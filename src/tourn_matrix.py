# ai generated, it runs faster then my version

"""Round-robin / gauntlet points-per-game matrix over a checkpoint directory.

    python points_matrix.py checkpoints/ --anchors 16 --games 16 --half

Cell (row i, col j) = mean points per game of checkpoint i playing ALONE against
three copies of checkpoint j. Scoring is by finishing place, where place comes
from death order (last snake standing is 1st):

    1st 2 pts   2nd 1 pt   3rd/4th 0 pts

Ties for 1st absorb the second-place point and split the pot 3/k: two share 1st
-> 1.5 each, three -> 1.0 each, all four die on the same tick -> 0.75 each.
A tie for 2nd splits only the single point: two -> 0.5 each, three -> 1/3 each.
Exactly 3 points are awarded every game, so PARITY IS 0.75 points/game.

Speed. Three things make this tractable at hundreds of checkpoints:

  * TILE FUSION. The engine state is a batch of games and nothing requires those
    games to be the same matchup, so a whole rectangle of the matrix is played in
    ONE state: game slot c*B+b is game b of cell c. An RxC tile costs R+C forwards
    per step instead of 2 forwards x R*C cells, and 450 sequential engine steps
    instead of 450*R*C. This is worth ~1000x and is the reason the script is fast.

  * GAUNTLET. --anchors K measures rows x anchors and anchors x cols instead of
    the full MxM. Both directions of every measured pair still exist, so the
    seat-bias cancellation and the pairwise Elo below are unaffected. At M=300
    that is 9,344 cells instead of 90,000.

  * RETIREMENT. A resolved game is frozen by the engine and a dead seat's action
    is ignored, so both are dropped from the forward pass every --regroup steps.
    LSTM state is carried across a regroup by masking, not rebuilt.

Per step there is exactly one host->device transfer (the live observation rows,
gathered contiguously so each group reads a slice) and exactly one sync.

priv_enc and v are ~a third of the parameters and forward() never reads them, so
they are not built and not loaded. Files without an `ep<N>` tag (pool.pt,
latest.pt, ...) are skipped: they are not points on the training trajectory.
"""
import argparse, os, re, sys, time
import numpy as np, torch, torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument('ckpt_dir')
ap.add_argument('--games', type=int, default=16, help='games per cell (rounded up to even)')
ap.add_argument('--steps', type=int, default=450)
ap.add_argument('--anchors', type=int, default=0,
                help='0 = full MxM grid; K = gauntlet against K checkpoints spread over training')
ap.add_argument('--tile-games', type=int, default=4096, help='approx games in flight per fused tile')
ap.add_argument('--regroup', type=int, default=16, help='drop finished games from the forward every N steps')
ap.add_argument('--stride', type=int, default=1, help='take every Nth checkpoint')
ap.add_argument('--limit', type=int, default=0, help='cap number of checkpoints (0 = all)')
ap.add_argument('--device', default='cuda')
ap.add_argument('--half', action='store_true', help='fp16 weights (halves GPU memory)')
ap.add_argument('--out', default='points_matrix')
ap.add_argument('--resume', action='store_true')
a = ap.parse_args()

import game, obsmem as om
B = a.games + (a.games & 1)      # engine packs 2 games per flag byte -> batch must be even
if B != a.games:
    print(f'note: --games {a.games} is odd; the flag buffer packs two games per byte, using {B}')
PARITY = 0.75
dev = torch.device(a.device)
WDTYPE = torch.float16 if a.half else torch.float32


# ---------------------------------------------------------------- model
from snakenet import SnakeNet


def cfg_from(sd):
    """Infer architecture from tensor shapes so mixed-size pools just work."""
    squeeze = sd['conv.4.weight'].shape[0]
    n_scal = sd['fc.0.weight'].shape[1] - squeeze * om.CROP * om.CROP
    return dict(ch=(sd['conv.0.weight'].shape[0], sd['conv.2.weight'].shape[0]),
                squeeze=squeeze, hid=sd['fc.0.weight'].shape[0],
                mem=sd['mem_cell.weight_hh'].shape[1],
                n_scal=n_scal)


def ep_of(fn):
    m = re.search(r'ep(\d+)', fn)
    return int(m.group(1)) if m else -1


files = [f for f in sorted(os.listdir(a.ckpt_dir), key=ep_of)
         if f.endswith('.pt') and ep_of(f) >= 0]
files = files[::a.stride]
if a.limit:
    files = files[:a.limit]
names = [f[:-3] for f in files]
eps = [ep_of(f) for f in files]
M = len(files)
if M == 0:
    sys.exit(f'no ep-tagged .pt checkpoints in {a.ckpt_dir}')

print(f'loading {M} checkpoints to {dev} ({"fp16" if a.half else "fp32"}) ...')
nets, tot, drop = [], 0, 0
for f in files:
    sd = torch.load(os.path.join(a.ckpt_dir, f), map_location='cpu')['model']
    drop += sum(v.numel() for k, v in sd.items() if k.startswith(('priv_enc', 'v.')))
    sd = {k: v for k, v in sd.items() if not k.startswith(('priv_enc', 'v.'))}
    cfg = cfg_from(sd)
    n = SnakeNet(**cfg)
    n.load_state_dict(sd, strict=False)
    n.n_scal = cfg['n_scal']                          # store for runtime
    n = n.to(dev, dtype=WDTYPE).eval()
    for p in n.parameters():
        p.requires_grad_(False)
    nets.append(n)
    tot += sum(p.numel() for p in n.parameters())
NEED_ALIVE = any(n.n_scal > 5 for n in nets)
assert len(nets) == M, f'{len(nets)} nets loaded but M={M}'
w = 2 if a.half else 4
print(f'  {tot/1e6:.1f}M params live, ~{tot*w/2**20:.0f} MiB of weights '
      f'({drop/1e6:.1f}M params in priv_enc/v skipped, saving {drop*w/2**20:.0f} MiB)\n')


# ------------------------------------------- game state, sized per tile
def alloc(G):
    """(Re)build the engine state for G simultaneous games."""
    global GB, board, head, foodg, hp, tord, mo, mf, mh, ma, el, rot, ob, sc, act, _init, gs, _IDX
    GB = G
    game.batch_size = G
    board = np.empty((G,15,15), np.uint8); head = np.empty((G,15,15), np.uint8)
    foodg = np.empty((G,15,15), np.uint8); hp = np.empty((G,4,2), np.uint8)
    tord = np.zeros((G,4,15,15), np.uint8)
    mo = np.zeros((G,4,15,15), np.uint8); mf = np.zeros((G,4,15,15), np.uint8)
    mh = np.zeros((G,4,15,15), np.uint8); ma = np.full((G,4,15,15),255,np.uint8)
    el = np.full((G,4,4),3,np.uint8); rot = np.zeros((G,4), np.int8)
    ob = np.zeros((G,4,om.C,om.CROP,om.CROP), np.uint8); sc = np.zeros((G,4,5), np.uint8)
    act = np.zeros((G,4), np.uint8)
    _init = game.generate_init_data(G)
    gs = dict(data=_init.copy(), flags=np.empty(G//2, np.uint8), food=np.empty((G,29), np.uint8),
              length=np.empty((G,4), np.uint8), health=np.empty((G,4), np.uint8),
              timestep=np.uint32(0), new_apple=np.full((G,2),255,np.uint8))
    _IDX = np.arange(G)


INV = np.array([[0,1,2,3],[3,2,0,1],[1,0,3,2],[2,3,1,0]], np.uint8)
GB = 0


def reset():
    np.copyto(gs['data'], _init); gs['flags'].fill(0xFF); gs['food'].fill(0)
    gs['length'].fill(3); gs['health'].fill(100); gs['timestep'] = np.uint32(0)
    gs['new_apple'].fill(255)
    mo.fill(0); mf.fill(0); mh.fill(0); ma.fill(255); el.fill(3)


def alive():
    """Liveness from the engine flag nibbles. Never read the 225-slot ring
    buffer: a resolved game is frozen there, so the ring reports the winner
    dead one tick later and re-animates it every 225 ticks."""
    nib = (gs['flags'][_IDX >> 1] >> ((_IDX & 1) * 4).astype(np.uint8)) & 0xF
    return ((nib[:, None] >> np.arange(4)) & 1).astype(bool)


def refresh(t):
    om.build_true_boards(gs['data'], gs['food'], gs['length'], t, board, head, foodg, hp, tord)
    om.update_memory(board, head, foodg, hp, gs['new_apple'], gs['length'],
                     mo, mf, mh, ma, el, om.VIEW_DX, om.VIEW_DY)
    om.heading_rot_all(gs['data'], t, rot)
    om.write_channels(board, foodg, hp, tord, mo, mf, mh, ma, el, gs['length'],
                      rot, om.ROT_OFF, ob, sc, t)


def pack(a3, r):
    raw = INV[r.astype(np.int64), om.ACT3_TO_CANON[a3]]
    return (raw[:,0] | (raw[:,1]<<2) | (raw[:,2]<<4) | (raw[:,3]<<6)).astype(np.uint8)


def points_from_death(d):
    """(G,4) death tick -> (G,4) points. Larger tick = survived longer = better.

    A tie for 1st swallows the second-place point and splits 3 ways-many: 2 ->
    1.5, 3 -> 1.0, 4 -> 0.75. A lone winner takes 2 and the next group down
    splits only the single second-place point. Every game awards exactly 3.
    """
    first = d == d.max(1, keepdims=True)
    k1 = first.sum(1, keepdims=True)
    pts = np.where(first, 3.0 / k1, 0.0)
    lone = k1[:, 0] == 1
    if lone.any():
        rest = np.where(first[lone], -1, d[lone])
        second = rest == rest.max(1, keepdims=True)
        k2 = second.sum(1, keepdims=True)
        pts[lone] = np.where(first[lone], 2.0, np.where(second, 1.0 / k2, 0.0))
    return pts


@torch.inference_mode()
def run_tile(cells, seed):
    """Play every (i,j) in `cells` at once. Returns per-cell (mean, sem) of seat-0 points.

    Slot layout is game c*B+b for cell c. Each group is one net plus a sorted
    array of flat (game*4+seat) ids it drives; groups shrink as games resolve and
    the LSTM state is carried across by masking, so ordering must stay sorted.
    """
    G = len(cells) * B
    if GB != G:
        alloc(G)
    np.random.seed(seed); reset()

    slots = {}                                   # (net index, seats) -> list of cell indices
    for c, (i, j) in enumerate(cells):
        slots.setdefault((i, (0,)), []).append(c)
        slots.setdefault((j, (1, 2, 3)), []).append(c)

    def flat_ids(cs, seats):
        g = (np.asarray(cs)[:, None] * B + np.arange(B)).ravel()
        return np.sort((g[:, None] * 4 + np.asarray(seats)).ravel())

    groups = [[nets[k], flat_ids(cs, seats), None] for (k, seats), cs in slots.items()]
    for g in groups:
        g[2] = (torch.zeros(len(g[1]), g[0].mem, device=dev, dtype=WDTYPE),
                torch.zeros(len(g[1]), g[0].mem, device=dev, dtype=WDTYPE))

    death = np.full((G, 4), a.steps, np.int32)   # survivors keep the sentinel -> 1st
    obf, scf, actf = ob.reshape(G*4, om.C, om.CROP, om.CROP), sc.reshape(G*4, 5), act.reshape(G*4)
    ids = np.concatenate([g[1] for g in groups])
    for t in range(a.steps):
        refresh(t)
        om.safe_moves3(board, hp, rot, om.ROT_OFF, act)

        # one gather, one upload, one sync per step; each group reads a slice
        x = om.dequantise(torch.from_numpy(obf[ids]).to(dev, non_blocking=True)).to(WDTYPE)
        s = torch.from_numpy(scf[ids]).to(dev, non_blocking=True).float()
        s = torch.stack((s[:,0]/100., s[:,1]/om.LEN_MAX, s[:,2]/200.,
                         s[:,3]/14., s[:,4]/14.), 1).to(WDTYPE)
        if NEED_ALIVE:
            na = alive().sum(1)                       # (G,) total alive per game
            alive_col = torch.tensor(na[ids // 4], device=dev, dtype=WDTYPE).unsqueeze(1) / 4.0
        outs, off = [], 0
        for g in groups:
            n, gid, hc = g
            lo, hi = off, off + len(gid)
            s_g = s[lo:hi]
            if n.n_scal > 5:
                s_g = torch.cat((s_g, alive_col[lo:hi]), 1)
            logits, _, g[2] = n(x[lo:hi], s_g, hc)
            outs.append(logits.argmax(-1))
            off = hi
        actf[ids] = torch.cat(outs).to(torch.uint8).cpu().numpy()

        game.step_wrapper(gs, pack(act, rot))
        al = alive()
        death[(death == a.steps) & ~al] = t
        undecided = al.sum(1) > 1
        if not undecided.any():
            break                      # every game decided; the engine freezes the rest
        if t % a.regroup == a.regroup - 1:
            live = np.flatnonzero((undecided[:, None] & al).ravel())   # live seat of a live game
            for g in groups:
                keep = np.isin(g[1], live, assume_unique=True)
                if not keep.all():
                    g[1] = g[1][keep]
                    kt = torch.from_numpy(keep).to(dev)
                    g[2] = (g[2][0][kt], g[2][1][kt])
            groups = [g for g in groups if len(g[1])]
            ids = np.concatenate([g[1] for g in groups]) if groups else np.empty(0, np.int64)

    p = points_from_death(death)[:, 0].reshape(len(cells), B)
    return p.mean(1), p.std(1, ddof=1) / B**0.5


# --------------------------------------------------------------- run grid
if a.anchors:
    anchors = sorted(set(np.linspace(0, M-1, min(a.anchors, M)).round().astype(int).tolist()))
    want = np.zeros((M, M), bool); want[:, anchors] = True; want[anchors, :] = True
else:
    anchors = list(range(M))
    want = np.ones((M, M), bool)

P = np.full((M, M), np.nan, np.float32)
SEM = np.full((M, M), np.nan, np.float32)
path = a.out + '.npz'
if a.resume and os.path.exists(path):
    z = np.load(path, allow_pickle=True)
    if list(z['names']) == names and z['P'].shape == (M, M):
        P = z['P']; SEM = z['sem'] if 'sem' in z else SEM
        print('resumed:', int(np.isfinite(P).sum()), 'cells already done\n')
    else:
        print('existing npz does not match this checkpoint set -- starting fresh\n')

cpt = max(1, a.tile_games // B)                  # cells per tile
cc = min(len(anchors), max(1, int(cpt ** 0.5)))  # tile is a rectangle so distinct nets stay low
rr = max(1, cpt // cc)
tiles = []
for r0 in range(0, M, rr):
    for c0 in range(0, M, cc):
        cs = [(i, j) for i in range(r0, min(r0+rr, M)) for j in range(c0, min(c0+cc, M))
              if want[i, j] and not np.isfinite(P[i, j])]
        if cs:
            tiles.append(cs)

todo = sum(len(t) for t in tiles)
print(f'{todo} cells to run ({"full grid" if a.anchors == 0 else f"{len(anchors)}-anchor gauntlet"}, '
      f'{B} games each, parity {PARITY})')
print(f'{len(tiles)} fused tiles of <={rr}x{cc} cells -> <={cpt*B} games in flight, '
      f'{len(tiles)*a.steps} sequential engine steps instead of {todo*a.steps}\n')

t0 = time.time(); done = 0
for ti, cs in enumerate(tiles):
    m, sd = run_tile(cs, seed=1000 + ti)
    for (i, j), mv, sv in zip(cs, m, sd):
        P[i, j] = mv; SEM[i, j] = sv
    done += len(cs)
    el_s = time.time() - t0
    eta = (todo - done) / max(done / max(el_s, 1e-9), 1e-9)
    print(f'tile {ti+1}/{len(tiles)}  {len(cs):4d} cells  rows {cs[0][0]}-{cs[-1][0]:<4d} '
          f'mean={np.mean(m):.3f}  [{el_s/60:.1f}m elapsed, ETA {eta/60:.1f}m]', flush=True)
    np.savez(path, P=P, sem=SEM, names=np.array(names), eps=np.array(eps),
             anchors=np.array(anchors))

# ------------------------------------------------------------------ Elo
# score share from the points split of a pairing: s = P[i,j] / (P[i,j] + P[j,i]).
# The gauntlet keeps both directions of every measured pair, so this is unchanged.
elo = np.full(M, 1500.0)
pairs = [(i, j) for i in range(M) for j in range(M)
         if i != j and np.isfinite(P[i,j]) and np.isfinite(P[j,i]) and P[i,j] + P[j,i] > 0]
for _ in range(6000):
    for i, j in pairs:
        s = P[i,j] / (P[i,j] + P[j,i])
        exp = 1/(1 + 10**((elo[j]-elo[i])/400))
        d = 2.0*(s-exp); elo[i] += d; elo[j] -= d
np.savez(path, P=P, sem=SEM, names=np.array(names), eps=np.array(eps),
         anchors=np.array(anchors), elo=elo)

# ----------------------------------------------------------------- plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

fig = plt.figure(figsize=(max(9, M*0.34)+3, max(8, M*0.30)+3))
gsp = fig.add_gridspec(2, 2, width_ratios=[3.4, 1], height_ratios=[3.4, 1],
                       hspace=0.16, wspace=0.14)

ax = fig.add_subplot(gsp[0, 0])
cmap = plt.get_cmap('RdBu_r').copy(); cmap.set_bad('#f2f2f0')
norm = TwoSlopeNorm(vmin=0.0, vcenter=PARITY, vmax=min(2.0, max(1.2, np.nanmax(P))))
im = ax.imshow(np.ma.masked_invalid(P), cmap=cmap, norm=norm, origin='upper',
               interpolation='nearest')
ax.set_title('points/game: row (alone, seat 0) vs 3x column   —   parity 0.75', pad=10)
ax.set_xlabel('opponent (3 copies)'); ax.set_ylabel('lone agent')
step = max(1, M//30)
ax.set_xticks(range(0, M, step)); ax.set_xticklabels([str(eps[k]) for k in range(0, M, step)],
                                                     rotation=90, fontsize=7)
ax.set_yticks(range(0, M, step)); ax.set_yticklabels([str(eps[k]) for k in range(0, M, step)],
                                                     fontsize=7)
if M <= 16:
    for i in range(M):
        for j in range(M):
            if np.isfinite(P[i, j]):
                ax.text(j, i, f'{P[i,j]:.2f}', ha='center', va='center', fontsize=7,
                        color='white' if abs(P[i,j]-PARITY) > 0.5 else '#222')
fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)

axr = fig.add_subplot(gsp[0, 1])
axr.plot(elo, range(M), lw=1.4, color='#2a78d6')
axr.set_ylim(M-0.5, -0.5); axr.set_yticks([]); axr.grid(alpha=.25)
axr.set_title('Elo', fontsize=10)
axr.axvline(elo[np.nanargmax(elo)], color='#1baf7a', ls='--', lw=1)

# trajectory always averages over the SAME opponent set (the anchors), otherwise
# anchor rows would be scored against a wider pool than everyone else
axb = fig.add_subplot(gsp[1, 0])
mean_pts = np.nanmean(P[:, anchors], axis=1)
band = np.nanmean(SEM[:, anchors], axis=1) / max(1, len(anchors))**0.5
axb.plot(eps, mean_pts, lw=1.5, color='#2a78d6', label=f'mean points/game vs {len(anchors)} anchors')
axb.fill_between(eps, mean_pts-band, mean_pts+band, color='#2a78d6', alpha=.18, lw=0)
axb.plot(eps, np.diag(P), lw=1.0, color='#eb6834', alpha=.8, label='self-play (should be ~0.75)')
axb.axhline(PARITY, color='#888780', ls=':', lw=1)
axb.set_xlabel('episode'); axb.set_ylabel('points/game'); axb.grid(alpha=.25)
axb.legend(fontsize=8, loc='best')

best = int(np.nanargmax(elo))
fig.suptitle(f'checkpoint gauntlet  ({B} games/cell, {M} checkpoints, {todo} cells)   '
             f'best Elo: {names[best]} ({elo[best]:.0f})', y=0.985)
fig.savefig(a.out + '.png', dpi=150, bbox_inches='tight')
print(f'\nwrote {a.out}.png and {a.out}.npz')

ordr = np.argsort(-elo)
def show(title, idx):
    print('\n' + title)
    for k in idx:
        print(f'  {names[k]:>18s}  elo={elo[k]:7.1f}  '
              f'mean_vs_anchors={np.nanmean(P[k, anchors]):.3f}  self={P[k,k]:.3f}')
show('ranked by Elo (best first):' if M <= 12 else 'top 10 by Elo:',
     ordr if M <= 12 else ordr[:10])
if M > 12:
    show('bottom 5:', ordr[-5:])

dse = np.diag(SEM)
off = np.abs(np.diag(P) - PARITY) / np.where(dse > 0, dse, np.inf)
print(f'\nself-play diagonal: mean={np.nanmean(np.diag(P)):.3f} '
      f'(expect {PARITY}, mean SE={np.nanmean(dse):.3f} at {B} games)')
if np.nanmax(off) > 3:
    print(f'  NOTE: {names[int(np.nanargmax(off))]} is {np.nanmax(off):.1f} SE off parity '
          f'-- raise --games if this persists.')