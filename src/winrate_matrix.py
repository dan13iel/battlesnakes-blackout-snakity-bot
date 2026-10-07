# ai generated, it runs faster then my version

"""Full round-robin win-rate matrix over a checkpoint directory + heatmap.

    python winrate_matrix.py checkpoints/ --games 512 --device cuda

Cell (row i, col j) = win rate of checkpoint i playing ALONE against three
copies of checkpoint j. Parity is 0.25. The matrix is deliberately asymmetric:
(i,j) and (j,i) are separate measurements, and comparing them cancels any
seat/format bias. The diagonal is self-play and should sit near 0.25 --
if it doesn't, the harness is broken, not the model.

Files without an `ep<N>` tag in the name (pool.pt, latest.pt, ...) are skipped:
they are not points on the training trajectory and would otherwise sort to the
front of the grid with a bogus episode of -1.

All checkpoints are loaded to the device up front; exactly one game state
exists at a time and is reused across matches. Everything runs under
inference_mode, so no autograd graph is ever built.
"""
import argparse, os, re, sys, time
import numpy as np, torch, torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument('ckpt_dir')
ap.add_argument('--games', type=int, default=512)
ap.add_argument('--steps', type=int, default=450)
ap.add_argument('--stride', type=int, default=1, help='take every Nth checkpoint')
ap.add_argument('--limit', type=int, default=0, help='cap number of checkpoints (0 = all)')
ap.add_argument('--device', default='cuda')
ap.add_argument('--half', action='store_true', help='fp16 weights (halves GPU memory)')
ap.add_argument('--out', default='winrate_matrix')
ap.add_argument('--resume', action='store_true')
a = ap.parse_args()

import game, obsmem as om
B = a.games
game.batch_size = B
dev = torch.device(a.device)
WDTYPE = torch.float16 if a.half else torch.float32


# ---------------------------------------------------------------- model
class SnakeNet(nn.Module):
    def __init__(s, in_ch=om.C, crop=om.CROP, n_scal=5, n_act=3,
                 ch=(32, 64), squeeze=24, hid=256, mem=128, priv_ch=7):
        super().__init__()
        s.mem = mem
        s.conv = nn.Sequential(nn.Conv2d(in_ch, ch[0], 3, padding=1), nn.ReLU(),
                               nn.Conv2d(ch[0], ch[1], 3, padding=1), nn.ReLU(),
                               nn.Conv2d(ch[1], squeeze, 1), nn.ReLU(), nn.Flatten())
        s.fc = nn.Sequential(nn.Linear(squeeze * crop * crop + n_scal, hid), nn.ReLU())
        s.mem_cell = nn.LSTMCell(hid, mem)
        s.pi = nn.Linear(mem, n_act)
        s.priv_enc = nn.Sequential(nn.Conv2d(priv_ch, ch[0], 3, padding=1), nn.ReLU(),
                                   nn.Conv2d(ch[0], ch[1], 3, padding=1), nn.ReLU(),
                                   nn.Conv2d(ch[1], squeeze, 1), nn.ReLU(), nn.Flatten(),
                                   nn.Linear(squeeze * 225, 128), nn.ReLU())
        s.v = nn.Sequential(nn.Linear(mem + 128, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(s, x, scal, hc):
        h, c = s.mem_cell(s.fc(torch.cat((s.conv(x), scal), 1)), hc)
        return s.pi(h), (h, c)


def cfg_from(sd):
    """Infer architecture from tensor shapes so mixed-size pools just work."""
    return dict(ch=(sd['conv.0.weight'].shape[0], sd['conv.2.weight'].shape[0]),
                squeeze=sd['conv.4.weight'].shape[0], hid=sd['fc.0.weight'].shape[0],
                mem=sd['mem_cell.weight_hh'].shape[1],
                priv_ch=sd['priv_enc.0.weight'].shape[1])


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
nets, tot = [], 0
for f in files:
    sd = torch.load(os.path.join(a.ckpt_dir, f), map_location='cpu')['model']
    n = SnakeNet(**cfg_from(sd))
    n.load_state_dict(sd)
    n = n.to(dev, dtype=WDTYPE).eval()
    for p in n.parameters():
        p.requires_grad_(False)
    nets.append(n)
    tot += sum(p.numel() for p in n.parameters())
assert len(nets) == M, f'{len(nets)} nets loaded but M={M}'
print(f'  {tot/1e6:.1f}M params total, ~{tot*(2 if a.half else 4)/2**20:.0f} MiB of weights\n')


# ------------------------------------------------- one reusable game state
INV = np.array([[0,1,2,3],[3,2,0,1],[1,0,3,2],[2,3,1,0]], np.uint8)
board = np.empty((B,15,15), np.uint8); head = np.empty((B,15,15), np.uint8)
foodg = np.empty((B,15,15), np.uint8); hp = np.empty((B,4,2), np.uint8)
tord = np.zeros((B,4,15,15), np.uint8)
mo = np.zeros((B,4,15,15), np.uint8); mf = np.zeros((B,4,15,15), np.uint8)
mh = np.zeros((B,4,15,15), np.uint8); ma = np.full((B,4,15,15),255,np.uint8)
el = np.full((B,4,4),3,np.uint8); rot = np.zeros((B,4), np.int8)
ob = np.zeros((B,4,om.C,om.CROP,om.CROP), np.uint8); sc = np.zeros((B,4,5), np.uint8)
act = np.zeros((B,4), np.uint8)
_init = game.generate_init_data(B)
gs = dict(data=_init.copy(), flags=np.empty(B//2, np.uint8), food=np.empty((B,29), np.uint8),
          length=np.empty((B,4), np.uint8), health=np.empty((B,4), np.uint8),
          timestep=np.uint32(0), new_apple=np.full((B,2),255,np.uint8))
_IDX = np.arange(B)


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


def seat_batch(seats):
    """Stack several seats into one forward pass."""
    x = om.dequantise(torch.from_numpy(ob[:, seats].reshape(-1, om.C, om.CROP, om.CROP)).to(dev))
    s = torch.from_numpy(sc[:, seats].reshape(-1, 5)).to(dev).float()
    s = torch.stack((s[:,0]/100., s[:,1]/om.LEN_MAX, s[:,2]/200.,
                     s[:,3]/14., s[:,4]/14.), 1)
    return x.to(WDTYPE), s.to(WDTYPE)


@torch.inference_mode()
def match(net_a, net_b, seed):
    """net_a in seat 0, net_b in seats 1-3. Returns seat-0 win rate."""
    np.random.seed(seed); reset()
    groups = [(net_a, [0]), (net_b, [1, 2, 3])] if net_b is not net_a else [(net_a, [0,1,2,3])]
    state = [(n, s, (torch.zeros(B*len(s), n.mem, device=dev, dtype=WDTYPE),
                     torch.zeros(B*len(s), n.mem, device=dev, dtype=WDTYPE)))
             for n, s in groups]
    for t in range(a.steps):
        refresh(t)
        om.safe_moves3(board, hp, rot, om.ROT_OFF, act)
        new_state = []
        for n, seats, hc in state:
            x, s = seat_batch(seats)
            logits, hc = n(x, s, hc)
            act[:, seats] = logits.argmax(-1).view(B, len(seats)).cpu().numpy().astype(np.uint8)
            new_state.append((n, seats, hc))
        state = new_state
        game.step_wrapper(gs, pack(act, rot))
        if not (alive().sum(1) > 1).any():
            break                      # every game decided; the rest is frozen
    fa = alive()
    return float((fa[:,0] & (fa.sum(1) == 1)).mean())


# --------------------------------------------------------------- run grid
W = np.full((M, M), np.nan, np.float32)
path = a.out + '.npz'
if a.resume and os.path.exists(path):
    z = np.load(path, allow_pickle=True)
    if list(z['names']) == names and z['W'].shape == (M, M):
        W = z['W']; print('resumed:', int(np.isfinite(W).sum()), 'cells already done\n')
    else:
        print('existing npz does not match this checkpoint set -- starting fresh\n')

todo = int((~np.isfinite(W)).sum())
print(f'running {todo} matches ({M}x{M} grid, {B} games each)\n')
t0 = time.time(); done = 0
for i in range(M):
    for j in range(M):
        if np.isfinite(W[i, j]):
            continue
        W[i, j] = match(nets[i], nets[j], seed=1000 + i*M + j)
        done += 1
    el_s = time.time() - t0
    rate = done / max(el_s, 1e-9)
    eta = (todo - done) / max(rate, 1e-9)
    print(f'row {i+1}/{M}  {names[i]:>16s}  '
          f'mean={np.nanmean(W[i]):.3f}  diag={W[i,i]:.3f}  '
          f'[{el_s/60:.1f}m elapsed, ETA {eta/60:.1f}m]', flush=True)
    np.savez(path, W=W, names=np.array(names), eps=np.array(eps))

# ------------------------------------------------------------------ Elo
elo = np.full(M, 1500.0)
for _ in range(6000):
    for i in range(M):
        for j in range(M):
            if i == j or not np.isfinite(W[i,j]) or not np.isfinite(W[j,i]):
                continue
            tot = W[i,j] + W[j,i]
            if tot <= 0: continue
            s = W[i,j] / tot
            exp = 1/(1 + 10**((elo[j]-elo[i])/400))
            d = 2.0*(s-exp); elo[i] += d; elo[j] -= d
np.savez(path, W=W, names=np.array(names), eps=np.array(eps), elo=elo)

# ----------------------------------------------------------------- plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

fig = plt.figure(figsize=(max(9, M*0.34)+3, max(8, M*0.30)+3))
gsp = fig.add_gridspec(2, 2, width_ratios=[3.4, 1], height_ratios=[3.4, 1],
                       hspace=0.16, wspace=0.14)

ax = fig.add_subplot(gsp[0, 0])
norm = TwoSlopeNorm(vmin=0.0, vcenter=0.25, vmax=min(1.0, max(0.5, np.nanmax(W))))
im = ax.imshow(W, cmap='RdBu_r', norm=norm, origin='upper', interpolation='nearest')
ax.set_title('win rate: row (alone, seat 0) vs 3x column   —   parity 0.25', pad=10)
ax.set_xlabel('opponent (3 copies)'); ax.set_ylabel('lone agent')
step = max(1, M//30)
ax.set_xticks(range(0, M, step)); ax.set_xticklabels([str(eps[k]) for k in range(0, M, step)],
                                                     rotation=90, fontsize=7)
ax.set_yticks(range(0, M, step)); ax.set_yticklabels([str(eps[k]) for k in range(0, M, step)],
                                                     fontsize=7)
if M <= 16:
    for i in range(M):
        for j in range(M):
            if np.isfinite(W[i, j]):
                ax.text(j, i, f'{W[i,j]:.2f}', ha='center', va='center', fontsize=7,
                        color='white' if abs(W[i,j]-0.25) > 0.18 else '#222')
fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)

axr = fig.add_subplot(gsp[0, 1])
axr.plot(elo, range(M), lw=1.4, color='#2a78d6')
axr.set_ylim(M-0.5, -0.5); axr.set_yticks([]); axr.grid(alpha=.25)
axr.set_title('Elo', fontsize=10)
axr.axvline(elo[np.nanargmax(elo)], color='#1baf7a', ls='--', lw=1)

axb = fig.add_subplot(gsp[1, 0])
axb.plot(eps, np.nanmean(W, axis=1), lw=1.5, color='#2a78d6', label='mean win rate vs pool')
axb.plot(eps, np.diag(W), lw=1.0, color='#eb6834', alpha=.8, label='self-play (should be ~0.25)')
axb.axhline(0.25, color='#888780', ls=':', lw=1)
axb.set_xlabel('episode'); axb.set_ylabel('win rate'); axb.grid(alpha=.25)
axb.legend(fontsize=8, loc='best')

best = int(np.nanargmax(elo))
fig.suptitle(f'checkpoint round-robin  ({B} games/cell, {M} checkpoints)   '
             f'best Elo: {names[best]} ({elo[best]:.0f})', y=0.985)
fig.savefig(a.out + '.png', dpi=150, bbox_inches='tight')
print(f'\nwrote {a.out}.png and {a.out}.npz')

ordr = np.argsort(-elo)
def show(title, idx):
    print('\n' + title)
    for k in idx:
        print(f'  {names[k]:>18s}  elo={elo[k]:7.1f}  '
              f'mean_vs_pool={np.nanmean(W[k]):.3f}  self={W[k,k]:.3f}')
show(f'ranked by Elo (best first):' if M <= 12 else 'top 10 by Elo:',
     ordr if M <= 12 else ordr[:10])
if M > 12:
    show('bottom 5:', ordr[-5:])

se = (0.25*0.75/B)**0.5
off = np.abs(np.diag(W) - 0.25) / se
print(f'\nself-play diagonal: mean={np.nanmean(np.diag(W)):.3f} '
      f'(expect 0.25, SE={se:.3f} at {B} games)')
if np.nanmax(off) > 3:
    print(f'  NOTE: {names[int(np.nanargmax(off))]} is {np.nanmax(off):.1f} SE off parity '
          f'-- raise --games if this persists.')