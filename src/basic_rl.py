import numpy as np
import csv
import obsmem as om
from numba import jit, njit
import game; print('compiled remapping luts')
from game import generate_init_data, game_state_types, pack_single_flags, remap_canonical
from tqdm import tqdm
from plyer import notification
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["FIXED_DEPTH"] = "6"
print("using a fixed depth for ExIT of", os.environ["FIXED_DEPTH"])
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import requests
from pyinstrument import Profiler
import sys
import logging
from shapeshifter_repo.proxy import optimal_moves, solve_batch, ENGINE_URL;print("setup proxy to shapeshifter engine")

def _check_engine():
    import urllib.request as _ur, json as _json
    base = ENGINE_URL.rsplit("/", 1)[0] + "/"
    try:
        with _ur.urlopen(base, timeout=5) as r:
            print(f"engine health: {ENGINE_URL} -> HTTP {r.status}")
            return
    except Exception as e:
        print(f"engine root unreachable ({base}): {type(e).__name__}: {e}")
    test = {"game":{"id":"t","ruleset":{"name":"standard","version":"v1","settings":{}},"map":"standard","timeout":200,"source":"custom"},
            "turn":0,"board":{"height":15,"width":15,"food":[],"hazards":[],"snakes":[
                {"id":"0","name":"s0","health":100,"length":3,"head":{"x":7,"y":7},
                 "body":[{"x":7,"y":7},{"x":7,"y":6},{"x":7,"y":5}],"shout":None,"squad":None,"next_move":None}]},
            "you":{"id":"0","name":"s0","health":100,"length":3,"head":{"x":7,"y":7},
                   "body":[{"x":7,"y":7},{"x":7,"y":6},{"x":7,"y":5}],"shout":None,"squad":None,"next_move":None}}
    try:
        req = _ur.Request(ENGINE_URL, data=_json.dumps(test).encode(), headers={"Content-Type":"application/json"})
        with _ur.urlopen(req, timeout=10) as r:
            print(f"engine move endpoint: HTTP {r.status}, body: {r.read()[:200]}")
    except Exception as e2:
        print(f"engine move endpoint FAILED ({ENGINE_URL}): {type(e2).__name__}: {e2}")
_check_engine()

def error_handler(exc_type, exc_value, exc_traceback):
    # ignore KeyboardInterrupt
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    try:
        # I placed canary tokens here as it was the easiest way to get an email/notification when something happened.
        if issubclass(exc_type, torch.OutOfMemoryError):
            requests.get("") # OOM url
        else:
            requests.get("") # General error url
    except Exception:
        pass # in case the canary token fails
    sys.__excepthook__(exc_type, exc_value, exc_traceback) # still print the traceback

IMPORTED = __name__ != "__main__"
if not IMPORTED:
    sys.excepthook = error_handler # was defined but never installed

try:
    import platform
except ImportError:
    class platform:
        def system():
            return "Linux" # linux being set here makes notifications not show which is the safe case if u dont know the display status
            
print('imported')

game_step_dur = 600 # max ticks per rollout
game_batch_count = 0 if IMPORTED else 100_000 # number of rollouts. 

batch_size = 100
game.batch_size = 100
# if this is not set then u will get this error:
# No implementation of function Function(<built-in function zeros>) found for signature:
# >>> zeros(Tuple(Literal[int](4), none), dtype=class(bool))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("training on:", device)

torch.set_num_threads(8)

gamma = 0.995
gae_lambda = 0.97
value_coef = 0.5
lr = 2e-5
ppo_epochs = 2 # replay passes over each collected episode; >1 is what makes the clip actually do something
clip_eps = 0.2 # PPO clipped-surrogate epsilon

ExIT = False # Self plays but instead of learning of the terminal reward, it does cross entropy towards the best engine move. Highly discouraged.
EXIT_STRIDE = 4 # query engine every N ticks (1 = every tick, 4 = 4x faster)
ENGINE_TIMEOUT = 20 # ms per game, but it gets overriden by the env var FIXED_DEPTH so this is kinda useless rn.
SELFPLAY = True # False reproduces the original safe_moves3-only opponents
OPP_POOL_CAP = 196 # max frozen opponents retained
OPP_SAMPLE = 5 # distinct pool models loaded per episode, I found that 5 worked best
ADMIT_WIN_RATE = 0.55 # seat-0 eval win rate required to enter the pool, compared to random chance wr of 25%
SCRIPTED_FRAC = 0.30 # share of opponent slots always left to safe_moves3
OPP_SAMPLE_ACTIONS = True # opponents sample (diversity) rather than argmax
POOL_KEEP_ANCHORS = 2 # oldest N pool entries are never evicted
POOL_STRIP_CRITIC = True # drop priv_enc/v from stored weights (opponents never use them)
POOL_DTYPE = torch.float16 # storage dtype for the pool; load_state_dict casts on copy
SEED_POOL_WITH_INIT = False # True starts self-play immediately vs a random net
EVAL_EVERY = 200 # episodes between evaluations / checkpoints
EVAL_VS = 'pool' # 'pool' or 'scripted', is what ADMIT_WIN_RATE is measured against
BASELINE_EVAL = True # additionally eval vs safe_moves3 for an absolute number to see for debug
EVAL_SEED = 1234 # fixes the eval matchup so the number is comparable run to run

# Shapeshifter engine
SHAPESHIFTER_OPP_FRAC = 0.00          # fraction of games that get an engine opponent
SHAPESHIFTER_OPP_SEAT_RANDOM = True  # if True, seat is random per episode (1–3)

PFSP = False
PFSP_POWER = 1.5        # higher = more aggressive focus on hard opponents
PFSP_EPS = 0.005          # floor so beaten opponents never reach zero probability

# fixed opp's to make a model to beat the general model easily to make the more general model rlly good
# Put .pt checkpoint paths here to train against them instead of safe_moves3 (replaces scripted opponents).
# leave empty for normal self-play.
OPP_CKPT_PATHS = [
#"checkpoints/ckpt_ep86249.pt"
]
FIXED_OPP = len(OPP_CKPT_PATHS) > 0

if FIXED_OPP:
    SELFPLAY            = True  # reuse opponent net infrastructure
    ADMIT_WIN_RATE      = 999.0 # never self-admit; pool stays fixed
    SEED_POOL_WITH_INIT = False
    PFSP                = False # uniform sampling over the targets

print("fixed opponent?", FIXED_OPP)
print("using ExIT?", ExIT)
print("using PFSP?", PFSP)
print("opponents sampled each time:", OPP_SAMPLE)

from snakenet import SnakeNet, PRIV_CH, INV_ROT_ACTION

N_ACT = 3

# remapping rotation bs between the engine and the model
ROT_ACTION = np.zeros_like(INV_ROT_ACTION)
for _r in range(INV_ROT_ACTION.shape[0]):
    for _c in range(INV_ROT_ACTION.shape[1]):
        ROT_ACTION[_r, INV_ROT_ACTION[_r, _c]] = _c
CANON_TO_ACT3 = np.full(4, -1, dtype=np.int64)
for _a3 in range(N_ACT):
    CANON_TO_ACT3[om.ACT3_TO_CANON[_a3]] = _a3

def engine_to_act3(engine_moves, rot):
    B = len(engine_moves)
    actions = np.zeros(B, dtype=np.int64)
    valid = np.zeros(B, dtype=bool)
    for g in range(B):
        mv = engine_moves[g]
        if mv is None:
            continue
        
        r = int(rot[g])
        
        canon = ROT_ACTION[r, int(mv)]
        a3 = CANON_TO_ACT3[canon]
        if a3 >= 0:
            actions[g] = a3
            valid[g] = True
    return actions, valid

# interesting dict()
NET_CFG = dict(ch=(16, 16), squeeze=8, hid=192, mem=96)
CLUSTER_CFG = dict(ch=(32, 64), squeeze=24, hid=256, mem=128)

net = SnakeNet(priv_ch=PRIV_CH, **CLUSTER_CFG).to(device)
memory_latent_dim = net.mem
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

USE_AMP = (str(device) != 'cpu' and torch.cuda.is_available()
           and torch.cuda.is_bf16_supported())
amp_ctx = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if USE_AMP \
          else torch.enable_grad

# compile only the time-independent submodules. dynamic=True because they are called with batch B during rollout and T*B during the update, and T varies per episode.
# mem_cell is deliberately left alone: its python loop would unroll per T and recompile constantly.

# also Module.compile() is in-place so for a good day do NOT use net.conv = torch.compile(net.conv)
# (that rebinds to an OptimizedModule and prefixes state_dict keys with `_orig_mod.`, which breaks load_state_dict against every existing checkpoint)
COMPILE = USE_AMP
if COMPILE:
    net.conv.compile(dynamic=True, mode="reduce-overhead")
    net.fc.compile(dynamic=True, mode="reduce-overhead")
    if net.priv_enc is not None:
        net.priv_enc.compile(dynamic=True, mode="reduce-overhead")

# placement reward, mean-centred: [2,1,0,0] - 0.75
PLACE_R = np.array([1.25, 0.25, -0.75, -0.75], dtype=np.float32)

import os, glob, re
from safetensors.torch import save_file as st_save, load_file as st_load

CKPT_DIR = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

def save_checkpoint(episode):
    if FIXED_OPP: return
    path = os.path.join(CKPT_DIR, f"ckpt_ep{episode:04d}.pt")
    tmp_path = path + ".tmp"
    try:
        torch.save({
            "episode": episode,
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, tmp_path)
        os.replace(tmp_path, path) # atomic on same filesystem; avoids partial/locked writes
    except (RuntimeError, OSError) as e:
        print(f"[warn] checkpoint save failed at episode {episode}, skipping: {e}")
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except OSError: pass
        return
    ckpt_index = episode // EVAL_EVERY # checkpoints happen every EVAL_EVERY eps
    if ckpt_index % 2 == 0:
        st_path = os.path.join(CKPT_DIR, f"model_ep{episode:04d}.safetensors")
        #try:
            #st_save(net.state_dict(), st_path, metadata={"in_dim": str(in_dim), "hidden": str(hidden_dim), "episode": str(episode)})
        #except OSError as e:
        #    print(f"[warn] safetensors save failed at episode {episode}, skipping: {e}")

def latest_checkpoint():
    pts = glob.glob(os.path.join(CKPT_DIR, "ckpt_ep*.pt"))
    if not pts:
        return None, -1
    def ep(p):
        m = re.search(r"ckpt_ep(\d+)\.pt", p)
        return int(m.group(1)) if m else -1
    best = max(pts, key=ep)
    return best, ep(best)

start_episode = 0
ckpt_path, ckpt_ep = latest_checkpoint()
if ckpt_path:
    ckpt = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_episode = 0 if IMPORTED else ckpt_ep + 1
    print(f"resumed from episode {ckpt_ep} ({ckpt_path})")

_warm_flags = np.full(batch_size // 2, np.uint8(0xFF), dtype=np.uint8)
dummy1 = np.zeros((batch_size + 7) >> 3, dtype=np.uint8)
dummy2 = np.empty(batch_size, dtype=np.uint8)
pack_single_flags(_warm_flags, dummy1, dummy2) # compiling

def notify_start():
    if IMPORTED or platform.system() == "Linux": return # running on cloud gpu
    notification.notify(
        title="rl",
        message="rl training has started",
        app_name="rl script",
        timeout=10
    )

m = np.empty((batch_size, 4), dtype=np.int64)
moves = np.empty(batch_size, dtype=np.uint8)

# world-frame ground truth
board_t = np.empty((batch_size, 15, 15), dtype=np.uint8)
head_t = np.empty((batch_size, 15, 15), dtype=np.uint8)
food_t = np.empty((batch_size, 15, 15), dtype=np.uint8)
headpos = np.empty((batch_size, 4, 2), dtype=np.uint8)
tailord = np.zeros((batch_size, 4, 15, 15), dtype=np.uint8)

# per-seat memory (world frame and never rotated)
mem_occ = np.zeros((batch_size, 4, 15, 15), dtype=np.uint8)
mem_food = np.zeros((batch_size, 4, 15, 15), dtype=np.uint8)
mem_head = np.zeros((batch_size, 4, 15, 15), dtype=np.uint8)
mem_age = np.full((batch_size, 4, 15, 15), 255, dtype=np.uint8)
est_len = np.full((batch_size, 4, 4), 3, dtype=np.uint8)
rot_buf = np.zeros((batch_size, 4), dtype=np.int8)

# observation planes (uint8; float32 at (T,B,9,15,15), ~1.4 GB)
obs_u8 = np.zeros((batch_size, 4, om.C, om.CROP, om.CROP), dtype=np.uint8)
scal_u8 = np.zeros((batch_size, 4, 5), dtype=np.uint8) # health, len, timestep, hx, hy
priv_u8 = np.zeros((batch_size, PRIV_CH, 15, 15), dtype=np.uint8)
act3 = np.zeros((batch_size, 4), dtype=np.uint8)

def reset_memory():
    mem_occ.fill(0); mem_food.fill(0); mem_head.fill(0)
    mem_age.fill(255); est_len.fill(3)

SEAT = 0 # the learner's seat. Under self-play the other three are pool policies.
OBS_SEAT = -1 if SELFPLAY else SEAT # -1 = build memory + channels for all four seats

def refresh_obs(gs, timestep, seat=OBS_SEAT):
    """ground truth -> per-seat memory -> canonical 9-channel crop."""
    om.build_true_boards(gs['data'], gs['food'], gs['length'], timestep, board_t, head_t, food_t, headpos, tailord, seat)
    om.update_memory(board_t, head_t, food_t, headpos, gs['new_apple'], gs['length'], mem_occ, mem_food, mem_head, mem_age, est_len, om.VIEW_DX, om.VIEW_DY, seat)
    om.heading_rot_all(gs['data'], timestep, rot_buf)
    om.write_channels(board_t, food_t, headpos, tailord, mem_occ, mem_food, mem_head, mem_age, est_len, gs['length'], rot_buf, om.ROT_OFF,obs_u8, scal_u8, timestep, seat)


@njit(cache=True)
def build_priv(board, head, foodg, headpos, rot, rot_off, out):
    """privileged unfogged full-board encoding for the critic, in seat 0's canonical frame.
    Not used in inference"""
    B = board.shape[0]
    h = 15 // 2
    for g in range(B):
        hx = int(headpos[g, 0, 0])
        for c in range(out.shape[1]):
            for y in range(15):
                for x in range(15):
                    out[g, c, y, x] = 0
        if hx == 255:
            continue
        hy = int(headpos[g, 0, 1])
        r = int(rot[g, 0])
        for cy in range(15):
            for cx in range(15):
                wx = hx + rot_off[r, cy, cx, 0]
                wy = hy + rot_off[r, cy, cx, 1]
                if wx < 0 or wx > 14 or wy < 0 or wy > 14:
                    out[g, 6, cy, cx] = 1 # offboard plane
                    continue
                v = board[g, wy, wx]
                if v != 255:
                    out[g, v, cy, cx] = 1 # planes 0-3: true occupancy by seat
                if head[g, wy, wx] != 0:
                    out[g, 4, cy, cx] = 1
                if foodg[g, wy, wx] != 0:
                    out[g, 5, cy, cx] = 1


def _scal_norm(sc):
    # health, length, timestep (stored capped at 255 so /255 stays in [0,1]), hx, hy
    return torch.stack((sc[:, 0] / 100.0, sc[:, 1] / om.LEN_MAX, sc[:, 2] / 255.0, sc[:, 3] / 14.0, sc[:, 4] / 14.0), dim=1)

def obs_tensor(seat=0):
    x = torch.from_numpy(obs_u8[:, seat]).to(device)
    sc = torch.from_numpy(scal_u8[:, seat]).to(device).float()
    return om.dequantise(x), _scal_norm(sc)

def gather_obs(gi, ki):
    """observations for an arbitrary set of (game, seat) slots -> (N, ...)"""
    x = torch.from_numpy(obs_u8[gi, ki]).to(device)
    sc = torch.from_numpy(scal_u8[gi, ki]).to(device).float()
    return om.dequantise(x), _scal_norm(sc)

def priv_tensor():
    return torch.from_numpy(priv_u8).to(device).float()

def random_moves(m, out):
    m[:] = np.random.randint(0, 4, (batch_size, 4))
    out[:] = (m[:, 0] | (m[:, 1] << 2) | (m[:, 2] << 4) | (m[:, 3] << 6)).astype(np.uint8)

@njit(cache=True)
def pack_moves3(a3, rot):
    """canonical 3-action (left/straight/right) -> engine move, per seat."""
    canon = om.ACT3_TO_CANON[a3]
    raw = INV_ROT_ACTION[rot.astype(np.int64), canon]
    return (raw[:, 0] | (raw[:, 1] << 2) |
            (raw[:, 2] << 4) | (raw[:, 3] << 6)).astype(np.uint8)

# pool entries are CPU state_dicts. opponent modules are built with the SAME
# priv_ch as the learner so key names line up; they are simply called with
# priv=None, which makes forward() return value=None and skip the critic.
POOL_PATH = os.path.join(CKPT_DIR, "pool.pt")
TOTAL_OPP_SLOTS = 3 * batch_size
N_SLOT = int(TOTAL_OPP_SLOTS * (1.0 - SCRIPTED_FRAC)) // OPP_SAMPLE
N_NET_SLOTS = N_SLOT * OPP_SAMPLE
N_ENGINE_SLOTS = int(TOTAL_OPP_SLOTS * SHAPESHIFTER_OPP_FRAC)

pool = []
pool_wins = []  # parallel to pool: learner wins against entry i
pool_games = [] # parallel to pool: total games played against entry i
opp_nets = []
if SELFPLAY and N_SLOT > 0:
    opp_nets = [SnakeNet(priv_ch=PRIV_CH, **CLUSTER_CFG).to(device).eval()
                for _ in range(OPP_SAMPLE)]
    for n in opp_nets:
        for p in n.parameters():
            p.requires_grad_(False)
elif SELFPLAY:
    print("[warn] batch_size too small for OPP_SAMPLE; falling back to scripted opponents")

def snapshot_weights():
    sd = net.state_dict()
    if POOL_STRIP_CRITIC:
        sd = {k: v for k, v in sd.items()
              if not (k.startswith("priv_enc.") or k.startswith("v."))}
    return {k: v.detach().to("cpu", POOL_DTYPE).clone() for k, v in sd.items()}

def load_pool():
    if not (SELFPLAY and os.path.exists(POOL_PATH)):
        return
    try:
        data = torch.load(POOL_PATH, map_location="cpu")
        pool.extend(data["pool"])
        pool_wins.extend(data.get("pool_wins", [0] * len(pool)))
        pool_games.extend(data.get("pool_games", [0] * len(pool)))
        print(f"loaded opponent pool: {len(pool)} models")
    except Exception as e:
        print(f"[warn] pool load failed, starting empty: {e}")

def save_pool(episode):
    try:
        tmp = POOL_PATH + ".tmp"
        torch.save({"pool": pool, "episode": episode, "cfg": CLUSTER_CFG,
                     "pool_wins": pool_wins, "pool_games": pool_games}, tmp)
        os.replace(tmp, POOL_PATH)
    except (RuntimeError, OSError) as e:
        print(f"[warn] pool save failed at episode {episode}, skipping: {e}")


# in the final full training run no evictions were ever made.
def maybe_admit(win_rate, episode):
    """admit the current policy if it clears ADMIT_WIN_RATE, then evict.

    eviction is random over the non-anchor, non-newest range rather than FIFO:
    dropping the oldest first would steadily purge exactly the early opponents
    that stop the learner cycling back into strategies it already beat."""
    if not SELFPLAY or win_rate < ADMIT_WIN_RATE:
        return False
    pool.append(snapshot_weights())
    pool_wins.append(0)
    pool_games.append(0)
    while len(pool) > OPP_POOL_CAP:
        lo, hi = POOL_KEEP_ANCHORS, len(pool) - 1
        evict = np.random.randint(lo, hi) if hi > lo else 0
        pool.pop(evict)
        pool_wins.pop(evict)
        pool_games.pop(evict)
    save_pool(episode)
    return True

def slice_gs(gs, g_idx):
    """safely slice game_state dict, repack the 'flags' array."""
    sub_gs = {}
    for k, v in gs.items():
        if k == "flags":
            new_flags = np.zeros((len(g_idx) + 1) // 2, dtype=np.uint8)
            for i, g in enumerate(g_idx):
                nib = (v[g >> 1] >> ((g & 1) * 4)) & 0xF
                new_flags[i >> 1] |= (nib << ((i & 1) * 4))
            sub_gs[k] = new_flags
        elif isinstance(v, np.ndarray):
            sub_gs[k] = v[g_idx]
        else:
            sub_gs[k] = v
    return sub_gs

def new_engine_matchup(seed=None):
    if N_ENGINE_SLOTS == 0:
        return None, None
    rng = np.random.default_rng(seed)
    s = rng.choice(TOTAL_OPP_SLOTS, N_ENGINE_SLOTS, replace=False)
    gi = s // 3
    ki = (s % 3) + 1
    return gi, ki

def pfsp_probs():
    """PFSP sampling weights: p(i) = (1 - win_rate_i + epsilon)^power.
    Returns None (= uniform) when PFSP is off or no stats exist yet."""
    n = len(pool)
    if not PFSP or n == 0:
        return None
    w = np.ones(n, dtype=np.float64)
    for i in range(n):
        if pool_games[i] > 0:
            wr = pool_wins[i] / pool_games[i]
            w[i] = (1.0 - wr + PFSP_EPS) ** PFSP_POWER
    return w / w.sum()


def new_matchup(seed=None, use_pfsp=True):
    """load OPP_SAMPLE pool models and fix a slot->model map for one episode.

    Returns ((g_idx, k_idx), pool_indices), or (None, None) when the pool
    is empty / self-play is off, in which case every seat stays on safe_moves3.
    pool_indices has shape (OPP_SAMPLE,) and records which pool entry each
    opp_net was loaded from, so PFSP stats can be attributed after the episod"""
    if not (SELFPLAY and pool and opp_nets):
        return None, None
    rng = np.random.default_rng(seed)
    probs = pfsp_probs() if use_pfsp else None
    idx = rng.choice(len(pool), OPP_SAMPLE, replace=len(pool) < OPP_SAMPLE, p=probs)
    for n, i in zip(opp_nets, idx):
        n.load_state_dict(pool[i], strict=not POOL_STRIP_CRITIC)
    s = rng.permutation(TOTAL_OPP_SLOTS)[:N_NET_SLOTS]
    gi = (s // 3).reshape(OPP_SAMPLE, N_SLOT)
    ki = (s % 3 + 1).reshape(OPP_SAMPLE, N_SLOT)
    return (gi, ki), idx

def update_pfsp_stats(matchup_idx, gi, won):
    """attribute training-episode outcomes to pool entries for PFSP weighting"""
    if matchup_idx is None or not PFSP:
        return
    for i in range(OPP_SAMPLE):
        games = np.unique(gi[i])
        pidx = matchup_idx[i]
        pool_games[pidx] += len(games)
        pool_wins[pidx] += int(won[games].sum())


def new_opp_state():
    return [(torch.zeros(N_SLOT, memory_latent_dim, device=device),
             torch.zeros(N_SLOT, memory_latent_dim, device=device))
            for _ in range(OPP_SAMPLE)]

def opp_step(matchup, eng_matchup, hc):
    """One opponent tick. Writes canonical 3-actions into act3 in place, leaving
    scripted slots and dead seats with whatever safe_moves3 already put there."""
    if matchup is not None:
        gi, ki = matchup
        flat_g, flat_k = gi.ravel(), ki.ravel()
        x, sc = gather_obs(flat_g, flat_k)
        x = x.view(OPP_SAMPLE, N_SLOT, om.C, om.CROP, om.CROP)
        sc = sc.view(OPP_SAMPLE, N_SLOT, 5)
        outs = []
        for i, n in enumerate(opp_nets):
            with torch.no_grad(), amp_ctx():
                lg, _, hc[i] = n(x[i], sc[i], hc[i], priv=None)
            lg = lg.float()
            outs.append(sample_actions(lg)[0] if OPP_SAMPLE_ACTIONS else lg.argmax(-1))
        act3[flat_g, flat_k] = torch.stack(outs).view(-1).cpu().numpy().astype(np.uint8)

_init_data = generate_init_data(batch_size)
game_state = {
    "data":      _init_data.copy(),
    "flags":     np.empty(batch_size // 2, dtype=np.uint8),
    "food":      np.empty((batch_size, 29), dtype=np.uint8),
    "length":    np.empty((batch_size, 4), dtype=np.uint8),
    "health":    np.empty((batch_size, 4), dtype=np.uint8),
    "timestep":  np.uint32(0),
    "new_apple": np.full((batch_size, 2), np.uint8(255), dtype=np.uint8),
}

def reset_game_state(gs):
    np.copyto(gs["data"], _init_data)
    gs["flags"].fill(0xFF)
    gs["food"].fill(0)
    gs["length"].fill(3)
    gs["health"].fill(100)
    gs["timestep"] = np.uint32(0)
    gs["new_apple"].fill(255)

def alive_mask(data, timestep):
    # same sentinel the rest of the file uses: x == 0xF means no head this tick (dead)
    cur = timestep % 225
    vals = data[:, :, cur]
    return (vals & 0xF) != 0xF

# potential shaping weights
W_HEALTH, W_LEN, W_OPP = 0.07, 0.03, 0.05
LEN_MAX = 30.0

def potential(alive, health, length, n_opp_alive):
    phi = (W_HEALTH * (health.astype(np.float32) / 100.0)
           + W_LEN * (length.astype(np.float32) / LEN_MAX)
           - W_OPP * n_opp_alive)
    return np.where(alive, phi, 0.0).astype(np.float32)

# prefix sums of PLACE_R so k snakes dying together can SHARE their k tied placements.
cumulative_placement = np.concatenate(([0.0], np.cumsum(PLACE_R))).astype(np.float32)

def step_rewards(alive_prev, alive_cur, health_prev, health_cur, length_prev, length_cur):
    """terminal reward is placement-based, mean-centred [2,1,0,0] -> PLACE_R"""
    just_died = alive_prev & ~alive_cur
    rewards = np.zeros((batch_size, 4), dtype=np.float32)
    n_prev = alive_prev.sum(1)
    n_cur = alive_cur.sum(1)
    contested = n_prev >= 2

    # k deaths leaving s survivors occupy places s+1..s+k -> average their rewards
    n_died = just_died.sum(1)
    span = np.minimum(n_cur + n_died, 4)
    shared = ((cumulative_placement[span] - cumulative_placement[np.minimum(n_cur, 4)])
              / np.maximum(n_died, 1)).astype(np.float32)
    shared *= contested
    rewards[just_died] = shared[np.where(just_died)[0]]

    # if there is a single snake, its first
    decided = contested & (n_cur == 1)
    if decided.any():
        w = np.argmax(alive_cur, axis=1)
        rewards[decided, w[decided]] += PLACE_R[0]

    opp_prev = n_prev[:, None] - alive_prev
    opp_cur = n_cur[:, None] - alive_cur
    phi_prev = potential(alive_prev, health_prev, length_prev, opp_prev)
    phi_cur = potential(alive_cur, health_cur, length_cur, opp_cur)
    phi_cur = phi_cur * (n_cur > 1)[:, None] # decided games are terminal states, last tick doesnt work since games vary in length
    rewards += alive_prev * (gamma * phi_cur - phi_prev)
    return rewards


def sample_actions(logits):
    # torch.distributions.Categorical was just way to slow for some reason
    log_probs_all = F.log_softmax(logits, dim=-1)
    u = torch.rand_like(logits).clamp_min(1e-20)
    gumbel = -torch.log(-torch.log(u))
    actions = (logits + gumbel).argmax(dim=-1)
    log_prob = log_probs_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    probs_all = log_probs_all.exp()
    entropy = -(probs_all * log_probs_all).sum(-1)
    return actions, log_prob, entropy

def evaluate_actions(logits, actions):
    log_probs_all = F.log_softmax(logits, dim=-1)
    log_prob = log_probs_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    probs_all = log_probs_all.exp()
    entropy = -(probs_all * log_probs_all).sum(-1)
    return log_prob, entropy


def compute_gae(rewards_buf, values_buf, nonterm_buf, last_values, gamma, lam):
    # nonterm_buf[t] gates the bootstrap from state t+1, so being alive needs to be calculated from alive_cur not alive_prev
    # otherwise the GAE will leak signal past when the snake died
    T = len(rewards_buf)
    advantages = [None] * T
    gae = torch.zeros_like(last_values)
    next_value = last_values
    for t in reversed(range(T)):
        delta = rewards_buf[t] + gamma * next_value * nonterm_buf[t] - values_buf[t]
        gae = delta + gamma * lam * nonterm_buf[t] * gae
        advantages[t] = gae
        next_value = values_buf[t]
    advantages = torch.stack(advantages)
    returns = advantages + torch.stack(values_buf)
    return returns, advantages


DEATH_CAUSES = {1: "starved", 2: "wall", 3: "head_to_head", 4: "body", 5: "invalid_state",
                0: "survived"}
DEATH_CSV = os.path.join(CKPT_DIR, "death_causes.csv")
_CAUSE_ORDER = (1, 2, 3, 4, 5, 0)


def log_death_causes(episode, counts, n_games): # model debug
    try:
        header_needed = not os.path.exists(DEATH_CSV) or os.path.getsize(DEATH_CSV) == 0
        with open(DEATH_CSV, "a", newline="") as fh:
            w = csv.writer(fh)
            if header_needed:
                w.writerow(["episode", "n_games"] + [DEATH_CAUSES[k] for k in _CAUSE_ORDER])
            w.writerow([episode, n_games] + [int(counts.get(k, 0)) for k in _CAUSE_ORDER])
    except Exception:
        pass


def evaluate_winrate(opponents=EVAL_VS, seed=EVAL_SEED):
    reset_game_state(game_state)
    reset_memory()
    matchup, _ = new_matchup(seed, use_pfsp=False) if opponents == 'pool' else (None, None)
    opp_hc = new_opp_state() if matchup else None
    eval_h = (torch.zeros(batch_size, memory_latent_dim, device=device),
              torch.zeros(batch_size, memory_latent_dim, device=device))
    alive_prev = alive_mask(game_state['data'], 0)
    done = np.zeros(batch_size, dtype=bool)
    won = np.zeros(batch_size, dtype=bool)
    death_tick = np.full(batch_size, game_step_dur, dtype=np.int32)
    cause0 = np.zeros(batch_size, dtype=np.uint8)

    with torch.no_grad():
        for t in range(game_step_dur):
            refresh_obs(game_state, t)
            x, sc = obs_tensor(0)
            with amp_ctx():
                logits, _, eval_h = net(x, sc, eval_h, priv=None)
            om.safe_moves3(board_t, headpos, rot_buf, om.ROT_OFF, act3)
            if matchup or eng_matchup is not None:
                opp_step(matchup, eng_matchup, opp_hc)
            act3[:, 0] = logits.argmax(-1).cpu().numpy().astype(np.uint8)
            packed = pack_moves3(act3, rot_buf)
            _, _, death_cause = game.step_wrapper(game_state, packed)

            alive_cur = alive_mask(game_state['data'], t + 1)
            # ~done: once the game is decided the winner's ring buffer goes stale
            # and alive_mask fakes a death next tick -- don't log it as one.
            just_died0 = alive_prev[:, 0] & ~alive_cur[:, 0] & ~done
            death_tick[just_died0] = t + 1
            cause0[just_died0] = death_cause[0][just_died0]

            n_prev, n_cur = alive_prev.sum(1), alive_cur.sum(1)
            finished = (n_prev > 1) & (n_cur == 1)
            if finished.any():
                won[finished] |= (np.argmax(alive_cur, axis=1)[finished] == 0)
            done |= n_cur <= 1
            alive_prev = alive_cur
            print(f"\reval progress: {(t+1)/game_step_dur*100:5.1f}%", end="", flush=True)

    print("\r" + " " * 30 + "\r", end="", flush=True)
    counts = {k: int((cause0 == k).sum()) for k in _CAUSE_ORDER}
    return won.mean(), death_tick.mean(), (death_tick <= 5).mean(), counts


print('np data alloc done')

if FIXED_OPP:
    for _p in OPP_CKPT_PATHS:
        _ckpt = torch.load(_p, map_location="cpu")
        _sd = _ckpt["model"] if "model" in _ckpt else _ckpt
        if POOL_STRIP_CRITIC:
            _sd = {k: v for k, v in _sd.items()
                   if not (k.startswith("priv_enc.") or k.startswith("v."))}
        _sd = {k: v.to(POOL_DTYPE) for k, v in _sd.items()}
        pool.append(_sd)
        pool_wins.append(0)
        pool_games.append(0)
    print(f"[fixed-opp] loaded {len(pool)} checkpoint opponent(s): "
          + ", ".join(os.path.basename(p) for p in OPP_CKPT_PATHS))
else:
    load_pool()
    if SELFPLAY and not pool and SEED_POOL_WITH_INIT:
        pool.append(snapshot_weights())
        pool_wins.append(0)
        pool_games.append(0)
        print("seeded opponent pool with the current (untrained) net")
if SELFPLAY:
    _mb = sum(v.numel() * v.element_size() for v in snapshot_weights().values()) / 2**20
    print(f"self-play: {len(pool)}/{OPP_POOL_CAP} in pool, {OPP_SAMPLE} sampled/ep, "
          f"{N_SLOT} slots each ({N_NET_SLOTS}/{TOTAL_OPP_SLOTS} opponent seats, rest scripted), "
          f"~{_mb:.1f} MB per entry / ~{_mb * OPP_POOL_CAP:.0f} MB at cap")

print("warming up:")
m = np.empty((batch_size, 4), dtype=np.int64)
moves = np.empty(batch_size, dtype=np.uint8)
reset_game_state(game_state)
refresh_obs(game_state, 0)
print("\t- refresh_obs (memory + channels)")
build_priv(board_t, head_t, food_t, headpos, rot_buf, om.ROT_OFF, priv_u8)
print("\t- build_priv")
om.safe_moves3(board_t, headpos, rot_buf, om.ROT_OFF, act3)
print("\t- safe_moves3")
for i in range(50):
    random_moves(m, moves)
    game.step_wrapper(game_state, moves)
    print(f"\t- game step wrapper {i+1}/50", end='\r')
print()

print('warm up done')

notify_start()

start = time.time()
total_ticks = 0
print("note: total ticks ignores any ticks prior to resuming from a checkpoint")
pbar = tqdm(total=game_batch_count, desc="Rollouts", unit="Ep", ncols=160, disable=IMPORTED, initial=start_episode) # ep = episode

profiler = Profiler()
if not IMPORTED:
    profiler.start()

for episode in range(start_episode, game_batch_count):
    reset_game_state(game_state)
    h0 = (torch.zeros(batch_size, memory_latent_dim, device=device),
          torch.zeros(batch_size, memory_latent_dim, device=device))
    alive_prev = alive_mask(game_state['data'], 0)
    done = np.zeros(batch_size, dtype=bool) # game decided (<=1 alive): mask out everything after
    entropy_coef = max(0.01, 0.02 * (1 - episode / (game_batch_count * 3)))

    reset_memory()
    matchup, matchup_idx = new_matchup()
    matchup, matchup_idx = new_matchup()
    eng_matchup = new_engine_matchup()
    opp_hc = new_opp_state() if matchup else None
    won = np.zeros(batch_size, dtype=bool)
    refresh_obs(game_state, 0)
    build_priv(board_t, head_t, food_t, headpos, rot_buf, om.ROT_OFF, priv_u8)
    cur_x, cur_sc = obs_tensor(0)
    cur_priv = priv_tensor()

    obs_seq, scal_seq, priv_seq, actions_seq = [], [], [], []
    old_logp_seq, old_val_seq, rewards_seq, masks_seq, nonterm_seq = [], [], [], [], []
    exit_target_seq, exit_valid_seq = [], []
    h_roll = h0
    with torch.no_grad():
        for t in range(game_step_dur):
            live0 = alive_prev[:, 0] & ~done
            if not live0.any():
                break

            with amp_ctx():
                logits, values, h_roll = net(cur_x, cur_sc, h_roll, priv=cur_priv)
            logits, values = logits.float(), values.float()
            actions_live, log_prob_live, _ = sample_actions(logits)

            obs_seq.append(torch.from_numpy(obs_u8[:, 0].copy()))
            scal_seq.append(cur_sc) 
            priv_seq.append(torch.from_numpy(priv_u8.copy()))
            actions_seq.append(actions_live)
            old_logp_seq.append(log_prob_live)
            old_val_seq.append(values)

            if ExIT:
                if t % EXIT_STRIDE == 0:
                    eng_moves, eng_reasons = solve_batch(game_state, seat=0, timeout_ms=ENGINE_TIMEOUT)
                    tgt, tgt_valid = engine_to_act3(eng_moves, rot_buf)
                    if t == 0 and episode == start_episode:
                        n_ok = sum(1 for r in eng_reasons if r == 'ok')
                        n_non_none = sum(1 for m in eng_moves if m is not None)
                        tqdm.write(f"[ExIT diag] tick 0: engine returned {n_non_none}/{len(eng_moves)} non-None moves, "
                                   f"{n_ok} ok reasons, {tgt_valid.sum()} valid after conversion")
                        for g in range(min(5, len(eng_moves))):
                            mv = eng_moves[g]
                            r = int(rot_buf[g, 0])
                            canon = ROT_ACTION[r, int(mv)] if mv is not None else -1
                            a3 = int(CANON_TO_ACT3[canon]) if canon >= 0 else -1
                            tqdm.write(f"  g={g}: engine={mv} reason={eng_reasons[g]} rot={r} "
                                       f"canon={canon} a3={a3} INV_ROT shape={INV_ROT_ACTION.shape}")
                    exit_target_seq.append(torch.from_numpy(tgt.copy()))
                    exit_valid_seq.append(torch.from_numpy(tgt_valid.copy()))
                else:
                    exit_target_seq.append(torch.zeros(batch_size, dtype=torch.int64))
                    exit_valid_seq.append(torch.zeros(batch_size, dtype=torch.bool))

            # safe_moves3 first: it fills scripted slots and dead seats, then the
            # pool policies overwrite the slots they own, the learner takes seat 0
            om.safe_moves3(board_t, headpos, rot_buf, om.ROT_OFF, act3)
            if matchup or eng_matchup is not None:
                opp_step(matchup, eng_matchup, opp_hc)
            act3[:, 0] = actions_live.cpu().numpy().astype(np.uint8)
            moves = pack_moves3(act3, rot_buf)

            health_prev = game_state['health'].copy()
            length_prev = game_state['length'].copy()
            game.step_wrapper(game_state, moves)

            alive_cur = alive_mask(game_state['data'], t + 1)
            rewards = step_rewards(alive_prev, alive_cur, health_prev, game_state['health'], length_prev, game_state['length'])
            n_cur = alive_cur.sum(1)
            nonterm0 = alive_cur[:, 0] & (n_cur > 1)
            rewards_seq.append(torch.from_numpy(rewards[:, 0]).to(device))
            masks_seq.append(torch.from_numpy(live0.astype(np.float32)).to(device))
            nonterm_seq.append(torch.from_numpy(nonterm0.astype(np.float32)).to(device))
            n_prev = alive_prev.sum(1)
            finished = (n_prev >= 2) & (n_cur == 1) & ~done
            if finished.any():
                won[finished] = alive_cur[finished, 0]
            done |= n_cur <= 1
            alive_prev = alive_cur

            refresh_obs(game_state, t + 1)
            build_priv(board_t, head_t, food_t, headpos, rot_buf, om.ROT_OFF, priv_u8)
            cur_x, cur_sc = obs_tensor(0)
            cur_priv = priv_tensor()

        # truncation bootstrap: V(s_T) for trajectories still alive at the horizon
        with amp_ctx():
            _, last_values, _ = net(cur_x, cur_sc, h_roll, priv=cur_priv)
        last_values = last_values.float()

    if matchup is not None:
        update_pfsp_stats(matchup_idx, matchup[0], won)

    T = len(rewards_seq)
    ep_policy_loss = ep_value_term = ep_entropy_term = ep_entropy = 0.0
    ep_grad_norm_sum, ep_update_count = 0.0, 0

    if T > 0 and ExIT:
        masks_t = torch.stack(masks_seq)
        exit_targets = torch.stack(exit_target_seq).to(device)   # (T, B) int64
        exit_valids = torch.stack(exit_valid_seq).to(device)     # (T, B) bool
        ce_mask = masks_t * exit_valids.float().to(device)
        ce_denom = ce_mask.sum().clamp_min(1.0)

        returns, _ = compute_gae(rewards_seq, old_val_seq, nonterm_seq,
                                 last_values, gamma, gae_lambda)
        val_denom = masks_t.sum().clamp_min(1.0)
        returns = returns.detach()

        for _ in range(ppo_epochs):
            x_all = om.dequantise(torch.stack(obs_seq).to(device).flatten(0, 1))
            p_all = torch.stack(priv_seq).to(device).flatten(0, 1).float()
            sc_all = torch.stack(scal_seq).flatten(0, 1)
            with amp_ctx():
                logits, new_val_seq, _ = net.forward_seq(
                    x_all, sc_all, h0, p_all, T, batch_size)
            logits = logits.float().view(T, batch_size, -1)
            new_val_seq = new_val_seq.float()

            ce = F.cross_entropy(logits.reshape(T * batch_size, -1),
                                 exit_targets.reshape(-1), reduction='none')
            policy_loss = (ce.view(T, batch_size) * ce_mask).sum() / ce_denom

            value_loss = ((new_val_seq - returns) ** 2 * masks_t).sum() / val_denom

            log_p = F.log_softmax(logits, dim=-1)
            ent = -(log_p.exp() * log_p).sum(-1)
            entropy_loss = -(ent * ce_mask).sum() / ce_denom

            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
            optimizer.step()

            ep_policy_loss += policy_loss.item()
            ep_value_term += (value_coef * value_loss).item()
            ep_entropy_term += (entropy_coef * entropy_loss).item()
            ep_entropy += (ent * ce_mask).sum().item() / ce_denom.item()
            ep_grad_norm_sum += grad_norm.item()
            ep_update_count += 1

    elif T > 0:
        old_logp_seq = torch.stack(old_logp_seq).detach() # (T, B)
        masks_t = torch.stack(masks_seq) # (T, B)
        actions_t = torch.stack(actions_seq) # (T, B)

        returns, advantages = compute_gae(rewards_seq, old_val_seq, nonterm_seq, last_values, gamma, gae_lambda)
        denom = masks_t.sum().clamp_min(1.0)

        adv_mean = (advantages * masks_t).sum() / denom
        adv_var = ((advantages - adv_mean) ** 2 * masks_t).sum() / denom
        advantages = ((advantages - adv_mean) / (adv_var.sqrt() + 1e-8)).detach()
        returns = returns.detach()

        for _ in range(ppo_epochs):
            x_all = om.dequantise(torch.stack(obs_seq).to(device).flatten(0, 1))
            p_all = torch.stack(priv_seq).to(device).flatten(0, 1).float()
            sc_all = torch.stack(scal_seq).flatten(0, 1)
            with amp_ctx():
                logits, new_val_seq, _ = net.forward_seq(
                    x_all, sc_all, h0, p_all, T, batch_size)
                new_logp_seq, new_ent_seq = evaluate_actions(
                    logits.view(T * batch_size, -1), actions_t.view(-1))
            new_logp_seq = new_logp_seq.view(T, batch_size).float()
            new_ent_seq = new_ent_seq.view(T, batch_size).float()
            new_val_seq = new_val_seq.float()

            ratio = torch.exp(new_logp_seq - old_logp_seq)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
            policy_loss = -(torch.min(surr1, surr2) * masks_t).sum() / denom
            value_loss = ((new_val_seq - returns) ** 2 * masks_t).sum() / denom
            entropy_loss = -(new_ent_seq * masks_t).sum() / denom
            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
            optimizer.step()

            ep_policy_loss += policy_loss.item()
            ep_value_term += (value_coef * value_loss).item()
            ep_entropy_term += (entropy_coef * entropy_loss).item()
            ep_entropy += (new_ent_seq * masks_t).sum().item() / denom.item()
            ep_grad_norm_sum += grad_norm.item()
            ep_update_count += 1

    if T > 0 and (episode + 1) % 20 == 0:
        stacked_r = torch.stack(rewards_seq)
        masked_r = stacked_r[masks_t.bool()]
        extra = ""
        if ExIT:
            act_t = torch.stack(actions_seq)  # (T, B)
            et = torch.stack(exit_target_seq).to(device)
            ev = torch.stack(exit_valid_seq).to(device)
            agree_mask = masks_t.bool() & ev
            if agree_mask.any():
                agree = (act_t.to(device)[agree_mask] == et[agree_mask]).float().mean().item()
                extra = f"  engine_agree: {agree:.1%}"
        tqdm.write(f"reward mean: {masked_r.mean().item():.4f} std: {masked_r.std().item():.2f} | entropy (sum): {ep_entropy:.4f}\tpolicy_loss: {ep_policy_loss:.3f}  value_term: {ep_value_term:.2f}  entropy_term: {ep_entropy_term:.2f} |\tavg grad_norm: {ep_grad_norm_sum / max(1, ep_update_count):.2f} | episode_len: {T}{extra}")

    if (episode + 1) % EVAL_EVERY == 0:
        win_rate, avg_survival, instant_death_rate, death_counts = evaluate_winrate()
        log_death_causes(episode, death_counts, batch_size)
        admitted = maybe_admit(win_rate, episode)
        msg = (f"episode {episode + 1}: win rate vs {EVAL_VS} = {win_rate:.3f}  "
               f"avg snake0 survival = {avg_survival:.1f} ticks  "
               f"died<=5 ticks = {instant_death_rate:.2%}")
        if SELFPLAY:
            msg += (f"  pool = {len(pool)}/{OPP_POOL_CAP}"
                    f"{' [ADMITTED]' if admitted else ''}")
        # self-play win rate is self-referential -- it can sit flat while skill
        # climbs or collapses. The scripted number is the only absolute yardstick.
        if BASELINE_EVAL and EVAL_VS == 'pool':
            base_wr, base_surv, _, _ = evaluate_winrate(opponents='scripted')
            msg += f"  | vs scripted: win {base_wr:.3f}, survival {base_surv:.1f}"
        tqdm.write(msg)
        save_checkpoint(episode)
    pbar.update(1)
    total_ticks += T * batch_size
    pbar.set_postfix(ticks=total_ticks, tick_rate=f"{total_ticks/(time.time()-start):.0f}/s")

pbar.close()
end = time.time()

if not IMPORTED:
    profiler.stop()
    profiler.print()

print(game_state['data'].shape)

def notify_done():
    if IMPORTED: return
    if platform.system() == "Linux": 
        requests.get("") # canary token for the emaling
        return
    notification.notify(
        title="rl",
        message="rl training has finished",
        app_name="rl script",
        timeout=10
    )

notify_done()