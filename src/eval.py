import os
os.environ["FIXED_DEPTH"] = "6" # shapeshifter

import argparse
import sys
import numpy as np
import torch
from tqdm import tqdm
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import game
from game import generate_init_data
import obsmem as om
from snakenet import SnakeNet, INV_ROT_ACTION
from shapeshifter_repo.proxy import solve_batch

CHECKPOINTS = [
    "ckpt_ep67099.pt",
    "ckpt_ep61299.pt",
    "ckpt_ep59699.pt",
    "ckpt_ep57499.pt",
]

NET_KWARGS = dict(ch=(32, 64), squeeze=24, hid=256, mem=128, priv_ch=7)
ENGINE_TIMEOUT_MS = 1000

N_ACT = 3

ROT_ACTION = np.zeros_like(INV_ROT_ACTION)
for _r in range(INV_ROT_ACTION.shape[0]):
    for _c in range(INV_ROT_ACTION.shape[1]):
        ROT_ACTION[_r, INV_ROT_ACTION[_r, _c]] = _c

CANON_TO_ACT3 = np.full(4, -1, dtype=np.int64)
for _a3 in range(N_ACT):
    CANON_TO_ACT3[om.ACT3_TO_CANON[_a3]] = _a3

def engine_to_act3(engine_moves, rot, seat=0):
    B = len(engine_moves)
    actions = np.zeros(B, dtype=np.int64)
    valid = np.zeros(B, dtype=bool)
    for g in range(B):
        mv = engine_moves[g]
        if mv is None:
            continue
        r = int(rot[g, seat])
        canon = ROT_ACTION[r, int(mv)]
        a3 = CANON_TO_ACT3[canon]
        if a3 >= 0:
            actions[g] = a3
            valid[g] = True
    return actions, valid

def pack_moves3(a3, rot):
    canon = om.ACT3_TO_CANON[a3]
    raw = INV_ROT_ACTION[rot.astype(np.int64), canon]
    return (raw[:, 0] | (raw[:, 1] << 2) |
            (raw[:, 2] << 4) | (raw[:, 3] << 6)).astype(np.uint8)

def load_player(path, device):
    if path == "shapeshifter":
        return "engine"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = SnakeNet(**NET_KWARGS).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    return net

def run_tournament(paths, N, max_ticks, device, seed=None, engine_timeout=ENGINE_TIMEOUT_MS, debug=False):
    engine_pos = [i for i, p in enumerate(paths) if p == "shapeshifter"]
    if engine_pos and engine_pos[0] != 0:
        idx = engine_pos[0]
        print(f"Warning: engine only works as seat 0. Moving it from seat {idx} to seat 0.")
        paths = ["shapeshifter"] + [p for i, p in enumerate(paths) if i != idx]
        print("New order:", paths)

    assert len(paths) == 4
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    names = []
    players = []
    for p in paths:
        if p == "shapeshifter":
            names.append("engine")
            players.append("engine")
        else:
            base = os.path.splitext(os.path.basename(p))[0]
            names.append(base)
            players.append(load_player(p, device))

    game.batch_size = N

    gs = {
        "data": generate_init_data(N),
        "flags": np.full((N + 1) // 2, np.uint8(0xFF), dtype=np.uint8),
        "food": np.zeros((N, 29), dtype=np.uint8),
        "length": np.full((N, 4), np.uint8(3), dtype=np.uint8),
        "health": np.full((N, 4), np.uint8(100), dtype=np.uint8),
        "timestep": np.uint32(0),
        "new_apple": np.full((N, 2), np.uint8(255), dtype=np.uint8),
    }

    board_t = np.empty((N, 15, 15), dtype=np.uint8)
    head_t = np.empty((N, 15, 15), dtype=np.uint8)
    food_t = np.empty((N, 15, 15), dtype=np.uint8)
    headpos = np.empty((N, 4, 2), dtype=np.uint8)
    tailord = np.zeros((N, 4, 15, 15), dtype=np.uint8)

    mem_occ = np.zeros((N, 4, 15, 15), dtype=np.uint8)
    mem_food = np.zeros((N, 4, 15, 15), dtype=np.uint8)
    mem_head = np.zeros((N, 4, 15, 15), dtype=np.uint8)
    mem_age = np.full((N, 4, 15, 15), 255, dtype=np.uint8)
    est_len = np.full((N, 4, 4), 3, dtype=np.uint8)
    rot_buf = np.zeros((N, 4), dtype=np.int8)

    obs_u8 = np.zeros((N, 4, om.C, om.CROP, om.CROP), dtype=np.uint8)
    scal_u8 = np.zeros((N, 4, 5), dtype=np.uint8)

    h_states, c_states = [], []
    for p in players:
        if p != "engine":
            h_states.append(torch.zeros(N, NET_KWARGS["mem"], device=device))
            c_states.append(torch.zeros(N, NET_KWARGS["mem"], device=device))
        else:
            h_states.append(None)
            c_states.append(None)

    def refresh_obs(timestep):
        om.build_true_boards(gs["data"], gs["food"], gs["length"], timestep, board_t, head_t, food_t, headpos, tailord, -1)
        om.update_memory(board_t, head_t, food_t, headpos, gs["new_apple"], gs["length"], mem_occ, mem_food, mem_head, mem_age, est_len, om.VIEW_DX, om.VIEW_DY, -1)
        om.heading_rot_all(gs["data"], timestep, rot_buf)
        om.write_channels(board_t, food_t, headpos, tailord, mem_occ, mem_food, mem_head, mem_age, est_len, gs["length"], rot_buf, om.ROT_OFF, obs_u8, scal_u8, timestep, -1)

    def scal_norm(sc):
        return torch.stack((sc[:, 0] / 100.0, sc[:, 1] / om.LEN_MAX, sc[:, 2] / 255.0, sc[:, 3] / 14.0, sc[:, 4] / 14.0), dim=1)

    def obs_tensor(seat):
        x = torch.from_numpy(obs_u8[:, seat]).to(device)
        sc = torch.from_numpy(scal_u8[:, seat]).float().to(device)
        return om.dequantise(x), scal_norm(sc)

    act3 = np.zeros((N, 4), dtype=np.uint8)
    placement = np.full((N, 4), -1, dtype=np.int64)
    survive_ticks = np.zeros((N, 4), dtype=np.int64)

    safe_act3 = np.zeros((N, 4), dtype=np.uint8)

    reason_counter = Counter()
    total_calls = 0

    pbar = tqdm(total=max_ticks, desc="evaluating", unit="tick")
    for t in range(max_ticks):
        nib = gs["flags"]
        alive_ct = np.array([bin(int((nib[g >> 1] >> ((g & 1) * 4)) & 0xF)).count("1")
                              for g in range(N)])
        if np.all(alive_ct <= 1):
            break

        refresh_obs(gs["timestep"])
        om.safe_moves3(board_t, headpos, rot_buf, om.ROT_OFF, safe_act3)

        for k in range(4):
            if players[k] == "engine":
                eng_moves, reasons = solve_batch(gs, seat=0, timeout_ms=engine_timeout)
                total_calls += 1
                for r in reasons:
                    reason_counter[r] += 1

                tgt, valid = engine_to_act3(eng_moves, rot_buf, seat=k)
                act3[:, k] = np.where(valid, tgt, safe_act3[:, k]).astype(np.uint8)
            else:
                x, sc = obs_tensor(k)
                with torch.no_grad():
                    logits, _, (h_states[k], c_states[k]) = players[k](x, sc, (h_states[k], c_states[k]))
                act3[:, k] = logits.argmax(-1).cpu().numpy().astype(np.uint8)

        # Use the correct LUT-based packing
        moves = pack_moves3(act3, rot_buf)

        pre_alive = np.zeros((N, 4), dtype=bool)
        for g in range(N):
            n = (gs["flags"][g >> 1] >> ((g & 1) * 4)) & 0xF
            for sk in range(4):
                pre_alive[g, sk] = bool((n >> sk) & 1)

        snakes_died, food_to_add, death_cause = game.step_wrapper(gs, moves)

        for g in range(N):
            n_alive_before = int(pre_alive[g].sum())
            for sk in range(4):
                if pre_alive[g, sk]:
                    survive_ticks[g, sk] += 1
                if snakes_died[sk, g] and placement[g, sk] == -1:
                    placement[g, sk] = n_alive_before

        pbar.update(1)
    pbar.close()

    # final placement
    for g in range(N):
        nb = (gs["flags"][g >> 1] >> ((g & 1) * 4)) & 0xF
        for sk in range(4):
            if (nb >> sk) & 1 and placement[g, sk] == -1:
                placement[g, sk] = 4

    wins = np.zeros(4, dtype=int)
    for g in range(N):
        nb = (gs["flags"][g >> 1] >> ((g & 1) * 4)) & 0xF
        if bin(int(nb)).count("1") == 1:
            for sk in range(4):
                if (nb >> sk) & 1:
                    wins[sk] += 1

    game_len = survive_ticks.max(axis=1)
    order = np.argsort(game_len)
    half = len(order) // 2
    short_mask = np.zeros(N, dtype=bool); short_mask[order[:half]] = True
    long_mask = np.zeros(N, dtype=bool); long_mask[order[half:]] = True
    median_len = np.median(game_len)

    print(f"\ngames played: {N}, ticks run: {t + 1}, device: {device}")
    print()
    for sk in range(4):
        print(f"{names[sk]:24s} wins={wins[sk]:4d} ({100*wins[sk]/N:.1f}%)  "
              f"avg_survival_ticks={survive_ticks[:, sk].mean():.1f}  "
              f"avg_placement={placement[:, sk].mean():.2f}")

    print(f"\nmedian game length: {median_len:.0f} ticks (range {game_len.min()}-{game_len.max()})")
    for label, mask in [("SHORT (<= median)", short_mask), ("LONG (> median)", long_mask)]:
        n_sub = mask.sum()
        print(f"\n{label}, n={n_sub}")
        for sk in range(4):
            sub_wins = 0
            for g in np.where(mask)[0]:
                nb = (gs["flags"][g >> 1] >> ((g & 1) * 4)) & 0xF
                if bin(int(nb)).count("1") == 1 and (nb >> sk) & 1:
                    sub_wins += 1
            sub_surv = survive_ticks[mask, sk].mean() if n_sub > 0 else float("nan")
            sub_place = placement[mask, sk].mean() if n_sub > 0 else float("nan")
            print(f"  {names[sk]:24s} wins={sub_wins:4d} ({100*sub_wins/max(n_sub,1):.1f}%) avg_surv={sub_surv:.1f}  avg_place={sub_place:.2f}")
    return {"names": names, "wins": wins, "survive_ticks": survive_ticks, "placement": placement, "game_len": game_len}

def main():
    parser = argparse.ArgumentParser(description="4-way FFA benchmark")
    parser.add_argument("--ckpt", nargs=4, default=CHECKPOINTS,help="exactly 4 entries; use 'shapeshifter' for engine")
    parser.add_argument("--n", type=int, default=100, help="number of parallel games")
    parser.add_argument("--max-ticks", type=int, default=2000, help="tick cap")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    parser.add_argument("--cpu", action="store_true", help="force CPU")
    parser.add_argument("--engine-timeout", type=int, default=ENGINE_TIMEOUT_MS, help="engine timeout (ms)")
    parser.add_argument("--debug", action="store_true", help="print engine move conversions")
    args = parser.parse_args()

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    run_tournament(args.ckpt, args.n, args.max_ticks, device, seed=args.seed, engine_timeout=args.engine_timeout, debug=args.debug)

if __name__ == "__main__":
    main()