import numpy as np
from numba import njit

# occupancy categories (spatial channels 0-4)
UNKNOWN, EMPTY, OWN_BODY, OPP_BODY, OFFBOARD = 0, 1, 2, 3, 4
M_UNKNOWN, M_EMPTY, M_OPP = 0, 1, 2 # mem_occ channel stores only UNKNOWN, EMPTY, OPP_BODY and these are the ids
N_OCC = 5

CROP = 13 # head-centred crop side (radius 7)
C = 9 # spatial channels
AGE_CAP = 32 # age normaliser
LEN_MAX = 40 # length / turns-until-vacated normaliser
GHOST_TTL = 3 # OPP_BODY older than this is cleared to UNKNOWN
VIEW_R = 5 # Manhattan view radius (matches the sim's diamond)
ADV_ZERO = 128

# 3-action space in the canonical frame: left / straight / right.
# canonical straight = up = engine action 0; left = 2; right = 3. 
ACT3_TO_CANON = np.array([2, 0, 3], dtype=np.uint8)


def _build_rot_off(): # builds luts for rotation
    off = np.zeros((4, CROP, CROP, 2), dtype=np.int32)
    h = CROP // 2
    for r in range(4):
        for cy in range(CROP):
            for cx in range(CROP):
                ox, oy = cx - h, cy - h
                if r == 0:
                    wx, wy = ox, oy
                elif r == 1:
                    wx, wy = -oy, ox
                elif r == 2:
                    wx, wy = -ox, -oy
                else:
                    wx, wy = oy, -ox
                off[r, cy, cx, 0] = wx
                off[r, cy, cx, 1] = wy
    return off

ROT_OFF = _build_rot_off()
_dy, _dx = np.mgrid[-VIEW_R:VIEW_R + 1, -VIEW_R:VIEW_R + 1]
_d = (np.abs(_dx) + np.abs(_dy)) <= VIEW_R
VIEW_DX = _dx[_d].astype(np.int32)
VIEW_DY = _dy[_d].astype(np.int32)


@njit(cache=True)
def build_true_boards(data, food, length, timestep, board, head, foodg, headpos, tailord, seat=-1):
    B = data.shape[0]
    head_ring = timestep % 225
    k0 = 0 if seat < 0 else seat
    k1 = 4 if seat < 0 else seat + 1
    for g in range(B):
        for y in range(15):
            for x in range(15):
                board[g, y, x] = 255
                head[g, y, x] = 0
                foodg[g, y, x] = 0
        for k in range(k0, k1):
            for y in range(15):
                for x in range(15):
                    tailord[g, k, y, x] = 0
        for p in range(225):
            if (food[g, p >> 3] >> (p & 7)) & 1:
                foodg[g, p // 15, p % 15] = 1
        for k in range(4):
            L = int(length[g, k])
            headpos[g, k, 0] = 255
            headpos[g, k, 1] = 255
            if L <= 0:
                continue
            hc = data[g, k, head_ring]
            hx = hc & 0xF
            if hx == 0xF:
                continue # dead: cells are sentinels
            hy = (hc >> 4) & 0xF
            headpos[g, k, 0] = hx
            headpos[g, k, 1] = hy
            head[g, hy, hx] = k + 1
            ring = (head_ring - L + 1) % 225
            want_tail = k0 <= k < k1
            for i in range(L):
                cell = data[g, k, ring]
                x = cell & 0xF
                if x != 0xF:
                    y = (cell >> 4) & 0xF
                    board[g, y, x] = k
                    if want_tail:
                        tailord[g, k, y, x] = min(i + 1, 255)
                ring = ring + 1 if ring != 224 else 0


@njit(cache=True)
def update_memory(board, head, foodg, headpos, new_apple, length, mem_occ, mem_food, mem_head, mem_age, est_len, vdx, vdy, seat=-1):
    """Per-seat memory update using only what that seat should see. That way the model doesnt need to waste parameters on trying to remember where everything is"""
    B = board.shape[0]
    n = vdx.shape[0]
    k0 = 0 if seat < 0 else seat
    k1 = 4 if seat < 0 else seat + 1
    for g in range(B):
        ax = int(new_apple[g, 0])
        ay = int(new_apple[g, 1])
        for k in range(k0, k1):
            # age everything (saturating)
            for y in range(15):
                for x in range(15):
                    if mem_age[g, k, y, x] < 255:
                        mem_age[g, k, y, x] += 1
            hx = int(headpos[g, k, 0])
            if hx == 255:
                continue # dead seat: freeze its map
            hy = int(headpos[g, k, 1])

            # overwrite everything inside the view diamond, including clearing
            seen = np.zeros(4, dtype=np.int32)
            for i in range(n):
                sx = hx + vdx[i]
                sy = hy + vdy[i]
                if sx < 0 or sx > 14 or sy < 0 or sy > 14:
                    continue
                mem_age[g, k, sy, sx] = 0
                occ = board[g, sy, sx]
                if occ == 255 or occ == k:
                    mem_occ[g, k, sy, sx] = M_EMPTY # own body is not memorised
                else:
                    mem_occ[g, k, sy, sx] = M_OPP
                    seen[occ] += 1
                mem_food[g, k, sy, sx] = foodg[g, sy, sx]
                hv = head[g, sy, sx]
                if hv != 0 and hv - 1 != k:
                    mem_head[g, k, sy, sx] = hv
                else:
                    mem_head[g, k, sy, sx] = 0

            # global one-tick food-spawn flash
            if ax != 255:
                mem_food[g, k, ay, ax] = 1
                mem_age[g, k, ay, ax] = 0

            # bodies move: hard-clear stale opponent segments so the map does
            # not silt up with ghosts that wall off the board. Food is static
            # and is deliberately NOT cleared by age.
            for y in range(15):
                for x in range(15):
                    if mem_age[g, k, y, x] > GHOST_TTL:
                        if mem_occ[g, k, y, x] == M_OPP:
                            mem_occ[g, k, y, x] = M_UNKNOWN
                        mem_head[g, k, y, x] = 0

            # opponent length estimate: lengths never shrink
            for j in range(4):
                if j != k and seen[j] > int(est_len[g, k, j]):
                    est_len[g, k, j] = min(seen[j], 255)


@njit()
def write_channels(board, foodg, headpos, tailord, mem_occ, mem_food, mem_head, mem_age, est_len, length, rot, rot_off, out, scal, timestep, seat=-1):
    """Write the 9 uint8 channel planes. out:(B,4,C,CROP,CROP). Only `seat` is
    populated unless seat < 0; other seats keep whatever they last held.

    scal layout (5): length, timestep, head_x, head_y. n_alive was removed:
    under fog of war a seat cannot know how many opponents are still alive, and the
    deploy API does not provide it."""
    B = board.shape[0]
    h = CROP // 2
    k0 = 0 if seat < 0 else seat
    k1 = 4 if seat < 0 else seat + 1
    for g in range(B):
        for k in range(k0, k1):
            hx = int(headpos[g, k, 0])
            my_len = int(length[g, k])
            scal[g, k, 0] = 100 # Avoid retraining without health feature.
            scal[g, k, 1] = min(my_len, 255)
            scal[g, k, 2] = min(np.uint32(timestep), np.uint32(255))
            if hx == 255:
                for c in range(C):
                    for y in range(CROP):
                        for x in range(CROP):
                            out[g, k, c, y, x] = ADV_ZERO if c == 7 else 0
                scal[g, k, 3] = 255
                scal[g, k, 4] = 255
                continue
            hy = int(headpos[g, k, 1])
            scal[g, k, 3] = hx
            scal[g, k, 4] = hy
            r = int(rot[g, k])
            for cy in range(CROP):
                for cx in range(CROP):
                    wx = hx + rot_off[r, cy, cx, 0]
                    wy = hy + rot_off[r, cy, cx, 1]
                    for c in range(C):
                        out[g, k, c, cy, cx] = ADV_ZERO if c == 7 else 0
                    if wx < 0 or wx > 14 or wy < 0 or wy > 14:
                        out[g, k, OFFBOARD, cy, cx] = 1
                        continue
                    if board[g, wy, wx] == k:
                        out[g, k, OWN_BODY, cy, cx] = 1
                        out[g, k, 8, cy, cx] = tailord[g, k, wy, wx]
                    else:
                        m = mem_occ[g, k, wy, wx]
                        if m == M_UNKNOWN:
                            out[g, k, UNKNOWN, cy, cx] = 1
                        elif m == M_EMPTY:
                            out[g, k, EMPTY, cy, cx] = 1
                        else:
                            out[g, k, OPP_BODY, cy, cx] = 1
                    out[g, k, 5, cy, cx] = mem_age[g, k, wy, wx]
                    out[g, k, 6, cy, cx] = mem_food[g, k, wy, wx]

            # if there is a head advantage then values are placed on squares adjacent to where the snake head can go 
            for wy in range(15):
                for wx in range(15):
                    hv = mem_head[g, k, wy, wx]
                    if hv == 0:
                        continue
                    j = hv - 1
                    # est_len only counts simultaneously-visible segments so it
                    # underestimates: demand a margin before calling the contest won, and treat est == my_len as a loss (the true length is >= est).
                    # doesnt really work/help in end or middle games
                    ol = int(est_len[g, k, j])
                    if my_len > ol + 1: 
                        v = 228 # +1.0  we win
                    elif my_len == ol + 1:
                        v = 78 # -0.5  margin of 1 over an underestimate: coin flip
                    else:
                        v = 28 # -1.0  est >= my_len: assume we lose
                    for d in range(4):
                        nx = wx + (0 if d < 2 else (1 if d == 2 else -1))
                        ny = wy + (1 if d == 0 else (-1 if d == 1 else 0))
                        if nx < 0 or nx > 14 or ny < 0 or ny > 14:
                            continue
                        ddx = nx - hx
                        ddy = ny - hy
                        if r == 0:
                            cxo, cyo = ddx, ddy
                        elif r == 1:
                            cxo, cyo = ddy, -ddx
                        elif r == 2:
                            cxo, cyo = -ddx, -ddy
                        else:
                            cxo, cyo = -ddy, ddx
                        cx = cxo + h
                        cy = cyo + h
                        if 0 <= cx < CROP and 0 <= cy < CROP:
                            if abs(v - ADV_ZERO) > abs(int(out[g, k, 7, cy, cx]) - ADV_ZERO):
                                out[g, k, 7, cy, cx] = v


@njit(cache=True)
def heading_rot_all(data, timestep, out_rot):
    """
    per-seat rotation putting that seat's heading at canonical 'up'
    it makes training faster since the model doesnt need to learn how to deal with rotation
    """
    cur = timestep % 225
    prev = (timestep - 1) % 225 if timestep > 0 else 0
    for g in range(data.shape[0]):
        for k in range(4):
            hc = data[g, k, cur]
            hx = np.int32(hc & np.uint32(0xF))
            if hx == 15:
                out_rot[g, k] = 0
                continue
            hy = np.int32((hc >> np.uint32(4)) & np.uint32(0xF))
            dx = np.int32(0)
            dy = np.int32(0)
            if timestep > 0:
                pc = data[g, k, prev]
                px = np.int32(pc & np.uint32(0xF))
                if px != 15:
                    py = np.int32((pc >> np.uint32(4)) & np.uint32(0xF))
                    dx = hx - px
                    dy = hy - py
            if dx == 0 and dy == 0:
                if hx == 13 and hy == 7:
                    out_rot[g, k] = 1
                elif hx == 7 and hy == 13:
                    out_rot[g, k] = 2
                elif hx == 1 and hy == 7:
                    out_rot[g, k] = 3
                else:
                    out_rot[g, k] = 0
            elif dy < 0:
                out_rot[g, k] = 0
            elif dx > 0:
                out_rot[g, k] = 1
            elif dy > 0:
                out_rot[g, k] = 2
            else:
                out_rot[g, k] = 3


@njit(cache=True)
def safe_moves3(board, headpos, rot, rot_off, actions):
    """
    this opponent doesnt really die quickly by itself unless something (e.g model, environment, random mistake) kills it
    """
    B = board.shape[0]
    h = CROP // 2
    for g in range(B):
        for k in range(4):
            hx = int(headpos[g, k, 0])
            if hx == 255:
                actions[g, k] = 1
                continue
            hy = int(headpos[g, k, 1])
            r = int(rot[g, k])
            cnt = 0
            valid = np.zeros(3, dtype=np.int32)
            for a in range(3):
                cxo = -1 if a == 0 else (0 if a == 1 else 1)
                cyo = 0 if a != 1 else -1
                wx = hx + rot_off[r, cyo + h, cxo + h, 0]
                wy = hy + rot_off[r, cyo + h, cxo + h, 1]
                if wx < 0 or wx > 14 or wy < 0 or wy > 14:
                    continue
                if board[g, wy, wx] != 255:
                    continue
                valid[cnt] = a
                cnt += 1
            actions[g, k] = valid[np.random.randint(0, cnt)] if cnt > 0 else 1


def dequantise(buf_u8):
    """uint8 planes -> float32 network input, required since vram is expensive."""
    x = buf_u8.float()
    x[:, 5] = (x[:, 5] / AGE_CAP).clamp(max=1.0)
    x[:, 7] = (x[:, 7] - ADV_ZERO) / 100.0
    x[:, 8] = (x[:, 8] / LEN_MAX).clamp(max=1.0)
    return x