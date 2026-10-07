import json
import numpy as np
import torch
import obsmem as om
import snakenet as brl # net + constants only; basic_rl is a script and would start TRAINING on import

MOVE_NAMES = np.array(["up", "down", "left", "right"])  # engine action 0..3
EMPTY = 255

# (61,) view-diamond offsets; cells[i] must describe head + (VIEW_DX[i], VIEW_DY[i])
VIEW_DX, VIEW_DY = om.VIEW_DX, om.VIEW_DY
_NET_KEYS = ("in_ch", "crop", "n_scal", "n_act", "ch", "squeeze", "hid", "mem", "priv_ch")

def _load_cfg(path):
    cfg = json.load(open(path))
    cfg = cfg.get("arch", cfg.get("net", cfg.get("model", cfg)))
    kw = {k: cfg[k] for k in _NET_KEYS if k in cfg}
    kw.setdefault("priv_ch", brl.PRIV_CH)
    if "ch" in kw:
        kw["ch"] = tuple(kw["ch"])
    return kw

def _load_state(path, device):
    obj = torch.load(path, map_location=device, weights_only=False)
    sd = obj.get("model", obj) if isinstance(obj, dict) and "model" in obj else obj
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


class model:
    def __init__(self, ckpt_path, config_path, device=None, seat=0):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cfg = _load_cfg(config_path)
        self.net = brl.SnakeNet(**self.cfg).to(self.device).eval()
        self.net.load_state_dict(_load_state(ckpt_path, self.device))
        self.seat = seat
        self.view_offsets = np.stack([VIEW_DX, VIEW_DY], 1)

        self.board = np.full((1, 15, 15), EMPTY, np.uint8)
        self.head_g = np.zeros((1, 15, 15), np.uint8)
        self.food_g = np.zeros((1, 15, 15), np.uint8)
        self.headpos = np.full((1, 4, 2), 255, np.uint8)
        self.tailord = np.zeros((1, 4, 15, 15), np.uint8)
        self.mem_occ = np.zeros((1, 4, 15, 15), np.uint8)
        self.mem_food = np.zeros((1, 4, 15, 15), np.uint8)
        self.mem_head = np.zeros((1, 4, 15, 15), np.uint8)
        self.mem_age = np.full((1, 4, 15, 15), 255, np.uint8)
        self.est_len = np.full((1, 4, 4), 3, np.uint8)
        self.length = np.zeros((1, 4), np.uint8)
        self.health = np.zeros((1, 4), np.uint8)
        self.new_apple = np.full((1, 2), 255, np.uint8)
        self.rot = np.zeros((1, 4), np.int8)
        self.obs_u8 = np.zeros((1, 4, om.C, om.CROP, om.CROP), np.uint8)
        self.scal_u8 = np.zeros((1, 4, 5), np.uint8)  # health, len, timestep, hx, hy
        self.reset()

    def reset(self):
        self.mem_occ.fill(0); self.mem_food.fill(0); self.mem_head.fill(0)
        self.mem_age.fill(255); self.est_len.fill(3)
        n = self.net.mem
        self.hc = (torch.zeros(1, n, device=self.device), torch.zeros(1, n, device=self.device))
        self.t = 0
        self.prev_head = None
        self.log = []

    def _heading(self, hx, hy):
        if self.prev_head is not None:
            dx, dy = hx - self.prev_head[0], hy - self.prev_head[1]
            if dy < 0: return 0
            if dx > 0: return 1
            if dy > 0: return 2
            if dx < 0: return 3
        return {(13, 7): 1, (7, 13): 2, (1, 7): 3}.get((hx, hy), 0)

    def _ingest(self, cells, apple, head, body, health):
        k = self.seat
        hx, hy = int(head[0]), int(head[1])
        body = np.asarray(body, np.int32).reshape(-1, 2)
        L = len(body)

        self.board.fill(EMPTY); self.head_g.fill(0); self.food_g.fill(0)
        self.tailord[0, k].fill(0)

        # own body, walked tail -> head so headward segments win on self-overlap.
        # tailord = turns until vacated, so the tail cell is 1.
        for j in range(L - 1, -1, -1):
            x, y = int(body[j, 0]), int(body[j, 1])
            self.board[0, y, x] = k
            self.tailord[0, k, y, x] = min(L - j, 255)

        # visible diamond: occupant_id (-1/255 empty, 0..3 snake), is_food, is_head
        cells = np.asarray(cells).reshape(-1, 3)
        for i in range(len(cells)):
            x, y = hx + int(VIEW_DX[i]), hy + int(VIEW_DY[i])
            if not (0 <= x < 15 and 0 <= y < 15):
                continue
            occ, food, is_head = int(cells[i, 0]), int(cells[i, 1]), int(cells[i, 2])
            if 0 <= occ < 4:
                self.board[0, y, x] = occ
                if is_head:
                    self.head_g[0, y, x] = occ + 1
            if food:
                self.food_g[0, y, x] = 1

        self.headpos.fill(255)
        self.headpos[0, k] = (hx, hy)
        self.length[0, k] = min(L, 255)
        self.health[0, k] = min(int(health), 255)
        self.new_apple[0] = (255, 255) if apple is None else (int(apple[0]), int(apple[1]))
        self.rot[0, k] = self._heading(hx, hy)
        self.prev_head = (hx, hy)

    def _obs(self):
        k = self.seat
        om.update_memory(self.board, self.head_g, self.food_g, self.headpos, self.new_apple,
                         self.length, self.mem_occ, self.mem_food, self.mem_head, self.mem_age,
                         self.est_len, om.VIEW_DX, om.VIEW_DY, k)
        om.write_channels(self.board, self.food_g, self.headpos, self.tailord, self.mem_occ,
                          self.mem_food, self.mem_head, self.mem_age, self.est_len, self.length,
                          self.rot, om.ROT_OFF, self.obs_u8, self.scal_u8,
                          min(self.t, 255), k)
        x = om.dequantise(torch.from_numpy(self.obs_u8[:, k]).to(self.device))
        s = torch.from_numpy(self.scal_u8[:, k]).to(self.device).float()
        s = torch.stack((s[:, 0] / 100.0, s[:, 1] / om.LEN_MAX, s[:, 2] / 255.0, s[:, 3] / 14.0, s[:, 4] / 14.0), 1)
        return x, s

    @torch.no_grad()
    def step(self, cells, apple, head, body, health=100, sample=False):
        self._ingest(cells, apple, head, body, health)
        logits, _, self.hc = self.net(*self._obs(), self.hc, priv=None)
        logits = logits.float()
        a3 = int(torch.multinomial(logits.softmax(-1)[0], 1) if sample else logits.argmax(-1))
        raw = int(brl.INV_ROT_ACTION[int(self.rot[0, self.seat]), om.ACT3_TO_CANON[a3]])

        self.log.append((self.t, self.board[0].copy(), self.food_g[0].copy(), np.array(head, np.uint8), self.new_apple[0].copy(),
                         int(self.length[0, self.seat]), int(self.health[0, self.seat]), int(self.rot[0, self.seat]), a3, raw, logits[0].cpu().numpy()))
        self.t += 1
        return str(MOVE_NAMES[raw])

    __call__ = step

    def dump(self, path):
        """Compressed .npz of every tick: board, food, head, move. No replay needed."""
        f = list(zip(*self.log))
        np.savez_compressed(
            path, move_names=MOVE_NAMES, timestep=np.array(f[0], np.int32),
            board=np.stack(f[1]), food=np.stack(f[2]), head=np.stack(f[3]),
            apple=np.stack(f[4]), length=np.array(f[5], np.uint8),
            health=np.array(f[6], np.uint8),
            rot=np.array(f[7], np.int8), action3=np.array(f[8], np.uint8),
            move=np.array(f[9], np.uint8), logits=np.stack(f[10]).astype(np.float32))
        return path