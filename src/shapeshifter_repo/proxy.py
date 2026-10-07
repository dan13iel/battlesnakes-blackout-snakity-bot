"""Batch proxy: engine game_state dict in, list of optimal seat-0 moves out.

One HTTP request carries a whole batch. The proxy unpacks each game from the ring
buffer, builds a Battlesnake payload, fans the batch out to shapeshifter across a
thread pool, and returns moves in the ENGINE's encoding.

    server:  python shapeshifter_proxy.py            # listens on :8090
    client:  from shapeshifter_proxy import optimal_moves
             moves = optimal_moves(game_state)       # -> [0, 3, 1, None, ...]

COORDINATE FRAMES - the thing that silently breaks if you get it wrong.
  The engine has y growing DOWNWARD; the Battlesnake API has y growing UP. Bodies
  and food are flipped on the way out (api_y = 14 - engine_y).
  Move NAMES survive that flip unchanged, which looks like a bug and isn't:
  API "up" means api_y+1, which is engine_y-1, which is engine move 0. So the
  name->code table below is the identity mapping, and that is correct *because*
  the coordinates were flipped. Feed unflipped coords and up/down invert.

SEARCH BUDGET
  timeout_ms is per-request. To search by DEPTH instead, start the engine with
  FIXED_DEPTH=n in its environment: it is read once at process start, applies to
  every request, and makes results deterministic. timeout_ms is then ignored.

RETURN VALUE
  moves[g] is 0/1/2/3 (up/down/left/right, engine convention) or None when seat 0
  is dead or absent in that game. Feed straight into your moves word:
      moves_u8[g] |= code << (2 * seat)
"""
import base64
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

ENGINE_URL = os.environ.get("SHAPESHIFTER_URL", "http://127.0.0.1:8081/move")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8090"))
WORKERS = int(os.environ.get("PROXY_WORKERS", "15")) # 2 less then core count
SIZE = 15
RING = 225
DEFAULT_TIMEOUT_MS = 200

NAME_TO_CODE = {"up": 0, "down": 1, "left": 2, "right": 3}
ARRAYS = ("data", "flags", "food", "length", "health")


def encode_state(gs):
    """numpy game_state dict -> JSON-safe dict (base64, dtype and shape preserved)."""
    out = {"timestep": int(gs["timestep"])}
    for k in ARRAYS:
        a = np.ascontiguousarray(gs[k])
        out[k] = {"b64": base64.b64encode(a.tobytes()).decode(),
                  "dtype": a.dtype.str, "shape": list(a.shape)}
    return out


def decode_state(d):
    gs = {"timestep": int(d["timestep"])}
    for k in ARRAYS:
        v = d[k]
        if isinstance(v, dict) and "b64" in v:
            gs[k] = np.frombuffer(base64.b64decode(v["b64"]),
                                  dtype=np.dtype(v["dtype"])).reshape(v["shape"])
        else:
            gs[k] = np.asarray(v)
    return gs


def alive_nibble(flags, g):
    return int((flags[g >> 1] >> np.uint8((g & 1) * 4)) & np.uint8(0xF))


def read_body(data, length, timestep, g, sk):
    """Head-first list of (x, y) in ENGINE coords. Empty if the snake is cleared.

    timestep is forced to a Python int: game_state['timestep'] is np.uint32, and
    (uint32 - j) underflows to ~4e9 for j > timestep, which silently reads the
    wrong ring slots on the first few ticks and truncates the body.
    """
    t = int(timestep)
    L = int(length[g, sk])
    cells = []
    for j in range(L):
        cell = int(data[g, sk, (t - j) % RING])
        x = cell & 0xF
        if x == 0xF:
            break
        cells.append((x, (cell >> 4) & 0xF))
    return cells


def read_food(food, g):
    return [(p % SIZE, p // SIZE) for p in range(RING)
            if (food[g, p >> 3] >> np.uint8(p & 7)) & 1]


def build_payload(gs, g, seat, timeout_ms):
    """(payload, reason). payload is None unless reason == 'ok'.

    reason is one of:
      ok         - a live position, move returned
      seat_dead  - that seat's alive bit is clear
      game_over  - <=1 snake left. step() returns early for finished games and
                   stops writing head cells while step_wrapper keeps incrementing
                   timestep, so the ring data is frozen and the head is no longer
                   at timestep % 225. There is no position left to solve.
    """
    data, length, health = gs["data"], gs["length"], gs["health"]
    t = int(gs["timestep"])
    mask = alive_nibble(gs["flags"], g)

    if not (mask >> seat) & 1:
        return None, "seat_dead"
    if bin(mask).count("1") <= 1:
        return None, "game_over"

    snakes = []
    me = None
    for sk in range(4):
        if not (mask >> sk) & 1:
            continue
        cells = read_body(data, length, t, g, sk)
        if not cells:
            continue
        flip = [{"x": x, "y": SIZE - 1 - y} for x, y in cells]
        obj = {"id": str(sk), "name": f"seat{sk}", "health": int(health[g, sk]),
               "length": len(flip), "head": flip[0], "body": flip,
               "shout": None, "squad": None, "next_move": None}
        snakes.append(obj)
        if sk == seat:
            me = obj
    if me is None:
        return None, "seat_dead"
    if len(snakes) > 4:
        snakes = [me] + [s for s in snakes if s is not me][:3]

    return {
        "game": {"id": f"proxy-{g}",
                 "ruleset": {"name": "standard", "version": "v1", "settings": {}},
                 "map": "standard", "timeout": int(timeout_ms), "source": "custom"},
        "turn": int(t),
        "board": {"height": SIZE, "width": SIZE,
                  "food": [{"x": x, "y": SIZE - 1 - y} for x, y in read_food(gs["food"], g)],
                  "hazards": [], "snakes": snakes},
        "you": me,
    }, "ok"


def ask(payload):
    req = urllib.request.Request(ENGINE_URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return NAME_TO_CODE[json.loads(r.read())["move"]]
    except Exception:
        # shapeshifter panics (unsupported snake count / board size) drop the
        # connection rather than returning an HTTP error, so catch broadly.
        return None


_pool = ThreadPoolExecutor(max_workers=WORKERS)


def solve_batch(gs, seat=0, timeout_ms=DEFAULT_TIMEOUT_MS):
    """-> (moves, reasons), both length n. moves[g] is None unless reasons[g]=='ok'."""
    n = gs["data"].shape[0]
    built = [build_payload(gs, g, seat, timeout_ms) for g in range(n)]
    reasons = [r for _, r in built]
    futs = {g: _pool.submit(ask, p) for g, (p, _) in enumerate(built) if p is not None}
    moves = []
    for g in range(n):
        if g not in futs:
            moves.append(None)
            continue
        m = futs[g].result()
        if m is None:
            reasons[g] = "engine_error"
        moves.append(m)
    return moves, reasons


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            try:
                with urllib.request.urlopen(ENGINE_URL.rsplit("/", 1)[0] + "/",
                                            timeout=5) as r:
                    up = r.status == 200
            except Exception:
                up = False
            self._send(200, {"proxy": "ok", "engine": ENGINE_URL, "engine_up": up,
                             "workers": WORKERS})
        else:
            self._send(404, {"error": "POST /moves, or GET /health"})

    def do_POST(self):
        if self.path != "/moves":
            return self._send(404, {"error": "POST /moves"})
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            gs = decode_state(body["game_state"])
            seat = int(body.get("seat", 0))
            tmo = int(body.get("timeout_ms", DEFAULT_TIMEOUT_MS))
        except Exception as e:
            return self._send(400, {"error": f"bad request: {type(e).__name__}: {e}"})

        t0 = time.time()
        try:
            moves, reasons = solve_batch(gs, seat, tmo)
        except Exception as e:
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})
        ms = (time.time() - t0) * 1000
        names = [None if m is None else ["up", "down", "left", "right"][m] for m in moves]
        tally = {}
        for r in reasons:
            tally[r] = tally.get(r, 0) + 1
        self._send(200, {"moves": moves, "names": names, "reasons": reasons,
                         "n": len(moves), "seat": seat,
                         "answered": sum(m is not None for m in moves),
                         "tally": tally, "ms": round(ms, 1)})

    def log_message(self, *a):
        pass


def optimal_moves(game_state, seat=0, timeout_ms=DEFAULT_TIMEOUT_MS,
                  url=f"http://127.0.0.1:{PROXY_PORT}/moves", full=False):
    """One request for a whole batch. Returns [code|None] per game.

    full=True returns the whole response dict instead, including `reasons`
    (ok / seat_dead / game_over / engine_error) and timing.
    """
    body = json.dumps({"game_state": encode_state(game_state),
                       "seat": seat, "timeout_ms": timeout_ms}).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        res = json.loads(r.read())
    return res if full else res["moves"]


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PROXY_PORT), Handler)
    print(f"proxy   http://127.0.0.1:{PROXY_PORT}/moves   (POST)")
    print(f"engine  {ENGINE_URL}   workers={WORKERS}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()