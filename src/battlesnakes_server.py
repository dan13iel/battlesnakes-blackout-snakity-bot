import os
import json
import queue
import threading
from datetime import datetime, timezone
import numpy as np
from flask import Flask, request
import runtime_model as rt

author = "Snakity"
name = "Snakity"
color = "#FFFFFF"

CKPT = os.environ.get("CKPT", "ckpt_ep88599.pt")
CONFIG = os.environ.get("CONFIG", "model_config.json")
DUMP_DIR = os.environ.get("DUMP_DIR")
POOL_TARGET = int(os.environ.get("POOL_TARGET", "4"))
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
if DUMP_DIR:
    os.makedirs(DUMP_DIR, exist_ok=True)

app = Flask("Battlesnake")

_pool = queue.LifoQueue()
_games = {}
_glock = threading.Lock()
_loglock = threading.Lock()
_want = threading.Event()

def _refill():
    while True:
        while _pool.qsize() < POOL_TARGET:
            try:
                _pool.put(rt.model(CKPT, CONFIG))
            except Exception as e:
                app.logger.error("preload failed: %s", e)
                break
        _want.clear()
        _want.wait(timeout=30.0)

_pool.put(rt.model(CKPT, CONFIG))
threading.Thread(target=_refill, daemon=True).start()

def _today():return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def _log(endpoint):
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "method": request.method,
        "path": request.path,
        "args": dict(request.args),
        "headers": dict(request.headers),
        "remote_addr": request.remote_addr,
        "body": request.get_json(silent=True),
    }
    with _loglock, open(os.path.join(LOG_DIR, f"{_today()}.jsonl"), "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry

def _read_day(day):
    path = os.path.join(LOG_DIR, f"{day}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def _acquire(gid, me_id, snakes):
    with _glock:
        m = _games.get(gid)
        if m is None:
            try:
                m = _pool.get_nowait()
                m.reset()
            except queue.Empty:
                m = rt.model(CKPT, CONFIG)
            _want.set()

            m.seats = {me_id: 0}
            for sid in sorted(s["id"] for s in snakes if s["id"] != me_id)[:3]:
                m.seats[sid] = len(m.seats)
            m.lock = threading.Lock()
            _games[gid] = m
        return m

def _release(gid):
    with _glock:
        m = _games.pop(gid, None)
    if m is None:
        return
    if DUMP_DIR:
        try:
            m.dump(os.path.join(DUMP_DIR, f"{gid}.npz"))
        except Exception as e:
            app.logger.warning("dump failed for %s: %s", gid, e)
    m.reset()
    _pool.put(m)

def _seat(seats, sid):
    s = seats.get(sid)
    if s is None and len(seats) < 4:
        s = seats[sid] = len(seats)
    return s

def _alive(s): return bool(s.get("body"))

def _build_cells(offsets, board, seats, hx, hy):
    w, h = board["width"], board["height"]
    occ, heads = {}, set()
    for s in board["snakes"]:
        if not _alive(s):
            continue
        k = _seat(seats, s["id"])
        if k is None:
            continue
        for p in s["body"]:
            if p["x"] >= 0:
                occ[(p["x"], h - 1 - p["y"])] = k
        hd = s.get("head")
        if k != 0 and hd is not None and hd["x"] >= 0:
            heads.add((hd["x"], h - 1 - hd["y"]))
    food = {(f["x"], h - 1 - f["y"]) for f in board["food"]}

    cells = np.zeros((len(offsets), 3), dtype=np.int32)
    cells[:, 0] = -1
    for i, (dx, dy) in enumerate(offsets):
        x, y = hx + int(dx), hy + int(dy)
        if not (0 <= x < w and 0 <= y < h):
            continue
        cells[i, 0] = occ.get((x, y), -1)
        cells[i, 1] = (x, y) in food
        cells[i, 2] = (x, y) in heads
    return cells


@app.get("/")
def on_info():
    _log("info")
    return {"author": author, "color": color, "name": name}

@app.post("/start")
def on_start():
    d = _log("start")["body"]
    _acquire(d["game"]["id"], d["you"]["id"], d["board"]["snakes"])
    return "ok"

@app.post("/move")
def on_move():
    d = _log("move")["body"]
    board, you, turn = d["board"], d["you"], d["turn"]
    h = board["height"]
    m = _acquire(d["game"]["id"], you["id"], board["snakes"])

    with m.lock:
        hx, hy = you["head"]["x"], h - 1 - you["head"]["y"]
        body = [(p["x"], h - 1 - p["y"]) for p in you["body"]]
        cells = _build_cells(m.view_offsets, board, m.seats, hx, hy)

        apple = None
        if turn > 0:
            for f in board["food"]:
                if f.get("spawn_turn") == turn:
                    apple = (f["x"], h - 1 - f["y"])
                    break
        mv = m(cells,apple,(hx, hy),body,health=you["health"],)
    return {"move": mv}

@app.post("/end")
def on_end():
    d = _log("end")["body"]
    _release(d["game"]["id"])
    return "ok"

@app.get("/dump")
def on_dump():
    day = _today()
    return {"day": day, "entries": _read_day(day)}

@app.get("/dumpall")
def on_dumpall():
    days = sorted(f[:-6] for f in os.listdir(LOG_DIR) if f.endswith(".jsonl"))
    return {day: _read_day(day) for day in days}


app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), threaded=True)