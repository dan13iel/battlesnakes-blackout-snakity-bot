# mostly ai generated.

import os, json, glob, socket, time, threading, uuid, random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import torch
import obsmem as om
import game
from game import generate_init_data
from snakenet import SnakeNet, PRIV_CH, INV_ROT_ACTION

CKPT_DIR = r"./checkpoints_sampled/"
PORT = int(os.environ.get("PORT", 8080))
B = 2  # Engine needs an even batch; each session plays game 0;
GAME = 0
SESSION_TTL = 1800 # Seconds of inactivity before a session is dropped 

CHAIN = int(os.environ.get("CHAIN", 6))   # how many times one direction is pre-rolled

SNAKES = [
    {"label": "You",    "body": "#c06a52", "head": "#8a3823", "dead": "#5e3a2f"},
    {"label": "Snakity 1", "body": "#5b7f99", "head": "#2f5169", "dead": "#2c3f4c"},
    {"label": "Snakity 2", "body": "#7a9a63", "head": "#456334", "dead": "#3b4a31"},
    {"label": "Snakity 3", "body": "#c79a45", "head": "#8a6218", "dead": "#5c4823"},
]
FOOD = "#a8641f"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
game.batch_size = B
torch.set_num_threads(2)

def infer_cfg(sd):
    return dict(ch=(sd["conv.0.weight"].shape[0], sd["conv.2.weight"].shape[0]), squeeze=sd["conv.4.weight"].shape[0], hid=sd["fc.0.weight"].shape[0], mem=sd["mem_cell.weight_hh"].shape[1])

def load_net(path):
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("model", sd)
    if "conv.0.weight" not in sd:
        raise ValueError("not a SnakeNet state dict") 
    n = SnakeNet(priv_ch=PRIV_CH if any(k.startswith("priv_enc.") for k in sd) else None, **infer_cfg(sd)).to(device).eval()
    n.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    for p in n.parameters():
        p.requires_grad_(False)
    return n

print(f"scanning {CKPT_DIR} ...")
NETS, NAMES = {}, []
for path in sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt"))):
    nm = os.path.splitext(os.path.basename(path))[0]
    try:
        NETS[nm] = load_net(path)
        NAMES.append(nm)
    except Exception as e:
        print(f"  [skip] {nm}: {e}")
if not NETS:
    print("No usable checkpoints found :*(")
    exit(1)
print(f"  loaded {len(NAMES)} checkpoints")

# nets are read-only after load and torch is thread-safe for inference, so all sessions share one copy; only the recurrent state is per-session.

_INIT = generate_init_data(B)

RECENT_POOL = 8 # opponents are drawn from the newest N checkpoints

def pick_seats():
    pool = NAMES[-RECENT_POOL:]
    return [None] + [random.choice(pool) for _ in range(3)]

def new_session():
    s = {
        "board": np.empty((B, 15, 15), dtype=np.uint8),
        "head": np.empty((B, 15, 15), dtype=np.uint8),
        "foodg": np.empty((B, 15, 15), dtype=np.uint8),
        "headpos": np.empty((B, 4, 2), dtype=np.uint8),
        "tailord": np.zeros((B, 4, 15, 15), dtype=np.uint8),
        "mem_occ": np.zeros((B, 4, 15, 15), dtype=np.uint8),
        "mem_food": np.zeros((B, 4, 15, 15), dtype=np.uint8),
        "mem_head": np.zeros((B, 4, 15, 15), dtype=np.uint8),
        "mem_age": np.full((B, 4, 15, 15), 255, dtype=np.uint8),
        "est_len": np.full((B, 4, 4), 3, dtype=np.uint8),
        "rot": np.zeros((B, 4), dtype=np.int8),
        "obs": np.zeros((B, 4, om.C, om.CROP, om.CROP), dtype=np.uint8),
        "scal": np.zeros((B, 4, 5), dtype=np.uint8),
        "act3": np.zeros((B, 4), dtype=np.uint8),
        "gs": {"data": _INIT.copy(),
               "flags": np.full(B // 2, np.uint8(0xFF), dtype=np.uint8),
               "food": np.zeros((B, 29), dtype=np.uint8),
               "length": np.full((B, 4), 3, dtype=np.uint8),
               "health": np.full((B, 4), 100, dtype=np.uint8),
               "timestep": np.uint32(0),
               "new_apple": np.full((B, 2), np.uint8(255), dtype=np.uint8)},
        "seat_name": pick_seats(),
        "hidden": [None] * 4,
        "spawned": [],
        "gen": 0,            # bumped on reset; stale client replies are dropped
        "hist": [],          # absolute move sequence; node keys are prefixed with it
        "nodes": {},         # relative path tuple -> node, pure repeat chains only
        "dirs": [],          # legal directions out of the current root
        "lock": threading.Lock(),
        "touched": time.time(),
    }
    reset(s)
    table_reset(s)
    return s

SESSIONS = {}
SESSIONS_LOCK = threading.Lock()

def get_session(sid):
    with SESSIONS_LOCK:
        now = time.time()
        for k in [k for k, v in SESSIONS.items() if now - v["touched"] > SESSION_TTL]:
            del SESSIONS[k]
        s = SESSIONS.get(sid)
        if s is None:
            sid = uuid.uuid4().hex
            s = SESSIONS[sid] = new_session()
        s["touched"] = now
        return sid, s

def refresh_obs(s, t):
    gs = s["gs"]
    om.build_true_boards(gs['data'], gs['food'], gs['length'], t, s["board"], s["head"],
                         s["foodg"], s["headpos"], s["tailord"], -1)
    om.update_memory(s["board"], s["head"], s["foodg"], s["headpos"], gs['new_apple'],
                     gs['length'], s["mem_occ"], s["mem_food"], s["mem_head"], s["mem_age"],
                     s["est_len"], om.VIEW_DX, om.VIEW_DY, -1)
    om.heading_rot_all(gs['data'], t, s["rot"])
    om.write_channels(s["board"], s["foodg"], s["headpos"], s["tailord"], s["mem_occ"],
                      s["mem_food"], s["mem_head"], s["mem_age"], s["est_len"], gs['length'],
                      s["rot"], om.ROT_OFF, s["obs"], s["scal"], t, -1)

def seat_obs(s, k):
    x = om.dequantise(torch.from_numpy(s["obs"][GAME:GAME + 1, k]).to(device))
    sc = torch.from_numpy(s["scal"][GAME:GAME + 1, k]).to(device).float()
    sc = torch.stack((sc[:, 0] / 100.0, sc[:, 1] / om.LEN_MAX, sc[:, 2] / 255.0,
                      sc[:, 3] / 14.0, sc[:, 4] / 14.0), dim=1)
    return x, sc

def clear_hidden(s, k):
    n = NETS[s["seat_name"][k]]
    s["hidden"][k] = (torch.zeros(1, n.mem, device=device),
                      torch.zeros(1, n.mem, device=device))

def set_food(gs, cells):
    gs['food'].fill(0)
    for g in range(B):
        for (x, y) in cells:
            p = y * 15 + x
            gs['food'][g, p >> 3] |= np.uint8(1 << (p & 7))

def opening_apples(gs):
    """Centre apple, plus one within a diagonal step of each snake's spawn."""
    heads = []
    for k in range(4):
        c = int(gs['data'][GAME, k, 0])
        heads.append((c & 0xF, (c >> 4) & 0xF))
    cells, taken = [(7, 7)], {(7, 7)} | set(heads)
    for (hx, hy) in heads:
        # prefer the diagonal pointing inward so the apple is never off-board
        for dx, dy in sorted([(1, 1), (1, -1), (-1, 1), (-1, -1)],
                             key=lambda d: abs(hx + d[0] - 7) + abs(hy + d[1] - 7)):
            x, y = hx + dx, hy + dy
            if 0 <= x <= 14 and 0 <= y <= 14 and (x, y) not in taken:
                cells.append((x, y)); taken.add((x, y))
                break
    return cells

def reset(s, reseat=False):
    gs = s["gs"]
    if reseat:
        s["seat_name"] = pick_seats()   # fresh random draw from the recent pool
    np.copyto(gs['data'], _INIT)
    gs['flags'].fill(0xFF); gs['length'].fill(3); gs['health'].fill(100)
    gs['timestep'] = np.uint32(0); gs['new_apple'].fill(255)
    s["mem_occ"].fill(0); s["mem_food"].fill(0); s["mem_head"].fill(0)
    s["mem_age"].fill(255); s["est_len"].fill(3)
    cells = opening_apples(gs)
    set_food(gs, cells)
    # every opening apple counts as a spawn flash, so a blind player starts with
    # the same one-tick global glimpse the networks get from mem_food
    s["spawned"] = [[int(x), int(y)] for (x, y) in cells]
    for k in (1, 2, 3):
        clear_hidden(s, k)
    refresh_obs(s, 0)

REVERSE = [1, 0, 3, 2]

def real_heading(s):
    """Engine dir the human is travelling, or None before any real movement."""
    gs = s["gs"]; t = int(gs['timestep'])
    if t == 0:
        return None
    cur = int(gs['data'][GAME, 0, t % 225])
    prv = int(gs['data'][GAME, 0, (t - 1) % 225])
    if cur & 0xF == 0xF or prv & 0xF == 0xF:
        return None
    dx = (cur & 0xF) - (prv & 0xF)
    dy = ((cur >> 4) & 0xF) - ((prv >> 4) & 0xF)
    if dx == 0 and dy == 0:
        return None
    return 0 if dy < 0 else 1 if dy > 0 else 2 if dx < 0 else 3

def frame(s, sid, ignored=False):
    gs = s["gs"]; t = int(gs['timestep'])
    snakes = []
    for k in range(4):
        L = int(gs['length'][GAME, k]); segs = []
        for i in range(L):
            c = int(gs['data'][GAME, k, (t - L + 1 + i) % 225])
            x, y = c & 0xF, (c >> 4) & 0xF
            if x != 0xF:
                segs.append([x, y])
        snakes.append(segs)
    bits = np.unpackbits(gs['food'][GAME], bitorder='little')[:225]
    alive = [(int(gs['data'][GAME, k, t % 225]) & 0xF) != 0xF for k in range(4)]
    return {"sid": sid, "t": t, "snakes": snakes, "view_r": int(om.VIEW_R),
            "food": [[int(p % 15), int(p // 15)] for p in np.nonzero(bits)[0]],
            "spawned": s["spawned"], "health": gs['health'][GAME].tolist(), "alive": alive,
            "seats": s["seat_name"], "ignored": ignored,
            "over": sum(alive) <= 1, "won": alive[0] and sum(alive) <= 1}

# ---------------------------------------------------------------- state copy
# The live arrays on the session are pure scratch. Every real position lives in a
# snapshot hanging off a table node, so expansion can hop around freely.

STATE_KEYS = ("board", "head", "foodg", "headpos", "tailord", "mem_occ", "mem_food",
              "mem_head", "mem_age", "est_len", "rot", "obs", "scal", "act3")

def snapshot(s):
    d = {k: s[k].copy() for k in STATE_KEYS}
    d["gs"] = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in s["gs"].items()}
    d["hidden"] = [None if h is None else (h[0].clone(), h[1].clone()) for h in s["hidden"]]
    d["spawned"] = list(s["spawned"])
    return d

def restore(s, d):
    for k in STATE_KEYS:
        np.copyto(s[k], d[k])
    for k, v in d["gs"].items():
        if isinstance(v, np.ndarray):
            np.copyto(s["gs"][k], v)
        else:
            s["gs"][k] = v
    s["hidden"] = [None if h is None else (h[0].clone(), h[1].clone()) for h in d["hidden"]]
    s["spawned"] = list(d["spawned"])

def opp_moves(s):
    """Sample seats 1-3 once. They observe time t, so your move cannot affect them,
    which is what lets one inference pass cover every direction out of a node."""
    om.safe_moves3(s["board"], s["headpos"], s["rot"], om.ROT_OFF, s["act3"])
    hidden = list(s["hidden"])
    with torch.no_grad():
        for k in (1, 2, 3):
            x, sc = seat_obs(s, k)
            lg, _, hidden[k] = NETS[s["seat_name"][k]](x, sc, s["hidden"][k], priv=None)
            s["act3"][GAME, k] = int(torch.multinomial(torch.softmax(lg.float(), -1), 1))
    return INV_ROT_ACTION[s["rot"].astype(np.int64), om.ACT3_TO_CANON[s["act3"]]], hidden

def advance(s, raw, hidden, want_dir):
    gs = s["gs"]; t = int(gs['timestep'])
    raw = raw.copy()
    if want_dir is not None:
        raw[GAME, 0] = want_dir
    moves = (raw[:, 0] | (raw[:, 1] << 2) | (raw[:, 2] << 4) | (raw[:, 3] << 6)).astype(np.uint8)
    s["hidden"] = hidden
    game.step_wrapper(gs, moves)
    refresh_obs(s, t + 1)
    ax, ay = int(gs['new_apple'][GAME, 0]), int(gs['new_apple'][GAME, 1])
    s["spawned"] = [[ax, ay]] if ax != 255 else []

def tick(s, want_dir):
    """Single unbranched step, from the live arrays. Auto-play after death only."""
    raw, hidden = opp_moves(s)
    advance(s, raw, hidden, want_dir)
    return frame(s, None)

# ---------------------------------------------------------------- repeat table
# One entry per player action sequence the client can reach without asking again:
# every legal direction, each repeated 1..CHAIN times. Keys are absolute (prefixed
# with the committed history) so a commit never re-keys anything on the client.

def mknode(f, snap):
    # dead or decided: keep the frame so it can be rendered, but never extend it
    term = (not f["alive"][0]) or f["over"]
    return {"frame": f, "snap": snap, "term": term, "opp": None}

def table_reset(s):
    s["nodes"] = {(): mknode(frame(s, None), snapshot(s))}
    s["dirs"] = []
    build_table(s)

def node_dirs(s, path):
    nd = s["nodes"][path]
    if nd["term"]:
        return []
    if path:
        return [d for d in range(4) if d != REVERSE[path[-1]]]
    restore(s, nd["snap"])
    cur = real_heading(s)
    return [d for d in range(4) if cur is None or d != REVERSE[cur]]

def child(s, path, d):
    """Extend one node by one move, reusing that node's single opponent sample."""
    nd = s["nodes"][path]
    if nd["opp"] is None:
        restore(s, nd["snap"])
        nd["opp"] = opp_moves(s)
    raw, hidden = nd["opp"]
    restore(s, nd["snap"])
    advance(s, raw, hidden, d)
    return mknode(frame(s, None), snapshot(s))

def build_table(s):
    keep = {(): s["nodes"][()]}
    dirs = node_dirs(s, ())
    for d in dirs:
        p = ()
        for _ in range(CHAIN):
            nxt = p + (d,)
            if nxt not in s["nodes"]:            # cache hit means zero inference here
                if s["nodes"][p]["term"]:
                    break
                s["nodes"][nxt] = child(s, p, d)
            keep[nxt] = s["nodes"][nxt]
            p = nxt
            if keep[p]["term"]:
                break                            # chain dies here; no move N+1 exists
    s["nodes"] = keep
    s["dirs"] = dirs

def reroot(s, d):
    """Commit one move. Only pure chains are stored, so the surviving nodes are
    exactly that direction's chain, shifted down by one."""
    s["nodes"] = {p[1:]: nd for p, nd in s["nodes"].items() if p[:1] == (d,)}
    s["hist"].append(d)

def apply_path(s, path):
    hist = s["hist"]
    if list(path[:len(hist)]) != list(hist):
        return False
    for d in path[len(hist):]:
        if not isinstance(d, int):
            return False
        if (d,) not in s["nodes"]:
            build_table(s)                       # mid-chain turn: widen, then commit
        if (d,) not in s["nodes"]:
            return False
        reroot(s, d)
    return True

def payload(s, sid, seq, resync=False):
    pre = s["hist"]
    tab, leg = {}, {}
    for p, nd in s["nodes"].items():
        k = ".".join(map(str, pre + list(p)))
        tab[k] = nd["frame"]
        leg[k] = node_dirs(s, p)
    return {"sid": sid, "gen": s["gen"], "seq": seq, "hist": pre,
            "table": tab, "leg": leg, "resync": resync}

PAGE = r"""<!doctype html><meta charset=utf-8><title>play the snake</title>
<meta name=viewport content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<style>
:root{--bg:#f4f1ea;--panel:#fbfaf6;--line:#cec6b5;--ink:#26241f;--soft:#6b665c;
--board:#eae5d9;--font:ui-monospace,SFMono-Regular,Menlo,monospace}
*{box-sizing:border-box;font-family:var(--font)}
body{background:var(--bg);color:var(--ink);font-size:14px;line-height:1.6;margin:0;
min-height:100vh;display:flex;align-items:center;justify-content:center;padding:28px;
-webkit-text-size-adjust:100%}
#wrap{display:flex;gap:28px;align-items:flex-start}
canvas{background:var(--board);border:1px solid var(--line);border-radius:6px;display:block;
touch-action:none;-webkit-user-select:none;user-select:none;-webkit-tap-highlight-color:transparent}
#panel{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:16px;width:500px}
.row{display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid var(--line)}
.row:last-child{border-bottom:none}
.who{flex:1;min-width:0}
.nm{font-weight:600}
.st{font-size:12px;opacity:.8}
select{font-size:12px;font-family:var(--font);color:inherit;background:transparent;
border:none;border-bottom:1px solid var(--line);border-radius:0;padding:2px 16px 2px 4px;
max-width:150px;cursor:pointer;appearance:none;-webkit-appearance:none;
background-image:linear-gradient(45deg,transparent 50%,currentColor 50%),
linear-gradient(135deg,currentColor 50%,transparent 50%);
background-position:calc(100% - 8px) 55%,calc(100% - 4px) 55%;
background-size:4px 4px,4px 4px;background-repeat:no-repeat;opacity:.85}
select:hover{opacity:1;border-bottom-color:currentColor}
select:focus{outline:none;border-bottom-color:currentColor}
select option{background:#fff;color:var(--ink)}
button{background:#fff;color:var(--ink);border:1px solid var(--line);border-radius:4px;
padding:7px 18px;cursor:pointer;font-size:13px}
button:hover{background:#f0ece1}
#msg{font-size:19px;min-height:26px;text-align:center;margin:8px 0}
#hint{color:var(--soft);font-size:12px;text-align:center;margin-top:10px}
label{color:var(--soft);font-size:12px;display:flex;align-items:center;gap:6px;margin-top:12px}
#board{display:flex;flex-direction:column;align-items:center;gap:14px}

/* the d-pad is a fallback for when a swipe gets eaten mid-game */
#pad{display:none;grid-template-columns:repeat(3,58px);grid-template-rows:repeat(2,52px);gap:8px}
#pad button{padding:0;font-size:20px;line-height:1;touch-action:manipulation}
#pad .up{grid-area:1/2}#pad .lf{grid-area:2/1}#pad .dn{grid-area:2/2}#pad .rt{grid-area:2/3}
@media (pointer:coarse){#pad{display:grid}}

@media (max-width:900px){
  body{padding:12px;align-items:flex-start}
  #wrap{flex-direction:column;gap:16px;width:100%;align-items:stretch}
  #panel{width:100%;order:-1}      /* menu above the game on narrow screens */
  #board{width:100%}
  canvas{width:100%;height:auto;max-width:600px}
  select{max-width:44%}
  #msg{font-size:17px}
}
</style>
<div id=wrap>
  <div id=board>
    <canvas id=c width=600 height=600></canvas>
    <div id=pad>
      <button class=up data-dir=0>&#9650;</button>
      <button class=lf data-dir=2>&#9664;</button>
      <button class=dn data-dir=1>&#9660;</button>
      <button class=rt data-dir=3>&#9654;</button>
    </div>
  </div>
  <div id=panel>
    <div id=side></div>
    <label><input type=checkbox id=fog checked> show snake POVs</label>
    <label><input type=checkbox id=blind> blind mode (you see only what other snakes would see)</label>
    <div id=msg></div>
    <div style="text-align:center"><button id=r>Restart</button></div>
    <div id=hint>arrow keys/wasd for movement &amp; space restarts</div>
  </div>
</div>
<script>
const S=__SNAKES__,FOOD="__FOOD__",CELL=40,AUTO_MS=400;   // 2.5 moves/s after you die
const x=document.getElementById('c').getContext('2d');
let sid=null,over=false,last=null,MODELS=[];
let T={},L={},hist=[],queued=null,gen=0,seq=0,seen=0,pending=0;
let known=new Map();          // blind mode: apples glimpsed at spawn, not yet confirmed
const KEY={ArrowUp:0,ArrowDown:1,ArrowLeft:2,ArrowRight:3,w:0,s:1,a:2,d:3};
const blind=()=>document.getElementById('blind').checked;
const key=(p)=>p[0]+','+p[1];
if(matchMedia('(pointer:coarse)').matches)
  document.getElementById('hint').textContent='swipe the board to turn, or use the pad';

function flash(t){const m=document.getElementById('msg');m.textContent=t;
  setTimeout(()=>{if(m.textContent===t)m.textContent='';},1200);}

// T holds one entry per action sequence the server precomputed: each legal
// direction repeated 1..6 times. A hit renders with no network in the way.
function input(d){
  if(over||!last||!last.alive[0])return;
  const h=hist.join('.'),k=hist.concat(d).join('.');
  if(k in T){hist.push(d);draw(T[k]);ask();return;}
  if(L[h]&&!L[h].includes(d)){flash("you tried to twist your snakes neck?");return;}
  queued=d;ask();                 // off the repeat chain: go and get that branch
}

addEventListener('keydown',e=>{
  if(e.key===' '){e.preventDefault();restart();return;}
  if(!(e.key in KEY))return;
  e.preventDefault();
  if(e.repeat)return;
  input(KEY[e.key]);
});

// ---- touch: swipe on the board, plus an explicit pad ------------------------
let tx=0,ty=0,tt=0;
const cv=document.getElementById('c');
cv.addEventListener('touchstart',e=>{
  const t=e.changedTouches[0];tx=t.clientX;ty=t.clientY;tt=Date.now();
  e.preventDefault();},{passive:false});
cv.addEventListener('touchmove',e=>e.preventDefault(),{passive:false});
cv.addEventListener('touchend',e=>{
  e.preventDefault();
  const t=e.changedTouches[0],dx=t.clientX-tx,dy=t.clientY-ty;
  if(Math.abs(dx)<22&&Math.abs(dy)<22){
    if(over&&Date.now()-tt<400)restart();   // tap the dead board to play again
    return;}
  input(Math.abs(dx)>Math.abs(dy)?(dx>0?3:2):(dy>0?1:0));},{passive:false});
document.querySelectorAll('#pad button').forEach(b=>{
  b.addEventListener('click',()=>input(+b.dataset.dir));});

let panelBuilt=false;
function buildPanel(f){
  document.getElementById('side').innerHTML=f.seats.map((nm,k)=>{
    const sel=k===0?'':`<select data-seat="${k}">`+MODELS.map(m=>
      `<option${m===nm?' selected':''}>${m}</option>`).join('')+'</select>';
    return `<div class=row data-row="${k}"><div class=who>`
      +`<div class=nm></div><div class=st></div></div>${sel}</div>`;
  }).join('')+`<div class=row><span class=st id=tick style="color:var(--soft)"></span></div>`;
  document.querySelectorAll('select[data-seat]').forEach(s=>{
    s.onchange=()=>send('/opponent',{seat:+s.dataset.seat,model:s.value});});
  panelBuilt=true;
}
function updatePanel(f){
  // mutate text only: rebuilding innerHTML would tear down an open <select>
  f.seats.forEach((nm,k)=>{
    const row=document.querySelector(`.row[data-row="${k}"]`);
    if(!row)return;
    const dead=!f.alive[k];
    row.style.color=dead?S[k].dead:S[k].head;
    row.querySelector('.nm').textContent=S[k].label+(dead?' (ded)':'');
    row.querySelector('.st').textContent=`hp ${f.health[k]}  len ${f.snakes[k].length}`;
    const sel=row.querySelector('select');
    if(sel&&sel!==document.activeElement&&sel.value!==nm)sel.value=nm;});
  document.getElementById('tick').textContent=`tick ${f.t}`;
}

function inView(f,seat,gx,gy){
  const segs=f.snakes[seat];
  if(!f.alive[seat]||!segs.length)return false;
  const [hx,hy]=segs[segs.length-1];
  return Math.abs(gx-hx)+Math.abs(gy-hy)<=f.view_r;
}
function diamond(f,seat,fn){
  const segs=f.snakes[seat];
  if(!segs.length)return;
  const [hx,hy]=segs[segs.length-1];
  for(let dy=-f.view_r;dy<=f.view_r;dy++)for(let dx=-f.view_r;dx<=f.view_r;dx++){
    if(Math.abs(dx)+Math.abs(dy)>f.view_r)continue;
    const gx=hx+dx,gy=hy+dy;
    if(gx>=0&&gx<=14&&gy>=0&&gy<=14)fn(gx,gy);}
}
function trackApples(f){
  for(const p of f.spawned||[])known.set(key(p),p);
  const live=new Set(f.food.map(key));
  // anything inside your sight is resolved: either it is really there or it is gone
  for(const [k,p] of [...known])
    if(inView(f,0,p[0],p[1])&&!live.has(k))known.delete(k);
  for(const k of live)if(inView(f,0,...known.get(k)||[-9,-9]))known.delete(k);
}

function draw(f){
  last=f;
  if(blind()&&f.alive[0])trackApples(f);
  x.clearRect(0,0,600,600);
  const bl=blind()&&f.alive[0];   // death lifts the fog: nothing left to hide from you

  if(bl){
    // everything outside your range is unlit
    x.fillStyle='#ded8ca';x.fillRect(0,0,600,600);
    x.save();x.beginPath();
    diamond(f,0,(gx,gy)=>x.rect(gx*CELL,gy*CELL,CELL,CELL));
    x.clip();
    x.fillStyle=getComputedStyle(document.body).getPropertyValue('--board');
    x.fillRect(0,0,600,600);
    x.restore();
  }else if(document.getElementById('fog').checked){
    f.snakes.forEach((segs,k)=>{
      if(!f.alive[k])return;
      x.fillStyle=S[k].body;x.globalAlpha=.13;
      diamond(f,k,(gx,gy)=>x.fillRect(gx*CELL,gy*CELL,CELL,CELL));
      x.globalAlpha=1;});
  }

  x.strokeStyle='#d2cbba';
  for(let j=0;j<=15;j++){x.beginPath();x.moveTo(j*CELL,0);x.lineTo(j*CELL,600);x.stroke();
    x.beginPath();x.moveTo(0,j*CELL);x.lineTo(600,j*CELL);x.stroke();}

  x.fillStyle=FOOD;
  for(const [fx,fy] of f.food){
    if(bl&&!inView(f,0,fx,fy))continue;
    x.beginPath();x.arc(fx*CELL+20,fy*CELL+20,9,0,7);x.fill();}
  if(bl){   // remembered spawn flashes, held at half opacity until confirmed
    x.globalAlpha=.5;
    for(const [,p] of known){
      if(inView(f,0,p[0],p[1]))continue;
      x.beginPath();x.arc(p[0]*CELL+20,p[1]*CELL+20,9,0,7);x.fill();}
    x.globalAlpha=1;}

  f.snakes.forEach((segs,k)=>{
    segs.forEach(([sx,sy],n)=>{
      if(bl&&k!==0&&!inView(f,0,sx,sy))return;
      x.fillStyle=n===segs.length-1?S[k].head:S[k].body;
      x.fillRect(sx*CELL,sy*CELL,CELL,CELL);});});

  if(!panelBuilt)buildPanel(f);
  updatePanel(f);

  const msg=document.getElementById('msg');
  if(f.over){over=true;msg.textContent=f.won?'you win':(f.alive[0]?'draw':'You died');}
  else if(!f.alive[0])msg.textContent='You Died';
  else if(f.ignored)flash("you tried to twist your snakes neck?");
}

function prune(){
  const h=hist.join('.'),p=h?h+'.':'';
  for(const k in T)if(k!==h&&!k.startsWith(p))delete T[k];   // unreachable now
  for(const k in L)if(k!==h&&!k.startsWith(p))delete L[k];
}
function adopt(j){
  if(j.resync){T={};L={};hist=j.hist;draw(j.table[hist.join('.')]);}
  Object.assign(T,j.table||{});
  Object.assign(L,j.leg||{});
  prune();
  if(queued!==null){const q=queued;queued=null;input(q);}
}

// requests may overlap; each carries a sequence number and anything that comes
// back older than what has already been applied is thrown away
async function send(url,body){
  const mine=gen,n=++seq;
  pending++;
  try{
    const r=await fetch(url,{method:'POST',
      body:JSON.stringify(Object.assign({sid:sid,seq:n},body||{}))});
    const j=await r.json();
    if(j.sid)sid=j.sid;
    if(j.models)MODELS=j.models;
    if(mine!==gen)return null;                       // from before a restart
    if(j.gen!==undefined&&j.gen!==gen)return null;
    if(j.seq!==undefined&&j.seq<seen)return null;    // out of order
    seen=j.seq;
    adopt(j);
    return j;
  }catch(e){return null;}
  finally{pending--;}
}
const ask=()=>send('/move',{path:hist});

// one interval for the page lifetime, gated purely on current state, so there is
// no timer handle that can survive a restart
setInterval(()=>{
  if(pending||over||!last||last.alive[0])return;
  send('/step',{path:hist});
},AUTO_MS);

async function restart(){
  gen++;over=false;queued=null;
  T={};L={};hist=[];known=new Map();panelBuilt=false;
  document.getElementById('msg').textContent='';
  const mine=gen,n=++seq;
  const r=await fetch('/reset',{method:'POST',body:JSON.stringify({sid:sid,seq:n})});
  const j=await r.json();
  if(mine!==gen)return;
  if(j.sid)sid=j.sid;
  if(j.models)MODELS=j.models;
  gen=j.gen;seen=j.seq;adopt(j);
}
document.getElementById('fog').onchange=()=>{if(last)draw(last);};
document.getElementById('blind').onchange=()=>{known=new Map();if(last)draw(last);};
document.getElementById('r').onclick=restart;
restart();
</script>"""

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        page = PAGE.replace("__SNAKES__", json.dumps(SNAKES)).replace("__FOOD__", FOOD)
        self._send(page.encode(), "text/html; charset=utf-8")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        sid, s = get_session(req.get("sid"))
        seq = int(req.get("seq", 0))
        with s["lock"]:      # per-session: concurrent games never block each other
            if self.path == "/reset":
                s["gen"] += 1
                reset(s, reseat=True)
                s["hist"] = []
                table_reset(s)
                out = payload(s, sid, seq, resync=True)
                out["models"] = NAMES
            elif self.path == "/opponent":
                k = int(req["seat"])
                if 1 <= k <= 3 and req["model"] in NETS:
                    restore(s, s["nodes"][()]["snap"])
                    s["seat_name"][k] = req["model"]
                    clear_hidden(s, k)
                    s["hist"] = s["hist"] + [-1]   # new keyspace: the old table is void
                    table_reset(s)
                out = payload(s, sid, seq, resync=True)
                out["models"] = NAMES
            elif self.path == "/step":
                # auto-play once you are dead; nothing left to branch on
                ok = apply_path(s, req.get("path") or [])
                restore(s, s["nodes"][()]["snap"])
                if ok and not s["nodes"][()]["frame"]["over"]:
                    tick(s, None)
                s["hist"] = s["hist"] + [-1]
                table_reset(s)
                out = payload(s, sid, seq, resync=True)
            else:                                   # /move
                bad = not apply_path(s, req.get("path") or [])
                build_table(s)
                out = payload(s, sid, seq, resync=bad)
        self._send(json.dumps(out).encode(), "application/json")

    def log_message(self, *a):
        pass

print("warming kernels...")
_t0 = time.time()
_w = new_session(); restore(_w, _w["nodes"][()]["snap"]); tick(_w, 1)
print(f"warm in {time.time() - _t0:.1f}s")
_t0 = time.time(); restore(_w, _w["nodes"][()]["snap"]); tick(_w, 1)
print(f"per-tick: {(time.time() - _t0) * 1000:.1f} ms")
table_reset(_w)
_t0 = time.time(); build_table(_w)
print(f"cold table: {len(_w['nodes'])} nodes in {(time.time() - _t0) * 1000:.0f} ms")
reroot(_w, _w["dirs"][0])
_t0 = time.time(); build_table(_w)
print(f"after a repeat commit: {(time.time() - _t0) * 1000:.0f} ms (cache reused)")
print(f"serving on 0.0.0.0:{PORT}")
ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()