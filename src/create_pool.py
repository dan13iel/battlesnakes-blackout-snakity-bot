import torch, os, glob

# used to rebuild the pool if it gets corrupted/deleted

POOL_PATH = "checkpoints/pool.pt"
CKPT_DIR = "checkpoints"
POOL_DTYPE = torch.float16
STRIP_CRITIC = True

# find all checkpoints
ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "ckpt_ep*.pt")))
if not ckpts:
    print(f"no ckpt_ep*.pt files found in {CKPT_DIR}/")
    exit()

print(f"found {len(ckpts)} checkpoints:")
for i, c in enumerate(ckpts):
    print(f"  [{i}] {os.path.basename(c)}")

sel = input("\nadd which? (comma-sep indices, 'all', or 'every N'): ").strip()

if sel == "all":
    picks = ckpts
elif sel.startswith("every"):
    n = int(sel.split()[1])
    picks = ckpts[::n]
else:
    idxs = [int(x.strip()) for x in sel.split(",")]
    picks = [ckpts[i] for i in idxs]

if os.path.exists(POOL_PATH):
    data = torch.load(POOL_PATH, map_location="cpu")
else:
    data = {"pool": [], "pool_wins": [], "pool_games": [], "episode": 0, "cfg": None}

for p in picks:
    ckpt = torch.load(p, map_location="cpu")
    sd = ckpt["model"] if "model" in ckpt else ckpt
    if STRIP_CRITIC:
        sd = {k: v for k, v in sd.items()
              if not (k.startswith("priv_enc.") or k.startswith("v."))}
    sd = {k: v.to(POOL_DTYPE) for k, v in sd.items()}
    data["pool"].append(sd)
    data["pool_wins"].append(0)
    data["pool_games"].append(0)
    print(f"  added {os.path.basename(p)}")

torch.save(data, POOL_PATH)
print(f"\npool now has {len(data['pool'])} entries")