import torch, os

POOL_PATH = "checkpoints/pool.pt"
POOL_DTYPE = torch.float16
STRIP_CRITIC = True

path = input("checkpoint path: ").strip()
ckpt = torch.load(path, map_location="cpu")
sd = ckpt["model"] if "model" in ckpt else ckpt

if STRIP_CRITIC:
    sd = {k: v for k, v in sd.items() if not (k.startswith("priv_enc.") or k.startswith("v."))}
sd = {k: v.to(POOL_DTYPE) for k, v in sd.items()}

if os.path.exists(POOL_PATH):
    data = torch.load(POOL_PATH, map_location="cpu")
else:
    data = {"pool": [], "pool_wins": [], "pool_games": [], "episode": 0, "cfg": None}

data["pool"].append(sd)
data["pool_wins"].append(0)
data["pool_games"].append(0)

torch.save(data, POOL_PATH)
print(f"added {os.path.basename(path)}; pool now has {len(data['pool'])} entries")