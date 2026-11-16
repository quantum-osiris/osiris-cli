"""Exploratory before/after comparison (NOT the registered NCLM-ARCH-1 protocol).
Same corpus, same split, same seeds, same step budget; only the checkout differs.
usage: compare.py <label> <seed> <steps> <out.json>   (run with cwd = the osiris-cli checkout to test)"""
import gc, glob, hashlib, json, os, random, sys, time
import numpy as np
sys.path.insert(0, os.getcwd())
import osiris_termux_console as otc
from osiris.nclm import cross_entropy_loss
from osiris.nclm.autograd import no_grad, clip_grad_norm
label, seed, steps, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
W = "/home/osiris/osiris-assessment/work"
files = sorted(f for r in ("dnalang-core", "osiris-governance", "bridge", "organism_sim") for f in glob.glob(f"{W}/{r}/**/*.md", recursive=True))
train, held, manifest = [], [], []
for f in files:
    b = open(f, "rb").read()
    side = "heldout" if int(hashlib.sha256(os.path.relpath(f, W).encode()).hexdigest(), 16) % 10 == 0 else "train"
    (held if side == "heldout" else train).append(b); manifest.append([os.path.relpath(f, W), hashlib.sha256(b).hexdigest(), side])
tr = np.frombuffer(b"\n".join(train), dtype=np.uint8); he = np.frombuffer(b"\n".join(held), dtype=np.uint8)
T, B = 128, 8
np.random.seed(seed); random.seed(seed); rng = np.random.default_rng(seed)
model, opt = otc._organism_build("crsm")
ev = np.random.default_rng(12345); starts = ev.integers(0, len(he) - T - 1, size=64)   # fixed held-out windows, same for every run
def heldout():
    tot = 0.0
    with no_grad():
        for i in range(0, 64, 16):
            x = np.stack([he[s:s + T] for s in starts[i:i + 16]]).astype(np.int64); y = np.stack([he[s + 1:s + T + 1] for s in starts[i:i + 16]]).astype(np.int64)
            tot += float(cross_entropy_loss(model.forward(x), y).data)
    return tot / 4 / np.log(2)
curve = [[0, heldout()]]; t0 = time.time()
for step in range(1, steps + 1):
    s = rng.integers(0, len(tr) - T - 1, size=B)
    x = np.stack([tr[i:i + T] for i in s]).astype(np.int64); y = np.stack([tr[i + 1:i + T + 1] for i in s]).astype(np.int64)
    opt.zero_grad(); loss = cross_entropy_loss(model.forward(x), y); loss.backward(); clip_grad_norm(model.parameters(), 1.0); opt.step()
    if step % 10 == 0: gc.collect()   # the autograd graph holds reference cycles; without this a run grows past 6 GB
    if step % 200 == 0: curve.append([step, heldout()]); print(step, round(curve[-1][1], 4), flush=True)
json.dump({"label": label, "seed": seed, "steps": steps, "train_bytes": int(len(tr)), "heldout_bytes": int(len(he)), "files": len(files),
           "manifest_sha256": hashlib.sha256(json.dumps(manifest).encode()).hexdigest(), "curve": curve,
           "best_heldout_bpb": min(c[1] for c in curve), "final_heldout_bpb": curve[-1][1], "seconds": round(time.time() - t0)}, open(out, "w"))
