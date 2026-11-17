"""Numerical check of the NCLM backward pass, and a forward record for comparing two revisions.

    cd <an osiris-cli checkout> && python -B experiments/nclm_tlock_gradfix/validate.py <label> <outdir>

For every parameter tensor p with gradient g (float32, dropout 0, fixed seeds):
    d = g / ||g||;  slope(eps) = (L(p + eps d) - L(p - eps d)) / (2 eps)
    abs_err(eps) = |slope - ||g|||;  rel_err(eps) = abs_err / max(|slope|, ||g||, 1e-8)
All six eps are recorded. A tensor PASSES when rel_err <= 0.02 for at least one eps, and the number of eps
at or under 0.02 is recorded beside it. Tensors with ||g|| < 1e-10 are listed as zero_gradient and not judged;
tensors with no gradient at all are listed as no_gradient. 0.02 is about 60x the largest error the plain
("standard") architecture shows (0.0003) and far below the defect this was written to detect (0.09-0.16).
Limits: float32 only (the model keeps the loss in float32 even with float64 weights); no independent
autodiff reference; three inputs per architecture."""
import json, os, random, sys
import numpy as np
sys.path.insert(0, os.getcwd())
import osiris_termux_console as otc
from osiris.nclm import cross_entropy_loss
from osiris.nclm.autograd import no_grad
EPS = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4)
CASES = [(0, 2, 48), (1, 1, 128), (2, 3, 17)]   # (seed, batch, length); 128 is the full context window
TOL = 0.02

def L(m, x, y):
    with no_grad():
        return float(np.float64(cross_entropy_loss(m.forward(x), y).data))

label, outdir = sys.argv[1], sys.argv[2]
report, arrays = {"label": label, "eps": EPS, "tolerance": TOL, "cases": {}}, {}
for arch in ("crsm", "standard"):
    for case, (seed, B, T) in enumerate(CASES):
        np.random.seed(seed); random.seed(seed)
        m, opt = otc._organism_build(arch)
        rng = np.random.default_rng(100 + seed)
        x, y = rng.integers(0, 256, size=(B, T)), rng.integers(0, 256, size=(B, T))
        with no_grad():
            arrays[f"{arch}_{case}_logits"] = m.forward(x).data
        arrays[f"{arch}_{case}_loss"] = np.array(L(m, x, y))
        opt.zero_grad(); cross_entropy_loss(m.forward(x), y).backward()
        rows = []
        for i, p in enumerate(m.parameters()):
            name = getattr(p, "name", "")
            if p.grad is None: rows.append({"i": i, "name": name, "shape": list(p.data.shape), "status": "no_gradient"}); continue
            g = np.asarray(p.grad, dtype=np.float64); n = float(np.linalg.norm(g))
            if n < 1e-10: rows.append({"i": i, "name": name, "shape": list(p.data.shape), "status": "zero_gradient", "grad_norm": n}); continue
            d, old, errs = (g / n).astype(p.data.dtype), p.data.copy(), []
            for eps in EPS:
                p.data[...] = old + eps * d; up = L(m, x, y)
                p.data[...] = old - eps * d; dn = L(m, x, y)
                p.data[...] = old
                s = (up - dn) / (2 * eps); a = abs(s - n)
                errs.append({"eps": eps, "slope": s, "abs_err": a, "rel_err": a / max(abs(s), n, 1e-8)})
            ok = sum(e["rel_err"] <= TOL for e in errs)
            rows.append({"i": i, "name": name, "shape": list(p.data.shape), "grad_norm": n, "errors": errs, "eps_within_tol": ok,
                         "status": "pass" if ok else "FAIL"})
        report["cases"][f"{arch}/seed{seed}/B{B}xT{T}"] = rows
os.makedirs(outdir, exist_ok=True)
np.savez(os.path.join(outdir, f"{label}_forward.npz"), **arrays)
json.dump(report, open(os.path.join(outdir, f"{label}_gradcheck.json"), "w"), indent=1)
