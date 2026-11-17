import json, sys, numpy as np
E = sys.argv[1].rstrip("/") + "/"
o, n = json.load(open(E + "prefix_968b7516_gradcheck.json")), json.load(open(E + "fixed_be8f506d_gradcheck.json"))
def summ(rows):
    j = [r for r in rows if "errors" in r]
    st = [r["status"] for r in rows]
    return len(rows), st.count("FAIL"), st.count("no_gradient"), st.count("zero_gradient"), [max(r["errors"][k]["rel_err"] for r in j) for k in range(6)], max(min(e["abs_err"] for e in r["errors"]) for r in j)
print("case | rev | tensors fail no_grad zero_grad | max rel_err at eps 1e-1 3e-2 1e-2 3e-3 1e-3 3e-4 | max best abs_err")
for k in o["cases"]:
    for lab, d in (("pre-fix", o), ("fixed  ", n)):
        t, f, ng, zg, mx, ab = summ(d["cases"][k])
        print(f"{k:22s} {lab} {t:3d} {f:3d} {ng} {zg} | " + " ".join(f"{v:.4f}" for v in mx) + f" | {ab:.2e}")
a, b = np.load(E + "prefix_968b7516_forward.npz"), np.load(E + "fixed_be8f506d_forward.npz")
print("forward arrays array_equal:", all(np.array_equal(a[k], b[k]) for k in a.files), "of", len(a.files), "arrays; dtypes", sorted({str(a[k].dtype) for k in a.files}))
