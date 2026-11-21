"""Writes generated.tex (macros + tables) from the committed result files. No hand-copied numbers."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = Path(__file__).resolve().parent
hw = json.loads((ROOT / "docs/xtop005/qpu/results.json").read_text())
sim = json.loads((ROOT / "docs/xtop005/stage1_hh/xtop005_stage1.json").read_text())
qd = json.loads((ROOT / "docs/xtop005/quantinuum/xtop005_quantinuum_design.json").read_text())
lock = json.loads((ROOT / "docs/xtop005/qpu/lock.json").read_text())
f = lambda s: float(s)
out = []
m = lambda k, v: out.append(f"\\newcommand{{\\{k}}}{{{v}}}")
m("JobID", hw["job_id"]); m("QPUs", f"{f(hw['job_usage_seconds']):.0f}")
m("QPYsha", lock["qpy_sha256"][:16]); m("NCirc", lock["n_circuits"])
m("QSha", qd["circuits_sha256"][:16])
for s, v in qd["hqc_total_all_cells"].items():
    m("HQCtotal" + {"200": "A", "500": "B", "1000": "C"}[s], f"{f(v):,.0f}")

simF = {}
for c in sim["cells"]:
    k = (c["arm"], c["depth"], int(f(c["idle_ns"])))
    simF[k] = f(c.get("F_det", c["F_raw"]))
hwc = {(c["arm"], c["depth"], int(f(c["idle_ns"]))): c for c in hw["cells"]}

rows = []
for L in (1, 3):
    for idle in (2000, 10000, 20000):
        cells = []
        for a in "ABCD":
            c = hwc[(a, L, idle)]
            F = f(c["F_used"]); lo, hi = (f(v) for v in c["F_used_ci99"])
            best = max(f(hwc[(b, L, idle)]["F_used"]) for b in "ABCD")
            txt = f"{F:.3f}"
            if F == best:
                txt = f"\\textbf{{{txt}}}"
            if a in "CD":
                txt += f" ({100 * f(c['p_keep']):.0f}\\%)"
            cells.append(txt)
        rows.append(f"{L} & {idle // 1000} & " + " & ".join(cells) + r" \\")
m("HWTable", "\n".join(rows))

rows = []
for L in (1, 3):
    for idle in (2000, 10000, 20000):
        rows.append(f"{L} & {idle // 1000} & " + " & ".join(
            f"{simF[(a, L, idle)]:.3f} / {f(hwc[(a, L, idle)]['F_used']):.3f}" for a in "ABCD") + r" \\")
m("SimTable", "\n".join(rows))

name = {"H1": "Crossover (C$-$A at 20 vs 2\\,\\textmu s)", "H2": "DD on bare (B$-$A, 20\\,\\textmu s)",
        "H3": "DD on encoded (D$-$C, 20\\,\\textmu s)", "H4": "Encoded beats bare (C$-$A, 20\\,\\textmu s)"}
rows = []
for t in hw["tests"]:
    lo, hi = (f(v) for v in t["ci99"])
    rows.append(f"{name[t['test']]} & {t['depth']} & {f(t['estimate']):+.3f} & [{lo:+.3f}, {hi:+.3f}] & "
                f"{f(t['p_holm']):.3f} & {'yes' if t['supported'] in (True, 'True', 1) else 'no'}" + r" \\")
m("TestTable", "\n".join(rows))

rows = []
for c in qd["cells"]:
    lo, hi = (f(v) for v in c["F_used_ci99"])
    h = c["h2_counts_avg"]
    rows.append(f"{c['label'].replace('_', chr(92) + '_')} & {c['depth']} & {c['qubits']} & {f(h['n2']):.0f} & "
                f"{f(c['F_used']):.3f} [{lo:.3f}, {hi:.3f}] & {100 * f(c['p_keep']):.1f}\\% & {f(c['hqc_9_settings']['500']):,.0f}" + r" \\")
m("QTable", "\n".join(rows))
qc = {(c["label"], c["depth"]): c for c in qd["cells"]}
for shots, tag in (("500", "B"), ("1000", "C")):
    m("HQCmin" + tag, f"{f(qc[('BARE', 3)]['hqc_9_settings'][shots]) + f(qc[('ENC_ALL_CHK', 3)]['hqc_9_settings'][shots]):,.0f}")
m("QbareLthree", f"{f(qc[('BARE', 3)]['F_used']):.3f}"); m("QchkLthree", f"{f(qc[('ENC_ALL_CHK', 3)]['F_used']):.3f}")
m("QchkKeep", f"{100 * f(qc[('ENC_ALL_CHK', 3)]['p_keep']):.0f}")
(D / "generated.tex").write_text("\n".join(out) + "\n")
print("wrote", len(out), "macros")
