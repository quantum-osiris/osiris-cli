"""Exploratory (NOT preregistered) analyses for the RO-06/RO-07 paper. Writes docs/phase7/ro06_07_exploratory.json.

E1 H3 sensitivity: transition table over rounds 1-7 only (drops the 13.5 h gap between rounds 0 and 1).
E2 IBM update cadence: number of qubits whose IBM readout_error changed between consecutive snapshots
   (RO-06 rounds 0-7, then the RO-07 snapshot).
E3 For a device whose IBM values changed after RO-06 round 7, Spearman rho and median |gap| between the NEW
   IBM values and the probe of each RO-06 round, and between the OLD values and the round-7 probe.
"""
from __future__ import annotations
import json
import numpy as np
from scipy.stats import fisher_exact, spearmanr
from phase3 import readout_drift_06 as m
from phase3.readout_context_07 import OUT_DIR as P7


def main():
    out = {"note": "exploratory, not preregistered"}
    tab = np.zeros((2, 2), int)
    cad, e3 = {}, {}
    for b in m.BACKENDS:
        d = {r: m._round_errors(b, r) for r in range(m.ROUNDS)}
        for r in range(1, m.ROUNDS - 1):
            for q in d[r][0]:
                tab[int(d[r][0][q] > m.FAULT), int(d[r + 1][0][q] > m.FAULT)] += 1
        cal7 = json.loads((P7 / f"ro07_{b}_calibration.json").read_text())
        snaps = [d[r][2] for r in range(m.ROUNDS)] + [{int(q): v["readout_error"] for q, v in cal7["qubits"].items()}]
        times = [d[r][3] for r in range(m.ROUNDS)] + [cal7["captured_utc"]]
        cad[b] = {"snapshot_times": times,
                  "qubits_changed": [sum(1 for q in snaps[k] if snaps[k][q] != snaps[k + 1][q]) for k in range(len(snaps) - 1)]}
        if cad[b]["qubits_changed"][-1] > 0:
            new = snaps[-1]
            rows = []
            for r in range(m.ROUNDS):
                ep = d[r][0]
                qs = [q for q in ep if new[q] is not None]
                rows.append({"round": r, "time": d[r][3], "rho": float(spearmanr([new[q] for q in qs], [ep[q] for q in qs]).statistic),
                             "median_abs_gap": float(np.median([abs(new[q] - ep[q]) for q in qs]))})
            ep = d[m.ROUNDS - 1][0]
            old = d[m.ROUNDS - 1][2]
            qs = [q for q in ep if old[q] is not None]
            e3[b] = {"new_vs_round": rows, "old_vs_last_round": {
                "rho": float(spearmanr([old[q] for q in qs], [ep[q] for q in qs]).statistic),
                "median_abs_gap": float(np.median([abs(old[q] - ep[q]) for q in qs]))}}
    o, p = fisher_exact(tab)
    out["E1_H3_rounds_1_to_7"] = {"table": tab.tolist(), "odds_ratio": float(o), "p": float(p),
                                  "p_fault_next_given_fault": float(tab[1, 1] / tab[1].sum()),
                                  "p_fault_next_given_ok": float(tab[0, 1] / tab[0].sum())}
    out["E2_ibm_update_cadence"] = cad
    out["E3_update_alignment"] = e3
    (P7 / "ro06_07_exploratory.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "E2_ibm_update_cadence"}, indent=1)[:1500])


if __name__ == "__main__":
    main()
