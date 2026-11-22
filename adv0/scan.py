#!/usr/bin/env python3
"""ADV-0 scan: for each (n, depth), build a peaked circuit, run the classical attacks, and predict hardware.

Outputs (docs/adv0/):
  scan.jsonl          one line per (n, depth): attack results, CZ count, predicted peak probability, QPU seconds
  noisy_check.jsonl   predicted vs simulated peak probability under FakeMarrakesh's full noise model (small n)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from adv0 import attacks as A
from adv0 import peaked as P

OUT = Path(os.environ.get("ADV0_OUT", Path(__file__).resolve().parents[1] / "docs" / "adv0"))


def scan(ns, depths, seed=20261121, pp_terms=2 ** 15, lc_limit=26, chis=(64, 256)):
    OUT.mkdir(parents=True, exist_ok=True)
    be = P.backend()
    f = open(OUT / "scan.jsonl", "a")
    for n in ns:
        reg = P.choose_region(be, n)
        for d in depths:
            p = P.build(n, d, seed + 1000 * n + d, be, reg)
            row = {"n": n, "depth": d, "seed": p.seed, "region": p.region, "n_cz": p.n_cz,
                   "n_1q": sum(1 for g in p.gates if g[0] in ("sx", "x")), **P.hardware_estimate(p, be)}
            row["pauli_prop"] = A.pauli_propagation(p, eps=1e-5, max_terms=pp_terms)
            row["light_cone"] = A.light_cone_attack(p, limit=lc_limit)
            row["mps"] = [A.mps_attack(p, chi=c, shots=200) for c in chis]
            row["classical_win"] = (row["pauli_prop"]["win"] or row["light_cone"]["win"]
                                    or any(m["win"] for m in row["mps"]))
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(n, d, "cz", p.n_cz, "p_hw %.3g" % row["p_peak_pred"], "PP", row["pauli_prop"]["win"],
                  "LC", row["light_cone"]["win"], "MPS", [m["win"] for m in row["mps"]],
                  "t", round(sum([row["pauli_prop"]["seconds"], row["light_cone"]["seconds"]]
                                 + [m["seconds"] for m in row["mps"]]), 1), flush=True)


def noisy_check(n=12, depths=(2, 4, 6, 8, 12), shots=4000, seed=20261121):
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    OUT.mkdir(parents=True, exist_ok=True)
    be = P.backend()
    sim = AerSimulator.from_backend(be)
    reg = P.choose_region(be, n)
    with open(OUT / "noisy_check.jsonl", "a") as f:
        for d in depths:
            p = P.build(n, d, seed + 1000 * n + d, be, reg)
            isa = transpile(p.qiskit(), be, initial_layout=reg, optimization_level=0)
            assert isa.count_ops().get("cz", 0) == p.n_cz, "routing added gates"
            c = sim.run(isa, shots=shots, seed_simulator=seed).result().get_counts()
            row = {"n": n, "depth": d, "n_cz": p.n_cz, "pred": P.hardware_estimate(p, be)["p_peak_pred"],
                   "sim": c.get(p.secret[::-1], 0) / shots, "modal_is_secret": max(c, key=c.get) == p.secret[::-1],
                   "duration_us": isa.estimate_duration(be.target, unit="u") if hasattr(isa, "estimate_duration") else None}
            f.write(json.dumps(row) + "\n")
            print(row, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="12,20,28,40,56")
    ap.add_argument("--depths", default="2,4,6,8,12,16")
    ap.add_argument("--noisy", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    if a.noisy:
        noisy_check()
    else:
        scan([int(x) for x in a.ns.split(",")], [int(x) for x in a.depths.split(",")])
    print("done in %.0f s" % (time.time() - t0))
