#!/usr/bin/env python3
"""OSIRIS-XTOP-005 hardware run (heavy-hex design): lock -> [preregister] -> submit -> analyze.

  lock     compile all circuits against the LIVE backend calibration, run Stage 0 (noiseless ISA check) and write
           docs/xtop005/qpu/lock.json (parameters, layouts, per-circuit ISA counts, QPY SHA-256) + circuits.qpy.
  submit   refuses unless lock.json and circuits.qpy still match, the preregistration DOI is recorded in
           lock.json's sidecar prereg.json, and enough quota is left; then ONE SamplerV2 job with every circuit.
  analyze  fidelities, 99 % bootstrap CIs, and the preregistered contrasts with Holm correction.

Credentials: OSIRIS_IBM_KEY_FILE (JSON with "apikey"). Nothing here prints a credential.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np

from xtop005 import osiris_xtop_005 as x

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("XTOP005_QPU_OUT", ROOT / "docs" / "xtop005" / "qpu"))
BACKEND = "ibm_marrakesh"
DEPTHS = (1, 3)
IDLES = (2000.0, 10000.0, 20000.0)
SHOTS = 1000
SEED = 20261121
S_PER_SHOT = 0.30e-3   # this account's measured static-circuit rate (MAP-04: 0.29 ms/shot)
QUOTA_SAFETY = 1.5
ALPHA = 0.01

# Preregistered contrasts (difference of two-pair Bell fidelity; "C*" = post-selected encoded fidelity F_det):
#  H1  code-space crossover: [F_det(C) - F(A)] at 20 us  minus  the same at 2 us  > 0         (each depth)
#  H2  hardware DD on bare:  F(B) - F(A) at 20 us > 0                                          (each depth)
#  H3  DD on encoded:        F_det(D) - F_det(C) at 20 us > 0                                  (each depth)
#  H4  long-idle advantage:  F_det(C) - F(A) at 20 us > 0                                      (each depth)
# 8 one-sided tests, Holm-corrected at family-wise alpha = 0.01, bootstrap p-values (4000 resamples).


def _service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    kf = os.environ.get("OSIRIS_IBM_KEY_FILE")
    if not kf:
        raise SystemExit("set OSIRIS_IBM_KEY_FILE")
    tok = json.loads(Path(kf).expanduser().read_text())["apikey"]
    return QiskitRuntimeService(channel="ibm_quantum_platform", token=tok)


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _circuits(cells) -> List:
    return [isa for cell in cells for _, isa in cell["items"]]


def lock(backend_name: str = BACKEND) -> Dict:
    from qiskit import qpy
    from phase3.live_hardware_runner import snapshot_backend_calibration
    OUT.mkdir(parents=True, exist_ok=True)
    svc = _service()
    be = svc.backend(backend_name)
    layouts, cells = x.prepare(be, DEPTHS, IDLES, "r")
    x.stage0(cells)
    circs = _circuits(cells)
    buf = io.BytesIO(); qpy.dump(circs, buf); qb = buf.getvalue()
    (OUT / "circuits.qpy").write_bytes(qb)
    (OUT / "lock_time_calibration.json").write_text(json.dumps(snapshot_backend_calibration(be), indent=1) + "\n")
    meta = {"experiment": "OSIRIS-XTOP-005 (heavy-hex design: encoded quarantined reference)",
            "backend": backend_name, "depths": DEPTHS, "idles_ns": IDLES, "shots": SHOTS, "seed": SEED,
            "arms": x.ARM_LABEL, "settings": ["".join(s) for s in x.SETTINGS],
            "n_circuits": len(circs), "est_qpu_s": round(len(circs) * SHOTS * S_PER_SHOT + 3, 1),
            "layouts": {k: {"physical": v[0], **v[1]} for k, v in layouts.items()},
            "cells": [{k: v for k, v in c.items() if k != "items"} for c in cells],
            "circuit_order": [[c["arm"], c["depth"], c["idle_ns"], "".join(b.setting)] for c in cells for b, _ in c["items"]],
            "qpy_sha256": _sha_bytes(qb),
            "locked_utc": datetime.now(timezone.utc).isoformat(),
            "contrasts": ["H1 crossover", "H2 DD bare", "H3 DD encoded", "H4 long-idle advantage"],
            "alpha_familywise": ALPHA}
    body = x.canonical_json(meta)
    (OUT / "lock.json").write_text(body + "\n")
    print("locked", len(circs), "circuits; qpy", meta["qpy_sha256"][:16], "lock", _sha_bytes(body.encode())[:16],
          "est", meta["est_qpu_s"], "s")
    return meta


def _check_lock() -> Dict:
    lk = json.loads((OUT / "lock.json").read_text())
    if _sha_bytes((OUT / "circuits.qpy").read_bytes()) != lk["qpy_sha256"]:
        raise SystemExit("circuits.qpy does not match lock.json")
    return lk


def submit() -> None:
    from qiskit import qpy
    from qiskit_ibm_runtime import SamplerV2
    lk = _check_lock()
    pre = OUT / "prereg.json"
    if not pre.exists() or not json.loads(pre.read_text()).get("doi"):
        raise SystemExit("no published preregistration DOI in docs/xtop005/qpu/prereg.json; not submitting")
    rec = OUT / "job_receipt.json"
    if rec.exists():
        raise SystemExit(f"already submitted: {json.loads(rec.read_text())['job_id']}")
    svc = _service()
    be = svc.backend(lk["backend"])
    rem = float(svc.usage().get("usage_remaining_seconds", 0) or 0)
    need = QUOTA_SAFETY * float(lk["est_qpu_s"])
    if rem < need:
        raise SystemExit(f"{rem:.0f} s quota left, need {need:.0f} s; not submitting")
    with open(OUT / "circuits.qpy", "rb") as fh:
        circs = qpy.load(fh)
    job = SamplerV2(mode=be).run(circs, shots=int(lk["shots"]))
    rec.write_text(json.dumps({"job_id": job.job_id(), "backend": be.name, "shots": lk["shots"],
                               "n_pubs": len(circs), "qpy_sha256": lk["qpy_sha256"],
                               "prereg_doi": json.loads(pre.read_text())["doi"],
                               "submitted_utc": datetime.now(timezone.utc).isoformat(),
                               "quota_remaining_before_s": rem}, indent=2) + "\n")
    print("submitted", job.job_id(), "pubs", len(circs), flush=True)


def fetch() -> Dict:
    from phase3.execute_and_analyze_live import _job_usage_seconds, _public_usage
    from phase3.live_hardware_runner import extract_counts_from_pub_result, snapshot_backend_calibration
    raw = OUT / "raw_counts.json"
    if raw.exists():
        return json.loads(raw.read_text())
    svc = _service()
    rec = json.loads((OUT / "job_receipt.json").read_text())
    job = svc.job(rec["job_id"])
    res = job.result()
    out = {"job_id": rec["job_id"], "completed_utc": datetime.now(timezone.utc).isoformat(),
           "job_usage_seconds": _job_usage_seconds(job), "post_job_usage": _public_usage(svc.usage()),
           "execution_time_calibration": snapshot_backend_calibration(svc.backend(rec["backend"])),
           "pub_counts": [extract_counts_from_pub_result(res[i]) for i in range(rec["n_pubs"])]}
    raw.write_text(json.dumps(out) + "\n")
    return out


def _boot_F(per, frame, which, rng, n_boot):
    """Bootstrap replicates of F for one cell (multinomial resampling per setting, as in osiris_xtop_005)."""
    vals = np.empty(n_boot)
    probs = {}
    for k, d in per.items():
        v = d[which]; n = v["n"]
        p = np.array([(n + v["c1"] + v["c2"] + v["c12"]), (n + v["c1"] - v["c2"] - v["c12"]),
                      (n - v["c1"] + v["c2"] - v["c12"]), (n - v["c1"] - v["c2"] + v["c12"])], float) / (4 * n)
        probs[k] = (n, np.clip(p, 0, None) / np.clip(p, 0, None).sum())
    for i in range(n_boot):
        res = {}
        for k, (n, p) in probs.items():
            a, b, c_, d = rng.multinomial(n, p)
            res[k] = {which: {"n": n, "c1": a + b - c_ - d, "c2": a - b + c_ - d, "c12": a - b - c_ + d}}
        vals[i] = x.fidelity(res, frame, which)
    return vals


def analyze(n_boot: int = 4000) -> Dict:
    from qiskit import qpy
    lk = _check_lock()
    raw = fetch()
    with open(OUT / "circuits.qpy", "rb") as fh:
        circs = qpy.load(fh)
    # rebuild the Built bookkeeping (pure Python, deterministic) in lock order
    rng = np.random.default_rng(SEED)
    cells, idx = {}, 0
    for arm, L, idle, st in lk["circuit_order"]:
        b = x.build(arm, int(L), (st[0], st[1]), float(idle), "r")
        cells.setdefault((arm, int(L), float(idle)), []).append((b, raw["pub_counts"][idx])); idx += 1
    F, boots, summary = {}, {}, []
    for (arm, L, idle), items in cells.items():
        frame = x.logical_frame(L)
        per = {"".join(b.setting): x.correlators(b, cnt) for b, cnt in items}
        which = "acc" if x.ENCODED[arm] else "raw"
        F[(arm, L, idle)] = x.fidelity(per, frame, which)
        boots[(arm, L, idle)] = _boot_F(per, frame, which, rng, n_boot)
        row = {"arm": arm, "label": x.ARM_LABEL[arm], "depth": L, "idle_ns": idle,
               "F_raw": x.fidelity(per, frame, "raw"),
               "F_used": F[(arm, L, idle)], "F_used_ci99": list(np.quantile(boots[(arm, L, idle)], [0.005, 0.995]))}
        if x.ENCODED[arm]:
            row["p_keep"] = sum(v["acc"]["n"] for v in per.values()) / sum(v["raw"]["n"] for v in per.values())
        row["correlator_sums"] = per
        summary.append(row)
    lo, hi = min(IDLES), max(IDLES)
    tests = []
    for L in DEPTHS:
        g = lambda a, i: boots[(a, L, i)]
        f = lambda a, i: F[(a, L, i)]
        defs = {
            "H1": ((f("C", hi) - f("A", hi)) - (f("C", lo) - f("A", lo)), (g("C", hi) - g("A", hi)) - (g("C", lo) - g("A", lo))),
            "H2": (f("B", hi) - f("A", hi), g("B", hi) - g("A", hi)),
            "H3": (f("D", hi) - f("C", hi), g("D", hi) - g("C", hi)),
            "H4": (f("C", hi) - f("A", hi), g("C", hi) - g("A", hi)),
        }
        for h, (est, bt) in defs.items():
            p = (1 + np.sum(bt <= 0)) / (1 + len(bt))   # one-sided H: effect > 0
            tests.append({"test": h, "depth": L, "estimate": float(est), "ci99": list(np.quantile(bt, [0.005, 0.995])),
                          "p_one_sided": float(p)})
    order = sorted(range(len(tests)), key=lambda i: tests[i]["p_one_sided"])
    m, running = len(tests), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * tests[i]["p_one_sided"]))
        tests[i]["p_holm"] = running
        tests[i]["supported"] = running < ALPHA
    out = {"job_id": raw["job_id"], "job_usage_seconds": raw["job_usage_seconds"], "cells": summary, "tests": tests,
           "lock_qpy_sha256": lk["qpy_sha256"]}
    body = x.canonical_json(out)
    (OUT / "results.json").write_text(body + "\n")
    (OUT / "results.receipt.json").write_text(json.dumps({"canon": "exec-canon-1", "sha256": _sha_bytes(body.encode())}) + "\n")
    for r in summary:
        print(r["arm"], "L", r["depth"], "idle", int(r["idle_ns"]), "F", round(r["F_used"], 4),
              [round(float(v), 4) for v in r["F_used_ci99"]], "keep", round(r.get("p_keep", 1), 3))
    for t in tests:
        print(t["test"], "L", t["depth"], "est", round(t["estimate"], 4), "ci99", [round(float(v), 4) for v in t["ci99"]],
              "p_holm", round(t["p_holm"], 4), "SUPPORTED" if t["supported"] else "not supported")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("lock", "submit", "analyze"))
    a = ap.parse_args(argv)
    {"lock": lock, "submit": submit, "analyze": analyze}[a.stage]()


if __name__ == "__main__":
    main()
