"""TESSERACT-RO-06: how common, and how persistent, are readout faults that IBM's calibration does not show?

Follows TESSERACT-MAP-03/04/05 (doi:10.5281/zenodo.24406525, 24407001, MAP-05 record). ibm_marrakesh q46 went
54% -> 34% -> 0.6% readout error within about 95 minutes while IBM listed ~1%.

Design: R rounds, INTERVAL_S apart. Each round, on each of ibm_marrakesh / ibm_fez / ibm_kingston, it submits
one probe job (all-0 and all-1 on every qubit, PROBE_REPEATS repeats each, 2,048 shots) and snapshots IBM's
calibration. All three jobs in a round are submitted together. Rounds already on disk are skipped
(resumable); a round is not submitted if quota < 1.3 x estimate.

Definitions (fixed here, before any data):
  e_probe(q)  = mean of p(1|0) and p(0|1), pooled over the round's repeats
  e_ibm(q)    = IBM-reported readout_error in the same round's snapshot
  hidden fault: e_probe > 0.15 and e_ibm < 0.03
  recovered:    e_probe < 0.05

Preregistered analyses (analyze step):
  H1 prevalence    hidden faults per device-round (mean, min, max); predicted >= 1 on >= 2 devices in a
                   majority of rounds.
  H2 intermittency among qubits that are a hidden fault in >= 1 round, the fraction that are 'recovered'
                   in >= 1 other round; predicted > 0.5.
  H3 persistence   2x2 over consecutive round pairs (all device-qubits): fault at r vs fault at r+1;
                   Fisher exact odds ratio, reported with p.
  H4 tracking      Spearman rho between e_probe and e_ibm per device-round; median reported.
  Noise floor      within-round repeat-to-repeat |delta e| for flagged qubits, against between-round |delta e|,
                   to show changes exceed sampling noise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from phase3.tesseract_map_03 import per_qubit_readout
from phase3.tesseract_map_04 import SECONDS_PER_PUB, _service

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "phase6"
BACKENDS = ["ibm_marrakesh", "ibm_fez", "ibm_kingston"]
ROUNDS = 8
INTERVAL_S = 15 * 60
PROBE_REPEATS = 3
SHOTS = 2048
FAULT = 0.15
IBM_OK = 0.03
RECOVERED = 0.05


def f(backend: str, r: int, what: str) -> Path:
    return OUT_DIR / f"ro06_{backend}_r{r:02d}_{what}.json"


def _probe_pubs(be):
    from qiskit import QuantumCircuit, transpile
    n = be.num_qubits
    qs = list(range(n))
    pubs = []
    for rep in range(PROBE_REPEATS):
        for bit in (0, 1):
            qc = QuantumCircuit(n, n, name=f"ro06_rep{rep}_all{bit}")
            if bit:
                qc.x(qs)
            qc.measure(qs, qs)
            pubs.append(transpile(qc, backend=be, initial_layout=qs, optimization_level=0))
    return pubs


def run() -> None:
    from qiskit_ibm_runtime import SamplerV2
    from phase3.live_hardware_runner import extract_counts_from_pub_result, snapshot_backend_calibration

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svc = _service()
    bes = {b: svc.backend(b) for b in BACKENDS}
    pubs = {b: _probe_pubs(be) for b, be in bes.items()}
    est = SECONDS_PER_PUB * 2 * PROBE_REPEATS * len(BACKENDS)
    for r in range(ROUNDS):
        if all(f(b, r, "counts").exists() for b in BACKENDS):
            continue
        t0 = time.time()
        rem = float(svc.usage().get("usage_remaining_seconds", 0) or 0)
        if rem < 1.3 * est:
            print(f"round {r}: {rem:.0f} s quota left, need {1.3 * est:.0f}; stopping", flush=True)
            break
        jobs = {}
        for b, be in bes.items():
            if f(b, r, "counts").exists():
                continue
            f(b, r, "calibration").write_text(json.dumps(snapshot_backend_calibration(be), indent=1) + "\n")
            jobs[b] = (SamplerV2(mode=be).run(pubs[b], shots=SHOTS), datetime.now(timezone.utc).isoformat())
        for b, (job, sub) in jobs.items():
            res = job.result()
            ts = {}
            try:
                ts = {k: str(v) for k, v in (job.metrics().get("timestamps") or {}).items()}
            except Exception:
                pass
            usage = None
            try:
                usage = job.usage()
            except Exception:
                pass
            f(b, r, "counts").write_text(json.dumps({
                "backend": b, "round": r, "job_id": job.job_id(), "submitted_utc": sub,
                "completed_utc": datetime.now(timezone.utc).isoformat(), "job_timestamps": ts,
                "job_usage_seconds": usage,
                "pub_counts": [extract_counts_from_pub_result(res[i]) for i in range(len(pubs[b]))]}) + "\n")
            print(f"round {r} {b} {job.job_id()} {usage}s", flush=True)
        if r < ROUNDS - 1:
            time.sleep(max(0.0, INTERVAL_S - (time.time() - t0)))


def _round_errors(b: str, r: int):
    c = json.loads(f(b, r, "counts").read_text())
    cal = json.loads(f(b, r, "calibration").read_text())
    pc = c["pub_counts"]
    n = cal["num_qubits"]
    qs = list(range(n))
    pooled = per_qubit_readout([pc[i] for i in range(0, len(pc), 2)], [pc[i] for i in range(1, len(pc), 2)], qs)
    reps = [per_qubit_readout([pc[2 * k]], [pc[2 * k + 1]], qs) for k in range(len(pc) // 2)]
    ep = {q: 0.5 * sum(pooled[q]) for q in qs}
    er = [{q: 0.5 * sum(rp[q]) for q in qs} for rp in reps]
    ei = {q: cal["qubits"][str(q)]["readout_error"] for q in qs}
    t = (c.get("job_timestamps") or {}).get("finished") or c["completed_utc"]
    return ep, er, ei, t, c["job_id"]


def analyze() -> Dict[str, Any]:
    from scipy.stats import fisher_exact, spearmanr

    per = {}
    for b in BACKENDS:
        rs = [r for r in range(ROUNDS) if f(b, r, "counts").exists()]
        per[b] = {r: _round_errors(b, r) for r in rs}
    hidden_counts, rhos, rows = {}, {}, {}
    ever_hidden, recovered_too = 0, 0
    tab = np.zeros((2, 2), dtype=int)  # [fault at r][fault at r+1]
    within, between = [], []
    for b, d in per.items():
        rs = sorted(d)
        hc = []
        rho = []
        flags = {}
        for r in rs:
            ep, er, ei, t, jid = d[r]
            hid = [q for q in ep if ep[q] > FAULT and ei[q] is not None and ei[q] < IBM_OK]
            hc.append(len(hid))
            flags[r] = {q: ep[q] > FAULT for q in ep}
            ok = [q for q in ep if ei[q] is not None]
            rho.append(float(spearmanr([ep[q] for q in ok], [ei[q] for q in ok]).statistic))
            rows.setdefault(b, []).append({"round": r, "time": t, "job_id": jid, "hidden_faults": {str(q): round(ep[q], 4) for q in hid},
                                           "ibm_of_hidden": {str(q): ei[q] for q in hid}, "spearman": rho[-1]})
            for q in hid:
                vals = [x[q] for x in er]
                within.append(max(vals) - min(vals))
        hidden_counts[b] = hc
        rhos[b] = rho
        hq = {q for r in rs for q in rows[b][rs.index(r)]["hidden_faults"]}
        for q in hq:
            ever_hidden += 1
            if any(d[r][0][int(q)] < RECOVERED for r in rs):
                recovered_too += 1
        for i in range(len(rs) - 1):
            a, c2 = flags[rs[i]], flags[rs[i + 1]]
            for q in a:
                tab[int(a[q]), int(c2[q])] += 1
                if a[q] or c2[q]:
                    between.append(abs(d[rs[i]][0][q] - d[rs[i + 1]][0][q]))
        rows[b] = rows[b]
    odds, p = fisher_exact(tab) if tab.sum() else (float("nan"), float("nan"))
    n_rounds = min(len(v) for v in hidden_counts.values())
    devices_majority = sum(sum(1 for x in hc if x >= 1) > len(hc) / 2 for hc in hidden_counts.values())
    out = {
        "experiment_id": "TESSERACT-RO-06",
        "rounds_completed": {b: len(v) for b, v in hidden_counts.items()},
        "H1_hidden_faults_per_round": {b: {"counts": hc, "mean": float(np.mean(hc)), "min": int(min(hc)), "max": int(max(hc))}
                                       for b, hc in hidden_counts.items()},
        "H1_devices_with_fault_in_majority_of_rounds": devices_majority,
        "H1_supported": devices_majority >= 2,
        "H2_qubits_ever_hidden": ever_hidden, "H2_of_which_recovered_in_another_round": recovered_too,
        "H2_fraction": (recovered_too / ever_hidden) if ever_hidden else None,
        "H2_supported": bool(ever_hidden and recovered_too / ever_hidden > 0.5),
        "H3_transition_table_[fault_r][fault_r+1]": tab.tolist(), "H3_fisher_odds_ratio": float(odds), "H3_fisher_p": float(p),
        "H3_p_fault_next_given_fault": float(tab[1, 1] / tab[1].sum()) if tab[1].sum() else None,
        "H3_p_fault_next_given_ok": float(tab[0, 1] / tab[0].sum()) if tab[0].sum() else None,
        "H4_spearman_median": {b: float(np.median(v)) for b, v in rhos.items()},
        "noise_floor_within_round_range_median": float(np.median(within)) if within else None,
        "between_round_abs_change_median_flagged": float(np.median(between)) if between else None,
        "rounds": rows, "min_rounds_any_device": n_rounds,
    }
    (OUT_DIR / "ro06_analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    return {k: v for k, v in out.items() if k != "rounds"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "analyze"])
    a = ap.parse_args()
    if a.step == "run":
        run()
    else:
        print(json.dumps(analyze(), indent=1, default=str))


if __name__ == "__main__":
    main()
