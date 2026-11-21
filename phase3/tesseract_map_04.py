"""TESSERACT-MAP-04: does a per-job readout probe pick better qubits than published calibration?

Follow-up to TESSERACT-MAP-03 (doi:10.5281/zenodo.24406525), where a qubit on the calibration-ranked
best path read |0> as 1 in 54% of shots while IBM's calibration listed ~1%.

Per backend (ibm_marrakesh primary; ibm_fez, ibm_kingston replications), two jobs:
  1. PROBE job: union |0>^156 and |1>^156 readout circuits, 3 repeats each, 2,048 shots (6 PUBs).
  2. MAIN job, built only from the probe result + the lock-time calibration by the fixed rule below:
       B1_PROBE_BEST   chain on the best 40-qubit path, scored with PROBE readout + IBM CZ errors
       B2_CAL_BEST     chain on the best path scored with IBM calibration only (the MAP-02 rule)
       B3_STOCK        chain, Qiskit stock level-3 layout
       B4_PROBE_MED    chain on the median path under the probe scoring
       U0 / U1         union readout calibration on every qubit any arm uses
     10 shuffled blocks x 6 PUBs x 2,048 shots.
All chain arms: 39 CZ, optimisation level 3, MAP-02 logical circuit (theta 51.7 deg).

Path scoring = MAP-02's `score_simple_paths_on_backend` rule (greedy DFS, 400 candidate paths,
cost = -sum log(1-e_cz) - sum log(1-e_ro)); only the readout term's source differs between B1 and B2.

Commands (backend name as argument):
  python -m phase3.tesseract_map_04 probe   <backend>   # submit probe job, save counts
  python -m phase3.tesseract_map_04 lock    <backend>   # select paths, compile, lock, noiseless check
  python -m phase3.tesseract_map_04 submit  <backend>   # guarded main job
  python -m phase3.tesseract_map_04 analyze <backend>
Credentials: JSON file named by OSIRIS_IBM_KEY_FILE ("apikey" field); never printed.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from phase2.tesseract_map_02.tesseract_map_02 import (
    THETA_DEFAULT_DEG,
    build_chain_circuit,
    chain_logical_to_physical_layout,
    extract_final_physical_layout,
    score_tesseract_counts,
)
from phase3.tesseract_map_03 import evaluate_contrasts, mitigated_fcl, per_qubit_readout

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "phase4"
N_QUBITS = 40
N_BLOCKS = 10
SHOTS = 2048
PROBE_REPEATS = 3
SEED = 20261121
SECONDS_PER_PUB = 53.0 / 90.0 * 1.05
QUOTA_SAFETY_FACTOR = 1.3
FAULTY_READOUT = 0.45  # same exclusion threshold as MAP-02

TEST_ARMS = ["B1_PROBE_BEST", "B2_CAL_BEST", "B3_STOCK", "B4_PROBE_MED"]
SPAM_ARMS = ["U0_UNION_ALL0", "U1_UNION_ALL1"]

# Preregistered contrasts (pp of F_cl), Holm across the four, per backend.
CONTRASTS_04: List[Tuple[str, List[Tuple[str, str, float]]]] = [
    ("L1_probe_vs_calibration_path", [("raw", "B1_PROBE_BEST", 1.0), ("raw", "B2_CAL_BEST", -1.0)]),
    ("L2_probe_vs_stock_layout", [("raw", "B1_PROBE_BEST", 1.0), ("raw", "B3_STOCK", -1.0)]),
    ("L3_probe_best_vs_probe_median", [("raw", "B1_PROBE_BEST", 1.0), ("raw", "B4_PROBE_MED", -1.0)]),
    ("L4_L1_after_readout_mitigation", [("mit", "B1_PROBE_BEST", 1.0), ("mit", "B2_CAL_BEST", -1.0)]),
]


def p(backend: str, what: str) -> Path:
    return OUT_DIR / f"tesseract_map_04_{backend}_{what}"


def _service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    kf = os.environ.get("OSIRIS_IBM_KEY_FILE")
    if not kf:
        raise SystemExit("set OSIRIS_IBM_KEY_FILE")
    tok = json.loads(Path(kf).expanduser().read_text())["apikey"]
    return QiskitRuntimeService(channel="ibm_quantum_platform", token=tok)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _remaining(svc) -> float:
    return float(svc.usage().get("usage_remaining_seconds", 0) or 0)


def _run_job(svc, backend, pubs, shots: int, est_s: float, receipt: Path, raw: Path, extra: Mapping[str, Any]) -> Dict[str, Any]:
    from qiskit_ibm_runtime import SamplerV2
    from phase3.execute_and_analyze_live import _job_usage_seconds, _public_usage
    from phase3.live_hardware_runner import extract_counts_from_pub_result, snapshot_backend_calibration

    if raw.exists():
        return json.loads(raw.read_text())
    if receipt.exists():
        job = svc.job(json.loads(receipt.read_text())["job_id"])
    else:
        rem = _remaining(svc)
        if rem < QUOTA_SAFETY_FACTOR * est_s:
            raise SystemExit(f"{rem:.0f} s quota left, need {QUOTA_SAFETY_FACTOR * est_s:.0f} s; not submitting")
        job = SamplerV2(mode=backend).run(pubs, shots=shots)
        receipt.write_text(json.dumps({"job_id": job.job_id(), "backend": backend.name,
                                       "submitted_utc": datetime.now(timezone.utc).isoformat(),
                                       "quota_remaining_before_s": rem, **extra}, indent=2) + "\n")
        print("submitted", job.job_id(), flush=True)
    res = job.result()
    out = {"job_id": job.job_id(), "completed_utc": datetime.now(timezone.utc).isoformat(),
           "job_usage_seconds": _job_usage_seconds(job), "post_job_usage": _public_usage(svc.usage()),
           "execution_time_calibration": snapshot_backend_calibration(backend),
           "pub_counts": [extract_counts_from_pub_result(res[i]) for i in range(len(pubs))]}
    raw.write_text(json.dumps(out) + "\n")
    return out


# ---------------------------------------------------------------------------
# Stage 1: probe
# ---------------------------------------------------------------------------

def probe(backend_name: str) -> Dict[str, Any]:
    from qiskit import QuantumCircuit, transpile
    from phase3.live_hardware_runner import snapshot_backend_calibration

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svc = _service()
    be = svc.backend(backend_name)
    n = be.num_qubits
    cal_path = p(backend_name, "probe_time_calibration.json")
    if not cal_path.exists():
        cal_path.write_text(json.dumps(snapshot_backend_calibration(be), indent=1) + "\n")
    qs = list(range(n))
    pubs = []
    for r in range(PROBE_REPEATS):
        for bit in (0, 1):
            qc = QuantumCircuit(n, n, name=f"probe_r{r}_all{bit}")
            if bit:
                qc.x(qs)
            qc.measure(qs, qs)
            pubs.append(transpile(qc, backend=be, initial_layout=qs, optimization_level=0))
    raw = _run_job(svc, be, pubs, SHOTS, SECONDS_PER_PUB * len(pubs), p(backend_name, "probe_receipt.json"),
                   p(backend_name, "probe_raw_counts.json"), {"stage": "probe"})
    s0 = [c for i, c in enumerate(raw["pub_counts"]) if i % 2 == 0]
    s1 = [c for i, c in enumerate(raw["pub_counts"]) if i % 2 == 1]
    ro = per_qubit_readout(s0, s1, qs)
    summ = {str(q): round(0.5 * (a + b), 5) for q, (a, b) in ro.items()}
    p(backend_name, "probe_readout.json").write_text(json.dumps({"job_id": raw["job_id"], "mean_assign_err": summ,
                                                                 "p10_p01": {str(q): v for q, v in ro.items()}}, indent=1) + "\n")
    return {"job_id": raw["job_id"], "usage_s": raw["job_usage_seconds"],
            "worst": sorted(summ.items(), key=lambda kv: -kv[1])[:8]}


# ---------------------------------------------------------------------------
# Stage 2: selection + lock
# ---------------------------------------------------------------------------

def score_paths(cal: Mapping[str, Any], ro_err: Mapping[int, float], n_q: int, coupling_edges: Sequence[Tuple[int, int]],
                path_length: int = N_QUBITS, max_paths: int = 400) -> List[Tuple[float, List[int]]]:
    """MAP-02's path rule, with the readout term supplied explicitly (copied logic; see module docstring)."""
    ro = {q: min(max(float(ro_err[q]), 1e-6), 0.49) for q in range(n_q)}
    faulty = {q for q in range(n_q) if float(ro_err[q]) >= FAULTY_READOUT}
    adj: Dict[int, List[int]] = collections.defaultdict(list)
    edge_err: Dict[Tuple[int, int], float] = {}
    for u, v in coupling_edges:
        u, v = int(u), int(v)
        if u in faulty or v in faulty:
            continue
        e = cal["cz_errors"].get(f"{min(u, v)}-{max(u, v)}", 0.005)
        if e is None or e >= 0.45:
            continue
        if v not in adj[u]:
            adj[u].append(v)
        if u not in adj[v]:
            adj[v].append(u)
        edge_err[(min(u, v), max(u, v))] = min(max(float(e), 1e-6), 0.49)
    for q in adj:
        adj[q].sort(key=lambda nb: edge_err[(min(q, nb), max(q, nb))] + 0.5 * ro[nb])
    found: List[Tuple[float, List[int]]] = []
    seen = set()
    for start in [q for q in sorted(range(n_q), key=lambda q: ro[q]) if q not in faulty]:
        stack = [(start, [start], {start})]
        br = 0
        while stack and len(found) < max_paths and br < 150:
            cur, path, vis = stack.pop()
            if len(path) == path_length:
                canon = tuple(path) if path[0] < path[-1] else tuple(reversed(path))
                if canon not in seen:
                    seen.add(canon)
                    c = sum(-math.log(1 - edge_err[(min(path[k], path[k + 1]), max(path[k], path[k + 1]))]) for k in range(path_length - 1))
                    c += sum(-math.log(1 - ro[q]) for q in path)
                    found.append((c, list(path)))
                br += 1
                continue
            for nb in reversed(adj[cur]):
                if nb not in vis:
                    stack.append((nb, path + [nb], vis | {nb}))
    if not found:
        raise RuntimeError("no path found")
    found.sort(key=lambda t: t[0])
    return found


def lock(backend_name: str) -> Dict[str, Any]:
    from qiskit import QuantumCircuit, qpy, transpile
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_aer import AerSimulator
    from phase3.live_hardware_runner import verify_isa_circuit

    lp = p(backend_name, "live_prereg_lock.json")
    qp = p(backend_name, "live_pubs.qpy")
    if lp.exists():
        raise SystemExit("lock exists; refusing to overwrite")
    svc = _service()
    be = svc.backend(backend_name)
    n = be.num_qubits
    cal = json.loads(p(backend_name, "probe_time_calibration.json").read_text())
    probe_ro = json.loads(p(backend_name, "probe_readout.json").read_text())["mean_assign_err"]
    edges = [tuple(e) for e in be.coupling_map.get_edges()]
    ibm_ro = {q: (cal["qubits"][str(q)]["readout_error"] if cal["qubits"][str(q)]["readout_error"] is not None else 0.015) for q in range(n)}
    pr_ro = {q: float(probe_ro[str(q)]) for q in range(n)}
    pr_paths = score_paths(cal, pr_ro, n, edges)
    cal_paths = score_paths(cal, ibm_ro, n, edges)
    probe_best, probe_med, cal_best = pr_paths[0][1], pr_paths[len(pr_paths) // 2][1], cal_paths[0][1]

    def chain(nm):
        return build_chain_circuit(N_QUBITS, THETA_DEFAULT_DEG, name=nm)

    def pinned(path, nm):
        pm = generate_preset_pass_manager(optimization_level=3, backend=be, initial_layout=chain_logical_to_physical_layout(path),
                                          layout_method="trivial", routing_method="none", seed_transpiler=SEED)
        return pm.run(chain(nm))

    arms = {"B1_PROBE_BEST": pinned(probe_best, "B1"), "B2_CAL_BEST": pinned(cal_best, "B2"),
            "B3_STOCK": transpile(chain("B3"), backend=be, optimization_level=3, seed_transpiler=SEED),
            "B4_PROBE_MED": pinned(probe_med, "B4")}
    man = {}
    for a, t in arms.items():
        verify_isa_circuit(t, be)
        lay = extract_final_physical_layout(t, N_QUBITS)
        man[a] = {"twoq_gate_count": sum(1 for i in t.data if i.operation.num_qubits == 2), "circuit_depth": t.depth(),
                  "final_physical_layout": lay,
                  "probe_readout_mean": float(np.mean([pr_ro[q] for q in lay])), "probe_readout_max": float(max(pr_ro[q] for q in lay)),
                  "ibm_readout_mean": float(np.mean([ibm_ro[q] for q in lay])), "ibm_readout_max": float(max(ibm_ro[q] for q in lay))}
        assert man[a]["twoq_gate_count"] == 39, (a, man[a]["twoq_gate_count"])
    union = sorted({q for m in man.values() for q in m["final_physical_layout"]})
    k = len(union)
    u0 = QuantumCircuit(k, k, name="U0")
    u0.measure(range(k), range(k))
    u1 = QuantumCircuit(k, k, name="U1")
    u1.x(range(k))
    u1.measure(range(k), range(k))
    arms["U0_UNION_ALL0"] = transpile(u0, backend=be, initial_layout=union, optimization_level=0)
    arms["U1_UNION_ALL1"] = transpile(u1, backend=be, initial_layout=union, optimization_level=1)
    rng = np.random.default_rng(SEED)
    pubs, sched = [], []
    for b in range(N_BLOCKS):
        order = TEST_ARMS + SPAM_ARMS
        rng.shuffle(order)
        for pos, a in enumerate(order):
            sched.append({"pub_idx": len(pubs), "block": b, "position_in_block": pos, "arm_id": a})
            pubs.append(arms[a].copy(name=f"tmap04_b{b:02d}_p{pos}_{a}"))
    buf = io.BytesIO()
    qpy.dump(pubs, buf)
    qp.write_bytes(buf.getvalue())
    # noiseless check of block 0
    sim = AerSimulator(method="matrix_product_state")
    check = {}
    for r, qc in zip(sched, pubs):
        if r["block"] == 0 and r["arm_id"] in TEST_ARMS:
            c = {kk.replace(" ", ""): v for kk, v in sim.run(qc, shots=1024, seed_simulator=SEED).result().get_counts().items()}
            check[r["arm_id"]] = score_tesseract_counts(c, N_QUBITS, THETA_DEFAULT_DEG)["classical_fidelity"]
    assert all(v > 0.95 for v in check.values()), check
    L = {"experiment_id": "TESSERACT-MAP-04", "backend_name": backend_name,
         "locked_utc": datetime.now(timezone.utc).isoformat(),
         "probe_job_id": json.loads(p(backend_name, "probe_receipt.json").read_text())["job_id"],
         "probe_readout_sha256": _sha(p(backend_name, "probe_readout.json")),
         "probe_time_calibration_sha256": _sha(p(backend_name, "probe_time_calibration.json")),
         "n_blocks": N_BLOCKS, "shots_per_pub": SHOTS, "total_pubs": len(pubs),
         "estimated_qpu_seconds": round(SECONDS_PER_PUB * len(pubs), 1), "qpy_sha256": _sha(qp),
         "paths": {"probe_best": probe_best, "probe_median": probe_med, "calibration_best": cal_best,
                   "probe_best_cost": pr_paths[0][0], "probe_median_cost": pr_paths[len(pr_paths) // 2][0],
                   "calibration_best_cost": cal_paths[0][0], "n_probe_paths": len(pr_paths), "n_cal_paths": len(cal_paths),
                   "probe_and_calibration_best_identical": sorted(probe_best) == sorted(cal_best)},
         "union_active_physical_qubits": union, "arm_manifests": man, "noiseless_fcl_block0": check,
         "code_sha256": _sha(Path(__file__)), "pub_schedule": sched}
    lp.write_text(json.dumps(L, indent=1) + "\n")
    return {k: L[k] for k in ("estimated_qpu_seconds", "qpy_sha256", "noiseless_fcl_block0")} | {
        "identical_paths": L["paths"]["probe_and_calibration_best_identical"],
        "arms": {a: (round(m["probe_readout_max"], 3), round(m["ibm_readout_max"], 3)) for a, m in man.items()}}


def submit(backend_name: str) -> Dict[str, Any]:
    from qiskit import qpy
    lp = p(backend_name, "live_prereg_lock.json")
    L = json.loads(lp.read_text())
    qp = p(backend_name, "live_pubs.qpy")
    if _sha(qp) != L["qpy_sha256"]:
        raise SystemExit("QPY does not match lock")
    svc = _service()
    be = svc.backend(backend_name)
    with qp.open("rb") as f:
        pubs = qpy.load(f)
    raw = _run_job(svc, be, pubs, SHOTS, L["estimated_qpu_seconds"], p(backend_name, "submission_receipt.json"),
                   p(backend_name, "live_raw_counts.json"), {"prereg_lock_sha256": _sha(lp), "qpy_sha256": L["qpy_sha256"]})
    return {"job_id": raw["job_id"], "usage_s": raw["job_usage_seconds"]}


def analyze(backend_name: str) -> Dict[str, Any]:
    import phase3.tesseract_map_03 as m3
    lp = p(backend_name, "live_prereg_lock.json")
    L = json.loads(lp.read_text())
    raw = json.loads(p(backend_name, "live_raw_counts.json").read_text())
    union = L["union_active_physical_qubits"]
    sched = L["pub_schedule"]
    ro = per_qubit_readout([c for r, c in zip(sched, raw["pub_counts"]) if r["arm_id"] == "U0_UNION_ALL0"],
                           [c for r, c in zip(sched, raw["pub_counts"]) if r["arm_id"] == "U1_UNION_ALL1"], union)
    rawf = {a: [0.0] * N_BLOCKS for a in TEST_ARMS}
    mitf = {a: [0.0] * N_BLOCKS for a in TEST_ARMS}
    for r, c in zip(sched, raw["pub_counts"]):
        if r["arm_id"] in TEST_ARMS:
            rawf[r["arm_id"]][r["block"]] = score_tesseract_counts(c, N_QUBITS, THETA_DEFAULT_DEG)["classical_fidelity"]
            mitf[r["arm_id"]][r["block"]] = mitigated_fcl(c, L["arm_manifests"][r["arm_id"]]["final_physical_layout"], ro)
    saved = m3.CONTRASTS
    m3.CONTRASTS = CONTRASTS_04
    try:
        con = evaluate_contrasts(rawf, mitf)
    finally:
        m3.CONTRASTS = saved
    arms = {}
    for a in TEST_ARMS:
        lay = L["arm_manifests"][a]["final_physical_layout"]
        errs = [0.5 * sum(ro[q]) for q in lay]
        arms[a] = {"mean_fcl_raw": float(np.mean(rawf[a])), "sd_fcl_raw": float(np.std(rawf[a], ddof=1)),
                   "mean_fcl_mit": float(np.mean(mitf[a])), "max_readout_in_job": float(max(errs)),
                   "worst_qubit": int(lay[int(np.argmax(errs))]),
                   "block_fcl_raw": rawf[a], "block_fcl_mit": mitf[a]}
    out = {"experiment_id": "TESSERACT-MAP-04", "backend": backend_name, "job_id": raw["job_id"],
           "job_usage_seconds": raw["job_usage_seconds"], "prereg_lock_sha256": _sha(lp),
           "probe_and_calibration_best_identical": L["paths"]["probe_and_calibration_best_identical"],
           "arms": arms, "contrasts": con,
           "per_qubit_readout": {str(q): v for q, v in ro.items()}}
    p(backend_name, "live_analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    return {"arms": {a: (round(v["mean_fcl_raw"], 4), round(v["mean_fcl_mit"], 3), v["worst_qubit"], round(v["max_readout_in_job"], 3)) for a, v in arms.items()},
            "contrasts": {k: (round(v["mean_pp"], 2), [round(x, 2) for x in v["ci95_pp"]], v["p_holm"]) for k, v in con.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["probe", "lock", "submit", "analyze"])
    ap.add_argument("backend")
    a = ap.parse_args()
    print(json.dumps({"probe": probe, "lock": lock, "submit": submit, "analyze": analyze}[a.step](a.backend), indent=1, default=str))


if __name__ == "__main__":
    main()
