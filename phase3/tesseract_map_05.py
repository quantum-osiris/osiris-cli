"""TESSERACT-MAP-05: keep the calibration ranking, exclude only grossly faulty readout qubits.

Follow-up to TESSERACT-MAP-04 (doi:10.5281/zenodo.24407001): ranking paths by probe readout helped on
one of three devices, and after mitigation the calibration-ranked path was better on all three.
Hypothesis: score paths with IBM's calibration as before, but drop any qubit whose probe readout
error exceeds EXCLUDE_THRESHOLD before the search.

Per backend: a probe job (all-0/all-1 on every qubit, 3 repeats, 2,048 shots), then a main job:
  C1_CAL_BEST     best path, IBM readout (MAP-02 rule)                [= MAP-04 B2]
  C2_CAL_EXCL     best path, IBM readout, probe-faulty qubits removed  [new]
  C3_PROBE_BEST   best path, probe readout                           [= MAP-04 B1]
  C4_STOCK        Qiskit stock level-3 layout
  U0 / U1         union all-0 / all-1
10 shuffled blocks x 6 PUBs x 2,048 shots. The chain circuit is MAP-02's (39 CZ, theta 51.7 deg,
optimisation level 3).

Commands: probe | lock | submit | analyze  <backend>.
Credentials: JSON file named by OSIRIS_IBM_KEY_FILE.
"""

from __future__ import annotations

import argparse
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from phase2.tesseract_map_02.tesseract_map_02 import (
    THETA_DEFAULT_DEG,
    build_chain_circuit,
    chain_logical_to_physical_layout,
    extract_final_physical_layout,
    score_tesseract_counts,
)
from phase3.tesseract_map_03 import evaluate_contrasts, mitigated_fcl, per_qubit_readout
from phase3.tesseract_map_04 import (
    N_BLOCKS, N_QUBITS, PROBE_REPEATS, SECONDS_PER_PUB, SHOTS, _run_job, _service, _sha, score_paths,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "phase5"
SEED = 20261122
EXCLUDE_THRESHOLD = 0.15  # probe mean assignment error above which a qubit is excluded from C2
TEST_ARMS = ["C1_CAL_BEST", "C2_CAL_EXCL", "C3_PROBE_BEST", "C4_STOCK"]
SPAM_ARMS = ["U0_UNION_ALL0", "U1_UNION_ALL1"]
CONTRASTS_05: List[Tuple[str, List[Tuple[str, str, float]]]] = [
    ("M1_exclusion_vs_calibration", [("raw", "C2_CAL_EXCL", 1.0), ("raw", "C1_CAL_BEST", -1.0)]),
    ("M2_exclusion_vs_probe_ranking", [("raw", "C2_CAL_EXCL", 1.0), ("raw", "C3_PROBE_BEST", -1.0)]),
    ("M3_exclusion_vs_stock", [("raw", "C2_CAL_EXCL", 1.0), ("raw", "C4_STOCK", -1.0)]),
    ("M4_probe_vs_calibration_replication", [("raw", "C3_PROBE_BEST", 1.0), ("raw", "C1_CAL_BEST", -1.0)]),
]


def p(backend: str, what: str) -> Path:
    return OUT_DIR / f"tesseract_map_05_{backend}_{what}"


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
                   p(backend_name, "probe_raw_counts.json"), {"stage": "probe", "experiment": "TESSERACT-MAP-05"})
    ro = per_qubit_readout([c for i, c in enumerate(raw["pub_counts"]) if i % 2 == 0],
                           [c for i, c in enumerate(raw["pub_counts"]) if i % 2 == 1], qs)
    summ = {str(q): round(0.5 * (a + b), 5) for q, (a, b) in ro.items()}
    p(backend_name, "probe_readout.json").write_text(json.dumps({"job_id": raw["job_id"], "mean_assign_err": summ,
                                                                 "p10_p01": {str(q): v for q, v in ro.items()}}, indent=1) + "\n")
    return {"job_id": raw["job_id"], "usage_s": raw["job_usage_seconds"],
            "over_threshold": sorted([(q, v) for q, v in summ.items() if v > EXCLUDE_THRESHOLD], key=lambda t: -t[1])}


def lock(backend_name: str) -> Dict[str, Any]:
    from qiskit import QuantumCircuit, qpy, transpile
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_aer import AerSimulator
    from phase3.live_hardware_runner import verify_isa_circuit

    lp, qp = p(backend_name, "live_prereg_lock.json"), p(backend_name, "live_pubs.qpy")
    if lp.exists():
        raise SystemExit("lock exists; refusing to overwrite")
    svc = _service()
    be = svc.backend(backend_name)
    n = be.num_qubits
    cal = json.loads(p(backend_name, "probe_time_calibration.json").read_text())
    probe_ro = {int(q): float(v) for q, v in json.loads(p(backend_name, "probe_readout.json").read_text())["mean_assign_err"].items()}
    edges = [tuple(e) for e in be.coupling_map.get_edges()]
    ibm_ro = {q: (cal["qubits"][str(q)]["readout_error"] if cal["qubits"][str(q)]["readout_error"] is not None else 0.015) for q in range(n)}
    excluded = sorted(q for q in range(n) if probe_ro[q] > EXCLUDE_THRESHOLD)
    excl_ro = {q: (0.5 if q in excluded else ibm_ro[q]) for q in range(n)}  # >= 0.45 -> removed by score_paths
    cal_best = score_paths(cal, ibm_ro, n, edges)[0]
    excl_best = score_paths(cal, excl_ro, n, edges)[0]
    probe_best = score_paths(cal, probe_ro, n, edges)[0]

    def pinned(path, nm):
        pm = generate_preset_pass_manager(optimization_level=3, backend=be, initial_layout=chain_logical_to_physical_layout(path),
                                          layout_method="trivial", routing_method="none", seed_transpiler=SEED)
        return pm.run(build_chain_circuit(N_QUBITS, THETA_DEFAULT_DEG, name=nm))

    arms = {"C1_CAL_BEST": pinned(cal_best[1], "C1"), "C2_CAL_EXCL": pinned(excl_best[1], "C2"),
            "C3_PROBE_BEST": pinned(probe_best[1], "C3"),
            "C4_STOCK": transpile(build_chain_circuit(N_QUBITS, THETA_DEFAULT_DEG, name="C4"), backend=be,
                                  optimization_level=3, seed_transpiler=SEED)}
    man = {}
    for a, t in arms.items():
        verify_isa_circuit(t, be)
        lay = extract_final_physical_layout(t, N_QUBITS)
        man[a] = {"twoq_gate_count": sum(1 for i in t.data if i.operation.num_qubits == 2), "circuit_depth": t.depth(),
                  "final_physical_layout": lay,
                  "probe_readout_max": float(max(probe_ro[q] for q in lay)),
                  "ibm_readout_max": float(max(ibm_ro[q] for q in lay)),
                  "n_excluded_qubits_in_layout": sum(q in excluded for q in lay)}
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
            pubs.append(arms[a].copy(name=f"tmap05_b{b:02d}_p{pos}_{a}"))
    buf = io.BytesIO()
    qpy.dump(pubs, buf)
    qp.write_bytes(buf.getvalue())
    sim = AerSimulator(method="matrix_product_state")
    check = {}
    for r, qc in zip(sched, pubs):
        if r["block"] == 0 and r["arm_id"] in TEST_ARMS:
            c = {kk.replace(" ", ""): v for kk, v in sim.run(qc, shots=1024, seed_simulator=SEED).result().get_counts().items()}
            check[r["arm_id"]] = score_tesseract_counts(c, N_QUBITS, THETA_DEFAULT_DEG)["classical_fidelity"]
    assert all(v > 0.95 for v in check.values()), check
    same = sorted(cal_best[1]) == sorted(excl_best[1])
    L = {"experiment_id": "TESSERACT-MAP-05", "backend_name": backend_name, "locked_utc": datetime.now(timezone.utc).isoformat(),
         "probe_job_id": json.loads(p(backend_name, "probe_receipt.json").read_text())["job_id"],
         "probe_readout_sha256": _sha(p(backend_name, "probe_readout.json")),
         "probe_time_calibration_sha256": _sha(p(backend_name, "probe_time_calibration.json")),
         "exclude_threshold": EXCLUDE_THRESHOLD, "excluded_qubits": excluded,
         "c1_c2_identical_qubits": same,
         "n_blocks": N_BLOCKS, "shots_per_pub": SHOTS, "total_pubs": len(pubs),
         "estimated_qpu_seconds": round(SECONDS_PER_PUB * len(pubs), 1), "qpy_sha256": _sha(qp),
         "paths": {"calibration_best": cal_best[1], "exclusion_best": excl_best[1], "probe_best": probe_best[1]},
         "union_active_physical_qubits": union, "arm_manifests": man, "noiseless_fcl_block0": check,
         "code_sha256": _sha(Path(__file__)), "pub_schedule": sched}
    lp.write_text(json.dumps(L, indent=1) + "\n")
    return {"excluded": excluded, "c1_c2_identical": same, "est_s": L["estimated_qpu_seconds"],
            "arms_probe_max_ro": {a: round(m["probe_readout_max"], 3) for a, m in man.items()}}


def submit(backend_name: str) -> Dict[str, Any]:
    from qiskit import qpy
    lp, qp = p(backend_name, "live_prereg_lock.json"), p(backend_name, "live_pubs.qpy")
    L = json.loads(lp.read_text())
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
    sched, union = L["pub_schedule"], L["union_active_physical_qubits"]
    ro = per_qubit_readout([c for r, c in zip(sched, raw["pub_counts"]) if r["arm_id"] == "U0_UNION_ALL0"],
                           [c for r, c in zip(sched, raw["pub_counts"]) if r["arm_id"] == "U1_UNION_ALL1"], union)
    rawf = {a: [0.0] * N_BLOCKS for a in TEST_ARMS}
    mitf = {a: [0.0] * N_BLOCKS for a in TEST_ARMS}
    for r, c in zip(sched, raw["pub_counts"]):
        if r["arm_id"] in TEST_ARMS:
            rawf[r["arm_id"]][r["block"]] = score_tesseract_counts(c, N_QUBITS, THETA_DEFAULT_DEG)["classical_fidelity"]
            mitf[r["arm_id"]][r["block"]] = mitigated_fcl(c, L["arm_manifests"][r["arm_id"]]["final_physical_layout"], ro)
    saved = m3.CONTRASTS
    m3.CONTRASTS = CONTRASTS_05
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
                   "worst_qubit": int(lay[int(np.argmax(errs))]), "block_fcl_raw": rawf[a], "block_fcl_mit": mitf[a]}
    out = {"experiment_id": "TESSERACT-MAP-05", "backend": backend_name, "job_id": raw["job_id"],
           "job_usage_seconds": raw["job_usage_seconds"], "prereg_lock_sha256": _sha(lp),
           "c1_c2_identical_qubits": L["c1_c2_identical_qubits"], "excluded_qubits": L["excluded_qubits"],
           "arms": arms, "contrasts": con, "per_qubit_readout": {str(q): v for q, v in ro.items()}}
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
