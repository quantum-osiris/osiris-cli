#!/usr/bin/env python3
"""
stage_and_harvest_qec001.py — Offline staging (`--stage-only`, 0 QPU seconds spent)
and future owner-authorized harvesting for `OSIRIS-QEC-001`: Multi-round repetition-code
logical-error-per-syndrome-cycle scaling across code distances d in {3, 5, 7} and
syndrome cycles R in {1, 3, 5, 7} on `ibm_fez` (`[111,110,109,108,107,106,105,117,125,124,123,122,121]`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from qiskit import qpy, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, thermal_relaxation_error
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit_ibm_runtime.utils.validations import validate_isa_circuits

EXPERIMENT_DIR = Path(__file__).resolve().parent
MOD_PATH = EXPERIMENT_DIR / "qec001_repetition_scaling.py"
_SPEC = importlib.util.spec_from_file_location("qec001_repetition_scaling", MOD_PATH)
assert _SPEC is not None and _SPEC.loader is not None
qec = importlib.util.module_from_spec(_SPEC)
sys.modules["qec001_repetition_scaling"] = qec
_SPEC.loader.exec_module(qec)

BENCHMARK_ID = "OSIRIS-QEC-001"
BACKEND_NAME = "ibm_fez"
N_BLOCKS = 4
SHOTS_PER_CIRCUIT = 2048
SCHEDULE_SEED = 20261118
CHAIN_13Q: Tuple[int, ...] = tuple(qec.DEFAULT_FEZ_CHAIN_13Q)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _load_token() -> str:
    ak_path = Path("/home/osiris/apikey.json")
    if ak_path.is_file():
        data = json.loads(ak_path.read_text(encoding="utf-8"))
        if data.get("apikey"):
            return str(data["apikey"]).strip()
    env_path = Path("/home/osiris/osiris-ide/.env")
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("IBM_QUANTUM_TOKEN="):
                return line.strip().split("=", 1)[1].strip().strip('"').strip("'")
    tok = os.environ.get("IBM_QUANTUM_TOKEN", "").strip()
    if tok:
        return tok
    raise RuntimeError("No IBM Quantum credential found")


def _connect_service() -> QiskitRuntimeService:
    return QiskitRuntimeService(channel="ibm_quantum_platform", token=_load_token(), instance="open-instance")


def build_calibrated_chain_noise_model(tgt: Any, props: Any, subchain: Sequence[int]) -> Tuple[NoiseModel, float, float]:
    """Build a local 0..(2d-2) Aer NoiseModel matching live `ibm_fez` gate, readout, reset, and mid-circuit T1/T2 idle."""
    nm = NoiseModel()
    cz_errs: List[float] = []
    ro_errs: List[float] = []

    # Average ancilla readout + reset duration during which data qubits idle per round
    anc_indices = [i for i in range(1, len(subchain), 2)]
    ro_durs = [float(tgt["measure"][(subchain[i],)].duration or 1.4e-6) for i in anc_indices]
    rst_durs = [float(tgt["reset"][(subchain[i],)].duration or 1.2e-6) for i in anc_indices]
    t_round_idle = float(np.mean(ro_durs) + np.mean(rst_durs))

    for loc_i, pq in enumerate(subchain):
        t1 = max(float(props.t1(pq)), 1e-6)
        t2 = min(max(float(props.t2(pq)), 1e-6), 2.0 * t1)
        sx_err = float(tgt["sx"][(pq,)].error or 3e-4)
        sx_dur = float(tgt["sx"][(pq,)].duration or 36e-9)
        err_1q = depolarizing_error(min(0.2, 2.0 * sx_err), 1).compose(thermal_relaxation_error(t1, t2, sx_dur))
        nm.add_quantum_error(err_1q, ["sx", "x"], [loc_i])

        p10 = float(props.qubit_property(pq, "prob_meas1_prep0")[0])
        p01 = float(props.qubit_property(pq, "prob_meas0_prep1")[0])
        ro_errs.append(0.5 * (p10 + p01))
        nm.add_readout_error(ReadoutError([[1.0 - p10, p10], [p01, 1.0 - p01]]), [loc_i])

        # Reset error + ancilla relaxation; and attach mid-circuit idle thermal relaxation to ancilla measure?
        # Even better: attach half of t_round_idle thermal relaxation to each CNOT layer on data/ancilla wires!
        rst_err = depolarizing_error(min(0.2, max(0.005, p01)), 1)
        nm.add_quantum_error(rst_err, ["reset"], [loc_i])

    for i in range(len(subchain) - 1):
        u, v = subchain[i], subchain[i + 1]
        pair = (u, v) if (u, v) in tgt["cz"] else (v, u)
        cerr = float(tgt["cz"][pair].error or 2.5e-3)
        cdur = float(tgt["cz"][pair].duration or 68e-9)
        cz_errs.append(cerr)
        t1_u = max(float(props.t1(u)), 1e-6)
        t2_u = min(max(float(props.t2(u)), 1e-6), 2.0 * t1_u)
        t1_v = max(float(props.t1(v)), 1e-6)
        t2_v = min(max(float(props.t2(v)), 1e-6), 2.0 * t1_v)
        # Include 0.25 * t_round_idle per CZ to capture data decoherence during mid-circuit readout/reset
        eff_dur = cdur + 0.25 * t_round_idle
        th_uv = thermal_relaxation_error(t1_u, t2_u, eff_dur).expand(thermal_relaxation_error(t1_v, t2_v, eff_dur))
        err_2q = depolarizing_error(min(0.5, 1.25 * cerr), 2).compose(th_uv)
        nm.add_quantum_error(err_2q, ["cz"], [i, i + 1])
        nm.add_quantum_error(err_2q, ["cz"], [i + 1, i])

    p_data_est = float(2.0 * np.mean(cz_errs) + t_round_idle / (2.0 * 120e-6))
    p_meas_est = float(np.mean(ro_errs) + 0.006)
    return nm, p_data_est, p_meas_est


def stage_qec001() -> Dict[str, Any]:
    svc = _connect_service()
    usage = svc.usage()
    backend = svc.backend(BACKEND_NAME)
    status = backend.status()
    props = backend.properties()
    tgt = backend.target

    # Assemble 4 randomized blocks of (2 bases x 3 distances x 4 round counts x 2 logical states = 48 specs/block => 192 circuits)
    rng = np.random.default_rng(SCHEDULE_SEED)
    isa_circuits: List[Any] = []
    metadata: List[Dict[str, Any]] = []

    base_specs = [
        (basis, d, r, state)
        for basis in qec.CODE_BASES
        for d in qec.DISTANCES
        for r in qec.ROUNDS_LIST
        for state in (0, 1)
    ]

    for blk in range(N_BLOCKS):
        order = rng.permutation(len(base_specs))
        for idx_in_blk in order:
            basis, d, r, state = base_specs[int(idx_in_blk)]
            sub = qec.subchain_for_distance(CHAIN_13Q, d)
            raw_qc = qec.build_repetition_memory_circuit(
                distance=d,
                rounds=r,
                code_basis=basis,
                logical_state=state,
                dynamical_decoupling=False,
            )
            raw_qc.name = f"b{blk}_{basis}_d{d}_r{r}_s{state}"
            isa_qc = transpile(
                raw_qc,
                target=tgt,
                initial_layout=list(sub),
                routing_method="none",
                optimization_level=1,
                seed_transpiler=42,
            )
            used_phys = {isa_qc.find_bit(qb).index for inst in isa_qc.data for qb in inst.qubits}
            if not used_phys.issubset(set(sub)):
                raise RuntimeError(f"Layout leak on {isa_qc.name}: {sorted(used_phys - set(sub))}")
            isa_circuits.append(isa_qc)
            metadata.append(
                {
                    "circuit_name": isa_qc.name,
                    "block_index": blk,
                    "code_basis": basis,
                    "distance": d,
                    "rounds": r,
                    "logical_state": state,
                    "pinned_subchain": list(sub),
                    "cz_count": int(isa_qc.count_ops().get("cz", 0)),
                    "mid_measure_count": r * (d - 1),
                    "reset_count": int(isa_qc.count_ops().get("reset", 0)),
                }
            )

    validate_isa_circuits(isa_circuits, tgt)

    # Compute calibration-matched noisy predictions on ibm_fez for R=5 scaling and all (basis, d, R)
    predictions: Dict[str, Any] = {}
    for basis in qec.CODE_BASES:
        predictions[basis] = {}
        for d in qec.DISTANCES:
            sub = qec.subchain_for_distance(CHAIN_13Q, d)
            nm, p_dat, p_meas = build_calibrated_chain_noise_model(tgt, props, sub)
            sim = AerSimulator(noise_model=nm, seed_simulator=42 + d)
            predictions[basis][f"d{d}"] = {}
            for r in qec.ROUNDS_LIST:
                raw0 = qec.build_repetition_memory_circuit(d, r, basis, logical_state=0, dynamical_decoupling=False)
                # Transpile to local basis gates ['sx', 'x', 'rz', 'cz'] without 156Q expansion for fast exact simulation
                loc_qc = transpile(raw0, basis_gates=["sx", "x", "rz", "cz"], optimization_level=1, seed_transpiler=42)
                cnts0 = sim.run(loc_qc, shots=4096).result().get_counts()
                dec = qec.build_mwpm_decoder(d, r, p_data=p_dat, p_meas=p_meas, p_hook=0.35 * p_dat)
                ev = qec.evaluate_repetition_counts(cnts0, d, r, logical_state=0, decoder=dec)
                predictions[basis][f"d{d}"][f"R{r}"] = {
                    "predicted_p_logical_mwpm": round(ev["p_logical_mwpm"], 5),
                    "predicted_p_logical_majority": round(ev["p_logical_majority"], 5),
                    "predicted_epsilon_per_cycle": round(ev["epsilon_per_cycle"], 6),
                    "predicted_defect_prob": round(ev["mean_defect_probability"], 5),
                }

    # Compute predicted suppression factors Lambda_{3->5} and Lambda_{5->7} at R=5 and R=7
    suppression_summary: Dict[str, Any] = {}
    for basis in qec.CODE_BASES:
        suppression_summary[basis] = {}
        for r_key in ("R5", "R7"):
            e3 = max(predictions[basis]["d3"][r_key]["predicted_epsilon_per_cycle"], 1e-6)
            e5 = max(predictions[basis]["d5"][r_key]["predicted_epsilon_per_cycle"], 1e-6)
            e7 = max(predictions[basis]["d7"][r_key]["predicted_epsilon_per_cycle"], 1e-6)
            suppression_summary[basis][r_key] = {
                "eps_d3": round(e3, 6),
                "eps_d5": round(e5, 6),
                "eps_d7": round(e7, 6),
                "lambda_3_to_5": round(e3 / e5, 3),
                "lambda_5_to_7": round(e5 / e7, 3),
            }

    qpy_path = EXPERIMENT_DIR / "qec001_fez_staged_circuits.qpy"
    with open(qpy_path, "wb") as fh:
        qpy.dump(isa_circuits, fh)
    qpy_sha256 = hashlib.sha256(qpy_path.read_bytes()).hexdigest()

    chain_edges = {}
    for i in range(12):
        u, v = CHAIN_13Q[i], CHAIN_13Q[i + 1]
        pair = (u, v) if (u, v) in tgt["cz"] else (v, u)
        chain_edges[f"{min(u, v)}-{max(u, v)}"] = {
            "cz_error": float(tgt["cz"][pair].error),
            "duration_s": float(tgt["cz"][pair].duration),
        }

    manifest = {
        "benchmark_id": BENCHMARK_ID,
        "staged_at_utc": _utc_now(),
        "backend": backend.name,
        "operational": bool(status.operational),
        "pending_jobs_at_staging": int(status.pending_jobs),
        "usage_remaining_seconds_at_staging": usage.get("usage_remaining_seconds"),
        "usage_remaining_ru_at_staging": usage.get("usage_remaining_ru"),
        "pinned_13q_chain": list(CHAIN_13Q),
        "subchains_by_distance": {
            f"d{d}": list(qec.subchain_for_distance(CHAIN_13Q, d)) for d in qec.DISTANCES
        },
        "chain_edge_calibration": chain_edges,
        "n_blocks": N_BLOCKS,
        "shots_per_circuit": SHOTS_PER_CIRCUIT,
        "total_circuits": len(isa_circuits),
        "total_shots": len(isa_circuits) * SHOTS_PER_CIRCUIT,
        "estimated_qpu_seconds": 92,
        "schedule_seed": SCHEDULE_SEED,
        "qpy_sha256": qpy_sha256,
        "calibrated_noisy_predictions": predictions,
        "predicted_suppression_factors": suppression_summary,
        "circuit_metadata": metadata,
    }
    (EXPERIMENT_DIR / "qec001_fez_staging_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"[STAGED OFFLINE] {BENCHMARK_ID} on {backend.name} | {len(isa_circuits)} ISA circuits x "
        f"{SHOTS_PER_CIRCUIT} shots = {len(isa_circuits) * SHOTS_PER_CIRCUIT:,} shots | "
        f"QPY SHA-256: {qpy_sha256[:16]}... | 0 QPU seconds spent"
    )
    for basis, rmap in suppression_summary.items():
        print(f"  [{basis}] R=5: {rmap['R5']} | R=7: {rmap['R7']}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-only", action="store_true", default=True)
    parser.parse_args()
    stage_qec001()


if __name__ == "__main__":
    main()
