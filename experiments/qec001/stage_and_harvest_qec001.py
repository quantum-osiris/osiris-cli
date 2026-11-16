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
import gzip
import hashlib
import importlib.util
import json
import math
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


def _extract_counts(pub_result: Any) -> Dict[str, int]:
    data_bin = pub_result.data
    if hasattr(data_bin, "c"):
        return {str(k): int(v) for k, v in data_bin.c.get_counts().items()}
    for field in dir(data_bin):
        if not field.startswith("_"):
            obj = getattr(data_bin, field)
            if hasattr(obj, "get_counts"):
                return {str(k): int(v) for k, v in obj.get_counts().items()}
    raise RuntimeError("Could not locate BitArray with get_counts() in SamplerV2 PubResult")


def submit_frozen_qec001() -> str:
    """Verify frozen QPY SHA-256 against pre-registered manifest, snapshot live calibration, and dispatch 1 SamplerV2 job."""
    qpy_path = EXPERIMENT_DIR / "qec001_fez_staged_circuits.qpy"
    manifest_path = EXPERIMENT_DIR / "qec001_fez_staging_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))

    actual_qpy_sha256 = hashlib.sha256(qpy_path.read_bytes()).hexdigest()
    if actual_qpy_sha256 != manifest["qpy_sha256"]:
        raise RuntimeError(
            f"Pre-flight abort: frozen QPY SHA-256 mismatch ({actual_qpy_sha256} != {manifest['qpy_sha256']})"
        )
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    with open(qpy_path, "rb") as fh:
        isa_circuits = qpy.load(fh)
    if len(isa_circuits) != int(manifest["total_circuits"]):
        raise RuntimeError(
            f"Pre-flight abort: circuit count {len(isa_circuits)} != {manifest['total_circuits']}"
        )

    svc = _connect_service()
    usage_before = svc.usage()
    rem_sec = float(usage_before.get("usage_remaining_seconds", 0))
    if rem_sec < 180:
        raise RuntimeError(f"Pre-flight abort: usage_remaining_seconds={rem_sec} < 180s minimum")

    backend = svc.backend(manifest["backend"])
    status = backend.status()
    if not status.operational:
        raise RuntimeError(f"Pre-flight abort: {backend.name} operational={status.operational}")

    props = backend.properties()
    tgt = backend.target
    validate_isa_circuits(isa_circuits, tgt)

    edge_cal: Dict[str, Dict[str, float]] = {}
    for i in range(len(CHAIN_13Q) - 1):
        u, v = CHAIN_13Q[i], CHAIN_13Q[i + 1]
        pair = (u, v) if (u, v) in tgt["cz"] else (v, u)
        err = float(tgt["cz"][pair].error)
        dur = float(tgt["cz"][pair].duration)
        if err > 0.02:
            raise RuntimeError(f"Pre-flight abort: CZ error on {pair} is {err:.5f} > 0.02")
        edge_cal[f"{min(u, v)}-{max(u, v)}"] = {"cz_error": err, "duration_s": dur}

    qubit_cal: Dict[str, Dict[str, float]] = {}
    for idx, q in enumerate(CHAIN_13Q):
        ro = float(props.readout_error(q))
        if ro > 0.05:
            raise RuntimeError(f"Pre-flight abort: readout error on q{q} is {ro:.5f} > 0.05")
        qubit_cal[str(q)] = {
            "role_in_d7": "data" if idx % 2 == 0 else "ancilla",
            "t1_s": float(props.t1(q)),
            "t2_s": float(props.t2(q)),
            "sx_error": float(tgt["sx"][(q,)].error or 0.0),
            "readout_error": ro,
            "prob_meas1_prep0": float(props.qubit_property(q, "prob_meas1_prep0")[0]),
            "prob_meas0_prep1": float(props.qubit_property(q, "prob_meas0_prep1")[0]),
            "measure_duration_s": float(tgt["measure"][(q,)].duration or 0.0),
            "reset_duration_s": float(tgt["reset"][(q,)].duration or 0.0),
        }

    # Pre-submission calibration decoder priors per distance (zero post-hoc parameter tuning)
    decoder_priors: Dict[str, Dict[str, float]] = {}
    for d in qec.DISTANCES:
        sub = qec.subchain_for_distance(CHAIN_13Q, d)
        _, p_dat, p_meas = build_calibrated_chain_noise_model(tgt, props, sub)
        decoder_priors[f"d{d}"] = {
            "p_data": p_dat,
            "p_meas": p_meas,
            "p_hook": 0.35 * p_dat,
        }

    cal_snapshot = {
        "benchmark_id": BENCHMARK_ID,
        "captured_at_utc": _utc_now(),
        "backend": backend.name,
        "pending_jobs": int(status.pending_jobs),
        "usage_before": {k: v for k, v in usage_before.items() if k != "instance_id"},
        "pinned_13q_chain": list(CHAIN_13Q),
        "edge_calibration": edge_cal,
        "qubit_calibration": qubit_cal,
        "decoder_priors": decoder_priors,
    }
    (EXPERIMENT_DIR / "qec001_fez_calibration_snapshot.json").write_text(
        json.dumps(cal_snapshot, indent=2), encoding="utf-8"
    )

    sampler = SamplerV2(mode=backend)
    job = sampler.run(isa_circuits, shots=int(manifest["shots_per_circuit"]))
    job_id = job.job_id()
    sub_record = {
        "benchmark_id": BENCHMARK_ID,
        "job_id": job_id,
        "backend": backend.name,
        "submitted_at_utc": _utc_now(),
        "preregistered_git_commit": "1a33ffe7",
        "qpy_sha256": actual_qpy_sha256,
        "manifest_sha256": manifest_sha256,
        "shots_per_circuit": int(manifest["shots_per_circuit"]),
        "total_circuits": len(isa_circuits),
        "total_shots": int(manifest["total_shots"]),
        "pending_jobs_at_submission": int(status.pending_jobs),
    }
    (EXPERIMENT_DIR / "qec001_fez_submitted_job.json").write_text(
        json.dumps(sub_record, indent=2), encoding="utf-8"
    )
    print(
        f"[SUBMITTED FROZEN QPY] Job ID: {job_id} on {backend.name} "
        f"(pending_jobs={status.pending_jobs}) | QPY SHA-256: {actual_qpy_sha256}",
        flush=True,
    )
    return job_id


def harvest_qec001_job(job_id: str, poll_seconds: int = 10) -> Dict[str, Any]:
    svc = _connect_service()
    manifest = json.loads((EXPERIMENT_DIR / "qec001_fez_staging_manifest.json").read_text(encoding="utf-8"))
    cal_snap = json.loads((EXPERIMENT_DIR / "qec001_fez_calibration_snapshot.json").read_text(encoding="utf-8"))
    metadata: List[Dict[str, Any]] = manifest["circuit_metadata"]
    priors: Dict[str, Dict[str, float]] = cal_snap["decoder_priors"]

    job = svc.job(job_id)
    while True:
        st = str(job.status())
        print(f"[{_utc_now()}] Job {job_id} status on {manifest['backend']}: {st}", flush=True)
        if st in ("DONE", "ERROR", "CANCELLED"):
            break
        time.sleep(poll_seconds)

    if str(job.status()) != "DONE":
        raise RuntimeError(f"Job {job_id} ended with non-DONE status: {job.status()}")

    result = job.result()
    if len(result) != len(metadata):
        raise RuntimeError(f"PubResult count {len(result)} != staged metadata count {len(metadata)}")

    # Prebuild exact MWPM decoders from pre-submission calibration priors
    decoders: Dict[Tuple[int, int], qec.PrecomputedMWPMDecoder] = {}
    for d in qec.DISTANCES:
        pr = priors[f"d{d}"]
        for r in qec.ROUNDS_LIST:
            decoders[(d, r)] = qec.build_mwpm_decoder(
                distance=d,
                rounds=r,
                p_data=float(pr["p_data"]),
                p_meas=float(pr["p_meas"]),
                p_hook=float(pr["p_hook"]),
            )

    raw_archive_records: List[Dict[str, Any]] = []
    circuit_summaries: List[Dict[str, Any]] = []

    # Group by (code_basis, distance, rounds) -> per-block accumulators
    grouped: Dict[Tuple[str, int, int], List[Dict[str, float]]] = {}
    n_blocks = int(manifest["n_blocks"])
    for basis in qec.CODE_BASES:
        for d in qec.DISTANCES:
            for r in qec.ROUNDS_LIST:
                grouped[(basis, d, r)] = [
                    {"shots": 0.0, "mwpm_err": 0.0, "maj_err": 0.0, "det_weighted": 0.0}
                    for _ in range(n_blocks)
                ]

    for idx, meta in enumerate(metadata):
        counts = _extract_counts(result[idx])
        shots = sum(counts.values())
        if shots != int(manifest["shots_per_circuit"]):
            raise RuntimeError(f"Shot count mismatch on circuit {idx}: {shots} != {manifest['shots_per_circuit']}")

        basis = str(meta["code_basis"])
        d = int(meta["distance"])
        r = int(meta["rounds"])
        state = int(meta["logical_state"])
        blk = int(meta["block_index"])

        ev = qec.evaluate_repetition_counts(
            counts,
            distance=d,
            rounds=r,
            logical_state=state,
            decoder=decoders[(d, r)],
        )
        raw_archive_records.append(
            {
                **meta,
                "circuit_index": idx,
                "shots": shots,
                "raw_counts": dict(sorted(counts.items())),
            }
        )
        circuit_summaries.append(
            {
                **meta,
                "circuit_index": idx,
                "shots": shots,
                "mwpm_logical_errors": int(round(ev["mwpm_logical_errors"])),
                "majority_logical_errors": int(round(ev["p_logical_majority"] * shots)),
                "p_logical_mwpm": ev["p_logical_mwpm"],
                "p_logical_majority": ev["p_logical_majority"],
                "epsilon_per_cycle": ev["epsilon_per_cycle"],
                "mean_defect_probability": ev["mean_defect_probability"],
            }
        )

        acc = grouped[(basis, d, r)][blk]
        acc["shots"] += float(shots)
        acc["mwpm_err"] += ev["mwpm_logical_errors"]
        acc["maj_err"] += ev["p_logical_majority"] * shots
        acc["det_weighted"] += ev["mean_defect_probability"] * shots

    # Write complete verbatim 192-circuit raw bitstring histograms to compressed archive
    raw_gz_path = EXPERIMENT_DIR / "physical_qec001_fez_raw_counts.json.gz"
    raw_json_bytes = json.dumps(raw_archive_records, separators=(",", ":")).encode("utf-8")
    with gzip.open(raw_gz_path, "wb", compresslevel=6) as gzf:
        gzf.write(raw_json_bytes)
    raw_gz_sha256 = hashlib.sha256(raw_gz_path.read_bytes()).hexdigest()

    # Aggregate pooled (basis, d, R) metrics across all 4 blocks & both initial logical states (16,384 shots/point)
    grid_summary: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for basis in qec.CODE_BASES:
        grid_summary[basis] = {}
        for d in qec.DISTANCES:
            grid_summary[basis][f"d{d}"] = {}
            for r in qec.ROUNDS_LIST:
                blks = grouped[(basis, d, r)]
                tot_shots = sum(b["shots"] for b in blks)
                tot_mwpm = sum(b["mwpm_err"] for b in blks)
                tot_maj = sum(b["maj_err"] for b in blks)
                mean_det = sum(b["det_weighted"] for b in blks) / tot_shots

                p_mwpm = tot_mwpm / tot_shots
                p_maj = tot_maj / tot_shots
                block_ps = [b["mwpm_err"] / b["shots"] for b in blks]
                binom_se = math.sqrt(max(p_mwpm * (1.0 - p_mwpm), 0.5 / tot_shots) / tot_shots)
                block_sem = float(np.std(block_ps, ddof=1) / math.sqrt(len(block_ps))) if len(block_ps) > 1 else 0.0
                se_eff = max(binom_se, block_sem)

                eps = qec.logical_error_per_cycle(p_mwpm, r)
                eps_hi = qec.logical_error_per_cycle(min(0.499999, p_mwpm + se_eff), r)
                eps_lo = qec.logical_error_per_cycle(max(0.0, p_mwpm - se_eff), r)
                eps_se_eff = 0.5 * abs(eps_hi - eps_lo)
                eps_maj = qec.logical_error_per_cycle(p_maj, r)

                grid_summary[basis][f"d{d}"][f"R{r}"] = {
                    "total_shots": int(tot_shots),
                    "mwpm_logical_errors": int(round(tot_mwpm)),
                    "majority_logical_errors": int(round(tot_maj)),
                    "p_logical_mwpm": p_mwpm,
                    "p_logical_se_eff": se_eff,
                    "block_p_logicals": block_ps,
                    "p_logical_majority": p_maj,
                    "epsilon_per_cycle": eps,
                    "epsilon_per_cycle_se_eff": eps_se_eff,
                    "epsilon_per_cycle_majority": eps_maj,
                    "mean_defect_probability": mean_det,
                }

    # Compute suppression factors Lambda_{d -> d+2}(basis, R) with pre-registered floor 1 / (2 * 16384 * R)
    suppression_observed: Dict[str, Dict[str, Any]] = {}
    for basis in qec.CODE_BASES:
        suppression_observed[basis] = {}
        for r in qec.ROUNDS_LIST:
            rk = f"R{r}"
            floor = 1.0 / (2.0 * 16384.0 * float(r))
            e3 = float(grid_summary[basis]["d3"][rk]["epsilon_per_cycle"])
            e5 = float(grid_summary[basis]["d5"][rk]["epsilon_per_cycle"])
            e7 = float(grid_summary[basis]["d7"][rk]["epsilon_per_cycle"])
            suppression_observed[basis][rk] = {
                "eps_d3": e3,
                "eps_d5": e5,
                "eps_d7": e7,
                "lambda_3_to_5": e3 / max(e5, floor),
                "lambda_5_to_7": e5 / max(e7, floor),
                "lambda_3_to_7": e3 / max(e7, floor),
                "monotonic_suppression": bool(e3 > e5 > e7 or (e3 > e5 and e5 == 0.0 and e7 == 0.0)),
            }

    # Evaluate pre-registered hypotheses QEC-H1, QEC-H2, QEC-H3
    bf5 = suppression_observed["bit_flip_z"]["R5"]
    bf7 = suppression_observed["bit_flip_z"]["R7"]
    pf5 = suppression_observed["phase_flip_x"]["R5"]
    pf7 = suppression_observed["phase_flip_x"]["R7"]

    qec_h1_pass = bool(
        bf5["monotonic_suppression"]
        and bf7["monotonic_suppression"]
        and bf5["lambda_3_to_5"] > 1.25
        and bf7["lambda_3_to_5"] > 1.25
        and bf5["lambda_3_to_7"] > 2.0
        and bf7["lambda_3_to_7"] > 2.0
    )
    qec_h2_subthreshold = bool(
        pf5["lambda_3_to_5"] > 1.15 and pf5["lambda_5_to_7"] > 1.15
    )
    qec_h3_pass = bool(
        grid_summary["bit_flip_z"]["d5"]["R7"]["p_logical_mwpm"]
        <= grid_summary["bit_flip_z"]["d5"]["R7"]["p_logical_majority"]
        and grid_summary["bit_flip_z"]["d7"]["R7"]["p_logical_mwpm"]
        <= grid_summary["bit_flip_z"]["d7"]["R7"]["p_logical_majority"]
        and all(
            grid_summary[b][f"d{d}"]["R7"]["mean_defect_probability"] < 0.12
            for b in qec.CODE_BASES
            for d in qec.DISTANCES
        )
    )

    hypotheses = {
        "QEC_H1_bit_flip_subthreshold_monotonic_suppression": {
            "passed": qec_h1_pass,
            "R5": bf5,
            "R7": bf7,
        },
        "QEC_H2_phase_flip_midcircuit_dephasing_diagnostic": {
            "subthreshold_passed": qec_h2_subthreshold,
            "regime": "SUB_THRESHOLD" if qec_h2_subthreshold else (
                "ABOVE_THRESHOLD_SPECTATOR_DEPHASING" if pf5["lambda_3_to_5"] < 1.0 else "MARGINAL_NEAR_THRESHOLD"
            ),
            "R5": pf5,
            "R7": pf7,
        },
        "QEC_H3_spacetime_mwpm_advantage_and_defect_ceiling": {
            "passed": qec_h3_pass,
            "bit_flip_d5_R7_mwpm_vs_maj": [
                grid_summary["bit_flip_z"]["d5"]["R7"]["p_logical_mwpm"],
                grid_summary["bit_flip_z"]["d5"]["R7"]["p_logical_majority"],
            ],
            "bit_flip_d7_R7_mwpm_vs_maj": [
                grid_summary["bit_flip_z"]["d7"]["R7"]["p_logical_mwpm"],
                grid_summary["bit_flip_z"]["d7"]["R7"]["p_logical_majority"],
            ],
            "phase_flip_d5_R7_mwpm_vs_maj": [
                grid_summary["phase_flip_x"]["d5"]["R7"]["p_logical_mwpm"],
                grid_summary["phase_flip_x"]["d5"]["R7"]["p_logical_majority"],
            ],
            "phase_flip_d7_R7_mwpm_vs_maj": [
                grid_summary["phase_flip_x"]["d7"]["R7"]["p_logical_mwpm"],
                grid_summary["phase_flip_x"]["d7"]["R7"]["p_logical_majority"],
            ],
        },
    }

    usage_after = svc.usage()
    metrics = {}
    try:
        metrics = job.metrics() or {}
    except Exception:
        metrics = {}

    payload = {
        "benchmark_id": BENCHMARK_ID,
        "harvested_at_utc": _utc_now(),
        "preregistered_git_commit": "1a33ffe7",
        "job_id": job_id,
        "backend": manifest["backend"],
        "pinned_13q_chain": manifest["pinned_13q_chain"],
        "subchains_by_distance": manifest["subchains_by_distance"],
        "qpy_sha256": manifest["qpy_sha256"],
        "raw_counts_archive": raw_gz_path.name,
        "raw_counts_archive_sha256": raw_gz_sha256,
        "total_circuits": manifest["total_circuits"],
        "shots_per_circuit": manifest["shots_per_circuit"],
        "total_shots": manifest["total_shots"],
        "job_metrics": metrics,
        "usage_after": {k: v for k, v in usage_after.items() if k != "instance_id"},
        "grid_summary": grid_summary,
        "suppression_observed": suppression_observed,
        "preregistered_hypotheses": hypotheses,
        "circuit_summaries": circuit_summaries,
    }
    out_path = EXPERIMENT_DIR / "physical_qec001_fez_results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[HARVESTED] Saved {out_path} and {raw_gz_path}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--submit-frozen-and-harvest", action="store_true")
    parser.add_argument("--harvest-job", type=str, default=None)
    args = parser.parse_args()

    if args.harvest_job:
        res = harvest_qec001_job(args.harvest_job)
    elif args.submit_frozen_and_harvest:
        jid = submit_frozen_qec001()
        res = harvest_qec001_job(jid)
    elif args.stage_only:
        stage_qec001()
        return
    else:
        parser.error("Specify --stage-only, --submit-frozen-and-harvest, or --harvest-job <JOB_ID>")
        return

    print("\n=== OSIRIS-QEC-001 (ibm_fez) PHYSICAL SYNDROME-CYCLE SCALING SUMMARY ===")
    for basis in qec.CODE_BASES:
        print(f"\n  [{basis}]")
        for r in qec.ROUNDS_LIST:
            rk = f"R{r}"
            s = res["suppression_observed"][basis][rk]
            print(
                f"    {rk}: eps_L(d=3)={s['eps_d3']:.5f}, eps_L(d=5)={s['eps_d5']:.5f}, eps_L(d=7)={s['eps_d7']:.5f} "
                f"| Lambda_3->5={s['lambda_3_to_5']:.2f}x, Lambda_5->7={s['lambda_5_to_7']:.2f}x"
            )
    print("\n  Pre-registered hypotheses:", {k: v.get("passed", v.get("regime")) for k, v in res["preregistered_hypotheses"].items()})


if __name__ == "__main__":
    main()
