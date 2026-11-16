#!/usr/bin/env python3
"""
stage_and_harvest_hp004b_fez.py — Cross-processor replication harness (`OSIRIS-HP-004B`)
on `ibm_fez` (`[2, 24, 4, 3, 16, 23, 22, 21]`).

Default action `--stage-only` is 100% read-only/offline (0 QPU seconds spent):
validates all 286 ISA circuits against `ibm_fez.target`, enforces gate-level light-cone
isolation, computes the live-calibrated Kraus + T1/T2 + readout prediction, and writes
the frozen `.qpy` and SHA-256 manifest. QPU submission requires explicit `--submit-frozen-and-harvest`
after separate owner approval.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from qiskit import qpy, transpile
from qiskit.quantum_info import DensityMatrix, Kraus
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit_ibm_runtime.utils.validations import validate_isa_circuits

EXPERIMENT_DIR = Path(__file__).resolve().parent
HP004_MOD_PATH = EXPERIMENT_DIR.parent / "hp004" / "hp004_joint_stabilizer.py"
_SPEC = importlib.util.spec_from_file_location("hp004_joint_stabilizer", HP004_MOD_PATH)
assert _SPEC is not None and _SPEC.loader is not None
hp004 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hp004)

BENCHMARK_ID = "OSIRIS-HP-004B"
BACKEND_NAME = "ibm_fez"
N_BLOCKS = 6
SHOTS_PER_CIRCUIT = 1024
SCHEDULE_SEED = 20261117
# Optimal isomorphic 8-transmon heavy-hex tree on ibm_fez (mean CZ error ~0.00179):
#   w0=2(R2), w1=24(R1), w2=4(C1), w3=3(E1), w4=16(D1/M1'), w5=23(E2/M2'), w6=22(D2), w7=21(C2)
FEZ_LAYOUT: Tuple[int, ...] = (2, 24, 4, 3, 16, 23, 22, 21)
READOUT_PHYS: Tuple[int, ...] = (FEZ_LAYOUT[0], FEZ_LAYOUT[4], FEZ_LAYOUT[1], FEZ_LAYOUT[5])  # (2, 16, 24, 23)


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


def _thermal_kraus(t_sec: float, t1: float, t2: float) -> Kraus:
    p_amp = 1.0 - np.exp(-t_sec / max(t1, 1e-9))
    inv_tphi = max(0.0, 1.0 / max(t2, 1e-9) - 0.5 / max(t1, 1e-9))
    p_phi = 1.0 - np.exp(-t_sec * inv_tphi)
    k_amp = [np.array([[1, 0], [0, np.sqrt(1 - p_amp)]], dtype=complex),
             np.array([[0, np.sqrt(p_amp)], [0, 0]], dtype=complex)]
    k_phi = [np.array([[1, 0], [0, np.sqrt(1 - p_phi)]], dtype=complex),
             np.array([[0, 0], [0, np.sqrt(p_phi)]], dtype=complex)]
    return Kraus([b @ a for a in k_amp for b in k_phi])


def _depol_1q(err: float) -> Kraus:
    lam = min(1.0, 2.0 * max(0.0, err))
    I = np.eye(2, dtype=complex); X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex); Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return Kraus([np.sqrt(1 - 0.75 * lam) * I, np.sqrt(lam / 4) * X, np.sqrt(lam / 4) * Y, np.sqrt(lam / 4) * Z])


def _depol_2q(err: float) -> Kraus:
    lam = min(1.0, 1.25 * max(0.0, err))
    paulis = [np.eye(2, dtype=complex), np.array([[0, 1], [1, 0]], dtype=complex),
              np.array([[0, -1j], [1j, 0]], dtype=complex), np.array([[1, 0], [0, -1]], dtype=complex)]
    ops = []
    for i, p1 in enumerate(paulis):
        for j, p2 in enumerate(paulis):
            pk = np.kron(p2, p1)
            ops.append((np.sqrt(1 - (15.0 / 16.0) * lam) if (i == 0 and j == 0) else np.sqrt(lam / 16.0)) * pk)
    return Kraus(ops)


def predict_fez_kraus_fidelities(tgt: Any, props: Any) -> Dict[str, Any]:
    pos_of = {phys: idx for idx, phys in enumerate(FEZ_LAYOUT)}
    t1s = {q: float(props.t1(q)) for q in FEZ_LAYOUT}
    t2s = {q: min(float(props.t2(q)), 2.0 * t1s[q]) for q in FEZ_LAYOUT}

    def sim_one(qc_isa: Any, shots: int = 6 * 1024) -> Tuple[Dict[str, int], float]:
        rho = DensityMatrix.from_int(0, 2**8)
        q_time = {i: 0.0 for i in range(8)}
        readout_map = {}
        for inst in qc_isa.data:
            op = inst.operation
            name = op.name
            phys_qs = [qc_isa.find_bit(qb).index for qb in inst.qubits]
            loc_qs = [pos_of[pq] for pq in phys_qs]
            if name == "barrier":
                t_max = max(q_time[lq] for lq in loc_qs)
                for lq, pq in zip(loc_qs, phys_qs):
                    if t_max > q_time[lq]:
                        rho = rho.evolve(_thermal_kraus(t_max - q_time[lq], t1s[pq], t2s[pq]), [lq])
                        q_time[lq] = t_max
            elif name == "rz":
                rho = rho.evolve(op, loc_qs)
            elif name in ("sx", "x"):
                pq, lq = phys_qs[0], loc_qs[0]
                dur = float(tgt[name][(pq,)].duration or 36e-9)
                err = float(tgt[name][(pq,)].error or 3e-4)
                rho = rho.evolve(op, [lq]).evolve(_depol_1q(err), [lq]).evolve(_thermal_kraus(dur, t1s[pq], t2s[pq]), [lq])
                q_time[lq] += dur
            elif name == "cz":
                pq0, pq1 = phys_qs; lq0, lq1 = loc_qs
                t_sync = max(q_time[lq0], q_time[lq1])
                for lq, pq in [(lq0, pq0), (lq1, pq1)]:
                    if t_sync > q_time[lq]:
                        rho = rho.evolve(_thermal_kraus(t_sync - q_time[lq], t1s[pq], t2s[pq]), [lq])
                        q_time[lq] = t_sync
                edge = (pq0, pq1) if (pq0, pq1) in tgt["cz"] else (pq1, pq0)
                dur = float(tgt["cz"][edge].duration or 68e-9)
                err = float(tgt["cz"][edge].error or 3e-3)
                rho = rho.evolve(op, [lq0, lq1]).evolve(_depol_2q(err), [lq0, lq1])
                rho = rho.evolve(_thermal_kraus(dur, t1s[pq0], t2s[pq0]), [lq0]).evolve(_thermal_kraus(dur, t1s[pq1], t2s[pq1]), [lq1])
                q_time[lq0] += dur; q_time[lq1] += dur
            elif name == "measure":
                readout_map[qc_isa.find_bit(inst.clbits[0]).index] = (loc_qs[0], phys_qs[0])
        t_end = max(q_time.values())
        for cl_idx, (lq, pq) in readout_map.items():
            ro_dur = float(tgt["measure"][(pq,)].duration or 1.2e-6)
            rho = rho.evolve(_thermal_kraus((t_end - q_time[lq]) + 0.5 * ro_dur, t1s[pq], t2s[pq]), [lq])
        probs_8q = np.real(np.diag(rho.data))
        probs_4b = np.zeros(16)
        for s_idx, p in enumerate(probs_8q):
            if p < 1e-12:
                continue
            dist = np.array([1.0])
            for cl_idx in [0, 1, 2, 3]:
                lq, pq = readout_map[cl_idx]
                bval = (s_idx >> lq) & 1
                p10 = float(props.qubit_property(pq, "prob_meas1_prep0")[0])
                p01 = float(props.qubit_property(pq, "prob_meas0_prep1")[0])
                dist = np.kron(np.array([(1 - p10) if bval == 0 else p01, p10 if bval == 0 else (1 - p01)]), dist)
            probs_4b += p * dist
        return {format(i, "04b"): max(1, int(round(probs_4b[i] * shots))) for i in range(16) if probs_4b[i] * shots >= 0.5}, t_end

    predictions: Dict[str, Any] = {}
    for arm in hp004.ARMS:
        b_cnts = {}
        t_c = 0.0
        cz_c = 0
        for p, q in hp004.JOINT_BASES:
            raw_c = hp004.build_hp004_circuit(arm, p, q)
            isa_c = transpile(raw_c, target=tgt, initial_layout=list(FEZ_LAYOUT), routing_method="none", optimization_level=1, seed_transpiler=42)
            cz_c = int(isa_c.count_ops().get("cz", 0))
            b_cnts[f"{p}{q}"], t_c = sim_one(isa_c)
        est = hp004.estimate_joint_bell_fidelity(b_cnts)
        predictions[arm] = {
            "cz_count": cz_c,
            "duration_us": round(t_c * 1e6, 3),
            "predicted_raw_f1": round(est["f1_marginal"], 5),
            "predicted_raw_f2": round(est["f2_marginal"], 5),
            "predicted_raw_f_joint": round(est["f_joint"], 5),
            "predicted_shot_se": round(est["f_joint_shot_se"], 5),
            "predicted_z_above_0p50": round(est["z_above_1ebit_shot"], 2),
        }
    return predictions


def stage_fez() -> Dict[str, Any]:
    svc = _connect_service()
    usage = svc.usage()
    backend = svc.backend(BACKEND_NAME)
    status = backend.status()
    props = backend.properties()
    tgt = backend.target

    raw_circuits, metadata = hp004.assemble_hp004_schedule(n_blocks=N_BLOCKS, seed=SCHEDULE_SEED)
    isa_circuits = transpile(
        raw_circuits,
        target=tgt,
        initial_layout=list(FEZ_LAYOUT),
        routing_method="none",
        optimization_level=1,
        seed_transpiler=42,
    )
    validate_isa_circuits(isa_circuits, tgt)

    # Verify light cones & readout lock
    phys_set = set(FEZ_LAYOUT)
    for qc, meta in zip(isa_circuits, metadata):
        used = {qc.find_bit(q).index for inst in qc.data for q in inst.qubits}
        assert used.issubset(phys_set)
        meas = tuple(
            qc.find_bit(i.qubits[0]).index
            for i in sorted((x for x in qc.data if x.operation.name == "measure"), key=lambda x: qc.find_bit(x.clbits[0]).index)
        )
        assert meas == READOUT_PHYS
        if meta["kind"] == "tomography":
            bars = [idx for idx, inst in enumerate(qc.data) if inst.operation.name == "barrier"]
            dec_phys = {qc.find_bit(q).index for inst in qc.data[bars[1] + 1 : bars[2]] for q in inst.qubits}
            assert dec_phys.isdisjoint({FEZ_LAYOUT[0], FEZ_LAYOUT[1]})
            if meta["arm"] in ("Arm B", "Arm B-Ctrl", "Arm B-Held"):
                assert dec_phys.issubset({FEZ_LAYOUT[3], FEZ_LAYOUT[4], FEZ_LAYOUT[5], FEZ_LAYOUT[6]})
            elif meta["arm"] == "Arm B-Def":
                assert dec_phys.issubset({FEZ_LAYOUT[3], FEZ_LAYOUT[4], FEZ_LAYOUT[5]})

    qpy_path = EXPERIMENT_DIR / "hp004b_fez_staged_circuits.qpy"
    with open(qpy_path, "wb") as fh:
        qpy.dump(isa_circuits, fh)
    qpy_sha256 = hashlib.sha256(qpy_path.read_bytes()).hexdigest()

    predictions = predict_fez_kraus_fidelities(tgt, props)

    tree_phys_edges = sorted({tuple(sorted((FEZ_LAYOUT[a], FEZ_LAYOUT[b]))) for a, b in hp004.TREE_EDGES_8Q})
    edge_cal = {}
    for u, v in tree_phys_edges:
        pair = (u, v) if (u, v) in tgt["cz"] else (v, u)
        edge_cal[f"{u}-{v}"] = {
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
        "pinned_tree_layout": list(FEZ_LAYOUT),
        "readout_transmons_c0_c1_c2_c3": list(READOUT_PHYS),
        "edge_calibration": edge_cal,
        "kraus_noisy_predictions": predictions,
        "n_blocks": N_BLOCKS,
        "shots_per_circuit": SHOTS_PER_CIRCUIT,
        "total_circuits": len(isa_circuits),
        "total_shots": len(isa_circuits) * SHOTS_PER_CIRCUIT,
        "estimated_qpu_seconds": 81,
        "schedule_seed": SCHEDULE_SEED,
        "qpy_sha256": qpy_sha256,
        "circuit_metadata": metadata,
    }
    (EXPERIMENT_DIR / "hp004b_fez_staging_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[STAGED OFFLINE] {BENCHMARK_ID} on {backend.name} | 286 circuits | QPY SHA-256: {qpy_sha256[:16]}... | 0 QPU seconds spent")
    for arm, pred in predictions.items():
        print(f"  {arm:10s}: predicted raw F_joint = {pred['predicted_raw_f_joint']:.4f} +/- {pred['predicted_shot_se']:.4f} ({pred['predicted_z_above_0p50']:+.1f}s)")
    return manifest


def _extract_counts(pub_result: Any) -> Dict[str, int]:
    data_bin = pub_result.data
    if hasattr(data_bin, "c"):
        return {str(k): int(v) for k, v in data_bin.c.get_counts().items()}
    if hasattr(data_bin, "meas"):
        return {str(k): int(v) for k, v in data_bin.meas.get_counts().items()}
    for field in dir(data_bin):
        if not field.startswith("_"):
            obj = getattr(data_bin, field)
            if hasattr(obj, "get_counts"):
                return {str(k): int(v) for k, v in obj.get_counts().items()}
    raise RuntimeError("Could not locate BitArray with get_counts() in SamplerV2 PubResult")


def _bootstrap_ci_joint(
    pooled_counts: Mapping[str, Mapping[str, int]],
    n_boot: int = 5000,
    seed: int = 42,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    keys_and_probs = {}
    for bkey, hist in pooled_counts.items():
        labels = list(hist.keys())
        tot = sum(hist.values())
        probs = np.array([hist[k] / tot for k in labels], dtype=float)
        keys_and_probs[bkey] = (labels, tot, probs)
    samples = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        sim_counts = {}
        for bkey, (labels, tot, probs) in keys_and_probs.items():
            draws = rng.multinomial(tot, probs)
            sim_counts[bkey] = {lbl: int(n) for lbl, n in zip(labels, draws) if n > 0}
        samples[idx] = hp004.estimate_joint_bell_fidelity(sim_counts)["f_joint"]
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _spam_mitigated_joint(
    cal_matrix: np.ndarray,
    pooled_counts: Mapping[str, Mapping[str, int]],
) -> Dict[str, float]:
    """Secondary sensitivity check ONLY: invert empirical 16x16 readout matrix from circuits 0..15."""
    inv_m = np.linalg.pinv(cal_matrix, rcond=1e-4)
    mit_counts: Dict[str, Dict[str, int]] = {}
    for bkey, hist in pooled_counts.items():
        tot = sum(hist.values())
        y = np.zeros(16, dtype=float)
        for bits, n in hist.items():
            y[int(bits.replace(" ", ""), 2)] += n / tot
        x = np.clip(inv_m @ y, 0.0, None)
        s = float(x.sum())
        if s > 0:
            x /= s
        mit_counts[bkey] = {
            format(i, "04b"): max(1, int(round(x[i] * tot * 100)))
            for i in range(16)
            if x[i] * tot * 100 >= 0.5
        }
    est = hp004.estimate_joint_bell_fidelity(mit_counts)
    return {
        "f1_spam_mitigated": est["f1_marginal"],
        "f2_spam_mitigated": est["f2_marginal"],
        "f_joint_spam_mitigated": est["f_joint"],
    }


def submit_frozen_fez() -> str:
    """Verify frozen `.qpy` SHA-256 against pre-registered manifest, snapshot live calibration, and dispatch 1 SamplerV2 job."""
    qpy_path = EXPERIMENT_DIR / "hp004b_fez_staged_circuits.qpy"
    manifest_path = EXPERIMENT_DIR / "hp004b_fez_staging_manifest.json"
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

    # Enforce hardware calibration sanity gates on pinned 8-transmon tree before spending QPU seconds
    tree_phys_edges = sorted({tuple(sorted((FEZ_LAYOUT[a], FEZ_LAYOUT[b]))) for a, b in hp004.TREE_EDGES_8Q})
    edge_cal: Dict[str, Dict[str, float]] = {}
    for u, v in tree_phys_edges:
        pair = (u, v) if (u, v) in tgt["cz"] else (v, u)
        err = float(tgt["cz"][pair].error)
        dur = float(tgt["cz"][pair].duration)
        if err > 0.02:
            raise RuntimeError(f"Pre-flight abort: CZ error on {pair} is {err:.5f} > 0.02")
        edge_cal[f"{u}-{v}"] = {"cz_error": err, "duration_s": dur}

    qubit_cal: Dict[str, Dict[str, float]] = {}
    for q in FEZ_LAYOUT:
        ro = float(props.readout_error(q))
        if q in READOUT_PHYS and ro > 0.05:
            raise RuntimeError(f"Pre-flight abort: readout error on q{q} is {ro:.5f} > 0.05")
        qubit_cal[str(q)] = {
            "t1_s": float(props.t1(q)),
            "t2_s": float(props.t2(q)),
            "sx_error": float(tgt["sx"][(q,)].error or 0.0),
            "readout_error": ro,
            "prob_meas1_prep0": float(props.qubit_property(q, "prob_meas1_prep0")[0]),
            "prob_meas0_prep1": float(props.qubit_property(q, "prob_meas0_prep1")[0]),
        }

    cal_snapshot = {
        "benchmark_id": BENCHMARK_ID,
        "captured_at_utc": _utc_now(),
        "backend": backend.name,
        "pending_jobs": int(status.pending_jobs),
        "usage_before": {k: v for k, v in usage_before.items() if k != "instance_id"},
        "pinned_tree_layout": list(FEZ_LAYOUT),
        "readout_transmons_c0_c1_c2_c3": list(READOUT_PHYS),
        "edge_calibration": edge_cal,
        "qubit_calibration": qubit_cal,
    }
    (EXPERIMENT_DIR / "hp004b_fez_calibration_snapshot.json").write_text(
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
        "preregistered_git_commit": "0c5a5a39",
        "qpy_sha256": actual_qpy_sha256,
        "manifest_sha256": manifest_sha256,
        "shots_per_circuit": int(manifest["shots_per_circuit"]),
        "total_circuits": len(isa_circuits),
        "total_shots": int(manifest["total_shots"]),
        "pending_jobs_at_submission": int(status.pending_jobs),
    }
    (EXPERIMENT_DIR / "hp004b_fez_submitted_job.json").write_text(
        json.dumps(sub_record, indent=2), encoding="utf-8"
    )
    print(
        f"[SUBMITTED FROZEN QPY] Job ID: {job_id} on {backend.name} "
        f"(pending_jobs={status.pending_jobs}) | QPY SHA-256: {actual_qpy_sha256}",
        flush=True,
    )
    return job_id


def harvest_fez_job(job_id: str, poll_seconds: int = 10) -> Dict[str, Any]:
    svc = _connect_service()
    manifest = json.loads((EXPERIMENT_DIR / "hp004b_fez_staging_manifest.json").read_text(encoding="utf-8"))
    metadata: List[Dict[str, Any]] = manifest["circuit_metadata"]

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

    circuit_records: List[Dict[str, Any]] = []
    cal_matrix = np.zeros((16, 16), dtype=float)
    arm_block_counts: Dict[str, List[Dict[str, Dict[str, int]]]] = {
        arm: [{} for _ in range(int(manifest["n_blocks"]))] for arm in hp004.ARMS
    }

    for idx, meta in enumerate(metadata):
        counts = _extract_counts(result[idx])
        shots = sum(counts.values())
        if shots != int(manifest["shots_per_circuit"]):
            raise RuntimeError(f"Shot count mismatch on circuit {idx}: {shots} != {manifest['shots_per_circuit']}")
        rec = {**meta, "circuit_index": idx, "shots": shots, "raw_counts": dict(sorted(counts.items()))}
        circuit_records.append(rec)

        if meta["kind"] == "spam_cal":
            prep_idx = int(str(meta["prepared_bits"]), 2)
            for bits, n in counts.items():
                cal_matrix[int(bits.replace(" ", ""), 2), prep_idx] += n / shots
        else:
            arm = str(meta["arm"])
            b_idx = int(meta["block_index"])
            basis = str(meta["basis"])
            arm_block_counts[arm][b_idx][basis] = counts

    arm_summaries: Dict[str, Any] = {}
    for arm in hp004.ARMS:
        blocks = arm_block_counts[arm]
        summary = hp004.summarize_blocks(blocks)
        pooled: Dict[str, Dict[str, int]] = {f"{p}{q}": {} for p, q in hp004.JOINT_BASES}
        block_f_list: List[float] = []
        for b_hist in blocks:
            block_f_list.append(hp004.estimate_joint_bell_fidelity(b_hist)["f_joint"])
            for bkey, hist in b_hist.items():
                for bits, n in hist.items():
                    clean = bits.replace(" ", "")
                    pooled[bkey][clean] = pooled[bkey].get(clean, 0) + int(n)
        ci_lo, ci_hi = _bootstrap_ci_joint(pooled)
        spam_sec = _spam_mitigated_joint(cal_matrix, pooled)
        arm_summaries[arm] = {
            **summary,
            "block_f_joints": block_f_list,
            "f_joint_shot_boot_95ci": [ci_lo, ci_hi],
            "secondary_spam_sensitivity": spam_sec,
        }

    a_sum = arm_summaries["Arm A"]
    b_sum = arm_summaries["Arm B"]
    c_sum = arm_summaries["Arm B-Ctrl"]
    d_sum = arm_summaries["Arm B-Def"]
    h_sum = arm_summaries["Arm B-Held"]

    hypotheses = {
        "H1_arm_a_inverse_above_0p60": {
            "passed": bool(a_sum["f_joint"] > 0.6000),
            "observed_raw_f_joint": a_sum["f_joint"],
            "se_eff": a_sum["f_joint_se_eff"],
        },
        "H2_PRIMARY_arm_b_exceeds_1ebit_0p50_at_5sigma": {
            "passed": bool((b_sum["f_joint"] - 5.0 * b_sum["f_joint_se_eff"]) > 0.5000),
            "observed_raw_f_joint": b_sum["f_joint"],
            "se_eff": b_sum["f_joint_se_eff"],
            "lower_5sigma_bound": b_sum["f_joint"] - 5.0 * b_sum["f_joint_se_eff"],
            "z_above_0p50_eff": b_sum["z_above_1ebit_eff"],
        },
        "H3_arm_b_ctrl_within_0ebit_separable_band": {
            "passed": bool(
                c_sum["f_joint"] <= (0.2500 + 3.0 * c_sum["f_joint_se_eff"])
                and c_sum["f_joint"] > 0.1250
            ),
            "observed_raw_f_joint": c_sum["f_joint"],
            "se_eff": c_sum["f_joint_se_eff"],
        },
        "H4_arm_b_def_sub_capacity_collapse": {
            "passed": bool(
                d_sum["f_joint"] <= (0.1250 + 3.0 * d_sum["f_joint_se_eff"])
                and d_sum["f1_marginal"] <= (0.5000 + 3.0 * d_sum["f1_shot_se"])
            ),
            "observed_raw_f_joint": d_sum["f_joint"],
            "observed_raw_f1": d_sum["f1_marginal"],
            "se_eff": d_sum["f_joint_se_eff"],
        },
        "H5_arm_b_held_unitary_specificity": {
            "passed": bool(0.0400 <= h_sum["f_joint"] <= 0.0900),
            "observed_raw_f_joint": h_sum["f_joint"],
            "se_eff": h_sum["f_joint_se_eff"],
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
        "preregistered_git_commit": "0c5a5a39",
        "replicates_marrakesh_job_id": "dbti1fbsbsbs73cb2vu0",
        "job_id": job_id,
        "backend": manifest["backend"],
        "pinned_tree_layout": manifest["pinned_tree_layout"],
        "readout_transmons_c0_c1_c2_c3": manifest["readout_transmons_c0_c1_c2_c3"],
        "qpy_sha256": manifest["qpy_sha256"],
        "kraus_noisy_predictions_at_staging": manifest["kraus_noisy_predictions"],
        "total_circuits": manifest["total_circuits"],
        "shots_per_circuit": manifest["shots_per_circuit"],
        "total_shots": manifest["total_shots"],
        "job_metrics": metrics,
        "usage_after": {k: v for k, v in usage_after.items() if k != "instance_id"},
        "spam_assignment_diagonal_mean": float(np.mean(np.diag(cal_matrix))),
        "arm_summaries": arm_summaries,
        "preregistered_hypotheses": hypotheses,
        "circuit_records": circuit_records,
    }
    out_path = EXPERIMENT_DIR / "physical_hp004b_fez_results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[HARVESTED] Saved complete raw counts + analysis to {out_path}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--submit-frozen-and-harvest", action="store_true")
    parser.add_argument("--harvest-job", type=str, default=None)
    args = parser.parse_args()

    if args.harvest_job:
        res = harvest_fez_job(args.harvest_job)
    elif args.submit_frozen_and_harvest:
        jid = submit_frozen_fez()
        res = harvest_fez_job(jid)
    elif args.stage_only:
        stage_fez()
        return
    else:
        parser.error("Specify --stage-only, --submit-frozen-and-harvest, or --harvest-job <JOB_ID>")
        return

    print("\n=== OSIRIS-HP-004B (ibm_fez) PHYSICAL HARDWARE SUMMARY ===")
    for arm, s in res["arm_summaries"].items():
        sec = s["secondary_spam_sensitivity"]["f_joint_spam_mitigated"]
        print(
            f"  {arm:10s}: Raw F_joint = {s['f_joint']:.4f} +/- {s['f_joint_se_eff']:.4f} "
            f"(F1={s['f1_marginal']:.4f}, F2={s['f2_marginal']:.4f}, Z_>0.5={s['z_above_1ebit_eff']:+.2f}s | SPAM-sec={sec:.4f})"
        )
    print("  Pre-registered hypotheses:", {k: v["passed"] for k, v in res["preregistered_hypotheses"].items()})


if __name__ == "__main__":
    main()
