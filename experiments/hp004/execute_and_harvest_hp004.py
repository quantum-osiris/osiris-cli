#!/usr/bin/env python3
"""
execute_and_harvest_hp004.py — Pre-registered staging, pre-flight verification,
single-job QPU submission, and raw-bitstring harvesting for OSIRIS-HP-004.

Never prints credentials or tokens. Archives frozen `.qpy`, calibration snapshot,
submitted job receipt, and full raw 4-bit counts for all 286 circuits.
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from qiskit import qpy, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit_ibm_runtime.utils.validations import validate_isa_circuits

EXPERIMENT_DIR = Path(__file__).resolve().parent
MOD_PATH = EXPERIMENT_DIR / "hp004_joint_stabilizer.py"
_SPEC = importlib.util.spec_from_file_location("hp004_joint_stabilizer", MOD_PATH)
assert _SPEC is not None and _SPEC.loader is not None
hp004 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hp004)

BENCHMARK_ID = "OSIRIS-HP-004"
BACKEND_NAME = "ibm_marrakesh"
N_BLOCKS = 6
SHOTS_PER_CIRCUIT = 1024
SCHEDULE_SEED = 20261116
LAYOUT: Tuple[int, ...] = tuple(hp004.DEFAULT_MARRAKESH_TREE_LAYOUT)  # (34, 52, 32, 33, 39, 53, 54, 55)
READOUT_PHYS: Tuple[int, ...] = (LAYOUT[0], LAYOUT[4], LAYOUT[1], LAYOUT[5])  # (34, 39, 52, 53)


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
    raise RuntimeError("No IBM Quantum credential found in ~/apikey.json or ~/osiris-ide/.env")


def _connect_service() -> QiskitRuntimeService:
    return QiskitRuntimeService(channel="ibm_quantum_platform", token=_load_token(), instance="open-instance")


def _verify_compiled_light_cones(isa_circuits: Sequence[Any], metadata: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    phys_set = set(LAYOUT)
    cz_stats: Dict[str, List[int]] = {}
    depth_stats: Dict[str, List[int]] = {}
    for qc, meta in zip(isa_circuits, metadata):
        used_q = {qc.find_bit(q).index for inst in qc.data for q in inst.qubits}
        if not used_q.issubset(phys_set):
            raise RuntimeError(f"Circuit {qc.name} leaked outside pinned 8Q tree: {sorted(used_q - phys_set)}")
        meas_phys = tuple(
            qc.find_bit(inst.qubits[0]).index
            for inst in sorted(
                (i for i in qc.data if i.operation.name == "measure"),
                key=lambda i: qc.find_bit(i.clbits[0]).index,
            )
        )
        if meas_phys != READOUT_PHYS:
            raise RuntimeError(f"Readout transmons drifted on {qc.name}: {meas_phys} != {READOUT_PHYS}")
        label = str(meta["arm"]) if meta["kind"] == "tomography" else "spam_cal"
        cz_stats.setdefault(label, []).append(int(qc.count_ops().get("cz", 0)))
        depth_stats.setdefault(label, []).append(int(qc.depth()))

        if meta["kind"] == "tomography":
            bars = [i for i, inst in enumerate(qc.data) if inst.operation.name == "barrier"]
            if len(bars) != 3:
                raise RuntimeError(f"Expected 3 barriers on {qc.name}, got {len(bars)}")
            scram_phys = {qc.find_bit(q).index for inst in qc.data[bars[0] + 1 : bars[1]] for q in inst.qubits}
            dec_phys = {qc.find_bit(q).index for inst in qc.data[bars[1] + 1 : bars[2]] for q in inst.qubits}
            if not scram_phys.isdisjoint({LAYOUT[0], LAYOUT[1]}):
                raise RuntimeError(f"Reference qubits touched during scrambling in {qc.name}: {scram_phys}")
            if not dec_phys.isdisjoint({LAYOUT[0], LAYOUT[1]}):
                raise RuntimeError(f"Reference qubits touched during decoding in {qc.name}: {dec_phys}")
            if label in ("Arm B", "Arm B-Ctrl", "Arm B-Held"):
                allowed = {LAYOUT[3], LAYOUT[4], LAYOUT[5], LAYOUT[6]}
                if not dec_phys.issubset(allowed):
                    raise RuntimeError(f"Remnant light-cone violation in {qc.name}: {dec_phys - allowed}")
            elif label == "Arm B-Def":
                allowed = {LAYOUT[3], LAYOUT[4], LAYOUT[5]}
                if not dec_phys.issubset(allowed):
                    raise RuntimeError(f"Sub-capacity light-cone violation in {qc.name}: {dec_phys - allowed}")

    return {
        k: {
            "count": len(cz_stats[k]),
            "cz_min": min(cz_stats[k]),
            "cz_max": max(cz_stats[k]),
            "depth_min": min(depth_stats[k]),
            "depth_max": max(depth_stats[k]),
        }
        for k in cz_stats
    }


def stage_and_optionally_submit(submit: bool) -> Optional[str]:
    svc = _connect_service()
    usage_before = svc.usage()
    rem_sec = float(usage_before.get("usage_remaining_seconds", 0))
    if rem_sec < 180:
        raise RuntimeError(f"Pre-flight abort: usage_remaining_seconds={rem_sec} < 180s minimum")

    backend = svc.backend(BACKEND_NAME)
    status = backend.status()
    if not status.operational:
        raise RuntimeError(f"Pre-flight abort: {BACKEND_NAME} operational={status.operational}")

    props = backend.properties()
    tgt = backend.target

    # Snapshot calibration & enforce pre-flight hardware quality gates
    tree_phys_edges = sorted({tuple(sorted((LAYOUT[a], LAYOUT[b]))) for a, b in hp004.TREE_EDGES_8Q})
    edge_cal: Dict[str, Dict[str, float]] = {}
    for u, v in tree_phys_edges:
        pair = (u, v) if (u, v) in tgt["cz"] else (v, u)
        p = tgt["cz"][pair]
        err = float(p.error)
        dur = float(p.duration)
        if err > 0.02:
            raise RuntimeError(f"Pre-flight abort: CZ error on {pair} is {err:.5f} > 0.02")
        edge_cal[f"{u}-{v}"] = {"cz_error": err, "duration_s": dur}

    qubit_cal: Dict[str, Dict[str, float]] = {}
    for q in LAYOUT:
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

    raw_circuits, metadata = hp004.assemble_hp004_schedule(n_blocks=N_BLOCKS, seed=SCHEDULE_SEED)
    isa_circuits = transpile(
        raw_circuits,
        target=tgt,
        initial_layout=list(LAYOUT),
        routing_method="none",
        optimization_level=1,
        seed_transpiler=42,
    )
    validate_isa_circuits(isa_circuits, tgt)
    gate_audit = _verify_compiled_light_cones(isa_circuits, metadata)

    qpy_path = EXPERIMENT_DIR / "hp004_staged_circuits.qpy"
    with open(qpy_path, "wb") as fh:
        qpy.dump(isa_circuits, fh)
    qpy_sha256 = hashlib.sha256(qpy_path.read_bytes()).hexdigest()

    cal_snapshot = {
        "benchmark_id": BENCHMARK_ID,
        "captured_at_utc": _utc_now(),
        "backend": backend.name,
        "pending_jobs": int(status.pending_jobs),
        "usage_before": {
            k: v for k, v in usage_before.items() if k != "instance_id"
        },
        "pinned_tree_layout": list(LAYOUT),
        "readout_transmons_c0_c1_c2_c3": list(READOUT_PHYS),
        "edge_calibration": edge_cal,
        "qubit_calibration": qubit_cal,
    }
    (EXPERIMENT_DIR / "hp004_calibration_snapshot.json").write_text(
        json.dumps(cal_snapshot, indent=2), encoding="utf-8"
    )

    manifest = {
        "benchmark_id": BENCHMARK_ID,
        "staged_at_utc": _utc_now(),
        "backend": backend.name,
        "pinned_tree_layout": list(LAYOUT),
        "readout_transmons_c0_c1_c2_c3": list(READOUT_PHYS),
        "n_blocks": N_BLOCKS,
        "shots_per_circuit": SHOTS_PER_CIRCUIT,
        "total_circuits": len(isa_circuits),
        "total_shots": len(isa_circuits) * SHOTS_PER_CIRCUIT,
        "schedule_seed": SCHEDULE_SEED,
        "qpy_sha256": qpy_sha256,
        "gate_audit": gate_audit,
        "circuit_metadata": metadata,
    }
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
    manifest_path = EXPERIMENT_DIR / "hp004_staging_manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    print(f"[STAGED] {len(isa_circuits)} ISA circuits verified | QPY SHA-256: {qpy_sha256[:16]}... | Manifest SHA-256: {manifest_sha256[:16]}...")
    if not submit:
        return None

    sampler = SamplerV2(mode=backend)
    job = sampler.run(isa_circuits, shots=SHOTS_PER_CIRCUIT)
    job_id = job.job_id()
    sub_record = {
        "benchmark_id": BENCHMARK_ID,
        "job_id": job_id,
        "backend": backend.name,
        "submitted_at_utc": _utc_now(),
        "qpy_sha256": qpy_sha256,
        "manifest_sha256": manifest_sha256,
        "shots_per_circuit": SHOTS_PER_CIRCUIT,
        "total_circuits": len(isa_circuits),
        "total_shots": len(isa_circuits) * SHOTS_PER_CIRCUIT,
    }
    (EXPERIMENT_DIR / "hp004_submitted_job.json").write_text(
        json.dumps(sub_record, indent=2), encoding="utf-8"
    )
    print(f"[SUBMITTED] IBM Runtime Job ID: {job_id} on {backend.name}")
    return job_id


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


def _bootstrap_ci_joint(pooled_counts: Mapping[str, Mapping[str, int]], n_boot: int = 5000, seed: int = 42) -> Tuple[float, float]:
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
        # Scale to virtual integer counts preserving total shot weight for parity expectations
        mit_counts[bkey] = {format(i, "04b"): max(1, int(round(x[i] * tot * 100))) for i in range(16) if x[i] * tot * 100 >= 0.5}
    est = hp004.estimate_joint_bell_fidelity(mit_counts)
    return {
        "f1_spam_mitigated": est["f1_marginal"],
        "f2_spam_mitigated": est["f2_marginal"],
        "f_joint_spam_mitigated": est["f_joint"],
    }


def harvest_job(job_id: str, poll_seconds: int = 10) -> Dict[str, Any]:
    svc = _connect_service()
    manifest = json.loads((EXPERIMENT_DIR / "hp004_staging_manifest.json").read_text(encoding="utf-8"))
    metadata: List[Dict[str, Any]] = manifest["circuit_metadata"]

    job = svc.job(job_id)
    while True:
        st = str(job.status())
        print(f"[{_utc_now()}] Job {job_id} status: {st}", flush=True)
        if st in ("DONE", "ERROR", "CANCELLED"):
            break
        time.sleep(poll_seconds)

    if str(job.status()) != "DONE":
        raise RuntimeError(f"Job {job_id} ended with non-DONE status: {job.status()}")

    result = job.result()
    if len(result) != len(metadata):
        raise RuntimeError(f"PubResult count {len(result)} != staged metadata count {len(metadata)}")

    circuit_records: List[Dict[str, Any]] = []
    # Build 16x16 empirical assignment matrix M[measured_idx, prepared_idx]
    cal_matrix = np.zeros((16, 16), dtype=float)
    # Structure per-arm per-block counts
    arm_block_counts: Dict[str, List[Dict[str, Dict[str, int]]]] = {
        arm: [{} for _ in range(manifest["n_blocks"])] for arm in hp004.ARMS
    }

    for idx, meta in enumerate(metadata):
        counts = _extract_counts(result[idx])
        shots = sum(counts.values())
        if shots != manifest["shots_per_circuit"]:
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

    # Evaluate pre-registered hypotheses strictly on RAW counts
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
        "job_id": job_id,
        "backend": manifest["backend"],
        "pinned_tree_layout": manifest["pinned_tree_layout"],
        "readout_transmons_c0_c1_c2_c3": manifest["readout_transmons_c0_c1_c2_c3"],
        "qpy_sha256": manifest["qpy_sha256"],
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
    out_path = EXPERIMENT_DIR / "physical_hp004_results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[HARVESTED] Saved complete raw counts + analysis to {out_path}")
    return payload


def submit_frozen() -> str:
    """Load the already-staged, SHA-256-verified `.qpy` from disk and submit 1 SamplerV2 job."""
    qpy_path = EXPERIMENT_DIR / "hp004_staged_circuits.qpy"
    manifest_path = EXPERIMENT_DIR / "hp004_staging_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    actual_qpy_sha256 = hashlib.sha256(qpy_path.read_bytes()).hexdigest()
    if actual_qpy_sha256 != manifest["qpy_sha256"]:
        raise RuntimeError(f"Frozen QPY SHA-256 mismatch: {actual_qpy_sha256} != {manifest['qpy_sha256']}")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    with open(qpy_path, "rb") as fh:
        isa_circuits = qpy.load(fh)
    if len(isa_circuits) != manifest["total_circuits"]:
        raise RuntimeError("Circuit count mismatch in frozen QPY")

    svc = _connect_service()
    usage_before = svc.usage()
    rem_sec = float(usage_before.get("usage_remaining_seconds", 0))
    if rem_sec < 180:
        raise RuntimeError(f"Pre-flight abort: usage_remaining_seconds={rem_sec} < 180s minimum")

    backend = svc.backend(manifest["backend"])
    st = backend.status()
    if not st.operational:
        raise RuntimeError(f"Pre-flight abort: {backend.name} operational={st.operational}")
    validate_isa_circuits(isa_circuits, backend.target)

    sampler = SamplerV2(mode=backend)
    job = sampler.run(isa_circuits, shots=int(manifest["shots_per_circuit"]))
    job_id = job.job_id()
    sub_record = {
        "benchmark_id": BENCHMARK_ID,
        "job_id": job_id,
        "backend": backend.name,
        "submitted_at_utc": _utc_now(),
        "qpy_sha256": actual_qpy_sha256,
        "manifest_sha256": manifest_sha256,
        "shots_per_circuit": int(manifest["shots_per_circuit"]),
        "total_circuits": len(isa_circuits),
        "total_shots": int(manifest["total_shots"]),
    }
    (EXPERIMENT_DIR / "hp004_submitted_job.json").write_text(
        json.dumps(sub_record, indent=2), encoding="utf-8"
    )
    print(f"[SUBMITTED FROZEN QPY] Job ID: {job_id} on {backend.name} | QPY SHA-256: {actual_qpy_sha256}")
    return job_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--submit-and-harvest", action="store_true")
    parser.add_argument("--submit-frozen-and-harvest", action="store_true")
    parser.add_argument("--harvest-job", type=str, default=None)
    args = parser.parse_args()

    if args.harvest_job:
        res = harvest_job(args.harvest_job)
    elif args.stage_only:
        stage_and_optionally_submit(submit=False)
        return
    elif args.submit_frozen_and_harvest:
        jid = submit_frozen()
        res = harvest_job(jid)
    elif args.submit_and_harvest:
        jid = stage_and_optionally_submit(submit=True)
        assert jid is not None
        res = harvest_job(jid)
    else:
        parser.error("Specify --stage-only, --submit-frozen-and-harvest, --submit-and-harvest, or --harvest-job <JOB_ID>")
        return

    print("\n=== OSIRIS-HP-004 PHYSICAL HARDWARE SUMMARY ===")
    for arm, s in res["arm_summaries"].items():
        sec = s["secondary_spam_sensitivity"]["f_joint_spam_mitigated"]
        print(
            f"  {arm:10s}: Raw F_joint = {s['f_joint']:.4f} +/- {s['f_joint_se_eff']:.4f} "
            f"(F1={s['f1_marginal']:.4f}, F2={s['f2_marginal']:.4f}, Z_>0.5={s['z_above_1ebit_eff']:+.2f}s | SPAM-sec={sec:.4f})"
        )
    print("  Pre-registered hypotheses:", {k: v["passed"] for k, v in res["preregistered_hypotheses"].items()})


if __name__ == "__main__":
    main()
