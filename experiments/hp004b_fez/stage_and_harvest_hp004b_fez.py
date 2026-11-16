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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-only", action="store_true", default=True)
    parser.parse_args()
    stage_fez()


if __name__ == "__main__":
    main()
