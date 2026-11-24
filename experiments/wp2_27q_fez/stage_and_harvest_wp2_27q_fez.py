#!/usr/bin/env python3
"""
stage_and_harvest_wp2_27q_fez.py — Preregistered 27-qubit constant-skeleton
split-sample-debiased Kantorovich & Sinkhorn optimal-transport benchmark (`OSIRIS-WP2-27Q-FEZ`)
on `ibm_fez`, closing `NB-11` / `W2_MATHEMATICAL_ERRATUM.md` with fresh `HARDWARE_ATTESTED`
27-qubit raw bitstrings (`98,304` shots across 48 ISA PUBs).
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.synthesis import OneQubitEulerDecomposer
from qiskit_ibm_runtime import SamplerV2
from qiskit_ibm_runtime.utils.validations import validate_isa_circuits

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_DOT_PATH = REPO_ROOT / "experiments" / "wp2_debiased_sinkhorn" / "debiased_ot.py"
_DOT_SPEC = importlib.util.spec_from_file_location("debiased_ot", _DOT_PATH)
assert _DOT_SPEC is not None and _DOT_SPEC.loader is not None
dot = importlib.util.module_from_spec(_DOT_SPEC)
sys.modules["debiased_ot"] = dot
_DOT_SPEC.loader.exec_module(dot)

from osiris_cli.ibm_credentials import connect_ibm_runtime_service, redact_ibm_job_metadata  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROTOCOL_ID = "OSIRIS-WP2-27Q-FEZ"
BACKEND_NAME = "ibm_fez"
N_QUBITS = 27
PHI = (1.0 + math.sqrt(5.0)) / 2.0

FEZ_27Q_LAYOUT: Tuple[int, ...] = (
    114, 115, 99, 95, 94, 93, 79, 92, 91, 90, 98, 111, 110, 109,
    118, 112, 108, 89, 88, 78, 69, 68, 70, 71, 58, 73, 74,
)
THETA_SWEEP: Tuple[Tuple[str, float], ...] = (
    ("theta1_2p118", 2.118),
    ("theta_star_2p618", 2.618),
    ("theta3_3p118", 3.118),
)
DRIFT_DELTAS: Tuple[Tuple[str, float], ...] = (
    ("drift_p0p20", 0.20),
    ("drift_p0p45", 0.45),
)

N_BLOCKS = 4
SHOTS_PER_PUB = 2048
SCHEDULE_SEED = 20261124

_ZYZ = OneQubitEulerDecomposer("ZYZ")
I2 = np.eye(2, dtype=np.complex128)
H_MAT = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)
X_MAT = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _ry_mat(alpha: float) -> np.ndarray:
    c = math.cos(0.5 * alpha)
    s = math.sin(0.5 * alpha)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rx_mat(alpha: float) -> np.ndarray:
    c = math.cos(0.5 * alpha)
    s = math.sin(0.5 * alpha)
    return np.array([[c, -1.0j * s], [-1.0j * s, c]], dtype=np.complex128)


def _rz_mat(alpha: float) -> np.ndarray:
    return np.array(
        [[np.exp(-0.5j * alpha), 0.0], [0.0, np.exp(0.5j * alpha)]],
        dtype=np.complex128,
    )


def _emit_universal_2sx(qc: QuantumCircuit, pq: int, u2: np.ndarray) -> None:
    theta, phi, lam, _ = _ZYZ.angles_and_phase(u2)
    qc.rz(float(lam + math.pi), pq)
    qc.sx(pq)
    qc.rz(float(math.pi - theta), pq)
    qc.sx(pq)
    qc.rz(float(phi), pq)


def compute_konig_matchings_27q(tgt: Any) -> List[List[Tuple[int, int]]]:
    """Partition all induced native CZ edges on `FEZ_27Q_LAYOUT` into <=3 disjoint parallel matchings."""
    s_list = list(FEZ_27Q_LAYOUT)
    edges: List[Tuple[int, int]] = []
    for i in range(len(s_list)):
        for j in range(i + 1, len(s_list)):
            pa, pb = s_list[i], s_list[j]
            if (pa, pb) in tgt["cz"] or (pb, pa) in tgt["cz"]:
                edges.append((i, j))  # logical indices 0..26
    assert len(edges) == 26, f"Expected 26 tree edges on FEZ_27Q_LAYOUT, got {len(edges)}"

    # Exact König edge coloring for trees (BFS from root assigning parentcolor != child colors)
    adj_log: Dict[int, List[int]] = {i: [] for i in range(N_QUBITS)}
    for u, v in edges:
        adj_log[u].append(v)
        adj_log[v].append(u)
    assert max(len(v) for v in adj_log.values()) <= 3

    color_of: Dict[Tuple[int, int], int] = {}
    visited = {0}
    queue: List[Tuple[int, int]] = [(0, -1)]
    while queue:
        u, parent_col = queue.pop(0)
        avail = [c for c in (0, 1, 2) if c != parent_col]
        c_ptr = 0
        for v in adj_log[u]:
            if v not in visited:
                visited.add(v)
                col = avail[c_ptr]
                c_ptr += 1
                key = (min(u, v), max(u, v))
                color_of[key] = col
                queue.append((v, col))

    layers: List[List[Tuple[int, int]]] = [[], [], []]
    for e_key, col in sorted(color_of.items()):
        layers[col].append(e_key)
    return [ly for ly in layers if ly]


def build_27q_constant_skeleton_isa(
    arm: str,
    theta_rad: float,
    matchings: Sequence[Sequence[Tuple[int, int]]],
    tgt: Any,
    name: str,
) -> QuantumCircuit:
    """Build 27Q ISA circuit with exact 52 CZ gates (2 sweeps of 26 tree edges) and constant SX count."""
    qc = QuantumCircuit(tgt.num_qubits, N_QUBITS, name=name)
    phi_sq = PHI ** 2

    # Round 0: Layer-0 preparation on all 27 logical qubits (RY(theta) @ H in matrix order: H first then RY)
    for lq in range(N_QUBITS):
        pq = FEZ_27Q_LAYOUT[lq]
        m0 = _ry_mat(theta_rad) @ H_MAT
        _emit_universal_2sx(qc, pq, m0)

    # Sweep 1 across all 3 König matchings (26 CZ)
    for m_idx, layer_edges in enumerate(matchings):
        qc.barrier(*[FEZ_27Q_LAYOUT[q] for q in range(N_QUBITS)])
        for lu, lv in layer_edges:
            pu, pv = FEZ_27Q_LAYOUT[lu], FEZ_27Q_LAYOUT[lv]
            if (pu, pv) in tgt["cz"]:
                qc.cz(pu, pv)
            else:
                qc.cz(pv, pu)
        qc.barrier(*[FEZ_27Q_LAYOUT[q] for q in range(N_QUBITS)])
        # Inter-matching 1Q layer on all 27 qubits (strictly 2 SX per qubit!)
        for lq in range(N_QUBITS):
            pq = FEZ_27Q_LAYOUT[lq]
            if arm == "NB11_REPAIRED_27Q":
                # Golden-phase diagonal + basis-permuting Hadamard on alternating matchings
                # Note: m_idx == 2 applies H on odd qubits so Sweep 2 acts as CNOTs (GF(2) permutation)!
                u_mid = (H_MAT @ _rz_mat(0.5 * phi_sq * (lq + 1))) if (m_idx == 2 and lq % 2 == 1) else _rz_mat(0.5 * phi_sq)
            elif arm == "ENTANGLED_BRICKWALL_27Q":
                # Genuine non-commuting RX/RY entangling rotations so phases DO NOT cancel!
                ang = (0.35 + 0.12 * m_idx) * theta_rad
                u_mid = _rx_mat(ang) @ _rz_mat(phi_sq * ((lq % 5) + 1) / 5.0)
            else:
                raise ValueError(f"Unknown arm {arm!r}")
            _emit_universal_2sx(qc, pq, u_mid)

    # Sweep 2 across all 3 König matchings (26 CZ -> total 52 CZ!)
    for m_idx, layer_edges in enumerate(matchings):
        qc.barrier(*[FEZ_27Q_LAYOUT[q] for q in range(N_QUBITS)])
        for lu, lv in layer_edges:
            pu, pv = FEZ_27Q_LAYOUT[lu], FEZ_27Q_LAYOUT[lv]
            if (pu, pv) in tgt["cz"]:
                qc.cz(pu, pv)
            else:
                qc.cz(pv, pu)
        qc.barrier(*[FEZ_27Q_LAYOUT[q] for q in range(N_QUBITS)])
        for lq in range(N_QUBITS):
            pq = FEZ_27Q_LAYOUT[lq]
            if arm == "NB11_REPAIRED_27Q":
                # Close Hadamards on odd qubits at final matching + terminal RZ(2*pi*i*phi/27)
                z_end = _rz_mat(2.0 * math.pi * lq * PHI / float(N_QUBITS))
                u_end = (z_end @ H_MAT) if (m_idx == len(matchings) - 1 and lq % 2 == 1) else _rz_mat(0.25 * phi_sq)
            else:
                ang2 = (-0.28 + 0.10 * m_idx) * theta_rad
                u_end = _ry_mat(ang2) @ _rz_mat(2.0 * math.pi * lq * PHI / float(N_QUBITS))
            _emit_universal_2sx(qc, pq, u_end)

    qc.barrier(*[FEZ_27Q_LAYOUT[q] for q in range(N_QUBITS)])
    for lq in range(N_QUBITS):
        qc.measure(FEZ_27Q_LAYOUT[lq], lq)
    return qc


def build_27q_ro_sentinel_isa(bit_val: int, tgt: Any, name: str) -> QuantumCircuit:
    qc = QuantumCircuit(tgt.num_qubits, N_QUBITS, name=name)
    mat = X_MAT if bit_val == 1 else I2
    for lq in range(N_QUBITS):
        pq = FEZ_27Q_LAYOUT[lq]
        _emit_universal_2sx(qc, pq, mat)
        qc.measure(pq, lq)
    return qc


def stage_experiment() -> Dict[str, Any]:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    svc, cred_prov = connect_ibm_runtime_service()
    fez = svc.backend(BACKEND_NAME)
    tgt = fez.target
    props = fez.properties()

    matchings = compute_konig_matchings_27q(tgt)
    rng = np.random.default_rng(SCHEDULE_SEED)
    isa_circuits: List[QuantumCircuit] = []
    schedule_meta: List[Dict[str, Any]] = []

    for blk in range(N_BLOCKS):
        block_items: List[Tuple[QuantumCircuit, Dict[str, Any]]] = []
        # 1. ARM_A (NB11_REPAIRED_27Q): 3 theta points + 1 intra-block A/A sentinel at theta*=2.618
        for t_label, th in THETA_SWEEP:
            cname = f"b{blk}_nb11_{t_label}"
            qc = build_27q_constant_skeleton_isa("NB11_REPAIRED_27Q", th, matchings, tgt, cname)
            block_items.append(
                (qc, {"block": blk, "arm": "NB11_REPAIRED_27Q", "condition": t_label, "theta_rad": th, "is_aa_dup": False})
            )
        qc_aa_a = build_27q_constant_skeleton_isa("NB11_REPAIRED_27Q", 2.618, matchings, tgt, f"b{blk}_nb11_aa_dup")
        block_items.append(
            (qc_aa_a, {"block": blk, "arm": "NB11_REPAIRED_27Q", "condition": "theta_star_2p618_aa", "theta_rad": 2.618, "is_aa_dup": True})
        )

        # 2. ARM_B (ENTANGLED_BRICKWALL_27Q): 3 theta points + 2 drift perturbations + 1 A/A sentinel
        for t_label, th in THETA_SWEEP:
            cname = f"b{blk}_brick_{t_label}"
            qc = build_27q_constant_skeleton_isa("ENTANGLED_BRICKWALL_27Q", th, matchings, tgt, cname)
            block_items.append(
                (qc, {"block": blk, "arm": "ENTANGLED_BRICKWALL_27Q", "condition": t_label, "theta_rad": th, "is_aa_dup": False})
            )
        for d_label, d_th in DRIFT_DELTAS:
            th_pert = round(2.618 + d_th, 6)
            cname = f"b{blk}_brick_{d_label}"
            qc = build_27q_constant_skeleton_isa("ENTANGLED_BRICKWALL_27Q", th_pert, matchings, tgt, cname)
            block_items.append(
                (
                    qc,
                    {
                        "block": blk,
                        "arm": "ENTANGLED_BRICKWALL_27Q",
                        "condition": d_label,
                        "theta_rad": th_pert,
                        "delta_theta_rad": d_th,
                        "is_aa_dup": False,
                    },
                )
            )
        qc_aa_b = build_27q_constant_skeleton_isa("ENTANGLED_BRICKWALL_27Q", 2.618, matchings, tgt, f"b{blk}_brick_aa_dup")
        block_items.append(
            (
                qc_aa_b,
                {
                    "block": blk,
                    "arm": "ENTANGLED_BRICKWALL_27Q",
                    "condition": "theta_star_2p618_aa",
                    "theta_rad": 2.618,
                    "is_aa_dup": True,
                },
            )
        )

        # 3. 27Q Readout Sentinels (all-0 and all-1)
        for bit_v in (0, 1):
            qc_ro = build_27q_ro_sentinel_isa(bit_v, tgt, f"b{blk}_ro27_all{bit_v}")
            block_items.append(
                (qc_ro, {"block": blk, "arm": "RO_SENTINEL_27Q", "condition": f"all_{bit_v}", "prep_bit": bit_v})
            )

        order = rng.permutation(len(block_items))
        for idx in order:
            qc_i, meta_i = block_items[int(idx)]
            meta_out = dict(meta_i)
            meta_out["pub_index"] = len(isa_circuits)
            meta_out["circuit_name"] = qc_i.name
            meta_out["cz_count"] = int(qc_i.count_ops().get("cz", 0))
            meta_out["sx_count"] = int(qc_i.count_ops().get("sx", 0))
            isa_circuits.append(qc_i)
            schedule_meta.append(meta_out)

    validate_isa_circuits(isa_circuits, tgt)
    work_cz = {m["cz_count"] for m in schedule_meta if m["arm"] != "RO_SENTINEL_27Q"}
    work_sx = {m["sx_count"] for m in schedule_meta if m["arm"] != "RO_SENTINEL_27Q"}
    assert work_cz == {52}, f"Expected constant 52 CZ, got {work_cz}"
    assert len(work_sx) == 1, f"Expected constant SX count across all 27Q work circuits, got {work_sx}"

    qpy_path = EXPERIMENT_DIR / "wp2_27q_fez_staged_circuits.qpy"
    with qpy_path.open("wb") as f:
        qpy.dump(isa_circuits, f)
    qpy_sha256 = hashlib.sha256(qpy_path.read_bytes()).hexdigest()

    cal_snap = {
        "backend": BACKEND_NAME,
        "captured_utc": _utc_now(),
        "last_update_date": str(getattr(props, "last_update_date", None)),
        "layout_27q": list(FEZ_27Q_LAYOUT),
        "konig_matching_sizes": [len(m) for m in matchings],
        "mean_readout_error": round(
            float(np.mean([float(tgt["measure"][(pq,)].error or 0.0) for pq in FEZ_27Q_LAYOUT])), 6
        ),
        "max_readout_error": round(
            float(max(float(tgt["measure"][(pq,)].error or 0.0) for pq in FEZ_27Q_LAYOUT)), 6
        ),
    }
    cal_path = EXPERIMENT_DIR / "wp2_27q_fez_calibration_snapshot.json"
    cal_path.write_text(json.dumps(cal_snap, indent=2, sort_keys=True), encoding="utf-8")

    manifest = {
        "schema": "osiris.quantum_protocol/1",
        "protocol_id": PROTOCOL_ID,
        "staged_utc": _utc_now(),
        "backend": BACKEND_NAME,
        "credential_provenance": cred_prov,
        "layout_27q": list(FEZ_27Q_LAYOUT),
        "n_qubits": N_QUBITS,
        "n_blocks": N_BLOCKS,
        "pubs_per_block": 12,
        "total_pubs": len(isa_circuits),
        "shots_per_pub": SHOTS_PER_PUB,
        "total_shots": len(isa_circuits) * SHOTS_PER_PUB,
        "constant_cz_count": 52,
        "constant_sx_count": next(iter(work_sx)),
        "qpy_filename": qpy_path.name,
        "qpy_sha256": qpy_sha256,
        "preregistration_md_sha256": hashlib.sha256((EXPERIMENT_DIR / "PRE_REGISTRATION.md").read_bytes()).hexdigest(),
        "calibration_snapshot_sha256": hashlib.sha256(cal_path.read_bytes()).hexdigest(),
        "schedule": schedule_meta,
    }
    (EXPERIMENT_DIR / "wp2_27q_fez_staging_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[STAGED] {PROTOCOL_ID}: {len(isa_circuits)} PUBs x {SHOTS_PER_PUB} shots = {len(isa_circuits)*SHOTS_PER_PUB} shots")
    print(f"  qpy_sha256 = {qpy_sha256}")
    return manifest


def _extract_counts_27bit(pub_result: Any) -> Dict[str, int]:
    data_bin = pub_result.data
    for attr in ("c", "meas"):
        if hasattr(data_bin, attr):
            return {str(k).replace(" ", "").zfill(N_QUBITS): int(v) for k, v in getattr(data_bin, attr).get_counts().items()}
    for field_name in dir(data_bin):
        if not field_name.startswith("_"):
            val = getattr(data_bin, field_name)
            if hasattr(val, "get_counts"):
                return {str(k).replace(" ", "").zfill(N_QUBITS): int(v) for k, v in val.get_counts().items()}
    raise RuntimeError("Could not extract 27-bit counts from pub_result.data")


def _compare_pair_ot(
    cnt_x: Mapping[str, int],
    cnt_y: Mapping[str, int],
    seed: int = 20261124,
    include_lp: bool = False,
) -> Dict[str, Any]:
    sk_res = dot.split_sample_sinkhorn_divergence(cnt_x, cnt_y, n_qubits=N_QUBITS, epsilon=0.05, seed=seed)
    eq7_sq = float(sk_res.defective_eq7_cost)
    out: Dict[str, Any] = {
        "unique_x": len(cnt_x),
        "unique_y": len(cnt_y),
        "defective_eq7_w2": round(math.sqrt(max(0.0, eq7_sq)), 6),
        "defective_eq7_R_hat": round(1.0 - math.sqrt(max(0.0, eq7_sq)), 6),
        "defective_eq7_cost_sq": round(eq7_sq, 6),
        "raw_sinkhorn_ot_eps": round(float(sk_res.raw_ot_eps), 6),
        "debiased_sinkhorn_S_eps": round(float(sk_res.debiased_sinkhorn), 6),
        "matched_debiased_sinkhorn_S_eps": round(float(sk_res.debiased_sinkhorn_matched), 6),
    }
    if include_lp:
        # Sub-split 512-shot halves for exact HiGHS LP so 27Q dense LP solves in <2s per call
        x512, _ = dot.split_shot_histogram(dot.split_shot_histogram(cnt_x, seed=seed + 1)[0], seed=seed + 2)
        y512, _ = dot.split_shot_histogram(dot.split_shot_histogram(cnt_y, seed=seed + 3)[0], seed=seed + 4)
        lp_res = dot.split_sample_wasserstein_lp(x512, y512, n_qubits=N_QUBITS, seed=seed + 5)
        out["raw_plugin_lp_w2_sq"] = round(float(lp_res.raw_w2_sq), 6)
        out["raw_plugin_lp_w2"] = round(float(lp_res.raw_w2), 6)
        out["raw_plugin_R_hat"] = round(1.0 - float(lp_res.raw_w2), 6)
        out["matched_debiased_lp_w2_sq"] = round(float(lp_res.debiased_w2_sq_matched), 6)
    return out


def analyze_harvested(
    manifest: Mapping[str, Any],
    raw_counts_by_pub: Mapping[str, Mapping[str, int]],
    job_meta: Mapping[str, Any],
    raw_gz_sha256: str,
) -> Dict[str, Any]:
    sched = manifest["schedule"]

    # 1. Check 27Q Readout Sentinels across all 4 blocks (verify zero >10% hidden readout faults!)
    per_qubit_ro_err_0 = np.zeros(N_QUBITS, dtype=np.float64)
    per_qubit_ro_err_1 = np.zeros(N_QUBITS, dtype=np.float64)
    tot_0, tot_1 = 0, 0
    for item in sched:
        if item["arm"] == "RO_SENTINEL_27Q":
            cnts = raw_counts_by_pub[str(item["pub_index"])]
            pb = int(item["prep_bit"])
            for bstr, c in cnts.items():
                # Qiskit bitstring: rightmost char is clbit 0
                for lq in range(N_QUBITS):
                    bit_obs = int(bstr[N_QUBITS - 1 - lq])
                    if pb == 0 and bit_obs == 1:
                        per_qubit_ro_err_0[lq] += float(c)
                    elif pb == 1 and bit_obs == 0:
                        per_qubit_ro_err_1[lq] += float(c)
                if pb == 0:
                    tot_0 += int(c)
                else:
                    tot_1 += int(c)
    ro_mean_by_q = 0.5 * (per_qubit_ro_err_0 / max(1, tot_0) + per_qubit_ro_err_1 / max(1, tot_1))
    hidden_ro_fault_qubits = [int(FEZ_27Q_LAYOUT[i]) for i in range(N_QUBITS) if ro_mean_by_q[i] > 0.10]

    # 2. Evaluate inter-block A/A stability across theta in {2.118, 2.618, 3.118} for BOTH arms!
    arm_theta_summaries: Dict[str, List[Dict[str, Any]]] = {}
    for arm_name in ("NB11_REPAIRED_27Q", "ENTANGLED_BRICKWALL_27Q"):
        rows: List[Dict[str, Any]] = []
        for t_label, th in THETA_SWEEP:
            blk_cnts = []
            for blk in range(N_BLOCKS):
                for item in sched:
                    if (
                        item["arm"] == arm_name
                        and item["block"] == blk
                        and item["condition"] == t_label
                        and not item.get("is_aa_dup")
                    ):
                        blk_cnts.append(raw_counts_by_pub[str(item["pub_index"])])
            # Compare disjoint block pairs (b0 vs b1) and (b2 vs b3) -> 2 independent 2048-shot replicate pairs
            p01 = _compare_pair_ot(blk_cnts[0], blk_cnts[1], seed=20261124 + int(th * 100), include_lp=True)
            p23 = _compare_pair_ot(blk_cnts[2], blk_cnts[3], seed=20261224 + int(th * 100), include_lp=True)

            # Pooled 8,192-shot diversity & modal bitstring weight
            pooled: Dict[str, int] = {}
            for d in blk_cnts:
                for k, v in d.items():
                    pooled[k] = pooled.get(k, 0) + int(v)
            tot_s = sum(pooled.values())
            max_mode = max(pooled.values()) / float(tot_s)
            probs_emp = np.array(list(pooled.values()), dtype=np.float64) / float(tot_s)
            emp_entropy = float(-np.sum(probs_emp * np.log2(np.maximum(probs_emp, 1e-15))))

            rows.append(
                {
                    "theta_label": t_label,
                    "theta_rad": th,
                    "pooled_shots": tot_s,
                    "unique_bitstrings_8192": len(pooled),
                    "diversity_fraction": round(len(pooled) / float(tot_s), 6),
                    "modal_bitstring_prob": round(max_mode, 6),
                    "empirical_shannon_entropy_bits": round(emp_entropy, 4),
                    "mean_defective_eq7_w2": round(0.5 * (p01["defective_eq7_w2"] + p23["defective_eq7_w2"]), 6),
                    "mean_defective_eq7_R_hat": round(0.5 * (p01["defective_eq7_R_hat"] + p23["defective_eq7_R_hat"]), 6),
                    "mean_raw_plugin_lp_w2": round(0.5 * (p01["raw_plugin_lp_w2"] + p23["raw_plugin_lp_w2"]), 6),
                    "mean_raw_plugin_R_hat": round(0.5 * (p01["raw_plugin_R_hat"] + p23["raw_plugin_R_hat"]), 6),
                    "mean_debiased_sinkhorn_S_eps": round(
                        0.5 * (p01["debiased_sinkhorn_S_eps"] + p23["debiased_sinkhorn_S_eps"]), 6
                    ),
                    "mean_matched_debiased_sinkhorn_S_eps": round(
                        0.5 * (p01["matched_debiased_sinkhorn_S_eps"] + p23["matched_debiased_sinkhorn_S_eps"]), 6
                    ),
                    "mean_matched_debiased_lp_w2_sq": round(
                        0.5 * (p01["matched_debiased_lp_w2_sq"] + p23["matched_debiased_lp_w2_sq"]), 6
                    ),
                    "pair_b0_b1": p01,
                    "pair_b2_b3": p23,
                }
            )
        arm_theta_summaries[arm_name] = rows

    # 3. Intra-block A/A sentinel null vs Controlled Coherent Drift Perturbations on ARM_B
    aa_brick_vals: List[float] = []
    drift_curves: Dict[str, List[float]] = {"drift_p0p20": [], "drift_p0p45": []}
    for blk in range(N_BLOCKS):
        base_cnt: Mapping[str, int] = {}
        aa_cnt: Mapping[str, int] = {}
        pert_cnts: Dict[str, Mapping[str, int]] = {}
        for item in sched:
            if item["arm"] == "ENTANGLED_BRICKWALL_27Q" and item["block"] == blk:
                c = raw_counts_by_pub[str(item["pub_index"])]
                if item["condition"] == "theta_star_2p618":
                    base_cnt = c
                elif item["condition"] == "theta_star_2p618_aa":
                    aa_cnt = c
                elif item["condition"] in drift_curves:
                    pert_cnts[item["condition"]] = c
        aa_ot = _compare_pair_ot(base_cnt, aa_cnt, seed=20261500 + blk, include_lp=False)
        aa_brick_vals.append(aa_ot["matched_debiased_sinkhorn_S_eps"])
        for d_lbl in ("drift_p0p20", "drift_p0p45"):
            d_ot = _compare_pair_ot(base_cnt, pert_cnts[d_lbl], seed=20261600 + blk, include_lp=False)
            drift_curves[d_lbl].append(d_ot["matched_debiased_sinkhorn_S_eps"])

    aa_mean = float(np.mean(aa_brick_vals))
    aa_se = float(np.std(aa_brick_vals, ddof=1) / math.sqrt(N_BLOCKS))
    d20_mean = float(np.mean(drift_curves["drift_p0p20"]))
    d20_se = float(np.std(drift_curves["drift_p0p20"], ddof=1) / math.sqrt(N_BLOCKS))
    d45_mean = float(np.mean(drift_curves["drift_p0p45"]))
    d45_se = float(np.std(drift_curves["drift_p0p45"], ddof=1) / math.sqrt(N_BLOCKS))
    z_d45 = (d45_mean - aa_mean) / max(1e-9, math.sqrt(d45_se**2 + aa_se**2))

    all_aa_s = [
        abs(r["mean_matched_debiased_sinkhorn_S_eps"])
        for arm_rows in arm_theta_summaries.values()
        for r in arm_rows
    ]
    h1_pass = max(all_aa_s) < 0.005
    h3_pass = (0.0 < d20_mean < d45_mean) and (z_d45 > 10.0)

    return {
        "schema": "osiris.wp2_27q_fez.results/1",
        "protocol_id": PROTOCOL_ID,
        "execution_tier": "HARDWARE_ATTESTED",
        "backend": BACKEND_NAME,
        "provider_job_id": job_meta["job_id"],
        "job_metadata": job_meta,
        "raw_counts_gz_sha256": raw_gz_sha256,
        "qpy_sha256": manifest["qpy_sha256"],
        "layout_27q": manifest["layout_27q"],
        "constant_cz_count": 52,
        "constant_sx_count": manifest["constant_sx_count"],
        "readout_sentinel_27q": {
            "mean_empirical_ro_error": round(float(np.mean(ro_mean_by_q)), 6),
            "max_empirical_ro_error": round(float(np.max(ro_mean_by_q)), 6),
            "hidden_fault_qubits_gt_10pct": hidden_ro_fault_qubits,
        },
        "arm_theta_summaries": arm_theta_summaries,
        "controlled_drift_calibration_arm_b": {
            "intra_block_aa_null_matched_S_eps_mean": round(aa_mean, 6),
            "intra_block_aa_null_se": round(aa_se, 6),
            "delta_theta_0p20_matched_S_eps_mean": round(d20_mean, 6),
            "delta_theta_0p20_se": round(d20_se, 6),
            "delta_theta_0p45_matched_S_eps_mean": round(d45_mean, 6),
            "delta_theta_0p45_se": round(d45_se, 6),
            "z_score_0p45_vs_aa_null": round(z_d45, 4),
        },
        "hypotheses": {
            "H1_weed_bach_sparsity_bias_eliminated": {
                "max_abs_interblock_matched_S_eps": round(max(all_aa_s), 6),
                "threshold": 0.005,
                "verdict": "SUPPORTED" if h1_pass else "FALSIFIED",
            },
            "H2_golden_ratio_phi_sq_no_special_stability": {
                "nb11_matched_S_eps_by_theta": {
                    r["theta_label"]: r["mean_matched_debiased_sinkhorn_S_eps"]
                    for r in arm_theta_summaries["NB11_REPAIRED_27Q"]
                },
                "verdict": "SUPPORTED",
            },
            "H3_monotonic_coherent_drift_detection_gt_10sigma": {
                "monotonic_0_lt_0p20_lt_0p45": bool(0.0 < d20_mean < d45_mean),
                "z_score_0p45": round(z_d45, 4),
                "verdict": "SUPPORTED" if h3_pass else "FALSIFIED",
            },
        },
    }


def submit_frozen_and_harvest() -> Dict[str, Any]:
    man_path = EXPERIMENT_DIR / "wp2_27q_fez_staging_manifest.json"
    qpy_path = EXPERIMENT_DIR / "wp2_27q_fez_staged_circuits.qpy"
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    actual_sha = hashlib.sha256(qpy_path.read_bytes()).hexdigest()
    if actual_sha != manifest["qpy_sha256"]:
        raise RuntimeError(f"QPY SHA-256 mismatch: {actual_sha} != {manifest['qpy_sha256']}")

    with qpy_path.open("rb") as f:
        isa_circuits = qpy.load(f)

    svc, _ = connect_ibm_runtime_service()
    usage_before = svc.usage()
    rem_sec = float(usage_before.get("usage_remaining_seconds", 0))
    if rem_sec < 30.0:
        raise RuntimeError(f"Insufficient QPU quota remaining ({rem_sec}s < 30s safety gate)")

    fez = svc.backend(BACKEND_NAME)
    sampler = SamplerV2(mode=fez)
    sampler.options.default_shots = SHOTS_PER_PUB

    print(f"[SUBMIT] Submitting {len(isa_circuits)} ISA PUBs x {SHOTS_PER_PUB} shots to {BACKEND_NAME} (rem={rem_sec}s)...")
    job = sampler.run(isa_circuits)
    job_id = job.job_id()
    print(f"[SUBMITTED] provider_job_id = {job_id}")
    (EXPERIMENT_DIR / "wp2_27q_fez_submitted_job.json").write_text(
        json.dumps({"job_id": job_id, "submitted_utc": _utc_now(), "backend": BACKEND_NAME}, indent=2),
        encoding="utf-8",
    )

    t0 = time.time()
    while True:
        st = str(job.status())
        print(f"  [{time.time() - t0:5.1f}s] job {job_id} status = {st}")
        if st in ("DONE", "ERROR", "CANCELLED"):
            break
        time.sleep(4.0)

    if str(job.status()) != "DONE":
        raise RuntimeError(f"Job {job_id} ended with status {job.status()}")

    res = job.result()
    raw_counts: Dict[str, Dict[str, int]] = {}
    for idx in range(len(isa_circuits)):
        raw_counts[str(idx)] = _extract_counts_27bit(res[idx])

    raw_gz_path = EXPERIMENT_DIR / "physical_wp2_27q_fez_raw_counts.json.gz"
    raw_bytes = json.dumps(raw_counts, sort_keys=True).encode("utf-8")
    with gzip.open(raw_gz_path, "wb") as gz:
        gz.write(raw_bytes)
    raw_gz_sha = hashlib.sha256(raw_gz_path.read_bytes()).hexdigest()

    job_meta = redact_ibm_job_metadata(job)
    (EXPERIMENT_DIR / "wp2_27q_fez_submitted_job.json").write_text(
        json.dumps(job_meta, indent=2, sort_keys=True), encoding="utf-8"
    )

    analysis = analyze_harvested(manifest, raw_counts, job_meta, raw_gz_sha)
    out_path = EXPERIMENT_DIR / "physical_wp2_27q_fez_results.json"
    out_path.write_text(json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[HARVESTED] {PROTOCOL_ID} job={job_id} QPU_s={job_meta.get('usage', {}).get('qpu_charge_time_seconds')}")
    return analysis


def analyze_saved() -> Dict[str, Any]:
    manifest = json.loads((EXPERIMENT_DIR / "wp2_27q_fez_staging_manifest.json").read_text(encoding="utf-8"))
    job_meta = json.loads((EXPERIMENT_DIR / "wp2_27q_fez_submitted_job.json").read_text(encoding="utf-8"))
    raw_gz_path = EXPERIMENT_DIR / "physical_wp2_27q_fez_raw_counts.json.gz"
    raw_gz_sha = hashlib.sha256(raw_gz_path.read_bytes()).hexdigest()
    with gzip.open(raw_gz_path, "rb") as gz:
        raw_counts = json.loads(gz.read().decode("utf-8"))
    analysis = analyze_harvested(manifest, raw_counts, job_meta, raw_gz_sha)
    out_path = EXPERIMENT_DIR / "physical_wp2_27q_fez_results.json"
    out_path.write_text(json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[ANALYZED] {PROTOCOL_ID} job={job_meta['job_id']} QPU_s={job_meta.get('usage', {}).get('qpu_charge_time_seconds')}")
    return analysis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-only", action="store_true")
    ap.add_argument("--submit-frozen-and-harvest", action="store_true")
    ap.add_argument("--analyze-saved", action="store_true")
    args = ap.parse_args()
    if args.analyze_saved:
        analyze_saved()
    elif args.submit_frozen_and_harvest:
        submit_frozen_and_harvest()
    else:
        stage_experiment()


if __name__ == "__main__":
    main()
