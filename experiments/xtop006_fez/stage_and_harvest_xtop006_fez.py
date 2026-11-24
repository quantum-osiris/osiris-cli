#!/usr/bin/env python3
"""
stage_and_harvest_xtop006_fez.py — Preregistered constant-skeleton 7-qubit tunable
non-Clifford Hayden-Preskill Kraus trajectory & decoherence discriminator (`OSIRIS-XTOP-006-FEZ`)
on `ibm_fez` (`[99, 115, 95, 93, 94, 92, 79]`).

Default `--stage-only` spends 0 QPU seconds:
  1. Synthesizes all 232 constant-28-CZ, constant-SX ISA circuits directly in `ibm_fez` native gates.
  2. Verifies exact 7-qubit unitary/statevector equivalence (`< 1e-12`) against `xtop006.kraus_channel`
     closed-form `P_{Phi+}(theta)` and `F_6state(theta | Phi+)` on the compact 7Q representation.
  3. Validates all 232 156Q ISA circuits via `validate_isa_circuits(..., fez.target)`.
  4. Freezes `xtop006_fez_staged_circuits.qpy` + SHA-256 pre-registration lock BEFORE submission,
     and harvests the read-only `dc0uhdts2omc7389tmt0` (`SCI-XTOP-1`) provider timestamp receipt.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Statevector
from qiskit.synthesis import OneQubitEulerDecomposer
from qiskit_ibm_runtime import SamplerV2
from qiskit_ibm_runtime.utils.validations import validate_isa_circuits
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osiris_cli.ibm_credentials import connect_ibm_runtime_service, redact_ibm_job_metadata  # noqa: E402
from xtop006.kraus_channel import (  # noqa: E402
    INPUT_STATES_6,
    THETA_GRID_DEFAULT,
    analytical_fidelity_6state,
    analytical_p_phi_plus,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROTOCOL_ID = "OSIRIS-XTOP-006-FEZ"
BACKEND_NAME = "ibm_fez"
# Pinned 7-transmon heavy-hex T-junction [q0, q1, q2, q3, q4, q5, q6]:
#   (q1=115) -- (q0=99) -- (q2=95) -- (q4=94) -- (q3=93) -- {(q5=92), (q6=79)}
FEZ_7Q_LAYOUT: Tuple[int, ...] = (99, 115, 95, 93, 94, 92, 79)
MEAS_PHYS: Tuple[int, int, int] = (FEZ_7Q_LAYOUT[2], FEZ_7Q_LAYOUT[4], FEZ_7Q_LAYOUT[6])  # (95, 94, 79)

N_BLOCKS = 4
SHOTS_PER_PUB = 512
SCHEDULE_SEED = 20261123
THETA_NAMES: Tuple[str, ...] = ("0", "pi_8", "pi_4", "3pi_8", "pi_2")
DECOH_THETA_INDICES: Tuple[int, ...] = (2, 3, 4)  # pi/4, 3pi/8, pi/2
# Deterministic non-trivial Pauli-Z phase twirls on (q0, q1, q2) across the 4 blocks:
DECOH_TWIRLS_BY_BLOCK: Tuple[Tuple[int, int, int], ...] = (
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
)
INPUT_ORDER: Tuple[str, ...] = ("0", "1", "+", "-", "+i", "-i")

_ZYZ = OneQubitEulerDecomposer("ZYZ")
I2 = np.eye(2, dtype=np.complex128)
H_MAT = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)
X_MAT = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
Z_MAT = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
S_MAT = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128)
SDG_MAT = np.array([[1.0, 0.0], [0.0, -1.0j]], dtype=np.complex128)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _rx_mat(alpha: float) -> np.ndarray:
    c = math.cos(0.5 * alpha)
    s = math.sin(0.5 * alpha)
    return np.array([[c, -1.0j * s], [-1.0j * s, c]], dtype=np.complex128)


def _rz_mat(alpha: float) -> np.ndarray:
    return np.array(
        [[np.exp(-0.5j * alpha), 0.0], [0.0, np.exp(0.5j * alpha)]],
        dtype=np.complex128,
    )


def _input_prep_matrix(label: str) -> np.ndarray:
    if label == "0":
        return I2.copy()
    if label == "1":
        return X_MAT.copy()
    if label == "+":
        return H_MAT.copy()
    if label == "-":
        return H_MAT @ X_MAT
    if label == "+i":
        return S_MAT @ H_MAT
    if label == "-i":
        return SDG_MAT @ H_MAT
    raise ValueError(f"Unknown input state {label!r}")


def _basis_rot_matrix(basis: str) -> np.ndarray:
    if basis == "Z":
        return I2.copy()
    if basis == "X":
        return H_MAT.copy()
    if basis == "Y":
        return H_MAT @ SDG_MAT
    raise ValueError(f"Unknown basis {basis!r}")


def _emit_universal_2sx(qc: QuantumCircuit, pq: int, u2: np.ndarray) -> None:
    """Emit ANY 2x2 unitary u2 = Rz(phi) @ SX @ Rz(pi - theta) @ SX @ Rz(lam + pi)
    in Qiskit time-ordered gate sequence (rightmost matrix applied first!).
    """
    theta, phi, lam, _ = _ZYZ.angles_and_phase(u2)
    qc.rz(float(lam + math.pi), pq)
    qc.sx(pq)
    qc.rz(float(math.pi - theta), pq)
    qc.sx(pq)
    qc.rz(float(phi), pq)


def build_constant_skeleton_7q_ops(
    theta: float,
    input_label: str,
    decoh_twirl: Tuple[int, int, int] = (0, 0, 0),
) -> List[Tuple[str, Any]]:
    """Build exact 28-CZ sequence on logical wires 0..6 where ALL 1Q gates between CZs
    are contracted into 2x2 matrices and emitted via universal 2-SX slots.

    Logical wires:
      0=q0(in), 1=q1, 2=q2, 3=q3, 4=q4, 5=q5, 6=q6(Bob out)
      Adjacent edges on T-junction: (1,0), (0,2), (2,4), (4,3), (3,5), (3,6).
    """
    buf: Dict[int, np.ndarray] = {q: I2.copy() for q in range(7)}
    stream: List[Tuple[str, Any]] = []

    def apply_1q(q: int, m: np.ndarray) -> None:
        buf[q] = m @ buf[q]

    def apply_cz(a: int, b: int) -> None:
        # Flush buffered 1Q unitaries on BOTH participating wires via constant 2-SX slot
        stream.append(("1q", (a, buf[a].copy())))
        stream.append(("1q", (b, buf[b].copy())))
        buf[a] = I2.copy()
        buf[b] = I2.copy()
        stream.append(("cz", (a, b)))

    def apply_cx(ctrl: int, targ: int) -> None:
        apply_1q(targ, H_MAT)
        apply_cz(ctrl, targ)
        apply_1q(targ, H_MAT)

    def apply_shuttle_from_zero(src: int, dst_zero: int) -> None:
        """Move arbitrary qubit state from `src` to `dst_zero` (initially |0>), leaving `src` in |0>, in 2 CZ."""
        apply_cx(src, dst_zero)
        apply_cx(dst_zero, src)

    # 1. Prepare EPR(q1, q5) by creating |Phi+> on (1, 0) [1 CZ] and shuttling 0->2->4->3->5 [8 CZ]
    #    while wires 0, 2, 4, 3, 5 are all in |0>!
    apply_1q(1, H_MAT)
    apply_cx(1, 0)
    apply_shuttle_from_zero(0, 2)
    apply_shuttle_from_zero(2, 4)
    apply_shuttle_from_zero(4, 3)
    apply_shuttle_from_zero(3, 5)

    # 2. Now wires 0, 2, 4, 3, 6 are clean |0>! Prepare EPR(q2, q4) [1 CZ], EPR(q3, q6) [1 CZ], and input on q0!
    apply_1q(2, H_MAT)
    apply_cx(2, 4)
    apply_1q(3, H_MAT)
    apply_cx(3, 6)
    apply_1q(0, _input_prep_matrix(input_label))

    # 3. Tunable non-Clifford U(theta) on (0,1,2) and U*(theta)=U(-theta) on (3,5,4) [16 CZ total]
    def apply_tcx(ctrl: int, targ: int, th_val: float) -> None:
        apply_cz(ctrl, targ)
        apply_1q(targ, _rx_mat(-th_val))
        apply_cz(ctrl, targ)
        apply_1q(targ, _rx_mat(th_val))
        apply_1q(ctrl, _rz_mat(th_val))

    for q in (0, 1, 2, 3, 5, 4):
        apply_1q(q, H_MAT)
    apply_tcx(0, 1, theta)
    apply_tcx(0, 2, theta)
    apply_tcx(3, 5, -theta)
    apply_tcx(3, 4, -theta)

    # Optional OTOC-mimic mid-scrambler Pauli-Z phase twirl on Alice's (0, 1, 2) — ZERO extra CZ/SX!
    for idx_q, z_bit in enumerate(decoh_twirl):
        if z_bit:
            apply_1q(idx_q, Z_MAT)

    for q in (0, 1, 2, 3, 5, 4):
        apply_1q(q, H_MAT)
    apply_tcx(0, 1, theta)
    apply_tcx(0, 2, theta)
    apply_tcx(3, 5, -theta)
    apply_tcx(3, 4, -theta)

    # 4. Bell measurement on (q2, q4) [1 CZ] + Pauli basis rotation on Bob's output q6
    apply_cx(2, 4)
    apply_1q(2, H_MAT)
    _, basis, _ = INPUT_STATES_6[input_label]
    apply_1q(6, _basis_rot_matrix(basis))

    # Flush final 1Q buffers on all 7 wires so SX count is strictly constant across every circuit!
    for q in range(7):
        stream.append(("1q", (q, buf[q].copy())))

    return stream


def compile_stream_to_compact_7q(stream: Sequence[Tuple[str, Any]]) -> QuantumCircuit:
    qc = QuantumCircuit(7)
    for kind, payload in stream:
        if kind == "1q":
            q, mat = payload
            _emit_universal_2sx(qc, q, mat)
        elif kind == "cz":
            a, b = payload
            qc.cz(a, b)
    return qc


def compile_stream_to_fez_isa(
    stream: Sequence[Tuple[str, Any]],
    tgt: Any,
    name: str,
) -> QuantumCircuit:
    qc = QuantumCircuit(tgt.num_qubits, 3, name=name)
    for kind, payload in stream:
        if kind == "1q":
            lq, mat = payload
            pq = FEZ_7Q_LAYOUT[lq]
            _emit_universal_2sx(qc, pq, mat)
        elif kind == "cz":
            la, lb = payload
            pa, pb = FEZ_7Q_LAYOUT[la], FEZ_7Q_LAYOUT[lb]
            # Barrier on pair prevents any backend-side merging across CZs while preserving exact timing
            qc.barrier(pa, pb)
            if (pa, pb) in tgt["cz"]:
                qc.cz(pa, pb)
            elif (pb, pa) in tgt["cz"]:
                qc.cz(pb, pa)
            else:
                raise RuntimeError(f"Physical edge ({pa}, {pb}) not in ibm_fez target['cz']")
            qc.barrier(pa, pb)
    qc.barrier(*FEZ_7Q_LAYOUT)
    qc.measure(MEAS_PHYS[0], 0)  # q2 -> c[0]
    qc.measure(MEAS_PHYS[1], 1)  # q4 -> c[1]
    qc.measure(MEAS_PHYS[2], 2)  # q6 -> c[2]
    return qc


def build_ro_cal_fez_isa(bits_246: Tuple[int, int, int], tgt: Any, name: str) -> QuantumCircuit:
    """3-qubit readout calibration state |b2, b4, b6> on physical (95, 94, 79)."""
    qc = QuantumCircuit(tgt.num_qubits, 3, name=name)
    for cl_idx, (pq, bit_val) in enumerate(zip(MEAS_PHYS, bits_246)):
        mat = X_MAT if bit_val == 1 else I2
        _emit_universal_2sx(qc, pq, mat)
        qc.measure(pq, cl_idx)
    return qc


def verify_noiseless_7q_streams() -> Dict[str, Any]:
    """Verify exact statevector agreement (<1e-12) against closed-form Kraus theory."""
    rows: List[Dict[str, Any]] = []
    max_abs_err = 0.0
    for t_idx, th in enumerate(THETA_GRID_DEFAULT):
        p_phis = []
        f_vals = []
        cz_counts = set()
        sx_counts = set()
        for lbl in INPUT_ORDER:
            _, _, eig = INPUT_STATES_6[lbl]
            stream = build_constant_skeleton_7q_ops(th, lbl, decoh_twirl=(0, 0, 0))
            qc7 = compile_stream_to_compact_7q(stream)
            ops = qc7.count_ops()
            cz_counts.add(int(ops.get("cz", 0)))
            sx_counts.add(int(ops.get("sx", 0)))
            probs = np.abs(Statevector.from_instruction(qc7).data) ** 2
            target_b6 = 0 if eig == +1 else 1
            p_phi = 0.0
            p_ok = 0.0
            for idx, p in enumerate(probs):
                if ((idx >> 2) & 1) == 0 and ((idx >> 4) & 1) == 0:
                    p_phi += float(p)
                    if ((idx >> 6) & 1) == target_b6:
                        p_ok += float(p)
            p_phis.append(p_phi)
            f_vals.append(p_ok / p_phi)

        mean_p = float(np.mean(p_phis))
        mean_f = float(np.mean(f_vals))
        p_th = analytical_p_phi_plus(th)
        f_th = analytical_fidelity_6state(th)
        err = max(abs(mean_p - p_th), abs(mean_f - f_th))
        max_abs_err = max(max_abs_err, err)
        assert cz_counts == {28}, f"Expected constant 28 CZ, got {cz_counts}"
        assert len(sx_counts) == 1, f"Expected constant SX count, got {sx_counts}"

        # Also check 4-twirl average of DECOH_CTRL if in DECOH_THETA_INDICES
        decoh_f = None
        if t_idx in DECOH_THETA_INDICES:
            df_list = []
            for tw in DECOH_TWIRLS_BY_BLOCK:
                for lbl in INPUT_ORDER:
                    _, _, eig = INPUT_STATES_6[lbl]
                    qc_d = compile_stream_to_compact_7q(build_constant_skeleton_7q_ops(th, lbl, decoh_twirl=tw))
                    probs_d = np.abs(Statevector.from_instruction(qc_d).data) ** 2
                    target_b6 = 0 if eig == +1 else 1
                    pp = sum(float(probs_d[i]) for i in range(128) if ((i >> 2) & 1) == 0 and ((i >> 4) & 1) == 0)
                    po = sum(
                        float(probs_d[i])
                        for i in range(128)
                        if ((i >> 2) & 1) == 0 and ((i >> 4) & 1) == 0 and ((i >> 6) & 1) == target_b6
                    )
                    df_list.append(po / pp)
            decoh_f = float(np.mean(df_list))

        rows.append(
            {
                "theta_name": THETA_NAMES[t_idx],
                "theta_rad": round(th, 8),
                "cz_count": 28,
                "sx_count": next(iter(sx_counts)),
                "noiseless_p_phi_plus": round(mean_p, 8),
                "analytic_p_phi_plus": round(p_th, 8),
                "noiseless_f_6state": round(mean_f, 8),
                "analytic_f_6state": round(f_th, 8),
                "noiseless_decoh_ctrl_f_6state": round(decoh_f, 8) if decoh_f is not None else None,
            }
        )
    assert max_abs_err < 1e-12, f"Statevector vs analytical mismatch: {max_abs_err}"
    return {"max_abs_error_vs_kraus_formula": max_abs_err, "theta_verification": rows}


def stage_experiment() -> Dict[str, Any]:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    noiseless_check = verify_noiseless_7q_streams()

    svc, cred_prov = connect_ibm_runtime_service()
    fez = svc.backend(BACKEND_NAME)
    tgt = fez.target
    props = fez.properties()

    # Harvest read-only SCI-XTOP-1 receipt for dc0uhdts2omc7389tmt0 while connected (0 QPU seconds!)
    try:
        j_xtop5 = svc.job("dc0uhdts2omc7389tmt0")
        sci_xtop1 = redact_ibm_job_metadata(j_xtop5)
        (EXPERIMENT_DIR / "sci_xtop1_dc0uhdts2omc7389tmt0_provider_receipt.json").write_text(
            json.dumps(sci_xtop1, indent=2, sort_keys=True), encoding="utf-8"
        )
    except Exception as exc:
        sci_xtop1 = {"error": str(exc)}

    rng = np.random.default_rng(SCHEDULE_SEED)
    isa_circuits: List[QuantumCircuit] = []
    schedule_meta: List[Dict[str, Any]] = []

    for blk in range(N_BLOCKS):
        block_items: List[Tuple[QuantumCircuit, Dict[str, Any]]] = []
        # 1. COHERENT: 5 theta x 6 input states = 30 circuits
        for t_idx, th in enumerate(THETA_GRID_DEFAULT):
            for lbl in INPUT_ORDER:
                cname = f"b{blk}_coh_t{THETA_NAMES[t_idx]}_in_{lbl}"
                stream = build_constant_skeleton_7q_ops(th, lbl, decoh_twirl=(0, 0, 0))
                qc = compile_stream_to_fez_isa(stream, tgt, cname)
                block_items.append(
                    (
                        qc,
                        {
                            "block": blk,
                            "arm": "COHERENT",
                            "theta_idx": t_idx,
                            "theta_name": THETA_NAMES[t_idx],
                            "theta_rad": th,
                            "input_label": lbl,
                            "decoh_twirl": [0, 0, 0],
                        },
                    )
                )

        # 2. DECOH_CTRL: 3 theta (pi/4, 3pi/8, pi/2) x 6 input states = 18 circuits
        tw = DECOH_TWIRLS_BY_BLOCK[blk]
        for t_idx in DECOH_THETA_INDICES:
            th = THETA_GRID_DEFAULT[t_idx]
            for lbl in INPUT_ORDER:
                cname = f"b{blk}_decoh_t{THETA_NAMES[t_idx]}_in_{lbl}"
                stream = build_constant_skeleton_7q_ops(th, lbl, decoh_twirl=tw)
                qc = compile_stream_to_fez_isa(stream, tgt, cname)
                block_items.append(
                    (
                        qc,
                        {
                            "block": blk,
                            "arm": "DECOH_CTRL",
                            "theta_idx": t_idx,
                            "theta_name": THETA_NAMES[t_idx],
                            "theta_rad": th,
                            "input_label": lbl,
                            "decoh_twirl": list(tw),
                        },
                    )
                )

        # 3. 2 A/A identical-circuit drift sentinels at theta = pi/2, input '+'
        for s_rep in range(2):
            cname = f"b{blk}_aa_sentinel_{s_rep}"
            stream = build_constant_skeleton_7q_ops(math.pi / 2.0, "+", decoh_twirl=(0, 0, 0))
            qc = compile_stream_to_fez_isa(stream, tgt, cname)
            block_items.append(
                (
                    qc,
                    {
                        "block": blk,
                        "arm": "AA_SENTINEL",
                        "sentinel_rep": s_rep,
                        "theta_idx": 4,
                        "theta_name": "pi_2",
                        "theta_rad": math.pi / 2.0,
                        "input_label": "+",
                    },
                )
            )

        # 4. 8 computational-basis 3Q readout calibration states on (q2=95, q4=94, q6=79)
        for mask in range(8):
            bits = (mask & 1, (mask >> 1) & 1, (mask >> 2) & 1)
            cname = f"b{blk}_rocal_{bits[2]}{bits[1]}{bits[0]}"
            qc = build_ro_cal_fez_isa(bits, tgt, cname)
            block_items.append(
                (
                    qc,
                    {
                        "block": blk,
                        "arm": "RO_CAL",
                        "prep_bits_c2c1c0": f"{bits[2]}{bits[1]}{bits[0]}",
                    },
                )
            )

        # Randomize intra-block execution order deterministically!
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

    # Validate ALL 232 ISA circuits against live ibm_fez target!
    validate_isa_circuits(isa_circuits, tgt)

    # Verify strict constant-skeleton invariant across ALL COHERENT, DECOH_CTRL, and AA_SENTINEL circuits
    work_cz = {m["cz_count"] for m in schedule_meta if m["arm"] != "RO_CAL"}
    work_sx = {m["sx_count"] for m in schedule_meta if m["arm"] != "RO_CAL"}
    assert work_cz == {28}, f"Non-constant CZ skeleton: {work_cz}"
    assert len(work_sx) == 1, f"Non-constant SX skeleton: {work_sx}"

    qpy_path = EXPERIMENT_DIR / "xtop006_fez_staged_circuits.qpy"
    with qpy_path.open("wb") as f:
        qpy.dump(isa_circuits, f)
    qpy_sha256 = hashlib.sha256(qpy_path.read_bytes()).hexdigest()

    cal_snap = {
        "backend": BACKEND_NAME,
        "captured_utc": _utc_now(),
        "last_update_date": str(getattr(props, "last_update_date", None)),
        "layout_7q": list(FEZ_7Q_LAYOUT),
        "measured_phys_c0c1c2": list(MEAS_PHYS),
        "qubits": {
            str(pq): {
                "t1_us": round(float(props.t1(pq)) * 1e6, 3),
                "t2_us": round(float(props.t2(pq)) * 1e6, 3),
                "readout_error": round(float(tgt["measure"][(pq,)].error or 0.0), 6),
            }
            for pq in FEZ_7Q_LAYOUT
        },
        "active_cz_edges": {
            f"{a}-{b}": round(
                float(
                    (tgt["cz"][(a, b)] if (a, b) in tgt["cz"] else tgt["cz"][(b, a)]).error or 0.0
                ),
                6,
            )
            for a, b in [
                (FEZ_7Q_LAYOUT[1], FEZ_7Q_LAYOUT[0]),
                (FEZ_7Q_LAYOUT[0], FEZ_7Q_LAYOUT[2]),
                (FEZ_7Q_LAYOUT[2], FEZ_7Q_LAYOUT[4]),
                (FEZ_7Q_LAYOUT[4], FEZ_7Q_LAYOUT[3]),
                (FEZ_7Q_LAYOUT[3], FEZ_7Q_LAYOUT[5]),
                (FEZ_7Q_LAYOUT[3], FEZ_7Q_LAYOUT[6]),
            ]
        },
    }
    cal_path = EXPERIMENT_DIR / "xtop006_fez_calibration_snapshot.json"
    cal_path.write_text(json.dumps(cal_snap, indent=2, sort_keys=True), encoding="utf-8")

    prereg_sha = hashlib.sha256((EXPERIMENT_DIR / "PRE_REGISTRATION.md").read_bytes()).hexdigest()
    manifest = {
        "schema": "osiris.quantum_protocol/1",
        "protocol_id": PROTOCOL_ID,
        "staged_utc": _utc_now(),
        "backend": BACKEND_NAME,
        "credential_provenance": cred_prov,
        "layout_7q_q0_to_q6": list(FEZ_7Q_LAYOUT),
        "measured_phys_c0c1c2": list(MEAS_PHYS),
        "n_blocks": N_BLOCKS,
        "pubs_per_block": 58,
        "total_pubs": len(isa_circuits),
        "shots_per_pub": SHOTS_PER_PUB,
        "total_shots": len(isa_circuits) * SHOTS_PER_PUB,
        "constant_cz_count": 28,
        "constant_sx_count": next(iter(work_sx)),
        "qpy_filename": qpy_path.name,
        "qpy_sha256": qpy_sha256,
        "preregistration_md_sha256": prereg_sha,
        "calibration_snapshot_sha256": hashlib.sha256(cal_path.read_bytes()).hexdigest(),
        "noiseless_verification": noiseless_check,
        "sci_xtop1_provider_receipt_captured": "job_id" in sci_xtop1,
        "schedule": schedule_meta,
    }
    man_path = EXPERIMENT_DIR / "xtop006_fez_staging_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[STAGED] {PROTOCOL_ID}: {len(isa_circuits)} PUBs x {SHOTS_PER_PUB} shots = {len(isa_circuits)*SHOTS_PER_PUB} shots")
    print(f"  qpy_sha256 = {qpy_sha256}")
    return manifest


def _extract_counts_3bit(pub_result: Any) -> Dict[str, int]:
    """Extract 3-bit string ('c2 c1 c0') counts cleanly via Qiskit BitArray."""
    data_bin = pub_result.data
    for attr in ("c", "meas"):
        if hasattr(data_bin, attr):
            return {str(k): int(v) for k, v in getattr(data_bin, attr).get_counts().items()}
    # Fallback first BitArray field
    for field_name in dir(data_bin):
        if not field_name.startswith("_"):
            val = getattr(data_bin, field_name)
            if hasattr(val, "get_counts"):
                return {str(k): int(v) for k, v in val.get_counts().items()}
    raise RuntimeError("Could not find BitArray counts in pub_result.data")


def _score_counts(counts: Mapping[str, int], input_label: str) -> Tuple[int, int, int]:
    """Return `(total_shots, n_phi_plus, n_bob_correct_given_phi_plus)` for 3-bit strings `c2c1c0`."""
    _, _, eig = INPUT_STATES_6[input_label]
    target_c2 = "0" if eig == +1 else "1"
    tot = 0
    n_phi = 0
    n_ok = 0
    for bitstr, cnt in counts.items():
        s = bitstr.replace(" ", "").zfill(3)
        c2, c1, c0 = s[-3], s[-2], s[-1]
        tot += int(cnt)
        if c1 == "0" and c0 == "0":
            n_phi += int(cnt)
            if c2 == target_c2:
                n_ok += int(cnt)
    return tot, n_phi, n_ok


def _holm_bonferroni(p_vals: Sequence[float]) -> List[float]:
    m = len(p_vals)
    indexed = sorted(enumerate(p_vals), key=lambda kv: kv[1])
    adj = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, pv) in enumerate(indexed):
        val = min(1.0, pv * float(m - rank))
        running_max = max(running_max, val)
        adj[orig_idx] = running_max
    return adj


def analyze_harvested(
    manifest: Mapping[str, Any],
    raw_counts_by_pub: Mapping[str, Mapping[str, int]],
    job_meta: Mapping[str, Any],
    raw_gz_sha256: str,
) -> Dict[str, Any]:
    sched = manifest["schedule"]

    # 1. Build 8x8 empirical readout assignment matrix M_{obs, true} across all 4 blocks (c2c1c0 in 0..7)
    ro_mat = np.zeros((8, 8), dtype=np.float64)
    ro_totals = np.zeros(8, dtype=np.float64)
    for item in sched:
        if item["arm"] == "RO_CAL":
            true_idx = int(item["prep_bits_c2c1c0"], 2)
            cnts = raw_counts_by_pub[str(item["pub_index"])]
            for bstr, c in cnts.items():
                obs_idx = int(bstr.replace(" ", "").zfill(3)[-3:], 2)
                ro_mat[obs_idx, true_idx] += float(c)
                ro_totals[true_idx] += float(c)
    for j in range(8):
        if ro_totals[j] > 0:
            ro_mat[:, j] /= ro_totals[j]
    ro_cond = float(np.linalg.cond(ro_mat))
    ro_inv = np.linalg.pinv(ro_mat)

    def mitigated_probs_from_pooled(pooled_counts: Mapping[str, int]) -> np.ndarray:
        vec = np.zeros(8, dtype=np.float64)
        for bstr, c in pooled_counts.items():
            vec[int(bstr.replace(" ", "").zfill(3)[-3:], 2)] += float(c)
        vec /= max(1.0, float(np.sum(vec)))
        mit = ro_inv @ vec
        # Project onto probability simplex (Michelot / Duchi) so probabilities are valid in [0, 1]
        u = np.sort(mit)[::-1]
        cssv = np.cumsum(u) - 1.0
        ind = np.arange(1, 9)
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta_proj = cssv[cond][-1] / float(rho)
        return np.maximum(mit - theta_proj, 0.0)

    # 2. Evaluate COHERENT across all 5 theta points (per-block + pooled + mitigated)
    coherent_rows: List[Dict[str, Any]] = []
    block_f_coh: Dict[int, List[float]] = {t_idx: [] for t_idx in range(5)}
    block_p_coh: Dict[int, List[float]] = {t_idx: [] for t_idx in range(5)}

    for t_idx, th in enumerate(THETA_GRID_DEFAULT):
        pooled_by_lbl: Dict[str, Dict[str, int]] = {lbl: {} for lbl in INPUT_ORDER}
        for blk in range(N_BLOCKS):
            blk_phi = 0
            blk_ok = 0
            blk_tot = 0
            for item in sched:
                if item["arm"] == "COHERENT" and item["block"] == blk and item["theta_idx"] == t_idx:
                    cnts = raw_counts_by_pub[str(item["pub_index"])]
                    lbl = item["input_label"]
                    for k, v in cnts.items():
                        pooled_by_lbl[lbl][k] = pooled_by_lbl[lbl].get(k, 0) + int(v)
                    tot, n_phi, n_ok = _score_counts(cnts, lbl)
                    blk_tot += tot
                    blk_phi += n_phi
                    blk_ok += n_ok
            block_p_coh[t_idx].append(blk_phi / max(1, blk_tot))
            block_f_coh[t_idx].append(blk_ok / max(1, blk_phi))

        # Pooled raw & readout-mitigated across all 6 states
        tot_all, phi_all, ok_all = 0, 0, 0
        mit_p_list, mit_f_list = [], []
        for lbl in INPUT_ORDER:
            _, _, eig = INPUT_STATES_6[lbl]
            t_c, p_c, o_c = _score_counts(pooled_by_lbl[lbl], lbl)
            tot_all += t_c
            phi_all += p_c
            ok_all += o_c
            mp = mitigated_probs_from_pooled(pooled_by_lbl[lbl])
            # c1==0, c0==0 => indices 0 (000) and 4 (100)
            p_phi_m = float(mp[0] + mp[4])
            p_ok_m = float(mp[0] if eig == +1 else mp[4])
            mit_p_list.append(p_phi_m)
            mit_f_list.append(p_ok_m / max(1e-9, p_phi_m))

        raw_p = phi_all / max(1, tot_all)
        raw_f = ok_all / max(1, phi_all)
        raw_f_ent = 0.5 * (3.0 * raw_f - 1.0)
        mit_p = float(np.mean(mit_p_list))
        mit_f = float(np.mean(mit_f_list))
        mit_f_ent = 0.5 * (3.0 * mit_f - 1.0)

        f_arr = np.array(block_f_coh[t_idx], dtype=np.float64)
        se_blk = float(np.std(f_arr, ddof=1) / math.sqrt(N_BLOCKS))
        t_crit = float(stats.t.ppf(0.95, df=N_BLOCKS - 1))  # 1-sided 95% LCL
        lcl95_f = float(np.mean(f_arr) - t_crit * se_blk)

        coherent_rows.append(
            {
                "theta_name": THETA_NAMES[t_idx],
                "theta_rad": round(th, 8),
                "total_shots": tot_all,
                "postselected_phi_plus_shots": phi_all,
                "analytic_p_phi_plus": round(analytical_p_phi_plus(th), 6),
                "raw_p_phi_plus": round(raw_p, 6),
                "ro_mitigated_p_phi_plus": round(mit_p, 6),
                "analytic_f_6state": round(analytical_fidelity_6state(th), 6),
                "raw_f_6state": round(raw_f, 6),
                "raw_f_6state_block_se": round(se_blk, 6),
                "raw_f_6state_95pct_lcl": round(lcl95_f, 6),
                "ro_mitigated_f_6state": round(mit_f, 6),
                "raw_f_ent": round(raw_f_ent, 6),
                "ro_mitigated_f_ent": round(mit_f_ent, 6),
                "raw_yoshida_kitaev_product": round(raw_p * raw_f_ent, 6),
                "ro_mitigated_yoshida_kitaev_product": round(mit_p * mit_f_ent, 6),
                "block_raw_f_6state": [round(x, 6) for x in block_f_coh[t_idx]],
            }
        )

    # 3. Evaluate DECOH_CTRL vs COHERENT paired across blocks at theta in {pi/4, 3pi/8, pi/2}
    decoh_contrasts: List[Dict[str, Any]] = []
    raw_pvals_decoh: List[float] = []
    for t_idx in DECOH_THETA_INDICES:
        blk_f_dec: List[float] = []
        blk_p_dec: List[float] = []
        for blk in range(N_BLOCKS):
            b_tot, b_phi, b_ok = 0, 0, 0
            for item in sched:
                if item["arm"] == "DECOH_CTRL" and item["block"] == blk and item["theta_idx"] == t_idx:
                    cnts = raw_counts_by_pub[str(item["pub_index"])]
                    tc, pc, oc = _score_counts(cnts, item["input_label"])
                    b_tot += tc
                    b_phi += pc
                    b_ok += oc
            blk_p_dec.append(b_phi / max(1, b_tot))
            blk_f_dec.append(b_ok / max(1, b_phi))

        diffs = np.array(block_f_coh[t_idx]) - np.array(blk_f_dec)
        t_stat, p_two = stats.ttest_1samp(diffs, popmean=0.0)
        p_one = float(p_two / 2.0) if t_stat > 0 else float(1.0 - p_two / 2.0)
        raw_pvals_decoh.append(p_one)
        decoh_contrasts.append(
            {
                "theta_name": THETA_NAMES[t_idx],
                "theta_rad": round(THETA_GRID_DEFAULT[t_idx], 8),
                "coherent_mean_raw_f": round(float(np.mean(block_f_coh[t_idx])), 6),
                "decoh_ctrl_mean_raw_f": round(float(np.mean(blk_f_dec)), 6),
                "delta_f_coherent_minus_decoh": round(float(np.mean(diffs)), 6),
                "delta_f_block_se": round(float(np.std(diffs, ddof=1) / math.sqrt(N_BLOCKS)), 6),
                "paired_t_stat": round(float(t_stat), 4),
                "p_one_sided": round(p_one, 8),
            }
        )
    holm_decoh = _holm_bonferroni(raw_pvals_decoh)
    for row, hp in zip(decoh_contrasts, holm_decoh):
        row["p_holm_adjusted"] = round(hp, 8)

    # 4. H1 Climb test: theta=pi/2 vs theta=0 paired across blocks
    climb_diffs = np.array(block_f_coh[4]) - np.array(block_f_coh[0])
    t_climb, p_climb_two = stats.ttest_1samp(climb_diffs, popmean=0.0)
    p_climb_one = float(p_climb_two / 2.0) if t_climb > 0 else 1.0

    # 5. A/A Sentinel RMS Drift Floor
    aa_diffs: List[float] = []
    for blk in range(N_BLOCKS):
        s_f = []
        for s_rep in (0, 1):
            for item in sched:
                if item["arm"] == "AA_SENTINEL" and item["block"] == blk and item["sentinel_rep"] == s_rep:
                    _, pc, oc = _score_counts(raw_counts_by_pub[str(item["pub_index"])], "+")
                    s_f.append(oc / max(1, pc))
        if len(s_f) == 2:
            aa_diffs.append(s_f[0] - s_f[1])
    aa_rms = float(math.sqrt(np.mean(np.square(aa_diffs)))) if aa_diffs else 0.0

    h1_pass = (
        float(np.mean(climb_diffs)) > 0.20
        and p_climb_one < 0.01
        and coherent_rows[4]["raw_f_6state_95pct_lcl"] > (2.0 / 3.0)
    )
    h2_pass = all(
        c["delta_f_coherent_minus_decoh"] > (0.15 if c["theta_name"] == "pi_2" else 0.10)
        and c["p_holm_adjusted"] < 0.01
        for c in decoh_contrasts
        if c["theta_name"] in ("3pi_8", "pi_2")
    )

    return {
        "schema": "osiris.xtop006_fez.results/1",
        "protocol_id": PROTOCOL_ID,
        "execution_tier": "HARDWARE_ATTESTED",
        "backend": BACKEND_NAME,
        "provider_job_id": job_meta["job_id"],
        "job_metadata": job_meta,
        "raw_counts_gz_sha256": raw_gz_sha256,
        "qpy_sha256": manifest["qpy_sha256"],
        "layout_7q": manifest["layout_7q_q0_to_q6"],
        "constant_cz_count": 28,
        "constant_sx_count": manifest["constant_sx_count"],
        "readout_assignment_matrix_cond": round(ro_cond, 4),
        "aa_sentinel_intra_block_rms_f": round(aa_rms, 6),
        "coherent_trajectory": coherent_rows,
        "decoh_ctrl_contrasts": decoh_contrasts,
        "hypotheses": {
            "H1_non_clifford_kraus_climb": {
                "delta_f_pi2_minus_0": round(float(np.mean(climb_diffs)), 6),
                "paired_t_stat": round(float(t_climb), 4),
                "p_one_sided": round(p_climb_one, 8),
                "f_pi2_95pct_lcl": coherent_rows[4]["raw_f_6state_95pct_lcl"],
                "classical_2_3_bound": round(2.0 / 3.0, 6),
                "verdict": "SUPPORTED" if h1_pass else "FALSIFIED",
            },
            "H2_scrambling_vs_decoherence_discriminator": {
                "verdict": "SUPPORTED" if h2_pass else "FALSIFIED",
            },
        },
    }


def submit_frozen_and_harvest() -> Dict[str, Any]:
    man_path = EXPERIMENT_DIR / "xtop006_fez_staging_manifest.json"
    qpy_path = EXPERIMENT_DIR / "xtop006_fez_staged_circuits.qpy"
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    actual_sha = hashlib.sha256(qpy_path.read_bytes()).hexdigest()
    if actual_sha != manifest["qpy_sha256"]:
        raise RuntimeError(f"QPY SHA-256 mismatch: {actual_sha} != {manifest['qpy_sha256']}")

    with qpy_path.open("rb") as f:
        isa_circuits = qpy.load(f)

    svc, _ = connect_ibm_runtime_service()
    usage_before = svc.usage()
    rem_sec = float(usage_before.get("usage_remaining_seconds", 0))
    if rem_sec < 40.0:
        raise RuntimeError(f"Insufficient QPU quota remaining ({rem_sec}s < 40s safety gate)")

    fez = svc.backend(BACKEND_NAME)
    sampler = SamplerV2(mode=fez)
    sampler.options.default_shots = SHOTS_PER_PUB

    print(f"[SUBMIT] Submitting {len(isa_circuits)} ISA PUBs x {SHOTS_PER_PUB} shots to {BACKEND_NAME} (rem={rem_sec}s)...")
    job = sampler.run(isa_circuits)
    job_id = job.job_id()
    print(f"[SUBMITTED] provider_job_id = {job_id}")
    (EXPERIMENT_DIR / "xtop006_fez_submitted_job.json").write_text(
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
        raw_counts[str(idx)] = _extract_counts_3bit(res[idx])

    raw_gz_path = EXPERIMENT_DIR / "physical_xtop006_fez_raw_counts.json.gz"
    raw_bytes = json.dumps(raw_counts, sort_keys=True).encode("utf-8")
    with gzip.open(raw_gz_path, "wb") as gz:
        gz.write(raw_bytes)
    raw_gz_sha = hashlib.sha256(raw_gz_path.read_bytes()).hexdigest()

    job_meta = redact_ibm_job_metadata(job)
    (EXPERIMENT_DIR / "xtop006_fez_submitted_job.json").write_text(
        json.dumps(job_meta, indent=2, sort_keys=True), encoding="utf-8"
    )

    analysis = analyze_harvested(manifest, raw_counts, job_meta, raw_gz_sha)
    out_path = EXPERIMENT_DIR / "physical_xtop006_fez_results.json"
    out_path.write_text(json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[HARVESTED] {PROTOCOL_ID} job={job_id} QPU_s={job_meta.get('usage', {}).get('qpu_charge_time_seconds')}")
    return analysis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-only", action="store_true")
    ap.add_argument("--submit-frozen-and-harvest", action="store_true")
    args = ap.parse_args()
    if args.submit_frozen_and_harvest:
        submit_frozen_and_harvest()
    else:
        stage_experiment()


if __name__ == "__main__":
    main()
