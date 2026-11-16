"""
Phase 3 Authorized Live QPU Runner (`ibm_marrakesh`)
====================================================

Executes the Principal Investigator–authorized Phase 3 live hardware campaigns:
  1. `TESSERACT-MAP-02`:
     9-arm controlled compilation benchmark (7 matched 40-qubit compilation arms
     C1..C7 + 2 full-union SPAM readout circuits covering 100% of active physical
     qubits across all arms), B = 10 randomized interleaved blocks x 2,048 shots
     = 90 ISA PUBs (184,320 shots, ~55 s QPU).
  2. `CORRELATED-NOISE-01 Stage 1`:
     Multiplexed 10-coupler (>= 5-hop disjoint: 4 training + 6 held-out couplers)
     XY8 staggered dynamical decoupling & multi-observable diagnostic suite on
     exact `dt = 4.0 ns` ticks (`T_8 = 8.192 us = 2,048 dt`, `T_16 = 16.384 us =
     4,096 dt`, `w_p = 36.0 ns = 9 dt`, `base_start = 28 dt` ensuring zero boundary
     clipping and every delay >= 27 dt >> min_length=2 dt), B = 5 randomized
     interleaved blocks + 4x4 two-qubit readout assignment calibration = 244 ISA
     PUBs x 2,048 shots (499,712 circuit shots = 4,997,120 coupler-shots, ~150 s QPU).

Strict SYS-021 Governance:
  - Never prints or logs secret API keys/tokens.
  - Locks QPY binaries, physical layouts, coupler cohorts, randomized block orders,
    and SHA-256 digests to `docs/phase3/` BEFORE calling `SamplerV2.run()`.
  - Checks live OpenPlan `usage_remaining_seconds` before each submission.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, qpy, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

from phase2.correlated_noise_01.correlated_noise_models import (
    HERON_DT_S,
    MARRAKESH_PULSE_WIDTH_S,
    ProbeCondition,
)
from phase2.correlated_noise_01.simulate_correlated_noise import (
    STAGGER_OFFSETS_NS,
    T_REF_8US,
    T_REF_16US,
    fit_competing_models,
    select_disjoint_couplers_on_backend,
)
from phase2.tesseract_map_02.tesseract_map_02 import (
    THETA_DEFAULT_DEG,
    analyze_paired_blocks,
    build_tesseract_map_02_arms,
    build_union_spam_calibration_circuits,
    score_tesseract_counts,
)

REPO_ROOT = Path("/home/osiris/osiris-assessment/work/osiris-cli-phase2")
OUT_DIR = REPO_ROOT / "docs" / "phase3"


def connect_service() -> QiskitRuntimeService:
    """Connect to IBM Quantum Platform without printing credentials."""
    try:
        return QiskitRuntimeService()
    except Exception:
        key_path = Path("/home/osiris/apikey.json")
        data = json.loads(key_path.read_text())
        token = data.get("apikey") or data.get("token")
        if not token:
            raise RuntimeError("No IBM Quantum API token available")
        return QiskitRuntimeService(channel="ibm_quantum_platform", token=token)


def dump_qpy_sha256(circuits: Sequence[QuantumCircuit], path: Path) -> str:
    buf = io.BytesIO()
    qpy.dump(list(circuits), buf)
    raw = buf.getvalue()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def snapshot_backend_calibration(backend: Any) -> Dict[str, Any]:
    target = backend.target
    q_cal: Dict[str, Any] = {}
    for q in range(backend.num_qubits):
        qp = target.qubit_properties[q] if target.qubit_properties else None
        m_prop = target["measure"].get((q,)) if "measure" in target.operation_names else None
        sx_prop = target["sx"].get((q,)) if "sx" in target.operation_names else None
        q_cal[str(q)] = {
            "t1_s": getattr(qp, "t1", None) if qp else None,
            "t2_s": getattr(qp, "t2", None) if qp else None,
            "freq_hz": getattr(qp, "frequency", None) if qp else None,
            "readout_error": getattr(m_prop, "error", None) if m_prop else None,
            "sx_error": getattr(sx_prop, "error", None) if sx_prop else None,
        }
    cz_cal: Dict[str, float] = {}
    if "cz" in target.operation_names:
        for (u, v), prop in target["cz"].items():
            if u < v and prop is not None and prop.error is not None:
                cz_cal[f"{u}-{v}"] = float(prop.error)
    return {
        "backend_name": backend.name,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "num_qubits": backend.num_qubits,
        "dt_s": float(backend.dt),
        "qubits": q_cal,
        "cz_errors": cz_cal,
    }


def verify_isa_circuit(qc: QuantumCircuit, backend: Any) -> None:
    """Assert that every instruction and qarg in `qc` is strictly native ISA on `backend`."""
    target = backend.target
    allowed = set(target.operation_names) | {"barrier"}
    for inst in qc.data:
        name = inst.operation.name
        if name not in allowed:
            raise ValueError(f"Non-ISA instruction {name!r} in {qc.name}")
        q_tup = tuple(qc.find_bit(q).index for q in inst.qubits)
        if name == "delay":
            dur = int(round(float(inst.operation.duration)))
            if dur < 2:
                raise ValueError(f"Delay duration {dur} dt < min_length=2 in {qc.name}")
        elif name != "barrier":
            if q_tup not in target[name]:
                raise ValueError(f"Instruction {name}{q_tup} not supported on {backend.name} in {qc.name}")


# ---------------------------------------------------------------------------
# 1. TESSERACT-MAP-02 Live Package Builder (90 PUBs, B=10 blocks, 2048 shots)
# ---------------------------------------------------------------------------

def prepare_tesseract_map_02_live(
    backend: Any, n_blocks: int = 10, shots_per_pub: int = 2048, seed: int = 20261116
) -> Tuple[List[QuantumCircuit], Dict[str, Any]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    circuits, manifests, path_summary = build_tesseract_map_02_arms(
        backend, n_qubits=40, theta_deg=THETA_DEFAULT_DEG, seed=seed
    )
    qc_u0, qc_u1, union_qubits = build_union_spam_calibration_circuits(manifests)
    # Transpile the two union SPAM circuits trivially onto exact `union_qubits`
    t_u0 = transpile(
        qc_u0, backend=backend, initial_layout=union_qubits, optimization_level=0, seed_transpiler=seed
    )
    t_u1 = transpile(
        qc_u1, backend=backend, initial_layout=union_qubits, optimization_level=1, seed_transpiler=seed
    )

    arm_templates: Dict[str, QuantumCircuit] = {
        "C0_UNION_SPAM_ALL0": t_u0,
        "C0_UNION_SPAM_ALL1": t_u1,
    }
    for k, v in circuits.items():
        if not k.startswith("C0_SPAM"):
            arm_templates[k] = v

    for qc in arm_templates.values():
        verify_isa_circuit(qc, backend)

    rng = np.random.default_rng(seed)
    arm_keys = list(arm_templates.keys())
    pub_circuits: List[QuantumCircuit] = []
    pub_index_records: List[Dict[str, Any]] = []

    for b in range(n_blocks):
        order = list(arm_keys)
        rng.shuffle(order)
        for pos, arm_id in enumerate(order):
            c_copy = arm_templates[arm_id].copy(name=f"tmap02_b{b:02d}_p{pos}_{arm_id}")
            pub_index_records.append({
                "pub_idx": len(pub_circuits),
                "block": b,
                "position_in_block": pos,
                "arm_id": arm_id,
            })
            pub_circuits.append(c_copy)

    qpy_path = OUT_DIR / "tesseract_map_02_live_pubs.qpy"
    qpy_sha = dump_qpy_sha256(pub_circuits, qpy_path)

    est_seconds = round(len(pub_circuits) * shots_per_pub * 250.0e-6 + len(pub_circuits) * 0.10, 1)
    lock_manifest = {
        "experiment_id": "TESSERACT-MAP-02",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "backend_name": backend.name,
        "n_blocks": n_blocks,
        "shots_per_pub": shots_per_pub,
        "total_pubs": len(pub_circuits),
        "total_shots": len(pub_circuits) * shots_per_pub,
        "estimated_qpu_seconds": est_seconds,
        "qpy_path": str(qpy_path),
        "qpy_sha256": qpy_sha,
        "union_active_physical_qubits": union_qubits,
        "arm_manifests": {
            k: {
                "logical_structure": m.logical_structure,
                "layout_policy": m.layout_policy,
                "routing_method": m.routing_method,
                "optimization_level": m.optimization_level,
                "twoq_gate_count": m.twoq_gate_count,
                "circuit_depth": m.circuit_depth,
                "ops_breakdown": m.ops_breakdown,
                "final_physical_layout": m.final_physical_layout,
                "circuit_qasm_sha256": m.circuit_qasm_sha256,
            }
            for k, m in manifests.items()
        },
        "path_summary": path_summary,
        "pub_schedule": pub_index_records,
    }
    lock_path = OUT_DIR / "tesseract_map_02_live_prereg_lock.json"
    lock_path.write_text(json.dumps(lock_manifest, indent=2) + "\n")
    lock_manifest["lock_sha256"] = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    return pub_circuits, lock_manifest


# ---------------------------------------------------------------------------
# 2. CORRELATED-NOISE-01 Multiplexed 10-Coupler ISA Builder
# ---------------------------------------------------------------------------

def _pick_spectators_for_couplers(
    backend: Any, couplers: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Assign one distinct adjacent physical spectator qubit `s` to each coupler `(u, v)`."""
    target = backend.target
    adj: Dict[int, List[int]] = {q: [] for q in range(backend.num_qubits)}
    for u, v in target.build_coupling_map().get_edges():
        if v not in adj[u]:
            adj[u].append(v)
        if u not in adj[v]:
            adj[v].append(u)

    used_qubits = {q for c in couplers for q in c["coupler"]}
    enriched: List[Dict[str, Any]] = []
    for idx, c in enumerate(couplers):
        u, v = c["coupler"]
        cand_s = [nb for nb in (adj[u] + adj[v]) if nb not in used_qubits]
        # Prefer spectator with lowest sx/measure error
        cand_s.sort(
            key=lambda q: (
                float(target["measure"][(q,)].error)
                if (q,) in target["measure"] and target["measure"][(q,)].error is not None
                else 0.05
            )
        )
        spec_q = int(cand_s[0])
        used_qubits.add(spec_q)
        enriched.append({
            "slot_idx": idx,
            "role": "train" if idx < 4 else "held_out",
            "coupler": [int(u), int(v)],
            "spectator": spec_q,
            "twoq_error": c["twoq_error"],
        })
    return enriched


def _append_isa_h(qc: QuantumCircuit, q: int) -> None:
    """Exact native Heron r2 Hadamard: RZ(pi/2) · SX · RZ(pi/2)."""
    qc.rz(math.pi / 2.0, q)
    qc.sx(q)
    qc.rz(math.pi / 2.0, q)


def _append_isa_xy_dd_train(
    qc: QuantumCircuit,
    q: int,
    total_dt: int,
    n_pulses: int,
    shift_dt: int = 0,
    pulse_dt: int = 9,
) -> None:
    """
    Append an exact `total_dt`-tick native XY8/Hahn sequence on physical wire `q`:
      - n_pulses == 0: single `delay(total_dt, q, unit='dt')`
      - n_pulses == 1: Hahn echo `delay((total_dt-pulse_dt)//2) · X · delay(rem)`
      - n_pulses in {8, 16, 32}: XY8x(n/8) sequence (`X, Y, X, Y, Y, X, Y, X` repeated),
        where `Y = RZ(-pi/2) · X · RZ(pi/2)` uses the exact 9-dt `X` envelope + 0-dt virtual-Z!
        Uses `base_start = max(4, (slot_dt - pulse_dt) // 4)` so that for ALL `shift_dt <= slot_dt/2`,
        every pulse sits strictly inside `[base_start, total_dt - 4]` with ZERO boundary wraparound,
        every gap >= 4 dt (>= min_length=2 dt), alpha_u = alpha_v = 0, and alpha_uv(slot/2) = 0!
    """
    if n_pulses <= 0:
        qc.delay(total_dt, q, unit="dt")
        return
    if n_pulses == 1:
        lead = (total_dt - pulse_dt) // 2 + shift_dt
        trail = total_dt - lead - pulse_dt
        qc.delay(lead, q, unit="dt")
        qc.x(q)
        qc.delay(trail, q, unit="dt")
        return

    slot_dt = total_dt // n_pulses
    # For slot_dt=128 (T=8.192us, N=16) and shift_dt<=64: base_start=28 -> max end = 15*128+28+64+9 = 2021 <= 2048!
    # For slot_dt=256 (N=8): base_start=60. For slot_dt=64 (N=32, shift=0): base_start=27.
    base_start = 28 if (slot_dt == 128) else max(4, (slot_dt - pulse_dt - shift_dt) // 2)
    xy8_pattern = ("X", "Y", "X", "Y", "Y", "X", "Y", "X")

    cursor = 0
    for k in range(n_pulses):
        st = k * slot_dt + base_start + shift_dt
        gap = st - cursor
        if gap < 2:
            raise ValueError(f"Invalid DD gap {gap} dt < 2 on wire {q} (slot={slot_dt}, shift={shift_dt})")
        qc.delay(gap, q, unit="dt")
        axis = xy8_pattern[k % 8]
        if axis == "X":
            qc.x(q)
        else:
            qc.rz(-math.pi / 2.0, q)
            qc.x(q)
            qc.rz(math.pi / 2.0, q)
        cursor = st + pulse_dt

    rem = total_dt - cursor
    if rem < 2:
        raise ValueError(f"Invalid trailing DD gap {rem} dt < 2 on wire {q}")
    qc.delay(rem, q, unit="dt")


def _build_multiplexed_1q_isa_circuit(
    num_backend_qubits: int,
    cohort: Sequence[Dict[str, Any]],
    total_dt: int,
    n_pulses: int,
    name: str,
) -> QuantumCircuit:
    """Prepare |+> on all 10 `u` qubits simultaneously, apply idle/echo/XY8, measure X-basis."""
    k = len(cohort)
    qc = QuantumCircuit(num_backend_qubits, k, name=name)
    u_wires = [c["coupler"][0] for c in cohort]
    for u in u_wires:
        _append_isa_h(qc, u)
    qc.barrier(*u_wires)
    for u in u_wires:
        _append_isa_xy_dd_train(qc, u, total_dt=total_dt, n_pulses=n_pulses, shift_dt=0)
    qc.barrier(*u_wires)
    for idx, u in enumerate(u_wires):
        _append_isa_h(qc, u)
        qc.measure(u, idx)
    return qc


def _build_multiplexed_2q_bell_isa_circuit(
    num_backend_qubits: int,
    cohort: Sequence[Dict[str, Any]],
    total_dt: int,
    n_pulses: int,
    shift_dt: int,
    bell_state: str,
    pauli_basis: str,
    spectator_plus: bool,
    name: str,
) -> QuantumCircuit:
    """
    Multiplexed 10-coupler native ISA Bell + staggered XY8 + Pauli-basis readout circuit:
      Classical bits `[2*idx, 2*idx + 1]` record `(u, v)` for coupler `idx in 0..9`.
    """
    k = len(cohort)
    qc = QuantumCircuit(num_backend_qubits, 2 * k, name=name)
    involved: List[int] = []
    for c in cohort:
        u, v = c["coupler"]
        involved.extend([u, v])
        if spectator_plus:
            involved.append(c["spectator"])

    # State preparation: |Phi+> = (|00>+|11>)/sqrt(2) or |Psi+> = (|01>+|10>)/sqrt(2)
    for c in cohort:
        u, v = c["coupler"]
        _append_isa_h(qc, u)
        if bell_state == "psi_plus":
            qc.x(v)
        _append_isa_h(qc, v)
        qc.cz(u, v)
        _append_isa_h(qc, v)
        if spectator_plus:
            _append_isa_h(qc, c["spectator"])

    qc.barrier(*involved)
    for c in cohort:
        u, v = c["coupler"]
        _append_isa_xy_dd_train(qc, u, total_dt=total_dt, n_pulses=n_pulses, shift_dt=0)
        _append_isa_xy_dd_train(qc, v, total_dt=total_dt, n_pulses=n_pulses, shift_dt=shift_dt)
        if spectator_plus:
            # Active sublattice-B spectator pulsed alongside v (or held during window)
            _append_isa_xy_dd_train(qc, c["spectator"], total_dt=total_dt, n_pulses=n_pulses, shift_dt=shift_dt)
    qc.barrier(*involved)

    # Readout rotation on each (u, v) pair
    b0, b1 = pauli_basis[0].upper(), pauli_basis[1].upper()
    for idx, c in enumerate(cohort):
        u, v = c["coupler"]
        for wire, b_ch in ((u, b0), (v, b1)):
            if b_ch == "X":
                _append_isa_h(qc, wire)
            elif b_ch == "Y":
                # Measure Y eigenbasis (|+i> -> |0>, |-i> -> |1>): Sdg · H = RZ(-pi/2) · H
                qc.rz(-math.pi / 2.0, wire)
                _append_isa_h(qc, wire)
        qc.measure(u, 2 * idx)
        qc.measure(v, 2 * idx + 1)
    return qc


def _build_multiplexed_2q_spam_circuit(
    num_backend_qubits: int,
    cohort: Sequence[Dict[str, Any]],
    bit_u: int,
    bit_v: int,
    name: str,
) -> QuantumCircuit:
    k = len(cohort)
    qc = QuantumCircuit(num_backend_qubits, 2 * k, name=name)
    for idx, c in enumerate(cohort):
        u, v = c["coupler"]
        if bit_u == 1:
            qc.x(u)
        if bit_v == 1:
            qc.x(v)
        qc.measure(u, 2 * idx)
        qc.measure(v, 2 * idx + 1)
    return qc


def prepare_correlated_noise_01_live(
    backend: Any, n_blocks: int = 5, shots_per_pub: int = 2048, seed: int = 20261117
) -> Tuple[List[QuantumCircuit], Dict[str, Any]]:
    """
    Build the `dt = 4.0 ns`-aligned 10-coupler (`>= 5`-hop disjoint: 4 train + 6 held-out)
    multiplexed Stage-1 ISA PUB suite and lock its SHA-256 manifest prior to submission.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_couplers = select_disjoint_couplers_on_backend(backend, min_hop_distance=5, max_couplers=10)
    if len(raw_couplers) < 8:
        raise RuntimeError(f"Expected >=8 disjoint >=5-hop couplers on {backend.name}, got {len(raw_couplers)}")
    cohort = _pick_spectators_for_couplers(backend, raw_couplers)

    nq = backend.num_qubits
    templates: List[Tuple[str, str, QuantumCircuit, Dict[str, Any]]] = []

    # 1. 12 single-qubit Ramsey / Echo / Pulse-Density templates on all 10 `u` wires
    for label, t_dt in (("2us", 512), ("4us", 1024), ("8us", 2048), ("16us", 4096)):
        templates.append((
            f"ramsey_{label}", "X",
            _build_multiplexed_1q_isa_circuit(nq, cohort, t_dt, 0, f"cn01_ramsey_{label}"),
            {"kind": "1q", "total_dt": t_dt, "n_dd": 0, "offset_dt": 0},
        ))
        templates.append((
            f"echo1_{label}", "X",
            _build_multiplexed_1q_isa_circuit(nq, cohort, t_dt, 1, f"cn01_echo1_{label}"),
            {"kind": "1q", "total_dt": t_dt, "n_dd": 1, "offset_dt": 0},
        ))
    for n_dd in (8, 16, 32):
        templates.append((
            f"echo_dd{n_dd}_8us", "X",
            _build_multiplexed_1q_isa_circuit(nq, cohort, 2048, n_dd, f"cn01_echo_dd{n_dd}_8us"),
            {"kind": "1q", "total_dt": 2048, "n_dd": n_dd, "offset_dt": 0},
        ))
    templates.append((
        "echo_dd32_16us", "X",
        _build_multiplexed_1q_isa_circuit(nq, cohort, 4096, 32, "cn01_echo_dd32_16us"),
        {"kind": "1q", "total_dt": 4096, "n_dd": 32, "offset_dt": 0},
    ))

    # 2. 2Q |Phi+> stagger sweep (7 offsets + 1 duplicate 0ns A/A drift sentinel = 8 x 3 bases = 24 templates)
    for delta_ns in STAGGER_OFFSETS_NS:
        sh_dt = delta_ns // 4
        for basis in ("XX", "YY", "ZZ"):
            templates.append((
                f"phi_dd16_d{delta_ns}ns", basis,
                _build_multiplexed_2q_bell_isa_circuit(
                    nq, cohort, 2048, 16, sh_dt, "phi_plus", basis, False, f"cn01_phi_d{delta_ns}_{basis}"
                ),
                {"kind": "2q_bell", "bell": "phi_plus", "total_dt": 2048, "n_dd": 16, "offset_dt": sh_dt, "spec_plus": False},
            ))
    # Mandatory within-block A/A sentinel duplicate of phi_dd16_d0ns!
    for basis in ("XX", "YY", "ZZ"):
        templates.append((
            "phi_dd16_d0ns_aa_sentinel", basis,
            _build_multiplexed_2q_bell_isa_circuit(
                nq, cohort, 2048, 16, 0, "phi_plus", basis, False, f"cn01_phi_aa0_{basis}"
            ),
            {"kind": "2q_bell", "bell": "phi_plus", "total_dt": 2048, "n_dd": 16, "offset_dt": 0, "spec_plus": False},
        ))

    # 3. 2Q |Psi+> DFS contrast at {0, 256} ns (2 x 3 = 6 templates)
    for delta_ns in (0, 256):
        sh_dt = delta_ns // 4
        for basis in ("XX", "YY", "ZZ"):
            templates.append((
                f"psi_dd16_d{delta_ns}ns", basis,
                _build_multiplexed_2q_bell_isa_circuit(
                    nq, cohort, 2048, 16, sh_dt, "psi_plus", basis, False, f"cn01_psi_d{delta_ns}_{basis}"
                ),
                {"kind": "2q_bell", "bell": "psi_plus", "total_dt": 2048, "n_dd": 16, "offset_dt": sh_dt, "spec_plus": False},
            ))

    # 4. Spectator |+>_s activation contrast at {0, 256} ns (2 x 3 = 6 templates)
    for delta_ns in (0, 256):
        sh_dt = delta_ns // 4
        for basis in ("XX", "YY", "ZZ"):
            templates.append((
                f"phi_spec_plus_d{delta_ns}ns", basis,
                _build_multiplexed_2q_bell_isa_circuit(
                    nq, cohort, 2048, 16, sh_dt, "phi_plus", basis, True, f"cn01_specp_d{delta_ns}_{basis}"
                ),
                {"kind": "2q_bell", "bell": "phi_plus", "total_dt": 2048, "n_dd": 16, "offset_dt": sh_dt, "spec_plus": True},
            ))

    # Verify 100% of templates pass strict ISA verification
    for _, _, qc_t, _ in templates:
        verify_isa_circuit(qc_t, backend)

    # 4x4 2-qubit SPAM readout calibration templates (|00>, |01>, |10>, |11>) + T=0 Bell ceiling
    spam_templates: List[Tuple[str, str, QuantumCircuit, Dict[str, Any]]] = []
    for bu, bv in ((0, 0), (0, 1), (1, 0), (1, 1)):
        qc_sp = _build_multiplexed_2q_spam_circuit(nq, cohort, bu, bv, f"cn01_spam_{bu}{bv}")
        verify_isa_circuit(qc_sp, backend)
        spam_templates.append((f"spam_{bu}{bv}", "ZZ", qc_sp, {"kind": "2q_spam", "prep": f"{bu}{bv}"}))

    rng = np.random.default_rng(seed)
    pub_circuits: List[QuantumCircuit] = []
    pub_schedule: List[Dict[str, Any]] = []

    # Prepend 4x4 2-qubit SPAM calibration PUBs
    for cid, basis, qc_sp, meta in spam_templates:
        pub_schedule.append({
            "pub_idx": len(pub_circuits),
            "block": -1,
            "condition_id": cid,
            "basis": basis,
            **meta,
        })
        pub_circuits.append(qc_sp.copy(name=f"pub_{len(pub_circuits):03d}_{cid}"))

    # Interleave B randomized blocks of the 48 core diagnostic templates
    n_tpl = len(templates)
    for b in range(n_blocks):
        perm = rng.permutation(n_tpl)
        for pos, idx in enumerate(perm):
            cid, basis, qc_tpl, meta = templates[int(idx)]
            pub_schedule.append({
                "pub_idx": len(pub_circuits),
                "block": b,
                "position_in_block": pos,
                "condition_id": cid,
                "basis": basis,
                **meta,
            })
            pub_circuits.append(qc_tpl.copy(name=f"pub_{len(pub_circuits):03d}_b{b}_{cid}_{basis}"))

    qpy_path = OUT_DIR / "correlated_noise_01_live_pubs.qpy"
    qpy_sha = dump_qpy_sha256(pub_circuits, qpy_path)
    est_seconds = round(len(pub_circuits) * shots_per_pub * 250.0e-6 + len(pub_circuits) * 0.10, 1)

    lock_manifest = {
        "experiment_id": "CORRELATED-NOISE-01-STAGE1",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "backend_name": backend.name,
        "dt_s": HERON_DT_S,
        "pulse_width_s": MARRAKESH_PULSE_WIDTH_S,
        "n_blocks": n_blocks,
        "shots_per_pub": shots_per_pub,
        "total_pubs": len(pub_circuits),
        "total_circuit_shots": len(pub_circuits) * shots_per_pub,
        "total_multiplexed_coupler_shots": len(pub_circuits) * shots_per_pub * len(cohort),
        "estimated_qpu_seconds": est_seconds,
        "qpy_path": str(qpy_path),
        "qpy_sha256": qpy_sha,
        "coupler_cohort": cohort,
        "pub_schedule": pub_schedule,
    }
    lock_path = OUT_DIR / "correlated_noise_01_live_prereg_lock.json"
    lock_path.write_text(json.dumps(lock_manifest, indent=2) + "\n")
    lock_manifest["lock_sha256"] = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    return pub_circuits, lock_manifest


def extract_counts_from_pub_result(pub_res: Any) -> Dict[str, int]:
    """Extract bitstring counts dict cleanly across Qiskit SamplerV2 DataBin register names."""
    data = pub_res.data
    if hasattr(data, "meas"):
        return {str(k): int(v) for k, v in data.meas.get_counts().items()}
    if hasattr(data, "c"):
        return {str(k): int(v) for k, v in data.c.get_counts().items()}
    for field_name in dir(data):
        if not field_name.startswith("_"):
            attr = getattr(data, field_name)
            if hasattr(attr, "get_counts"):
                return {str(k): int(v) for k, v in attr.get_counts().items()}
    raise RuntimeError("Could not locate classical bitarray in SamplerV2 PUB result")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["lock", "submit_tesseract", "submit_cn01"], required=True)
    parser.add_argument("--backend", default="ibm_marrakesh")
    parser.add_argument("--cn01-blocks", type=int, default=5)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svc = connect_service()
    usage = svc.usage()
    rem_sec = usage.get("usage_remaining_seconds", 0)
    print(f"[Quota] Remaining OpenPlan seconds: {rem_sec}s (RU remaining: {usage.get('usage_remaining_ru')})")

    backend = svc.backend(args.backend)
    cal = snapshot_backend_calibration(backend)
    (OUT_DIR / f"{args.backend}_prelock_calibration.json").write_text(json.dumps(cal, indent=2) + "\n")

    if args.step == "lock":
        _, t_lock = prepare_tesseract_map_02_live(backend, n_blocks=10, shots_per_pub=2048)
        _, c_lock = prepare_correlated_noise_01_live(backend, n_blocks=args.cn01_blocks, shots_per_pub=2048)
        print(
            f"[Locked TESSERACT-MAP-02] pubs={t_lock['total_pubs']} shots={t_lock['total_shots']} "
            f"est_s={t_lock['estimated_qpu_seconds']} qpy_sha256={t_lock['qpy_sha256'][:16]}... "
            f"lock_sha256={t_lock['lock_sha256'][:16]}..."
        )
        print(
            f"[Locked CORRELATED-NOISE-01] pubs={c_lock['total_pubs']} shots={c_lock['total_circuit_shots']} "
            f"coupler_shots={c_lock['total_multiplexed_coupler_shots']} est_s={c_lock['estimated_qpu_seconds']} "
            f"qpy_sha256={c_lock['qpy_sha256'][:16]}... lock_sha256={c_lock['lock_sha256'][:16]}..."
        )
        return

    if args.step == "submit_tesseract":
        lock_path = OUT_DIR / "tesseract_map_02_live_prereg_lock.json"
        qpy_path = OUT_DIR / "tesseract_map_02_live_pubs.qpy"
        lock = json.loads(lock_path.read_text())
        with qpy_path.open("rb") as f:
            pubs = qpy.load(f)
        if rem_sec < lock["estimated_qpu_seconds"]:
            raise RuntimeError(f"Insufficient quota ({rem_sec}s) for TESSERACT-MAP-02 ({lock['estimated_qpu_seconds']}s)")
        sampler = SamplerV2(mode=backend)
        job = sampler.run(pubs, shots=int(lock["shots_per_pub"]))
        jid = job.job_id()
        sub_rec = {
            "experiment_id": "TESSERACT-MAP-02",
            "job_id": jid,
            "backend": backend.name,
            "submitted_utc": datetime.now(timezone.utc).isoformat(),
            "prereg_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
            "qpy_sha256": lock["qpy_sha256"],
        }
        (OUT_DIR / "tesseract_map_02_submission_receipt.json").write_text(json.dumps(sub_rec, indent=2) + "\n")
        print(f"[Submitted TESSERACT-MAP-02] job_id={jid} on {backend.name}")
        return

    if args.step == "submit_cn01":
        lock_path = OUT_DIR / "correlated_noise_01_live_prereg_lock.json"
        qpy_path = OUT_DIR / "correlated_noise_01_live_pubs.qpy"
        lock = json.loads(lock_path.read_text())
        with qpy_path.open("rb") as f:
            pubs = qpy.load(f)
        if rem_sec < lock["estimated_qpu_seconds"] * 0.80:
            raise RuntimeError(f"Insufficient quota ({rem_sec}s) for CORRELATED-NOISE-01 ({lock['estimated_qpu_seconds']}s)")
        sampler = SamplerV2(mode=backend)
        job = sampler.run(pubs, shots=int(lock["shots_per_pub"]))
        jid = job.job_id()
        sub_rec = {
            "experiment_id": "CORRELATED-NOISE-01-STAGE1",
            "job_id": jid,
            "backend": backend.name,
            "submitted_utc": datetime.now(timezone.utc).isoformat(),
            "prereg_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
            "qpy_sha256": lock["qpy_sha256"],
        }
        (OUT_DIR / "correlated_noise_01_submission_receipt.json").write_text(json.dumps(sub_rec, indent=2) + "\n")
        print(f"[Submitted CORRELATED-NOISE-01] job_id={jid} on {backend.name}")
        return


if __name__ == "__main__":
    main()
