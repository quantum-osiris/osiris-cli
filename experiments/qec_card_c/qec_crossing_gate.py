#!/usr/bin/env python3
"""Card C (`OSIRIS-QEC`): Offline Repeated-Correction Crossing & Auditable Decoder Gate.

Evaluates when repeated syndrome extraction and decoding reduce logical error per round
below BOTH the bare unencoded qubit (`arm1_unencoded`) AND the encoded-without-correction
memory baseline (`arm2_encoded_no_correction`) at matched physical task duration
`T_task(R) = R * T_round` across `R in {1, 2, 3, 5, 7, 9}`.

Governance & Epistemic Scope (`Card C`, `phase1/verification_queue.md`):
  - `execution_tier = "SIMULATED"`, `provider_job_id = None`, `qpu_seconds = 0`.
  - Self-contained, glass-box space-time Detector Error Model (DEM) + Minimum-Weight
    Perfect Matching (`AuditableMWPMDecoder`, solved via all-pairs Dijkstra + exact
    bitmask dynamic programming for `k <= 16` active defects and exact Edmonds Blossom
    fallback for `k > 16`), cross-checked against an exact maximum-likelihood
    syndrome-table decoder (`ExactMLSyndromeTableDecoder`) on `[[3,1,3]]`.
  - Four `Card C` arms (with Arm 4 separated into offline Pauli-frame tracking vs
    physical feed-forward):
      1. `arm1_unencoded`: 1 bare qubit idling for `R * T_round` + 1 final readout.
      2. `arm2_encoded_no_correction`: Encoded codeword idling for `R * T_round` with no
         mid-circuit syndrome extraction + 1 final transversal readout + terminal decode.
      3. `arm3_postselected_detection`: Repeated `R`-round syndrome extraction retaining
         zero-detector trajectories; conditional logical failure and unconditional
         acceptance rate `A(R)` reported strictly separately.
      4. `arm4a_decoded_pauli_frame` (`offline_pauli_frame`, `0` feed-forward latency/gate
         penalty) and `arm4b_decoded_physical_feedforward` (`physical_feedforward`,
         including classical controller latency `t_ff_us` + conditional 1Q gate error).
  - Scope limit: A single code distance (`d = 3`) answers the crossing question for that
    size only; scaling with distance (`d = 3 -> 5 -> 7`) remains separate, and a
    single-size result is not evidence of scalable fault tolerance.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
FEZ_SNAPSHOT_PATH = REPO_ROOT / "experiments" / "bt2t_fez" / "bt2t_fez_calibration_snapshot.json"

DEFAULT_SEED = 20261124
ROUNDS_GRID: Tuple[int, ...] = (1, 2, 3, 5, 7, 9)
N_SHOTS_REP3_DEFAULT = 30000
N_SHOTS_SURF9_DEFAULT = 40000
N_BOOT_DEFAULT = 1000

MODELED_MECHANISMS: Tuple[str, ...] = (
    "amplitude_damping_t1",
    "phase_damping_t2",
    "coherent_over_rotation_1q_2q",
    "mid_circuit_and_terminal_readout_confusion",
    "active_reset_bit_flip",
    "correlated_spectator_zz_crosstalk",
    "feedforward_classical_latency_and_conditional_gate_error",
    "heavy_hex_degree3_bridge_routing_overhead_for_surface_code",
)

OMITTED_MECHANISMS: Tuple[str, ...] = ("non_computational_state_2_leakage",)

OMITTED_MECHANISMS_IMPACT_NOTE = (
    "Non-computational |2> leakage (and hyperfine Zeeman leakage on trapped ions) is "
    "not explicitly simulated in the 2-level computational Hilbert space. On physical "
    "transmons without active Leakage Reduction Units (LRUs), leakage during strong "
    "mid-circuit readout or two-qubit CZ drives creates multi-round time-correlated "
    "space-time hook/syndrome streaks that reduce effective fault distance and raise "
    "the asymptotic logical error floor epsilon_L at large round counts R."
)

SCOPE_LIMIT_STATEMENT = (
    "A single code distance (d=3) answers the crossing question for that size only. "
    "Scaling with distance remains open in this gate, and a single-size result is not "
    "evidence of scalable fault tolerance."
)

PAULIS_1Q: Tuple[str, ...] = ("X", "Y", "Z")
PAULIS_2Q: Tuple[str, ...] = tuple(
    p1 + p2 for p1 in "IXYZ" for p2 in "IXYZ" if not (p1 == "I" and p2 == "I")
)
PAULI_2Q_BITS: Tuple[Tuple[bool, bool, bool, bool], ...] = tuple(
    (p[0] in ("X", "Y"), p[0] in ("Z", "Y"), p[1] in ("X", "Y"), p[1] in ("Z", "Y"))
    for p in PAULIS_2Q
)

# Exact 2x2 and 4x4 matrices for the 5-qubit Kraus engine
I2 = np.eye(2, dtype=np.complex128)
X2 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
Y2 = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
Z2 = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
H2 = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)
CX4 = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=np.complex128,
)
ZZ4 = np.kron(Z2, Z2)

# Collision-free, stabilizer-commuting, hook-error-safe 4-step Tomita-Svore schedule
# for rotated d=3 [[9,1,3]] surface code:
#   Data qubits 0..8 on 3x3 grid (row = q // 3, col = q % 3)
#   Logical Z_L = Z0 Z3 Z6 (vertical, col 0); Logical X_L = X0 X1 X2 (horizontal, row 0)
#   Z-ancillas 9..12 have horizontal last-two pairs (perpendicular to vertical Z_L)
#   X-ancillas 13..16 have vertical last-two pairs (perpendicular to horizontal X_L)
SURFACE_D3_Z_SCHEDULE: Dict[int, Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]] = {
    9: (None, 0, None, 1),
    10: (1, 2, 4, 5),
    11: (3, 4, 6, 7),
    12: (7, 8, None, None),
}
SURFACE_D3_X_SCHEDULE: Dict[int, Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]] = {
    13: (None, None, 3, 6),
    14: (0, 3, 1, 4),
    15: (4, 7, 5, 8),
    16: (2, 5, None, None),
}
SURFACE_D3_Z_SUPPORTS: Dict[int, Tuple[int, ...]] = {
    9: (0, 1),
    10: (1, 2, 4, 5),
    11: (3, 4, 6, 7),
    12: (7, 8),
}
SURFACE_D3_X_SUPPORTS: Dict[int, Tuple[int, ...]] = {
    13: (3, 6),
    14: (0, 1, 3, 4),
    15: (4, 5, 7, 8),
    16: (2, 5),
}
SURFACE_D3_DATA_GRID_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (3, 4),
    (4, 5),
    (6, 7),
    (7, 8),
    (0, 3),
    (3, 6),
    (1, 4),
    (4, 7),
    (2, 5),
    (5, 8),
)


@dataclass(frozen=True)
class TargetNoiseProfile:
    """Published or drift-perturbed hardware noise regime for Card C simulation."""

    profile_id: str
    target_family: str
    description: str
    p_2q: float
    p_1q: float
    p_readout: float
    p_reset: float
    t1_us: float
    t2_us: float
    t_round_us: float
    t_readout_us: float
    t_reset_us: float
    t_gates_us: float
    t_ff_us: float
    delta_theta_rad: float
    p_corr: float
    p_idle_nominal: float
    supports_native_d3_repetition: bool
    supports_native_d3_surface: bool
    heavy_hex_bridge_cz_overhead_per_round: int
    calibration_provenance: str


@dataclass
class SpaceTimeDEM:
    """Auditable space-time Detector Error Model extracted from a QEC circuit specification."""

    code_id: str
    basis: str
    rounds: int
    n_qubits: int
    n_measurements: int
    n_detectors: int
    detectors: List[List[int]]
    observable: List[int]
    ops: List[Tuple[Any, ...]]
    mechanisms: Dict[Tuple[int, int], float]


def _rx(theta: float) -> np.ndarray:
    c = math.cos(0.5 * theta)
    s = -1.0j * math.sin(0.5 * theta)
    return np.array([[c, s], [s, c]], dtype=np.complex128)


def _rz(theta: float) -> np.ndarray:
    em = np.exp(-0.5j * theta)
    ep = np.exp(0.5j * theta)
    return np.array([[em, 0.0], [0.0, ep]], dtype=np.complex128)


def _rzz(theta: float) -> np.ndarray:
    em = np.exp(-0.5j * theta)
    ep = np.exp(0.5j * theta)
    return np.diag([em, ep, ep, em]).astype(np.complex128)


def gta_t1_t2_pauli_probs(dt_us: float, t1_us: float, t2_us: float) -> Tuple[float, float, float]:
    """Exact Pauli-twirled Kraus probabilities (p_X, p_Y, p_Z) for T1 + T2 relaxation/dephasing."""
    if dt_us <= 0.0:
        return 0.0, 0.0, 0.0
    px = 0.25 * (1.0 - math.exp(-dt_us / t1_us)) if t1_us > 0.0 else 0.0
    py = px
    p_deph_total = 0.5 * (1.0 - math.exp(-dt_us / t2_us)) if t2_us > 0.0 else 0.0
    pz = max(0.0, p_deph_total - py)
    return px, py, pz


def build_published_target_profiles() -> Dict[str, TargetNoiseProfile]:
    """Construct the published `ibm_fez`, SPAM-drifted `ibm_fez`, and Quantinuum `H2-1` regimes."""
    fez_sha = "unavailable"
    if FEZ_SNAPSHOT_PATH.is_file():
        fez_sha = hashlib.sha256(FEZ_SNAPSHOT_PATH.read_bytes()).hexdigest()[:16]

    fez_pub = TargetNoiseProfile(
        profile_id="FEZ_PUBLISHED",
        target_family="ibm_fez (156Q Heron r2 superconducting heavy-hex)",
        description=(
            "Published IBM Heron r2 (`ibm_fez`) calibration regime: CZ=0.0023, SX=0.00031, "
            "RO=0.0085, reset=0.0100, T1=210 us, T2=145 us, T_round=1.65 us."
        ),
        p_2q=0.0023,
        p_1q=0.00031,
        p_readout=0.0085,
        p_reset=0.0100,
        t1_us=210.0,
        t2_us=145.0,
        t_round_us=1.65,
        t_readout_us=1.20,
        t_reset_us=0.30,
        t_gates_us=0.15,
        t_ff_us=0.35,
        delta_theta_rad=0.012,
        p_corr=0.0006,
        p_idle_nominal=round(0.5 * (1.0 - math.exp(-1.65 / 145.0)), 6),
        supports_native_d3_repetition=True,
        supports_native_d3_surface=False,
        heavy_hex_bridge_cz_overhead_per_round=8,
        calibration_provenance=(
            f"experiments/bt2t_fez/bt2t_fez_calibration_snapshot.json (sha256_16={fez_sha}) "
            "& experiments/qec001/qec001_fez_calibration_snapshot.json"
        ),
    )

    fez_drift = TargetNoiseProfile(
        profile_id="FEZ_DRIFTED_SPAM",
        target_family="ibm_fez (Heron r2 heavy-hex with live-observed SPAM staleness)",
        description=(
            "IBM Heron r2 (`ibm_fez`) regime with drifted p_readout=0.0320 reflecting live "
            "PREP_RO_CTRL SPAM degradation observed in OSIRIS-BT2T-001-FEZ."
        ),
        p_2q=0.0023,
        p_1q=0.00031,
        p_readout=0.0320,
        p_reset=0.0100,
        t1_us=210.0,
        t2_us=145.0,
        t_round_us=1.65,
        t_readout_us=1.20,
        t_reset_us=0.30,
        t_gates_us=0.15,
        t_ff_us=0.35,
        delta_theta_rad=0.012,
        p_corr=0.0006,
        p_idle_nominal=round(0.5 * (1.0 - math.exp(-1.65 / 145.0)), 6),
        supports_native_d3_repetition=True,
        supports_native_d3_surface=False,
        heavy_hex_bridge_cz_overhead_per_round=8,
        calibration_provenance=(
            "experiments/bt2t_fez/physical_bt2t_fez_results.json (H3_spam_control_integrity)"
        ),
    )

    h2_pub = TargetNoiseProfile(
        profile_id="H2_1_PUBLISHED",
        target_family="Quantinuum H2-1 (56Q QCCD all-to-all trapped-ion)",
        description=(
            "Published Quantinuum H2-1 specifications: ZZ=0.0012, 1Q=0.00003, RO=0.0015, "
            "reset=0.0015, shuttling memory idle=0.0004/round, all-to-all connectivity."
        ),
        p_2q=0.0012,
        p_1q=0.00003,
        p_readout=0.0015,
        p_reset=0.0015,
        t1_us=1.0e7,
        t2_us=1.5e6,
        t_round_us=1200.0,
        t_readout_us=600.0,
        t_reset_us=150.0,
        t_gates_us=450.0,
        t_ff_us=250.0,
        delta_theta_rad=0.004,
        p_corr=0.00005,
        p_idle_nominal=0.0004,
        supports_native_d3_repetition=True,
        supports_native_d3_surface=True,
        heavy_hex_bridge_cz_overhead_per_round=0,
        calibration_provenance="Published Quantinuum H2-1 system benchmark specifications",
    )

    return {
        fez_pub.profile_id: fez_pub,
        fez_drift.profile_id: fez_drift,
        h2_pub.profile_id: h2_pub,
    }


# ---------------------------------------------------------------------------
# Component 1: Space-Time Detector Error Model & Exact Auditable Decoders
# ---------------------------------------------------------------------------


def _propagate_pauli_fault(
    ops: Sequence[Tuple[Any, ...]],
    n_qubits: int,
    n_measurements: int,
    fault_op_idx: int,
    fault_qubits: Sequence[int],
    fault_paulis: Sequence[str],
    pre_measurement: bool = False,
) -> np.ndarray:
    """Propagate a single Pauli fault through a Clifford QEC circuit specification."""
    x = np.zeros(n_qubits, dtype=np.uint8)
    z = np.zeros(n_qubits, dtype=np.uint8)
    m = np.zeros(n_measurements, dtype=np.uint8)

    def _inject() -> None:
        for q, p in zip(fault_qubits, fault_paulis):
            if p in ("X", "Y"):
                x[q] ^= 1
            if p in ("Z", "Y"):
                z[q] ^= 1

    for idx, op in enumerate(ops):
        if idx == fault_op_idx and pre_measurement:
            _inject()
        kind = op[0]
        if kind == "R":
            x[op[1]] = 0
            z[op[1]] = 0
        elif kind == "H":
            q = op[1]
            x[q], z[q] = z[q], x[q]
        elif kind == "CX":
            c, t = op[1], op[2]
            x[t] ^= x[c]
            z[c] ^= z[t]
        elif kind == "MZ":
            m[op[2]] = x[op[1]]
        if idx == fault_op_idx and not pre_measurement:
            _inject()
    return m


def _extract_mechanisms_from_ops(
    ops: Sequence[Tuple[Any, ...]],
    n_qubits: int,
    n_measurements: int,
    detectors: Sequence[Sequence[int]],
    observable: Sequence[int],
) -> Dict[Tuple[int, int], float]:
    """Aggregate all elementary fault locations into independent DEM `(det_mask, obs_flip) -> p_e`."""
    mechs: Dict[Tuple[int, int], float] = {}

    def _add(m_vec: np.ndarray, prob: float) -> None:
        if prob <= 0.0:
            return
        dmask = 0
        for di, m_idxs in enumerate(detectors):
            if sum(int(m_vec[u]) for u in m_idxs) & 1:
                dmask |= 1 << di
        oflip = sum(int(m_vec[u]) for u in observable) & 1
        if dmask == 0 and oflip == 0:
            return
        key = (dmask, oflip)
        p_old = mechs.get(key, 0.0)
        mechs[key] = p_old * (1.0 - prob) + prob * (1.0 - p_old)

    for idx, op in enumerate(ops):
        kind = op[0]
        if kind == "R":
            m_vec = _propagate_pauli_fault(ops, n_qubits, n_measurements, idx, (op[1],), ("X",))
            _add(m_vec, float(op[2]))
        elif kind in ("H", "DEPOL1"):
            pe = float(op[2]) / 3.0
            for pa in PAULIS_1Q:
                m_vec = _propagate_pauli_fault(ops, n_qubits, n_measurements, idx, (op[1],), (pa,))
                _add(m_vec, pe)
        elif kind == "PAULI1":
            for pa, pe in zip(PAULIS_1Q, op[2:5]):
                m_vec = _propagate_pauli_fault(ops, n_qubits, n_measurements, idx, (op[1],), (pa,))
                _add(m_vec, float(pe))
        elif kind == "ZZ_CORR":
            m_vec = _propagate_pauli_fault(
                ops, n_qubits, n_measurements, idx, (op[1], op[2]), ("Z", "Z")
            )
            _add(m_vec, float(op[3]))
        elif kind == "CX":
            pe = float(op[3]) / 15.0
            for pa2 in PAULIS_2Q:
                m_vec = _propagate_pauli_fault(
                    ops, n_qubits, n_measurements, idx, (op[1], op[2]), (pa2[0], pa2[1])
                )
                _add(m_vec, pe)
        elif kind == "MZ":
            m_vec = _propagate_pauli_fault(
                ops, n_qubits, n_measurements, idx, (op[1],), ("X",), pre_measurement=True
            )
            _add(m_vec, float(op[3]))

    return mechs


def build_repetition_d3_dem(
    rounds: int, profile: TargetNoiseProfile, basis: str = "X"
) -> SpaceTimeDEM:
    """Build auditable space-time DEM for 5Q `[[3,1,3]]` repetition code (`Z` or `X` memory)."""
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    if basis not in ("Z", "X"):
        raise ValueError(f"basis must be 'Z' or 'X', got {basis}")

    px, py, pz = gta_t1_t2_pauli_probs(profile.t_round_us, profile.t1_us, profile.t2_us)
    p_coh = math.sin(0.5 * profile.delta_theta_rad) ** 2
    pz_eff = pz + p_coh * (1.0 - 2.0 * pz)
    p1_eff = min(0.49, profile.p_1q + p_coh)
    p2_eff = min(0.49, profile.p_2q + p_coh)

    ops: List[Tuple[Any, ...]] = []
    for d in (0, 1, 2):
        ops.append(("R", d, profile.p_reset))
        if basis == "X":
            ops.append(("H", d, p1_eff))

    mid = 0
    anc_meas: Dict[Tuple[int, int], int] = {}
    coupler_edges = ((0, 3), (3, 1), (1, 4), (4, 2))

    for r in range(rounds):
        for a in (3, 4):
            ops.append(("R", a, profile.p_reset))
            if basis == "X":
                ops.append(("H", a, p1_eff))
        if basis == "Z":
            ops.append(("CX", 0, 3, p2_eff))
            ops.append(("CX", 1, 4, p2_eff))
            ops.append(("CX", 1, 3, p2_eff))
            ops.append(("CX", 2, 4, p2_eff))
        else:
            ops.append(("CX", 3, 0, p2_eff))
            ops.append(("CX", 4, 1, p2_eff))
            ops.append(("CX", 3, 1, p2_eff))
            ops.append(("CX", 4, 2, p2_eff))

        for u, v in coupler_edges:
            ops.append(("ZZ_CORR", u, v, profile.p_corr))
        for d in (0, 1, 2):
            ops.append(("PAULI1", d, px, py, pz_eff))
        for s_idx, a in enumerate((3, 4)):
            if basis == "X":
                ops.append(("H", a, p1_eff))
            ops.append(("MZ", a, mid, profile.p_readout))
            anc_meas[(r, s_idx)] = mid
            mid += 1

    data_meas: Dict[int, int] = {}
    for d in (0, 1, 2):
        if basis == "X":
            ops.append(("H", d, p1_eff))
        ops.append(("MZ", d, mid, profile.p_readout))
        data_meas[d] = mid
        mid += 1

    detectors: List[List[int]] = []
    for r in range(rounds):
        for s_idx in (0, 1):
            if r == 0:
                detectors.append([anc_meas[(0, s_idx)]])
            else:
                detectors.append([anc_meas[(r, s_idx)], anc_meas[(r - 1, s_idx)]])
    detectors.append([anc_meas[(rounds - 1, 0)], data_meas[0], data_meas[1]])
    detectors.append([anc_meas[(rounds - 1, 1)], data_meas[1], data_meas[2]])

    observable = [data_meas[0], data_meas[1], data_meas[2]]
    mechs = _extract_mechanisms_from_ops(ops, 5, mid, detectors, observable)
    return SpaceTimeDEM(
        code_id="repetition_d3_5q",
        basis=basis,
        rounds=rounds,
        n_qubits=5,
        n_measurements=mid,
        n_detectors=len(detectors),
        detectors=detectors,
        observable=observable,
        ops=ops,
        mechanisms=mechs,
    )


def build_rotated_surface_d3_dem(
    rounds: int,
    profile: TargetNoiseProfile,
    basis: str = "X",
    include_topology_overhead: bool = True,
) -> SpaceTimeDEM:
    """Build auditable space-time DEM for 17Q rotated `d=3` `[[9,1,3]]` surface code."""
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    if basis not in ("Z", "X"):
        raise ValueError(f"basis must be 'Z' or 'X', got {basis}")

    bridge_cz = profile.heavy_hex_bridge_cz_overhead_per_round if include_topology_overhead else 0
    topo_2q_scale = (24.0 + float(bridge_cz)) / 24.0
    extra_gate_us = 0.04 * (bridge_cz / 4.0) if bridge_cz > 0 else 0.0

    px, py, pz = gta_t1_t2_pauli_probs(
        profile.t_round_us + extra_gate_us, profile.t1_us, profile.t2_us
    )
    p_coh = math.sin(0.5 * profile.delta_theta_rad) ** 2
    pz_eff = pz + p_coh * (1.0 - 2.0 * pz)
    p1_eff = min(0.49, profile.p_1q + p_coh)
    p2_eff = min(0.49, topo_2q_scale * (profile.p_2q + p_coh))

    ops: List[Tuple[Any, ...]] = []
    for d in range(9):
        ops.append(("R", d, profile.p_reset))
        if basis == "X":
            ops.append(("H", d, p1_eff))

    phys_coupler_edges: List[Tuple[int, int]] = []
    for a, qs in list(SURFACE_D3_Z_SCHEDULE.items()) + list(SURFACE_D3_X_SCHEDULE.items()):
        for q in qs:
            if q is not None:
                phys_coupler_edges.append((q, a))

    mid = 0
    anc_meas: Dict[Tuple[int, int], int] = {}
    for r in range(rounds):
        for a in range(9, 13):
            ops.append(("R", a, profile.p_reset))
        for a in range(13, 17):
            ops.append(("R", a, profile.p_reset))
            ops.append(("H", a, p1_eff))

        for step in range(4):
            for a, qs in SURFACE_D3_Z_SCHEDULE.items():
                dq = qs[step]
                if dq is not None:
                    ops.append(("CX", dq, a, p2_eff))
            for a, qs in SURFACE_D3_X_SCHEDULE.items():
                dq = qs[step]
                if dq is not None:
                    ops.append(("CX", a, dq, p2_eff))

        for u, v in phys_coupler_edges:
            ops.append(("ZZ_CORR", u, v, profile.p_corr))
        for d in range(9):
            ops.append(("PAULI1", d, px, py, pz_eff))

        for a in range(13, 17):
            ops.append(("H", a, p1_eff))
        for a in range(9, 17):
            ops.append(("MZ", a, mid, profile.p_readout))
            anc_meas[(r, a)] = mid
            mid += 1

    data_meas: Dict[int, int] = {}
    for d in range(9):
        if basis == "X":
            ops.append(("H", d, p1_eff))
        ops.append(("MZ", d, mid, profile.p_readout))
        data_meas[d] = mid
        mid += 1

    detectors: List[List[int]] = []
    if basis == "Z":
        supports = SURFACE_D3_Z_SUPPORTS
        active_ancs = tuple(range(9, 13))
        observable = [data_meas[0], data_meas[3], data_meas[6]]
    else:
        supports = SURFACE_D3_X_SUPPORTS
        active_ancs = tuple(range(13, 17))
        observable = [data_meas[0], data_meas[1], data_meas[2]]

    for r in range(rounds):
        for a in active_ancs:
            if r == 0:
                detectors.append([anc_meas[(0, a)]])
            else:
                detectors.append([anc_meas[(r, a)], anc_meas[(r - 1, a)]])
    for a in active_ancs:
        detectors.append([anc_meas[(rounds - 1, a)]] + [data_meas[d] for d in supports[a]])

    mechs = _extract_mechanisms_from_ops(ops, 17, mid, detectors, observable)
    return SpaceTimeDEM(
        code_id="rotated_surface_d3_17q",
        basis=basis,
        rounds=rounds,
        n_qubits=17,
        n_measurements=mid,
        n_detectors=len(detectors),
        detectors=detectors,
        observable=observable,
        ops=ops,
        mechanisms=mechs,
    )


class AuditableMWPMDecoder:
    """Glass-box Space-Time Minimum-Weight Perfect Matching (MWPM) Decoder."""

    def __init__(self, dem: SpaceTimeDEM) -> None:
        self.dem = dem
        self.n_det = dem.n_detectors
        self.boundary_node = self.n_det
        n_nodes = self.n_det + 1

        edge_probs: Dict[Tuple[int, int, int], float] = {}
        self.hyperedge_decomposed_count = 0

        for (dmask, oflip), prob in dem.mechanisms.items():
            if dmask == 0:
                continue
            bits = [i for i in range(self.n_det) if (dmask >> i) & 1]
            if len(bits) == 1:
                u, v = bits[0], self.boundary_node
                key = (u, v, oflip)
                p_old = edge_probs.get(key, 0.0)
                edge_probs[key] = p_old * (1.0 - prob) + prob * (1.0 - p_old)
            elif len(bits) == 2:
                u, v = (bits[0], bits[1]) if bits[0] < bits[1] else (bits[1], bits[0])
                key = (u, v, oflip)
                p_old = edge_probs.get(key, 0.0)
                edge_probs[key] = p_old * (1.0 - prob) + prob * (1.0 - p_old)
            else:
                self.hyperedge_decomposed_count += 1

        adj: List[List[Tuple[int, float, int]]] = [[] for _ in range(n_nodes)]
        self.graph_edges: List[Dict[str, Any]] = []
        for (u, v, oflip), pe in sorted(edge_probs.items()):
            pe_clamped = min(max(pe, 1e-12), 0.499999)
            w = -math.log(pe_clamped / (1.0 - pe_clamped))
            adj[u].append((v, w, oflip))
            adj[v].append((u, w, oflip))
            self.graph_edges.append(
                {
                    "u": u,
                    "v": "BOUNDARY" if v == self.boundary_node else v,
                    "prob": pe,
                    "weight": w,
                    "obs_flip": oflip,
                }
            )

        dist = np.full((n_nodes, n_nodes), math.inf, dtype=np.float64)
        obs = np.zeros((n_nodes, n_nodes), dtype=np.uint8)
        b_node = self.boundary_node

        for src in range(n_nodes):
            dist[src, src] = 0.0
            pq: List[Tuple[float, int, int]] = [(0.0, src, 0)]
            while pq:
                d_u, u, o_u = heapq.heappop(pq)
                if d_u > dist[src, u] + 1e-14:
                    continue
                if u == b_node and src != b_node:
                    continue
                for v, w, o_e in adj[u]:
                    nd = d_u + w
                    if nd < dist[src, v] - 1e-12:
                        dist[src, v] = nd
                        obs[src, v] = o_u ^ o_e
                        heapq.heappush(pq, (nd, v, o_u ^ o_e))

        self.dist_matrix = dist
        self.obs_matrix = obs
        self._cache: Dict[int, int] = {0: 0}

    def explain_matching(self, syn_mask: int) -> List[Dict[str, Any]]:
        """Return the exact matched defect pairs, weights, and observable flips for `syn_mask`."""
        b_node = self.boundary_node
        nodes = [i for i in range(b_node) if (syn_mask >> i) & 1]
        k = len(nodes)
        if k == 0:
            return []

        dist = self.dist_matrix
        obs = self.obs_matrix
        pairs: List[Tuple[int, Optional[int]]] = []

        if k <= 16:
            size = 1 << k
            dp = [math.inf] * size
            choice: List[Optional[Tuple[int, int]]] = [None] * size
            dp[0] = 0.0
            bdist = [float(dist[u, b_node]) for u in nodes]
            dmat = [[float(dist[u, v]) for v in nodes] for u in nodes]
            for mask in range(1, size):
                lsb = mask & -mask
                i = lsb.bit_length() - 1
                rem = mask ^ lsb
                best = bdist[i] + dp[rem]
                best_partner = -1
                sub = rem
                row_i = dmat[i]
                while sub:
                    bit = sub & -sub
                    j = bit.bit_length() - 1
                    cand = row_i[j] + dp[rem ^ bit]
                    if cand < best - 1e-12:
                        best = cand
                        best_partner = j
                    sub ^= bit
                dp[mask] = best
                choice[mask] = (i, best_partner)

            mask = size - 1
            while mask:
                step = choice[mask]
                assert step is not None
                i, p = step
                if p == -1:
                    pairs.append((nodes[i], None))
                    mask ^= 1 << i
                else:
                    pairs.append((nodes[i], nodes[p]))
                    mask ^= (1 << i) ^ (1 << p)
        else:
            g = nx.Graph()
            for i in range(k):
                g.add_edge(i, k + i, weight=float(dist[nodes[i], b_node]))
                for j in range(i + 1, k):
                    g.add_edge(i, j, weight=float(dist[nodes[i], nodes[j]]))
                    g.add_edge(k + i, k + j, weight=0.0)
            matching = nx.algorithms.matching.min_weight_matching(g)
            for u_idx, v_idx in matching:
                if u_idx > v_idx:
                    u_idx, v_idx = v_idx, u_idx
                if u_idx < k and v_idx < k:
                    pairs.append((nodes[u_idx], nodes[v_idx]))
                elif u_idx < k and v_idx >= k:
                    pairs.append((nodes[u_idx], None))

        explained: List[Dict[str, Any]] = []
        for u, v in pairs:
            if v is None:
                explained.append(
                    {
                        "u": u,
                        "v": "BOUNDARY",
                        "weight": float(dist[u, b_node]),
                        "obs_flip": int(obs[u, b_node]),
                    }
                )
            else:
                explained.append(
                    {
                        "u": u,
                        "v": v,
                        "weight": float(dist[u, v]),
                        "obs_flip": int(obs[u, v]),
                    }
                )
        return explained

    def decode_mask(self, syn_mask: int) -> int:
        """Decode a detector bitmask into the predicted logical observable flip (`0` or `1`)."""
        cached = self._cache.get(syn_mask)
        if cached is not None:
            return cached

        b_node = self.boundary_node
        dist = self.dist_matrix
        obs = self.obs_matrix
        nodes = [i for i in range(b_node) if (syn_mask >> i) & 1]
        k = len(nodes)

        if k == 1:
            res = int(obs[nodes[0], b_node])
        elif k == 2:
            u, v = nodes[0], nodes[1]
            if dist[u, v] <= dist[u, b_node] + dist[v, b_node] + 1e-12:
                res = int(obs[u, v])
            else:
                res = int(obs[u, b_node] ^ obs[v, b_node])
        else:
            res = 0
            for item in self.explain_matching(syn_mask):
                res ^= int(item["obs_flip"])

        if len(self._cache) < 131072:
            self._cache[syn_mask] = res
        return res

    def decode_batch(self, syn_masks: np.ndarray) -> np.ndarray:
        """Decode an array of integer detector bitmasks."""
        out = np.empty(syn_masks.shape[0], dtype=np.uint8)
        for i, sm in enumerate(syn_masks):
            out[i] = self.decode_mask(int(sm))
        return out


def auditable_mwpm_decoder(dem: SpaceTimeDEM) -> AuditableMWPMDecoder:
    """Factory returning an `AuditableMWPMDecoder` for any `SpaceTimeDEM`."""
    return AuditableMWPMDecoder(dem)


class ExactMLSyndromeTableDecoder:
    """Exact Maximum-Likelihood Syndrome-Coset Table Decoder for `[[3,1,3]]` (`n_det <= 20`)."""

    def __init__(self, dem: SpaceTimeDEM) -> None:
        if dem.n_detectors > 20:
            raise ValueError(
                f"Exact syndrome table requires n_detectors <= 20, got {dem.n_detectors}"
            )
        self.dem = dem
        self.n_det = dem.n_detectors
        n_syn = 1 << self.n_det
        table = np.zeros((n_syn, 2), dtype=np.float64)
        table[0, 0] = 1.0
        idx_all = np.arange(n_syn, dtype=np.int64)

        for (dmask, oflip), pe in dem.mechanisms.items():
            prev = table.copy()
            partner = idx_all ^ dmask
            if oflip == 0:
                table[:, 0] = (1.0 - pe) * prev[:, 0] + pe * prev[partner, 0]
                table[:, 1] = (1.0 - pe) * prev[:, 1] + pe * prev[partner, 1]
            else:
                table[:, 0] = (1.0 - pe) * prev[:, 0] + pe * prev[partner, 1]
                table[:, 1] = (1.0 - pe) * prev[:, 1] + pe * prev[partner, 0]

        self.joint_table = table
        self.predictions = (table[:, 1] > table[:, 0]).astype(np.uint8)
        self.exact_logical_failure_prob = float(
            np.sum(np.minimum(table[:, 0], table[:, 1]))
        )

    def decode_mask(self, syn_mask: int) -> int:
        return int(self.predictions[syn_mask])

    def decode_batch(self, syn_masks: np.ndarray) -> np.ndarray:
        return self.predictions[syn_masks]


def audit_single_fault_recovery(
    code: str, basis: str, rounds: int = 3, profile: Optional[TargetNoiseProfile] = None
) -> Dict[str, Any]:
    """Exhaustively verify 100% single-fault recovery across EVERY circuit fault location."""
    if profile is None:
        profile = build_published_target_profiles()["FEZ_PUBLISHED"]
    if code == "repetition_d3_5q":
        dem = build_repetition_d3_dem(rounds=rounds, profile=profile, basis=basis)
    elif code == "rotated_surface_d3_17q":
        dem = build_rotated_surface_d3_dem(rounds=rounds, profile=profile, basis=basis)
    else:
        raise ValueError(f"Unsupported code: {code}")

    decoder = auditable_mwpm_decoder(dem)
    total_locations = 0
    recovered_locations = 0
    failures: List[Dict[str, Any]] = []

    for idx, op in enumerate(dem.ops):
        kind = op[0]
        fault_specs: List[Tuple[Tuple[int, ...], Tuple[str, ...], bool]] = []
        if kind == "R":
            fault_specs.append(((op[1],), ("X",), False))
        elif kind in ("H", "PAULI1", "DEPOL1"):
            for pa in PAULIS_1Q:
                fault_specs.append(((op[1],), (pa,), False))
        elif kind == "ZZ_CORR":
            fault_specs.append(((op[1], op[2]), ("Z", "Z"), False))
        elif kind == "CX":
            for pa2 in PAULIS_2Q:
                fault_specs.append(((op[1], op[2]), (pa2[0], pa2[1]), False))
        elif kind == "MZ":
            fault_specs.append(((op[1],), ("X",), True))

        for qs, ps, pre_m in fault_specs:
            m_vec = _propagate_pauli_fault(
                dem.ops, dem.n_qubits, dem.n_measurements, idx, qs, ps, pre_measurement=pre_m
            )
            dmask = 0
            for di, m_idxs in enumerate(dem.detectors):
                if sum(int(m_vec[u]) for u in m_idxs) & 1:
                    dmask |= 1 << di
            true_obs = sum(int(m_vec[u]) for u in dem.observable) & 1
            pred_obs = decoder.decode_mask(dmask)
            total_locations += 1
            if pred_obs == true_obs:
                recovered_locations += 1
            else:
                failures.append(
                    {
                        "op_index": idx,
                        "op": str(op),
                        "paulis": ps,
                        "det_mask": dmask,
                        "true_obs": true_obs,
                        "pred_obs": pred_obs,
                    }
                )

    ml_crosscheck: Optional[Dict[str, Any]] = None
    if code == "repetition_d3_5q" and dem.n_detectors <= 16:
        ml_dec = ExactMLSyndromeTableDecoder(dem)
        n_syn = 1 << dem.n_detectors
        mwpm_all = np.array([decoder.decode_mask(s) for s in range(n_syn)], dtype=np.uint8)
        sf_masks = sorted({dm for (dm, _) in dem.mechanisms})
        sf_exact_match = all(int(mwpm_all[s]) == ml_dec.decode_mask(s) for s in sf_masks)
        mwpm_dem_p_fail = float(
            np.sum(ml_dec.joint_table[np.arange(n_syn), 1 - mwpm_all])
        )
        ml_crosscheck = {
            "n_syndromes": n_syn,
            "table_prob_sum": round(float(np.sum(ml_dec.joint_table)), 12),
            "all_syndromes_agree_count": int(np.sum(mwpm_all == ml_dec.predictions)),
            "single_fault_syndromes_exact_match": sf_exact_match,
            "exact_ml_dem_p_fail": round(ml_dec.exact_logical_failure_prob, 8),
            "mwpm_dem_p_fail": round(mwpm_dem_p_fail, 8),
        }

    return {
        "code_id": code,
        "basis": basis,
        "rounds": rounds,
        "n_detectors": dem.n_detectors,
        "n_graph_edges": len(decoder.graph_edges),
        "total_single_fault_locations": total_locations,
        "recovered_single_fault_locations": recovered_locations,
        "single_fault_recovery_rate": (
            recovered_locations / total_locations if total_locations > 0 else 1.0
        ),
        "zero_single_fault_failures": len(failures) == 0,
        "ml_syndrome_table_crosscheck": ml_crosscheck,
    }


# ---------------------------------------------------------------------------
# Component 2A: Exact 5Q Kraus Density-Matrix & Quantum-Instrument Engine
# ---------------------------------------------------------------------------


def _dm_apply_1q_unitary(rho: np.ndarray, u2: np.ndarray, q: int, n_qubits: int) -> np.ndarray:
    dim = 1 << n_qubits
    t = rho.reshape((2,) * (2 * n_qubits))
    t = np.moveaxis(np.tensordot(u2, t, axes=([1], [q])), 0, q)
    t = np.moveaxis(np.tensordot(u2.conj(), t, axes=([1], [n_qubits + q])), 0, n_qubits + q)
    return t.reshape((dim, dim))


def _dm_apply_2q_unitary(
    rho: np.ndarray, u4: np.ndarray, q1: int, q2: int, n_qubits: int
) -> np.ndarray:
    dim = 1 << n_qubits
    t = rho.reshape((2,) * (2 * n_qubits))
    ut = u4.reshape(2, 2, 2, 2)
    t = np.moveaxis(np.tensordot(ut, t, axes=([2, 3], [q1, q2])), [0, 1], [q1, q2])
    t = np.moveaxis(
        np.tensordot(ut.conj(), t, axes=([2, 3], [n_qubits + q1, n_qubits + q2])),
        [0, 1],
        [n_qubits + q1, n_qubits + q2],
    )
    return t.reshape((dim, dim))


def _dm_apply_t1_t2(
    rho: np.ndarray, dt_us: float, t1_us: float, t2_us: float, q: int, n_qubits: int
) -> np.ndarray:
    if dt_us <= 0.0:
        return rho
    dim = 1 << n_qubits
    gamma1 = 1.0 - math.exp(-dt_us / t1_us) if t1_us > 0.0 else 0.0
    rate_phi = max(
        0.0, (2.0 / t2_us if t2_us > 0.0 else 0.0) - (1.0 / t1_us if t1_us > 0.0 else 0.0)
    )
    gamma_phi = 1.0 - math.exp(-dt_us * rate_phi)

    k0 = np.array([[1.0, 0.0], [0.0, math.sqrt(1.0 - gamma1)]], dtype=np.complex128)
    k1 = np.array([[0.0, math.sqrt(gamma1)], [0.0, 0.0]], dtype=np.complex128)
    p0 = np.array([[1.0, 0.0], [0.0, math.sqrt(1.0 - gamma_phi)]], dtype=np.complex128)
    p1 = np.array([[0.0, 0.0], [0.0, math.sqrt(gamma_phi)]], dtype=np.complex128)

    out = np.zeros_like(rho)
    t = rho.reshape((2,) * (2 * n_qubits))
    for p_mat in (p0, p1):
        for k_mat in (k0, k1):
            op = p_mat @ k_mat
            if np.linalg.norm(op) < 1e-15:
                continue
            s = np.moveaxis(np.tensordot(op, t, axes=([1], [q])), 0, q)
            s = np.moveaxis(
                np.tensordot(op.conj(), s, axes=([1], [n_qubits + q])), 0, n_qubits + q
            )
            out += s.reshape((dim, dim))
    return out


def _dm_apply_depol1(rho: np.ndarray, p: float, q: int, n_qubits: int) -> np.ndarray:
    if p <= 0.0:
        return rho
    return (1.0 - p) * rho + (p / 3.0) * (
        _dm_apply_1q_unitary(rho, X2, q, n_qubits)
        + _dm_apply_1q_unitary(rho, Y2, q, n_qubits)
        + _dm_apply_1q_unitary(rho, Z2, q, n_qubits)
    )


def _dm_apply_depol2(
    rho: np.ndarray, p: float, q1: int, q2: int, n_qubits: int
) -> np.ndarray:
    if p <= 0.0:
        return rho
    out = (1.0 - p) * rho
    paulis = (I2, X2, Y2, Z2)
    w = p / 15.0
    for i in range(4):
        for j in range(4):
            if i == 0 and j == 0:
                continue
            out += w * _dm_apply_2q_unitary(rho, np.kron(paulis[i], paulis[j]), q1, q2, n_qubits)
    return out


def analytical_arm1_unencoded_failure(
    rounds: int,
    profile: TargetNoiseProfile,
    basis: str = "X",
    include_spam_and_coherent: bool = True,
) -> float:
    """Closed-form analytical single-qubit failure probability for `arm1_unencoded`."""
    dt_us = rounds * profile.t_round_us
    if not include_spam_and_coherent:
        if basis == "X":
            return 0.5 * (1.0 - math.exp(-dt_us / profile.t2_us))
        return 0.5 * (1.0 - math.exp(-dt_us / profile.t1_us))

    if basis == "X":
        phi = rounds * profile.delta_theta_rad
        dth = profile.delta_theta_rad
        pol = (
            (1.0 - 2.0 * profile.p_reset)
            * (1.0 - 4.0 * profile.p_1q / 3.0)
            * math.exp(-dt_us / profile.t2_us)
            * math.cos(phi + dth)
            * (1.0 - 4.0 * profile.p_1q / 3.0)
            * (1.0 - 2.0 * profile.p_readout)
        )
        return 0.5 * (1.0 - pol)

    z0 = 1.0 - 2.0 * profile.p_reset
    z1 = -(1.0 - 2.0 * profile.p_reset)
    e1 = math.exp(-dt_us / profile.t1_us)
    z0_after = 1.0 - (1.0 - z0) * e1
    z1_after = 1.0 - (1.0 - z1) * e1
    ro_scale = 1.0 - 2.0 * profile.p_readout
    p_err_0 = 0.5 * (1.0 - z0_after * ro_scale)
    p_err_1 = 0.5 * (1.0 + z1_after * ro_scale)
    return 0.5 * (p_err_0 + p_err_1)


def simulate_5q_arm1_kraus_exact(
    rounds: int,
    profile: TargetNoiseProfile,
    basis: str = "X",
    include_spam_and_coherent: bool = True,
) -> float:
    """Evolve `arm1_unencoded` inside the full `32 x 32` 5-qubit Kraus density-matrix space."""
    if not include_spam_and_coherent:
        rho = np.zeros((32, 32), dtype=np.complex128)
        rho[0, 0] = 1.0
        if basis == "X":
            rho = _dm_apply_1q_unitary(rho, H2, 0, 5)
            for _ in range(rounds):
                rho = _dm_apply_t1_t2(rho, profile.t_round_us, profile.t1_us, profile.t2_us, 0, 5)
            rho = _dm_apply_1q_unitary(rho, H2, 0, 5)
            diag = np.real(np.diag(rho)).reshape(2, 16)
            return float(np.sum(diag[1]))
        errs = []
        for init_bit in (0, 1):
            r_b = np.zeros((32, 32), dtype=np.complex128)
            r_b[0, 0] = 1.0
            if init_bit == 1:
                r_b = _dm_apply_1q_unitary(r_b, X2, 0, 5)
            for _ in range(rounds):
                r_b = _dm_apply_t1_t2(r_b, profile.t_round_us, profile.t1_us, profile.t2_us, 0, 5)
            diag = np.real(np.diag(r_b)).reshape(2, 16)
            errs.append(float(np.sum(diag[1 - init_bit])))
        return 0.5 * (errs[0] + errs[1])

    dth = profile.delta_theta_rad
    pro = profile.p_readout
    if basis == "X":
        rho = np.zeros((32, 32), dtype=np.complex128)
        rho[0, 0] = 1.0
        if profile.p_reset > 0.0:
            rho = (1.0 - profile.p_reset) * rho + profile.p_reset * _dm_apply_1q_unitary(
                rho, X2, 0, 5
            )
        rho = _dm_apply_1q_unitary(rho, _rx(dth) @ H2, 0, 5)
        rho = _dm_apply_depol1(rho, profile.p_1q, 0, 5)
        for _ in range(rounds):
            rho = _dm_apply_t1_t2(rho, profile.t_round_us, profile.t1_us, profile.t2_us, 0, 5)
            rho = _dm_apply_1q_unitary(rho, _rz(dth), 0, 5)
        rho = _dm_apply_1q_unitary(rho, _rx(dth) @ H2, 0, 5)
        rho = _dm_apply_depol1(rho, profile.p_1q, 0, 5)
        diag = np.real(np.diag(rho)).reshape(2, 16)
        p1_raw = float(np.sum(diag[1]))
        return p1_raw * (1.0 - pro) + (1.0 - p1_raw) * pro

    errs = []
    for init_bit in (0, 1):
        rho = np.zeros((32, 32), dtype=np.complex128)
        rho[0, 0] = 1.0
        if profile.p_reset > 0.0:
            rho = (1.0 - profile.p_reset) * rho + profile.p_reset * _dm_apply_1q_unitary(
                rho, X2, 0, 5
            )
        if init_bit == 1:
            rho = _dm_apply_1q_unitary(rho, X2, 0, 5)
        for _ in range(rounds):
            rho = _dm_apply_t1_t2(rho, profile.t_round_us, profile.t1_us, profile.t2_us, 0, 5)
        diag = np.real(np.diag(rho)).reshape(2, 16)
        p_wrong = float(np.sum(diag[1 - init_bit]))
        errs.append(p_wrong * (1.0 - pro) + (1.0 - p_wrong) * pro)
    return 0.5 * (errs[0] + errs[1])


def _build_rep3_kraus_superoperators(
    profile: TargetNoiseProfile,
) -> Dict[str, Any]:
    """Construct exact `64 x 64` 3-data-qubit Kraus quantum-instrument superoperators from `32 x 32` 5Q maps."""
    dth = profile.delta_theta_rad
    pres = profile.p_reset
    pro = profile.p_readout
    pc = profile.p_corr

    rho_init = np.zeros((8, 8), dtype=np.complex128)
    rho_init[0, 0] = 1.0
    for d in (0, 1, 2):
        if pres > 0.0:
            rho_init = (1.0 - pres) * rho_init + pres * _dm_apply_1q_unitary(rho_init, X2, d, 3)
        rho_init = _dm_apply_1q_unitary(rho_init, _rx(dth) @ H2, d, 3)
        rho_init = _dm_apply_depol1(rho_init, profile.p_1q, d, 3)

    def _one_idle_round_3q(rho8: np.ndarray) -> np.ndarray:
        out = rho8
        for d in (0, 1, 2):
            out = _dm_apply_t1_t2(out, profile.t_round_us, profile.t1_us, profile.t2_us, d, 3)
            out = _dm_apply_1q_unitary(out, _rz(dth), d, 3)
        for u, v in ((0, 1), (1, 2)):
            if pc > 0.0:
                out = (1.0 - pc) * out + pc * _dm_apply_2q_unitary(out, ZZ4, u, v, 3)
        return out

    idle_super = np.zeros((64, 64), dtype=np.complex128)
    for col in range(64):
        b = np.zeros(64, dtype=np.complex128)
        b[col] = 1.0
        idle_super[:, col] = _one_idle_round_3q(b.reshape(8, 8)).reshape(64)

    rho_anc = np.array([[1.0 - pres, 0.0], [0.0, pres]], dtype=np.complex128)

    def _one_syndrome_round_5q(rho8: np.ndarray) -> List[np.ndarray]:
        rho32 = np.kron(np.kron(rho8, rho_anc), rho_anc)
        for a in (3, 4):
            rho32 = _dm_apply_1q_unitary(rho32, _rx(dth) @ H2, a, 5)
            rho32 = _dm_apply_depol1(rho32, profile.p_1q, a, 5)
        for a, d in ((3, 0), (4, 1), (3, 1), (4, 2)):
            rho32 = _dm_apply_2q_unitary(rho32, _rzz(dth) @ CX4, a, d, 5)
            rho32 = _dm_apply_depol2(rho32, profile.p_2q, a, d, 5)
        for u, v in ((0, 3), (3, 1), (1, 4), (4, 2)):
            if pc > 0.0:
                rho32 = (1.0 - pc) * rho32 + pc * _dm_apply_2q_unitary(rho32, ZZ4, u, v, 5)
        for d in (0, 1, 2):
            rho32 = _dm_apply_t1_t2(
                rho32, profile.t_round_us, profile.t1_us, profile.t2_us, d, 5
            )
            rho32 = _dm_apply_1q_unitary(rho32, _rz(dth), d, 5)
        for a in (3, 4):
            rho32 = _dm_apply_1q_unitary(rho32, _rx(dth) @ H2, a, 5)
            rho32 = _dm_apply_depol1(rho32, profile.p_1q, a, 5)

        t = rho32.reshape(8, 4, 8, 4)
        true_blocks = [t[:, m, :, m] for m in range(4)]
        obs_blocks = [np.zeros((8, 8), dtype=np.complex128) for _ in range(4)]
        for m_true in range(4):
            t0, t1 = (m_true >> 1) & 1, m_true & 1
            for m_obs in range(4):
                o0, o1 = (m_obs >> 1) & 1, m_obs & 1
                p0 = (1.0 - pro) if o0 == t0 else pro
                p1 = (1.0 - pro) if o1 == t1 else pro
                obs_blocks[m_obs] += (p0 * p1) * true_blocks[m_true]
        return obs_blocks

    m_ops = [np.zeros((64, 64), dtype=np.complex128) for _ in range(4)]
    for col in range(64):
        b = np.zeros(64, dtype=np.complex128)
        b[col] = 1.0
        blks = _one_syndrome_round_5q(b.reshape(8, 8))
        for m_obs in range(4):
            m_ops[m_obs][:, col] = blks[m_obs].reshape(64)

    ff_ratio = profile.t_ff_us / profile.t_round_us if profile.t_round_us > 0.0 else 0.0
    dth_ff = dth * ff_ratio
    pc_ff = pc * ff_ratio
    target_for_syn = {0: None, 1: 2, 2: 0, 3: 1}
    ff_ops: Dict[int, np.ndarray] = {}
    for syn_code, corr_q in target_for_syn.items():
        ff_mat = np.zeros((64, 64), dtype=np.complex128)
        for col in range(64):
            b = np.zeros(64, dtype=np.complex128)
            b[col] = 1.0
            r8 = b.reshape(8, 8)
            for d in (0, 1, 2):
                r8 = _dm_apply_t1_t2(r8, profile.t_ff_us, profile.t1_us, profile.t2_us, d, 3)
                r8 = _dm_apply_1q_unitary(r8, _rz(dth_ff), d, 3)
            for u, v in ((0, 1), (1, 2)):
                if pc_ff > 0.0:
                    r8 = (1.0 - pc_ff) * r8 + pc_ff * _dm_apply_2q_unitary(r8, ZZ4, u, v, 3)
            if corr_q is not None:
                r8 = _dm_apply_1q_unitary(r8, _rx(dth) @ Z2, corr_q, 3)
                r8 = _dm_apply_depol1(r8, profile.p_1q, corr_q, 3)
            ff_mat[:, col] = r8.reshape(64)
        ff_ops[syn_code] = ff_mat

    ro_mat_1q = np.array([[1.0 - pro, pro], [pro, 1.0 - pro]], dtype=np.float64)
    ro_mat_3q = np.kron(np.kron(ro_mat_1q, ro_mat_1q), ro_mat_1q)
    meas_linear = np.zeros((8, 64), dtype=np.complex128)
    for col in range(64):
        b = np.zeros(64, dtype=np.complex128)
        b[col] = 1.0
        r8 = b.reshape(8, 8)
        for d in (0, 1, 2):
            r8 = _dm_apply_1q_unitary(r8, _rx(dth) @ H2, d, 3)
            r8 = _dm_apply_depol1(r8, profile.p_1q, d, 3)
        true_diag = np.diag(r8)
        meas_linear[:, col] = ro_mat_3q @ true_diag

    trace_vec = np.eye(8, dtype=np.complex128).reshape(64)
    return {
        "rho_init_vec": rho_init.reshape(64),
        "idle_super": idle_super,
        "m_ops": m_ops,
        "ff_ops": ff_ops,
        "meas_linear": meas_linear,
        "trace_vec": trace_vec,
    }


def _exact_rep3_arm4a_branch_sum(
    rounds: int, super_bundle: Mapping[str, Any], decoder: AuditableMWPMDecoder
) -> float:
    """Exact zero-variance Kraus density-matrix branch integral for `[[3,1,3]]` `arm4a` (`R <= 4`)."""
    m_ops = super_bundle["m_ops"]
    meas_linear = super_bundle["meas_linear"]
    states = super_bundle["rho_init_vec"].reshape(64, 1)
    for _ in range(rounds):
        next_blocks = [m_ops[m_obs] @ states for m_obs in range(4)]
        states = np.concatenate(next_blocks, axis=1)

    term_probs = np.maximum(0.0, np.real(meas_linear @ states))
    n_hist = 1 << (2 * rounds)
    p_fail = 0.0

    for h in range(n_hist):
        m_seq = [(h >> (2 * (rounds - 1 - r))) & 3 for r in range(rounds)]
        base_dmask = 0
        for r in range(rounds):
            m_curr = m_seq[r]
            m_prev = 0 if r == 0 else m_seq[r - 1]
            diff = m_curr ^ m_prev
            if (diff >> 1) & 1:
                base_dmask |= 1 << (2 * r)
            if diff & 1:
                base_dmask |= 1 << (2 * r + 1)
        m_last0 = (m_seq[rounds - 1] >> 1) & 1
        m_last1 = m_seq[rounds - 1] & 1

        for d_out in range(8):
            d0 = (d_out >> 2) & 1
            d1 = (d_out >> 1) & 1
            d2 = d_out & 1
            b0 = m_last0 ^ d0 ^ d1
            b1 = m_last1 ^ d1 ^ d2
            full_mask = base_dmask | (b0 << (2 * rounds)) | (b1 << (2 * rounds + 1))
            obs_raw = d0 ^ d1 ^ d2
            if decoder.decode_mask(full_mask) != obs_raw:
                p_fail += float(term_probs[d_out, h])
    return p_fail


# ---------------------------------------------------------------------------
# Statistical Fitting & Crossing Helpers
# ---------------------------------------------------------------------------


def logical_error_per_round(p_fail: float, rounds: int) -> float:
    """Compute effective logical error per round `0.5 * (1 - (1 - 2 * P_fail)^(1/R))`."""
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    if p_fail <= 0.0:
        return 0.0
    fid = max(1e-12, 1.0 - 2.0 * min(p_fail, 0.499999))
    return 0.5 * (1.0 - (fid ** (1.0 / float(rounds))))


def _fit_log_fidelity_slope(rounds_arr: np.ndarray, p_fails: np.ndarray) -> float:
    """Fit `ln(1 - 2*P_fail(R)) = b0 + b1*R` via OLS and return `epsilon_L_slope = 0.5*(1 - exp(b1))`."""
    fids = np.clip(1.0 - 2.0 * p_fails, 1e-9, 1.0)
    y = np.log(fids)
    slope, _ = np.polyfit(rounds_arr, y, 1)
    if slope >= 0.0:
        return 0.0
    return float(0.5 * (1.0 - math.exp(float(slope))))


def fit_epsilon_L_vs_rounds(
    rounds_list: Sequence[int],
    p_fails: Sequence[float],
    shot_counts: Sequence[int],
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Compute per-round `epsilon_L(R)`, log-linear decay slope `epsilon_L_fit`, and bootstrap 95% CIs."""
    r_arr = np.asarray(rounds_list, dtype=np.float64)
    p_arr = np.asarray(p_fails, dtype=np.float64)
    n_arr = np.asarray(shot_counts, dtype=np.int64)

    per_r_eps = [
        round(logical_error_per_round(float(p), int(r)), 8) for r, p in zip(rounds_list, p_fails)
    ]
    slope_point = _fit_log_fidelity_slope(r_arr, p_arr)
    terminal_point = logical_error_per_round(float(p_arr[-1]), int(rounds_list[-1]))

    rng = np.random.default_rng(seed)
    boot_slopes = np.empty(n_boot, dtype=np.float64)
    boot_term = np.empty(n_boot, dtype=np.float64)
    p_clipped = np.clip(p_arr, 0.0, 1.0)

    for b in range(n_boot):
        sampled = rng.binomial(n_arr, p_clipped) / np.maximum(n_arr, 1)
        boot_slopes[b] = _fit_log_fidelity_slope(r_arr, sampled)
        boot_term[b] = logical_error_per_round(float(sampled[-1]), int(rounds_list[-1]))

    return {
        "epsilon_L_slope_fit": round(slope_point, 8),
        "epsilon_L_slope_ci95": [
            round(float(np.percentile(boot_slopes, 2.5)), 8),
            round(float(np.percentile(boot_slopes, 97.5)), 8),
        ],
        "epsilon_L_at_max_round": round(terminal_point, 8),
        "epsilon_L_at_max_round_ci95": [
            round(float(np.percentile(boot_term, 2.5)), 8),
            round(float(np.percentile(boot_term, 97.5)), 8),
        ],
        "per_round_epsilon_L": {str(r): eps for r, eps in zip(rounds_list, per_r_eps)},
    }


def _find_crossing_rounds(
    rounds_list: Sequence[int],
    p_arm4: Sequence[float],
    p_arm1: Sequence[float],
    p_arm2: Sequence[float],
) -> Dict[str, Any]:
    """Determine first round `R_cross` where `arm4` drops strictly below `arm1`, `arm2`, and BOTH."""
    r_vs_arm1: Optional[int] = None
    r_vs_arm2: Optional[int] = None
    r_vs_both: Optional[int] = None
    sustained_from_r_both: Optional[int] = None

    below_both_flags = [
        (p4 < p1) and (p4 < p2) for p4, p1, p2 in zip(p_arm4, p_arm1, p_arm2)
    ]

    for idx, (r, p4, p1, p2) in enumerate(zip(rounds_list, p_arm4, p_arm1, p_arm2)):
        if r_vs_arm1 is None and p4 < p1:
            r_vs_arm1 = int(r)
        if r_vs_arm2 is None and p4 < p2:
            r_vs_arm2 = int(r)
        if r_vs_both is None and (p4 < p1) and (p4 < p2):
            r_vs_both = int(r)
        if sustained_from_r_both is None and all(below_both_flags[idx:]):
            sustained_from_r_both = int(r)

    return {
        "r_cross_vs_arm1_unencoded": r_vs_arm1,
        "r_cross_vs_arm2_no_correction": r_vs_arm2,
        "r_cross_vs_both_first": r_vs_both,
        "r_cross_vs_both_sustained": sustained_from_r_both,
        "beats_arm2_at_any_tested_round": r_vs_arm2 is not None,
        "crosses_below_both_sustained": sustained_from_r_both is not None,
    }


def evaluate_rep3_kraus_arms(
    profile: TargetNoiseProfile,
    rounds_list: Sequence[int] = ROUNDS_GRID,
    n_shots: int = N_SHOTS_REP3_DEFAULT,
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Evaluate all Card C arms on `[[3,1,3]]` (`5Q`) using the exact Kraus engine."""
    super_bundle = _build_rep3_kraus_superoperators(profile)
    m_ops = super_bundle["m_ops"]
    ff_ops = super_bundle["ff_ops"]
    meas_linear = super_bundle["meas_linear"]
    trace_vec = super_bundle["trace_vec"]

    rng = np.random.default_rng(seed)

    p_arm1_list: List[float] = []
    p_arm2_list: List[float] = []
    p_arm3_cond_list: List[float] = []
    acc_arm3_list: List[float] = []
    p_arm4a_list: List[float] = []
    p_arm4b_list: List[float] = []
    exact_dm_crosschecks: Dict[str, Dict[str, float]] = {}

    per_round_records: List[Dict[str, Any]] = []

    for r_val in rounds_list:
        dem = build_repetition_d3_dem(rounds=r_val, profile=profile, basis="X")
        decoder = auditable_mwpm_decoder(dem)

        p_arm1 = simulate_5q_arm1_kraus_exact(
            rounds=r_val, profile=profile, basis="X", include_spam_and_coherent=True
        )
        p_arm1_list.append(p_arm1)

        v_arm2 = super_bundle["rho_init_vec"].copy()
        for _ in range(r_val):
            v_arm2 = super_bundle["idle_super"] @ v_arm2
        probs_arm2 = np.maximum(0.0, np.real(meas_linear @ v_arm2))
        p_arm2 = float(probs_arm2[3] + probs_arm2[5] + probs_arm2[6] + probs_arm2[7])
        p_arm2_list.append(p_arm2)

        v_arm3 = super_bundle["rho_init_vec"].copy()
        for _ in range(r_val):
            v_arm3 = m_ops[0] @ v_arm3
        probs_arm3 = np.maximum(0.0, np.real(meas_linear @ v_arm3))
        p_acc_000 = float(probs_arm3[0])
        p_acc_111 = float(probs_arm3[7])
        acc_rate = p_acc_000 + p_acc_111
        p_arm3_cond = (p_acc_111 / acc_rate) if acc_rate > 1e-15 else 0.0
        acc_arm3_list.append(acc_rate)
        p_arm3_cond_list.append(p_arm3_cond)

        # Common random numbers shared by Arm 4a and Arm 4b (preserves exact marginals of each!)
        u_rounds_shared = rng.random((r_val, n_shots))

        states_4a = np.repeat(super_bundle["rho_init_vec"].reshape(64, 1), n_shots, axis=1)
        anc_hist_4a = np.zeros((n_shots, r_val), dtype=np.uint8)

        for r_step in range(r_val):
            w_branches = np.empty((4, n_shots), dtype=np.float64)
            next_vecs: List[np.ndarray] = []
            for m_code in range(4):
                cand = m_ops[m_code] @ states_4a
                next_vecs.append(cand)
                w_branches[m_code] = np.maximum(0.0, np.real(trace_vec @ cand))
            w_sum = np.maximum(np.sum(w_branches, axis=0), 1e-30)
            probs_m = w_branches / w_sum
            u = u_rounds_shared[r_step]
            cdf = np.cumsum(probs_m, axis=0)
            chosen = np.clip(np.sum(u[None, :] >= cdf, axis=0), 0, 3).astype(np.uint8)
            anc_hist_4a[:, r_step] = chosen
            for m_code in range(4):
                sel = chosen == m_code
                if np.any(sel):
                    nrm = np.maximum(w_branches[m_code, sel], 1e-30)
                    states_4a[:, sel] = next_vecs[m_code][:, sel] / nrm[None, :]

        # Exact conditional terminal readout integration (Rao-Blackwellized over all 8 data outcomes)
        term_4a = np.maximum(0.0, np.real(meas_linear @ states_4a))
        term_4a /= np.maximum(np.sum(term_4a, axis=0, keepdims=True), 1e-30)

        base_dmasks_4a = np.zeros(n_shots, dtype=np.int64)
        for r_step in range(r_val):
            mc = anc_hist_4a[:, r_step]
            mp = np.zeros(n_shots, dtype=np.uint8) if r_step == 0 else anc_hist_4a[:, r_step - 1]
            diff = mc ^ mp
            base_dmasks_4a |= (((diff >> 1) & 1).astype(np.int64)) << (2 * r_step)
            base_dmasks_4a |= ((diff & 1).astype(np.int64)) << (2 * r_step + 1)
        mlast_a = anc_hist_4a[:, r_val - 1]
        mlast0_a = (mlast_a >> 1) & 1
        mlast1_a = mlast_a & 1
        fail_prob_4a_per_shot = np.zeros(n_shots, dtype=np.float64)
        for d_out in range(8):
            d0_a = (d_out >> 2) & 1
            d1_a = (d_out >> 1) & 1
            d2_a = d_out & 1
            b0_a = mlast0_a ^ (d0_a ^ d1_a)
            b1_a = mlast1_a ^ (d1_a ^ d2_a)
            full_masks_a = (
                base_dmasks_4a
                | (b0_a.astype(np.int64) << (2 * r_val))
                | (b1_a.astype(np.int64) << (2 * r_val + 1))
            )
            obs_val_a = d0_a ^ d1_a ^ d2_a
            wrong_a = decoder.decode_batch(full_masks_a) != obs_val_a
            fail_prob_4a_per_shot += term_4a[d_out, :] * wrong_a
        p_arm4a = float(np.mean(fail_prob_4a_per_shot))
        p_arm4a_list.append(p_arm4a)

        if r_val <= 3:
            p_exact_4a = _exact_rep3_arm4a_branch_sum(r_val, super_bundle, decoder)
            exact_dm_crosschecks[str(r_val)] = {
                "exact_kraus_branch_p_fail": round(p_exact_4a, 8),
                "trajectory_mc_p_fail": round(p_arm4a, 8),
                "abs_diff": round(abs(p_exact_4a - p_arm4a), 8),
            }

        states_4b = np.repeat(super_bundle["rho_init_vec"].reshape(64, 1), n_shots, axis=1)
        eff_anc_4b = np.zeros((n_shots, r_val), dtype=np.uint8)
        applied_z = np.zeros((n_shots, 3), dtype=np.uint8)

        for r_step in range(r_val):
            w_branches = np.empty((4, n_shots), dtype=np.float64)
            next_vecs_b: List[np.ndarray] = []
            for m_code in range(4):
                cand = m_ops[m_code] @ states_4b
                next_vecs_b.append(cand)
                w_branches[m_code] = np.maximum(0.0, np.real(trace_vec @ cand))
            w_sum = np.maximum(np.sum(w_branches, axis=0), 1e-30)
            probs_m = w_branches / w_sum
            u = u_rounds_shared[r_step]
            chosen_raw = np.clip(
                np.sum(u[None, :] >= np.cumsum(probs_m, axis=0), axis=0), 0, 3
            ).astype(np.uint8)
            s0_applied = applied_z[:, 0] ^ applied_z[:, 1]
            s1_applied = applied_z[:, 1] ^ applied_z[:, 2]
            eff_anc_4b[:, r_step] = chosen_raw ^ (s0_applied << 1) ^ s1_applied

            for m_code in range(4):
                sel = chosen_raw == m_code
                if np.any(sel):
                    nrm = np.maximum(w_branches[m_code, sel], 1e-30)
                    normed = next_vecs_b[m_code][:, sel] / nrm[None, :]
                    states_4b[:, sel] = ff_ops[m_code] @ normed
                    if m_code == 1:
                        applied_z[sel, 2] ^= 1
                    elif m_code == 2:
                        applied_z[sel, 0] ^= 1
                    elif m_code == 3:
                        applied_z[sel, 1] ^= 1

        term_4b = np.maximum(0.0, np.real(meas_linear @ states_4b))
        term_4b /= np.maximum(np.sum(term_4b, axis=0, keepdims=True), 1e-30)

        base_dmasks_4b = np.zeros(n_shots, dtype=np.int64)
        for r_step in range(r_val):
            mc = eff_anc_4b[:, r_step]
            mp = np.zeros(n_shots, dtype=np.uint8) if r_step == 0 else eff_anc_4b[:, r_step - 1]
            diff = mc ^ mp
            base_dmasks_4b |= (((diff >> 1) & 1).astype(np.int64)) << (2 * r_step)
            base_dmasks_4b |= ((diff & 1).astype(np.int64)) << (2 * r_step + 1)
        mlast_b = eff_anc_4b[:, r_val - 1]
        mlast0_b = (mlast_b >> 1) & 1
        mlast1_b = mlast_b & 1
        fail_prob_4b_per_shot = np.zeros(n_shots, dtype=np.float64)
        for d_out_raw in range(8):
            d0_b = ((d_out_raw >> 2) & 1) ^ applied_z[:, 0]
            d1_b = ((d_out_raw >> 1) & 1) ^ applied_z[:, 1]
            d2_b = (d_out_raw & 1) ^ applied_z[:, 2]
            b0_b = mlast0_b ^ d0_b ^ d1_b
            b1_b = mlast1_b ^ d1_b ^ d2_b
            full_masks_b = (
                base_dmasks_4b
                | (b0_b.astype(np.int64) << (2 * r_val))
                | (b1_b.astype(np.int64) << (2 * r_val + 1))
            )
            obs_b = d0_b ^ d1_b ^ d2_b
            wrong_b = decoder.decode_batch(full_masks_b) != obs_b
            fail_prob_4b_per_shot += term_4b[d_out_raw, :] * wrong_b
        p_arm4b = float(np.mean(fail_prob_4b_per_shot))
        p_arm4b_list.append(p_arm4b)

        per_round_records.append(
            {
                "rounds": int(r_val),
                "matched_task_duration_us": round(r_val * profile.t_round_us, 4),
                "arm1_unencoded_p_fail": round(p_arm1, 7),
                "arm2_encoded_no_correction_p_fail": round(p_arm2, 7),
                "arm3_postselected_conditional_p_fail": round(p_arm3_cond, 7),
                "arm3_postselected_acceptance_rate": round(acc_rate, 7),
                "arm4a_decoded_pauli_frame_p_fail": round(p_arm4a, 7),
                "arm4b_decoded_physical_feedforward_p_fail": round(p_arm4b, 7),
            }
        )

    shots_vec = [n_shots] * len(rounds_list)
    acc_shots_vec = [max(10, int(round(a * n_shots))) for a in acc_arm3_list]

    crossing_4a = _find_crossing_rounds(rounds_list, p_arm4a_list, p_arm1_list, p_arm2_list)
    crossing_4b = _find_crossing_rounds(rounds_list, p_arm4b_list, p_arm1_list, p_arm2_list)

    return {
        "code_id": "repetition_d3_5q",
        "code_label": "[[3,1,3]] distance-3 phase-flip repetition code (5 qubits: 3 data + 2 ancillas)",
        "simulation_engine": "exact_5q_kraus_density_matrix_and_quantum_instrument",
        "memory_basis": "X",
        "target_profile_id": profile.profile_id,
        "n_shots_per_round": n_shots,
        "rounds_evaluated": list(rounds_list),
        "unconditional_arms": {
            "arm1_unencoded": {
                "arm_label": "unencoded_bare_qubit",
                "fit": fit_epsilon_L_vs_rounds(
                    rounds_list, p_arm1_list, shots_vec, n_boot, seed + 1
                ),
                "p_fail_by_round": {str(r): round(p, 7) for r, p in zip(rounds_list, p_arm1_list)},
            },
            "arm2_encoded_no_correction": {
                "arm_label": "encoded_idle_terminal_decode_only",
                "fit": fit_epsilon_L_vs_rounds(
                    rounds_list, p_arm2_list, shots_vec, n_boot, seed + 2
                ),
                "p_fail_by_round": {str(r): round(p, 7) for r, p in zip(rounds_list, p_arm2_list)},
            },
            "arm4a_decoded_pauli_frame": {
                "arm_label": "decoded_correction_offline_pauli_frame",
                "decoding_mode": "offline_pauli_frame",
                "feedforward_latency_us": 0.0,
                "conditional_physical_gates": False,
                "fit": fit_epsilon_L_vs_rounds(
                    rounds_list, p_arm4a_list, shots_vec, n_boot, seed + 3
                ),
                "p_fail_by_round": {
                    str(r): round(p, 7) for r, p in zip(rounds_list, p_arm4a_list)
                },
                "crossing_analysis": crossing_4a,
            },
            "arm4b_decoded_physical_feedforward": {
                "arm_label": "decoded_correction_physical_feedforward",
                "decoding_mode": "physical_feedforward",
                "feedforward_latency_us": profile.t_ff_us,
                "conditional_physical_gates": True,
                "fit": fit_epsilon_L_vs_rounds(
                    rounds_list, p_arm4b_list, shots_vec, n_boot, seed + 4
                ),
                "p_fail_by_round": {
                    str(r): round(p, 7) for r, p in zip(rounds_list, p_arm4b_list)
                },
                "crossing_analysis": crossing_4b,
            },
        },
        "postselected_arm_reported_separately": {
            "arm3_postselected_detection": {
                "arm_label": "postselected_zero_detector_events_only",
                "postselection_applied": True,
                "acceptance_rate_by_round": {
                    str(r): round(a, 7) for r, a in zip(rounds_list, acc_arm3_list)
                },
                "conditional_p_fail_by_round": {
                    str(r): round(p, 7) for r, p in zip(rounds_list, p_arm3_cond_list)
                },
                "conditional_fit": fit_epsilon_L_vs_rounds(
                    rounds_list, p_arm3_cond_list, acc_shots_vec, n_boot, seed + 5
                ),
            }
        },
        "exact_kraus_branch_crosschecks_r1_to_r3": exact_dm_crosschecks,
        "per_round_table": per_round_records,
        "cell_stop_go": {
            "falsifier_triggered": not crossing_4a["beats_arm2_at_any_tested_round"],
            "arm4a_crosses_below_both": crossing_4a["crosses_below_both_sustained"],
            "arm4b_crosses_below_both": crossing_4b["crosses_below_both_sustained"],
            "r_cross_arm4a_vs_both": crossing_4a["r_cross_vs_both_sustained"],
            "r_cross_arm4b_vs_both": crossing_4b["r_cross_vs_both_sustained"],
            "verdict": (
                "PASS_CROSSING_OBSERVED"
                if crossing_4a["crosses_below_both_sustained"]
                else "STOP_NO_CROSSING_AT_TESTED_ROUNDS"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Component 2B: 17Q Rotated d=3 Surface Code Circuit-Level Simulator
# ---------------------------------------------------------------------------


def _build_surface_d3_spatial_x_lut() -> Dict[int, int]:
    """Exact minimum-weight spatial Z-error lookup table for 4 X-checks on `[[9,1,3]]`."""
    lut: Dict[int, int] = {}
    supports = list(SURFACE_D3_X_SUPPORTS.values())
    for syn in range(16):
        best_w = 99
        best_obs = 0
        for zmask in range(512):
            w = bin(zmask).count("1")
            if w >= best_w:
                continue
            s_val = 0
            for si, qs in enumerate(supports):
                if sum((zmask >> q) & 1 for q in qs) & 1:
                    s_val |= 1 << si
            if s_val == syn:
                best_w = w
                best_obs = ((zmask >> 0) & 1) ^ ((zmask >> 1) & 1) ^ ((zmask >> 2) & 1)
        lut[syn] = best_obs
    return lut


def _build_surface_d3_single_round_recovery_qubit() -> Dict[int, Optional[int]]:
    """Map single-round 4-bit X-syndrome to single-qubit physical Z recovery target (if weight-1)."""
    supports = list(SURFACE_D3_X_SUPPORTS.values())
    mapping: Dict[int, Optional[int]] = {s: None for s in range(16)}
    for q in range(9):
        s_val = 0
        for si, qs in enumerate(supports):
            if q in qs:
                s_val |= 1 << si
        mapping[s_val] = q
    return mapping


def _simulate_dem_ops_batch(
    dem: SpaceTimeDEM, n_shots: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized circuit-level Monte Carlo execution of `SpaceTimeDEM.ops` across `n_shots`."""
    nq = dem.n_qubits
    nm = dem.n_measurements
    x = np.zeros((n_shots, nq), dtype=np.uint8)
    z = np.zeros((n_shots, nq), dtype=np.uint8)
    m = np.zeros((n_shots, nm), dtype=np.uint8)

    for op in dem.ops:
        kind = op[0]
        if kind == "R":
            q, p = op[1], float(op[2])
            x[:, q] = (rng.random(n_shots) < p).astype(np.uint8) if p > 0.0 else 0
            z[:, q] = 0
        elif kind == "H":
            q, p = op[1], float(op[2])
            tmp = x[:, q].copy()
            x[:, q] = z[:, q]
            z[:, q] = tmp
            if p > 0.0:
                u = rng.random(n_shots)
                p3 = p / 3.0
                e_x = u < p3
                e_y = (u >= p3) & (u < 2.0 * p3)
                e_z = (u >= 2.0 * p3) & (u < p)
                x[:, q] ^= (e_x | e_y).astype(np.uint8)
                z[:, q] ^= (e_y | e_z).astype(np.uint8)
        elif kind == "PAULI1":
            q, px, py, pz = op[1], float(op[2]), float(op[3]), float(op[4])
            u = rng.random(n_shots)
            e_x = u < px
            e_y = (u >= px) & (u < px + py)
            e_z = (u >= px + py) & (u < px + py + pz)
            x[:, q] ^= (e_x | e_y).astype(np.uint8)
            z[:, q] ^= (e_y | e_z).astype(np.uint8)
        elif kind == "ZZ_CORR":
            q1, q2, pc = op[1], op[2], float(op[3])
            if pc > 0.0:
                fl = (rng.random(n_shots) < pc).astype(np.uint8)
                z[:, q1] ^= fl
                z[:, q2] ^= fl
        elif kind == "CX":
            c, t, p = op[1], op[2], float(op[3])
            x[:, t] ^= x[:, c]
            z[:, c] ^= z[:, t]
            if p > 0.0:
                hit = np.where(rng.random(n_shots) < p)[0]
                if hit.size > 0:
                    kinds = rng.integers(0, 15, size=hit.size)
                    for code_i, (cx_b, cz_b, tx_b, tz_b) in enumerate(PAULI_2Q_BITS):
                        sel = hit[kinds == code_i]
                        if sel.size > 0:
                            if cx_b:
                                x[sel, c] ^= 1
                            if cz_b:
                                z[sel, c] ^= 1
                            if tx_b:
                                x[sel, t] ^= 1
                            if tz_b:
                                z[sel, t] ^= 1
        elif kind == "MZ":
            q, mid, pro = op[1], op[2], float(op[3])
            fl = (rng.random(n_shots) < pro).astype(np.uint8) if pro > 0.0 else 0
            m[:, mid] = x[:, q] ^ fl

    dmasks = np.zeros(n_shots, dtype=np.int64)
    for di, ms in enumerate(dem.detectors):
        par = np.zeros(n_shots, dtype=np.uint8)
        for u_idx in ms:
            par ^= m[:, u_idx]
        dmasks |= par.astype(np.int64) << di

    obs_vals = np.zeros(n_shots, dtype=np.uint8)
    for u_idx in dem.observable:
        obs_vals ^= m[:, u_idx]
    return dmasks, obs_vals


def _simulate_surface_d3_arm4b_feedforward(
    dem: SpaceTimeDEM,
    rounds: int,
    profile: TargetNoiseProfile,
    decoder: AuditableMWPMDecoder,
    n_shots: int,
    rng_base: np.random.Generator,
    rng_ff: np.random.Generator,
) -> float:
    """Simulate `[[9,1,3]]` `arm4b_decoded_physical_feedforward` coupled to `rng_base` via Markov semigroup splitting."""
    ff_ratio = profile.t_ff_us / profile.t_round_us if profile.t_round_us > 0.0 else 0.0
    px_ff, py_ff, pz_ff = gta_t1_t2_pauli_probs(profile.t_ff_us, profile.t1_us, profile.t2_us)
    p_coh_ff = math.sin(0.5 * profile.delta_theta_rad * ff_ratio) ** 2
    p_coh_gate = math.sin(0.5 * profile.delta_theta_rad) ** 2
    pz_ff_eff = pz_ff + p_coh_ff * (1.0 - 2.0 * pz_ff)
    p1_eff = min(0.49, profile.p_1q + p_coh_gate)
    pc_ff = min(0.49, profile.p_corr * ff_ratio)

    phys_edges = [
        (q, a)
        for a, qs in list(SURFACE_D3_Z_SCHEDULE.items()) + list(SURFACE_D3_X_SCHEDULE.items())
        for q in qs
        if q is not None
    ]
    rec_map = _build_surface_d3_single_round_recovery_qubit()
    x_supps = [SURFACE_D3_X_SUPPORTS[a] for a in range(13, 17)]

    nq = dem.n_qubits
    nm = dem.n_measurements
    x = np.zeros((n_shots, nq), dtype=np.uint8)
    z = np.zeros((n_shots, nq), dtype=np.uint8)
    m = np.zeros((n_shots, nm), dtype=np.uint8)
    applied_z = np.zeros((n_shots, 9), dtype=np.uint8)
    eff_anc = np.zeros((n_shots, rounds, 4), dtype=np.uint8)

    for op in dem.ops:
        kind = op[0]
        if kind == "R":
            q, p = op[1], float(op[2])
            x[:, q] = (rng_base.random(n_shots) < p).astype(np.uint8) if p > 0.0 else 0
            z[:, q] = 0
        elif kind == "H":
            q, p = op[1], float(op[2])
            tmp = x[:, q].copy()
            x[:, q] = z[:, q]
            z[:, q] = tmp
            if p > 0.0:
                u = rng_base.random(n_shots)
                p3 = p / 3.0
                e_x = u < p3
                e_y = (u >= p3) & (u < 2.0 * p3)
                e_z = (u >= 2.0 * p3) & (u < p)
                x[:, q] ^= (e_x | e_y).astype(np.uint8)
                z[:, q] ^= (e_y | e_z).astype(np.uint8)
        elif kind == "PAULI1":
            q, px, py, pz = op[1], float(op[2]), float(op[3]), float(op[4])
            u = rng_base.random(n_shots)
            e_x = u < px
            e_y = (u >= px) & (u < px + py)
            e_z = (u >= px + py) & (u < px + py + pz)
            x[:, q] ^= (e_x | e_y).astype(np.uint8)
            z[:, q] ^= (e_y | e_z).astype(np.uint8)
        elif kind == "ZZ_CORR":
            q1, q2, pc = op[1], op[2], float(op[3])
            if pc > 0.0:
                fl = (rng_base.random(n_shots) < pc).astype(np.uint8)
                z[:, q1] ^= fl
                z[:, q2] ^= fl
        elif kind == "CX":
            c, t, p = op[1], op[2], float(op[3])
            x[:, t] ^= x[:, c]
            z[:, c] ^= z[:, t]
            if p > 0.0:
                hit = np.where(rng_base.random(n_shots) < p)[0]
                if hit.size > 0:
                    kinds = rng_base.integers(0, 15, size=hit.size)
                    for code_i, (cx_b, cz_b, tx_b, tz_b) in enumerate(PAULI_2Q_BITS):
                        sel = hit[kinds == code_i]
                        if sel.size > 0:
                            if cx_b:
                                x[sel, c] ^= 1
                            if cz_b:
                                z[sel, c] ^= 1
                            if tx_b:
                                x[sel, t] ^= 1
                            if tz_b:
                                z[sel, t] ^= 1
        elif kind == "MZ":
            q, mid, pro = op[1], op[2], float(op[3])
            fl = (rng_base.random(n_shots) < pro).astype(np.uint8) if pro > 0.0 else 0
            m[:, mid] = x[:, q] ^ fl

            # At the end of round `r_step` (immediately after ancilla 16 is measured):
            if q == 16 and mid < 8 * rounds:
                r_step = mid // 8
                raw_syn = np.zeros(n_shots, dtype=np.uint8)
                for si in range(4):
                    m_bit = m[:, 8 * r_step + 4 + si]
                    raw_syn |= m_bit << si
                    app_par = np.zeros(n_shots, dtype=np.uint8)
                    for dq in x_supps[si]:
                        app_par ^= applied_z[:, dq]
                    eff_anc[:, r_step, si] = m_bit ^ app_par

                # Incremental feed-forward classical controller latency t_ff_us
                if profile.t_ff_us > 0.0:
                    for u_q, v_q in phys_edges:
                        if pc_ff > 0.0:
                            fl_ff = (rng_ff.random(n_shots) < pc_ff).astype(np.uint8)
                            z[:, u_q] ^= fl_ff
                            z[:, v_q] ^= fl_ff
                    for d in range(9):
                        u_ff = rng_ff.random(n_shots)
                        e_x = u_ff < px_ff
                        e_y = (u_ff >= px_ff) & (u_ff < px_ff + py_ff)
                        e_z = (u_ff >= px_ff + py_ff) & (u_ff < px_ff + py_ff + pz_ff_eff)
                        x[:, d] ^= (e_x | e_y).astype(np.uint8)
                        z[:, d] ^= (e_y | e_z).astype(np.uint8)

                # Conditional physical Z pulse on flagged data qubit
                for syn_code, target_q in rec_map.items():
                    if target_q is None:
                        continue
                    sel = np.where(raw_syn == syn_code)[0]
                    if sel.size > 0:
                        z[sel, target_q] ^= 1
                        applied_z[sel, target_q] ^= 1
                        if p1_eff > 0.0:
                            u_g = rng_ff.random(sel.size)
                            p3 = p1_eff / 3.0
                            x[sel, target_q] ^= (u_g < 2.0 * p3).astype(np.uint8)
                            z[sel, target_q] ^= ((u_g >= p3) & (u_g < p1_eff)).astype(np.uint8)

    data_m = np.zeros((n_shots, 9), dtype=np.uint8)
    for d in range(9):
        data_m[:, d] = m[:, 8 * rounds + d] ^ applied_z[:, d]

    dmasks = np.zeros(n_shots, dtype=np.int64)
    for r_step in range(rounds):
        for si in range(4):
            bit_idx = 4 * r_step + si
            if r_step == 0:
                par = eff_anc[:, 0, si]
            else:
                par = eff_anc[:, r_step, si] ^ eff_anc[:, r_step - 1, si]
            dmasks |= par.astype(np.int64) << bit_idx
    for si in range(4):
        bit_idx = 4 * rounds + si
        par = eff_anc[:, rounds - 1, si].copy()
        for dq in x_supps[si]:
            par ^= data_m[:, dq]
        dmasks |= par.astype(np.int64) << bit_idx

    obs_raw = data_m[:, 0] ^ data_m[:, 1] ^ data_m[:, 2]
    preds = decoder.decode_batch(dmasks)
    return float(np.mean(preds != obs_raw))


def evaluate_surface_d3_arms(
    profile: TargetNoiseProfile,
    rounds_list: Sequence[int] = ROUNDS_GRID,
    n_shots: int = N_SHOTS_SURF9_DEFAULT,
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Evaluate all Card C arms on rotated `d=3` `[[9,1,3]]` surface code (`17Q`)."""
    rng = np.random.default_rng(seed + 100)
    spatial_lut = _build_surface_d3_spatial_x_lut()
    bridge_cz = profile.heavy_hex_bridge_cz_overhead_per_round
    topo_2q_scale = (24.0 + float(bridge_cz)) / 24.0
    p_coh_gate = math.sin(0.5 * profile.delta_theta_rad) ** 2
    p1_eff = min(0.49, profile.p_1q + p_coh_gate)
    p2_enc = min(0.49, topo_2q_scale * (profile.p_2q + p_coh_gate))

    p_arm1_list: List[float] = []
    p_arm2_list: List[float] = []
    p_arm3_cond_list: List[float] = []
    acc_arm3_list: List[float] = []
    p_arm4a_list: List[float] = []
    p_arm4b_list: List[float] = []
    per_round_records: List[Dict[str, Any]] = []

    for r_val in rounds_list:
        p_arm1 = analytical_arm1_unencoded_failure(
            rounds=r_val, profile=profile, basis="X", include_spam_and_coherent=True
        )
        p_arm1_list.append(p_arm1)

        z_a2 = (rng.random((n_shots, 9)) < profile.p_reset).astype(np.uint8)
        for q in range(9):
            z_a2[:, q] ^= (rng.random(n_shots) < (2.0 * p1_eff / 3.0)).astype(np.uint8)
        az_z = np.zeros((n_shots, 4), dtype=np.uint8)
        for step in range(4):
            for ai, (_, qs) in enumerate(SURFACE_D3_Z_SCHEDULE.items()):
                dq = qs[step]
                if dq is not None:
                    z_a2[:, dq] ^= az_z[:, ai]
                    hit = np.where(rng.random(n_shots) < p2_enc)[0]
                    if hit.size > 0:
                        ks = rng.integers(0, 15, size=hit.size)
                        for ci, (_, czb, _, tzb) in enumerate(PAULI_2Q_BITS):
                            sel = hit[ks == ci]
                            if sel.size > 0:
                                if czb:
                                    z_a2[sel, dq] ^= 1
                                if tzb:
                                    az_z[sel, ai] ^= 1

        dt_tot = r_val * profile.t_round_us
        _, py_tot, pz_tot = gta_t1_t2_pauli_probs(dt_tot, profile.t1_us, profile.t2_us)
        p_coh_idle = math.sin(0.5 * r_val * profile.delta_theta_rad) ** 2
        for q in range(9):
            z_a2[:, q] ^= (rng.random(n_shots) < (py_tot + pz_tot)).astype(np.uint8)
            z_a2[:, q] ^= (rng.random(n_shots) < p_coh_idle).astype(np.uint8)

        p_edge_corr = 0.5 * (1.0 - ((1.0 - 2.0 * profile.p_corr) ** r_val))
        for u_q, v_q in SURFACE_D3_DATA_GRID_EDGES:
            fl = (rng.random(n_shots) < p_edge_corr).astype(np.uint8)
            z_a2[:, u_q] ^= fl
            z_a2[:, v_q] ^= fl

        for q in range(9):
            z_a2[:, q] ^= (rng.random(n_shots) < (2.0 * p1_eff / 3.0)).astype(np.uint8)
            z_a2[:, q] ^= (rng.random(n_shots) < profile.p_readout).astype(np.uint8)

        syn_a2 = np.zeros(n_shots, dtype=np.int64)
        for si, qs in enumerate(SURFACE_D3_X_SUPPORTS.values()):
            par = np.zeros(n_shots, dtype=np.uint8)
            for q in qs:
                par ^= z_a2[:, q]
            syn_a2 |= par.astype(np.int64) << si
        obs_a2 = z_a2[:, 0] ^ z_a2[:, 1] ^ z_a2[:, 2]
        pred_a2 = np.array([spatial_lut[int(s)] for s in syn_a2], dtype=np.uint8)
        p_arm2 = float(np.mean(pred_a2 != obs_a2))
        p_arm2_list.append(p_arm2)

        dem = build_rotated_surface_d3_dem(
            rounds=r_val, profile=profile, basis="X", include_topology_overhead=True
        )
        decoder = auditable_mwpm_decoder(dem)
        coupled_base_seed = int(rng.integers(0, 2**31 - 1))
        dmasks_4a, obs_4a = _simulate_dem_ops_batch(
            dem, n_shots=n_shots, rng=np.random.default_rng(coupled_base_seed)
        )
        preds_4a = decoder.decode_batch(dmasks_4a)
        p_arm4a = float(np.mean(preds_4a != obs_4a))
        p_arm4a_list.append(p_arm4a)

        zero_det = dmasks_4a == 0
        acc_rate = float(np.mean(zero_det))
        p_arm3_cond = float(np.mean(obs_4a[zero_det])) if np.any(zero_det) else 0.0
        acc_arm3_list.append(acc_rate)
        p_arm3_cond_list.append(p_arm3_cond)

        p_arm4b = _simulate_surface_d3_arm4b_feedforward(
            dem=dem,
            rounds=r_val,
            profile=profile,
            decoder=decoder,
            n_shots=n_shots,
            rng_base=np.random.default_rng(coupled_base_seed),
            rng_ff=rng,
        )
        p_arm4b_list.append(p_arm4b)

        per_round_records.append(
            {
                "rounds": int(r_val),
                "matched_task_duration_us": round(r_val * profile.t_round_us, 4),
                "arm1_unencoded_p_fail": round(p_arm1, 7),
                "arm2_encoded_no_correction_p_fail": round(p_arm2, 7),
                "arm3_postselected_conditional_p_fail": round(p_arm3_cond, 7),
                "arm3_postselected_acceptance_rate": round(acc_rate, 7),
                "arm4a_decoded_pauli_frame_p_fail": round(p_arm4a, 7),
                "arm4b_decoded_physical_feedforward_p_fail": round(p_arm4b, 7),
            }
        )

    shots_vec = [n_shots] * len(rounds_list)
    acc_shots_vec = [max(10, int(round(a * n_shots))) for a in acc_arm3_list]
    crossing_4a = _find_crossing_rounds(rounds_list, p_arm4a_list, p_arm1_list, p_arm2_list)
    crossing_4b = _find_crossing_rounds(rounds_list, p_arm4b_list, p_arm1_list, p_arm2_list)

    return {
        "code_id": "rotated_surface_d3_17q",
        "code_label": "[[9,1,3]] rotated d=3 surface code (17 qubits: 9 data + 8 ancillas)",
        "simulation_engine": "vectorized_17q_circuit_level_pauli_and_coherent_channel",
        "memory_basis": "X",
        "target_profile_id": profile.profile_id,
        "native_topology_supported": profile.supports_native_d3_surface,
        "heavy_hex_bridge_cz_overhead_per_round": bridge_cz,
        "n_shots_per_round": n_shots,
        "rounds_evaluated": list(rounds_list),
        "unconditional_arms": {
            "arm1_unencoded": {
                "arm_label": "unencoded_bare_qubit",
                "fit": fit_epsilon_L_vs_rounds(
                    rounds_list, p_arm1_list, shots_vec, n_boot, seed + 11
                ),
                "p_fail_by_round": {str(r): round(p, 7) for r, p in zip(rounds_list, p_arm1_list)},
            },
            "arm2_encoded_no_correction": {
                "arm_label": "encoded_idle_terminal_decode_only",
                "fit": fit_epsilon_L_vs_rounds(
                    rounds_list, p_arm2_list, shots_vec, n_boot, seed + 12
                ),
                "p_fail_by_round": {str(r): round(p, 7) for r, p in zip(rounds_list, p_arm2_list)},
            },
            "arm4a_decoded_pauli_frame": {
                "arm_label": "decoded_correction_offline_pauli_frame",
                "decoding_mode": "offline_pauli_frame",
                "feedforward_latency_us": 0.0,
                "conditional_physical_gates": False,
                "fit": fit_epsilon_L_vs_rounds(
                    rounds_list, p_arm4a_list, shots_vec, n_boot, seed + 13
                ),
                "p_fail_by_round": {
                    str(r): round(p, 7) for r, p in zip(rounds_list, p_arm4a_list)
                },
                "crossing_analysis": crossing_4a,
            },
            "arm4b_decoded_physical_feedforward": {
                "arm_label": "decoded_correction_physical_feedforward",
                "decoding_mode": "physical_feedforward",
                "feedforward_latency_us": profile.t_ff_us,
                "conditional_physical_gates": True,
                "fit": fit_epsilon_L_vs_rounds(
                    rounds_list, p_arm4b_list, shots_vec, n_boot, seed + 14
                ),
                "p_fail_by_round": {
                    str(r): round(p, 7) for r, p in zip(rounds_list, p_arm4b_list)
                },
                "crossing_analysis": crossing_4b,
            },
        },
        "postselected_arm_reported_separately": {
            "arm3_postselected_detection": {
                "arm_label": "postselected_zero_detector_events_only",
                "postselection_applied": True,
                "acceptance_rate_by_round": {
                    str(r): round(a, 7) for r, a in zip(rounds_list, acc_arm3_list)
                },
                "conditional_p_fail_by_round": {
                    str(r): round(p, 7) for r, p in zip(rounds_list, p_arm3_cond_list)
                },
                "conditional_fit": fit_epsilon_L_vs_rounds(
                    rounds_list, p_arm3_cond_list, acc_shots_vec, n_boot, seed + 15
                ),
            }
        },
        "per_round_table": per_round_records,
        "cell_stop_go": {
            "falsifier_triggered": not crossing_4a["beats_arm2_at_any_tested_round"],
            "arm4a_crosses_below_both": crossing_4a["crosses_below_both_sustained"],
            "arm4b_crosses_below_both": crossing_4b["crosses_below_both_sustained"],
            "r_cross_arm4a_vs_both": crossing_4a["r_cross_vs_both_sustained"],
            "r_cross_arm4b_vs_both": crossing_4b["r_cross_vs_both_sustained"],
            "verdict": (
                "PASS_CROSSING_OBSERVED"
                if crossing_4a["crosses_below_both_sustained"]
                else "STOP_NO_CROSSING_AT_TESTED_ROUNDS"
            ),
        },
    }


# ---------------------------------------------------------------------------
# End-to-End Card C Gate Runner
# ---------------------------------------------------------------------------


def run_card_c_qec_crossing_gate(
    output_path: Optional[Path] = None,
    seed: int = DEFAULT_SEED,
    rounds_list: Sequence[int] = ROUNDS_GRID,
    n_shots_rep3: int = N_SHOTS_REP3_DEFAULT,
    n_shots_surf9: int = N_SHOTS_SURF9_DEFAULT,
    n_boot: int = N_BOOT_DEFAULT,
) -> Dict[str, Any]:
    """Execute full offline Card C (`OSIRIS-QEC`) stop/go crossing evaluation."""
    profiles = build_published_target_profiles()

    decoder_audits = {
        "rep3_z_memory": audit_single_fault_recovery(
            "repetition_d3_5q", "Z", rounds=3, profile=profiles["FEZ_PUBLISHED"]
        ),
        "rep3_x_memory": audit_single_fault_recovery(
            "repetition_d3_5q", "X", rounds=3, profile=profiles["FEZ_PUBLISHED"]
        ),
        "surface_d3_z_memory": audit_single_fault_recovery(
            "rotated_surface_d3_17q", "Z", rounds=3, profile=profiles["H2_1_PUBLISHED"]
        ),
        "surface_d3_x_memory": audit_single_fault_recovery(
            "rotated_surface_d3_17q", "X", rounds=3, profile=profiles["H2_1_PUBLISHED"]
        ),
    }

    arm1_kraus_checks: List[Dict[str, Any]] = []
    for basis in ("X", "Z"):
        for r_val in (1, 5, 9):
            p_ana_pure = analytical_arm1_unencoded_failure(
                r_val, profiles["FEZ_PUBLISHED"], basis=basis, include_spam_and_coherent=False
            )
            p_sim_pure = simulate_5q_arm1_kraus_exact(
                r_val, profiles["FEZ_PUBLISHED"], basis=basis, include_spam_and_coherent=False
            )
            p_ana_full = analytical_arm1_unencoded_failure(
                r_val, profiles["FEZ_PUBLISHED"], basis=basis, include_spam_and_coherent=True
            )
            p_sim_full = simulate_5q_arm1_kraus_exact(
                r_val, profiles["FEZ_PUBLISHED"], basis=basis, include_spam_and_coherent=True
            )
            arm1_kraus_checks.append(
                {
                    "basis": basis,
                    "rounds": r_val,
                    "pure_t1_t2_abs_diff": float(abs(p_ana_pure - p_sim_pure)),
                    "full_spam_coherent_abs_diff": float(abs(p_ana_full - p_sim_full)),
                }
            )

    cells: Dict[str, Dict[str, Any]] = {}
    summary_matrix: List[Dict[str, Any]] = []
    idx_offset = 0
    for pid, prof in profiles.items():
        rep_res = evaluate_rep3_kraus_arms(
            prof,
            rounds_list=rounds_list,
            n_shots=n_shots_rep3,
            n_boot=n_boot,
            seed=seed + 1000 * idx_offset,
        )
        surf_res = evaluate_surface_d3_arms(
            prof,
            rounds_list=rounds_list,
            n_shots=n_shots_surf9,
            n_boot=n_boot,
            seed=seed + 1000 * idx_offset + 500,
        )
        idx_offset += 1
        cells[f"repetition_d3_5q__{pid}"] = rep_res
        cells[f"rotated_surface_d3_17q__{pid}"] = surf_res

        for res in (rep_res, surf_res):
            u_arms = res["unconditional_arms"]
            summary_matrix.append(
                {
                    "code_id": res["code_id"],
                    "target_profile_id": pid,
                    "arm1_eps_L_slope": u_arms["arm1_unencoded"]["fit"]["epsilon_L_slope_fit"],
                    "arm2_eps_L_slope": u_arms["arm2_encoded_no_correction"]["fit"][
                        "epsilon_L_slope_fit"
                    ],
                    "arm4a_pauli_frame_eps_L_slope": u_arms["arm4a_decoded_pauli_frame"]["fit"][
                        "epsilon_L_slope_fit"
                    ],
                    "arm4b_feedforward_eps_L_slope": u_arms[
                        "arm4b_decoded_physical_feedforward"
                    ]["fit"]["epsilon_L_slope_fit"],
                    "r_cross_arm4a_vs_both": res["cell_stop_go"]["r_cross_arm4a_vs_both"],
                    "r_cross_arm4b_vs_both": res["cell_stop_go"]["r_cross_arm4b_vs_both"],
                    "verdict": res["cell_stop_go"]["verdict"],
                }
            )

    all_audits_passed = all(a["zero_single_fault_failures"] for a in decoder_audits.values())
    max_kraus_diff = max(
        max(c["pure_t1_t2_abs_diff"], c["full_spam_coherent_abs_diff"])
        for c in arm1_kraus_checks
    )

    report: Dict[str, Any] = {
        "schema": "osiris.qec_card_c.crossing_report/1",
        "experiment_card": "Card C: OSIRIS-QEC (phase1/verification_queue.md)",
        "execution_tier": "SIMULATED",
        "provider_job_id": None,
        "qpu_seconds": 0,
        "hardware_execution_authorized": False,
        "seed": seed,
        "noise_model_documentation": {
            "modeled_mechanisms": list(MODELED_MECHANISMS),
            "omitted_mechanisms": list(OMITTED_MECHANISMS),
            "omitted_mechanisms_impact_note": OMITTED_MECHANISMS_IMPACT_NOTE,
            "target_profiles": {k: asdict(v) for k, v in profiles.items()},
        },
        "verification_controls": {
            "single_fault_decoder_audits": decoder_audits,
            "all_single_fault_audits_passed": all_audits_passed,
            "arm1_5q_kraus_vs_analytical_checks": arm1_kraus_checks,
            "arm1_5q_kraus_max_abs_diff": max_kraus_diff,
        },
        "summary_crossing_matrix": summary_matrix,
        "cells": cells,
        "stop_go_decision": {
            "decoder_audits_verified": all_audits_passed,
            "kraus_analytical_baseline_verified": bool(max_kraus_diff < 1e-12),
            "fez_published_rep3_verdict": cells["repetition_d3_5q__FEZ_PUBLISHED"][
                "cell_stop_go"
            ]["verdict"],
            "fez_drifted_spam_rep3_verdict": cells["repetition_d3_5q__FEZ_DRIFTED_SPAM"][
                "cell_stop_go"
            ]["verdict"],
            "fez_published_surface_d3_verdict": cells["rotated_surface_d3_17q__FEZ_PUBLISHED"][
                "cell_stop_go"
            ]["verdict"],
            "fez_drifted_spam_surface_d3_verdict": cells[
                "rotated_surface_d3_17q__FEZ_DRIFTED_SPAM"
            ]["cell_stop_go"]["verdict"],
            "h2_1_published_rep3_verdict": cells["repetition_d3_5q__H2_1_PUBLISHED"][
                "cell_stop_go"
            ]["verdict"],
            "h2_1_published_surface_d3_verdict": cells[
                "rotated_surface_d3_17q__H2_1_PUBLISHED"
            ]["cell_stop_go"]["verdict"],
            "scope_limit": SCOPE_LIMIT_STATEMENT,
            "governance_note": (
                "Offline SIMULATED stop/go crossing gate only (0 QPU seconds, provider_job_id=None). "
                "Postselected outcomes (arm3) and acceptance rates A(R) are isolated from "
                "unconditional arms (arm1, arm2, arm4a, arm4b), and offline Pauli-frame tracking "
                "(arm4a) is labeled separately from physical feed-forward (arm4b). "
                + SCOPE_LIMIT_STATEMENT
            ),
        },
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
        output_path.write_bytes(payload.encode("utf-8"))

    return report


def main() -> None:
    out_path = Path(__file__).resolve().parent / "artifacts" / "card_c_qec_crossing_report.json"
    rep = run_card_c_qec_crossing_gate(output_path=out_path)
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"Wrote {out_path} (sha256={sha[:16]}...)")
    for row in rep["summary_crossing_matrix"]:
        print(
            f"  {row['code_id']:24s} | {row['target_profile_id']:17s} | "
            f"eps4a={row['arm4a_pauli_frame_eps_L_slope']:.6f} | "
            f"eps4b={row['arm4b_feedforward_eps_L_slope']:.6f} | "
            f"R_cross(4a)={str(row['r_cross_arm4a_vs_both']):4s} | "
            f"R_cross(4b)={str(row['r_cross_arm4b_vs_both']):4s} | {row['verdict']}"
        )


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
