#!/usr/bin/env python3
"""Card C (`OSIRIS-QEC`): Exact 5-Qutrit (`3^5 = 243`-State) `|2>` Leakage & `LRU` Gate.

Closes the non-computational `|2>` leakage evidence gap documented in
`experiments/qec_card_c/qec_crossing_gate.py` by modeling each physical transmon
(and hyperfine ion reference) in the 3-level qutrit space `{|0>, |1>, |2>}` (`3^5 = 243`
pure states for `[[3,1,3]]`: 3 data qutrits + 2 syndrome ancillas):

  1. Two-qubit `CZ` diabatic `|11> <-> |02> / |20>` leakage heating (`p_leak_cz` swept across
     `[0.0, 2e-4, 5e-4, 1.2e-3]` on Heron r2 `ibm_fez`) alongside spontaneous seepage
     `|2> -> |1>` (`p_seep = T_round / (0.5 * T1)`).
  2. Time-correlated multi-round hook / spectator corruption: when a data or ancilla qutrit
     occupies `|2>`, `CZ` gates involving that qutrit fail to apply the computational
     `|11> |-> -|11>` phase and inject persistent 50% `Z` phase / syndrome randomization
     across consecutive rounds until `|2>` decays or is reset, while dispersive binary
     readout classifies `|2>` predominantly as `'1'` (`p_ro_2_as_1 = 0.92`).
  3. Three leakage-mitigation architectures across `R in {1, 2, 3, 5, 7, 9}`:
       - `NO_LRU`: computational reset (`|1> -> |0>`) leaving `|2>` unreset on ancillas & data.
       - `ANCILLA_LRU_ONLY`: unconditional multi-level ancilla reset (`|1>,|2> -> |0>`,
         `lru_infidelity = 5e-4`) each round, eliminating ancilla `|2>` streaks while data
         leakage accumulates.
       - `FULL_DATA_AND_ANCILLA_LRU` (`DQLR / SWAP-LRU`): ancilla multi-level reset plus
         every-round data-qutrit `|2> -> |1>` seepage pulse (`p_lru_1q = 3e-4`, `120 ns`
         duration), restoring per-round suppression and preserving the `Card C` `[[3,1,3]]`
         crossing even at `p_leak_cz = 1.2e-3`.
  4. Cross-architecture comparison against Quantinuum `H2-1` spontaneous Raman / Zeeman
     qubit-subspace leakage (`p_leak_zz = 5e-5`).

Governance & Epistemic Scope:
  - `execution_tier = "SIMULATED"`, `provider_job_id = None`, `qpu_seconds = 0`.
  - Scope limit: Single code distance (`d = 3`, `[[3,1,3]]`) characterizes `|2>` leakage
    floors and `LRU` mitigation requirements at `d = 3` only; multi-distance surface-code
    scaling (`d = 3 -> 5 -> 7`) remains separate.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parent
REPO_ROOT = HERE.parents[1]
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from qec_card_c import qec_crossing_gate as gate  # noqa: E402

CROSSING_REPORT_PATH = HERE / "artifacts" / "card_c_qec_crossing_report.json"
DEFAULT_OUTPUT_PATH = HERE / "artifacts" / "card_c_qutrit_leakage_lru_report.json"

DEFAULT_SEED = gate.DEFAULT_SEED
ROUNDS_GRID: Tuple[int, ...] = gate.ROUNDS_GRID
N_SHOTS_DEFAULT = gate.N_SHOTS_REP3_DEFAULT
N_BOOT_DEFAULT = gate.N_BOOT_DEFAULT

FEZ_LEAK_SWEEP: Tuple[float, ...] = (0.0, 2.0e-4, 5.0e-4, 1.2e-3)
FEZ_DRIFT_LEAK_SWEEP: Tuple[float, ...] = (0.0, 5.0e-4, 1.2e-3)
H2_LEAK_SWEEP: Tuple[float, ...] = (0.0, 5.0e-5)

LRU_ARCHITECTURES: Tuple[str, ...] = (
    "NO_LRU",
    "ANCILLA_LRU_ONLY",
    "FULL_DATA_AND_ANCILLA_LRU",
)

P_RO_2_AS_1_DEFAULT = 0.92
TRANSMON_LRU_INFIDELITY = 5.0e-4
TRANSMON_DQLR_1Q_ERROR = 3.0e-4
TRANSMON_DQLR_DURATION_US = 0.120  # 120 ns DQLR / SWAP-LRU pulse overhead

ION_LRU_INFIDELITY = 5.0e-5
ION_DQLR_1Q_ERROR = 1.5e-5
ION_DQLR_DURATION_US = 20.0  # 20 us optical repump / hyperfine leakage-reduction pulse

SCOPE_LIMIT_STATEMENT = (
    "Simulated 5-qutrit (3^5 = 243-state) evaluation of [[3,1,3]] (d=3) only (0 QPU seconds, "
    "provider_job_id=None). Quantifies non-computational |2> leakage floors and Leakage "
    "Reduction Unit (LRU) requirements at d=3; multi-distance surface-code fault-tolerance "
    "scaling (d = 3 -> 5 -> 7) and live hardware LRU calibration require separate experiments."
)


@dataclass(frozen=True)
class QutritLRUParams:
    """Physical qutrit leakage, seepage, readout, and LRU parameters."""

    p_leak_2q: float
    p_seep_round: float
    p_seep_reset: float
    p_ro_2_as_1: float
    lru_infidelity: float
    p_lru_1q: float
    dt_lru_us: float


def embed_2q_to_3q_1q(u2: np.ndarray) -> np.ndarray:
    """Embed a `2 x 2` computational unitary on `{|0>, |1>}` into `3 x 3` with `+1` on `|2>`."""
    u3 = np.eye(3, dtype=np.complex128)
    u3[:2, :2] = u2
    return u3


I3 = np.eye(3, dtype=np.complex128)
X3 = embed_2q_to_3q_1q(gate.X2)
Y3 = embed_2q_to_3q_1q(gate.Y2)
Z3 = embed_2q_to_3q_1q(gate.Z2)
H3 = embed_2q_to_3q_1q(gate.H2)

# Orthonormal Hermitian basis for single-qutrit block-coherent algebra B(C^2) (+) R|2><2|:
#   0: |0><0|
#   1: |1><1|
#   2: (|0><1| + |1><0|) / sqrt(2)
#   3: i(|1><0| - |0><1|) / sqrt(2)
#   4: |2><2|
_INV_SQRT2 = 1.0 / math.sqrt(2.0)
HERMITIAN_BASIS_1Q: Tuple[np.ndarray, ...] = (
    np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.complex128),
    np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.complex128),
    np.array(
        [[0.0, _INV_SQRT2, 0.0], [_INV_SQRT2, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.complex128,
    ),
    np.array(
        [[0.0, -1.0j * _INV_SQRT2, 0.0], [1.0j * _INV_SQRT2, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.complex128,
    ),
    np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.complex128),
)
HERMITIAN_BASIS_2Q_STACK: np.ndarray = np.stack(
    [np.kron(a, b) for a in HERMITIAN_BASIS_1Q for b in HERMITIAN_BASIS_1Q], axis=0
)


def default_lru_params_for_profile(
    profile: gate.TargetNoiseProfile,
    p_leak_2q: float,
    p_ro_2_as_1: float = P_RO_2_AS_1_DEFAULT,
) -> QutritLRUParams:
    """Return target-appropriate physical seepage and LRU pulse parameters."""
    p_seep_round = (
        min(1.0, profile.t_round_us / (0.5 * profile.t1_us)) if profile.t1_us > 0.0 else 0.0
    )
    p_seep_reset = (
        min(1.0, profile.t_reset_us / (0.5 * profile.t1_us)) if profile.t1_us > 0.0 else 0.0
    )
    is_ion = "H2" in profile.profile_id
    return QutritLRUParams(
        p_leak_2q=float(p_leak_2q),
        p_seep_round=p_seep_round,
        p_seep_reset=p_seep_reset,
        p_ro_2_as_1=float(p_ro_2_as_1),
        lru_infidelity=ION_LRU_INFIDELITY if is_ion else TRANSMON_LRU_INFIDELITY,
        p_lru_1q=ION_DQLR_1Q_ERROR if is_ion else TRANSMON_DQLR_1Q_ERROR,
        dt_lru_us=ION_DQLR_DURATION_US if is_ion else TRANSMON_DQLR_DURATION_US,
    )


# ---------------------------------------------------------------------------
# Exact CPTP 3-Level & 9-Level Qutrit Kraus Channels
# ---------------------------------------------------------------------------


def cptp_completeness_error(kraus_ops: Sequence[np.ndarray]) -> float:
    """Return `|| sum_m K_m^dag K_m - I ||_inf` to audit exact CPTP trace preservation."""
    dim = kraus_ops[0].shape[0]
    acc = np.zeros((dim, dim), dtype=np.complex128)
    for k in kraus_ops:
        acc += k.conj().T @ k
    return float(np.max(np.abs(acc - np.eye(dim, dtype=np.complex128))))


def build_qutrit_depol1_kraus(u2: np.ndarray, p_1q: float) -> List[np.ndarray]:
    """Computational 1Q unitary `u2` followed by computational depolarizing noise `p_1q`."""
    u3 = embed_2q_to_3q_1q(u2)
    if p_1q <= 0.0:
        return [u3]
    w = math.sqrt(p_1q / 3.0)
    return [math.sqrt(1.0 - p_1q) * u3] + [w * (p3 @ u3) for p3 in (X3, Y3, Z3)]


def build_qutrit_zz_corr_kraus(p_corr: float) -> List[np.ndarray]:
    """Correlated spectator `ZZ` phase-flip channel on computational subspace."""
    i9 = np.eye(9, dtype=np.complex128)
    if p_corr <= 0.0:
        return [i9]
    return [math.sqrt(1.0 - p_corr) * i9, math.sqrt(p_corr) * np.kron(Z3, Z3)]


def build_qutrit_t1_t2_seep_kraus(
    dt_us: float, t1_us: float, t2_us: float, p_seep_override: Optional[float] = None
) -> List[np.ndarray]:
    """Exact computational `T1`/`T2` damping composed with `|2> -> |1>` spontaneous seepage."""
    if dt_us <= 0.0 and (p_seep_override is None or p_seep_override <= 0.0):
        return [I3]
    gamma1 = 1.0 - math.exp(-dt_us / t1_us) if (dt_us > 0.0 and t1_us > 0.0) else 0.0
    rate_phi = max(
        0.0,
        (2.0 / t2_us if t2_us > 0.0 else 0.0) - (1.0 / t1_us if t1_us > 0.0 else 0.0),
    )
    gamma_phi = 1.0 - math.exp(-dt_us * rate_phi) if dt_us > 0.0 else 0.0

    k0 = np.array([[1.0, 0.0], [0.0, math.sqrt(1.0 - gamma1)]], dtype=np.complex128)
    k1 = np.array([[0.0, math.sqrt(gamma1)], [0.0, 0.0]], dtype=np.complex128)
    p0 = np.array([[1.0, 0.0], [0.0, math.sqrt(1.0 - gamma_phi)]], dtype=np.complex128)
    p1 = np.array([[0.0, 0.0], [0.0, math.sqrt(gamma_phi)]], dtype=np.complex128)

    comp_ops: List[np.ndarray] = []
    first = True
    for pm in (p0, p1):
        for km in (k0, k1):
            op3 = np.zeros((3, 3), dtype=np.complex128)
            op3[:2, :2] = pm @ km
            if first:
                op3[2, 2] = 1.0
                first = False
            if np.linalg.norm(op3) > 1e-15:
                comp_ops.append(op3)

    p_seep = (
        float(p_seep_override)
        if p_seep_override is not None
        else (min(1.0, dt_us / (0.5 * t1_us)) if (dt_us > 0.0 and t1_us > 0.0) else 0.0)
    )
    if p_seep <= 0.0:
        return comp_ops

    s0 = np.diag([1.0, 1.0, math.sqrt(1.0 - p_seep)]).astype(np.complex128)
    s1 = np.zeros((3, 3), dtype=np.complex128)
    s1[1, 2] = math.sqrt(p_seep)
    return [s @ c for s in (s0, s1) for c in comp_ops if np.linalg.norm(s @ c) > 1e-15]


def build_qutrit_cz_cx_kraus(
    delta_theta_rad: float, p_2q: float, p_leak_2q: float
) -> List[np.ndarray]:
    """Exact `9 x 9` CPTP Kraus operators for native `H_d - CZ(a, d) - H_d` on two qutrits.

    Subspace action on `(ancilla a, data d)` in `{|0>, |1>, |2>}^{x 2}`:
      - `{|0>,|1>} x {|0>,|1>}`: Diabatic `|11> -> |02> / |2,+>` leakage heating at gate-averaged
        rate `p_leak_2q` (`lambda_11 = 4 * p_leak_2q`), composed on the surviving computational
        branch with `R_ZZ(delta_theta) CX_{a->d}` and 2-qubit depolarizing noise `p_2q`.
      - `{|0>,|1>} x {|2>}` (leaked data `|2>`): Computational `CZ` fails; spectator `|2>_d`
        injects 50% `Z_a` phase/syndrome randomization on ancilla `a`.
      - `{|2>} x {|0>,|1>}` (leaked ancilla `|2>`): Computational `CZ` fails; leaked `|2>_a`
        injects persistent 50% `Z_d` phase randomization (space-time hook error) on data `d`.
      - `{|22>}`: Identity `|22><22|`.
    """
    comp_idx = [0, 1, 3, 4]
    u_comp4 = gate._rzz(delta_theta_rad) @ gate.CX4
    lam = min(0.99, 4.0 * max(0.0, p_leak_2q))
    ops: List[np.ndarray] = []

    k_surv4 = np.diag([1.0, 1.0, 1.0, math.sqrt(1.0 - lam)]).astype(np.complex128)
    base4 = u_comp4 @ k_surv4
    paulis2 = (gate.I2, gate.X2, gate.Y2, gate.Z2)

    m0 = np.zeros((9, 9), dtype=np.complex128)
    m0[np.ix_(comp_idx, comp_idx)] = math.sqrt(max(0.0, 1.0 - p_2q)) * base4
    ops.append(m0)

    if p_2q > 0.0:
        w = math.sqrt(p_2q / 15.0)
        for i in range(4):
            for j in range(4):
                if i == 0 and j == 0:
                    continue
                mp = np.zeros((9, 9), dtype=np.complex128)
                mp[np.ix_(comp_idx, comp_idx)] = w * (np.kron(paulis2[i], paulis2[j]) @ base4)
                ops.append(mp)

    if lam > 0.0:
        # Equal branching from |11> (idx 4) into data leakage |0,2> (idx 2) and ancilla leakage
        # |2,0> followed by post-CZ H_d -> |2,+> (indices 6 and 7)
        k_data_leak = np.zeros((9, 9), dtype=np.complex128)
        k_data_leak[2, 4] = math.sqrt(0.5 * lam)
        ops.append(k_data_leak)

        k_anc_leak = np.zeros((9, 9), dtype=np.complex128)
        k_anc_leak[6, 4] = math.sqrt(0.25 * lam)
        k_anc_leak[7, 4] = math.sqrt(0.25 * lam)
        ops.append(k_anc_leak)

    # Leaked data d=2 (indices 2,5) -> 50% Z_a randomization; leaked ancilla a=2 (6,7) -> 50% Z_d
    for i0, i1 in ((2, 5), (6, 7)):
        kh_id = np.zeros((9, 9), dtype=np.complex128)
        kh_id[i0, i0] = _INV_SQRT2
        kh_id[i1, i1] = _INV_SQRT2
        kh_z = np.zeros((9, 9), dtype=np.complex128)
        kh_z[i0, i0] = _INV_SQRT2
        kh_z[i1, i1] = -_INV_SQRT2
        ops.extend([kh_id, kh_z])

    k_22 = np.zeros((9, 9), dtype=np.complex128)
    k_22[8, 8] = 1.0
    ops.append(k_22)
    return ops


def build_data_dqlr_kraus(
    eta_dqlr: float, p_lru_1q: float, dt_lru_us: float, t1_us: float, t2_us: float
) -> List[np.ndarray]:
    """Data-qubit DQLR / SWAP-LRU seepage pulse `|2> -> |1>` + 1Q gate & duration overhead."""
    d0 = np.diag([1.0, 1.0, math.sqrt(max(0.0, 1.0 - eta_dqlr))]).astype(np.complex128)
    d1 = np.zeros((3, 3), dtype=np.complex128)
    d1[1, 2] = math.sqrt(eta_dqlr)
    pulse_ops = [d0, d1] if eta_dqlr > 0.0 else [I3]
    dep_ops = build_qutrit_depol1_kraus(gate.I2, p_lru_1q)
    rel_ops = build_qutrit_t1_t2_seep_kraus(dt_lru_us, t1_us, t2_us)
    return [
        r @ dp @ pu
        for r in rel_ops
        for dp in dep_ops
        for pu in pulse_ops
        if np.linalg.norm(r @ dp @ pu) > 1e-15
    ]


# ---------------------------------------------------------------------------
# Explicit Full 243 x 243 (3^5 = 243-State) 5-Qutrit Density-Matrix Engine
# ---------------------------------------------------------------------------


def _dm243_apply_kraus_1q(
    rho243: np.ndarray, kraus_ops: Sequence[np.ndarray], q: int
) -> np.ndarray:
    t = rho243.reshape((3,) * 10)
    out = np.zeros_like(rho243)
    for k in kraus_ops:
        s = np.moveaxis(np.tensordot(k, t, axes=([1], [q])), 0, q)
        s = np.moveaxis(np.tensordot(k.conj(), s, axes=([1], [5 + q])), 0, 5 + q)
        out += s.reshape((243, 243))
    return out


def _dm243_apply_kraus_2q(
    rho243: np.ndarray, kraus_ops_9x9: Sequence[np.ndarray], q1: int, q2: int
) -> np.ndarray:
    t = rho243.reshape((3,) * 10)
    out = np.zeros_like(rho243)
    for k9 in kraus_ops_9x9:
        kt = k9.reshape(3, 3, 3, 3)
        s = np.moveaxis(np.tensordot(kt, t, axes=([2, 3], [q1, q2])), [0, 1], [q1, q2])
        s = np.moveaxis(
            np.tensordot(kt.conj(), s, axes=([2, 3], [5 + q1, 5 + q2])),
            [0, 1],
            [5 + q1, 5 + q2],
        )
        out += s.reshape((243, 243))
    return out


def simulate_5qutrit_243_kraus_round(
    rho_data_27: np.ndarray,
    anc_leak_pair: Tuple[int, int],
    profile: gate.TargetNoiseProfile,
    lru_params: QutritLRUParams,
    architecture: str,
) -> Dict[int, np.ndarray]:
    """Evolve a `27 x 27` data-qutrit state inside the full `243 x 243` (`3^5`) Hilbert space.

    Returns the 16 unnormalized `27 x 27` post-measurement data-qutrit blocks indexed by
    `code = (((l3_next << 1) | l4_next) << 2) | m_obs`.
    """
    if architecture not in LRU_ARCHITECTURES:
        raise ValueError(f"Unsupported architecture: {architecture}")

    pres = profile.p_reset
    pro = profile.p_readout
    dth = profile.delta_theta_rad

    if architecture == "NO_LRU":
        rho_anc0 = np.diag([1.0 - pres, pres, 0.0]).astype(np.complex128)
        rho_anc1 = np.diag(
            [0.0, lru_params.p_seep_reset, 1.0 - lru_params.p_seep_reset]
        ).astype(np.complex128)
    else:
        pres_lru = min(0.49, pres + lru_params.lru_infidelity)
        res_2 = 0.2 * lru_params.lru_infidelity
        rho_anc0 = np.diag([1.0 - pres_lru, pres_lru, 0.0]).astype(np.complex128)
        rho_anc1 = np.diag(
            [(1.0 - res_2) * (1.0 - pres_lru), (1.0 - res_2) * pres_lru, res_2]
        ).astype(np.complex128)

    anc_pair_mats = (rho_anc0, rho_anc1)
    rho243 = np.kron(
        np.kron(rho_data_27, anc_pair_mats[anc_leak_pair[0]]),
        anc_pair_mats[anc_leak_pair[1]],
    )

    h_ops = build_qutrit_depol1_kraus(gate._rx(dth) @ gate.H2, profile.p_1q)
    cz_ops = build_qutrit_cz_cx_kraus(dth, profile.p_2q, lru_params.p_leak_2q)
    zz_ops = build_qutrit_zz_corr_kraus(profile.p_corr)
    idle_ops = [
        embed_2q_to_3q_1q(gate._rz(dth)) @ k
        for k in build_qutrit_t1_t2_seep_kraus(
            profile.t_round_us,
            profile.t1_us,
            profile.t2_us,
            p_seep_override=lru_params.p_seep_round,
        )
    ]

    for a in (3, 4):
        rho243 = _dm243_apply_kraus_1q(rho243, h_ops, a)
    for a, d in ((3, 0), (4, 1), (3, 1), (4, 2)):
        rho243 = _dm243_apply_kraus_2q(rho243, cz_ops, a, d)
    if profile.p_corr > 0.0:
        for u, v in ((0, 3), (3, 1), (1, 4), (4, 2)):
            rho243 = _dm243_apply_kraus_2q(rho243, zz_ops, u, v)
    for d in (0, 1, 2):
        rho243 = _dm243_apply_kraus_1q(rho243, idle_ops, d)
    for a in (3, 4):
        rho243 = _dm243_apply_kraus_1q(rho243, h_ops, a)

    if architecture == "FULL_DATA_AND_ANCILLA_LRU":
        dqlr_ops = build_data_dqlr_kraus(
            1.0 - lru_params.lru_infidelity,
            lru_params.p_lru_1q,
            lru_params.dt_lru_us,
            profile.t1_us,
            profile.t2_us,
        )
        for d in (0, 1, 2):
            rho243 = _dm243_apply_kraus_1q(rho243, dqlr_ops, d)

    t9 = rho243.reshape(27, 9, 27, 9)
    ro_p1 = (pro, 1.0 - pro, lru_params.p_ro_2_as_1)
    ro_p0 = (1.0 - pro, pro, 1.0 - lru_params.p_ro_2_as_1)

    blocks: Dict[int, np.ndarray] = {
        c: np.zeros((27, 27), dtype=np.complex128) for c in range(16)
    }
    for t3 in range(3):
        l3_n = 1 if t3 == 2 else 0
        for t4 in range(3):
            l4_n = 1 if t4 == 2 else 0
            m_true = 3 * t3 + t4
            blk = t9[:, m_true, :, m_true]
            for o3 in (0, 1):
                p3 = ro_p1[t3] if o3 == 1 else ro_p0[t3]
                for o4 in (0, 1):
                    p4 = ro_p1[t4] if o4 == 1 else ro_p0[t4]
                    code = (((l3_n << 1) | l4_n) << 2) | ((o3 << 1) | o4)
                    blocks[code] += (p3 * p4) * blk
    return blocks


# ---------------------------------------------------------------------------
# Fast Exact 125-Dimensional Real-Hermitian Quantum-Instrument Engine
# ---------------------------------------------------------------------------


def _kraus1_to_real5(kraus_ops: Sequence[np.ndarray]) -> np.ndarray:
    s5 = np.zeros((5, 5), dtype=np.float64)
    for col, b_in in enumerate(HERMITIAN_BASIS_1Q):
        out = sum(k @ b_in @ k.conj().T for k in kraus_ops)
        for row, b_out in enumerate(HERMITIAN_BASIS_1Q):
            s5[row, col] = float(np.real(np.trace(b_out.conj().T @ out)))
    return np.ascontiguousarray(s5)


def _kraus2_to_real25(kraus_ops_9x9: Sequence[np.ndarray]) -> np.ndarray:
    in_stack = HERMITIAN_BASIS_2Q_STACK
    out_stack = np.zeros_like(in_stack)
    for k in kraus_ops_9x9:
        out_stack += k[None, :, :] @ in_stack @ k.conj().T[None, :, :]
    hb2_flat = in_stack.conj().reshape(25, 81)
    out_flat = out_stack.reshape(25, 81)
    return np.ascontiguousarray(np.real(hb2_flat @ out_flat.T), dtype=np.float64)


def _apply_r5_1q(state: np.ndarray, s5: np.ndarray, q: int) -> np.ndarray:
    ax = 1 + q
    return np.moveaxis(np.tensordot(s5, state, axes=([1], [ax])), 0, ax)


def _apply_r25_2q(state: np.ndarray, s25: np.ndarray, q1: int, q2: int) -> np.ndarray:
    st = s25.reshape(5, 5, 5, 5)
    a1, a2 = 1 + q1, 1 + q2
    return np.moveaxis(np.tensordot(st, state, axes=([2, 3], [a1, a2])), [0, 1], [a1, a2])


def vec125_to_dm27(vec125: np.ndarray) -> np.ndarray:
    """Reconstruct the exact `27 x 27` 3-qutrit Hermitian density matrix from its `125`-vector."""
    t = vec125.reshape(5, 5, 5)
    out = np.zeros((27, 27), dtype=np.complex128)
    for i0 in range(5):
        for i1 in range(5):
            b01 = np.kron(HERMITIAN_BASIS_1Q[i0], HERMITIAN_BASIS_1Q[i1])
            for i2 in range(5):
                w = float(t[i0, i1, i2])
                if abs(w) > 0.0:
                    out += w * np.kron(b01, HERMITIAN_BASIS_1Q[i2])
    return out


def build_qutrit_lru_superoperators(
    profile: gate.TargetNoiseProfile,
    lru_params: QutritLRUParams,
    architecture: str,
) -> Dict[str, Any]:
    """Construct exact `125 x 125` real-Hermitian quantum-instrument superoperators."""
    if architecture not in LRU_ARCHITECTURES:
        raise ValueError(f"Unsupported architecture: {architecture}")

    dth = profile.delta_theta_rad
    pres = profile.p_reset
    pro = profile.p_readout
    pc = profile.p_corr

    s5_reset = _kraus1_to_real5(
        [math.sqrt(1.0 - pres) * I3, math.sqrt(pres) * X3] if pres > 0.0 else [I3]
    )
    s5_h = _kraus1_to_real5(build_qutrit_depol1_kraus(gate._rx(dth) @ gate.H2, profile.p_1q))
    s25_cz = _kraus2_to_real25(build_qutrit_cz_cx_kraus(dth, profile.p_2q, lru_params.p_leak_2q))
    s25_zz = _kraus2_to_real25(build_qutrit_zz_corr_kraus(pc))
    s5_idle = _kraus1_to_real5(
        [
            embed_2q_to_3q_1q(gate._rz(dth)) @ k
            for k in build_qutrit_t1_t2_seep_kraus(
                profile.t_round_us,
                profile.t1_us,
                profile.t2_us,
                p_seep_override=lru_params.p_seep_round,
            )
        ]
    )
    s5_dqlr = (
        _kraus1_to_real5(
            build_data_dqlr_kraus(
                1.0 - lru_params.lru_infidelity,
                lru_params.p_lru_1q,
                lru_params.dt_lru_us,
                profile.t1_us,
                profile.t2_us,
            )
        )
        if architecture == "FULL_DATA_AND_ANCILLA_LRU"
        else None
    )

    init_t = np.zeros((1, 5, 5, 5), dtype=np.float64)
    init_t[0, 0, 0, 0] = 1.0
    for d in (0, 1, 2):
        init_t = _apply_r5_1q(init_t, s5_reset, d)
        init_t = _apply_r5_1q(init_t, s5_h, d)
    rho_init_125 = np.ascontiguousarray(init_t.reshape(125))

    if architecture == "NO_LRU":
        anc_v0 = np.array([1.0 - pres, pres, 0.0, 0.0, 0.0], dtype=np.float64)
        anc_v1 = np.array(
            [0.0, lru_params.p_seep_reset, 0.0, 0.0, 1.0 - lru_params.p_seep_reset],
            dtype=np.float64,
        )
    else:
        pres_lru = min(0.49, pres + lru_params.lru_infidelity)
        res_2 = 0.2 * lru_params.lru_infidelity
        anc_v0 = np.array([1.0 - pres_lru, pres_lru, 0.0, 0.0, 0.0], dtype=np.float64)
        anc_v1 = np.array(
            [(1.0 - res_2) * (1.0 - pres_lru), (1.0 - res_2) * pres_lru, 0.0, 0.0, res_2],
            dtype=np.float64,
        )
    anc_vecs = (anc_v0, anc_v1)

    ro_p1 = (pro, 1.0 - pro, lru_params.p_ro_2_as_1)
    ro_p0 = (1.0 - pro, pro, 1.0 - lru_params.p_ro_2_as_1)
    diag_idx5 = (0, 1, 4)

    eye_125 = np.eye(125, dtype=np.float64).reshape(125, 5, 5, 5)
    ops_by_lin: Dict[int, List[np.ndarray]] = {}
    instr_stacks: Dict[int, np.ndarray] = {}
    active_codes: Dict[int, np.ndarray] = {}

    for l_in in range(4):
        if lru_params.p_leak_2q == 0.0 and l_in > 0:
            continue
        l3, l4 = (l_in >> 1) & 1, l_in & 1
        st = np.einsum("bijk,l,m->bijklm", eye_125, anc_vecs[l3], anc_vecs[l4])
        for a in (3, 4):
            st = _apply_r5_1q(st, s5_h, a)
        for a, d in ((3, 0), (4, 1), (3, 1), (4, 2)):
            st = _apply_r25_2q(st, s25_cz, a, d)
        if pc > 0.0:
            for u, v in ((0, 3), (3, 1), (1, 4), (4, 2)):
                st = _apply_r25_2q(st, s25_zz, u, v)
        for d in (0, 1, 2):
            st = _apply_r5_1q(st, s5_idle, d)
        for a in (3, 4):
            st = _apply_r5_1q(st, s5_h, a)
        if s5_dqlr is not None:
            for d in (0, 1, 2):
                st = _apply_r5_1q(st, s5_dqlr, d)

        st_flat = st.reshape(125, 125, 5, 5)
        ops_16 = [np.zeros((125, 125), dtype=np.float64) for _ in range(16)]
        for t3 in range(3):
            i3, l3_n = diag_idx5[t3], (1 if t3 == 2 else 0)
            for t4 in range(3):
                i4, l4_n = diag_idx5[t4], (1 if t4 == 2 else 0)
                blk = np.ascontiguousarray(st_flat[:, :, i3, i4].T)
                for o3 in (0, 1):
                    p3 = ro_p1[t3] if o3 == 1 else ro_p0[t3]
                    for o4 in (0, 1):
                        p4 = ro_p1[t4] if o4 == 1 else ro_p0[t4]
                        code = (((l3_n << 1) | l4_n) << 2) | ((o3 << 1) | o4)
                        ops_16[code] += (p3 * p4) * blk

        ops_by_lin[l_in] = ops_16
        nz_codes = [c for c in range(16) if np.max(np.abs(ops_16[c])) > 1e-16]
        active_codes[l_in] = np.array(nz_codes, dtype=np.uint8)
        instr_stacks[l_in] = np.ascontiguousarray(np.vstack([ops_16[c] for c in nz_codes]))

    term_st = eye_125.copy()
    for d in (0, 1, 2):
        term_st = _apply_r5_1q(term_st, s5_h, d)
    diag_27 = np.zeros((27, 125), dtype=np.float64)
    for q0 in range(3):
        for q1 in range(3):
            for q2 in range(3):
                diag_27[9 * q0 + 3 * q1 + q2, :] = term_st[
                    :, diag_idx5[q0], diag_idx5[q1], diag_idx5[q2]
                ]
    ro_1q = np.array(
        [[ro_p0[0], ro_p0[1], ro_p0[2]], [ro_p1[0], ro_p1[1], ro_p1[2]]], dtype=np.float64
    )
    meas_linear = np.ascontiguousarray(np.kron(np.kron(ro_1q, ro_1q), ro_1q) @ diag_27)

    trace_1q = np.array([1.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float64)
    trace_vec = np.ascontiguousarray(np.kron(np.kron(trace_1q, trace_1q), trace_1q))
    leak_1q = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    leak_data_vecs = np.ascontiguousarray(
        np.vstack(
            [
                np.kron(np.kron(leak_1q, trace_1q), trace_1q),
                np.kron(np.kron(trace_1q, leak_1q), trace_1q),
                np.kron(np.kron(trace_1q, trace_1q), leak_1q),
            ]
        )
    )

    return {
        "rho_init_125": rho_init_125,
        "ops_by_lin": ops_by_lin,
        "instr_stacks": instr_stacks,
        "active_codes": active_codes,
        "meas_linear": meas_linear,
        "trace_vec": trace_vec,
        "leak_data_vecs": leak_data_vecs,
    }


def exact_qutrit_branch_sum_arm4a(
    rounds: int,
    bundle: Mapping[str, Any],
    decoder: gate.AuditableMWPMDecoder,
) -> float:
    """Exact zero-variance multi-round qutrit quantum-instrument branch sum (`R <= 3`)."""
    active_codes: Mapping[int, np.ndarray] = bundle["active_codes"]
    ops_by_lin: Mapping[int, List[np.ndarray]] = bundle["ops_by_lin"]
    meas_linear: np.ndarray = bundle["meas_linear"]

    branches: Dict[Tuple[int, int], np.ndarray] = {(0, 0): bundle["rho_init_125"].copy()}
    for _ in range(rounds):
        next_branches: Dict[Tuple[int, int], np.ndarray] = {}
        for (m_hist, l_in), vec in branches.items():
            for code_raw in active_codes[l_in]:
                code = int(code_raw)
                m_obs = code & 3
                l_next = (code >> 2) & 3
                nxt_vec = ops_by_lin[l_in][code] @ vec
                key = ((m_hist << 2) | m_obs, l_next)
                if key in next_branches:
                    next_branches[key] = next_branches[key] + nxt_vec
                else:
                    next_branches[key] = nxt_vec
        branches = next_branches

    by_mhist: Dict[int, np.ndarray] = {}
    for (m_hist, _), vec in branches.items():
        if m_hist in by_mhist:
            by_mhist[m_hist] = by_mhist[m_hist] + vec
        else:
            by_mhist[m_hist] = vec

    p_fail = 0.0
    for h, vec in by_mhist.items():
        term_probs = np.maximum(0.0, meas_linear @ vec)
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
            if decoder.decode_mask(full_mask) != (d0 ^ d1 ^ d2):
                p_fail += float(term_probs[d_out])
    return p_fail


def evaluate_qutrit_lru_cell(
    profile: gate.TargetNoiseProfile,
    lru_params: QutritLRUParams,
    architecture: str,
    card_c_cell_ref: Mapping[str, Any],
    rounds_list: Sequence[int] = ROUNDS_GRID,
    n_shots: int = N_SHOTS_DEFAULT,
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Evaluate `[[3,1,3]]` `arm4a` under qutrit `|2>` leakage and `architecture` across `R`."""
    bundle = build_qutrit_lru_superoperators(profile, lru_params, architecture)
    instr_stacks = bundle["instr_stacks"]
    active_codes = bundle["active_codes"]
    meas_linear = bundle["meas_linear"]
    trace_vec = bundle["trace_vec"]
    leak_data_vecs = bundle["leak_data_vecs"]

    rng = np.random.default_rng(seed)

    p_arm4a_list: List[float] = []
    mean_data_p2_by_r: Dict[str, float] = {}
    mean_anc_p2_by_r: Dict[str, float] = {}
    anc_streak_ge2_r9 = 0.0
    exact_branch_checks: Dict[str, Dict[str, float]] = {}

    for r_val in rounds_list:
        dem = gate.build_repetition_d3_dem(rounds=r_val, profile=profile, basis="X")
        decoder = gate.auditable_mwpm_decoder(dem)

        u_rounds = rng.random((r_val, n_shots))
        states = np.repeat(bundle["rho_init_125"].reshape(125, 1), n_shots, axis=1)
        anc_l = np.zeros(n_shots, dtype=np.uint8)
        anc_hist = np.zeros((n_shots, r_val), dtype=np.uint8)
        anc_leak_hist = np.zeros((n_shots, r_val), dtype=np.uint8)

        for r_step in range(r_val):
            chosen_code = np.zeros(n_shots, dtype=np.uint8)
            next_states = np.empty_like(states)
            u = u_rounds[r_step]
            for l_in, codes in active_codes.items():
                sel_l = np.where(anc_l == l_in)[0]
                if sel_l.size == 0:
                    continue
                k_br = codes.size
                all_cands = (instr_stacks[l_in] @ states[:, sel_l]).reshape(
                    k_br, 125, sel_l.size
                )
                w_br = np.maximum(0.0, np.einsum("d,kds->ks", trace_vec, all_cands))
                w_sum = np.maximum(np.sum(w_br, axis=0), 1e-30)
                cdf = np.cumsum(w_br / w_sum, axis=0)
                idx_ch = np.clip(np.sum(u[sel_l][None, :] >= cdf, axis=0), 0, k_br - 1)
                chosen_code[sel_l] = codes[idx_ch]
                for ki in range(k_br):
                    loc = idx_ch == ki
                    if np.any(loc):
                        nrm = np.maximum(w_br[ki, loc], 1e-30)
                        next_states[:, sel_l[loc]] = all_cands[ki][:, loc] / nrm[None, :]
            states = next_states
            anc_hist[:, r_step] = chosen_code & 3
            anc_l = (chosen_code >> 2) & 3
            anc_leak_hist[:, r_step] = anc_l

        data_p2_per_q_shot = np.maximum(0.0, leak_data_vecs @ states)
        mean_data_p2_by_r[str(r_val)] = round(float(np.mean(data_p2_per_q_shot)), 8)
        anc_l3 = (anc_leak_hist >> 1) & 1
        anc_l4 = anc_leak_hist & 1
        mean_anc_p2_by_r[str(r_val)] = round(float(0.5 * np.mean(anc_l3 + anc_l4)), 8)
        if r_val == rounds_list[-1] and r_val >= 2:
            streak3 = np.any((anc_l3[:, :-1] == 1) & (anc_l3[:, 1:] == 1), axis=1)
            streak4 = np.any((anc_l4[:, :-1] == 1) & (anc_l4[:, 1:] == 1), axis=1)
            anc_streak_ge2_r9 = round(float(np.mean(streak3 | streak4)), 7)

        term = np.maximum(0.0, meas_linear @ states)
        term /= np.maximum(np.sum(term, axis=0, keepdims=True), 1e-30)

        base_dmasks = np.zeros(n_shots, dtype=np.int64)
        for r_step in range(r_val):
            mc = anc_hist[:, r_step]
            mp = np.zeros(n_shots, dtype=np.uint8) if r_step == 0 else anc_hist[:, r_step - 1]
            diff = mc ^ mp
            base_dmasks |= (((diff >> 1) & 1).astype(np.int64)) << (2 * r_step)
            base_dmasks |= ((diff & 1).astype(np.int64)) << (2 * r_step + 1)

        ml = anc_hist[:, r_val - 1]
        ml0, ml1 = (ml >> 1) & 1, ml & 1
        fail_prob_per_shot = np.zeros(n_shots, dtype=np.float64)
        for d_out in range(8):
            d0, d1, d2 = (d_out >> 2) & 1, (d_out >> 1) & 1, d_out & 1
            full_masks = (
                base_dmasks
                | ((ml0 ^ d0 ^ d1).astype(np.int64) << (2 * r_val))
                | ((ml1 ^ d1 ^ d2).astype(np.int64) << (2 * r_val + 1))
            )
            wrong = decoder.decode_batch(full_masks) != (d0 ^ d1 ^ d2)
            fail_prob_per_shot += term[d_out, :] * wrong

        p_fail = float(np.mean(fail_prob_per_shot))
        p_arm4a_list.append(p_fail)

        max_exact_r = 3 if lru_params.p_leak_2q == 0.0 else 2
        if r_val <= max_exact_r:
            p_exact = exact_qutrit_branch_sum_arm4a(r_val, bundle, decoder)
            exact_branch_checks[str(r_val)] = {
                "exact_qutrit_branch_p_fail": round(p_exact, 8),
                "trajectory_mc_p_fail": round(p_fail, 8),
                "abs_diff": round(abs(p_exact - p_fail), 8),
            }

    shots_vec = [n_shots] * len(rounds_list)
    fit = gate.fit_epsilon_L_vs_rounds(rounds_list, p_arm4a_list, shots_vec, n_boot, seed + 3)

    u_ref = card_c_cell_ref["unconditional_arms"]
    p_arm1 = [float(u_ref["arm1_unencoded"]["p_fail_by_round"][str(r)]) for r in rounds_list]
    p_arm2 = [
        float(u_ref["arm2_encoded_no_correction"]["p_fail_by_round"][str(r)])
        for r in rounds_list
    ]
    crossing = gate._find_crossing_rounds(rounds_list, p_arm4a_list, p_arm1, p_arm2)

    zero_leak_slope = float(u_ref["arm4a_decoded_pauli_frame"]["fit"]["epsilon_L_slope_fit"])
    zero_leak_r9 = float(u_ref["arm4a_decoded_pauli_frame"]["fit"]["epsilon_L_at_max_round"])

    return {
        "target_profile_id": profile.profile_id,
        "architecture": architecture,
        "lru_parameters": asdict(lru_params),
        "n_shots_per_round": n_shots,
        "rounds_evaluated": list(rounds_list),
        "p_fail_by_round": {str(r): round(p, 7) for r, p in zip(rounds_list, p_arm4a_list)},
        "fit": fit,
        "delta_eps_L_slope_vs_zero_leak": round(
            float(fit["epsilon_L_slope_fit"]) - zero_leak_slope, 8
        ),
        "delta_eps_L_r9_vs_zero_leak": round(
            float(fit["epsilon_L_at_max_round"]) - zero_leak_r9, 8
        ),
        "leakage_diagnostics": {
            "mean_data_qutrit_p2_by_round": mean_data_p2_by_r,
            "mean_ancilla_qutrit_p2_by_round": mean_anc_p2_by_r,
            "ancilla_streak_ge2_fraction_at_max_round": anc_streak_ge2_r9,
        },
        "exact_qutrit_branch_crosschecks": exact_branch_checks,
        "crossing_vs_card_c_baselines": {
            **crossing,
            "verdict": (
                "PASS_CROSSING_OBSERVED"
                if crossing["crosses_below_both_sustained"]
                else "STOP_NO_CROSSING_AT_TESTED_ROUNDS"
            ),
        },
    }


def _audit_243x243_vs_liouville125(profile: gate.TargetNoiseProfile) -> Dict[str, Any]:
    """Verify full `243 x 243` (`3^5`) Kraus round matches `125`-dim real superoperators."""
    params = default_lru_params_for_profile(profile, p_leak_2q=1.2e-3)
    cptp_errs = {
        "cz_cx_qutrit_9x9": cptp_completeness_error(
            build_qutrit_cz_cx_kraus(profile.delta_theta_rad, profile.p_2q, 1.2e-3)
        ),
        "t1_t2_seep_3x3": cptp_completeness_error(
            build_qutrit_t1_t2_seep_kraus(
                profile.t_round_us, profile.t1_us, profile.t2_us, params.p_seep_round
            )
        ),
        "data_dqlr_3x3": cptp_completeness_error(
            build_data_dqlr_kraus(
                1.0 - params.lru_infidelity,
                params.p_lru_1q,
                params.dt_lru_us,
                profile.t1_us,
                profile.t2_us,
            )
        ),
    }

    rng = np.random.default_rng(20261124)
    psi_comp = rng.normal(size=2) + 1.0j * rng.normal(size=2)
    psi_comp /= np.linalg.norm(psi_comp)
    rho_1q_a = 0.85 * np.outer(psi_comp, psi_comp.conj())
    rho3_a = np.zeros((3, 3), dtype=np.complex128)
    rho3_a[:2, :2] = rho_1q_a
    rho3_a[2, 2] = 0.15

    v5 = np.array(
        [float(np.real(np.trace(b.conj().T @ rho3_a))) for b in HERMITIAN_BASIS_1Q],
        dtype=np.float64,
    )
    vec125_in = np.kron(np.kron(v5, v5), v5)
    dm27_in = vec125_to_dm27(vec125_in)

    max_diffs_by_arch: Dict[str, float] = {}
    for arch in LRU_ARCHITECTURES:
        bundle = build_qutrit_lru_superoperators(profile, params, arch)
        arch_max = 0.0
        for l_in in (0, 1, 2, 3):
            anc_pair = ((l_in >> 1) & 1, l_in & 1)
            blocks_243 = simulate_5qutrit_243_kraus_round(
                dm27_in, anc_pair, profile, params, arch
            )
            for code in range(16):
                dm27_from_125 = vec125_to_dm27(bundle["ops_by_lin"][l_in][code] @ vec125_in)
                diff = float(np.max(np.abs(blocks_243[code] - dm27_from_125)))
                arch_max = max(arch_max, diff)
        max_diffs_by_arch[arch] = arch_max

    return {
        "cptp_completeness_errors": cptp_errs,
        "max_cptp_error": max(cptp_errs.values()),
        "full_243x243_vs_liouville125_max_abs_diff_by_arch": max_diffs_by_arch,
        "full_243x243_vs_liouville125_max_abs_diff": max(max_diffs_by_arch.values()),
    }


def run_card_c_qutrit_leakage_lru_gate(
    output_path: Optional[Path] = None,
    seed: int = DEFAULT_SEED,
    rounds_list: Sequence[int] = ROUNDS_GRID,
    n_shots: int = N_SHOTS_DEFAULT,
    n_boot: int = N_BOOT_DEFAULT,
) -> Dict[str, Any]:
    """Execute the full Card C 5-qutrit `|2>` leakage and `LRU` ablation suite."""
    if not CROSSING_REPORT_PATH.is_file():
        raise FileNotFoundError(
            f"Missing prerequisite Card C crossing report: {CROSSING_REPORT_PATH}"
        )
    crossing_rep = json.loads(CROSSING_REPORT_PATH.read_text(encoding="utf-8"))
    crossing_sha = hashlib.sha256(CROSSING_REPORT_PATH.read_bytes()).hexdigest()

    profiles = gate.build_published_target_profiles()
    audit_243 = _audit_243x243_vs_liouville125(profiles["FEZ_PUBLISHED"])

    sweep_plan: List[Tuple[str, Tuple[float, ...], int]] = [
        ("FEZ_PUBLISHED", FEZ_LEAK_SWEEP, seed),
        ("FEZ_DRIFTED_SPAM", FEZ_DRIFT_LEAK_SWEEP, seed + 1000),
        ("H2_1_PUBLISHED", H2_LEAK_SWEEP, seed + 2000),
    ]

    cells: Dict[str, Dict[str, Any]] = {}
    summary_rows: List[Dict[str, Any]] = []
    zero_leak_checks: Dict[str, Dict[str, float]] = {}

    for pid, leak_grid, prof_seed in sweep_plan:
        prof = profiles[pid]
        ref_cell = crossing_rep["cells"][f"repetition_d3_5q__{pid}"]
        for p_leak in leak_grid:
            lparams = default_lru_params_for_profile(prof, p_leak_2q=p_leak)
            for arch in LRU_ARCHITECTURES:
                cell_key = f"{pid}__pleak_{p_leak:.5f}__{arch}"
                res = evaluate_qutrit_lru_cell(
                    profile=prof,
                    lru_params=lparams,
                    architecture=arch,
                    card_c_cell_ref=ref_cell,
                    rounds_list=rounds_list,
                    n_shots=n_shots,
                    n_boot=n_boot,
                    seed=prof_seed,
                )
                cells[cell_key] = res
                summary_rows.append(
                    {
                        "target_profile_id": pid,
                        "p_leak_2q": p_leak,
                        "architecture": arch,
                        "eps_L_slope_fit": res["fit"]["epsilon_L_slope_fit"],
                        "eps_L_at_max_round": res["fit"]["epsilon_L_at_max_round"],
                        "delta_eps_L_slope_vs_zero_leak": res["delta_eps_L_slope_vs_zero_leak"],
                        "p_fail_max_round": res["p_fail_by_round"][str(rounds_list[-1])],
                        "data_qutrit_p2_max_round": res["leakage_diagnostics"][
                            "mean_data_qutrit_p2_by_round"
                        ][str(rounds_list[-1])],
                        "ancilla_streak_ge2_max_round": res["leakage_diagnostics"][
                            "ancilla_streak_ge2_fraction_at_max_round"
                        ],
                        "r_cross_vs_both": res["crossing_vs_card_c_baselines"][
                            "r_cross_vs_both_sustained"
                        ],
                        "verdict": res["crossing_vs_card_c_baselines"]["verdict"],
                    }
                )

                if p_leak == 0.0 and arch == "NO_LRU":
                    ref_p = ref_cell["unconditional_arms"]["arm4a_decoded_pauli_frame"][
                        "p_fail_by_round"
                    ]
                    max_p_diff = max(
                        abs(float(res["p_fail_by_round"][str(r)]) - float(ref_p[str(r)]))
                        for r in rounds_list
                    )
                    zero_leak_checks[pid] = {
                        "max_p_fail_diff_vs_card_c_report": round(max_p_diff, 12),
                        "eps_L_slope_diff_vs_card_c_report": round(
                            abs(float(res["delta_eps_L_slope_vs_zero_leak"])), 12
                        ),
                    }

    fez_high_no = cells["FEZ_PUBLISHED__pleak_0.00120__NO_LRU"]
    fez_high_anc = cells["FEZ_PUBLISHED__pleak_0.00120__ANCILLA_LRU_ONLY"]
    fez_high_full = cells["FEZ_PUBLISHED__pleak_0.00120__FULL_DATA_AND_ANCILLA_LRU"]

    strict_ordering_1p2e3 = bool(
        fez_high_no["fit"]["epsilon_L_slope_fit"]
        > fez_high_anc["fit"]["epsilon_L_slope_fit"]
        > fez_high_full["fit"]["epsilon_L_slope_fit"]
    )
    full_lru_preserves_all_crossings = all(
        row["verdict"] == "PASS_CROSSING_OBSERVED"
        for row in summary_rows
        if row["architecture"] == "FULL_DATA_AND_ANCILLA_LRU"
    )

    report: Dict[str, Any] = {
        "schema": "osiris.qec_card_c.qutrit_leakage_lru_report/1",
        "experiment_card": (
            "Card C: OSIRIS-QEC — Transmon Qutrit |2> Leakage & LRU Ablation "
            "(phase1/verification_queue.md)"
        ),
        "execution_tier": "SIMULATED",
        "provider_job_id": None,
        "qpu_seconds": 0,
        "hardware_execution_authorized": False,
        "seed": seed,
        "prerequisite_card_c_crossing_report_sha256": crossing_sha,
        "qutrit_model_documentation": {
            "hilbert_space": (
                "5-qutrit {|0>, |1>, |2>}^{x 5} (3^5 = 243 pure states; 125-dim exact "
                "block-coherent 3-data-qutrit Liouville instrument)"
            ),
            "code_evaluated": (
                "[[3,1,3]] distance-3 phase-flip repetition code "
                "(X-memory, 3 data + 2 syndrome ancillas)"
            ),
            "modeled_qutrit_mechanisms": [
                "cz_diabatic_11_to_02_20_leakage_heating",
                "spontaneous_qutrit_seepage_2_to_1_at_2x_t1_rate",
                "leaked_data_to_ancilla_50pct_syndrome_randomization",
                "leaked_ancilla_to_data_50pct_phase_hook_corruption",
                "dispersive_binary_readout_state2_as_1_bias_0p92",
                "ancilla_multilevel_reset_lru_infidelity",
                "data_dqlr_swap_lru_pulse_gate_and_120ns_idle_overhead",
                "ion_trap_h2_1_spontaneous_raman_hyperfine_leakage_baseline",
            ],
            "p_ro_2_as_1": P_RO_2_AS_1_DEFAULT,
            "fez_leak_sweep": list(FEZ_LEAK_SWEEP),
            "fez_drifted_spam_leak_sweep": list(FEZ_DRIFT_LEAK_SWEEP),
            "h2_1_leak_sweep": list(H2_LEAK_SWEEP),
            "lru_architectures": list(LRU_ARCHITECTURES),
        },
        "verification_controls": {
            **audit_243,
            "zero_leakage_exact_card_c_agreement": zero_leak_checks,
            "strict_lru_ordering_at_fez_1p2e3": strict_ordering_1p2e3,
            "full_lru_preserves_crossing_all_regimes": full_lru_preserves_all_crossings,
        },
        "summary_ablation_table": summary_rows,
        "cells": cells,
        "stop_go_decision": {
            "qutrit_243_state_engine_verified": bool(
                audit_243["full_243x243_vs_liouville125_max_abs_diff"] < 1e-13
                and audit_243["max_cptp_error"] < 1e-13
            ),
            "zero_leakage_matches_card_c_baseline": all(
                v["max_p_fail_diff_vs_card_c_report"] <= 1e-7 for v in zero_leak_checks.values()
            ),
            "strict_monotonicity_no_lru_gt_ancilla_gt_full_at_1p2e3": strict_ordering_1p2e3,
            "full_data_and_ancilla_lru_preserves_card_c_crossing": full_lru_preserves_all_crossings,
            "verdict": (
                "PASS_QUTRIT_LEAKAGE_GAP_CLOSED_WITH_LRU"
                if (strict_ordering_1p2e3 and full_lru_preserves_all_crossings)
                else "STOP_LRU_INSUFFICIENT"
            ),
            "scope_limit": SCOPE_LIMIT_STATEMENT,
        },
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
        output_path.write_text(payload, encoding="utf-8", newline="\n")

    return report


def main() -> None:
    rep = run_card_c_qutrit_leakage_lru_gate(output_path=DEFAULT_OUTPUT_PATH)
    sha = hashlib.sha256(DEFAULT_OUTPUT_PATH.read_bytes()).hexdigest()
    print(f"Wrote {DEFAULT_OUTPUT_PATH} (sha256={sha[:16]}...)")
    for row in rep["summary_ablation_table"]:
        print(
            f"  {row['target_profile_id']:17s} | p_leak={row['p_leak_2q']:.5f} | "
            f"{row['architecture']:25s} | eps_slope={row['eps_L_slope_fit']:.6f} | "
            f"d_eps={row['delta_eps_L_slope_vs_zero_leak']:+.6f} | "
            f"R_cross={str(row['r_cross_vs_both']):4s} | {row['verdict']}"
        )


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
