"""Analytical Kraus channel & Landsman-Figgatt discriminator for OSIRIS-XTOP-006.

Implements the exact 7-qubit Bell-postselected Hayden-Preskill / Yoshida-Kitaev recovery channel:
  Given any 3-qubit unitary U (8x8) acting on (q0, q1, q2) and its complex conjugate U* acting on
  (q3, q5, q4), with initial EPR pairs (q1, q5), (q2, q4), (q3, q6), projecting (q2, q4) onto
  Bell state |Phi+> yields 16 remnant 2x2 Kraus operators indexed by r = (i, j, m, n) in {0,1}^4:

      (M_r)_{x a} = (1 / 4) * sum_{b, c, k in {0, 1}} U_{i j k, a b c} * conj(U_{n m k, x b c})

  and Bob's unnormalized output state on q6 for input density matrix rho_in on q0:

      E_{Phi+}(rho_in) = sum_{r in {0,1}^4} M_r @ rho_in @ M_r^dagger,
      P_{Phi+} = Tr(E_{Phi+}(rho_in)),   rho_{out | Phi+} = E_{Phi+}(rho_in) / P_{Phi+}.

Across all four Bell outcomes beta in {Phi+, Phi-, Psi+, Psi-}, the 64 Kraus operators satisfy
exact CPTP completeness:
      sum_{beta} sum_{r=0}^{15} M_{beta, r}^dagger @ M_{beta, r} = I_2.

References:
  * B. Yoshida & A. Kitaev, arXiv:1710.03363 (2017).
  * K. A. Landsman, C. Figgatt, T. Schuster, N. M. Linke, B. Yoshida, N. Y. Yao, C. Monroe,
    "Verified quantum information scrambling", Nature 567, 61-65 (2019).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

I2 = np.eye(2, dtype=np.complex128)
PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
HADAMARD = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)
PAULIS_1Q: Tuple[np.ndarray, ...] = (I2, PAULI_X, PAULI_Y, PAULI_Z)

BELL_OUTCOMES: Tuple[str, ...] = ("Phi+", "Phi-", "Psi+", "Psi-")

# Bell-basis coefficient matrices B^{(beta)}_{k1, k2} such that |beta> = sum_{k1,k2} B_{k1,k2} |k1, k2>:
# Under CNOT(q2 -> q4) followed by H(q2), these map to |00>, |10>, |01>, |11> respectively.
BELL_MATRICES: Dict[str, np.ndarray] = {
    "Phi+": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128) / math.sqrt(2.0),
    "Phi-": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0),
    "Psi+": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128) / math.sqrt(2.0),
    "Psi-": np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.complex128) / math.sqrt(2.0),
}

THETA_GRID_DEFAULT: Tuple[float, ...] = (
    0.0,
    math.pi / 8.0,
    math.pi / 4.0,
    3.0 * math.pi / 8.0,
    math.pi / 2.0,
)

# 6 Pauli eigenstates: label -> (statevector, Pauli basis, ideal eigenvalue)
INPUT_STATES_6: Dict[str, Tuple[np.ndarray, str, int]] = {
    "0": (np.array([1.0, 0.0], dtype=np.complex128), "Z", +1),
    "1": (np.array([0.0, 1.0], dtype=np.complex128), "Z", -1),
    "+": (np.array([1.0, 1.0], dtype=np.complex128) / math.sqrt(2.0), "X", +1),
    "-": (np.array([1.0, -1.0], dtype=np.complex128) / math.sqrt(2.0), "X", -1),
    "+i": (np.array([1.0, 1.0j], dtype=np.complex128) / math.sqrt(2.0), "Y", +1),
    "-i": (np.array([1.0, -1.0j], dtype=np.complex128) / math.sqrt(2.0), "Y", -1),
}


def _kron_all(mats: Sequence[np.ndarray]) -> np.ndarray:
    out = np.array([[1.0]], dtype=np.complex128)
    for m in mats:
        out = np.kron(out, m)
    return out


def _rz_1q(phi: float) -> np.ndarray:
    return np.array(
        [[np.exp(-0.5j * phi), 0.0], [0.0, np.exp(0.5j * phi)]],
        dtype=np.complex128,
    )


def _tunable_cx_3q(theta: float, ctrl: int, tgt: int) -> np.ndarray:
    """Tunable 2-qubit entangler on 3 qubits:
    (I_ctrl x H_tgt) @ diag(1, 1, 1, exp(i * 2 * theta)) @ (I_ctrl x H_tgt).
    - At theta = 0:     exact 3-qubit identity I_8.
    - At theta = pi/2:  exact CNOT(ctrl -> tgt).
    - Native H2-1 decomposition: 1 ZZPhase(-theta/pi) + 2 virtual Rz(theta/pi) + 2 Hadamards.
    """
    z_ops = [I2, I2, I2]
    z_ops[ctrl] = PAULI_Z
    z_ops[tgt] = PAULI_Z
    zz = _kron_all(z_ops)

    rz_ops = [I2, I2, I2]
    rz_ops[ctrl] = _rz_1q(theta)
    rz_ops[tgt] = _rz_1q(theta)
    rz_layer = _kron_all(rz_ops)

    cphase = np.exp(0.5j * theta) * (
        rz_layer @ (math.cos(0.5 * theta) * np.eye(8, dtype=np.complex128) + 1.0j * math.sin(0.5 * theta) * zz)
    )
    h_ops = [I2, I2, I2]
    h_ops[tgt] = HADAMARD
    h_layer = _kron_all(h_ops)
    return h_layer @ cphase @ h_layer


def landsman_figgatt_scrambler(theta: float) -> np.ndarray:
    """Explicit 1-parameter tunable non-Clifford 3-qubit scrambler U(theta) in U(8).

    Properties across theta in [0, pi/2]:
      * theta = 0:
          U(0) = I_8 (zero scrambling; M_r(0) not proportional to I_2; P_{Phi+} = 1.0, F = 0.5).
      * theta in (0, pi/2) (e.g., pi/8, pi/4, 3*pi/8):
          Non-Clifford continuous-angle ZZPhase unitary with partial operator spreading;
          exact analytical P_{Phi+}(theta) = (1 - 0.5 * sin^2(theta))^2 and
          6-state fidelity F_avg(theta | Phi+) = 1/3 + 1 / (6 * P_{Phi+}(theta)).
      * theta = pi/2:
          Maximal 3-qubit scrambler (maps X_0 -> X_0 X_1 X_2, Z_0 -> Z_0 Z_1 Z_2, Y_0 -> -Y_0 Y_1 Y_2);
          every remnant Kraus matrix satisfies M_r(pi/2) = c_r * I_2, P_{Phi+}(pi/2) = 1/4, F = 1.0.
    """
    h3 = _kron_all((HADAMARD, HADAMARD, HADAMARD))
    cx01 = _tunable_cx_3q(theta, 0, 1)
    cx02 = _tunable_cx_3q(theta, 0, 2)
    return cx02 @ cx01 @ h3 @ cx02 @ cx01 @ h3


def analytical_p_phi_plus(theta: float) -> float:
    """Exact closed-form Bell |Phi+> postselection probability for U(theta):
    P_{Phi+}(theta) = (1 - 0.5 * sin^2(theta))^2.
    """
    s2 = math.sin(theta) ** 2
    return (1.0 - 0.5 * s2) ** 2


def analytical_fidelity_6state(theta: float) -> float:
    """Exact closed-form 6-state average recovery fidelity for U(theta) given |Phi+>:
    Saturates the Yoshida-Kitaev unitary bound P_{Phi+} * F_ent = 1/4,
    with F_6state = (2 * F_ent + 1) / 3 = 1/3 + 1 / (6 * P_{Phi+}(theta)).
    """
    p_phi = analytical_p_phi_plus(theta)
    return (1.0 / 3.0) + 1.0 / (6.0 * p_phi)


def is_non_clifford_unitary(unitary: np.ndarray, tol: float = 1e-6) -> Tuple[bool, float]:
    """Check whether an 8x8 unitary is non-Clifford by testing Pauli-group non-closure.

    For any Clifford U and 3-qubit Pauli P != I, U P U^dagger is a single signed 3-qubit Pauli
    (so sum_{Q in Pauli_3} |Tr(Q U P U^dagger) / 8|^4 == 1). Non-Clifford unitaries spread at least
    one generator into a superposition of multiple Paulis, yielding inverse participation ratio < 1
    and stabilizer Renyi-2 magic witness M_2 = -log2(min_ipr) > 0.
    Returns `(is_non_clifford, magic_m2_bits)`.
    """
    u = np.asarray(unitary, dtype=np.complex128)
    if u.shape != (8, 8):
        raise ValueError(f"Expected (8, 8) unitary, got {u.shape}")

    paulis_3q = [
        _kron_all((a, b, c))
        for a in PAULIS_1Q
        for b in PAULIS_1Q
        for c in PAULIS_1Q
    ]
    generators = [
        _kron_all((PAULI_X, I2, I2)),
        _kron_all((PAULI_Z, I2, I2)),
        _kron_all((I2, PAULI_X, I2)),
        _kron_all((I2, PAULI_Z, I2)),
        _kron_all((I2, I2, PAULI_X)),
        _kron_all((I2, I2, PAULI_Z)),
    ]
    min_ipr = 1.0
    for g in generators:
        heis = u @ g @ u.conj().T
        weights_sq = [abs(np.trace(q @ heis) / 8.0) ** 2 for q in paulis_3q]
        ipr = float(sum(w * w for w in weights_sq))
        min_ipr = min(min_ipr, ipr)
    min_ipr_clamped = max(min(min_ipr, 1.0), 1e-16)
    magic_m2 = -math.log2(min_ipr_clamped) if min_ipr_clamped < 1.0 - tol else 0.0
    return (min_ipr < 1.0 - tol), float(magic_m2)


def compute_remnant_kraus_operators(
    unitary: np.ndarray,
    bell_outcome: str = "Phi+",
) -> np.ndarray:
    """Compute the 16 remnant 2x2 Kraus matrices M_{beta, r} for r = (i, j, m, n) in {0,1}^4.

    For `bell_outcome == "Phi+"`, implements the exact contraction:
        (M_r)_{x a} = (1 / 4) * sum_{b, c, k} U_{i j k, a b c} * conj(U_{n m k, x b c}).

    For general `bell_outcome in {"Phi+", "Phi-", "Psi+", "Psi-"}` with Bell matrix B_{k1, k2}:
        (M_{beta, r})_{x a} = (1 / (2 * sqrt(2))) * sum_{b, c, k1, k2}
                              conj(B_{k1, k2}) * U_{i j k1, a b c} * conj(U_{n m k2, x b c}).

    Returns an array of shape `(16, 2, 2)` ordered lexicographically by `r = 8*i + 4*j + 2*m + n`.
    """
    u = np.asarray(unitary, dtype=np.complex128)
    if u.shape != (8, 8):
        raise ValueError(f"Expected 3-qubit unitary of shape (8, 8), got {u.shape}")
    if bell_outcome not in BELL_MATRICES:
        raise ValueError(f"Unsupported Bell outcome {bell_outcome!r}; expected one of {BELL_OUTCOMES}")

    t = u.reshape(2, 2, 2, 2, 2, 2)  # (i, j, k, a, b, c)
    if bell_outcome == "Phi+":
        m_tensor = 0.25 * np.einsum("ijkabc,nmkxbc->ijmnxa", t, np.conj(t))
    else:
        b_mat = BELL_MATRICES[bell_outcome]
        pref = 1.0 / (2.0 * math.sqrt(2.0))
        m_tensor = pref * np.einsum(
            "kl,ijkabc,nmlxbc->ijmnxa",
            np.conj(b_mat),
            t,
            np.conj(t),
        )
    return m_tensor.reshape(16, 2, 2)


def compute_all_bell_kraus_channels(unitary: np.ndarray) -> Dict[str, np.ndarray]:
    """Return `{bell_outcome: M_array_(16,2,2)}` across all four Bell outcomes."""
    return {beta: compute_remnant_kraus_operators(unitary, beta) for beta in BELL_OUTCOMES}


def verify_cptp_completeness(unitary: np.ndarray, atol: float = 1e-12) -> Dict[str, Any]:
    """Verify trace-preserving completeness sum_{beta, r} M_{beta, r}^dagger M_{beta, r} == I_2
    and complete positivity of every Bell sub-channel's Choi matrix.
    """
    channels = compute_all_bell_kraus_channels(unitary)
    completeness_sum = np.zeros((2, 2), dtype=np.complex128)
    per_outcome_povm: Dict[str, np.ndarray] = {}
    choi_min_eig: Dict[str, float] = {}

    # Unnormalized Choi matrix J(E_beta) = sum_{a, b in {0,1}} |a><b| x E_beta(|a><b|)
    for beta, kraus_arr in channels.items():
        povm = sum((m.conj().T @ m for m in kraus_arr), np.zeros((2, 2), dtype=np.complex128))
        per_outcome_povm[beta] = povm
        completeness_sum = completeness_sum + povm

        choi = np.zeros((4, 4), dtype=np.complex128)
        for a in range(2):
            for b in range(2):
                eab = np.zeros((2, 2), dtype=np.complex128)
                eab[a, b] = 1.0
                out_ab = sum((m @ eab @ m.conj().T for m in kraus_arr), np.zeros((2, 2), dtype=np.complex128))
                choi = choi + np.kron(eab, out_ab)
        eigvals = np.linalg.eigvalsh(0.5 * (choi + choi.conj().T))
        choi_min_eig[beta] = float(np.min(eigvals))

    frob_err = float(np.linalg.norm(completeness_sum - I2, ord="fro"))
    cp_ok = all(v >= -atol for v in choi_min_eig.values())
    tp_ok = frob_err <= atol
    return {
        "cptp_valid": bool(cp_ok and tp_ok),
        "completeness_frobenius_error": frob_err,
        "completeness_matrix": completeness_sum,
        "per_outcome_povm": per_outcome_povm,
        "choi_min_eigenvalues": choi_min_eig,
    }


def decoupling_deviation(kraus_ops: np.ndarray) -> float:
    """Return max_{r in 0..15} ||M_r - (Tr(M_r)/2) I_2||_F.
    Vanishes (< 1e-14) iff every M_r = c_r I_2 (exact maximal scrambling recovery without correction).
    """
    return float(
        max(np.linalg.norm(m - 0.5 * np.trace(m) * I2, ord="fro") for m in kraus_ops)
    )


@dataclass(frozen=True)
class ChannelEvaluation:
    bell_outcome: str
    unnormalized_rho_out: np.ndarray
    success_probability: float
    normalized_rho_out: np.ndarray
    fidelity_with_input: float


def apply_bell_channel(
    unitary: np.ndarray,
    rho_in: np.ndarray,
    bell_outcome: str = "Phi+",
) -> ChannelEvaluation:
    """Evaluate E_beta(rho_in) = sum_r M_{beta, r} rho_in M_{beta, r}^dagger, P_beta, and state fidelity."""
    rho = np.asarray(rho_in, dtype=np.complex128)
    if rho.ndim == 1:
        rho = np.outer(rho, rho.conj())
    if rho.shape != (2, 2):
        raise ValueError(f"Expected (2, 2) input density matrix or length-2 statevector, got {rho.shape}")

    kraus_ops = compute_remnant_kraus_operators(unitary, bell_outcome=bell_outcome)
    rho_tilde = sum((m @ rho @ m.conj().T for m in kraus_ops), np.zeros((2, 2), dtype=np.complex128))
    prob = float(np.real(np.trace(rho_tilde)))
    if prob <= 1e-15:
        rho_norm = 0.5 * I2
        fid = 0.5
    else:
        rho_norm = rho_tilde / prob
        # Uhlmann fidelity specialized to pure or general 1-qubit states:
        # For pure rho_in, Tr(rho_in @ rho_norm) is exact; for mixed states use general 2x2 formula
        purity_in = float(np.real(np.trace(rho @ rho)))
        if abs(purity_in - 1.0) < 1e-10:
            fid = float(np.real(np.trace(rho @ rho_norm)))
        else:
            det_prod = max(0.0, float(np.real(np.linalg.det(rho) * np.linalg.det(rho_norm))))
            fid = float(np.real(np.trace(rho @ rho_norm)) + 2.0 * math.sqrt(det_prod))
    return ChannelEvaluation(
        bell_outcome=bell_outcome,
        unnormalized_rho_out=rho_tilde,
        success_probability=prob,
        normalized_rho_out=rho_norm,
        fidelity_with_input=fid,
    )


def evaluate_6state_summary(unitary: np.ndarray, bell_outcome: str = "Phi+") -> Dict[str, Any]:
    """Evaluate P_beta, 6-state average fidelity, entanglement fidelity, and decoupling error."""
    kraus_ops = compute_remnant_kraus_operators(unitary, bell_outcome=bell_outcome)
    per_state: Dict[str, Dict[str, float]] = {}
    probs: List[float] = []
    fids: List[float] = []
    for label, (psi, _, _) in INPUT_STATES_6.items():
        ev = apply_bell_channel(unitary, psi, bell_outcome=bell_outcome)
        per_state[label] = {
            "success_probability": ev.success_probability,
            "fidelity": ev.fidelity_with_input,
        }
        probs.append(ev.success_probability)
        fids.append(ev.fidelity_with_input)

    p_avg = float(np.mean(probs))
    f_avg = float(np.mean(fids))
    # For d=2 qubits, F_avg = (2 * F_ent + 1) / 3  =>  F_ent = (3 * F_avg - 1) / 2
    f_ent = 0.5 * (3.0 * f_avg - 1.0)
    non_cliff, magic_m2 = is_non_clifford_unitary(unitary)
    return {
        "bell_outcome": bell_outcome,
        "p_success": p_avg,
        "fidelity_6state_avg": f_avg,
        "entanglement_fidelity": f_ent,
        "yk_product_p_times_f_ent": p_avg * f_ent,
        "verified_scrambling_witness": f_avg - 0.5,
        "max_decoupling_deviation": decoupling_deviation(kraus_ops),
        "is_non_clifford": non_cliff,
        "stabilizer_renyi2_magic_bits": magic_m2,
        "per_state": per_state,
    }


# ---------------------------------------------------------------------------
# Exact 7-Qubit Density-Matrix Scrambling-vs-Decoherence Discriminator
# ---------------------------------------------------------------------------


def _apply_1q_pauli_noise_7q(
    rho7: np.ndarray,
    qubit: int,
    px: float,
    py: float,
    pz: float,
) -> np.ndarray:
    """Apply single-qubit Pauli channel `(1 - px - py - pz) I + px X + py Y + pz Z` to `qubit` of a 7Q density matrix."""
    p0 = 1.0 - px - py - pz
    if abs(1.0 - p0) < 1e-15:
        return rho7
    out = p0 * rho7
    for prob, pauli in ((px, PAULI_X), (py, PAULI_Y), (pz, PAULI_Z)):
        if prob <= 0.0:
            continue
        ops = [I2] * 7
        ops[qubit] = pauli
        g = _kron_all(ops)
        out = out + prob * (g @ rho7 @ g.conj().T)
    return out


def _embed_3q_unitary_in_7q(u3: np.ndarray, qubits: Tuple[int, int, int]) -> np.ndarray:
    """Embed an 8x8 unitary acting on `qubits = (qa, qb, qc)` into the 7-qubit (128x128) Hilbert space."""
    # Build index map directly on 7-qubit basis b0..b6
    u7 = np.zeros((128, 128), dtype=np.complex128)
    qa, qb, qc = qubits
    other = [q for q in range(7) if q not in qubits]
    for col in range(128):
        c_bits = [(col >> (6 - q)) & 1 for q in range(7)]
        in3 = (c_bits[qa] << 2) | (c_bits[qb] << 1) | c_bits[qc]
        for out3 in range(8):
            amp = u3[out3, in3]
            if abs(amp) < 1e-16:
                continue
            r_bits = list(c_bits)
            r_bits[qa] = (out3 >> 2) & 1
            r_bits[qb] = (out3 >> 1) & 1
            r_bits[qc] = out3 & 1
            # other bits unchanged
            for oq in other:
                assert r_bits[oq] == c_bits[oq]
            row = 0
            for q in range(7):
                row = (row << 1) | r_bits[q]
            u7[row, col] = amp
    return u7


def _simulate_7q_density_matrix_point(
    theta: float,
    depol_rate: float = 0.0,
    dephase_rate: float = 0.0,
) -> Dict[str, float]:
    """Exact 7-qubit density-matrix evaluation under U(theta) on (0,1,2) and U*(theta) on (3,5,4)
    followed by local depolarizing (`depol_rate`) and dephasing (`dephase_rate`) on active qubits (0..5).
    """
    u3 = landsman_figgatt_scrambler(theta)
    u7_fwd = _embed_3q_unitary_in_7q(u3, (0, 1, 2))
    u7_conj = _embed_3q_unitary_in_7q(np.conj(u3), (3, 5, 4))
    u7_tot = u7_conj @ u7_fwd

    # Build 7-qubit state for each input |psi> on q0
    px = depol_rate / 3.0
    py = depol_rate / 3.0
    pz = depol_rate / 3.0 + dephase_rate * (1.0 - depol_rate)

    # Projector onto |Phi+>_{2,4} (shape 128 x 32 mapping (i,j,m,n,x) -> 7Q index)
    proj_phi_plus = np.zeros((32, 128), dtype=np.complex128)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    # 5 remaining qubits order: (0, 1, 5, 3, 6)
    for r5 in range(32):
        b0 = (r5 >> 4) & 1
        b1 = (r5 >> 3) & 1
        b5 = (r5 >> 2) & 1
        b3 = (r5 >> 1) & 1
        b6 = r5 & 1
        for k in (0, 1):
            bits7 = [b0, b1, k, b3, k, b5, b6]
            idx7 = 0
            for bit in bits7:
                idx7 = (idx7 << 1) | bit
            proj_phi_plus[r5, idx7] = inv_sqrt2

    probs: List[float] = []
    fids: List[float] = []
    for psi, _, _ in INPUT_STATES_6.values():
        # Construct 7Q initial statevector: psi_0 * Phi+_{1,5} * Phi+_{2,4} * Phi+_{3,6}
        sv7 = np.zeros(128, dtype=np.complex128)
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    for x in (0, 1):
                        bits7 = [a, b, c, x, c, b, x]
                        idx7 = 0
                        for bit in bits7:
                            idx7 = (idx7 << 1) | bit
                        sv7[idx7] = psi[a] * (inv_sqrt2 ** 3)
        sv_after = u7_tot @ sv7
        rho7 = np.outer(sv_after, sv_after.conj())
        for q_act in range(6):
            rho7 = _apply_1q_pauli_noise_7q(rho7, q_act, px, py, pz)

        # Post-select (q2, q4) onto |Phi+> -> 32x32 density matrix on (q0, q1, q5, q3, q6)
        rho5 = proj_phi_plus @ rho7 @ proj_phi_plus.conj().T
        # Trace out 4 remnant qubits (first 4 qubits of rho5 -> 16 dim) leaving q6 (2x2)
        rho5_t = rho5.reshape(16, 2, 16, 2)
        rho_out_tilde = np.einsum("rxry->xy", rho5_t)
        p_s = float(np.real(np.trace(rho_out_tilde)))
        rho_out = rho_out_tilde / p_s
        fid = float(np.real(psi.conj() @ rho_out @ psi))
        probs.append(p_s)
        fids.append(fid)

    p_avg = float(np.mean(probs))
    f_avg = float(np.mean(fids))
    f_ent = 0.5 * (3.0 * f_avg - 1.0)
    return {
        "theta_rad": float(theta),
        "depol_rate": float(depol_rate),
        "dephase_rate": float(dephase_rate),
        "p_phi_plus": p_avg,
        "fidelity_6state": f_avg,
        "entanglement_fidelity": f_ent,
        "yk_product_p_times_f_ent": p_avg * f_ent,
        "verified_scrambling_witness": f_avg - 0.5,
    }


def scrambling_vs_decoherence_discriminator(
    thetas: Sequence[float] = THETA_GRID_DEFAULT,
    noisy_depol_rate: float = 0.015,
    noisy_dephase_rate: float = 0.010,
    decoherence_rates: Sequence[float] = (0.0, 0.08, 0.20, 0.40, 0.75),
) -> Dict[str, Any]:
    """Compute exact Landsman-Figgatt Scrambling-vs-Decoherence Discriminator curves:
      (i)   Noiseless coherent tunable scrambling (`p = 0`, sweeping `theta`):
            P_{Phi+} drops 1.0 -> 0.25 while F_{6state | Phi+} rises 0.50 -> 1.00,
            identically saturating Yoshida-Kitaev `P_{Phi+} * F_ent = 0.25`.
      (ii)  Coherent tunable scrambling + depolarizing/dephasing noise (`p > 0`, sweeping `theta`):
            P_{Phi+} drops toward 0.25 while F_{6state | Phi+} rises strictly above the 0.50
            decoherence ceiling (`verified_scrambling_witness = F - 0.5 > 0` for all theta > 0).
      (iii) Pure decoherence without scrambling (`theta = 0`, `U = I_8`, sweeping `p`):
            Decoherence drives P_{Phi+} down from 1.0 -> 0.25 (mimicking OTOC decay),
            yet F_{6state | Phi+} remains identically pinned at 0.500000 (`witness == 0.0`).
    """
    coherent_noiseless = [
        _simulate_7q_density_matrix_point(th, depol_rate=0.0, dephase_rate=0.0)
        for th in thetas
    ]
    coherent_noisy = [
        _simulate_7q_density_matrix_point(
            th,
            depol_rate=noisy_depol_rate,
            dephase_rate=noisy_dephase_rate,
        )
        for th in thetas
    ]
    pure_decoherence = [
        _simulate_7q_density_matrix_point(0.0, depol_rate=p_dec, dephase_rate=0.0)
        for p_dec in decoherence_rates
    ]
    max_decoherence_fidelity = max(pt["fidelity_6state"] for pt in pure_decoherence)
    max_noisy_scrambling_fidelity = max(pt["fidelity_6state"] for pt in coherent_noisy)
    return {
        "coherent_noiseless": coherent_noiseless,
        "coherent_noisy": coherent_noisy,
        "pure_decoherence_no_scrambling": pure_decoherence,
        "separation_summary": {
            "pure_decoherence_max_fidelity": max_decoherence_fidelity,
            "coherent_noisy_peak_fidelity": max_noisy_scrambling_fidelity,
            "fidelity_gap_over_decoherence_ceiling": max_noisy_scrambling_fidelity - max_decoherence_fidelity,
            "separates_scrambling_from_decoherence": bool(
                max_decoherence_fidelity <= 0.5 + 1e-12
                and max_noisy_scrambling_fidelity > 0.85
            ),
        },
    }
