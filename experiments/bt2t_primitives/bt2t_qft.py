"""OSIRIS-BT2T-002 Milestone 2 (Part 2): exact non-abelian Quantum Fourier Transform over 2T.

Constructs all 7 irreducible unitary representations (irreps) of the binary tetrahedral group 2T
(dimensions 1, 1, 1, 2, 2, 2, 3; sum d_alpha^2 = 24) from first principles on frozen encoding
``bt2t-enc-1``, assembles the exact 24x24 Peter-Weyl Fourier matrix F_{2T} and its 5-qubit 32x32
unitary extension F_tilde_{2T} = F_{2T} (+) I_8, proves Schur orthogonality, Plancherel & convolution
block-diagonalization across all 24 group translations, verifies Hidden Subgroup coset-state Fourier
support across 6 canonical subgroups, and synthesizes a 5-qubit Qiskit QuantumCircuit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

if __package__:
    from . import bt2t_mul
else:
    import bt2t_mul  # type: ignore[no-redef]

N_QFT_QUBITS = 5
N_QFT_STATES = 32
N_GROUP_ORDER = 24

IRREP_NAMES: Tuple[str, ...] = (
    "chi_0",
    "chi_1",
    "chi_2",
    "rho_2_0",
    "rho_2_1",
    "rho_2_2",
    "rho_3",
)
IRREP_DIMS: Dict[str, int] = {
    "chi_0": 1,
    "chi_1": 1,
    "chi_2": 1,
    "rho_2_0": 2,
    "rho_2_1": 2,
    "rho_2_2": 2,
    "rho_3": 3,
}

PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
PAULIS = (PAULI_X, PAULI_Y, PAULI_Z)
ZETA_3 = complex(np.exp(2.0j * np.pi / 3.0))


@dataclass(frozen=True)
class PeterWeylRow:
    """Metadata for one row of the 24x24 Peter-Weyl matrix F_{2T}."""

    row_index: int
    irrep: str
    dim: int
    j: int
    k: int


def coset_z3(g: int) -> int:
    """Quotient homomorphism c: 2T -> 2T / Q_8 ~= Z_3 = {0, 1, 2}."""
    return bt2t_mul.coset_index_z3(g)


def su2_matrix(g: int) -> np.ndarray:
    """Faithful 2x2 SU(2) representation of element g in 2T from doubled quaternion coords (a, b, c, d):
      1 -> I_2,  i -> -i sigma_x,  j -> -i sigma_y,  k -> -i sigma_z.
    """
    if not (0 <= g < N_GROUP_ORDER):
        raise ValueError(f"Element index {g} outside 0..23")
    a, b, c, d = bt2t_mul.bt2t_base.elements()[g]
    return 0.5 * np.array(
        [[a - 1.0j * d, -c - 1.0j * b], [c - 1.0j * b, a + 1.0j * d]],
        dtype=complex,
    )


def irrep_chi(r: int, g: int) -> np.ndarray:
    """1D unitary irrep chi_r(g) = [zeta_3^{r * c(g)}] for r in {0, 1, 2}."""
    if r not in (0, 1, 2):
        raise ValueError(f"1D irrep index r={r} must be in {{0, 1, 2}}")
    exp = (r * coset_z3(g)) % 3
    val = 1.0 + 0.0j if exp == 0 else (ZETA_3 if exp == 1 else ZETA_3 * ZETA_3)
    return np.array([[val]], dtype=complex)


def irrep_rho2(r: int, g: int) -> np.ndarray:
    """2D spinor unitary irrep rho_{2,r}(g) = chi_r(g) * U_{SU(2)}(g) for r in {0, 1, 2}."""
    if r not in (0, 1, 2):
        raise ValueError(f"2D spinor irrep index r={r} must be in {{0, 1, 2}}")
    return irrep_chi(r, g)[0, 0] * su2_matrix(g)


def irrep_rho3(g: int) -> np.ndarray:
    """3D tetrahedral SO(3) unitary irrep via adjoint action on traceless Pauli matrices:
      U(g) sigma_a U(g)^dagger = sum_{b=1}^3 [rho_3(g)]_{b, a} sigma_b.
    """
    u = su2_matrix(g)
    ud = u.conj().T
    mat = np.zeros((3, 3), dtype=complex)
    for a_idx in range(3):
        conjugated = u @ PAULIS[a_idx] @ ud
        for b_idx in range(3):
            mat[b_idx, a_idx] = 0.5 * np.trace(PAULIS[b_idx] @ conjugated)
    return mat


def irreps_2t() -> Dict[str, List[np.ndarray]]:
    """Return all 7 irreps of 2T, each as a list of 24 unitary matrices indexed by g in 0..23."""
    return {
        "chi_0": [irrep_chi(0, g) for g in range(N_GROUP_ORDER)],
        "chi_1": [irrep_chi(1, g) for g in range(N_GROUP_ORDER)],
        "chi_2": [irrep_chi(2, g) for g in range(N_GROUP_ORDER)],
        "rho_2_0": [irrep_rho2(0, g) for g in range(N_GROUP_ORDER)],
        "rho_2_1": [irrep_rho2(1, g) for g in range(N_GROUP_ORDER)],
        "rho_2_2": [irrep_rho2(2, g) for g in range(N_GROUP_ORDER)],
        "rho_3": [irrep_rho3(g) for g in range(N_GROUP_ORDER)],
    }


def character_table_2t() -> np.ndarray:
    """7x24 complex array of character values chi_alpha(g) = Tr(rho_alpha(g))."""
    all_irreps = irreps_2t()
    chars = np.zeros((len(IRREP_NAMES), N_GROUP_ORDER), dtype=complex)
    for alpha_idx, name in enumerate(IRREP_NAMES):
        for g in range(N_GROUP_ORDER):
            chars[alpha_idx, g] = np.trace(all_irreps[name][g])
    return chars


def peter_weyl_row_layout() -> List[PeterWeylRow]:
    """Canonical ordering of the 24 Peter-Weyl basis states (alpha, j, k).

    Ordered by irrep alpha in IRREP_NAMES, multiplicity index k in 0..d_alpha-1 (outer),
    and representation row index j in 0..d_alpha-1 (inner), so that left-regular translations
    F_{2T} L_g F_{2T}^dagger = (+)_{alpha} (I_{d_alpha} (x) rho_alpha(g)) form 12 contiguous
    diagonal blocks of sizes (1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3).
    """
    rows: List[PeterWeylRow] = []
    idx = 0
    for name in IRREP_NAMES:
        d = IRREP_DIMS[name]
        for k in range(d):
            for j in range(d):
                rows.append(PeterWeylRow(row_index=idx, irrep=name, dim=d, j=j, k=k))
                idx += 1
    return rows


def irrep_row_slices() -> Dict[str, slice]:
    """Contiguous row slice in 0..23 occupied by each of the 7 irreps (size d_alpha^2)."""
    out: Dict[str, slice] = {}
    start = 0
    for name in IRREP_NAMES:
        span = IRREP_DIMS[name] ** 2
        out[name] = slice(start, start + span)
        start += span
    return out


def peter_weyl_matrix_24() -> np.ndarray:
    """Exact 24x24 Peter-Weyl Non-Abelian Fourier Matrix F_{2T}:
      [F_{2T}]_{(alpha, j, k), g} = sqrt(d_alpha / 24) * [rho_alpha(g)]_{j, k}.
    """
    all_irreps = irreps_2t()
    layout = peter_weyl_row_layout()
    f24 = np.zeros((N_GROUP_ORDER, N_GROUP_ORDER), dtype=complex)
    for r in layout:
        scale = np.sqrt(r.dim / float(N_GROUP_ORDER))
        mats = all_irreps[r.irrep]
        for g in range(N_GROUP_ORDER):
            f24[r.row_index, g] = scale * mats[g][r.j, r.k]
    return f24


def bt2t_qft_matrix_32() -> np.ndarray:
    """5-qubit 32x32 non-abelian QFT unitary F_tilde_{2T} = F_{2T} (+) I_8 (fixing unused 24..31)."""
    f32 = np.eye(N_QFT_STATES, dtype=complex)
    f32[:N_GROUP_ORDER, :N_GROUP_ORDER] = peter_weyl_matrix_24()
    return f32


def left_regular_matrix_24(g: int) -> np.ndarray:
    """24x24 permutation matrix L_g with L_g |x> = |g * x>."""
    tbl = bt2t_mul.mul_table_24()
    mat = np.zeros((N_GROUP_ORDER, N_GROUP_ORDER), dtype=complex)
    for x in range(N_GROUP_ORDER):
        mat[tbl[g][x], x] = 1.0
    return mat


def right_regular_matrix_24(g: int) -> np.ndarray:
    """24x24 permutation matrix R_g with R_g |x> = |x * g^{-1}> (left action R_{gh} = R_g R_h)."""
    tbl = bt2t_mul.mul_table_24()
    inv = bt2t_mul.bt2t_base.inverse_from_table(tbl)
    g_inv = inv[g]
    mat = np.zeros((N_GROUP_ORDER, N_GROUP_ORDER), dtype=complex)
    for x in range(N_GROUP_ORDER):
        mat[tbl[x][g_inv], x] = 1.0
    return mat


# ---------------------------------------------------------------------------
# Programmatic Mathematical Proofs & Verification Suites
# ---------------------------------------------------------------------------


def verify_irreps_and_schur_orthogonality() -> Dict[str, object]:
    """Verify 7-irrep unitarity, 576-pair homomorphism, Schur orthogonality, and F_tilde_{2T} unitarity."""
    tbl = bt2t_mul.mul_table_24()
    all_irreps = irreps_2t()

    sum_d_sq = sum(d * d for d in IRREP_DIMS.values())
    hom_errors: Dict[str, float] = {}
    unitary_errors: Dict[str, float] = {}
    for name in IRREP_NAMES:
        d = IRREP_DIMS[name]
        mats = all_irreps[name]
        u_err = max(
            float(np.linalg.norm(mats[g] @ mats[g].conj().T - np.eye(d), ord=np.inf))
            for g in range(N_GROUP_ORDER)
        )
        h_err = max(
            float(np.max(np.abs(mats[tbl[g][h]] - (mats[g] @ mats[h]))))
            for g in range(N_GROUP_ORDER)
            for h in range(N_GROUP_ORDER)
        )
        unitary_errors[name] = u_err
        hom_errors[name] = h_err

    # Verify rho_3 is real orthogonal in SO(3) (det == +1, imag == 0):
    rho3_mats = all_irreps["rho_3"]
    rho3_imag_err = max(float(np.max(np.abs(np.imag(m)))) for m in rho3_mats)
    rho3_det_err = max(float(abs(np.linalg.det(np.real(m)) - 1.0)) for m in rho3_mats)

    # Character orthogonality: (1/24) * chars @ chars^dagger == I_7
    chars = character_table_2t()
    char_gram = (chars @ chars.conj().T) / float(N_GROUP_ORDER)
    char_orth_err = float(np.linalg.norm(char_gram - np.eye(len(IRREP_NAMES)), ord=np.inf))

    # Distinct character vectors across all 7 irreps:
    min_char_dist = min(
        float(np.linalg.norm(chars[a] - chars[b]))
        for a in range(len(IRREP_NAMES))
        for b in range(a + 1, len(IRREP_NAMES))
    )

    # Peter-Weyl 24x24 & 32x32 exact unitarity (equivalent to full Schur matrix-element orthogonality):
    f24 = peter_weyl_matrix_24()
    f32 = bt2t_qft_matrix_32()
    f24_unitarity_inf = float(
        np.linalg.norm(f24 @ f24.conj().T - np.eye(N_GROUP_ORDER), ord=np.inf)
    )
    f32_unitarity_inf = float(
        np.linalg.norm(f32 @ f32.conj().T - np.eye(N_QFT_STATES), ord=np.inf)
    )
    unused_identity_err = float(
        np.max(np.abs(f32[N_GROUP_ORDER:, N_GROUP_ORDER:] - np.eye(8)))
        + np.max(np.abs(f32[:N_GROUP_ORDER, N_GROUP_ORDER:]))
        + np.max(np.abs(f32[N_GROUP_ORDER:, :N_GROUP_ORDER]))
    )

    return {
        "sum_d_alpha_squared": sum_d_sq,
        "sum_d_alpha_squared_equals_24": sum_d_sq == N_GROUP_ORDER,
        "irrep_unitary_max_inf_errors": unitary_errors,
        "irrep_homomorphism_max_abs_errors": hom_errors,
        "all_irreps_unitary": max(unitary_errors.values()) < 1e-14,
        "all_irreps_homomorphisms_all_576_pairs": max(hom_errors.values()) < 1e-14,
        "rho3_so3_real_orthogonal": rho3_imag_err < 1e-14 and rho3_det_err < 1e-14,
        "character_orthogonality_inf_error": char_orth_err,
        "characters_orthonormal_and_distinct": char_orth_err < 1e-14 and min_char_dist > 1.0,
        "f24_unitarity_inf_error": f24_unitarity_inf,
        "f32_unitarity_inf_error": f32_unitarity_inf,
        "f32_strictly_unitary_below_1e14": f32_unitarity_inf < 1e-14,
        "unused_codewords_exact_identity": unused_identity_err == 0.0,
    }


def verify_simultaneous_block_diagonalization() -> Dict[str, object]:
    """Verify F_{2T} simultaneously block-diagonalizes all 24 L_g and R_g translations, plus Plancherel
    and the non-abelian group convolution theorem.
    """
    f24 = peter_weyl_matrix_24()
    f24_dag = f24.conj().T
    all_irreps = irreps_2t()
    slices = irrep_row_slices()

    # Contiguous diagonal blocks for L_g: 12 blocks (d_alpha copies of rho_alpha(g)):
    contiguous_blocks: List[Tuple[str, int, int]] = []
    pos = 0
    for name in IRREP_NAMES:
        d = IRREP_DIMS[name]
        for _ in range(d):
            contiguous_blocks.append((name, pos, pos + d))
            pos += d

    mask_left = np.zeros((N_GROUP_ORDER, N_GROUP_ORDER), dtype=bool)
    for _, r0, r1 in contiguous_blocks:
        mask_left[r0:r1, r0:r1] = True

    mask_irrep_sectors = np.zeros((N_GROUP_ORDER, N_GROUP_ORDER), dtype=bool)
    for sl in slices.values():
        mask_irrep_sectors[sl, sl] = True

    max_left_off_block = 0.0
    max_left_on_block_err = 0.0
    max_right_off_irrep = 0.0
    max_right_on_block_err = 0.0

    for g in range(N_GROUP_ORDER):
        lg = left_regular_matrix_24(g)
        conj_l = f24 @ lg @ f24_dag
        max_left_off_block = max(max_left_off_block, float(np.max(np.abs(conj_l[~mask_left]))))
        for name, r0, r1 in contiguous_blocks:
            err_blk = float(np.max(np.abs(conj_l[r0:r1, r0:r1] - all_irreps[name][g])))
            max_left_on_block_err = max(max_left_on_block_err, err_blk)

        # For R_g: F_{2T} R_g F_{2T}^dagger in (k outer, j inner) order equals conj(rho_alpha(g)) (x) I_{d_alpha}!
        rg = right_regular_matrix_24(g)
        conj_r = f24 @ rg @ f24_dag
        max_right_off_irrep = max(
            max_right_off_irrep, float(np.max(np.abs(conj_r[~mask_irrep_sectors])))
        )
        for name, sl in slices.items():
            d = IRREP_DIMS[name]
            expected_r = np.kron(np.conj(all_irreps[name][g]), np.eye(d))
            err_r = float(np.max(np.abs(conj_r[sl, sl] - expected_r)))
            max_right_on_block_err = max(max_right_on_block_err, err_r)

    # Plancherel & Non-Abelian Convolution Theorem check on seeded complex functions a, b: 2T -> C:
    rng = np.random.default_rng(20261124)
    a_vec = rng.normal(size=N_GROUP_ORDER) + 1j * rng.normal(size=N_GROUP_ORDER)
    b_vec = rng.normal(size=N_GROUP_ORDER) + 1j * rng.normal(size=N_GROUP_ORDER)
    a_vec /= np.linalg.norm(a_vec)
    b_vec /= np.linalg.norm(b_vec)
    plancherel_err = abs(float(np.linalg.norm(f24 @ a_vec) - np.linalg.norm(a_vec)))

    # Group convolution (a * b)(g) = sum_{h in 2T} a(h) b(h^{-1} g) = sum_h a(h) (L_h b)(g):
    conv_ab = sum(a_vec[h] * (left_regular_matrix_24(h) @ b_vec) for h in range(N_GROUP_ORDER))
    # Matrix-valued Fourier transform: hat{f}(alpha) = sum_{g in 2T} f(g) rho_alpha(g) satisfies
    # hat{a * b}(alpha) == hat{a}(alpha) @ hat{b}(alpha) for all 7 irreps!
    conv_err = 0.0
    for name in IRREP_NAMES:
        mats = all_irreps[name]
        hat_a = sum(a_vec[g] * mats[g] for g in range(N_GROUP_ORDER))
        hat_b = sum(b_vec[g] * mats[g] for g in range(N_GROUP_ORDER))
        hat_ab = sum(conv_ab[g] * mats[g] for g in range(N_GROUP_ORDER))
        conv_err = max(conv_err, float(np.max(np.abs(hat_ab - (hat_a @ hat_b)))))

    return {
        "block_sizes_contiguous": [r1 - r0 for _, r0, r1 in contiguous_blocks],
        "irrep_unique_dims": [IRREP_DIMS[n] for n in IRREP_NAMES],
        "left_regular_max_off_block_abs": max_left_off_block,
        "left_regular_max_on_block_error": max_left_on_block_err,
        "right_regular_max_off_irrep_abs": max_right_off_irrep,
        "right_regular_max_on_block_error": max_right_on_block_err,
        "all_24_left_and_right_block_diagonal_below_1e14": (
            max_left_off_block < 1e-14
            and max_left_on_block_err < 1e-14
            and max_right_off_irrep < 1e-14
            and max_right_on_block_err < 1e-14
        ),
        "plancherel_norm_error": plancherel_err,
        "nonabelian_convolution_max_error": conv_err,
        "plancherel_and_convolution_hold": plancherel_err < 1e-13 and conv_err < 1e-13,
    }


def canonical_subgroups_2t() -> Dict[str, List[int]]:
    """Six canonical subgroups H <= 2T of orders 1, 2, 4, 3, 8, and 24."""
    return {
        "trivial_1": [0],
        "Z2_center": [0, 1],
        "Z4_i": [0, 2, 1, 3],
        "Z3_omega": [0, bt2t_mul.OMEGA_INDEX, bt2t_mul.OMEGA_SQ_INDEX],
        "Q8_normal": list(range(8)),
        "full_2T": list(range(N_GROUP_ORDER)),
    }


def verify_subgroup_coset_sampling() -> Dict[str, object]:
    """Verify Hidden Subgroup Problem (HSP) Fourier concentration for all left cosets |gH> of 6 subgroups."""
    tbl = bt2t_mul.mul_table_24()
    inv = bt2t_mul.bt2t_base.inverse_from_table(tbl)
    f32 = bt2t_qft_matrix_32()
    chars = character_table_2t()
    slices = irrep_row_slices()
    subgroups = canonical_subgroups_2t()

    results: Dict[str, Dict[str, object]] = {}
    all_ok = True

    for sg_name, h_elems in subgroups.items():
        h_set = set(h_elems)
        # 1. Verify H is a genuine subgroup (contains identity, closed under product and inverse):
        is_subgroup = (
            0 in h_set
            and all(tbl[a][b] in h_set for a in h_elems for b in h_elems)
            and all(inv[a] in h_set for a in h_elems)
        )
        order_h = len(h_elems)

        # 2. Exact analytical multiplicity m_alpha(H) = (1/|H|) sum_{h in H} chi_alpha(h)
        #    and predicted Fourier weight P(alpha | gH) = d_alpha * |H| * m_alpha(H) / 24:
        multiplicities: Dict[str, int] = {}
        expected_weights: Dict[str, float] = {}
        for a_idx, ir_name in enumerate(IRREP_NAMES):
            m_raw = np.sum(chars[a_idx, h_elems]) / float(order_h)
            m_int = int(np.rint(np.real(m_raw)))
            if abs(m_raw - m_int) > 1e-12:
                all_ok = False
            multiplicities[ir_name] = m_int
            expected_weights[ir_name] = (
                IRREP_DIMS[ir_name] * order_h * m_int / float(N_GROUP_ORDER)
            )

        # 3. Enumerate ALL left cosets gH (g in 0..23) and evolve uniform state |gH> through F_tilde_{2T}:
        max_weight_err = 0.0
        max_forbidden_leakage = 0.0
        max_unused_leakage = 0.0
        sample_weights: Dict[str, float] = {}
        raw_weights_e: Dict[str, float] = {}

        for g in range(N_GROUP_ORDER):
            coset_members = sorted({tbl[g][h] for h in h_elems})
            psi = np.zeros(N_QFT_STATES, dtype=complex)
            psi[coset_members] = 1.0 / np.sqrt(float(order_h))
            out = f32 @ psi
            probs = np.abs(out) ** 2

            unused_leak = float(np.sum(probs[N_GROUP_ORDER:]))
            max_unused_leakage = max(max_unused_leakage, unused_leak)

            for ir_name in IRREP_NAMES:
                w = float(np.sum(probs[slices[ir_name]]))
                if g == 0:
                    raw_weights_e[ir_name] = w
                    sample_weights[ir_name] = round(w, 12)
                max_weight_err = max(max_weight_err, abs(w - expected_weights[ir_name]))
                if multiplicities[ir_name] == 0:
                    max_forbidden_leakage = max(max_forbidden_leakage, w)

        sg_ok = (
            is_subgroup
            and max_weight_err < 1e-14
            and max_forbidden_leakage < 1e-28
            and max_unused_leakage == 0.0
        )
        all_ok = all_ok and sg_ok
        results[sg_name] = {
            "order": order_h,
            "elements": h_elems,
            "is_valid_subgroup": is_subgroup,
            "trivial_irrep_multiplicities": multiplicities,
            "expected_irrep_weights": {k: round(v, 12) for k, v in expected_weights.items()},
            "measured_irrep_weights_coset_eH": sample_weights,
            "_raw_weights_eH": raw_weights_e,
            "max_coset_weight_abs_error_all_24_g": max_weight_err,
            "max_forbidden_irrep_leakage": max_forbidden_leakage,
            "max_unused_codeword_leakage": max_unused_leakage,
            "passed": sg_ok,
        }

    # Explicit spotlight check on normal subgroup Q_8: 100% weight on {chi_0, chi_1, chi_2} (1/3 each):
    q8_raw = results["Q8_normal"]["_raw_weights_eH"]
    q8_1d_total = sum(q8_raw[n] for n in ("chi_0", "chi_1", "chi_2"))  # type: ignore[index]
    q8_2d3d_total = sum(
        q8_raw[n] for n in ("rho_2_0", "rho_2_1", "rho_2_2", "rho_3")  # type: ignore[index]
    )
    for info in results.values():
        info.pop("_raw_weights_eH", None)

    return {
        "subgroups": results,
        "q8_1d_total_weight": q8_1d_total,
        "q8_2d_3d_leakage": q8_2d3d_total,
        "q8_concentrates_100pct_on_1d_irreps": (
            abs(q8_1d_total - 1.0) < 1e-14 and q8_2d3d_total < 1e-28
        ),
        "all_subgroups_hsp_verified": all_ok,
    }


def bt2t_qft_circuit():
    """Synthesize exact 5-qubit QuantumCircuit for F_tilde_{2T} = F_{2T} (+) I_8 via Quantum Shannon
    Decomposition with global phase aligned so Operator(qc).data == F_tilde_{2T} to < 1e-13.
    """
    from qiskit.quantum_info import Operator
    from qiskit.synthesis.unitary.qsd import qs_decomposition

    target = bt2t_qft_matrix_32()
    qc = qs_decomposition(target)
    qc.name = "bt2t_qft_5q"
    u_raw = Operator(qc).data
    phase = np.vdot(target.flatten(), u_raw.flatten())
    phase /= abs(phase)
    qc.global_phase -= float(np.angle(phase))
    return qc
