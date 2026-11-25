"""OSIRIS-BT2T-003 Milestone 3: Kogut-Susskind 2T non-abelian lattice gauge plaquette Trotter dynamics.

Implements the exact discrete non-abelian Kogut-Susskind lattice gauge Hamiltonian and real-time
Lie-Trotter / Strang-splitting propagators for a single elementary plaquette with gauge group
2T subset SU(2) (the binary tetrahedral group of order 24) on frozen 5-qubit encoding ``bt2t-enc-1``
(sha256 = 3e3b02d13539daaf431af39056d031f777a785a556eb9da21ef4ea6097dfc4c5).

Core mathematical & circuit pillars:
  1. Gauge-Fixed Single-Plaquette Formulation (5 qubits, 32 computational states, 24 group states):
     * Maximal-tree gauge fixing sets 3 of the 4 plaquette links to the identity e, reducing the
       physical holonomy to one 5-qubit link register |g> subject to residual basepoint conjugation
       Gauss's law |g> -> |h g h^{-1}> (7 conjugacy classes matching the 7 irreps of 2T).
     * Magnetic potential H_B = -(1 / g_YM^2) sum_{g in 2T} Re Tr(rho_{2D}(g)) |g><g|, where
       chi_{2D}(g) in {+2, -2, 0, +1, -1}. Under bt2t-enc-1, all 16 Hurwitz units satisfy
       chi_{2D}(g) = (Z_4 - Z_3)/2 (zero two-qubit gates!) while Q_8 center {+-1} is isolated on
       |0000>_{q4..q1} (x) Z_0, yielding a 16-CNOT Walsh-Hadamard Gray-code circuit U_B(dt) acting
       identically as +1 on all 8 unused codewords (24..31).
     * Electric kinetic operator H_E = (g_YM^2 / 2) Delta_{2T}, where Delta_{2T} is the bi-invariant
       24-cell nearest-neighbor Cayley graph Laplacian over S_8 = {8..15} calibrated to the continuum
       SU(2) fundamental quadratic Casimir c_2(rho_{2,0}) = 3/4:
         Delta_{2T} = (3/16) sum_{gamma in S_8} (I - L_gamma) = F_{2T}^dagger (sum_rho c_2(rho) Pi_rho) F_{2T},
       with exact dyadic Casimir eigenvalues {0, 9/4, 9/4, 3/4, 15/8, 15/8, 3/2} on the 7 irreps.
       Synthesized both via canonical 5Q Peter-Weyl QFT F_tilde_{2T}^dagger D_E(dt) F_tilde_{2T}
       and via a Casimir-adapted semidirect circuit V_PW^dagger D_E^{Cas}(dt) V_PW.
  2. Unfixed Multi-Link Holonomy Accumulation (2-link 10Q/15Q & 4-link 20Q/25Q):
     * Explicit reversible ordered holonomy circuits accumulating |g_1, g_2, g_3, g_4, e> ->
       |g_1, g_2, g_3, g_4, g_1 g_2 g_3^{-1} g_4^{-1}> from structured 10Q multiplication
       (bt2t_mul.py) and 5Q inversion (bt2t.py), plus phase-kickback magnetic evolution and
       uncomputation.
     * Exact proof across all 4-vertex Gauss-law-invariant states H_phys (dim = 7) that unfixed
       4-link evolution matches 5-qubit gauge-fixed evolution to <= 1e-14 while compressing
       register width 4x-5x (20Q/25Q -> 5Q) and native CZ count by >100x.
  3. Lie-Trotter (1st-order) & Strang Splitting (2nd-order) Dynamics & Native Compilation:
     * Strong/intermediate/weak coupling benchmarks g_YM^2 in {0.5, 1.0, 2.0}, tracking Wilson loop
       <W_plaq(t)>, electric energy <H_E(t)>, all 7 Peter-Weyl sector probabilities, 0.0 unused-8
       leakage, empirical O(dt^1) vs O(dt^2) spectral norm error scaling, and Qiskit CZ + pytket
       ZZPhase compilations.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import expm

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
BT2T_INVERSION_DIR = EXPERIMENTS_DIR / "bt2t_inversion"
BT2T_PRIMITIVES_DIR = EXPERIMENTS_DIR / "bt2t_primitives"


def _load_sibling_module(mod_key: str, file_path: Path):
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    parent_str = str(file_path.parent)
    added = False
    if parent_str not in sys.path:
        sys.path.insert(0, parent_str)
        added = True
    try:
        spec = importlib.util.spec_from_file_location(mod_key, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module {mod_key} from {file_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_key] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        if added and parent_str in sys.path:
            sys.path.remove(parent_str)


bt2t_base = _load_sibling_module("_bt2t_enc1_base", BT2T_INVERSION_DIR / "bt2t.py")
bt2t_mul = _load_sibling_module("bt2t_mul", BT2T_PRIMITIVES_DIR / "bt2t_mul.py")
bt2t_qft = _load_sibling_module("bt2t_qft", BT2T_PRIMITIVES_DIR / "bt2t_qft.py")

ENCODING_VERSION = "bt2t-enc-1"
ENCODING_SHA256 = "3e3b02d13539daaf431af39056d031f777a785a556eb9da21ef4ea6097dfc4c5"

N_LINK_QUBITS = 5
N_LINK_STATES = 32
N_GROUP_ORDER = 24
UNUSED_CODEWORDS: Tuple[int, ...] = tuple(range(N_GROUP_ORDER, N_LINK_STATES))

# 8 nearest neighbors of identity e=+1 on the 24-cell / SU(2) with Tr_{SU(2)}(gamma) = +1:
CAYLEY_NEIGHBORS_S8: Tuple[int, ...] = (8, 9, 10, 11, 12, 13, 14, 15)
# Continuum SU(2) calibration factor matching spin-1/2 Casimir j(j+1) = 3/4 on rho_{2,0}:
LAPLACIAN_SU2_SCALE = 3.0 / 16.0

CAYLEY_GRAPH_EIGENVALUES: Dict[str, int] = {
    "chi_0": 0,
    "chi_1": 12,
    "chi_2": 12,
    "rho_2_0": 4,
    "rho_2_1": 10,
    "rho_2_2": 10,
    "rho_3": 8,
}

CASIMIR_EIGENVALUES: Dict[str, float] = {
    name: LAPLACIAN_SU2_SCALE * float(val)
    for name, val in CAYLEY_GRAPH_EIGENVALUES.items()
}

COUPLING_REGIMES: Tuple[float, ...] = (0.5, 1.0, 2.0)

# 5-qubit T-shaped heavy-hex cell (identical to Milestones 1 & 2):
HEAVY_HEX_5Q: List[List[int]] = [
    [0, 1],
    [1, 0],
    [1, 2],
    [2, 1],
    [2, 3],
    [3, 2],
    [2, 4],
    [4, 2],
]


def heavy_hex_coupling_edges(n_cells: int) -> List[List[int]]:
    """Chain ``n_cells`` 5-qubit T5 heavy-hex cells into a connected 5*n_cells-qubit graph."""
    edges: List[List[int]] = []
    for cell in range(n_cells):
        off = 5 * cell
        for u, v in ((0, 1), (1, 2), (2, 3), (2, 4)):
            edges.append([off + u, off + v])
            edges.append([off + v, off + u])
        if cell + 1 < n_cells:
            edges.append([off + 3, off + 5])
            edges.append([off + 5, off + 3])
    return edges


HEAVY_HEX_20Q: List[List[int]] = heavy_hex_coupling_edges(4)
HEAVY_HEX_25Q: List[List[int]] = heavy_hex_coupling_edges(5)

GateTuple = Tuple[str, Tuple[int, ...], Tuple[int, ...]]


# ---------------------------------------------------------------------------
# 1. Group Structure, Conjugacy Classes, Magnetic & Electric Hamiltonians
# ---------------------------------------------------------------------------


def verify_frozen_encoding() -> Dict[str, object]:
    """Verify frozen encoding ``bt2t-enc-1`` SHA-256 integrity."""
    return bt2t_mul.verify_frozen_encoding()


def conjugacy_classes_2t() -> List[List[int]]:
    """Compute the 7 exact conjugacy classes of 2T in canonical representative order."""
    tbl = bt2t_mul.mul_table_24()
    inv = bt2t_base.inverse_from_table(tbl)
    visited: set[int] = set()
    classes: List[List[int]] = []
    for g in range(N_GROUP_ORDER):
        if g in visited:
            continue
        cls = sorted({tbl[tbl[h][g]][inv[h]] for h in range(N_GROUP_ORDER)})
        visited.update(cls)
        classes.append(cls)
    return classes


def adjoint_matrix_24(h: int) -> np.ndarray:
    """24x24 conjugation permutation matrix Ad_h |g> = |h g h^{-1}> = L_h R_h |g>."""
    return bt2t_qft.left_regular_matrix_24(h) @ bt2t_qft.right_regular_matrix_24(h)


def adjoint_matrix_32(h: int) -> np.ndarray:
    """32x32 conjugation unitary Ad_h (+) I_8 on the 5-qubit link register."""
    mat = np.eye(N_LINK_STATES, dtype=complex)
    mat[:N_GROUP_ORDER, :N_GROUP_ORDER] = adjoint_matrix_24(h)
    return mat


def fundamental_character_24() -> np.ndarray:
    """Exact fundamental SU(2) character chi_{2D}(g) = Re Tr(rho_{2,0}(g)) in {+2, -2, 0, +1, -1}."""
    out = np.zeros(N_GROUP_ORDER, dtype=float)
    for g in range(N_GROUP_ORDER):
        out[g] = float(np.real(np.trace(bt2t_qft.su2_matrix(g))))
    return out


def fundamental_character_32() -> np.ndarray:
    """Extended 32-element fundamental character chi_{2D}(x) (identically 0.0 on unused 24..31)."""
    out = np.zeros(N_LINK_STATES, dtype=float)
    out[:N_GROUP_ORDER] = fundamental_character_24()
    return out


def plaquette_wilson_operator_32() -> np.ndarray:
    """Normalized plaquette Wilson loop observable W_plaq = (1/2) Re Tr(rho_{2D}(g)) on 32 states."""
    return 0.5 * np.diag(fundamental_character_32()).astype(complex)


def magnetic_hamiltonian_24(g_ym_sq: float) -> np.ndarray:
    """24x24 Kogut-Susskind magnetic potential H_B = -(1 / g_YM^2) diag(chi_{2D}(g))."""
    if g_ym_sq <= 0.0:
        raise ValueError(f"g_ym_sq must be positive, got {g_ym_sq}")
    return -(1.0 / float(g_ym_sq)) * np.diag(fundamental_character_24()).astype(complex)


def magnetic_hamiltonian_32(g_ym_sq: float) -> np.ndarray:
    """32x32 Kogut-Susskind magnetic potential H_B (+) 0_8 (zero generator on unused 24..31)."""
    if g_ym_sq <= 0.0:
        raise ValueError(f"g_ym_sq must be positive, got {g_ym_sq}")
    return -(1.0 / float(g_ym_sq)) * np.diag(fundamental_character_32()).astype(complex)


def magnetic_propagator_matrix_32(dt: float, g_ym_sq: float) -> np.ndarray:
    """Exact 32x32 magnetic propagator U_B(dt) = exp(-i H_B dt) (identically +1 on unused 24..31)."""
    if g_ym_sq <= 0.0:
        raise ValueError(f"g_ym_sq must be positive, got {g_ym_sq}")
    phases = np.exp(1j * (float(dt) / float(g_ym_sq)) * fundamental_character_32())
    return np.diag(phases)


def cayley_laplacian_24(normalized: bool = True) -> np.ndarray:
    """24x24 bi-invariant Cayley graph Laplacian Delta_{2T} over S_8 = {8..15}."""
    scale = LAPLACIAN_SU2_SCALE if normalized else 1.0
    mat = np.zeros((N_GROUP_ORDER, N_GROUP_ORDER), dtype=complex)
    eye24 = np.eye(N_GROUP_ORDER, dtype=complex)
    for gamma in CAYLEY_NEIGHBORS_S8:
        mat += scale * (eye24 - bt2t_qft.left_regular_matrix_24(gamma))
    return mat


def cayley_laplacian_32(normalized: bool = True) -> np.ndarray:
    """32x32 extended Cayley graph Laplacian Delta_{2T} (+) 0_8 (zero on unused 24..31)."""
    mat = np.zeros((N_LINK_STATES, N_LINK_STATES), dtype=complex)
    mat[:N_GROUP_ORDER, :N_GROUP_ORDER] = cayley_laplacian_24(normalized=normalized)
    return mat


def casimir_diagonal_peter_weyl_24() -> np.ndarray:
    """24-entry quadratic Casimir diagonal vector sum_{rho} c_2(rho) Pi_rho in Peter-Weyl order."""
    diag = np.zeros(N_GROUP_ORDER, dtype=float)
    slices = bt2t_qft.irrep_row_slices()
    for name in bt2t_qft.IRREP_NAMES:
        diag[slices[name]] = CASIMIR_EIGENVALUES[name]
    return diag


def casimir_diagonal_peter_weyl_32() -> np.ndarray:
    """32-entry Casimir diagonal vector in Peter-Weyl basis (0.0 on unused 24..31)."""
    diag = np.zeros(N_LINK_STATES, dtype=float)
    diag[:N_GROUP_ORDER] = casimir_diagonal_peter_weyl_24()
    return diag


def electric_hamiltonian_24(g_ym_sq: float) -> np.ndarray:
    """24x24 Kogut-Susskind electric kinetic Hamiltonian H_E = (g_YM^2 / 2) Delta_{2T}."""
    if g_ym_sq <= 0.0:
        raise ValueError(f"g_ym_sq must be positive, got {g_ym_sq}")
    return 0.5 * float(g_ym_sq) * cayley_laplacian_24(normalized=True)


def electric_hamiltonian_32(g_ym_sq: float) -> np.ndarray:
    """32x32 Kogut-Susskind electric kinetic Hamiltonian H_E (+) 0_8."""
    if g_ym_sq <= 0.0:
        raise ValueError(f"g_ym_sq must be positive, got {g_ym_sq}")
    return 0.5 * float(g_ym_sq) * cayley_laplacian_32(normalized=True)


def electric_diagonal_propagator_pw_32(dt: float, g_ym_sq: float) -> np.ndarray:
    """32x32 diagonal phase gate D_E(dt) = exp(-i (g_YM^2 / 2) c_2(rho) dt) in Peter-Weyl basis."""
    if g_ym_sq <= 0.0:
        raise ValueError(f"g_ym_sq must be positive, got {g_ym_sq}")
    phases = np.exp(-1j * 0.5 * float(g_ym_sq) * casimir_diagonal_peter_weyl_32() * float(dt))
    return np.diag(phases)


def electric_propagator_matrix_32(dt: float, g_ym_sq: float) -> np.ndarray:
    """Exact 32x32 electric propagator U_E(dt) = F_tilde_{2T}^dagger D_E(dt) F_tilde_{2T}."""
    f32 = bt2t_qft.bt2t_qft_matrix_32()
    d_e = electric_diagonal_propagator_pw_32(dt, g_ym_sq)
    return f32.conj().T @ d_e @ f32


def kogut_susskind_hamiltonian_24(g_ym_sq: float) -> np.ndarray:
    """24x24 single-plaquette gauge-fixed Kogut-Susskind Hamiltonian H_KS = H_E + H_B."""
    return electric_hamiltonian_24(g_ym_sq) + magnetic_hamiltonian_24(g_ym_sq)


def kogut_susskind_hamiltonian_32(g_ym_sq: float) -> np.ndarray:
    """32x32 5-qubit single-plaquette Kogut-Susskind Hamiltonian H_KS = H_E + H_B (0_8 on unused)."""
    return electric_hamiltonian_32(g_ym_sq) + magnetic_hamiltonian_32(g_ym_sq)


def exact_propagator_matrix_32(t: float, g_ym_sq: float) -> np.ndarray:
    """Exact continuum-time 32x32 propagator exp(-i H_KS t) = exp(-i H_KS^{(24)} t) (+) I_8."""
    return expm(-1j * kogut_susskind_hamiltonian_32(g_ym_sq) * float(t))


def lie_trotter_step_matrix_32(dt: float, g_ym_sq: float) -> np.ndarray:
    """Single 1st-order Lie-Trotter step S_1(dt) = U_E(dt) U_B(dt)."""
    return electric_propagator_matrix_32(dt, g_ym_sq) @ magnetic_propagator_matrix_32(dt, g_ym_sq)


def strang_step_matrix_32(dt: float, g_ym_sq: float) -> np.ndarray:
    """Single 2nd-order Strang splitting step S_2(dt) = U_B(dt/2) U_E(dt) U_B(dt/2)."""
    ub_half = magnetic_propagator_matrix_32(0.5 * float(dt), g_ym_sq)
    ue = electric_propagator_matrix_32(dt, g_ym_sq)
    return ub_half @ ue @ ub_half


def verify_gauss_law_bi_invariance(
    g_ym_sq: float = 1.0, dt: float = 0.2
) -> Dict[str, object]:
    """Prove Cayley bi-invariance, Peter-Weyl Casimir diagonalization, Gauss-law conservation,
    and unused-8 decoupling in pure NumPy.
    """
    tbl = bt2t_mul.mul_table_24()
    inv = bt2t_base.inverse_from_table(tbl)

    s8_list = list(CAYLEY_NEIGHBORS_S8)
    s8_symmetric = sorted(inv[g] for g in s8_list) == s8_list
    s8_normal = all(
        sorted(tbl[tbl[h][g]][inv[h]] for g in s8_list) == s8_list
        for h in range(N_GROUP_ORDER)
    )

    generated = {0}
    frontier = {0}
    diameter = 0
    while frontier:
        nxt = {tbl[g][x] for x in frontier for g in s8_list} - generated
        if nxt:
            diameter += 1
            generated |= nxt
        frontier = nxt

    delta_left = cayley_laplacian_24(normalized=True)
    delta_right = np.zeros((N_GROUP_ORDER, N_GROUP_ORDER), dtype=complex)
    eye24 = np.eye(N_GROUP_ORDER, dtype=complex)
    for gamma in CAYLEY_NEIGHBORS_S8:
        delta_right += LAPLACIAN_SU2_SCALE * (eye24 - bt2t_qft.right_regular_matrix_24(gamma))
    left_right_laplacian_diff = float(np.linalg.norm(delta_left - delta_right, ord=2))

    max_comm_l = 0.0
    max_comm_r = 0.0
    for h in range(N_GROUP_ORDER):
        lh = bt2t_qft.left_regular_matrix_24(h)
        rh = bt2t_qft.right_regular_matrix_24(h)
        max_comm_l = max(
            max_comm_l, float(np.linalg.norm(delta_left @ lh - lh @ delta_left, ord=2))
        )
        max_comm_r = max(
            max_comm_r, float(np.linalg.norm(delta_left @ rh - rh @ delta_left, ord=2))
        )

    f24 = bt2t_qft.peter_weyl_matrix_24()
    delta_pw = f24 @ delta_left @ f24.conj().T
    expected_casimir = np.diag(casimir_diagonal_peter_weyl_24().astype(complex))
    pw_diag_err = float(np.linalg.norm(delta_pw - expected_casimir, ord=2))
    pw_off_diag_max = float(np.max(np.abs(delta_pw - np.diag(np.diag(delta_pw)))))

    h_b32 = magnetic_hamiltonian_32(g_ym_sq)
    h_e32 = electric_hamiltonian_32(g_ym_sq)
    h_ks32 = kogut_susskind_hamiltonian_32(g_ym_sq)

    herm_err = max(
        float(np.linalg.norm(h_b32 - h_b32.conj().T, ord=2)),
        float(np.linalg.norm(h_e32 - h_e32.conj().T, ord=2)),
        float(np.linalg.norm(h_ks32 - h_ks32.conj().T, ord=2)),
    )

    max_gauss_hb = 0.0
    max_gauss_he = 0.0
    max_gauss_hks = 0.0
    for h in range(N_GROUP_ORDER):
        adh = adjoint_matrix_32(h)
        max_gauss_hb = max(
            max_gauss_hb, float(np.linalg.norm(h_b32 @ adh - adh @ h_b32, ord=2))
        )
        max_gauss_he = max(
            max_gauss_he, float(np.linalg.norm(h_e32 @ adh - adh @ h_e32, ord=2))
        )
        max_gauss_hks = max(
            max_gauss_hks, float(np.linalg.norm(h_ks32 @ adh - adh @ h_ks32, ord=2))
        )

    ub_err = float(
        np.linalg.norm(
            magnetic_propagator_matrix_32(dt, g_ym_sq) - expm(-1j * h_b32 * dt), ord=2
        )
    )
    ue_err = float(
        np.linalg.norm(
            electric_propagator_matrix_32(dt, g_ym_sq) - expm(-1j * h_e32 * dt), ord=2
        )
    )

    unused_hks_norm = float(
        np.linalg.norm(h_ks32[N_GROUP_ORDER:, :], ord=2)
        + np.linalg.norm(h_ks32[:, N_GROUP_ORDER:], ord=2)
    )
    ub_unused_err = float(
        np.linalg.norm(
            magnetic_propagator_matrix_32(dt, g_ym_sq)[N_GROUP_ORDER:, N_GROUP_ORDER:] - np.eye(8),
            ord=2,
        )
    )
    ue_unused_err = float(
        np.linalg.norm(
            electric_propagator_matrix_32(dt, g_ym_sq)[N_GROUP_ORDER:, N_GROUP_ORDER:] - np.eye(8),
            ord=2,
        )
    )

    classes = conjugacy_classes_2t()
    chi24 = fundamental_character_24()
    chi_values_sorted = sorted({float(v) for v in chi24})

    return {
        "conjugacy_class_count": len(classes),
        "conjugacy_class_sizes": [len(c) for c in classes],
        "s8_neighbors": list(CAYLEY_NEIGHBORS_S8),
        "s8_symmetric_and_normal": s8_symmetric and s8_normal,
        "s8_generates_2t_diameter": diameter,
        "s8_generated_order": len(generated),
        "fundamental_character_distinct_values": chi_values_sorted,
        "cayley_graph_eigenvalues": CAYLEY_GRAPH_EIGENVALUES,
        "casimir_su2_calibrated_eigenvalues": CASIMIR_EIGENVALUES,
        "left_right_laplacian_2norm_diff": left_right_laplacian_diff,
        "max_commutator_delta_left_translation_2norm": max_comm_l,
        "max_commutator_delta_right_translation_2norm": max_comm_r,
        "peter_weyl_casimir_diagonalization_2norm_error": pw_diag_err,
        "peter_weyl_off_diagonal_max_abs": pw_off_diag_max,
        "hamiltonian_hermiticity_2norm_error": herm_err,
        "max_commutator_hb_gauss_adjoint_2norm": max_gauss_hb,
        "max_commutator_he_gauss_adjoint_2norm": max_gauss_he,
        "max_commutator_hks_gauss_adjoint_2norm": max_gauss_hks,
        "ub_matrix_vs_expm_2norm_error": ub_err,
        "ue_pw_qft_vs_expm_2norm_error": ue_err,
        "unused_8_hamiltonian_coupling_norm": unused_hks_norm,
        "unused_8_propagator_identity_error": max(ub_unused_err, ue_unused_err),
        "checks": {
            "conjugacy_classes_equal_7_irreps": len(classes) == 7 and sum(len(c) for c in classes) == 24,
            "s8_cayley_set_symmetric_normal_generating": (
                s8_symmetric and s8_normal and len(generated) == 24 and diameter == 3
            ),
            "laplacian_bi_invariant_below_1e14": (
                left_right_laplacian_diff < 1e-14 and max_comm_l < 1e-14 and max_comm_r < 1e-14
            ),
            "peter_weyl_diagonalizes_he_below_1e14": pw_diag_err < 1e-14 and pw_off_diag_max < 1e-14,
            "gauss_law_adjoint_conserved_below_1e14": (
                max_gauss_hb < 1e-14 and max_gauss_he < 1e-14 and max_gauss_hks < 1e-14
            ),
            "ub_and_ue_match_expm_below_1e14": ub_err < 1e-14 and ue_err < 1e-14,
            "unused_8_exact_zero_generator_and_identity": (
                unused_hks_norm == 0.0 and max(ub_unused_err, ue_unused_err) < 1e-14
            ),
        },
    }


# ---------------------------------------------------------------------------
# 2. Multi-Link Reversible Holonomy Gate Lists & Gauge-Invariant Isomorphism
# ---------------------------------------------------------------------------


def structured_inversion_gate_list_5q(offset: int = 0) -> List[GateTuple]:
    """5-qubit structured inversion gate tuples shifted to qubit register ``offset..offset+4``."""
    o = offset
    gates: List[GateTuple] = []
    for t in (0, 1, 2):
        gates.append(("cx", (o + 3,), (o + t,)))
        gates.append(("cx", (o + 4,), (o + t,)))
    gates.append(("x", (), (o + 3,)))
    gates.append(("x", (), (o + 4,)))
    gates.append(("ccx", (o + 3, o + 4), (o + 0,)))
    gates.append(("x", (), (o + 1,)))
    gates.append(("x", (), (o + 2,)))
    gates.append(("mcx", (o + 1, o + 2, o + 3, o + 4), (o + 0,)))
    for q in (1, 2, 3, 4):
        gates.append(("x", (), (o + q,)))
    return gates


def shift_10q_mul_gate_list(
    reg_x_offset: int, reg_y_offset: int, base_gates: Optional[Sequence[GateTuple]] = None
) -> List[GateTuple]:
    """Map 10Q structured multiplication gate tuples from (0..4, 5..9) onto arbitrary 5Q registers."""
    src = bt2t_mul.multiplication_gate_list_structured() if base_gates is None else base_gates
    qmap = [reg_x_offset + q for q in range(5)] + [reg_y_offset + q for q in range(5)]
    shifted: List[GateTuple] = []
    for kind, ctrls, tgts in src:
        shifted.append(
            (kind, tuple(qmap[c] for c in ctrls), tuple(qmap[t] for t in tgts))
        )
    return shifted


def holonomy_2link_gate_list_15q() -> List[GateTuple]:
    """15-qubit reversible 2-link digon holonomy gate list:
      |g_1>_{0..4} |g_2>_{5..9} |y>_{10..14} -> |g_1> |g_2> |g_1 * g_2^{-1} * y>.
    """
    mul_base = bt2t_mul.multiplication_gate_list_structured()
    gates: List[GateTuple] = []
    gates.extend(structured_inversion_gate_list_5q(5))
    gates.extend(shift_10q_mul_gate_list(5, 10, mul_base))
    gates.extend(shift_10q_mul_gate_list(0, 10, mul_base))
    gates.extend(structured_inversion_gate_list_5q(5))
    return gates


def holonomy_4link_gate_list_25q() -> List[GateTuple]:
    """25-qubit reversible 4-link plaquette holonomy gate list:
      |g_1>_{0..4} |g_2>_{5..9} |g_3>_{10..14} |g_4>_{15..19} |y>_{20..24}
        -> |g_1, g_2, g_3, g_4, g_1 * g_2 * g_3^{-1} * g_4^{-1} * y>.
    """
    mul_base = bt2t_mul.multiplication_gate_list_structured()
    gates: List[GateTuple] = []
    gates.extend(structured_inversion_gate_list_5q(10))
    gates.extend(structured_inversion_gate_list_5q(15))
    gates.extend(shift_10q_mul_gate_list(15, 20, mul_base))
    gates.extend(shift_10q_mul_gate_list(10, 20, mul_base))
    gates.extend(shift_10q_mul_gate_list(5, 20, mul_base))
    gates.extend(shift_10q_mul_gate_list(0, 20, mul_base))
    gates.extend(structured_inversion_gate_list_5q(10))
    gates.extend(structured_inversion_gate_list_5q(15))
    return gates


def apply_gate_list_vectorized_uint32(
    gates: Sequence[GateTuple], states: np.ndarray
) -> np.ndarray:
    """Bit-parallel classical gate-list simulator on an array of uint32 computational basis integers."""
    cur = states.astype(np.uint32, copy=True)
    one = np.uint32(1)
    for kind, ctrls, tgts in gates:
        if kind == "x":
            cur ^= one << np.uint32(tgts[0])
        elif kind == "cx":
            mask = (cur >> np.uint32(ctrls[0])) & one
            cur ^= mask << np.uint32(tgts[0])
        elif kind == "ccx":
            mask = ((cur >> np.uint32(ctrls[0])) & (cur >> np.uint32(ctrls[1]))) & one
            cur ^= mask << np.uint32(tgts[0])
        elif kind == "cswap":
            c_bit = (cur >> np.uint32(ctrls[0])) & one
            diff = ((cur >> np.uint32(tgts[0])) ^ (cur >> np.uint32(tgts[1]))) & c_bit
            cur ^= (diff << np.uint32(tgts[0])) | (diff << np.uint32(tgts[1]))
        elif kind == "mcx":
            mask = np.ones_like(cur, dtype=np.uint32)
            for c in ctrls:
                mask &= (cur >> np.uint32(c)) & one
            cur ^= mask << np.uint32(tgts[0])
        else:
            raise ValueError(f"Unsupported gate tuple kind: {kind}")
    return cur


def verify_multilink_gauge_isomorphism(
    dt: float = 0.15,
    g_ym_sq: float = 1.0,
    n_steps: int = 5,
    seed: int = 20261124,
) -> Dict[str, object]:
    """Prove exact classical/quantum multi-link holonomy accumulation and 4-link vs 5Q gauge-fixed
    dynamical isomorphism across the 4-vertex Gauss-law-invariant physical Hilbert space H_phys.
    """
    tbl = np.asarray(bt2t_mul.mul_table_24(), dtype=np.int32)
    inv = np.asarray(bt2t_base.inverse_from_table(tbl.tolist()), dtype=np.int32)

    # 1. Exhaustive 15Q 2-link holonomy gate-list check on all 24^3 = 13,824 valid (g1, g2, y) triples:
    g1_2, g2_2, y_2 = np.meshgrid(
        np.arange(24, dtype=np.uint32),
        np.arange(24, dtype=np.uint32),
        np.arange(24, dtype=np.uint32),
        indexing="ij",
    )
    packed_15q = (g1_2 | (g2_2 << np.uint32(5)) | (y_2 << np.uint32(10))).ravel()
    out_15q = apply_gate_list_vectorized_uint32(holonomy_2link_gate_list_15q(), packed_15q)
    hol2_expected = tbl[g1_2, tbl[inv[g2_2], y_2]].ravel().astype(np.uint32)
    expected_15q = (
        g1_2.ravel() | (g2_2.ravel() << np.uint32(5)) | (hol2_expected << np.uint32(10))
    )
    hol2_15q_exact = bool(np.array_equal(out_15q, expected_15q))

    # 2. Exhaustive 25Q 4-link holonomy gate-list check across ALL 24^4 = 331,776 valid 4-link states
    #    with |e>_{anc} = |00000>, plus uncomputation check back to |00000>_{anc}:
    g12 = tbl[:, :, None, None]
    g3i_g4i = tbl[inv[:, None], inv[None, :]][None, None, :, :]
    hol_4 = tbl[g12, g3i_g4i]  # shape (24, 24, 24, 24)

    idx_24 = np.arange(24, dtype=np.uint32)
    a1, a2, a3, a4 = np.meshgrid(idx_24, idx_24, idx_24, idx_24, indexing="ij")
    packed_25q_in = (
        a1.ravel()
        | (a2.ravel() << np.uint32(5))
        | (a3.ravel() << np.uint32(10))
        | (a4.ravel() << np.uint32(15))
    )
    gates_25q = holonomy_4link_gate_list_25q()
    packed_25q_mid = apply_gate_list_vectorized_uint32(gates_25q, packed_25q_in)
    expected_25q_mid = packed_25q_in | (hol_4.ravel().astype(np.uint32) << np.uint32(20))
    hol4_25q_all_331776_exact = bool(np.array_equal(packed_25q_mid, expected_25q_mid))

    packed_25q_uncomp = apply_gate_list_vectorized_uint32(
        list(reversed(gates_25q)), packed_25q_mid
    )
    hol4_uncompute_exact = bool(np.array_equal(packed_25q_uncomp, packed_25q_in))

    # 3. Construct the 7 orthonormal 4-link character basis states of H_phys via isometric gauge lift
    #    J_{1->4} |g> = 24^{-3/2} sum_{g1 g2 g3^{-1} g4^{-1} = g} |g1, g2, g3, g4>:
    chars = bt2t_qft.character_table_2t()  # (7, 24)
    psi1_chars = chars.conj() / np.sqrt(float(N_GROUP_ORDER))  # (7, 24)
    norm_4 = float(N_GROUP_ORDER) ** (-1.5)
    psi4_chars = psi1_chars[:, hol_4] * norm_4  # (7, 24, 24, 24, 24)

    gram_7 = np.einsum("aijkl,bijkl->ab", psi4_chars.conj(), psi4_chars)
    isometry_gram_err = float(np.max(np.abs(gram_7 - np.eye(7))))

    rng = np.random.default_rng(seed)
    c7 = rng.normal(size=7) + 1j * rng.normal(size=7)
    c7 /= np.linalg.norm(c7)
    psi4_test = np.tensordot(c7, psi4_chars, axes=(0, 0))  # (24, 24, 24, 24)

    max_gauss_4v_all24 = 0.0
    for h in range(N_GROUP_ORDER):
        hi = int(inv[h])
        lh_idx = tbl[hi]
        rh_idx = tbl[:, h]
        gv0 = psi4_test[lh_idx, :, :, :][:, :, :, lh_idx]
        gv1 = psi4_test[rh_idx, :, :, :][:, lh_idx, :, :]
        gv2 = psi4_test[:, rh_idx, :, :][:, :, rh_idx, :]
        gv3 = psi4_test[:, :, lh_idx, :][:, :, :, rh_idx]
        for gv in (gv0, gv1, gv2, gv3):
            max_gauss_4v_all24 = max(
                max_gauss_4v_all24, float(np.linalg.norm((gv - psi4_test).ravel()))
            )

    max_gauss_7basis_gens = 0.0
    for h in (1, 2, 4, 8, 23):
        hi = int(inv[h])
        lh_idx = tbl[hi]
        rh_idx = tbl[:, h]
        b_v0 = psi4_chars[:, lh_idx, :, :, :][:, :, :, :, lh_idx]
        b_v1 = psi4_chars[:, rh_idx, :, :, :][:, :, lh_idx, :, :]
        b_v2 = psi4_chars[:, :, rh_idx, :, :][:, :, :, rh_idx, :]
        b_v3 = psi4_chars[:, :, :, lh_idx, :][:, :, :, :, rh_idx]
        for bv in (b_v0, b_v1, b_v2, b_v3):
            max_gauss_7basis_gens = max(
                max_gauss_7basis_gens, float(np.max(np.abs(bv - psi4_chars)))
            )

    # 4. Verify single-link electric Laplacian intertwining on all 4 links:
    delta24 = cayley_laplacian_24(normalized=True)
    psi1_init = c7 @ psi1_chars
    target_delta_lifted = (delta24 @ psi1_init)[hol_4] * norm_4
    max_link_laplacian_intertwine_err = 0.0
    for ax in range(4):
        acted = np.moveaxis(np.tensordot(delta24, psi4_test, axes=([1], [ax])), 0, ax)
        max_link_laplacian_intertwine_err = max(
            max_link_laplacian_intertwine_err,
            float(np.linalg.norm((acted - target_delta_lifted).ravel())),
        )

    # 5. Multi-step Strang Trotter evolution across all 7 character basis states + 4 random superpositions:
    chi24 = fundamental_character_24()
    ub_half_24 = np.exp(1j * (0.5 * float(dt) / float(g_ym_sq)) * chi24)
    ue_1link_24 = expm(-1j * 0.5 * float(g_ym_sq) * delta24 * float(dt))
    ue_per_link_4 = expm(-1j * (0.5 * float(g_ym_sq) / 4.0) * delta24 * float(dt))

    cur_1 = psi1_chars.copy()
    cur_4 = psi4_chars.copy()
    for _ in range(n_steps):
        cur_1 = cur_1 * ub_half_24[None, :]
        cur_1 = cur_1 @ ue_1link_24.T
        cur_1 = cur_1 * ub_half_24[None, :]

        cur_4 = cur_4 * ub_half_24[hol_4][None, ...]
        for ax in range(1, 5):
            cur_4 = np.moveaxis(
                np.tensordot(ue_per_link_4, cur_4, axes=([1], [ax])), 0, ax
            )
        cur_4 = cur_4 * ub_half_24[hol_4][None, ...]

    cur_1_lifted = cur_1[:, hol_4] * norm_4
    max_basis_4link_vs_1link_l2_err = max(
        float(np.linalg.norm((cur_4[a] - cur_1_lifted[a]).ravel())) for a in range(7)
    )

    max_rand_4link_vs_1link_l2_err = 0.0
    for _ in range(4):
        w = rng.normal(size=7) + 1j * rng.normal(size=7)
        w /= np.linalg.norm(w)
        diff_w = np.tensordot(w, cur_4 - cur_1_lifted, axes=(0, 0))
        max_rand_4link_vs_1link_l2_err = max(
            max_rand_4link_vs_1link_l2_err, float(np.linalg.norm(diff_w.ravel()))
        )

    # 6. Also verify 2-link digon (24x24) unfixed vs 5Q gauge-fixed evolution:
    hol_2 = tbl[:, inv]
    norm_2 = 1.0 / np.sqrt(float(N_GROUP_ORDER))
    psi2_init = psi1_init[hol_2] * norm_2
    ue_per_link_2 = expm(-1j * (0.5 * float(g_ym_sq) / 2.0) * delta24 * float(dt))
    cur_2 = psi2_init.copy()
    cur_1_single = psi1_init.copy()
    for _ in range(n_steps):
        cur_1_single = ub_half_24 * (ue_1link_24 @ (ub_half_24 * cur_1_single))
        cur_2 = ub_half_24[hol_2] * cur_2
        cur_2 = ue_per_link_2 @ cur_2 @ ue_per_link_2.T
        cur_2 = ub_half_24[hol_2] * cur_2
    two_link_vs_1link_l2_err = float(
        np.linalg.norm((cur_2 - cur_1_single[hol_2] * norm_2).ravel())
    )

    return {
        "holonomy_2link_15q_triples_checked": int(packed_15q.size),
        "holonomy_4link_25q_configs_checked": int(packed_25q_in.size),
        "physical_gauss_invariant_subspace_dim": 7,
        "qubit_reduction_unfixed_20q_to_5q": 4.0,
        "qubit_reduction_ancilla_25q_to_5q": 5.0,
        "isometry_7x7_gram_max_abs_error": isometry_gram_err,
        "max_4vertex_gauss_law_l2_error_all_24_h": max_gauss_4v_all24,
        "max_7sector_gauss_generator_abs_error": max_gauss_7basis_gens,
        "max_link_laplacian_intertwining_l2_error": max_link_laplacian_intertwine_err,
        "n_strang_steps_compared": n_steps,
        "two_link_vs_5q_gauge_fixed_l2_error": two_link_vs_1link_l2_err,
        "four_link_vs_5q_basis_max_l2_error": max_basis_4link_vs_1link_l2_err,
        "four_link_vs_5q_random_superposition_max_l2_error": max_rand_4link_vs_1link_l2_err,
        "checks": {
            "holonomy_2link_15q_all_13824_exact": hol2_15q_exact,
            "holonomy_4link_25q_all_331776_exact": hol4_25q_all_331776_exact,
            "holonomy_4link_25q_ancilla_uncomputes_exact": hol4_uncompute_exact,
            "gauss_law_4vertex_invariant_below_1e13": (
                isometry_gram_err < 1e-11
                and max_gauss_4v_all24 < 1e-13
                and max_gauss_7basis_gens < 1e-14
            ),
            "link_laplacians_intertwine_below_1e13": max_link_laplacian_intertwine_err < 1e-13,
            "two_link_matches_5q_gauge_fixed_below_1e13": two_link_vs_1link_l2_err <= 1e-13,
            "four_link_matches_5q_gauge_fixed_below_1e13": (
                max_basis_4link_vs_1link_l2_err <= 1e-13
                and max_rand_4link_vs_1link_l2_err <= 1e-13
            ),
        },
    }


# ---------------------------------------------------------------------------
# 3. Real-Time Plaquette Observables & Lie-Trotter / Strang Error Scaling
# ---------------------------------------------------------------------------


def initial_state_electric_vacuum_32() -> np.ndarray:
    """Strong-coupling electric ground state |Omega_E> = |chi_0> = (1/sqrt(24)) sum_{g in 2T} |g>."""
    psi = np.zeros(N_LINK_STATES, dtype=complex)
    psi[:N_GROUP_ORDER] = 1.0 / np.sqrt(float(N_GROUP_ORDER))
    return psi


def initial_state_magnetic_identity_32() -> np.ndarray:
    """Weak-coupling magnetic flux-free identity holonomy state |e> = |00000>."""
    psi = np.zeros(N_LINK_STATES, dtype=complex)
    psi[0] = 1.0
    return psi


def evaluate_plaquette_observables(
    psi_32: np.ndarray, g_ym_sq: float
) -> Dict[str, object]:
    """Compute <W_plaq>, <H_E>, <H_B>, <H_KS>, 7 irrep sector weights, and unused-8 leakage."""
    w_op = plaquette_wilson_operator_32()
    h_e = electric_hamiltonian_32(g_ym_sq)
    h_b = magnetic_hamiltonian_32(g_ym_sq)
    f32 = bt2t_qft.bt2t_qft_matrix_32()
    slices = bt2t_qft.irrep_row_slices()

    wilson = float(np.real(psi_32.conj() @ w_op @ psi_32))
    e_kin = float(np.real(psi_32.conj() @ h_e @ psi_32))
    e_mag = float(np.real(psi_32.conj() @ h_b @ psi_32))
    pw_amps = f32 @ psi_32
    pw_probs = np.abs(pw_amps) ** 2
    sector_probs = {
        name: float(np.sum(pw_probs[slices[name]])) for name in bt2t_qft.IRREP_NAMES
    }
    unused_leakage = float(np.sum(np.abs(psi_32[N_GROUP_ORDER:]) ** 2))
    return {
        "wilson_loop": wilson,
        "electric_energy": e_kin,
        "magnetic_energy": e_mag,
        "total_energy": e_kin + e_mag,
        "irrep_sector_probabilities": sector_probs,
        "unused_8_leakage": unused_leakage,
    }


def simulate_plaquette_regimes(
    t_final: float = 1.0,
    n_strang_steps: int = 64,
    couplings: Sequence[float] = COUPLING_REGIMES,
) -> Dict[str, object]:
    """Benchmark physical observables across strong/intermediate/weak couplings g_YM^2 in {0.5, 1.0, 2.0}."""
    dt = float(t_final) / float(n_strang_steps)
    psi_vac0 = initial_state_electric_vacuum_32()
    psi_id0 = initial_state_magnetic_identity_32()

    regimes_out: Dict[str, object] = {}
    max_unused_leakage = 0.0
    max_strang_state_err = 0.0
    max_gauss_drift = 0.0

    ad_generators = [adjoint_matrix_32(h) for h in (2, 4, 8, 23)]

    for g_sq in couplings:
        u_exact = exact_propagator_matrix_32(t_final, g_sq)
        s2_step = strang_step_matrix_32(dt, g_sq)
        u_strang = np.linalg.matrix_power(s2_step, n_strang_steps)

        regime_entry: Dict[str, object] = {}
        for init_label, psi0 in (
            ("electric_vacuum_chi0", psi_vac0),
            ("magnetic_identity_e", psi_id0),
        ):
            psi_ex = u_exact @ psi0
            psi_st = u_strang @ psi0
            st_err = float(np.linalg.norm(psi_st - psi_ex))
            max_strang_state_err = max(max_strang_state_err, st_err)

            obs_ex = evaluate_plaquette_observables(psi_ex, g_sq)
            obs_st = evaluate_plaquette_observables(psi_st, g_sq)
            max_unused_leakage = max(
                max_unused_leakage,
                float(obs_ex["unused_8_leakage"]),  # type: ignore[arg-type]
                float(obs_st["unused_8_leakage"]),  # type: ignore[arg-type]
            )
            for adh in ad_generators:
                max_gauss_drift = max(
                    max_gauss_drift, float(np.linalg.norm(adh @ psi_st - psi_st))
                )

            regime_entry[init_label] = {
                "exact": obs_ex,
                "strang": obs_st,
                "strang_vs_exact_state_l2_error": st_err,
            }
        regimes_out[f"g_ym_sq_{g_sq:.1f}"] = regime_entry

    psi_unused = np.zeros(N_LINK_STATES, dtype=complex)
    psi_unused[24:] = 1.0 / np.sqrt(8.0)
    s2_ref = strang_step_matrix_32(dt, 1.0)
    psi_unused_out = np.linalg.matrix_power(s2_ref, n_strang_steps) @ psi_unused
    unused_confinement_err = float(np.linalg.norm(psi_unused_out - psi_unused))

    return {
        "t_final": float(t_final),
        "n_strang_steps": int(n_strang_steps),
        "dt": dt,
        "regimes": regimes_out,
        "max_unused_8_leakage": max_unused_leakage,
        "unused_8_stationary_error": unused_confinement_err,
        "max_strang_vs_exact_state_l2_error": max_strang_state_err,
        "max_gauss_conjugation_state_drift": max_gauss_drift,
        "checks": {
            "unused_8_leakage_below_1e28": max_unused_leakage < 1e-28,
            "unused_8_subspace_exact_identity": unused_confinement_err < 1e-14,
            "gauss_invariance_preserved_under_strang": max_gauss_drift < 1e-14,
            "strang_64step_state_error_below_1e4": max_strang_state_err < 1e-4,
        },
    }


def benchmark_trotter_error_scaling(
    t_final: float = 1.0,
    couplings: Sequence[float] = COUPLING_REGIMES,
    step_counts: Sequence[int] = (8, 16, 32, 64, 128, 256, 512, 1024, 2048),
) -> Dict[str, object]:
    """Quantify 1st-order Lie-Trotter O(dt^1) vs 2nd-order Strang O(dt^2) spectral norm error scaling."""
    results_by_coupling: Dict[str, object] = {}
    lie_slopes: List[float] = []
    strang_slopes: List[float] = []
    moderate_strang_errors: List[float] = []
    fine_strang_errors: List[float] = []

    for g_sq in couplings:
        u_exact = exact_propagator_matrix_32(t_final, g_sq)
        dts: List[float] = []
        lie_errs: List[float] = []
        strang_errs: List[float] = []
        rows: List[Dict[str, float]] = []

        for n_steps in step_counts:
            dt = float(t_final) / float(n_steps)
            dts.append(dt)
            u_lie = np.linalg.matrix_power(lie_trotter_step_matrix_32(dt, g_sq), n_steps)
            u_str = np.linalg.matrix_power(strang_step_matrix_32(dt, g_sq), n_steps)
            e_lie = float(np.linalg.norm(u_lie - u_exact, ord=2))
            e_str = float(np.linalg.norm(u_str - u_exact, ord=2))
            lie_errs.append(e_lie)
            strang_errs.append(e_str)
            rows.append(
                {
                    "n_steps": int(n_steps),
                    "dt": dt,
                    "lie_2norm_error": e_lie,
                    "strang_2norm_error": e_str,
                }
            )
            if n_steps == 64:
                moderate_strang_errors.append(e_str)
            if n_steps == 2048:
                fine_strang_errors.append(e_str)

        fit_idx = [i for i, n in enumerate(step_counts) if n >= 32]
        log_dt = np.log([dts[i] for i in fit_idx])
        slope_lie = float(np.polyfit(log_dt, np.log([lie_errs[i] for i in fit_idx]), 1)[0])
        slope_str = float(np.polyfit(log_dt, np.log([strang_errs[i] for i in fit_idx]), 1)[0])
        lie_slopes.append(slope_lie)
        strang_slopes.append(slope_str)

        results_by_coupling[f"g_ym_sq_{g_sq:.1f}"] = {
            "g_ym_sq": float(g_sq),
            "lie_loglog_slope": slope_lie,
            "strang_loglog_slope": slope_str,
            "series": rows,
        }

    return {
        "t_final": float(t_final),
        "step_counts": [int(n) for n in step_counts],
        "by_coupling": results_by_coupling,
        "min_lie_slope": min(lie_slopes),
        "max_lie_slope": max(lie_slopes),
        "min_strang_slope": min(strang_slopes),
        "max_strang_slope": max(strang_slopes),
        "max_moderate_step_strang_error_n64": max(moderate_strang_errors),
        "max_fine_step_strang_error_n2048": max(fine_strang_errors),
        "checks": {
            "lie_trotter_first_order_slope_near_1": all(
                0.98 <= s <= 1.02 for s in lie_slopes
            ),
            "strang_second_order_slope_near_2": all(
                1.98 <= s <= 2.02 for s in strang_slopes
            ),
            "strang_moderate_step_error_below_1e4": max(moderate_strang_errors) < 1e-4,
            "strang_fine_step_error_below_1e7": max(fine_strang_errors) < 1e-7,
        },
    }


# ---------------------------------------------------------------------------
# 4. Structured Quantum Circuit Synthesis & Native Gate-Count Compilation
# ---------------------------------------------------------------------------


def _walsh_hadamard_1d(values: np.ndarray) -> np.ndarray:
    """Normalized fast Walsh-Hadamard transform of length 2^k."""
    w = values.astype(float, copy=True)
    n = w.shape[0]
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x, y = w[j], w[j + h]
                w[j] = x + y
                w[j + h] = x - y
        h *= 2
    return w / float(n)


def _lowest_bit_index(mask: int) -> int:
    """Python 3.9-compatible index of least significant set bit of positive integer ``mask``."""
    return (mask & -mask).bit_length() - 1


def _append_gray_multiplexed_rot(
    qc,
    target: int,
    controls: Sequence[int],
    angles_by_ctrl_val: Dict[int, float],
    axis: str = "z",
) -> None:
    """Synthesize multiplexed R_axis(angles[val])_target controlled on ``controls`` via Gray code."""
    k = len(controls)
    n_ctrl = 1 << k
    f = np.zeros(n_ctrl, dtype=float)
    for val, ang in angles_by_ctrl_val.items():
        f[val] = float(ang)
    coeffs = _walsh_hadamard_1d(f)

    if axis == "y":
        qc.rx(0.5 * np.pi, target)

    active_mask = 0
    for step in range(n_ctrl):
        g_mask = step ^ (step >> 1)
        if abs(coeffs[g_mask]) > 1e-15:
            diff = active_mask ^ g_mask
            while diff:
                lsb = diff & -diff
                bit_idx = _lowest_bit_index(diff)
                qc.cx(controls[bit_idx], target)
                active_mask ^= lsb
                diff ^= lsb
            qc.rz(float(coeffs[g_mask]), target)
    while active_mask:
        lsb = active_mask & -active_mask
        bit_idx = _lowest_bit_index(active_mask)
        qc.cx(controls[bit_idx], target)
        active_mask ^= lsb

    if axis == "y":
        qc.rx(-0.5 * np.pi, target)


def _append_sparse_walsh_diagonal_5q(qc, phases_rad: np.ndarray) -> None:
    """Synthesize exact 5-qubit diagonal unitary diag(exp(i * phases_rad[x])) via sparse Walsh-Gray ladder."""
    w = _walsh_hadamard_1d(phases_rad)
    qc.global_phase += float(w[0])
    for target in range(N_LINK_QUBITS):
        base = 1 << target
        sub_w = w[base : 2 * base]
        if np.all(np.abs(sub_w) <= 1e-15):
            continue
        n_sub = 1 << target
        active_mask = 0
        for step in range(n_sub):
            g_mask = step ^ (step >> 1)
            if abs(sub_w[g_mask]) > 1e-15:
                diff = active_mask ^ g_mask
                while diff:
                    lsb = diff & -diff
                    bit_idx = _lowest_bit_index(diff)
                    qc.cx(bit_idx, target)
                    active_mask ^= lsb
                    diff ^= lsb
                qc.rz(-2.0 * float(sub_w[g_mask]), target)
        while active_mask:
            lsb = active_mask & -active_mask
            bit_idx = _lowest_bit_index(active_mask)
            qc.cx(bit_idx, target)
            active_mask ^= lsb


def magnetic_propagator_circuit_5q(dt: float, g_ym_sq: float):
    """Exact 5-qubit structured magnetic propagator circuit U_B(dt) = exp(-i H_B dt) (16 CX, 18 RZ).

    Exploits bt2t-enc-1 structure:
      * Hurwitz cosets (8..23) + unused (24..31): chi_{2D}(x) = (Z_4 - Z_3)/2 -> 2 single-qubit RZs!
      * Q_8 center {0, 1}: 2 * 1_{q4=q3=q2=q1=0} * Z_0 -> 16-step Gray-code parity cycle onto q0!
    """
    if g_ym_sq <= 0.0:
        raise ValueError(f"g_ym_sq must be positive, got {g_ym_sq}")
    from qiskit import QuantumCircuit

    theta = float(dt) / float(g_ym_sq)
    qc = QuantumCircuit(N_LINK_QUBITS, name="bt2t_ub_5q")
    qc.rz(-theta, 4)
    qc.rz(theta, 3)

    active_mask = 0
    for step in range(16):
        g_mask = step ^ (step >> 1)
        diff = active_mask ^ g_mask
        if diff:
            bit_idx = _lowest_bit_index(diff)
            qc.cx(bit_idx + 1, 0)
            active_mask = g_mask
        qc.rz(-0.25 * theta, 0)
    while active_mask:
        lsb = active_mask & -active_mask
        bit_idx = _lowest_bit_index(active_mask)
        qc.cx(bit_idx + 1, 0)
        active_mask ^= lsb
    return qc


def casimir_peter_weyl_circuit_5q():
    """Structured 5-qubit unitary V_PW mapping computational basis |x> into an exact orthonormal
    Casimir-adapted Peter-Weyl eigenbasis diagonalizing Delta_{2T} (fixing unused subspace to zero).
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(N_LINK_QUBITS, name="bt2t_casimir_pw_5q")
    for t in (0, 1, 2):
        qc.cx(4, t)
    qc.cx(3, 4)
    qc.cx(4, 3)
    qc.cswap(4, 3, 0)
    qc.h(0)
    for t in (1, 2, 3):
        qc.ch(4, t)
    qc.cswap(4, 1, 2)
    qc.cx(3, 2)
    qc.cx(3, 1)
    qc.ccx(4, 2, 0)
    qc.ccx(1, 0, 3)
    qc.ccx(4, 2, 0)
    qc.ccx(1, 0, 3)
    qc.cx(3, 1)
    qc.cx(3, 2)
    qc.x([0, 4])
    for t in (1, 2):
        qc.ry(-0.25 * np.pi, t)
        qc.ccz(0, 4, t)
        qc.ry(0.25 * np.pi, t)
    qc.x([0, 4])
    theta_magic = -2.0 * float(np.arctan(np.sqrt(2.0)))
    angles = {0: theta_magic, 1: theta_magic, 3: theta_magic, 5: theta_magic, 7: theta_magic}
    _append_gray_multiplexed_rot(qc, target=4, controls=(0, 1, 2, 3), angles_by_ctrl_val=angles, axis="y")
    return qc


def casimir_eigenvalues_in_structured_basis() -> np.ndarray:
    """Exact 32-element Casimir vector c_2 in the structured V_PW basis (pure dyadic rationals)."""
    c2 = np.zeros(N_LINK_STATES, dtype=float)
    c2[0] = 0.0
    for idx in (1, 3, 5, 7):
        c2[idx] = 0.75
    for idx in (2, 4, 6):
        c2[idx] = 1.5
    for idx in range(16, 32):
        if idx & 1:
            c2[idx] = 1.875
        elif idx in (16, 30):
            c2[idx] = 2.25
        else:
            c2[idx] = 1.5
    return c2


def electric_propagator_circuit_structured_5q(dt: float, g_ym_sq: float):
    """Structured 5-qubit electric propagator U_E(dt) = V_PW^dagger D_E^{Cas}(dt) V_PW."""
    if g_ym_sq <= 0.0:
        raise ValueError(f"g_ym_sq must be positive, got {g_ym_sq}")
    from qiskit import QuantumCircuit

    v_pw = casimir_peter_weyl_circuit_5q()
    c2_vec = casimir_eigenvalues_in_structured_basis()
    phases_rad = -0.5 * float(g_ym_sq) * c2_vec * float(dt)

    qc = QuantumCircuit(N_LINK_QUBITS, name="bt2t_ue_structured_5q")
    qc.compose(v_pw, inplace=True)
    _append_sparse_walsh_diagonal_5q(qc, phases_rad)
    qc.compose(v_pw.inverse(), inplace=True)
    return qc


def electric_propagator_circuit_qft_5q(dt: float, g_ym_sq: float):
    """Canonical Peter-Weyl QFT electric propagator U_E(dt) = F_tilde_{2T}^dagger D_E(dt) F_tilde_{2T}."""
    if g_ym_sq <= 0.0:
        raise ValueError(f"g_ym_sq must be positive, got {g_ym_sq}")
    from qiskit import QuantumCircuit

    f_qc = bt2t_qft.bt2t_qft_circuit()
    phases_rad = -0.5 * float(g_ym_sq) * casimir_diagonal_peter_weyl_32() * float(dt)

    qc = QuantumCircuit(N_LINK_QUBITS, name="bt2t_ue_peter_weyl_qft_5q")
    qc.compose(f_qc, inplace=True)
    _append_sparse_walsh_diagonal_5q(qc, phases_rad)
    qc.compose(f_qc.inverse(), inplace=True)
    return qc


def strang_step_circuit_structured_5q(dt: float, g_ym_sq: float):
    """5-qubit structured Strang splitting step S_2(dt) = U_B(dt/2) U_E(dt) U_B(dt/2)."""
    from qiskit import QuantumCircuit

    ub_half = magnetic_propagator_circuit_5q(0.5 * float(dt), g_ym_sq)
    ue = electric_propagator_circuit_structured_5q(dt, g_ym_sq)
    qc = QuantumCircuit(N_LINK_QUBITS, name="bt2t_strang_structured_5q")
    qc.compose(ub_half, inplace=True)
    qc.compose(ue, inplace=True)
    qc.compose(ub_half, inplace=True)
    return qc


def strang_step_circuit_qft_5q(dt: float, g_ym_sq: float):
    """5-qubit Strang step using dense QSD Peter-Weyl QFT F_tilde_{2T} from bt2t_qft.py."""
    from qiskit import QuantumCircuit

    ub_half = magnetic_propagator_circuit_5q(0.5 * float(dt), g_ym_sq)
    ue_qft = electric_propagator_circuit_qft_5q(dt, g_ym_sq)
    qc = QuantumCircuit(N_LINK_QUBITS, name="bt2t_strang_qsd_qft_5q")
    qc.compose(ub_half, inplace=True)
    qc.compose(ue_qft, inplace=True)
    qc.compose(ub_half, inplace=True)
    return qc


def dense_unitary_step_circuit_5q(dt: float, g_ym_sq: float, kind: str = "exact"):
    """Generic 5-qubit 32x32 dense UnitaryGate synthesis for exp(-i H_KS dt) or S_2(dt)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate

    if kind == "exact":
        mat = exact_propagator_matrix_32(dt, g_ym_sq)
    elif kind == "strang":
        mat = strang_step_matrix_32(dt, g_ym_sq)
    else:
        raise ValueError(f"Unknown dense step kind: {kind}")

    qc = QuantumCircuit(N_LINK_QUBITS, name=f"bt2t_dense_{kind}_5q")
    qc.append(UnitaryGate(mat), list(range(N_LINK_QUBITS)))
    return qc


def holonomy_2link_circuit_15q():
    """15-qubit reversible 2-link digon holonomy Qiskit circuit |g_1, g_2, y> -> |g_1, g_2, g_1 g_2^{-1} y>."""
    from qiskit import QuantumCircuit

    qc_inv = bt2t_base.inversion_circuit_structured()
    qc_mul = bt2t_mul.multiplication_circuit_structured()
    qc = QuantumCircuit(15, name="bt2t_holonomy_2link_15q")
    qc.compose(qc_inv, qubits=list(range(5, 10)), inplace=True)
    qc.compose(qc_mul, qubits=list(range(5, 10)) + list(range(10, 15)), inplace=True)
    qc.compose(qc_mul, qubits=list(range(0, 5)) + list(range(10, 15)), inplace=True)
    qc.compose(qc_inv, qubits=list(range(5, 10)), inplace=True)
    return qc


def magnetic_step_2link_circuit_15q(dt: float, g_ym_sq: float):
    """15-qubit phase-kickback magnetic propagator U_B^{2-link}(dt) on |g_1, g_2, e>."""
    from qiskit import QuantumCircuit

    qc_inv = bt2t_base.inversion_circuit_structured()
    qc_mul = bt2t_mul.multiplication_circuit_structured()
    qc_ub = magnetic_propagator_circuit_5q(dt, g_ym_sq)

    qc = QuantumCircuit(15, name="bt2t_ub_2link_15q")
    qc.compose(qc_inv, qubits=list(range(5, 10)), inplace=True)
    qc.compose(qc_mul, qubits=list(range(5, 10)) + list(range(10, 15)), inplace=True)
    qc.compose(qc_mul, qubits=list(range(0, 5)) + list(range(10, 15)), inplace=True)
    qc.compose(qc_ub, qubits=list(range(10, 15)), inplace=True)
    qc.compose(qc_mul.inverse(), qubits=list(range(0, 5)) + list(range(10, 15)), inplace=True)
    qc.compose(qc_mul.inverse(), qubits=list(range(5, 10)) + list(range(10, 15)), inplace=True)
    qc.compose(qc_inv, qubits=list(range(5, 10)), inplace=True)
    return qc


def holonomy_4link_circuit_25q():
    """25-qubit reversible 4-link plaquette holonomy circuit |g_1,g_2,g_3,g_4,y> -> |..., g_1 g_2 g_3^{-1} g_4^{-1} y>."""
    from qiskit import QuantumCircuit

    qc_inv = bt2t_base.inversion_circuit_structured()
    qc_mul = bt2t_mul.multiplication_circuit_structured()
    qc = QuantumCircuit(25, name="bt2t_holonomy_4link_25q")
    qc.compose(qc_inv, qubits=list(range(10, 15)), inplace=True)
    qc.compose(qc_inv, qubits=list(range(15, 20)), inplace=True)
    for link_off in (15, 10, 5, 0):
        qc.compose(qc_mul, qubits=list(range(link_off, link_off + 5)) + list(range(20, 25)), inplace=True)
    qc.compose(qc_inv, qubits=list(range(10, 15)), inplace=True)
    qc.compose(qc_inv, qubits=list(range(15, 20)), inplace=True)
    return qc


def magnetic_step_4link_circuit_25q(dt: float, g_ym_sq: float):
    """25-qubit 4-link + holonomy-ancilla magnetic phase-kickback circuit U_B^{4-link}(dt)."""
    from qiskit import QuantumCircuit

    qc_inv = bt2t_base.inversion_circuit_structured()
    qc_mul = bt2t_mul.multiplication_circuit_structured()
    qc_mul_inv = qc_mul.inverse()
    qc_ub = magnetic_propagator_circuit_5q(dt, g_ym_sq)

    qc = QuantumCircuit(25, name="bt2t_ub_4link_25q")
    qc.compose(qc_inv, qubits=list(range(10, 15)), inplace=True)
    qc.compose(qc_inv, qubits=list(range(15, 20)), inplace=True)
    for link_off in (15, 10, 5, 0):
        qc.compose(qc_mul, qubits=list(range(link_off, link_off + 5)) + list(range(20, 25)), inplace=True)
    qc.compose(qc_ub, qubits=list(range(20, 25)), inplace=True)
    for link_off in (0, 5, 10, 15):
        qc.compose(qc_mul_inv, qubits=list(range(link_off, link_off + 5)) + list(range(20, 25)), inplace=True)
    qc.compose(qc_inv, qubits=list(range(10, 15)), inplace=True)
    qc.compose(qc_inv, qubits=list(range(15, 20)), inplace=True)
    return qc


def strang_step_4link_circuit_25q(dt: float, g_ym_sq: float):
    """25-qubit unfixed 4-link + holonomy-ancilla Strang step S_2^{25Q}(dt)."""
    from qiskit import QuantumCircuit

    ub_half = magnetic_step_4link_circuit_25q(0.5 * float(dt), g_ym_sq)
    ue_link = electric_propagator_circuit_structured_5q(0.25 * float(dt), g_ym_sq)

    qc = QuantumCircuit(25, name="bt2t_strang_4link_25q")
    qc.compose(ub_half, inplace=True)
    for link_off in (0, 5, 10, 15):
        qc.compose(ue_link, qubits=list(range(link_off, link_off + 5)), inplace=True)
    qc.compose(ub_half, inplace=True)
    return qc


def strang_step_4link_inplace_circuit_20q(dt: float, g_ym_sq: float):
    """20-qubit zero-ancilla unfixed 4-link Strang step accumulating holonomy in-place on link 4."""
    from qiskit import QuantumCircuit

    qc_inv = bt2t_base.inversion_circuit_structured()
    qc_mul = bt2t_mul.multiplication_circuit_structured()
    qc_mul_inv = qc_mul.inverse()
    qc_ub_half = magnetic_propagator_circuit_5q(0.5 * float(dt), g_ym_sq)
    ue_link = electric_propagator_circuit_structured_5q(0.25 * float(dt), g_ym_sq)

    def _ub_inplace_20q():
        sub = QuantumCircuit(20, name="bt2t_ub_inplace_20q")
        sub.compose(qc_inv, qubits=list(range(10, 15)), inplace=True)
        sub.compose(qc_inv, qubits=list(range(15, 20)), inplace=True)
        for link_off in (10, 5, 0):
            sub.compose(qc_mul, qubits=list(range(link_off, link_off + 5)) + list(range(15, 20)), inplace=True)
        sub.compose(qc_ub_half, qubits=list(range(15, 20)), inplace=True)
        for link_off in (0, 5, 10):
            sub.compose(qc_mul_inv, qubits=list(range(link_off, link_off + 5)) + list(range(15, 20)), inplace=True)
        sub.compose(qc_inv, qubits=list(range(10, 15)), inplace=True)
        sub.compose(qc_inv, qubits=list(range(15, 20)), inplace=True)
        return sub

    ub20 = _ub_inplace_20q()
    qc = QuantumCircuit(20, name="bt2t_strang_4link_inplace_20q")
    qc.compose(ub20, inplace=True)
    for link_off in (0, 5, 10, 15):
        qc.compose(ue_link, qubits=list(range(link_off, link_off + 5)), inplace=True)
    qc.compose(ub20, inplace=True)
    return qc


def _phase_aligned_2norm_error(target: np.ndarray, candidate: np.ndarray) -> float:
    overlap = np.vdot(target.flatten(), candidate.flatten())
    if abs(overlap) < 1e-12:
        return float(np.linalg.norm(candidate - target, ord=2))
    phase = overlap / abs(overlap)
    return float(np.linalg.norm(candidate * np.conj(phase) - target, ord=2))


def benchmark_native_compilations(
    dt: float = 0.2,
    g_ym_sq: float = 1.0,
    seed: int = 20261124,
) -> Dict[str, object]:
    """Verify exact circuit unitaries and compile 5Q/20Q/25Q circuits to native CZ + SX + RZ and ZZPhase."""
    if importlib.util.find_spec("qiskit") is None:
        return {
            "status": "SKIPPED_NO_QISKIT",
            "qiskit_available": False,
            "pytket_available": False,
            "checks": {},
        }

    from qiskit import qasm2, transpile
    from qiskit.quantum_info import Operator, Statevector
    from qiskit.transpiler import CouplingMap

    ub_mat = magnetic_propagator_matrix_32(dt, g_ym_sq)
    ue_mat = electric_propagator_matrix_32(dt, g_ym_sq)
    s2_mat = strang_step_matrix_32(dt, g_ym_sq)
    ex_mat = exact_propagator_matrix_32(dt, g_ym_sq)

    qc_ub = magnetic_propagator_circuit_5q(dt, g_ym_sq)
    qc_ue_str = electric_propagator_circuit_structured_5q(dt, g_ym_sq)
    qc_ue_qft = electric_propagator_circuit_qft_5q(dt, g_ym_sq)
    qc_s2_str = strang_step_circuit_structured_5q(dt, g_ym_sq)
    qc_s2_qft = strang_step_circuit_qft_5q(dt, g_ym_sq)
    qc_dense_ex = dense_unitary_step_circuit_5q(dt, g_ym_sq, kind="exact")
    qc_dense_s2 = dense_unitary_step_circuit_5q(dt, g_ym_sq, kind="strang")

    ub_circuit_2norm_err = float(np.linalg.norm(Operator(qc_ub).data - ub_mat, ord=2))
    ue_str_2norm_err = float(np.linalg.norm(Operator(qc_ue_str).data - ue_mat, ord=2))
    ue_qft_2norm_err = float(np.linalg.norm(Operator(qc_ue_qft).data - ue_mat, ord=2))
    ue_str_vs_qft_2norm_err = float(
        np.linalg.norm(Operator(qc_ue_str).data - Operator(qc_ue_qft).data, ord=2)
    )
    s2_str_2norm_err = float(np.linalg.norm(Operator(qc_s2_str).data - s2_mat, ord=2))

    # Verify 15Q 2-link phase-kickback magnetic circuit via exact 32,768-state Statevector evolution:
    qc_ub15 = magnetic_step_2link_circuit_15q(dt, g_ym_sq)
    rng = np.random.default_rng(seed)
    tbl = np.asarray(bt2t_mul.mul_table_24(), dtype=np.int32)
    inv = np.asarray(bt2t_base.inverse_from_table(tbl.tolist()), dtype=np.int32)
    chi24 = fundamental_character_24()
    sv15_in = np.zeros(1 << 15, dtype=complex)
    amps_24x24 = rng.normal(size=(24, 24)) + 1j * rng.normal(size=(24, 24))
    amps_24x24 /= np.linalg.norm(amps_24x24)
    for g1 in range(24):
        for g2 in range(24):
            sv15_in[g1 | (g2 << 5)] = amps_24x24[g1, g2]
    sv15_out = Statevector(sv15_in).evolve(qc_ub15).data
    sv15_expected = np.zeros_like(sv15_in)
    for g1 in range(24):
        for g2 in range(24):
            hol = int(tbl[g1, inv[g2]])
            sv15_expected[g1 | (g2 << 5)] = amps_24x24[g1, g2] * np.exp(
                1j * (float(dt) / float(g_ym_sq)) * chi24[hol]
            )
    ub15_sv_l2_err = float(np.linalg.norm(sv15_out - sv15_expected))
    ub15_ancilla_leak = float(
        np.sum(np.abs(sv15_out.reshape(32, 1024)[1:, :]) ** 2)
    )

    compilations_5q: List[Dict[str, object]] = []
    cz_5q: Dict[str, int] = {}
    five_q_specs = (
        ("ub_structured_5q", qc_ub, ub_mat),
        ("ue_structured_5q", qc_ue_str, ue_mat),
        ("strang_structured_5q", qc_s2_str, s2_mat),
        ("strang_qsd_qft_5q", qc_s2_qft, s2_mat),
        ("dense_unitary_exact_5q", qc_dense_ex, ex_mat),
        ("dense_unitary_strang_5q", qc_dense_s2, s2_mat),
    )
    for cname, qc_obj, ref_mat in five_q_specs:
        for tlabel, cmap in (
            ("all_to_all", None),
            ("heavy_hex_T5", CouplingMap(HEAVY_HEX_5Q)),
        ):
            tr = transpile(
                qc_obj,
                basis_gates=["rz", "sx", "x", "cz"],
                coupling_map=cmap,
                optimization_level=3,
                seed_transpiler=seed,
            )
            u_tr = Operator.from_circuit(tr).data
            err_tr = _phase_aligned_2norm_error(ref_mat, u_tr)
            cz_val = int(tr.count_ops().get("cz", 0))
            cz_5q[f"{cname}_{tlabel}"] = cz_val
            compilations_5q.append(
                {
                    "execution_tier": "PROXY",
                    "provider_job_id": None,
                    "toolchain": "qiskit",
                    "circuit": cname,
                    "num_qubits": 5,
                    "target": tlabel,
                    "optimization_level": 3,
                    "seed": seed,
                    "two_qubit_gates": {"cz": cz_val},
                    "depth": int(tr.depth()),
                    "size": int(tr.size()),
                    "phase_aligned_2norm_error": err_tr,
                    "equivalent_up_to_global_phase": err_tr < 1e-9,
                }
            )

    qc_s2_20q = strang_step_4link_inplace_circuit_20q(dt, g_ym_sq)
    qc_s2_25q = strang_step_4link_circuit_25q(dt, g_ym_sq)

    compilations_multilink: List[Dict[str, object]] = []
    cz_multi: Dict[str, int] = {}
    for cname, qc_obj, hh_edges in (
        ("strang_4link_inplace_20q", qc_s2_20q, HEAVY_HEX_20Q),
        ("strang_4link_ancilla_25q", qc_s2_25q, HEAVY_HEX_25Q),
    ):
        for tlabel, cmap in (
            ("all_to_all", None),
            (f"heavy_hex_{qc_obj.num_qubits}Q", CouplingMap(hh_edges)),
        ):
            tr = transpile(
                qc_obj,
                basis_gates=["rz", "sx", "x", "cz"],
                coupling_map=cmap,
                optimization_level=3,
                seed_transpiler=seed,
            )
            cz_val = int(tr.count_ops().get("cz", 0))
            cz_multi[f"{cname}_{tlabel}"] = cz_val
            compilations_multilink.append(
                {
                    "execution_tier": "PROXY",
                    "provider_job_id": None,
                    "toolchain": "qiskit",
                    "circuit": cname,
                    "num_qubits": int(qc_obj.num_qubits),
                    "target": tlabel,
                    "optimization_level": 3,
                    "seed": seed,
                    "two_qubit_gates": {"cz": cz_val},
                    "depth": int(tr.depth()),
                    "size": int(tr.size()),
                }
            )

    pytket_available = importlib.util.find_spec("pytket") is not None
    compilations_pytket: List[Dict[str, object]] = []
    zzphase_5q: Dict[str, int] = {}
    if pytket_available:
        from pytket.circuit import OpType
        from pytket.passes import AutoRebase, FullPeepholeOptimise
        from pytket.qasm import circuit_from_qasm_str

        rev_idx = [bt2t_base.reverse_bits(x) for x in range(N_LINK_STATES)]
        for cname, qc_obj, ref_mat in (
            ("ub_structured_5q", qc_ub, ub_mat),
            ("strang_structured_5q", qc_s2_str, s2_mat),
            ("dense_unitary_exact_5q", qc_dense_ex, ex_mat),
        ):
            qc_u3 = transpile(
                qc_obj, basis_gates=["u3", "cx"], optimization_level=3, seed_transpiler=seed
            )
            tk_circ = circuit_from_qasm_str(qasm2.dumps(qc_u3))
            FullPeepholeOptimise().apply(tk_circ)
            AutoRebase({OpType.ZZPhase, OpType.PhasedX, OpType.Rz}).apply(tk_circ)
            u_tk = tk_circ.get_unitary()
            u_qiskit_order = u_tk[np.ix_(rev_idx, rev_idx)]
            tk_err = _phase_aligned_2norm_error(ref_mat, u_qiskit_order)
            zz_cnt = int(tk_circ.n_gates_of_type(OpType.ZZPhase))
            zzphase_5q[cname] = zz_cnt
            compilations_pytket.append(
                {
                    "execution_tier": "PROXY",
                    "provider_job_id": None,
                    "toolchain": "pytket",
                    "circuit": cname,
                    "num_qubits": 5,
                    "FullPeepholeOptimise": True,
                    "two_qubit_gates": {"ZZPhase": zz_cnt},
                    "depth": int(tk_circ.depth()),
                    "size": int(tk_circ.n_gates),
                    "phase_aligned_2norm_error": tk_err,
                    "equivalent_up_to_global_phase": tk_err < 1e-9,
                }
            )

    cz_s2_5q_ata = cz_5q["strang_structured_5q_all_to_all"]
    cz_s2_5q_t5 = cz_5q["strang_structured_5q_heavy_hex_T5"]
    cz_dense_ata = cz_5q["dense_unitary_exact_5q_all_to_all"]
    cz_dense_t5 = cz_5q["dense_unitary_exact_5q_heavy_hex_T5"]
    cz_qft_ata = cz_5q["strang_qsd_qft_5q_all_to_all"]
    cz_20q_ata = cz_multi["strang_4link_inplace_20q_all_to_all"]
    cz_25q_ata = cz_multi["strang_4link_ancilla_25q_all_to_all"]
    cz_25q_hh = cz_multi["strang_4link_ancilla_25q_heavy_hex_25Q"]

    ratio_dense_ata = cz_dense_ata / float(cz_s2_5q_ata)
    ratio_dense_t5 = cz_dense_t5 / float(cz_s2_5q_t5)
    ratio_25q_ata = cz_25q_ata / float(cz_s2_5q_ata)
    ratio_25q_hh = cz_25q_hh / float(cz_s2_5q_t5)
    ratio_20q_ata = cz_20q_ata / float(cz_s2_5q_ata)

    checks = {
        "ub_5q_circuit_exact_below_1e14": ub_circuit_2norm_err < 1e-14,
        "ue_structured_5q_circuit_exact_below_1e14": ue_str_2norm_err < 1e-14,
        "ue_qft_5q_circuit_exact_below_1e12": ue_qft_2norm_err < 1e-12,
        "ue_structured_matches_canonical_qft_below_1e12": ue_str_vs_qft_2norm_err < 1e-12,
        "strang_structured_5q_circuit_exact_below_1e14": s2_str_2norm_err < 1e-14,
        "ub_2link_15q_statevector_exact_and_zero_ancilla_leak": (
            ub15_sv_l2_err < 1e-13 and ub15_ancilla_leak < 1e-28
        ),
        "all_5q_qiskit_compilations_unitary_equivalent": all(
            bool(c["equivalent_up_to_global_phase"]) for c in compilations_5q
        ),
        "structured_5q_fewer_cz_than_dense_unitary": (
            cz_s2_5q_ata < cz_dense_ata and cz_s2_5q_t5 < cz_dense_t5
        ),
        "gauge_fixed_5q_over_10x_cz_reduction_vs_unfixed_4link": (
            ratio_25q_ata > 10.0 and ratio_25q_hh > 10.0 and ratio_20q_ata > 10.0
        ),
    }
    if pytket_available:
        checks["all_pytket_compilations_unitary_equivalent"] = all(
            bool(c["equivalent_up_to_global_phase"]) for c in compilations_pytket
        )
        checks["structured_5q_fewer_zzphase_than_dense_unitary"] = (
            zzphase_5q["strang_structured_5q"] < zzphase_5q["dense_unitary_exact_5q"]
        )

    return {
        "status": "OK",
        "qiskit_available": True,
        "pytket_available": pytket_available,
        "dt": float(dt),
        "g_ym_sq": float(g_ym_sq),
        "direct_circuit_errors": {
            "ub_5q_2norm_error": ub_circuit_2norm_err,
            "ue_structured_5q_2norm_error": ue_str_2norm_err,
            "ue_peter_weyl_qft_5q_2norm_error": ue_qft_2norm_err,
            "ue_structured_vs_qft_2norm_error": ue_str_vs_qft_2norm_err,
            "strang_structured_5q_2norm_error": s2_str_2norm_err,
            "ub_2link_15q_statevector_l2_error": ub15_sv_l2_err,
            "ub_2link_15q_ancilla_leakage": ub15_ancilla_leak,
        },
        "gate_count_summary": {
            "cz_ub_structured_5q_all_to_all": cz_5q["ub_structured_5q_all_to_all"],
            "cz_ub_structured_5q_heavy_hex_T5": cz_5q["ub_structured_5q_heavy_hex_T5"],
            "cz_ue_structured_5q_all_to_all": cz_5q["ue_structured_5q_all_to_all"],
            "cz_ue_structured_5q_heavy_hex_T5": cz_5q["ue_structured_5q_heavy_hex_T5"],
            "cz_strang_structured_5q_all_to_all": cz_s2_5q_ata,
            "cz_strang_structured_5q_heavy_hex_T5": cz_s2_5q_t5,
            "cz_strang_qsd_qft_5q_all_to_all": cz_qft_ata,
            "cz_strang_qsd_qft_5q_heavy_hex_T5": cz_5q["strang_qsd_qft_5q_heavy_hex_T5"],
            "cz_dense_unitary_5q_all_to_all": cz_dense_ata,
            "cz_dense_unitary_5q_heavy_hex_T5": cz_dense_t5,
            "cz_strang_4link_inplace_20q_all_to_all": cz_20q_ata,
            "cz_strang_4link_inplace_20q_heavy_hex_20Q": cz_multi[
                "strang_4link_inplace_20q_heavy_hex_20Q"
            ],
            "cz_strang_4link_ancilla_25q_all_to_all": cz_25q_ata,
            "cz_strang_4link_ancilla_25q_heavy_hex_25Q": cz_25q_hh,
            "zzphase_ub_structured_5q": zzphase_5q.get("ub_structured_5q"),
            "zzphase_strang_structured_5q": zzphase_5q.get("strang_structured_5q"),
            "zzphase_dense_unitary_5q": zzphase_5q.get("dense_unitary_exact_5q"),
            "cz_compression_dense_vs_structured_all_to_all": round(ratio_dense_ata, 4),
            "cz_compression_dense_vs_structured_heavy_hex_T5": round(ratio_dense_t5, 4),
            "cz_compression_unfixed_20q_vs_5q_all_to_all": round(ratio_20q_ata, 4),
            "cz_compression_unfixed_25q_vs_5q_all_to_all": round(ratio_25q_ata, 4),
            "cz_compression_unfixed_25q_vs_5q_heavy_hex": round(ratio_25q_hh, 4),
        },
        "proxy_compilations_5q": compilations_5q,
        "proxy_compilations_multilink": compilations_multilink,
        "proxy_compilations_pytket": compilations_pytket,
        "checks": checks,
    }
