#!/usr/bin/env python3
"""Deterministic offline verification & manifest generator for OSIRIS-BT2T-002 (Milestone 2).

Verifies:
  Part 1 — Exact 10-qubit reversible 2T left-multiplication (generic cycle reference vs structured
           Q_8 |>< Z_3 semidirect synthesis): frozen bt2t-enc-1 digest, classical 1024-state bit
           simulation, exact 1024x1024 complex Operator matrix equality (including phases), 16+16
           seeded complex 10-qubit superpositions, CZ phase-error positive control, and local PROXY
           transpilation onto rz/sx/x/cz (all-to-all and 10Q heavy-hex fragment).
  Part 2 — Exact 5-qubit non-abelian 2T Quantum Fourier Transform: all 7 irreps from first principles,
           Schur orthogonality & 32x32 Peter-Weyl unitarity (< 1e-14), simultaneous block
           diagonalization of all 24 group translations L_g and R_g (< 1e-14), Plancherel &
           convolution theorem, Hidden Subgroup coset-state sampling across 6 canonical subgroups,
           and local PROXY Qiskit (all-to-all & HEAVY_HEX_5Q) + pytket H2-native compilation.

Execution tier: SIMULATED (exact linear algebra + local PROXY transpilation; provider_job_id = null;
0 QPU seconds).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
if __package__:
    from . import bt2t_mul, bt2t_qft
else:
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import bt2t_mul  # type: ignore[no-redef]
    import bt2t_qft  # type: ignore[no-redef]

SEED = 20261124
DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parent / "manifest_bt2t002.json"

# 5-qubit T-shaped heavy-hex fragment (identical to Milestone 1 verify.HEAVY_HEX_5Q):
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

# 10-qubit heavy-hex fragment (two T5 cells bridged at 3 <-> 5):
HEAVY_HEX_10Q: List[List[int]] = [
    [0, 1],
    [1, 0],
    [1, 2],
    [2, 1],
    [2, 3],
    [3, 2],
    [2, 4],
    [4, 2],
    [3, 5],
    [5, 3],
    [5, 6],
    [6, 5],
    [6, 7],
    [7, 6],
    [7, 8],
    [8, 7],
    [7, 9],
    [9, 7],
]


def _clean_float(val: float) -> float:
    return float(f"{float(val):.12e}")


def permutation_matrix(perm: Sequence[int]) -> np.ndarray:
    """Exact permutation matrix P where column s is the image |perm[s]>."""
    n = len(perm)
    mat = np.zeros((n, n), dtype=complex)
    for s, img in enumerate(perm):
        mat[img, s] = 1.0
    return mat


def _phase_aligned_max_abs_error(target: np.ndarray, candidate: np.ndarray) -> float:
    overlap = np.vdot(target.flatten(), candidate.flatten())
    if abs(overlap) < 1e-12:
        return float(np.max(np.abs(candidate - target)))
    phase = overlap / abs(overlap)
    return float(np.max(np.abs(candidate * np.conj(phase) - target)))


def _unitary_10q_transpiled(qc) -> np.ndarray:
    """Evaluate layout-reconstructed 10Q transpiled circuit unitary matching Operator.from_circuit(qc)."""
    try:
        from qiskit_aer import AerSimulator

        qc_u = qc.copy()
        qc_u.save_unitary()
        res = AerSimulator(method="unitary").run(qc_u).result()
        u_phys = np.asarray(res.get_unitary(qc_u), dtype=complex)
        if qc.layout is not None:
            n = qc.num_qubits
            init_idx = qc.layout.initial_index_layout(filter_ancillas=True)
            final_idx = qc.layout.final_index_layout(filter_ancillas=True)
            in_map = [sum(((s >> q) & 1) << init_idx[q] for q in range(n)) for s in range(1 << n)]
            out_map = [
                sum(((s >> q) & 1) << final_idx[q] for q in range(n)) for s in range(1 << n)
            ]
            return u_phys[np.ix_(out_map, in_map)]
        return u_phys
    except ImportError:
        from qiskit.quantum_info import Operator

        return Operator.from_circuit(qc).data


# ---------------------------------------------------------------------------
# Part 1 Verification: 10-Qubit Reversible 2T Multiplication
# ---------------------------------------------------------------------------


def verify_semidirect_algebra() -> Dict[str, object]:
    """Exhaustively check 2T ~= Q_8 |>< Z_3 semidirect arithmetic against integer quaternion table."""
    rec = bt2t_mul.verify_frozen_encoding()
    tbl = bt2t_mul.mul_table_24()
    inv = bt2t_mul.bt2t_base.inverse_from_table(tbl)
    w = bt2t_mul.OMEGA_INDEX
    w2 = bt2t_mul.OMEGA_SQ_INDEX

    # Check omega order 3 and Ad_omega(i, j, k) == (j, k, i):
    omega_order_3 = tbl[w][w] == w2 and tbl[w2][w] == 0 and w != 0
    ad_matches_conj = all(
        bt2t_mul.ad_omega_q8(q, power=p)
        == tbl[tbl[(0, w, w2)[p]][q]][inv[(0, w, w2)[p]]]
        for p in (0, 1, 2)
        for q in range(8)
    )
    ad_cycles_ijk = (
        bt2t_mul.ad_omega_q8(2, 1) == 4
        and bt2t_mul.ad_omega_q8(4, 1) == 6
        and bt2t_mul.ad_omega_q8(6, 1) == 2
        and bt2t_mul.ad_omega_q8(3, 1) == 5
        and bt2t_mul.ad_omega_q8(5, 1) == 7
        and bt2t_mul.ad_omega_q8(7, 1) == 3
    )

    # Check 3-bit Q_8 cocycle multiplication formula on all 64 pairs:
    q8_formula_exact = all(
        bt2t_mul.q8_multiply_bits(qx, qy) == tbl[qx][qy]
        for qx in range(8)
        for qy in range(8)
    )

    # Check right (g = q * omega^c) and left (g = omega^c * q_left) decompositions on all 24 elements:
    right_pairs = [bt2t_mul.semidirect_decompose(g) for g in range(24)]
    left_pairs = [bt2t_mul.semidirect_decompose_left(g) for g in range(24)]
    right_unique_and_exact = len(set(right_pairs)) == 24 and all(
        tbl[q][(0, w, w2)[c]] == g for g, (c, q) in enumerate(right_pairs)
    )
    left_unique_and_exact = len(set(left_pairs)) == 24 and all(
        tbl[(0, w, w2)[c]][ql] == g for g, (c, ql) in enumerate(left_pairs)
    )

    # Check semidirect product law on all 576 pairs in 2T x 2T:
    semidirect_law_all_576 = True
    for x in range(24):
        cx, qx = right_pairs[x]
        for y in range(24):
            cy, qy = right_pairs[y]
            c_xy = (cx + cy) % 3
            q_xy = bt2t_mul.q8_multiply_bits(qx, bt2t_mul.ad_omega_q8(qy, power=cx))
            c_want, q_want = right_pairs[tbl[x][y]]
            if (c_xy, q_xy) != (c_want, q_want):
                semidirect_law_all_576 = False

    # Check 5-qubit E_sd circuit classically and via exact 32x32 Operator:
    from qiskit.quantum_info import Operator

    esd_perm = bt2t_mul.semidirect_bijection_5q()
    esd_gates = bt2t_mul.semidirect_bijection_gate_list(0)
    esd_classical_ok = [
        bt2t_mul.apply_gate_list_classically(esd_gates, s) for s in range(32)
    ] == esd_perm and esd_perm[24:] == list(range(24, 32))
    esd_op_err = float(
        np.max(np.abs(Operator(bt2t_mul.semidirect_bijection_circuit()).data - permutation_matrix(esd_perm)))
    )

    return {
        "encoding_sha256_verified": rec["sha256"] == bt2t_mul.ENCODING_SHA256,
        "omega_order_3": omega_order_3,
        "ad_omega_matches_quaternion_conjugation": ad_matches_conj and ad_cycles_ijk,
        "q8_3bit_cocycle_exact_all_64": q8_formula_exact,
        "semidirect_right_and_left_bijections_24": right_unique_and_exact and left_unique_and_exact,
        "semidirect_product_law_all_576_pairs": semidirect_law_all_576,
        "esd_5q_classical_and_unitary_exact": esd_classical_ok and esd_op_err < 1e-12,
        "esd_5q_operator_max_abs_error": _clean_float(esd_op_err),
    }


def verify_multiplication_10q(
    u_generic_override: np.ndarray | None = None,
) -> Dict[str, object]:
    """Full Part 1 verification: 1024 basis states, 1024x1024 Operator matrices (including phases),
    complex superpositions, phase-error positive control, and PROXY compilations.
    """
    from qiskit import qasm2, transpile
    from qiskit.quantum_info import Operator, Statevector
    from qiskit.transpiler import CouplingMap

    perm_1024 = bt2t_mul.multiplication_permutation_10q()
    p_1024 = permutation_matrix(perm_1024)

    # Check Cayley bijection on 576 valid pairs and identity on 448 unused pairs:
    tbl = bt2t_mul.mul_table_24()
    cayley_bijection_24 = all(
        sorted(bt2t_mul.mu(x, y, tbl) for y in range(24)) == list(range(24))
        for x in range(24)
    )
    unused_pairs_count = sum(
        1 for x in range(32) for y in range(32) if (x >= 24 or y >= 24)
    )
    unused_fixed_448 = unused_pairs_count == 448 and all(
        perm_1024[bt2t_mul.pack_xy(x, y)] == bt2t_mul.pack_xy(x, y)
        for x in range(32)
        for y in range(32)
        if (x >= 24 or y >= 24)
    )

    # Classical bit-level gate simulation on all 1024 inputs (no quantum library used):
    gates_gen = bt2t_mul.multiplication_gate_list_generic()
    gates_str = bt2t_mul.multiplication_gate_list_structured()
    classical_gen_ok = [
        bt2t_mul.apply_gate_list_classically(gates_gen, s) for s in range(1024)
    ] == perm_1024
    classical_str_ok = [
        bt2t_mul.apply_gate_list_classically(gates_str, s) for s in range(1024)
    ] == perm_1024

    # Qiskit zero-ancilla circuits & pure computational-basis gate audit:
    qc_gen = bt2t_mul.multiplication_circuit_generic()
    qc_str = bt2t_mul.multiplication_circuit_structured()
    allowed_ops = {"x", "cx", "ccx", "mcx", "cswap"}
    pure_basis_gates_only = (
        qc_gen.num_qubits == 10
        and qc_str.num_qubits == 10
        and qc_gen.num_ancillas == 0
        and qc_str.num_ancillas == 0
        and set(qc_gen.count_ops().keys()).issubset(allowed_ops)
        and set(qc_str.count_ops().keys()).issubset(allowed_ops)
    )

    # Exact 1024x1024 Operator complex matrix equality (phases included!):
    u_gen = Operator(qc_gen).data if u_generic_override is None else u_generic_override
    u_str = Operator(qc_str).data
    err_gen = float(np.max(np.abs(u_gen - p_1024)))
    err_str = float(np.max(np.abs(u_str - p_1024)))

    # 16 random complex superpositions on 576 valid pairs + 16 on all 1024 states:
    rng = np.random.default_rng(SEED)
    valid_indices = [
        bt2t_mul.pack_xy(x, y) for x in range(24) for y in range(24)
    ]
    all_indices = list(range(1024))
    max_sup_err_gen = 0.0
    max_sup_err_str = 0.0
    for support in (valid_indices, all_indices):
        for _ in range(16):
            amp = np.zeros(1024, dtype=complex)
            amp[support] = rng.normal(size=len(support)) + 1j * rng.normal(size=len(support))
            amp /= np.linalg.norm(amp)
            expected = np.zeros(1024, dtype=complex)
            for s_idx in range(1024):
                expected[perm_1024[s_idx]] = amp[s_idx]
            out_str = Statevector(amp).evolve(qc_str).data
            out_gen = u_gen @ amp
            max_sup_err_str = max(max_sup_err_str, float(np.max(np.abs(out_str - expected))))
            max_sup_err_gen = max(max_sup_err_gen, float(np.max(np.abs(out_gen - expected))))

    # Positive phase-error injection control: append CZ(0, 5) to structured circuit:
    qc_bad = qc_str.copy(name="bt2t_mul_structured_bad_phase")
    qc_bad.cz(0, 5)
    u_bad = Operator(qc_bad).data
    bad_basis_still_matches = [
        int(np.argmax(np.abs(u_bad[:, s]))) for s in range(1024)
    ] == perm_1024
    bad_matrix_phase_err = float(np.max(np.abs(u_bad - p_1024)))
    phase_control_caught = bad_basis_still_matches and bad_matrix_phase_err > 1.0

    # Local PROXY compilations (rz, sx, x, cz; optimization_level=3):
    compilations: List[Dict[str, object]] = []
    cz_summary: Dict[str, int] = {}
    for cname, qc_obj in (("generic", qc_gen), ("structured", qc_str)):
        for target_label, cmap in (
            ("all_to_all", None),
            ("heavy_hex_fragment_10Q", CouplingMap(HEAVY_HEX_10Q)),
        ):
            tr = transpile(
                qc_obj,
                basis_gates=["rz", "sx", "x", "cz"],
                coupling_map=cmap,
                optimization_level=3,
                seed_transpiler=SEED,
            )
            u_tr = _unitary_10q_transpiled(tr)
            ph_err = _phase_aligned_max_abs_error(p_1024, u_tr)
            cz_cnt = int(tr.count_ops().get("cz", 0))
            cz_summary[f"{cname}_{target_label}"] = cz_cnt
            compilations.append(
                {
                    "execution_tier": "PROXY",
                    "provider_job_id": None,
                    "kind": "PROXY local Qiskit transpile (not hardware output)",
                    "circuit": cname,
                    "target": target_label,
                    "optimization_level": 3,
                    "seed": SEED,
                    "two_qubit_gates": {"cz": cz_cnt},
                    "depth": int(tr.depth()),
                    "size": int(tr.size()),
                    "equivalent_up_to_global_phase": ph_err < 1e-9,
                    "phase_aligned_max_abs_error": _clean_float(ph_err),
                }
            )

    compression_all_to_all = cz_summary["generic_all_to_all"] / float(
        cz_summary["structured_all_to_all"]
    )
    compression_heavy_hex = cz_summary["generic_heavy_hex_fragment_10Q"] / float(
        cz_summary["structured_heavy_hex_fragment_10Q"]
    )

    q2_gen = qasm2.dumps(qc_gen.decompose(reps=2))
    q2_str = qasm2.dumps(qc_str.decompose(reps=2))
    sha = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()  # noqa: E731

    return {
        "logical_circuits": {
            "generic": {
                "num_qubits": qc_gen.num_qubits,
                "num_ancillas": qc_gen.num_ancillas,
                "gate_counts": {k: int(v) for k, v in sorted(qc_gen.count_ops().items())},
                "depth": int(qc_gen.depth()),
                "decomposed_qasm2_sha256": sha(q2_gen),
            },
            "structured": {
                "num_qubits": qc_str.num_qubits,
                "num_ancillas": qc_str.num_ancillas,
                "gate_counts": {k: int(v) for k, v in sorted(qc_str.count_ops().items())},
                "depth": int(qc_str.depth()),
                "decomposed_qasm2_sha256": sha(q2_str),
            },
        },
        "metrics": {
            "valid_pairs_checked": 576,
            "unused_pairs_checked": unused_pairs_count,
            "generic_operator_max_abs_error": _clean_float(err_gen),
            "structured_operator_max_abs_error": _clean_float(err_str),
            "generic_superposition_max_abs_error": _clean_float(max_sup_err_gen),
            "structured_superposition_max_abs_error": _clean_float(max_sup_err_str),
            "phase_control_injected_cz_max_abs_error": _clean_float(bad_matrix_phase_err),
            "cz_generic_all_to_all_opt3": cz_summary["generic_all_to_all"],
            "cz_structured_all_to_all_opt3": cz_summary["structured_all_to_all"],
            "cz_compression_ratio_all_to_all": round(compression_all_to_all, 4),
            "cz_generic_heavy_hex_10q_opt3": cz_summary["generic_heavy_hex_fragment_10Q"],
            "cz_structured_heavy_hex_10q_opt3": cz_summary["structured_heavy_hex_fragment_10Q"],
            "cz_compression_ratio_heavy_hex_10q": round(compression_heavy_hex, 4),
        },
        "proxy_compilations": compilations,
        "checks": {
            "mul_cayley_bijection_all_24_left_translations": cayley_bijection_24,
            "mul_unused_448_pairs_identity": unused_fixed_448,
            "mul_classical_sim_all_1024_generic": classical_gen_ok,
            "mul_classical_sim_all_1024_structured": classical_str_ok,
            "mul_zero_ancillas_pure_basis_gates": pure_basis_gates_only,
            "mul_generic_operator_1024x1024_exact_with_phases": err_gen < 1e-10,
            "mul_structured_operator_1024x1024_exact_with_phases": err_str < 1e-10,
            "mul_superpositions_exact_with_phases": (
                max_sup_err_gen < 1e-10 and max_sup_err_str < 1e-10
            ),
            "mul_positive_phase_error_control_caught": phase_control_caught,
            "mul_structured_dramatically_fewer_cz_than_generic": (
                compression_all_to_all > 10.0 and compression_heavy_hex > 10.0
            ),
            "mul_proxy_compilations_all_equivalent": all(
                bool(c["equivalent_up_to_global_phase"]) for c in compilations
            ),
        },
    }


# ---------------------------------------------------------------------------
# Part 2 Verification: 5-Qubit Non-Abelian 2T QFT
# ---------------------------------------------------------------------------


def verify_qft_5q() -> Dict[str, object]:
    """Full Part 2 verification: 7 irreps, Schur orthogonality, Peter-Weyl unitarity, simultaneous
    block-diagonalization of all 24 translations, subgroup HSP sampling, and PROXY 5Q compilations.
    """
    from qiskit import qasm2, transpile
    from qiskit.quantum_info import Operator
    from qiskit.transpiler import CouplingMap

    irrep_report = bt2t_qft.verify_irreps_and_schur_orthogonality()
    block_report = bt2t_qft.verify_simultaneous_block_diagonalization()
    hsp_report = bt2t_qft.verify_subgroup_coset_sampling()

    f32 = bt2t_qft.bt2t_qft_matrix_32()
    qc_qft = bt2t_qft.bt2t_qft_circuit()
    op_qft = Operator(qc_qft)
    direct_circuit_err = float(np.max(np.abs(op_qft.data - f32)))

    compilations: List[Dict[str, object]] = []
    for target_label, cmap in (
        ("all_to_all", None),
        ("heavy_hex_fragment_T5", CouplingMap(HEAVY_HEX_5Q)),
    ):
        for opt_level in (1, 3):
            tr = transpile(
                qc_qft,
                basis_gates=["rz", "sx", "x", "cz"],
                coupling_map=cmap,
                optimization_level=opt_level,
                seed_transpiler=SEED,
            )
            op_tr = Operator.from_circuit(tr)
            equiv = bool(op_tr.equiv(op_qft, atol=1e-9))
            ph_err = _phase_aligned_max_abs_error(f32, op_tr.data)
            compilations.append(
                {
                    "execution_tier": "PROXY",
                    "provider_job_id": None,
                    "kind": "PROXY local Qiskit transpile (not hardware output)",
                    "circuit": "bt2t_qft_5q",
                    "target": target_label,
                    "optimization_level": opt_level,
                    "seed": SEED,
                    "two_qubit_gates": {"cz": int(tr.count_ops().get("cz", 0))},
                    "depth": int(tr.depth()),
                    "size": int(tr.size()),
                    "equivalent_up_to_global_phase": equiv and ph_err < 1e-9,
                    "phase_aligned_max_abs_error": _clean_float(ph_err),
                }
            )

    try:
        from pytket.circuit import OpType
        from pytket.passes import AutoRebase, FullPeepholeOptimise
        from pytket.qasm import circuit_from_qasm_str

        qc_u3_cx = transpile(
            qc_qft, basis_gates=["u3", "cx"], optimization_level=3, seed_transpiler=SEED
        )
        for opt_flag in (False, True):
            tk = circuit_from_qasm_str(qasm2.dumps(qc_u3_cx))
            if opt_flag:
                FullPeepholeOptimise().apply(tk)
            AutoRebase({OpType.ZZPhase, OpType.PhasedX, OpType.Rz}).apply(tk)
            u_tk = tk.get_unitary()
            rev = [bt2t_mul.bt2t_base.reverse_bits(x) for x in range(32)]
            u_q = u_tk[np.ix_(rev, rev)]
            ph_err_tk = _phase_aligned_max_abs_error(f32, u_q)
            compilations.append(
                {
                    "execution_tier": "PROXY",
                    "provider_job_id": None,
                    "kind": "PROXY local pytket H2-native rebase (not Nexus)",
                    "circuit": "bt2t_qft_5q",
                    "FullPeepholeOptimise": opt_flag,
                    "two_qubit_gates": {"ZZPhase": int(tk.n_gates_of_type(OpType.ZZPhase))},
                    "depth": int(tk.depth()),
                    "size": int(tk.n_gates),
                    "equivalent_up_to_global_phase": ph_err_tk < 1e-9,
                    "phase_aligned_max_abs_error": _clean_float(ph_err_tk),
                }
            )
    except ImportError:
        compilations.append(
            {
                "execution_tier": "PROXY",
                "provider_job_id": None,
                "kind": "PROXY local pytket H2-native rebase",
                "status": "SKIPPED: pytket not installed",
            }
        )

    q2_qft = qasm2.dumps(
        transpile(qc_qft, basis_gates=["u3", "cx"], optimization_level=1, seed_transpiler=SEED)
    )
    qft_sha256 = hashlib.sha256(q2_qft.encode("utf-8")).hexdigest()

    qiskit_opt3 = {
        c["target"]: c
        for c in compilations
        if c.get("kind", "").startswith("PROXY local Qiskit") and c.get("optimization_level") == 3
    }

    return {
        "irrep_dimensions": bt2t_qft.IRREP_DIMS,
        "schur_and_unitarity": {
            "sum_d_alpha_squared": irrep_report["sum_d_alpha_squared"],
            "irrep_unitary_max_inf_errors": {
                k: _clean_float(v)
                for k, v in irrep_report["irrep_unitary_max_inf_errors"].items()  # type: ignore[union-attr]
            },
            "irrep_homomorphism_max_abs_errors": {
                k: _clean_float(v)
                for k, v in irrep_report["irrep_homomorphism_max_abs_errors"].items()  # type: ignore[union-attr]
            },
            "character_orthogonality_inf_error": _clean_float(
                irrep_report["character_orthogonality_inf_error"]  # type: ignore[arg-type]
            ),
            "f24_unitarity_inf_error": _clean_float(
                irrep_report["f24_unitarity_inf_error"]  # type: ignore[arg-type]
            ),
            "f32_unitarity_inf_error": _clean_float(
                irrep_report["f32_unitarity_inf_error"]  # type: ignore[arg-type]
            ),
        },
        "block_diagonalization_and_convolution": {
            "block_sizes_contiguous": block_report["block_sizes_contiguous"],
            "irrep_unique_dims": block_report["irrep_unique_dims"],
            "left_regular_max_off_block_abs": _clean_float(
                block_report["left_regular_max_off_block_abs"]  # type: ignore[arg-type]
            ),
            "left_regular_max_on_block_error": _clean_float(
                block_report["left_regular_max_on_block_error"]  # type: ignore[arg-type]
            ),
            "right_regular_max_off_irrep_abs": _clean_float(
                block_report["right_regular_max_off_irrep_abs"]  # type: ignore[arg-type]
            ),
            "right_regular_max_on_block_error": _clean_float(
                block_report["right_regular_max_on_block_error"]  # type: ignore[arg-type]
            ),
            "plancherel_norm_error": _clean_float(
                block_report["plancherel_norm_error"]  # type: ignore[arg-type]
            ),
            "nonabelian_convolution_max_error": _clean_float(
                block_report["nonabelian_convolution_max_error"]  # type: ignore[arg-type]
            ),
        },
        "subgroup_coset_hsp_sampling": {
            "q8_1d_total_weight": round(float(hsp_report["q8_1d_total_weight"]), 12),  # type: ignore[arg-type]
            "q8_2d_3d_leakage": _clean_float(float(hsp_report["q8_2d_3d_leakage"])),  # type: ignore[arg-type]
            "subgroups": {
                sg: {
                    "order": info["order"],
                    "elements": info["elements"],
                    "trivial_irrep_multiplicities": info["trivial_irrep_multiplicities"],
                    "expected_irrep_weights": info["expected_irrep_weights"],
                    "measured_irrep_weights_coset_eH": info["measured_irrep_weights_coset_eH"],
                    "max_coset_weight_abs_error_all_24_g": _clean_float(
                        info["max_coset_weight_abs_error_all_24_g"]
                    ),
                    "max_forbidden_irrep_leakage": _clean_float(
                        info["max_forbidden_irrep_leakage"]
                    ),
                    "max_unused_codeword_leakage": _clean_float(
                        info["max_unused_codeword_leakage"]
                    ),
                    "passed": info["passed"],
                }
                for sg, info in hsp_report["subgroups"].items()  # type: ignore[union-attr]
            },
        },
        "logical_circuit": {
            "num_qubits": qc_qft.num_qubits,
            "gate_counts": {k: int(v) for k, v in sorted(qc_qft.count_ops().items())},
            "depth": int(qc_qft.depth()),
            "direct_operator_max_abs_error": _clean_float(direct_circuit_err),
            "u3_cx_qasm2_sha256": qft_sha256,
            "cz_all_to_all_opt3": qiskit_opt3["all_to_all"]["two_qubit_gates"]["cz"],  # type: ignore[index]
            "depth_all_to_all_opt3": qiskit_opt3["all_to_all"]["depth"],
            "cz_heavy_hex_T5_opt3": qiskit_opt3["heavy_hex_fragment_T5"]["two_qubit_gates"]["cz"],  # type: ignore[index]
            "depth_heavy_hex_T5_opt3": qiskit_opt3["heavy_hex_fragment_T5"]["depth"],
        },
        "proxy_compilations": compilations,
        "checks": {
            "qft_sum_irrep_dims_squared_is_24": bool(irrep_report["sum_d_alpha_squared_equals_24"]),
            "qft_all_7_irreps_unitary": bool(irrep_report["all_irreps_unitary"]),
            "qft_all_7_irreps_homomorphisms_576_pairs": bool(
                irrep_report["all_irreps_homomorphisms_all_576_pairs"]
            ),
            "qft_rho3_so3_real_orthogonal": bool(irrep_report["rho3_so3_real_orthogonal"]),
            "qft_characters_orthonormal_and_distinct": bool(
                irrep_report["characters_orthonormal_and_distinct"]
            ),
            "qft_f32_unitarity_below_1e14": bool(irrep_report["f32_strictly_unitary_below_1e14"]),
            "qft_unused_codewords_exact_identity": bool(
                irrep_report["unused_codewords_exact_identity"]
            ),
            "qft_all_24_translations_block_diagonalized_below_1e14": bool(
                block_report["all_24_left_and_right_block_diagonal_below_1e14"]
            ),
            "qft_plancherel_and_nonabelian_convolution_hold": bool(
                block_report["plancherel_and_convolution_hold"]
            ),
            "qft_q8_coset_100pct_on_1d_irreps": bool(
                hsp_report["q8_concentrates_100pct_on_1d_irreps"]
            ),
            "qft_all_6_subgroups_hsp_coset_sampling_exact": bool(
                hsp_report["all_subgroups_hsp_verified"]
            ),
            "qft_circuit_operator_exact_below_1e12": direct_circuit_err < 1e-12,
            "qft_proxy_compilations_all_equivalent": all(
                bool(c.get("equivalent_up_to_global_phase", False))
                for c in compilations
                if "status" not in c
            ),
        },
    }


def build_manifest_bt2t002(
    u_generic_override: np.ndarray | None = None,
) -> Dict[str, object]:
    """Build deterministic OSIRIS-BT2T-002 Milestone 2 verification manifest."""
    sd_report = verify_semidirect_algebra()
    mul_report = verify_multiplication_10q(u_generic_override=u_generic_override)
    qft_report = verify_qft_5q()

    all_checks: Dict[str, bool] = {}
    for k, v in sd_report.items():
        if isinstance(v, bool):
            all_checks[f"semidirect_{k}"] = v
    all_checks.update(mul_report["checks"])  # type: ignore[arg-type]
    all_checks.update(qft_report["checks"])  # type: ignore[arg-type]

    return {
        "experiment_id": "OSIRIS-BT2T-002",
        "milestone": (
            "M2: exact 10-qubit reversible 2T multiplication and 5-qubit non-abelian 2T QFT (offline)"
        ),
        "execution_tier": "SIMULATED",
        "execution_class": "NO_EXECUTION (exact linear algebra and local PROXY compilation only)",
        "provider_job_id": None,
        "qpu_seconds": 0,
        "preregistration": None,
        "encoding_version": bt2t_mul.ENCODING_VERSION,
        "encoding_sha256": bt2t_mul.ENCODING_SHA256,
        "seed": SEED,
        "semidirect_algebra": sd_report,
        "part1_multiplication_10q": {
            k: v for k, v in mul_report.items() if k != "checks"
        },
        "part2_nonabelian_qft_5q": {
            k: v for k, v in qft_report.items() if k != "checks"
        },
        "checks": all_checks,
        "all_checks_passed": all(all_checks.values()),
        "governance_note": (
            "Offline exact group-primitive verification and local PROXY gate compilation only; "
            "no quantum hardware or cloud emulator was invoked (0 QPU seconds, provider_job_id=null)."
        ),
    }


def write_manifest_bt2t002(output_path: Path = DEFAULT_MANIFEST_PATH) -> Dict[str, object]:
    """Generate and write deterministic manifest_bt2t002.json."""
    manifest = build_manifest_bt2t002()
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_MANIFEST_PATH)
    args = parser.parse_args()
    manifest = write_manifest_bt2t002(args.out)
    ok = bool(manifest["all_checks_passed"])
    print(f"wrote {args.out}; all_checks_passed={ok} ({len(manifest['checks'])} checks)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
