"""OSIRIS-BT2T-002 Milestone 2 tests: exact 10Q reversible 2T multiplication & 5Q non-abelian 2T QFT."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osiris_cli import claims  # noqa: E402

PRIMITIVES_DIR = REPO_ROOT / "experiments" / "bt2t_primitives"
MANIFEST_PATH = PRIMITIVES_DIR / "manifest_bt2t002.json"


def _load_prim_module(name: str) -> Any:
    if str(PRIMITIVES_DIR) not in sys.path:
        sys.path.insert(0, str(PRIMITIVES_DIR))
    spec = importlib.util.spec_from_file_location(name, PRIMITIVES_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None, f"Cannot load {name}.py"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bt2t_mul = _load_prim_module("bt2t_mul")
bt2t_qft = _load_prim_module("bt2t_qft")
verify_bt2t002 = _load_prim_module("verify_bt2t002")


@pytest.fixture(scope="module")
def perm_1024() -> list[int]:
    return bt2t_mul.multiplication_permutation_10q()


@pytest.fixture(scope="module")
def p_matrix_1024(perm_1024: list[int]) -> np.ndarray:
    return verify_bt2t002.permutation_matrix(perm_1024)


@pytest.fixture(scope="module")
def u_generic_1024() -> np.ndarray:
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Operator

    return Operator(bt2t_mul.multiplication_circuit_generic()).data


@pytest.fixture(scope="module")
def u_structured_1024() -> np.ndarray:
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Operator

    return Operator(bt2t_mul.multiplication_circuit_structured()).data


# ---------------------------------------------------------------------------
# Part 1: Exact 10-Qubit Reversible 2T Group Multiplication
# ---------------------------------------------------------------------------


def test_frozen_bt2t_enc1_and_cayley_unused_extension(perm_1024: list[int]) -> None:
    rec = bt2t_mul.verify_frozen_encoding()
    assert rec["encoding_version"] == "bt2t-enc-1"
    assert rec["sha256"] == "3e3b02d13539daaf431af39056d031f777a785a556eb9da21ef4ea6097dfc4c5"

    tbl = bt2t_mul.mul_table_24()
    # Left translation L_x is a bijection on 0..23 for all 24 elements (576 valid pairs):
    valid_count = 0
    for x in range(24):
        row_imgs = [bt2t_mul.mu(x, y, tbl) for y in range(24)]
        assert sorted(row_imgs) == list(range(24))
        valid_count += len(row_imgs)
    assert valid_count == 576

    # Unitary identity extension on all 448 unused-codeword pairs:
    unused_count = 0
    for x in range(32):
        for y in range(32):
            if x >= 24 or y >= 24:
                unused_count += 1
                assert bt2t_mul.mu(x, y, tbl) == y
                idx = bt2t_mul.pack_xy(x, y)
                assert perm_1024[idx] == idx
    assert unused_count == 448
    assert sorted(perm_1024) == list(range(1024))


def test_semidirect_product_algebra_and_esd_5q_bijection() -> None:
    pytest.importorskip("qiskit")
    sd = verify_bt2t002.verify_semidirect_algebra()
    assert sd["encoding_sha256_verified"] is True
    assert sd["omega_order_3"] is True
    assert sd["ad_omega_matches_quaternion_conjugation"] is True
    assert sd["q8_3bit_cocycle_exact_all_64"] is True
    assert sd["semidirect_right_and_left_bijections_24"] is True
    assert sd["semidirect_product_law_all_576_pairs"] is True
    assert sd["esd_5q_classical_and_unitary_exact"] is True
    assert sd["esd_5q_operator_max_abs_error"] < 1e-12


def test_classical_gate_lists_map_all_1024_basis_states(perm_1024: list[int]) -> None:
    gates_gen = bt2t_mul.multiplication_gate_list_generic()
    gates_str = bt2t_mul.multiplication_gate_list_structured()

    assert [bt2t_mul.apply_gate_list_classically(gates_gen, s) for s in range(1024)] == perm_1024
    assert [bt2t_mul.apply_gate_list_classically(gates_str, s) for s in range(1024)] == perm_1024

    allowed_kinds = {"x", "cx", "ccx", "mcx", "cswap"}
    assert {k for k, _, _ in gates_gen}.issubset(allowed_kinds)
    assert {k for k, _, _ in gates_str}.issubset(allowed_kinds)


def test_exact_1024x1024_operator_unitaries_including_phases(
    p_matrix_1024: np.ndarray,
    u_generic_1024: np.ndarray,
    u_structured_1024: np.ndarray,
) -> None:
    qc_gen = bt2t_mul.multiplication_circuit_generic()
    qc_str = bt2t_mul.multiplication_circuit_structured()
    assert qc_gen.num_qubits == 10 and qc_gen.num_ancillas == 0
    assert qc_str.num_qubits == 10 and qc_str.num_ancillas == 0

    err_gen = float(np.max(np.abs(u_generic_1024 - p_matrix_1024)))
    err_str = float(np.max(np.abs(u_structured_1024 - p_matrix_1024)))
    assert err_gen < 1e-10, f"Generic 1024x1024 matrix error {err_gen} >= 1e-10"
    assert err_str < 1e-10, f"Structured 1024x1024 matrix error {err_str} >= 1e-10"


def test_random_10q_complex_superpositions_with_relative_phases(
    perm_1024: list[int],
    u_generic_1024: np.ndarray,
) -> None:
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector

    qc_str = bt2t_mul.multiplication_circuit_structured()
    rng = np.random.default_rng(20261124)
    valid_indices = [bt2t_mul.pack_xy(x, y) for x in range(24) for y in range(24)]
    all_indices = list(range(1024))

    for support in (valid_indices, all_indices):
        for _ in range(16):
            amp = np.zeros(1024, dtype=complex)
            amp[support] = rng.normal(size=len(support)) + 1j * rng.normal(size=len(support))
            amp /= np.linalg.norm(amp)

            expected = np.zeros(1024, dtype=complex)
            for s_idx in range(1024):
                expected[perm_1024[s_idx]] = amp[s_idx]

            out_str = Statevector(amp).evolve(qc_str).data
            out_gen = u_generic_1024 @ amp
            assert float(np.max(np.abs(out_str - expected))) < 1e-10
            assert float(np.max(np.abs(out_gen - expected))) < 1e-10


def test_positive_phase_error_injection_control_caught(
    perm_1024: list[int],
    p_matrix_1024: np.ndarray,
) -> None:
    """Injecting CZ(0, 5) leaves all 1024 computational-basis outputs unchanged but flips relative
    phases by -1 on states with x0=1 and y0=1, which MUST fail both Operator and superposition checks.
    """
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Operator, Statevector

    qc_bad = bt2t_mul.multiplication_circuit_structured()
    qc_bad.cz(0, 5)
    u_bad = Operator(qc_bad).data

    # Computational basis support test still sees exact permutation on all 1024 states:
    assert [int(np.argmax(np.abs(u_bad[:, s]))) for s in range(1024)] == perm_1024
    # Phase-aware 1024x1024 Operator comparison catches the -1 phase error (diff == 2.0):
    assert float(np.max(np.abs(u_bad - p_matrix_1024))) > 1.0

    # Superposition check also catches the relative phase flip:
    psi = np.zeros(1024, dtype=complex)
    psi[bt2t_mul.pack_xy(0, 0)] = 1.0 / np.sqrt(2.0)
    psi[bt2t_mul.pack_xy(1, 1)] = 1.0 / np.sqrt(2.0)  # x0=1, y0=1 (mu(1,1) = 0 -> y0_out=0; wait: cz is at end!)
    # Note: for x=1 (-1), y=3 (-i), mu(1, 3) = 2 (+i, y0_out=0); for x=1, y=2 (+i), mu(1, 2) = 3 (-i, y0_out=1)!
    psi = np.zeros(1024, dtype=complex)
    psi[bt2t_mul.pack_xy(0, 0)] = 1.0 / np.sqrt(2.0)
    psi[bt2t_mul.pack_xy(1, 2)] = 1.0 / np.sqrt(2.0)
    evolved_bad = Statevector(psi).evolve(qc_bad).data
    expected = p_matrix_1024 @ psi
    assert float(np.max(np.abs(evolved_bad - expected))) > 1.0


def test_structured_vs_generic_native_cz_compression(p_matrix_1024: np.ndarray) -> None:
    pytest.importorskip("qiskit")
    from qiskit import transpile

    qc_gen = bt2t_mul.multiplication_circuit_generic()
    qc_str = bt2t_mul.multiplication_circuit_structured()

    tr_gen = transpile(
        qc_gen,
        basis_gates=["rz", "sx", "x", "cz"],
        optimization_level=3,
        seed_transpiler=verify_bt2t002.SEED,
    )
    tr_str = transpile(
        qc_str,
        basis_gates=["rz", "sx", "x", "cz"],
        optimization_level=3,
        seed_transpiler=verify_bt2t002.SEED,
    )

    cz_gen = int(tr_gen.count_ops().get("cz", 0))
    cz_str = int(tr_str.count_ops().get("cz", 0))
    assert cz_str < 2000
    assert cz_gen > 25000
    assert cz_gen / float(cz_str) > 18.0

    u_tr_str = verify_bt2t002._unitary_10q_transpiled(tr_str)
    assert verify_bt2t002._phase_aligned_max_abs_error(p_matrix_1024, u_tr_str) < 1e-9


# ---------------------------------------------------------------------------
# Part 2: Exact Non-Abelian Quantum Fourier Transform over 2T
# ---------------------------------------------------------------------------


def test_seven_irreps_unitarity_and_homomorphism_all_576_pairs() -> None:
    rep = bt2t_qft.verify_irreps_and_schur_orthogonality()
    assert rep["sum_d_alpha_squared"] == 24
    assert rep["sum_d_alpha_squared_equals_24"] is True
    assert rep["all_irreps_unitary"] is True
    assert rep["all_irreps_homomorphisms_all_576_pairs"] is True
    assert rep["rho3_so3_real_orthogonal"] is True


def test_character_table_orthogonality_and_distinct_traces() -> None:
    chars = bt2t_qft.character_table_2t()
    assert chars.shape == (7, 24)
    gram = (chars @ chars.conj().T) / 24.0
    assert float(np.linalg.norm(gram - np.eye(7), ord=np.inf)) < 1e-14

    # Dimensions at identity element g=0 equal (1, 1, 1, 2, 2, 2, 3):
    dims_at_e = [int(np.rint(np.real(chars[i, 0]))) for i in range(7)]
    assert dims_at_e == [1, 1, 1, 2, 2, 2, 3]

    # Spinor irreps rho_{2,0}, rho_{2,1}, rho_{2,2} have distinct traces on Hurwitz cosets:
    assert abs(chars[3, 23] - chars[4, 23]) > 1.0
    assert abs(chars[4, 23] - chars[5, 23]) > 1.0
    assert abs(chars[3, 23] - chars[5, 23]) > 1.0


def test_peter_weyl_32x32_exact_unitarity_and_unused_identity() -> None:
    f24 = bt2t_qft.peter_weyl_matrix_24()
    f32 = bt2t_qft.bt2t_qft_matrix_32()
    assert f24.shape == (24, 24)
    assert f32.shape == (32, 32)
    assert float(np.linalg.norm(f24 @ f24.conj().T - np.eye(24), ord=np.inf)) < 1e-14
    assert float(np.linalg.norm(f32 @ f32.conj().T - np.eye(32), ord=np.inf)) < 1e-14
    assert np.array_equal(f32[24:, 24:], np.eye(8, dtype=complex))
    assert np.count_nonzero(f32[:24, 24:]) == 0
    assert np.count_nonzero(f32[24:, :24]) == 0


def test_simultaneous_block_diagonalization_all_24_translations_and_convolution() -> None:
    blk = bt2t_qft.verify_simultaneous_block_diagonalization()
    assert blk["irrep_unique_dims"] == [1, 1, 1, 2, 2, 2, 3]
    assert blk["block_sizes_contiguous"] == [1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3]
    assert blk["left_regular_max_off_block_abs"] < 1e-14
    assert blk["left_regular_max_on_block_error"] < 1e-14
    assert blk["right_regular_max_off_irrep_abs"] < 1e-14
    assert blk["right_regular_max_on_block_error"] < 1e-14
    assert blk["all_24_left_and_right_block_diagonal_below_1e14"] is True
    assert blk["plancherel_and_convolution_hold"] is True


def test_nonabelian_subgroup_coset_hsp_fourier_concentration() -> None:
    hsp = bt2t_qft.verify_subgroup_coset_sampling()
    assert hsp["all_subgroups_hsp_verified"] is True
    assert hsp["q8_concentrates_100pct_on_1d_irreps"] is True
    assert hsp["q8_1d_total_weight"] == pytest.approx(1.0, abs=1e-14)
    assert hsp["q8_2d_3d_leakage"] < 1e-28

    sgs = hsp["subgroups"]
    assert set(sgs.keys()) == {
        "trivial_1",
        "Z2_center",
        "Z4_i",
        "Z3_omega",
        "Q8_normal",
        "full_2T",
    }
    # Spot-check Z_3 subgroup (non-normal, order 3): kills chi_1, chi_2, and rho_{2,0}!
    z3_w = sgs["Z3_omega"]["measured_irrep_weights_coset_eH"]
    assert z3_w["chi_0"] == pytest.approx(0.125, abs=1e-12)
    assert z3_w["chi_1"] == pytest.approx(0.0, abs=1e-12)
    assert z3_w["chi_2"] == pytest.approx(0.0, abs=1e-12)
    assert z3_w["rho_2_0"] == pytest.approx(0.0, abs=1e-12)
    assert z3_w["rho_2_1"] == pytest.approx(0.25, abs=1e-12)
    assert z3_w["rho_2_2"] == pytest.approx(0.25, abs=1e-12)
    assert z3_w["rho_3"] == pytest.approx(0.375, abs=1e-12)


def test_bt2t_qft_5q_circuit_and_heavy_hex_5q_compilation_equivalence() -> None:
    pytest.importorskip("qiskit")
    from qiskit import transpile
    from qiskit.quantum_info import Operator
    from qiskit.transpiler import CouplingMap

    f32 = bt2t_qft.bt2t_qft_matrix_32()
    qc = bt2t_qft.bt2t_qft_circuit()
    op_ref = Operator(qc)
    assert float(np.max(np.abs(op_ref.data - f32))) < 1e-12

    for cmap in (None, CouplingMap(verify_bt2t002.HEAVY_HEX_5Q)):
        tr = transpile(
            qc,
            basis_gates=["rz", "sx", "x", "cz"],
            coupling_map=cmap,
            optimization_level=3,
            seed_transpiler=verify_bt2t002.SEED,
        )
        op_tr = Operator.from_circuit(tr)
        assert op_tr.equiv(op_ref, atol=1e-9)
        assert verify_bt2t002._phase_aligned_max_abs_error(f32, op_tr.data) < 1e-9


# ---------------------------------------------------------------------------
# Part 3: Deterministic Manifest, Provenance & SYS-024 Cleanliness
# ---------------------------------------------------------------------------


def test_manifest_bt2t002_deterministic_schema_and_reproducibility(
    u_generic_1024: np.ndarray,
) -> None:
    assert MANIFEST_PATH.is_file(), f"Missing manifest: {MANIFEST_PATH}"
    saved_text = MANIFEST_PATH.read_text(encoding="utf-8")
    saved = json.loads(saved_text)

    assert saved["experiment_id"] == "OSIRIS-BT2T-002"
    assert saved["execution_tier"] == "SIMULATED"
    assert saved["provider_job_id"] is None
    assert saved["qpu_seconds"] == 0
    assert saved["encoding_version"] == "bt2t-enc-1"
    assert (
        saved["encoding_sha256"]
        == "3e3b02d13539daaf431af39056d031f777a785a556eb9da21ef4ea6097dfc4c5"
    )
    assert saved["all_checks_passed"] is True
    assert all(saved["checks"].values())

    for comp in (
        saved["part1_multiplication_10q"]["proxy_compilations"]
        + saved["part2_nonabelian_qft_5q"]["proxy_compilations"]
    ):
        assert comp["execution_tier"] == "PROXY"
        assert comp["provider_job_id"] is None

    if importlib.util.find_spec("qiskit") is not None and importlib.util.find_spec("pytket") is not None:
        regenerated = verify_bt2t002.build_manifest_bt2t002(u_generic_override=u_generic_1024)
        regen_text = json.dumps(regenerated, indent=2, sort_keys=True) + "\n"
        assert regen_text == saved_text


def test_sys024_claims_firewall_cleanliness() -> None:
    assert MANIFEST_PATH.is_file()
    manifest_text = MANIFEST_PATH.read_text(encoding="utf-8")
    assert claims.flagged_in_reply(manifest_text) == []

    banned_tokens = ("theta_lock", "51.843", "tau_phase", "2.176435e-08", "phi_threshold")
    for path in sorted(PRIMITIVES_DIR.glob("*")):
        if path.is_file() and path.suffix in {".py", ".json", ".md"}:
            txt = path.read_text(encoding="utf-8")
            assert claims.flagged_in_reply(txt) == [], f"SYS-024 flagged claim in {path.name}"
            lower = txt.lower()
            for tok in banned_tokens:
                assert tok not in lower, f"Quarantined token {tok!r} found in {path.name}"
