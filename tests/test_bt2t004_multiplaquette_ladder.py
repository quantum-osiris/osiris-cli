"""OSIRIS-BT2T-004 Milestone 4 tests: 10-qubit coupled 2-plaquette 2T Kogut-Susskind ladder."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osiris_cli import claim_provenance, claims, plan_gates  # noqa: E402

LADDER_DIR = REPO_ROOT / "experiments" / "bt2t_multiplaquette"
MANIFEST_PATH = LADDER_DIR / "manifest_bt2t004.json"


def _load_ladder_module(name: str) -> Any:
    if str(LADDER_DIR) not in sys.path:
        sys.path.insert(0, str(LADDER_DIR))
    spec = importlib.util.spec_from_file_location(name, LADDER_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None, f"Cannot load {name}.py"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ladder = _load_ladder_module("bt2t_two_plaquette_ladder")
verify_bt2t004 = _load_ladder_module("verify_bt2t004")


@pytest.fixture(scope="module")
def reduction_report() -> Dict[str, Any]:
    return ladder.verify_two_plaquette_gauge_reduction()


@pytest.fixture(scope="module")
def hamiltonian_report() -> Dict[str, Any]:
    return ladder.verify_coupled_hamiltonian_and_shared_link(g_ym_sq=1.0, dt=0.2)


@pytest.fixture(scope="module")
def confinement_report() -> Dict[str, Any]:
    return ladder.benchmark_wilson_loop_confinement_crossover()


@pytest.fixture(scope="module")
def trotter_report() -> Dict[str, Any]:
    return ladder.benchmark_coupled_trotter_scaling(t_final=0.5)


@pytest.fixture(scope="module")
def live_manifest() -> Dict[str, Any]:
    return verify_bt2t004.build_manifest_bt2t004()


# ---------------------------------------------------------------------------
# Part 1: Gauge-Fixed 10Q vs 35Q Reduction & 76-D Physical Orbit Subspace
# ---------------------------------------------------------------------------


def test_frozen_encoding_and_35q_to_10q_hilbert_space_reduction(
    reduction_report: Dict[str, Any],
) -> None:
    rec = ladder.verify_frozen_encoding()
    assert rec["encoding_version"] == "bt2t-enc-1"
    assert rec["sha256"] == "3e3b02d13539daaf431af39056d031f777a785a556eb9da21ef4ea6097dfc4c5"

    assert reduction_report["num_vertices"] == 6
    assert reduction_report["num_directed_links_unfixed"] == 7
    assert reduction_report["num_spanning_tree_fixed_links"] == 5
    assert reduction_report["unfixed_group_configurations"] == 24**7 == 4586471424
    assert reduction_report["gauge_fixed_active_configurations"] == 24**2 == 576
    assert reduction_report["hilbert_space_config_reduction_factor"] == 24**5 == 7962624
    assert reduction_report["unfixed_qubits"] == 35
    assert reduction_report["gauge_fixed_qubits"] == 10
    assert reduction_report["qubit_reduction_factor"] == 3.5
    assert reduction_report["padded_10q_hilbert_dim"] == 1024
    assert reduction_report["active_2t_x_2t_subspace_dim"] == 576
    assert reduction_report["decoupled_padded_subspace_dim"] == 448


def test_burnside_character_and_orbit_enumeration_prove_76d_physical_subspace(
    reduction_report: Dict[str, Any],
) -> None:
    assert reduction_report["conjugacy_class_sizes"] == [1, 1, 6, 4, 4, 4, 4]
    assert reduction_report["class_centralizer_sizes"] == [24, 24, 4, 6, 6, 6, 6]
    assert reduction_report["burnside_sum_centralizer_squared"] == 1824
    assert reduction_report["burnside_physical_subspace_dim"] == 76
    assert reduction_report["class_centralizer_sum_dim"] == 76
    assert reduction_report["clebsch_gordan_endomorphism_dim"] == 76
    assert reduction_report["trivial_irrep_pair_count"] == 7
    assert reduction_report["explicit_conjugation_orbit_count"] == 76
    assert all(reduction_report["checks"].values())


def test_physical_76d_orbit_basis_orthonormality_and_conjugation_invariance(
    reduction_report: Dict[str, Any],
) -> None:
    assert reduction_report["physical_basis_76_orthonormality_max_error"] < 1e-14
    assert reduction_report["physical_basis_76_global_conjugation_max_error"] < 1e-14

    phi_76 = ladder.physical_orbit_basis_576()
    assert phi_76.shape == (76, 576)
    for h in (1, 2, 4, 8, 16, 23):
        adh_2p = ladder.global_adjoint_matrix_576(h)
        assert float(np.linalg.norm(phi_76 @ adh_2p.T - phi_76, ord=2)) < 1e-14


# ---------------------------------------------------------------------------
# Part 2: Coupled 2-Plaquette Hamiltonian & Shared-Link Electric Operator
# ---------------------------------------------------------------------------


def test_shared_link_electric_operator_matches_direct_cayley_and_is_hermitian_psd(
    hamiltonian_report: Dict[str, Any],
) -> None:
    assert hamiltonian_report["shared_link_mul_vs_direct_cayley_2norm_error"] < 1e-14
    assert hamiltonian_report["shared_link_vs_rectangular_isospectral_max_diff"] < 1e-13
    assert hamiltonian_report["hamiltonian_hermiticity_2norm_error"] < 1e-14
    assert hamiltonian_report["shared_electric_min_eigenvalue"] >= -1e-14
    assert hamiltonian_report["shared_electric_max_eigenvalue"] == pytest.approx(2.25, abs=1e-12)
    assert hamiltonian_report["total_electric_min_eigenvalue"] >= -1e-14


def test_mutual_commutativity_of_outer_and_shared_electric_operators_on_1024(
    hamiltonian_report: Dict[str, Any],
) -> None:
    assert hamiltonian_report["max_electric_mutual_commutator_2norm"] < 1e-14
    assert hamiltonian_report["ub_10q_vs_expm_2norm_error"] < 1e-14
    assert hamiltonian_report["ue_10q_factorized_vs_expm_2norm_error"] < 1e-14


def test_576_active_subspace_zero_leakage_and_global_gauge_conjugation_invariance(
    hamiltonian_report: Dict[str, Any],
) -> None:
    assert hamiltonian_report["active_576_to_padded_448_te_shared_leakage_norm"] == 0.0
    assert hamiltonian_report["active_576_to_padded_448_hamiltonian_leakage_norm"] == 0.0
    assert hamiltonian_report["active_576_to_padded_448_strang_leakage_norm"] < 1e-15
    assert hamiltonian_report["max_commutator_te_shared_global_adjoint_2norm"] < 1e-14
    assert hamiltonian_report["max_commutator_he_global_adjoint_2norm"] < 1e-14
    assert hamiltonian_report["max_commutator_hb_global_adjoint_2norm"] < 1e-14
    assert hamiltonian_report["max_commutator_h2p_global_adjoint_2norm"] < 1e-14
    assert all(hamiltonian_report["checks"].values())


def test_unfixed_3link_13824_ladder_intertwines_with_10q_gauge_fixed_hamiltonian(
    hamiltonian_report: Dict[str, Any],
) -> None:
    assert hamiltonian_report["unfixed_3link_13824_laplacian_intertwining_l2_error"] < 1e-13


# ---------------------------------------------------------------------------
# Part 3: Wilson-Loop Area-Law Confinement Crossover & String Tension
# ---------------------------------------------------------------------------


def test_ground_state_lies_in_76d_physical_subspace_with_plaquette_exchange_symmetry(
    confinement_report: Dict[str, Any],
) -> None:
    assert confinement_report["max_576_vs_76_ground_energy_abs_diff"] < 1e-13
    assert confinement_report["max_ground_state_outside_76d_phys_l2"] < 1e-14
    assert confinement_report["max_plaquette_exchange_asymmetry"] < 1e-14
    assert confinement_report["max_1024_padded_448_leakage"] == 0.0


def test_wilson_loop_area_law_crossover_connected_correlator_and_string_tension(
    confinement_report: Dict[str, Any],
) -> None:
    assert all(confinement_report["checks"].values())
    rows = confinement_report["sweep"]
    assert len(rows) == 7
    assert [r["g_ym_sq"] for r in rows] == [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

    for i, row in enumerate(rows):
        w1 = row["wilson_1x1"]
        w12 = row["wilson_1x2_rectangular"]
        c12 = row["connected_correlator_c12"]
        sigma_eff = row["effective_string_tension_sigma_eff"]
        assert 0.0 < w12 < w1 < 1.0
        assert c12 == pytest.approx(w12 - w1 * w1, abs=1e-15)
        assert c12 > 0.0
        assert sigma_eff == pytest.approx(-np.log(w12 / w1), abs=1e-14)
        if i > 0:
            assert w1 < rows[i - 1]["wilson_1x1"]
            assert w12 < rows[i - 1]["wilson_1x2_rectangular"]
            assert sigma_eff > rows[i - 1]["effective_string_tension_sigma_eff"]


# ---------------------------------------------------------------------------
# Part 4: Coupled Strang Trotterization O(dt^2) Scaling & Real-Time Dynamics
# ---------------------------------------------------------------------------


def test_coupled_lie_and_strang_trotter_empirical_convergence_slopes(
    trotter_report: Dict[str, Any],
) -> None:
    assert all(trotter_report["checks"].values())
    assert trotter_report["step_counts"] == [4, 8, 16, 32, 64, 128]
    for g_sq in ladder.TROTTER_COUPLING_REGIMES:
        entry = trotter_report["by_coupling"][f"g_ym_sq_{g_sq:.1f}"]
        assert 0.99 <= entry["lie_loglog_slope"] <= 1.01
        assert 1.99 <= entry["strang_loglog_slope"] <= 2.01
    assert trotter_report["max_strang_error_n64"] < 1e-4
    assert trotter_report["max_strang_error_n128"] < 1e-5
    assert trotter_report["realtime_10q_padded_448_leakage"] < 1e-28
    assert trotter_report["realtime_10q_global_gauss_conjugation_drift"] < 1e-13


# ---------------------------------------------------------------------------
# Part 5: Guarded Qiskit & pytket 10Q/35Q Compilations & Gate Compression
# ---------------------------------------------------------------------------


def test_qiskit_10q_and_35q_compilations_and_gate_compression_ratios(
    live_manifest: Dict[str, Any],
) -> None:
    pytest.importorskip("qiskit")
    comp = live_manifest["part5_native_10q_and_35q_circuit_compilations"]
    assert comp["status"] == "OK"
    assert comp["qiskit_available"] is True

    errs = comp["direct_circuit_errors"]
    assert errs["ub_10q_2norm_error"] < 1e-14
    assert errs["ue_structured_10q_2norm_error"] < 1e-14
    assert errs["strang_structured_10q_2norm_error"] < 1e-14

    summary = comp["gate_count_summary"]
    assert summary["cz_ub_structured_10q_all_to_all"] == 32
    assert summary["cz_ub_structured_10q_heavy_hex_2xT5_patches"] == 38
    assert (
        summary["cz_strang_structured_10q_all_to_all"]
        < summary["cz_strang_dense5q_blocks_10q_all_to_all"]
    )
    assert (
        summary["cz_strang_structured_10q_heavy_hex_10Q"]
        < summary["cz_strang_dense5q_blocks_10q_heavy_hex_10Q"]
    )
    assert summary["cz_generic_dense_1024x1024_qsd_exact_all_to_all"] == 479063
    assert summary["cz_compression_generic_dense_10q_vs_structured_10q_all_to_all"] > 100.0
    assert summary["cz_compression_unfixed_35q_vs_10q_all_to_all"] > 10.0
    assert summary["cz_compression_unfixed_35q_vs_10q_heavy_hex"] > 10.0

    for entry in comp["proxy_compilations_10q"]:
        assert entry["execution_tier"] == "PROXY"
        assert entry["provider_job_id"] is None
        assert entry["equivalent_up_to_global_phase"] is True


def test_pytket_10q_zzphase_compilation(
    live_manifest: Dict[str, Any],
) -> None:
    pytest.importorskip("qiskit")
    pytest.importorskip("pytket")
    comp = live_manifest["part5_native_10q_and_35q_circuit_compilations"]
    assert comp["pytket_available"] is True

    summary = comp["gate_count_summary"]
    assert summary["zzphase_ub_structured_10q"] == 32
    assert summary["zzphase_strang_structured_10q"] < summary["zzphase_strang_dense5q_blocks_10q"]

    for entry in comp["proxy_compilations_pytket"]:
        assert entry["execution_tier"] == "PROXY"
        assert entry["provider_job_id"] is None


# ---------------------------------------------------------------------------
# Part 6: Manifest Determinism, Offline No-SDK Fallback & SYS-024 Cleanliness
# ---------------------------------------------------------------------------


def test_manifest_bt2t004_determinism_and_rfc8785_jcs(
    live_manifest: Dict[str, Any],
) -> None:
    assert MANIFEST_PATH.is_file(), f"Missing manifest: {MANIFEST_PATH}"
    saved_text = MANIFEST_PATH.read_text(encoding="utf-8")
    saved = json.loads(saved_text)

    assert saved["experiment_id"] == "OSIRIS-BT2T-004"
    assert saved["execution_tier"] == "SIMULATED"
    assert saved["evidence_class"] == "SIMULATED"
    assert saved["provider_job_id"] is None
    assert saved["qpu_seconds"] == 0
    assert saved["encoding_version"] == "bt2t-enc-1"
    assert (
        saved["encoding_sha256"]
        == "3e3b02d13539daaf431af39056d031f777a785a556eb9da21ef4ea6097dfc4c5"
    )
    assert saved["seed"] == 20261124
    assert saved["all_passed"] is True
    assert saved["all_checks_passed"] is True
    assert len(saved["checks"]) >= 28
    assert all(saved["checks"].values())

    body = {k: v for k, v in saved.items() if k != "rfc8785_canonical_sha256"}
    pure_bytes = verify_bt2t004.pure_python_jcs_bytes(body)
    assert hashlib.sha256(pure_bytes).hexdigest() == saved["rfc8785_canonical_sha256"]

    if importlib.util.find_spec("rfc8785") is not None:
        import rfc8785

        assert bytes(rfc8785.dumps(body)) == pure_bytes

    if (
        importlib.util.find_spec("qiskit") is not None
        and importlib.util.find_spec("pytket") is not None
    ):
        regen_text = json.dumps(live_manifest, indent=2, sort_keys=True) + "\n"
        assert regen_text == saved_text


def test_offline_pure_numpy_execution_without_qiskit_pytket_or_rfc8785() -> None:
    """Simulate a pure-NumPy Python 3.9 environment with qiskit, pytket, and rfc8785 absent."""
    blocked_script = (
        "import importlib.abc, json, sys\n"
        "from pathlib import Path\n"
        "BLOCKED = {'qiskit', 'qiskit_aer', 'qiskit_ibm_runtime', 'pytket', 'rfc8785'}\n"
        "import importlib.util as _iu\n"
        "_orig_find_spec = _iu.find_spec\n"
        "def _guarded_find_spec(name, package=None):\n"
        "    if name.split('.')[0] in BLOCKED:\n"
        "        return None\n"
        "    return _orig_find_spec(name, package)\n"
        "_iu.find_spec = _guarded_find_spec\n"
        "class _RaiseBlocker(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path, target=None):\n"
        "        if fullname.split('.')[0] in BLOCKED:\n"
        "            raise ModuleNotFoundError(f'Blocked {fullname}')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _RaiseBlocker())\n"
        f"sys.path.insert(0, {str(LADDER_DIR)!r})\n"
        "import verify_bt2t004\n"
        "m = verify_bt2t004.build_manifest_bt2t004()\n"
        "assert m['all_passed'] is True\n"
        "assert m['all_checks_passed'] is True\n"
        "assert m['part5_native_10q_and_35q_circuit_compilations']['status'] == 'SKIPPED_NO_QISKIT'\n"
        "assert len(m['checks']) >= 28\n"
        f"saved = json.loads(Path({str(MANIFEST_PATH)!r}).read_text(encoding='utf-8'))\n"
        "body = {k: v for k, v in saved.items() if k != 'rfc8785_canonical_sha256'}\n"
        "assert verify_bt2t004.jcs_sha256(body) == saved['rfc8785_canonical_sha256']\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", blocked_script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Offline no-SDK check failed:\n{proc.stderr}\n{proc.stdout}"


def test_sys024_claims_firewall_and_secret_cleanliness() -> None:
    quarantined_verdicts = {
        "REFUTED",
        "ARTIFACT",
        "NOT_MEASURED",
        "RULED_OUT",
        "OVERCLAIM",
        "NO_EVIDENCE",
    }
    for path in sorted(LADDER_DIR.glob("*")):
        if path.is_file() and path.suffix in {".py", ".json", ".md"}:
            txt = path.read_text(encoding="utf-8")
            assert claims.flagged_in_reply(txt) == [], f"SYS-024 flagged in {path.name}"
            bad = [c for c in claims.match(txt) if c.verdict in quarantined_verdicts]
            assert not bad, f"Quarantined claim in {path.name}: {[(c.id, c.verdict) for c in bad]}"
            for p_id, p_re in plan_gates._QUARANTINED_PLAN_PATTERNS:
                assert not p_re.search(txt), f"Banned pattern {p_id} in {path.name}"
            for q_id, q_re in claim_provenance._QUARANTINED_CONSTANTS:
                assert not q_re.search(txt), f"Quarantined constant {q_id} in {path.name}"
            assert "crn:v1:bluemix:" not in txt, f"CRN leaked in {path.name}"
            assert "ZENODO_TOKEN=" not in txt, f"Token leaked in {path.name}"
