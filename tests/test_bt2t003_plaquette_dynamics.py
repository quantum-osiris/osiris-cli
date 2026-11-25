"""OSIRIS-BT2T-003 Milestone 3 tests: Kogut-Susskind 2T non-abelian plaquette Trotter dynamics."""

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

PLAQUETTE_DIR = REPO_ROOT / "experiments" / "bt2t_plaquette"
MANIFEST_PATH = PLAQUETTE_DIR / "manifest_bt2t003.json"


def _load_plaq_module(name: str) -> Any:
    if str(PLAQUETTE_DIR) not in sys.path:
        sys.path.insert(0, str(PLAQUETTE_DIR))
    spec = importlib.util.spec_from_file_location(name, PLAQUETTE_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None, f"Cannot load {name}.py"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dyn = _load_plaq_module("bt2t_plaquette_dynamics")
verify_bt2t003 = _load_plaq_module("verify_bt2t003")


@pytest.fixture(scope="module")
def gauss_report() -> Dict[str, Any]:
    return dyn.verify_gauss_law_bi_invariance(g_ym_sq=1.0, dt=0.2)


@pytest.fixture(scope="module")
def multilink_report() -> Dict[str, Any]:
    return dyn.verify_multilink_gauge_isomorphism(dt=0.15, g_ym_sq=1.0, n_steps=5, seed=20261124)


@pytest.fixture(scope="module")
def regime_report() -> Dict[str, Any]:
    return dyn.simulate_plaquette_regimes(t_final=1.0, n_strang_steps=64)


@pytest.fixture(scope="module")
def scaling_report() -> Dict[str, Any]:
    return dyn.benchmark_trotter_error_scaling(t_final=1.0)


@pytest.fixture(scope="module")
def live_manifest() -> Dict[str, Any]:
    return verify_bt2t003.build_manifest_bt2t003()


# ---------------------------------------------------------------------------
# Part 1: Pure-NumPy Group, Gauss Law, Peter-Weyl & Hamiltonian Checks
# ---------------------------------------------------------------------------


def test_frozen_encoding_conjugacy_classes_and_cayley_neighbors(
    gauss_report: Dict[str, Any],
) -> None:
    rec = dyn.verify_frozen_encoding()
    assert rec["encoding_version"] == "bt2t-enc-1"
    assert rec["sha256"] == "3e3b02d13539daaf431af39056d031f777a785a556eb9da21ef4ea6097dfc4c5"

    classes = dyn.conjugacy_classes_2t()
    assert len(classes) == 7
    assert [len(c) for c in classes] == [1, 1, 6, 4, 4, 4, 4]
    assert sorted(x for c in classes for x in c) == list(range(24))

    assert gauss_report["fundamental_character_distinct_values"] == [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert gauss_report["s8_symmetric_and_normal"] is True
    assert gauss_report["s8_generated_order"] == 24
    assert gauss_report["s8_generates_2t_diameter"] == 3


def test_gauss_law_bi_invariance_across_all_group_elements(
    gauss_report: Dict[str, Any],
) -> None:
    assert gauss_report["left_right_laplacian_2norm_diff"] < 1e-14
    assert gauss_report["max_commutator_delta_left_translation_2norm"] < 1e-14
    assert gauss_report["max_commutator_delta_right_translation_2norm"] < 1e-14
    assert gauss_report["max_commutator_hb_gauss_adjoint_2norm"] < 1e-14
    assert gauss_report["max_commutator_he_gauss_adjoint_2norm"] < 1e-14
    assert gauss_report["max_commutator_hks_gauss_adjoint_2norm"] < 1e-14
    assert all(gauss_report["checks"].values())

    # Check Gauss's law [H_KS(g^2), Ad_h] = 0 across all three benchmark couplings:
    for g_sq in dyn.COUPLING_REGIMES:
        h_ks = dyn.kogut_susskind_hamiltonian_32(g_sq)
        for h in range(24):
            adh = dyn.adjoint_matrix_32(h)
            assert float(np.linalg.norm(h_ks @ adh - adh @ h_ks, ord=2)) < 1e-14


def test_peter_weyl_qft_diagonalizes_electric_hamiltonian_and_casimir_spectrum(
    gauss_report: Dict[str, Any],
) -> None:
    assert gauss_report["peter_weyl_casimir_diagonalization_2norm_error"] < 1e-14
    assert gauss_report["peter_weyl_off_diagonal_max_abs"] < 1e-14
    assert gauss_report["ub_matrix_vs_expm_2norm_error"] < 1e-14
    assert gauss_report["ue_pw_qft_vs_expm_2norm_error"] < 1e-14

    expected_c2 = {
        "chi_0": 0.0,
        "chi_1": 2.25,
        "chi_2": 2.25,
        "rho_2_0": 0.75,
        "rho_2_1": 1.875,
        "rho_2_2": 1.875,
        "rho_3": 1.5,
    }
    assert dyn.CASIMIR_EIGENVALUES == expected_c2


def test_unused_8_subspace_exact_zero_generator_and_zero_leakage(
    gauss_report: Dict[str, Any],
    regime_report: Dict[str, Any],
) -> None:
    assert gauss_report["unused_8_hamiltonian_coupling_norm"] == 0.0
    assert gauss_report["unused_8_propagator_identity_error"] < 1e-14
    assert regime_report["max_unused_8_leakage"] < 1e-28
    assert regime_report["unused_8_stationary_error"] < 1e-14

    s2 = dyn.strang_step_matrix_32(0.25, 1.0)
    assert float(np.linalg.norm(s2[24:, 24:] - np.eye(8), ord=2)) < 1e-14
    assert float(np.linalg.norm(s2[:24, 24:], ord=2)) == 0.0
    assert float(np.linalg.norm(s2[24:, :24], ord=2)) == 0.0


# ---------------------------------------------------------------------------
# Part 2: Multi-Link Holonomy Accumulation & Gauge-Invariant Isomorphism
# ---------------------------------------------------------------------------


def test_multilink_holonomy_gate_lists_and_4link_vs_5q_isomorphism(
    multilink_report: Dict[str, Any],
) -> None:
    assert multilink_report["holonomy_2link_15q_triples_checked"] == 13824
    assert multilink_report["holonomy_4link_25q_configs_checked"] == 331776
    assert multilink_report["physical_gauss_invariant_subspace_dim"] == 7
    assert multilink_report["qubit_reduction_unfixed_20q_to_5q"] == 4.0
    assert multilink_report["qubit_reduction_ancilla_25q_to_5q"] == 5.0
    assert multilink_report["max_4vertex_gauss_law_l2_error_all_24_h"] < 1e-13
    assert multilink_report["max_link_laplacian_intertwining_l2_error"] < 1e-13
    assert multilink_report["two_link_vs_5q_gauge_fixed_l2_error"] <= 1e-13
    assert multilink_report["four_link_vs_5q_basis_max_l2_error"] <= 1e-13
    assert multilink_report["four_link_vs_5q_random_superposition_max_l2_error"] <= 1e-13
    assert all(multilink_report["checks"].values())


# ---------------------------------------------------------------------------
# Part 3: Real-Time Regimes & Trotter-Suzuki Convergence Slopes
# ---------------------------------------------------------------------------


def test_plaquette_regime_observables_energy_conservation_and_sector_symmetry(
    regime_report: Dict[str, Any],
) -> None:
    assert all(regime_report["checks"].values())
    regimes = regime_report["regimes"]
    for g_sq in dyn.COUPLING_REGIMES:
        key = f"g_ym_sq_{g_sq:.1f}"
        vac_exact = regimes[key]["electric_vacuum_chi0"]["exact"]
        # Starting from |chi_0>, <H_KS(0)> = 0 => <H_E(t)> == (2 / g_YM^2) * <W_plaq(t)>:
        assert vac_exact["total_energy"] == pytest.approx(0.0, abs=1e-14)
        assert vac_exact["electric_energy"] == pytest.approx(
            (2.0 / g_sq) * vac_exact["wilson_loop"], abs=1e-14
        )
        probs = vac_exact["irrep_sector_probabilities"]
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-14)
        assert probs["chi_1"] == pytest.approx(probs["chi_2"], abs=1e-14)
        assert probs["rho_2_1"] == pytest.approx(probs["rho_2_2"], abs=1e-14)


def test_trotter_first_and_second_order_empirical_convergence_slopes(
    scaling_report: Dict[str, Any],
) -> None:
    assert all(scaling_report["checks"].values())
    for g_sq in dyn.COUPLING_REGIMES:
        entry = scaling_report["by_coupling"][f"g_ym_sq_{g_sq:.1f}"]
        assert 0.99 <= entry["lie_loglog_slope"] <= 1.01
        assert 1.99 <= entry["strang_loglog_slope"] <= 2.01
    assert scaling_report["max_moderate_step_strang_error_n64"] < 1e-4
    assert scaling_report["max_fine_step_strang_error_n2048"] < 1e-7


# ---------------------------------------------------------------------------
# Part 4: Guarded Qiskit & pytket Circuit Synthesis & Native Gate Counts
# ---------------------------------------------------------------------------


def test_qiskit_5q_and_multilink_circuits_and_native_cz_compression(
    live_manifest: Dict[str, Any],
) -> None:
    pytest.importorskip("qiskit")
    comp = live_manifest["part5_native_circuit_compilations"]
    assert comp["status"] == "OK"
    assert comp["qiskit_available"] is True

    errs = comp["direct_circuit_errors"]
    assert errs["ub_5q_2norm_error"] < 1e-14
    assert errs["ue_structured_5q_2norm_error"] < 1e-14
    assert errs["ue_peter_weyl_qft_5q_2norm_error"] < 1e-12
    assert errs["ue_structured_vs_qft_2norm_error"] < 1e-12
    assert errs["strang_structured_5q_2norm_error"] < 1e-14
    assert errs["ub_2link_15q_statevector_l2_error"] < 1e-13
    assert errs["ub_2link_15q_ancilla_leakage"] < 1e-28

    summary = comp["gate_count_summary"]
    assert summary["cz_ub_structured_5q_all_to_all"] == 16
    assert summary["cz_ub_structured_5q_heavy_hex_T5"] == 19
    assert summary["cz_strang_structured_5q_all_to_all"] < summary["cz_dense_unitary_5q_all_to_all"]
    assert (
        summary["cz_strang_structured_5q_heavy_hex_T5"]
        < summary["cz_dense_unitary_5q_heavy_hex_T5"]
    )
    assert summary["cz_compression_unfixed_20q_vs_5q_all_to_all"] > 10.0
    assert summary["cz_compression_unfixed_25q_vs_5q_all_to_all"] > 10.0
    assert summary["cz_compression_unfixed_25q_vs_5q_heavy_hex"] > 10.0

    for entry in comp["proxy_compilations_5q"]:
        assert entry["execution_tier"] == "PROXY"
        assert entry["provider_job_id"] is None
        assert entry["equivalent_up_to_global_phase"] is True


def test_pytket_zzphase_compilation_and_unitary_equivalence(
    live_manifest: Dict[str, Any],
) -> None:
    pytest.importorskip("qiskit")
    pytest.importorskip("pytket")
    comp = live_manifest["part5_native_circuit_compilations"]
    assert comp["pytket_available"] is True

    summary = comp["gate_count_summary"]
    assert summary["zzphase_ub_structured_5q"] == 16
    assert summary["zzphase_strang_structured_5q"] < summary["zzphase_dense_unitary_5q"]

    for entry in comp["proxy_compilations_pytket"]:
        assert entry["execution_tier"] == "PROXY"
        assert entry["provider_job_id"] is None
        assert entry["equivalent_up_to_global_phase"] is True


# ---------------------------------------------------------------------------
# Part 5: Manifest Determinism, Offline No-SDK Fallback & SYS-024 Cleanliness
# ---------------------------------------------------------------------------


def test_manifest_bt2t003_determinism_and_rfc8785_jcs(
    live_manifest: Dict[str, Any],
) -> None:
    assert MANIFEST_PATH.is_file(), f"Missing manifest: {MANIFEST_PATH}"
    saved_text = MANIFEST_PATH.read_text(encoding="utf-8")
    saved = json.loads(saved_text)

    assert saved["experiment_id"] == "OSIRIS-BT2T-003"
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
    assert saved["all_checks_passed"] is True
    assert all(saved["checks"].values())

    body = {k: v for k, v in saved.items() if k != "rfc8785_canonical_sha256"}
    pure_bytes = verify_bt2t003.pure_python_jcs_bytes(body)
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
        "class _Blocker(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path, target=None):\n"
        "        if fullname.split('.')[0] in BLOCKED:\n"
        "            return None\n"
        "        return None\n"
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
        f"sys.path.insert(0, {str(PLAQUETTE_DIR)!r})\n"
        "import verify_bt2t003\n"
        "m = verify_bt2t003.build_manifest_bt2t003()\n"
        "assert m['all_checks_passed'] is True\n"
        "assert m['part5_native_circuit_compilations']['status'] == 'SKIPPED_NO_QISKIT'\n"
        "assert len(m['checks']) >= 22\n"
        f"saved = json.loads(Path({str(MANIFEST_PATH)!r}).read_text(encoding='utf-8'))\n"
        "body = {k: v for k, v in saved.items() if k != 'rfc8785_canonical_sha256'}\n"
        "assert verify_bt2t003.jcs_sha256(body) == saved['rfc8785_canonical_sha256']\n"
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
    for path in sorted(PLAQUETTE_DIR.glob("*")):
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
