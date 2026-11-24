"""Tests for Offline Card B (`OSIRIS-ECHO` Stop/Go Gate) and Check #6 (`PHY-9` Relabeling + Injection-Recovery)."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RXGate, RZZGate
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import RemoveBarriers
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, coherent_unitary_error, depolarizing_error

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osiris_cli import claims  # noqa: E402

ECHO_GATE_PY = REPO_ROOT / "experiments" / "echo_placement" / "echo_placement_gate.py"
ECHO_REPORT_PATH = (
    REPO_ROOT / "experiments" / "echo_placement" / "artifacts" / "echo_stopgo_report.json"
)
AFIELD_HARNESS_PY = (
    REPO_ROOT / "research" / "a_field_study" / "experiments" / "falsification_harness.py"
)
AFIELD_SPEC_PATH = REPO_ROOT / "research" / "a_field_study" / "experiments" / "experiment_spec.json"
AFIELD_RESULTS_PATH = (
    REPO_ROOT / "research" / "a_field_study" / "experiments" / "experiment_results.json"
)


def _load_file_module(mod_name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    assert spec is not None and spec.loader is not None, f"Cannot load {file_path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


echo_mod = _load_file_module("_test_echo_placement_gate", ECHO_GATE_PY)
phy9_mod = _load_file_module("_test_afield_falsification_harness", AFIELD_HARNESS_PY)


@pytest.fixture(scope="module")
def echo_report() -> dict:
    assert ECHO_REPORT_PATH.is_file(), f"Missing artifact: {ECHO_REPORT_PATH}"
    return json.loads(ECHO_REPORT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def afield_spec() -> dict:
    return json.loads(AFIELD_SPEC_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def afield_results() -> dict:
    assert AFIELD_RESULTS_PATH.is_file(), f"Missing artifact: {AFIELD_RESULTS_PATH}"
    return json.loads(AFIELD_RESULTS_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Part 1: Card B (OSIRIS-ECHO) Offline Placement-Ranking Stop/Go Gate
# ---------------------------------------------------------------------------


def test_heron_r2_156q_lattice_and_k_disjoint_t5_placements() -> None:
    edges = echo_mod.heron_r2_156q_coupling_edges()
    assert len(edges) == 176
    try:
        from qiskit_ibm_runtime.fake_provider import FakeFez

        fez_edges = sorted({tuple(sorted(e)) for e in FakeFez().coupling_map.get_edges()})
        assert edges == fez_edges
    except ImportError:
        pass

    placements = echo_mod.select_t5_candidate_placements(k_placements=10)
    assert len(placements) == 10
    assert placements[0]["physical_qubits"] == [21, 22, 23, 16, 24]

    seen_qubits: set[int] = set()
    edge_set = set(edges)
    for p in placements:
        qs = p["physical_qubits"]
        assert len(qs) == 5
        assert seen_qubits.isdisjoint(qs), "Candidate T5 placements must be vertex-disjoint"
        seen_qubits.update(qs)
        expected_internal = {
            tuple(sorted((qs[0], qs[1]))),
            tuple(sorted((qs[1], qs[2]))),
            tuple(sorted((qs[2], qs[3]))),
            tuple(sorted((qs[2], qs[4]))),
        }
        induced = {
            tuple(sorted((u, v)))
            for i, u in enumerate(qs)
            for v in qs[i + 1 :]
            if tuple(sorted((u, v))) in edge_set
        }
        assert induced == expected_internal

    topo_scores = [p["topology_degree_score"] for p in placements]
    assert np.std(topo_scores) > 0.0


def test_echo_simulator_matches_qiskit_aer_to_machine_precision() -> None:
    qc = QuantumCircuit(5)
    qc.sx(0)
    qc.rz(0.41, 0)
    qc.x(2)
    qc.cz(0, 1)
    qc.sx(1)
    qc.cz(1, 2)
    qc.cz(2, 3)
    qc.sx(3)
    qc.cz(2, 4)

    p_1q = [0.002, 0.003, 0.001, 0.004, 0.0025]
    p_cz = {(0, 1): 0.015, (1, 2): 0.032, (2, 3): 0.008, (2, 4): 0.021}
    overrot = [0.02, -0.015, 0.03, 0.01, -0.025]
    p_ro = [0.012, 0.025, 0.008, 0.040, 0.015]

    nm = NoiseModel()
    for q in range(5):
        err_1q = coherent_unitary_error(Operator(RXGate(overrot[q]))).compose(
            depolarizing_error(p_1q[q], 1)
        )
        nm.add_quantum_error(err_1q, ["sx", "x"], [q])
    for (u, v), lam in p_cz.items():
        zz_ang = 0.5 * (overrot[u] + overrot[v])
        err_2q = coherent_unitary_error(Operator(RZZGate(zz_ang))).compose(
            depolarizing_error(lam, 2)
        )
        nm.add_quantum_error(err_2q, ["cz"], [u, v])
        nm.add_quantum_error(err_2q, ["cz"], [v, u])

    qc_aer = qc.copy()
    qc_aer.save_density_matrix()
    dm_aer = np.asarray(
        AerSimulator(method="density_matrix", noise_model=nm)
        .run(qc_aer)
        .result()
        .data(0)["density_matrix"]
    )
    diag_aer = np.real(np.diag(dm_aer)).reshape(2, 2, 2, 2, 2)
    for q in range(5):
        ax = 4 - q
        m_ro = np.array([[1.0 - p_ro[q], p_ro[q]], [p_ro[q], 1.0 - p_ro[q]]])
        diag_aer = np.moveaxis(np.tensordot(m_ro, diag_aer, axes=([1], [ax])), 0, ax)
    aer_ro_probs = diag_aer.reshape(32)

    our_probs = echo_mod.simulate_t5_noisy_probabilities(qc, p_cz, p_1q, overrot, p_ro)[0]
    assert float(np.max(np.abs(our_probs - aer_ro_probs))) < 1e-13


def test_card_b_controls_identity_perturbation_and_non_cancellation(echo_report: dict) -> None:
    controls = echo_report["controls"]

    # Control 1: 0-CZ identity return > 16-CZ workload echo return on every cell
    c1 = controls["control_1_no_workload_identity_echo"]
    assert c1["cz_count"] == 0
    assert c1["passed"] is True
    assert c1["cells_where_identity_exceeds_workload_echo"] == c1["total_placement_window_cells"]
    assert c1["mean_identity_return"] > c1["mean_workload_echo_return"]

    # Control 2: Coherent over-rotation perturbation strictly lowers return probability
    c2 = controls["control_2_deliberate_overrotation_perturbation"]
    assert c2["passed"] is True
    assert c2["monotone_drop_verified"] is True
    ideal_curve = c2["ideal_return_curve"]
    noisy_curve = c2["mean_noisy_return_curve"]
    assert ideal_curve[0] == pytest.approx(1.0, abs=1e-9)
    assert all(ideal_curve[i] > ideal_curve[i + 1] for i in range(len(ideal_curve) - 1))
    assert all(noisy_curve[i] > noisy_curve[i + 1] for i in range(len(noisy_curve) - 1))

    # Control 3: Non-cancellation audit WITHOUT relying solely on barriers
    bundle = echo_mod.build_echo_probe_and_controls()
    audit = bundle["audit"]
    assert audit["non_cancelled_cz_verified"] is True
    assert audit["ideal_noiseless_return_prob"] == pytest.approx(1.0, abs=1e-11)
    assert audit["forward_cz_count"] == 8
    assert audit["echo_compiled_cz_count"] >= 2 * audit["forward_cz_count"]
    assert audit["echo_nobarrier_opt1_cz_count"] >= 2 * audit["forward_cz_count"]
    assert audit["echo_nobarrier_opt3_cz_count"] >= 2 * audit["forward_cz_count"]
    assert audit["adjacent_cancellable_cz_pairs"] == 0
    assert audit["naive_palindrome_without_barrier_cz_count"] == 0
    assert audit["global_unitary_is_non_identity_reflection"] is True

    # Independent check: strip barriers directly in test and verify CZ gates remain intact
    cmap = CouplingMap([[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [2, 4], [4, 2]])
    stripped = RemoveBarriers()(bundle["echo_circuit"])
    reopt = transpile(
        stripped,
        basis_gates=["rz", "sx", "x", "cz"],
        coupling_map=cmap,
        optimization_level=3,
        seed_transpiler=999,
    )
    assert int(reopt.count_ops().get("cz", 0)) >= 16


def test_card_b_three_unseen_heldout_circuit_classes() -> None:
    heldout = echo_mod.build_heldout_reference_circuits()
    assert set(heldout.keys()) == {
        "bt2t_72cz",
        "random_clifford_30cz",
        "non_clifford_mirror_24cz",
    }
    assert heldout["bt2t_72cz"]["cz_count"] == 72
    assert 24 <= heldout["random_clifford_30cz"]["cz_count"] <= 36
    assert heldout["non_clifford_mirror_24cz"]["cz_count"] == 24
    assert heldout["random_clifford_30cz"]["ideal_peak_prob"] == pytest.approx(1.0, abs=1e-12)
    assert heldout["non_clifford_mirror_24cz"]["ideal_peak_prob"] == pytest.approx(1.0, abs=1e-12)


def test_card_b_stopgo_window_bootstrap_and_provenance(echo_report: dict) -> None:
    assert echo_report["execution_tier"] == "SIMULATED"
    assert echo_report["provider_job_id"] is None
    assert echo_report["qpu_seconds"] == 0
    assert echo_report["hardware_execution_authorized"] is False
    assert echo_report["mechanism_inferred"] == "none"

    design = echo_report["design"]
    assert design["candidate_t5_placements_K"] >= 8
    assert design["calibration_windows_M"] >= 12
    assert design["unit_of_inference"] == "calibration_window"
    assert design["bootstrap_replicates_B"] == 2000

    rc = echo_report["ranking_comparison"]
    rankers = rc["rankers"]
    assert set(rankers.keys()) == {
        "ECHO_RANKED",
        "CALIBRATION_RANKED",
        "TOPOLOGY_DEGREE_ONLY",
        "RANDOM_PLACEMENT",
    }
    assert rankers["ECHO_RANKED"]["mean_spearman"] > 0.85
    assert rc["delta_spearman_point"] > 0.30
    assert rc["delta_spearman_lcl95"] > 0.0
    assert rc["delta_spearman_lcl95_one_sided"] > 0.0
    assert rc["top1_regret_reduction_vs_best_comparator"]["point"] > 0.03
    assert rc["top1_regret_reduction_vs_best_comparator"]["ci95"][0] > 0.0

    for hc_name, hc_stats in rc["per_heldout_circuit_class"].items():
        assert hc_stats["delta_echo_minus_cal_mean"] > 0.30, hc_name
        assert hc_stats["delta_echo_minus_cal_ci95"][0] > 0.0, hc_name

    sg = echo_report["stop_go_decision"]
    assert sg["echo_recovers_ranking"] is True
    assert sg["delta_spearman_lcl95_gt_zero"] is True
    assert sg["all_mandatory_controls_passed"] is True
    assert sg["mechanism_inferred"] == "none"
    assert sg["verdict"] == "PASS_OFFLINE_STOPGO_GATE"


def test_card_b_deterministic_report_regeneration(tmp_path: Path) -> None:
    out = tmp_path / "echo_stopgo_report_regen.json"
    echo_mod.run_echo_placement_stopgo_gate(output_path=out)
    assert out.read_text(encoding="utf-8") == ECHO_REPORT_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Part 2: Check #6 (PHY-9) Relabeling & Injection-Recovery Study
# ---------------------------------------------------------------------------


def test_phy9_zero_signal_relabeling_never_emits_refuted(
    afield_spec: dict, afield_results: dict
) -> None:
    raw_text = AFIELD_RESULTS_PATH.read_text(encoding="utf-8")
    assert "REFUTED" not in raw_text
    assert "HYPOTHESIS_REFUTED_BY_EMPIRICAL_SIMULATION" not in raw_text

    assert afield_results["execution_tier"] == "SIMULATED"
    assert afield_results["provider_job_id"] is None
    assert afield_results["physical_claim_assessment"] == "UNVERIFIED"
    assert afield_results["all_hypotheses_falsified"] is False
    assert (
        afield_results["organism_status_update"]
        == "SIMULATION_CALIBRATED_PHYSICAL_CLAIM_UNVERIFIED"
    )
    assert afield_results["zero_signal_controls_below_threshold"] is True

    exps = afield_results["experiments_executed"]
    assert len(exps) == 2
    for exp in exps:
        assert exp["execution_tier"] == "SIMULATED"
        assert exp["provider_job_id"] is None
        assert exp["physical_claim_assessment"] == "UNVERIFIED"
        assert exp["simulation_outcome"] == "BELOW_THRESHOLD_WITH_ZERO_INJECTED_SIGNAL"

    pos_cav = phy9_mod.run_cavity_experiment_sim(
        afield_spec["experiments"][0], seed=42, injected_signal_hz=3.0
    )
    pos_flux = phy9_mod.run_fluxoid_experiment_sim(
        afield_spec["experiments"][1], seed=42, injected_signal=0.030
    )
    assert pos_cav["simulation_outcome"] == "INJECTED_SIGNAL_DETECTED_ABOVE_THRESHOLD"
    assert pos_cav["physical_claim_assessment"] == "UNVERIFIED"
    assert pos_flux["simulation_outcome"] == "INJECTED_SIGNAL_DETECTED_ABOVE_THRESHOLD"
    assert pos_flux["physical_claim_assessment"] == "UNVERIFIED"


def test_phy9_injection_recovery_fpr_power_bias_and_rmse(afield_results: dict) -> None:
    inj = afield_results["injection_recovery"]
    assert inj["execution_tier"] == "SIMULATED"
    assert inj["provider_job_id"] is None
    assert inj["physical_claim_assessment"] == "UNVERIFIED"
    assert inj["n_trials_per_amplitude"] >= 2000
    assert inj["sigma_multipliers"] == [0.0, 1.0, 2.0, 3.0, 5.0, 6.0]

    for exp_id, expected_sigma in (("EXP-AFIELD-01", 0.5), ("EXP-AFIELD-02", 0.005)):
        proto = inj["protocols"][exp_id]
        assert proto["execution_tier"] == "SIMULATED"
        assert proto["provider_job_id"] is None
        assert proto["physical_claim_assessment"] == "UNVERIFIED"
        assert proto["measurement_sigma"] == pytest.approx(expected_sigma)
        assert proto["threshold_3sigma"] == pytest.approx(3.0 * expected_sigma)
        assert proto["monotone_power_curve_verified"] is True

        # False-positive rate at 0*sigma (< 0.01), ~50% power at 3*sigma, >99% recovery at 6*sigma
        assert 0.0 <= proto["false_positive_rate_0sigma"] < 0.01
        assert 0.005 < proto["power_at_1sigma"] < 0.06
        assert 0.10 < proto["power_at_2sigma"] < 0.22
        assert 0.45 <= proto["power_at_3sigma"] <= 0.55
        assert proto["power_at_5sigma"] > 0.95
        assert proto["power_at_6sigma"] > 0.99

        levels = {lv["injected_sigma_multiple"]: lv for lv in proto["levels"]}
        if exp_id == "EXP-AFIELD-01":
            for mult, lv in levels.items():
                assert abs(lv["estimation_bias"]) < 0.04, (exp_id, mult)
                assert lv["estimation_rmse"] == pytest.approx(expected_sigma, rel=0.08)
        else:
            expected_zero_bias = expected_sigma * math.sqrt(2.0 / math.pi)
            assert levels[0.0]["estimation_bias"] == pytest.approx(expected_zero_bias, rel=0.08)
            assert abs(levels[3.0]["estimation_bias"]) < 0.0005
            assert abs(levels[6.0]["estimation_bias"]) < 0.0005
            assert levels[6.0]["estimation_rmse"] == pytest.approx(expected_sigma, rel=0.08)


def test_phy9_deterministic_results_regeneration(afield_spec: dict) -> None:
    regen = phy9_mod.execute_all_experiments()
    saved = json.loads(AFIELD_RESULTS_PATH.read_text(encoding="utf-8"))
    assert regen == saved

    with pytest.raises(ValueError, match=">= 2000"):
        phy9_mod.run_injection_recovery_study(spec_data=afield_spec, n_trials=500)


def test_sys024_claims_firewall_clean(echo_report: dict, afield_results: dict) -> None:
    echo_text = json.dumps(echo_report, sort_keys=True)
    afield_text = json.dumps(afield_results, sort_keys=True)
    assert claims.flagged_in_reply(echo_text) == []
    assert claims.flagged_in_reply(afield_text) == []
    assert claims.flagged_in_reply(echo_report["stop_go_decision"]["scope_note"]) == []
    assert claims.flagged_in_reply(afield_results["governance_note"]) == []
