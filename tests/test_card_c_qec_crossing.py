"""Tests for Card C (`OSIRIS-QEC`): Offline Crossing Gate & Auditable Space-Time Decoder."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import qec_card_c  # noqa: E402

from osiris_cli import claims  # noqa: E402

GATE_PY = REPO_ROOT / "experiments" / "qec_card_c" / "qec_crossing_gate.py"
REPORT_PATH = (
    REPO_ROOT / "experiments" / "qec_card_c" / "artifacts" / "card_c_qec_crossing_report.json"
)


def _load_gate_module() -> Any:
    spec = importlib.util.spec_from_file_location("_test_qec_card_c_gate", GATE_PY)
    assert spec is not None and spec.loader is not None, f"Cannot load {GATE_PY}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_test_qec_card_c_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


gate_mod = _load_gate_module()


@pytest.fixture(scope="module")
def card_c_report() -> dict[str, Any]:
    assert REPORT_PATH.is_file(), f"Missing Card C artifact: {REPORT_PATH}"
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Auditable Space-Time DEM, MWPM, Single-Fault Recovery & Exact ML Lookup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code_id", "basis", "min_expected_locations"),
    [
        ("repetition_d3_5q", "Z", 200),
        ("repetition_d3_5q", "X", 250),
        ("rotated_surface_d3_17q", "Z", 1300),
        ("rotated_surface_d3_17q", "X", 1350),
    ],
)
def test_100_percent_single_fault_recovery_across_all_codes_and_bases(
    code_id: str, basis: str, min_expected_locations: int
) -> None:
    profiles = gate_mod.build_published_target_profiles()
    prof = profiles["H2_1_PUBLISHED"] if "surface" in code_id else profiles["FEZ_PUBLISHED"]
    audit = gate_mod.audit_single_fault_recovery(
        code=code_id, basis=basis, rounds=3, profile=prof
    )
    assert audit["total_single_fault_locations"] >= min_expected_locations
    assert audit["recovered_single_fault_locations"] == audit["total_single_fault_locations"]
    assert audit["single_fault_recovery_rate"] == 1.0
    assert audit["zero_single_fault_failures"] is True


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_mwpm_exact_agreement_with_ml_syndrome_table_on_rep3(basis: str) -> None:
    prof = gate_mod.build_published_target_profiles()["FEZ_PUBLISHED"]

    # For R=1 (16 syndromes) and R=2 (64 syndromes), MWPM and Exact ML agree 100% on EVERY syndrome
    for rounds in (1, 2):
        dem = gate_mod.build_repetition_d3_dem(rounds=rounds, profile=prof, basis=basis)
        mwpm = qec_card_c.auditable_mwpm_decoder(dem)
        ml = gate_mod.ExactMLSyndromeTableDecoder(dem)

        n_syn = 1 << dem.n_detectors
        assert float(np.sum(ml.joint_table)) == pytest.approx(1.0, abs=1e-12)

        all_masks = np.arange(n_syn, dtype=np.int64)
        mwpm_preds = mwpm.decode_batch(all_masks)
        ml_preds = ml.decode_batch(all_masks)
        assert np.array_equal(mwpm_preds, ml_preds), (
            f"Expected 100% syndrome-table match at R={rounds} ({basis})"
        )

    # At R=3 (256 syndromes), verify 100% agreement on all single-fault syndromes & tight P_fail match
    dem3 = gate_mod.build_repetition_d3_dem(rounds=3, profile=prof, basis=basis)
    mwpm3 = gate_mod.auditable_mwpm_decoder(dem3)
    ml3 = gate_mod.ExactMLSyndromeTableDecoder(dem3)
    for dmask, _ in dem3.mechanisms:
        assert mwpm3.decode_mask(dmask) == ml3.decode_mask(dmask)

    mwpm3_all = mwpm3.decode_batch(np.arange(1 << dem3.n_detectors, dtype=np.int64))
    p_fail_mwpm = float(
        np.sum(ml3.joint_table[np.arange(1 << dem3.n_detectors), 1 - mwpm3_all])
    )
    assert abs(p_fail_mwpm - ml3.exact_logical_failure_prob) < 5e-5


def test_bitmask_dp_and_blossom_fallback_consistency() -> None:
    prof = gate_mod.build_published_target_profiles()["FEZ_PUBLISHED"]
    dem = gate_mod.build_rotated_surface_d3_dem(rounds=5, profile=prof, basis="X")
    decoder = gate_mod.auditable_mwpm_decoder(dem)

    # Test explain_matching audit trail and exact Blossom equivalence on synthetic k=6 and k=18 defect masks
    mask_k6 = (1 << 0) | (1 << 3) | (1 << 5) | (1 << 9) | (1 << 14) | (1 << 19)
    explanation = decoder.explain_matching(mask_k6)
    assert len(explanation) >= 3
    xor_obs = 0
    for edge in explanation:
        assert edge["weight"] >= 0.0
        xor_obs ^= int(edge["obs_flip"])
    assert decoder.decode_mask(mask_k6) == xor_obs

    # Exercise k=18 (> 16) Blossom path and verify weight matches NetworkX min_weight_matching directly
    nodes_18 = list(range(18))
    mask_k18 = sum(1 << i for i in nodes_18)
    exp_18 = decoder.explain_matching(mask_k18)
    b_node = decoder.boundary_node
    g = nx.Graph()
    for i, u in enumerate(nodes_18):
        g.add_edge(i, 18 + i, weight=float(decoder.dist_matrix[u, b_node]))
        for j in range(i + 1, 18):
            v = nodes_18[j]
            g.add_edge(i, j, weight=float(decoder.dist_matrix[u, v]))
            g.add_edge(18 + i, 18 + j, weight=0.0)
    m_nx = nx.algorithms.matching.min_weight_matching(g)
    cost_nx = sum(g[u][v]["weight"] for u, v in m_nx)
    cost_dec = sum(item["weight"] for item in exp_18)
    assert cost_dec == pytest.approx(cost_nx, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. 5Q Kraus Density-Matrix Verification Against Analytical & Qiskit Aer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("rounds", [1, 2, 3, 5, 7, 9])
def test_5q_kraus_simulator_matches_analytical_arm1_to_machine_precision(
    basis: str, rounds: int
) -> None:
    for prof in gate_mod.build_published_target_profiles().values():
        p_ana_pure = gate_mod.analytical_arm1_unencoded_failure(
            rounds=rounds, profile=prof, basis=basis, include_spam_and_coherent=False
        )
        p_sim_pure = gate_mod.simulate_5q_arm1_kraus_exact(
            rounds=rounds, profile=prof, basis=basis, include_spam_and_coherent=False
        )
        assert abs(p_ana_pure - p_sim_pure) < 1e-14

        p_ana_full = gate_mod.analytical_arm1_unencoded_failure(
            rounds=rounds, profile=prof, basis=basis, include_spam_and_coherent=True
        )
        p_sim_full = gate_mod.simulate_5q_arm1_kraus_exact(
            rounds=rounds, profile=prof, basis=basis, include_spam_and_coherent=True
        )
        assert abs(p_ana_full - p_sim_full) < 1e-14


def test_5q_kraus_t1_t2_matches_qiskit_aer_density_matrix() -> None:
    prof = gate_mod.build_published_target_profiles()["FEZ_PUBLISHED"]
    rounds = 5
    dt_ns = rounds * prof.t_round_us * 1000.0

    qc = QuantumCircuit(1)
    qc.h(0)
    qc.id(0)
    qc.h(0)
    qc.save_density_matrix()

    nm = NoiseModel()
    err = thermal_relaxation_error(prof.t1_us * 1000.0, prof.t2_us * 1000.0, dt_ns)
    nm.add_all_qubit_quantum_error(err, ["id"])

    sim = AerSimulator(method="density_matrix", noise_model=nm)
    dm_aer = np.asarray(sim.run(qc).result().data(0)["density_matrix"])
    p_aer = float(np.real(dm_aer[1, 1]))
    p_kraus = gate_mod.simulate_5q_arm1_kraus_exact(
        rounds=rounds, profile=prof, basis="X", include_spam_and_coherent=False
    )
    assert abs(p_aer - p_kraus) < 1e-12


# ---------------------------------------------------------------------------
# 3. Card C Artifact Schema, Epistemic Separation, Crossings & Determinism
# ---------------------------------------------------------------------------


def test_artifact_governance_noise_documentation_and_arm_separation(
    card_c_report: dict[str, Any],
) -> None:
    assert card_c_report["schema"] == "osiris.qec_card_c.crossing_report/1"
    assert card_c_report["execution_tier"] == "SIMULATED"
    assert card_c_report["provider_job_id"] is None
    assert card_c_report["qpu_seconds"] == 0
    assert card_c_report["hardware_execution_authorized"] is False

    ndoc = card_c_report["noise_model_documentation"]
    assert ndoc["omitted_mechanisms"] == ["non_computational_state_2_leakage"]
    assert "leakage" in ndoc["omitted_mechanisms_impact_note"].lower()
    for mech in (
        "amplitude_damping_t1",
        "phase_damping_t2",
        "coherent_over_rotation_1q_2q",
        "mid_circuit_and_terminal_readout_confusion",
        "active_reset_bit_flip",
        "correlated_spectator_zz_crosstalk",
    ):
        assert mech in ndoc["modeled_mechanisms"]

    vc = card_c_report["verification_controls"]
    assert vc["all_single_fault_audits_passed"] is True
    assert vc["arm1_5q_kraus_max_abs_diff"] < 1e-12

    # Check every cell separates unconditional arms from postselected arm3 + acceptance A(R),
    # and labels Pauli-frame vs physical feed-forward explicitly
    assert len(card_c_report["cells"]) == 6
    for cell_key, cell in card_c_report["cells"].items():
        assert cell["rounds_evaluated"] == [1, 2, 3, 5, 7, 9]
        u_arms = cell["unconditional_arms"]
        assert set(u_arms.keys()) == {
            "arm1_unencoded",
            "arm2_encoded_no_correction",
            "arm4a_decoded_pauli_frame",
            "arm4b_decoded_physical_feedforward",
        }
        assert "arm3_postselected_detection" not in u_arms

        a4a = u_arms["arm4a_decoded_pauli_frame"]
        a4b = u_arms["arm4b_decoded_physical_feedforward"]
        assert a4a["decoding_mode"] == "offline_pauli_frame"
        assert a4a["feedforward_latency_us"] == 0.0
        assert a4a["conditional_physical_gates"] is False
        assert a4b["decoding_mode"] == "physical_feedforward"
        assert a4b["feedforward_latency_us"] > 0.0
        assert a4b["conditional_physical_gates"] is True

        # Physical feed-forward latency + conditional pulses must not outperform zero-latency Pauli frame
        assert (
            a4b["fit"]["epsilon_L_at_max_round"] >= a4a["fit"]["epsilon_L_at_max_round"] - 1e-4
        ), cell_key

        ps = cell["postselected_arm_reported_separately"]["arm3_postselected_detection"]
        assert ps["postselection_applied"] is True
        acc_vals = [ps["acceptance_rate_by_round"][str(r)] for r in (1, 2, 3, 5, 7, 9)]
        assert all(0.0 < a < 1.0 for a in acc_vals), cell_key
        # Acceptance rate A(R) must decrease strictly monotonically with rounds R
        assert all(acc_vals[i] > acc_vals[i + 1] for i in range(len(acc_vals) - 1)), cell_key

        if cell["code_id"] == "repetition_d3_5q":
            for r_key, chk in cell["exact_kraus_branch_crosschecks_r1_to_r3"].items():
                assert chk["abs_diff"] < 0.002, (cell_key, r_key, chk)


def test_crossing_verdicts_across_published_and_drifted_regimes(
    card_c_report: dict[str, Any],
) -> None:
    cells = card_c_report["cells"]

    # 1. [[3,1,3]] crosses on FEZ_PUBLISHED, FEZ_DRIFTED_SPAM (delayed for 4b), and H2_1_PUBLISHED
    fez_rep = cells["repetition_d3_5q__FEZ_PUBLISHED"]["cell_stop_go"]
    fez_drift_rep = cells["repetition_d3_5q__FEZ_DRIFTED_SPAM"]["cell_stop_go"]
    h2_rep = cells["repetition_d3_5q__H2_1_PUBLISHED"]["cell_stop_go"]
    assert fez_rep["verdict"] == "PASS_CROSSING_OBSERVED"
    assert fez_drift_rep["verdict"] == "PASS_CROSSING_OBSERVED"
    assert h2_rep["verdict"] == "PASS_CROSSING_OBSERVED"
    assert (
        cells["repetition_d3_5q__FEZ_DRIFTED_SPAM"]["unconditional_arms"][
            "arm4a_decoded_pauli_frame"
        ]["fit"]["epsilon_L_at_max_round"]
        > cells["repetition_d3_5q__FEZ_PUBLISHED"]["unconditional_arms"][
            "arm4a_decoded_pauli_frame"
        ]["fit"]["epsilon_L_at_max_round"]
    )

    # 2. [[9,1,3]] rotated surface code:
    #    - FEZ_PUBLISHED: offline Pauli-frame crosses, while physical feed-forward fails to beat Arm 1
    #    - FEZ_DRIFTED_SPAM (p_readout=0.032): crossing erased -> STOP_NO_CROSSING_AT_TESTED_ROUNDS
    #    - H2_1_PUBLISHED (native all-to-all): both 4a and 4b cross cleanly -> PASS_CROSSING_OBSERVED
    fez_surf = cells["rotated_surface_d3_17q__FEZ_PUBLISHED"]["cell_stop_go"]
    fez_drift_surf = cells["rotated_surface_d3_17q__FEZ_DRIFTED_SPAM"]["cell_stop_go"]
    h2_surf = cells["rotated_surface_d3_17q__H2_1_PUBLISHED"]["cell_stop_go"]

    assert fez_surf["verdict"] == "PASS_CROSSING_OBSERVED"
    assert fez_surf["arm4a_crosses_below_both"] is True
    assert fez_surf["arm4b_crosses_below_both"] is False

    assert fez_drift_surf["verdict"] == "STOP_NO_CROSSING_AT_TESTED_ROUNDS"
    assert fez_drift_surf["arm4a_crosses_below_both"] is False
    assert fez_drift_surf["arm4b_crosses_below_both"] is False

    assert h2_surf["verdict"] == "PASS_CROSSING_OBSERVED"
    assert h2_surf["arm4a_crosses_below_both"] is True
    assert h2_surf["arm4b_crosses_below_both"] is True
    assert h2_surf["r_cross_arm4a_vs_both"] in (1, 2, 3, 5, 7, 9)

    scope = card_c_report["stop_go_decision"]["scope_limit"]
    assert "single code distance" in scope.lower()
    assert "not evidence of scalable fault tolerance" in scope.lower()


def test_byte_for_byte_deterministic_regeneration(tmp_path: Path) -> None:
    out_a = tmp_path / "rep_a.json"
    out_b = tmp_path / "rep_b.json"
    gate_mod.run_card_c_qec_crossing_gate(
        output_path=out_a,
        seed=20261124,
        rounds_list=(1, 3, 5),
        n_shots_rep3=1024,
        n_shots_surf9=1024,
        n_boot=100,
    )
    gate_mod.run_card_c_qec_crossing_gate(
        output_path=out_b,
        seed=20261124,
        rounds_list=(1, 3, 5),
        n_shots_rep3=1024,
        n_shots_surf9=1024,
        n_boot=100,
    )
    assert out_a.read_bytes() == out_b.read_bytes()

    # Also verify canonical formatting and exact cell provenance against committed artifact
    raw_saved = REPORT_PATH.read_text(encoding="utf-8")
    parsed_saved = json.loads(raw_saved)
    assert raw_saved == json.dumps(parsed_saved, indent=2, sort_keys=True) + "\n"

    fez_prof = gate_mod.build_published_target_profiles()["FEZ_PUBLISHED"]
    regen_fez_rep = gate_mod.evaluate_rep3_kraus_arms(
        fez_prof,
        rounds_list=gate_mod.ROUNDS_GRID,
        n_shots=gate_mod.N_SHOTS_REP3_DEFAULT,
        n_boot=gate_mod.N_BOOT_DEFAULT,
        seed=gate_mod.DEFAULT_SEED,
    )
    assert regen_fez_rep == parsed_saved["cells"]["repetition_d3_5q__FEZ_PUBLISHED"]


def test_sys024_claims_firewall_clean(card_c_report: dict[str, Any]) -> None:
    serialized = json.dumps(card_c_report, sort_keys=True)
    assert claims.flagged_in_reply(serialized) == []
    assert claims.flagged_in_reply(card_c_report["stop_go_decision"]["governance_note"]) == []
    assert claims.flagged_in_reply(card_c_report["stop_go_decision"]["scope_limit"]) == []
    assert (
        claims.flagged_in_reply(
            card_c_report["noise_model_documentation"]["omitted_mechanisms_impact_note"]
        )
        == []
    )
