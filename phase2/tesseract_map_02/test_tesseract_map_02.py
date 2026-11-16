"""
Unit & Simulation Verification Tests for EXPERIMENT TESSERACT-MAP-02
====================================================================
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_ibm_runtime")

from qiskit_ibm_runtime.fake_provider import FakeKingston

from phase2.tesseract_map_02.tesseract_map_02 import (
    build_tesseract_map_02_arms,
    compute_required_blocks,
    simulate_40q_hardware_block_outcomes,
    verify_archived_kingston_rerun,
    verify_logical_equivalence,
)


def test_logical_equivalence_and_fault_propagation_difference():
    eq = verify_logical_equivalence(n_qubits_sv=10, n_qubits_gf2=40, theta_deg=51.7)
    assert math.isclose(eq["statevector_overlap_star_vs_chain"], 1.0, abs_tol=1e-12)
    assert eq["max_prob_diff_vs_analytic_ideal"] < 1e-12
    assert eq["max_prob_diff_with_inert_rz"] < 1e-12
    assert eq["gf2_40q_core_propagation_identical"] is True
    assert eq["fault_weight_x_on_q2_star"] == 1
    assert eq["fault_weight_x_on_q2_chain"] == 19


def test_40q_compilation_and_layout_saving_on_fake_kingston():
    backend = FakeKingston()
    circuits, manifests, path_info = build_tesseract_map_02_arms(backend, n_qubits=40)

    assert len(circuits) == 9
    assert path_info["best_path_cost"] < path_info["median_path_cost"] <= path_info["worst_path_cost"]

    # Every compiled circuit MUST save all 40 distinct physical qubit indices (fixes E-83!)
    for arm_id, man in manifests.items():
        assert len(man.final_physical_layout) == 40
        assert len(set(man.final_physical_layout)) == 40

    # Chain arms on contiguous paths MUST use exactly 39 two-qubit gates (zero SWAPs!)
    assert manifests["C4_CHAIN_BESTPATH_OPT1"].twoq_gate_count == 39
    assert manifests["C5_CHAIN_BESTPATH_OPT3"].twoq_gate_count == 39
    assert manifests["C6_CHAIN_MEDPATH_OPT3"].twoq_gate_count == 39

    # Star arms on heavy-hex MUST incur substantial SWAP overhead vs 39-CZ chain:
    # C1 (free layout) > 100 CZ; C2 (midpoint path + opt3 KAK consolidation) >= 75 CZ (~2x chain!)
    assert manifests["C1_STAR_FREE_OPT3"].twoq_gate_count > 100
    assert manifests["C2_STAR_BESTPATH_OPT3"].twoq_gate_count >= 75
    assert manifests["C3_STAR_BESTPATH_OPT1"].twoq_gate_count > manifests["C2_STAR_BESTPATH_OPT3"].twoq_gate_count
    assert manifests["C0_SPAM_ALL0"].twoq_gate_count == 0
    assert manifests["C0_SPAM_ALL1"].twoq_gate_count == 0


def test_blocked_simulation_and_statistical_plan():
    req = compute_required_blocks(target_mde=0.015, between_block_sd_diff=0.014)
    assert 8 <= req["required_blocks"] <= 20
    assert req["achieved_power"] >= 0.85

    backend = FakeKingston()
    circuits, manifests, _ = build_tesseract_map_02_arms(backend, n_qubits=40)
    sim = simulate_40q_hardware_block_outcomes(
        backend, circuits, manifests, n_blocks=12, shots_per_circuit=2048, seed=12345
    )
    assert sim["qpu_submitted"] is False

    # Routing + logical rewrite contrast (C5 chain vs C2 star on IDENTICAL best_path qubits)
    # must be strongly positive and Holm-significant!
    contrasts = {
        c["contrast"]: c
        for c in sim["paired_contrasts_classical_fidelity"]["contrasts"]
    }
    c_route = contrasts["routing_and_rewrite_at_opt3"]
    assert c_route["mean_diff"] > 0.05
    assert c_route["significant_holm_0_05"] is True

    # Qubit-selection contrast (best_path vs median_path at identical 39 CZ gates) must be positive!
    c_qub = contrasts["qubit_selection_at_fixed_39cz"]
    assert c_qub["mean_diff"] > 0.0


def test_archived_kingston_rerun_reproduction():
    arc = verify_archived_kingston_rerun()
    if arc["available"]:
        assert arc["circuits"]["P0_replication"]["matches_archive_fidelity"] is True
        assert arc["circuits"]["P1_ghz_control"]["matches_archive_fidelity"] is True
        assert arc["circuits"]["P2_chain_mapped"]["matches_archive_fidelity"] is True
        assert math.isclose(
            arc["circuits"]["P0_replication"]["recomputed"]["four_string_support"], 0.05305, abs_tol=1e-6
        )
        assert math.isclose(
            arc["circuits"]["P2_chain_mapped"]["recomputed"]["four_string_support"], 0.17995, abs_tol=1e-6
        )
