"""
Tests for Phase 2 Bounded Offline Workflows & Structured Claim Ingestion
========================================================================

Covers:
  - Structured STAGGERED_DD duration vs comparator schema & Tesseract namespace disambiguation
  - Section-addressed document review coverage (`sections_read` / `sections_not_read` guard)
  - Workflow A (`arXiv:2610.24077v1` color-code offline replay, group-locked split,
    abstention preservation, authors' (n_F/n)/(r+0.5) estimator, d=7 3/29000 sparse check)
  - Workflow B (`arXiv:2609.17102v1` NMR 1,420-circuit / 35,500-shot budget equation,
    5-tier diagnostic separation, window ISE, and signed-spectrum Wasserstein guard).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from osiris_cli.workflows import (
    DecoderPrediction,
    NMRErrorDiagnostics,
    NMRResourceConfig,
    ReviewCoverage,
    SyndromeRecord,
    authors_per_round_ler_estimator,
    build_canonical_staggered_dd_record,
    compute_attempted_shot_budget,
    disambiguate_tesseract_identifier,
    evaluate_nmr_spectral_window_error,
    evaluate_paired_decoders,
    transform_spectrum_to_measure,
    validate_locked_group_split,
)


def test_staggered_dd_structured_record_prevents_duration_conflation():
    rec = build_canonical_staggered_dd_record()
    d = rec.to_dict()
    assert d["claim_id"] == "STAGGERED_DD"
    assert d["source_status"] == "user_supplied_register"
    pts = {p["duration_us"]: p["estimate"] for p in d["points"]}
    assert pts == {8.0: 0.946, 16.0: 0.888, 32.0: 0.795}
    # Contrast vs simultaneous is ONLY defined at 32 us (0.795 - 0.59 = +0.205), NOT 0.946 - 0.888 at 8 us!
    comp_dur, delta = rec.contrast_vs_simultaneous_at_comparator()
    assert comp_dur == 32.0
    assert math.isclose(delta, 0.795 - 0.59, abs_tol=1e-12)


def test_tesseract_namespace_disambiguation():
    assert (
        disambiguate_tesseract_identifier("arXiv:2610.24077v1 color-code A* syndrome decoder")
        == "TESSERACT_COLOR_CODE_DECODER_EXT"
    )
    assert (
        disambiguate_tesseract_identifier("TESSERACT-MAP-02 chain-mapped 17.995% vs 5.305% star")
        == "TESSERACT_MAP_02"
    )
    assert (
        disambiguate_tesseract_identifier("40-qubit tesseract circuit 10^6x gain on ibm_torino")
        == "ERROR_SUPPRESSION_1E6"
    )


def test_review_coverage_blocks_unread_section_speculation():
    cov = ReviewCoverage(
        document_id="integration_brief_v1",
        declared_sections=(
            "source_manifest",
            "color_code_claims",
            "color_code_methods",
            "nmr_claims",
            "nmr_methods",
            "proposed_integrations",
        ),
        sections_read=("source_manifest", "color_code_claims", "color_code_methods"),
        source_versions=("arxiv:2610.24077v1", "arxiv:2609.17102v1"),
        external_unreviewed_claims=("quantinuum_nmr_21spin_projection",),
    )
    assert cov.sections_not_read == ("nmr_claims", "nmr_methods", "proposed_integrations")
    cov.assert_section_was_read("color_code_methods")
    with pytest.raises(PermissionError, match="UNREAD"):
        cov.assert_section_was_read("nmr_methods")


def test_workflow_a_color_code_group_split_and_sparse_d7_estimator():
    # Reproduce the paper's d=7 Tesseract benchmark statistic: 3 failures in 29,000 records, r=20
    d7 = authors_per_round_ler_estimator(n_failures=3, n_records=29_000, rounds=20)
    assert math.isclose(d7["per_round_ler_estimate"], (3.0 / 29_000.0) / 20.5, rel_tol=1e-12)
    assert d7["estimator_kind"] == "authors_half_round_estimator_arxiv_2610_24077v1"
    assert d7["sparse_failure_warning"] is True
    assert d7["per_round_ler_ci95_exact"][0] < d7["per_round_ler_estimate"] < d7["per_round_ler_ci95_exact"][1]

    # Reject round-sliced record
    with pytest.raises(ValueError, match="partial/split record forbidden"):
        SyndromeRecord("r_bad", "batch_1", 3, 20, ((0, 1),) * 10, 0)

    train = [SyndromeRecord(f"tr_{i}", "batch_A", 5, 4, ((0, 0),) * 4, 0) for i in range(4)]
    tune = [SyndromeRecord(f"tu_{i}", "batch_B", 5, 4, ((0, 0),) * 4, 0) for i in range(2)]
    test = [
        SyndromeRecord("te_0", "batch_C", 5, 4, ((1, 1), (0, 0), (1, 0), (0, 0)), 1),
        SyndromeRecord("te_1", "batch_C", 5, 4, ((0, 0),) * 4, 0),
        SyndromeRecord("te_2", "batch_D", 5, 4, ((1, 0), (1, 1), (0, 1), (0, 0)), 1),
    ]
    split_summary = validate_locked_group_split(train, tune, test, test_set_locked=True)
    assert split_summary["n_test"] == 3

    # Acquisition batch leakage must raise
    leaking_test = test + [SyndromeRecord("te_leak", "batch_A", 5, 4, ((0, 0),) * 4, 0)]
    with pytest.raises(ValueError, match="Acquisition-batch leakage"):
        validate_locked_group_split(train, tune, leaking_test, test_set_locked=True)

    # Paired evaluation with abstention preserved as failure
    base_preds = [
        DecoderPrediction("te_0", predicted_logical_observable=0, latency_us=120.0, peak_memory_kib=512.0),  # fail
        DecoderPrediction("te_1", predicted_logical_observable=0, latency_us=80.0, peak_memory_kib=512.0),   # pass
        DecoderPrediction("te_2", predicted_logical_observable=None, latency_us=950.0, abstained=True, peak_memory_kib=640.0),  # abstain -> fail
    ]
    cand_preds = [
        DecoderPrediction("te_0", predicted_logical_observable=1, latency_us=95.0, peak_memory_kib=480.0),   # rescued!
        DecoderPrediction("te_1", predicted_logical_observable=0, latency_us=70.0, peak_memory_kib=480.0),   # pass
        DecoderPrediction("te_2", predicted_logical_observable=1, latency_us=110.0, peak_memory_kib=500.0),  # rescued!
    ]
    report = evaluate_paired_decoders(
        test, base_preds, cand_preds,
        processor_model="x86_64-local-cpu",
        execution_mode="offline_replay",
        deadline_us=500.0,
    )
    assert report["baseline"]["n_failures"] == 2
    assert report["baseline"]["abstentions_or_errors"] == 1
    assert math.isclose(report["baseline"]["deadline_miss_fraction"], 1.0 / 3.0)
    assert report["candidate"]["n_failures"] == 0
    assert report["paired_disagreement_audit"]["rescued_by_candidate_ids"] == ["te_0", "te_2"]
    assert report["permissions"]["hardware_submission"] is False


def test_workflow_b_nmr_budget_and_signed_spectrum_guard():
    # Paper configuration: N_t=70 (71 time points), 10 initial states, 2 quadratures, 25 shots/circuit
    cfg = NMRResourceConfig(n_trotter_steps=70, n_initial_states=10, n_quadratures=2, shots_per_circuit=25)
    assert cfg.n_time_points == 71
    assert cfg.n_circuits == 1_420
    accepted = [22] * 1_420
    budget = compute_attempted_shot_budget(cfg, accepted_shots_by_circuit=accepted)
    assert budget["attempted_shots_total"] == 35_500
    assert budget["accepted_shots_total"] == 31_240
    assert math.isclose(budget["overall_acceptance_fraction"], 22.0 / 25.0)
    assert len(budget["acceptance_fraction_by_time_index"]) == 71

    ppm = np.linspace(-5.0, 15.0, 2001)
    ref_spec = np.exp(-0.5 * ((ppm - 3.20) / 0.25) ** 2) - 0.1 * np.exp(-0.5 * ((ppm - 8.0) / 0.4) ** 2)
    cand_spec = np.exp(-0.5 * ((ppm - 3.24) / 0.25) ** 2) - 0.1 * np.exp(-0.5 * ((ppm - 8.0) / 0.4) ** 2)

    err = evaluate_nmr_spectral_window_error(ppm, ref_spec, cand_spec, windows_ppm=[(1.0, 6.0)])
    assert err["integrated_squared_error"] > 0.0
    assert math.isclose(err["dominant_peak_shift_error_ppm"], 0.04, abs_tol=0.01)

    diag = NMRErrorDiagnostics(
        model_reduction_ise=0.012,
        trotter_discretization_ise=0.004,
        finite_time_truncation_ise=0.019,
        shot_sampling_ise=0.008,
        implementation_noise_ise=0.015,
    ).to_dict()
    assert diag["additive_assumption"] is False

    # Raw signed spectrum MUST be rejected before Wasserstein calculation
    with pytest.raises(ValueError, match="negative"):
        transform_spectrum_to_measure(ref_spec, mode="raw")
    # Explicit non-negative transformation succeeds cleanly
    p_pow = transform_spectrum_to_measure(ref_spec, mode="power_density")
    assert math.isclose(float(np.sum(p_pow)), 1.0, abs_tol=1e-12)
