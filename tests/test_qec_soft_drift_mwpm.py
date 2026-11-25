"""Unit tests for Card C (`OSIRIS-QEC-SOFTMWPM`): Soft-Drift & Streak-Aware Blossom MWPM Decoder."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from qec_card_c import qec_crossing_gate as gate  # noqa: E402
from qec_card_c import qec_multidistance_scaling as ms  # noqa: E402
from qec_card_c import qec_soft_drift_mwpm as sdm  # noqa: E402


def test_compute_effective_time_weight_monotone_and_clamped() -> None:
    """Verify Stage B time-like weight reduction is monotone non-increasing and clamped at 0.05."""
    w0 = 3.85
    assert sdm.compute_effective_time_weight(w0, 0.8) == w0
    assert sdm.compute_effective_time_weight(w0, 1.0) == w0

    etas = [1.0, 1.25, 1.5, 2.0, 2.5, 2.9, 4.0, 10.0, 500.0]
    weights = [sdm.compute_effective_time_weight(w0, e, min_weight=0.05) for e in etas]
    for i in range(len(weights) - 1):
        assert weights[i] >= weights[i + 1]
    assert weights[0] == w0
    assert weights[-1] == 0.05


def test_estimate_empirical_streak_ratio_bernoulli_vs_correlated() -> None:
    """Verify empirical consecutive-round streak estimator separates Bernoulli from streaks."""
    rng = np.random.default_rng(20261124)
    bern = (rng.random((20000, 6)) < 0.06).astype(np.uint8)
    res_bern = sdm.estimate_empirical_streak_ratio(bern)
    assert 0.88 <= res_bern["eta_streak"] <= 1.15

    # Inject paired consecutive streaks (D_{t,a}=1, D_{t+1,a}=1)
    streaked = bern.copy()
    burst = rng.random((20000, 5)) < 0.04
    for t in range(5):
        mask = burst[:, t]
        streaked[mask, t] = 1
        streaked[mask, t + 1] = 1
    res_streak = sdm.estimate_empirical_streak_ratio(streaked)
    assert res_streak["eta_streak"] > 2.0


def test_soft_drift_blossom_decoder_stage_a_and_b_edge_weights() -> None:
    """Verify Stage A and Stage B reduce targeted hot-ancilla time-like weights while preserving b(u)."""
    prof_base = ms.build_circuit_depolarizing_profile(0.0022, idle_multiplier=0.45)
    prof_cal = gate.TargetNoiseProfile(
        **{**prof_base.__dict__, "p_readout": sdm.P_RO_CAL_PUBLISHED, "p_reset": 0.0005}
    )
    dem = ms.build_rotated_surface_dem(3, 3, prof_cal, basis="X", include_topology_overhead=False)
    dec_stale = ms.PolynomialMWPMDecoder(dem)
    dec_prior = sdm.SoftDriftBlossomDecoder(
        dem, n_anc=4, delta_w_prior_by_anc={1: 0.65}, eta_streak_by_anc={1: 1.0}
    )
    dec_soft = sdm.SoftDriftBlossomDecoder(
        dem, n_anc=4, delta_w_prior_by_anc={1: 0.65}, eta_streak_by_anc={1: 2.80}
    )

    for t in range(3):
        w_stale_hot = float(dec_stale.dist_matrix[t * 4 + 1, (t + 1) * 4 + 1])
        w_prior_hot = dec_prior.time_edge_weight(t, 1)
        w_soft_hot = dec_soft.time_edge_weight(t, 1)
        assert w_stale_hot > w_prior_hot > w_soft_hot >= 0.05

        # Cold ancilla 0 time-like weight and all boundary distances remain invariant
        assert (
            abs(dec_soft.time_edge_weight(t, 0) - float(dec_stale.dist_matrix[t * 4, (t + 1) * 4]))
            < 1e-12
        )

    b_node = dec_stale.boundary_node
    np.testing.assert_allclose(
        dec_soft.dist_matrix[:b_node, b_node], dec_stale.dist_matrix[:b_node, b_node], atol=1e-12
    )


def test_soft_drift_blossom_decoder_exact_blossom_equivalence() -> None:
    """Verify 100% agreement between component-DP Blossom and pure NetworkX Edmonds Blossom."""
    controls = sdm.audit_soft_drift_decoder_controls(seed=sdm.DEFAULT_SEED)
    assert controls["exact_100pct_blossom_match"] is True
    assert controls["component_dp_vs_pure_edmonds_blossom_mismatches"] == 0
    assert controls["monotone_streak_weight_reduction_verified"] is True


def test_soft_drift_blossom_decoder_resolves_sinkhole_triangles() -> None:
    """Verify Stage B streak reweighting resolves interior-to-boundary sinkhole triangles on d=5,7."""
    for d, expected_total in ((5, 10), (7, 28)):
        _, spec, dec_stale, dec_prior, dec_soft = (
            sdm.build_benchmark1_heterogeneous_dems_and_decoders("rotated_surface", d, 0.032)
        )
        n_anc = int(spec["n_anc"])
        b_node = dec_stale.boundary_node
        stale_fails = 0
        prior_fails = 0
        soft_fails = 0
        tot = 0
        for a_hot, a_int in spec["sinkhole_pairs"]:
            for t in range(d):
                u = t * n_anc + a_hot
                v = (t + 1) * n_anc + a_hot
                w = t * n_anc + a_int
                dm = (1 << u) | (1 << v) | (1 << w)
                true_of = int(dec_stale.obs_matrix[w, b_node])
                tot += 1
                if dec_stale.decode_mask(dm) != true_of:
                    stale_fails += 1
                if dec_prior.decode_mask(dm) != true_of:
                    prior_fails += 1
                if dec_soft.decode_mask(dm) != true_of:
                    soft_fails += 1
        assert tot == expected_total
        assert stale_fails >= tot - 1
        assert 0 < prior_fails < stale_fails
        assert soft_fails == 0


def test_surface_code_lambda_3_to_5_recovery_under_heterogeneous_drift() -> None:
    """Verify rotated surface code Lambda_{3->5} > 2.0 recovery across all p_ro_hot regimes."""
    rep = json.loads(sdm.DEFAULT_OUTPUT_PATH.read_text(encoding="utf-8"))
    b1 = rep["benchmark_1_heterogeneous_heavy_hex_drift_recovery"]
    for p_hot in sdm.P_RO_HOT_SWEEP:
        cell = b1["cells"][f"rotated_surface__p_ro_hot_{p_hot:.3f}"]
        l35_stale = float(cell["decoders"]["STALE_UNIFORM_MWPM"]["lambda_3_to_5"])
        l35_prior = float(cell["decoders"]["DRIFT_PRIOR_MWPM"]["lambda_3_to_5"])
        l35_soft = float(cell["decoders"]["SOFT_DRIFT_STREAK_MWPM"]["lambda_3_to_5"])
        l57_soft = float(cell["decoders"]["SOFT_DRIFT_STREAK_MWPM"]["lambda_5_to_7"])
        assert l35_stale < 1.35
        assert l35_stale < l35_prior < l35_soft
        assert l35_soft > 2.0
        assert l57_soft > 2.0


def test_repetition_code_lambda_3_to_5_recovery_under_heterogeneous_drift() -> None:
    """Verify 1D repetition code Lambda_{3->5} recovery across all p_ro_hot regimes."""
    rep = json.loads(sdm.DEFAULT_OUTPUT_PATH.read_text(encoding="utf-8"))
    b1 = rep["benchmark_1_heterogeneous_heavy_hex_drift_recovery"]
    for p_hot in sdm.P_RO_HOT_SWEEP:
        cell = b1["cells"][f"repetition__p_ro_hot_{p_hot:.3f}"]
        l35_stale = float(cell["decoders"]["STALE_UNIFORM_MWPM"]["lambda_3_to_5"])
        l35_prior = float(cell["decoders"]["DRIFT_PRIOR_MWPM"]["lambda_3_to_5"])
        l35_soft = float(cell["decoders"]["SOFT_DRIFT_STREAK_MWPM"]["lambda_3_to_5"])
        assert l35_stale < l35_prior < l35_soft
        assert l35_soft > 3.0


def test_live_fez_ro_cal_asymmetric_priors() -> None:
    """Verify RO_CAL asymmetric readout calibration on ibm_fez dc35d4inj2vs73dhmsog."""
    rep = json.loads(sdm.DEFAULT_OUTPUT_PATH.read_text(encoding="utf-8"))
    ro_priors = rep["benchmark_2_live_fez_dc35d4inj2vs73dhmsog_offline_rescoring"][
        "ro_cal_asymmetric_data_priors"
    ]
    assert set(ro_priors.keys()) == {"d0_q58", "d1_q70", "d2_q78"}
    for q_info in ro_priors.values():
        assert float(q_info["p_meas0_given_prep1"]) > float(q_info["p_meas1_given_prep0"])
        assert float(q_info["asymmetry_ratio_p01_over_p10"]) > 2.0


def test_live_fez_empirical_streak_ratios_q71_q69() -> None:
    """Verify consecutive time-like detector streak excess (2.34x-2.90x) on q71 and q69."""
    rep = json.loads(sdm.DEFAULT_OUTPUT_PATH.read_text(encoding="utf-8"))
    priors = rep["benchmark_2_live_fez_dc35d4inj2vs73dhmsog_offline_rescoring"][
        "empirical_detector_and_streak_priors"
    ]
    for r_key in ("R3", "R5"):
        eta_q71 = float(priors[r_key]["ancilla_a0_q71"]["eta_streak_detector"])
        eta_q69 = float(priors[r_key]["ancilla_a1_q69"]["eta_streak_detector"])
        assert 2.30 <= eta_q71 <= 2.95
        assert 2.30 <= eta_q69 <= 2.95


def test_live_fez_offline_rescoring_exact_baseline_and_monotone_gain() -> None:
    """Verify offline re-scoring of dc35d4inj2vs73dhmsog matches baseline and cuts slope by >70%."""
    rep = json.loads(sdm.DEFAULT_OUTPUT_PATH.read_text(encoding="utf-8"))
    b2 = rep["benchmark_2_live_fez_dc35d4inj2vs73dhmsog_offline_rescoring"]
    assert b2["provider_job_id"] == "dc35d4inj2vs73dhmsog"
    assert b2["new_qpu_seconds"] == 0
    assert b2["total_job_shots"] == 49152
    assert b2["arm4a_shots_rescored"] == 9216
    assert b2["baseline_arm4a_exact_reproduction_verified"] is True

    rows = b2["hardware_rounds_summary"]
    assert [r["rounds"] for r in rows] == [1, 3, 5]
    assert rows[0]["uniform_static_mwpm_p_fail"] == 0.01237
    assert rows[1]["uniform_static_mwpm_p_fail"] == 0.054036
    assert rows[2]["uniform_static_mwpm_p_fail"] == 0.08724

    for r_row in rows:
        u = float(r_row["uniform_static_mwpm_p_fail"])
        p = float(r_row["drift_prior_mwpm_p_fail"])
        s = float(r_row["soft_drift_streak_mwpm_p_fail"])
        assert u >= p > s

    assert float(b2["headline_improvements"]["slope_reduction_soft_drift_vs_uniform_pct"]) > 70.0
    assert len(b2["extended_rounds_grid_1_2_3_5_7_9"]) == 6


def test_live_fez_paired_block_wins() -> None:
    """Verify paired block-level wins (3/3 blocks) on R=3,5 and R_cross=1 vs Arm 2 no-corr."""
    rep = json.loads(sdm.DEFAULT_OUTPUT_PATH.read_text(encoding="utf-8"))
    tests = rep["benchmark_2_live_fez_dc35d4inj2vs73dhmsog_offline_rescoring"][
        "paired_block_comparisons"
    ]
    for r in (3, 5):
        assert tests[f"R{r}_uniform_minus_drift_prior"]["block_wins"] == 3
        assert tests[f"R{r}_uniform_minus_soft_drift_streak"]["block_wins"] == 3
        assert tests[f"R{r}_drift_prior_minus_soft_drift_streak"]["block_wins"] == 3
    for r in (1, 3, 5):
        assert tests[f"R{r}_arm2_nocorr_minus_soft_drift_streak"]["block_wins"] == 3


def test_report_jcs_canonical_hash_and_schema_integrity() -> None:
    """Verify RFC 8785 JCS SHA-256 integrity, zero wall-clock timestamps, and PASS verdict."""
    rep = json.loads(sdm.DEFAULT_OUTPUT_PATH.read_text(encoding="utf-8"))
    assert rep["schema"] == "osiris.qec_card_c.soft_drift_mwpm_report/1"
    assert rep["canonicalization"] == "RFC8785-JCS"
    assert rep["new_qpu_seconds"] == 0
    assert (
        rep["scientific_conclusions"]["verdict"]
        == "PASS_SOFT_DRIFT_STREAK_MWPM_LAMBDA_RECOVERY_AND_FEZ_RESCORING_VERIFIED"
    )

    recorded_sha = rep["jcs_payload_sha256"]
    body = {k: v for k, v in rep.items() if k != "jcs_payload_sha256"}
    assert ms.jcs_sha256(body) == recorded_sha


def test_deterministic_reexecution_small_sample() -> None:
    """Verify deterministic Benchmark 1 re-execution under fixed PRNG seed."""
    r1 = sdm.run_benchmark1_heterogeneous_drift_study(n_shots=128, n_boot=16, seed=20261124)
    r2 = sdm.run_benchmark1_heterogeneous_drift_study(n_shots=128, n_boot=16, seed=20261124)
    assert ms.jcs_sha256(r1) == ms.jcs_sha256(r2)
