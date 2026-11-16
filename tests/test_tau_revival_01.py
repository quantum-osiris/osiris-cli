"""Verification tests for Phase 2 Section 7: EXPERIMENT TAU-REVIVAL-01."""

from __future__ import annotations

import math

from phase2.tau_revival_01.tau_revival_models import (
    TAU_0_UNIVERSAL_US,
    ConventionalBeatModel,
    audit_archived_torino_tau_sweeps,
    compute_aliased_frequency_khz,
    simulate_detuning_injection_falsification,
)


def test_beat_period_and_aliasing_math():
    mdl = ConventionalBeatModel(delta_f_khz=21.692)
    assert math.isclose(mdl.beat_period_us(0.0), 46.10, abs_tol=0.05)
    # Shifting detuning by +10 kHz moves period from ~46.1 us to 1000 / 31.692 = 31.55 us!
    assert math.isclose(mdl.beat_period_us(10.0), 1000.0 / 31.692, abs_tol=0.02)
    # Aliasing check: 83.333 kHz sampled every 10 us (f_s = 100 kHz) aliases to 16.667 kHz (60 us period!)
    aliased = compute_aliased_frequency_khz(83.3333, dt_step_us=10.0)
    assert math.isclose(aliased, 16.6667, abs_tol=1e-3)


def test_detuning_injection_slope_and_echo_elimination():
    sim = simulate_detuning_injection_falsification()
    # Conventional detuning shifts frequency with unit slope +1.0 (+/- 0.05)
    assert math.isclose(sim["fitted_frequency_slope_df_obs_df_inj"], 1.0, abs_tol=0.05)
    assert math.isclose(sim["fitted_intercept_delta_f0_khz"], 21.7, abs_tol=0.5)
    # Under Hahn echo, 46 us is strictly lower than 36 us (pure T2 decay, no revival!)
    assert sim["hahn_echo_46us_minus_36us"] < 0.0

    # Signed conjugate-quadrature fit also recovers +1.0 slope when Delta_f0 < 0 or near zero (no V-folding!)
    sim_neg = simulate_detuning_injection_falsification(true_delta_f_khz=-6.5, seed=99)
    assert math.isclose(sim_neg["fitted_frequency_slope_df_obs_df_inj"], 1.0, abs_tol=0.05)
    assert math.isclose(sim_neg["fitted_intercept_delta_f0_khz"], -6.5, abs_tol=0.6)


def test_archived_torino_sweeps_falsify_pair_invariance():
    arc = audit_archived_torino_tau_sweeps()
    assert math.isclose(TAU_0_UNIVERSAL_US, 46.9787, abs_tol=1e-3)
    if arc["available"]:
        assert arc["fez_prospective_3basis_tomography_job_dbiobimne4hc73bf2am0"]["separable_below_0_50"] is True
        assert len(arc["archived_json_records"]) == 3
        assert arc["archived_json_records"][0]["tau_0_observed"] == 8
