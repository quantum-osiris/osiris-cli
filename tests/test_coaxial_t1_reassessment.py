"""Verification tests for Phase 2 Section 8: Coaxial T1 Reassessment."""

from __future__ import annotations

import pytest

pytest.importorskip("qiskit_ibm_runtime")

from phase2.coaxial_t1.coaxial_t1_reassessment import run_full_coaxial_t1_reassessment


def test_coaxial_t1_reassessment_preserves_negative_slope_and_rejects_retrofitting():
    res = run_full_coaxial_t1_reassessment()
    assert res["n_testable_snapshots"] >= 10
    # Negative T1/frequency slope is preserved and highly statistically significant globally
    assert res["negative_t1_freq_slope_preserved"]["pooled_slope_us_per_ghz"] < -50.0
    assert res["negative_t1_freq_slope_preserved"]["pooled_z"] < -4.0

    # At 3.5256 GHz, every device spans < 0.25 cycles and exhibits severe collinearity
    # with a smooth quadratic background (quadratic VIF > 200 on every single device,
    # even where straddling the 5.29 GHz extremum lowers linear VIF to ~3.65!)
    for d in res["per_device_results"]:
        assert d["cycles_spanned_at_3p526ghz"] < 0.25
        assert d["collinearity_vif_after_linear"] >= 1.0
        assert d["collinearity_vif_after_quadratic"] > 200.0
        assert d["mde_80pct_over_mean_t1"] > 1.0  # MDE exceeds 100% of mean T1!

    # Collective-mode mechanism cannot be distinguished from linear/smooth backgrounds -> INCONCLUSIVE
    assert res["mechanism_classification"] == "INCONCLUSIVE"
