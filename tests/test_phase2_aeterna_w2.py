"""Unit tests for Phase 2 Section 4: Aeterna controls repair & W2 27-qubit reanalysis."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("qiskit")

from phase2.aeterna_w2.aeterna_controls_repaired import (
    HERON_R2_DT_S,
    FloquetScheduleSpec,
    audit_historical_aeterna_bug,
    build_repaired_aeterna_stroboscopic_circuit,
    compute_zeno_cycle_count,
    verify_floquet_survives_transpile,
)
from phase2.aeterna_w2.w2_circuit_reanalysis import (
    run_w2_27q_erratum_analysis,
    verify_phase_cancellation_small_width,
)


def test_historical_aeterna_1ghz_1ns_aliasing_and_transpile_elimination():
    bug = audit_historical_aeterna_bug()
    assert bug["max_abs_floquet_angle_rad"] < 3e-15
    assert bug["pre_transpile_rz_count"] == 100
    assert bug["post_transpile_opt3_gate_count"] == 0
    assert bug["floquet_survived_transpilation"] is False
    assert bug["zeno_v2_0_cycles"] == 1
    assert bug["zeno_v2_1_cycles_truncated"] == 1


def test_repaired_aeterna_nyquist_dt_alignment_and_transpile_survival():
    # 1. Reject 1 GHz at 4 ns or 8 ns sampling (Nyquist violation)
    bad_nyq = FloquetScheduleSpec(
        drive_frequency_hz=1.0e9,
        drive_amplitude_rad=0.5,
        n_steps=10,
        sample_interval_s=8.0e-9,
        dt_s=HERON_R2_DT_S,
    )
    with pytest.raises(ValueError, match="Nyquist aliasing violation"):
        bad_nyq.validate()

    # 2. Reject non-integer dt sample interval (e.g. 5.0 ns on 4.0 ns dt grid)
    bad_dt = FloquetScheduleSpec(
        drive_frequency_hz=20.0e6,
        drive_amplitude_rad=0.5,
        n_steps=10,
        sample_interval_s=5.0e-9,
        dt_s=HERON_R2_DT_S,
    )
    with pytest.raises(ValueError, match="integer multiple of hardware dt"):
        bad_dt.validate()

    # 3. Reject 130 qubits on 120-qubit backend
    with pytest.raises(ValueError, match="exceeding backend limit"):
        build_repaired_aeterna_stroboscopic_circuit(
            n_left=60, n_right=60, n_anc=10, backend_qubit_limit=120
        )

    # 4. Zeno cycle rounding distinguishes 1.0 MHz over 4 us (=4) vs 1.25 MHz over 4 us (=5)
    assert compute_zeno_cycle_count(4.0e-6, 1.0e6) == 4
    assert compute_zeno_cycle_count(4.0e-6, 1.25e6) == 5

    # 5. Valid stroboscopic Floquet intervention survives Qiskit optimization_level=3!
    qc, meta = build_repaired_aeterna_stroboscopic_circuit(
        n_left=4, n_right=4, n_anc=2, backend_qubit_limit=156
    )
    assert meta["floquet"]["max_abs_angle_rad"] > 0.30
    surv = verify_floquet_survives_transpile(qc, optimization_level=3)
    assert surv["floquet_modulation_preserved"] is True
    assert surv["delay_count"] >= 40


def test_w2_27q_phase_cancellation_uniform_baseline_and_ot_recomputation():
    # Exact phase cancellation verified against Qiskit Statevector at 8 and 10 qubits
    assert verify_phase_cancellation_small_width(8, 2.118) < 1e-14
    assert verify_phase_cancellation_small_width(10, 2.618) < 1e-14

    res = run_w2_27q_erratum_analysis(shots_exact_lp=256)
    assert math.isclose(res["true_uniform_reference_prob"], 2.0 ** -27, rel_tol=1e-12)
    assert math.isclose(res["paper_112x_reproduced_from_unique_support_bug"], 112.16, abs_tol=0.1)
    assert math.isclose(res["analytic_defective_eq7_w2_on_identical_uniform"], 0.509175, abs_tol=1e-5)

    rows = res["theta_sweep_reanalysis"]
    # At theta=2.118, true ideal mode probability is ~12.91% (> observed 3.98%), NOT 6e-4!
    assert math.isclose(rows[0]["ideal_mode_prob_27q"], 0.129135, abs_tol=1e-4)
    assert math.isclose(rows[1]["ideal_mode_prob_27q"], 0.000423, abs_tol=1e-5)
    assert rows[2]["ideal_mode_prob_27q"] < 1e-7

    # Self-distance: exact LP gives 0.0 identically, whereas defective Eq. 7 gives 0.18..0.51!
    for r in rows:
        assert r["exact_lp_w2_self"] < 1e-11
        assert r["defective_eq7_w2_self"] > 0.14
    # Even with ZERO hardware noise, two finite-shot draws from theta=3.118 exhibit
    # large empirical W2 (~0.22-0.25) purely from 27D multinomial sampling bias,
    # monotonic across theta, while defective Eq. 7 inflates W2 past 0.50 (>2x true OT)!
    assert rows[2]["exact_lp_w2_noiseless_inter_rep"] > 0.20
    assert rows[2]["exact_lp_w2_noiseless_inter_rep"] > rows[0]["exact_lp_w2_noiseless_inter_rep"]
    assert rows[2]["defective_eq7_w2_noiseless_inter_rep"] > 1.8 * rows[2]["exact_lp_w2_noiseless_inter_rep"]
