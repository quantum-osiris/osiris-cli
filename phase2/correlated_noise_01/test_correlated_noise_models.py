"""
Unit & Identifiability Verification Tests for EXPERIMENT CORRELATED-NOISE-01
============================================================================
"""

from __future__ import annotations

import math

import pytest

from phase2.correlated_noise_01.correlated_noise_models import (
    ApparatusPulseCollisionModel,
    H0IndependentLocalModel,
    H1StaticSpectatorZZModel,
    H2CorrelatedMarkovianModel,
    H3NonMarkovianMemoryModel,
    ProbeCondition,
    build_isa_staggered_dd_bell_circuit,
    exact_pair_toggling_exposure,
    verify_heavy_hex_psd_covariance,
)
from phase2.correlated_noise_01.simulate_correlated_noise import (
    run_identifiability_and_null_study,
    select_disjoint_couplers_on_backend,
)


def test_toggling_exposure_exact_refocusing_at_half_slot():
    # Synchronous DD (offset = 0) gives alpha_uv = +1.0
    assert math.isclose(exact_pair_toggling_exposure(8.0e-6, 16, 0.0), 1.0, abs_tol=1e-12)
    # Half-slot stagger (slot = 8us / 16 = 500 ns -> half-slot = 250 ns) gives alpha_uv = 0.0!
    assert math.isclose(exact_pair_toggling_exposure(8.0e-6, 16, 250.0e-9), 0.0, abs_tol=1e-12)


def test_distinct_physical_signatures_across_h0_h1_h2_h3_and_apparatus():
    h0 = H0IndependentLocalModel()
    h1 = H1StaticSpectatorZZModel(f_zz_spectator_hz=28_000.0)
    h2 = H2CorrelatedMarkovianModel(rho_spatial=0.75)
    h3 = H3NonMarkovianMemoryModel(tau_c_s=6.0e-6, spatial_corr=0.65)
    app = ApparatusPulseCollisionModel(per_collision_error=0.01)

    sync_phi = ProbeCondition("sync_phi", "bell_phi_plus", 8.0e-6, 16, 0.0)
    stag_phi = ProbeCondition("stag_phi", "bell_phi_plus", 8.0e-6, 16, 250.0e-9)
    sync_psi = ProbeCondition("sync_psi", "bell_psi_plus", 8.0e-6, 16, 0.0)
    stag_psi = ProbeCondition("stag_psi", "bell_psi_plus", 8.0e-6, 16, 250.0e-9)

    # 1. H0 is completely invariant to stagger offset
    assert math.isclose(h0.predict(sync_phi), h0.predict(stag_phi), abs_tol=1e-12)

    # 2. H1 improves BOTH |Phi+> and |Psi+> under half-slot stagger and responds strongly to |+>_s
    assert h1.predict(stag_phi) > h1.predict(sync_phi) + 0.01
    assert h1.predict(stag_psi) > h1.predict(sync_psi) + 0.01
    spec_0 = ProbeCondition("spec0", "bell_phi_plus", 8.0e-6, 16, 0.0, spectator_superposition=False)
    spec_p = ProbeCondition("specp", "bell_phi_plus", 8.0e-6, 16, 0.0, spectator_superposition=True)
    assert h1.predict(spec_0) > h1.predict(spec_p) + 0.05

    # 3. H2 splits |Phi+> vs |Psi+>: staggering IMPROVES |Phi+> and DEGRADES |Psi+>!
    assert h2.predict(stag_phi) > h2.predict(sync_phi) + 0.02
    assert h2.predict(stag_psi) < h2.predict(sync_psi) - 0.02
    # And H2 single-qubit echo is completely invariant to DD pulse count (Markovian white noise)
    e1 = ProbeCondition("e1", "ramsey_echo_u", 8.0e-6, 1, 0.0)
    e32 = ProbeCondition("e32", "ramsey_echo_u", 8.0e-6, 32, 0.0)
    assert math.isclose(h2.predict(e1), h2.predict(e32), abs_tol=1e-12)

    # 4. H3 exhibits non-Markovian improvement with DD pulse density (N=0 -> N=1 -> N=32)
    r0_16 = ProbeCondition("r0_16", "ramsey_echo_u", 16.0e-6, 0, 0.0)
    e1_16 = ProbeCondition("e1_16", "ramsey_echo_u", 16.0e-6, 1, 0.0)
    e32_16 = ProbeCondition("e32_16", "ramsey_echo_u", 16.0e-6, 32, 0.0)
    assert h3.predict(e32_16) > h3.predict(e1_16) + 0.02
    assert h3.predict(e1_16) > h3.predict(r0_16) + 0.04

    # 5. Apparatus collision exhibits step recovery by 48 ns (>75% of total 250 ns recovery) and flat plateau
    d0 = app.predict(ProbeCondition("d0", "bell_phi_plus", 8.0e-6, 16, 0.0))
    d48 = app.predict(ProbeCondition("d48", "bell_phi_plus", 8.0e-6, 16, 48.0e-9))
    d250 = app.predict(ProbeCondition("d250", "bell_phi_plus", 8.0e-6, 16, 250.0e-9))
    step_frac = (d48 - d0) / (d250 - d0)
    assert math.isclose(step_frac, 1.0, abs_tol=1e-9)


def test_bic_identifiability_confusion_matrix_and_null_calibration():
    study = run_identifiability_and_null_study(
        n_trials_per_truth=5, shots_per_basis_per_block=4200, n_blocks=8, seed=20261116
    )
    assert study["shots_per_basis_total"] >= 33_500
    # Complex models must NOT win on H0 null data (high specificity)
    assert study["null_h0_specificity"] >= 0.80
    # Overall 5-way model recovery accuracy must be high (>= 85%)
    assert study["overall_identifiability_accuracy"] >= 0.85
    # Held-out coupler B prediction under H3 (noisy, zero leakage) beats H0 by > 3x lower RMSE
    assert study["held_out_transfer"]["parameter_leakage_from_truth"] is False
    assert study["held_out_transfer"]["evaluated_on_noisy_observations"] is True
    assert (
        study["held_out_transfer"]["held_out_rmse_h3"] * 3.0
        < study["held_out_transfer"]["held_out_rmse_h0"]
    )


def test_exact_piecewise_h3_no_aliasing_and_isa_dt_alignment_and_disjoint_couplers():
    pytest.importorskip("qiskit_ibm_runtime")  # CI does not install qiskit_ibm-runtime
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

    # 1. Sub-slot offsets 16, 32, 48, 80 ns must NOT alias to identical values in H3,
    #    and must increase strictly monotonically across all 7 offsets!
    h3 = H3NonMarkovianMemoryModel(tau_c_s=1.4e-6, spatial_corr=0.65)
    vals = [
        h3.predict(ProbeCondition(f"d{d}", "bell_phi_plus", 8.192e-6, 16, d * 1e-9))
        for d in (0, 16, 32, 48, 80, 160, 256)
    ]
    assert len(set(round(v, 10) for v in vals)) == 7
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))

    # When bath correlation time matches the CPMG passband (tau_c ~ 0.25 us), stagger gain exceeds +1.2 pp
    h3_passband = H3NonMarkovianMemoryModel(
        sigma_ou_u_rad_s=220_000.0, sigma_ou_v_rad_s=220_000.0, tau_c_s=0.25e-6, spatial_corr=0.75
    )
    f_sync = h3_passband.predict(ProbeCondition("s0", "bell_phi_plus", 8.192e-6, 16, 0.0))
    f_stag = h3_passband.predict(ProbeCondition("s256", "bell_phi_plus", 8.192e-6, 16, 256e-9))
    assert f_stag > f_sync + 0.012

    # 2. ISA circuit builder enforces 4.0 ns dt alignment and rejects 250 ns (62.5 dt)
    with pytest.raises(ValueError, match="exact integer multiple of dt"):
        build_isa_staggered_dd_bell_circuit(8.0e-6, 16, 250.0e-9)
    qc_isa, meta = build_isa_staggered_dd_bell_circuit(8.192e-6, 16, 256.0e-9, pauli_basis="XX")
    assert meta["total_dt"] == 2048
    assert meta["offset_dt"] == 64
    assert qc_isa.count_ops()["x"] == 32

    # 3. Heavy-hex PSD check & >=5-hop disjoint coupler selector on FakeMarrakesh
    assert verify_heavy_hex_psd_covariance(0.60, 2.0) is True
    assert verify_heavy_hex_psd_covariance(0.95, 1.0) is False
    couplers = select_disjoint_couplers_on_backend(FakeMarrakesh(), min_hop_distance=5, max_couplers=10)
    assert len(couplers) == 10
