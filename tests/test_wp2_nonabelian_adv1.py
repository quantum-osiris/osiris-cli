"""
Comprehensive Unit & Artifact Verification Suite for Quantum Directions 4-6 & 7.2:
  1. Direction 4 (`WP2`) & 7.2: `experiments/wp2_debiased_sinkhorn/debiased_ot.py`
  2. Direction 5: `qldpc_nonabelian/nonabelian_2bga.py`
  3. Direction 6: `adv1/looped_peaked.py` & `adv1/pregate.py`
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEBIASED_OT_PATH = REPO_ROOT / "experiments" / "wp2_debiased_sinkhorn" / "debiased_ot.py"
WP2_REPORT_PATH = (
    REPO_ROOT
    / "experiments"
    / "wp2_debiased_sinkhorn"
    / "artifacts"
    / "wp2_debiased_ot_report.json"
)
QLDPC_REPORT_PATH = (
    REPO_ROOT / "qldpc_nonabelian" / "artifacts" / "nonabelian_2bga_report.json"
)
ADV1_REPORT_PATH = REPO_ROOT / "adv1" / "artifacts" / "adv1_pregate_report.json"


def _load_debiased_ot():
    spec = importlib.util.spec_from_file_location("wp2_debiased_ot", DEBIASED_OT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wp2_debiased_ot"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. Direction 4 (WP2) & 7.2 — Split-Sample-Debiased Sinkhorn & LP OT Tests
# ---------------------------------------------------------------------------


def test_normalized_squared_hamming_cost_arbitrary_width_96_and_156() -> None:
    dot = _load_debiased_ot()
    for q_bits in (27, 50, 96, 156):
        a0 = 0
        a_all = (1 << q_bits) - 1
        a_half = sum(1 << i for i in range(0, q_bits, 2))  # ceil(q_bits / 2) ones
        xs = [a0, a_all, a_half]
        cmat = dot.normalized_squared_hamming_cost(xs, xs, n_qubits=q_bits)
        assert cmat.shape == (3, 3)
        assert np.allclose(np.diag(cmat), 0.0, atol=1e-15)
        assert np.allclose(cmat, cmat.T, atol=1e-15)
        assert cmat[0, 1] == pytest.approx(1.0, abs=1e-15)
        for i in range(3):
            for j in range(3):
                dh_exact = dot.hamming_distance_int(xs[i], xs[j])
                assert cmat[i, j] == pytest.approx((dh_exact / q_bits) ** 2, abs=1e-15)


def test_entropic_sinkhorn_converges_and_approaches_exact_lp() -> None:
    dot = _load_debiased_ot()
    keys_x = ["00000000", "00110000", "01110000"]
    keys_y = ["00010000", "00111000", "01111000"]
    p = np.array([0.45, 0.35, 0.20])
    q = np.array([0.35, 0.40, 0.25])
    cmat = dot.normalized_squared_hamming_cost(keys_x, keys_y, n_qubits=8)

    lp_w2_sq = dot.exact_kantorovich_w2_sq(p, q, cmat)
    sk_default = dot.entropic_sinkhorn_ot(p, q, cmat, epsilon=0.05)
    sk_sharp = dot.entropic_sinkhorn_ot(p, q, cmat, epsilon=0.005, max_iter=3000, tol=1e-7)

    assert sk_default.converged
    assert sk_default.max_marginal_error < 1e-9
    assert sk_sharp.converged
    assert sk_sharp.transport_cost == pytest.approx(lp_w2_sq, abs=2e-3)


def test_split_shot_histogram_determinism_and_conservation() -> None:
    dot = _load_debiased_ot()
    hist = {(1 << 90) | 7: 120, (1 << 75) | 3: 85, 42: 51}  # total = 256 (even)
    x1, x2 = dot.split_shot_histogram(hist, seed=19)
    x1_b, x2_b = dot.split_shot_histogram(hist, seed=19)
    assert x1 == x1_b and x2 == x2_b
    assert sum(x1.values()) == 128
    assert sum(x2.values()) == 128
    for k, c in hist.items():
        assert x1.get(k, 0) + x2.get(k, 0) == c


def test_split_sample_debiasing_eliminates_96q_uniform_bias_and_detects_drift() -> None:
    dot = _load_debiased_ot()
    null_sk = []
    null_lp_matched = []
    raw_lp = []
    eq7 = []
    for r in range(8):
        hx = dot.sample_uniform_hypercube_histogram(96, shots=256, eps_relax=0.0, seed=500 + 2 * r)
        hy = dot.sample_uniform_hypercube_histogram(96, shots=256, eps_relax=0.0, seed=501 + 2 * r)
        sk = dot.split_sample_sinkhorn_divergence(hx, hy, n_qubits=96, epsilon=0.05, seed=r)
        lp = dot.split_sample_wasserstein_lp(hx, hy, n_qubits=96, seed=r)
        null_sk.append(sk.debiased_sinkhorn)
        null_lp_matched.append(lp.debiased_w2_sq_matched)
        raw_lp.append(lp.raw_w2_sq)
        eq7.append(lp.defective_eq7_cost)

    # Defective Eq. 7 (~0.2526) and raw LP (~0.133) are heavily biased positive, while debiased ~ 0
    assert float(np.mean(eq7)) > 0.24
    assert float(np.mean(raw_lp)) > 0.12
    assert abs(float(np.mean(null_sk))) < 0.002
    assert abs(float(np.mean(null_lp_matched))) < 0.003

    # Real physical drift (eps_depol = 0.05 on 50Q transport circuit) yields clear positive signal
    hx_clean = dot.sample_transport_circuit_histogram(50, 2.118, shots=256, eps_depol=0.0, seed=91)
    hy_drift = dot.sample_transport_circuit_histogram(50, 2.118, shots=256, eps_depol=0.05, seed=92)
    sk_d = dot.split_sample_sinkhorn_divergence(hx_clean, hy_drift, n_qubits=50, epsilon=0.05, seed=9)
    lp_d = dot.split_sample_wasserstein_lp(hx_clean, hy_drift, n_qubits=50, seed=9)
    assert sk_d.debiased_sinkhorn > 0.015
    assert lp_d.debiased_w2_sq > 0.015


def test_wp2_artifact_report_and_hardware_attestation() -> None:
    assert WP2_REPORT_PATH.is_file(), f"Missing artifact {WP2_REPORT_PATH}"
    rep = json.loads(WP2_REPORT_PATH.read_text(encoding="utf-8"))
    assert rep["schema"] == "osiris.wp2.debiased_ot_report/1"

    pa = rep["part_a_simulated_high_entropy_benchmark"]
    assert pa["evidence_class"] == "SIMULATED"
    assert pa["provider_job_id"] is None
    assert pa["null_unbiasedness_gate"]["passed_under_0_002"] is True
    assert pa["null_unbiasedness_gate"]["max_abs_mean_debiased_sinkhorn_eps0"] < 0.002
    assert pa["drift_detection_gate_eps_0_05"]["passed_over_10_sigma"] is True
    assert pa["drift_detection_gate_eps_0_05"]["min_z_score_across_all_widths_and_thetas"] > 10.0

    pb = rep["part_b_hardware_attested_heron_r2_rescoring"]
    m_res = pb["marrakesh_dd_rescoring"]
    assert m_res["evidence_class"] == "HARDWARE_ATTESTED"
    assert m_res["provider_job_id"] == "dau0q3qhcrkc73durtgg"
    m_disk_sha = hashlib.sha256((REPO_ROOT / m_res["raw_counts_relpath"]).read_bytes()).hexdigest()
    assert m_res["raw_counts_sha256"] == m_disk_sha

    f_res = pb["fez_qec001_rescoring"]
    assert f_res["evidence_class"] == "HARDWARE_ATTESTED"
    assert f_res["provider_job_id"] == "dbtl2vb2kf8s73c9a0d0"
    f_disk_sha = hashlib.sha256((REPO_ROOT / f_res["raw_counts_relpath"]).read_bytes()).hexdigest()
    assert f_res["raw_counts_sha256"] == f_disk_sha


# ---------------------------------------------------------------------------
# 2. Direction 5 — Non-Abelian 2BGA Odd-k qLDPC Discovery Tests
# ---------------------------------------------------------------------------


def test_nonabelian_group_tables_axioms_and_non_commutativity() -> None:
    from qldpc_nonabelian import (
        alternating_group_a4,
        dihedral_group,
        frobenius_metacyclic_group,
        quaternion_group_q8,
        symmetric_group_s4,
    )

    groups = [
        (quaternion_group_q8(), 8),
        (alternating_group_a4(), 12),
        (symmetric_group_s4(), 24),
        (dihedral_group(7), 14),
        (dihedral_group(25), 50),
        (frobenius_metacyclic_group(7, 3), 21),
        (frobenius_metacyclic_group(13, 3), 39),
        (frobenius_metacyclic_group(19, 3), 57),
        (frobenius_metacyclic_group(11, 5), 55),
        (frobenius_metacyclic_group(31, 3), 93),
    ]
    for grp, expected_order in groups:
        assert grp.order == expected_order
        assert grp.verify_axioms() is True
        assert grp.is_abelian() is False


def test_2bga_universal_css_orthogonality_and_coset_odd_k() -> None:
    from qldpc_nonabelian import (
        build_2bga_css,
        build_coset_2bga_css,
        dihedral_group,
        frobenius_metacyclic_group,
        gf2_rank,
        left_regular_matrix,
        right_regular_matrix,
    )

    c7c3 = frobenius_metacyclic_group(7, 3)
    a_sub, b_sub = [0, 3, 11], [0, 5, 17]
    la = left_regular_matrix(c7c3, a_sub).astype(np.int32)
    rb = right_regular_matrix(c7c3, b_sub).astype(np.int32)
    assert np.array_equal(la @ rb, rb @ la)  # commutes over Z identically!

    hx, hz = build_2bga_css(c7c3, a_sub, b_sub)
    assert hx.shape == (21, 42) and hz.shape == (21, 42)
    assert np.all((hx.astype(np.int32) @ hz.astype(np.int32).T) % 2 == 0)
    assert gf2_rank(hx) == gf2_rank(hz)

    # Coset 2BGA on D_7 / Z_2 gives odd k = 3 directly
    d7 = dihedral_group(7)
    hx_c, hz_c = build_coset_2bga_css(d7, [0, 7], [0, 1, 3], [0])
    assert np.all((hx_c.astype(np.int32) @ hz_c.astype(np.int32).T) % 2 == 0)
    k_c = hx_c.shape[1] - gf2_rank(hx_c) - gf2_rank(hz_c)
    assert k_c == 3


def test_odd_k_d9_36_3_6_exact_distance_and_exhaustive_bposd_recovery() -> None:
    from qldpc_nonabelian import (
        build_odd_k_pinned_2bga,
        compute_css_distance,
        css_logical_bases,
        decode_bp_osd_cs,
        dihedral_group,
        gf2_rank,
    )

    d9 = dihedral_group(9)
    pinned = build_odd_k_pinned_2bga(
        d9, [0, 4, 8], [0, 9, 11], n_pinned_logicals=1, max_pin_weight=6, seed=20261123
    )
    assert pinned is not None
    hx, hz, pin_wts = pinned
    assert pin_wts == [6]
    assert int(hx.sum(axis=1).max()) == 6 and int(hz.sum(axis=1).max()) == 6
    assert np.all((hx.astype(np.int32) @ hz.astype(np.int32).T) % 2 == 0)

    k = hx.shape[1] - gf2_rank(hx) - gf2_rank(hz)
    assert k == 3
    dist = compute_css_distance(hx, hz, max_exact_n=48)
    assert dist["distance_type"] == "exact_d"
    assert dist["d"] == 6

    # Since exact d = 6, BP+OSD-CS must correct 100% of weight-1 AND all 630 weight-2 errors!
    lx, lz = css_logical_bases(hx, hz)
    n = hx.shape[1]
    for q in range(n):
        err = np.zeros(n, dtype=np.uint8)
        err[q] = 1
        rx = err ^ decode_bp_osd_cs(hz, (hz @ err) % 2, p_err=0.002)
        rz = err ^ decode_bp_osd_cs(hx, (hx @ err) % 2, p_err=0.002)
        assert np.all((hz @ rx) % 2 == 0) and np.all((lx @ rx) % 2 == 0)
        assert np.all((hx @ rz) % 2 == 0) and np.all((lz @ rz) % 2 == 0)

    w2_recovered = 0
    w2_total = 0
    for i in range(n):
        for j in range(i + 1, n):
            w2_total += 1
            err = np.zeros(n, dtype=np.uint8)
            err[i] = 1
            err[j] = 1
            rx = err ^ decode_bp_osd_cs(hz, (hz @ err) % 2, p_err=0.002)
            if np.all((hz @ rx) % 2 == 0) and np.all((lx @ rx) % 2 == 0):
                w2_recovered += 1
    assert w2_recovered == w2_total == 630


def test_nonabelian_2bga_report_artifact() -> None:
    assert QLDPC_REPORT_PATH.is_file(), f"Missing artifact {QLDPC_REPORT_PATH}"
    rep = json.loads(QLDPC_REPORT_PATH.read_text(encoding="utf-8"))
    assert rep["evidence_class"] == "SIMULATED"
    assert rep["provider_job_id"] is None
    assert rep["odd_k_d_ge_4_discoveries_count"] >= 8
    for cand in rep["odd_k_candidates"]:
        assert cand["n"] <= 200
        assert cand["k"] % 2 == 1 and cand["k"] >= 1
        assert cand["d"] >= 4
        assert cand["css_orthogonality_verified"] is True
        assert cand["distance_type"] in ("exact_d", "upper_bound_d")
        if cand["n"] <= 48:
            assert cand["distance_type"] == "exact_d"
    assert len(rep["helios_1_bposd_cs_benchmarks"]) >= 5


# ---------------------------------------------------------------------------
# 3. Direction 6 — ADV-1 Classical-Hardness Pre-Gate Tests
# ---------------------------------------------------------------------------


def test_adv1_non_mirror_signature_and_statevector_peak() -> None:
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit.quantum_info import Statevector

    from adv1 import (
        build_adv0_mirror_control,
        build_looped_chorded_gadget_peaked,
        verify_non_mirror_signature,
    )

    ctrl = build_adv0_mirror_control(8, 3, seed=42)
    m_ctrl = verify_non_mirror_signature(ctrl)
    assert m_ctrl["mirror_structure_detected"] is True

    cand = build_looped_chorded_gadget_peaked(8, loop_layers=3, loop_gamma=0.14, seed=42)
    m_cand = verify_non_mirror_signature(cand)
    assert m_cand["cz_edge_palindrome"] is False
    assert m_cand["mirror_structure_detected"] is False
    assert m_cand["mean_block_mirror_overlap"] < 0.75

    probs = np.abs(Statevector.from_instruction(cand.to_qiskit(measure=False)).data) ** 2
    secret_idx = int(cand.secret[::-1], 2)
    assert int(np.argmax(probs)) == secret_idx
    assert probs[secret_idx] > 0.50

    # Verify M6 = J_6 - I_6 Pauli-weight-5 amplifier preserves exact peak while zeroing w_max<=4 Pauli paths
    from adv1 import attack_truncated_pauli_path

    cand_m6 = build_looped_chorded_gadget_peaked(
        12, loop_layers=3, loop_gamma=0.14, enable_m6_pauli_amplifier=True, seed=42
    )
    probs_m6 = np.abs(Statevector.from_instruction(cand_m6.to_qiskit(measure=False)).data) ** 2
    secret_m6_idx = int(cand_m6.secret[::-1], 2)
    assert int(np.argmax(probs_m6)) == secret_m6_idx
    assert probs_m6[secret_m6_idx] > 0.40
    pp_res = attack_truncated_pauli_path(cand_m6, w_max_sweep=(2, 3, 4))
    assert pp_res["any_win"] is False
    for row in pp_res["w_max_curves"]:
        if row["w_max"] <= 3:
            assert row["mean_abs_z"] == pytest.approx(0.0, abs=1e-9)
        else:
            assert row["mean_abs_z"] < 1e-3


def test_adv1_pregate_rejects_mirror_and_small_lightcone() -> None:
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    pytest.importorskip("qiskit_ibm_runtime")

    from adv1 import (
        build_adv0_mirror_control,
        build_looped_chorded_gadget_peaked,
        evaluate_adv1_pregate,
    )

    v_mirror = evaluate_adv1_pregate(
        build_adv0_mirror_control(8, 3, seed=7),
        candidate_id="test_mirror_8q",
        chi_sweep=(8, 16),
        w_max_sweep=(2, 3),
        tractable_cone_limit=12,
        mps_shots=64,
    )
    assert v_mirror.qpu_allocation_permitted is False
    assert "MIRROR_CANCELLATION_STRUCTURE_DETECTED" in v_mirror.rejection_reasons
    assert v_mirror.attack1_bounded_mps["any_win"] is True

    v_small = evaluate_adv1_pregate(
        build_looped_chorded_gadget_peaked(10, loop_layers=3, loop_gamma=0.14, seed=11),
        candidate_id="test_small_cone_10q",
        chi_sweep=(8, 16),
        w_max_sweep=(2, 3),
        tractable_cone_limit=12,
        mps_shots=64,
    )
    assert v_small.qpu_allocation_permitted is False
    assert v_small.attack3_light_cone["win"] is True


def test_adv1_pregate_report_manifest_invariants() -> None:
    assert ADV1_REPORT_PATH.is_file(), f"Missing artifact {ADV1_REPORT_PATH}"
    manifest = json.loads(ADV1_REPORT_PATH.read_text(encoding="utf-8"))
    assert manifest["schema"] == "osiris.quantum_protocol/1"
    assert manifest["evidence_class"] == "SIMULATED"
    assert manifest["provider_job_id"] is None
    assert manifest["hardware_executed"] is False
    assert manifest["summary"]["adv0_100pct_rejected"] is True
    assert all(
        not c["qpu_allocation_permitted"] for c in manifest["adv0_twirled_mirror_controls"]
    )
    assert len(manifest["summary"]["adv1_permitted_candidates"]) >= 1
    for c in manifest["adv1_looped_non_mirror_candidates"]:
        if c["qpu_allocation_permitted"]:
            assert c["mirror_audit"]["mirror_structure_detected"] is False
            assert c["attack1_bounded_mps"]["any_win"] is False
            assert c["attack2_pauli_path"]["any_win"] is False
            assert c["attack3_light_cone"]["win"] is False
            assert c["rejection_reasons"] == []
        else:
            assert len(c["rejection_reasons"]) >= 1
