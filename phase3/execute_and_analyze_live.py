"""
End-to-End Live QPU Execution & Statistical Analysis for Phase 3 (`ibm_marrakesh`)
==================================================================================

Submits the pre-locked QPY manifests (`tesseract_map_02_live_pubs.qpy` and
`correlated_noise_01_live_pubs.qpy`), records real IBM Quantum Job IDs immediately
upon submission, collects raw bitstring histograms, archives post-execution usage
and calibration properties, and runs the complete pre-registered statistical analyses:
  1. `TESSERACT-MAP-02`:
     Per-block `score_tesseract_counts`, union SPAM readout error per physical qubit,
     and 5-contrast paired block t-tests with Holm-Bonferroni FWER control (`df = 9`).
  2. `CORRELATED-NOISE-01 Stage 1`:
     Per-coupler 4x4 readout assignment matrix calibration, raw & readout-mitigated
     3-basis Pauli Bell fidelities + 1Q Ramsey/Echo/XY8 fidelities across all 10
     >=5-hop disjoint couplers (4 training + 6 held-out), intra-block A/A sentinel
     drift RMS, structured block-covariance GLS 5-model fitting (`H0, H1, H2, H3,
     Apparatus`), and cluster-level zero-retraining held-out coupler validation (`df = 5`).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from qiskit import qpy
from qiskit_ibm_runtime import SamplerV2
from scipy import stats

from phase2.correlated_noise_01.correlated_noise_models import (
    H0IndependentLocalModel,
    H1StaticSpectatorZZModel,
    H2CorrelatedMarkovianModel,
    H3NonMarkovianMemoryModel,
    MARRAKESH_PULSE_WIDTH_S,
    ProbeCondition,
)
from phase2.correlated_noise_01.simulate_correlated_noise import (
    STAGGER_OFFSETS_NS,
    T_REF_8US,
    T_REF_16US,
    build_block_covariance_matrix,
    fit_competing_models,
)
from phase2.tesseract_map_02.tesseract_map_02 import (
    THETA_DEFAULT_DEG,
    analyze_paired_blocks,
    score_tesseract_counts,
)
from phase3.live_hardware_runner import (
    OUT_DIR,
    connect_service,
    extract_counts_from_pub_result,
    snapshot_backend_calibration,
)


# ---------------------------------------------------------------------------
# 1. TESSERACT-MAP-02 Raw Hardware Analysis
# ---------------------------------------------------------------------------

def analyze_tesseract_live_counts(
    lock_manifest: Mapping[str, Any],
    pub_counts: Sequence[Mapping[str, int]],
    job_id: str,
) -> Dict[str, Any]:
    union_qubits: List[int] = [int(q) for q in lock_manifest["union_active_physical_qubits"]]
    k_union = len(union_qubits)
    n_blocks = int(lock_manifest["n_blocks"])

    # Accumulate union SPAM readout errors p(1|0) and p(0|1) per physical qubit
    p1_given_0_num = np.zeros(k_union, dtype=np.float64)
    p1_given_0_den = 0.0
    p0_given_1_num = np.zeros(k_union, dtype=np.float64)
    p0_given_1_den = 0.0

    test_arms = [
        k for k in lock_manifest["arm_manifests"].keys() if not k.startswith("C0_SPAM")
    ]
    block_f_cl: Dict[str, List[float]] = {k: [0.0] * n_blocks for k in test_arms}
    block_support: Dict[str, List[float]] = {k: [0.0] * n_blocks for k in test_arms}
    block_tvd: Dict[str, List[float]] = {k: [0.0] * n_blocks for k in test_arms}

    for rec, counts in zip(lock_manifest["pub_schedule"], pub_counts):
        arm_id = rec["arm_id"]
        b_idx = int(rec["block"])
        tot = float(sum(counts.values()))
        if arm_id == "C0_UNION_SPAM_ALL0":
            p1_given_0_den += tot
            for bs, c in counts.items():
                for idx_u in range(k_union):
                    if bs[k_union - 1 - idx_u] == "1":
                        p1_given_0_num[idx_u] += c
        elif arm_id == "C0_UNION_SPAM_ALL1":
            p0_given_1_den += tot
            for bs, c in counts.items():
                for idx_u in range(k_union):
                    if bs[k_union - 1 - idx_u] == "0":
                        p0_given_1_num[idx_u] += c
        else:
            sc = score_tesseract_counts(counts, n_qubits=40, theta_deg=THETA_DEFAULT_DEG)
            block_f_cl[arm_id][b_idx] = sc["classical_fidelity"]
            block_support[arm_id][b_idx] = sc["four_string_support"]
            block_tvd[arm_id][b_idx] = sc["tvd_to_ideal"]

    p10 = p1_given_0_num / max(p1_given_0_den, 1.0)
    p01 = p0_given_1_num / max(p0_given_1_den, 1.0)
    per_qubit_spam = {
        str(union_qubits[i]): {
            "p1_given_0": round(float(p10[i]), 5),
            "p0_given_1": round(float(p01[i]), 5),
            "mean_assign_err": round(float(0.5 * (p10[i] + p01[i])), 5),
        }
        for i in range(k_union)
    }

    arm_summaries = {}
    for arm_id in test_arms:
        man = lock_manifest["arm_manifests"][arm_id]
        lay_errs = [
            per_qubit_spam[str(q)]["mean_assign_err"] for q in man["final_physical_layout"]
        ]
        arm_summaries[arm_id] = {
            "twoq_gates": man["twoq_gate_count"],
            "depth": man["circuit_depth"],
            "mean_layout_spam_err": round(float(np.mean(lay_errs)), 5),
            "max_layout_spam_err": round(float(np.max(lay_errs)), 5),
            "mean_classical_fidelity": round(float(np.mean(block_f_cl[arm_id])), 5),
            "sd_classical_fidelity": round(float(np.std(block_f_cl[arm_id], ddof=1)), 5),
            "mean_four_string_support": round(float(np.mean(block_support[arm_id])), 5),
            "sd_four_string_support": round(float(np.std(block_support[arm_id], ddof=1)), 5),
            "mean_tvd": round(float(np.mean(block_tvd[arm_id])), 5),
            "block_classical_fidelities": [round(x, 5) for x in block_f_cl[arm_id]],
            "block_four_string_supports": [round(x, 5) for x in block_support[arm_id]],
        }

    return {
        "experiment_id": "TESSERACT-MAP-02",
        "status": "Hardware-observed",
        "job_id": job_id,
        "backend": lock_manifest["backend_name"],
        "n_blocks": n_blocks,
        "shots_per_pub": lock_manifest["shots_per_pub"],
        "total_shots": lock_manifest["total_shots"],
        "prereg_lock_sha256": lock_manifest.get("lock_sha256"),
        "qpy_sha256": lock_manifest["qpy_sha256"],
        "union_spam_qubit_count": k_union,
        "union_mean_readout_error": round(float(np.mean(0.5 * (p10 + p01))), 5),
        "per_qubit_spam_calibration": per_qubit_spam,
        "arm_summaries": arm_summaries,
        "paired_contrasts_classical_fidelity": analyze_paired_blocks(block_f_cl),
        "paired_contrasts_four_string_support": analyze_paired_blocks(block_support),
    }


# ---------------------------------------------------------------------------
# 2. CORRELATED-NOISE-01 Stage 1 Raw Hardware Analysis & Held-Out Test
# ---------------------------------------------------------------------------

def _marginal_1q_prob0(counts: Mapping[str, int], slot_idx: int, n_slots: int) -> float:
    pos = n_slots - 1 - slot_idx
    tot = 0
    c0 = 0
    for bs, cnt in counts.items():
        tot += cnt
        if bs[pos] == "0":
            c0 += cnt
    return c0 / float(max(tot, 1))


def _marginal_2q_probs(counts: Mapping[str, int], slot_idx: int, n_slots: int) -> np.ndarray:
    """Return [P(00), P(01), P(10), P(11)] for `(b_u, b_v)` at coupler `slot_idx`."""
    n_bits = 2 * n_slots
    pos_u = n_bits - 1 - (2 * slot_idx)
    pos_v = n_bits - 1 - (2 * slot_idx + 1)
    vec = np.zeros(4, dtype=np.float64)
    for bs, cnt in counts.items():
        bu = 1 if bs[pos_u] == "1" else 0
        bv = 1 if bs[pos_v] == "1" else 0
        vec[2 * bu + bv] += cnt
    tot = float(np.sum(vec))
    return vec / max(tot, 1.0)


def _build_cn01_stage1_probe_conditions() -> List[ProbeCondition]:
    """23 per-coupler physical conditions measured in every Stage-1 block."""
    conds: List[ProbeCondition] = []
    for label, t_s in (("2us", 2.048e-6), ("4us", 4.096e-6), ("8us", T_REF_8US), ("16us", T_REF_16US)):
        conds.append(ProbeCondition(f"ramsey_{label}", "ramsey_echo_u", t_s, 0, 0.0, pulse_width_s=MARRAKESH_PULSE_WIDTH_S))
        conds.append(ProbeCondition(f"echo1_{label}", "ramsey_echo_u", t_s, 1, 0.0, pulse_width_s=MARRAKESH_PULSE_WIDTH_S))
    for n_dd in (8, 16, 32):
        conds.append(ProbeCondition(f"echo_dd{n_dd}_8us", "ramsey_echo_u", T_REF_8US, n_dd, 0.0, pulse_width_s=MARRAKESH_PULSE_WIDTH_S))
    conds.append(ProbeCondition("echo_dd32_16us", "ramsey_echo_u", T_REF_16US, 32, 0.0, pulse_width_s=MARRAKESH_PULSE_WIDTH_S))

    for delta_ns in STAGGER_OFFSETS_NS:
        conds.append(ProbeCondition(f"phi_dd16_d{delta_ns}ns", "bell_phi_plus", T_REF_8US, 16, delta_ns * 1e-9, False, 1, MARRAKESH_PULSE_WIDTH_S))
    for delta_ns in (0, 256):
        conds.append(ProbeCondition(f"psi_dd16_d{delta_ns}ns", "bell_psi_plus", T_REF_8US, 16, delta_ns * 1e-9, False, 1, MARRAKESH_PULSE_WIDTH_S))
    for delta_ns in (0, 256):
        conds.append(ProbeCondition(f"phi_spec_plus_d{delta_ns}ns", "bell_phi_plus", T_REF_8US, 16, delta_ns * 1e-9, True, 1, MARRAKESH_PULSE_WIDTH_S))
    return conds


def analyze_correlated_noise_01_live_counts(
    lock_manifest: Mapping[str, Any],
    pub_counts: Sequence[Mapping[str, int]],
    cal_snapshot: Mapping[str, Any],
    job_id: str,
) -> Dict[str, Any]:
    cohort = lock_manifest["coupler_cohort"]
    n_slots = len(cohort)
    n_blocks = int(lock_manifest["n_blocks"])
    suite = _build_cn01_stage1_probe_conditions()
    cond_ids = [c.condition_id for c in suite]

    # 1. Calibrate 4x4 2-qubit readout assignment matrix M_s[:, col] = P(meas | prep=col)
    m_mats = [np.eye(4, dtype=np.float64) for _ in range(n_slots)]
    prep_map = {"spam_00": 0, "spam_01": 1, "spam_10": 2, "spam_11": 3}

    # Block storage: raw & readout-mitigated observables per coupler and block
    # Structure: raw_data[slot_idx][block][cond_id][basis] -> float
    basis_vals_raw: List[List[Dict[str, Dict[str, float]]]] = [
        [{} for _ in range(n_blocks)] for _ in range(n_slots)
    ]
    basis_vals_mit: List[List[Dict[str, Dict[str, float]]]] = [
        [{} for _ in range(n_blocks)] for _ in range(n_slots)
    ]

    # First pass: extract 4x4 SPAM matrices
    for rec, counts in zip(lock_manifest["pub_schedule"], pub_counts):
        cid = rec["condition_id"]
        if cid in prep_map:
            col = prep_map[cid]
            for s_idx in range(n_slots):
                m_mats[s_idx][:, col] = _marginal_2q_probs(counts, s_idx, n_slots)

    m_invs = []
    for s_idx in range(n_slots):
        try:
            m_invs.append(np.linalg.pinv(m_mats[s_idx], rcond=1e-3))
        except np.linalg.LinAlgError:
            m_invs.append(np.eye(4, dtype=np.float64))

    # Second pass: compute per-block 1Q probabilities and 2Q Pauli correlators (raw & mitigated)
    for rec, counts in zip(lock_manifest["pub_schedule"], pub_counts):
        b_idx = int(rec["block"])
        if b_idx < 0:
            continue
        cid = rec["condition_id"]
        basis = rec["basis"]
        kind = rec["kind"]
        for s_idx in range(n_slots):
            if kind == "1q":
                p0 = _marginal_1q_prob0(counts, s_idx, n_slots)
                # 1Q readout mitigation on u using marginal of 4x4 SPAM matrix
                p1_given_0_u = m_mats[s_idx][2, 0] + m_mats[s_idx][3, 0]
                p0_given_1_u = m_mats[s_idx][0, 2] + m_mats[s_idx][1, 2]
                denom_u = max(0.2, 1.0 - p1_given_0_u - p0_given_1_u)
                p0_mit = float(np.clip((p0 - p0_given_1_u) / denom_u, 0.0, 1.0))
                basis_vals_raw[s_idx][b_idx].setdefault(cid, {})[basis] = p0
                basis_vals_mit[s_idx][b_idx].setdefault(cid, {})[basis] = p0_mit
            else:
                p_vec = _marginal_2q_probs(counts, s_idx, n_slots)
                # Parity expectation: P(00) + P(11) - P(01) - P(10)
                parity_sign = np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float64)
                ev_raw = float(np.dot(parity_sign, p_vec))
                # Mitigated probability vector projected onto simplex
                p_unb = np.maximum(m_invs[s_idx] @ p_vec, 0.0)
                if np.sum(p_unb) > 0:
                    p_unb /= np.sum(p_unb)
                ev_mit = float(np.dot(parity_sign, p_unb))
                basis_vals_raw[s_idx][b_idx].setdefault(cid, {})[basis] = ev_raw
                basis_vals_mit[s_idx][b_idx].setdefault(cid, {})[basis] = ev_mit

    def _assemble_fidelities(
        store: List[List[Dict[str, Dict[str, float]]]]
    ) -> Tuple[np.ndarray, List[float]]:
        """Return array of shape `(n_slots, n_blocks, n_conds)` + A/A sentinel diffs."""
        arr = np.zeros((n_slots, n_blocks, len(suite)), dtype=np.float64)
        aa_diffs: List[float] = []
        for s_idx in range(n_slots):
            for b_idx in range(n_blocks):
                d_map = store[s_idx][b_idx]
                for c_idx, cond in enumerate(suite):
                    b_dict = d_map[cond.condition_id]
                    if cond.observable == "ramsey_echo_u":
                        val = b_dict["X"]
                    elif cond.observable == "bell_phi_plus":
                        val = 0.25 * (1.0 + b_dict["XX"] - b_dict["YY"] + b_dict["ZZ"])
                    elif cond.observable == "bell_psi_plus":
                        val = 0.25 * (1.0 + b_dict["XX"] + b_dict["YY"] - b_dict["ZZ"])
                    else:
                        val = 0.0
                    arr[s_idx, b_idx, c_idx] = float(np.clip(val, 0.0, 1.0))
                # A/A sentinel difference within the exact same block & coupler!
                aa_b = d_map["phi_dd16_d0ns_aa_sentinel"]
                f_aa = 0.25 * (1.0 + aa_b["XX"] - aa_b["YY"] + aa_b["ZZ"])
                f_ref = arr[s_idx, b_idx, cond_ids.index("phi_dd16_d0ns")]
                aa_diffs.append(float(f_aa - f_ref))
        return arr, aa_diffs

    f_raw_arr, aa_diffs_raw = _assemble_fidelities(basis_vals_raw)
    f_mit_arr, aa_diffs_mit = _assemble_fidelities(basis_vals_mit)

    aa_rms_raw = float(math.sqrt(np.mean(np.square(aa_diffs_raw))))
    aa_rms_mit = float(math.sqrt(np.mean(np.square(aa_diffs_mit))))

    # Empirical block-averaged training curve (slots 0..3) and empirical block covariance
    train_slots = [c["slot_idx"] for c in cohort if c["role"] == "train"]
    heldout_slots = [c["slot_idx"] for c in cohort if c["role"] == "held_out"]

    # Block-level training averages: shape (n_blocks, n_conds)
    train_block_curves = np.mean(f_raw_arr[train_slots, :, :], axis=0)
    y_train_mean = np.mean(train_block_curves, axis=0)
    emp_sd_per_cond = np.std(train_block_curves, axis=0, ddof=1) / math.sqrt(float(n_blocks))
    sigma_emp = float(max(np.median(emp_sd_per_cond), 0.0015))
    # Regularized empirical + structured gain covariance for GLS (`n_blocks=5` shrinkage)
    emp_cov = np.cov(train_block_curves, rowvar=False, ddof=1) / float(n_blocks)
    struct_cov = build_block_covariance_matrix(suite, sigma_emp)
    cov_gls = 0.5 * emp_cov + 0.5 * struct_cov + np.eye(len(suite)) * 1e-7

    # Note: In Stage 1 single-coupler-hop cohort (graph_distance=1 across all 23 conditions),
    # spatial decay length xi is fixed at 2.0; fit all 5 competing models via GLS!
    gls_fit = fit_competing_models(suite, y_train_mean, sigma_emp, cov_matrix=cov_gls)

    # Four pre-registered mechanistic contrasts across all 10 couplers (Holm-Bonferroni):
    idx_phi0 = cond_ids.index("phi_dd16_d0ns")
    idx_phi48 = cond_ids.index("phi_dd16_d48ns")
    idx_phi256 = cond_ids.index("phi_dd16_d256ns")
    idx_psi0 = cond_ids.index("psi_dd16_d0ns")
    idx_psi256 = cond_ids.index("psi_dd16_d256ns")
    idx_spec0 = cond_ids.index("phi_spec_plus_d0ns")
    idx_spec256 = cond_ids.index("phi_spec_plus_d256ns")
    idx_e1_16 = cond_ids.index("echo1_16us")
    idx_e32_16 = cond_ids.index("echo_dd32_16us")

    coupler_means_raw = np.mean(f_raw_arr, axis=1)  # (n_slots, n_conds)
    coupler_means_mit = np.mean(f_mit_arr, axis=1)

    c_stagger_phi = coupler_means_raw[:, idx_phi256] - coupler_means_raw[:, idx_phi0]
    c_dfs = (coupler_means_raw[:, idx_phi256] - coupler_means_raw[:, idx_phi0]) - (
        coupler_means_raw[:, idx_psi256] - coupler_means_raw[:, idx_psi0]
    )
    c_spec = (coupler_means_raw[:, idx_spec256] - coupler_means_raw[:, idx_spec0]) - (
        coupler_means_raw[:, idx_phi256] - coupler_means_raw[:, idx_phi0]
    )
    c_mem = coupler_means_raw[:, idx_e32_16] - coupler_means_raw[:, idx_e1_16]

    def _cluster_ttest(vec: np.ndarray) -> Dict[str, float]:
        mean_v = float(np.mean(vec))
        se_v = float(np.std(vec, ddof=1) / math.sqrt(vec.size))
        t_val = mean_v / max(se_v, 1e-12)
        p_val = float(2.0 * stats.t.sf(abs(t_val), df=vec.size - 1))
        return {
            "mean": round(mean_v, 5),
            "se_across_couplers": round(se_v, 5),
            "t_stat_df9": round(t_val, 3),
            "p_two_sided": p_val,
        }

    mech_raw = [
        ("C_stagger_phi_256_minus_0", _cluster_ttest(c_stagger_phi)),
        ("C_DFS_asymmetry", _cluster_ttest(c_dfs)),
        ("C_spectator_activation", _cluster_ttest(c_spec)),
        ("C_memory_dd32_minus_echo1_16us", _cluster_ttest(c_mem)),
    ]
    # Holm-Bonferroni adjustment across the 4 mechanistic contrasts
    order_m = sorted(range(len(mech_raw)), key=lambda i: mech_raw[i][1]["p_two_sided"])
    run_max = 0.0
    mech_contrasts: Dict[str, Any] = {}
    for rank, idx_m in enumerate(order_m):
        name_m, d_m = mech_raw[idx_m]
        p_adj = min(1.0, d_m["p_two_sided"] * (len(mech_raw) - rank))
        run_max = max(run_max, p_adj)
        d_m["p_holm_adjusted"] = round(run_max, 6)
        d_m["significant_holm_0_05"] = run_max < 0.05
        mech_contrasts[name_m] = d_m

    denom_step = float(np.mean(coupler_means_raw[:, idx_phi256] - coupler_means_raw[:, idx_phi0]))
    step_s48 = (
        float(np.mean(coupler_means_raw[:, idx_phi48] - coupler_means_raw[:, idx_phi0])) / denom_step
        if abs(denom_step) > 0.002
        else 0.0
    )

    # Zero-retraining cluster-level held-out evaluation across all 6 held-out couplers (`df = 5`)!
    # Transfer inferred model parameters from training cohort, substituting ONLY each held-out
    # coupler's own calibration T1_{u,v} and local 1Q N=32 echo baseline + T=0 SPAM!
    offset_indices = [cond_ids.index(f"phi_dd16_d{d}ns") for d in STAGGER_OFFSETS_NS]
    offset_conds = [suite[i] for i in offset_indices]
    inferred_objs = gls_fit["fitted_objects"]
    selected_key = gls_fit["selected_model"]

    heldout_rmse_h0: List[float] = []
    heldout_rmse_h3: List[float] = []
    heldout_rmse_sel: List[float] = []
    per_coupler_profiles: List[Dict[str, Any]] = []

    q_cal = cal_snapshot.get("qubits", {})
    for s_idx, c_info in enumerate(cohort):
        u, v = c_info["coupler"]
        t1_u = float(q_cal.get(str(u), {}).get("t1_s") or 170e-6)
        t1_v = float(q_cal.get(str(v), {}).get("t1_s") or 170e-6)
        # Local 1Q white floor from this coupler's `echo_dd32_8us`
        f_1q_32 = float(coupler_means_raw[s_idx, cond_ids.index("echo_dd32_8us")])
        f_1q_r0 = float(coupler_means_raw[s_idx, cond_ids.index("ramsey_8us")])
        # Estimate local SPAM ceiling from 2Q readout assignment diagonal average
        ro_diag = float(np.mean(np.diag(m_mats[s_idx])))
        f0_local = float(np.clip(ro_diag * (1.0 - 2.0 * float(c_info["twoq_error"])), 0.85, 0.995))

        coh_32 = max(1e-4, min(0.9999, (2.0 * f_1q_32 - 1.0) / max(0.1, 2.0 * math.sqrt(f0_local) - 1.0)))
        gw_local = max(0.0, -math.log(coh_32) / T_REF_8US - 0.5 / max(t1_u, 1e-5))
        coh_r0 = max(1e-4, min(0.9999, (2.0 * f_1q_r0 - 1.0) / max(0.1, 2.0 * math.sqrt(f0_local) - 1.0)))
        gphi_local = max(0.0, -math.log(coh_r0) / T_REF_8US - 0.5 / max(t1_u, 1e-5))

        if c_info["role"] == "held_out":
            obs_curve = coupler_means_raw[s_idx, offset_indices]
            h0_tr: H0IndependentLocalModel = inferred_objs["H0"]
            h3_tr: H3NonMarkovianMemoryModel = inferred_objs["H3"]

            pred_h0 = H0IndependentLocalModel(
                t1_u_s=t1_u, t1_v_s=t1_v, gamma_phi_u_hz=gphi_local, gamma_phi_v_hz=gphi_local, f0_spam=f0_local
            ).predict_suite(offset_conds)
            pred_h3 = H3NonMarkovianMemoryModel(
                t1_u_s=t1_u,
                t1_v_s=t1_v,
                gamma_white_u_hz=gw_local,
                gamma_white_v_hz=gw_local,
                sigma_ou_u_rad_s=h3_tr.sigma_ou_u_rad_s,
                sigma_ou_v_rad_s=h3_tr.sigma_ou_v_rad_s,
                tau_c_s=h3_tr.tau_c_s,
                spatial_corr=h3_tr.spatial_corr,
                f0_spam=f0_local,
            ).predict_suite(offset_conds)

            r_h0 = float(math.sqrt(np.mean((pred_h0 - obs_curve) ** 2)))
            r_h3 = float(math.sqrt(np.mean((pred_h3 - obs_curve) ** 2)))
            heldout_rmse_h0.append(r_h0)
            heldout_rmse_h3.append(r_h3)
            heldout_rmse_sel.append(r_h3 if selected_key == "H3" else r_h0)

        per_coupler_profiles.append({
            "slot_idx": s_idx,
            "role": c_info["role"],
            "coupler": c_info["coupler"],
            "spectator": c_info["spectator"],
            "twoq_error": c_info["twoq_error"],
            "readout_assignment_diag_mean": round(ro_diag, 5),
            "phi_offset_curve_raw": {
                f"{d}ns": round(float(coupler_means_raw[s_idx, cond_ids.index(f"phi_dd16_d{d}ns")]), 5)
                for d in STAGGER_OFFSETS_NS
            },
            "phi_offset_curve_mitigated": {
                f"{d}ns": round(float(coupler_means_mit[s_idx, cond_ids.index(f"phi_dd16_d{d}ns")]), 5)
                for d in STAGGER_OFFSETS_NS
            },
            "stagger_256_minus_0_pp_raw": round(float(c_stagger_phi[s_idx] * 100.0), 3),
            "dfs_asymmetry_pp_raw": round(float(c_dfs[s_idx] * 100.0), 3),
            "memory_16us_dd32_minus_echo1_pp_raw": round(float(c_mem[s_idx] * 100.0), 3),
        })

    rmse_diff = np.array(heldout_rmse_h0) - np.array(heldout_rmse_h3)  # > 0 means H3 has lower RMSE than H0
    t_ho = float(np.mean(rmse_diff) / max(np.std(rmse_diff, ddof=1) / math.sqrt(rmse_diff.size), 1e-12))
    p_ho_two_sided = float(2.0 * stats.t.sf(abs(t_ho), df=rmse_diff.size - 1))

    return {
        "experiment_id": "CORRELATED-NOISE-01-STAGE1",
        "status": "Hardware-observed",
        "job_id": job_id,
        "backend": lock_manifest["backend_name"],
        "n_blocks": n_blocks,
        "shots_per_pub": lock_manifest["shots_per_pub"],
        "total_circuit_shots": lock_manifest["total_circuit_shots"],
        "total_multiplexed_coupler_shots": lock_manifest["total_multiplexed_coupler_shots"],
        "prereg_lock_sha256": lock_manifest.get("lock_sha256"),
        "qpy_sha256": lock_manifest["qpy_sha256"],
        "live_aa_sentinel_rms_pp_raw": round(aa_rms_raw * 100.0, 4),
        "live_aa_sentinel_rms_pp_mitigated": round(aa_rms_mit * 100.0, 4),
        "gls_model_selection_on_training_cohort": {
            "selected_model": gls_fit["selected_model"],
            "raw_min_bic_model": gls_fit["raw_min_bic_model"],
            "delta_bic_over_h0": round(float(gls_fit["delta_bic_over_h0"]), 3),
            "fits": gls_fit["fits"],
        },
        "mechanistic_contrasts_10couplers_holm": mech_contrasts,
        "step_fraction_s48": round(step_s48, 4),
        "held_out_6coupler_cluster_validation_df5": {
            "n_held_out_couplers": len(heldout_rmse_h0),
            "mean_rmse_h0": round(float(np.mean(heldout_rmse_h0)), 5),
            "mean_rmse_h3": round(float(np.mean(heldout_rmse_h3)), 5),
            "per_coupler_rmse_h0": [round(x, 5) for x in heldout_rmse_h0],
            "per_coupler_rmse_h3": [round(x, 5) for x in heldout_rmse_h3],
            "t_stat_h0_minus_h3_df5": round(t_ho, 3),
            "p_two_sided_df5": p_ho_two_sided,
            "transfer_beats_h0_p_0_05": bool(np.mean(rmse_diff) > 0 and p_ho_two_sided < 0.05),
        },
        "training_mean_curve_23conds": {
            c.condition_id: round(float(y_train_mean[i]), 5) for i, c in enumerate(suite)
        },
        "per_coupler_profiles": per_coupler_profiles,
    }


# ---------------------------------------------------------------------------
# 3. Sequential Live Submission & Collection Driver
# ---------------------------------------------------------------------------

QUOTA_SAFETY_FACTOR = 1.3  # HP-001 on ibm_fez used 12 s against a 10.1 s estimate (1.19x)


def _remaining_seconds(svc: Any) -> float:
    return float(svc.usage().get("usage_remaining_seconds", 0) or 0)


def _job_usage_seconds(job: Any) -> float | None:
    try:
        return float(job.usage())
    except Exception:
        try:
            return float(job.metrics()["usage"]["quantum_seconds"])
        except Exception:
            return None


def _run_or_resume(
    svc: Any,
    backend: Any,
    experiment_id: str,
    stem: str,
    remaining_s: float,
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    """Submit one locked campaign once, or resume it, and return (lock, raw payload, QPU s used now).

    - Raw counts on disk: nothing is submitted; they are re-analyzed.
    - Submission receipt on disk: the job is retrieved by its ID, never resubmitted.
    - Otherwise: the QPY must match its lock's SHA-256 and the remaining quota must cover
      QUOTA_SAFETY_FACTOR x the lock's estimate, or nothing is submitted.
    """
    lock_path = OUT_DIR / f"{stem}_live_prereg_lock.json"
    qpy_path = OUT_DIR / f"{stem}_live_pubs.qpy"
    receipt_path = OUT_DIR / f"{stem}_submission_receipt.json"
    raw_path = OUT_DIR / f"{stem}_live_raw_counts.json"
    lock = json.loads(lock_path.read_text())
    lock["lock_sha256"] = hashlib.sha256(lock_path.read_bytes()).hexdigest()

    if raw_path.exists():
        return lock, json.loads(raw_path.read_text()), 0.0

    if receipt_path.exists():
        jid = json.loads(receipt_path.read_text())["job_id"]
        print(f"[{experiment_id}] Resuming job_id={jid} from its receipt (no resubmission).", flush=True)
        job = svc.job(jid)
        n_pubs = int(lock["total_pubs"])
    else:
        if hashlib.sha256(qpy_path.read_bytes()).hexdigest() != lock["qpy_sha256"]:
            raise RuntimeError(f"{experiment_id}: QPY does not match its prereg lock; not submitting")
        need = QUOTA_SAFETY_FACTOR * float(lock["estimated_qpu_seconds"])
        if remaining_s < need:
            raise RuntimeError(
                f"{experiment_id}: {remaining_s:.0f} s of quota left, need {need:.0f} s "
                f"({QUOTA_SAFETY_FACTOR}x the {lock['estimated_qpu_seconds']} s estimate); not submitting"
            )
        with qpy_path.open("rb") as f:
            pubs = qpy.load(f)
        job = SamplerV2(mode=backend).run(pubs, shots=int(lock["shots_per_pub"]))
        jid = job.job_id()
        receipt_path.write_text(
            json.dumps(
                {
                    "experiment_id": experiment_id,
                    "job_id": jid,
                    "backend": backend.name,
                    "submitted_utc": datetime.now(timezone.utc).isoformat(),
                    "prereg_lock_sha256": lock["lock_sha256"],
                    "qpy_sha256": lock["qpy_sha256"],
                    "quota_remaining_before_s": remaining_s,
                },
                indent=2,
            )
            + "\n"
        )
        n_pubs = len(pubs)
        print(f"[{experiment_id}] Submitted job_id={jid} ({n_pubs} PUBs). Waiting for result...", flush=True)

    res = job.result()
    usage_s = _job_usage_seconds(job)
    raw = {
        "job_id": jid,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "job_usage_seconds": usage_s,
        "estimated_qpu_seconds": lock["estimated_qpu_seconds"],
        "post_job_usage": svc.usage(),
        "execution_time_calibration": snapshot_backend_calibration(backend),
        "pub_counts": [extract_counts_from_pub_result(res[i]) for i in range(n_pubs)],
    }
    raw_path.write_text(json.dumps(raw) + "\n")
    print(
        f"[{experiment_id}] Completed job_id={jid}: {usage_s} s used "
        f"(estimate {lock['estimated_qpu_seconds']} s). Raw counts saved.",
        flush=True,
    )
    return lock, raw, float(usage_s or 0.0)


def run_live_campaigns(campaigns: Sequence[str]) -> None:
    svc = connect_service()
    remaining = _remaining_seconds(svc)
    print(f"[Start] OpenPlan remaining: {remaining:.0f} s", flush=True)
    backend = svc.backend("ibm_marrakesh")

    if "tmap02" in campaigns:
        t_lock, t_raw, t_used = _run_or_resume(svc, backend, "TESSERACT-MAP-02", "tesseract_map_02", remaining)
        t_analysis = analyze_tesseract_live_counts(t_lock, t_raw["pub_counts"], t_raw["job_id"])
        (OUT_DIR / "tesseract_map_02_live_analysis.json").write_text(json.dumps(t_analysis, indent=2) + "\n")
        print("[TESSERACT-MAP-02] Analysis complete.", flush=True)
        # The usage endpoint can lag a job that just finished, so also subtract what the job reported.
        remaining = min(_remaining_seconds(svc), remaining - t_used)

    if "cn01" in campaigns:
        c_lock, c_raw, _ = _run_or_resume(
            svc, backend, "CORRELATED-NOISE-01-STAGE1", "correlated_noise_01", remaining
        )
        c_analysis = analyze_correlated_noise_01_live_counts(
            c_lock, c_raw["pub_counts"], c_raw["execution_time_calibration"], c_raw["job_id"]
        )
        (OUT_DIR / "correlated_noise_01_live_analysis.json").write_text(json.dumps(c_analysis, indent=2) + "\n")
        print("[CORRELATED-NOISE-01] Analysis complete.", flush=True)

    print(f"[Done] OpenPlan remaining: {_remaining_seconds(svc):.0f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit or resume the locked Phase 3 campaigns, then analyze.")
    parser.add_argument(
        "--campaign",
        choices=["tmap02", "cn01", "both"],
        required=True,
        help="Which locked campaign(s) to run. Each is submitted at most once; reruns resume or re-analyze.",
    )
    args = parser.parse_args()
    run_live_campaigns(("tmap02", "cn01") if args.campaign == "both" else (args.campaign,))


if __name__ == "__main__":
    main()
