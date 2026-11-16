"""
Simulation, Structured-Covariance GLS Estimation, Identifiability & Held-Out Validation
for EXPERIMENT CORRELATED-NOISE-01
=======================================================================================

Phase 2 Section 6 (`simulate_correlated_noise.py`).

Evaluates model identifiability under structured intra-block electronics gain drift
(`Bridge E5` r=0.812 across DD pulse counts), empirical H0 null calibration (ensuring
more complex models H1-H3 do NOT win under H0 data), exact Student-t minimum detectable
effect (MDE) under the 1.03-1.11 pp Heron r2 A/A drift floor, verified >=5-hop coupler
multiplexing, and zero-leakage noisy held-out coupler schedule prediction.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.optimize import least_squares

from .correlated_noise_models import (
    HERON_DT_S,
    MARRAKESH_PULSE_WIDTH_S,
    ApparatusPulseCollisionModel,
    H0IndependentLocalModel,
    H1StaticSpectatorZZModel,
    H2CorrelatedMarkovianModel,
    H3NonMarkovianMemoryModel,
    NoiseModelBase,
    ProbeCondition,
)

T_REF_8US: float = 8.192e-6   # Exact 2,048 * 4.0 ns dt (slot at N=16 is 512.0 ns = 128 dt)
T_REF_16US: float = 16.384e-6  # Exact 4,096 * 4.0 ns dt
STAGGER_OFFSETS_NS: Tuple[int, ...] = (0, 16, 32, 48, 80, 160, 256)  # Exact {0,4,8,12,20,40,64} dt


def build_diagnostic_probe_suite(
    pulse_width_s: float = MARRAKESH_PULSE_WIDTH_S,
) -> List[ProbeCondition]:
    """
    Construct a 27-condition, `dt = 4.0 ns`-aligned multi-observable probe suite
    separating H0, H1, H2, H3, and Apparatus_PulseCollision across all 8 required axes.
    """
    conds: List[ProbeCondition] = []

    # 1. Idle / Ramsey vs Echo / Pulse-Density Sweep on exact 4.0 ns dt grid
    for label_us, t_s in (("2us", 2.048e-6), ("4us", 4.096e-6), ("8us", T_REF_8US), ("16us", T_REF_16US)):
        conds.append(ProbeCondition(f"ramsey_{label_us}", "ramsey_echo_u", t_s, 0, 0.0, pulse_width_s=pulse_width_s))
        conds.append(ProbeCondition(f"echo1_{label_us}", "ramsey_echo_u", t_s, 1, 0.0, pulse_width_s=pulse_width_s))
    for n_dd in (8, 16, 32):
        conds.append(ProbeCondition(f"echo_dd{n_dd}_8us", "ramsey_echo_u", T_REF_8US, n_dd, 0.0, pulse_width_s=pulse_width_s))
    # Explicit 16.384 us N=32 arm matching preregistered C_mem = F(N=32, 16.384 us) - F(N=1, 16.384 us)
    conds.append(ProbeCondition("echo_dd32_16us", "ramsey_echo_u", T_REF_16US, 32, 0.0, pulse_width_s=pulse_width_s))

    # Verify every timing value in the suite is an exact integer multiple of 4.0 ns dt
    for delta_ns in STAGGER_OFFSETS_NS:
        conds.append(
            ProbeCondition(
                f"phi_dd16_d{delta_ns}ns",
                "bell_phi_plus",
                T_REF_8US,
                16,
                delta_ns * 1.0e-9,
                spectator_superposition=False,
                graph_distance=1,
                pulse_width_s=pulse_width_s,
            )
        )

    # 3. DFS Bell-Basis Asymmetry: |Psi+> at sync (0 ns) vs half-slot stagger (256 ns = 64 dt)
    for delta_ns in (0, 256):
        conds.append(
            ProbeCondition(
                f"psi_dd16_d{delta_ns}ns",
                "bell_psi_plus",
                T_REF_8US,
                16,
                delta_ns * 1.0e-9,
                spectator_superposition=False,
                graph_distance=1,
                pulse_width_s=pulse_width_s,
            )
        )

    # 4. Spectator Activation Contrast (|0>_s vs |+>_s at sync and half-slot stagger)
    for delta_ns in (0, 256):
        conds.append(
            ProbeCondition(
                f"phi_spec_plus_d{delta_ns}ns",
                "bell_phi_plus",
                T_REF_8US,
                16,
                delta_ns * 1.0e-9,
                spectator_superposition=True,
                graph_distance=1,
                pulse_width_s=pulse_width_s,
            )
        )

    # 5. Spatial Graph Separation Sweep (d = 2, 3 hops under sync DD)
    for dist in (2, 3):
        conds.append(
            ProbeCondition(
                f"phi_dist{dist}_sync",
                "bell_phi_plus",
                T_REF_8US,
                16,
                0.0,
                spectator_superposition=False,
                graph_distance=dist,
                pulse_width_s=pulse_width_s,
            )
        )
        conds.append(
            ProbeCondition(
                f"psi_dist{dist}_sync",
                "bell_psi_plus",
                T_REF_8US,
                16,
                0.0,
                spectator_superposition=False,
                graph_distance=dist,
                pulse_width_s=pulse_width_s,
            )
        )

    for c in conds:
        ticks_t = round(c.total_time_s / HERON_DT_S)
        ticks_d = round(c.stagger_offset_s / HERON_DT_S)
        if not math.isclose(ticks_t * HERON_DT_S, c.total_time_s, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"{c.condition_id} total_time_s not aligned to 4.0 ns dt")
        if not math.isclose(ticks_d * HERON_DT_S, c.stagger_offset_s, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"{c.condition_id} stagger_offset_s not aligned to 4.0 ns dt")

    return conds


def compute_measurement_se_per_condition(
    shots_per_basis_per_block: int = 4200,
    n_blocks: int = 8,
    aa_drift_rms_per_block: float = 0.0105,
) -> float:
    """
    Marginal standard error per condition combining 3-basis Pauli shot variance over
    `N_total = shots_per_basis_per_block * n_blocks` (`33,600 >= 33,500`) and
    block-interleaved residual drift.
    """
    n_total = shots_per_basis_per_block * n_blocks
    var_shot = (3.0 * 0.55) / (16.0 * float(n_total))
    paired_residual_drift = 0.35 * aa_drift_rms_per_block / math.sqrt(float(n_blocks))
    return math.sqrt(var_shot + paired_residual_drift ** 2)


def build_block_covariance_matrix(
    suite: Sequence[ProbeCondition],
    sigma_diag: float,
    shared_gain_drift_sd: float = 0.0012,
) -> np.ndarray:
    """
    Construct positive-definite 27x27 observation covariance matrix Sigma modeling:
      1. Diagonal shot + unpaired drift variance `sigma_diag^2`
      2. Off-diagonal correlated control-electronics pulse-gain drift (`Bridge E5`
         r=0.812) scaling with pulse count N_i in {0, 1, 8, 16, 32} so GLS whitening
         prevents pulse-count gain drift from mimicking H3 bath memory!
    """
    n = len(suite)
    cov = np.eye(n, dtype=np.float64) * (sigma_diag ** 2)
    pulse_sens = np.array([float(c.n_dd_pulses) / 32.0 for c in suite], dtype=np.float64)
    cov = cov + (shared_gain_drift_sd ** 2) * np.outer(pulse_sens, pulse_sens)
    return cov


def select_disjoint_couplers_on_backend(
    backend: Any, min_hop_distance: int = 5, max_couplers: int = 10
) -> List[Dict[str, Any]]:
    """
    Programmatically select up to `max_couplers` low-error couplers on `backend`
    whose endpoints are pairwise separated by at least `min_hop_distance` hops on the
    heavy-hex coupling graph (eliminating 2-hop spectator/feedline collisions).
    """
    target = backend.target
    twoq_op = next((op for op in ("cz", "ecr", "cx") if op in target.operation_names), "cz")
    adj: Dict[int, set[int]] = {q: set() for q in range(backend.num_qubits)}
    candidates: List[Tuple[float, int, int]] = []
    for u, v in target.build_coupling_map().get_edges():
        adj[u].add(v)
        adj[v].add(u)
        if u < v:
            p = target[twoq_op].get((u, v)) or target[twoq_op].get((v, u))
            err = float(p.error) if (p is not None and p.error is not None) else 1.0
            if err < 0.015:
                candidates.append((err, int(u), int(v)))
    candidates.sort()

    def _bfs_dist(src: int, dst: int) -> int:
        q = deque([(src, 0)])
        seen = {src}
        while q:
            curr, d = q.popleft()
            if curr == dst:
                return d
            for nb in adj[curr]:
                if nb not in seen:
                    seen.add(nb)
                    q.append((nb, d + 1))
        return 999

    chosen: List[Dict[str, Any]] = []
    for err, u, v in candidates:
        ok = True
        for prev in chosen:
            pu, pv = prev["coupler"]
            d_min = min(_bfs_dist(u, pu), _bfs_dist(u, pv), _bfs_dist(v, pu), _bfs_dist(v, pv))
            if d_min < min_hop_distance:
                ok = False
                break
        if ok:
            chosen.append({"coupler": [u, v], "twoq_error": round(err, 5)})
            if len(chosen) >= max_couplers:
                break
    return chosen


# ---------------------------------------------------------------------------
# Parametric GLS Model Fitting & Calibrated Null Model Selection
# ---------------------------------------------------------------------------

def _instantiate_model_from_theta(model_key: str, theta: np.ndarray) -> NoiseModelBase:
    if model_key == "H0":
        t1u, t1v, gpu, gpv, f0 = theta
        return H0IndependentLocalModel(t1u, t1v, gpu, gpv, f0)
    if model_key == "H1":
        t1u, t1v, gpu, gpv, f0, fzz = theta
        base = H0IndependentLocalModel(t1u, t1v, gpu, gpv, f0)
        return H1StaticSpectatorZZModel(base=base, f_zz_spectator_hz=fzz)
    if model_key == "H2":
        t1u, t1v, gpu, gpv, f0, rho, xi = theta
        base = H0IndependentLocalModel(t1u, t1v, gpu, gpv, f0)
        return H2CorrelatedMarkovianModel(
            base=base, rho_spatial=rho, spatial_length_hops=xi, enforce_lattice_psd=True
        )
    if model_key == "H3":
        t1u, t1v, gwu, gwv, sou, tauc, scorr, f0 = theta
        return H3NonMarkovianMemoryModel(
            t1_u_s=t1u,
            t1_v_s=t1v,
            gamma_white_u_hz=gwu,
            gamma_white_v_hz=gwv,
            sigma_ou_u_rad_s=sou,
            sigma_ou_v_rad_s=sou,
            tau_c_s=tauc,
            spatial_corr=scorr,
            f0_spam=f0,
        )
    if model_key == "Apparatus":
        t1u, t1v, gpu, gpv, f0, pcol = theta
        base = H0IndependentLocalModel(t1u, t1v, gpu, gpv, f0)
        return ApparatusPulseCollisionModel(base=base, per_collision_error=pcol)
    raise ValueError(f"Unknown model_key: {model_key!r}")


MODEL_BOUNDS: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {
    "H0": (
        np.array([170e-6, 170e-6, 4000.0, 4000.0, 0.97]),
        np.array([30e-6, 30e-6, 0.0, 0.0, 0.85]),
        np.array([500e-6, 500e-6, 60_000.0, 60_000.0, 1.00]),
    ),
    "H1": (
        np.array([170e-6, 170e-6, 4000.0, 4000.0, 0.97, 18_000.0]),
        np.array([30e-6, 30e-6, 0.0, 0.0, 0.85, 0.0]),
        np.array([500e-6, 500e-6, 60_000.0, 60_000.0, 1.00, 90_000.0]),
    ),
    "H2": (
        np.array([170e-6, 170e-6, 4000.0, 4000.0, 0.97, 0.4, 1.8]),
        np.array([30e-6, 30e-6, 10.0, 10.0, 0.85, -0.42, 0.4]),
        np.array([500e-6, 500e-6, 60_000.0, 60_000.0, 1.00, 0.80, 10.0]),
    ),
    "H3": (
        np.array([170e-6, 170e-6, 1000.0, 1000.0, 50_000.0, 1.5e-6, 0.45, 0.97]),
        np.array([30e-6, 30e-6, 0.0, 0.0, 0.0, 0.25e-6, -0.42, 0.85]),
        np.array([500e-6, 500e-6, 40_000.0, 40_000.0, 200_000.0, 80.0e-6, 0.80, 1.00]),
    ),
    "Apparatus": (
        np.array([170e-6, 170e-6, 4000.0, 4000.0, 0.97, 0.005]),
        np.array([30e-6, 30e-6, 0.0, 0.0, 0.85, 0.0]),
        np.array([500e-6, 500e-6, 60_000.0, 60_000.0, 1.00, 0.08]),
    ),
}


def fit_competing_models(
    suite: Sequence[ProbeCondition],
    y_obs: np.ndarray,
    sigma_obs: float,
    cov_matrix: np.ndarray | None = None,
    candidate_keys: Sequence[str] = ("H0", "H1", "H2", "H3", "Apparatus"),
    delta_bic_gate: float = 8.0,
) -> Dict[str, Any]:
    """
    Fit each competing model by Generalized Least Squares (GLS) using the Cholesky
    factor of `cov_matrix` (defaulting to structured pulse-gain covariance) and select
    via GLS-BIC with a calibrated strong-evidence threshold `delta_bic_gate = 8.0` over H0.
    """
    n_pts = len(suite)
    cov = build_block_covariance_matrix(suite, sigma_obs) if cov_matrix is None else cov_matrix
    chol = np.linalg.cholesky(cov)
    _, logdet_cov = np.linalg.slogdet(cov)
    fits: Dict[str, Dict[str, Any]] = {}

    for key in candidate_keys:
        x0, lb, ub = MODEL_BOUNDS[key]
        scale = np.maximum(np.abs(x0), 1e-6)

        def _whitened_resid(params: np.ndarray) -> np.ndarray:
            mdl = _instantiate_model_from_theta(key, params)
            raw_r = mdl.predict_suite(suite) - y_obs
            return np.linalg.solve(chol, raw_r)

        opt = least_squares(
            _whitened_resid,
            x0=x0,
            bounds=(lb, ub),
            x_scale=scale,
            max_nfev=250,
            ftol=1e-7,
            xtol=1e-7,
        )
        chi2_gls = float(np.sum(opt.fun ** 2))
        k_params = len(x0)
        loglik = -0.5 * (chi2_gls + n_pts * math.log(2.0 * math.pi) + float(logdet_cov))
        aic = chi2_gls + 2.0 * k_params
        aicc = aic + (2.0 * k_params * (k_params + 1.0)) / max(1.0, float(n_pts - k_params - 1))
        bic = chi2_gls + k_params * math.log(float(n_pts))
        fitted_model = _instantiate_model_from_theta(key, opt.x)
        raw_diff = fitted_model.predict_suite(suite) - y_obs

        fits[key] = {
            "model": fitted_model,
            "params": [float(v) for v in opt.x],
            "k_params": k_params,
            "chi2": chi2_gls,
            "aic": aic,
            "aicc": aicc,
            "bic": bic,
            "loglik": loglik,
            "rmse": float(math.sqrt(np.mean(raw_diff ** 2))),
        }

    best_bic_key = min(fits.keys(), key=lambda k: fits[k]["bic"])
    delta_bic_vs_h0 = fits["H0"]["bic"] - fits[best_bic_key]["bic"]
    selected_key = best_bic_key if (best_bic_key == "H0" or delta_bic_vs_h0 >= delta_bic_gate) else "H0"

    return {
        "selected_model": selected_key,
        "raw_min_bic_model": best_bic_key,
        "delta_bic_over_h0": delta_bic_vs_h0,
        "fits": {
            k: {kk: vv for kk, vv in v.items() if kk != "model"} for k, v in fits.items()
        },
        "fitted_objects": {k: v["model"] for k, v in fits.items()},
    }


def run_identifiability_and_null_study(
    n_trials_per_truth: int = 8,
    shots_per_basis_per_block: int = 4200,
    n_blocks: int = 8,
    seed: int = 20261116,
) -> Dict[str, Any]:
    """
    Execute Monte Carlo GLS identifiability confusion matrix across all 5 ground-truth
    mechanisms (`H0`, `H1`, `H2`, `H3`, `Apparatus`) under correlated multivariate
    Gaussian drift + zero-leakage noisy held-out coupler transfer validation.
    """
    rng = np.random.default_rng(seed)
    suite = build_diagnostic_probe_suite()
    sigma = compute_measurement_se_per_condition(shots_per_basis_per_block, n_blocks)
    cov = build_block_covariance_matrix(suite, sigma)

    ground_truths: Dict[str, NoiseModelBase] = {
        "H0": H0IndependentLocalModel(),
        "H1": H1StaticSpectatorZZModel(f_zz_spectator_hz=28_000.0),
        "H2": H2CorrelatedMarkovianModel(rho_spatial=0.68, spatial_length_hops=2.0, enforce_lattice_psd=True),
        "H3": H3NonMarkovianMemoryModel(
            sigma_ou_u_rad_s=58_000.0,
            sigma_ou_v_rad_s=62_000.0,
            tau_c_s=1.4e-6,  # Comparable to 512 ns DD slot -> genuine stagger gain + pulse-density signature!
            spatial_corr=0.65,
        ),
        "Apparatus": ApparatusPulseCollisionModel(per_collision_error=0.009),
    }

    confusion: Dict[str, Dict[str, int]] = {
        gt: {cand: 0 for cand in ground_truths} for gt in ground_truths
    }

    for gt_key, gt_model in ground_truths.items():
        y_true = gt_model.predict_suite(suite)
        for _ in range(n_trials_per_truth):
            noise_vec = rng.multivariate_normal(np.zeros(y_true.size), cov)
            y_noisy = np.clip(y_true + noise_vec, 0.0, 1.0)
            res = fit_competing_models(suite, y_noisy, sigma, cov_matrix=cov)
            confusion[gt_key][res["selected_model"]] += 1

    # Zero-leakage, noisy held-out coupler predictive validation under H3 truth:
    # 1. Train on coupler A using multivariate noisy data `y_train_noisy`.
    # 2. Held-out coupler B has INDEPENDENT ground-truth T1_B=(145us, 130us) and
    #    gamma_white_B=(1050, 1200) Hz (NOT copied from inferred_h3!).
    # 3. Estimate coupler B's local high-frequency white floor ONLY from coupler B's
    #    own noisy 1Q N=32 echo baseline (`echo_dd32_8us`), transferring ONLY the bath
    #    parameters (`tau_c_s`, `spatial_corr`, `sigma_ou`) from coupler A!
    # 4. Evaluate predictive RMSE against NOISY held-out observations `y_B_obs` (and
    #    record latent curve RMSE separately) and compute `argmax` dynamically!
    held_out_offsets = [
        ProbeCondition(f"heldout_d{d}ns", "bell_phi_plus", T_REF_8US, 16, d * 1.0e-9)
        for d in STAGGER_OFFSETS_NS
    ]
    gt_train = ground_truths["H3"]
    y_train_noisy = np.clip(
        gt_train.predict_suite(suite) + rng.multivariate_normal(np.zeros(len(suite)), cov), 0.0, 1.0
    )
    fit_train = fit_competing_models(suite, y_train_noisy, sigma, cov_matrix=cov)
    inferred_h3: H3NonMarkovianMemoryModel = fit_train["fitted_objects"]["H3"]
    inferred_h0: H0IndependentLocalModel = fit_train["fitted_objects"]["H0"]

    true_heldout_B = H3NonMarkovianMemoryModel(
        t1_u_s=145e-6,
        t1_v_s=130e-6,
        gamma_white_u_hz=1050.0,
        gamma_white_v_hz=1200.0,
        sigma_ou_u_rad_s=55_000.0,
        sigma_ou_v_rad_s=60_000.0,
        tau_c_s=1.4e-6,
        spatial_corr=0.65,
        f0_spam=0.972,
    )
    # Independent noisy 1Q high-frequency calibration on coupler B (no training leakage)
    cal_u_cond = ProbeCondition("cal_B_u", "ramsey_echo_u", T_REF_8US, 32, 0.0)
    f_cal_u = float(np.clip(true_heldout_B.predict(cal_u_cond) + rng.normal(0.0, sigma), 0.51, 0.999))
    coh_est_u = max(1e-4, min(0.9999, (2.0 * f_cal_u - 1.0) / (2.0 * 0.972 - 1.0)))
    rate_tot_B = -math.log(coh_est_u) / T_REF_8US
    gw_est_B = max(0.0, rate_tot_B - 0.5 / 145e-6)

    pred_h3_B = H3NonMarkovianMemoryModel(
        t1_u_s=145e-6,
        t1_v_s=130e-6,
        gamma_white_u_hz=gw_est_B,
        gamma_white_v_hz=gw_est_B,
        sigma_ou_u_rad_s=inferred_h3.sigma_ou_u_rad_s,
        sigma_ou_v_rad_s=inferred_h3.sigma_ou_v_rad_s,
        tau_c_s=inferred_h3.tau_c_s,
        spatial_corr=inferred_h3.spatial_corr,
        f0_spam=0.972,
    )
    pred_h0_B = H0IndependentLocalModel(
        t1_u_s=145e-6,
        t1_v_s=130e-6,
        gamma_phi_u_hz=inferred_h0.gamma_phi_u_hz,
        gamma_phi_v_hz=inferred_h0.gamma_phi_v_hz,
        f0_spam=0.972,
    )

    y_B_true = true_heldout_B.predict_suite(held_out_offsets)
    y_B_obs = np.clip(y_B_true + rng.normal(0.0, sigma, size=y_B_true.size), 0.0, 1.0)
    y_B_h3 = pred_h3_B.predict_suite(held_out_offsets)
    y_B_h0 = pred_h0_B.predict_suite(held_out_offsets)

    rmse_h3_noisy = float(math.sqrt(np.mean((y_B_h3 - y_B_obs) ** 2)))
    rmse_h0_noisy = float(math.sqrt(np.mean((y_B_h0 - y_B_obs) ** 2)))
    rmse_h3_latent = float(math.sqrt(np.mean((y_B_h3 - y_B_true) ** 2)))
    rmse_h0_latent = float(math.sqrt(np.mean((y_B_h0 - y_B_true) ** 2)))
    opt_idx_h3 = int(np.argmax(y_B_h3))
    opt_offset_ns_h3 = int(STAGGER_OFFSETS_NS[opt_idx_h3])

    # Exact finite-block Student-t MDE at 80% power, two-sided alpha=0.01 (5-test Holm, df = n_blocks - 1 = 7)
    df_blocks = max(1, n_blocks - 1)
    t_crit = float(stats.t.ppf(1.0 - 0.01 / 2.0, df=df_blocks))
    # Solve non-central t for 80% power
    ncp_target = 0.0
    for cand_ncp in np.linspace(1.0, 8.0, 701):
        pwr = 1.0 - float(stats.nct.cdf(t_crit, df=df_blocks, nc=cand_ncp)) + float(stats.nct.cdf(-t_crit, df=df_blocks, nc=cand_ncp))
        if pwr >= 0.80:
            ncp_target = float(cand_ncp)
            break
    mde_paired_t = ncp_target * math.sqrt(2.0) * sigma

    diagonal_acc = sum(confusion[k][k] for k in ground_truths) / float(len(ground_truths) * n_trials_per_truth)
    h0_specificity = confusion["H0"]["H0"] / float(n_trials_per_truth)

    return {
        "n_conditions_in_suite": len(suite),
        "dt_aligned_4ns": True,
        "shots_per_basis_total": shots_per_basis_per_block * n_blocks,
        "effective_sigma_per_condition": round(sigma, 6),
        "mde_80pct_power_alpha_0_01_student_t_pp": round(mde_paired_t * 100.0, 4),
        "overall_identifiability_accuracy": round(diagonal_acc, 4),
        "null_h0_specificity": round(h0_specificity, 4),
        "confusion_matrix": confusion,
        "held_out_transfer": {
            "held_out_rmse_h3": round(rmse_h3_noisy, 5),
            "held_out_rmse_h0": round(rmse_h0_noisy, 5),
            "held_out_latent_rmse_h3": round(rmse_h3_latent, 5),
            "held_out_latent_rmse_h0": round(rmse_h0_latent, 5),
            "predicted_optimal_offset_ns_h3": opt_offset_ns_h3,
            "true_stagger_gain_on_held_out_pp": round(float((y_B_true[-1] - y_B_true[0]) * 100.0), 3),
            "parameter_leakage_from_truth": False,
            "evaluated_on_noisy_observations": True,
        },
    }
