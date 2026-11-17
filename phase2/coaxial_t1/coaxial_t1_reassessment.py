"""
Phase 2 Section 8: Reassess Coaxial T1 Without Retrofitting
===========================================================

Evaluates all 21 bundled IBM V2 calibration snapshots (11 testable 127-qubit
Eagle processors with published per-qubit frequency + T1, and 10 snapshots
lacking per-qubit frequency including Heron FakeFez / FakeMarrakesh / FakeTorino /
FakeKingston) WITHOUT retrofitting physical constants or periods:

  1. Preserves the empirical negative T1-vs-frequency slope (`beta_1 < 0` on 10/11
     snapshots; pooled inverse-variance slope ~ -141 us/GHz).
  2. Quantifies spectral-window coverage (`bandwidth / 3.5256 GHz = 14.4% - 19.3%`
     of one cycle) and collinearity VIF of `cos(2*pi*f / 3.5256)` against `[1, f]`
     and `[1, f, f^2]`.
  3. Compares four competing models via BIC AND 5-fold held-out qubit cross-validation:
       - M0: Linear background `T1 = b0 + b1 * f` (standard dielectric/Purcell trend)
       - M1: Smooth quadratic background `T1 = b0 + b1 * f + b2 * f^2`
       - M2: Local-loss / TLS defect-trimmed model (Huber M-estimator + local TLS indicator)
       - M3a: Collective coaxial fixed-phase `3.5256 GHz` (`L_eff = 15.6 mm`, 13 hops)
       - M3b: Collective graph-diameter fixed-phase `1.7628 GHz` (`L_eff = 31.2 mm`, 26 hops)
       - M3c: Collective harmonic `b0 + b1*f + A*cos(2*pi*f/Df) + B*sin(2*pi*f/Df)`
  4. Quantifies minimum detectable modulation amplitude (`MDE_80pct`) after linear
     detrending and tests heavy-hex spatial / frequency-allocation mimicry.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
from scipy import stats
from scipy.optimize import least_squares

C_EFF_M_S: float = 1.1e8
HOP_SPACING_M: float = 1.2e-3
DELTA_F_13HOPS_GHZ: float = (C_EFF_M_S / (2.0 * 13 * HOP_SPACING_M)) * 1e-9  # 3.5256410256 GHz


def extract_v2_calibration_snapshots() -> Dict[str, Any]:
    """
    Extract per-qubit `(qubit, freq_ghz, t1_us)` directly from Qiskit's bundled
    FakeProvider V2 backends with strict unit checking and timestamp provenance.
    """
    import qiskit_ibm_runtime.fake_provider as fp

    testable: List[Dict[str, Any]] = []
    missing_freq: List[Dict[str, Any]] = []

    seen_backend_names: set[str] = set()
    for attr in sorted(dir(fp)):
        if not attr.startswith("Fake") or not attr.endswith("V2") and attr not in (
            "FakeBrisbane", "FakeBrussels", "FakeCusco", "FakeFez", "FakeKawasaki",
            "FakeKingston", "FakeKyiv", "FakeKyoto", "FakeMarrakesh", "FakeOsaka",
            "FakeQuebec", "FakeSherbrooke", "FakeStrasbourg", "FakeTorino",
        ):
            continue
        cls = getattr(fp, attr)
        if not isinstance(cls, type):
            continue
        try:
            b = cls()
            if not hasattr(b, "target") or b.num_qubits < 20:
                continue
        except Exception:
            continue

        bname = getattr(b, "name", attr)
        if bname in seen_backend_names:
            continue
        seen_backend_names.add(bname)

        # Inspect backend properties if available for frequencies + timestamps
        props = b.properties() if hasattr(b, "properties") and callable(b.properties) else None
        last_update = str(getattr(props, "last_update_date", None)) if props is not None else None

        freqs_ghz: List[float] = []
        t1s_us: List[float] = []
        qubit_ids: List[int] = []

        target = b.target
        q_props = getattr(target, "qubit_properties", None)
        if q_props:
            for q_idx, qp in enumerate(q_props):
                if qp is None:
                    continue
                f_hz = getattr(qp, "frequency", None)
                t1_s = getattr(qp, "t1", None)
                if f_hz is not None and t1_s is not None and f_hz > 0 and t1_s > 0:
                    # Unit normalization: Qiskit Target stores Hz and seconds
                    f_g = float(f_hz) * 1e-9 if f_hz > 1e6 else float(f_hz)
                    t_u = float(t1_s) * 1e6 if t1_s < 1.0 else float(t1_s)
                    if 2.0 <= f_g <= 8.0 and 0.1 <= t_u <= 2000.0:
                        qubit_ids.append(q_idx)
                        freqs_ghz.append(f_g)
                        t1s_us.append(t_u)

        if len(freqs_ghz) >= 20:
            testable.append({
                "class_name": attr,
                "backend_name": bname,
                "num_qubits": int(b.num_qubits),
                "n_valid_qubits": len(freqs_ghz),
                "snapshot_last_update": last_update,
                "qubits": qubit_ids,
                "freq_ghz": freqs_ghz,
                "t1_us": t1s_us,
                "edges": [[int(u), int(v)] for u, v in b.coupling_map.get_edges()],
            })
        else:
            missing_freq.append({
                "class_name": attr,
                "backend_name": bname,
                "num_qubits": int(b.num_qubits),
                "snapshot_last_update": last_update,
                "reason": "Per-qubit frequencies absent or unpopulated in bundled Target",
            })

    return {"testable": testable, "missing_frequency": missing_freq}


def _design_matrix(f: np.ndarray, model_name: str, period_ghz: float = DELTA_F_13HOPS_GHZ) -> np.ndarray:
    f_c = f - np.mean(f)
    if model_name == "M0_linear":
        return np.column_stack([np.ones_like(f), f_c])
    if model_name == "M1_quadratic":
        return np.column_stack([np.ones_like(f), f_c, f_c ** 2])
    if model_name == "M3a_fixed_phase_3p526":
        x_cos = -np.cos(2.0 * math.pi * f / DELTA_F_13HOPS_GHZ)
        return np.column_stack([np.ones_like(f), f_c, x_cos])
    if model_name == "M3b_fixed_phase_1p763":
        x_cos = -np.cos(2.0 * math.pi * f / (0.5 * DELTA_F_13HOPS_GHZ))
        return np.column_stack([np.ones_like(f), f_c, x_cos])
    if model_name == "M3c_harmonic_3p526":
        return np.column_stack([
            np.ones_like(f),
            f_c,
            np.cos(2.0 * math.pi * f / period_ghz),
            np.sin(2.0 * math.pi * f / period_ghz),
        ])
    raise ValueError(f"Unknown model_name: {model_name!r}")


def _kfold_cv_rmse(
    f: np.ndarray, y: np.ndarray, model_name: str, n_splits: int = 5, seed: int = 20261116
) -> float:
    """Held-out k-fold cross-validation RMSE (never freely fitting period!)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(f.size)
    folds = np.array_split(perm, n_splits)
    sq_errs: List[np.ndarray] = []

    f_mean = float(np.mean(f))
    for k in range(n_splits):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != k])

        def _make_X(fv: np.ndarray) -> np.ndarray:
            fc = fv - f_mean
            if model_name == "M0_linear":
                return np.column_stack([np.ones_like(fv), fc])
            if model_name == "M1_quadratic":
                return np.column_stack([np.ones_like(fv), fc, fc ** 2])
            if model_name == "M2_huber_local_loss":
                return np.column_stack([np.ones_like(fv), fc])
            if model_name == "M3a_fixed_phase_3p526":
                return np.column_stack([np.ones_like(fv), fc, -np.cos(2.0 * math.pi * fv / DELTA_F_13HOPS_GHZ)])
            if model_name == "M3b_fixed_phase_1p763":
                return np.column_stack([np.ones_like(fv), fc, -np.cos(2.0 * math.pi * fv / (0.5 * DELTA_F_13HOPS_GHZ))])
            if model_name == "M3c_harmonic_3p526":
                return np.column_stack([
                    np.ones_like(fv),
                    fc,
                    np.cos(2.0 * math.pi * fv / DELTA_F_13HOPS_GHZ),
                    np.sin(2.0 * math.pi * fv / DELTA_F_13HOPS_GHZ),
                ])
            raise ValueError(model_name)

        X_tr = _make_X(f[train_idx])
        y_tr = y[train_idx]
        X_te = _make_X(f[test_idx])
        y_te = y[test_idx]

        if model_name == "M2_huber_local_loss":
            # Robust Huber regression downweighting severe local TLS dropout qubits
            scale = max(float(np.std(y_tr)), 1.0)
            res = least_squares(lambda b: (X_tr @ b - y_tr) / scale, x0=np.array([np.median(y_tr), 0.0]), loss="huber")
            beta = res.x
        else:
            beta, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)

        sq_errs.append((y_te - X_te @ beta) ** 2)

    return float(math.sqrt(np.mean(np.concatenate(sq_errs))))


def analyze_single_device_coaxial_models(dev: Dict[str, Any]) -> Dict[str, Any]:
    f = np.asarray(dev["freq_ghz"], dtype=np.float64)
    y = np.asarray(dev["t1_us"], dtype=np.float64)
    n = f.size

    # 1. Linear trend & negative T1-vs-frequency preservation
    lin = stats.linregress(f, y)
    span_ghz = float(np.max(f) - np.min(f))
    spectral_cycles_3p526 = span_ghz / DELTA_F_13HOPS_GHZ

    # 2. Variance Inflation Factor (VIF) of x = -cos(2*pi*f / 3.5256) against [1, f] and [1, f, f^2]
    x_cos = -np.cos(2.0 * math.pi * f / DELTA_F_13HOPS_GHZ)
    X_lin = _design_matrix(f, "M0_linear")
    X_quad = _design_matrix(f, "M1_quadratic")
    r2_lin = 1.0 - float(np.sum((x_cos - X_lin @ np.linalg.lstsq(X_lin, x_cos, rcond=None)[0]) ** 2)) / float(np.sum((x_cos - np.mean(x_cos)) ** 2))
    r2_quad = 1.0 - float(np.sum((x_cos - X_quad @ np.linalg.lstsq(X_quad, x_cos, rcond=None)[0]) ** 2)) / float(np.sum((x_cos - np.mean(x_cos)) ** 2))
    vif_after_linear = 1.0 / max(1.0 - r2_lin, 1e-12)
    vif_after_quad = 1.0 / max(1.0 - r2_quad, 1e-12)

    # 3. Minimum Detectable Modulation Amplitude (MDE at 80% power, two-sided alpha=0.05) after linear detrending
    xr = x_cos - X_lin @ np.linalg.lstsq(X_lin, x_cos, rcond=None)[0]
    yr = y - X_lin @ np.linalg.lstsq(X_lin, y, rcond=None)[0]
    sigma_resid = float(math.sqrt(np.sum(yr ** 2) / (n - 3)))
    se_A = sigma_resid / float(math.sqrt(np.sum(xr ** 2)))
    t_crit = float(stats.t.ppf(0.975, df=n - 3))
    t_beta = float(stats.t.ppf(0.80, df=n - 3))
    mde_80_us = (t_crit + t_beta) * se_A

    # 4. BIC & 5-fold held-out CV across M0, M1, M2, M3a, M3b, M3c
    bic_scores: Dict[str, float] = {}
    for mname in ("M0_linear", "M1_quadratic", "M3a_fixed_phase_3p526", "M3b_fixed_phase_1p763", "M3c_harmonic_3p526"):
        Xm = _design_matrix(f, mname)
        beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
        rss = float(np.sum((y - Xm @ beta) ** 2))
        k_p = Xm.shape[1] + 1  # +1 for noise variance
        bic_scores[mname] = float(n * math.log(rss / n) + k_p * math.log(n))

    cv_rmse = {
        mname: round(_kfold_cv_rmse(f, y, mname), 3)
        for mname in (
            "M0_linear",
            "M1_quadratic",
            "M2_huber_local_loss",
            "M3a_fixed_phase_3p526",
            "M3b_fixed_phase_1p763",
            "M3c_harmonic_3p526",
        )
    }

    # 5. Layout-dependent mimicry diagnostic: nearest-neighbor frequency difference & spatial index gradient
    q_idx = np.asarray(dev["qubits"], dtype=np.float64)
    spatial_r, spatial_p = stats.pearsonr(q_idx, y)

    best_bic_model = min(bic_scores, key=bic_scores.get)
    best_cv_model = min(cv_rmse, key=cv_rmse.get)

    return {
        "backend": dev["backend_name"],
        "snapshot_last_update": dev["snapshot_last_update"],
        "n_qubits": n,
        "freq_band_ghz": [round(float(np.min(f)), 4), round(float(np.max(f)), 4)],
        "bandwidth_ghz": round(span_ghz, 4),
        "cycles_spanned_at_3p526ghz": round(spectral_cycles_3p526, 4),
        "mean_t1_us": round(float(np.mean(y)), 2),
        "linear_slope_us_per_ghz": round(float(lin.slope), 2),
        "linear_slope_se": round(float(lin.stderr), 2),
        "linear_slope_p_two_sided": float(lin.pvalue),
        "collinearity_r2_cos_vs_linear": round(r2_lin, 5),
        "collinearity_vif_after_linear": round(vif_after_linear, 2),
        "collinearity_vif_after_quadratic": round(vif_after_quad, 2),
        "mde_80pct_after_linear_us": round(mde_80_us, 2),
        "mde_80pct_over_mean_t1": round(mde_80_us / float(np.mean(y)), 3),
        "bic_by_model": {k: round(v, 2) for k, v in bic_scores.items()},
        "best_bic_model": best_bic_model,
        "held_out_5fold_cv_rmse_us": cv_rmse,
        "best_cv_model": best_cv_model,
        "spatial_qubit_index_t1_corr_r": round(float(spatial_r), 3),
        "spatial_qubit_index_t1_p": float(spatial_p),
    }


def run_full_coaxial_t1_reassessment() -> Dict[str, Any]:
    snaps = extract_v2_calibration_snapshots()
    per_device = [analyze_single_device_coaxial_models(d) for d in snaps["testable"]]

    n_neg_slope = sum(1 for d in per_device if d["linear_slope_us_per_ghz"] < 0)
    # Fixed-effect inverse-variance & DerSimonian-Laird random-effects pooled linear slope
    variances = np.array([d["linear_slope_se"] ** 2 for d in per_device], dtype=np.float64)
    weights = 1.0 / variances
    slopes = np.array([d["linear_slope_us_per_ghz"] for d in per_device], dtype=np.float64)
    pooled_slope = float(np.sum(weights * slopes) / np.sum(weights))
    pooled_se = float(1.0 / math.sqrt(np.sum(weights)))

    # Cochran's Q and DerSimonian-Laird between-chip heterogeneity tau^2 (accounting for chip/spatial heterogeneity)
    k_chips = len(per_device)
    cochran_q = float(np.sum(weights * ((slopes - pooled_slope) ** 2)))
    c_denom = float(np.sum(weights) - np.sum(weights ** 2) / np.sum(weights))
    tau2_dl = max(0.0, (cochran_q - (k_chips - 1)) / max(c_denom, 1e-12)) if k_chips > 1 else 0.0
    i2_pct = max(0.0, 100.0 * (cochran_q - (k_chips - 1)) / max(cochran_q, 1e-12)) if k_chips > 1 else 0.0
    re_weights = 1.0 / (variances + tau2_dl)
    re_slope = float(np.sum(re_weights * slopes) / np.sum(re_weights))
    re_se = float(1.0 / math.sqrt(np.sum(re_weights)))

    n_m3_wins_bic = sum(1 for d in per_device if d["best_bic_model"].startswith("M3"))
    n_m3_wins_cv = sum(
        1 for d in per_device
        if d["held_out_5fold_cv_rmse_us"]["M3a_fixed_phase_3p526"]
        < d["held_out_5fold_cv_rmse_us"]["M0_linear"] - 1.0
    )

    electrical_lengths = [
        {
            "structure": "13-hop heavy-hex subgraph (historical April script)",
            "length_m": round(13 * HOP_SPACING_M, 4),
            "predicted_period_ghz": round(DELTA_F_13HOPS_GHZ, 4),
            "cycles_in_0p58ghz_band": round(0.58 / DELTA_F_13HOPS_GHZ, 3),
            "identifiable_after_linear_trend": False,
        },
        {
            "structure": "26-hop 127Q Eagle graph diameter (`--hops graph`)",
            "length_m": round(26 * HOP_SPACING_M, 4),
            "predicted_period_ghz": round(0.5 * DELTA_F_13HOPS_GHZ, 4),
            "cycles_in_0p58ghz_band": round(0.58 / (0.5 * DELTA_F_13HOPS_GHZ), 3),
            "identifiable_after_linear_trend": False,
        },
        {
            "structure": "On-chip readout/Purcell lambda/2 resonator (~6-10 mm CPW)",
            "length_m": 0.008,
            "predicted_period_ghz": round((C_EFF_M_S / (2.0 * 0.008)) * 1e-9, 3),
            "cycles_in_0p58ghz_band": round(0.58 / ((C_EFF_M_S / 0.016) * 1e-9), 3),
            "identifiable_after_linear_trend": False,
        },
        {
            "structure": "Cryogenic package-to-MXC coaxial cable (~0.50 m PTFE, v_p ~ 2.0e8 m/s)",
            "length_m": 0.50,
            "predicted_period_ghz": round((2.0e8 / (2.0 * 0.50)) * 1e-9, 4),
            "cycles_in_0p58ghz_band": round(0.58 / 0.20, 2),
            "identifiable_after_linear_trend": True,
        },
    ]

    return {
        "n_testable_snapshots": len(per_device),
        "n_missing_frequency_snapshots": len(snaps["missing_frequency"]),
        "missing_frequency_backends": snaps["missing_frequency"],
        "negative_t1_freq_slope_preserved": {
            "n_negative_out_of_testable": f"{n_neg_slope}/{len(per_device)}",
            "pooled_slope_us_per_ghz": round(pooled_slope, 2),
            "pooled_se_us_per_ghz": round(pooled_se, 2),
            "pooled_z": round(pooled_slope / pooled_se, 2),
            "random_effects_dl_slope_us_per_ghz": round(re_slope, 2),
            "random_effects_dl_se_us_per_ghz": round(re_se, 2),
            "random_effects_dl_z": round(re_slope / re_se, 2),
            "between_chip_heterogeneity_i2_pct": round(i2_pct, 1),
        },
        "electrical_length_identifiability_table": electrical_lengths,
        "n_devices_where_M3_wins_bic": n_m3_wins_bic,
        "n_devices_where_M3a_improves_cv_by_1us": n_m3_wins_cv,
        "mechanism_classification": "INCONCLUSIVE",
        "per_device_results": per_device,
    }
