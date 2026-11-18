"""CORRELATED-NOISE-01 Stage 1: do the white-floor bounds drive A-3's model selection on the hardware data?

Diagnostic only. It is not preregistered and is not part of the locked analysis or of amendment A-3; it
was written after the hardware data existed. A-3 gave H0_1f and H3_local the bounds of the Phase 2 H3
model, which cap each qubit's white dephasing floor at 40 kHz; H0, H1, H2 and Apparatus allow 60 kHz.
On the hardware training couplers the fitted v-qubit floor is about 57 kHz, so the three capped models
fit far worse than H0, although each of them contains H0. This rebuilds A-3's exact training-fit inputs
and refits every model with the caps raised, to show whether any A-3 decision depends on the cap.

    python -m phase3.cn01_a3_bound_diagnostic   # writes docs/phase3/correlated_noise_01_A3_bound_diagnostic.json
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Sequence

import numpy as np
from scipy.optimize import least_squares

from phase2.correlated_noise_01.simulate_correlated_noise import build_block_covariance_matrix
from phase3 import cn01_amendment_a3 as a3
from phase3.execute_and_analyze_live import cn01_fidelity_arrays_from_counts
from phase3.live_hardware_runner import OUT_DIR

WHITE_FLOOR_INDEX = (2, 3)  # (u, v) white dephasing floor in Hz, in every model's parameter vector
CAPS_HZ = (6e4, 2e5)


def training_fit_inputs(arrays: Dict[str, Any], train_slots: Sequence[int]):
    """The suite, mean curve and GLS covariance that `a3_inference` fits, built the same way."""
    suite = list(arrays["suite"])
    f_raw = np.asarray(arrays["f_raw_arr"], dtype=np.float64)
    n_blocks = f_raw.shape[1]
    fit_idx = [i for i, c in enumerate(suite) if not c.condition_id.startswith(a3.EXCLUDED_FROM_FIT_PREFIX)]
    fit_suite = [suite[i] for i in fit_idx]
    curves = np.mean(f_raw[sorted(train_slots)][:, :, fit_idx], axis=0)
    y = np.mean(curves, axis=0)
    sigma = float(max(np.median(np.std(curves, axis=0, ddof=1) / math.sqrt(n_blocks)), 0.0015))
    cov = (0.5 * np.cov(curves, rowvar=False, ddof=1) / float(n_blocks)
           + 0.5 * build_block_covariance_matrix(fit_suite, sigma) + np.eye(len(fit_idx)) * 1e-7)
    return fit_suite, y, cov


def _fit(key: str, starts: List[np.ndarray], lb: np.ndarray, ub: np.ndarray, suite, y, chol) -> Dict[str, Any]:
    def resid(p: np.ndarray) -> np.ndarray:
        return np.linalg.solve(chol, a3.instantiate_a3(key, p).predict_suite(suite) - y)

    best = None
    for s in starts:
        x0 = np.clip(s, lb, ub)
        opt = least_squares(resid, x0=x0, bounds=(lb, ub), x_scale=np.maximum(np.abs(x0), 1e-6),
                            max_nfev=5000, ftol=1e-10, xtol=1e-10)
        if best is None or opt.cost < best.cost:
            best = opt
    assert best is not None
    chi2 = float(np.sum(best.fun ** 2))
    return {"chi2": chi2, "bic": chi2 + len(lb) * math.log(len(y)), "params": [float(v) for v in best.x]}


def bound_diagnostic(arrays: Dict[str, Any], train_slots: Sequence[int]) -> Dict[str, Any]:
    suite, y, cov = training_fit_inputs(arrays, train_slots)
    chol = np.linalg.cholesky(cov)
    reference = a3.fit_models_a3(suite, y, cov)
    h0 = np.asarray(reference["fits"]["H0"]["params"])

    def h0_start(key: str) -> np.ndarray:
        """The A-3 H0 solution mapped onto `key`'s parameters, with any extra noise amplitude off."""
        s = np.array(a3.A3_BOUNDS[key][0], dtype=np.float64)
        s[[0, 1, 2, 3, -1]] = h0[[0, 1, 2, 3, 4]]
        if key in ("H0_1f", "H3_local", "H3"):
            s[4] = 0.0
        return s

    lb0, ub0 = (np.array(v, dtype=np.float64) for v in a3.A3_BOUNDS["H0"][1:])
    ub_40k = ub0.copy()
    ub_40k[list(WHITE_FLOOR_INDEX)] = 4e4
    out: Dict[str, Any] = {
        "train_slots": sorted(int(s) for s in train_slots),
        "n_fit_conditions": len(y),
        "a3_fits": {k: {"chi2": v["chi2"], "bic": v["bic"]} for k, v in reference["fits"].items()},
        "h0_with_40khz_cap": _fit("H0", [h0], lb0, ub_40k, suite, y, chol),
        "refits": {},
    }
    for cap in CAPS_HZ:
        fits = {}
        for key in a3.ALL_KEYS:
            lb, ub = (np.array(v, dtype=np.float64) for v in a3.A3_BOUNDS[key][1:])
            ub[list(WHITE_FLOOR_INDEX)] = np.maximum(ub[list(WHITE_FLOOR_INDEX)], cap)
            starts = [np.asarray(reference["fits"][key]["params"]), h0_start(key), np.array(a3.A3_BOUNDS[key][0])]
            fits[key] = _fit(key, starts, lb, ub, suite, y, chol)
        local = [k for k in a3.ALL_KEYS if k in a3.LOCAL_KEYS]
        best_local = min(local, key=lambda k: fits[k]["bic"])
        best_all = min(a3.ALL_KEYS, key=lambda k: fits[k]["bic"])
        delta = fits[best_local]["bic"] - fits[best_all]["bic"]
        selected = best_all if (best_all in a3.LOCAL_KEYS or delta >= a3.DELTA_BIC_GATE) else best_local
        out["refits"][f"white_floor_cap_{int(cap)}_hz"] = {
            "fits": fits,
            "best_local_model": best_local,
            "raw_min_bic_model": best_all,
            "delta_bic_over_best_local": float(delta),
            "selected_model": selected,
        }
    out["a3_selection"] = reference["selected_model"]
    out["selection_unchanged_by_cap"] = all(r["selected_model"] == reference["selected_model"] for r in out["refits"].values())
    return out


def main() -> None:
    lock = json.loads((OUT_DIR / "correlated_noise_01_live_prereg_lock.json").read_text())
    raw = json.loads((OUT_DIR / "correlated_noise_01_live_raw_counts.json").read_text())
    amendment = json.loads((OUT_DIR / "correlated_noise_01_A3_amendment.json").read_text())
    result = {
        "status": "Diagnostic only: not preregistered, not part of the locked analysis or of A-3; written after the data existed.",
        "job_id": raw["job_id"],
        **bound_diagnostic(cn01_fidelity_arrays_from_counts(lock, raw["pub_counts"]), amendment["train_slots"]),
    }
    path = OUT_DIR / "correlated_noise_01_A3_bound_diagnostic.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    for name, r in result["refits"].items():
        print(name, "selected", r["selected_model"], "delta over best local", round(r["delta_bic_over_best_local"], 2))
    print("selection unchanged by cap:", result["selection_unchanged_by_cap"], "->", path.name)


if __name__ == "__main__":
    main()
