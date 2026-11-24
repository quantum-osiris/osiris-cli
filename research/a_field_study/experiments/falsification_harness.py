#!/usr/bin/env python3
"""Pre-Registered Simulation & Detector Injection-Recovery Harness for A-Field Protocols.

Simulates detector response for the pre-registered experimental protocols in `experiment_spec.json`:
  - EXP-AFIELD-01: Microwave Cavity QED Transmission (Nb 3D TE011 Cavity, sigma = 0.5 Hz).
  - EXP-AFIELD-02: Mesoscopic Aharonov-Bohm Fractional Fluxoid Amplitude (sigma = 0.005).

Epistemic & Governance Discipline (`phase1/errata_qbyte_lambda_phi.md` item #12, `PHY-9`):
  - Every output is explicitly tagged `execution_tier = "SIMULATED"`, `provider_job_id = None`,
    and `physical_claim_assessment = "UNVERIFIED"`.
  - A simulated run with `injected_signal == 0.0` falling within the 3-sigma noise floor is a
    null detector-calibration control (`BELOW_THRESHOLD_WITH_ZERO_INJECTED_SIGNAL`), NEVER an
    empirical refutation of a physical hypothesis.
  - `run_injection_recovery_study` characterizes false-positive rate alpha at `0 sigma`,
    sub-threshold response (`1 sigma`, `2 sigma`), near-threshold power (`3 sigma`, ~50%),
    and high-SNR recovery (`5 sigma`, `6 sigma`, >99%), plus amplitude bias and RMSE.
"""

from __future__ import annotations

import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_INJECTION_TRIALS = 4000
DEFAULT_INJECTION_SEED = 20261124
DEFAULT_SIGMA_MULTIPLIERS: Tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 5.0, 6.0)


def _normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _analytic_two_sided_3sigma_power(sigma_mult: float) -> float:
    """Exact P(|mu + Z| > 3) for Z ~ N(0, 1) and mu = sigma_mult."""
    return (1.0 - _normal_cdf(3.0 - sigma_mult)) + _normal_cdf(-3.0 - sigma_mult)


def _classify_simulation_outcome(injected_signal: float, below_threshold: bool) -> str:
    if injected_signal == 0.0:
        if below_threshold:
            return "BELOW_THRESHOLD_WITH_ZERO_INJECTED_SIGNAL"
        return "FALSE_POSITIVE_ABOVE_THRESHOLD_AT_ZERO_SIGNAL"
    if below_threshold:
        return "BELOW_THRESHOLD_WITH_NONZERO_INJECTED_SIGNAL"
    return "INJECTED_SIGNAL_DETECTED_ABOVE_THRESHOLD"


def run_cavity_experiment_sim(
    spec: Dict[str, Any],
    seed: int = 42,
    injected_signal_hz: float = 0.0,
) -> Dict[str, Any]:
    """Simulate cryogenic Nb 3D cavity frequency-pull readout with optional injected shift."""
    rng = random.Random(seed)
    sigma_hz = float(spec["measurement_uncertainty"]["target_sigma_Hz"])
    f0 = float(spec["hardware_substrate"]["fundamental_frequency_GHz"]) * 1e9

    noise_hz = rng.gauss(0.0, sigma_hz)
    measured_shift_hz = float(injected_signal_hz) + noise_hz
    measured_relative_shift = measured_shift_hz / f0

    threshold_hz = 3.0 * sigma_hz
    below_threshold = abs(measured_shift_hz) <= threshold_hz
    outcome = _classify_simulation_outcome(float(injected_signal_hz), below_threshold)

    return {
        "experiment_id": spec["experiment_id"],
        "execution_tier": "SIMULATED",
        "provider_job_id": None,
        "physical_claim_assessment": "UNVERIFIED",
        "hardware_profile": "Niobium 3D TE011 Cavity (f0=5.0 GHz, Q=2e8, T=20 mK)",
        "injected_signal_Hz": float(injected_signal_hz),
        "measured_observable_Hz": measured_shift_hz,
        "recovered_amplitude_Hz": measured_shift_hz,
        "measured_relative_shift": measured_relative_shift,
        "measurement_sigma_Hz": sigma_hz,
        "detection_threshold_3sigma_Hz": threshold_hz,
        "is_below_threshold": below_threshold,
        "simulation_outcome": outcome,
        "calibration_note": (
            f"Simulated cavity readout ({measured_shift_hz:+.3f} Hz vs 3-sigma threshold "
            f"{threshold_hz:.3f} Hz; injected_signal_Hz={injected_signal_hz:+.3f}). "
            "Zero-signal simulation verifies baseline detector noise behavior only; "
            "physical claim status remains UNVERIFIED pending physical hardware measurement."
        ),
    }


def run_fluxoid_experiment_sim(
    spec: Dict[str, Any],
    seed: int = 42,
    injected_signal: float = 0.0,
) -> Dict[str, Any]:
    """Simulate hybrid Al/Cu ring magnetoconductance Fourier amplitude at 4/Phi_0."""
    rng = random.Random(seed)
    sigma = float(spec["measurement_uncertainty"]["target_sigma"])

    measured_power = abs(float(injected_signal) + rng.gauss(0.0, sigma))
    threshold = 3.0 * sigma
    below_threshold = measured_power <= threshold
    outcome = _classify_simulation_outcome(float(injected_signal), below_threshold)

    return {
        "experiment_id": spec["experiment_id"],
        "execution_tier": "SIMULATED",
        "provider_job_id": None,
        "physical_claim_assessment": "UNVERIFIED",
        "hardware_profile": "Bimetallic Al/Cu Ring (Diameter 1.5 um, T=20 mK)",
        "injected_signal": float(injected_signal),
        "fractional_power_measured": measured_power,
        "recovered_amplitude": measured_power,
        "measurement_sigma": sigma,
        "noise_floor_3sigma": threshold,
        "is_below_threshold": below_threshold,
        "simulation_outcome": outcome,
        "calibration_note": (
            f"Simulated Fourier magnitude at 4/Phi_0 ({measured_power:.5f} vs 3-sigma threshold "
            f"{threshold:.5f}; injected_signal={injected_signal:.5f}). "
            "Zero-signal simulation is a null detector-calibration control; "
            "physical claim status remains UNVERIFIED."
        ),
    }


def _load_spec_data() -> Dict[str, Any]:
    spec_path = os.path.join(os.path.dirname(__file__), "experiment_spec.json")
    with open(spec_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_injection_recovery_study(
    spec_data: Optional[Dict[str, Any]] = None,
    n_trials: int = DEFAULT_INJECTION_TRIALS,
    seed: int = DEFAULT_INJECTION_SEED,
    sigma_multipliers: Sequence[float] = DEFAULT_SIGMA_MULTIPLIERS,
) -> Dict[str, Any]:
    """Run seeded Monte Carlo false-positive and signal injection-recovery curves.

    Evaluates `n_trials >= 2,000` trials per amplitude level across both `EXP-AFIELD-01`
    and `EXP-AFIELD-02`, quantifying false-positive rate alpha at `0 sigma`, detection
    power across `1 sigma .. 6 sigma`, and recovered-amplitude estimation bias & RMSE.
    """
    if n_trials < 2000:
        raise ValueError("PHY-9 injection-recovery study requires n_trials >= 2000")

    data = spec_data if spec_data is not None else _load_spec_data()
    exp1_spec = data["experiments"][0]
    exp2_spec = data["experiments"][1]

    sigma1 = float(exp1_spec["measurement_uncertainty"]["target_sigma_Hz"])
    thresh1 = 3.0 * sigma1
    sigma2 = float(exp2_spec["measurement_uncertainty"]["target_sigma"])
    thresh2 = 3.0 * sigma2

    rng = random.Random(seed)

    def _evaluate_protocol(
        exp_id: str,
        unit: str,
        sigma: float,
        threshold: float,
        folded_magnitude: bool,
    ) -> Dict[str, Any]:
        levels: List[Dict[str, Any]] = []
        by_mult: Dict[str, float] = {}
        for mult in sigma_multipliers:
            m_val = float(mult)
            amp = m_val * sigma
            detections = 0
            sum_rec = 0.0
            sum_err = 0.0
            sum_sq_err = 0.0
            for _ in range(n_trials):
                raw = amp + rng.gauss(0.0, sigma)
                obs = abs(raw) if folded_magnitude else raw
                if abs(obs) > threshold:
                    detections += 1
                err = obs - amp
                sum_rec += obs
                sum_err += err
                sum_sq_err += err * err

            det_rate = detections / float(n_trials)
            mean_rec = sum_rec / float(n_trials)
            bias = sum_err / float(n_trials)
            rmse = math.sqrt(sum_sq_err / float(n_trials))
            by_mult[f"{m_val:.1f}sigma"] = det_rate
            row: Dict[str, Any] = {
                "injected_sigma_multiple": m_val,
                "injected_amplitude": round(amp, 8),
                "unit": unit,
                "n_trials": n_trials,
                "detections_above_3sigma": detections,
                "detection_rate": round(det_rate, 6),
                "analytical_gaussian_3sigma_rate": round(_analytic_two_sided_3sigma_power(m_val), 6),
                "mean_recovered_amplitude": round(mean_rec, 8),
                "estimation_bias": round(bias, 8),
                "estimation_rmse": round(rmse, 8),
            }
            if m_val == 0.0:
                row["false_positive_rate_alpha"] = round(det_rate, 6)
            levels.append(row)

        rates = [lv["detection_rate"] for lv in levels]
        monotone = all(rates[i] < rates[i + 1] for i in range(len(rates) - 1))
        return {
            "experiment_id": exp_id,
            "execution_tier": "SIMULATED",
            "provider_job_id": None,
            "physical_claim_assessment": "UNVERIFIED",
            "estimator_type": "folded_magnitude" if folded_magnitude else "signed_linear",
            "measurement_sigma": sigma,
            "threshold_3sigma": threshold,
            "unit": unit,
            "n_trials_per_level": n_trials,
            "false_positive_rate_0sigma": round(by_mult.get("0.0sigma", float("nan")), 6),
            "power_at_1sigma": round(by_mult.get("1.0sigma", float("nan")), 6),
            "power_at_2sigma": round(by_mult.get("2.0sigma", float("nan")), 6),
            "power_at_3sigma": round(by_mult.get("3.0sigma", float("nan")), 6),
            "power_at_5sigma": round(by_mult.get("5.0sigma", float("nan")), 6),
            "power_at_6sigma": round(by_mult.get("6.0sigma", float("nan")), 6),
            "monotone_power_curve_verified": monotone,
            "levels": levels,
        }

    exp1_report = _evaluate_protocol(
        exp_id=exp1_spec["experiment_id"],
        unit="Hz",
        sigma=sigma1,
        threshold=thresh1,
        folded_magnitude=False,
    )
    exp2_report = _evaluate_protocol(
        exp_id=exp2_spec["experiment_id"],
        unit="dimensionless_fourier_amplitude",
        sigma=sigma2,
        threshold=thresh2,
        folded_magnitude=True,
    )

    return {
        "execution_tier": "SIMULATED",
        "provider_job_id": None,
        "physical_claim_assessment": "UNVERIFIED",
        "seed": seed,
        "n_trials_per_amplitude": n_trials,
        "sigma_multipliers": [float(m) for m in sigma_multipliers],
        "protocols": {
            exp1_spec["experiment_id"]: exp1_report,
            exp2_spec["experiment_id"]: exp2_report,
        },
    }


def execute_all_experiments(
    seed: int = 42,
    injection_trials: int = DEFAULT_INJECTION_TRIALS,
    injection_seed: int = DEFAULT_INJECTION_SEED,
) -> Dict[str, Any]:
    spec_data = _load_spec_data()
    exp1_spec = spec_data["experiments"][0]
    exp2_spec = spec_data["experiments"][1]

    res1 = run_cavity_experiment_sim(exp1_spec, seed=seed, injected_signal_hz=0.0)
    res2 = run_fluxoid_experiment_sim(exp2_spec, seed=seed, injected_signal=0.0)
    inj_report = run_injection_recovery_study(
        spec_data=spec_data,
        n_trials=injection_trials,
        seed=injection_seed,
    )

    zero_signal_controls_below = (
        res1["simulation_outcome"] == "BELOW_THRESHOLD_WITH_ZERO_INJECTED_SIGNAL"
        and res2["simulation_outcome"] == "BELOW_THRESHOLD_WITH_ZERO_INJECTED_SIGNAL"
    )

    return {
        "organism_id": spec_data["organism_id"],
        "execution_tier": "SIMULATED",
        "provider_job_id": None,
        "physical_claim_assessment": "UNVERIFIED",
        "experiments_executed": [res1, res2],
        "zero_signal_controls_below_threshold": zero_signal_controls_below,
        "all_hypotheses_falsified": False,
        "organism_status_update": "SIMULATION_CALIBRATED_PHYSICAL_CLAIM_UNVERIFIED",
        "governance_note": (
            "Zero-signal null simulations verify baseline detector noise floors and 3-sigma "
            "false-positive calibration only; they are NOT physical measurements and do NOT "
            "empirically confirm or falsify any physical field hypothesis "
            "(physical_claim_assessment: UNVERIFIED). Quantitative detector sensitivity is "
            "characterized in injection_recovery across 0 sigma to 6 sigma."
        ),
        "injection_recovery": inj_report,
    }


if __name__ == "__main__":
    results = execute_all_experiments()
    out_file = os.path.join(os.path.dirname(__file__), "experiment_results.json")
    with open(out_file, "w", encoding="utf-8", newline="\n") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")

    print("=== A-FIELD DETECTOR CALIBRATION & INJECTION-RECOVERY HARNESS (SIMULATED) ===")
    for exp in results["experiments_executed"]:
        print(f"[{exp['experiment_id']}] Hardware Profile: {exp['hardware_profile']}")
        print(f"  Execution Tier:   {exp['execution_tier']}")
        print(f"  Sim Outcome:      {exp['simulation_outcome']}")
        print(f"  Physical Claim:   {exp['physical_claim_assessment']}")
    print("-----------------------------------------------------------------------------")
    print(f"Organism Status: {results['organism_status_update']}")
    print(f"Results File:    {out_file}")
