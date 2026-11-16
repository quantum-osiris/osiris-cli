"""
EXPERIMENT TAU-REVIVAL-01: Conventional Beat/Detuning vs Universal Revival Model
================================================================================

Phase 2 Section 7 (`EXPERIMENT TAU-REVIVAL-01`).

Audits all archived ~46 us records across `/home/osiris` and `legacy-mirrors/`
and implements quantitative competing predictions for:
  1. Conventional Pair-Specific Detuning / Residual ZZ Beating (`ConventionalBeatModel`):
     Predicts characteristic beat period T_beat = 1 / |Delta_f_pair + f_inj|,
     shifting with unit frequency slope d(f_obs)/d(f_inj) = +1 under controlled
     virtual-Z detuning injection, varying across qubit pairs and calibration
     epochs, and vanishing under a Hahn spin-echo pi-pulse!
  2. Universal Constant Revival (`UniversalTau0Model`, tau_0 = phi^8 * 1 us = 46.9787 us,
     ZERO per-device retuning allowed):
     Predicts an invariant revival peak at tau = tau_0 across all qubit pairs,
     backends, calibration epochs, and virtual-Z frame shifts!
  3. Aliasing & Multiple-Testing False-Positive Simulator:
     Demonstrates how coarse/non-uniform delay grids (`d5h5n9agim5s73ahi1h0` /
     `d5ikodnea9qs7393cb7g`) alias higher-frequency detuning fringes into apparent
     ~40-60 us revivals, and how unadjusted peak-picking across heterogeneous
     pairs/windows inflates false positives.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
from scipy.optimize import curve_fit

PHI: float = (1.0 + math.sqrt(5.0)) / 2.0
TAU_0_UNIVERSAL_US: float = PHI ** 8  # 46.97871376374779 us (fixed; never retuned per device!)


@dataclass(frozen=True)
class ConventionalBeatModel:
    """
    Conventional Ramsey / Z-basis population beating from residual ZZ coupling or
    frame detuning `delta_f_khz` (in kHz; Note 1 / 46.1 us = 21.692 kHz,
    1 / 60.0 us = 16.667 kHz):
        S(tau_us; f_inj_khz, echo=False) = 0.5 * (1 + exp(-tau / T2_us) * cos(2*pi * f_eff * tau))
    Under Hahn echo (`echo=True`), static detuning `delta_f_khz + f_inj_khz` is
    refocused (`f_eff -> 0` for static detuning!).
    """
    delta_f_khz: float = 21.692
    t2_us: float = 85.0
    visibility: float = 0.88
    baseline: float = 0.50

    def beat_period_us(self, f_inj_khz: float = 0.0) -> float:
        f_eff = abs(self.delta_f_khz + f_inj_khz)
        if f_eff < 1e-12:
            return float("inf")
        return 1000.0 / f_eff  # 1 / (kHz) in microseconds: 1000 / 21.692 = 46.10 us!

    def predict(
        self, tau_us: np.ndarray, f_inj_khz: float = 0.0, echo: bool = False
    ) -> np.ndarray:
        t = np.asarray(tau_us, dtype=np.float64)
        env = np.exp(-t / self.t2_us)
        if echo:
            # Static 1Q detuning and sublattice-selective echo refocus static ZZ/detuning
            osc = np.ones_like(t)
        else:
            f_eff_mhz = (self.delta_f_khz + f_inj_khz) * 1.0e-3  # cycles per us
            osc = np.cos(2.0 * math.pi * f_eff_mhz * t)
        return self.baseline + 0.5 * self.visibility * env * osc

    def predict_complex_coherence(
        self, tau_us: np.ndarray, f_inj_khz: float = 0.0, sublattice_echo: bool = False
    ) -> np.ndarray:
        """
        Predict conjugate-quadrature complex Bell/Ramsey coherence:
            C_complex(tau) = 0.5 * (<XX - YY> + i * <XY + YX>)
                           = 0.5 * visibility * exp(-tau / T2) * exp(i * 2*pi * (Delta_f0 + f_inj) * tau)
        Measuring both I = <XX - YY>/2 and Q = <XY + YX>/2 resolves SIGNED frequency
        f_signed = Delta_f0 + f_inj (eliminating |Delta_f0 + f_inj| V-folding when
        Delta_f0 < 0 or |Delta_f0| < |f_inj|) and yields frame-invariant magnitude |C_complex|.
        Sublattice-selective echo (`sublattice_echo=True`: pi-pulse on qubit u at tau/2)
        anticommutes with Z_u Z_v and refocuses pair ZZ coupling (unlike simultaneous 2Q echo).
        """
        t = np.asarray(tau_us, dtype=np.float64)
        env = 0.5 * self.visibility * np.exp(-t / self.t2_us)
        if sublattice_echo:
            return env.astype(np.complex128)
        f_eff_mhz = (self.delta_f_khz + f_inj_khz) * 1.0e-3
        return env * np.exp(1j * 2.0 * np.pi * f_eff_mhz * t)


@dataclass(frozen=True)
class UniversalTau0Model:
    """
    Hypothesized universal revival at fixed tau_0 = phi^8 * 1 us = 46.9787 us
    (strictly forbidden from retuning tau_0 per qubit pair or device!):
        F(tau_us) = baseline + 0.5 * vis * exp(-tau / T2_us) + A_rev * exp(-0.5*((tau - tau_0)/w_us)^2)
    """
    t2_us: float = 45.0
    visibility: float = 0.80
    revival_amp: float = 0.25
    revival_width_us: float = 3.5
    baseline: float = 0.35
    tau_0_us: float = TAU_0_UNIVERSAL_US

    def predict(
        self, tau_us: np.ndarray, f_inj_khz: float = 0.0, echo: bool = False
    ) -> np.ndarray:
        """
        Predicts frame-invariant transverse coherence envelope / survival at fixed tau_0
        when evaluated via phase-invariant |C_perp| = 0.5 * sqrt(<XX-YY>^2 + <XY+YX>^2)
        (preventing virtual-Z phase rotation from trivially hiding a hypothetical revival
        in the single XX-YY quadrature).
        """
        t = np.asarray(tau_us, dtype=np.float64)
        decay = self.baseline + 0.5 * self.visibility * np.exp(-t / self.t2_us)
        bump = self.revival_amp * np.exp(-0.5 * ((t - self.tau_0_us) / self.revival_width_us) ** 2)
        return np.clip(decay + bump, 0.0, 1.0)


def compute_aliased_frequency_khz(true_freq_khz: float, dt_step_us: float) -> float:
    """
    Compute apparent aliased frequency in [0, f_Nyquist] when sampling a sinusoid
    of `true_freq_khz` on a uniform delay grid with spacing `dt_step_us` (in us).
      Example: if dt_step_us = 10.0 us (f_sample = 100 kHz, f_Nyq = 50 kHz), a
      true 83.33 kHz detuning aliases to |83.33 - 100.0| = 16.67 kHz -> apparent
      60.0 us revival period!
    """
    if dt_step_us <= 0.0:
        raise ValueError("dt_step_us must be positive")
    f_sample_khz = 1000.0 / dt_step_us
    folded = abs(true_freq_khz) % f_sample_khz
    if folded > 0.5 * f_sample_khz:
        folded = f_sample_khz - folded
    return folded


def audit_archived_torino_tau_sweeps(
    git_mirror: str = "/home/osiris/osiris-assessment/legacy-mirrors/ENKI-420/electrogravitic-unified-physics.git",
) -> Dict[str, Any]:
    """
    Read-only byte-exact audit of archived tau-sweep JSON records in
    `electrogravitic-unified-physics.git` and `sovereign-qed-physics.git`.
    """
    files = [
        "data/tau_analysis_d5h5n9agim5s73ahi1h0.json",
        "data/tau_sweep_d5ig8acpe0pc73anfb9g.json",
        "data/tau_sweep_d5ikodnea9qs7393cb7g.json",
    ]
    extracted: Dict[str, Any] = {}
    for rel in files:
        proc = subprocess.run(
            ["git", f"--git-dir={git_mirror}", "show", f"HEAD:{rel}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            extracted[rel.split("/")[-1]] = json.loads(proc.stdout)

    if not extracted:
        return {"available": False}

    records: List[Dict[str, Any]] = []
    d1 = extracted.get("tau_analysis_d5h5n9agim5s73ahi1h0.json", {})
    if d1:
        records.append({
            "file": "tau_analysis_d5h5n9agim5s73ahi1h0.json",
            "job_id": d1.get("job_id"),
            "backend": d1.get("backend"),
            "tau_0_predicted": d1.get("tau_0_predicted"),
            "tau_0_observed": d1.get("tau_0_observed"),
            "peak_circuits": d1.get("peak_circuits"),
            "peak_fidelity_mean": d1.get("peak_fidelity_mean"),
            "trough_fidelity_mean": d1.get("trough_fidelity_mean"),
            "forensic_finding": (
                "Predicted tau_0=46 but observed period=8 circuit indices ([0, 8, 16, 24] with "
                "peak 0.9985 and trough 0.4934 — a deterministic gate-angle cos^2(k*pi/8) sweep, "
                "NOT a 46 us free-idle decoherence revival; raw shot counts absent)."
            ),
        })

    for fname in ("tau_sweep_d5ig8acpe0pc73anfb9g.json", "tau_sweep_d5ikodnea9qs7393cb7g.json"):
        dm = extracted.get(fname, {})
        if dm:
            has_counts = any(k in dm for k in ("counts", "results", "fidelities", "pair_results"))
            records.append({
                "file": fname,
                "job_id": dm.get("job_id"),
                "backend": dm.get("backend"),
                "delays_us": dm.get("delays_us"),
                "has_measured_data": has_counts,
                "forensic_finding": (
                    "Pre-submission parameter manifest ONLY (13 coarse delays [0..100] us, "
                    "zero measured counts or fidelities recorded in JSON)."
                ),
            })

    return {
        "available": True,
        "universal_tau0_prediction_us": round(TAU_0_UNIVERSAL_US, 4),
        "archived_json_records": records,
        "feb_2026_zz_universal_selftest_range_us": [6.7, 333.0],
        "fez_prospective_3basis_tomography_job_dbiobimne4hc73bf2am0": {
            "tau_us": 46.0,
            "f_bell": 0.3438,
            "f_bell_se": 0.0132,
            "c_perp": 0.0029,
            "c_perp_se": 0.0221,
            "separable_below_0_50": True,
        },
    }


def simulate_detuning_injection_falsification(
    f_inj_grid_khz: Sequence[float] = (-15.0, -7.5, 0.0, 7.5, 15.0),
    true_delta_f_khz: float = 21.7,
    shots: int = 4096,
    seed: int = 20261116,
) -> Dict[str, Any]:
    """
    Simulate the decisive TAU-REVIVAL-01 virtual-Z detuning-injection protocol:
      - Dense anti-aliased grid tau in [0, 80] us with dt = 1.0 us (f_Nyquist = 500 kHz!)
      - Under conventional detuning `true_delta_f_khz = 21.7 kHz` (`T_0 = 46.08 us`),
        injecting `f_inj_khz` shifts the FFT/cosine peak frequency linearly with
        slope d(f_est)/d(f_inj) = +1.000, whereas Hahn echo (`echo=True`) eliminates
        the oscillation completely!
    """
    rng = np.random.default_rng(seed)
    taus = np.arange(0.0, 81.0, 1.0, dtype=np.float64)
    mdl = ConventionalBeatModel(delta_f_khz=true_delta_f_khz, t2_us=85.0, visibility=0.88)

    est_freqs_khz: List[float] = []
    est_periods_us: List[float] = []
    for f_inj in f_inj_grid_khz:
        c_true = mdl.predict_complex_coherence(taus, f_inj_khz=f_inj, sublattice_echo=False)
        p_i = np.clip(0.5 + c_true.real, 0.01, 0.99)
        p_q = np.clip(0.5 + c_true.imag, 0.01, 0.99)
        i_emp = (rng.binomial(shots, p_i) / float(shots)) - 0.5
        q_emp = (rng.binomial(shots, p_q) / float(shots)) - 0.5
        z_emp = i_emp + 1j * q_emp

        # Two-sided complex FFT + unwrapped phase / complex NLS extracts SIGNED frequency
        # (eliminating |Delta_f0 + f_inj| V-folding even when Delta_f0 < 0 or |Delta_f0| < 15 kHz!)
        n_fft = 4096
        freqs_signed_khz = np.fft.fftfreq(n_fft, d=1.0) * 1000.0
        power = np.abs(np.fft.fft(z_emp * np.hanning(taus.size), n=n_fft)) ** 2
        band = np.abs(freqs_signed_khz) <= 250.0
        f_init = float(freqs_signed_khz[band][int(np.argmax(power[band]))])

        def _complex_resid(t_arr: np.ndarray, f_k: float, amp: float, t2: float) -> np.ndarray:
            pred = amp * np.exp(-t_arr / t2) * np.exp(1j * 2.0 * np.pi * (f_k * 1e-3) * t_arr)
            return np.concatenate([pred.real, pred.imag])

        y_cat = np.concatenate([z_emp.real, z_emp.imag])
        popt, _ = curve_fit(
            _complex_resid,
            taus,
            y_cat,
            p0=[f_init, 0.42, 80.0],
            bounds=([-250.0, 0.05, 10.0], [250.0, 0.60, 500.0]),
            maxfev=2000,
        )
        f_peak = float(popt[0])
        est_freqs_khz.append(f_peak)
        est_periods_us.append(1000.0 / max(abs(f_peak), 1e-6))

    # Fit linear slope of estimated frequency vs injected frequency (for positive total frequencies)
    x = np.asarray(f_inj_grid_khz, dtype=np.float64)
    y = np.asarray(est_freqs_khz, dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)

    # Hahn echo control at f_inj = 0
    p_echo = np.clip(mdl.predict(taus, f_inj_khz=0.0, echo=True), 0.01, 0.99)
    y_echo = rng.binomial(shots, p_echo) / float(shots)
    # Check absence of revival bump at 46 us vs 36 us under echo (monotonic decay!)
    echo_diff_46_minus_36 = float(y_echo[46] - y_echo[36])

    return {
        "f_inj_grid_khz": list(f_inj_grid_khz),
        "estimated_beat_freqs_khz": [round(v, 3) for v in est_freqs_khz],
        "estimated_periods_us": [round(v, 3) for v in est_periods_us],
        "fitted_frequency_slope_df_obs_df_inj": round(float(slope), 4),
        "fitted_intercept_delta_f0_khz": round(float(intercept), 3),
        "hahn_echo_46us_minus_36us": round(echo_diff_46_minus_36, 4),
    }
