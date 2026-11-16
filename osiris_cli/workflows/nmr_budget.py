"""
Workflow B: NMR Spectral Reconstruction Resource Budgeting & Error Separation
=============================================================================

Implements the bounded resource-budgeted spectral evaluation scaffold aligned
with `arXiv:2609.17102v1` (Quantinuum NMR study: 22 -> 21 projected spins,
N_t = 70 Trotter steps -> 71 time points including t=0, 10 initial states,
2 quadratures -> 1,420 circuits, 25 shots/circuit -> 35,500 attempted shots).

Enforced OSIRIS invariants:
  1. Explicit attempted-shot budget equation:
         B_attempted = (N_t + 1) * N_initial * N_quadratures * S_per_circuit
     (or exact per-circuit sum under adaptive shot reallocation), alongside
     post-selected accepted-shot fractions by circuit and evolution time.
  2. Strict separation of the 5 non-additive diagnostic error sources:
       - model_reduction (22-spin full vs 21-spin projected reference)
       - trotter_discretization (exact projected vs Trotterized projected)
       - finite_time_truncation (long-duration vs truncated noiseless signal)
       - shot_sampling (noiseless expectation vs finite-shot reconstruction)
       - hardware_implementation_noise (matched ideal vs noisy/hardware execution)
  3. Primary endpoint fixed in advance: integrated squared spectral error over
     declared chemical-shift windows [ppm_lo, ppm_hi] with fixed normalization.
  4. Hard guard prohibiting raw Wasserstein distance on signed or complex NMR
     signals without an explicit probability-measure transformation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from osiris_cli.metrics.wasserstein_verified import (
    validate_probability_measure,
)


@dataclass(frozen=True)
class NMRResourceConfig:
    n_trotter_steps: int = 70
    n_initial_states: int = 10
    n_quadratures: int = 2
    shots_per_circuit: Optional[int] = 25
    adaptive_shots_by_circuit: Optional[Tuple[int, ...]] = None

    @property
    def n_time_points(self) -> int:
        return self.n_trotter_steps + 1

    @property
    def n_circuits(self) -> int:
        return self.n_time_points * self.n_initial_states * self.n_quadratures


def compute_attempted_shot_budget(
    cfg: NMRResourceConfig,
    accepted_shots_by_circuit: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    """Compute exact attempted and accepted shot budgets under uniform or adaptive allocation."""
    if cfg.n_trotter_steps < 0 or cfg.n_initial_states <= 0 or cfg.n_quadratures <= 0:
        raise ValueError("NMRResourceConfig counts must be positive (n_trotter_steps >= 0)")

    if cfg.adaptive_shots_by_circuit is not None:
        if len(cfg.adaptive_shots_by_circuit) != cfg.n_circuits:
            raise ValueError(
                f"adaptive_shots_by_circuit length {len(cfg.adaptive_shots_by_circuit)} "
                f"!= n_circuits {cfg.n_circuits}"
            )
        if any(s < 0 for s in cfg.adaptive_shots_by_circuit):
            raise ValueError("Adaptive shot counts must be non-negative")
        attempted_vec = np.asarray(cfg.adaptive_shots_by_circuit, dtype=np.int64)
        allocation_mode = "adaptive_sum"
    else:
        if cfg.shots_per_circuit is None or cfg.shots_per_circuit <= 0:
            raise ValueError("shots_per_circuit must be positive when adaptive_shots_by_circuit is None")
        attempted_vec = np.full(cfg.n_circuits, cfg.shots_per_circuit, dtype=np.int64)
        allocation_mode = "uniform_grid"

    total_attempted = int(np.sum(attempted_vec))
    out: Dict[str, object] = {
        "allocation_mode": allocation_mode,
        "n_time_points": cfg.n_time_points,
        "n_initial_states": cfg.n_initial_states,
        "n_quadratures": cfg.n_quadratures,
        "n_circuits": cfg.n_circuits,
        "attempted_shots_total": total_attempted,
    }

    if accepted_shots_by_circuit is not None:
        acc_vec = np.asarray(accepted_shots_by_circuit, dtype=np.int64)
        if acc_vec.shape != attempted_vec.shape:
            raise ValueError("accepted_shots_by_circuit length must equal n_circuits")
        if np.any(acc_vec < 0) or np.any(acc_vec > attempted_vec):
            raise ValueError("accepted shots must satisfy 0 <= accepted <= attempted per circuit")
        total_accepted = int(np.sum(acc_vec))
        # Reshape (n_time_points, n_initial * n_quadratures) to get acceptance by evolution time
        per_time_att = attempted_vec.reshape(cfg.n_time_points, -1).sum(axis=1)
        per_time_acc = acc_vec.reshape(cfg.n_time_points, -1).sum(axis=1)
        frac_by_time = np.where(per_time_att > 0, per_time_acc / np.maximum(per_time_att, 1), 0.0)
        out["accepted_shots_total"] = total_accepted
        out["overall_acceptance_fraction"] = (
            float(total_accepted / total_attempted) if total_attempted > 0 else 0.0
        )
        out["acceptance_fraction_by_time_index"] = [float(x) for x in frac_by_time]

    return out


def evaluate_nmr_spectral_window_error(
    ppm_grid: Sequence[float],
    reference_spectrum: Sequence[float],
    candidate_spectrum: Sequence[float],
    windows_ppm: Optional[Sequence[Tuple[float, float]]] = None,
) -> Dict[str, float]:
    """
    Compute prespecified integrated squared spectral error (primary endpoint) and
    dominant peak-position error (secondary endpoint) over fixed chemical-shift windows.
    """
    ppm = np.asarray(ppm_grid, dtype=np.float64)
    ref = np.asarray(reference_spectrum, dtype=np.float64)
    cand = np.asarray(candidate_spectrum, dtype=np.float64)
    if np.iscomplexobj(reference_spectrum) or np.iscomplexobj(candidate_spectrum):
        raise ValueError("Pass real absorption-mode spectra; complex FIDs must be phased/transformed first")
    if ppm.ndim != 1 or ppm.size < 2 or ref.shape != ppm.shape or cand.shape != ppm.shape:
        raise ValueError("ppm_grid, reference_spectrum, and candidate_spectrum must be matching 1D arrays (len >= 2)")

    order = np.argsort(ppm)
    ppm, ref, cand = ppm[order], ref[order], cand[order]

    if windows_ppm is None:
        mask = np.ones_like(ppm, dtype=bool)
    else:
        mask = np.zeros_like(ppm, dtype=bool)
        for lo, hi in windows_ppm:
            a, b = (lo, hi) if lo <= hi else (hi, lo)
            mask |= (ppm >= a) & (ppm <= b)
        if not np.any(mask):
            raise ValueError("Declared chemical-shift windows contain zero grid points")

    diff_sq = (cand - ref) ** 2
    # Trapezoidal integration over masked intervals
    ise = float(np.trapezoid(diff_sq * mask, ppm))
    ref_energy = float(np.trapezoid((ref ** 2) * mask, ppm))
    rel_ise = ise / ref_energy if ref_energy > 1e-30 else float("nan")

    ref_peak_ppm = float(ppm[mask][int(np.argmax(ref[mask]))])
    cand_peak_ppm = float(ppm[mask][int(np.argmax(cand[mask]))])
    return {
        "integrated_squared_error": ise,
        "relative_integrated_squared_error": rel_ise,
        "dominant_peak_shift_error_ppm": abs(cand_peak_ppm - ref_peak_ppm),
        "reference_peak_ppm": ref_peak_ppm,
        "candidate_peak_ppm": cand_peak_ppm,
    }


@dataclass(frozen=True)
class NMRErrorDiagnostics:
    """
    Five strictly separated diagnostic comparisons (NOT assumed additive):
      1. model_reduction_ise:         full 22-spin ref vs projected 21-spin exact ref
      2. trotter_discretization_ise:  exact projected ref vs Trotterized projected noiseless
      3. finite_time_truncation_ise:  long-duration noiseless vs truncated-window noiseless
      4. shot_sampling_ise:           truncated noiseless vs finite-shot ideal reconstruction
      5. implementation_noise_ise:    matched ideal vs noisy/hardware reconstruction
    """
    model_reduction_ise: Optional[float] = None
    trotter_discretization_ise: Optional[float] = None
    finite_time_truncation_ise: Optional[float] = None
    shot_sampling_ise: Optional[float] = None
    implementation_noise_ise: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "additive_assumption": False,
            "tiers": {
                "model_reduction_ise": self.model_reduction_ise,
                "trotter_discretization_ise": self.trotter_discretization_ise,
                "finite_time_truncation_ise": self.finite_time_truncation_ise,
                "shot_sampling_ise": self.shot_sampling_ise,
                "implementation_noise_ise": self.implementation_noise_ise,
            },
        }


def transform_spectrum_to_measure(
    spectrum: Sequence[complex | float], mode: str = "power_density"
) -> np.ndarray:
    """
    Explicitly map a signed or complex spectrum into a valid discrete probability
    measure ONLY when requested via a named transformation (`power_density` or
    `clipped_absorption`); `mode='raw'` validates directly and rejects signed/complex data.
    """
    arr = np.asarray(spectrum)
    if mode == "raw":
        return validate_probability_measure(arr, name="raw_spectrum")
    if mode == "power_density":
        mag_sq = np.abs(arr) ** 2
        total = float(np.sum(mag_sq))
        if total <= 0.0:
            raise ValueError("Zero-energy spectrum cannot be normalized to a probability measure")
        return validate_probability_measure(mag_sq / total, name="power_density_spectrum")
    if mode == "clipped_absorption":
        if np.iscomplexobj(arr):
            raise ValueError("clipped_absorption requires a real-phased spectrum")
        pos = np.maximum(np.asarray(arr, dtype=np.float64), 0.0)
        total = float(np.sum(pos))
        if total <= 0.0:
            raise ValueError("Spectrum has no positive absorption mass")
        return validate_probability_measure(pos / total, name="clipped_absorption_spectrum")
    raise ValueError(f"Unknown measure transformation mode: {mode!r}")
