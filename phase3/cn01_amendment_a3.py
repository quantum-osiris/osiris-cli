"""
CORRELATED-NOISE-01 Stage 1: preregistration amendment A-3 (analysis only; circuits unchanged)
===============================================================================================

Recorded before any CN01 data exist. As locked, the analysis would report "non-Markovian H3 wins"
from ordinary local 1/f dephasing:

  * H0, the only null, is white noise, so it predicts the same decay for Ramsey, echo and DD. Any
    real difference between them (which local 1/f noise produces) favours H3 in the BIC comparison.
  * In the held-out test H0 is calibrated from the Ramsey arm and H3 from the DD32 arm, then both
    are scored on DD16 Bell conditions, so the calibration data decide the winner. (The
    preregistration specifies "high-frequency echo calibration" for every model.)
  * The models place DD pulses at centred, wrapped slots, while the locked circuits start each
    pulse 28 dt into a 128 dt slot and never wrap. This one is immaterial: the pair exposures are
    identical and H3's predictions move by at most 0.01 pp. It is corrected anyway, for exactness.

A-3 changes, with the QPY, the schedule and the circuits untouched:
  1. Local non-white nulls. H0_1f: independent 1/f plus a white floor on each qubit. H3_local: H3
     with spatial_corr = 0. A non-local model (H1, H2, H3, Apparatus) is selected only if it beats
     the best local model (H0, H0_1f, H3_local) by delta-BIC >= 8.
  2. Ramsey arms measure only the X quadrature, so detuning and decay are confounded. They are
     excluded from model selection and calibration and reported descriptively.
  3. Held-out transfer. Each model's local white floor is solved so that the model reproduces the
     same calibration observable: the coupler's own `echo_dd32_8us`. The primary held-out score is
     the change in Phi+ fidelity from 0 ns at the six non-zero stagger offsets: the preregistration's
     "sign and magnitude of dF_Bell(delta)". The absolute levels the locked code scores become
     secondary, because they carry per-coupler calibration error common to every model.
  4. Toggling functions use the hardware pulse placement of the locked circuits.
  5. A seeded random 4/6 train/held-out split replaces the quality-ranked split.
  6. Decisions keep the preregistered requirements per candidate (C_DFS > 0 for H2/H3, C_memory > 0
     for H3, C_spec > 0 for H1, S48 >= 0.75 for Apparatus, Holm alpha = 0.05) and the kill criterion
     (a candidate that wins in-sample but does not beat the null on held-out couplers is falsified),
     with the null being the best local model instead of H0.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy import stats
from scipy.optimize import brentq, least_squares

from phase2.correlated_noise_01.correlated_noise_models import (
    ApparatusPulseCollisionModel,
    H0IndependentLocalModel,
    H1StaticSpectatorZZModel,
    H2CorrelatedMarkovianModel,
    H3NonMarkovianMemoryModel,
    NoiseModelBase,
    ProbeCondition,
)
from phase2.correlated_noise_01.simulate_correlated_noise import (
    MODEL_BOUNDS,
    STAGGER_OFFSETS_NS,
    build_block_covariance_matrix,
)

AMENDMENT_ID = "CN01-A3"
DT_S = 4.0e-9
PULSE_DT = 9
DELTA_BIC_GATE = 8.0
LOCAL_KEYS: Tuple[str, ...] = ("H0", "H0_1f", "H3_local")
NONLOCAL_KEYS: Tuple[str, ...] = ("H1", "H2", "H3", "Apparatus")
ALL_KEYS: Tuple[str, ...] = LOCAL_KEYS + NONLOCAL_KEYS
# Equal variance per decade of correlation time over 0.1 us .. 1 ms: a 1/f spectrum between ~160 Hz and ~1.6 MHz.
TAU_1F_GRID_S: Tuple[float, ...] = tuple(float(t) for t in np.logspace(-7, -3, 9))
CALIBRATION_CONDITION = "echo_dd32_8us"
EXCLUDED_FROM_FIT_PREFIX = "ramsey_"
N_TRAIN = 4


# ---------------------------------------------------------------------------
# 1. Hardware pulse placement (as phase3.live_hardware_runner._append_isa_xy_dd_train builds it)
# ---------------------------------------------------------------------------

def hw_pulse_starts_dt(total_dt: int, n_dd_pulses: int, shift_dt: int = 0) -> List[int]:
    """Start tick of every X/Y pulse in the locked circuits."""
    if n_dd_pulses <= 0:
        return []
    if n_dd_pulses == 1:
        return [(total_dt - PULSE_DT) // 2 + shift_dt]
    slot_dt = total_dt // n_dd_pulses
    base = 28 if slot_dt == 128 else max(4, (slot_dt - PULSE_DT - shift_dt) // 2)
    return [k * slot_dt + base + shift_dt for k in range(n_dd_pulses)]


def hw_pulse_centers_s(total_time_s: float, n_dd_pulses: int, offset_s: float = 0.0) -> np.ndarray:
    """Toggling-frame flip times: the centre of each 9 dt pulse."""
    if n_dd_pulses <= 0 or total_time_s <= 0.0:
        return np.empty(0, dtype=np.float64)
    starts = hw_pulse_starts_dt(int(round(total_time_s / DT_S)), n_dd_pulses, int(round(offset_s / DT_S)))
    return (np.asarray(starts, dtype=np.float64) + 0.5 * PULSE_DT) * DT_S


def _signs(mids: np.ndarray, centers: np.ndarray) -> np.ndarray:
    if centers.size == 0:
        return np.ones_like(mids)
    return np.where(np.sum(mids[:, None] >= centers[None, :], axis=1) % 2 == 0, 1.0, -1.0)


@lru_cache(maxsize=None)
def _hw_intervals_cached(total_time_s: float, n_dd_pulses: int, stagger_offset_s: float) -> Tuple[np.ndarray, ...]:
    c_u = hw_pulse_centers_s(total_time_s, n_dd_pulses, 0.0)
    c_v = hw_pulse_centers_s(total_time_s, n_dd_pulses, stagger_offset_s)
    knots = np.unique(np.concatenate(([0.0, total_time_s], c_u, c_v)))
    keep = np.diff(knots) > 1e-17
    a, b = knots[:-1][keep], knots[1:][keep]
    mids = 0.5 * (a + b)
    out = (a, b, _signs(mids, c_u), _signs(mids, c_v))
    for arr in out:
        arr.setflags(write=False)
    return out


def hw_toggling_intervals(
    total_time_s: float, n_dd_pulses: int, stagger_offset_s: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`(a_starts, b_ends, s_u, s_v)` with s_u, s_v constant on each `[a_k, b_k]` (hardware placement)."""
    return _hw_intervals_cached(float(total_time_s), int(n_dd_pulses), float(stagger_offset_s))  # type: ignore[return-value]


def hw_pair_exposure(total_time_s: float, n_dd_pulses: int, stagger_offset_s: float) -> float:
    if n_dd_pulses <= 0 or total_time_s <= 0.0:
        return 1.0
    a, b, su, sv = hw_toggling_intervals(total_time_s, n_dd_pulses, stagger_offset_s)
    return float(np.sum(su * sv * (b - a)) / total_time_s)


def hw_single_exposure(total_time_s: float, n_dd_pulses: int) -> float:
    if n_dd_pulses <= 0 or total_time_s <= 0.0:
        return 1.0
    a, b, su, _ = hw_toggling_intervals(total_time_s, n_dd_pulses, 0.0)
    return float(np.sum(su * (b - a)) / total_time_s)


def ou_kernel_integrals(
    a: np.ndarray, b: np.ndarray, su: np.ndarray, sv: np.ndarray, tau_c_s: float, f_bath_hz: float = 0.0
) -> Tuple[float, float, float]:
    """The closed form of H3NonMarkovianMemoryModel._exact_piecewise_kernel_integrals, on given intervals."""
    lengths = b - a
    z = complex(1.0 / max(tau_c_s, 1.0e-12), 2.0 * math.pi * f_bath_hz)
    z_l = z * lengths
    w_mat = np.diag(((2.0 / (z * z)) * (z_l + np.expm1(-z_l))).real)
    m = lengths.size
    if m > 1:
        edge = (-np.expm1(-z_l)) / z
        gap = np.maximum(0.0, a[:, None] - b[None, :])
        lower = np.tril(np.ones((m, m), dtype=bool), k=-1)
        exp_gap = np.zeros((m, m), dtype=np.complex128)
        exp_gap[lower] = np.exp(-z * gap[lower])
        off = ((edge[:, None] * edge[None, :]) * exp_gap).real
        w_mat = w_mat + off + off.T
    return max(0.0, float(su @ w_mat @ su)), max(0.0, float(sv @ w_mat @ sv)), float(su @ w_mat @ sv)


@lru_cache(maxsize=None)
def _one_over_f_integrals(total_time_s: float, n_dd_pulses: int, stagger_offset_s: float) -> Tuple[float, float]:
    a, b, su, sv = hw_toggling_intervals(total_time_s, n_dd_pulses, stagger_offset_s)
    iuu = ivv = 0.0
    for tau in TAU_1F_GRID_S:
        x, y, _ = ou_kernel_integrals(a, b, su, sv, tau)
        iuu += x
        ivv += y
    w = 1.0 / len(TAU_1F_GRID_S)
    return iuu * w, ivv * w


# ---------------------------------------------------------------------------
# 2. Models with hardware pulse timing, and the two local non-white nulls
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class H3HardwareTiming(H3NonMarkovianMemoryModel):
    """H3 (correlated OU memory); with spatial_corr = 0 it is the H3_local null."""
    name: str = "H3_NonMarkovianMemory_hw"

    def _exact_piecewise_kernel_integrals(
        self, total_time_s: float, n_dd_pulses: int, stagger_offset_s: float
    ) -> Tuple[float, float, float]:
        a, b, su, sv = hw_toggling_intervals(total_time_s, n_dd_pulses, stagger_offset_s)
        return ou_kernel_integrals(a, b, su, sv, self.tau_c_s, self.f_bath_hz)


@dataclass(frozen=True)
class H0Local1fHardwareTiming(H3NonMarkovianMemoryModel):
    """H0_1f: independent local 1/f dephasing plus a white floor on each qubit; no spatial correlation."""
    amp_1f_u_rad_s: float = 40_000.0
    amp_1f_v_rad_s: float = 40_000.0
    name: str = "H0_Local1f_hw"
    n_free_params: int = 6

    def _cumulants(self, cond: ProbeCondition) -> Tuple[float, float, float]:
        T = cond.total_time_s
        if T <= 0.0:
            return 0.0, 0.0, 0.0
        iuu, ivv = _one_over_f_integrals(float(T), int(cond.n_dd_pulses), float(cond.stagger_offset_s))
        chi_u = self.gamma_white_u_hz * T + 0.5 * (self.amp_1f_u_rad_s ** 2) * iuu
        chi_v = self.gamma_white_v_hz * T + 0.5 * (self.amp_1f_v_rad_s ** 2) * ivv
        return chi_u, chi_v, 0.0


@dataclass(frozen=True)
class H2HardwareTiming(H2CorrelatedMarkovianModel):
    name: str = "H2_CorrelatedMarkovian_hw"

    def predict(self, cond: ProbeCondition) -> float:
        if cond.observable == "ramsey_echo_u":
            return self.base.predict(cond)
        T = cond.total_time_s
        alpha_uv = hw_pair_exposure(T, cond.n_dd_pulses, cond.stagger_offset_s)
        g_cov = self._gamma_cov(cond.graph_distance)
        chi_base = (self.base.gamma_phi_u_hz + self.base.gamma_phi_v_hz) * T
        if cond.observable in ("bell_phi_plus", "bell_psi_plus"):
            is_phi = cond.observable == "bell_phi_plus"
            chi = max(0.0, chi_base + (1.0 if is_phi else -1.0) * 2.0 * alpha_uv * g_cov * T)
            return self.base._bell_fidelity_from_dephasing_exponent(
                T, chi, is_phi, cond.n_dd_pulses, cond.graph_distance
            )
        if cond.observable == "connected_zz_corr":
            return 2.0 * alpha_uv * g_cov * T
        raise ValueError(f"Unknown observable: {cond.observable!r}")


@dataclass(frozen=True)
class H1HardwareTiming(H1StaticSpectatorZZModel):
    name: str = "H1_StaticSpectatorZZ_hw"

    def predict(self, cond: ProbeCondition) -> float:
        T = cond.total_time_s
        zz_hz = (
            self.f_zz_spectator_hz
            if cond.graph_distance == 1
            else self.f_zz_spectator_hz * (0.08 ** (cond.graph_distance - 1))
        )
        omega_zz = 2.0 * math.pi * zz_hz
        p_vis = 1.0 if cond.spectator_superposition else self.passive_spectator_visibility
        base = self.base
        if cond.observable == "ramsey_echo_u":
            alpha_1q = hw_single_exposure(T, cond.n_dd_pulses)
            coh_zz_1q = (1.0 - p_vis) + p_vis * math.cos(omega_zz * T * alpha_1q)
            coh_u = math.exp(-T * (0.5 / base.t1_u_s + base.gamma_phi_u_hz)) * coh_zz_1q
            return 0.5 * (1.0 + (2.0 * base.f0_spam - 1.0) * coh_u)
        if self.spectator_pulsed_with_v and cond.n_dd_pulses > 0:
            alpha_eff = hw_pair_exposure(T, cond.n_dd_pulses, cond.stagger_offset_s)
        else:
            alpha_eff = hw_single_exposure(T, cond.n_dd_pulses)
        coh_zz = (1.0 - p_vis) + p_vis * math.cos(omega_zz * T * alpha_eff)
        if cond.observable in ("bell_phi_plus", "bell_psi_plus"):
            is_phi = cond.observable == "bell_phi_plus"
            e1_u = math.exp(-T / base.t1_u_s)
            e1_v = math.exp(-T / base.t1_v_s)
            if cond.n_dd_pulses >= 1:
                pop_term = 0.5 * (1.0 + e1_u * e1_v)
            else:
                pop_term = 0.5 * (2.0 - e1_u - e1_v + 2.0 * e1_u * e1_v) if is_phi else 0.5 * (e1_u + e1_v)
            chi_local = (base.gamma_phi_u_hz + base.gamma_phi_v_hz) * T
            coh = math.exp(-0.5 * T * (1.0 / base.t1_u_s + 1.0 / base.t1_v_s) - chi_local) * coh_zz
            f_raw = 0.5 * (pop_term + coh)
            alpha_spam = (4.0 * base._effective_spam_ceiling(cond.graph_distance) - 1.0) / 3.0
            return 0.25 + 0.75 * alpha_spam * ((4.0 * f_raw - 1.0) / 3.0)
        if cond.observable == "connected_zz_corr":
            return 0.0
        raise ValueError(f"Unknown observable: {cond.observable!r}")


def _h0(t1u: float, t1v: float, gpu: float, gpv: float, f0: float) -> H0IndependentLocalModel:
    return H0IndependentLocalModel(t1_u_s=t1u, t1_v_s=t1v, gamma_phi_u_hz=gpu, gamma_phi_v_hz=gpv, f0_spam=f0)


def instantiate_a3(key: str, theta: Sequence[float]) -> NoiseModelBase:
    th = [float(x) for x in theta]
    if key == "H0":
        return _h0(*th)
    if key == "H0_1f":
        t1u, t1v, gwu, gwv, amp, f0 = th
        return H0Local1fHardwareTiming(
            t1_u_s=t1u, t1_v_s=t1v, gamma_white_u_hz=gwu, gamma_white_v_hz=gwv,
            sigma_ou_u_rad_s=0.0, sigma_ou_v_rad_s=0.0, spatial_corr=0.0, f0_spam=f0,
            amp_1f_u_rad_s=amp, amp_1f_v_rad_s=amp,
        )
    if key in ("H3", "H3_local"):
        if key == "H3":
            t1u, t1v, gwu, gwv, sou, tauc, scorr, f0 = th
        else:
            t1u, t1v, gwu, gwv, sou, tauc, f0 = th
            scorr = 0.0
        return H3HardwareTiming(
            t1_u_s=t1u, t1_v_s=t1v, gamma_white_u_hz=gwu, gamma_white_v_hz=gwv,
            sigma_ou_u_rad_s=sou, sigma_ou_v_rad_s=sou, tau_c_s=tauc, spatial_corr=scorr, f0_spam=f0,
        )
    if key == "H1":
        t1u, t1v, gpu, gpv, f0, fzz = th
        return H1HardwareTiming(base=_h0(t1u, t1v, gpu, gpv, f0), f_zz_spectator_hz=fzz)
    if key == "H2":
        t1u, t1v, gpu, gpv, f0, rho, xi = th
        return H2HardwareTiming(
            base=_h0(t1u, t1v, gpu, gpv, f0), rho_spatial=rho, spatial_length_hops=xi, enforce_lattice_psd=True
        )
    if key == "Apparatus":
        t1u, t1v, gpu, gpv, f0, pcol = th
        return ApparatusPulseCollisionModel(base=_h0(t1u, t1v, gpu, gpv, f0), per_collision_error=pcol)
    raise ValueError(f"Unknown model key: {key!r}")


_H3_X0, _H3_LB, _H3_UB = MODEL_BOUNDS["H3"]
A3_BOUNDS: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {
    **{k: MODEL_BOUNDS[k] for k in ("H0", "H1", "H2", "H3", "Apparatus")},
    "H0_1f": (
        np.array([170e-6, 170e-6, 1000.0, 1000.0, 40_000.0, 0.97]),
        np.array([30e-6, 30e-6, 0.0, 0.0, 0.0, 0.85]),
        np.array([500e-6, 500e-6, 40_000.0, 40_000.0, 400_000.0, 1.00]),
    ),
    "H3_local": (np.delete(_H3_X0, 6), np.delete(_H3_LB, 6), np.delete(_H3_UB, 6)),
}
_TAU_INDEX = {"H3": 5, "H3_local": 5}
_TAU_STARTS_S = (0.5e-6, 8.0e-6)


def fit_models_a3(
    suite: Sequence[ProbeCondition],
    y_obs: np.ndarray,
    cov_matrix: np.ndarray,
    keys: Sequence[str] = ALL_KEYS,
    delta_bic_gate: float = DELTA_BIC_GATE,
) -> Dict[str, Any]:
    """GLS fit of every model. A non-local model wins only by beating the best local model by the gate."""
    n_pts = len(suite)
    chol = np.linalg.cholesky(cov_matrix)
    fits: Dict[str, Dict[str, Any]] = {}
    objs: Dict[str, NoiseModelBase] = {}
    for key in keys:
        x0, lb, ub = A3_BOUNDS[key]
        starts = [x0.copy()]
        for tau in _TAU_STARTS_S if key in _TAU_INDEX else ():
            s = x0.copy()
            s[_TAU_INDEX[key]] = tau
            starts.append(s)

        def resid(p: np.ndarray, key: str = key) -> np.ndarray:
            return np.linalg.solve(chol, instantiate_a3(key, p).predict_suite(suite) - y_obs)

        best = None
        for s in starts:
            opt = least_squares(resid, x0=s, bounds=(lb, ub), x_scale=np.maximum(np.abs(x0), 1e-6),
                                max_nfev=250, ftol=1e-7, xtol=1e-7)
            if best is None or opt.cost < best.cost:
                best = opt
        assert best is not None
        chi2 = float(np.sum(best.fun ** 2))
        obj = instantiate_a3(key, best.x)
        fits[key] = {
            "params": [float(v) for v in best.x],
            "k_params": len(x0),
            "chi2": chi2,
            "bic": chi2 + len(x0) * math.log(float(n_pts)),
            "rmse": float(math.sqrt(np.mean((obj.predict_suite(suite) - y_obs) ** 2))),
        }
        objs[key] = obj
    local = [k for k in keys if k in LOCAL_KEYS]
    best_local = min(local, key=lambda k: fits[k]["bic"])
    best_all = min(keys, key=lambda k: fits[k]["bic"])
    delta = fits[best_local]["bic"] - fits[best_all]["bic"]
    selected = best_all if (best_all in LOCAL_KEYS or delta >= delta_bic_gate) else best_local
    return {
        "selected_model": selected,
        "best_local_model": best_local,
        "raw_min_bic_model": best_all,
        "delta_bic_over_best_local": float(delta),
        "fits": fits,
        "fitted_objects": objs,
    }


# ---------------------------------------------------------------------------
# 3. Split, held-out transfer and the full amended inference
# ---------------------------------------------------------------------------

def a3_random_split(qpy_sha256: str, n_couplers: int = 10, n_train: int = N_TRAIN) -> List[int]:
    """Training slots: the first `n_train` of a permutation seeded from the locked QPY's SHA-256."""
    seed = int(hashlib.sha256(f"{AMENDMENT_ID}:split:{qpy_sha256}".encode()).hexdigest()[:16], 16)
    perm = np.random.default_rng(seed).permutation(n_couplers)
    return sorted(int(i) for i in perm[:n_train])


def transfer_predict(
    key: str,
    trained: NoiseModelBase,
    t1_u_s: float,
    t1_v_s: float,
    f0_local: float,
    y_cal: float,
    cal_cond: ProbeCondition,
    target_conds: Sequence[ProbeCondition],
) -> Tuple[np.ndarray, float]:
    """Predict a held-out coupler from trained bath parameters, with this coupler's T1 and SPAM ceiling
    and a local white floor solved so that the model reproduces `y_cal` on `cal_cond`."""

    def make(g: float) -> NoiseModelBase:
        if key == "H0":
            return _h0(t1_u_s, t1_v_s, g, g, f0_local)
        if key in ("H0_1f", "H3", "H3_local"):
            return replace(trained, t1_u_s=t1_u_s, t1_v_s=t1_v_s, gamma_white_u_hz=g, gamma_white_v_hz=g,
                           f0_spam=f0_local)
        return replace(trained, base=_h0(t1_u_s, t1_v_s, g, g, f0_local))

    def gap(g: float) -> float:
        return make(g).predict(cal_cond) - y_cal

    g_hi = 2.0e5
    if gap(0.0) <= 0.0:
        g = 0.0
    elif gap(g_hi) > 0.0:
        g = g_hi
    else:
        g = float(brentq(gap, 0.0, g_hi, xtol=1e-3))
    return make(g).predict_suite(target_conds), g


def _cluster_ttest(vec: np.ndarray) -> Dict[str, float]:
    mean_v = float(np.mean(vec))
    se_v = float(np.std(vec, ddof=1) / math.sqrt(vec.size))
    t_val = mean_v / max(se_v, 1e-12)
    return {
        "mean": round(mean_v, 5),
        "se": round(se_v, 5),
        "t_stat": round(t_val, 3),
        "df": int(vec.size - 1),
        "p_two_sided": float(2.0 * stats.t.sf(abs(t_val), df=vec.size - 1)),
    }


def a3_inference(
    arrays: Mapping[str, Any],
    cohort: Sequence[Mapping[str, Any]],
    train_slots: Sequence[int],
    cal_snapshot: Mapping[str, Any],
) -> Dict[str, Any]:
    """Amended inference from per-coupler, per-block raw fidelities (`cn01_fidelity_arrays_from_counts`)."""
    suite: List[ProbeCondition] = list(arrays["suite"])
    cond_ids: List[str] = [c.condition_id for c in suite]
    f_raw = np.asarray(arrays["f_raw_arr"], dtype=np.float64)  # (n_slots, n_blocks, n_conds)
    m_mats = arrays["m_mats"]
    n_slots, n_blocks, _ = f_raw.shape
    train = sorted(int(s) for s in train_slots)
    held_out = [s for s in range(n_slots) if s not in train]

    # Model selection on the training cohort, Ramsey arms excluded; locked GLS covariance construction.
    fit_idx = [i for i, c in enumerate(cond_ids) if not c.startswith(EXCLUDED_FROM_FIT_PREFIX)]
    fit_suite = [suite[i] for i in fit_idx]
    train_curves = np.mean(f_raw[train][:, :, fit_idx], axis=0)  # (n_blocks, n_fit)
    y_train = np.mean(train_curves, axis=0)
    sigma_emp = float(max(np.median(np.std(train_curves, axis=0, ddof=1) / math.sqrt(n_blocks)), 0.0015))
    emp_cov = np.cov(train_curves, rowvar=False, ddof=1) / float(n_blocks)
    cov = 0.5 * emp_cov + 0.5 * build_block_covariance_matrix(fit_suite, sigma_emp) + np.eye(len(fit_idx)) * 1e-7
    gls = fit_models_a3(fit_suite, y_train, cov)

    # The four locked mechanistic contrasts across all couplers, Holm-adjusted (unchanged by A-3).
    means = np.mean(f_raw, axis=1)
    ix = {c: cond_ids.index(c) for c in (
        "phi_dd16_d0ns", "phi_dd16_d256ns", "psi_dd16_d0ns", "psi_dd16_d256ns",
        "phi_spec_plus_d0ns", "phi_spec_plus_d256ns", "echo1_16us", "echo_dd32_16us")}
    phi_step = means[:, ix["phi_dd16_d256ns"]] - means[:, ix["phi_dd16_d0ns"]]
    raw_contrasts = {
        "C_stagger_phi_256_minus_0": phi_step,
        "C_DFS_asymmetry": phi_step - (means[:, ix["psi_dd16_d256ns"]] - means[:, ix["psi_dd16_d0ns"]]),
        "C_spectator_activation": (means[:, ix["phi_spec_plus_d256ns"]] - means[:, ix["phi_spec_plus_d0ns"]]) - phi_step,
        "C_memory_dd32_minus_echo1_16us": means[:, ix["echo_dd32_16us"]] - means[:, ix["echo1_16us"]],
    }
    tests = {k: _cluster_ttest(v) for k, v in raw_contrasts.items()}
    run_max = 0.0
    for rank, name in enumerate(sorted(tests, key=lambda k: tests[k]["p_two_sided"])):
        run_max = max(run_max, min(1.0, tests[name]["p_two_sided"] * (len(tests) - rank)))
        tests[name]["p_holm_adjusted"] = round(run_max, 6)
        tests[name]["significant_holm_0_05"] = run_max < 0.05

    # Locked step-fraction statistic S48 (Apparatus criterion S48 >= 0.75), unchanged.
    step_den = float(np.mean(phi_step))
    step_s48 = (
        float(np.mean(means[:, cond_ids.index("phi_dd16_d48ns")] - means[:, ix["phi_dd16_d0ns"]])) / step_den
        if abs(step_den) > 0.002
        else 0.0
    )

    # Held-out transfer: every model is calibrated from the same observable, then scored on the
    # phi_dd16 stagger curve of each held-out coupler. Primary: the change from 0 ns at the six
    # non-zero offsets, which is the preregistration's target ("the sign and magnitude of
    # dF_Bell(delta)"). Secondary: the absolute fidelities at all seven offsets, as the locked code
    # scores them. Absolute levels carry each coupler's SPAM, T1 and v-qubit calibration error, which
    # is common to all models and, on synthetic data, masks a ~1 pp correlated stagger signal.
    q_cal = cal_snapshot.get("qubits", {})
    offset_idx = [cond_ids.index(f"phi_dd16_d{d}ns") for d in STAGGER_OFFSETS_NS]
    offset_conds = [suite[i] for i in offset_idx]
    cal_cond = suite[cond_ids.index(CALIBRATION_CONDITION)]
    rmse: Dict[str, List[float]] = {k: [] for k in ALL_KEYS}
    rmse_step: Dict[str, List[float]] = {k: [] for k in ALL_KEYS}
    floors: Dict[str, List[float]] = {k: [] for k in ALL_KEYS}
    for s in held_out:
        u, v = cohort[s]["coupler"]
        t1_u = float(q_cal.get(str(u), {}).get("t1_s") or 170e-6)
        t1_v = float(q_cal.get(str(v), {}).get("t1_s") or 170e-6)
        ro_diag = float(np.mean(np.diag(m_mats[s])))
        f0_local = float(np.clip(ro_diag * (1.0 - 2.0 * float(cohort[s]["twoq_error"])), 0.85, 0.995))
        y_cal = float(means[s, cond_ids.index(CALIBRATION_CONDITION)])
        obs = means[s, offset_idx]
        for k in ALL_KEYS:
            pred, g = transfer_predict(k, gls["fitted_objects"][k], t1_u, t1_v, f0_local, y_cal, cal_cond, offset_conds)
            rmse[k].append(float(math.sqrt(np.mean((pred - obs) ** 2))))
            rmse_step[k].append(float(math.sqrt(np.mean(((pred - pred[0]) - (obs - obs[0]))[1:] ** 2))))
            floors[k].append(round(g, 2))

    selected = gls["selected_model"]
    best_local = gls["best_local_model"]
    candidate = selected if selected in NONLOCAL_KEYS else "H3"

    def _transfer_test(store: Mapping[str, List[float]]) -> Dict[str, Any]:
        diff = np.array(store[best_local]) - np.array(store[candidate])  # > 0: candidate transfers better
        out: Dict[str, Any] = dict(_cluster_ttest(diff))
        out["candidate_beats_best_local_p_0_05"] = bool(out["mean"] > 0 and out["p_two_sided"] < 0.05)
        return out

    transfer_primary = _transfer_test(rmse_step)
    transfer_absolute = _transfer_test(rmse)
    passed = transfer_primary["candidate_beats_best_local_p_0_05"]

    def _sig_pos(name: str) -> bool:
        return bool(tests[name]["significant_holm_0_05"] and tests[name]["mean"] > 0)

    dfs_ok = _sig_pos("C_DFS_asymmetry")
    decisions = {
        # Preregistered requirements per candidate, with the null replaced by the best local model.
        "correlated_dephasing_supported": bool(selected in ("H2", "H3") and dfs_ok and passed),
        "non_markovian_correlated_memory_supported": bool(
            selected == "H3" and dfs_ok and _sig_pos("C_memory_dd32_minus_echo1_16us") and passed),
        "spectator_zz_supported": bool(selected == "H1" and _sig_pos("C_spectator_activation") and passed),
        "pulse_collision_apparatus_supported": bool(selected == "Apparatus" and step_s48 >= 0.75),
        # Preregistered kill criterion: wins in-sample but does not transfer.
        "candidate_falsified_by_held_out": bool(selected in ("H1", "H2", "H3") and not passed),
        "local_noise_only": bool(selected in LOCAL_KEYS),
        # Descriptive: dephasing is coloured (echo/DD-dependent) rather than white. Expected, not novel.
        "local_non_white_dephasing": bool(
            best_local != "H0" and gls["fits"]["H0"]["bic"] - gls["fits"][best_local]["bic"] >= DELTA_BIC_GATE),
    }
    return {
        "amendment_id": AMENDMENT_ID,
        "train_slots": train,
        "held_out_slots": held_out,
        "fit_conditions": [cond_ids[i] for i in fit_idx],
        "model_selection": {
            "selected_model": selected,
            "best_local_model": best_local,
            "raw_min_bic_model": gls["raw_min_bic_model"],
            "delta_bic_over_best_local": round(gls["delta_bic_over_best_local"], 3),
            "delta_bic_h0_minus_best_local": round(gls["fits"]["H0"]["bic"] - gls["fits"][best_local]["bic"], 3),
            "delta_bic_gate": DELTA_BIC_GATE,
            "fits": gls["fits"],
        },
        "mechanistic_contrasts_holm": tests,
        "step_fraction_s48": round(step_s48, 4),
        "memory_contrast_note": (
            "C_memory (DD32 minus echo at 16 us) is expected to be positive under ordinary local 1/f "
            "noise; it is required for H3 but is not, on its own, evidence of correlated memory."
        ),
        "held_out_transfer": {
            "calibration_condition": CALIBRATION_CONDITION,
            "candidate_model": candidate,
            "candidate_is_selected_model": candidate == selected,
            "best_local_model": best_local,
            "per_coupler_rmse_stagger_change": {k: [round(x, 5) for x in v] for k, v in rmse_step.items()},
            "mean_rmse_stagger_change": {k: round(float(np.mean(v)), 5) for k, v in rmse_step.items()},
            "per_coupler_rmse_absolute": {k: [round(x, 5) for x in v] for k, v in rmse.items()},
            "mean_rmse_absolute": {k: round(float(np.mean(v)), 5) for k, v in rmse.items()},
            "solved_white_floor_hz": floors,
            "primary_stagger_change_best_local_minus_candidate": transfer_primary,
            "secondary_absolute_best_local_minus_candidate": transfer_absolute,
        },
        "ramsey_descriptive_mean": {
            c: round(float(np.mean(means[:, i])), 5) for i, c in enumerate(cond_ids) if c.startswith("ramsey_")
        },
        "decisions": decisions,
    }
