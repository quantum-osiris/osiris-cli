"""
Competing Physical Open-System Models for EXPERIMENT CORRELATED-NOISE-01
========================================================================

Phase 2 Section 6 (`EXPERIMENT CORRELATED-NOISE-01`).

Implements exact analytical / piecewise open-quantum-system observable predictors
and hardware-`dt`-aligned ISA circuit builders for four competing physical
hypotheses plus an apparatus pulse-collision control:
  - H0 (`H0_IndependentLocal`):
      Independent local Markovian T1 + pure dephasing T_phi Lindblad channels,
      with DD-symmetrized longitudinal T1 populations when n_dd_pulses >= 1.
  - H1 (`H1_StaticSpectatorZZ`):
      Local Lindblad + static Hamiltonian ZZ coupling (pair zeta_uv + spectator
      zeta_us, zeta_vs) integrated exactly in the toggling frame. Refocuses 1Q
      echo when spectators are unpulsed (`alpha_u = 0`) and separates active
      |+>_s spectator preparation from background thermal/pulsed spectator weight.
  - H2 (`H2_CorrelatedMarkovian`):
      Local Lindblad + spatially correlated memoryless collective dephasing
      gamma_phi_uv enforcing multi-qubit lattice Kossakowski PSD constraints.
  - H3 (`H3_NonMarkovianMemory`):
      Local T1 + temporally correlated colored / Ornstein-Uhlenbeck (or damped
      oscillatory) Gaussian bath evaluated via EXACT closed-form piecewise
      analytical integration over pulse-flip intervals (zero quadrature aliasing).
  - Apparatus (`Apparatus_PulseCollision`):
      H0 + finite-width microwave drive crosstalk (`36 ns = 9 dt` on ibm_marrakesh,
      `24 ns = 6 dt` on ibm_fez) during overlapping DD envelopes (`|Delta| < w_p`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit

HERON_DT_S: float = 4.0e-9
MARRAKESH_PULSE_WIDTH_S: float = 36.0e-9  # 9 dt native X/SX on ibm_marrakesh
FEZ_PULSE_WIDTH_S: float = 24.0e-9        # 6 dt native X/SX on ibm_fez


@dataclass(frozen=True)
class ProbeCondition:
    """Single experimental measurement condition in the CORRELATED-NOISE-01 design."""
    condition_id: str
    observable: str           # 'bell_phi_plus', 'bell_psi_plus', 'ramsey_echo_u', 'connected_zz_corr'
    total_time_s: float       # Total evolution window T (seconds), exact multiple of 4.0 ns dt
    n_dd_pulses: int          # 0 (Ramsey/idle), 1 (Hahn echo), 8 (XY8), 16 (XY8x2), 32 (XY8x4)
    stagger_offset_s: float   # Sublattice offset Delta on qubit v (exact multiple of 4.0 ns dt)
    spectator_superposition: bool = False  # False -> spectator in |0>; True -> spectator in |+>
    graph_distance: int = 1   # Heavy-hex lattice distance between u and v (1 = adjacent)
    pulse_width_s: float = MARRAKESH_PULSE_WIDTH_S


def pulse_centers_s(total_time_s: float, n_dd_pulses: int, offset_s: float = 0.0) -> np.ndarray:
    """
    Return sorted modulo-wrapped periodic pulse centers in [0, total_time_s):
        c_k = ((k + 0.5) * slot + offset_s) mod total_time_s
    Using exact periodic modulo wrapping preserves all `n_dd_pulses` flips inside
    [0, T) even at the half-slot boundary `offset_s = 0.5 * slot`, keeping single-qubit
    toggling exposure identically zero for even `n_dd_pulses`.
    """
    if n_dd_pulses <= 0 or total_time_s <= 0.0:
        return np.empty(0, dtype=np.float64)
    slot = total_time_s / float(n_dd_pulses)
    raw = (np.arange(n_dd_pulses, dtype=np.float64) + 0.5) * slot + offset_s
    wrapped = np.mod(raw, total_time_s)
    # Snap near-T floating-point wraparound (e.g. T - 1e-22) cleanly to 0.0
    wrapped = np.where(np.isclose(wrapped, total_time_s, rtol=0.0, atol=1e-16), 0.0, wrapped)
    return np.sort(wrapped)


def toggling_sign_waveform(
    t_grid: np.ndarray, total_time_s: float, n_dd_pulses: int, offset_s: float = 0.0
) -> np.ndarray:
    """Evaluate toggling-frame sign s(t) in {+1, -1} at points `t_grid` in [0, T]."""
    centers = pulse_centers_s(total_time_s, n_dd_pulses, offset_s)
    if centers.size == 0:
        return np.ones_like(t_grid, dtype=np.float64)
    n_flips = np.sum(t_grid[:, None] >= centers[None, :], axis=1)
    return np.where((n_flips % 2) == 0, 1.0, -1.0)


def piecewise_toggling_intervals(
    total_time_s: float, n_dd_pulses: int, stagger_offset_s: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Partition [0, T] at all exact pulse-flip knots of qubits u and v, returning
    `(a_starts, b_ends, s_u_const, s_v_const)` where s_u and s_v are strictly
    constant on every sub-interval `[a_k, b_k]`.
    """
    c_u = pulse_centers_s(total_time_s, n_dd_pulses, 0.0)
    c_v = pulse_centers_s(total_time_s, n_dd_pulses, stagger_offset_s)
    knots = np.unique(np.concatenate(([0.0, total_time_s], c_u, c_v)))
    # Filter degenerate zero-length intervals from coincident pulses (e.g. offset=0)
    keep = np.diff(knots) > 1e-17
    a = knots[:-1][keep]
    b = knots[1:][keep]
    mids = 0.5 * (a + b)
    su = toggling_sign_waveform(mids, total_time_s, n_dd_pulses, 0.0)
    sv = toggling_sign_waveform(mids, total_time_s, n_dd_pulses, stagger_offset_s)
    return a, b, su, sv


def exact_pair_toggling_exposure(
    total_time_s: float, n_dd_pulses: int, stagger_offset_s: float
) -> float:
    """Compute exact alpha_uv = (1/T) int_0^T s_u(t) s_v(t) dt over exact pulse knots."""
    if n_dd_pulses <= 0 or total_time_s <= 0.0:
        return 1.0
    a, b, su, sv = piecewise_toggling_intervals(total_time_s, n_dd_pulses, stagger_offset_s)
    return float(np.sum(su * sv * (b - a)) / total_time_s)


def exact_single_toggling_exposure(
    total_time_s: float, n_dd_pulses: int, offset_s: float = 0.0
) -> float:
    """Compute exact alpha_u = (1/T) int_0^T s_u(t) dt (0 for balanced echo / even DD)."""
    if n_dd_pulses <= 0 or total_time_s <= 0.0:
        return 1.0
    c_u = pulse_centers_s(total_time_s, n_dd_pulses, offset_s)
    knots = np.unique(np.concatenate(([0.0, total_time_s], c_u)))
    keep = np.diff(knots) > 1e-17
    a, b = knots[:-1][keep], knots[1:][keep]
    mids = 0.5 * (a + b)
    su = toggling_sign_waveform(mids, total_time_s, n_dd_pulses, offset_s)
    return float(np.sum(su * (b - a)) / total_time_s)


def pulse_overlap_fraction(
    total_time_s: float, n_dd_pulses: int, stagger_offset_s: float, pulse_width_s: float = MARRAKESH_PULSE_WIDTH_S
) -> float:
    """Fraction of finite 1Q pulse width [0, 1] during which pulses on u and v overlap."""
    if n_dd_pulses <= 0 or pulse_width_s <= 0.0 or total_time_s <= 0.0:
        return 0.0
    slot = total_time_s / float(n_dd_pulses)
    # Periodic shortest separation between pulse centers on u and v
    delta_mod = abs(stagger_offset_s) % slot
    sep = min(delta_mod, slot - delta_mod)
    return max(0.0, 1.0 - sep / pulse_width_s)


def verify_heavy_hex_psd_covariance(
    rho_spatial: float, spatial_length_hops: float, max_degree: int = 3
) -> bool:
    """
    Check multi-qubit Kossakowski positive-semidefiniteness (K >= 0) of the exponential
    spatial correlation model C_ab = rho_spatial * exp(-(d_ab - 1) / xi) on a heavy-hex
    degree-3 star (hub + 3 neighbors at pairwise distance d=2).
    """
    if abs(rho_spatial) > 1.0 or spatial_length_hops <= 0.0:
        return False
    n = 1 + max_degree
    c_mat = np.eye(n, dtype=np.float64)
    c1 = rho_spatial
    c2 = rho_spatial * math.exp(-1.0 / spatial_length_hops)
    for j in range(1, n):
        c_mat[0, j] = c_mat[j, 0] = c1
        for k in range(j + 1, n):
            c_mat[j, k] = c_mat[k, j] = c2
    min_eig = float(np.min(np.linalg.eigvalsh(c_mat)))
    return min_eig >= -1e-12


def clamp_heavy_hex_psd_rho(rho_spatial: float, spatial_length_hops: float) -> float:
    """
    Clamp `rho_spatial` to the exact Kossakowski-PSD interval on a degree-3 heavy-hex
    hub: 1 + 2*c2 - 3*rho^2 >= 0 and 1 - c2 >= 0, guaranteeing CPTP multi-qubit evolution.
    """
    xi = max(spatial_length_hops, 1e-6)
    e1 = math.exp(-1.0 / xi)
    # Quadratic roots of 3*rho^2 - 2*e1*rho - 1 <= 0:
    # rho in [(e1 - sqrt(e1^2 + 3))/3, (e1 + sqrt(e1^2 + 3))/3]
    disc = math.sqrt(e1 * e1 + 3.0)
    rho_min = max(-0.999, (e1 - disc) / 3.0)
    rho_max = min(0.999, (e1 + disc) / 3.0)
    return float(np.clip(rho_spatial, rho_min, rho_max))


def build_isa_staggered_dd_bell_circuit(
    total_time_s: float = 8.192e-6,
    n_dd_pulses: int = 16,
    stagger_offset_s: float = 0.0,
    bell_state: str = "phi_plus",
    pauli_basis: str = "ZZ",
    dt_s: float = HERON_DT_S,
    pulse_width_s: float = MARRAKESH_PULSE_WIDTH_S,
) -> Tuple[QuantumCircuit, Dict[str, int | float]]:
    """
    Construct a hardware-executable, `dt = 4.0 ns`-aligned 2-qubit staggered-DD Bell
    tomography circuit with finite-duration `x` pulses and barrier-protected `delay`
    slots (preventing `optimization_level >= 2` gate absorption across idle boundaries).
    """
    total_dt = int(round(total_time_s / dt_s))
    offset_dt = int(round(stagger_offset_s / dt_s))
    pulse_dt = int(round(pulse_width_s / dt_s))
    if not math.isclose(total_dt * dt_s, total_time_s, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(f"total_time_s={total_time_s} is not an exact integer multiple of dt={dt_s}")
    if not math.isclose(offset_dt * dt_s, stagger_offset_s, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(f"stagger_offset_s={stagger_offset_s} is not an exact integer multiple of dt={dt_s}")
    if n_dd_pulses > 0 and total_dt % n_dd_pulses != 0:
        raise ValueError(f"total_dt={total_dt} must be divisible by n_dd_pulses={n_dd_pulses}")

    qc = QuantumCircuit(2, 2, name=f"isa_{bell_state}_d{offset_dt}dt_{pauli_basis}")
    # Bell preparation: |Phi+> or |Psi+>
    qc.sx(0)
    qc.rz(math.pi / 2.0, 0)
    qc.sx(0)
    if bell_state == "psi_plus":
        qc.x(1)
    elif bell_state != "phi_plus":
        raise ValueError(f"Unknown bell_state: {bell_state!r}")
    qc.sx(1)
    qc.rz(math.pi / 2.0, 1)
    qc.cz(0, 1)
    qc.sx(1)
    qc.rz(math.pi / 2.0, 1)
    qc.barrier()

    def _append_wire_dd(wire: int, shift_dt: int) -> int:
        if n_dd_pulses <= 0:
            qc.delay(total_dt, wire, unit="dt")
            return total_dt
        slot_dt = total_dt // n_dd_pulses
        if slot_dt <= pulse_dt:
            raise ValueError(f"slot_dt={slot_dt} must exceed finite pulse_dt={pulse_dt}")
        # Place pulse starts on exact dt ticks inside [0, total_dt - pulse_dt]
        base_start = (slot_dt - pulse_dt) // 2
        starts = sorted(((k * slot_dt + base_start + shift_dt) % (total_dt - pulse_dt + 1)) for k in range(n_dd_pulses))
        cursor = 0
        for st in starts:
            gap = max(0, st - cursor)
            if gap > 0:
                qc.delay(gap, wire, unit="dt")
                cursor += gap
            qc.x(wire)
            cursor += pulse_dt
        rem = max(0, total_dt - cursor)
        if rem > 0:
            qc.delay(rem, wire, unit="dt")
            cursor += rem
        return cursor

    dur_0 = _append_wire_dd(0, 0)
    dur_1 = _append_wire_dd(1, offset_dt)
    qc.barrier()

    # Pauli readout basis rotation
    for w_idx, b_ch in enumerate(pauli_basis.upper()):
        if b_ch == "X":
            qc.rz(math.pi / 2.0, w_idx)
            qc.sx(w_idx)
            qc.rz(math.pi / 2.0, w_idx)
        elif b_ch == "Y":
            qc.sx(w_idx)
            qc.rz(math.pi / 2.0, w_idx)
        elif b_ch != "Z":
            raise ValueError(f"Unsupported Pauli basis char: {b_ch!r}")
    qc.measure([0, 1], [0, 1])

    return qc, {
        "total_dt": total_dt,
        "offset_dt": offset_dt,
        "pulse_dt": pulse_dt,
        "wire0_dd_dt": dur_0,
        "wire1_dd_dt": dur_1,
    }


# ---------------------------------------------------------------------------
# Competing Physical Models (H0, H1, H2, H3 + Apparatus Control)
# ---------------------------------------------------------------------------

class NoiseModelBase:
    name: str = "base"
    n_free_params: int = 0

    def predict(self, cond: ProbeCondition) -> float:
        raise NotImplementedError

    def predict_suite(self, suite: Sequence[ProbeCondition]) -> np.ndarray:
        return np.array([self.predict(c) for c in suite], dtype=np.float64)


@dataclass(frozen=True)
class H0IndependentLocalModel(NoiseModelBase):
    """
    H0: Independent local Markovian T1 + pure dephasing gamma_phi Lindblad model.
      Includes per-hop routing baseline `swap_fidelity_per_hop` (calibrated at T=0 on
      d=2,3 pairs so SWAP gate infidelity is never misattributed to spatial bath decay).
    """
    t1_u_s: float = 180.0e-6
    t1_v_s: float = 160.0e-6
    gamma_phi_u_hz: float = 4500.0
    gamma_phi_v_hz: float = 5500.0
    f0_spam: float = 0.975
    swap_fidelity_per_hop: float = 0.985
    name: str = "H0_IndependentLocal"
    n_free_params: int = 5

    def _effective_spam_ceiling(self, graph_distance: int) -> float:
        extra_hops = max(0, graph_distance - 1)
        if extra_hops == 0:
            return self.f0_spam
        surv_route = self.swap_fidelity_per_hop ** extra_hops
        return 0.25 + (self.f0_spam - 0.25) * surv_route

    def _bell_fidelity_from_dephasing_exponent(
        self,
        T: float,
        chi_dephase: float,
        is_phi_plus: bool,
        n_dd_pulses: int = 16,
        graph_distance: int = 1,
    ) -> float:
        e1_u = math.exp(-T / self.t1_u_s)
        e1_v = math.exp(-T / self.t1_v_s)
        if n_dd_pulses >= 1:
            # Balanced pi-pulses equalize |0>/<->|1> residence, symmetrizing T1 parity decay
            # across |Phi+> and |Psi+> (<Z_u Z_v>(T) = +/- exp(-T*(1/T1_u + 1/T1_v)))
            pop_term = 0.5 * (1.0 + e1_u * e1_v)
        else:
            pop_phi = 0.5 * (2.0 - e1_u - e1_v + 2.0 * e1_u * e1_v)
            pop_psi = 0.5 * (e1_u + e1_v)
            pop_term = pop_phi if is_phi_plus else pop_psi

        coh_t1 = math.exp(-0.5 * T * (1.0 / self.t1_u_s + 1.0 / self.t1_v_s))
        coh = coh_t1 * math.exp(-chi_dephase)
        f_raw = 0.5 * (pop_term + coh)
        f_ceil = self._effective_spam_ceiling(graph_distance)
        alpha_spam = (4.0 * f_ceil - 1.0) / 3.0
        return 0.25 + 0.75 * alpha_spam * ((4.0 * f_raw - 1.0) / 3.0)

    def predict(self, cond: ProbeCondition) -> float:
        T = cond.total_time_s
        chi_local_pair = (self.gamma_phi_u_hz + self.gamma_phi_v_hz) * T
        if cond.observable == "bell_phi_plus":
            return self._bell_fidelity_from_dephasing_exponent(
                T, chi_local_pair, True, cond.n_dd_pulses, cond.graph_distance
            )
        if cond.observable == "bell_psi_plus":
            return self._bell_fidelity_from_dephasing_exponent(
                T, chi_local_pair, False, cond.n_dd_pulses, cond.graph_distance
            )
        if cond.observable == "ramsey_echo_u":
            coh_u = math.exp(-T * (0.5 / self.t1_u_s + self.gamma_phi_u_hz))
            return 0.5 * (1.0 + (2.0 * self.f0_spam - 1.0) * coh_u)
        if cond.observable == "connected_zz_corr":
            return 0.0
        raise ValueError(f"Unknown observable: {cond.observable!r}")


@dataclass(frozen=True)
class H1StaticSpectatorZZModel(NoiseModelBase):
    """
    H1: Local Markovian noise + static Hamiltonian spectator ZZ interactions:
        H_ZZ = (zeta_uv / 2) Z_u Z_v + (zeta_us / 2) Z_u Z_s + (zeta_vs / 2) Z_v Z_s
    with angular coupling zeta = 2*pi * f_zz.
      - Isolated pair Z_u Z_v commutes with both |Phi+> and |Psi+>; static Bell dephasing
        is driven by spectator couplings Z_u Z_s, Z_v Z_s.
      - On 1Q `ramsey_echo_u` (spectator unpulsed), balanced DD (`n_dd >= 1`) gives
        alpha_u = 0 and refocuses static ZZ completely.
      - On 2Q Bell arms where active sublattice-B spectators are pulsed alongside v
        (`spectator_pulsed_with_v=True`), synchronous DD keeps alpha_eff = 1 while
        half-slot staggering drives alpha_eff -> 0; preparing |+>_s (`spectator_superposition=True`)
        maximally activates visibility `p_s = 1.0` vs passive/active-chain weight `p_s = 0.35`.
    """
    base: H0IndependentLocalModel = H0IndependentLocalModel()
    f_zz_spectator_hz: float = 28_000.0
    spectator_pulsed_with_v: bool = True
    passive_spectator_visibility: float = 0.35
    name: str = "H1_StaticSpectatorZZ"
    n_free_params: int = 6

    def predict(self, cond: ProbeCondition) -> float:
        T = cond.total_time_s
        zz_hz = (
            self.f_zz_spectator_hz
            if cond.graph_distance == 1
            else self.f_zz_spectator_hz * (0.08 ** (cond.graph_distance - 1))
        )
        omega_zz = 2.0 * math.pi * zz_hz

        if cond.observable == "ramsey_echo_u":
            # Single-qubit echo on u with unpulsed spectator refocuses static ZZ (alpha_u = 0 for n_dd >= 1)
            alpha_1q = exact_single_toggling_exposure(T, cond.n_dd_pulses, 0.0)
            p_vis = 1.0 if cond.spectator_superposition else self.passive_spectator_visibility
            coh_zz_1q = (1.0 - p_vis) + p_vis * math.cos(omega_zz * T * alpha_1q)
            coh_u = math.exp(-T * (0.5 / self.base.t1_u_s + self.base.gamma_phi_u_hz)) * coh_zz_1q
            return 0.5 * (1.0 + (2.0 * self.base.f0_spam - 1.0) * coh_u)

        if self.spectator_pulsed_with_v and cond.n_dd_pulses > 0:
            alpha_eff = exact_pair_toggling_exposure(T, cond.n_dd_pulses, cond.stagger_offset_s)
        else:
            alpha_eff = exact_single_toggling_exposure(T, cond.n_dd_pulses, 0.0)

        p_vis = 1.0 if cond.spectator_superposition else self.passive_spectator_visibility
        coh_zz_factor = (1.0 - p_vis) + p_vis * math.cos(omega_zz * T * alpha_eff)
        chi_local = (self.base.gamma_phi_u_hz + self.base.gamma_phi_v_hz) * T

        if cond.observable in ("bell_phi_plus", "bell_psi_plus"):
            is_phi = cond.observable == "bell_phi_plus"
            e1_u = math.exp(-T / self.base.t1_u_s)
            e1_v = math.exp(-T / self.base.t1_v_s)
            if cond.n_dd_pulses >= 1:
                pop_term = 0.5 * (1.0 + e1_u * e1_v)
            else:
                pop_term = (
                    0.5 * (2.0 - e1_u - e1_v + 2.0 * e1_u * e1_v)
                    if is_phi
                    else 0.5 * (e1_u + e1_v)
                )
            coh = math.exp(-0.5 * T * (1.0 / self.base.t1_u_s + 1.0 / self.base.t1_v_s) - chi_local) * coh_zz_factor
            f_raw = 0.5 * (pop_term + coh)
            f_ceil = self.base._effective_spam_ceiling(cond.graph_distance)
            alpha_spam = (4.0 * f_ceil - 1.0) / 3.0
            return 0.25 + 0.75 * alpha_spam * ((4.0 * f_raw - 1.0) / 3.0)

        if cond.observable == "connected_zz_corr":
            return 0.0
        raise ValueError(f"Unknown observable: {cond.observable!r}")


@dataclass(frozen=True)
class H2CorrelatedMarkovianModel(NoiseModelBase):
    """
    H2: Spatially correlated Markovian collective-dephasing Lindblad model.
      Enforces heavy-hex lattice Kossakowski PSD positivity via `clamp_heavy_hex_psd_rho`
      when `enforce_lattice_psd=True` (default pairwise CPTP `|rho_spatial| <= 1`).
    """
    base: H0IndependentLocalModel = H0IndependentLocalModel()
    rho_spatial: float = 0.72
    spatial_length_hops: float = 2.0
    enforce_lattice_psd: bool = False
    name: str = "H2_CorrelatedMarkovian"
    n_free_params: int = 7

    def _gamma_cov(self, graph_distance: int) -> float:
        if abs(self.rho_spatial) > 1.0:
            raise ValueError("CPTP positivity requires |rho_spatial| <= 1.0")
        rho_eff = (
            clamp_heavy_hex_psd_rho(self.rho_spatial, self.spatial_length_hops)
            if self.enforce_lattice_psd
            else self.rho_spatial
        )
        g_max = math.sqrt(self.base.gamma_phi_u_hz * self.base.gamma_phi_v_hz)
        decay = math.exp(-max(0, graph_distance - 1) / max(self.spatial_length_hops, 1e-6))
        return rho_eff * g_max * decay

    def predict(self, cond: ProbeCondition) -> float:
        T = cond.total_time_s
        alpha_uv = exact_pair_toggling_exposure(T, cond.n_dd_pulses, cond.stagger_offset_s)
        g_cov = self._gamma_cov(cond.graph_distance)
        chi_base = (self.base.gamma_phi_u_hz + self.base.gamma_phi_v_hz) * T

        if cond.observable == "bell_phi_plus":
            chi_phi = max(0.0, chi_base + 2.0 * alpha_uv * g_cov * T)
            return self.base._bell_fidelity_from_dephasing_exponent(
                T, chi_phi, True, cond.n_dd_pulses, cond.graph_distance
            )
        if cond.observable == "bell_psi_plus":
            chi_psi = max(0.0, chi_base - 2.0 * alpha_uv * g_cov * T)
            return self.base._bell_fidelity_from_dephasing_exponent(
                T, chi_psi, False, cond.n_dd_pulses, cond.graph_distance
            )
        if cond.observable == "ramsey_echo_u":
            return self.base.predict(cond)
        if cond.observable == "connected_zz_corr":
            return 2.0 * alpha_uv * g_cov * T
        raise ValueError(f"Unknown observable: {cond.observable!r}")


@dataclass(frozen=True)
class H3NonMarkovianMemoryModel(NoiseModelBase):
    """
    H3: Temporally non-Markovian colored Ornstein-Uhlenbeck (or damped-oscillatory) bath:
        K_ab(dt) = C_ab * sigma_a * sigma_b * exp(-|dt| / tau_c_s) * cos(2*pi*f_bath_hz * dt)
      Evaluated via EXACT closed-form piecewise analytical double integration over the
      sorted pulse-flip knots of s_u(t) and s_v(t) (eliminating Riemann-grid aliasing
      across sub-slot offsets such as 16, 32, 48, 80, 160, 256 ns to ~1e-16 accuracy).
    """
    t1_u_s: float = 180.0e-6
    t1_v_s: float = 160.0e-6
    gamma_white_u_hz: float = 900.0
    gamma_white_v_hz: float = 1100.0
    sigma_ou_u_rad_s: float = 55_000.0
    sigma_ou_v_rad_s: float = 60_000.0
    tau_c_s: float = 1.5e-6
    spatial_corr: float = 0.65
    spatial_length_hops: float = 2.0
    f_bath_hz: float = 0.0
    f0_spam: float = 0.975
    swap_fidelity_per_hop: float = 0.985
    name: str = "H3_NonMarkovianMemory"
    n_free_params: int = 8

    def _exact_piecewise_kernel_integrals(
        self, total_time_s: float, n_dd_pulses: int, stagger_offset_s: float
    ) -> Tuple[float, float, float]:
        a, b, su, sv = piecewise_toggling_intervals(total_time_s, n_dd_pulses, stagger_offset_s)
        lengths = b - a
        tau_c = max(self.tau_c_s, 1.0e-12)
        z = complex(1.0 / tau_c, 2.0 * math.pi * self.f_bath_hz)

        # Diagonal block integral on [a_k, b_k]^2: Re[ (2 / z^2) * (z*L_k + expm1(-z*L_k)) ]
        z_l = z * lengths
        diag_vals = ((2.0 / (z * z)) * (z_l + np.expm1(-z_l))).real
        w_mat = np.diag(diag_vals)

        # Off-diagonal strictly lower triangle j > k (where a_j >= b_k >= 0):
        # Re[ ((-expm1(-z*L_j))/z) * ((-expm1(-z*L_k))/z) * exp(-z * (a_j - b_k)) ]
        m = lengths.size
        if m > 1:
            edge_factor = (-np.expm1(-z_l)) / z
            gap = np.maximum(0.0, a[:, None] - b[None, :])
            lower_mask = np.tril(np.ones((m, m), dtype=bool), k=-1)
            exp_gap = np.zeros((m, m), dtype=np.complex128)
            exp_gap[lower_mask] = np.exp(-z * gap[lower_mask])
            off_lower = ((edge_factor[:, None] * edge_factor[None, :]) * exp_gap).real
            w_mat = w_mat + off_lower + off_lower.T

        iuu = max(0.0, float(su @ w_mat @ su))
        ivv = max(0.0, float(sv @ w_mat @ sv))
        iuv = float(su @ w_mat @ sv)
        return iuu, ivv, iuv

    def _cumulants(self, cond: ProbeCondition) -> Tuple[float, float, float]:
        T = cond.total_time_s
        if T <= 0.0:
            return 0.0, 0.0, 0.0
        iuu, ivv, iuv = self._exact_piecewise_kernel_integrals(
            T, cond.n_dd_pulses, cond.stagger_offset_s
        )
        c_spatial = self.spatial_corr * math.exp(
            -max(0, cond.graph_distance - 1) / max(self.spatial_length_hops, 1e-6)
        )
        chi_u = self.gamma_white_u_hz * T + 0.5 * (self.sigma_ou_u_rad_s ** 2) * iuu
        chi_v = self.gamma_white_v_hz * T + 0.5 * (self.sigma_ou_v_rad_s ** 2) * ivv
        chi_cross = c_spatial * self.sigma_ou_u_rad_s * self.sigma_ou_v_rad_s * iuv
        return chi_u, chi_v, chi_cross

    def predict(self, cond: ProbeCondition) -> float:
        T = cond.total_time_s
        chi_u, chi_v, chi_cross = self._cumulants(cond)
        helper = H0IndependentLocalModel(
            t1_u_s=self.t1_u_s,
            t1_v_s=self.t1_v_s,
            gamma_phi_u_hz=0.0,
            gamma_phi_v_hz=0.0,
            f0_spam=self.f0_spam,
            swap_fidelity_per_hop=self.swap_fidelity_per_hop,
        )
        if cond.observable == "bell_phi_plus":
            return helper._bell_fidelity_from_dephasing_exponent(
                T, max(0.0, chi_u + chi_v + chi_cross), True, cond.n_dd_pulses, cond.graph_distance
            )
        if cond.observable == "bell_psi_plus":
            return helper._bell_fidelity_from_dephasing_exponent(
                T, max(0.0, chi_u + chi_v - chi_cross), False, cond.n_dd_pulses, cond.graph_distance
            )
        if cond.observable == "ramsey_echo_u":
            coh_u = math.exp(-0.5 * T / self.t1_u_s - max(0.0, chi_u))
            return 0.5 * (1.0 + (2.0 * self.f0_spam - 1.0) * coh_u)
        if cond.observable == "connected_zz_corr":
            return chi_cross
        raise ValueError(f"Unknown observable: {cond.observable!r}")


@dataclass(frozen=True)
class ApparatusPulseCollisionModel(NoiseModelBase):
    """
    Apparatus Control Model: Independent local Markovian noise + finite-width
    (`36 ns` on ibm_marrakesh; `24 ns` on ibm_fez) microwave drive crosstalk during
    overlapping simultaneous DD pulses (`|Delta| < w_p`).
    Predicts sharp step recovery once `Delta >= w_p` (`48 ns > 36 ns`) and a flat
    plateau across `Delta in [48, 256] ns`.
    """
    base: H0IndependentLocalModel = H0IndependentLocalModel()
    per_collision_error: float = 0.0085
    name: str = "Apparatus_PulseCollision"
    n_free_params: int = 6

    def predict(self, cond: ProbeCondition) -> float:
        f_base = self.base.predict(cond)
        if cond.observable in ("bell_phi_plus", "bell_psi_plus") and cond.n_dd_pulses > 0 and cond.graph_distance == 1:
            ov = pulse_overlap_fraction(
                cond.total_time_s, cond.n_dd_pulses, cond.stagger_offset_s, cond.pulse_width_s
            )
            surv_col = (1.0 - self.per_collision_error * ov) ** cond.n_dd_pulses
            return 0.25 + (f_base - 0.25) * surv_col
        return f_base
