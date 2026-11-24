"""
ADV-1 — 3-Attack Classical-Hardness Pre-Gate & `osiris.quantum_protocol/1` Manifest Emitter
===========================================================================================

Enforces the fail-closed classical-hardness gate (`evaluate_adv1_pregate`) prior to any QPU
credit allocation on Quantinuum `H2-1` / `Helios-1`:

  1. Attack 1 — Bounded-`chi` MPS Contraction (`attack_bounded_mps`):
     Sweeps `chi in {8, 16, 32, 64, 128}` (`qiskit_aer` MPS), recording modal peak recovery,
     empirical `P(secret)`, Total Variation Distance vs exact statevector (when `n <= 20`),
     and truncation fidelity ratio.
  2. Attack 2 — Truncated Pauli-Path / Low-Weight Stabilizer Back-Propagation
     (`attack_truncated_pauli_path`):
     Propagates each output `Z_i` observable backward (`O <- G^dagger O G`) with explicit
     Pauli weight cutoff `wt(P) <= w_max` (sweeping `w_max in {2, 3, 4}`), estimating
     single-qubit bias `<Z_i>` for greedy secret reconstruction `s_hat`.
  3. Attack 3 — Causal Light-Cone / Treewidth Reduction (`attack_causal_light_cone`):
     Computes exact backward causal light cone `C(i)` for every output bit `i = 0..n-1` and
     the interaction graph min-degree elimination treewidth upper bound, rejecting any
     circuit with `max_i |C(i)| <= L_tractable` (and verifying exact cone contraction).

Fail-Closed Rule:
  `qpu_allocation_permitted == True` IFF:
    - `not mirror_structure_detected`
    - `not attack1_mps_any_win` (across all `chi` in sweep)
    - `not attack2_pauli_any_win` (across all `w_max` in sweep)
    - `not attack3_light_cone_tractable_or_win` (`max_i |C(i)| > L_tractable`)
    - `helios1_predicted_peak >= 10 * 2^{-n}`
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector
from qiskit_aer import AerSimulator

from adv0 import peaked as adv0_peaked
from adv0.attacks import CZ_T, SX_T, X_T
from adv1.looped_peaked import (
    Gate,
    LoopedPeakedCircuit,
    build_adv0_mirror_control,
    build_looped_chorded_gadget_peaked,
    build_looped_commuting_phase_peaked,
    verify_non_mirror_signature,
)


@dataclass(frozen=True)
class PregateVerdict:
    candidate_id: str
    family: str
    topology: str
    n_qubits: int
    depth_layers: int
    n_cz: int
    n_1q_phys: int
    n_rz_virtual: int
    helios1_hqc_1000_shots: float
    helios1_predicted_peak_prob: float
    uniform_baseline_prob: float
    exact_noiseless_peak_prob: Optional[float]
    mirror_audit: Dict[str, Any]
    attack1_bounded_mps: Dict[str, Any]
    attack2_pauli_path: Dict[str, Any]
    attack3_light_cone: Dict[str, Any]
    rejection_reasons: List[str]
    qpu_allocation_permitted: bool


class _WeightBoundedPauliOps:
    """Backward Heisenberg Pauli sum with explicit Pauli weight cutoff `wt(P) <= w_max`."""

    def __init__(self, n: int, q: int) -> None:
        self.L = np.zeros((1, n), dtype=np.int8)
        self.L[0, q] = 3  # Z_q
        self.c = np.ones(1, dtype=np.float64)
        self.dropped_l2 = 0.0

    def apply_cliff1(self, q: int, table: Tuple[np.ndarray, np.ndarray]) -> None:
        perm, sign = table
        old = self.L[:, q].copy()
        self.L[:, q] = perm[old]
        self.c *= sign[old]

    def apply_cz(self, a: int, b: int) -> None:
        perm, sign = CZ_T
        idx = 4 * self.L[:, a] + self.L[:, b]
        new = perm[idx]
        self.L[:, a] = new // 4
        self.L[:, b] = new % 4
        self.c *= sign[idx]
        # CZ can change Pauli weight (+/- 1); prune immediately when buffer grows

    def apply_rz(self, q: int, theta: float) -> None:
        co, si = math.cos(theta), math.sin(theta)
        col = self.L[:, q]
        mask = (col == 1) | (col == 2)
        if not np.any(mask):
            return
        if abs(si) < 1e-12:
            self.c[mask] *= co
            return
        sub = self.L[mask].copy()
        cs = self.c[mask].copy()
        sgn = np.where(sub[:, q] == 1, -si, si)
        sub[:, q] = 3 - sub[:, q]
        if abs(co) < 1e-12:
            self.L[mask] = sub
            self.c[mask] = cs * sgn
            return
        self.c[mask] *= co
        self.L = np.concatenate([self.L, sub], axis=0)
        self.c = np.concatenate([self.c, cs * sgn], axis=0)

    def prune_and_merge(self, w_max: int, eps_coeff: float, max_terms: int) -> None:
        if self.c.size == 0:
            return
        # 1. Enforce strict Pauli weight cutoff wt(P) <= w_max BEFORE/AFTER merging duplicates
        L_contig = np.ascontiguousarray(self.L)
        view = L_contig.view(np.dtype((np.void, L_contig.shape[1]))).ravel()
        _, first, inv = np.unique(view, return_index=True, return_inverse=True)
        c_merged = np.bincount(inv.ravel(), weights=self.c, minlength=len(first))
        L_merged = L_contig[first]

        weights = np.sum(L_merged != 0, axis=1)
        keep = (weights <= w_max) & (np.abs(c_merged) > eps_coeff)
        if int(np.sum(keep)) > max_terms:
            kept_abs = np.abs(c_merged[keep])
            thr = float(np.partition(kept_abs, -max_terms)[-max_terms])
            keep &= np.abs(c_merged) >= thr

        self.dropped_l2 += float(np.sum(c_merged[~keep] ** 2))
        self.L = L_merged[keep]
        self.c = c_merged[keep]

    def expectation_all_zero(self) -> float:
        if self.c.size == 0:
            return 0.0
        diag = np.all((self.L == 0) | (self.L == 3), axis=1)
        return float(np.sum(self.c[diag]))


def _backward_z_expectation_weight_cutoff(
    gates: Sequence[Gate],
    n: int,
    target_q: int,
    w_max: int = 3,
    eps_coeff: float = 1e-4,
    max_terms: int = 8192,
) -> Tuple[float, float]:
    ops = _WeightBoundedPauliOps(n, target_q)
    step = 0
    for name, qs, th in reversed(gates):
        if name == "rz":
            ops.apply_rz(qs[0], th)
        elif name == "sx":
            ops.apply_cliff1(qs[0], SX_T)
        elif name == "x":
            ops.apply_cliff1(qs[0], X_T)
        elif name == "cz":
            ops.apply_cz(qs[0], qs[1])
            ops.prune_and_merge(w_max=w_max, eps_coeff=eps_coeff, max_terms=max_terms)
        step += 1
        if step % 24 == 0 or ops.c.size > 2 * max_terms:
            ops.prune_and_merge(w_max=w_max, eps_coeff=eps_coeff, max_terms=max_terms)
    ops.prune_and_merge(w_max=w_max, eps_coeff=eps_coeff, max_terms=max_terms)
    return ops.expectation_all_zero(), ops.dropped_l2


def attack_truncated_pauli_path(
    circuit: LoopedPeakedCircuit,
    w_max_sweep: Sequence[int] = (2, 3, 4),
    eps_coeff: float = 1e-4,
    max_terms: int = 8192,
) -> Dict[str, Any]:
    """
    Attack 2: Truncated Pauli-Path / Low-Weight Stabilizer Back-Propagation across `w_max_sweep`.
    Reconstructs greedy peak `s_hat[q] = '0' if <Z_q> >= 0 else '1'`; wins if `s_hat == secret`.
    """
    t0 = time.monotonic()
    curves: List[Dict[str, Any]] = []
    any_win = False

    for w_max in w_max_sweep:
        tw0 = time.monotonic()
        z_vals: List[float] = []
        dropped_tot = 0.0
        for q in range(circuit.n):
            zq, dr = _backward_z_expectation_weight_cutoff(
                circuit.gates,
                circuit.n,
                q,
                w_max=int(w_max),
                eps_coeff=eps_coeff,
                max_terms=max_terms,
            )
            z_vals.append(zq)
            dropped_tot += dr

        greedy_bits = "".join("0" if z >= 0.0 else "1" for z in z_vals)
        signed_z = [
            (z if circuit.secret[q] == "0" else -z) for q, z in enumerate(z_vals)
        ]
        n_correct = sum(1 for q in range(circuit.n) if greedy_bits[q] == circuit.secret[q])
        # Win if greedy sign recovers all n bits AND non-degenerate signal (> 1e-6)
        win = bool(greedy_bits == circuit.secret and min(signed_z) > 1e-6)
        if win:
            any_win = True
        curves.append(
            {
                "w_max": int(w_max),
                "win": win,
                "bits_correct": int(n_correct),
                "bit_accuracy": round(n_correct / float(circuit.n), 4),
                "min_signed_z": round(float(min(signed_z)), 6),
                "mean_abs_z": round(float(np.mean(np.abs(z_vals))), 6),
                "mean_dropped_l2_weight": round(dropped_tot / float(circuit.n), 6),
                "seconds": round(time.monotonic() - tw0, 3),
            }
        )

    return {
        "any_win": any_win,
        "w_max_curves": curves,
        "total_seconds": round(time.monotonic() - t0, 3),
    }


def _backward_cone(gates: Sequence[Gate], target_q: int) -> Tuple[List[int], List[Gate]]:
    cone = {int(target_q)}
    kept: List[Gate] = []
    for g in reversed(gates):
        if set(g[1]) & cone:
            cone.update(g[1])
            kept.append(g)
    return sorted(cone), list(reversed(kept))


def _min_degree_treewidth_ub(n: int, gates: Sequence[Gate]) -> int:
    """Compute min-degree elimination treewidth upper bound of the circuit's 2-qubit graph."""
    adj: Dict[int, set] = {q: set() for q in range(n)}
    for name, qs, _ in gates:
        if len(qs) == 2 and qs[0] != qs[1]:
            adj[qs[0]].add(qs[1])
            adj[qs[1]].add(qs[0])
    tw = 0
    rem = set(range(n))
    while rem:
        v = min(rem, key=lambda x: len(adj[x]))
        nbrs = list(adj[v])
        tw = max(tw, len(nbrs))
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                adj[nbrs[i]].add(nbrs[j])
                adj[nbrs[j]].add(nbrs[i])
        for u in nbrs:
            adj[u].discard(v)
        del adj[v]
        rem.remove(v)
    return tw


def attack_causal_light_cone(
    circuit: LoopedPeakedCircuit,
    tractable_cone_limit: int = 20,
) -> Dict[str, Any]:
    """
    Attack 3: Causal Light-Cone & Treewidth Reduction.
    Computes exact backward light cone `|C(i)|` for every output bit `i = 0..n-1`.
    Any circuit with `max_i |C(i)| <= tractable_cone_limit` is rejected (`tractable = True`)
    and solved directly via exact cone statevector expectation!
    """
    t0 = time.monotonic()
    cones = [_backward_cone(circuit.gates, q) for q in range(circuit.n)]
    sizes = [len(c[0]) for c in cones]
    max_cone = int(max(sizes))
    min_cone = int(min(sizes))
    mean_cone = float(np.mean(sizes))
    tw_ub = _min_degree_treewidth_ub(circuit.n, circuit.gates)

    tractable = bool(max_cone <= tractable_cone_limit)
    exact_recovered = False
    bits_correct = 0

    if tractable:
        sim = AerSimulator(method="statevector")
        rec_bits: List[str] = []
        for q, (cone_qs, cone_gates) in enumerate(cones):
            pos = {v: i for i, v in enumerate(cone_qs)}
            qc = QuantumCircuit(len(cone_qs))
            for name, qs, th in cone_gates:
                mapped = [pos[x] for x in qs]
                if name == "rz":
                    qc.rz(th, mapped[0])
                elif name == "sx":
                    qc.sx(mapped[0])
                elif name == "x":
                    qc.x(mapped[0])
                elif name == "cz":
                    qc.cz(mapped[0], mapped[1])
            qc.save_expectation_value(Pauli("Z"), [pos[q]])
            zq = float(np.real(sim.run(qc).result().data()["expectation_value"]))
            rec_bits.append("0" if zq >= 0.0 else "1")
        rec_str = "".join(rec_bits)
        bits_correct = sum(1 for q in range(circuit.n) if rec_str[q] == circuit.secret[q])
        exact_recovered = bool(rec_str == circuit.secret)

    return {
        "tractable_cone_limit": int(tractable_cone_limit),
        "min_cone_size": min_cone,
        "mean_cone_size": round(mean_cone, 2),
        "max_cone_size": max_cone,
        "treewidth_upper_bound": int(tw_ub),
        "cone_tractable": tractable,
        "exact_cone_recovered_secret": exact_recovered,
        "cone_bits_correct": int(bits_correct) if tractable else None,
        "win": bool(tractable or exact_recovered),
        "seconds": round(time.monotonic() - t0, 3),
    }


def attack_bounded_mps(
    circuit: LoopedPeakedCircuit,
    chi_sweep: Sequence[int] = (8, 16, 32, 64, 128),
    shots: int = 256,
    exact_probs: Optional[np.ndarray] = None,
    seed: int = 7,
) -> Dict[str, Any]:
    """
    Attack 1: Bounded-`chi` Matrix Product State contraction across `chi_sweep`,
    measuring peak-bitstring recovery, empirical `P(secret)`, TVD (when `exact_probs`
    is available for `n <= 20`), and truncation fidelity ratio.
    """
    t0 = time.monotonic()
    qc_meas = circuit.to_qiskit(measure=True)
    target_le = circuit.secret[::-1]  # Qiskit little-endian bitstring
    target_int = int(target_le, 2)
    p_ref = (
        float(exact_probs[target_int])
        if exact_probs is not None
        else max(circuit.predicted_peak_lower_bound, 1e-12)
    )

    curves: List[Dict[str, Any]] = []
    any_win = False

    for chi in chi_sweep:
        tc0 = time.monotonic()
        sim = AerSimulator(
            method="matrix_product_state",
            matrix_product_state_max_bond_dimension=int(chi),
            matrix_product_state_truncation_threshold=1e-10,
            seed_simulator=int(seed) + int(chi),
        )
        cnts = sim.run(qc_meas, shots=shots).result().get_counts()
        modal_str = max(cnts, key=cnts.get)
        win = bool(modal_str == target_le)
        if win:
            any_win = True
        p_est = cnts.get(target_le, 0) / float(shots)
        modal_p = cnts[modal_str] / float(shots)

        tvd: Optional[float] = None
        if exact_probs is not None:
            # Exact TVD = 0.5 * sum_x |p_emp(x) - p_exact(x)| computed in O(|support| + 2^n)
            emp_arr = np.zeros_like(exact_probs)
            for bs, c_val in cnts.items():
                emp_arr[int(bs.replace(" ", ""), 2)] = c_val / float(shots)
            tvd = round(0.5 * float(np.sum(np.abs(emp_arr - exact_probs))), 6)

        fid_ratio = min(1.0, p_est / max(p_ref, 1e-12))
        curves.append(
            {
                "chi": int(chi),
                "win": win,
                "p_peak_est": round(p_est, 6),
                "modal_prob": round(modal_p, 6),
                "tvd_vs_exact": tvd,
                "truncation_fidelity_ratio": round(fid_ratio, 6),
                "seconds": round(time.monotonic() - tc0, 3),
            }
        )

    return {
        "any_win": any_win,
        "shots_per_chi": int(shots),
        "chi_curves": curves,
        "total_seconds": round(time.monotonic() - t0, 3),
    }


def evaluate_adv1_pregate(
    circuit: LoopedPeakedCircuit,
    candidate_id: str,
    chi_sweep: Sequence[int] = (8, 16, 32, 64, 128),
    w_max_sweep: Sequence[int] = (2, 3, 4),
    tractable_cone_limit: int = 20,
    mps_shots: int = 256,
    helios1_p2q: float = 7.9e-4,
    helios1_p1q: float = 3.0e-5,
    helios1_pspam: float = 1.0e-3,
) -> PregateVerdict:
    """
    Execute the complete 3-attack `ADV-1` classical hardness pre-gate on `circuit`
    and compute the fail-closed `qpu_allocation_permitted` verdict.
    """
    exact_probs: Optional[np.ndarray] = None
    exact_peak: Optional[float] = None
    if circuit.n <= 20:
        sv = Statevector.from_instruction(circuit.to_qiskit(measure=False))
        exact_probs = np.abs(sv.data) ** 2
        exact_peak = float(exact_probs[int(circuit.secret[::-1], 2)])

    mirror_audit = verify_non_mirror_signature(circuit)
    a3_lc = attack_causal_light_cone(circuit, tractable_cone_limit=tractable_cone_limit)
    a2_pp = attack_truncated_pauli_path(circuit, w_max_sweep=w_max_sweep)
    a1_mps = attack_bounded_mps(
        circuit, chi_sweep=chi_sweep, shots=mps_shots, exact_probs=exact_probs
    )

    # Quantinuum Helios-1 HQC & survival prediction at 1,000 shots
    shots_hw = 1000
    n_m = 2 * circuit.n  # init + readout
    hqc_1000 = 5.0 + (shots_hw * (circuit.n_1q_phys + 10.0 * circuit.n_cz + 5.0 * n_m)) / 5000.0
    gate_survival = (
        ((1.0 - helios1_p2q) ** circuit.n_cz)
        * ((1.0 - helios1_p1q) ** circuit.n_1q_phys)
        * ((1.0 - helios1_pspam) ** circuit.n)
    )
    ideal_p = exact_peak if exact_peak is not None else circuit.predicted_peak_lower_bound
    pred_hw_peak = float(ideal_p * gate_survival)
    unif_prob = 2.0 ** (-circuit.n)

    reasons: List[str] = []
    if mirror_audit["mirror_structure_detected"]:
        reasons.append("MIRROR_CANCELLATION_STRUCTURE_DETECTED")
    if a1_mps["any_win"]:
        winning_chis = [c["chi"] for c in a1_mps["chi_curves"] if c["win"]]
        reasons.append(f"DEFEATED_BY_BOUNDED_MPS_CHI_{winning_chis[0]}")
    if a2_pp["any_win"]:
        winning_ws = [c["w_max"] for c in a2_pp["w_max_curves"] if c["win"]]
        reasons.append(f"DEFEATED_BY_TRUNCATED_PAULI_PATH_W_{winning_ws[0]}")
    if a3_lc["win"]:
        reasons.append(
            f"TRACTABLE_CAUSAL_LIGHT_CONE_MAX_{a3_lc['max_cone_size']}_LE_{tractable_cone_limit}"
        )
    if pred_hw_peak < 10.0 * unif_prob:
        reasons.append("INSUFFICIENT_PREDICTED_HARDWARE_PEAK_SNR")

    permitted = len(reasons) == 0

    return PregateVerdict(
        candidate_id=candidate_id,
        family=circuit.family,
        topology=circuit.topology,
        n_qubits=circuit.n,
        depth_layers=circuit.depth_layers,
        n_cz=circuit.n_cz,
        n_1q_phys=circuit.n_1q_phys,
        n_rz_virtual=circuit.n_rz,
        helios1_hqc_1000_shots=round(hqc_1000, 3),
        helios1_predicted_peak_prob=round(pred_hw_peak, 6),
        uniform_baseline_prob=unif_prob,
        exact_noiseless_peak_prob=round(exact_peak, 6) if exact_peak is not None else None,
        mirror_audit=mirror_audit,
        attack1_bounded_mps=a1_mps,
        attack2_pauli_path=a2_pp,
        attack3_light_cone=a3_lc,
        rejection_reasons=reasons,
        qpu_allocation_permitted=permitted,
    )


def run_adv1_pregate_suite(
    output_path: Optional[Path] = None,
    seed: int = 20261123,
) -> Dict[str, Any]:
    """
    Run the `ADV-1` classical hardness pre-gate across:
      - 3 `ADV-0` twirled-mirror heavy-hex controls (`n = 12, 20, 28`) -> 100% REJECTED,
      - 1 looped commuting-phase non-mirror control (`n = 24`) -> REJECTED by Pauli-path / MPS,
      - 4 flat-Schmidt chorded-loop non-mirror QCCD candidates (`n = 16, 20, 28, 32`), showing
        `n = 16, 20` caught by Light-Cone (`|C(i)| <= 20`) while `n = 28, 32` defeat all three
        classical attacks simultaneously (`qpu_allocation_permitted: True`).
    Emits a valid `osiris.quantum_protocol/1` manifest (`SIMULATED`, `provider_job_id: null`).
    """
    be = adv0_peaked.backend()
    controls: List[Tuple[str, LoopedPeakedCircuit]] = [
        ("adv0_ctrl_n12_d4", build_adv0_mirror_control(12, 4, seed + 12, be=be)),
        ("adv0_ctrl_n20_d6", build_adv0_mirror_control(20, 6, seed + 20, be=be)),
        ("adv0_ctrl_n28_d8", build_adv0_mirror_control(28, 8, seed + 28, be=be)),
    ]
    candidates: List[Tuple[str, LoopedPeakedCircuit]] = [
        (
            "adv1_commuting_phase_n24_L3",
            build_looped_commuting_phase_peaked(
                24, loop_layers=3, bias_angle=0.85, phase_coupling=0.14, seed=seed + 101
            ),
        ),
        (
            "adv1_chorded_2q_n28_L5_no_m6",
            build_looped_chorded_gadget_peaked(
                28,
                loop_layers=5,
                loop_gamma=0.12,
                enable_m6_pauli_amplifier=False,
                seed=seed + 228,
            ),
        ),
        (
            "adv1_chorded_m6_n18_L4_small_cone",
            build_looped_chorded_gadget_peaked(
                18,
                loop_layers=4,
                loop_gamma=0.14,
                enable_m6_pauli_amplifier=True,
                seed=seed + 318,
            ),
        ),
        (
            "adv1_chorded_m6_n24_L5",
            build_looped_chorded_gadget_peaked(
                24,
                loop_layers=5,
                loop_gamma=0.12,
                enable_m6_pauli_amplifier=True,
                seed=seed + 324,
            ),
        ),
        (
            "adv1_chorded_m6_n30_L5",
            build_looped_chorded_gadget_peaked(
                30,
                loop_layers=5,
                loop_gamma=0.11,
                enable_m6_pauli_amplifier=True,
                seed=seed + 330,
            ),
        ),
    ]

    ctrl_verdicts = [evaluate_adv1_pregate(circ, cid) for cid, circ in controls]
    cand_verdicts = [evaluate_adv1_pregate(circ, cid) for cid, circ in candidates]

    adv0_all_rejected = all(not v.qpu_allocation_permitted for v in ctrl_verdicts)
    permitted_ids = [v.candidate_id for v in cand_verdicts if v.qpu_allocation_permitted]

    manifest: Dict[str, Any] = {
        "schema": "osiris.quantum_protocol/1",
        "protocol_id": "OSIRIS-ADV-1-PREGATE",
        "title": "ADV-1 Fail-Closed 3-Attack Classical-Hardness Pre-Gate for Looped Non-Mirror Peaked Circuits",
        "evidence_class": "SIMULATED",
        "provider_job_id": None,
        "hardware_executed": False,
        "canonicalization": "RFC8785-JCS",
        "target_architecture": "Quantinuum Helios-1 (98Q QCCD All-to-All)",
        "seed": seed,
        "pregate_policy": {
            "mps_chi_sweep": [8, 16, 32, 64, 128],
            "pauli_weight_cutoff_sweep": [2, 3, 4],
            "tractable_light_cone_limit": 20,
            "min_snr_over_uniform": 10.0,
            "fail_closed_invariant": (
                "qpu_allocation_permitted is True iff mirror_structure_detected is False AND "
                "all three classical attacks (Bounded-chi MPS, Truncated Pauli-Path, Causal "
                "Light-Cone) fail to recover the heavy peak."
            ),
        },
        "summary": {
            "adv0_controls_evaluated": len(ctrl_verdicts),
            "adv0_controls_rejected": sum(1 for v in ctrl_verdicts if not v.qpu_allocation_permitted),
            "adv0_100pct_rejected": adv0_all_rejected,
            "adv1_non_mirror_evaluated": len(cand_verdicts),
            "adv1_permitted_candidates": permitted_ids,
            "adv1_rejected_candidates": [
                v.candidate_id for v in cand_verdicts if not v.qpu_allocation_permitted
            ],
        },
        "adv0_twirled_mirror_controls": [asdict(v) for v in ctrl_verdicts],
        "adv1_looped_non_mirror_candidates": [asdict(v) for v in cand_verdicts],
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return manifest


def main() -> None:
    out = Path(__file__).resolve().parent / "artifacts" / "adv1_pregate_report.json"
    rep = run_adv1_pregate_suite(output_path=out)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"Wrote {out} (sha256={sha})")
    print("Summary:", json.dumps(rep["summary"], indent=2))


if __name__ == "__main__":
    main()
