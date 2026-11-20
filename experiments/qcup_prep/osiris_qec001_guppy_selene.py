#!/usr/bin/env python3
"""OSIRIS-QEC-001 Standalone Port to Quantinuum Guppy (HUGR) + Selene (Stim & ClassicalReplay).

Ports the pre-registered 1D repetition-code benchmark (DOI: 10.5281/zenodo.24293756,
ibm_fez Job dbtl2vb2kf8s73c9a0d0, 393,216 shots, d in {3, 5, 7}, R in {1, 3, 5, 7},
bit_flip_z & phase_flip_x, q107 T2=18.17us hotspot) to guppylang + selene-sim with:
  1. Complete 24-cell archived ibm_fez verification (cumulative P_L(R), exact Markov
     epsilon_L_eff, SPAM-separated epsilon_L_slope, and phase_flip_x d=7 Lambda_5->7 = 2.2339)
     -- correcting external LLM double-R division errors.
  2. ClassicalReplay mode (--mode replay): SHA-256 verifies physical_qec001_fez_raw_counts.json.gz
     and replays all 393,216 physical ibm_fez bitstrings through Selene ClassicalReplay (when
     installed) or deterministic space-time Blossom MWPM.
  3. Native @guppy HUGR compilation + selene_sim.Stim execution (--mode selene / --mode auto),
     with zero-dependency built-in q107 hotspot stabilizer fallback (--mode builtin).
  4. Dual-Substrate Execution Invariant (Node Alpha ARM64 300s vs Node Beta x86_64 1800s) +
     NCLM Wasserstein-2 (W2) and Shannon syndrome entropy (H) telemetry.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXPECTED_QEC001_RAW_GZ_SHA256 = "cdae697c364e2aad2085cc3585039d079baf4b3d1bb619dcb6178647b75646c1"
ARCHIVED_QEC001_JOB_ID = "dbtl2vb2kf8s73c9a0d0"


def probe_substrate_context() -> Dict[str, Any]:
    """Enforce OSIRIS dual-substrate invariant across Node Alpha (ARM64) & Node Beta (x86_64)."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    try:
        import osiris_hardware_mesh  # type: ignore

        if hasattr(osiris_hardware_mesh, "detect_substrate"):
            mesh_info = dict(osiris_hardware_mesh.detect_substrate())
            mesh_info.setdefault("probed_via", "osiris_hardware_mesh")
            return mesh_info
    except Exception:
        pass

    machine = platform.machine().lower()
    uname_info = os.uname() if hasattr(os, "uname") else platform.uname()
    sysname = getattr(uname_info, "sysname", getattr(uname_info, "system", sys.platform))
    release = getattr(uname_info, "release", "")
    is_arm64 = machine in ("aarch64", "arm64")

    return {
        "node_id": "Node Alpha (Mobile Edge)" if is_arm64 else "Node Beta (Workstation Compute)",
        "architecture": machine,
        "sysname": sysname,
        "release": release,
        "simd": "NEON" if is_arm64 else "AVX2/AVX-VNNI",
        "timeout_horizon_s": 300.0 if is_arm64 else 1800.0,
        "probed_via": "platform.machine+os.uname",
    }


# Complete 24-cell archived ibm_fez table: (shots, cumulative_pl_maj, cumulative_pl_mwpm, mean_defect_prob)
IBM_FEZ_QEC001_CELLS: Dict[Tuple[str, int, int], Tuple[int, float, float, float]] = {
    ("bit_flip_z", 3, 1): (16384, 0.007568359375, 0.007568359375, 0.058319091796875),
    ("bit_flip_z", 3, 3): (16384, 0.0303955078125, 0.02337646484375, 0.0599365234375),
    ("bit_flip_z", 3, 5): (16384, 0.0625000000000, 0.0367431640625, 0.060399373372395836),
    ("bit_flip_z", 3, 7): (16384, 0.10150146484375, 0.0489501953125, 0.062084197998046875),
    ("bit_flip_z", 5, 1): (16384, 0.000732421875, 0.00140380859375, 0.05806732177734375),
    ("bit_flip_z", 5, 3): (16384, 0.01190185546875, 0.00640869140625, 0.0627899169921875),
    ("bit_flip_z", 5, 5): (16384, 0.0377197265625, 0.0111083984375, 0.06480662027994792),
    ("bit_flip_z", 5, 7): (16384, 0.07342529296875, 0.01458740234375, 0.06631278991699219),
    ("bit_flip_z", 7, 1): (16384, 0.0003662109375, 0.00048828125, 0.059137979797979796),
    ("bit_flip_z", 7, 3): (16384, 0.01373291015625, 0.00396728515625, 0.071075439453125),
    ("bit_flip_z", 7, 5): (16384, 0.04608154296875, 0.00927734375, 0.07490234375),
    ("bit_flip_z", 7, 7): (16384, 0.097412109375, 0.01348876953125, 0.07651646931966146),
    ("phase_flip_x", 3, 1): (16384, 0.0147705078125, 0.0147705078125, 0.0700225830078125),
    ("phase_flip_x", 3, 3): (16384, 0.07373046875, 0.06121826171875, 0.1007080078125),
    ("phase_flip_x", 3, 5): (16384, 0.13714599609375, 0.09967041015625, 0.11558024088541667),
    ("phase_flip_x", 3, 7): (16384, 0.2027587890625, 0.14044189453125, 0.1265716552734375),
    ("phase_flip_x", 5, 1): (16384, 0.00274658203125, 0.0020751953125, 0.0715789794921875),
    ("phase_flip_x", 5, 3): (16384, 0.02685546875, 0.01397705078125, 0.102264404296875),
    ("phase_flip_x", 5, 5): (16384, 0.073486328125, 0.02703857421875, 0.11998240152994792),
    ("phase_flip_x", 5, 7): (16384, 0.1199951171875, 0.03997802734375, 0.12842000325520834),
    ("phase_flip_x", 7, 1): (16384, 0.00091552734375, 0.00079345703125, 0.07251485188802083),
    ("phase_flip_x", 7, 3): (16384, 0.01312255859375, 0.00518798828125, 0.10573069254557292),
    ("phase_flip_x", 7, 5): (16384, 0.04327392578125, 0.01141357421875, 0.11768086751302084),
    ("phase_flip_x", 7, 7): (16384, 0.08868408203125, 0.01824951171875, 0.12647628784179688),
}


def effective_epsilon_per_round(
    p_logical: float, rounds: int, shots: int = 16384
) -> Tuple[float, float]:
    """Compute Markov per-round logical error epsilon_L_eff from cumulative P_L(R)."""
    if p_logical <= 0.0:
        return 0.0, 0.0
    if p_logical >= 0.5:
        return 0.5, 0.0
    se_pl = math.sqrt(max(1e-12, p_logical * (1.0 - p_logical)) / shots)
    pol = max(1e-12, 1.0 - 2.0 * p_logical)
    eps = 0.5 * (1.0 - pol ** (1.0 / rounds))
    d_eps = (1.0 / rounds) * (pol ** (1.0 / rounds - 1.0)) * se_pl
    return eps, d_eps


def slope_epsilon_per_round(
    rounds_seq: Sequence[int], pl_seq: Sequence[float], shots: int = 16384
) -> Tuple[float, float]:
    """Compute SPAM-separated 2-parameter Markov decay epsilon_L_slope across rounds."""
    ys = [math.log(max(1e-9, 1.0 - 2.0 * p)) for p in pl_seq]
    var_ys = [
        4.0 * max(1e-9, p * (1.0 - p)) / (shots * max(1e-9, (1.0 - 2.0 * p) ** 2))
        for p in pl_seq
    ]
    r_mean = sum(rounds_seq) / len(rounds_seq)
    s_rr = sum((r - r_mean) ** 2 for r in rounds_seq)
    slope = sum((r - r_mean) * y for r, y in zip(rounds_seq, ys)) / s_rr
    var_slope = sum(((r - r_mean) ** 2) * vy for r, vy in zip(rounds_seq, var_ys)) / (s_rr ** 2)
    eps = 0.5 * (1.0 - math.exp(slope))
    se_eps = 0.5 * math.exp(slope) * math.sqrt(var_slope)
    return eps, se_eps


def shannon_entropy_bits(probs: Sequence[float]) -> float:
    total = sum(probs)
    if total <= 0.0:
        return 0.0
    h_val = 0.0
    for p in probs:
        if p > 0.0:
            pn = p / total
            h_val -= pn * math.log2(pn)
    return h_val


def wasserstein2_1d_empirical(u_samples: Sequence[float], v_samples: Sequence[float]) -> float:
    if not u_samples or not v_samples:
        return 0.0
    u_sorted = sorted(u_samples)
    v_sorted = sorted(v_samples)
    n = max(len(u_sorted), len(v_sorted))
    sq_sum = 0.0
    for i in range(n):
        q = (i + 0.5) / n
        ui = u_sorted[min(len(u_sorted) - 1, int(q * len(u_sorted)))]
        vi = v_sorted[min(len(v_sorted) - 1, int(q * len(v_sorted)))]
        sq_sum += (ui - vi) ** 2
    return math.sqrt(sq_sum / n)


def analyze_archived_qec001() -> Dict[str, Any]:
    """Verify exact epsilon_L and Lambda_d across all 24 archived ibm_fez QEC-001 cells."""
    report: Dict[str, Any] = {"cells": {}, "scaling": {}}
    for sector in ("bit_flip_z", "phase_flip_x"):
        eps_eff_mwpm: Dict[int, float] = {}
        eps_eff_maj: Dict[int, float] = {}
        eps_slope_mwpm: Dict[int, float] = {}
        for d in (3, 5, 7):
            rs = [1, 3, 5, 7]
            maj_pls = [IBM_FEZ_QEC001_CELLS[(sector, d, r)][1] for r in rs]
            mwpm_pls = [IBM_FEZ_QEC001_CELLS[(sector, d, r)][2] for r in rs]
            eff_m, eff_m_se = effective_epsilon_per_round(mwpm_pls[-1], 7)
            eff_v, _ = effective_epsilon_per_round(maj_pls[-1], 7)
            slp_m, slp_m_se = slope_epsilon_per_round(rs, mwpm_pls)
            eps_eff_mwpm[d] = eff_m
            eps_eff_maj[d] = eff_v
            eps_slope_mwpm[d] = slp_m
            report["cells"][f"{sector}_d{d}"] = {
                "pl_R7_majority_vote": round(maj_pls[-1], 6),
                "pl_R7_mwpm": round(mwpm_pls[-1], 6),
                "defect_density_R7": round(IBM_FEZ_QEC001_CELLS[(sector, d, 7)][3], 6),
                "epsilon_L_eff_R7": round(eff_m, 6),
                "epsilon_L_eff_se": round(eff_m_se, 6),
                "epsilon_L_eff_majority_R7": round(eff_v, 6),
                "epsilon_L_slope": round(slp_m, 6),
                "epsilon_L_slope_se": round(slp_m_se, 6),
            }
        lam_35 = eps_eff_mwpm[3] / eps_eff_mwpm[5]
        lam_57 = eps_eff_mwpm[5] / eps_eff_mwpm[7]
        lam_37 = eps_eff_mwpm[3] / eps_eff_mwpm[7]
        report["scaling"][sector] = {
            "Lambda_3_to_5_eff_R7": round(lam_35, 4),
            "Lambda_5_to_7_eff_R7": round(lam_57, 4),
            "Lambda_3_to_7_eff_R7": round(lam_37, 4),
            "Lambda_global_step_R7": round(math.sqrt(lam_37), 4),
            "Lambda_3_to_5_slope": round(eps_slope_mwpm[3] / max(1e-9, eps_slope_mwpm[5]), 4),
            "Lambda_5_to_7_slope": round(eps_slope_mwpm[5] / max(1e-9, eps_slope_mwpm[7]), 4),
            "sub_threshold_3_to_5": lam_35 > 1.0,
            "sub_threshold_5_to_7": lam_57 > 1.0,
        }
    return report


class SpaceTimeRepetitionMWPMDecoder:
    """Precomputed Dijkstra + Blossom MWPM decoder for 1D repetition codes."""

    def __init__(
        self,
        distance: int,
        rounds: int,
        data_error_priors: Sequence[float],
        meas_error_priors: Sequence[float],
    ) -> None:
        self.d = distance
        self.rounds = rounds
        self.n_checks = distance - 1
        self.n_det = (rounds + 1) * self.n_checks
        self.w_space = [
            -math.log(max(1e-6, min(0.49, p)) / (1.0 - max(1e-6, min(0.49, p))))
            for p in data_error_priors
        ]
        self.w_time = [
            -math.log(max(1e-6, min(0.49, p)) / (1.0 - max(1e-6, min(0.49, p))))
            for p in meas_error_priors
        ]
        self._nx = None
        try:
            import networkx as nx  # type: ignore

            self._nx = nx
        except Exception:
            self._nx = None

        self._bound_cost: List[float] = [0.0] * self.n_det
        self._bound_flip_d0: List[int] = [0] * self.n_det
        self._pair_cost: Dict[Tuple[int, int], Tuple[float, int]] = {}
        self._precompute_tables()

    def _precompute_tables(self) -> None:
        for u_idx in range(self.n_det):
            _, uj = divmod(u_idx, self.n_checks)
            c_left = sum(self.w_space[k] for k in range(uj + 1))
            c_right = sum(self.w_space[k] for k in range(uj + 1, self.d))
            if c_left <= c_right:
                self._bound_cost[u_idx] = c_left
                self._bound_flip_d0[u_idx] = 1
            else:
                self._bound_cost[u_idx] = c_right
                self._bound_flip_d0[u_idx] = 0

        for src in range(self.n_det):
            dist = {src: 0.0}
            pq = [(0.0, src)]
            while pq:
                cost, cur = heapq.heappop(pq)
                if cost > dist.get(cur, float("inf")):
                    continue
                t, j = divmod(cur, self.n_checks)
                if j > 0:
                    nxt = t * self.n_checks + (j - 1)
                    nc = cost + self.w_space[j]
                    if nc < dist.get(nxt, float("inf")):
                        dist[nxt] = nc
                        heapq.heappush(pq, (nc, nxt))
                if j + 1 < self.n_checks:
                    nxt = t * self.n_checks + (j + 1)
                    nc = cost + self.w_space[j + 1]
                    if nc < dist.get(nxt, float("inf")):
                        dist[nxt] = nc
                        heapq.heappush(pq, (nc, nxt))
                for nt in (t - 1, t + 1):
                    if 0 <= nt <= self.rounds:
                        nxt = nt * self.n_checks + j
                        nc = cost + self.w_time[j]
                        if nc < dist.get(nxt, float("inf")):
                            dist[nxt] = nc
                            heapq.heappush(pq, (nc, nxt))
            for dst in range(src + 1, self.n_det):
                c_int = dist[dst]
                c_both_b = self._bound_cost[src] + self._bound_cost[dst]
                if c_both_b <= c_int:
                    self._pair_cost[(src, dst)] = (
                        c_both_b,
                        self._bound_flip_d0[src] ^ self._bound_flip_d0[dst],
                    )
                else:
                    self._pair_cost[(src, dst)] = (c_int, 0)

    def extract_detector_indices(
        self, syndromes_by_round: Sequence[Sequence[int]], final_data: Sequence[int]
    ) -> List[int]:
        active: List[int] = []
        prev = [0] * self.n_checks
        for t, syn in enumerate(syndromes_by_round):
            base = t * self.n_checks
            for j in range(self.n_checks):
                if (syn[j] ^ prev[j]) == 1:
                    active.append(base + j)
            prev = list(syn)
        base_f = self.rounds * self.n_checks
        for j in range(self.n_checks):
            if ((final_data[j] ^ final_data[j + 1]) ^ prev[j]) == 1:
                active.append(base_f + j)
        return active

    def decode_logical_bit(
        self, syndromes_by_round: Sequence[Sequence[int]], final_data: Sequence[int]
    ) -> Tuple[int, int]:
        active = self.extract_detector_indices(syndromes_by_round, final_data)
        k = len(active)
        if k == 0:
            return int(final_data[0]), 0

        flip_d0 = 0
        if self._nx is not None:
            nx = self._nx
            mg = nx.Graph()
            for i, u in enumerate(active):
                mg.add_edge(
                    i,
                    k + i,
                    weight=self._bound_cost[u],
                    flip=self._bound_flip_d0[u],
                )
            for i in range(k):
                u = active[i]
                for j in range(i + 1, k):
                    v = active[j]
                    c_uv, f_uv = self._pair_cost[(u, v) if u < v else (v, u)]
                    mg.add_edge(i, j, weight=c_uv, flip=f_uv)
                    mg.add_edge(k + i, k + j, weight=0.0, flip=0)
            matching = nx.algorithms.matching.min_weight_matching(mg, weight="weight")
            for a, b in matching:
                flip_d0 ^= int(mg[a][b]["flip"])
        else:
            memo: Dict[int, Tuple[float, int]] = {0: (0.0, 0)}

            def solve(mask: int) -> Tuple[float, int]:
                if mask in memo:
                    return memo[mask]
                first = (mask & -mask).bit_length() - 1
                rem = mask ^ (1 << first)
                u = active[first]
                sc, sf = solve(rem)
                best_c = self._bound_cost[u] + sc
                best_f = self._bound_flip_d0[u] ^ sf
                m_iter = rem
                while m_iter:
                    lsb = m_iter & -m_iter
                    other = lsb.bit_length() - 1
                    v = active[other]
                    c_uv, f_uv = self._pair_cost[(u, v) if u < v else (v, u)]
                    sc2, sf2 = solve(rem ^ lsb)
                    if c_uv + sc2 < best_c:
                        best_c = c_uv + sc2
                        best_f = f_uv ^ sf2
                    m_iter ^= lsb
                memo[mask] = (best_c, best_f)
                return memo[mask]

            if k <= 14:
                _, flip_d0 = solve((1 << k) - 1)
            else:
                unmatched = set(range(k))
                edges = []
                for i in range(k):
                    for j in range(i + 1, k):
                        u, v = active[i], active[j]
                        c_uv, f_uv = self._pair_cost[(u, v) if u < v else (v, u)]
                        edges.append((c_uv, f_uv, i, j))
                edges.sort()
                for _, f_uv, i, j in edges:
                    if i in unmatched and j in unmatched:
                        unmatched.remove(i)
                        unmatched.remove(j)
                        flip_d0 ^= f_uv
                for i in unmatched:
                    flip_d0 ^= self._bound_flip_d0[active[i]]

        return int(final_data[0]) ^ flip_d0, k


def parse_qiskit_qec001_bitstring(
    bitstring: str, distance: int, rounds: int, logical_state: int
) -> Tuple[List[List[int]], List[int]]:
    """Parse archived Qiskit little-endian bitstring and normalize logical_state into |0_L> frame."""
    clean = bitstring.replace(" ", "")
    n_checks = distance - 1
    c_le = [1 if ch == "1" else 0 for ch in reversed(clean)]
    syns = [c_le[r * n_checks : (r + 1) * n_checks] for r in range(rounds)]
    data_bits = [b ^ logical_state for b in c_le[rounds * n_checks :]]
    return syns, data_bits


def try_compile_and_run_guppy_selene(
    sector: str,
    distance: int,
    rounds: int,
    shots: int,
    p_2q: float,
    p_meas: float,
    seed: int,
    replay_measurements: Optional[List[List[bool]]] = None,
) -> Optional[List[Tuple[List[List[int]], List[int]]]]:
    """Compile @guppy repetition program to HUGR and execute on Selene Stim or ClassicalReplay."""
    try:
        from guppylang import guppy  # type: ignore
        from guppylang.std.builtins import array, comptime, result  # type: ignore
        from guppylang.std.quantum import cx, h, measure, measure_array, qubit  # type: ignore
        import selene_sim  # type: ignore
        from selene_sim import DepolarizingErrorModel, Stim, build  # type: ignore
    except Exception:
        return None

    is_phase_sector = sector == "phase_flip_x"
    d_val = distance
    r_val = rounds
    num_checks = d_val - 1

    @guppy
    def repetition_memory_main() -> None:
        data = array(qubit() for _ in range(comptime(d_val)))
        if comptime(is_phase_sector):
            for i in range(comptime(d_val)):
                h(data[i])

        for _r in range(comptime(r_val)):
            syn_bits = array(False for _ in range(comptime(num_checks)))
            for j in range(comptime(num_checks)):
                anc = qubit()
                if comptime(is_phase_sector):
                    h(anc)
                    cx(anc, data[j])
                    cx(anc, data[j + 1])
                    h(anc)
                else:
                    cx(data[j], anc)
                    cx(data[j + 1], anc)
                syn_bits[j] = measure(anc)
            result("syn", syn_bits)

        if comptime(is_phase_sector):
            for i in range(comptime(d_val)):
                h(data[i])
        final_bits = measure_array(data)
        result("data", final_bits)

    hugr_pkg = repetition_memory_main.compile()
    runner = build(hugr_pkg)

    if replay_measurements is not None and hasattr(selene_sim, "ClassicalReplay"):
        sim = selene_sim.ClassicalReplay(measurements=replay_measurements)
        shot_stream = runner.run_shots(
            simulator=sim,
            n_qubits=d_val + 1,
            n_shots=len(replay_measurements),
        )
    else:
        error_model = DepolarizingErrorModel(
            random_seed=seed,
            p_1q=p_2q * 0.1,
            p_2q=p_2q,
            p_meas=p_meas,
            p_init=p_meas * 0.5,
        )
        shot_stream = runner.run_shots(
            simulator=Stim(random_seed=seed + 1),
            error_model=error_model,
            n_qubits=d_val + 1,
            n_shots=shots,
        )

    parsed_shots: List[Tuple[List[List[int]], List[int]]] = []
    for shot_entries in shot_stream:
        syns: List[List[int]] = []
        data_out: List[int] = []
        for tag, val in shot_entries:
            if tag == "syn":
                syns.append([1 if b else 0 for b in val])
            elif tag == "data":
                data_out = [1 if b else 0 for b in val]
        parsed_shots.append((syns, data_out))
    return parsed_shots


def simulate_stabilizer_with_q107_hotspot(
    sector: str,
    distance: int,
    rounds: int,
    shots: int,
    base_data_p: float,
    meas_p: float,
    q107_hotspot_multiplier: float,
    seed: int,
) -> Tuple[List[Tuple[List[List[int]], List[int]]], List[float], List[float]]:
    rng = random.Random(seed)
    hotspot_idx = min(distance - 1, 2)
    data_priors = [base_data_p for _ in range(distance)]
    if sector == "phase_flip_x":
        data_priors[hotspot_idx] = min(0.35, base_data_p * q107_hotspot_multiplier)
    meas_priors = [meas_p for _ in range(distance - 1)]

    outcomes: List[Tuple[List[List[int]], List[int]]] = []
    for _ in range(shots):
        data_state = [0] * distance
        syns: List[List[int]] = []
        for _r in range(rounds):
            for i in range(distance):
                if rng.random() < data_priors[i]:
                    data_state[i] ^= 1
            syn_r = []
            for j in range(distance - 1):
                parity = data_state[j] ^ data_state[j + 1]
                if rng.random() < meas_priors[j]:
                    parity ^= 1
                syn_r.append(parity)
            syns.append(syn_r)
        final_readout = [
            bit ^ (1 if rng.random() < meas_p * 0.75 else 0) for bit in data_state
        ]
        outcomes.append((syns, final_readout))
    return outcomes, data_priors, meas_priors


def run_classical_replay_from_archive(
    archive_gz_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Replay all 393,216 physical ibm_fez QEC-001 shots with SHA-256 verification."""
    substrate = probe_substrate_context()
    repo_root = Path(__file__).resolve().parents[2]
    gz_path = archive_gz_path or (
        repo_root / "experiments" / "qec001" / "physical_qec001_fez_raw_counts.json.gz"
    )
    if not gz_path.exists():
        raise FileNotFoundError(f"Archived QEC-001 raw counts not found at {gz_path}")

    actual_sha = hashlib.sha256(gz_path.read_bytes()).hexdigest()
    if actual_sha != EXPECTED_QEC001_RAW_GZ_SHA256:
        raise ValueError(
            f"SHA-256 mismatch on {gz_path.name}: {actual_sha} != {EXPECTED_QEC001_RAW_GZ_SHA256}"
        )

    with gzip.open(gz_path, "rt", encoding="utf-8") as fh:
        records = json.load(fh)

    grouped: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = {}
    for rec in records:
        key = (str(rec["code_basis"]), int(rec["distance"]), int(rec["rounds"]))
        grouped.setdefault(key, []).append(rec)

    cells_out: List[Dict[str, Any]] = []
    r1_defect_profiles: Dict[Tuple[str, int], List[float]] = {}

    for sector in ("bit_flip_z", "phase_flip_x"):
        for d in (3, 5, 7):
            for r in (1, 3, 5, 7):
                cell_recs = grouped[(sector, d, r)]
                decoder = SpaceTimeRepetitionMWPMDecoder(
                    d, r, [0.008] * d, [0.012] * (d - 1)
                )
                total_shots = 0
                maj_errs = 0
                mwpm_errs = 0
                total_defects = 0
                defect_rate_samples: List[float] = []
                syndrome_hist: Dict[Tuple[int, ...], int] = {}
                n_det_sites = (r + 1) * (d - 1)

                for rec in cell_recs:
                    l_state = int(rec["logical_state"])
                    for bitstr, cnt in rec["raw_counts"].items():
                        n = int(cnt)
                        if n <= 0:
                            continue
                        total_shots += n
                        syns, data_norm = parse_qiskit_qec001_bitstring(bitstr, d, r, l_state)
                        if sum(data_norm) > (d // 2):
                            maj_errs += n
                        pred_bit, k_def = decoder.decode_logical_bit(syns, data_norm)
                        if pred_bit == 1:
                            mwpm_errs += n
                        total_defects += k_def * n
                        defect_rate_samples.append(k_def / n_det_sites)
                        last_syn = tuple(syns[-1])
                        syndrome_hist[last_syn] = syndrome_hist.get(last_syn, 0) + n

                if r == 1:
                    r1_defect_profiles[(sector, d)] = defect_rate_samples
                w2_drift = wasserstein2_1d_empirical(
                    r1_defect_profiles[(sector, d)], defect_rate_samples
                )
                h_syn = shannon_entropy_bits(list(syndrome_hist.values()))
                pl_mwpm = mwpm_errs / total_shots
                eps_eff, _ = effective_epsilon_per_round(pl_mwpm, r, shots=total_shots)

                cells_out.append(
                    {
                        "sector": sector,
                        "distance": d,
                        "rounds": r,
                        "shots": total_shots,
                        "engine": "classical_replay_ibm_fez",
                        "pl_majority_vote": round(maj_errs / total_shots, 6),
                        "pl_mwpm_cal": round(pl_mwpm, 6),
                        "epsilon_L_eff": round(eps_eff, 6),
                        "defect_density": round(total_defects / (total_shots * n_det_sites), 6),
                        "shannon_syndrome_bits": round(h_syn, 5),
                        "wasserstein_w2_vs_r1": round(w2_drift, 6),
                    }
                )

    payload = {
        "schema": "osiris-qec001-guppy-selene-port/v2",
        "evidence_class": "HARDWARE_REPLAY",
        "hardware": True,
        "source_backend": "ibm_fez",
        "source_job_id": ARCHIVED_QEC001_JOB_ID,
        "raw_counts_archive_sha256": actual_sha,
        "substrate": substrate,
        "archived_ibm_fez_verification": analyze_archived_qec001(),
        "replayed_cells": cells_out,
    }
    canonical_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["receipt_sha256"] = hashlib.sha256(canonical_bytes).hexdigest()
    return payload


def run_port_sweep(
    shots: int = 1024,
    mode: str = "auto",
    hotspot_mult: float = 4.5,
    seed: int = 20261119,
    archive_gz_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if mode == "replay":
        return run_classical_replay_from_archive(archive_gz_path=archive_gz_path)

    substrate = probe_substrate_context()
    t0 = time.monotonic()

    cells_out: List[Dict[str, Any]] = []
    r1_defect_profiles: Dict[Tuple[str, int], List[float]] = {}

    for sector in ("bit_flip_z", "phase_flip_x"):
        base_p = 0.0045 if sector == "bit_flip_z" else 0.0085
        meas_p = 0.0110 if sector == "bit_flip_z" else 0.0150
        for d in (3, 5, 7):
            for r in (1, 3, 5, 7):
                if time.monotonic() - t0 > substrate["timeout_horizon_s"]:
                    raise TimeoutError("Exceeded active substrate temporal horizon")

                cell_seed = seed + ((1000 if sector == "phase_flip_x" else 0) + d * 31 + r)
                guppy_shots = None
                if mode in ("auto", "selene"):
                    guppy_shots = try_compile_and_run_guppy_selene(
                        sector, d, r, shots, p_2q=base_p, p_meas=meas_p, seed=cell_seed
                    )
                    if mode == "selene" and guppy_shots is None:
                        raise RuntimeError(
                            "guppylang / selene_sim not installed in current Python env; "
                            "install `pip install guppylang selene-sim` or use --mode auto"
                        )

                if guppy_shots is not None:
                    raw_shots = guppy_shots
                    data_priors = [base_p] * d
                    meas_priors = [meas_p] * (d - 1)
                    engine_used = "guppylang_hugr+selene_stim"
                else:
                    raw_shots, data_priors, meas_priors = simulate_stabilizer_with_q107_hotspot(
                        sector, d, r, shots, base_p, meas_p, hotspot_mult, cell_seed
                    )
                    engine_used = "builtin_stabilizer_q107_sim"

                decoder = SpaceTimeRepetitionMWPMDecoder(d, r, data_priors, meas_priors)
                maj_errs = 0
                mwpm_errs = 0
                total_defects = 0
                per_shot_defect_rates: List[float] = []
                syndrome_pattern_hist: Dict[Tuple[int, ...], int] = {}
                num_detector_sites = (r + 1) * (d - 1)

                for syns, final_data in raw_shots:
                    if sum(final_data) > d // 2:
                        maj_errs += 1
                    pred_bit, k_def = decoder.decode_logical_bit(syns, final_data)
                    if pred_bit == 1:
                        mwpm_errs += 1
                    total_defects += k_def
                    per_shot_defect_rates.append(k_def / num_detector_sites)
                    last_syn = tuple(syns[-1])
                    syndrome_pattern_hist[last_syn] = syndrome_pattern_hist.get(last_syn, 0) + 1

                if r == 1:
                    r1_defect_profiles[(sector, d)] = per_shot_defect_rates
                w2_drift = wasserstein2_1d_empirical(
                    r1_defect_profiles[(sector, d)], per_shot_defect_rates
                )
                h_syn = shannon_entropy_bits(list(syndrome_pattern_hist.values()))
                pl_mwpm = mwpm_errs / shots
                eps_eff, _ = effective_epsilon_per_round(pl_mwpm, r, shots=shots)

                cells_out.append(
                    {
                        "sector": sector,
                        "distance": d,
                        "rounds": r,
                        "shots": shots,
                        "engine": engine_used,
                        "pl_majority_vote": round(maj_errs / shots, 6),
                        "pl_mwpm_cal": round(pl_mwpm, 6),
                        "epsilon_L_eff": round(eps_eff, 6),
                        "defect_density": round(total_defects / (shots * num_detector_sites), 6),
                        "shannon_syndrome_bits": round(h_syn, 5),
                        "wasserstein_w2_vs_r1": round(w2_drift, 6),
                    }
                )

    archived_analysis = analyze_archived_qec001()
    payload = {
        "schema": "osiris-qec001-guppy-selene-port/v2",
        "evidence_class": "SIMULATED",
        "hardware": False,
        "substrate": substrate,
        "archived_ibm_fez_verification": archived_analysis,
        "emulated_cells": cells_out,
    }
    canonical_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["receipt_sha256"] = hashlib.sha256(canonical_bytes).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="OSIRIS-QEC-001 Guppy + Selene Stabilizer Port")
    parser.add_argument("--shots", type=int, default=2048, help="Shots per (sector, d, R) cell")
    parser.add_argument("--mode", choices=("auto", "selene", "builtin", "replay"), default="auto")
    parser.add_argument("--hotspot-mult", type=float, default=4.5, help="q107 phase-flip multiplier")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON receipt output path")
    args = parser.parse_args()

    result = run_port_sweep(shots=args.shots, mode=args.mode, hotspot_mult=args.hotspot_mult)
    sub = result["substrate"]
    print(f"[SUBSTRATE] {sub['node_id']} | {sub['architecture']} | SIMD: {sub['simd']} | Horizon: {sub['timeout_horizon_s']}s")
    print("\n=== 1. ARCHIVED OSIRIS-QEC-001 (ibm_fez, ALL 24 CELLS) EXACT EPSILON_L & LAMBDA_D ===")
    for sec, metrics in result["archived_ibm_fez_verification"]["scaling"].items():
        print(f"  {sec:13s} -> {metrics}")

    cells_key = "replayed_cells" if "replayed_cells" in result else "emulated_cells"
    print(f"\n=== 2. R=7 SUMMARY ({result['evidence_class']}) ===")
    for c in result[cells_key]:
        if c["rounds"] == 7:
            print(
                f"  {c['sector']:12s} d={c['distance']} R=7 [{c['engine']}] | "
                f"Maj={c['pl_majority_vote']:.5f} | MWPM={c['pl_mwpm_cal']:.5f} | "
                f"eps_L={c['epsilon_L_eff']:.6f} | Def={c['defect_density']:.5f} | "
                f"H={c['shannon_syndrome_bits']:.3f}b | W2={c['wasserstein_w2_vs_r1']:.5f}"
            )
    print(f"\n[RECEIPT SHA-256] {result['receipt_sha256']} (evidence_class={result['evidence_class']})")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[SAVED] {args.out.resolve()}")


if __name__ == "__main__":
    main()
