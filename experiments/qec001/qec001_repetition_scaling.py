"""
qec001_repetition_scaling.py — Multi-round repetition-code logical-error-per-cycle
scaling benchmark (`OSIRIS-QEC-001`) across code distances d in {3, 5, 7} and
syndrome extraction cycles R in {1, 3, 5, 7} on IBM Heron r2 heavy-hex chains.

Implements:
  1. Layout-locked Z-basis (bit-flip) and X-basis (phase-flip, with optional Hahn
     spin-echo DD during ancilla readout/reset) multi-round syndrome circuits using
     native mid-circuit `measure` + `reset` into a single flat ClassicalRegister.
  2. Exact space-time detector extraction sigma_{r, j} = m_{r, j} ^ m_{r-1, j}
     (with m_{-1, j} = 0 and final data parity m_{R, j} = d_j ^ d_{j+1}).
  3. Exact Minimum-Weight Perfect Matching (MWPM) via Edmonds' Blossom algorithm
     (`networkx.algorithms.matching.min_weight_matching`) on the space-time syndrome
     graph, compared against un-decoded majority vote.
  4. Logical error per syndrome cycle epsilon_L(d, R) = 0.5 * (1 - (1 - 2*P_L)^(1/R))
     and error-suppression factor Lambda_{d -> d+2} = epsilon_L(d) / epsilon_L(d+2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

# Optimal 13-transmon contiguous heavy-hex chain on `ibm_fez`:
#   mean CZ error = 0.00226 (max 0.00274), mean ancilla readout = 0.00724, mean data readout = 0.00970
DEFAULT_FEZ_CHAIN_13Q: Tuple[int, ...] = (
    111, 110, 109, 108, 107, 106, 105, 117, 125, 124, 123, 122, 121
)

DISTANCES: Tuple[int, ...] = (3, 5, 7)
ROUNDS_LIST: Tuple[int, ...] = (1, 3, 5, 7)
CODE_BASES: Tuple[str, ...] = ("bit_flip_z", "phase_flip_x")


def subchain_for_distance(chain_13q: Sequence[int], d: int) -> Tuple[int, ...]:
    """Return the centered (2*d - 1)-qubit contiguous subchain so d=3, 5, 7 share center qubits."""
    if len(chain_13q) != 13:
        raise ValueError(f"Expected 13-qubit chain, got {len(chain_13q)}")
    if d not in DISTANCES:
        raise ValueError(f"Unsupported distance d={d}; expected one of {DISTANCES}")
    width = 2 * d - 1
    start = (13 - width) // 2  # d=3 -> 4..8 (5Q), d=5 -> 2..10 (9Q), d=7 -> 0..12 (13Q)
    return tuple(int(q) for q in chain_13q[start : start + width])


def build_repetition_memory_circuit(
    distance: int,
    rounds: int,
    code_basis: str = "bit_flip_z",
    logical_state: int = 0,
    dynamical_decoupling: bool = True,
) -> QuantumCircuit:
    """
    Build a (2*d - 1)-qubit multi-round repetition-code memory circuit:
      - Even wires 0, 2, ..., 2d-2 are data qubits D_0 .. D_{d-1}.
      - Odd wires  1, 3, ..., 2d-3 are syndrome ancillas A_0 .. A_{d-2}.
      - Single flat ClassicalRegister `c` of length `rounds * (d - 1) + d`:
          c[r*(d-1) + j]      = round-r syndrome bit on A_j (0 <= r < rounds)
          c[rounds*(d-1) + k] = final data measurement on D_k (0 <= k < d)
    """
    if distance not in DISTANCES:
        raise ValueError(f"distance must be in {DISTANCES}, got {distance}")
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    if code_basis not in CODE_BASES:
        raise ValueError(f"code_basis must be in {CODE_BASES}, got {code_basis}")
    if logical_state not in (0, 1):
        raise ValueError(f"logical_state must be 0 or 1, got {logical_state}")

    n_qubits = 2 * distance - 1
    n_checks = distance - 1
    n_clbits = rounds * n_checks + distance

    qr = QuantumRegister(n_qubits, "q")
    cr = ClassicalRegister(n_clbits, "c")
    qc = QuantumCircuit(qr, cr, name=f"qec001_{code_basis}_d{distance}_r{rounds}_s{logical_state}")

    data_wires = [2 * k for k in range(distance)]
    anc_wires = [2 * j + 1 for j in range(n_checks)]

    # 1. Logical state preparation
    if logical_state == 1:
        for w in data_wires:
            qc.x(qr[w])
    if code_basis == "phase_flip_x":
        # |0_L> -> |+_L> = |+>^d, |1_L> -> |-_L> = |->^d
        for w in data_wires:
            qc.h(qr[w])
    qc.barrier()

    # 2. R syndrome extraction cycles
    for r in range(rounds):
        if code_basis == "bit_flip_z":
            # Measure Z_j Z_{j+1} into ancilla A_j via CNOT(D_j -> A_j), CNOT(D_{j+1} -> A_j)
            for j, aw in enumerate(anc_wires):
                qc.cx(qr[data_wires[j]], qr[aw])
            qc.barrier()
            for j, aw in enumerate(anc_wires):
                qc.cx(qr[data_wires[j + 1]], qr[aw])
        else:
            # Measure X_j X_{j+1} into ancilla A_j: H(A_j), CNOT(A_j -> D_j), CNOT(A_j -> D_{j+1}), H(A_j)
            for aw in anc_wires:
                qc.h(qr[aw])
            for j, aw in enumerate(anc_wires):
                qc.cx(qr[aw], qr[data_wires[j]])
            qc.barrier()
            for j, aw in enumerate(anc_wires):
                qc.cx(qr[aw], qr[data_wires[j + 1]])
            for aw in anc_wires:
                qc.h(qr[aw])

        qc.barrier()
        for j, aw in enumerate(anc_wires):
            qc.measure(qr[aw], cr[r * n_checks + j])
        if dynamical_decoupling and code_basis == "phase_flip_x":
            # Balanced X-X Hahn echo on data qubits while ancillas undergo readout/reset
            for w in data_wires:
                qc.x(qr[w])
                qc.x(qr[w])
        if r < rounds - 1:
            for aw in anc_wires:
                qc.reset(qr[aw])
        qc.barrier()

    # 3. Final destructive data readout in the logical basis
    if code_basis == "phase_flip_x":
        for w in data_wires:
            qc.h(qr[w])
    base = rounds * n_checks
    for k, w in enumerate(data_wires):
        qc.measure(qr[w], cr[base + k])

    return qc


def parse_bitstring(bitstring: str, distance: int, rounds: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a Qiskit little-endian ClassicalRegister bitstring (`c_{N-1}...c_1 c_0`) into:
      - `syndromes`: shape `(rounds, distance - 1)` uint8 array `m[r, j]`
      - `data_bits`: shape `(distance,)` uint8 array `d_final[k]`
    """
    clean = bitstring.replace(" ", "")
    n_checks = distance - 1
    expected = rounds * n_checks + distance
    if len(clean) != expected:
        raise ValueError(f"Bitstring length {len(clean)} != expected {expected}")
    # Reverse so index i corresponds to classical bit c[i]
    c_le = np.fromiter((1 if ch == "1" else 0 for ch in reversed(clean)), dtype=np.uint8, count=expected)
    syndromes = c_le[: rounds * n_checks].reshape((rounds, n_checks))
    data_bits = c_le[rounds * n_checks :]
    return syndromes, data_bits


def compute_detectors(syndromes: np.ndarray, data_bits: np.ndarray) -> np.ndarray:
    """
    Compute the `(rounds + 1, distance - 1)` space-time detector array:
      sigma[0, j]      = m[0, j]
      sigma[r, j]      = m[r, j] ^ m[r - 1, j]   for 1 <= r < rounds
      sigma[rounds, j] = (d_final[j] ^ d_final[j + 1]) ^ m[rounds - 1, j]
    """
    rounds, n_checks = syndromes.shape
    final_checks = (data_bits[:-1] ^ data_bits[1:]).reshape((1, n_checks))
    all_checks = np.vstack([syndromes, final_checks])
    prev = np.vstack([np.zeros((1, n_checks), dtype=np.uint8), all_checks[:-1]])
    return (all_checks ^ prev).astype(np.uint8)


@dataclass(frozen=True)
class PrecomputedMWPMDecoder:
    """Exact precomputed all-pairs shortest-path + Edmonds Blossom MWPM decoder for 1D repetition codes."""

    distance: int
    rounds: int
    pair_cost: np.ndarray        # shape (N_det, N_det)
    pair_flip_d0: np.ndarray     # shape (N_det, N_det) bool: whether path crosses left boundary / D_0
    bound_cost: np.ndarray       # shape (N_det,) minimum boundary cost
    bound_flip_d0: np.ndarray    # shape (N_det,) bool: True iff nearest boundary is left boundary (flips D_0)


def build_mwpm_decoder(
    distance: int,
    rounds: int,
    p_data: float = 0.008,
    p_meas: float = 0.012,
    p_hook: float = 0.003,
) -> PrecomputedMWPMDecoder:
    """Construct the space-time detector graph for `(distance, rounds)` and precompute shortest paths."""
    n_checks = distance - 1
    n_rows = rounds + 1
    n_det = n_rows * n_checks

    def _w(p: float) -> float:
        p_clip = min(0.499, max(1e-6, float(p)))
        return -math.log(p_clip / (1.0 - p_clip))

    w_space = _w(p_data)
    w_time = _w(p_meas)
    w_hook = _w(p_hook)

    g = nx.Graph()
    for idx in range(n_det):
        g.add_node(idx)
    left_b = "B_LEFT"
    right_b = "B_RIGHT"
    g.add_node(left_b)
    g.add_node(right_b)
    # Zero-weight edge between left and right virtual boundaries? NO: connecting left_b to right_b
    # with zero weight would let two interior defects match via left_b--right_b at cost < interior!
    # Instead, boundary costs are tracked per detector separately (standard Stim/PyMatching formulation).

    for r in range(n_rows):
        # Left boundary edge flips D_0 (`flip=1`); right boundary flips D_{d-1} (`flip=0` on D_0)
        u_left = r * n_checks + 0
        u_right = r * n_checks + (n_checks - 1)
        g.add_edge(u_left, left_b, weight=w_space, flip=1)
        g.add_edge(u_right, right_b, weight=w_space, flip=0)

        # Bulk spatial edges (data qubit D_{j+1}, does not flip D_0)
        for j in range(n_checks - 1):
            u = r * n_checks + j
            v = r * n_checks + (j + 1)
            g.add_edge(u, v, weight=w_space, flip=0)

        # Temporal edges (ancilla measurement / reset error, does not flip D_0)
        if r < n_rows - 1:
            for j in range(n_checks):
                u = r * n_checks + j
                v = (r + 1) * n_checks + j
                g.add_edge(u, v, weight=w_time, flip=0)
            # CNOT hook space-time diagonal edges: (r, j) <-> (r+1, j-1)
            for j in range(1, n_checks):
                u = r * n_checks + j
                v = (r + 1) * n_checks + (j - 1)
                g.add_edge(u, v, weight=w_hook, flip=0)

    # Interior graph (excluding virtual boundary nodes) for defect-to-defect shortest paths
    g_int = g.subgraph(range(n_det)).copy()
    pair_cost = np.full((n_det, n_det), np.inf, dtype=float)
    pair_flip = np.zeros((n_det, n_det), dtype=bool)
    for src in range(n_det):
        lengths, paths = nx.single_source_dijkstra(g_int, src, weight="weight")
        for dst, cost in lengths.items():
            pair_cost[src, dst] = float(cost)
            fl = 0
            path = paths[dst]
            for a, b in zip(path[:-1], path[1:]):
                fl ^= int(g_int[a][b]["flip"])
            pair_flip[src, dst] = bool(fl)

    # Shortest path from each detector to left_b vs right_b through interior
    len_left = nx.single_source_dijkstra_path_length(
        g.subgraph(list(range(n_det)) + [left_b]), left_b, weight="weight"
    )
    len_right = nx.single_source_dijkstra_path_length(
        g.subgraph(list(range(n_det)) + [right_b]), right_b, weight="weight"
    )
    bound_cost = np.empty(n_det, dtype=float)
    bound_flip = np.empty(n_det, dtype=bool)
    for u in range(n_det):
        cl = float(len_left[u])
        cr = float(len_right[u])
        if cl <= cr:
            bound_cost[u] = cl
            bound_flip[u] = True   # Matched to left boundary => flips tracked observable D_0
        else:
            bound_cost[u] = cr
            bound_flip[u] = False  # Matched to right boundary => does not flip D_0

    # Also check whether matching u and v via BOTH boundaries (cl[u]+cl[v] or cr[u]+cr[v] etc.) is cheaper
    for u in range(n_det):
        for v in range(u + 1, n_det):
            via_bounds = bound_cost[u] + bound_cost[v]
            if via_bounds < pair_cost[u, v]:
                pair_cost[u, v] = pair_cost[v, u] = via_bounds
                pair_flip[u, v] = pair_flip[v, u] = bool(bound_flip[u] ^ bound_flip[v])

    return PrecomputedMWPMDecoder(
        distance=distance,
        rounds=rounds,
        pair_cost=pair_cost,
        pair_flip_d0=pair_flip,
        bound_cost=bound_cost,
        bound_flip_d0=bound_flip,
    )


def decode_shot_mwpm(decoder: PrecomputedMWPMDecoder, syndromes: np.ndarray, data_bits: np.ndarray) -> int:
    """Decode a single shot's logical value (`0` or `1`) by correcting tracked observable `data_bits[0]`."""
    det = compute_detectors(syndromes, data_bits).ravel()
    active = np.flatnonzero(det)
    k = int(active.size)
    if k == 0:
        return int(data_bits[0])
    if k == 1:
        u = int(active[0])
        return int(data_bits[0]) ^ int(decoder.bound_flip_d0[u])
    if k == 2:
        u, v = int(active[0]), int(active[1])
        c_pair = decoder.pair_cost[u, v]
        c_bnd = decoder.bound_cost[u] + decoder.bound_cost[v]
        if c_bnd < c_pair:
            flip = int(decoder.bound_flip_d0[u]) ^ int(decoder.bound_flip_d0[v])
        else:
            flip = int(decoder.pair_flip_d0[u, v])
        return int(data_bits[0]) ^ flip

    # General Edmonds Blossom MWPM with k boundary mirror nodes (0..k-1 real, k..2k-1 boundary mirrors)
    mg = nx.Graph()
    for i in range(k):
        u = int(active[i])
        mg.add_edge(i, k + i, weight=float(decoder.bound_cost[u]), flip=int(decoder.bound_flip_d0[u]))
    for i in range(k):
        u = int(active[i])
        for j in range(i + 1, k):
            v = int(active[j])
            mg.add_edge(i, j, weight=float(decoder.pair_cost[u, v]), flip=int(decoder.pair_flip_d0[u, v]))
            # Mirror-to-mirror zero-cost edge so unmatched mirrors pair up freely
            mg.add_edge(k + i, k + j, weight=0.0, flip=0)

    matching = nx.algorithms.matching.min_weight_matching(mg, weight="weight")
    total_flip = 0
    for a, b in matching:
        total_flip ^= int(mg[a][b]["flip"])
    return int(data_bits[0]) ^ total_flip


def evaluate_repetition_counts(
    counts: Mapping[str, int],
    distance: int,
    rounds: int,
    logical_state: int,
    decoder: PrecomputedMWPMDecoder | None = None,
) -> Dict[str, float]:
    """Evaluate MWPM logical error rate, naive majority-vote error rate, and logical error per cycle."""
    dec = decoder or build_mwpm_decoder(distance, rounds)
    total_shots = 0
    mwpm_errors = 0
    majority_errors = 0
    det_sum = 0.0
    n_det_sites = (rounds + 1) * (distance - 1)

    for bitstr, cnt in counts.items():
        n = int(cnt)
        if n <= 0:
            continue
        total_shots += n
        syn, dat = parse_bitstring(bitstr, distance, rounds)
        det = compute_detectors(syn, dat)
        det_sum += float(det.sum()) * n

        pred_mwpm = decode_shot_mwpm(dec, syn, dat)
        if pred_mwpm != logical_state:
            mwpm_errors += n

        pred_maj = 1 if int(dat.sum()) > (distance // 2) else 0
        if pred_maj != logical_state:
            majority_errors += n

    if total_shots == 0:
        raise ValueError("Empty shot histogram")

    p_err = mwpm_errors / total_shots
    p_err_se = math.sqrt(max(p_err * (1.0 - p_err), 0.5 / total_shots) / total_shots)
    p_maj = majority_errors / total_shots
    eps_cycle = logical_error_per_cycle(p_err, rounds)
    eps_cycle_hi = logical_error_per_cycle(min(0.499999, p_err + p_err_se), rounds)
    eps_cycle_lo = logical_error_per_cycle(max(0.0, p_err - p_err_se), rounds)

    return {
        "shots": float(total_shots),
        "mwpm_logical_errors": float(mwpm_errors),
        "p_logical_mwpm": p_err,
        "p_logical_mwpm_se": p_err_se,
        "p_logical_majority": p_maj,
        "epsilon_per_cycle": eps_cycle,
        "epsilon_per_cycle_se": 0.5 * abs(eps_cycle_hi - eps_cycle_lo),
        "mean_defect_probability": det_sum / (total_shots * n_det_sites),
    }


def logical_error_per_cycle(p_logical: float, rounds: int) -> float:
    """Standard logical error per syndrome cycle: epsilon_L = 0.5 * (1 - (1 - 2 * P_L)^(1 / R))."""
    if p_logical <= 0.0:
        return 0.0
    if p_logical >= 0.5:
        return 0.5
    return 0.5 * (1.0 - (1.0 - 2.0 * p_logical) ** (1.0 / float(rounds)))
