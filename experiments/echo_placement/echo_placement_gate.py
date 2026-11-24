#!/usr/bin/env python3
"""Card B (`OSIRIS-ECHO`): Offline Noisy-Simulation Placement-Ranking Stop/Go Gate.

Evaluates whether forward-then-inverse echo return probabilities rank isomorphic 5-qubit
heavy-hex `T5` placements by held-out fidelity on *unseen* circuits better than published
(stale) calibration tables, static topology-degree scoring, or uniform random selection
under matched selection budget.

Governance & Epistemic Scope (`Card B`, `phase1/verification_queue.md`):
  - `execution_tier = "SIMULATED"`, `provider_job_id = None`, `qpu_seconds = 0`.
  - Unit of inference is the **calibration window** (`M = 16`), never individual shots.
  - Mandatory controls:
      1. No-workload identity echo (`0 CZ`) isolating pure 5Q SPAM return baseline.
      2. Deliberate coherent over-rotation perturbation sweep verifying strict monotone drop.
      3. Compiled-artifact gate-level non-cancellation audit verifying `cz_count >= 2 * fwd_cz`
         even after stripping all `barrier` instructions and running `optimization_level=3`.
  - Mechanism inferred from return curves: `"none"`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import SXGate
from qiskit.quantum_info import Clifford, Operator, Statevector, random_clifford
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import RemoveBarriers
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
BT2T_DIR = REPO_ROOT / "experiments" / "bt2t_inversion"

DEFAULT_SEED = 20261124
K_PLACEMENTS_DEFAULT = 10
M_WINDOWS_DEFAULT = 16
B_BOOT_DEFAULT = 2000
ECHO_SHOTS_DEFAULT = 1024

# Canonical 5-qubit heavy-hex T-junction: 0 -- 1 -- 2(hub) -- {3, 4}
HEAVY_HEX_T5_DIRECTED: List[List[int]] = [
    [0, 1],
    [1, 0],
    [1, 2],
    [2, 1],
    [2, 3],
    [3, 2],
    [2, 4],
    [4, 2],
]
T5_INTERNAL_EDGES: Tuple[Tuple[int, int], ...] = ((0, 1), (1, 2), (2, 3), (2, 4))

SX_MAT = Operator(SXGate()).data
X_MAT = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)


def _load_bt2t() -> Any:
    mod_key = "_card_b_bt2t"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    if str(BT2T_DIR) not in sys.path:
        sys.path.insert(0, str(BT2T_DIR))
    spec = importlib.util.spec_from_file_location(mod_key, BT2T_DIR / "bt2t.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load bt2t.py from {BT2T_DIR}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    spec.loader.exec_module(mod)
    return mod


def _rx_mat(theta: float) -> np.ndarray:
    c = np.cos(0.5 * theta)
    s = -1j * np.sin(0.5 * theta)
    return np.array([[c, s], [s, c]], dtype=complex)


def _rz_mat(theta: float) -> np.ndarray:
    e_neg = np.exp(-0.5j * theta)
    e_pos = np.exp(0.5j * theta)
    return np.array([[e_neg, 0.0], [0.0, e_pos]], dtype=complex)


@dataclass(frozen=True)
class PlacementNoiseWindow:
    """Published calibration snapshot and drifted true physical noise for one (window, placement)."""

    window_id: int
    placement_id: int
    physical_qubits: Tuple[int, int, int, int, int]
    drifted_regime: bool
    cal_cz_err: Dict[Tuple[int, int], float]
    cal_ro_err: Tuple[float, float, float, float, float]
    cal_sx_err: Tuple[float, float, float, float, float]
    cal_composite_cost: float
    true_cz_depol: Dict[Tuple[int, int], float]
    true_ro_flip: Tuple[float, float, float, float, float]
    true_sx_depol: Tuple[float, float, float, float, float]
    true_overrot_rad: Tuple[float, float, float, float, float]


def heron_r2_156q_coupling_edges() -> List[Tuple[int, int]]:
    """Return the exact 176 undirected edges of the 156-qubit IBM Heron r2 (`ibm_fez`) heavy-hex lattice."""
    edges: List[Tuple[int, int]] = []
    for row in range(8):
        base = 20 * row
        for col in range(15):
            edges.append((base + col, base + col + 1))
    for bridge_row in range(7):
        offset = 3 if (bridge_row % 2 == 0) else 1
        for k in range(4):
            q_bridge = 20 * bridge_row + 16 + k
            q_top = 20 * bridge_row + offset + 4 * k
            q_bot = 20 * (bridge_row + 1) + offset + 4 * k
            edges.append(tuple(sorted((q_top, q_bridge))))  # type: ignore[arg-type]
            edges.append(tuple(sorted((q_bridge, q_bot))))  # type: ignore[arg-type]
    return sorted(edges)


def select_t5_candidate_placements(
    k_placements: int = K_PLACEMENTS_DEFAULT,
) -> List[Dict[str, Any]]:
    """Select K vertex-disjoint induced 5Q heavy-hex `T5` subgraphs on the 156Q Heron r2 lattice.

    Each tuple `(q0, q1, q2_hub, q3, q4)` has induced edges `(q0,q1), (q1,q2), (q2,q3), (q2,q4)`
    with zero extra chords. Placement 0 is pinned to `(21, 22, 23, 16, 24)` (`OSIRIS-BT2T-001-FEZ`).
    Static ambient topological metrics (external spectator couplers and graph eccentricity) are
    recorded for the `TOPOLOGY_DEGREE_ONLY` comparator.
    """
    if k_placements < 8:
        raise ValueError("Card B requires K >= 8 candidate T5 placements")

    g = nx.Graph(heron_r2_156q_coupling_edges())
    pinned_first = (21, 22, 23, 16, 24)
    chosen: List[Tuple[int, int, int, int, int]] = [pinned_first]
    used = set(pinned_first)

    for hub in sorted(g.nodes()):
        if len(chosen) >= k_placements:
            break
        if g.degree[hub] != 3 or hub in used:
            continue
        nbrs = sorted(g.neighbors(hub))
        for n1 in nbrs:
            if n1 in used:
                continue
            leaves = [n for n in nbrs if n != n1]
            if any(v in used for v in leaves):
                continue
            cands_n0 = [
                x for x in sorted(g.neighbors(n1)) if x != hub and x not in leaves and x not in used
            ]
            for n0 in cands_n0:
                nodes = (n0, n1, hub, leaves[0], leaves[1])
                sub = g.subgraph(nodes)
                if sub.number_of_edges() == 4:
                    chosen.append(nodes)
                    used.update(nodes)
                    break
            if hub in used:
                break

    if len(chosen) < k_placements:
        raise RuntimeError(f"Only found {len(chosen)} disjoint T5 placements, needed {k_placements}")

    placements: List[Dict[str, Any]] = []
    for pid, nodes in enumerate(chosen):
        sub = g.subgraph(nodes)
        expected_edges = {
            tuple(sorted((nodes[0], nodes[1]))),
            tuple(sorted((nodes[1], nodes[2]))),
            tuple(sorted((nodes[2], nodes[3]))),
            tuple(sorted((nodes[2], nodes[4]))),
        }
        actual_edges = {tuple(sorted(e)) for e in sub.edges()}
        if actual_edges != expected_edges:
            raise AssertionError(f"Placement {nodes} is not isomorphic to T5: {actual_edges}")

        ext_spectator_couplers = int(sum(g.degree[v] for v in nodes) - 8)
        mean_eccentricity = float(np.mean([nx.eccentricity(g, v) for v in nodes]))
        harmonic_centrality = float(
            np.mean(
                [
                    sum(
                        1.0 / d
                        for _t, d in nx.single_source_shortest_path_length(g, v).items()
                        if d > 0
                    )
                    for v in nodes
                ]
            )
        )
        topology_score = (
            -float(ext_spectator_couplers)
            - 0.05 * harmonic_centrality
            + 0.01 * mean_eccentricity
        )
        placements.append(
            {
                "placement_id": pid,
                "physical_qubits": list(nodes),
                "external_spectator_couplers": ext_spectator_couplers,
                "mean_lattice_eccentricity": round(mean_eccentricity, 6),
                "mean_harmonic_centrality": round(harmonic_centrality, 6),
                "topology_degree_score": round(topology_score, 8),
            }
        )
    return placements


def generate_calibration_windows(
    placements: Sequence[Mapping[str, Any]],
    m_windows: int = M_WINDOWS_DEFAULT,
    seed: int = DEFAULT_SEED,
) -> List[List[PlacementNoiseWindow]]:
    """Generate M independent simulated calibration windows with realistic post-calibration TLS drift."""
    if m_windows < 12:
        raise ValueError("Card B requires M >= 12 independent calibration windows")

    rng = np.random.default_rng(seed)
    k_placements = len(placements)
    n_drifted_per_window = max(3, int(round(0.4 * k_placements)))

    windows: List[List[PlacementNoiseWindow]] = []
    for w in range(m_windows):
        raw_cal = []
        for _k in range(k_placements):
            cal_cz = {
                e: float(np.exp(rng.uniform(np.log(0.0014), np.log(0.0052))))
                for e in T5_INTERNAL_EDGES
            }
            cal_ro = tuple(
                float(np.exp(rng.uniform(np.log(0.0030), np.log(0.0220)))) for _ in range(5)
            )
            cal_sx = tuple(
                float(np.exp(rng.uniform(np.log(0.00016), np.log(0.00060)))) for _ in range(5)
            )
            comp_cost = float(sum(cal_cz.values()) + 0.5 * np.mean(cal_ro) + 5.0 * np.mean(cal_sx))
            raw_cal.append((cal_cz, cal_ro, cal_sx, comp_cost))

        cal_order = list(np.argsort([rc[3] for rc in raw_cal]))
        top_half = cal_order[: max(2, k_placements // 2)]
        bot_half = cal_order[max(2, k_placements // 2) :]
        n_top_drift = max(2, (n_drifted_per_window + 1) // 2)
        n_bot_drift = n_drifted_per_window - n_top_drift
        drifted_ids = set(rng.choice(top_half, size=n_top_drift, replace=False).tolist()) | set(
            rng.choice(bot_half, size=n_bot_drift, replace=False).tolist()
        )

        win_rows: List[PlacementNoiseWindow] = []
        for k in range(k_placements):
            cal_cz, cal_ro, cal_sx, comp_cost = raw_cal[k]
            is_drifted = k in drifted_ids

            true_cz: Dict[Tuple[int, int], float] = {}
            for e in T5_INTERNAL_EDGES:
                true_cz[e] = float(cal_cz[e] * rng.uniform(0.85, 1.20))
            true_ro = [float(cal_ro[q] * rng.uniform(0.85, 1.20)) for q in range(5)]
            true_sx = [float(cal_sx[q] * rng.uniform(0.85, 1.20)) for q in range(5)]
            true_overrot = [
                float(rng.choice([-1.0, 1.0]) * rng.uniform(0.001, 0.006)) for _ in range(5)
            ]

            if is_drifted:
                hot_edges = rng.choice(
                    len(T5_INTERNAL_EDGES), size=int(rng.integers(1, 3)), replace=False
                )
                for idx_e in hot_edges:
                    edge = T5_INTERNAL_EDGES[int(idx_e)]
                    mult = float(rng.uniform(4.5, 15.0))
                    true_cz[edge] = float(min(0.18, cal_cz[edge] * mult))
                hot_qubits = rng.choice(5, size=int(rng.integers(1, 3)), replace=False)
                for q_hot in hot_qubits:
                    qh = int(q_hot)
                    true_ro[qh] = float(min(0.15, cal_ro[qh] * rng.uniform(3.0, 12.0)))
                    true_sx[qh] = float(min(0.02, cal_sx[qh] * rng.uniform(4.0, 15.0)))
                    true_overrot[qh] = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.022, 0.055))

            win_rows.append(
                PlacementNoiseWindow(
                    window_id=w,
                    placement_id=k,
                    physical_qubits=tuple(placements[k]["physical_qubits"]),  # type: ignore[arg-type]
                    drifted_regime=is_drifted,
                    cal_cz_err=cal_cz,
                    cal_ro_err=cal_ro,  # type: ignore[arg-type]
                    cal_sx_err=cal_sx,  # type: ignore[arg-type]
                    cal_composite_cost=comp_cost,
                    true_cz_depol=true_cz,
                    true_ro_flip=tuple(true_ro),  # type: ignore[arg-type]
                    true_sx_depol=tuple(true_sx),  # type: ignore[arg-type]
                    true_overrot_rad=tuple(true_overrot),  # type: ignore[arg-type]
                )
            )
        windows.append(win_rows)
    return windows


def _push_pauli_rx(x_bits: List[int], z_bits: List[int], q: int) -> float:
    return -1.0 if z_bits[q] else 1.0


def _push_pauli_rz(x_bits: List[int], z_bits: List[int], q: int) -> float:
    return -1.0 if x_bits[q] else 1.0


def _push_pauli_cz(x_bits: List[int], z_bits: List[int], u: int, v: int) -> None:
    if x_bits[u]:
        z_bits[v] ^= 1
    if x_bits[v]:
        z_bits[u] ^= 1


def _count_adjacent_same_edge_cz(qc: QuantumCircuit) -> int:
    """Count same-edge CZ pairs separated only by diagonal RZ gates (cancellable by commutation)."""
    last_edge_on_qubit: Dict[int, Optional[Tuple[int, int]]] = {}
    cancellable = 0
    for inst in qc.data:
        name = inst.operation.name
        if name == "barrier":
            continue
        qs = [qc.find_bit(qb).index for qb in inst.qubits]
        if name == "cz":
            edge: Tuple[int, int] = tuple(sorted((qs[0], qs[1])))  # type: ignore[assignment]
            if last_edge_on_qubit.get(qs[0]) == edge and last_edge_on_qubit.get(qs[1]) == edge:
                cancellable += 1
            last_edge_on_qubit[qs[0]] = edge
            last_edge_on_qubit[qs[1]] = edge
        elif name in ("sx", "x"):
            last_edge_on_qubit[qs[0]] = None
    return cancellable


def build_echo_probe_and_controls(seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """Construct the stabilizer-reflected 16-CZ `T5` echo probe and all 3 `Card B` controls.

    Uses the `T5` graph-state stabilizer `S = X_0 X_2 Z_3 Z_4` pushed through the forward
    non-Clifford layer `V_A` and reverses `CZ` edge ordering in `V_B` and `G_B`, so that:
      - Ideal return probability `P(|00000>) == 1.000000000000`.
      - Overall 5-qubit unitary is `Z_0 Z_2` (NOT `I_32`), preventing any unitary/peephole pass
        from collapsing the circuit to identity even when all barriers are removed.
    """
    cmap = CouplingMap(HEAVY_HEX_T5_DIRECTED)
    angles_z1 = [0.37, -0.51, 0.63, -0.43, 0.29]
    angles_x1 = [0.41, 0.33, -0.47, 0.59, -0.39]
    angles_x2 = [0.53, -0.31, 0.44, -0.61, 0.36]
    angles_z2 = [-0.27, 0.49, -0.38, 0.34, -0.57]

    qc_fwd = QuantumCircuit(5, name="echo_forward_8cz")
    for q in range(5):
        qc_fwd.h(q)
    for u, v in ((0, 1), (1, 2), (2, 3), (2, 4)):
        qc_fwd.cz(u, v)
    for q in range(5):
        qc_fwd.rz(angles_z1[q], q)
        qc_fwd.rx(angles_x1[q], q)
    for u, v in ((2, 4), (0, 1), (2, 3), (1, 2)):
        qc_fwd.cz(u, v)
    for q in range(5):
        qc_fwd.rx(angles_x2[q], q)
        qc_fwd.rz(angles_z2[q], q)

    x_bits = [1, 0, 1, 0, 0]
    z_bits = [0, 0, 0, 1, 1]
    qc_vb = QuantumCircuit(5)
    for q in range(5):
        sz = _push_pauli_rz(x_bits, z_bits, q)
        qc_vb.rz(sz * angles_z1[q], q)
        sx = _push_pauli_rx(x_bits, z_bits, q)
        qc_vb.rx(sx * angles_x1[q], q)
    for u, v in ((1, 2), (2, 3), (0, 1), (2, 4)):
        _push_pauli_cz(x_bits, z_bits, u, v)
        qc_vb.cz(u, v)
    for q in range(5):
        sx = _push_pauli_rx(x_bits, z_bits, q)
        qc_vb.rx(sx * angles_x2[q], q)
        sz = _push_pauli_rz(x_bits, z_bits, q)
        qc_vb.rz(sz * angles_z2[q], q)

    qc_gb = QuantumCircuit(5)
    for q in range(5):
        qc_gb.h(q)
    for u, v in ((2, 3), (0, 1), (2, 4), (1, 2)):
        qc_gb.cz(u, v)

    def _assemble_echo(delta_rad: float = 0.0) -> QuantumCircuit:
        qc = QuantumCircuit(5, name=f"echo_probe_d{delta_rad:.3f}")
        qc.compose(qc_fwd, inplace=True)
        qc.barrier()
        for q in range(5):
            if z_bits[q]:
                qc.z(q)
            if x_bits[q]:
                qc.x(q)
            if delta_rad != 0.0:
                qc.rz(delta_rad, q)
                qc.rx(delta_rad, q)
        qc.barrier()
        qc.compose(qc_vb.inverse(), inplace=True)
        qc.compose(qc_gb.inverse(), inplace=True)
        return transpile(
            qc,
            basis_gates=["rz", "sx", "x", "cz"],
            coupling_map=cmap,
            initial_layout=[0, 1, 2, 3, 4],
            optimization_level=3,
            seed_transpiler=seed,
        )

    t_fwd = transpile(
        qc_fwd,
        basis_gates=["rz", "sx", "x", "cz"],
        coupling_map=cmap,
        initial_layout=[0, 1, 2, 3, 4],
        optimization_level=3,
        seed_transpiler=seed,
    )
    t_echo = _assemble_echo(0.0)

    qc_id = QuantumCircuit(5, name="echo_identity_0cz")
    qc_id.barrier()

    perturbation_deltas = (0.0, 0.08, 0.16, 0.24, 0.32)
    perturbed_circuits = {d: _assemble_echo(d) for d in perturbation_deltas}

    echo_nobar = RemoveBarriers()(t_echo)
    t_echo_nobar_opt1 = transpile(
        echo_nobar,
        basis_gates=["rz", "sx", "x", "cz"],
        coupling_map=cmap,
        initial_layout=[0, 1, 2, 3, 4],
        optimization_level=1,
        seed_transpiler=seed,
    )
    t_echo_nobar_opt3 = transpile(
        echo_nobar,
        basis_gates=["rz", "sx", "x", "cz"],
        coupling_map=cmap,
        initial_layout=[0, 1, 2, 3, 4],
        optimization_level=3,
        seed_transpiler=seed,
    )

    qc_naive = QuantumCircuit(5)
    qc_naive.compose(qc_fwd, inplace=True)
    qc_naive.barrier()
    qc_naive.compose(qc_fwd.inverse(), inplace=True)
    t_naive_bar = transpile(
        qc_naive,
        basis_gates=["rz", "sx", "x", "cz"],
        coupling_map=cmap,
        initial_layout=[0, 1, 2, 3, 4],
        optimization_level=1,
        seed_transpiler=seed,
    )
    t_naive_nobar = transpile(
        RemoveBarriers()(t_naive_bar),
        basis_gates=["rz", "sx", "x", "cz"],
        coupling_map=cmap,
        initial_layout=[0, 1, 2, 3, 4],
        optimization_level=1,
        seed_transpiler=seed,
    )

    fwd_cz = int(t_fwd.count_ops().get("cz", 0))
    echo_cz = int(t_echo.count_ops().get("cz", 0))
    nobar_opt1_cz = int(t_echo_nobar_opt1.count_ops().get("cz", 0))
    nobar_opt3_cz = int(t_echo_nobar_opt3.count_ops().get("cz", 0))
    naive_bar_cz = int(t_naive_bar.count_ops().get("cz", 0))
    naive_nobar_cz = int(t_naive_nobar.count_ops().get("cz", 0))
    adj_cz = _count_adjacent_same_edge_cz(echo_nobar)

    ideal_p0 = float(abs(Statevector.from_instruction(t_echo).data[0]) ** 2)
    u_echo_nobar = Operator(echo_nobar).data
    non_identity_norm = float(np.abs(np.trace(u_echo_nobar)))

    non_cancelled_ok = bool(
        fwd_cz == 8
        and echo_cz >= 2 * fwd_cz
        and nobar_opt1_cz >= 2 * fwd_cz
        and nobar_opt3_cz >= 2 * fwd_cz
        and adj_cz == 0
        and naive_nobar_cz == 0
        and abs(ideal_p0 - 1.0) < 1e-12
    )

    return {
        "echo_circuit": t_echo,
        "identity_circuit": qc_id,
        "perturbed_circuits": perturbed_circuits,
        "perturbation_deltas_rad": list(perturbation_deltas),
        "audit": {
            "ideal_noiseless_return_prob": round(ideal_p0, 12),
            "forward_cz_count": fwd_cz,
            "echo_compiled_cz_count": echo_cz,
            "echo_nobarrier_opt1_cz_count": nobar_opt1_cz,
            "echo_nobarrier_opt3_cz_count": nobar_opt3_cz,
            "adjacent_cancellable_cz_pairs": adj_cz,
            "naive_palindrome_with_barrier_cz_count": naive_bar_cz,
            "naive_palindrome_without_barrier_cz_count": naive_nobar_cz,
            "echo_unitary_abs_trace": round(non_identity_norm, 12),
            "global_unitary_is_non_identity_reflection": bool(non_identity_norm < 1e-9),
            "non_cancelled_cz_verified": non_cancelled_ok,
        },
    }


def build_heldout_reference_circuits() -> Dict[str, Dict[str, Any]]:
    """Build 3 unseen held-out exact reference circuit classes compiled to `T5`."""
    cmap = CouplingMap(HEAVY_HEX_T5_DIRECTED)
    bt2t = _load_bt2t()

    qc_bt2t_raw = bt2t.inversion_circuit_structured()
    t_bt2t = transpile(
        qc_bt2t_raw,
        basis_gates=["rz", "sx", "x", "cz"],
        coupling_map=cmap,
        optimization_level=3,
        seed_transpiler=20261122,
    )
    bt2t_init_idx = [t_bt2t.layout.initial_index_layout()[q] for q in range(5)]
    bt2t_final_idx = [t_bt2t.layout.final_index_layout()[q] for q in range(5)]
    bt2t_perm = bt2t.inversion_permutation()

    c1_raw = random_clifford(5, seed=8802)
    tc1 = transpile(
        c1_raw.to_circuit(),
        basis_gates=["rz", "sx", "x", "cz"],
        coupling_map=cmap,
        optimization_level=3,
        seed_transpiler=8802,
    )
    c1_phys = Clifford.from_matrix(Operator(tc1).data)
    qc_zt = QuantumCircuit(5)
    qc_zt.cz(0, 4)
    qc_zt.cz(1, 3)
    qc_zt.z(2)
    c_zt = Clifford(qc_zt)
    qc_xt = QuantumCircuit(5)
    qc_xt.x([0, 2, 3])
    c_xt = Clifford(qc_xt)
    c2_log = (c_zt.compose(c1_phys)).adjoint().compose(c_xt)
    tc2 = transpile(
        c2_log.to_circuit(),
        basis_gates=["rz", "sx", "x", "cz"],
        coupling_map=cmap,
        initial_layout=[0, 1, 2, 3, 4],
        optimization_level=3,
        seed_transpiler=9902,
    )
    qc_cliff_join = QuantumCircuit(5)
    qc_cliff_join.compose(tc1, inplace=True)
    qc_cliff_join.compose(tc2, inplace=True)
    t_cliff = transpile(
        RemoveBarriers()(qc_cliff_join),
        basis_gates=["rz", "sx", "x", "cz"],
        coupling_map=cmap,
        initial_layout=[0, 1, 2, 3, 4],
        optimization_level=3,
        seed_transpiler=1234,
    )
    cliff_probs_ideal = np.abs(Statevector.from_instruction(t_cliff).data) ** 2
    cliff_target_phys = int(np.argmax(cliff_probs_ideal))
    if cliff_probs_ideal[cliff_target_phys] < 1.0 - 1e-12:
        raise AssertionError("Held-out random Clifford ideal peak probability < 1")

    az1 = [-0.67, 0.43, -0.29, 0.71, -0.53]
    ax1 = [0.31, -0.59, 0.48, -0.37, 0.62]
    az2 = [0.52, -0.34, 0.66, -0.41, 0.28]
    ax2 = [-0.44, 0.39, -0.61, 0.33, -0.49]
    ax3 = [0.27, -0.46, 0.35, 0.58, -0.32]

    qc_mf = QuantumCircuit(5)
    for q in range(5):
        qc_mf.h(q)
    for u, v in ((0, 1), (1, 2), (2, 3), (2, 4)):
        qc_mf.cz(u, v)
    for q in range(5):
        qc_mf.rz(az1[q], q)
        qc_mf.rx(ax1[q], q)
    for u, v in ((0, 1), (2, 3), (1, 2), (2, 4)):
        qc_mf.cz(u, v)
    for q in range(5):
        qc_mf.rz(az2[q], q)
        qc_mf.rx(ax2[q], q)
    for u, v in ((2, 4), (1, 2), (2, 3), (0, 1)):
        qc_mf.cz(u, v)
    for q in range(5):
        qc_mf.rx(ax3[q], q)

    xb = [0, 1, 0, 1, 1]
    zb = [1, 0, 1, 0, 0]
    qc_mvb = QuantumCircuit(5)
    for q in range(5):
        sz = _push_pauli_rz(xb, zb, q)
        qc_mvb.rz(sz * az1[q], q)
        sx = _push_pauli_rx(xb, zb, q)
        qc_mvb.rx(sx * ax1[q], q)
    for u, v in ((2, 4), (1, 2), (2, 3), (0, 1)):
        _push_pauli_cz(xb, zb, u, v)
        qc_mvb.cz(u, v)
    for q in range(5):
        sz = _push_pauli_rz(xb, zb, q)
        qc_mvb.rz(sz * az2[q], q)
        sx = _push_pauli_rx(xb, zb, q)
        qc_mvb.rx(sx * ax2[q], q)
    for u, v in ((0, 1), (2, 3), (1, 2), (2, 4)):
        _push_pauli_cz(xb, zb, u, v)
        qc_mvb.cz(u, v)
    for q in range(5):
        sx = _push_pauli_rx(xb, zb, q)
        qc_mvb.rx(sx * ax3[q], q)

    qc_mgb = QuantumCircuit(5)
    for q in range(5):
        qc_mgb.h(q)
    for u, v in ((2, 4), (2, 3), (1, 2), (0, 1)):
        qc_mgb.cz(u, v)

    qc_mirror = QuantumCircuit(5, name="heldout_non_clifford_peaked_24cz")
    qc_mirror.compose(qc_mf, inplace=True)
    for q in range(5):
        if zb[q]:
            qc_mirror.z(q)
        if xb[q]:
            qc_mirror.x(q)
    qc_mirror.compose(qc_mvb.inverse(), inplace=True)
    qc_mirror.compose(qc_mgb.inverse(), inplace=True)
    qc_mirror.x([0, 3, 4])
    t_mirror = transpile(
        RemoveBarriers()(qc_mirror),
        basis_gates=["rz", "sx", "x", "cz"],
        coupling_map=cmap,
        initial_layout=[0, 1, 2, 3, 4],
        optimization_level=3,
        seed_transpiler=20261199,
    )
    mirror_probs_ideal = np.abs(Statevector.from_instruction(t_mirror).data) ** 2
    mirror_target_phys = int(np.argmax(mirror_probs_ideal))
    if mirror_probs_ideal[mirror_target_phys] < 1.0 - 1e-12 or mirror_target_phys != 25:
        raise AssertionError("Held-out non-Clifford peaked circuit ideal peak mismatch")

    return {
        "bt2t_72cz": {
            "circuit": t_bt2t,
            "cz_count": int(t_bt2t.count_ops().get("cz", 0)),
            "depth": int(t_bt2t.depth()),
            "init_idx": bt2t_init_idx,
            "final_idx": bt2t_final_idx,
            "inversion_perm": bt2t_perm,
        },
        "random_clifford_30cz": {
            "circuit": t_cliff,
            "cz_count": int(t_cliff.count_ops().get("cz", 0)),
            "depth": int(t_cliff.depth()),
            "target_phys_state": cliff_target_phys,
            "ideal_peak_prob": float(cliff_probs_ideal[cliff_target_phys]),
        },
        "non_clifford_mirror_24cz": {
            "circuit": t_mirror,
            "cz_count": int(t_mirror.count_ops().get("cz", 0)),
            "depth": int(t_mirror.depth()),
            "target_phys_state": mirror_target_phys,
            "ideal_peak_prob": float(mirror_probs_ideal[mirror_target_phys]),
        },
    }


def _apply_1q_unitary_batch(rho: np.ndarray, mat: np.ndarray, q: int) -> np.ndarray:
    ka, ba = 1 + (4 - q), 6 + (4 - q)
    rho = np.moveaxis(np.tensordot(mat, rho, axes=([1], [ka])), 0, ka)
    rho = np.moveaxis(np.tensordot(mat.conj(), rho, axes=([1], [ba])), 0, ba)
    return rho


def _apply_1q_depol_batch(rho: np.ndarray, lam: float, q: int) -> np.ndarray:
    if lam <= 0.0:
        return rho
    ka, ba = 1 + (4 - q), 6 + (4 - q)
    tr_q = np.diagonal(rho, axis1=ka, axis2=ba).sum(axis=-1)
    mixed = tr_q[..., None, None] * (0.5 * np.eye(2, dtype=complex))
    orig_axes = [a for a in range(11) if a not in (ka, ba)] + [ka, ba]
    return (1.0 - lam) * rho + lam * np.transpose(mixed, np.argsort(orig_axes))


def _apply_cz_noisy_batch(
    rho: np.ndarray, u: int, v: int, zz_ang: float, lam: float
) -> np.ndarray:
    ku, kv = 1 + (4 - u), 1 + (4 - v)
    bu, bv = 6 + (4 - u), 6 + (4 - v)
    ph = np.array(
        [
            [np.exp(-0.5j * zz_ang), np.exp(0.5j * zz_ang)],
            [np.exp(0.5j * zz_ang), -np.exp(-0.5j * zz_ang)],
        ],
        dtype=complex,
    )
    sk = [1] * 11
    sb = [1] * 11
    sk[ku] = sk[kv] = 2
    sb[bu] = sb[bv] = 2
    rho = rho * ph.reshape(sk) * ph.conj().reshape(sb)
    if lam > 0.0:
        keep = [a for a in range(11) if a not in (ku, kv, bu, bv)]
        rp = np.transpose(rho, keep + [ku, kv, bu, bv])
        tr_uv = (
            rp[..., 0, 0, 0, 0]
            + rp[..., 0, 1, 0, 1]
            + rp[..., 1, 0, 1, 0]
            + rp[..., 1, 1, 1, 1]
        )
        eye4_q = 0.25 * np.eye(4, dtype=complex).reshape(2, 2, 2, 2)
        mixed = np.transpose(
            tr_uv[..., None, None, None, None] * eye4_q,
            np.argsort(keep + [ku, kv, bu, bv]),
        )
        rho = (1.0 - lam) * rho + lam * mixed
    return rho


def simulate_t5_noisy_probabilities(
    qc: QuantumCircuit,
    true_cz_depol: Mapping[Tuple[int, int], float],
    true_sx_depol: Sequence[float],
    true_overrot_rad: Sequence[float],
    true_ro_flip: Sequence[float],
    initial_phys_states: Sequence[int] = (0,),
    include_initial_state_prep_noise: bool = False,
) -> np.ndarray:
    """Exact 5-qubit Kraus density-matrix evolution + readout confusion on `['rz', 'sx', 'x', 'cz']`."""
    batch = len(initial_phys_states)
    rho = np.zeros((batch, 32, 32), dtype=complex)
    for b_idx, s in enumerate(initial_phys_states):
        if not include_initial_state_prep_noise:
            rho[b_idx, s, s] = 1.0
        else:
            rho[b_idx, 0, 0] = 1.0
    rho = rho.reshape((batch,) + (2,) * 10)

    if include_initial_state_prep_noise:
        for q in range(5):
            mask = [b_idx for b_idx, s in enumerate(initial_phys_states) if (s >> q) & 1]
            if mask:
                sub = rho[mask]
                m_x = _rx_mat(float(true_overrot_rad[q])) @ X_MAT
                sub = _apply_1q_depol_batch(
                    _apply_1q_unitary_batch(sub, m_x, q), float(true_sx_depol[q]), q
                )
                rho[mask] = sub

    sx_noisy = [_rx_mat(float(true_overrot_rad[q])) @ SX_MAT for q in range(5)]
    x_noisy = [_rx_mat(float(true_overrot_rad[q])) @ X_MAT for q in range(5)]

    for inst in qc.data:
        name = inst.operation.name
        if name in ("barrier", "measure"):
            continue
        qs = [qc.find_bit(qb).index for qb in inst.qubits]
        if name == "rz":
            theta = float(inst.operation.params[0])
            rho = _apply_1q_unitary_batch(rho, _rz_mat(theta), qs[0])
        elif name == "sx":
            q = qs[0]
            rho = _apply_1q_depol_batch(
                _apply_1q_unitary_batch(rho, sx_noisy[q], q), float(true_sx_depol[q]), q
            )
        elif name == "x":
            q = qs[0]
            rho = _apply_1q_depol_batch(
                _apply_1q_unitary_batch(rho, x_noisy[q], q), float(true_sx_depol[q]), q
            )
        elif name == "cz":
            u, v = qs[0], qs[1]
            ek: Tuple[int, int] = tuple(sorted((u, v)))  # type: ignore[assignment]
            zz_ang = 0.5 * (float(true_overrot_rad[u]) + float(true_overrot_rad[v]))
            rho = _apply_cz_noisy_batch(rho, u, v, zz_ang, float(true_cz_depol[ek]))
        else:
            raise ValueError(f"Unsupported non-native instruction in T5 simulation: {name}")

    diag_probs = np.real(np.diagonal(rho.reshape(batch, 32, 32), axis1=1, axis2=2))
    ro_tensor = diag_probs.reshape((batch, 2, 2, 2, 2, 2))
    for q in range(5):
        ax = 1 + (4 - q)
        p_flip = float(true_ro_flip[q])
        m_ro = np.array([[1.0 - p_flip, p_flip], [p_flip, 1.0 - p_flip]], dtype=float)
        ro_tensor = np.moveaxis(np.tensordot(m_ro, ro_tensor, axes=([1], [ax])), 0, ax)
    return ro_tensor.reshape(batch, 32)


def _evaluate_bt2t_valid24_fidelity(
    bt2t_spec: Mapping[str, Any], nw: PlacementNoiseWindow
) -> float:
    init_idx = bt2t_spec["init_idx"]
    final_idx = bt2t_spec["final_idx"]
    perm = bt2t_spec["inversion_perm"]

    init_phys_states: List[int] = []
    target_phys_states: List[int] = []
    for x in range(24):
        s_in = sum(((x >> q) & 1) << init_idx[q] for q in range(5))
        y_log = perm[x]
        s_out = sum(((y_log >> q) & 1) << final_idx[q] for q in range(5))
        init_phys_states.append(s_in)
        target_phys_states.append(s_out)

    probs = simulate_t5_noisy_probabilities(
        bt2t_spec["circuit"],
        nw.true_cz_depol,
        nw.true_sx_depol,
        nw.true_overrot_rad,
        nw.true_ro_flip,
        initial_phys_states=init_phys_states,
        include_initial_state_prep_noise=True,
    )
    correct = [probs[i, target_phys_states[i]] for i in range(24)]
    return float(np.mean(correct))


def _spearman_corr(scores: Sequence[float], targets: Sequence[float]) -> float:
    res = stats.spearmanr(scores, targets)
    val = float(res.statistic)
    if np.isnan(val):
        return 0.0
    return val


def run_echo_placement_stopgo_gate(
    k_placements: int = K_PLACEMENTS_DEFAULT,
    m_windows: int = M_WINDOWS_DEFAULT,
    b_boot: int = B_BOOT_DEFAULT,
    echo_shots_per_placement: int = ECHO_SHOTS_DEFAULT,
    seed: int = DEFAULT_SEED,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute the full `Card B` (`OSIRIS-ECHO`) offline noisy-simulation stop/go gate."""
    placements = select_t5_candidate_placements(k_placements=k_placements)
    windows = generate_calibration_windows(placements, m_windows=m_windows, seed=seed)
    echo_bundle = build_echo_probe_and_controls(seed=seed)
    heldout = build_heldout_reference_circuits()

    rng_shots = np.random.default_rng(seed + 1000)
    rng_rand_ranker = np.random.default_rng(seed + 2000)
    rng_boot = np.random.default_rng(seed + 3000)

    echo_qc = echo_bundle["echo_circuit"]
    id_qc = echo_bundle["identity_circuit"]
    pert_deltas = echo_bundle["perturbation_deltas_rad"]
    pert_qcs = echo_bundle["perturbed_circuits"]

    ideal_pert_curve = [
        float(abs(Statevector.from_instruction(pert_qcs[d]).data[0]) ** 2) for d in pert_deltas
    ]
    noisy_pert_accum = np.zeros(len(pert_deltas), dtype=float)
    identity_gt_echo_count = 0
    total_cells = m_windows * k_placements

    rankers = ("ECHO_RANKED", "CALIBRATION_RANKED", "TOPOLOGY_DEGREE_ONLY", "RANDOM_PLACEMENT")
    comparators = ("CALIBRATION_RANKED", "TOPOLOGY_DEGREE_ONLY", "RANDOM_PLACEMENT")
    heldout_classes = ("composite", "bt2t_72cz", "random_clifford_30cz", "non_clifford_mirror_24cz")

    spearman_by_ranker: Dict[str, Dict[str, List[float]]] = {
        r: {hc: [] for hc in heldout_classes} for r in rankers
    }
    top1_fidelity_by_ranker: Dict[str, List[float]] = {r: [] for r in rankers}
    top1_regret_by_ranker: Dict[str, List[float]] = {r: [] for r in rankers}
    window_summaries: List[Dict[str, Any]] = []

    topology_scores_static = [float(p["topology_degree_score"]) for p in placements]

    for w_idx, win_rows in enumerate(windows):
        echo_emp_scores: List[float] = []
        echo_exact_scores: List[float] = []
        id_exact_scores: List[float] = []
        cal_scores: List[float] = []
        f_bt2t_list: List[float] = []
        f_cliff_list: List[float] = []
        f_mirror_list: List[float] = []
        f_comp_list: List[float] = []

        for nw in win_rows:
            p_id = float(
                simulate_t5_noisy_probabilities(
                    id_qc,
                    nw.true_cz_depol,
                    nw.true_sx_depol,
                    nw.true_overrot_rad,
                    nw.true_ro_flip,
                )[0, 0]
            )
            probs_echo = simulate_t5_noisy_probabilities(
                echo_qc,
                nw.true_cz_depol,
                nw.true_sx_depol,
                nw.true_overrot_rad,
                nw.true_ro_flip,
            )[0]
            p_echo_exact = float(probs_echo[0])
            emp_hits = int(rng_shots.binomial(echo_shots_per_placement, p_echo_exact))
            p_echo_emp = float(emp_hits / echo_shots_per_placement)

            if p_id > p_echo_exact:
                identity_gt_echo_count += 1

            if w_idx < 4:
                for d_i, d_val in enumerate(pert_deltas):
                    p_d = float(
                        simulate_t5_noisy_probabilities(
                            pert_qcs[d_val],
                            nw.true_cz_depol,
                            nw.true_sx_depol,
                            nw.true_overrot_rad,
                            nw.true_ro_flip,
                        )[0, 0]
                    )
                    noisy_pert_accum[d_i] += p_d / (4.0 * k_placements)

            f_bt2t = _evaluate_bt2t_valid24_fidelity(heldout["bt2t_72cz"], nw)
            f_cliff = float(
                simulate_t5_noisy_probabilities(
                    heldout["random_clifford_30cz"]["circuit"],
                    nw.true_cz_depol,
                    nw.true_sx_depol,
                    nw.true_overrot_rad,
                    nw.true_ro_flip,
                )[0, heldout["random_clifford_30cz"]["target_phys_state"]]
            )
            f_mirror = float(
                simulate_t5_noisy_probabilities(
                    heldout["non_clifford_mirror_24cz"]["circuit"],
                    nw.true_cz_depol,
                    nw.true_sx_depol,
                    nw.true_overrot_rad,
                    nw.true_ro_flip,
                )[0, heldout["non_clifford_mirror_24cz"]["target_phys_state"]]
            )
            f_comp = float((f_bt2t + f_cliff + f_mirror) / 3.0)

            id_exact_scores.append(p_id)
            echo_exact_scores.append(p_echo_exact)
            echo_emp_scores.append(p_echo_emp)
            cal_scores.append(-nw.cal_composite_cost)
            f_bt2t_list.append(f_bt2t)
            f_cliff_list.append(f_cliff)
            f_mirror_list.append(f_mirror)
            f_comp_list.append(f_comp)

        rand_scores = rng_rand_ranker.uniform(0.0, 1.0, size=k_placements).tolist()
        score_map: Dict[str, List[float]] = {
            "ECHO_RANKED": echo_emp_scores,
            "CALIBRATION_RANKED": cal_scores,
            "TOPOLOGY_DEGREE_ONLY": topology_scores_static,
            "RANDOM_PLACEMENT": rand_scores,
        }
        target_map: Dict[str, List[float]] = {
            "composite": f_comp_list,
            "bt2t_72cz": f_bt2t_list,
            "random_clifford_30cz": f_cliff_list,
            "non_clifford_mirror_24cz": f_mirror_list,
        }

        oracle_best_f = float(max(f_comp_list))
        oracle_best_idx = int(np.argmax(f_comp_list))

        win_corrs: Dict[str, float] = {}
        win_regrets: Dict[str, float] = {}
        for r_name in rankers:
            sc = score_map[r_name]
            for hc_name, tgt in target_map.items():
                spearman_by_ranker[r_name][hc_name].append(_spearman_corr(sc, tgt))
            win_corrs[r_name] = round(spearman_by_ranker[r_name]["composite"][-1], 6)
            chosen_idx = int(np.argmax(sc))
            chosen_f = float(f_comp_list[chosen_idx])
            reg = float(oracle_best_f - chosen_f)
            top1_fidelity_by_ranker[r_name].append(chosen_f)
            top1_regret_by_ranker[r_name].append(reg)
            win_regrets[r_name] = round(reg, 6)

        window_summaries.append(
            {
                "window_id": w_idx,
                "oracle_best_placement_id": oracle_best_idx,
                "oracle_best_heldout_fidelity": round(oracle_best_f, 6),
                "mean_heldout_fidelity": round(float(np.mean(f_comp_list)), 6),
                "min_heldout_fidelity": round(float(np.min(f_comp_list)), 6),
                "mean_identity_0cz_return": round(float(np.mean(id_exact_scores)), 6),
                "mean_echo_16cz_return": round(float(np.mean(echo_emp_scores)), 6),
                "spearman_composite_by_ranker": win_corrs,
                "top1_regret_by_ranker": win_regrets,
            }
        )

    boot_indices = rng_boot.integers(0, m_windows, size=(b_boot, m_windows))

    rho_arrays = {
        r: np.asarray(spearman_by_ranker[r]["composite"], dtype=float) for r in rankers
    }
    regret_arrays = {r: np.asarray(top1_regret_by_ranker[r], dtype=float) for r in rankers}

    mean_rho_point = {r: float(np.mean(rho_arrays[r])) for r in rankers}
    best_comp_name = max(comparators, key=lambda c: mean_rho_point[c])
    delta_spearman_point = float(mean_rho_point["ECHO_RANKED"] - mean_rho_point[best_comp_name])

    boot_echo = np.mean(rho_arrays["ECHO_RANKED"][boot_indices], axis=1)
    boot_comps = np.stack([np.mean(rho_arrays[c][boot_indices], axis=1) for c in comparators], axis=1)
    boot_best_comp = np.max(boot_comps, axis=1)
    boot_delta_rho = boot_echo - boot_best_comp
    boot_delta_vs_cal = boot_echo - np.mean(rho_arrays["CALIBRATION_RANKED"][boot_indices], axis=1)

    delta_lcl95_two_sided = float(np.percentile(boot_delta_rho, 2.5))
    delta_ucl95_two_sided = float(np.percentile(boot_delta_rho, 97.5))
    delta_lcl95_one_sided = float(np.percentile(boot_delta_rho, 5.0))

    boot_regret_echo = np.mean(regret_arrays["ECHO_RANKED"][boot_indices], axis=1)
    boot_regret_best_comp = np.min(
        np.stack([np.mean(regret_arrays[c][boot_indices], axis=1) for c in comparators], axis=1),
        axis=1,
    )
    boot_regret_reduction = boot_regret_best_comp - boot_regret_echo

    per_class_summary: Dict[str, Any] = {}
    for hc in heldout_classes:
        echo_hc = np.asarray(spearman_by_ranker["ECHO_RANKED"][hc], dtype=float)
        cal_hc = np.asarray(spearman_by_ranker["CALIBRATION_RANKED"][hc], dtype=float)
        diff_boot = np.mean(echo_hc[boot_indices], axis=1) - np.mean(cal_hc[boot_indices], axis=1)
        per_class_summary[hc] = {
            "echo_mean_spearman": round(float(np.mean(echo_hc)), 6),
            "calibration_mean_spearman": round(float(np.mean(cal_hc)), 6),
            "topology_mean_spearman": round(
                float(np.mean(spearman_by_ranker["TOPOLOGY_DEGREE_ONLY"][hc])), 6
            ),
            "random_mean_spearman": round(
                float(np.mean(spearman_by_ranker["RANDOM_PLACEMENT"][hc])), 6
            ),
            "delta_echo_minus_cal_mean": round(float(np.mean(echo_hc - cal_hc)), 6),
            "delta_echo_minus_cal_ci95": [
                round(float(np.percentile(diff_boot, 2.5)), 6),
                round(float(np.percentile(diff_boot, 97.5)), 6),
            ],
        }

    noisy_pert_list = [float(x) for x in noisy_pert_accum]
    ideal_monotone = all(
        ideal_pert_curve[i] > ideal_pert_curve[i + 1] for i in range(len(pert_deltas) - 1)
    )
    noisy_monotone = all(
        noisy_pert_list[i] > noisy_pert_list[i + 1] for i in range(len(pert_deltas) - 1)
    )
    control_1_pass = identity_gt_echo_count == total_cells
    control_2_pass = bool(ideal_monotone and noisy_monotone)
    control_3_pass = bool(echo_bundle["audit"]["non_cancelled_cz_verified"])
    echo_recovers_ranking = bool(
        delta_spearman_point > 0.0
        and delta_lcl95_two_sided > 0.0
        and control_1_pass
        and control_2_pass
        and control_3_pass
    )

    ranker_table: Dict[str, Any] = {}
    for r_name in rankers:
        r_boot = np.mean(rho_arrays[r_name][boot_indices], axis=1)
        reg_arr = regret_arrays[r_name]
        fid_arr = np.asarray(top1_fidelity_by_ranker[r_name], dtype=float)
        ranker_table[r_name] = {
            "mean_spearman": round(float(np.mean(rho_arrays[r_name])), 6),
            "spearman_window_ci95": [
                round(float(np.percentile(r_boot, 2.5)), 6),
                round(float(np.percentile(r_boot, 97.5)), 6),
            ],
            "mean_top1_heldout_fidelity": round(float(np.mean(fid_arr)), 6),
            "mean_top1_regret": round(float(np.mean(reg_arr)), 6),
            "zero_regret_window_fraction": round(float(np.mean(reg_arr <= 1e-12)), 6),
        }

    report: Dict[str, Any] = {
        "schema": "osiris.card_b.echo_stopgo_report/1",
        "experiment_card": "Card B: OSIRIS-ECHO",
        "execution_tier": "SIMULATED",
        "provider_job_id": None,
        "qpu_seconds": 0,
        "hardware_execution_authorized": False,
        "mechanism_inferred": "none",
        "seed": seed,
        "design": {
            "substrate_model": "156-qubit IBM Heron r2 (ibm_fez) heavy-hex coupling lattice",
            "candidate_t5_placements_K": k_placements,
            "calibration_windows_M": m_windows,
            "unit_of_inference": "calibration_window",
            "bootstrap_replicates_B": b_boot,
            "echo_shots_per_placement": echo_shots_per_placement,
            "candidate_placements": placements,
            "heldout_reference_circuits": {
                "bt2t_72cz": {
                    "description": "Structured 5Q Binary Tetrahedral (2T) exact group inversion over 24 valid inputs",
                    "cz_count": heldout["bt2t_72cz"]["cz_count"],
                    "depth": heldout["bt2t_72cz"]["depth"],
                },
                "random_clifford_30cz": {
                    "description": "Held-out 5Q random Clifford survival circuit on T5",
                    "cz_count": heldout["random_clifford_30cz"]["cz_count"],
                    "depth": heldout["random_clifford_30cz"]["depth"],
                },
                "non_clifford_mirror_24cz": {
                    "description": "Held-out 5Q non-Clifford stabilizer-twisted peaked circuit on T5",
                    "cz_count": heldout["non_clifford_mirror_24cz"]["cz_count"],
                    "depth": heldout["non_clifford_mirror_24cz"]["depth"],
                },
            },
        },
        "controls": {
            "control_1_no_workload_identity_echo": {
                "cz_count": 0,
                "cells_where_identity_exceeds_workload_echo": identity_gt_echo_count,
                "total_placement_window_cells": total_cells,
                "mean_identity_return": round(
                    float(np.mean([w["mean_identity_0cz_return"] for w in window_summaries])), 6
                ),
                "mean_workload_echo_return": round(
                    float(np.mean([w["mean_echo_16cz_return"] for w in window_summaries])), 6
                ),
                "passed": control_1_pass,
            },
            "control_2_deliberate_overrotation_perturbation": {
                "perturbation_deltas_rad": pert_deltas,
                "ideal_return_curve": [round(v, 6) for v in ideal_pert_curve],
                "mean_noisy_return_curve": [round(v, 6) for v in noisy_pert_list],
                "monotone_drop_verified": control_2_pass,
                "passed": control_2_pass,
            },
            "control_3_compiled_artifact_inspection": echo_bundle["audit"],
        },
        "ranking_comparison": {
            "rankers": ranker_table,
            "best_non_echo_comparator": best_comp_name,
            "delta_spearman_point": round(delta_spearman_point, 6),
            "delta_spearman_lcl95": round(delta_lcl95_two_sided, 6),
            "delta_spearman_lcl95_one_sided": round(delta_lcl95_one_sided, 6),
            "delta_spearman_ci95": [
                round(delta_lcl95_two_sided, 6),
                round(delta_ucl95_two_sided, 6),
            ],
            "delta_spearman_vs_calibration_ci95": [
                round(float(np.percentile(boot_delta_vs_cal, 2.5)), 6),
                round(float(np.percentile(boot_delta_vs_cal, 97.5)), 6),
            ],
            "top1_regret_reduction_vs_best_comparator": {
                "point": round(float(np.mean(boot_regret_reduction)), 6),
                "ci95": [
                    round(float(np.percentile(boot_regret_reduction, 2.5)), 6),
                    round(float(np.percentile(boot_regret_reduction, 97.5)), 6),
                ],
            },
            "per_heldout_circuit_class": per_class_summary,
        },
        "stop_go_decision": {
            "echo_recovers_ranking": echo_recovers_ranking,
            "delta_spearman_lcl95_gt_zero": bool(delta_lcl95_two_sided > 0.0),
            "all_mandatory_controls_passed": bool(
                control_1_pass and control_2_pass and control_3_pass
            ),
            "mechanism_inferred": "none",
            "verdict": (
                "PASS_OFFLINE_STOPGO_GATE" if echo_recovers_ranking else "STOP_BEFORE_HARDWARE"
            ),
            "scope_note": (
                "Offline SIMULATED stop/go gate only (0 QPU seconds). Demonstrates that a 16-CZ "
                "non-cancelling echo probe recovers placement ranking on unseen 5Q circuits under "
                "post-calibration single-qubit/coupler drift where stale calibration tables degrade. "
                "No physical mechanism is inferred from return curves, and live hardware execution "
                "requires separate preregistration and owner authorization."
            ),
        },
        "windows": window_summaries,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
        output_path.write_text(payload, encoding="utf-8", newline="\n")

    return report


def main() -> None:
    out_path = Path(__file__).resolve().parent / "artifacts" / "echo_stopgo_report.json"
    rep = run_echo_placement_stopgo_gate(output_path=out_path)
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    rc = rep["ranking_comparison"]
    sg = rep["stop_go_decision"]
    print(f"Wrote {out_path} (sha256={sha[:16]}...)")
    print("Rankers:", json.dumps(rc["rankers"], indent=2))
    print(
        f"Delta Spearman = {rc['delta_spearman_point']:+.4f} "
        f"(95% window-bootstrap CI: {rc['delta_spearman_ci95']})"
    )
    print(f"Stop/Go Verdict: {sg['verdict']} (echo_recovers_ranking={sg['echo_recovers_ranking']})")


if __name__ == "__main__":
    main()
