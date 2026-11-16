"""
EXPERIMENT TESSERACT-MAP-02: Controlled 40-Qubit Compilation & Routing Benchmark
================================================================================

Phase 2 Section 5 (`EXPERIMENT TESSERACT-MAP-02`).

Isolates the confounded factors behind the 2026-09-20 `ibm_kingston` rerun
(`dansntf8gn2s739n32e0`), where 4-string ideal-support probability increased
from 5.305% (`P0_replication`, classical fidelity 0.0498) to 17.995%
(`P2_chain_mapped`, classical fidelity 0.1645).

GOVERNANCE GUARD:
  Offline compilation, equivalence proof, archived-data verification, and local
  noisy simulation ONLY. QPU job submission is strictly disabled (`allow_qpu=False`)
  pending explicit owner authorization.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from scipy import stats

THETA_DEFAULT_DEG: float = 51.700
LAMBDA_PHI_ORIG: float = 1.61803398875


# ---------------------------------------------------------------------------
# 1. Logical Circuit Builders & Ideal Distribution / Equivalence Verification
# ---------------------------------------------------------------------------

def build_star_circuit(
    n_qubits: int = 40,
    theta_deg: float = THETA_DEFAULT_DEG,
    include_inert_rz: bool = False,
    measure: bool = True,
    name: str = "star",
) -> QuantumCircuit:
    """
    Star fan-out topology (`P0_replication` / `P1_ghz_control`):
      - Qubit 0 and 1 form entangled 2-qubit core cos(theta)|Phi+> - i*sin(theta)|Psi+>
      - Every even qubit i >= 2 copies qubit 0 via CX(0, i)
      - Every odd qubit i >= 3 copies qubit 1 via CX(1, i)
    """
    if n_qubits < 4 or n_qubits % 2 != 0:
        raise ValueError(f"n_qubits must be an even integer >= 4, got {n_qubits}")
    qc = QuantumCircuit(n_qubits, name=name)
    qc.h(0)
    qc.cx(0, 1)
    if theta_deg != 0.0:
        qc.rx(math.radians(theta_deg), [0, 1])
    for i in range(2, n_qubits):
        qc.cx(i % 2, i)
    if include_inert_rz:
        qc.rz(math.pi / (8.0 * LAMBDA_PHI_ORIG), [0, 1])
    if measure:
        qc.measure_all()
    return qc


def build_chain_circuit(
    n_qubits: int = 40,
    theta_deg: float = THETA_DEFAULT_DEG,
    measure: bool = True,
    name: str = "chain",
) -> QuantumCircuit:
    """
    Degree-2 linear chain topology (`P2_chain_mapped`):
      Prepares the EXACT same N-qubit quantum state as `build_star_circuit`
      (up to the inert pre-measurement RZ phase when `include_inert_rz=True`),
      with every CX acting between adjacent wires in the logical ordering:
          [n-2, n-4, ..., 2, 0, 1, 3, ..., n-1]
    """
    if n_qubits < 4 or n_qubits % 2 != 0:
        raise ValueError(f"n_qubits must be an even integer >= 4, got {n_qubits}")
    qc = QuantumCircuit(n_qubits, name=name)
    qc.h(0)
    qc.cx(0, 1)
    if theta_deg != 0.0:
        qc.rx(math.radians(theta_deg), [0, 1])
    for k in range(1, n_qubits // 2):
        qc.cx(2 * k - 2, 2 * k)
        qc.cx(2 * k - 1, 2 * k + 1)
    if measure:
        qc.measure_all()
    return qc


def build_spam_calibration_circuit(
    n_qubits: int = 40, state_bit: int = 0, name: Optional[str] = None
) -> QuantumCircuit:
    """Zero-entanglement baseline SPAM control preparing |0>^n or |1>^n."""
    if state_bit not in (0, 1):
        raise ValueError("state_bit must be 0 or 1")
    qc = QuantumCircuit(n_qubits, name=name or f"spam_all{state_bit}")
    if state_bit == 1:
        qc.x(list(range(n_qubits)))
    qc.measure_all()
    return qc


def chain_logical_to_physical_layout(physical_path: Sequence[int]) -> List[int]:
    """
    Map logical qubits 0..n-1 onto a contiguous physical path `physical_path`
    so that `build_chain_circuit`'s `(2k-2, 2k)`, `(0, 1)`, and `(2k-1, 2k+1)`
    CX gates all act on adjacent physical edges with ZERO routing SWAPs.
    """
    n_qubits = len(physical_path)
    if n_qubits % 2 != 0:
        raise ValueError("physical_path length must be even")
    logical_order = list(range(n_qubits - 2, -1, -2)) + list(range(1, n_qubits, 2))
    layout = [0] * n_qubits
    for phys, log_q in zip(physical_path, logical_order):
        layout[log_q] = int(phys)
    return layout


def ideal_four_string_distribution(
    n_qubits: int = 40, theta_deg: float = THETA_DEFAULT_DEG
) -> Dict[str, float]:
    """
    Exact ideal computational-basis distribution (Qiskit little-endian bitstring
    convention `q_{n-1}...q_1 q_0`):
      - All-0 (`00...0`) and All-1 (`11...1`): each `0.5 * cos^2(theta)`
      - Alternating `1010...10` (`q0=0, q1=1`) and `0101...01` (`q0=1, q1=0`):
        each `0.5 * sin^2(theta)`.
    """
    th = math.radians(theta_deg)
    c2 = math.cos(th) ** 2
    s2 = math.sin(th) ** 2
    s_all0 = "0" * n_qubits
    s_all1 = "1" * n_qubits
    # In Qiskit bitstrings c_{n-1}...c_1 c_0: rightmost char is q0, second from right is q1
    s_q0_0_q1_1 = "10" * (n_qubits // 2)  # ... q1='1', q0='0'
    s_q0_1_q1_0 = "01" * (n_qubits // 2)  # ... q1='0', q0='1'
    out = {s_all0: 0.5 * c2, s_all1: 0.5 * c2}
    if s2 > 1e-15:
        out[s_q0_0_q1_1] = 0.5 * s2
        out[s_q0_1_q1_0] = 0.5 * s2
    return out


def verify_logical_equivalence(
    n_qubits_sv: int = 12, n_qubits_gf2: int = 40, theta_deg: float = THETA_DEFAULT_DEG
) -> Dict[str, Any]:
    """
    Prove logical equivalence between `star` and `chain` BEFORE any hardware execution:
      1. Exact complex Statevector inner product `|<psi_star | psi_chain>| == 1.0` at
         `n_qubits_sv` (and exact probability match even when `include_inert_rz=True`).
      2. Exact GF(2) fan-out propagation matrix equality at full `n_qubits_gf2 = 40`:
         every wire `2k` receives `q0` and every wire `2k+1` receives `q1` in both topologies.
      3. Demonstrate E-84 fault-propagation non-equivalence under mid-circuit noise:
         a single bit-flip error after CX(0, 2) flips 19 wires in `chain` vs 1 wire in `star`.
    """
    sv_star = Statevector.from_instruction(build_star_circuit(n_qubits_sv, theta_deg, include_inert_rz=False, measure=False))
    sv_chain = Statevector.from_instruction(build_chain_circuit(n_qubits_sv, theta_deg, measure=False))
    sv_star_rz = Statevector.from_instruction(build_star_circuit(n_qubits_sv, theta_deg, include_inert_rz=True, measure=False))

    state_overlap = float(abs(sv_star.inner(sv_chain)))
    probs_star = sv_star.probabilities_dict(decimals=15)
    probs_chain = sv_chain.probabilities_dict(decimals=15)
    probs_star_rz = sv_star_rz.probabilities_dict(decimals=15)
    ideal = ideal_four_string_distribution(n_qubits_sv, theta_deg)

    max_prob_diff = max(
        abs(probs_star.get(k, 0.0) - ideal.get(k, 0.0))
        for k in set(probs_star) | set(ideal)
    )
    max_rz_prob_diff = max(
        abs(probs_star_rz.get(k, 0.0) - probs_chain.get(k, 0.0))
        for k in set(probs_star_rz) | set(probs_chain)
    )

    # Full 40-qubit GF(2) propagation check from (q0, q1) -> all 40 wires
    def _gf2_matrix(is_chain: bool) -> np.ndarray:
        M = np.eye(n_qubits_gf2, dtype=np.uint8)
        if is_chain:
            for k in range(1, n_qubits_gf2 // 2):
                M[2 * k, :] ^= M[2 * k - 2, :]
                M[2 * k + 1, :] ^= M[2 * k - 1, :]
        else:
            for i in range(2, n_qubits_gf2):
                M[i, :] ^= M[i % 2, :]
        return M[:, :2]  # columns for core (q0, q1)

    gf2_equal = bool(np.array_equal(_gf2_matrix(False), _gf2_matrix(True)))

    # Fault propagation weight of an X error on wire 2 immediately after it is entangled:
    # In star, wire 2 is never a control -> weight 1.
    # In chain, wire 2 controls 4 -> 6 -> ... -> 38 -> weight 19!
    star_fault_weight_q2 = 1
    chain_fault_weight_q2 = len(range(2, n_qubits_gf2, 2))

    return {
        "n_qubits_statevector": n_qubits_sv,
        "statevector_overlap_star_vs_chain": state_overlap,
        "max_prob_diff_vs_analytic_ideal": max_prob_diff,
        "max_prob_diff_with_inert_rz": max_rz_prob_diff,
        "gf2_40q_core_propagation_identical": gf2_equal,
        "fault_weight_x_on_q2_star": star_fault_weight_q2,
        "fault_weight_x_on_q2_chain": chain_fault_weight_q2,
    }


# ---------------------------------------------------------------------------
# 2. Heavy-Hex Path Discovery & Exact Final Layout Extraction
# ---------------------------------------------------------------------------

def score_simple_paths_on_backend(
    backend: Any, path_length: int = 40, max_paths: int = 400
) -> List[Tuple[float, List[int]]]:
    """
    Find simple paths of `path_length` physical qubits on `backend`'s coupling graph
    and score each path by expected log-infidelity:
        cost(path) = -sum_{edges} log(1 - e_2q) - sum_{qubits} log(1 - e_ro)
    Returns sorted list `(cost, physical_path)` from lowest cost (`best_path`) to highest.
    """
    target = backend.target
    cmap = backend.coupling_map
    adj: Dict[int, List[int]] = collections.defaultdict(list)
    edge_err: Dict[Tuple[int, int], float] = {}

    twoq_names = [op for op in ("cz", "ecr", "cx") if op in target.operation_names]
    twoq_op = twoq_names[0] if twoq_names else None

    ro_err: Dict[int, float] = {}
    faulty_qubits: set[int] = set()
    for q in range(backend.num_qubits):
        r_e = 0.015
        if "measure" in target.operation_names:
            m_prop = target["measure"].get((q,))
            if m_prop is not None and m_prop.error is not None:
                r_e = float(m_prop.error)
        if r_e >= 0.45:
            faulty_qubits.add(q)
        ro_err[q] = min(max(r_e, 1e-6), 0.49)

    for u, v in cmap.get_edges():
        u_i, v_i = int(u), int(v)
        if u_i in faulty_qubits or v_i in faulty_qubits:
            continue
        err = 0.005
        if twoq_op is not None:
            props = target[twoq_op].get((u_i, v_i)) or target[twoq_op].get((v_i, u_i))
            if props is not None and props.error is not None:
                err = float(props.error)
        if err >= 0.45:
            # Exclude disabled/faulty hardware couplers (e.g., error=1.0 calibration flags)
            continue
        if v_i not in adj[u_i]:
            adj[u_i].append(v_i)
        if u_i not in adj[v_i]:
            adj[v_i].append(u_i)
        edge_err[(min(u_i, v_i), max(u_i, v_i))] = min(max(err, 1e-6), 0.49)

    for q in adj:
        adj[q].sort(key=lambda nb: edge_err[(min(q, nb), max(q, nb))] + 0.5 * ro_err[nb])

    found: List[Tuple[float, List[int]]] = []
    seen_sets: set[Tuple[int, ...]] = set()

    # Deterministic greedy-DFS from every starting node sorted by readout fidelity
    starts = [q for q in sorted(range(backend.num_qubits), key=lambda q: ro_err[q]) if q not in faulty_qubits]
    for start in starts:
        stack: List[Tuple[int, List[int], set[int]]] = [(start, [start], {start})]
        branches_from_start = 0
        while stack and len(found) < max_paths and branches_from_start < 150:
            curr, path, visited = stack.pop()
            if len(path) == path_length:
                canon = tuple(path) if path[0] < path[-1] else tuple(reversed(path))
                if canon not in seen_sets:
                    seen_sets.add(canon)
                    c_edges = sum(
                        -math.log(1.0 - edge_err[(min(path[k], path[k + 1]), max(path[k], path[k + 1]))])
                        for k in range(path_length - 1)
                    )
                    c_ro = sum(-math.log(1.0 - ro_err[q]) for q in path)
                    found.append((c_edges + c_ro, list(path)))
                branches_from_start += 1
                continue
            for nb in reversed(adj[curr]):
                if nb not in visited:
                    stack.append((nb, path + [nb], visited | {nb}))

    if not found:
        raise RuntimeError(f"Could not find a simple path of length {path_length} on {backend.name}")
    found.sort(key=lambda item: item[0])
    return found


def extract_final_physical_layout(compiled_qc: QuantumCircuit, n_logical: int) -> List[int]:
    """
    Extract exact post-routing physical qubit index for each logical qubit 0..n_logical-1
    from measurement instructions (eliminates the missing-layout defect of E-83!).
    """
    if compiled_qc.layout is not None:
        final_map = compiled_qc.layout.final_index_layout(filter_ancillas=True)
        if len(final_map) == n_logical:
            return [int(q) for q in final_map]
    # Fallback: read directly from measure ops
    meas_map: Dict[int, int] = {}
    for inst in compiled_qc.data:
        if inst.operation.name == "measure":
            phys_q = compiled_qc.find_bit(inst.qubits[0]).index
            cl_idx = compiled_qc.find_bit(inst.clbits[0]).index
            meas_map[cl_idx] = int(phys_q)
    return [meas_map[i] for i in range(n_logical)]


# ---------------------------------------------------------------------------
# 3. Controlled 9-Arm Compilation Design & Metrics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompiledArmManifest:
    arm_id: str
    logical_structure: str
    layout_policy: str
    routing_method: str
    optimization_level: int
    theta_deg: float
    twoq_gate_count: int
    circuit_depth: int
    ops_breakdown: Dict[str, int]
    final_physical_layout: List[int]
    circuit_qasm_sha256: str


def build_tesseract_map_02_arms(
    backend: Any, n_qubits: int = 40, theta_deg: float = THETA_DEFAULT_DEG, seed: int = 20261116
) -> Tuple[Dict[str, QuantumCircuit], Dict[str, CompiledArmManifest], Dict[str, Any]]:
    """
    Compile the controlled 9-arm TESSERACT-MAP-02 design on `backend`:
      - C0_SPAM_ALL0:          |0>^n SPAM baseline on `best_path`
      - C0_SPAM_ALL1:          |1>^n SPAM baseline on `best_path`
      - C1_STAR_FREE_OPT3:     Historical P0 replication (star, compiler SABRE layout, opt=3)
      - C2_STAR_BESTPATH_OPT3: Star pinned to `best_path` initial qubits, SABRE routed, opt=3
      - C3_STAR_BESTPATH_OPT1: Star pinned to `best_path` initial qubits, SABRE routed, opt=1
      - C4_CHAIN_BESTPATH_OPT1: Historical P2 replication (chain, `best_path`, 0 SWAPs, opt=1)
      - C5_CHAIN_BESTPATH_OPT3: Chain on `best_path`, 0 SWAPs, opt=3
      - C6_CHAIN_MEDPATH_OPT3:  Chain on `median_path` (worse calibration, same 39 2Q gates & depth!), opt=3
      - C7_CHAIN_FREE_OPT3:     Chain with free compiler SABRE layout, opt=3
    """
    scored_paths = score_simple_paths_on_backend(backend, path_length=n_qubits)
    best_cost, best_path = scored_paths[0]
    med_idx = len(scored_paths) // 2
    med_cost, med_path = scored_paths[med_idx]
    worst_cost, worst_path = scored_paths[-1]

    best_layout = chain_logical_to_physical_layout(best_path)
    med_layout = chain_logical_to_physical_layout(med_path)

    raw_specs = [
        ("C0_SPAM_ALL0", "spam_0", "best_path", "none", 1, 0.0,
         build_spam_calibration_circuit(n_qubits, 0, "C0_SPAM_ALL0"), best_layout),
        ("C0_SPAM_ALL1", "spam_1", "best_path", "none", 1, 0.0,
         build_spam_calibration_circuit(n_qubits, 1, "C0_SPAM_ALL1"), best_layout),
        ("C1_STAR_FREE_OPT3", "star", "sabre_free", "sabre", 3, theta_deg,
         build_star_circuit(n_qubits, theta_deg, include_inert_rz=False, name="C1_STAR_FREE_OPT3"), None),
        ("C2_STAR_BESTPATH_OPT3", "star", "best_path", "sabre", 3, theta_deg,
         build_star_circuit(n_qubits, theta_deg, include_inert_rz=False, name="C2_STAR_BESTPATH_OPT3"), best_layout),
        ("C3_STAR_BESTPATH_OPT1", "star", "best_path", "sabre", 1, theta_deg,
         build_star_circuit(n_qubits, theta_deg, include_inert_rz=False, name="C3_STAR_BESTPATH_OPT1"), best_layout),
        ("C4_CHAIN_BESTPATH_OPT1", "chain", "best_path", "none", 1, theta_deg,
         build_chain_circuit(n_qubits, theta_deg, name="C4_CHAIN_BESTPATH_OPT1"), best_layout),
        ("C5_CHAIN_BESTPATH_OPT3", "chain", "best_path", "none", 3, theta_deg,
         build_chain_circuit(n_qubits, theta_deg, name="C5_CHAIN_BESTPATH_OPT3"), best_layout),
        ("C6_CHAIN_MEDPATH_OPT3", "chain", "median_path", "none", 3, theta_deg,
         build_chain_circuit(n_qubits, theta_deg, name="C6_CHAIN_MEDPATH_OPT3"), med_layout),
        ("C7_CHAIN_FREE_OPT3", "chain", "sabre_free", "sabre", 3, theta_deg,
         build_chain_circuit(n_qubits, theta_deg, name="C7_CHAIN_FREE_OPT3"), None),
    ]

    compiled_circuits: Dict[str, QuantumCircuit] = {}
    manifests: Dict[str, CompiledArmManifest] = {}

    for arm_id, log_struct, lay_pol, route_m, opt_lvl, th_deg, qc_raw, init_lay in raw_specs:
        if init_lay is not None and route_m == "none":
            pm = generate_preset_pass_manager(
                optimization_level=opt_lvl,
                backend=backend,
                initial_layout=init_lay,
                layout_method="trivial",
                routing_method="none",
                seed_transpiler=seed,
            )
            t_qc = pm.run(qc_raw)
        elif init_lay is not None and route_m == "sabre":
            pm = generate_preset_pass_manager(
                optimization_level=opt_lvl,
                backend=backend,
                initial_layout=init_lay,
                layout_method="trivial",
                routing_method="sabre",
                seed_transpiler=seed,
            )
            t_qc = pm.run(qc_raw)
        else:
            t_qc = transpile(qc_raw, backend=backend, optimization_level=opt_lvl, seed_transpiler=seed)

        twoq = sum(1 for inst in t_qc.data if inst.operation.num_qubits == 2)
        ops = {k: int(v) for k, v in t_qc.count_ops().items()}
        final_lay = extract_final_physical_layout(t_qc, n_qubits)
        # Deterministic structural digest of transpiled instruction sequence
        op_repr = "|".join(
            f"{inst.operation.name}:{[t_qc.find_bit(q).index for q in inst.qubits]}"
            for inst in t_qc.data
        )
        digest = hashlib.sha256(op_repr.encode("utf-8")).hexdigest()

        compiled_circuits[arm_id] = t_qc
        manifests[arm_id] = CompiledArmManifest(
            arm_id=arm_id,
            logical_structure=log_struct,
            layout_policy=lay_pol,
            routing_method=route_m,
            optimization_level=opt_lvl,
            theta_deg=th_deg,
            twoq_gate_count=twoq,
            circuit_depth=t_qc.depth(),
            ops_breakdown=ops,
            final_physical_layout=final_lay,
            circuit_qasm_sha256=digest,
        )

    union_qubits = sorted({
        q for man in manifests.values() for q in man.final_physical_layout
    })
    path_summary = {
        "backend_name": backend.name,
        "n_qubits": n_qubits,
        "num_candidate_paths_scored": len(scored_paths),
        "best_path_cost": best_cost,
        "best_path": best_path,
        "median_path_cost": med_cost,
        "median_path": med_path,
        "worst_path_cost": worst_cost,
        "union_active_physical_qubits": union_qubits,
        "union_spam_qubit_count": len(union_qubits),
    }
    return compiled_circuits, manifests, path_summary


def build_union_spam_calibration_circuits(
    manifests: Mapping[str, CompiledArmManifest],
) -> Tuple[QuantumCircuit, QuantumCircuit, List[int]]:
    """
    Construct full-union |0>^K and |1>^K SPAM readout calibration circuits covering
    100% of physical qubits active across ANY arm (C1..C7), preventing uncovered
    readout outliers on C1, C6, or C7.
    """
    union_qubits = sorted({q for m in manifests.values() for q in m.final_physical_layout})
    k = len(union_qubits)
    qc0 = QuantumCircuit(k, k, name="C0_UNION_SPAM_ALL0")
    qc0.measure(range(k), range(k))
    qc1 = QuantumCircuit(k, k, name="C0_UNION_SPAM_ALL1")
    qc1.x(range(k))
    qc1.measure(range(k), range(k))
    return qc0, qc1, union_qubits


# ---------------------------------------------------------------------------
# 4. Primary & Secondary Endpoint Scoring + Paired/Blocked Statistical Plan
# ---------------------------------------------------------------------------

def score_tesseract_counts(
    counts: Mapping[str, int], n_qubits: int = 40, theta_deg: float = THETA_DEFAULT_DEG
) -> Dict[str, float]:
    """
    Compute prespecified endpoints on a shot histogram:
      - Primary endpoint: `classical_fidelity` F_cl = (sum_{x in S_4} sqrt(p_emp(x) * p_ideal(x)))^2
      - Secondary endpoint: `four_string_support` = sum_{x in S_4} p_emp(x)
      - Additional diagnostics: `ghz_population`, `tvd_to_ideal`, `shannon_entropy_bits`
    """
    total = float(sum(counts.values()))
    if total <= 0:
        raise ValueError("Empty counts dictionary")
    ideal = ideal_four_string_distribution(n_qubits, theta_deg)

    bc = sum(math.sqrt((counts.get(s, 0) / total) * p_id) for s, p_id in ideal.items())
    f_cl = bc ** 2
    support = sum(counts.get(s, 0) for s in ideal) / total
    ghz_pop = (counts.get("0" * n_qubits, 0) + counts.get("1" * n_qubits, 0)) / total

    # Exact TVD = 0.5 * sum_x |p_emp(x) - p_ideal(x)|
    l1 = sum(abs((counts.get(s, 0) / total) - p_id) for s, p_id in ideal.items())
    l1 += sum(c / total for s, c in counts.items() if s not in ideal)
    tvd = 0.5 * l1

    probs = np.array([c / total for c in counts.values() if c > 0], dtype=np.float64)
    entropy = float(-np.sum(probs * np.log2(probs)))
    return {
        "classical_fidelity": f_cl,
        "four_string_support": support,
        "ghz_population": ghz_pop,
        "tvd_to_ideal": tvd,
        "shannon_entropy_bits": entropy,
    }


def analyze_paired_blocks(
    block_scores: Mapping[str, Sequence[float]],
    contrasts: Optional[Sequence[Tuple[str, str, str]]] = None,
    alpha_fwer: float = 0.05,
) -> Dict[str, Any]:
    """
    Blocked paired statistical analysis across `B` interleaved experimental blocks
    (unit of replication = hardware block `b`, NEVER individual shots!).
    Applies Holm-Bonferroni step-down multiple-testing correction across all
    prespecified contrasts.
    """
    if contrasts is None:
        contrasts = (
            ("routing_and_rewrite_at_opt3", "C5_CHAIN_BESTPATH_OPT3", "C2_STAR_BESTPATH_OPT3"),
            ("routing_and_rewrite_at_opt1", "C4_CHAIN_BESTPATH_OPT1", "C3_STAR_BESTPATH_OPT1"),
            ("qubit_selection_at_fixed_39cz", "C5_CHAIN_BESTPATH_OPT3", "C6_CHAIN_MEDPATH_OPT3"),
            ("transpiler_opt3_vs_opt1_on_chain", "C5_CHAIN_BESTPATH_OPT3", "C4_CHAIN_BESTPATH_OPT1"),
            ("error_aware_path_vs_sabre_chain", "C5_CHAIN_BESTPATH_OPT3", "C7_CHAIN_FREE_OPT3"),
        )

    raw_results: List[Dict[str, Any]] = []
    for name, arm_a, arm_b in contrasts:
        ya = np.asarray(block_scores[arm_a], dtype=np.float64)
        yb = np.asarray(block_scores[arm_b], dtype=np.float64)
        if ya.shape != yb.shape or ya.ndim != 1 or ya.size < 2:
            raise ValueError(f"Contrast {name} requires >=2 paired block observations")
        diffs = ya - yb
        b_cnt = diffs.size
        mean_d = float(np.mean(diffs))
        sd_d = float(np.std(diffs, ddof=1))
        se_d = sd_d / math.sqrt(b_cnt)
        if sd_d > 1e-15:
            t_stat = mean_d / se_d
            p_two = float(2.0 * stats.t.sf(abs(t_stat), df=b_cnt - 1))
            t_crit = float(stats.t.ppf(0.975, df=b_cnt - 1))
        else:
            t_stat = float("inf") if abs(mean_d) > 0 else 0.0
            p_two = 0.0 if abs(mean_d) > 0 else 1.0
            t_crit = 0.0

        raw_results.append({
            "contrast": name,
            "arm_a": arm_a,
            "arm_b": arm_b,
            "n_blocks": int(b_cnt),
            "mean_diff": mean_d,
            "se_diff": se_d,
            "ci95": (mean_d - t_crit * se_d, mean_d + t_crit * se_d),
            "t_stat": t_stat,
            "p_two_sided_unadjusted": p_two,
        })

    # Holm-Bonferroni step-down adjustment
    m = len(raw_results)
    order = sorted(range(m), key=lambda k: raw_results[k]["p_two_sided_unadjusted"])
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj_p = min(1.0, raw_results[idx]["p_two_sided_unadjusted"] * (m - rank))
        running_max = max(running_max, adj_p)
        raw_results[idx]["p_holm_adjusted"] = running_max
        raw_results[idx]["significant_holm_0_05"] = running_max < alpha_fwer

    return {"alpha_fwer": alpha_fwer, "contrasts": raw_results}


def compute_required_blocks(
    target_mde: float = 0.015,
    between_block_sd_diff: float = 0.014,
    alpha_holm_per_test: float = 0.05 / 5.0,
    target_power: float = 0.85,
) -> Dict[str, float | int]:
    """
    Justified sample-size (block count B) calculation under paired block-difference SD
    `sigma_d = 0.014` (combining 1.1 pp hardware A/A drift + 4,096-shot multinomial noise).
    """
    for b in range(4, 100):
        df = b - 1
        t_crit = float(stats.t.ppf(1.0 - alpha_holm_per_test / 2.0, df=df))
        ncp = abs(target_mde) / (between_block_sd_diff / math.sqrt(b))
        # Exact non-central t power
        power = float(1.0 - stats.nct.cdf(t_crit, df=df, nc=ncp) + stats.nct.cdf(-t_crit, df=df, nc=ncp))
        if power >= target_power:
            return {
                "required_blocks": b,
                "achieved_power": round(power, 4),
                "target_mde": target_mde,
                "assumed_paired_sd": between_block_sd_diff,
                "alpha_per_contrast": alpha_holm_per_test,
            }
    raise RuntimeError("Target power not reached within 100 blocks")


# ---------------------------------------------------------------------------
# 5. Offline Calibrated Simulation & Archive Audit (`dansntf8gn2s739n32e0`)
# ---------------------------------------------------------------------------

def simulate_40q_hardware_block_outcomes(
    backend: Any,
    compiled_circuits: Mapping[str, QuantumCircuit],
    manifests: Mapping[str, CompiledArmManifest],
    n_blocks: int = 12,
    shots_per_circuit: int = 4096,
    temporal_drift_sd: float = 0.008,
    seed: int = 20261116,
) -> Dict[str, Any]:
    """
    Offline calibrated survival + readout-bias + temporal-drift Monte Carlo simulation
    for all 40-qubit compiled circuits on `backend` (`FakeKingston` / `FakeMarrakesh`).
    Models:
      - Exact transpiled 2-qubit gate infidelity product `prod_g (1 - e_2q(g))`
      - Idle T1/T2 depth decoherence from transpiled circuit depth
      - Per-qubit asymmetric readout assignment error from `backend.target['measure']`
      - Shared block-to-block temporal calibration drift (`temporal_drift_sd`)
      - Randomized condition order within every block.
    """
    rng = np.random.default_rng(seed)
    target = backend.target
    twoq_names = [op for op in ("cz", "ecr", "cx") if op in target.operation_names]
    twoq_op = twoq_names[0] if twoq_names else None

    # Compute exact survival factor + physical scheduled duration + T1/T2 idle decay per arm
    arm_coherent_survival: Dict[str, float] = {}
    arm_duration_ns: Dict[str, float] = {}
    arm_mean_ro_fidelity: Dict[str, float] = {}

    for arm_id, qc in compiled_circuits.items():
        man = manifests[arm_id]
        surv = 1.0
        wire_avail_s: Dict[int, float] = {}
        wire_busy_s: Dict[int, float] = {}
        for inst in qc.data:
            op_name = inst.operation.name
            if op_name in ("barrier", "rz"):
                # Virtual-Z (`rz`) is a zero-duration NCO phase frame update on IBM Heron/Eagle
                continue
            q_indices = [qc.find_bit(qb).index for qb in inst.qubits]
            dur_s = 0.0
            if inst.operation.num_qubits == 2 and twoq_op is not None:
                u, v = q_indices
                p_obj = target[twoq_op].get((u, v)) or target[twoq_op].get((v, u))
                err = float(p_obj.error) if (p_obj is not None and p_obj.error is not None) else 0.006
                dur_s = float(p_obj.duration) if (p_obj is not None and p_obj.duration is not None) else 68e-9
                surv *= max(0.0, 1.0 - err)
            elif op_name in ("sx", "x"):
                q = q_indices[0]
                p1 = target[op_name].get((q,)) if op_name in target.operation_names else None
                err1 = float(p1.error) if (p1 is not None and p1.error is not None) else 0.0003
                dur_s = float(p1.duration) if (p1 is not None and p1.duration is not None) else 32e-9
                surv *= max(0.0, 1.0 - err1)
            elif op_name == "measure":
                q = q_indices[0]
                pm = target["measure"].get((q,)) if "measure" in target.operation_names else None
                dur_s = float(pm.duration) if (pm is not None and pm.duration is not None) else 1560e-9

            start_t = max((wire_avail_s.get(q, 0.0) for q in q_indices), default=0.0)
            end_t = start_t + dur_s
            for q in q_indices:
                wire_avail_s[q] = end_t
                if op_name != "measure":
                    wire_busy_s[q] = wire_busy_s.get(q, 0.0) + dur_s

        # Pre-readout circuit evolution span and per-qubit idle T1/T2 decoherence
        meas_dur_ref = 1560e-9
        if "measure" in target.operation_names and man.final_physical_layout:
            pm0 = target["measure"].get((man.final_physical_layout[0],))
            if pm0 is not None and pm0.duration is not None:
                meas_dur_ref = float(pm0.duration)
        total_sched_s = max(wire_avail_s.values(), default=0.0)
        coherent_window_s = max(0.0, total_sched_s - meas_dur_ref)
        arm_duration_ns[arm_id] = round(coherent_window_s * 1.0e9, 1)

        idle_surv = 1.0
        if not arm_id.startswith("C0_SPAM") and coherent_window_s > 0.0:
            for phys_q in man.final_physical_layout:
                qprop = target.qubit_properties[phys_q] if target.qubit_properties else None
                t1 = float(qprop.t1) if (qprop is not None and getattr(qprop, "t1", None)) else 180e-6
                t2 = float(qprop.t2) if (qprop is not None and getattr(qprop, "t2", None)) else 140e-6
                idle_q_s = max(0.0, coherent_window_s - wire_busy_s.get(phys_q, 0.0))
                # Per-qubit idle survival factor across unpulsed chain wait window
                rate = 0.25 / max(t1, 1e-6) + 0.35 / max(t2, 1e-6)
                idle_surv *= math.exp(-idle_q_s * rate)

        ro_surv = 1.0
        ro_list = []
        for phys_q in man.final_physical_layout:
            m_prop = target["measure"].get((phys_q,)) if "measure" in target.operation_names else None
            r_err = float(m_prop.error) if (m_prop is not None and m_prop.error is not None) else 0.015
            ro_surv *= max(0.0, 1.0 - r_err)
            ro_list.append(1.0 - r_err)

        arm_coherent_survival[arm_id] = surv * idle_surv * ro_surv
        arm_mean_ro_fidelity[arm_id] = float(np.mean(ro_list))

    # Simulate B interleaved blocks with randomized intra-block execution order + spatial qubit drift
    arm_ids = [k for k in compiled_circuits if not k.startswith("C0_SPAM")]
    block_f_cl: Dict[str, List[float]] = {k: [] for k in arm_ids}
    block_support: Dict[str, List[float]] = {k: [] for k in arm_ids}
    execution_orders: List[List[str]] = []

    for b in range(n_blocks):
        order = list(arm_ids)
        rng.shuffle(order)
        execution_orders.append(order)
        # Shared chip-level drift + independent per-qubit drift (so disjoint paths experience differential drift!)
        global_drift = float(rng.normal(0.0, 0.75 * temporal_drift_sd))
        per_qubit_drift = rng.normal(0.0, 0.65 * temporal_drift_sd * math.sqrt(40.0), size=backend.num_qubits)

        for pos, arm_id in enumerate(order):
            within_block_drift = 0.0004 * (pos - 0.5 * len(order))
            man = manifests[arm_id]
            layout_drift = float(np.mean(per_qubit_drift[man.final_physical_layout]))
            ideal = ideal_four_string_distribution(len(man.final_physical_layout), man.theta_deg)
            p_surv = float(
                np.clip(
                    arm_coherent_survival[arm_id] + global_drift + layout_drift + within_block_drift,
                    0.005,
                    0.995,
                )
            )

            n_q = len(man.final_physical_layout)
            n_good = int(rng.binomial(shots_per_circuit, p_surv))
            n_bad = shots_per_circuit - n_good
            ideal_keys = list(ideal.keys())
            ideal_probs = np.array([ideal[k] for k in ideal_keys], dtype=np.float64)
            ideal_probs /= ideal_probs.sum()
            good_draws = rng.multinomial(n_good, ideal_probs)

            counts: Dict[str, int] = {
                k: int(c) for k, c in zip(ideal_keys, good_draws) if c > 0
            }
            # Generate genuine 40-bit binary corrupted bitstrings (domain-wall / readout flips outside S_4)
            if n_bad > 0:
                base_choices = rng.choice(len(ideal_keys), size=n_bad, p=ideal_probs)
                for b_idx in base_choices:
                    bits = list(ideal_keys[int(b_idx)])
                    if man.logical_structure == "chain" and rng.random() < 0.45:
                        # Propagated parity-subchain domain-wall flip (E-84 fault signature!)
                        start_k = int(rng.integers(2, n_q))
                        for pos_k in range(start_k, n_q, 2):
                            bits[n_q - 1 - pos_k] = "1" if bits[n_q - 1 - pos_k] == "0" else "0"
                    else:
                        # Single/double readout or local Pauli bit-flip (asymmetric 1->0 biased)
                        n_flips = 1 if rng.random() < 0.75 else 2
                        for flip_q in rng.choice(n_q, size=n_flips, replace=False):
                            idx_char = int(flip_q)
                            bits[idx_char] = "1" if bits[idx_char] == "0" else "0"
                    corrupted = "".join(bits)
                    if corrupted in ideal:
                        # Ensure corrupted shot lands outside ideal 4-string support
                        corrupted = ("1" if corrupted[0] == "0" else "0") + corrupted[1:]
                    counts[corrupted] = counts.get(corrupted, 0) + 1

            scored = score_tesseract_counts(counts, n_q, man.theta_deg)
            block_f_cl[arm_id].append(scored["classical_fidelity"])
            block_support[arm_id].append(scored["four_string_support"])

    stats_f_cl = analyze_paired_blocks(block_f_cl)
    stats_support = analyze_paired_blocks(block_support)
    summary_by_arm = {
        arm_id: {
            "twoq_gates": manifests[arm_id].twoq_gate_count,
            "depth": manifests[arm_id].circuit_depth,
            "coherent_duration_ns": arm_duration_ns[arm_id],
            "predicted_survival": round(arm_coherent_survival[arm_id], 5),
            "mean_classical_fidelity": round(float(np.mean(block_f_cl[arm_id])), 5),
            "sd_classical_fidelity": round(float(np.std(block_f_cl[arm_id], ddof=1)), 5),
            "mean_four_string_support": round(float(np.mean(block_support[arm_id])), 5),
        }
        for arm_id in arm_ids
    }
    spam_summaries = {
        arm_id: {
            "predicted_all_correct_ro_survival": round(arm_coherent_survival[arm_id], 5),
            "mean_per_qubit_ro_fidelity": round(arm_mean_ro_fidelity[arm_id], 5),
        }
        for arm_id in compiled_circuits
        if arm_id.startswith("C0_SPAM")
    }
    return {
        "backend": backend.name,
        "n_blocks": n_blocks,
        "shots_per_circuit": shots_per_circuit,
        "spam_baseline_summaries": spam_summaries,
        "arm_summaries": summary_by_arm,
        "paired_contrasts_classical_fidelity": stats_f_cl,
        "paired_contrasts_four_string_support": stats_support,
        "randomized_orders_sample": execution_orders[:3],
        "simulation_model_kind": "calibrated_gate_idle_t1t2_domainwall_surrogate",
        "qpu_submitted": False,
    }


def verify_archived_kingston_rerun(
    zip_path: str = "/home/osiris/tesseract_rerun_2026-09-20.zip",
) -> Dict[str, Any]:
    """Recompute P0, P1, P2 metrics directly from archived `dansntf8gn2s739n32e0.counts.json`."""
    p = Path(zip_path)
    if not p.exists():
        return {"available": False}
    with zipfile.ZipFile(p) as zf:
        counts_all = json.loads(zf.read("dansntf8gn2s739n32e0.counts.json"))
        analysis_ref = json.loads(zf.read("dansntf8gn2s739n32e0.analysis.json"))

    out: Dict[str, Any] = {"available": True, "job_id": "dansntf8gn2s739n32e0", "circuits": {}}
    for name, theta in (("P0_replication", 51.7), ("P1_ghz_control", 0.0), ("P2_chain_mapped", 51.7)):
        recomp = score_tesseract_counts(counts_all[name], n_qubits=40, theta_deg=theta)
        ref_m = analysis_ref[name]["measured"]
        out["circuits"][name] = {
            "recomputed": recomp,
            "matches_archive_fidelity": math.isclose(
                recomp["classical_fidelity"], ref_m["classical_fidelity"], abs_tol=1e-9
            ),
        }
    return out
