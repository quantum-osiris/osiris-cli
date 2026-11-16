"""
Offline Quantinuum H2 / Helios HQC Workload Estimator & Pure-NumPy BP+OSD Hypergraph Decoder.

Strictly offline preparation module for:
  - Track A: Quantinuum Startup Partner Program (Nexus / TKET / Helios milestones)
  - Track B: DOE OLCF QCUP Open-Science Proposal (WP1, WP2, WP3 workload & myOLCF credit sizing)

Contains ZERO network calls and ZERO QPU job submissions.

Quantinuum H-Series / Helios Hardware Quantum Credit (HQC) Formula:
    HQC = 5.0 + C * (N_1q + 10 * N_2q + 5 * N_m) / 5000.0
where:
  - C    : number of shots per circuit
  - N_1q : physical single-qubit phased-X / RY / RX gates (`rz` is virtual phase tracking = 0 HQC)
  - N_2q : native two-qubit ZZPhase / RZZ entangling gates (0 SWAPs under QCCD all-to-all transport)
  - N_m  : state-init + mid-circuit measure/reset + terminal measurement operations
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from qiskit import QuantumCircuit, transpile

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HP004_MOD_PATH = _REPO_ROOT / "experiments" / "hp004" / "hp004_joint_stabilizer.py"


# ---------------------------------------------------------------------------
# 1. CSS Code Constructors: Steane [[7,1,3]], Tesseract [[16,6,4]], BB [[72,12,6]]
# ---------------------------------------------------------------------------

def gf2_rank(matrix: np.ndarray) -> int:
    """Exact rank of an integer/binary matrix over GF(2)."""
    a = (np.asarray(matrix, dtype=np.uint8) & 1).copy()
    rows, cols = a.shape
    r = 0
    for c in range(cols):
        pivots = np.nonzero(a[r:, c])[0]
        if pivots.size == 0:
            continue
        piv = r + int(pivots[0])
        if piv != r:
            a[[r, piv]] = a[[piv, r]]
        for rr in np.nonzero(a[:, c])[0]:
            if rr != r:
                a[rr] ^= a[r]
        r += 1
        if r == rows:
            break
    return r


def gf2_nullspace(h_mat: np.ndarray) -> np.ndarray:
    """Basis rows for ker(H) over GF(2): vectors v such that H @ v^T = 0 mod 2."""
    a = (np.asarray(h_mat, dtype=np.uint8) & 1).copy()
    rows, cols = a.shape
    pivot_cols: List[int] = []
    r = 0
    for c in range(cols):
        pivots = np.nonzero(a[r:, c])[0]
        if pivots.size == 0:
            continue
        piv = r + int(pivots[0])
        if piv != r:
            a[[r, piv]] = a[[piv, r]]
        for rr in np.nonzero(a[:, c])[0]:
            if rr != r:
                a[rr] ^= a[r]
        pivot_cols.append(c)
        r += 1
        if r == rows:
            break

    free_cols = [c for c in range(cols) if c not in pivot_cols]
    basis = np.zeros((len(free_cols), cols), dtype=np.uint8)
    for idx, fc in enumerate(free_cols):
        basis[idx, fc] = 1
        for pr, pc in enumerate(pivot_cols):
            if a[pr, fc]:
                basis[idx, pc] = 1
    return basis


def css_logical_operators(hx: np.ndarray, hz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return logical X (ker(HZ) \\ rowspace(HX)) and logical Z (ker(HX) \\ rowspace(HZ)) bases."""
    if np.any((hx @ hz.T) % 2 != 0):
        raise ValueError("CSS orthogonality violated: HX @ HZ^T != 0 mod 2")

    def quotient_basis(ker_basis: np.ndarray, stab_rows: np.ndarray) -> np.ndarray:
        cur = (stab_rows.astype(np.uint8) & 1).copy()
        cur_rank = gf2_rank(cur)
        logicals: List[np.ndarray] = []
        for cand in ker_basis:
            trial = np.vstack([cur, cand[None, :]]) if cur.size else cand[None, :]
            nr = gf2_rank(trial)
            if nr > cur_rank:
                cur = trial
                cur_rank = nr
                logicals.append(cand.copy())
        return np.array(logicals, dtype=np.uint8)

    lz = quotient_basis(gf2_nullspace(hx), hz)
    lx = quotient_basis(gf2_nullspace(hz), hx)
    return lx, lz


def build_steane_7_1_3() -> Tuple[np.ndarray, np.ndarray]:
    """Steane [[7, 1, 3]] 2D color code (weight-4 checks, max column weight 3 hyperedges)."""
    h = np.array(
        [
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    return h.copy(), h.copy()


def build_tesseract_16_6_4() -> Tuple[np.ndarray, np.ndarray]:
    """4D hypercube RM(1,4) CSS [[16, 6, 4]] Tesseract code (5 independent weight-8 hypercube facets)."""
    coords = np.array([[(i >> b) & 1 for b in range(4)] for i in range(16)], dtype=np.uint8)
    facets = []
    for axis in range(4):
        facets.append((coords[:, axis] == 0).astype(np.uint8))
        facets.append((coords[:, axis] == 1).astype(np.uint8))
    h_indep = np.vstack([facets[0], facets[1], facets[2], facets[4], facets[6]])
    return h_indep.copy(), h_indep.copy()


def build_bivariate_bicycle_72_12_6() -> Tuple[np.ndarray, np.ndarray]:
    """Bravyi et al. (2024) [[72, 12, 6]] weight-6 Bivariate Bicycle qLDPC code on Z_6 x Z_6.

    Fits inside Quantinuum Helios (98 physical ions): 72 data ions + 24 reused ancilla ions = 96 ions.
    """
    def shift(size: int, power: int) -> np.ndarray:
        m = np.zeros((size, size), dtype=np.uint8)
        for i in range(size):
            m[i, (i + power) % size] = 1
        return m

    l_dim, m_dim = 6, 6
    ix = np.eye(l_dim, dtype=np.uint8)
    iy = np.eye(m_dim, dtype=np.uint8)
    x = np.kron(shift(l_dim, 1), iy).astype(np.uint8)
    y = np.kron(ix, shift(m_dim, 1)).astype(np.uint8)
    x2 = (x @ x) % 2
    x3 = (x2 @ x) % 2
    y2 = (y @ y) % 2
    y3 = (y2 @ y) % 2

    a_poly = (x3 ^ y ^ y2).astype(np.uint8)
    b_poly = (y3 ^ x ^ x2).astype(np.uint8)
    hx = np.hstack([a_poly, b_poly]).astype(np.uint8)
    hz = np.hstack([b_poly.T, a_poly.T]).astype(np.uint8)
    return hx, hz


# ---------------------------------------------------------------------------
# 2. Pure-NumPy Sum-Product BP + Ordered Statistics Decoding (BP + OSD-CS)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BPOSDResult:
    correction: np.ndarray
    bp_converged: bool
    bp_iterations: int
    used_osd: bool
    weight: int


def decode_bp_osd(
    h_mat: np.ndarray,
    syndrome: np.ndarray,
    channel_probs: Union[np.ndarray, float] = 0.01,
    max_iter: int = 30,
    damping: float = 0.20,
    osd_order: int = 12,
) -> BPOSDResult:
    """Log-domain Sum-Product Belief Propagation + Combination-Sweep OSD (`BP+OSD-CS`) over GF(2).

    - Runs damped tanh/arctanh Sum-Product message passing on the Tanner graph.
    - When BP converges to a weight-<=1 correction, returns immediately.
    - When BP fails to converge OR returns weight >= 2 on a loopy graph, runs `OSD-CS` over the
      BP reliability ordering and selects the minimum-prior-cost valid syndrome solution.
    """
    h = (np.asarray(h_mat, dtype=np.uint8) & 1)
    s = (np.asarray(syndrome, dtype=np.uint8) & 1).reshape(-1)
    m_checks, n_vars = h.shape
    if s.shape[0] != m_checks:
        raise ValueError(f"Syndrome length {s.shape[0]} != number of checks {m_checks}")

    if np.all(s == 0):
        return BPOSDResult(np.zeros(n_vars, dtype=np.uint8), True, 0, False, 0)

    if np.ndim(channel_probs) == 0:
        p = np.full(n_vars, float(channel_probs), dtype=np.float64)
    else:
        p = np.asarray(channel_probs, dtype=np.float64).reshape(n_vars)
    p = np.clip(p, 1e-12, 0.499999)
    llr0 = np.log((1.0 - p) / p)  # positive prior NLL cost per flipped qubit

    check_nbrs = [np.nonzero(h[i])[0] for i in range(m_checks)]
    var_nbrs = [np.nonzero(h[:, j])[0] for j in range(n_vars)]

    m_cv = np.zeros((m_checks, n_vars), dtype=np.float64)
    m_vc = np.zeros((m_checks, n_vars), dtype=np.float64)
    for j in range(n_vars):
        m_vc[var_nbrs[j], j] = llr0[j]

    posterior_llr = llr0.copy()
    bp_cand: Optional[np.ndarray] = None
    bp_conv_iter = max_iter

    for it in range(1, max_iter + 1):
        # Exact log-domain Sum-Product check-to-variable update: 2 * atanh((-1)^s_i * prod tanh(m_vc / 2))
        for i in range(m_checks):
            nbrs = check_nbrs[i]
            if nbrs.size == 0:
                continue
            t_vals = np.tanh(0.5 * np.clip(m_vc[i, nbrs], -30.0, 30.0))
            t_vals = np.where(np.abs(t_vals) < 1e-15, 1e-15, t_vals)
            sign_s = -1.0 if s[i] else 1.0
            total_prod = sign_s * float(np.prod(t_vals))
            for k_loc, j in enumerate(nbrs):
                extrinsic = np.clip(total_prod / float(t_vals[k_loc]), -0.999999999999, 0.999999999999)
                m_cv[i, j] = 2.0 * math.atanh(extrinsic)

        for j in range(n_vars):
            checks = var_nbrs[j]
            total_llr = llr0[j] + float(np.sum(m_cv[checks, j]))
            posterior_llr[j] = total_llr
            for i in checks:
                new_msg = total_llr - m_cv[i, j]
                m_vc[i, j] = (1.0 - damping) * new_msg + damping * m_vc[i, j]

        e_hat = (posterior_llr < 0.0).astype(np.uint8)
        if np.array_equal((h @ e_hat) % 2, s):
            bp_cand = e_hat.copy()
            bp_conv_iter = it
            if int(e_hat.sum()) <= 1:
                return BPOSDResult(bp_cand, True, it, False, int(bp_cand.sum()))
            break

    # OSD-CS post-processing over reliability-ordered columns
    col_order = np.argsort(posterior_llr)
    h_perm = h[:, col_order].copy()
    aug = np.hstack([h_perm, s[:, None]]).astype(np.uint8)

    pivot_cols: List[int] = []
    r = 0
    for c in range(n_vars):
        pivots = np.nonzero(aug[r:, c])[0]
        if pivots.size == 0:
            continue
        piv = r + int(pivots[0])
        if piv != r:
            aug[[r, piv]] = aug[[piv, r]]
        for rr in np.nonzero(aug[:, c])[0]:
            if rr != r:
                aug[rr] ^= aug[r]
        pivot_cols.append(c)
        r += 1
        if r == m_checks:
            break

    rhs_piv = aug[:r, n_vars].copy()
    free_cols = [c for c in range(n_vars) if c not in pivot_cols]
    dep = aug[:r, free_cols] if free_cols else np.zeros((r, 0), dtype=np.uint8)
    prior_cost_perm = llr0[col_order]

    def eval_free_assignment(active_free_indices: Sequence[int]) -> np.ndarray:
        e_perm = np.zeros(n_vars, dtype=np.uint8)
        piv_vals = rhs_piv.copy()
        for f_idx in active_free_indices:
            fc = free_cols[f_idx]
            e_perm[fc] = 1
            piv_vals ^= dep[:, f_idx]
        for p_idx, pc in enumerate(pivot_cols):
            e_perm[pc] = piv_vals[p_idx]
        return e_perm

    best_perm = eval_free_assignment(())
    best_nll = float(np.dot(best_perm, prior_cost_perm))

    k_sweep1 = min(len(free_cols), max(0, osd_order))
    k_sweep2 = min(len(free_cols), min(8, max(0, osd_order)))

    candidates: List[Tuple[int, ...]] = [(i,) for i in range(k_sweep1)]
    for i in range(k_sweep2):
        for j in range(i + 1, k_sweep2):
            candidates.append((i, j))

    for active in candidates:
        cand_perm = eval_free_assignment(active)
        nll = float(np.dot(cand_perm, prior_cost_perm))
        if nll < best_nll - 1e-12:
            best_perm = cand_perm
            best_nll = nll

    e_osd = np.zeros(n_vars, dtype=np.uint8)
    e_osd[col_order] = best_perm

    if bp_cand is not None:
        bp_nll = float(np.dot(bp_cand, llr0))
        if bp_nll <= best_nll + 1e-12:
            return BPOSDResult(bp_cand, True, bp_conv_iter, False, int(bp_cand.sum()))
        return BPOSDResult(e_osd, True, bp_conv_iter, True, int(e_osd.sum()))

    return BPOSDResult(e_osd, False, max_iter, True, int(e_osd.sum()))


# ---------------------------------------------------------------------------
# 3. Quantinuum H2 / Helios HQC Compiler & Realistic QCUP Allocation Sizing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HQCWorkloadRow:
    work_package: str
    protocol: str
    target_system: str
    physical_qubits: int
    circuits: int
    shots_per_circuit: int
    total_shots: int
    mean_n_1q: float
    mean_n_2q: float
    mean_n_m: float
    routing_swaps: int
    hqc_per_circuit: float
    total_hqc: float


def compute_hqc_for_circuit(qc: QuantumCircuit, shots: int) -> Dict[str, float]:
    """Decompose `qc` into trapped-ion basis (`rx`, `ry`, `rz`, `rzz`) with 0 routing SWAPs and compute HQC."""
    has_reset = any(inst.operation.name == "reset" for inst in qc.data)
    if not has_reset:
        t_qc = transpile(
            qc,
            basis_gates=["rx", "ry", "rz", "rzz"],
            optimization_level=2,
            seed_transpiler=42,
        )
    else:
        t_qc = transpile(
            qc,
            basis_gates=["rx", "ry", "rz", "rzz", "reset"],
            optimization_level=1,
            seed_transpiler=42,
        )

    n_1q_phys = 0  # rx, ry (`rz` is virtual phase frame update = 0 HQC on Quantinuum H2/Helios)
    n_rz_virtual = 0
    n_2q = 0
    n_meas_reset = 0
    active_qubits = set()

    for inst in t_qc.data:
        name = inst.operation.name
        for q in inst.qubits:
            active_qubits.add(t_qc.find_bit(q).index)
        if name in ("rx", "ry"):
            n_1q_phys += 1
        elif name == "rz":
            n_rz_virtual += 1
        elif name == "rzz":
            n_2q += 1
        elif name in ("measure", "reset"):
            n_meas_reset += 1

    n_m = len(active_qubits) + n_meas_reset
    hqc = 5.0 + (shots * (n_1q_phys + 10.0 * n_2q + 5.0 * n_m)) / 5000.0
    return {
        "active_qubits": float(len(active_qubits)),
        "n_1q": float(n_1q_phys),
        "n_rz_virtual": float(n_rz_virtual),
        "n_2q": float(n_2q),
        "n_m": float(n_m),
        "hqc": hqc,
    }


def _build_wp1_k3_all_to_all_sample() -> QuantumCircuit:
    """12-qubit (k=3 pairs, d=8 Schmidt-rank extension) Hayden-Preskill circuit under QCCD all-to-all connectivity."""
    qc = QuantumCircuit(12, 6)
    for r_q, m_q in [(0, 3), (1, 4), (2, 5), (6, 9), (7, 10), (8, 11)]:
        qc.h(r_q)
        qc.cx(r_q, m_q)
    qc.barrier()
    scram_pairs = [(3, 6), (4, 7), (5, 8), (3, 7), (4, 8), (5, 6), (3, 5), (6, 8), (4, 6), (5, 7)]
    for idx, (a, b) in enumerate(scram_pairs):
        qc.h(a)
        if idx % 2 == 0:
            qc.s(b)
        qc.cz(a, b)
    qc.barrier()
    dec_pairs = [(6, 9), (7, 10), (8, 11), (6, 10), (7, 11), (8, 9), (6, 7), (7, 8), (9, 10), (10, 11), (6, 9), (7, 10), (8, 11), (6, 11)]
    for a, b in dec_pairs:
        qc.h(b)
        qc.cx(a, b)
    qc.barrier()
    qc.measure([0, 6, 1, 7, 2, 8], [0, 1, 2, 3, 4, 5])
    return qc


def _build_wp2_transport_circuit(n_qubits: int, depth_layers: int = 4) -> QuantumCircuit:
    """All-to-all nonlocal random-graph parameterized circuit on `n_qubits` (50Q H2 or 96Q Helios)."""
    rng = np.random.default_rng(20261116 + n_qubits)
    qc = QuantumCircuit(n_qubits, n_qubits)
    for q in range(n_qubits):
        qc.h(q)
        qc.ry(0.73, q)
    for layer in range(depth_layers):
        perm = rng.permutation(n_qubits)
        for i in range(0, n_qubits - 1, 2):
            a, b = int(perm[i]), int(perm[i + 1])
            qc.rzz(0.618 * (layer + 1), a, b)
            qc.rx(0.41, a)
            qc.ry(0.29, b)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def _build_wp3_syndrome_circuit(hx: np.ndarray, hz: np.ndarray, rounds: int, n_ancillas: int) -> QuantumCircuit:
    """Multi-round CSS syndrome extraction with QCCD ancilla reuse (`measure` + `reset` between half-rounds)."""
    n_data = hx.shape[1]
    n_total = n_data + n_ancillas
    n_checks = hx.shape[0]
    total_clbits = rounds * 2 * n_checks + n_data
    qc = QuantumCircuit(n_total, total_clbits)
    cl_idx = 0

    for _ in range(rounds):
        for start in range(0, n_checks, n_ancillas):
            batch = range(start, min(n_checks, start + n_ancillas))
            for loc_a, chk_i in enumerate(batch):
                anc = n_data + loc_a
                qc.h(anc)
                for dq in np.nonzero(hx[chk_i])[0]:
                    qc.cx(anc, int(dq))
                qc.h(anc)
                qc.measure(anc, cl_idx)
                qc.reset(anc)
                cl_idx += 1
        for start in range(0, n_checks, n_ancillas):
            batch = range(start, min(n_checks, start + n_ancillas))
            for loc_a, chk_i in enumerate(batch):
                anc = n_data + loc_a
                for dq in np.nonzero(hz[chk_i])[0]:
                    qc.cx(int(dq), anc)
                qc.measure(anc, cl_idx)
                qc.reset(anc)
                cl_idx += 1

    for dq in range(n_data):
        qc.measure(dq, cl_idx)
        cl_idx += 1
    return qc


def evaluate_qcup_workloads_and_decoders() -> Dict[str, Any]:
    """Compute realistic QRUC-sized Phase-1 HQC estimates (~16,500 HQC) and verify BP+OSD decoders."""
    spec = importlib.util.spec_from_file_location("hp004_joint_stabilizer", _HP004_MOD_PATH)
    assert spec is not None and spec.loader is not None
    hp004 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hp004)

    rows: List[HQCWorkloadRow] = []

    # WP1a: 8Q k=2 HP-004 replication on H2/Helios (3 randomized blocks x 3 core arms x 9 bases = 81 circuits x 300 shots)
    # Power note: 900 shots/basis x 9 bases = 8,100 shots/arm -> SE_shot ~ 0.0055 (>39 sigma above 0.50 and >25 sigma above 0.6839 at 99.85% 2Q fidelity)
    wp1a_stats = []
    for arm in ("Arm A", "Arm B", "Arm B-Ctrl"):
        for p, q in [("X", "X"), ("Y", "Z"), ("Z", "Z")]:
            qc = hp004.build_hp004_circuit(arm, p, q)
            wp1a_stats.append(compute_hqc_for_circuit(qc, shots=300))
    mean_1q_a = float(np.mean([s["n_1q"] for s in wp1a_stats]))
    mean_2q_a = float(np.mean([s["n_2q"] for s in wp1a_stats]))
    mean_nm_a = float(np.mean([s["n_m"] for s in wp1a_stats]))
    mean_hqc_a = float(np.mean([s["hqc"] for s in wp1a_stats]))
    rows.append(
        HQCWorkloadRow(
            work_package="WP1a",
            protocol="8Q k=2 9-basis joint stabilizer replication (zero SWAP)",
            target_system="Quantinuum H2-1 / Helios-1",
            physical_qubits=8,
            circuits=81,
            shots_per_circuit=300,
            total_shots=81 * 300,
            mean_n_1q=round(mean_1q_a, 2),
            mean_n_2q=round(mean_2q_a, 2),
            mean_n_m=round(mean_nm_a, 2),
            routing_swaps=0,
            hqc_per_circuit=round(mean_hqc_a, 3),
            total_hqc=round(81 * mean_hqc_a, 1),
        )
    )

    # WP1b: 12Q k=3 (d=8 Schmidt-rank) all-to-all extension (54 circuits x 300 shots)
    wp1b_stat = compute_hqc_for_circuit(_build_wp1_k3_all_to_all_sample(), shots=300)
    rows.append(
        HQCWorkloadRow(
            work_package="WP1b",
            protocol="12Q k=3 all-to-all Schmidt-rank extension",
            target_system="Quantinuum H2-1 / Helios-1",
            physical_qubits=12,
            circuits=54,
            shots_per_circuit=300,
            total_shots=54 * 300,
            mean_n_1q=round(wp1b_stat["n_1q"], 2),
            mean_n_2q=round(wp1b_stat["n_2q"], 2),
            mean_n_m=round(wp1b_stat["n_m"], 2),
            routing_swaps=0,
            hqc_per_circuit=round(wp1b_stat["hqc"], 3),
            total_hqc=round(54 * wp1b_stat["hqc"], 1),
        )
    )

    # WP2a (50Q on H2/Helios, 12 circuits x 512 shots) & WP2b (96Q on Helios, 12 circuits x 512 shots)
    for tag, nq, n_circ, sys_name in [
        ("WP2a", 50, 12, "Quantinuum H2-1 / Helios-1"),
        ("WP2b", 96, 12, "Quantinuum Helios-1 (98Q QCCD)"),
    ]:
        st = compute_hqc_for_circuit(_build_wp2_transport_circuit(nq, depth_layers=4), shots=512)
        rows.append(
            HQCWorkloadRow(
                work_package=tag,
                protocol=f"{nq}Q 4-layer nonlocal debiased transport stability",
                target_system=sys_name,
                physical_qubits=nq,
                circuits=n_circ,
                shots_per_circuit=512,
                total_shots=n_circ * 512,
                mean_n_1q=round(st["n_1q"], 2),
                mean_n_2q=round(st["n_2q"], 2),
                mean_n_m=round(st["n_m"], 2),
                routing_swaps=0,
                hqc_per_circuit=round(st["hqc"], 3),
                total_hqc=round(n_circ * st["hqc"], 1),
            )
        )

    # WP3a: Steane [[7,1,3]] + Tesseract [[16,6,4]] multi-round (R=3) syndrome extraction (24 circuits x 512 shots)
    hx_s, hz_s = build_steane_7_1_3()
    hx_t, hz_t = build_tesseract_16_6_4()
    st_steane = compute_hqc_for_circuit(_build_wp3_syndrome_circuit(hx_s, hz_s, rounds=3, n_ancillas=3), shots=512)
    st_tess = compute_hqc_for_circuit(_build_wp3_syndrome_circuit(hx_t, hz_t, rounds=3, n_ancillas=5), shots=512)
    rows.append(
        HQCWorkloadRow(
            work_package="WP3a",
            protocol="Steane [[7,1,3]] & Tesseract [[16,6,4]] R=3 mid-circuit QEC",
            target_system="Quantinuum H2-1 / Helios-1",
            physical_qubits=21,
            circuits=24,
            shots_per_circuit=512,
            total_shots=24 * 512,
            mean_n_1q=round(0.5 * (st_steane["n_1q"] + st_tess["n_1q"]), 2),
            mean_n_2q=round(0.5 * (st_steane["n_2q"] + st_tess["n_2q"]), 2),
            mean_n_m=round(0.5 * (st_steane["n_m"] + st_tess["n_m"]), 2),
            routing_swaps=0,
            hqc_per_circuit=round(0.5 * (st_steane["hqc"] + st_tess["hqc"]), 3),
            total_hqc=round(12 * st_steane["hqc"] + 12 * st_tess["hqc"], 1),
        )
    )

    # WP3b: Bivariate-Bicycle [[72,12,6]] qLDPC R=2 syndrome extraction on Helios (8 circuits x 256 shots, 96Q)
    hx_bb, hz_bb = build_bivariate_bicycle_72_12_6()
    st_bb = compute_hqc_for_circuit(_build_wp3_syndrome_circuit(hx_bb, hz_bb, rounds=2, n_ancillas=24), shots=256)
    rows.append(
        HQCWorkloadRow(
            work_package="WP3b",
            protocol="Bivariate-Bicycle [[72,12,6]] qLDPC R=2 (96Q Helios ancilla reuse)",
            target_system="Quantinuum Helios-1 (98Q QCCD)",
            physical_qubits=96,
            circuits=8,
            shots_per_circuit=256,
            total_shots=8 * 256,
            mean_n_1q=round(st_bb["n_1q"], 2),
            mean_n_2q=round(st_bb["n_2q"], 2),
            mean_n_m=round(st_bb["n_m"], 2),
            routing_swaps=0,
            hqc_per_circuit=round(st_bb["hqc"], 3),
            total_hqc=round(8 * st_bb["hqc"], 1),
        )
    )

    # Decoder verification benchmarks
    decoder_benchmarks: Dict[str, Any] = {}
    for code_name, (hx, hz), check_w2 in [
        ("Steane_7_1_3", (hx_s, hz_s), False),
        ("Tesseract_16_6_4", (hx_t, hz_t), False),
        ("BivariateBicycle_72_12_6", (hx_bb, hz_bb), True),
    ]:
        lx, _ = css_logical_operators(hx, hz)
        n_data = hx.shape[1]
        k_log = lx.shape[0]

        w1_pass = 0
        for q in range(n_data):
            err = np.zeros(n_data, dtype=np.uint8)
            err[q] = 1
            syn = (hz @ err) % 2
            res = decode_bp_osd(hz, syn, channel_probs=0.015)
            residual = err ^ res.correction
            if np.all((hz @ residual) % 2 == 0) and np.all((lx @ residual) % 2 == 0):
                w1_pass += 1

        w2_summary = None
        if check_w2:
            # Exhaustive weight-2 check across all 72*71/2 = 2,556 pairs on BB [[72, 12, 6]] (d=6 corrects all wt <= 2)
            w2_pass = 0
            w2_total = 0
            for q1 in range(n_data):
                for q2 in range(q1 + 1, n_data):
                    w2_total += 1
                    err = np.zeros(n_data, dtype=np.uint8)
                    err[q1] = 1
                    err[q2] = 1
                    syn = (hz @ err) % 2
                    res = decode_bp_osd(hz, syn, channel_probs=0.015)
                    residual = err ^ res.correction
                    if np.all((hz @ residual) % 2 == 0) and np.all((lx @ residual) % 2 == 0):
                        w2_pass += 1
            w2_summary = f"{w2_pass}/{w2_total}"

        rng = np.random.default_rng(20261116)
        n_mc = 500
        bp_only_fail = 0
        bposd_logical_err = 0
        osd_invocations = 0
        for _ in range(n_mc):
            err = (rng.random(n_data) < 0.02).astype(np.uint8)
            syn = (hz @ err) % 2
            res = decode_bp_osd(hz, syn, channel_probs=0.02)
            if res.used_osd:
                osd_invocations += 1
            if not res.bp_converged:
                bp_only_fail += 1
            residual = err ^ res.correction
            if np.any((hz @ residual) % 2 != 0) or np.any((lx @ residual) % 2 != 0):
                bposd_logical_err += 1
                if res.bp_converged:
                    bp_only_fail += 1

        entry: Dict[str, Any] = {
            "n": int(n_data),
            "k": int(k_log),
            "rank_hx": int(gf2_rank(hx)),
            "rank_hz": int(gf2_rank(hz)),
            "max_check_weight": int(hx.sum(axis=1).max()),
            "max_qubit_hyperedge_degree": int(hz.sum(axis=0).max()),
            "exhaustive_weight1_recovery": f"{w1_pass}/{n_data}",
            "mc_500_p002_pure_bp_failure_rate": round(bp_only_fail / n_mc, 4),
            "mc_500_p002_bposd_logical_error_rate": round(bposd_logical_err / n_mc, 4),
            "mc_500_osd_invocations": int(osd_invocations),
        }
        if w2_summary is not None:
            entry["exhaustive_weight2_recovery"] = w2_summary
        decoder_benchmarks[code_name] = entry

    total_hqc = round(sum(r.total_hqc for r in rows), 1)
    total_circuits = sum(r.circuits for r in rows)
    total_shots = sum(r.total_shots for r in rows)

    summary = {
        "hqc_formula": "HQC = 5 + C * (N_1q + 10 * N_2q + 5 * N_m) / 5000 (rz virtual = 0)",
        "total_quantinuum_circuits": total_circuits,
        "total_quantinuum_shots": total_shots,
        "total_estimated_hqc": total_hqc,
        "recommended_initial_hqc_allocation": int(math.ceil(total_hqc / 500.0) * 500),
        "recommended_ibm_qpu_seconds": 600,
        "workloads": [asdict(r) for r in rows],
        "decoder_benchmarks": decoder_benchmarks,
    }
    return summary


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    summary = evaluate_qcup_workloads_and_decoders()
    out_path = out_dir / "qcup_workload_and_bposd_estimates.json"
    payload = json.dumps(summary, indent=2) + "\n"
    out_path.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"Wrote {out_path} (sha256: {digest})")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
