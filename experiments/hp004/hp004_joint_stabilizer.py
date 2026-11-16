"""
OSIRIS-HP-004: Tree-Native Light-Cone-Isolated 9-Basis 4-Qubit Joint Bell Stabilizer Harness.

Designed for 8-transmon heavy-hex trees (`w0--w3, w2--w3, w3--w4--w5--w6--w7, w1--w5`, e.g.
`[34, 52, 32, 33, 39, 53, 54, 55]` on `ibm_marrakesh`) with STRICT physical light-cone isolation:
  - Stage 1 (6 CZs): Prepares Bell pairs `(R2=w0, M1=w4)`, `(R1=w1, M2=w5)`, `(B1=w3, E1=w2)`,
    and `(B2=w6, E2=w7)` (last two omitted in `Arm B-Ctrl`).
  - Barrier 1: `R2=w0` and `R1=w1` receive ZERO 1Q or 2Q gates for the remainder of the circuit.
  - Stage 2 (16 CZs for U1, 19 CZs for Held-Out U2): Applies irreducible 4-chain dual-unitary
    Clifford scrambler (`seed=111`, 7 CZs, `I(R:D)=I(R:C)=I(R:E)=0`, `I(R:DE)=4` bits,
    `I(R2:D1,E1)=0` bits) on `[w3, w4, w5, w6]`, followed by local radiation emission swaps placing
    remnants at degree-1 leaves `C1=w2`, `C2=w7` and radiation on `[w3=E1, w4=D1, w5=E2, w6=D2]`.
  - Barrier 2: Horizon boundary.
  - Stage 3 (Decoding with zero router intervention):
      * `Arm A` (16 CZs, total 38 CZs): Exact barriered inverse of Stage 2 on `[w2..w7]`.
      * `Arm B` / `Arm B-Ctrl` / `Arm B-Held` (10 CZs, total 32 / 30 / 35 CZs): Exact symplectic
        Clifford decoder acting STRICTLY on `[w3, w4, w5, w6]` — ZERO gates touch `C1=w2`, `C2=w7`,
        `R2=w0`, or `R1=w1`.
      * `Arm B-Def` (3 CZs, total 25 CZs): Optimal 1-photon sub-capacity Clifford decoder acting
        STRICTLY on `[w3, w4, w5]` — ZERO gates touch quarantined photon `D2=w6`, `C1=w2`, `C2=w7`,
        `R2=w0`, or `R1=w1`.
  - Stage 4: Independent `{X, Y, Z}_1 x {X, Y, Z}_2` basis rotations and unconditional 4-bit readout
    on the IDENTICAL 4 physical transmons `[w0(R2), w4(M1'), w1(R1), w5(M2')]` across all 5 arms.

Strictly offline module: contains NO QPU job submission calls.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Clifford

ARMS: Tuple[str, ...] = ("Arm A", "Arm B", "Arm B-Def", "Arm B-Ctrl", "Arm B-Held")
PAULIS: Tuple[str, ...] = ("X", "Y", "Z")
JOINT_BASES: Tuple[Tuple[str, str], ...] = tuple((p, q) for p in PAULIS for q in PAULIS)
BELL_SIGN: Mapping[str, int] = {"I": 1, "X": 1, "Y": -1, "Z": 1}

# Canonical 8-transmon heavy-hex tree edges on wire indices 0..7:
#   0(R2)--3, 2(C1)--3(E1)--4(D1)--5(E2)--6(D2)--7(C2), and 1(R1)--5(E2)
TREE_EDGES_8Q: Tuple[Tuple[int, int], ...] = (
    (0, 3), (3, 0),
    (2, 3), (3, 2),
    (3, 4), (4, 3),
    (4, 5), (5, 4),
    (1, 5), (5, 1),
    (5, 6), (6, 5),
    (6, 7), (7, 6),
)
CHAIN4_EDGES: Tuple[Tuple[int, int], ...] = ((0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2))
CHAIN3_EDGES: Tuple[Tuple[int, int], ...] = ((0, 1), (1, 0), (1, 2), (2, 1))

# Default lowest-error 8-transmon heavy-hex tree on `ibm_marrakesh` (mean CZ error ~0.00185):
#   w0=34(R2), w1=52(R1), w2=32(C1), w3=33(E1), w4=39(D1/M1'), w5=53(E2/M2'), w6=54(D2), w7=55(C2)
DEFAULT_MARRAKESH_TREE_LAYOUT: Tuple[int, ...] = (34, 52, 32, 33, 39, 53, 54, 55)


def _make_brickwall_4chain(seed: int, n_layers: int = 4) -> QuantumCircuit:
    r = np.random.default_rng(seed)
    qc = QuantumCircuit(4, name=f"scram_bw_{seed}")
    for layer in range(n_layers):
        for q in range(4):
            g = str(r.choice(["id", "h", "s", "hs", "sh"]))
            if g.startswith("h"):
                qc.h(q)
            if "s" in g:
                qc.s(q)
            if g == "sh":
                qc.h(q)
        edges = [(0, 1), (2, 3)] if layer % 2 == 0 else [(1, 2)]
        for a, b in edges:
            gt = str(r.choice(["cz", "cx", "xc", "swap_like"]))
            if gt == "cz":
                qc.cz(a, b)
            elif gt == "cx":
                qc.cx(a, b)
            elif gt == "xc":
                qc.cx(b, a)
            else:
                qc.cx(a, b)
                qc.cx(b, a)
    return qc


def _strip_to_local_isa(t_qc: QuantumCircuit, n_wires: int) -> QuantumCircuit:
    bare = QuantumCircuit(n_wires)
    for inst in t_qc.data:
        bare.append(inst.operation, [t_qc.find_bit(q).index for q in inst.qubits])
    return bare


# Scrambler U1 (seed=111, 7 CZs) and Held-Out Scrambler U2 (seed=28, 10 CZs)
SCRAMBLER_U1_RAW = _make_brickwall_4chain(seed=111, n_layers=4)
SCRAMBLER_U2_RAW = _make_brickwall_4chain(seed=28, n_layers=4)

# Exact 10-CZ symplectic 2-photon decoder on [w3=E1, w4=D1, w5=E2, w6=D2] (outputs M1'->w4, M2'->w5)
DECODER_2P_TABLE = {
    "destabilizer": ["ZYXI", "-IXYX", "YYZI", "IXYI"],
    "stabilizer": ["YXYX", "XIYZ", "IYXX", "YXXZ"],
}
_DEC_2P_T = transpile(
    Clifford.from_dict(DECODER_2P_TABLE).adjoint().to_circuit(),
    coupling_map=[[a, b] for a, b in CHAIN4_EDGES],
    basis_gates=["cz", "sx", "rz", "x"],
    initial_layout=[0, 1, 2, 3],
    optimization_level=3,
    seed_transpiler=0,
)
assert list(_DEC_2P_T.layout.final_index_layout())[1:3] == [1, 2]
DECODER_2P_ISA = _strip_to_local_isa(_DEC_2P_T, 4)

# Exact 3-CZ optimal 1-photon sub-capacity decoder on [w3=E1, w4=D1, w5=E2] (outputs M1'->w4, leaves w6=D2 untouched)
DECODER_1P_TABLE = {
    "stabilizer": ["+IZY", "-XYX", "+ZYZ"],
    "destabilizer": ["-IXY", "+IYZ", "+XII"],
}
_DEC_1P_T = transpile(
    Clifford.from_dict(DECODER_1P_TABLE).to_circuit(),
    coupling_map=[[a, b] for a, b in CHAIN3_EDGES],
    basis_gates=["cz", "sx", "rz", "x"],
    initial_layout=[0, 1, 2],
    optimization_level=3,
    seed_transpiler=42,
)
assert list(_DEC_1P_T.layout.final_index_layout())[1] == 1
DECODER_1P_ISA = _strip_to_local_isa(_DEC_1P_T, 3)


def _apply_pair_basis(qc: QuantumCircuit, q_ref: int, q_msg: int, pauli: str) -> None:
    if pauli == "X":
        qc.h(q_ref)
        qc.h(q_msg)
    elif pauli == "Y":
        qc.sdg(q_ref)
        qc.h(q_ref)
        qc.sdg(q_msg)
        qc.h(q_msg)
    elif pauli == "Z":
        return
    else:
        raise ValueError(f"Unsupported Pauli basis: {pauli!r}")


def build_hp004_circuit(arm_name: str, basis_pair1: str, basis_pair2: str) -> QuantumCircuit:
    """Build one tree-native, light-cone-isolated 8-qubit ISA circuit on canonical wires `0..7`."""
    if arm_name not in ARMS:
        raise ValueError(f"Unknown arm: {arm_name!r}")
    if basis_pair1 not in PAULIS or basis_pair2 not in PAULIS:
        raise ValueError(f"Invalid basis pair: {(basis_pair1, basis_pair2)!r}")

    qc = QuantumCircuit(8, 4, name=f"{arm_name}_{basis_pair1}{basis_pair2}")

    # Stage 1: Exact 6-CZ State Preparation on tree edges (4 CZs in Arm B-Ctrl)
    prep = QuantumCircuit(8)
    prep.h(0); prep.cx(0, 3); prep.cx(3, 4); prep.cx(4, 3)  # |Phi+> on (w0=R2, w4=M1), w3=|0>
    prep.h(1); prep.cx(1, 5)                                # |Phi+> on (w1=R1, w5=M2)
    if arm_name != "Arm B-Ctrl":
        prep.h(3); prep.cx(3, 2)                            # |Phi+> on (w3=B1, w2=E1)
        prep.h(6); prep.cx(6, 7)                            # |Phi+> on (w6=B2, w7=E2)
    prep_isa = _strip_to_local_isa(
        transpile(
            prep,
            coupling_map=[[a, b] for a, b in TREE_EDGES_8Q],
            basis_gates=["cz", "sx", "rz", "x"],
            initial_layout=list(range(8)),
            routing_method="none",
            optimization_level=3,
        ),
        8,
    )
    qc.compose(prep_isa, range(8), inplace=True)

    # Barrier 1: Reference qubits w0(R2) and w1(R1) are permanently isolated from here until readout
    qc.barrier(range(8))

    # Stage 2: Scrambling on [w3, w4, w5, w6] + local Hawking emission swaps to leaves w2(C1), w7(C2)
    scram = QuantumCircuit(6)  # maps to [w2, w3, w4, w5, w6, w7]
    u_raw = SCRAMBLER_U2_RAW if arm_name == "Arm B-Held" else SCRAMBLER_U1_RAW
    scram.compose(u_raw, [1, 2, 3, 4], inplace=True)
    scram.swap(0, 1)  # w2=C1, w3=E1
    scram.swap(4, 5)  # w6=E2, w7=C2
    scram.swap(3, 4)  # w5=E2, w6=D2
    chain6_edges = [[i, i + 1] for i in range(5)] + [[i + 1, i] for i in range(5)]
    scram_isa = _strip_to_local_isa(
        transpile(
            scram,
            coupling_map=chain6_edges,
            basis_gates=["cz", "sx", "rz", "x"],
            initial_layout=list(range(6)),
            routing_method="none",
            optimization_level=3,
        ),
        6,
    )
    qc.compose(scram_isa, [2, 3, 4, 5, 6, 7], inplace=True)

    # Barrier 2: Horizon boundary (prevents U * U^dagger collapse in Arm A and demarcates decoding)
    qc.barrier(range(8))

    # Stage 3: Strict physical light-cone-isolated decoding
    if arm_name == "Arm A":
        inv_isa = _strip_to_local_isa(
            transpile(
                scram.inverse(),
                coupling_map=chain6_edges,
                basis_gates=["cz", "sx", "rz", "x"],
                initial_layout=list(range(6)),
                routing_method="none",
                optimization_level=3,
            ),
            6,
        )
        qc.compose(inv_isa, [2, 3, 4, 5, 6, 7], inplace=True)
    elif arm_name in ("Arm B", "Arm B-Ctrl", "Arm B-Held"):
        # Strictly inside [w3, w4, w5, w6]; ZERO gates on remnants w2(C1), w7(C2) or refs w0, w1
        qc.compose(DECODER_2P_ISA, [3, 4, 5, 6], inplace=True)
    elif arm_name == "Arm B-Def":
        # Strictly inside [w3, w4, w5]; ZERO gates on quarantined D2=w6, remnants w2, w7, or refs w0, w1
        qc.compose(DECODER_1P_ISA, [3, 4, 5], inplace=True)

    qc.barrier([0, 4, 1, 5])

    # Stage 4: Tomographic basis rotations on Pair 1 (w0=R2, w4=M1') and Pair 2 (w1=R1, w5=M2')
    rot = QuantumCircuit(4)
    _apply_pair_basis(rot, 0, 1, basis_pair1)
    _apply_pair_basis(rot, 2, 3, basis_pair2)
    rot_isa = _strip_to_local_isa(
        transpile(rot, basis_gates=["cz", "sx", "rz", "x"], optimization_level=3),
        4,
    )
    qc.compose(rot_isa, [0, 4, 1, 5], inplace=True)

    # Stage 5: Identical 4-transmon readout across all 5 arms: c0=w0(R2), c1=w4(M1'), c2=w1(R1), c3=w5(M2')
    qc.measure([0, 4, 1, 5], [0, 1, 2, 3])
    return qc


def build_spam_calibration_circuits() -> List[Tuple[QuantumCircuit, str]]:
    """Build all 16 computational-basis state preparation circuits on `[w0, w4, w1, w5]`."""
    cal: List[Tuple[QuantumCircuit, str]] = []
    readout_qubits = [0, 4, 1, 5]  # maps to c0, c1, c2, c3
    for idx in range(16):
        bitstr = format(idx, "04b")  # 'c3c2c1c0'
        qc = QuantumCircuit(8, 4, name=f"spam_cal_{bitstr}")
        for clbit_idx, q in enumerate(readout_qubits):
            if bitstr[3 - clbit_idx] == "1":
                qc.x(q)
        qc.barrier(readout_qubits)
        qc.measure(readout_qubits, [0, 1, 2, 3])
        cal.append((qc, bitstr))
    return cal


def assemble_hp004_schedule(
    n_blocks: int = 6,
    seed: int = 20261116,
    arms: Sequence[str] = ARMS,
) -> Tuple[List[QuantumCircuit], List[Dict[str, object]]]:
    """Assemble block-randomized 9-basis circuits plus 16-state readout calibration."""
    if n_blocks < 2:
        raise ValueError("n_blocks must be >= 2 to compute unbiased block SEM (ddof=1)")
    circuits: List[QuantumCircuit] = []
    metadata: List[Dict[str, object]] = []

    for qc, bitstr in build_spam_calibration_circuits():
        circuits.append(qc)
        metadata.append({"kind": "spam_cal", "prepared_bits": bitstr, "circuit_name": qc.name})

    for block_idx in range(n_blocks):
        block_items: List[Tuple[QuantumCircuit, Dict[str, object]]] = []
        for arm in arms:
            for b1, b2 in JOINT_BASES:
                cname = f"hp004_b{block_idx}_{arm.lower().replace(' ', '_').replace('-', '_')}_{b1.lower()}{b2.lower()}"
                qc = build_hp004_circuit(arm, b1, b2)
                qc.name = cname
                block_items.append(
                    (
                        qc,
                        {
                            "kind": "tomography",
                            "block_index": block_idx,
                            "arm": arm,
                            "basis_pair1": b1,
                            "basis_pair2": b2,
                            "basis": f"{b1}{b2}",
                            "circuit_name": cname,
                        },
                    )
                )
        rng = random.Random(seed + block_idx * 1009)
        rng.shuffle(block_items)
        for qc, meta in block_items:
            circuits.append(qc)
            metadata.append(meta)

    return circuits, metadata


def compile_layout_locked(
    circuits: Sequence[QuantumCircuit],
    initial_layout: Sequence[int] = DEFAULT_MARRAKESH_TREE_LAYOUT,
    coupling_map: Sequence[Sequence[int]] | None = None,
    num_chip_qubits: int | None = None,
) -> List[QuantumCircuit]:
    """Embed tree-native circuits onto physical `initial_layout` with `routing_method='none'` (zero SWAP drift)."""
    if len(initial_layout) != 8 or len(set(initial_layout)) != 8:
        raise ValueError("initial_layout must contain 8 distinct physical qubit indices")
    if coupling_map is None:
        n_chip = num_chip_qubits or (max(initial_layout) + 1)
        cmap = [[initial_layout[a], initial_layout[b]] for a, b in TREE_EDGES_8Q]
    else:
        cmap = [list(e) for e in coupling_map]
        n_chip = num_chip_qubits or (max(max(e) for e in cmap) + 1)
        edge_set = {(e[0], e[1]) for e in cmap} | {(e[1], e[0]) for e in cmap}
        for a, b in TREE_EDGES_8Q:
            pa, pb = initial_layout[a], initial_layout[b]
            if (pa, pb) not in edge_set:
                raise ValueError(f"Physical layout missing required tree edge ({pa}, {pb}) for logical ({a}, {b})")

    compiled = transpile(
        list(circuits),
        coupling_map=cmap,
        basis_gates=["cz", "sx", "rz", "x"],
        initial_layout=list(initial_layout),
        routing_method="none",
        optimization_level=0,
    )
    expected = tuple(initial_layout)
    expected_readout = (initial_layout[0], initial_layout[4], initial_layout[1], initial_layout[5])
    for qc in compiled:
        init_phys = tuple(qc.layout.initial_index_layout()[:8])
        if init_phys != expected:
            raise RuntimeError(f"Initial layout drifted on {qc.name}: {init_phys} != {expected}")
        meas_phys = tuple(
            qc.find_bit(inst.qubits[0]).index
            for inst in sorted(
                (i for i in qc.data if i.operation.name == "measure"),
                key=lambda i: qc.find_bit(i.clbits[0]).index,
            )
        )
        if meas_phys != expected_readout:
            raise RuntimeError(f"Readout transmons drifted on {qc.name}: {meas_phys} != {expected_readout}")
    return compiled


def pair_parities_from_counts(counts: Mapping[str, int]) -> Tuple[float, float, float, int]:
    """Return `(mean_pi1, mean_pi2, mean_pi1_pi2, total_shots)` from raw 4-bit Qiskit counts `'c3c2c1c0'`."""
    total = 0
    sum_p1 = 0.0
    sum_p2 = 0.0
    sum_p12 = 0.0
    for raw_bits, c in counts.items():
        bits = raw_bits.replace(" ", "")
        if len(bits) != 4 or any(ch not in "01" for ch in bits):
            raise ValueError(f"Expected 4-bit readout string, got {raw_bits!r}")
        n = int(c)
        if n < 0:
            raise ValueError("Negative shot count")
        p1 = 1.0 if bits[3] == bits[2] else -1.0
        p2 = 1.0 if bits[1] == bits[0] else -1.0
        sum_p1 += n * p1
        sum_p2 += n * p2
        sum_p12 += n * (p1 * p2)
        total += n
    if total <= 0:
        raise ValueError("Empty counts dictionary")
    return sum_p1 / total, sum_p2 / total, sum_p12 / total, total


def estimate_joint_bell_fidelity(
    basis_counts: Mapping[str, Mapping[str, int]],
) -> Dict[str, float]:
    """Compute exact 16-stabilizer 4-qubit joint Bell overlap `F_joint` and marginal fidelities from 9 bases."""
    missing = [f"{p}{q}" for p, q in JOINT_BASES if f"{p}{q}" not in basis_counts]
    if missing:
        raise ValueError(f"Missing required 9-basis settings: {missing}")

    p1_by_setting: Dict[str, Tuple[float, int]] = {}
    p2_by_setting: Dict[str, Tuple[float, int]] = {}
    p12_by_setting: Dict[str, Tuple[float, int]] = {}

    for p, q in JOINT_BASES:
        key = f"{p}{q}"
        m1, m2, m12, shots = pair_parities_from_counts(basis_counts[key])
        p1_by_setting[key] = (m1, shots)
        p2_by_setting[key] = (m2, shots)
        p12_by_setting[key] = (m12, shots)

    exp_p1: Dict[str, float] = {}
    exp_p2: Dict[str, float] = {}
    var_p1: Dict[str, float] = {}
    var_p2: Dict[str, float] = {}

    for p in PAULIS:
        shots_p = sum(p1_by_setting[f"{p}{q}"][1] for q in PAULIS)
        mean_p = sum(p1_by_setting[f"{p}{q}"][0] * p1_by_setting[f"{p}{q}"][1] for q in PAULIS) / shots_p
        exp_p1[p] = mean_p
        var_p1[p] = max(0.0, 1.0 - mean_p * mean_p) / shots_p

    for q in PAULIS:
        shots_q = sum(p2_by_setting[f"{p}{q}"][1] for p in PAULIS)
        mean_q = sum(p2_by_setting[f"{p}{q}"][0] * p2_by_setting[f"{p}{q}"][1] for p in PAULIS) / shots_q
        exp_p2[q] = mean_q
        var_p2[q] = max(0.0, 1.0 - mean_q * mean_q) / shots_q

    f1 = 0.25 * (1.0 + sum(BELL_SIGN[p] * exp_p1[p] for p in PAULIS))
    f2 = 0.25 * (1.0 + sum(BELL_SIGN[q] * exp_p2[q] for q in PAULIS))

    joint_corr_sum = 0.0
    var_f_joint = 0.0

    for p, q in JOINT_BASES:
        key = f"{p}{q}"
        sp = BELL_SIGN[p]
        sq = BELL_SIGN[q]
        joint_corr_sum += sp * sq * p12_by_setting[key][0]

        counts = basis_counts[key]
        n_shots = p12_by_setting[key][1]
        vals = []
        probs = []
        for raw_bits, c in counts.items():
            bits = raw_bits.replace(" ", "")
            pi1 = 1.0 if bits[3] == bits[2] else -1.0
            pi2 = 1.0 if bits[1] == bits[0] else -1.0
            score = ((sp * pi1) / 3.0 + (sq * pi2) / 3.0 + (sp * sq * pi1 * pi2)) / 16.0
            vals.append(score)
            probs.append(int(c) / n_shots)
        mean_score = sum(v * pr for v, pr in zip(vals, probs))
        setting_var = sum(pr * ((v - mean_score) ** 2) for v, pr in zip(vals, probs)) / n_shots
        var_f_joint += setting_var

    f_joint = (
        1.0
        + sum(BELL_SIGN[p] * exp_p1[p] for p in PAULIS)
        + sum(BELL_SIGN[q] * exp_p2[q] for q in PAULIS)
        + joint_corr_sum
    ) / 16.0

    se_f1 = 0.25 * math.sqrt(sum(var_p1.values()))
    se_f2 = 0.25 * math.sqrt(sum(var_p2.values()))
    se_joint = math.sqrt(var_f_joint)

    return {
        "f1_marginal": f1,
        "f1_shot_se": se_f1,
        "f2_marginal": f2,
        "f2_shot_se": se_f2,
        "f_marginal_mean": 0.5 * (f1 + f2),
        "f_marginal_product": f1 * f2,
        "f_union_lower_bound": f1 + f2 - 1.0,
        "f_joint": f_joint,
        "f_joint_shot_se": se_joint,
        "z_above_1ebit_shot": (f_joint - 0.5) / se_joint if se_joint > 0 else float("inf"),
    }


def summarize_blocks(block_basis_counts: Sequence[Mapping[str, Mapping[str, int]]]) -> Dict[str, float]:
    """Aggregate across `B >= 2` blocks with conservative `SE_eff = max(SE_block_ddof1, SE_shot)`."""
    if len(block_basis_counts) < 2:
        raise ValueError("At least 2 blocks required for unbiased ddof=1 SEM")

    pooled: Dict[str, Dict[str, int]] = {f"{p}{q}": {} for p, q in JOINT_BASES}
    block_f_joints: List[float] = []
    for b_counts in block_basis_counts:
        est = estimate_joint_bell_fidelity(b_counts)
        block_f_joints.append(est["f_joint"])
        for key, hist in b_counts.items():
            for bits, n in hist.items():
                clean = bits.replace(" ", "")
                pooled[key][clean] = pooled[key].get(clean, 0) + int(n)

    pooled_est = estimate_joint_bell_fidelity(pooled)
    se_block_ddof1 = float(np.std(block_f_joints, ddof=1) / math.sqrt(len(block_f_joints)))
    se_eff = max(se_block_ddof1, pooled_est["f_joint_shot_se"])
    return {
        **pooled_est,
        "n_blocks": float(len(block_basis_counts)),
        "f_joint_block_sem_ddof1": se_block_ddof1,
        "f_joint_se_eff": se_eff,
        "z_above_1ebit_eff": (pooled_est["f_joint"] - 0.5) / se_eff if se_eff > 0 else float("inf"),
    }
