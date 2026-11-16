"""
OSIRIS-HP-004: Layout-Locked 9-Basis 4-Qubit Joint Bell Stabilizer Harness.

Corrects the estimator and protocol limitations identified in the HP-002/HP-003 forensic audit:
  1. Measures all 9 two-pair Pauli basis settings `{X, Y, Z}_1 x {X, Y, Z}_2` so all 16 four-qubit
     Bell stabilizers of `|Phi+>_{R1 M1'} (x) |Phi+>_{R2 M2'}` are estimated directly without
     marginal factorization assumptions.
  2. Certifies `> 1 ebit` (Schmidt number >= 3 across `(R1 R2) : (M1' M2')`) via the exact tight
     bound `F_joint > 0.5000` (whereas 3-basis marginal mean `0.7500` and product `0.5625` can be
     saturated by a 1-ebit Schmidt-rank-2 mixture).
  3. Enforces a single pinned `initial_layout` across all arms plus 16 computational-basis SPAM
     calibration circuits on the exact 4 readout transmons `[layout[0], layout[2], layout[1], layout[3]]`.
  4. Preserves full raw 4-bit histograms per circuit, block-shuffles every block, and reports
     conservative uncertainty `SE_eff = max(SE_block_ddof1, SE_shot)`.

Strictly offline module: contains NO QPU job submission calls.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Clifford

# Reuse the audited 4-qubit scrambler generator and exact Clifford decoders from HP-003.
import sys
from pathlib import Path

_HP003_DIR = Path(__file__).resolve().parent.parent / "hp003"
if str(_HP003_DIR) not in sys.path:
    sys.path.insert(0, str(_HP003_DIR))

from find_hp003_scrambler_decoder import build_candidate_scrambler  # noqa: E402

ARMS: Tuple[str, ...] = ("Arm A", "Arm B", "Arm B-Def", "Arm B-Ctrl", "Arm B-Held")
PAULIS: Tuple[str, ...] = ("X", "Y", "Z")
JOINT_BASES: Tuple[Tuple[str, str], ...] = tuple((p, q) for p in PAULIS for q in PAULIS)
BELL_SIGN: Mapping[str, int] = {"I": 1, "X": 1, "Y": -1, "Z": 1}

SCRAMBLER_U1 = build_candidate_scrambler(seed=128)
SCRAMBLER_U2 = build_candidate_scrambler(seed=158)

DECODER_TABLE = {
    "destabilizer": ["-IYIX", "-IXIZ", "+IXYZ", "+XXIZ"],
    "stabilizer": ["-XXYI", "+YIXX", "+IYXX", "+YYIX"],
}
DECODER_CIRC = Clifford.from_dict(DECODER_TABLE).adjoint().to_circuit().decompose()

SINGLE_PHOTON_DECODER_TABLE = {
    "stabilizer": ["-IXY", "-XYZ", "+ZXI"],
    "destabilizer": ["-YZI", "-ZXY", "+IZX"],
}
SINGLE_PHOTON_DECODER_CIRC = Clifford.from_dict(SINGLE_PHOTON_DECODER_TABLE).to_circuit().decompose()


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
    """Build one 8-qubit, 4-clbit circuit in `{X,Y,Z}_1 x {X,Y,Z}_2` readout basis."""
    if arm_name not in ARMS:
        raise ValueError(f"Unknown arm: {arm_name!r}")
    if basis_pair1 not in PAULIS or basis_pair2 not in PAULIS:
        raise ValueError(f"Invalid basis pair: {(basis_pair1, basis_pair2)!r}")

    qc = QuantumCircuit(8, 4, name=f"{arm_name}_{basis_pair1}{basis_pair2}")

    # 1. Reference-Message Bell pairs on (R1=0, M1=2) and (R2=1, M2=3)
    qc.h(0)
    qc.cx(0, 2)
    qc.h(1)
    qc.cx(1, 3)

    if arm_name != "Arm B-Ctrl":
        # Early Hawking-partner entanglement on (B1=4, E1=6) and (B2=5, E2=7)
        qc.h(4)
        qc.cx(4, 6)
        qc.h(5)
        qc.cx(5, 7)

    # Barrier isolates initial state preparation across all arms
    qc.barrier(range(8))

    # 2. Scrambling unitary U_MB on (M1=2, M2=3, B1=4, B2=5)
    scrambler = SCRAMBLER_U2 if arm_name == "Arm B-Held" else SCRAMBLER_U1
    qc.compose(scrambler.decompose(), qubits=[2, 3, 4, 5], inplace=True)

    # Mandatory barrier preventing compiler algebraic cancellation across U and decoder
    qc.barrier(range(8))

    # 3. Decoding stage
    if arm_name == "Arm A":
        qc.compose(SCRAMBLER_U1.inverse().decompose(), qubits=[2, 3, 4, 5], inplace=True)
    elif arm_name in ("Arm B", "Arm B-Ctrl", "Arm B-Held"):
        # Remnants (4, 5) strictly unaccessed
        qc.compose(DECODER_CIRC, qubits=[2, 3, 6, 7], inplace=True)
    elif arm_name == "Arm B-Def":
        # Remnants (4, 5) and late photon D2 (3) strictly unaccessed
        qc.compose(SINGLE_PHOTON_DECODER_CIRC, qubits=[2, 6, 7], inplace=True)

    qc.barrier([0, 1, 2, 3])

    # 4. Independent Pauli basis rotations on Pair 1 (0, 2) and Pair 2 (1, 3)
    _apply_pair_basis(qc, 0, 2, basis_pair1)
    _apply_pair_basis(qc, 1, 3, basis_pair2)

    # 5. Unconditional 4-bit readout: c[0]=R1, c[1]=M1', c[2]=R2, c[3]=M2'
    qc.measure([0, 2, 1, 3], [0, 1, 2, 3])
    return qc


def build_spam_calibration_circuits() -> List[Tuple[QuantumCircuit, str]]:
    """Build all 16 computational-basis state preparation circuits on logical `[0, 2, 1, 3]`."""
    cal: List[Tuple[QuantumCircuit, str]] = []
    # Bitstring order in Qiskit classical register is 'c3 c2 c1 c0'
    readout_qubits = [0, 2, 1, 3]  # maps to c0, c1, c2, c3
    for idx in range(16):
        bitstr = format(idx, "04b")  # 'c3c2c1c0'
        qc = QuantumCircuit(8, 4, name=f"spam_cal_{bitstr}")
        for clbit_idx, q in enumerate(readout_qubits):
            bit_char = bitstr[3 - clbit_idx]
            if bit_char == "1":
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
    """Assemble block-randomized 9-basis circuits plus interleaved 16-state SPAM calibration."""
    if n_blocks < 2:
        raise ValueError("n_blocks must be >= 2 to compute unbiased block SEM (ddof=1)")
    circuits: List[QuantumCircuit] = []
    metadata: List[Dict[str, object]] = []

    # Prepend 16-state SPAM calibration block on the exact same 8-qubit register
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
    initial_layout: Sequence[int],
    coupling_map: Sequence[Sequence[int]],
    seed_transpiler: int = 42,
) -> List[QuantumCircuit]:
    """Compile all circuits onto an identical pinned 8-qubit `initial_layout` and verify readout lock."""
    if len(initial_layout) != 8 or len(set(initial_layout)) != 8:
        raise ValueError("initial_layout must contain 8 distinct physical qubit indices")
    compiled = transpile(
        list(circuits),
        coupling_map=list(coupling_map),
        basis_gates=["cz", "sx", "rz", "x"],
        initial_layout=list(initial_layout),
        optimization_level=1,
        seed_transpiler=seed_transpiler,
    )
    expected_initial = tuple(initial_layout)
    for qc in compiled:
        init_phys = tuple(qc.layout.initial_index_layout()[:8])
        if init_phys != expected_initial:
            raise RuntimeError(f"Initial layout drifted on {qc.name}: {init_phys} != {expected_initial}")
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
        # Qiskit bitstring order: bits[0]=c3(M2'), bits[1]=c2(R2), bits[2]=c1(M1'), bits[3]=c0(R1)
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

    # Shot-weighted marginal Pauli expectations <P_1 P_1'> and <Q_2 Q_2'>
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

    # Full 16-stabilizer sum:
    #   1 + sum_P s_P <P_1> + sum_Q s_Q <Q_2> + sum_{P,Q} s_P s_Q <P_1 Q_2>
    # Notice that in setting (P, Q), the single-shot contribution to F_joint is:
    #   w_{P,Q}(pi1, pi2) = (1/16) * ( (1/3)*s_P*pi1 + (1/3)*s_Q*pi2 + s_P*s_Q*pi1*pi2 )
    # which lets us compute the EXACT independent-setting multinomial shot variance!
    joint_corr_sum = 0.0
    var_f_joint = 0.0

    for p, q in JOINT_BASES:
        key = f"{p}{q}"
        sp = BELL_SIGN[p]
        sq = BELL_SIGN[q]
        joint_corr_sum += sp * sq * p12_by_setting[key][0]

        # Exact per-setting shot variance of the linear estimator over 4-bit outcomes
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
