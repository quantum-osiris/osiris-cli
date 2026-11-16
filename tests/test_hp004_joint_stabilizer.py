"""Offline verification tests for OSIRIS-HP-004 9-basis 4-qubit joint Bell stabilizer harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace, state_fidelity

_MOD_PATH = Path(__file__).resolve().parent.parent / "experiments" / "hp004" / "hp004_joint_stabilizer.py"
_SPEC = importlib.util.spec_from_file_location("hp004_joint_stabilizer", _MOD_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

ARMS = _MOD.ARMS
JOINT_BASES = _MOD.JOINT_BASES
assemble_hp004_schedule = _MOD.assemble_hp004_schedule
build_hp004_circuit = _MOD.build_hp004_circuit
compile_layout_locked = _MOD.compile_layout_locked
estimate_joint_bell_fidelity = _MOD.estimate_joint_bell_fidelity
summarize_blocks = _MOD.summarize_blocks


def _exact_counts_from_circuit(qc: QuantumCircuit, shots: int = 100_000) -> Dict[str, int]:
    """Deterministic statevector probability table scaled to integer counts for exact analytical checks."""
    # Strip measurements & final barriers to compute exact pre-measurement statevector
    bare = QuantumCircuit(qc.num_qubits)
    for inst in qc.data:
        if inst.operation.name not in ("measure", "barrier"):
            bare.append(inst.operation, [ bare.qubits[qc.find_bit(q).index] for q in inst.qubits ])
    sv = Statevector.from_instruction(bare)
    # Readout maps q0->c0, q2->c1, q1->c2, q3->c3 (bitstring 'c3 c2 c1 c0')
    probs_4b: Dict[str, float] = {}
    probs = sv.probabilities()
    for idx, p in enumerate(probs):
        if p < 1e-15:
            continue
        q0 = (idx >> 0) & 1
        q1 = (idx >> 1) & 1
        q2 = (idx >> 2) & 1
        q3 = (idx >> 3) & 1
        bitstr = f"{q3}{q1}{q2}{q0}"
        probs_4b[bitstr] = probs_4b.get(bitstr, 0.0) + float(p)
    return {k: int(round(v * shots)) for k, v in probs_4b.items() if int(round(v * shots)) > 0}


@pytest.mark.parametrize(
    ("arm", "expected_f1", "expected_f2", "expected_joint"),
    [
        ("Arm A", 1.0, 1.0, 1.0),
        ("Arm B", 1.0, 1.0, 1.0),
        ("Arm B-Def", 0.5, 0.25, 0.125),
        # Unentangled B,E still transmits classical stabilizer correlations through (D1, D2),
        # hitting the exact 0-ebit separable ceiling 4/16 = 0.25 (missed by HP-003's F1*F2 = 0.0625!).
        ("Arm B-Ctrl", 0.25, 0.25, 0.25),
        ("Arm B-Held", 0.25, 0.25, 0.0625),
    ],
)
def test_noiseless_9basis_joint_fidelity_matches_exact_analytical_channel_values(
    arm: str, expected_f1: float, expected_f2: float, expected_joint: float
) -> None:
    basis_counts = {}
    for p, q in JOINT_BASES:
        qc = build_hp004_circuit(arm, p, q)
        basis_counts[f"{p}{q}"] = _exact_counts_from_circuit(qc)

    # Also cross-check against exact Qiskit partial_trace state_fidelity on Z-basis pre-measurement state
    qc_zz = build_hp004_circuit(arm, "Z", "Z")
    bare = QuantumCircuit(8)
    for inst in qc_zz.data:
        if inst.operation.name not in ("measure", "barrier"):
            bare.append(inst.operation, [bare.qubits[qc_zz.find_bit(q).index] for q in inst.qubits])
    rho_4q = partial_trace(Statevector.from_instruction(bare), [4, 5, 6, 7])
    # Target |Phi+> on (0, 2) and |Phi+> on (1, 3)
    target_qc = QuantumCircuit(4)
    target_qc.h(0); target_qc.cx(0, 2)
    target_qc.h(1); target_qc.cx(1, 3)
    exact_dm_overlap = float(state_fidelity(rho_4q, Statevector.from_instruction(target_qc)))

    res = estimate_joint_bell_fidelity(basis_counts)
    assert exact_dm_overlap == pytest.approx(expected_joint, abs=1e-6)
    assert res["f1_marginal"] == pytest.approx(expected_f1, abs=1e-4)
    assert res["f2_marginal"] == pytest.approx(expected_f2, abs=1e-4)
    assert res["f_joint"] == pytest.approx(exact_dm_overlap, abs=1e-4)


def test_1ebit_schmidt_rank_2_mixture_fools_marginal_product_but_is_caught_by_f_joint() -> None:
    """Prove why HP-004's 9-basis F_joint is required over HP-003's 3-basis marginal product.

    The 1-ebit Schmidt-rank-2 state:
      rho_SR2 = 1/4 ( |Phi+>_1|00>_2 + |Phi+>_1|11>_2 + |00>_1|Phi+>_2 + |11>_1|Phi+>_2 )
    achieves F1 = F2 = 0.75 and F1*F2 = 0.5625 (> 0.5), yet has true joint Bell overlap 0.5000.
    """
    phi_plus = Statevector.from_label("00") + Statevector.from_label("11")
    phi_dm = DensityMatrix(phi_plus.data / np.linalg.norm(phi_plus.data))
    z00 = DensityMatrix.from_label("00")
    z11 = DensityMatrix.from_label("11")

    # Tensor order: Pair2 (c3,c2) (x) Pair1 (c1,c0) so bitstring index matches 'c3c2c1c0'
    rho_sr2 = 0.25 * (
        z00.tensor(phi_dm)
        + z11.tensor(phi_dm)
        + phi_dm.tensor(z00)
        + phi_dm.tensor(z11)
    )

    shots = 100_000
    basis_counts: Dict[str, Dict[str, int]] = {}
    for p, q in JOINT_BASES:
        rot = QuantumCircuit(4)
        # Pair 1 is qubits (0, 1); Pair 2 is qubits (2, 3)
        if p == "X":
            rot.h(0); rot.h(1)
        elif p == "Y":
            rot.sdg(0); rot.h(0); rot.sdg(1); rot.h(1)
        if q == "X":
            rot.h(2); rot.h(3)
        elif q == "Y":
            rot.sdg(2); rot.h(2); rot.sdg(3); rot.h(3)
        rotated = rho_sr2.evolve(rot)
        probs = rotated.probabilities()
        basis_counts[f"{p}{q}"] = {
            format(i, "04b"): int(round(prob * shots))
            for i, prob in enumerate(probs)
            if int(round(prob * shots)) > 0
        }

    res = estimate_joint_bell_fidelity(basis_counts)
    assert res["f1_marginal"] == pytest.approx(0.75, abs=1e-4)
    assert res["f2_marginal"] == pytest.approx(0.75, abs=1e-4)
    assert res["f_marginal_product"] == pytest.approx(0.5625, abs=1e-4)
    # Exact 9-basis F_joint pins rho_SR2 right at the 0.5000 1-ebit ceiling!
    assert res["f_joint"] == pytest.approx(0.5000, abs=1e-4)


def test_schedule_randomization_layout_lock_and_conservative_se() -> None:
    circuits, meta = assemble_hp004_schedule(n_blocks=2, seed=99, arms=("Arm A", "Arm B"))
    # 16 SPAM cal + 2 blocks * 2 arms * 9 bases = 52 circuits
    assert len(circuits) == 16 + 36
    assert len(meta) == 52

    # Verify blocks are shuffled (not sorted by arm/basis) and differ across blocks
    b0_names = [m["basis"] for m in meta if m.get("block_index") == 0]
    b1_names = [m["basis"] for m in meta if m.get("block_index") == 1]
    assert b0_names != b1_names

    # Compile onto a pinned 8-qubit ring/heavy-hex subgraph within a 16-qubit chip
    edges = [[i, i + 1] for i in range(15)] + [[i + 1, i] for i in range(15)]
    pinned_layout = [4, 5, 6, 7, 8, 9, 10, 11]
    compiled = compile_layout_locked(circuits[:20], initial_layout=pinned_layout, coupling_map=edges)
    for qc in compiled:
        assert tuple(qc.layout.initial_index_layout()[:8]) == tuple(pinned_layout)
        # Arm A barrier prevents U * U^dagger algebraic collapse
        if qc.name.startswith("hp004_b0_arm_a"):
            assert qc.count_ops().get("cz", 0) >= 16

    # Conservative SE_eff never drops below shot-noise SE even if 2 blocks agree identically
    b_counts = {f"{p}{q}": {"0000": 400, "1111": 400, "0001": 112, "1110": 112} for p, q in JOINT_BASES}
    summary = summarize_blocks([b_counts, b_counts])
    assert summary["f_joint_block_sem_ddof1"] == 0.0
    assert summary["f_joint_se_eff"] == pytest.approx(summary["f_joint_shot_se"])
    assert summary["f_joint_se_eff"] > 0.002
