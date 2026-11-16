"""Offline verification tests for tree-native, light-cone-isolated OSIRIS-HP-004 harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

pytest.importorskip("qiskit")

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace, state_fidelity  # noqa: E402

_MOD_PATH = Path(__file__).resolve().parent.parent / "experiments" / "hp004" / "hp004_joint_stabilizer.py"
_SPEC = importlib.util.spec_from_file_location("hp004_joint_stabilizer", _MOD_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

ARMS = _MOD.ARMS
DEFAULT_MARRAKESH_TREE_LAYOUT = _MOD.DEFAULT_MARRAKESH_TREE_LAYOUT
JOINT_BASES = _MOD.JOINT_BASES
TREE_EDGES_8Q = _MOD.TREE_EDGES_8Q
assemble_hp004_schedule = _MOD.assemble_hp004_schedule
build_hp004_circuit = _MOD.build_hp004_circuit
compile_layout_locked = _MOD.compile_layout_locked
estimate_joint_bell_fidelity = _MOD.estimate_joint_bell_fidelity
summarize_blocks = _MOD.summarize_blocks


def _exact_counts_from_circuit(qc: QuantumCircuit, shots: int = 100_000) -> Dict[str, int]:
    """Deterministic statevector probability table scaled to integer counts for exact analytical checks."""
    bare = QuantumCircuit(qc.num_qubits)
    meas_wire_for_clbit: Dict[int, int] = {}
    for inst in qc.data:
        op = inst.operation.name
        if op == "measure":
            meas_wire_for_clbit[qc.find_bit(inst.clbits[0]).index] = qc.find_bit(inst.qubits[0]).index
        elif op != "barrier":
            bare.append(inst.operation, [bare.qubits[qc.find_bit(q).index] for q in inst.qubits])
    sv = Statevector.from_instruction(bare)
    probs_4b: Dict[str, float] = {}
    probs = sv.probabilities()
    w0, w1, w2, w3 = (meas_wire_for_clbit[i] for i in range(4))
    for idx, p in enumerate(probs):
        if p < 1e-15:
            continue
        c0 = (idx >> w0) & 1
        c1 = (idx >> w1) & 1
        c2 = (idx >> w2) & 1
        c3 = (idx >> w3) & 1
        bitstr = f"{c3}{c2}{c1}{c0}"
        probs_4b[bitstr] = probs_4b.get(bitstr, 0.0) + float(p)
    return {k: int(round(v * shots)) for k, v in probs_4b.items() if int(round(v * shots)) > 0}


@pytest.mark.parametrize(
    ("arm", "expected_f1", "expected_f2", "expected_joint", "max_cz"),
    [
        ("Arm A", 1.0, 1.0, 1.0, 38),
        ("Arm B", 1.0, 1.0, 1.0, 32),
        ("Arm B-Def", 0.5, 0.25, 0.125, 25),
        ("Arm B-Ctrl", 0.5, 0.25, 0.25, 30),
        ("Arm B-Held", 0.25, 0.25, 0.0625, 35),
    ],
)
def test_noiseless_9basis_joint_fidelity_and_physical_light_cone_isolation(
    arm: str, expected_f1: float, expected_f2: float, expected_joint: float, max_cz: int
) -> None:
    basis_counts = {}
    for p, q in JOINT_BASES:
        qc = build_hp004_circuit(arm, p, q)
        assert qc.count_ops().get("cz", 0) <= max_cz
        # Audit physical wire light-cones across Barrier 1 (post-prep) and Barrier 2 (decoding)
        barriers = [i for i, inst in enumerate(qc.data) if inst.operation.name == "barrier"]
        assert len(barriers) == 3
        scram_wires = {qc.find_bit(qb).index for inst in qc.data[barriers[0] + 1 : barriers[1]] for qb in inst.qubits}
        dec_wires = {qc.find_bit(qb).index for inst in qc.data[barriers[1] + 1 : barriers[2]] for qb in inst.qubits}
        # Reference qubits w0(R2) and w1(R1) receive ZERO gates during scrambling and decoding!
        assert scram_wires.isdisjoint({0, 1})
        assert dec_wires.isdisjoint({0, 1})
        # Remnants w2(C1) and w7(C2) receive ZERO gates during restricted decoding!
        if arm in ("Arm B", "Arm B-Ctrl", "Arm B-Held"):
            assert dec_wires.issubset({3, 4, 5, 6})
        elif arm == "Arm B-Def":
            # Quarantined late photon D2=w6 ALSO receives ZERO gates!
            assert dec_wires.issubset({3, 4, 5})
        basis_counts[f"{p}{q}"] = _exact_counts_from_circuit(qc)

    # Cross-check against exact partial_trace state_fidelity on [w0(R2), w1(R1), w4(M1'), w5(M2')]
    qc_zz = build_hp004_circuit(arm, "Z", "Z")
    bare = QuantumCircuit(8)
    for inst in qc_zz.data:
        if inst.operation.name not in ("measure", "barrier"):
            bare.append(inst.operation, [bare.qubits[qc_zz.find_bit(q).index] for q in inst.qubits])
    rho_4q = partial_trace(Statevector.from_instruction(bare), [2, 3, 6, 7])
    # Remaining 4 qubits in ascending wire order are [0(R2), 1(R1), 4(M1'), 5(M2')] -> Bell(0,2) & Bell(1,3)
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
    phi_plus = Statevector.from_label("00") + Statevector.from_label("11")
    phi_dm = DensityMatrix(phi_plus.data / np.linalg.norm(phi_plus.data))
    z00 = DensityMatrix.from_label("00")
    z11 = DensityMatrix.from_label("11")

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
    assert res["f_joint"] == pytest.approx(0.5000, abs=1e-4)


def test_marrakesh_tree_layout_lock_zero_routing_and_conservative_se() -> None:
    circuits, meta = assemble_hp004_schedule(n_blocks=2, seed=99, arms=("Arm A", "Arm B", "Arm B-Def"))
    assert len(circuits) == 16 + 2 * 3 * 9
    assert len(meta) == 70

    b0_names = [m["basis"] for m in meta if m.get("block_index") == 0]
    b1_names = [m["basis"] for m in meta if m.get("block_index") == 1]
    assert b0_names != b1_names

    cmap = [[DEFAULT_MARRAKESH_TREE_LAYOUT[a], DEFAULT_MARRAKESH_TREE_LAYOUT[b]] for a, b in TREE_EDGES_8Q]
    compiled = compile_layout_locked(circuits[:25], initial_layout=DEFAULT_MARRAKESH_TREE_LAYOUT, coupling_map=cmap)
    for qc in compiled:
        assert tuple(qc.layout.initial_index_layout()[:8]) == DEFAULT_MARRAKESH_TREE_LAYOUT
        if qc.name.startswith("hp004_b0_arm_a"):
            assert qc.count_ops().get("cz", 0) == 38

    b_counts = {f"{p}{q}": {"0000": 400, "1111": 400, "0001": 112, "1110": 112} for p, q in JOINT_BASES}
    summary = summarize_blocks([b_counts, b_counts])
    assert summary["f_joint_block_sem_ddof1"] == 0.0
    assert summary["f_joint_se_eff"] == pytest.approx(summary["f_joint_shot_se"])
    assert summary["f_joint_se_eff"] > 0.002
