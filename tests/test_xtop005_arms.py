"""Exact checks of the OSIRIS-XTOP-005 four-arm generator (no noise, no hardware)."""

import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford, Operator, Statevector

from xtop005 import code422 as c
from xtop005 import logical_search as ls
from xtop005 import osiris_xtop_005 as x

DEPTHS = (1, 2, 3)


@pytest.mark.parametrize("L", DEPTHS)
def test_scrambler_hides_reference(L):
    """After U_L, R shares no information with M alone or B alone (only with B and E jointly)."""
    sv = Statevector(ls.logical_circuit(x.SCRAMBLERS[L]))
    assert ls.mutual_info(sv, [0, 1], [2, 3]) == pytest.approx(0, abs=1e-9)
    assert ls.mutual_info(sv, [0, 1], [4, 5]) == pytest.approx(0, abs=1e-9)
    assert ls.mutual_info(sv, [0, 1], [4, 5, 6, 7]) == pytest.approx(4, abs=1e-9)  # all of R's 2 qubits


@pytest.mark.parametrize("L", DEPTHS)
def test_decoder_restores_bell_pairs(L):
    assert ls.bell_frame(Statevector(ls.logical_circuit(x.SCRAMBLERS[L], x.DECODERS[L]))) is not None


@pytest.mark.parametrize("L", DEPTHS)
def test_logical_frame_exact(L):
    fr = x.logical_frame(L)
    assert set(fr) == {a + b for a, b in x.SETTINGS}


@pytest.mark.parametrize("L", DEPTHS)
@pytest.mark.parametrize("opts", [{}, {"verify": True}, {"checks": True, "verify": True}])
def test_encoded_arm_noiseless_all_settings(L, opts):
    """Full-encoding program: F = 1 and every shot passes every check, for all 9 settings."""
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method="stabilizer")
    frame = x.logical_frame(L)
    per = {}
    for st in x.SETTINGS:
        b = x.build("C", L, st, 0, "full", **opts)
        qc = x.to_qiskit(b)
        qc.data = [i for i in qc.data if i.operation.name != "delay"]
        per["".join(st)] = x.correlators(b, sim.run(qc, shots=128, seed_simulator=5).result().get_counts())
    assert x.fidelity(per, frame, "raw") == pytest.approx(1)
    assert x.fidelity(per, frame, "acc") == pytest.approx(1)
    assert all(v["acc"]["n"] == v["raw"]["n"] for v in per.values())


def _logical_images(qc):
    cl = Clifford(qc)
    return {n: c.logical_pauli(P, i).evolve(cl, frame="s") for n, (P, i) in
            {"X1": ("X", 1), "Z1": ("Z", 1), "X2": ("X", 2), "Z2": ("Z", 2)}.items()}


def _L(*terms):
    from qiskit.quantum_info import Pauli
    out = Pauli("IIII")
    for P, i in terms:
        out = out.compose(c.logical_pauli(P, i))
    return out


def _equiv_mod_stab(a, b):
    return any(a.equiv(s.compose(b)) for s in c.STABILIZER_GROUP)


def test_encoded_cz_is_exact_logical_cz():
    qc = QuantumCircuit(4); qc.s(0); qc.sdg(1); qc.sdg(2); qc.s(3)
    im = _logical_images(qc)
    assert im["X1"] == _L(("X", 1), ("Z", 2)) or _equiv_mod_stab(im["X1"], _L(("X", 1), ("Z", 2)))
    assert _equiv_mod_stab(im["X2"], _L(("Z", 1), ("X", 2)))
    assert _equiv_mod_stab(im["Z1"], _L(("Z", 1))) and _equiv_mod_stab(im["Z2"], _L(("Z", 2)))
    for prep in (c.prep_zero, c.prep_plus):  # stays in the +1 code space on encoded states
        pr = c.Program(4); prep(pr, (0, 1, 2, 3))
        sv = Statevector(pr.to_qiskit().compose(qc))
        for s in ("XXXX", "ZZZZ"):
            assert float(np.real(sv.expectation_value(c._pauli(s)))) == pytest.approx(1)


def test_encoded_h4_is_logical_h_h_swap():
    qc = QuantumCircuit(4); qc.h(range(4))
    im = _logical_images(qc)
    assert _equiv_mod_stab(im["X1"], _L(("Z", 2))) and _equiv_mod_stab(im["Z1"], _L(("X", 2)))
    assert _equiv_mod_stab(im["X2"], _L(("Z", 1))) and _equiv_mod_stab(im["Z2"], _L(("X", 1)))


def test_relabels_are_logical_cnot21_and_swap():
    def relabeled(perm, P, i):
        lab = c.LOGICAL_LABELS[f"{P}{i}"]
        return c._pauli("".join(lab[perm[k]] for k in range(4)))
    # C21: slots 0<->1  => X2 -> X1 X2, Z1 -> Z1 Z2 (CNOT control 2, target 1)
    p01 = [1, 0, 2, 3]
    assert _equiv_mod_stab(relabeled(p01, "X", 2), _L(("X", 1), ("X", 2)))
    assert _equiv_mod_stab(relabeled(p01, "Z", 1), _L(("Z", 1), ("Z", 2)))
    assert _equiv_mod_stab(relabeled(p01, "X", 1), _L(("X", 1)))
    # SW: slots 1<->2  => X1 <-> X2, Z1 <-> Z2
    p12 = [0, 2, 1, 3]
    assert _equiv_mod_stab(relabeled(p12, "X", 1), _L(("X", 2)))
    assert _equiv_mod_stab(relabeled(p12, "Z", 1), _L(("Z", 2)))


def test_native_xy4_is_identity():
    seq, sp_a, sp_b = x._xy4_native()
    qc = QuantumCircuit(1)
    for g in seq:
        qc.append(g, [0])
    assert Operator(qc).equiv(Operator(QuantumCircuit(1)))
    assert {g.name for g in seq} == {"x", "rz"}
    for sp in (sp_a, sp_b):
        assert len(sp) == len(seq) + 1 and sum(sp) == pytest.approx(1) and min(sp) >= 0


def test_heavy_hex_is_bipartite_and_layout_avoids_hotspots():
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    be = FakeMarrakesh()
    va, vb = x.sublattices(be)
    assert va.isdisjoint(vb) and len(va | vb) == be.num_qubits
    for a, b in be.coupling_map.get_edges():
        assert (a in va) != (b in va)
    qc = x.to_qiskit(x.build("A", 3, ("X", "X")))
    lay, info = x.choose_layout(qc, be)
    assert not set(lay) & set(info["hotspots"])


def test_canonical_json_has_no_floats():
    s = x.canonical_json({"b": 0.5, "a": [1, float("nan"), np.float64(2.0)]})
    assert s == '{"a":[1,"nan","2.000000"],"b":"0.500000"}'


@pytest.mark.parametrize("L", DEPTHS)
def test_r_encoded_arm_noiseless_all_settings(L):
    """Heavy-hex design (only R encoded): F = 1 and no shot rejected, all 9 settings."""
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method="stabilizer")
    per = {}
    for st in x.SETTINGS:
        b = x.build("C", L, st, 0, "r")
        assert b.prog.n_qubits == 10 and len(b.postselect) == 2
        qc = x.to_qiskit(b)
        qc.data = [i for i in qc.data if i.operation.name != "delay"]
        per["".join(st)] = x.correlators(b, sim.run(qc, shots=128, seed_simulator=6).result().get_counts())
    assert x.fidelity(per, x.logical_frame(L), "acc") == pytest.approx(1)
    assert all(v["acc"]["n"] == v["raw"]["n"] for v in per.values())


def test_r_encoder_is_star_and_detects_every_single_qubit_error():
    enc = QuantumCircuit(4)
    for g, *q in x.R_ENCODER:
        getattr(enc, g)(*q)
        if g == "cx":
            assert 0 in q  # star centred on qubit 0 (a degree-3 heavy-hex node)
    from qiskit.quantum_info import random_statevector
    for seed in range(3):
        qc = QuantumCircuit(4); qc.initialize(random_statevector(4, seed=seed).data, [1, 2]); qc.compose(enc, inplace=True)
        sv = Statevector(qc.decompose(reps=6))
        for lab in ("XXXX", "ZZZZ"):
            assert float(np.real(sv.expectation_value(c._pauli(lab)))) == pytest.approx(1)
        for q in range(4):
            for e in "xyz":
                q2 = qc.copy(); getattr(q2, e)(q); q2.compose(enc.inverse(), inplace=True)
                assert Statevector(q2.decompose(reps=6)).probabilities([0, 3])[0] == pytest.approx(0, abs=1e-9)


def test_r_encoded_arm_detects_idle_error_on_reference():
    """An X error on any R qubit during the transit window is caught by the syndrome bits."""
    from qiskit_aer import AerSimulator
    for q in range(4):
        b = x.build("C", 1, ("Z", "Z"), 0, "r")
        qc = x.to_qiskit(b)
        k = next(i for i, ins in enumerate(qc.data) if ins.operation.name == "delay")
        qc.data = [ins for ins in qc.data if ins.operation.name != "delay"]
        from qiskit.circuit.library import XGate
        from qiskit.circuit import CircuitInstruction
        qc.data.insert(k, CircuitInstruction(XGate(), [qc.qubits[q]]))
        cnt = AerSimulator(method="stabilizer").run(qc, shots=64, seed_simulator=7).result().get_counts()
        assert x.correlators(b, cnt)["acc"]["n"] == 0
