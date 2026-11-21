"""Exact checks of the [[4,2,2]] builder (statevector / stabilizer simulation, no hardware)."""

from itertools import product

import numpy as np
import pytest
from qiskit.quantum_info import Clifford, Pauli, Statevector

from xtop005 import code422 as c

B = (0, 1, 2, 3)


def _sv(prog):
    qc = prog.to_qiskit()
    qc.remove_final_measurements()
    return Statevector.from_instruction(qc)


def _exp(sv, label_by_qubit):
    return float(np.real(sv.expectation_value(Pauli(label_by_qubit[::-1]))))


def test_logical_algebra():
    X1, Z1, X2, Z2 = (c.logical_pauli(p, i) for p, i in (("X", 1), ("Z", 1), ("X", 2), ("Z", 2)))
    assert not X1.commutes(Z1) and not X2.commutes(Z2)
    assert X1.commutes(Z2) and X2.commutes(Z1) and X1.commutes(X2) and Z1.commutes(Z2)
    for L in (X1, Z1, X2, Z2):
        for s in ("XXXX", "ZZZZ"):
            assert L.commutes(Pauli(s))
    for i in (1, 2):
        y = c.logical_pauli("Y", i)
        assert y == y.adjoint()  # Hermitian


def test_prep_zero_and_plus():
    p = c.Program(4); c.prep_zero(p, B); sv = _sv(p)
    for lab in ("XXXX", "ZZZZ", "ZIZI", "ZZII"):
        assert _exp(sv, lab) == pytest.approx(1)
    p = c.Program(4); c.prep_plus(p, B); sv = _sv(p)
    for lab in ("XXXX", "ZZZZ", "XXII", "XIXI"):
        assert _exp(sv, lab) == pytest.approx(1)


def test_logical_bell_pairs_all_nine_correlators():
    p = c.Program(8); c.logical_bell_pairs(p, (0, 1, 2, 3), (4, 5, 6, 7)); sv = _sv(p)
    sign = {"X": 1, "Y": -1, "Z": 1}
    for i in (1, 2):
        for P in "XYZ":
            op = c.logical_pauli(P, i).tensor(c.logical_pauli(P, i))  # (M block) x (R block)
            assert float(np.real(sv.expectation_value(op))) == pytest.approx(sign[P])


def test_transversal_cnot_is_logical_cnot_pair():
    """Control block = qubits 0-3 (low), target = 4-7 (high). A.tensor(B) puts B on the low qubits."""
    p = c.Program(8); c.transversal_cnot(p, (0, 1, 2, 3), (4, 5, 6, 7))
    cl = Clifford(p.to_qiskit())
    I4 = Pauli("IIII")
    for i in (1, 2):
        X, Z = c.logical_pauli("X", i), c.logical_pauli("Z", i)
        assert I4.tensor(X).evolve(cl, frame="s").equiv(X.tensor(X))   # X_ctrl -> X_ctrl X_tgt
        assert Z.tensor(I4).evolve(cl, frame="s").equiv(Z.tensor(Z))   # Z_tgt  -> Z_ctrl Z_tgt
        assert X.tensor(I4).evolve(cl, frame="s").equiv(X.tensor(I4))  # X_tgt unchanged
        assert I4.tensor(Z).evolve(cl, frame="s").equiv(I4.tensor(Z))  # Z_ctrl unchanged


@pytest.mark.parametrize("p1,p2", list(product("XYZ", repeat=2)))
def test_readout_decodes_logical_values(p1, p2):
    """On |Phi+>|Phi+> the decoded product L_i(R) L_i(M) must equal the Bell sign in every shot."""
    from qiskit_aer import AerSimulator
    p = c.Program(8)
    c.logical_bell_pairs(p, (0, 1, 2, 3), (4, 5, 6, 7))
    cr = c.measure_logical(p, (0, 1, 2, 3), p1, p2)
    cm = c.measure_logical(p, (4, 5, 6, 7), p1, p2)
    res = AerSimulator(method="stabilizer").run(p.to_qiskit(), shots=200, seed_simulator=1).result().get_counts()
    sign = {"X": 1, "Y": -1, "Z": 1}
    for key in res:
        bits = [int(b) for b in key.replace(" ", "")[::-1]]
        r = c.decode_block([bits[i] for i in cr], p1, p2)
        m = c.decode_block([bits[i] for i in cm], p1, p2)
        assert r["L1"] * m["L1"] == sign[p1] and r["L2"] * m["L2"] == sign[p2]
        assert r["check_ok"] and m["check_ok"]


def test_readout_table_every_setting_measures_correct_logicals():
    for (p1, p2), ro in c.READOUT.items():
        for (sign, supp), (P, i) in ((ro.logical1, (p1, 1)), (ro.logical2, (p2, 2))):
            lab = ["I"] * 4
            for q in supp:
                lab[q] = ro.basis[q]
            op = Pauli("".join(lab)[::-1])
            if sign < 0:
                op = -op
            assert any(op.equiv(s.compose(c.logical_pauli(P, i))) and (op == s.compose(c.logical_pauli(P, i)))
                       for s in c.STABILIZER_GROUP), (p1, p2, i)
    n_with_checks = sum(1 for ro in c.READOUT.values() if ro.checks)
    assert n_with_checks == 2  # only XX and ZZ readouts contain a stabilizer


@pytest.mark.parametrize("q", range(4))
@pytest.mark.parametrize("err", ["x", "z", "y"])
def test_every_single_qubit_error_is_detected(q, err):
    from qiskit_aer import AerSimulator
    p = c.Program(6)
    c.prep_zero(p, B)
    p.add(err, [q])
    sx = c.check(p, B, "X", 4, 5)
    sz = c.check(p, B, "Z", 4, 5)
    res = AerSimulator(method="stabilizer").run(p.to_qiskit(), shots=64, seed_simulator=2).result().get_counts()
    for key in res:
        bits = key.replace(" ", "")[::-1]
        syn = (int(bits[sx[0]]), int(bits[sz[0]]))
        want = (int(err in "zy"), int(err in "xy"))  # X check flags Z/Y, Z check flags X/Y
        assert syn == want and bits[sx[1]] == "0" and bits[sz[1]] == "0"


def test_verified_prep_catches_logical_cnot_fault():
    """X fault on q0 after the 2nd CNOT of prep_zero propagates to X0 X3 = X1 X2 (logical); verification flags it."""
    from qiskit_aer import AerSimulator
    p = c.Program(5)
    p.add("h", [0]); p.add("cx", [0, 1]); p.add("cx", [0, 2]); p.add("x", [0]); p.add("cx", [0, 3])
    p.add("reset", [4]); p.add("cx", [0, 4]); p.add("cx", [1, 4])
    cb = p.new_clbits(1)[0]; p.add("measure", [4], [cb])
    res = AerSimulator(method="stabilizer").run(p.to_qiskit(), shots=32, seed_simulator=3).result().get_counts()
    assert all(k.replace(" ", "")[::-1][cb] == "1" for k in res)


def test_pytket_and_qiskit_emit_same_unitary():
    from qiskit.quantum_info import Operator
    p = c.Program(8)
    c.logical_bell_pairs(p, (0, 1, 2, 3), (4, 5, 6, 7))
    for q in range(4):
        p.add("s", [q]); p.add("sdg", [q + 4]); p.add("cz", [q, q + 4])
    u_tk = p.to_pytket().get_unitary()                       # ILO-BE: qubit 0 most significant
    u_qk = Operator(p.to_qiskit().reverse_bits()).data        # make qiskit big-endian too
    assert np.allclose(u_tk, u_qk)
