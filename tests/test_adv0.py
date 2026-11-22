import numpy as np
import pytest

# ADV-0 needs the IBM stack (FakeMarrakesh calibration, Aer MPS); CI installs neither, so skip there.
pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")
pytest.importorskip("qiskit_ibm_runtime")

from adv0 import attacks as A
from adv0 import peaked as P


@pytest.fixture(scope="module")
def be():
    return P.backend()


def _statevector(p):
    from qiskit.quantum_info import Statevector
    return Statevector(p.qiskit(measure=False))


def test_region_connected_and_native(be):
    reg = P.choose_region(be, 20)
    assert len(set(reg)) == 20
    edges = P.region_edges(be, reg)
    seen, stack, adj = {0}, [0], {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    while stack:
        for nb in adj.get(stack.pop(), []):
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    assert len(seen) == 20
    for col in P.matchings(edges):
        qs = [q for e in col for q in e]
        assert len(qs) == len(set(qs))


@pytest.mark.parametrize("n,d,seed", [(6, 2, 1), (8, 4, 2), (10, 6, 3)])
def test_circuit_is_peaked_on_secret(be, n, d, seed):
    p = P.build(n, d, seed, be)
    assert {g[0] for g in p.gates} <= {"rz", "sx", "x", "cz"}
    probs = _statevector(p).probabilities_dict()
    assert probs.get(p.secret[::-1], 0) == pytest.approx(1.0, abs=1e-9)
    assert p.n_cz == 2 * sum(len(P.matchings(P.region_edges(be, p.region))[k % len(P.matchings(P.region_edges(be, p.region)))]) for k in range(d))


def test_no_literal_palindrome(be):
    p = P.build(8, 4, 5, be)
    oneq = [g for g in p.gates if g[0] == "rz"]
    k = len(oneq) // 2
    first = sorted(round(g[2], 6) for g in oneq[:k])
    second = sorted(round((-g[2]) % (2 * np.pi), 6) for g in oneq[k:])
    assert first != second


def test_rz_table_matches_closed_form():
    for th in (0.3, 1.1, -2.0):
        a = A._table1(np.diag([np.exp(-0.5j * th), np.exp(0.5j * th)]))
        assert a[1, 1] == pytest.approx(np.cos(th)) and a[1, 2] == pytest.approx(-np.sin(th))
        assert a[2, 2] == pytest.approx(np.cos(th)) and a[2, 1] == pytest.approx(np.sin(th))


def test_pauli_propagation_exact_on_random_circuit():
    from qiskit.quantum_info import SparsePauliOp, Statevector
    rng = np.random.default_rng(0)
    n, gates = 5, []
    for _ in range(60):
        r = rng.integers(4)
        q = int(rng.integers(n))
        if r == 0:
            gates.append(("rz", (q,), float(rng.uniform(0, 6.28))))
        elif r == 1:
            gates.append(("sx", (q,), 0.0))
        elif r == 2:
            gates.append(("x", (q,), 0.0))
        else:
            a, b = rng.choice(n, 2, replace=False)
            gates.append(("cz", (int(a), int(b)), 0.0))
    p = P.Peaked(n=n, depth=0, seed=0, region=list(range(n)), secret="0" * n, gates=gates)
    sv = Statevector(p.qiskit(measure=False))
    for q in range(n):
        lab = ["I"] * n
        lab[n - 1 - q] = "Z"
        want = float(np.real(sv.expectation_value(SparsePauliOp("".join(lab)))))
        got = A.pp_expect_z(gates, n, q, eps=0.0, max_terms=10 ** 9)["z"]
        assert got == pytest.approx(want, abs=1e-9)


def test_attacks_win_small_and_pp_fails_when_starved(be):
    p = P.build(8, 4, 11, be)
    assert A.pauli_propagation(p, eps=0.0, max_terms=10 ** 9)["win"]
    assert A.light_cone_attack(p)["win"]
    assert A.mps_attack(p, chi=16, shots=64)["win"]
    big = P.build(16, 10, 12, be)
    assert not A.pauli_propagation(big, eps=1e-2, max_terms=8)["win"]


def test_hardware_estimate_monotone_in_depth(be):
    reg = P.choose_region(be, 12)
    e = [P.hardware_estimate(P.build(12, d, 3, be, reg), be)["p_peak_pred"] for d in (2, 6, 10)]
    assert 1 > e[0] > e[1] > e[2] > 0
