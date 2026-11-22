"""ADV-0 classical attack suite. Each attack tries to recover the hidden string s of a peaked circuit.

  pauli_propagation  Heisenberg-picture Pauli propagation of every Z_i with coefficient and term-count
                     truncation; bit i is recovered when the sign of <Z_i> is right and |<Z_i>| >= 0.5.
  light_cone         exact statevector of each qubit's backward light cone (when the cone fits in memory).
  mps                matrix-product-state sampling with a bond-dimension cap; wins if the modal sample is s.

An attack "wins" a circuit when it recovers all n bits. All attacks are generic: none looks for the mirror
structure. A structure-aware attack is listed as an open item in the report.
"""

from __future__ import annotations

import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

from adv0.peaked import Gate, Peaked

_P = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1.0, -1.0])]
_SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_CZ = np.diag([1, 1, 1, -1]).astype(complex)


def _table1(g: np.ndarray) -> np.ndarray:
    """A[l, k] with g^dagger P_l g = sum_k A[l, k] P_k (real for Hermitian P_l)."""
    a = np.zeros((4, 4))
    for l in range(4):
        m = g.conj().T @ _P[l] @ g
        for k in range(4):
            a[l, k] = np.real(np.trace(_P[k] @ m)) / 2
    return a


def _table2(g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Clifford 2-qubit gate: permutation and sign over the 16 Pauli pairs (index 4*la + lb)."""
    perm, sign = np.zeros(16, dtype=np.int8), np.zeros(16)
    for la in range(4):
        for lb in range(4):
            m = g.conj().T @ np.kron(_P[la], _P[lb]) @ g
            for ka in range(4):
                for kb in range(4):
                    c = np.real(np.trace(np.kron(_P[ka], _P[kb]) @ m)) / 4
                    if abs(c) > 1e-9:
                        perm[4 * la + lb], sign[4 * la + lb] = 4 * ka + kb, c
    return perm, sign


def _clifford1(a: np.ndarray):
    perm = np.argmax(np.abs(a), axis=1).astype(np.int8)
    return perm, a[np.arange(4), perm]


SX_T = _clifford1(_table1(_SX))
X_T = _clifford1(_table1(_X))
CZ_T = _table2(_CZ)


class _Ops:
    """Propagated observable: rows of Pauli letters (0=I,1=X,2=Y,3=Z) with real coefficients."""

    def __init__(self, n: int, q: int):
        self.L = np.zeros((1, n), dtype=np.int8)
        self.L[0, q] = 3
        self.c = np.ones(1)
        self.dropped = 0.0  # sum of squared coefficients removed by truncation

    def cliff1(self, q: int, t):
        perm, sign = t
        old = self.L[:, q].copy()
        self.L[:, q] = perm[old]
        self.c *= sign[old]

    def cz(self, a: int, b: int):
        perm, sign = CZ_T
        idx = 4 * self.L[:, a] + self.L[:, b]
        new = perm[idx]
        self.L[:, a], self.L[:, b] = new // 4, new % 4
        self.c *= sign[idx]

    def rz(self, q: int, theta: float):
        # rz^dag X rz = cos X - sin Y ; rz^dag Y rz = cos Y + sin X  (checked against _table1 in tests)
        co, si = np.cos(theta), np.sin(theta)
        col = self.L[:, q]
        m = (col == 1) | (col == 2)
        if not m.any():
            return
        if abs(si) < 1e-12:
            self.c[m] *= co
            return
        sub = self.L[m].copy()
        cs = self.c[m].copy()
        sgn = np.where(sub[:, q] == 1, -si, si)
        sub[:, q] = 3 - sub[:, q]  # 1 <-> 2
        if abs(co) < 1e-12:
            self.L[m] = sub
            self.c[m] = cs * sgn
            return
        self.c[m] *= co
        self.L = np.concatenate([self.L, sub])
        self.c = np.concatenate([self.c, cs * sgn])

    def merge(self, eps: float, max_terms: int):
        L = np.ascontiguousarray(self.L)
        v = L.view(np.dtype((np.void, L.shape[1]))).ravel()
        _, first, inv = np.unique(v, return_index=True, return_inverse=True)
        c = np.bincount(inv.ravel(), weights=self.c, minlength=len(first))
        L = L[first]
        keep = np.abs(c) > eps
        if keep.sum() > max_terms:
            thr = np.partition(np.abs(c), -max_terms)[-max_terms]
            keep &= np.abs(c) >= thr
        self.dropped += float(np.sum(c[~keep] ** 2))
        self.L, self.c = L[keep], c[keep]

    def expect_zero(self) -> float:
        diag = np.all((self.L == 0) | (self.L == 3), axis=1)
        return float(self.c[diag].sum())


def pp_expect_z(gates: Sequence[Gate], n: int, q: int, eps: float = 1e-5, max_terms: int = 2 ** 15,
                merge_every: int = 2 ** 14) -> Dict:
    """<0| C^dag Z_q C |0> by backward Heisenberg propagation (O <- G^dag O G, last gate first)."""
    o = _Ops(n, q)
    peak = 1
    for name, qs, th in reversed(gates):
        if name == "rz":
            o.rz(qs[0], th)
            if len(o.c) > merge_every:
                o.merge(eps, max_terms)
        elif name == "sx":
            o.cliff1(qs[0], SX_T)
        elif name == "x":
            o.cliff1(qs[0], X_T)
        elif name == "cz":
            o.cz(*qs)
        peak = max(peak, len(o.c))
    o.merge(eps, max_terms)
    return {"z": o.expect_zero(), "dropped": o.dropped, "max_terms_seen": peak}


def pauli_propagation(p: Peaked, eps: float = 1e-5, max_terms: int = 2 ** 15, order_seed: int = 0) -> Dict:
    """Stops at the first qubit it fails to recover (an attack must recover every bit)."""
    t0 = time.time()
    order = np.random.default_rng(order_seed).permutation(p.n)
    got, worst = 0, 1.0
    for q in order:
        r = pp_expect_z(p.gates, p.n, int(q), eps, max_terms)
        want = 1.0 if p.secret[q] == "0" else -1.0
        worst = min(worst, r["z"] * want)
        if r["z"] * want < 0.5:
            return {"win": False, "bits_recovered_before_fail": got, "failed_z_signed": r["z"] * want,
                    "dropped_weight": r["dropped"], "seconds": time.time() - t0}
        got += 1
    return {"win": True, "bits_recovered_before_fail": got, "min_z_signed": worst, "seconds": time.time() - t0}


def light_cone(gates: Sequence[Gate], q: int) -> Tuple[List[int], List[Gate]]:
    cone, keep = {q}, []
    for g in reversed(gates):
        if set(g[1]) & cone:
            cone.update(g[1])
            keep.append(g)
    return sorted(cone), list(reversed(keep))


def light_cone_attack(p: Peaked, limit: int = 26) -> Dict:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Pauli
    from qiskit_aer import AerSimulator
    t0 = time.time()
    sim = AerSimulator(method="statevector")
    sizes = []
    for q in range(p.n):
        cone, gs = light_cone(p.gates, q)
        sizes.append(len(cone))
        if len(cone) > limit:
            return {"win": False, "max_cone": max(sizes), "cone_limit": limit, "qubit": q,
                    "seconds": time.time() - t0}
        pos = {v: i for i, v in enumerate(cone)}
        qc = QuantumCircuit(len(cone))
        for name, qs, th in gs:
            m = [pos[x] for x in qs]
            getattr(qc, name)(*([th] if name == "rz" else []), *m)
        qc.save_expectation_value(Pauli("Z"), [pos[q]])
        z = float(np.real(sim.run(qc).result().data()["expectation_value"]))
        want = 1.0 if p.secret[q] == "0" else -1.0
        if z * want < 0.5:
            return {"win": False, "max_cone": max(sizes), "qubit": q, "z_signed": z * want,
                    "seconds": time.time() - t0}
    return {"win": True, "max_cone": max(sizes), "seconds": time.time() - t0}


def mps_attack(p: Peaked, chi: int = 64, shots: int = 200, seed: int = 7) -> Dict:
    from qiskit_aer import AerSimulator
    t0 = time.time()
    sim = AerSimulator(method="matrix_product_state", matrix_product_state_max_bond_dimension=chi,
                       matrix_product_state_truncation_threshold=1e-10, seed_simulator=seed)
    counts = sim.run(p.qiskit(), shots=shots).result().get_counts()
    target = p.secret[::-1]  # qiskit strings are little-endian
    modal = max(counts, key=counts.get)
    return {"win": modal == target, "p_peak_est": counts.get(target, 0) / shots, "chi": chi,
            "seconds": time.time() - t0}
