"""Search for a transversal-only Hayden-Preskill scrambler U (on blocks M,B) and decoder V (on blocks B,E).

Logical register: 4 blocks x 2 logical qubits. Block k holds logical qubits (2k, 2k+1).
  R = block 0 (reference), M = 1 (message), B = 2 (black hole -> radiation D), E = 3 (early radiation).
Gate set G (each is fault-detecting on [[4,2,2]]; see osiris_xtop_005.py for the physical versions):
  ("CX", a, b)  transversal CX: CNOT(a1->b1) CNOT(a2->b2)
  ("H4", a)     H(x)H then SWAP inside block a
  ("CZ", a)     CZ(a1, a2)
  ("C21", a)    CNOT(a2 -> a1)      (encoded: physical relabel q0<->q1, zero gates)
  ("SW", a)     SWAP(a1, a2)        (encoded: physical relabel q1<->q2, zero gates)
Requirements on U (with R-M and B-E Bell pairs): I(R:M) = 0 and I(R:B) = 0 after U, so R's information is
only in B together with E. Requirement on V: afterwards (R1,E1) and (R2,E2) are Bell pairs up to a Pauli frame.
"""

from __future__ import annotations

import json
from typing import Dict, List, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Pauli, Statevector, entropy, partial_trace

Op = Tuple
R, M, B, E = 0, 1, 2, 3


def lq(block: int, i: int) -> int:
    return 2 * block + i


def apply(qc: QuantumCircuit, op: Op) -> None:
    k = op[0]
    if k == "CX":
        a, b = op[1], op[2]
        qc.cx(lq(a, 0), lq(b, 0)); qc.cx(lq(a, 1), lq(b, 1))
    elif k == "H4":
        a = op[1]; qc.h(lq(a, 0)); qc.h(lq(a, 1)); qc.swap(lq(a, 0), lq(a, 1))
    elif k == "CZ":
        a = op[1]; qc.cz(lq(a, 0), lq(a, 1))
    elif k == "C21":
        a = op[1]; qc.cx(lq(a, 1), lq(a, 0))
    elif k == "SW":
        a = op[1]; qc.swap(lq(a, 0), lq(a, 1))
    else:
        raise ValueError(op)


def prep_logical() -> QuantumCircuit:
    qc = QuantumCircuit(8)
    for a, b in ((R, M), (B, E)):  # |++> on a, |00> on b, transversal CX a->b  => Bell pairs (a1,b1),(a2,b2)
        qc.h(lq(a, 0)); qc.h(lq(a, 1))
        apply(qc, ("CX", a, b))
    return qc


def logical_circuit(U: Sequence[Op], V: Sequence[Op] = ()) -> QuantumCircuit:
    qc = prep_logical()
    for op in list(U) + list(V):
        apply(qc, op)
    return qc


def mutual_info(sv: Statevector, A: Sequence[int], Bq: Sequence[int]) -> float:
    rho = DensityMatrix(sv)
    keep = lambda ks: partial_trace(rho, [q for q in range(8) if q not in ks])
    return float(entropy(keep(A)) + entropy(keep(Bq)) - entropy(keep(list(A) + list(Bq))))


def bell_frame(sv: Statevector, pairs=((0, 6), (1, 7))) -> Dict[str, int] | None:
    """If (R_i, E_i) are Bell pairs up to Pauli, return the sign of XX, YY, ZZ per pair, else None."""
    out = {}
    for i, (r, e) in enumerate(pairs):
        for P in "XYZ":
            lab = ["I"] * 8
            lab[r] = lab[e] = P
            v = float(np.real(sv.expectation_value(Pauli("".join(lab)[::-1]))))
            if abs(abs(v) - 1) > 1e-9:
                return None
            out[f"{P}{P}_{i + 1}"] = int(round(v))
    return out


def random_ops(rng, blocks: Tuple[int, int], n: int) -> List[Op]:
    a, b = blocks
    ops: List[Op] = []
    for _ in range(n):
        k = rng.choice(["CX", "CX", "H4", "CZ", "C21", "SW"])
        if k == "CX":
            ops.append(("CX", a, b) if rng.random() < 0.5 else ("CX", b, a))
        else:
            ops.append((k, int(rng.choice([a, b]))))
    return ops


def find_scrambler(rng, length: int, tries: int = 4000):
    for _ in range(tries):
        U = random_ops(rng, (M, B), length)
        sv = Statevector(logical_circuit(U))
        if mutual_info(sv, [0, 1], [2, 3]) < 1e-9 and mutual_info(sv, [0, 1], [4, 5]) < 1e-9:
            return U
    return None


def find_decoder(rng, U, max_len: int = 10, tries: int = 20000):
    for _ in range(tries):
        V = random_ops(rng, (B, E), int(rng.integers(1, max_len + 1)))
        fr = bell_frame(Statevector(logical_circuit(U, V)))
        if fr is not None:
            return V, fr
    return None, None


def main() -> None:
    rng = np.random.default_rng(20261121)
    found = {}
    for L, length in ((1, 4), (2, 8), (3, 12)):
        U = find_scrambler(rng, length)
        if U is None:
            print("L", L, "no scrambler"); continue
        V, fr = find_decoder(rng, U)
        print("L", L, "U", U, "V", V, fr)
        found[L] = {"U": U, "V": V, "frame": fr}
    print(json.dumps({k: {"nU": len(v["U"]), "nV": None if v["V"] is None else len(v["V"])} for k, v in found.items()}))


if __name__ == "__main__":
    main()
