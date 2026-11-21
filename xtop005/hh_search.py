"""Heavy-hex redesign search: scramblers/decoders WITHOUT in-block relabels (so inter-block transversal CX always
pairs the same physically adjacent qubits on an R M B E | R M B E | ... line). Relabels of E are allowed only at the
very end (they just change how E's readout is decoded). Uses the binary tableau BFS of decoder_bfs.py."""
from __future__ import annotations
import sys
from collections import deque
import numpy as np
from xtop005 import decoder_bfs as d
from xtop005.logical_search import B, E, M, R, logical_circuit, mutual_info
from qiskit.quantum_info import Statevector

GEN_U = [("CX", M, B), ("CX", B, M)] + [(k, a) for k in ("H4", "CZ") for a in (M, B)]
GEN_V = [("CX", B, E), ("CX", E, B)] + [(k, a) for k in ("H4", "CZ") for a in (B, E)]
FINAL_E = [(), (("C21", E),), (("SW", E),), (("C21", E), ("SW", E)), (("SW", E), ("C21", E)),
           (("C21", E), ("SW", E), ("C21", E))]


def target_mod_final(T):
    for seq in FINAL_E:
        T2 = T
        for op in seq:
            T2 = d.apply(T2, op)
        if d.target(T2):
            return seq
    return None


def scrambles(U):
    sv = Statevector(logical_circuit(U))
    return mutual_info(sv, [0, 1], [2, 3]) < 1e-9 and mutual_info(sv, [0, 1], [4, 5]) < 1e-9


def random_U(rng, n):
    while True:
        U = [GEN_U[int(rng.integers(len(GEN_U)))] for _ in range(n)]
        # no immediate self-inverse repeats
        if all(U[i] != U[i + 1] or U[i][0] == "CZ" and False for i in range(n - 1)) and scrambles(U):
            return U


def bfs(T0, limit=2_000_000):
    start = d.rref(T0); seen = {start.tobytes()}; q = deque([(start, [])])
    while q:
        T, path = q.popleft()
        fin = target_mod_final(T)
        if fin is not None:
            return path, fin, len(seen)
        for g in GEN_V:
            T2 = d.rref(d.apply(T, g)); k = T2.tobytes()
            if k not in seen:
                seen.add(k)
                if len(seen) > limit:
                    return None, None, len(seen)
                q.append((T2, path + [g]))
    return None, None, len(seen)


if __name__ == "__main__":
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
    for L, n in ((1, 4), (2, 8), (3, 12)):
        for attempt in range(6):
            U = random_U(rng, n)
            T = d.initial()
            for op in U:
                T = d.apply(T, op)
            V, fin, ns = bfs(T)
            if V is not None:
                print("L", L, "U", [(str(o[0]),) + tuple(int(v) for v in o[1:]) for o in U], "V", V, "final", fin,
                      "states", ns, flush=True)
                break
            print("L", L, "attempt", attempt, "no decoder, states", ns, flush=True)
