"""Exhaustive BFS for a transversal-only decoder (see logical_search.py). Sign-free binary symplectic tableau.

State: 8 stabilizer generators of the 8-logical-qubit state as rows [x(8) | z(8)], canonicalized by GF(2)
row reduction. Target: the stabilizer group contains X_R1X_E1, Z_R1Z_E1, X_R2X_E2, Z_R2Z_E2 (any signs).
"""
from __future__ import annotations
import sys
from collections import deque
import numpy as np
from xtop005.logical_search import B, E, M, R, find_scrambler, lq

N = 8


def rref(T):
    T = T.copy(); r = 0
    for c in range(2 * N):
        piv = next((i for i in range(r, N) if T[i, c]), None)
        if piv is None:
            continue
        T[[r, piv]] = T[[piv, r]]
        for i in range(N):
            if i != r and T[i, c]:
                T[i] ^= T[r]
        r += 1
        if r == N:
            break
    return T


def cx(T, c, t): T[:, t] ^= T[:, c]; T[:, N + c] ^= T[:, N + t]
def h(T, q): T[:, [q, N + q]] = T[:, [N + q, q]]
def cz(T, a, b): T[:, N + a] ^= T[:, b]; T[:, N + b] ^= T[:, a]
def sw(T, a, b): T[:, [a, b]] = T[:, [b, a]]; T[:, [N + a, N + b]] = T[:, [N + b, N + a]]


def apply(T, op):
    k = op[0]; T = T.copy()
    if k == "CX":
        a, b = op[1], op[2]; cx(T, lq(a, 0), lq(b, 0)); cx(T, lq(a, 1), lq(b, 1))
    elif k == "H4":
        a = op[1]; h(T, lq(a, 0)); h(T, lq(a, 1)); sw(T, lq(a, 0), lq(a, 1))
    elif k == "CZ":
        a = op[1]; cz(T, lq(a, 0), lq(a, 1))
    elif k == "C21":
        a = op[1]; cx(T, lq(a, 1), lq(a, 0))
    elif k == "SW":
        a = op[1]; sw(T, lq(a, 0), lq(a, 1))
    return T


def initial():
    T = np.zeros((N, 2 * N), dtype=np.uint8)
    # start from |0..0>: Z_i
    for i in range(N):
        T[i, N + i] = 1
    for a, b in ((R, M), (B, E)):
        h(T, lq(a, 0)); h(T, lq(a, 1)); cx(T, lq(a, 0), lq(b, 0)); cx(T, lq(a, 1), lq(b, 1))
    return T


def in_group(T, row):
    return np.linalg.matrix_rank(np.vstack([T, row]).astype(float)) == np.linalg.matrix_rank(T.astype(float)) if False else _gf2_in(T, row)


def _gf2_in(T, row):
    A = rref(T); v = row.copy()
    for i in range(N):
        piv = np.flatnonzero(A[i])
        if piv.size and v[piv[0]]:
            v ^= A[i]
    return not v.any()


def target(T):
    for i in (0, 1):
        r, e = lq(R, i), lq(E, i)
        x = np.zeros(2 * N, np.uint8); x[r] = x[e] = 1
        z = np.zeros(2 * N, np.uint8); z[N + r] = z[N + e] = 1
        if not (_gf2_in(T, x) and _gf2_in(T, z)):
            return False
    return True


GEN = [("CX", B, E), ("CX", E, B)] + [(k, a) for k in ("H4", "CZ", "C21", "SW") for a in (B, E)]


def bfs(T0, limit=3_000_000):
    start = rref(T0); seen = {start.tobytes(): None}; q = deque([(start, [])])
    while q:
        T, path = q.popleft()
        if target(T):
            return path, len(seen)
        for g in GEN:
            T2 = rref(apply(T, g)); key = T2.tobytes()
            if key not in seen:
                seen[key] = None
                if len(seen) > limit:
                    return None, len(seen)
                q.append((T2, path + [g]))
    return None, len(seen)


if __name__ == "__main__":
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 20261121)
    for L, length in ((1, 4), (2, 8), (3, 12)):
        U = find_scrambler(rng, length)
        T = initial()
        for op in U:
            T = apply(T, op)
        V, n = bfs(T)
        print("L", L, "U", [(str(o[0]),) + tuple(int(x) for x in o[1:]) for o in U], "| orbit states", n, "| V", V, flush=True)
