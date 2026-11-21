"""Find the best simple 16-qubit path (R M B E repeated) on a heavy-hex backend, avoiding hotspot qubits."""
from __future__ import annotations
from typing import List, Tuple
import numpy as np


def edge_error(backend) -> dict:
    t = backend.target
    out = {}
    for qs, pr in t["cz"].items():
        if pr is not None and pr.error is not None:
            out[tuple(sorted(qs))] = pr.error
    return out


def best_path(backend, n: int = 16, avoid=frozenset()) -> Tuple[List[int], float]:
    t = backend.target
    ee = edge_error(backend)
    ro = {q: (t["measure"][(q,)].error or 0.0) for q in range(backend.num_qubits)}
    adj = {q: set() for q in range(backend.num_qubits)}
    for a, b in ee:
        if a not in avoid and b not in avoid and ee[(a, b)] < 0.5:
            adj[a].add(b); adj[b].add(a)
    best = (None, float("inf"))

    def score(path):
        return sum(ee[tuple(sorted(e))] for e in zip(path, path[1:])) + sum(ro[q] for q in path) / 4

    def dfs(path, seen):
        nonlocal best
        if len(path) == n:
            s = score(path)
            if s < best[1]:
                best = (list(path), s)
            return
        for v in adj[path[-1]]:
            if v not in seen:
                seen.add(v); path.append(v); dfs(path, seen); path.pop(); seen.discard(v)

    for s in range(backend.num_qubits):
        if s not in avoid and adj[s]:
            dfs([s], {s})
    if best[0] is None:
        raise RuntimeError("no clean path")
    p = best[0]
    return (p if p[0] < p[-1] else p[::-1]), best[1]
