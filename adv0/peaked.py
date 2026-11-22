"""ADV-0: peaked circuits on a heavy-hex region, and their hardware cost.

Construction (twirled mirror). Pick a connected n-qubit region of the device's coupling graph, grown greedily
from the lowest-error edge. Split its edges into matchings (CZ layers). Build R = d layers of (random SU(2) on
every qubit, then CZ on one matching), plus a final SU(2) layer. The circuit is

    C = X^s . R^dagger . R

so C|0...0> = |s> exactly: one hidden bitstring s carries probability 1.

To keep the two halves from reading as gate-by-gate inverses, every CZ gets an identity gauge: random RZ(phi)
on both qubits before it and RZ(-phi) after (RZ commutes with CZ), plus a random Pauli twirl before it with the
CZ-conjugated Pauli after. All single-qubit runs are then merged into the native basis {rz, sx, x}, and X^s is
merged into the last layer. The result has native gates only and no literal palindrome.

Scope, stated plainly: a mirror-type construction can be undone by an attacker who re-synthesises the circuit and
spots the R^dagger R structure. It is a family for measuring (a) where GENERIC classical simulators stop
recovering s and (b) whether hardware still produces the peak there. It is not, by itself, an advantage candidate;
see docs/adv0/ADV0_REPORT.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

Gate = Tuple[str, Tuple[int, ...], float]  # (name, virtual qubits, parameter or 0.0)
SHOT_SECONDS = 63.0 / 216_000  # measured on XTOP-005 (ibm_marrakesh, 216 circuits x 1000 shots, 63 QPU s)


def backend():
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    return FakeMarrakesh()


def _err(target, op: str, qargs: Tuple[int, ...]) -> float:
    try:
        p = target[op][qargs]
    except KeyError:
        p = target[op].get(tuple(reversed(qargs))) if len(qargs) == 2 else None
    return float(p.error) if p is not None and p.error is not None else 0.0


def device_graph(be) -> Tuple[Dict[int, set], Dict[Tuple[int, int], float]]:
    t = be.target
    adj: Dict[int, set] = {}
    cz: Dict[Tuple[int, int], float] = {}
    for a, b in t["cz"].keys():
        e = (min(a, b), max(a, b))
        if e in cz:
            continue
        cz[e] = _err(t, "cz", (a, b))
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj, cz


def choose_region(be, n: int) -> List[int]:
    """Grow a connected region from the lowest-error CZ edge, adding the frontier qubit whose best connecting
    edge plus readout error is smallest. Qubits with a CZ edge error >= 5% or readout error >= 5% are excluded."""
    t = be.target
    adj, cz = device_graph(be)
    ro = {q: _err(t, "measure", (q,)) for q in adj}
    bad = {q for q in adj if ro[q] >= 0.05}
    good = {e: v for e, v in cz.items() if v < 0.05 and not (set(e) & bad)}
    start = min(good, key=good.get)
    region = list(start)
    while len(region) < n:
        best, score = None, None
        for q in region:
            for nb in adj[q]:
                if nb in region or nb in bad:
                    continue
                e = (min(q, nb), max(q, nb))
                if e not in good:
                    continue
                s = good[e] + ro[nb]
                if score is None or s < score:
                    best, score = nb, s
        if best is None:
            raise RuntimeError(f"region cannot grow past {len(region)} qubits")
        region.append(best)
    return region


def region_edges(be, region: Sequence[int]) -> List[Tuple[int, int]]:
    """Edges of the induced subgraph, as VIRTUAL index pairs (i < j)."""
    adj, _ = device_graph(be)
    pos = {p: i for i, p in enumerate(region)}
    out = set()
    for p in region:
        for nb in adj[p]:
            if nb in pos:
                out.add((min(pos[p], pos[nb]), max(pos[p], pos[nb])))
    return sorted(out)


def matchings(edges: Sequence[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    """Greedy proper edge colouring (heavy-hex has max degree 3, so this gives 3 or 4 colours)."""
    colours: List[List[Tuple[int, int]]] = []
    used: List[set] = []
    for e in edges:
        for k, u in enumerate(used):
            if e[0] not in u and e[1] not in u:
                colours[k].append(e)
                u.update(e)
                break
        else:
            colours.append([e])
            used.append(set(e))
    return colours


@dataclass
class Peaked:
    n: int
    depth: int
    seed: int
    region: List[int]
    secret: str                      # s, character k = virtual qubit k
    gates: List[Gate] = field(default_factory=list)

    @property
    def n_cz(self) -> int:
        return sum(1 for g in self.gates if g[0] == "cz")

    def qiskit(self, measure: bool = True):
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(self.n, self.n if measure else 0, name=f"adv0_n{self.n}_d{self.depth}_s{self.seed}")
        for name, qs, p in self.gates:
            if name == "rz":
                qc.rz(p, qs[0])
            elif name == "sx":
                qc.sx(qs[0])
            elif name == "x":
                qc.x(qs[0])
            elif name == "cz":
                qc.cz(*qs)
            else:
                raise ValueError(name)
        if measure:
            qc.measure(range(self.n), range(self.n))
        return qc


_PAULI = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]),
          "Z": np.diag([1, -1])}


def _haar_su2(rng) -> np.ndarray:
    z = (rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    q = q @ np.diag(np.diag(r) / np.abs(np.diag(r)))
    return q / np.sqrt(np.linalg.det(q))


def _rz(t: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])


def _decompose_zsx(u: np.ndarray, q: int) -> List[Gate]:
    """u (2x2 unitary) as native rz/sx gates on virtual qubit q, via qiskit's ZSX Euler decomposer."""
    from qiskit.synthesis import OneQubitEulerDecomposer
    qc = OneQubitEulerDecomposer("ZSX")(u, simplify=True, atol=1e-12)
    out: List[Gate] = []
    for inst in qc.data:
        nm = inst.operation.name
        if nm == "rz":
            out.append(("rz", (q,), float(inst.operation.params[0]) % (2 * np.pi)))
        elif nm in ("sx", "x"):
            out.append((nm, (q,), 0.0))
        else:
            raise ValueError(nm)
    return out


def _cz_conj(pa: str, pb: str) -> Tuple[str, str]:
    """CZ (pa x pb) CZ = (pa' x pb') up to sign: X/Y on one side picks up a Z on the other."""
    def mul(p: str, flip: bool) -> str:
        if not flip:
            return p
        return {"I": "Z", "Z": "I", "X": "Y", "Y": "X"}[p]
    return mul(pa, pb in "XY"), mul(pb, pa in "XY")


def build(n: int, depth: int, seed: int, be=None, region: Sequence[int] | None = None) -> Peaked:
    be = be or backend()
    region = list(region) if region is not None else choose_region(be, n)
    cols = matchings(region_edges(be, region))
    rng = np.random.default_rng(seed)
    secret = "".join(rng.choice(["0", "1"], size=n))

    # Logical layers: list of ("u", {q: 2x2}) and ("cz", [edges]).
    R: List[Tuple[str, object]] = []
    for k in range(depth):
        R.append(("u", {q: _haar_su2(rng) for q in range(n)}))
        R.append(("cz", cols[k % len(cols)]))
    R.append(("u", {q: _haar_su2(rng) for q in range(n)}))
    Rdg: List[Tuple[str, object]] = []
    for kind, val in reversed(R):
        Rdg.append((kind, {q: m.conj().T for q, m in val.items()} if kind == "u" else val))
    layers = R + Rdg
    layers.append(("u", {q: (_PAULI["X"] if secret[q] == "1" else _PAULI["I"]).astype(complex) for q in range(n)}))

    # Pending single-qubit unitary per qubit; flush into native gates before each CZ that touches the qubit.
    pend = {q: np.eye(2, dtype=complex) for q in range(n)}
    gates: List[Gate] = []

    def flush(q: int):
        gates.extend(_decompose_zsx(pend[q], q))
        pend[q] = np.eye(2, dtype=complex)

    for kind, val in layers:
        if kind == "u":
            for q, m in val.items():
                pend[q] = m @ pend[q]
            continue
        for a, b in val:
            pa, pb = rng.choice(list("IXYZ")), rng.choice(list("IXYZ"))
            fa, fb = rng.uniform(0, 2 * np.pi, size=2)
            pend[a] = _PAULI[pa] @ _rz(fa) @ pend[a]
            pend[b] = _PAULI[pb] @ _rz(fb) @ pend[b]
            flush(a)
            flush(b)
            gates.append(("cz", (a, b), 0.0))
            qa, qb = _cz_conj(pa, pb)
            pend[a] = _rz(-fa) @ _PAULI[qa]
            pend[b] = _rz(-fb) @ _PAULI[qb]
    for q in range(n):
        flush(q)
    return Peaked(n=n, depth=depth, seed=seed, region=region, secret=secret, gates=gates)


def hardware_estimate(p: Peaked, be=None, detect_counts: int = 20) -> Dict[str, float]:
    """Predicted peak probability on the device = product of (1 - error) over every native gate and readout,
    from the backend's reported calibration. This ignores idle decoherence and crosstalk (both made XTOP-005
    worse than its simulator), so it is an optimistic first-order number; validate with noisy simulation."""
    be = be or backend()
    t = be.target
    logf = 0.0
    for name, qs, _ in p.gates:
        if name == "rz":
            continue
        phys = tuple(p.region[q] for q in qs)
        logf += np.log1p(-_err(t, name, phys))
    for q in range(p.n):
        logf += np.log1p(-_err(t, "measure", (p.region[q],)))
    ps = float(np.exp(logf))
    shots = int(max(1000, np.ceil(detect_counts / max(ps, 1e-12))))
    return {"p_peak_pred": ps, "shots_needed": shots, "qpu_seconds_pred": shots * SHOT_SECONDS,
            "uniform_baseline": 2.0 ** -p.n}
