#!/usr/bin/env python3
"""OSIRIS-XTOP-005: bare vs [[4,2,2]]-encoded Hayden-Preskill recovery, idle vs staggered DD.

Four arms, identical logical program in every arm:
  A  BARE_IDLE      8 physical qubits (R, M, B, E: 2 logical qubits each), idle windows left as delays
  B  BARE_DD        same, idle windows filled with bipartite-staggered native XY4
  C  ENC_IDLE       every block encoded in [[4,2,2]] (16 data + 2 ancilla/flag = 18 qubits), delays
  D  ENC_DD         same as C with staggered XY4

Logical protocol (all logical gates are fault-detecting on [[4,2,2]]):
  prep   Bell pairs (R_i, M_i) and (B_i, E_i), i = 1, 2
  U_L    scrambler on (M, B), depth L = 1, 2, 3  (found by xtop005/logical_search.py + decoder_bfs.py;
         satisfies I(R:M) = I(R:B) = 0 exactly, i.e. R's information is only in B together with E)
  idle   transit window of `idle_ns` on all data qubits (where DD acts)
  V_L    decoder on (B, E), found by exhaustive BFS; restores Bell pairs (R_i, E_i) up to a Pauli frame
  read   R and E in one of 9 two-qubit Pauli settings; encoded arms additionally run flagged X/Z checks
         on R and E, use verified state preparation, and Z-read M and B (ZZZZ parity)

Gate set and its [[4,2,2]] realization (slot order of each block = its code qubits q0..q3):
  CX(a,b)  transversal CX slot-by-slot           -> logical CNOT(a1->b1) CNOT(a2->b2)
  H4(a)    H on the 4 slots                      -> logical (H x H) then SWAP
  CZ(a)    S, Sdg, Sdg, S on slots 0..3          -> logical CZ(a1, a2)           (virtual Z on Heron)
  C21(a)   relabel slots 0<->1 (no gates)        -> logical CNOT(a2 -> a1)
  SW(a)    relabel slots 1<->2 (no gates)        -> logical SWAP(a1, a2)
Bare arms apply the same logical gates on 2-qubit blocks; SWAPs are relabels there too (same cost model).

Stage 0  noiseless: compiled ISA circuits must reproduce the ideal Bell frame (F = 1 within shot noise,
         and every one of the 9 settings must give the exact +-1 correlator signs).
Stage 1  AerSimulator.from_backend(FakeMarrakesh): noisy fidelity per arm/depth, raw and post-selected
         (encoded arms), keep fraction, 99 % bootstrap CIs. Nothing is sent to hardware from this module.

Layout: one hotspot-avoiding layout per encoding (bare / encoded), shared by idle and DD arms, so DD is
the only difference between A/B and between C/D. Hotspots = bottom-decile T2, top-decile readout error,
or endpoint of a top-decile CZ edge of the backend's calibration snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from xtop005 import code422 as c

# ---------------------------------------------------------------------------------------------------------
# Logical program
# ---------------------------------------------------------------------------------------------------------
R, M, B, E = 0, 1, 2, 3
BLOCK_NAMES = "RMBE"
Op = Tuple

SCRAMBLERS: Dict[int, List[Op]] = {
    1: [("CX", 2, 1), ("C21", 2), ("SW", 1), ("CX", 1, 2)],
    2: [("CX", 2, 1), ("SW", 1), ("CZ", 2), ("CZ", 2), ("C21", 2), ("CX", 1, 2), ("C21", 1), ("SW", 2)],
    3: [("CZ", 1), ("C21", 2), ("CX", 1, 2), ("CX", 2, 1), ("H4", 2), ("CX", 1, 2), ("CZ", 2), ("SW", 1),
        ("C21", 2), ("CX", 2, 1), ("H4", 2), ("CZ", 2)],
}
DECODERS: Dict[int, List[Op]] = {
    1: [("C21", 2), ("CX", 3, 2), ("C21", 2), ("SW", 2), ("CX", 2, 3)],
    2: [("C21", 3), ("SW", 2), ("CX", 3, 2), ("C21", 3), ("SW", 2), ("CX", 2, 3)],
    3: [("H4", 2), ("CX", 2, 3), ("CX", 3, 2), ("CZ", 2), ("C21", 3), ("SW", 2), ("CX", 3, 2), ("H4", 2),
        ("CX", 2, 3)],
}
SETTINGS: List[Tuple[str, str]] = list(product("XYZ", repeat=2))
ARMS = ("A", "B", "C", "D")
ARM_LABEL = {"A": "BARE_IDLE", "B": "BARE_DD", "C": "ENC_IDLE", "D": "ENC_DD"}
ENCODED = {"A": False, "B": False, "C": True, "D": True}
WITH_DD = {"A": False, "B": True, "C": False, "D": True}


@dataclass
class Built:
    """A logical-level circuit on physical (virtual) qubits plus the bookkeeping to decode its shots."""
    arm: str
    depth: int
    setting: Tuple[str, str]
    prog: c.Program
    readout: Dict[str, List[int]] = field(default_factory=dict)  # name -> clbits
    postselect: List[int] = field(default_factory=list)         # clbits that must read 0
    parity_zero: List[List[int]] = field(default_factory=list)  # clbit groups that must have even parity
    data_qubits: List[int] = field(default_factory=list)


# R-block encoder into the [[4,2,2]] code space (XXXX = ZZZZ = +1): inputs on qubits 1 and 2, qubits 0 and 3
# start in |0>. Star-shaped (every CX touches qubit 0), so it maps onto a degree-3 heavy-hex node with no SWAPs.
# Found by BFS (5 CX is the minimum on a star). R is never operated on while encoded, so the logical basis
# the encoder picks does not matter: the inverse (reversed list; every gate is self-inverse) undoes it, and the
# two stabilizer syndromes come back out on qubits 0 and 3.
R_ENCODER = [("cx", 1, 0), ("cx", 2, 0), ("h", 3), ("cx", 3, 0), ("cx", 0, 1), ("cx", 0, 2)]


def _bare(depth: int, setting: Tuple[str, str], idle_ns: float, encode_r: bool = False) -> Built:
    """Bare M, B, E (2 qubits each). R is bare (8 qubits total) or, with encode_r, one [[4,2,2]] block on
    qubits 0-3 that is encoded right after the Bell pairs are made, idles through U / transit / V, and is
    unencoded before readout; the unencoder leaves the two stabilizer syndromes on qubits 0 and 3."""
    n = 10 if encode_r else 8
    p = c.Program(n)
    if encode_r:
        perm = {R: [1, 2], M: [4, 5], B: [6, 7], E: [8, 9]}
    else:
        perm = {k: [2 * k, 2 * k + 1] for k in range(4)}
    for a, b in ((R, M), (B, E)):
        for i in (0, 1):
            p.add("h", [perm[a][i]]); p.add("cx", [perm[a][i], perm[b][i]])
    if encode_r:
        for g, *qs in R_ENCODER:
            p.add(g, qs)

    def do(op):
        k = op[0]
        if k == "CX":
            for i in (0, 1):
                p.add("cx", [perm[op[1]][i], perm[op[2]][i]])
        elif k == "H4":
            for q in perm[op[1]]:
                p.add("h", [q])
            perm[op[1]].reverse()
        elif k == "CZ":
            p.add("cz", perm[op[1]])
        elif k == "C21":
            p.add("cx", [perm[op[1]][1], perm[op[1]][0]])
        elif k == "SW":
            perm[op[1]].reverse()
        else:
            raise ValueError(op)

    for op in SCRAMBLERS[depth]:
        do(op)
    p.add("barrier", list(range(n)))
    p.add("delay_ns", list(range(n)), ())  # placeholder, expanded at emit time
    p.add("barrier", list(range(n)))
    for op in DECODERS[depth]:
        do(op)
    built = Built("bare", depth, setting, p, data_qubits=list(range(n)))
    built.kind = "renc" if encode_r else "bare"  # type: ignore[attr-defined]
    if encode_r:
        for g, *qs in reversed(R_ENCODER):  # every gate is self-inverse
            p.add(g, qs)
        syn = p.new_clbits(2)
        p.add("measure", [0], [syn[0]]); p.add("measure", [3], [syn[1]])
        built.postselect = syn
    for blk in (R, E):
        cs = p.new_clbits(2)
        for i, (q, P) in enumerate(zip(perm[blk], setting)):
            if P == "X":
                p.add("h", [q])
            elif P == "Y":
                p.add("sdg", [q]); p.add("h", [q])
            p.add("measure", [q], [cs[i]])
        built.readout[BLOCK_NAMES[blk]] = cs
    built.idle_ns = idle_ns  # type: ignore[attr-defined]
    return built


def _encoded(depth: int, setting: Tuple[str, str], idle_ns: float, checks: bool = False, verify: bool = False) -> Built:
    """16 data qubits (4 blocks x 4) + ancilla 16 + flag 17."""
    p = c.Program(18)
    anc, flag = 16, 17
    slots = {k: [4 * k + j for j in range(4)] for k in range(4)}
    post: List[int] = []
    for a, b in ((R, M), (B, E)):
        post += c.logical_bell_pairs(p, tuple(slots[a]), tuple(slots[b]), verify_anc=anc if verify else None)

    def do(op):
        k = op[0]
        if k == "CX":
            c.transversal_cnot(p, tuple(slots[op[1]]), tuple(slots[op[2]]))
        elif k == "H4":
            for q in slots[op[1]]:
                p.add("h", [q])
        elif k == "CZ":
            s = slots[op[1]]
            p.add("s", [s[0]]); p.add("sdg", [s[1]]); p.add("sdg", [s[2]]); p.add("s", [s[3]])
        elif k == "C21":
            s = slots[op[1]]; s[0], s[1] = s[1], s[0]
        elif k == "SW":
            s = slots[op[1]]; s[1], s[2] = s[2], s[1]
        else:
            raise ValueError(op)

    for op in SCRAMBLERS[depth]:
        do(op)
    p.add("barrier", list(range(16)))
    p.add("delay_ns", list(range(16)), ())
    p.add("barrier", list(range(16)))
    for op in DECODERS[depth]:
        do(op)
    if checks:  # flagged weight-4 checks: ~3x the routed CZ count on heavy-hex (see README)
        for blk in (R, E):
            for kind in ("X", "Z"):
                post += list(c.check(p, tuple(slots[blk]), kind, anc, flag))
    built = Built("enc", depth, setting, p, postselect=post, data_qubits=list(range(16)))
    for blk in (R, E):
        built.readout[BLOCK_NAMES[blk]] = c.measure_logical(p, tuple(slots[blk]), *setting)
    for blk in (M, B):
        cs = p.new_clbits(4)
        for q, cb in zip(slots[blk], cs):
            p.add("measure", [q], [cb])
        built.parity_zero.append(cs)
    built.idle_ns = idle_ns  # type: ignore[attr-defined]
    return built


def build(arm: str, depth: int, setting: Tuple[str, str], idle_ns: float = 2000.0, mode: str = "r",
          **enc_opts) -> Built:
    """mode 'r' (default, heavy-hex design): only the quarantined reference R is encoded.
    mode 'full': all four blocks encoded (needs all-to-all connectivity; see README)."""
    if not ENCODED[arm]:
        b = _bare(depth, setting, idle_ns)
    elif mode == "r":
        b = _bare(depth, setting, idle_ns, encode_r=True)
    elif mode == "full":
        b = _encoded(depth, setting, idle_ns, **enc_opts)
        b.kind = "enc"  # type: ignore[attr-defined]
    else:
        raise ValueError(mode)
    b.arm = arm
    return b


def to_qiskit(b: Built):
    from qiskit import QuantumCircuit
    p = b.prog
    qc = QuantumCircuit(p.n_qubits, p.n_clbits, name=f"xtop005_{b.arm}_L{b.depth}_{''.join(b.setting)}")
    for g, qs, cs in p.gates:
        if g == "measure":
            qc.measure(qs[0], cs[0])
        elif g == "reset":
            qc.reset(qs[0])
        elif g == "barrier":
            qc.barrier(*qs)
        elif g == "delay_ns":
            for q in qs:
                qc.delay(b.idle_ns, q, unit="ns")  # type: ignore[attr-defined]
        else:
            getattr(qc, g)(*qs)
    return qc


# ---------------------------------------------------------------------------------------------------------
# Decoding shots
# ---------------------------------------------------------------------------------------------------------
def _bits(key: str) -> List[int]:
    return [int(x) for x in key.replace(" ", "")[::-1]]


def correlators(b: Built, counts: Mapping) -> Dict[str, object]:  # type: ignore[name-defined]
    """Returns sums of <P P>_1, <Q Q>_2, <PP QQ> over raw and accepted shots."""
    raw = {"n": 0, "c1": 0, "c2": 0, "c12": 0}
    acc = {"n": 0, "c1": 0, "c2": 0, "c12": 0}
    P, Q = b.setting
    for key, n in counts.items():
        bits = _bits(key)
        if getattr(b, "kind", "bare") in ("bare", "renc"):
            r = [1 - 2 * bits[i] for i in b.readout["R"]]
            e = [1 - 2 * bits[i] for i in b.readout["E"]]
            v1, v2 = r[0] * e[0], r[1] * e[1]
            ok = all(bits[i] == 0 for i in b.postselect)
        else:
            dr = c.decode_block([bits[i] for i in b.readout["R"]], P, Q)
            de = c.decode_block([bits[i] for i in b.readout["E"]], P, Q)
            v1, v2 = dr["L1"] * de["L1"], dr["L2"] * de["L2"]
            ok = (dr["check_ok"] and de["check_ok"] and all(bits[i] == 0 for i in b.postselect)
                  and all(sum(bits[i] for i in grp) % 2 == 0 for grp in b.parity_zero))
        for d, take in ((raw, True), (acc, ok)):
            if take:
                d["n"] += n; d["c1"] += n * v1; d["c2"] += n * v2; d["c12"] += n * v1 * v2
    return {"raw": raw, "acc": acc}


def fidelity(per_setting: Dict[str, Dict], frame: Dict[str, Tuple[int, int]], which: str) -> float:
    """Two-pair Bell fidelity F = (1/16) sum_{a,b in IXYZ} s_a s_b <sigma_a sigma_a (x) sigma_b sigma_b>."""
    one = {P: [] for P in "XYZ"}; two = {P: [] for P in "XYZ"}; both = {}
    for key, d in per_setting.items():
        P, Q = key[0], key[1]
        x = d[which]
        if x["n"] == 0:
            return float("nan")
        s1, s2 = frame[key]
        one[P].append(s1 * x["c1"] / x["n"]); two[Q].append(s2 * x["c2"] / x["n"])
        both[key] = s1 * s2 * x["c12"] / x["n"]
    return (1 + sum(np.mean(v) for v in one.values()) + sum(np.mean(v) for v in two.values())
            + sum(both.values())) / 16


# ---------------------------------------------------------------------------------------------------------
# Backend: hotspots, layout, compilation, staggered DD
# ---------------------------------------------------------------------------------------------------------
def hotspots(backend, decile: float = 0.10) -> set:
    t = backend.target
    n = backend.num_qubits
    t2 = np.array([t.qubit_properties[q].t2 or 0.0 for q in range(n)])
    ro = np.array([(t["measure"][(q,)].error or 0.0) for q in range(n)])
    cz = {qs: (pr.error or 0.0) for qs, pr in t["cz"].items() if pr is not None}
    bad = set(np.flatnonzero(t2 <= np.quantile(t2, decile)).tolist())
    bad |= set(np.flatnonzero(ro >= np.quantile(ro, 1 - decile)).tolist())
    thr = np.quantile(list(cz.values()), 1 - decile)
    for qs, e in cz.items():
        if e >= thr:
            bad |= set(qs)
    return bad


def choose_layout(qc, backend, seed: int = 11, tries: int = 40) -> Tuple[List[int], Dict]:
    """Route on the coupling graph with hotspots removed, then keep that virtual->physical layout."""
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap
    bad = hotspots(backend)
    edges = [e for e in backend.coupling_map.get_edges() if e[0] not in bad and e[1] not in bad]
    cmap = CouplingMap(edges)
    best = None
    for s in range(seed, seed + tries):
        tq = transpile(qc, coupling_map=cmap, basis_gates=["cz", "rz", "sx", "x", "id", "measure", "reset", "delay"],
                       optimization_level=3, seed_transpiler=s)
        lay = tq.layout.initial_virtual_layout(filter_ancillas=True)
        phys = [lay[qc.qubits[i]] for i in range(qc.num_qubits)]
        ncz = tq.count_ops().get("cz", 0)
        if best is None or ncz < best[0]:
            best = (ncz, phys, s)
    return best[1], {"hotspots": sorted(bad), "routing_seed": best[2], "cz_on_clean_graph": best[0]}


def sublattices(backend) -> Tuple[set, set]:
    """2-colouring of the heavy-hex coupling graph (bipartite)."""
    import rustworkx as rx
    g = backend.coupling_map.graph.to_undirected()
    colour = {}
    for comp in rx.connected_components(g):
        start = min(comp)
        colour[start] = 0
        stack = [start]
        while stack:
            u = stack.pop()
            for v in g.neighbors(u):
                if v not in colour:
                    colour[v] = 1 - colour[u]; stack.append(v)
                elif colour[v] == colour[u]:
                    raise RuntimeError("coupling graph is not bipartite")
    return ({q for q, k in colour.items() if k == 0}, {q for q, k in colour.items() if k == 1})


def _xy4_native():
    """XY4 = X Y X Y with Y = Rz(pi/2) X Rz(-pi/2) (exact up to global phase); only x and virtual rz."""
    from qiskit.circuit.library import RZGate, XGate
    seq, sp_a, sp_b = [], [], []
    for k in range(4):
        if k % 2 == 0:
            seq += [XGate()]
        else:
            seq += [RZGate(-math.pi / 2), XGate(), RZGate(math.pi / 2)]
    # pulse k (k=0..3) centred at (2k+1)/8 tau on V_A and (2k+2)/8 tau on V_B; rz gates get zero spacing
    def spacing(offsets):
        out, last = [], 0.0
        for k, t in enumerate(offsets):
            if k % 2 == 0:
                out += [t - last]
            else:
                out += [t - last, 0.0, 0.0]
            last = t
        out.append(1.0 - last)
        return out
    sp_a = spacing([1 / 8, 3 / 8, 5 / 8, 7 / 8])
    sp_b = spacing([2 / 8, 4 / 8, 6 / 8, 1.0 - 1e-6])
    return seq, sp_a, sp_b


def compile_isa(qc, backend, layout: Sequence[int], dd: bool, seed: int = 7, tries: int = 8):
    """Transpile on the fixed layout (best of `tries` routing seeds by CZ count), schedule ALAP, then pad delays
    (or staggered XY4) on active wires only."""
    from qiskit import transpile
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDelay, PadDynamicalDecoupling
    t = backend.target
    tq = min((transpile(qc, backend=backend, initial_layout=list(layout), optimization_level=3, seed_transpiler=seed + k)
              for k in range(tries)), key=lambda c_: (c_.count_ops().get("cz", 0), c_.depth()))
    active = sorted({tq.find_bit(q).index for inst in tq.data for q in inst.qubits
                     if inst.operation.name not in ("barrier", "delay")})
    passes = [ALAPScheduleAnalysis(target=t)]
    if dd:
        va, vb = sublattices(backend)
        seq, sp_a, sp_b = _xy4_native()
        qa = [q for q in active if q in va]; qb = [q for q in active if q in vb]
        passes += [PadDynamicalDecoupling(target=t, dd_sequence=seq, qubits=qa, spacing=sp_a, skip_reset_qubits=True),
                   ALAPScheduleAnalysis(target=t),
                   PadDynamicalDecoupling(target=t, dd_sequence=seq, qubits=qb, spacing=sp_b, skip_reset_qubits=True)]
    passes += [PadDelay(target=t)]
    sched = PassManager(passes).run(tq)
    # drop delays on wires that carry no operation (keeps Aer's qubit truncation effective)
    from qiskit import QuantumCircuit
    out = QuantumCircuit(*sched.qregs, *sched.cregs, name=qc.name)
    for inst in sched.data:
        qi = [sched.find_bit(q).index for q in inst.qubits]
        if inst.operation.name == "delay" and qi[0] not in active:
            continue
        out.append(inst)
    out._layout = sched.layout
    return out, active


def isa_counts(isa) -> Dict[str, int]:
    ops = isa.count_ops()
    xs = ops.get("x", 0)
    return {"cz": ops.get("cz", 0), "sx": ops.get("sx", 0), "x": xs, "measure": ops.get("measure", 0),
            "reset": ops.get("reset", 0), "delay": ops.get("delay", 0), "depth": isa.depth()}


# ---------------------------------------------------------------------------------------------------------
# Stage 0 / Stage 1
# ---------------------------------------------------------------------------------------------------------
def logical_frame(depth: int) -> Dict[str, Tuple[int, int]]:
    """Ideal signs of <P_R1 P_E1> and <Q_R2 Q_E2> from the bare noiseless circuit (exact, stabilizer sim)."""
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method="stabilizer")
    out = {}
    for P, Q in SETTINGS:
        b = build("A", depth, (P, Q), idle_ns=0)
        qc = to_qiskit(b)
        qc.data = [i for i in qc.data if i.operation.name != "delay"]
        cnt = sim.run(qc, shots=64, seed_simulator=1).result().get_counts()
        d = correlators(b, cnt)["raw"]
        s1, s2 = d["c1"] / d["n"], d["c2"] / d["n"]
        if abs(abs(s1) - 1) > 1e-12 or abs(abs(s2) - 1) > 1e-12 or abs(d["c12"] / d["n"] - s1 * s2) > 1e-12:
            raise AssertionError(f"logical program is not an exact Bell-pair recovery at L={depth} {P}{Q}")
        out[P + Q] = (int(round(s1)), int(round(s2)))
    return out


def bootstrap_ci(per_setting, frame, which, rng, n_boot=400, alpha=0.01):
    keys = list(per_setting)
    vals = []
    for _ in range(n_boot):
        res = {}
        for k in keys:
            x = per_setting[k][which]
            n = x["n"]
            if n == 0:
                return (float("nan"), float("nan"))
            # resample the joint distribution of (v1, v2) from its 4-cell multinomial
            p11 = (n + x["c1"] + x["c2"] + x["c12"]) / (4 * n)
            p1m = (n + x["c1"] - x["c2"] - x["c12"]) / (4 * n)
            pm1 = (n - x["c1"] + x["c2"] - x["c12"]) / (4 * n)
            pmm = max(0.0, 1 - p11 - p1m - pm1)
            pr = np.clip([p11, p1m, pm1, pmm], 0, None); pr = pr / pr.sum()
            a, b_, cc, d = rng.multinomial(n, pr)
            res[k] = {which: {"n": n, "c1": a + b_ - cc - d, "c2": a - b_ + cc - d, "c12": a - b_ - cc + d}}
        vals.append(fidelity(res, frame, which))
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def _canon(obj):
    """exec-canon-1: sorted keys, no floats (floats -> fixed 6-decimal strings)."""
    if isinstance(obj, float) or isinstance(obj, np.floating):
        return "nan" if math.isnan(obj) else f"{float(obj):.6f}"
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _canon(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    return obj


def canonical_json(obj) -> str:
    return json.dumps(_canon(obj), sort_keys=True, separators=(",", ":"))


def qpy_sha(circuits) -> str:
    from qiskit import qpy
    buf = io.BytesIO(); qpy.dump(list(circuits), buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def prepare(backend, depths=(1, 3), idles=(2000.0, 10000.0), mode: str = "r", enc_opts=None):
    """Build and compile every circuit. Returns (layouts, cells); cell = dict(meta) + 'items' [(Built, isa)].
    One hotspot-avoiding layout per encoding, chosen on the largest circuit and shared by idle/DD arms."""
    enc_opts = dict(enc_opts or {})
    big = (max(depths), max(idles))
    layouts = {}
    for key, arm in (("bare", "A"), ("enc", "C")):
        layouts[key] = choose_layout(to_qiskit(build(arm, big[0], ("X", "X"), big[1], mode,
                                                     **(enc_opts if key == "enc" else {}))), backend)
    cells = []
    for L in depths:
        for idle in idles:
            for arm in ARMS:
                key = "enc" if ENCODED[arm] else "bare"
                items = []
                for st in SETTINGS:
                    b = build(arm, L, st, idle, mode, **(enc_opts if ENCODED[arm] else {}))
                    isa, active = compile_isa(to_qiskit(b), backend, layouts[key][0], WITH_DD[arm])
                    items.append((b, isa))
                cells.append({"arm": arm, "label": ARM_LABEL[arm], "depth": L, "idle_ns": idle,
                              "active_qubits": len(active), "isa": isa_counts(items[0][1]), "items": items})
    return layouts, cells


def analyze(cell, counts_list, rng, frame=None) -> Dict:
    frame = frame or logical_frame(cell["depth"])
    per = {"".join(b.setting): correlators(b, cnt) for (b, _), cnt in zip(cell["items"], counts_list)}
    out = {k: v for k, v in cell.items() if k != "items"}
    out["F_raw"] = fidelity(per, frame, "raw")
    out["F_raw_ci99"] = bootstrap_ci(per, frame, "raw", rng)
    if ENCODED[cell["arm"]]:
        n_raw = sum(v["raw"]["n"] for v in per.values()); n_acc = sum(v["acc"]["n"] for v in per.values())
        out.update({"F_det": fidelity(per, frame, "acc"), "F_det_ci99": bootstrap_ci(per, frame, "acc", rng),
                    "p_keep": n_acc / n_raw})
    out["correlator_sums"] = per
    return out


def stage0(cells, seed: int = 1, shots: int = 64) -> None:
    """Noiseless compiled-ISA check: F = 1 exactly and no shot rejected, every cell."""
    from qiskit_aer import AerSimulator
    ideal = AerSimulator()
    for cell in cells:
        res = ideal.run([i[1] for i in cell["items"]], shots=shots, seed_simulator=seed).result()
        per = {"".join(b.setting): correlators(b, res.get_counts(k)) for k, (b, _) in enumerate(cell["items"])}
        f0 = fidelity(per, logical_frame(cell["depth"]), "raw")
        if abs(f0 - 1) > 1e-9 or any(v["acc"]["n"] != v["raw"]["n"] for v in per.values()):
            raise AssertionError(f"Stage 0 failed: {cell['arm']} L={cell['depth']} idle={cell['idle_ns']} F={f0}")
        cell["stage0_F"] = f0


def run(depths=(1, 3), shots=1000, idles=(2000.0, 10000.0), seed=20261121, out_dir: Path | None = None,
        mode: str = "r", checks: bool = False, verify: bool = False) -> Dict:
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    enc_opts = {"checks": checks, "verify": verify} if mode == "full" else {}
    backend = FakeMarrakesh()
    noisy = AerSimulator.from_backend(backend)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    layouts, cells = prepare(backend, depths, idles, mode, enc_opts)
    stage0(cells)
    report = {"backend": backend.name, "shots": shots, "seed": seed, "mode": mode, "encoded_options": enc_opts,
              "layouts": {k: {"physical": v[0], **v[1]} for k, v in layouts.items()}, "cells": []}
    for n, cell in enumerate(cells):
        res = noisy.run([i[1] for i in cell["items"]], shots=shots, seed_simulator=seed + n).result()
        out = analyze(cell, [res.get_counts(k) for k in range(len(cell["items"]))], rng)
        report["cells"].append(out)
        print(json.dumps(_canon({k: out[k] for k in out if k not in ("isa", "correlator_sums")})), flush=True)
    report["qpy_sha256_all_isa"] = qpy_sha([i[1] for cell in cells for i in cell["items"]])
    report["wall_s"] = round(time.time() - t0, 1)
    body = canonical_json(report)
    report_receipt = {"canon": "exec-canon-1", "sha256": hashlib.sha256(body.encode()).hexdigest()}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "xtop005_stage1.json").write_text(body + "\n")
        (out_dir / "xtop005_stage1.receipt.json").write_text(json.dumps(report_receipt, sort_keys=True) + "\n")
    report["receipt"] = report_receipt
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--depths", default="1,3")
    ap.add_argument("--idles-ns", default="2000,10000")
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--mode", choices=("r", "full"), default="r",
                    help="r: encode only the quarantined reference (heavy-hex design); full: all four blocks")
    ap.add_argument("--checks", action="store_true", help="full mode: flagged X/Z checks on R and E")
    ap.add_argument("--verify", action="store_true", help="full mode: verified |00>/|++> preparation")
    a = ap.parse_args(argv)
    rep = run(tuple(int(x) for x in a.depths.split(",")), a.shots,
              tuple(float(x) for x in a.idles_ns.split(",")), a.seed, a.out, a.mode, a.checks, a.verify)
    print("receipt", rep["receipt"]["sha256"], "wall", rep["wall_s"], "s")


if __name__ == "__main__":
    main()
