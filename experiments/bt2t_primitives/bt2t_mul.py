"""OSIRIS-BT2T-002 Milestone 2 (Part 1): exact 10-qubit reversible 2T group multiplication.

Under frozen encoding ``bt2t-enc-1`` (sha256 = 3e3b02d13539daaf431af39056d031f777a785a556eb9da21ef4ea6097dfc4c5):
  * Register x occupies qubits 0..4 (qubit 0 = LSB of x in 0..31).
  * Register y occupies qubits 5..9 (qubit 5 = LSB of y in 0..31).
  * Computational basis state |x>_{0..4} |y>_{5..9} has 10-qubit integer index x | (y << 5) in 0..1023.

Left-multiplication map U_mul |x>|y> = |x>|mu(x, y)>:
  * If x in 0..23 AND y in 0..23: mu(x, y) = mul_table[x][y] in 0..23 (bijection of 0..23 by Cayley's theorem).
  * If x in 24..31 OR y in 24..31: mu(x, y) = y (unitary identity extension on all 448 unused-codeword pairs).

Two independent zero-ancilla 10-qubit reversible syntheses (pure computational-basis gates X, CX, CCX, MCX,
CSWAP so every computational basis amplitude has phase identically +1):
  1. ``multiplication_circuit_generic()``: generic permutation reference decomposing each left translation L_x
     (x in 1..23) on 0..23 into disjoint cycles and paired disjoint transpositions controlled on |x>_{0..4}.
  2. ``multiplication_circuit_structured()``: algebraic semidirect-product synthesis exploiting
     2T ~= Q_8 |>< Z_3 via g = q(g) * omega^{c(g)} with omega = (-1 - i - j - k)/2 (index 23, order 3),
     where Ad_omega(i, j, k) = (j, k, i) acts as a 2-CNOT cycle on the imaginary-axis bits of Q_8 while
     preserving the Q_8 sign bit, combined with 3-bit Q_8 multiplication and Z_3 modular addition.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

BT2T_INVERSION_DIR = Path(__file__).resolve().parents[1] / "bt2t_inversion"


def _load_bt2t_base():
    mod_name = "_bt2t_enc1_base"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, BT2T_INVERSION_DIR / "bt2t.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load base encoding module from {BT2T_INVERSION_DIR / 'bt2t.py'}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


bt2t_base = _load_bt2t_base()

ENCODING_VERSION = "bt2t-enc-1"
ENCODING_SHA256 = "3e3b02d13539daaf431af39056d031f777a785a556eb9da21ef4ea6097dfc4c5"
N_REG_QUBITS = 5
N_MUL_QUBITS = 10
N_REG_STATES = 1 << N_REG_QUBITS  # 32
N_BASIS_STATES = 1 << N_MUL_QUBITS  # 1024
N_ELEMENTS = 24
N_VALID_PAIRS = N_ELEMENTS * N_ELEMENTS  # 576
N_UNUSED_PAIRS = N_BASIS_STATES - N_VALID_PAIRS  # 448
UNUSED_CODEWORDS = tuple(range(N_ELEMENTS, N_REG_STATES))  # 24..31

# Canonical Z_3 generator omega = (-1 - i - j - k) / 2 (index 23, doubled coords (-1, -1, -1, -1)):
#   omega^2 = (-1 + i + j + k) / 2 (index 16), omega^3 = +1 (index 0),
#   and Ad_omega(q) = omega * q * omega^{-1} maps (i, j, k) -> (j, k, i).
OMEGA_INDEX = 23
OMEGA_SQ_INDEX = 16

CtrlSpec = Union[int, Tuple[int, int]]
GateTuple = Tuple[str, Tuple[int, ...], Tuple[int, ...]]


def verify_frozen_encoding() -> Dict[str, object]:
    """Verify base bt2t-enc-1 record and SHA-256 digest are intact."""
    rec = bt2t_base.encoding_record()
    if rec["encoding_version"] != ENCODING_VERSION or rec["sha256"] != ENCODING_SHA256:
        raise RuntimeError(
            f"Frozen encoding mismatch: got {rec['encoding_version']} ({rec['sha256']})"
        )
    return rec


def mul_table_24() -> List[List[int]]:
    """24x24 group multiplication table on 2T from exact integer quaternion arithmetic."""
    return bt2t_base.multiplication_table()


def mu(x: int, y: int, table: Sequence[Sequence[int]] | None = None) -> int:
    """Extended left-multiplication target mu(x, y) on {0..31}^2."""
    if not (0 <= x < N_REG_STATES and 0 <= y < N_REG_STATES):
        raise ValueError(f"(x, y)=({x}, {y}) out of 5-qubit range 0..31")
    if x < N_ELEMENTS and y < N_ELEMENTS:
        tbl = mul_table_24() if table is None else table
        return int(tbl[x][y])
    return y


def pack_xy(x: int, y: int) -> int:
    """Pack (x, y) in {0..31}^2 into 10-qubit Qiskit computational basis index x | (y << 5)."""
    return (x & 0x1F) | ((y & 0x1F) << N_REG_QUBITS)


def unpack_xy(state: int) -> Tuple[int, int]:
    """Unpack 10-qubit basis index into (x, y) in {0..31}^2."""
    return state & 0x1F, (state >> N_REG_QUBITS) & 0x1F


def multiplication_permutation_10q() -> List[int]:
    """Exact 1024-element bijection for U_mul |x>_{0..4}|y>_{5..9} = |x>_{0..4}|mu(x, y)>_{5..9}."""
    tbl = mul_table_24()
    perm = [0] * N_BASIS_STATES
    for y in range(N_REG_STATES):
        for x in range(N_REG_STATES):
            idx_in = pack_xy(x, y)
            y_out = tbl[x][y] if (x < N_ELEMENTS and y < N_ELEMENTS) else y
            perm[idx_in] = pack_xy(x, y_out)
    return perm


# ---------------------------------------------------------------------------
# Semidirect-product structure 2T ~= Q_8 |>< Z_3
# ---------------------------------------------------------------------------


def coset_index_z3(g: int) -> int:
    """Homomorphism c: 2T -> Z_3 = {0, 1, 2} with kernel Q_8 (indices 0..7).

    On Q_8 (0..7): c(g) = 0.
    On Hurwitz units (8..23) with doubled coords (s_a, s_b, s_c, s_d) in {+1, -1}^4:
      * c(g) = 1 when s_a * s_b * s_c * s_d == +1 (even number of minus signs, coset Q_8 * omega)
      * c(g) = 2 when s_a * s_b * s_c * s_d == -1 (odd number of minus signs, coset Q_8 * omega^2)
    """
    if not (0 <= g < N_ELEMENTS):
        raise ValueError(f"Element index {g} outside 2T range 0..23")
    if g < 8:
        return 0
    els = bt2t_base.elements()
    sa, sb, sc, sd = els[g]
    return 1 if (sa * sb * sc * sd == 1) else 2


def ad_omega_q8(q: int, power: int = 1) -> int:
    """Conjugation automorphism Ad_{omega^power}(q) = omega^power * q * omega^{-power} on Q_8 (0..7).

    For omega = (-1 - i - j - k)/2 (index 23):
      Ad_omega(+-1) = +-1, Ad_omega(+-i) = +-j, Ad_omega(+-j) = +-k, Ad_omega(+-k) = +-i.
    In Q_8 index bits (q2, q1, q0) where q0 is sign and (q2, q1) in {00:1, 01:i, 10:j, 11:k}:
      * power == 1 mod 3: (q2, q1) -> (q1 ^ q2, q2)   [CX(q1->q2); CX(q2->q1)], q0 unchanged.
      * power == 2 mod 3: (q2, q1) -> (q1, q1 ^ q2)   [CX(q2->q1); CX(q1->q2)], q0 unchanged.
    """
    if not (0 <= q < 8):
        raise ValueError(f"Q_8 element index {q} outside 0..7")
    p = power % 3
    if p == 0:
        return q
    q0 = q & 1
    q1 = (q >> 1) & 1
    q2 = (q >> 2) & 1
    if p == 1:
        n2, n1 = q1 ^ q2, q2
    else:
        n2, n1 = q1, q1 ^ q2
    return (n2 << 2) | (n1 << 1) | q0


def q8_multiply_bits(qx: int, qy: int) -> int:
    """3-bit multiplication in Q_8 (indices 0..7) from exact GF(2) cocycle formula:
      prod_2 = x2 ^ y2
      prod_1 = x1 ^ y1
      prod_0 = x0 ^ y0 ^ (x1 ^ x2)*y1 ^ x2*y2
    """
    if not (0 <= qx < 8 and 0 <= qy < 8):
        raise ValueError(f"Q_8 indices ({qx}, {qy}) outside 0..7")
    x0, x1, x2 = qx & 1, (qx >> 1) & 1, (qx >> 2) & 1
    y0, y1, y2 = qy & 1, (qy >> 1) & 1, (qy >> 2) & 1
    p2 = x2 ^ y2
    p1 = x1 ^ y1
    p0 = x0 ^ y0 ^ ((x1 ^ x2) & y1) ^ (x2 & y2)
    return (p2 << 2) | (p1 << 1) | p0


def semidirect_decompose(g: int) -> Tuple[int, int]:
    """Unique decomposition g = q * omega^c with c in {0, 1, 2} and q in Q_8 (0..7)."""
    if not (0 <= g < N_ELEMENTS):
        raise ValueError(f"Element index {g} outside 0..23")
    tbl = mul_table_24()
    c = coset_index_z3(g)
    wc_inv = (0, OMEGA_SQ_INDEX, OMEGA_INDEX)[c]  # omega^{-c}
    q = tbl[g][wc_inv]
    if not (0 <= q < 8):
        raise ArithmeticError(f"Right decomposition failed for g={g}: q={q}")
    return c, q


def semidirect_decompose_left(g: int) -> Tuple[int, int]:
    """Unique left decomposition g = omega^c * q_left with c in {0, 1, 2} and q_left in Q_8 (0..7)."""
    if not (0 <= g < N_ELEMENTS):
        raise ValueError(f"Element index {g} outside 0..23")
    tbl = mul_table_24()
    c = coset_index_z3(g)
    wc_inv = (0, OMEGA_SQ_INDEX, OMEGA_INDEX)[c]
    q_left = tbl[wc_inv][g]
    if not (0 <= q_left < 8):
        raise ArithmeticError(f"Left decomposition failed for g={g}: q_left={q_left}")
    return c, q_left


def semidirect_bijection_5q() -> List[int]:
    """5-qubit bijection E_sd: |g> -> |(c_g << 3) | q_g> on 0..23, fixing unused 24..31."""
    perm = list(range(N_REG_STATES))
    for g in range(N_ELEMENTS):
        c, q = semidirect_decompose(g)
        perm[g] = (c << 3) | q
    return perm


# ---------------------------------------------------------------------------
# Gate-list primitives & classical bit-level simulation (no Qiskit needed)
# ---------------------------------------------------------------------------


def _expand_ctrl_op_to_gate_tuples(
    kind: str,
    ctrls: Sequence[CtrlSpec],
    tgts: Sequence[int],
) -> List[GateTuple]:
    """Expand high-level (possibly zero-controlled) X or SWAP into pure (x, cx, ccx, cswap, mcx) tuples."""
    zero_qs = [c[0] for c in ctrls if isinstance(c, tuple) and c[1] == 0]
    ctrl_qs = tuple(c[0] if isinstance(c, tuple) else c for c in ctrls)
    flips = [("x", (), (q,)) for q in zero_qs]
    if kind == "x":
        t = tgts[0]
        k = len(ctrl_qs)
        if k == 0:
            core: List[GateTuple] = [("x", (), (t,))]
        elif k == 1:
            core = [("cx", ctrl_qs, (t,))]
        elif k == 2:
            core = [("ccx", ctrl_qs, (t,))]
        else:
            core = [("mcx", ctrl_qs, (t,))]
    elif kind == "swap":
        a, b = tgts
        if len(ctrl_qs) == 0:
            core = [("cx", (a,), (b,)), ("cx", (b,), (a,)), ("cx", (a,), (b,))]
        elif len(ctrl_qs) == 1:
            core = [("cswap", ctrl_qs, (a, b))]
        else:
            sub_ctrls = ctrl_qs + (a,)
            mid_kind = "ccx" if len(sub_ctrls) == 2 else "mcx"
            core = [("cx", (b,), (a,)), (mid_kind, sub_ctrls, (b,)), ("cx", (b,), (a,))]
    else:
        raise ValueError(f"Unsupported gate kind: {kind}")
    return flips + core + flips


def apply_gate_list_classically(gates: Sequence[GateTuple], state: int) -> int:
    """Exact classical bit-level simulation of pure computational-basis gates (x, cx, ccx, mcx, cswap)."""
    v = state
    for kind, ctrls, tgts in gates:
        if ctrls and not all((v >> c) & 1 for c in ctrls):
            continue
        if kind in ("x", "cx", "ccx", "mcx"):
            v ^= 1 << tgts[0]
        elif kind == "cswap":
            a, b = tgts
            ba, bb = (v >> a) & 1, (v >> b) & 1
            if ba != bb:
                v ^= (1 << a) | (1 << b)
        else:
            raise ValueError(f"Unknown gate kind in classical simulator: {kind}")
    return v


# ---------------------------------------------------------------------------
# Synthesis 1: Generic Disjoint-Cycle / Double-Transposition Reference
# ---------------------------------------------------------------------------


def disjoint_cycles_24(perm24: Sequence[int]) -> List[List[int]]:
    """Canonical disjoint non-trivial cycles of a bijection on {0..23}."""
    seen = [False] * N_ELEMENTS
    cycles: List[List[int]] = []
    for s in range(N_ELEMENTS):
        if seen[s] or perm24[s] == s:
            seen[s] = True
            continue
        cyc: List[int] = []
        cur = s
        while not seen[cur]:
            seen[cur] = True
            cyc.append(cur)
            cur = perm24[cur]
        cycles.append(cyc)
    return cycles


def _pair_2cycles_greedy(pairs: Sequence[Tuple[int, int]]) -> List[Tuple[int, int, int, int]]:
    rem = list(pairs)
    out: List[Tuple[int, int, int, int]] = []
    while rem:
        a, b = rem.pop(0)
        match_idx = next((i for i, (c, d) in enumerate(rem) if (a ^ b) == (c ^ d)), 0)
        c, d = rem.pop(match_idx)
        out.append((a, b, c, d))
    return out


def cycle_double_transpositions(cycles: Sequence[Sequence[int]]) -> List[Tuple[int, int, int, int]]:
    """Factor disjoint cycles of L_x into two involutions I_1 o I_2 and pair disjoint 2-cycles."""
    i2_pairs: List[Tuple[int, int]] = []
    i1_pairs: List[Tuple[int, int]] = []
    for cyc in cycles:
        length = len(cyc)
        if length == 2:
            i1_pairs.append((cyc[0], cyc[1]))
        else:
            for k in range(length // 2):
                i2_pairs.append((cyc[k], cyc[length - 1 - k]))
            for k in range(1, (length + 1) // 2):
                i1_pairs.append((cyc[k], cyc[length - k]))
    if len(i2_pairs) % 2 != 0 or len(i1_pairs) % 2 != 0:
        raise ValueError("Permutation is not an even product of disjoint 2-cycles in each involution")
    return _pair_2cycles_greedy(i2_pairs) + _pair_2cycles_greedy(i1_pairs)


def _double_transposition_conj_5q(
    a: int, b: int, c: int, d: int
) -> Tuple[List[GateTuple], int, int, List[int]]:
    """Construct 5-qubit gates (on 0..4) mapping {a, b} -> {w, w^e_p0} and {c, d} -> {w^e_p1, w^e_p0^e_p1}
    with all three remaining bits rem = [r0, r1, r2] equal to 1 in w.
    """
    ab = a ^ b
    p0 = [q for q in range(5) if (ab >> q) & 1][0]
    gates: List[GateTuple] = []

    def apply_cx(ctrl: int, tgt: int, vals: List[int]) -> List[int]:
        gates.append(("cx", (ctrl,), (tgt,)))
        return [v ^ (1 << tgt) if ((v >> ctrl) & 1) else v for v in vals]

    vals = [a, b, c, d]
    for q in range(5):
        if q != p0 and ((ab >> q) & 1):
            vals = apply_cx(p0, q, vals)
    ac = vals[0] ^ vals[2]
    p1 = [q for q in range(5) if q != p0 and ((ac >> q) & 1)][0]
    for q in range(5):
        if q != p1 and ((ac >> q) & 1):
            vals = apply_cx(p1, q, vals)

    rem = [q for q in range(5) if q not in (p0, p1)]
    if (vals[2] ^ vals[3]) != (1 << p0):
        if (vals[0] >> p1) & 1:
            gates.append(("x", (), (p1,)))
            vals = [v ^ (1 << p1) for v in vals]
        if (vals[0] >> p0) & 1:
            gates.append(("x", (), (p0,)))
            vals = [v ^ (1 << p0) for v in vals]
        diff_u3 = vals[2] ^ vals[3]
        p2 = [q for q in rem if (diff_u3 >> q) & 1][0]
        if (vals[0] >> p2) & 1:
            gates.append(("x", (), (p2,)))
            vals = [v ^ (1 << p2) for v in vals]
        extra = vals[2] ^ vals[3] ^ (1 << p2)
        for q in range(5):
            if q != p2 and ((extra >> q) & 1):
                vals = apply_cx(p2, q, vals)
        gates.append(("cswap", (p1,), (p0, p2)))
        vals[3] = vals[2] ^ (1 << p0)

    w = vals[0]
    for r in rem:
        if not ((w >> r) & 1):
            gates.append(("x", (), (r,)))
    return gates, p0, p1, rem


def multiplication_gate_list_generic() -> List[GateTuple]:
    """Generic reference 10Q gate list decomposing each L_x (x in 1..23) via disjoint cycles."""
    tbl = mul_table_24()
    gates: List[GateTuple] = []
    for x_val in range(1, N_ELEMENTS):
        zero_x = [q for q in range(N_REG_QUBITS) if not ((x_val >> q) & 1)]
        for q in zero_x:
            gates.append(("x", (), (q,)))
        cycs = disjoint_cycles_24(tbl[x_val])
        for a, b, c, d in cycle_double_transpositions(cycs):
            conj_5q, p0, p1, rem = _double_transposition_conj_5q(a, b, c, d)
            shifted = [
                (k, tuple(5 + u for u in ct), tuple(5 + v for v in tg))
                for k, ct, tg in conj_5q
            ]
            gates.extend(shifted)
            # Barenco Lemma 7.3 exact decomposition of 8-control MCX([0,1,2,3,4, r0,r1,r2] -> tgt)
            # using untouched qubit anc = 5 + p1 and qubit 0 for the 5-control half:
            anc = 5 + p1
            tgt = 5 + p0
            r0, r1, r2 = 5 + rem[0], 5 + rem[1], 5 + rem[2]
            for _ in range(2):
                gates.append(("mcx", (r2, anc, 0), (tgt,)))
                gates.append(("mcx", (4, r0, r1), (0,)))
                gates.append(("mcx", (r2, anc, 0), (tgt,)))
                gates.append(("mcx", (4, r0, r1), (0,)))
                gates.append(("mcx", (0, 1, 2, 3), (anc,)))
            gates.extend(reversed(shifted))
        for q in zero_x:
            gates.append(("x", (), (q,)))
    return gates


# ---------------------------------------------------------------------------
# Synthesis 2: Structured Semidirect-Product Circuit (2T ~= Q_8 |>< Z_3)
# ---------------------------------------------------------------------------


def _esd_high_ops(offset: int = 0) -> List[Tuple[str, List[CtrlSpec], List[int]]]:
    """High-level reversible gate sequence for 5-qubit semidirect bijection E_sd: |g> -> |c_g, q_g>."""
    q = lambda i: offset + i  # noqa: E731
    return [
        # 1. Extract Z_3 coset bits (c1, c0) into (q4, q3) via parity p = b0 ^ b1 ^ b2 and CSWAP:
        ("x", [q(1)], [q(0)]),
        ("x", [q(2)], [q(0)]),
        ("swap", [q(0)], [q(3), q(4)]),
        ("x", [q(2)], [q(0)]),
        ("x", [q(1)], [q(0)]),
        # 2. Set q4 <- c1 ^ c0 (= 1 iff Hurwitz 8..23; = 0 on both Q_8 0..7 and unused 24..31):
        ("x", [q(3)], [q(4)]),
        # 3. Controlled on q4 == 1: linear + cocycle extraction of Q_8 element (q2, q1, q0):
        ("x", [q(4), q(1)], [q(0)]),
        ("x", [q(4), q(2)], [q(1)]),
        ("swap", [q(4)], [q(1), q(2)]),
        ("swap", [q(4)], [q(0), q(1)]),
        ("x", [q(4), q(3), q(1)], [q(2)]),
        ("x", [q(4), q(3), q(2)], [q(1)]),
        ("x", [q(4), q(1)], [q(0)]),
        ("x", [q(4), q(3)], [q(0)]),
        ("x", [q(4), q(3), q(1), (q(2), 0)], [q(0)]),
        # 4. Restore q4 <- c1:
        ("x", [q(3)], [q(4)]),
    ]


def semidirect_bijection_gate_list(offset: int = 0) -> List[GateTuple]:
    """Pure computational-basis gate list for E_sd on 5 qubits starting at ``offset``."""
    out: List[GateTuple] = []
    for kind, ctrls, tgts in _esd_high_ops(offset):
        out.extend(_expand_ctrl_op_to_gate_tuples(kind, ctrls, tgts))
    return out


def semidirect_bijection_circuit():
    """5-qubit Qiskit QuantumCircuit implementing E_sd: |y> <-> |c_y, q_y> (fixing 24..31)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import C3XGate, C4XGate

    qc = QuantumCircuit(N_REG_QUBITS, name="bt2t_esd_5q")
    for kind, ctrls, tgts in semidirect_bijection_gate_list(0):
        if kind == "x":
            qc.x(tgts[0])
        elif kind == "cx":
            qc.cx(ctrls[0], tgts[0])
        elif kind == "ccx":
            qc.ccx(ctrls[0], ctrls[1], tgts[0])
        elif kind == "cswap":
            qc.cswap(ctrls[0], tgts[0], tgts[1])
        elif len(ctrls) == 3:
            qc.append(C3XGate(), list(ctrls) + [tgts[0]])
        elif len(ctrls) == 4:
            qc.append(C4XGate(), list(ctrls) + [tgts[0]])
        else:
            qc.mcx(list(ctrls), tgts[0])
    return qc


def _barenco_lemma72_ccx(
    ctrls: Sequence[int], target: int, dirty: Sequence[int]
) -> List[GateTuple]:
    """Exact Barenco Lemma 7.2 decomposition of k-control MCX into 4k-8 pure CCX gates using k-2 dirty ancillas."""
    k = len(ctrls)
    if k == 3:
        a = dirty[0]
        return [
            ("ccx", (ctrls[0], ctrls[1]), (a,)),
            ("ccx", (ctrls[2], a), (target,)),
            ("ccx", (ctrls[0], ctrls[1]), (a,)),
            ("ccx", (ctrls[2], a), (target,)),
        ]
    d = list(dirty[: k - 2])
    m = k - 2
    down: List[GateTuple] = [("ccx", (ctrls[k - 1], d[m - 1]), (target,))]
    for i in range(m - 1, 0, -1):
        down.append(("ccx", (ctrls[i + 1], d[i - 1]), (d[i],)))
    mid: List[GateTuple] = [("ccx", (ctrls[0], ctrls[1]), (d[0],))]
    up = list(reversed(down))
    down2 = down[1:]
    up2 = list(reversed(down2))
    return down + mid + up + down2 + mid + up2


def _expand_mcx_dirty_ccx_10q(ctrls: Sequence[int], target: int) -> List[GateTuple]:
    """Expand k-control MCX on 10 qubits into pure computational-basis CCX gates via Barenco Lemmas 7.2 & 7.3."""
    k = len(ctrls)
    if k == 0:
        return [("x", (), (target,))]
    if k == 1:
        return [("cx", (ctrls[0],), (target,))]
    if k == 2:
        return [("ccx", (ctrls[0], ctrls[1]), (target,))]
    used = set(ctrls) | {target}
    dirty = [q for q in range(N_MUL_QUBITS) if q not in used]
    if len(dirty) >= k - 2:
        return _barenco_lemma72_ccx(ctrls, target, dirty)
    a = dirty[0]
    half = (k + 1) // 2
    left_c, right_c = list(ctrls[:half]), list(ctrls[half:])
    part1 = _expand_mcx_dirty_ccx_10q(right_c + [a], target)
    part2 = _expand_mcx_dirty_ccx_10q(left_c, a)
    return part1 + part2 + part1 + part2


def multiplication_gate_list_structured() -> List[GateTuple]:
    """Structured 10Q gate list exploiting 2T ~= Q_8 |>< Z_3:
      (q_x omega^{c_x})(q_y omega^{c_y}) = (q_x * Ad_omega^{c_x}(q_y)) * omega^{(c_x + c_y) mod 3},
    gated off whenever x in 24..31 (c_x == 3) or y in 24..31 (c_y == 3).
    """
    high_ops: List[Tuple[str, List[CtrlSpec], List[int]]] = []

    # 1. Transform x (0..4) and y (5..9) into semidirect coordinates |c, q>:
    high_ops.extend(_esd_high_ops(0))
    high_ops.extend(_esd_high_ops(5))

    # 2a. Automorphism Ad_omega^{c_x}(q_y) on (y2, y1) = (7, 6), gated off when c_y == 3 (q9=1, q8=1).
    # Temporarily set x4 <- c_x1 ^ c_x0 so x4 == 1 iff c_x in {1, 2} (automatically 0 on c_x=0 and c_x=3):
    high_ops.append(("x", [3], [4]))
    ad_core: List[Tuple[str, List[CtrlSpec], List[int]]] = [
        ("x", [4, 3, 7], [6]),
        ("swap", [4], [6, 7]),
        ("x", [4, (3, 0), 7], [6]),
    ]
    high_ops.extend(ad_core)
    for kind, ctrls, tgts in reversed(ad_core):
        high_ops.append((kind, [9, 8] + ctrls, tgts))
    high_ops.append(("x", [3], [4]))

    # 2b. Q_8 3-bit left multiplication q_y <- q_x * q_y, gated by valid(x, y) = (1 ^ x3*x4)*(1 ^ y3*y4):
    high_ops.append(("x", [1], [2]))
    for extra in ([], [3, 4], [8, 9], [3, 4, 8, 9]):
        high_ops.append(("x", extra + [2, 6], [5]))
    high_ops.append(("x", [1], [2]))
    for base_ctrls, target in (([2, 7], 5), ([0], 5), ([1], 6), ([2], 7)):
        for extra in ([], [3, 4], [8, 9], [3, 4, 8, 9]):
            high_ops.append(("x", extra + base_ctrls, [target]))

    # 2c. Z_3 modular addition c_y <- (c_x + c_y) mod 3 (automatically fixes c_y=3 and skips c_x in {0, 3}):
    high_ops.append(("x", [3], [4]))
    high_ops.append(("x", [4, (3, 0), (9, 0)], [8]))
    high_ops.append(("swap", [4], [8, 9]))
    high_ops.append(("x", [4, 3, (9, 0)], [8]))
    high_ops.append(("x", [3], [4]))

    # 3. Uncompute E_sd^dagger on y (5..9) and x (0..4):
    high_ops.extend(reversed(_esd_high_ops(5)))
    high_ops.extend(reversed(_esd_high_ops(0)))

    gates: List[GateTuple] = []
    for kind, ctrls, tgts in high_ops:
        for gk, gc, gt in _expand_ctrl_op_to_gate_tuples(kind, ctrls, tgts):
            if gk == "mcx":
                gates.extend(_expand_mcx_dirty_ccx_10q(gc, gt[0]))
            else:
                gates.append((gk, gc, gt))
    return gates


def _build_qiskit_10q_from_gate_list(gates: Sequence[GateTuple], name: str):
    """Materialize a 10-qubit Qiskit QuantumCircuit from pure computational-basis GateTuples."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import C3XGate, C4XGate

    qc = QuantumCircuit(N_MUL_QUBITS, name=name)
    for kind, ctrls, tgts in gates:
        if kind == "x":
            qc.x(tgts[0])
        elif kind == "cx":
            qc.cx(ctrls[0], tgts[0])
        elif kind == "ccx":
            qc.ccx(ctrls[0], ctrls[1], tgts[0])
        elif kind == "cswap":
            qc.cswap(ctrls[0], tgts[0], tgts[1])
        elif kind == "mcx":
            if len(ctrls) == 3:
                qc.append(C3XGate(), list(ctrls) + [tgts[0]])
            elif len(ctrls) == 4:
                qc.append(C4XGate(), list(ctrls) + [tgts[0]])
            else:
                qc.mcx(list(ctrls), tgts[0])
        else:
            raise ValueError(f"Unsupported gate kind: {kind}")
    return qc


def multiplication_circuit_generic():
    """10-qubit generic reference Qiskit circuit for exact reversible 2T left multiplication."""
    return _build_qiskit_10q_from_gate_list(
        multiplication_gate_list_generic(), name="bt2t_mul_generic"
    )


def multiplication_circuit_structured():
    """10-qubit structured semidirect-product Qiskit circuit for exact reversible 2T left multiplication."""
    return _build_qiskit_10q_from_gate_list(
        multiplication_gate_list_structured(), name="bt2t_mul_structured"
    )
