"""OSIRIS-BT2T-001, milestone 1: exact inversion in the binary tetrahedral group 2T on one 5-qubit register.

Everything here is exact integer arithmetic or an exact permutation; nothing is fitted or sampled.

Group
-----
2T is the group of the 24 Hurwitz units: +-1, +-i, +-j, +-k and the 16 quaternions (+-1 +-i +-j +-k)/2.
An element q = (a + b i + c j + d k) / 2 is stored as its doubled integer coordinates (a, b, c, d), so every
coordinate is an integer and a^2 + b^2 + c^2 + d^2 = 4. Products are computed with the Hamilton rules on the
doubled coordinates (giving 4 q1 q2) and halved; the halving is checked to be exact.

Encoding (frozen; ENCODING_VERSION)
-----------------------------------
Index 0..7:   1, -1, i, -i, j, -j, k, -k                      (the quaternion group Q8)
Index 8..23:  (s_a 1 + s_b i + s_c j + s_d k) / 2, with signs given by the 4-bit number n = index - 8:
              bit 3 of n -> s_a, bit 2 -> s_b, bit 1 -> s_c, bit 0 -> s_d; bit value 0 means +, 1 means -.
Index 24..31: unused codewords. Inversion leaves them fixed (a valid unitary extension of the group map).

Basis and bit order
-------------------
Basis state |x>, x in 0..31, holds bit q of x on qubit q (qubit 0 is the least significant bit). This is
Qiskit's convention: a counts key "b4 b3 b2 b1 b0" reads qubit 4 first. Any other toolchain must be
converted explicitly (see ``reverse_bits``).

Inversion
---------
For a unit quaternion the inverse is the conjugate (a, -b, -c, -d). This module derives the inverse map in
two independent ways (search of the multiplication table, and conjugation) and checks they agree. The map is
an involution whose fixed points are exactly +-1 (indices 0 and 1) and the 8 unused codewords; the other 22
elements form 11 transpositions.

Circuit
-------
``inversion_circuit`` realises the 32x32 permutation as a product of the 11 disjoint transpositions. Each
transposition (a b) is C . T . C, where C is a set of CNOTs from a pivot bit p (a bit where a and b differ)
onto the other differing bits, which makes the two images differ only at p, and T is an X on p controlled on
the remaining four bits taking the images' common values. Every gate is a permutation with no phases, so the
circuit's matrix must equal the permutation matrix exactly, including phase; the tests check that.
"""

from __future__ import annotations

import hashlib
import json
from itertools import product
from typing import Dict, List, Sequence, Tuple

ENCODING_VERSION = "bt2t-enc-1"
N_QUBITS = 5
N_STATES = 1 << N_QUBITS
N_ELEMENTS = 24
UNUSED = tuple(range(N_ELEMENTS, N_STATES))

Quat = Tuple[int, int, int, int]  # doubled coordinates (a, b, c, d) of (a + b i + c j + d k) / 2


def elements() -> List[Quat]:
    """The 24 elements of 2T in frozen index order (see module docstring)."""
    q8 = [(2, 0, 0, 0), (-2, 0, 0, 0), (0, 2, 0, 0), (0, -2, 0, 0),
          (0, 0, 2, 0), (0, 0, -2, 0), (0, 0, 0, 2), (0, 0, 0, -2)]
    hurwitz = []
    for n in range(16):
        s = [(-1 if (n >> bit) & 1 else 1) for bit in (3, 2, 1, 0)]
        hurwitz.append((s[0], s[1], s[2], s[3]))
    return q8 + hurwitz


def qmul(p: Quat, q: Quat) -> Quat:
    """Product of two elements given as doubled coordinates; returns doubled coordinates of p*q."""
    a1, b1, c1, d1 = p
    a2, b2, c2, d2 = q
    raw = (a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
           a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
           a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
           a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2)  # (2p)(2q) = 4pq; the doubled coordinates of pq are raw / 2
    if any(x % 2 for x in raw):
        raise ArithmeticError(f"product of {p} and {q} is not a Hurwitz unit: {raw}")
    return tuple(x // 2 for x in raw)  # type: ignore[return-value]


def conjugate(q: Quat) -> Quat:
    return (q[0], -q[1], -q[2], -q[3])


def index_of() -> Dict[Quat, int]:
    return {g: i for i, g in enumerate(elements())}


def multiplication_table() -> List[List[int]]:
    """table[i][j] = index of element_i * element_j. Raises if the 24 elements are not closed."""
    els, idx = elements(), index_of()
    table = []
    for g in els:
        row = []
        for h in els:
            gh = qmul(g, h)
            if gh not in idx:
                raise ArithmeticError(f"{g} * {h} = {gh} is outside the encoded set")
            row.append(idx[gh])
        table.append(row)
    return table


def inverse_from_table(table: Sequence[Sequence[int]]) -> List[int]:
    """inv[i] = the unique j with table[i][j] == table[j][i] == identity (index 0)."""
    inv = []
    for i in range(N_ELEMENTS):
        js = [j for j in range(N_ELEMENTS) if table[i][j] == 0 and table[j][i] == 0]
        if len(js) != 1:
            raise ArithmeticError(f"element {i} has {len(js)} two-sided inverses")
        inv.append(js[0])
    return inv


def inversion_permutation() -> List[int]:
    """perm[x] = image of basis state x under inversion; unused codewords are fixed."""
    inv = inverse_from_table(multiplication_table())
    return inv + list(UNUSED)


def transpositions(perm: Sequence[int]) -> List[Tuple[int, int]]:
    """Disjoint transpositions of an involution, each (a, b) with a < b. Raises if perm is not an involution."""
    if any(perm[perm[x]] != x for x in range(len(perm))):
        raise ValueError("permutation is not an involution")
    return [(x, perm[x]) for x in range(len(perm)) if x < perm[x]]


def _bits(x: int) -> List[int]:
    return [(x >> q) & 1 for q in range(N_QUBITS)]


def reverse_bits(x: int, n: int = N_QUBITS) -> int:
    """Map a Qiskit (qubit 0 = LSB) basis index to the qubit-0-is-MSB convention, and back."""
    return int(format(x, f"0{n}b")[::-1], 2)


def transposition_gates(a: int, b: int) -> List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]]:
    """Gate list for the transposition of basis states a and b.

    Returns tuples (kind, controls, targets): ("cx", (p,), (q,)), ("x", (), (q,)), ("mcx", controls, (p,)).
    """
    diff = [q for q in range(N_QUBITS) if ((a ^ b) >> q) & 1]
    if not diff:
        raise ValueError("a transposition needs two different states")
    p = diff[0]
    cnots = [("cx", (p,), (q,)) for q in diff[1:]]

    def apply_c(x: int) -> int:
        for _, (c,), (t,) in cnots:
            if (x >> c) & 1:
                x ^= 1 << t
        return x

    a2, b2 = apply_c(a), apply_c(b)
    assert (a2 ^ b2) == (1 << p), (a, b, a2, b2)
    others = [q for q in range(N_QUBITS) if q != p]
    zero_controls = [q for q in others if not (a2 >> q) & 1]
    flips = [("x", (), (q,)) for q in zero_controls]
    core = flips + [("mcx", tuple(others), (p,))] + flips
    return cnots + core + cnots[::-1]


def inversion_gate_list() -> List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]]:
    gates = []
    for a, b in transpositions(inversion_permutation()):
        gates.extend(transposition_gates(a, b))
    return gates


def apply_gate_list_classically(gates, x: int) -> int:
    """Independent classical simulation of the gate list on one basis state (no quantum library)."""
    for kind, controls, (t,) in gates:
        if kind == "x" or all((x >> c) & 1 for c in controls):
            x ^= 1 << t
    return x


def inversion_circuit():
    """Qiskit circuit for the inversion permutation (needs qiskit)."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(N_QUBITS, name="bt2t_inversion")
    for kind, controls, (t,) in inversion_gate_list():
        if kind == "x":
            qc.x(t)
        elif kind == "cx":
            qc.cx(controls[0], t)
        else:
            qc.mcx(list(controls), t)
    return qc


def encoding_record() -> Dict[str, object]:
    """Frozen, hashable description of the encoding and the inversion map."""
    els = elements()
    perm = inversion_permutation()
    rec = {
        "encoding_version": ENCODING_VERSION,
        "coordinates": "doubled: (a, b, c, d) means (a + b i + c j + d k) / 2",
        "bit_order": "basis |x>: bit q of x on qubit q (qubit 0 = least significant bit)",
        "elements": [{"index": i, "doubled_coords": list(g), "bits_q4_to_q0": format(i, "05b")}
                     for i, g in enumerate(els)],
        "unused_codewords": list(UNUSED),
        "unused_behaviour": "fixed by inversion",
        "inversion_permutation": perm,
        "fixed_points": [x for x in range(N_STATES) if perm[x] == x],
        "transpositions": [list(t) for t in transpositions(perm)],
    }
    rec["sha256"] = hashlib.sha256(json.dumps(rec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return rec


def all_products_associative() -> bool:
    """Exhaustive associativity over 24^3 triples (a sanity check on qmul, not on the circuit)."""
    t = multiplication_table()
    return all(t[t[i][j]][k] == t[i][t[j][k]] for i, j, k in product(range(N_ELEMENTS), repeat=3))


def inversion_circuit_structured():
    """Same permutation as ``inversion_circuit``, built from the structure of this encoding (needs qiskit).

    Derivation (checked against the generic reference by the tests, not trusted on its own):
      * Hurwitz elements (index 8..23) have (q4, q3) in {(0, 1), (1, 0)}, i.e. q3 XOR q4 = 1, and their low
        three bits are (s_b, s_c, s_d). Conjugation negates b, c, d: X on q0, q1, q2 exactly when
        q3 XOR q4 = 1, which is CX(q3 -> t) CX(q4 -> t) for t in {0, 1, 2}.
      * Q8 elements (q3 = q4 = 0): 1 and -1 (index 0, 1) are fixed; i <-> -i, j <-> -j, k <-> -k are the pairs
        (2, 3), (4, 5), (6, 7): X on q0 when q1 OR q2. Written as X_q0[not q3, not q4] followed by
        X_q0[not q1, not q2, not q3, not q4]; the two cancel when q1 = q2 = 0.
      * Unused codewords (q3 = q4 = 1) satisfy neither condition and are fixed.
    The two parts commute: the Q8 part needs q3 = q4 = 0, where the Hurwitz part does nothing.
    """
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(N_QUBITS, name="bt2t_inversion_structured")
    for t in (0, 1, 2):
        qc.cx(3, t)
        qc.cx(4, t)
    qc.x([3, 4])
    qc.ccx(3, 4, 0)
    qc.x([1, 2])
    qc.mcx([1, 2, 3, 4], 0)
    qc.x([1, 2, 3, 4])
    return qc
