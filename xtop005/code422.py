"""[[4,2,2]] code builder for OSIRIS-XTOP-005, emitting both Qiskit and pytket circuits from one gate list.

Code (one block = physical qubits q0..q3):
  stabilizers  S_X = XXXX, S_Z = ZZZZ
  logicals     X1 = X0 X1, Z1 = Z0 Z2, X2 = X0 X2, Z2 = Z0 Z1
               (Y_i := i X_i Z_i)

Pieces (all verified in tests/test_xtop005_code422.py):
  prep_zero(block)          |00>_L = (|0000> + |1111>)/sqrt2, optional verification ancilla that
                            measures Z_2 = Z0 Z1 (+1 on |00>_L) and catches the weight-2 X fault
                            X0 X3 = X1 X2 (a logical error) a single CNOT fault can cause
  prep_plus(block)          |++>_L = H^{x4} |00>_L, optional X_2-parity verification
  transversal_cnot(a, b)    logical CNOT(a1->b1) x CNOT(a2->b2)
  logical_bell_pairs(r, m)  |Phi+>_{r1 m1} |Phi+>_{r2 m2}: prep_plus(r), prep_zero(m), transversal CNOT
  check(block, 'X'|'Z')     flagged weight-4 stabilizer measurement (1 syndrome + 1 flag ancilla)
  measure_logical(block, p1, p2)
                            destructive single-qubit readout that yields logical P1 on qubit 1 and P2
                            on qubit 2. Basis choice from an exhaustive search (READOUT). Only the
                            XX and ZZ settings also contain a stabilizer in the final readout; the other
                            seven need the mid-circuit checks for detection.

`Program` is a flat list of (gate, qubits, clbits) tuples; `to_qiskit()` and `to_pytket()` emit
native circuits from the same list, so both backends receive identical logical programs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

from qiskit.quantum_info import Pauli

LOGICAL_LABELS = {"X1": "XXII", "Z1": "ZIZI", "X2": "XIXI", "Z2": "ZZII"}  # char i acts on qubit i
STABILIZER_LABELS = {"SX": "XXXX", "SZ": "ZZZZ"}


def _pauli(label: str) -> Pauli:
    return Pauli(label[::-1])  # qiskit labels are little-endian


def logical_pauli(p: str, i: int) -> Pauli:
    """Logical P_i as a 4-qubit Pauli (Y_i = i X_i Z_i, Hermitian)."""
    if p == "X":
        return _pauli(LOGICAL_LABELS[f"X{i}"])
    if p == "Z":
        return _pauli(LOGICAL_LABELS[f"Z{i}"])
    if p == "Y":
        out = _pauli(LOGICAL_LABELS[f"X{i}"]).compose(_pauli(LOGICAL_LABELS[f"Z{i}"]))
        return Pauli((out.z, out.x, (out.phase - 1) % 4))  # multiply by i
    raise ValueError(p)


STABILIZER_GROUP = [_pauli("IIII"), _pauli("XXXX"), _pauli("ZZZZ"), _pauli("XXXX").compose(_pauli("ZZZZ"))]


def _as_basis_product(op: Pauli, basis: Sequence[str]) -> Optional[Tuple[int, Tuple[int, ...]]]:
    """If op = sign * prod_{q in support} basis[q]_q, return (sign, support); else None."""
    lab = op.to_label()
    phase = 0
    while lab and lab[0] in "-i":
        phase += 2 if lab[0] == "-" else 1
        lab = lab[1:]
    lab = lab[::-1]
    support = []
    for q, c in enumerate(lab):
        if c == "I":
            continue
        if c != basis[q]:
            return None
        support.append(q)
    if phase % 2:
        return None  # non-Hermitian representative
    return (1 if phase % 4 == 0 else -1), tuple(support)


@dataclass(frozen=True)
class Readout:
    basis: Tuple[str, ...]                     # per physical qubit
    logical1: Tuple[int, Tuple[int, ...]]      # (sign, support) giving logical P1
    logical2: Tuple[int, Tuple[int, ...]]      # (sign, support) giving logical P2
    checks: Tuple[Tuple[int, Tuple[int, ...]], ...]  # stabilizers visible in this readout (expected +1)


def _search_readouts() -> Dict[Tuple[str, str], Readout]:
    out = {}
    for p1, p2 in product("XYZ", repeat=2):
        best = None
        for basis in product("XYZ", repeat=4):
            r1 = [x for x in (_as_basis_product(s.compose(logical_pauli(p1, 1)), basis) for s in STABILIZER_GROUP) if x]
            r2 = [x for x in (_as_basis_product(s.compose(logical_pauli(p2, 2)), basis) for s in STABILIZER_GROUP) if x]
            if not r1 or not r2:
                continue
            chk = tuple(x for x in (_as_basis_product(s, basis) for s in STABILIZER_GROUP[1:]) if x)
            cand = Readout(tuple(basis), min(r1, key=lambda t: len(t[1])), min(r2, key=lambda t: len(t[1])), chk)
            key = (len(chk), -len(cand.logical1[1]) - len(cand.logical2[1]), "".join(basis))
            if best is None or key > best[0]:
                best = (key, cand)
        if best is None:
            raise RuntimeError(f"no single-qubit readout for {p1}{p2}")
        out[(p1, p2)] = best[1]
    return out


READOUT: Dict[Tuple[str, str], Readout] = _search_readouts()

Gate = Tuple[str, Tuple[int, ...], Tuple[int, ...]]


@dataclass
class Program:
    n_qubits: int
    n_clbits: int = 0
    gates: List[Gate] = field(default_factory=list)

    def add(self, name: str, qubits: Sequence[int], clbits: Sequence[int] = ()) -> None:
        self.gates.append((name, tuple(qubits), tuple(clbits)))

    def new_clbits(self, k: int) -> List[int]:
        start = self.n_clbits
        self.n_clbits += k
        return list(range(start, start + k))

    # ---- emitters -------------------------------------------------------------------------------
    def to_qiskit(self, name: str = "xtop005"):
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(self.n_qubits, self.n_clbits, name=name)
        for g, qs, cs in self.gates:
            if g == "measure":
                qc.measure(qs[0], cs[0])
            elif g == "reset":
                qc.reset(qs[0])
            elif g == "barrier":
                qc.barrier(*qs)
            else:
                getattr(qc, g)(*qs)
        return qc

    def to_pytket(self, name: str = "xtop005"):
        from pytket import Circuit, OpType
        c = Circuit(self.n_qubits, self.n_clbits, name=name)
        table = {"h": OpType.H, "x": OpType.X, "y": OpType.Y, "z": OpType.Z, "s": OpType.S, "sdg": OpType.Sdg,
                 "cx": OpType.CX, "cz": OpType.CZ, "reset": OpType.Reset}
        for g, qs, cs in self.gates:
            if g == "measure":
                c.Measure(qs[0], cs[0])
            elif g == "barrier":
                c.add_barrier(list(qs))
            else:
                c.add_gate(table[g], list(qs))
        return c

    def counts(self) -> Dict[str, int]:
        n1 = sum(1 for g, qs, _ in self.gates if g not in ("measure", "reset", "barrier") and len(qs) == 1)
        n2 = sum(1 for g, qs, _ in self.gates if g not in ("barrier",) and len(qs) == 2)
        nm = sum(1 for g, _, _ in self.gates if g == "measure")
        nr = sum(1 for g, _, _ in self.gates if g == "reset")
        return {"one_qubit": n1, "two_qubit": n2, "measure": nm, "reset": nr}


Block = Tuple[int, int, int, int]


def prep_zero(p: Program, b: Block, verify_anc: Optional[int] = None) -> Optional[int]:
    p.add("h", [b[0]])
    for t in (b[1], b[2], b[3]):
        p.add("cx", [b[0], t])
    if verify_anc is None:
        return None
    # measure Z2 = Z0 Z1 (eigenvalue +1 on |00>_L); a single CNOT fault that leaves X0 X3 flips it
    p.add("reset", [verify_anc])
    p.add("cx", [b[0], verify_anc])
    p.add("cx", [b[1], verify_anc])
    c = p.new_clbits(1)[0]
    p.add("measure", [verify_anc], [c])
    return c


def prep_plus(p: Program, b: Block, verify_anc: Optional[int] = None) -> Optional[int]:
    prep_zero(p, b, None)
    for q in b:
        p.add("h", [q])
    if verify_anc is None:
        return None
    # H^{x4} maps Z0 Z1 to X0 X1 = X1_L; measure it (eigenvalue +1 on |++>_L)
    p.add("reset", [verify_anc])
    p.add("h", [verify_anc])
    p.add("cx", [verify_anc, b[0]])
    p.add("cx", [verify_anc, b[1]])
    p.add("h", [verify_anc])
    c = p.new_clbits(1)[0]
    p.add("measure", [verify_anc], [c])
    return c


def transversal_cnot(p: Program, ctrl: Block, tgt: Block) -> None:
    for a, t in zip(ctrl, tgt):
        p.add("cx", [a, t])


def logical_bell_pairs(p: Program, r: Block, m: Block, verify_anc: Optional[int] = None) -> List[int]:
    flags = [prep_plus(p, r, verify_anc), prep_zero(p, m, verify_anc)]
    transversal_cnot(p, r, m)
    return [c for c in flags if c is not None]


def check(p: Program, b: Block, kind: str, anc: int, flag: int) -> Tuple[int, int]:
    """Flagged weight-4 stabilizer measurement. Returns (syndrome clbit, flag clbit); both 0 if clean."""
    p.add("reset", [anc])
    p.add("reset", [flag])
    if kind == "Z":
        p.add("h", [flag])
        p.add("cx", [b[0], anc])
        p.add("cx", [flag, anc])
        p.add("cx", [b[1], anc])
        p.add("cx", [b[2], anc])
        p.add("cx", [flag, anc])
        p.add("cx", [b[3], anc])
        p.add("h", [flag])
    elif kind == "X":
        p.add("h", [anc])
        p.add("cx", [anc, b[0]])
        p.add("cx", [anc, flag])
        p.add("cx", [anc, b[1]])
        p.add("cx", [anc, b[2]])
        p.add("cx", [anc, flag])
        p.add("cx", [anc, b[3]])
        p.add("h", [anc])
    else:
        raise ValueError(kind)
    cs = p.new_clbits(2)
    p.add("measure", [anc], [cs[0]])
    p.add("measure", [flag], [cs[1]])
    return cs[0], cs[1]


def measure_logical(p: Program, b: Block, p1: str, p2: str) -> List[int]:
    """Rotate each physical qubit into READOUT basis and measure; returns the 4 clbits (qubit order)."""
    ro = READOUT[(p1, p2)]
    cs = p.new_clbits(4)
    for q, basis, c in zip(b, ro.basis, cs):
        if basis == "X":
            p.add("h", [q])
        elif basis == "Y":
            p.add("sdg", [q])
            p.add("h", [q])
        p.add("measure", [q], [c])
    return cs


def decode_block(bits: Sequence[int], p1: str, p2: str) -> Dict[str, int]:
    """bits: the 4 measured values (0/1) of one block in qubit order. Returns logical eigenvalues and checks."""
    ro = READOUT[(p1, p2)]

    def ev(sign_support):
        sign, support = sign_support
        par = sum(bits[q] for q in support) % 2
        return sign * (1 - 2 * par)
    return {"L1": ev(ro.logical1), "L2": ev(ro.logical2), "check_ok": int(all(ev(c) == 1 for c in ro.checks))}
