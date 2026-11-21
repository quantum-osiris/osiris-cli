"""Deterministic OpenQASM / Guppy parser and local QCCD trajectory noise simulator.

Models trapped-ion QCCD execution (Quantinuum H2 / Helios) entirely offline using
Python stdlib + NumPy:
  - Static OpenQASM 2.0 / 3.0 and Guppy (`@guppy`) AST parsing & QCCD gate lowering
  - Mid-circuit projective measurement collapse, optical-pumping `reset`, and
    real-time classical register feedforward (`if (creg == val) <op>;`)
  - Per-shot stochastic Pauli channels for physical 1Q (`PhasedX`), 2Q (`ZZMax`/`ZZPhase`),
    ion-shuttling memory transport dephasing, reset state-prep, and SPAM readout errors
  - `G-14` raw integer shot counts + canonical SHA-256 preservation (`sum == shots`)
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..physics_model import check_sys024_clean

MAX_TRAJECTORY_QUBITS = 20

_COMMENT_LINE_RE = re.compile(r"//[^\n]*")
_COMMENT_BLOCK_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_GATE_DEF_RE = re.compile(
    r"\bgate\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(([^)]*)\))?\s+([^{]+)\{([^}]*)\}",
    re.DOTALL,
)
_QREG_RE = re.compile(r"^qreg\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*(\d+)\s*\]\s*$")
_CREG_RE = re.compile(r"^creg\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*(\d+)\s*\]\s*$")
_QUBIT3_RE = re.compile(r"^qubit(?:\s*\[\s*(\d+)\s*\])?\s+([A-Za-z_][A-Za-z0-9_]*)\s*$")
_BIT3_RE = re.compile(r"^bit(?:\s*\[\s*(\d+)\s*\])?\s+([A-Za-z_][A-Za-z0-9_]*)\s*$")
_IF_RE = re.compile(
    r"^if\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)(?:\s*\[\s*(\d+)\s*\])?\s*==\s*(\d+)\s*\)\s*(.+)$"
)
_MEAS_ARROW_RE = re.compile(r"^measure\s+(.+?)\s*->\s*(.+)$")
_MEAS_ASSIGN_RE = re.compile(r"^(.+?)\s*=\s*measure\s+(.+)$")
_RESET_RE = re.compile(r"^reset\s+(.+)$")
_BARRIER_RE = re.compile(r"^barrier(?:\s+(.+))?$")
_OP_CALL_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\s*\(([^)]*)\))?\s+(.+)$")
_INDEXED_REF_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\s*\[\s*(\d+)\s*\])?$")

VIRTUAL_Z_GATES = frozenset({"id", "i", "z", "s", "sdg", "t", "tdg", "rz", "p", "u1"})
PHYSICAL_1Q_GATES = frozenset({"x", "y", "h", "sx", "sxdg", "rx", "ry", "u2", "u3", "u", "phasedx"})
NATIVE_2Q_EQUIV_GATES = frozenset(
    {"cx", "cnot", "cz", "cy", "ch", "rxx", "ryy", "rzz", "zzmax", "zzphase", "cp", "cu1"}
)
SUPPORTED_GATES = VIRTUAL_Z_GATES | PHYSICAL_1Q_GATES | NATIVE_2Q_EQUIV_GATES | frozenset(
    {"swap", "ccx", "toffoli", "cswap", "fredkin"}
)


class QCCDCircuitError(ValueError):
    """Raised when a candidate OpenQASM or Guppy circuit fails deterministic validation."""


@dataclass(frozen=True)
class QCCDNoiseModel:
    """Physical error parameters for Quantinuum QCCD trapped-ion architectures."""

    name: str
    err_1q: float
    err_2q: float
    err_spam: float
    err_reset: float
    err_transport: float

    @classmethod
    def ideal(cls) -> "QCCDNoiseModel":
        return cls(
            name="qccd_ideal",
            err_1q=0.0,
            err_2q=0.0,
            err_spam=0.0,
            err_reset=0.0,
            err_transport=0.0,
        )

    @classmethod
    def h2_1(cls) -> "QCCDNoiseModel":
        # Published Quantinuum H2-1 benchmark regime (~99.997% 1Q, ~99.87% 2Q, ~99.85% SPAM)
        return cls(
            name="quantinuum_h2_1_qccd",
            err_1q=3.0e-5,
            err_2q=1.3e-3,
            err_spam=1.5e-3,
            err_reset=5.0e-4,
            err_transport=1.5e-4,
        )

    @classmethod
    def helios(cls) -> "QCCDNoiseModel":
        # Next-gen Quantinuum Helios barium QCCD regime
        return cls(
            name="quantinuum_helios_qccd",
            err_1q=1.5e-5,
            err_2q=6.0e-4,
            err_spam=8.0e-4,
            err_reset=3.0e-4,
            err_transport=8.0e-5,
        )


@dataclass(frozen=True)
class ParsedOp:
    kind: str  # "gate" | "measure" | "reset" | "barrier" | "if"
    name: str
    qubits: Tuple[int, ...] = ()
    params: Tuple[float, ...] = ()
    clbit: Optional[Tuple[str, int]] = None
    cond_reg: Optional[str] = None
    cond_bit: Optional[int] = None
    cond_val: Optional[int] = None
    sub_op: Optional["ParsedOp"] = None
    lineno: int = 0


@dataclass(frozen=True)
class ParsedCircuit:
    source_format: str
    n_qubits: int
    qregs: Tuple[Tuple[str, int], ...]
    cregs: Tuple[Tuple[str, int], ...]
    ops: Tuple[ParsedOp, ...]
    single_qubit_gates: int
    physical_1q_gates: int
    virtual_z_gates: int
    two_qubit_gates: int
    measurements: int
    mid_circuit_measurements: int
    resets: int
    feedforward_conditionals: int
    transport_barriers: int
    active_qubits: int
    two_qubit_edges: Tuple[Tuple[int, int], ...]
    circuit_sha256: str


@dataclass(frozen=True)
class QCCDEmulationResult:
    target: str
    noise_model: str
    shots: int
    seed: int
    raw_counts: Dict[str, int]
    raw_counts_sha256: str
    register_counts: Dict[str, Dict[str, int]]
    dominant_bitstring: str
    dominant_probability: float
    syndrome_zero_rate: Optional[float]


def _eval_angle(expr: str, env: Optional[Mapping[str, float]] = None) -> float:
    """Deterministic AST arithmetic evaluator for OpenQASM angle expressions."""
    symbols: Dict[str, float] = {"pi": math.pi, "PI": math.pi, "tau": 2.0 * math.pi}
    if env:
        symbols.update(env)
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise QCCDCircuitError(f"Invalid parameter expression {expr!r}: {exc}") from exc

    def _walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in symbols:
                return float(symbols[node.id])
            raise QCCDCircuitError(f"Unknown parameter symbol {node.id!r}")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            val = _walk(node.operand)
            return val if isinstance(node.op, ast.UAdd) else -val
        if isinstance(node, ast.BinOp):
            left = _walk(node.left)
            right = _walk(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0.0:
                    raise QCCDCircuitError("Division by zero in gate angle")
                return left / right
        raise QCCDCircuitError(f"Unsupported expression syntax in {expr!r}")

    return _walk(tree)


def _decompose_high_level_gate(
    name: str,
    qubits: Tuple[int, ...],
    params: Tuple[float, ...],
    lineno: int,
) -> List[ParsedOp]:
    """Lower 3Q / composite gates (`swap`, `ccx`, `cswap`) to 1Q + 2Q primitives."""
    g = name.lower()
    if g == "swap":
        if len(qubits) != 2:
            raise QCCDCircuitError(f"Line {lineno}: 'swap' expects 2 qubits, got {len(qubits)}")
        q0, q1 = qubits
        return [
            ParsedOp(kind="gate", name="cx", qubits=(q0, q1), lineno=lineno),
            ParsedOp(kind="gate", name="cx", qubits=(q1, q0), lineno=lineno),
            ParsedOp(kind="gate", name="cx", qubits=(q0, q1), lineno=lineno),
        ]
    if g in ("ccx", "toffoli"):
        if len(qubits) != 3:
            raise QCCDCircuitError(f"Line {lineno}: '{g}' expects 3 qubits, got {len(qubits)}")
        c0, c1, t = qubits
        return [
            ParsedOp(kind="gate", name="h", qubits=(t,), lineno=lineno),
            ParsedOp(kind="gate", name="cx", qubits=(c1, t), lineno=lineno),
            ParsedOp(kind="gate", name="tdg", qubits=(t,), lineno=lineno),
            ParsedOp(kind="gate", name="cx", qubits=(c0, t), lineno=lineno),
            ParsedOp(kind="gate", name="t", qubits=(t,), lineno=lineno),
            ParsedOp(kind="gate", name="cx", qubits=(c1, t), lineno=lineno),
            ParsedOp(kind="gate", name="tdg", qubits=(t,), lineno=lineno),
            ParsedOp(kind="gate", name="cx", qubits=(c0, t), lineno=lineno),
            ParsedOp(kind="gate", name="t", qubits=(c1,), lineno=lineno),
            ParsedOp(kind="gate", name="t", qubits=(t,), lineno=lineno),
            ParsedOp(kind="gate", name="h", qubits=(t,), lineno=lineno),
            ParsedOp(kind="gate", name="cx", qubits=(c0, c1), lineno=lineno),
            ParsedOp(kind="gate", name="t", qubits=(c0,), lineno=lineno),
            ParsedOp(kind="gate", name="tdg", qubits=(c1,), lineno=lineno),
            ParsedOp(kind="gate", name="cx", qubits=(c0, c1), lineno=lineno),
        ]
    if g in ("cswap", "fredkin"):
        if len(qubits) != 3:
            raise QCCDCircuitError(f"Line {lineno}: '{g}' expects 3 qubits, got {len(qubits)}")
        c, a, b = qubits
        out: List[ParsedOp] = [ParsedOp(kind="gate", name="cx", qubits=(b, a), lineno=lineno)]
        out.extend(_decompose_high_level_gate("ccx", (c, a, b), (), lineno))
        out.append(ParsedOp(kind="gate", name="cx", qubits=(b, a), lineno=lineno))
        return out

    expected_arity = 2 if g in NATIVE_2Q_EQUIV_GATES else 1
    if len(qubits) != expected_arity:
        raise QCCDCircuitError(
            f"Line {lineno}: gate '{g}' expects {expected_arity} qubit(s), got {len(qubits)}"
        )
    if len(set(qubits)) != len(qubits):
        raise QCCDCircuitError(f"Line {lineno}: duplicate qubit argument in gate '{g}': {qubits}")
    return [ParsedOp(kind="gate", name=g, qubits=qubits, params=params, lineno=lineno)]


def _finalize_parsed_circuit(
    source_format: str,
    raw_source: str,
    qregs: List[Tuple[str, int]],
    cregs: List[Tuple[str, int]],
    ops: List[ParsedOp],
) -> ParsedCircuit:
    n_qubits = sum(sz for _, sz in qregs)
    if n_qubits <= 0:
        raise QCCDCircuitError("Circuit declares zero qubits")

    single_q = 0
    phys_1q = 0
    virt_z = 0
    two_q = 0
    meas_total = 0
    resets = 0
    conditionals = 0
    barriers = 0
    touched: set[int] = set()
    edges: List[Tuple[int, int]] = []

    def _account_gate(op: ParsedOp) -> None:
        nonlocal single_q, phys_1q, virt_z, two_q
        touched.update(op.qubits)
        if len(op.qubits) == 1:
            single_q += 1
            if op.name in VIRTUAL_Z_GATES:
                virt_z += 1
            else:
                phys_1q += 1
        elif len(op.qubits) == 2:
            two_q += 1
            a, b = op.qubits
            edges.append((a, b))

    meas_indices: List[Tuple[int, int, str]] = []
    for idx, op in enumerate(ops):
        if op.kind == "gate":
            _account_gate(op)
        elif op.kind == "measure":
            meas_total += 1
            touched.update(op.qubits)
            assert op.clbit is not None
            meas_indices.append((idx, op.qubits[0], op.clbit[0]))
        elif op.kind == "reset":
            resets += 1
            touched.update(op.qubits)
        elif op.kind == "barrier":
            barriers += 1
        elif op.kind == "if":
            conditionals += 1
            if op.sub_op is not None:
                _account_gate(op.sub_op)

    mid_meas = 0
    for m_idx, mq, mreg in meas_indices:
        is_mid = False
        for later in ops[m_idx + 1 :]:
            if later.kind in ("gate", "reset") and mq in later.qubits:
                is_mid = True
                break
            if later.kind == "if":
                if later.cond_reg == mreg:
                    is_mid = True
                    break
                if later.sub_op is not None and mq in later.sub_op.qubits:
                    is_mid = True
                    break
        if is_mid:
            mid_meas += 1

    if not cregs:
        cregs = [("meas", max(1, meas_total or n_qubits))]

    circuit_sha = hashlib.sha256(raw_source.encode("utf-8")).hexdigest()
    return ParsedCircuit(
        source_format=source_format,
        n_qubits=n_qubits,
        qregs=tuple(qregs),
        cregs=tuple(cregs),
        ops=tuple(ops),
        single_qubit_gates=single_q,
        physical_1q_gates=phys_1q,
        virtual_z_gates=virt_z,
        two_qubit_gates=two_q,
        measurements=meas_total,
        mid_circuit_measurements=mid_meas,
        resets=resets,
        feedforward_conditionals=conditionals,
        transport_barriers=barriers,
        active_qubits=len(touched) if touched else n_qubits,
        two_qubit_edges=tuple(edges),
        circuit_sha256=circuit_sha,
    )


def parse_openqasm(source: str) -> ParsedCircuit:
    """Parse and statically validate an OpenQASM 2.0 or 3.0 circuit."""
    check_sys024_clean(source)
    stripped = _COMMENT_BLOCK_RE.sub("", source)
    stripped = _COMMENT_LINE_RE.sub("", stripped)

    macros: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...], str]] = {}

    def _record_macro(match: re.Match[str]) -> str:
        gname = match.group(1).strip()
        p_raw = (match.group(2) or "").strip()
        q_raw = match.group(3).strip()
        body = match.group(4).strip()
        p_names = tuple(x.strip() for x in p_raw.split(",") if x.strip())
        q_names = tuple(x.strip() for x in q_raw.split(",") if x.strip())
        macros[gname] = (p_names, q_names, body)
        return ""

    stripped = _GATE_DEF_RE.sub(_record_macro, stripped)
    # Normalize OpenQASM 3 single-statement `if (...) { op; }` blocks
    stripped = re.sub(
        r"if\s*\(([^)]+)\)\s*\{\s*([^;{}]+?)\s*;?\s*\}",
        r"if (\1) \2;",
        stripped,
    )

    source_format = "openqasm3" if re.search(r"\bOPENQASM\s+3", stripped) else "openqasm2"
    qreg_offsets: Dict[str, Tuple[int, int]] = {}
    qregs: List[Tuple[str, int]] = []
    cregs: List[Tuple[str, int]] = []
    creg_sizes: Dict[str, int] = {}
    total_qubits = 0
    ops: List[ParsedOp] = []

    def _resolve_qarg(token: str, lineno: int, local_qmap: Optional[Mapping[str, int]] = None) -> List[int]:
        tok = token.strip()
        if local_qmap is not None and tok in local_qmap:
            return [local_qmap[tok]]
        m = _INDEXED_REF_RE.match(tok)
        if not m:
            raise QCCDCircuitError(f"Line {lineno}: invalid qubit reference {tok!r}")
        reg_name, idx_str = m.group(1), m.group(2)
        if reg_name not in qreg_offsets:
            raise QCCDCircuitError(f"Line {lineno}: undeclared quantum register {reg_name!r}")
        base, size = qreg_offsets[reg_name]
        if idx_str is None:
            return [base + i for i in range(size)]
        idx = int(idx_str)
        if idx < 0 or idx >= size:
            raise QCCDCircuitError(
                f"Line {lineno}: qubit index {reg_name}[{idx}] out of bounds (size {size})"
            )
        return [base + idx]

    def _resolve_carg(token: str, lineno: int) -> List[Tuple[str, int]]:
        tok = token.strip()
        m = _INDEXED_REF_RE.match(tok)
        if not m:
            raise QCCDCircuitError(f"Line {lineno}: invalid classical bit reference {tok!r}")
        reg_name, idx_str = m.group(1), m.group(2)
        if reg_name not in creg_sizes:
            raise QCCDCircuitError(f"Line {lineno}: undeclared classical register {reg_name!r}")
        size = creg_sizes[reg_name]
        if idx_str is None:
            return [(reg_name, i) for i in range(size)]
        idx = int(idx_str)
        if idx < 0 or idx >= size:
            raise QCCDCircuitError(
                f"Line {lineno}: classical bit {reg_name}[{idx}] out of bounds (size {size})"
            )
        return [(reg_name, idx)]

    def _parse_gate_stmt(
        stmt: str,
        lineno: int,
        local_qmap: Optional[Mapping[str, int]] = None,
        local_pmap: Optional[Mapping[str, float]] = None,
    ) -> List[ParsedOp]:
        m = _OP_CALL_RE.match(stmt)
        if not m:
            raise QCCDCircuitError(f"Line {lineno}: unrecognized OpenQASM statement {stmt!r}")
        gname, p_str, q_str = m.group(1), m.group(2), m.group(3)
        glower = gname.lower()
        params: Tuple[float, ...] = ()
        if p_str is not None and p_str.strip():
            params = tuple(_eval_angle(p, local_pmap) for p in p_str.split(","))

        raw_qtokens = [t.strip() for t in q_str.split(",") if t.strip()]
        resolved_lists = [_resolve_qarg(t, lineno, local_qmap) for t in raw_qtokens]

        # Broadcast register arguments if needed
        lengths = {len(lst) for lst in resolved_lists}
        max_len = max(lengths) if lengths else 0
        if lengths - {1, max_len}:
            raise QCCDCircuitError(f"Line {lineno}: mismatched register widths in {stmt!r}")

        expanded_ops: List[ParsedOp] = []
        for k in range(max_len):
            qtuple = tuple(lst[0] if len(lst) == 1 else lst[k] for lst in resolved_lists)
            if gname in macros:
                p_names, q_names, body = macros[gname]
                if len(params) != len(p_names) or len(qtuple) != len(q_names):
                    raise QCCDCircuitError(f"Line {lineno}: macro '{gname}' signature mismatch")
                sub_pmap = dict(local_pmap or {})
                sub_pmap.update(zip(p_names, params))
                sub_qmap = dict(zip(q_names, qtuple))
                for sub_raw in body.split(";"):
                    sub_s = sub_raw.strip()
                    if sub_s:
                        expanded_ops.extend(_parse_gate_stmt(sub_s, lineno, sub_qmap, sub_pmap))
            elif glower in SUPPORTED_GATES:
                expanded_ops.extend(_decompose_high_level_gate(glower, qtuple, params, lineno))
            else:
                raise QCCDCircuitError(f"Line {lineno}: unsupported gate '{gname}' for QCCD target")
        return expanded_ops

    raw_statements = stripped.split(";")
    for idx, raw_stmt in enumerate(raw_statements, start=1):
        stmt = " ".join(raw_stmt.strip().split())
        if not stmt:
            continue
        if stmt.startswith("OPENQASM ") or stmt.startswith("include "):
            continue
        m_qreg = _QREG_RE.match(stmt)
        if m_qreg:
            name, sz = m_qreg.group(1), int(m_qreg.group(2))
            if sz <= 0:
                raise QCCDCircuitError(f"Statement {idx}: qreg '{name}' size must be >= 1")
            if name in qreg_offsets:
                raise QCCDCircuitError(f"Statement {idx}: duplicate qreg '{name}'")
            qreg_offsets[name] = (total_qubits, sz)
            qregs.append((name, sz))
            total_qubits += sz
            continue

        m_q3 = _QUBIT3_RE.match(stmt)
        if m_q3:
            sz = int(m_q3.group(1)) if m_q3.group(1) is not None else 1
            name = m_q3.group(2)
            if sz <= 0 or name in qreg_offsets:
                raise QCCDCircuitError(f"Statement {idx}: invalid qubit declaration '{stmt}'")
            qreg_offsets[name] = (total_qubits, sz)
            qregs.append((name, sz))
            total_qubits += sz
            continue

        m_creg = _CREG_RE.match(stmt)
        if m_creg:
            name, sz = m_creg.group(1), int(m_creg.group(2))
            if sz <= 0 or name in creg_sizes:
                raise QCCDCircuitError(f"Statement {idx}: invalid creg declaration '{stmt}'")
            creg_sizes[name] = sz
            cregs.append((name, sz))
            continue

        m_b3 = _BIT3_RE.match(stmt)
        if m_b3:
            sz = int(m_b3.group(1)) if m_b3.group(1) is not None else 1
            name = m_b3.group(2)
            if sz <= 0 or name in creg_sizes:
                raise QCCDCircuitError(f"Statement {idx}: invalid bit declaration '{stmt}'")
            creg_sizes[name] = sz
            cregs.append((name, sz))
            continue

        m_bar = _BARRIER_RE.match(stmt)
        if m_bar:
            arg_str = m_bar.group(1)
            b_qubits: List[int] = []
            if arg_str:
                for tok in arg_str.split(","):
                    if tok.strip():
                        b_qubits.extend(_resolve_qarg(tok, idx))
            ops.append(ParsedOp(kind="barrier", name="barrier", qubits=tuple(b_qubits), lineno=idx))
            continue

        m_rst = _RESET_RE.match(stmt)
        if m_rst:
            for q in _resolve_qarg(m_rst.group(1), idx):
                ops.append(ParsedOp(kind="reset", name="reset", qubits=(q,), lineno=idx))
            continue

        m_meas = _MEAS_ARROW_RE.match(stmt)
        if not m_meas:
            m_assn = _MEAS_ASSIGN_RE.match(stmt)
            if m_assn:
                q_tok, c_tok = m_assn.group(2), m_assn.group(1)
            else:
                q_tok, c_tok = "", ""
        else:
            q_tok, c_tok = m_meas.group(1), m_meas.group(2)
        if q_tok and c_tok:
            qs = _resolve_qarg(q_tok, idx)
            cs = _resolve_carg(c_tok, idx)
            if len(qs) != len(cs):
                raise QCCDCircuitError(
                    f"Statement {idx}: measure width mismatch ({len(qs)} qubits vs {len(cs)} bits)"
                )
            for q_i, c_i in zip(qs, cs):
                ops.append(
                    ParsedOp(kind="measure", name="measure", qubits=(q_i,), clbit=c_i, lineno=idx)
                )
            continue

        m_if = _IF_RE.match(stmt)
        if m_if:
            creg_name = m_if.group(1)
            cbit_str = m_if.group(2)
            cval = int(m_if.group(3))
            inner_stmt = m_if.group(4).strip()
            if creg_name not in creg_sizes:
                raise QCCDCircuitError(
                    f"Statement {idx}: undeclared classical register {creg_name!r} in feedforward"
                )
            cbit = int(cbit_str) if cbit_str is not None else None
            if cbit is not None and (cbit < 0 or cbit >= creg_sizes[creg_name]):
                raise QCCDCircuitError(
                    f"Statement {idx}: bit index {creg_name}[{cbit}] out of bounds"
                )
            sub_ops = _parse_gate_stmt(inner_stmt, idx)
            for sub in sub_ops:
                ops.append(
                    ParsedOp(
                        kind="if",
                        name="if",
                        qubits=sub.qubits,
                        cond_reg=creg_name,
                        cond_bit=cbit,
                        cond_val=cval,
                        sub_op=sub,
                        lineno=idx,
                    )
                )
            continue

        ops.extend(_parse_gate_stmt(stmt, idx))

    return _finalize_parsed_circuit(source_format, source, qregs, cregs, ops)


def parse_guppy_ast(source: str) -> ParsedCircuit:
    """Deterministically validate a Guppy (`.py` `@guppy`) circuit via Python AST."""
    check_sys024_clean(source)
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise QCCDCircuitError(f"Guppy Python syntax error: {exc}") from exc

    qubit_vars: Dict[str, int] = {}
    meas_vars: Dict[str, int] = {}
    ops: List[ParsedOp] = []

    def _alloc_qubit(var_name: str) -> int:
        if var_name not in qubit_vars:
            qubit_vars[var_name] = len(qubit_vars)
        return qubit_vars[var_name]

    def _resolve_qubit_expr(node: ast.AST, lineno: int) -> int:
        if isinstance(node, ast.Name):
            return _alloc_qubit(node.id)
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, int)
        ):
            return _alloc_qubit(f"{node.value.id}[{node.slice.value}]")
        raise QCCDCircuitError(f"Line {lineno}: unsupported Guppy qubit reference")

    def _walk_stmts(stmts: Sequence[ast.stmt], cond: Optional[Tuple[str, int]] = None) -> None:
        for st in stmts:
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.value, ast.Call):
                func_name = _call_name(st.value.func)
                tgt = st.targets[0]
                if func_name == "qubit" and isinstance(tgt, ast.Name):
                    _alloc_qubit(tgt.id)
                    continue
                if func_name in ("array", "qubit_array") and isinstance(tgt, ast.Name):
                    if len(st.value.args) >= 2 and isinstance(st.value.args[1], ast.Constant):
                        cnt = int(st.value.args[1].value)
                        for idx in range(cnt):
                            _alloc_qubit(f"{tgt.id}[{idx}]")
                        continue
                if func_name in ("measure", "project_z") and isinstance(tgt, ast.Name):
                    if not st.value.args:
                        raise QCCDCircuitError(f"Line {st.lineno}: measure() missing qubit arg")
                    q_idx = _resolve_qubit_expr(st.value.args[0], st.lineno)
                    bit_idx = len(meas_vars)
                    meas_vars[tgt.id] = bit_idx
                    ops.append(
                        ParsedOp(
                            kind="measure",
                            name="measure",
                            qubits=(q_idx,),
                            clbit=("guppy_c", bit_idx),
                            lineno=st.lineno,
                        )
                    )
                    if func_name == "project_z":
                        # project_z keeps qubit alive; no automatic discard
                        pass
                    continue
            if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call):
                call = st.value
                fname = _call_name(call.func).lower()
                if fname == "reset" and call.args:
                    q_idx = _resolve_qubit_expr(call.args[0], st.lineno)
                    ops.append(ParsedOp(kind="reset", name="reset", qubits=(q_idx,), lineno=st.lineno))
                    continue
                if fname == "measure" and call.args:
                    q_idx = _resolve_qubit_expr(call.args[0], st.lineno)
                    bit_idx = len(meas_vars)
                    meas_vars[f"_m{bit_idx}"] = bit_idx
                    ops.append(
                        ParsedOp(
                            kind="measure",
                            name="measure",
                            qubits=(q_idx,),
                            clbit=("guppy_c", bit_idx),
                            lineno=st.lineno,
                        )
                    )
                    continue
                gate_map = {"zz_max": "zzmax", "zz_phase": "zzphase"}
                gcanon = gate_map.get(fname, fname)
                if gcanon in SUPPORTED_GATES:
                    if gcanon in ("rx", "ry", "rz", "zzphase", "rzz") and len(call.args) >= 2:
                        q_args = tuple(_resolve_qubit_expr(a, st.lineno) for a in call.args[:-1])
                        ang = _eval_angle(ast.unparse(call.args[-1]))
                        lowered = _decompose_high_level_gate(gcanon, q_args, (ang,), st.lineno)
                    else:
                        q_args = tuple(_resolve_qubit_expr(a, st.lineno) for a in call.args)
                        lowered = _decompose_high_level_gate(gcanon, q_args, (), st.lineno)
                    for lop in lowered:
                        if cond is not None:
                            ops.append(
                                ParsedOp(
                                    kind="if",
                                    name="if",
                                    qubits=lop.qubits,
                                    cond_reg="guppy_c",
                                    cond_bit=cond[1],
                                    cond_val=1,
                                    sub_op=lop,
                                    lineno=st.lineno,
                                )
                            )
                        else:
                            ops.append(lop)
            if isinstance(st, ast.If):
                cond_info: Optional[Tuple[str, int]] = None
                if isinstance(st.test, ast.Name) and st.test.id in meas_vars:
                    cond_info = (st.test.id, meas_vars[st.test.id])
                _walk_stmts(st.body, cond=cond_info)
                if st.orelse:
                    _walk_stmts(st.orelse, cond=cond_info)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            _walk_stmts(node.body)

    if not qubit_vars:
        raise QCCDCircuitError("Guppy module defines zero qubits or @guppy quantum kernels")

    qregs = [("guppy_q", len(qubit_vars))]
    cregs = [("guppy_c", max(1, len(meas_vars)))]
    return _finalize_parsed_circuit("guppy", source, qregs, cregs, ops)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def parse_quantum_source(source: str, filename: str = "") -> ParsedCircuit:
    """Auto-detect OpenQASM vs. Guppy Python source and return `ParsedCircuit`."""
    trimmed = source.lstrip()
    if filename.endswith(".py") or "@guppy" in trimmed or "guppylang" in trimmed:
        return parse_guppy_ast(source)
    return parse_openqasm(source)


def _unitary_1q(name: str, params: Tuple[float, ...]) -> np.ndarray:
    g = name.lower()
    if g in ("id", "i"):
        return np.eye(2, dtype=np.complex128)
    if g == "x":
        return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    if g == "y":
        return np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    if g == "z":
        return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    if g == "h":
        inv = 1.0 / math.sqrt(2.0)
        return np.array([[inv, inv], [inv, -inv]], dtype=np.complex128)
    if g == "s":
        return np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128)
    if g == "sdg":
        return np.array([[1.0, 0.0], [0.0, -1.0j]], dtype=np.complex128)
    if g == "t":
        return np.array([[1.0, 0.0], [0.0, np.exp(1.0j * math.pi / 4.0)]], dtype=np.complex128)
    if g == "tdg":
        return np.array([[1.0, 0.0], [0.0, np.exp(-1.0j * math.pi / 4.0)]], dtype=np.complex128)
    if g == "sx":
        return 0.5 * np.array([[1.0 + 1.0j, 1.0 - 1.0j], [1.0 - 1.0j, 1.0 + 1.0j]], dtype=np.complex128)
    if g == "sxdg":
        return 0.5 * np.array([[1.0 - 1.0j, 1.0 + 1.0j], [1.0 + 1.0j, 1.0 - 1.0j]], dtype=np.complex128)
    if g == "rx":
        th = params[0]
        c, s = math.cos(th / 2.0), math.sin(th / 2.0)
        return np.array([[c, -1.0j * s], [-1.0j * s, c]], dtype=np.complex128)
    if g == "ry":
        th = params[0]
        c, s = math.cos(th / 2.0), math.sin(th / 2.0)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)
    if g == "rz":
        th = params[0]
        return np.array(
            [[np.exp(-0.5j * th), 0.0], [0.0, np.exp(0.5j * th)]], dtype=np.complex128
        )
    if g in ("p", "u1"):
        lam = params[0]
        return np.array([[1.0, 0.0], [0.0, np.exp(1.0j * lam)]], dtype=np.complex128)
    if g == "u2":
        phi, lam = params[0], params[1]
        inv = 1.0 / math.sqrt(2.0)
        return inv * np.array(
            [[1.0, -np.exp(1.0j * lam)], [np.exp(1.0j * phi), np.exp(1.0j * (phi + lam))]],
            dtype=np.complex128,
        )
    if g in ("u3", "u"):
        th, phi, lam = params[0], params[1], params[2]
        c, s = math.cos(th / 2.0), math.sin(th / 2.0)
        return np.array(
            [
                [c, -np.exp(1.0j * lam) * s],
                [np.exp(1.0j * phi) * s, np.exp(1.0j * (phi + lam)) * c],
            ],
            dtype=np.complex128,
        )
    if g == "phasedx":
        th, phi = params[0], params[1] if len(params) > 1 else 0.0
        c, s = math.cos(th / 2.0), math.sin(th / 2.0)
        return np.array(
            [[c, -1.0j * np.exp(-1.0j * phi) * s], [-1.0j * np.exp(1.0j * phi) * s, c]],
            dtype=np.complex128,
        )
    raise QCCDCircuitError(f"Unsupported 1Q gate '{name}'")


def _unitary_2q(name: str, params: Tuple[float, ...]) -> np.ndarray:
    g = name.lower()
    if g in ("cx", "cnot"):
        return np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128
        )
    if g == "cz":
        return np.diag([1.0, 1.0, 1.0, -1.0]).astype(np.complex128)
    if g == "cy":
        return np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]], dtype=np.complex128
        )
    if g == "ch":
        inv = 1.0 / math.sqrt(2.0)
        return np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, inv, inv], [0, 0, inv, -inv]],
            dtype=np.complex128,
        )
    if g in ("cp", "cu1"):
        return np.diag([1.0, 1.0, 1.0, np.exp(1.0j * params[0])]).astype(np.complex128)
    if g == "zzmax":
        return _unitary_2q("rzz", (math.pi / 2.0,))
    if g in ("rzz", "zzphase"):
        th = params[0]
        em = np.exp(-0.5j * th)
        ep = np.exp(0.5j * th)
        return np.diag([em, ep, ep, em]).astype(np.complex128)
    if g == "rxx":
        th = params[0]
        c, s = math.cos(th / 2.0), -1.0j * math.sin(th / 2.0)
        return np.array(
            [[c, 0, 0, s], [0, c, s, 0], [0, s, c, 0], [s, 0, 0, c]], dtype=np.complex128
        )
    if g == "ryy":
        th = params[0]
        c, sm, sp = math.cos(th / 2.0), 1.0j * math.sin(th / 2.0), -1.0j * math.sin(th / 2.0)
        return np.array(
            [[c, 0, 0, sm], [0, c, sp, 0], [0, sp, c, 0], [sm, 0, 0, c]], dtype=np.complex128
        )
    raise QCCDCircuitError(f"Unsupported 2Q gate '{name}'")


def _apply_1q_inplace(psi: np.ndarray, q: int, mat: np.ndarray, n: int) -> None:
    s0: List[Any] = [slice(None)] * n
    s1: List[Any] = [slice(None)] * n
    s0[q] = 0
    s1[q] = 1
    t0, t1 = tuple(s0), tuple(s1)
    a0 = psi[t0].copy()
    a1 = psi[t1].copy()
    psi[t0] = mat[0, 0] * a0 + mat[0, 1] * a1
    psi[t1] = mat[1, 0] * a0 + mat[1, 1] * a1


def _apply_pauli_inplace(psi: np.ndarray, q: int, pauli: int, n: int) -> None:
    """Apply Pauli 1=X, 2=Y, 3=Z to axis `q` in-place."""
    if pauli == 0:
        return
    s0: List[Any] = [slice(None)] * n
    s1: List[Any] = [slice(None)] * n
    s0[q] = 0
    s1[q] = 1
    t0, t1 = tuple(s0), tuple(s1)
    if pauli == 1:
        tmp = psi[t0].copy()
        psi[t0] = psi[t1]
        psi[t1] = tmp
    elif pauli == 2:
        tmp = psi[t0].copy()
        psi[t0] = -1.0j * psi[t1]
        psi[t1] = 1.0j * tmp
    elif pauli == 3:
        psi[t1] *= -1.0


def _apply_2q_inplace(psi: np.ndarray, q0: int, q1: int, name: str, mat: np.ndarray, n: int) -> None:
    if name in ("cx", "cnot"):
        c10: List[Any] = [slice(None)] * n
        c11: List[Any] = [slice(None)] * n
        c10[q0], c10[q1] = 1, 0
        c11[q0], c11[q1] = 1, 1
        t10, t11 = tuple(c10), tuple(c11)
        tmp = psi[t10].copy()
        psi[t10] = psi[t11]
        psi[t11] = tmp
        return
    if name == "cz":
        c11 = [slice(None)] * n
        c11[q0], c11[q1] = 1, 1
        psi[tuple(c11)] *= -1.0
        return

    slices: List[Tuple[Any, ...]] = []
    for b0 in (0, 1):
        for b1 in (0, 1):
            sl: List[Any] = [slice(None)] * n
            sl[q0] = b0
            sl[q1] = b1
            slices.append(tuple(sl))
    views = [psi[sl].copy() for sl in slices]
    for row in range(4):
        acc = mat[row, 0] * views[0]
        for col in range(1, 4):
            if mat[row, col] != 0.0:
                acc = acc + mat[row, col] * views[col]
        psi[slices[row]] = acc


def _project_qubit_inplace(psi: np.ndarray, q: int, n: int, rng: np.random.Generator) -> int:
    s1: List[Any] = [slice(None)] * n
    s1[q] = 1
    t1 = tuple(s1)
    v1 = psi[t1]
    p1 = float(np.vdot(v1, v1).real)
    p1 = min(1.0, max(0.0, p1))
    outcome = 1 if float(rng.random()) < p1 else 0
    s_zero_out: List[Any] = [slice(None)] * n
    s_zero_out[q] = 1 - outcome
    psi[tuple(s_zero_out)] = 0.0
    norm = math.sqrt(p1 if outcome == 1 else max(1e-30, 1.0 - p1))
    psi /= norm
    return outcome


def emulate_qccd(
    circuit: ParsedCircuit,
    target: str = "H2-1E",
    noise_model: Optional[QCCDNoiseModel] = None,
    shots: int = 1024,
    seed: int = 7,
) -> QCCDEmulationResult:
    """Execute Monte Carlo statevector trajectories with mid-circuit collapse & feedforward."""
    if shots <= 0:
        raise QCCDCircuitError(f"shots must be >= 1, got {shots}")
    if circuit.n_qubits > MAX_TRAJECTORY_QUBITS:
        raise QCCDCircuitError(
            f"Local statevector QCCD emulator supports <= {MAX_TRAJECTORY_QUBITS} qubits "
            f"(circuit uses {circuit.n_qubits})"
        )
    if noise_model is None:
        noise_model = QCCDNoiseModel.helios() if "helios" in target.lower() else QCCDNoiseModel.h2_1()

    n = circuit.n_qubits
    digest_seed = int(hashlib.sha256(f"{seed}:{circuit.circuit_sha256}".encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(digest_seed)

    # Pre-build gate unitaries once outside shot loop
    compiled_mats: Dict[int, np.ndarray] = {}
    for idx, op in enumerate(circuit.ops):
        gop = op.sub_op if op.kind == "if" else (op if op.kind == "gate" else None)
        if gop is not None:
            if len(gop.qubits) == 1:
                compiled_mats[idx] = _unitary_1q(gop.name, gop.params)
            elif len(gop.qubits) == 2:
                compiled_mats[idx] = _unitary_2q(gop.name, gop.params)

    has_explicit_meas = circuit.measurements > 0
    raw_counts: Dict[str, int] = {}
    reg_counts: Dict[str, Dict[str, int]] = {rname: {} for rname, _ in circuit.cregs}

    def _exec_gate(psi: np.ndarray, gop: ParsedOp, mat: np.ndarray) -> None:
        if len(gop.qubits) == 1:
            q = gop.qubits[0]
            _apply_1q_inplace(psi, q, mat, n)
            if gop.name not in VIRTUAL_Z_GATES and noise_model.err_1q > 0.0:
                if float(rng.random()) < noise_model.err_1q:
                    _apply_pauli_inplace(psi, q, int(rng.integers(1, 4)), n)
        elif len(gop.qubits) == 2:
            q0, q1 = gop.qubits
            _apply_2q_inplace(psi, q0, q1, gop.name, mat, n)
            if noise_model.err_2q > 0.0 and float(rng.random()) < noise_model.err_2q:
                p_pair = int(rng.integers(1, 16))
                _apply_pauli_inplace(psi, q0, p_pair // 4, n)
                _apply_pauli_inplace(psi, q1, p_pair % 4, n)
            if noise_model.err_transport > 0.0:
                if float(rng.random()) < noise_model.err_transport:
                    _apply_pauli_inplace(psi, q0, 3, n)
                if float(rng.random()) < noise_model.err_transport:
                    _apply_pauli_inplace(psi, q1, 3, n)

    for _ in range(shots):
        psi = np.zeros((2,) * n, dtype=np.complex128)
        psi[(0,) * n] = 1.0 + 0.0j
        creg_state: Dict[str, List[int]] = {rname: [0] * rsz for rname, rsz in circuit.cregs}

        for idx, op in enumerate(circuit.ops):
            if op.kind == "gate":
                _exec_gate(psi, op, compiled_mats[idx])
            elif op.kind == "measure":
                q = op.qubits[0]
                collapsed = _project_qubit_inplace(psi, q, n, rng)
                recorded = collapsed
                if noise_model.err_spam > 0.0 and float(rng.random()) < noise_model.err_spam:
                    recorded = 1 - recorded
                assert op.clbit is not None
                creg_state[op.clbit[0]][op.clbit[1]] = recorded
            elif op.kind == "reset":
                q = op.qubits[0]
                collapsed = _project_qubit_inplace(psi, q, n, rng)
                if collapsed == 1:
                    _apply_pauli_inplace(psi, q, 1, n)
                if noise_model.err_reset > 0.0 and float(rng.random()) < noise_model.err_reset:
                    _apply_pauli_inplace(psi, q, 1, n)
            elif op.kind == "if":
                assert op.cond_reg is not None and op.cond_val is not None and op.sub_op is not None
                bits = creg_state[op.cond_reg]
                if op.cond_bit is not None:
                    reg_val = bits[op.cond_bit]
                else:
                    reg_val = sum((b << bit_i) for bit_i, b in enumerate(bits))
                if reg_val == op.cond_val:
                    _exec_gate(psi, op.sub_op, compiled_mats[idx])

        if not has_explicit_meas:
            rname, rsz = circuit.cregs[0]
            for q in range(min(n, rsz)):
                collapsed = _project_qubit_inplace(psi, q, n, rng)
                if noise_model.err_spam > 0.0 and float(rng.random()) < noise_model.err_spam:
                    collapsed = 1 - collapsed
                creg_state[rname][q] = collapsed

        reg_strs: List[str] = []
        for rname, _ in reversed(circuit.cregs):
            bstr = "".join(str(b) for b in reversed(creg_state[rname]))
            reg_strs.append(bstr)
            bucket = reg_counts[rname]
            bucket[bstr] = bucket.get(bstr, 0) + 1

        full_bstr = " ".join(reg_strs)
        raw_counts[full_bstr] = raw_counts.get(full_bstr, 0) + 1

    sorted_counts = dict(sorted(raw_counts.items()))
    sorted_reg_counts = {k: dict(sorted(v.items())) for k, v in reg_counts.items()}
    counts_canon = json.dumps(sorted_counts, sort_keys=True, separators=(",", ":"))
    counts_sha = hashlib.sha256(counts_canon.encode("utf-8")).hexdigest()

    dominant_bstr, dominant_cnt = max(sorted_counts.items(), key=lambda kv: (kv[1], kv[0]))
    dom_prob = dominant_cnt / float(shots)

    syn_regs = [
        rname for rname, _ in circuit.cregs if "syn" in rname.lower() or "flag" in rname.lower()
    ]
    syn_zero: Optional[float] = None
    if syn_regs:
        last_syn = syn_regs[-1]
        zero_key = "0" * len(next(iter(sorted_reg_counts[last_syn].keys())))
        syn_zero = sorted_reg_counts[last_syn].get(zero_key, 0) / float(shots)

    return QCCDEmulationResult(
        target=target,
        noise_model=noise_model.name,
        shots=shots,
        seed=seed,
        raw_counts=sorted_counts,
        raw_counts_sha256=counts_sha,
        register_counts=sorted_reg_counts,
        dominant_bitstring=dominant_bstr,
        dominant_probability=dom_prob,
        syndrome_zero_rate=syn_zero,
    )
