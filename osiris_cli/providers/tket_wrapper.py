"""TKET compilation wrapper and OpenQASM 3.0 / Guppy / DNA-Lang -> Quantinuum OpenQASM 2.0 lowering.

Provides deterministic lowering from OpenQASM 3.0, OpenQASM 2.0, Guppy (`@guppy`), and
minimal DNA-Lang circuit descriptions into Quantinuum-compatible OpenQASM 2.0 preserving
mid-circuit measurements, qubit resets, and classical feedforward controls. Uses ``pytket``
when installed and falls back to a deterministic static optimizer otherwise.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from .local_qccd import VIRTUAL_Z_GATES, parse_guppy_ast

QuantinuumTarget = Literal[
    "H2-1SC",
    "H2-2SC",
    "Helios-SC",
    "H2-1E",
    "H2-2E",
    "Helios-E",
    "H2-1",
    "H2-2",
    "Helios",
]

QUANTINUUM_QUBIT_CAPS: Dict[str, int] = {
    "H2-1": 56,
    "H2-1SC": 56,
    "H2-1E": 56,
    "H2-2": 56,
    "H2-2SC": 56,
    "H2-2E": 56,
    "Helios": 98,
    "Helios-SC": 98,
    "Helios-E": 98,
}

_CANONICAL_TARGETS: Dict[str, str] = {
    name.lower(): name for name in QUANTINUUM_QUBIT_CAPS
}

_SELF_INVERSE_1Q_GATES = frozenset({"x", "y", "z", "h"})
_ZERO_ANGLE_GATES = frozenset({"rx", "ry", "rz", "u1", "p"})

_QASM_HEADER_RE = re.compile(r"^OPENQASM\s+([0-9]+(?:\.[0-9]+)?)\s*$", re.IGNORECASE)
_INCLUDE_RE = re.compile(r'^include\s+"([^"]+)"\s*$')
_QASM3_QUBIT_RE = re.compile(
    r"^qubit(?:\s*\[\s*(\d+)\s*\])?\s+([A-Za-z_][A-Za-z0-9_]*)\s*$"
)
_QASM3_BIT_RE = re.compile(
    r"^bit(?:\s*\[\s*(\d+)\s*\])?\s+([A-Za-z_][A-Za-z0-9_]*)\s*$"
)
_QREG_RE = re.compile(r"^qreg\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*(\d+)\s*\]\s*$")
_CREG_RE = re.compile(r"^creg\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*(\d+)\s*\]\s*$")
_QASM3_MEASURE_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*(?:\s*\[\s*\d+\s*\])?)\s*=\s*measure\s+"
    r"([A-Za-z_][A-Za-z0-9_]*(?:\s*\[\s*\d+\s*\])?)\s*$"
)
_QASM2_MEASURE_RE = re.compile(
    r"^measure\s+([A-Za-z_][A-Za-z0-9_]*(?:\s*\[\s*\d+\s*\])?)\s*->\s*"
    r"([A-Za-z_][A-Za-z0-9_]*(?:\s*\[\s*\d+\s*\])?)\s*$"
)
_RESET_RE = re.compile(
    r"^reset\s+([A-Za-z_][A-Za-z0-9_]*(?:\s*\[\s*\d+\s*\])?)\s*$"
)
_BARRIER_RE = re.compile(r"^barrier(?:\s+(.*))?$")
_IF_RE = re.compile(r"^if\s*\(([^)]+)\)\s*(.+)$", re.DOTALL)
_GATE_CALL_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*)(?:\s*\(([^)]*)\))?\s+(.+)$"
)
_QUBIT_REF_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)(?:\s*\[\s*(\d+)\s*\])?")


@dataclass(frozen=True)
class QuantinuumCompilerOptions:
    """Configuration options controlling TKET / static optimization passes."""

    tket_opt_level: int = 2
    no_opt: bool = False
    no_reduce: bool = False
    optimization_level: Optional[int] = None
    rebase_to_qccd: bool = True
    allow_mid_circuit_meas: bool = True
    allow_classical_feedforward: bool = True

    def __post_init__(self) -> None:
        eff_level = (
            self.optimization_level
            if self.optimization_level is not None
            else self.tket_opt_level
        )
        if eff_level not in (0, 1, 2):
            raise ValueError(
                f"tket_opt_level must be 0, 1, or 2; got {eff_level}"
            )
        object.__setattr__(self, "tket_opt_level", eff_level)
        object.__setattr__(self, "optimization_level", eff_level)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CircuitMetrics:
    """Static circuit metrics computed on lowered/compiled Quantinuum QASM 2.0."""

    n_qubits: int
    n_cbits: int
    total_gates: int
    two_qubit_gates: int
    depth: int
    has_mid_circuit_measurement: bool
    has_reset: bool
    has_classical_control: bool
    single_qubit_gates: int = 0
    n_measurements: int = 0
    n_resets: int = 0
    physical_1q_gates: int = 0
    two_qubit_depth: int = 0
    mid_circuit_measurements: int = 0
    feedforward_conditionals: int = 0
    routing_swaps: int = 0

    @property
    def measurements(self) -> int:
        return self.n_measurements

    @property
    def resets(self) -> int:
        return self.n_resets

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_qubits": self.n_qubits,
            "n_cbits": self.n_cbits,
            "total_gates": self.total_gates,
            "single_qubit_gates": self.single_qubit_gates,
            "physical_1q_gates": self.physical_1q_gates,
            "two_qubit_gates": self.two_qubit_gates,
            "depth": self.depth,
            "two_qubit_depth": self.two_qubit_depth,
            "n_measurements": self.n_measurements,
            "mid_circuit_measurements": self.mid_circuit_measurements,
            "n_resets": self.n_resets,
            "feedforward_conditionals": self.feedforward_conditionals,
            "routing_swaps": self.routing_swaps,
            "has_mid_circuit_measurement": self.has_mid_circuit_measurement,
            "has_reset": self.has_reset,
            "has_classical_control": self.has_classical_control,
        }


TketCircuitMetrics = CircuitMetrics


@dataclass(frozen=True)
class TketCompilationResult:
    """Result of lowering and compiling a circuit for a Quantinuum target."""

    source_sha256: str
    compiled_sha256: str
    qasm2: str
    target: str
    compiler_backend: str
    metrics: CircuitMetrics
    diagnostics: List[str] = field(default_factory=list)
    ok: bool = True
    compiler_options: Dict[str, Any] = field(default_factory=dict)

    @property
    def compiled_qasm2(self) -> str:
        return self.qasm2

    @property
    def backend_engine(self) -> str:
        return self.compiler_backend

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "source_sha256": self.source_sha256,
            "compiled_sha256": self.compiled_sha256,
            "qasm2": self.qasm2,
            "target": self.target,
            "compiler_backend": self.compiler_backend,
            "compiler_options": dict(self.compiler_options),
            "metrics": self.metrics.to_dict(),
            "diagnostics": list(self.diagnostics),
        }


def canonicalize_quantinuum_target(target: str) -> str:
    """Normalize a Quantinuum target identifier and fail closed if unknown."""
    key = (target or "").strip().lower()
    if key not in _CANONICAL_TARGETS:
        allowed = ", ".join(sorted(QUANTINUUM_QUBIT_CAPS))
        raise ValueError(
            f"Unknown Quantinuum target {target!r}; supported targets: {allowed}"
        )
    return _CANONICAL_TARGETS[key]


def resolve_qubit_cap(target: str) -> int:
    """Return the physical qubit cap for a canonical Quantinuum target."""
    canonical = canonicalize_quantinuum_target(target)
    return QUANTINUUM_QUBIT_CAPS[canonical]


def _strip_comments(source: str) -> str:
    """Remove C-style block comments (/* ... */) and line comments (// ...)."""
    without_block = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    lines: List[str] = []
    for line in without_block.splitlines():
        idx = line.find("//")
        if idx != -1:
            line = line[:idx]
        lines.append(line)
    return "\n".join(lines)


def _split_top_level_statements(text: str) -> List[str]:
    """Split QASM source into statements while preserving brace-delimited blocks."""
    statements: List[str] = []
    buf: List[str] = []
    brace_depth = 0
    in_string = False
    escape = False

    for ch in text:
        if in_string:
            buf.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            buf.append(ch)
        elif ch == "{":
            brace_depth += 1
            buf.append(ch)
        elif ch == "}":
            brace_depth = max(0, brace_depth - 1)
            buf.append(ch)
            if brace_depth == 0:
                stmt = "".join(buf).strip()
                if stmt:
                    statements.append(stmt)
                buf = []
        elif ch == ";" and brace_depth == 0:
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
        else:
            buf.append(ch)

    trailing = "".join(buf).strip()
    if trailing:
        statements.append(trailing)
    return statements


def _normalize_ref(ref: str, scalar_regs: Sequence[str]) -> str:
    compact = re.sub(r"\s+", "", ref.strip())
    if "[" not in compact and compact in scalar_regs:
        return f"{compact}[0]"
    return compact


def _normalize_condition(cond: str, scalar_cbits: Sequence[str]) -> str:
    text = cond.strip()
    if "==" in text:
        lhs, rhs = text.split("==", 1)
        lhs_norm = _normalize_ref(lhs, scalar_cbits)
        rhs_norm = re.sub(r"\s+", "", rhs.strip())
        return f"{lhs_norm}=={rhs_norm}"
    lhs_norm = _normalize_ref(text, scalar_cbits)
    return f"{lhs_norm}==1"


def _rewrite_gate_statement(stmt: str, scalar_qregs: Sequence[str]) -> Optional[str]:
    """Normalize gate aliases and scalar register refs in a single gate call."""
    cleaned = " ".join(stmt.strip().split())
    if not cleaned:
        return None
    if cleaned.lower().startswith("gphase") or cleaned.lower().startswith("delay"):
        return None

    m_reset = _RESET_RE.match(cleaned)
    if m_reset:
        target = _normalize_ref(m_reset.group(1), scalar_qregs)
        return f"reset {target};"

    m_meas3 = _QASM3_MEASURE_RE.match(cleaned)
    if m_meas3:
        cbit = re.sub(r"\s+", "", m_meas3.group(1))
        qbit = _normalize_ref(m_meas3.group(2), scalar_qregs)
        return f"measure {qbit} -> {cbit};"

    m_meas2 = _QASM2_MEASURE_RE.match(cleaned)
    if m_meas2:
        qbit = _normalize_ref(m_meas2.group(1), scalar_qregs)
        cbit = re.sub(r"\s+", "", m_meas2.group(2))
        return f"measure {qbit} -> {cbit};"

    m_bar = _BARRIER_RE.match(cleaned)
    if m_bar:
        args_str = (m_bar.group(1) or "").strip()
        if not args_str:
            return "barrier;"
        args = [_normalize_ref(a, scalar_qregs) for a in args_str.split(",") if a.strip()]
        return f"barrier {', '.join(args)};"

    m_gate = _GATE_CALL_RE.match(cleaned)
    if not m_gate:
        return f"{cleaned};"

    gate_name = m_gate.group(1)
    params = m_gate.group(2)
    args_raw = m_gate.group(3)
    args = [_normalize_ref(a, scalar_qregs) for a in args_raw.split(",") if a.strip()]
    gate_low = gate_name.lower()

    if gate_low == "p" and params is not None:
        gate_out = f"rz({params.strip()})"
    elif gate_low == "u" and params is not None:
        gate_out = f"u3({params.strip()})"
    elif gate_low == "cp" and params is not None:
        gate_out = f"cu1({params.strip()})"
    elif gate_low == "sx" and params is None:
        gate_out = "rx(pi/2)"
    elif gate_low == "zzmax" and params is None:
        gate_out = "rzz(pi/2)"
    elif gate_low == "zzphase" and params is not None:
        gate_out = f"rzz({params.strip()})"
    elif params is not None:
        gate_out = f"{gate_name}({params.strip()})"
    else:
        gate_out = gate_name

    return f"{gate_out} {', '.join(args)};"


def _lower_guppy_to_qasm2(source: str) -> str:
    parsed = parse_guppy_ast(source)
    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{parsed.n_qubits}];",
    ]
    for cname, csz in parsed.cregs:
        lines.append(f"creg {cname}[{csz}];")
    for op in parsed.ops:
        if op.kind == "gate":
            qs = ", ".join(f"q[{q}]" for q in op.qubits)
            if op.params:
                ps = ", ".join(str(p) for p in op.params)
                lines.append(f"{op.name}({ps}) {qs};")
            else:
                lines.append(f"{op.name} {qs};")
        elif op.kind == "measure" and op.clbit is not None:
            lines.append(f"measure q[{op.qubits[0]}] -> {op.clbit[0]}[{op.clbit[1]}];")
        elif op.kind == "reset":
            lines.append(f"reset q[{op.qubits[0]}];")
        elif op.kind == "if" and op.sub_op is not None and op.cond_reg is not None:
            qs = ", ".join(f"q[{q}]" for q in op.sub_op.qubits)
            cbit_ref = (
                f"{op.cond_reg}[{op.cond_bit}]" if op.cond_bit is not None else op.cond_reg
            )
            lines.append(f"if({cbit_ref}=={op.cond_val}) {op.sub_op.name} {qs};")
    return "\n".join(lines) + "\n"


def _lower_dnalang_minimal(source: str) -> str:
    """Minimal lowering from DNA-Lang circuit statements into Quantinuum QASM 2.0."""
    cleaned = _strip_comments(source)
    body_lines: List[str] = []
    max_q = -1
    max_c = -1
    declared_q: Optional[int] = None
    declared_c: Optional[int] = None

    def _qref(tok: str) -> str:
        nonlocal max_q
        t = tok.strip()
        m_idx = re.match(r"^(?:q\[(\d+)\]|q(\d+)|(\d+))$", t, re.IGNORECASE)
        if not m_idx:
            raise ValueError(f"Invalid DNA-Lang qubit reference: {tok!r}")
        idx = int(next(g for g in m_idx.groups() if g is not None))
        if idx > max_q:
            max_q = idx
        return f"q[{idx}]"

    def _cref(tok: str) -> str:
        nonlocal max_c
        t = tok.strip()
        m_idx = re.match(r"^(?:c\[(\d+)\]|c(\d+)|(\d+))$", t, re.IGNORECASE)
        if not m_idx:
            raise ValueError(f"Invalid DNA-Lang classical bit reference: {tok!r}")
        idx = int(next(g for g in m_idx.groups() if g is not None))
        if idx > max_c:
            max_c = idx
        return f"c[{idx}]"

    raw_items = re.split(r"[\n;]+", cleaned)
    for raw in raw_items:
        line = raw.strip()
        if not line or line in ("{", "}"):
            continue
        if re.match(r"^(?:organism|genome|gene|circuit)\b", line, re.IGNORECASE):
            continue
        m_decl_q = re.match(r"^qubits?\s*(?:\[\s*(\d+)\s*\]|\s+(\d+))", line, re.IGNORECASE)
        if m_decl_q:
            declared_q = int(m_decl_q.group(1) or m_decl_q.group(2))
            continue
        m_decl_c = re.match(r"^(?:c?bits?)\s*(?:\[\s*(\d+)\s*\]|\s+(\d+))", line, re.IGNORECASE)
        if m_decl_c:
            declared_c = int(m_decl_c.group(1) or m_decl_c.group(2))
            continue

        cond_prefix = ""
        m_if = re.match(r"^IF\s*\(\s*([^)]+)\s*\)\s*(.+)$", line, re.IGNORECASE)
        if m_if:
            cond_raw = m_if.group(1).strip()
            line = m_if.group(2).strip()
            m_cond = re.match(r"^c(?:\[(\d+)\]|(\d+))\s*==\s*(\d+)$", cond_raw, re.IGNORECASE)
            if m_cond:
                cidx = int(m_cond.group(1) or m_cond.group(2))
                if cidx > max_c:
                    max_c = cidx
                cond_prefix = f"if(c[{cidx}]=={m_cond.group(3)}) "
            else:
                compact_cond = re.sub(r"\s+", "", cond_raw)
                cond_prefix = f"if({compact_cond}) "

        m_call = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)$", line)
        if not m_call:
            continue
        op = m_call.group(1).upper()
        args = [a.strip() for a in m_call.group(2).split(",") if a.strip()]

        if op in ("HELIX", "H") and len(args) == 1:
            body_lines.append(f"{cond_prefix}h {_qref(args[0])};")
        elif op in ("BOND", "CX", "CNOT") and len(args) == 2:
            body_lines.append(f"{cond_prefix}cx {_qref(args[0])}, {_qref(args[1])};")
        elif op in ("X", "Y", "Z", "S", "T") and len(args) == 1:
            body_lines.append(f"{cond_prefix}{op.lower()} {_qref(args[0])};")
        elif op in ("RX", "RY", "RZ") and len(args) == 2:
            body_lines.append(f"{cond_prefix}{op.lower()}({args[0]}) {_qref(args[1])};")
        elif op == "RZZ" and len(args) == 3:
            body_lines.append(
                f"{cond_prefix}rzz({args[0]}) {_qref(args[1])}, {_qref(args[2])};"
            )
        elif op == "RESET" and len(args) == 1:
            body_lines.append(f"{cond_prefix}reset {_qref(args[0])};")
        elif op == "MEASURE" and len(args) in (1, 2):
            q_target = _qref(args[0])
            c_target = _cref(args[1]) if len(args) == 2 else _cref(args[0])
            body_lines.append(f"{cond_prefix}measure {q_target} -> {c_target};")
        else:
            raise ValueError(f"Unsupported DNA-Lang operation: {line!r}")

    if not body_lines and declared_q is None:
        raise ValueError("Source is neither valid OpenQASM nor a recognized DNA-Lang circuit")

    n_q = max(declared_q or 0, max_q + 1)
    n_c = max(declared_c or 0, max_c + 1)
    if n_q <= 0:
        raise ValueError("DNA-Lang circuit declares 0 qubits")

    out = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{n_q}];",
    ]
    if n_c > 0:
        out.append(f"creg c[{n_c}];")
    out.extend(body_lines)
    return "\n".join(out) + "\n"


def lower_qasm3_to_quantinuum_qasm2(source: str) -> str:
    """Lower OpenQASM 3.0 / OpenQASM 2.0 / Guppy / minimal DNA-Lang to Quantinuum QASM 2.0."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Empty quantum circuit source")

    if "@guppy" in source or "guppylang" in source:
        return _lower_guppy_to_qasm2(source)

    stripped = _strip_comments(source).strip()
    if not stripped:
        raise ValueError("Quantum circuit source contains only comments/whitespace")

    if "OPENQASM" not in stripped.upper():
        return _lower_dnalang_minimal(source)

    statements = _split_top_level_statements(stripped)
    out: List[str] = ["OPENQASM 2.0;"]
    has_qelib_include = False
    scalar_qregs: List[str] = []
    scalar_cbits: List[str] = []
    body: List[str] = []

    for stmt in statements:
        stmt_clean = stmt.strip()
        if not stmt_clean:
            continue

        if _QASM_HEADER_RE.match(stmt_clean):
            continue

        m_inc = _INCLUDE_RE.match(stmt_clean)
        if m_inc:
            inc_name = m_inc.group(1).strip()
            if inc_name in ("stdgates.inc", "qelib1.inc"):
                if not has_qelib_include:
                    out.append('include "qelib1.inc";')
                    has_qelib_include = True
            else:
                out.append(f'include "{inc_name}";')
            continue

        m_q3 = _QASM3_QUBIT_RE.match(stmt_clean)
        if m_q3:
            size_str, name = m_q3.group(1), m_q3.group(2)
            size = int(size_str) if size_str is not None else 1
            if size_str is None:
                scalar_qregs.append(name)
            body.append(f"qreg {name}[{size}];")
            continue

        m_b3 = _QASM3_BIT_RE.match(stmt_clean)
        if m_b3:
            size_str, name = m_b3.group(1), m_b3.group(2)
            size = int(size_str) if size_str is not None else 1
            if size_str is None:
                scalar_cbits.append(name)
            body.append(f"creg {name}[{size}];")
            continue

        m_q2 = _QREG_RE.match(stmt_clean)
        if m_q2:
            body.append(f"qreg {m_q2.group(1)}[{int(m_q2.group(2))}];")
            continue

        m_c2 = _CREG_RE.match(stmt_clean)
        if m_c2:
            body.append(f"creg {m_c2.group(1)}[{int(m_c2.group(2))}];")
            continue

        if stmt_clean.startswith("gate ") and "{" in stmt_clean and stmt_clean.endswith("}"):
            body.append(stmt_clean)
            continue

        m_if = _IF_RE.match(stmt_clean)
        if m_if:
            cond = _normalize_condition(m_if.group(1), scalar_cbits)
            rest = m_if.group(2).strip()
            if rest.startswith("{") and rest.endswith("}"):
                inner_src = rest[1:-1].strip()
                inner_stmts = _split_top_level_statements(inner_src)
            else:
                inner_stmts = [rest]
            for inner in inner_stmts:
                lowered_inner = _rewrite_gate_statement(inner, scalar_qregs)
                if lowered_inner:
                    body.append(f"if({cond}) {lowered_inner}")
            continue

        m_meas3 = _QASM3_MEASURE_RE.match(stmt_clean)
        if m_meas3:
            cbit = _normalize_ref(m_meas3.group(1), scalar_cbits)
            qbit = _normalize_ref(m_meas3.group(2), scalar_qregs)
            body.append(f"measure {qbit} -> {cbit};")
            continue

        m_meas2 = _QASM2_MEASURE_RE.match(stmt_clean)
        if m_meas2:
            qbit = _normalize_ref(m_meas2.group(1), scalar_qregs)
            cbit = _normalize_ref(m_meas2.group(2), scalar_cbits)
            body.append(f"measure {qbit} -> {cbit};")
            continue

        lowered_stmt = _rewrite_gate_statement(stmt_clean, scalar_qregs)
        if lowered_stmt:
            body.append(lowered_stmt)

    if not has_qelib_include:
        out.append('include "qelib1.inc";')
    out.extend(body)
    return "\n".join(out) + "\n"


def _expand_qubit_arg(arg: str, qreg_sizes: Dict[str, int]) -> List[str]:
    clean = re.sub(r"\s+", "", arg.strip())
    if not clean:
        return []
    m = _QUBIT_REF_RE.fullmatch(clean)
    if not m:
        return [clean]
    reg, idx = m.group(1), m.group(2)
    if idx is not None:
        return [f"{reg}[{int(idx)}]"]
    size = qreg_sizes.get(reg, 1)
    return [f"{reg}[{i}]" for i in range(size)]


def _parse_gate_op(
    stmt: str,
) -> Optional[Tuple[bool, str, Optional[str], List[str]]]:
    """Return (is_conditional, gate_name, params, args) or None for non-gate statements."""
    s = stmt.strip().rstrip(";").strip()
    if not s:
        return None
    is_cond = False
    m_if = _IF_RE.match(s)
    if m_if:
        is_cond = True
        s = m_if.group(2).strip().rstrip(";").strip()

    if (
        _QASM_HEADER_RE.match(s)
        or _INCLUDE_RE.match(s)
        or _QREG_RE.match(s)
        or _CREG_RE.match(s)
        or _QASM2_MEASURE_RE.match(s)
        or _RESET_RE.match(s)
        or _BARRIER_RE.match(s)
        or s.startswith("gate ")
    ):
        return None

    m_gate = _GATE_CALL_RE.match(s)
    if not m_gate:
        return None
    name = m_gate.group(1).lower()
    params = m_gate.group(2).strip() if m_gate.group(2) is not None else None
    args = [re.sub(r"\s+", "", a) for a in m_gate.group(3).split(",") if a.strip()]
    return (is_cond, name, params, args)


def _is_zero_angle(params: Optional[str]) -> bool:
    if params is None:
        return False
    p = params.strip()
    try:
        return abs(float(p)) == 0.0
    except ValueError:
        return False


def _static_optimize_qasm2(
    qasm2: str,
    options: QuantinuumCompilerOptions,
) -> Tuple[str, List[str]]:
    """Deterministic static optimization pass used when pytket is unavailable."""
    if options.no_opt or options.no_reduce or options.tket_opt_level == 0:
        return qasm2, ["Static optimizer skipped due to compiler options"]

    lines = [ln.strip() for ln in qasm2.splitlines() if ln.strip()]
    out_lines: List[str] = []
    removed = 0

    for line in lines:
        parsed = _parse_gate_op(line)
        if parsed is None:
            out_lines.append(line)
            continue

        is_cond, name, params, args = parsed
        if not is_cond and name in _ZERO_ANGLE_GATES and _is_zero_angle(params):
            removed += 1
            continue

        if not is_cond and out_lines:
            prev_parsed = _parse_gate_op(out_lines[-1])
            if prev_parsed is not None:
                prev_cond, prev_name, prev_params, prev_args = prev_parsed
                if (
                    not prev_cond
                    and prev_name == name
                    and prev_params == params
                    and prev_args == args
                    and (
                        (name in _SELF_INVERSE_1Q_GATES and params is None and len(args) == 1)
                        or (name in ("cx", "cnot", "cz", "cy", "swap") and params is None and len(args) == 2)
                    )
                ):
                    out_lines.pop()
                    removed += 2
                    continue

        out_lines.append(line)

    diags = [f"static_fallback: removed {removed} redundant gate(s)"]
    return "\n".join(out_lines) + "\n", diags


def _analyze_qasm2(qasm2: str) -> CircuitMetrics:
    """Compute circuit metrics (qubits, gates, depth, mid-circuit flags) from QASM 2.0."""
    qreg_sizes: Dict[str, int] = {}
    creg_sizes: Dict[str, int] = {}
    wire_depth: Dict[str, int] = {}
    wire_2q_depth: Dict[str, int] = {}

    single_q = 0
    phys_1q = 0
    two_q = 0
    n_meas = 0
    n_reset = 0
    if_count = 0
    seen_measure = False
    has_mid_circuit_measurement = False
    has_reset = False
    has_classical_control = False
    meas_events: List[Tuple[int, List[str], str]] = []
    stmt_records: List[Tuple[str, List[str], Optional[str]]] = []

    statements = _split_top_level_statements(_strip_comments(qasm2))
    for stmt in statements:
        s = stmt.strip()
        if not s or _QASM_HEADER_RE.match(s) or _INCLUDE_RE.match(s):
            continue
        if s.startswith("gate ") and "{" in s:
            continue

        m_q = _QREG_RE.match(s)
        if m_q:
            qreg_sizes[m_q.group(1)] = int(m_q.group(2))
            continue

        m_c = _CREG_RE.match(s)
        if m_c:
            creg_sizes[m_c.group(1)] = int(m_c.group(2))
            continue

        is_cond = False
        cond_reg: Optional[str] = None
        inner = s
        m_if = _IF_RE.match(s)
        if m_if:
            is_cond = True
            if_count += 1
            has_classical_control = True
            cond_expr = m_if.group(1).strip()
            cond_reg = re.split(r"[\[\s=]", cond_expr, maxsplit=1)[0]
            inner = m_if.group(2).strip().rstrip(";").strip()
            if seen_measure:
                has_mid_circuit_measurement = True

        m_bar = _BARRIER_RE.match(inner)
        if m_bar:
            args_str = (m_bar.group(1) or "").strip()
            qubits: List[str] = []
            if args_str:
                for tok in args_str.split(","):
                    qubits.extend(_expand_qubit_arg(tok, qreg_sizes))
            else:
                for reg, sz in qreg_sizes.items():
                    qubits.extend(f"{reg}[{i}]" for i in range(sz))
            if qubits:
                sync_layer = max((wire_depth.get(q, 0) for q in qubits), default=0)
                for q in qubits:
                    wire_depth[q] = sync_layer
            continue

        m_meas = _QASM2_MEASURE_RE.match(inner)
        if m_meas:
            q_wires = _expand_qubit_arg(m_meas.group(1), qreg_sizes)
            c_reg = re.split(r"[\[\s]", m_meas.group(2).strip(), maxsplit=1)[0]
            rec_idx = len(stmt_records)
            stmt_records.append(("measure", q_wires, c_reg))
            for qw in q_wires:
                meas_events.append((rec_idx, [qw], c_reg))
            n_meas += max(1, len(q_wires))
            seen_measure = True
            for q in q_wires:
                wire_depth[q] = wire_depth.get(q, 0) + 1
            continue

        m_res = _RESET_RE.match(inner)
        if m_res:
            has_reset = True
            if seen_measure:
                has_mid_circuit_measurement = True
            q_wires = _expand_qubit_arg(m_res.group(1), qreg_sizes)
            stmt_records.append(("reset", q_wires, cond_reg))
            n_reset += max(1, len(q_wires))
            for q in q_wires:
                wire_depth[q] = wire_depth.get(q, 0) + 1
            continue

        m_gate = _GATE_CALL_RE.match(inner)
        if m_gate:
            if seen_measure:
                has_mid_circuit_measurement = True
            gname = m_gate.group(1).lower()
            raw_args = [a.strip() for a in m_gate.group(3).split(",") if a.strip()]
            expanded: List[str] = []
            for a in raw_args:
                expanded.extend(_expand_qubit_arg(a, qreg_sizes))
            stmt_records.append(("if" if is_cond else "gate", expanded, cond_reg))
            if len(raw_args) >= 2:
                two_q += 1
                l2 = max((wire_2q_depth.get(q, 0) for q in expanded), default=0) + 1
                for q in expanded:
                    wire_2q_depth[q] = l2
            else:
                single_q += 1
                if gname not in VIRTUAL_Z_GATES:
                    phys_1q += 1
            layer = max((wire_depth.get(q, 0) for q in expanded), default=0) + 1
            for q in expanded:
                wire_depth[q] = layer

    mid_meas_cnt = 0
    for m_idx, mq_list, mc_reg in meas_events:
        mq = mq_list[0]
        is_mid = False
        for kind, later_qs, later_creg in stmt_records[m_idx + 1 :]:
            if kind in ("gate", "reset", "if") and mq in later_qs:
                is_mid = True
                break
            if kind == "if" and later_creg == mc_reg:
                is_mid = True
                break
        if is_mid:
            mid_meas_cnt += 1

    n_qubits = sum(qreg_sizes.values())
    n_cbits = sum(creg_sizes.values())
    depth = max(wire_depth.values(), default=0)
    depth_2q = max(wire_2q_depth.values(), default=0)
    return CircuitMetrics(
        n_qubits=n_qubits,
        n_cbits=n_cbits,
        total_gates=single_q + two_q,
        two_qubit_gates=two_q,
        depth=depth,
        has_mid_circuit_measurement=has_mid_circuit_measurement,
        has_reset=has_reset,
        has_classical_control=has_classical_control,
        single_qubit_gates=single_q,
        n_measurements=n_meas,
        n_resets=n_reset,
        physical_1q_gates=phys_1q,
        two_qubit_depth=depth_2q,
        mid_circuit_measurements=mid_meas_cnt,
        feedforward_conditionals=if_count,
        routing_swaps=0,
    )


def _try_pytket_compile(
    qasm2: str,
    options: QuantinuumCompilerOptions,
) -> Optional[Tuple[str, List[str]]]:
    """Attempt compilation using pytket passes; return None if unavailable."""
    try:
        from pytket.passes import FullPeepholeOptimise, RemoveRedundancies
        from pytket.qasm import circuit_from_qasm_str, circuit_to_qasm_str
    except ImportError:
        return None

    diags: List[str] = []
    try:
        circ = circuit_from_qasm_str(qasm2)
    except Exception:
        return None

    if not options.no_opt:
        if not options.no_reduce and options.tket_opt_level >= 1:
            RemoveRedundancies().apply(circ)
            diags.append("Applied pytket.passes.RemoveRedundancies")
        if options.tket_opt_level >= 2:
            FullPeepholeOptimise().apply(circ)
            diags.append("Applied pytket.passes.FullPeepholeOptimise")
    else:
        diags.append("pytket optimization disabled via no_opt=True")

    compiled = circuit_to_qasm_str(circ)
    if not compiled.endswith("\n"):
        compiled += "\n"
    return compiled, diags


def compile_for_quantinuum(
    source: str,
    target: str = "H2-1SC",
    options: Optional[QuantinuumCompilerOptions] = None,
) -> TketCompilationResult:
    """Lower, validate qubit caps, and optimize a circuit for a Quantinuum target."""
    canonical_target = canonicalize_quantinuum_target(target)
    cap = resolve_qubit_cap(canonical_target)
    opts = options if options is not None else QuantinuumCompilerOptions()

    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    lowered_qasm2 = lower_qasm3_to_quantinuum_qasm2(source)
    pre_metrics = _analyze_qasm2(lowered_qasm2)

    if pre_metrics.n_qubits <= 0:
        raise ValueError("Circuit declares 0 qubits; at least 1 qubit is required")
    if pre_metrics.n_qubits > cap:
        raise ValueError(
            f"Circuit requires {pre_metrics.n_qubits} qubits, exceeding target "
            f"{canonical_target} cap of {cap} qubits"
        )

    policy_diags: List[str] = []
    if not opts.allow_mid_circuit_meas and (
        pre_metrics.has_mid_circuit_measurement or pre_metrics.has_reset
    ):
        policy_diags.append("Mid-circuit measurement/reset disabled by compiler options")
    if not opts.allow_classical_feedforward and pre_metrics.has_classical_control:
        policy_diags.append("Classical feedforward disabled by compiler options")
    if policy_diags:
        return TketCompilationResult(
            source_sha256=source_sha256,
            compiled_sha256="",
            qasm2="",
            target=canonical_target,
            compiler_backend="static_fallback",
            metrics=pre_metrics,
            diagnostics=policy_diags,
            ok=False,
            compiler_options=opts.to_dict(),
        )

    diagnostics: List[str] = []
    pytket_out = _try_pytket_compile(lowered_qasm2, opts)
    if pytket_out is not None:
        compiled_qasm2, pass_diags = pytket_out
        backend = "pytket"
        diagnostics.extend(pass_diags)
    else:
        compiled_qasm2, pass_diags = _static_optimize_qasm2(lowered_qasm2, opts)
        backend = "static_fallback"
        diagnostics.append("pytket unavailable or extended QASM2 fallback active")
        diagnostics.extend(pass_diags)

    final_metrics = _analyze_qasm2(compiled_qasm2)
    if final_metrics.n_qubits > cap:
        raise ValueError(
            f"Compiled circuit requires {final_metrics.n_qubits} qubits, exceeding "
            f"target {canonical_target} cap of {cap} qubits"
        )

    compiled_sha256 = hashlib.sha256(compiled_qasm2.encode("utf-8")).hexdigest()
    return TketCompilationResult(
        source_sha256=source_sha256,
        compiled_sha256=compiled_sha256,
        qasm2=compiled_qasm2,
        target=canonical_target,
        compiler_backend=backend,
        metrics=final_metrics,
        diagnostics=diagnostics,
        ok=True,
        compiler_options=opts.to_dict(),
    )
