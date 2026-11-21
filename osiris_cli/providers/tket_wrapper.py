"""Deterministic TKET / QCCD compilation & lowering wrapper for Quantinuum targets.

Compiles OpenQASM 2.0/3.0 and Guppy circuits into canonical OpenQASM 2.0 programs
validated against Quantinuum H2 (`H2-1`, `H2-2`, `56q`) and `Helios` (`98q`) QCCD
architectures. Uses `pytket` passes when installed, with a zero-dependency built-in
peephole optimizer and OpenQASM 2.0 emitter so compilation is 100% deterministic
across Python 3.9–3.12 offline environments.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from .local_qccd import (
    ParsedCircuit,
    ParsedOp,
    QCCDCircuitError,
    VIRTUAL_Z_GATES,
    parse_quantum_source,
)
from .quantinuum import resolve_quantinuum_target

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

_SELF_INVERSE_1Q = frozenset({"x", "y", "z", "h"})
_SELF_INVERSE_2Q = frozenset({"cx", "cnot", "cz"})


@dataclass(frozen=True)
class QuantinuumCompilerOptions:
    optimization_level: int = 2
    rebase_to_qccd: bool = True
    allow_mid_circuit_meas: bool = True
    allow_classical_feedforward: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TketCircuitMetrics:
    n_qubits: int
    n_cbits: int
    total_gates: int
    single_qubit_gates: int
    physical_1q_gates: int
    two_qubit_gates: int
    two_qubit_depth: int
    measurements: int
    mid_circuit_measurements: int
    resets: int
    feedforward_conditionals: int
    routing_swaps: int = 0


@dataclass(frozen=True)
class TketCompilationResult:
    ok: bool
    target: QuantinuumTarget
    backend_engine: str
    source_sha256: str
    compiled_qasm2: str
    compiled_sha256: str
    compiler_options: Dict[str, Any]
    metrics: TketCircuitMetrics
    diagnostics: List[str]


def _two_qubit_depth_from_ops(ops: Sequence[ParsedOp]) -> int:
    layer: Dict[int, int] = {}
    for op in ops:
        gop = op.sub_op if op.kind == "if" else (op if op.kind == "gate" else None)
        if gop is not None and len(gop.qubits) == 2:
            q0, q1 = gop.qubits
            d = max(layer.get(q0, 0), layer.get(q1, 0)) + 1
            layer[q0] = d
            layer[q1] = d
    return max(layer.values(), default=0)


def _peephole_optimize_ops(ops: Sequence[ParsedOp], level: int) -> List[ParsedOp]:
    """Deterministic adjacent self-inverse cancellation pass (`x;x`, `h;h`, `cx;cx`, etc.)."""
    if level <= 0:
        return list(ops)
    passes = 2 if level >= 2 else 1
    current = list(ops)
    for _ in range(passes):
        stack: List[ParsedOp] = []
        for op in current:
            if op.kind == "gate" and not op.params and stack:
                prev = stack[-1]
                if (
                    prev.kind == "gate"
                    and not prev.params
                    and prev.qubits == op.qubits
                    and prev.name == op.name
                    and (
                        (len(op.qubits) == 1 and op.name in _SELF_INVERSE_1Q)
                        or (len(op.qubits) == 2 and op.name in _SELF_INVERSE_2Q)
                    )
                ):
                    stack.pop()
                    continue
                if (
                    level >= 2
                    and prev.kind == "gate"
                    and prev.qubits == op.qubits
                    and (prev.name, op.name) in (("s", "sdg"), ("sdg", "s"), ("t", "tdg"), ("tdg", "t"))
                ):
                    stack.pop()
                    continue
            stack.append(op)
        current = stack
    return current


def _format_angle(rad: float) -> str:
    if abs(rad) < 1e-15:
        return "0"
    ratio = rad / math.pi
    for denom in (1, 2, 3, 4, 6, 8, 12):
        numer = round(ratio * denom)
        if numer != 0 and abs(ratio - (numer / denom)) < 1e-12:
            sign = "-" if numer < 0 else ""
            an = abs(numer)
            head = f"{sign}pi" if an == 1 else f"{sign}{an}*pi"
            return head if denom == 1 else f"{head}/{denom}"
    return f"{rad:.12g}"


def _emit_gate_qasm2(gop: ParsedOp) -> str:
    g = gop.name.lower()
    if g == "zzmax" and len(gop.qubits) == 2:
        return f"rzz(pi/2) q[{gop.qubits[0]}],q[{gop.qubits[1]}];"
    if g == "zzphase" and len(gop.qubits) == 2:
        ang = _format_angle(gop.params[0])
        return f"rzz({ang}) q[{gop.qubits[0]}],q[{gop.qubits[1]}];"
    if g == "phasedx" and len(gop.qubits) == 1:
        th = _format_angle(gop.params[0])
        ph = _format_angle(gop.params[1] if len(gop.params) > 1 else 0.0)
        return f"u3({th},{ph},-({ph})) q[{gop.qubits[0]}];"
    q_str = ",".join(f"q[{q}]" for q in gop.qubits)
    if gop.params:
        p_str = ",".join(_format_angle(p) for p in gop.params)
        return f"{g}({p_str}) {q_str};"
    return f"{g} {q_str};"


def _emit_canonical_qasm2(
    n_qubits: int,
    cregs: Sequence[Tuple[str, int]],
    ops: Sequence[ParsedOp],
) -> str:
    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{n_qubits}];",
    ]
    for cname, csz in cregs:
        lines.append(f"creg {cname}[{csz}];")
    for op in ops:
        if op.kind == "gate":
            lines.append(_emit_gate_qasm2(op))
        elif op.kind == "measure":
            assert op.clbit is not None
            lines.append(f"measure q[{op.qubits[0]}] -> {op.clbit[0]}[{op.clbit[1]}];")
        elif op.kind == "reset":
            lines.append(f"reset q[{op.qubits[0]}];")
        elif op.kind == "barrier":
            if op.qubits:
                qs = ",".join(f"q[{q}]" for q in op.qubits)
                lines.append(f"barrier {qs};")
            else:
                lines.append("barrier q;")
        elif op.kind == "if":
            assert op.cond_reg is not None and op.cond_val is not None and op.sub_op is not None
            if op.cond_bit is not None:
                # Single-bit condition on 1-bit register or indexed bit
                reg_sizes = dict(cregs)
                if reg_sizes.get(op.cond_reg, 1) == 1 and op.cond_bit == 0:
                    lines.append(f"if({op.cond_reg}=={op.cond_val}) {_emit_gate_qasm2(op.sub_op)}")
                else:
                    shifted = op.cond_val << op.cond_bit
                    lines.append(f"if({op.cond_reg}=={shifted}) {_emit_gate_qasm2(op.sub_op)}")
            else:
                lines.append(f"if({op.cond_reg}=={op.cond_val}) {_emit_gate_qasm2(op.sub_op)}")
    return "\n".join(lines) + "\n"


def _try_pytket_optimize(qasm2_str: str, opt_level: int, has_mid_or_if: bool) -> Optional[str]:
    if has_mid_or_if:
        # Preserve mid-circuit reset & classical feedforward structure intact
        return None
    try:
        from pytket.passes import FullPeepholeOptimise, RemoveRedundancies  # type: ignore[import-not-found]
        from pytket.qasm import circuit_from_qasm_str, circuit_to_qasm_str  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        circ = circuit_from_qasm_str(qasm2_str)
        if opt_level >= 2:
            FullPeepholeOptimise().apply(circ)
        elif opt_level == 1:
            RemoveRedundancies().apply(circ)
        return str(circuit_to_qasm_str(circ))
    except Exception:
        return None


def compile_for_quantinuum(
    source: str,
    *,
    target: QuantinuumTarget = "H2-1SC",
    options: Optional[QuantinuumCompilerOptions] = None,
) -> TketCompilationResult:
    """Validate, optimize, and lower `source` into canonical OpenQASM 2.0 for Quantinuum Nexus."""
    opts = options or QuantinuumCompilerOptions()
    source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    diagnostics: List[str] = []

    if opts.optimization_level not in (0, 1, 2):
        return TketCompilationResult(
            ok=False,
            target=target,
            backend_engine="osiris_qccd_lowerer",
            source_sha256=source_sha,
            compiled_qasm2="",
            compiled_sha256="",
            compiler_options=opts.to_dict(),
            metrics=TketCircuitMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            diagnostics=[f"optimization_level must be 0, 1, or 2 (got {opts.optimization_level})"],
        )

    try:
        spec = resolve_quantinuum_target(target)
    except QCCDCircuitError as exc:
        return TketCompilationResult(
            ok=False,
            target=target,
            backend_engine="osiris_qccd_lowerer",
            source_sha256=source_sha,
            compiled_qasm2="",
            compiled_sha256="",
            compiler_options=opts.to_dict(),
            metrics=TketCircuitMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            diagnostics=[str(exc)],
        )

    try:
        parsed: ParsedCircuit = parse_quantum_source(source)
    except Exception as exc:
        return TketCompilationResult(
            ok=False,
            target=target,
            backend_engine="osiris_qccd_lowerer",
            source_sha256=source_sha,
            compiled_qasm2="",
            compiled_sha256="",
            compiler_options=opts.to_dict(),
            metrics=TketCircuitMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            diagnostics=[str(exc)],
        )

    if parsed.n_qubits > spec.max_qubits:
        diagnostics.append(
            f"Circuit declares {parsed.n_qubits} qubits > {spec.name} ceiling ({spec.max_qubits})"
        )
    if not opts.allow_mid_circuit_meas and (parsed.mid_circuit_measurements > 0 or parsed.resets > 0):
        diagnostics.append("Mid-circuit measurement/reset disabled by compiler options")
    if not opts.allow_classical_feedforward and parsed.feedforward_conditionals > 0:
        diagnostics.append("Classical feedforward disabled by compiler options")

    if diagnostics:
        return TketCompilationResult(
            ok=False,
            target=target,
            backend_engine="osiris_qccd_lowerer",
            source_sha256=source_sha,
            compiled_qasm2="",
            compiled_sha256="",
            compiler_options=opts.to_dict(),
            metrics=TketCircuitMetrics(
                n_qubits=parsed.n_qubits,
                n_cbits=sum(sz for _, sz in parsed.cregs),
                total_gates=parsed.single_qubit_gates + parsed.two_qubit_gates,
                single_qubit_gates=parsed.single_qubit_gates,
                physical_1q_gates=parsed.physical_1q_gates,
                two_qubit_gates=parsed.two_qubit_gates,
                two_qubit_depth=_two_qubit_depth_from_ops(parsed.ops),
                measurements=parsed.measurements,
                mid_circuit_measurements=parsed.mid_circuit_measurements,
                resets=parsed.resets,
                feedforward_conditionals=parsed.feedforward_conditionals,
                routing_swaps=0,
            ),
            diagnostics=diagnostics,
        )

    optimized_ops = _peephole_optimize_ops(parsed.ops, opts.optimization_level)
    compiled_qasm = _emit_canonical_qasm2(parsed.n_qubits, parsed.cregs, optimized_ops)
    engine = "osiris_qccd_lowerer"

    has_dynamic = (
        parsed.mid_circuit_measurements > 0
        or parsed.resets > 0
        or parsed.feedforward_conditionals > 0
    )
    tk_qasm = _try_pytket_optimize(compiled_qasm, opts.optimization_level, has_dynamic)
    if tk_qasm is not None:
        compiled_qasm = tk_qasm
        engine = "pytket"
        reparsed = parse_quantum_source(compiled_qasm)
        final_ops = list(reparsed.ops)
        n_cbits = sum(sz for _, sz in reparsed.cregs)
        mid_m = reparsed.mid_circuit_measurements
    else:
        final_ops = optimized_ops
        n_cbits = sum(sz for _, sz in parsed.cregs)
        mid_m = parsed.mid_circuit_measurements

    s_1q = 0
    phys_1q = 0
    s_2q = 0
    meas_cnt = 0
    rst_cnt = 0
    if_cnt = 0
    for op in final_ops:
        gop = op.sub_op if op.kind == "if" else (op if op.kind == "gate" else None)
        if op.kind == "if":
            if_cnt += 1
        elif op.kind == "measure":
            meas_cnt += 1
        elif op.kind == "reset":
            rst_cnt += 1
        if gop is not None:
            if len(gop.qubits) == 1:
                s_1q += 1
                if gop.name not in VIRTUAL_Z_GATES:
                    phys_1q += 1
            elif len(gop.qubits) == 2:
                s_2q += 1

    compiled_sha = hashlib.sha256(compiled_qasm.encode("utf-8")).hexdigest()
    metrics = TketCircuitMetrics(
        n_qubits=parsed.n_qubits,
        n_cbits=n_cbits,
        total_gates=s_1q + s_2q,
        single_qubit_gates=s_1q,
        physical_1q_gates=phys_1q,
        two_qubit_gates=s_2q,
        two_qubit_depth=_two_qubit_depth_from_ops(final_ops),
        measurements=meas_cnt,
        mid_circuit_measurements=mid_m,
        resets=rst_cnt,
        feedforward_conditionals=if_cnt,
        routing_swaps=0,
    )
    return TketCompilationResult(
        ok=True,
        target=target,
        backend_engine=engine,
        source_sha256=source_sha,
        compiled_qasm2=compiled_qasm,
        compiled_sha256=compiled_sha,
        compiler_options=opts.to_dict(),
        metrics=metrics,
        diagnostics=diagnostics,
    )
