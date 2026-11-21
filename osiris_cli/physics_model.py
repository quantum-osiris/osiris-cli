"""Wave 4 Physics Research Capabilities (#9 & #10):

1. Formal Model Builder (`Dimension`, `VariableSpec`, `SymmetrySpec`, `FormalModel`,
   `build_and_check_model`):
   - Exact rational SI base dimension vectors `(kg, m, s, A, K, mol, cd)` over `Fraction`.
   - Composite algebraic SI unit parser (`J`, `W`, `N`, `Pa`, `Hz`, `V`, `C`, `T`, `Ohm`,
     `eV`, `kg*m^2/s^2`, `m/s^2`, `1`, etc.).
   - Static AST algebraic & dimensional propagation across `+`, `-`, `*`, `/`, `**`,
     `sqrt`, `abs`, and dimensionless-argument transcendental functions (`exp`, `log`,
     `sin`, `cos`, `tan`, etc.).
   - Deterministic transformational symmetry / conservation checks (`time_translation`,
     `space_translation`, `parity[:vars]`, `shift:<vars>`, `scale:<var=exp,...>`,
     `exchange:<v1,v2>`).
   - `SYS-024` firewall rejecting quarantined/refuted CRSM constants (`theta_lock`,
     `51.843`, `PHI_THRESHOLD`, `0.7734`, `F_max=0.97871`, `Lambda_Phi`, `tau_phase`).

2. Prediction vs. Standard-Theory Reference Comparator (`STANDARD_REFERENCES`,
   `StandardReference`, `TheoryComparisonResult`, `compare_to_standard_theory`):
   - Authoritative CODATA / SI / QM / QED / thermodynamic reference registry with explicit
     citation provenance (`source_ref`).
   - Dimensional compatibility check & unit-scale conversion between prediction and reference.
   - Residual, relative error, combined uncertainty `sigma` / `z_score`, and tolerance
     verdict (`CONSISTENT` vs `RULED_OUT`), never treating quarantined CRSM constants as
     standard theory (`SYS-024`).
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

SI_BASE_SYMBOLS: Tuple[str, ...] = ("kg", "m", "s", "A", "K", "mol", "cd")

# SYS-024 quarantined / refuted CRSM identifiers and magic numbers banned from formal models
# and banned as "standard-theory" reference claims.
_QUARANTINED_MODEL_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("theta_lock=51.843", re.compile(r"(?:\btheta_lock\b|(?<![\d.])51\.843\d*(?![\d]))", re.IGNORECASE)),
    ("PHI_THRESHOLD=0.7734", re.compile(
        r"(?:\bphi_threshold\b|\bphi_consciousness\b|(?<![\d.])0\.7734\d*(?![\d]))", re.IGNORECASE
    )),
    ("F_max=0.97871", re.compile(r"(?:\bf_max\b|(?<![\d.])0\.97871\d*(?![\d]))", re.IGNORECASE)),
    ("Lambda_Phi", re.compile(r"\blambda_phi\b", re.IGNORECASE)),
    ("tau_phase=46.9787", re.compile(
        r"(?:\btau_phase\b|\btau_0\b|(?<![\d.])46\.9787\d*(?![\d]))", re.IGNORECASE
    )),
    ("CRSM_framework", re.compile(r"\bcrsm\b", re.IGNORECASE)),
)


class PhysicsModelError(ValueError):
    """Raised when a model specification, unit expression, or reference violates syntax or SYS-024."""


def check_sys024_clean(text: str, *, context: str = "physics model") -> Optional[str]:
    """Return a SYS-024 violation message if `text` invokes a quarantined CRSM constant, else None."""
    for label, pat in _QUARANTINED_MODEL_PATTERNS:
        if pat.search(text):
            return (
                f"SYS-024 violation in {context}: quarantined/refuted constant {label!r} "
                "is not an established physical invariant (see /legit)."
            )
    return None


@dataclass(frozen=True)
class Dimension:
    """Exact rational exponent vector over SI base units `(kg, m, s, A, K, mol, cd)`."""

    exponents: Tuple[Fraction, Fraction, Fraction, Fraction, Fraction, Fraction, Fraction] = (
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
    )

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Union[int, Fraction]]) -> Dimension:
        idx_map = {sym: idx for idx, sym in enumerate(SI_BASE_SYMBOLS)}
        exps = [Fraction(0)] * 7
        for key, val in mapping.items():
            if key not in idx_map:
                raise PhysicsModelError(f"unknown SI base symbol {key!r}")
            exps[idx_map[key]] = Fraction(val)
        return cls(tuple(exps))  # type: ignore[arg-type]

    def is_dimensionless(self) -> bool:
        return all(exp == 0 for exp in self.exponents)

    def __mul__(self, other: Dimension) -> Dimension:
        return Dimension(tuple(a + b for a, b in zip(self.exponents, other.exponents)))  # type: ignore[arg-type]

    def __truediv__(self, other: Dimension) -> Dimension:
        return Dimension(tuple(a - b for a, b in zip(self.exponents, other.exponents)))  # type: ignore[arg-type]

    def __pow__(self, power: Union[int, Fraction]) -> Dimension:
        frac = power if isinstance(power, Fraction) else Fraction(power)
        return Dimension(tuple(a * frac for a in self.exponents))  # type: ignore[arg-type]

    def format_si(self) -> str:
        parts: List[str] = []
        for sym, exp in zip(SI_BASE_SYMBOLS, self.exponents):
            if exp == 0:
                continue
            if exp == 1:
                parts.append(sym)
            elif exp.denominator == 1:
                parts.append(f"{sym}^{exp.numerator}")
            else:
                parts.append(f"{sym}^({exp.numerator}/{exp.denominator})")
        return "·".join(parts) if parts else "1"


DIM_ONE = Dimension()
DIM_MASS = Dimension.from_dict({"kg": 1})
DIM_LEN = Dimension.from_dict({"m": 1})
DIM_TIME = Dimension.from_dict({"s": 1})
DIM_CURR = Dimension.from_dict({"A": 1})
DIM_TEMP = Dimension.from_dict({"K": 1})
DIM_MOL = Dimension.from_dict({"mol": 1})
DIM_LUM = Dimension.from_dict({"cd": 1})

DIM_FREQ = DIM_TIME ** -1
DIM_VEL = DIM_LEN / DIM_TIME
DIM_ACC = DIM_LEN / (DIM_TIME ** 2)
DIM_FORCE = DIM_MASS * DIM_ACC
DIM_ENERGY = DIM_FORCE * DIM_LEN
DIM_POWER = DIM_ENERGY / DIM_TIME
DIM_PRESSURE = DIM_FORCE / (DIM_LEN ** 2)
DIM_CHARGE = DIM_CURR * DIM_TIME
DIM_VOLTAGE = DIM_POWER / DIM_CURR
DIM_RESISTANCE = DIM_VOLTAGE / DIM_CURR
DIM_CAPACITANCE = DIM_CHARGE / DIM_VOLTAGE
DIM_MAG_FLUX = DIM_VOLTAGE * DIM_TIME
DIM_MAG_FIELD = DIM_MAG_FLUX / (DIM_LEN ** 2)
DIM_INDUCTANCE = DIM_MAG_FLUX / DIM_CURR
DIM_ACTION = DIM_ENERGY * DIM_TIME

# Atomic unit table: symbol -> (Dimension, scale_to_SI_base)
_ATOMIC_UNITS: Dict[str, Tuple[Dimension, float]] = {
    "1": (DIM_ONE, 1.0),
    "dimensionless": (DIM_ONE, 1.0),
    "rad": (DIM_ONE, 1.0),
    "sr": (DIM_ONE, 1.0),
    "bit": (DIM_ONE, 1.0),
    "bits": (DIM_ONE, 1.0),
    "%": (DIM_ONE, 0.01),
    # Mass
    "kg": (DIM_MASS, 1.0),
    "g": (DIM_MASS, 1e-3),
    "mg": (DIM_MASS, 1e-6),
    "ug": (DIM_MASS, 1e-9),
    "µg": (DIM_MASS, 1e-9),
    "t": (DIM_MASS, 1e3),
    "tonne": (DIM_MASS, 1e3),
    # Length
    "m": (DIM_LEN, 1.0),
    "km": (DIM_LEN, 1e3),
    "cm": (DIM_LEN, 1e-2),
    "mm": (DIM_LEN, 1e-3),
    "um": (DIM_LEN, 1e-6),
    "µm": (DIM_LEN, 1e-6),
    "nm": (DIM_LEN, 1e-9),
    "pm": (DIM_LEN, 1e-12),
    # Time
    "s": (DIM_TIME, 1.0),
    "ms": (DIM_TIME, 1e-3),
    "us": (DIM_TIME, 1e-6),
    "µs": (DIM_TIME, 1e-6),
    "ns": (DIM_TIME, 1e-9),
    "ps": (DIM_TIME, 1e-12),
    "min": (DIM_TIME, 60.0),
    "h": (DIM_TIME, 3600.0),
    "day": (DIM_TIME, 86400.0),
    "yr": (DIM_TIME, 3.15576e7),
    # Current, Temperature, Amount, Luminous intensity
    "A": (DIM_CURR, 1.0),
    "mA": (DIM_CURR, 1e-3),
    "uA": (DIM_CURR, 1e-6),
    "µA": (DIM_CURR, 1e-6),
    "K": (DIM_TEMP, 1.0),
    "mK": (DIM_TEMP, 1e-3),
    "mol": (DIM_MOL, 1.0),
    "mmol": (DIM_MOL, 1e-3),
    "cd": (DIM_LUM, 1.0),
    # Frequency
    "Hz": (DIM_FREQ, 1.0),
    "hz": (DIM_FREQ, 1.0),
    "kHz": (DIM_FREQ, 1e3),
    "khz": (DIM_FREQ, 1e3),
    "MHz": (DIM_FREQ, 1e6),
    "mhz": (DIM_FREQ, 1e6),
    "GHz": (DIM_FREQ, 1e9),
    "ghz": (DIM_FREQ, 1e9),
    # Force, Energy, Power, Pressure
    "N": (DIM_FORCE, 1.0),
    "kN": (DIM_FORCE, 1e3),
    "MN": (DIM_FORCE, 1e6),
    "J": (DIM_ENERGY, 1.0),
    "kJ": (DIM_ENERGY, 1e3),
    "MJ": (DIM_ENERGY, 1e6),
    "eV": (DIM_ENERGY, 1.602176634e-19),
    "ev": (DIM_ENERGY, 1.602176634e-19),
    "keV": (DIM_ENERGY, 1.602176634e-16),
    "MeV": (DIM_ENERGY, 1.602176634e-13),
    "GeV": (DIM_ENERGY, 1.602176634e-10),
    "W": (DIM_POWER, 1.0),
    "mW": (DIM_POWER, 1e-3),
    "kW": (DIM_POWER, 1e3),
    "MW": (DIM_POWER, 1e6),
    "GW": (DIM_POWER, 1e9),
    "Pa": (DIM_PRESSURE, 1.0),
    "kPa": (DIM_PRESSURE, 1e3),
    "MPa": (DIM_PRESSURE, 1e6),
    "GPa": (DIM_PRESSURE, 1e9),
    # Electromagnetism
    "C": (DIM_CHARGE, 1.0),
    "V": (DIM_VOLTAGE, 1.0),
    "mV": (DIM_VOLTAGE, 1e-3),
    "kV": (DIM_VOLTAGE, 1e3),
    "Ohm": (DIM_RESISTANCE, 1.0),
    "ohm": (DIM_RESISTANCE, 1.0),
    "Ω": (DIM_RESISTANCE, 1.0),
    "F": (DIM_CAPACITANCE, 1.0),
    "Wb": (DIM_MAG_FLUX, 1.0),
    "T": (DIM_MAG_FIELD, 1.0),
    "mT": (DIM_MAG_FIELD, 1e-3),
    "H": (DIM_INDUCTANCE, 1.0),
}


def _eval_rational_exponent(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return Fraction(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        inner = _eval_rational_exponent(node.operand)
        return inner if isinstance(node.op, ast.UAdd) else -inner
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        num = _eval_rational_exponent(node.left)
        den = _eval_rational_exponent(node.right)
        if den == 0:
            raise PhysicsModelError("zero denominator in unit exponent")
        return num / den
    raise PhysicsModelError("unit exponent must be an integer or rational fraction (e.g. ^2, ^-1, ^(1/2))")


def _eval_unit_ast(node: ast.AST) -> Tuple[Dimension, float]:
    if isinstance(node, ast.Constant) and node.value == 1:
        return DIM_ONE, 1.0
    if isinstance(node, ast.Name):
        if node.id not in _ATOMIC_UNITS:
            raise PhysicsModelError(f"unknown unit symbol {node.id!r}")
        return _ATOMIC_UNITS[node.id]
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Mult):
            d1, s1 = _eval_unit_ast(node.left)
            d2, s2 = _eval_unit_ast(node.right)
            return d1 * d2, s1 * s2
        if isinstance(node.op, ast.Div):
            d1, s1 = _eval_unit_ast(node.left)
            d2, s2 = _eval_unit_ast(node.right)
            return d1 / d2, s1 / s2
        if isinstance(node.op, ast.Pow):
            base_dim, base_scale = _eval_unit_ast(node.left)
            exp = _eval_rational_exponent(node.right)
            return base_dim ** exp, base_scale ** float(exp)
    raise PhysicsModelError("unsupported unit syntax; use *, /, ^ or ** (e.g. 'kg*m^2/s^2')")


def parse_unit_dimension(unit_text: str) -> Tuple[Dimension, float]:
    """Parse a unit string (e.g. `'J'`, `'kg*m^2/s^2'`, `'m/s'`, `'1'`) -> `(Dimension, scale_to_SI)`."""
    clean = unit_text.strip()
    if not clean or clean in ("1", "dimensionless"):
        return DIM_ONE, 1.0
    if clean in _ATOMIC_UNITS:
        return _ATOMIC_UNITS[clean]
    normalized = clean.replace("·", "*").replace("^", "**")
    # Support space-separated products like "kg m / s**2" when unambiguous
    normalized = re.sub(r"(?<=[A-Za-z0-9)\]])\s+(?=[A-Za-z(])", "*", normalized)
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise PhysicsModelError(f"invalid unit expression {unit_text!r}: {exc.msg}") from exc
    return _eval_unit_ast(tree.body)


# ── Capability #9: Formal Model Builder ──────────────────────────────────────


@dataclass(frozen=True)
class VariableSpec:
    name: str
    unit: str
    dimension: Dimension
    si_scale: float = 1.0
    description: str = ""


@dataclass(frozen=True)
class SymmetrySpec:
    raw: str
    kind: str  # time_translation | space_translation | parity | shift | scale | exchange
    targets: Tuple[str, ...]
    exponents: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class EquationReport:
    raw: str
    lhs_dim: Dimension
    rhs_dim: Dimension
    consistent: bool
    detail: str


@dataclass(frozen=True)
class SymmetryReport:
    raw: str
    conserved_law: str
    invariant: bool
    detail: str


@dataclass(frozen=True)
class FormalModelResult:
    verdict: str  # CONSISTENT | RULED_OUT
    variables: Dict[str, VariableSpec]
    equations: Tuple[EquationReport, ...]
    symmetries: Tuple[SymmetryReport, ...]
    violations: Tuple[str, ...]
    summary: str


_TRANSCENDENTAL_FUNCS: Dict[str, Callable[[float], float]] = {
    "exp": math.exp,
    "log": math.log,
    "ln": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
}

_BUILTIN_DIMENSIONLESS: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}

_VALID_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_variable_specs(
    spec_input: Union[str, Mapping[str, str], Sequence[Tuple[str, str]]]
) -> Dict[str, VariableSpec]:
    """Parse variable declarations like `'E:J, m:kg, c:m/s'` or `{'E': 'J', 'm': 'kg'}`."""
    pairs: List[Tuple[str, str]] = []
    if isinstance(spec_input, str):
        for token in [t.strip() for t in spec_input.split(",") if t.strip()]:
            if ":" not in token:
                raise PhysicsModelError(f"expected 'var:unit' in variable declaration, got {token!r}")
            vname, ustr = token.split(":", 1)
            pairs.append((vname.strip(), ustr.strip()))
    elif isinstance(spec_input, Mapping):
        for k, v in spec_input.items():
            pairs.append((str(k).strip(), str(v).strip()))
    else:
        for k, v in spec_input:
            pairs.append((str(k).strip(), str(v).strip()))

    if not pairs:
        raise PhysicsModelError("formal model requires at least one declared variable")

    out: Dict[str, VariableSpec] = {}
    for name, unit in pairs:
        sys024 = check_sys024_clean(name, context=f"variable {name!r}")
        if sys024:
            raise PhysicsModelError(sys024)
        if not _VALID_IDENT.match(name):
            raise PhysicsModelError(f"invalid variable identifier {name!r}")
        if name in _TRANSCENDENTAL_FUNCS or name in ("sqrt", "abs"):
            raise PhysicsModelError(f"variable name {name!r} collides with reserved mathematical function")
        dim, scale = parse_unit_dimension(unit)
        out[name] = VariableSpec(name=name, unit=unit, dimension=dim, si_scale=scale)
    return out


def parse_symmetry_spec(raw: str) -> SymmetrySpec:
    """Parse a symmetry declaration:
    - `'time_translation'` (shifts `t -> t + dt`; Noether: energy conservation)
    - `'space_translation'` (shifts `x -> x + dx`; Noether: momentum conservation)
    - `'parity'` or `'parity:x,v'` (inverts `x -> -x`, etc.)
    - `'shift:x1,x2'` (simultaneous additive shift `v -> v + delta`)
    - `'scale:x=1,t=1'` or `'scale:r=-2,F=4'` (`v -> lambda^exp * v`)
    - `'exchange:m1,m2'` (swaps two variables `m1 <-> m2`)
    """
    clean = raw.strip()
    if not clean:
        raise PhysicsModelError("empty symmetry specification")
    if ":" not in clean:
        kind = clean.lower()
        if kind == "time_translation":
            return SymmetrySpec(raw=clean, kind="time_translation", targets=("t",))
        if kind == "space_translation":
            return SymmetrySpec(raw=clean, kind="space_translation", targets=("x",))
        if kind == "parity":
            return SymmetrySpec(raw=clean, kind="parity", targets=("x", "r", "v", "p"))
        raise PhysicsModelError(
            f"unknown bare symmetry {clean!r}; expected time_translation, space_translation, "
            "parity[:vars], shift:<vars>, scale:<var=exp,...>, or exchange:<v1,v2>"
        )

    head, rest = clean.split(":", 1)
    kind = head.strip().lower()
    items = [it.strip() for it in rest.split(",") if it.strip()]
    if kind in ("parity", "shift"):
        if not items:
            raise PhysicsModelError(f"{kind} symmetry requires at least one variable")
        return SymmetrySpec(raw=clean, kind=kind, targets=tuple(items))
    if kind == "exchange":
        if len(items) != 2:
            raise PhysicsModelError(f"exchange symmetry requires exactly 2 variables, got {items!r}")
        return SymmetrySpec(raw=clean, kind="exchange", targets=(items[0], items[1]))
    if kind == "scale":
        exps: Dict[str, float] = {}
        for item in items:
            if "=" in item:
                vn, ev = item.split("=", 1)
                exps[vn.strip()] = float(Fraction(ev.strip()))
            else:
                exps[item] = 1.0
        if not exps:
            raise PhysicsModelError("scale symmetry requires at least one var=exp entry")
        return SymmetrySpec(raw=clean, kind="scale", targets=tuple(exps.keys()), exponents=exps)
    raise PhysicsModelError(f"unknown symmetry kind {kind!r} in {clean!r}")


def _infer_dimension(node: ast.AST, variables: Mapping[str, VariableSpec], violations: List[str]) -> Dimension:
    """Propagate exact rational SI dimensions across an algebraic AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return DIM_ONE
    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id].dimension
        if node.id in _BUILTIN_DIMENSIONLESS:
            return DIM_ONE
        violations.append(f"undeclared symbol {node.id!r} in equation")
        return DIM_ONE
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _infer_dimension(node.operand, variables, violations)
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, (ast.Add, ast.Sub)):
            dl = _infer_dimension(node.left, variables, violations)
            dr = _infer_dimension(node.right, variables, violations)
            if dl != dr:
                op_sym = "+" if isinstance(node.op, ast.Add) else "-"
                violations.append(
                    f"dimensional mismatch in '{op_sym}': left has [{dl.format_si()}] "
                    f"!= right [{dr.format_si()}]"
                )
            return dl
        if isinstance(node.op, ast.Mult):
            return _infer_dimension(node.left, variables, violations) * _infer_dimension(
                node.right, variables, violations
            )
        if isinstance(node.op, ast.Div):
            return _infer_dimension(node.left, variables, violations) / _infer_dimension(
                node.right, variables, violations
            )
        if isinstance(node.op, ast.Pow):
            base_dim = _infer_dimension(node.left, variables, violations)
            try:
                exp_frac = _eval_rational_exponent(node.right)
                return base_dim ** exp_frac
            except PhysicsModelError:
                exp_dim = _infer_dimension(node.right, variables, violations)
                if not exp_dim.is_dimensionless():
                    violations.append(
                        f"exponent must be dimensionless [1], got [{exp_dim.format_si()}]"
                    )
                if not base_dim.is_dimensionless():
                    violations.append(
                        f"variable exponent requires dimensionless base [1], got [{base_dim.format_si()}]"
                    )
                return DIM_ONE
        violations.append(f"unsupported operator {type(node.op).__name__} in formal model equation (use ** for powers)")
        return DIM_ONE
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.keywords or len(node.args) != 1:
            violations.append("only unary mathematical functions (sqrt, abs, exp, log, sin, cos, ...) are allowed")
            return DIM_ONE
        fname = node.func.id
        arg_dim = _infer_dimension(node.args[0], variables, violations)
        if fname == "sqrt":
            return arg_dim ** Fraction(1, 2)
        if fname == "abs":
            return arg_dim
        if fname in _TRANSCENDENTAL_FUNCS:
            if not arg_dim.is_dimensionless():
                violations.append(
                    f"transcendental function {fname}() requires dimensionless argument [1], "
                    f"got [{arg_dim.format_si()}]"
                )
            return DIM_ONE
        violations.append(f"unsupported function {fname!r} in formal model")
        return DIM_ONE
    violations.append(f"disallowed AST construct {type(node).__name__} in formal model")
    return DIM_ONE


def _eval_numeric(node: ast.AST, env: Mapping[str, float]) -> float:
    """Pure, sandboxed numeric evaluator over validated algebraic AST nodes."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        if node.id in _BUILTIN_DIMENSIONLESS:
            return _BUILTIN_DIMENSIONLESS[node.id]
        raise PhysicsModelError(f"unbound variable {node.id!r}")
    if isinstance(node, ast.UnaryOp):
        val = _eval_numeric(node.operand, env)
        return val if isinstance(node.op, ast.UAdd) else -val
    if isinstance(node, ast.BinOp):
        left = _eval_numeric(node.left, env)
        right = _eval_numeric(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return float(left ** right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and len(node.args) == 1:
        arg = _eval_numeric(node.args[0], env)
        fname = node.func.id
        if fname == "sqrt":
            return math.sqrt(abs(arg))
        if fname == "abs":
            return abs(arg)
        if fname in _TRANSCENDENTAL_FUNCS:
            fn = _TRANSCENDENTAL_FUNCS[fname]
            if fname in ("asin", "acos"):
                arg = max(-0.95, min(0.95, arg))
            elif fname in ("log", "ln", "log10", "log2"):
                arg = abs(arg) if abs(arg) > 1e-12 else 0.5
            elif fname in ("exp", "sinh", "cosh"):
                arg = max(-20.0, min(20.0, arg))
            return fn(arg)
    raise PhysicsModelError(f"cannot evaluate AST node {type(node).__name__}")


_SYMMETRY_CONSERVATION_LABELS: Dict[str, str] = {
    "time_translation": "Noether: conservation of energy (no explicit coordinate-time dependence)",
    "space_translation": "Noether: conservation of linear momentum (spatial shift invariance)",
    "parity": "Spatial inversion / parity symmetry",
    "shift": "Additive gauge / coordinate shift invariance",
    "scale": "Scale / similarity invariance",
    "exchange": "Particle / index exchange symmetry",
}


def _check_symmetry(
    sym: SymmetrySpec,
    parsed_eqs: Sequence[Tuple[str, ast.AST, ast.AST]],
    variables: Mapping[str, VariableSpec],
) -> SymmetryReport:
    """Verify whether all model equations remain satisfied under the symmetry transformation."""
    law = _SYMMETRY_CONSERVATION_LABELS.get(sym.kind, sym.kind)

    # Validate explicit variable targets exist (for bare time/space/parity, apply to matching declared vars)
    if ":" in sym.raw:
        missing = [v for v in sym.targets if v not in variables]
        if missing:
            return SymmetryReport(
                raw=sym.raw,
                conserved_law=law,
                invariant=False,
                detail=f"symmetry references undeclared variable(s): {', '.join(missing)}",
            )

    active_targets = [v for v in sym.targets if v in variables]
    if not active_targets:
        return SymmetryReport(
            raw=sym.raw,
            conserved_law=law,
            invariant=True,
            detail=f"trivially invariant ({law}; no transformed coordinate appears explicitly)",
        )

    # Deterministic irrational-like positive test points to avoid accidental cancellations
    seeds = (0.37, 0.71)
    ordered_vars = sorted(variables.keys())

    for trial_idx, base_offset in enumerate(seeds):
        env: Dict[str, float] = {}
        for idx, vname in enumerate(ordered_vars):
            env[vname] = 0.23 + base_offset * (idx + 1) * 0.41

        for raw_eq, lhs_node, rhs_node in parsed_eqs:
            local_env = dict(env)
            # Satisfy `y = RHS(...)` on the base point when LHS is a declared variable
            if isinstance(lhs_node, ast.Name) and lhs_node.id in local_env:
                local_env[lhs_node.id] = _eval_numeric(rhs_node, local_env)
            base_residual = _eval_numeric(lhs_node, local_env) - _eval_numeric(rhs_node, local_env)

            transformed = dict(local_env)
            if sym.kind in ("time_translation", "space_translation", "shift"):
                delta = 1.37 + 0.53 * trial_idx
                for v in active_targets:
                    transformed[v] = local_env[v] + delta
            elif sym.kind == "parity":
                for v in active_targets:
                    transformed[v] = -local_env[v]
            elif sym.kind == "exchange":
                v1, v2 = active_targets[0], active_targets[1]
                transformed[v1], transformed[v2] = local_env[v2], local_env[v1]
            elif sym.kind == "scale":
                lam = 2.0 + 0.5 * trial_idx
                for v, exp in sym.exponents.items():
                    if v in transformed:
                        transformed[v] = local_env[v] * (lam ** exp)

            new_lhs = _eval_numeric(lhs_node, transformed)
            new_rhs = _eval_numeric(rhs_node, transformed)
            new_residual = new_lhs - new_rhs
            scale_norm = max(1.0, abs(new_lhs), abs(new_rhs), abs(_eval_numeric(rhs_node, local_env)))
            if abs(new_residual - base_residual) > 1e-8 * scale_norm:
                return SymmetryReport(
                    raw=sym.raw,
                    conserved_law=law,
                    invariant=False,
                    detail=(
                        f"broken in {raw_eq!r}: transformed residual Δ={new_residual - base_residual:.4g} "
                        f"!= 0 ({law})"
                    ),
                )

    return SymmetryReport(
        raw=sym.raw,
        conserved_law=law,
        invariant=True,
        detail=f"verified invariant across test transformations ({law})",
    )


def build_and_check_model(
    variables: Union[str, Mapping[str, str], Sequence[Tuple[str, str]]],
    equations: Union[str, Sequence[str]],
    symmetries: Union[str, Sequence[str]] = (),
) -> FormalModelResult:
    """Build and verify a formal physics model (variables + SI units, equations, symmetries)."""
    var_map = parse_variable_specs(variables)

    eq_list: List[str] = (
        [e.strip() for e in equations.split(";") if e.strip()]
        if isinstance(equations, str)
        else [str(e).strip() for e in equations if str(e).strip()]
    )
    if not eq_list:
        raise PhysicsModelError("formal model requires at least one equation ('LHS = RHS')")

    sym_strings: List[str] = []
    if isinstance(symmetries, str):
        # Split on ';' first; if no ';', split top-level symmetries carefully (e.g. 'time_translation;parity:x,v'
        # or 'time_translation,space_translation')
        if ";" in symmetries:
            sym_strings = [s.strip() for s in symmetries.split(";") if s.strip()]
        elif symmetries.strip():
            # Group comma tokens that belong to a preceding 'kind:v1,v2'
            parts = [p.strip() for p in symmetries.split(",") if p.strip()]
            known_heads = ("time_translation", "space_translation", "parity", "shift", "scale", "exchange")
            for p in parts:
                head = p.split(":", 1)[0].strip().lower()
                if head in known_heads or not sym_strings:
                    sym_strings.append(p)
                else:
                    sym_strings[-1] = sym_strings[-1] + "," + p
    else:
        sym_strings = [str(s).strip() for s in symmetries if str(s).strip()]

    sym_specs = [parse_symmetry_spec(s) for s in sym_strings]

    all_violations: List[str] = []
    eq_reports: List[EquationReport] = []
    parsed_eqs: List[Tuple[str, ast.AST, ast.AST]] = []

    for raw_eq in eq_list:
        sys024 = check_sys024_clean(raw_eq, context=f"equation {raw_eq!r}")
        if sys024:
            all_violations.append(sys024)
            eq_reports.append(EquationReport(raw_eq, DIM_ONE, DIM_ONE, False, sys024))
            continue
        if "=" not in raw_eq or "==" in raw_eq:
            msg = f"equation {raw_eq!r} must have the form 'LHS = RHS'"
            all_violations.append(msg)
            eq_reports.append(EquationReport(raw_eq, DIM_ONE, DIM_ONE, False, msg))
            continue

        lhs_txt, rhs_txt = raw_eq.split("=", 1)
        try:
            lhs_ast = ast.parse(lhs_txt.strip(), mode="eval").body
            rhs_ast = ast.parse(rhs_txt.strip(), mode="eval").body
        except SyntaxError as exc:
            msg = f"syntax error in equation {raw_eq!r}: {exc.msg}"
            all_violations.append(msg)
            eq_reports.append(EquationReport(raw_eq, DIM_ONE, DIM_ONE, False, msg))
            continue

        eq_violations: List[str] = []
        lhs_dim = _infer_dimension(lhs_ast, var_map, eq_violations)
        rhs_dim = _infer_dimension(rhs_ast, var_map, eq_violations)
        if lhs_dim != rhs_dim:
            eq_violations.append(
                f"LHS [{lhs_dim.format_si()}] != RHS [{rhs_dim.format_si()}] in {raw_eq!r}"
            )

        if eq_violations:
            all_violations.extend(eq_violations)
            eq_reports.append(
                EquationReport(raw_eq, lhs_dim, rhs_dim, False, "; ".join(eq_violations))
            )
        else:
            parsed_eqs.append((raw_eq, lhs_ast, rhs_ast))
            eq_reports.append(
                EquationReport(
                    raw_eq,
                    lhs_dim,
                    rhs_dim,
                    True,
                    f"[{lhs_dim.format_si()}] == [{rhs_dim.format_si()}]",
                )
            )

    sym_reports: List[SymmetryReport] = []
    if not all_violations and sym_specs:
        for sspec in sym_specs:
            srep = _check_symmetry(sspec, parsed_eqs, var_map)
            sym_reports.append(srep)
            if not srep.invariant:
                all_violations.append(f"symmetry {srep.raw!r} violated: {srep.detail}")

    verdict = "CONSISTENT" if not all_violations else "RULED_OUT"
    if verdict == "CONSISTENT":
        sym_note = f" and {len(sym_reports)} symmetry check(s)" if sym_reports else ""
        summary = (
            f"All {len(eq_reports)} equation(s){sym_note} are dimensionally and algebraically "
            "consistent; dimensional consistency does not by itself confirm physical truth."
        )
    else:
        summary = "; ".join(all_violations)

    return FormalModelResult(
        verdict=verdict,
        variables=var_map,
        equations=tuple(eq_reports),
        symmetries=tuple(sym_reports),
        violations=tuple(all_violations),
        summary=summary,
    )


# ── Capability #10: Prediction vs. Standard-Theory Reference Comparator ──────


@dataclass(frozen=True)
class StandardReference:
    key: str
    name: str
    value: Decimal
    uncertainty: Decimal
    unit: str
    source_ref: str
    bound_kind: str = "MEASURED_REFERENCE"  # EXACT_VALUE | MEASURED_REFERENCE | UPPER_BOUND


STANDARD_REFERENCES: Dict[str, StandardReference] = {
    "speed_of_light": StandardReference(
        key="speed_of_light",
        name="Speed of light in vacuum c",
        value=Decimal("299792458"),
        uncertainty=Decimal("0"),
        unit="m/s",
        source_ref="SI 2019 exact redefinition (BIPM SI Brochure 9th ed.)",
        bound_kind="EXACT_VALUE",
    ),
    "planck_constant_hbar": StandardReference(
        key="planck_constant_hbar",
        name="Reduced Planck constant hbar",
        value=Decimal("1.054571817E-34"),
        uncertainty=Decimal("0"),
        unit="J*s",
        source_ref="SI 2019 / CODATA 2022 exact h/(2*pi)",
        bound_kind="EXACT_VALUE",
    ),
    "boltzmann_constant": StandardReference(
        key="boltzmann_constant",
        name="Boltzmann constant k_B",
        value=Decimal("1.380649E-23"),
        uncertainty=Decimal("0"),
        unit="J/K",
        source_ref="SI 2019 exact redefinition (BIPM SI Brochure 9th ed.)",
        bound_kind="EXACT_VALUE",
    ),
    "fine_structure_inv": StandardReference(
        key="fine_structure_inv",
        name="Inverse fine-structure constant alpha^-1",
        value=Decimal("137.035999177"),
        uncertainty=Decimal("0.000000021"),
        unit="1",
        source_ref="CODATA 2022 (Mohr et al., Rev. Mod. Phys. 97, 025002, 2025)",
        bound_kind="MEASURED_REFERENCE",
    ),
    "electron_g_minus_2": StandardReference(
        key="electron_g_minus_2",
        name="Electron anomalous magnetic moment a_e = (g-2)/2",
        value=Decimal("0.00115965218059"),
        uncertainty=Decimal("0.00000000000013"),
        unit="1",
        source_ref="Fan et al., Phys. Rev. Lett. 130, 071801 (2023) / 5-loop QED",
        bound_kind="MEASURED_REFERENCE",
    ),
    "rydberg_energy_ev": StandardReference(
        key="rydberg_energy_ev",
        name="Rydberg unit of energy hc R_inf",
        value=Decimal("13.605693122990"),
        uncertainty=Decimal("0.000000000015"),
        unit="eV",
        source_ref="CODATA 2022 (Mohr et al., Rev. Mod. Phys. 97, 025002, 2025)",
        bound_kind="MEASURED_REFERENCE",
    ),
    "planck_mass_kg": StandardReference(
        key="planck_mass_kg",
        name="Planck mass sqrt(hbar*c/G)",
        value=Decimal("2.176434E-8"),
        uncertainty=Decimal("0.000024E-8"),
        unit="kg",
        source_ref="CODATA 2022 (Mohr et al., Rev. Mod. Phys. 97, 025002, 2025)",
        bound_kind="MEASURED_REFERENCE",
    ),
    "tsirelson_chsh": StandardReference(
        key="tsirelson_chsh",
        name="Tsirelson quantum CHSH correlator ceiling 2*sqrt(2)",
        value=Decimal("2.8284271247461903"),
        uncertainty=Decimal("0"),
        unit="1",
        source_ref="Cirel'son (Tsirelson), Lett. Math. Phys. 4, 93-100 (1980)",
        bound_kind="UPPER_BOUND",
    ),
    "bell_local_chsh_ceiling": StandardReference(
        key="bell_local_chsh_ceiling",
        name="Bell-CHSH local hidden-variable ceiling",
        value=Decimal("2.0"),
        uncertainty=Decimal("0"),
        unit="1",
        source_ref="Clauser, Horne, Shimony, Holt, Phys. Rev. Lett. 23, 880 (1969)",
        bound_kind="UPPER_BOUND",
    ),
    "bell_2q_depolarizing_visibility": StandardReference(
        key="bell_2q_depolarizing_visibility",
        name="2-qubit Werner state CHSH-violation visibility threshold 1/sqrt(2)",
        value=Decimal("0.7071067811865475"),
        uncertainty=Decimal("0"),
        unit="1",
        source_ref="Werner, Phys. Rev. A 40, 4277 (1989)",
        bound_kind="EXACT_VALUE",
    ),
    "carnot_efficiency_300_600": StandardReference(
        key="carnot_efficiency_300_600",
        name="Carnot thermal efficiency ceiling (T_c=300 K, T_h=600 K)",
        value=Decimal("0.5"),
        uncertainty=Decimal("0"),
        unit="1",
        source_ref="Carnot / Second Law of Thermodynamics: eta_max = 1 - T_c / T_h",
        bound_kind="UPPER_BOUND",
    ),
    "free_space_impedance_ohm": StandardReference(
        key="free_space_impedance_ohm",
        name="Characteristic impedance of vacuum Z_0",
        value=Decimal("376.730313412"),
        uncertainty=Decimal("0.000000059"),
        unit="Ohm",
        source_ref="CODATA 2022 (Mohr et al., Rev. Mod. Phys. 97, 025002, 2025)",
        bound_kind="MEASURED_REFERENCE",
    ),
}


@dataclass(frozen=True)
class TheoryComparisonResult:
    verdict: str  # CONSISTENT | RULED_OUT
    reference: StandardReference
    prediction_in_ref_unit: float
    pred_uncertainty_in_ref_unit: float
    residual: float
    relative_error: float
    combined_uncertainty: float
    z_score: Optional[float]
    tol_sigma: float
    rel_tol: float
    why: str


def compare_to_standard_theory(
    prediction: Union[float, Decimal, str],
    *,
    ref_key: Optional[str] = None,
    ref_value: Optional[Union[float, Decimal, str]] = None,
    ref_uncertainty: Union[float, Decimal, str] = 0.0,
    ref_unit: str = "1",
    source_ref: str = "",
    pred_uncertainty: Union[float, Decimal, str] = 0.0,
    pred_unit: Optional[str] = None,
    tol_sigma: float = 3.0,
    rel_tol: float = 1e-4,
) -> TheoryComparisonResult:
    """Compare a predicted numeric value against an established standard-theory reference."""
    if tol_sigma <= 0 or rel_tol < 0:
        raise PhysicsModelError("tol_sigma must be > 0 and rel_tol must be >= 0")

    if ref_key is not None:
        key_clean = ref_key.strip()
        sys024 = check_sys024_clean(key_clean, context="reference key")
        if sys024:
            raise PhysicsModelError(sys024)
        if key_clean not in STANDARD_REFERENCES:
            known = ", ".join(sorted(STANDARD_REFERENCES.keys()))
            raise PhysicsModelError(f"unknown standard-theory reference {key_clean!r}; known: {known}")
        ref = STANDARD_REFERENCES[key_clean]
    else:
        if ref_value is None:
            raise PhysicsModelError("provide either ref_key=... or explicit ref_value=... with source_ref=...")
        clean_src = str(source_ref or "").strip()
        if not clean_src:
            raise PhysicsModelError("custom standard-theory comparison requires non-empty source_ref provenance")
        sys024 = check_sys024_clean(clean_src, context="source_ref")
        if sys024:
            raise PhysicsModelError(sys024)
        sys024_val = check_sys024_clean(str(ref_value), context="ref_value")
        if sys024_val:
            raise PhysicsModelError(sys024_val)
        parse_unit_dimension(ref_unit)
        ref = StandardReference(
            key="custom_reference",
            name=f"Custom standard-theory reference ({clean_src})",
            value=Decimal(str(ref_value)),
            uncertainty=Decimal(str(ref_uncertainty)),
            unit=ref_unit,
            source_ref=clean_src,
            bound_kind="MEASURED_REFERENCE" if float(ref_uncertainty) > 0 else "EXACT_VALUE",
        )

    pred_val = float(prediction)
    pred_unc = float(pred_uncertainty)
    if pred_unc < 0 or not math.isfinite(pred_val) or not math.isfinite(pred_unc):
        raise PhysicsModelError("prediction and uncertainty must be finite with uncertainty >= 0")

    ref_dim, ref_scale = parse_unit_dimension(ref.unit)
    if pred_unit is not None and pred_unit.strip():
        p_dim, p_scale = parse_unit_dimension(pred_unit)
        if p_dim != ref_dim:
            return TheoryComparisonResult(
                verdict="RULED_OUT",
                reference=ref,
                prediction_in_ref_unit=pred_val,
                pred_uncertainty_in_ref_unit=pred_unc,
                residual=float("nan"),
                relative_error=float("nan"),
                combined_uncertainty=float("nan"),
                z_score=None,
                tol_sigma=tol_sigma,
                rel_tol=rel_tol,
                why=(
                    f"Dimensional mismatch: prediction unit {pred_unit!r} [{p_dim.format_si()}] "
                    f"!= reference {ref.name} unit {ref.unit!r} [{ref_dim.format_si()}] ({ref.source_ref})."
                ),
            )
        conv = p_scale / ref_scale
        pred_in_ref = pred_val * conv
        pred_unc_in_ref = pred_unc * conv
    else:
        pred_in_ref = pred_val
        pred_unc_in_ref = pred_unc

    ref_float = float(ref.value)
    ref_unc_float = float(ref.uncertainty)
    residual = pred_in_ref - ref_float
    rel_err = abs(residual) / abs(ref_float) if ref_float != 0.0 else abs(residual)
    comb_unc = math.hypot(pred_unc_in_ref, ref_unc_float)
    z_score: Optional[float] = (abs(residual) / comb_unc) if comb_unc > 0.0 else None

    if ref.bound_kind == "UPPER_BOUND":
        margin = (tol_sigma * comb_unc) if comb_unc > 0.0 else (rel_tol * max(1.0, abs(ref_float)))
        if pred_in_ref <= ref_float + margin:
            verdict = "CONSISTENT"
            why = (
                f"Prediction {pred_in_ref:.6g} <= upper bound {ref_float:.6g} {ref.unit} "
                f"within tolerance ({ref.source_ref})."
            )
        else:
            verdict = "RULED_OUT"
            why = (
                f"Prediction {pred_in_ref:.6g} exceeds standard-theory upper bound "
                f"{ref_float:.6g} {ref.unit} by {residual:+.4g} ({ref.source_ref})."
            )
    elif comb_unc > 0.0 and z_score is not None:
        if z_score <= tol_sigma:
            verdict = "CONSISTENT"
            why = (
                f"Within {z_score:.2f}σ <= {tol_sigma:.2f}σ of {ref.name} "
                f"({ref.source_ref})."
            )
        else:
            verdict = "RULED_OUT"
            why = (
                f"Discrepant by {z_score:.2f}σ > {tol_sigma:.2f}σ from {ref.name} "
                f"({ref.source_ref})."
            )
    else:
        if rel_err <= rel_tol:
            verdict = "CONSISTENT"
            why = (
                f"Relative error {rel_err:.3e} <= {rel_tol:.3e} vs exact standard-theory "
                f"value {ref.name} ({ref.source_ref})."
            )
        else:
            verdict = "RULED_OUT"
            why = (
                f"Relative error {rel_err:.3e} > {rel_tol:.3e} vs exact standard-theory "
                f"value {ref.name} ({ref.source_ref})."
            )

    return TheoryComparisonResult(
        verdict=verdict,
        reference=ref,
        prediction_in_ref_unit=pred_in_ref,
        pred_uncertainty_in_ref_unit=pred_unc_in_ref,
        residual=residual,
        relative_error=rel_err,
        combined_uncertainty=comb_unc,
        z_score=z_score,
        tol_sigma=tol_sigma,
        rel_tol=rel_tol,
        why=why,
    )


def format_standard_references() -> str:
    lines = ["REGISTERED STANDARD-THEORY REFERENCES (SYS-024 governed):"]
    for key in sorted(STANDARD_REFERENCES.keys()):
        ref = STANDARD_REFERENCES[key]
        unc_part = f" ± {ref.uncertainty}" if ref.uncertainty > 0 else " (exact)"
        lines.append(f"  {ref.key:<32} {ref.value}{unc_part} [{ref.unit}]")
        lines.append(f"    source: {ref.source_ref}")
    return "\n".join(lines)
