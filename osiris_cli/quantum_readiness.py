"""Wave 4 Quantum Research Capability (#11 + G-14 + REQ-Q-1..8):
Simulator-first local hardware-readiness gate (resource model & noise-sensitivity sweep).

Enforces:
  * Resource & coherence budget modeling: qubit ceiling, 1Q/2Q gate counts & depth,
    estimated duration vs. backend `min(T1, T2)` coherence window, and gate+decoherence
    survival floor.
  * Deterministic local noise-sensitivity sweep across depolarizing error rates using a
    built-in exact density-matrix Kraus channel simulator (pure Python stdlib, so Python
    3.9+ works offline without external packages) with optional Qiskit `QuantumCircuit`
    extraction.
  * `G-14` raw-counts preservation: unmutated integer shot counts (`sum == shots`) and
    canonical SHA-256 digest (`raw_counts_sha256`) persisted at every sweep point.
  * Governance & provenance (`REQ-Q-1..8` & `SYS-024`):
    - Evidence classes `SIMULATED`, `PREVIEW`, `ILLUSTRATIVE`, `HARDWARE_ATTESTED`
      (simulator runs are strictly `SIMULATED` with `hardware=False` and refuse
      `HARDWARE_ATTESTED` spoofing).
    - `EXPLORATORY` vs. `CONFIRMATORY` claim modes (`CONFIRMATORY` fails closed without
      `preregistration_id` + 64-hex `preregistration_sha256`).
    - Zero quarantined CRSM constants (`SYS-024`).
    - Append-only `exec-canon-1` hash-chained JSONL receipts (`0600`, zero floats) under
      `$OSIRIS_HOME/quantum/readiness_receipts.jsonl`.
    - Live QPU submission adapter (`submit_to_hardware`) stays strictly fail-closed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shlex
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .exec_kernel import ExecJournal, canon
from .physics_model import check_sys024_clean

SCHEMA = "osiris.quantum_readiness_receipt/1"
EVIDENCE_CLASSES: Tuple[str, ...] = ("SIMULATED", "PREVIEW", "ILLUSTRATIVE", "HARDWARE_ATTESTED")
CLAIM_MODES: Tuple[str, ...] = ("EXPLORATORY", "CONFIRMATORY")
VERDICTS: Tuple[str, ...] = (
    "READY_FOR_REVIEW",
    "BLOCKED_GOVERNANCE",
    "BLOCKED_RESOURCES",
    "BLOCKED_COHERENCE",
    "BLOCKED_NOISE_SENSITIVITY",
)
DEFAULT_NOISE_GRID: Tuple[float, ...] = (0.0, 0.001, 0.005, 0.01, 0.02)
MAX_DENSITY_MATRIX_QUBITS = 6

PathArg = Optional[Union[os.PathLike, str]]
GateTuple = Tuple[str, Tuple[int, ...], Tuple[float, ...]]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class QuantumReadinessError(ValueError):
    """Invalid circuit specification, governance parameter, or tampered readiness ledger."""


class HardwareSubmissionDisabledError(RuntimeError):
    """Raised unconditionally by `submit_to_hardware`: live QPU dispatch is fail-closed."""


def default_osiris_home(osiris_home: PathArg = None) -> Path:
    raw = osiris_home or os.environ.get("OSIRIS_HOME")
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".osiris").resolve()


def default_readiness_receipts_path(osiris_home: PathArg = None) -> Path:
    if osiris_home is None:
        override = os.environ.get("OSIRIS_QUANTUM_READINESS_LOG")
        if override:
            return Path(override).expanduser().resolve()
    return default_osiris_home(osiris_home) / "quantum" / "readiness_receipts.jsonl"


def _canon_safe(val: Any) -> Any:
    """Recursively convert floats to exact `Decimal` strings for `exec-canon-1`."""
    if isinstance(val, float):
        if not math.isfinite(val):
            return "NaN"
        return str(Decimal(str(round(val, 12))))
    if isinstance(val, Decimal):
        return str(val)
    if isinstance(val, dict):
        return {str(k): _canon_safe(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_canon_safe(v) for v in val]
    return val


@dataclass(frozen=True)
class BackendProfile:
    name: str
    max_qubits: int
    t1_us: float
    t2_us: float
    t_1q_ns: float
    t_2q_ns: float
    err_1q: float
    err_2q: float
    readout_err: float
    max_shots: int = 100_000
    coherence_ratio_ceiling: float = 0.25
    min_survival_floor: float = 0.35
    min_operating_fidelity: float = 0.60


BACKEND_PROFILES: Dict[str, BackendProfile] = {
    "heron_r2_156q": BackendProfile(
        name="heron_r2_156q",
        max_qubits=156,
        t1_us=180.0,
        t2_us=150.0,
        t_1q_ns=32.0,
        t_2q_ns=68.0,
        err_1q=2.5e-4,
        err_2q=3.5e-3,
        readout_err=1.2e-2,
    ),
    "eagle_r3_127q": BackendProfile(
        name="eagle_r3_127q",
        max_qubits=127,
        t1_us=220.0,
        t2_us=120.0,
        t_1q_ns=35.55,
        t_2q_ns=660.0,
        err_1q=3.0e-4,
        err_2q=7.5e-3,
        readout_err=1.8e-2,
    ),
    "local_ideal_32q": BackendProfile(
        name="local_ideal_32q",
        max_qubits=32,
        t1_us=10_000.0,
        t2_us=10_000.0,
        t_1q_ns=20.0,
        t_2q_ns=40.0,
        err_1q=1.0e-5,
        err_2q=1.0e-4,
        readout_err=1.0e-3,
    ),
}


@dataclass(frozen=True)
class CircuitResourceSpec:
    name: str
    family: str
    num_qubits: int
    single_qubit_gates: int
    two_qubit_gates: int
    single_qubit_depth: int
    two_qubit_depth: int
    idle_delay_ns: float = 0.0
    shots: int = 4096
    seed: int = 7
    gates: Tuple[GateTuple, ...] = ()


def build_family_circuit(
    family: str = "ghz",
    *,
    num_qubits: int = 3,
    layers: int = 1,
    idle_delay_ns: float = 0.0,
    shots: int = 4096,
    seed: int = 7,
    name: str = "",
) -> CircuitResourceSpec:
    """Construct a concrete benchmark circuit specification with explicit gate instructions."""
    fam = family.strip().lower()
    if num_qubits < 1:
        raise QuantumReadinessError(f"num_qubits must be >= 1, got {num_qubits}")
    if layers < 1:
        raise QuantumReadinessError(f"layers must be >= 1, got {layers}")

    sim_n = min(num_qubits, MAX_DENSITY_MATRIX_QUBITS)
    gates: List[GateTuple] = []

    if fam == "bell":
        n_eff = max(2, num_qubits) if num_qubits >= 2 else 2
        sim_n = min(n_eff, MAX_DENSITY_MATRIX_QUBITS)
        gates.append(("h", (0,), ()))
        for _ in range(layers):
            gates.append(("cx", (0, 1), ()))
        return CircuitResourceSpec(
            name=name or f"bell_{n_eff}q_L{layers}",
            family="bell",
            num_qubits=n_eff,
            single_qubit_gates=1,
            two_qubit_gates=layers,
            single_qubit_depth=1,
            two_qubit_depth=layers,
            idle_delay_ns=float(idle_delay_ns),
            shots=int(shots),
            seed=int(seed),
            gates=tuple(gates),
        )

    if fam == "ghz":
        gates.append(("h", (0,), ()))
        for _ in range(layers):
            for q in range(sim_n - 1):
                gates.append(("cx", (q, q + 1), ()))
        total_2q = max(0, num_qubits - 1) * layers
        return CircuitResourceSpec(
            name=name or f"ghz_{num_qubits}q_L{layers}",
            family="ghz",
            num_qubits=num_qubits,
            single_qubit_gates=1,
            two_qubit_gates=total_2q,
            single_qubit_depth=1,
            two_qubit_depth=total_2q,
            idle_delay_ns=float(idle_delay_ns),
            shots=int(shots),
            seed=int(seed),
            gates=tuple(gates),
        )

    if fam in ("ising_trotter", "trotter"):
        for _ in range(layers):
            for q in range(sim_n):
                gates.append(("rx", (q,), (0.35,)))
            for q in range(0, sim_n - 1, 2):
                gates.append(("cx", (q, q + 1), ()))
                gates.append(("rz", (q + 1,), (0.25,)))
                gates.append(("cx", (q, q + 1), ()))
            for q in range(1, sim_n - 1, 2):
                gates.append(("cx", (q, q + 1), ()))
                gates.append(("rz", (q + 1,), (0.25,)))
                gates.append(("cx", (q, q + 1), ()))
        bonds = max(0, num_qubits - 1)
        total_1q = layers * (num_qubits + bonds)
        total_2q = layers * 2 * bonds
        depth_2q = layers * (4 if num_qubits >= 3 else (2 if num_qubits == 2 else 0))
        depth_1q = layers * 3
        return CircuitResourceSpec(
            name=name or f"ising_trotter_{num_qubits}q_L{layers}",
            family="ising_trotter",
            num_qubits=num_qubits,
            single_qubit_gates=total_1q,
            two_qubit_gates=total_2q,
            single_qubit_depth=depth_1q,
            two_qubit_depth=depth_2q,
            idle_delay_ns=float(idle_delay_ns),
            shots=int(shots),
            seed=int(seed),
            gates=tuple(gates),
        )

    if fam == "ramsey_idle":
        for q in range(sim_n):
            gates.append(("h", (q,), ()))
        for _ in range(layers):
            for q in range(sim_n):
                gates.append(("rz", (q,), (0.2,)))
        for q in range(sim_n):
            gates.append(("h", (q,), ()))
        return CircuitResourceSpec(
            name=name or f"ramsey_idle_{num_qubits}q_L{layers}",
            family="ramsey_idle",
            num_qubits=num_qubits,
            single_qubit_gates=num_qubits * (2 + layers),
            two_qubit_gates=0,
            single_qubit_depth=2 + layers,
            two_qubit_depth=0,
            idle_delay_ns=float(idle_delay_ns),
            shots=int(shots),
            seed=int(seed),
            gates=tuple(gates),
        )

    raise QuantumReadinessError(
        f"unknown circuit family {family!r}; expected bell, ghz, ising_trotter, or ramsey_idle"
    )


def from_qiskit_circuit(
    qc: Any,
    *,
    idle_delay_ns: float = 0.0,
    shots: int = 4096,
    seed: int = 7,
    name: str = "",
) -> CircuitResourceSpec:
    """Extract a `CircuitResourceSpec` from a Qiskit `QuantumCircuit` without importing qiskit globally."""
    num_qubits = int(getattr(qc, "num_qubits", 0))
    if num_qubits < 1:
        raise QuantumReadinessError("QuantumCircuit must have num_qubits >= 1")

    n_1q = 0
    n_2q = 0
    extracted_gates: List[GateTuple] = []
    q_index_map = {q: idx for idx, q in enumerate(getattr(qc, "qubits", ()))}

    for inst in getattr(qc, "data", ()):
        op = getattr(inst, "operation", inst[0] if isinstance(inst, tuple) else None)
        qargs = getattr(inst, "qubits", inst[1] if isinstance(inst, tuple) else ())
        if op is None:
            continue
        op_name = str(getattr(op, "name", "")).lower()
        if op_name in ("measure", "barrier", "delay"):
            continue
        q_idxs = tuple(int(q_index_map.get(q, 0)) for q in qargs)
        params = tuple(float(p) for p in getattr(op, "params", ()))
        if len(q_idxs) == 1:
            n_1q += 1
        elif len(q_idxs) >= 2:
            n_2q += 1
        if all(idx < MAX_DENSITY_MATRIX_QUBITS for idx in q_idxs):
            if op_name in ("h", "x", "y", "z", "s", "sdg", "rx", "ry", "rz", "cx", "cz", "id"):
                extracted_gates.append((op_name, q_idxs[:2], params))

    total_depth = int(qc.depth()) if callable(getattr(qc, "depth", None)) else (n_1q + n_2q)
    depth_2q = min(total_depth, n_2q)
    depth_1q = max(0, total_depth - depth_2q)
    return CircuitResourceSpec(
        name=name or str(getattr(qc, "name", "qiskit_circuit")),
        family="qiskit",
        num_qubits=num_qubits,
        single_qubit_gates=n_1q,
        two_qubit_gates=n_2q,
        single_qubit_depth=depth_1q,
        two_qubit_depth=depth_2q,
        idle_delay_ns=float(idle_delay_ns),
        shots=int(shots),
        seed=int(seed),
        gates=tuple(extracted_gates),
    )


# ── Exact Pure-Stdlib Density-Matrix Noisy Simulator ─────────────────────────

ComplexMat = List[List[complex]]

_SQRT2_INV = 1.0 / math.sqrt(2.0)

_PAULIS_1Q: Dict[str, Tuple[Tuple[complex, complex], Tuple[complex, complex]]] = {
    "I": ((1.0 + 0j, 0j), (0j, 1.0 + 0j)),
    "X": ((0j, 1.0 + 0j), (1.0 + 0j, 0j)),
    "Y": ((0j, -1.0j), (1.0j, 0j)),
    "Z": ((1.0 + 0j, 0j), (0j, -1.0 + 0j)),
}


def _single_qubit_unitary(
    name: str, params: Tuple[float, ...]
) -> Tuple[Tuple[complex, complex], Tuple[complex, complex]]:
    if name in ("id", "i"):
        return _PAULIS_1Q["I"]
    if name == "h":
        return ((_SQRT2_INV + 0j, _SQRT2_INV + 0j), (_SQRT2_INV + 0j, -_SQRT2_INV + 0j))
    if name == "x":
        return _PAULIS_1Q["X"]
    if name == "y":
        return _PAULIS_1Q["Y"]
    if name == "z":
        return _PAULIS_1Q["Z"]
    if name == "s":
        return ((1.0 + 0j, 0j), (0j, 1.0j))
    if name == "sdg":
        return ((1.0 + 0j, 0j), (0j, -1.0j))
    theta = params[0] if params else 0.0
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    if name == "rx":
        return ((c + 0j, -1.0j * s), (-1.0j * s, c + 0j))
    if name == "ry":
        return ((c + 0j, -s + 0j), (s + 0j, c + 0j))
    if name == "rz":
        return ((complex(c, -s), 0j), (0j, complex(c, s)))
    raise QuantumReadinessError(f"unsupported single-qubit gate {name!r}")


def _apply_1q_unitary(
    rho: ComplexMat,
    n_qubits: int,
    q: int,
    u: Tuple[Tuple[complex, complex], Tuple[complex, complex]],
) -> ComplexMat:
    """Compute `U_q rho U_q^dagger` in-place copy for an `n_qubits` density matrix."""
    dim = 1 << n_qubits
    bit = 1 << q
    u00, u01 = u[0]
    u10, u11 = u[1]
    # Left-multiply: tmp = U_q @ rho
    tmp: ComplexMat = [[0j] * dim for _ in range(dim)]
    for r in range(dim):
        if r & bit:
            continue
        r1 = r | bit
        row0 = rho[r]
        row1 = rho[r1]
        t0 = tmp[r]
        t1 = tmp[r1]
        for c in range(dim):
            v0 = row0[c]
            v1 = row1[c]
            t0[c] = u00 * v0 + u01 * v1
            t1[c] = u10 * v0 + u11 * v1

    # Right-multiply: out = tmp @ U_q^dagger
    u00c, u01c = u00.conjugate(), u01.conjugate()
    u10c, u11c = u10.conjugate(), u11.conjugate()
    out: ComplexMat = [[0j] * dim for _ in range(dim)]
    for r in range(dim):
        tr = tmp[r]
        orow = out[r]
        for c in range(dim):
            if c & bit:
                continue
            c1 = c | bit
            v0 = tr[c]
            v1 = tr[c1]
            orow[c] = v0 * u00c + v1 * u01c
            orow[c1] = v0 * u10c + v1 * u11c
    return out


def _apply_2q_gate(rho: ComplexMat, n_qubits: int, name: str, q_ctrl: int, q_targ: int) -> ComplexMat:
    dim = 1 << n_qubits
    c_bit = 1 << q_ctrl
    t_bit = 1 << q_targ
    out: ComplexMat = [[0j] * dim for _ in range(dim)]
    if name == "cx":
        for r in range(dim):
            rr = (r ^ t_bit) if (r & c_bit) else r
            src_row = rho[r]
            dst_row = out[rr]
            for c in range(dim):
                cc = (c ^ t_bit) if (c & c_bit) else c
                dst_row[cc] = src_row[c]
        return out
    if name == "cz":
        mask = c_bit | t_bit
        for r in range(dim):
            r_sign = -1.0 if (r & mask) == mask else 1.0
            src_row = rho[r]
            dst_row = out[r]
            for c in range(dim):
                c_sign = -1.0 if (c & mask) == mask else 1.0
                dst_row[c] = (r_sign * c_sign) * src_row[c]
        return out
    raise QuantumReadinessError(f"unsupported two-qubit gate {name!r}")


def _depolarize_1q(rho: ComplexMat, n_qubits: int, q: int, p: float) -> ComplexMat:
    if p <= 0.0:
        return rho
    p_clamped = min(1.0, max(0.0, p))
    dim = 1 << n_qubits
    rx = _apply_1q_unitary(rho, n_qubits, q, _PAULIS_1Q["X"])
    ry = _apply_1q_unitary(rho, n_qubits, q, _PAULIS_1Q["Y"])
    rz = _apply_1q_unitary(rho, n_qubits, q, _PAULIS_1Q["Z"])
    w0 = 1.0 - p_clamped
    wp = p_clamped / 3.0
    return [
        [w0 * rho[r][c] + wp * (rx[r][c] + ry[r][c] + rz[r][c]) for c in range(dim)]
        for r in range(dim)
    ]


def _depolarize_2q(rho: ComplexMat, n_qubits: int, q0: int, q1: int, p: float) -> ComplexMat:
    if p <= 0.0:
        return rho
    p_clamped = min(1.0, max(0.0, p))
    dim = 1 << n_qubits
    w0 = 1.0 - p_clamped
    wp = p_clamped / 15.0
    acc: ComplexMat = [[w0 * rho[r][c] for c in range(dim)] for r in range(dim)]
    pauli_keys = ("I", "X", "Y", "Z")
    for k0 in pauli_keys:
        r0 = rho if k0 == "I" else _apply_1q_unitary(rho, n_qubits, q0, _PAULIS_1Q[k0])
        for k1 in pauli_keys:
            if k0 == "I" and k1 == "I":
                continue
            r01 = r0 if k1 == "I" else _apply_1q_unitary(r0, n_qubits, q1, _PAULIS_1Q[k1])
            for r in range(dim):
                arow = acc[r]
                prow = r01[r]
                for c in range(dim):
                    arow[c] += wp * prow[c]
    return acc


def _dephase_idle_all(rho: ComplexMat, n_qubits: int, gamma_phi: float) -> ComplexMat:
    if gamma_phi <= 0.0:
        return rho
    cur = rho
    dim = 1 << n_qubits
    w0 = 1.0 - 0.5 * gamma_phi
    wz = 0.5 * gamma_phi
    for q in range(n_qubits):
        rz = _apply_1q_unitary(cur, n_qubits, q, _PAULIS_1Q["Z"])
        cur = [[w0 * cur[r][c] + wz * rz[r][c] for c in range(dim)] for r in range(dim)]
    return cur


def _simulate_density_matrix(
    gates: Sequence[GateTuple],
    n_qubits: int,
    *,
    p_1q: float,
    p_2q: float,
    gamma_phi: float = 0.0,
) -> ComplexMat:
    dim = 1 << n_qubits
    rho: ComplexMat = [[0j] * dim for _ in range(dim)]
    rho[0][0] = 1.0 + 0j
    for gname, qubits, params in gates:
        if len(qubits) == 1:
            u = _single_qubit_unitary(gname, params)
            rho = _apply_1q_unitary(rho, n_qubits, qubits[0], u)
            rho = _depolarize_1q(rho, n_qubits, qubits[0], p_1q)
        elif len(qubits) == 2:
            rho = _apply_2q_gate(rho, n_qubits, gname, qubits[0], qubits[1])
            rho = _depolarize_2q(rho, n_qubits, qubits[0], qubits[1], p_2q)
    if gamma_phi > 0.0:
        rho = _dephase_idle_all(rho, n_qubits, gamma_phi)
    return rho


def _state_fidelity_with_pure(rho_pure: ComplexMat, rho_noisy: ComplexMat) -> float:
    """For pure `rho_pure = |psi><psi|`, fidelity `F = Re(Tr(rho_pure @ rho_noisy))` in `[0, 1]`."""
    dim = len(rho_pure)
    total = 0.0
    for r in range(dim):
        pr = rho_pure[r]
        for c in range(dim):
            total += (pr[c] * rho_noisy[c][r]).real
    return max(0.0, min(1.0, total))


def _sample_shot_counts(probs: Sequence[float], n_qubits: int, shots: int, rng: random.Random) -> Dict[str, int]:
    """Sample exact integer shot counts (`sum == shots`, G-14 unmutated raw bitstrings)."""
    cdf: List[float] = []
    running = 0.0
    for p in probs:
        running += max(0.0, p)
        cdf.append(running)
    total_mass = cdf[-1] if cdf and cdf[-1] > 0 else 1.0
    counts: Dict[str, int] = {}
    dim = len(probs)
    for _ in range(shots):
        u = rng.random() * total_mass
        lo, hi = 0, dim - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if u <= cdf[mid]:
                hi = mid
            else:
                lo = mid + 1
        bitstr = format(lo, f"0{n_qubits}b")
        counts[bitstr] = counts.get(bitstr, 0) + 1
    return dict(sorted(counts.items()))


def _hash_raw_counts(raw_counts: Mapping[str, int], shots: int) -> str:
    payload = canon({"counts": {str(k): int(v) for k, v in sorted(raw_counts.items())}, "shots": int(shots)})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ── Resource Model & Sweep Dataclasses ───────────────────────────────────────


@dataclass(frozen=True)
class ResourceModelReport:
    backend_name: str
    num_qubits: int
    max_qubits: int
    single_qubit_gates: int
    two_qubit_gates: int
    single_qubit_depth: int
    two_qubit_depth: int
    duration_ns: float
    effective_coherence_ns: float
    coherence_ratio: float
    gate_survival: float
    decoherence_survival: float
    gate_decoherence_survival: float
    readout_assignment_fidelity: float
    within_qubit_ceiling: bool
    within_shot_ceiling: bool
    within_coherence_budget: bool
    within_survival_floor: bool


@dataclass(frozen=True)
class NoiseSweepPoint:
    p_2q: float
    p_1q: float
    state_fidelity: float
    tvd_similarity: float
    shots: int
    raw_counts: Dict[str, int]
    raw_counts_sha256: str


@dataclass(frozen=True)
class ReadinessGateReport:
    readiness_id: str
    verdict: str
    ready_for_human_review: bool
    hardware_executed: bool
    evidence_class: str
    claim_mode: str
    preregistration_id: str
    preregistration_sha256: str
    circuit_name: str
    active_simulated_qubits: int
    resource_model: ResourceModelReport
    noise_sweep: Tuple[NoiseSweepPoint, ...]
    operating_point_fidelity: float
    fidelity_drop_per_pct_2q: float
    reasons: Tuple[str, ...]
    receipt_hash: str
    receipt_path: str


def evaluate_resource_model(spec: CircuitResourceSpec, profile: BackendProfile) -> ResourceModelReport:
    duration_ns = (
        spec.single_qubit_depth * profile.t_1q_ns
        + spec.two_qubit_depth * profile.t_2q_ns
        + max(0.0, spec.idle_delay_ns)
    )
    eff_coh_ns = min(profile.t1_us, profile.t2_us) * 1000.0
    coh_ratio = duration_ns / eff_coh_ns if eff_coh_ns > 0 else float("inf")
    gate_surv = ((1.0 - profile.err_1q) ** max(0, spec.single_qubit_gates)) * (
        (1.0 - profile.err_2q) ** max(0, spec.two_qubit_gates)
    )
    decoh_surv = math.exp(-duration_ns / (profile.t2_us * 1000.0)) if profile.t2_us > 0 else 0.0
    combined_surv = gate_surv * decoh_surv
    ro_fid = (1.0 - profile.readout_err) ** max(0, spec.num_qubits)

    return ResourceModelReport(
        backend_name=profile.name,
        num_qubits=spec.num_qubits,
        max_qubits=profile.max_qubits,
        single_qubit_gates=spec.single_qubit_gates,
        two_qubit_gates=spec.two_qubit_gates,
        single_qubit_depth=spec.single_qubit_depth,
        two_qubit_depth=spec.two_qubit_depth,
        duration_ns=duration_ns,
        effective_coherence_ns=eff_coh_ns,
        coherence_ratio=coh_ratio,
        gate_survival=gate_surv,
        decoherence_survival=decoh_surv,
        gate_decoherence_survival=combined_surv,
        readout_assignment_fidelity=ro_fid,
        within_qubit_ceiling=(1 <= spec.num_qubits <= profile.max_qubits),
        within_shot_ceiling=(64 <= spec.shots <= profile.max_shots),
        within_coherence_budget=(coh_ratio <= profile.coherence_ratio_ceiling),
        within_survival_floor=(combined_surv >= profile.min_survival_floor),
    )


def run_noise_sensitivity_sweep(
    spec: CircuitResourceSpec,
    profile: BackendProfile,
    *,
    noise_grid: Sequence[float] = DEFAULT_NOISE_GRID,
) -> Tuple[int, Tuple[NoiseSweepPoint, ...], float, float]:
    """Run local density-matrix depolarizing noise sweep, preserving unmutated G-14 shot counts."""
    sim_qubits = max(1, min(spec.num_qubits, MAX_DENSITY_MATRIX_QUBITS))
    gates = list(spec.gates)
    if not gates:
        # Synthesize a faithful representative lightcone gate list matching spec depths
        fallback = build_family_circuit(
            "ghz" if spec.two_qubit_gates > 0 else "ramsey_idle",
            num_qubits=sim_qubits,
            layers=max(1, spec.two_qubit_depth // max(1, sim_qubits - 1)) if spec.two_qubit_gates > 0 else 1,
            idle_delay_ns=spec.idle_delay_ns,
            shots=spec.shots,
            seed=spec.seed,
        )
        gates = list(fallback.gates)

    grid = sorted({0.0} | {float(p) for p in noise_grid if p >= 0.0})
    ratio_1q_to_2q = (profile.err_1q / profile.err_2q) if profile.err_2q > 0 else 0.1
    gamma_idle = (
        1.0 - math.exp(-max(0.0, spec.idle_delay_ns) / (profile.t2_us * 1000.0))
        if spec.idle_delay_ns > 0 and profile.t2_us > 0
        else 0.0
    )

    rho_ideal = _simulate_density_matrix(gates, sim_qubits, p_1q=0.0, p_2q=0.0, gamma_phi=0.0)
    dim = 1 << sim_qubits
    probs_ideal = [max(0.0, rho_ideal[i][i].real) for i in range(dim)]

    points: List[NoiseSweepPoint] = []
    shots_eff = max(1, int(spec.shots))
    for idx, p2 in enumerate(grid):
        p1 = min(1.0, p2 * ratio_1q_to_2q)
        g_phi = gamma_idle if p2 > 0.0 else 0.0
        rho_p = _simulate_density_matrix(gates, sim_qubits, p_1q=p1, p_2q=p2, gamma_phi=g_phi)
        probs_p = [max(0.0, rho_p[i][i].real) for i in range(dim)]
        sfid = _state_fidelity_with_pure(rho_ideal, rho_p)

        rng = random.Random(int(spec.seed) * 1009 + idx * 37)
        raw_counts = _sample_shot_counts(probs_p, sim_qubits, shots_eff, rng)
        counts_sha = _hash_raw_counts(raw_counts, shots_eff)

        tvd = 0.5 * sum(
            abs((raw_counts.get(format(i, f"0{sim_qubits}b"), 0) / shots_eff) - probs_ideal[i])
            for i in range(dim)
        )
        tvd_sim = max(0.0, min(1.0, 1.0 - tvd))
        points.append(
            NoiseSweepPoint(
                p_2q=p2,
                p_1q=p1,
                state_fidelity=sfid,
                tvd_similarity=tvd_sim,
                shots=shots_eff,
                raw_counts=raw_counts,
                raw_counts_sha256=counts_sha,
            )
        )

    # Operating-point fidelity at `profile.err_2q` (exact density-matrix evaluation at backend err_2q)
    rho_op = _simulate_density_matrix(
        gates,
        sim_qubits,
        p_1q=profile.err_1q,
        p_2q=profile.err_2q,
        gamma_phi=gamma_idle,
    )
    op_fid_lightcone = _state_fidelity_with_pure(rho_ideal, rho_op)
    # Account for full-circuit gate count when `spec.num_qubits > sim_qubits`
    sim_2q_count = sum(1 for _n, qs, _ps in gates if len(qs) == 2)
    extra_2q = max(0, spec.two_qubit_gates - sim_2q_count)
    op_fidelity = op_fid_lightcone * ((1.0 - profile.err_2q) ** extra_2q)

    max_p = grid[-1]
    slope_per_pct = ((points[0].state_fidelity - points[-1].state_fidelity) / (max_p * 100.0)) if max_p > 0 else 0.0
    return sim_qubits, tuple(points), op_fidelity, slope_per_pct


def evaluate_hardware_readiness(
    spec: CircuitResourceSpec,
    *,
    backend: Union[str, BackendProfile] = "heron_r2_156q",
    claim_mode: str = "EXPLORATORY",
    evidence_class: str = "SIMULATED",
    preregistration_id: str = "",
    preregistration_sha256: str = "",
    noise_grid: Sequence[float] = DEFAULT_NOISE_GRID,
    osiris_home: PathArg = None,
    receipts_path: PathArg = None,
) -> ReadinessGateReport:
    """Run the full simulator-first hardware-readiness gate and append an `exec-canon-1` receipt."""
    profile = (
        BACKEND_PROFILES[backend]
        if isinstance(backend, str) and backend in BACKEND_PROFILES
        else (backend if isinstance(backend, BackendProfile) else None)
    )
    if profile is None:
        known = ", ".join(sorted(BACKEND_PROFILES.keys()))
        raise QuantumReadinessError(f"unknown backend profile {backend!r}; known: {known}")

    gov_reasons: List[str] = []
    mode_up = str(claim_mode or "").strip().upper()
    ev_up = str(evidence_class or "").strip().upper()

    if mode_up not in CLAIM_MODES:
        gov_reasons.append(f"invalid claim_mode {claim_mode!r}; must be EXPLORATORY or CONFIRMATORY")
    if ev_up not in EVIDENCE_CLASSES:
        gov_reasons.append(f"invalid evidence_class {evidence_class!r}; must be one of {EVIDENCE_CLASSES}")
    if ev_up == "HARDWARE_ATTESTED":
        gov_reasons.append(
            "simulator readiness gate cannot claim HARDWARE_ATTESTED evidence_class (hardware=False)"
        )

    clean_prereg_id = str(preregistration_id or "").strip()
    clean_prereg_sha = str(preregistration_sha256 or "").strip().lower()
    if mode_up == "CONFIRMATORY":
        if not clean_prereg_id:
            gov_reasons.append("CONFIRMATORY claim_mode requires a non-empty preregistration_id")
        if not _SHA256_RE.match(clean_prereg_sha):
            gov_reasons.append(
                "CONFIRMATORY claim_mode requires a 64-hex SHA-256 preregistration_sha256"
            )

    for ctx_name, text_val in (
        ("circuit name", spec.name),
        ("preregistration_id", clean_prereg_id),
    ):
        sys024 = check_sys024_clean(text_val, context=ctx_name)
        if sys024:
            gov_reasons.append(sys024)

    res_model = evaluate_resource_model(spec, profile)
    sim_qubits, sweep_points, op_fid, slope_pct = run_noise_sensitivity_sweep(
        spec, profile, noise_grid=noise_grid
    )

    reasons: List[str] = list(gov_reasons)
    verdict = "READY_FOR_REVIEW"

    if gov_reasons:
        verdict = "BLOCKED_GOVERNANCE"
    elif not res_model.within_qubit_ceiling or not res_model.within_shot_ceiling:
        verdict = "BLOCKED_RESOURCES"
        if not res_model.within_qubit_ceiling:
            reasons.append(
                f"num_qubits={res_model.num_qubits} outside backend {profile.name} ceiling [1, {profile.max_qubits}]"
            )
        if not res_model.within_shot_ceiling:
            reasons.append(
                f"shots={spec.shots} outside allowed range [64, {profile.max_shots}]"
            )
    elif not res_model.within_coherence_budget or not res_model.within_survival_floor:
        verdict = "BLOCKED_COHERENCE"
        if not res_model.within_coherence_budget:
            reasons.append(
                f"circuit duration {res_model.duration_ns:.1f} ns consumes "
                f"{res_model.coherence_ratio * 100:.1f}% > {profile.coherence_ratio_ceiling * 100:.1f}% "
                f"of {profile.name} coherence window ({res_model.effective_coherence_ns:.0f} ns)"
            )
        if not res_model.within_survival_floor:
            reasons.append(
                f"estimated gate+decoherence survival {res_model.gate_decoherence_survival:.3f} "
                f"< floor {profile.min_survival_floor:.3f}"
            )
    elif op_fid < profile.min_operating_fidelity:
        verdict = "BLOCKED_NOISE_SENSITIVITY"
        reasons.append(
            f"operating-point state fidelity {op_fid:.3f} at err_2q={profile.err_2q:.4g} "
            f"< minimum {profile.min_operating_fidelity:.3f}"
        )
    else:
        reasons.append(
            f"resource budget ({res_model.coherence_ratio * 100:.1f}% coherence used) and "
            f"noise sweep (F_op={op_fid:.3f}) pass simulator readiness; human review still required before any QPU run"
        )

    effective_ev_class = ev_up if ev_up in ("SIMULATED", "PREVIEW", "ILLUSTRATIVE") else "SIMULATED"
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rid_seed = f"{spec.name}:{profile.name}:{ts}:{spec.shots}:{spec.seed}"
    readiness_id = "qready-" + hashlib.sha256(rid_seed.encode("utf-8")).hexdigest()[:12]

    rpath = (
        Path(receipts_path).expanduser().resolve()
        if receipts_path is not None
        else default_readiness_receipts_path(osiris_home)
    )
    journal = ExecJournal(rpath)

    receipt_body: Dict[str, Any] = _canon_safe({
        "schema": SCHEMA,
        "readiness_id": readiness_id,
        "ts": ts,
        "verdict": verdict,
        "ready_for_human_review": verdict == "READY_FOR_REVIEW",
        "hardware": False,
        "evidence_class": effective_ev_class,
        "requested_evidence_class": ev_up,
        "claim_mode": mode_up if mode_up in CLAIM_MODES else "INVALID",
        "preregistration_id": clean_prereg_id,
        "preregistration_sha256": clean_prereg_sha,
        "circuit": {
            "name": spec.name,
            "family": spec.family,
            "num_qubits": spec.num_qubits,
            "active_simulated_qubits": sim_qubits,
            "single_qubit_gates": spec.single_qubit_gates,
            "two_qubit_gates": spec.two_qubit_gates,
            "single_qubit_depth": spec.single_qubit_depth,
            "two_qubit_depth": spec.two_qubit_depth,
            "idle_delay_ns": spec.idle_delay_ns,
            "shots": spec.shots,
            "seed": spec.seed,
        },
        "resource_model": {
            "backend_name": res_model.backend_name,
            "max_qubits": res_model.max_qubits,
            "duration_ns": res_model.duration_ns,
            "effective_coherence_ns": res_model.effective_coherence_ns,
            "coherence_ratio": res_model.coherence_ratio,
            "gate_survival": res_model.gate_survival,
            "decoherence_survival": res_model.decoherence_survival,
            "gate_decoherence_survival": res_model.gate_decoherence_survival,
            "readout_assignment_fidelity": res_model.readout_assignment_fidelity,
        },
        "noise_sweep": [
            {
                "p_2q": pt.p_2q,
                "p_1q": pt.p_1q,
                "state_fidelity": pt.state_fidelity,
                "tvd_similarity": pt.tvd_similarity,
                "shots": pt.shots,
                "raw_counts": pt.raw_counts,
                "raw_counts_sha256": pt.raw_counts_sha256,
            }
            for pt in sweep_points
        ],
        "operating_point_fidelity": op_fid,
        "fidelity_drop_per_pct_2q": slope_pct,
        "reasons": reasons,
    })
    canon(receipt_body)
    rhash = journal.append(receipt_body)
    try:
        os.chmod(rpath, 0o600)
    except OSError:
        pass

    return ReadinessGateReport(
        readiness_id=readiness_id,
        verdict=verdict,
        ready_for_human_review=(verdict == "READY_FOR_REVIEW"),
        hardware_executed=False,
        evidence_class=effective_ev_class,
        claim_mode=mode_up,
        preregistration_id=clean_prereg_id,
        preregistration_sha256=clean_prereg_sha,
        circuit_name=spec.name,
        active_simulated_qubits=sim_qubits,
        resource_model=res_model,
        noise_sweep=sweep_points,
        operating_point_fidelity=op_fid,
        fidelity_drop_per_pct_2q=slope_pct,
        reasons=tuple(reasons),
        receipt_hash=rhash,
        receipt_path=str(rpath),
    )


def submit_to_hardware(*_args: Any, **_kwargs: Any) -> None:
    """Fail-closed QPU submission adapter: OSIRIS never dispatches live QPU jobs autonomously."""
    raise HardwareSubmissionDisabledError(
        "HARDWARE_SUBMISSION_DISABLED: Live QPU submission is strictly fail-closed in OSIRIS CLI. "
        "Readiness receipts authorize human review of simulator evidence only, never live QPU execution."
    )


def assess_experiment_brief(
    brief: Mapping[str, Any],
    *,
    backend: str = "heron_r2_156q",
    claim_mode: str = "EXPLORATORY",
    preregistration_id: str = "",
    preregistration_sha256: str = "",
    osiris_home: PathArg = None,
) -> ReadinessGateReport:
    """Derive a `CircuitResourceSpec` from an `experiments.py` brief dict and gate it."""
    sys_text = str(brief.get("system", "")) + " " + str(brief.get("question", ""))
    controls_text = " ".join(str(c) for c in (brief.get("controls") or ()))

    q_match = re.search(r"(\d+)\s*[- ]*qubit", sys_text, re.IGNORECASE)
    num_qubits = int(q_match.group(1)) if q_match else 3

    d_match = re.search(r"depth\s*(?:\d+\s*[-–]\s*)?(\d+)", sys_text, re.IGNORECASE)
    layers = max(1, int(d_match.group(1)) // max(1, num_qubits - 1)) if d_match else 1

    s_match = re.search(r"(\d[\d,]*)\s*shots?", controls_text, re.IGNORECASE)
    shots = int(s_match.group(1).replace(",", "")) if s_match else 4096

    seed_match = re.search(r"seed\s*[=:]?\s*(\d+)", controls_text, re.IGNORECASE)
    seed = int(seed_match.group(1)) if seed_match else 7

    low_sys = sys_text.lower()
    if "bell" in low_sys or "chsh" in low_sys:
        family = "bell"
    elif "trotter" in low_sys or "ising" in low_sys:
        family = "ising_trotter"
    elif "ramsey" in low_sys or "idle" in low_sys:
        family = "ramsey_idle"
    else:
        family = "ghz"

    spec = build_family_circuit(
        family,
        num_qubits=num_qubits,
        layers=layers,
        shots=shots,
        seed=seed,
        name=f"brief_{brief.get('id', 'draft')}_{family}",
    )
    return evaluate_hardware_readiness(
        spec,
        backend=backend,
        claim_mode=claim_mode,
        preregistration_id=preregistration_id,
        preregistration_sha256=preregistration_sha256,
        osiris_home=osiris_home,
    )


def verify_receipts(path: PathArg = None) -> Tuple[bool, int, str]:
    rpath = Path(path).expanduser().resolve() if path is not None else default_readiness_receipts_path()
    return ExecJournal(rpath).verify()


def format_readiness_report(rep: ReadinessGateReport, *, width: int = 92) -> str:
    rm = rep.resource_model
    cap = max(60, min(width, 96))
    lines = [
        f"QUANTUM HARDWARE-READINESS GATE · {rep.verdict} ({rep.readiness_id})"[:cap],
        f"  circuit:   {rep.circuit_name} ({rm.num_qubits}q, 1Q={rm.single_qubit_gates}, "
        f"2Q={rm.two_qubit_gates}, 2Q-depth={rm.two_qubit_depth})",
        f"  backend:   {rm.backend_name} (ceiling {rm.max_qubits}q · hardware_executed={rep.hardware_executed})",
        f"  provenance: evidence={rep.evidence_class} · mode={rep.claim_mode}"
        + (f" · prereg={rep.preregistration_id}" if rep.preregistration_id else ""),
        f"  coherence: {rm.duration_ns:.1f} ns / {rm.effective_coherence_ns:.0f} ns "
        f"({rm.coherence_ratio * 100:.2f}%) · survival={rm.gate_decoherence_survival:.3f}",
        f"  operating: F_op={rep.operating_point_fidelity:.3f} · "
        f"sensitivity={rep.fidelity_drop_per_pct_2q:.3f} drop / 1% 2Q err",
        "  noise sweep (G-14 raw counts preserved):",
    ]
    for pt in rep.noise_sweep:
        lines.append(
            f"    p_2q={pt.p_2q:<6.4f}  F_state={pt.state_fidelity:.4f}  "
            f"TVD_sim={pt.tvd_similarity:.4f}  shots={pt.shots}  sha256={pt.raw_counts_sha256[:12]}"
        )
    for reason in rep.reasons:
        lines.append(f"  verdict: {reason}")
    lines.append(f"  receipt: {rep.receipt_hash[:16]} ({rep.receipt_path})")
    return "\n".join(lines)


USAGE = (
    "Usage:\n"
    "  /readiness check [family=ghz] [qubits=4] [layers=1] [idle_ns=0] [shots=4096] "
    "[backend=heron_r2_156q] [mode=EXPLORATORY|CONFIRMATORY] [prereg_id=...] [prereg_sha256=...]\n"
    "  /readiness brief <brief_id> [backend=heron_r2_156q]\n"
    "  /readiness backends\n"
    "  /readiness list\n"
    "  /readiness verify"
)


def run_command(args: str = "", *, osiris_home: PathArg = None) -> str:
    """CLI router for `/readiness` and `/experiment readiness`."""
    try:
        tokens = shlex.split(args or "")
    except ValueError:
        tokens = (args or "").strip().split()

    if not tokens:
        tokens = ["list"]
    sub = tokens[0].lower()

    if sub in ("help", "-h", "--help"):
        return USAGE

    if sub == "backends":
        out = ["CALIBRATED HARDWARE-READINESS PROFILES (offline envelope specs, 0 live QPU calls):"]
        for key in sorted(BACKEND_PROFILES.keys()):
            bp = BACKEND_PROFILES[key]
            out.append(
                f"  {bp.name:<18} {bp.max_qubits:>3}q · T1={bp.t1_us:.0f}us T2={bp.t2_us:.0f}us · "
                f"t_2q={bp.t_2q_ns:.1f}ns · err_2q={bp.err_2q:.2e}"
            )
        return "\n".join(out)

    rpath = default_readiness_receipts_path(osiris_home)
    if sub == "verify":
        from . import probes

        ok, count, err = verify_receipts(rpath)
        tail = probes.verify_chain_tail(rpath, tail_lines=20)
        status = "INTACT" if (ok and tail.intact) else "TAMPERED"
        return f"Quantum readiness receipt chain {status}: {count} receipts, tail={tail.status}" + (
            f" ({err})" if err else ""
        )

    if sub == "list":
        if not rpath.is_file():
            return "Quantum readiness ledger empty (0 simulator receipts). Run `/readiness check` to gate a circuit."
        ok, count, err = verify_receipts(rpath)
        if not ok:
            return f"Quantum readiness ledger TAMPERED ({err}) -- fail-closed."
        rows: List[Dict[str, Any]] = []
        with open(rpath, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
        out = [f"QUANTUM READINESS RECEIPTS · chain INTACT ({count} total, showing last {min(8, len(rows))}):"]
        for row in rows[-8:]:
            b = row.get("body") or {}
            circ = b.get("circuit") or {}
            rm = b.get("resource_model") or {}
            out.append(
                f"  [{b.get('verdict')}] {b.get('readiness_id')} · {circ.get('name')} "
                f"({circ.get('num_qubits')}q -> {rm.get('backend_name')}) · F_op={b.get('operating_point_fidelity')}"
            )
        return "\n".join(out)

    if sub == "brief" and len(tokens) >= 2:
        import experiments

        try:
            bid = int(tokens[1].lstrip("#"))
        except ValueError:
            return f"Expected integer brief id, got {tokens[1]!r}"
        kv_b: Dict[str, str] = {}
        for tok in tokens[2:]:
            if "=" in tok:
                k, v = tok.split("=", 1)
                kv_b[k.lower()] = v
        matches = [b for b in experiments.briefs() if b.get("id") == bid]
        if not matches:
            return f"Experiment brief #{bid} not found."
        try:
            rep = assess_experiment_brief(
                matches[0],
                backend=kv_b.get("backend", "heron_r2_156q"),
                claim_mode=kv_b.get("mode", "EXPLORATORY"),
                preregistration_id=kv_b.get("prereg_id", ""),
                preregistration_sha256=kv_b.get("prereg_sha256", ""),
                osiris_home=osiris_home,
            )
        except QuantumReadinessError as exc:
            return f"Readiness gate error: {exc}"
        return format_readiness_report(rep)

    # Allow `/readiness check ...` or shorthand `/readiness family=ghz qubits=4`
    kv_tokens = tokens[1:] if sub == "check" else (tokens if "=" in tokens[0] else None)
    if kv_tokens is not None:
        kv: Dict[str, str] = {}
        for tok in kv_tokens:
            if "=" not in tok:
                return f"Expected key=value, got {tok!r}\n{USAGE}"
            k, v = tok.split("=", 1)
            kv[k.lower()] = v
        try:
            spec = build_family_circuit(
                kv.get("family", "ghz"),
                num_qubits=int(kv.get("qubits", "3")),
                layers=int(kv.get("layers", "1")),
                idle_delay_ns=float(kv.get("idle_ns", "0")),
                shots=int(kv.get("shots", "4096")),
                seed=int(kv.get("seed", "7")),
                name=kv.get("name", ""),
            )
            rep = evaluate_hardware_readiness(
                spec,
                backend=kv.get("backend", "heron_r2_156q"),
                claim_mode=kv.get("mode", "EXPLORATORY"),
                evidence_class=kv.get("evidence", "SIMULATED"),
                preregistration_id=kv.get("prereg_id", ""),
                preregistration_sha256=kv.get("prereg_sha256", ""),
                osiris_home=osiris_home,
            )
        except (QuantumReadinessError, ValueError) as exc:
            return f"Readiness gate error: {exc}"
        return format_readiness_report(rep)

    return USAGE
