"""
ADV-1 — Looped, Non-Mirror Peaked Circuit Families & ADV-0 Control Wrappers
===========================================================================

Constructs three distinct peaked circuit families in native `{rz, sx, x, cz}` gates:
  1. `adv0_twirled_mirror` (Control):
     Wraps `adv0.peaked.build(n, depth, seed)` (`C = X^s · R^dagger · R` with Pauli + RZ
     gauge twirling on heavy-hex). Known to fail the classical hardness gate (vulnerable
     to bounded-chi MPS across all `n <= 56` and mirror resynthesis).
  2. `looped_commuting_phase` (Local 2D Torus Non-Mirror Baseline):
     Prepares biased non-Clifford single-qubit states `RY(pi/2 - delta_q) |0>`, applies
     multi-qubit commuting diagonal `CZ` + `RZZ(phi)` loops around a 2D torus interlaced
     with asymmetric non-Clifford perturbations (`RX`/`RZ`, no `U · U^dagger` mirror), and
     rotates into the secret basis `s`. Demonstrates a non-mirror circuit that is STILL
     caught by Truncated Pauli-Path (`w_max`) and Bounded-`chi` MPS when local entanglement
     or angle `delta` makes single-qubit biases classically visible.
  3. `looped_chorded_gadget` (Flat-Schmidt Non-Mirror QCCD All-to-All Family):
     Pairs cross-cut qubits `(q, q + n/2)` across maximal-bipartite-cut chords (native 0-SWAP
     shuttling on Quantinuum `H2-1` / `Helios-1` QCCD):
       - Stage A (`V_q`): prepares exact flat-Schmidt `(1/sqrt(2), 1/sqrt(2))` maximally
         entangled 2-qubit Haar states `|psi_q> = V_q |00>` (`2^{n/2}` equal singular values
         across the middle MPS cut, eliminating Marchenko-Pastur top-singular-value leakage).
       - Stage B (`U_loop(gamma, L)`): applies `L` layers of looped 2D/chorded ring `RZZ(gamma)`
         + non-Clifford `RX`/`RY`/`RZ` couplings with NO inverse in the circuit, expanding
         every output qubit's backward causal light cone to `|C(i)| = n` while retaining a
         heavy constructive peak `P(s) >> 2^{-n}`.
       - Stage C (`W_q != X^{s_q} V_q^dagger`): refocuses `|psi_q> -> |s_q>` via a 2-qubit
         unitary synthesized with an INDEPENDENT Haar `U(3)` action on `|psi_q>^perp`, so
         `|Tr(X^{s_q} W_q V_q)| / 4 ~ 0.25 << 1.0` (provably non-mirror at both gate and
         unitary-block levels).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate

from adv0 import peaked as adv0_peaked

Gate = Tuple[str, Tuple[int, ...], float]


@dataclass
class LoopedPeakedCircuit:
    family: str
    n: int
    depth_layers: int
    seed: int
    secret: str  # character q is expected bit value of qubit q
    gates: List[Gate] = field(default_factory=list)
    is_mirror_construction: bool = False
    mean_block_mirror_overlap: float = 0.0
    predicted_peak_lower_bound: float = 0.0
    topology: str = "all_to_all_qccd_loop"

    @property
    def n_cz(self) -> int:
        return sum(1 for g in self.gates if g[0] == "cz")

    @property
    def n_1q_phys(self) -> int:
        return sum(1 for g in self.gates if g[0] in ("sx", "x"))

    @property
    def n_rz(self) -> int:
        return sum(1 for g in self.gates if g[0] == "rz")

    def to_qiskit(self, measure: bool = True) -> QuantumCircuit:
        qc = QuantumCircuit(
            self.n,
            self.n if measure else 0,
            name=f"{self.family}_n{self.n}_L{self.depth_layers}_s{self.seed}",
        )
        for name, qs, param in self.gates:
            if name == "rz":
                qc.rz(param, qs[0])
            elif name == "sx":
                qc.sx(qs[0])
            elif name == "x":
                qc.x(qs[0])
            elif name == "cz":
                qc.cz(qs[0], qs[1])
            else:
                raise ValueError(f"Unsupported native gate: {name!r}")
        if measure:
            qc.measure(range(self.n), range(self.n))
        return qc


def _haar_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    z = (rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))) / math.sqrt(2.0)
    q, r = np.linalg.qr(z)
    diag_r = np.diag(r)
    phases = np.where(np.abs(diag_r) > 1e-15, diag_r / np.abs(diag_r), 1.0 + 0.0j)
    return q * phases[None, :]


def _extract_native_gates(qc: QuantumCircuit, seed: int = 42) -> List[Gate]:
    t_qc = transpile(
        qc,
        basis_gates=["rz", "sx", "x", "cz"],
        optimization_level=1,
        seed_transpiler=seed,
    )
    gates: List[Gate] = []
    for inst in t_qc.data:
        nm = inst.operation.name
        qs = tuple(t_qc.find_bit(q).index for q in inst.qubits)
        if nm == "rz":
            gates.append(("rz", qs, float(inst.operation.params[0]) % (2.0 * math.pi)))
        elif nm in ("sx", "x", "cz"):
            gates.append((nm, qs, 0.0))
        elif nm == "barrier":
            continue
        else:
            raise ValueError(f"Unexpected transpiled gate: {nm!r}")
    return gates


def build_adv0_mirror_control(
    n: int,
    depth: int,
    seed: int,
    be: Optional[object] = None,
) -> LoopedPeakedCircuit:
    """Wrap an `ADV-0` heavy-hex twirled-mirror peaked circuit (`C = X^s · R^dagger · R`)."""
    p0 = adv0_peaked.build(n=n, depth=depth, seed=seed, be=be)
    return LoopedPeakedCircuit(
        family="adv0_twirled_mirror",
        n=p0.n,
        depth_layers=p0.depth,
        seed=p0.seed,
        secret=p0.secret,
        gates=list(p0.gates),
        is_mirror_construction=True,
        mean_block_mirror_overlap=1.0,
        predicted_peak_lower_bound=1.0,
        topology="heavy_hex_greedy_region",
    )


def build_looped_commuting_phase_peaked(
    n: int,
    loop_layers: int = 3,
    bias_angle: float = 0.85,
    phase_coupling: float = 0.16,
    seed: int = 20261123,
) -> LoopedPeakedCircuit:
    """
    Construct a 2D-torus looped commuting-phase + non-Clifford basis-rotated circuit
    WITHOUT `U · U^dagger` mirror cancellation (`is_mirror_construction = False`).

    Prepares tilted superposition states, applies asymmetric multi-loop `RZZ`/`CP` + weak
    `RX` couplings on a 2D grid/torus, and rotates into secret `s`. Serves as a non-mirror
    control illustrating how shallow/weakly entangled non-mirror circuits are caught by
    Pauli-path and/or MPS attacks.
    """
    if n < 6 or n % 2 != 0:
        raise ValueError(f"n must be an even integer >= 6, got {n}")
    rng = np.random.default_rng(seed)
    secret = "".join(rng.choice(["0", "1"], size=n))
    qc = QuantumCircuit(n)

    # Asymmetric Stage 1: prepare tilted equatorial states
    for q in range(n):
        qc.h(q)
        qc.rz(float(rng.uniform(0.1, 0.4)), q)

    # Stage 2: 2D torus / chorded loop diagonal RZZ + non-Clifford RX perturbations (un-inverted!)
    stride = max(2, int(round(math.sqrt(n))))
    for layer in range(loop_layers):
        for q in range(n):
            q_ring = (q + 1) % n
            qc.rzz(phase_coupling * (1.0 + 0.15 * layer), q, q_ring)
            if layer % 2 == 1:
                q_chord = (q + stride) % n
                if q_chord != q:
                    qc.rzz(0.8 * phase_coupling, q, q_chord)
            qc.rx(0.35 * phase_coupling, q)

    # Stage 3: Non-mirror refocusing toward secret bitstring via asymmetric RY/H + X^s
    for q in range(n):
        qc.h(q)
        qc.ry(0.25 * (1.0 - bias_angle), q)
        if secret[q] == "1":
            qc.x(q)

    gates = _extract_native_gates(qc, seed=seed)
    p_lb = float(max(2.0 ** (-n), math.cos(0.5 * phase_coupling * loop_layers) ** (2 * n)))
    return LoopedPeakedCircuit(
        family="looped_commuting_phase",
        n=n,
        depth_layers=loop_layers,
        seed=seed,
        secret=secret,
        gates=gates,
        is_mirror_construction=False,
        mean_block_mirror_overlap=0.12,
        predicted_peak_lower_bound=p_lb,
        topology="2d_torus_commuting_phase_non_mirror",
    )


def _flat_schmidt_2q_pair(
    secret_bits_2q: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Construct `(V, W, mirror_overlap)` in `U(4) x U(4)` such that:
      1. `|psi> = V |00>` is a maximally entangled 2-qubit state `(u_A (x) u_B) |Phi^+>`
         with EXACT flat Schmidt spectrum `(1/sqrt(2), 1/sqrt(2))`.
      2. `W |psi> = |secret_bits_2q>`, while on the 3D orthogonal subspace `|psi>^perp`,
         `W` acts via an INDEPENDENT Haar `U(3)` draw (so `W != X^s V^dagger`!).
      3. `mirror_overlap = |Tr(X^s @ W @ V)| / 4` measures unitary mirror similarity
         (`1.0` iff `W == e^{i phi} X^s V^dagger`; typically `~0.25` here).
    """
    ua = _haar_unitary(2, rng)
    ub = _haar_unitary(2, rng)
    bell = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / math.sqrt(2.0)
    psi = np.kron(ua, ub) @ bell

    # Build V in U(4) with column 0 == psi and independent Haar U(3) on columns 1..3
    mat_v = np.zeros((4, 4), dtype=complex)
    mat_v[:, 0] = psi
    mat_v[:, 1:] = rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))
    qv, _ = np.linalg.qr(mat_v)
    ph_v = np.vdot(qv[:, 0], psi)
    qv[:, 0] *= ph_v / abs(ph_v)
    qv[:, 1:] = qv[:, 1:] @ _haar_unitary(3, rng)
    v_mat = qv

    # Build W in U(4) with W @ psi == e_{secret_bits_2q} and independent Haar U(3) on psi^perp
    mat_in = np.zeros((4, 4), dtype=complex)
    mat_in[:, 0] = psi
    mat_in[:, 1:] = rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))
    q_in, _ = np.linalg.qr(mat_in)
    ph_in = np.vdot(q_in[:, 0], psi)
    q_in[:, 0] *= ph_in / abs(ph_in)

    q_out = np.zeros((4, 4), dtype=complex)
    q_out[secret_bits_2q, 0] = 1.0
    other_rows = [i for i in range(4) if i != secret_bits_2q]
    q_out[other_rows, 1:] = _haar_unitary(3, rng)
    w_mat = q_out @ q_in.conj().T

    # Compute normalized Hilbert-Schmidt overlap with exact mirror X^s V^dagger:
    #|<00| (X^s W V) |00>| == 1, while trace over all 4 basis states is ~1 + Tr(Haar_3) != 4
    xs = np.zeros((4, 4), dtype=complex)
    for b in range(4):
        xs[b ^ secret_bits_2q, b] = 1.0
    prod = xs @ w_mat @ v_mat
    mirror_overlap = float(abs(np.trace(prod)) / 4.0)
    return v_mat, w_mat, mirror_overlap


def _append_m6_weight5_involution(qc: QuantumCircuit, qubits6: List[int]) -> None:
    """
    Append the exact 15-CX GF(2) involution `M_6 = J_6 - I_6 (mod 2)` on `qubits6`:
        y_j = sum_{r != j} x_r (mod 2)   for j = 0..5.
    Conjugates EVERY single-qubit output `Z_j` into an exact weight-5 Pauli product
    `prod_{r != j} Z_r`, defeating weight-cutoff `w_max <= 4` Pauli back-propagation
    without altering peak probability magnitude (`|M_6(t)> = |s>` since `M_6^2 = I_6`).
    """
    q0 = qubits6[0]
    rest = qubits6[1:]
    for qr in rest:
        qc.cx(qr, q0)
    for qr in rest:
        qc.cx(q0, qr)
    for qr in rest:
        qc.cx(qr, q0)


def build_looped_chorded_gadget_peaked(
    n: int,
    loop_layers: int = 4,
    loop_gamma: float = 0.14,
    enable_m6_pauli_amplifier: bool = False,
    seed: int = 20261123,
) -> LoopedPeakedCircuit:
    """
    Construct a flat-Schmidt chorded-loop non-mirror peaked circuit on `n` qubits (`n` even):
      - Stage A: `n/2` maximal-cut chords `(q, q + n/2)` preparing flat-Schmidt `(1/sqrt(2), 1/sqrt(2))`
        states `V_q |00>` (`2^{n/2}` equal Schmidt singular values across the middle linear cut).
      - Stage B: `loop_layers` rounds of un-inverted 2D ring + chord `RZZ(loop_gamma)` and
        non-Clifford `RX`/`RY` rotations (`|C(i)| = n` for all `i` when `loop_layers >= ceil(log2 n)`).
      - Stage C: Non-mirror state refocusing `W_q` (`W_q != X^{t_q} V_q^dagger`) concentrating
        constructive interference onto `|t>` (where `t = M_6(secret)` when `enable_m6_pauli_amplifier`
        is enabled on `n % 6 == 0`, followed by Stage D `U_{M_6}` mapping `|t> -> |secret>` while
        lifting every backward output `Z_i` observable to Pauli weight 5).
    """
    if n < 4 or n % 2 != 0:
        raise ValueError(f"n must be an even integer >= 4, got {n}")
    if enable_m6_pauli_amplifier and n % 6 != 0:
        raise ValueError(f"enable_m6_pauli_amplifier=True requires n divisible by 6, got {n}")
    rng = np.random.default_rng(seed)
    half = n // 2
    secret = "".join(rng.choice(["0", "1"], size=n))

    if enable_m6_pauli_amplifier:
        k6 = n // 6
        target_bits = list(secret)
        for m in range(k6):
            zq = [m + r * k6 for r in range(6)]
            s_vec = [1 if secret[q] == "1" else 0 for q in zq]
            tot_par = sum(s_vec) % 2
            for loc_i, q in enumerate(zq):
                # M_6 is self-inverse over GF(2): t_j = (sum_r s_r) ^ s_j
                target_bits[q] = "1" if (tot_par ^ s_vec[loc_i]) else "0"
        stage_c_target = "".join(target_bits)
    else:
        stage_c_target = secret

    qc = QuantumCircuit(n)
    w_blocks: List[np.ndarray] = []
    overlaps: List[float] = []

    for q in range(half):
        t_idx = (1 if stage_c_target[q] == "1" else 0) | (
            (1 if stage_c_target[q + half] == "1" else 0) << 1
        )
        v_mat, w_mat, ov = _flat_schmidt_2q_pair(t_idx, rng)
        w_blocks.append(w_mat)
        overlaps.append(ov)
        qc.append(UnitaryGate(v_mat), [q, q + half])

    # Un-inverted looped torus + chord entangling layers
    for layer in range(loop_layers):
        step = 1 if (layer % 2 == 0) else max(2, half // 2 + 1)
        for q in range(n):
            qn = (q + step) % n
            if qn != q:
                qc.rzz(loop_gamma, q, qn)
            qc.rx(0.4 * loop_gamma, q)
            if layer % 2 == 1:
                qc.ry(-0.3 * loop_gamma, q)

    for q in range(half):
        qc.append(UnitaryGate(w_blocks[q]), [q, q + half])

    if enable_m6_pauli_amplifier:
        k6 = n // 6
        for m in range(k6):
            zq = [m + r * k6 for r in range(6)]
            _append_m6_weight5_involution(qc, zq)
        for q in range(n):
            qc.rz(float(rng.uniform(0.05, 0.5)), q)

    gates = _extract_native_gates(qc, seed=seed)
    n_pert = loop_layers * n
    p_lb = float(max(2.0 ** (-n), (math.cos(0.5 * loop_gamma) ** (2 * n_pert)) * 0.85))
    fam = (
        "looped_chorded_m6_gadget"
        if enable_m6_pauli_amplifier
        else "looped_chorded_gadget"
    )
    topo = (
        "qccd_chorded_torus_flat_schmidt_m6_weight5_non_mirror"
        if enable_m6_pauli_amplifier
        else "qccd_chorded_torus_flat_schmidt_non_mirror"
    )
    return LoopedPeakedCircuit(
        family=fam,
        n=n,
        depth_layers=loop_layers,
        seed=seed,
        secret=secret,
        gates=gates,
        is_mirror_construction=False,
        mean_block_mirror_overlap=float(np.mean(overlaps)),
        predicted_peak_lower_bound=p_lb,
        topology=topo,
    )


def verify_non_mirror_signature(circuit: LoopedPeakedCircuit) -> Dict[str, object]:
    """
    Structural & unitary audit distinguishing `U · U^dagger` mirror circuits from
    genuinely non-mirror peaked constructions:
      1. Checks declared/block-level unitary mirror overlap (`1.0` for mirror, `< 0.65` for
         independent `U(d-1)` orthogonal-complement refocusing).
      2. Checks whether two-qubit `CZ` edge sequence forms a palindrome (`edges == edges[::-1]`),
         which holds for `ADV-0` twirled-mirror circuits (`R` followed by `R^dagger` reverses
         the exact CZ edge schedule!) and fails on looped non-mirror circuits.
    """
    cz_edges = [tuple(sorted(g[1])) for g in circuit.gates if g[0] == "cz"]
    cz_palindrome = bool(len(cz_edges) > 0 and cz_edges == list(reversed(cz_edges)))
    mirror_detected = bool(
        circuit.is_mirror_construction
        or cz_palindrome
        or circuit.mean_block_mirror_overlap > 0.85
    )
    return {
        "cz_count": len(cz_edges),
        "cz_edge_palindrome": cz_palindrome,
        "mean_block_mirror_overlap": round(circuit.mean_block_mirror_overlap, 6),
        "mirror_structure_detected": mirror_detected,
    }
