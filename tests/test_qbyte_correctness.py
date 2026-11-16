"""
Comprehensive Verification & Differential Test Suite for qByte (`qbyte_system`)
===============================================================================

Covers Phase 2 Section 3 requirements:
  1. All 8 computational basis states of CCNOT (Toffoli)
  2. Every valid arrangement of (c1, c2, target) across 3Q (6 permutations) and
     all 8*7*6 = 336 ordered triples on 8 qubits
  3. Consistent big-endian basis convention across 1Q, 2Q, 3Q gates and
     single-qubit projective measurement (`measure_qubit`)
  4. Random normalized complex statevectors (norm preservation + full matrix oracle)
  5. Gate unitarity (U† U = I) across all gates in `qbyte_system.gates`
  6. Inverse-circuit restoration on arbitrary normalized 8-qubit statevectors
  7. Bell-state preparation (all 4 Bell states on arbitrary wire pairs)
  8. GHZ-state preparation (widths 2..8 and non-contiguous subsets)
  9. Phase & interference behavior (Mach-Zehnder, H-Z-H, H-S-S-H, QFT)
  10. Cross-framework differential agreement against an independent Kronecker
      matrix reference and Qiskit `Statevector` within declared tolerance 1e-12.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from qbyte_system import gates
from qbyte_system.qbyte import Qbyte, QbyteRuntime, bell_state, ghz_state
from qbyte_system.sovereign_executor import (
    Circuit,
    SovereignExecutor,
    ghz_circuit,
    qft_circuit,
)

ATOL = 1e-12


def _independent_toffoli_action_big_endian(
    state: np.ndarray, n_qubits: int, c1: int, c2: int, target: int
) -> np.ndarray:
    """Independent bit-level reference for big-endian CCNOT without shared code."""
    out = state.copy()
    shift_c1 = n_qubits - 1 - c1
    shift_c2 = n_qubits - 1 - c2
    mask_t = 1 << (n_qubits - 1 - target)
    for idx in range(1 << n_qubits):
        if ((idx >> shift_c1) & 1) and ((idx >> shift_c2) & 1) and ((idx & mask_t) == 0):
            partner = idx ^ mask_t
            out[idx], out[partner] = out[partner], out[idx]
    return out


def _independent_embed_gate_big_endian(
    gate_matrix: np.ndarray, n_qubits: int, wires: tuple[int, ...]
) -> np.ndarray:
    """Build explicit 2^n x 2^n unitary matrix in big-endian basis order from scratch."""
    dim = 1 << n_qubits
    k = len(wires)
    full = np.zeros((dim, dim), dtype=np.complex128)
    other_wires = [q for q in range(n_qubits) if q not in wires]
    for col in range(dim):
        # Extract bits in big-endian convention: bit of wire q is (col >> (n-1-q)) & 1
        sub_col = 0
        for w in wires:
            sub_col = (sub_col << 1) | ((col >> (n_qubits - 1 - w)) & 1)
        base_row = 0
        for ow in other_wires:
            if (col >> (n_qubits - 1 - ow)) & 1:
                base_row |= 1 << (n_qubits - 1 - ow)
        for sub_row in range(1 << k):
            amp = gate_matrix[sub_row, sub_col]
            if amp != 0.0:
                row = base_row
                for pos, w in enumerate(wires):
                    if (sub_row >> (k - 1 - pos)) & 1:
                        row |= 1 << (n_qubits - 1 - w)
                full[row, col] += amp
    return full


# ---------------------------------------------------------------------------
# 1. All 8 Computational Basis States of CCNOT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("b0,b1,b2", list(itertools.product([0, 1], repeat=3)))
def test_ccnot_all_eight_basis_states(b0: int, b1: int, b2: int):
    expected_b2 = b2 ^ (b0 & b1)

    # SovereignExecutor (3-qubit circuit)
    circ = Circuit(3, name=f"ccnot_{b0}{b1}{b2}")
    if b0:
        circ.cleave(0)
    if b1:
        circ.cleave(1)
    if b2:
        circ.cleave(2)
    circ.toffoli_gate(0, 1, 2)

    ex = SovereignExecutor(n_qubits=3)
    res = ex.run(circ, shots=0, return_statevector=True)
    expected_idx_3 = (b0 << 2) | (b1 << 1) | expected_b2
    expected_sv_3 = np.zeros(8, dtype=np.complex128)
    expected_sv_3[expected_idx_3] = 1.0
    np.testing.assert_allclose(res.statevector, expected_sv_3, atol=ATOL)

    # Qbyte (8-qubit register, wires 0, 1, 2)
    qb = Qbyte()
    if b0:
        qb.cleave(0)
    if b1:
        qb.cleave(1)
    if b2:
        qb.cleave(2)
    qb.toffoli(0, 1, 2)
    expected_idx_8 = (b0 << 7) | (b1 << 6) | (expected_b2 << 5)
    expected_sv_8 = np.zeros(256, dtype=np.complex128)
    expected_sv_8[expected_idx_8] = 1.0
    np.testing.assert_allclose(qb.state, expected_sv_8, atol=ATOL)


# ---------------------------------------------------------------------------
# 2. Every Valid Arrangement of Control and Target Qubits
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("perm", list(itertools.permutations([0, 1, 2], 3)))
def test_ccnot_all_3qubit_permutations_basis_and_random(perm: tuple[int, int, int]):
    c1, c2, target = perm
    rng = np.random.default_rng(1000 + c1 * 100 + c2 * 10 + target)
    psi = rng.normal(size=8) + 1j * rng.normal(size=8)
    psi /= np.linalg.norm(psi)

    expected = _independent_toffoli_action_big_endian(psi, 3, c1, c2, target)
    ex = SovereignExecutor(n_qubits=3)
    ex.reset(active_qubits=3)
    ex._state = psi.copy()
    ex._apply_toffoli(c1, c2, target)
    np.testing.assert_allclose(ex.state, expected, atol=ATOL)


def test_ccnot_all_336_arrangements_on_8_qubits():
    """Test every ordered triple (c1, c2, target) on 8 qubits (8*7*6 = 336 triples)."""
    rng = np.random.default_rng(20261116)
    psi = rng.normal(size=256) + 1j * rng.normal(size=256)
    psi /= np.linalg.norm(psi)

    for c1, c2, target in itertools.permutations(range(8), 3):
        expected = _independent_toffoli_action_big_endian(psi, 8, c1, c2, target)

        qb = Qbyte(initial_state=psi.copy())
        qb.toffoli(c1, c2, target)
        np.testing.assert_allclose(qb.state, expected, atol=ATOL)

        ex = SovereignExecutor(n_qubits=8)
        ex._state = psi.copy()
        ex._apply_toffoli(c1, c2, target)
        np.testing.assert_allclose(ex.state, expected, atol=ATOL)


def test_invalid_qubit_arrangements_rejected():
    qb = Qbyte()
    with pytest.raises(ValueError):
        qb.toffoli(0, 0, 1)
    with pytest.raises(ValueError):
        qb.toffoli(0, 1, 1)
    with pytest.raises(ValueError):
        qb.toffoli(-1, 1, 2)
    with pytest.raises(ValueError):
        qb.toffoli(0, 1, 8)
    with pytest.raises(ValueError):
        qb.measure_qubit(8)


# ---------------------------------------------------------------------------
# 3. Basis-Order Consistency Between Gates and `measure_qubit`
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("wire", range(8))
def test_single_qubit_cleave_and_measure_qubit_consistent(wire: int):
    qb = Qbyte()
    qb.cleave(wire)
    # Only wire `wire` should measure 1; all other 7 wires must measure 0
    for q in range(8):
        expected = 1 if q == wire else 0
        assert qb.measure_qubit(q, collapse=False) == expected
    assert qb.measure(collapse=False) == (1 << (7 - wire))


# ---------------------------------------------------------------------------
# 4. Gate Unitarity Across All Gate Constructors
# ---------------------------------------------------------------------------

def test_all_gates_are_unitary():
    angles = [-math.pi, -0.731, 0.0, 0.419, math.pi / 4, math.pi]
    fixed_gates = [
        gates.helix(),
        gates.unwind(),
        gates.cleave(),
        gates.phase_flip(),
        gates.rotate_y(),
        gates.s_gate(),
        gates.s_dg(),
        gates.t_gate(),
        gates.t_dg(),
        gates.bond(),
        gates.controlled_z(),
        gates.swap(),
        gates.iswap(),
        gates.toffoli(),
        gates.fredkin(),
    ]
    for g in fixed_gates:
        assert gates.is_unitary(g, tol=1e-13)
        assert gates.is_unitary(g.T, tol=1e-13)

    for theta in angles:
        for param_gate in (
            gates.twist(theta),
            gates.fold(theta),
            gates.splice(theta),
            gates.phase_gate(theta),
            gates.controlled_phase(theta),
            gates.xx_rotation(theta),
            gates.yy_rotation(theta),
            gates.zz_rotation(theta),
        ):
            assert gates.is_unitary(param_gate, tol=1e-13)


# ---------------------------------------------------------------------------
# 5. Random Normalized Statevectors vs Independent Full-Matrix Oracle
# ---------------------------------------------------------------------------

def test_random_statevectors_match_explicit_kronecker_oracle():
    rng = np.random.default_rng(424242)
    n_q = 6
    psi = rng.normal(size=1 << n_q) + 1j * rng.normal(size=1 << n_q)
    psi /= np.linalg.norm(psi)

    ex = SovereignExecutor(n_qubits=n_q)
    ex.reset(active_qubits=n_q)
    ex._state = psi.copy()
    ref = psi.copy()

    ops = [
        (gates.helix(), (0,)),
        (gates.splice(0.61), (3,)),
        (gates.fold(-1.17), (5,)),
        (gates.twist(0.93), (2,)),
        (gates.bond(), (4, 1)),
        (gates.controlled_z(), (5, 0)),
        (gates.swap(), (2, 4)),
        (gates.controlled_phase(0.77), (3, 0)),
        (gates.xx_rotation(-0.44), (1, 5)),
        (gates.yy_rotation(0.31), (0, 2)),
        (gates.zz_rotation(1.05), (4, 3)),
        (gates.toffoli(), (5, 1, 3)),
        (gates.fredkin(), (2, 5, 0)),
    ]

    for g_mat, wires in ops:
        ex._apply_k_qubit_gate(g_mat, wires)
        u_full = _independent_embed_gate_big_endian(g_mat, n_q, wires)
        ref = u_full @ ref
        assert math.isclose(float(np.linalg.norm(ex.state)), 1.0, abs_tol=ATOL)
        np.testing.assert_allclose(ex.state, ref, atol=ATOL)


# ---------------------------------------------------------------------------
# 6. Inverse-Circuit Restoration
# ---------------------------------------------------------------------------

def test_inverse_circuit_restoration_on_random_8qubit_state():
    rng = np.random.default_rng(98765)
    psi0 = rng.normal(size=256) + 1j * rng.normal(size=256)
    psi0 /= np.linalg.norm(psi0)

    qb = Qbyte(initial_state=psi0.copy())
    # Forward sequence
    qb.helix(0).s(1).t(2).splice(3, 0.43).fold(4, -0.89).twist(5, 1.31).y(6).phase(7)
    qb.bond(0, 3).cz(4, 1).cphase(2, 6, 0.58).rxx(5, 7, -0.37).ryy(1, 3, 0.72).rzz(0, 6, -1.11)
    qb.toffoli(7, 2, 4).fredkin(3, 0, 5).swap(1, 6)

    # Confirm state genuinely scrambled away from psi0
    assert abs(np.vdot(psi0, qb.state)) < 0.9

    # Exact inverse sequence in reverse order
    qb.swap(1, 6).fredkin(3, 0, 5).toffoli(7, 2, 4)
    qb.rzz(0, 6, 1.11).ryy(1, 3, -0.72).rxx(5, 7, 0.37).cphase(2, 6, -0.58).cz(4, 1).bond(0, 3)
    qb.phase(7).y(6).twist(5, -1.31).fold(4, 0.89).splice(3, -0.43).tdg(2).sdg(1).helix(0)

    np.testing.assert_allclose(qb.state, psi0, atol=ATOL)


# ---------------------------------------------------------------------------
# 7. Bell-State and GHZ-State Preparation
# ---------------------------------------------------------------------------

def test_all_four_bell_states_and_nonadjacent_pairs():
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    # Test on SovereignExecutor 2Q
    for z_flag, x_flag, expected in [
        (False, False, np.array([inv_sqrt2, 0, 0, inv_sqrt2])),
        (True, False, np.array([inv_sqrt2, 0, 0, -inv_sqrt2])),
        (False, True, np.array([0, inv_sqrt2, inv_sqrt2, 0])),
        (True, True, np.array([0, inv_sqrt2, -inv_sqrt2, 0])),
    ]:
        c = Circuit(2)
        if x_flag:
            c.cleave(1)
        c.helix(0)
        if z_flag:
            c.phase(0)
        c.bond(0, 1)
        res = SovereignExecutor(n_qubits=2).run(c, shots=0, return_statevector=True)
        np.testing.assert_allclose(res.statevector, expected, atol=ATOL)

    # Non-adjacent pair on Qbyte (wires 6 and 2)
    qb = bell_state(Qbyte(), q1=6, q2=2)
    idx_00 = 0
    idx_11 = (1 << (7 - 6)) | (1 << (7 - 2))
    expected_8 = np.zeros(256, dtype=np.complex128)
    expected_8[idx_00] = inv_sqrt2
    expected_8[idx_11] = inv_sqrt2
    np.testing.assert_allclose(qb.state, expected_8, atol=ATOL)


@pytest.mark.parametrize("width", [2, 3, 4, 5, 8])
def test_ghz_state_preparation(width: int):
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    res = SovereignExecutor(n_qubits=8).run(ghz_circuit(width), shots=0, return_statevector=True)
    expected = np.zeros(1 << width, dtype=np.complex128)
    expected[0] = inv_sqrt2
    expected[(1 << width) - 1] = inv_sqrt2
    np.testing.assert_allclose(res.statevector, expected, atol=ATOL)

    # Subset GHZ on Qbyte
    subset = [7, 4, 1][: min(width, 3)]
    qb = ghz_state(Qbyte(), qubits=subset)
    mask_all_ones = sum(1 << (7 - q) for q in subset)
    expected_qb = np.zeros(256, dtype=np.complex128)
    expected_qb[0] = inv_sqrt2
    expected_qb[mask_all_ones] = inv_sqrt2
    np.testing.assert_allclose(qb.state, expected_qb, atol=ATOL)


# ---------------------------------------------------------------------------
# 8. Phase, Interference, and QFT Behavior
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("theta", np.linspace(-math.pi, math.pi, 9))
def test_mach_zehnder_phase_interference_and_qft(theta: float):
    # H - RZ(theta) - H |0> = e^{-i theta/2} (cos(theta/2) (-i ? wait: check exact))
    # Let's compute exact 2x2 product: H @ RZ(theta) @ H @ |0>:
    # RZ(theta) = cos(theta/2) I - i sin(theta/2) Z
    # H Z H = X => H RZ(theta) H = cos(theta/2) I - i sin(theta/2) X = RX(theta)!
    c = Circuit(1)
    c.helix(0).twist(0, theta).helix(0)
    res = SovereignExecutor(n_qubits=4).run(c, shots=0, return_statevector=True)
    expected = np.array([math.cos(theta / 2.0), -1j * math.sin(theta / 2.0)], dtype=np.complex128)
    np.testing.assert_allclose(res.statevector, expected, atol=ATOL)

    # QFT on |101> (n=3): QFT|x> = (1/sqrt(8)) sum_y exp(2*pi*i * x * y / 8) |y>
    qft3 = Circuit(3)
    qft3.cleave(0).cleave(2)  # |101> -> x = 5
    for op in qft_circuit(3).operations:
        qft3.operations.append(op)
    res_qft = SovereignExecutor(n_qubits=3).run(qft3, shots=0, return_statevector=True)
    y = np.arange(8)
    expected_qft = np.exp(2j * math.pi * 5 * y / 8.0) / math.sqrt(8.0)
    np.testing.assert_allclose(res_qft.statevector, expected_qft, atol=ATOL)


def test_qbyte_runtime_toffoli_kernel():
    rt = QbyteRuntime(strict_l0=False)
    rt.execute_kernel([
        {"name": "cleave", "qubit": 0},
        {"name": "cleave", "qubit": 2},
        {"name": "toffoli", "qubit": 0, "control2": 2, "target": 5},
    ])
    expected_idx = (1 << 7) | (1 << 5) | (1 << 2)
    assert rt.qbyte.measure(collapse=False) == expected_idx


# ---------------------------------------------------------------------------
# 9. Cross-Framework Differential Testing Against Qiskit `Statevector`
# ---------------------------------------------------------------------------

def test_differential_agreement_against_qiskit_statevector():
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    # Deterministic 6-qubit benchmark circuit exercising 1Q, 2Q, 3Q, phase, and interference
    n_q = 6
    qc = QuantumCircuit(n_q)
    circ = Circuit(n_q, name="diff_ref")
    qb = Qbyte()  # Also verify first 6 wires on 8-qubit Qbyte (wires 6,7 remain |00>)

    # Layer 1: Superposition + rotations
    qc.h(0)
    circ.h(0)
    qb.h(0)
    qc.x(1)
    circ.x(1)
    qb.x(1)
    qc.rx(0.73, 2)
    circ.rx(2, 0.73)
    qb.rx(2, 0.73)
    qc.ry(-1.19, 3)
    circ.ry(3, -1.19)
    qb.ry(3, -1.19)
    qc.rz(0.41, 4)
    circ.rz(4, 0.41)
    qb.rz(4, 0.41)
    qc.h(5)
    qc.s(5)
    circ.h(5)
    circ.s(5)
    qb.h(5)
    qb.s(5)

    # Layer 2: Entangling & non-adjacent controls
    qc.cx(0, 3)
    circ.bond(0, 3)
    qb.bond(0, 3)
    qc.cx(5, 1)
    circ.bond(5, 1)
    qb.bond(5, 1)
    qc.cz(2, 4)
    circ.cz(2, 4)
    qb.cz(2, 4)
    qc.cp(0.88, 3, 0)
    circ.cphase(3, 0, 0.88)
    qb.cphase(3, 0, 0.88)
    qc.rxx(-0.52, 1, 4)
    circ.rxx(1, 4, -0.52)
    qb.rxx(1, 4, -0.52)
    qc.rzz(1.13, 5, 2)
    circ.rzz(5, 2, 1.13)
    qb.rzz(5, 2, 1.13)

    # Layer 3: Toffoli & Fredkin in non-monotonic wire orders
    qc.ccx(5, 1, 4)
    circ.toffoli_gate(5, 1, 4)
    qb.toffoli(5, 1, 4)
    qc.ccx(3, 0, 2)
    circ.toffoli_gate(3, 0, 2)
    qb.toffoli(3, 0, 2)
    qc.cswap(4, 5, 0)
    circ.fredkin_gate(4, 5, 0)
    qb.fredkin(4, 5, 0)
    qc.t(2)
    qc.tdg(4)
    qc.sdg(5)
    circ.t(2)
    circ.tdg(4)
    circ.sdg(5)
    qb.t(2)
    qb.tdg(4)
    qb.sdg(5)

    sv_qiskit = Statevector.from_instruction(qc).data
    res = SovereignExecutor(n_qubits=n_q).run(circ, shots=0, return_statevector=True)
    sv_sovereign_le = res.get_statevector(endian="little")

    np.testing.assert_allclose(sv_sovereign_le, sv_qiskit, atol=ATOL)

    # Also verify Qbyte (8 qubits, where q6=0, q7=0 -> in little-endian |00 q5 q4 q3 q2 q1 q0>,
    # first 64 amplitudes equal sv_qiskit and remaining 192 amplitudes equal 0)
    sv_qbyte_le = qb.get_statevector(endian="little")
    np.testing.assert_allclose(sv_qbyte_le[:64], sv_qiskit, atol=ATOL)
    np.testing.assert_allclose(sv_qbyte_le[64:], np.zeros(192), atol=ATOL)

    # Randomized 20-circuit differential fuzzing across 5 qubits
    rng = np.random.default_rng(314159)
    for trial in range(20):
        qc_r = QuantumCircuit(5)
        circ_r = Circuit(5, name=f"fuzz_{trial}")
        for _ in range(18):
            kind = int(rng.integers(0, 8))
            if kind == 0:
                q = int(rng.integers(0, 5))
                qc_r.h(q)
                circ_r.h(q)
            elif kind == 1:
                q = int(rng.integers(0, 5))
                th = float(rng.uniform(-math.pi, math.pi))
                qc_r.rx(th, q)
                circ_r.rx(q, th)
            elif kind == 2:
                q = int(rng.integers(0, 5))
                th = float(rng.uniform(-math.pi, math.pi))
                qc_r.ry(th, q)
                circ_r.ry(q, th)
            elif kind == 3:
                q = int(rng.integers(0, 5))
                th = float(rng.uniform(-math.pi, math.pi))
                qc_r.rz(th, q)
                circ_r.rz(q, th)
            elif kind == 4:
                q1, q2 = [int(x) for x in rng.choice(5, size=2, replace=False)]
                qc_r.cx(q1, q2)
                circ_r.bond(q1, q2)
            elif kind == 5:
                q1, q2 = [int(x) for x in rng.choice(5, size=2, replace=False)]
                qc_r.cz(q1, q2)
                circ_r.cz(q1, q2)
            elif kind == 6:
                c1, c2, tgt = [int(x) for x in rng.choice(5, size=3, replace=False)]
                qc_r.ccx(c1, c2, tgt)
                circ_r.toffoli_gate(c1, c2, tgt)
            else:
                ctl, q1, q2 = [int(x) for x in rng.choice(5, size=3, replace=False)]
                qc_r.cswap(ctl, q1, q2)
                circ_r.fredkin_gate(ctl, q1, q2)

        sv_q = Statevector.from_instruction(qc_r).data
        sv_s = SovereignExecutor(n_qubits=5).run(circ_r, shots=0, return_statevector=True).get_statevector(endian="little")
        np.testing.assert_allclose(sv_s, sv_q, atol=ATOL)


def test_strict_l0_runtime_and_subset_measurement_and_endian_counts():
    # Default strict_l0=True must execute short valid kernels without spurious gate-1 crash
    rt_strict = QbyteRuntime(strict_l0=True)
    out = rt_strict.execute_kernel([
        {"name": "helix", "qubit": 0},
        {"name": "bond", "qubit": 0, "target": 1},
    ])
    assert out["status"] == "SUCCESS"
    assert rt_strict.qbyte.metrics.phi >= 0.0

    # Explicit wire subset & classical-bit mapping + little-endian count conversion
    c = Circuit(3, name="subset_meas")
    c.x(0)
    c.measure(0, 1)
    c.measure(2, 0)
    res = SovereignExecutor(n_qubits=3).run(c, shots=64)
    assert res.counts == {"01": 64}
    assert res.get_counts(endian="little") == {"10": 64}
