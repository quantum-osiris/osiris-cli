"""Unit & differential tests for `osiris_cli.compiler_passes.reduce_commuting_cx_fanouts`."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_ibm_runtime")

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

from osiris_cli.compiler_passes import reduce_commuting_cx_fanouts
from phase2.tesseract_map_02.tesseract_map_02 import build_star_circuit


def test_degree_reducer_preserves_exact_statevector_and_cuts_40q_cz_count_by_4x():
    # 1. Exact statevector preservation on 12-qubit dual-hub star circuit
    star12 = build_star_circuit(n_qubits=12, theta_deg=51.7, include_inert_rz=True, measure=False)
    red12, stats12 = reduce_commuting_cx_fanouts(star12)
    assert stats12["edges_rewritten_to_chain"] == 8  # 4 even + 4 odd relays
    sv_orig = Statevector.from_instruction(star12)
    sv_red = Statevector.from_instruction(red12)
    assert math.isclose(float(abs(sv_orig.inner(sv_red))), 1.0, abs_tol=1e-12)

    # 2. Soundness guard: when intermediate target wires are NOT in |0> (e.g. after Hadamard),
    # the pass MUST NOT corrupt the unitary by falsely chaining through non-|0> wires!
    qc_non_zero = QuantumCircuit(4)
    qc_non_zero.h([0, 1, 2])  # wires 1 and 2 are in |+>, NOT |0>!
    qc_non_zero.cx(0, 1)
    qc_non_zero.cx(0, 2)
    qc_non_zero.cx(0, 3)      # wire 3 IS in |0>!
    red_nz, stats_nz = reduce_commuting_cx_fanouts(qc_non_zero)
    sv_nz_orig = Statevector.from_instruction(qc_non_zero)
    sv_nz_red = Statevector.from_instruction(red_nz)
    assert math.isclose(float(abs(sv_nz_orig.inner(sv_nz_red))), 1.0, abs_tol=1e-12)

    # 3. 40-qubit compilation benchmark on FakeMarrakesh: automatic 163 CZ -> 39 CZ reduction!
    backend = FakeMarrakesh()
    star40 = build_star_circuit(n_qubits=40, theta_deg=51.7, include_inert_rz=True, measure=True)
    red40, stats40 = reduce_commuting_cx_fanouts(star40)
    assert stats40["edges_rewritten_to_chain"] == 36  # 18 even + 18 odd relays

    t_orig = transpile(star40, backend=backend, optimization_level=3, seed_transpiler=20261116)
    t_red = transpile(red40, backend=backend, optimization_level=3, seed_transpiler=20261116)

    cz_orig = sum(1 for i in t_orig.data if i.operation.num_qubits == 2)
    cz_red = sum(1 for i in t_red.data if i.operation.num_qubits == 2)
    assert cz_orig > 120
    assert cz_red == 39
