"""OSIRIS-BT2T-001 milestone 1: exact checks of 2T inversion (offline; no provider calls)."""

import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "experiments" / "bt2t_inversion"


def _load(name):
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bt2t = _load("bt2t")


# --- exact group and map (no quantum library needed) -------------------------------------------------

def test_group_axioms_hold_exhaustively():
    els = bt2t.elements()
    assert len(set(els)) == 24 and all(sum(c * c for c in g) == 4 for g in els)
    table = bt2t.multiplication_table()  # raises if not closed
    assert all(table[0][j] == j == table[j][0] for j in range(24))
    assert bt2t.all_products_associative()
    i, j, k = (0, 2, 0, 0), (0, 0, 2, 0), (0, 0, 0, 2)
    assert bt2t.qmul(i, j) == k and bt2t.qmul(j, i) == (0, 0, 0, -2)
    assert bt2t.qmul(bt2t.qmul(i, j), k) == (-2, 0, 0, 0)


def test_encoding_is_frozen():
    rec = bt2t.encoding_record()
    assert rec["encoding_version"] == "bt2t-enc-1"
    assert rec["elements"][8]["doubled_coords"] == [1, 1, 1, 1]          # (1 + i + j + k) / 2
    assert rec["elements"][23]["doubled_coords"] == [-1, -1, -1, -1]
    assert rec["fixed_points"] == [0, 1] + list(range(24, 32))
    # Changing the encoding, bit order or map must change this hash; update it only with a new version.
    assert rec["sha256"] == "3e3b02d13539daaf431af39056d031f777a785a556eb9da21ef4ea6097dfc4c5"


def test_inverse_from_table_matches_conjugation_and_is_an_involution():
    els, idx = bt2t.elements(), bt2t.index_of()
    inv = bt2t.inverse_from_table(bt2t.multiplication_table())
    assert inv == [idx[bt2t.conjugate(g)] for g in els]
    perm = bt2t.inversion_permutation()
    assert sorted(perm) == list(range(32))
    assert all(perm[perm[x]] == x for x in range(32))
    assert {x for x in range(32) if perm[x] == x} == {0, 1} | set(range(24, 32))
    assert len(bt2t.transpositions(perm)) == 11


def test_gate_list_maps_all_32_basis_inputs_classically():
    perm = bt2t.inversion_permutation()
    gates = bt2t.inversion_gate_list()
    assert [bt2t.apply_gate_list_classically(gates, x) for x in range(32)] == perm


def test_bit_reversal_helper():
    assert bt2t.reverse_bits(0b00001) == 0b10000
    assert all(bt2t.reverse_bits(bt2t.reverse_bits(x)) == x for x in range(32))


# --- circuit, exact including phases (needs qiskit) ---------------------------------------------------

@pytest.fixture(scope="module")
def verify():
    pytest.importorskip("qiskit")
    return _load("verify")


def test_circuit_matrices_equal_the_permutation_exactly(verify):
    np = pytest.importorskip("numpy")
    from qiskit.quantum_info import Operator
    p = verify.permutation_matrix(bt2t.inversion_permutation())
    for qc in (bt2t.inversion_circuit(), bt2t.inversion_circuit_structured()):
        assert np.max(np.abs(Operator(qc).data - p)) < 1e-9, qc.name  # phases included, not just support


def test_superpositions_with_relative_phases(verify):
    c = verify.circuit_checks()
    assert c["circuit_superpositions_exact_with_phases"] and c["circuit_classical_sim_all_32_inputs"]


def test_a_phase_error_is_caught_by_the_phase_tests(verify):
    """Positive control: a CZ that only adds a relative phase passes a basis-output test but must fail."""
    np = pytest.importorskip("numpy")
    from qiskit.quantum_info import Operator
    bad = bt2t.inversion_circuit_structured()
    bad.cz(0, 1)
    u = Operator(bad).data
    perm = bt2t.inversion_permutation()
    assert [int(np.argmax(np.abs(u[:, x]))) for x in range(32)] == perm      # basis outputs still right
    assert np.max(np.abs(u - verify.permutation_matrix(perm))) > 1.0           # phases are not


def test_proxy_compilation_stays_equivalent(verify):
    from qiskit import transpile
    from qiskit.quantum_info import Operator
    from qiskit.transpiler import CouplingMap
    qc = bt2t.inversion_circuit_structured()
    t = transpile(qc, basis_gates=["rz", "sx", "x", "cz"], coupling_map=CouplingMap(verify.HEAVY_HEX_5Q),
                  optimization_level=3, seed_transpiler=verify.SEED)
    assert Operator.from_circuit(t).equiv(Operator(bt2t.inversion_circuit()))


def test_prior_prototype_fails_the_necessary_conditions(verify):
    """The earlier, unvalidated harness: its preparation covers all 32 states and its 'inversion' is not
    an involution, so it is not group inversion under its own stated encoding."""
    f = verify.prior_prototype_findings()
    assert f["prep_support_states"] == 32 and not f["prep_restricted_to_24"]
    assert not f["inversion_is_involution"] and not f["meets_necessary_conditions"]
