"""Verification of OSIRIS-BT2T-001 milestone 1 against independent references.

References, each independent of the circuit construction:
  R1  group axioms checked by exhaustive enumeration on exact integers (closure, identity, associativity,
      the Hamilton relations i^2 = j^2 = k^2 = ijk = -1);
  R2  the inverse map from searching the multiplication table, compared with quaternion conjugation;
  R3  a classical bit-level simulation of the gate list on all 32 basis inputs;
  R4  the circuit's full 32x32 matrix compared with the permutation matrix EXACTLY (phases included);
  R5  seeded random complex superpositions (valid codewords only, and all 32 states) evolved through the
      circuit and compared with the permuted amplitudes, which catches relative-phase errors a basis test
      would miss;
  R6  local proxy compilations (Qiskit to rz/sx/x/cz on all-to-all and a heavy-hex fragment; pytket rebase
      to the H2 native set) checked for unitary equivalence up to global phase. PROXY: not provider output.

``prior_prototype_findings`` measures the unvalidated earlier harness against the conditions any correct
inversion must meet (``prior`` is kept out of ``checks``: it describes the old code, not this one).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

import bt2t

SEED = 20261122


def permutation_matrix(perm: List[int]) -> np.ndarray:
    m = np.zeros((len(perm), len(perm)), dtype=complex)
    for x, y in enumerate(perm):
        m[y, x] = 1.0  # column x is the image of basis state x
    return m


def group_checks() -> Dict[str, bool]:
    els = bt2t.elements()
    idx = bt2t.index_of()
    one, minus_one = (2, 0, 0, 0), (-2, 0, 0, 0)
    i, j, k = (0, 2, 0, 0), (0, 0, 2, 0), (0, 0, 0, 2)
    m = bt2t.qmul
    try:
        bt2t.multiplication_table()
        closed = True
    except ArithmeticError:
        closed = False
    return {
        "group_24_distinct_elements": len(set(els)) == 24,
        "group_all_unit_norm": all(sum(c * c for c in g) == 4 for g in els),
        "group_closed": closed,
        "group_identity_is_index_0": all(m(els[0], g) == g == m(g, els[0]) for g in els),
        "group_hamilton_relations": (m(i, i) == m(j, j) == m(k, k) == m(m(i, j), k) == minus_one
                                     and m(i, j) == k and m(j, k) == i and m(k, i) == j and m(j, i) == (0, 0, 0, -2)),
        "group_associative_exhaustive": bt2t.all_products_associative(),
        "group_index_roundtrip": all(idx[g] == n for n, g in enumerate(els)) and els[1] == minus_one,
    }


def map_checks() -> Dict[str, bool]:
    els, idx = bt2t.elements(), bt2t.index_of()
    table_inv = bt2t.inverse_from_table(bt2t.multiplication_table())
    conj_inv = [idx[bt2t.conjugate(g)] for g in els]
    perm = bt2t.inversion_permutation()
    fixed = {x for x in range(bt2t.N_STATES) if perm[x] == x}
    return {
        "map_table_inverse_equals_conjugate": table_inv == conj_inv,
        "map_is_bijection_on_32": sorted(perm) == list(range(32)),
        "map_is_involution": all(perm[perm[x]] == x for x in range(32)),
        "map_fixed_points_are_pm1_and_unused": fixed == {0, 1} | set(bt2t.UNUSED),
        "map_valid_set_preserved": all(perm[x] < 24 for x in range(24)),
        "map_11_transpositions": len(bt2t.transpositions(perm)) == 11,
    }


def circuit_checks() -> Dict[str, object]:
    perm = bt2t.inversion_permutation()
    gates = bt2t.inversion_gate_list()
    checks: Dict[str, object] = {
        "circuit_classical_sim_all_32_inputs": all(bt2t.apply_gate_list_classically(gates, x) == perm[x]
                                                   for x in range(32)),
    }
    from qiskit.quantum_info import Operator, Statevector
    qc = bt2t.inversion_circuit()
    u = Operator(qc).data
    p = permutation_matrix(perm)
    err = float(np.max(np.abs(u - p)))
    checks["circuit_matrix_equals_permutation_exactly"] = err < 1e-9
    checks["_circuit_matrix_max_abs_error"] = err
    checks["circuit_squared_is_identity"] = bool(np.allclose(u @ u, np.eye(32), atol=1e-9))
    us = Operator(bt2t.inversion_circuit_structured()).data
    serr = float(np.max(np.abs(us - p)))
    checks["structured_matrix_equals_permutation_exactly"] = serr < 1e-9
    checks["_structured_matrix_max_abs_error"] = serr

    rng = np.random.default_rng(SEED)
    worst = 0.0
    for support in (range(24), range(32)):
        for _ in range(16):
            amp = np.zeros(32, dtype=complex)
            s = list(support)
            amp[s] = rng.normal(size=len(s)) + 1j * rng.normal(size=len(s))
            amp /= np.linalg.norm(amp)
            out = Statevector(amp).evolve(qc).data
            expected = np.zeros(32, dtype=complex)
            for x in range(32):
                expected[perm[x]] = amp[x]
            worst = max(worst, float(np.max(np.abs(out - expected))))
    checks["circuit_superpositions_exact_with_phases"] = worst < 1e-9
    checks["_superposition_max_abs_error"] = worst
    return checks


HEAVY_HEX_5Q = [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [2, 4], [4, 2]]  # a T-shaped heavy-hex fragment


def proxy_compilations() -> List[Dict[str, object]]:
    from qiskit import qasm2, transpile
    from qiskit.quantum_info import Operator
    from qiskit.transpiler import CouplingMap
    ref = Operator(bt2t.inversion_circuit())
    out = []
    for name, qc in (("generic", bt2t.inversion_circuit()), ("structured", bt2t.inversion_circuit_structured())):
      for label, cmap in (("all_to_all", None), ("heavy_hex_fragment_T5", CouplingMap(HEAVY_HEX_5Q))):
        for level in (1, 3):
            t = transpile(qc, basis_gates=["rz", "sx", "x", "cz"], coupling_map=cmap,
                          optimization_level=level, seed_transpiler=SEED)
            equiv = Operator.from_circuit(t).equiv(ref)
            ops = dict(t.count_ops())
            out.append({"kind": "PROXY qiskit transpile", "circuit": name, "target": label,
                        "optimization_level": level,
                        "seed": SEED, "two_qubit_gates": {"cz": int(ops.get("cz", 0))},
                        "depth": t.depth(), "size": t.size(), "equivalent_up_to_global_phase": bool(equiv)})
    try:
        from pytket.circuit import OpType
        from pytket.passes import AutoRebase, FullPeepholeOptimise
        from pytket.qasm import circuit_from_qasm_str
    except ImportError:
        out.append({"kind": "PROXY pytket H2-native rebase", "status": "SKIPPED: pytket not installed"})
        return out
    for (name, qc), opt in [((n, c), o) for n, c in (("generic", bt2t.inversion_circuit()),
                                                    ("structured", bt2t.inversion_circuit_structured()))
                             for o in (False, True)]:
        tk = circuit_from_qasm_str(qasm2.dumps(qc.decompose(reps=4)))
        if opt:
            FullPeepholeOptimise().apply(tk)
        AutoRebase({OpType.ZZPhase, OpType.PhasedX, OpType.Rz}).apply(tk)
        # pytket orders basis states with qubit 0 as the MOST significant bit; convert to Qiskit order.
        u_tk = tk.get_unitary()
        r = [bt2t.reverse_bits(x) for x in range(32)]
        u_q = u_tk[np.ix_(r, r)]
        a = ref.data
        phase = np.vdot(a.flatten(), u_q.flatten())
        equiv = abs(abs(phase) - 32.0) < 1e-6 and np.allclose(u_q * np.conj(phase / abs(phase)), a, atol=1e-6)
        out.append({"kind": "PROXY pytket H2-native rebase (local pytket, not Nexus)", "circuit": name,
                    "FullPeepholeOptimise": opt,
                    "two_qubit_gates": {"ZZPhase": int(tk.n_gates_of_type(OpType.ZZPhase))},
                    "depth": tk.depth(), "size": tk.n_gates, "equivalent_up_to_global_phase": bool(equiv)})
    return out


def prior_prototype_findings() -> Dict[str, object]:
    """The earlier harness's 'inversion' (build_bt_inversion_circuit), reconstructed gate for gate.

    Its stated encoding: q0 sign bit, (q1, q2) Q8 index, (q3, q4) Z3 coset index, so coset value 3
    (q3 = q4 = 1) is unused and 24 states are valid. A correct inversion on any such encoding must map the
    valid set to itself, be an involution there, and fix exactly 2 valid elements (+-1).
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator, Statevector
    prep = QuantumCircuit(5)
    for q in range(5):
        prep.h(q)
    prep.cx(0, 1); prep.cx(2, 3); prep.cx(3, 4)  # noqa: E702
    inv = QuantumCircuit(5)
    inv.cx(1, 0); inv.cx(2, 0); inv.ccx(1, 2, 0); inv.swap(3, 4); inv.ccx(3, 4, 0)  # noqa: E702
    inv.cx(3, 1); inv.cx(4, 2)  # noqa: E702
    support = int(np.sum(np.abs(Statevector(prep).data) ** 2 > 1e-12))
    u = Operator(inv).data
    perm = [int(np.argmax(np.abs(u[:, x]))) for x in range(32)]
    valid = [x for x in range(32) if not (((x >> 3) & 1) and ((x >> 4) & 1))]
    return {
        "prep_support_states": support,
        "prep_restricted_to_24": support == 24,
        "inversion_is_involution": all(perm[perm[x]] == x for x in range(32)),
        "inversion_preserves_valid_set": all(perm[x] in valid for x in valid),
        "valid_fixed_points": sum(1 for x in valid if perm[x] == x),
        "meets_necessary_conditions": (all(perm[perm[x]] == x for x in range(32))
                                       and all(perm[x] in valid for x in valid)
                                       and sum(1 for x in valid if perm[x] == x) == 2),
    }


def run_all() -> Dict[str, object]:
    checks: Dict[str, object] = {}
    checks.update(group_checks())
    checks.update(map_checks())
    circ = circuit_checks()
    metrics = {k: v for k, v in circ.items() if k.startswith("_")}
    checks.update({k: v for k, v in circ.items() if not k.startswith("_")})
    comps = proxy_compilations()
    checks["proxy_compilations_all_equivalent"] = all(c.get("equivalent_up_to_global_phase", False)
                                                      for c in comps if "status" not in c)
    return {"checks": checks, "metrics": metrics, "compilations": comps, "prior": prior_prototype_findings()}


if __name__ == "__main__":
    import json
    print(json.dumps(run_all(), indent=1, default=str))
