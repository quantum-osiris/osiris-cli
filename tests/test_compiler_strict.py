"""Strict compile mode: the promotion path must never accept a genome merely because a default
circuit could be produced. Legacy (non-strict) behavior is pinned in
test_compiler_fallback_characterization.py and must stay unchanged."""

import pytest

from osiris.compiler.converter import OrganismConverter
from osiris_cli.dnalang.loop import CandidateEvolutionLoop


def strict_ops(src):
    c = OrganismConverter().compile(src, strict=True, optimize=False)
    return [(o.op_type.name, o.qubits) for o in c.operations]


def test_strict_compiles_the_genome_the_loop_actually_emits():
    assert strict_ops("organism Gen_1 { quantum_state { h 0; cx 0, 1; } }") == [
        ("H", [0]), ("CX", [0, 1]), ("MEASURE", [0, 1])]


def test_strict_circuit_depends_on_genome_content():
    a = strict_ops("organism G { quantum_state { h 0; } }")
    b = strict_ops("organism G { quantum_state { x 0; y 1; } }")
    assert a != b and a[0][0] == "H" and b[0][0] == "X"


@pytest.mark.parametrize("src", [
    "BellState",                                              # bare name is not a genome
    "this is not dna at all",
    "organism X { }",
    "organism X { quantum_state { } }",
    "organism X { gene Y { role: \"z\" } }",
    "organism G { quantum_state { exec_shell 0; } }",         # unknown op
    "organism G { quantum_state { h 0, 1; } }",               # wrong arity
    "organism G { quantum_state { cx 0, 0; } }",              # repeated qubit
    "organism G { quantum_state { h 99; } }",                 # out of strict range
    "organism G { quantum_state { h 0 } }",                   # missing ;
    "organism G { quantum_state { h 0; } } trailing junk",
    "organism A { quantum_state { h 0; } } organism B { quantum_state { h 0; } }",
    "organism G { quantum_state { h 0; } gene Y { } }",
])
def test_strict_rejects_what_it_cannot_honor(src):
    with pytest.raises((ValueError, SyntaxError)):
        OrganismConverter().compile(src, strict=True)


def test_legacy_mode_is_unchanged_for_the_same_inputs():
    for src in ("BellState", "organism X { }", "organism G { quantum_state { exec_shell 0; } }"):
        assert [o.op_type.name for o in OrganismConverter().compile(src).operations] == ["H", "CX", "MEASURE"]


def _loop(tmp_path):
    return CandidateEvolutionLoop(provider=object(), lineage_ledger_path=str(tmp_path / "lineage.jsonl"))


def test_loop_strict_validation_rejects_a_forbidden_or_malformed_genome(tmp_path):
    loop = _loop(tmp_path)
    for src in ("organism Evil { quantum_state { exec_shell 0; } }", "organism Empty { }", "BellState",
                "organism R { quantum_state { reset 0; } }"):     # RESET is not in ALLOWED_CAPABILITIES
        ok, ir, why = loop.parse_and_validate(src, strict=True)
        assert ok is False and ir is None, src
    ok, ir, why = loop.parse_and_validate("organism G { quantum_state { h 0; cx 0, 1; } }", strict=True)
    assert ok is True and [o.op_type.name for o in ir.operations][:2] == ["H", "CX"]


def test_run_cycle_cannot_promote_through_the_fallback(tmp_path):
    """A parent whose genome the strict compiler cannot honor yields a candidate that is refused at
    validation, so acceptance_fn is never called and nothing is promoted."""
    from osiris_cli.dnalang.loop import GenomeCandidate
    loop = _loop(tmp_path)
    called = []
    parent = GenomeCandidate("p0", None, "organism P { }")
    # mutate_parent_genome falls back to a valid default form when no quantum_state is found, so
    # force an unsupported candidate source directly through the validator the loop uses:
    loop.mutate_parent_genome = lambda *a, **k: "organism Gen_1 { quantum_state { exec_shell 0; } }"
    res = loop.run_cycle("x", lambda ir: called.append(1) or (True, 1.0), parent_candidate=parent)
    assert res["promoted"] is False and res["step"] == "validation" and called == []
    assert loop.current_champion is None


def test_run_cycle_still_promotes_a_valid_genome(tmp_path):
    loop = _loop(tmp_path)
    res = loop.run_cycle("start", lambda ir: (True, 0.5))
    assert res["promoted"] is True
