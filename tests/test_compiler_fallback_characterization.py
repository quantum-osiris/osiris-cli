"""Characterization of osiris.compiler.OrganismConverter.compile as of main @ 9896f04a.

Pins what the code DOES today, including the defect that motivated bugfix/compiler-fallback-trace:
Parser._parse_organism skips every token inside an organism's braces, so organism bodies never
reach the IR and every body compiles to IRCompiler's default H/CX/MEASURE circuit. These tests
describe behavior; they do not endorse it. The strict-mode tests live in test_compiler_strict.py.
"""

import pytest

from osiris.compiler.converter import OrganismConverter
from osiris.compiler.dna_parser import Lexer, Parser

DEFAULT = ["H", "CX", "MEASURE"]


def ops(src):
    return [o.op_type.name for o in OrganismConverter().compile(src).operations]


def test_name_only_input_is_the_documented_convenience_api():
    # converter.py: "Preserve the documented convenience API for organism names"; also used by tests
    assert ops("BellState") == DEFAULT
    assert ops("this is not dna at all") == DEFAULT      # no marker chars -> treated as a name


def test_text_with_dna_markers_but_no_organism_raises():
    with pytest.raises(ValueError):
        OrganismConverter().compile("gene Y { role: z }")
    with pytest.raises(Exception):
        OrganismConverter().compile("organism {")         # pinned by tests/test_backends.py


def test_organism_body_is_never_parsed():
    p = Parser(Lexer("organism X { quantum_state { x 0; y 1; z 2; } }").tokenize()).parse()
    assert len(p) == 1 and p[0].quantum_state is None and p[0].genome is None


@pytest.mark.parametrize("src", [
    "organism X { }",
    "organism X { gene Y { role: \"z\" } }",
    "organism G { quantum_state { h 0; } }",
    "organism G { quantum_state { x 0; y 1; z 2; swap 0, 1; } }",
    "organism G { quantum_state { totally_unsupported_op 99; } }",
    "organism G { !!! garbage }",
])
def test_every_organism_body_compiles_to_the_default_circuit(src):
    assert ops(src) == DEFAULT


def test_the_converters_op_building_loop_is_unreachable_with_this_parser():
    # converter.compile contains `raise ValueError("Unsupported DNA-Lang operation")`, but it is
    # reachable only when organism.quantum_state is set, which this parser never does.
    assert ops("organism G { quantum_state { not_a_gate 0; } }") == DEFAULT


def test_loop_validation_accepts_a_genome_whose_body_names_a_forbidden_op():
    """The consequence for the evolutionary loop: SYS-025's capability allowlist is checked on
    the compiled IR, which never contains the genome's own operations."""
    from osiris_cli.dnalang.loop import CandidateEvolutionLoop
    loop = CandidateEvolutionLoop(provider=object(), lineage_ledger_path="/nonexistent/never-written")
    ok, ir, why = loop.parse_and_validate("organism Evil { quantum_state { exec_shell 0; } }")
    assert ok is True and [o.op_type.name for o in ir.operations] == DEFAULT and why == "Valid genome"
    ok2, _, _ = loop.parse_and_validate("organism Empty { }")
    assert ok2 is True


def test_two_different_genomes_yield_identical_circuits():
    a = OrganismConverter().compile("organism G { quantum_state { h 0; } }")
    b = OrganismConverter().compile("organism G { quantum_state { x 0; y 1; z 0; } }")
    assert [o.op_type for o in a.operations] == [o.op_type for o in b.operations]
    assert a.lineage_hash == b.lineage_hash   # lineage is name:qubits:len(ops) only
