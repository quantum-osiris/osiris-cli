"""Characterization of what dna::}{::lang (dnalang-core) does TODAY with an agent-shaped key-value
gene. These tests pin current behavior so RFC-0001 cannot over-claim; they do not add or imply
agent-gene support. Skipped when dnalang-core is not importable (CI core install)."""

import os
import sys

import pytest


def _dnalang():
    try:
        import dnalang  # noqa: F401
    except ImportError:
        for cand in (os.environ.get("OSIRIS_DNALANG_CORE", ""), os.path.expanduser("~/dnalang-core")):
            if cand and os.path.isdir(os.path.join(cand, "dnalang")):
                sys.path.insert(0, cand)
                break
    pytest.importorskip("dnalang")
    from dnalang import parser, rules_ir, sema
    return parser, sema, rules_ir


AGENT_FIELDS = '    role: "research"\n    capabilities: ["READ_ONLY_ANALYSIS"]\n    budget_tokens: 4096\n'
RULE = 'organism Agents {\n  gene Researcher {\n    condition: "1#"\n    action: "(emit found)"\n%s  }\n}\n'
DATA_ONLY = "organism Agents {\n  gene Researcher {\n" + AGENT_FIELDS + "  }\n}\n"


def test_agent_fields_parse_and_are_preserved_on_the_ast():
    parser, _, _ = _dnalang()
    ast = parser.parse(DATA_ONLY)
    assert sorted(ast.kv_genes[0].fields) == ["budget_tokens", "capabilities", "role"]


def test_data_only_gene_passes_check_with_only_a_warning_and_lowers_to_nothing():
    parser, sema, rules_ir = _dnalang()
    ast = parser.parse(DATA_ONLY)
    d = sema.check(ast)
    assert d.errors == [] and any("data only" in w for w in d.warnings)
    assert rules_ir.lower_all(ast) == {"circuit": None, "rules": None, "regulation": None}


def test_unknown_agent_fields_do_not_change_the_canonical_hash():
    """Consequence: today a compiled organism's sha256 does NOT cover an agent's capabilities or
    budget. Identity for an agent genome therefore cannot rely on the language hash."""
    parser, _, rules_ir = _dnalang()
    with_f = rules_ir.lower_all(parser.parse(RULE % AGENT_FIELDS))["rules"].sha256()
    without = rules_ir.lower_all(parser.parse(RULE % ""))["rules"].sha256()
    assert with_f == without


def test_no_capability_validation_exists_today():
    parser, sema, _ = _dnalang()
    bogus = DATA_ONLY.replace("READ_ONLY_ANALYSIS", "ROOT_EVERYTHING")
    assert sema.check(parser.parse(bogus)).errors == []   # accepted: the language does not know capabilities
