"""Executable dialect matrix: which implementation accepts which .dna form (docs/dialects/DIALECT_MATRIX.md).
Observed 2026-11-09. A change in any cell is a language-compatibility change and needs review."""

import importlib.util
import os
import sys

import pytest

from osiris.compiler.converter import OrganismConverter

D = os.path.join(os.path.dirname(__file__), "fixtures", "dialects")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(name):
    with open(os.path.join(D, name), encoding="utf-8") as f:
        return f.read()


def legacy(src):
    try:
        return "ops=" + ",".join(o.op_type.name for o in OrganismConverter().compile(src).operations)
    except Exception as e:
        return "ERR " + type(e).__name__


def strict(src):
    try:
        return "ops=" + ",".join(o.op_type.name for o in OrganismConverter().compile(src, strict=True).operations)
    except Exception as e:
        return "ERR " + type(e).__name__


# fixture -> (osiris.compiler legacy, osiris.compiler strict)
OSIRIS_COMPILER = {
    "bare_name.txt": ("ops=H,CX,MEASURE", "ERR SyntaxError"),
    "garbage.txt": ("ops=H,CX,MEASURE", "ERR SyntaxError"),
    "empty_organism.dna": ("ops=H,CX,MEASURE", "ERR SyntaxError"),
    "osiris_loop_form.dna": ("ops=H,CX,MEASURE", "ops=H,CX,MEASURE"),   # legacy ignores the body
    "currency_card_form.dna": ("ops=H,CX,MEASURE", "ERR SyntaxError"),
    "dnalang_core_rule_gene.dna": ("ops=H,CX,MEASURE", "ERR SyntaxError"),
    # legacy lexer has '#' comments only: a '//' comment containing the word "organism" breaks it
    "dnalang_core_bell.dna": ("ERR SyntaxError", "ERR SyntaxError"),
    "repo_organism_scimitar_sse_128.dna": ("ERR SyntaxError", "ERR SyntaxError"),
}

# fixture -> dnalang-core parse+check
DNALANG_CORE = {
    "bare_name.txt": "ERR ParseError", "garbage.txt": "ERR ParseError",
    "empty_organism.dna": "ERR ParseError", "osiris_loop_form.dna": "ERR ParseError",
    "currency_card_form.dna": "ERR LexError", "repo_organism_scimitar_sse_128.dna": "ERR LexError",
    "dnalang_core_bell.dna": "ok errors=0 warnings=0", "dnalang_core_rule_gene.dna": "ok errors=0 warnings=0",
}

# fixture -> copilot-sdk-dnalang DNACompiler.parse_organism(filepath) is not None
COPILOT_SDK = {n: (n == "repo_organism_scimitar_sse_128.dna") for n in DNALANG_CORE}


def test_fixture_set_is_complete():
    assert set(os.listdir(D)) == set(OSIRIS_COMPILER) == set(DNALANG_CORE)


@pytest.mark.parametrize("name", sorted(OSIRIS_COMPILER))
def test_osiris_compiler_column(name):
    assert (legacy(read(name)), strict(read(name))) == OSIRIS_COMPILER[name]


def _core():
    for cand in (os.environ.get("OSIRIS_DNALANG_CORE", ""), os.path.expanduser("~/dnalang-core")):
        if cand and os.path.isdir(os.path.join(cand, "dnalang")) and cand not in sys.path:
            sys.path.insert(0, cand)
    pytest.importorskip("dnalang")
    from dnalang import parser, sema
    return parser, sema


@pytest.mark.parametrize("name", sorted(DNALANG_CORE))
def test_dnalang_core_column(name):
    parser, sema = _core()
    try:
        d = sema.check(parser.parse(read(name)))
        got = f"ok errors={len(d.errors)} warnings={len(d.warnings)}"
    except Exception as e:
        got = "ERR " + type(e).__name__
    assert got == DNALANG_CORE[name]


@pytest.mark.parametrize("name", sorted(COPILOT_SDK))
def test_copilot_sdk_file_compiler_column(name):
    path = os.path.join(ROOT, "copilot-sdk-dnalang", "src", "dnalang_sdk", "organisms_compiler.py")
    if not os.path.isfile(path):
        pytest.skip("copilot-sdk-dnalang not present")
    spec = importlib.util.spec_from_file_location("_orgc_probe", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        pytest.skip("organisms_compiler not importable here")
    got = mod.DNACompiler().parse_organism(os.path.join(D, name)) is not None
    assert got is COPILOT_SDK[name]
