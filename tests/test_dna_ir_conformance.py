"""Conformance slice for the DNA-IR core contract (dna-ir/0.1).

Authority: osiris-cli/dna_lang.py (main @ 5549d0c9) owns the canonical
dna-ir/0.1 document IR, its pinned field spec (SPEC / SPEC_HASH) and the
RFC 8785 (JCS) canonical byte encoding + dna:sha256: content ids.
dnalang-core owns a *different* semantic layer (executable circuit/genome
IRs in dnalang.qc_ir / rules_ir); these tests deliberately do not merge them.

Each test names the contract clause it pins. Fixtures live in
tests/fixtures/dna_ir/. Run: python -m pytest tests/test_dna_ir_conformance.py
"""

import copy
import hashlib
import json
from pathlib import Path

import pytest

import dna_lang

FIXTURES = Path(__file__).parent / "fixtures" / "dna_ir"
MINIMAL_DNA = (FIXTURES / "minimal_organism.dna").read_text(encoding="utf-8")

# Known-answer pins: any change to SPEC, LANG_VERSION/IR_VERSION, or the
# canonicalizer must be a deliberate version bump, not a silent drift.
PINNED_SPEC_HASH = "sha256:90be4b47595b1528139041e344e7ae00ab2991e63acf1893874fbeb41a1a7a34"
PINNED_MINIMAL_CONTENT_ID = (
    "dna:sha256:f81d76b18b0c925ac660e03c4979dabca69f60466c7cb00df987f8dff5bb913c"
)
VALID_REF = "dna:sha256:" + "ab" * 32


def _rules(fn):
    """Run fn, return sorted diagnostic rule ids (empty means accepted)."""
    try:
        fn()
        return set()
    except dna_lang.Invalid as e:
        return {d.rule for d in e.diags}


def _minimal_ir():
    return dna_lang.compile_source(MINIMAL_DNA)


# ---------------------------------------------------------------- core shape

def test_pinned_spec_and_versions_are_current():
    assert dna_lang.LANG_VERSION == "dna-lang/0.1"
    assert dna_lang.IR_VERSION == "dna-ir/0.1"
    assert dna_lang.SPEC_HASH == PINNED_SPEC_HASH


def test_minimal_valid_document_compiles_and_validates():
    ir = _minimal_ir()
    assert ir["schema_version"] == "dna-ir/0.1"
    assert ir["kind"] == "organism_definition"
    assert ir["name"] == "minimal"
    assert ir["spec_hash"] == PINNED_SPEC_HASH
    # named collections are always present in IR, possibly empty
    assert ir["capabilities"] == {} and ir["evidence"] == {} and ir["metamorphoses"] == {}
    # revalidation of the lowered IR is a no-op
    assert dna_lang.validate_ir(ir) == ir


def test_unsupported_version_rejected():
    ir = copy.deepcopy(_minimal_ir())
    ir["schema_version"] = "dna-ir/0.2"
    assert "DNA-E-VERSION" in _rules(lambda: dna_lang.validate_ir(ir))
    # source side: wrong meta.schema constant
    bad_src = MINIMAL_DNA.replace('schema "dna-lang/0.1"', 'schema "dna-lang/0.2"')
    assert "DNA-E-VERSION" in _rules(lambda: dna_lang.compile_source(bad_src))
    # a foreign spec_hash (e.g. from a hypothetical other compiler) is rejected
    ir = copy.deepcopy(_minimal_ir())
    ir["spec_hash"] = "sha256:" + "00" * 32
    assert "DNA-E-VERSION" in _rules(lambda: dna_lang.validate_ir(ir))


def test_unknown_field_rejected_in_source_and_json():
    bad_src = MINIMAL_DNA.replace('title "Minimal conformance organism"',
                                  'title "Minimal conformance organism"\n    mystery 1')
    assert "DNA-E-UNKNOWN" in _rules(lambda: dna_lang.compile_source(bad_src))
    ir = copy.deepcopy(_minimal_ir())
    ir["mystery"] = 1
    assert "DNA-E-UNKNOWN" in _rules(lambda: dna_lang.validate_ir(ir))
    ir = copy.deepcopy(_minimal_ir())
    ir["meta"]["mystery"] = 1
    assert "DNA-E-UNKNOWN" in _rules(lambda: dna_lang.validate_ir(ir))


def test_duplicate_key_rejected():
    # duplicate JSON object key (raw bytes, before any dict collapse)
    raw = dna_lang.canonical_bytes(_minimal_ir())[:-1] + b',"name":"x"}'
    assert "DNA-E-DUP" in _rules(lambda: dna_lang.load_ir(raw))
    # duplicate block in source: a second intent block before the closing brace
    bad_src = MINIMAL_DNA.rstrip()[:-1] + '\n  intent { objective "again" }\n}\n'
    assert "DNA-E-DUP" in _rules(lambda: dna_lang.compile_source(bad_src))


def test_invalid_primitive_and_out_of_range_values_rejected():
    # non-integer number in compiled JSON is refused at parse time
    raw = dna_lang.canonical_bytes(_minimal_ir()).replace(b'"generation":0', b'"generation":1.5')
    assert "DNA-E-TYPE" in _rules(lambda: dna_lang.load_ir(raw))
    # float literal in source is a lexer type error
    assert "DNA-E-TYPE" in _rules(
        lambda: dna_lang.compile_source(MINIMAL_DNA.replace("generation 0", "generation 0.5")))
    # integer bounds: generation is int in [0, 2**53-1]
    for bad in (-1, 2**53):
        ir = copy.deepcopy(_minimal_ir())
        ir["identity"]["generation"] = bad
        assert "DNA-E-RANGE" in _rules(lambda: dna_lang.validate_ir(ir))
    # bool is not an int
    ir = copy.deepcopy(_minimal_ir())
    ir["identity"]["generation"] = True
    assert "DNA-E-TYPE" in _rules(lambda: dna_lang.validate_ir(ir))


# ------------------------------------------------- ordered vs unordered sets

def test_set_is_unordered_and_canonical_sorted():
    a = dna_lang.compile_source(MINIMAL_DNA.replace('require ["read_only"]',
                                                     'require ["read_only", "aaa"]'))
    b = dna_lang.compile_source(MINIMAL_DNA.replace('require ["read_only"]',
                                                     'require ["aaa", "read_only"]'))
    assert a["constraints"]["require"] == ["aaa", "read_only"] == b["constraints"]["require"]
    assert dna_lang.content_id(a) == dna_lang.content_id(b)
    # ...but a hand-edited compiled JSON with an unsorted set is refused
    ir = copy.deepcopy(a)
    ir["constraints"]["require"] = ["read_only", "aaa"]
    assert "DNA-E-CANONICAL" in _rules(lambda: dna_lang.validate_ir(ir))


def test_seq_is_ordered_and_order_is_part_of_the_content_id():
    s1 = MINIMAL_DNA.replace('objective "Exercise the dna-ir/0.1 conformance slice"',
                             'objective "Exercise the dna-ir/0.1 conformance slice"\n    success ["b", "a"]')
    s2 = s1.replace('success ["b", "a"]', 'success ["a", "b"]')
    ir1, ir2 = dna_lang.compile_source(s1), dna_lang.compile_source(s2)
    assert ir1["intent"]["success"] == ["b", "a"]  # source order kept verbatim
    assert ir2["intent"]["success"] == ["a", "b"]
    assert dna_lang.content_id(ir1) != dna_lang.content_id(ir2)


# ------------------------------------------------- canonical bytes + digest

def test_stable_canonical_bytes_and_known_content_digest():
    ir1 = _minimal_ir()
    ir2 = dna_lang.compile_source(MINIMAL_DNA)
    b1, b2 = dna_lang.canonical_bytes(ir1), dna_lang.canonical_bytes(ir2)
    assert b1 == b2  # deterministic across runs
    # RFC 8785 JCS shape: compact separators, sorted keys, no whitespace;
    # for this all-ASCII document JCS coincides with sorted-key compact JSON
    assert b1 == json.dumps(ir1, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=False).encode("utf-8")
    assert dna_lang.content_id(ir1) == PINNED_MINIMAL_CONTENT_ID
    assert dna_lang.content_id(ir1) == "dna:sha256:" + hashlib.sha256(b1).hexdigest()


def test_floats_are_outside_the_value_domain():
    with pytest.raises(ValueError):
        dna_lang.canonicalize({"x": 1.5})


# ------------------------------------------- reference syntax vs resolution

def test_reference_syntax_checked_but_not_resolved():
    # A well-formed dna:sha256: ref to an artifact that does not exist anywhere
    # still validates: DNA-IR 0.1 format-checks refs only; resolution against an
    # artifact store is explicitly out of scope (dna_lang.py docstring).
    src = MINIMAL_DNA.replace("generation 0", f'generation 0\n    genome_ref "{VALID_REF}"')
    ir = dna_lang.compile_source(src)
    assert ir["identity"]["genome_ref"] == VALID_REF
    # malformed refs are rejected at the syntax layer
    for bad in ("dna:sha256:xyz", "dna:sha256:" + "AB" * 32, "sha256:" + "ab" * 32):
        src = MINIMAL_DNA.replace("generation 0", f'generation 0\n    genome_ref "{bad}"')
        assert "DNA-E-REF" in _rules(lambda: dna_lang.compile_source(src))


# ------------------------------------------- no document-level authorization

def test_self_authorization_words_rejected():
    src = MINIMAL_DNA.replace('title "Minimal conformance organism"',
                              'title "Minimal conformance organism"\n    grant true')
    assert "DNA-E-AUTHORITY" in _rules(lambda: dna_lang.compile_source(src))
    ir = copy.deepcopy(_minimal_ir())
    ir["authorize"] = True
    assert "DNA-E-AUTHORITY" in _rules(lambda: dna_lang.validate_ir(ir))


def test_execution_authority_needs_trusted_runtime_and_promotion_gates_cannot_weaken():
    # execution capability held by a model adapter is self-authorization
    src = MINIMAL_DNA.replace("constraints {", """capability exec_cap {
    class "execution"
    authority "model_adapter"
    limits { wall_seconds 10 memory_mb 64 network "deny" native_execution "deny" }
  }
  constraints {""")
    assert "DNA-E-AUTHORITY" in _rules(lambda: dna_lang.compile_source(src))

    # a complete, correctly-gated metamorphosis proposal is accepted...
    good = MINIMAL_DNA.replace(
        "generation 0", f'generation 0\n    genome_ref "{VALID_REF}"').replace(
        "constraints {", f"""metamorphosis m1 {{
    parent "{VALID_REF}"
    target_generation 1
    objective "conformance"
    mutation {{ kind "config" patch_ref "{VALID_REF}" allowed_paths ["work"] }}
    verification {{ required_profiles ["p"] }}
    promotion {{ require_operator_approval true require_sandbox_pass true
                 require_base_hash_match true require_ledger_attestation true }}
  }}
  constraints {{""")
    ir = dna_lang.compile_source(good)
    assert ir["metamorphoses"]["m1"]["promotion"]["require_operator_approval"] is True
    # ...but weakening any promotion gate from inside the document is refused
    weak = good.replace("require_operator_approval true", "require_operator_approval false")
    assert "DNA-E-WEAKEN" in _rules(lambda: dna_lang.compile_source(weak))


# ------------------------------------------------------- replay semantics

def test_replay_projection_is_canonical_bytes_not_file_bytes(tmp_path):
    """Replay of a dna-ir document: exact artifact identity = content id over
    canonical (JCS) bytes. DNA-IR carries no timestamps or audit heads, so the
    projection has no exclusions; a non-canonical encoding of the same document
    is valid, has the same content id, and is flagged non-canonical."""
    ir = _minimal_ir()
    canonical = dna_lang.canonical_bytes(ir)

    # compiled output round-trips byte-identically
    out = tmp_path / "minimal.json"
    out.write_bytes(canonical)
    ir_rt, is_canon = dna_lang.load_any(str(out))
    assert ir_rt == ir and is_canon is True
    assert dna_lang.canonical_bytes(ir_rt) == canonical

    # pretty-printed re-encoding: same document identity, different file bytes
    pretty = tmp_path / "pretty.json"
    pretty.write_bytes(json.dumps(ir, indent=2).encode("utf-8"))
    ir_p, is_canon_p = dna_lang.load_any(str(pretty))
    assert ir_p == ir
    assert dna_lang.content_id(ir_p) == dna_lang.content_id(ir) == PINNED_MINIMAL_CONTENT_ID
    assert is_canon_p is False  # valid, but sha256(file) != content id
    # re-serializing the loaded document reproduces the canonical bytes exactly
    assert dna_lang.canonical_bytes(ir_p) == canonical
