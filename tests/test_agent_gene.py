"""RFC-0001 Phase 0: strict agent-gene adapter. Synthetic genes only; no live state."""

import copy
import os
import sys

import pytest

from osiris_cli.orchestration import agent_gene as ag
from osiris_cli.orchestration.agent_gene import AgentGeneError, spec_digest, validate_agent_gene
from osiris_cli.orchestration.trio import CRITIC, DEVELOPER, RESEARCH

RESEARCH_GENE = {"kind": "agent", "role": "research", "version": "1.0",
                 "capabilities": ["READ_ONLY_ANALYSIS", "RETURN_STRUCTURED_PROPOSAL"],
                 "input_schema": "RepoTask", "output_schema": "ResearchFindings"}
DEVELOPER_GENE = {"kind": "agent", "role": "developer", "version": "1.0",
                  "capabilities": ["RETURN_STRUCTURED_PROPOSAL"],
                  "input_schema": "ResearchFindings", "output_schema": "DeveloperOutput"}
CRITIC_GENE = {"kind": "agent", "role": "critic", "version": "1.0",
               "capabilities": ["READ_ONLY_ANALYSIS", "REPLAY_FIXTURE", "SUBPROCESS_EXEC"],
               "input_schema": "CandidatePatch", "output_schema": "CriticVerdict"}


def bad(fields, needle, name="Agent1"):
    with pytest.raises(AgentGeneError) as e:
        validate_agent_gene(name, fields)
    assert needle in " | ".join(e.value.problems), e.value.problems


# ── positive ──

@pytest.mark.parametrize("name,gene,hand", [("research", RESEARCH_GENE, RESEARCH),
                                            ("developer", DEVELOPER_GENE, DEVELOPER),
                                            ("critic", CRITIC_GENE, CRITIC)])
def test_gene_builds_exactly_the_spec_trio_uses(name, gene, hand):
    v = validate_agent_gene(name, gene)
    assert v.spec == hand
    assert v.digest == spec_digest(hand)[1]          # identity matches the hand-built agent


def test_human_approval_flag_follows_the_tier_table():
    assert validate_agent_gene("c", CRITIC_GENE).requires_human_approval is True     # SUBPROCESS_EXEC
    assert validate_agent_gene("r", RESEARCH_GENE).requires_human_approval is False


def test_canonical_json_is_stable_and_key_order_independent():
    a = validate_agent_gene("r", RESEARCH_GENE)
    shuffled = dict(reversed(list(RESEARCH_GENE.items())))
    shuffled["capabilities"] = list(reversed(RESEARCH_GENE["capabilities"]))
    b = validate_agent_gene("r", shuffled)
    assert a.canonical_json == b.canonical_json and a.digest == b.digest and len(a.digest) == 64


def test_integral_float_budget_normalizes_like_an_int():
    g = dict(RESEARCH_GENE, deadline_s=120.0)
    assert validate_agent_gene("r", g).digest == validate_agent_gene("r", dict(RESEARCH_GENE, deadline_s=120)).digest


# ── digest covers every authorization-relevant field ──

@pytest.mark.parametrize("mutation", [
    {"version": "1.1"}, {"capabilities": ["READ_ONLY_ANALYSIS"]}, {"tool_permissions": ["read_file"]},
    {"provider": "ollama"}, {"model": "qwen2.5:7b"}, {"max_tokens": 2048}, {"deadline_s": 60},
    {"evidence_required": False}, {"input_schema": "Other"}, {"output_schema": "Other"}, {"id": "other"},
])
def test_changing_any_authorization_relevant_field_changes_the_digest(mutation):
    base = validate_agent_gene("r", RESEARCH_GENE).digest
    assert validate_agent_gene("r", {**RESEARCH_GENE, **mutation}).digest != base


def test_role_changes_the_digest():
    r = validate_agent_gene("x", {**RESEARCH_GENE}).digest
    d = validate_agent_gene("x", {**RESEARCH_GENE, "role": "developer"}).digest
    assert r != d


def test_policy_change_changes_the_digest(monkeypatch):
    before = validate_agent_gene("r", RESEARCH_GENE).digest
    monkeypatch.setitem(ag.CAPABILITY_TIERS, "REPLAY_FIXTURE", 2)   # a tier-table change must be visible
    assert validate_agent_gene("r", RESEARCH_GENE).digest != before


# ── negative: closed schema, allowlist, ceilings, budgets, types ──

def test_unknown_capability_is_rejected_by_default():
    bad({**RESEARCH_GENE, "capabilities": ["READ_ONLY_ANALYSIS", "ROOT_EVERYTHING"]}, "unknown capability 'ROOT_EVERYTHING'")
    bad({**RESEARCH_GENE, "capabilities": ["read_only_analysis"]}, "unknown capability")   # case-sensitive


@pytest.mark.parametrize("cap", ["WORKSPACE_WRITE", "SUBPROCESS_EXEC", "SESSION_STATE_WRITE"])
def test_developer_cannot_hold_elevated_or_unlisted_capabilities(cap):
    bad({**DEVELOPER_GENE, "capabilities": ["RETURN_STRUCTURED_PROPOSAL", cap]}, cap)


def test_workspace_write_is_not_expressible_for_any_role():
    for role in ("research", "developer", "critic"):
        bad({**CRITIC_GENE, "role": role, "capabilities": ["WORKSPACE_WRITE"]}, "human grant")


def test_unknown_fields_and_wrong_kind_and_missing_fields():
    bad({**RESEARCH_GENE, "sudo": True}, "unknown field(s): ['sudo']")
    bad({**RESEARCH_GENE, "kind": "rule"}, 'kind must be exactly "agent"')
    bad({"kind": "agent"}, "missing required field(s)")
    bad({**RESEARCH_GENE, "role": "overlord"}, "role must be one of")


@pytest.mark.parametrize("field,value,needle", [
    ("max_tokens", 0, "max_tokens"), ("max_tokens", 10**9, "max_tokens"), ("max_tokens", 10.5, "whole number"),
    ("max_tokens", "4096", "max_tokens"), ("max_tokens", True, "max_tokens"), ("max_tokens", float("nan"), "max_tokens"),
    ("deadline_s", -1, "deadline_s"), ("deadline_s", 99999, "deadline_s"), ("deadline_s", float("inf"), "deadline_s"),
    ("evidence_required", "true", "boolean"), ("evidence_required", 1, "boolean"),
    ("provider", "openai-unreviewed", "provider"), ("model", "../../etc/passwd", "model"),
    ("version", "latest", "version"), ("version", "1", "version"),
    ("capabilities", "READ_ONLY_ANALYSIS", "list of strings"), ("capabilities", [], "must not be empty"),
    ("capabilities", ["READ_ONLY_ANALYSIS", "READ_ONLY_ANALYSIS"], "duplicate"),
    ("tool_permissions", ["Bad Name"], "malformed"), ("tool_permissions", "read_file", "list of strings"),
    ("id", "not valid!", "id is malformed"),
])
def test_budgets_types_and_shapes_are_validated(field, value, needle):
    bad({**RESEARCH_GENE, field: value}, needle)


def test_all_problems_are_reported_together():
    with pytest.raises(AgentGeneError) as e:
        validate_agent_gene("x", {"kind": "agent", "role": "nope", "version": "v", "capabilities": ["ROOT"],
                                  "max_tokens": -5})
    assert len(e.value.problems) >= 4


def test_bad_gene_name():
    bad(RESEARCH_GENE, "gene name", name="1bad name")


def test_input_is_not_mutated():
    g = copy.deepcopy(RESEARCH_GENE)
    validate_agent_gene("r", g)
    assert g == RESEARCH_GENE


def test_no_execution_authority_without_validation():
    """The only constructor path from gene text to an AgentSpec is validate_agent_gene; a refused
    gene yields no spec at all (an exception, not a partially built object)."""
    with pytest.raises(AgentGeneError):
        validate_agent_gene("r", {**RESEARCH_GENE, "capabilities": ["WORKSPACE_WRITE"]})


# ── drift guard against the governor's real table (only where that tree exists) ──

def test_capability_table_matches_the_governor_when_present():
    p = os.path.expanduser("~/osiris/agents/gateway")
    if not os.path.isfile(os.path.join(p, "governor.py")):
        pytest.skip("governor tree not present")
    import re
    src = open(os.path.join(p, "governor.py"), encoding="utf-8").read()
    found = dict(re.findall(r'"([A-Z_]+)":\s*RiskTier\.TIER_(\d)_', src))
    assert {k: int(v) for k, v in found.items()} == ag.CAPABILITY_TIERS


# ── from dna::}{::lang text (dialect B), only when dnalang-core is importable ──

def _core():
    for cand in (os.environ.get("OSIRIS_DNALANG_CORE", ""), os.path.expanduser("~/dnalang-core")):
        if cand and os.path.isdir(os.path.join(cand, "dnalang")) and cand not in sys.path:
            sys.path.insert(0, cand)
    pytest.importorskip("dnalang")


DNA = '''// synthetic
organism Trio {
  gene Researcher {
    kind: "agent"
    id: "research"
    role: "research"
    version: "1.0"
    capabilities: ["READ_ONLY_ANALYSIS", "RETURN_STRUCTURED_PROPOSAL"]
    input_schema: "RepoTask"
    output_schema: "ResearchFindings"
  }
  gene Plain { condition: "1#" action: "(emit x)" }
}
'''


def test_from_dna_text_validates_agent_genes_and_ignores_others():
    _core()
    genes = ag.agent_genes_from_dna(DNA)
    assert [g.gene_name for g in genes] == ["Researcher"] and genes[0].spec == RESEARCH


def test_from_dna_text_refuses_unknown_capability():
    _core()
    with pytest.raises(AgentGeneError):
        ag.agent_genes_from_dna(DNA.replace('"RETURN_STRUCTURED_PROPOSAL"', '"ROOT_EVERYTHING"'))


def test_from_dna_text_catches_duplicate_keys_the_language_would_resolve_silently():
    _core()
    from dnalang import parser
    dup = DNA.replace('version: "1.0"', 'version: "1.0"\n    capabilities: ["WORKSPACE_WRITE"]')
    # the language itself accepts this and keeps the last value:
    assert parser.parse(dup).kv_genes[0].fields["capabilities"] == ["READ_ONLY_ANALYSIS", "RETURN_STRUCTURED_PROPOSAL"]
    with pytest.raises(AgentGeneError) as e:
        ag.agent_genes_from_dna(dup)
    assert "duplicate key Researcher.capabilities" in " ".join(e.value.problems)


def test_from_dna_text_refuses_duplicate_agent_genes():
    _core()
    two = DNA.replace("gene Plain", "gene Researcher").replace('condition: "1#" action: "(emit x)"',
                                                                'kind: "agent" role: "research" version: "1.0" capabilities: ["READ_ONLY_ANALYSIS"]')
    with pytest.raises(AgentGeneError):
        ag.agent_genes_from_dna(two)
