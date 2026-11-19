"""Regression tests for OSIRIS REPL paste, claim-matching, /ignite init, and form guards."""

from __future__ import annotations

import io
import json
from unittest.mock import patch

import genome
import osiris_termux_console as otc
from osiris_cli import claims, living, osiris_repl, paste_learning
from osiris_cli.osiris_repl import OsirisReplState, dispatch_command, execute_ignite
from tests.test_living import ScriptedMentor, make

ARCH_SPEC_FIXTURE = (
    "OSIRIS System Architecture Specification: Transitioning Simulated Workspaces to a Governed Operating Platform\n\n"
    "1. Strategic Objectives & Architectural Scope\n\n"
    "Code cells compiled or staged but intentionally not dispatched.\n\n"
    "1. Preservation Invariant: Engineering teams are explicitly commanded NOT to rewrite the platform, "
    "NOT to create a new IDE, and NOT to build additional dashboards.\n"
    "2. Integration Method: Engineering teams must conduct a thorough line-by-line platform inventory.\n"
    "3. Execution Pathway: Engineering teams must reconcile all payload schemas.\n\n"
    + ("Architecture paragraph line for section one padding.\n" * 90)
    + "\n2. Canonical End-to-End System Topology\n\n"
    + ("Canonical control plane topology details.\n" * 60)
    + "\n3. P0 Remediation: Contract Standardization & Schema Unification\n\n"
    "1. Frontend API Route Proxy: app/api/osiris/plan/route.ts\n"
    "2. Backend Python Planner: osiris/planner.py\n"
    "3. Legacy JSON Contract: osiris/contracts/plan_schema_v1.json\n\n"
    "1. Zero Silent Fallbacks: Runtimes shall never auto-correct missing keys or perform silent coercion.\n"
    "2. Immediate Rejection: Requests failing schema validation must be rejected at the boundary.\n\n"
    + ("Schema enforcement body text for section three.\n" * 60)
    + "\n4. P0 Remediation: Replacing Simulated Execution & Cryptographic Verification\n\n"
    "Gate 0 signature verified before allowing promotion from staging to production.\n"
    + ("Evidence service receipt invariants.\n" * 60)
    + "\n5. Functional Re-Alignment of Workspace Components & Departments\n\n"
    "Subjects proposals to multi-stage policy evaluation.\n"
    + ("Departmental model details.\n" * 60)
    + "\n6. Repository Integration Topology & Release Sequencing Matrix\n\n"
    + ("Phased repository sequencing details and final implementation summary.\n" * 45)
)


def test_claims_no_false_positive_on_staged_staging_or_parametric_prose():
    assert claims.match(ARCH_SPEC_FIXTURE) == []
    assert claims.match("Parametric quantum circuits compiled and staged for preflight.") == []
    # True domain mentions still match accurately
    heron_hits = [c.id for c in claims.match("Staggered XY8 decoupling validated on ibm_marrakesh Heron r2")]
    assert "HERON_R2_STAG_XY8" in heron_hits
    amoc_hits = [c.id for c in claims.match("AMOC Sverdrup ocean observing telemetry and parametric settlement")]
    assert "AMOC_PROVENANCE" in amoc_hits
    # Ensure zero ASCII control characters (< 0x20, e.g. \x08 backspace) in any claim regex pattern
    for c in claims.REGISTER:
        for pat in c.patterns:
            assert not any(ord(ch) < 32 and ch not in "\n\r\t" for ch in pat), (c.id, repr(pat))


def test_headings_extracts_true_sections_and_skips_numbered_list_items():
    assert living.headings(ARCH_SPEC_FIXTURE) == [
        "1. Strategic Objectives & Architectural Scope",
        "2. Canonical End-to-End System Topology",
        "3. P0 Remediation: Contract Standardization & Schema Unification",
        "4. P0 Remediation: Replacing Simulated Execution & Cryptographic Verification"[:70],
        "5. Functional Re-Alignment of Workspace Components & Departments",
        "6. Repository Integration Topology & Release Sequencing Matrix",
    ]


def test_excerpt_snaps_to_line_boundaries_and_lists_omitted_sections():
    assert len(ARCH_SPEC_FIXTURE) > living.MESSAGE_BUDGET * 2
    h, t = living.Osiris.excerpt_sizes(ARCH_SPEC_FIXTURE)
    assert ARCH_SPEC_FIXTURE[h] == "\n"
    assert ARCH_SPEC_FIXTURE[len(ARCH_SPEC_FIXTURE) - t - 1] == "\n"
    ex = living.Osiris.excerpt(ARCH_SPEC_FIXTURE)
    assert "Omitted middle section headings:" in ex
    assert "3. P0 Remediation: Contract Standardization & Schema Unification" in ex
    assert "the user's document is complete" in ex


def test_bare_paste_records_last_paste_and_serves_requested_middle_section(tmp_path):
    mentor = ScriptedMentor()
    o, _ = make(tmp_path, mentor=mentor)
    acknowledgement = o.converse(ARCH_SPEC_FIXTURE, input_transport="bracketed_paste")
    assert o.last_paste is not None and o.last_paste["user"] == ARCH_SPEC_FIXTURE
    assert "Run /review for a section-by-section review" in acknowledgement
    o.converse("review section 3")
    sent = mentor.calls[-1][-1]["content"]
    assert "[Requested section from your most recent pasted document]" in sent
    assert "3. P0 Remediation: Contract Standardization & Schema Unification" in sent
    assert "Frontend API Route Proxy: app/api/osiris/plan/route.ts" in sent
    assert "only that section was additionally provided" in sent


def test_generic_long_document_review_is_explicitly_limited_to_excerpt(tmp_path):
    mentor = ScriptedMentor()
    o, _ = make(tmp_path, mentor=mentor)
    doc = (
        "Document start\n"
        + ("Opening text.\n" * 500)
        + "UNSEEN_MIDDLE_MARKER\n"
        + ("Middle-only material.\n" * 700)
        + "Document end\n"
        + ("Closing text.\n" * 500)
    )
    o.converse(doc, input_transport="bracketed_paste")
    o.converse("review it")
    messages = mentor.calls[-1]
    sent = "\n".join(message["content"] for message in messages)
    assert "Do not claim to have reviewed the complete document" in sent
    assert "head-and-tail excerpt" in sent
    assert "UNSEEN_MIDDLE_MARKER" not in sent


def test_document_review_chunks_preserve_all_source_text():
    document = (
        "# Overview\n"
        + ("Overview detail.\n" * 250)
        + "\n# Risks\n"
        + ("Risk detail.\n" * 220)
    )
    chunks = living.document_review_chunks(document, chunk_chars=300)
    assert len(chunks) == 2
    assert "".join(chunks[0]["parts"]) + "".join(chunks[1]["parts"]) == document
    assert all(len(part) <= 300 for section in chunks for part in section["parts"])


def test_review_last_paste_reviews_each_chunk_and_records_coverage(tmp_path):
    mentor = ScriptedMentor()
    o, out = make(tmp_path, mentor=mentor)
    document = (
        "# Overview\n"
        + ("Overview detail.\n" * 250)
        + "\n# Risks\n"
        + ("Risk detail.\n" * 220)
    )
    o.converse(document, input_transport="bracketed_paste")

    result = o.review_last_paste()
    coverage = result["coverage"]["review_coverage"]
    assert result["status"] == "COMPLETE"
    assert coverage["sections_not_read"] == []
    assert len(coverage["sections_read"]) == 2
    assert len(mentor.calls) == sum(len(section["parts"]) for section in living.document_review_chunks(document))
    assert all(len(call) == 2 for call in mentor.calls)
    assert all("[Source segment]" in call[-1]["content"] for call in mentor.calls)
    assert o.last_paste["user"] == document
    assert o.last_paste["review_coverage"] == result["coverage"]
    assert not o.core.learned
    assert "[DOCUMENT REVIEW COMPLETE]" in "".join(out)
    assert not hasattr(o, "force_core_voice")
    with open(o.log_path, encoding="utf-8") as ledger:
        receipt = json.loads(ledger.readlines()[-1])
    assert receipt["hash"] == result["receipt_hash"]
    assert receipt["source_hash"] == coverage["source_versions"][0].removeprefix("sha256:")
    assert receipt["coverage"]["sections_not_read"] == []
    assert paste_learning.get_last_paste_candidate(o.home).text == document


def test_document_review_reports_unread_sections_when_mentor_fails(tmp_path):
    mentor = ScriptedMentor(fail=ConnectionError("offline"))
    o, _ = make(tmp_path, mentor=mentor)
    document = "# One\n" + ("Evidence.\n" * 400) + "\n# Two\nMore evidence.\n"
    o.converse(document, input_transport="bracketed_paste")

    result = o.review_last_paste()
    coverage = result["coverage"]["review_coverage"]
    assert result["status"] == "PARTIAL"
    assert coverage["sections_read"] == []
    assert coverage["sections_not_read"] == ["1: # One", "2: # Two"]
    assert len(mentor.calls) == 1


def test_document_review_refuses_unconfirmed_remote_mentor(tmp_path):
    mentor = ScriptedMentor()
    mentor.local_review_allowed = False
    o, _ = make(tmp_path, mentor=mentor)
    document = "# Private\n" + ("Private content.\n" * 250)
    o.converse(document, input_transport="bracketed_paste")

    result = o.review_last_paste()
    assert result == {"status": "LOCAL_MENTOR_REQUIRED", "coverage": None}
    assert mentor.calls == []


def test_document_review_only_allows_loopback_ollama():
    assert living.OllamaMentor(base="http://localhost:11434").local_review_allowed
    assert living.OllamaMentor(base="http://127.0.0.1:11434").local_review_allowed
    assert not living.OllamaMentor(base="https://models.example.test").local_review_allowed


def test_review_command_routes_to_last_paste_review(monkeypatch):
    called = []

    class FakeLiving:
        def review_last_paste(self):
            called.append(True)

    monkeypatch.setattr(osiris_repl, "get_living", FakeLiving)
    assert osiris_repl.bare_command("review") == "/review"
    dispatch_command(OsirisReplState(), "/review")
    assert called == [True]


def test_execute_ignite_initializes_missing_genome_without_catch22(tmp_path, monkeypatch):
    gpath = tmp_path / "genome.json"
    monkeypatch.setattr(genome, "GENOME_PATH", str(gpath))
    monkeypatch.setattr(genome, "PROMPT_LIBRARY_PATH", str(tmp_path / "prompt_library.json"))
    state = OsirisReplState()
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        res = execute_ignite(state)
    out = buf.getvalue()
    assert res["status"] == "SUCCESS"
    assert gpath.exists()
    assert "does not exist -- run /ignite once to initialize it" not in out
    assert state.evidence_ledger[-1]["details"]["generation"] == 1
    assert state.evidence_ledger[-1]["details"]["arch"] != ""


def test_swarm_refuses_non_numeric_pasted_document_instead_of_running_dd(monkeypatch, capsys):
    state = OsirisReplState()
    called = []
    monkeypatch.setattr(osiris_repl, "execute_swarm", lambda st, steps=10: called.append(steps))
    dispatch_command(state, "/swarm " + ARCH_SPEC_FIXTURE)
    out = capsys.readouterr().out
    assert called == []
    assert "Refusing to discard your input into an unrelated DD loop" in out
    assert otc._pending_suggestions.get("1", "").startswith("/plan ")
    assert otc._pending_suggestions.get("2") == "/swarm 10"

    dispatch_command(state, "/swarm 4")
    assert called == [4]


def test_experiment_and_gap_refuse_multiline_document_paste_cleanly(monkeypatch, capsys):
    monkeypatch.setattr(otc, "_ask", lambda *a, **k: (_ for _ in ()).throw(AssertionError("_ask must not run")))
    otc._experiment(ARCH_SPEC_FIXTURE)
    out_exp = capsys.readouterr().out
    assert "MULTI-LINE PASTE REFUSED" in out_exp
    assert otc._intent_state["text"] == ARCH_SPEC_FIXTURE.strip()
    assert otc._pending_suggestions["2"] == "/import"

    otc._gap(ARCH_SPEC_FIXTURE)
    out_gap = capsys.readouterr().out
    assert "MULTI-LINE PASTE REFUSED" in out_gap
    assert otc._pending_suggestions["2"] == "/import"


def test_probes_and_knowledge_discover_osiris_assessment_work_layout(tmp_path):
    from osiris_cli.knowledge import Knowledge
    from osiris_cli.probes import Probes

    work = tmp_path / "osiris-assessment" / "work"
    res_dir = work / "organism_sim" / "results"
    res_dir.mkdir(parents=True)
    (res_dir / "m1_eval_seeds0-4.json").write_text('{"verdict": true}', encoding="utf-8")
    (work / "organism_sim" / "README.md").write_text(
        "# organism_sim\n\n<!-- scorecard:start -->\n| experiment | verdict |\n|---|---|\n| m1 | PASS |\n<!-- scorecard:end -->\n",
        encoding="utf-8",
    )
    # Simulate a git worktree where .git is a file rather than a directory
    (work / "osiris-cli").mkdir(parents=True)
    (work / "osiris-cli" / ".git").write_text("gitdir: /tmp/fake\n", encoding="utf-8")

    p = Probes(str(tmp_path), base=str(tmp_path))
    assert p._resolve_subpath("osiris-cli", want_git=True) == str(work / "osiris-cli")
    ev = p.evidence("scorecard evidence")
    assert "m1_eval_seeds0-4.json" in ev["out"] and "verdict: PASS" in ev["out"]

    kb = Knowledge(living_home=str(tmp_path / "living"), base=str(tmp_path))
    assert len(kb.texts) >= 1
    assert any("m1: PASS" in r for r in kb.on_record("what did m1 find?"))

