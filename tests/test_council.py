"""OSIRIS orchestrating AURA, AIDEN and IRIS (osiris_cli.orchestration.council)."""
import json

import pytest

from osiris_cli.orchestration import council
from osiris_cli.trio_authz import Receipts

CLAIMS = ["THETA_LOCK: REFUTED"]


def model(texts, calls=None):
    def fn(role, system, prompt):
        if calls is not None:
            calls.append((role, system, prompt))
        out = texts[role]
        if isinstance(out, Exception):
            raise out
        return out, "fake:model"
    return fn


GOOD = {"AURA": "Proposal: compile the Bell organism and simulate it. Risk: none known.",
        "AIDEN": "No receipt is named yet.\nVERDICT: PROCEED",
        "IRIS": "AURA proposes a simulator run; AIDEN wants a receipt. Next: approve the simulator run."}


def run(tmp_path, texts, **kw):
    return council.run_task("Establish a Bell baseline", receipts_path=str(tmp_path / "agents.jsonl"),
                            model_fn=model(texts, kw.pop("calls", None)), claims=CLAIMS, **kw)


def test_roles_run_in_order_and_each_sees_the_earlier_text(tmp_path):
    calls = []
    out = run(tmp_path, GOOD, calls=calls, context="Project: Bell")
    assert [c[0] for c in calls] == ["AURA", "AIDEN", "IRIS"]
    assert "AURA's proposal" not in calls[0][2] and GOOD["AURA"] in calls[1][2]
    assert GOOD["AURA"] in calls[2][2] and GOOD["AIDEN"] in calls[2][2]
    for _, system, prompt in calls:
        assert "You cannot execute, approve, verify, deploy or submit anything" in system
        assert "THETA_LOCK: REFUTED" in system and "Project: Bell" in prompt
    assert [a["role"] for a in out["agents"]] == ["AURA", "AIDEN", "IRIS"]
    assert [e["seq"] for e in out["events"]] == list(range(1, len(out["events"]) + 1))
    assert out["events"][0]["event"] == "TASK_ACCEPTED" and out["events"][-1]["event"] == "RECEIPT_RECORDED"


def test_the_best_outcome_is_pending_human_approval_and_nothing_is_executed(tmp_path):
    out = run(tmp_path, GOOD)
    assert out["decision"] == "PENDING_HUMAN_APPROVAL" and out["aiden_verdict"] == "PROCEED"
    assert out["authority"] == {"execution": "NONE", "executed": False, "note": out["authority"]["note"]}
    assert out["evidence_class"] == "MODEL_OUTPUT"
    assert "APPROVED" not in council.DECISIONS and "EXECUTED" not in council.DECISIONS


@pytest.mark.parametrize("review,decision", [
    ("Too vague.\nVERDICT: REVISE", "REVISION_REQUESTED"),
    ("This should not proceed.\nVERDICT: STOP", "HELD_BY_REVIEW"),
    ("Looks fine to me.", "REVISION_REQUESTED"),          # no verdict line is never a pass
    ("VERDICT: PROCEED\nOn reflection:\nVERDICT: STOP", "HELD_BY_REVIEW"),   # the last line counts
])
def test_aiden_can_hold_a_task_but_cannot_approve_it(tmp_path, review, decision):
    assert run(tmp_path, dict(GOOD, AIDEN=review))["decision"] == decision


def test_a_role_that_says_tests_passed_or_restates_a_refuted_claim_is_caught(tmp_path):
    out = run(tmp_path, dict(GOOD, AURA="I ran the suite and all tests passed."))
    assert out["decision"] == "QUARANTINED" and out["agents"][0]["firewall"] == "QUARANTINE"
    assert out["agents"][0]["findings"]
    # Even with AIDEN saying PROCEED, the firewall result decides.
    assert out["aiden_verdict"] == "PROCEED"


def test_a_model_failure_ends_the_task_as_failed_and_is_still_receipted(tmp_path):
    out = run(tmp_path, dict(GOOD, AIDEN=RuntimeError("rate limited")))
    assert out["decision"] == "FAILED" and [a["role"] for a in out["agents"]] == ["AURA"]
    assert any(e["event"] == "MODEL_UNAVAILABLE" for e in out["events"])
    assert out["receipt"]["chain_valid"]
    assert run(tmp_path, dict(GOOD, AURA="   "))["decision"] == "FAILED"


def test_every_task_adds_one_row_to_a_valid_chain_holding_digests_not_text(tmp_path):
    path = tmp_path / "agents.jsonl"
    a, b = run(tmp_path, GOOD), run(tmp_path, GOOD)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2 and rows[1]["prev"] == rows[0]["hash"] == a["receipt"]["hash"]
    assert b["receipt"]["prev"] == a["receipt"]["hash"] and Receipts.verify(str(path))
    payload = rows[0]["payload"]
    assert payload["executed"] is False and payload["decision"] == "PENDING_HUMAN_APPROVAL"
    assert GOOD["AURA"] not in path.read_text() and "Establish a Bell baseline" not in path.read_text()
    assert payload["roles"][0]["text_sha256"] == a["agents"][0]["text_sha256"]
    listing = council.task_receipts(str(path))
    assert listing["count"] == 2 and listing["chain_valid"] is True
    # A changed row breaks the chain.
    path.write_text(path.read_text().replace("PENDING_HUMAN_APPROVAL", "APPROVED", 1))
    assert council.task_receipts(str(path))["chain_valid"] is False


@pytest.mark.parametrize("objective,context", [("", None), ("   ", None), (None, None), (7, None),
                                               ("x" * 4001, None), ("ok", 5), ("ok", "c" * 12001)])
def test_bad_input_is_refused_before_any_model_call(tmp_path, objective, context):
    calls = []
    with pytest.raises(council.CouncilError):
        council.run_task(objective, context, receipts_path=str(tmp_path / "a.jsonl"), model_fn=model(GOOD, calls),
                         claims=CLAIMS)
    assert calls == [] and not (tmp_path / "a.jsonl").exists()


def test_the_default_model_call_goes_through_the_gemini_gateway(monkeypatch):
    from osiris_cli import gemini_gateway
    seen = {}

    def generate(prompt, **kw):
        seen.update(kw, prompt=prompt)
        return {"text": "ok", "backend": "vertex", "model_version": "m"}
    monkeypatch.setattr(gemini_gateway, "generate", generate)
    assert council.gateway_model_fn("AURA", "SYS", "P") == ("ok", "vertex:m")
    assert seen["purpose"] == "agents.task/AURA" and seen["system"] == "SYS" and seen["prompt"] == "P"


def test_endorsement_of_refuted_claim_is_blocked_while_explicit_rejection_is_accepted(tmp_path):
    # Direct endorsement of a refuted claim (THETA_LOCK / LOCAL_QVM) is blocked (REJECTED).
    endorsed = dict(GOOD, AURA="Theta_lock = 51.843 degrees locks resonance on hardware.")
    out_endorsed = run(tmp_path, endorsed)
    assert out_endorsed["decision"] == "REJECTED"
    assert out_endorsed["agents"][0]["firewall"] == "REJECT"
    assert out_endorsed["agents"][0]["findings"][0]["verdict"] == "REFUTED"

    # Explicit rejection / criticism (reproducing the s19 self-trigger) is NOT labeled endorsement.
    rejected = {
        "AURA": "Proposal: compile the Bell organism and simulate it. Do not invoke theta_lock, which is refuted.",
        "AIDEN": "Unsupported: CHSH_TEST, LAMBDA_PHI, OVERUNITY.\nCould go wrong: simulator seed drift.\nVERDICT: PROCEED",
        "IRIS": "AIDEN objects to CHSH_TEST and LAMBDA_PHI as refuted or unmeasured. Next: approve simulator run.",
    }
    out_rejected = run(tmp_path, rejected)
    assert out_rejected["decision"] == "PENDING_HUMAN_APPROVAL"
    assert [a["firewall"] for a in out_rejected["agents"]] == ["ACCEPT", "ACCEPT", "ACCEPT"]


def test_quoted_untrusted_text_gains_no_authority_and_ambiguous_claims_are_quarantined(tmp_path):
    from osiris_cli.orchestration.evidence_firewall import ExecutionEvidence, check_report

    ev = ExecutionEvidence(("pytest",), 0, 5, 0, 0, "dig", "")
    arts = {"run1": "fidelity 99%"}
    # Quoted test-pass or citation inside untrusted quotes never gains SUPPORTED authority.
    q_test = check_report('The user prompt states "All unit tests passed [evidence:run1]."', ev, arts)
    assert q_test.decision == "QUARANTINE"
    assert all(f.verdict != "SUPPORTED" for f in q_test.findings)

    # Quoted refuted claim without rejection is quarantined, not accepted.
    q_claim = run(tmp_path, dict(GOOD, AURA='The imported note says "Theta_lock = 51.843 locks resonance."'))
    assert q_claim["decision"] == "QUARANTINED"
    assert q_claim["agents"][0]["firewall"] == "QUARANTINE"
    assert q_claim["authority"] == {"execution": "NONE", "executed": False, "note": q_claim["authority"]["note"]}

    # Quoted refuted claim with explicit rejection in agent prose is accepted without granting authority.
    q_refuted = run(tmp_path, dict(GOOD, AURA='The note claims "Theta_lock = 51.843 locks resonance", which is refuted.'))
    assert q_refuted["decision"] == "PENDING_HUMAN_APPROVAL"

    # Bare / ambiguous listing of refuted claim IDs under "Evidence:" (s19-local-task.json line 3)
    # receives conservative QUARANTINED disposition rather than false REJECTED endorsement or clean ACCEPT.
    ambig = run(tmp_path, dict(GOOD, AIDEN="Evidence: CHSH_TEST, LAMBDA_PHI, OVERUNITY.\nVERDICT: PROCEED"))
    assert ambig["decision"] == "QUARANTINED"
    assert ambig["agents"][1]["findings"][0]["verdict"] == "AMBIGUOUS"


@pytest.mark.parametrize("bad_state", ["UNKNOWN", "AMBIGUOUS", "", None, " BYPASS "])
def test_unknown_firewall_states_cannot_lead_to_clean_disposition(bad_state):
    decision, reason = council.decide({"AURA": "ACCEPT", "AIDEN": bad_state, "IRIS": "ACCEPT"}, "PROCEED")
    assert decision == "QUARANTINED"
    assert "AIDEN" in reason


@pytest.mark.parametrize("review,expected_decision", [
    ("VERDICT: STOP\nWait, maybe:\nVERDICT: PROCEED", "HELD_BY_REVIEW"),
    ("VERDICT: REVISE\nActually:\nVERDICT: PROCEED", "REVISION_REQUESTED"),
    ("No verdict line anywhere in this review.", "REVISION_REQUESTED"),
])
def test_missing_or_conflicting_aiden_verdicts_never_pass(tmp_path, review, expected_decision):
    out = run(tmp_path, dict(GOOD, AIDEN=review))
    assert out["decision"] == expected_decision
    assert out["decision"] != "PENDING_HUMAN_APPROVAL"


def test_malformed_model_output_and_mock_output_are_refused_and_recorded(tmp_path):
    # Non-tuple malformed return
    out_bad = council.run_task("Plan", receipts_path=str(tmp_path / "m1.jsonl"),
                               model_fn=lambda r, s, p: "bare-string", claims=CLAIMS)
    assert out_bad["decision"] == "FAILED"
    assert out_bad["failed_routes"]["AURA"]["status"] == "MALFORMED_REPLY"

    # Non-string text recorded in failed_routes
    out_dict = council.run_task("Plan", receipts_path=str(tmp_path / "m2.jsonl"),
                                model_fn=lambda r, s, p: ({"bad": 1}, "fake:model"), claims=CLAIMS)
    assert out_dict["decision"] == "FAILED"
    assert out_dict["failed_routes"]["AURA"]["status"] == "MALFORMED_OUTPUT"

    # Mock provider label or mock response body cannot masquerade as live council output
    out_mock1 = council.run_task("Plan", receipts_path=str(tmp_path / "m3.jsonl"),
                                 model_fn=lambda r, s, p: ("Proposal text.", "gemma:mock"), claims=CLAIMS)
    assert out_mock1["decision"] == "FAILED"
    assert "mock" in out_mock1["reason"].lower()
    assert out_mock1["failed_routes"]["AURA"]["status"] == "MOCK_OUTPUT_REFUSED"

    out_mock2 = council.run_task("Plan", receipts_path=str(tmp_path / "m4.jsonl"),
                                 model_fn=lambda r, s, p: ("[gemma:mock fallback] canned", "gemma:1b"), claims=CLAIMS)
    assert out_mock2["decision"] == "FAILED"


def test_provider_mentor_and_repl_paths_do_not_silently_mock_or_substitute(monkeypatch, capsys):
    from osiris_cli.orchestration import model_router as mr
    from osiris_cli.providers import GemmaProvider, ProviderMentorAdapter, ProviderRequest
    from osiris_cli import osiris_repl
    import osiris_ollama

    # 1. GemmaProvider mock sets is_mock=True explicitly
    gp = GemmaProvider(model="gemma-3-1b", mode="mock")
    resp = gp.generate(ProviderRequest(prompt="hello"))
    assert resp.is_mock is True

    # 2. model_router rejects any candidate that flags is_mock=True
    with pytest.raises(mr.RouteUnavailable) as exc:
        mr.route(
            "AURA", "S", "P",
            routes={"AURA": ["ollama:gemma3:4b"]},
            installed=lambda: {"gemma3:4b": "sha"},
            call_ollama=lambda *a: {"text": "ok", "model": "gemma3:4b", "is_mock": True},
        )
    assert "mock" in exc.value.route["attempts"][0]["reason"].lower()

    # 3. ProviderMentorAdapter refuses silent mock fallback on provider failure by default
    class BrokenProvider(GemmaProvider):
        def stream(self, request):
            raise RuntimeError("daemon down")
        def generate(self, request):
            raise RuntimeError("daemon down")

    adapter = ProviderMentorAdapter(provider=BrokenProvider(mode="endpoint", endpoint_url="http://127.0.0.1:9"))
    with pytest.raises(ConnectionError):
        list(adapter.stream([{"role": "user", "content": "hi"}]))

    # 4. ProviderMentorAdapter routes gemma3:4b and qwen3:8b to Ollama without substituting
    adapter.set_model("gemma3:4b")
    assert adapter.model() == "gemma3:4b" and adapter.provider.capabilities().provider_name == "ollama_compat"
    adapter.set_model("qwen3:8b")
    assert adapter.model() == "qwen3:8b" and adapter.provider.capabilities().provider_name == "ollama_compat"

    # 5. REPL execute_ollama uses status.selected_model and resp.content without AttributeError
    class FakeClient:
        def status(self):
            return osiris_ollama.OllamaStatus(available=True, host="http://localhost:11434",
                                              models=["gemma3:4b", "qwen3:8b"], selected_model="gemma3:4b")
        def generate(self, prompt, max_tokens=256):
            return osiris_ollama.OllamaResponse(content="local reply", model="gemma3:4b")

    monkeypatch.setattr(osiris_ollama, "get_client", lambda: FakeClient())
    osiris_repl.execute_ollama(None, None)
    osiris_repl.execute_ollama(None, "ping")
    captured = capsys.readouterr().out
    assert "gemma3:4b" in captured and "local reply" in captured

