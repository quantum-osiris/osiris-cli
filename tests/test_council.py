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
