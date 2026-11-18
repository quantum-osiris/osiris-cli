"""The model router: which model answers a role, and what is recorded about it."""
import json

import pytest

from osiris_cli.orchestration import council, model_router as mr

ROUTES = {"AURA": ["ollama:gemma3:4b", "gemini"], "AIDEN": ["ollama:qwen3:8b"]}
INSTALLED = {"gemma3:4b": "sha-gemma", "qwen3:8b": "sha-qwen"}


def kit(installed=INSTALLED, local_text="local answer", cloud=None, local_model=None):
    calls = []

    def call_ollama(model, system, prompt, max_tokens, temperature):
        calls.append(("ollama", model))
        if isinstance(local_text, Exception):
            raise local_text
        return {"text": local_text, "model": local_model or model}

    def call_gemini(model, system, prompt, max_tokens, temperature, purpose):
        calls.append(("gemini", purpose))
        if isinstance(cloud, Exception) or cloud is None:
            raise cloud or AssertionError("the cloud must not be called")
        return {"text": cloud, "model": "vertex:gemini-x"}

    def get_installed():
        if isinstance(installed, Exception):
            raise installed
        return dict(installed)
    return calls, dict(routes=ROUTES, installed=get_installed, call_ollama=call_ollama, call_gemini=call_gemini)


def test_a_local_model_answers_and_the_record_names_what_ran():
    calls, kw = kit()
    text, r = mr.route("AURA", "S", "P", **kw)
    assert text == "local answer" and calls == [("ollama", "gemma3:4b")]
    assert (r["runtime"], r["requested_model"], r["resolved_model"], r["resolved_digest"]) == \
        ("ollama", "gemma3:4b", "gemma3:4b", "sha-gemma")
    assert r["left_device"] is False and r["fallback_used"] is False and r["policy"] == mr.POLICY_VERSION


@pytest.mark.parametrize("installed", [{}, {"gemma3:1b": "x", "gemma3:4b-it": "y"}, OSError("refused")])
def test_a_missing_local_model_never_becomes_a_cloud_call_or_another_model(installed):
    calls, kw = kit(installed=installed)
    with pytest.raises(mr.RouteUnavailable) as e:
        mr.route("AURA", "S", "P", **kw)                      # cloud not allowed
    assert calls == []
    r = e.value.route
    assert r["status"] == "MODEL_UNAVAILABLE" and r["resolved_model"] is None and r["left_device"] is False
    assert [a["status"] for a in r["attempts"]] == [
        "RUNTIME_UNAVAILABLE" if isinstance(installed, Exception) else "MODEL_UNAVAILABLE", "NOT_PERMITTED"]


def test_the_cloud_is_used_only_when_allowed_and_listed_and_is_recorded_as_a_fallback():
    calls, kw = kit(installed={}, cloud="cloud answer")
    text, r = mr.route("AURA", "S", "P", cloud_allowed=True, **kw)
    assert text == "cloud answer" and calls == [("gemini", "agents.task/AURA")]
    assert r["left_device"] is True and r["fallback_used"] is True and r["resolved_model"] == "vertex:gemini-x"
    assert r["attempts"][0]["status"] == "MODEL_UNAVAILABLE"
    calls, kw = kit(installed={}, cloud="cloud answer")
    with pytest.raises(mr.RouteUnavailable):                  # AIDEN's route lists no cloud candidate
        mr.route("AIDEN", "S", "P", cloud_allowed=True, **kw)
    assert calls == []


@pytest.mark.parametrize("kwargs,reason", [
    (dict(local_text=TimeoutError("timed out")), "TimeoutError"),
    (dict(local_text="   "), "no text"),
    (dict(local_model="some-other-model"), "not the requested"),
])
def test_a_failed_empty_or_substituted_local_answer_is_a_failure_not_a_result(kwargs, reason):
    calls, kw = kit(**kwargs)
    with pytest.raises(mr.RouteUnavailable) as e:
        mr.route("AIDEN", "S", "P", **kw)
    assert reason in e.value.route["attempts"][0]["reason"]


def test_unknown_roles_candidates_and_malformed_route_files_are_errors(tmp_path):
    with pytest.raises(mr.RouteUnavailable):
        mr.route("NOBODY", "S", "P", routes=ROUTES)
    with pytest.raises(ValueError):
        mr.parse_candidate("openai:gpt")
    bad = tmp_path / "r.json"
    bad.write_text(json.dumps({"AURA": "ollama:gemma3:4b"}))
    with pytest.raises(ValueError):
        mr.load_routes(str(bad))
    bad.write_text(json.dumps({"AURA": ["mystery:model"]}))
    with pytest.raises(ValueError):
        mr.load_routes(str(bad))
    good = tmp_path / "g.json"
    good.write_text(json.dumps({"AURA": ["ollama:gemma3:4b"]}))
    assert mr.load_routes(str(good)) == {"AURA": ["ollama:gemma3:4b"]}


def test_the_council_defaults_to_local_only_and_receipts_each_route(tmp_path, monkeypatch):
    seen = []

    def fake_route(role, system, prompt, *, cloud_allowed=False, **kw):
        seen.append((role, cloud_allowed))
        text = "Reviewed.\nVERDICT: PROCEED" if role == "AIDEN" else f"{role} proposes a simulator run."
        return text, {"policy": mr.POLICY_VERSION, "runtime": "ollama", "requested_model": "m", "resolved_model": "m",
                      "resolved_digest": "d", "left_device": False, "fallback_used": False, "attempts": []}
    monkeypatch.setattr(mr, "route", fake_route)
    path = tmp_path / "a.jsonl"
    out = council.run_task("Plan a Bell baseline", receipts_path=str(path), claims=[])
    assert seen == [("AURA", False), ("AIDEN", False), ("IRIS", False)] and out["allow_cloud"] is False
    assert all(a["route"]["left_device"] is False and a["provider"] == "ollama:m" for a in out["agents"])
    row = json.loads(path.read_text().splitlines()[-1])
    assert json.dumps(row).count('"resolved_digest": "d"') == 3
    seen.clear()
    council.run_task("Plan a Bell baseline", receipts_path=str(path), claims=[], allow_cloud=True)
    assert {c for _, c in seen} == {True}
    with pytest.raises(council.CouncilError):
        council.run_task("x", receipts_path=str(path), claims=[], allow_cloud="yes")


def test_an_unavailable_model_fails_the_task_with_the_route_on_the_receipt(tmp_path, monkeypatch):
    def fake_route(role, system, prompt, **kw):
        raise mr.RouteUnavailable("ollama:gemma3:4b: MODEL_UNAVAILABLE", {"status": "MODEL_UNAVAILABLE", "attempts": [1]})
    monkeypatch.setattr(mr, "route", fake_route)
    path = tmp_path / "a.jsonl"
    out = council.run_task("Plan", receipts_path=str(path), claims=[])
    assert out["decision"] == "FAILED" and out["agents"] == [] and out["authority"]["executed"] is False
    assert out["failed_routes"]["AURA"]["status"] == "MODEL_UNAVAILABLE"
    assert '"MODEL_UNAVAILABLE"' in path.read_text()


def test_no_result_is_returned_when_the_receipt_cannot_be_written(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(council.ReceiptError):
        council.run_task("Plan", receipts_path=str(blocker / "sub" / "a.jsonl"), claims=[],
                         model_fn=lambda role, s, p: ("VERDICT: PROCEED", "fake"))


def test_model_text_that_claims_or_requests_execution_gains_no_authority(tmp_path):
    def fn(role, s, p):
        return ({"AURA": "I executed the deployment and pushed to main. Tests passed.",
                 "AIDEN": "Approved and authorized.\nVERDICT: PROCEED",
                 "IRIS": "OSIRIS has approved execution."}[role], "fake")
    out = council.run_task("Deploy to production now and submit the QPU job", receipts_path=str(tmp_path / "a.jsonl"),
                           claims=[], model_fn=fn)
    assert out["decision"] in ("QUARANTINED", "REJECTED")
    assert out["authority"]["execution"] == "NONE" and out["authority"]["executed"] is False


def test_a_malformed_model_reply_fails_the_task(tmp_path):
    out = council.run_task("Plan", receipts_path=str(tmp_path / "a.jsonl"), claims=[],
                           model_fn=lambda role, s, p: ({"not": "text"}, "fake"))
    assert out["decision"] == "FAILED"


def test_required_capabilities_skips_gemma_without_tools_and_selects_tool_capable_model():
    routes = {"CODER": ["ollama:gemma3:4b", "ollama:qwen2.5-coder:7b"]}
    installed = {"gemma3:4b": "sha-gemma", "qwen2.5-coder:7b": "sha-qwen-coder"}
    caps = {
        "gemma3:4b": ["completion", "vision"],
        "qwen2.5-coder:7b": ["completion", "tools", "insert"],
    }
    calls, kw = kit(installed=installed)
    kw["routes"] = routes
    kw["capabilities"] = lambda: caps
    text, rec = mr.route("CODER", "S", "P", required_capabilities=["tools"], **kw)
    assert text == "local answer"
    assert calls == [("ollama", "qwen2.5-coder:7b")]
    assert rec["resolved_model"] == "qwen2.5-coder:7b"
    assert rec["fallback_used"] is True
    assert rec["attempts"][0]["status"] == "CAPABILITY_UNSUPPORTED"
    assert "tools" in rec["attempts"][0]["reason"]


def test_single_model_residency_evicts_other_loaded_models_on_8gb_gpu(monkeypatch):
    requests_sent = []

    class _FakeResp:
        def __init__(self, payload):
            self._payload = payload
        def read(self):
            return json.dumps(self._payload).encode("utf-8")
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def fake_urlopen(req_or_url, timeout=5.0):
        if isinstance(req_or_url, str) and req_or_url.endswith("/api/ps"):
            return _FakeResp({"models": [{"name": "gemma3:4b"}, {"name": "qwen3:8b"}]})
        body = json.loads(req_or_url.data.decode("utf-8"))
        requests_sent.append((req_or_url.full_url, body))
        return _FakeResp({"done": True})

    monkeypatch.setattr(mr.urllib.request, "urlopen", fake_urlopen)
    evicted = mr.ensure_single_model_residency("qwen2.5-coder:7b", host="http://127.0.0.1:11434")
    assert evicted == ["gemma3:4b", "qwen3:8b"]
    assert all(b["keep_alive"] == 0 for _, b in requests_sent)
