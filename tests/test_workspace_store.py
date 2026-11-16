"""OSIRIS's Supabase store: explicit failure, one project only, digests not text. Offline: a fake HTTP layer."""
import json

import pytest

from osiris_cli.workspace_store import PROJECT_REF, StoreError, WorkspaceStore

URL = f"https://{PROJECT_REF}.supabase.co"
ID, ID2 = "11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222"
RESULT = {"schema": "osiris-agent-task/v1", "task_id": "task-abc", "objective": "Plan a Bell baseline", "decision": "REJECTED",
          "reason": "firewall", "aiden_verdict": "PROCEED", "allow_cloud": False,
          "authority": {"execution": "NONE", "executed": False},
          "agents": [{"role": "AURA", "provider": "ollama:gemma3:4b", "text": "SECRET PROSE", "text_sha256": "a" * 64,
                      "firewall": "ACCEPT", "route": {"left_device": False, "resolved_model": "gemma3:4b", "resolved_digest": "d"}}],
          "receipt": {"kind": "agents.task", "hash": "b" * 64}}


def store(reply=None, calls=None, **kw):
    def http(method, url, headers, body):
        if calls is not None:
            calls.append((method, url, headers, json.loads(body) if body else None))
        if isinstance(reply, Exception):
            raise reply
        return [{"id": ID}] if reply is None else reply
    return WorkspaceStore(kw.pop("url", URL), kw.pop("key", "service-key"), http=http, **kw)


def test_without_configuration_nothing_pretends_to_save(monkeypatch):
    monkeypatch.delenv("OSIRIS_SUPABASE_URL", raising=False)
    monkeypatch.delenv("OSIRIS_SUPABASE_SERVICE_KEY", raising=False)
    with pytest.raises(StoreError) as e:
        WorkspaceStore()
    assert e.value.code == "NOT_CONFIGURED"


@pytest.mark.parametrize("url,code", [("https://abcdefghijklmnopqrst.supabase.co", "WRONG_PROJECT"),
                                      ("http://" + PROJECT_REF + ".supabase.co", "BAD_CONFIG"),
                                      ("https://evil.example/" + PROJECT_REF, "BAD_CONFIG")])
def test_only_the_intended_project_is_ever_written(url, code):
    with pytest.raises(StoreError) as e:
        store(url=url)
    assert e.value.code == code


def test_a_disposition_stores_digests_and_references_not_model_text():
    calls = []
    store(calls=calls).record_disposition(ID, RESULT, task_id=ID2)
    method, url, headers, row = calls[0]
    assert method == "POST" and url == f"{URL}/rest/v1/osiris_dispositions" and headers["apikey"] == "service-key"
    assert "SECRET PROSE" not in json.dumps(row) and row["executed"] is False and row["receipt_hash"] == "b" * 64
    assert row["routes"][0] == {"role": "AURA", "provider": "ollama:gemma3:4b", "text_sha256": "a" * 64, "firewall": "ACCEPT",
                                "left_device": False, "resolved_model": "gemma3:4b", "resolved_digest": "d"}


@pytest.mark.parametrize("change", [{"authority": {"execution": "GRANTED", "executed": False}},
                                    {"authority": {"execution": "NONE", "executed": True}}, {"schema": "other"}])
def test_a_result_that_claims_execution_is_not_recorded(change):
    calls = []
    with pytest.raises(StoreError):
        store(calls=calls).record_disposition(ID, {**RESULT, **change})
    assert calls == []


@pytest.mark.parametrize("reply,code", [([], "BAD_RESPONSE"), ([{"id": ID}, {"id": ID2}], "BAD_RESPONSE"), ({"id": ID}, "BAD_RESPONSE"),
                                        ([{"id": "nope"}], "BAD_RESPONSE"), (StoreError("UNAVAILABLE"), "UNAVAILABLE"),
                                        (StoreError("UNAUTHORIZED"), "UNAUTHORIZED")])
def test_a_write_counts_only_when_exactly_one_saved_row_comes_back(reply, code):
    with pytest.raises(StoreError) as e:
        store(reply).create_project("Bell")
    assert e.value.code == code


def test_inputs_are_checked_before_any_request():
    calls = []
    s = store(calls=calls)
    for bad in (lambda: s.create_project(""), lambda: s.create_task("not-a-uuid", "t"), lambda: s.create_task(ID, ""),
                lambda: s.create_task(ID, "t", status="bogus"), lambda: s.update_task(ID),
                lambda: s.update_task(ID, status="bogus"), lambda: s.dispositions("bad"),
                lambda: s.project("1 or 1=1")):
        with pytest.raises(StoreError):
            bad()
    assert calls == []


def test_task_lifecycle_and_disposition_retrieval_queries(monkeypatch):
    from osiris_cli.workspace_store import store_status

    monkeypatch.delenv("OSIRIS_SUPABASE_URL", raising=False)
    monkeypatch.delenv("OSIRIS_SUPABASE_SERVICE_KEY", raising=False)
    assert store_status() == {"configured": False, "project_ref": PROJECT_REF, "reason": "NOT_CONFIGURED"}
    monkeypatch.setenv("OSIRIS_SUPABASE_URL", URL)
    monkeypatch.setenv("OSIRIS_SUPABASE_SERVICE_KEY", "secret-service-key")
    st = store_status()
    assert st == {"configured": True, "project_ref": PROJECT_REF, "reason": None}
    assert "secret-service-key" not in json.dumps(st)

    calls = []
    s = store(reply=[{"id": ID2, "status": "doing", "title": "Run Bell sim"}], calls=calls)
    updated = s.update_task(ID2, status="doing")
    assert updated["status"] == "doing"
    assert calls[-1][0] == "PATCH" and f"/osiris_tasks?id=eq.{ID2}" in calls[-1][1]

    s2 = store(reply=[{"id": ID, "council_task_id": "task-abc", "receipt_hash": "b" * 64}], calls=calls)
    rows = s2.dispositions(ID, task_id=ID2)
    assert len(rows) == 1 and rows[0]["council_task_id"] == "task-abc"
    assert f"project_id=eq.{ID}&task_id=eq.{ID2}" in calls[-1][1]

