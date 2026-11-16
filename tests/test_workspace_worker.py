import json

import pytest

from osiris_cli.orchestration import council
from osiris_cli.workspace_worker import process_request
from osiris_cli.workspace_store import PROJECT_REF, StoreError, WorkspaceStore

REQUEST = "11111111-1111-4111-8111-111111111111"
LEASE = "22222222-2222-4222-8222-222222222222"
DISPOSITION = "33333333-3333-4333-8333-333333333333"


def test_council_worker_records_a_verified_local_proposal(tmp_path, monkeypatch):
    calls = []
    roles = []

    def local_model(cloud_allowed):
        assert cloud_allowed is False

        def run(role, _system, prompt):
            roles.append(role)
            assert "Design a Bell test" in prompt
            return ("Check the control.\nVERDICT: PROCEED" if role == "AIDEN" else "Compare the circuit with a control.",
                    "ollama:fixture", {"left_device": False})

        return run

    def http(method, url, headers, content):
        calls.append(url.rsplit("/", 1)[1])
        assert method == "POST" and headers["apikey"] == "sb_secret_fixture"
        assert "Authorization" not in headers
        body = json.loads(content)
        if calls[-1] == "osiris_claim_proposal_request":
            assert body == {"p_request_id": REQUEST}
            return [{"id": REQUEST, "lease_id": LEASE, "objective": "Design a Bell test"}]
        assert calls[-1] == "osiris_finish_proposal_request"
        assert body["p_request_id"] == REQUEST and body["p_lease_id"] == LEASE
        row = body["p_disposition"]
        assert row["executed"] is False and row["allow_cloud"] is False
        assert "project_id" not in row and "task_id" not in row
        assert row["decision"] == "PENDING_HUMAN_APPROVAL"
        assert len(row["receipt_hash"]) == 64
        return [{"id": DISPOSITION, "request_id": REQUEST, **row}]

    monkeypatch.setattr(council, "routed_model_fn", local_model)
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11434")
    store = WorkspaceStore(f"https://{PROJECT_REF}.supabase.co", "sb_secret_fixture", http=http)
    output = process_request(store, tmp_path, REQUEST)
    receipts = council.task_receipts(str(tmp_path / REQUEST / f"{LEASE}.jsonl"))
    assert output["status"] == "SAVED" and output["request_id"] == REQUEST
    assert receipts["chain_valid"] and output["receipt_hash"] == receipts["rows"][0]["hash"]
    assert calls == ["osiris_claim_proposal_request", "osiris_finish_proposal_request"]
    assert roles == ["AURA", "AIDEN", "IRIS"]


def test_no_available_request_does_not_run_models(tmp_path, monkeypatch):
    def http(_method, url, _headers, _body):
        assert url.endswith("/rpc/osiris_claim_proposal_request")
        return []

    store = WorkspaceStore(f"https://{PROJECT_REF}.supabase.co", "fixture-key", http=http)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert process_request(store, tmp_path, None) == {"status": "NO_REQUEST_AVAILABLE"}
    assert list(tmp_path.iterdir()) == []


def test_remote_ollama_is_refused_before_claim(tmp_path, monkeypatch):
    def http(*_):
        pytest.fail("The worker claimed a request with a remote model endpoint")

    monkeypatch.setenv("OLLAMA_HOST", "https://remote.example.test")
    store = WorkspaceStore(f"https://{PROJECT_REF}.supabase.co", "fixture-key", http=http)
    with pytest.raises(StoreError, match="BAD_CONFIG"):
        process_request(store, tmp_path, REQUEST)


@pytest.mark.parametrize("allow_cloud,valid", [(True, True), (False, False)])
def test_worker_refuses_cloud_or_unverified_result(allow_cloud, valid):
    def http(*_):
        pytest.fail("The worker tried to save a result it could not verify")

    store = WorkspaceStore(f"https://{PROJECT_REF}.supabase.co", "fixture-key", http=http)
    with pytest.raises(StoreError, match="INVALID_INPUT"):
        store.finish_proposal_request(REQUEST, LEASE, {"allow_cloud": allow_cloud, "receipt": {"chain_valid": valid}})


@pytest.mark.parametrize("url,key,code", [
    ("http://127.0.0.1:54321", "sb_secret_local_x", None),
    ("http://localhost:54321/", "sb_secret_local_x", None),
    ("https://eirdsaeutzkzepibnfxf.supabase.co", "sb_secret_local_x", "BAD_CONFIG"),
    ("http://10.0.0.5:54321", "sb_secret_local_x", "BAD_CONFIG"),
    ("http://127.0.0.1.example.test:54321", "sb_secret_local_x", "BAD_CONFIG"),
    ("http://127.0.0.1:54321", "sb_secret_hostedkey", "BAD_CONFIG"),
    ("", "sb_secret_local_x", "NOT_CONFIGURED"),
])
def test_local_store_accepts_only_the_loopback_stand_in(url, key, code):
    if code is None:
        assert WorkspaceStore.local(url, key, http=lambda *_: [])._base.startswith(url.rstrip("/"))
    else:
        with pytest.raises(StoreError, match=code):
            WorkspaceStore.local(url, key, http=lambda *_: [])


def test_hosted_constructor_still_refuses_a_loopback_url():
    with pytest.raises(StoreError, match="BAD_CONFIG"):
        WorkspaceStore("http://127.0.0.1:54321", "sb_secret_local_x", http=lambda *_: [])
