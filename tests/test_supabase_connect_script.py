"""Unit tests for `scripts/connect_and_verify_supabase.py`."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from osiris_cli.workspace_store import PROJECT_REF, WorkspaceStore

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "connect_and_verify_supabase.py"
_SPEC = importlib.util.spec_from_file_location("connect_and_verify_supabase", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def test_apply_remote_migrations_idempotently_applies_both_sql_files() -> None:
    queries: List[str] = []

    def fake_mgmt(method: str, url: str, headers: Dict[str, str], body: Optional[bytes]) -> Any:
        assert method == "POST"
        assert PROJECT_REF in url
        assert headers["Authorization"] == "Bearer sbp_test_token"
        q = json.loads((body or b"{}").decode("utf-8"))["query"]
        queries.append(q)
        if "from pg_tables" in q:
            return []
        if "from information_schema.columns" in q:
            return []
        if "from pg_class" in q:
            return [{"relname": "osiris_projects", "relrowsecurity": True}]
        return []

    res = _MOD.apply_remote_migrations("sbp_test_token", http=fake_mgmt)
    assert res["applied_migrations"] == [
        "20261116120000_osiris_workspace.sql",
        "20261116130000_osiris_workspace_ownership_and_lifecycle.sql",
    ]
    assert len(queries) == 5


def test_verify_store_connection_roundtrip_without_leaking_secrets() -> None:
    proj_id = "11111111-1111-4111-8111-111111111111"
    task_id = "22222222-2222-4222-8222-222222222222"
    disp_id = "33333333-3333-4333-8333-333333333333"
    rows_tasks: List[Dict[str, Any]] = []
    rows_disps: List[Dict[str, Any]] = []

    def fake_postgrest(method: str, url: str, headers: Dict[str, str], body: Optional[bytes]) -> Any:
        payload = json.loads(body.decode("utf-8")) if body else None
        if method == "GET" and "/osiris_projects?select=id,name" in url:
            return []
        if method == "POST" and url.endswith("/osiris_projects"):
            return [{"id": proj_id, **payload}]
        if method == "POST" and url.endswith("/osiris_tasks"):
            row = {"id": task_id, "status": "todo", **payload}
            rows_tasks.append(row)
            return [row]
        if method == "PATCH" and "/osiris_tasks?id=eq." in url:
            rows_tasks[0].update(payload)
            return [rows_tasks[0]]
        if method == "POST" and url.endswith("/osiris_dispositions"):
            row = {"id": disp_id, **payload}
            rows_disps.append(row)
            return [row]
        if method == "GET" and f"/osiris_projects?id=eq.{proj_id}" in url:
            return [{"id": proj_id, "name": "P", "osiris_tasks": rows_tasks, "osiris_dispositions": rows_disps}]
        raise AssertionError(f"Unexpected call: {method} {url}")

    ws = WorkspaceStore(
        url=f"https://{PROJECT_REF}.supabase.co",
        key="secret_service_key_never_printed",
        http=fake_postgrest,
    )
    out = _MOD.verify_store_connection(smoke_write=True, store=ws)
    assert out["smoke_write_executed"] is True
    assert out["retrieved_task_count"] == 1
    assert out["retrieved_disposition_count"] == 1
    assert "secret_service_key_never_printed" not in json.dumps(out)


def test_missing_credentials_reports_clean_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for k in ("SUPABASE_ACCESS_TOKEN", "OSIRIS_SUPABASE_URL", "OSIRIS_SUPABASE_SERVICE_KEY"):
        monkeypatch.delenv(k, raising=False)
    presence = _MOD.load_supabase_env(env_files=(tmp_path / "nonexistent.env",))
    assert presence == {
        "SUPABASE_ACCESS_TOKEN": False,
        "OSIRIS_SUPABASE_URL": False,
        "OSIRIS_SUPABASE_SERVICE_KEY": False,
    }
