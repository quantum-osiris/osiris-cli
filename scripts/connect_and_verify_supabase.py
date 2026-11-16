#!/usr/bin/env python3
"""
connect_and_verify_supabase.py — Governed Migration & Live Smoke-Test Runner
for Supabase Project `eirdsaeutzkzepibnfxf` ("osiris").

Capabilities (strictly secret-safe; never prints tokens or keys):
  1. Loads `SUPABASE_ACCESS_TOKEN`, `OSIRIS_SUPABASE_URL`, and `OSIRIS_SUPABASE_SERVICE_KEY`
     from environment variables or `.env` files (`~/.env`, `~/osiris-ide/.env`, `./.env`).
  2. If `SUPABASE_ACCESS_TOKEN` is present and `--apply-migrations` is requested, applies
     `20261116120000_osiris_workspace.sql` and `20261116130000_osiris_workspace_ownership_and_lifecycle.sql`
     to `eirdsaeutzkzepibnfxf` via the Supabase Management SQL API (`v1/projects/{ref}/database/query`)
     only when `public.osiris_projects` is not yet created (or applies `20261116130000` idempotently).
  3. If `OSIRIS_SUPABASE_SERVICE_KEY` is present, verifies `WorkspaceStore` connectivity against
     `https://eirdsaeutzkzepibnfxf.supabase.co` (`projects()` read check, plus optional
     `--smoke-write` end-to-end `project -> task -> council disposition -> retrieval` round-trip).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osiris_cli.workspace_store import PROJECT_REF, StoreError, WorkspaceStore, store_status  # noqa: E402

MIGRATION_1 = REPO_ROOT / "supabase" / "migrations" / "20261116145612_osiris_workspace.sql"
MIGRATION_2 = REPO_ROOT / "supabase" / "migrations" / "20261116145623_osiris_workspace_ownership_and_lifecycle.sql"
DEFAULT_ENV_FILES = (
    Path.home() / ".env",
    Path.home() / "osiris-ide" / ".env",
    REPO_ROOT / ".env",
)

MgmtHttp = Callable[[str, str, Dict[str, str], Optional[bytes]], Any]


def load_supabase_env(env_files: Tuple[Path, ...] = DEFAULT_ENV_FILES) -> Dict[str, bool]:
    """Populate missing Supabase env vars from standard `.env` files without echoing values."""
    wanted = ("SUPABASE_ACCESS_TOKEN", "OSIRIS_SUPABASE_URL", "OSIRIS_SUPABASE_SERVICE_KEY")
    for path in env_files:
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = k.strip()
            val = v.strip().strip('"').strip("'")
            if key in wanted and val and not os.environ.get(key):
                os.environ[key] = val

    if os.environ.get("OSIRIS_SUPABASE_SERVICE_KEY") and not os.environ.get("OSIRIS_SUPABASE_URL"):
        os.environ["OSIRIS_SUPABASE_URL"] = f"https://{PROJECT_REF}.supabase.co"

    return {k: bool(os.environ.get(k, "").strip()) for k in wanted}


def _mgmt_http(method: str, url: str, headers: Dict[str, str], body: Optional[bytes]) -> Any:
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            return json.loads(raw.decode("utf-8")) if raw else []
    except urllib.error.HTTPError as exc:
        raise StoreError("MGMT_HTTP_ERROR", f"Supabase Management API answered HTTP {exc.code}") from None
    except Exception as exc:  # noqa: BLE001
        raise StoreError("MGMT_UNAVAILABLE", type(exc).__name__) from None


def apply_remote_migrations(
    access_token: str,
    project_ref: str = PROJECT_REF,
    http: MgmtHttp = _mgmt_http,
) -> Dict[str, Any]:
    """Idempotently apply `20261116120000` and `20261116130000` to `project_ref`."""
    if project_ref != PROJECT_REF:
        raise StoreError("WRONG_PROJECT", f"Refusing to migrate {project_ref}; expected {PROJECT_REF}")
    if not access_token.strip():
        raise StoreError("MISSING_ACCESS_TOKEN", "SUPABASE_ACCESS_TOKEN is required for remote DDL")

    endpoint = f"https://api.supabase.com/v1/projects/{project_ref}/database/query"
    headers = {
        "Authorization": f"Bearer {access_token.strip()}",
        "Content-Type": "application/json",
    }

    def run_sql(query: str) -> Any:
        return http("POST", endpoint, headers, json.dumps({"query": query}).encode("utf-8"))

    existing = run_sql(
        "select tablename from pg_tables where schemaname = 'public' and tablename like 'osiris_%' order by 1;"
    )
    existing_names: List[str] = [str(r.get("tablename")) for r in (existing or []) if isinstance(r, dict)]

    applied: List[str] = []
    if "osiris_projects" not in existing_names:
        run_sql(MIGRATION_1.read_text(encoding="utf-8"))
        applied.append(MIGRATION_1.name)

    # Check if `owner_id` column from migration 2 is already present
    cols = run_sql(
        "select column_name from information_schema.columns "
        "where table_schema = 'public' and table_name = 'osiris_projects' and column_name = 'owner_id';"
    )
    if not cols:
        run_sql(MIGRATION_2.read_text(encoding="utf-8"))
        applied.append(MIGRATION_2.name)

    rls_rows = run_sql(
        "select relname, relrowsecurity from pg_class "
        "where relname in ('osiris_projects', 'osiris_tasks', 'osiris_dispositions') and relkind = 'r' order by 1;"
    )
    return {
        "project_ref": project_ref,
        "applied_migrations": applied,
        "rls_verified": rls_rows,
    }


def verify_store_connection(smoke_write: bool = False, store: Optional[WorkspaceStore] = None) -> Dict[str, Any]:
    """Verify PostgREST connectivity (and optional workflow round-trip) via `WorkspaceStore`."""
    ws = store or WorkspaceStore()
    projects = ws.projects()
    out: Dict[str, Any] = {
        "configured": True,
        "project_ref": PROJECT_REF,
        "existing_project_count": len(projects),
        "smoke_write_executed": False,
    }
    if smoke_write:
        proj = ws.create_project("OSIRIS-HP-004 Verification Workspace", "Automated live Supabase workflow smoke check")
        task = ws.create_task(proj["id"], "Verify IBM Marrakesh Job dbti1fbsbsbs73cb2vu0 artifact", status="todo")
        task = ws.update_task(task["id"], status="doing")
        receipt_hash = hashlib.sha256(f"smoke:{proj['id']}:{task['id']}".encode("utf-8")).hexdigest()
        council_stub = {
            "schema": "osiris-agent-task/v1",
            "task_id": f"smoke-{proj['id'][:8]}",
            "objective": "Verify archived OSIRIS-HP-004 raw counts against 1-ebit Schmidt-rank-2 bound",
            "decision": "PENDING_HUMAN_APPROVAL",
            "reason": "Automated smoke test disposition; awaiting human review.",
            "aiden_verdict": "PROCEED",
            "allow_cloud": False,
            "authority": {"execution": "NONE", "executed": False},
            "agents": [],
            "receipt": {"kind": "agents.task", "hash": receipt_hash, "chain_valid": True},
        }
        disp = ws.record_disposition(proj["id"], council_stub, task_id=task["id"])
        detail = ws.project(proj["id"])
        out["smoke_write_executed"] = True
        out["smoke_project_id"] = proj["id"]
        out["smoke_task_id"] = task["id"]
        out["smoke_disposition_id"] = disp["id"]
        out["retrieved_task_count"] = len(detail.get("osiris_tasks") or [])
        out["retrieved_disposition_count"] = len(detail.get("osiris_dispositions") or [])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Connect and verify Supabase project eirdsaeutzkzepibnfxf")
    parser.add_argument("--apply-migrations", action="store_true", help="Apply pending SQL migrations using SUPABASE_ACCESS_TOKEN")
    parser.add_argument("--smoke-write", action="store_true", help="Run live project->task->disposition->retrieval write check")
    args = parser.parse_args()

    presence = load_supabase_env()
    report: Dict[str, Any] = {
        "project_ref": PROJECT_REF,
        "credentials_present": presence,
        "store_status": store_status(),
    }

    if not presence["SUPABASE_ACCESS_TOKEN"] and not presence["OSIRIS_SUPABASE_SERVICE_KEY"]:
        report["status"] = "CREDENTIALS_NOT_SET"
        report["instructions"] = (
            "Set SUPABASE_ACCESS_TOKEN (for --apply-migrations) and OSIRIS_SUPABASE_SERVICE_KEY "
            "in ~/osiris-ide/.env or environment, then re-run this script."
        )
        print(json.dumps(report, indent=2))
        return 2

    try:
        if args.apply_migrations and presence["SUPABASE_ACCESS_TOKEN"]:
            report["migrations"] = apply_remote_migrations(os.environ["SUPABASE_ACCESS_TOKEN"])
        if presence["OSIRIS_SUPABASE_SERVICE_KEY"]:
            report["connection"] = verify_store_connection(smoke_write=args.smoke_write)
        report["status"] = "CONNECTED"
        print(json.dumps(report, indent=2))
        return 0
    except StoreError as exc:
        report["status"] = "ERROR"
        report["error_code"] = exc.code
        report["error"] = str(exc)
        print(json.dumps(report, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
