"""OSIRIS's application store in Supabase: projects, tasks and council dispositions.

This is the application's own database access (PostgREST with the service-role key, held by the
OSIRIS backend only). It is separate from developer MCP access and is never given to a browser.

Configuration, from the environment: ``OSIRIS_SUPABASE_URL`` (https://<ref>.supabase.co) and
``OSIRIS_SUPABASE_SERVICE_KEY``. Without both, every call raises ``StoreError("NOT_CONFIGURED")``:
there is no local stand-in that would make an unsaved record look saved.

The tables are created by supabase/migrations/20261116145612_osiris_workspace.sql.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

PROJECT_REF = "eirdsaeutzkzepibnfxf"
TASK_STATUSES = ("todo", "doing", "done")
_URL = re.compile(r"^https://([a-z0-9]{20})\.supabase\.co/?$")
_LOCAL_URL = re.compile(r"^http://(127\.0\.0\.1|localhost):\d{2,5}/?$")
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
Http = Callable[[str, str, Dict[str, str], Optional[bytes]], Any]


def store_status(expected_ref: str = PROJECT_REF) -> Dict[str, Any]:
    """Configuration readiness without exposing URLs or keys."""
    url = os.environ.get("OSIRIS_SUPABASE_URL", "").strip()
    key = os.environ.get("OSIRIS_SUPABASE_SERVICE_KEY", "").strip()
    if not url or not key:
        return {"configured": False, "project_ref": expected_ref, "reason": "NOT_CONFIGURED"}
    m = _URL.match(url)
    if not m:
        return {"configured": False, "project_ref": expected_ref, "reason": "BAD_CONFIG"}
    if m.group(1) != expected_ref:
        return {"configured": False, "project_ref": expected_ref, "reason": "WRONG_PROJECT"}
    return {"configured": True, "project_ref": expected_ref, "reason": None}


class StoreError(Exception):
    def __init__(self, code: str, message: str = ""):
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code


def _http(method: str, url: str, headers: Dict[str, str], body: Optional[bytes]) -> Any:
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        code = "UNAUTHORIZED" if e.code in (401, 403) else "CONFLICT" if e.code == 409 else "REJECTED"
        raise StoreError(code, f"Supabase answered HTTP {e.code}") from None   # the body may echo values
    except Exception as e:  # noqa: BLE001
        raise StoreError("UNAVAILABLE", type(e).__name__) from None
    return json.loads(raw) if raw else []


class WorkspaceStore:
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None, http: Http = _http,
                 expected_ref: str = PROJECT_REF):
        url = (os.environ.get("OSIRIS_SUPABASE_URL", "") if url is None else url).strip()
        key = (os.environ.get("OSIRIS_SUPABASE_SERVICE_KEY", "") if key is None else key).strip()
        if not url or not key:
            raise StoreError("NOT_CONFIGURED", "OSIRIS_SUPABASE_URL and OSIRIS_SUPABASE_SERVICE_KEY are not both set")
        m = _URL.match(url)
        if not m:
            raise StoreError("BAD_CONFIG", "OSIRIS_SUPABASE_URL is not https://<ref>.supabase.co")
        if m.group(1) != expected_ref:   # never write to another project because a variable changed
            raise StoreError("WRONG_PROJECT", f"configured for {m.group(1)}, expected {expected_ref}")
        self._base, self._key, self._http = url.rstrip("/") + "/rest/v1", key, http

    @classmethod
    def local(cls, url: Optional[str] = None, key: Optional[str] = None, http: Http = _http) -> "WorkspaceStore":
        """The loopback stand-in (supabase/local_stack/server.mjs) only: never a remote host.

        Reads OSIRIS_SUPABASE_LOCAL_URL and OSIRIS_SUPABASE_LOCAL_SECRET_KEY. The hosted constructor's
        project lock is unchanged; this is a separate, explicit entry point for local development.
        """
        url = (os.environ.get("OSIRIS_SUPABASE_LOCAL_URL", "") if url is None else url).strip()
        key = (os.environ.get("OSIRIS_SUPABASE_LOCAL_SECRET_KEY", "") if key is None else key).strip()
        if not url or not key:
            raise StoreError("NOT_CONFIGURED", "OSIRIS_SUPABASE_LOCAL_URL and OSIRIS_SUPABASE_LOCAL_SECRET_KEY are not both set")
        if not _LOCAL_URL.match(url):
            raise StoreError("BAD_CONFIG", "OSIRIS_SUPABASE_LOCAL_URL must be http://127.0.0.1:<port> or http://localhost:<port>")
        if not key.startswith("sb_secret_local_"):
            raise StoreError("BAD_CONFIG", "the local stand-in's secret key is expected (sb_secret_local_...)")
        store = cls.__new__(cls)
        store._base, store._key, store._http = url.rstrip("/") + "/rest/v1", key, http
        return store

    def _call(self, method: str, path: str, body: Any = None) -> List[Dict[str, Any]]:
        headers = {"apikey": self._key, "Content-Type": "application/json", "Prefer": "return=representation"}
        if not self._key.startswith("sb_secret_"):
            headers["Authorization"] = f"Bearer {self._key}"
        rows = self._http(method, self._base + path, headers, None if body is None else json.dumps(body).encode())
        if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
            raise StoreError("BAD_RESPONSE", "Supabase did not return a list of rows")
        return rows

    def _one(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(rows) != 1 or not _UUID.match(str(rows[0].get("id", ""))):
            raise StoreError("BAD_RESPONSE", "the write did not return exactly one saved row")
        return rows[0]

    def create_project(self, name: str, description: str = "") -> Dict[str, Any]:
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 200:
            raise StoreError("INVALID_INPUT", "name must be 1 to 200 characters")
        return self._one(self._call("POST", "/osiris_projects", {"name": name.strip(), "description": str(description)[:4000]}))

    def create_task(self, project_id: str, title: str, status: str = "todo") -> Dict[str, Any]:
        self._uuid(project_id)
        if not isinstance(title, str) or not 1 <= len(title.strip()) <= 400:
            raise StoreError("INVALID_INPUT", "title must be 1 to 400 characters")
        if status not in TASK_STATUSES:
            raise StoreError("INVALID_INPUT", "status must be todo, doing or done")
        body: Dict[str, Any] = {"project_id": project_id, "title": title.strip()}
        if status != "todo":
            body["status"] = status
        return self._one(self._call("POST", "/osiris_tasks", body))

    def update_task(self, task_id: str, *, status: Optional[str] = None, title: Optional[str] = None) -> Dict[str, Any]:
        self._uuid(task_id)
        patch: Dict[str, Any] = {}
        if status is not None:
            if status not in TASK_STATUSES:
                raise StoreError("INVALID_INPUT", "status must be todo, doing or done")
            patch["status"] = status
        if title is not None:
            if not isinstance(title, str) or not 1 <= len(title.strip()) <= 400:
                raise StoreError("INVALID_INPUT", "title must be 1 to 400 characters")
            patch["title"] = title.strip()
        if not patch:
            raise StoreError("INVALID_INPUT", "nothing to update")
        rows = self._call("PATCH", f"/osiris_tasks?id=eq.{task_id}", patch)
        if not rows:
            raise StoreError("NOT_FOUND", "no such task")
        return self._one(rows)

    def record_disposition(self, project_id: str, result: Dict[str, Any], task_id: Optional[str] = None) -> Dict[str, Any]:
        """Save the outcome of one council run (``council.run_task``'s result). Digests and references
        only; the roles' text stays in the response and is not stored."""
        self._uuid(project_id)
        if task_id is not None:
            self._uuid(task_id)
        row = {"project_id": project_id, "task_id": task_id, **self._disposition(result)}
        return self._one(self._call("POST", "/osiris_dispositions", row))

    def claim_proposal_request(self, request_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if request_id is not None:
            self._uuid(request_id)
        rows = self._call("POST", "/rpc/osiris_claim_proposal_request", {"p_request_id": request_id})
        if not rows:
            return None
        claimed = self._one(rows)
        self._uuid(claimed.get("lease_id"))
        if not isinstance(claimed.get("objective"), str):
            raise StoreError("BAD_RESPONSE", "the claimed objective is missing")
        return claimed

    def finish_proposal_request(self, request_id: str, lease_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        self._uuid(request_id)
        self._uuid(lease_id)
        if result.get("allow_cloud") is not False or (result.get("receipt") or {}).get("chain_valid") is not True:
            raise StoreError("INVALID_INPUT", "a local council result with a verified receipt is required")
        response = self._one(self._call("POST", "/rpc/osiris_finish_proposal_request", {
            "p_request_id": request_id,
            "p_lease_id": lease_id,
            "p_disposition": self._disposition(result),
        }))
        if response.get("request_id") != request_id:
            raise StoreError("BAD_RESPONSE", "the saved disposition refers to another request")
        return response

    @staticmethod
    def _disposition(result: Dict[str, Any]) -> Dict[str, Any]:
        auth = result.get("authority") or {}
        if result.get("schema") != "osiris-agent-task/v1" or auth.get("execution") != "NONE" or auth.get("executed") is not False:
            raise StoreError("INVALID_INPUT", "not an OSIRIS council result without execution authority")
        import hashlib
        return {
            "council_task_id": result["task_id"],
            "objective_sha256": hashlib.sha256(result["objective"].encode("utf-8")).hexdigest(),
            "objective_excerpt": str(result["objective"]).strip()[:280],
            "decision": result["decision"], "reason": str(result["reason"])[:2000], "aiden_verdict": result.get("aiden_verdict"),
            "allow_cloud": bool(result.get("allow_cloud", False)),
            "receipt_kind": result["receipt"]["kind"], "receipt_hash": result["receipt"]["hash"],
            "routes": [{"role": a["role"], "provider": a["provider"], "text_sha256": a["text_sha256"], "firewall": a["firewall"],
                        "left_device": (a.get("route") or {}).get("left_device"),
                        "resolved_model": (a.get("route") or {}).get("resolved_model"),
                        "resolved_digest": (a.get("route") or {}).get("resolved_digest")} for a in result.get("agents", [])],
            "executed": False,
        }

    def dispositions(self, project_id: str, task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        self._uuid(project_id)
        filt = f"project_id=eq.{project_id}"
        if task_id is not None:
            self._uuid(task_id)
            filt += f"&task_id=eq.{task_id}"
        return self._call("GET", f"/osiris_dispositions?{filt}&order=created_at.desc&limit=100")

    def project(self, project_id: str) -> Dict[str, Any]:
        self._uuid(project_id)
        rows = self._call("GET", f"/osiris_projects?id=eq.{project_id}&select=*,osiris_tasks(*),osiris_dispositions(*)")
        if len(rows) != 1:
            raise StoreError("NOT_FOUND", "no such project")
        return rows[0]

    def projects(self) -> List[Dict[str, Any]]:
        return self._call("GET", "/osiris_projects?select=id,name,description,archived,created_at&order=created_at.desc&limit=200")

    @staticmethod
    def _uuid(value: Any) -> None:
        if not isinstance(value, str) or not _UUID.match(value):
            raise StoreError("INVALID_INPUT", "not a UUID")
