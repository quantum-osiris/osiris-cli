"""Signed-in users' project files for the DNA-Lang editor, through Supabase Auth and PostgREST.

Every data call here carries the *user's* access token and the project's publishable (anon) key.
The service-role key is never used, so what a user can read or change is decided by the row level
security policies in supabase/migrations (owner = ``osiris_projects.owner_id``), not by this code.
This module never sends ``owner_id``: the database sets it from the token.

Configuration, from the environment (two mutually exclusive targets):

  hosted   OSIRIS_SUPABASE_URL=https://<ref>.supabase.co  and  OSIRIS_SUPABASE_ANON_KEY
  local    OSIRIS_SUPABASE_LOCAL_URL=http://127.0.0.1:<port>  and  OSIRIS_SUPABASE_LOCAL_ANON_KEY
           (the development stand-in in supabase/local_stack; loopback only)

Without a complete target every call raises ``StoreError("NOT_CONFIGURED")``. There is no
in-memory fallback: a file that was not stored is never reported as saved.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional

import urllib.error
import urllib.request

from osiris_cli.workspace_store import PROJECT_REF, Http, StoreError, _URL, _UUID

MAX_FILE_BYTES = 64 * 1024            # the same limit as vertical_slice.MAX_SOURCE_BYTES and the table check
MAX_PATH_CHARS = 200
_LOCAL_URL = re.compile(r"^http://(127\.0\.0\.1|localhost):(\d{2,5})/?$")
# Relative, forward-slash paths of plain segments. No leading slash, no "." or ".." segment, no backslash.
_SEGMENT = r"[A-Za-z0-9_][A-Za-z0-9_.-]*"
_PATH = re.compile(rf"^{_SEGMENT}(/{_SEGMENT})*$")
_EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}$")
_CODE = re.compile(r"^[0-9]{6,10}$")
_FILE_COLUMNS = "id,project_id,path,revision,content_sha256,created_at,updated_at"


def _http(method: str, url: str, headers: Dict[str, str], body: Optional[bytes]) -> Any:
    """Like workspace_store._http, but keeps "not signed in" (401) apart from "not permitted" (403)."""
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        code = {401: "UNAUTHORIZED", 403: "FORBIDDEN", 409: "CONFLICT", 413: "TOO_LARGE",
                429: "RATE_LIMITED"}.get(e.code, "REJECTED" if e.code < 500 else "UNAVAILABLE")
        raise StoreError(code, f"the store answered HTTP {e.code}") from None   # the body may echo values
    except Exception as e:  # noqa: BLE001
        raise StoreError("UNAVAILABLE", type(e).__name__) from None
    try:
        return json.loads(raw) if raw else []
    except ValueError:
        raise StoreError("BAD_RESPONSE", "the store did not return JSON") from None


def valid_path(path: Any) -> bool:
    return (isinstance(path, str) and len(path) <= MAX_PATH_CHARS and bool(_PATH.match(path))
            and ".." not in path)


def target_status() -> Dict[str, Any]:
    """Which store the editor would use, without exposing URLs or keys."""
    local_url = os.environ.get("OSIRIS_SUPABASE_LOCAL_URL", "").strip()
    if local_url:
        if not _LOCAL_URL.match(local_url):
            return {"configured": False, "target": "local-stand-in", "reason": "BAD_CONFIG"}
        if not os.environ.get("OSIRIS_SUPABASE_LOCAL_ANON_KEY", "").strip():
            return {"configured": False, "target": "local-stand-in", "reason": "NOT_CONFIGURED"}
        return {"configured": True, "target": "local-stand-in", "reason": None}
    url = os.environ.get("OSIRIS_SUPABASE_URL", "").strip()
    key = os.environ.get("OSIRIS_SUPABASE_ANON_KEY", "").strip()
    target = f"supabase:{PROJECT_REF}"
    if not url or not key:
        return {"configured": False, "target": target, "reason": "NOT_CONFIGURED"}
    m = _URL.match(url)
    if not m:
        return {"configured": False, "target": target, "reason": "BAD_CONFIG"}
    if m.group(1) != PROJECT_REF:
        return {"configured": False, "target": target, "reason": "WRONG_PROJECT"}
    return {"configured": True, "target": target, "reason": None}


class EditorStore:
    def __init__(self, url: Optional[str] = None, anon_key: Optional[str] = None, http: Http = _http,
                 target: Optional[str] = None, verify_session: bool = True):
        # PostgREST accepts an access token until it expires, even after sign-out. With verify_session the
        # auth service is asked first whether the session is still live, so sign-out takes effect at once.
        self._verify_session = verify_session
        if url is None:
            status = target_status()
            if not status["configured"]:
                raise StoreError(status["reason"], "the editor's sign-in and file store is not configured")
            local = status["target"] == "local-stand-in"
            url = os.environ["OSIRIS_SUPABASE_LOCAL_URL" if local else "OSIRIS_SUPABASE_URL"].strip()
            anon_key = os.environ["OSIRIS_SUPABASE_LOCAL_ANON_KEY" if local else "OSIRIS_SUPABASE_ANON_KEY"].strip()
            target = status["target"]
        if not anon_key:
            raise StoreError("NOT_CONFIGURED", "no publishable key")
        self._base, self._key, self._http, self.target = url.rstrip("/"), anon_key, http, target or "injected"

    # ── transport ──
    def _call(self, method: str, path: str, token: Optional[str], body: Any = None) -> Any:
        headers = {"apikey": self._key, "Authorization": f"Bearer {token or self._key}",
                   "Content-Type": "application/json", "Prefer": "return=representation"}
        return self._http(method, self._base + path, headers, None if body is None else json.dumps(body).encode())

    def _rows(self, method: str, path: str, token: str, body: Any = None) -> List[Dict[str, Any]]:
        self._token(token)
        if self._verify_session:
            self.user(token)
        rows = self._call(method, "/rest/v1" + path, token, body)
        if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
            raise StoreError("BAD_RESPONSE", "the store did not return a list of rows")
        return rows

    @staticmethod
    def _token(token: Any) -> None:
        if not isinstance(token, str) or not 20 <= len(token) <= 8192 or re.search(r"\s", token):
            raise StoreError("UNAUTHORIZED", "no signed-in session")

    @staticmethod
    def _uuid(value: Any) -> None:
        if not isinstance(value, str) or not _UUID.match(value):
            raise StoreError("INVALID_INPUT", "not a UUID")

    # ── sign-in (Supabase Auth, email one-time code) ──
    def request_code(self, email: Any) -> Dict[str, Any]:
        if not isinstance(email, str) or not _EMAIL.match(email.strip()):
            raise StoreError("INVALID_INPUT", "an email address is required")
        self._call("POST", "/auth/v1/otp", None, {"email": email.strip().lower(), "create_user": True})
        return {"requested": True, "delivery": "by the configured auth service; this API cannot confirm delivery"}

    def _session(self, data: Any) -> Dict[str, Any]:
        if not isinstance(data, dict) or not isinstance(data.get("access_token"), str) or \
                not isinstance(data.get("refresh_token"), str) or not isinstance(data.get("user"), dict):
            raise StoreError("BAD_RESPONSE", "the auth service did not return a session")
        user = data["user"]
        self._uuid(user.get("id"))
        return {"access_token": data["access_token"], "refresh_token": data["refresh_token"],
                "expires_in": int(data.get("expires_in") or 0),
                "user": {"id": user["id"], "email": user.get("email")}}

    def verify_code(self, email: Any, code: Any) -> Dict[str, Any]:
        if not isinstance(email, str) or not _EMAIL.match(email.strip()):
            raise StoreError("INVALID_INPUT", "an email address is required")
        if not isinstance(code, str) or not _CODE.match(code.strip()):
            raise StoreError("INVALID_INPUT", "the code is 6 to 10 digits")
        try:
            data = self._call("POST", "/auth/v1/verify", None,
                              {"type": "email", "email": email.strip().lower(), "token": code.strip()})
        except StoreError as e:
            if e.code in ("UNAUTHORIZED", "FORBIDDEN", "REJECTED"):
                raise StoreError("CODE_REJECTED", "the code is wrong or has expired") from None
            raise
        return self._session(data)

    def refresh(self, refresh_token: Any) -> Dict[str, Any]:
        if not isinstance(refresh_token, str) or not 6 <= len(refresh_token) <= 4096:
            raise StoreError("UNAUTHORIZED", "no refresh token")
        try:
            data = self._call("POST", "/auth/v1/token?grant_type=refresh_token", None, {"refresh_token": refresh_token})
        except StoreError as e:
            if e.code in ("FORBIDDEN", "REJECTED"):
                raise StoreError("UNAUTHORIZED", "the session has ended; sign in again") from None
            raise
        return self._session(data)

    def user(self, token: Any) -> Dict[str, Any]:
        self._token(token)
        try:
            data = self._call("GET", "/auth/v1/user", token)
        except StoreError as e:
            if e.code in ("FORBIDDEN", "REJECTED"):
                raise StoreError("UNAUTHORIZED", "no signed-in session") from None
            raise
        if not isinstance(data, dict):
            raise StoreError("UNAUTHORIZED", "no signed-in session")
        self._uuid(data.get("id"))
        return {"id": data["id"], "email": data.get("email")}

    def sign_out(self, token: Any) -> Dict[str, Any]:
        self._token(token)
        self._call("POST", "/auth/v1/logout", token, {})
        return {"signed_out": True}

    # ── projects (the caller's own, by row level security) ──
    def projects(self, token: str) -> List[Dict[str, Any]]:
        return self._rows("GET", "/osiris_projects?select=id,name,description,archived,owner_id,created_at"
                                 "&order=created_at.desc&limit=200", token)

    def create_project(self, token: str, name: Any) -> Dict[str, Any]:
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 200:
            raise StoreError("INVALID_INPUT", "name must be 1 to 200 characters")
        return self._one(self._rows("POST", "/osiris_projects", token, {"name": name.strip()}))   # no owner_id: the database sets it

    # ── files ──
    def files(self, token: str, project_id: Any) -> List[Dict[str, Any]]:
        self._uuid(project_id)
        return self._rows("GET", f"/osiris_files?select={_FILE_COLUMNS}&project_id=eq.{project_id}"
                                 "&order=path.asc&limit=500", token)

    def file(self, token: str, file_id: Any) -> Dict[str, Any]:
        self._uuid(file_id)
        rows = self._rows("GET", f"/osiris_files?select={_FILE_COLUMNS},content&id=eq.{file_id}", token)
        if len(rows) != 1:
            raise StoreError("NOT_FOUND", "no such file, or it is not yours")
        row = rows[0]
        # The digest is computed by the database; a mismatch means the row is not what it claims to be.
        if not isinstance(row.get("content"), str) or \
                hashlib.sha256(row["content"].encode("utf-8")).hexdigest() != row.get("content_sha256"):
            raise StoreError("BAD_RESPONSE", "stored content does not match its recorded digest")
        return row

    def create_file(self, token: str, project_id: Any, path: Any, content: Any = "") -> Dict[str, Any]:
        self._uuid(project_id)
        self._check_path(path)
        self._check_content(content)
        try:
            return self._one(self._rows("POST", "/osiris_files", token,
                                        {"project_id": project_id, "path": path, "content": content}))
        except StoreError as e:
            if e.code == "CONFLICT":
                raise StoreError("PATH_EXISTS", "a file with this path already exists in the project") from None
            raise

    def save_file(self, token: str, file_id: Any, base_revision: Any, content: Any) -> Dict[str, Any]:
        """Store ``content`` as the next revision only if the stored revision is still ``base_revision``.
        Returns {"status": "SAVED", "file": row} or {"status": "CONFLICT", "current": metadata}."""
        self._uuid(file_id)
        self._check_content(content)
        if isinstance(base_revision, bool) or not isinstance(base_revision, int) or base_revision < 1:
            raise StoreError("INVALID_INPUT", "base_revision must be a positive integer")
        rows = self._rows("PATCH", f"/osiris_files?id=eq.{file_id}&revision=eq.{base_revision}"
                                   f"&select={_FILE_COLUMNS}", token, {"content": content})
        if len(rows) == 1:
            row = rows[0]
            if row.get("revision") != base_revision + 1 or \
                    row.get("content_sha256") != hashlib.sha256(content.encode("utf-8")).hexdigest():
                raise StoreError("BAD_RESPONSE", "the store did not confirm the saved revision and digest")
            return {"status": "SAVED", "file": row}
        current = self._rows("GET", f"/osiris_files?select={_FILE_COLUMNS}&id=eq.{file_id}", token)
        if len(current) != 1:
            raise StoreError("NOT_FOUND", "no such file, or it is not yours")
        return {"status": "CONFLICT", "current": current[0]}

    def delete_file(self, token: str, file_id: Any) -> Dict[str, Any]:
        self._uuid(file_id)
        rows = self._rows("DELETE", f"/osiris_files?id=eq.{file_id}&select=id,path,revision", token)
        if len(rows) != 1:
            raise StoreError("NOT_FOUND", "no such file, or it is not yours")
        return {"deleted": rows[0]}

    # ── helpers ──
    @staticmethod
    def _check_path(path: Any) -> None:
        if not valid_path(path):
            raise StoreError("INVALID_INPUT", "path must be relative, at most 200 characters, made of segments of "
                                              "letters, digits, '_', '-' and '.', separated by '/'")

    @staticmethod
    def _check_content(content: Any) -> None:
        if not isinstance(content, str):
            raise StoreError("INVALID_INPUT", "content must be text")
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            raise StoreError("TOO_LARGE", f"a file may be at most {MAX_FILE_BYTES} bytes")

    def _one(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(rows) != 1 or not _UUID.match(str(rows[0].get("id", ""))):
            raise StoreError("BAD_RESPONSE", "the write did not return exactly one saved row")
        return rows[0]
