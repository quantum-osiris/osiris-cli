"""Durable run records for headless /code tasks.

One JSON file per run (``<dir>/<run_id>.json``), rewritten atomically at every state
transition, so a crash leaves the last committed state on disk. Side files:

* ``<run_id>.lock``  -- flock held while a process executes the run (no double resume);
* ``<run_id>.cancel`` -- written by ``request_cancel``; the runner checks it at every
  checkpoint, and a run whose cancel arrived during verification ends CANCELLED, never
  SUCCEEDED.

Lifecycle::

    CREATED -> CONTEXT_READY -> RUNNING -> AWAITING_APPROVAL -> VERIFYING -> SUCCEEDED
                                       \\-> VERIFYING (no proposals)     \\-> FAILED | BLOCKED
    any non-terminal -> FAILED | BLOCKED | CANCELLED | UNAVAILABLE | STOPPED_BUDGET

``STOPPED_BUDGET`` is its own terminal state, not a kind of FAILED or BLOCKED: the tool loop
reached a cap of the run's task context (``osiris_cli.task_context``: steps, tool calls, tokens,
wall clock, or failures in a row) before it answered. Nothing it staged is applied.

``RETRYING`` is recorded as a history event (the runner's bounded write nudge), not a
resting state. There is no separate planner yet, so there is no PLANNED state.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

SCHEMA = "osiris.code_run/1"
TERMINAL = frozenset({"SUCCEEDED", "FAILED", "BLOCKED", "CANCELLED", "UNAVAILABLE", "STOPPED_BUDGET"})
_END = {"FAILED", "BLOCKED", "CANCELLED", "UNAVAILABLE", "STOPPED_BUDGET"}
ALLOWED: Dict[str, frozenset] = {
    "CREATED": frozenset({"CONTEXT_READY"} | _END),
    "CONTEXT_READY": frozenset({"RUNNING"} | _END),
    "RUNNING": frozenset({"AWAITING_APPROVAL", "VERIFYING"} | _END),
    "AWAITING_APPROVAL": frozenset({"VERIFYING"} | _END),
    "VERIFYING": frozenset({"SUCCEEDED"} | _END),
}
DEFAULT_DIR = os.path.join(os.path.expanduser("~"), ".osiris", "code", "runs")


class RunError(Exception):
    """An operation the run's current state does not allow."""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def source_snapshot(root: str) -> Dict[str, Any]:
    """git HEAD and a digest of `git status --porcelain` (None outside a repository)."""
    def git(*args: str) -> Optional[str]:
        try:
            r = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, timeout=20,
                               stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return r.stdout if r.returncode == 0 else None

    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    return {"git_head": head.strip() if head else None,
            "git_status_sha256": hashlib.sha256(status.encode()).hexdigest() if status is not None else None,
            "git_dirty": bool(status.strip()) if status is not None else None}


class RunStore:
    def __init__(self, directory: str = DEFAULT_DIR):
        self.dir = directory
        os.makedirs(self.dir, exist_ok=True)

    def _path(self, run_id: str, ext: str = "json") -> str:
        if not run_id or os.sep in run_id or run_id.startswith("."):
            raise RunError(f"bad run id {run_id!r}")
        return os.path.join(self.dir, f"{run_id}.{ext}")

    # ── records ──────────────────────────────────────────────────────────────
    def create(self, run_id: str, **fields: Any) -> Dict[str, Any]:
        if os.path.exists(self._path(run_id)):
            raise RunError(f"run {run_id} already exists")
        run = {"schema": SCHEMA, "run_id": run_id, "state": "CREATED", "created_at": _now(),
               "history": [{"state": "CREATED", "ts": _now(), "reason": ""}], "receipts": [], **fields}
        self.save(run)
        return run

    def load(self, run_id: str) -> Dict[str, Any]:
        try:
            with open(self._path(run_id), encoding="utf-8") as f:
                run = json.load(f)
        except FileNotFoundError:
            raise RunError(f"no run {run_id} in {self.dir}")
        if run.get("schema") != SCHEMA:
            raise RunError(f"run {run_id} has unknown schema {run.get('schema')!r}")
        return run

    def save(self, run: Dict[str, Any]) -> None:
        path = self._path(run["run_id"])
        tmp = f"{path}.tmp-{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(run, f, indent=1, sort_keys=True, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def transition(self, run: Dict[str, Any], state: str, reason: str = "") -> Dict[str, Any]:
        cur = run["state"]
        if state not in ALLOWED.get(cur, frozenset()):
            raise RunError(f"{run['run_id']}: {cur} -> {state} is not an allowed transition")
        run["state"] = state
        run["history"].append({"state": state, "ts": _now(), "reason": reason[:500]})
        self.save(run)
        return run

    def event(self, run: Dict[str, Any], name: str, reason: str = "") -> None:
        """A recorded transition that is not a resting state (e.g. RETRYING)."""
        run["history"].append({"event": name, "ts": _now(), "reason": reason[:500]})
        self.save(run)

    def list(self) -> List[Dict[str, Any]]:
        rows = []
        for name in sorted(os.listdir(self.dir)):
            if name.endswith(".json"):
                try:
                    r = self.load(name[:-5])
                except (RunError, ValueError):
                    continue
                rows.append({k: r.get(k) for k in ("run_id", "state", "created_at", "root")})
        return rows

    # ── cancel and lock ──────────────────────────────────────────────────────
    def request_cancel(self, run_id: str) -> Dict[str, Any]:
        run = self.load(run_id)
        if run["state"] in TERMINAL:
            raise RunError(f"run {run_id} is already {run['state']}")
        with open(self._path(run_id, "cancel"), "w", encoding="utf-8") as f:
            f.write(_now() + "\n")
        return run

    def cancel_requested(self, run_id: str) -> bool:
        return os.path.exists(self._path(run_id, "cancel"))

    @contextmanager
    def locked(self, run_id: str) -> Iterator[None]:
        fd = os.open(self._path(run_id, "lock"), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RunError(f"run {run_id} is being executed by another process")
            yield
        finally:
            os.close(fd)
