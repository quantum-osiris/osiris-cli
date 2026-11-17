"""OSIRIS IDE API v1: a loopback-only HTTP service the OSIRIS console talks to.

    python -m osiris_cli.ide_api --port 8765 --console-dir <quantum-ide>/public/osiris-console

Contract: docs/api/osiris-ide-api-v1.json (every response is validated against it in tests).

Exposure rules, all enforced here:
  * binds to 127.0.0.1 / ::1 only; any other address is refused at startup;
  * every /api/ request must carry X-OSIRIS-Token, a per-process secret injected into the console page
    and written to ~/.osiris/ide_session.token (0600); it never appears in logs or responses;
  * the Host header must be the loopback address the server is bound to (DNS-rebinding defence);
    a present Origin header must match it;
  * POST bodies must be application/json and at most 256 KiB;
  * there is deliberately no endpoint that executes a GitHub write. The console can only propose and
    dry-run; execution needs the owner's SYS-021 grant at a terminal (osiris_cli.github_gateway).
"""

from __future__ import annotations

import argparse
import hmac
import ipaddress
import json
import os
import secrets
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import urlparse

API_VERSION = "1.4.0"
MAX_BODY = 256 * 1024
TOKEN_HEADER = "X-OSIRIS-Token"
TOKEN_PLACEHOLDER = "__OSIRIS_SESSION_TOKEN__"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
CONTRACT_PATH = os.path.join(REPO_ROOT, "docs", "api", "osiris-ide-api-v1.json")
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".json": "application/json"}
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; "
       "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")


class ApiError(Exception):
    def __init__(self, status: int, kind: str, message: str):
        super().__init__(message)
        self.status, self.kind, self.message = status, kind, message


def _git(path: str, *args: str) -> str:
    r = subprocess.run(["git", "-C", path, *args], capture_output=True, text=True, timeout=10, check=False,
                       env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
    return r.stdout.strip() if r.returncode == 0 else ""


def local_repo_status(path: str) -> Dict[str, Any]:
    if not os.path.isdir(os.path.join(path, ".git")):
        return {"name": os.path.basename(path), "path": path, "present": False}
    head = _git(path, "rev-parse", "HEAD")
    branch = _git(path, "branch", "--show-current")
    dirty = bool(_git(path, "status", "--porcelain", "--untracked-files=no"))
    ahead = _git(path, "rev-list", "--count", "main..HEAD") if branch != "main" else "0"
    push = _git(path, "remote", "get-url", "--push", "origin")
    return {"name": os.path.basename(path), "path": path, "present": True, "branch": branch, "head": head[:12],
            "dirty": dirty, "commits_ahead_of_main": int(ahead) if ahead.isdigit() else None,
            "push_enabled": bool(push) and "DISABLED" not in push,
            "last_commit": _git(path, "log", "-1", "--format=%s")[:120]}


class Backend:
    """The service logic behind the HTTP layer. Tests call it directly as well as over HTTP."""

    def __init__(self, *, gateway_factory: Callable[[], Any], slice_receipts: str, examples_dir: str,
                 workspace_dirs: Tuple[str, ...] = (), slice_runner: Optional[Callable[..., Dict[str, Any]]] = None,
                 agent_receipts: Optional[str] = None, agent_model_fn: Optional[Callable[..., Any]] = None,
                 store_factory: Optional[Callable[[], Any]] = None,
                 editor_store_factory: Optional[Callable[[], Any]] = None,
                 assist_route_fn: Optional[Callable[..., Any]] = None):
        self._editor_store_factory = editor_store_factory
        self._assist_route_fn = assist_route_fn
        self._gateway_factory = gateway_factory
        self._gateway = None
        self._store_factory = store_factory
        self.slice_receipts = slice_receipts
        self.examples_dir = examples_dir
        self.workspace_dirs = workspace_dirs
        self._slice_runner = slice_runner
        self.agent_receipts = agent_receipts or os.path.join(os.path.dirname(os.path.abspath(slice_receipts)),
                                                             "agent_receipts.jsonl")
        self._agent_model_fn = agent_model_fn
        self._agent_lock = threading.Lock()
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def _get_store(self):
        from osiris_cli.workspace_store import StoreError, WorkspaceStore
        try:
            return self._store_factory() if self._store_factory is not None else WorkspaceStore()
        except StoreError as e:
            raise self._store_api_error(e)

    @staticmethod
    def _store_api_error(e: Any) -> ApiError:
        status = (400 if e.code == "INVALID_INPUT"
                  else 404 if e.code == "NOT_FOUND"
                  else 503 if e.code in ("NOT_CONFIGURED", "BAD_CONFIG", "WRONG_PROJECT")
                  else 502)
        return ApiError(status, f"STORE_{e.code}", str(e))

    @property
    def gateway(self):
        with self._lock:
            if self._gateway is None:
                self._gateway = self._gateway_factory()
            return self._gateway

    def _cached(self, key: str, ttl: float, fn: Callable[[], Any]) -> Any:
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < ttl:
            return hit[1]
        value = fn()
        self._cache[key] = (time.monotonic(), value)
        return value

    # ── endpoints ──
    def meta(self) -> Dict[str, Any]:
        contract_sha = None
        if os.path.isfile(CONTRACT_PATH):
            import hashlib
            contract_sha = "sha256:" + hashlib.sha256(open(CONTRACT_PATH, "rb").read()).hexdigest()
        return {"api_version": API_VERSION, "service": "osiris-ide-api", "contract_sha256": contract_sha,
                "evidence_labels": ["IMPLEMENTED", "TESTED", "SIMULATED", "MEASURED", "REPRODUCED", "HYPOTHESIS",
                                    "REFUTED"],
                "github_writes_via_api": False}

    def workspace(self) -> Dict[str, Any]:
        from osiris_cli.github_gateway import GatewayError
        try:
            gw = self.gateway
            github = gw.status()
            repos = self._cached("ecosystem", 60.0, lambda: gw.ecosystem()["repositories"]) \
                if github.get("identity_matches") else []
        except GatewayError as e:
            github, repos = {"authenticated": False, "login": None, "identity_matches": False, "error": e.kind,
                             "reason": e.message}, []
        return {"github": github, "repositories": repos,
                "local": [local_repo_status(p) for p in self.workspace_dirs]}

    def examples(self) -> Dict[str, Any]:
        out = []
        if os.path.isdir(self.examples_dir):
            for name in sorted(os.listdir(self.examples_dir)):
                if name.endswith(".dna"):
                    with open(os.path.join(self.examples_dir, name), encoding="utf-8") as fh:
                        out.append({"name": name, "source": fh.read()})
        return {"examples": out}

    def compile(self, body: Dict[str, Any]) -> Dict[str, Any]:
        from osiris_cli.vertical_slice import SliceInputError, compile_source
        src = body.get("source")
        try:
            res = compile_source(src)
        except SliceInputError as e:
            raise ApiError(400, "INVALID_INPUT", str(e))
        return {k: v for k, v in res.items() if not k.startswith("_")}

    def run_slice(self, body: Dict[str, Any]) -> Dict[str, Any]:
        from osiris_cli.vertical_slice import SliceInputError, run_slice
        runner = self._slice_runner or run_slice
        try:
            return runner(body.get("source"), shots=body.get("shots", 2000), seed=body.get("seed", 1234),
                          receipts_path=self.slice_receipts)
        except SliceInputError as e:
            raise ApiError(400, "INVALID_INPUT", str(e))

    def organism(self, body: Dict[str, Any]) -> Dict[str, Any]:
        from organism_sim.cli import load_spec
        from organism_sim.noise import Environment
        from organism_sim.organism import Organism
        ticks, seed, burst = body.get("ticks", 200), body.get("seed", 0), body.get("burst_prob", 0.05)
        if isinstance(ticks, bool) or not isinstance(ticks, int) or not 1 <= ticks <= 5000:
            raise ApiError(400, "INVALID_INPUT", "ticks must be an integer in [1, 5000]")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
            raise ApiError(400, "INVALID_INPUT", "seed must be an integer in [0, 2^31)")
        if isinstance(burst, bool) or not isinstance(burst, (int, float)) or not 0.0 <= burst <= 1.0:
            raise ApiError(400, "INVALID_INPUT", "burst_prob must be a number in [0, 1]")
        spec = load_spec(None)
        env = Environment(n_genes=len(spec.genome), seed=seed, noise_floor=spec.triggers.noise_floor,
                          burst_prob=float(burst))
        org = Organism(spec, env=env, seed=seed)
        series = []
        for _ in range(ticks):
            rec = org.step()
            st = rec.state if isinstance(rec.state, dict) else vars(rec.state)
            series.append({"tick": rec.tick, "coherence": st.get("coherence"), "noise_rate": st.get("noise_rate"),
                           "entropy_bits": st.get("entropy_bits"), "events": list(rec.events or [])[:4]})
        summary = org.summary().to_dict()
        return {"evidence_class": "SIMULATED", "engine": "organism_sim.Organism (spec ALPHA)", "ticks": ticks,
                "seed": seed, "burst_prob": float(burst), "summary": summary, "series": series}

    def receipts(self) -> Dict[str, Any]:
        from osiris_cli.trio_authz import Receipts
        rows = []
        valid = True
        if os.path.isfile(self.slice_receipts):
            valid = Receipts.verify(self.slice_receipts)
            with open(self.slice_receipts, encoding="utf-8") as fh:
                rows = [json.loads(ln) for ln in fh if ln.strip()]
        try:
            gh = self.gateway.receipts()
        except Exception as e:  # receipts must still render if GitHub is unavailable
            gh = {"path": None, "chain_valid": None, "count": 0, "rows": [], "error": str(e)}
        return {"slice": {"path": self.slice_receipts, "chain_valid": valid, "count": len(rows), "rows": rows[-50:]},
                "github": gh}

    def propose(self, body: Dict[str, Any]) -> Dict[str, Any]:
        from osiris_governance.canonical import strict_parse_json
        from osiris_governance.github_policy import WRITE_CAPABILITIES

        from osiris_cli.github_gateway import OPERATIONS, GatewayError
        op, repo, args = body.get("op"), body.get("repo"), body.get("args") or {}
        if not isinstance(args, dict):
            raise ApiError(400, "INVALID_INPUT", "args must be an object")
        try:
            gw = self.gateway
            proposal = gw.propose(op, repo, **args)
            decision = gw.decide(proposal)
            out = {"proposal": strict_parse_json(proposal), "decision": decision.to_dict(),
                   "is_write": OPERATIONS[op]["cap"] in WRITE_CAPABILITIES}
            if out["is_write"]:
                res = gw.execute(proposal, dry_run=True)
                out["dry_run"] = {k: res[k] for k in ("status", "plan", "reasons") if k in res}
                out["execution"] = ("not available through the API: run osiris_cli.github_gateway at a terminal "
                                    "with the owner's SYS-021 grant")
            return out
        except GatewayError as e:
            raise ApiError(400 if e.kind in ("INVALID_ARGUMENT", "OPERATION_NOT_ALLOWLISTED") else 502, e.kind, e.message)

    def claims(self) -> Dict[str, Any]:
        from osiris_cli import claims
        return {"source": "osiris_cli.claims.REGISTER", "count": len(claims.REGISTER),
                "claims": [{"id": c.id, "statement": c.statement, "standing": c.standing(), "verdict": c.verdict,
                            "finding": c.finding[:600], "testable": c.testable[:300],
                            "evidence": [{"kind": k, "ref": r, "note": n} for k, r, n in c.evidence]}
                           for c in claims.REGISTER]}

    def agent_task(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """OSIRIS runs AURA, AIDEN and IRIS on one objective and records the decision. Nothing is executed.
        When ``project_id`` is provided, the resulting disposition and receipt hash are persisted to Supabase."""
        from osiris_cli.orchestration import council
        from osiris_cli.workspace_store import StoreError, WorkspaceStore
        project_id = body.get("project_id")
        task_id = body.get("task_id")
        store = None
        if project_id is not None:
            try:
                WorkspaceStore._uuid(project_id)
                if task_id is not None:
                    WorkspaceStore._uuid(task_id)
            except StoreError as e:
                raise self._store_api_error(e)
            store = self._get_store()
        with self._agent_lock:  # one task at a time: the receipt chain is appended in order
            try:
                result = council.run_task(body.get("objective"), body.get("context"),
                                          receipts_path=self.agent_receipts, model_fn=self._agent_model_fn,
                                          allow_cloud=body.get("allow_cloud", False))
            except council.CouncilError as e:
                raise ApiError(400, "INVALID_INPUT", str(e))
            except council.ReceiptError as e:
                raise ApiError(500, "RECEIPT_NOT_WRITTEN", str(e))
            if store is not None:
                try:
                    saved = store.record_disposition(project_id, result, task_id=task_id)
                except StoreError as e:
                    raise self._store_api_error(e)
                result = {**result, "saved_disposition": saved}
            return result

    def agent_tasks(self) -> Dict[str, Any]:
        from osiris_cli.orchestration import council
        return council.task_receipts(self.agent_receipts)

    def store_status(self) -> Dict[str, Any]:
        from osiris_cli.workspace_store import PROJECT_REF, StoreError, store_status
        if self._store_factory is not None:
            try:
                self._store_factory()
                return {"configured": True, "project_ref": PROJECT_REF, "reason": None}
            except StoreError as e:
                return {"configured": False, "project_ref": PROJECT_REF, "reason": e.code}
        return store_status()

    def store_projects(self) -> Dict[str, Any]:
        from osiris_cli.workspace_store import StoreError
        try:
            return {"projects": self._get_store().projects()}
        except StoreError as e:
            raise self._store_api_error(e)

    def store_create_project(self, body: Dict[str, Any]) -> Dict[str, Any]:
        from osiris_cli.workspace_store import StoreError
        try:
            return {"project": self._get_store().create_project(body.get("name"), body.get("description", ""))}
        except StoreError as e:
            raise self._store_api_error(e)

    def store_project_detail(self, body: Dict[str, Any]) -> Dict[str, Any]:
        from osiris_cli.workspace_store import StoreError
        try:
            return {"project": self._get_store().project(body.get("project_id"))}
        except StoreError as e:
            raise self._store_api_error(e)

    def store_tasks(self, body: Dict[str, Any]) -> Dict[str, Any]:
        from osiris_cli.workspace_store import StoreError
        store = self._get_store()
        try:
            if body.get("task_id") is not None:
                return {"task": store.update_task(body.get("task_id"), status=body.get("status"), title=body.get("title"))}
            return {"task": store.create_task(body.get("project_id"), body.get("title"), status=body.get("status", "todo"))}
        except StoreError as e:
            raise self._store_api_error(e)

    # ── DNA-Lang editor (API 1.3.0): compiler identity, sign-in, a user's project files, bound runs ──
    def dnalang_info(self) -> Dict[str, Any]:
        """What this service can actually do for the editor. Versions come from the imported packages."""
        from osiris_cli import vertical_slice as vs
        from osiris_cli.editor_store import MAX_FILE_BYTES
        return {"compiler": {"name": "dnalang", "version": vs._versions()["dnalang"]},
                "versions": vs._versions(),
                "operations": {"parse": True, "check": True, "compile": True, "simulate": True,
                               "evolve": False, "hardware": False},
                "simulator": {"backend": "qiskit_aer.AerSimulator (noiseless, optimization_level=0)", "label": "SIMULATED",
                              "limits": {"qubits": vs.MAX_QUBITS, "operations": vs.MAX_OPS, "depth": vs.MAX_DEPTH,
                                         "shots": vs.MAX_SHOTS, "source_bytes": vs.MAX_SOURCE_BYTES}},
                "hardware": {"available": False,
                             "reason": "this API has no route that submits to a quantum processor"},
                "queue": {"supported": False, "cancel_supported": False,
                          "reason": "a run is one synchronous request; nothing is queued and nothing can be cancelled"},
                "files": {**(self._editor_status()), "max_file_bytes": MAX_FILE_BYTES}}

    def _editor_status(self) -> Dict[str, Any]:
        from osiris_cli.editor_store import target_status
        from osiris_cli.workspace_store import StoreError
        if self._editor_store_factory is not None:
            try:
                return {"configured": True, "target": self._editor_store_factory().target, "reason": None}
            except StoreError as e:
                return {"configured": False, "target": None, "reason": e.code}
        return target_status()

    def _editor(self):
        from osiris_cli.editor_store import EditorStore
        from osiris_cli.workspace_store import StoreError
        try:
            return self._editor_store_factory() if self._editor_store_factory is not None else EditorStore()
        except StoreError as e:
            raise self._editor_error(e)

    @staticmethod
    def _editor_error(e: Any) -> ApiError:
        status = {"INVALID_INPUT": 400, "CODE_REJECTED": 400, "UNAUTHORIZED": 401, "FORBIDDEN": 403, "NOT_FOUND": 404,
                  "PATH_EXISTS": 409, "CONFLICT": 409, "TOO_LARGE": 413, "RATE_LIMITED": 429,
                  "NOT_CONFIGURED": 503, "BAD_CONFIG": 503, "WRONG_PROJECT": 503, "UNAVAILABLE": 503}.get(e.code, 502)
        return ApiError(status, f"EDITOR_{e.code}", str(e))

    def _editor_call(self, fn: Callable[[Any], Dict[str, Any]]) -> Dict[str, Any]:
        from osiris_cli.workspace_store import StoreError
        store = self._editor()
        try:
            return fn(store)
        except StoreError as e:
            raise self._editor_error(e)

    def auth_request_code(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return self._editor_call(lambda s: s.request_code(body.get("email")))

    def auth_verify(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return self._editor_call(lambda s: {"session": s.verify_code(body.get("email"), body.get("code"))})

    def auth_refresh(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return self._editor_call(lambda s: {"session": s.refresh(body.get("refresh_token"))})

    def auth_user(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return self._editor_call(lambda s: {"user": s.user(body.get("user_token")), "target": s.target})

    def auth_signout(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return self._editor_call(lambda s: s.sign_out(body.get("user_token")))

    def files_projects(self, body: Dict[str, Any]) -> Dict[str, Any]:
        tok = body.get("user_token")
        if body.get("name") is not None:
            return self._editor_call(lambda s: {"project": s.create_project(tok, body.get("name"))})
        return self._editor_call(lambda s: {"projects": s.projects(tok)})

    def files_list(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return self._editor_call(lambda s: {"files": s.files(body.get("user_token"), body.get("project_id"))})

    def files_open(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return self._editor_call(lambda s: {"file": s.file(body.get("user_token"), body.get("file_id"))})

    def files_create(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return self._editor_call(lambda s: {"file": s.create_file(body.get("user_token"), body.get("project_id"),
                                                                  body.get("path"), body.get("content", ""))})

    def files_save(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return self._editor_call(lambda s: s.save_file(body.get("user_token"), body.get("file_id"),
                                                       body.get("base_revision"), body.get("content")))

    def files_delete(self, body: Dict[str, Any]) -> Dict[str, Any]:
        return self._editor_call(lambda s: s.delete_file(body.get("user_token"), body.get("file_id")))

    def files_run(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate one stored revision of a file the caller can read. The source is read from the store
        with the caller's token, never taken from the request, so the result is bound to that revision."""
        from osiris_cli.vertical_slice import SliceInputError, run_slice
        revision = body.get("revision")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ApiError(400, "INVALID_INPUT", "revision must be a positive integer")
        row = self._editor_call(lambda s: s.file(body.get("user_token"), body.get("file_id")))
        if row["revision"] != revision:
            return {"status": "REVISION_MISMATCH", "requested_revision": revision,
                    "current": {"revision": row["revision"], "content_sha256": row["content_sha256"]},
                    "executed": False}
        binding = {"project_id": row["project_id"], "file_id": row["id"], "path": row["path"],
                   "revision": row["revision"], "content_sha256": "sha256:" + row["content_sha256"]}
        runner = self._slice_runner or run_slice
        try:
            result = runner(row["content"], shots=body.get("shots", 2000), seed=body.get("seed", 1234),
                            receipts_path=self.slice_receipts, binding=binding)
        except SliceInputError as e:
            raise ApiError(400, "INVALID_INPUT", str(e))
        return {**result, "executed": result.get("status") in ("COMPLETED", "FAILED")}

    def notebook_run(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate one DNA-Lang cell of one stored notebook revision (API 1.4.0). As with files_run, the
        notebook is read from the store with the caller's token; no source text is taken from the request.
        The receipt's binding names the notebook file, its revision and digest, the cell id and the
        digest of that cell's source."""
        from osiris_cli import notebook_format as nbf
        from osiris_cli.vertical_slice import SliceInputError, run_slice
        revision, cell_id = body.get("revision"), body.get("cell_id")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ApiError(400, "INVALID_INPUT", "revision must be a positive integer")
        if not isinstance(cell_id, str) or not cell_id:
            raise ApiError(400, "INVALID_INPUT", "cell_id is required")
        row = self._editor_call(lambda s: s.file(body.get("user_token"), body.get("file_id")))
        if row["revision"] != revision:
            return {"status": "REVISION_MISMATCH", "requested_revision": revision,
                    "current": {"revision": row["revision"], "content_sha256": row["content_sha256"]},
                    "executed": False}
        try:
            nb = nbf.parse(row["content"])
        except nbf.NotebookFormatError as e:
            raise ApiError(400, "NOTEBOOK_INVALID", str(e))
        try:
            c = nbf.cell(nb, cell_id)
        except KeyError:
            raise ApiError(404, "NOTEBOOK_CELL_NOT_FOUND", f"revision {revision} has no cell {cell_id!r}")
        if c["type"] not in nbf.EXECUTABLE:
            raise ApiError(400, "NOTEBOOK_CELL_NOT_EXECUTABLE", f"cell {cell_id!r} is {c['type']}; only DNA-Lang cells run")
        binding = {"project_id": row["project_id"], "file_id": row["id"], "path": row["path"],
                   "revision": row["revision"], "content_sha256": "sha256:" + row["content_sha256"],
                   "notebook_format": f"{nbf.FORMAT}/{nbf.VERSION}", "cell_id": c["id"],
                   "cell_source_sha256": nbf.source_sha256(c["source"])}
        runner = self._slice_runner or run_slice
        try:
            result = runner(c["source"], shots=body.get("shots", 2000), seed=body.get("seed", 1234),
                            receipts_path=self.slice_receipts, binding=binding)
        except SliceInputError as e:
            raise ApiError(400, "INVALID_INPUT", str(e))
        return {**result, "executed": result.get("status") in ("COMPLETED", "FAILED")}

    def files_runs(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Receipted runs of one file, read from the local receipt chain. The caller must be able to read the file."""
        from osiris_cli.trio_authz import Receipts
        row = self._editor_call(lambda s: s.file(body.get("user_token"), body.get("file_id")))
        runs, valid = [], None
        if os.path.isfile(self.slice_receipts):
            valid = Receipts.verify(self.slice_receipts)
            with open(self.slice_receipts, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    p = r.get("payload") or {}
                    if (p.get("source_binding") or {}).get("file_id") == row["id"]:
                        runs.append({"run_id": p.get("run_id"), "kind": r.get("kind"), "ts": r.get("ts"),
                                     "receipt_hash": r.get("hash"), "source_binding": p.get("source_binding"),
                                     "source_sha256": p.get("source_sha256"), "circuit_sha256": p.get("circuit_sha256"),
                                     "shots": p.get("shots"), "seed": p.get("seed"), "counts": p.get("counts"),
                                     "evidence_class": p.get("evidence_class"), "versions": p.get("versions"),
                                     "failed_stage": p.get("failed_stage")})
        return {"file_id": row["id"], "chain_valid": valid, "runs": runs[-50:]}

    def assist(self, body: Dict[str, Any]) -> Dict[str, Any]:
        from osiris_cli import editor_assist
        try:
            return editor_assist.assist(body, route_fn=self._assist_route_fn)
        except editor_assist.AssistError as e:
            raise ApiError(400, "INVALID_INPUT", str(e))

    def experiments(self) -> Dict[str, Any]:
        rows = self.receipts()["slice"]
        runs = [{"run_id": r["payload"].get("run_id"), "kind": r["kind"], "ts": r["ts"], "hash": r["hash"],
                 "circuit_sha256": r["payload"].get("circuit_sha256"), "shots": r["payload"].get("shots"),
                 "evidence_class": r["payload"].get("evidence_class"),
                 "claim_category": r["payload"].get("claim_category"), "tvd": r["payload"].get("tvd"),
                 "failed_stage": r["payload"].get("failed_stage")} for r in rows["rows"]]
        return {"chain_valid": rows["chain_valid"], "runs": runs}


ROUTES = {
    ("GET", "/api/v1/meta"): "meta",
    ("GET", "/api/v1/workspace"): "workspace",
    ("GET", "/api/v1/examples"): "examples",
    ("POST", "/api/v1/dnalang/compile"): "compile",
    ("POST", "/api/v1/slice/run"): "run_slice",
    ("POST", "/api/v1/organism/run"): "organism",
    ("GET", "/api/v1/governance/receipts"): "receipts",
    ("POST", "/api/v1/governance/github/propose"): "propose",
    ("GET", "/api/v1/research/claims"): "claims",
    ("GET", "/api/v1/research/experiments"): "experiments",
    ("POST", "/api/v1/agents/task"): "agent_task",
    ("GET", "/api/v1/agents/tasks"): "agent_tasks",
    ("GET", "/api/v1/store/status"): "store_status",
    ("GET", "/api/v1/store/projects"): "store_projects",
    ("POST", "/api/v1/store/projects"): "store_create_project",
    ("POST", "/api/v1/store/project"): "store_project_detail",
    ("POST", "/api/v1/store/tasks"): "store_tasks",
    ("GET", "/api/v1/dnalang/info"): "dnalang_info",
    ("POST", "/api/v1/auth/request-code"): "auth_request_code",
    ("POST", "/api/v1/auth/verify"): "auth_verify",
    ("POST", "/api/v1/auth/refresh"): "auth_refresh",
    ("POST", "/api/v1/auth/user"): "auth_user",
    ("POST", "/api/v1/auth/signout"): "auth_signout",
    ("POST", "/api/v1/files/projects"): "files_projects",
    ("POST", "/api/v1/files/list"): "files_list",
    ("POST", "/api/v1/files/open"): "files_open",
    ("POST", "/api/v1/files/create"): "files_create",
    ("POST", "/api/v1/files/save"): "files_save",
    ("POST", "/api/v1/files/delete"): "files_delete",
    ("POST", "/api/v1/files/run"): "files_run",
    ("POST", "/api/v1/files/runs"): "files_runs",
    ("POST", "/api/v1/notebook/run"): "notebook_run",
    ("POST", "/api/v1/assist"): "assist",
}


# Every operation that reaches Qiskit/Aer runs on this one long-lived thread. With qiskit 2.6.2 and
# qiskit-aer 0.18.2 on Python 3.14, a second simulation started from a *different* short-lived thread
# (ThreadingHTTPServer opens one per request) segfaults the process inside qiskit._accelerate
# (get_standard_gate_name_mapping). Reproduced outside the server; one shared worker thread avoids it.
# This also serialises simulations, which the API already documents as synchronous and unqueued.
QISKIT_OPERATIONS = frozenset({"compile", "run_slice", "organism", "files_run", "notebook_run"})
_QISKIT_THREAD = ThreadPoolExecutor(max_workers=1, thread_name_prefix="osiris-qiskit")


def make_handler(backend: Backend, token: str, console_dir: Optional[str], allowed_hosts: Tuple[str, ...]):
    class Handler(BaseHTTPRequestHandler):
        server_version = "osiris-ide-api/" + API_VERSION
        sys_version = ""

        def log_message(self, fmt, *args):  # never log headers (they carry the token)
            sys.stderr.write("[ide-api] %s %s\n" % (self.command, self.path.split("?")[0]))

        def _send(self, status: int, payload: bytes, ctype: str, extra: Optional[Dict[str, str]] = None):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", CSP)
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, status: int, obj: Any):
            self._send(status, json.dumps(obj, default=str).encode(), "application/json")

        def _error(self, e: ApiError):
            self._json(e.status, {"error": {"kind": e.kind, "message": e.message}, "api_version": API_VERSION})

        def _check_origin(self):
            host = (self.headers.get("Host") or "").lower()
            if host not in allowed_hosts:
                raise ApiError(421, "HOST_REJECTED", "Host header is not this loopback service")
            origin = self.headers.get("Origin")
            if origin is not None and urlparse(origin).netloc.lower() not in allowed_hosts:
                raise ApiError(403, "ORIGIN_REJECTED", "cross-origin requests are refused")

        def _dispatch(self, method: str):
            path = self.path.split("?", 1)[0]
            try:
                self._check_origin()
                if not path.startswith("/api/"):
                    if method != "GET":
                        raise ApiError(405, "METHOD_NOT_ALLOWED", "static files are GET only")
                    return self._static(path)
                provided_token = (
                    self.headers.get(TOKEN_HEADER)
                    or self.headers.get("X-Session-Token")
                    or (
                        self.headers.get("Authorization", "")[7:].strip()
                        if self.headers.get("Authorization", "").lower().startswith("bearer ")
                        else ""
                    )
                )
                if not hmac.compare_digest(provided_token, token):
                    raise ApiError(
                        401,
                        "TOKEN_REQUIRED",
                        f"missing or wrong {TOKEN_HEADER}, X-Session-Token, or Authorization Bearer credential",
                    )
                name = ROUTES.get((method, path))
                if name is None:
                    raise ApiError(404 if not any(p == path for _, p in ROUTES) else 405, "NO_ROUTE",
                                   f"{method} {path} is not part of API v1")
                body: Dict[str, Any] = {}
                if method == "POST":
                    if (self.headers.get("Content-Type") or "").split(";")[0].strip() != "application/json":
                        raise ApiError(415, "JSON_REQUIRED", "POST bodies must be application/json")
                    n = int(self.headers.get("Content-Length") or 0)
                    if n > MAX_BODY:
                        raise ApiError(413, "BODY_TOO_LARGE", f"body exceeds {MAX_BODY} bytes")
                    try:
                        body = json.loads(self.rfile.read(n) or b"{}")
                    except json.JSONDecodeError:
                        raise ApiError(400, "BAD_JSON", "body is not valid JSON")
                    if not isinstance(body, dict):
                        raise ApiError(400, "BAD_JSON", "body must be a JSON object")
                fn = getattr(backend, name)
                call = (lambda: fn(body)) if method == "POST" else fn
                result = _QISKIT_THREAD.submit(call).result() if name in QISKIT_OPERATIONS else call()
                self._json(200, {"api_version": API_VERSION, **result})
            except ApiError as e:
                self._error(e)
            except Exception as e:  # never leak a traceback to the client
                sys.stderr.write(f"[ide-api] internal error on {path}: {type(e).__name__}: {e}\n")
                self._error(ApiError(500, "INTERNAL", f"{type(e).__name__}"))

        def _static(self, path: str):
            if not console_dir:
                return self._send(200, b"OSIRIS IDE API v1 is running. Start with --console-dir to serve the console.",
                                  "text/plain; charset=utf-8")
            rel = "index.html" if path in ("/", "") else path.lstrip("/")
            full = os.path.realpath(os.path.join(console_dir, rel))
            if not full.startswith(os.path.realpath(console_dir) + os.sep) or not os.path.isfile(full):
                raise ApiError(404, "NOT_FOUND", "no such file")
            ext = os.path.splitext(full)[1]
            if ext not in STATIC_TYPES:
                raise ApiError(404, "NOT_FOUND", "unsupported file type")
            data = open(full, "rb").read()
            if rel == "index.html":
                data = data.replace(TOKEN_PLACEHOLDER.encode(), token.encode())
            self._send(200, data, STATIC_TYPES[ext])

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

        def do_PUT(self):
            self._error(ApiError(405, "METHOD_NOT_ALLOWED", "not allowed"))

        do_DELETE = do_PATCH = do_PUT

    return Handler


def serve(backend: Backend, *, host: str = "127.0.0.1", port: int = 8765, console_dir: Optional[str] = None,
          token: Optional[str] = None, token_file: Optional[str] = None) -> Tuple[ThreadingHTTPServer, str]:
    """Create (not start) the server. Refuses any non-loopback bind address."""
    if not ipaddress.ip_address(host).is_loopback:
        raise ValueError(f"refusing to bind {host}: the OSIRIS IDE API is loopback-only")
    token = token or secrets.token_urlsafe(32)
    httpd = ThreadingHTTPServer((host, port), None)
    bound_port = httpd.server_address[1]
    hostpart = f"[{host}]" if ":" in host else host
    allowed = (f"{hostpart}:{bound_port}", f"localhost:{bound_port}")
    httpd.RequestHandlerClass = make_handler(backend, token, console_dir, allowed)
    if token_file:
        os.makedirs(os.path.dirname(token_file), mode=0o700, exist_ok=True)
        fd = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(token)
    return httpd, token


def default_backend(examples_dir: Optional[str] = None, workspace_dirs: Tuple[str, ...] = (),
                    slice_receipts: Optional[str] = None, github_receipts: Optional[str] = None,
                    agent_receipts: Optional[str] = None) -> Backend:
    from osiris_cli.github_gateway import DEFAULT_RECEIPTS_PATH, GitHubGateway
    from osiris_cli.vertical_slice import DEFAULT_RECEIPTS_PATH as SLICE_DEFAULT
    if examples_dir is None:
        try:
            import dnalang
            examples_dir = os.path.join(os.path.dirname(os.path.dirname(dnalang.__file__)), "examples")
        except ImportError:
            examples_dir = ""
    return Backend(gateway_factory=lambda: GitHubGateway(receipts_path=github_receipts or DEFAULT_RECEIPTS_PATH),
                   slice_receipts=slice_receipts or SLICE_DEFAULT, examples_dir=examples_dir,
                   workspace_dirs=workspace_dirs, agent_receipts=agent_receipts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m osiris_cli.ide_api", description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--console-dir", default=os.environ.get("OSIRIS_CONSOLE_DIR"))
    ap.add_argument("--examples-dir")
    ap.add_argument("--workspace", action="append", default=[], help="local git checkout to report (repeatable)")
    ap.add_argument("--slice-receipts")
    ap.add_argument("--github-receipts")
    ap.add_argument("--agent-receipts", help="receipt chain for AURA/AIDEN/IRIS tasks (default: beside --slice-receipts)")
    ap.add_argument("--token-file", default=os.path.join(os.path.expanduser("~"), ".osiris", "ide_session.token"))
    a = ap.parse_args(argv)
    backend = default_backend(a.examples_dir, tuple(a.workspace), a.slice_receipts, a.github_receipts,
                              a.agent_receipts)
    httpd, _ = serve(backend, host=a.host, port=a.port, console_dir=a.console_dir, token_file=a.token_file)
    print(f"OSIRIS IDE API v{API_VERSION} on http://{a.host}:{httpd.server_address[1]}/  "
          f"(session token in {a.token_file})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
