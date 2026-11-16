"""OSIRIS GitHub Gateway: governed, allowlisted access to GitHub through the authenticated gh CLI.

Every operation, read or write, becomes an ``osiris_governance.contracts.ActionProposal`` whose
canonical bytes are decided by ``osiris_governance.github_policy.evaluate_github_proposal``. The
HTTP request is rebuilt from those exact bytes, so what was decided is what runs.

Reads run as soon as the policy allows them. Writes additionally need, in order:
  1. policy ALLOW (writes_enabled, every security gate PASSED, capability granted for the repo);
  2. not a duplicate of a proposal already attempted (by proposal digest or request id);
  3. the owner's SYS-021 grant (``osiris_cli.grants.authorize``) naming this proposal's digest and
     the scope ``github:<capability>:<owner/repo>``: an ssh-signed, single-use approval.
A write-ahead receipt is appended before the request is sent, and a result receipt after.

Credentials: only gh's own stored login is used. This module never reads, prints or stores a token.
"""

from __future__ import annotations

import base64
import os
import re
import secrets
import threading
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import quote, urlencode

from osiris_governance.canonical import canonical_sha256, canonicalize_json, strict_parse_json
from osiris_governance.contracts import ActionProposal
from osiris_governance.github_policy import (
    GITHUB_ACTION_KIND,
    WRITE_CAPABILITIES,
    GitHubPolicy,
    PolicyDecision,
    evaluate_github_proposal,
)

from osiris_cli.github_bridge import GitHubBridge, redact
from osiris_cli.trio_authz import Receipts

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_POLICY_PATH = os.path.join(os.path.dirname(HERE), "governance", "github_policy.json")
DEFAULT_RECEIPTS_PATH = os.path.join(os.path.expanduser("~"), ".osiris", "receipts", "github_gateway.jsonl")
PROPOSAL_TTL_S = 15 * 60

_REPO = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})/[A-Za-z0-9._-]{1,100}$")
# A git ref: no "..", no leading "-" or "/", no trailing "/" (git check-ref-format forbids all three).
_REF = re.compile(r"^(?!.*\.\.)(?![-/])[A-Za-z0-9._/-]{1,200}(?<!/)$")
_PATH = re.compile(r"^(?!.*\.\.)[A-Za-z0-9._/-]{1,300}$")


class GatewayError(Exception):
    """A request the gateway refuses or cannot complete. ``kind`` is stable for callers and tests."""

    def __init__(self, kind: str, message: str, detail: Optional[Dict[str, Any]] = None):
        super().__init__(f"{kind}: {message}")
        self.kind, self.message, self.detail = kind, message, detail or {}

    def to_dict(self) -> Dict[str, Any]:
        return {"status": "ERROR", "kind": self.kind, "reason": self.message, **({"detail": self.detail}
                                                                                  if self.detail else {})}


# ── argument validation ──────────────────────────────────────────────────────────────────────────
def _int(lo: int, hi: int) -> Callable[[Any], int]:
    def check(v: Any) -> int:
        if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
            raise GatewayError("INVALID_ARGUMENT", f"expected an integer in [{lo}, {hi}], got {v!r}")
        return v
    return check


def _str(pattern: Optional[re.Pattern] = None, max_len: int = 256, min_len: int = 1) -> Callable[[Any], str]:
    def check(v: Any) -> str:
        if not isinstance(v, str) or not min_len <= len(v) <= max_len or (pattern and not pattern.match(v)):
            raise GatewayError("INVALID_ARGUMENT", f"invalid string argument {v!r:.80}")
        if "\x00" in v:
            raise GatewayError("INVALID_ARGUMENT", "NUL byte in argument")
        return v
    return check


def _opt_ref(v: Any) -> str:
    """An optional git ref: empty means "the default branch"."""
    return "" if v == "" else _str(_REF)(v)


def _bool(v: Any) -> bool:
    if not isinstance(v, bool):
        raise GatewayError("INVALID_ARGUMENT", f"expected a boolean, got {v!r}")
    return v


# Each operation: capability, HTTP method, required args, optional args (with defaults), request builder.
# Builders receive validated args and return (endpoint, body). Nothing else reaches gh.
def _q(repo: str) -> str:
    owner, name = repo.split("/", 1)
    return f"repos/{quote(owner, safe='')}/{quote(name, safe='')}"


OPERATIONS: Dict[str, Dict[str, Any]] = {
    "identity": dict(cap="github.identity.read", method="GET", req={}, opt={},
                     build=lambda r, a: ("user", None)),
    "repos.discover": dict(cap="github.identity.read", method="GET", req={}, opt={"limit": (_int(1, 100), 100)},
                           build=lambda r, a: ("user/repos?" + urlencode({"per_page": a["limit"],
                                                                         "sort": "pushed"}), None)),
    "repo.view": dict(cap="github.repo.read", method="GET", req={}, opt={}, build=lambda r, a: (_q(r), None)),
    "branches.list": dict(cap="github.branch.read", method="GET", req={}, opt={"limit": (_int(1, 100), 30)},
                          build=lambda r, a: (f"{_q(r)}/branches?" + urlencode({"per_page": a["limit"]}), None)),
    "commits.list": dict(cap="github.commit.read", method="GET", req={},
                         opt={"ref": (_opt_ref, ""), "limit": (_int(1, 100), 10)},
                         build=lambda r, a: (f"{_q(r)}/commits?" + urlencode(
                             {"per_page": a["limit"], **({"sha": a["ref"]} if a["ref"] else {})}), None)),
    "issues.list": dict(cap="github.issue.read", method="GET", req={}, opt={"limit": (_int(1, 100), 20)},
                        build=lambda r, a: (f"{_q(r)}/issues?" + urlencode({"state": "open",
                                                                            "per_page": a["limit"]}), None)),
    "pulls.list": dict(cap="github.pr.read", method="GET", req={}, opt={"limit": (_int(1, 100), 20)},
                       build=lambda r, a: (f"{_q(r)}/pulls?" + urlencode({"state": "open",
                                                                          "per_page": a["limit"]}), None)),
    "pull.view": dict(cap="github.pr.read", method="GET", req={"number": _int(1, 10**9)}, opt={},
                      build=lambda r, a: (f"{_q(r)}/pulls/{a['number']}", None)),
    "workflow_runs.list": dict(cap="github.workflow.read", method="GET", req={}, opt={"limit": (_int(1, 100), 10)},
                               build=lambda r, a: (f"{_q(r)}/actions/runs?" + urlencode({"per_page": a["limit"]}),
                                                   None)),
    "contents.get": dict(cap="github.contents.read", method="GET", req={"path": _str(_PATH, 300)},
                         opt={"ref": (_opt_ref, "")},
                         build=lambda r, a: (f"{_q(r)}/contents/{quote(a['path'])}"
                                             + (("?" + urlencode({"ref": a["ref"]})) if a["ref"] else ""), None)),
    # ── writes ──
    "pr.create": dict(cap="github.pr.create", method="POST",
                      req={"title": _str(max_len=256), "head": _str(_REF), "base": _str(_REF)},
                      opt={"body": (_str(max_len=65536, min_len=0), ""), "draft": (_bool, True)},
                      build=lambda r, a: (f"{_q(r)}/pulls", {"title": a["title"], "head": a["head"],
                                                             "base": a["base"], "body": a["body"],
                                                             "draft": a["draft"]})),
    "issue.create": dict(cap="github.issue.create", method="POST",
                         req={"title": _str(max_len=256)}, opt={"body": (_str(max_len=65536, min_len=0), "")},
                         build=lambda r, a: (f"{_q(r)}/issues", {"title": a["title"], "body": a["body"]})),
    "issue.comment": dict(cap="github.issue.comment", method="POST",
                          req={"number": _int(1, 10**9), "body": _str(max_len=65536)}, opt={},
                          build=lambda r, a: (f"{_q(r)}/issues/{a['number']}/comments", {"body": a["body"]})),
}
CAP_TO_OP = {}
for _name, _spec in OPERATIONS.items():
    CAP_TO_OP.setdefault(_spec["cap"], []).append(_name)


def validate_arguments(op: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if op not in OPERATIONS:
        raise GatewayError("OPERATION_NOT_ALLOWLISTED", f"{op!r} is not a gateway operation")
    spec = OPERATIONS[op]
    if not isinstance(args, dict):
        raise GatewayError("INVALID_ARGUMENT", "arguments must be an object")
    unknown = set(args) - set(spec["req"]) - set(spec["opt"]) - {"op"}
    if unknown:
        raise GatewayError("INVALID_ARGUMENT", f"unknown arguments for {op}: {sorted(unknown)}")
    out: Dict[str, Any] = {}
    for k, check in spec["req"].items():
        if k not in args:
            raise GatewayError("INVALID_ARGUMENT", f"{op} requires {k!r}")
        out[k] = check(args[k])
    for k, (check, default) in spec["opt"].items():
        out[k] = check(args[k]) if k in args else default
    return out


# ── response shaping: return only fields the IDE and CLI use, never the raw API objects ──────────
def _shape(op: str, data: Any) -> Any:
    if op == "identity":
        return {k: data.get(k) for k in ("login", "id", "html_url", "created_at")}
    if op in ("repos.discover",):
        return [{"full_name": d["full_name"], "private": d["private"], "default_branch": d.get("default_branch"),
                 "pushed_at": d.get("pushed_at"), "fork": d.get("fork")} for d in data]
    if op == "repo.view":
        perms = data.get("permissions") or {}
        return {"full_name": data["full_name"], "private": data["private"], "default_branch": data["default_branch"],
                "pushed_at": data.get("pushed_at"), "open_issues_count": data.get("open_issues_count"),
                "archived": data.get("archived"),
                "viewer_permission": next((p for p in ("admin", "maintain", "push", "triage", "pull")
                                           if perms.get(p)), None)}
    if op == "branches.list":
        return [{"name": b["name"], "sha": b["commit"]["sha"], "protected": b.get("protected")} for b in data]
    if op == "commits.list":
        return [{"sha": c["sha"], "author": (c["commit"]["author"] or {}).get("name"),
                 "date": (c["commit"]["author"] or {}).get("date"),
                 "message": c["commit"]["message"].splitlines()[0][:200] if c["commit"]["message"] else "",
                 "verified": (c["commit"].get("verification") or {}).get("verified")} for c in data]
    if op == "issues.list":
        return [{"number": i["number"], "title": i["title"], "user": (i.get("user") or {}).get("login"),
                 "updated_at": i.get("updated_at")} for i in data if "pull_request" not in i]
    if op in ("pulls.list",):
        return [{"number": p["number"], "title": p["title"], "head": p["head"]["ref"], "base": p["base"]["ref"],
                 "draft": p.get("draft"), "user": (p.get("user") or {}).get("login")} for p in data]
    if op == "pull.view":
        return {"number": data["number"], "title": data["title"], "state": data["state"], "head": data["head"]["ref"],
                "base": data["base"]["ref"], "draft": data.get("draft"), "mergeable": data.get("mergeable"),
                "html_url": data.get("html_url")}
    if op == "workflow_runs.list":
        return [{"id": w["id"], "name": w.get("name"), "status": w.get("status"), "conclusion": w.get("conclusion"),
                 "head_sha": w.get("head_sha"), "branch": w.get("head_branch"), "created_at": w.get("created_at")}
                for w in (data or {}).get("workflow_runs", [])]
    if op == "contents.get":
        if not isinstance(data, dict) or data.get("type") != "file":
            raise GatewayError("NOT_A_FILE", "path is not a file")
        raw = base64.b64decode(data.get("content", "")) if data.get("encoding") == "base64" else b""
        return {"path": data["path"], "sha": data["sha"], "size": data["size"], "text": raw.decode("utf-8", "replace")}
    if op in ("pr.create", "issue.create"):
        return {"number": data.get("number"), "html_url": data.get("html_url")}
    if op == "issue.comment":
        return {"id": data.get("id"), "html_url": data.get("html_url")}
    return data


class GitHubGateway:
    """Governed GitHub access. Construct once per process; it is thread-safe."""

    def __init__(self, policy_path: str = DEFAULT_POLICY_PATH, receipts_path: str = DEFAULT_RECEIPTS_PATH,
                 bridge: Optional[GitHubBridge] = None, clock: Callable[[], datetime] = None,
                 proposer_id: str = "osiris-cli/github_gateway"):
        try:
            with open(policy_path, "rb") as fh:
                self.policy = GitHubPolicy.from_json(fh.read())
        except OSError as e:
            raise GatewayError("POLICY_UNAVAILABLE", f"cannot read policy {policy_path}: {e}") from e
        self.policy_path = policy_path
        self.receipts_path = receipts_path
        self.bridge = bridge or GitHubBridge()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.proposer_id = proposer_id
        self._lock = threading.Lock()
        self._login: Optional[str] = None

    # ── identity ──
    def observed_login(self, refresh: bool = False) -> str:
        """The authenticated login according to the API. Raises GatewayError if it cannot be determined."""
        if self._login and not refresh:
            return self._login
        r = self.bridge.authenticated_login()
        if r["status"] != "SUCCESS":
            kind = {"NOT_INSTALLED": "GH_NOT_INSTALLED", "AUTH": "NOT_AUTHENTICATED"}.get(r.get("kind"), r.get("kind"))
            raise GatewayError(kind or "IDENTITY_UNAVAILABLE", r.get("reason", "cannot determine GitHub identity"))
        self._login = r["login"]
        return self._login

    def status(self) -> Dict[str, Any]:
        """Authentication summary for the IDE: never raises, never includes a token."""
        out = {"expected_login": self.policy.expected_login, "policy_version": self.policy.policy_version,
               "policy_sha256": self.policy.policy_sha256, "writes_enabled": self.policy.writes_enabled,
               "security_gates": dict(self.policy.security_gates)}
        try:
            login = self.observed_login(refresh=True)
            out.update(authenticated=True, login=login, identity_matches=(login == self.policy.expected_login))
        except GatewayError as e:
            out.update(authenticated=False, login=None, identity_matches=False, error=e.kind, reason=e.message)
        return out

    # ── proposals ──
    def propose(self, op: str, repo: Optional[str] = None, **args: Any) -> bytes:
        """Canonical proposal bytes for one operation. Grants nothing by itself."""
        spec = OPERATIONS.get(op)
        if spec is None:
            raise GatewayError("OPERATION_NOT_ALLOWLISTED", f"{op!r} is not a gateway operation")
        if spec["cap"] == "github.identity.read":
            repo = repo or ""
        elif not isinstance(repo, str) or not _REPO.match(repo):
            raise GatewayError("INVALID_ARGUMENT", f"invalid repository {repo!r}")
        clean = validate_arguments(op, args)
        now = self.clock()
        prop = ActionProposal(
            proposal_id=f"gh-{secrets.token_hex(8)}", request_id=f"req-{secrets.token_hex(8)}",
            proposer_id=self.proposer_id, action_kind=GITHUB_ACTION_KIND, capability=spec["cap"],
            fixture_id=repo.lower() if repo else "", arguments={"op": op, **clean},
            nonce=secrets.token_hex(16), created_at_utc=now.isoformat(),
            expires_at_utc=(now + timedelta(seconds=PROPOSAL_TTL_S)).isoformat())
        return prop.canonical_bytes()

    def decide(self, proposal_bytes: bytes) -> PolicyDecision:
        try:
            login = self.observed_login()
        except GatewayError:
            login = None
        return evaluate_github_proposal(proposal_bytes, self.policy, observed_login=login, now=self.clock())

    @staticmethod
    def _request_from(proposal_bytes: bytes) -> Tuple[str, str, Dict[str, Any], str, Optional[Dict[str, Any]], str]:
        """Rebuild (op, repo, args, endpoint, body, method) from the exact decided bytes."""
        p = strict_parse_json(proposal_bytes)
        args = dict(p["arguments"])
        op = args.pop("op", None)
        if op not in OPERATIONS or OPERATIONS[op]["cap"] != p["capability"]:
            raise GatewayError("BINDING_MISMATCH", "proposal operation does not match its capability")
        clean = validate_arguments(op, args)
        endpoint, body = OPERATIONS[op]["build"](p["fixture_id"], clean)
        return op, p["fixture_id"], clean, endpoint, body, OPERATIONS[op]["method"]

    def _receipt(self, kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        os.makedirs(os.path.dirname(self.receipts_path), mode=0o700, exist_ok=True)
        with self._lock:
            row = Receipts(self.receipts_path).append(kind, payload)
        try:
            os.chmod(self.receipts_path, 0o600)
        except OSError:
            pass
        return row

    def _already_attempted(self, proposal_sha: str, request_id: str) -> bool:
        if not os.path.isfile(self.receipts_path):
            return False
        import json
        with open(self.receipts_path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("kind") == "github.write.attempt":
                    pl = row.get("payload", {})
                    if pl.get("proposal_sha256") == proposal_sha or pl.get("request_id") == request_id:
                        return True
        return False

    # ── reads ──
    def read(self, op: str, repo: Optional[str] = None, **args: Any) -> Dict[str, Any]:
        if op in OPERATIONS and OPERATIONS[op]["cap"] in WRITE_CAPABILITIES:
            raise GatewayError("NOT_A_READ", f"{op} is a write; use propose() and execute()")
        proposal = self.propose(op, repo, **args)
        decision = self.decide(proposal)
        if not decision.allowed:
            self._receipt("github.read.denied", {"op": op, "repository": decision.repository,
                                                 "decision": decision.to_dict()})
            raise GatewayError("POLICY_DENIED", "; ".join(decision.reasons), {"decision": decision.to_dict()})
        _, _, _, endpoint, _, method = self._request_from(proposal)
        r = self.bridge.api(endpoint, method=method)
        if r["status"] != "SUCCESS":
            raise GatewayError(r.get("kind") or "GH_ERROR", r.get("reason", "request failed"))
        return {"status": "SUCCESS", "op": op, "repository": decision.repository, "data": _shape(op, r["data"])}

    def ecosystem(self) -> Dict[str, Any]:
        """Every allowlisted repository with its metadata, or the error that prevented reading it."""
        repos = []
        for name in sorted(self.policy.repositories):
            try:
                repos.append({"repository": name, "ok": True, **self.read("repo.view", name)["data"]})
            except GatewayError as e:
                repos.append({"repository": name, "ok": False, "error": e.kind, "reason": e.message})
        return {"status": "SUCCESS", "repositories": repos}

    def dependencies(self, path: str = "pyproject.toml") -> Dict[str, Any]:
        """Cross-repository dependency graph from each allowlisted repo's pyproject on its default branch."""
        try:
            import tomllib
        except ModuleNotFoundError:  # Python < 3.11
            raise GatewayError("UNSUPPORTED", "dependency inspection needs Python 3.11+ (tomllib)")
        nodes: Dict[str, Dict[str, Any]] = {}
        for repo, caps in sorted(self.policy.repositories.items()):
            if "github.contents.read" not in caps:
                continue
            try:
                f = self.read("contents.get", repo, path=path)["data"]
                proj = tomllib.loads(f["text"]).get("project", {})
            except GatewayError as e:
                nodes[repo] = {"error": e.kind, "reason": e.message}
                continue
            except tomllib.TOMLDecodeError as e:
                nodes[repo] = {"error": "BAD_PYPROJECT", "reason": str(e)}
                continue
            reqs = list(proj.get("dependencies", []))
            for extra, items in (proj.get("optional-dependencies") or {}).items():
                reqs += [f"{i} ; extra={extra}" for i in items]
            nodes[repo] = {"package": proj.get("name"), "version": proj.get("version"), "requires": reqs,
                           "blob_sha": f["sha"]}
        def norm(n: str) -> str:
            return re.sub(r"[-_.]+", "-", n).lower()
        by_pkg = {norm(v["package"]): r for r, v in nodes.items() if v.get("package")}
        edges = []
        for repo, v in nodes.items():
            for req in v.get("requires", []):
                m = re.match(r"\s*([A-Za-z0-9._-]+)", req)
                if m and norm(m.group(1)) in by_pkg and by_pkg[norm(m.group(1))] != repo:
                    url = re.search(r"git\+https://github\.com/([^/@\s]+/[^/@\s.]+)", req)
                    edges.append({"from": repo, "to": by_pkg[norm(m.group(1))], "requirement": req.strip(),
                                  "source": f"git:{url.group(1)}" if url else "index",
                                  "pinned": bool(re.search(r"@[0-9a-f]{40}", req) or "==" in req)})
        return {"status": "SUCCESS", "nodes": nodes, "edges": edges}

    # ── writes ──
    def execute(self, proposal_bytes: bytes, *, dry_run: bool = True, grant_path: Optional[str] = None,
                signature_path: Optional[str] = None, authorize: Optional[Callable[..., Any]] = None,
                authorize_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run one write proposal, or explain exactly why not. Every outcome is receipted."""
        if not isinstance(proposal_bytes, bytes):
            raise GatewayError("INVALID_PROPOSAL", "proposal must be bytes")
        decision = self.decide(proposal_bytes)
        base = {"proposal_sha256": decision.proposal_sha256, "decision": decision.to_dict()}
        if not decision.allowed:
            self._receipt("github.write.denied", base)
            return {"status": "DENIED", "reasons": list(decision.reasons), **base}
        if not decision.is_write:
            raise GatewayError("NOT_A_WRITE", "use read() for read operations")
        op, repo, clean, endpoint, body, method = self._request_from(proposal_bytes)
        request_id = strict_parse_json(proposal_bytes)["request_id"]
        plan = {"op": op, "repository": repo, "method": method, "endpoint": endpoint,
                "body_sha256": canonical_sha256(canonicalize_json(body)) if body is not None else None,
                "request_id": request_id}
        if dry_run:
            self._receipt("github.write.dry_run", {**base, "plan": plan})
            return {"status": "DRY_RUN", "plan": plan, "body": body, **base}
        if self._already_attempted(decision.proposal_sha256, request_id):
            self._receipt("github.write.duplicate_refused", {**base, "plan": plan})
            return {"status": "DUPLICATE_REFUSED", "plan": plan, **base}
        if not grant_path:
            self._receipt("github.write.denied", {**base, "plan": plan, "reason": "NO_GRANT"})
            return {"status": "DENIED", "reasons": ["SYS021_GRANT_REQUIRED"], "plan": plan, **base}
        if authorize is None:
            from osiris_cli.grants import authorize as authorize
        scope = f"github:{decision.capability}:{repo}"
        grant = authorize(grant_path, decision.proposal_sha256, scope, signature_path=signature_path,
                          **(authorize_kwargs or {}))
        if not grant:
            self._receipt("github.write.denied", {**base, "plan": plan, "reason": "GRANT_REFUSED",
                                                  "grant_detail": redact(getattr(grant, "detail", str(grant)))})
            return {"status": "DENIED", "reasons": [f"SYS021_GRANT_REFUSED: {getattr(grant, 'detail', '')}"],
                    "plan": plan, **base}
        self._receipt("github.write.attempt", {**base, "plan": plan, "request_id": request_id, "scope": scope})
        r = self.bridge.api(endpoint, method=method, body=body)
        if r["status"] != "SUCCESS":
            self._receipt("github.write.result", {**base, "status": "FAILED", "kind": r.get("kind"),
                                                  "reason": redact(r.get("reason"))})
            return {"status": "FAILED", "kind": r.get("kind"), "reason": r.get("reason"), "plan": plan, **base}
        result = _shape(op, r["data"])
        self._receipt("github.write.result", {**base, "status": "EXECUTED", "result": result})
        return {"status": "EXECUTED", "result": result, "plan": plan, **base}

    def receipts(self, limit: int = 50) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []
        valid = True
        if os.path.isfile(self.receipts_path):
            import json
            valid = Receipts.verify(self.receipts_path)
            with open(self.receipts_path, encoding="utf-8") as fh:
                rows = [json.loads(l) for l in fh if l.strip()]
        return {"path": self.receipts_path, "chain_valid": valid, "count": len(rows), "rows": rows[-limit:]}


# ── terminal entry point ──────────────────────────────────────────────────────────────────────────
def _kv_args(pairs):
    out = {}
    for p in pairs:
        k, sep, v = p.partition("=")
        if not sep:
            raise GatewayError("INVALID_ARGUMENT", f"expected key=value, got {p!r}")
        if v.isdigit():
            out[k] = int(v)
        elif v in ("true", "false"):
            out[k] = v == "true"
        else:
            out[k] = v
    return out


def main(argv=None) -> int:
    """python -m osiris_cli.github_gateway status | ecosystem | deps | read OP [REPO] [k=v ...]
                                          | propose OP REPO k=v ... --out FILE
                                          | execute FILE [--grant G --signature S] [--live]"""
    import argparse
    import json
    import sys
    ap = argparse.ArgumentParser(prog="python -m osiris_cli.github_gateway", description="Governed GitHub gateway")
    ap.add_argument("--policy", default=DEFAULT_POLICY_PATH)
    ap.add_argument("--receipts", default=DEFAULT_RECEIPTS_PATH)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status"); sub.add_parser("ecosystem"); sub.add_parser("deps"); sub.add_parser("receipts")
    rd = sub.add_parser("read"); rd.add_argument("op"); rd.add_argument("repo", nargs="?"); rd.add_argument("kv", nargs="*")
    pr = sub.add_parser("propose"); pr.add_argument("op"); pr.add_argument("repo"); pr.add_argument("kv", nargs="*")
    pr.add_argument("--out", required=True, help="where to write the canonical proposal bytes")
    ex = sub.add_parser("execute"); ex.add_argument("proposal")
    ex.add_argument("--grant"); ex.add_argument("--signature")
    ex.add_argument("--live", action="store_true", help="send the request (default: dry run)")
    a = ap.parse_args(argv)
    try:
        gw = GitHubGateway(policy_path=a.policy, receipts_path=a.receipts)
        if a.cmd == "status":
            out = gw.status()
        elif a.cmd == "ecosystem":
            out = gw.ecosystem()
        elif a.cmd == "deps":
            out = gw.dependencies()
        elif a.cmd == "receipts":
            out = gw.receipts()
        elif a.cmd == "read":
            out = gw.read(a.op, a.repo, **_kv_args(a.kv))
        elif a.cmd == "propose":
            data = gw.propose(a.op, a.repo, **_kv_args(a.kv))
            with open(a.out, "wb") as fh:
                fh.write(data)
            d = gw.decide(data)
            out = {"written": a.out, "proposal_sha256": d.proposal_sha256, "decision": d.to_dict(),
                   "grant_scope": f"github:{d.capability}:{d.repository}",
                   "next": ("python -m osiris_cli.grants draft --payload-digest <proposal_sha256> --scope <grant_scope>"
                            " > grant.json; sign it at the owner's terminal; then execute --grant ... --live")}
        else:
            with open(a.proposal, "rb") as fh:
                data = fh.read()
            out = gw.execute(data, dry_run=not a.live, grant_path=a.grant, signature_path=a.signature)
    except GatewayError as e:
        print(json.dumps(e.to_dict(), indent=1))
        return 2
    print(json.dumps(out, indent=1, default=str))
    return 0 if out.get("status") not in ("DENIED", "FAILED", "DUPLICATE_REFUSED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
