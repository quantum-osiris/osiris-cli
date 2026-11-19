"""
osiris_cli/session_approval.py
One readable approval for a bounded coding session.

The operator approves a session once: the workspace, the files that may be edited (glob
patterns relative to the workspace) and the exact test/build commands that may run, for a
limited time. The approval is a hybrid Ed25519 + ML-DSA-65 grant signed by the operator's
local key (`event_type = "session.approve"`, schema `osiris.session-scope.v1`).

During the session the service signs a single-use child grant for each concrete action that
falls inside that scope (`approval_mode = "session"`). The child binds the exact payload (file
contents digest or argv/cwd), names its parent session grant, and expires no later than the
session. Anything outside the scope falls back to the normal per-action [y/N] prompt.

What the agent cannot do:
- create, widen or extend a session (the scope comes only from the operator's flags/prompt;
  the model has no tool that reaches this module);
- approve itself (children are signed only after the session grant verifies against the
  trusted key store, never against keys embedded in a grant);
- run excluded actions under a session: shells, network tools, package installs, privilege
  changes, deletes, pushes, deploys, credential files. Those are refused when the session is
  created, and paths such as `.git/`, `.env*` or key files can never be edited under a session.

The grant serialization (OSIRIS-HYBRID-GRANT-V2) is unchanged. Session data lives inside the
signed `operation_schema`, so older grants and their verification are unaffected.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import posixpath
import shlex
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

SESSION_EVENT = "session.approve"
SCOPE_SCHEMA = "osiris.session-scope.v1"
CHILD_SCHEMA = "osiris.session-child.v1"
DEFAULT_SESSION_MINUTES = 60
MAX_SESSION_MINUTES = 8 * 60
CHILD_TTL_S = 120.0

# Paths that a session never covers, whatever glob the operator gives.
EXCLUDED_PATH_PATTERNS = (
    ".git", ".git/*", "*/.git/*", ".env", ".env.*", "*/.env", "*/.env.*",
    "*.pem", "*.key", "*id_rsa*", "*id_ed25519*", ".osiris/*", "*/.osiris/*",
    ".github/workflows/*", "*.p12", "*.pfx", "*credentials*", "*secret*",
)

_SHELLS = {"sh", "bash", "zsh", "dash", "fish", "ksh", "csh", "tcsh", "env", "xargs", "nohup",
           "powershell", "powershell.exe", "pwsh", "cmd", "cmd.exe"}
_PRIVILEGE = {"sudo", "su", "doas", "pkexec", "chmod", "chown", "chgrp", "setfacl"}
_NETWORK = {"curl", "wget", "ssh", "scp", "sftp", "rsync", "nc", "ncat", "netcat", "telnet", "ftp",
            "http", "httpie", "aria2c"}
_INSTALLERS = {"pip", "pip3", "pipx", "uv", "conda", "mamba", "apt", "apt-get", "dpkg", "snap",
               "brew", "gem", "cargo", "go", "npx", "pnpx", "bunx"}
_DESTRUCTIVE = {"rm", "rmdir", "dd", "mkfs", "shred", "truncate", "mv", "find", "kill", "pkill", "killall"}
_DEPLOY = {"vercel", "gh", "supabase", "docker", "podman", "kubectl", "helm", "railway", "render",
           "terraform", "aws", "gcloud", "az", "fly", "flyctl", "netlify", "twine", "ibmcloud"}
_READONLY_GIT = {"status", "diff", "log", "show"}
_JS_RUNNERS = {"npm", "pnpm", "yarn", "bun"}
_JS_SCRIPTS = {"test", "build", "lint", "typecheck"}
_INTERPRETER_EVAL_FLAGS = {"-c", "-e", "--eval", "-E", "--exec"}


class SessionScopeError(ValueError):
    """The requested session scope includes something a session may never cover."""


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha(obj: Any) -> str:
    return hashlib.sha256(_canon(obj).encode("utf-8")).hexdigest()


def excluded_command_reason(argv: Sequence[str]) -> Optional[str]:
    """Why this command may not be pre-approved by a session, or None when it may."""
    if not argv:
        return "empty command"
    prog = os.path.basename(str(argv[0])).lower()
    rest = [str(a) for a in argv[1:]]
    if prog in _SHELLS:
        return f"`{prog}` runs arbitrary commands (shell)"
    if prog in _PRIVILEGE:
        return f"`{prog}` changes privileges or permissions"
    if prog in _NETWORK:
        return f"`{prog}` uses the network"
    if prog in _INSTALLERS:
        return f"`{prog}` installs or fetches packages"
    if prog in _DESTRUCTIVE:
        return f"`{prog}` deletes, moves or kills"
    if prog in _DEPLOY:
        return f"`{prog}` publishes, deploys or reaches a hosted service"
    if prog == "git":
        sub = next((a for a in rest if not a.startswith("-")), "")
        if sub not in _READONLY_GIT or any(a.startswith("-c") for a in rest):
            return "only read-only git (status, diff, log, show) may be pre-approved"
    if prog in _JS_RUNNERS:
        script = rest[1] if rest[:1] == ["run"] and len(rest) > 1 else (rest[0] if rest else "")
        if script not in _JS_SCRIPTS or (rest[:1] != ["run"] and rest[:1] != ["test"]):
            return f"only `{prog} test` or `{prog} run build|test|lint|typecheck` may be pre-approved"
    if prog.startswith("python") or prog in {"node", "deno", "ruby", "perl", "php"}:
        if any(a in _INTERPRETER_EVAL_FLAGS for a in rest):
            return f"`{prog}` with an inline-code flag runs arbitrary code"
        if prog.startswith("python") and "-m" in rest:
            mod = rest[rest.index("-m") + 1] if rest.index("-m") + 1 < len(rest) else ""
            if mod.split(".")[0] in {"pip", "http", "venv", "ensurepip", "twine", "webbrowser", "smtplib"}:
                return f"`python -m {mod}` installs, serves or reaches the network"
    return None


def _path_excluded(rel: str) -> bool:
    return any(fnmatch.fnmatchcase(rel, pat) for pat in EXCLUDED_PATH_PATTERNS)


def _clean_rel(rel: str) -> Optional[str]:
    """A normalized workspace-relative POSIX path, or None if it is absolute or escapes."""
    rel = str(rel).replace("\\", "/")
    if not rel or rel.startswith("/") or (len(rel) > 1 and rel[1] == ":"):
        return None
    norm = posixpath.normpath(rel)
    if norm == ".." or norm.startswith("../"):
        return None
    return norm


@dataclass(frozen=True)
class SessionScope:
    """What the operator approves for one session. Built only from operator input."""

    root: str
    edit_globs: Tuple[str, ...] = ()
    commands: Tuple[Tuple[str, ...], ...] = ()
    minutes: int = DEFAULT_SESSION_MINUTES
    excluded: Tuple[str, ...] = field(default=(
        "shell commands", "network access", "package installs", "credential or key files",
        "git push and other git writes", "deploys and hosted services", "deleting files",
        "paid or QPU actions",
    ))

    @classmethod
    def build(cls, root: str, edit_globs: Sequence[str] = (), commands: Sequence[str] = (),
              minutes: int = DEFAULT_SESSION_MINUTES) -> "SessionScope":
        root_abs = os.path.realpath(os.path.expanduser(root))
        if not os.path.isdir(root_abs):
            raise SessionScopeError(f"workspace {root} is not a directory")
        if not isinstance(minutes, int) or not 1 <= minutes <= MAX_SESSION_MINUTES:
            raise SessionScopeError(f"session length must be 1–{MAX_SESSION_MINUTES} minutes")
        globs: List[str] = []
        for g in edit_globs:
            clean = _clean_rel(g)
            if clean is None:
                raise SessionScopeError(f"edit pattern {g!r} must be relative to the workspace")
            if _path_excluded(clean) or clean in {"*", "**", "**/*", "."}:
                raise SessionScopeError(
                    f"edit pattern {g!r} is too broad or covers protected files; name the files or folders"
                )
            globs.append(clean)
        cmds: List[Tuple[str, ...]] = []
        for c in commands:
            argv = tuple(shlex.split(c)) if isinstance(c, str) else tuple(str(a) for a in c)
            why = excluded_command_reason(argv)
            if why:
                raise SessionScopeError(f"command {shlex.join(argv)!r} cannot be pre-approved: {why}")
            cmds.append(argv)
        if not globs and not cmds:
            raise SessionScopeError("a session needs at least one edit pattern or one command")
        return cls(root=root_abs, edit_globs=tuple(sorted(set(globs))),
                   commands=tuple(sorted(set(cmds))), minutes=minutes)

    def document(self) -> Dict[str, Any]:
        return {
            "schema": SCOPE_SCHEMA,
            "root": self.root,
            "edit_globs": list(self.edit_globs),
            "commands": [list(c) for c in self.commands],
            "minutes": self.minutes,
            "excluded": list(self.excluded),
        }

    def summary(self) -> str:
        lines = [f"Work in {self.root} for up to {self.minutes} minutes."]
        if self.edit_globs:
            lines.append("May edit: " + ", ".join(self.edit_globs))
        if self.commands:
            lines.append("May run: " + "; ".join(shlex.join(c) for c in self.commands))
        lines.append("Will not, without asking you: " + ", ".join(self.excluded) + ".")
        lines.append("Anything else is asked about one action at a time.")
        return "\n".join(lines)


def scope_from_document(doc: Dict[str, Any]) -> SessionScope:
    if not isinstance(doc, dict) or doc.get("schema") != SCOPE_SCHEMA:
        raise SessionScopeError("not a session scope document")
    return SessionScope(
        root=str(doc["root"]),
        edit_globs=tuple(str(g) for g in doc.get("edit_globs", [])),
        commands=tuple(tuple(str(a) for a in c) for c in doc.get("commands", [])),
        minutes=int(doc.get("minutes", DEFAULT_SESSION_MINUTES)),
        excluded=tuple(str(x) for x in doc.get("excluded", [])),
    )


def covers_paths(scope: SessionScope, root: str, rel_paths: Sequence[str]) -> Tuple[bool, str]:
    if os.path.realpath(root) != scope.root:
        return False, "the change is in a different workspace than the session"
    if not rel_paths:
        return False, "no files to cover"
    for rel in rel_paths:
        clean = _clean_rel(rel)
        if clean is None:
            return False, f"{rel} is outside the workspace"
        if _path_excluded(clean):
            return False, f"{rel} is a protected file; sessions never cover it"
        if not any(fnmatch.fnmatchcase(clean, g) for g in scope.edit_globs):
            return False, f"{rel} is not in the session's edit list"
    return True, "covered"


def covers_command(scope: SessionScope, argv: Sequence[str], cwd: str) -> Tuple[bool, str]:
    cwd_real = os.path.realpath(cwd)
    if cwd_real != scope.root and not cwd_real.startswith(scope.root + os.sep):
        return False, "the command runs outside the session's workspace"
    if excluded_command_reason(argv):
        return False, "the command is excluded from sessions"
    if tuple(str(a) for a in argv) not in scope.commands:
        return False, "the command is not in the session's command list"
    return True, "covered"


def request_session_grant(gate: Any, scope: SessionScope, *, confirm: Callable[[str], bool],
                          out: Callable[[str], None] = lambda s: None) -> Dict[str, Any]:
    """Show the session scope in plain words and sign it only after the operator says yes."""
    doc = scope.document()
    return gate.request_approved_grant(
        action=SESSION_EVENT,
        scope=["SESSION", scope.root],
        summary=scope.summary(),
        payload_sha256=_sha(doc),
        operation_schema=doc,
        confirm=confirm,
        out=out,
        prompt="Approve this coding session? [y/N] ",
        ttl_seconds=float(scope.minutes) * 60.0,
    )


def verify_session_grant(gate: Any, session_grant: Dict[str, Any], *,
                         now: Optional[float] = None) -> Tuple[bool, str, Optional[SessionScope]]:
    """Verify a session grant against the trusted key store (not consumed: it covers many actions)."""
    if not isinstance(session_grant, dict):
        return False, "no session approval", None
    doc = session_grant.get("operation_schema")
    if not isinstance(doc, dict) or doc.get("schema") != SCOPE_SCHEMA:
        return False, "session grant has no session scope", None
    try:
        scope = scope_from_document(doc)
    except (SessionScopeError, KeyError, TypeError, ValueError) as exc:
        return False, f"malformed session scope: {exc}", None
    ok, why = gate.verify_grant_integrity(
        session_grant,
        expected_payload_sha256=_sha(doc),
        expected_scope=["SESSION", scope.root],
        expected_event_type=SESSION_EVENT,
        now=now,
        consume=False,
    )
    if not ok:
        return False, f"session approval did not verify: {why}", None
    return True, "session verified", scope


def _child_schema(session_grant: Dict[str, Any], operation_schema: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **operation_schema,
        "session": {
            "schema": CHILD_SCHEMA,
            "parent_grant_id": session_grant["grant_id"],
            "session_scope_sha256": session_grant["payload_sha256"],
        },
    }


def issue_child_grant(gate: Any, session_grant: Dict[str, Any], *, action: str, scope: Sequence[str],
                      payload_sha256: str, operation_schema: Dict[str, Any],
                      covered: Tuple[bool, str], now: Optional[float] = None) -> Dict[str, Any]:
    """Sign a single-use grant for one concrete action inside a verified session.

    Raises GrantVerificationError when the session does not verify or the action is not
    covered. The caller then asks the operator about this action instead; it never runs the
    action unsigned.
    """
    from osiris_cli.pqc_enclave import GrantVerificationError

    ok, why, _scope = verify_session_grant(gate, session_grant, now=now)
    if not ok:
        raise GrantVerificationError(why)
    if not covered[0]:
        raise GrantVerificationError(covered[1])
    t = float(time.time() if now is None else now)
    remaining = float(session_grant["expires_at"]) - t
    if remaining <= 1.0:
        raise GrantVerificationError("the session approval has expired")
    child = gate.generate_hybrid_grant(
        sequence_index=1,
        prev_block_hash="0" * 64,
        event_type=action,
        payload_sha256=payload_sha256,
        operation_schema=_child_schema(session_grant, operation_schema),
        scope=list(scope),
        ttl_seconds=min(CHILD_TTL_S, remaining),
        issued_at=t,
        approval_mode="session",
    )
    ok, why = gate.verify_grant_integrity(child, expected_payload_sha256=payload_sha256,
                                          expected_scope=list(scope), expected_event_type=action,
                                          now=t, consume=False)
    if not ok:
        raise GrantVerificationError(f"child grant failed its self-check: {why}")
    return child


def verify_child_link(gate: Any, child: Dict[str, Any], session_grant: Optional[Dict[str, Any]], *,
                      now: Optional[float] = None) -> Tuple[bool, str]:
    """For a grant whose approval_mode is "session": its parent must be this verified session.

    Grants approved one at a time (approval_mode "interactive"/"operator-flag"/"direct") pass
    unchanged, so legacy grants keep verifying exactly as before.
    """
    if child.get("approval_mode") != "session":
        link = (child.get("operation_schema") or {}).get("session")
        if isinstance(link, dict):
            return False, "grant carries a session link but is not marked as a session grant"
        return True, "not a session grant"
    link = (child.get("operation_schema") or {}).get("session")
    if not isinstance(link, dict) or link.get("schema") != CHILD_SCHEMA:
        return False, "session grant has no parent link"
    if session_grant is None:
        return False, "session grant's parent approval is missing"
    if link.get("parent_grant_id") != session_grant.get("grant_id"):
        return False, "session grant names a different parent approval"
    if link.get("session_scope_sha256") != session_grant.get("payload_sha256"):
        return False, "session grant's scope digest does not match its parent"
    ok, why, _scope = verify_session_grant(gate, session_grant, now=now)
    if not ok:
        return False, why
    if float(child.get("expires_at", 0)) > float(session_grant["expires_at"]) + 1e-6:
        return False, "session grant outlives its parent approval"
    return True, "session link verified"
