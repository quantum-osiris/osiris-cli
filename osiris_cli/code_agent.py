"""/code -- a governed, tool-using coding loop over a local model.

Models propose, code decides. The model may call five tools; none of them grants itself:

* read_file / list_dir / grep_code -- read-only, confined (symlinks resolved) to the
  working directory, the same safe root the REPL's write check uses.
* write_file -- never writes. It hands the content to the REPL's own proposal path
  (osiris_termux_console._propose_write): the diff is shown and nothing changes until
  the user types /apply, which keeps its changed-since-shown check, backup, atomic
  write and genome-ledger record. One pending write at a time.
* run_command -- one program with arguments, never a shell. Each call needs a y/N from
  the user, which mints an ExecutionGrant bound to the digest of (argv, cwd); the grant
  is verified again immediately before the process starts, under the interactive
  horizon from osiris.substrate.

There are deliberately no memory tools: OSIRIS learns only from eligible material the
user explicitly approves, and a model-callable "remember" would bypass that.

Models that write tool calls as JSON text instead of structured tool_calls
(TEXT_TOOL_CALL_MODELS) are parsed by parse_text_tool_calls and governed identically.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

DEFAULT_MODEL = os.environ.get("OSIRIS_CODE_MODEL", "qwen2.5-coder:7b")
PREFERRED_CODE_MODELS = ("qwen2.5-coder:7b", "qwen3:8b", "gemma3:4b")
MAX_STEPS = 8
_MAX_TOOL_RESULT = 6000
_MAX_READ_BYTES = 600_000

#: Model families that write tool calls into the reply text. Prefix match, case-insensitive.
TEXT_TOOL_CALL_MODELS = ("qwen2.5-coder",)

#: Shell syntax that has no meaning without a shell; refused with a clear reason.
_SHELL_ONLY_TOKENS = {"|", "||", "&&", ";", "&", ">", ">>", "<", "<<", "2>", "2>&1", "&>"}
_DANGEROUS = [re.compile(p) for p in (r"^rm$", r"^dd$", r"^mkfs", r"^shutdown$", r"^reboot$")]

SYSTEM = (
    "You are OSIRIS's coding tool loop. Use the tools to look at real files before you "
    "answer; never invent file contents, results or test outcomes. Paths are relative to "
    "the working directory; anything outside it is refused. write_file only PROPOSES a "
    "change -- the user applies it with /apply, so never claim a file was written. "
    "run_command runs one program with arguments (no pipes, redirects or &&) and the user "
    "must approve each one. A DENIED result is final for that call: explain it, do not "
    "retry the same thing. Finish with a short plain answer."
)

TOOLS: List[Dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file in the working directory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer"}, "end_line": {"type": "integer"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "list_dir", "description": "List files under a directory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "pattern": {"type": "string"},
            "recursive": {"type": "boolean"}}, "required": []}}},
    {"type": "function", "function": {
        "name": "grep_code", "description": "Search files for a regex; returns file:line matches.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}, "path": {"type": "string"},
            "file_type": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Propose new full contents for a file. Shown as a diff; applied only by the user's /apply.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_command",
        "description": "Run one program with arguments (no shell). The user approves each call.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}, "cwd": {"type": "string"},
            "timeout": {"type": "integer"}}, "required": ["command"]}}},
]
TOOL_NAMES = tuple(t["function"]["name"] for t in TOOLS)


class Denied(Exception):
    """A tool call refused by policy; its message goes back to the model verbatim."""


# ── Text tool calls ──────────────────────────────────────────────────────────

_FENCE = re.compile(r"```(?:json)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _as_tool_call(obj: Any, known: Iterable[str]) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    if "name" in obj and "arguments" in obj:
        name, arguments = obj.get("name"), obj.get("arguments")
    elif "tool" in obj and "kwargs" in obj:
        name, arguments = obj.get("tool"), obj.get("kwargs")
    else:
        return None
    if not isinstance(name, str) or name not in set(known):
        return None
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return None
    if not isinstance(arguments, dict):
        return None
    return {"function": {"name": name, "arguments": arguments}}


def parse_text_tool_calls(content: str, known: Iterable[str] = TOOL_NAMES) -> List[Dict[str, Any]]:
    """Tool calls a model wrote as JSON text (fenced ```json blocks first, then bare objects),
    in order. Only objects naming a known tool with a JSON-object argument set count.
    Parsing grants nothing: every result is governed like a native call."""
    if not content or "{" not in content:
        return []
    known = list(known)
    decoder = json.JSONDecoder()
    calls: List[Dict[str, Any]] = []

    def scan(text: str) -> None:
        i = text.find("{")
        while i != -1:
            try:
                obj, end = decoder.raw_decode(text, i)
            except ValueError:
                i = text.find("{", i + 1)
                continue
            call = _as_tool_call(obj, known)
            if call is not None:
                calls.append(call)
            i = text.find("{", end)

    for m in _FENCE.finditer(content):
        scan(m.group(1))
    scan(_FENCE.sub(" ", content))
    return calls


def text_tool_calls_enabled(model: Optional[str]) -> bool:
    flag = os.environ.get("OSIRIS_TEXT_TOOL_CALLS", "").strip().lower()
    if flag in ("1", "true", "yes", "on"):
        return True
    if flag in ("0", "false", "no", "off"):
        return False
    name = (model or "").lower()
    return any(name.startswith(p) for p in TEXT_TOOL_CALL_MODELS)


# ── The loop ─────────────────────────────────────────────────────────────────

@dataclass
class CodeResult:
    answer: str
    status: str  # SUCCESS | ERROR | MAX_STEPS | MODEL_UNAVAILABLE | UNAVAILABLE
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    error: str = ""
    run_id: str = ""
    requested_model: Optional[str] = None
    resolved_model: Optional[str] = None
    resolved_digest: Optional[str] = None
    receipt_hash: Optional[str] = None


def _digest(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class CodeAgent:
    """One /code task. Collaborators are injectable so tests need no model or terminal."""

    def __init__(
        self,
        root: Optional[str] = None,
        model: Optional[str] = None,
        chat: Optional[Callable[[List[Dict[str, Any]], List[Dict[str, Any]], str], Dict[str, Any]]] = None,
        confirm: Optional[Callable[[str], bool]] = None,
        propose_write: Optional[Callable[[str, str], bool]] = None,
        has_pending_write: Optional[Callable[[], bool]] = None,
        log: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        out: Callable[[str], None] = print,
        max_steps: int = MAX_STEPS,
        installed_fn: Optional[Callable[[], Dict[str, str]]] = None,
        receipts_path: Optional[str] = None,
    ):
        self.root = os.path.realpath(root or os.getcwd())
        env_model = os.environ.get("OSIRIS_CODE_MODEL", "").strip() or None
        self._pinned_model: Optional[str] = model or env_model
        self.model = self._pinned_model or DEFAULT_MODEL
        self._live_ollama = chat is None
        self._chat = chat or self._ollama_chat
        self._confirm = confirm or _tty_confirm
        self._propose_write = propose_write or _otc_propose_write
        self._has_pending_write = has_pending_write or _otc_has_pending_write
        self._log = log or (lambda event, details: None)
        self._out = out
        self.max_steps = max_steps
        self._installed_fn = installed_fn
        self.receipts_path = receipts_path
        self.decisions: List[Dict[str, Any]] = []
        self._wrote = False
        self._pending_proposal: Optional[Dict[str, Any]] = None
        self.resolved_digest: Optional[str] = None

    # ── model ────────────────────────────────────────────────────────────────
    def _preflight_live_model(self) -> Optional[Tuple[str, str]]:
        """Return (status, error) if live Ollama cannot serve an eligible model; None when ready."""
        from osiris_cli.orchestration import model_router

        fn = self._installed_fn or model_router.ollama_installed
        try:
            installed = fn()
        except Exception as exc:  # noqa: BLE001
            return "UNAVAILABLE", f"DAEMON_UNAVAILABLE: Ollama is not reachable ({type(exc).__name__}: {exc})"

        if self._pinned_model is not None:
            if self._pinned_model not in installed:
                avail = ", ".join(sorted(k for k in installed if k)) or "none"
                return (
                    "MODEL_UNAVAILABLE",
                    f"MODEL_NOT_INSTALLED: {self._pinned_model!r} is not installed in Ollama "
                    f"(installed: {avail}); no other model was substituted",
                )
            self.model = self._pinned_model
            self.resolved_digest = installed.get(self.model) or None
            return None

        for candidate in PREFERRED_CODE_MODELS:
            if candidate in installed:
                self.model = candidate
                self.resolved_digest = installed.get(candidate) or None
                return None
        if installed:
            chosen = sorted(k for k in installed if k)[0]
            self.model = chosen
            self.resolved_digest = installed.get(chosen) or None
            return None
        return "MODEL_UNAVAILABLE", "MODEL_NOT_INSTALLED: Ollama has 0 installed models"

    def _ollama_chat(self, messages, tools, model):
        from osiris_cli.providers.base import Message, ProviderRequest
        from osiris_cli.providers.ollama_compat import OllamaCompatibilityProvider

        from osiris.substrate import execution_timeout_s

        provider = OllamaCompatibilityProvider(model=model)
        request = ProviderRequest(
            messages=[Message(m["role"], m.get("content", ""),
                              {k: m[k] for k in ("tool_calls", "name") if k in m}) for m in messages],
            model=model, tools=tools, max_tokens=1024, temperature=0.0,
            think=False, num_ctx=8192, timeout=execution_timeout_s(),
        )
        resp = provider.chat(request)
        if resp.error:
            return {"error": resp.error}
        return {"content": resp.content, "tool_calls": resp.tool_calls or []}

    def _record_receipt(self, run_id: str, task: str, status: str, error: str = "") -> Optional[str]:
        if not self.receipts_path:
            return None
        from osiris_cli.trio_authz import Receipts

        os.makedirs(os.path.dirname(os.path.abspath(self.receipts_path)), exist_ok=True)
        row = Receipts(self.receipts_path).append("code.agent.run", {
            "run_id": run_id,
            "root": self.root,
            "task_sha256": _sha256_bytes(task.encode("utf-8")),
            "requested_model": self._pinned_model,
            "resolved_model": self.model if status not in ("UNAVAILABLE", "MODEL_UNAVAILABLE") else None,
            "resolved_digest": self.resolved_digest,
            "status": status,
            "error": error[:400] if error else "",
            "decisions": list(self.decisions),
        })
        return row["hash"]

    def _finish(self, run_id: str, task: str, answer: str, status: str, error: str = "") -> CodeResult:
        r_hash = self._record_receipt(run_id, task, status, error)
        resolved = self.model if status not in ("UNAVAILABLE", "MODEL_UNAVAILABLE") else None
        return CodeResult(
            answer=answer,
            status=status,
            decisions=self.decisions,
            error=error,
            run_id=run_id,
            requested_model=self._pinned_model,
            resolved_model=resolved,
            resolved_digest=self.resolved_digest,
            receipt_hash=r_hash,
        )

    def apply_pending_write(self, approve: Callable[[Dict[str, Any]], bool]) -> Dict[str, Any]:
        """Verify unchanged target, check explicit operator approval, atomically apply proposed write,
        and record a receipted decision."""
        if not self._pending_proposal:
            raise Denied("no pending write proposal to apply")
        prop = dict(self._pending_proposal)
        target = prop["abs_path"]
        existed_now = os.path.exists(target)
        current_sha = _sha256_bytes(open(target, "rb").read()) if existed_now else None
        if existed_now != prop["existed_at_proposal"] or current_sha != prop["before_sha256"]:
            self._pending_proposal = None
            raise Denied("target file changed on disk since the proposal was generated; refusing stale apply")
        if not approve(prop):
            rec = {"tool": "apply_write", "decision": "DENIED", "path": prop["rel_path"],
                   "after_sha256": prop["after_sha256"], "reason": "operator declined patch"}
            self.decisions.append(rec)
            self._log("CODE_TOOL", rec)
            return rec
        os.makedirs(os.path.dirname(target), exist_ok=True)
        tmp = f"{target}.osiris-tmp-{uuid.uuid4().hex[:8]}"
        data = prop["content"].encode("utf-8")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
        self._pending_proposal = None
        self._wrote = False
        rec = {
            "tool": "apply_write",
            "decision": "GRANTED",
            "path": prop["rel_path"],
            "before_sha256": prop["before_sha256"],
            "after_sha256": _sha256_bytes(open(target, "rb").read()),
        }
        self.decisions.append(rec)
        self._log("CODE_TOOL", rec)
        if self.receipts_path:
            from osiris_cli.trio_authz import Receipts
            os.makedirs(os.path.dirname(os.path.abspath(self.receipts_path)), exist_ok=True)
            row = Receipts(self.receipts_path).append("code.agent.apply_write", rec)
            rec["receipt_hash"] = row["hash"]
        return rec

    def run(self, task: str) -> CodeResult:
        run_id = f"code-{uuid.uuid4().hex[:12]}"
        if self._live_ollama:
            fail = self._preflight_live_model()
            if fail is not None:
                st, err = fail
                return self._finish(run_id, task, "", st, err)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": f"{SYSTEM}\nWorking directory: {self.root}"},
            {"role": "user", "content": task},
        ]
        for _ in range(self.max_steps):
            resp = self._chat(messages, TOOLS, self.model)
            if "error" in resp:
                err = str(resp["error"])
                st = ("MODEL_UNAVAILABLE" if "MODEL_NOT_INSTALLED" in err
                      else "UNAVAILABLE" if "DAEMON_UNAVAILABLE" in err else "ERROR")
                return self._finish(run_id, task, "", st, err)
            content = resp.get("content", "") or ""
            calls = list(resp.get("tool_calls") or [])
            origin = "native"
            if not calls and text_tool_calls_enabled(self.model):
                calls, origin = parse_text_tool_calls(content), "text"
            if not calls:
                return self._finish(run_id, task, content or "(no answer)", "SUCCESS")
            messages.append({"role": "assistant", "content": content, "tool_calls": calls})
            for call in calls:
                fn = call.get("function", {}) or {}
                name = fn.get("name", "")
                raw = fn.get("arguments", {}) or {}
                try:
                    args = json.loads(raw) if isinstance(raw, str) else dict(raw)
                except (ValueError, TypeError):
                    args = None
                if args is None:
                    result = f"ERROR: arguments for {name} are not a JSON object"
                    bad = {"tool": name, "origin": origin, "decision": "ERROR",
                           "reason": "arguments are not a JSON object"}
                    self.decisions.append(bad)
                    self._log("CODE_TOOL", bad)
                else:
                    result = self.call(name, args, origin)
                if len(result) > _MAX_TOOL_RESULT:
                    result = result[:3000] + f"\n...[{len(result) - 3000} chars omitted]...\n" + result[-1500:]
                messages.append({"role": "tool", "content": result, "name": name})
        return self._finish(run_id, task, "", "MAX_STEPS", f"no answer within {self.max_steps} steps")

    # ── one tool call ────────────────────────────────────────────────────────
    def call(self, name: str, args: Dict[str, Any], origin: str = "native") -> str:
        shown = ", ".join(f"{k}={str(v)[:50]!r}" for k, v in args.items() if k != "content")
        self._out(f"\033[90m  ▸ {name}({shown}){' [from text]' if origin != 'native' else ''}\033[0m")
        decision = {"tool": name, "origin": origin, "args": {k: str(v)[:200] for k, v in args.items()
                                                              if k != "content"}}
        try:
            handler = {
                "read_file": self._read_file, "list_dir": self._list_dir, "grep_code": self._grep,
                "write_file": self._write_file, "run_command": self._run_command,
            }.get(name)
            if handler is None:
                raise Denied(f"unknown tool {name!r}")
            result, extra = handler(**args)
            decision.update(decision="GRANTED", **extra)
        except Denied as exc:
            result = f"DENIED: {exc}"
            decision.update(decision="DENIED", reason=str(exc))
        except TypeError as exc:
            result = f"ERROR: bad arguments for {name}: {exc}"
            decision.update(decision="ERROR", reason=str(exc))
        self.decisions.append(decision)
        self._log("CODE_TOOL", decision)
        if decision["decision"] != "GRANTED":
            self._out(f"\033[33m    ✗ {result}\033[0m")
        return result

    def _confine(self, raw: Optional[str]) -> str:
        """realpath under the working directory, or Denied (symlinks cannot escape)."""
        candidate = os.path.realpath(os.path.join(self.root, os.path.expanduser(str(raw or "."))))
        if not (candidate == self.root or candidate.startswith(self.root.rstrip(os.sep) + os.sep)):
            raise Denied(f"'{raw}' resolves outside the working directory ({self.root})")
        return candidate

    def _read_file(self, path, start_line=None, end_line=None):
        p = self._confine(path)
        if not os.path.isfile(p):
            raise Denied(f"'{path}' is not a file")
        if os.path.getsize(p) > _MAX_READ_BYTES:
            raise Denied(f"'{path}' is over {_MAX_READ_BYTES:,} bytes; use grep_code or a line range")
        with open(p, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if start_line or end_line:
            lines = lines[int(start_line or 1) - 1:int(end_line or len(lines))]
        text = "".join(lines)
        if len(text) > 40_000:
            text = text[:20_000] + "\n...[truncated: ask for a line range]...\n"
        return text, {"path": p}

    def _list_dir(self, path=".", pattern="*", recursive=False):
        p = self._confine(path)
        if not os.path.isdir(p):
            raise Denied(f"'{path}' is not a directory")
        base = Path(p)
        entries = sorted((base.rglob if recursive else base.glob)(pattern or "*"))
        rows = []
        for e in entries:
            if ".git" in e.parts or "__pycache__" in e.parts:
                continue
            rows.append(("DIR  " if e.is_dir() else f"{e.stat().st_size:>9,}B ") + str(e.relative_to(base)))
            if len(rows) >= 300:
                break
        return "\n".join(rows) or "(empty)", {"path": p}

    def _grep(self, pattern, path=".", file_type=""):
        p = self._confine(path)
        argv = ["grep", "-rnE"]
        if file_type:
            argv.append(f"--include=*.{str(file_type).lstrip('.')}")
        argv += ["--", str(pattern), p]
        try:
            r = subprocess.run(argv, shell=False, capture_output=True, text=True, timeout=30,
                               stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise Denied(f"grep unavailable or timed out: {exc}")
        lines = [ln.replace(self.root + os.sep, "", 1) for ln in r.stdout.splitlines()[:120]]
        return "\n".join(lines) or "(no matches)", {"path": p}

    def _write_file(self, path, content):
        abs_path = self._confine(path)  # refuse early with a reason the model can read
        if self._wrote or self._has_pending_write():
            raise Denied("a write is already pending; the user must /apply or discard it first")
        rel_path = os.path.relpath(abs_path, self.root)
        new_text = str(content)
        existed = os.path.exists(abs_path)
        old_bytes = open(abs_path, "rb").read() if existed else b""
        old_text = old_bytes.decode("utf-8", errors="replace") if existed else ""
        before_sha = _sha256_bytes(old_bytes) if existed else None
        after_sha = _sha256_bytes(new_text.encode("utf-8"))
        diff = "".join(difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=f"a/{rel_path}" if existed else "/dev/null",
            tofile=f"b/{rel_path}",
        ))
        if not self._propose_write(str(path), new_text):
            raise Denied("the REPL's write check refused this path")
        self._wrote = True
        self._pending_proposal = {
            "abs_path": abs_path,
            "rel_path": rel_path,
            "content": new_text,
            "existed_at_proposal": existed,
            "before_sha256": before_sha,
            "after_sha256": after_sha,
            "diff": diff,
        }
        return ("PROPOSED: the diff is on the user's screen. It is applied only if they type /apply; "
                "do not say the file was written."), {
            "path": str(path), "chars": len(new_text), "before_sha256": before_sha, "after_sha256": after_sha}

    def _run_command(self, command, cwd=".", timeout=None):
        from osiris.substrate import interactive_timeout_s
        from osiris_cli.orchestration.schemas import ExecutionGrant

        try:
            argv = shlex.split(str(command))
        except ValueError as exc:
            raise Denied(f"cannot parse command: {exc}")
        if not argv:
            raise Denied("empty command")
        shell_bits = [t for t in argv if t in _SHELL_ONLY_TOKENS or t.startswith("$(")]
        if shell_bits:
            raise Denied(f"shell syntax {shell_bits} is not supported: one program, no shell")
        if any(rx.search(os.path.basename(argv[0])) for rx in _DANGEROUS):
            raise Denied(f"'{argv[0]}' is not allowed from the tool loop")
        work = self._confine(cwd)

        payload = {"argv": argv, "cwd": work}
        digest = _digest(payload)
        if not self._confirm(f"Run `{shlex.join(argv)}` in {work}? [y/N] "):
            raise Denied("the user did not approve this command")
        now = time.time()
        grant = ExecutionGrant(grant_id=f"grant-{uuid.uuid4().hex[:12]}", payload_digest=digest,
                               authorized_by="operator", authorized_at=now, expires_at=now + 60,
                               scope=["SUBPROCESS_EXEC", work])
        horizon = interactive_timeout_s()
        try:
            limit = min(float(timeout), horizon) if timeout else horizon
        except (TypeError, ValueError):
            limit = horizon
        # Verified against the exact argv/cwd about to run, not what was asked about.
        if not grant.is_valid(_digest({"argv": argv, "cwd": work})):
            raise Denied("execution grant no longer matches the command")
        try:
            proc = subprocess.Popen(
                argv, shell=False, cwd=work, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                start_new_session=True,
            )
        except FileNotFoundError:
            raise Denied(f"program not found: {argv[0]}")
        try:
            stdout, stderr = proc.communicate(timeout=limit)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                proc.kill()
            proc.communicate()
            return f"TIMEOUT after {limit:.0f}s", {"grant_id": grant.grant_id, "digest": digest, "timed_out": True}
        out = stdout + (f"\n[stderr]\n{stderr}" if stderr.strip() else "")
        return (out or f"(exit code {proc.returncode}, no output)"), {
            "grant_id": grant.grant_id,
            "digest": digest,
            "exit_code": proc.returncode,
            "stdout_sha256": _sha256_bytes(stdout.encode("utf-8")),
            "stderr_sha256": _sha256_bytes(stderr.encode("utf-8")),
        }


# ── REPL bindings ────────────────────────────────────────────────────────────

def _otc():
    import osiris_termux_console as otc
    return otc


def _otc_propose_write(path: str, content: str) -> bool:
    return bool(_otc()._propose_write(path, content, allow_self_modify=False))


def _otc_has_pending_write() -> bool:
    return bool((_otc()._pending_write or {}).get("path"))


def _tty_confirm(prompt: str) -> bool:
    try:
        return _otc()._ask(prompt).strip().lower() in ("y", "yes")
    except Exception:  # noqa: BLE001 - no terminal, no approval
        return False


def run_code_command(state: Any, rest: str) -> CodeResult:
    """`/code [--model NAME] <task>` from the REPL."""
    parts = rest.split()
    model = None
    if len(parts) >= 2 and parts[0] == "--model":
        model, rest = parts[1], " ".join(parts[2:])
    task = rest.strip()
    if not task:
        print("Usage: /code [--model NAME] <task>   -- tools: read/list/grep, proposed writes "
              "(/apply), approved commands")
        return CodeResult("", "ERROR", [], "no task")
    log = getattr(state, "log_event", None)
    agent = CodeAgent(model=model, log=(lambda e, d: log(e, d)) if log else None)
    print(f"\n\033[1;36m[/code]\033[0m {agent.model} · working directory {agent.root}")
    result = agent.run(task)
    granted = sum(1 for d in result.decisions if d["decision"] == "GRANTED")
    denied = len(result.decisions) - granted
    if result.status == "SUCCESS":
        print(f"\n{result.answer}\n")
    else:
        print(f"\n\033[33m[/code {result.status}]\033[0m {result.error}\n")
    print(f"\033[2m  tools: {granted} granted · {denied} denied/failed"
          f"{' · a write is pending: /apply to confirm' if agent._wrote else ''}\033[0m\n")
    return result
