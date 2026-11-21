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
import itertools
import json
import os
import platform
import re
import shlex
import shutil
import stat
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

DEFAULT_MODEL = os.environ.get("OSIRIS_CODE_MODEL", "qwen2.5-coder:7b")
PREFERRED_CODE_MODELS = ("qwen2.5-coder:7b", "qwen3.5:9b", "gemma4:e4b", "qwen3:8b", "qwen3.5:4b", "gemma3:4b")
MAX_STEPS = 8
_MAX_TOOL_RESULT = 6000
_MAX_READ_BYTES = 600_000

#: Model families that write tool calls into the reply text. Prefix match, case-insensitive.
TEXT_TOOL_CALL_MODELS = ("qwen2.5-coder",)

#: Shell syntax that has no meaning without a shell; refused with a clear reason.
_SHELL_ONLY_TOKENS = {"|", "||", "&&", ";", "&", ">", ">>", "<", "<<", "2>", "2>&1", "&>"}
_DANGEROUS = [re.compile(p) for p in (r"^rm$", r"^dd$", r"^mkfs", r"^shutdown$", r"^reboot$")]
_BARE_INTERACTIVE_REPLS = frozenset({
    "powershell", "powershell.exe", "pwsh", "pwsh.exe", "cmd", "cmd.exe",
    "python", "python3", "bash", "sh", "zsh", "node", "irb",
})


def _is_wsl() -> bool:
    try:
        return "microsoft" in platform.release().lower()
    except Exception:  # noqa: BLE001
        return False

# Commands that change credentials or account state, or publish outside this machine. The
# tool loop never runs them, whatever the model was told: a task often contains pasted text
# (reports, logs, READMEs) whose instructions are not the user's, and a y/N prompt reading
# "Run `gh auth logout`?" does not say that it signs this machine out of GitHub.
_WRAPPERS = {"sudo", "doas", "env", "command", "nohup", "nice", "time", "exec"}
_ACCOUNT_RULES = {
    "gh": {"auth", "secret", "ssh-key", "gpg-key", "variable", "release", "workflow",
           ("pr", "merge"), ("pr", "create"), ("repo", "create"), ("repo", "delete"),
           ("repo", "edit"), ("repo", "rename"), ("repo", "archive"), ("api", None)},
    "git": {"push", "credential", ("config", "--global"), ("config", "--system"),
            ("remote", "set-url"), ("remote", "add")},
    "vercel": None, "supabase": None, "gcloud": None, "aws": None, "az": None, "netlify": None,
    "ssh-keygen": None, "ssh-add": None, "gpg": None, "passwd": None,
    "npm": {"publish", "login", "adduser", "token"}, "pnpm": {"publish", "login"},
    "yarn": {"publish", "npm"}, "twine": {"upload"}, "docker": {"push", "login"},
    "ollama": {"push", "rm"},
}


def _account_action(argv: List[str]) -> Optional[str]:
    """Why `argv` changes credentials/account state or publishes, or None."""
    args = list(argv)
    while args and os.path.basename(args[0]) in _WRAPPERS:
        args = args[1:]
        while args and (args[0].startswith("-") or "=" in args[0]):
            args = args[1:]
    if not args:
        return None
    prog = os.path.basename(args[0])
    if prog not in _ACCOUNT_RULES:
        return None
    rules = _ACCOUNT_RULES[prog]
    if rules is None:
        return f"'{prog}' acts on a cloud or credential account"
    rest = [a for a in args[1:]]
    sub = next((a for a in rest if not a.startswith("-")), "")
    for rule in rules:
        if isinstance(rule, str) and rule == sub:
            return f"'{prog} {sub}' changes credentials, account state or published state"
        if isinstance(rule, tuple) and rule[0] == sub:
            if rule[1] is None or rule[1] in rest:
                return f"'{prog} {' '.join(x for x in rule if x)}' changes credentials, account state or published state"
    return None

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


#: Environment names that look like credentials. They are removed from every child
#: process the loop starts, so an approved `env`/`printenv` cannot hand a key to the model.
_SECRET_ENV_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSW|CREDENTIAL|PRIVATE|AUTH|COOKIE|SESSION|_PAT$", re.I)


def child_env(base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """A copy of the environment without credential-shaped variables."""
    src = os.environ if base is None else base
    return {k: v for k, v in src.items() if not _SECRET_ENV_NAME.search(k)}


#: A final answer that says a file was changed. Checked against apply_write receipts.
_EDIT_CLAIM = re.compile(
    r"\b(?:I(?:'ve|\s+have)?|(?:has|have)\s+been|was|were|is\s+now|successfully)\s+"
    r"(?:\w+\s+){0,2}?(?:edited|updated|modified|written|wrote|created|fixed|applied|changed|"
    r"patched|saved|replaced|rewrote|rewritten)\b",
    re.IGNORECASE,
)


def edit_claims(answer: str) -> List[str]:
    """Sentences in `answer` that assert a file change (lossless, at most 10, 200 chars each)."""
    out: List[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer or ""):
        if _EDIT_CLAIM.search(sentence) and not re.search(r"\b(?:not|never|no)\b", sentence, re.I):
            out.append(sentence.strip()[:200])
            if len(out) >= 10:
                break
    return out


def vet_argv(command: str) -> List[str]:
    """Parse one command (no shell) and refuse shell syntax, destructive and account programs."""
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
    account = _account_action(argv)
    if account:
        raise Denied(f"{account}; the tool loop never runs these. If you mean it, run it yourself in a terminal.")
    return argv


_MAX_VERIFY_OUTPUT = 4000


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
    #: SUCCESS | ERROR | MAX_STEPS | CIRCUIT_BREAKER | MODEL_UNAVAILABLE | UNAVAILABLE, and for a run
    #: given a task context (osiris_cli.task_context): STOPPED_BUDGET | BLOCKED.
    status: str
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    error: str = ""
    run_id: str = ""
    requested_model: Optional[str] = None
    resolved_model: Optional[str] = None
    resolved_digest: Optional[str] = None
    receipt_hash: Optional[str] = None
    #: Sentences in the answer that claim a file change no apply_write receipt backs.
    unbacked_claims: List[str] = field(default_factory=list)
    #: For a run given a task context: why it was stopped (max_steps, max_tool_calls, max_tokens,
    #: max_wall_s, repeated_failure, not_permitted) or None, what it spent, and the context's digest.
    stop_reason: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)
    context_sha256: Optional[str] = None


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
        max_writes: int = 1,
        installed_fn: Optional[Callable[[], Dict[str, str]]] = None,
        receipts_path: Optional[str] = None,
        write_nudges: int = 0,
        enclave: Optional[Any] = None,
        session_grant: Optional[Dict[str, Any]] = None,
        context: Optional[Any] = None,
        task_ledger: Optional[Any] = None,
        clock: Callable[[], float] = time.monotonic,
        exec_journal: Optional[Any] = None,
        exec_artifacts_dir: Optional[Any] = None,
        include_workspace_map: Optional[bool] = None,
    ):
        self.root = os.path.realpath(root or os.getcwd())
        # The task's run context (osiris_cli.task_context.TaskRunContext): which tools this run may
        # use and what it may spend. When given, its BudgetMeter governs the loop instead of
        # max_steps, and the run is opened and closed in task_ledger. The model has no tool that
        # can read or change it. Without one, the loop behaves exactly as it did before contexts.
        self._context = context
        self._task_ledger = task_ledger
        self._clock = clock
        if isinstance(exec_journal, (str, os.PathLike)):
            from osiris_cli.exec_kernel import ExecJournal
            exec_journal = ExecJournal(exec_journal)
        self._exec_journal = exec_journal
        self._exec_artifacts_dir = exec_artifacts_dir
        self.meter: Optional[Any] = None
        self._stop: Optional[Any] = None
        self._ledger_opened = False
        self._enclave = enclave
        # A signed session approval from the operator (osiris_cli.session_approval). The model has
        # no tool that can create or change it; it only lets listed commands run without a prompt.
        self._session_grant = session_grant
        env_model = os.environ.get("OSIRIS_CODE_MODEL", "").strip() or None
        self._pinned_model: Optional[str] = model or env_model
        self.model = self._pinned_model or DEFAULT_MODEL
        self._live_ollama = chat is None
        self._include_workspace_map = self._live_ollama if include_workspace_map is None else bool(include_workspace_map)
        self._chat = chat or self._ollama_chat
        self._confirm = confirm or _tty_confirm
        self._propose_write = propose_write or _otc_propose_write
        self._has_pending_write = has_pending_write or _otc_has_pending_write
        self._log = log or (lambda event, details: None)
        self._out = out
        self.max_steps = max_steps
        self.max_writes = max(1, int(max_writes))
        self._installed_fn = installed_fn
        self.receipts_path = receipts_path
        #: Times the runner may tell a model that answered without write_file that nothing
        #: changed (bounded, recorded as a RETRY decision). 0 for the REPL.
        self.write_nudges = max(0, int(write_nudges))
        self.decisions: List[Dict[str, Any]] = []
        self._wrote = False
        self._pending_proposal: Optional[Dict[str, Any]] = None
        self._pending_proposals: Dict[str, Dict[str, Any]] = {}
        self.resolved_digest: Optional[str] = None
        self.run_id: str = ""
        self.last_status: str = ""
        self.last_answer: str = ""

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

        try:
            from osiris_cli import evidence_router

            ranked = evidence_router.rank_models("code", installed=sorted(k for k in installed if k))
            for cand_score in ranked:
                if cand_score.model in installed and not cand_score.demoted:
                    self.model = cand_score.model
                    self.resolved_digest = installed.get(self.model) or None
                    return None
        except Exception:
            pass

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
        return {"content": resp.content, "tool_calls": resp.tool_calls or [],
                "tokens": int(resp.eval_count or 0) + int(resp.prompt_eval_count or 0)}

    def _record_receipt(self, run_id: str, task: str, status: str, error: str = "",
                        unbacked: Optional[List[str]] = None) -> Optional[str]:
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
            "unbacked_claims": list(unbacked or []),
            **self._task_run_fields(),
        })
        return row["hash"]

    def _task_run_fields(self) -> Dict[str, Any]:
        """What a governed run adds to its receipt; nothing for a run without a context."""
        if self._context is None:
            return {}
        return {"task_run": {"run_id": self._context.run_id, "context_sha256": self._context.context_sha256,
                             "risk_tier": self._context.risk_tier, "cost_class": self._context.cost_class,
                             "stop_reason": self._stop.reason if self._stop else None,
                             "usage": self.meter.usage() if self.meter is not None else {}}}

    def _applied_writes(self) -> List[Dict[str, Any]]:
        return [d for d in self.decisions if d.get("tool") == "apply_write" and d.get("decision") == "GRANTED"]

    def _stopped(self, run_id: str, task: str, stop: Any) -> CodeResult:
        """End a governed run at a budget cap or a permission block."""
        self._stop = stop
        self._out(f"\033[33m  ■ {stop}\033[0m")
        return self._finish(run_id, task, "", stop.status, str(stop))

    def _begin_governed(self, run_id: str) -> Optional[Any]:
        """Before any model call: the context must be this run's, must permit what the agent was
        built to do, and must be recorded. Returns a Stop (BLOCKED) or None."""
        from osiris_cli import task_context as tc
        ctx = self._context
        if ctx.run_id != run_id:
            return tc.Stop(tc.BLOCKED, "context_invalid",
                           f"run {run_id}: the context given was issued to run {ctx.run_id}")
        try:
            tc.TaskRunContext.from_dict(ctx.to_dict())   # still hashes to its own digest
        except tc.TaskContextError as exc:
            return tc.Stop(tc.BLOCKED, "context_invalid", f"run {run_id}: {exc}")
        stop = tc.require(ctx, max_writes=self.max_writes)
        if stop is not None:
            return stop
        self.meter = ctx.meter(self._clock)
        if self._task_ledger is not None:
            try:
                self._task_ledger.open(ctx)
                self._ledger_opened = True
            except tc.TaskLedgerError as exc:
                return tc.Stop(tc.BLOCKED, "ledger_unavailable", f"run {run_id}: {exc}; nothing was run")
        return None

    def _finish(self, run_id: str, task: str, answer: str, status: str, error: str = "") -> CodeResult:
        if self._ledger_opened:
            from osiris_cli.task_context import BLOCKED, TaskLedgerError
            self._ledger_opened = False
            try:
                self._task_ledger.close(self._context, status, stop=self._stop,
                                        usage=self.meter.usage() if self.meter is not None else {},
                                        detail=error)
            except TaskLedgerError as exc:
                # The run happened but its close could not be recorded: nothing it proposed advances.
                status, answer, error = BLOCKED, "", f"{exc}; run left open in the task ledger"
        unbacked = edit_claims(answer) if status == "SUCCESS" and not self._applied_writes() else []
        self.run_id, self.last_status, self.last_answer = run_id, status, answer
        r_hash = self._record_receipt(run_id, task, status, error, unbacked)
        resolved = self.model if status not in ("UNAVAILABLE", "MODEL_UNAVAILABLE") else None
        return CodeResult(
            unbacked_claims=unbacked,
            answer=answer,
            status=status,
            decisions=self.decisions,
            error=error,
            run_id=run_id,
            requested_model=self._pinned_model,
            resolved_model=resolved,
            resolved_digest=self.resolved_digest,
            receipt_hash=r_hash,
            stop_reason=self._stop.reason if self._stop else None,
            usage=self.meter.usage() if self.meter is not None else {},
            context_sha256=self._context.context_sha256 if self._context is not None else None,
        )

    def _check_confined_regular_target(self, rel_path: str) -> str:
        if not isinstance(rel_path, str) or not rel_path or "\x00" in rel_path:
            raise Denied(f"invalid target path {rel_path!r}")
        norm_slash = rel_path.replace("\\", "/")
        if norm_slash.startswith("/") or (len(norm_slash) > 1 and norm_slash[1] == ":"):
            raise Denied(f"target path {rel_path!r} must be workspace-relative")
        parts = [seg for seg in norm_slash.split("/") if seg]
        if not parts or any(seg in (".", "..") for seg in parts):
            raise Denied(f"target path {rel_path!r} escapes workspace root")
        cur = self.root
        for idx, part in enumerate(parts):
            cur = os.path.join(cur, part)
            is_last = idx == len(parts) - 1
            try:
                st = os.lstat(cur)
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(st.st_mode):
                raise Denied(f"target path '{rel_path}' traverses or points to a symlink; refusing write")
            if is_last and not stat.S_ISREG(st.st_mode):
                raise Denied(f"target path '{rel_path}' is not a regular file; refusing write")
            if not is_last and not stat.S_ISDIR(st.st_mode):
                raise Denied(f"parent of '{rel_path}' is not a directory; refusing write")
        candidate = os.path.normpath(os.path.join(self.root, *parts))
        if not candidate.startswith(self.root.rstrip(os.sep) + os.sep):
            raise Denied(f"target path '{rel_path}' resolves outside working directory ({self.root})")
        return candidate

    def _validate_proposal_target_and_content(self, prop: Dict[str, Any]) -> Tuple[str, bytes]:
        target = self._check_confined_regular_target(prop["rel_path"])
        data = prop["content"].encode("utf-8")
        if _sha256_bytes(data) != prop["after_sha256"]:
            raise Denied(f"proposal content SHA-256 mismatch for '{prop['rel_path']}'; refusing apply")
        try:
            st = os.lstat(target)
            existed_now = True
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
                raise Denied(f"target '{prop['rel_path']}' is a symlink or non-regular file; refusing apply")
            current_sha = _sha256_bytes(open(target, "rb").read())
        except FileNotFoundError:
            existed_now = False
            current_sha = None
        if existed_now != prop["existed_at_proposal"] or current_sha != prop["before_sha256"]:
            raise Denied(f"target file '{prop['rel_path']}' changed on disk since the proposal was generated; refusing stale apply")
        return target, data

    def apply_pending_write(self, approve: Callable[[Dict[str, Any]], bool]) -> Dict[str, Any]:
        """Verify unchanged confined target before and after explicit operator approval, apply via
        per-file atomic replacement, and record a receipted decision."""
        if not self._pending_proposal:
            raise Denied("no pending write proposal to apply")
        prop = dict(self._pending_proposal)
        try:
            self._validate_proposal_target_and_content(prop)
        except Denied:
            self._pending_proposal = None
            raise
        enclave = self._get_enclave()
        write_schema = {
            "action": "code.apply_writes",
            "files": [{"rel_path": prop["rel_path"], "before_sha256": prop["before_sha256"], "after_sha256": prop["after_sha256"]}],
            "root": self.root,
        }
        write_digest = _digest(write_schema)
        write_scope = ["code.apply_writes", self.root, prop["rel_path"]]
        from osiris_cli.pqc_enclave import ApprovalDeniedError
        try:
            pqc_grant = enclave.request_approved_grant(
                action="code.apply_writes",
                scope=write_scope,
                summary=f"Apply {prop['rel_path']} ({(prop['before_sha256'] or 'new')[:10]}->{prop['after_sha256'][:10]})",
                payload_sha256=write_digest,
                operation_schema=write_schema,
                confirm=lambda _prompt: bool(approve(prop)),
                out=self._out,
            )
        except ApprovalDeniedError:
            rec = {"tool": "apply_write", "decision": "DENIED", "path": prop["rel_path"],
                   "after_sha256": prop["after_sha256"], "reason": "operator declined patch"}
            self.decisions.append(rec)
            self._log("CODE_TOOL", rec)
            return rec
        # Re-verify confinement, non-symlink status, content hash, and unchanged disk state AFTER approval.
        try:
            target, data = self._validate_proposal_target_and_content(prop)
        except Denied:
            self._pending_proposal = None
            raise
        verifier = enclave.as_verifier()
        ok_g, why_g = verifier.verify_grant_integrity(
            pqc_grant,
            expected_payload_sha256=write_digest,
            expected_scope=write_scope,
            expected_event_type="code.apply_writes",
            consume=True,
        )
        if not ok_g:
            raise Denied(f"grant verification failed: {why_g}")
        atomic_write(target, data)
        self._pending_proposal = None
        self._pending_proposals.pop(prop["rel_path"], None)
        self._wrote = bool(self._pending_proposals)
        rec = {
            "tool": "apply_write",
            "decision": "GRANTED",
            "path": prop["rel_path"],
            "before_sha256": prop["before_sha256"],
            "after_sha256": _sha256_bytes(open(target, "rb").read()),
            "grant_id": pqc_grant["grant_id"],
            "key_fingerprint": pqc_grant["key_fingerprint"],
        }
        self.decisions.append(rec)
        self._log("CODE_TOOL", rec)
        if self.receipts_path:
            from osiris_cli.trio_authz import Receipts
            os.makedirs(os.path.dirname(os.path.abspath(self.receipts_path)), exist_ok=True)
            row = Receipts(self.receipts_path).append("code.agent.apply_write", rec)
            rec["receipt_hash"] = row["hash"]
        return rec

    def apply_all_pending_writes(self, approve: Callable[[List[Dict[str, Any]]], bool]) -> List[Dict[str, Any]]:
        """Validate and sequentially apply staged multi-file proposals via per-file atomic replacement
        (not an atomic multi-file transaction) after explicit operator approval."""
        if not self._pending_proposals:
            raise Denied("no pending write proposals to apply")
        proposals = [dict(p) for p in self._pending_proposals.values()]
        for prop in proposals:
            try:
                self._validate_proposal_target_and_content(prop)
            except Denied:
                self._pending_proposals.clear()
                self._pending_proposal = None
                self._wrote = False
                raise
        enclave = self._get_enclave()
        ordered = sorted(proposals, key=lambda p: str(p["rel_path"]))
        batch_schema = {
            "action": "code.apply_writes",
            "files": [{"rel_path": p["rel_path"], "before_sha256": p["before_sha256"], "after_sha256": p["after_sha256"]} for p in ordered],
            "root": self.root,
        }
        batch_digest = _digest(batch_schema)
        batch_scope = ["code.apply_writes", self.root, *[p["rel_path"] for p in ordered]]
        from osiris_cli.pqc_enclave import ApprovalDeniedError
        try:
            pqc_grant = enclave.request_approved_grant(
                action="code.apply_writes",
                scope=batch_scope,
                summary=f"Apply {len(ordered)} staged file(s): " + ", ".join(p["rel_path"] for p in ordered),
                payload_sha256=batch_digest,
                operation_schema=batch_schema,
                confirm=lambda _prompt: bool(approve(proposals)),
                out=self._out,
            )
        except ApprovalDeniedError:
            records = []
            for prop in proposals:
                rec = {"tool": "apply_write", "decision": "DENIED", "path": prop["rel_path"],
                       "after_sha256": prop["after_sha256"], "reason": "operator declined patch batch"}
                self.decisions.append(rec)
                self._log("CODE_TOOL", rec)
                records.append(rec)
            return records

        # Re-verify every target AFTER approval before consuming the grant or writing any file.
        validated_batch: List[Tuple[Dict[str, Any], str, bytes]] = []
        for prop in ordered:
            try:
                target, data = self._validate_proposal_target_and_content(prop)
            except Denied:
                self._pending_proposals.clear()
                self._pending_proposal = None
                self._wrote = False
                raise
            validated_batch.append((prop, target, data))

        verifier = enclave.as_verifier()
        ok_g, why_g = verifier.verify_grant_integrity(
            pqc_grant,
            expected_payload_sha256=batch_digest,
            expected_scope=batch_scope,
            expected_event_type="code.apply_writes",
            consume=True,
        )
        if not ok_g:
            raise Denied(f"grant verification failed: {why_g}")

        records = []
        for prop, _target, _data in validated_batch:
            target, data = self._validate_proposal_target_and_content(prop)
            atomic_write(target, data)
            rec = {
                "tool": "apply_write",
                "decision": "GRANTED",
                "path": prop["rel_path"],
                "before_sha256": prop["before_sha256"],
                "after_sha256": _sha256_bytes(open(target, "rb").read()),
                "grant_id": pqc_grant["grant_id"],
                "key_fingerprint": pqc_grant["key_fingerprint"],
            }
            self.decisions.append(rec)
            self._log("CODE_TOOL", rec)
            if self.receipts_path:
                from osiris_cli.trio_authz import Receipts
                os.makedirs(os.path.dirname(os.path.abspath(self.receipts_path)), exist_ok=True)
                row = Receipts(self.receipts_path).append("code.agent.apply_write", rec)
                rec["receipt_hash"] = row["hash"]
            records.append(rec)
        self._pending_proposals.clear()
        self._pending_proposal = None
        self._wrote = False
        return records

    def run(self, task: str, run_id: Optional[str] = None) -> CodeResult:
        run_id = run_id or (self._context.run_id if self._context is not None else f"code-{uuid.uuid4().hex[:12]}")
        self._stop, self.meter = None, None
        if self._context is not None:
            blocked = self._begin_governed(run_id)
            if blocked is not None:
                return self._stopped(run_id, task, blocked)
        meter = self.meter
        if self._live_ollama:
            fail = self._preflight_live_model()
            if fail is not None:
                st, err = fail
                return self._finish(run_id, task, "", st, err)
        ws_map_block = ""
        if self._include_workspace_map:
            try:
                from osiris_cli import codebase_visibility

                ws_map = codebase_visibility.format_workspace_map_for_prompt(self.root, query=task, max_files=12, max_chars=1800)
                if ws_map:
                    ws_map_block = "\n" + ws_map
            except Exception:
                ws_map_block = ""
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": f"{SYSTEM}\nWorking directory: {self.root}{ws_map_block}"},
            {"role": "user", "content": task},
        ]
        seen_calls: Dict[str, Dict[str, Any]] = {}
        consecutive_breaker_steps = 0
        # Without a context: max_steps turns, as before. With one: the meter ends the loop, at the
        # step cap or at any other cap, and every turn and tool call is charged before it happens.
        for _ in (range(self.max_steps) if meter is None else itertools.count()):
            if meter is not None:
                stop = meter.begin_step()
                if stop is not None:
                    return self._stopped(run_id, task, stop)
            resp = self._chat(messages, TOOLS, self.model)
            if "error" in resp:
                err = str(resp["error"])
                st = ("MODEL_UNAVAILABLE" if "MODEL_NOT_INSTALLED" in err
                      else "UNAVAILABLE" if "DAEMON_UNAVAILABLE" in err else "ERROR")
                return self._finish(run_id, task, "", st, err)
            content = resp.get("content", "") or ""
            calls = list(resp.get("tool_calls") or [])
            if meter is not None:
                meter.charge_tokens(resp.get("tokens"), content + json.dumps(calls, default=str))
            origin = "native"
            if not calls and text_tool_calls_enabled(self.model):
                calls, origin = parse_text_tool_calls(content), "text"
            if not calls and self.write_nudges and not self._pending_proposals and not self._applied_writes():
                self.write_nudges -= 1
                retry = {"tool": "runner", "decision": "RETRY",
                         "reason": "answer without a write_file call while a change is required"}
                self.decisions.append(retry)
                self._log("CODE_TOOL", retry)
                self._out("\033[33m  ↻ no write_file call: nothing changed; asking the model once more\033[0m")
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": (
                    "RUNNER: no file was changed. A diff or code block in your reply is not a change. "
                    "Call the write_file tool now with the path and the complete new file content.")})
                continue
            if not calls:
                return self._finish(run_id, task, content or "(no answer)", "SUCCESS")
            messages.append({"role": "assistant", "content": content, "tool_calls": calls})
            all_tripped = True
            for call in calls:
                if meter is not None:
                    stop = meter.begin_tool_call()
                    if stop is not None:
                        return self._stopped(run_id, task, stop)
                fn = call.get("function", {}) or {}
                name = fn.get("name", "")
                raw = fn.get("arguments", {}) or {}
                try:
                    args = json.loads(raw) if isinstance(raw, str) else dict(raw)
                except (ValueError, TypeError):
                    args = None
                if args is None:
                    all_tripped = False
                    result = f"ERROR: arguments for {name} are not a JSON object"
                    bad = {"tool": name, "origin": origin, "decision": "ERROR",
                           "reason": "arguments are not a JSON object"}
                    self.decisions.append(bad)
                    self._log("CODE_TOOL", bad)
                else:
                    call_sig = _digest({"tool": name, "args": args})
                    prev = seen_calls.get(call_sig)
                    if prev is not None and (prev["decision"] in ("DENIED", "ERROR") or prev["count"] >= 2):
                        prev["count"] += 1
                        result = (
                            f"DENIED: duplicate call blocked by circuit breaker -- identical {name} call "
                            f"already returned {prev['decision']} ({prev['summary']}). Do not repeat this call; "
                            f"use a different command/tool or answer from what you observed."
                        )
                        blocked = {
                            "tool": name,
                            "origin": origin,
                            "args": {k: str(v)[:200] for k, v in args.items() if k != "content"},
                            "decision": "DENIED",
                            "reason": "duplicate_call_circuit_breaker",
                            "previous_decision": prev["decision"],
                        }
                        self.decisions.append(blocked)
                        self._log("CODE_TOOL", blocked)
                        self._out(f"\033[33m    ✗ {result}\033[0m")
                    else:
                        all_tripped = False
                        result = self.call(name, args, origin)
                        last_dec = self.decisions[-1]["decision"] if self.decisions else "GRANTED"
                        seen_calls[call_sig] = {
                            "decision": last_dec,
                            "count": (prev["count"] + 1) if prev else 1,
                            "summary": result[:160],
                        }
                if meter is not None:
                    meter.note(bool(self.decisions) and self.decisions[-1].get("decision") == "GRANTED")
                if len(result) > _MAX_TOOL_RESULT:
                    result = result[:3000] + f"\n...[{len(result) - 3000} chars omitted]...\n" + result[-1500:]
                messages.append({"role": "tool", "content": result, "name": name})
            if all_tripped:
                consecutive_breaker_steps += 1
                if consecutive_breaker_steps >= 2:
                    why = "halted repeated identical tool call after duplicate-call circuit breaker tripped twice"
                    if meter is not None:
                        from osiris_cli.task_context import STOPPED_BUDGET, Stop
                        return self._stopped(run_id, task, Stop(STOPPED_BUDGET, "repeated_failure",
                                                                f"run {run_id}: {why}"))
                    return self._finish(run_id, task, "", "CIRCUIT_BREAKER", why)
            else:
                consecutive_breaker_steps = 0
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
            if self._context is not None and not self._context.permissions.allows_tool(name):
                raise Denied(f"this task run ({self._context.run_id}) does not permit {name}; "
                             f"it may use: {', '.join(self._context.permissions.tools) or 'no tools'}")
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

    def _get_exec_journal(self):
        if self._exec_journal is not None:
            return self._exec_journal
        from osiris_cli.exec_kernel import ExecJournal
        return ExecJournal()

    def _grep(self, pattern, path=".", file_type=""):
        p = self._confine(path)
        try:
            rx = re.compile(str(pattern))
        except re.error as exc:
            raise Denied(f"invalid grep regex: {exc}")
        ext_filter = f".{str(file_type).lstrip('.')}" if file_type else ""
        # Refuse secret paths directly and skip them during recursive walk so workspace
        # credentials (.env, *.pem, *.key, service_account*.json, etc.) can never leak.
        secret_name_re = re.compile(
            r"(?i)(?:^\.env(?:\..+)?$|^.+\.env(?:\.txt)?$|\.(?:pem|key|p12|pfx)$"
            r"|^\.netrc$|^\.git-credentials$|^id_(?:rsa|ed25519|ecdsa|dsa|xmss)(?:\.pub)?$"
            r"|service[-_]account.*\.json$|ledger[^/]*\.(?:db|sqlite3?)$)"
        )
        secret_dir_parts = {".ssh", ".gnupg", ".aws", "keyrings"}
        secret_multi_seqs = ((".config", "gcloud"), (".config", "gh"))
        skip_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv"}

        def _is_secret_rel(rel_posix: str) -> bool:
            parts = [seg for seg in rel_posix.split("/") if seg]
            lowered = [seg.lower() for seg in parts]
            for seg in parts:
                if seg.lower() in secret_dir_parts or secret_name_re.search(seg):
                    return True
            for seq in secret_multi_seqs:
                n = len(seq)
                for i in range(len(lowered) - n + 1):
                    if tuple(lowered[i:i + n]) == seq:
                        return True
            return False

        root_prefix = self.root.rstrip(os.sep) + os.sep
        matches: List[str] = []

        def _scan_file(abs_file: str) -> None:
            if len(matches) >= 120:
                return
            try:
                st = os.lstat(abs_file)
            except OSError:
                return
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
                return
            if st.st_size > _MAX_READ_BYTES:
                return
            real_f = os.path.realpath(abs_file)
            if not (real_f == self.root or real_f.startswith(root_prefix)):
                return
            rel = os.path.relpath(real_f, self.root).replace(os.sep, "/")
            if _is_secret_rel(rel):
                return
            if ext_filter and not rel.endswith(ext_filter):
                return
            try:
                with open(real_f, "rb") as fh:
                    raw = fh.read(_MAX_READ_BYTES + 1)
            except OSError:
                return
            if b"\x00" in raw or len(raw) > _MAX_READ_BYTES:
                return
            for lineno, line in enumerate(raw.decode("utf-8", errors="replace").splitlines(), 1):
                if rx.search(line):
                    matches.append(f"{rel}:{lineno}:{line}")
                    if len(matches) >= 120:
                        return

        if os.path.isfile(p):
            rel_single = os.path.relpath(p, self.root).replace(os.sep, "/")
            if _is_secret_rel(rel_single):
                raise Denied(f"refusing to search secret path '{path}'")
            _scan_file(p)
        elif os.path.isdir(p):
            for dirpath, dirnames, filenames in os.walk(p, followlinks=False):
                rel_dir = os.path.relpath(dirpath, self.root).replace(os.sep, "/")
                if rel_dir != "." and _is_secret_rel(rel_dir):
                    dirnames[:] = []
                    continue
                kept_dirs = []
                for d in sorted(dirnames):
                    if d in skip_dirs or d.lower() in secret_dir_parts:
                        continue
                    full_d = os.path.join(dirpath, d)
                    try:
                        if stat.S_ISLNK(os.lstat(full_d).st_mode):
                            continue
                    except OSError:
                        continue
                    sub_rel = f"{rel_dir}/{d}" if rel_dir != "." else d
                    if _is_secret_rel(sub_rel):
                        continue
                    kept_dirs.append(d)
                dirnames[:] = kept_dirs
                for fname in sorted(filenames):
                    _scan_file(os.path.join(dirpath, fname))
                    if len(matches) >= 120:
                        break
                if len(matches) >= 120:
                    break
        else:
            raise Denied(f"'{path}' is not a file or directory")

        receipt_id = "exec-" + uuid.uuid4().hex[:12]
        try:
            from osiris_cli.exec_policy import load_policy as _load_exec_policy
            _, policy_sha = _load_exec_policy()
            self._get_exec_journal().append({
                "kind": "exec-receipt",
                "receipt_id": receipt_id,
                "ts_ms": int(time.time() * 1000),
                "origin": "model",
                "skill": None,
                "decision": "ALLOW",
                "classified": "ALLOW",
                "tier": "READ",
                "reasons": ["in-process SafePathGuard grep (secret-filtered, no subprocess)"],
                "argv": ["inprocess-grep", str(pattern), p],
                "cwd": self.root,
                "grant_id": None,
                "policy_sha256": policy_sha,
                "env_keys": [],
                "exit_code": 0,
                "timed_out": False,
                "duration_ms": 0,
                "artifact_sha256": None,
            })
        except Exception:
            pass
        return "\n".join(matches) or "(no matches)", {"path": p, "exec_receipt_id": receipt_id}

    def _write_file(self, path, content):
        abs_path = self._confine(path)  # refuse early with a reason the model can read
        rel_path = os.path.relpath(abs_path, self.root)
        if self.max_writes == 1:
            if self._wrote or self._has_pending_write():
                raise Denied("a write is already pending; the user must /apply or discard it first")
        else:
            if not self._wrote and self._has_pending_write():
                raise Denied("a write is already pending; the user must /apply or discard it first")
            if rel_path not in self._pending_proposals and len(self._pending_proposals) >= self.max_writes:
                raise Denied(f"maximum staged writes ({self.max_writes}) reached for this task")
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
        prop = {
            "abs_path": abs_path,
            "rel_path": rel_path,
            "content": new_text,
            "existed_at_proposal": existed,
            "before_sha256": before_sha,
            "after_sha256": after_sha,
            "diff": diff,
        }
        self._pending_proposal = prop
        self._pending_proposals[rel_path] = prop
        return ("PROPOSED: the diff is on the user's screen. It is applied only if they type /apply; "
                "do not say the file was written."), {
            "path": str(path), "chars": len(new_text), "before_sha256": before_sha, "after_sha256": after_sha}

    def _resolve_command_binary(self, prog: str, work: str) -> Optional[str]:
        if prog.startswith(("./", "../")) or (os.sep in prog and not os.path.isabs(prog)):
            cand = self._confine(os.path.join(work, prog))
            return cand if (os.path.isfile(cand) and os.access(cand, os.X_OK)) else None
        return shutil.which(prog)

    def _command_binding_payload(self, argv: List[str], work: str) -> Tuple[Optional[str], Dict[str, Any]]:
        resolved_bin = self._resolve_command_binary(argv[0], work)
        bin_sha = file_sha256(resolved_bin) if resolved_bin else None
        bound_files: Dict[str, str] = {}
        root_prefix = self.root.rstrip(os.sep) + os.sep
        for arg in argv[1:]:
            if not arg or arg.startswith("-") or "\x00" in arg:
                continue
            cand_raw = os.path.join(work, arg)
            try:
                cand_real = os.path.realpath(cand_raw)
            except OSError:
                continue
            if (cand_real == self.root or cand_real.startswith(root_prefix)) and os.path.isfile(cand_real):
                digest = file_sha256(cand_real)
                if digest is not None:
                    bound_files[os.path.relpath(cand_real, self.root)] = digest
        return resolved_bin, {
            "argv": argv,
            "cwd": work,
            "resolved_bin": resolved_bin,
            "resolved_bin_sha256": bin_sha,
            "bound_files": bound_files,
        }

    def _run_command(self, command, cwd=".", timeout=None):
        from osiris.substrate import interactive_timeout_s
        from osiris_cli.orchestration.schemas import ExecutionGrant

        argv = vet_argv(command)
        work = self._confine(cwd)

        resolved_bin, payload = self._command_binding_payload(argv, work)
        if resolved_bin is None:
            exe_name = argv[0] + ".exe" if not argv[0].lower().endswith(".exe") else argv[0]
            wsl_ps = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
            if _is_wsl() or shutil.which(exe_name) or (
                os.path.basename(argv[0]).lower() == "powershell" and os.path.exists(wsl_ps)
            ):
                raise Denied(
                    f"program not found: {argv[0]} (WSL2 detected: Windows binaries require '.exe' and "
                    f"non-interactive flags, e.g. 'powershell.exe -NoProfile -NonInteractive -Command \"...\"')"
                )
            raise Denied(f"program not found: {argv[0]}")

        if len(argv) == 1 and os.path.basename(argv[0]).lower() in _BARE_INTERACTIVE_REPLS:
            raise Denied(
                f"bare '{argv[0]}' starts an interactive REPL with stdin=/dev/null; pass non-interactive "
                f"execution flags (e.g. -c or -NoProfile -NonInteractive -Command)"
            )

        from osiris_cli import exec_kernel
        from osiris_cli.exec_policy import CommandRequest as _ExecRequest, classify as _exec_classify

        pre_req = _ExecRequest(argv=tuple([resolved_bin, *argv[1:]]), cwd=work, origin="model", root=self.root)
        pre_cls = _exec_classify(pre_req)
        if pre_cls.decision == "DENY":
            # Record the immutable DENY receipt via exec_kernel without prompting or minting a grant.
            exec_kernel.run(
                pre_req,
                confirm=lambda _p, _c: False,
                journal=self._get_exec_journal(),
                artifacts_dir=self._exec_artifacts_dir,
            )
            reason_txt = "; ".join(pre_cls.reasons) or "policy denied"
            raise Denied(f"execution policy refused this command ({pre_cls.tier}): {reason_txt}")

        digest = _digest(payload)
        cmd_scope = ["SUBPROCESS_EXEC", work]
        enclave = self._get_enclave()
        from osiris_cli.pqc_enclave import ApprovalDeniedError, GrantVerificationError
        from osiris_cli import session_approval as sa
        pqc_grant = None
        if self._session_grant is not None:
            ok_s, _why, scope = sa.verify_session_grant(enclave, self._session_grant)
            covered = sa.covers_command(scope, argv, work) if ok_s else (False, _why)
            if covered[0]:
                try:
                    pqc_grant = sa.issue_child_grant(
                        enclave, self._session_grant, action="code.run_command", scope=cmd_scope,
                        payload_sha256=digest, operation_schema=payload, covered=covered)
                    self._out(f"  [session] running `{shlex.join(argv)}` (listed in your session approval)")
                except GrantVerificationError as exc:
                    self._out(f"  [session] not covered ({exc}); asking you instead")
            else:
                self._out(f"  [session] `{shlex.join(argv)}` is not covered ({covered[1]}); asking you")
        if pqc_grant is None:
            try:
                pqc_grant = enclave.request_approved_grant(
                    action="code.run_command",
                    scope=cmd_scope,
                    summary=f"[{pre_cls.tier}] Run `{shlex.join(argv)}` in {work}",
                    payload_sha256=digest,
                    operation_schema=payload,
                    confirm=self._confirm,
                    out=self._out,
                    prompt=f"[{pre_cls.tier}] Run `{shlex.join(argv)}` in {work}? [y/N] ",
                    ttl_seconds=60.0,
                )
            except ApprovalDeniedError:
                raise Denied("the user did not approve this command")
        now = time.time()
        grant = ExecutionGrant(grant_id=pqc_grant["grant_id"], payload_digest=digest,
                               authorized_by="operator", authorized_at=now, expires_at=now + 60,
                               scope=cmd_scope)
        horizon = interactive_timeout_s()
        try:
            limit = min(float(timeout), horizon) if timeout else horizon
        except (TypeError, ValueError):
            limit = horizon
        # Re-bind resolved executable and workspace-local script hashes immediately before launch,
        # verify child-session link first (no spend on invalid link), then verify & consume via pure verifier.
        resolved_bin_now, recomputed_payload = self._command_binding_payload(argv, work)
        if resolved_bin_now is None:
            raise Denied(f"program not found: {argv[0]}")
        recomputed_digest = _digest(recomputed_payload)
        verifier = enclave.as_verifier()
        ok_pqc, why_pqc = sa.verify_child_link(verifier, pqc_grant, self._session_grant)
        if ok_pqc:
            ok_pqc, why_pqc = verifier.verify_grant_integrity(
                pqc_grant,
                expected_payload_sha256=recomputed_digest,
                expected_scope=cmd_scope,
                expected_event_type="code.run_command",
                consume=True,
            )
        if not ok_pqc or not grant.is_valid(recomputed_digest):
            raise Denied(f"execution grant verification failed: {why_pqc}")

        # Execute strictly through exec_kernel under origin="model": policy DENY always blocks
        # before Popen, while ASK is satisfied by the PQC grant verified above.
        res = exec_kernel.run(
            _ExecRequest(
                argv=tuple([resolved_bin_now, *argv[1:]]),
                cwd=work,
                origin="model",
                timeout_s=limit,
                root=self.root,
            ),
            confirm=lambda _prompt, _cls: True,
            journal=self._get_exec_journal(),
            artifacts_dir=self._exec_artifacts_dir,
        )
        refused = _kernel_refused_before_spawn(res)
        if refused is not None:
            raise Denied(f"execution policy refused this command: {refused}")
        failed = _kernel_spawn_failure(res)
        if failed is not None:
            raise Denied(f"program not found: {argv[0]} ({failed})")
        meta: Dict[str, Any] = {
            "grant_id": grant.grant_id,
            "key_fingerprint": pqc_grant["key_fingerprint"],
            "approval_mode": pqc_grant.get("approval_mode"),
            "digest": digest,
            "exec_receipt_id": res.receipt_id,
        }
        if res.timed_out or res.decision == "TIMEOUT":
            meta["timed_out"] = True
            return f"TIMEOUT after {limit:.0f}s", meta
        stdout = res.stdout_view or ""
        stderr = res.stderr_view or ""
        out = stdout + (f"\n[stderr]\n{stderr}" if stderr.strip() else "")
        meta["exit_code"] = res.exit_code
        meta["stdout_sha256"] = res.stdout_sha256 or _sha256_bytes(stdout.encode("utf-8"))
        meta["stderr_sha256"] = res.stderr_sha256 or _sha256_bytes(stderr.encode("utf-8"))
        if res.truncated or res.artifact_truncated:
            meta["output_truncated"] = True
        return (out or f"(exit code {res.exit_code}, no output)"), meta

    def _get_enclave(self):
        # No throwaway self-trusting key: approvals are signed by the operator's local key and
        # verified against its trust store. With no key set up, the operator is offered setup.
        if self._enclave is None:
            from osiris_cli.pqc_enclave import KeySetupRequired, load_or_onboard
            try:
                self._enclave = load_or_onboard(confirm=self._confirm, out=self._out)
            except KeySetupRequired as exc:
                raise Denied(str(exc))
        return self._enclave


    # ── acceptance gate ──────────────────────────────────────────────────────
    def verify(self, command: str, *, require_change: bool = True,
               timeout: float = 600.0) -> Dict[str, Any]:
        """Run the operator's acceptance command and decide the task's disposition.

        The model never calls this and its answer never counts as evidence. VERIFIED needs
        all of: the run ended SUCCESS; no proposal is still unapplied; at least one applied
        write when ``require_change``; every applied file still has the hash recorded at
        apply time; and the acceptance command exits 0 within ``timeout``. Anything else is
        FAILED or BLOCKED with the reasons listed. The result is receipted when receipts are on.
        """
        gate: List[Tuple[str, str]] = []
        if self.last_status != "SUCCESS":
            gate.append(("BLOCKED", f"run ended {self.last_status or 'NOT_RUN'}, not SUCCESS"))
        if self._pending_proposals:
            gate.append(("BLOCKED", f"{len(self._pending_proposals)} proposed write(s) not applied"))
        applied = self._applied_writes()
        record = verify_applied(self.root, applied, command, require_change=require_change,
                                timeout=timeout, gate=gate, run_id=self.run_id,
                                unbacked=edit_claims(self.last_answer) if not applied else [],
                                exec_journal=self._exec_journal,
                                exec_artifacts_dir=self._exec_artifacts_dir)
        self.decisions.append({k: v for k, v in record.items() if k != "output_tail"})
        self._log("CODE_VERIFY", {k: v for k, v in record.items() if k != "output_tail"})
        if self.receipts_path:
            record["receipt_hash"] = append_receipt(
                self.receipts_path, "code.agent.verify", {k: v for k, v in record.items() if k != "output_tail"})
        return record


# ── verification (shared by CodeAgent.verify and resumed headless runs) ───────

def append_receipt(path: str, kind: str, payload: Dict[str, Any]) -> str:
    from osiris_cli.trio_authz import Receipts

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return Receipts(path).append(kind, payload)["hash"]


def file_sha256(path: str) -> Optional[str]:
    return _sha256_bytes(open(path, "rb").read()) if os.path.isfile(path) else None


def atomic_write(target: str, data: bytes) -> None:
    parent = os.path.dirname(target) or "."
    os.makedirs(parent, exist_ok=True)
    tmp = f"{target}.osiris-tmp-{uuid.uuid4().hex[:8]}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(tmp, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    if os.name == "posix":
        try:
            dir_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass


def _kernel_refused_before_spawn(result: Any) -> Optional[str]:
    """Why exec_kernel refused before a process existed, or None if it spawned."""
    if result.timed_out or result.decision in ("ALLOW", "ASK_APPROVED", "TIMEOUT"):
        return None
    if result.decision == "ERROR" and result.exit_code is not None:
        return None
    if any(str(reason).startswith("spawn failed") for reason in result.reasons):
        return None
    if result.decision in ("DENY", "ASK_DECLINED", "ERROR"):
        return "; ".join(str(reason) for reason in result.reasons) or result.decision
    return None


def _kernel_spawn_failure(result: Any) -> Optional[str]:
    if result.timed_out or result.exit_code is not None:
        return None
    for reason in result.reasons:
        text = str(reason)
        if text.startswith("spawn failed"):
            return text
    return None


def run_acceptance(
    root: str,
    command: str,
    timeout: float,
    *,
    exec_journal: Optional[Any] = None,
    exec_artifacts_dir: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run one vetted acceptance command in ``root`` through ``exec_kernel.run`` under
    ``origin='model'`` (no shell, credential-free env, process group killed on timeout,
    policy DENY enforced before Popen). Raises Denied when the command is refused."""
    from osiris_cli import exec_kernel
    from osiris_cli.exec_policy import CommandRequest as _ExecRequest

    argv = vet_argv(command)
    rec: Dict[str, Any] = {"argv": argv}
    started = time.time()
    journal = exec_journal if exec_journal is not None else exec_kernel.ExecJournal()
    res = exec_kernel.run(
        _ExecRequest(argv=tuple(argv), cwd=root, origin="model", timeout_s=timeout, root=root),
        confirm=lambda _prompt, _cls: True,
        journal=journal,
        artifacts_dir=exec_artifacts_dir,
    )
    refused = _kernel_refused_before_spawn(res)
    if refused is not None:
        raise Denied(f"execution policy refused acceptance command: {refused}")
    failed = _kernel_spawn_failure(res)
    if failed is not None:
        rec["start_error"] = failed
        return rec
    rec["elapsed_s"] = round(time.time() - started, 3)
    rec["exec_receipt_id"] = res.receipt_id
    fallback_out = (res.stdout_view or "") + (res.stderr_view or "")
    rec["output_sha256"] = res.combined_sha256 or _sha256_bytes(fallback_out.encode("utf-8"))
    rec["output_bytes"] = res.combined_bytes if res.combined_sha256 is not None else len(fallback_out.encode("utf-8"))
    tail_src = res.combined_tail if res.combined_sha256 is not None else fallback_out
    rec["output_tail"] = tail_src[-_MAX_VERIFY_OUTPUT:]
    if res.artifact_truncated:
        rec["output_truncated"] = True
    if res.timed_out or res.decision == "TIMEOUT":
        rec["exit_code"] = None
        rec["timed_out"] = True
    else:
        rec["exit_code"] = res.exit_code
    return rec


def verify_applied(root: str, applied: List[Dict[str, Any]], command: str, *, require_change: bool = True,
                   timeout: float = 600.0, gate: Optional[List[Tuple[str, str]]] = None,
                   run_id: str = "", unbacked: Optional[List[str]] = None,
                   exec_journal: Optional[Any] = None,
                   exec_artifacts_dir: Optional[Any] = None) -> Dict[str, Any]:
    """Decide VERIFIED / FAILED / BLOCKED for writes already applied under ``root``.

    ``applied`` rows need ``path`` (relative) and ``after_sha256``. ``gate`` carries
    (disposition, reason) pairs the caller already knows; the command runs only when no
    reason has been found yet.
    """
    reasons = [r for _, r in (gate or [])]
    disposition = "BLOCKED" if any(d == "BLOCKED" for d, _ in (gate or [])) else "VERIFIED"
    files = []
    for d in applied:
        now = file_sha256(os.path.join(root, d["path"]))
        files.append({"path": d["path"], "applied_sha256": d["after_sha256"], "disk_sha256": now})
        if now != d["after_sha256"]:
            reasons.append(f"{d['path']} changed after it was applied")
    if require_change and not applied:
        reasons.append("no applied change (a proposed or claimed edit is not a change)")
    record: Dict[str, Any] = {"tool": "verify", "run_id": run_id, "command": command, "files": files,
                              "unbacked_claims": list(unbacked or [])}
    try:
        vet_argv(command)
    except Denied as exc:
        disposition = "BLOCKED"
        reasons.append(f"acceptance command refused: {exc}")
    if not reasons:  # run only when the outcome is still open
        try:
            rec = run_acceptance(
                root, command, timeout,
                exec_journal=exec_journal,
                exec_artifacts_dir=exec_artifacts_dir,
            )
        except Denied as exc:
            disposition = "BLOCKED"
            reasons.append(f"acceptance command refused: {exc}")
        else:
            record.update(rec)
            if "start_error" in rec:
                reasons.append(f"acceptance command could not start: {rec['start_error']}")
            elif rec.get("timed_out"):
                reasons.append(f"acceptance command timed out after {timeout:.0f}s (process group killed)")
            elif rec["exit_code"] != 0:
                reasons.append(f"acceptance command exited {rec['exit_code']}")
    if reasons and disposition == "VERIFIED":
        disposition = "FAILED"
    record.update(decision=disposition, reasons=reasons)
    return record

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


_LEADING_CODE_ACTION = re.compile(
    r"^\s*(?:please\s+)?(?:explain|read|list|inspect|grep|find|search|write|edit|fix|add|create|"
    r"implement|build|generate|scaffold|refactor|update|run|test|check|verify)\b",
    re.IGNORECASE,
)


def normalize_code_task(task: str) -> str:
    """Unwrap outer quotes on pasted specs and add an explicit tool-loop directive when absent."""
    raw = task.strip()
    if not raw:
        return ""
    was_quoted = False
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ('"', "'") and ("\n" in raw or len(raw) > 400):
        raw = raw[1:-1].strip()
        was_quoted = True
    if ("\n" in raw and (was_quoted or len(raw) > 400)) and not _LEADING_CODE_ACTION.match(raw):
        return (
            "[Task from pasted specification: inspect relevant files in the working directory first "
            "(list_dir / read_file / grep_code), then propose concrete code changes via write_file "
            "(/apply to confirm) rather than merely summarizing the pasted text.]\n\n" + raw
        )
    return raw


#: /code --accept: the agent whose proposal waits for the console's /apply, and the command.
_repl_acceptance: Optional[Dict[str, Any]] = None
REPL_RECEIPTS = os.path.join(os.path.expanduser("~"), ".osiris", "code", "receipts.jsonl")


def _print_verification(v: Dict[str, Any]) -> None:
    colour = "32" if v["decision"] == "VERIFIED" else "33"
    print(f"\n\033[1;{colour}m[/code {v['decision']}]\033[0m acceptance `{v['command']}`"
          + (f" exited {v.get('exit_code')} in {v.get('elapsed_s')}s" if "exit_code" in v else ""))
    for r in v["reasons"]:
        print(f"\033[33m    - {r}\033[0m")
    if v.get("output_tail"):
        print("\033[2m" + "\n".join(v["output_tail"].splitlines()[-8:]) + "\033[0m")
    if v.get("receipt_hash"):
        print(f"\033[2m  receipt {v['receipt_hash'][:16]}\033[0m")
    print()


def after_console_apply() -> Optional[Dict[str, Any]]:
    """Called by the REPL after the console's /apply. Runs the pending acceptance command
    only if every proposal is on disk exactly as proposed (the console's own apply, with its
    backup and genome-ledger record, did the write)."""
    global _repl_acceptance
    pending, _repl_acceptance = _repl_acceptance, None
    if not pending:
        return None
    agent: CodeAgent = pending["agent"]
    for prop in list(agent._pending_proposals.values()):
        disk = file_sha256(prop["abs_path"])
        if disk != prop["after_sha256"]:
            print(f"\n\033[33m[/code]\033[0m {prop['rel_path']} is not on disk as proposed "
                  f"(/apply refused or the file changed); acceptance not run.\n")
            return None
        rec = {"tool": "apply_write", "decision": "GRANTED", "path": prop["rel_path"],
               "before_sha256": prop["before_sha256"], "after_sha256": disk,
               "approved_via": "/apply (console; genome ledger)", "run_id": agent.run_id}
        agent.decisions.append(rec)
        if agent.receipts_path:
            rec["receipt_hash"] = append_receipt(agent.receipts_path, "code.agent.apply_write", rec)
    agent._pending_proposals.clear()
    agent._pending_proposal = None
    agent._wrote = False
    v = agent.verify(pending["command"])
    _print_verification(v)
    return v


def discard_repl_acceptance() -> None:
    global _repl_acceptance
    _repl_acceptance = None


_ACCEPT_OPT = re.compile(r"""^\s*--accept\s+(?:"([^"]+)"|'([^']+)')(?:\s+(.*))?$""", re.DOTALL)


def _repl_task_ledger() -> str:
    """Where /code's task runs are recorded: the shared ledger, or beside REPL_RECEIPTS when
    that has been pointed elsewhere (so a test never writes ~/.osiris/tasks)."""
    from osiris_cli import task_context
    default = os.path.join(os.path.expanduser("~"), ".osiris", "code", "receipts.jsonl")
    if os.path.realpath(REPL_RECEIPTS) == os.path.realpath(default):
        return task_context.DEFAULT_LEDGER
    return os.path.join(os.path.dirname(os.path.abspath(REPL_RECEIPTS)), "task_runs.jsonl")


def run_code_command(state: Any, rest: str) -> CodeResult:
    """`/code [--multi] [--model NAME] [--accept "CMD"] <task>` from the REPL."""
    global _repl_acceptance
    model = None
    max_writes = 1
    accept: Optional[str] = None
    while True:
        m_accept = _ACCEPT_OPT.match(rest)
        if m_accept:
            accept = m_accept.group(1) or m_accept.group(2)
            rest = m_accept.group(3) or ""
            continue
        m_multi = re.match(r"^\s*--multi(?:\s+(.*))?$", rest, re.DOTALL)
        if m_multi:
            max_writes = 4
            rest = m_multi.group(1) or ""
            continue
        m_model = re.match(r"^\s*--model\s+(\S+)(?:\s+(.*))?$", rest, re.DOTALL)
        if m_model:
            model = m_model.group(1)
            rest = m_model.group(2) or ""
            continue
        break
    task = normalize_code_task(rest)
    if not task:
        print("Usage: /code [--multi] [--model NAME] [--accept \"CMD\"] <task>   -- tools: read/list/grep, "
              "proposed writes (/apply), approved commands; --accept runs CMD after /apply")
        return CodeResult("", "ERROR", [], "no task")
    if accept and max_writes > 1:
        print("[/code] --accept works with one proposed write; drop --multi or use "
              "`python -m osiris_cli.code_task --multi`.")
        return CodeResult("", "ERROR", [], "--accept with --multi")
    _repl_acceptance = None
    log = getattr(state, "log_event", None)
    from osiris_cli.pqc_enclave import KeySetupRequired, load_or_onboard
    try:
        repl_enclave = load_or_onboard(confirm=_tty_confirm, out=print)
    except KeySetupRequired as exc:
        print(f"\n[/code] {exc}\n")
        return CodeResult("", "ERROR", [], "no signing key set up")
    # This task's run: an ID, the risk tier of the operator's words, what the loop may use, and
    # its budget. Commands are permitted to be asked for; each still needs its own y/N.
    from osiris_cli import task_context
    try:
        context = task_context.new_context(
            task, kind="code", run_id=f"code-{uuid.uuid4().hex[:12]}", max_writes=max_writes,
            perms=task_context.permissions(propose_writes=True, run_commands=True))
    except task_context.TaskContextError as exc:
        print(f"\n[/code] not started: {exc}\n")
        return CodeResult("", task_context.BLOCKED, [], f"no task context: {exc}")
    agent = CodeAgent(model=model, max_writes=max_writes, log=(lambda e, d: log(e, d)) if log else None,
                      receipts_path=REPL_RECEIPTS if accept else None, write_nudges=1 if accept else 0,
                      enclave=repl_enclave, context=context,
                      task_ledger=task_context.TaskLedger(_repl_task_ledger()))
    print(f"\n\033[1;36m[/code]\033[0m {agent.model} · working directory {agent.root}")
    print(f"\033[2m  task run {context.summary()}\033[0m")
    result = agent.run(task, run_id=context.run_id)
    granted = sum(1 for d in result.decisions if d["decision"] == "GRANTED")
    denied = len(result.decisions) - granted
    if result.status == "SUCCESS":
        print(f"\n{result.answer}\n")
        if result.unbacked_claims:
            print("\033[33m  [/code] the answer claims a file change, but no change was applied "
                  "(no apply_write receipt):\033[0m")
            for claim in result.unbacked_claims:
                print(f"\033[33m    - {claim}\033[0m")
    else:
        print(f"\n\033[33m[/code {result.status}]\033[0m {result.error}\n")
    print(f"\033[2m  tools: {granted} granted · {denied} denied/failed"
          f"{' · a write is pending: /apply to confirm' if agent._wrote else ''}\033[0m\n")
    if accept and result.status == "SUCCESS":
        if agent._pending_proposals:
            _repl_acceptance = {"agent": agent, "command": accept}
            print(f"\033[2m  after /apply the runner runs: {accept}\033[0m\n")
        else:
            _print_verification(agent.verify(accept))
    return result
