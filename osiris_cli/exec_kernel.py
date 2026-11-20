"""Execution kernel: the one place OSIRIS spawns a program on behalf of a human, a model or a skill.

Flow: classify -> (ASK: confirm) -> mint grant -> verify grant -> spawn with shell=False ->
capture -> exactly one hash-chained receipt. The policy lives in ``exec_policy``.

Usage (explain only, runs nothing):
    python -m osiris_cli.exec_kernel --explain [--origin model] -- rm -rf x
"""

from __future__ import annotations

import argparse
import codecs
import contextlib
import hashlib
import hmac
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from .exec_policy import (
    AGENT_ORIGINS,
    ORIGINS,
    Classification,
    CommandRequest,
    PolicyError,
    classify,
    is_sensitive_argv,
    load_policy,
    scrubbed_env,
)

try:  # POSIX only; the kernel's process-group handling is POSIX-only too
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]

__all__ = [
    "CommandRequest",
    "Classification",
    "CommandResult",
    "ExecJournal",
    "ExecutionGrant",
    "classify",
    "mint_grant",
    "reset_session_approvals",
    "run",
    "verify_grant",
]

CANON_VERSION = "exec-canon-1"
_GENESIS = "0" * 64
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")


@dataclass(frozen=True)
class CommandResult:
    exit_code: Optional[int]
    stdout_view: str
    stderr_view: str
    truncated: bool
    duration_s: float
    timed_out: bool
    artifact_path: Optional[str]
    artifact_sha256: Optional[str]
    grant_id: Optional[str]
    receipt_id: str
    decision: str  # ALLOW | ASK_APPROVED | ASK_DECLINED | DENY | TIMEOUT | ERROR
    reasons: Tuple[str, ...]
    provenance: str = "COMPUTED"
    stdout_sha256: Optional[str] = None
    stderr_sha256: Optional[str] = None
    combined_sha256: Optional[str] = None
    combined_bytes: int = 0
    combined_tail: str = ""
    artifact_truncated: bool = False


@dataclass(frozen=True)
class ExecutionGrant:
    """A single-use permission bound to one normalized argv, cwd, env key set and policy hash."""

    grant_id: str
    digest: str
    expires_ms: int


# --------------------------------------------------------------------------- canonical JSON

def canon(obj: object) -> str:
    """Canonical JSON for journal bodies (canon-1). Floats are rejected so the bytes are stable."""
    _reject_floats(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _reject_floats(obj: object) -> None:
    if isinstance(obj, float):
        raise ValueError("floats are not allowed in journal bodies (canon-1)")
    if isinstance(obj, dict):
        for value in obj.values():
            _reject_floats(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _reject_floats(value)


def _link_hash(prev: str, body: dict) -> str:
    return hashlib.sha256((prev + canon(body)).encode("utf-8")).hexdigest()


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- receipts journal

def _default_state_home() -> Path:
    osiris_home = os.environ.get("OSIRIS_HOME")
    if osiris_home:
        return Path(osiris_home).expanduser()
    return Path.home() / ".osiris"


def default_journal_path() -> Path:
    override = os.environ.get("OSIRIS_EXEC_JOURNAL")
    if override:
        return Path(override).expanduser()
    return _default_state_home() / "exec_journal.jsonl"


def default_artifacts_dir() -> Path:
    override = os.environ.get("OSIRIS_EXEC_ARTIFACTS_DIR")
    if override:
        return Path(override).expanduser()
    return _default_state_home() / "runs"


@contextlib.contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    if fcntl is None:  # pragma: no cover
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = str(path) + ".lock"
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _last_line(path: Path) -> Optional[bytes]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    size = path.stat().st_size
    chunks = b""
    pos = size
    with open(path, "rb") as handle:
        while True:
            step = min(4096, pos)
            pos -= step
            handle.seek(pos)
            chunks = handle.read(step) + chunks
            body = chunks.rstrip(b"\n")
            if pos == 0 or b"\n" in body:
                break
    return chunks.rstrip(b"\n").rsplit(b"\n", 1)[-1] or None


class ExecJournal:
    """Append-only, hash-chained JSONL of execution decisions. One line per decision."""

    def __init__(self, path: Optional[object] = None) -> None:
        self.path = Path(path) if path is not None else default_journal_path()
        self._thread_lock = threading.Lock()

    def append(self, body: dict) -> str:
        with self._thread_lock, _file_lock(self.path):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            last = _last_line(self.path)
            if last is None:
                prev, seq = _GENESIS, 0
            else:
                entry = json.loads(last.decode("utf-8"))
                prev, seq = entry["hash"], int(entry["seq"])
            digest = _link_hash(prev, body)
            entry = {"canon_version": CANON_VERSION, "seq": seq + 1, "prev": prev,
                     "body": body, "hash": digest}
            line = json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                              allow_nan=False)
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return digest

    def verify(self) -> Tuple[bool, int, str]:
        """Recompute the whole chain. Returns (ok, entries checked, first error)."""
        if not self.path.exists():
            return True, 0, ""
        prev, count = _GENESIS, 0
        with open(self.path, "r", encoding="utf-8") as handle:
            for lineno, raw in enumerate(handle, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                except ValueError:
                    return False, count, f"line {lineno}: not JSON"
                if entry.get("canon_version") != CANON_VERSION:
                    return False, count, f"line {lineno}: unknown canon_version"
                if entry.get("prev") != prev or entry.get("seq") != count + 1:
                    return False, count, f"line {lineno}: broken link or sequence"
                try:
                    want = _link_hash(prev, entry["body"])
                except (KeyError, ValueError) as exc:
                    return False, count, f"line {lineno}: {exc}"
                if want != entry.get("hash"):
                    return False, count, f"line {lineno}: hash mismatch"
                prev = want
                count += 1
        return True, count, ""


# --------------------------------------------------------------------------- grants

_APPROVED_LOCK = threading.Lock()
_APPROVED_PROGRAMS: set = set()


def reset_session_approvals() -> None:
    """Forget ask-once approvals. The REPL calls this on session start; tests use it for isolation."""
    with _APPROVED_LOCK:
        _APPROVED_PROGRAMS.clear()


def _program_key(argv: Sequence[str]) -> str:
    base = os.path.basename(argv[0])
    if "-m" in argv[1:]:
        idx = list(argv).index("-m", 1)
        if idx + 1 < len(argv):
            return f"{base} -m {argv[idx + 1]}"
    return base


def _grant_digest(argv: Sequence[str], cwd: str, env_keys: Sequence[str], sha: str) -> str:
    return _sha256_hex(canon({"argv": list(argv), "cwd": cwd, "env_keys": sorted(env_keys),
                              "policy_sha256": sha}).encode("utf-8"))


def mint_grant(argv: Sequence[str], cwd: str, env_keys: Sequence[str], sha: str) -> ExecutionGrant:
    ttl_s = float(load_policy()[0]["grant_ttl_s"])
    now_ms = int(time.time() * 1000)
    return ExecutionGrant(grant_id="grant-" + uuid.uuid4().hex[:12],
                          digest=_grant_digest(argv, cwd, env_keys, sha),
                          expires_ms=now_ms + int(ttl_s * 1000))


def verify_grant(grant: ExecutionGrant, argv: Sequence[str], cwd: str,
                 env_keys: Sequence[str], sha: str) -> bool:
    """True only if the grant is unexpired and its digest matches exactly this command."""
    if int(time.time() * 1000) > grant.expires_ms:
        return False
    return hmac.compare_digest(grant.digest, _grant_digest(argv, cwd, env_keys, sha))


# --------------------------------------------------------------------------- output capture

class _Capture:
    """Keeps the first HEAD and last TAIL characters of a stream, plus streaming SHA-256 & byte count."""

    def __init__(self, head_chars: int, tail_chars: int) -> None:
        self.head_chars = head_chars
        self.tail_chars = tail_chars
        self.total = 0
        self.total_bytes = 0
        self._head = ""
        self._tail = ""
        self._sha = hashlib.sha256()
        self._lock = threading.Lock()

    def feed(self, text: str) -> None:
        enc = text.encode("utf-8")
        with self._lock:
            self.total += len(text)
            self.total_bytes += len(enc)
            self._sha.update(enc)
            if len(self._head) < self.head_chars:
                self._head += text[: self.head_chars - len(self._head)]
            self._tail = (self._tail + text)[-self.tail_chars:]

    @property
    def truncated(self) -> bool:
        return self.total > self.head_chars + self.tail_chars

    def sha256_hex(self) -> str:
        with self._lock:
            return self._sha.hexdigest()

    def tail_view(self) -> str:
        with self._lock:
            return _ANSI.sub("", self._tail)

    def view(self) -> str:
        with self._lock:
            if not self.truncated:
                if self.total <= self.head_chars:
                    whole = self._head
                else:
                    whole = self._head + self._tail[len(self._tail) - (self.total - self.head_chars):]
                return _ANSI.sub("", whole)
            omitted = self.total - self.head_chars - self.tail_chars
            marker = f"\n[... {omitted} characters truncated ...]\n"
            return _ANSI.sub("", self._head) + marker + _ANSI.sub("", self._tail)


class _ArtifactWriter:
    """Writes every chunk as one JSON line to the run's artifact file and hashes what it wrote."""

    def __init__(self, path: Path, max_bytes: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.max_bytes = max_bytes
        self._hash = hashlib.sha256()
        self._written = 0
        self.truncated = False
        self._lock = threading.Lock()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        self._handle = os.fdopen(fd, "wb")

    def write(self, kind: str, text: str) -> None:
        line = (json.dumps({"stream": kind, "text": text}, ensure_ascii=False) + "\n").encode("utf-8")
        with self._lock:
            if self._written + len(line) > self.max_bytes:
                self.truncated = True
                return
            self._handle.write(line)
            self._hash.update(line)
            self._written += len(line)

    def close(self) -> str:
        with self._lock:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            return self._hash.hexdigest()


def _pump(pipe, kind: str, capture: _Capture, combined: Optional[_Capture],
          writer: Optional[_ArtifactWriter],
          callback: Optional[Callable[[str, str], None]]) -> None:
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    fd = pipe.fileno()
    try:
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            text = decoder.decode(chunk)
            if text:
                _deliver(kind, text, capture, combined, writer, callback)
        tail = decoder.decode(b"", final=True)
        if tail:
            _deliver(kind, tail, capture, combined, writer, callback)
    finally:
        pipe.close()


def _deliver(kind: str, text: str, capture: _Capture, combined: Optional[_Capture],
             writer: Optional[_ArtifactWriter],
             callback: Optional[Callable[[str, str], None]]) -> None:
    capture.feed(text)
    if combined is not None:
        combined.feed(text)
    if writer is not None:
        writer.write(kind, text)
    if callback is not None:
        try:
            callback(kind, text)
        except Exception:  # a broken live view must not break the run or its receipt
            pass


# --------------------------------------------------------------------------- process groups

def _group_alive(pgid: int) -> bool:
    """True if any process in the group is still running (zombies do not count)."""
    if not os.path.isdir("/proc"):  # pragma: no cover - fallback for non-Linux
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", "r") as handle:
                data = handle.read()
        except OSError:
            continue
        fields = data[data.rfind(")") + 2:].split()
        if len(fields) >= 3 and fields[0] != "Z" and int(fields[2]) == pgid:
            return True
    return False


def _wait_group_gone(pgid: int, grace_s: float) -> bool:
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if not _group_alive(pgid):
            return True
        time.sleep(0.05)
    return not _group_alive(pgid)


def _signal_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def _kill_group(pgid: int, grace_s: float) -> bool:
    """SIGTERM the group, then SIGKILL after the grace period. Returns True if survivors remain."""
    _signal_group(pgid, signal.SIGTERM)
    if _wait_group_gone(pgid, grace_s):
        return False
    _signal_group(pgid, signal.SIGKILL)
    return not _wait_group_gone(pgid, grace_s)


# --------------------------------------------------------------------------- timeouts

def time_limit(requested: Optional[float]) -> float:
    """The substrate's interactive horizon. A request may lower it, never raise it."""
    from osiris.substrate import interactive_timeout_s  # lazy: keeps classify() import-light

    horizon = float(interactive_timeout_s())
    if requested is None:
        return horizon
    try:
        value = float(requested)
    except (TypeError, ValueError):
        return horizon
    if not value > 0:  # also catches NaN
        return horizon
    return min(value, horizon)


# --------------------------------------------------------------------------- run

def _safe_argv(req: CommandRequest) -> List[str]:
    raw = req.argv if isinstance(req.argv, (tuple, list)) else ()
    return [str(tok) for tok in raw]


class _Attempt:
    """Accumulates the facts of one run and writes exactly one receipt when it finishes."""

    def __init__(self, req: CommandRequest, journal: ExecJournal) -> None:
        self.req = req
        self.journal = journal
        self.receipt_id = "rcpt-" + uuid.uuid4().hex
        self.t0 = time.monotonic()
        self.tier = "UNKNOWN"
        self.classified = ""
        self.reasons: List[str] = []
        self.argv: List[str] = _safe_argv(req)
        self.sha: Optional[str] = None
        self.env_keys: List[str] = []
        self.grant_id: Optional[str] = None
        self.cwd = str(req.cwd)

    def set_classification(self, cls: Classification) -> None:
        self.tier = cls.tier
        self.classified = cls.decision
        self.reasons = list(cls.reasons)
        self.argv = list(cls.normalized_argv)
        self.sha = cls.policy_sha256

    def finish(self, outcome: str, *, exit_code: Optional[int] = None, timed_out: bool = False,
               stdout_view: str = "", stderr_view: str = "", truncated: bool = False,
               artifact_path: Optional[str] = None, artifact_sha256: Optional[str] = None,
               stdout_sha256: Optional[str] = None, stderr_sha256: Optional[str] = None,
               combined_sha256: Optional[str] = None, combined_bytes: int = 0,
               combined_tail: str = "", artifact_truncated: bool = False) -> CommandResult:
        duration_ms = int((time.monotonic() - self.t0) * 1000)
        body = {
            "kind": "exec-receipt",
            "receipt_id": self.receipt_id,
            "ts_ms": int(time.time() * 1000),
            "origin": self.req.origin,
            "skill": self.req.skill,
            "decision": outcome,
            "classified": self.classified,
            "tier": self.tier,
            "reasons": self.reasons,
            "argv": self.argv,
            "cwd": self.cwd,
            "grant_id": self.grant_id,
            "policy_sha256": self.sha,
            "env_keys": self.env_keys,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "duration_ms": duration_ms,
            "artifact_sha256": artifact_sha256,
        }
        self.journal.append(body)
        return CommandResult(
            exit_code=exit_code,
            stdout_view=stdout_view,
            stderr_view=stderr_view,
            truncated=truncated,
            duration_s=duration_ms / 1000.0,
            timed_out=timed_out,
            artifact_path=artifact_path,
            artifact_sha256=artifact_sha256,
            grant_id=self.grant_id,
            receipt_id=self.receipt_id,
            decision=outcome,
            reasons=tuple(self.reasons),
            stdout_sha256=stdout_sha256,
            stderr_sha256=stderr_sha256,
            combined_sha256=combined_sha256,
            combined_bytes=combined_bytes,
            combined_tail=combined_tail,
            artifact_truncated=artifact_truncated,
        )


def _confirmed(confirm: Optional[Callable[[str, Classification], bool]], req: CommandRequest,
               cls: Classification) -> bool:
    if confirm is None:
        return False
    prompt = (f"[{cls.tier}] {shlex.join(cls.normalized_argv)}  "
              f"(cwd {req.cwd}, origin {req.origin}) approve? [y/N] ")
    try:
        return bool(confirm(prompt, cls))
    except Exception:
        return False


def run(req: CommandRequest,
        confirm: Optional[Callable[[str, Classification], bool]],
        journal: ExecJournal,
        stream: Optional[Callable[[str, str], None]] = None,
        artifacts_dir: Optional[object] = None) -> CommandResult:
    """Classify, confirm if needed, spawn with shell=False, capture, and write one receipt.

    ``confirm(prompt, classification) -> bool`` answers ASK decisions; None means decline.
    ``stream(kind, text)`` gets live chunks unless the output is credential-bearing.
    """
    att = _Attempt(req, journal)
    try:
        cls = classify(req)
    except PolicyError as exc:
        att.reasons = [str(exc)]
        return att.finish("ERROR")
    att.set_classification(cls)
    if cls.decision == "DENY":
        return att.finish("DENY")

    approved = False
    if cls.decision == "ASK":
        once = cls.tier == "VERIFY" and req.origin in AGENT_ORIGINS
        with _APPROVED_LOCK:
            already = once and _program_key(cls.normalized_argv) in _APPROVED_PROGRAMS
        if not already:
            if not _confirmed(confirm, req, cls):
                return att.finish("ASK_DECLINED")
            approved = True
            if once:
                with _APPROVED_LOCK:
                    _APPROVED_PROGRAMS.add(_program_key(cls.normalized_argv))

    try:
        env = scrubbed_env(os.environ)
        limit = time_limit(req.timeout_s)
    except Exception as exc:
        att.reasons.append(f"setup failed: {exc}")
        return att.finish("ERROR")

    env_keys = sorted(env)
    att.env_keys = env_keys
    cwd = os.path.realpath(req.cwd)
    att.cwd = cwd
    argv = tuple(cls.normalized_argv)
    grant = mint_grant(argv, cwd, env_keys, cls.policy_sha256)
    att.grant_id = grant.grant_id

    sensitive = is_sensitive_argv(argv)
    writer: Optional[_ArtifactWriter] = None
    artifact_path: Optional[Path] = None
    if not sensitive:
        try:
            root = Path(artifacts_dir) if artifacts_dir is not None else default_artifacts_dir()
            artifact_path = root / ("run-" + uuid.uuid4().hex) / "output.jsonl"
            pol = load_policy()[0]
            writer = _ArtifactWriter(artifact_path, int(pol["output"]["artifact_max_bytes"]))
        except Exception as exc:
            att.reasons.append(f"artifact setup failed: {exc}")
            return att.finish("ERROR")

    if not verify_grant(grant, argv, cwd, env_keys, cls.policy_sha256):
        if writer is not None:
            writer.close()
        att.reasons.append("grant does not match the command about to run")
        return att.finish("ERROR")

    try:
        proc = subprocess.Popen(
            list(argv), cwd=cwd, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,
            close_fds=True, start_new_session=True)
    except OSError as exc:
        if writer is not None:
            writer.close()
        att.reasons.append(f"spawn failed: {exc}")
        return att.finish("ERROR")

    pol = load_policy()[0]
    head_chars = int(pol["output"]["head_chars"])
    tail_chars = int(pol["output"]["tail_chars"])
    grace_s = float(pol["kill_grace_s"])
    cap_out = _Capture(head_chars, tail_chars)
    cap_err = _Capture(head_chars, tail_chars)
    cap_all = _Capture(head_chars, tail_chars)
    callback = None if sensitive else stream
    threads = [
        threading.Thread(target=_pump, args=(proc.stdout, "stdout", cap_out, cap_all, writer, callback),
                         daemon=True),
        threading.Thread(target=_pump, args=(proc.stderr, "stderr", cap_err, cap_all, writer, callback),
                         daemon=True),
    ]
    for thread in threads:
        thread.start()

    pgid = proc.pid
    timed_out = False
    survivors = False
    try:
        rc = proc.wait(timeout=limit)
    except subprocess.TimeoutExpired:
        timed_out = True
        survivors = _kill_group(pgid, grace_s)
        rc = proc.wait()
    for thread in threads:
        thread.join(timeout=5)
    stray_pipes = any(thread.is_alive() for thread in threads)

    artifact_sha = writer.close() if writer is not None else None
    artifact_truncated = bool(writer.truncated) if writer is not None else False
    if survivors:
        att.reasons.append("processes survived SIGKILL in the process group")
    if stray_pipes:
        att.reasons.append("output pipe held open by a process outside the group")

    if sensitive:
        out_view = err_view = "[output withheld: credential-bearing command]"
        truncated = False
        stdout_sha = stderr_sha = combined_sha = None
        combined_bytes = 0
        combined_tail = "[output withheld: credential-bearing command]"
    else:
        out_view, err_view = cap_out.view(), cap_err.view()
        truncated = cap_out.truncated or cap_err.truncated
        stdout_sha = cap_out.sha256_hex()
        stderr_sha = cap_err.sha256_hex()
        combined_sha = cap_all.sha256_hex()
        combined_bytes = cap_all.total_bytes
        combined_tail = cap_all.tail_view()

    outcome = "ERROR" if survivors else ("TIMEOUT" if timed_out else ("ASK_APPROVED" if approved else "ALLOW"))
    return att.finish(
        outcome,
        exit_code=rc,
        timed_out=timed_out,
        stdout_view=out_view,
        stderr_view=err_view,
        truncated=truncated,
        artifact_path=str(artifact_path) if artifact_path is not None else None,
        artifact_sha256=artifact_sha,
        stdout_sha256=stdout_sha,
        stderr_sha256=stderr_sha,
        combined_sha256=combined_sha,
        combined_bytes=combined_bytes,
        combined_tail=combined_tail,
        artifact_truncated=artifact_truncated,
    )


# --------------------------------------------------------------------------- CLI

def _split_shell_command(command: str) -> Tuple[List[str], Optional[str]]:
    """Split a shell command string preserving shell operator tokens without spawning a shell.

    Refuses multiline strings (newlines act as command separators in Bash) and NUL bytes.
    """
    if not isinstance(command, str) or not command.strip():
        return [], "empty command string"
    if "\x00" in command or "\n" in command or "\r" in command:
        return [], "multiline command or control character refused"
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars="|&;<>")
        lexer.whitespace_split = True
        lexer.commenters = "#"
        tokens = list(lexer)
    except ValueError as exc:
        return [], f"malformed shell syntax: {exc}"
    if not tokens:
        return [], "empty command string"
    return tokens, None


def classify_json_payload(
    raw_json: str,
    *,
    origin: str = "claude_code",
    cwd: Optional[str] = None,
    root: Optional[str] = None,
) -> Dict[str, Any]:
    """Classify a JSON hook payload without spawning any process.

    Caller-supplied ``origin`` and ``root`` are strictly authoritative and CANNOT be
    overridden by untrusted JSON stdin fields.
    """
    try:
        _, policy_sha = load_policy()
    except Exception:
        policy_sha = ""

    def _fail_closed(reason: str, req_origin: str) -> Dict[str, Any]:
        return {
            "tier": "FORBIDDEN",
            "decision": "DENY",
            "reasons": [reason],
            "normalized_argv": [],
            "policy_sha256": policy_sha,
            "origin": req_origin,
        }

    req_origin = origin
    req_cwd = cwd or os.getcwd()
    req_root = root
    try:
        data = json.loads(raw_json)
    except Exception as exc:
        return _fail_closed(f"malformed json payload: {exc}", req_origin)
    if not isinstance(data, dict):
        return _fail_closed("json payload must be an object", req_origin)

    # Validate tool_name when present: this classifier governs Bash / command invocations only.
    tool_name = data.get("tool_name")
    if tool_name is not None and tool_name != "Bash":
        return _fail_closed(f"unsupported tool_name {tool_name!r} for bash classifier", req_origin)

    if isinstance(data.get("cwd"), str) and data["cwd"]:
        req_cwd = data["cwd"]

    tool_input = data.get("tool_input")
    if isinstance(tool_input, dict) and isinstance(tool_input.get("cwd"), str) and tool_input["cwd"]:
        req_cwd = tool_input["cwd"]

    argv_list: Optional[List[str]] = None
    if isinstance(data.get("argv"), list):
        raw_argv = data["argv"]
        if not raw_argv or not all(isinstance(x, str) for x in raw_argv) or not raw_argv[0].strip():
            return _fail_closed("argv must be a non-empty list of strings with a non-empty program", req_origin)
        argv_list = list(raw_argv)
    else:
        cmd_str: Optional[str] = None
        if isinstance(tool_input, dict) and isinstance(tool_input.get("command"), str):
            cmd_str = tool_input["command"]
        elif isinstance(data.get("command"), str):
            cmd_str = data["command"]
        if cmd_str is None:
            return _fail_closed("missing command or argv in json payload", req_origin)
        tokens, err = _split_shell_command(cmd_str)
        if err is not None:
            return _fail_closed(err, req_origin)
        argv_list = tokens

    cls = classify(CommandRequest(argv=tuple(argv_list), cwd=req_cwd, origin=req_origin, root=req_root))
    return {
        "tier": cls.tier,
        "decision": cls.decision,
        "reasons": list(cls.reasons),
        "normalized_argv": list(cls.normalized_argv),
        "policy_sha256": cls.policy_sha256,
        "origin": req_origin,
    }


def _explain(argv: List[str], origin: str, cwd: Optional[str], root: Optional[str]) -> int:
    request = CommandRequest(argv=tuple(argv), cwd=cwd or os.getcwd(), origin=origin, root=root)
    cls = classify(request)
    print(f"tier: {cls.tier}")
    print(f"decision: {cls.decision} (origin {origin})")
    print("reasons: " + ("; ".join(cls.reasons) if cls.reasons else "-"))
    print("normalized argv: " + json.dumps(list(cls.normalized_argv)))
    print(f"policy sha256: {cls.policy_sha256}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m osiris_cli.exec_kernel")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--explain", action="store_true",
                      help="print the tier, decision and reasons; runs nothing")
    mode.add_argument("--classify-json", action="store_true",
                      help="read JSON hook payload from stdin and print machine-readable verdict; runs nothing")
    parser.add_argument("--origin", default="human", choices=ORIGINS)
    parser.add_argument("--cwd", default=None)
    parser.add_argument("--root", default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.classify_json:
        raw_in = sys.stdin.read()
        verdict = classify_json_payload(raw_in, origin=args.origin, cwd=args.cwd, root=args.root)
        print(json.dumps(verdict, sort_keys=True))
        return 0 if verdict.get("decision") == "ALLOW" else 2
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("no command given after --")
    return _explain(command, args.origin, args.cwd, args.root)


if __name__ == "__main__":
    sys.exit(main())
