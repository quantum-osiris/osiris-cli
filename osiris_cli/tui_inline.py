"""Claude-Code-style inline TUI viewport and governed skill/command UI for OSIRIS v6.0.0.

Design invariants:
- No full-screen alt-buffer: preserves native terminal scrollback and Termux copy/paste.
- Width-adaptive rendering: clamps every rendered frame/footer line to terminal columns
  (safe on narrow <100-col Termux Node Alpha displays, including 76-col screens).
- Strict Two-Gate Separation:
  * EXEC_GATE (STRICT | AUTO, toggled via Ctrl+G or /gate): governs human [y/N]
    confirmation before workspace writes, !cmd subprocesses, or /skill execution.
  * SYS-026 Speaking Gate (living.py): governs osiris.nclm direct speaking; NEVER
    modified, bypassed, or toggled by any UI hotkey or /gate command.
- Optional prompt_toolkit + rich integration with zero-crash fallback:
  * Uses prompt_toolkit PromptSession + dynamic NestedCompleter + Ctrl+G KeyBindings
    when prompt_toolkit is installed, stdin is a TTY, and OSIRIS_TUI != "plain".
  * Falls back cleanly to rich/ANSI inline framing + stdlib input() + pure-Python
    prefix completion when prompt_toolkit is absent, OSIRIS_TUI=plain, or non-TTY.
  * Never hangs on piped/non-TTY stdin or EOF.
- Governed /skills and /skill execution:
  * Uses skills_runtime.discover_skills / hydrate_skill / format_skills_table.
  * Enforces EXEC_GATE: STRICT confirmation before model invocation.
  * Runs SYS-024 claims.match + deterministic plan_and_run_probes pre-model.
  * Routes through OSIRIS's real active mentor/provider (never stub/mock URLs).
  * Runs audit_mentor_reply post-generation and records hash-chained exchanges with
    skill_receipt, probe_receipts, and grounding_audit into exchanges.jsonl.
"""

from __future__ import annotations

import collections
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

from osiris_cli import claims as claims_mod
from osiris_cli import skills_runtime
from osiris_cli.probes import (
    format_probe_receipts_for_prompt,
    plan_and_run_probes,
    probe_substrate,
    resolve_osiris_home,
    verify_chain_tail,
)

try:
    from rich.console import Console as RichConsole
    from rich.live import Live as RichLive
    from rich.panel import Panel as RichPanel
    from rich.text import Text as RichText

    HAS_RICH = True
except ImportError:
    RichConsole = None  # type: ignore[assignment]
    RichLive = None  # type: ignore[assignment]
    RichPanel = None  # type: ignore[assignment]
    RichText = None  # type: ignore[assignment]
    HAS_RICH = False

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import NestedCompleter
    from prompt_toolkit.key_binding import KeyBindings

    HAS_PROMPT_TOOLKIT = True
except ImportError:
    PromptSession = None  # type: ignore[assignment]
    NestedCompleter = None  # type: ignore[assignment]
    KeyBindings = None  # type: ignore[assignment]
    HAS_PROMPT_TOOLKIT = False


ExecGateMode = Literal["STRICT", "AUTO"]

DEFAULT_ROLLING_LINES = 12
MIN_VIEWPORT_WIDTH = 40
MAX_VIEWPORT_WIDTH = 120

_DANGEROUS_SHELL_RE = re.compile(
    r"(?:^|\s)(?:rm\s+-[a-zA-Z]*r[a-zA-Z]*f\s+[/~]|dd\s+if=|mkfs\.|shutdown\b|reboot\b|:(){)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ViewportSnapshot:
    substrate_tag: str
    sys026_status: str
    exec_gate: ExecGateMode
    mentor_tag: str
    ledger_head: str
    skill_count: int


@dataclass(frozen=True)
class CommandStreamResult:
    argv: Tuple[str, ...]
    exit_code: int
    output: str
    tail_lines: Tuple[str, ...]
    sha256: str
    blocked: bool = False
    reason: str = ""


@dataclass(frozen=True)
class SkillExecutionResult:
    status: str
    skill_name: str
    reply: str
    ledger_hash: Optional[str]
    skill_receipt: Optional[Dict[str, Any]]
    grounding_passed: bool
    probe_count: int
    detail: str = ""


def resolve_viewport_width(columns: Optional[int] = None) -> int:
    """Return safe viewport width clamped to terminal columns (never exceeds columns)."""
    if columns is None:
        try:
            columns = shutil.get_terminal_size(fallback=(80, 24)).columns
        except Exception:  # noqa: BLE001
            columns = 80
    cols = max(20, int(columns))
    return min(cols, MAX_VIEWPORT_WIDTH)


def should_use_inline_tui(
    stdin: Any = None,
    stdout: Any = None,
    env: Optional[Dict[str, str]] = None,
) -> bool:
    """Determine whether to activate the inline TUI prompt loop.

    Returns False on non-TTY streams or when OSIRIS_TUI=plain so piped commands
    and test harnesses never hang. When OSIRIS_TUI=inline is explicitly set on a
    TTY, activates the inline viewport (with rich/stdlib fallback if prompt_toolkit
    is absent). When OSIRIS_TUI is unset, activates automatically on interactive
    TTYs only when both prompt_toolkit and rich are installed.
    """
    in_stream = sys.stdin if stdin is None else stdin
    out_stream = sys.stdout if stdout is None else stdout
    env_map = os.environ if env is None else env

    mode = str(env_map.get("OSIRIS_TUI", "")).strip().lower()
    if mode in ("plain", "off", "0", "false"):
        return False

    is_tty = bool(getattr(in_stream, "isatty", lambda: False)()) and bool(
        getattr(out_stream, "isatty", lambda: False)()
    )
    if not is_tty:
        return False

    if mode in ("inline", "on", "1", "true"):
        return True

    return bool(HAS_PROMPT_TOOLKIT and HAS_RICH)


def normalize_exec_gate(value: str) -> ExecGateMode:
    """Normalize EXEC_GATE string strictly to 'STRICT' or 'AUTO' (fail-closed to STRICT)."""
    cleaned = (value or "").strip().upper()
    return "AUTO" if cleaned == "AUTO" else "STRICT"


def toggle_exec_gate(state: Any) -> ExecGateMode:
    """Toggle EXEC_GATE (STRICT <-> AUTO) on state without touching SYS-026."""
    current = normalize_exec_gate(getattr(state, "exec_gate", "STRICT"))
    nxt: ExecGateMode = "AUTO" if current == "STRICT" else "STRICT"
    if state is not None:
        state.exec_gate = nxt
    return nxt


def set_exec_gate(state: Any, mode: str) -> ExecGateMode:
    """Set EXEC_GATE explicitly on state without touching SYS-026."""
    nxt = normalize_exec_gate(mode)
    if state is not None:
        state.exec_gate = nxt
    return nxt


def _truncate_line(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def _short_substrate_tag(node_id: str, is_termux: bool) -> str:
    upper = (node_id or "").upper()
    if is_termux or "ALPHA" in upper:
        return "Alpha"
    if "BETA" in upper:
        return "Beta"
    return "Edge"


def collect_viewport_snapshot(
    state: Any = None,
    living: Any = None,
    workspace_root: Path | str | None = None,
    home_dir: Path | str | None = None,
) -> ViewportSnapshot:
    """Build deterministic status-footer telemetry using O(1) tail verification."""
    # Substrate
    hp = getattr(state, "hardware_profile", None)
    if hp is not None:
        node_val = getattr(getattr(hp, "node_type", None), "value", "")
        sub_tag = _short_substrate_tag(str(node_val), bool(getattr(hp, "is_termux", False)))
    else:
        sub = probe_substrate()
        sub_tag = _short_substrate_tag(sub.node_id, sub.is_termux)

    # EXEC_GATE (strictly separate from SYS-026)
    exec_gate = normalize_exec_gate(getattr(state, "exec_gate", "STRICT"))

    # SYS-026 Speaking Gate + Mentor + exchanges.jsonl O(1) tail hash
    sys026_status = "CLOSED"
    mentor_tag = "offline"
    ledger_head = "empty"

    if living is not None:
        try:
            gate = living.gate()
            if gate.get("open"):
                sys026_status = "OPEN"
            elif gate.get("core_bpb") is not None:
                sys026_status = f"CLOSED {float(gate['core_bpb']):.2f}bpb"
            else:
                sys026_status = "CLOSED"
        except Exception:  # noqa: BLE001
            sys026_status = "CLOSED"

        try:
            mentor_obj = getattr(living, "mentor", None)
            active = mentor_obj.model() if mentor_obj is not None else None
            mentor_tag = str(active) if active else "offline"
        except Exception:  # noqa: BLE001
            mentor_tag = "offline"

        log_path = getattr(living, "log_path", None)
        if log_path:
            tail = verify_chain_tail(log_path, tail_window=6)
            if tail.status == "MISSING":
                ledger_head = "empty"
            elif not tail.intact:
                ledger_head = "TAMPERED"
            else:
                head_hex = tail.tail_hash or ""
                ledger_head = head_hex[:8] if head_hex and head_hex != "genesis" else "empty"
    else:
        osiris_home = resolve_osiris_home(home_dir if home_dir is None else Path(home_dir) / ".osiris")
        ex_path = osiris_home / "living" / "exchanges.jsonl"
        if ex_path.is_file():
            tail = verify_chain_tail(ex_path, tail_window=6)
            if not tail.intact:
                ledger_head = "TAMPERED"
            elif tail.tail_hash and tail.tail_hash != "genesis":
                ledger_head = tail.tail_hash[:8]

    # Discovered valid skills count
    try:
        discovered = skills_runtime.discover_skills(
            workspace_root=workspace_root,
            home_dir=home_dir,
            include_invalid=False,
        )
        skill_count = len(discovered)
    except Exception:  # noqa: BLE001
        skill_count = 0

    return ViewportSnapshot(
        substrate_tag=sub_tag,
        sys026_status=sys026_status,
        exec_gate=exec_gate,
        mentor_tag=mentor_tag,
        ledger_head=ledger_head,
        skill_count=skill_count,
    )


def render_status_footer(snapshot: ViewportSnapshot, width: Optional[int] = None) -> str:
    """Render a compact, single-line width-clamped status footer."""
    max_w = resolve_viewport_width(width)
    full = (
        f"[{snapshot.substrate_tag}] "
        f"SYS-026:{snapshot.sys026_status} | "
        f"EXEC:{snapshot.exec_gate} | "
        f"M:{snapshot.mentor_tag} | "
        f"LEDGER:{snapshot.ledger_head} | "
        f"SKILLS:{snapshot.skill_count}"
    )
    if len(full) <= max_w:
        return full

    # Compact variant for narrow Termux displays (<100 cols, e.g. 76 cols)
    sys_short = snapshot.sys026_status.replace("CLOSED ", "C:").replace("CLOSED", "C").replace("OPEN", "O")
    mentor_short = snapshot.mentor_tag
    if len(mentor_short) > 14:
        mentor_short = mentor_short[:11] + "..."
    compact = (
        f"[{snapshot.substrate_tag}] "
        f"S26:{sys_short}|"
        f"EG:{snapshot.exec_gate}|"
        f"M:{mentor_short}|"
        f"L:{snapshot.ledger_head[:6]}|"
        f"SK:{snapshot.skill_count}"
    )
    return _truncate_line(compact, max_w)


def render_inline_frame(title: str, body: str, width: Optional[int] = None) -> str:
    """Render an inline bordered box strictly clamped to `width` columns."""
    max_w = resolve_viewport_width(width)
    inner_w = max(10, max_w - 4)

    title_clean = _truncate_line(title.strip(), max(4, inner_w - 2))
    top_dashes = max(0, max_w - len(title_clean) - 5)
    top = f"+- {title_clean} " + ("-" * top_dashes) + "+"
    top = top[:max_w]

    lines = [top]
    raw_lines = body.splitlines() or [""]
    for raw in raw_lines:
        seg = raw
        if not seg:
            lines.append("| " + (" " * inner_w) + " |")
            continue
        while len(seg) > inner_w:
            chunk = seg[:inner_w]
            lines.append(f"| {chunk:<{inner_w}} |")
            seg = seg[inner_w:]
        lines.append(f"| {seg:<{inner_w}} |")

    bottom = "+" + ("-" * max(0, max_w - 2)) + "+"
    lines.append(bottom[:max_w])
    return "\n".join(line[:max_w] for line in lines)


def default_confirm_prompt(
    prompt_text: str,
    stdin: Any = None,
    stdout: Any = None,
) -> bool:
    """Ask an explicit [y/N] confirmation; fails closed on EOF, error, or non-yes."""
    in_stream = sys.stdin if stdin is None else stdin
    out_stream = sys.stdout if stdout is None else stdout
    try:
        out_stream.write(f"{prompt_text} [y/N]: ")
        out_stream.flush()
        raw = in_stream.readline()
    except (OSError, KeyboardInterrupt, EOFError):
        return False
    if not raw:
        return False
    return raw.strip().lower() in ("y", "yes")


def stream_command_bounded(
    argv: Sequence[str],
    *,
    exec_gate: ExecGateMode = "STRICT",
    require_approval: bool = True,
    max_lines: int = DEFAULT_ROLLING_LINES,
    width: Optional[int] = None,
    cwd: Path | str | None = None,
    timeout: float = 120.0,
    confirm_fn: Optional[Callable[[str], bool]] = None,
    use_rich_live: Optional[bool] = None,
    out: Optional[Callable[[str], None]] = None,
) -> CommandStreamResult:
    """Execute a subprocess with bounded rolling output and canonical SHA-256 receipt.

    Enforces EXEC_GATE: STRICT confirmation before launching any process when
    `require_approval=True`, and blocks destructive shell patterns even in AUTO mode.
    """
    write = out or (lambda s: sys.stdout.write(s) and None)
    cmd_tuple = tuple(str(a) for a in argv if str(a))
    if not cmd_tuple:
        return CommandStreamResult(
            argv=(),
            exit_code=2,
            output="",
            tail_lines=(),
            sha256=hashlib.sha256(b"").hexdigest(),
            blocked=True,
            reason="Empty command refused.",
        )

    joined_cmd = " ".join(cmd_tuple)
    if _DANGEROUS_SHELL_RE.search(joined_cmd):
        msg = f"[EXEC_GATE BLOCKED] Destructive command pattern refused: {joined_cmd}\n"
        write(msg)
        return CommandStreamResult(
            argv=cmd_tuple,
            exit_code=126,
            output="",
            tail_lines=(),
            sha256=hashlib.sha256(b"").hexdigest(),
            blocked=True,
            reason="Destructive command pattern refused.",
        )

    gate_mode = normalize_exec_gate(exec_gate)
    if require_approval and gate_mode == "STRICT":
        asker = confirm_fn or default_confirm_prompt
        approved = bool(asker(f"[EXEC_GATE: STRICT] Execute command `{joined_cmd}`?"))
        if not approved:
            write(f"[EXEC_GATE: STRICT] Declined command `{joined_cmd}` (not executed).\n")
            return CommandStreamResult(
                argv=cmd_tuple,
                exit_code=126,
                output="",
                tail_lines=(),
                sha256=hashlib.sha256(b"").hexdigest(),
                blocked=True,
                reason="Operator declined EXEC_GATE: STRICT confirmation.",
            )

    max_w = resolve_viewport_width(width)
    window: collections.deque[str] = collections.deque(maxlen=max(1, int(max_lines)))
    full_lines: List[str] = []
    hasher = hashlib.sha256()

    live_enabled = (
        HAS_RICH
        and (use_rich_live if use_rich_live is not None else bool(getattr(sys.stdout, "isatty", lambda: False)()))
        and os.environ.get("OSIRIS_TUI", "").strip().lower() != "plain"
    )

    try:
        proc = subprocess.Popen(
            list(cmd_tuple),
            cwd=str(cwd) if cwd is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except OSError as exc:
        err_text = f"Command failed to start: {exc}\n"
        digest = hashlib.sha256(err_text.encode("utf-8")).hexdigest()
        write(f"[CMD ERROR] {err_text}")
        return CommandStreamResult(
            argv=cmd_tuple,
            exit_code=127,
            output=err_text,
            tail_lines=(err_text.strip(),),
            sha256=digest,
            blocked=False,
            reason=str(exc),
        )

    deadline = time.monotonic() + max(1.0, float(timeout))
    timed_out = False

    def _build_panel(status_label: str) -> Any:
        body_text = "\n".join(_truncate_line(ln, max(10, max_w - 6)) for ln in window) or "(no output)"
        return RichPanel(
            RichText(body_text),
            title=_truncate_line(f"{joined_cmd} [{status_label}]", max(10, max_w - 6)),
            width=max_w,
        )

    assert proc.stdout is not None
    if live_enabled and RichConsole is not None and RichLive is not None and RichPanel is not None:
        console = RichConsole(width=max_w)
        with RichLive(_build_panel("RUNNING"), console=console, refresh_per_second=8, transient=False) as live:
            for raw_line in proc.stdout:
                if time.monotonic() > deadline:
                    proc.kill()
                    timed_out = True
                    break
                hasher.update(raw_line.encode("utf-8", errors="replace"))
                clean = raw_line.rstrip("\r\n")
                full_lines.append(raw_line)
                window.append(clean)
                live.update(_build_panel("RUNNING"))
            exit_code = -1 if timed_out else proc.wait(timeout=max(1.0, deadline - time.monotonic()))
            digest = hasher.hexdigest()
            badge = "TIMEOUT" if timed_out else ("VERIFIED" if exit_code == 0 else f"EXIT {exit_code}")
            live.update(_build_panel(f"{badge} sha256:{digest[:12]}"))
    else:
        for raw_line in proc.stdout:
            if time.monotonic() > deadline:
                proc.kill()
                timed_out = True
                break
            hasher.update(raw_line.encode("utf-8", errors="replace"))
            clean = raw_line.rstrip("\r\n")
            full_lines.append(raw_line)
            window.append(clean)
        exit_code = -1 if timed_out else proc.wait(timeout=max(1.0, deadline - time.monotonic()))
        digest = hasher.hexdigest()
        badge = "TIMEOUT" if timed_out else ("VERIFIED" if exit_code == 0 else f"EXIT {exit_code}")
        frame = render_inline_frame(
            f"{joined_cmd} [{badge} sha256:{digest[:12]}]",
            "\n".join(window) or "(no output)",
            width=max_w,
        )
        write(frame + "\n")

    return CommandStreamResult(
        argv=cmd_tuple,
        exit_code=int(exit_code),
        output="".join(full_lines),
        tail_lines=tuple(window),
        sha256=digest,
        blocked=False,
        reason="timeout" if timed_out else "",
    )


def execute_skills_command(
    subcommand: str = "list",
    *,
    workspace_root: Path | str | None = None,
    home_dir: Path | str | None = None,
    width: Optional[int] = None,
    out: Optional[Callable[[str], None]] = None,
) -> str:
    """Run read-only `/skills [list|audit]` and return the formatted table."""
    write = out or (lambda s: sys.stdout.write(s) and None)
    sub = (subcommand or "list").strip().lower()
    include_invalid = sub == "audit"
    max_w = resolve_viewport_width(width)

    table = skills_runtime.format_skills_table(
        workspace_root=workspace_root,
        home_dir=home_dir,
        max_width=max_w,
        include_invalid=include_invalid,
    )
    lines = [table]
    if include_invalid:
        all_recs = skills_runtime.discover_skills(
            workspace_root=workspace_root,
            home_dir=home_dir,
            include_invalid=True,
        )
        invalid = [r for r in all_recs.values() if not r.valid]
        if invalid:
            lines.append("")
            lines.append(f"Audit failures ({len(invalid)}):")
            for rec in invalid:
                for iss in rec.issues:
                    lines.append(_truncate_line(f"  - {rec.name} [{iss.code}]: {iss.message}", max_w))
        else:
            lines.append("")
            lines.append(f"Audit: PASS ({len(all_recs)} valid skill(s), 0 schema issues).")

    rendered = "\n".join(lines)
    write("\n" + rendered + "\n\n")
    return rendered


def execute_skill_command(
    raw_args: str,
    *,
    state: Any = None,
    living: Any = None,
    workspace_root: Path | str | None = None,
    home_dir: Path | str | None = None,
    confirm_fn: Optional[Callable[[str], bool]] = None,
    width: Optional[int] = None,
    out: Optional[Callable[[str], None]] = None,
) -> SkillExecutionResult:
    """Execute `/skill <name> [task]` under EXEC_GATE, SYS-024, probes, grounding audit, and ledger chain."""
    write = out or (lambda s: sys.stdout.write(s) and None)
    tokens = (raw_args or "").strip().split(None, 1)
    if not tokens:
        msg = "Usage: /skill <skill-name> [task]  (run /skills to list installed skills)"
        write(f"\n[OSIRIS] {msg}\n\n")
        return SkillExecutionResult(
            status="INVALID_USAGE",
            skill_name="",
            reply="",
            ledger_hash=None,
            skill_receipt=None,
            grounding_passed=False,
            probe_count=0,
            detail=msg,
        )

    skill_name = tokens[0].strip()
    user_task = tokens[1].strip() if len(tokens) > 1 else ""

    ws_root = Path(workspace_root).expanduser().resolve() if workspace_root else Path.cwd().resolve()
    osiris_home = resolve_osiris_home(
        Path(living.home).parent if (living is not None and getattr(living, "home", None)) else None
    )

    try:
        bundle = skills_runtime.hydrate_skill(
            skill_name,
            user_task=user_task,
            workspace_root=ws_root,
            home_dir=home_dir,
        )
    except (KeyError, ValueError, PermissionError, FileNotFoundError) as exc:
        msg = f"[OSIRIS::SKILL BLOCKED] {exc}"
        write(f"\n{msg}\n\n")
        return SkillExecutionResult(
            status="BLOCKED",
            skill_name=skill_name,
            reply="",
            ledger_hash=None,
            skill_receipt=None,
            grounding_passed=False,
            probe_count=0,
            detail=str(exc),
        )

    # Enforce EXEC_GATE: STRICT requires explicit operator confirmation before invoking the skill.
    exec_gate = normalize_exec_gate(getattr(state, "exec_gate", "STRICT"))
    if exec_gate == "STRICT":
        asker = confirm_fn or default_confirm_prompt
        approved = bool(
            asker(
                f"[EXEC_GATE: STRICT] Run skill '{bundle.name}' "
                f"(scope={bundle.scope}, sha256={bundle.skill_sha256[:12]})?"
            )
        )
        if not approved:
            msg = f"[EXEC_GATE: STRICT] Declined skill '{bundle.name}' (not executed)."
            write(f"\n{msg}\n\n")
            return SkillExecutionResult(
                status="CANCELLED",
                skill_name=bundle.name,
                reply="",
                ledger_hash=None,
                skill_receipt=dict(bundle.ledger_receipt),
                grounding_passed=False,
                probe_count=0,
                detail=msg,
            )

    # Pre-model SYS-024 claims check
    claim_hits = claims_mod.match(bundle.user_prompt)
    claims_context = ""
    if claim_hits:
        write("\n" + claims_mod.console_block(claim_hits) + "\n")
        claims_context = claims_mod.context_block(claim_hits)

    # Deterministic pre-answer probes (SafePathGuard enforced)
    probe_receipts = plan_and_run_probes(
        bundle.user_prompt,
        workspace_root=ws_root,
        osiris_home=osiris_home,
    )
    probe_context = format_probe_receipts_for_prompt(probe_receipts)

    retrieved_files: set[str] = {str(bundle.skill_file)}
    for rec in probe_receipts:
        if rec.probe == "read_file" and not rec.blocked and rec.target:
            retrieved_files.add(rec.target)

    if living is None or getattr(living, "mentor", None) is None:
        msg = "[OSIRIS::SKILL] No mentor runtime is connected; skill not executed."
        write(f"\n{msg}\n\n")
        return SkillExecutionResult(
            status="OFFLINE",
            skill_name=bundle.name,
            reply="",
            ledger_hash=None,
            skill_receipt=dict(bundle.ledger_receipt),
            grounding_passed=False,
            probe_count=len(probe_receipts),
            detail=msg,
        )

    model_name = living.mentor.model()
    if not model_name:
        msg = (
            "[OSIRIS::SKILL] No reachable local mentor model; cannot run skill "
            f"'{bundle.name}'. Start Ollama or check installed models."
        )
        write(f"\n{msg}\n\n")
        return SkillExecutionResult(
            status="OFFLINE",
            skill_name=bundle.name,
            reply="",
            ledger_hash=None,
            skill_receipt=dict(bundle.ledger_receipt),
            grounding_passed=False,
            probe_count=len(probe_receipts),
            detail=msg,
        )

    user_parts = [bundle.user_prompt]
    if claims_context:
        user_parts.append(claims_context)
    if probe_context:
        user_parts.append(probe_context)
    assembled_user = "\n\n".join(user_parts)

    messages = [
        {"role": "system", "content": bundle.system_prompt},
        {"role": "user", "content": assembled_user},
    ]

    write(f"\n[SKILL:{bundle.name} · {model_name} · sha256:{bundle.skill_sha256[:12]}]\n")
    reply_chunks: List[str] = []
    interrupted = False
    try:
        for piece in living.mentor.stream(messages, model=model_name):
            reply_chunks.append(piece)
            write(piece)
    except KeyboardInterrupt:
        interrupted = True
        write(" ...(interrupted)")
    except (OSError, ConnectionError, ValueError) as exc:
        interrupted = True
        write(f"\n[OSIRIS::SKILL ERROR] mentor {model_name} failed: {exc}\n")
    write("\n")

    reply = "".join(reply_chunks)

    # Post-generation grounding & citation audit
    from osiris_cli.living import audit_mentor_reply

    context_corpus = bundle.system_prompt + "\n" + assembled_user
    audit = audit_mentor_reply(
        reply,
        retrieved_files=retrieved_files,
        context_corpus_text=context_corpus,
    )
    if reply and not audit.passed:
        write(audit.warning_banner())

    # Record into canonical hash-chained living/exchanges.jsonl with skill_receipt
    idx = living.stats.get("exchanges", 0)
    living.stats["exchanges"] = idx + 1
    rec_data: Dict[str, Any] = {
        "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "voice": f"mentor:{model_name}",
        "heldout": False,
        "learnable": False,
        "distill_eligible": False,
        "user": f"/skill {bundle.name}" + (f" {user_task}" if user_task else ""),
        "reply": reply,
        "index": idx,
        "interrupted": interrupted,
        "input_transport": "skill_command",
        "source_type": "skill_command",
        "skill_receipt": dict(bundle.ledger_receipt),
        "probe_receipts": [r.to_ledger_dict() for r in probe_receipts],
        "grounding_audit": audit.to_dict(),
        "flagged_claims": [c.id for c in claim_hits] or None,
    }
    ledger_hash = living._record(rec_data)
    living._save_stats()

    status = "INTERRUPTED" if interrupted else "SUCCESS"
    max_w = resolve_viewport_width(width)
    receipt_line = _truncate_line(
        f"  · skill_receipt:{bundle.skill_sha256[:12]} · ledger:{ledger_hash[:12]} · "
        f"probes:{len(probe_receipts)} · grounding:{'PASS' if audit.passed else 'FLAGGED'}",
        max_w,
    )
    write(receipt_line + "\n\n")

    return SkillExecutionResult(
        status=status,
        skill_name=bundle.name,
        reply=reply,
        ledger_hash=ledger_hash,
        skill_receipt=dict(bundle.ledger_receipt),
        grounding_passed=audit.passed,
        probe_count=len(probe_receipts),
    )


class InlineViewport:
    """Width-adaptive inline viewport with dynamic skill completion and Ctrl+G EXEC_GATE toggle."""

    def __init__(
        self,
        *,
        state: Any = None,
        living_getter: Optional[Callable[[], Any]] = None,
        workspace_root: Path | str | None = None,
        home_dir: Path | str | None = None,
        prompt_text: str = "osiris::}{> ",
        base_commands: Optional[Dict[str, Optional[Dict[str, None]]]] = None,
        stdin: Any = None,
        stdout: Any = None,
        columns: Optional[int] = None,
    ) -> None:
        self.state = state
        self.living_getter = living_getter
        self.workspace_root = Path(workspace_root).expanduser().resolve() if workspace_root else Path.cwd().resolve()
        self.home_dir = Path(home_dir).expanduser().resolve() if home_dir is not None else None
        self.prompt_text = prompt_text
        self.base_commands = base_commands
        self.stdin = stdin
        self.stdout = stdout
        self.columns = columns
        self._session: Any = None
        self._key_bindings: Any = None

    def _living(self) -> Any:
        if self.living_getter is None:
            return None
        try:
            return self.living_getter()
        except Exception:  # noqa: BLE001
            return None

    def handle_ctrl_g(self, _event: Any = None) -> ExecGateMode:
        """Toggle EXEC_GATE (STRICT <-> AUTO) cleanly; never touches SYS-026."""
        return toggle_exec_gate(self.state)

    def build_completer_tree(self) -> Dict[str, Optional[Dict[str, None]]]:
        """Rebuild slash + live /skill completion map from disk before each prompt."""
        return skills_runtime.build_completer_dict(
            base_commands=self.base_commands,
            workspace_root=self.workspace_root,
            home_dir=self.home_dir,
        )

    def complete_input(self, raw_buffer: str) -> List[str]:
        """Pure-Python prefix completion for fallback mode (works without prompt_toolkit)."""
        stripped = (raw_buffer or "").lstrip()
        if stripped.startswith("/skill"):
            return skills_runtime.complete_skill_input(
                stripped,
                workspace_root=self.workspace_root,
                home_dir=self.home_dir,
            )
        tree = self.build_completer_tree()
        if stripped.startswith("/"):
            return sorted(cmd for cmd in tree if cmd.startswith(stripped))
        return []

    def render_footer(self) -> str:
        """Render the current status toolbar clamped to active terminal width."""
        snap = collect_viewport_snapshot(
            state=self.state,
            living=self._living(),
            workspace_root=self.workspace_root,
            home_dir=self.home_dir,
        )
        return render_status_footer(snap, width=self.columns)

    def _ensure_pt_session(self) -> Any:
        if not HAS_PROMPT_TOOLKIT or PromptSession is None or KeyBindings is None:
            return None
        if self._session is None:
            kb = KeyBindings()

            @kb.add("c-g")
            def _on_ctrl_g(event: Any) -> None:
                self.handle_ctrl_g(event)

            self._key_bindings = kb
            self._session = PromptSession(key_bindings=kb)
        return self._session

    def read_input(self) -> str:
        """Read one user prompt line, using prompt_toolkit on interactive TTYs or safe fallback.

        Raises EOFError on closed/exhausted stdin (never hangs on piped non-TTY input).
        """
        in_stream = sys.stdin if self.stdin is None else self.stdin
        out_stream = sys.stdout if self.stdout is None else self.stdout
        is_tty = bool(getattr(in_stream, "isatty", lambda: False)()) and bool(
            getattr(out_stream, "isatty", lambda: False)()
        )
        plain_forced = os.environ.get("OSIRIS_TUI", "").strip().lower() in ("plain", "off", "0", "false")

        if is_tty and not plain_forced and HAS_PROMPT_TOOLKIT and NestedCompleter is not None:
            session = self._ensure_pt_session()
            if session is not None:
                completer = NestedCompleter.from_nested_dict(self.build_completer_tree())
                return session.prompt(
                    self.prompt_text,
                    completer=completer,
                    bottom_toolbar=self.render_footer,
                )

        # Safe fallback (works without prompt_toolkit, on OSIRIS_TUI=plain, and on non-TTY pipes)
        if is_tty and not plain_forced:
            footer = self.render_footer()
            out_stream.write(f"\033[2m{footer}\033[0m\n")
            out_stream.flush()

        out_stream.write(self.prompt_text)
        out_stream.flush()
        raw = in_stream.readline()
        if raw == "":
            raise EOFError
        return raw.rstrip("\r\n")


def run_bang_command(
    raw_line: str,
    *,
    state: Any = None,
    confirm_fn: Optional[Callable[[str], bool]] = None,
    width: Optional[int] = None,
    out: Optional[Callable[[str], None]] = None,
) -> CommandStreamResult:
    """Parse and run a gated `!command ...` through the bounded viewport."""
    write = out or (lambda s: sys.stdout.write(s) and None)
    body = raw_line[1:].strip() if raw_line.startswith("!") else raw_line.strip()
    if not body:
        write("\n[OSIRIS] Usage: !<command> [args...]  (gated by EXEC_GATE)\n\n")
        return CommandStreamResult(
            argv=(),
            exit_code=2,
            output="",
            tail_lines=(),
            sha256=hashlib.sha256(b"").hexdigest(),
            blocked=True,
            reason="Empty !command.",
        )
    try:
        argv = shlex.split(body)
    except ValueError as exc:
        write(f"\n[OSIRIS] Invalid command syntax: {exc}\n\n")
        return CommandStreamResult(
            argv=(),
            exit_code=2,
            output="",
            tail_lines=(),
            sha256=hashlib.sha256(b"").hexdigest(),
            blocked=True,
            reason=f"Invalid command syntax: {exc}",
        )
    gate_mode = normalize_exec_gate(getattr(state, "exec_gate", "STRICT"))
    return stream_command_bounded(
        argv,
        exec_gate=gate_mode,
        require_approval=True,
        width=width,
        confirm_fn=confirm_fn,
        out=write,
    )
