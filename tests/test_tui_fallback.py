"""Stage 6 tests for inline TUI viewport fallback, width clamping, EXEC_GATE separation, and /skill routing."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pytest

from osiris_cli import osiris_repl, tui_inline
from osiris_cli.living import Osiris
from osiris_cli.probes import verify_chain, verify_chain_tail


class _FakeCore:
    name = "osiris.nclm"

    def __init__(self, bpb: float = 4.25) -> None:
        self._bpb = bpb
        self._step = 12
        self.learned: List[str] = []

    def step(self) -> int:
        return self._step

    def bits_per_byte(self, text: str) -> float:
        return self._bpb

    def learn(self, text: str, steps: int) -> float:
        self.learned.append(text)
        self._step += steps
        return 1.2

    def draft(self, prompt: str) -> str:
        return "raw core draft"

    def training_locked(self) -> bool:
        return False


class _ScriptedSkillMentor:
    def __init__(self, reply: str = "Verified preflight status from probe output.") -> None:
        self.reply = reply
        self.calls: List[List[Dict[str, str]]] = []
        self.last_done: Dict[str, object] = {"done_reason": "stop"}

    def model(self) -> Optional[str]:
        return "qwen2.5:7b"

    def set_model(self, name: Optional[str]) -> None:
        pass

    def stream(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> Iterator[str]:
        self.calls.append(messages)
        yield self.reply


def _write_valid_skill(base_dir: Path, name: str, description: str, body: str) -> Path:
    sdir = base_dir / ".agents" / "skills" / name
    sdir.mkdir(parents=True, exist_ok=True)
    path = sdir / "SKILL.md"
    content = f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n"
    path.write_bytes(content.encode("utf-8"))
    return path


def test_non_tty_fallback_never_hangs_and_reads_piped_stdin(tmp_path: Path) -> None:
    fake_in = io.StringIO("/skills list\n/gate\n")
    fake_out = io.StringIO()

    assert tui_inline.should_use_inline_tui(stdin=fake_in, stdout=fake_out, env={"OSIRIS_TUI": "inline"}) is False
    assert tui_inline.should_use_inline_tui(stdin=fake_in, stdout=fake_out, env={"OSIRIS_TUI": "plain"}) is False

    vp = tui_inline.InlineViewport(
        state=osiris_repl.OsirisReplState(),
        workspace_root=tmp_path,
        home_dir=tmp_path,
        stdin=fake_in,
        stdout=fake_out,
        columns=76,
    )
    assert vp.read_input() == "/skills list"
    assert vp.read_input() == "/gate"
    with pytest.raises(EOFError):
        vp.read_input()


def test_width_clamping_stays_within_76_columns_for_node_alpha(tmp_path: Path) -> None:
    _write_valid_skill(
        tmp_path,
        "osiris-preflight-check",
        "Use when running preflight health checks across Node Alpha and Node Beta substrates.",
        "# Preflight\nVerify git and ledger integrity.",
    )
    snap = tui_inline.ViewportSnapshot(
        substrate_tag="Alpha",
        sys026_status="CLOSED 4.82bpb",
        exec_gate="STRICT",
        mentor_tag="qwen2.5-coder:7b-instruct",
        ledger_head="05b5451f8c25",
        skill_count=5,
    )
    footer = tui_inline.render_status_footer(snap, width=76)
    assert len(footer) <= 76
    assert "Alpha" in footer
    assert "STRICT" in footer

    frame = tui_inline.render_inline_frame(
        "Long Diagnostic Panel Header For Narrow Samsung Fold Termux Viewport",
        "A" * 180 + "\n" + "short line",
        width=76,
    )
    for line in frame.splitlines():
        assert len(line) <= 76, f"Frame line exceeded 76 chars ({len(line)}): {line!r}"

    table_buf: List[str] = []
    rendered_table = tui_inline.execute_skills_command(
        "list",
        workspace_root=tmp_path,
        home_dir=tmp_path,
        width=76,
        out=table_buf.append,
    )
    for line in rendered_table.splitlines():
        assert len(line) <= 76, f"Skills table line exceeded 76 chars ({len(line)}): {line!r}"


def test_ctrl_g_and_gate_command_toggle_exec_gate_without_touching_sys026(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    state = osiris_repl.OsirisReplState()
    assert state.exec_gate == "STRICT"

    core = _FakeCore(bpb=4.25)
    living = Osiris(
        core=core,
        mentor=_ScriptedSkillMentor(),
        home=str(tmp_path / "living"),
        out=lambda s: None,
        background=False,
        tty=False,
    )
    gate_before = living.gate()
    assert gate_before["open"] is False

    vp = tui_inline.InlineViewport(
        state=state,
        living_getter=lambda: living,
        workspace_root=tmp_path,
        home_dir=tmp_path,
        columns=76,
    )
    assert vp.handle_ctrl_g() == "AUTO"
    assert state.exec_gate == "AUTO"
    assert living.gate() == gate_before
    assert getattr(living, "force_core_voice", False) is False

    assert vp.handle_ctrl_g() == "STRICT"
    assert state.exec_gate == "STRICT"
    assert living.gate() == gate_before

    # Verify REPL /gate command respects strict two-gate separation
    osiris_repl.dispatch_command(state, "/gate auto")
    assert state.exec_gate == "AUTO"
    assert "SYS-026 Speaking Gate is unchanged" in capsys.readouterr().out
    assert living.gate()["open"] is False

    osiris_repl.dispatch_command(state, "/gate strict")
    assert state.exec_gate == "STRICT"
    assert living.gate()["open"] is False


def test_dynamic_skill_completion_refreshes_immediately_on_new_skill(tmp_path: Path) -> None:
    ws = tmp_path / "repo"
    home = tmp_path / "home"
    ws.mkdir()
    home.mkdir()
    _write_valid_skill(
        ws,
        "alpha-check",
        "Use when checking initial workspace state before running tests.",
        "# Alpha Check\nRun deterministic check.",
    )

    vp = tui_inline.InlineViewport(state=osiris_repl.OsirisReplState(), workspace_root=ws, home_dir=home)
    tree1 = vp.build_completer_tree()
    assert tree1["/skill"] == {"alpha-check": None}
    assert vp.complete_input("/skill al") == ["/skill alpha-check"]

    _write_valid_skill(
        ws,
        "beta-audit",
        "Use when auditing claims and hash chains across OSIRIS ledgers.",
        "# Beta Audit\nVerify hash chain.",
    )
    tree2 = vp.build_completer_tree()
    assert tree2["/skill"] == {"alpha-check": None, "beta-audit": None}
    assert vp.complete_input("/skill be") == ["/skill beta-audit"]


def test_bounded_command_stream_enforces_strict_gate_and_computes_sha256(tmp_path: Path) -> None:
    captured: List[str] = []
    # Declined in STRICT mode -> never executes
    declined = tui_inline.stream_command_bounded(
        [sys.executable, "-c", "print('should_not_run')"],
        exec_gate="STRICT",
        require_approval=True,
        confirm_fn=lambda _: False,
        use_rich_live=False,
        out=captured.append,
    )
    assert declined.blocked is True
    assert declined.exit_code == 126
    assert "should_not_run" not in declined.output

    # Approved in STRICT mode -> runs, bounds tail_lines to 12, hashes full output
    py_script = "for i in range(25): print(f'line-{i:02d}')"
    approved = tui_inline.stream_command_bounded(
        [sys.executable, "-c", py_script],
        exec_gate="STRICT",
        require_approval=True,
        max_lines=12,
        width=76,
        confirm_fn=lambda _: True,
        use_rich_live=False,
        out=captured.append,
    )
    assert approved.blocked is False
    assert approved.exit_code == 0
    assert len(approved.tail_lines) == 12
    assert approved.tail_lines[0] == "line-13"
    assert approved.tail_lines[-1] == "line-24"
    expected_sha = hashlib.sha256(approved.output.encode("utf-8")).hexdigest()
    assert approved.sha256 == expected_sha

    # Destructive pattern is blocked even under AUTO
    blocked_rm = tui_inline.stream_command_bounded(
        ["rm", "-rf", "/"],
        exec_gate="AUTO",
        require_approval=True,
        use_rich_live=False,
        out=captured.append,
    )
    assert blocked_rm.blocked is True
    assert blocked_rm.exit_code == 126


def test_skill_command_enforces_exec_gate_runs_sys024_probes_audit_and_records_chain(tmp_path: Path) -> None:
    ws = tmp_path / "repo"
    home = tmp_path / "home"
    ws.mkdir()
    home.mkdir()
    (ws / "notes.txt").write_text("Deterministic evidence file.\n", encoding="utf-8")
    _write_valid_skill(
        ws,
        "osiris-claims-grounding-audit",
        "Use when auditing physics claims and grounding citations against repository evidence.",
        "# Audit\nReport SYS-024 verdicts and cite only verified files.",
    )

    state = osiris_repl.OsirisReplState()
    state.exec_gate = "STRICT"
    mentor = _ScriptedSkillMentor(reply="Confirmed from notes.txt: theta_lock is REFUTED.")
    living = Osiris(
        core=_FakeCore(),
        mentor=mentor,
        home=str(tmp_path / "osiris_home" / "living"),
        out=lambda s: None,
        background=False,
        tty=False,
    )

    # 1. STRICT confirmation declined -> fail closed, mentor not called, ledger untouched
    out_declined: List[str] = []
    res_declined = tui_inline.execute_skill_command(
        "osiris-claims-grounding-audit inspect notes.txt and theta_lock",
        state=state,
        living=living,
        workspace_root=ws,
        home_dir=home,
        confirm_fn=lambda _: False,
        out=out_declined.append,
    )
    assert res_declined.status == "CANCELLED"
    assert mentor.calls == []
    assert not Path(living.log_path).exists()

    # 2. STRICT confirmation approved -> runs SYS-024 + probes + mentor + audit + hash-chained receipt
    out_ok: List[str] = []
    res_ok = tui_inline.execute_skill_command(
        "osiris-claims-grounding-audit inspect notes.txt and theta_lock",
        state=state,
        living=living,
        workspace_root=ws,
        home_dir=home,
        confirm_fn=lambda _: True,
        width=76,
        out=out_ok.append,
    )
    assert res_ok.status == "SUCCESS"
    assert res_ok.grounding_passed is True
    assert res_ok.ledger_hash is not None
    assert res_ok.skill_receipt is not None
    assert res_ok.skill_receipt["schema"] == "osiris.skill_receipt.v1"
    assert res_ok.skill_receipt["skill_name"] == "osiris-claims-grounding-audit"
    assert len(mentor.calls) == 1
    assert "REFUTED" in mentor.calls[0][1]["content"]  # SYS-024 injected pre-model
    assert "notes.txt" in mentor.calls[0][1]["content"]  # Probe injected pre-model

    rows = [json.loads(ln) for ln in Path(living.log_path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) == 1
    assert rows[0]["skill_receipt"]["skill_name"] == "osiris-claims-grounding-audit"
    assert rows[0]["grounding_audit"]["passed"] is True
    assert rows[0]["distill_eligible"] is False
    assert "THETA_LOCK" in (rows[0]["flagged_claims"] or [])

    assert verify_chain(living.log_path)["intact"] is True
    assert verify_chain_tail(living.log_path, tail_window=5).intact is True

    # 3. Phantom citation in skill reply triggers grounding audit warning and stays hash-chained
    mentor.reply = "Fabricated claim citing `docs/NONEXISTENT_PHANTOM.md` with 0.1234 bits/byte."
    out_flag: List[str] = []
    res_flag = tui_inline.execute_skill_command(
        "osiris-claims-grounding-audit check notes.txt",
        state=state,
        living=living,
        workspace_root=ws,
        home_dir=home,
        confirm_fn=lambda _: True,
        out=out_flag.append,
    )
    assert res_flag.status == "SUCCESS"
    assert res_flag.grounding_passed is False
    assert "[GOVERNANCE WARNING]" in "".join(out_flag)
    assert verify_chain(living.log_path)["intact"] is True
    assert verify_chain_tail(living.log_path, tail_window=5).intact is True


def test_repl_dispatch_routes_skills_and_gates_bang_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_valid_skill(
        tmp_path,
        "osiris-preflight-check",
        "Use when running deterministic workspace and ledger preflight checks.",
        "# Preflight\nVerify git and ledger integrity.",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OSIRIS_GLOBAL_SKILLS_DIR", str(tmp_path / "empty_global"))

    state = osiris_repl.OsirisReplState()
    assert state.exec_gate == "STRICT"

    osiris_repl.dispatch_command(state, "/skills list")
    out_list = capsys.readouterr().out
    assert "osiris-preflight-check" in out_list

    osiris_repl.dispatch_command(state, "/skills audit")
    out_audit = capsys.readouterr().out
    assert "Audit: PASS" in out_audit

    # !cmd in STRICT mode with closed stdin fails closed without executing
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    osiris_repl.dispatch_command(state, "!echo should_never_print")
    out_bang = capsys.readouterr().out
    assert "Declined command" in out_bang
    assert "should_never_print" not in out_bang or "Declined command `echo should_never_print`" in out_bang


def test_tui_inline_passes_sys025_zero_stub_and_lf_gate() -> None:
    target = Path(tui_inline.__file__)
    raw = target.read_bytes()
    assert b"\r" not in raw, "CRLF detected in osiris_cli/tui_inline.py"
    tree = ast.parse(raw.decode("utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = [
                stmt
                for stmt in node.body
                if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str))
            ]
            assert body, f"Empty function body in {node.name}"
            if len(body) == 1:
                only = body[0]
                assert not isinstance(only, ast.Pass), f"Stub pass function: {node.name}"
                assert not (
                    isinstance(only, ast.Expr)
                    and isinstance(only.value, ast.Constant)
                    and only.value.value is Ellipsis
                ), f"Stub ellipsis function: {node.name}"
