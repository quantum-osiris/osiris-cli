"""Offline end-to-end verification suite for Iterations R0-R4:
- R0 (B1, B2, B7): REPL Ctrl+C survival, `/learn <paste>` governed routing, narrow non-Termux
  banner, real `dnalang` v0.4.0 `/dna` compiler verification, and `/code` pre-prompt binary check +
  WSL2 interop hint + duplicate-call circuit breaker + multi-file atomic staging.
- R1 (B3): Delimited multi-block proposal transport (eliminating JSON double-escaping) + inner fence
  stripping + deterministic AST contract extractor & preservation verifier.
- R2 (B4): Structured invariant decomposition + strictly isolated Independent L5 Critic.
- R3 (B5): Closed-loop `verified_first_passing` (`k=3`) synthesis with `ProductionConfinementGuard`,
  digest-bound receipts, duplicate-hash circuit breaking, and repository `pytest` overlay execution.
- R4 (B6): Zero unapproved `_spawn_organism_learn` calls on rejected/unapplied synthesis,
  `protege` mentor evidence routing, `SYNTHESIS_MAX_TOKENS >= 4096`, and preserved `runner_sha256`.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import osiris_bench
import osiris_termux_console as otc
import payload_gate
import prompt_strategies
import protege
from osiris_cli import code_agent, osiris_repl, paste_learning, production_synth
from osiris_cli.code_agent import CodeAgent, Denied
from osiris_cli.living import Osiris
from tests.test_living import ScriptedMentor

REPO_ROOT = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════════════════
# ITERATION R0: REPL, /learn <paste>, /dna, AND /code HARDENING
# ═══════════════════════════════════════════════════════════════════════════════


def test_r0_boot_repl_ctrl_c_cancels_line_without_closing_session(monkeypatch, capsys):
    """Pressing Ctrl+C at `osiris::}{>` must print `^C` and continue, exiting only on EOFError."""
    events = []
    inputs = iter([
        KeyboardInterrupt(),
        "/clean",
        EOFError(),
    ])

    def fake_collect():
        item = next(inputs)
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setenv("OSIRIS_PREWARM", "0")
    monkeypatch.setattr(osiris_repl, "print_banner", lambda: None)
    monkeypatch.setattr(otc, "_terminal_setup", lambda _on: None)
    monkeypatch.setattr(otc, "collect_multiline_input", fake_collect)
    monkeypatch.setattr(osiris_repl, "dispatch_capturing", lambda _st, line: events.append(line))

    osiris_repl.boot_repl()
    out = capsys.readouterr().out
    assert "^C" in out
    assert events == ["/clean"], "REPL must survive KeyboardInterrupt and process the next command"
    assert "[OSIRIS] Session closed." in out


def test_r0_learn_pasted_document_routes_to_governed_review_card_while_numeric_runs_swarm(tmp_path, monkeypatch, capsys):
    living_home = tmp_path / "living"
    living_home.mkdir()
    living_inst = Osiris(
        core=None,
        mentor=ScriptedMentor(),
        home=str(living_home),
        out=lambda _s: None,
        background=False,
        tty=False,
    )
    monkeypatch.setattr(osiris_repl, "get_living", lambda: living_inst)
    swarm_generations = []
    monkeypatch.setattr(
        osiris_repl,
        "execute_learn",
        lambda state, generations=5: swarm_generations.append(generations),
    )

    state = osiris_repl.OsirisReplState()
    pasted_session = (
        '"Windows PowerShell 5.1\n'
        'PS C:\\Users\\there> Get-PnpDevice -PresentOnly\n'
        'Galaxy S26 Ultra ADB Interface OK"\n'
    )
    osiris_repl.dispatch_command(state, f"/learn {pasted_session}")
    out = capsys.readouterr().out

    assert swarm_generations == [], "/learn <paste> must not fall through to the [SWARM] stub"
    assert "GOVERNED LEARNING CANDIDATE REVIEW" in out
    assert "Galaxy S26 Ultra ADB Interface" in out
    cand = paste_learning.get_last_paste_candidate(str(living_home))
    assert cand is not None
    assert cand.input_transport == "bracketed_paste"
    assert cand.line_count == 3
    # Zero unconfirmed writes to destination stores
    for store in ("notes.jsonl", "candidate_facts.jsonl", "training_queue.jsonl", "learning_ledger.jsonl"):
        assert not (living_home / store).exists()

    # Numeric and bare `/learn` preserve backward-compatible swarm behavior
    osiris_repl.dispatch_command(state, "/learn 3")
    osiris_repl.dispatch_command(state, "/learn")
    assert swarm_generations == [3, 5]


def test_r0_banner_narrow_non_termux_workstation_does_not_mislabel_as_termux(monkeypatch, capsys):
    monkeypatch.delenv("OSIRIS_COMPACT", raising=False)
    monkeypatch.setattr(osiris_repl.SESSION, "compact_mode", False, raising=False)
    monkeypatch.setattr(osiris_repl, "get_terminal_width", lambda: 80)
    monkeypatch.setattr(osiris_repl, "otc", None)

    wsl_hp = SimpleNamespace(
        is_termux=False,
        node_type="NODE_BETA_WORKSTATION",
        arch="x86_64",
        execution_timeout_s=1800.0,
        allow_extended_horizons=True,
        coherence_floor=0.092,
    )
    monkeypatch.setattr(osiris_repl.SESSION, "hardware_profile", wsl_hp)
    monkeypatch.setattr(osiris_repl, "substrate_label", lambda _hp: "Node Beta (compute workstation) | x86_64 | WSL")
    osiris_repl.print_banner()
    out_wsl = capsys.readouterr().out
    assert "Mode: compact narrow (<100 cols)" in out_wsl
    assert "TERMUX" not in out_wsl

    termux_hp = SimpleNamespace(
        is_termux=True,
        node_type="NODE_ALPHA_MOBILE",
        arch="aarch64",
        execution_timeout_s=300.0,
        allow_extended_horizons=False,
        coherence_floor=0.092,
    )
    monkeypatch.setattr(osiris_repl.SESSION, "hardware_profile", termux_hp)
    monkeypatch.setattr(osiris_repl, "substrate_label", lambda _hp: "Node Alpha (Termux) | aarch64")
    osiris_repl.print_banner()
    out_termux = capsys.readouterr().out
    assert "MOBILE COMPACT / TERMUX (<100 cols)" in out_termux


def test_r0_execute_dna_compiles_valid_dnalang_v040_organism_without_refuted_constants(capsys):
    dnalang = pytest.importorskip("dnalang")
    state = osiris_repl.OsirisReplState()
    res = osiris_repl.execute_dna(state, intent="Synthesize XY4 decoupling organism for Heron r2")
    out = capsys.readouterr().out

    assert res["status"] == "SUCCESS"
    assert res["verified"] is True
    assert res["op_count"] > 0
    src = res["dna_source"]
    # Zero refuted constants or invalid legacy syntax
    for forbidden in ("46.9787", "51.843", "0.7734", "2.176435e-8", "express {"):
        assert forbidden not in src and forbidden not in out
    org = dnalang.parse(src)
    diags = dnalang.check(org)
    assert diags.ok() and diags.warnings == []


def test_r0_code_agent_missing_powershell_and_bare_repl_fail_before_prompt_and_trip_circuit_breaker(tmp_path, monkeypatch):
    """Reproduces the exact 6x `run_command('powershell')` death-spiral from the WSL2 devlog."""
    prompts = []
    monkeypatch.setattr(code_agent, "_is_wsl", lambda: True)
    monkeypatch.setattr(code_agent.shutil, "which", lambda prog: None if prog.startswith("powershell") else "/usr/bin/true")

    repeated_ps_call = {
        "content": '{"name": "run_command", "arguments": {"command": "powershell", "cwd": ".", "timeout": 300}}',
    }
    turns = [dict(repeated_ps_call) for _ in range(6)]
    agent = CodeAgent(
        root=str(tmp_path),
        model="qwen2.5-coder:7b",
        chat=lambda messages, tools, model: turns.pop(0) if turns else {"content": "done"},
        confirm=lambda p: prompts.append(p) or True,
        out=lambda _s: None,
    )
    res = agent.run("Diagnose Windows USB device from WSL2")
    assert prompts == [], "Missing binary must NEVER prompt operator [y/N]"
    assert res.status == "CIRCUIT_BREAKER"
    assert len(agent.decisions) == 3, "Circuit breaker must halt after initial denial + 2 blocked retries (not 6-8 steps)"
    assert "WSL2 detected" in agent.decisions[0]["reason"]
    assert "powershell.exe -NoProfile -NonInteractive -Command" in agent.decisions[0]["reason"]
    assert agent.decisions[1]["reason"] == "duplicate_call_circuit_breaker"
    assert agent.decisions[2]["reason"] == "duplicate_call_circuit_breaker"

    # Bare interactive interpreter (existing binary, 0 flags) also fails closed before [y/N]
    monkeypatch.undo()
    bare_prompts = []
    bare_turns = iter([
        {"content": "", "tool_calls": [{"function": {"name": "run_command", "arguments": {"command": sys.executable}}}]},
        {"content": "stopped"},
    ])
    bare_agent = CodeAgent(
        root=str(tmp_path),
        chat=lambda m, t, mod: next(bare_turns),
        confirm=lambda p: bare_prompts.append(p) or True,
        out=lambda _s: None,
    )
    r_bare = bare_agent.run("run python")
    assert r_bare.status == "SUCCESS"
    assert bare_prompts == []
    assert "interactive REPL" in bare_agent.decisions[0]["reason"]


def test_r0_code_agent_multi_file_atomic_staging_and_stale_guard(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "mod.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    turns = iter([
        {
            "content": "",
            "tool_calls": [
                {"function": {"name": "write_file", "arguments": {"path": "mod.py", "content": "def f():\n    return 2\n"}}},
                {"function": {"name": "write_file", "arguments": {"path": "test_mod.py", "content": "from mod import f\n\ndef test_f():\n    assert f() == 2\n"}}},
            ],
        },
        {"content": "staged module and test"},
    ])
    agent = CodeAgent(
        root=str(ws),
        max_writes=2,
        chat=lambda m, t, mod: next(turns),
        propose_write=lambda _p, _c: True,
        has_pending_write=lambda: False,
        out=lambda _s: None,
    )
    res = agent.run("update mod and add test")
    assert res.status == "SUCCESS"
    assert set(agent._pending_proposals.keys()) == {"mod.py", "test_mod.py"}
    assert not (ws / "test_mod.py").exists()

    applied = agent.apply_all_pending_writes(approve=lambda props: len(props) == 2)
    assert [r["decision"] for r in applied] == ["GRANTED", "GRANTED"]
    assert (ws / "mod.py").read_text(encoding="utf-8") == "def f():\n    return 2\n"
    assert (ws / "test_mod.py").exists()

    # Stale batch guard refuses entire batch if any staged file changed on disk
    stale_turns = iter([
        {
            "content": "",
            "tool_calls": [
                {"function": {"name": "write_file", "arguments": {"path": "mod.py", "content": "def f():\n    return 3\n"}}},
                {"function": {"name": "write_file", "arguments": {"path": "extra.py", "content": "VAL = 3\n"}}},
            ],
        },
        {"content": "staged"},
    ])
    stale_agent = CodeAgent(
        root=str(ws),
        max_writes=2,
        chat=lambda m, t, mod: next(stale_turns),
        propose_write=lambda _p, _c: True,
        has_pending_write=lambda: False,
        out=lambda _s: None,
    )
    stale_agent.run("batch")
    (ws / "mod.py").write_text("# external edit\n", encoding="utf-8")
    with pytest.raises(Denied, match="changed on disk"):
        stale_agent.apply_all_pending_writes(approve=lambda _p: True)
    assert not (ws / "extra.py").exists()


# ═══════════════════════════════════════════════════════════════════════════════
# ITERATIONS R1-R3: DELIMITED PARSER, AST CONTRACT, L5 CRITIC & CLOSED LOOP
# ═══════════════════════════════════════════════════════════════════════════════


def test_r1_delimited_and_fenced_parser_eliminates_json_double_escaping_and_strips_inner_fences():
    delimited_raw = (
        "=== FILE: claims_scan.py ===\n"
        "import re\n\n"
        "NUM_RE = re.compile(r\"(?<!\\w)\\d+(?:\\.\\d+)?(?!\\w)\")\n\n"
        "def find_nums(text: str) -> list[str]:\n"
        "    return NUM_RE.findall(text)\n"
        "=== TEST: test_proposal.py ===\n"
        "import claims_scan\n\n"
        "def test_proposal():\n"
        "    assert claims_scan.find_nums('fidelity 0.97.') == ['0.97']\n"
        "=== END ===\n"
    )
    p1 = production_synth.parse_delimited_or_json_proposal(delimited_raw)
    assert p1 is not None and p1["format"] == "delimited"
    assert p1["path"] == "claims_scan.py"
    assert r"(?<!\w)\d+(?:\.\d+)?(?!\w)" in p1["module"]

    # gemma3:4b failure mode: JSON object with ```python fences embedded inside string values
    fenced_json = json.dumps({
        "path": "mod.py",
        "module": "```python\ndef inc(x):\n    return x + 1\n```",
        "test": "```python\nimport mod\ndef test_proposal():\n    assert mod.inc(1) == 2\n```",
    })
    p2 = production_synth.parse_delimited_or_json_proposal(fenced_json)
    assert p2 is not None
    assert not p2["module"].lstrip().startswith("```")
    assert payload_gate.check_payload(p2["module"]) is None
    assert payload_gate.check_test(p2["test"]) is None


def test_r1_ast_contract_extractor_catches_backlog_list_to_dict_and_return_none_drift():
    backlog_existing = (REPO_ROOT / "bench_tasks" / "backlog" / "existing.py").read_text(encoding="utf-8")
    contract = production_synth.extract_module_contract(backlog_existing)
    assert contract.classes["Backlog"].init_attr_types == {"stories": "list", "next_id": "int"}
    assert contract.classes["Backlog"].methods["add"].returns_non_none is True

    # Exact false-confidence candidate shape from 20261118T011242_backlog.py_fail.log
    false_conf_backlog = (
        "class Backlog:\n"
        "    def __init__(self):\n"
        "        self.stories = {}\n"
        "        self.next_id = 1\n"
        "    def add(self, text, source='manual'):\n"
        "        if text in self.stories:\n"
        "            return None\n"
        "        self.stories[text] = {'id': self.next_id, 'text': text, 'source': source, 'status': 'backlog'}\n"
        "        self.next_id += 1\n"
        "        return self.next_id - 1\n"
        "    def set_status(self, story_id, status):\n"
        "        return False\n"
        "    def open_stories(self):\n"
        "        return list(self.stories.values())\n"
    )
    violations = production_synth.verify_contract_preservation(contract, false_conf_backlog)
    assert any("self.stories" in v and "list to dict" in v for v in violations)
    assert any("Backlog.add" in v and "return None" in v for v in violations)

    # Bounded regex cap detection on bench_tasks/gaps/existing.py
    gaps_existing = (REPO_ROOT / "bench_tasks" / "gaps" / "existing.py").read_text(encoding="utf-8")
    gaps_contract = production_synth.extract_module_contract(gaps_existing)
    assert ".{0,160}" in gaps_contract.bounded_regex_caps


def test_r2_r3_closed_loop_defeats_backlog_false_confidence_and_repairs_on_attempt_2(tmp_path):
    """Attempt 1 reproduces the exact self-confirming `backlog` false-confidence candidate;
    the R1-R3 pipeline rejects it at L5, feeds back contract diagnostics, and verifies Attempt 2
    against `bench_tasks/backlog/acceptance.py`."""
    backlog_existing = (REPO_ROOT / "bench_tasks" / "backlog" / "existing.py").read_text(encoding="utf-8")
    backlog_acceptance = (REPO_ROOT / "bench_tasks" / "backlog" / "acceptance.py").read_text(encoding="utf-8")
    backlog_task = json.loads((REPO_ROOT / "bench_tasks" / "backlog" / "task.json").read_text(encoding="utf-8"))

    attempt_1_bad = (
        "=== FILE: backlog.py ===\n"
        "class Backlog:\n"
        "    def __init__(self):\n"
        "        self.stories = {}\n"
        "        self.next_id = 1\n"
        "    def add(self, text, source='manual'):\n"
        "        if text in self.stories:\n"
        "            return None\n"
        "        sid = self.next_id\n"
        "        self.stories[text] = {'id': sid, 'text': text, 'source': source, 'status': 'backlog'}\n"
        "        self.next_id += 1\n"
        "        return sid\n"
        "    def set_status(self, story_id, status):\n"
        "        for s in self.stories.values():\n"
        "            if s['id'] == story_id:\n"
        "                s['status'] = status\n"
        "                return True\n"
        "        return False\n"
        "    def open_stories(self):\n"
        "        return [s for s in self.stories.values() if s['status'] in ('backlog', 'active')]\n"
        "=== TEST: test_proposal.py ===\n"
        "import backlog\n"
        "def test_proposal():\n"
        "    b = backlog.Backlog()\n"
        "    assert b.add('Fix parser') == 1\n"
        "    assert b.add('Fix parser') is None\n"
        "=== END ===\n"
    )

    attempt_2_good = (
        "=== FILE: backlog.py ===\n"
        "class Backlog:\n"
        "    def __init__(self):\n"
        "        self.stories = []\n"
        "        self.next_id = 1\n\n"
        "    def add(self, text, source='manual'):\n"
        "        norm = ' '.join(str(text).split()).lower()\n"
        "        for s in self.stories:\n"
        "            if s['status'] in ('backlog', 'active') and ' '.join(s['text'].split()).lower() == norm:\n"
        "                return s['id']\n"
        "        story = {'id': self.next_id, 'text': text, 'source': source, 'status': 'backlog'}\n"
        "        self.stories.append(story)\n"
        "        self.next_id += 1\n"
        "        return story['id']\n\n"
        "    def set_status(self, story_id, status):\n"
        "        for s in self.stories:\n"
        "            if s['id'] == story_id:\n"
        "                s['status'] = status\n"
        "                return True\n"
        "        return False\n\n"
        "    def open_stories(self):\n"
        "        return [s for s in self.stories if s['status'] in ('backlog', 'active')]\n"
        "=== TEST: test_proposal.py ===\n"
        "import backlog\n"
        "def test_proposal():\n"
        "    b = backlog.Backlog()\n"
        "    first = b.add('Fix parser')\n"
        "    assert b.add('Fix parser') == first\n"
        "    assert isinstance(b.stories, list) and len(b.stories) == 1\n"
        "=== END ===\n"
    )

    prompts_seen = []
    replies = iter([attempt_1_bad, attempt_2_good])

    def fake_propose(prompt: str) -> str:
        prompts_seen.append(prompt)
        return next(replies)

    outcome = production_synth.run_production_synthesis(
        root=str(tmp_path),
        target_rel_path="backlog.py",
        story_text=backlog_task.get("story") or backlog_task.get("prompt") or "Avoid duplicate open stories in Backlog",
        propose_raw_fn=fake_propose,
        sandbox_test_fn=otc._run_sandboxed_test_inner,
        existing_source=backlog_existing,
        k=3,
    )
    assert outcome.winner is not None
    assert len(outcome.attempts) == 2
    assert outcome.attempts[0]["ok"] is False
    assert "contract drift" in outcome.attempts[0]["reason"]
    assert "PREVIOUS CANDIDATE FAILED DETERMINISTIC L5 VERIFICATION" in prompts_seen[1]
    assert outcome.attempts[1]["ok"] is True
    assert outcome.receipt is not None and outcome.receipt["candidate_digest"].startswith("sha256:")

    # Score winner against the untouched hidden acceptance test
    hidden_ok, hidden_out = otc._run_sandboxed_test_inner("backlog.py", outcome.winner["module"], backlog_acceptance)
    assert hidden_ok is True, hidden_out


def test_r2_r3_closed_loop_defeats_gaps_160_char_truncation_with_independent_critic(tmp_path):
    """Attempt 1 reproduces the `.{1,160}` 160-char truncation bug from `20261118T011320_gaps.py_fail.log`;
    the isolated L5 Critic (which never sees Attempt 1) falsifies it with a >250-char input, and Attempt 2 passes
    both L5 Critic and `bench_tasks/gaps/acceptance.py`."""
    gaps_existing = (REPO_ROOT / "bench_tasks" / "gaps" / "existing.py").read_text(encoding="utf-8")
    gaps_acceptance = (REPO_ROOT / "bench_tasks" / "gaps" / "acceptance.py").read_text(encoding="utf-8")

    attempt_1_truncated = (
        "=== FILE: gaps.py ===\n"
        "import re\n"
        "_GAP_RE = re.compile(r'capability gap[:\\s]*(.{1,160})', re.IGNORECASE)\n"
        "def extract_gaps(prompts):\n"
        "    found = []\n"
        "    for text in prompts:\n"
        "        m = _GAP_RE.search(text)\n"
        "        if m:\n"
        "            found.append(m.group(1).strip().rstrip('.'))\n"
        "    return found\n"
        "=== TEST: test_proposal.py ===\n"
        "import gaps\n"
        "def test_proposal():\n"
        "    assert gaps.extract_gaps(['capability gap: short item.']) == ['short item']\n"
        "=== END ===\n"
    )

    attempt_2_fixed = (
        "=== FILE: gaps.py ===\n"
        "import re\n"
        "_HEAD_RE = re.compile(r'capability gap[:\\s]+([^\\n]+)', re.IGNORECASE)\n"
        "_SENT_END_RE = re.compile(r'\\.(?!\\d)(?:\\s|$)')\n"
        "def extract_gaps(prompts):\n"
        "    found = []\n"
        "    for text in prompts:\n"
        "        m = _HEAD_RE.search(text)\n"
        "        if not m:\n"
        "            continue\n"
        "        span = m.group(1).strip()\n"
        "        end = _SENT_END_RE.search(span)\n"
        "        gap = (span[:end.start()] if end else span).strip()\n"
        "        if gap:\n"
        "            found.append(gap)\n"
        "    return found\n"
        "=== TEST: test_proposal.py ===\n"
        "import gaps\n"
        "def test_proposal():\n"
        "    long_gap = 'x' * 260\n"
        "    assert gaps.extract_gaps([f'capability gap: {long_gap}. Tail.']) == [long_gap]\n"
        "=== END ===\n"
    )

    critic_prompts = []

    def isolated_critic_model(c_prompt: str) -> str:
        critic_prompts.append(c_prompt)
        return (
            "def _osiris_critic_invariants():\n"
            "    import gaps\n"
            "    long_desc = 'governed verification pipeline ' * 9 + 'with receipts'\n"
            "    assert len(long_desc) > 250\n"
            "    assert gaps.extract_gaps([f'capability gap: {long_desc}. Next sentence.']) == [long_desc]\n"
            "    assert gaps.extract_gaps(['Capability gap: track 0.97 fidelity. Done.']) == ['track 0.97 fidelity']\n"
        )

    replies = iter([attempt_1_truncated, attempt_2_fixed])
    outcome = production_synth.run_production_synthesis(
        root=str(tmp_path),
        target_rel_path="gaps.py",
        story_text="Extract full capability gap sentence without 160-char truncation or splitting decimals like 0.97",
        propose_raw_fn=lambda _p: next(replies),
        sandbox_test_fn=otc._run_sandboxed_test_inner,
        existing_source=gaps_existing,
        critic_model_fn=isolated_critic_model,
        k=3,
    )
    # Critic never saw Attempt 1 or Attempt 2 source code
    assert len(critic_prompts) == 1
    assert ".{1,160}" not in critic_prompts[0] and "_SENT_END_RE" not in critic_prompts[0]
    assert outcome.attempts[0]["ok"] is False
    assert "AssertionError" in outcome.attempts[0]["reason"]
    assert outcome.winner is not None and outcome.attempts[1]["ok"] is True

    hidden_ok, hidden_out = otc._run_sandboxed_test_inner("gaps.py", outcome.winner["module"], gaps_acceptance)
    assert hidden_ok is True, hidden_out


def test_r3_duplicate_candidate_hash_rejected_and_repo_pytest_overlay_enforced(tmp_path):
    ws = tmp_path / "repo"
    (ws / "tests").mkdir(parents=True)
    (ws / "adder.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (ws / "tests" / "test_adder.py").write_text(
        "import adder\n\ndef test_existing_suite():\n    assert adder.add(2, 3) == 5\n    assert adder.add(-1, 1) == 0\n",
        encoding="utf-8",
    )

    # Candidate 1 & 2 are byte-identical and break negative numbers (own test only checks 2+3);
    # Candidate 3 fixes add() and adds mul()
    bad_cand = (
        "=== FILE: adder.py ===\n"
        "def add(a, b):\n"
        "    return abs(a) + abs(b)\n"
        "def mul(a, b):\n"
        "    return a * b\n"
        "=== TEST: test_proposal.py ===\n"
        "import adder\n"
        "def test_proposal():\n"
        "    assert adder.add(2, 3) == 5\n"
        "    assert adder.mul(2, 4) == 8\n"
        "=== END ===\n"
    )
    good_cand = (
        "=== FILE: adder.py ===\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "def mul(a, b):\n"
        "    return a * b\n"
        "=== TEST: test_proposal.py ===\n"
        "import adder\n"
        "def test_proposal():\n"
        "    assert adder.add(-2, 3) == 1\n"
        "    assert adder.mul(3, 4) == 12\n"
        "=== END ===\n"
    )

    replies = iter([bad_cand, bad_cand, good_cand])
    outcome = production_synth.run_production_synthesis(
        root=str(ws),
        target_rel_path="adder.py",
        story_text="Add mul(a, b) to adder.py while keeping add(a, b) working",
        propose_raw_fn=lambda _p: next(replies),
        sandbox_test_fn=otc._run_sandboxed_test_inner,
        existing_source=(ws / "adder.py").read_text(encoding="utf-8"),
        k=3,
        run_repo_pytest=True,
    )
    assert outcome.winner is not None
    assert "existing repository pytest failed" in outcome.attempts[0]["reason"]
    assert "DUPLICATE_CANDIDATE" in outcome.attempts[1]["reason"]
    assert outcome.attempts[2]["ok"] is True
    assert outcome.receipt["repo_pytest_targets"] == ["tests/test_adder.py"]
    # Real workspace file was NEVER mutated prior to /apply
    assert "mul" not in (ws / "adder.py").read_text(encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# ITERATION R4: GOVERNANCE CLOSURE, MENTOR ROUTING & FROZEN RUNNER INVARIANTS
# ═══════════════════════════════════════════════════════════════════════════════


def test_r4_run_synergy_pipeline_never_calls_unapproved_organism_learn_on_rejected_or_unapplied_code(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(otc, "PROMPT_LIBRARY_PATH", str(tmp_path / "prompt_lib.json"))
    monkeypatch.setattr(
        otc,
        "_pending_write",
        {"path": None, "content": None, "existed": None, "raw_path": None, "story_meta": None},
    )
    learned = []
    monkeypatch.setattr(otc, "_spawn_organism_learn", lambda payload: learned.append(payload))

    # 1. Empty Stage 1 output
    monkeypatch.setattr(otc, "query_model", lambda *a, **k: "")
    assert otc.run_synergy_pipeline("build a parser") is False
    assert learned == []

    # 2. Placeholder/stub output rejected
    monkeypatch.setattr(otc, "query_model", lambda *a, **k: "Implement compute_score(xs) returning mean.")
    monkeypatch.setattr(otc, "_synthesize", lambda *a, **k: "```python\ndef compute_score(xs):\n    pass\n```")
    assert otc.run_synergy_pipeline("build compute_score") is False
    assert learned == []

    # 3. Valid verified candidate staged for /apply (still unconfirmed by human -> zero unapproved training!)
    valid_delimited = (
        "=== FILE: score_mod.py ===\n"
        "def compute_score(xs: list[float]) -> float:\n"
        "    if not xs:\n"
        "        raise ValueError('empty')\n"
        "    return sum(xs) / len(xs)\n"
        "=== TEST: test_proposal.py ===\n"
        "import score_mod\n"
        "def test_proposal():\n"
        "    assert score_mod.compute_score([2.0, 4.0]) == 3.0\n"
        "=== END ===\n"
    )
    monkeypatch.setattr(otc, "_synthesize", lambda *a, **k: valid_delimited)
    assert otc.run_synergy_pipeline("build compute_score") is True
    assert otc._pending_write["path"] is not None
    assert learned == [], "Unapplied synthesis candidates must NEVER train Engine 3 before human approval"


def test_r4_protege_mentor_routing_prefers_verified_coder_and_demotes_zero_pass_model(tmp_path, monkeypatch):
    monkeypatch.delenv("OSIRIS_CHAT_MODEL", raising=False)
    monkeypatch.delenv("OSIRIS_CODE_MODEL", raising=False)

    suite = protege.current_suite()
    runner = protege.current_runner()
    prev = protege.GENESIS
    bodies = [
        {
            "kind": "bench_run_start",
            "run_id": "r-gemma",
            "suite_sha256": suite,
            "runner_sha256": runner,
            "backend": "ollama",
            "model": "gemma3:4b",
        },
        *[
            {
                "kind": "bench_task",
                "run_id": "r-gemma",
                "task": t,
                "result": {"attempts": [{"n": 1, "own_ok": False, "hidden_ok": False, "module_sha256": None, "hidden_reason": "malformed"}]},
            }
            for t in ("backlog", "claims", "gaps")
        ],
        {
            "kind": "bench_run_start",
            "run_id": "r-qwen",
            "suite_sha256": suite,
            "runner_sha256": runner,
            "backend": "ollama",
            "model": "qwen2.5-coder:7b",
        },
        {
            "kind": "bench_task",
            "run_id": "r-qwen",
            "task": "nclm_eval",
            "result": {"attempts": [{"n": 1, "own_ok": True, "hidden_ok": True, "module_sha256": "a" * 64, "hidden_reason": "ok"}]},
        },
    ]
    ledger_file = tmp_path / "results.ledger.jsonl"
    rows = []
    for b in bodies:
        h = protege.hashlib.sha256((prev + protege._canon(b)).encode()).hexdigest()
        rows.append(json.dumps({**b, "prev": prev, "hash": h}))
        prev = h
    ledger_file.write_text("\n".join(rows) + "\n", encoding="utf-8")

    saved_chat, saved_code = otc.CHAT_MODEL, otc.CODE_MODEL
    try:
        otc._resolve_active_models(
            installed=["gemma3:4b", "qwen2.5-coder:7b"],
            bench_dir=str(tmp_path),
        )
        assert otc.CODE_MODEL == "qwen2.5-coder:7b"
        assert otc.CHAT_MODEL == "qwen2.5-coder:7b"
        assert otc.SYNTHESIS_MAX_TOKENS >= 4096
    finally:
        otc.CHAT_MODEL, otc.CODE_MODEL = saved_chat, saved_code


def test_r4_frozen_runner_and_invariant_first_hashes_unchanged_and_contract_critic_registered():
    assert osiris_bench.runner_sha256() == "e48927f381297e3d05f7f0e9c42226548c905998390939293bdf45fb720d8f96"
    assert osiris_bench.runner_sha256() == protege.current_runner()
    assert "invariant-first-v1" in prompt_strategies.STRATEGIES
    assert "contract-critic-v1" in prompt_strategies.STRATEGIES
    applied = prompt_strategies.apply("contract-critic-v1", "TASK PROMPT")
    assert applied.endswith("TASK PROMPT") and "self.stories" in applied


def test_r3_sprint_execute_runs_verified_first_passing_and_records_run_artifact(tmp_path, monkeypatch):
    import genome_ledger
    import run_record
    import sprint_manager

    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.chdir(ws)
    monkeypatch.setattr(sprint_manager, "SPRINT_BACKLOG_PATH", str(tmp_path / "sprint_backlog.json"))
    monkeypatch.setattr(run_record, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(genome_ledger, "DEFAULT_PATH", str(tmp_path / "genome_ledger.jsonl"))
    monkeypatch.setattr(otc, "SANDBOX_LOG_DIR", str(tmp_path / "sandbox_logs"))
    monkeypatch.setattr(otc, "APPLY_BLOCK_FILE", str(tmp_path / "APPLY_BLOCKED"))
    monkeypatch.setattr(otc, "_apply_blocked", {"reason": None})
    monkeypatch.setattr(
        otc,
        "_pending_write",
        {"path": None, "content": None, "existed": None, "raw_path": None, "story_meta": None},
    )

    story = sprint_manager.add_story("Implement mean_abs(values) rejecting empty lists", 1, "manual")
    sprint_manager.set_status(story["id"], "active")

    # Attempt 1 is stub -> rejected by ProductionConfinementGuard; Attempt 2 passes L5 + sandbox
    stub_attempt = (
        f"=== FILE: osiris_story_{story['id']}.py ===\n"
        "def mean_abs(values):\n"
        "    pass\n"
        "=== TEST: test_proposal.py ===\n"
        f"import osiris_story_{story['id']} as m\n"
        "def test_proposal():\n"
        "    assert True\n"
        "=== END ===\n"
    )
    good_attempt = (
        f"=== FILE: osiris_story_{story['id']}.py ===\n"
        "def mean_abs(values):\n"
        "    if not values:\n"
        "        raise ValueError('values must not be empty')\n"
        "    return sum(abs(x) for x in values) / len(values)\n"
        "=== TEST: test_proposal.py ===\n"
        f"import osiris_story_{story['id']} as m\n"
        "def test_proposal():\n"
        "    assert m.mean_abs([-2.0, 4.0]) == 3.0\n"
        "=== END ===\n"
    )
    synth_replies = iter([stub_attempt, good_attempt])

    def fake_synth(prompt, system_prompt="", structured=False, only=None, ollama_model=None):
        otc._last_backend.update(name="ollama", model="qwen2.5-coder:7b", error=None)
        return next(synth_replies)

    monkeypatch.setattr(otc, "_synthesize", fake_synth)
    otc._sprint_execute(f"/sprint execute #{story['id']} --k 3")

    assert otc._pending_write["path"] == str(ws / f"osiris_story_{story['id']}.py")
    assert not (ws / f"osiris_story_{story['id']}.py").exists(), "Must remain staged until /apply"
    led = genome_ledger.GenomeLedger()
    assert led.problem() is None and len(led.chain) == 1
    assert led.chain[0]["kind"] == "sprint_run"

