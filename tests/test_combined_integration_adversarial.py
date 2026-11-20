"""Cross-branch integration and adversarial verification tests for osiris/candidate/v6-combined-integration.

These tests are newly created specifically by the combined integration (counted separately
from origin/main tests and candidate-branch tests). They validate cross-branch interactions
that no single candidate branch tested on its own:

1. Priority 1: ide_api.py HOME-path disclosure remediation on PR #47 task_run.ledger.
2. 3-way CodeAgent integration: PR #47 TaskRunContext budgets + grant-hardening TOCTOU/symlink
   defense + capability-reconciliation-s1 exec_kernel receipted execution.
3. 2-way code_task integration: PR #47 task_run governance + grant-hardening F-01/F-02/F-03.
4. Reconciled probes.py: O(1) verify_chain_tail_report CRLF rejection + orphan detection +
   ChainTailReport dict compatibility + SafePathGuard + substrate aliases.
5. Reconciled console/paths/preflight: PR #45 ledger order + WP2 paths._resolve_* +
   bare /sprint + read-only /runs --orphans + unified _mentor_model_guard + canonical
   gemma3:1b defaults + RTX 5050 Node Beta harness + preserved suite_sha256 & runner_sha256.
6. Whole-candidate static governance gate: Python 3.9 AST parseability, UTF-8 LF hygiene,
   and SYS-025 payload_gate zero-stub verification across every new/modified runtime module.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from osiris_cli import code_agent, code_task, ide_api, osiris_repl, paths, probes, task_context
from osiris_cli.code_runs import RunStore
from osiris_cli.pqc_enclave import GrantVerificationError, PQCEnclaveGate, setup_local_keys

otc = osiris_repl.otc


def _scripted(*steps: Dict[str, Any]):
    queue = list(steps) + [{"content": "done"}]
    return lambda m, t, model: queue.pop(0) if queue else {"content": "done"}


def _tool_call(name: str, **args: Any) -> Dict[str, Any]:
    return {"content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


# ── 1. Priority 1: ide_api.py HOME-path disclosure fix ──────────────────────

def test_priority1_ide_api_sanitizes_task_run_ledger_home_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "isolated_home"
    (home / ".osiris" / "code").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))

    repo = home / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (repo / "test_calc.py").write_text("from calc import add\ndef test_add():\n    assert add(2, 3) == 5\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)

    runs = home / ".osiris" / "code" / "runs"
    receipts = home / ".osiris" / "code" / "receipts.jsonl"
    store = RunStore(str(runs))
    setup_local_keys(str(runs / "_enclave"), confirm=lambda _p: True)

    fixed = "def add(a, b):\n    return a + b\n"
    accept = f"{sys.executable} -m pytest -q -p no:cacheprovider test_calc.py"
    summary = code_task.run_task(
        "fix add in calc.py",
        root=str(repo),
        chat=_scripted(_tool_call("write_file", path="calc.py", content=fixed)),
        out=lambda _s: None,
        store=store,
        receipts=str(receipts),
        confirm=lambda _p: False,
        accept=accept,
        approve_writes=True,
    )

    backend = ide_api.Backend(
        gateway_factory=lambda: None,
        slice_receipts=str(tmp_path / "slice.jsonl"),
        examples_dir="",
        code_runs_dir=str(runs),
        code_receipts=str(receipts),
    )
    detail = backend.code_run({"run_id": summary["run_id"]})
    blob = json.dumps(detail)

    assert str(home) not in blob
    assert detail["root"] == "~/repo"
    assert detail["task_run"]["ledger"].startswith("~/.osiris/code/runs/_tasks/task_runs.jsonl")


# ── 2. 3-way CodeAgent integration (PR #47 + grant-hardening + exec_kernel) ─

def test_three_way_code_agent_budget_grant_toctou_and_exec_kernel(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    script = repo / "check.py"
    script.write_text("print('CLEAN_OK')\n", encoding="utf-8")
    (repo / "notes.txt").write_text("alpha needle beta\n", encoding="utf-8")

    enclave_dir = tmp_path / "enclave"
    enclave = PQCEnclaveGate.load_or_init_local(str(enclave_dir), allow_init=True)
    exec_journal_path = tmp_path / "exec_journal.jsonl"
    exec_artifacts = tmp_path / "exec_artifacts"
    from osiris_cli.exec_kernel import ExecJournal

    # 2a. Grep + approved command route through exec_kernel (emitting exec_receipt_id) under TaskRunContext
    ctx = task_context.new_context(
        "grep and run check.py",
        kind="code",
        run_id="code-3way00000001",
        perms=task_context.permissions(propose_writes=True, run_commands=True),
    )
    agent = code_agent.CodeAgent(
        root=str(repo),
        chat=_scripted(
            _tool_call("grep_code", pattern="needle", path="notes.txt"),
            _tool_call("run_command", command=f"{sys.executable} check.py"),
        ),
        confirm=lambda _p: True,
        out=lambda _s: None,
        enclave=enclave,
        context=ctx,
        task_ledger=task_context.TaskLedger(str(tmp_path / "tasks.jsonl")),
        exec_journal=ExecJournal(exec_journal_path),
        exec_artifacts_dir=str(exec_artifacts),
    )
    res = agent.run("grep and run check.py", run_id=ctx.run_id)
    assert res.status == "SUCCESS"
    grep_dec = next(d for d in res.decisions if d["tool"] == "grep_code")
    cmd_dec = next(d for d in res.decisions if d["tool"] == "run_command")
    assert grep_dec["decision"] == "GRANTED" and grep_dec.get("exec_receipt_id")
    assert cmd_dec["decision"] == "GRANTED" and cmd_dec.get("exec_receipt_id") and cmd_dec.get("exit_code") == 0

    # 2b. Adversarial script TOCTOU after approval fails closed BEFORE exec_kernel executes
    def tamper_during_confirm(_prompt: str) -> bool:
        script.write_text("print('TAMPERED_PAYLOAD')\n", encoding="utf-8")
        return True

    ctx2 = task_context.new_context(
        "run tampered check.py",
        kind="code",
        run_id="code-3way00000002",
        perms=task_context.permissions(propose_writes=False, run_commands=True),
    )
    agent2 = code_agent.CodeAgent(
        root=str(repo),
        chat=_scripted(_tool_call("run_command", command=f"{sys.executable} check.py")),
        confirm=tamper_during_confirm,
        out=lambda _s: None,
        enclave=enclave,
        context=ctx2,
        task_ledger=task_context.TaskLedger(str(tmp_path / "tasks2.jsonl")),
        exec_journal=ExecJournal(tmp_path / "exec_journal2.jsonl"),
        exec_artifacts_dir=str(tmp_path / "exec_artifacts2"),
    )
    res2 = agent2.run("run tampered check.py", run_id=ctx2.run_id)
    cmd_dec2 = next(d for d in res2.decisions if d["tool"] == "run_command")
    assert cmd_dec2["decision"] == "DENIED"
    assert "verification failed" in cmd_dec2["reason"].lower()


# ── 3. 2-way code_task integration (PR #47 + grant-hardening) ───────────────

def test_code_task_combines_task_context_and_grant_hardening_trust_pinning(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("X = 1\n", encoding="utf-8")

    store = RunStore(str(tmp_path / "runs"))
    enclave_dir = tmp_path / "runs" / "_enclave"
    setup_local_keys(str(enclave_dir), confirm=lambda _p: True)

    attacker_dir = tmp_path / "attacker_enclave"
    attacker = PQCEnclaveGate.load_or_init_local(str(attacker_dir), allow_init=True)
    new_content = "X = 2\n"
    after_sha = hashlib.sha256(new_content.encode("utf-8")).hexdigest()
    before_sha = hashlib.sha256(b"X = 1\n").hexdigest()
    prop = {
        "rel_path": "app.py",
        "abs_path": str(repo / "app.py"),
        "content": new_content,
        "existed_at_proposal": True,
        "before_sha256": before_sha,
        "after_sha256": after_sha,
        "diff": "",
    }
    psha, scope, schema, _ = code_task.proposals_grant_spec(str(repo), [prop])
    forged_grant = attacker.generate_hybrid_grant(
        sequence_index=1,
        prev_block_hash="0" * 64,
        event_type="code.apply_writes",
        payload_sha256=psha,
        operation_schema=schema,
        scope=scope,
    )
    run = store.create("code-2way00000001", task="update X", root=str(repo), options={
        "enclave_dir": str(attacker_dir),
        "trusted_keys_path": str(attacker_dir / "trusted_operator_keys.json"),
    })
    run["proposals"] = [prop]
    run["approval_grant"] = forged_grant
    store.save(run)

    with pytest.raises(GrantVerificationError):
        code_task.apply_idempotent(run, store)
    assert (repo / "app.py").read_text(encoding="utf-8") == "X = 1\n"


# ── 4. Reconciled probes.py (v6 + capability-reconciliation-s1) ─────────────

def test_reconciled_probes_crlf_fail_closed_orphans_and_compat(tmp_path: Path) -> None:
    osiris_home = tmp_path / ".osiris"
    living_dir = osiris_home / "living"
    runs_dir = osiris_home / "runs"
    living_dir.mkdir(parents=True)
    runs_dir.mkdir(parents=True)

    (runs_dir / "run_done.json").write_text(json.dumps({"id": "ok", "status": "completed"}), encoding="utf-8")
    (runs_dir / "run_orphan.json").write_text(json.dumps({"id": "o1", "status": "running"}), encoding="utf-8")

    # Build valid living-style prev/hash ledger with LF
    prev = "0" * 64
    lines: List[str] = []
    for i in range(4):
        row = {"user": f"q{i}", "reply": f"a{i}", "prev": prev}
        h = hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        row["hash"] = h
        prev = h
        lines.append(json.dumps(row, sort_keys=True))

    lf_path = living_dir / "exchanges.jsonl"
    lf_path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    rep = probes.verify_chain_tail_report(lf_path, tail_lines=10, osiris_home=osiris_home)
    assert rep.intact is True
    assert rep.crlf_detected is False
    assert rep.orphan_runs == 1
    assert rep.tail_head == prev[:12]
    assert rep["intact"] is True and rep["tail_count"] == 4 and rep["head"] == prev[:12]

    # CRLF byte injection must fail closed with crlf_detected=True
    crlf_path = living_dir / "exchanges_crlf.jsonl"
    crlf_path.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))
    rep_crlf = probes.verify_chain_tail_report(crlf_path, tail_lines=10, osiris_home=osiris_home)
    assert rep_crlf.intact is False
    assert rep_crlf.crlf_detected is True
    assert rep_crlf["status"] == "TAMPERED"

    # SubstrateInfo aliases + power_supply_dir kwarg compatibility
    sub = probes.probe_substrate(power_supply_dir=tmp_path / "missing_power")
    assert sub.power_status == "unreadable_attended_only"
    assert sub.node_role == sub.node_id
    assert sub.horizon_s == float(sub.horizon_sec)


# ── 5. Reconciled WP2/PR45/v6 paths, console, RTX 5050 harness & pins ───────

def test_reconciled_console_wp2_paths_sprint_orphans_guard_and_bench_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # 5a. paths._resolve_osiris_home and _resolve_dnalang_ledger (Node Alpha layout)
    home = tmp_path / "home"
    alpha_ledger = home / "osiris-repos" / "dnalang-core" / "dnalang" / "ledger.py"
    alpha_ledger.parent.mkdir(parents=True)
    alpha_ledger.write_text("# ledger\n", encoding="utf-8")
    assert paths._resolve_osiris_home(environ={"OSIRIS_HOME": str(tmp_path / "oh")}) == tmp_path / "oh"
    assert paths._resolve_dnalang_ledger(tmp_path / "oh", repo_root=tmp_path / "repo", environ={}, home=home) == alpha_ledger

    # 5b. Canonical gemma3:1b defaults & zero legacy gemma-3-1b literal in console source
    assert otc is not None
    src = inspect.getsource(otc)
    assert 'os.environ.get("OSIRIS_CHAT_MODEL", "gemma3:1b")' in src
    assert 'os.environ.get("OSIRIS_CODE_MODEL", "gemma3:1b")' in src
    assert '"gemma-3-1b"' not in src
    # Legacy alias still normalizes dynamically via providers.router
    assert otc.canonicalize_ollama_tag("gemma-3-1b") == "gemma3:1b"

    # 5c. Bare /sprint dispatches to usage; /runs --orphans routes cleanly in console & REPL
    called: List[str] = []
    monkeypatch.setattr(otc, "_sprint_usage", lambda: called.append("sprint_usage"))
    monkeypatch.setattr(otc, "_runs_orphans", lambda: called.append("runs_orphans"))
    assert otc.run_command("/sprint") is True
    assert otc.run_command("/runs --orphans") is True
    osiris_repl.dispatch_command(osiris_repl.SESSION, "/runs --orphans")
    assert called == ["sprint_usage", "runs_orphans", "runs_orphans"]
    assert "/runs --orphans" in otc.READ_ONLY_COMMANDS

    # 5d. Unified _mentor_model_guard delegates to _bench_blocker (checks daemon, model, and recorder)
    monkeypatch.setattr(otc, "_ollama_models", lambda: None)
    argv = ["osiris_bench.py", "--events", "--backend", "ollama", "--model", "qwen2.5-coder:7b", "--k", "1"]
    refusal = otc._mentor_model_guard(argv, otc.osiris_ui.Canvas())
    assert refusal is not None and "BACKEND UNREACHABLE" in refusal

    # 5e. RTX 5050 harness refuses non-Beta substrate
    spec = importlib.util.spec_from_file_location("bench_rtx5050", ROOT / "scripts" / "bench_rtx5050_multimentor.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "is_node_beta", lambda: (False, "aarch64"))
    assert mod.preflight() == 2
    assert "Refusing" in capsys.readouterr().out

    # 5f. Both benchmark comparability hashes (runner_sha256 AND suite_sha256) preserved intact
    import osiris_bench
    assert osiris_bench.runner_sha256() == "af49b2bd7707c36ff94cc0f87d307846294afefb5a35cb808f8524d41e53264c"
    assert osiris_bench.suite_sha256().startswith("09037df63614")


# ── 6. Whole-candidate Python 3.9 AST, UTF-8 LF & SYS-025 zero-stub gate ────

def test_combined_candidate_modules_pass_py39_ast_lf_and_sys025_gates() -> None:
    import payload_gate

    modules = [
        "bench_trial.py",
        "genome_ledger.py",
        "osiris_shell.py",
        "osiris_termux_console.py",
        "publish_v600_to_zenodo.py",
        "scripts/bootstrap_osiris_agents.py",
        "scripts/bench_rtx5050_multimentor.py",
        "osiris_cli/architecture.py",
        "osiris_cli/code_agent.py",
        "osiris_cli/code_task.py",
        "osiris_cli/exec_kernel.py",
        "osiris_cli/ide_api.py",
        "osiris_cli/knowledge.py",
        "osiris_cli/living.py",
        "osiris_cli/orchestration/evidence_firewall.py",
        "osiris_cli/orchestration/evolution_bridge.py",
        "osiris_cli/orchestration/trio.py",
        "osiris_cli/osiris_repl.py",
        "osiris_cli/paths.py",
        "osiris_cli/pqc_enclave.py",
        "osiris_cli/probes.py",
        "osiris_cli/providers/__init__.py",
        "osiris_cli/providers/router.py",
        "osiris_cli/session_approval.py",
        "osiris_cli/skills_runtime.py",
        "osiris_cli/train.py",
        "osiris_cli/tui_inline.py",
    ]
    for rel in modules:
        path = ROOT / rel
        raw = path.read_bytes()
        assert b"\r" not in raw, f"CRLF detected in {rel}"
        src = raw.decode("utf-8")
        ast.parse(src, filename=rel, feature_version=(3, 9))
        if not rel.endswith("__init__.py"):
            stub_err = payload_gate.check_payload(src)
            assert stub_err is None, f"SYS-025 payload_gate rejected {rel}: {stub_err}"
