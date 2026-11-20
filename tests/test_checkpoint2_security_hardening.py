"""Checkpoint 2 exhaustive adversarial & regression tests covering CRIT-01..05,
HIGH-01..06, MED-01..06, V6-01..05, and T-01..T-13.

All tests run strictly offline against disposable temporary workspaces and enclaves.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

import osiris_shell
from osiris_cli import code_agent, code_task, exec_kernel, exec_policy, probes, skills_runtime, tui_inline
from osiris_cli.code_agent import CodeAgent, Denied, run_acceptance, verify_applied
from osiris_cli.code_runs import RunStore
from osiris_cli.exec_kernel import CommandRequest, ExecJournal, classify_json_payload
from osiris_cli.living import Osiris
from osiris_cli.orchestration.evidence_firewall import ExecutionEvidence
from osiris_cli.pqc_enclave import PQCEnclaveGate


def _scripted(*replies):
    it = iter(replies)

    def chat(messages, tools, model):
        try:
            return next(it)
        except StopIteration:
            return {"content": "done", "tool_calls": []}

    return chat


def _call(name: str, **arguments: Any) -> Dict[str, Any]:
    return {"content": "", "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}


# ── T-01 / HIGH-01 / MED-02: env scrubbing & journal non-diversion ───────────

def test_t01_med02_scrubbed_env_drops_auth_pat_and_exec_overrides() -> None:
    parent = {
        "PATH": "/usr/bin:/bin",
        "OSIRIS_SAFE_FLAG": "1",
        "OSIRIS_AUTH_HEADER": "Bearer secret",
        "OSIRIS_GITHUB_PAT": "ghp_secret",
        "OSIRIS_EXEC_JOURNAL": "/tmp/evil.jsonl",
        "OSIRIS_EXEC_ARTIFACTS_DIR": "/tmp/evil_runs",
        "QISKIT_IBM_TOKEN": "tok",
    }
    env = exec_policy.scrubbed_env(parent)
    assert env["PATH"] == "/usr/bin:/bin"
    assert env["OSIRIS_SAFE_FLAG"] == "1"
    for forbidden in (
        "OSIRIS_AUTH_HEADER",
        "OSIRIS_GITHUB_PAT",
        "OSIRIS_EXEC_JOURNAL",
        "OSIRIS_EXEC_ARTIFACTS_DIR",
        "QISKIT_IBM_TOKEN",
    ):
        assert forbidden not in env, f"{forbidden} leaked into scrubbed_env"


# ── T-02 / CRIT-03: newline chaining & Bash redirect operators denied ────────

@pytest.mark.parametrize("cmd", [
    "git status\nrm -rf /",
    "ls\r\nid",
    "git status # comment\nrm -rf /",
    "ls >& /tmp/pwn",
    "ls >| /tmp/pwn",
    "ls &>> /tmp/pwn",
    "cat <<< hello",
    "ls 2>&1",
    "ls <& 0",
    "ls ;;& id",
    "ls &&& id",
    "echo $(id)",
    "echo `id`",
    "cat <(id)",
    "cat .env",
    'cat .en""v',
])
def test_t02_crit03_shell_smuggling_and_redirects_denied(tmp_path: Path, cmd: str) -> None:
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": cmd}, "cwd": str(tmp_path)})
    verdict = classify_json_payload(payload, origin="claude_code", cwd=str(tmp_path), root=str(tmp_path))
    assert verdict["decision"] == "DENY", f"Bypassed classifier for {cmd!r}: {verdict}"
    assert verdict["tier"] == "FORBIDDEN"


# ── T-03 / CRIT-04: authoritative CLI origin/root, tool_name, empty argv ─────

def test_t03_crit04_stdin_json_cannot_spoof_origin_or_root(tmp_path: Path) -> None:
    safe_root = tmp_path / "repo"
    outside = tmp_path / "outside"
    safe_root.mkdir()
    outside.mkdir()

    # Spoofing origin="human" must NOT downgrade claude_code DENY on network command.
    spoof_origin = json.dumps({
        "origin": "human",
        "tool_name": "Bash",
        "tool_input": {"command": "curl https://evil.example"},
        "cwd": str(safe_root),
    })
    v1 = classify_json_payload(spoof_origin, origin="claude_code", cwd=str(safe_root), root=str(safe_root))
    assert v1["origin"] == "claude_code"
    assert v1["decision"] == "DENY"

    # Spoofing root="/" must NOT escape safe_root confinement.
    spoof_root = json.dumps({
        "root": "/",
        "cwd": str(outside),
        "tool_name": "Bash",
        "tool_input": {"command": "ls"},
    })
    v2 = classify_json_payload(spoof_root, origin="claude_code", cwd=str(safe_root), root=str(safe_root))
    assert v2["decision"] == "DENY"
    assert v2["tier"] == "FORBIDDEN"

    # Non-Bash tool_name refused fail-closed.
    for tool in ("Read", "Write", "Edit"):
        v3 = classify_json_payload(
            json.dumps({"tool_name": tool, "tool_input": {"command": "ls"}, "cwd": str(safe_root)}),
            origin="claude_code",
            cwd=str(safe_root),
            root=str(safe_root),
        )
        assert v3["decision"] == "DENY"

    # Empty/blank argv[0] refused as FORBIDDEN / DENY.
    for bad_argv in ([""], ["   "], []):
        v4 = classify_json_payload(
            json.dumps({"argv": bad_argv, "cwd": str(safe_root)}),
            origin="claude_code",
            cwd=str(safe_root),
            root=str(safe_root),
        )
        assert v4["decision"] == "DENY"
        assert v4["tier"] == "FORBIDDEN"


# ── T-04 / CRIT-01: CodeAgent._run_command enforces origin='model' ───────────

@pytest.mark.parametrize("cmd", [
    "curl https://evil.example",
    "bash -c 'echo pwn'",
    f"{sys.executable} -c 'print(1)'",
    "timeout 5 rm -rf .",
    "git reset --hard",
    "git commit -m spoof",
    f"{sys.executable} publish_v600_to_zenodo.py",
    f"{sys.executable} -m pip install evilpkg",
    "find . -delete",
])
def test_t04_crit01_code_agent_run_command_enforces_model_origin(tmp_path: Path, cmd: str) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "publish_v600_to_zenodo.py").write_text("print('pub')\n", encoding="utf-8")
    journal = ExecJournal(tmp_path / "exec_journal.jsonl")
    prompts: List[str] = []
    agent = CodeAgent(
        root=str(ws),
        chat=_scripted(_call("run_command", command=cmd)),
        confirm=lambda p: (prompts.append(p) or True),
        out=lambda _s: None,
        exec_journal=journal,
        exec_artifacts_dir=tmp_path / "runs",
    )
    agent.run("try blocked command")
    assert prompts == [], f"Operator should never be prompted for agent-denied command {cmd!r}"
    assert agent.decisions[0]["tool"] == "run_command"
    assert agent.decisions[0]["decision"] == "DENIED"
    assert "unknown tool" not in agent.decisions[0].get("reason", "").lower()
    if journal.path.exists():
        rows = [json.loads(ln)["body"] for ln in journal.path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert all(r["origin"] == "model" for r in rows)


def test_t04_crit01_allowed_model_command_records_model_origin_in_journal(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "ok.py").write_text("print('MODEL_OK')\n", encoding="utf-8")
    journal = ExecJournal(tmp_path / "exec_journal.jsonl")
    prompts: List[str] = []
    agent = CodeAgent(
        root=str(ws),
        chat=_scripted(_call("run_command", command=f"{sys.executable} ok.py")),
        confirm=lambda p: (prompts.append(p) or True),
        out=lambda _s: None,
        exec_journal=journal,
        exec_artifacts_dir=tmp_path / "runs",
    )
    agent.run("run workspace script")
    assert agent.decisions[0]["decision"] == "GRANTED"
    assert prompts and "[MUTATE]" in prompts[0]
    rows = [json.loads(ln)["body"] for ln in journal.path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) == 1
    assert rows[0]["origin"] == "model"
    assert rows[0]["decision"] == "ASK_APPROVED"


# ── T-05 / CRIT-02: run_acceptance policy & tampered run["accept"] rejection ─

@pytest.mark.parametrize("cmd", [
    "curl https://evil.example",
    "bash -c 'echo pwn'",
    "timeout 5 rm -rf .",
    "git reset --hard",
    f"{sys.executable} publish_v600_to_zenodo.py",
    "printenv",
])
def test_t05_crit02_run_acceptance_raises_denied_and_verify_blocks(tmp_path: Path, cmd: str) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.py").write_text("X = 1\n", encoding="utf-8")
    journal = ExecJournal(tmp_path / "acc_journal.jsonl")
    with pytest.raises(Denied):
        run_acceptance(str(ws), cmd, 5.0, exec_journal=journal, exec_artifacts_dir=tmp_path / "runs")
    rec = verify_applied(
        str(ws),
        [{"path": "a.py", "after_sha256": hashlib.sha256(b"X = 1\n").hexdigest()}],
        cmd,
        require_change=True,
        timeout=5.0,
        exec_journal=journal,
        exec_artifacts_dir=tmp_path / "runs",
    )
    assert rec["decision"] == "BLOCKED"


def test_t05_crit02_tampered_persisted_accept_command_blocked_on_resume(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text("X = 1\n", encoding="utf-8")
    (repo / "check.py").write_text("print('LEGIT_CHECK')\n", encoding="utf-8")
    (repo / "evil.py").write_text("open('PWNED', 'w').write('1')\n", encoding="utf-8")
    from osiris_cli.pqc_enclave import setup_local_keys
    enc = tmp_path / "enclave"
    setup_local_keys(str(enc), confirm=lambda _p: True)
    store = RunStore(str(tmp_path / "runs"))

    summary = code_task.run_task(
        "update calc",
        root=str(repo),
        store=store,
        chat=_scripted(_call("write_file", path="calc.py", content="X = 2\n")),
        enclave_dir=str(enc),
        confirm=lambda _p: True,
        out=lambda _m: None,
        approve_writes=True,
        accept=f"{sys.executable} check.py",
        stop_after="AWAITING_APPROVAL",
    )
    run_id = summary["run_id"]
    # Approve & apply, then simulate tampering run["accept"] on disk before VERIFYING resumes.
    run = store.load(run_id)
    payload_sha, scope, schema, summary_txt = code_task.proposals_grant_spec(
        run["root"], run["proposals"], accept=run["accept"]
    )
    signer = PQCEnclaveGate.load_or_init_local(str(enc))
    grant = signer.request_approved_grant(
        action="code.apply_writes",
        scope=scope,
        summary=summary_txt,
        payload_sha256=payload_sha,
        operation_schema=schema,
        confirm=lambda _p: True,
        out=lambda _m: None,
    )
    run["approval_grant"] = grant
    store.save(run)
    code_task.apply_idempotent(run, store, enclave_dir=str(enc))
    store.transition(run, "VERIFYING", "applied")

    # Attacker tampers persisted run["accept"] on disk to point to evil.py!
    run = store.load(run_id)
    run["accept"] = f"{sys.executable} evil.py"
    store.save(run)

    res = code_task.resume(run_id, store=store, enclave_dir=str(enc), out=lambda _m: None)
    assert res["state"] == "BLOCKED"
    assert not (repo / "PWNED").exists(), "Tampered run['accept'] must never execute"


# ── T-06 / HIGH-03: streaming SHA-256 & tail accurate even when artifact caps ─

def test_t06_high03_streaming_sha256_and_tail_exact_on_artifact_truncation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "big.py").write_text(
        "for i in range(400):\n    print(f'ROW-{i:04d}-' + 'A'*60)\n", encoding="utf-8"
    )
    orig_load = exec_policy.load_policy

    def small_artifact_policy():
        pol, sha = orig_load()
        pol2 = json.loads(json.dumps(pol))
        pol2["output"]["artifact_max_bytes"] = 2048  # force artifact truncation far below output size
        return pol2, sha

    monkeypatch.setattr(exec_kernel, "load_policy", small_artifact_policy)
    journal = ExecJournal(tmp_path / "j.jsonl")
    res = exec_kernel.run(
        CommandRequest(argv=(sys.executable, str(ws / "big.py")), cwd=str(ws), origin="model", root=str(ws)),
        confirm=lambda _p, _c: True,
        journal=journal,
        artifacts_dir=tmp_path / "runs",
    )
    expected_full = "".join(f"ROW-{i:04d}-" + "A" * 60 + "\n" for i in range(400))
    assert res.artifact_truncated is True
    assert res.stdout_sha256 == hashlib.sha256(expected_full.encode("utf-8")).hexdigest()
    assert res.combined_sha256 == hashlib.sha256(expected_full.encode("utf-8")).hexdigest()
    assert res.combined_bytes == len(expected_full.encode("utf-8"))
    assert "ROW-0399-" in res.combined_tail


# ── T-07 / HIGH-04: _grep blocks secrets & allows metacharacter patterns ─────

def test_t07_high04_grep_skips_secrets_and_searches_legitimate_patterns(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / ".env").write_text("SECRET_API_KEY=CANARY_12345\n", encoding="utf-8")
    (ws / "priv.key").write_text("CANARY_12345\n", encoding="utf-8")
    (ws / "service_account_prod.json").write_text('{"k": "CANARY_12345"}\n', encoding="utf-8")
    (ws / "app.py").write_text(
        'cfg = ".env"\ncmd = "a && b | c $(date)"\nSAFE_MARKER = "OK"\n', encoding="utf-8"
    )
    agent = CodeAgent(root=str(ws), chat=_scripted(), out=lambda _s: None)

    out_canary, _ = agent._grep("CANARY_12345", ".")
    assert "CANARY_12345" not in out_canary
    assert out_canary == "(no matches)"

    for pat in (r"\.env", r"&&", r"\|", r"\$\(date\)"):
        out_legit, _ = agent._grep(pat, ".")
        assert "app.py:" in out_legit, f"Legitimate search pattern {pat!r} failed: {out_legit}"

    with pytest.raises(Denied):
        agent._grep("CANARY", ".env")


# ── T-08 / MED-03: osiris_shell governance, confinement, ANSI, non-TTY ───────

def test_t08_med03_osiris_shell_governed_execution_and_confinement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    outside = tmp_path / "outside"
    ws.mkdir()
    outside.mkdir()
    (ws / "ansi.py").write_text("print('\\x1b[31mRED_TEXT\\x1b[0m')\n", encoding="utf-8")
    journal = ExecJournal(tmp_path / "shell_journal.jsonl")

    # Non-TTY confirm fails closed on ASK-tier commands.
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    refused_nontty = osiris_shell._exec_shell(
        f"{sys.executable} ansi.py", root=str(ws), cwd=str(ws), journal=journal, artifacts_dir=tmp_path / "runs"
    )
    assert "refused" in refused_nontty.lower()

    # Shell operators & newline injection refused.
    for bad in ("git status; id", "cat foo>bar", "ls >& /tmp/pwn", "git status\nrm -rf /"):
        res = osiris_shell._exec_shell(
            bad, root=str(ws), cwd=str(ws), confirm=lambda _p, _c: True,
            journal=journal, artifacts_dir=tmp_path / "runs",
        )
        assert "refused" in res.lower(), f"osiris_shell allowed {bad!r}: {res}"

    # Confinement: cwd outside root is refused.
    esc = osiris_shell._exec_shell(
        "ls", root=str(ws), cwd=str(outside), confirm=lambda _p, _c: True,
        journal=journal, artifacts_dir=tmp_path / "runs",
    )
    assert "refused" in esc.lower()

    # ANSI stripped cleanly on approved execution.
    ok_out = osiris_shell._exec_shell(
        f"{sys.executable} ansi.py", root=str(ws), cwd=str(ws), confirm=lambda _p, _c: True,
        journal=journal, artifacts_dir=tmp_path / "runs",
    )
    assert "RED_TEXT" in ok_out
    assert "\x1b[" not in ok_out


# ── T-09 / T-10 / T-11 / CRIT-05 / HIGH-02 / V6-03..05 / MED-04..05: probes ──

def test_t09_med04_v604_safepathguard_and_skills_block_symlink_bounce_and_secrets(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    outside = tmp_path / "outside"
    ws.mkdir()
    outside.mkdir()
    (ws / "good.txt").write_text("hello Φ⚛—\n", encoding="utf-8")
    (ws / "token_counter.py").write_text("COUNT = 1\n", encoding="utf-8")
    (ws / "id_a").write_text("not an ssh key\n", encoding="utf-8")

    # Intermediate symlink bounce: ws/dir_link -> outside, outside/back.txt -> ws/good.txt
    (outside / "back.txt").symlink_to(ws / "good.txt")
    (ws / "dir_link").symlink_to(outside)

    guard = probes.SafePathGuard(ws, max_read_bytes=256)
    with pytest.raises(PermissionError):
        guard.resolve("dir_link/back.txt")

    # Legitimate code filenames should NOT be false-positived.
    assert guard.resolve("token_counter.py").name == "token_counter.py"
    assert guard.resolve("id_a").name == "id_a"

    # Multi-segment & newly hardened secrets blocked in BOTH probes and skills_runtime.
    secret_rel_paths = [
        ".config/gcloud/credentials.db",
        ".config/gh/hosts.yml",
        ".git-credentials",
        "service-account-prod.json",
        "keyrings/default.kbx",
        "state/ledger_main.db",
        "certs/client.p12",
        "certs/client.pfx",
        ".ssh/id_ecdsa",
        ".aws/config",
        "notes.env.txt",
    ]
    for rel in secret_rel_paths:
        target = ws / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("SECRET\n", encoding="utf-8")
        with pytest.raises(PermissionError, match="blocked|Secret|secret|Credential"):
            guard.resolve(rel)
        with pytest.raises(PermissionError):
            skills_runtime.safe_resolve_skill_resource(ws, rel)

    # Multi-byte UTF-8 truncation produces clean UTF-8 without U+FFFD.
    (ws / "utf8.txt").write_text("Φ" * 200, encoding="utf-8")  # 400 bytes, 2 bytes per char
    small_guard = probes.SafePathGuard(ws, max_read_bytes=257)  # odd boundary splits a 2-byte char
    _, text = small_guard.read_text_bounded("utf8.txt")
    assert "\ufffd" not in text
    assert text.startswith("Φ")

    # max_files=0 reads zero files.
    plan0 = probes.plan_probes("check good.txt", max_files=0)
    assert plan0.file_targets == ()


def test_t10_v605_med05_probe_substrate_arm64_not_termux_and_orphan_runs_code_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ARM64 Linux / Apple Silicon workstation without TERMUX_VERSION/com.termux MUST NOT be Termux.
    monkeypatch.delenv("TERMUX_VERSION", raising=False)
    monkeypatch.setenv("PREFIX", "/usr")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    sub_ws = probes.probe_substrate(power_dir=tmp_path / "no_power")
    assert sub_ws.is_termux is False
    assert sub_ws.node_id == "NODE_BETA_WORKSTATION"
    assert sub_ws.horizon_sec == 1800

    # TERMUX_VERSION explicitly triggers Node Alpha.
    monkeypatch.setenv("TERMUX_VERSION", "0.118.1")
    sub_tx = probes.probe_substrate(power_dir=tmp_path / "no_power")
    assert sub_tx.is_termux is True
    assert sub_tx.node_id == "NODE_ALPHA_EDGE"
    assert sub_tx.horizon_sec == 300

    # _count_orphan_runs detects unfinished runs in $OSIRIS_HOME/code/runs/*.json.
    ohome = tmp_path / ".osiris"
    cruns = ohome / "code" / "runs"
    cruns.mkdir(parents=True)
    (cruns / "code-done.json").write_text(json.dumps({"run_id": "code-done", "state": "SUCCEEDED"}), encoding="utf-8")
    (cruns / "code-stuck.json").write_text(json.dumps({"run_id": "code-stuck", "state": "RUNNING"}), encoding="utf-8")
    cnt, names = probes._count_orphan_runs(ohome)
    assert cnt == 1 and names == ["code-stuck.json"]


def _living_row(prev: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    e = {**payload, "prev": prev}
    e["hash"] = hashlib.sha256(json.dumps(e, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return e


def test_t11_crit05_high02_v603_med05_verify_chain_tail_hardening(tmp_path: Path) -> None:
    g = "0" * 64

    # 1. Missing file -> status="MISSING"
    miss = probes.verify_chain_tail(tmp_path / "absent.jsonl")
    assert miss.exists is False and miss.status == "MISSING"

    # 2. Invalid tail_lines <= 0 -> fail closed
    assert probes.verify_chain_tail(tmp_path / "absent.jsonl", tail_lines=0).intact is False
    assert probes.verify_chain_tail(tmp_path / "absent.jsonl", tail_lines=-2).intact is False

    # 3. Trailing entry >200 KiB: intact passes, tampered fails (NEVER EMPTY/intact=True!)
    big_path = tmp_path / "big.jsonl"
    r0 = _living_row(g, {"user": "u0", "reply": "r0"})
    r1 = _living_row(r0["hash"], {"user": "x" * 210_000, "reply": "r1"})
    big_path.write_text(json.dumps(r0) + "\n" + json.dumps(r1) + "\n", encoding="utf-8")
    rep_ok = probes.verify_chain_tail_report(big_path, tail_lines=1)
    assert rep_ok.intact is True and rep_ok.status == "INTACT" and rep_ok.inspected_count == 1

    r1_bad = dict(r1, reply="TAMPERED")
    big_path.write_text(json.dumps(r0) + "\n" + json.dumps(r1_bad) + "\n", encoding="utf-8")
    rep_bad = probes.verify_chain_tail_report(big_path, tail_lines=1)
    assert rep_bad.intact is False and rep_bad.status == "TAMPERED"

    # 4. Multi-byte UTF-8 ledger >80 KiB (Φ/⚛/—) across multiple tail_lines values -> never false UNREADABLE
    utf8_path = tmp_path / "utf8.jsonl"
    prev = g
    rows = []
    for i in range(60):
        row = _living_row(prev, {"user": f"msg-{i}-" + "Φ⚛—" * 200, "reply": "✓" * 100})
        rows.append(json.dumps(row, ensure_ascii=False))
        prev = row["hash"]
    utf8_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    for tl in (1, 2, 3, 5, 7, 10, 15):
        r = probes.verify_chain_tail_report(utf8_path, tail_lines=tl)
        assert r.intact is True and r.status == "INTACT", (tl, r)

    # 5. Forged-head small file (full read, prev != genesis) -> TAMPERED
    forged = tmp_path / "forged.jsonl"
    rf = _living_row("a" * 64, {"user": "forged", "reply": "head"})
    forged.write_text(json.dumps(rf) + "\n", encoding="utf-8")
    rep_forged = probes.verify_chain_tail_report(forged, tail_lines=10)
    assert rep_forged.intact is False and rep_forged.status == "TAMPERED"

    # 6. Historical fork -> INTACT (parity with verify_chain)
    fork_path = tmp_path / "fork.jsonl"
    f0 = _living_row(g, {"user": "0"})
    f1 = _living_row(f0["hash"], {"user": "1"})
    f2 = _living_row(f0["hash"], {"user": "2"})
    fork_path.write_text("\n".join(json.dumps(x) for x in (f0, f1, f2)) + "\n", encoding="utf-8")
    assert probes.verify_chain(str(fork_path))["intact"] is True
    rep_fork = probes.verify_chain_tail_report(fork_path, tail_lines=10)
    assert rep_fork.intact is True and rep_fork.status == "INTACT"

    # 7. V6-03 fail-closed on unrecognized hash schema
    unk_path = tmp_path / "unknown.jsonl"
    unk_path.write_text(
        json.dumps({"record_hash": "b" * 64, "prev_id": g, "payload": "tampered"}) + "\n",
        encoding="utf-8",
    )
    rep_unk = probes.verify_chain_tail_report(unk_path, tail_lines=5)
    assert rep_unk.intact is False and rep_unk.status == "TAMPERED"


# ── T-12 / HIGH-05 / MED-01: honest pytest parsing & PromotionPolicy ─────────

def test_t12_high05_med01_pytest_xfail_xpass_spoof_and_skip_promotion(tmp_path: Path) -> None:
    from osiris_cli.orchestration import evolution_bridge, trio
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "test_suite.py").write_text(
        "import pytest\n"
        "def test_pass_and_spoof():\n"
        "    print('SKIPPED [50] spoofed')\n"
        "    print('PASSED fake::test_ghost')\n"
        "    assert True\n"
        "def test_pass_2():\n"
        "    assert True\n"
        "@pytest.mark.xfail(reason='known')\n"
        "def test_xf():\n"
        "    assert False\n"
        "@pytest.mark.xfail(reason='unexpected pass')\n"
        "def test_xp():\n"
        "    assert True\n",
        encoding="utf-8",
    )
    ev, passed, failed = trio.run_pytest(str(repo), sandboxed=False)
    assert ev.passed == 2
    assert ev.skipped == 0, f"Stdout spoofing leaked into skipped count: {ev}"
    assert ev.xfailed == 1
    assert ev.xpassed == 1
    assert evolution_bridge.pass_rate(ev) == pytest.approx(0.5)

    # All-skipped suite -> green=False, pass_rate=0.0
    ev_all_skip = ExecutionEvidence(
        argv=("pytest",), returncode=0, passed=0, failed=0, errors=0,
        stdout_digest="0" * 64, stdout_tail="", skipped=2,
    )
    assert ev_all_skip.green is False
    assert evolution_bridge.pass_rate(ev_all_skip) == 0.0

    # Truncated evidence -> green=False
    ev_trunc = ExecutionEvidence(
        argv=("pytest",), returncode=0, passed=2, failed=0, errors=0,
        stdout_digest="0" * 64, stdout_tail="", truncated=True,
    )
    assert ev_trunc.green is False

    # MED-01: Candidate with 1 pass + 1 skip (fitness=0.5) is REJECTED by default.
    repo2 = tmp_path / "repo2"
    repo2.mkdir()
    (repo2 / "mod.py").write_text("def f(): return 0\n", encoding="utf-8")
    (repo2 / "test_mod.py").write_text(
        "import pytest\nfrom mod import f\n"
        "def test_ok(): assert f() == 1\n"
        "@pytest.mark.skip(reason='pending')\n"
        "def test_sk(): assert True\n",
        encoding="utf-8",
    )
    trio_res = trio.run_trio(
        str(repo2),
        lambda _f, _sb: trio.DeveloperOutput({"mod.py": "def f(): return 1\n"}, "Fixed f()."),
        str(tmp_path / "trio_out"),
        sandboxed=False,
    )
    dec = evolution_bridge.evaluate_candidate(
        trio_res,
        str(tmp_path / "lineage.jsonl"),
        authorizer=lambda _d, _s: (True, "ok"),
    )
    assert dec.candidate_fitness == pytest.approx(0.5)
    assert dec.status == "REJECTED"


# ── T-13 / V6-01 / V6-02 / HIGH-06: TUI footer, AUTO exec_kernel, live wiring ─

def test_t13_v601_v602_high06_tui_footer_and_auto_mode_enforces_exec_kernel(tmp_path: Path) -> None:
    living = Osiris(
        core=None, mentor=None, home=str(tmp_path / "living"), out=lambda _s: None, background=False, tty=False
    )
    # V6-01: collect_viewport_snapshot and render_footer work on missing, intact, and tampered ledgers.
    snap_missing = tui_inline.collect_viewport_snapshot(living=living, workspace_root=tmp_path, home_dir=tmp_path)
    assert snap_missing.ledger_head == "empty"
    vp = tui_inline.InlineViewport(
        living_getter=lambda: living, workspace_root=tmp_path, home_dir=tmp_path, columns=76
    )
    footer = vp.render_footer()
    assert len(footer) <= 76 and "empty" in footer

    living._record({"user": "hello", "reply": "world"})
    snap_ok = tui_inline.collect_viewport_snapshot(living=living, workspace_root=tmp_path, home_dir=tmp_path)
    assert snap_ok.ledger_head != "empty" and snap_ok.ledger_head != "TAMPERED"

    # V6-02: In EXEC_GATE=AUTO, dangerous/mutating/network/secret commands are blocked via exec_kernel.
    journal = ExecJournal(tmp_path / "tui_journal.jsonl")
    for bad_argv in (
        ["bash", "-c", "echo pwn"],
        ["git", "push", "--force"],
        ["curl", "https://evil.example"],
        ["cat", str(Path.home() / ".ssh" / "id_rsa")],
        ["rm", "-rf", "/"],
    ):
        res = tui_inline.stream_command_bounded(
            bad_argv,
            exec_gate="AUTO",
            cwd=tmp_path,
            root=tmp_path,
            use_rich_live=False,
            out=lambda _s: None,
            exec_journal=journal,
            exec_artifacts_dir=tmp_path / "runs",
        )
        assert res.blocked is True and res.exit_code == 126, (bad_argv, res)

    # run_bang_command rejects multiline injection via _split_shell_command.
    bang_bad = tui_inline.run_bang_command("!git status\nrm -rf /", out=lambda _s: None)
    assert bang_bad.blocked is True and bang_bad.exit_code == 126

    # HIGH-06: verify probes.py is genuinely imported/wired into live modules.
    from osiris_cli import knowledge, living as living_mod
    from osiris_cli.providers import router as router_mod
    assert hasattr(living_mod, "audit_mentor_reply")
    assert hasattr(knowledge, "dense_retrieval_allowed")
    assert hasattr(router_mod, "collect_system_telemetry")
