"""Offline deterministic security & provenance pytest suite for osiris_cli.probes."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from osiris_cli import probes
from osiris_cli.living import Osiris
from osiris_cli.probes import (
    SafePathGuard,
    compute_chained_entry_hash,
    format_probe_receipts_for_prompt,
    plan_and_run_probes,
    probe_ledger,
    probe_substrate,
    receipts_from_legacy_results,
    resolve_sibling_repo,
    run_check_command,
    verify_chain_tail,
)


def _write_lf(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = text.replace("\r\n", "\n").replace("\r", "\n")
    if not clean.endswith("\n"):
        clean += "\n"
    path.write_bytes(clean.encode("utf-8"))
    return path


def _append_canonical_ledger_row(ledger_file: Path, prev_hash: str, payload: dict[str, object]) -> str:
    row = dict(payload)
    row["hash_mode"] = "canonical_v1"
    row["prev_hash"] = prev_hash
    entry_hash = compute_chained_entry_hash(prev_hash, row)
    row["entry_hash"] = entry_hash
    ledger_file.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
    with ledger_file.open("ab") as fh:
        fh.write(line.encode("utf-8"))
    return entry_hash


def test_safepathguard_blocks_traversal_and_escape_vectors(tmp_path: Path) -> None:
    outside_secret = _write_lf(tmp_path / "outside.env", "SECRET_API_KEY=leak_123")
    ws = tmp_path / "repo"
    _write_lf(ws / "osiris_cli" / "living.py", "GATE_BPB = 2.0\n")

    guard = SafePathGuard(ws)
    resolved_ok, content = guard.read_text_bounded("osiris_cli/living.py")
    assert resolved_ok == (ws / "osiris_cli" / "living.py").resolve()
    assert content == "GATE_BPB = 2.0\n"

    for attack in (
        "../outside.env",
        "../../.env",
        "osiris_cli/../../outside.env",
        str(outside_secret),
        "/etc/passwd",
        "~/.ssh/id_ed25519",
        "osiris_cli\\living.py",
        "osiris_cli/living.py\x00.txt",
        "",
    ):
        with pytest.raises(PermissionError):
            guard.resolve(attack)


def test_safepathguard_blocks_all_credential_names_and_symlinks(tmp_path: Path) -> None:
    fake_ssh_dir = tmp_path / "fake_home" / ".ssh"
    ssh_key = _write_lf(fake_ssh_dir / "id_ed25519", "OPENSSH_PRIVATE_KEY_PAYLOAD")

    ws = tmp_path / "repo"
    secret_files = (
        ".env",
        ".env.local",
        "apikey.json",
        "config/aws_credentials.json",
        "config/client_secret.toml",
        "config/session_token.txt",
        "keys/id_rsa",
        "keys/id_ed25519",
        "keys/server.pem",
        "keys/tls.key",
        ".netrc",
        ".ssh/config",
    )
    for rel in secret_files:
        _write_lf(ws / rel, "TOP_SECRET_CREDENTIAL_MATERIAL")

    (ws / "links").mkdir(parents=True, exist_ok=True)
    os.symlink(str(ssh_key), ws / "links" / "external_ssh.py")
    os.symlink(str(tmp_path / "nonexistent" / "outside.py"), ws / "links" / "dangling_escape.py")
    os.symlink("../.env", ws / "links" / "innocent_notes.md")

    guard = SafePathGuard(ws)
    for blocked_rel in (
        *secret_files,
        "links/external_ssh.py",
        "links/dangling_escape.py",
        "links/innocent_notes.md",
    ):
        with pytest.raises(PermissionError):
            guard.resolve(blocked_rel)


def test_fuzz_plan_and_run_probes_zero_secret_leak_and_exact_sha256(tmp_path: Path) -> None:
    ssh_target = _write_lf(tmp_path / "outside_id_ed25519", "SUPER_SECRET_SSH_BYTES")
    ws = tmp_path / "repo"
    _write_lf(ws / ".env", "ANTHROPIC_API_KEY=SUPER_SECRET_ENV_BYTES")
    _write_lf(ws / "apikey.json", '{"key":"SUPER_SECRET_JSON_BYTES"}')
    valid_src = "def verify() -> bool:\n    return True\n"
    _write_lf(ws / "osiris_cli" / "living.py", valid_src)

    (ws / "docs").mkdir(parents=True, exist_ok=True)
    os.symlink(str(ssh_target), ws / "docs" / "escaped_link.md")

    prompt = (
        "Compare osiris_cli/living.py against ../../.env, apikey.json, "
        "docs/escaped_link.md, and future_uncreated_module.py."
    )
    receipts = plan_and_run_probes(prompt, workspace_root=ws, osiris_home=tmp_path / "osiris_home")

    by_target = {r.target: r for r in receipts}
    assert "osiris_cli/living.py" in by_target
    ok_rec = by_target["osiris_cli/living.py"]
    assert ok_rec.exit_code == 0
    assert ok_rec.blocked is False
    assert ok_rec.output_sha256 == hashlib.sha256(valid_src.encode("utf-8")).hexdigest()

    for malicious in ("../../.env", "apikey.json", "docs/escaped_link.md"):
        assert malicious in by_target
        blocked_rec = by_target[malicious]
        assert blocked_rec.blocked is True
        assert blocked_rec.exit_code == 13

    assert "future_uncreated_module.py" not in by_target

    rendered = format_probe_receipts_for_prompt(receipts)
    for forbidden_leak in (
        "SUPER_SECRET_SSH_BYTES",
        "SUPER_SECRET_ENV_BYTES",
        "SUPER_SECRET_JSON_BYTES",
    ):
        assert forbidden_leak not in rendered


def test_verify_chain_tail_supports_both_living_exchanges_and_canonical_v1(tmp_path: Path) -> None:
    # 1. Living Osiris._record format ("prev" / "hash") + legacy dict indexing compatibility
    living_home = tmp_path / "osiris_living" / "living"
    o = Osiris(core=None, mentor=None, home=str(living_home), out=lambda s: None, background=False, tty=False)
    for idx in range(10):
        o._record({"user": f"q{idx}", "reply": f"a{idx}"})
    rep_living = verify_chain_tail(o.log_path, tail_lines=6)
    assert rep_living.intact is True
    assert rep_living["intact"] is True
    assert rep_living["tail_count"] == 6
    assert rep_living["head"] == o.stats["last_head"][:12]

    # Tamper last entry in living log -> fails closed
    lines = Path(o.log_path).read_text(encoding="utf-8").splitlines()
    last = json.loads(lines[-1])
    last["reply"] = "TAMPERED"
    lines[-1] = json.dumps(last)
    _write_lf(Path(o.log_path), "\n".join(lines))
    assert verify_chain_tail(o.log_path, tail_lines=6).intact is False

    # 2. canonical_v1 format + orphan run detection
    osiris_home = tmp_path / ".osiris"
    exchanges = osiris_home / "living" / "exchanges.jsonl"
    prev = "0" * 64
    for idx in range(1, 26):
        prev = _append_canonical_ledger_row(
            exchanges,
            prev,
            {"idx": idx, "user": f"msg-{idx}", "held_out": (idx % 5 == 0)},
        )

    tail_ok = verify_chain_tail(exchanges, tail_window=6)
    assert tail_ok.intact is True
    assert tail_ok.inspected_count == 6
    assert tail_ok.tail_hash == prev[:12]

    ledger_receipt_ok = probe_ledger(osiris_home)
    assert ledger_receipt_ok.exit_code == 0
    assert "exchanges_ledger: INTACT" in ledger_receipt_ok.summary

    _write_lf(osiris_home / "runs" / "run_001.json", '{"run_id": "run_001", "status": "running"}')
    orphan_receipt = probe_ledger(osiris_home)
    assert orphan_receipt.exit_code == 1
    assert "orphan_runs: 1 (run_001.json)" in orphan_receipt.summary
    (osiris_home / "runs" / "run_001.json").unlink()

    lines = exchanges.read_text(encoding="utf-8").splitlines()
    last_obj = json.loads(lines[-1])
    last_obj["prev_hash"] = "f" * 64
    lines[-1] = json.dumps(last_obj)
    _write_lf(exchanges, "\n".join(lines))

    tail_bad = verify_chain_tail(exchanges, tail_window=6)
    assert tail_bad.intact is False
    assert probe_ledger(osiris_home).exit_code == 1


def test_git_trainer_and_sibling_repo_orchestration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(fake_home))

    repos_root = fake_home / "osiris-repos"
    ws = repos_root / "osiris-cli"
    sibling = repos_root / "dnalang-core"
    _write_lf(ws / "README.md", "# osiris-cli\n")
    _write_lf(sibling / "dnalang" / "ledger.py", "LEDGER_SCHEMA = 1\n")

    for repo_dir in (ws, sibling):
        subprocess.run(["git", "-C", str(repo_dir), "init"], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(repo_dir), "config", "user.email", "test@example.com"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_dir), "config", "user.name", "OSIRIS Test"],
            check=True,
            capture_output=True,
        )
        subprocess.run(["git", "-C", str(repo_dir), "add", "."], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(repo_dir), "commit", "-m", "init"],
            check=True,
            capture_output=True,
        )

    osiris_home = fake_home / ".osiris"
    exchanges = osiris_home / "living" / "exchanges.jsonl"
    prev = "0" * 64
    for idx in range(1, 16):
        prev = _append_canonical_ledger_row(exchanges, prev, {"idx": idx})

    _write_lf(
        osiris_home / "living" / "trainer_state.json",
        json.dumps({"step": 927, "held_out_bpb": 6.15, "unigram_bpb": 4.96}),
    )

    assert resolve_sibling_repo("dnalang-core", workspace_root=ws, osiris_home=osiris_home) == sibling.resolve()

    receipts = plan_and_run_probes(
        "Why did the last git commit affect the ledger chain, nclm step, and dnalang-core?",
        workspace_root=ws,
        osiris_home=osiris_home,
    )
    kinds = {r.probe for r in receipts}
    assert {"git", "ledger", "trainer", "sibling_repo"}.issubset(kinds)
    for rec in receipts:
        assert rec.exit_code == 0
        assert rec.output_sha256 == hashlib.sha256(rec.summary.encode("utf-8")).hexdigest()
        ledger_dict = rec.to_ledger_dict()
        assert ledger_dict["sha256"] == rec.output_sha256

    # Legacy adapter preserves SHA-256 receipts cleanly
    legacy = receipts_from_legacy_results([{"name": "file", "cmd": "read README.md", "out": "# hi"}])
    assert legacy[0]["probe"] == "read_file"
    assert legacy[0]["sha256"] == hashlib.sha256(b"# hi").hexdigest()


def test_termux_unreadable_power_sensor_graceful_degradation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
    monkeypatch.setattr("platform.machine", lambda: "aarch64")

    fake_meminfo = _write_lf(tmp_path / "meminfo", "MemTotal: 12288000 kB\nMemAvailable: 4194304 kB\n")
    missing_power = tmp_path / "unreadable_sysfs_power"

    sub = probe_substrate(power_dir=missing_power, meminfo_path=fake_meminfo)
    assert sub.node_id == "NODE_ALPHA_EDGE"
    assert sub.is_termux is True
    assert sub.horizon_sec == 300
    assert sub.free_ram_mb == 4096
    assert sub.power_status == "unreadable_attended_only"


def test_run_check_command_and_sys025_zero_stub_compliance(tmp_path: Path) -> None:
    ws = tmp_path / "repo"
    osiris_home = tmp_path / ".osiris"
    _write_lf(ws / "osiris_cli" / "claims.py", "CLAIMS = {}\n")

    all_checks = run_check_command("all", workspace_root=ws, osiris_home=osiris_home)
    assert {r.probe for r in all_checks} == {"ledger", "git", "trainer", "system", "evidence"}

    blocked_check = run_check_command("git", target="../../.env", workspace_root=ws, osiris_home=osiris_home)
    assert blocked_check[-1].blocked is True
    assert blocked_check[-1].exit_code == 13

    module_path = Path(probes.__file__).resolve()
    raw = module_path.read_bytes()
    assert b"\r" not in raw, "CR byte detected in osiris_cli/probes.py"

    tree = ast.parse(raw.decode("utf-8"))
    stubs: list[str] = []

    def _is_stub_stmt(stmt: ast.stmt) -> bool:
        if isinstance(stmt, ast.Pass):
            return True
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is Ellipsis:
            return True
        if isinstance(stmt, ast.Raise) and stmt.exc is not None:
            exc = stmt.exc.func if isinstance(stmt.exc, ast.Call) else stmt.exc
            return isinstance(exc, ast.Name) and exc.id == "NotImplementedError"
        if isinstance(stmt, ast.Return):
            return stmt.value is None or (isinstance(stmt.value, ast.Constant) and stmt.value.value is None)
        return False

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]
            if all(_is_stub_stmt(s) for s in body):
                stubs.append(node.name)

    assert stubs == [], f"SYS-025 stub functions found in probes.py: {stubs}"
