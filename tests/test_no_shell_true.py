"""AST gate forbidding live shell=True, os.system, and direct subprocess bypasses,
plus regression tests for exec_kernel --classify-json and wired callers (CAP-0005)."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest

from osiris_cli import exec_kernel

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST_PATH = Path(__file__).resolve().parent / "fixtures" / "shell_true_allowlist.json"

SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".tox",
    "build",
    "dist",
    "tests",
    ".venv",
    "venv",
}

FORBIDDEN_CALLS = {
    "os.system",
    "os.popen",
    "subprocess.getoutput",
    "subprocess.getstatusoutput",
    "pty.spawn",
    "os.posix_spawn",
    "os.posix_spawnp",
}
FORBIDDEN_PREFIXES = ("os.exec", "os.spawn")
KERNEL_OWNED_MODULES = (
    "osiris_cli/code_agent.py",
    "osiris_shell.py",
)
SUBPROCESS_SPAWN_ATTRS = {"Popen", "run", "call", "check_output", "check_call"}


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _scan_forbidden_calls() -> List[Tuple[str, int, str]]:
    hits: List[Tuple[str, int, str]] = []
    for path in sorted(REPO_ROOT.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.parts) or path.name.endswith(".bak"):
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        src = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(src, filename=rel)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = _call_name(node.func)
            for kw in node.keywords:
                if kw.arg == "shell":
                    is_const_false = isinstance(kw.value, ast.Constant) and kw.value.value is False
                    if not is_const_false:
                        hits.append((rel, node.lineno, f"{fn}(shell=True)"))
            if fn in FORBIDDEN_CALLS or fn.startswith(FORBIDDEN_PREFIXES):
                hits.append((rel, node.lineno, fn))
    return hits


def _load_allowlist() -> Dict[str, object]:
    return json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))


def test_allowlist_has_zero_live_entries():
    data = _load_allowlist()
    entries = data.get("entries", [])
    assert isinstance(entries, list)
    for entry in entries:
        assert entry["status"] in ("gated", "dead"), f"Forbidden status in allowlist: {entry}"
        assert entry["status"] != "live"
        assert (REPO_ROOT / entry["path"]).is_file(), f"Stale allowlist path: {entry['path']}"


def test_no_unallowlisted_shell_true_or_os_system():
    data = _load_allowlist()
    allowed: Dict[Tuple[str, str], int] = {}
    for entry in data["entries"]:
        key = (entry["path"], entry["kind"])
        allowed[key] = allowed.get(key, 0) + 1

    hits = _scan_forbidden_calls()
    seen_counts: Dict[Tuple[str, str], int] = {}
    unexpected: List[Tuple[str, int, str]] = []
    for rel, lineno, kind in hits:
        key = (rel, kind)
        seen_counts[key] = seen_counts.get(key, 0) + 1
        if seen_counts[key] > allowed.get(key, 0):
            unexpected.append((rel, lineno, kind))

    assert not unexpected, f"Unallowlisted shell/exec calls found: {unexpected}"


@pytest.mark.parametrize("rel_path", KERNEL_OWNED_MODULES)
def test_kernel_owned_modules_have_no_direct_subprocess_spawns(rel_path: str):
    path = REPO_ROOT / rel_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel_path)
    direct: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = _call_name(node.func)
        if fn.startswith("subprocess.") and fn.split(".")[-1] in SUBPROCESS_SPAWN_ATTRS:
            direct.append((node.lineno, fn))
    assert not direct, f"{rel_path} still bypasses exec_kernel via direct subprocess spawns: {direct}"


def test_exec_kernel_classify_json_stdin_and_argv(tmp_path: Path):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)

    # 1. Claude Code PreToolUse stdin format (READ command)
    payload = json.dumps({
        "tool_name": "Bash",
        "tool_input": {"command": "git status"},
        "cwd": str(tmp_path),
    })
    proc = subprocess.run(
        [sys.executable, "-m", "osiris_cli.exec_kernel", "--classify-json",
         "--origin", "claude_code", "--root", str(tmp_path)],
        input=payload,
        capture_output=True,
        text=True,
        env=env,
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["origin"] == "claude_code"
    assert out["tier"] == "READ"
    assert out["decision"] == "ALLOW"
    assert isinstance(out["normalized_argv"], list) and out["normalized_argv"]

    # 2. Denied git push via stdin PreToolUse payload
    deny_payload = json.dumps({
        "tool_name": "Bash",
        "tool_input": {"command": "git push origin main"},
        "cwd": str(tmp_path),
    })
    proc_deny = subprocess.run(
        [sys.executable, "-m", "osiris_cli.exec_kernel", "--classify-json",
         "--origin", "claude_code", "--root", str(tmp_path)],
        input=deny_payload,
        capture_output=True,
        text=True,
        env=env,
        timeout=15,
    )
    assert proc_deny.returncode == 0, proc_deny.stderr
    out_deny = json.loads(proc_deny.stdout)
    assert out_deny["decision"] == "DENY"

    # 3. Trailing -- argv fallback
    proc_argv = subprocess.run(
        [sys.executable, "-m", "osiris_cli.exec_kernel", "--classify-json",
         "--origin", "model", "--cwd", str(tmp_path), "--root", str(tmp_path),
         "--", "pytest", "-q"],
        input="",
        capture_output=True,
        text=True,
        env=env,
        timeout=15,
    )
    assert proc_argv.returncode == 0, proc_argv.stderr
    out_argv = json.loads(proc_argv.stdout)
    assert out_argv["tier"] == "VERIFY"
    assert out_argv["decision"] == "ASK"

    # 4. Malformed stdin fails closed (still emits JSON + exit 0)
    bad = exec_kernel.classify_json_payload("{not-json", origin="claude_code", root=str(tmp_path))
    assert bad["tier"] == "FORBIDDEN"
    assert bad["decision"] == "DENY"
