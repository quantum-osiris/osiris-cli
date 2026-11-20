"""Hardened repo-wide AST gate forbidding shell=True, os.system, os.popen, os.exec*,
asyncio.create_subprocess_shell, dynamic getattr escapes, and direct subprocess spawns
in kernel-owned modules (MED-06)."""

from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST_PATH = Path(__file__).resolve().parent / "fixtures" / "shell_true_allowlist.json"

SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".venv",
    "venv",
    "build",
    "dist",
    "tests",
    "docs",
}

FORBIDDEN_OS_FUNCS = {
    "system",
    "popen",
    "execl",
    "execle",
    "execlp",
    "execlpe",
    "execv",
    "execve",
    "execvp",
    "execvpe",
    "spawnl",
    "spawnle",
    "spawnlp",
    "spawnlpe",
    "spawnv",
    "spawnve",
    "spawnvp",
    "spawnvpe",
}

SUBPROCESS_FUNCS = {
    "Popen",
    "run",
    "call",
    "check_call",
    "check_output",
    "getoutput",
    "getstatusoutput",
}

# Modules whose subprocess execution MUST flow exclusively through osiris_cli.exec_kernel.
KERNEL_OWNED_MODULES = (
    "osiris_cli/code_agent.py",
    "osiris_shell.py",
    "osiris_cli/tui_inline.py",
)


def _iter_repo_python_files() -> List[Path]:
    out: List[Path] = []
    for p in sorted(REPO_ROOT.rglob("*.py")):
        rel_parts = p.relative_to(REPO_ROOT).parts
        if any(part in SKIP_DIRS for part in rel_parts):
            continue
        out.append(p)
    return out


def _collect_import_aliases(tree: ast.AST) -> Tuple[Dict[str, str], Dict[str, Tuple[str, str]]]:
    """Map module aliases (`import os as _o` -> `{'_o': 'os'}`) and direct symbol imports
    (`from os import system as _s` -> `{'_s': ('os', 'system')}`)."""
    mod_aliases: Dict[str, str] = {}
    sym_aliases: Dict[str, Tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                mod_aliases[local] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                local = alias.asname or alias.name
                sym_aliases[local] = (node.module, alias.name)
    return mod_aliases, sym_aliases


def _is_truthy_shell_kw(node: ast.Call) -> bool:
    for kw in node.keywords:
        if kw.arg == "shell":
            if isinstance(kw.value, ast.Constant):
                if bool(kw.value.value):
                    return True
            else:
                return True
        elif kw.arg is None and isinstance(kw.value, ast.Dict):
            for k_node, v_node in zip(kw.value.keys, kw.value.values):
                if isinstance(k_node, ast.Constant) and k_node.value == "shell":
                    if not isinstance(v_node, ast.Constant) or bool(v_node.value):
                        return True
    return False


def _popen_positional_shell_true(node: ast.Call) -> bool:
    # subprocess.Popen(args, bufsize, executable, stdin, stdout, stderr, preexec_fn, close_fds, shell)
    if len(node.args) >= 9:
        shell_arg = node.args[8]
        if not isinstance(shell_arg, ast.Constant) or bool(shell_arg.value):
            return True
    return False


def scan_source_for_shell_hazards(source: str, rel_path: str = "<mem>") -> List[Tuple[str, int, str]]:
    """Scan Python source AST for shell=True, os.system/popen/exec*/spawn*,
    asyncio.create_subprocess_shell, and dynamic getattr(os, 'system') escapes.
    Fails closed on SyntaxError (never skips unparseable files)."""
    try:
        tree = ast.parse(source, filename=rel_path)
    except SyntaxError as exc:
        return [(rel_path, int(exc.lineno or 1), f"SyntaxError:{exc.msg}")]

    mod_aliases, sym_aliases = _collect_import_aliases(tree)
    findings: List[Tuple[str, int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        lineno = getattr(node, "lineno", 0)
        func = node.func

        # Dynamic getattr(os, "system") / getattr(subprocess, ...)
        if isinstance(func, ast.Name) and func.id == "getattr" and len(node.args) >= 2:
            target_obj, attr_node = node.args[0], node.args[1]
            if isinstance(target_obj, ast.Name) and isinstance(attr_node, ast.Constant) and isinstance(attr_node.value, str):
                resolved_mod = mod_aliases.get(target_obj.id, target_obj.id)
                if resolved_mod == "os" and attr_node.value in FORBIDDEN_OS_FUNCS:
                    findings.append((rel_path, lineno, f"os.{attr_node.value}"))
                elif resolved_mod == "asyncio" and attr_node.value == "create_subprocess_shell":
                    findings.append((rel_path, lineno, "asyncio.create_subprocess_shell"))
            continue

        # Direct call via `from os import system` / `from subprocess import Popen` / `from asyncio import ...`
        if isinstance(func, ast.Name) and func.id in sym_aliases:
            orig_mod, orig_sym = sym_aliases[func.id]
            if orig_mod == "os" and orig_sym in FORBIDDEN_OS_FUNCS:
                findings.append((rel_path, lineno, f"os.{orig_sym}"))
            elif orig_mod == "asyncio" and orig_sym == "create_subprocess_shell":
                findings.append((rel_path, lineno, "asyncio.create_subprocess_shell"))
            elif orig_mod == "subprocess" and orig_sym in SUBPROCESS_FUNCS:
                if orig_sym in ("getoutput", "getstatusoutput"):
                    findings.append((rel_path, lineno, f"subprocess.{orig_sym}"))
                elif _is_truthy_shell_kw(node) or (orig_sym == "Popen" and _popen_positional_shell_true(node)):
                    findings.append((rel_path, lineno, f"subprocess.{orig_sym}(shell=True)"))
            continue

        # Attribute call: mod.func(...)
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            resolved_mod = mod_aliases.get(func.value.id, func.value.id)
            attr = func.attr
            if resolved_mod == "os" and attr in FORBIDDEN_OS_FUNCS:
                findings.append((rel_path, lineno, f"os.{attr}"))
            elif resolved_mod == "asyncio" and attr == "create_subprocess_shell":
                findings.append((rel_path, lineno, "asyncio.create_subprocess_shell"))
            elif resolved_mod == "subprocess" and attr in SUBPROCESS_FUNCS:
                if attr in ("getoutput", "getstatusoutput"):
                    findings.append((rel_path, lineno, f"subprocess.{attr}"))
                elif _is_truthy_shell_kw(node) or (attr == "Popen" and _popen_positional_shell_true(node)):
                    findings.append((rel_path, lineno, f"subprocess.{attr}(shell=True)"))
            elif _is_truthy_shell_kw(node):
                findings.append((rel_path, lineno, f"{resolved_mod}.{attr}(shell=True)"))

    return findings


def scan_source_for_direct_subprocess(source: str, rel_path: str = "<mem>") -> List[Tuple[str, int, str]]:
    """Detect ANY direct subprocess spawn (including aliased/from-imports) in kernel-owned modules."""
    try:
        tree = ast.parse(source, filename=rel_path)
    except SyntaxError as exc:
        return [(rel_path, int(exc.lineno or 1), f"SyntaxError:{exc.msg}")]

    mod_aliases, sym_aliases = _collect_import_aliases(tree)
    hits: List[Tuple[str, int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        lineno = getattr(node, "lineno", 0)
        func = node.func
        if isinstance(func, ast.Name) and func.id in sym_aliases:
            orig_mod, orig_sym = sym_aliases[func.id]
            if orig_mod == "subprocess" and orig_sym in SUBPROCESS_FUNCS:
                hits.append((rel_path, lineno, f"subprocess.{orig_sym}"))
            elif orig_mod == "asyncio" and orig_sym in ("create_subprocess_exec", "create_subprocess_shell"):
                hits.append((rel_path, lineno, f"asyncio.{orig_sym}"))
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            resolved_mod = mod_aliases.get(func.value.id, func.value.id)
            if resolved_mod == "subprocess" and func.attr in SUBPROCESS_FUNCS:
                hits.append((rel_path, lineno, f"subprocess.{func.attr}"))
            elif resolved_mod == "asyncio" and func.attr in ("create_subprocess_exec", "create_subprocess_shell"):
                hits.append((rel_path, lineno, f"asyncio.{func.attr}"))
    return hits


def test_allowlist_has_zero_live_entries() -> None:
    data = json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    live = [e for e in data["entries"] if e.get("status") == "live"]
    assert live == [], f"Live shell=True/os.system entries are forbidden: {live}"


def test_repo_has_no_unapproved_shell_hazards_and_no_stale_allowlist() -> None:
    data = json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))
    allowed: Counter[Tuple[str, str]] = Counter(
        (e["path"], e["kind"]) for e in data["entries"]
    )
    seen: Counter[Tuple[str, str]] = Counter()
    all_findings: List[Tuple[str, int, str]] = []

    for py_file in _iter_repo_python_files():
        rel = str(py_file.relative_to(REPO_ROOT)).replace("\\", "/")
        src = py_file.read_text(encoding="utf-8", errors="replace")
        file_hits = scan_source_for_shell_hazards(src, rel)
        for path_s, lineno, kind in file_hits:
            seen[(path_s, kind)] += 1
            all_findings.append((path_s, lineno, kind))

    unexpected = [
        (p, ln, k) for p, ln, k in all_findings if seen[(p, k)] > allowed.get((p, k), 0)
    ]
    assert unexpected == [], f"Unexpected shell=True/os.system/SyntaxError findings: {unexpected}"

    stale = [
        (key, cnt, seen.get(key, 0))
        for key, cnt in allowed.items()
        if seen.get(key, 0) < cnt
    ]
    assert stale == [], f"Stale allowlist entries must be removed: {stale}"


def test_kernel_owned_modules_have_no_direct_subprocess_spawns() -> None:
    violations: List[Tuple[str, int, str]] = []
    for rel in KERNEL_OWNED_MODULES:
        p = REPO_ROOT / rel
        assert p.is_file(), f"Expected kernel-owned module missing: {rel}"
        src = p.read_text(encoding="utf-8")
        violations.extend(scan_source_for_direct_subprocess(src, rel))
    assert violations == [], f"Kernel-owned modules must route all spawns through exec_kernel: {violations}"


# ── Meta-tests proving MED-06 AST gate catches every evasion pattern ─────────

@pytest.mark.parametrize("snippet,expected_kind", [
    ("from os import system\nsystem('id')\n", "os.system"),
    ("import os as _o\n_o.system('id')\n", "os.system"),
    ("import os\ngetattr(os, 'system')('id')\n", "os.system"),
    ("import asyncio\nasyncio.create_subprocess_shell('id')\n", "asyncio.create_subprocess_shell"),
    ("import subprocess\nsubprocess.run(['id'], **{'shell': True})\n", "subprocess.run(shell=True)"),
    ("from subprocess import Popen\nPopen(['id'], -1, None, None, None, None, None, True, True)\n", "subprocess.Popen(shell=True)"),
    ("def broken(:\n    pass\n", "SyntaxError:"),
])
def test_ast_gate_meta_catches_evasion_and_syntax_errors(snippet: str, expected_kind: str) -> None:
    hits = scan_source_for_shell_hazards(snippet, "meta.py")
    assert hits, f"AST gate missed hazard: {snippet!r}"
    assert any(kind.startswith(expected_kind) for _, _, kind in hits), hits


@pytest.mark.parametrize("snippet", [
    "from subprocess import Popen\nPopen(['ls'])\n",
    "import subprocess as _sp\n_sp.run(['ls'])\n",
    "from asyncio import create_subprocess_exec\ncreate_subprocess_exec('ls')\n",
])
def test_ast_gate_meta_catches_aliased_direct_subprocess_spawns(snippet: str) -> None:
    hits = scan_source_for_direct_subprocess(snippet, "meta.py")
    assert hits, f"Direct-subprocess gate missed: {snippet!r}"
