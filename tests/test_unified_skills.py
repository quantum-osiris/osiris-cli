"""Offline deterministic pytest suite for osiris_cli.skills_runtime and repo-local bootstrap."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from osiris_cli import skills_runtime
from osiris_cli.skills_runtime import (
    build_completer_dict,
    complete_skill_input,
    discover_skills,
    format_skills_table,
    hydrate_skill,
    load_skill_resource,
    resolve_osiris_home,
    safe_resolve_skill_resource,
    validate_skill_file,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write_lf(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = content.replace("\r\n", "\n").replace("\r", "\n").lstrip("\n")
    if not normalized.endswith("\n"):
        normalized += "\n"
    path.write_bytes(normalized.encode("utf-8"))
    return path


def _make_valid_skill_md(name: str, summary: str, body: str = "# Protocol\n\n1. Run deterministic check.") -> str:
    return (
        f"---\n"
        f"name: {name}\n"
        f"description: {summary} Use when validating deterministic OSIRIS workflows.\n"
        f"---\n\n"
        f"{body}\n"
    )


def test_project_overrides_global_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_home = tmp_path / "home"
    fake_repo = tmp_path / "repo"
    monkeypatch.delenv("OSIRIS_GLOBAL_SKILLS_DIR", raising=False)

    _write_lf(
        fake_home / ".agents" / "skills" / "osiris-preflight-check" / "SKILL.md",
        _make_valid_skill_md("osiris-preflight-check", "Global preflight baseline.", "# Global Version\n\nGlobal step."),
    )
    _write_lf(
        fake_home / ".agents" / "skills" / "global-only-probe" / "SKILL.md",
        _make_valid_skill_md("global-only-probe", "Global fallback probe.", "# Global Probe\n\nInspect host."),
    )
    _write_lf(
        fake_repo / ".agents" / "skills" / "osiris-preflight-check" / "SKILL.md",
        _make_valid_skill_md("osiris-preflight-check", "Project preflight override.", "# Project Version\n\nRepo step."),
    )

    found = discover_skills(workspace_root=fake_repo, home_dir=fake_home)
    assert set(found.keys()) == {"global-only-probe", "osiris-preflight-check"}

    preflight = found["osiris-preflight-check"]
    assert preflight.scope == "project"
    assert "Project Version" in preflight.body
    assert found["global-only-probe"].scope == "global"


def test_invalid_project_skill_fails_closed_over_global_shadow(tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    fake_repo = tmp_path / "repo"

    _write_lf(
        fake_home / ".agents" / "skills" / "osiris-sandbox-patch" / "SKILL.md",
        _make_valid_skill_md("osiris-sandbox-patch", "Valid global sandbox skill."),
    )
    _write_lf(
        fake_repo / ".agents" / "skills" / "osiris-sandbox-patch" / "SKILL.md",
        (
            "---\n"
            "name: osiris-sandbox-patch\n"
            "description: Invalid override skill. Use when testing fail-closed behavior.\n"
            "allowed-tools: Bash\n"
            "---\n\n"
            "# Broken Override\n"
        ),
    )

    valid_only = discover_skills(workspace_root=fake_repo, home_dir=fake_home, include_invalid=False)
    assert "osiris-sandbox-patch" not in valid_only

    audit_all = discover_skills(workspace_root=fake_repo, home_dir=fake_home, include_invalid=True)
    assert not audit_all["osiris-sandbox-patch"].valid
    issue_codes = {i.code for i in audit_all["osiris-sandbox-patch"].issues}
    assert "NON_UNIVERSAL_KEYS" in issue_codes


def test_portable_symlinks_claude_and_gemini_parity(tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    fake_repo = tmp_path / "repo"

    _write_lf(fake_repo / "AGENTS.md", "# AGENTS.md\n\nCanonical rules.\n")
    _write_lf(
        fake_repo / ".agents" / "skills" / "deep-research" / "references" / "researcher.md",
        "# Researcher Protocol\n\n1. Tag primary evidence.\n" + ("\n".join(f"- Rule {i}" for i in range(10))),
    )
    _write_lf(
        fake_repo / ".agents" / "skills" / "deep-research" / "SKILL.md",
        _make_valid_skill_md(
            "deep-research",
            "Coordinate multi-source research.",
            "# Deep Research\n\nRead references/researcher.md first.",
        ),
    )

    os.symlink("AGENTS.md", fake_repo / "CLAUDE.md")
    (fake_repo / ".claude").mkdir(parents=True, exist_ok=True)
    os.symlink("../.agents/skills", fake_repo / ".claude" / "skills")

    (fake_home / ".agents" / "skills").mkdir(parents=True, exist_ok=True)
    (fake_home / ".claude").mkdir(parents=True, exist_ok=True)
    (fake_home / ".gemini").mkdir(parents=True, exist_ok=True)
    os.symlink(str(fake_home / ".agents" / "skills"), fake_home / ".claude" / "skills")
    os.symlink(str(fake_home / ".agents" / "skills"), fake_home / ".gemini" / "skills")

    claude_skill = fake_repo / ".claude" / "skills" / "deep-research" / "SKILL.md"
    assert claude_skill.is_file()
    assert (fake_repo / "CLAUDE.md").read_bytes() == (fake_repo / "AGENTS.md").read_bytes()

    discovered = discover_skills(workspace_root=fake_repo, home_dir=fake_home)
    assert "deep-research" in discovered
    assert discovered["deep-research"].referenced_files == ("references/researcher.md",)


def test_universal_schema_catches_crlf_dashes_and_name_mismatch(tmp_path: Path) -> None:
    skill_dir = tmp_path / ".agents" / "skills" / "bad-skill"
    skill_dir.mkdir(parents=True)
    skill_md = skill_dir / "SKILL.md"

    raw_crlf = (
        "---\r\n"
        "name: wrong-name\r\n"
        "description: Missing trigger phrase and uses em-dash \u2014 here.\r\n"
        "model: claude-opus\r\n"
        "---\r\n\r\n"
        "# Heading\r\n"
    ).encode("utf-8")
    skill_md.write_bytes(raw_crlf)

    rec = validate_skill_file(skill_md, scope="project")
    assert not rec.valid
    codes = {iss.code for iss in rec.issues}
    assert "CRLF_DETECTED" in codes
    assert "TYPOGRAPHIC_DASH" in codes
    assert "NON_UNIVERSAL_KEYS" in codes
    assert "DIR_NAME_MISMATCH" in codes
    assert "MISSING_TRIGGER_PHRASE" in codes


def test_safe_path_guard_blocks_traversal_secrets_and_escaping_symlinks(tmp_path: Path) -> None:
    outside_secret = _write_lf(tmp_path / "outside_apikey.txt", "TOP_SECRET_VALUE")
    skill_dir = tmp_path / "repo" / ".agents" / "skills" / "sec-test"
    _write_lf(skill_dir / ".env", "API_KEY=leak")
    _write_lf(skill_dir / "apikey.json", '{"key": "leak"}')
    _write_lf(skill_dir / "references" / "id_ed25519", "PRIVATE_KEY")
    _write_lf(skill_dir / "references" / "safe.md", "# Safe Doc\n\nVerified notes.")

    escape_link = skill_dir / "references" / "escape.md"
    os.symlink(str(outside_secret), escape_link)

    resolved_safe = safe_resolve_skill_resource(skill_dir, "references/safe.md")
    assert resolved_safe.name == "safe.md"

    for blocked in (
        "../outside_apikey.txt",
        "../../outside_apikey.txt",
        "/etc/passwd",
        ".env",
        "apikey.json",
        "references/id_ed25519",
        "references/escape.md",
        "references/safe.md\x00.txt",
    ):
        with pytest.raises(PermissionError):
            safe_resolve_skill_resource(skill_dir, blocked)


def test_deep_research_coordinator_isolation_and_explicit_reference_hydration(tmp_path: Path) -> None:
    fake_repo = tmp_path / "repo"
    fake_home = tmp_path / "home"
    skill_base = fake_repo / ".agents" / "skills" / "deep-research"

    _write_lf(
        skill_base / "references" / "researcher.md",
        "# Researcher Subagent Protocol\n\nSECRET_SUBAGENT_MARKER_ALPHA\n",
    )
    _write_lf(
        skill_base / "references" / "report-writer.md",
        "# Report Writer Protocol\n\nSECRET_WRITER_MARKER_BETA\n",
    )
    _write_lf(
        skill_base / "SKILL.md",
        _make_valid_skill_md(
            "deep-research",
            "Coordinate multi-source research with provenance.",
            (
                "# Coordinator\n\n"
                "Spawn subagents pointing at {path_to_skill}/references/researcher.md "
                "and {path_to_skill}/references/report-writer.md."
            ),
        ),
    )

    coord_bundle = hydrate_skill(
        "deep-research",
        user_task="Compare qLDPC decoders on FPGA",
        workspace_root=fake_repo,
        home_dir=fake_home,
    )
    assert "SECRET_SUBAGENT_MARKER_ALPHA" not in coord_bundle.system_prompt
    assert "SECRET_WRITER_MARKER_BETA" not in coord_bundle.system_prompt
    assert str(skill_base.resolve()) in coord_bundle.system_prompt
    assert coord_bundle.loaded_resources == {}
    assert coord_bundle.ledger_receipt["schema"] == "osiris.skill_receipt.v1"
    assert coord_bundle.ledger_receipt["skill_name"] == "deep-research"

    subagent_bundle = hydrate_skill(
        "deep-research",
        user_task="Research FPGA BP+OSD latency",
        workspace_root=fake_repo,
        home_dir=fake_home,
        include_references=("references/researcher.md",),
    )
    assert "SECRET_SUBAGENT_MARKER_ALPHA" in subagent_bundle.system_prompt
    assert "references/researcher.md" in subagent_bundle.loaded_resources

    rec = discover_skills(workspace_root=fake_repo, home_dir=fake_home)["deep-research"]
    text, digest = load_skill_resource(rec, "references/researcher.md")
    assert "SECRET_SUBAGENT_MARKER_ALPHA" in text
    assert subagent_bundle.loaded_resources["references/researcher.md"] == digest


def test_dynamic_autocomplete_osiris_home_and_termux_compact_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_repo = tmp_path / "repo"
    fake_home = tmp_path / "home"
    custom_osiris = tmp_path / "custom_osiris_state"
    monkeypatch.setenv("OSIRIS_HOME", str(custom_osiris))
    assert resolve_osiris_home() == custom_osiris.resolve()

    _write_lf(
        fake_repo / ".agents" / "skills" / "osiris-preflight-check" / "SKILL.md",
        _make_valid_skill_md("osiris-preflight-check", "Run preflight ledger and Ollama checks."),
    )

    comp_initial = build_completer_dict(workspace_root=fake_repo, home_dir=fake_home)
    assert comp_initial["/skill"] == {"osiris-preflight-check": None}

    _write_lf(
        fake_repo / ".agents" / "skills" / "osiris-dual-node-sync" / "SKILL.md",
        _make_valid_skill_md("osiris-dual-node-sync", "Synchronize Node Alpha and Node Beta safely across SSH."),
    )

    comp_refreshed = build_completer_dict(workspace_root=fake_repo, home_dir=fake_home)
    assert comp_refreshed["/skill"] == {
        "osiris-dual-node-sync": None,
        "osiris-preflight-check": None,
    }

    matches = complete_skill_input("/skill osiris-", workspace_root=fake_repo, home_dir=fake_home)
    assert matches == ["/skill osiris-dual-node-sync", "/skill osiris-preflight-check"]
    assert complete_skill_input("/sk", workspace_root=fake_repo, home_dir=fake_home) == ["/skill", "/skills"]

    rendered = format_skills_table(workspace_root=fake_repo, home_dir=fake_home, max_width=76)
    lines = rendered.splitlines()
    assert len(lines) == 4
    assert all(len(line) <= 76 for line in lines)


def test_repo_local_bootstrap_and_five_starter_skills_verified(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "bootstrap_osiris_agents.py"), "--repo-root", str(REPO_ROOT), "--verify-only"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[VERIFY PASS]" in proc.stdout

    discovered = discover_skills(workspace_root=REPO_ROOT, home_dir=tmp_path / "isolated_home")
    assert set(discovered.keys()) == {
        "deep-research",
        "osiris-claims-grounding-audit",
        "osiris-dual-node-sync",
        "osiris-preflight-check",
        "osiris-sandbox-patch",
    }
    assert all(rec.valid and rec.scope == "project" for rec in discovered.values())


def test_module_passes_sys025_zero_stub_and_lf_gate() -> None:
    module_path = Path(skills_runtime.__file__).resolve()
    raw = module_path.read_bytes()
    assert b"\r" not in raw, "CR byte detected in skills_runtime.py"

    tree = ast.parse(raw.decode("utf-8"))
    stub_functions: list[str] = []

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
                stub_functions.append(node.name)

    assert stub_functions == [], f"SYS-025 stub functions detected: {stub_functions}"
