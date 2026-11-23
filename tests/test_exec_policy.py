"""Offline tests for the execution policy: classification, normalization and environment scrubbing.

Nothing here spawns a process. Every case classifies inside a temporary root.
"""

import json
import os
import sys
from pathlib import Path

import pytest

from osiris_cli import exec_policy
from osiris_cli.exec_policy import (
    CommandRequest,
    PolicyError,
    classify,
    is_sensitive_argv,
    scrubbed_env,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "exec_policy_cases.json"
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))



def cls_reasons(v):
    return v.reasons


def _req(argv, origin, root, cwd=None, **kw):
    return CommandRequest(argv=tuple(argv), cwd=str(cwd or root), origin=origin, root=str(root), **kw)


def test_fixture_has_at_least_eighty_cases_and_covers_every_bypass():
    assert len(CASES) >= 80
    ids = {case["id"] for case in CASES}
    for required in ("bypass-bash-c-model", "bypass-sh-c-model", "shell-env-assign-model",
                     "bypass-nohup-rm-model", "bypass-timeout-rm-model", "bypass-dotdot-rm-model",
                     "shell-find-exec-semicolon-model", "mutate-find-delete-model",
                     "shell-git-c-alias-model", "shell-git-c-pager-model", "mutate-git-quoted-push-model",
                     "shell-git-dir-env-push-model", "secret-cat-env-model", "secret-rg-env-local-model",
                     "forbid-sudo-model", "net-curl-model", "net-pip-install-model",
                     "net-run-ibm-model", "bypass-python-I-c-model", "forbid-xargs-model"):
        assert required in ids, required


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_fixture_case(case, tmp_path):
    cls = classify(_req(case["argv"], case["origin"], tmp_path))
    assert cls.tier == case["tier"], cls.reasons
    assert cls.decision == case["decision"], cls.reasons


def test_policy_hash_is_recorded_and_stable():
    first = classify(_req(["ls"], "model", Path.cwd()))
    second = classify(_req(["ls"], "model", Path.cwd()))
    assert len(first.policy_sha256) == 64
    assert first.policy_sha256 == second.policy_sha256


def test_unknown_origin_is_denied(tmp_path):
    cls = classify(_req(["ls"], "stranger", tmp_path))
    assert cls.decision == "DENY"


def test_empty_or_non_string_argv_is_denied(tmp_path):
    assert classify(CommandRequest(argv=(), cwd=str(tmp_path), origin="human", root=str(tmp_path))).decision == "DENY"
    assert classify(CommandRequest(argv=("ls", 3), cwd=str(tmp_path), origin="human", root=str(tmp_path))).decision == "DENY"


def test_nul_and_control_characters_are_denied(tmp_path):
    assert classify(_req(["l\x01s"], "human", tmp_path)).decision == "DENY"
    assert classify(_req(["ls", "a\x00b"], "human", tmp_path)).decision == "DENY"


def test_symlink_named_ls_pointing_at_rm_is_treated_as_rm(tmp_path):
    rm_path = None
    for candidate in ("/usr/bin/rm", "/bin/rm"):
        if os.path.exists(candidate):
            rm_path = candidate
            break
    if rm_path is None:
        pytest.skip("no rm binary on this system")
    link = tmp_path / "ls"
    link.symlink_to(rm_path)
    cls = classify(_req(["./ls", "-rf", "x"], "model", tmp_path))
    assert cls.decision == "DENY"
    assert cls.tier == "MUTATE"


def test_symlink_chain_through_renamed_rm_in_a_system_dir_is_treated_as_rm(tmp_path, monkeypatch):
    # Ubuntu 26.04: /usr/bin/rm -> gnurm. The final name is unknown and sits in a system dir, so
    # judging only the last hop let ./ls -> rm -> gnurm through as ls.
    sysdir = tmp_path / "sysbin"
    sysdir.mkdir()
    real = sysdir / "gnurm"
    real.write_text("#!/bin/sh\n")
    real.chmod(0o755)
    (sysdir / "rm").symlink_to("gnurm")
    (tmp_path / "ls").symlink_to(sysdir / "rm")
    monkeypatch.setattr(exec_policy, "_SYSTEM_DIRS", (str(sysdir) + os.sep,))
    cls = classify(_req(["./ls", "-rf", "x"], "model", tmp_path))
    assert cls.decision == "DENY"
    assert cls.tier == "MUTATE"


def test_symlink_straight_to_gnu_prefixed_rm_is_treated_as_rm(tmp_path, monkeypatch):
    sysdir = tmp_path / "sysbin"
    sysdir.mkdir()
    real = sysdir / "gnurm"
    real.write_text("#!/bin/sh\n")
    real.chmod(0o755)
    (tmp_path / "ls").symlink_to(real)
    monkeypatch.setattr(exec_policy, "_SYSTEM_DIRS", (str(sysdir) + os.sep,))
    cls = classify(_req(["./ls", "-rf", "x"], "model", tmp_path))
    assert cls.decision == "DENY"
    assert cls.tier == "MUTATE"


def test_cwd_outside_root_is_denied(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    cls = classify(_req(["ls"], "human", root, cwd=outside))
    assert cls.decision == "DENY"


def test_symlink_inside_root_that_points_outside_is_denied(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "escape").symlink_to(outside)
    cls = classify(_req(["ls"], "human", root, cwd=root / "escape"))
    assert cls.decision == "DENY"


def test_cwd_that_is_not_a_directory_is_denied(tmp_path):
    cls = classify(_req(["ls"], "human", tmp_path, cwd=tmp_path / "missing"))
    assert cls.decision == "DENY"


def test_secret_path_is_denied_even_for_human(tmp_path):
    assert classify(_req(["cat", ".env"], "human", tmp_path)).decision == "DENY"
    assert classify(_req(["cat", "~/.ssh/id_rsa"], "human", tmp_path)).decision == "DENY"


def test_publish_flag_is_denied_for_agents(tmp_path):
    cls = classify(_req(["python", "-m", "pytest", "--publish"], "model", tmp_path))
    assert cls.decision == "DENY"
    cls = classify(_req(["python", "-m", "pytest", "--publish"], "human", tmp_path))
    assert cls.decision == "ASK"


def test_unwrapped_command_is_recorded_in_normalized_argv(tmp_path):
    cls = classify(_req(["timeout", "5", "rm", "-rf", "x"], "human", tmp_path))
    assert os.path.basename(cls.normalized_argv[0]) == "rm"
    assert cls.normalized_argv[1:] == ("-rf", "x")


def test_normalized_argv_keeps_the_path_not_the_realpath_for_venv_interpreters(tmp_path):
    # The venv path must survive normalization; realpath would drop pyvenv.cfg discovery when exec'd.
    import sys

    cls = classify(_req([sys.executable, "--version"], "model", tmp_path))
    assert cls.normalized_argv[0] == os.path.abspath(sys.executable)
    assert cls.tier == "READ"


def test_classify_is_pure_apart_from_reads(tmp_path):
    before = set(os.listdir(tmp_path))
    classify(_req(["rm", "-rf", "x"], "human", tmp_path))
    classify(_req(["git", "commit", "-m", "m"], "model", tmp_path))
    assert set(os.listdir(tmp_path)) == before


def test_missing_policy_file_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(exec_policy, "_POLICY_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(exec_policy, "_POLICY_CACHE", None)
    with pytest.raises(PolicyError):
        classify(_req(["ls"], "human", tmp_path))
    monkeypatch.setattr(exec_policy, "_POLICY_CACHE", None)


def test_malformed_policy_fails_closed(tmp_path, monkeypatch):
    bad = tmp_path / "bad.json"
    bad.write_text('{"tiers": {"READ": {"human": "ALLOW", "agent": "ALLOW"}}}', encoding="utf-8")
    monkeypatch.setattr(exec_policy, "_POLICY_PATH", bad)
    monkeypatch.setattr(exec_policy, "_POLICY_CACHE", None)
    with pytest.raises(PolicyError):
        exec_policy.load_policy()
    monkeypatch.setattr(exec_policy, "_POLICY_CACHE", None)


def test_scrubbed_env_drops_secret_names_even_when_allowlisted():
    parent = {
        "PATH": "/usr/bin",
        "HOME": "/home/x",
        "LC_ALL": "C",
        "OSIRIS_MODEL": "qwen",
        "QISKIT_IBM_TOKEN": "t",
        "OSIRIS_PG_URL": "postgres://u:p@h/db",
        "GITHUB_TOKEN": "g",
        "AWS_SECRET_ACCESS_KEY": "a",
        "MY_API_KEY": "k",
        "IBM_QUANTUM_KEY": "q",
        "SSH_AUTH_SOCK": "/tmp/agent",
        "RANDOM_VAR": "r",
    }
    env = scrubbed_env(parent)
    assert set(env) == {"PATH", "HOME", "LC_ALL", "OSIRIS_MODEL"}


def test_is_sensitive_argv_flags_only_the_token_command():
    assert is_sensitive_argv(("/usr/bin/gcloud", "auth", "print-access-token"))
    assert not is_sensitive_argv(("/usr/bin/gcloud", "compute", "instances", "list"))
    assert not is_sensitive_argv(("/bin/echo", "print-access-token"))



def test_bare_python_alias_resolves_when_python_is_missing(tmp_path, monkeypatch):
    """python is not installed on this substrate; the PEP 394 alias must still allow a real spawn."""
    sysdir = tmp_path / "aliasbin"
    sysdir.mkdir()
    os.symlink(sys.executable, sysdir / "python3")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "ghost" / "python"))
    monkeypatch.setenv("PATH", str(sysdir))
    cls = classify(_req(["python", "-m", "pytest", "-q"], "model", tmp_path))
    assert cls.tier == "VERIFY", cls.reasons
    assert cls.decision == "ASK", cls.reasons
    assert os.path.basename(cls.normalized_argv[0]) == "python3", cls.normalized_argv
    assert "alias" in cls.reasons[0], cls.reasons


def test_bare_pip_alias_resolves_but_kills_installs(tmp_path, monkeypatch):
    sysdir = tmp_path / "aliasbin"
    sysdir.mkdir()
    os.symlink(sys.executable, sysdir / "python3")
    os.symlink(sys.executable, sysdir / "pip3")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "ghost" / "python"))
    monkeypatch.setenv("PATH", str(sysdir))
    cls = classify(_req(["pip", "install", "something"], "model", tmp_path))
    assert cls.decision == "DENY", cls.reasons  # alias never smuggles a network install through
    assert os.path.basename(cls.normalized_argv[0]) == "pip3", cls.normalized_argv


def test_alias_not_applied_when_python_exists(tmp_path, monkeypatch):
    sysdir = tmp_path / "aliasbin"
    sysdir.mkdir()
    fake = sysdir / "python"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "ghost" / "python"))
    monkeypatch.setenv("PATH", str(sysdir))
    cls = classify(_req(["python", "-V"], "model", tmp_path))
    assert os.path.basename(cls.normalized_argv[0]) == "python", cls.normalized_argv
    assert not any("alias" in r for r in cls.reasons), cls.reasons
    assert cls.tier == "READ", cls.reasons


def test_alias_cannot_decoy_a_python3_named_binary(tmp_path, monkeypatch):
    """A same-named wrapper is judged by its own name and chain, not by the alias."""
    sysdir = tmp_path / "aliasbin"
    sysdir.mkdir()
    decoy = sysdir / "python3"
    decoy.write_text("#!/bin/sh\nexit 0\n")
    decoy.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "ghost" / "python"))
    monkeypatch.setenv("PATH", str(sysdir))
    cls = classify(_req(["python", "-m", "pytest", "-q"], "model", tmp_path))
    assert cls.normalized_argv[0] == str(decoy), cls.normalized_argv  # resolved, judged, kept
    assert cls.tier == "VERIFY", cls.reasons  # judge by what it is named and does


def test_unresolvable_verifier_fails_closed(tmp_path, monkeypatch):
    """An agent-requested verifier that exists nowhere is refused instead of approved then dead."""
    monkeypatch.delenv("OSIRIS_PROJECT_PYTHON", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "nothing"))
    cls = exec_policy.classify(CommandRequest(
        argv=("definitely-not-installed-verifier", "-q"), cwd=str(tmp_path),
        origin="model", root=str(tmp_path)))
    assert cls.decision == "DENY", (cls.tier, cls.reasons)
    assert cls.tier == "FORBIDDEN", (cls.tier, cls.reasons)
    assert "not installed on this system (checked PATH" in cls.reasons[0], cls.reasons


def test_unresolvable_verifier_is_not_blocked_for_humans(tmp_path, monkeypatch):
    """A human at a TTY keeps the historical behavior; the spawn error is its own feedback."""
    monkeypatch.delenv("OSIRIS_PROJECT_PYTHON", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "nothing"))
    cls = exec_policy.classify(CommandRequest(
        argv=("definitely-not-installed-verifier", "-q"), cwd=str(tmp_path),
        origin="human", root=str(tmp_path)))
    assert cls.tier != "FORBIDDEN", (cls.tier, cls.reasons)


def test_active_interpreters_own_environment_is_a_resolution_source(tmp_path, monkeypatch):
    """pytest is not on PATH and the workspace has no venv, but the environment running
    OSIRIS itself ships pytest (as production always does)."""
    monkeypatch.delenv("OSIRIS_PROJECT_PYTHON", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "nothing"))
    cls = exec_policy.classify(CommandRequest(
        argv=("pytest", "-q"), cwd=str(tmp_path), origin="model", root=str(tmp_path)))
    assert cls.tier == "VERIFY", cls.reasons
    assert "resolved pytest from the running interpreter's environment" in cls.reasons[0], cls.reasons


def test_unlisted_program_can_still_fail_at_spawn(tmp_path, monkeypatch):
    """Unknown bare programs keep the historical behavior: classified, honestly dead at spawn."""
    monkeypatch.setenv("PATH", str(tmp_path / "nothing"))
    cls = classify(_req(["totally-absent-tool", "--run"], "model", tmp_path))
    assert cls.decision != "DENY" or "not installed" not in " ".join(cls.reasons)


def _make_venv(root, interp):
    bind = root / ".venv" / "bin"
    bind.mkdir(parents=True)
    (bind / "python").symlink_to(interp)
    return bind / "python"


def test_workspace_venv_interpreter_is_preferred(tmp_path, monkeypatch):
    """Bare `python` in a workspace with a .venv must resolve to that venv (its deps)."""
    monkeypatch.delenv("OSIRIS_PROJECT_PYTHON", raising=False)
    venv_py = _make_venv(tmp_path, sys.executable)
    (tmp_path / "src").mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / "nothing"))
    cls = exec_policy.classify(CommandRequest(
        argv=("python", "-m", "pytest", "-q"), cwd=str(tmp_path / "src"),
        origin="model", root=str(tmp_path)))
    assert Path(cls.normalized_argv[0]) == venv_py, cls.normalized_argv
    assert "workspace virtualenv" in cls.reasons[0], cls.reasons
    assert cls.tier == "VERIFY", cls.reasons


def test_configured_project_interpreter_wins(tmp_path, monkeypatch):
    monkeypatch.delenv("OSIRIS_PROJECT_PYTHON", raising=False)
    monkeypatch.setenv("OSIRIS_PROJECT_PYTHON", str(sys.executable))
    monkeypatch.setenv("PATH", str(tmp_path / "nothing"))
    cls = exec_policy.classify(CommandRequest(
        argv=("python", "-V"), cwd=str(tmp_path), origin="human", root=str(tmp_path)))
    assert Path(cls.normalized_argv[0]) == Path(sys.executable), cls.normalized_argv
    assert "OSIRIS_PROJECT_PYTHON" in cls.reasons[0], cls.reasons
    monkeypatch.delenv("OSIRIS_PROJECT_PYTHON")


def test_version_pinned_interpreter_falls_back_to_installed_sibling(tmp_path, monkeypatch):
    """python3.12 is not installed here; the documented sibling fallback keeps it runnable."""
    monkeypatch.delenv("OSIRIS_PROJECT_PYTHON", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "nothing"))
    bin = tmp_path / "bin"
    bin.mkdir()
    os.symlink(sys.executable, bin / "python3")
    monkeypatch.setenv("PATH", str(bin))
    cls = exec_policy.classify(CommandRequest(
        argv=("python3.12", "--version"), cwd=str(tmp_path), origin="model", root=str(tmp_path)))
    assert os.path.basename(cls.normalized_argv[0]) == "python3", cls.normalized_argv
    assert "python3.12" in cls.reasons[0], cls.reasons
    # A pinned interpreter required a sibling: the receipt says so.
    assert "resolved to the installed" in cls.reasons[0], cls.reasons


def test_venv_python_symlinked_to_forbidden_target_is_still_denied(tmp_path, monkeypatch):
    """A .venv python that is really a renamed destructive tool is judged by its chain."""
    bind = tmp_path / ".venv" / "bin"
    bind.mkdir(parents=True)
    weapon = tmp_path / "gnurm"
    weapon.write_text("#!/bin/sh" + chr(10) + "exit 7" + chr(10))
    weapon.chmod(0o755)
    os.symlink(weapon, bind / "python")
    monkeypatch.delenv("OSIRIS_PROJECT_PYTHON", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "nothing"))
    cls = exec_policy.classify(CommandRequest(
        argv=("python", "-m", "pytest", "-q"), cwd=str(tmp_path), origin="model", root=str(tmp_path)))
    # VERIFY from the bare name merged with the chain's unknown gnu-rm -> MUTATE -> deny
    assert cls.decision == "DENY", (cls.tier, cls.reasons)
    assert cls.tier == "MUTATE", cls.reasons
