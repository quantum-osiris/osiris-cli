"""/trio front end: dry-run default; planning, parsing and REPL checks (authorization: test_trio_authz.py).
Not wired into the live REPL; these tests call the command function directly."""

import json
import os
import socket
import subprocess

import pytest

from osiris_cli import trio_command as tc
from osiris_cli.orchestration.trio import tree_digest

BUGGY = "def mean(xs):\n    return sum(xs) / (len(xs) + 1)\n"
FIXED = "def mean(xs):\n    return sum(xs) / len(xs)\n"
TESTS = "from calc import mean\n\ndef test_mean():\n    assert mean([2, 4]) == 3\n"


@pytest.fixture
def ws(tmp_path):
    repo = tmp_path / "work" / "repo"
    repo.mkdir(parents=True)
    (repo / "calc.py").write_text(BUGGY)
    (repo / "test_calc.py").write_text(TESTS)
    patch = tmp_path / "work" / "patch.json"
    patch.write_text(json.dumps({"patched_files": {"calc.py": FIXED}, "report": "Changed the divisor."}))
    return {"root": str(tmp_path / "work"), "repo": str(repo), "patch": str(patch), "out": str(tmp_path / "out"),
            "tmp": tmp_path}


def cmd(ws, *a):
    return tc.run_trio_command(list(a), workspace_root=ws["root"])


def test_plan_is_a_pure_dry_run(ws, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("dry run must not spawn processes or open sockets")
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    monkeypatch.setattr(socket, "socket", boom)
    before = tree_digest(ws["root"])
    text = cmd(ws, "plan", ws["repo"], "--patch", ws["patch"])
    assert "dry run" in text and "plan digest" in text and "[needs human approval]" in text   # critic: SUBPROCESS_EXEC
    assert tree_digest(ws["root"]) == before and not os.path.exists(ws["out"])
    assert cmd(ws, ws["repo"]).startswith("/trio plan")            # bare form defaults to plan


def test_only_provider_none_and_no_model_calls(ws):
    with pytest.raises(tc.TrioCommandError, match="no model integration"):
        tc.build_plan(ws["repo"], ws["patch"], workspace_root=ws["root"], provider="gemini")


def test_confinement_and_input_checks(ws):
    with pytest.raises(tc.TrioCommandError, match="inside the workspace"):
        tc.run_trio_command(["plan", str(ws["tmp"])], workspace_root=ws["repo"])
    with pytest.raises(tc.TrioCommandError, match="not a directory"):
        cmd(ws, "plan", os.path.join(ws["root"], "nope"))
    os.symlink("/etc/hostname", os.path.join(ws["repo"], "link.txt"))
    with pytest.raises(tc.TrioCommandError, match="symlink"):
        cmd(ws, "plan", ws["repo"])


def test_bad_patch_files_and_options(ws):
    for body in ('{"patched_files": {}, "report": "x"}', '{"patched_files": {"a.py": 1}, "report": "x"}',
                 '{"patched_files": {"a.py": "x"}, "report": "x", "apply": true}', "not json"):
        open(ws["patch"], "w").write(body)
        with pytest.raises(tc.TrioCommandError):
            cmd(ws, "plan", ws["repo"], "--patch", ws["patch"])
    with pytest.raises(tc.TrioCommandError, match="unknown option"):
        cmd(ws, "plan", ws["repo"], "--apply")
    assert cmd(ws) == tc.USAGE


def test_repl_trio_is_plan_only_and_never_runs(ws, monkeypatch, capsys):
    from osiris_cli import osiris_repl
    monkeypatch.chdir(ws["root"])
    monkeypatch.setattr(tc, "run_trio_command", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called")) if a and a[0] and a[0][0] == "run" else "PLAN-OK")
    osiris_repl.execute_trio("run %s --patch %s --out %s --approve x" % (ws["repo"], ws["patch"], ws["out"]))
    out = capsys.readouterr().out
    assert "disabled in the REPL" in out and not os.path.exists(ws["out"])
    osiris_repl.execute_trio("plan %s" % ws["repo"])
    assert "PLAN-OK" in capsys.readouterr().out
    monkeypatch.undo()
    monkeypatch.chdir(ws["root"])
    osiris_repl.execute_trio("plan /nonexistent/x")
    assert "[!]" in capsys.readouterr().out
