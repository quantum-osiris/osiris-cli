"""/trio front end: dry-run default, digest-bound approval, no apply, no promote, no network, no model.
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


def digest_of(ws):
    out = cmd(ws, "plan", ws["repo"], "--patch", ws["patch"])
    return out.split("plan digest")[1].split()[0]


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


def test_run_without_or_with_wrong_approval_is_refused_and_executes_nothing(ws, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("executed")))
    for approve in ([], ["--approve", "0" * 64], ["--approve", "deadbeef"]):
        with pytest.raises(tc.TrioCommandError, match="approval"):
            cmd(ws, "run", ws["repo"], "--patch", ws["patch"], "--out", ws["out"], *approve)
    assert not os.path.exists(ws["out"])


def test_approval_is_invalidated_when_the_repo_or_patch_changes(ws):
    dg = digest_of(ws)
    open(os.path.join(ws["repo"], "calc.py"), "a").write("# changed after review\n")
    with pytest.raises(tc.TrioCommandError, match="does not match"):
        cmd(ws, "run", ws["repo"], "--patch", ws["patch"], "--out", ws["out"], "--approve", dg)
    dg2 = digest_of(ws)
    open(ws["patch"], "w").write(json.dumps({"patched_files": {"calc.py": FIXED + "#\n"}, "report": "x"}))
    with pytest.raises(tc.TrioCommandError, match="does not match"):
        cmd(ws, "run", ws["repo"], "--patch", ws["patch"], "--out", ws["out"], "--approve", dg2)


def test_approved_run_records_but_never_applies_or_promotes_or_uses_network(ws, monkeypatch):
    dg = digest_of(ws)
    monkeypatch.setattr(socket, "socket", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    before = tree_digest(ws["repo"])
    text = cmd(ws, "run", ws["repo"], "--patch", ws["patch"], "--out", ws["out"], "--approve", dg)
    assert "VERIFIED_PENDING_AUTHORIZATION" in text and "nothing promoted" in text
    assert tree_digest(ws["repo"]) == before and open(os.path.join(ws["repo"], "calc.py")).read() == BUGGY
    rows = [json.loads(line) for line in open(os.path.join(ws["out"], "lineage.jsonl"))]
    assert [r["kind"] for r in rows] == ["CANDIDATE_VERIFIED_PENDING_AUTHORIZATION"]
    assert rows[0]["payload"]["applied_to_repo"] is False


def test_unsupported_claims_in_the_patch_report_are_quarantined_through_the_command(ws):
    open(ws["patch"], "w").write(json.dumps({"patched_files": {"calc.py": FIXED},
                                             "report": "Fixed. Accuracy up 37.2% on ibm_torino."}))
    dg = digest_of(ws)
    text = cmd(ws, "run", ws["repo"], "--patch", ws["patch"], "--out", ws["out"], "--approve", dg)
    assert "QUARANTINED" in text


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


def test_run_requires_explicit_out_outside_the_repo_and_a_patch(ws):
    dg = digest_of(ws)
    with pytest.raises(tc.TrioCommandError, match="explicit --out"):
        cmd(ws, "run", ws["repo"], "--patch", ws["patch"], "--approve", dg)
    with pytest.raises(tc.TrioCommandError, match="inside the repo"):
        cmd(ws, "run", ws["repo"], "--patch", ws["patch"], "--out", os.path.join(ws["repo"], "o"), "--approve", dg)
    with pytest.raises(tc.TrioCommandError, match="--patch"):
        cmd(ws, "run", ws["repo"], "--out", ws["out"], "--approve", dg)


def test_the_live_repl_does_not_know_about_trio_yet():
    src = open(os.path.join(os.path.dirname(tc.__file__), "osiris_repl.py"), encoding="utf-8").read()
    assert "trio_command" not in src and '"/trio"' not in src
