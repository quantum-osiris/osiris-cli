"""Sprint 1: durable run lifecycle, crash/resume idempotency, cancel, critic, REPL --accept.

Models are scripted; acceptance commands are real subprocesses in a temporary repo.
"""

import json
import os
import subprocess
import sys

import pytest

from osiris_cli import code_agent, code_critic
from osiris_cli.code_runs import RunError, RunStore
from osiris_cli.code_task import main as task_main, resume, run_task

BUGGY = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"
TEST = "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
ACCEPT = f"{sys.executable} -m pytest -q -p no:cacheprovider test_calc.py"


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "calc.py").write_text(BUGGY)
    (root / "util.py").write_text("X = 1\n")
    (root / "test_calc.py").write_text(TEST)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "b"], cwd=root, check=True)
    monkeypatch.delenv("OSIRIS_TEXT_TOOL_CALLS", raising=False)
    return root


@pytest.fixture
def store(tmp_path):
    return RunStore(str(tmp_path / "runs"))


def call(name, **args):
    return {"content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def scripted(*steps, final="done"):
    script = list(steps) + [{"content": final}]
    return lambda messages, tools, model: script.pop(0) if script else {"content": final}


def go(repo, store, *steps, **kw):
    kw.setdefault("accept", ACCEPT)
    kw.setdefault("receipts", str(repo.parent / "receipts.jsonl"))
    kw.setdefault("confirm", lambda prompt: False)
    return run_task("fix add", root=str(repo), chat=scripted(*steps), out=lambda s: None, store=store, **kw)


def states(summary):
    return [h["state"] for h in summary["history"] if "state" in h]


def receipts(repo):
    p = repo.parent / "receipts.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


# ── lifecycle ────────────────────────────────────────────────────────────────────

def test_full_lifecycle_is_recorded_and_durable(repo, store):
    s = go(repo, store, call("write_file", path="calc.py", content=FIXED), approve_writes=True)
    assert s["state"] == "SUCCEEDED" and s["disposition"] == "VERIFIED"
    assert states(s) == ["CREATED", "CONTEXT_READY", "RUNNING", "AWAITING_APPROVAL", "VERIFYING", "SUCCEEDED"]
    on_disk = store.load(s["run_id"])
    assert on_disk["state"] == "SUCCEEDED" and on_disk["source"]["git_head"]
    assert on_disk["critic"]["verdict"] == "PASS"


def test_retry_is_a_recorded_event(repo, store):
    s = go(repo, store, {"content": "```diff\n+ return a + b\n```"},
           call("write_file", path="calc.py", content=FIXED), approve_writes=True)
    assert s["state"] == "SUCCEEDED"
    assert [h["event"] for h in s["history"] if "event" in h] == ["RETRYING"]


def test_illegal_transition_refused(repo, store):
    run = store.create("code-x", root=str(repo))
    with pytest.raises(RunError):
        store.transition(run, "SUCCEEDED")


def test_unavailable_model_is_unavailable_not_failed(repo, store):
    s = go(repo, store, {"error": "DAEMON_UNAVAILABLE: refused"})
    assert s["state"] == "UNAVAILABLE"


# ── crash and resume ─────────────────────────────────────────────────────────────

def test_resume_after_crash_awaiting_approval(repo, store):
    s = go(repo, store, call("write_file", path="calc.py", content=FIXED), stop_after="AWAITING_APPROVAL")
    assert s["state"] == "AWAITING_APPROVAL" and (repo / "calc.py").read_text() == BUGGY
    r = resume(s["run_id"], store=store, approve_writes=True, out=lambda x: None)
    assert r["state"] == "SUCCEEDED" and (repo / "calc.py").read_text() == FIXED
    assert [h["event"] for h in r["history"] if "event" in h] == ["RESUMED"]


def test_resume_mid_apply_does_not_rewrite_or_double_receipt(repo, store):
    s = go(repo, store, call("write_file", path="calc.py", content=FIXED),
           call("write_file", path="util.py", content="X = 2\n"), multi=True, stop_after="AWAITING_APPROVAL")
    # crash after the first file was written but before the state moved on
    (repo / "calc.py").write_text(FIXED)
    mtime = os.stat(repo / "calc.py").st_mtime_ns
    r = resume(s["run_id"], store=store, approve_writes=True, out=lambda x: None)
    assert r["state"] == "SUCCEEDED"
    by_path = {a["path"]: a for a in r["applied"]}
    assert by_path["calc.py"]["already_applied"] is True and by_path["util.py"]["already_applied"] is False
    assert os.stat(repo / "calc.py").st_mtime_ns == mtime  # not written a second time
    writes = [x for x in receipts(repo) if x["kind"] == "code.agent.apply_write"]
    assert sorted(w["payload"]["path"] for w in writes) == ["calc.py", "util.py"]


def test_resume_refuses_a_target_changed_by_someone_else(repo, store):
    s = go(repo, store, call("write_file", path="calc.py", content=FIXED), stop_after="AWAITING_APPROVAL")
    (repo / "calc.py").write_text("def add(a, b):\n    return b + a  # human edit\n")
    r = resume(s["run_id"], store=store, approve_writes=True, out=lambda x: None)
    assert r["state"] == "BLOCKED" and "human edit" in (repo / "calc.py").read_text()


def test_crash_in_running_resumes_to_failed_with_no_write(repo, store):
    s = go(repo, store, call("write_file", path="calc.py", content=FIXED), stop_after="RUNNING")
    r = resume(s["run_id"], store=store, approve_writes=True, out=lambda x: None)
    assert r["state"] == "FAILED" and "nothing was written" in r["history"][-1]["reason"]
    assert (repo / "calc.py").read_text() == BUGGY


def test_terminal_run_cannot_be_resumed_or_rerun(repo, store):
    s = go(repo, store, call("write_file", path="calc.py", content=FIXED), approve_writes=True)
    n = len(receipts(repo))
    with pytest.raises(RunError):
        resume(s["run_id"], store=store, approve_writes=True)
    assert len(receipts(repo)) == n


def test_resume_uses_stored_acceptance_command(repo, store, capsys):
    s = go(repo, store, call("write_file", path="calc.py", content="def add(a, b):\n    return 0\n"),
           stop_after="AWAITING_APPROVAL")
    # a resumer cannot swap in a weaker acceptance command: CLI has no --accept on resume path
    rc = task_main(["--resume", s["run_id"], "--approve-writes", "--accept", "true", "--runs-dir", store.dir])
    assert rc == 1 and json.loads(capsys.readouterr().out)["state"] == "FAILED"


def test_concurrent_resume_is_refused(repo, store):
    s = go(repo, store, call("write_file", path="calc.py", content=FIXED), stop_after="AWAITING_APPROVAL")
    with store.locked(s["run_id"]):
        with pytest.raises(RunError, match="another process"):
            resume(s["run_id"], store=store, approve_writes=True)


# ── cancel ───────────────────────────────────────────────────────────────────────

def test_cancel_before_approval_writes_nothing(repo, store):
    s = go(repo, store, call("write_file", path="calc.py", content=FIXED), stop_after="AWAITING_APPROVAL")
    store.request_cancel(s["run_id"])
    r = resume(s["run_id"], store=store, approve_writes=True, out=lambda x: None)
    assert r["state"] == "CANCELLED" and (repo / "calc.py").read_text() == BUGGY


def test_cancel_during_verification_is_never_promoted(repo, store, monkeypatch):
    from osiris_cli import code_task
    real = code_task.verify_applied

    def verify_then_cancel(*a, **kw):
        out = real(*a, **kw)
        store.request_cancel(kw["run_id"])
        return out
    monkeypatch.setattr(code_task, "verify_applied", verify_then_cancel)
    s = go(repo, store, call("write_file", path="calc.py", content=FIXED), approve_writes=True)
    assert s["verification"]["decision"] == "VERIFIED" and s["state"] == "CANCELLED"


def test_cancel_terminal_run_refused(repo, store):
    s = go(repo, store, call("write_file", path="calc.py", content=FIXED), approve_writes=True)
    with pytest.raises(RunError):
        store.request_cancel(s["run_id"])


# ── critic ───────────────────────────────────────────────────────────────────────

def test_critic_blocks_test_edit_before_any_write(repo, store):
    weakened = "def test_add():\n    assert True\n"
    s = go(repo, store, call("write_file", path="test_calc.py", content=weakened), approve_writes=True)
    assert s["state"] == "BLOCKED" and s["critic"]["findings"][0]["check"] == "modifies_tests"
    assert (repo / "test_calc.py").read_text() == TEST


def test_critic_blocks_protected_path(repo, store):
    s = go(repo, store, call("write_file", path=".github/workflows/ci.yml", content="on: push\n"),
           approve_writes=True)
    assert s["state"] == "BLOCKED" and not (repo / ".github").exists()


def test_critic_blocks_introduced_eval_but_not_preexisting(repo, store):
    s = go(repo, store, call("write_file", path="calc.py", content="def add(a, b):\n    return eval('a+b')\n"),
           approve_writes=True)
    assert s["state"] == "BLOCKED" and "eval" in s["critic"]["findings"][0]["reason"]
    pre = [{"rel_path": "m.py", "content": "eval('1')\nx = 2\n", "before_content": "eval('1')\nx = 1\n"}]
    assert code_critic.deterministic_findings(pre, accept=None) == []


def test_critic_blocks_syntax_error(repo, store):
    s = go(repo, store, call("write_file", path="calc.py", content="def add(a, b)\n  return\n"), approve_writes=True)
    assert s["state"] == "BLOCKED" and s["critic"]["findings"][0]["check"] == "python_syntax"


def test_model_critic_reject_blocks(repo, store):
    s = go(repo, store, call("write_file", path="calc.py", content=FIXED), approve_writes=True,
           critic_model="qwen3:8b",
           critic_chat=lambda m, t, model: {"content": '{"verdict": "REJECT", "reasons": ["wrong"]}'})
    assert s["state"] == "BLOCKED" and (repo / "calc.py").read_text() == BUGGY


def test_model_critic_approve_still_needs_tests(repo, store):
    s = go(repo, store, call("write_file", path="calc.py", content="def add(a, b):\n    return 0\n"),
           approve_writes=True, critic_model="qwen3:8b",
           critic_chat=lambda m, t, model: {"content": '<think>x</think>{"verdict": "APPROVE", "reasons": []}'})
    assert s["critic"]["verdict"] == "PASS" and s["state"] == "FAILED"


def test_model_critic_invalid_json_retries_once_then_blocks(repo, store):
    calls = []

    def bad(m, t, model):
        calls.append(1)
        return {"content": "looks fine to me"}
    s = go(repo, store, call("write_file", path="calc.py", content=FIXED), approve_writes=True,
           critic_model="qwen3:8b", critic_chat=bad)
    assert s["state"] == "BLOCKED" and len(calls) == 2


def test_model_critic_must_differ_from_author(repo, store):
    s = go(repo, store, call("write_file", path="calc.py", content=FIXED), approve_writes=True,
           model=None, critic_model=code_agent.DEFAULT_MODEL,
           critic_chat=lambda m, t, model: {"content": '{"verdict": "APPROVE", "reasons": []}'})
    assert s["state"] == "BLOCKED" and "must differ" in s["critic"]["findings"][0]["reason"]


def test_model_critic_missing_model_blocks(repo, store):
    s = go(repo, store, call("write_file", path="calc.py", content=FIXED), approve_writes=True,
           critic_model="nope:1b", critic_installed={"qwen3:8b": "d"})
    assert s["state"] == "BLOCKED" and "MODEL_NOT_INSTALLED" in s["critic"]["findings"][0]["reason"]


# ── REPL /code --accept ──────────────────────────────────────────────────────────

class _State:
    log_event = None


def test_repl_accept_runs_after_console_apply(repo, monkeypatch, tmp_path):
    monkeypatch.chdir(repo)
    monkeypatch.setattr(code_agent, "REPL_RECEIPTS", str(tmp_path / "repl.jsonl"))
    staged = {}
    monkeypatch.setattr(code_agent, "_otc_propose_write", lambda p, c: staged.update(path=p, content=c) or True)
    monkeypatch.setattr(code_agent, "_otc_has_pending_write", lambda: bool(staged))
    script = [call("write_file", path="calc.py", content=FIXED), {"content": "done"}]
    monkeypatch.setattr(code_agent.CodeAgent, "_preflight_live_model", lambda self: None)
    monkeypatch.setattr(code_agent.CodeAgent, "_ollama_chat", lambda self, m, t, model: script.pop(0))
    code_agent.run_code_command(_State(), f'--accept "{ACCEPT}" fix add')
    assert code_agent._repl_acceptance is not None
    (repo / "calc.py").write_text(staged["content"])  # what the console's /apply does
    v = code_agent.after_console_apply()
    assert v["decision"] == "VERIFIED" and v["receipt_hash"] and code_agent._repl_acceptance is None


def test_repl_accept_not_run_when_apply_refused(repo, monkeypatch, tmp_path):
    monkeypatch.chdir(repo)
    monkeypatch.setattr(code_agent, "REPL_RECEIPTS", str(tmp_path / "repl.jsonl"))
    monkeypatch.setattr(code_agent, "_otc_propose_write", lambda p, c: True)
    monkeypatch.setattr(code_agent, "_otc_has_pending_write", lambda: False)
    script = [call("write_file", path="calc.py", content=FIXED), {"content": "done"}]
    monkeypatch.setattr(code_agent.CodeAgent, "_preflight_live_model", lambda self: None)
    monkeypatch.setattr(code_agent.CodeAgent, "_ollama_chat", lambda self, m, t, model: script.pop(0))
    code_agent.run_code_command(_State(), f"--accept '{ACCEPT}' fix add")
    assert code_agent.after_console_apply() is None  # file still BUGGY: nothing verified


def test_repl_accept_without_write_reports_failed(repo, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(repo)
    monkeypatch.setattr(code_agent, "REPL_RECEIPTS", str(tmp_path / "repl.jsonl"))
    script = [{"content": "I fixed calc.py."}, {"content": "I fixed calc.py."}]
    monkeypatch.setattr(code_agent.CodeAgent, "_preflight_live_model", lambda self: None)
    monkeypatch.setattr(code_agent.CodeAgent, "_ollama_chat", lambda self, m, t, model: script.pop(0))
    code_agent.run_code_command(_State(), f'--accept "{ACCEPT}" fix add')
    out = capsys.readouterr().out
    assert "[/code FAILED]" in out and "no applied change" in out


# ── CLI ──────────────────────────────────────────────────────────────────────────

def test_cli_list_status_cancel(repo, store, capsys):
    s = go(repo, store, call("write_file", path="calc.py", content=FIXED), stop_after="AWAITING_APPROVAL")
    assert task_main(["--list", "--runs-dir", store.dir]) == 0
    assert json.loads(capsys.readouterr().out)[0]["state"] == "AWAITING_APPROVAL"
    assert task_main(["--cancel", s["run_id"], "--runs-dir", store.dir]) == 0
    capsys.readouterr()
    assert task_main(["--resume", s["run_id"], "--runs-dir", store.dir]) == 1
    assert json.loads(capsys.readouterr().out)["state"] == "CANCELLED"
    assert task_main(["--status", s["run_id"], "--runs-dir", store.dir]) == 0
