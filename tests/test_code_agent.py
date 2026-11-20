"""/code: the governed tool loop. The model is always scripted; nothing reaches Ollama."""

import io
import json
import sys
from unittest.mock import patch

import pytest

from osiris_cli import code_agent
from osiris_cli.code_agent import CodeAgent, parse_text_tool_calls


@pytest.fixture
def ws(tmp_path, monkeypatch):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "notes.txt").write_text("theta_lock = 51.843\n")
    (tmp_path / "secret.txt").write_text("SECRET-OUTSIDE\n")
    monkeypatch.chdir(root)
    monkeypatch.delenv("OSIRIS_TEXT_TOOL_CALLS", raising=False)
    return root


def agent(ws, *steps, **kw):
    """A CodeAgent whose model replays `steps` (dicts) and then answers 'done'."""
    script = list(steps) + [{"content": "done"}]
    seen = []

    def chat(messages, tools, model):
        seen.append([dict(m) for m in messages])
        return script.pop(0) if script else {"content": "done"}

    kw.setdefault("confirm", lambda prompt: False)
    kw.setdefault("propose_write", lambda path, content: True)
    kw.setdefault("has_pending_write", lambda: False)
    a = CodeAgent(root=str(ws), chat=chat, out=lambda s: None, **kw)
    a.seen = seen
    return a


def call(name, **args):
    return {"content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def tool_results(a):
    return [m["content"] for m in a.seen[-1] if m["role"] == "tool"]


# ── reads ──────────────────────────────────────────────────────────────────────

def test_read_inside_the_working_directory(ws):
    a = agent(ws, call("read_file", path="notes.txt"))
    result = a.run("read it")
    assert result.status == "SUCCESS" and result.answer == "done"
    assert tool_results(a) == ["theta_lock = 51.843\n"]
    assert a.decisions[0]["decision"] == "GRANTED"


@pytest.mark.parametrize("path", ["../secret.txt", "ABS"])
def test_read_outside_is_denied(ws, path):
    if path == "ABS":
        path = str(ws.parent / "secret.txt")
    a = agent(ws, call("read_file", path=path))
    a.run("read it")
    shown = tool_results(a)[0]
    assert shown.startswith("DENIED") and "SECRET-OUTSIDE" not in shown


def test_symlink_escape_is_denied(ws):
    (ws / "link.txt").symlink_to(ws.parent / "secret.txt")
    a = agent(ws, call("read_file", path="link.txt"))
    a.run("read it")
    assert "outside the working directory" in tool_results(a)[0]


def test_grep_pattern_never_reaches_a_shell(ws):
    a = agent(ws, call("grep_code", pattern="it's $(touch PWNED)", path="."))
    a.run("search")
    assert not (ws / "PWNED").exists()


def test_list_dir_and_grep_return_workspace_relative_results(ws):
    a = agent(ws, call("list_dir"), call("grep_code", pattern="theta"))
    a.run("look")
    listing, matches = tool_results(a)
    assert "notes.txt" in listing
    assert matches.startswith("notes.txt:1:")


# ── writes: proposed only, through the REPL's own path ─────────────────────────

def test_write_is_proposed_never_written(ws):
    proposed = []
    a = agent(ws, call("write_file", path="out.py", content="x = 1\n"),
              propose_write=lambda p, c: proposed.append((p, c)) or True)
    a.run("write it")
    assert proposed == [("out.py", "x = 1\n")]
    assert not (ws / "out.py").exists()
    assert tool_results(a)[0].startswith("PROPOSED")


def test_only_one_write_per_task_and_none_over_a_pending_write(ws):
    a = agent(ws, call("write_file", path="a.py", content="1"), call("write_file", path="b.py", content="2"))
    a.run("two writes")
    assert tool_results(a)[1].startswith("DENIED: a write is already pending")
    b = agent(ws, call("write_file", path="c.py", content="3"), has_pending_write=lambda: True)
    b.run("over a pending one")
    assert tool_results(b)[0].startswith("DENIED: a write is already pending")


def test_write_outside_is_refused_before_proposing(ws):
    proposed = []
    a = agent(ws, call("write_file", path="../evil.py", content="x"),
              propose_write=lambda p, c: proposed.append(p) or True)
    a.run("escape")
    assert proposed == [] and tool_results(a)[0].startswith("DENIED")


# ── commands: one program, approved per call, granted and verified ─────────────

def test_command_needs_approval(ws):
    (ws / "touch_ran.py").write_text("open('RAN', 'w').write('x')\n", encoding="utf-8")
    a = agent(ws, call("run_command", command=f"{sys.executable} touch_ran.py"))
    a.run("run it")
    assert "did not approve" in tool_results(a)[0]
    assert not (ws / "RAN").exists()


def test_approved_command_runs_under_a_grant(ws):
    (ws / "say_ok.py").write_text("print('GOVERNED_OK')\n", encoding="utf-8")
    prompts = []
    a = agent(ws, call("run_command", command=f"{sys.executable} say_ok.py"),
              confirm=lambda p: prompts.append(p) or True)
    a.run("run it")
    assert "GOVERNED_OK" in tool_results(a)[0]
    assert prompts and "Run `" in prompts[0]
    d = a.decisions[0]
    assert d["decision"] == "GRANTED" and d["grant_id"].startswith("grant-") and len(d["digest"]) == 64


@pytest.mark.parametrize("command", ["ls | head", "echo a && echo b", "cat x > y", "echo $(id)"])
def test_shell_syntax_is_refused_without_asking(ws, command):
    prompts = []
    a = agent(ws, call("run_command", command=command), confirm=lambda p: prompts.append(p) or True)
    a.run("shell")
    assert prompts == [] and tool_results(a)[0].startswith("DENIED")


def test_command_cwd_outside_is_refused_without_asking(ws):
    prompts = []
    a = agent(ws, call("run_command", command="ls", cwd=".."), confirm=lambda p: prompts.append(p) or True)
    a.run("escape")
    assert prompts == [] and "outside the working directory" in tool_results(a)[0]


def test_command_obeys_the_interactive_horizon(ws, monkeypatch):
    monkeypatch.setenv("OSIRIS_INTERACTIVE_TIMEOUT_S", "1")
    (ws / "slow_cmd.py").write_text("import time\ntime.sleep(5)\n", encoding="utf-8")
    a = agent(ws, call("run_command", command=f"{sys.executable} slow_cmd.py"),
              confirm=lambda p: True)
    a.run("slow")
    assert tool_results(a)[0] == "TIMEOUT after 1s"


# ── text tool calls ────────────────────────────────────────────────────────────

LIVE_REPLY = (
    '{"name": "read_file", "arguments": {"path": "notes.txt"}}\n\nNow I will write it.\n\n'
    '{"name": "write_file", "arguments": {"path": "out.txt", "content": <read_content>}}'
)


def test_text_calls_parse_valid_and_drop_malformed():
    calls = parse_text_tool_calls(LIVE_REPLY)
    assert [c["function"]["name"] for c in calls] == ["read_file"]


@pytest.mark.parametrize("text", ["plain answer", '{"name": "rm_rf", "arguments": {}}',
                                  '{"name": "read_file", "arguments": ["x"]}', "{broken"])
def test_non_tool_text_yields_nothing(text):
    assert parse_text_tool_calls(text) == []


def test_coder_model_text_calls_are_governed_like_native(ws):
    a = agent(ws, {"content": LIVE_REPLY}, model="qwen2.5-coder:7b")
    a.run("read then write")
    assert tool_results(a) == ["theta_lock = 51.843\n"]
    assert a.decisions[0]["origin"] == "text"


def test_other_models_never_execute_json_they_mention(ws):
    mention = 'call {"name": "read_file", "arguments": {"path": "notes.txt"}}'
    a = agent(ws, {"content": mention}, model="qwen2.5:7b")
    result = a.run("explain")
    assert result.answer == mention and a.decisions == []


# ── loop edges ─────────────────────────────────────────────────────────────────

def test_model_error_is_reported(ws):
    a = agent(ws, {"error": "ollama down"})
    result = a.run("x")
    assert result.status == "ERROR" and result.error == "ollama down"


def test_step_budget(ws):
    a = agent(ws, *[call("list_dir")] * 3, max_steps=2)
    assert a.run("loop").status == "MAX_STEPS"


def test_decisions_are_logged(ws):
    events = []
    a = agent(ws, call("read_file", path="../secret.txt"), log=lambda e, d: events.append((e, d)))
    a.run("x")
    assert events and events[0][0] == "CODE_TOOL" and events[0][1]["decision"] == "DENIED"


# ── REPL and provider wiring ───────────────────────────────────────────────────

def test_repl_knows_code_command():
    from osiris_cli.osiris_repl import OsirisReplState, dispatch_command

    buf = io.StringIO()
    with patch("sys.stdout", buf):
        dispatch_command(OsirisReplState(), "/code")
    out = buf.getvalue()
    assert "Usage: /code" in out and "Unknown command" not in out


def test_repl_routes_code_task(monkeypatch):
    from osiris_cli.osiris_repl import OsirisReplState, dispatch_command

    got = []
    monkeypatch.setattr(code_agent, "run_code_command", lambda state, rest: got.append(rest))
    dispatch_command(OsirisReplState(), "/code --model qwen2.5:3b list the tests")
    assert got == ["--model qwen2.5:3b list the tests"]


def test_provider_chat_sends_tools_and_returns_native_calls():
    from osiris_cli.providers.base import Message, ProviderRequest
    from osiris_cli.providers.ollama_compat import OllamaCompatibilityProvider

    sent = {}

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(req, timeout=None):
        sent.update(json.loads(req.data.decode()))
        sent["url"] = req.full_url
        body = {"message": {"content": "", "tool_calls": [{"function": {"name": "list_dir", "arguments": {}}}]},
                "done": True}
        return Resp(json.dumps(body).encode())

    with patch("urllib.request.urlopen", urlopen):
        r = OllamaCompatibilityProvider(host="http://h:1").chat(ProviderRequest(
            messages=[Message("user", "hi"), Message("tool", "x", {"name": "list_dir"})],
            tools=code_agent.TOOLS, model="m"))
    assert sent["url"] == "http://h:1/api/chat" and sent["tools"] == code_agent.TOOLS
    assert sent["messages"][1] == {"role": "tool", "content": "x", "name": "list_dir"}
    assert r.tool_calls[0]["function"]["name"] == "list_dir" and not r.error


def test_provider_chat_reports_errors_without_raising():
    from osiris_cli.providers.base import ProviderRequest
    from osiris_cli.providers.ollama_compat import OllamaCompatibilityProvider

    with patch("urllib.request.urlopen", side_effect=OSError("refused")):
        r = OllamaCompatibilityProvider(host="http://h:1").chat(ProviderRequest(model="m"))
    assert r.error.startswith("Ollama chat failed")


def test_write_goes_through_the_real_repl_proposal_path(ws, monkeypatch, capsys):
    import osiris_termux_console as otc

    monkeypatch.setattr(otc, "_pending_write", {"path": None, "content": None, "existed": None,
                                                "raw_path": None, "story_meta": None})
    replies = iter([call("write_file", path="out.py", content="x = 1\n"), {"content": "done"}])
    a = CodeAgent(root=str(ws), out=lambda s: None, confirm=lambda p: False,
                  chat=lambda messages, tools, model: next(replies))
    a.run("write")
    pending = otc._pending_write
    assert pending["path"] == str(ws / "out.py") and pending["content"] == "x = 1\n"
    assert not (ws / "out.py").exists()                 # nothing until the user's /apply
    assert "Type /apply to confirm" in capsys.readouterr().out
