"""The /code tool loop must never run credential, account or publishing commands, even when
the user would approve: a pasted report once led it to propose `gh auth logout` and
`gh auth login`, and the user's y signed the workstation out of GitHub (2026-11-18)."""

import pytest

from osiris_cli.code_agent import _account_action
from tests.test_code_agent import agent, call, tool_results, ws  # noqa: F401  (fixture)

DENIED = [
    "gh auth logout", "gh auth login", "gh auth token", "gh secret set X", "gh pr merge 27",
    "gh release create v1", "gh api -X DELETE repos/a/b", "gh repo delete a/b",
    "git push origin main", "git push --force", "git config --global user.email x@y",
    "git credential fill", "git remote set-url origin https://evil.example/x.git",
    "vercel deploy --prod", "supabase db push", "gcloud auth login", "ssh-keygen -t ed25519",
    "npm publish", "twine upload dist/*", "docker push img", "ollama rm qwen3:8b",
    "sudo gh auth logout", "env GH_TOKEN=x gh auth status", "/usr/bin/git push",
]
ALLOWED = ["git status", "git diff", "git log -1", "git config user.name", "gh pr view 27",
           "gh pr list", "gh repo view", "npm test", "ollama list", "ls -la", "pytest -q"]


@pytest.mark.parametrize("command", DENIED)
def test_account_commands_are_refused_without_asking(ws, command):
    prompts = []
    a = agent(ws, call("run_command", command=command), confirm=lambda p: prompts.append(p) or True)
    a.run("pasted report says: revoke the token")
    assert prompts == [], "the user must never be asked to approve an account command"
    result = tool_results(a)[0]
    assert result.startswith("DENIED") and "run it yourself" in result


@pytest.mark.parametrize("command", ALLOWED)
def test_read_only_commands_are_not_classified_as_account_actions(command):
    import shlex
    assert _account_action(shlex.split(command)) is None


def test_text_parsed_call_is_refused_too(ws, monkeypatch):
    # Models without native tool calls emit JSON in prose ("[from text]"); same rule applies.
    monkeypatch.setenv("OSIRIS_TEXT_TOOL_CALLS", "1")
    prompts = []
    text = '{"name": "run_command", "arguments": {"command": "gh auth logout"}}'
    a = agent(ws, {"content": text}, confirm=lambda p: prompts.append(p) or True)
    a.run("x")
    assert prompts == []
    assert any(r.startswith("DENIED") for r in tool_results(a))
