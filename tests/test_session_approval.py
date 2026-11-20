"""Session approval: one readable approval for a bounded coding session.

Spec (owner, 2026-11-19): normal local coding work must not need finding grants, copying grant
IDs, calculating payload hashes or supplying signature hex. Each test below names the
requirement it checks.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from osiris_cli import session_approval as sa
from osiris_cli.code_runs import RunStore
from osiris_cli.code_task import apply_idempotent, main as task_main, run_task
from osiris_cli.pqc_enclave import (
    GrantVerificationError,
    KeySetupRequired,
    PQCEnclaveGate,
    load_or_onboard,
    setup_local_keys,
)

BUGGY = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"
TEST = "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
PYTEST = [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_calc.py"]
FIXTURE = Path(__file__).parent / "fixtures" / "legacy_hybrid_grant_v2.json"


def call(name, **args):
    return {"content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def scripted(*steps, final="done"):
    script = list(steps) + [{"content": final}]
    return lambda messages, tools, model: script.pop(0) if script else {"content": final}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "calc.py").write_text(BUGGY)
    (root / "util.py").write_text("X = 1\n")
    (root / "test_calc.py").write_text(TEST)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    monkeypatch.delenv("OSIRIS_TEXT_TOOL_CALLS", raising=False)
    return root


@pytest.fixture
def keys(tmp_path):
    d = tmp_path / "enclave"
    setup_local_keys(str(d), confirm=lambda p: True)
    return d


def gate(keys):
    return PQCEnclaveGate.load_or_init_local(str(keys))


def session_for(repo, keys, edits=("calc.py",), commands=(PYTEST,), minutes=30):
    g = gate(keys)
    scope = sa.SessionScope.build(str(repo), list(edits), [list(c) for c in commands], minutes)
    return g, scope, sa.request_session_grant(g, scope, confirm=lambda p: True)


def run(repo, keys, tmp_path, *steps, session=None, confirm=None, prompts=None, **kw):
    prompts = [] if prompts is None else prompts

    def ask(p):
        prompts.append(p)
        return confirm(p) if confirm else False
    return run_task("fix add", root=str(repo), chat=scripted(*steps), out=lambda s: None,
                    store=RunStore(str(tmp_path / "runs")), receipts=str(tmp_path / "receipts.jsonl"),
                    enclave_dir=str(keys), confirm=ask, session=session, **kw)


# --- 1. an allowed fixture edit plus its tests run without any hex ------------------------
def test_session_edit_and_test_run_need_one_readable_approval_and_no_hex(repo, keys, tmp_path):
    prompts, shown = [], []
    scope = sa.SessionScope.build(str(repo), ["calc.py"], [" ".join(PYTEST)])
    s = run_task("fix add", root=str(repo), store=RunStore(str(tmp_path / "runs")),
                 chat=scripted(call("write_file", path="calc.py", content=FIXED),
                               call("run_command", command=" ".join(PYTEST))),
                 receipts=str(tmp_path / "receipts.jsonl"), enclave_dir=str(keys),
                 accept=" ".join(PYTEST), allow_commands=True, session=scope,
                 confirm=lambda p: (prompts.append(p) or True), out=shown.append)
    assert s["state"] == "SUCCEEDED" and s["disposition"] == "VERIFIED", (s["history"], s.get("verification"))
    assert prompts == ["Approve this coding session? [y/N] "]  # the only question asked
    card = "\n".join(shown)
    assert "May edit: calc.py" in card and "May run:" in card and "Will not, without asking you" in card
    assert (repo / "calc.py").read_text() == FIXED
    rec = json.loads(Path(tmp_path / "runs" / f"{s['run_id']}.json").read_text())
    assert rec["applied"][0]["approval_mode"] == "session"
    assert rec["applied"][0]["session_grant_id"] == rec["session_grant"]["grant_id"]
    cmd = [t for t in rec["tool_calls"] if t["tool"] == "run_command"]
    assert cmd and cmd[0]["decision"] == "GRANTED" and cmd[0]["approval_mode"] == "session"
    for p in prompts:
        assert "hex" not in p.lower() and "grant" not in p.lower() and "sha" not in p.lower()


def test_cli_session_flags_build_the_scope(repo, keys, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("osiris_cli.code_task._stdin_confirm", lambda p: False)
    rc = task_main(["fix", "--root", str(repo), "--enclave-dir", str(keys), "--runs-dir", str(tmp_path / "r"),
                    "--session-command", "curl https://example.com"])
    assert rc == 2 and "cannot be pre-approved" in capsys.readouterr().err


# --- 2. no self-authorization, no scope expansion ------------------------------------------
def test_write_outside_session_files_falls_back_to_a_prompt(repo, keys, tmp_path):
    scope = sa.SessionScope.build(str(repo), ["calc.py"], [])
    prompts = []
    s = run(repo, keys, tmp_path, call("write_file", path="util.py", content="X = 2\n"),
            session=scope, confirm=lambda p: p.startswith("Approve this coding session"),
            prompts=prompts, accept=" ".join(PYTEST), require_change=False)
    assert prompts[0].startswith("Approve this coding session")
    assert any(p.startswith("Apply 1 change") for p in prompts)  # asked, not auto-approved
    assert s["state"] == "FAILED" and (repo / "util.py").read_text() == "X = 1\n"


def test_command_outside_session_list_is_asked_not_run(repo, keys, tmp_path):
    g, scope, sg = session_for(repo, keys, commands=(PYTEST,))
    from osiris_cli.code_agent import CodeAgent
    asked = []
    agent = CodeAgent(root=str(repo), chat=scripted(), confirm=lambda p: asked.append(p) or False,
                      out=lambda s: None, enclave=g, session_grant=sg)
    with pytest.raises(Exception, match="did not approve"):
        agent._run_command(f"{sys.executable} -m pytest -q")  # not the exact listed argv
    assert len(asked) == 1


def test_tampered_session_scope_does_not_verify(repo, keys):
    g, scope, sg = session_for(repo, keys)
    wider = copy.deepcopy(sg)
    wider["operation_schema"]["edit_globs"].append("util.py")
    ok, why, _ = sa.verify_session_grant(g, wider)
    assert not ok and "schema" in why.lower()
    longer = copy.deepcopy(sg)
    longer["expires_at"] += 3600
    assert not sa.verify_session_grant(g, longer)[0]
    with pytest.raises(GrantVerificationError):
        sa.issue_child_grant(g, wider, action="code.apply_writes", scope=["x"], payload_sha256="a" * 64,
                             operation_schema={}, covered=(True, "claimed"))


def test_model_cannot_pass_a_session_through_tool_arguments(repo, keys, tmp_path):
    g, scope, sg = session_for(repo, keys)
    from osiris_cli.code_agent import CodeAgent
    agent = CodeAgent(root=str(repo), chat=scripted(call("run_command", command=" ".join(PYTEST),
                                                         session_grant=sg)),
                      confirm=lambda p: False, out=lambda s: None, enclave=g)
    agent.run("x")
    d = [x for x in agent.decisions if x["tool"] == "run_command"][0]
    assert d["decision"] == "ERROR" and "session_grant" in d["reason"]


# --- 3. outside-workspace writes and excluded actions are denied -------------------------
@pytest.mark.parametrize("cmd", [
    "bash -c 'pytest'", "sh -c ls", "curl https://x", "wget x", "ssh host", "pip install x",
    f"{sys.executable} -m pip install x", "npm install", "npx something", "git push", "git reset --hard",
    "git -c core.pager=x log", "rm -rf build", "find . -delete", "sudo pytest", "chmod 777 x", "vercel deploy",
    "gh pr create", "supabase db push", "docker run x", f"{sys.executable} -c 'print(1)'", "node -e 1",
])
def test_excluded_commands_cannot_be_put_in_a_session(repo, cmd):
    with pytest.raises(sa.SessionScopeError, match="cannot be pre-approved"):
        sa.SessionScope.build(str(repo), [], [cmd])


@pytest.mark.parametrize("glob", ["../*", "/etc/*", "*", "**", ".env", ".git/*", "keys/id_ed25519", "secrets/*",
                                  ".github/workflows/*", ".osiris/*"])
def test_broad_outside_or_protected_edit_patterns_are_refused(repo, glob):
    with pytest.raises(sa.SessionScopeError):
        sa.SessionScope.build(str(repo), [glob], [])


def test_paths_outside_or_protected_are_never_covered(repo, tmp_path):
    scope = sa.SessionScope.build(str(repo), ["conf/*", "calc.py"], [])
    assert sa.covers_paths(scope, str(repo), ["calc.py"])[0]
    for bad in ["../calc.py", "/etc/passwd", "conf/.env", "conf/x.pem", "util.py"]:
        assert not sa.covers_paths(scope, str(repo), [bad])[0], bad
    assert not sa.covers_paths(scope, str(tmp_path), ["calc.py"])[0]  # different workspace
    assert not sa.covers_command(sa.SessionScope.build(str(repo), [], [" ".join(PYTEST)]), PYTEST, str(tmp_path))[0]


# --- 4. changed payloads and substituted targets are rejected -----------------------------
def test_child_grant_binds_the_exact_payload_and_target(repo, keys):
    g, scope, sg = session_for(repo, keys)
    digest = "c" * 64
    child = sa.issue_child_grant(g, sg, action="code.run_command", scope=["SUBPROCESS_EXEC", str(repo)],
                                 payload_sha256=digest, operation_schema={"argv": PYTEST, "cwd": str(repo)},
                                 covered=sa.covers_command(scope, PYTEST, str(repo)))
    assert g.verify_grant_integrity(child, expected_payload_sha256=digest, expected_event_type="code.run_command")[0]
    assert not g.verify_grant_integrity(child, expected_payload_sha256="d" * 64)[0]
    assert not g.verify_grant_integrity(child, expected_scope=["SUBPROCESS_EXEC", "/elsewhere"])[0]
    assert not g.verify_grant_integrity(child, expected_event_type="code.apply_writes")[0]
    swapped = copy.deepcopy(child)
    swapped["operation_schema"]["argv"] = ["rm", "-rf", "."]
    assert not g.verify_grant_integrity(swapped)[0]


def test_changed_proposal_after_session_signing_is_not_applied(repo, keys, tmp_path):
    scope = sa.SessionScope.build(str(repo), ["calc.py"], [])
    store = RunStore(str(tmp_path / "runs"))
    s = run_task("fix", root=str(repo), store=store, chat=scripted(call("write_file", path="calc.py", content=FIXED)),
                 receipts=None, enclave_dir=str(keys), confirm=lambda p: True, out=lambda m: None,
                 session=scope, accept=None, stop_after=None)
    rec = store.load(s["run_id"])
    rec["proposals"][0]["after_sha256"] = "e" * 64
    rec["applied"] = []
    with pytest.raises(GrantVerificationError):
        apply_idempotent(rec, store, enclave_dir=str(keys))


def test_relabelled_session_child_or_foreign_parent_is_rejected(repo, keys):
    g, scope, sg = session_for(repo, keys)
    _g2, _s2, other = session_for(repo, keys)
    child = sa.issue_child_grant(g, sg, action="code.apply_writes", scope=["w"], payload_sha256="a" * 64,
                                 operation_schema={"f": 1}, covered=(True, "ok"))
    assert sa.verify_child_link(g, child, sg)[0]
    assert not sa.verify_child_link(g, child, other)[0]  # a different session
    assert not sa.verify_child_link(g, child, None)[0]
    relabelled = dict(child, approval_mode="interactive")  # approval_mode is not signed; the link is
    assert not sa.verify_child_link(g, relabelled, None)[0]


# --- 5. untrusted keys, malformed signatures, expired and replayed grants -----------------
def test_session_signed_by_an_untrusted_key_is_rejected(repo, keys):
    g = gate(keys)
    stranger = PQCEnclaveGate()  # valid keys, but not in this operator's trust store
    scope = sa.SessionScope.build(str(repo), ["calc.py"], [])
    foreign = sa.request_session_grant(stranger, scope, confirm=lambda p: True)
    ok, why, _ = sa.verify_session_grant(g, foreign)
    assert not ok and "untrusted signer" in why
    with pytest.raises(GrantVerificationError):
        sa.issue_child_grant(g, foreign, action="code.apply_writes", scope=["w"], payload_sha256="a" * 64,
                             operation_schema={}, covered=(True, "ok"))


@pytest.mark.parametrize("field", ["ed25519_signature", "ml_dsa_signature"])
def test_malformed_session_signature_is_rejected(repo, keys, field):
    g, scope, sg = session_for(repo, keys)
    bad = copy.deepcopy(sg)
    sig = bad["pqc_envelope"][field]
    bad["pqc_envelope"][field] = ("0" if sig[0] != "0" else "1") + sig[1:]
    assert not sa.verify_session_grant(g, bad)[0]
    bad["pqc_envelope"][field] = "zz"
    assert not sa.verify_session_grant(g, bad)[0]


def test_expired_session_issues_no_child(repo, keys):
    g, scope, sg = session_for(repo, keys, minutes=1)
    later = time.time() + 120
    with pytest.raises(GrantVerificationError, match="expired"):
        sa.issue_child_grant(g, sg, action="code.apply_writes", scope=["w"], payload_sha256="a" * 64,
                             operation_schema={}, covered=(True, "ok"), now=later)


def test_child_grant_is_single_use(repo, keys):
    g, scope, sg = session_for(repo, keys)
    child = sa.issue_child_grant(g, sg, action="code.apply_writes", scope=["w"], payload_sha256="a" * 64,
                                 operation_schema={}, covered=(True, "ok"))
    assert g.verify_grant_integrity(child, consume=True)[0]
    ok, why = gate(keys).verify_grant_integrity(child, consume=True)  # fresh process, same ledger
    assert not ok and "replay" in why.lower()


# --- 6. readable onboarding when the signer is missing ------------------------------------
def test_missing_signer_blocks_with_setup_instructions_and_creates_nothing(repo, tmp_path):
    empty = tmp_path / "no-keys"
    prompts = []
    s = run(repo, empty, tmp_path, call("write_file", path="calc.py", content=FIXED),
            prompts=prompts, accept=" ".join(PYTEST), approve_writes=True)
    assert s["state"] == "BLOCKED", s
    reason = " ".join(s["verification"]["reasons"])
    assert "no signing key set up yet" in reason and "python -m osiris_cli.pqc_enclave setup" in reason
    assert prompts and prompts[0].startswith("Create the signing key")  # offered, operator said no
    assert not (empty / "operator_ed25519.key").exists() and not (empty / "trusted_operator_keys.json").exists()
    assert (repo / "calc.py").read_text() == BUGGY


def test_onboarding_creates_keys_only_after_a_yes(tmp_path):
    d = tmp_path / "k"
    with pytest.raises(KeySetupRequired):
        PQCEnclaveGate.load_or_init_local(str(d))  # no silent creation
    assert not d.joinpath("operator_ed25519.key").exists()
    shown = []
    g = load_or_onboard(str(d), confirm=lambda p: True, out=shown.append)
    assert d.joinpath("operator_ed25519.key").exists() and g.key_fingerprint in json.dumps(g.trusted_key_records())
    text = "\n".join(shown)
    assert "never leaves this machine" in text
    seed = d.joinpath("operator_ed25519.key").read_bytes().hex()
    assert seed not in text


def test_keys_without_a_trust_list_are_not_silently_enrolled(tmp_path):
    d = tmp_path / "k"
    setup_local_keys(str(d), confirm=lambda p: True)
    (d / "trusted_operator_keys.json").unlink()
    with pytest.raises(KeySetupRequired, match="trusted-key list is missing"):
        PQCEnclaveGate.load_or_init_local(str(d))
    assert not (d / "trusted_operator_keys.json").exists()


def test_existing_trust_list_without_keys_is_not_paired_with_new_keys(tmp_path):
    d = tmp_path / "k"
    setup_local_keys(str(d), confirm=lambda p: True)
    for n in ("operator_ed25519.key", "operator_mldsa65.key"):
        (d / n).unlink()
    with pytest.raises(KeySetupRequired, match="already lists trusted keys"):
        setup_local_keys(str(d), confirm=lambda p: True)
    assert not (d / "operator_ed25519.key").exists()


def test_code_agent_has_no_throwaway_self_trusting_signer(repo, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    # Isolate from any real operator keys on this machine and from keys another
    # test may have enrolled: with no enclave available this must fail closed.
    import osiris_cli.pqc_enclave as pqc

    def no_key(*args, **kwargs):
        raise pqc.KeySetupRequired("no signing key set up")

    monkeypatch.setattr(pqc, "load_or_onboard", no_key)
    from osiris_cli.code_agent import CodeAgent, Denied
    agent = CodeAgent(root=str(repo), chat=scripted(), confirm=lambda p: False, out=lambda s: None)
    with pytest.raises(Denied, match="no signing key set up"):
        agent._get_enclave()


# --- 7. legacy verification still works ---------------------------------------------------
def test_grant_signed_by_the_previous_code_still_verifies():
    doc = json.loads(FIXTURE.read_text())
    g = PQCEnclaveGate(trusted_public_keys=doc["trusted_keys"], verifier_only=True)
    grant = doc["grant"]
    ok, why = g.verify_grant_integrity(grant, now=grant["issued_at"])
    assert ok, why
    assert sa.verify_child_link(g, grant, None) == (True, "not a session grant")
    tampered = copy.deepcopy(grant)
    tampered["operation_schema"]["files"][0]["rel_path"] = "other.py"
    assert not g.verify_grant_integrity(tampered, now=grant["issued_at"])[0]


# --- 8. no silent hybrid downgrade --------------------------------------------------------
@pytest.mark.parametrize("change", ["drop_ml_dsa", "drop_ed25519", "relabel_ed25519_only", "relabel_ml_dsa_only"])
def test_hybrid_grants_cannot_be_downgraded(repo, keys, change):
    g, scope, sg = session_for(repo, keys)
    child = sa.issue_child_grant(g, sg, action="code.apply_writes", scope=["w"], payload_sha256="a" * 64,
                                 operation_schema={}, covered=(True, "ok"))
    for grant in (sg, child):
        bad = copy.deepcopy(grant)
        env = bad["pqc_envelope"]
        if change == "drop_ml_dsa":
            env["ml_dsa_signature"] = ""
        elif change == "drop_ed25519":
            del env["ed25519_signature"]
        elif change == "relabel_ed25519_only":
            env["algorithm"] = "Ed25519"
        else:
            env["algorithm"] = "ML-DSA-65"
        ok, why = g.verify_grant_integrity(bad)
        assert not ok, (change, why)


def test_no_private_key_material_in_session_records(repo, keys, tmp_path):
    scope = sa.SessionScope.build(str(repo), ["calc.py"], [])
    store = RunStore(str(tmp_path / "runs"))
    s = run_task("fix", root=str(repo), store=store, chat=scripted(call("write_file", path="calc.py", content=FIXED)),
                 receipts=str(tmp_path / "receipts.jsonl"), enclave_dir=str(keys), confirm=lambda p: True,
                 out=lambda m: None, session=scope, accept=None, require_change=True)
    blob = Path(tmp_path / "runs" / f"{s['run_id']}.json").read_text() + (tmp_path / "receipts.jsonl").read_text()
    for name in ("operator_ed25519.key", "operator_mldsa65.key"):
        assert (keys / name).read_bytes().hex() not in blob
    assert os.stat(keys / "operator_ed25519.key").st_mode & 0o077 == 0
