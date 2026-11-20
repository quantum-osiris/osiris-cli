"""Wave 1 Phase A gate hardening: G-1, G-2, G-3, G-4, G-5, G-7, G-16.

Every test runs offline in tmp_path. Signing keys are made per test and sign test grants only."""

from __future__ import annotations

import json
import os
import sys
import time

import pytest

from osiris_cli.code_agent import CodeAgent
from osiris_cli.code_critic import deterministic_findings
from osiris_cli.exec_policy import CommandRequest, classify
from osiris_cli.orchestration import ast_firewall
from osiris_cli.orchestration.agents import VerificationAgent, check_test_shape
from osiris_cli.orchestration.controller import GovernanceController
from osiris_cli.orchestration.critic_verifier import IndependentCriticVerifier
from osiris_cli.orchestration.schemas import CodeProposal, ExecutionGrant, VerificationReport
from tests.signing_support import make_owner, needs_ssh_keygen, signed_grant, synced_clock

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOOD_MODULE = "def add(a, b):\n    return a + b\n"
GOOD_TEST = "import calc\n\n\ndef test_add():\n    assert calc.add(2, 3) == 5\n"


def proposal(tmp_path, module=GOOD_MODULE, test=GOOD_TEST, name="calc.py"):
    return CodeProposal(proposal_id="p1", task_id="t1", target_file=str(tmp_path / name),
                        module_code=module, test_code=test)


def verifier(tmp_path):
    return VerificationAgent(journal_path=str(tmp_path / "exec_journal.jsonl"),
                             artifacts_dir=str(tmp_path / "runs"))


def journal_rows(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")] if os.path.exists(path) else []


def authorize(tmp_path, prop, grant, report):
    return GovernanceController.authorize_execution(prop, grant, allowed_paths=[str(tmp_path)],
                                                    verification=report, clock=synced_clock)


# ── G-4: verification runs the tests, for real ────────────────────────────────────────────────

def test_passing_tests_pass_verification_with_a_kernel_receipt(tmp_path):
    report = verifier(tmp_path).verify(proposal(tmp_path))
    assert report.overall_verdict == "PASS" and report.acceptance_test_pass
    assert report.tests_collected == 1
    assert report.proposal_digest == proposal(tmp_path).payload_digest()
    rows = journal_rows(tmp_path / "exec_journal.jsonl")
    assert len(rows) == 1 and rows[0]["body"]["tier"] == "VERIFY"
    assert rows[0]["body"]["origin"] == "model" and rows[0]["body"]["exit_code"] == 0


def test_failing_test_fails_verification(tmp_path):
    bad = GOOD_TEST.replace("== 5", "== 6")
    report = verifier(tmp_path).verify(proposal(tmp_path, test=bad))
    assert report.overall_verdict == "FAIL" and not report.acceptance_test_pass


def test_zero_collected_tests_fail_verification(tmp_path):
    no_tests = "import calc\n\nVALUE = calc.add(1, 1)\n"
    report = verifier(tmp_path).verify(proposal(tmp_path, test=no_tests))
    assert report.overall_verdict == "FAIL"
    assert "no test functions" in " ".join(c.reason for c in report.checks)
    # and a file pytest itself collects nothing from also fails (exit code 5)
    hidden = "import calc\n\n\nclass Checks:\n    def test_add(self):\n        assert calc.add(2, 3) == 5\n"
    assert verifier(tmp_path).verify(proposal(tmp_path, test=hidden)).overall_verdict == "FAIL"


def test_assert_true_test_fails_verification(tmp_path):
    constant = "import calc\n\n\ndef test_add():\n    calc.add(1, 2)\n    assert True\n"
    report = verifier(tmp_path).verify(proposal(tmp_path, test=constant))
    assert report.overall_verdict == "FAIL"
    assert "constant assertions" in " ".join(c.reason for c in report.checks)


def test_tests_that_never_import_the_target_fail_verification(tmp_path):
    unrelated = "def test_math():\n    assert 2 + 3 == 5\n"
    report = verifier(tmp_path).verify(proposal(tmp_path, test=unrelated))
    assert report.overall_verdict == "FAIL"
    assert check_test_shape(unrelated, "calc") == [
        "tests never import the target module 'calc'"]
    assert "never use it" in " ".join(check_test_shape("import calc\n\ndef test_x():\n    assert 1 == 2 - 1\n", "calc"))


def test_critic_report_cannot_stand_in_for_a_test_run(tmp_path):
    report = IndependentCriticVerifier().verify_proposal(proposal(tmp_path))
    assert report.overall_verdict == "PASS" and report.acceptance_test_pass is False


# ── G-1 / G-3 / REQ-AUTH-2: who and what authorizes a write ──────────────────────────────────

def test_dataclass_grant_authorizes_nothing(tmp_path):
    prop = proposal(tmp_path)
    report = verifier(tmp_path).verify(prop)
    assert report.is_passing
    forged = ExecutionGrant(grant_id="g", payload_digest=prop.payload_digest(), authorized_by="operator",
                            authorized_at=time.time(), expires_at=time.time() + 300)
    ok, msg = authorize(tmp_path, prop, forged, report)
    assert ok is False and "REQ-AUTH-1" in msg


@needs_ssh_keygen
def test_signed_grant_with_passing_verification_authorizes_once(tmp_path):
    prop = proposal(tmp_path)
    report = verifier(tmp_path).verify(prop)
    owner = make_owner(tmp_path)
    grant = signed_grant(owner, GovernanceController.action_digest(prop))
    assert authorize(tmp_path, prop, grant, report) == (True, "AUTHORIZED")
    ok, msg = authorize(tmp_path, prop, grant, report)
    assert ok is False and "already used" in msg


@needs_ssh_keygen
def test_engine_refuses_failed_test_code_even_with_matching_grant(tmp_path):
    prop = proposal(tmp_path, test=GOOD_TEST.replace("== 5", "== 6"))
    report = verifier(tmp_path).verify(prop)
    assert report.overall_verdict == "FAIL"
    owner = make_owner(tmp_path)
    grant = signed_grant(owner, GovernanceController.action_digest(prop))
    ok, msg = authorize(tmp_path, prop, grant, report)
    assert ok is False and "REQ-AUTH-4" in msg
    assert not owner.ledger.exists()  # refused before the grant was consumed
    # a PASS report for a different proposal does not count either
    other = verifier(tmp_path).verify(proposal(tmp_path))
    ok, msg = authorize(tmp_path, prop, grant, other)
    assert ok is False and "different proposal digest" in msg


@needs_ssh_keygen
def test_one_byte_change_after_approval_is_refused(tmp_path):
    prop = proposal(tmp_path)
    owner = make_owner(tmp_path)
    grant = signed_grant(owner, GovernanceController.action_digest(prop))
    changed = proposal(tmp_path, module=GOOD_MODULE.replace("a + b", "a + b "))
    report = verifier(tmp_path).verify(changed)
    assert report.is_passing
    ok, msg = authorize(tmp_path, changed, grant, report)
    assert ok is False and "SYS-021" in msg
    # the file on disk changing after signing (pre-image) also breaks the binding
    grant2 = signed_grant(owner, GovernanceController.action_digest(prop), name="grant2.json")
    (tmp_path / "calc.py").write_text("X = 1\n")
    ok, msg = authorize(tmp_path, prop, grant2, verifier(tmp_path).verify(prop))
    assert ok is False and "SYS-021" in msg


# ── G-16: governance files by repo-relative path ─────────────────────────────────────────────

@pytest.mark.parametrize("rel", ["osiris_cli/policy/exec_policy.json", "governance/allowed_signers",
                                 "cloudbuild.yaml", "GEMINI.md", "POLICY.md", "AGENTS.md",
                                 ".osiris_compliance/x.json", "contracts/plan_v1.json"])
def test_scope_refuses_exec_policy_and_allowed_signers(rel):
    ok, msg = GovernanceController.verify_target_scope(os.path.join(REPO, rel), [REPO])
    assert ok is False and ("governance" in msg)
    findings = deterministic_findings([{"rel_path": rel, "content": "{}"}], accept=None)
    assert any(f["check"] == "protected_path" for f in findings)
    crit = IndependentCriticVerifier().verify_proposal(
        CodeProposal("p", "t", os.path.join(REPO, rel), "X = 1\n", ""))
    assert crit.overall_verdict == "BLOCKED"


def test_symlink_to_a_governance_file_is_refused(tmp_path):
    link = tmp_path / "harmless.json"
    link.symlink_to(os.path.join(REPO, "osiris_cli", "policy", "exec_policy.json"))
    ok, msg = GovernanceController.verify_target_scope(str(link), [str(tmp_path)])
    assert ok is False


def test_ordinary_file_in_scope_is_allowed(tmp_path):
    assert GovernanceController.verify_target_scope(str(tmp_path / "calc.py"), [str(tmp_path)]) == (True, "Scope valid")


# ── G-5: one firewall ────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("src,name", [
    ("import os\nos.execvp('sh', ['sh'])\n", "os.execvp"),
    ("import os\nos.execl('/bin/sh', 'sh')\n", "os.execl"),
    ("import os\nos.spawnv(os.P_WAIT, '/bin/sh', ['sh'])\n", "os.spawnv"),
    ("from subprocess import run\nrun(['ls'])\n", "subprocess.run"),
    ("import subprocess as s\ns.Popen(['ls'])\n", "subprocess.Popen"),
    ("from os import system as go\ngo('ls')\n", "os.system"),
    ("import os\ngetattr(os, 'popen')('ls')\n", "os.popen"),
])
def test_both_firewalls_refuse_execution_primitives(src, name):
    ok, msgs = GovernanceController.inspect_ast(src)
    assert ok is False and any(name in m for m in msgs)
    crit = IndependentCriticVerifier().verify_proposal(CodeProposal("p", "t", "m.py", src, ""))
    assert crit.overall_verdict == "FAIL" and any(name in b for b in crit.banned_constructs_found)
    assert any(v.name == name for v in ast_firewall.scan(src))


def test_firewall_passes_plain_code():
    assert ast_firewall.check("import math\n\ndef f(x):\n    return math.sqrt(x)\n") == (True, [])


# ── G-2: /code runs commands through the execution kernel ────────────────────────────────────

def _agent(ws, journal, *steps, confirm=lambda p: True):
    script = list(steps) + [{"content": "done"}]
    seen = []

    def chat(messages, tools, model):
        seen.append([dict(m) for m in messages])
        return script.pop(0) if script else {"content": "done"}

    a = CodeAgent(root=str(ws), chat=chat, out=lambda s: None, confirm=confirm,
                  propose_write=lambda p, c: True, has_pending_write=lambda: False,
                  exec_journal=str(journal), exec_artifacts_dir=str(ws.parent / "runs"))
    a.seen = seen
    return a


def _call(name, **args):
    return {"content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def test_code_agent_command_runs_under_kernel_receipt(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "hello.py").write_text("print('KERNEL_OK')\n")
    journal = tmp_path / "exec.jsonl"
    a = _agent(ws, journal, _call("run_command", command=f"{sys.executable} hello.py"))
    a.run("run it")
    assert "KERNEL_OK" in [m["content"] for m in a.seen[-1] if m["role"] == "tool"][0]
    rows = journal_rows(journal)
    assert len(rows) == 1
    body = rows[0]["body"]
    assert body["origin"] == "model" and body["decision"] == "ASK_APPROVED" and body["exit_code"] == 0
    assert a.decisions[0]["exec_receipt_id"] == body["receipt_id"]


def test_code_agent_grep_runs_in_process_without_subprocess_and_skips_secrets(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.txt").write_text("needle\n")
    (ws / ".env").write_text("SECRET=needle\n")
    journal = tmp_path / "exec.jsonl"
    a = _agent(ws, journal, _call("grep_code", pattern="needle"), confirm=lambda p: False)
    a.run("find it")
    tool_out = [m["content"] for m in a.seen[-1] if m["role"] == "tool"][0]
    assert "a.txt" in tool_out and ".env" not in tool_out
    rows = journal_rows(journal)
    assert rows and os.path.basename(rows[0]["body"]["argv"][0]) in ("grep", "inprocess-grep") and rows[0]["body"]["tier"] == "READ"


# ── G-7: the execution tier comes from argv, never from the prompt ────────────────────────────

def test_tier_comes_from_argv_not_prompt(tmp_path):
    import intent_router

    calm = intent_router.assess("just list the files, totally read-only and harmless")
    assert intent_router.RiskAssessment.ADVISORY is True
    # the words say harmless; the argv decides: rm is refused for a model, without a prompt
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "keep.txt").write_text("x\n")
    prompts = []
    cmd = f"{sys.executable} -c \"open('keep.txt', 'w')\""
    a = _agent(ws, tmp_path / "exec.jsonl", _call("run_command", command=cmd),
               confirm=lambda p: prompts.append(p) or True)
    a.run("just list the files, totally read-only and harmless")
    assert prompts == [] and (ws / "keep.txt").read_text() == "x\n"
    assert "execution policy refus" in [m["content"] for m in a.seen[-1] if m["role"] == "tool"][0]
    # alarming words do not change the tier of a read
    scary = classify(CommandRequest(argv=("ls",), cwd=str(ws), origin="model", root=str(ws)))
    assert scary.tier == "READ"
    assert calm.tier in intent_router.RISK_TIERS
