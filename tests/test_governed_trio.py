"""Three-agent slice: Research -> Developer -> Critic with deterministic verification.
Synthetic repo only; the Developer is a scripted function, never a live model."""

import json
import os

import pytest

from osiris_cli.orchestration.agent_spec import (AgentRun, AgentSpec, AgentState, BudgetExceeded,
                                                 IllegalTransition, RunBudget)
from osiris_cli.orchestration.evidence_firewall import ExecutionEvidence, check_report
from osiris_cli.orchestration.trio import (DeveloperOutput, run_trio, tree_digest, verify_chain)

BUGGY = "def mean(xs):\n    return sum(xs) / (len(xs) + 1)\n"
FIXED = "def mean(xs):\n    return sum(xs) / len(xs)\n"
TESTS = ("from calc import mean\n\n"
         "def test_mean():\n    assert mean([2, 4]) == 3\n\n"
         "def test_single():\n    assert mean([5]) == 5\n")


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    (r / "calc.py").write_text(BUGGY)
    (r / "test_calc.py").write_text(TESTS)
    return r


def _dev(files, report):
    return lambda findings, sandbox: DeveloperOutput(files, report)


def _run(repo, tmp_path, dev):
    before = tree_digest(str(repo))
    res = run_trio(str(repo), dev, str(tmp_path / "out"), timeout=60)
    assert tree_digest(str(repo)) == before, "the repo under test must never change"
    return res


def test_honest_fix_is_pending_human_approval_and_repo_untouched(repo, tmp_path):
    res = _run(repo, tmp_path, _dev({"calc.py": FIXED}, "Changed the divisor to len(xs)."))
    assert res.outcome == "PENDING_HUMAN_APPROVAL", res.reasons
    assert res.candidate_status == "MEASURED"
    assert res.baseline.failed == 2 and res.candidate.green
    assert sorted(res.critic.fail_to_pass) == ["test_calc.py::test_mean", "test_calc.py::test_single"]
    assert (repo / "calc.py").read_text() == BUGGY
    rec = json.load(open(os.path.join(res.run_dir, "record.json")))
    assert rec["approval"] == "PENDING"
    assert rec["repo_digest_before"] == rec["repo_digest_after"]
    assert verify_chain(str(tmp_path / "out" / "receipts.jsonl"))
    assert [r["role"] for r in res.receipts] == ["research", "developer", "critic"]


def test_unexecuted_test_claim_is_quarantined(repo, tmp_path):
    # Developer claims success but its patch is a no-op change that still fails. The claim must
    # not be believed; the critic sees red, the firewall sees a false 'tests passed'.
    res = _run(repo, tmp_path,
               _dev({"calc.py": BUGGY + "# touched\n"}, "All tests pass now. I ran pytest."))
    assert res.outcome == "REJECTED"
    kinds = {(f.kind, f.verdict) for f in res.firewall.findings}
    assert ("test_result", "REFUTED") in kinds


def test_invented_measurement_is_quarantined_even_with_a_working_patch(repo, tmp_path):
    res = _run(repo, tmp_path, _dev(
        {"calc.py": FIXED}, "Fixed the divisor. Accuracy improved by 37.2% (95% CI 35.1-39.4) on ibm_torino."))
    assert res.outcome == "QUARANTINED"
    verdicts = {f.verdict for f in res.firewall.findings}
    assert "UNMEASURED" in verdicts and "UNVERIFIED" in verdicts
    assert res.critic.fail_to_pass  # the code itself did work; the claims did not


def test_developer_cannot_edit_tests_or_escape_the_repo(repo, tmp_path):
    res = _run(repo, tmp_path, _dev(
        {"calc.py": FIXED, "test_calc.py": "def test_mean():\n    assert True\n",
         "../evil.py": "x = 1\n"}, "Fixed."))
    assert res.outcome == "REJECTED"
    joined = " ".join(res.critic.reasons)
    assert "may not edit or add tests" in joined and "escapes the repo" in joined
    assert not (tmp_path / "evil.py").exists()


def test_ast_firewall_blocks_dangerous_patch(repo, tmp_path):
    res = _run(repo, tmp_path, _dev({"calc.py": "import os\nos.system('id')\n"}, "Fixed."))
    assert res.outcome == "REJECTED" and "AST firewall" in " ".join(res.critic.reasons)


def test_regression_against_frozen_baseline_is_rejected(repo, tmp_path):
    (repo / "test_calc.py").write_text(TESTS + "\ndef test_empty_is_zero():\n    assert mean([]) == 0\n")
    # baseline: test_empty_is_zero passes (sum([])/(0+1) == 0); the fix makes it raise.
    res = _run(repo, tmp_path, _dev({"calc.py": FIXED}, "Fixed the divisor."))
    assert res.outcome == "REJECTED"
    assert res.critic.pass_to_fail == ["test_calc.py::test_empty_is_zero"]


def test_green_repo_is_no_defect(repo, tmp_path):
    (repo / "calc.py").write_text(FIXED)
    assert _run(repo, tmp_path, _dev({}, "")).outcome == "NO_DEFECT"


def test_out_dir_inside_repo_is_refused(repo):
    with pytest.raises(ValueError):
        run_trio(str(repo), _dev({}, ""), str(repo / "out"))


def test_supplementary_critic_is_recorded_but_not_authoritative(repo, tmp_path):
    res = run_trio(str(repo), _dev({"calc.py": BUGGY + "#\n"}, "Fixed."), str(tmp_path / "o"),
                   supplementary_fn=lambda d: "LGTM, ship it")
    assert res.outcome == "REJECTED" and res.critic.supplementary == "LGTM, ship it"


# ── firewall unit tests ──

def _ev(green=True):
    return ExecutionEvidence(("pytest",), 0 if green else 1, 2 if green else 0, 0 if green else 2, 0, "d", "")


def test_firewall_verdicts():
    assert check_report("All tests pass.", None).decision == "QUARANTINE"
    assert check_report("All tests pass.", _ev(True)).decision == "ACCEPT"
    assert check_report("All tests pass.", _ev(False)).decision == "REJECT"
    assert check_report("Latency fell 12%.", None).findings[0].verdict == "UNMEASURED"
    arts = {"run7": "latency_ms before=100 after=88 delta 12%"}
    assert check_report("Latency fell 12% [evidence:run7].", None, arts).decision == "ACCEPT"
    assert check_report("Latency fell 40% [evidence:run7].", None, arts).decision == "REJECT"
    assert check_report("Refactored the helper.", None).decision == "ACCEPT"


def test_firewall_registered_refuted_claim_unless_negated():
    from osiris_cli import claims
    c = next(c for c in claims.REGISTER if c.verdict == "ARTIFACT" and c.id == "NEGENTROPY_136")
    stmt = "We found a 136-bit negentropy gap on a 156-qubit Heron processor"
    assert check_report(stmt + ".", None).decision == "REJECT", c.id
    assert check_report("The " + stmt.lower() + " is not supported; it was an artifact.", None
                        ).decision == "ACCEPT"


# ── agent contract ──

def test_state_machine_and_budget():
    spec = AgentSpec("a", "1", "research", frozenset({"READ_ONLY_ANALYSIS"}))
    b = RunBudget(max_agents=1)
    run = AgentRun(spec, "r", b)
    with pytest.raises(IllegalTransition):
        run.to(AgentState.DONE)           # CREATED -> DONE is not allowed
    run.start({"x": 1}); run.finish({"y": 2})
    assert run.receipt().state == "DONE" and len(run.receipt().input_digest) == 64
    with pytest.raises(BudgetExceeded):
        AgentRun(spec, "r", b)            # second agent exceeds max_agents
    with pytest.raises(BudgetExceeded):
        AgentRun(spec, "r", RunBudget(max_depth=0), depth=1)   # no nested spawning
    assert not spec.allows("WORKSPACE_WRITE")
