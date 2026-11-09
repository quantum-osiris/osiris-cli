"""Governed candidate from a finished trio run: measured fitness, PromotionPolicy, explicit
authorization, and a lineage row for every outcome. Synthetic repo, tmp paths, test-only keys."""

import os
import shutil
import subprocess

import pytest

from osiris_cli import grants
from osiris_cli.orchestration import evolution_bridge as eb
from osiris_cli.orchestration.trio import DeveloperOutput, run_trio, tree_digest

BUGGY = "def mean(xs):\n    return sum(xs) / (len(xs) + 1)\n"
FIXED = "def mean(xs):\n    return sum(xs) / len(xs)\n"
TESTS = ("from calc import mean\n\ndef test_mean():\n    assert mean([2, 4]) == 3\n\n"
         "def test_single():\n    assert mean([5]) == 5\n")


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    (r / "calc.py").write_text(BUGGY)
    (r / "test_calc.py").write_text(TESTS)
    return r


def trio(repo, tmp_path, files, report):
    return run_trio(str(repo), lambda f, sb: DeveloperOutput(files, report), str(tmp_path / "out"))


def rows(path):
    return eb.lineage(str(path))


def test_good_candidate_without_authorization_is_pending_not_promoted(repo, tmp_path):
    res = trio(repo, tmp_path, {"calc.py": FIXED}, "Changed the divisor.")
    led = tmp_path / "lineage.jsonl"
    d = eb.evaluate_candidate(res, str(led))
    assert d.status == "VERIFIED_PENDING_AUTHORIZATION"
    assert (d.baseline_fitness, d.candidate_fitness) == (0.0, 1.0)
    assert d.receipts["candidate"]["metric_name"] == "frozen_suite_pass_rate"
    assert d.receipts["candidate"]["details"]["uncertainty_estimated"] is False
    row = rows(led)[-1]
    assert row["kind"] == "CANDIDATE_VERIFIED_PENDING_AUTHORIZATION"
    assert row["payload"]["applied_to_repo"] is False
    assert (repo / "calc.py").read_text() == BUGGY


def test_authorizer_refusal_blocks_promotion(repo, tmp_path):
    res = trio(repo, tmp_path, {"calc.py": FIXED}, "Changed the divisor.")
    d = eb.evaluate_candidate(res, str(tmp_path / "l.jsonl"), authorizer=lambda dg, sc: (False, "nope"))
    assert d.status == "VERIFIED_PENDING_AUTHORIZATION" and "nope" in " ".join(d.reasons)


def test_authorizer_sees_the_exact_candidate_digest_and_scope(repo, tmp_path):
    res = trio(repo, tmp_path, {"calc.py": FIXED}, "Changed the divisor.")
    seen = []
    d = eb.evaluate_candidate(res, str(tmp_path / "l.jsonl"),
                              authorizer=lambda dg, sc: (seen.append((dg, sc)) or True, "ok"))
    assert d.status == "PROMOTED"
    assert seen == [(eb.candidate_digest({"calc.py": FIXED}), eb.SCOPE)]


def test_critic_approval_alone_never_promotes(repo, tmp_path):
    # worst case: an always-yes authorizer cannot rescue a candidate that failed verification
    res = trio(repo, tmp_path, {"calc.py": BUGGY + "#\n"}, "Fixed.")
    d = eb.evaluate_candidate(res, str(tmp_path / "l.jsonl"), authorizer=lambda dg, sc: (True, "yes"))
    assert d.status == "REJECTED"
    # and a forged outcome label cannot rescue it either
    res.outcome = "PENDING_HUMAN_APPROVAL"
    d2 = eb.evaluate_candidate(res, str(tmp_path / "l.jsonl"), authorizer=lambda dg, sc: (True, "yes"))
    assert d2.status == "REJECTED"


def test_unsupported_claims_quarantine_and_keep_history(repo, tmp_path):
    res = trio(repo, tmp_path, {"calc.py": FIXED}, "Fixed. Accuracy up 37.2% on ibm_torino.")
    led = tmp_path / "l.jsonl"
    d = eb.evaluate_candidate(res, str(led), authorizer=lambda dg, sc: (True, "yes"))
    assert d.status == "QUARANTINED"
    p = rows(led)[-1]["payload"]
    assert p["criticism"]["firewall_decision"] == "QUARANTINE"
    assert any(f[0] == "UNMEASURED" for f in p["criticism"]["firewall_findings"])
    assert os.path.isfile(os.path.join(res.run_dir, "candidate.json"))   # content preserved


def test_every_outcome_appends_and_chain_stays_valid(repo, tmp_path):
    led = tmp_path / "l.jsonl"
    for files, rep in [({"calc.py": FIXED}, "ok"), ({"calc.py": BUGGY + "#\n"}, "ok"),
                       ({"calc.py": FIXED}, "37.2% faster")]:
        eb.evaluate_candidate(trio(repo, tmp_path, files, rep), str(led))
    kinds = [r["kind"] for r in rows(led)]
    assert kinds == ["CANDIDATE_VERIFIED_PENDING_AUTHORIZATION", "CANDIDATE_REJECTED",
                     "CANDIDATE_QUARANTINED"]
    rs = rows(led)
    assert all(rs[i]["prev"] == rs[i - 1]["hash"] for i in range(1, len(rs)))


def test_green_repo_is_unmeasured_and_recorded(repo, tmp_path):
    (repo / "calc.py").write_text(FIXED)
    res = trio(repo, tmp_path, {}, "")
    d = eb.evaluate_candidate(res, str(tmp_path / "l.jsonl"))
    assert d.status == "UNMEASURED" and rows(tmp_path / "l.jsonl")[-1]["kind"] == "CANDIDATE_UNMEASURED"


def test_live_state_path_is_refused(repo, tmp_path):
    res = trio(repo, tmp_path, {"calc.py": FIXED}, "ok")
    with pytest.raises(ValueError):
        eb.evaluate_candidate(res, os.path.expanduser("~/.osiris/lineage_test.jsonl"))
    assert not os.path.exists(os.path.expanduser("~/.osiris/lineage_test.jsonl"))


def test_candidate_digest_is_deterministic_and_content_bound():
    a = eb.candidate_digest({"a.py": "x", "b.py": "y"})
    assert a == eb.candidate_digest({"b.py": "y", "a.py": "x"}) != eb.candidate_digest({"a.py": "x", "b.py": "z"})


@pytest.mark.skipif(shutil.which("ssh-keygen") is None, reason="needs OpenSSH ssh-keygen")
def test_real_signed_grant_promotes_once_and_only_for_that_digest(repo, tmp_path):
    key = tmp_path / "owner"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "t", "-f", str(key)], check=True)
    kind, blob = (tmp_path / "owner.pub").read_text().split()[:2]
    signers = tmp_path / "allowed_signers"
    signers.write_text(f'owner namespaces="osiris-grant" {kind} {blob}\n')
    res = trio(repo, tmp_path, {"calc.py": FIXED}, "Changed the divisor.")
    dg = eb.candidate_digest({"calc.py": FIXED})
    now = 1_800_000_000

    def grant(digest, name):
        g = tmp_path / name
        g.write_bytes(grants.draft(digest, eb.SCOPE, 900, now=now - 60))
        subprocess.run(["ssh-keygen", "-q", "-Y", "sign", "-f", str(key), "-n", "osiris-grant", str(g)],
                       check=True, capture_output=True)
        return g

    def auth(g):
        return eb.grants_authorizer(str(g), allowed_signers=str(signers),
                                    ledger_path=str(tmp_path / "auth.jsonl"), now=now,
                                    clock=lambda: (True, "test clock"))

    wrong = eb.evaluate_candidate(res, str(tmp_path / "l.jsonl"), authorizer=auth(grant("cd" * 32, "wrong.json")))
    assert wrong.status == "VERIFIED_PENDING_AUTHORIZATION"
    g = grant(dg, "right.json")
    ok = eb.evaluate_candidate(res, str(tmp_path / "l.jsonl"), authorizer=auth(g))
    assert ok.status == "PROMOTED"
    replay = eb.evaluate_candidate(res, str(tmp_path / "l.jsonl"), authorizer=auth(g))
    assert replay.status == "VERIFIED_PENDING_AUTHORIZATION" and "already used" in " ".join(replay.reasons)
    assert tree_digest(str(repo)) == tree_digest(str(repo))
