"""qLDPC adapter regression tests. The real challenge checkout is used read-only (skipped when absent);
a fixture copy with an EMPTY board stands in where a passing gate verdict is needed, so a known public
code can exercise the plumbing. That is a test of the adapter, not a discovery."""

import json
import os
import shutil
import subprocess

import pytest

from osiris_cli import qldpc_adapter as qa

REAL = qa.DEFAULT_REPO
pytestmark = pytest.mark.skipif(not os.path.isfile(os.path.join(REAL, "verify", "validate_candidate.py")),
                                reason="qLDPC challenge checkout not present")

BB72 = qa.bb_spec(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)], "BB [[72,12,6]] fixture")
BB30 = qa.bb_spec(3, 5, [(0, 0), (1, 0), (0, 1)], [(0, 0), (2, 0), (0, 2)], "k=0 example")
CAMP = "research/campaigns/smoke-bb-72/campaign.json"


def porcelain(root):
    return subprocess.run(["git", "-C", root, "status", "--porcelain"], capture_output=True, text=True).stdout


@pytest.fixture(scope="module")
def real():
    return qa.ChallengeRepo.open(REAL)


@pytest.fixture
def fixture_repo(tmp_path):
    f = tmp_path / "fixrepo"
    (f / "research").mkdir(parents=True)
    (f / "codes").mkdir()                                  # EMPTY board
    for rel in ("verify", "schema"):
        shutil.copytree(os.path.join(REAL, rel), f / rel, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(os.path.join(REAL, "research", "kit"), f / "research" / "kit",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(os.path.join(REAL, "research", "QUICKSTART.md"), f / "research" / "QUICKSTART.md")
    shutil.copytree(os.path.join(REAL, "research", "campaigns", "smoke-bb-72"),
                    f / "research" / "campaigns" / "smoke-bb-72")
    # the gate under test is byte-identical to the real one
    assert open(f / "verify" / "validate_candidate.py", "rb").read() == \
        open(os.path.join(REAL, "verify", "validate_candidate.py"), "rb").read()
    return qa.ChallengeRepo.open(str(f), require_git=False, run_integrity=False)


def cfg(tmp_path, **kw):
    return qa.RunConfig(out_dir=str(tmp_path / "out"), staging_root=str(tmp_path / "stage"), screen_trials=100, **kw)


def rows(tmp_path):
    return [json.loads(x) for x in open(tmp_path / "out" / "evidence.jsonl")]


def test_repo_state_and_integrity_recorded(real):
    s = real.snapshot()
    assert len(s["git_rev"]) == 40 and s["validator_integrity_ok"] and len(s["validator_sha256"]) == 64


def test_direction_comes_from_the_repo_campaign(real):
    d = qa.load_direction(real, CAMP)
    assert d.families == ("bivariate-bicycle",) and d.n_range == (60, 80) and d.max_check_weight == 6
    with pytest.raises(qa.AdapterError):
        qa.load_direction(real, "/etc/passwd")


def test_known_board_code_is_rejected_as_duplicate_and_nothing_is_staged(real, tmp_path):
    before = porcelain(REAL)
    s = qa.run_campaign(real, qa.load_direction(real, CAMP), [BB72], cfg(tmp_path))
    assert s["counts"] == {"REJECTED": 1} and s["validated"] == []
    r = rows(tmp_path)
    c = next(x for x in r if x["kind"] == "CANDIDATE")["payload"]
    assert c["gate_verdict"]["gates"]["dedup"]["exact_duplicate_of"] and c["labels"][0] == "NOT a discovery"
    assert c["gate_verdict"]["validator"]["source_sha256"] == real.validator_sha256
    assert not os.path.exists(tmp_path / "stage")
    assert porcelain(REAL) == before, "the challenge checkout must be untouched"
    assert qa.Evidence.verify(s["evidence"])
    start = r[0]["payload"]
    assert start["repo"]["git_rev"] == real.git_rev and start["environment"]["python"]


def test_validated_candidate_is_staged_with_honest_labels_on_an_empty_board(fixture_repo, tmp_path):
    s = qa.run_campaign(fixture_repo, qa.load_direction(fixture_repo, CAMP), [BB72], cfg(tmp_path))
    assert s["counts"] == {"VALIDATED": 1}
    path = s["validated"][0]
    doc = json.load(open(path))
    assert path.startswith(str(tmp_path / "stage")) and (doc["n"], doc["k"], doc["distance"]["d"]) == (72, 12, 6)
    assert doc["distance"]["X"]["confidence"] == "upper_bound" and doc["distance"]["X"]["witness"]
    c = [x for x in rows(tmp_path) if x["kind"] == "CANDIDATE"][0]["payload"]
    assert c["status"] == "VALIDATED" and c["gate_verdict"]["passed"] is True and c["gate_exit_code"] == 0
    assert "literature novelty UNVERIFIED" in c["labels"] and len(c["candidate_digest"]) == 64
    assert os.path.exists(tmp_path / "out" / "candidates")           # the witness doc is kept as evidence too
    assert "discover" not in json.dumps(s).lower().replace("not a discovery", "")


def test_unpackageable_and_not_permitted_are_recorded_not_dropped(fixture_repo, tmp_path):
    other = qa.CandidateSpec("lifted-product", {"x": 1})
    big = qa.bb_spec(12, 12, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)], "n=288")
    import dataclasses
    d = dataclasses.replace(qa.load_direction(fixture_repo, CAMP), max_candidates=5)
    s = qa.run_campaign(fixture_repo, d, [BB30, other, big], cfg(tmp_path))
    st = {x["payload"]["spec_key"]: x["payload"]["status"] for x in rows(tmp_path) if x["kind"] == "CANDIDATE"}
    assert st[BB30.key()] in ("UNPACKAGEABLE", "SCREENED_OUT", "NOT_PERMITTED")
    assert st[other.key()] == "NOT_PERMITTED" and st[big.key()] == "NOT_PERMITTED"
    assert s["validated"] == []


def _fake_gate(verdict, rc, tmp_hash=None):
    def g(repo, path):
        v = json.loads(json.dumps(verdict)) if verdict is not None else None
        if v is not None:
            v.setdefault("validator", {})["source_sha256"] = tmp_hash or repo.validator_sha256
        return v, rc, "stderr"
    return g


@pytest.mark.parametrize("verdict,rc,hash_,expect", [
    ({"passed": False, "gates": {"refute": {"refuted": True, "detail": "lighter logical, weight 4"}}, "labels": []},
     1, None, "REFUTED"),
    ({"passed": False, "gates": {"refute": {"refuted": False}}, "labels": ["duplicate: x"]}, 1, None, "REJECTED"),
    ({"passed": True, "gates": {}, "labels": []}, 0, "0" * 64, "GATE_ERROR"),      # verdict from another validator
    ({"passed": True, "gates": {}, "labels": []}, 1, None, "REJECTED"),            # passed but nonzero exit: not trusted
    (None, 2, None, "GATE_ERROR"),
])
def test_only_a_trusted_passed_true_verdict_stages(fixture_repo, tmp_path, verdict, rc, hash_, expect):
    c = cfg(tmp_path)
    c.gate = _fake_gate(verdict, rc, hash_)
    s = qa.run_campaign(fixture_repo, qa.load_direction(fixture_repo, CAMP), [BB72], c)
    assert s["counts"] == {expect: 1} and s["validated"] == [] and not os.path.exists(tmp_path / "stage")
    cand = [x for x in rows(tmp_path) if x["kind"] == "CANDIDATE"][0]["payload"]
    assert "NOT a discovery" in cand["labels"]


def test_bounds_top_k_candidate_cap_ram_and_wall(fixture_repo, tmp_path, monkeypatch):
    d = qa.load_direction(fixture_repo, CAMP)
    with pytest.raises(qa.AdapterError, match="exceeds the bound"):
        qa.run_campaign(fixture_repo, d, [BB72, BB30], cfg(tmp_path))          # campaign budget is 1 candidate
    monkeypatch.setattr(qa, "mem_available_mb", lambda: 100)
    with pytest.raises(qa.AdapterError, match="RAM"):
        qa.run_campaign(fixture_repo, d, [BB72], cfg(tmp_path))
    monkeypatch.undo()
    c = cfg(tmp_path, wall_budget_s=-1)
    s = qa.run_campaign(fixture_repo, d, [BB72], c)
    assert s["counts"] == {"SCREENED_OUT": 1}


def test_output_locations_are_confined(real, fixture_repo, tmp_path):
    d = qa.load_direction(real, CAMP)
    for bad in (os.path.expanduser("~/.osiris/x"), os.path.join(REAL, "verify", "o"), REAL,
                os.path.join(REAL, "codes")):
        with pytest.raises(qa.AdapterError):
            qa.run_campaign(real, d, [BB72], qa.RunConfig(out_dir=bad, staging_root=str(tmp_path / "s")))


def test_modified_trusted_stack_is_refused(tmp_path):
    r = tmp_path / "r"
    (r / "research" / "kit").mkdir(parents=True)
    for rel in ("verify", "schema"):
        shutil.copytree(os.path.join(REAL, rel), r / rel, ignore=shutil.ignore_patterns("__pycache__"))
    (r / "codes").mkdir()
    (r / "research" / "QUICKSTART.md").write_text("x")
    (r / "research" / "kit" / "submit.py").write_text("")
    g = ["git", "-C", str(r), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "-C", str(r), "init", "-q"], check=True)
    subprocess.run(g + ["add", "-A"], check=True)
    subprocess.run(g + ["commit", "-q", "-m", "x"], check=True)
    qa.ChallengeRepo.open(str(r), run_integrity=False)                        # clean: accepted
    open(r / "verify" / "validate_candidate.py", "a").write("\n# softened gate\n")
    with pytest.raises(qa.AdapterError, match="verify/ has local modifications"):
        qa.ChallengeRepo.open(str(r), run_integrity=False)


def test_evidence_chain_detects_tampering(fixture_repo, tmp_path):
    s = qa.run_campaign(fixture_repo, qa.load_direction(fixture_repo, CAMP), [BB72], cfg(tmp_path))
    assert qa.Evidence.verify(s["evidence"])
    t = open(s["evidence"]).read().replace('"passed": true', '"passed": false')
    open(s["evidence"], "w").write(t)
    assert not qa.Evidence.verify(s["evidence"])
