import json
import os
import pytest
from osiris_cli import qldpc_sweep as sw
from osiris_cli import qldpc_adapter as qa

REPO = qa.DEFAULT_REPO
needs_repo = pytest.mark.skipif(not os.path.isdir(os.path.join(REPO, "verify")), reason="challenge repo absent")
PROPOSAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "docs", "qldpc", "PROPOSED_bb-w6-unrestricted-n60-126.json")


def test_sampler_is_deterministic_bounded_and_canonical():
    a, b = sw.sample_specs(50), sw.sample_specs(50)
    assert [s.key() for s in a] == [s.key() for s in b] and len({s.key() for s in a}) == 50
    for s in a:
        ell, m = s.params["l"], s.params["m"]
        assert 60 <= 2 * ell * m <= 126 and len(s.params["A"]) == 3 and len(s.params["B"]) == 3
    # translating A and swapping A<->B give the same canonical key
    A, B = [(0, 0), (1, 2), (3, 1)], [(0, 1), (2, 2), (4, 0)]
    A2 = [((i + 2) % 6, (j + 1) % 6) for i, j in A]
    assert sw.canonical_key(6, 6, A, B) == sw.canonical_key(6, 6, B, A2)


def test_materialize_keeps_parameters_and_lists_every_change():
    prop = json.load(open(PROPOSAL))
    obj, changes = sw.materialize_campaign(prop)
    c, p = obj["campaign"], prop["campaign"]
    assert c["status"] == "active" and any("status" in x for x in changes)
    assert c["constraints"]["n"] == p["constraints"]["n"] and c["constraints"]["min_distance"] == 6
    assert {k: v for k, v in p["methods"]["screening"].items() if k != "use"} == c["methods"]["screening"]
    assert c["budget"]["candidates_screened"] == 400 and c["stopping"] == p["stopping"]
    assert "forbidden" in c["notes"] and "gate_candidates" in c["notes"]


def test_guards_refuse_low_ram_and_full_disk(tmp_path):
    with pytest.raises(sw.SweepStop):
        sw.guard(str(tmp_path), mem=lambda: 1000, disk=lambda p: (50000, 0.1))
    with pytest.raises(sw.SweepStop):
        sw.guard(str(tmp_path), mem=lambda: 5000, disk=lambda p: (100, 0.99))
    sw.guard(str(tmp_path), mem=lambda: 5000, disk=lambda p: (50000, 0.1))


def test_progress_requires_passed_exit0_hash_and_no_duplicate():
    ok = {"passed": True, "validator": {"source_sha256": "h"}, "gates": {"dedup": {"exact_duplicate_of": None}}}
    assert sw._is_progress(ok, 0, "h")
    assert not sw._is_progress(dict(ok, passed=False), 1, "h")
    assert not sw._is_progress(ok, 1, "h")
    assert not sw._is_progress(ok, 0, "other")
    assert not sw._is_progress(dict(ok, gates={"dedup": {"exact_duplicate_of": "x.json"}}), 0, "h")
    assert not sw._is_progress(None, None, "h")


@needs_repo
def test_stop_after_eight_consecutive_no_progress_resume_and_no_retry(tmp_path):
    import subprocess
    clone = str(tmp_path / "clone")
    obj, _ = sw.materialize_campaign(json.load(open(PROPOSAL)))
    rel = sw.make_checkout(REPO, clone, obj)
    repo = qa.ChallengeRepo.open(clone, run_integrity=False)
    d = qa.load_direction(repo, rel)
    specs = sw.sample_specs(400)
    calls = []
    def gate(r, path):
        calls.append(path)
        return {"passed": False, "validator": {"source_sha256": repo.validator_sha256},
                "gates": {"dedup": {"exact_duplicate_of": "x.json"}, "refute": {"seed": 7, "refuted": False}},
                "labels": ["dup"]}, 1, ""
    cfg = sw.SweepConfig(out_dir=str(tmp_path / "out"), staging_root=str(tmp_path / "stage"),
                         screen_fn=lambda HX, HZ, t, s: 8, gate_fn=gate)
    res = sw.run_sweep(repo, d, specs, cfg)
    assert res["gate_evaluations"] == 8 and "eight consecutive" in res["stop_reason"] and not res["accepted"]
    assert len(calls) == 8
    res2 = sw.run_sweep(repo, d, specs, cfg)          # resume: nothing is re-run, nothing is retried
    assert len(calls) == 8 and res2["gate_evaluations"] == 8
    assert qa.Evidence.verify(os.path.join(cfg.out_dir, "evidence.jsonl"))
    assert subprocess.run(["git", "-C", REPO, "status", "--porcelain"], capture_output=True, text=True).stdout == ""
