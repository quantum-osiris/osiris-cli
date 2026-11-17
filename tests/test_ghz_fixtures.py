"""Scientific-correctness fixtures, kept apart from compile and execution success.

A circuit can compile and run and still be wrong. tests/fixtures/aura_ghz_proposal_2026-11-16.json
is the change a local model (gemma3:4b via AURA) proposed when asked for a 3-qubit GHZ state; it
compiled and executed, and its counts were uniform over all 8 outcomes. This file checks a
documented ideal GHZ fixture against the GHZ outcome support with an explicit finite-shot bound,
and checks that the recorded proposal fails that same check. The check is specific to these
fixtures. It is not a general correctness test for user circuits and the product does not use it.
"""
import json
import math
import os

import pytest

pytest.importorskip("dnalang")
pytest.importorskip("qiskit_aer")
pytest.importorskip("osiris_governance.github_policy")

from osiris_cli.vertical_slice import compile_source, run_slice  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
SHOTS = 4000
Z = 5.0   # half-width of the binomial bound in standard deviations (two-sided p ~ 6e-7 per check)


def ghz_support_check(counts, shots):
    """True iff counts are consistent with ideal noiseless GHZ sampling: nothing outside {000, 111}
    (the simulator is noiseless, so any such count is disqualifying) and P(000) within Z sigma of 1/2."""
    total = sum(counts.values())
    outside = sum(v for k, v in counts.items() if k not in ("000", "111"))
    p0 = counts.get("000", 0) / total
    return total == shots and outside == 0 and abs(p0 - 0.5) <= Z * math.sqrt(0.25 / shots)


def test_ideal_ghz_fixture_has_ghz_support(tmp_path):
    src = open(os.path.join(FIX, "ghz3_ideal.dna")).read()
    r = run_slice(src, shots=SHOTS, seed=11, receipts_path=str(tmp_path / "r.jsonl"))
    assert r["status"] == "COMPLETED"
    assert ghz_support_check(r["simulation"]["counts"], SHOTS), r["simulation"]["counts"]


def test_recorded_aura_proposal_compiles_runs_and_is_not_ghz(tmp_path):
    fx = json.load(open(os.path.join(FIX, "aura_ghz_proposal_2026-11-16.json")))
    comp = compile_source(fx["source"])
    assert comp["ok"] is True                                       # compilation success …
    r = run_slice(fx["source"], shots=SHOTS, seed=11, receipts_path=str(tmp_path / "r.jsonl"))
    assert r["status"] == "COMPLETED"                               # … and execution success …
    counts = r["simulation"]["counts"]
    assert not ghz_support_check(counts, SHOTS)                     # … are not correctness
    assert len(counts) == 8                                         # H on every qubit: all 8 outcomes
    # the counts observed in the notebook session were the same kind: 8 outcomes, none dominant
    obs = fx["observed_run"]["counts"]
    assert len(obs) == 8 and not ghz_support_check(obs, sum(obs.values()))


def test_support_check_bound_depends_on_shots():
    # With 10 shots the bound is wide (sigma = 0.16): a 10/0 split is only 3.2 sigma from 1/2 and is NOT
    # rejected. Few shots cannot establish the 1/2 balance; the same split at 100 shots is rejected.
    assert ghz_support_check({"000": 10, "111": 0}, 10)
    assert not ghz_support_check({"000": 100, "111": 0}, 100)
    assert ghz_support_check({"000": 2010, "111": 1990}, 4000)
    assert not ghz_support_check({"000": 1999, "111": 2000, "010": 1}, 4000)
