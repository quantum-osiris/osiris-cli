"""Bounded demonstration of osiris_cli.qldpc_adapter. Writes only under --out (outside the challenge repo).

  A. REAL BOARD   : the known public [[72,12,6]] BB code through the repo's own gate -> REJECTED (it is
                    already board entry 72-12-6.json). Expected, and proves the gate is not bypassed.
  B. EMPTY BOARD  : the same code against a byte-identical copy of the gate with an empty codes/ dir ->
                    VALIDATED. This is a plumbing fixture, NOT a discovery: the code is public and the
                    board in the fixture is artificially empty.
  C. k=0 BB code  : recorded as UNPACKAGEABLE evidence.
Usage: python scripts/demo_qldpc_adapter.py --out DIR [--repo PATH]
"""
import argparse
import dataclasses
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from osiris_cli import qldpc_adapter as qa  # noqa: E402

CAMP = "research/campaigns/smoke-bb-72/campaign.json"
BB72 = qa.bb_spec(6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)], "BB [[72,12,6]]")
BB30 = qa.bb_spec(3, 5, [(0, 0), (1, 0), (0, 1)], [(0, 0), (2, 0), (0, 2)], "BB l=3,m=5")


def fixture(real_root, dest):
    os.makedirs(os.path.join(dest, "research"))
    os.makedirs(os.path.join(dest, "codes"))                   # empty board
    ign = shutil.ignore_patterns("__pycache__")
    for rel in ("verify", "schema"):
        shutil.copytree(os.path.join(real_root, rel), os.path.join(dest, rel), ignore=ign)
    shutil.copytree(os.path.join(real_root, "research", "kit"), os.path.join(dest, "research", "kit"), ignore=ign)
    shutil.copy(os.path.join(real_root, "research", "QUICKSTART.md"), os.path.join(dest, "research"))
    shutil.copytree(os.path.join(real_root, "research", "campaigns", "smoke-bb-72"),
                    os.path.join(dest, "research", "campaigns", "smoke-bb-72"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=qa.DEFAULT_REPO)
    a = ap.parse_args()
    out = os.path.realpath(a.out)
    real = qa.ChallengeRepo.open(a.repo)
    d = dataclasses.replace(qa.load_direction(real, CAMP), max_candidates=4)
    res = {"A_real_board": qa.run_campaign(real, d, [BB72], qa.RunConfig(
        out_dir=os.path.join(out, "A_real_board"), staging_root=os.path.join(out, "A_stage"), screen_trials=200))}
    tmp = tempfile.mkdtemp(prefix="qldpc_fixture_")
    try:
        fixture(real.root, os.path.join(tmp, "repo"))
        fx = qa.ChallengeRepo.open(os.path.join(tmp, "repo"), require_git=False, run_integrity=False)
        res["B_empty_board_fixture"] = qa.run_campaign(fx, dataclasses.replace(qa.load_direction(fx, CAMP), max_candidates=4),
                                                       [BB72, BB30], qa.RunConfig(
            out_dir=os.path.join(out, "B_fixture"), staging_root=os.path.join(out, "B_stage"), screen_trials=200))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
