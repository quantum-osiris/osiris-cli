# Preregistration: bounded BB sweep (direction `smoke-bb-72`)

Written 2026-11-09, BEFORE any search. Status: **NOT EXECUTED. Blocked by the campaign gate.**

## Gate findings (read-only inspection, challenge rev 0d9808ed, tree clean)
- `smoke-bb-72`: `status: draft`. `w8-2dlocal-n700-1000`: `status: draft`. No non-draft campaign exists.
- Rule (research/campaigns/README.md): campaigns stage for review and never open a PR. A draft campaign is
  not cleared for unattended autoresearch, and I do not edit `campaign.json` to change that: it is the
  challenge repo's file and promoting it is the owner's decision.
- `w8-2dlocal-n700-1000` is excluded on feasibility as well: 500 CPU-h / 72 h budget, n=700-1000,
  boundary_engine at 100000 trials. This host has ~7.4 GB RAM (5.9 GB available at inspection).
- `./qldpc targets` could not run (`uv` not installed; no installs allowed), so board cells were not read
  from the live board. Nothing below claims a board cell is open.
- The D:\ paths in the request are not mounted in WSL (`/mnt/d` is empty), so those documents were not read.

## Chosen direction (conditional on the owner marking the campaign non-draft)
`smoke-bb-72`: bivariate-bicycle, CSS, n in [60,80], max check weight 6, d >= 4. Feasible within ~1.5 GB.
- Eligibility: CSS, n 60-80, weight <= 6, d >= 4, official gate `passed: true`.
- Family and bounds: BB only (`bb.build_bb`), ell,m with 2*ell*m in [60,80]; A,B each 3 monomials in x^i y^j.
- Budget: <= 16 candidates, screen_trials 200, top_k 2, wall 900 s, gate timeout 600 s, 3 GB AS rlimit.
- Seeds: screening seeds 101, 211, 307 (tuning range; evaluation uses fresh seeds 0-4). Fixed per candidate.
- Ranking: surrogate distance used for ordering only. A surrogate value is never reported as a distance.
- Validation: only `verify/validate_candidate.py`, source sha256 pinned and `check_validator_integrity.py`
  run before the sweep; `verify/` must be unmodified.
- Success: >= 1 candidate with verdict `passed: true`, exit 0, matching validator sha256.
- Rejection: any non-pass, `refuted`, exact duplicate, or gate error is recorded and not staged.
- Stopping: budget exhausted, or 8 consecutive experiments without a passing candidate.
- Prior: known BB codes in this window (e.g. [[72,12,6]]) are expected to be exact duplicates; zero new
  discoveries is the expected, reportable outcome.
- Artifacts: evidence JSONL and packaged candidates under an out-of-repo directory; passing candidates staged
  only in `research/candidates/` (gitignored). No commits, PRs, publication or hardware use.

## To execute
The owner sets the campaign non-draft; then run `osiris_cli.qldpc_adapter.run_campaign` with the settings above.

## Addendum A (2026-11-09, before the run): owner authorization and binding budget
- The owner authorized marking `smoke-bb-72` non-draft. Only the `status` field was changed
  (`draft` -> `active`); campaign.json sha256 8d37142f... -> 67ab19fc.... The challenge repo is not committed.
- The campaign's own budget is binding and narrower than the plan above: `candidates_screened: 1`,
  screening seed [0], 200 trials, stop on `candidates_found: 1`. Exactly ONE candidate is run:
  BB [[72,12,6]] (l=6, m=6, A = x^3+y+y^2, B = y^3+x+x^2). The 16-candidate and seed-101 settings above
  are superseded. Expected verdict: REJECTED as an exact duplicate of board entry 72-12-6.
