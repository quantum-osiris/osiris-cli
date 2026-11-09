# Execution interpretation: osiris-bb-w6-unrestricted-n60-126

Written 2026-11-09 BEFORE any candidate is generated. Frozen proposal: commit 4dbf13cb,
`docs/qldpc/PROPOSED_bb-w6-unrestricted-n60-126.json`, sha256 55bfb2999b048246f10c83b8ee724cc041233f1cff34c3faac66e27f8c893532
(unchanged). Owner approval: given in conversation 2026-11-09 with refinements, applied below.

## Campaign loading
The challenge repo's schema (`schema/campaign.schema.json`, `additionalProperties: false`) rejects four fields
of the proposal (`validation`, `prior`, `forbidden`, and non-schema objective/budget/screening keys). Nothing is
bypassed: `qldpc_sweep.materialize_campaign` produces a schema-valid copy, listing every transformation, with
unrepresentable fields carried verbatim in `notes`. Only substantive edit: `status` -> `active` (owner approval).
The copy is added to a throw-away `git clone --local` of the challenge repo; the repo's own `load_campaign`
validates it. The canonical challenge repo, its `verify/` and its existing campaign files are not written.
No challenge rule prohibits externally defined campaigns (AGENTS.md / campaigns README: campaigns are JSON files
validated by the repo loader); if the repo loader rejects the materialised file the run stops.

## Parameters (unchanged from the proposal)
Family BB; 2*l*m in [60,126], l<=m; A and B each 3 distinct monomials; up to 400 screened, 12 gate
evaluations, 3 GB gate rlimit, one process at a time, 6 h wall from the first evidence row; surrogate 200 trials
at seeds 101, 211, 307 (ranking only). Sampler: Python `random.Random(101)`, candidates canonicalised under
monomial translation and A<->B swap, duplicates skipped. 101 is a sampler seed, not a validation seed.
Candidates with k=0 or n/weight outside the campaign are recorded and are not gated.

## Gate selection and ranking (deterministic)
Eligible: k>0 and min over the 3 surrogate seeds >= campaign min_distance (6). Ranked by k*d^2/n using that
minimum (a conservative reading of an upper bound), ties by spec key; the top 12 are gated in order, one at a time.
If fewer than 12 are eligible, fewer are gated; that is reported as is.

## Progress and stopping
Progress = official verdict `passed: true`, exit code 0, `validator.source_sha256` equal to the checked-out gate,
and `gates.dedup.exact_duplicate_of` null. (The gate's `passed` already excludes exact duplicates.)
Stop at the first of: 3 accepted; 8 consecutive official evaluations without progress (gate errors count as no
progress); 12 evaluations; 400 screened and no eligible candidates left; 6 h wall; safe stop on RAM < 1.5 GB or
disk free < 2 GB / > 95 % used. Wall time is measured from the first evidence row and includes pauses.

## Integrity
A gate evaluation is never repeated and validation seeds are never chosen: the gate draws its own. Errors are
recorded and not retried. Accepted candidates are staged only under the archive's `staged/` directory (outside
the challenge repo). "Board-advancing" is reported separately from "accepted" and is `gates.novelty.board_advancing`
non-null. Distances are upper bounds; literature novelty is unverified. Gate peak memory is reported as the
running `ru_maxrss` of all children so far (an upper envelope, not per-candidate).
Resumable: all state is replayed from the hash-chained evidence file.
