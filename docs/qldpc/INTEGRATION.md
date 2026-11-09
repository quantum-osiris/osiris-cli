# qLDPC autoresearch integration (OSIRIS adapter)

Branch `feature/qldpc-autoresearch`. Adds `osiris_cli/qldpc_adapter.py`, `tests/test_qldpc_adapter.py`,
`scripts/demo_qldpc_adapter.py`. Nothing in the OSIRIS REPL, governance modules, learning state or the
dna::}{::lang compiler is touched.

## 1. Sources (read before building; challenge checkout `~/github-backup/qldpc-challenge` @ `0d9808ed`, clean)

* `AGENTS.md` "Choose the workflow": with no explicit publication authorization the applicable workflow is
  **unattended autoresearch** (`research/AUTORESEARCH.md`): stage in `research/candidates/`, never write
  `codes/`, never commit or open a PR. No instruction to submit was given, so none is implemented.
* `research/QUICKSTART.md` (there is no top-level QUICKSTART.md): the loop `pick direction -> build (HX,HZ)
  -> estimate distance -> package -> VALIDATE -> stage`; "No code is a find until
  `verify/validate_candidate.py` returns `passed: true`"; persist through `submit.make_submission` +
  `save_submission`; stage under `coordination.staging_dir()`.
* `GEMINI.md` (binds all agents in that repo): you propose, the human commits; no installs without approval;
  never read `.env`/credentials; no network in tests; read-only first pass. Honoured: the adapter runs no git
  write, installs nothing, and the challenge checkout's `git status` was identical before and after (tests
  assert it).
* `verify/validate_candidate.py` (docstring + `validate_candidate`, `main`): the gate is
  `qldpc_verify.verify(refute=True)` + dedup + novelty label; `passed = verify_ok and not refuted and not
  exact_dup`; verdict carries `gates`, `labels`, `validator.source_sha256`, and a random refutation seed.
  `verify/check_validator_integrity.py` pins 31 files (it reported "matches the pin").
* `research/campaigns/*/campaign.json` + `campaign.load_campaign`: the permitted direction. Applicable one:
  `smoke-bb-72` (family `bivariate-bicycle`, n 60-80, weight <= 6, status `draft`, budget 1 candidate).
* `./qldpc targets` (read-only) lists open board cells; the adapter does not choose cells, the campaign does.

## 2. What the adapter does (and refuses)

1. `ChallengeRepo.open`: records git rev and dirty paths, hashes `validate_candidate.py`, runs the repo's own
   integrity check, and **refuses** if anything under `verify/` is modified (this caught a real bug in my first
   draft: `git status --porcelain` output was stripped and a modified gate went unnoticed; test added).
2. `load_direction`: through the repo's validating loader; campaign must live in the repo.
3. Construct (`bb.build_bb`) -> screen (`surrogate.distance_rand`, recorded only as
   `screen_distance_upper_bound`, a ranking signal; top-k by it) -> package (`make_submission`,
   `confidence="upper_bound"`, witness embedded) -> **official gate run as a subprocess of the repo's own
   script**, rlimit-bounded (3 GB address space, 600 s), scrubbed environment.
4. Classification: `VALIDATED` only if the verdict says `passed: true`, the process exit code is 0 **and** the
   verdict's `validator.source_sha256` equals the checked-out gate's hash. Else `REFUTED` (the gate's own
   refutation fired), `REJECTED` (e.g. exact duplicate), `UNPACKAGEABLE` (kit refused, e.g. k=0),
   `NOT_PERMITTED` (family/n/weight outside the campaign), `SCREENED_OUT` (rank/budget), `GATE_ERROR`.
5. Only `VALIDATED` candidates are staged (via `coordination.staging_dir/unique_path` + `save_submission`).
   Everything else, with its full verdict and the packaged doc (witness), lives in `--out/evidence.jsonl` and
   `--out/candidates/`, labelled "NOT a discovery". VALIDATED rows carry "upper bound / literature novelty
   UNVERIFIED / staged for human review".
6. Bounds: campaign `budget.candidates_screened` and a config cap on candidates, top-k to the gate, wall budget,
   RAM floor (`MemAvailable` >= 1.5 GB), sequential (no concurrency). Output dir must be outside the challenge
   repo, not `verify/`, not `~/.osiris`.
7. Evidence rows (hash-chained, tamper test included): repo rev + dirty paths, validator hash + integrity,
   environment (python/numpy/scipy/jsonschema versions, platform, RAM), campaign hash, candidate digest
   (the repo's `coordination.content_digest`), full gate verdict and exit code.

Not done, by instruction: no custom validator, no distance claims of our own, nothing under `verify/` edited,
no submission/PR/publish, no paid hardware, no canonical OSIRIS state touched, no GPU use.

## 3. Evidence (demo run, `scripts/demo_qldpc_adapter.py`, 2026-11-09)

| Scenario | Candidate | Surrogate UB (ranking only) | Gate | Outcome |
|---|---|---|---|---|
| A. real board (rev `0d9808ed`) | public BB [[72,12,6]] | 6 | `passed: false`, `exact_duplicate_of: 72-12-6.json`, refutation found nothing lighter (5380 trials) | **REJECTED** (already on the board; not a discovery) |
| B. byte-identical gate, **empty** board fixture | same code | 6 | `passed: true`, exit 0, validator hash match | **VALIDATED** (plumbing fixture only; the code is public, novelty meaningless here) |
| B. | BB l=3,m=5 | n=30 outside campaign range 60-80 | not sent | NOT_PERMITTED |
| (test) | BB l=3,m=5 k=0 attempt | | | UNPACKAGEABLE (kit refused) |

Nothing here is a screened-and-new code. There are **zero** real validated discoveries: the only `passed: true`
verdict was produced on an artificially empty board. The adapter's value so far is that the loop runs end to
end, refuses wrongly, and keeps evidence; a real search (a family/parameter sweep over a campaign that is not
`draft`) has not been run.

Tests: `tests/test_qldpc_adapter.py` (14): repo state/integrity; direction from repo; duplicate rejected and
checkout untouched; VALIDATED staged with honest labels; unpackageable / not-permitted recorded; five simulated
gate outputs (refuted, rejected, wrong-validator-hash, passed-but-nonzero-exit, no verdict) never stage; bounds
(candidate cap, RAM, wall); output confinement; modified `verify/` refused; evidence tamper detection. ~2 min
(the real-board gate dominates, ~110 s, because the dedup/signature check walks the whole board).

## 4. Limitations

* Surrogate distances are upper bounds; the gate's refutation is random-seeded (a different seed could find a
  lighter logical). `passed: true` never certifies exact distance, and literature novelty is unverified.
* Only the `bivariate-bicycle` constructor is registered; other families need a registered builder.
* The gate subprocess is rlimit-bounded but not in the bwrap sandbox (the repo is a trusted checkout of the
  user's own); if untrusted checkouts matter, reuse `osiris_cli.sandbox`.
* This workstation shows ~7.4 GB RAM total (the brief said 16 GB); bounds are set for the smaller figure.
* `research/candidates/` is gitignored local staging and can never be cited as evidence (repo rule).
* Challenge-side checks (prose, authorship) apply only to a later human-driven submission and are not run.

## 5. dna::}{::lang

Optional and future. Whether a qLDPC candidate can be expressed as a dna::}{::lang (dnalang-core) circuit/rule
gene, and what its canonical-JSON digest would mean for a parity-check matrix, has **not** been verified. A
later adapter would map `CandidateSpec` to a gene and keep the repo's gate as the only authority; until
semantics are checked against dnalang-core tests, no integration is claimed.

## 6. Next steps (need approval)

1. Choose a non-draft campaign / board cell via `./qldpc targets`, pre-register the criterion, then run a small
   bounded sweep (seeds recorded; tuning seeds separate from evaluation seeds per research-program discipline).
2. Add lifted-product / coset builders as registered families.
3. If a candidate validates, a human promotes it (`research/kit/promote.py` + CONTRIBUTING.md) — out of scope here.
