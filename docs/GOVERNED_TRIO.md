# Governed three-agent slice (Research → Developer → Critic)

Branch `feature/governed-trio` (worktree `~/osiris-cli-wt/governed-trio`), from `main` @ `9896f04a`.
Nothing here is merged, deployed, or wired into the live REPL.

## 1. Baseline (before any change)

`main` @ `9896f04a`, `osiris_env` pytest: **1589 passed, 13 skipped, 1 failed**
(`tests/test_exec_policy.py::test_symlink_named_ls_pointing_at_rm_is_treated_as_rm`, pre-existing,
unrelated). Learning ledger `~/.osiris/living/learning_ledger.jsonl`: 1 line; `notes.jsonl`: 1 note
(`note-9a9bd331`, status `active`). None of those files were written by this work.
After: **1595 passed, 13 skipped, same 1 failure** on this branch (the count differs from the
baseline run partly because the baseline ran in the main checkout, which also holds untracked
`tests/test_nclm_corpus.py`; this worktree does not).

## 2. Layer inventory (what exists, as found)

| Layer | State | Where |
|---|---|---|
| L0 Intake | implemented | `orchestration/run_record.py` (`TaskRunRecord`: budget, walltime, permissions) |
| L1 Intent | implemented, keyword/provider-based | `orchestration/agents.py:34` `IntentAgent`, `intent_router.py` |
| L2 Orchestration | implemented, single linear engine | `orchestration/engine.py:41` `GovernedAgentEngine` |
| L3 Evidence | partial: provenance records + claims register; no gate on agent prose until this branch | `evidence_provenance.py`, `osiris_cli/claims.py` |
| L4 Proposal | implemented | `schemas.py` `CodeProposal`, `agents.py:95` |
| L5 Verification | **partial**: `VerificationAgent` (`agents.py:158`) only AST-checks and `compile()`s; it never runs tests | `agents.py`, `critic_verifier.py` |
| L6 Benchmark | exists separately, not wired to agents | `dnalang/loop.py`, `dnalang/receipts.py` |
| L7 Promotion | exists separately (`PromotionPolicy`, `promotion_gate.py`), not wired to agents | `dnalang/receipts.py:84`, `promotion_gate.py:193` |
| L8 Ledger | implemented (hash-chained genome ledger) | `agents.py` `MemoryLineageAgent` → `genome_ledger` |

## 3. Approved-note retrieval defect (fixed, commit `a5cdc075`)

Root cause (confirmed in source): `Knowledge` (`osiris_cli/knowledge.py`) built its index from markdown
files only; `paste_learning.confirm_proposal` writes approved notes to `notes.jsonl`, which nothing
read for conversation. So the note was authorized and stored but never retrieved.
Fix: active notes join `lookup()` only (not `brief()`/`on_record()`), source `approved-note:<id>`
(stable id), labelled *user-provided, unverified*, bounded by the existing per-message
`NOTES_PER_MESSAGE`/chunking, tombstoned/malformed/non-`active` records skipped, never turned into
training data. Tests: `tests/test_knowledge_notes.py` (synthetic only).
Caveat: this makes the note *reachable*; whether the mentor model reproduces its contents faithfully
is a separate, still-unmeasured property.

## 4. What was built (commit `29b2a016`)

* `orchestration/agent_spec.py` — `AgentSpec` (identity, version, role, capabilities using
  `CapabilityGovernor` names, tool permissions, provider/model, schemas, token/deadline budgets,
  `evidence_required`), `AgentRun` with a checked state machine
  (`CREATED→RUNNING→DONE|FAILED|REJECTED|QUARANTINED`), `RunBudget` (max agents, depth, wall
  time — no recursive spawning), `AgentReceipt` (input/output digests + transitions).
* `orchestration/evidence_firewall.py` — deterministic checks over report sentences:
  test-result claims vs `ExecutionEvidence`; completed-action claims; numbers/intervals needing an
  `[evidence:<id>]` citation whose artifact contains those numbers; backend names; restatements of
  registered REFUTED/ARTIFACT/NULL claims (skipped when the sentence negates them). Verdicts
  `SUPPORTED/REFUTED/ARTIFACT/UNVERIFIED/UNMEASURED`; any REFUTED/ARTIFACT → REJECT, else any
  UNVERIFIED/UNMEASURED → QUARANTINE. No model is consulted.
* `orchestration/trio.py` — `run_trio(repo, propose_fn, out_dir)`: frozen baseline run → Research
  (traceback frames + modules imported by failing tests, each with sha256) → Developer
  (`propose_fn`; `provider_propose_fn(provider)` adapts an existing `osiris_cli.providers`
  provider) in a sandbox copy → Critic (fresh sandbox, own pytest run, fail→pass / pass→fail vs
  baseline, patch-scope/size checks, firewall over the Developer's report, optional model critique
  recorded as non-authoritative). Developer cannot edit/add tests, escape the repo, or pass the AST
  firewall with `exec/eval/os.system`. Outcome is at best `PENDING_HUMAN_APPROVAL`; the repo's
  tree digest is asserted unchanged; the run record (`record.json`, candidate diffs) and a
  hash-chained `receipts.jsonl` go in `out_dir` (refused if inside the repo). Candidates are
  `UNMEASURED` until the runner measures them; rejected/quarantined ones keep their diff + evidence.

Run it: `pytest tests/test_governed_trio.py -v` (12 tests, ~3 s). Covered: honest fix → pending
approval; **Developer claims "all tests pass / I ran pytest" with a failing patch → REJECTED**;
**invented 37.2 % / CI / `ibm_torino` figures with a working patch → QUARANTINED**; test-editing and
path escape refused; dangerous AST refused; regression vs baseline rejected; already-green repo;
out_dir-in-repo refused; supplementary critic never overrides; firewall verdict table; state
machine and budgets.

## 4b. dna::}{::lang integration — analysis only, nothing implemented

`dnalang-core/docs/LANGUAGE.md` defines three gene kinds (circuit, rule, regulator) with fixed
key-value fields, an organism `fitness` clause, and canonical JSON + SHA-256 per compiled target.
`osiris_cli/dnalang/loop.py` already runs a parent→candidate→receipt→promotion loop over `.dna`
genomes through `osiris.compiler`. I did **not** verify that either parser accepts extra fields
such as `capabilities` / `budget` on a key-value gene, so I have not claimed an agent genome
exists in the language. Recommended next step: an RFC (syntax example, parser change, back-compat,
tests) for an `agent` gene kind, with a typed adapter `AgentSpec ⇄ canonical organism JSON`
instead of putting orchestration inside the language runtime.

## 5. Limitations / not done

* The Research agent is deterministic (no retrieval over docs/experiment records); the Developer is
  only exercised with scripted proposers. `provider_propose_fn` is untested against a live model.
* Tests run through `subprocess` with a scrubbed environment and a copy of the repo, **not** a
  container/namespace sandbox and not via `exec_kernel`'s grant path; do not run untrusted repos.
* Firewall is regex-based: it will miss paraphrases and can flag benign numbers (it errs toward
  QUARANTINE). It does not check "confusion between observables" or contradictions beyond the claims
  register.
* The trio's receipt chain is its own file; it is not appended to the genome ledger and does not
  feed `PromotionPolicy`/`promotion_gate`, so L6/L7 are not yet wired.
* No evolutionary loop over agent prompts/routing was added; no GPU telemetry/scheduling; no
  live-REPL command. No scores were produced, so nothing is claimed as improved.

## 6. Next steps

1. Wire `run_trio` outcomes into `PromotionPolicy` + `exec_kernel` grants and the genome ledger.
2. Retrieval-backed Research agent (reuse `Knowledge` incl. approved notes, with citations).
3. `agent` gene RFC + adapter; frozen-baseline benchmark set for prompt/routing candidates.
4. Provider/GPU budgeting (Ollama unload, VRAM telemetry) behind `RunBudget`.
5. REPL surface (`/trio <repo>`), approval via the existing `HumanApproval` flow.
