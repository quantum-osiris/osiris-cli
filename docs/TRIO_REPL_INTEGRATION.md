# `/trio` REPL integration: design and readiness

Status: **implemented as a standalone module (`osiris_cli/trio_command.py`), not wired into the live
REPL.** `osiris_repl.py` is unchanged; a test asserts it does not reference `/trio`.

## Behavior (all covered by `tests/test_trio_command.py`)
| form | does | does not |
|---|---|---|
| `/trio <repo>` / `/trio plan <repo> [--patch F]` | read files, print agent digests, budgets, capabilities, **plan digest** | spawn a process, open a socket, call a model, write anything |
| `/trio run <repo> --patch F --out DIR --approve <plan_digest>` | run the repo's tests on the patch in a copy, record a lineage row under DIR | apply the patch, promote, use a provider/paid service, write outside DIR |

* **Approval** is the plan digest (sha256 over the repo tree digest, patch digest, the three agent
  spec digests and the budget). It is recomputed at `run`; any change to the repo, patch, agents or
  budget invalidates it. It is the same idea as a digest-bound grant, typed by the human.
  *The repo has no `HumanApproval` class*: that lives in the home tree's
  `osiris/agents/gateway/` (not part of this repository). If you want the exact gateway object, say so;
  the adapter point is `run_trio_command`'s approval check.
* **Execution risk, stated plainly:** `run` executes the repo's own test suite. It uses a scrubbed
  environment, a throwaway copy and a wall-clock timeout; it is *not* a container. Use only on repos
  you would run `pytest` in anyway. Symlinks in the repo are refused; size is bounded (2000 files / 50 MB).
* **No model use:** the Developer's proposal is a human-supplied JSON file; any provider other than
  `none` is refused. Promotion needs a separate signed SYS-021 grant (`osiris_cli.grants`) bound to
  the candidate digest and is not reachable from `/trio`.
* `--out` is mandatory and must be outside the repo: no hidden default location, nothing under `~/.osiris`.

## Wiring (NOT applied; for review)
1. `osiris_repl.py` command list (near line 1882): add `"/trio"`.
2. Dispatch (near line 1981, beside `/apply`):
   ```python
   if cmd == "/trio":
       from osiris_cli.trio_command import run_trio_command, split, TrioCommandError
       try:
           print(run_trio_command(split(rest), workspace_root=self.workspace))
       except TrioCommandError as e:
           print(f"[!] {e}")
       continue
   ```
3. Do **not** add it to any model-visible tool list or let a model-originated turn invoke it; the
   approval digest must be typed by a human.

## Review checklist before wiring
- [ ] Reviewer agrees `--approve <digest>` satisfies "HumanApproval" or wants the gateway object.
- [ ] Reviewer accepts subprocess (non-container) test execution for human-selected repos.
- [ ] REPL help text and `/home` entry added; one REPL integration test using the existing harness.
- [ ] Decide whether `workspace_root` should be the REPL's `--workspace` (recommended) or cwd.

## Readiness
Engine, Phase 0 adapter, bridge and command all pass their regression tests. Gaps: no container
sandbox; no model-backed Developer; the command is not yet exercised through the REPL harness.
