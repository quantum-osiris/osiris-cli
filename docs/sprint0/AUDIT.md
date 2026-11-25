# Sprint 0 forensic audit

Audit date: 2026-11-25. Primary tree: `quantum-osiris/osiris-cli` at `c1f0444`. Machine-readable rows are in `capability_inventory.json`. A status is `VERIFIED` only where this run executed the code. Docs, Zenodo receipts, and generated papers were not treated as proof.

## What is actually here

The console entrypoint is `osiris` (`osiris_cli.osiris_repl:main`). Node Beta's acceptor (`osiris_cli/node_beta.py`) checks a signed SHA-256 challenge and replays a finished request id. `osiris_cli/sync_bridge.py` is a dual-node runbook and socket probe, not a watcher daemon. There is no Alpha/Beta watcher daemon in this repository.

The AST firewall in `osiris_cli/orchestration/ast_firewall.py` is a static deny-list over a parsed tree. It does not execute the tree, and it is not an OS sandbox. `osiris/runtime/worker.py` refuses to run when no isolation backend is configured. The private IDE repo `quantum-osiris/quantum-ide` has `worker/osiris_cli_worker.py`, which runs `dnalang.cli run` and labels that output simulated. That worker says a working directory is not a sandbox. There is no repository named `quantum-ide/osiris-ide`.

Hash-chained JSONL appends go through `osiris_cli/chain_lock.py`. QByte and organism code in this tree are local programs. Their tests passing does not make them quantum-advantage evidence. Zenodo publish clients exist; this run did not call Zenodo.

## Repos

Reached: `osiris-cli` (local), `osiris-governance`, `dnalang-core`, `organism_sim`, `bridge` (installed from the pins in `pyproject.toml`), `quantum-ide` (worker source read), `osiris-edge-bridge` (tree only), `devinphillipdavis/osiris-android` (tree only), `osiris-research-archive` (top-level listing only).

Not reached: `quantum-ide/osiris-ide` (no such repo). Branch `claude/wave0-1` is not on this remote and was not deleted.

`dnalang` 0.4.0 from pin `5ae07489` imports. The `dnalang-core` main tree observed via the API (`90b68053`) is not that pin.

## P0 status

The five named defects were not present as those functions on `main`. The unsafe shapes are now rejected by code under test, instead of being left as missing checks.

| Item | Status |
| --- | --- |
| `probe_alpha_state` semicolon/`json.dumps` shell quote | Fixed. Versioned helper `osiris_cli/remote_helpers/alpha_state_v1.py`, argv only. The old one-line `for` after a semicolon still does not parse. |
| AST firewall dynamic import | Fixed for the three REPL home-directory `exec_module` fallbacks. `IS_SANDBOX` is false. Untrusted paths are not loaded. |
| Return sync overwriting a newer chain | Fixed. `accept_return_chain` allows only an empty local chain or a strict extension of the local tip. |
| Exactly-once after execution, before ack | Fixed. Persistent experiment id and idempotency key in a JSONL ledger. A second begin returns the stored result. |
| Manifest check-then-unlink | Fixed. Atomic rename into quarantine, then verify, then delete that renamed file. |
| Strict git alignment | Fixed. Missing commit info fails closed in `probe_git` and `assert_strict_git_alignment`. |
| Success telemetry | Fixed. Both `dispatch` and `result` must belong to the current experiment id. |
| Mock Zenodo fixtures | Fixed. Mock or fixture records, including fixture confidence 0.6, are quarantined before `PublishingWorkflow` contacts Zenodo. Unmarked records are not assigned a confidence. |

## Suite

`python3 -m pytest -o cache_dir=/tmp/osiris_pytest_cache -q tests/ --tb=line -rfE` on Python 3.12.3: **1 failed, 3166 passed, 97 skipped**, exit code 1, 417.32 s. Twelve further items were skipped at collection. The one failure is `test_active_interpreters_own_environment_is_a_resolution_source`: pytest is not installed beside `/usr/bin/python3`. The Sprint 0 file `tests/test_sprint0_p0.py` passed (9 tests).

GitHub Actions run 45513791914 was red only on the Python 3.9 job: `tests/test_node_beta.py` imports `dnalang` at collection, and `dnalang-core` requires Python 3.10+. That module is now in the 3.9 skip list. `.github/workflows/dnalang-pytest.yml` installs the pinned `dnalang-core` and runs pytest on Python 3.12.

## Sprint 1

1. Deploy the versioned Alpha helper over the existing adb/ssh transports in `key_sync.py` without shell-quoting, and make return sync call `accept_return_chain` on the real evidence files.
2. Add an OS isolation backend behind `osiris/runtime/worker.py` before any untrusted candidate is executed. Keep the AST firewall as the static policy in front of that backend.
3. Decide whether to move the `dnalang-core` pin from `5ae07489` to current `main`, and record the diff. Do not treat the pin and `main` as the same tree.
4. Connect the idempotency ledger to Node Beta request ids so a retried phone request cannot start a second execution.
5. Leave QByte, Aer, and the RTX bench script classified as simulation until a pre-registered hardware run is hashed into the ledger. Do not publish mock fixtures.
