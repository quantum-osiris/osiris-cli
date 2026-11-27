# Sprint 1

Branch `cursor/sprint1-integration-c9aa`, based on current `main` after PRs #79 and #87. The `dnalang-core` pin stays `5ae07489`. See `DNALANG_CORE_PIN_REVIEW.md`.

## Wired

- `key_sync.sync_via_adb` and `sync_via_ssh` push `osiris_cli/remote_helpers/alpha_state_v1.py` and run `python3 <path>` (`adb exec-out` or `ssh` argv). They do not build a `python -c` string.
- A remote `exchanges.jsonl` is pulled into `incoming/exchanges.jsonl`, never onto the live ledger. `apply_return_chain` decides ancestry under `chain_lock` and restores the previous bytes if the rewrite fails.
- Historical exchange rows have `prev` and `hash` and no `seq`. `verify_rows` follows that hash link. It does not require `seq`, because adding `seq` would change the recorded hashes.
- Node Beta calls `begin_experiment` / `record_execution` / `acknowledge` with `osiris_request_id` as both the experiment id and the idempotency key. A claim that is still `started` returns `PRIOR_ATTEMPT_INCOMPLETE` and does not run `sha256sum` again. A finished id is replayed by the existing ledger.
- Discovery results whose job ids start with `qvm_` or `fallback_` get `evidence_class=SIMULATED`. `PublishingWorkflow` quarantines them. QByte, Aer, and the RTX bench script are not given a new execution path.

## Isolation

Chosen backend: a subprocess with POSIX rlimits (`osiris/runtime/isolation.py`, `RlimitSubprocessBackend`). `bwrap` and `nsjail` are not installed here, and this repo already refuses to run when `WorkerRunner` has no backend.

The child gets `RLIMIT_CPU`, `RLIMIT_AS`, `RLIMIT_FSIZE`, `RLIMIT_NPROC`, and `RLIMIT_NOFILE`. The backend marks itself verified only after a probe child reports the CPU limit it was given. If that probe fails, `run_untrusted` raises `IsolationUnavailable`. There is no in-process `exec` fallback.

`osiris_cli.orchestration.ast_firewall.IS_SANDBOX` stays false. The rlimit backend's own `IS_SANDBOX` is also false: rlimits are not a mount, user, or network namespace. The AST firewall remains a static check only.

## Copilot requests from PR #79

| Request | On main before this branch | This branch |
| --- | --- | --- |
| Home directory on `sys.path` can execute a planted module before the REPL refusal | Open | Fixed. Home itself is not appended. Imports of `osiris_tetra1d_benchmark`, `osiris_claim_closure`, and `osiris_autonomous_mutation` require a trusted origin. |
| `seq` rejects `living/exchanges.jsonl` | Open | Fixed. Hash-link order is enough when `seq` is absent. |
| `os.write` short write of the recovery file | Already fixed on main by `_write_all` before the live rewrite | Kept, and the temp file is read back with `_fd_matches` before the live inode is touched. |
| `record_execution` check-then-append race | Already fixed on main: one `locked_tail` covers the read, the phase check, and the write | Unchanged. |
| `ExperimentResult` mock/`qvm_`/`fallback_` jobs bypass the Zenodo gate | Open | Fixed. Those job ids are `SIMULATED` and quarantined. |

## Sprint 2

- A namespace sandbox (bubblewrap or nsjail) if `IS_SANDBOX` is ever to become true. Rlimits do not hide the filesystem or the network.
- Run the osiris-cli suite against dnalang-core `90b68053` and update `to_qiskit` / `to_qasm3` callers before any pin bump.
- Exercise the adb and ssh commands on Node Alpha. These tests mock the transport.
