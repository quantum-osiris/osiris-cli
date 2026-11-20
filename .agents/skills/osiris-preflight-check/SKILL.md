---
name: osiris-preflight-check
description: Run deterministic read-only OSIRIS v6.0.0 preflight audits across SHA-256 JSONL ledgers, sibling repositories, Ollama model tags, and sandbox suites. Use when starting a session, diagnosing runtime or benchmark errors, checking orphan runs, or verifying workspace health before or after code changes.
---

# OSIRIS Preflight Check

Execute deterministic, read-only health checks across active state directories, hash-chained JSONL ledgers, local Ollama tags, and sibling repositories before mutating code or launching benchmarks.

## When to Use

- Starting a development or debugging session in `osiris-cli`
- Diagnosing failed mentor rounds, unclosed run records, or missing `dnalang-core` paths
- Verifying installed Ollama model tags to prevent HTTP 404 errors and ledger contamination
- Confirming baseline health before and after applying a repository patch

## Steps

1. **Probe Substrate and Dynamic Paths:**
   - Resolve the active state directory from `$OSIRIS_HOME` falling back to `~/.osiris` via `Path.resolve()`.
   - Locate sibling repositories (`dnalang-core`, `organism_sim`) dynamically via `genome_ledger.ledger_src_candidates()` and `osiris_cli.probes.resolve_sibling_repo()`.

   ```bash
   python3 -c "
   import os, pathlib, platform, genome_ledger
   repo = pathlib.Path.cwd().resolve()
   osiris_home = pathlib.Path(os.environ.get('OSIRIS_HOME') or (pathlib.Path.home() / '.osiris')).resolve()
   print(f'Arch: {platform.machine()} | Repo: {repo} | OSIRIS_HOME: {osiris_home} (exists={osiris_home.is_dir()})')
   print('LEDGER_SRC:', genome_ledger.LEDGER_SRC, 'exists:', os.path.exists(genome_ledger.LEDGER_SRC))
   "
   ```

2. **Verify Canonical SHA-256 Ledgers and Orphan Runs:**
   - Run OSIRIS's built-in read-only `osiris check ledger git system evidence` and `python3 protege.py verify` to confirm hash-chain continuity (`verify_chain_tail`), `0` failing verifications, and `0` orphan runs.
   - Never create competing SQLite databases or sidecar logs outside `$OSIRIS_HOME`.

3. **Reconcile Installed Ollama Tags Against Configured Roles:**
   - Query the local Ollama `/api/tags` endpoint via `osiris_cli.providers.router.normalize_ollama_tag` before dispatching `/bench`, `[N]`, `[G]`, `/trial`, or `/suggest`.
   - Flag canonical tag mismatches (for example `gemma-3-1b` vs `gemma3:1b`) and block benchmarks targeting uninstalled 7B models on mobile edge nodes before a run entry is opened.

4. **Audit Git Cleanliness and Secret Isolation:**
   - Run `git status --short` and `git diff --check` and confirm zero credential files (`.env`, `apikey.json`, `*.pem`, `*.key`, `*token*`) are staged or modified.

## Gotchas

- **Unreadable mobile power sensors:** On unrooted Android Termux (`aarch64`), `/sys/class/power_supply` reads normally fail under SELinux; treat this as expected for attended interactive work while keeping unattended overnight batch jobs gated.
- **Split checkouts:** Verify `python3 -c "import osiris_cli; print(osiris_cli.__file__)"` resolves to the exact git working tree under test so `pytest` does not run against a stale installation.
