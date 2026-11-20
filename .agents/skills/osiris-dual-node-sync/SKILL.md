---
name: osiris-dual-node-sync
description: Verify and synchronize code, telemetry ledgers, and quarantine bundles across Node Alpha (Termux ARM64 edge) and Node Beta (WSL2 x86_64 workstation). Use when checking SSH or ADB transport, aligning git commit SHAs across substrates, staging incoming bundles, or delegating heavy benchmarks and tests to Node Beta.
---

# OSIRIS Dual-Node Sync

Coordinate deterministic Git alignment, quarantine staging, and workload routing between Node Alpha (`aarch64` mobile edge) and Node Beta (`x86_64` workstation) without leaking credentials or breaking substrate invariants.

## When to Use

- Verifying non-interactive SSH or ADB port-forwarded connectivity between Node Alpha and Node Beta
- Synchronizing `osiris-cli` or `dnalang-core` working trees and checking commit SHA parity
- Transferring QASM circuit bundles, benchmark artifacts, or JSONL ledgers across nodes
- Offloading heavy `7B+` mentor rounds, `olmo3-embed-7b` indexing, or full regression suites from Termux to WSL2

## Steps

1. **Identify Active Substrate and Workload Ceiling:**
   - Detect architecture via `platform.machine()` (`aarch64` + `$PREFIX` indicates Node Alpha with `300.0 s` ceiling and compact `< 100` col UI; `x86_64` indicates Node Beta with `1800.0 s` ceiling).
   - Keep lightweight orchestration and `gemma3:1b` edge checks on Node Alpha; delegate `qwen2.5-coder:7b` synthesis, SciFact dense retrieval, and large `pytest` runs to Node Beta.

2. **Verify Fail-Fast Non-Interactive SSH Transport:**
   - Probe connectivity with `BatchMode=yes` so automated workflows fail cleanly in seconds rather than hanging on interactive password prompts:

   ```bash
   ssh -o BatchMode=yes -o ConnectTimeout=5 "$BETA_HOST" \
     'uname -m && hostname && git --version && python3 --version'
   ```

   - When bridging over USB or wireless debugging, bind loopback forwarding explicitly (`adb forward tcp:8022 tcp:8022`) before running SSH or `rsync`.

3. **Stage Incoming Bundles in Quarantine First:**
   - Place incoming unverified archives or cross-node artifacts into `$OSIRIS_HOME/.quarantine/<sha256_prefix>/` (never `/sdcard/Download/` or `/tmp`).
   - Verify SHA-256 file digests against `manifest.json` and run `SYS-025` AST inspection (`ast.parse` + placeholder gate) before promoting any `.py` or `.dna` file into an active checkout.
   - Exclude `.env`, `apikey.json`, `.netrc`, SSH keys, and `.git/objects/pack/tmp*` from every `rsync` command:

   ```bash
   rsync -avz --checksum --delete-delay \
     --exclude='.git/objects/pack/tmp*' \
     --exclude='.env*' \
     --exclude='apikey.json' \
     --exclude='*.pem' \
     --exclude='*.key' \
     "$SRC_PATH/" "$DEST_PATH/"
   ```

4. **Confirm Canonical Path and Commit SHA Parity:**
   - Verify both substrates resolve to identical commit hashes (`git rev-parse HEAD`) with UTF-8 LF line endings (`git config core.autocrlf input`).
   - Remember that file synchronization never implies `SYS-023` or L7 promotion authority; run `osiris check` after sync completes.

## Gotchas

- **CRLF hash drift on WSL2:** Editing files from Windows-mounted paths with CRLF line endings alters SHA-256 artifact hashes when synced to Termux. Keep repositories inside native Linux/WSL2 and Termux filesystems with LF endings.
- **Hardcoded shell assumptions:** Never hardcode `apt` vs `pkg` or `/bin/bash` paths; always use `#!/usr/bin/env bash` or `python3` for portable scripts.
