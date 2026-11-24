---
name: osiris-dual-node-sync
description: Pair or connect Android Wireless Debugging, verify SSH/ADB or Bluetooth PAN transport, and synchronize code, telemetry ledgers, and quarantine bundles across Node Alpha (Termux ARM64 edge) and Node Beta (WSL2 x86_64 workstation). Use when pairing or connecting wireless ADB, checking SSH or ADB port forwards, aligning git commit SHAs across substrates, staging incoming bundles, or delegating heavy benchmarks and tests to Node Beta.
---

# OSIRIS Dual-Node Sync

Coordinate deterministic Git alignment, quarantine staging, and workload routing between Node Alpha (`aarch64` mobile edge) and Node Beta (`x86_64` workstation) without leaking credentials or breaking substrate invariants.

## When to Use

- Pairing or connecting Android Wireless Debugging (`adb mdns services`, `adb connect`, `adb pair`) with Node Alpha
- Verifying non-interactive SSH or ADB port-forwarded connectivity between Node Alpha and Node Beta
- Synchronizing `osiris-cli` or `dnalang-core` working trees and checking commit SHA parity
- Transferring QASM circuit bundles, benchmark artifacts, or JSONL ledgers across nodes
- Offloading heavy `7B+` mentor rounds, `olmo3-embed-7b` indexing, or full regression suites from Termux to WSL2

## Steps

1. **Identify Active Substrate and Workload Ceiling:**
   - Detect architecture via `platform.machine()` (`aarch64` + `$PREFIX` indicates Node Alpha with `300.0 s` ceiling and compact `< 100` col UI; `x86_64` indicates Node Beta with `1800.0 s` ceiling).
   - Keep lightweight orchestration and `gemma3:1b` edge checks on Node Alpha; delegate `qwen2.5-coder:7b` synthesis, SciFact dense retrieval, and large `pytest` runs to Node Beta.

2. **Establish Wireless ADB or Bluetooth PAN Transport:**
   - **Check mDNS before pairing:** Run `adb mdns services` first to distinguish `_adb-tls-connect._tcp` (main Wireless Debugging connection port) from `_adb-tls-pairing._tcp` (6-digit pairing dialog port).
   - If the workstation is already paired with Node Alpha, connect directly to the `_adb-tls-connect._tcp` endpoint (`adb connect <IP>:<Connect_Port>`) without requesting a new pairing code. Use `adb pair <IP>:<Pairing_Port> <Code>` only when unpaired or targeting `_adb-tls-pairing._tcp`.
   - Start `sshd -p 8022` in Termux on Node Alpha (`pkg install -y openssh rsync` if absent), then bind loopback forwarding (`adb forward tcp:8022 tcp:8022`) on the same host/WSL2 ADB daemon that will run `ssh`/`rsync`.
   - **Bluetooth PAN alternative:** Pair Node Alpha and Node Beta via Bluetooth, enable Bluetooth Tethering (`bnep0`, e.g. `192.168.44.X`), and target `<Phone_PAN_IP>` on port `8022`.
   - Probe SSH connectivity with `BatchMode=yes` so automated checks fail fast without hanging on password prompts:

   ```bash
   ssh -p 8022 -o BatchMode=yes -o ConnectTimeout=5 localhost \
     'uname -m && hostname && git --version && python3 --version'
   ```

3. **Stage Incoming Bundles in a Fresh Quarantine/Staging Directory First:**
   - Prefer `git fetch` / `git pull` for existing repository checkouts.
   - For cross-node file or ledger transfers, always create a **new, empty** staging/quarantine directory such as `$OSIRIS_HOME/.quarantine/<run_or_sha256_prefix>/` or `~/src/staging/<bundle>/` (never sync directly over an active checkout, `/sdcard/`, or `/tmp`).
   - **Preview first with `--dry-run --itemize-changes`**, exclude all secrets and temporary packfiles, and **never pass `--delete` or `--delete-delay`** (which can destroy destination-only files):

   ```bash
   rsync -avz --checksum --dry-run --itemize-changes \
     --exclude='.git/objects/pack/tmp*' \
     --exclude='.env*' \
     --exclude='apikey.json' \
     --exclude='*.pem' \
     --exclude='*.key' \
     -e 'ssh -p 8022' \
     "$SRC_PATH/" "$STAGE_DEST_PATH/"
   ```

   - Inspect the itemized preview, then re-run without `--dry-run` to stage the files.
   - Verify SHA-256 file digests against `manifest.json` and run `SYS-025` AST inspection (`ast.parse` + placeholder gate) before promoting any `.py` or `.dna` file into an active checkout.

4. **Confirm Canonical Path and Commit SHA Parity:**
   - Verify both substrates resolve to identical commit hashes (`git rev-parse HEAD`) with UTF-8 LF line endings (`git config core.autocrlf input`).
   - Remember that file synchronization never implies `SYS-023` or L7 promotion authority; run `osiris check` after sync completes.

## Gotchas

- **Android `/sdcard/` scoped storage:** Never sync Git repositories into `/sdcard/` (`noexec` mount + FUSE latency); keep active checkouts in Termux private storage (`~/src/`, `~/osiris-repos/`, or PRoot `/root/repos`).
- **Windows host vs. WSL2 ADB split:** Connecting wireless ADB via Windows `adb.exe` does not automatically register `tcp:8022` port forwards inside WSL2's separate Linux `adb` server. Verify `adb devices` and `adb forward --list` inside the exact environment running `ssh`.
- **CRLF hash drift on WSL2:** Editing files from Windows-mounted paths with CRLF line endings alters SHA-256 artifact hashes when synced to Termux. Keep repositories inside native Linux/WSL2 and Termux filesystems with LF endings.
- **Hardcoded shell assumptions:** Never hardcode `apt` vs `pkg` or `/bin/bash` paths; always use `#!/usr/bin/env bash` or `python3` for portable scripts.
