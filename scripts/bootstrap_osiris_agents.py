#!/usr/bin/env python3
"""Bootstrap AGENTS.md, unified .agents/skills, and portable symlinks for OSIRIS v6.0.0.

Enforces:
- Pure UTF-8 with LF (0x0A) line endings across Termux (aarch64) and WSL2 (x86_64)
- Universal 2-key YAML frontmatter (name, description) across Claude Code, agy, Vibe, Copilot CLI, and OSIRIS
- Portable relative repo symlinks: CLAUDE.md -> AGENTS.md and .claude/skills -> ../.agents/skills
- Non-destructive backup of any pre-existing regular files/directories before symlinking
- Repository-local by default; global ~/.agents/skills wiring requires explicit --global-links
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

FENCE = "`" * 3


def _md(template: str) -> str:
    """Normalize line endings to LF and expand tilde fences to backtick fences."""
    normalized = template.replace("\r\n", "\n").replace("\r", "\n").lstrip("\n")
    if not normalized.endswith("\n"):
        normalized += "\n"
    return normalized.replace("~~~", FENCE)


AGENTS_MD = _md(
    r"""
# AGENTS.md - OSIRIS Sovereign Runtime Governance (v6.0.0)

This file is the repository-local agent governance reference for coding CLIs and runtimes operating in `osiris-cli` (`Claude Code`, `Google Antigravity / agy`, `Mistral Vibe`, `GitHub Copilot CLI`, and `osiris-cli`). `CLAUDE.md` is a relative symlink to `AGENTS.md` - edit `AGENTS.md` directly so both remain byte-identical. Where `GEMINI.md` Gate-0 rules are stricter than any rule here, `GEMINI.md` wins.

---

## 1. Core Architectural Invariants

1. **Primary Law:** *"Models propose; deterministic code decides, measures, and records."*
2. **Empirical Measurement Law:** *"No number gets a name until a histogram produced it, and no AI-written summary substitutes for opening the counts."* Never introduce speculative abstractions, unverified numerical claims, or pseudo-physics constants.
3. **Single Canonical Ledger Topology:**
   - Never create competing or split-brain state stores (such as standalone SQLite `ledger.db` files or ad-hoc `/tmp` logs) and never invent fictional wrapper entrypoints.
   - Bind all runtime provenance to OSIRIS's existing append-only SHA-256 hash-chained JSONL ledgers under `$OSIRIS_HOME` (default `~/.osiris/`):
     - `living/exchanges.jsonl` (verified via `verify_chain` / O(1) `verify_chain_tail` with `osiris_cli/chain_lock.py` zero-fork locking)
     - `genome_ledger.jsonl` (via `genome_ledger.py` + `dnalang-core` Ledger)
     - `tasks/task_runs.jsonl` (via `osiris_cli/task_context.py`)
     - `bench/results.ledger.jsonl` (via `osiris_bench.py` / `protege.py`)
     - `retrieval/ledger.jsonl`
4. **Strict Separation of the Two Gates:**
   - **Speaking Gate (`SYS-026`, `osiris_cli/living.py`):** Governs whether the online-trained core (`osiris.nclm`) has earned its voice (`<= 2.0 bits/byte` across `30/30` held-out exchanges while beating the unigram baseline, where every 5th exchange `exchange_idx % 5 == 0` is strictly held out and never trained on). Never bypass, mock, or weaken `SYS-026` via UI flags or hotkeys.
   - **Action Execution Gate (`EXEC_GATE: STRICT | AUTO`):** Governs interactive human operator approval (`[y/N]`) before state-mutating workspace writes (`/apply`), shell subprocesses (`!cmd`), or external `/skill` dispatches.
5. **Human-Governed Workspace Mutation:** Agents may read confined repository files and propose diffs or sandboxed candidates, but workspace writes require explicit operator authorization (`/apply`, signed grants under `SYS-021`, or approved interactive session intent). Never auto-commit, force-reset, or `git push` without explicit instruction.

---

## 2. Pipeline Layer Contracts (L0 - L8)

Every modification to `osiris-cli` must preserve the deterministic boundaries of the L0-L8 pipeline:

| Layer | Canonical Modules | Enforced Contract |
| :--- | :--- | :--- |
| **L0 Intake** | `osiris_cli/living.py`, `osiris_cli/paste_learning.py`, `osiris_cli/osiris_repl.py`, `osiris_cli/task_context.py`, `session_state.py` | Record every exchange to `exchanges.jsonl` and every task run to `tasks/task_runs.jsonl`; preserve bracketed paste (`keys_typed`) and `TypeAhead`; never auto-promote pasted or external text into durable notes, facts, or training queues without explicit approval (`/learn confirm`). |
| **L1 Intent** | `intent_router.py`, `osiris_termux_console.py` (`_route`) | Route and assess risk tiers (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`) strictly via deterministic Python rules with zero LLM calls. Slash commands require `/` at index 0 of the trimmed input buffer. |
| **L2 Orchestrate** | `osiris_termux_console.py` (`_next_best`), `sprint_manager.py`, `osiris_cli/task_context.py` (`BudgetMeter`), `osiris_cli/skills_runtime.py` | Compute Next Best Action deterministically from live state; enforce per-task execution budgets (`STOPPED_BUDGET` terminal state); require human sign-off for `/apply` and `/discard`. |
| **L3 Evidence** | `osiris_cli/claims.py`, `osiris_cli/knowledge.py`, `research.py` | Enforce **SYS-024 Claims Firewall** *before* model generation. Never assert `REFUTED` or `NOT_MEASURED` historical constants (`theta_lock = 51.843 deg` REFUTED, `F_max = 0.97871` REFUTED, `Lambda_Phi = 2.176435e-08` NOT_MEASURED) as valid physics. Cite exact source files for retrieved notes. |
| **L4 Propose** | `osiris_cli/providers/`, `osiris_cli/code_agent.py`, `osiris_termux_console.py` (`_synthesize`, `_bench_blocker`) | Verify Ollama reachability and installed canonical model tags (e.g., `gemma3:1b`, `qwen2.5-coder:7b`) during preflight across *all* entrypoints (`[N]`, `[G]`, `/bench`, `/trial`, `/suggest`) to prevent HTTP 404 ledger contamination. |
| **L5 Verify** | `osiris_cli/production_synth.py`, `payload_gate.py`, `osiris_cli/physics_checks.py`, `osiris_cli/probes.py` | Enforce **SYS-025** fail-closed `ast.parse`, AST contract extraction, isolated independent L5 critic, placeholder/stub rejection (`pass`, `...`, `NotImplementedError`, bare `return None`), `SafePathGuard` confinement, and sandboxed `pytest` verification. |
| **L6 Benchmark** | `protege.py`, `osiris_bench.py`, `osiris_cli/prepost.py`, `osiris_cli/retrieval/` | Preserve reference-pass and `known_bad` / `unchanged`-fail invariants across the 9-task bench suite (`09037df63614`). Keep `osiris_bench.runner_sha256()` comparability honest. SciFact `olmo3-embed-7b` ranks evidence capsules only and never issues truth verdicts. |
| **L7 Promote** | `osiris_cli/living.py` (`Osiris.gate`, `audit_mentor_reply`), `osiris_cli/train.py`, `gaps.py` | Filter training strictly on `learnable == True`, `distill_eligible != False`, and `heldout == False`. Force `distill_eligible = False` on degraded low-RAM fallbacks, truncated outputs, or replies failing post-generation file-citation / numerical-metric audits. |
| **L8 Ledger** | `genome_ledger.py`, `run_record.py`, `dnalang-core` (`dnalang/ledger.py`) | Maintain write-once hash chains and ensure every started run writes a deterministic terminal entry (`0 failing verification`, `0 orphans`). |

---

## 3. Dual-Substrate Execution Invariant (Node Alpha vs. Node Beta)

`osiris-cli` executes across two distinct hardware substrates. Never hardcode paths, terminal widths, package managers (`pkg` vs. `apt`), or timeout ceilings.

### Substrate Profiles

- **Node Alpha (`NODE_ALPHA_EDGE` - Android / Termux / `aarch64`):**
  - **Hardware & Runtime:** ARM64 (`aarch64`), NEON SIMD, Android Termux (`$PREFIX` set), `300.0 s` execution horizon.
  - **UI & Ergonomics:** Must render cleanly under `< 100` columns (`MOBILE COMPACT / TERMUX`, including 76-80 col viewports). Prefer inline `prompt_toolkit` + `rich` viewports (with stdlib fallback) over full-screen alternate-buffer TUIs so native terminal scrollback, text copy, and Android soft keyboards work without lock-in.
  - **Resource & Sensor Guards:** Tolerate unreadable `/sys/class/power_supply` sysfs permissions gracefully (allow attended interactive work; block unattended overnight batch jobs unless explicitly overridden). Keep heavy models (`7B+` local coders, `olmo3-embed-7b`) gated behind RAM checks and delegate heavy synthesis/benchmarks to Node Beta.
- **Node Beta (`NODE_BETA_WORKSTATION` - Linux / WSL2 / `x86_64`):**
  - **Hardware & Runtime:** `x86_64`, AVX2/AVX-VNNI + discrete GPU acceleration, `1800.0 s` execution horizon.
  - **Workload Role:** Primary substrate for heavy multi-file refactoring, `qwen2.5:7b` / `qwen2.5-coder:7b` mentor benchmarks, `olmo3-embed-7b` SciFact indexing, quantum circuit compilation (`Qiskit`, `TKET`, `PennyLane`), and full `pytest` regression sweeps.

### Dynamic Path & Sibling Repository Resolution

1. **State Root (`$OSIRIS_HOME`):** Always resolve state directories via `os.environ.get("OSIRIS_HOME")` falling back to `Path.home() / ".osiris"`, canonicalized with `.resolve()`.
2. **Sibling Packages (`dnalang-core`, `organism_sim`):** Use `genome_ledger.ledger_src_candidates()` and `osiris_cli/probes.py` (`resolve_sibling_repo`) to locate sibling checkouts dynamically across `$OSIRIS_DNALANG_LEDGER`, `$OSIRIS_HOME/src/`, `repo_root.parent`, `~/osiris-repos/`, `~/`, and installed packages without hardcoding machine paths.
3. **Forbidden Hardcoded Paths:** Never hardcode `/home/<user>`, `/root/`, `/data/data/com.termux/...`, `~/fold/`, `/tmp`, or `/sdcard/Download/` inside library or test modules. Use `tmp_path` fixtures in `pytest`.
4. **Cross-Node Sync & Quarantine:** Incoming cross-substrate bundles must stage in `$OSIRIS_HOME/.quarantine/<sha256_prefix>/`, pass SHA-256 `manifest.json` verification and `SYS-025` AST inspection before checkout promotion, and maintain identical Git commit SHAs (`git rev-parse HEAD`) across checkouts.

---

## 4. Security, Secret Isolation & Safe I/O (`SafePathGuard`)

1. **Credential Non-Read Rule:** Deterministic probes (`osiris_cli/probes.py`), skills runtime (`osiris_cli/skills_runtime.py`), agent tool loops (`osiris_cli/code_agent.py`), and external coding CLIs must **never** read, print, index, stage, symlink, or log files matching:
   - `.env`, `.env.*`, `apikey.json`, `*credentials*`, `*secret*`, `*token*`, `id_rsa*`, `id_ed25519*`, `*.pem`, `*.key`, `.netrc`
2. **Workspace Confinement:** Resolve every candidate file path via `Path.resolve()` and verify `candidate.is_relative_to(workspace_root)` before reading or writing, blocking `..` traversal and external symlink escapes.
3. **Encoding & Line Endings:** Write all Python, Markdown, shell, JSON, JSONL, and `.dna` files as UTF-8 with LF (`\n`) line endings so hashes match byte-for-byte between Termux and WSL2.

---

## 5. Cross-CLI Unified Skills Standard (`./.agents/skills` & `~/.agents/skills`)

To ensure every reusable skill works identically across **Claude Code**, **Antigravity (`agy`)**, **Mistral Vibe**, **Copilot CLI**, and **OSIRIS**:

1. **Canonical Directory Layout & Relative Symlinks:**
   - **Repo Instructions:** `./AGENTS.md` with relative symlink `./CLAUDE.md -> AGENTS.md`
   - **Repo Skills:** `./.agents/skills/<skill-name>/SKILL.md` with relative symlink `./.claude/skills -> ../.agents/skills`
   - **Global Skills (opt-in only):** `~/.agents/skills/<skill-name>/SKILL.md`
   - Precedence: Project skills (`./.agents/skills`) override global skills (`~/.agents/skills`) of the same name, failing closed if a project skill is invalid.
2. **Universal `SKILL.md` YAML Frontmatter:**
   - Include **only** `name` (lowercase `a-z0-9-`, max 64 chars) and `description` (concise capability summary + explicit `"Use when..."` trigger conditions).
   - Do **not** put tool-proprietary keys (`allowed-tools`, `model`, custom permissions) in YAML frontmatter, as strict parsers in sibling CLIs will reject the skill. Place execution constraints inside the Markdown body.
   - Reference bundled scripts and templates relative to the skill directory (`./scripts/...`, `./references/...`).

---

## 6. Mandatory Verification Protocol Before Completing Any Task

Before declaring any code change, refactor, or bugfix complete, run these checks and report exact exit codes and counts:

~~~bash
# 1. Unified skill & symlink verification
python3 scripts/bootstrap_osiris_agents.py --verify-only

# 2. Syntax & AST sanity across modified modules
python3 -m py_compile osiris_cli/*.py osiris_cli/providers/*.py

# 3. Automated offline unit & regression suite
pytest -o cache_dir=/tmp/osiris_pytest_cache -q tests/

# 4. Working tree & diff whitespace hygiene audit (ensure zero staged secrets)
git diff --check
git status --short
~~~

### Definition of Done

- All new/modified tests pass offline and deterministically on both `aarch64` and `x86_64`.
- `verify_chain` and O(1) `verify_chain_tail` pass on existing and newly written JSONL ledger entries.
- `0` stub/placeholder functions (`payload_gate.py`), `0` orphan runs, and `0` phantom citations or unbacked metrics marked `distill_eligible`.
"""
)


SKILL_PREFLIGHT = _md(
    r"""
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

   ~~~bash
   python3 -c "
   import os, pathlib, platform, genome_ledger
   repo = pathlib.Path.cwd().resolve()
   osiris_home = pathlib.Path(os.environ.get('OSIRIS_HOME') or (pathlib.Path.home() / '.osiris')).resolve()
   print(f'Arch: {platform.machine()} | Repo: {repo} | OSIRIS_HOME: {osiris_home} (exists={osiris_home.is_dir()})')
   print('LEDGER_SRC:', genome_ledger.LEDGER_SRC, 'exists:', os.path.exists(genome_ledger.LEDGER_SRC))
   "
   ~~~

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
"""
)


SKILL_CLAIMS_AUDIT = _md(
    r"""
---
name: osiris-claims-grounding-audit
description: Audit responses, research briefs, and training candidates against the SYS-024 claims register, file-citation provenance, histogram-backed metrics, and SYS-026 Speaking Gate rules. Use when evaluating physics or quantum claims, reviewing mentor outputs, checking citations, or filtering exchanges for NCLM continual learning.
---

# OSIRIS Claims and Grounding Audit

Enforce OSIRIS epistemic governance across prompts, code comments, experiment briefs, and mentor replies so unverified claims or hallucinated metrics never enter durable memory or `osiris.nclm` training.

## When to Use

- Reviewing quantum, physics, or benchmark claims before writing docs, briefs, or code
- Auditing mentor model outputs for phantom file citations or fabricated precision numbers
- Updating or testing `osiris_cli/claims.py`, `osiris_cli/knowledge.py`, `osiris_cli/living.py`, or `osiris_cli/train.py`
- Deciding whether a recorded exchange qualifies for online continual learning or batch distillation

## Steps

1. **Enforce the Pre-Model SYS-024 Claims Register:**
   - Cross-check named constants and experimental assertions against `osiris_cli/claims.py` (`/legit list`) before generating conclusions.
   - Never treat refuted or unmeasured historical CRSM parameters as active invariants (`theta_lock = 51.843 deg` is `REFUTED`, `F_max = 0.97871` is `REFUTED`, `Lambda_Phi = 2.176435e-08` is `NOT_MEASURED`).

2. **Verify Explicit File Attribution:**
   - Every local repository or note citation in a synthesized answer must match a path actually present in that turn's retrieved context or deterministic `ProbeReceipt` list.
   - Flag any cited path absent from the retrieved bundle as a phantom citation.

3. **Enforce the Histogram Procedural Rule:**
   - Apply the governing rule: *"No number gets a name until a histogram produced it, and no AI-written summary substitutes for opening the counts."*
   - Require every precision metric (`bits/byte`, `bpb`, `nDCG@10`, `fidelity`, `p-value`, `z-score`, shot counts, or HQC estimates) to trace verbatim to a raw ledger row, benchmark artifact, or deterministic computation receipt with explicit units and uncertainty bounds.

4. **Protect SYS-026 Speaking Gate and Held-Out Isolation:**
   - Keep every 5th exchange (`exchange_idx % 5 == 0`) strictly isolated as held-out evaluation data (`heldout = True`, never trained on).
   - Mark `distill_eligible = False` whenever a reply comes from a degraded low-RAM fallback model, truncates mid-stream, or fails citation or numerical-metric verification.
   - Keep `osiris.nclm` behind the `SYS-026` Speaking Gate until held-out evaluation reaches `<= 2.0 bits/byte` across `30/30` held-out exchanges while beating the unigram baseline.

## Gotchas

- **Confusing retrieval ranking with truth:** `olmo3-embed-7b` and BM25F rank candidate passages by relevance (`nDCG@10`), not truth. Only deterministic code (`claims.py`, `probes.py`, verified ledger counts) issues a verdict.
- **Triggering SYS-024 accidentally in slash commands:** In the REPL, slash commands must begin at index 0 of a clean line; concatenating a slash command onto natural-language text containing registered claim numbers routes the whole buffer through chat and triggers pre-model claim injection.
"""
)


SKILL_DUAL_NODE = _md(
    r"""
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

   ~~~bash
   ssh -o BatchMode=yes -o ConnectTimeout=5 "$BETA_HOST" \
     'uname -m && hostname && git --version && python3 --version'
   ~~~

   - When bridging over USB or wireless debugging, bind loopback forwarding explicitly (`adb forward tcp:8022 tcp:8022`) before running SSH or `rsync`.

3. **Stage Incoming Bundles in Quarantine First:**
   - Place incoming unverified archives or cross-node artifacts into `$OSIRIS_HOME/.quarantine/<sha256_prefix>/` (never `/sdcard/Download/` or `/tmp`).
   - Verify SHA-256 file digests against `manifest.json` and run `SYS-025` AST inspection (`ast.parse` + placeholder gate) before promoting any `.py` or `.dna` file into an active checkout.
   - Exclude `.env`, `apikey.json`, `.netrc`, SSH keys, and `.git/objects/pack/tmp*` from every `rsync` command:

   ~~~bash
   rsync -avz --checksum --delete-delay \
     --exclude='.git/objects/pack/tmp*' \
     --exclude='.env*' \
     --exclude='apikey.json' \
     --exclude='*.pem' \
     --exclude='*.key' \
     "$SRC_PATH/" "$DEST_PATH/"
   ~~~

4. **Confirm Canonical Path and Commit SHA Parity:**
   - Verify both substrates resolve to identical commit hashes (`git rev-parse HEAD`) with UTF-8 LF line endings (`git config core.autocrlf input`).
   - Remember that file synchronization never implies `SYS-023` or L7 promotion authority; run `osiris check` after sync completes.

## Gotchas

- **CRLF hash drift on WSL2:** Editing files from Windows-mounted paths with CRLF line endings alters SHA-256 artifact hashes when synced to Termux. Keep repositories inside native Linux/WSL2 and Termux filesystems with LF endings.
- **Hardcoded shell assumptions:** Never hardcode `apt` vs `pkg` or `/bin/bash` paths; always use `#!/usr/bin/env bash` or `python3` for portable scripts.
"""
)


SKILL_SANDBOX_PATCH = _md(
    r"""
---
name: osiris-sandbox-patch
description: Author, validate, and test minimal zero-stub patches for osiris-cli under SYS-025 AST firewall, SafePathGuard, and dual-substrate constraints. Use when fixing bugs, implementing L0-L8 modules, refactoring REPL or TUI components, or preparing verified diffs and pytest suites for operator approval.
---

# OSIRIS Sandbox Patch

Design and verify surgical, production-ready Python changes inside `osiris-cli` that pass `SYS-025` static AST checks, `payload_gate.py` anti-stub validation, and offline `pytest` verification on both `aarch64` and `x86_64`.

## When to Use

- Fixing bugs or implementing upgrades across `osiris_cli/` modules
- Building or refining inline `prompt_toolkit` + `rich` TUI components, deterministic probes, or mentor routers
- Writing unit and integration tests under `tests/`
- Preparing review-only diffs or sandboxed `/apply` candidates under `EXEC_GATE: STRICT`

## Steps

1. **Inspect Live Call Sites Read-Only:**
   - Locate exact function signatures, imports, and callers with `grep -rn` or `SafePathGuard`-confined reads before drafting edits.
   - Respect `SafePathGuard`: resolve paths via `Path.resolve()`, confine operations to the repository root, and never read `.env`, `apikey.json`, or key material.

2. **Author Zero-Stub, Dual-Substrate-Safe Code:**
   - Satisfy `payload_gate.py` and `stub_detector.py`: never leave functions whose bodies consist solely of `pass`, `...`, `raise NotImplementedError`, a bare `return` or `return None`, or a standalone docstring.
   - Avoid hardcoded user paths, fixed terminal widths wider than 80-100 columns on Termux, or non-stdlib dependencies not declared in `pyproject.toml`.
   - Provide clean non-TTY fallbacks whenever touching interactive `prompt_toolkit` or `rich.Live` terminal code so CI pipes and headless test runners never hang.

3. **Run Deterministic Syntax, Stub, and Test Gates:**
   - Validate compilation, AST stub absence, and offline `pytest` execution before presenting the change as ready:

   ~~~bash
   python3 -m py_compile osiris_cli/*.py osiris_cli/providers/*.py
   pytest -o cache_dir=/tmp/osiris_pytest_cache -q tests/
   ~~~

   - Use `tmp_path` and monkeypatched `$OSIRIS_HOME` in every new test so unit tests never mutate the operator's live `~/.osiris/` ledgers.

4. **Present Diffs Clearly for Human Sign-Off:**
   - When the operator requests review-only mode or `EXEC_GATE` is `STRICT`, output a minimal unified diff, summarize ledger/schema impact (ensuring backward compatibility with `verify_chain` and `verify_chain_tail`), and wait for explicit authorization before mutating files or committing.

## Gotchas

- **Breaking `verify_chain_tail` backward compatibility:** When adding metadata fields (such as `routing`, `probe_receipts`, or `grounding_audit`) to `exchanges.jsonl`, ensure existing ledger entries verify against their original recorded payload format while new entries include the added fields deterministically.
- **Blocking the async TUI loop:** Wrap synchronous file hashing, BM25F scoring, shadow `osiris.nclm` bits/byte scoring, or HTTP calls in background threads when called from an interactive loop so keystrokes and `TypeAhead` stay responsive.
"""
)


SKILL_DEEP_RESEARCH = _md(
    r"""
---
name: deep-research
description: Coordinate multi-source, evidence-governed deep research across parallel subagents and synthesize findings into a verified narrative report with SHA-256 provenance. Use when investigating a topic across multiple sources, comparing technical or market alternatives, auditing literature or benchmarks, analyzing trends, or testing hypotheses under OSIRIS governance when an Agent or Task subagent tool is available.
---

# OSIRIS Deep Research Coordinator

Coordinate parallel research subagents and a synthesis subagent under OSIRIS L0-L8 governance while keeping the coordinator context window strictly isolated from raw source payloads.

## When to Use

- Researching a technical, scientific, regulatory, or market question across multiple primary sources
- Comparing competing architectures, algorithms, hardware targets, or vendor options with quantitative baselines
- Reviewing academic literature, benchmark suites, or historical trends into a structured narrative report
- Testing hypotheses where disconfirming evidence and provenance tracking are required (and an `Agent` or `Task` subagent tool is available; otherwise research directly)

## Coordinator Role and Invariants

Act strictly as the **L2 Orchestrator**: clarify material ambiguity, propagate L3 `SYS-024` evidence constraints, spawn subagents within substrate limits, and deliver the verified report.

- **Context hygiene:** Never conduct raw web scraping or write research notes/reports directly in the coordinator turn. Do not read `./references/researcher.md`, `./references/report-writer.md`, or raw note files into coordinator context; rely on subagent return summaries.
- **Governed memory isolation (L0):** Keep research notes and reports staged in the workspace directory. Never auto-promote external findings into durable `/remember` facts or `osiris.nclm` training queues without explicit user confirmation.
- **Progress visibility:** Maintain user-facing milestones phrased as research outcomes without referencing internal subagent mechanics.

## Steps

### 1. Check ambiguity and local L3 evidence constraints

Ask clarifying questions **only** when missing details (ambiguous acronyms, unspecified jurisdiction/hardware/version bounds, or colliding entity names) would materially change how research is decomposed. Otherwise state sensible defaults and proceed.

Check active conversation context or lightweight read-only OSIRIS state for established `SYS-024` verdicts (`REFUTED` or `SUPPORTED` claims, kept facts, or `/lab` hypotheses) so every subagent receives them upfront and never resurrects refuted premises.

### 2. Resolve substrate horizon, research title, and workspace directories

Resolve `{workspace_root}` dynamically via `realpath` using `$OSIRIS_RESEARCH_DIR` if set, an earlier research root from this session, or the current working directory (never `/tmp`, upload scratch folders, `/sdcard/Download`, or hardcoded paths, and never copy `.env` or `apikey.json`).

Cap parallel subagent fan-out per wave by active substrate:

- **Node Alpha (`aarch64` / Termux mobile edge, 300 s horizon):** Spawn **1 to 3** parallel subagents per wave.
- **Node Beta (`x86_64` / workstation or cloud runner, 1800 s horizon):** Spawn **3 to 6+** parallel subagents when breadth warrants it.

Choose `{research title}` in the user's language and sentence case: **3 to 6 words AND under 50 characters**, using only letters, numbers, and single spaces, with zero hyphens, underscores, apostrophes, or punctuation.

Never write into, rename, or move an earlier request's folder. Create quoted directories:

~~~bash
mkdir -p "{workspace_root}/research_notes/{research title}" "{workspace_root}/reports"
~~~

### 3. Decompose query with falsification coverage

Decompose the prompt into mutually exclusive, collectively exhaustive (MECE) subtopics (or only net-new facets when extending prior research):

| Researchers (Beta / Alpha) | Decomposition Strategy | Example |
| :--- | :--- | :--- |
| 1 / 1 | Single bounded factual verification | "Current NIST FIPS standard for ML-DSA signatures" |
| 3 / 2-3 | Mechanism, empirical benchmarks, and scaling limits | "qLDPC decoding latency" -> (1) algorithms, (2) FPGA/ASIC latency data, (3) error-floor limits |
| 4 / 3 | Multi-criteria evaluation plus failure modes | "Small-business CRM comparison" -> (1) pricing/TCO, (2) workflow features, (3) API limits, (4) migration failures |
| 5+ / 3 (batched) | Multi-region, multi-entity, or clustered landscape | "Sovereign AI regulation" -> US, EU, East Asia, UK, open-weight compliance |

Include at least one **falsification / disconfirming-evidence question** per subagent so benchmarks, forecasts, and claims are tested against null results, baselines, and failure modes. Default to **3** subagents when uncertain.

### 4. Spawn parallel foreground research subagents

Launch all research subagents **in a single parallel tool-call turn** pointing each researcher at `{path_to_skill}/references/researcher.md` and saving UTF-8 LF notes to `{workspace_root}/research_notes/{research title}/{topic_slug}.md`.

### 5. [If strictly necessary - max once] Close critical evidence gaps

Inspect subagent return summaries. Only if a missing primary metric or unresolved contradiction leaves the core question incomplete, spawn one targeted parallel gap-closing wave. Execute Step 5 **at most once**, and always output this exact line when doing so:

`After these researchers finish, I will move directly to coordinating the report writer`

### 6. Spawn foreground report-writer subagent

Spawn a single foreground report writer pointing at `{path_to_skill}/references/report-writer.md`, writing `{workspace_root}/reports/{research title}.md` and `{workspace_root}/research_notes/{research title}/manifest.json`.

### 7. Deliver report to user

Deliver `{workspace_root}/reports/{research title}.md`, verify `manifest.json` was written, then provide a tight **3 to 5 sentence summary** stating the core verdict, key confidence bounds, and the exact path or delivery location of the full report.

## Gotchas

- **Unquoted spaces in paths:** Always double-quote `"{workspace_root}/research_notes/{research title}"` and `"{workspace_root}/reports/{research title}.md"` in shell commands and prompts.
- **Mobile thermal/timeout limits:** Do not launch 5+ simultaneous heavy subagents on Termux (`aarch64`); cap waves at 1-3 on Node Alpha or offload wide sweeps to Node Beta (`x86_64`).
"""
)


REF_RESEARCHER = _md(
    r"""
# OSIRIS Deep Research - Researcher Subagent Protocol

Operate as an **L3/L4 Evidence Gatherer and Falsifier** under the OSIRIS core law: *"Models propose; deterministic code decides, measures, and records."* Your job is to collect verifiable, source-backed facts, test claims against counter-evidence, write dense structured notes to disk, and return a compact summary to the coordinator.

## 1. Source Hierarchy and Epistemic Tags

Classify every major factual or quantitative claim in your output notes with one of these five tags:

- **`[PRIMARY]`:** Peer-reviewed papers (with DOI or arXiv ID), official standards/RFCs, vendor hardware datasheets, regulatory/SEC filings, canonical git commits, or raw benchmark ledgers.
- **`[COMPUTED]`:** Deterministic values derived by running Python or shell arithmetic over `[PRIMARY]` raw counts (always record the exact formula and inputs used).
- **`[SECONDARY]`:** Reputable technical journalism, industry analyst summaries, or engineering blog posts. Never rely solely on `[SECONDARY]` sources when `[PRIMARY]` artifacts are reachable.
- **`[CONTESTED]`:** Credible sources report conflicting measurements or conclusions. Document both sides alongside sample sizes, hardware configurations, and dates.
- **`[UNVERIFIED_GAP]`:** A target metric or question could not be verified from accessible sources. State exactly what queries/repositories were checked and why the gap remains open - never guess, interpolate, or fabricate numbers.

## 2. Non-Negotiable Governance Rules

1. **Histogram Procedural Rule:** *"No number gets a name until a histogram produced it, and no AI-written summary substitutes for opening the counts."* For every quantitative metric, record exact value, explicit units, comparison baseline, sample size (`N`), hardware profile, and uncertainty bounds.
2. **SYS-024 Claims Compliance:** Never assert locally `REFUTED` claims passed in your prompt constraints as valid premises.
3. **Adversarial Falsification:** Actively search for negative results, replication failures, regressions, and edge cases where the primary mechanism or product breaks down.
4. **SafePathGuard Secret Isolation:** Confine local reads to authorized project directories and **never** read `.env`, `apikey.json`, SSH keys, `.pem`, `.key`, `.netrc`, or token files.

## 3. Required Output Note Structure (UTF-8, LF)

Write notes using UTF-8 encoding and LF (`\n`) line endings with sections for Core Verified Findings, Quantitative Evidence Table, Falsification/Null Results/Failure Modes, and Open Evidence Gaps.

## 4. Coordinator Return Summary

After confirming your note file is saved on disk, return a concise **150 to 250 word** task summary to the coordinator covering answered key questions, headline numbers with units/baselines, and critical `[CONTESTED]` or `[UNVERIFIED_GAP]` items.
"""
)


REF_REPORT_WRITER = _md(
    r"""
# OSIRIS Deep Research - Report Writer Synthesis Protocol

Operate as the **L5/L6/L8 Synthesis and Provenance Engine**. Read every research note file in the assigned `research_notes/{research title}/` directory (plus any prior report/notes explicitly supplied in your prompt), synthesize a rigorous narrative report, and write a cryptographic SHA-256 `manifest.json` receipt.

## 1. Grounding and Epistemic Integrity Rules

1. **Direct Structural Opening:** Open the report immediately with the core substantive answer and decision verdict in sentence 1 - no meta-introductions or filler.
2. **Zero Phantom Citations:** Cite only URLs, DOIs, and file paths that genuinely appear in the consumed research notes or verified prior research (`phantom_files == 0`).
3. **Zero Unbacked Precision Metrics:** Every precision number (`bits/byte`, `ms`, `%`, `USD`, `nDCG@10`, `fidelity`, `p-value`) in prose or tables must trace directly to the research notes with units, comparison baseline, and uncertainty/sample context intact (`unverified_numbers == 0`).
4. **Honest Epistemic Calibration:** Distinguish settled `[PRIMARY]` facts from `[SECONDARY]` reporting, `[CONTESTED]` debates, and `[UNVERIFIED_GAP]` items.

## 2. Required Report Layout (`reports/{research title}.md`)

Write the finalized report using UTF-8 encoding and LF (`\n`) line endings containing:
- Executive Verdict
- Methodology and Evidence Scope
- Detailed Findings (with inline citations and quantitative tables)
- Falsification, Risks, and Limitations
- Sources

## 3. Deterministic SHA-256 Provenance Manifest (`manifest.json`)

Immediately after saving `reports/{research title}.md`, execute a deterministic Python stdlib snippet to write `research_notes/{research title}/manifest.json` (UTF-8, LF) with schema `osiris.deep_research.manifest.v1` hashing every consumed note file and the final report.

Return a concise completion summary to the coordinator confirming the report path, `manifest.json` SHA-256 report digest, total cited sources, and top 3 conclusions.
"""
)


MANIFEST: dict[str, str] = {
    "AGENTS.md": AGENTS_MD,
    ".agents/skills/osiris-preflight-check/SKILL.md": SKILL_PREFLIGHT,
    ".agents/skills/osiris-claims-grounding-audit/SKILL.md": SKILL_CLAIMS_AUDIT,
    ".agents/skills/osiris-dual-node-sync/SKILL.md": SKILL_DUAL_NODE,
    ".agents/skills/osiris-sandbox-patch/SKILL.md": SKILL_SANDBOX_PATCH,
    ".agents/skills/deep-research/SKILL.md": SKILL_DEEP_RESEARCH,
    ".agents/skills/deep-research/references/researcher.md": REF_RESEARCHER,
    ".agents/skills/deep-research/references/report-writer.md": REF_REPORT_WRITER,
}

REPO_SYMLINKS: tuple[tuple[str, str], ...] = (
    ("CLAUDE.md", "AGENTS.md"),
    (".claude/skills", "../.agents/skills"),
)

SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.+)$", re.DOTALL)
REF_LINK_RE = re.compile(r"references/[a-zA-Z0-9_.-]+\.md")


@dataclass(frozen=True)
class SubstrateContext:
    node_id: str
    arch: str
    is_termux: bool
    horizon_sec: int


def probe_substrate() -> SubstrateContext:
    arch = platform.machine().lower()
    prefix = os.environ.get("PREFIX", "")
    is_termux = "com.termux" in prefix or arch in ("aarch64", "arm64")
    if is_termux:
        return SubstrateContext("NODE_ALPHA_EDGE", arch, True, 300)
    return SubstrateContext("NODE_BETA_WORKSTATION", arch, False, 1800)


def resolve_repo_root(explicit_root: str | None) -> Path:
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if out:
            return Path(out).resolve()
    except (OSError, subprocess.SubprocessError):
        pass
    return Path.cwd().resolve()


def write_utf8_lf(path: Path, text: str, dry_run: bool) -> tuple[str, int]:
    clean = text.replace("\r\n", "\n").replace("\r", "\n")
    raw = clean.encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    lines = clean.count("\n")
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return digest, lines


def ensure_relative_symlink(link_path: Path, rel_target: str, dry_run: bool) -> str:
    """Create or repair a relative symlink, backing up any non-symlink collision."""
    if link_path.is_symlink():
        current = os.readlink(link_path)
        if current == rel_target:
            return "unchanged"
        if not dry_run:
            link_path.unlink()
            os.symlink(rel_target, link_path)
        return f"relinked ({current} -> {rel_target})"

    if link_path.exists():
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = link_path.with_name(f"{link_path.name}.bak.{stamp}")
        if not dry_run:
            link_path.rename(backup)
            link_path.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(rel_target, link_path)
        return f"backed up collision -> {backup.name}"

    if not dry_run:
        link_path.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(rel_target, link_path)
    return "created"


def setup_global_links(dry_run: bool) -> list[str]:
    """Wire ~/.agents/skills as canonical store for Claude Code and Antigravity (opt-in only)."""
    home = Path.home().resolve()
    canonical = home / ".agents" / "skills"
    notes: list[str] = []

    if not dry_run:
        canonical.mkdir(parents=True, exist_ok=True)

    legacy_dirs = [
        home / ".claude" / "skills",
        home / ".gemini" / "skills",
        home / ".vibe" / "skills",
        home / ".copilot" / "skills",
    ]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for legacy in legacy_dirs:
        if legacy.exists() and not legacy.is_symlink() and legacy.is_dir():
            for child in sorted(legacy.iterdir()):
                dest = canonical / child.name
                if not dest.exists() and not dry_run:
                    if child.is_dir():
                        shutil.copytree(child, dest)
                    else:
                        shutil.copy2(child, dest)
            notes.append(f"migrated non-conflicting entries from {legacy}")

    for global_link in (home / ".claude" / "skills", home / ".gemini" / "skills"):
        if global_link.is_symlink():
            if Path(os.readlink(global_link)).expanduser().resolve() == canonical:
                notes.append(f"global symlink ok: {global_link}")
                continue
            if not dry_run:
                global_link.unlink()
                os.symlink(str(canonical), global_link)
            notes.append(f"updated global symlink: {global_link} -> {canonical}")
        elif global_link.exists():
            backup = global_link.with_name(f"{global_link.name}.bak.{stamp}")
            if not dry_run:
                global_link.rename(backup)
                global_link.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(str(canonical), global_link)
            notes.append(f"backed up {global_link.name} -> {backup.name} and linked {canonical}")
        else:
            if not dry_run:
                global_link.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(str(canonical), global_link)
            notes.append(f"created global symlink: {global_link} -> {canonical}")

    return notes


def verify_skill_schema(path: Path, text: str) -> list[str]:
    errors: list[str] = []
    if "\u2014" in text or "\u2013" in text:
        errors.append(f"{path}: contains em-dash or en-dash")

    match = FRONTMATTER_RE.match(text)
    if not match:
        errors.append(f"{path}: invalid or missing YAML frontmatter block")
        return errors

    fm_text, body = match.group(1), match.group(2)
    meta: dict[str, str] = {}
    for line in fm_text.splitlines():
        if ":" not in line:
            errors.append(f"{path}: multi-line or malformed frontmatter line: {line!r}")
            continue
        k, v = line.split(":", 1)
        meta[k.strip()] = v.strip()

    if set(meta.keys()) != {"name", "description"}:
        errors.append(f"{path}: frontmatter keys {sorted(meta)} != ['description', 'name']")
        return errors

    name = meta["name"]
    desc = meta["description"]
    if not SKILL_NAME_RE.match(name) or len(name) > 64:
        errors.append(f"{path}: invalid skill name {name!r}")
    if name != path.parent.name:
        errors.append(f"{path}: name {name!r} != directory {path.parent.name!r}")
    if not (20 <= len(desc) <= 1024):
        errors.append(f"{path}: description length {len(desc)} outside 20..1024")
    if "use when" not in desc.lower():
        errors.append(f"{path}: description missing explicit 'Use when' trigger")

    body_lines = body.splitlines()
    if not body.strip() or len(body_lines) > 500:
        errors.append(f"{path}: body line count {len(body_lines)} outside 1..500")

    for rel_ref in sorted(set(REF_LINK_RE.findall(body))):
        ref_file = path.parent / rel_ref
        if not ref_file.is_file():
            errors.append(f"{path}: missing referenced companion file {rel_ref}")

    return errors


def verify_reference_doc(path: Path, text: str) -> list[str]:
    errors: list[str] = []
    if "\u2014" in text or "\u2013" in text:
        errors.append(f"{path}: contains em-dash or en-dash")
    lines = text.splitlines()
    if not lines or not lines[0].startswith("# "):
        errors.append(f"{path}: must start with a top-level '# ' Markdown heading")
    if not (10 <= len(lines) <= 500):
        errors.append(f"{path}: reference doc line count {len(lines)} outside 10..500")
    return errors


def verify_installation(repo_root: Path) -> list[str]:
    errors: list[str] = []

    for rel_path in MANIFEST:
        full_path = repo_root / rel_path
        if not full_path.is_file():
            errors.append(f"missing file: {rel_path}")
            continue
        raw = full_path.read_bytes()
        if b"\r" in raw:
            errors.append(f"CR byte (0x0D) detected in {rel_path}")
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            errors.append(f"invalid UTF-8 in {rel_path}: {exc}")
            continue
        if not decoded.endswith("\n"):
            errors.append(f"missing trailing LF in {rel_path}")
        if rel_path.endswith("/SKILL.md"):
            errors.extend(verify_skill_schema(full_path, decoded))
        elif "/references/" in rel_path and rel_path.endswith(".md"):
            errors.extend(verify_reference_doc(full_path, decoded))

    for rel_link, expected_target in REPO_SYMLINKS:
        link_path = repo_root / rel_link
        if not link_path.is_symlink():
            errors.append(f"not a symlink: {rel_link}")
            continue
        actual_target = os.readlink(link_path)
        if actual_target != expected_target:
            errors.append(
                f"non-portable symlink {rel_link}: {actual_target!r} != {expected_target!r}"
            )
        resolved = link_path.resolve()
        if not resolved.exists():
            errors.append(f"dangling symlink {rel_link} -> {resolved}")
        if not resolved.is_relative_to(repo_root):
            errors.append(f"symlink escapes repo root: {rel_link} -> {resolved}")

    claude_ref = repo_root / ".claude" / "skills" / "deep-research" / "references" / "researcher.md"
    if not claude_ref.is_file():
        errors.append("symlink traversal failed for .claude/skills/deep-research/references/researcher.md")

    agents_file = repo_root / "AGENTS.md"
    claude_file = repo_root / "CLAUDE.md"
    if agents_file.is_file() and claude_file.is_file():
        if agents_file.read_bytes() != claude_file.read_bytes():
            errors.append("CLAUDE.md and AGENTS.md byte contents diverge")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap OSIRIS v6.0.0 AGENTS.md, .agents/skills (with deep-research), and symlinks."
    )
    parser.add_argument("--repo-root", help="Target repository directory (default: git root / CWD)")
    parser.add_argument("--dry-run", action="store_true", help="Preview SHA-256 digests and actions only")
    parser.add_argument(
        "--global-links",
        action="store_true",
        help="Opt-in: configure ~/.agents/skills and symlink ~/.claude/skills + ~/.gemini/skills",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run UTF-8 LF, YAML schema, reference-link, and symlink checks without writing",
    )
    args = parser.parse_args(argv)

    sub = probe_substrate()
    repo_root = resolve_repo_root(args.repo_root)

    print(f"[OSIRIS::BOOTSTRAP] Substrate : {sub.node_id} ({sub.arch}, horizon={sub.horizon_sec}s)")
    print(f"[OSIRIS::BOOTSTRAP] Repo Root : {repo_root}")

    if not args.verify_only:
        mode = "DRY-RUN" if args.dry_run else "WRITE"
        print(f"\n[{mode}] Staging UTF-8 LF artifacts ({len(MANIFEST)} files):")
        for rel_path, content in MANIFEST.items():
            digest, line_count = write_utf8_lf(repo_root / rel_path, content, args.dry_run)
            print(f"  {digest[:12]}  {line_count:>3}L  {rel_path}")

        print(f"\n[{mode}] Linking portable relative symlinks:")
        for rel_link, target in REPO_SYMLINKS:
            state = ensure_relative_symlink(repo_root / rel_link, target, args.dry_run)
            print(f"  {rel_link} -> {target} [{state}]")

        if args.global_links:
            print(f"\n[{mode}] Syncing global ~/.agents/skills links:")
            for note in setup_global_links(args.dry_run):
                print(f"  {note}")

        if args.dry_run:
            print("\n[DRY-RUN COMPLETE] Zero files modified on disk.")
            return 0

    errors = verify_installation(repo_root)
    if errors:
        print("\n[VERIFY FAILED]", file=sys.stderr)
        for err in errors:
            print(f"  [x] {err}", file=sys.stderr)
        return 1

    print(
        f"\n[VERIFY PASS] {len(MANIFEST)} UTF-8 LF files (5 skills + 2 reference docs + AGENTS.md) "
        f"and {len(REPO_SYMLINKS)} portable symlinks verified cleanly."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
