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
| **L6 Benchmark** | `protege.py`, `osiris_bench.py`, `osiris_cli/prepost.py`, `osiris_cli/retrieval/` | Preserve reference-pass and `known_bad` / `unchanged`-fail invariants across the 9-task bench suite (`125a27de6d2fe733` as of PR #55; prior runs at `09037df63614` are not directly comparable). Keep `osiris_bench.runner_sha256()` comparability honest. SciFact `olmo3-embed-7b` ranks evidence capsules only and never issues truth verdicts. |
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

```bash
# 1. Unified skill & symlink verification
python3 scripts/bootstrap_osiris_agents.py --verify-only

# 2. Syntax & AST sanity across modified modules
python3 -m py_compile osiris_cli/*.py osiris_cli/providers/*.py

# 3. Automated offline unit & regression suite
pytest -o cache_dir=/tmp/osiris_pytest_cache -q tests/

# 4. Working tree & diff whitespace hygiene audit (ensure zero staged secrets)
git diff --check
git status --short
```

### Definition of Done

- All new/modified tests pass offline and deterministically on both `aarch64` and `x86_64`.
- `verify_chain` and O(1) `verify_chain_tail` pass on existing and newly written JSONL ledger entries.
- `0` stub/placeholder functions (`payload_gate.py`), `0` orphan runs, and `0` phantom citations or unbacked metrics marked `distill_eligible`.
