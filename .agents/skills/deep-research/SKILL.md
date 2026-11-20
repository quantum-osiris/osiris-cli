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

```bash
mkdir -p "{workspace_root}/research_notes/{research title}" "{workspace_root}/reports"
```

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
