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

   ```bash
   python3 -m py_compile osiris_cli/*.py osiris_cli/providers/*.py
   pytest -o cache_dir=/tmp/osiris_pytest_cache -q tests/
   ```

   - Use `tmp_path` and monkeypatched `$OSIRIS_HOME` in every new test so unit tests never mutate the operator's live `~/.osiris/` ledgers.

4. **Present Diffs Clearly for Human Sign-Off:**
   - When the operator requests review-only mode or `EXEC_GATE` is `STRICT`, output a minimal unified diff, summarize ledger/schema impact (ensuring backward compatibility with `verify_chain` and `verify_chain_tail`), and wait for explicit authorization before mutating files or committing.

## Gotchas

- **Breaking `verify_chain_tail` backward compatibility:** When adding metadata fields (such as `routing`, `probe_receipts`, or `grounding_audit`) to `exchanges.jsonl`, ensure existing ledger entries verify against their original recorded payload format while new entries include the added fields deterministically.
- **Blocking the async TUI loop:** Wrap synchronous file hashing, BM25F scoring, shadow `osiris.nclm` bits/byte scoring, or HTTP calls in background threads when called from an interactive loop so keystrokes and `TypeAhead` stay responsive.
