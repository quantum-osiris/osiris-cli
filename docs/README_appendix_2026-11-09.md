# README appendix (moved 2026-11-09)

These fragments were appended to the end of `README.md` after its License section. They were moved here unchanged when the README was cleaned up for the move to github.com/quantum-osiris. The earlier full README is `docs/README_v4_legacy.md`.

* Deconstruction of Pseudo-Physics Terminology: Examination of physics-named code abstractions in the legacy osiris module demonstrated functional disconnects between nomenclature and computation:
  * "Pilot-wave attention" was implemented as a static scalar score multiplier: 1 + e^{-|i-j|/T}.
  * "Golden scale" consisted of a constant multiplication factor (0.618) applied to Feed-Forward Network (FFN) outputs, which was trivially absorbed by subsequent weight matrices.
  * "Phase-conjugate positional encoding" was a sinusoidal positional encoding using base \phi \approx 1.618 instead of 10^4. Whereas standard positional encoding spans four orders of magnitude to prevent positional aliasing across sequence lengths, base-\phi encoding restricted spectral representation to a narrow window of [0.38, 1]\text{ rad/position}, forcing severe frequency aliasing and creating a fundamental mathematical defect.
* Decoder Architectural Defects: The decoder module osiris/decoders/tesseract.py calculated residual syndromes via elementwise Bitwise XOR: \text{syndrome}_{\text{residual}} = \text{syndrome} \oplus \text{correction}. Standard Quantum Error Correction (QEC) decoders map data-qubit corrections through a parity-check matrix H. Treating syndrome bits and correction bits as identical structural objects invalidated the module as a functional QEC decoder.

```bash
python -m pip install -e .
```

Optional feature sets are available:

```bash
python -m pip install -e '.[tui]'
python -m pip install -e '.[quantum]'
python -m pip install -e '.[plot]'
python -m pip install -e '.[forge]'
python -m pip install -e '.[retrieval]'
```

Run the console:

```bash
osiris
```

Or invoke the package directly:

```bash
python -m osiris_cli
```

## High-level architecture

The repository has a few major layers:

### 1. Console and orchestration

The main runtime lives in `osiris_cli/` and includes modules for:

- REPL behavior and command orchestration
- knowledge access and retrieval metadata
- living system state and recordkeeping
- provider/gateway integrations
- training and learning loops

The console entry point is declared in `pyproject.toml`:

```toml
[project.scripts]
osiris = "osiris_cli.osiris_repl:main"
```

### 2. Language-model and learning pipeline

The project includes a local, continuously learning model stack and infrastructure for gating model voice or output. The design emphasizes local execution, structured memory, and governance over unverified claims.

### 3. Provenance and evidence infrastructure

The repo contains a significant amount of code and documentation about write-ahead ledgers, hash chaining, and measured execution logs. This is a key design principle: evidence is captured before and after execution, and recorded with enough structure to support later audit.

### 4. Quantum and hardware research tooling

In addition to the CLI, OSIRIS includes protocols and tooling for:

- IBM QPU job analysis
- benchmark generation and reporting
- dynamical-decoupling studies
- anomaly detection and research notebooks/scripts
- quantum-discovery project scaffolding

## Repository layout

A representative view of the project structure:

- `README.md` — project overview and entry point
- `pyproject.toml` — packaging metadata and optional dependency groups
- `osiris_cli/` — primary console and runtime modules
- `tests/` — repository test suite
- `quantum_discovery/` — research framework for discovery-oriented analysis
- `experiments/` — execution and experiment folders
- `bench_tasks/` — benchmark task definitions
- `docs/` and top-level research markdown files — design notes, evidence, audit docs, and operational summaries

Notable files:

- `LICENSE`
- `pyproject.toml`
- `osiris_cli/osiris_repl.py`
- `osiris_cli/knowledge.py`
- `quantum_discovery/README.md`
- `experiments/`

## Quick start examples

### Start the CLI

```bash
osiris
```

### Install with TUI support

```bash
python -m pip install -e '.[tui]'
```

### Install for quantum research workflows

```bash
python -m pip install -e '.[quantum,tui,plot]'
```

## Research and evidence posture

The repo makes an explicit distinction between:

- local-first model development
- empirical evidence collection
- provenance and auditability
- research exploration versus production readiness

The codebase and docs stress measured behavior, reproducibility, and recorded evidence. This is stronger than a generic model wrapper; it is oriented around evaluating whether the system learns under controlled conditions and whether the observations are auditable.

## Tests

The repository includes a Python test suite under `tests/` and is configured with pytest in `pyproject.toml`.

Typical usage:

```bash
pytest -o cache_dir=/tmp/osiris_pytest_cache tests/
```

## Key design principles

- local-first execution
- explicit governance of model voice and output
- audit-friendly record keeping
- measured evaluation instead of narrative-only claims
- experimentation that can coexist with a production CLI shell

## Notable project themes

The repo repeatedly touches these themes:

| Part | What it does | Where |
|---|---|---|
| Conversation | Plain text talks to OSIRIS. Replies stream; keys typed during a reply are held and sent next, never mixed into the answer. | `osiris_cli/living.py` |
| Grounding | A stable brief of what is on record (pre-registered results, projects) plus per-message retrieval over your notes; answers name the file they used. | `osiris_cli/knowledge.py` |
| Read-only checks | Code, not the model, runs checks before answering — training status, git history, exchange-ledger integrity, system load, a file you name (credential files are never read). `/check` shows them directly. | `osiris_cli/probes.py` |
| Memory | Recent exchanges carry into the next session from the ledger (no model summarises them); `/remember` and `/forget` keep explicit facts. | `osiris_cli/living.py` |
| Speaking gate | The core answers in its own voice only after 30 held-out exchanges at ≤ 2.0 bits/byte and below a unigram baseline, both scored on every byte of the reply. Every 5th exchange is held out and never trained on. | `osiris_cli/living.py` |
| Probe set | About 100 held-out replies, frozen once, scored whole at every 1,000-step milestone checkpoint, so learning can be told apart from a change of topic. `/train probe` scores new milestones; `/train probe freeze` draws the set. | `osiris_cli/probeset.py` |
| Batch training | `osiris train --hours 8 --detach` trains the core on a configured corpus, chat lessons and grounded distillation; distillation pauses while you chat. | `osiris_cli/train.py` |
| Provenance | Every exchange is appended to a SHA-256 hash chain (`~/.osiris/living/exchanges.jsonl`); `/check ledger` recomputes it. | `osiris_cli/probes.py` |

## License and provenance

This repository is source-available under the OSIRIS Dual License v1.0. The project also includes a research record and provenance-oriented documentation intended to support evidence review and scientific traceability.

## Summary

**Until scorer 3 (2026-10-27) the gate read only the opening of each held-out reply:** the core
was scored on the first two 128-byte windows (~254 bytes, mostly the mentor's formulaic opening)
while the unigram baseline was scored on the whole reply. Amendment 4's fix (`4e42bc90`) had never
reached `main`. Both are now scored on every byte the core predicts. On the live core's last 30
held-out replies (step 9250) the whole reply gives 3.20 bits/byte where the first two windows gave
2.90. Scorer-2 scores no longer count; `osiris train --rescore-only` replaces them and first keeps a
copy of `stats.json`, whose scores trace the core's learning one exchange at a time.

## Install and run

- local model operation
- training and learning loops
- governance and evaluation gates
- provenance-aware records
- quantum and hardware analysis tooling

That combination makes it a useful project for reproducible, evidence-anchored experimentation in AI and quantum-adjacent research.

## Repository composition

The GitHub repository composition is roughly:

- Python: 96.6%
- TeX: 1.2%
- Shell: 0.9%
- OpenQASM: 0.7%
- Jupyter Notebook: 0.3%
- Cython: 0.2%
- C: 0.1%

## Related documentation

A few useful starting points in this repo include:

- `README.md`
- `pyproject.toml`
- `quantum_discovery/README.md`
- `osiris_cli/knowledge.py`
- `osiris_cli/osiris_repl.py`
- `LICENSE`

If you want a single high-signal way to enter the project, start with the console entry point and then move into the research documents and experimental folders.
