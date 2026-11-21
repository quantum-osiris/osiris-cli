# osiris-cli

**Governed CLI for AI-assisted quantum programming: LLMs propose circuits, deterministic verifiers (`H2-1SC`, QCCD noise emulation, HQC budget gates) decide, measure, and record cryptographic receipts.**  
*Models propose; deterministic code decides, measures, and records.*

[![CI](https://github.com/quantum-osiris/osiris-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/quantum-osiris/osiris-cli/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-OSIRIS--Dual--v1.0-green.svg)](LICENSE)

`osiris-cli` is a local-first, cryptographically auditable console that sits between language models and quantum execution backends (including **Quantinuum Nexus** `H2-1` / `H2-2` / `Helios`, IBM Quantum, and local QCCD/statevector simulators). It prevents AI agents from burning hardware credits on invalid or unapproved circuits by enforcing deterministic pre-flight verification, exact HQC/shot budget caps, and tamper-evident SHA-256 execution receipts.

---

## 3-Command Quickstart

```bash
pip install -e .
osiris propose "5-qubit bit-flip QEC with mid-circuit reset" --target quantinuum --out circuit.qasm
osiris verify circuit.qasm --target H2-1SC --emulate qccd --record-ledger
```

---

## Why `osiris-cli`?

Autonomous coding agents are useful at proposing quantum circuits and QEC routines—but unreliable at self-verifying them. `osiris-cli` separates **proposal** from **execution**:

1. **Deterministic Pre-Flight (`H2-1SC` & Local QCCD Emulation)**: Every model-generated OpenQASM / Guppy circuit is statically parsed, qubit-ceiling checked (`56q` on H2, `98q` on Helios), lowered to native QCCD gates (`PhasedX`, virtual `Rz`, `ZZMax`/`ZZPhase`), and simulated under realistic trapped-ion trajectory noise with mid-circuit collapse, optical-pumping `reset`, and classical feedforward (`if(creg == val)`) before any paid emulator or QPU call is allowed.
2. **Hard Budget & Tiered Approval Gates**: Enforces per-job shot caps (`OSIRIS_MAX_SHOTS`), exact `Decimal` HQC budgets (`OSIRIS_MAX_HQC` / `--budget-hqc`), and operator-signed Tier 3 approval tokens bound to `(circuit_sha256, target, max_shots, max_hqc)` so neither a human typo nor an AI loop can submit an unapproved or mutated hardware job.
3. **Cryptographic Evidence Ledger**: Every run writes an immutable `exec-canon-1` SHA-256 hash-chained receipt (`SIMULATED`, `PREVIEW_COMPLETE`, or `HARDWARE_ATTESTED`) with circuit hash, native gate breakdown, `G-14` raw integer shot counts, and HQC estimate—verifiable offline via `osiris ledger verify`.

---

## Core Workflows

### 1. Install
```bash
pip install -e .
```

### 2. Propose & Verify a Quantinuum QEC Circuit
```bash
# 1. Generate a candidate circuit with mid-circuit measurement, reset & feedforward
osiris propose "3-qubit bit-flip QEC with mid-circuit syndrome reset" \
  --target quantinuum \
  --out examples/quantinuum/bitflip_qec.qasm

# 2. Run deterministic pre-flight (H2-1SC static check + local QCCD trajectory emulation)
osiris verify examples/quantinuum/bitflip_qec.qasm \
  --target H2-1SC \
  --emulate qccd \
  --shots 1024 \
  --record-ledger
```

### 3. Benchmark All-to-All QCCD vs. Heavy-Hex Routing
```bash
osiris bench examples/quantinuum/steane_7q_x_syndrome.qasm \
  --compare quantinuum_h2_1,quantinuum_helios,ibm_fez \
  --format markdown
```

### 4. Submit to Live Hardware (Tier 3 Governed)
```bash
# Mint an operator-signed approval token bound to exact circuit SHA-256, target, shots & HQC cap
export OSIRIS_TIER3_TOKEN="$(osiris grant-tier3 examples/quantinuum/bitflip_qec.qasm --target H2-1 --shots 500 --budget-hqc 25 --quiet)"

# Fails closed unless H2-1SC passes, HQC estimate <= budget, and approval token verifies
osiris run examples/quantinuum/bitflip_qec.qasm \
  --target H2-1 \
  --shots 500 \
  --budget-hqc 25 \
  --approval-token "$OSIRIS_TIER3_TOKEN"
```

### 5. Verify Receipt Chains Offline
```bash
osiris ledger verify
```

---

## Governance & Execution Tiers

| Tier | Targets | Approval Required | Disposition Label |
| --- | --- | --- | --- |
| **Tier 1 (Free / Local)** | Local Statevector, `H2-1SC` / `H2-2SC` / `Helios-SC` Syntax Checkers | None | `SIMULATED` |
| **Tier 2 (Emulation)** | Local QCCD Trajectory Simulator, `H2-1E` / `H2-2E` / `Helios-E` | Shot & HQC Budget Cap | `SIMULATED` |
| **Tier 3 (Live QPU)** | Quantinuum `H2-1`, `H2-2`, `Helios` (via Nexus), IBM Quantum | Signed Approval Token + HQC Cap | `HARDWARE_ATTESTED` |

> **Strict Integrity Guarantee**: Simulated, syntax-checker, and pre-flight dry-run executions never emit `hardware: true` or `HARDWARE_ATTESTED`.

---

## Repository Structure

```text
osiris-cli/
├── osiris_cli/
│   ├── osiris_repl.py         # Universal CLI & interactive console entrypoint
│   ├── quantinuum_cli.py      # Subcommands: propose, verify, bench, grant-tier3, run, ledger
│   ├── quantinuum_gate.py     # H2-1SC gatekeeper, Tier 3 signed tokens, SHA-256 Nexus ledger
│   ├── quantum_readiness.py   # Coherence budget & Kraus noise-sensitivity readiness gate
│   ├── exec_kernel.py         # Single governed execution kernel & exec-canon-1 hash chain
│   └── providers/
│       ├── quantinuum.py      # Quantinuum H2 / Helios specs, HQC model, QCCD vs. heavy-hex router
│       └── local_qccd.py      # Deterministic OpenQASM/Guppy parser & QCCD trajectory emulator
├── examples/
│   └── quantinuum/            # Runnable QEC & mid-circuit feedforward circuits ([[3,1,1]], [[4,2,2]], [[7,1,3]])
├── tests/                     # Deterministic offline pytest suite (Python 3.9–3.12)
├── POLICY.md                  # Execution governance, Tier 1–3 controls & integrity policy
└── pyproject.toml
```

---

## Configuration

Configure governance limits and optional provider credentials via environment variables:

```bash
export QUANTINUUM_NEXUS_TOKEN="..."           # Quantinuum Nexus API token (Tier 3 only)
export OSIRIS_MAX_SHOTS=10000                 # Hard per-job shot ceiling
export OSIRIS_MAX_HQC=50                      # Hard per-job HQC budget ceiling
export OSIRIS_TIER3_HMAC_SECRET="..."         # Optional HMAC secret (defaults to local Ed25519 keypair)
```

---

## Testing

```bash
pytest -q
```

All tests execute deterministically offline with zero external network calls required.

---

## License

OSIRIS Source-Available Dual License v1.0 — see [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff).
