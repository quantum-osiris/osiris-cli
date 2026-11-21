# OSIRIS Local Runtime Policy

This policy governs the canonical OSIRIS repository and its local runtime. It is
deliberately narrower than the project's research and product aspirations.

## Authority model

- Human intent and explicit constraints are inputs, not execution grants.
- Models, agents, Φ/CCCE/W2 values, confidence, classifications, and generated
  explanations may propose work but never authorize it.
- Host-side policy validates schemas, paths, artifacts, budgets, and execution mode.
- An isolated worker may execute only a host-issued, expiring capability.
- Promotion from staging to a protected workspace requires explicit human approval
  bound to the exact candidate and evidence hashes.

## Default-deny capabilities

The following are denied unless a separate policy version, implementation, negative
tests, evidence record, and human approval establish a narrower capability:

- Network access, cloud APIs, remote models, MCP servers, and web research.
- Shell execution, arbitrary subprocesses, native-module loading, and dynamic code
  execution.
- Exchange, broker, wallet, market-order, and financial-account access.
- Quantum hardware submission, publication, public deployment, and automatic Git push.
- Credential discovery, export, rotation, or access to SSH, cloud, provider, or
  browser/session credentials.
- Deletion, recursive cleanup, filesystem mounts, host sockets, and access outside
  an explicit workspace or job root.

## Tiered quantum execution & HQC governance

Quantum circuit verification and hardware submission follow a strict three-tier gate:

- **Tier 1 (Static Syntax & Resource Gate)**: Local OpenQASM / Guppy validation against
  `H2-1SC`, `H2-2SC`, and `Helios-SC` resource ceilings. Emits `disposition: "SIMULATED"`
  with `hardware: false`.
- **Tier 2 (Local QCCD & Noise Emulation)**: Deterministic local trajectory or density-matrix
  simulation bounded by `OSIRIS_MAX_SHOTS`. Preserves unmutated `G-14` raw integer shot
  counts and SHA-256 digests (`disposition: "SIMULATED"`, `hardware: false`).
- **Tier 3 (Live QPU via Quantinuum Nexus / IBM Quantum)**: Fails closed unless:
  1. Mandatory Tier 1 syntax-checker pre-flight (`H2-1SC` / `Helios-SC`) succeeds,
  2. Exact `Decimal` HQC estimate satisfies both `--budget-hqc` and `OSIRIS_MAX_HQC`,
  3. An operator-signed approval token (`Ed25519` or `OSIRIS_TIER3_HMAC_SECRET`) matches the
     exact `(circuit_sha256, target, max_shots, max_hqc)` tuple prior to expiry, and
  4. Live submission is explicitly enabled (`OSIRIS_ALLOW_LIVE_NEXUS_SUBMIT=1` + valid
     provider credentials). Dry-run pre-flight receipts record `PREVIEW_COMPLETE` with
     `hardware: false` and never claim `HARDWARE_ATTESTED`.

## Artifact handling

- Treat `.dna`, model output, archives, generated code, and experiment files as
  untrusted input.
- Require a bounded schema and content hash before an artifact enters a workflow.
- Reject native binaries, unknown executable payloads, oversized files, and
  unsupported archive formats before loader or worker dispatch.
- Store references and hashes instead of raw secrets or unbounded payloads.

## Cloud and external services

The GCP project `dnalang` and the Cloud Shell identity observed during planning are
operator resources, not OSIRIS runtime credentials. Cloud Shell ADC, browser state,
tokens, API keys, service-account keys, and project metadata must not enter agents,
workers, model prompts, repositories, artifacts, MCP, or local ledgers.

No cloud resource creation, API enablement, public endpoint, agent deployment, or
remote provider call is approved by this policy.

## Verification standard

A capability is `verified` only when implementation, focused tests, a declared clean
environment result, boundary documentation, and an evidence record all exist.
Otherwise report `planned`, `partial`, `implemented-unverified`, `blocked`, or
`disabled`. A syntax check alone is not a security or policy verification.
