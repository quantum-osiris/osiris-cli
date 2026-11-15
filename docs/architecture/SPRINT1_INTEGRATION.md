# Sprint 1 integration: governed GitHub access, the DNA-Lang vertical slice, and the IDE API

Status: implemented and tested locally on branch `sprint1/integration` (2026-11-15). Not pushed.

## Layers and owners (verified against the code)

| Layer | Component | Owner module |
|---|---|---|
| 1. Interaction | OSIRIS console (5 panels) | `quantum-ide/public/osiris-console` |
| 1. Interaction | Terminal | `python -m osiris_cli.github_gateway`, `python -m osiris_cli.ide_api` |
| 2. Local service | IDE API v1 (loopback only) | `osiris_cli/ide_api.py` · contract `docs/api/osiris-ide-api-v1.json` |
| 3. Deterministic governance | GitHub policy decision | `osiris_governance.github_policy` (new) |
| 3. Deterministic governance | Execution admission | `osiris_governance.execution_gate.ExecutionGate` |
| 3. Deterministic governance | Evidence recomputation and sealing | `osiris_governance.evidence_verifier.ExternalVerifier` |
| 3. Deterministic governance | Owner authorization (SYS-021) | `osiris_cli.grants.authorize` (ssh-signed, single-use) |
| 4. Simulation | Circuits | `dnalang` → `dnalang.backends.aer` (qiskit-aer) |
| 4. Simulation | Organisms | `organism_sim.Organism` |
| 5. Execution adapters | GitHub | `osiris_cli.github_bridge.GitHubBridge` (gh CLI, argv lists, bodies on stdin) |
| 6. Evidence | Receipts | `osiris_cli.trio_authz.Receipts` (SHA-256 chain) |
| 6. Evidence | Claims register | `osiris_cli.claims` |
| 7. Evaluation | Slice evaluation | `osiris_cli.vertical_slice.evaluate` (exact-distribution TVD, bootstrap CI) |

## Flow 1: a GitHub operation

```
caller ─► GitHubGateway.propose(op, repo, **args)        validated against a fixed operation table
        ─► ActionProposal canonical bytes (RFC 8785)       nonce, request id, 15-minute expiry
        ─► github_policy.evaluate_github_proposal          identity == policy.expected_login
                                                           repo allowlisted, capability granted for the repo
                                                           writes: writes_enabled and all gates PASSED
        ─► read:  request rebuilt from the decided bytes ─► gh api GET ─► shaped fields only
        ─► write: dry run (default) | duplicate check | grants.authorize(digest, "github:<cap>:<repo>")
                  ─► receipt github.write.attempt ─► gh api POST (JSON on stdin) ─► receipt github.write.result
```

Denials are receipted (`github.read.denied`, `github.write.denied`).

## Flow 2: the DNA-Lang vertical slice

```
source ─► dnalang.parse / check / lower ─► Circuit (sha256, OpenQASM 3)
       ─► ExecutionGate.admit_execution(mode=SIMULATION) ─► dnalang.backends.aer.run (seeded)
       ─► evaluate: exact distribution (Statevector), TVD vs p99 multinomial bound,
                    classical fidelity + bootstrap CI, claimed metrics
       ─► ExternalVerifier.verify_and_seal(evidence_class=SIMULATED)  recomputes every claimed number
       ─► Receipts.append("slice.evidence" | "slice.failed")
       ─► result schema osiris-slice-result/v1 ─► IDE
```

Compile errors stop the flow before admission, and nothing is receipted because nothing ran. Any failure after admission is receipted as a negative result.

## Changes to existing behaviour

- `GitHubBridge.check_auth()["raw"]` is now redacted.
- `ExternalVerifier.verify_and_seal` accepts `evidence_class`. The default stays `MEASURED`, so existing callers are unchanged.
- `osiris-governance/errors.py` is restored: merge `649fc5a` had dropped six classes and broken the package import.

## Known limits

- **Python 3.11+ only for dependency inspection.** `GitHubGateway.dependencies()` needs `tomllib`; it raises `UNSUPPORTED` on older versions.
- **New tests skip in CI.** They need the sibling repos installed, and `osiris-cli` CI does not install them.
- **No Next.js integration.** The console is served by the OSIRIS service, not by Next.js; see `quantum-ide/docs/OSIRIS_CONSOLE.md`.
- **The key-creation problem is still open.** `node_beta_governance.get_or_create_operator_key()` creates its own HMAC key in-process. The gateway does not use it, because SYS-021 ssh grants are the write authority. The issue itself remains.
