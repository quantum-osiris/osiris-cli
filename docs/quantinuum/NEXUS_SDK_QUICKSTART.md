# Quantinuum Nexus with osiris-cli (qnexus SDK)

`osiris nexus` drives Quantinuum Nexus through the official `qnexus` SDK, pinned to 0.51.4. Every call it makes is tested against that version's real function signatures (`tests/test_nexus_sdk.py`).

The older `QuantinuumNexusClient` in `providers/quantinuum_nexus.py` sends hand-written REST calls to `/api/v1/backends`, `/api/v1/targets` and `/api/v1/jobs`. Those paths are not part of the Nexus API, so use `osiris nexus` for real runs.

## Once access is granted

```bash
pip install 'osiris-cli[nexus]'          # installs qnexus==0.51.4
qnx login                                # interactive; tokens are stored by qnexus, not by OSIRIS
```

## 1. Preflight (Tier 1, free)

```bash
osiris nexus preflight docs/xtop005/quantinuum/circuits_h2_1_tket.json --target H2-1 --shots 500 \
  --out preflight.json
```

This command:

1. uploads the circuits to the Nexus project `OSIRIS`;
2. compiles them for H2-1 on Nexus at `-O 0`;
3. prices each compiled circuit with Quantinuum's syntax checker (`circuits.cost(..., syntax_checker="H2-1SC")`);
4. cross-checks that price against the local HQC formula;
5. writes a Tier 1 receipt to `$OSIRIS_HOME/quantum/nexus_dispositions.jsonl`.

The preflight output includes the compiled-batch SHA-256. A hardware approval token is bound to that hash.

**First check:** compare Nexus's HQC total with the offline design figure for the XTOP-005 decisive comparison (1,336 HQC at 500 shots). Any difference shows exactly where the local model is wrong.

## 2. Emulate (Tier 2, emulator quota)

```bash
osiris nexus run docs/xtop005/quantinuum/circuits_h2_1_tket.json --target H2-1E --shots 100 \
  --max-emulator-hqc 200 --match '_(Q0|Q3)_L3_' --out h2-1e.json
```

The run refuses to start if the estimated cost is above `--max-emulator-hqc`. Each circuit is also submitted with its own `max_cost` cap. Results are recorded as `SIMULATED` with `hardware=false`.

Circuit names follow `xtop005q_<arm>_L<depth>_<setting>`, so `--match '_(Q0|Q3)_L3_'` selects the decisive comparison: bare against fully encoded with checks at depth 3, 18 circuits in total.

## 3. Hardware (Tier 3, only after an explicit decision)

```bash
osiris nexus grant --compiled-sha <compiled_batch_sha256 from preflight> --target H2-1 --shots 500 --budget-hqc 1400
osiris nexus run <circuits> --target H2-1 --shots 500 --approval-token <token>
```

The run refuses to start in any of these cases:

- there is no token;
- the token was signed for a different compiled batch, target, shot count or HQC budget;
- Nexus compiles the circuits to a different batch than the one the token names.

`HARDWARE_ATTESTED` is written only for a completed Tier 3 job with a real job ID.

## Ledger

```bash
osiris nexus status        # verifies the hash chain and lists recent receipts
```
