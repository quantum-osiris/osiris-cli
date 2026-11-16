# OSIRIS-HP-004B Pre-Registration: Cross-Processor Replication on `ibm_fez`

- **Benchmark ID**: `OSIRIS-HP-004B` (Cross-processor replication of `OSIRIS-HP-004` Job `dbti1fbsbsbs73cb2vu0`)
- **Target Processor**: `ibm_fez` (156-qubit IBM Heron r2, `open-instance` — distinct physical cryostat & QPU from `ibm_marrakesh`)
- **Pinned Physical Tree Layout (`routing_method="none"`)**:
  `[w0=2(R2), w1=24(R1), w2=4(C1), w3=3(E1), w4=16(D1/M1'), w5=23(E2/M2'), w6=22(D2), w7=21(C2)]`
  (mean internal CZ error `0.00179`, readout errors `[0.00391, 0.00830, 0.00537, 0.00244]`, $T_1 \in [68, 206]\,\mu\text{s}$, $T_2 \in [78, 174]\,\mu\text{s}$)
- **Pinned Readout Transmons (Identical Across All 286 Circuits)**:
  `c0=q2(R2), c1=q16(M1'), c2=q24(R1), c3=q23(M2')`
- **Payload & Budget**: `1` `SamplerV2` job, `286` circuits (`16` SPAM states + `6` randomized blocks × `5` arms × `9` Pauli bases, `seed=20261117`) × `1,024` shots = **`292,864` shots** (`~81s` / `~1.34 RU` out of `519s` / `8.66 RU` remaining), `resilience_level=0`, zero postselection.
- **Status**: **STAGED OFFLINE ONLY — REQUIRES SEPARATE EXPLICIT OWNER APPROVAL BEFORE DISPATCH.**

## Pre-Registered Replication Hypotheses (Evaluated Strictly on Raw Unmitigated Counts)

- **H1 (`Arm A`, 38 CZs)**: Raw $F_{\text{joint}} > 0.6000$.
- **H2 (Primary Cross-Chip Replication — `Arm B`, 32 CZs)**: Raw $F_{\text{joint}} - 5\,\text{SE}_{\text{eff}} > \mathbf{0.5000}$ (certifying $> 1\text{ ebit}$ / Schmidt rank $\ge 3$ on an independent quantum processor with zero physical gates applied to horizon remnants `q4, q21` or references `q2, q24` during decoding).
- **H3 (`Arm B-Ctrl`, 30 CZs, $|0000\rangle_{BE}$)**: Raw $F_{\text{joint}} \le 0.2500 + 3\,\text{SE}_{\text{eff}}$ and $> 0.1250$.
- **H4 (`Arm B-Def`, 25 CZs, `q22` quarantined)**: Raw $F_{\text{joint}} \le 0.1250 + 3\,\text{SE}_{\text{eff}}$ and raw $F_1 \le 0.5000 + 3\,\sigma_{F_1}$.
- **H5 (`Arm B-Held`, 35 CZs, mismatched $U_2$)**: Raw $F_{\text{joint}} \in [0.0400, 0.0900]$.
