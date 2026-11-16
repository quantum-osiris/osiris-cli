# OSIRIS-QEC-001 Pre-Registration: Multi-Round Repetition-Code Logical Error Scaling ($d \in \{3, 5, 7\}$) on `ibm_fez`

- **Benchmark ID**: `OSIRIS-QEC-001`
- **Target Processor**: `ibm_fez` (156-qubit IBM Heron r2, `open-instance`)
- **Pinned 13-Transmon Heavy-Hex Chain (`routing_method="none"`)**:
  `[111, 110, 109, 108, 107, 106, 105, 117, 125, 124, 123, 122, 121]` (mean `CZ` error `0.00226`, max `CZ` error `0.00274`, mean ancilla readout error `0.00724`, mean data readout error `0.00970`):
  - **$d = 3$ (`5Q`, center)**: `[107, 106, 105, 117, 125]` (`3` data + `2` syndrome ancillas)
  - **$d = 5$ (`9Q`, centered)**: `[109, 108, 107, 106, 105, 117, 125, 124, 123]` (`5` data + `4` syndrome ancillas)
  - **$d = 7$ (`13Q`, full chain)**: `[111, 110, 109, 108, 107, 106, 105, 117, 125, 124, 123, 122, 121]` (`7` data + `6` syndrome ancillas)
- **Frozen ISA Payload**: `qec001_fez_staged_circuits.qpy` (`SHA-256: 357a29ad2746e2be94522702516dce3e04319894c1dc981596c9a3640041c056`)
- **Payload & Budget**: `1` `SamplerV2` job, `192` dynamic mid-circuit `measure` + `reset` circuits (`4` shuffled blocks × `2` code bases `[bit_flip_z, phase_flip_x]` × `3` distances `[3, 5, 7]` × `4` syndrome cycle counts `[1, 3, 5, 7]` × `2` logical states `[0, 1]`, `seed=20261118`) × `2,048` shots = **`393,216` total shots** (`~92s` / `~1.53 RU` out of `440s` / `7.36 RU` remaining), `resilience_level=0`, zero postselection.
- **Status**: **STAGED OFFLINE ONLY (`0` QPU SECONDS SPENT) — REQUIRES SEPARATE EXPLICIT OWNER APPROVAL BEFORE QPU DISPATCH.**

## Pre-Registered Falsifiable Hypotheses (Evaluated via Exact Blossom MWPM on Raw Unmitigated Counts)

Let $\bar{P}_L(b, d, R)$ denote the pooled raw MWPM logical error probability across both initial states $\{0_L, 1_L\}$ and all `4` interleaved blocks (`16,384` shots per `(basis, d, R)` point), $\epsilon_L(b, d, R) = \frac{1}{2}[1 - (1 - 2\bar{P}_L(b, d, R))^{1/R}]$ the logical error per syndrome cycle, and $\Lambda_{d \to d+2}(b, R) = \epsilon_L(b, d, R) / \max(\epsilon_L(b, d+2, R), \frac{1}{2\cdot 16384\cdot R})$ the cycle suppression factor:

1. **QEC-H1 (Bit-Flip Sub-Threshold Monotonic Suppression at $R = 5$ and $R = 7$)**:
   $\epsilon_L(\text{bit\_flip\_z}, 3, R) > \epsilon_L(\text{bit\_flip\_z}, 5, R) > \epsilon_L(\text{bit\_flip\_z}, 7, R)$ with $\Lambda_{3 \to 5}(\text{bit\_flip\_z}, R) > 1.25$ and $\Lambda_{3 \to 7} = \epsilon_L(3, R)/\epsilon_L(7, R) > 2.0$.
2. **QEC-H2 (Phase-Flip Sub-Threshold Suppression Under Real Mid-Circuit Readout/Reset Dephasing)**:
   Test whether `ibm_fez` mid-circuit ancilla `measure` + `reset` (`~2.6 µs`/cycle idle on data transmons) stays below the 1D phase-flip repetition threshold ($\Lambda_{3 \to 5}(\text{phase\_flip\_x}, R=5) > 1.15$ and $\Lambda_{5 \to 7}(\text{phase\_flip\_x}, R=5) > 1.15$). Note: A measured $\Lambda < 1$ on `phase_flip_x` (e.g., due to measurement-induced spectator dephasing exceeding $T_2$ predictions) is explicitly pre-registered as a valid falsification / diagnostic finding rather than failure.
3. **QEC-H3 (Space-Time MWPM Advantage Over Un-Decoded Final Majority Vote at $R = 7$)**:
   Pooling $R = 7$, space-time Blossom MWPM achieves strictly lower or equal logical error than final-round-only majority vote on $d \in \{5, 7\}$ while resolving per-round syndrome defect rates $\bar{p}_{\text{det}}(b, d, R) < 0.12$.
