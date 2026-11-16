# OSIRIS-HP-004 Pre-Registration: Layout-Locked 9-Basis 4-Qubit Joint Bell Stabilizer Capacity Test

- **Benchmark ID**: `OSIRIS-HP-004`
- **Pre-Registration Timestamp**: `2026-11-16T10:21:00-05:00` (frozen prior to QPU job submission)
- **Target Processor**: `ibm_marrakesh` (156-qubit IBM Heron r2, `open-instance`)
- **Pinned Physical Tree Layout (`routing_method="none"`)**:
  `[w0=34(R2), w1=52(R1), w2=32(C1), w3=33(E1), w4=39(D1/M1'), w5=53(E2/M2'), w6=54(D2), w7=55(C2)]`
- **Pinned Readout Transmons (Identical Across All 286 Circuits)**:
  `c0=q34(R2), c1=q39(M1'), c2=q52(R1), c3=q53(M2')`
- **Execution Payload**: `1` `SamplerV2` job, `286` circuits (`16` computational-basis SPAM calibration states + `6` randomized blocks × `5` arms × `9` Pauli basis settings `{X,Y,Z}_1 × {X,Y,Z}_2`), `1,024` shots/circuit (`292,864` total shots), zero error mitigation in execution (`resilience_level=0`), zero postselection.

## 1. Exact Mathematical Estimator

For each block and pooled across all `6` blocks, the 4-qubit joint reference-output Bell state overlap is evaluated directly from all 16 four-qubit Bell stabilizers measured across the 9 basis settings $(P, Q) \in \{X, Y, Z\}^2$ with $s_X = +1, s_Y = -1, s_Z = +1$:

$$F_{\text{joint}} = \frac{1}{16}\left(1 + \sum_{P \in \{X,Y,Z\}} s_P \langle P_{R_2} P_{M_1'} \rangle + \sum_{Q \in \{X,Y,Z\}} s_Q \langle Q_{R_1} Q_{M_2'} \rangle + \sum_{P,Q \in \{X,Y,Z\}} s_P s_Q \langle P_{R_2} P_{M_1'} Q_{R_1} Q_{M_2'} \rangle \right)$$

Uncertainty is bounded conservatively by:
$$\text{SE}_{\text{eff}} = \max\!\left(\text{SEM}_{\text{block, ddof=1}}, \; \sigma_{\text{shot}}\right)$$

## 2. Pre-Registered Hypotheses (Evaluated Strictly on Raw, Unmitigated Counts)

- **H1 (Barriered Unitary Inverse Baseline — `Arm A`, 38 CZs)**:
  Raw $F_{\text{joint}}(\text{Arm A}) > 0.6000$.
- **H2 (Primary Hypothesis — Certified `> 1 Ebit` / Schmidt-Rank $\ge 3$ Restricted Recovery — `Arm B`, 32 CZs)**:
  Raw $F_{\text{joint}}(\text{Arm B}) - 5\,\text{SE}_{\text{eff}}(\text{Arm B}) > \mathbf{0.5000}$ (strictly exceeding the exact 1-ebit Schmidt-rank-2 ceiling across $(R_1 R_2) : (M_1' M_2')$ at $> 5\sigma$, with zero physical gates applied to horizon remnants `q32, q55` or references `q34, q52` during decoding).
- **H3 (0-Ebit Classical Separable Ceiling — `Arm B-Ctrl`, 30 CZs, $|0000\rangle_{BE}$)**:
  Raw $F_{\text{joint}}(\text{Arm B-Ctrl}) \le 0.2500 + 3\,\text{SE}_{\text{eff}}$ and $> 0.1250$ (confirming classical late-radiation correlation without quantum entanglement).
- **H4 (Single-Photon Sub-Capacity Collapse — `Arm B-Def`, 25 CZs, `q54` quarantined)**:
  Raw $F_{\text{joint}}(\text{Arm B-Def}) \le 0.1250 + 3\,\text{SE}_{\text{eff}}$ and raw $F_1 \le 0.5000 + 3\,\sigma_{F_1}$.
- **H5 (Unitary Specificity — `Arm B-Held`, 35 CZs, mismatched scrambler $U_2$)**:
  Raw $F_{\text{joint}}(\text{Arm B-Held}) \in [0.0400, 0.0900]$ (consistent with the maximally mixed $1/16 = 0.0625$ floor).

## 3. Secondary Sensitivity Analysis

Empirical $16 \times 16$ readout assignment matrix $M_{\text{cal}}$ measured on `[34, 39, 52, 53]` from circuits `0..15` (`spam_cal_0000`..`1111`) in the **same job** is inverted via non-negative least-squares / clipped linear inversion strictly as a secondary sensitivity report; it may **not** be used to overturn a raw-count failure on H1–H5.
