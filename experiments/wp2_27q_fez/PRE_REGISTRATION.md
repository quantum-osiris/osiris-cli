# Pre-Registration: `OSIRIS-WP2-27Q-FEZ` — Constant-Skeleton 27-Qubit Split-Sample-Debiased Kantorovich & Sinkhorn Optimal-Transport Benchmark on `ibm_fez`

- **Protocol ID:** `OSIRIS-WP2-27Q-FEZ` (Closes `NB-11` / `W2_MATHEMATICAL_ERRATUM.md` & executes DOE QCUP `WP2` methodology on 27Q Heron r2)
- **Target Processor:** `ibm_fez` (156Q IBM Heron r2, `open-instance` via governed `troykey.json`)
- **Pinned 27-Transmon Connected Heavy-Hex Tree (`26` native `CZ` edges, `0` SWAPs):**
  `[114, 115, 99, 95, 94, 93, 79, 92, 91, 90, 98, 111, 110, 109, 118, 112, 108, 89, 88, 78, 69, 68, 70, 71, 58, 73, 74]`
- **Constant-Skeleton Zero-Confound Invariant:**
  Every circuit in `ARM_A` (`NB11_REPAIRED_27Q`) and `ARM_B` (`ENTANGLED_BRICKWALL_27Q`) executes **exact `52 CZ` gates** (two full 3-color König edge-matching sweeps over all 26 native edges) and **constant `SX` count** via universal 2-`SX` slots, eliminating the 129–132 depth confound of `w2_robustness_paper.tex`.
- **Shot Budget:** `4` interleaved blocks $\times$ `12` PUBs/block = `48` PUBs $\times$ `2,048` shots = `98,304` shots (`8,192` shots per condition).

## Preregistered Hypotheses & Kill Rules

1. **H1 — Elimination of Weed–Bach $O(S^{-1/27})$ Sparsity Bias on Live 27Q Hardware:**
   - **Prediction:** Across identical-circuit inter-block replicates (`b0..b3`) at $\theta \in \{2.118, 2.618, 3.118\}$ rad, both defective Eq. (7) and raw plug-in LP $W_2^2$ inflate strongly with Shannon entropy ($\theta_1 \to \theta_3$), whereas split-sample-debiased Sinkhorn $\widehat{S}_\varepsilon$ ($\varepsilon=0.05$) and matched cross-split LP $\widehat{W}_{2,\text{matched}}^2$ collapse inter-block A/A drift to $|\widehat{S}_\varepsilon| < 0.005$ across **all three $\theta$ points**.
2. **H2 — Falsification of Golden-Ratio $\theta^\star = \varphi^2 \approx 2.618$ Stability Specialness:**
   - **Prediction:** Under split-sample-debiased optimal transport (`ARM_A` and `ARM_B`), $\theta^\star = 2.618$ rad exhibits no statistically significant stability advantage ($p > 0.05$) over $\theta_1 = 2.118$ or $\theta_3 = 3.118$ rad.
3. **H3 — Monotonic Detection of Real Coherent Drift (`ARM_B` $\Delta\theta \in \{0.20, 0.45\}$ rad):**
   - **Prediction:** Comparing baseline $\theta^\star = 2.618$ against $\theta^\star + 0.20$ and $\theta^\star + 0.45$ rad on `ARM_B`, $\widehat{S}_\varepsilon$ increases strictly monotonically ($0 < \widehat{S}_\varepsilon(0.20) < \widehat{S}_\varepsilon(0.45)$) and exceeds the inter-block A/A null by $> 10\sigma$ at $\Delta\theta = 0.45$ rad.
