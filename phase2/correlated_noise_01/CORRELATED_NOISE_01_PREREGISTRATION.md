# Separate Historical Evidence Reconciliation & Pre-Registration (`CORRELATED_NOISE_01_PREREGISTRATION.md`)

**Experiment ID:** `CORRELATED-NOISE-01`  
**Status:** `Implemented and tested` offline (`phase2/correlated_noise_01/`); QPU hardware execution `Proposed only` (**0 QPU jobs submitted; halted at authorization boundary**).

---

## 1. Separate Reconciliation of Historical Evidence (`E0`–`E4` & Bridge `E1`–`E5`)

Per Phase 2 Section 6 rules, historical runs are reconciled **individually** without merging distinct endpoints, couplers, or calibration epochs, and all negative/null findings are preserved verbatim:

### 1.1 Coupler-Aware Staggered DD Campaign (`experiments/experiment_e/`, Nov 2026)
- **`E Pilot` (`ibm_marrakesh`, Job `dblujopkbkrc73coem40`, Ledger `#6`):** **Negative / Bug.** Sequence `H(u); CZ(u,v); H(v)` prepared separable $|++\rangle$ ($F_{\text{Bell}} \approx 0.465 < 0.50$).
- **`E0` (`ibm_marrakesh`, Job `dblup49kbkrc73cof20g`, `221,184` shots, Ledger `#7`):** Certified $0\,\mu\text{s}$ Bell baseline (`97.30%` on `(14,15)`, `93.55%` on `(19,35)`, `94.07%` on `(33,32)`) and $4\,\mu\text{s}$ bare-idle decay (`87.81%, 86.08%, 80.57%`). *Forensic limitation:* Raw `.py`/`.qpy` files were not committed to local disk.
- **`E1` (`ibm_marrakesh`, Job `dblutq9kbkrc73cofcg0`, `276,480` shots, Ledger `#8`):** Exploratory stagger sweep. Fast coupler `(14,15)` gained `+2.28 pp` at `8 µs / 80 ns`, **but** slow coupler `(19,35)` (`80 ns` CZ) favored synchronous XY8 (`-0.54 pp` at `4 µs`, `-0.16 pp` at `8 µs`), and `(33,32)` suffered CPMG breakdown at `8 µs` (`-0.90 pp`).
- **`E2` (`ibm_marrakesh`, Job `dblv7sm74qds73dr8ho0`, `294,912` shots, 4 unseen `68 ns` couplers `(15,19),(32,31),(34,33),(35,34)`, Ledger `#10–#11`):**
  - Fixed `STAG_XY8_80` beat `SYNC_XY8` by **`+0.446 ± 0.113 pp`** coupler-clustered ($t_3 = 3.95, p_{2s} = 0.0289$; naive $N=32$ $p_{2s} = 0.0621$; **0/8 cells significant under 8-cell Bonferroni $\alpha = 0.00625$**).
  - **Preserved Negative Finding:** Frozen `Adaptive v1.0` **lost** to fixed `STAG_80` (`-0.276 ± 0.090 pp`, $p_{2s} = 0.0545$) and was **null** against `SYNC_XY8` (`+0.169 ± 0.200 pp`, $p_{2s} = 0.4588$).
- **`E3 Pilot` (`ibm_marrakesh`, Job `dbm00gd8mdtc73d6vjcg`, `221,184` shots, `(12,13),(24,25),(44,45)`, Ledger `#15`):**
  - **Preserved Null & Drift Finding:** `OSIRIS_V2 − FIXED_80 = +0.651 ± 0.344 pp` ($p_{2s} = 0.199$, **null**), trailing `QISKIT_OPT_STAG` (`-0.100 pp`), and collapsing to **`+0.038 pp`** under strict pre-registered arm scoring. Accidental A/A cells drifted by **`+0.70, +1.09, -0.63 pp`**.
- **`E3 Cross-Chip Transfer` (`ibm_fez`, Job `dbm86158ft8s738ucqi0`, `82,944` shots, `(87,88),(61,62),(100,101)`, Ledger `#20`):**
  - **Preserved Negative Finding:** Zero-retraining transfer **failed** (`Adaptive − STAG80 = -0.803 ± 0.601 pp`, $p_{2s} = 0.218$; `STAG80 − SYNC` on `(61,62)` = `+0.109 ± 0.461 pp`, $p = 0.820$; accidental A/A drift on `(100,101)` = **`-2.16 pp`**).
- **`ZZ-Ramsey Stage 1` (`ibm_marrakesh`, Jobs `dbm8ni674qds73drr68g` & `dbm8q31kbkrc73cp2bng`, `65,536` shots, Ledger `#21–#22`):**
  - **Preserved Null & Fitting Fix:** Corrected fundamental refit (`refit_zz_ramsey.py`) measured static $\zeta_{14,15}/2\pi = \mathbf{3.17 \pm 3.47\text{ kHz}}$ ($p = 0.36$) and $\zeta_{61,62}/2\pi = \mathbf{0.29\text{ kHz}}$ — far too small to produce appreciable idle Bell dephasing when spectators remain unpulsed in $|0\rangle$.
- **`E4` (`ibm_marrakesh`, Job `dbm9ctu74qds73drs9j0`, `73,728` shots, `(14,15)` & `(61,62)`, Ledger `#24`):**
  - **Preserved Negative Finding:** Step fraction $S_{32} = 9.5\% \ll 75\%$ falsified microwave pulse-collision dominance, **and** flat-constant goodness-of-fit passed ($\chi^2_5 = 1.87, p = 0.866$ on `(14,15)`; $\chi^2_5 = 1.96, p = 0.854$ on `(61,62)`), establishing that `2,048` shots/offset was underpowered against the empirical **`1.03–1.11 pp` RMS A/A drift floor**.

### 1.2 Bridge Apparatus Drift Campaign (`bridge/results/drift_checks/`, `81` `ibm_fez` Jobs, Oct 27–28, 2026)
- Independently established (`E2`–`E5`, `1,092,000` total shots) intra-calibration coherent rotation-angle drift (`140–211×` binomial variance on `x100`, `E2/E3`) with pairwise inter-qubit gain-increment correlation **$r = 0.812$ ($p = 0.0005$, `E5`)** from shared control electronics, plus `340 kHz` qubit frequency jumps (`Q82`, `E4a`).

---

## 2. Pre-Registered Statistical, Apparatus-Control & Falsification Specification

1. **Primary Observable & Apparatus De-Confounding Controls:**
   - Three-basis Pauli Bell-state fidelity $\mathcal{F}_{\text{Bell}}(\rho) = \frac{1}{4}(1 + \langle XX\rangle - \langle YY\rangle + \langle ZZ\rangle)$ for $|\Phi^+\rangle$ and $\frac{1}{4}(1 + \langle XX\rangle + \langle YY\rangle - \langle ZZ\rangle)$ for $|\Psi^+\rangle$, evaluated across the `dt = 4.0 ns`-aligned 27-condition diagnostic suite (`T = 8.192 µs = 2048 dt`, offsets `{0, 16, 32, 48, 80, 160, 256} ns` = `{0, 4, 8, 12, 20, 40, 64} dt`, plus `16.384 µs` pulse-density arm).
   - **Mandatory Classical-Mimic Controls:** (a) $4\times 4$ two-qubit readout assignment matrix calibration per pair (controlling parity-correlated readout crosstalk $P(00|11) \neq P(01|10)$); (b) $T=0$ SWAP-routed Bell baselines on $d=2,3$ pairs; (c) simultaneous far-separated ($d \ge 6$ hops) shared-LO vs disjoint-LO control pairs to distinguish classical shared-LO/AWG phase noise ($1\text{–}20\,\mu\text{s}$) from on-chip substrate correlation.
2. **Sample Size & Block Floor:**
   - $B = 8$ randomized interleaved hardware blocks $\times$ `4,200` shots/basis/block = **`33,600` shots/basis/condition** (`>= 33,500` requirement), with mandatory within-block A/A duplicate sentinel circuits (`0 ns` sync run twice per block) and live execution-time `backend.properties()` archival.
3. **Structured-Covariance GLS Fit & Model-Comparison Statistic:**
   - **Generalized Least Squares (GLS)** (`fit_competing_models`) using the empirical block-difference covariance matrix $\hat{\Sigma}$ (capturing off-diagonal intra-block pulse-gain covariance across $N \in \{0,1,8,16,32\}$ so `Bridge E5` gain drift cannot masquerade as `H3` bath memory).
   - Penalized **GLS-BIC** $\mathrm{BIC}_m = \chi^2_{\text{GLS},m} + k_m \ln(n_{\text{cond}})$ (reported alongside small-sample $\mathrm{AIC}_c$) with a null-calibrated **$\Delta\mathrm{BIC} = \mathrm{BIC}_{H0} - \mathrm{BIC}_m \ge 8.0$** threshold over `H0`, plus Holm–Bonferroni FWER control ($\alpha = 0.05$) across the 4 mechanistic contrasts:
     - $C_{\text{DFS}}$: $[F_{\Phi^+}(256\text{ ns}) - F_{\Phi^+}(0)] - [F_{\Psi^+}(256\text{ ns}) - F_{\Psi^+}(0)]$ (`> 0` required for `H2`/`H3` spatial dephasing).
     - $C_{\text{spec}}$: $[F_{\Phi^+}(256\text{ ns}) - F_{\Phi^+}(0)]_{|+\rangle_s} - [F_{\Phi^+}(256\text{ ns}) - F_{\Phi^+}(0)]_{|0\rangle_s}$ (`> 0` required for `H1`).
     - $C_{\text{mem}}$: $F_{\text{echo}}(N=32, 16.384\,\mu\text{s}) - F_{\text{echo}}(N=1, 16.384\,\mu\text{s})$ (`> 0` required for `H3`).
     - $C_{\text{step48}}$: Step fraction $S_{48} = \frac{F(48\text{ ns})-F(0)}{F(256\text{ ns})-F(0)} \ge 0.75$ (`Apparatus` `24/36 ns` microwave collision).
4. **Mandatory Falsifiable Cluster-Level Held-Out Prediction (Could Fail!):**
   - Parameters fitted on **$G_{\text{train}} = 4$ training couplers** must predict (with **zero refitting** of bath/coupling parameters, using only independent local 1Q $T_1$ and high-frequency echo calibration on target qubits) both the **sign and magnitude** of $\Delta\mathcal{F}_{\text{Bell}}(\Delta)$ across **$G_{\text{test}} = 6$ disjoint held-out couplers** (`>= 5` hops apart, multiplexed simultaneously in the same PUBs; **cluster-level degrees of freedom $df = G_{\text{test}} - 1 = 5$**, never inflating $df$ by temporal blocks).
   - **Kill / Falsification Criterion:** If a candidate model (`H1`, `H2`, or `H3`) wins in-sample $\Delta\mathrm{BIC} \ge 8.0$ on training couplers **but fails to beat `H0` in cluster-level held-out out-of-sample noisy RMSE** (paired $t_5$, $p < 0.05$), the transferable quantum-control claim is **FALSIFIED** and classified as non-portable local calibration heterogeneity.
