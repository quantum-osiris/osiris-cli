# Quantitative Identifiability, Null Calibration & Held-Out Power Study (`CORRELATED_NOISE_01_IDENTIFIABILITY.md`)

**Experiment ID:** `CORRELATED-NOISE-01`  
**Status:** `Simulated` & `Implemented and tested` (`phase2/correlated_noise_01/simulate_correlated_noise.py`, `docs/phase2/correlated_noise_01_identifiability_results.json`)

---

## 1. Identifiability Conditions & Structural Rank

Let $\boldsymbol{\theta}_m$ denote the parameter vector of model $m \in \{\mathrm{H0}, \mathrm{H1}, \mathrm{H2}, \mathrm{H3}, \mathrm{Apparatus}\}$ evaluated over the `dt = 4.0 ns`-aligned 27-condition diagnostic probe suite $\mathcal{D}_{27}$ (`build_diagnostic_probe_suite()`):
1. **Local Rates ($T_{1,u}, T_{1,v}, \gamma_{\phi,u}, \gamma_{\phi,v}, f_0$):** Identified uniquely by $T \in \{2.048, 4.096, 8.192, 16.384\}\,\mu\text{s}$ 1Q echo and 2Q Bell baselines (with $T=0$ SWAP-routed ceilings for $d=2,3$).
2. **Spectator $ZZ$ ($f_{ZZ}$, `H1`):** Uniquely identified by the orthogonal contrast $|+\rangle_s$ vs $|0\rangle_s$ across $\Delta \in \{0, 256\}\text{ ns}$, where `H0`, `H2`, `H3`, and `Apparatus` predict identically zero spectator-state sensitivity.
3. **Markovian Spatial Covariance ($\rho_{\text{spatial}}, \xi$, `H2`):** Uniquely identified by the opposite-sign stagger response on $|\Phi^+\rangle$ (`+`) vs $|\Psi^+\rangle$ (`-`) combined with invariance across $N \in \{1, 8, 16, 32\}$ pulses and exponential decay across $d_{\text{graph}} \in \{1, 2, 3\}$ subject to heavy-hex degree-3 Kossakowski PSD bounds.
4. **Non-Markovian Memory ($\sigma_{\text{OU}}, \tau_c$, `H3`):** Uniquely identified via exact piecewise knot integrals across $N \in \{0, 1, 8, 16, 32\}$ (including explicit $N=32$ at $16.384\,\mu\text{s}$) plus non-exponential Ramsey curvature, whitened via Generalized Least Squares (GLS) against shared pulse-count gain drift (`Bridge E5` $r = 0.812$).
5. **Drive Collision (`Apparatus`):** Uniquely identified by the finite-pulse step ratio $S_{48} = \frac{F(48\text{ ns})-F(0)}{F(256\text{ ns})-F(0)} = 1.00$ (`48 ns > 36 ns` `ibm_marrakesh` / `24 ns` `ibm_fez` `X` envelope) vs $< 0.35$ for `H1–H3`.

---

## 2. Empirical Noise Floor, Exact Student-$t$ Power, and Minimum Detectable Effect (MDE)

From `docs/phase2/correlated_noise_01_identifiability_results.json` ($B = 8$ blocks $\times$ `4,200` shots/basis/block = **`33,600` shots/basis/condition**, incorporating the empirical Heron r2 `1.05 pp` per-block A/A drift plus off-diagonal shared pulse-gain covariance):
- **Per-condition marginal standard error:** $\sigma_{\text{eff}} = \mathbf{0.218\text{ pp}}$ (`0.002181`).
- **Exact finite-block Student-$t_7$ Paired MDE (`80%` power, 5-contrast Holm $\alpha_1 = 0.01$, $df = 7$):**
  $$\mathrm{MDE}_{80\%,\;\alpha=0.01,\;df=7} = \mathbf{1.394\text{ pp}} \quad (\text{vs } 1.054\text{ pp asymptotic normal } z\text{-approximation})$$
  Compare this directly to historical `E4` (`2,048` shots/offset, $\mathrm{MDE} > 3.1\text{ pp}$), explaining why `E4` was flat ($\chi^2_5 = 1.87, p = 0.866$) and why $N \ge 33{,}500$ across $B \ge 8$ interleaved blocks is essential.

---

## 3. Monte Carlo 5-Model GLS Confusion Matrix (`50` Blind Multivariate-Drift Fits)

Using Cholesky-whitened Generalized Least Squares (`build_block_covariance_matrix`) and penalized GLS-BIC selection with the calibrated parsimony gate $\Delta\mathrm{BIC}_{m \text{ vs } H0} \ge 8.0$:

| True Generating Mechanism $\downarrow$ \ Selected Model $\rightarrow$ | `H0` | `H1` | `H2` | `H3` | `Apparatus` | Recall |
|---|---:|---:|---:|---:|---:|---:|
| **`H0` (`IndependentLocal` + Correlated Gain Drift)** | **10** | 0 | 0 | 0 | 0 | **100.0%** (`0%` false `H1/H2/H3`!) |
| **`H1` (`StaticSpectatorZZ`, `28 kHz`)** | 0 | **10** | 0 | 0 | 0 | **100.0%** |
| **`H2` (`CorrelatedMarkovian`, $\rho=0.68$)** | 0 | 0 | **10** | 0 | 0 | **100.0%** |
| **`H3` (`NonMarkovianMemory`, $\tau_c=1.4\,\mu\text{s}$)** | 0 | 0 | 0 | **10** | 0 | **100.0%** |
| **`Apparatus` (`PulseCollision`, `0.90%/pulse`)** | 0 | 0 | 0 | 0 | **10** | **100.0%** |

- **Overall 5-way classification accuracy:** **`100.0%` (`50 / 50`)**.
- **Null specificity:** Under pure `H0` local noise + multivariate `Bridge E5` gain drift, **10 out of 10** realizations selected `H0` (`0%` false positives for `H1`, `H2`, `H3`, or `Apparatus`).
- **Zero-Leakage, Noisy Held-Out Coupler Transfer (`Coupler A -> Unseen Coupler B`):** With independent ground-truth `gamma_white_B` calibrated solely from noisy 1Q `N=32` echo on Coupler B (`parameter_leakage_from_truth = False`) and scored against noisy held-out observations (`evaluated_on_noisy_observations = True`), `H3` achieved noisy out-of-sample $\mathrm{RMSE} = \mathbf{0.00309}$ vs `H0` $\mathrm{RMSE} = \mathbf{0.01705}$ (**5.5× noisy error reduction**; latent curve $\mathrm{RMSE} = \mathbf{0.00068}$ vs $\mathbf{0.01598}$, **23.5× latent error reduction**).
