# Phase 2 Scientific Claims Ledger Corrections (`PHASE2_CLAIM_CORRECTIONS.md`)

**Status:** `Implemented and tested` (`phase2/claims_reconciliation_v2.py`, `docs/phase2/claims_ledger_phase2_proposed.jsonl`, `tests/test_phase2_claims_ledger.py`)  
**Source Baseline Preserved Unchanged:**
- `/home/osiris/claims_ledger.jsonl` (`44` rows; `sha256: 567a1ffd6c425eab3b031d5191ecd3fbd1198b89233bbbe87f5c1ee0fa25fb1f`)
- `/home/osiris/CLAIM_RECONCILIATION.md` (`sha256: 37b6c9c6f9f5dcdc3a7d6ce99db57514dc474840c00e13f3007050132712883a`)
- `/home/osiris/osiris-assessment/work/osiris-cli-phase2/osiris_cli/claims.py` (`sha256: 8563ac0fa78d69b259ea6ad35d498182737f5ac2ddfcd525c5a4a7ad47f47fb1`) — **NOT MODIFIED** pending explicit owner authorization.
- **Missing External Dependency Recorded:** `OSIRIS_Signal_Noise_Forensic_Audit_2026-11-16.md` is absent from `/home/osiris` and all 171 git mirrors (consistent with `CLAIM_RECONCILIATION.md:176`). All Phase 2 analyses were verified directly against primary code, raw archives, and calibration snapshots.

---

## 1. Correction A — `NB-11` (`W2_ROBUSTNESS`): Separation of Hardware Runs from Defective $W_2$ Metric

### 1.1 Forensic Audit of Historical $W_2$ Implementations
Read-only inspection across the entire corpus identified **six distinct implementations** labeled `W2` or `Wasserstein`:
1. **`/home/osiris/w2_robustness_paper.tex` (`sha256: b1386f27...`, Appendix A Eq. 7):**
   $$C(p, q) = \sum_{b \in \mathrm{supp}(p)} \sum_{b' \in \mathrm{supp}(q)} \left(\frac{d_H(b, b')}{Q}\right)^2 p(b) q(b')$$
   - **Verdict:** **DEFECTIVE.** Computes expected squared Hamming distance under independent product sampling $\pi = p \otimes q$, not optimal transport $\inf_{\pi \in \Pi(p,q)}$.
   - **Counterexample (`tests/test_wasserstein_verified.py`):** For $p = q = \frac{1}{2}\delta_0 + \frac{1}{2}\delta_1$ with unit ground cost $c(0,1)=c(1,0)=1$, $C(p,p) = 0.5 \neq 0 = W_2^2(p,p)$. On $Q=27$ uniform distributions, $\sqrt{C(p_{\text{unif}}, p_{\text{unif}})} = \sqrt{\frac{1}{4} + \frac{1}{108}} = \mathbf{0.50918}$ ($\hat{R} = 0.4908$), explaining the paper's $\theta_3 = 3.118$ value ($\hat{R} = 0.494$).
2. **`dnalang-sovereign-sdk-v2.git:lib/metrics/wasserstein.ts` (`sha256: ec3c5a02...`):**
   - `sinkhornDistance` returns raw entropic cost $\sqrt{\langle K \odot C, u \otimes v\rangle}$ without Sinkhorn debiasing, so `sinkhornDistance(p, p) > 0`; `wasserstein1D` omits caller weight normalization; `slicedWasserstein` uses unseeded `Math.random()`.
3. **`Zenodo 18450505` Black-Hole Draft (`ERRATUM_AND_FORENSIC_AUDIT_ZENODO_18450505_W2_BLACK_HOLE.md`):**
   - Mislabeled quantum Hellinger distance $\sqrt{\mathrm{Tr}[(\sqrt{\rho_1}-\sqrt{\rho_2})^2]}$ as $W_2$ and inverted $I_{\text{pres}} = D / D_{\max}$ (`0%` for identity, `100%` for orthogonal destruction).
4. **`copilot-sdk-dnalang` (`omega_engine.py`, `penteract.py`, `aiden.py`, `correlation_analyzer.py`):**
   - Plain Euclidean $\ell_2$ norm or normalized Shannon entropy difference mislabeled `w2_distance`.
5. **`11dCRSM-non-causal-non-local-validation-main` (`qwc_pass_manager.py`):**
   - Standard Qiskit `SabreLayout` + `SabreSwap` labeled "Quantum Wasserstein Compilation ($W_1$)" without computing any transport metric.
6. **`ddpm_markov_sim.py:170` (`discrete_w2`):**
   - Valid 1D two-pointer discrete $W_2$ implementation (used for 1D CTMC diffusion verification only, not 27Q hardware analysis).

### 1.2 Verified Replacement & Proposed Reclassification
- Implemented `osiris_cli/metrics/wasserstein_verified.py` (`exact_wasserstein_lp` via HiGHS Kantorovich LP, `exact_wasserstein_1d`, `sinkhorn_wasserstein_bounded` with certified Altschuler primal + c-transform dual error bounds, and `sliced_wasserstein_hypercube`), verified on identity, symmetry, triangle inequality, and closed-form analytic distributions (`tests/test_wasserstein_verified.py`).
- Raw count files for the six Dec 23, 2025 `ibm_fez` jobs (`d554h...`) are not in `/home/osiris`, so historical scores are **not** recomputed from guessed counts.
- **Proposed `NB-11` Split Classification:**
  - Historical 6-job `ibm_fez` execution: `HARDWARE-OBSERVED / OWNER-REPORTED`
  - $W_2$ robustness & entropy-correlation conclusion: `QUARANTINED_PENDING_RECOMPUTATION` (also requires split-sample multinomial debiasing per `docs/phase2/W2_MATHEMATICAL_ERRATUM.md`).

---

## 2. Correction B — `NB-07` (`F_MAX`): Separating Gate Fidelity from Like-for-Like Bell-State Preparation Fidelity

### 2.1 Quantity Bounded by `NB-07` / `F_MAX`
The claim $F_{\max} = 1 - \varphi^{-8} \approx 0.978701$ asserts a universal upper bound on **two-qubit Bell-state preparation fidelity**:
$$\mathcal{F}_{\text{Bell}}(\rho_{\text{exp}}) = \langle \Phi^+ | \rho_{\text{exp}} | \Phi^+ \rangle = \frac{1}{4}\left(P_{00} + P_{11} + \mathcal{C}_{\text{parity}}\right) = \frac{1 + \langle XX\rangle - \langle YY\rangle + \langle ZZ\rangle}{4}$$
Phase 1 (`CLAIM_RECONCILIATION.md` L112–114; `NB-07`) cited Ballance et al. (2016) `99.9(1)%` two-qubit **gate** fidelity. Randomized-benchmarking (RB / interleaved RB) average two-qubit gate fidelity isolates single-gate error from state preparation and measurement (SPAM) and is **not** interchangeable with $\mathcal{F}_{\text{Bell}}$.

### 2.2 Like-for-Like Bell-State Preparation Experiments Comparing State Prep, Tomography, SPAM & Uncertainty

| Experiment | Platform | State Preparation | Tomography / Estimator | SPAM Correction? | Reported $\mathcal{F}_{\text{Bell}}$ | Excess Over `0.978701` |
|---|---|---|---|---|---|---|
| **Benhelm et al.** (*Nat. Phys.* 4, 463, 2008) | $^{40}\text{Ca}^+$ (2 ions) | Single $100\,\mu\text{s}$ bichromatic Mølmer–Sørensen entangling pulse | Direct $P_{00}+P_{11}$ fluorescence + parity oscillation contrast | **No** (raw end-to-end state prep + readout) | **$0.993 \pm 0.001$** | **`+1.43 pp` (`+14.3σ`)** |
| **Gaebler et al.** (*PRL* 117, 060505, 2016) | $^9\text{Be}^+$ (2 ions) | Deterministic Mølmer–Sørensen Bell preparation | Direct population + parity oscillation (`0.9991 ± 0.0004` SPAM-deducted) | **No** (`0.9955` raw) / Yes (`0.9991` SPAM-deducted) | **$0.9955 \pm 0.0005$ (raw)** | **`+1.68 pp` (`+33.6σ`)** |
| **Ballance et al.** (*PRL* 117, 060504, 2016) | $^{43}\text{Ca}^+$ hyperfine clock | Raman geometric-phase gate preparing Bell state | Population + parity contrast (SPAM characterized at $1.7 \times 10^{-3}$) | Raw `~0.9973`; SPAM-corrected `0.999(1)` | **$0.9973 \pm 0.0010$ (raw)** | **`+1.86 pp` (`+18.6σ`)** |
| **Clark et al.** (*PRL* 127, 130505, 2021) | $^{40}\text{Ca}^+$ (2 ions) | Mølmer–Sørensen entangling gate | Direct population + parity oscillation **without** SPAM subtraction | **No** (raw state fidelity) | **$0.9994 \pm 0.0003$** | **`+2.07 pp` (`+69.0σ`)** |
| **Ding et al.** (*PRX* 13, 031035, 2023) | Superconducting fluxonium (2Q) | Microwave-activated `CZ` + single-qubit $\pi/2$ | Quantum state tomography (`>0.992` SPAM-mitigated) | **No** (`0.985` raw) | **$0.985 \pm 0.002$ (raw)** | **`+0.63 pp` (`+3.1σ`)** |
| *Contrast (Gate RB only, NOT like-for-like):* **Kandala et al.** (*PRL* 127, 130501, 2021) | Fixed-frequency transmon | Interleaved 2Q Clifford RB | Exponential RB sequence decay (SPAM-insensitive) | SPAM-free by RB design | $F_{\text{gate}} = 0.9977(2)$ | *Excluded from direct $\mathcal{F}_{\text{Bell}}$ test* |

**Conclusion for `NB-07` / `F_MAX`:** Even restricting strictly to **raw, non-SPAM-subtracted Bell-state preparation fidelity**, independent trapped-ion and superconducting experiments exceed $0.978701$ by $3.1\sigma$ to $69\sigma$. Meanwhile, OSIRIS's 189 IBM transmon runs had maximum $\mathcal{F}_{\text{Bell}} = 0.9773$ (`E0` on `ibm_marrakesh` `(14,15)` reached `0.9730`), where typical Heron r2 readout + 1Q/2Q error is $\sim 2\text{–}3\%$, giving zero statistical sensitivity to a $0.9787$ ceiling. `NB-07` remains `REJECTED` (`REFUTED`) on strictly like-for-like grounds.

---

## 3. Correction C — Audit of Every `MEASURED` and `VERIFIED` Row + Structured Ingestion Fixes

Every one of the `7` Phase-1 `MEASURED` rows and `5` Phase-1 `VERIFIED` rows in `claims_ledger.jsonl` was audited across all five mandatory dimensions in `docs/phase2/claims_ledger_phase2_proposed.jsonl`:

| ID | Phase 1 Status | Phase 2 Proposed Status | Direct vs Derived | Raw Data Access | Math vs Hardware | Replication & Key Limitation |
|---|---|---|---|---|---|---|
| `NB-11` | `MEASURED` | `QUARANTINED_PENDING_RECOMPUTATION` (runs: `HARDWARE-OBSERVED / OWNER-REPORTED`) | Derived metric from counts | Owner-reported table only | Hardware reported; **math defective** | Single chip (`R=2`); Eq. 7 violates $W_2(p,p)=0$. |
| `NB-14` | `MEASURED` | `HARDWARE-OBSERVED / OWNER-REPORTED` | Derived $\Delta F$ from counts | Owner-reported (`zenodo.22855102`, raw counts not local) | Hardware observed + compiler math verified (`RZ` commutes with `CX`) | Null result ($\Delta F = +0.002 \pm 0.006$); exact Z-commutation verified mathematically. |
| `NB-15` | `MEASURED` | `HARDWARE-OBSERVED / OWNER-REPORTED` | Derived $|+\rangle$ survival | Owner-reported for `ibm_fez`; local raw data for `ibm_marrakesh` (`dau0q3qh...`) | Hardware observed on specific chains | Structured points `[8us: 0.946, 16us: 0.888, 32us: 0.795]` vs `32us` sync `0.59` / none `0.55`; prior art Tripathi et al. (2022); differs from `NB-17` (`0.944/0.906/0.832`). |
| `NB-17` | `MEASURED` | `HARDWARE-OBSERVED / OWNER-REPORTED` | Derived $|+\rangle$ survival | Owner-reported (`zenodo.22870287`, raw counts not local) | Hardware observed | GA matched staggered XY4 within $\pm 0.006$ (null on GA superiority); job IDs needed to reconcile with `NB-15`. |
| `NB-27` | `MEASURED` | `HARDWARE-OBSERVED / OWNER-REPORTED` | Derived XEB from counts | Owner-reported (`ibm_torino`, no local raw counts) | Hardware reported | 12Q XEB $>0.1$ is trivially classically simulable; not quantum advantage. |
| `NB-28` | `MEASURED` | `HARDWARE-OBSERVED / OWNER-REPORTED` | Derived plug-in entropy | Owner-reported (no local raw counts) | Hardware reported | Ordinary decoherence entropy growth ($3.2 \to 4.8$ bits). |
| `NB-37` | `MEASURED` | `HARDWARE-OBSERVED / OWNER-REPORTED` | Derived connected correlator $C_{0k}(d)$ | Owner-reported (no local circuits/counts) | Hardware reported | Null result ($Z < 5$ outside light cone); lacks minimum-detectable-coupling power curve. |
| `NB-38`..`NB-42` | `VERIFIED` | `MATHEMATICALLY_VERIFIED_ONLY` | Derived analytic identities | Reproducible locally (`build_ledger.py`) | **Mathematical verification ONLY** (zero hardware measurement) | Standard regular-tetrahedron geometry & $\lVert A\rVert_* / \lVert A\rVert_F \le \sqrt{\mathrm{rank}(A)}$ Cauchy–Schwarz inequality; confers zero physical validation to CRSM. |

### Additional Disambiguation Enforced (`osiris_cli/workflows/review_coverage.py`)
- **`STAGGERED_DD` Duration Fix:** Stored as structured `points` (`8 µs: 0.946`, `16 µs: 0.888`, `32 µs: 0.795`) with comparator locked strictly to `duration_us: 32` (`simultaneous_approx: 0.59`, `no_dd_approx: 0.55`), preventing `0.946 vs 0.888` from ever being misread as an `8 µs` treatment-vs-control contrast.
- **`Tesseract` Namespace Separation:** `TESSERACT_COLOR_CODE_DECODER_EXT` (`arXiv:2610.24077v1` A* syndrome decoder), `ERROR_SUPPRESSION_1E6` (refuted 40-qubit OSIRIS circuit), and `TESSERACT_MAP_02` (Phase 2 compilation experiment) are assigned strictly disjoint identifiers with zero cross-linking of evidence.
