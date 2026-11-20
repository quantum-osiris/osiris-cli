# PWS Section 1.2.6 (Outcome 6) & Phase 3 Evidence Consolidation

**Record Standing:** `CANDIDATE_REPORTED` (proposed for human adjudication on Node Beta per Gate 0 `GEMINI.md`; no claim, value, or threshold is marked `"verified"` here).

---

## 1. Proposed Addition to Section 1.2.6

### 1.2.6 Outcome 6: Reproducible Evaluation and Platform Acceptance

The customer shall be able to evaluate candidate computational methods against defined reference implementations and determine whether observed results satisfy acceptance criteria established before model training or experimental execution.

The delivered capability (`osiris_cli/reproducible_eval.py`) shall:

1. **Partition Isolation (`1.2.6-1`):** Separate training, tuning, and test data, with learned preprocessing fitted only within the applicable training partition.
2. **Reference Comparison (`1.2.6-2`):** Compare candidate methods against approved reference methods using documented datasets, configurations, computational budgets, and evaluation metrics.
3. **Pre-Recorded Decision Rule (`1.2.6-3`):** Record the decision rule before training or execution and report whether the observed result satisfies that rule.
4. **Artifact & Provenance Integrity (`1.2.6-4`):** Preserve configurations, generated artifacts, evaluation outputs, and hash-chained integrity records sufficient to inspect and repeat the evaluation within documented tolerances.
5. **Immutable Outcome Reporting (`1.2.6-5`):** Report negative, inconclusive, and favorable results without changing the acceptance rule after observing outcomes.
6. **Plane & Data Scope Classification (`1.2.6-6`):** Distinguish synthetic-data results from real-data results, and simulation results from physical-hardware observations.
7. **Limitation & Superseded-Run Disclosure (`1.2.6-7`):** Identify untested interfaces, omitted comparisons, superseded runs, and other limitations affecting interpretation.

Successful performance shall be demonstrated by executing an approved benchmark protocol and delivering an auditable evidence package. **A candidate method's failure to outperform the reference methods shall not constitute contractor nonperformance when the required evaluation has been completed correctly and its limitations have been disclosed.**

---

## 2. Candidate Evidence Register Entries

### 2.1 Quantum-Kernel Genomics Benchmark (`dnalang.qkernel`)

| Field | Proposed Record |
| :--- | :--- |
| **Record standing** | `CANDIDATE_REPORTED` (supports Outcome 6 & simulator portion of Outcome 3; not a quantum-advantage claim) |
| **Capability supported** | Reproducible comparative evaluation of quantum-kernel and classical classifiers |
| **Implementation** | `dnalang.qkernel` in `dnalang-core` (`osiris_cli/reproducible_eval.py` protocol inspector) |
| **Reported verification** | Local `dnalang-core` pytest suite passing (including LD pruning, kinship-aware splits, RF/GB baselines, and probability calibration); candidate record awaiting human commit |
| **Evaluation scope** | Two synthetic Balding-Nichols genotype settings: `4` populations of `50` individuals (`n = 200`), `2,000` SNPs, `6` simulated qubits, `2` feature-map repetitions, `5` outer folds and `3` inner folds |
| **Data origin / Execution plane** | `SYNTHETIC` / `SIMULATOR` |
| **Candidate methods** | `product` (no entanglement negative control), `nn` (nearest-neighbor CZ chain), and Havlicek `zz` feature-map fidelity kernels; exact statevector and finite-measurement evaluations (`128`, `512`, `2,048`, `8,192` shots) with eigenvalue-clipping PSD repair |
| **Reference methods** | Linear, RBF, and degree-3 polynomial SVMs; multinomial logistic regression; `k`-nearest-neighbor; matched-dimension (`2^n`) classical random-Fourier-feature control; plus pure-NumPy Random Forest and Gradient Boosting baselines added in `dnalang/qkernel/baselines.py` |
| **Recorded success criterion** | Written to `ledger.jsonl` (`qkernel.preregister`, index `0`) before fitting: `H0` rejected only if best quantum minus best classical balanced accuracy exceeds `0.02` with the entire `95%` paired-bootstrap confidence interval above `0.02` |
| **Hard-setting result (`F_ST = 0.004`)** | Best classical (`logistic`): `0.595`. Best quantum (`product`, no entanglement): `0.545`. Difference `95%` paired-bootstrap CI: `-0.117` to `+0.018`. `H0` not rejected (criterion not met) |
| **Easy-setting result (`F_ST = 0.02`)** | Best classical (`linear_svm`, `rbf_svm`, `logistic`) and best quantum (`product`, `nn`) tied at `1.000`. `H0` not rejected (criterion not met) |
| **Kernel concentration diagnostic** | `zz` was weakest in both settings (mean off-diagonal fidelity `~0.06`); reported as a descriptive concentration indicator, not causal proof |
| **Evidence outputs** | `report.md`, `results.json`, OpenQASM 3 circuits (`circuits/*.qasm`), saved raw & PSD-repaired kernel archives (`kernels/*.npz`), and hash-chained `ledger.jsonl` |
| **Explicit exclusions** | No physical QPU execution; no completed real-VCF (`1000 Genomes`) evaluation; no clinical or biological inference |
| **Newly closed code items & remaining limits** | Sliding-window `r^2` LD pruning, GRM kinship-aware group splits, Random Forest / Gradient Boosting baselines, and cross-validated probability calibration (`log_loss`, `brier_score`, `ece`) are now implemented and unit-tested in `dnalang.qkernel`; real `1000 Genomes` VCF execution and physical QPU execution remain unrun |
| **Superseded evidence** | Initial `test-runs/qkernel/synthetic-fst0.004/` directory (signed binary KTA diagnostic on 4 classes) is superseded by `v2-synthetic-fst0.004` and `v2-synthetic-fst0.02` and excluded |

---

### 2.2 `TESSERACT-MAP-02` Controlled 40-Qubit Compilation Benchmark (`ibm_marrakesh`)

| Field | Proposed Record |
| :--- | :--- |
| **Record standing** | `CANDIDATE_REPORTED` (`Hardware-observed`, `ibm_marrakesh` job `dbto408n26qs739tnrj0`) |
| **Execution scope** | `10` randomized interleaved blocks x `9` circuits = `90` PUBs x `2,048` shots (`184,320` total shots, `53` s QPU charged; Amendment A-1 sample sizes; lock `sha256: 5f4a04c45933e046...`, QPY `sha256: 409dc7365405ed20...`) |
| **Data origin / Execution plane** | `REAL_OBSERVATIONAL` / `PHYSICAL_HARDWARE` |
| **Primary contrasts (`F_cl`, paired `t`, `df = 9`, Holm-corrected)** | (1) `C5 - C2` (degree-2 chain rewrite + SWAP removal at opt `3`): `+14.27` pp (`95%` CI `[+13.52, +15.01]`, Holm `p = 4.6e-11`); (2) `C4 - C3` (rewrite + SWAP removal at opt `1`): `+12.47` pp (`[+11.52, +13.42]`, Holm `p = 8.4e-10`); (3) `C5 - C6` (greedy best vs median path at `39` CZ): `+10.54` pp (`[+9.95, +11.12]`, Holm `p = 6.2e-11`); (4) `C5 - C4` (virtual-Z A/A drift control): `-0.24` pp (`[-0.88, +0.41]`, Holm `p = 0.43`); (5) `C5 - C7` (greedy best path `0.2150` vs stock `VF2Layout` `0.1580`): `+5.70` pp (`[+5.13, +6.28]`, Holm `p = 6.6e-09`) |
| **Adverse / divergent finding preserved** | FakeMarrakesh surrogate simulation predicted `VF2Layout` (`C7`, `0.4404`) would beat the greedy best path (`C5`, `0.4054`); physical hardware showed the opposite (`C7` scored `5.70` pp below `C5`). Observed absolute `F_cl` on `C5` (`0.2150`) was roughly half the surrogate prediction (`0.4054`), with `22.3%` and `12.1%` union-SPAM assignment errors observed on qubits `42` and `41` |
| **Explicit exclusions** | Demonstrates compilation and physical-layout sensitivity only; does **not** demonstrate genomics classification or computational quantum advantage |

---

### 2.3 `CORRELATED-NOISE-01 Stage 1` Multiplexed 10-Coupler Staggered-DD Benchmark (`ibm_marrakesh`)

| Field | Proposed Record |
| :--- | :--- |
| **Record standing** | `CANDIDATE_REPORTED` (`Hardware-observed`, `ibm_marrakesh` job `dbtp67dav06s73cfbsm0`) |
| **Execution scope** | `244` PUBs x `2,048` shots (`499,712` circuit shots = `4,997,120` multiplexed coupler-shots across `10` disjoint `>= 5`-hop couplers, `5` blocks + `4` readout-calibration PUBs, `139` s QPU charged; lock `sha256: 38ebfe7f7c007633...`, QPY `sha256: 158de03260c429c8...`) |
| **Data origin / Execution plane** | `REAL_OBSERVATIONAL` / `PHYSICAL_HARDWARE` |
| **Mechanistic contrasts (`10` couplers, `df = 9`, Holm-corrected)** | `C_DFS = +0.43` pp (`95%` CI `-0.13` to `+1.00` pp, Holm `p = 0.47`); `C_spec = -0.20` pp (`[-0.50, +0.11]`, Holm `p = 0.53`); `C_stagger = +0.20` pp (`[-0.22, +0.62]`, Holm `p = 0.61`); `C_mem = -0.14` pp (`[-2.25, +1.97]`, Holm `p = 0.89`); step fraction `S48 = 0.6154 < 0.75` |
| **Locked analysis outcome** | Selected `H3` in-sample on the `4` training couplers over white `H0` (`delta BIC = 10,262.849`, attributable to colored local Ramsey decay vs white `H0`), but failed `6`-coupler held-out validation (`mean_rmse_h3 = 0.14329` vs `mean_rmse_h0 = 0.13023`, paired `t = -0.238`, `p = 0.8214`, `transfer_beats_h0_p_0_05 = false`), firing the preregistered kill criterion |
| **Amendment A-3 outcome** | Seeded `19`-condition echo/DD fit with local colored nulls (`H0_1f`, `H3_local`) selected local `H0` (`BIC = 115.05` vs best non-local `H1` `BIC = 115.27`; `delta BIC < 8` gate). Outcome: **local noise only** |
| **Stage-2 gate disposition** | Closed under both analyses; Stage 2 on `ibm_fez` does not run. Bounds fast correlated dephasing (`< 512` ns `DD16` slot), not slow (`> 1` us) correlated noise refocused by `DD16` |

---

## 3. Technical Capabilities Annex Language & Scope Matrix

### Approved Accomplishment Wording
> Implemented and executed a simulator-based quantum-kernel evaluation package (`dnalang.qkernel`) comparing three circuit feature maps against classical classifiers on two synthetic genotype datasets, alongside governed physical-QPU compilation (`TESSERACT-MAP-02`) and noise-diagnostics (`CORRELATED-NOISE-01 Stage 1`) campaigns on `ibm_marrakesh`. The workflows generated structured reports, kernel/count archives, OpenQASM/QPY circuit artifacts, and hash-chained execution records. Neither synthetic genomics setting satisfied the recorded quantum-advantage criterion, and `CORRELATED-NOISE-01 Stage 1` closed its Stage-2 gate after falsifying transferable non-local noise claims. Real-VCF genomics evaluation and physical-QPU genomics execution remain outside the demonstrated scope.

### Prohibited Overclaims (Enforced by `osiris_cli.reproducible_eval.check_annex_language`)
- Do **not** describe `dnalang.qkernel` as *validated quantum advantage for genomics*, *demonstrated real-world population classification*, *physical quantum-computing validation*, *proof that entanglement provides no benefit generally*, or *confirmation that kernel concentration caused the observed performance*.

### Complementary Evidence Matrix

| Evidence Package | Plane / Data | What It Demonstrates | What It Does Not Demonstrate |
| :--- | :--- | :--- | :--- |
| `TESSERACT-MAP-02` (`dbto408n26qs739tnrj0`) | `PHYSICAL_HARDWARE` / `REAL_OBSERVATIONAL` | Physical-QPU compilation & layout comparisons with preregistered Holm-corrected block statistics (including where hardware contradicts surrogate `VF2Layout` rankings) | Genomics classification performance or computational quantum advantage over classical classifiers |
| `CORRELATED-NOISE-01 Stage 1` (`dbtp67dav06s73cfbsm0`) | `PHYSICAL_HARDWARE` / `REAL_OBSERVATIONAL` | Physical-QPU 10-coupler staggered-DD model selection & held-out falsification closing the Stage-2 hardware gate | Transferable spatial noise memory, slow (`> 1` us) correlated noise, or quantum advantage |
| `dnalang.qkernel` Genomics Benchmark | `SIMULATOR` / `SYNTHETIC` | Leakage-safe comparative evaluation against tuned classical baselines with hash-chained provenance and an immutable null (`H0` not rejected) verdict | Physical-QPU performance, real-world clinical/population validity, or a quantum advantage |
