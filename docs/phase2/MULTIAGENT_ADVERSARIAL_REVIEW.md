# Phase 2 Multiagent Adversarial Review Record (`MULTIAGENT_ADVERSARIAL_REVIEW.md`)

**Status:** `Implemented and tested` (all 7 independent read-only adversarial review passes completed prior to freezing final preregistrations; explicit reviewer disagreements preserved verbatim without artificial consensus).

---

## 1. Independent Review Roster & Transcript Provenance

Per Phase 2 Section 9, seven separate `research` subagents executed read-only forensic, mathematical, hardware, statistical, simulator, governance, and architectural audits across the workspace:

| Role | Subagent Conversation ID | Verdict on Initial Draft | Primary Focus |
|---|---|---|---|
| **Physics Theorist** | `fafcdbbf-77ba-4b37-bb0e-d9727449ccb9` | `CONDITIONAL PASS` | Open-system Hamiltonians, CPTP/Kossakowski positivity, `H3` quadrature aliasing, `TAU-REVIVAL-01` frame/sign math |
| **Quantum Hardware Engineer** | `9ebacff4-f961-4733-a5b2-fdfb7bbe623b` | `TESSERACT-MAP-02: CONDITIONAL PASS`; `CORRELATED-NOISE-01: BLOCK -> RESOLVED` | IBM Heron r2 `dt = 4.0 ns`, `24/36 ns` `SX/X` widths, `3.14/5.30 µs` reset latency, `>=5`-hop multiplexing, `<600 s` OpenPlan quota |
| **Statistician** | `a7418f8b-15f6-4e9e-a771-d0c28ba91db2` | `CONDITIONAL PASS` | Intra-block gain covariance (`Bridge E5` $r=0.812$), GLS vs WLS, held-out parameter leakage, coupler-cluster $df$, Student-$t$ MDE |
| **Simulator Engineer** | `fdf61c59-71c4-4acc-bb63-a7c515cc3b4c` | `CONDITIONAL PASS` | `qByte` unitarity/endianness, `QbyteRuntime(strict_l0=True)` crash, sparse Kantorovich LP, $>64\text{Q}$ Hamming, 40Q $T_1/T_2$ surrogate |
| **Adversarial Reviewer** | `d0e41f23-3be1-415e-9775-0888d501a0ee` | `CRITICAL OBJECTIONS -> RESOLVED` | Classical shared-LO mimics, prior art (Tripathi 2022), stock `VF2Layout` credit, modulo-wrap boundary bug, `NB-11` disposition |
| **Research Governance Auditor** | `5e31169d-7744-4f02-9998-27647387dbc6` | `CONDITIONAL PASS` | SHA-256 immutability (`99/103` untouched + `4` `qbyte_system` repairs), zero QPU spend, credential hygiene, canonical 6-label coverage |
| **Product Architect** | `2f94bcce-f95e-45c8-918f-24f0665c09e7` | `CONDITIONAL PASS` | Reusable compiler pass (`fanout_degree_reducer.py`), split-sample OT debiasing, `CCCE` technical debt, QPU quota allocation |

---

## 2. Role-by-Role Technical Findings & Pre-Preregistration Repairs Applied

### 2.1 Physics Theorist (`fafcdbbf`)
- **Critiques Raised:**
  1. `H3NonMarkovianMemoryModel` used an 80-point uniform Riemann grid (`dt = 100 ns` at `8 µs`), causing offsets `{16, 32, 48, 80} ns` to evaluate identical `sv` arrays (quadrature aliasing).
  2. `H1StaticSpectatorZZModel` failed to refocus 1Q `ramsey_echo_u` (`n_dd >= 1`) and placed `0.35` inside $\cos(0.35\omega_{ZZ}T\alpha)$ rather than as mixture visibility; `THEORY.md` omitted $\frac{1}{2}$ on $\sum_{a\neq b}\gamma_{\phi,ab}$.
  3. Pairwise $|\rho_{\text{sp}}| \le 1$ is insufficient for multi-qubit CPTP positivity on heavy-hex degree-3 hubs ($3\rho^2 - 2\rho e^{-1/\xi} - 1 \le 0$).
  4. Bare-idle $T_1$ populations created a `+0.27 pp` $|\Phi^+\rangle$/$|\Psi^+\rangle$ split during balanced $N=16$ DD; $d=2,3$ pairs lacked per-hop SWAP SPAM baselines.
  5. `TAU-REVIVAL-01` real-quadrature cosine fits measure $|\Delta f_0 + f_{\text{inj}}|$ (V-folding when $\Delta f_0 < 0$ or $|\Delta f_0| < 15\text{ kHz}$), and virtual-$Z$ rotates any $\tau_0$ peak out of $\langle XX - YY\rangle$.
- **Pre-Preregistration Repairs Applied:**
  - Replaced `H3` Riemann quadrature with **exact closed-form piecewise analytical integration** over sorted pulse-flip knots (`_exact_piecewise_kernel_integrals`, verified to $3 \times 10^{-16}$ against `scipy.integrate.dblquad`), fixed `H1` 1Q-echo refocusing + visibility mixture, enforced heavy-hex degree-3 Kossakowski PSD bounds (`verify_heavy_hex_psd_covariance`, `clamp_heavy_hex_psd_rho`), symmetrized DD $T_1$ parity decay, added `swap_fidelity_per_hop`, and upgraded `TAU-REVIVAL-01` to conjugate-quadrature complex coherence $C_{\text{complex}}(\tau) = \frac{1}{2}(\langle XX - YY\rangle + i\langle XY + YX\rangle)$ + sublattice-selective echo.

### 2.2 Quantum Hardware Engineer (`9ebacff4`)
- **Critiques Raised:**
  1. `build_diagnostic_probe_suite()` hardcoded `250 ns` (`62.5 dt`, illegal on Heron r2 `dt = 4.0 ns`) while `HARDWARE_PLAN.md` retained draft text (`"wait: note 250 ns..."`), and assumed `32 ns` pulses instead of calibrated **`36 ns` (`9 dt`) on `ibm_marrakesh`** and **`24 ns` (`6 dt`) on `ibm_fez`**.
  2. Proposed couplers `(24,25)` and `(44,45)` were only **2 hops apart** (`25–37–45`) on `ibm_marrakesh`, violating the `>3`-hop multiplexing rule, and Stage 1 (`13–15 min` with held-out couplers) breached IBM's `10 min` (`600 s`) OpenPlan cap.
  3. In `TESSERACT-MAP-02`, `C5` (`depth 66`) and `C4` (`depth 86`) compile to **identical physical schedules (`81 sx, 39 cz`, `4,772 ns`)**, differing only by 40 zero-duration virtual-`RZ` updates; `C0_SPAM` omitted non-`best_path` qubits.
  4. `Aeterna` `HERON_R2_MEASURE_RESET_NS = 1560 ns` understated calibrated `measure + reset` (`3,144 ns` on `ibm_fez`; `5,304 ns` on `ibm_marrakesh`) and 2-`SX` `RX(θ)` synthesis (`48–72 ns`).
- **Pre-Preregistration Repairs Applied:**
  - Locked `T = 8.192 µs` (`2,048 dt`) and `{0, 16, 32, 48, 80, 160, 256} ns` (`{0, 4, 8, 12, 20, 40, 64} dt`) across all code/docs, added `build_isa_staggered_dd_bell_circuit()` with `36 ns`/`24 ns` pulse widths, programmatically selected **10 verified `>= 5`-hop disjoint couplers** on `ibm_marrakesh` (`4` train + `6` held-out multiplexed in ONE PUB $\implies$ **`5.4 min < 10 min` OpenPlan cap**), reclassified `C5 vs C4` as a Virtual-$Z$ A/A Null Sentinel, added `build_union_spam_calibration_circuits()`, and updated `aeterna_controls_repaired.py` with exact `3,144 ns`/`5,304 ns` reset latencies and 2-`SX` step durations.

### 2.3 Statistician (`a7418f8b`)
- **Critiques Raised:**
  1. Diagonal WLS ignored off-diagonal intra-block gain covariance (`Bridge E5` $r = 0.812$) across pulse counts $N \in \{0, 1, 8, 16, 32\}$; asymptotic $z_{0.995} = 2.58$ understated $B=8$ Student-$t_7$ MDE (`3.50`) by ~31%.
  2. `simulate_correlated_noise.py` copied `inferred_h3.gamma_white` into `true_heldout_B` and evaluated `y_B_true` with zero observation noise (`0.00010` RMSE = `22×` below $\sigma_{\text{eff}}$).
  3. Coupler-transfer inference has cluster degrees of freedom $df = G_{\text{test}} - 1$, not temporal block $df = B - 1$; `TESSERACT-MAP-02` needed pre-lock path-selection separation; `color_code_eval.py` and `coaxial_t1_reassessment.py` needed batch-clustered SEs and DerSimonian–Laird random-effects pooling.
- **Pre-Preregistration Repairs Applied:**
  - Implemented structured block-covariance **Generalized Least Squares (GLS)** (`build_block_covariance_matrix`, $\Delta\text{BIC}_{\text{GLS}} \ge 8.0$), removed all parameter leakage in `true_heldout_B` and evaluated against noisy `y_B_obs`, upgraded MDE to exact non-central $t_7$, expanded held-out cohort to $G_{\text{test}} = 6$ couplers ($df = 5$), enforced pre-lock calibration snapshot separation in `TESSERACT-MAP-02`, added `batch_cluster_se_failure_rate_diff` in `color_code_eval.py`, and added DerSimonian–Laird random-effects meta-analysis in `coaxial_t1_reassessment.py`.

### 2.4 Simulator Engineer (`fdf61c59`)
- **Critiques Raised:**
  1. `Qbyte` set $\Gamma = 0.092(1 + 0.001\,\text{ops}) = 0.09209 > 0.092$ on gate #1, causing default `QbyteRuntime(strict_l0=True)` to crash deterministically on any 1-gate kernel; `_compute_phi` could go slightly negative (`-3.6e-16`); `SovereignExecutor.run` ignored explicit `measure(q, c)` wire subsets.
  2. `exact_wasserstein_lp` allocated a dense `(m+n-1)×(m*n)` constraint matrix (`17.2 GB` at $m=n=1024$), and `hamming_cost_matrix` overflowed `uint64` above 64Q.
  3. `simulate_40q_hardware_block_outcomes` omitted $T_1/T_2$ idle decay and emitted `"noise_..."` strings instead of 40-bit binary corrupted bitstrings.
- **Pre-Preregistration Repairs Applied:**
  - Aligned `Qbyte._update_metrics` (`0.0005/op`) with `L0FormalProver` (verified via `test_strict_l0_runtime_and_subset_measurement_and_endian_counts`), clamped `entropy >= 0.0`, disabled `enable_ccce` by default in `SovereignExecutor`, implemented `measure(q, c)` subset mapping + `get_counts(endian="little")`, converted `exact_wasserstein_lp` to `scipy.sparse.csc_matrix`, extended `hamming_cost_matrix` to arbitrary $>64\text{Q}$ widths, and added physical gate-duration + idle-$T_1/T_2$ + 40-bit binary domain-wall/readout corruption in `tesseract_map_02.py`.

### 2.5 Adversarial Reviewer (`d0e41f23`), Governance Auditor (`5e31169d`), & Product Architect (`2f94bcce`)
- **Additional Repairs Applied:**
  - Fixed modulo-wrapped pulse boundaries (`pulse_centers_s`) and dynamic `argmax` optimal-offset prediction in `CORRELATED-NOISE-01`; added classical shared-LO ($d \ge 6$) and $4\times 4$ readout-crosstalk controls; cited Tripathi et al. (2022), Paz-Silva et al. (2017), Sung et al. (2021), and Nation & Treinish (2023); added `debiased_empirical_wasserstein()` to `wasserstein_verified.py`; enforced `deadline_us` in `color_code_eval.py`; fixed `/home/osiris/.../claims.py` path and `Ballance 2016` raw vs SPAM-adjusted fields; and added full canonical 6-label status accounting.

---

## 3. Explicit Unresolved Reviewer Disagreements (Preserved Without Artificial Consensus)

Per Section 9 (*"Reviewers must identify disagreements explicitly. Do not synthesize artificial consensus"*), three substantive strategic and taxonomic disagreements remain across the panel and are escalated transparently to the Principal Investigator:

1. **QPU Quota Allocation: `CORRELATED-NOISE-01` vs `TESSERACT-MAP-02` First?**
   - **Product Architect (`2f94bcce`) & Adversarial Reviewer (`d0e41f23`):** Recommend allocating initial IBM OpenPlan QPU quota **exclusively to `TESSERACT-MAP-02` (`~2.8 min`)**, arguing that `reduce_commuting_cx_fanouts` + `VF2Layout` delivers a large, deterministic `163 -> 39 CZ` (`+24–30 pp` fidelity) engineering gain, whereas sub-percent heavy-hex DD stagger effects risk being dominated by fast shared-LO/AWG drift and prior art (Tripathi 2022).
   - **Physics Theorist (`fafcdbbf`) & PI Research Directive:** Prioritize **`CORRELATED-NOISE-01`** (`5.4 min` compact multiplexed Stage 1) as the primary novel-physics characterization experiment capable of testing 8-axis model discrimination (`H0` vs `H1` vs `H2` vs `H3` vs `Apparatus`) and zero-retraining cross-coupler transfer, which cannot be answered by compilation benchmarks alone.
   - **Statistician (`a7418f8b`) & Hardware Engineer (`9ebacff4`):** Approve running **both** (`2.8 min + 5.4 min = 8.2 min < 10.0 min` monthly OpenPlan budget!) *provided* `CORRELATED-NOISE-01` uses the repaired 10-coupler `>=5`-hop multiplexed PUB layout and GLS covariance estimator.

2. **Taxonomic Disposition of `NB-11` (`W2_ROBUSTNESS`):**
   - **Adversarial Reviewer (`d0e41f23`):** Argues `NB-11` should be classified directly as **`REFUTED` / `ARTIFACT`**, because Z-basis phase cancellation (`W2_MATHEMATICAL_ERRATUM.md` Defects #2–3) eliminates post-Layer-0 quantum phase interference regardless of how raw `ibm_fez` bitstrings are rescored.
   - **Research Governance Auditor (`5e31169d`) & Phase 2 Directive Sec 2A:** Require preserving the strict split between historical hardware execution (**`HARDWARE-OBSERVED / OWNER-REPORTED`**) and the manuscript's $W_2$ robustness conclusion (**`QUARANTINED_PENDING_RECOMPUTATION`**, pending owner retrieval of the 6 `d554h...` count histograms or owner sign-off to close as `REFUTED`). Both positions are recorded in `CLAIM_CHANGESET_FOR_OWNER_APPROVAL.md`.

3. **Future of `.dna` / `CCCE` Runtime Surface (`qbyte_system` & `osiris/compiler`):**
   - **Product Architect (`2f94bcce`) & Simulator Engineer (`fdf61c59`):** Recommend deprecating biological gate verbs (`helix`, `cleave`, `splice`) and heuristic `CCCE` consciousness telemetry (`is_conscious`, `PHI_THRESHOLD = 0.7734`) in Phase 3 so enterprise/DARPA reviewers interact solely with standard Qiskit/OpenQASM3 passes (`osiris_cli/compiler_passes/`).
   - **Governance Compatibility Constraint:** Existing unit tests (`tests/test_core.py`) require backward compatibility of `Qbyte` and `CCCEMetrics` symbols during Phase 2; therefore `SovereignExecutor` defaults to `enable_ccce=False` and `enable_healing=False` while preserving legacy symbols until owner-approved Phase 3 deprecation.
