# OSIRIS Frontier Research — Phase 2 Executive Research Report (`PHASE2_EXECUTIVE_REPORT.md`)

**Principal Investigator:** Devin Phillip Davis (OSIRIS Quantum Systems / Agile Defense Systems LLC)  
**Isolated Development Branch:** `feat/phase2-scientific-repair` (branched from `main` `97f5db21` in `/home/osiris/osiris-assessment/work/osiris-cli-phase2`)  
**Execution Governance:** Local-first, evidence-governed (`0` QPU jobs submitted, `0` IBM RU spent, `0` external pushes, `0` bytes modified in `/home/osiris/claims_ledger.jsonl` or `osiris_cli/claims.py`).

---

## 1. Phase 2 Executive Research Summary

Phase 2 restored the 44-claim Phase 1 scientific baseline, executed SHA-256 provenance verification across 125 baseline/archived files, repaired mathematical and simulator defects across `qByte`, optimal-transport ($W_2$), and time-dependent Floquet/Zeno controls, conducted a 7-agent adversarial review with pre-preregistration code/protocol tightening, and delivered three simulation-backed experimental packages plus two bounded literature-integration workflows:

1. **Marketable Quantum Compilation Capability (`TESSERACT-MAP-02` + `osiris_cli/compiler_passes/fanout_degree_reducer.py`):**
   - Decomposed the archived 40-qubit `ibm_kingston` improvement (`5.305% -> 17.995%` 4-string support; `dansntf8gn2s739n32e0`) and proved that rewriting high-degree commuting CNOT stars (`degree 20`) into degree-2 chains (`reduce_commuting_cx_fanouts`) unlocks stock Qiskit's exact error-weighted subgraph isomorphism (`VF2Layout`), cutting 2-qubit `CZ` gates by **$4.18\times$ (`163 -> 39 CZ`, `0` SWAPs)** and raising calibrated 40-qubit classical fidelity from `0.0639` to `0.3847` on `FakeKingston` (`0.0171 -> 0.4404` on `FakeMarrakesh`).
2. **Falsifiable Open-System Physics Characterization (`CORRELATED-NOISE-01`):**
   - Formulated five competing models (`H0` independent local Lindblad, `H1` static spectator $ZZ$, `H2` Kossakowski-PSD correlated Markovian dephasing, `H3` exact piecewise-integrated non-Markovian memory kernel, and `Apparatus` `24/36 ns` pulse collision) across a `dt = 4.0 ns`-aligned 27-condition diagnostic suite (`T = 8.192 µs`, `16.384 µs`).
   - Under structured block-covariance Generalized Least Squares (GLS) accounting for `Bridge E5` ($r = 0.812$) electronics gain drift, blind 5-way recovery achieved **`50/50` (`100.0%` overall accuracy, `100.0%` `H0` null specificity)** and **5.5× lower noisy out-of-sample RMSE** (`0.00309` vs `0.01705`, `23.5×` latent) on unseen held-out couplers with zero parameter leakage.
   - Programmatically verified **10 `>= 5`-hop disjoint couplers** on `ibm_marrakesh` (`4` train + `6` held-out multiplexed simultaneously in one PUB), fitting compact Stage 1 into **`~5.4 minutes` (`< 600 s` IBM OpenPlan cap)**.
3. **Anomaly Resolution (`TAU-REVIVAL-01` & Coaxial $T_1$):**
   - Byte-exact archival inspection proved `tau_analysis_d5h5n9agim5s73ahi1h0.json` recorded a period-8 gate-angle sweep (`tau_0_observed: 8`, peaks `[0,8,16,24]`), while `d5ig8...` and `d5ikod...` were empty pre-submission manifests; implemented signed conjugate-quadrature virtual-$Z$ detuning falsification (`slope = +1.0002`, zero V-folding).
   - Preserved the negative $T_1(f)$ trend across **27/28** processors ($\hat{\beta}_{\text{FE}} = -56.26 \pm 6.07\,\mu\text{s/GHz}$, $z = -9.26$; DerSimonian–Laird $\hat{\beta}_{\text{RE}} = -114.38 \pm 18.04\,\mu\text{s/GHz}$, $z = -6.34$) while classifying `3.526 GHz` / `1.763 GHz` collective-mode periodicity as **`INCONCLUSIVE`** (`0.12–0.33` cycles/chip, quadratic `VIF = 200–15,791`, `M0_linear` winning BIC on `27/28` chips).

---

## 2. Canonical Six-Label Status Accounting & Corrected Claims Ledger

Every deliverable and claim record in Phase 2 is strictly classified under the six canonical epistemic labels (`PHASE2_DELIVERABLE_HASHES.json`):

| Canonical Status Label | Phase 2 Scope & Count | Representative Deliverables / Claims |
|---|---|---|
| **`Implemented and tested`** | **All 18 Phase 2 code & test modules** (`0` ruff errors; full pytest suite green) | `wasserstein_verified.py`, `qbyte_system/*`, `fanout_degree_reducer.py`, `color_code_eval.py`, `nmr_budget.py`, `review_coverage.py`, `aeterna_controls_repaired.py`, `w2_circuit_reanalysis.py`, `tesseract_map_02.py`, `correlated_noise_models.py`, `simulate_correlated_noise.py`, `tau_revival_models.py`, `coaxial_t1_reassessment.py` |
| **`Implemented but untested`** | **`0` Phase 2 modules** (legacy quarantine scripts remain unexecuted by design) | Zero newly written Phase 2 modules remain untested; quarantined historical `deploy_aeterna_porta_v2_IGNITION.py` is inspected-only and intentionally never executed. |
| **`Simulated`** | **6 offline simulation & benchmark studies** (plus $>125\text{ MHz}$ Floquet regime) | `tesseract_map_02_simulation_results.json`, `correlated_noise_01_identifiability_results.json`, `w2_27q_erratum_reanalysis.json`, `tau_revival_01_audit_and_sim.json`, `qbyte_benchmarks.json`, `NB-18` |
| **`Hardware-observed`** | **Historical IBM Quantum runs (`NB-19` raw counts in hand; `NB-11,14,15,17,27,28,37` owner-reported)** | `dansntf8gn2s739n32e0` (`ibm_kingston`), `E0–E4` (`ibm_marrakesh`/`ibm_fez`), `Bridge E1–E5` (`81` `ibm_fez` jobs), `HP-004` (`0.633423` joint Bell overlap); owner-reported `NB-14/15/17/27/28/37` qualified as `HARDWARE-OBSERVED / OWNER-REPORTED`. |
| **`Independently replicated`** | **`0` OSIRIS claims** (`5` external trapped-ion/fluxonium Bell benchmarks cited for `NB-07`) | Zero internal OSIRIS hardware claims have third-party independent lab replication yet; external literature benchmarks (`Benhelm 2008`, `Ballance 2016`, `Gaebler 2016`, `Clark 2021`, `Ding 2023`) are independently replicated external results refuting `F_max = 0.9787`. |
| **`Proposed only`** | **All 44 claim transitions & all QPU hardware execution plans** | `claims_ledger_phase2_proposed.jsonl`, `CLAIM_CHANGESET_FOR_OWNER_APPROVAL.md`, and QPU submission stages of `TESSERACT-MAP-02`, `CORRELATED-NOISE-01`, and `TAU-REVIVAL-01`. |

- **Protected Authoritative Files (Verified Unchanged):**
  - `/home/osiris/claims_ledger.jsonl`: `sha256: 567a1ffd6c425eab3b031d5191ecd3fbd1198b89233bbbe87f5c1ee0fa25fb1f`
  - `/home/osiris/osiris-assessment/work/osiris-cli-phase2/osiris_cli/claims.py`: `sha256: 8563ac0fa78d69b259ea6ad35d498182737f5ac2ddfcd525c5a4a7ad47f47fb1`
- **Proposed Ledger Artifact:** `docs/phase2/claims_ledger_phase2_proposed.jsonl` (`sha256: 87638fcf7a065d2bcb0bdfa03898b9fdffda5764a2a74020b0a7f5c0805ee241`).
- **Missing External File Logged:** `OSIRIS_Signal_Noise_Forensic_Audit_2026-11-16.md` is absent across `/home/osiris` and all 171 local git mirrors (`PHASE2_STATE_INVENTORY.json`).
- **Credential Exposure Advisory (Metadata Only — No Secrets Printed):** Rotate credentials exposed in `/home/osiris/apikey.json`, world-readable `/home/osiris/apikey - Copy.json` (`chmod 600` or shred), `/home/osiris/osiris-ide/.env`, and legacy mirror `dnalang-sovereign-sdk-v2.git` (`HEAD:apikey.json`).

---

## 3. Mathematical & Software Repair Summary (`qByte`, $W_2$, Aeterna, Workflows)

1. **`qByte` Classical Simulator (`qbyte_system/`, `docs/phase2/QBYTE_CORRECTNESS_REPORT.md`):**
   - Repaired `SovereignExecutor._apply_toffoli` double-swap identity defect, unified big-endian wire-to-axis indexing (`n - 1 - q`) across `measure_qubit` and 1Q/2Q/3Q gates, disabled default non-unitary mid-circuit conjugation (`enable_healing=False`) and heuristic `CCCE` (`enable_ccce=False`), aligned `QbyteRuntime(strict_l0=True)` per-gate decoherence increments (`0.0005/op`), added explicit `measure(q, c)` subset mapping and `.get_statevector(endian="little")` / `.get_counts(endian="little")`, and achieved `0.0` (`<= 4.4e-16`) max amplitude discrepancy vs Qiskit `Statevector` (`110` tests passing across `test_qbyte_correctness.py` and `test_core.py`).
2. **Verified Optimal Transport & $W_2$ Erratum (`osiris_cli/metrics/wasserstein_verified.py`, `docs/phase2/W2_MATHEMATICAL_ERRATUM.md`):**
   - Proved `w2_robustness_paper.tex` Eq. (7) computes $C(p,q) = \sum c(x,y)p(x)q(y)$ ($\sqrt{C(p_{\text{unif}},p_{\text{unif}})} = 0.50918 \neq 0$), omitted Layer-0 Hadamards ($p_{\text{dom}}^{27} = 12.91\% > 3.98\%$ at $\theta_1 = 2.118$), had exact Z-basis `CP`/`RZ` phase cancellation ($4.44\times 10^{-16}$), and miscomputed `112×` via $1/\overline{N_{\text{unique}}}$ (`2,818.5`) instead of $2^{27}$.
   - Implemented sparse HiGHS Kantorovich LP `exact_wasserstein_lp`, `exact_wasserstein_1d`, certified `[LB, UB]` `sinkhorn_wasserstein_bounded`, `>64Q` `hamming_cost_matrix`, and split-sample `debiased_empirical_wasserstein` (reducing 27Q finite-shot plug-in bias from `>0.20` to `<0.05`).
3. **Aeterna Controls (`phase2/aeterna_w2/aeterna_controls_repaired.py`, `docs/phase2/AETERNA_METHODS_AUDIT_AND_REPAIR.md`):**
   - Repaired `1 GHz @ 1 ns` integer-phase aliasing, `int(1.25) == 1` Zeno truncation, finite 2-`SX` step Nyquist checks (`64–88 ns`), calibrated `3,144 ns` (`ibm_fez`) / `5,304 ns` (`ibm_marrakesh`) `measure+reset` bounds, and `opt_level=3` transpilation survival; quarantined fabricated `IGNITION.py`.
4. **Bounded Literature Workflows (`osiris_cli/workflows/`):**
   - `color_code_eval.py`: Enforces atomic `SyndromeRecord` batch-locked splits, abstention/deadline failure accounting, Clopper–Pearson CIs prior to $(r + 0.5)^{-1}$ scaling (`arXiv:2610.24077v1`), exact McNemar tests, and acquisition-batch cluster-robust SEs.
   - `nmr_budget.py`: Enforces the `1,420`-circuit / `35,500`-shot budget equation (`arXiv:2609.17102v1`), 5 non-additive error separation tiers, and signed/complex Wasserstein rejection.
   - `review_coverage.py`: Enforces structured `STAGGERED_DD` duration/comparator schemas (`32 µs` contrast, never `8 µs` vs `16 µs`), 3-way `Tesseract` namespace separation, and unread-section speculation blocking.

---

## 4. Experimental Packages Summary (`TESSERACT-MAP-02`, `CORRELATED-NOISE-01`, `TAU-REVIVAL-01`)

| Package | Directory | Key Quantitative Result (Offline Verification) | Hardware Budget (Awaiting Approval) |
|---|---|---|---|
| **`TESSERACT-MAP-02`** | `phase2/tesseract_map_02/` + `osiris_cli/compiler_passes/` | Exact 12Q statevector & 40Q $\mathrm{GF}(2)$ equivalence; `163 -> 39 CZ` (`0` SWAPs); $B = 15$ blocks gives `85.8%` power at `1.5 pp` MDE; `C5 vs C4` verified as `2,188 ns` virtual-$Z$ A/A null sentinel. | `135` PUBs, `552,960` shots (`~2.8 min` QPU on `ibm_marrakesh` or `ibm_fez`) |
| **`CORRELATED-NOISE-01`** | `phase2/correlated_noise_01/` | Exact piecewise OU/oscillatory integration (`3e-16` error); `dt = 4.0 ns` ISA builder (`T = 8.192 µs`); `50/50` (`100%`) GLS-BIC recovery; Student-$t_7$ MDE = `1.394 pp`; `5.5×` lower noisy held-out RMSE (`0.00309` vs `0.01705`). | Compact Stage 1 (`10` `>=5`-hop couplers in 1 PUB): `432` PUBs, `1.81M` shots (`~5.4 min < 10 min` OpenPlan cap) |
| **`TAU-REVIVAL-01`** | `phase2/tau_revival_01/` | Archival JSON audit refutes universal `46.9787 µs` invariance (`6.7–333 µs` span; `ibm_fez` $F = 0.3438 < 0.50$); signed conjugate-quadrature fit achieves slope `+1.0002` at $\Delta f_0 = -6.5$ & $+21.7\text{ kHz}$. | `0` RU recommended (already refuted; optional protocol ready if requested) |

---

## 5. Updated Research Priority Matrix

| Rank | Initiative | Scientific / Commercial Value | Feasibility & Signal-to-Drift | Recommended Disposition |
|---:|---|---|---|---|
| **1** | **`TESSERACT-MAP-02` + `fanout_degree_reducer` Pass** | **High Commercial SDK Value:** Deterministic `4.2×` 2Q-gate reduction (`163 -> 39 CZ`) unlocking Qiskit `VF2Layout` on planar heavy-hex processors. | **Very High:** Predicted `+20–35 pp` $F_{\text{cl}}$ gain (`>15×` above `1.5 pp` MDE) in **`~2.8 min`** QPU time. | **READY FOR STAGE-1 QPU APPROVAL** & immediate integration into `osiris.compiler`. |
| **2** | **`CORRELATED-NOISE-01` (Multiplexed 10-Coupler GLS Suite)** | **High Novel-Physics / Characterization Value:** Tests 5-way mechanistic separation (`H0–H3` + `Apparatus`) and zero-retraining schedule transfer on unseen couplers. | **Moderate-High:** Requires $B=8$ (`33,600` shots/basis, `MDE = 1.39 pp`) + GLS whitening against `Bridge E5` gain drift; fits in **`~5.4 min`** OpenPlan window. | **READY FOR GATED STAGE-1 QPU APPROVAL** (note Product vs Physics quota trade-off in `MULTIAGENT_ADVERSARIAL_REVIEW.md`). |
| **3** | **Bounded Color-Code & NMR Verification Harnesses (`Workflows A & B`)** | **High DARPA / Enterprise Verification Value:** Leak-free batch-locked decoder replay and 5-tier non-additive spectral error budgeting. | **High (0 QPU RU):** Pure offline evaluation once external syndrome/Hamiltonian datasets are ingested. | **ACTIVE OFFLINE CAPABILITY** — obtain external `arXiv:2610.24077v1` syndrome records for baseline replay. |
| **4** | **Coaxial $T_1(f)$ Characterization** | **Low-Moderate:** Smooth negative $T_1(f)$ slope is real (`27/28` chips), but collective chip-mode period is unidentifiable on fixed-frequency transmons. | **Blocked on Fixed-Frequency Transmons:** `<0.33` cycles/band; requires flux-tunable multi-GHz sweep hardware. | **HOLD / ARCHIVE AS `INCONCLUSIVE`** (do not retrofit $L_{\text{eff}}$). |
| **5** | **`TAU-REVIVAL-01` (`~46 µs` Universal Constant)** | **Resolved / Refuted:** Historical artifacts fully explained by gate-angle sweeps, empty manifests, $T_1$ ground-state decay, and pair-specific detuning. | **Low Return on QPU Quota:** Already falsified by archived `ibm_fez` 3-basis tomography (`F = 0.3438`). | **CLOSE / DO NOT SPEND QPU RU** unless PI explicitly requests confirmatory detuning-ramp demo. |

---

## 6. Phase 3 Execution Roadmap & Decision Gates

### Gate 1 — Immediate Owner Governance & Security Actions (`0` QPU RU)
1. **Credential Rotation:** Rotate IBM Quantum / API keys in `/home/osiris/apikey.json`, `/home/osiris/apikey - Copy.json`, `/home/osiris/osiris-ide/.env`, and purge `apikey.json` from `dnalang-sovereign-sdk-v2.git` history.
2. **Claim Register Sign-Off:** Review `docs/phase2/CLAIM_CHANGESET_FOR_OWNER_APPROVAL.md` and authorize applying approved verdict updates (`W2_ROBUSTNESS`, `F_MAX`, `STAGGERED_DD`, `HERON_R2_STAG_XY8`, `ERROR_SUPPRESSION_1E6` regex, `LOCAL_QVM`) to `osiris_cli/claims.py` and `/home/osiris/claims_ledger.jsonl`.

### Gate 2 — Authorized Hardware Execution (~8.2 min Combined QPU Time Within 10-min OpenPlan Window)
1. **Sprint 3.1 (`TESSERACT-MAP-02`, `~2.8 min` QPU on `ibm_marrakesh`):** Freeze path selection on pre-lock snapshot $t_{\text{select}}$, export QPY + SHA-256 manifest + union SPAM circuits (`build_union_spam_calibration_circuits`), execute $B = 15$ randomized interleaved blocks (`552,960` shots), archive live `backend.properties()`, and run `analyze_paired_blocks()`.
2. **Sprint 3.2 (`CORRELATED-NOISE-01` Stage 1, `~5.4 min` QPU on `ibm_marrakesh`):** Execute multiplexed 10-coupler (`>=5`-hop disjoint: `4` train + `6` held-out) ISA suite (`432` PUBs, `1.81M` shots), run GLS-BIC model selection (`fit_competing_models`), and evaluate the pre-registered cluster-level held-out kill gate ($t_5, p < 0.05$). Proceed to `ibm_fez` Stage 2 (`~2.8 min`) **only if** Stage 1 passes both $\Delta\mathrm{BIC}_{\text{GLS}} \ge 8.0$ and held-out transfer.

### Gate 3 — OSIRIS SDK Productization (`0` QPU RU)
1. Wire `osiris_cli/compiler_passes/fanout_degree_reducer.py` + `pec_budget.py` + `calibration_spc.py` into a standard Qiskit `PassManager` plugin (with automatic QEC-gadget exclusion guard).
2. Replace `osiris/compiler/backends/sovereign_adapter.py`'s coin-flip stub with `SovereignExecutor(enable_healing=False, enable_ccce=False)` and deprecate heuristic `CCCE` consciousness metrics from enterprise/DARPA-facing CLI surfaces.

---

## 7. Exact Verification Commands & Unresolved Blockers

```bash
# 1. Lint verification (observed: "All checks passed!", 0 errors)
cd /home/osiris/osiris-assessment/work/osiris-cli-phase2
/home/osiris/osiris-assessment/venvs/osiris-cli/bin/ruff check \
  osiris_cli/metrics osiris_cli/workflows osiris_cli/compiler_passes phase2 \
  tests/test_qbyte_correctness.py tests/test_wasserstein_verified.py tests/test_phase2_*.py \
  tests/test_tesseract_map_02_suite.py tests/test_correlated_noise_01_suite.py \
  tests/test_tau_revival_01.py tests/test_coaxial_t1_reassessment.py tests/test_fanout_degree_reducer.py

# 2. Phase 2 test suite (observed: 140 passed)
PYTHONPATH=. /home/osiris/osiris-assessment/venvs/osiris-cli/bin/pytest \
  tests/test_wasserstein_verified.py tests/test_phase2_workflows.py tests/test_qbyte_correctness.py \
  tests/test_core.py tests/test_phase2_aeterna_w2.py tests/test_phase2_claims_ledger.py \
  tests/test_tesseract_map_02_suite.py tests/test_correlated_noise_01_suite.py \
  tests/test_tau_revival_01.py tests/test_coaxial_t1_reassessment.py tests/test_fanout_degree_reducer.py -v
```

### Unresolved External Blockers
1. **QPU Submission Authorization:** Hardware execution of `TESSERACT-MAP-02` and `CORRELATED-NOISE-01` is intentionally halted pending explicit PI authorization.
2. **Claim Promotion Authorization:** Protected registers `osiris_cli/claims.py` and `/home/osiris/claims_ledger.jsonl` remain untouched pending PI approval of `CLAIM_CHANGESET_FOR_OWNER_APPROVAL.md`.
3. **Missing Historical Files:** `OSIRIS_Signal_Noise_Forensic_Audit_2026-11-16.md` and raw bitstring count files for the six Dec 23, 2025 `ibm_fez` $W_2$ jobs (`d554h...`) and external `arXiv:2610.24077v1` syndrome archives are not present on local disk.
