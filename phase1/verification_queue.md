# Phase 1 verification queue and experiment cards (2026-11-22)

- **Reviewer:** self-review (Claude). No independent reviewer ran.
- **Claim IDs** refer to `claim_evidence_map.jsonl`.
- **Owner abbreviations:** "owner" is the project owner; "agent" is a Claude Code session working under the owner's approval.

## 1. Prioritized checks

The queue is ordered by decision value per unit of cost. Checks that can invalidate a whole downstream comparison come first.

| # | Check | Claim | Owner | Prerequisite | Expected artifact | If it passes | If it fails |
|---|---|---|---|---|---|---|---|
| 1 | BT2T inversion against an exact oracle: all 32 inputs, phases, unused codewords | BT2T-2 | agent | none | `tests/test_bt2t_inversion.py`, `manifest.json` | **Done this session: pass.** Advance to offline compiler validation (G1 to G2). | Revise the circuit; hardware stays blocked. |
| 2 | Independent rederivation of encoding `bt2t-enc-1` and rerun of the tests | BT2T-2 | second reviewer (human or separate agent) | the reviewer has the repo | review note | G1 is signed off by someone other than the author. | Freeze nothing; fix the defect first. |
| 3 | Retrieve IBM `creation_date` for job `dc0uhdts2omc7389tmt0` | SCI-XTOP-1 | owner | **BLOCKED: owner reported on 2026-11-22 that they no longer have access to the IBM account that ran the job** | job metadata JSON (CRN redacted), only if IBM support or a former account holder can supply it | Timing claim becomes SUPPORTED or REFUTED on the provider clock. | Default while access is lost: the claim stays INCONCLUSIVE, and the paper should not say the preregistration preceded the job. |
| 4 | Decide the CRSM_ARCH verdict wording (R-3) | SCI-ARCH1-5 | owner | none | register PR | Register matches both decision rules. | — |
| 5 | Correct Zenodo `publication_date` in a new version note | SCI-XTOP-2 | owner | Zenodo access | new version record | Metadata is accurate; nothing is backdated. | — |
| 6 | Relabel the A-field harness outputs and add injection-recovery | PHY-9 | agent, if approved | none (local) | harness patch and test | The harness can show detector sensitivity. | — |
| 7 | Errata pointer in `falsifiable_predictions.md` | PHY-7 | owner decision | none | one-line header change | — | — |
| 8 | First approved Nexus preflight on one circuit, with Nexus HQC compared to `local_hqc` | ENG-NEXUS-4 | owner approves; agent runs | Nexus credentials, explicit approval | preflight receipt | Live compatibility becomes SUPPORTED. | File adapter defects before any paid job. |
| 9 | Install the Vercel GitHub App on `quantum-osiris`; redeploy from Git | ENG-DEPLOY-1/4 | owner | org admin | Git-linked deployment | Source binding moves from operator-supplied to platform-recorded. | — |
| 10 | Retrieve NCLM-ARCH-1 run directories (evaluation logs) | SCI-ARCH1-4 | owner | the machine that ran them | logs | The question of when results were viewed becomes answerable. | It stays unknown. |

## 2. Provisional research landscape

Scores run from 1 to 5, where higher is better; for cost, 5 means cheapest. They reflect this reviewer's judgement from the literature and the inspected repo, not a measurement. "Null info" is what a negative result would teach us.

| Area | Representative task | Classical comparator | Theory | Classical gap | HW readiness | Cost | Null info | Total | Class |
|---|---|---|---|---|---|---|---|---|---|
| QEC: memory and repeated correction | logical memory vs rounds, small codes | none needed (the reference is the unencoded qubit) | 5 | n/a | 4 | 4 | 5 | **18** | useful hardware primitive |
| Compiler and placement diagnostics (echo) | predict held-out circuit fidelity | calibration-ranked and topology-only placement | 4 | n/a | 5 | 5 | 4 | **18** | engineering benchmark |
| Exact group primitives (BT2T) | inversion, then multiplication and Fourier transform on 2T | exact classical oracle (trivial) | 5 | none (classically trivial) | 5 | 5 | 3 | **18** | scientific benchmark and compiler test bed |
| Many-body dynamics | Trotterized 2D Ising and Heisenberg dynamics past exact-diagonalization size | tensor networks, Pauli propagation | 4 | 2 (classical methods keep catching up) | 3 | 2 | 3 | 14 | advantage claim, contested |
| Chemistry and materials | ground-state energy to chemical accuracy, FeMoco-class | DMRG, CCSD(T), AFQMC | 5 | 3 (only at fault-tolerant scale) | 1 (needs about 10⁶ physical qubits; T-count about 10⁹ to 10¹⁰, roughly) | 1 | 2 | 12 | end-to-end advantage, FT era |
| Random-circuit sampling and peaked circuits | sampling or peak finding (ADV-0) | MPS, Pauli propagation, light cone | 4 | 1 (ADV-0: classical attack won 30 of 30 cells) | 4 | 3 | 3 | 15 | advantage claim, **STOP at current sizes** |
| Metrology | phase estimation with entangled probes | standard quantum limit | 5 | n/a (physical-resource advantage) | 3 | 3 | 3 | 14 | physics benchmark |
| Cryptography (Shor) | RSA-2048 factoring | GNFS | 5 | 5 | 1 (about 10⁶ physical qubits, around a week; unknown for this stack) | 1 | 1 | 13 | FT-era advantage |
| Optimization (QAOA, annealing) | MaxCut and similar on NISQ | Goemans–Williamson, simulated annealing, heuristics | 2 | 1 | 4 | 4 | 2 | 13 | weak advantage case |
| ML with quantum kernels | classification | classical kernels and NNs | 2 | 1 (data loading and dequantization) | 4 | 4 | 1 | 12 | weak |
| Communication (teleportation, QKD) | teleportation fidelity with feed-forward | — | 5 | n/a | 4 | 4 | 3 | 16 | useful primitive (exercises feed-forward) |

**Cost accounting.** Every comparison against a classical method must count:
- compilation;
- shots and repetitions;
- postselection loss (acceptance rate);
- decoding;
- result processing.

Where a row above omits one of these, it is **unknown**, not zero.

## 3. Experiment cards

### Card A: OSIRIS-BT2T-001, exact 2T inversion (milestone 1)

- **Hypothesis (engineering):** a 5-qubit reversible circuit implements group inversion on encoding `bt2t-enc-1` exactly. The structured synthesis needs fewer native two-qubit gates than the generic transposition reference, with the logical operation held fixed.
- **Reference:** an exact multiplication table from integer quaternion arithmetic. Inverses are found from the table and cross-checked against conjugation. The generic transposition circuit is the synthesis reference.
- **Controls:**
  - a positive control: an injected CZ phase error is detected while basis outputs are unchanged;
  - the earlier prototype, which fails necessary conditions.
- **Primary metric:**
  - *Ideal stage:* the maximum absolute difference between the circuit matrix and the permutation matrix (must be below 1e-9). Measured values: 3.5e-14 for the generic circuit and 3.6e-15 for the structured one.
  - *Later hardware stage, not authorized:* correct-output probability under uniform weighting over the 24 valid inputs, reported separately from unused-codeword outcomes and from preparation and readout controls.
- **Local PROXY results** (not provider output):

  | Target | Generic | Structured |
  |---|---|---|
  | Qiskit, all-to-all, opt 3 | 400 CZ | 48 CZ |
  | Qiskit, 5-qubit heavy-hex fragment, opt 3 | 723 CZ | 72 CZ |
  | pytket H2-native rebase, with FullPeepholeOptimise | 400 ZZPhase | 47 ZZPhase |

  These are differences between syntheses, not hardware advantages.
- **Falsifier:** any basis mismatch, any phase mismatch beyond tolerance, or an unused codeword that is not fixed.
- **Stop/go:**
  - **GO** to offline compiler validation: pinned compiler versions, saved compiled artifacts, equivalence on each target's native set. All ideal checks pass.
  - **Hardware: BLOCKED** until (a) an independent reviewer signs off G1; (b) a frozen preregistration exists covering input weighting, shots, the unit of inference (calibration window or job, not shots), and the equal-shot rule; (c) target access and cost are verified at run time; (d) the owner gives explicit approval.

### Card B: OSIRIS-ECHO, does echo-based placement predict held-out performance?

- **Hypothesis:** placements ranked by the return probability of a forward-then-inverse echo predict the fidelity of *unseen* circuits better than calibration-ranked, topology-only or random placements, under the same selection budget.
- **Reference:** held-out circuits whose ideal outputs are exactly computable, such as BT2T inversion, Clifford circuits and mirror circuits.
- **Controls:**
  - a no-workload echo (identity);
  - a known perturbation, such as a deliberate over-rotation, which must lower the return;
  - a compiled-artifact inspection showing the workload was not cancelled. A barrier alone is not enough evidence.
- **Primary metric:** Spearman rank correlation between the echo score and held-out fidelity, minus the same correlation for the best non-echo comparator. Uncertainty comes from a bootstrap over calibration windows, which are the unit of inference.
- **Minimum effect of interest:** not yet set, so this starts as an **exploratory pilot** with a separate confirmatory dataset.
- **Falsifier:** echo ranking is no better than calibration ranking on held-out windows.
- **First stop/go:** an offline noisy simulation with a known placement-dependent noise model must recover the right ranking. If it cannot, STOP before hardware.
- **Mechanism:** none is inferred from return curves.

### Card C: OSIRIS-QEC, when does repeated correction improve logical reliability?

- **Audit first.** dnalang-core `wen_plaquette_pass.py` is a TKET pass with a provenance capsule. Inspected evidence covers compilation and tests only; no syndrome or logical-outcome dataset for Wen-plaquette or qLDPC was found in this pass. XTOP-005's [[4,2,2]] encoding did not beat bare qubits on hardware (SCI-XTOP-4).
- **Hypothesis:** for the smallest code with an auditable decoder (a distance-3 repetition code; a d=3 surface code if the target supports it), logical failure per round under decoded correction falls below both the unencoded and the encoded-without-correction arms, at matched task duration.
- **Arms:**
  1. unencoded;
  2. encoded, no correction;
  3. postselected detection, with the acceptance rate reported;
  4. decoded correction, comparing offline decoding or Pauli-frame tracking against physical feed-forward, labelled separately.
- **Reference:** stim and pymatching (or an equivalent auditable decoder) under stated circuit-level noise: relaxation, dephasing, coherent over-rotation, readout and reset, leakage where modelled, and correlated noise. Any mechanism omitted is documented.
- **Primary metric:** logical error per round, from a fit of failure against rounds 1 to N. Unconditional outcomes are reported separately from postselected ones.
- **Falsifier:** decoded correction does not beat arm 2 at any tested round count.
- **First stop/go:** the simulation must show the crossing at the target's published error rates, verified at run time. If not, STOP.
- **Scope limit:** a single code distance answers the question for that size only. Scaling with distance remains open, and a single-size result is not evidence of scalable fault tolerance.

## 4. Go/no-go

**GO:** BT2T-001 advances to **offline compiler validation only** (G1 to G2).

**NO-GO (BLOCKED):** any hardware, emulator, Nexus upload, remote compile or costing for BT2T. Unblocking needs:
- an independent G1 review (check 2);
- a frozen preregistration;
- explicit owner approval naming the provider, target, shot budget and cost ceiling.

---

## 5. Reconciliation & Execution Ledger (2026-11-24)

The table below records the post-2026-11-22 execution and verification outcomes for the prioritized checks and experiment cards above, preserving the original 2026-11-22 baselines verbatim for auditability.

### 5.1 Prioritized Checks Status Update

| # | Check | Claim | Updated Status (2026-11-24) | Evidence Artifact(s) |
|---|---|---|---|---|
| 2 | Independent rederivation of `bt2t-enc-1` & test rerun | `BT2T-2` | **CLOSED — PASS (`Gate G1` signed off)** via clean-room $SU(2)$ $2\times 2$ matrix adjoint & Boolean coset-parity derivation | `experiments/bt2t_inversion/G1_INDEPENDENT_REVIEW.md` |
| 3 | Retrieve IBM `creation_date` for job `dc0uhdts2omc7389tmt0` | `SCI-XTOP-1` | **CLOSED — `SUPPORTED` on provider clocks:** Zenodo `24425416` server `created = 2026-11-21T18:53:24.06Z` precedes IBM Quantum `timestamps.created = 2026-11-21T18:53:43.172751Z` by **`+19.11 s`** | `experiments/xtop006_fez/sci_xtop1_dc0uhdts2omc7389tmt0_provider_receipt.json` |
| 6 | Relabel A-field harness outputs & add injection-recovery | `PHY-9` | **CLOSED — REMEDIATED (`SIMULATED`):** emits `BELOW_THRESHOLD_WITH_ZERO_INJECTED_SIGNAL` + `physical_claim_assessment: UNVERIFIED` + 4,000-trial injection-recovery (`0.35%/0.25%` FPR at $0\sigma$, `49.7%` at $3\sigma$, `>99.8%` at $6\sigma$) | `research/a_field_study/experiments/falsification_harness.py`, `experiment_results.json`, `tests/test_echo_placement_and_phy9.py` |
| 8 | Approved Nexus 1-circuit preflight & HQC comparison | `ENG-NEXUS-4` | **OFFLINE PREFLIGHT READY (`UNVERIFIED` live):** `36.9000` HQC single-canary `xtop005q_Q0_L3_ZZ` verified offline (`0` paid HQC, Tier 3 hard-blocked); awaits live `qnexus` login | `scripts/nexus_onboard_preflight.py`, `xtop005/eng_nexus_4_preflight_receipt.json` |

### 5.2 Experiment Cards & Milestone 2 Status Update

1. **Card A (`OSIRIS-BT2T-001`, Milestone 1 Hardware Stage) — EXECUTED (`HARDWARE_ATTESTED`, `N_windows = 1`):**
   - **Pre-gates satisfied:** (a) independent `G1` review (`G1_INDEPENDENT_REVIEW.md`); (b) write-ahead preregistration commit `240046bd` (`PRE_REGISTRATION.md`, `qpy_sha256 = 81a90cf8...`); (c) runtime quota check (`19 s` available $\to$ `11 s` QPU charged, `8 s` remaining); (d) explicit owner approval.
   - **Result (`ibm_fez`, `job_id = dc2ee3uq4uuc73a56o6g`, `288` PUBs $\times$ `96` = `27,648` equal shots on `[21, 22, 23, 16, 24]`):** `H1 SUPPORTED` (`72-CZ` structured `0.5337 +/- 0.0059` raw, `0.6104` RO-mitigated, `24/24` modal vs `723-CZ` generic `0.0661 +/- 0.0030`; `+46.76 pp`, Wilcoxon $p = 1.19\times 10^{-7}$); `H2 SUPPORTED` (unused-8 `0.5595` vs `0.0777`, leakage `0.0940 < 0.15`); `H3 REFUTED` (`PREP_RO_CTRL` valid-24 `0.8643 < 0.90` due to `14.91x` live readout drift on `q24`). Manuscript package: `papers/bt2t001_binary_tetrahedral_fez/`.
   - **Milestone 2 (`OSIRIS-BT2T-002`, `SIMULATED`/`PROXY`):** Exact 10Q zero-ancilla $2T$ multiplication (`1,739` vs `33,042 CZ` all-to-all = `19.00x`; `3,662` vs `78,035 CZ` heavy-hex = `21.31x`; `0.0` `1024x1024` unitary error) and exact 5Q Peter–Weyl non-abelian $2T$ QFT (`9.80e-16` unitarity, `3.11e-16` simultaneous block-diagonalization across all 7 irreps) verified in `experiments/bt2t_primitives/`.
2. **Card B (`OSIRIS-ECHO`) — FIRST STOP/GO GATE PASSED (`SIMULATED`, `0 s` QPU):**
   - Stabilizer-reflected `16-CZ` probe retains `16 CZ` after `RemoveBarriers() + optimization_level=3`. Across `K = 10` disjoint `T5` subgraphs on `ibm_fez` and `M = 16` simulated calibration windows, `ECHO_RANKED` achieved $\bar{\rho} = +0.9333$ vs stale `CALIBRATION_RANKED` $+0.3871$ ($\Delta\rho = +0.5462$, 95% window-bootstrap CI `[+0.4594, +0.6288]`) across held-out `BT2T`, Clifford, and mirror circuits (`experiments/echo_placement/artifacts/echo_stopgo_report.json`). Hardware multi-window pilot remains gated on replenished IBM QPU quota.
3. **Card C (`OSIRIS-QEC`) — FIRST STOP/GO GATE COMPLETED (`SIMULATED`, `0 s` QPU):**
   - Auditable Dijkstra + exact bitmask-DP/Blossom MWPM space-time decoder (`100%` single-fault recovery across `[[3,1,3]]` and Tomita-Svore `[[9,1,3]]` rotated surface code; 5Q Kraus engine `< 1.9e-16` vs analytical $T_1/T_2$; `|2>` leakage documented as omitted) evaluated across `R in {1, 2, 3, 5, 7, 9}` (`experiments/qec_card_c/artifacts/card_c_qec_crossing_report.json`):
     - **`[[3,1,3]]` (`5Q`):** **GO** (`PASS_CROSSING_OBSERVED` at `R = 1` on `FEZ_PUBLISHED`, `R = 1/2` on `FEZ_DRIFTED_SPAM`, and `R = 1` on `H2_1_PUBLISHED`).
     - **`[[9,1,3]]` (`17Q` surface code):** **GO on native `H2-1`** (`R_cross = 3` for both Pauli-frame `2.26e-4` and physical feed-forward `2.39e-4`); **CONDITIONAL / STOP on heavy-hex** (`FEZ_PUBLISHED` crosses only under software Pauli-frame tracking at `R = 2`, fails under physical feed-forward, and `3.2%` drifted heavy-hex readout erases the crossing completely: `STOP_NO_CROSSING_AT_TESTED_ROUNDS`). Single-distance ($d=3$) scope strictly enforced.

### v6.2.2 Follow-On Verification Closure (2026-11-24)
- **`QEC-CARDC-2` (`SUPPORTED as SIMULATED`):** Closed the documented `|2>` transmon leakage omission in Card C via exact `3^5 = 243`-state 5-qutrit Kraus + Liouville simulation (`experiments/qec_card_c/qec_leakage_lru.py`, `<= 2.08e-16` agreement vs full density matrix, `0.0` deviation vs 2-level Card C at `p_leak = 0`), establishing a `2.74x` unmitigated slope inflation at `p_leak,CZ = 1.2e-3` (`0.000691 -> 0.001896`) and `78.1%` recovery (`0.000955`) under `FULL_DATA_AND_ANCILLA_LRU` (`DQLR`).
- **`ECHO-2` & `QEC-CARDC-3` (`OFFLINE_STAGED`, `0s` QPU):** Frozen write-ahead `.qpy` + SHA-256 staging manifests validated against live `ibm_fez.target` for `OSIRIS-ECHO-FEZ` (`240` PUBs, `61,440` shots, `sha256=538af9be...`) and `OSIRIS-QEC-REP3-FEZ` (`96` dynamic `measure+reset+if_test` PUBs on `[58,71,70,69,78]`, `49,152` shots, `sha256=f240d445...`), fail-closed below `25.0s` remaining QPU quota.
- **Manuscripts 7 & 8 (`papers/bt2t002_mul_and_nonabelian_qft/`, `papers/echo_and_card_c_qec_gates/`):** Macro-locked REVTeX 4.2 + standalone multi-page `%PDF-1.4` preprints (`5` pp and `6` pp) verified (`SYS-024`, `SHA256SUMS.txt`, and `zenodo_staging_manifest_nov23.json` `paper_count = 7`).

### v6.2.3 Live Hardware, Zenodo & Multi-Distance/LGT Closure (2026-11-25)
- **Zenodo Archival DOIs Published (`74` site deposits, `85` lineage DOIs):** `10.5281/zenodo.24520723` (`bt2t001_binary_tetrahedral_fez`), `10.5281/zenodo.24520738` (`bt2t002_mul_and_nonabelian_qft`), and `10.5281/zenodo.24520746` (`echo_and_card_c_qec_gates`).
- **`QEC-CARDC-3` Upgraded to `HARDWARE_ATTESTED` (`dc35d4inj2vs73dhmsog`, `ibm_fez`, `16s` QPU, `49,152` shots):** Both `Arm 4a` (space-time MWPM Pauli-frame) and `Arm 4b` (real-time `if_test` physical feed-forward) crossed below `Arm 1` unencoded and `Arm 2` no-correction at `R = 3` and `R = 5` across `3/3` blocks (`R = 5`: `Arm 4a = 0.0872`, `Arm 4b = 0.0980 < Arm 2 = 0.1230 < Arm 1 = 0.2770`; `Arm 4b < Arm 2` paired-$t = 9.01, p = 0.00604$).
- **`ECHO-2` Upgraded to `HARDWARE_ATTESTED` (`dc35e35s2omc738cvcug`, `ibm_fez`, `19s` QPU, `61,440` shots across `10` disjoint `T5` subgraphs):** `ECHO_PROBE_16CZ` selected the `#1` oracle-best placement (`pid = 1`, `[5,6,7,8,17]`, `0.0` top-1 regret on `0.6594` composite) and beat calibration on `MIRROR_24CZ` (`+0.9030` vs `+0.7576`), while single-window unweighted composite measured `+0.7212` vs `+0.7818` (`H2 = REFUTED`; post-hoc `ECHO/ID_0CZ` reached `+0.9030` composite and `+0.9515` on `BT2T_72CZ`).
- **`QEC-CARDC-4` (`OSIRIS-QEC-MULTIDISTANCE`, `0s` QPU):** Closed the `d=3`-only limitation across `d = 3, 5, 7` (`p_th^surf = 1.124%`; `H2-1` surface `Lambda_3->5 = 6.96, Lambda_5->7 = 13.78`; `FEZ_DRIFTED_SPAM` SPAM-debiased `Lambda_3->5^diff = 1.023 [0.74, 1.33]`).
- **`BT2T-6` (`OSIRIS-BT2T-003`, `0s` QPU):** Verified 5Q gauge-fixed Kogut-Susskind `2T` Strang Trotter dynamics (`slope = 2.000008`, `209 CZ` all-to-all / `350 CZ` heavy-hex `T5`, `104x-178x` fewer `CZ` than unfixed 20Q/25Q 4-link circuits with `2.27e-15` gauge-invariant subspace agreement).
| `BT2T-7` | `OSIRIS-BT2T-003-FEZ` live 40-CZ magnetic plaquette character-sector trajectory (`r=0.9976` Center, `r=0.9979` Hurwitz, `0.8503` vs `0.0417` `+80.86 pp` structured-over-generic) & 2-window `OSIRIS-ECHO` drift catch (`rho_echo=+0.9483` vs `rho_cal=+0.4545`, `0.0000` regret, catching Q7 `31.12%` SPAM excursion on `pid=1`) | `SUPPORTED (H1, H2, H3)` (`HARDWARE_ATTESTED`, job `dc38nreq4uuc73a6g0k0`, DOI `10.5281/zenodo.24525501`) |
| `QEC-CARDC-5` | `OSIRIS-QEC-REP5-FEZ` write-ahead 72-PUB (`36,864` shots) 9Q `[[5,1,5]]` vs 5Q `[[3,1,3]]` dynamic MCMR + `if_test` distance-scaling `Lambda_{3->5}` bundle on `[120,121,122,123,136,143,142,141,140]` | `STAGED (WRITE-AHEAD LOCKED)` (`OFFLINE_STAGED`, `qpy_sha256 = da8061639e8eaa7e...`, `0s` QPU) |
| `QEC-CARDC-5 (LIVE)` | `OSIRIS-QEC-REP5-FEZ` live 72-PUB (`36,864` shots, `12s` QPU) 9Q `[[5,1,5]]` vs 5Q `[[3,1,3]]` dynamic MCMR distance-scaling (`Lambda_3->5 = 2.8333` at `R=3`, `1.5294` at `R=5`, `2.1209` mean, `0.0000` postselected error) | `SUPPORTED (H1, H2, H3)` (`HARDWARE_ATTESTED`, job `dc3bmlinj2vs73di19s0`, DOI `10.5281/zenodo.24531362`) |
| `QEC-CARDC-6` | `OSIRIS-QEC-SOFTMWPM` drift- & streak-aware soft-weight Edmonds Blossom decoder (restoring surface-code `Lambda_3->5 = 3.537` under `3.2%` hot ancilla drift and cutting `dc35d4inj2vs73dhmsog` slope `-73.91%` to `0.004883/round`, advancing crossing to `R=1`) | `SUPPORTED as SIMULATED + HARDWARE RE-SCORED` (`0s` new QPU, DOI `10.5281/zenodo.24531362`) |
| `BT2T-8` | `OSIRIS-BT2T-004` 10-qubit coupled 2-plaquette `2T` Kogut-Susskind ladder (`7,962,624x` reduction from `35Q`, `76-D` physical orbit subspace, `C_12 > 0`, `sigma_eff` `0.00136 -> 3.02580`, Strang slope `2.0001`, `116.14x` `CZ` compression vs 10Q dense QSD) | `SUPPORTED as SIMULATED/PROXY` (`37/37` checks, DOI `10.5281/zenodo.24531362`) |
