# Phase 3 Preregistration Amendments (`ibm_marrakesh`)

Each amendment was recorded before any data existed for the campaign it changes. A-1 and A-2 predate every Phase 3 job. A-3 was recorded after TESSERACT-MAP-02 had run (`TESSERACT_MAP_02_RESULTS.md`) but before any CORRELATED-NOISE-01 job was submitted. The locked artifacts in `docs/phase3/` are unchanged unless an amendment says otherwise.

| Artifact | SHA-256 |
| --- | --- |
| `tesseract_map_02_live_pubs.qpy` | `409dc7365405ed20bde035683ae022a09778a1c58e186edd64fe5077154d7bf9` |
| `tesseract_map_02_live_prereg_lock.json` (locked 2026-11-16T19:52:00Z) | `5f4a04c45933e046c0fff0574a426fd3a876ec5f19d808d69514374c2f931f1f` |
| `correlated_noise_01_live_pubs.qpy` | `158de03260c429c8170cd3e3b296ade8045881e385370f1774619344e57b9da1` |
| `correlated_noise_01_live_prereg_lock.json` (locked 2026-11-16T19:52:01Z) | `38ebfe7f7c007633281039bca551af963e8335db54e6e6915a1e3f5be4a64fd8` |

## A-1 — TESSERACT-MAP-02: fewer blocks and shots (sizes only; analysis unchanged)

| | Preregistered (`TESSERACT_MAP_02_RESOURCE_ESTIMATE.md`, `TESSERACT_MAP_02_PREREGISTRATION.md`) | Locked |
| --- | --- | --- |
| Blocks `B` | 15 | 10 |
| Shots per PUB | 4,096 | 2,048 |
| PUBs / shots | 135 / 552,960 | 90 / 184,320 |
| Jobs | 1, or 3 sequential jobs of 5 blocks | 1 |
| Paired t-test df | 14 | 9 |

Reason: the Open Plan quota (600 QPU-seconds per 28 days) has to cover both Phase 3 campaigns.

Consequence for power. The preregistered sample-size justification assumes a paired block-difference SD of `σ_d = 0.014` (1.1 pp drift plus 4,096-shot multinomial noise) and Holm's strictest per-test `α = 0.01`. Halving the shots raises `σ_d` to about `0.0165` (the multinomial variance doubles; drift is unchanged). Recomputed with the same non-central t power calculation as `compute_required_blocks`:

| | `B = 15`, 4,096 shots (prereg) | `B = 10`, 2,048 shots (locked) |
| --- | --- | --- |
| Power at MDE = 1.5 pp | 85.8 % | 41.5 % (57.5 % if `σ_d` stays 0.014) |
| MDE at 85 % power | 1.5 pp | 2.3 pp (2.0 pp if `σ_d` stays 0.014) |

Endpoints, contrasts, Holm correction and decision rules are otherwise unchanged. A non-significant contrast below about 2.3 pp is inconclusive, not evidence of no effect. In particular, the `C5 − C4` virtual-Z / A/A drift control can only rule out differences of roughly that size.

## A-2 — CORRELATED-NOISE-01 Stage 1: fewer blocks and shots; cohort re-selected at lock time

| | Preregistered (`CORRELATED_NOISE_01_HARDWARE_PLAN.md`, Stage 1 core) | Locked |
| --- | --- | --- |
| Blocks `B` | 8 | 5 |
| PUBs | 432 (54 per block) | 244 (4 two-qubit readout-calibration PUBs + 5 blocks × 48) |
| Shots per PUB | 4,200 | 2,048 |
| Circuit shots | 1,814,400 | 499,712 |
| Estimated QPU time | ≈ 325 s | 149.3 s |

Coupler cohort. The plan lists a cohort selected from an earlier calibration: training `(13,14)`, `(1,2)`, `(46,47)`, `(98,111)`; held out `(136,143)`, `(62,63)`, `(135,139)`, `(54,55)`, `(7,8)`, `(127,128)`. The lock re-ran the same selection rule (`select_disjoint_couplers_on_backend`: couplers at least 5 hops apart, the 4 with the lowest CZ error for training, the next 6 held out) on the 2026-11-16T19:52Z calibration snapshot. That gave training `(135,139)`, `(149,150)`, `(1,2)`, `(13,14)` and held out `(53,54)`, `(91,98)`, `(47,57)`, `(7,17)`, `(104,105)`, `(142,143)`.

CORRELATED-NOISE-01 is not submitted until its analysis amendment (A-3, below) is committed and pushed. A-3 changes the analysis only, not the circuits.

## A-3 — CORRELATED-NOISE-01 Stage 1: analysis amendment (circuits unchanged)

Recorded 2026-11-16T23:02Z, before any CORRELATED-NOISE-01 data existed. These are all unchanged, and the prereg lock file is byte-identical:
- the QPY
- the PUB schedule
- the coupler cohort
- the blocks and the shots

| File | What it is | SHA-256 |
| --- | --- | --- |
| `correlated_noise_01_A3_amendment.json` | The amendment record: the hashes of the lock, the QPY and the analysis code, the split, the models and the decision rules | `7fb094d7a347496527db32d6a2b668a0723e7806cb5cefdb35ce7129e40a6947` |
| `phase3/cn01_amendment_a3.py` | The amended analysis | `056256b734890b22056582ffb66636815db9478175a7c9e51ce8def5ee5e38c3` |
| `phase3/execute_and_analyze_live.py` | The runner, the shared counts-to-fidelity step, and the locked analysis | `e5e194bd67fd3e1b5db428c04ea04927ada292f231ddb968f47a6e7ff503091c` |
| `correlated_noise_01_A3_validation.json` | Validation on synthetic counts (`phase3/cn01_a3_validation.py`) | `c390ce57feec10271752c33f9f5c13163d3aeb5bd1c82c36d9af9e2ecfbd7b7b` |

The runner refuses to run CN01 unless these hashes match the files on disk. Results are reported in two parts:
- **Primary:** the A-3 analysis.
- **Secondary:** the analysis as locked, kept for comparison.

### Why

On synthetic counts from independent coloured dephasing with no spatial correlation, the locked analysis selects H3 ("non-Markovian memory") in 100% of replicates. It also passes its own held-out test and makes the H3 memory claim in 95–100% of them. That opens the hardware plan's Stage-2 gate on ordinary local 1/f noise, which every superconducting qubit has. There are three causes:

1. H0, the only null, is white noise. Coloured local noise decays differently under Ramsey, echo and DD, and any such difference favours H3; on the local 1/f truth, by thousands of BIC units.
2. The held-out test calibrates H0 from the Ramsey arm and H3 from the DD32 arm. The preregistration specifies "high-frequency echo calibration" for every model.
3. The Ramsey arms are measured in one quadrature, so a frame detuning looks like decay.

The locked models also place DD pulses at centred slots instead of where the circuits put them. That moves predictions by at most 0.01 pp, and the pair exposures are identical. It is corrected for exactness only.

### Changes

| | Locked | A-3 |
| --- | --- | --- |
| Null models | H0 (white) | H0; H0_1f (independent 1/f plus a white floor); H3_local (H3 without spatial correlation) |
| Selection | Any model beating H0 by ΔBIC ≥ 8 | A non-local model (H1, H2, H3, Apparatus) beating the best local model by ΔBIC ≥ 8 |
| Fit conditions | All 23 | 19. The four Ramsey arms are reported descriptively only. |
| Held-out calibration | H0 from `ramsey_8us`; H3 from `echo_dd32_8us` | Each model's white floor is solved to reproduce the coupler's `echo_dd32_8us` |
| Held-out score | Absolute Φ+ fidelity at the 7 offsets | Change from 0 ns at the 6 non-zero offsets, which is the preregistration's "sign and magnitude of ΔF_Bell(Δ)". Absolute levels become secondary. |
| Held-out comparison | H3 vs H0 | Selected candidate vs best local model |
| Train / held-out split | The 4 lowest-CZ-error couplers train | 4 couplers drawn by a generator seeded from the QPY's SHA-256. Training: `(149,150)`, `(1,2)`, `(53,54)`, `(7,17)`. Held out: `(135,139)`, `(13,14)`, `(91,98)`, `(47,57)`, `(104,105)`, `(142,143)`. |
| DD pulse timing in the models | Centred, wrapped | As the circuits run it |

Unchanged:
- **Counts-to-fidelity extraction.** It is now one shared function. On synthetic data, the locked analysis's JSON output is byte-identical before and after the refactor.
- **Statistics.** The GLS covariance, the ΔBIC gate of 8, the four mechanistic contrasts with Holm correction, and df = 5 on held-out couplers.

The held-out score is the one change made after the validation began. With absolute levels, A-3 confirmed correlated dephasing in only 4–6% of truly correlated replicates (table below), because each coupler's calibration error is common to every model and hides a stagger signal of about 1 pp. The new score is the target the preregistration states.

### Decision rules

- **Correlated dephasing supported:** H2 or H3 is selected, C_DFS > 0 with Holm p < 0.05, and the candidate beats the best local model on the held-out couplers (paired t, df = 5, p < 0.05).
- **Correlated non-Markovian memory supported:** all of the above with H3 selected, plus C_memory > 0 (Holm).
- **Spectator ZZ supported:** H1 is selected, C_spec > 0 (Holm), and the held-out test passes.
- **Pulse collision supported:** Apparatus is selected and S48 ≥ 0.75.
- **Kill criterion (preregistered):** H1, H2 or H3 is selected but the held-out test fails. At this size the held-out test confirms only about half of true correlated signals (below). After significant in-sample evidence, a failure therefore means "transfer not demonstrated at this size", not evidence that the correlation is absent.
- **Local noise only:** any other outcome. "Local non-white dephasing" is reported when the best local model beats H0 by ΔBIC ≥ 8. That is expected and is not a finding.

### Validation (synthetic counts; `correlated_noise_01_A3_validation.json`)

Each replicate is a complete 244-PUB count set in the locked bit layout, built with:
- per-coupler T1 from the lock-time calibration, with 15% jitter
- per-qubit readout error
- a per-block drift offset and per-PUB jitter, giving a median A/A RMS of 0.72–0.81 pp
- 2,048 shots per PUB

Both analyses run on every replicate. The seeds are deterministic, and a second full run on the final code reproduced every rate.

| Truth | Replicates | Expected C_DFS, pp (range) | Locked: H3 selected | Locked: Stage-2 gate | Locked: H3 memory claim | A-3: any non-local claim (95% CI) | A-3: correlated dephasing supported |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| White (H0) | 100 | 0 | 0% | 0% | 0% | 0% (0–4) | 0% |
| Local 1/f | 100 | 0 | 100% | 100% | 100% | 0% (0–4) | 0% |
| Local 1/f, Ramsey detuned ±15 kHz | 100 | 0 | 100% | 99% | 96% | 0% (0–4) | 0% |
| Local OU memory, τ_c 1–4 µs | 100 | 0 | 100% | 95% | 95% | 0% (0–4) | 0% |
| Correlated Markovian (H2), ρ 0.5–0.7 | 50 | 2.70 (2.18–2.97) | 0% | 0% | 0% | 56% (42–69) | 56% |
| Correlated OU, τ_c 0.25–0.35 µs | 50 | 1.66 (1.37–2.00) | 100% | 100% | 100% | 44% (31–58) | 44% |
| Correlated OU, τ_c 1–4 µs | 50 | 0.04 (0.02–0.05) | 100% | 96% | 96% | 0% (0–7) | 0% |

| Truth | In-sample evidence (H2/H3 selected and C_DFS > 0, Holm) | Held-out pass, stagger change (primary) | Held-out pass, absolute levels | Correlated dephasing supported |
| --- | ---: | ---: | ---: | ---: |
| Local truths (all four) | 0% | 1–3% | 0–21% | 0% |
| Correlated Markovian (H2) | 100% | 56% | 4% | 56% |
| Correlated OU, τ_c 0.25–0.35 µs | 100% | 44% | 6% | 44% |
| Correlated OU, τ_c 1–4 µs | 0% | 0% | 22% | 0% |

Acceptance criteria, recorded in the validation module before the full run:
- **No non-local claim on any local truth in more than 5% of replicates.** Passed: 0 of 400.
- **At least 80% power on each fast correlated truth.** Not met: 56% and 44%. In-sample evidence is 100% on both. The held-out confirmation at this size is the limiting step, and its power is limited by two things:
  - each held-out coupler's stagger rise has a standard error of 0.15–0.6 pp at 5 blocks × 2,048 shots;
  - qubit v's own dephasing rate is never measured, so a predicted rise can be off by tens of percent.

The locked analysis has no power on the H2 truth. Its result on fast correlated OU (100%) carries no information, because it gives the same verdict without correlation.

The alternative truths were revised once before the full run, and that is disclosed in the validation module. A smoke test of three datasets, followed by a calculation of each truth's expected C_DFS, showed that slow correlated OU noise is undetectable by design. It is kept as a descriptive truth, and two fast truths carry the power criterion.

### What CN01 Stage 1 can and cannot show at this size

- **Slow correlated noise:** with a correlation time above about 1 µs, DD16 refocuses it. Its expected Φ+/Ψ+ asymmetry is below 0.1 pp, so no analysis of these circuits can detect it.
- **Fast correlated dephasing:** dephasing faster than the 512 ns DD slot, at the simulated strengths, is detected in-sample essentially always. It is confirmed on held-out couplers about half the time.
- **A null result:** "no non-local model selected" bounds only fast correlated dephasing. It says nothing about slow correlated noise.

## Operational changes (no effect on circuits or analysis)

- One guarded submission path: `python -m phase3.execute_and_analyze_live --campaign tmap02|cn01|both`. The runner's `--step submit_*` delegates to it. Before submitting, it checks the QPY against the lock's SHA-256 and requires the remaining quota to be at least 1.3× the lock's estimate. (HP-001 on `ibm_fez` used 12 s against 10.1 s from the same estimate formula.) It never submits a campaign twice: with a submission receipt on disk, it retrieves the job by its ID.
- Each job's reported QPU usage is saved with its raw counts.
- The runner no longer overwrites existing prereg locks or the pre-lock calibration snapshot.
- Credentials come from the saved Qiskit account, `QISKIT_IBM_TOKEN`, or the JSON file named by `OSIRIS_IBM_KEY_FILE`; no local path is hardcoded.
- Added after TESSERACT-MAP-02 ran:
  - **Usage polling:** the runner waits for IBM to finalise a job's QPU usage before recording it. `job.usage()` reads 0 until then.
  - **Account identifiers:** the runner does not save the IBM Cloud instance CRN or plan ID.
