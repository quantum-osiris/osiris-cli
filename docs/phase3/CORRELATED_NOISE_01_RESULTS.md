# CORRELATED-NOISE-01 Stage 1: hardware results (`ibm_marrakesh`)

**Status:** Hardware-observed. Two analyses are reported.
- **The analysis as locked:** the only one published before the data.
- **Amendment A-3:** hashed before the job but published after it (see [Provenance](#provenance)).

The owner decides which analysis is primary. Both reach the same verdict. The claims register is not changed here. Any claim built on these results needs the owner's review.

## Verdict

Neither analysis supports a non-local noise mechanism: no correlated dephasing, correlated non-Markovian memory, spectator ZZ or pulse collision. The Stage-2 gate (`CORRELATED_NOISE_01_HARDWARE_PLAN.md` §1) needs both ΔBIC ≥ 8 over H0 and a significant held-out improvement, so it is closed under both analyses. Stage 2 on `ibm_fez` does not run.

| | Analysis as locked | Amendment A-3 |
| --- | --- | --- |
| Model selected on the training couplers | H3, ΔBIC 10,263 over H0 | H0. No model beats the best local model. |
| Held-out test | H3 does not beat H0 (t₅ = −0.24, p = 0.82) | Not reached: no non-local model is selected |
| Outcome | Preregistered kill criterion: the transferable H3 claim is **falsified** | **Local noise only** |

The locked analysis's H3 selection is not evidence of memory. On A-3's synthetic validation it selects H3 in 100% of replicates of independent local 1/f or OU noise (`PHASE3_PREREG_AMENDMENTS.md`, A-3).

## Run

| | |
| --- | --- |
| Job | `dbtp67dav06s73cfbsm0` on `ibm_marrakesh`, SamplerV2, job mode |
| Created / running / finished (IBM) | 2026-11-16 23:34:53Z / 23:34:55Z / 23:37:44Z |
| Size | 244 PUBs × 2,048 shots (499,712 shots): 4 readout-calibration PUBs plus 5 blocks × 48. All 10 couplers are measured in every PUB. |
| QPU time charged | 139 s (2.3076 RU), against the lock's 149.3 s estimate. Open Plan quota went from 278 s to 139 s. |
| Prereg lock / QPY | `38ebfe7f…` / `158de032…` (full hashes in `PHASE3_PREREG_AMENDMENTS.md`). Both have been public since PR #15 was opened with them at 22:21:31Z, 73 minutes before the job. |
| Other jobs | IBM lists two jobs on the account since 21:00Z: this one and TESSERACT-MAP-02's. Nothing was resubmitted. |
| Counts check | The job was re-queried read-only. The saved counts match IBM's stored result for all 244 PUBs (`correlated_noise_01_job_usage.json`). |

Two fields were removed from the raw-counts file's `post_job_usage` before it was committed: `instance_id` and `plan_id`. They identify the IBM Cloud account. The file's `redacted_fields` key lists them, and nothing else changed. The owner keeps the unredacted original locally (SHA-256 `a459413cbd5faa001c7a18d4d71adb8801eb038d234b31316b23d69a375e6981`).

## Provenance

### Who ran it, and with what code

The owner submitted the job from a second local checkout of `main`, not with this branch's guarded runner.

In that checkout, `phase3/live_hardware_runner.py` had `REPO_ROOT` hard-coded to this checkout. So the run read the QPY and the lock from here, and wrote its outputs here. A later commit there restored the relative path. That one-line change is the only difference from the version that ran. The compiled module left by the run still contains the hard-coded path, which is how this was established.

That runner:
- **Circuits:** submitted the locked QPY against the locked lock. The receipt records both hashes, and they match.
- **Analysis:** ran only the analysis as locked. Its `analyze_correlated_noise_01_live_counts` differs from the version committed with the lock (`494558ef`) only in two unused local variables, removed by a lint cleanup. This branch's pinned runner reproduces its output exactly (`tests/test_phase3_cn01_hardware_artifact.py`).
- **Checks:** does not check for A-3, so it ran without it. It requires 120 s of quota rather than 1.3× the estimate; that margin (194 s) was met anyway.
- **Side effects:** rewrote `tesseract_map_02_live_analysis.json` with bytes identical to the committed file, and saved the two account identifiers removed above.

### When A-3 was recorded

| Event (2026-11-16, UTC) | Time | Evidence |
| --- | --- | --- |
| CN01 lock and QPY committed (`494558ef`) | 22:02:41 | git |
| PR #15 opened, containing the lock and QPY | 22:21:31 | GitHub |
| A-3 code and validation written | 22:45–23:01:53 | Local file times |
| A-3 record written (`correlated_noise_01_A3_amendment.json`) | 23:02:15 | The record's `recorded_utc`, its file time, the authoring session's log |
| PR #15 merged, without A-3 | 23:02:29 | GitHub |
| A-3 text in `PHASE3_PREREG_AMENDMENTS.md` last edited | 23:02:56 | Local file time |
| CN01 job created | 23:34:53 | IBM |
| A-3 first committed | After the job (this PR) | git |

A-2 says CN01 "is not submitted until its analysis amendment (A-3, below) is committed and pushed." That condition was not met.

The evidence that A-3 predates the data is local only, and nobody else can check it. The record pins the lock, the QPY, the four analysis files and the validation by hash, and all of them still match. That shows the A-3 analysis run here is the one the record describes. It cannot show when the record was written. A reader who does not accept the local evidence should treat A-3 as a post-hoc analysis.

## Data (descriptive)

These are raw fidelities: the mean over 10 couplers × 5 blocks, with the SD across couplers in parentheses.

| Condition | Mean (SD across couplers) |
| --- | --- |
| Echo, N = 1, at 2 / 4 / 8 / 16 µs | 0.975 (0.012) / 0.963 (0.016) / 0.940 (0.027) / 0.904 (0.042) |
| DD at 8.192 µs, N = 8 / 16 / 32 | 0.940 (0.024) / 0.940 (0.026) / 0.946 (0.026) |
| DD32 at 16.384 µs | 0.902 (0.043) |
| Ramsey at 2 / 4 / 8 / 16 µs | 0.960 (0.018) / 0.927 (0.058) / 0.841 (0.172) / 0.723 (0.220) |
| Φ+ under DD16, at the 7 offsets from 0 to 256 ns | 0.7216–0.7241 (0.081–0.084) |
| Ψ+ under DD16, at 0 / 256 ns | 0.7243 (0.082) / 0.7220 (0.084) |
| Φ+ with the spectator in \|+⟩, at 0 / 256 ns | 0.7190 (0.085) / 0.7191 (0.083) |

- **Stagger:** Φ+ is flat across the offsets to within 0.25 pp.
- **Ramsey:** it decays much faster than echo and varies far more between couplers. At 16 µs it ranges over 0.18–0.90, against 0.82–0.95 for echo. That points to low-frequency local noise, a frame detuning, or both. The Ramsey arm measures one quadrature, so it cannot tell these apart.
- **A/A sentinel** (the 0 ns Φ+ circuit, run twice per block): the RMS difference is 1.15 pp raw (1.32 pp readout-mitigated) over 50 coupler-blocks, with a mean of +0.12 pp. This matches the 1.03–1.11 pp drift floor found in the earlier ZZ-Ramsey Stage 1.
- **Readout:** coupler (1,2) has a mean two-qubit assignment fidelity of 0.89. The other nine are 0.95–0.98.

## Mechanistic contrasts (identical in both analyses)

Each contrast is a mean across the 10 couplers, tested with a t-test (df = 9) and Holm correction across the four.

| Contrast | Needed for | Mean (pp) | 95% CI (pp) | Holm p |
| --- | --- | ---: | ---: | ---: |
| C_DFS: the Φ+ rise from 0 to 256 ns, minus the Ψ+ rise | H2, H3 (> 0) | +0.43 | [−0.13, +1.00] | 0.47 |
| C_spec: the Φ+ rise with the spectator in \|+⟩, minus the rise with it in \|0⟩ | H1 (> 0) | −0.20 | [−0.50, +0.11] | 0.53 |
| C_stagger: Φ+ at 256 ns minus Φ+ at 0 ns | — | +0.20 | [−0.22, +0.62] | 0.61 |
| C_mem: DD32 minus echo, at 16.384 µs | H3 (> 0) | −0.14 | [−2.25, +1.97] | 0.89 |

No contrast is significant. The step fraction S48 is 0.62, below the 0.75 needed for the pulse-collision (Apparatus) model. With a stagger rise of only 0.2 pp, that ratio is mostly noise.

## Analysis as locked

The training couplers are the four with the lowest CZ error: (135,139), (149,150), (1,2), (13,14). The fit uses all 23 conditions.

| Model | k | χ² | BIC |
| --- | ---: | ---: | ---: |
| H3 | 8 | 148.6 | 173.7 |
| H1 | 6 | 2,482.6 | 2,501.4 |
| H2 | 7 | 10,361.5 | 10,383.4 |
| Apparatus | 6 | 10,402.6 | 10,421.4 |
| H0 | 5 | 10,420.9 | 10,436.6 |

H3 is selected, by ΔBIC = 10,263 over H0. Two of its fitted parameters sit at their lower bounds: the spatial correlation at −0.42, and qubit v's relaxation time at 30 µs.

H0 is white noise, so it cannot fit the fast Ramsey decay together with echo and DD. On local 1/f noise, A-3's validation found that this alone gives H3 an advantage of thousands of BIC units.

Held-out test (6 couplers, df = 5): the mean absolute-level RMSE is 0.130 for H0 and 0.143 for H3 (paired t = −0.24, p = 0.82). Per coupler, the errors run from 1 to 32 pp and differ widely between couplers. H3 does not beat H0.

Under the preregistered kill criterion (`CORRELATED_NOISE_01_PREREGISTRATION.md` §2.4), H3 wins in-sample by ΔBIC ≥ 8 but fails the held-out test. So "the transferable quantum-control claim is FALSIFIED and classified as non-portable local calibration heterogeneity."

A-3's validation qualifies this. On synthetic local noise, this analysis selected H3 every time, and its held-out test usually passed: the Stage-2 gate opened in 95–100% of replicates. Here the held-out test failed. Neither outcome says anything about spatial correlation.

## Amendment A-3

The training couplers were drawn by a generator seeded from the QPY's SHA-256: (149,150), (1,2), (53,54), (7,17). The four Ramsey arms are left out of the fit, which uses 19 conditions.

| Model | Local | k | χ² | BIC |
| --- | :---: | ---: | ---: | ---: |
| H0 | yes | 5 | 100.3 | 115.05 |
| H1 | no | 6 | 97.6 | 115.27 |
| H2 | no | 7 | 95.2 | 115.83 |
| Apparatus | no | 6 | 100.3 | 118.00 |
| H0_1f | yes | 6 | 825.6 | 843.2 |
| H3_local | yes | 7 | 825.4 | 846.0 |
| H3 | no | 8 | 825.4 | 848.9 |

H0 is the best local model, and no non-local model beats it. The gate is ΔBIC ≥ 8, and the best non-local model (H1) is 0.22 behind H0. The decision is **local noise only**. "Local non-white dephasing" is not reported, because H0_1f does not beat H0.

No decision rule reaches the held-out stagger-change test. For the record, H0's per-coupler RMSE of the stagger change on the six held-out couplers is 0.3–0.9 pp.

**Bounds diagnostic** (not preregistered: `phase3/cn01_a3_bound_diagnostic.py` → `correlated_noise_01_A3_bound_diagnostic.json`). H0_1f, H3_local and H3 fit far worse than H0, although each of them contains H0. The cause is a bound:
- A-3 gave its two new local models the bounds of the Phase 2 H3 model, which cap each qubit's white dephasing floor at 40 kHz. The fitted v-qubit floor here is 57 kHz.
- H0 refitted with a 40 kHz cap gives χ² = 825.4, which is exactly what the three capped models reach.
- With the cap raised to 60 kHz or 200 kHz, all three collapse onto H0: the extra noise amplitude goes to 0 and χ² to 100.3. They then lose on the parameter penalty.
- At either raised cap the selection stays H0, and the best non-local model (H2, BIC 115.6) stays far short of the gate.

So the cap changes no decision, but the three capped BICs reflect the cap, not the data.

## What this does and does not show

- **Fast correlated dephasing** (faster than the 512 ns DD slot): not seen. C_DFS's 95% interval, [−0.13, +1.00] pp, excludes the asymmetries A-3's validation simulated for fast correlated noise (1.4–3.0 pp). A-3 found in-sample evidence on every one of those simulated replicates.
- **Slow correlated noise** (correlation time above about 1 µs): not tested. DD16 refocuses it, and its expected Φ+/Ψ+ asymmetry is below 0.1 pp.
- **Spectator ZZ and pulse collision:** not seen. See C_spec and S48 above.
- **Correlated memory:** not seen, but C_mem's interval is wide ([−2.25, +1.97] pp).
- **Ramsey decay:** descriptive only, for the reasons under Data.

## Artifacts (`docs/phase3/`)

| File | SHA-256 |
| --- | --- |
| `correlated_noise_01_submission_receipt.json` | `4a940172cf64fc6915a13e17913090992166b9c7f7b33a69977c99021fdd48bd` |
| `correlated_noise_01_live_raw_counts.json` (redacted) | `585bb1bc1d48fe3ad2f7b18e92930442dae11ada22988d65b61972b39cc10b9d` |
| `correlated_noise_01_live_analysis.json` (analysis as locked, written by the run) | `22116f59ae4ef1753767b8551bf293506a67016eda6744af2b9cfafaac718688` |
| `correlated_noise_01_live_analysis_A3.json` | `b9d4640222bffd818a095a1c2b1b34a88fe02f0f8f221f86ca05ba485dfb3d24` |
| `correlated_noise_01_job_usage.json` | `f1109abdc2d9c96694b3da6c640bce2f8a36416927919013880fb1df525f1d85` |
| `correlated_noise_01_A3_bound_diagnostic.json` | `faa07ec141e352303da2ac53e2b137925f27a3bd0292be508cca2baf6c104070` |

`tests/test_phase3_cn01_hardware_artifact.py` checks three things:
- the files agree with each other and with the lock;
- the raw counts carry no account identifiers;
- both analyses reproduce from the archived counts.
