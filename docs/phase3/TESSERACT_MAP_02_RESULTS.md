# TESSERACT-MAP-02: hardware results (`ibm_marrakesh`)

**Status:** Hardware-observed. These results are analysed exactly as preregistered (`phase2/tesseract_map_02/TESSERACT_MAP_02_PREREGISTRATION.md`), with the sizes changed by amendment A-1 (`PHASE3_PREREG_AMENDMENTS.md`). The claims register is not changed here. Any claim built on these results needs the owner's review.

## Run

| | |
| --- | --- |
| Job | `dbto408n26qs739tnrj0` on `ibm_marrakesh`, SamplerV2, job mode |
| Created / running / finished (IBM) | 2026-11-16 22:21:53Z / 22:21:56Z / 22:23:02Z |
| Size | 10 blocks × 9 circuits = 90 PUBs × 2,048 shots (184,320 shots) |
| QPU time charged | 53 s (0.8689 RU), against the lock's 55.1 s estimate. Open Plan quota went from 331 s to 278 s. |
| Prereg lock / QPY | `5f4a04c4…` / `409dc736…`. Both were pushed before the job was submitted (see the amendments file for full hashes). |

`tesseract_map_02_live_raw_counts.json` gives `job_usage_seconds: 0.0`. The runner read `job.usage()` before IBM had finalised the charge. The finalised figure, re-queried afterwards, is in `tesseract_map_02_job_usage.json`. The runner now polls until IBM marks a job's usage complete.

Two fields were removed from the raw-counts file's `post_job_usage` before it was committed: `instance_id` and `plan_id`. They identify the IBM Cloud account. The file's `redacted_fields` key lists them. No other field changed, the counts included. The runner no longer saves these two fields.

## Arms

Each arm's F_cl and P(S4) is the mean over 10 blocks, with the SD across blocks in parentheses. The predictions come from the Phase 2 FakeMarrakesh surrogate simulation (`TESSERACT_MAP_02_EXPECTED_OUTCOMES.md`).

| Arm | CZ | Depth | F_cl | Predicted F_cl | P(S4) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `C1_STAR_FREE_OPT3` | 163 | 213 | 0.0160 (0.0019) | 0.0171 | 0.0256 (0.0028) |
| `C2_STAR_BESTPATH_OPT3` | 79 | 91 | 0.0723 (0.0073) | 0.2594 | 0.0748 (0.0076) |
| `C3_STAR_BESTPATH_OPT1` | 147 | 179 | 0.0927 (0.0088) | 0.1072 | 0.1286 (0.0063) |
| `C4_CHAIN_BESTPATH_OPT1` | 39 | 86 | 0.2173 (0.0095) | 0.4048 | 0.2475 (0.0107) |
| `C5_CHAIN_BESTPATH_OPT3` | 39 | 66 | 0.2150 (0.0055) | 0.4054 | 0.2381 (0.0069) |
| `C6_CHAIN_MEDPATH_OPT3` | 39 | 66 | 0.1096 (0.0048) | 0.2783 | 0.1246 (0.0043) |
| `C7_CHAIN_FREE_OPT3` | 39 | 66 | 0.1580 (0.0085) | 0.4404 | 0.1716 (0.0080) |

The readout calibration was a union SPAM calibration over 96 qubits. It found a mean assignment error of 2.47%.

## Preregistered contrasts (primary endpoint F_cl)

The tests are two-sided paired t-tests over 10 blocks (df = 9), with Holm correction across the five contrasts. The Wilcoxon signed-rank check was preregistered as a sensitivity analysis. The locked pipeline did not produce it, so it was computed afterwards from the same per-block values (`phase3/tmap02_supplement.py` → `tesseract_map_02_live_sensitivity.json`). It agrees with the t-tests on all 10 contrasts across both endpoints.

| Contrast | Difference (pp) | 95% CI (pp) | Holm p | Significant | Wilcoxon Holm p |
| --- | ---: | ---: | ---: | :---: | ---: |
| 1. C5 − C2: rewrite + SWAP removal, opt 3 | +14.27 | [+13.52, +15.01] | 4.6e-11 | yes | 0.0098 |
| 2. C4 − C3: rewrite + SWAP removal, opt 1 | +12.47 | [+11.52, +13.42] | 8.4e-10 | yes | 0.0098 |
| 3. C5 − C6: best vs median path at 39 CZ | +10.54 | [+9.95, +11.12] | 6.2e-11 | yes | 0.0098 |
| 4. C5 − C4: virtual-Z A/A drift control | −0.24 | [−0.88, +0.41] | 0.43 | no | 0.32 |
| 5. C5 − C7: greedy best path vs VF2Layout | +5.70 | [+5.13, +6.28] | 6.6e-09 | yes | 0.0098 |

A-1 warned that, at 10 blocks × 2,048 shots, contrasts below about 2.3 pp could be inconclusive. That limit did not bind:
- **Variability:** the observed SDs of the paired block differences were 0.008–0.013, against the 0.0165 assumed in A-1.
- **Precision:** every 95% CI has a half-width under 1 pp.

## Decision matrix (`TESSERACT_MAP_02_EXPECTED_OUTCOMES.md` §3)

| Preregistered pattern | Outcome |
| --- | --- |
| `C5 > C2` and `C5 > C6`, both Holm p < 0.05 | **Met.** SWAP elimination and physical path quality both contribute significantly. The matrix also says they combine additively, which this design does not test: there is no interaction contrast. |
| `C7 ≥ C5 ≫ C1` | **Not met.** C7 is 5.70 pp below C5. The simulation predicted that stock `VF2Layout` on the degree-reduced circuit would beat the greedy best path (0.4404 vs 0.4054). This run shows the opposite. |
| `C5 − C2` small while SPAM flags 1–2 outlier qubits | **Not applicable.** C5 − C2 is +14.27 pp. |
| `C5 − C4 > +1.5 pp`, Holm p < 0.05 | **Not met.** The A/A control passes: −0.24 pp, with the 95% CI excluding drift larger than 0.9 pp in either direction. |

## Secondary endpoint P(S4)

All five contrasts are significant after Holm correction. Contrasts 1, 2, 3 and 5 match the primary endpoint in sign and size: +16.32, +11.88, +11.35 and +6.65 pp.

The A/A control (contrast 4) is −0.94 pp, 95% CI [−1.81, −0.06], with Holm p = 0.038 (Wilcoxon 0.037). This is a small difference, below the 1.5 pp decision threshold, and opposite in sign to the hypothesis that compressing the single-qubit layers helps. The two arms have identical `sx` and `cz` counts (81 and 39) and differ by 40 virtual RZ gates. The preregistration says they share one physical schedule, but that was not re-checked against the backend here. This is a secondary-endpoint result and triggers no decision-matrix row.

## Descriptive observations (not preregistered; no tests)

- **Absolute fidelities:** they are about half of what the surrogate simulation predicted for every arm except C1: C5 scored 0.215 against a predicted 0.405. C1 matched its prediction (0.016 against 0.017).
- **C2 vs C3:** C2 (opt 3, 79 CZ) scored below C3 (opt 1, 147 CZ), 0.072 against 0.093, and was lower in all 10 blocks. The simulation predicted the opposite (0.259 against 0.107).
- **Readout on two qubits:** the union SPAM circuits measured mean assignment errors of 22.3% on qubit 42 and 12.1% on qubit 41. IBM's reported readout errors for these qubits were 1.0% and 1.3% in both the lock-time and the execution-time snapshots. Both qubits are in the C6 (median path) and C7 (VF2) layouts and not in the best path. This could account for part of the C5 − C7 gap, but that was not tested.

## Artifacts (`docs/phase3/`)

| File | SHA-256 |
| --- | --- |
| `tesseract_map_02_submission_receipt.json` | `277eaa51f684653ee0ae69c3fde0ec641825365eb79aae5370452099a457436c` |
| `tesseract_map_02_live_raw_counts.json` | `957c1c94e8fa9c514ec0d515fe307c21e6aa9ee68cce19c769d463e9cf132270` |
| `tesseract_map_02_live_analysis.json` | `0a53301c0a67889cc18ee942856f132c62ddd35a7accbcd2b9487386cc326185` |
| `tesseract_map_02_job_usage.json` | `ad795c81dc9a155fdd752e4b054da8ec7f81b5048a8a74ee2f418a8dcbef3c34` |
| `tesseract_map_02_live_sensitivity.json` | `955a42eac7dad94c68900c2826d0267a9e7b1677f0dc8917f89f73e553239c10` |
