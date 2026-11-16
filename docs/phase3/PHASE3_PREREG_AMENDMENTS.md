# Phase 3 Preregistration Amendments (`ibm_marrakesh`)

Every amendment here is recorded before any Phase 3 data exist: no job has been submitted for either campaign. The locked artifacts in `docs/phase3/` are unchanged unless an amendment says otherwise.

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

CORRELATED-NOISE-01 is not submitted until its analysis amendment (A-3) is committed and pushed. A-3 changes the analysis only, not the circuits.

## Operational changes (no effect on circuits or analysis)

- One guarded submission path: `python -m phase3.execute_and_analyze_live --campaign tmap02|cn01|both`. The runner's `--step submit_*` delegates to it. Before submitting, it checks the QPY against the lock's SHA-256 and requires the remaining quota to be at least 1.3× the lock's estimate. (HP-001 on `ibm_fez` used 12 s against 10.1 s from the same estimate formula.) It never submits a campaign twice: with a submission receipt on disk, it retrieves the job by its ID.
- Each job's reported QPU usage is saved with its raw counts.
- The runner no longer overwrites existing prereg locks or the pre-lock calibration snapshot.
- Credentials come from the saved Qiskit account, `QISKIT_IBM_TOKEN`, or the JSON file named by `OSIRIS_IBM_KEY_FILE`; no local path is hardcoded.
