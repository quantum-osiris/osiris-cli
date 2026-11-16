# Resource Estimate: `EXPERIMENT TESSERACT-MAP-02`

**Status:** `Proposed only` for QPU execution (**0 QPU seconds spent in Phase 2**; all estimates verified via offline compilation on `FakeKingston` and `FakeMarrakesh`).

---

## 1. Shot, Circuit, and QPU Time Budget

| Parameter | Value | Basis / Notes |
|---|---|---|
| Active circuit width ($N$) | `40` qubits | Matches `dansntf8gn2s739n32e0` (`ibm_kingston`) |
| Circuits per block | `9` (`7` test arms + `2` SPAM controls) | Randomized order within every block |
| Interleaved blocks ($B$) | `15` blocks | Powered at `85.8%` for `1.5 pp` MDE after 5-test Holm ($\alpha_1 = 0.01$) |
| Total PUBs (circuit executions) | `135` PUBs | Can be submitted as `3` sequential jobs of `5` blocks (`45` PUBs/job) to sample inter-job calibration/scheduling |
| Shots per PUB ($S$) | `4,096` shots | Multinomial SE $\le 0.0078$ per block; `61,440` shots/arm pooled |
| Total attempted shots | **`552,960` shots** | `135 × 4,096` |
| Archived empirical QPU rate | `0.30 ms / shot` (`18 s` for `60,000` shots in `dansntf8gn2s739n32e0`) | Includes Heron r2 active reset / repetition delay (`~250 µs` rep time) |
| Estimated total QPU execution time | **`~150–175 QPU seconds` (`~2.5–2.9 min`)** |Fits within a single 10-minute IBM Quantum OpenPlan monthly window or `< 3 min` Pay-As-You-Go runtime |
| Classical pre-compilation & post-analysis | `< 4.0 s` local CPU, `< 250 MiB` RAM | Verified in `test_tesseract_map_02.py` (`1.95 s`) |

---

## 2. Explicit QPU Submission Gate

Per Phase 2 Section 5 & Section 10 governance rules:
- **Current spend:** `0` QPU jobs, `0.0` IBM Resource Units.
- **Required before submission:** Explicit owner sign-off specifying target backend (`ibm_marrakesh` or `ibm_fez`), confirmation of available quota, and authorization to lock the pre-submission calibration snapshot SHA-256.
