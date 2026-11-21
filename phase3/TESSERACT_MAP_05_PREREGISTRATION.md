# Preregistration: TESSERACT-MAP-05

**Status:** published before any MAP-05 job. The selection rule, arms, contrasts and analysis are fixed by `phase3/tesseract_map_05.py`. Its SHA-256 is in this Zenodo record. Locks are produced after each probe and published with the results.

**Follows:** TESSERACT-MAP-04 (doi:10.5281/zenodo.24407001). There, ranking 40-qubit paths by probe readout helped on 1 of 3 devices (`ibm_marrakesh` +25.0 pp, `ibm_fez` −3.6 pp, `ibm_kingston` −0.25 pp). After readout mitigation, the calibration-ranked path was better on all three.

## Hypothesis

Keep IBM's calibration ranking. Only remove qubits whose readout is grossly faulty at probe time: probe mean assignment error > **0.15**. This should keep the calibration path's gate quality and avoid faults like q46 on `ibm_marrakesh`.

## Design (per backend: `ibm_marrakesh`, `ibm_fez`, `ibm_kingston`)

- **Probe job.** All-0 and all-1 on every qubit, 3 repeats each, 2,048 shots (6 PUBs).
- **Path rule.** Same as MAP-04 (`score_paths`): greedy search, 400 candidates, cost −Σlog(1−e_CZ) − Σlog(1−e_RO), CZ errors from the probe-time calibration snapshot.
- **Main job.** 10 shuffled blocks × 6 PUBs × 2,048 shots, using the MAP-02 chain circuit (39 CZ, optimisation level 3).

| Arm | Physical qubits |
| --- | --- |
| C1_CAL_BEST | best path, IBM readout (MAP-02 rule; = MAP-04 B2) |
| C2_CAL_EXCL | best path, IBM readout, qubits with probe error > 0.15 removed |
| C3_PROBE_BEST | best path, probe readout (= MAP-04 B1) |
| C4_STOCK | Qiskit stock level-3 layout |
| U0 / U1 | union all-0 / all-1 |

- **Endpoints.** F_cl to the exact four-string distribution. Readout-mitigated F_cl as in MAP-03/04.
- **Guards before submission.** Noiseless MPS check F_cl > 0.95 for every arm; quota ≥ 1.3 × the estimate.

## Contrasts

Per backend: paired t over 10 blocks, Holm across the four contrasts; exact Wilcoxon as a sensitivity check.

| ID | Contrast | Prediction |
| --- | --- | --- |
| M1 | C2 − C1 | > 0 where they differ: exclusion beats the plain calibration path |
| M2 | C2 − C3 | ≥ 0: exclusion is at least as good as probe ranking |
| M3 | C2 − C4 | reported |
| M4 | C3 − C1 | replication of MAP-04 L1 (sign per device) |

## Decision rules

- **M1 testable.** If C1 and C2 use the same 40 qubits on a backend (the calibration path contains no excluded qubit), M1 is *not tested* there, and this is reported.
- **Exclusion helps** on a backend if M1 > 0 with Holm p < 0.05.
- **Exclusion is safe** on a backend if M2 is not significantly negative.
- **Overall conclusion: "exclude-only beats both MAP-04 rules"** requires all three of the following:
  - exclusion helps on every backend where M1 is testable;
  - at least one such backend exists;
  - exclusion is safe on every backend.

## Budget

About 42 s of QPU time per backend, about 125 s in total, from the remaining Open Plan allocation (about 477 s).
