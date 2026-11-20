# Preregistration: TESSERACT-MAP-04

**Status:** written and published before any MAP-04 job. Selection rule, arms, contrasts and analysis code are fixed by `phase3/tesseract_map_04.py` (SHA-256 recorded in the Zenodo record). Circuits are compiled after the probe, by that code, and their lock is published with the results.

**Follows:** TESSERACT-MAP-02 (doi:10.5281/zenodo.24406426) and TESSERACT-MAP-03 (doi:10.5281/zenodo.24406525). In MAP-03, a qubit on the calibration-ranked best path read |0⟩ as 1 in 54% of shots while IBM's calibration listed about 1%, and the MAP-02 layout advantage reversed.

## Question

Does a few-second readout probe, run just before the experiment, choose better physical qubits than IBM's published calibration, and better than Qiskit's stock level-3 layout? Does the answer hold across three Heron r2 processors?

## Design (per backend)

- **Backends:** `ibm_marrakesh` is primary. `ibm_fez` and `ibm_kingston` are replications, analysed identically and reported separately.
- **Probe job:** prepares all-|0⟩ and all-|1⟩ on every qubit, 3 repeats each, 2,048 shots (6 PUBs). A qubit's probe readout error is the mean of p(1|0) and p(0|1).
- **Selection rule:** MAP-02's path rule, reimplemented as `score_paths` with an explicit readout term:
  - greedy DFS; 400 candidate 40-qubit simple paths;
  - cost = −Σ log(1 − e_CZ) − Σ log(1 − e_RO);
  - qubits with e_RO ≥ 0.45 excluded;
  - CZ errors from IBM's calibration snapshot taken at probe time.
  - *Probe paths* use the probe's e_RO; *calibration paths* use IBM's reported e_RO.
- **Main job:** 10 shuffled blocks × 6 PUBs × 2,048 shots. Every chain arm is the MAP-02 chain circuit (θ = 51.7°, 39 CZ, optimisation level 3, no routing):

| Arm | Physical qubits |
| --- | --- |
| B1_PROBE_BEST | lowest-cost path under probe readout |
| B2_CAL_BEST | lowest-cost path under IBM readout (the MAP-02 rule) |
| B3_STOCK | Qiskit's stock level-3 layout (free) |
| B4_PROBE_MED | median-cost path under probe readout |
| U0 / U1 | union all-0 / all-1 on every qubit any arm uses |

- **Endpoint:** classical fidelity F_cl to the exact four-string ideal distribution, as in MAP-02/03. Readout-mitigated F_cl uses the tensored inverse with rates from U0/U1, as in MAP-03.
- **Before submission:** every arm in block 0 must give F_cl > 0.95 in a noiseless matrix-product-state simulation, or nothing is submitted. Quota must be at least 1.3 × the estimate.

## Preregistered contrasts (per backend; paired t over 10 blocks, df = 9; Holm across the four; exact Wilcoxon as sensitivity)

| ID | Contrast | Prediction |
| --- | --- | --- |
| L1 | B1 − B2 | > 0: the probe path beats the calibration path |
| L2 | B1 − B3 | > 0: the probe path beats the stock layout |
| L3 | B1 − B4 | > 0: path quality matters under probe ranking (replicates MAP-02's path effect) |
| L4 | (B1 − B2) after readout mitigation | ≈ 0 if the L1 advantage is readout alone |

## Decision rules

- **Probe selection helps** on a backend if L1 > 0 with Holm p < 0.05.
- If B1 and B2 select the same 40 qubits, L1 and L4 are not tested on that backend, and this is reported.
- **Generalises:** probe selection helps on at least 2 of the 3 backends. If it helps only on `ibm_marrakesh`, it is a device-specific finding.
- **L4 interpretation:**
  - L4 not significant while L1 is: the advantage is readout.
  - L4 significant with the same sign as L1: part of the advantage is not readout (for example, qubits correlated with bad readout are also worse otherwise).
- **L2 and L3:** reported with sign and significance.

## Budget

Each backend uses about 4 s for the probe and 37 s for the main job, roughly 125 s across all three. This comes from a fresh 600 s Open Plan allocation.

## Order of operations (recorded in the results)

1. This preregistration is published on Zenodo.
2. For each backend, in turn: probe job, then lock (path selection, compile, noiseless check, hashes), then main job.

The probe and main jobs run minutes apart, as a practical pre-job probe would.
