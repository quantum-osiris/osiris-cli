# Preregistration: TESSERACT-MAP-03

**Status:** locked 2026-11-20T20:12:19Z; not yet run. This document, the lock file, the QPY and the analysis code are published on Zenodo before the job is submitted.

**Follows:** TESSERACT-MAP-02 (`ibm_marrakesh`, job `dbto408n26qs739tnrj0`; paper: doi:10.5281/zenodo.24406426).

## Questions

MAP-02 found that, at a fixed 39 CZ and depth 66, a 40-qubit chain on a calibration-ranked best path beat the same chain on a median path by 10.54 pp of classical fidelity F_cl. It also beat Qiskit's stock level-3 layout by 5.70 pp. Union readout calibration then found two qubits on the losing layouts (q41: 12.1%, q42: 22.3%) whose IBM-reported readout error was about 1%. MAP-02 left three questions open:

1. **Replication.** Does the best-vs-median path gap persist on the same physical qubits four days later?
2. **Readout share.** How much of that gap is readout error, and how much is gate, idle and crosstalk error?
3. **Additivity.** Is the path effect the same for the routed star circuit as for the SWAP-free chain? MAP-02 had no star/median-path cell.

## Design

One job on `ibm_marrakesh`: 10 blocks × 8 circuits × 2,048 shots (80 PUBs, 163,840 shots, estimated 49.5 s QPU). Circuit order is shuffled in each block with seed 20261120. Every arm uses optimisation level 3 and the MAP-02 logical circuits (θ = 51.7°, exact four-string ideal support).

| Arm | Logical | Physical qubits | Routing | CZ | Depth | Role |
| --- | --- | --- | --- | ---: | ---: | --- |
| A1_CHAIN_BEST | chain | MAP-02 best path (= C5's qubits) | none | 39 | 66 | replays C5 |
| A2_CHAIN_MED | chain | MAP-02 median path (= C6's qubits) | none | 39 | 66 | replays C6 |
| A3_CHAIN_C7REPLAY | chain | MAP-02 C7's 40 qubits, pinned | none | 39 | 66 | replays C7's qubits |
| A4_CHAIN_STOCK | chain | stock level-3 layout chosen today | stock | 39 | 66 | fresh stock layout |
| A5_STAR_MED | star | MAP-02 median path | SABRE | 79 | 91 | **new**: the missing 2×2 cell |
| A6_STAR_BEST | star | MAP-02 best path | SABRE | 79 | 91 | replays C2 |
| U0 / U1 | — | all 81 qubits used by any arm | — | 0 | — | union readout calibration |

The paths come from the published MAP-02 lock (SHA-256 `5f4a04c4…`), so path choice cannot be influenced by today's calibration or by the MAP-02 outcome. A4 is the one arm chosen on today's calibration. It includes q41 and q42, which IBM's 2026-11-20 calibration lists at 1.3% readout error.

## Endpoints

- **Primary:** F_cl, the classical (Bhattacharyya) fidelity to the exact four-string ideal distribution, per PUB. This is computed exactly as in MAP-02 (`score_tesseract_counts`).
- **Readout-mitigated F_cl (used in K2):** apply the tensored single-qubit inverse of the readout assignment matrix, evaluated only on the four ideal strings. Each qubit's p(1|0) and p(0|1) come from the union calibration pooled over all 10 blocks. Negative quasi-probabilities are clipped to 0. Quasi-probabilities can push mitigated F_cl slightly above 1; values are not clipped further.

## Preregistered contrasts

All five use per-block values in pp, a two-sided paired t-test over 10 blocks (df = 9) and Holm correction across the five at FWER 0.05. An exact Wilcoxon signed-rank test (Holm) is a sensitivity check.

| ID | Contrast | Question |
| --- | --- | --- |
| K1 | A1 − A2 (raw) | Does the path gap replicate? |
| K2 | (A1 − A2)_raw − (A1 − A2)_mitigated | How much of the gap is readout? |
| K3 | (A1 − A2) − (A6 − A5) (raw) | Is the path effect additive across chain and star? |
| K4 | A1 − A3 (raw) | Best path vs C7's qubits, replay |
| K5 | A1 − A4 (raw) | Best path vs today's stock layout |

**Descriptive (no test):** η = 1 − (mitigated gap / raw gap) for K1's gap, with a 10,000-sample block bootstrap 95% interval. Also reported: each arm against its MAP-02 counterpart, and per-qubit readout errors against IBM's reported values.

## Decision rules

- **K1:** Holm p < 0.05 and positive means the path gap replicated. A non-significant K1 means it did not replicate at this size.
- **K2:** given a replicated K1, a positive significant K2 means readout explains part of the gap.
  - η's interval entirely above 0.8: readout explains most of the gap.
  - η's interval entirely below 0.5: most of the gap is not readout.
  - Otherwise: the split is reported as an estimate only.
- **K3:** Holm p < 0.05 means the path effect differs between chain and star (not additive). Non-significant means additivity is not rejected at this size. It does not show additivity.
- **K4, K5:** sign and significance are reported, with no further decision attached.

## Expectations recorded before the run

These are from the dry run (`tesseract_map_03_dryrun.json`). No test depends on them.

- **Readout alone.** Readout noise alone, at MAP-02's measured per-qubit rates, gives F_cl = 0.382 on the best path and 0.261 on the median path, a 12.1 pp gap. That exceeds the 10.5 pp observed in MAP-02. If readout rates are unchanged, K2 should be large and η near 1. If q41/q42 have recovered, K1 should shrink.
- **Stock layout.** A4 places the circuit on q41 and q42 again. If those qubits still read out at 12–22%, A4 should score below A1.

## Size and power

At 10 blocks × 2,048 shots, MAP-02's observed paired SDs (0.8–1.3 pp) give 95% half-widths under 1 pp, so K1 at 10 pp is well powered. K3 needs an interaction of roughly 2 pp to be detected. K2's power depends on the readout share. Both may be inconclusive, and that will be reported as such.

## Checks before submission (all passed; `tesseract_map_03_dryrun.json`)

- **Noiseless simulation** (Aer matrix-product-state) of every compiled arm in block 0: F_cl ≥ 0.9997.
- **Readout calibration circuits:** return exactly all-0 and all-1.
- **Mitigation:** on synthetic counts with readout-only noise at MAP-02's rates, it restores F_cl to 1.00 (best path) and 1.01 (median path).
- **Lock:** the A1, A2 and A3 layouts equal the MAP-02 lock's C5, C6 and C7 layouts, and each arm has 39 CZ.

## Artifacts

| File | SHA-256 |
| --- | --- |
| `docs/phase3/tesseract_map_03_live_prereg_lock.json` | `ea7e8765a9d82f4c7c5604f0d2cf06eef444f4bd78fc8f6f245b64a79c895817` |
| `docs/phase3/tesseract_map_03_live_pubs.qpy` | `f15167c40519ee3d7b4cb7b37e871f7f19b48d21df032097d0e28cbe754f816f` |
| `docs/phase3/tesseract_map_03_prelock_calibration.json` | `70cc69bc226e6022e6a9bdfdc07af6d80fa66949464ecb6747255fcd5be760ce` |
| `docs/phase3/tesseract_map_03_dryrun.json` | `59479795271a71e399207a007984fed1840027d575902c369e0e65cda4c60b0f` |
| `phase3/tesseract_map_03.py` (build, analysis, submission) | `20beb4e3a9cc982f2eed05b6eaa6809c2bf20458111e375c777bef580169f8ff` |

The runner refuses to submit unless the QPY matches the lock and the remaining quota is at least 1.3 × the estimate. It never submits twice.
