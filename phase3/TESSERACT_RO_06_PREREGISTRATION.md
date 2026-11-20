# Preregistration: TESSERACT-RO-06, a time series of readout faults on three IBM Heron r2 processors

**Status:** published before any RO-06 job. The design, definitions, thresholds and analysis are fixed in `phase3/readout_drift_06.py` (included in this record). The analysis code was checked offline on the MAP-04 and MAP-05 probe data (already published) used as two pseudo-rounds. No RO-06 data existed at the time of that check.

**Motivation:** in MAP-03, MAP-04 and MAP-05, on the same afternoon, `ibm_marrakesh` q46 measured 54% → 34% → 0.6% readout error while IBM listed about 1%. Qubit-selection rules reversed their ranking between runs.

## Design

- **Rounds:** 8, 15 minutes apart.
- **Per round, per backend** (`ibm_marrakesh`, `ibm_fez`, `ibm_kingston`):
  - one probe job: all-0 and all-1 on every qubit, 3 repeats, 2,048 shots;
  - an IBM calibration snapshot taken at submission.
- **Cost:** about 145 s of QPU time.
- **Quota guard:** a round is not submitted if the remaining quota is less than 1.3 × its estimate.

## Definitions

| Term | Meaning |
| --- | --- |
| e_probe | mean of p(1\|0) and p(0\|1), pooled over the 3 repeats |
| e_ibm | IBM-reported readout_error in the same round |
| hidden fault | e_probe > 0.15 and e_ibm < 0.03 |
| recovered | e_probe < 0.05 |

## Hypotheses

| ID | Hypothesis | Measure | Supported if |
| --- | --- | --- | --- |
| H1 | Prevalence | hidden faults per device-round | ≥ 1 hidden fault in a majority of rounds on ≥ 2 of 3 devices |
| H2 | Intermittency | among qubits that are a hidden fault in some round, the fraction recovered in another round | > 0.5 |
| H3 | Persistence | consecutive-round 2×2 table (fault at r vs fault at r+1), Fisher exact | reported: odds ratio, p, and P(fault at r+1 \| fault at r) |
| H4 | Tracking | per device-round Spearman ρ(e_probe, e_ibm) | reported: median per device |

**Noise floor:** within-round repeat range against between-round change for flagged qubits.
