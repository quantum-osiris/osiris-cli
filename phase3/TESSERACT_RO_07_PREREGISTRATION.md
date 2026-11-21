# Preregistration: TESSERACT-RO-07, is readout error on IBM Heron r2 context-dependent?

**Status:** published before any RO-07 job. Selection rule, circuits, contrasts and criteria are fixed in `phase3/readout_context_07.py` (included in this record). The selection was checked offline on RO-06 rounds 0–3. The final selection uses RO-06's last completed round at lock time, and that selection is published with the results.

## Question

A qubit's all-qubit probe readout error can be far worse than IBM's reported value. RO-06 found examples: `ibm_marrakesh` q31 at 27.8% against 4.6% reported, and q5 at 11.6% against 0.3%. Does such a qubit's error depend on its measurement **context**?

- **Multiplexing:** how many other qubits are measured at the same time.
- **Neighbour state:** the state of every other qubit.

If it does, that explains why per-qubit calibration and simultaneous-readout probes disagree.

## Selection (per device, from RO-06's last round; no two selected qubits are coupled)

| Group | Size | Rule |
| --- | --- | --- |
| DISAGREE | up to 3 | largest e_probe − e_ibm, with e_probe > 0.05 and e_probe − e_ibm > 0.03 |
| AGREE | up to 2 | e_probe > 0.15 and \|e_probe − e_ibm\| < 0.03 (bad readout that IBM reports) |
| CONTROLS | 2 | lowest e_probe |

## Circuits (one job per device; 6 types × 6 interleaved repeats × 2,048 shots)

| Type | Prepared | Measured |
| --- | --- | --- |
| ISO0 / ISO1 | the selected qubits in 0 / 1 | only the selected qubits |
| ALL0 / ALL1 | every qubit in 0 / 1 | every qubit |
| FLIP0 / FLIP1 | selected qubits in 0 / 1, every other qubit in 1 / 0 | every qubit |

## Measures

- **e:** for each qubit and context, the mean of p(1|0) and p(0|1).
- **D_mux** = e(ALL) − e(ISO).
- **D_nbr** = e(FLIP) − e(ALL).
- Both are tested with a paired t over the 6 repeats (df = 5) and reported with a 95% CI.

## Criteria

- **R1 (multiplexing matters):** on at least 2 of 3 devices, at least half the DISAGREE qubits have \|D_mux\| > 0.05 with the CI excluding 0.
- **R2 (neighbour state matters):** the same criterion for D_nbr.
- **R2b (reported):** D_mux and D_nbr for DISAGREE vs AGREE qubits.
- **R3 (sanity):** every control has \|D_mux\| < 0.01 and \|D_nbr\| < 0.01.

## Budget

About 22 s of QPU per device, about 67 s in total.

## Scope note

Readout crosstalk in multiplexed superconducting readout is known in the literature. The question here is narrower: is the specific gap between IBM's reported readout errors and in-job readout on public Heron r2 devices explained by context?
