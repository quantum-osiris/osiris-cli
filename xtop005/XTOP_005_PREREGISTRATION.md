# Preregistration: OSIRIS-XTOP-005 (heavy-hex design)

**Does encoding the idle reference in [[4,2,2]] improve Hayden–Preskill recovery on IBM Heron r2, and does staggered dynamical decoupling help?**

This document is published before any XTOP-005 hardware job. The circuits are fixed by `docs/xtop005/qpu/lock.json`, which records the SHA-256 of `circuits.qpy`. The submit step refuses to run if the QPY file does not match the lock or if no preregistration DOI is recorded.

## Why the design changed from the original proposal

The original proposal encoded all four registers (R, M, B, E) in [[4,2,2]] blocks. Stage-1 simulation on FakeMarrakesh (`docs/xtop005/stage1/`) showed two problems:

- **Routing overhead.** Heavy-hex has no layout that makes the transversal gates between blocks nearest-neighbour. The circuits compile to 147–396 CZ, against 30–60 for the bare protocol.
- **No benefit.** Post-selected encoded fidelity ended up below the unprotected bare arm.

A 16-qubit-line layout made the gates between blocks local but not the state preparation, which alone cost about 132 CZ.

The heavy-hex design therefore encodes only the quarantined reference R, the register that idles for the whole protocol:

- **Encoder:** a 5-CX star circuit (found by BFS) on one degree-3 heavy-hex node, applied right after the Bell pairs are made.
- **Unencoder:** the inverse circuit, applied just before readout. It returns both stabilizer syndromes on two qubits.
- **Detection:** measuring those two qubits detects every single-qubit error on R during the idle window, in all 9 readout settings.

Overhead is about 13 CZ, and the total is 10 qubits.

## Protocol

The logical program is the same in every arm:

1. Prepare Bell pairs (R_i, M_i) and (B_i, E_i).
2. Apply the scrambler U_L on (M, B). U_L is Clifford and transversal-compatible, and gives I(R:M) = I(R:B) = 0 exactly.
3. Wait through a transit window of length τ on all data qubits.
4. Apply the decoder V_L on (B, E). It restores the Bell pairs (R_i, E_i) up to a known Pauli frame.
5. Read out R and E in each of the 9 two-qubit Pauli settings.

Parameters:

- **Depths:** L ∈ {1, 3}.
- **Transit windows:** τ ∈ {2, 10, 20} µs.
- **Arms:**
  - A: bare, idle.
  - B: bare, staggered XY4.
  - C: R encoded, idle.
  - D: R encoded, staggered XY4.
- **Staggered XY4:** built from native x and virtual rz gates only. Pulses fall at odd eighths of each delay on one heavy-hex sublattice and at even eighths on the other.
- **Layouts:** one hotspot-avoiding layout per encoding, chosen from the live calibration at lock time and shared by the idle and DD arms. Hotspots are qubits with bottom-decile T2 or top-decile readout error, and the endpoints of top-decile CZ edges.
- **Run:** 216 circuits, 1000 shots each, in one SamplerV2 job on ibm_marrakesh. Estimated 68 QPU seconds.

## Measured quantity

The measured quantity is the two-pair Bell fidelity F of the recovered (R, E) pairs, computed from all 9 settings as (1/16) Σ s_a s_b ⟨σ_a σ_a ⊗ σ_b σ_b⟩. The signs s come from the exact noiseless frame.

- Encoded arms use F_det, computed on shots whose two syndrome bits are both 0. The keep fraction is reported alongside it.
- Bare arms use all shots.
- Confidence intervals come from a parametric multinomial bootstrap with 4000 resamples, applied per setting.

## Hypotheses

All tests are one-sided (effect > 0) and evaluated at each depth, giving 8 tests in total. They are Holm-corrected at a family-wise α = 0.01.

| ID | Contrast | Meaning |
| --- | --- | --- |
| H1 | [F_det(C) − F(A)] at 20 µs − [F_det(C) − F(A)] at 2 µs | Encoding R helps more as the idle window grows (crossover). |
| H2 | F(B) − F(A) at 20 µs | Staggered XY4 helps bare qubits during long idles. |
| H3 | F_det(D) − F_det(C) at 20 µs | Staggered XY4 helps on top of encoding. |
| H4 | F_det(C) − F(A) at 20 µs | Encoded R beats bare R after a long idle. |

## Stage-1 predictions (FakeMarrakesh simulation)

The simulator predicts H1 and H4 for L = 1 and a smaller effect for L = 3. It cannot test H2 or H3: its noise model has no ZZ crosstalk or low-frequency dephasing, so DD only adds gate error there and the simulation predicts B ≤ A and D ≤ C. A positive H2 or H3 on hardware would therefore show noise that the simulator's model omits.

## Lock

- **Lock files:** `docs/xtop005/qpu/lock.json` and `circuits.qpy`, compiled against ibm_marrakesh's live calibration.
- **QPY SHA-256:** `d78d15b7a2040fb1…`; the full hash is in lock.json.
- **Physical qubits:**
  - bare arms: 140, 142, 141, 143, 136, 144, 123, 145;
  - encoded arms: 28, 31, 29, 38, 32, 30, 33, 34, 39, 35.
- **Compiled sizes:** 29 to 54 CZ for the bare arms and 45 to 64 CZ for the encoded arms, depending on depth.

**Known confound.** The bare and encoded arms run on different physical qubits, so H4 combines the effect of encoding with the difference in qubit quality. H1 is a difference of differences across idle windows on the same qubits, so most of that difference cancels. H2 and H3 compare arms on the same qubits.

**Dry run.** The full analysis was run end-to-end on these exact locked circuits, using FakeMarrakesh noise in place of hardware counts. All 24 cells were analysed and all 8 tests computed. H1(L=1), H4(L=1) and H4(L=3) came out supported, and H2/H3 did not. This checks only that the pipeline runs. It is not hardware data.

## Stopping and deviations

- **Single submission:** one job, with no re-submission. If the job fails, the failure is reported, and any re-run gets a new lock and an amendment.
- **No data-dependent changes:** analysis code, contrasts, the post-selection rule and α are fixed here.
- **Reporting:** all cells are reported, including raw (non-post-selected) encoded fidelity.
