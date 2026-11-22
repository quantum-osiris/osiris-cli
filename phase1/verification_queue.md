# Phase 1 verification queue and experiment cards (2026-11-22)

- **Reviewer:** self-review (Claude). No independent reviewer ran.
- **Claim IDs** refer to `claim_evidence_map.jsonl`.
- **Owner abbreviations:** "owner" is the project owner; "agent" is a Claude Code session working under the owner's approval.

## 1. Prioritized checks

The queue is ordered by decision value per unit of cost. Checks that can invalidate a whole downstream comparison come first.

| # | Check | Claim | Owner | Prerequisite | Expected artifact | If it passes | If it fails |
|---|---|---|---|---|---|---|---|
| 1 | BT2T inversion against an exact oracle: all 32 inputs, phases, unused codewords | BT2T-2 | agent | none | `tests/test_bt2t_inversion.py`, `manifest.json` | **Done this session: pass.** Advance to offline compiler validation (G1 to G2). | Revise the circuit; hardware stays blocked. |
| 2 | Independent rederivation of encoding `bt2t-enc-1` and rerun of the tests | BT2T-2 | second reviewer (human or separate agent) | the reviewer has the repo | review note | G1 is signed off by someone other than the author. | Freeze nothing; fix the defect first. |
| 3 | Retrieve IBM `creation_date` for job `dc0uhdts2omc7389tmt0` | SCI-XTOP-1 | owner, who must run it or authorize use of the key | IBM account access | job metadata JSON (CRN redacted) | Timing claim becomes SUPPORTED or REFUTED on the provider clock. | Claim stays INCONCLUSIVE; drop it from the paper. |
| 4 | Decide the CRSM_ARCH verdict wording (R-3) | SCI-ARCH1-5 | owner | none | register PR | Register matches both decision rules. | — |
| 5 | Correct Zenodo `publication_date` in a new version note | SCI-XTOP-2 | owner | Zenodo access | new version record | Metadata is accurate; nothing is backdated. | — |
| 6 | Relabel the A-field harness outputs and add injection-recovery | PHY-9 | agent, if approved | none (local) | harness patch and test | The harness can show detector sensitivity. | — |
| 7 | Errata pointer in `falsifiable_predictions.md` | PHY-7 | owner decision | none | one-line header change | — | — |
| 8 | First approved Nexus preflight on one circuit, with Nexus HQC compared to `local_hqc` | ENG-NEXUS-4 | owner approves; agent runs | Nexus credentials, explicit approval | preflight receipt | Live compatibility becomes SUPPORTED. | File adapter defects before any paid job. |
| 9 | Install the Vercel GitHub App on `quantum-osiris`; redeploy from Git | ENG-DEPLOY-1/4 | owner | org admin | Git-linked deployment | Source binding moves from operator-supplied to platform-recorded. | — |
| 10 | Retrieve NCLM-ARCH-1 run directories (evaluation logs) | SCI-ARCH1-4 | owner | the machine that ran them | logs | The question of when results were viewed becomes answerable. | It stays unknown. |

## 2. Provisional research landscape

Scores run from 1 to 5, where higher is better; for cost, 5 means cheapest. They reflect this reviewer's judgement from the literature and the inspected repo, not a measurement. "Null info" is what a negative result would teach us.

| Area | Representative task | Classical comparator | Theory | Classical gap | HW readiness | Cost | Null info | Total | Class |
|---|---|---|---|---|---|---|---|---|---|
| QEC: memory and repeated correction | logical memory vs rounds, small codes | none needed (the reference is the unencoded qubit) | 5 | n/a | 4 | 4 | 5 | **18** | useful hardware primitive |
| Compiler and placement diagnostics (echo) | predict held-out circuit fidelity | calibration-ranked and topology-only placement | 4 | n/a | 5 | 5 | 4 | **18** | engineering benchmark |
| Exact group primitives (BT2T) | inversion, then multiplication and Fourier transform on 2T | exact classical oracle (trivial) | 5 | none (classically trivial) | 5 | 5 | 3 | **18** | scientific benchmark and compiler test bed |
| Many-body dynamics | Trotterized 2D Ising and Heisenberg dynamics past exact-diagonalization size | tensor networks, Pauli propagation | 4 | 2 (classical methods keep catching up) | 3 | 2 | 3 | 14 | advantage claim, contested |
| Chemistry and materials | ground-state energy to chemical accuracy, FeMoco-class | DMRG, CCSD(T), AFQMC | 5 | 3 (only at fault-tolerant scale) | 1 (needs about 10⁶ physical qubits; T-count about 10⁹ to 10¹⁰, roughly) | 1 | 2 | 12 | end-to-end advantage, FT era |
| Random-circuit sampling and peaked circuits | sampling or peak finding (ADV-0) | MPS, Pauli propagation, light cone | 4 | 1 (ADV-0: classical attack won 30 of 30 cells) | 4 | 3 | 3 | 15 | advantage claim, **STOP at current sizes** |
| Metrology | phase estimation with entangled probes | standard quantum limit | 5 | n/a (physical-resource advantage) | 3 | 3 | 3 | 14 | physics benchmark |
| Cryptography (Shor) | RSA-2048 factoring | GNFS | 5 | 5 | 1 (about 10⁶ physical qubits, around a week; unknown for this stack) | 1 | 1 | 13 | FT-era advantage |
| Optimization (QAOA, annealing) | MaxCut and similar on NISQ | Goemans–Williamson, simulated annealing, heuristics | 2 | 1 | 4 | 4 | 2 | 13 | weak advantage case |
| ML with quantum kernels | classification | classical kernels and NNs | 2 | 1 (data loading and dequantization) | 4 | 4 | 1 | 12 | weak |
| Communication (teleportation, QKD) | teleportation fidelity with feed-forward | — | 5 | n/a | 4 | 4 | 3 | 16 | useful primitive (exercises feed-forward) |

**Cost accounting.** Every comparison against a classical method must count:
- compilation;
- shots and repetitions;
- postselection loss (acceptance rate);
- decoding;
- result processing.

Where a row above omits one of these, it is **unknown**, not zero.

## 3. Experiment cards

### Card A: OSIRIS-BT2T-001, exact 2T inversion (milestone 1)

- **Hypothesis (engineering):** a 5-qubit reversible circuit implements group inversion on encoding `bt2t-enc-1` exactly. The structured synthesis needs fewer native two-qubit gates than the generic transposition reference, with the logical operation held fixed.
- **Reference:** an exact multiplication table from integer quaternion arithmetic. Inverses are found from the table and cross-checked against conjugation. The generic transposition circuit is the synthesis reference.
- **Controls:**
  - a positive control: an injected CZ phase error is detected while basis outputs are unchanged;
  - the earlier prototype, which fails necessary conditions.
- **Primary metric:**
  - *Ideal stage:* the maximum absolute difference between the circuit matrix and the permutation matrix (must be below 1e-9). Measured values: 3.5e-14 for the generic circuit and 3.6e-15 for the structured one.
  - *Later hardware stage, not authorized:* correct-output probability under uniform weighting over the 24 valid inputs, reported separately from unused-codeword outcomes and from preparation and readout controls.
- **Local PROXY results** (not provider output):

  | Target | Generic | Structured |
  |---|---|---|
  | Qiskit, all-to-all, opt 3 | 400 CZ | 48 CZ |
  | Qiskit, 5-qubit heavy-hex fragment, opt 3 | 723 CZ | 72 CZ |
  | pytket H2-native rebase, with FullPeepholeOptimise | 400 ZZPhase | 47 ZZPhase |

  These are differences between syntheses, not hardware advantages.
- **Falsifier:** any basis mismatch, any phase mismatch beyond tolerance, or an unused codeword that is not fixed.
- **Stop/go:**
  - **GO** to offline compiler validation: pinned compiler versions, saved compiled artifacts, equivalence on each target's native set. All ideal checks pass.
  - **Hardware: BLOCKED** until (a) an independent reviewer signs off G1; (b) a frozen preregistration exists covering input weighting, shots, the unit of inference (calibration window or job, not shots), and the equal-shot rule; (c) target access and cost are verified at run time; (d) the owner gives explicit approval.

### Card B: OSIRIS-ECHO, does echo-based placement predict held-out performance?

- **Hypothesis:** placements ranked by the return probability of a forward-then-inverse echo predict the fidelity of *unseen* circuits better than calibration-ranked, topology-only or random placements, under the same selection budget.
- **Reference:** held-out circuits whose ideal outputs are exactly computable, such as BT2T inversion, Clifford circuits and mirror circuits.
- **Controls:**
  - a no-workload echo (identity);
  - a known perturbation, such as a deliberate over-rotation, which must lower the return;
  - a compiled-artifact inspection showing the workload was not cancelled. A barrier alone is not enough evidence.
- **Primary metric:** Spearman rank correlation between the echo score and held-out fidelity, minus the same correlation for the best non-echo comparator. Uncertainty comes from a bootstrap over calibration windows, which are the unit of inference.
- **Minimum effect of interest:** not yet set, so this starts as an **exploratory pilot** with a separate confirmatory dataset.
- **Falsifier:** echo ranking is no better than calibration ranking on held-out windows.
- **First stop/go:** an offline noisy simulation with a known placement-dependent noise model must recover the right ranking. If it cannot, STOP before hardware.
- **Mechanism:** none is inferred from return curves.

### Card C: OSIRIS-QEC, when does repeated correction improve logical reliability?

- **Audit first.** dnalang-core `wen_plaquette_pass.py` is a TKET pass with a provenance capsule. Inspected evidence covers compilation and tests only; no syndrome or logical-outcome dataset for Wen-plaquette or qLDPC was found in this pass. XTOP-005's [[4,2,2]] encoding did not beat bare qubits on hardware (SCI-XTOP-4).
- **Hypothesis:** for the smallest code with an auditable decoder (a distance-3 repetition code; a d=3 surface code if the target supports it), logical failure per round under decoded correction falls below both the unencoded and the encoded-without-correction arms, at matched task duration.
- **Arms:**
  1. unencoded;
  2. encoded, no correction;
  3. postselected detection, with the acceptance rate reported;
  4. decoded correction, comparing offline decoding or Pauli-frame tracking against physical feed-forward, labelled separately.
- **Reference:** stim and pymatching (or an equivalent auditable decoder) under stated circuit-level noise: relaxation, dephasing, coherent over-rotation, readout and reset, leakage where modelled, and correlated noise. Any mechanism omitted is documented.
- **Primary metric:** logical error per round, from a fit of failure against rounds 1 to N. Unconditional outcomes are reported separately from postselected ones.
- **Falsifier:** decoded correction does not beat arm 2 at any tested round count.
- **First stop/go:** the simulation must show the crossing at the target's published error rates, verified at run time. If not, STOP.
- **Scope limit:** a single code distance answers the question for that size only. Scaling with distance remains open, and a single-size result is not evidence of scalable fault tolerance.

## 4. Go/no-go

**GO:** BT2T-001 advances to **offline compiler validation only** (G1 to G2).

**NO-GO (BLOCKED):** any hardware, emulator, Nexus upload, remote compile or costing for BT2T. Unblocking needs:
- an independent G1 review (check 2);
- a frozen preregistration;
- explicit owner approval naming the provider, target, shot budget and cost ceiling.
