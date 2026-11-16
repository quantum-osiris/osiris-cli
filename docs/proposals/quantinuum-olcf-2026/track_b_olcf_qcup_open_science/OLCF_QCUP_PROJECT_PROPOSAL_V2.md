# OLCF Quantum Computing User Program (QCUP): Project Proposal (`Track B` — Open Fundamental Science)

## 1. Project Title

**Pre-Registered, Provenance-Governed Benchmarks of High-Dimensional Entanglement, Debiased Optimal-Transport Stability, and Nonlocal Syndrome Decoding on Trapped-Ion and Superconducting Quantum Processors**

## 2. Principal Investigator

| Field | Entry |
| :--- | :--- |
| **PI** | Devin Phillip Davis |
| **Organization** | Agile Defense Systems LLC |
| **Email** | `research@dnalang.dev` |
| **Phone** | `[fill]` |
| **Mailing Address** | Lexington, KY 40503, USA `[add street address]` |
| **Citizenship / Residency** | United States `[confirm per OLCF account requirements]` |
| **Co-Investigators / Project Users** | PI only (initial 6-month allocation) `[or add co-investigators]` |

## 3. Abstract

This project develops and executes prospectively pre-registered, cryptographically provenance-tracked benchmarking experiments comparing trapped-ion (Quantinuum H2 and Helios via Nexus) and superconducting (IBM Quantum) architectures. Our methodology freezes hypotheses, OpenQASM 3 / QPY circuit digests, qubit layouts, shot budgets, and pass/fail thresholds in RFC 8785 hash-chained records prior to QPU job submission, and verifies every reported statistic by independent recomputation from raw shot histograms. Building on three completed pre-registered superconducting campaigns totaling 978,944 raw shots—(1) dual-processor 9-basis joint Bell-stabilizer experiments (`10.5281/zenodo.24287505`, 585,728 shots) certifying Schmidt number >= 3 on `ibm_marrakesh` (raw joint fidelity 0.633423) and both Schmidt number >= 3 and witness-derived entanglement of formation >= 1.147 ebits (1.124 ebits at the 95% lower confidence limit) on `ibm_fez` (raw joint fidelity 0.730713), and (2) a 393,216-shot dynamic mid-circuit measurement/reset repetition-code scaling campaign (`OSIRIS-QEC-001`, `10.5281/zenodo.24293756`, Job `dbtl2vb2kf8s73c9a0d0`, d in {3, 5, 7}, R in {1, 3, 5, 7}) demonstrating 3.75x bit-flip and 8.69x phase-flip logical error suppression alongside a 7.22x space-time Blossom V MWPM gain—we will isolate architectural limits across platforms in three work packages: **(WP1)** routing-free high-Schmidt-number entanglement witnesses (k = 2 and k = 3) under QCCD all-to-all connectivity, **(WP2)** classical-simulation-free, split-sample-debiased Kantorovich/Sinkhorn distributional stability benchmarks at 50 to 96 physical qubits, and **(WP3)** multi-round syndrome extraction and co-compiled error detection/qLDPC decoding (`BP+OSD` vs. `A*` and MWPM on Steane `[[7,1,3]]`, Tesseract `[[16,6,4]]`, and Bivariate-Bicycle `[[72,12,6]]`) in QCCD isolated measurement zones, directly testing relief of the 3.344 us/round spectator-dephasing defect floor observed on planar transmons. All circuits, raw histograms, code, and negative results will be published openly. This is open, fundamental research; no export-controlled, PHI, or other controlled data will be used.

## 4. Nature and Significance

Near-term and early fault-tolerant quantum benchmarking faces two acute methodological gaps: post-hoc selection of metrics or qubit layouts, and reliance on statevector fidelity references that become intractable past roughly 50 qubits. We address both through a write-ahead pre-registration and independent raw-count verification pipeline, paired with explicit mathematical scope boundaries separating direct observables from witness-derived bounds.

Our completed campaigns demonstrate both the referee-auditable rigor of this protocol and the physical bottlenecks motivating cross-platform trapped-ion experiments:

1. **High-Dimensional Entanglement Recovery (`OSIRIS-HP-004` & `HP-004B`, 585,728 raw shots, `10.5281/zenodo.24287505`):** Layout-locked 8-qubit Hayden–Preskill recovery across 9 Pauli bases witnessed Schmidt number >= 3 (`F_joint > 0.5000`) at +50.64 sigma (`ibm_marrakesh`, 0.633423) and +84.45 sigma (`ibm_fez`, 0.730713), while crossing the d = 4 Terhal–Vollbrecht / Vollbrecht–Werner 1-ebit entanglement-of-formation threshold (`F* = 0.683900`) at +17.13 sigma on `ibm_fez` (`E_F >= 1.147 ebits`; `1.124 ebits` at the 95% Student-t lower confidence limit `0.7237`). These are witness-derived convex-roof bounds within standard quantum mechanics, conditional on target-state mapping and Pauli basis implementation, not direct von Neumann entropy measurements. Extending beyond k = 2 pairs or non-tree scramblers on heavy-hex hardware is blocked by planar SWAP routing: in our controlled 40-qubit rerun (`10.5281/zenodo.22870279`), mapping a star graph onto heavy-hex inflated two-qubit gates from 39 to 166 and degraded exact support by 3.3x.
2. **Dynamic Mid-Circuit Syndrome Scaling & Spectator Dephasing (`OSIRIS-QEC-001`, 393,216 raw shots, Job `dbtl2vb2kf8s73c9a0d0`):** Across 192 dynamic mid-circuit `measure` + `reset` circuits on a 13-transmon chain (`d in {3, 5, 7}`, `R in {1, 3, 5, 7}`), space-time Blossom V MWPM with pre-submission calibration priors achieved monotonic sub-threshold suppression across all 8 `(basis, R)` curves and reduced `R = 7, d = 7` cumulative logical error by 7.22x (`bit_flip_z`: 9.74% to 1.35%) and 4.86x (`phase_flip_x`: 8.87% to 1.82%) over final-round majority voting. However, `phase_flip_x` at `R = 7` exceeded our pre-registered `0.1200` detector defect ceiling (`0.1265` to `0.1284`), traced quantitatively to `3.344 us/round` readout+reset idle exposure interacting with a localized `T2 = 18.17 us` data-qubit hotspot (`8.38%/round` thermal phase-flip probability) and heavy-hex degree constraints that restrict 1D chains to one Pauli sector per run.

Quantinuum H2 and Helios QCCD architectures combine arbitrary two-qubit connectivity with physically shuttled, optically isolated mid-circuit measurement/reset zones. Comparing identical OpenQASM 3 logical protocols across Quantinuum QCCD and IBM Heron r2 processors under one pre-registered framework cleanly separates hardware-topology and spectator-dephasing effects from intrinsic algorithmic and decoder performance.

## 5. Methodology

1. **Write-Ahead Pre-Registration:** Hypotheses, OpenQASM 3 / `pytket`-compiled circuit digests, qubit/zone assignments, shot counts, decoder priors (zero post-hoc edge fitting), and pass/fail decision rules are canonicalized (RFC 8785), SHA-256 hash-chained, and committed prior to submission.
2. **Emulator-Gated Execution:** Every work package is profiled offline in native trapped-ion basis (`rx`, `ry`, virtual `rz`, `rzz`, mid-circuit `measure`/`reset`) and verified on Quantinuum Nexus emulators (`H2-1E` / `Helios-1E`) before QPU dispatch.
3. **Independent Raw-Count Verification & Debiasing:** Separate verification scripts recompute stabilizer witnesses, logical error rates, and split-sample-debiased optimal-transport distances directly from raw integer histograms. Negative or ceiling-exceeding outcomes are reported un-modified alongside calibration diagnostics.
4. **Open Release:** All raw shot archives, pre-registrations, OpenQASM 3 / `.dna` sources, and `BP+OSD` / MWPM decoder benchmarks are deposited on Zenodo and public Git repositories under permissive open licenses.

## 6. Why QCUP Access Is Required

- **QCCD All-to-All Connectivity & Shuttled Mid-Circuit Readout (`H2` / `Helios`):** Required to embed non-planar `k = 3` (`d = 8` Schmidt rank) scrambling/recovery graphs with 0 routing SWAPs (`WP1`) and to execute weight-4, weight-6, and weight-8 CSS syndrome checks (`Steane [[7,1,3]]`, `Tesseract [[16,6,4]]`, and `Bivariate-Bicycle [[72,12,6]]` with 24 reused ancillas = 96 ions on Helios) with low cross-talk mid-circuit reset (`WP3`).
- **Unified Cross-Vendor Replication:** QCUP uniquely provides Quantinuum trapped-ion and IBM superconducting access within one merit-reviewed open-science allocation.
- **Scale Beyond Self-Funded Open Windows:** Our baseline runs consumed 269 QPU seconds (~979k shots) on single open windows; multi-block 50-to-96-qubit cross-vendor experiments require dedicated QCUP allocations.

## 7. Work Packages (6-Month Allocation)

- **WP1 — Routing-Free High-Schmidt-Rank Entanglement Witnesses (`H2 / Helios` + `IBM`):** Replicate the 8-qubit 9-basis joint-stabilizer protocol (`WP1a`, 81 circuits x 300 shots) on H2/Helios with 0 routing SWAPs, and extend to 12-qubit `k = 3` (`d = 8` Schmidt-rank certification, `WP1b`, 54 circuits x 300 shots), quantifying the routing-attributable fidelity gap against matched IBM Heron r2 replicates.
- **WP2 — Split-Sample-Debiased Kantorovich/Sinkhorn Stability at 50 to 96 Qubits (`H2 / Helios` + `IBM`):** Measure replicate-to-replicate distributional stability on 50 qubits (`WP2a`) and 96 qubits (`WP2b` on 98-qubit Helios) using exact Kantorovich linear programming and certified Sinkhorn optimal transport under normalized Hamming cost, subtracting within-job split-sample multinomial baselines to eliminate `O(S^-1/Q)` finite-shot sparsity bias without classical statevector references.
- **WP3 — Nonlocal Error-Detection Co-Compilation & Hypergraph `BP+OSD` Decoding (`H2 / Helios` + `IBM`):** Execute multi-round (`R = 3`) syndrome extraction on `Steane [[7,1,3]]` and `Tesseract [[16,6,4]]` (`WP3a`, 21 physical ions) and `R = 2` syndrome extraction on the nonlocal weight-6 `Bivariate-Bicycle [[72,12,6]]` qLDPC code (`WP3b`, 72 data + 24 reused ancillas = 96 ions on Helios). Compare detector defect rates against our `ibm_fez` `3.344 us/round` baseline (`0.1265` to `0.1284`), and benchmark Sum-Product **Belief Propagation + Ordered Statistics Decoding (`BP+OSD-CS`)** (verified offline at `7/7`, `16/16`, and `72/72` weight-1 plus `2556/2556` weight-2 recovery) against heuristic `A*` and graph-decomposed Blossom V MWPM.

## 8. Resource Estimates (Compiled Gate-Count HQC Sizing)

*Computed directly from native ion-trap decompositions (`rx`, `ry`, virtual `rz = 0`, `rzz`, and mid-circuit `measure`+`reset`) via `HQC = 5 + C * (N_1q + 10*N_2q + 5*N_m) / 5000`, to be cross-checked on `H2-1E` / `Helios-1E` emulators in Month 1:*

| Work Package | Target Platform | Qubits | Circuits x Shots | Mean `(N_1q, N_2q, N_m)` | SWAPs | Estimated Workload |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **WP1a** (`k=2` 9-basis replicate) | Quantinuum H2-1 / Helios-1 | 8 | `81 x 300` (`24,300`) | `(99.67, 33.33, 12.0)` | 0 | **`2,801.0 HQC`** |
| **WP1b** (`k=3` all-to-all extension) | Quantinuum H2-1 / Helios-1 | 12 | `54 x 300` (`16,200`) | `(53.0, 30.0, 18.0)` | 0 | **`1,705.3 HQC`** |
| **WP2a** (`50Q` 4-layer transport) | Quantinuum H2-1 / Helios-1 | 50 | `12 x 512` (`6,144`) | `(300.0, 100.0, 100.0)` | 0 | **`2,271.8 HQC`** |
| **WP2b** (`96Q` 4-layer transport) | Quantinuum Helios-1 | 96 | `12 x 512` (`6,144`) | `(574.0, 192.0, 192.0)` | 0 | **`4,304.3 HQC`** |
| **WP3a** (`[[7,1,3]]` & `[[16,6,4]]` `R=3`) | Quantinuum H2-1 / Helios-1 | 21 | `24 x 512` (`12,288`) | `(152.5, 156.0, 75.0)` | 0 | **`5,250.2 HQC`** |
| **WP3b** (`BB [[72,12,6]]` `R=2` reuse) | Quantinuum Helios-1 | 96 | `8 x 256` (`2,048`) | `(792.0, 864.0, 456.0)` | 0 | **`4,837.2 HQC`** |
| **Quantinuum Subtotal** | **H2-1 / Helios-1 (Nexus)** | **8–96** | **`191` (`67,124` shots)** | — | **0** | **`21,169.8 HQC` (`21,500 HQC` request)** |
| **Matched Superconducting Replicates** | **IBM Quantum (Heron r2)** | **8–96** | **`~600` (`~750k` shots)** | Native `cz/sx/rz` | Planar | **`600 QPU seconds` (`10 RU`)** |

## 9. Timeline (6 Months)

| Month | Milestone |
| :--- | :--- |
| **1** | Validate OpenQASM 3 / `pytket` circuits on `H2-1E` and `Helios-1E` Nexus emulators; submit `myOLCF` credit request; freeze `WP1` pre-registration |
| **2–3** | Execute and independently verify `WP1a` and `WP1b` on H2/Helios and IBM Quantum; freeze `WP2` pre-registration |
| **4** | Execute `WP2a` (50Q) and `WP2b` (96Q Helios) split-sample-debiased optimal-transport runs; freeze `WP3` pre-registration |
| **5** | Execute `WP3a` (`[[7,1,3]]`, `[[16,6,4]]`) and `WP3b` (`BB [[72,12,6]]`) mid-circuit QEC runs; run `BP+OSD-CS` vs. `A*` and MWPM benchmarks |
| **6** | Deposit all raw counts, QPY/OpenQASM 3 archives, and verifiers on Zenodo/Git; submit quarterly and closeout OLCF reports and peer-reviewed manuscripts |

## 10. Open Research Commitment and Compliance

All proposed work is open, fundamental research. No export-controlled, PHI, or other controlled data will be produced, processed, or stored on QCUP resources. All results, including negative and falsifying results, will be published openly with permissive-licensed code, and every publication using QCUP resources will include the verbatim acknowledgement:

> "This research used resources of the Oak Ridge Leadership Computing Facility, which is a DOE Office of Science User Facility supported under Contract DE-AC05-00OR22725."

Quarterly and closeout reports will be submitted on schedule per QCUP policy.

## 11. Merits and Fit to QCUP Priorities

- **Enable Research:** Publishes reusable open tooling for write-ahead pre-registration, split-sample optimal-transport debiasing, and hypergraph `BP+OSD` verification across vendor platforms.
- **Evaluate Technology:** Performs architecture-controlled comparisons of QCCD all-to-all connectivity and isolated mid-circuit reset against planar heavy-hex routing and spectator dephasing.
- **Engage the Community:** Releases complete raw shot histograms, circuit manifests, and transparent errata records to strengthen QIS reproducibility standards.

## 12. Track Record and Open Outputs

- **Dual-Processor Entanglement Certification (`OSIRIS-HP-004` & `HP-004B`, Zenodo `10.5281/zenodo.24287505`):** 585,728 raw shots across `ibm_marrakesh` (`dbti1fbsbsbs73cb2vu0`) and `ibm_fez` (`dbtj61bnlcms73ajc380`) certifying Schmidt number >= 3 (+50.64 sigma, +84.45 sigma) and witness-derived entanglement of formation >= 1.147 ebits (1.124 ebits at 95% LCL), independently reconstructed from raw job exports.
- **Multi-Round Dynamic QEC Scaling (`OSIRIS-QEC-001`, Zenodo `10.5281/zenodo.24293756`, Job `dbtl2vb2kf8s73c9a0d0`):** 393,216 raw shots on `ibm_fez` across `d in {3, 5, 7}` and `R in {1, 3, 5, 7}`, demonstrating monotonic sub-threshold suppression on 8/8 curves, 7.22x space-time Blossom V MWPM error reduction, and calibration-resolved spectator `T2` hotspot diagnostics.
- **Open Compiler & Rerun Errata (`10.5281/zenodo.22862567`, `10.5281/zenodo.22870279`):** Conformance-tested OpenQASM 3 circuit-family compiler (`20,000` random programs, `< 0.015%` failure rate), bivariate-bicycle qLDPC Pareto screening, and public withdrawal/errata of superseded claims.
