# OSIRIS-QEC-001 Hardware Execution Result: Multi-Round Syndrome-Cycle Scaling ($d \in \{3, 5, 7\}$) on `ibm_fez` (`Job dbtl2vb2kf8s73c9a0d0`)

- **Status**:
  - **`QEC-H1` (Bit-Flip Sub-Threshold Monotonic Suppression)**: **`SUPPORTED / PASS`**
  - **`QEC-H2` (Phase-Flip Mid-Circuit Readout/Reset Dephasing Diagnostic)**: **`SUB_THRESHOLD / PASS`** ($\Lambda_{3\to 5} = 3.93\times$, $\Lambda_{5\to 7} = 2.40\times$ at $R=5$; $\Lambda_{3\to 5} = 3.89\times$, $\Lambda_{5\to 7} = 2.23\times$ at $R=7$)
  - **`QEC-H3` (Space-Time MWPM Advantage + Defect Ceiling `< 0.1200`)**: **`PARTIAL / FALSIFIED ON DEFECT CEILING ONLY`** — Space-time Blossom MWPM beat final-round majority vote by **`3.00x–7.22x`** across every $d \in \{5, 7\}$ point at $R=7$, and `bit_flip_z` defect rates stayed at `0.0621–0.0765 < 0.1200`, whereas `phase_flip_x` at $R=7$ reached mean detector defect probability **`0.1266–0.1284 > 0.1200`**, quantifying cumulative spectator/reset dephasing over 7 mid-circuit readout cycles (`~18 µs`).
- **Pre-Registration Commit (GitHub-Timestamped Prior to Job Creation)**: `1a33ffe7` (`staged_at_utc: 2026-11-16T17:52:31.575162+00:00` < IBM `created: 2026-11-16T18:54:53.740499Z`)
- **Processor**: `ibm_fez` (156-qubit IBM Heron r2, `open-instance`)
- **Job ID**: `dbtl2vb2kf8s73c9a0d0` (`running: 18:54:55.207671Z`, `finished: 18:57:07.143743Z`, `circuits_execution_time_ns: 106,638,331,648`)
- **Frozen QPY SHA-256**: `357a29ad2746e2be94522702516dce3e04319894c1dc981596c9a3640041c056`
- **Verbatim Raw Bitstrings Archive**: `physical_qec001_fez_raw_counts.json.gz` (`SHA-256: cdae697c364e2aad2085cc3585039d079baf4b3d1bb619dcb6178647b75646c1`, `192` circuits × `2,048` shots = **`393,216` shots**, `16,384` pooled shots per `(basis, d, R)` cell, zero postselection)
- **Pinned Concentric 13-Transmon Chain (`routing_method="none"`)**:
  `[111, 110, 109, 108, 107, 106, 105, 117, 125, 124, 123, 122, 121]`
  (`d=3`: `[107..125]`; `d=5`: `[109..123]`; `d=7`: `[111..121]`)
- **QPU Accounting**: **`109` QPU charge seconds (`1.812076 RU`)** consumed (`269 / 600s` = `4.456568 / 10.0 RU` cumulative across all three hardware runs); **`331 / 600` seconds (`5.543432 / 10.0 RU`, `55.4%`) remaining**.

---

## 1. Logical Error per Syndrome Cycle $\epsilon_L(d, R)$ & Suppression Factors $\Lambda_{d \to d+2}$

All MWPM logical error rates below use the **pre-submission calibration edge priors** captured in `qec001_fez_calibration_snapshot.json` prior to hardware execution (zero post-hoc weight fitting), with $\text{SE}_{\text{eff}} = \max(\sigma_{\text{binom}}, \text{SEM}_{\text{block, ddof=1}})$ across the `4` shuffled blocks (`16,384` shots/cell):

| Code Basis | Cycles $R$ | $\epsilon_L(d=3) \pm \text{SE}$ (`Errors/16384`) | $\epsilon_L(d=5) \pm \text{SE}$ (`Errors/16384`) | **$\epsilon_L(d=7) \pm \text{SE}$ (`Errors/16384`)** | $\Lambda_{3 \to 5}$ | $\Lambda_{5 \to 7}$ | **$\Lambda_{3 \to 7}$** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`bit_flip_z`** | `R = 1` | `0.00757 ± 0.00068` (`124`) | `0.00140 ± 0.00042` (`23`) | **`0.00049 ± 0.00017` (`8`)** | **`5.39x`** | **`2.88x`** | **`15.50x`** |
| **`bit_flip_z`** | `R = 3` | `0.00792 ± 0.00041` (`383`) | `0.00215 ± 0.00026` (`105`) | **`0.00133 ± 0.00016` (`65`)** | **`3.69x`** | **`1.62x`** | **`5.97x`** |
| **`bit_flip_z`** | `R = 5` | `0.00757 ± 0.00031` (`602`) | `0.00224 ± 0.00017` (`182`) | **`0.00187 ± 0.00015` (`152`)** | **`3.38x`** | **`1.20x`** | **`4.05x`** |
| **`bit_flip_z`** | `R = 7` | `0.00731 ± 0.00026` (`802`) | `0.00211 ± 0.00025` (`239`) | **`0.00195 ± 0.00018` (`221`)** | **`3.46x`** | **`1.08x`** | **`3.75x`** |
| **`phase_flip_x`** | `R = 1` | `0.01477 ± 0.00094` (`242`) | `0.00208 ± 0.00036` (`34`) | **`0.00079 ± 0.00022` (`13`)** | **`7.12x`** | **`2.62x`** | **`18.62x`** |
| **`phase_flip_x`** | `R = 3` | `0.02130 ± 0.00068` (`1003`) | `0.00470 ± 0.00041` (`229`) | **`0.00174 ± 0.00020` (`85`)** | **`4.53x`** | **`2.71x`** | **`12.27x`** |
| **`phase_flip_x`** | `R = 5` | `0.02174 ± 0.00066` (`1633`) | `0.00553 ± 0.00026` (`443`) | **`0.00230 ± 0.00017` (`187`)** | **`3.93x`** | **`2.40x`** | **`9.44x`** |
| **`phase_flip_x`** | `R = 7` | `0.02301 ± 0.00051` (`2301`) | `0.00592 ± 0.00025` (`655`) | **`0.00265 ± 0.00020` (`299`)** | **`3.89x`** | **`2.23x`** | **`8.69x`** |

---

## 2. Space-Time Blossom MWPM vs. Final-Round Majority Vote (`R = 7`) & Defect Diagnostics

| Code Basis (`R = 7`) | Distance $d$ | Space-Time MWPM $P_L$ (`Errors`) | Final Majority $P_L$ (`Errors`) | **MWPM Error Reduction** | Mean Detector Defect $\bar{p}_{\text{det}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`bit_flip_z`** | $d = 3$ | `0.04895` (`802`) | `0.10150` (`1,663`) | **`2.07x`** | `0.06208` (`< 0.12`) |
| **`bit_flip_z`** | $d = 5$ | `0.01459` (`239`) | `0.07343` (`1,203`) | **`5.03x`** | `0.06631` (`< 0.12`) |
| **`bit_flip_z`** | $d = 7$ | **`0.01349` (`221`)** | `0.09741` (`1,596`) | **`7.22x`** | `0.07652` (`< 0.12`) |
| **`phase_flip_x`** | $d = 3$ | `0.14044` (`2,301`) | `0.20276` (`3,322`) | **`1.44x`** | `0.12657` (`> 0.12`) |
| **`phase_flip_x`** | $d = 5$ | `0.03998` (`655`) | `0.12000` (`1,966`) | **`3.00x`** | `0.12842` (`> 0.12`) |
| **`phase_flip_x`** | $d = 7$ | **`0.01825` (`299`)** | `0.08868` (`1,453`) | **`4.86x`** | `0.12648` (`> 0.12`) |

## 3. Physical Interpretation & Scope Boundaries

1. **Unconditional Monotonic Suppression Across All `8` `(basis, R)` Curves**: Every single `(code_basis, R)` combination (`8 / 8`) exhibits strict monotonic reduction in logical error per cycle $\epsilon_L(3) > \epsilon_L(5) > \epsilon_L(7)$ using dynamic mid-circuit ancilla `measure` + `reset` without postselection.
2. **Why `phase_flip_x` Shows Both Higher Defect Density (`~12.7%`) and Stronger $d=5 \to 7$ Scaling (`2.23x` vs `1.08x`)**:
   - In `phase_flip_x`, data transmons sit on the Bloch equator ($|+\rangle / |-\rangle$) during ancilla readout + reset (`~2.6 µs`/round), experiencing genuine $T_2$ + measurement-induced dephasing (`d=3` $\epsilon_L \approx 2.30\%$/cycle vs `0.73%`/cycle in `bit_flip_z`, pushing $R=7$ detector density to `0.1265–0.1284` just above the pre-registered `0.1200` ceiling in `QEC-H3`).
   - Because single-qubit dephasing is spatially uncorrelated across data transmons, increasing distance $d = 3 \to 5 \to 7$ suppresses `phase_flip_x` cleanly at **`3.89x`** then **`2.23x`** (`8.69x` overall at $R=7$), whereas `bit_flip_z` approaches a `~0.19%`/cycle noise floor between $d=5$ (`0.211%`) and $d=7$ (`0.195%`, `1.08x`) driven by rare correlated two-qubit/leakage events on outer couplers (`q111-q110`, `q122-q121`).
3. **Explicit Scope Boundary**: A 1D repetition code protects one Pauli sector (`Z` or `X`) at a time rather than both simultaneously (which requires a 2D surface, color, or bivariate-bicycle qLDPC code). These measurements benchmark physical mid-circuit syndrome extraction, reset, and space-time MWPM scaling on heavy-hex transmons within standard open-system quantum mechanics.
