# OSIRIS-HP-004 Hardware Execution Result (`ibm_marrakesh`, Job `dbti1fbsbsbs73cb2vu0`)

- **Status**: `SUPPORTED (RAW UNMITIGATED, 5/5 PRE-REGISTERED HYPOTHESES PASSED)`
- **Pre-Registration Commit (GitHub-Timestamped Prior to Job Creation)**: `fa9ef46db8e68ff7874d9a91a6ff241028916678` (`2026-11-16T15:26:18Z` < IBM `created: 2026-11-16T15:26:53.494751Z`)
- **Processor**: `ibm_marrakesh` (156-qubit IBM Heron r2, `open-instance`)
- **Job ID**: `dbti1fbsbsbs73cb2vu0` (`running: 15:26:55Z`, `finished: 15:28:45Z`)
- **Frozen QPY SHA-256**: `c3daaa48d89a74cec39932c310365b3eb23660ac4922219586543f8fe473d02c`
- **Pinned 8-Transmon Tree Layout (`routing_method="none"`)**: `[34, 52, 32, 33, 39, 53, 54, 55]`
- **Pinned 4-Transmon Readout (`c0..c3`, Identical Across All 286 Circuits)**: `[34, 39, 52, 53]` (`mean 4Q SPAM diagonal = 0.9398`)
- **Total Payload**: `286` circuits (`16` SPAM states + `6` randomized blocks × `5` arms × `9` Pauli bases) × `1,024` shots = **`292,864` shots** (`resilience_level=0`, zero postselection, all `286` raw 4-bit histograms archived in `physical_hp004_results.json`)
- **QPU Accounting**: `81` QPU charge seconds (`1.335542 RU`) consumed; `519 / 600` seconds (`8.664458 / 10.0 RU`) remaining.

## Primary Raw Unmitigated Results

| Arm | ISA CZs | Raw $F_1 \pm \sigma$ | Raw $F_2 \pm \sigma$ | **Raw 9-Basis $F_{\text{joint}} \pm \text{SE}_{\text{eff}}$** | 95% Shot-Bootstrap CI | $Z_{\text{eff}}$ vs `0.5000` | Secondary SPAM $F_{\text{joint}}$ | Hypothesis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Arm A` (Barriered Inverse)** | `38` | `0.7516 ± 0.0024` | `0.8006 ± 0.0022` | **`0.6591 ± 0.0026`** | `[0.6543, 0.6642]` | `+61.67σ` | `0.7194` | **H1 PASS** (`> 0.60`) |
| **`Arm B` (2-Photon Restricted)** | **`32`** | `0.7225 ± 0.0025` | `0.7650 ± 0.0023` | **`0.6334 ± 0.0026`** | **`[0.6284, 0.6385]`** | **`+50.64σ`** | `0.6921` | **H2 PASS (`> 0.50` at `50.6σ`)** |
| **`Arm B-Ctrl` ($|0000\rangle_{BE}$)** | `30` | `0.4274 ± 0.0029` | `0.2476 ± 0.0032` | **`0.1799 ± 0.0027`** | `[0.1743, 0.1852]` | `-117.03σ` | `0.1930` | **H3 PASS** (`∈ (0.125, 0.25]`) |
| **`Arm B-Def` (1-Photon Sub-Cap.)** | `25` | `0.4174 ± 0.0029` | `0.2575 ± 0.0032` | **`0.1086 ± 0.0035`** | `[0.1032, 0.1142]` | `-112.84σ` | `0.1123` | **H4 PASS** (`<= 0.125 + 3 SE`) |
| **`Arm B-Held` (Wrong $U_2$)** | `35` | `0.2520 ± 0.0032` | `0.2560 ± 0.0032` | **`0.0634 ± 0.0027`** | `[0.0583, 0.0688]` | `-164.67σ` | `0.0632` | **H5 PASS** (`≈ 0.0625`) |

## Key Scientific Findings

1. **Unconditional Certification of Schmidt Number $\mathrm{SN}(\rho) \ge 3$ (and Exact Vollbrecht--Werner $E_F$ Bound)**:
   Raw, unmitigated `Arm B` achieves $F_{\text{joint}} = 0.633423 \pm 0.002634$ across 6 randomized blocks (`[0.6343, 0.6327, 0.6427, 0.6231, 0.6366, 0.6311]`, block `ddof=1` SEM `0.002634` vs shot SE `0.002617`, 6-block 95% Student-$t$ CI `[0.6267, 0.6402]`), exceeding the exact Terhal--Horodecki (`quant-ph/9911117`) `0.5000` Schmidt-number-$\le 2$ ceiling by **`+50.64σ`** (`lower 5σ bound = 0.6203 > 0.5000`) while physical horizon remnants (`q32, q55`) and references (`q34, q52`) experienced **zero gates** during decoding.
   - *Terminology Clarification (v2)*: Exceeding $F_{\text{joint}} = 0.5000$ certifies **Schmidt number $\mathrm{SN}(\rho_{R_2 R_1 : M_1' M_2'}) \ge 3$** (ruling out any rank-$\le 2$ / single-qubit channel + LOCC), which is distinct from certifying Entanglement of Formation $E_F(\rho) > 1\text{ ebit}$: by Vollbrecht & Werner (*Phys. Rev. A* 64, 062307, 2001), the exact $d=4$ convex-roof threshold for $E_F(\rho) > 1\text{ ebit}$ is $F_* = 0.683910$, so raw `ibm_marrakesh` `Arm B` (`0.6334`) establishes $E_F(\rho) \ge 0.8451 \pm 0.0080\text{ ebits}$ (whereas the cross-processor replication `OSIRIS-HP-004B` on `ibm_fez`, Job `dbtj61bnlcms73ajc380`, achieves raw $F_{\text{joint}} = 0.7307 \pm 0.0027$, certifying both $\mathrm{SN} \ge 3$ at $+84.45\sigma$ and $E_F(\rho) \ge 1.1465 \pm 0.0086\text{ ebits} > 1\text{ ebit}$ at $+17.13\sigma$).
2. **Necessity of Full 9-Basis Stabilizer Tomography Over 3-Basis Marginals**:
   On the exact same dataset, `Arm B`'s 2-qubit marginals were $\bar{F} = 0.7438 < 0.7500$, $F_1 F_2 = 0.5528 < 0.5625$, and $F_1 + F_2 - 1 = 0.4875 < 0.5000$—demonstrating that 3-basis marginal estimators (`HP-003`) would have remained below the 1-ebit Schmidt-rank-2 threshold even on a state whose true 16-stabilizer joint overlap is `0.6334`. Furthermore, `Arm B-Ctrl` and `Arm B-Def` share indistinguishable marginal fidelities (`0.3375` vs `0.3375`), yet 9-basis joint stabilizers cleanly separate 2-photon classical correlation (`0.1799`) from 1-photon sub-capacity collapse (`0.1086`) by `+16.2σ`.
