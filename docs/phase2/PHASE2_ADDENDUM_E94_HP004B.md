# Phase 2 Addendum (`E-94`): `OSIRIS-HP-004B` (`ibm_fez`) Completion & Terhal–Vollbrecht $E_F$ Continuity (2026-11-16)

Continues the E-series from `PHASE2_REPAIRS.md` (last entry `E-93`) without modifying `PHASE2_REPAIRS.md`, `CLAIM_RECONCILIATION.md`, `claims_ledger.jsonl`, or `osiris_cli/claims.py` (§13.2 immutability; baseline SHA-256s preserved; claim-register transitions remain **PROPOSED, pending owner review under §11.5 / SYS-021**).

## 1. Entry `E-94` (Supersedes `QUEUED` Status in `E-93` for `ibm_fez` Job `dbtj61bnlcms73ajc380`)

| # | Evidence | Class | Proposed impact |
| --- | --- | --- | --- |
| E-94 | `OSIRIS-HP-004B` (`ibm_fez`, Job `dbtj61bnlcms73ajc380`, `286` circuits × `1,024` shots = `292,864` raw shots, pre-registered commit `0c5a5a39` `16:01:38Z` < IBM `created: 16:44:53Z`, QPY SHA-256 `6d7e5a6d17a002b5...`) completed and is archived in `work/osiris-cli/experiments/hp004b_fez/` and Zenodo v2 (`10.5281/zenodo.24287505`, commit `1a33ffe7`, superseding v1 `10.5281/zenodo.24285577`). Independent reconstruction from IBM job exports reproduces `Arm B` $F_{\mathrm{joint}} = 0.633423$ on `ibm_marrakesh` (`dbti1fbsbsbs73cb2vu0`) and $F_{\mathrm{joint}} = 0.730713$ on `ibm_fez` (`dbtj61bnlcms73ajc380`, 6-block 95% Student-$t$ CI `[0.7237, 0.7377]`). Applying the $4 \times 4$ isotropic-state convex-roof bound $\gamma_4(F)$ of Terhal & Vollbrecht (*Phys. Rev. Lett.* 85, 2625, 2000) and Vollbrecht & Werner (*Phys. Rev. A* 64, 062307, 2001), where $\gamma_4(F_*) = 1\text{ ebit}$ at $F_* \approx 0.683910$: `ibm_marrakesh` yields $E_F \ge 0.8451\text{ ebits}$ (`< 1 ebit`), whereas `ibm_fez` yields $E_F \ge 1.1465\text{ ebits}$ (`1.147 ebits`), and at the reported 95% lower fidelity limit $F = 0.7237$ yields $E_F \ge 1.1244\text{ ebits}$ (`1.124 ebits > 1 ebit`). | MEASURED / INDEPENDENTLY RECONSTRUCTED / DERIVED | Retain `E-93` restriction for `ibm_marrakesh` (supports Schmidt number $\ge 3$, **not** $E_F > 1\text{ ebit}$). Update `ibm_fez` disposition: conditionally supports **both** Schmidt number $\ge 3$ and entanglement of formation $E_F > 1\text{ ebit}$ (including at the 95% lower confidence limit), as a witness-derived bound within standard quantum mechanics. |

## 2. Exact Numerical Concordance Table ($d = 4$ Terhal–Vollbrecht / Vollbrecht–Werner Bound)

For a $4 \times 4$ bipartite state $\rho$ with maximally entangled overlap $F = \langle \Psi_4 | \rho | \Psi_4 \rangle \in [\frac{1}{4}, \frac{3}{4}]$,
$$p_0(F) = \frac{1}{4}\left(\sqrt{F} + \sqrt{3(1 - F)}\right)^2, \qquad E_F(\rho) \ge \gamma_4(F) = H_2\!\left(p_0, \frac{1-p_0}{3}, \frac{1-p_0}{3}, \frac{1-p_0}{3}\right),$$
with Schmidt-number-$\le 2$ ceiling $F_{\mathrm{SN}\le 2} = 0.5000$ (Terhal & Horodecki, *Phys. Rev. A* 61, 040301(R), 2000) and $1$-ebit threshold $F_* \approx 0.683910$:

| Device & Estimate (`Arm B`, 32 CZs) | Joint Fidelity $F_{\mathrm{joint}}$ | Exact $\gamma_4(F)$ (ebits) | Reviewer-Reconstructed Bound | Supported Claim Scope |
| --- | --- | --- | --- | --- |
| **`ibm_marrakesh`** point estimate (`dbti1fbsbsbs73cb2vu0`) | `0.633423` | `0.845131` | `0.845 ebits` | $\mathrm{SN}(\rho) \ge 3$ (`+50.64σ`); **not** $E_F > 1\text{ ebit}$ |
| `ibm_marrakesh` 95% Student-$t$ lower limit | `0.626700` | `0.824807` | — | $\mathrm{SN}(\rho) \ge 3$; **not** $E_F > 1\text{ ebit}$ |
| **`ibm_fez`** point estimate (`dbtj61bnlcms73ajc380`) | **`0.730713`** | **`1.146530`** | **`1.147 ebits`** | **Both $\mathrm{SN}(\rho) \ge 3$ (`+84.45σ`) and $E_F > 1\text{ ebit}$ (`+17.13σ`)** |
| **`ibm_fez` 95% Student-$t$ lower limit** (6 blocks, `df=5`) | **`0.723700`** | **`1.124434`** | **`1.124 ebits`** | **Both $\mathrm{SN}(\rho) \ge 3$ and $E_F > 1\text{ ebit}$** |
| `ibm_fez` 95% shot-bootstrap lower limit | `0.726100` | `1.131991` | — | Both $\mathrm{SN}(\rho) \ge 3$ and $E_F > 1\text{ ebit}$ |
| `ibm_fez` $5\sigma_{\mathrm{eff}}$ lower limit | `0.717053` | `1.103534` | — | Both $\mathrm{SN}(\rho) \ge 3$ and $E_F > 1\text{ ebit}$ |

## 3. Mandatory Continuity & Scope Invariants Across All Documents and Pitches

1. **Witness-Derived Bound, Not Direct Entropy Measurement:** The `0.845`, `1.147`, and `1.124 ebits` values are convex-roof lower bounds on *entanglement of formation* $E_F(\rho)$ computed from the 9-basis joint Bell-stabilizer fidelity witness, **not** direct measurements of von Neumann entanglement entropy $S(\rho_R)$.
2. **Conditional Validity:** Every bound remains conditional on the target-state mapping $|\Psi_4\rangle = |\Phi^+\rangle_{R_2 M_1'} \otimes |\Phi^+\rangle_{R_1 M_2'}$, faithful implementation of the local Pauli measurement settings (`H`, `Sdg` + `H`, `Z`), and the block/shot uncertainty model.
3. **Standard Quantum Mechanics Only:** Both runs test established quantum entanglement dimensionality and channel recovery on noisy superconducting transmons; neither claims a new physical law or non-Clifford computational advantage.
4. **Code & Bibliography Hygiene Notes (Do Not Mutate Frozen Zenodo `24287505` Artifacts In Place):**
   - **Legacy JSON Key Warning:** In `experiments/hp004/hp004_joint_stabilizer.py` (`L458`, `L485`) and archived `physical_hp004*_results.json`, the field `z_above_1ebit_eff` computes `(f_joint - 0.5) / se_eff` ($Z$ above the `0.5000` **Schmidt-number-2** bound, giving `+50.64` on Marrakesh and `+84.45` on Fez), **not** $Z$ above the $1$-ebit $E_F$ threshold $F_* = 0.683910$ (`-19.17σ` on Marrakesh, `+17.13σ` on Fez). Downstream readers must treat `z_above_1ebit_eff` strictly as `z_above_sn2_bound`.
   - **Primary Citation for v2.1:** Add B. M. Terhal & K. G. H. Vollbrecht, *"Entanglement of Formation for Isotropic States"*, *Phys. Rev. Lett.* **85**, 2625 (2000) (`doi:10.1103/PhysRevLett.85.2625`) alongside Vollbrecht & Werner, *Phys. Rev. A* **64**, 062307 (2001) in any future v2.1 manuscript revision.

## 4. Proposed `osiris_cli/claims.py` Entry (Pending Owner Approval Under SYS-021 / §11.5)

- **Proposed ID:** `HP004_JOINT_STABILIZER`
- **Proposed Verdict:** `SUPPORTED` (with explicit device-split Marrakesh vs. Fez bounds and witness caveats)
- **Statement:** *"Layout-locked 8-qubit Hayden–Preskill 9-basis joint stabilizer recovery witnesses Schmidt number ≥ 3 on ibm_marrakesh and ibm_fez, and witnesses entanglement of formation > 1 ebit on ibm_fez"*
- **Scoped Wording:** Cite `10.5281/zenodo.24287505` (superseding `10.5281/zenodo.24285577`); state explicitly that `ibm_marrakesh` (`F_joint = 0.6334`, `E_F >= 0.845 ebits`) certifies $\mathrm{SN} \ge 3$ only, while `ibm_fez` (`F_joint = 0.7307`, 95% LCL `0.7237`, `E_F >= 1.147 ebits` / `1.124 ebits` at 95% LCL) certifies both $\mathrm{SN} \ge 3$ and $E_F > 1\text{ ebit}$ via the Terhal–Vollbrecht fidelity witness (conditional on measurement/target assumptions; not a direct entropy measurement or new physics).
