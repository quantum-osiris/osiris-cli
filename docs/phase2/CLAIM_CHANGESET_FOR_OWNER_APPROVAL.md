# Proposed Claim Register Changeset for Owner Approval (`CLAIM_CHANGESET_FOR_OWNER_APPROVAL.md`)

**Governance Status:** `Proposed only` — **Awaiting Explicit Owner (PI: Devin Phillip Davis) Approval under SYS-021 / §11.5.**  
**Protected File:** `/home/osiris/osiris-assessment/work/osiris-cli-phase2/osiris_cli/claims.py` (`sha256: 8563ac0fa78d69b259ea6ad35d498182737f5ac2ddfcd525c5a4a7ad47f47fb1`) — **0 bytes modified.**  
**Proposed Phase 2 JSONL Ledger:** `docs/phase2/claims_ledger_phase2_proposed.jsonl` (`sha256: 87638fcf7a065d2bcb0bdfa03898b9fdffda5764a2a74020b0a7f5c0805ee241`)

---

## 1. Proposed Updates to Authoritative `osiris_cli/claims.py` (Do Not Apply Without Approval)

| Claim ID in `osiris_cli/claims.py` | Current Verdict | Proposed Verdict | Exact Proposed Modification & Rationale |
|---|---|---|---|
| **`W2_ROBUSTNESS`** (L352–358) | `REPORTED` (*"legit: measured and on record, not pre-registered"*) | **`ARTIFACT`** (or split: `W2_THETA_SWEEP_COUNTS` = `REPORTED` [owner-reported], `W2_ROBUSTNESS` = `ARTIFACT` / `QUARANTINED`) | Current entry states *"the one idea the 2026-09-20 assessment found no artifact behind."* Inspection of `w2_robustness_paper.tex` (`sha256: b1386f27...`) revealed four mathematical artifacts: Eq. (7) computes independent-product cost $C(p,q)$ where $\sqrt{C(p_{\text{unif}},p_{\text{unif}})} = 0.5092 \neq 0$, omits Layer-0 Hadamards in the $0.760^{27}$ interference argument, overlooks exact Z-basis phase cancellation of `CP`/`RZ`, and divides $0.0398$ by $1/2818.5$ (`112×`) instead of $2^{-27}$. |
| **`F_MAX`** (L159–176) | `REFUTED` | **`REFUTED`** (retain verdict; refine finding & `_check_lines` wording) | Separate RB gate fidelities (`Kandala 2021` `0.9977`, `Ballance 2016` gate RB `0.999`) from strictly like-for-like raw Bell-state preparation fidelities (`Benhelm 2008` `0.993(1)` raw, `Gaebler 2016` `0.9955(5)` raw / `0.9991(4)` SPAM-adjusted, `Clark 2021` `0.9994(3)` raw, `Ding 2023` `0.985(2)` raw superconducting fluxonium). Update `one_minus_phi_power` tuple to `(0.993, 0.9955, 0.9994)` (all raw non-SPAM-subtracted Bell states). |
| **`STAGGERED_DD`** (L319–328) | `SUPPORTED` | **`SUPPORTED`** (narrow scope + structured comparator note) | Keep `SUPPORTED` within its stated chain/backend scope, explicitly note that `ibm_fez` raw histograms (`10.5281/zenodo.22855102`, `22870287`) are owner-deposited external records while `ibm_marrakesh` (`dau0q3qhcrkc73durtgg`, `qpu_stag_xy8_v2`, `E2`) has local raw counts, cite `Tripathi et al., PRApplied 18, 024068 (2022)` prior art, and add negative scope boundary from `E3` (`ibm_fez` adaptive transfer null `-0.80 ± 0.60 pp`, $p=0.218$) and `E4` (`1.03–1.11 pp` A/A drift floor). |
| **`HERON_R2_STAG_XY8`** (L302–309) | `SUPPORTED` | **`REPORTED`** or **`SUPPORTED` (single-edge scoped)** | Clarify that in `qpu_stag_xy8_v2/final_synthesis.json`, protection against `no-DD` (`+0.4360`) held across all 7 edges, whereas the gain **over synchronous XY8** was statistically significant on benchmark edge `(14, 15)` (`+0.0186`, `95% CI [+0.0107, +0.0250]`) while the 7-edge pooled contrast over synchronous XY8 had `95% CI [-0.0045, +0.0138]` (including zero), and later `E4` (`dbm9ctu74qds73drs9j0`) on `(14, 15)` was flat within the `~1.05 pp` A/A drift floor. |
| **`ERROR_SUPPRESSION_1E6`** (L111–119) | `REFUTED` | **`REFUTED`** (narrow regex `r"tesseract"` so external color-code decoder is not misflagged) | Replace bare pattern `r"tesseract"` with context-qualified pattern excluding `arXiv:2610.24077` / `color[- ]code` / `TESSERACT-MAP-02`, preventing the external Quantinuum A* color-code decoder (`TESSERACT_COLOR_CODE_DECODER_EXT`) from colliding with the refuted 40-qubit OSIRIS circuit. |
| **`LOCAL_QVM`** (L205–210) | `REFUTED` | **`REFUTED`** (clarify `osiris_local_qvm` quaternion module vs repaired `qbyte_system` classical statevector module) | Distinguish `osiris_local_qvm` (single-Bloch-quaternion per qubit, incapable of entanglement) from `qbyte_system` (256-amplitude classical NumPy statevector simulator repaired in Phase 2, which simulates 8-qubit unitaries accurately on CPU but is **not** a physical quantum substrate). |

---

## 2. Proposed Transitions in the 44-Row Phase 1/2 Ledger (`claims_ledger_phase2_proposed.jsonl`)

1. **`NB-11`:** `MEASURED` $\longrightarrow$ `QUARANTINED_PENDING_RECOMPUTATION` (with `historical_measurements_classification = "HARDWARE-OBSERVED / OWNER-REPORTED"`).
2. **`NB-07`:** Retain `REJECTED` (`REFUTED`), replacing gate-fidelity reasoning with strictly like-for-like Bell-state preparation benchmarks (`Benhelm 2008`, `Gaebler 2016`, `Ballance 2016`, `Clark 2021`, `Ding 2023`).
3. **`NB-14, NB-15, NB-17, NB-27, NB-28, NB-37`:** `MEASURED` $\longrightarrow$ `HARDWARE-OBSERVED / OWNER-REPORTED` (attaching structured duration/comparator schema to `NB-15`).
4. **`NB-38, NB-39, NB-40, NB-41, NB-42`:** `VERIFIED` $\longrightarrow$ `MATHEMATICALLY_VERIFIED_ONLY` (explicitly marking textbook geometry/Cauchy–Schwarz derivations as non-hardware, non-novel identities).

---

## 3. Owner Authorization Checklist

To approve applying any subset of Section 1 or Section 2 transitions:
- [ ] Approve `W2_ROBUSTNESS` (`NB-11`) transition to `QUARANTINED_PENDING_RECOMPUTATION` / `ARTIFACT`.
- [ ] Approve `F_MAX` (`NB-07`) like-for-like Bell-state citation update.
- [ ] Approve `STAGGERED_DD` & `HERON_R2_STAG_XY8` scope & A/A drift-floor annotations.
- [ ] Approve `ERROR_SUPPRESSION_1E6` regex disambiguation for `TESSERACT_COLOR_CODE_DECODER_EXT` and `TESSERACT_MAP_02`.
