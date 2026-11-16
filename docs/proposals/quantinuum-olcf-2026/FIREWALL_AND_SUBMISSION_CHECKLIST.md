# Dual-Track Firewall & Pre-Submission Checklist (`quantinuum-olcf-2026`)

## 1. Mandatory Separation of Scope (Commercial vs. DOE Open Science)

| Policy Dimension | Track A: Quantinuum Startup Partner Program | Track B: DOE OLCF QCUP Open-Science Proposal |
| :--- | :--- | :--- |
| **Submission File** | `track_a_quantinuum_startup_partner/QUANTINUUM_STARTUP_PARTNER_SUBMISSION.md` | `track_b_olcf_qcup_open_science/OLCF_QCUP_PROJECT_PROPOSAL_V2.md` |
| **Credit Sizing File** | N/A (commercial partner program scoping) | `track_b_olcf_qcup_open_science/MYOLCF_CREDIT_ALLOCATION_JUSTIFICATION.md` |
| **Permitted Scope** | Commercial `DNA-Lang` / `OSIRIS` productization, enterprise & defense Nexus Evidence Capsules, TKET integration | **100% open, publishable fundamental research ONLY** (benchmarking entanglement witnesses, debiased optimal transport, and `BP+OSD` / MWPM QEC) |
| **Prohibited on Track B** | N/A | **NO** export-controlled data, **NO** PHI, **NO** defense deliverables, **NO** proprietary closed-source artifacts on QCUP allocations |
| **Repository Isolation** | Commercial/product repositories (`quantum-ide` / enterprise adapters) | Dedicated public open-science benchmark repository + Zenodo open deposits (`CC-BY-4.0` / `MIT` / `Apache-2.0`) |
| **Required Citation** | Standard commercial terms | Verbatim DOE OLCF Contract `DE-AC05-00OR22725` acknowledgement in every paper/preprint |

## 2. Final Owner Steps Before External Submission

1. **Track A (Quantinuum Startup Partner Web Form):**
   - Fill contact fields (`Devin Phillip Davis`, `Agile Defense Systems LLC`, email, country of residency).
   - Check `Software`, `AI`, `Optimization`, `Physics` *(optional: `Cryptography`)*.
   - Paste the essay from `track_a_quantinuum_startup_partner/QUANTINUUM_STARTUP_PARTNER_SUBMISSION.md`.
2. **Track B (OLCF QCUP Application Form):**
   - Fill the 4 bracketed PI contact/citizenship fields in Section 2 of `track_b_olcf_qcup_open_science/OLCF_QCUP_PROJECT_PROPOSAL_V2.md`.
   - Select **"OLCF Quantum Computing User Program"** on the OLCF Project Application Form and submit Sections 1–12.
   - Upon QRUC approval and Project ID issuance, run the Month-1 `H2-1E` / `Helios-1E` Nexus emulator check and paste `track_b_olcf_qcup_open_science/MYOLCF_CREDIT_ALLOCATION_JUSTIFICATION.md` (`21,500 HQC` + `600s` IBM) into `myOLCF`.
3. **Permanent Zenodo DOIs Linked Across Both Tracks (`COMPLETE`):**
   - `OSIRIS-HP-004` & `HP-004B` (`585,728` raw shots): **`10.5281/zenodo.24287505`**
   - `OSIRIS-QEC-001` (`393,216` raw shots, Job `dbtl2vb2kf8s73c9a0d0`): **`10.5281/zenodo.24293756`**
