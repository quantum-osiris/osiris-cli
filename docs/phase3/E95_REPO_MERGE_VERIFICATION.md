# Phase 3 / E-95 Repository Merge, v2.1 Manuscript, and Security Hardening Record (2026-11-16)

Continues the E-series from `docs/phase2/PHASE2_ADDENDUM_E94_HP004B.md` (`E-94`) and `docs/phase2/PHASE2_ADDENDUM_E95_HP004B_APPLICATION.md` (`E-95`) on the unified `main` branch.

## 1. Repository-Side E-95 Merge & Firewall Verification

1. **Unified Register & Number-Provenance Firewall (`osiris_cli/claims.py`):**
   - SHA-256: `9d4fbd6a80a3fb1ceb01269dce84a74ebd71cbe7c5f1963b3c5f562ea9e48b86`
   - Integrates:
     - `check_claim(text, evidence)` (sandbox 6-assertion number-provenance firewall) and `verify_register(claim_id, evidence_text)` + CLI entrypoint (`python osiris_cli/claims.py <claim_id> <evidence_files...>`).
     - `HP004_JOINT_STABILIZER` (`SUPPORTED`, with `terhal_vollbrecht_d4` convex-roof calculator, evidence-aligned `scoped_wording`, `scoped_wording_verbatim`, and 5 invariants as `caveats`).
     - `QEC001_SYNDROME_SCALING` (`SUPPORTED`, preserving un-doctored `QEC-H3 = false` on `0.1265–0.1284 > 0.1200`).
     - Phase 2 / SYS-021 approved claim repairs (`W2_ROBUSTNESS` -> `ARTIFACT`, `W2_THETA_SWEEP_COUNTS` -> `REPORTED`, `F_MAX` raw Bell DOIs, `STAGGERED_DD` duration scoping + Tripathi et al. 2022, `ERROR_SUPPRESSION_1E6` regex disambiguation, `LOCAL_QVM` classical simulator scoping).
2. **E-95 Section 3 Step-2 Verification Against Archived Evidence:**
   - Against `experiments/hp004b_fez/RESULT.md` alone: `verify_register("HP004_JOINT_STABILIZER", result_md)["statement"] == []` (confirming tokens `8` and `9` in `8-qubit` / `9-basis` are grounded in `RESULT.md`).
   - Against `docs/phase2/PHASE2_ADDENDUM_E94_HP004B.md` + `experiments/hp004b_fez/RESULT.md`:
     ```json
     {
       "statement": [],
       "scoped_wording": []
     }
     ```
     Both lists are empty (`0` problems).

## 2. Non-Mutating Version 2.1 (`v2.1`) Manuscript (`papers/hp004_joint_stabilizer_heron_r2_v2_1/`)

Without modifying any file in `papers/hp004_joint_stabilizer_heron_r2/` (`Zenodo 10.5281/zenodo.24287505` v2.0), Version 2.1 adds:
1. Primary isotropic-state $E_F$ citation: **B. M. Terhal and K. G. H. Vollbrecht, *Phys. Rev. Lett.* 85, 2625–2628 (2000), `doi:10.1103/PhysRevLett.85.2625`** alongside Vollbrecht & Werner, *Phys. Rev. A* 64, 062307 (2001).
2. Exact 1-ebit threshold root $F_* = 0.683900$ (`0.683900317`, correcting v2.0's fifth-decimal `0.683910` typo; $+17.13\sigma$ and all bounds unchanged).
3. Explicit schema note in Remark 1 and Section 2 documenting that legacy JSON key `z_above_1ebit_eff` in frozen `physical_hp004*_results.json` computes $(F_{\mathrm{joint}} - 0.5000)/\mathrm{SE}_{\mathrm{eff}}$ ($Z$ above the `0.5000` $\mathrm{SN}\le 2$ bound).

| File (`papers/hp004_joint_stabilizer_heron_r2_v2_1/`) | SHA-256 |
| --- | --- |
| `hp004_joint_stabilizer_heron_r2_v2_1.pdf` | `766f75216c6ee73ce1b69ca7a56311ad643289e785da34c716b19e9e7fd08593` |
| `hp004_joint_stabilizer_heron_r2_v2_1.tex` | `ee128ff3a824a5d61dc4a5b49ea684e6a6b8285b8be00811ddec8a1d672c29aa` |
| `generate_paper_assets_v2_1.py` | `a35b608b88968f85cdd1837c11ee78fe4585a6b8ab26d4bb2e58b5912d52777a` |

## 3. Frozen Artifact & Hardened Archive Verification

| Artifact | Verified SHA-256 | Status |
| --- | --- | --- |
| `/home/osiris/CLAIM_RECONCILIATION.md` | `37b6c9c6f9f5dcdc3a7d6ce99db57514dc474840c00e13f3007050132712883a` | Byte-identical (frozen) |
| `/home/osiris/PHASE2_REPAIRS.md` | `943d826c55bc78e960a5463267c9a5c19091b36277ce447f839ceae71f3f2fb1` | Byte-identical (frozen) |
| `/home/osiris/claims_ledger.jsonl` | `567a1ffd6c425eab3b031d5191ecd3fbd1198b89233bbbe87f5c1ee0fa25fb1f` | Byte-identical (frozen; Phase 3 approved copy at `docs/phase3/claims_ledger_phase3_approved.jsonl`) |
| `/home/osiris/PHASE2_ADDENDUM_E94_HP004B.md` | `00a2b9bf90a3e2f2c2ee58b7378c6767caa89a1d64e932cb1fb10a790afdc052` | Byte-identical (frozen) |
| `/home/osiris/PHASE2_ADDENDUM_E95_HP004B_APPLICATION.md` | `bc6986be755df069fd6ffec5baa72444cabff253334ede27ba0c321a894caf2c` | Byte-identical (frozen) |
| `papers/hp004_joint_stabilizer_heron_r2/` & `experiments/hp004*` | All 24/24 Zenodo `24287505` digests | Byte-identical (frozen) |
| `/home/osiris/quantum-ide-11227528.zip` | `f1166cb2ea060dddb054ca65def67e1897e6b7c6dfa9b945efb1b5a57ed2e474` | Hardened 8-pattern exclusion verified (`553` entries; only template placeholder `apikey.json.template`, `0` secrets) |
