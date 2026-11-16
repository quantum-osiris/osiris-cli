# Phase 2 Addendum (`E-95`): `HP004_JOINT_STABILIZER` Application Record — SYS-021 Owner Approval Applied (2026-11-16)

Continues the E-series from `PHASE2_ADDENDUM_E94_HP004B.md` (`E-94`). Frozen artifacts are untouched: `PHASE2_REPAIRS.md`, `CLAIM_RECONCILIATION.md`, `claims_ledger.jsonl`, `work/osiris-cli/experiments/hp004*`, and the contents of Zenodo `10.5281/zenodo.24287505` were not present in the application workspace and no write was issued against any of them.

Inputs used (SHA-256):

| File | SHA-256 |
| --- | --- |
| `PHASE2_ADDENDUM_E94_HP004B.md` (as uploaded) | `00a2b9bf90a3e2f2c2ee58b7378c6767caa89a1d64e932cb1fb10a790afdc052` |
| Sandbox PASS modules (`20261115T135621_claims.py_pass.log`, `20261116T100651_claims.py_pass.log`) | `be24ce4268889d41f15507fd5177d645880c88d406d7e7c20d6d55ef1b7bec9b` (reconstructed byte-for-byte from the logs and re-run) |
| Sandbox FAIL modules (both FAIL logs) | `4546e862979e1f2134a76da63f433e0bc05e65eae63fdf3803d0a9618fb73acc` |

## 1. Approval and Application

- `E-94` Section 4 staged the `osiris_cli/claims.py` entry `HP004_JOINT_STABILIZER` as **PROPOSED**, pending owner review (§11.5 / SYS-021).
- Owner approval was given on 2026-11-16. Status under this record: **PROPOSED → APPROVED / APPLIED**.
- Applied artifact: `claims.py` (SHA-256 `7f15dea425c2f7a576c4610ccb1765233b84c3d2be3dd9a14afb2a88034f9741`), containing:
  1. The number-provenance `check_claim` module (the sandbox PASS proposal), and
  2. The `HP004_JOINT_STABILIZER` register entry (verdict `SUPPORTED`), with the approved statement and scoped wording embedded byte-verbatim from `E-94` Section 4, plus the invariants of `E-94` Section 3 as caveat fields.

## 2. Sandbox Reproduction (Independent)

- The baseline module (no number check) fails assertion 1 of `test_proposal` ("an unsourced number must be reported by value") — reproducing both uploaded FAIL verdicts (`20261115T135621`, `20261116T100651`).
- The enhanced module passes all 6 assertions — reproducing both uploaded PASS verdicts.
- The final applied `claims.py` (checker + register) also passes all 6 assertions.

## 3. Firewall Verification of the Entry Itself (`check_claim` vs. the `E-94` record)

| Text checked | vs. `E-94` entry (Section 1) | vs. Sections 1–3 |
| --- | --- | --- |
| Statement (verbatim approved) | flags `8`, `9` | flags `8` |
| Scoped wording (verbatim approved) | flags `0.6334`, `0.7307` | flags `0.6334`, `0.7307` |
| Scoped wording (evidence-aligned) | **0 problems** | **0 problems** |

Findings and dispositions:

1. **Rounded tokens.** The approved Section 4 wording uses `0.6334` / `0.7307`; the `E-94` evidence contains `0.633423` / `0.730713`. The verbatim wording only "passes" against the full addendum because Section 4 quotes itself (circular). The register's canonical `scoped_wording` is therefore the **evidence-aligned** form: full-precision fidelities plus job IDs added, **no numeric value changed**. The verbatim approved wording is preserved in `scoped_wording_verbatim_e94` for audit. This delta is recorded here and involves no change to any measured or derived value.
2. **`8-qubit` / `9-basis`.** The statement's tokens `8` and `9` are not present in the addendum (the `9-basis` witness appears only in the Section 3 invariant prose). They are evidenced by the archived `RESULT.md` in Zenodo `10.5281/zenodo.24287505`. Before merging, run in the repo:
   `python osiris_cli/claims.py HP004_JOINT_STABILIZER work/osiris-cli/experiments/hp004b_fez/RESULT.md`
   and confirm both lists are empty.

## 4. Independent Numerical Re-Verification of `E-94`

- All six `γ₄(F)` bounds in the Section 2 table reproduce to within `5 × 10⁻⁷` ebits (floating-point rounding) from the stated formula `p₀(F) = ¼(√F + √(3(1−F)))²`, `γ₄ = H₂(p₀, (1−p₀)/3 ×3)`: `0.845131`, `0.824807`, `1.146530`, `1.124434`, `1.131991`, `1.103534`.
- Internal σ consistency: σ_eff(Marrakesh) = `0.002635` implied by `z = +50.64` vs the `0.5000` SN ≤ 2 bound, matching the reported `±0.0026` and giving `z = −19.16` vs `F*` (claimed `−19.17`); σ_eff(Fez) = `0.002732` implied by `z = +84.45`, giving `z = +17.13` vs `F*` (as claimed), the `5σ` LCL `0.717053` exactly, and t-LCLs consistent at `df = 5` within rounding.
- **Minor erratum (no claim impact):** the exact 1-ebit threshold root is `F* = 0.683900317`; `E-94` prints `0.683910` (a fifth-decimal typo). `γ₄(0.683910) = 1.00003`, still above 1 ebit, so every conclusion, z-score (17.13 either way), and bound in `E-94` is unaffected. `E-94` is left unmodified; this section supersedes the printed constant for future revisions.

## 5. Hygiene Items Disposition

- **Terhal–Vollbrecht primary citation (PRL 85, 2625 (2000), doi:10.1103/PhysRevLett.85.2625) for the v2.1 manuscript:** NOT applied — the manuscript `.tex` was not provided in the application workspace. Remains pending; do not mutate frozen v2 artifacts in place.
- **`z_above_1ebit_eff` legacy JSON key:** documented in `E-94` Section 3 and carried as a caveat field on the register entry. Archived code and JSONs untouched per the frozen-artifact policy (Zenodo `24287505` SHA256SUMS remain valid).

## 6. Merge Instructions (Repository Side)

1. Back up `osiris-assessment/work/osiris-cli/osiris_cli/claims.py`.
2. If the repo module matches the sandbox baseline (evidence-presence check only), replace it with the applied `claims.py`. If it contains further register content, merge the REGISTER block into it instead of replacing the file; the `check_claim` upgrade is additive and safe to replace wholesale.
3. Re-run the sandbox `test_proposal` (all 6 assertions) and the Section 3 step-2 command against the real `RESULT.md`.
4. Confirm the frozen files are byte-identical after merge: `PHASE2_REPAIRS.md`, `CLAIM_RECONCILIATION.md`, `claims_ledger.jsonl`, `work/osiris-cli/experiments/hp004*`.
5. Record the new `claims.py` SHA-256 (`7f15dea425c2f7a576c4610ccb1765233b84c3d2be3dd9a14afb2a88034f9741`, or the merged variant's) in the phase-2 baseline file per local convention, and mint the `claims_ledger.jsonl` transition for `HP004_JOINT_STABILIZER` per §11.5 if the ledger is to be extended (the ledger file itself remains frozen in this application).
