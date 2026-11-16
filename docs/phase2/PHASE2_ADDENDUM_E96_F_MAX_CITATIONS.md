# Phase 2 Addendum (`E-96`): `F_MAX` Literature Figures Checked Against the Cited Papers (2026-11-16)

Continues the E-series from `PHASE2_ADDENDUM_E95_HP004B_APPLICATION.md` (`E-95`). The frozen Phase 2 records and the Phase 3 approved ledger are left byte-identical; this entry supersedes the figures below wherever they appear in them.

## 1. Figures that are not in the cited papers

Each paper was checked on 2026-11-16: the arXiv full text for Benhelm, Ballance, Gaebler and Ding, the APS abstract for Clark, and the arXiv abstract for Kandala.

| Cited in the Phase 2/3 records as | What the paper reports |
| --- | --- |
| Gaebler et al., PRL 117, 060505 (2016): raw Bell-state fidelity `0.9955(5)`, SPAM-adjusted `0.9991(4)` | Neither number appears. Maximum-likelihood Bell-state fidelity `0.9992(4)`, which "does not include errors due to imperfect \|2,2⟩ state preparation and measurement" and gives the deduced gate error `8(4)×10⁻⁴`; with those effects included, a lower bound of `0.997` on the overall Bell-state fidelity. |
| Ballance et al., PRL 117, 060504 (2016): raw Bell-state fidelity `~0.9973(10)` | Not in the paper. Bell-state infidelity `2.5(7)×10⁻³` *after* SPAM correction (`F = 0.9975(7)` before correcting spin-echo error); inferred two-qubit gate fidelity `99.9(1) %`. Without SPAM correction the apparent infidelity would be larger by about `3ε_SPAM`, with `ε_SPAM = 1.74(7)×10⁻³`. |
| Ballance et al. (2016) as an "RB gate metric" | Its two-qubit gate fidelity is inferred from a prepared Bell state, not from randomized benchmarking. |
| Ding et al., PRX 13, 031035 (2023): raw fluxonium Bell-state fidelity `0.985(2)`, SPAM-mitigated `>0.992` | The paper reports no Bell-state fidelity, state tomography or parity measurement. It reports CZ gate fidelities of `99.85–99.92 %` by interleaved randomized benchmarking. |

Checked and retained:

- Benhelm et al., Nat. Phys. 4, 463 (2008): Bell-state fidelity `99.3(1) %`, computed from measured populations `0.9965(4)` and parity contrast `0.990(1)`; no detection or preparation correction is described.
- Clark et al., PRL 127, 130505 (2021): Bell-state infidelity `6(3)×10⁻⁴`, measured "without subtraction of experimental errors".
- Kandala et al., PRL 127, 130501 (2021): CNOT fidelity `99.77(2) %` by interleaved randomized benchmarking (a gate fidelity, not a Bell-state fidelity).

## 2. Effect on the claim

None on the verdict. `F_MAX` stays `REFUTED`. Benhelm 2008 (`0.993(1)`) and Clark 2021 (`≈ 0.9994`, no error subtraction) each exceed `1 − φ⁻⁸ = 0.978714`, and so do the gate fidelities Ballance and Gaebler inferred from Bell states (`0.999(1)`, `0.9992(4)`). With Ding 2023 withdrawn, no superconducting Bell-state measurement is cited.

Minor erratum, no comparison affected: the Phase 2 records print `1 − φ⁻⁸` as `0.978701`. It is `0.978714` (`φ⁸ = 46.978714`), as `osiris_cli/claims.py`'s `one_minus_phi_power` check computes.

## 3. Applied (owner approved dropping the unverified figures, 2026-11-16)

- `osiris_cli/claims.py`, `F_MAX`: finding and evidence restated from the checked figures; Ding 2023 removed; `one_minus_phi_power` check values `(0.993, 0.999, 0.9992, 0.9994)`.
- `tests/test_phase2_claims_ledger.py`: asserts that Benhelm 2008 and Clark 2021 are cited, and that Ding 2023 and the unsupported values are not.

Not modified (frozen records; read their `NB-07` / `F_MAX` figures through this entry): `phase2/claims_reconciliation_v2.py` (`LIKE_FOR_LIKE_BELL_STATE_BENCHMARKS`), `docs/phase2/claims_ledger_phase2_proposed.jsonl`, `docs/phase2/PHASE2_CLAIM_CORRECTIONS.md` §2.2, `docs/phase2/CLAIM_CHANGESET_FOR_OWNER_APPROVAL.md`, `docs/phase2/PHASE2_EXECUTIVE_REPORT.md`, and `docs/phase3/claims_ledger_phase3_approved.jsonl`.
