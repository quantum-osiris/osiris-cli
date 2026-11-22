# Phase 1 reconciliation (2026-11-22)

- **Reviewer:** Claude, an AI agent in a Claude Code session. This is self-review: no independent reviewer ran.
- **Pinned revisions:**
  - osiris-cli `8e963c77`
  - dnalang-core `90b6805`
  - quantum-ide `190c326`
  - quantumosiris-site `7ae1104`
- **Claim IDs** refer to `claim_evidence_map.jsonl`.
- **What this record changes:** nothing in the canonical register, production or provider state. Each proposed correction below is for the owner to decide.

## R-1 Engineering closures (E-44, E-45, E-49 to E-52)

| Lead | Verified now | Still open |
|---|---|---|
| E-49/E-50 pytket and Nexus tests | CI run 44989410176 at `8e963c77`: the `quantinuum` job installs pytket 2.18.5 and qnexus 0.51.4, collects 38 items, and fails on any skip. Local rerun: 34 passed, 0 skipped in the two named files (ENG-NEXUS-2). The 12 failures on main before the fix were reproduced and are fixed at the root cause (ENG-TKET-1, ENG-TKET-2). | Compatibility with the live Nexus service (ENG-NEXUS-4). Real-service contents of the fields the attestation step reads (ENG-NEXUS-5). |
| E-51 dnalang-core CI | Run 44989084645 at `90b6805`: success. | None. |
| E-52 CAGE wording | No old strings remain in source at `190c326`. The alias `www.quantumagentlab.com` points to deployment `dpl_6gdekc1…`. | That deployment's source binding is metadata the operator supplied (`--meta sourceCommit`), not a Vercel Git link. The rendered check covered only 3 routes. |
| E-45 "reconnect quantum-ide's Vercel Git link from ENKI-420" | **The premise is false** (ENG-DEPLOY-3). The Vercel API reports no Git link at all for quantum-ide or quantumosiris-site. | Linking needs the Vercel GitHub App to be installed on `quantum-osiris`. Git-linked deploys of the private site repo were BLOCKED on the current plan. |
| E-44 domain move | The alias `www.quantumosiris.com` points to quantumosiris-site deployment `dpl_2Vqd7…`. | Authorization rests on the owner decision record pasted into the conversation (declared, not inspected). |

**Correction to an earlier statement by this agent (ENG-TKET-3).** The agent first said the old tket wrapper "drops a feed-forward correction". Reproduction showed pytket only rewrote `x` as `u3`; all corrections were present. The PR #72 description was corrected, and the claim is recorded here as REFUTED.

**Second correction to this agent's own statements.** In the conversation the agent said arccos(1/3) ≈ 70.53° "is not the dihedral angle". That is wrong: the dihedral angle of a regular tetrahedron is arccos(1/3) ≈ 70.529°.

## R-2 NCLM-ARCH-1 chronology (E-46)

| Local time, 2026-10-31 to 11-01 (-05:00 assumed) | Event | Source |
|---|---|---|
| 10-31 21:48 to about 11-01 00:13 | crsm s200 runs, 12,000 steps, 2.418 h | `results.json` run_id and hours |
| 00:15 to about 01:36 | standard s200 runs, 1.349 h | same |
| 01:37 (to about 03:09) | crsm s201 starts | same |
| 02:56:55 | **Amendment 3 committed** (`da96afed`): scope set to N = 2 | git |
| 02:57:54 | standard s201 relaunched from step 0 | `results.json` |
| about 04:22 | last run ends | derived |
| 04:25:19 | result committed (`e2679dad`) | git |

- **Supported:** Amendment 3 was adopted after both seed-200 runs had finished (SCI-ARCH1-3). This assumes run IDs use the same time zone as the commits.
- **Unknown:** whether seed-200 held-out scores were looked at before the amendment (SCI-ARCH1-4). The run directories are not on this machine.
- **Preserve both rules:**
  - FAIL under Amendment 3, N = 2 (SCI-ARCH1-1);
  - incomplete against the 8-seed registration (SCI-ARCH1-2).

**R-3 Proposed correction, for the owner.** PR #75 set `CRSM_ARCH` to `REFUTED`, carrying the exact Amendment 3 wording, as approved in the owner record (E-46). The register's verdict set has no value for "failed under an amended, incomplete design". `REFUTED` ("tested and failed") can be read as stronger than the evidence. Two options, neither applied:
- (a) add a verdict such as `FAIL_AMENDED_INCOMPLETE`, with standing "not supported";
- (b) keep `REFUTED` and make the caveats render with the verdict wherever it is shown.

The registration itself must not be rewritten.

## R-4 ADV-0 (E-47)

| Proposition | Status |
|---|---|
| PR #73 merged, and main CI passed at the merge | SUPPORTED (SCI-ADV0-1) |
| A classical attack wins all 30 cells | SUPPORTED. Recomputed from the raw rows (SCI-ADV0-2); the attacks were not rerun in this pass. |
| Predicted hardware peak | A model output only (SCI-ADV0-3) |

The board's "Done" is not used as evidence for any of these.

## R-5 XTOP-005 timing (E-48)

All times are UTC, original values kept. "Local" means recorded by this machine's clock.

| Time | Clock | Event |
|---|---|---|
| 18:37:20.195 | local | circuits locked (`lock.json`) |
| 18:53:24.065 | Zenodo server | preregistration record 24425416 created |
| 18:53:43.247 | local | job submitted (`job_receipt.json`) |
| 18:55:12.526 | local | results received |
| 19:06:02.677 | Zenodo server | paper record 24425628 created |

- **Ordering:** the preregistration record precedes local submission by 19 seconds, but the two times come from different clocks. The claim "preregistered before the job ran" is INCONCLUSIVE until IBM's own `creation_date` is retrieved (SCI-XTOP-1). That retrieval is BLOCKED in this session.
- **Metadata defect:** Zenodo's `publication_date` field says 2026-11-20, before the record existed (SCI-XTOP-2). Correct it in a new version or note it; never backdate.
- **Simulation result:** the H2-like noise-model result stays SIMULATED (SCI-XTOP-5).

## R-6 Physics drafts (CRSM/qByte, ΛΦ, A-field)

The full errata are in `errata_qbyte_lambda_phi.md`; the historical drafts are kept as they are.

1. **The pasted correction table misattributes θ_lock (PHY-2).** The draft writes θ_lock = arctan(4/π) = 51.843°. That arithmetic is wrong too: arctan(4/π) = 51.854° (PHY-1). The draft separately calls 70.53° "projected as 51.84°", without defining any projection.
2. **The repo copy of `falsifiable_predictions.md` is v2.1.0** and still says "STRONGLY VALIDATED (100% compliance)" (PHY-7). The softer v2.2.0 text exists only in a pasted message.
3. **The A-field harness labels null simulations as REFUTED** (PHY-9). That is a labelling defect in code, not a finding about physics.

## R-7 Statements not to repeat without qualification

| Statement | Must be qualified as |
|---|---|
| "osiris nexus works with Quantinuum Nexus" | unit-tested against SDK signatures; never run against the live service |
| "HARDWARE_ATTESTED is verified by Nexus" | the logic is in place; it has not yet been exercised on a real job |
| "XTOP-005 was preregistered before the job ran" | provider timing is unverified (R-5) |
| "NCLM-ARCH-1 refuted CRSM" | FAIL under Amendment 3 (N = 2); incomplete against the 8-seed registration |
| "the site is deployed from Git" or "has commit provenance" | the commit SHA is metadata the operator supplied, with no Vercel Git link |
| "θ_lock = 51.843° = arctan(4/π)" | arithmetically false |
| "F_Bell ≤ 0.9787 validated" | REFUTED as a universal bound |
| "A-field hypothesis refuted" | simulated nulls only; detector sensitivity untested |
| Any "2T group primitive" from the earlier harness | the earlier harness fails necessary conditions (BT2T-1) |
