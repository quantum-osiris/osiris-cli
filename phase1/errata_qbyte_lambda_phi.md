# Errata: qByte whitepaper, ΛΦ "Falsifiable Predictions" and A-field harness (2026-11-22)

- **Historical texts kept unchanged:**
  - `physics/QBYTE_PHYSICS_WHITEPAPER.md`
  - `falsifiable_predictions.md` (v2.1.0)
  - `research/a_field_study/experiments/falsification_harness.py`
- **How numbers were checked:** each was recomputed from CODATA 2018 constants (c, G, ħ, e) with double-precision arithmetic.
- **Status of this document:** an errata proposal. It changes no register verdict.

| # | Location | Original | Defect | Correction | Regression check | Effect on conclusion |
|---|---|---|---|---|---|---|
| 1 | whitepaper §1.1 | θ_lock = 51.843° = arctan(4/π) | arithmetic | arctan(4/π) = 51.85397°. Separately, arccos(1/3) = 70.52878° is the regular-tetrahedron dihedral angle; the "projection" from 70.53° to 51.84° is undefined. | `math.degrees(math.atan(4/math.pi))` | None of the listed coincidences derives the angle. |
| 2 | whitepaper §1.4, §5; predictions §2 | λ_Φ = 2.176435 × 10⁻⁸ s⁻¹ | units | 2.176434 × 10⁻⁸ is the Planck mass in kg. The Planck angular frequency √(c⁵/Għ) is 1.8549 × 10⁴³ s⁻¹. | `sqrt(hbar*c/G)`, `sqrt(c**5/(G*hbar))` | Every equation using λ_Φ as a rate is dimensionally unsupported. The register already has NOT_MEASURED. |
| 3 | whitepaper eq. at line 505 | E = λ_Φ ħ = 2.30 × 10⁻⁴² J ≈ 10⁻¹⁸ eV | conversion | 2.2961 × 10⁻⁴² J = 1.4331 × 10⁻²³ eV | `E/1.602176634e-19` | The energy scale is 5 orders of magnitude smaller than stated. |
| 4 | whitepaper §1.4 | ∇ × E = (2π/λ_φ) n̂ | dimensional | The left side is in V m⁻². The right side is in s when λ_φ is in s⁻¹. | units table | The equation has no physical content as written. |
| 5 | whitepaper (quaternion section) | q ⊗ q* = 1 + δE_release | dimensional | For a unit quaternion, q q* = \|q\|² = 1 identically, and it is dimensionless. | `bt2t.qmul(g, bt2t.conjugate(g)) == (2,0,0,0)` for all 24 Hurwitz units | Unit-quaternion inversion is valid mathematics (used in BT2T). No energy term can appear in it. |
| 6 | predictions §3, §5 | F_Bell ≤ 1 − φ⁻⁸ = 0.9787, "100% validated" | logic | The arithmetic is right (0.978714), but ideal Bell preparation has F = 1 and no derivation of the bound exists. A sample lying below the bound cannot tell a universal ceiling from device noise. | Ideal statevector Bell fidelity is 1 | REFUTED as a universal bound; consistent with the register's `F_MAX`. |
| 7 | predictions header | "STRONGLY VALIDATED (100% compliance)" | status claim | Conflicts with register verdicts. The pasted v2.2.0 ("provisional") is not in the repo. | — | The header should point here. |
| 8 | predictions §6; whitepaper | χ_pc = 0.946 and χ_pc = 0.869 | versioning | These are two different predictions: recovery fractions χ² = 0.894916 and 0.755161. Do not treat them as a range. | — | The recovery law must name one operation and one value before data are collected. |
| 9 | predictions §8 (v2.2.0 text) | α = 1/(90° − θ_lock)² | none | 6.8683 × 10⁻⁴ deg⁻². Correct as arithmetic. | — | Testing a free-minimum fit is not a test of the fixed curvature. |
| 10 | predictions §9 (v2.2.0 text) | T2(N) table with τ_mem | internal consistency | The rows imply τ_mem of 47.09, 45.66, 45.46 and 47.07 ns, so there is no single τ_mem. This 46 ns scale also differs from the 46 μs T2 grid by a factor of 1000. | Recompute `T/(N**-0.5*exp(-0.092*N))` | Freeze one τ_mem and regenerate the table before any test. |
| 11 | predictions §7 (v2.2.0 text) | Ξ = ΛΦ/Γ > 0.7734 | dimensional | With Λ and Φ dimensionless and Γ a rate, Ξ has units of time, so the threshold needs units. | — | Not operationally defined, so not testable yet. |
| 12 | A-field harness lines 45–47 and 101–107 | `verdict = "REFUTED" if is_null_result` | labelling | A simulated observable with zero injected signal, falling below a configured threshold, is a null control, not an empirical refutation. Proposed labels: `BELOW_THRESHOLD_WITH_ZERO_INJECTED_SIGNAL`, and `physical_claim_assessment: UNVERIFIED`. | Add an injection-recovery test: detection probability for signals below, near and above threshold, and a false-positive rate at zero signal. | Detector sensitivity is untested; no physical exclusion follows. |

**What survives as valid mathematics:**
- the binary tetrahedral group 2T ⊂ SU(2);
- unit-quaternion inversion;
- echo, or time-reversal, constructions as diagnostics.

None of these supports qByte emission, vacuum-energy extraction, nonlocal workload effects or a consciousness metric. Those remain unverified hypotheses with no defined observable.
