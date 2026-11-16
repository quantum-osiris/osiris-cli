"""Independent re-verification of the E-94 Terhal-Vollbrecht / Vollbrecht-Werner table
(PHASE2_ADDENDUM_E94_HP004B.md, Section 2) from the stated formulas."""
import math

TCRIT_95_DF5 = 2.570582  # two-sided 95% Student-t critical value, df=5


def p0(F):
    return 0.25 * (math.sqrt(F) + math.sqrt(3.0 * (1.0 - F))) ** 2


def gamma4(F):
    """4-ary entropy of (p0, (1-p0)/3 x3), per the addendum's stated formula."""
    p = p0(F)
    q = (1.0 - p) / 3.0
    return -(p * math.log2(p) + 3.0 * q * math.log2(q))


rows = [
    ("ibm_marrakesh point", 0.633423, 0.845131),
    ("ibm_marrakesh 95% t-LCL", 0.626700, 0.824807),
    ("ibm_fez point", 0.730713, 1.146530),
    ("ibm_fez 95% t-LCL", 0.723700, 1.124434),
    ("ibm_fez bootstrap LCL", 0.726100, 1.131991),
    ("ibm_fez 5-sigma LCL", 0.717053, 1.103534),
]
print("== gamma_4(F) table reproduction ==")
maxdiff = 0.0
for name, F, claimed in rows:
    g = gamma4(F)
    d = g - claimed
    maxdiff = max(maxdiff, abs(d))
    print(f"  {name:26} F={F:.6f}  recomputed={g:.6f}  table={claimed:.6f}  diff={d:+.2e}")

lo, hi = 0.5, 0.75
for _ in range(200):
    mid = 0.5 * (lo + hi)
    if gamma4(mid) < 1.0:
        lo = mid
    else:
        hi = mid
print(f"\n== 1-ebit threshold F*: solved = {hi:.6f}  (addendum: 0.683910) ==")

print("\n== sigma_eff internal consistency ==")
sig_m = (0.633423 - 0.5) / 50.64
sig_f = (0.730713 - 0.5) / 84.45
print(f"  sigma_eff marrakesh implied by z=+50.64 vs 0.5: {sig_m:.6f}  (reported +/-0.0026)")
print(f"  sigma_eff fez implied by z=+84.45 vs 0.5:      {sig_f:.6f}")
print(f"  marrakesh z vs F* (claimed -19.17): {(0.633423 - 0.683910) / sig_m:+.2f}")
print(f"  fez z vs F* (claimed +17.13):       {(0.730713 - 0.683910) / sig_f:+.2f}")
print(f"  fez 5-sigma LCL (claimed 0.717053): {0.730713 - 5 * sig_f:.6f}")
print(f"  fez 95% t-LCL (claimed 0.723700):   {0.730713 - TCRIT_95_DF5 * sig_f:.6f}")
print(f"  marrakesh 95% t-LCL (claimed 0.626700): {0.633423 - TCRIT_95_DF5 * sig_m:.6f}")
print(f"\nmax |recomputed - table| across all six rows: {maxdiff:.2e} ebits")
