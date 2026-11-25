# Write-Ahead Preregistration — `OSIRIS-QEC-REP5-FEZ` (`[[5,1,5]]` vs `[[3,1,3]]` Distance-Scaling $\Lambda_{3\to 5}$ on `ibm_fez`)

- **Protocol ID:** `OSIRIS-QEC-REP5-FEZ`
- **Target Backend:** `ibm_fez` (156-qubit IBM Heron r2 processor)
- **Pinned 9-Transmon Linear Chain (`[d0, a0, d1, a1, d2, a2, d3, a3, d4]`):**
  `FEZ_REP5_CHAIN = [120, 121, 122, 123, 136, 143, 142, 141, 140]` (`ro_mean = 0.01000`, `cz_sum = 0.01832`, all `T1 >= 60 us`, `T2 >= 40 us`), with co-located `[[3,1,3]]` embedded on the central 5 transmons `[122, 123, 136, 143, 142]`.
- **Offline Stop/Go Prerequisite:** Passed in `experiments/qec_card_c/artifacts/card_c_multidistance_threshold_report.json` (`FEZ_PUBLISHED` `Lambda_{3->5} = 10.565` `[7.86, 14.11]`).

## 1. Co-Scheduled Distance-Scaling Arms (`R in {1, 3, 5}`, Logical States `|0_L>, |1_L>`)
Matched round duration `ROUND_DT = 904 dt` (`3.616 us/round`):
1. `ARM1_UNENCODED`: Central data qubit `d2 = 136` idling for `R * 904 dt`.
2. `ARM2_REP3_NOCORR`: Co-located 5-qubit `[[3,1,3]]` (`[122, 136, 142]`) idling for `R * 904 dt` + 3-bit terminal majority readout.
3. `ARM2_REP5_NOCORR`: Full 9-qubit `[[5,1,5]]` (`[120, 122, 136, 142, 140]`) idling for `R * 904 dt` + 5-bit terminal majority readout.
4. `ARM4A_REP3_MWPM`: `R` rounds of mid-circuit `Z` syndrome extraction (`measure + reset` on `a1=123, a2=143`) decoded via `PolynomialMWPMDecoder(d=3)`.
5. `ARM4A_REP5_MWPM`: `R` rounds of mid-circuit `Z` syndrome extraction (`measure + reset` on `a0=121, a1=123, a2=143, a3=141`) decoded via `PolynomialMWPMDecoder(d=5)`.
6. `RO_CAL_5D`: `6` stratified 5-data-qubit basis states (`00000, 11111, 01010, 10101, 00100, 11011`).

## 2. Schedule & Quota Safety Gate
- `N_BLOCKS = 2` randomized blocks x `36` PUBs/block (`3 rounds * 2 states * 5 arms + 6 RO_CAL_5D`) = **`72` PUBs** x `512` shots/PUB = **`36,864` total shots**.
- Quota gate: `--submit-frozen-and-harvest` fails closed unless `usage_remaining_seconds >= 25.0`.
