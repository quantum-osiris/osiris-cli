# Write-Ahead Preregistration — `OSIRIS-QEC-REP3-FEZ` (`Card C` `[[3,1,3]]` Crossing)

- **Protocol ID:** `OSIRIS-QEC-REP3-FEZ` (`Card C` in `phase1/verification_queue.md`, Distance-3 Repetition Code)
- **Target Backend:** `ibm_fez` (156-qubit IBM Heron r2 processor)
- **Pinned 5-Transmon Linear Chain (`[d0, a0, d1, a1, d2]`):** `FEZ_REP3_CHAIN = [58, 71, 70, 69, 78]` (`ro_mean = 0.0083`, `cz_sum = 0.01381`, valid `T1 > 80 us`, `T2 > 50 us`).
- **Offline Stop/Go Prerequisite:** Passed in `experiments/qec_card_c/artifacts/card_c_qec_crossing_report.json` (`PASS_CROSSING_OBSERVED` at `R = 1` on `FEZ_PUBLISHED` and `R = 1/2` on `FEZ_DRIFTED_SPAM`).

## 1. Arms (`R in {1, 3, 5}`, Logical Inputs `|0_L>, |1_L>`)
Matched round duration `ROUND_DT = 904 dt` (`3.616 us`, matching 2 sequential native `CX` layers + `1.660 us` `measure` + `1.684 us` `reset` on `ibm_fez`):
1. `ARM1_UNENCODED`: Bare data qubit `d1 = 70` idling for `R * 904 dt` + 1 readout.
2. `ARM2_ENCODED_NO_CORR`: `[[3,1,3]]` (`[58, 70, 78]`) idling for `R * 904 dt` + 3-bit terminal readout (majority decode).
3. `ARM3_4A_SYNDROME_STREAM`: `R` rounds of mid-circuit `Z_0 Z_1` (`a0=71`) and `Z_1 Z_2` (`a1=69`) syndrome extraction (`measure` + `reset`) + 3-bit terminal data readout; scored separately as:
   - **`Arm 3` (Postselected Detection):** Conditional logical failure on zero-detector runs + unconditional acceptance rate `A(R)`.
   - **`Arm 4a` (Auditable Space-Time MWPM Pauli-Frame Tracking):** Unconditional logical failure decoded offline via `AuditableMWPMDecoder`.
4. `ARM4B_PHYSICAL_FEEDFORWARD`: `R` rounds of mid-circuit `measure` + `reset` + real-time hardware `if_test((syn_r, 1/3/2))` conditional `X` correction pulses on `d0=58, d1=70, d2=78` (reported separately from `Arm 4a`!).
5. `RO_CAL` (`8` 3-bit basis states on `[58, 70, 78]`): Terminal data readout calibration control.

## 2. Schedule, Equal-Shot Rule & Quota Safety Gate
- `N_BLOCKS = 3` randomized blocks x `32` PUBs/block (`3 rounds * 2 states * 4 arms + 8 RO_CAL`) = **`96` PUBs** (`<= 300`).
- Equal-shot rule: **`512` shots/PUB** (`1,536` shots per `(round, state, arm)` cell; **`49,152` total shots**; predicted `~17 s` QPU).
- Quota gate: `--submit-frozen-and-harvest` fails closed unless `usage_remaining_seconds >= 25.0`.
