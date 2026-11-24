# Write-Ahead Preregistration — `OSIRIS-ECHO-FEZ` (`Card B` Exploratory Pilot)

- **Protocol ID:** `OSIRIS-ECHO-FEZ` (`Card B` in `phase1/verification_queue.md`, Exploratory Hardware Pilot)
- **Target Backend:** `ibm_fez` (156-qubit IBM Heron r2 processor)
- **Candidate Placements (`K = 10` vertex-disjoint `T5` heavy-hex subgraphs):**
  - `pid=0`: `[21, 22, 23, 16, 24]`, `pid=1`: `[5, 6, 7, 8, 17]`, `pid=2`: `[9, 10, 11, 12, 18]`,
    `pid=3`: `[27, 28, 29, 30, 38]`, `pid=4`: `[31, 32, 33, 34, 39]`, `pid=5`: `[43, 42, 41, 36, 40]`,
    `pid=6`: `[25, 37, 45, 44, 46]`, `pid=7`: `[49, 50, 51, 52, 58]`, `pid=8`: `[63, 62, 61, 60, 76]`,
    `pid=9`: `[67, 66, 65, 64, 77]` (`50` distinct transmons total; zero shared vertices).
- **Offline Stop/Go Prerequisite:** Passed in `experiments/echo_placement/artifacts/echo_stopgo_report.json` (`Delta Spearman = +0.5462`, 95% window-bootstrap CI `[+0.4594, +0.6288]`).

## 1. Probes, Mandatory Controls & Held-Out Workloads (Per Placement `pid in 0..9`)
1. `ECHO_PROBE_16CZ`: Stabilizer-reflected (`S = X_0 X_2 Z_3 Z_4`) barrier-independent `16-CZ` echo probe (`U_global = Z_0 Z_2 != I_32`, ideal return `|00000>` = `1.0`).
2. `IDENTITY_CTRL_0CZ`: `0-CZ` no-workload SPAM return baseline (`|00000>`).
3. `OVERROT_CTRL_016`: Deliberate `RZ(0.16) + RX(0.16)` coherent perturbation positive control.
4. Held-out exact workloads (unseen by echo score):
   - `CLIFFORD_30CZ`: `30-CZ` 5Q random Clifford survival circuit.
   - `MIRROR_24CZ`: `24-CZ` non-Clifford stabilizer-twisted peaked circuit (target `|11001>` = `25`).
   - `BT2T_72CZ` (`x in {0, 2, 8}`): Stratified `72-CZ` structured $2T$ inversion inputs ($Q_8$ center `+1`, $Q_8$ imaginary `+i`, Hurwitz `(+1+i+j+k)/2`).

## 2. Schedule, Equal-Shot Rule & Quota Safety Gate
- `N_BLOCKS = 3` temporally interleaved randomized blocks x `80` PUBs/block (`10` placements x `8` circuits) = **`240` PUBs** (`<= 300`).
- Equal-shot rule: **`256` shots/PUB** (`768` shots per `(placement, circuit)` cell; **`61,440` total shots**; predicted `~18 s` QPU).
- Quota gate: `--submit-frozen-and-harvest` fails closed unless `usage_remaining_seconds >= 25.0`.
- Unit of inference: Single-job run yields `N_windows = 1` exploratory pilot Spearman $\rho(\text{echo}, F_{\text{heldout}}) - \rho(\text{cal}, F_{\text{heldout}})$ across the `K = 10` disjoint placements; confirmatory inference requires repeating across independent calibration windows.
