# `myOLCF` Initial 6-Month Credit Allocation Request — Technical Justification

- **Program:** OLCF Quantum Computing User Program (QCUP)
- **Requested Quantinuum Allocation (`H2-1` / `Helios-1` via Nexus):** **`21,500 HQC`** (`21,169.8 HQC` compiled estimate)
- **Requested IBM Quantum Allocation (`ibm_fez` / `ibm_marrakesh` Heron r2):** **`600 QPU seconds`** (`10.0 RU`)
- **Reproducible Estimator Script & JSON Receipt:** `experiments/qcup_prep/hqc_and_bposd_prep.py` (`qcup_workload_and_bposd_estimates.json`, SHA-256 `cceb8bc04d09145745f40dea4d99d5029a875b9282853fcf472f19e9a2c88995`)

---

## Copy-Paste Technical Justification for `myOLCF`

All circuits have been compiled offline into native trapped-ion operations (`rx`, `ry`, virtual `rz` at 0 HQC cost, two-qubit `rzz`/`ZZPhase`, and mid-circuit `measure` + `reset` with 0 routing SWAPs) and sized using Quantinuum’s standard H-Series/Helios credit formula `HQC = 5 + C * (N_1q + 10*N_2q + 5*N_m) / 5000`, to be cross-verified on the Nexus emulators (`H2-1E` and `Helios-1E`) prior to hardware dispatch:

1. **WP1 — Routing-Free High-Schmidt-Rank Entanglement Witnesses (`4,506.3 HQC` total):**
   - `WP1a` (8 qubits, `k = 2` pairs): 3 randomized blocks x 3 core arms (`Arm A`, `Arm B`, `Arm B-Ctrl`) x 9 Pauli bases = `81` circuits x `300` shots (`24,300` shots), mean `(N_1q = 99.67, N_2q = 33.33, N_m = 12.0)` -> `34.58 HQC/circuit` = **`2,801.0 HQC`**. At `8,100` pooled shots/arm (`SE_shot ~ 0.0055`), trapped-ion 2Q fidelity (`~99.85%`) provides `> 35 sigma` power above the `0.5000` Schmidt-number-2 ceiling and `> 20 sigma` power above the `0.683900` 1-ebit entanglement-of-formation threshold, benchmarked directly against our completed `79s` `ibm_fez` run (`10.5281/zenodo.24287505`).
   - `WP1b` (12 qubits, `k = 3` pairs, `d = 8` Schmidt-rank extension): `54` circuits x `300` shots (`16,200` shots), mean `(N_1q = 53.0, N_2q = 30.0, N_m = 18.0)` -> `31.58 HQC/circuit` = **`1,705.3 HQC`**.
2. **WP2 — Split-Sample-Debiased Kantorovich/Sinkhorn Stability at 50 to 96 Qubits (`6,576.1 HQC` total):**
   - `WP2a` (50 qubits, 4 nonlocal `rzz` layers on H2-1/Helios-1): `12` circuits x `512` shots (`6,144` shots), `(N_1q = 300.0, N_2q = 100.0, N_m = 100.0)` -> `189.32 HQC/circuit` = **`2,271.8 HQC`**.
   - `WP2b` (96 qubits, 4 nonlocal `rzz` layers on 98-qubit Helios-1): `12` circuits x `512` shots (`6,144` shots), `(N_1q = 574.0, N_2q = 192.0, N_m = 192.0)` -> `358.69 HQC/circuit` = **`4,304.3 HQC`**.
3. **WP3 — Multi-Round Mid-Circuit Syndrome Extraction & `BP+OSD` Decoding (`10,087.4 HQC` total):**
   - `WP3a` (`Steane [[7,1,3]]` & `Tesseract [[16,6,4]]`, `R = 3` rounds with QCCD mid-circuit `measure` + `reset`, up to 21 physical ions): `24` circuits x `512` shots (`12,288` shots), mean `(N_1q = 152.5, N_2q = 156.0, N_m = 75.0)` -> `218.76 HQC/circuit` = **`5,250.2 HQC`**.
   - `WP3b` (`Bivariate-Bicycle [[72,12,6]]` weight-6 qLDPC code on 98-qubit Helios-1 using 72 data ions + 24 reused ancilla ions = `96` physical ions, `R = 2` rounds): `8` circuits x `256` shots (`2,048` shots), `(N_1q = 792.0, N_2q = 864.0, N_m = 456.0)` -> `604.654 HQC/circuit` = **`4,837.2 HQC`**. Evaluated offline with our verified Sum-Product `BP+OSD-CS` hypergraph decoder (`7/7`, `16/16`, and `72/72` exhaustive weight-1 plus `2556/2556` weight-2 recovery) against our completed `109s` `ibm_fez` mid-circuit syndrome baseline (`10.5281/zenodo.24293756`, Job `dbtl2vb2kf8s73c9a0d0`).

**Total Initial 6-Month Allocation Request:** **`21,500 HQC`** on Quantinuum (`H2-1` / `Helios-1` via Nexus, covering `21,169.8 HQC` plus `< 2%` calibration/diagnostic margin) and **`600 QPU seconds`** (`10.0 RU`) on IBM Quantum for matched planar superconducting replicates.
