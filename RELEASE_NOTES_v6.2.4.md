# OSIRIS CLI v6.2.4 Release Notes (2026-11-25)

## Highlights

1. **Live `OSIRIS-BT2T-003-FEZ` Non-Abelian Magnetic Plaquette Trajectory & Two-Window `OSIRIS-ECHO` Drift Catch (`experiments/bt2t003_fez/`, job `dc38nreq4uuc73a6g0k0`, `HARDWARE_ATTESTED`, `19s` QPU, `240` PUBs x `256` = `61,440` shots across `10` disjoint `T5` subgraphs = `50` transmons):**
   - **Write-Ahead Governance:** Staged and pushed to `origin/main` at commit `82cfd46a8ef091fbb6919f5112578d8e657eef7a` (`qpy_sha256 = 349a7ab40a1a55db6ca85dbfd27c24f75aa3e4c67c20accad82a7a41b2c87b9a`) prior to hardware execution (`raw_counts_gz_sha256 = 4667adc71b3af9b7dd3c9dff1fde9d4d6acaf768597992bfbc39bbde2c0abeb3`).
   - **Magnetic Plaquette Character-Frequency Trajectories (`40-CZ` total / `19-CZ` $U_B$ core vs `2,727 CZ` generic dense QFT-conjugated synthesis = `68.18x` native `CZ` reduction):**
     - On `#1` Echo-selected `pid = 0` (`[21, 22, 23, 16, 24]`): Center sector ($\Delta\chi_{2D}=4$, $\cos^2 2\tau$) measured $P_{|+\rangle}(\tau) = [0.876302, 0.411458, 0.057292, 0.841146]$ (**Pearson $r = 0.997576$**), with $P_{|-\rangle}(\pi/4) = \mathbf{0.850260}$ vs `0.041667` generic dense (**`+80.86 pp` / `20.41x` structured compiler advantage**); Hurwitz half-frequency sector ($\Delta\chi_{2D}=2$, $\cos^2 \tau$) measured $P_{\Phi^+}(\tau) = [0.829427, 0.669271, 0.404948, 0.130208, 0.050781]$ (**Pearson $r = 0.997940$**, $P_{\Phi^-}(\pi/2) = \mathbf{0.796875}$); $\Delta\chi_{2D}=0$ null controls preserved `0.861979` (imaginary $Q_8$) and `0.785156` (unused-8 subspace).
   - **Live Two-Window (`N_windows = 2`, `+3.763 h`) `OSIRIS-ECHO` Intra-Calibration Drift Catch (`dc35e35s2omc738cvcug` -> `dc38nreq4uuc73a6g0k0`):**
     - Hub qubit `Q7` of formerly top-ranked `pid = 1` (`[5, 6, 7, 8, 17]`) suffered an unflagged `+31.12%` $|0\rangle \to |1\rangle$ SPAM/readout drift excursion (`IDENTITY_CTRL_0CZ` dropped `0.962240 -> 0.677083`, `ECHO_PROBE_16CZ` dropped `0.815104 -> 0.562500`).
     - Frozen published calibration collapsed to **$\rho_{\text{cal}} = +0.454545$** with **`0.361002` top-1 selection regret**, whereas live `ECHO_PROBE_16CZ` switched top-1 selection to `pid = 0` with **$\rho_{\text{echo}} = +0.948333$ ($\Delta\rho = +0.493787$)** and **`0.000000` top-1 selection regret across both windows** (2-window mean $\bar{\rho}_{\text{echo}} = +0.834772$ vs $\bar{\rho}_{\text{cal}} = +0.618182$, $\Delta\bar{\rho} = +0.216591$).
2. **Write-Ahead 9-Qubit `[[5,1,5]]` vs 5-Qubit `[[3,1,3]]` Dynamic MCMR Distance-Scaling Staging (`experiments/rep5_fez_staged/`, `OSIRIS-QEC-REP5-FEZ`, `0s` QPU):**
   - Locked `72` ISA-validated dynamic `measure + reset + if_test` PUBs (`36,864` shots on `[120, 121, 122, 123, 136, 143, 142, 141, 140]`, `qpy_sha256 = da8061639e8eaa7e766b026a2cf2361db40886d8aa2282a9d5594c76728cb518`) with fail-closed `25.0s` quota protection.
3. **9th & 10th Macro-Locked REVTeX 4.2 + Standalone `%PDF-1.4` Manuscript Packages & Zenodo Archival DOIs (`0` duplicate DOIs):**
   - **Paper 9 (`papers/rep3_echo_fez_and_multidistance_qec/`):** `27/27` locked files in `SHA256SUMS.txt`, published to Zenodo at **`10.5281/zenodo.24524910`** (`deposition_id = 24524910`).
   - **Paper 10 (`papers/bt2t003_kogut_susskind_plaquette/`):** `20/20` locked files in `SHA256SUMS.txt`, published to Zenodo at **`10.5281/zenodo.24525501`** (`deposition_id = 24525501`).
   - All `10` active manuscript packages (`166/166` locked `SHA256SUMS.txt` entries `OK`) and `9` standalone `%PDF-1.4` preprints (`tests/test_standalone_paper_pdfs.py`, `19/19` tests) verified.
4. **Provenance Ledger & Site Synchronization:**
   - `32` chain-verified RFC 8785 JCS reproducibility capsules (`19` `HARDWARE_ATTESTED` + `13` `OFFLINE_COMPILED`/`SIMULATED`, `file_sha256 = b7ec96e7dac0e7dc...`, `tail_sha256 = edd70b90970e082f...`).
   - `49` audited claims in `phase1/claim_evidence_map.jsonl` (`BT2T-7`, `QEC-CARDC-5`).
   - `76` Zenodo deposits in `quantumosiris-site/public/data/records.json` and `87` indexed DOIs in `~/.osiris/evidence/doi_lineage.json` (`lineage_sha256 = 33f8b67a9d4071bd...`).
