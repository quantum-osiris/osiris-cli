# OSIRIS CLI v6.2.3 Release Notes (2026-11-25)

## Summary

`v6.2.3` extends `v6.2.2` (`9f9a0f3d`) with:
1. **Two Write-Ahead Preregistered `HARDWARE_ATTESTED` IBM Heron r2 (`ibm_fez`) Runs (`110,592` Shots, `35s` QPU):**
   - **`OSIRIS-QEC-REP3-FEZ` (`job_id = dc35d4inj2vs73dhmsog`, `16s` QPU, `49,152` shots on `[58, 71, 70, 69, 78]`):** Both auditable space-time MWPM Pauli-frame tracking (`Arm 4a`) and real-time hardware `if_test` physical feed-forward (`Arm 4b`) cross cleanly below unencoded (`Arm 1`) and no-correction (`Arm 2`) baselines at `R = 3` and `R = 5` across `3/3` interleaved blocks (`R = 5`: `Arm 4a = 0.0872`, `Arm 4b = 0.0980` vs `Arm 2 = 0.1230`, `Arm 1 = 0.2770`; `Arm 4b < Arm 2` paired-$t = 9.01, p = 0.00604$).
   - **`OSIRIS-ECHO-FEZ` (`job_id = dc35e35s2omc738cvcug`, `19s` QPU, `61,440` shots across `10` vertex-disjoint `T5` subgraphs = `50` transmons):** `ECHO_PROBE_16CZ` selects the `#1` oracle-best placement (`pid = 1`, `[5, 6, 7, 8, 17]`, `0.0` top-1 regret on `0.6594` composite) and beats calibration on `MIRROR_24CZ` (`+0.9030` vs `+0.7576` Spearman), while single-window unweighted composite sits `-0.0606` below fresh calibration (`+0.7212` vs `+0.7818`, `H2 = REFUTED`; post-hoc SPAM-normalized `ECHO / ID_0CZ` achieves `+0.9030` composite and `+0.9515` on `BT2T_72CZ`).
2. **Three New Published Zenodo Archival DOIs (`74` Site Deposits, `85` Indexed Lineage DOIs):**
   - `10.5281/zenodo.24520723` (`bt2t001_binary_tetrahedral_fez`)
   - `10.5281/zenodo.24520738` (`bt2t002_mul_and_nonabelian_qft`)
   - `10.5281/zenodo.24520746` (`echo_and_card_c_qec_gates`)
3. **Multi-Distance QEC Threshold & Exponential Suppression Gate (`OSIRIS-QEC-MULTIDISTANCE`, `0s` QPU):**
   - Exact polynomial-time Edmonds Blossom MWPM (`0 / 3,600` mismatches vs exact bitmask-DP on `d = 3`) across hook-safe `d = 3, 5, 7` repetition (`5Q, 9Q, 13Q`) and rotated surface (`[[9,1,3]]` `17Q`, `[[25,1,5]]` `49Q`, `[[49,1,7]]` `97Q`) codes (`p_th^surf = 1.124%`; `H2-1` surface `Lambda_3->5 = 6.96, Lambda_5->7 = 13.78`; `FEZ_DRIFTED_SPAM` SPAM-debiased `Lambda_3->5^diff = 1.023 [0.74, 1.33]`).
4. **Kogut–Susskind `2T` Non-Abelian Lattice Gauge Plaquette Trotter Dynamics (`OSIRIS-BT2T-003`, `0s` QPU):**
   - 5-qubit gauge-fixed `2T` Strang splitting (`slope = 2.000008`, `16-CZ` magnetic step `U_B`, `209 CZ` all-to-all / `350 CZ` heavy-hex `T5` Strang step) matches unfixed 20Q/25Q 4-link Gauss-law-invariant evolution to `2.27e-15` while compressing `CZ` count `2.02x` vs dense 5Q unitary and `104x-178x` vs unfixed 4-link circuits.
5. **30 Chain-Verified RFC 8785 JCS Reproducibility Capsules (`18` `HARDWARE_ATTESTED` + `12` `SIMULATED`/`OFFLINE_COMPILED`) & `47` Reconciled Claims.**
