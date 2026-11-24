# OSIRIS CLI v6.2.2 Release Notes (2026-11-24)

## Summary

`v6.2.2` extends `v6.2.1` (`f84b8dfe`) and Phase 1 follow-ons (`5549d0c9`) with:
1. **Exact 5-Qutrit (`3^5 = 243`-State) Transmon `|2>` Leakage & Leakage-Reduction Unit (`LRU`) Ablation (`experiments/qec_card_c/qec_leakage_lru.py`)**, closing the documented non-computational `|2>` leakage gap in `Card C: OSIRIS-QEC`.
2. **7th & 8th Macro-Locked REVTeX 4.2 + Standalone `%PDF-1.4` Manuscript Packages** (`papers/bt2t002_mul_and_nonabelian_qft/` and `papers/echo_and_card_c_qec_gates/`), bringing the Nov 23–24 preprint Staging Manifest (`papers/zenodo_staging_manifest_nov23.json`) to **7 zero-prose-float manuscripts** (8 total with `qdecode_heron_calibration_horizon`).
3. **Write-Ahead Hardware `.qpy` & SHA-256 Staging (`0s` QPU)** for `OSIRIS-ECHO-FEZ` (`240` PUBs across `10` disjoint `T5` trees, `61,440` shots) and `OSIRIS-QEC-REP3-FEZ` (`96` dynamic `measure + reset + if_test` `[[3,1,3]]` PUBs on `[58, 71, 70, 69, 78]`, `49,152` shots), validated against live `ibm_fez.target` and fail-closed below `25s` remaining QPU quota.
4. **Clean-Environment GitHub Actions CI Matrix Hardening** across Python `3.9`–`3.12` (when optional `qiskit`, `qiskit_ibm_runtime`, `pytket`, `ldpc`, or `rfc8785` SDKs are absent) and **28 chain-verified RFC 8785 JCS reproducibility capsules**.

---

## Highlights

### 1. Card C Extension: Exact 5-Qutrit `|2>` Leakage & `LRU` Crossover (`OSIRIS-QEC-LEAKAGE-LRU`, `0s` QPU)
- Exact `243 x 243` density-matrix verification vs `125`-dimensional (`5^3`) real-Hermitian Liouville instrument (`<= 2.08e-16` discrepancy across `NO_LRU`, `ANCILLA_LRU_ONLY`, and `FULL_DATA_AND_ANCILLA_LRU`), CPTP completeness `<= 8.88e-16`, and exact `0.0` deviation vs 2-level `card_c_qec_crossing_report.json` at `p_leak = 0`.
- On `FEZ_PUBLISHED` (`[[3,1,3]]`), unmitigated `p_leak,CZ = 1.2e-3` inflates the per-round logical error slope **`2.74x`** (`0.000691 -> 0.001896`) via multi-round ancilla `|2>` syndrome streaks (`1.86%` at `R = 9`), whereas `FULL_DATA_AND_ANCILLA_LRU` (`DQLR`) pins data `|2>` occupancy at `3.9e-7` and recovers **`78.1%`** of the leakage penalty (`0.000955`), exhibiting an exact pulse-cost crossover between `2.0e-4` and `5.0e-4` per-`CZ` leakage.

### 2. Manuscripts 7 & 8 (`papers/bt2t002_mul_and_nonabelian_qft/`, `papers/echo_and_card_c_qec_gates/`)
- **Paper 7 (`bt2t002_mul_and_nonabelian_qft.pdf`, `5` pp, `171` `\\btm*` macros, `17/17` `SHA256SUMS.txt` OK):** 10Q zero-ancilla `2T` multiplication (`19.00x` all-to-all and `21.31x` heavy-hex `CZ` compression) and 5Q 7-irrep Peter–Weyl non-abelian QFT (`9.80e-16` unitarity, `3.11e-16` block-diagonalization).
- **Paper 8 (`echo_and_card_c_qec_gates.pdf`, `6` pp, `140+` `\\ecq*` macros, `15/15` `SHA256SUMS.txt` OK):** Unified `OSIRIS-ECHO` (`+0.9333` vs `+0.3871` Spearman), `Card C` 4-arm space-time MWPM crossing (`[[3,1,3]]` & `[[9,1,3]]`), and 5-qutrit `|2>` `LRU` synthesis (`SIMULATED`, `provider_job_id: null`).

### 3. Write-Ahead `ibm_fez` Staging (`experiments/echo_fez_staged/`, `experiments/rep3_fez_staged/`)
- **`OSIRIS-ECHO-FEZ` (`qpy_sha256 = 538af9be...`):** `240` ISA-validated PUBs (`61,440` shots) across all `10` vertex-disjoint `T5` trees (`50` transmons).
- **`OSIRIS-QEC-REP3-FEZ` (`qpy_sha256 = f240d445...`):** `96` ISA-validated dynamic MCMR + `if_test` PUBs (`49,152` shots) on `[58, 71, 70, 69, 78]` (`ro_mean = 0.0083`, matched `904 dt = 3.616 us` round schedule).
