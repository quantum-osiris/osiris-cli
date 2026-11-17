# Forensic Archive Audit: ~46 µs Revival Records (`TAU_REVIVAL_01_ARCHIVE_AUDIT.md`)

**Experiment ID:** `TAU-REVIVAL-01`  
**Status:** `Implemented and tested` (`phase2/tau_revival_01/tau_revival_models.py`, `docs/phase2/tau_revival_01_audit_and_sim.json`)

---

## 1. Exhaustive Audit of Every Archived ~46 µs Record Across `/home/osiris` & Legacy Mirrors

Direct byte-level read-only inspection across `/home/osiris` and all 171 bare git mirrors identified **six distinct archival strata**, revealing severe provenance, observable, and sampling defects in every positive claim and confirming prospective hardware refutation:

| Archive / File & SHA-256 | Backend & Date | Delay Grid & Qubits | Forensic Finding |
|---|---|---|---|
| **1. Wall-Clock Timestamp Corpus (`TAU_PHASE`)**<br>`tau_phase_anomaly_package.zip` (`5801823d...`), `dnalang-quantum-corpus-v1.0.zip` (`32a36250...`) | `ibm_fez` & `ibm_torino` (Nov 2025; `580` jobs collapsing to `103` unique runs) | **Zero circuit idle delay sweep.** Computed `(unix_timestamp mod 46 µs) / 46 µs`. | **Server-Clock Artifact (`ARTIFACT`).** Self-refuted inside the same archive: `tau_alignment_statistics.json` (`6393ece4...`) gives $t = 1.233, p = 0.2194$ (null), and `K8_CAUSALITY_BREAKTHROUGH.md` (`172794f9...`) records `Status: K8 = 0` ($A/\sigma_A = 1.34$). |
| **2. `tau_analysis_d5h5n9agim5s73ahi1h0.json`**<br>`electrogravitic-unified-physics.git` (`a9da57c7...`) | `ibm_fez` (`32` circuits) | JSON lists `"tau_0_predicted": 46`, **`"tau_0_observed": 8`**, `"peak_circuits": [0, 8, 16, 24]`, `"peak_fidelity_mean": 0.9985`, `"trough_fidelity_mean": 0.4934`. | **Gate-Angle Sweep Mislabelled as Delay Periodicity.** Peaks of `0.9985` at indices `0, 8, 16, 24` and troughs of `0.4934` are a deterministic gate-rotation $\cos^2(k\pi/8)$ curve (period `8` circuits, NOT `46 µs` decoherence!). Raw counts absent. |
| **3. `tau_sweep_d5ig8acpe0pc73anfb9g.json` & `tau_sweep_d5ikodnea9qs7393cb7g.json`**<br>`electrogravitic-unified-physics.git` (`d0ce134c...`, `0e746350...`) | `ibm_fez` (`2026-01-12`) | Coarse 13-point grid `[0, 10, 20, 30, 40, 46, 50, 60, 70, 80, 90, 92, 100] µs` (`8,192` shots planned). | **Unpopulated Pre-Submission Manifests Only.** Contain zero measured counts, fidelities, or calibration metadata. Furthermore, `10 µs` coarse spacing ($f_{\text{Nyq}} = 50\text{ kHz}$) aliases any $100 m \pm 21.7\text{ kHz}$ detuning onto `~46 µs`. |
| **4. `sovereign_tau_sweep_d0a2afa967a7282dbecc7a7664cb0026.json` & `braket_tau_sweep.py`**<br>`sovereign-qed-physics.git` (`d5844619...`, `8cf998bf...`) | `SOVEREIGN_LOCAL_SIMULATOR` (`2026-01-30`, seed `51843`) | 21 points `[1..100] µs`. | **Synthetic Injected Bump.** Simulator explicitly adds an `ANLPCC` Gaussian revival around $\varphi^8 = 46.9787\,\mu\text{s}$ in Python. `braket_tau_sweep.py` was never executed on hardware and substitutes `int(tau_us / 10)` identity-gate pairs for physical time delays. |
| **5. Feb 2026 Hardware Self-Test (`ZZ_UNIVERSAL`) & Monte Carlo Audit (`EXP-TAU-ERRATUM-001`)**<br>`experiment_results.json` (`06bb51af...`) | `ibm_fez` / `ibm_torino` (`10.5281/zenodo.18781261`) + OU Monte Carlo (`2026-04-08`) | Multi-pair ZZ/Ramsey sweeps + `500` OU drift trials × `126` frequencies. | **Pair-Specific & Selection-Biased.** Measured beat periods span **`6.7–333 µs` (`CV = 123%`)**, refuting universality. Naive unadjusted periodogram peak-picking produced **`87.6%` false positives** under pure OU drift (`0.0%` after BH-FDR + block permutation). |
| **6. Prospective 3-Basis Tomography (`dbiobimne4hc73bf2am0`)**<br>`osiris_cli/claims.py` (`K8_REVIVAL`), `10.5281/zenodo.23832283` | `ibm_fez` (`2026-10-31`, `59` PUBs, `60,416` shots, `Q[0,1]`) | Controlled idle delay at $T = 46\,\mu\text{s}$ with full `XX, YY, ZZ` Pauli tomography + 8 angle controls. | **Definitive Hardware Refutation (`REFUTED`).** $\mathcal{F}_{\text{Bell}}(46\,\mu\text{s}) = \mathbf{0.3438 \pm 0.0132} < 0.50$ (separable) and transverse coherence $C_\perp = \mathbf{+0.0029 \pm 0.0221}$ (`0.13σ` from zero); earlier $P_{00}+P_{11}$ increases were $T_1$ ground-state relaxation (`R² = 0.971–0.976`), not coherent entanglement revival. |

---

## 2. Archive Resolution Limitations & Scientific Conclusion

1. **Resolution & Metadata Gaps:** None of the early Jan 2026 `tau_sweep_*.json` files preserved raw per-shot counts, `XX/YY` parity bases, or contemporaneous per-qubit Ramsey detuning calibrations ($\Delta f_0$).
2. **Disposition:** All archived evidence is completely explained by **(a) server-clock timestamp binning (`TAU_PHASE`)**, **(b) $T_1$ relaxation toward $|00\rangle$ misread via Z-only population $P_{00}+P_{11}$ (`K8_REVIVAL`)**, and **(c) ordinary pair-dependent $ZZ$/detuning Ramsey fringes (`6.7–333 µs`, where $46\,\mu\text{s} \leftrightarrow 21.7\text{ kHz}$)**. No unresolved physical anomaly remains in the archive.
