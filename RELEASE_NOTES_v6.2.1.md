# OSIRIS CLI v6.2.1 Release Notes (2026-11-24)

## Summary

`v6.2.1` extends `v6.2.0` (`e680e20c`) with the live IBM Heron r2 Binary Tetrahedral Group (`|2T| = 24`) exact-inversion synthesis benchmark (`OSIRIS-BT2T-001-FEZ`), its macro-locked REVTeX 4.2 + standalone `%PDF-1.4` manuscript package, the pure-Python deterministic preprint typesetter (`papers/_pdf_preprint_renderer.py`), offline `Card B` (`OSIRIS-ECHO` placement-ranking stop/go gate), `Check #6` (`PHY-9` A-field simulation relabeling and 4,000-trial injection-recovery study), `ENG-NEXUS-4` Quantinuum Nexus onboarding preflight tooling, and full `v6.2.0` Zenodo DOI lineage & RFC 8785 provenance synchronization.

---

## Highlights

### 1. Live `OSIRIS-BT2T-001-FEZ` Hardware Execution & Gate `G1` Sign-Off (`experiments/bt2t_fez/`, `papers/bt2t001_binary_tetrahedral_fez/`)
- **Independent Gate `G1` Sign-Off (`experiments/bt2t_inversion/G1_INDEPENDENT_REVIEW.md`):** Clean-room $SU(2)$ matrix & Boolean coset-parity verification of frozen encoding `bt2t-enc-1` (`sha256 = 3e3b02d1...`).
- **Live 156Q IBM Heron r2 Run (`ibm_fez`, `job_id = dc2ee3uq4uuc73a56o6g`, `11s` QPU, `288` PUBs x `96` = `27,648` equal shots on `[21, 22, 23, 16, 24]`, write-ahead preregistration `240046bd`):**
  - **`H1` (`SUPPORTED`):** Structured `72-CZ` synthesis achieved `0.5337 +/- 0.0059` raw (`0.6104` readout-mitigated, **`24/24` modal exact group-inverse matches**) vs `0.0661 +/- 0.0030` for generic `723-CZ` transposition synthesis (**`+46.76 pp`**, paired block-$t = 73.23$, one-sided $p = 9.32\times 10^{-5}$; `24/24` elements won, exact two-sided Wilcoxon $p = 1.192\times 10^{-7}$).
  - **`H2` (`SUPPORTED`):** Unused-8 codeword preservation `0.5595` vs `0.0777` (`+48.18 pp`, `8/8` modal) and valid-to-unused subspace leakage `0.0940 < 0.15`.
  - **`H3` (`REFUTED` — Live Calibration Staleness Caught):** Interleaved `PREP_RO_CTRL` (`0 CZ`) measured `0.9076` unused-8 and `0.8643 < 0.90` valid-24 (`24/24` modal, $\kappa(M_{32}) = 1.4524$) due to live `14.91x` post-calibration readout drift on transmon `q24` (`0.33% -> 4.92%`).
- **6th Macro-Locked REVTeX 4.2 + Standalone `%PDF-1.4` Package:** `papers/bt2t001_binary_tetrahedral_fez/` (`218` programmatic macros, 2 dual `.pdf`/`.svg` vector figures, 5-page `bt2t001_binary_tetrahedral_fez.pdf`, `22/22` `SHA256SUMS.txt` entries verified).

### 2. Standalone Deterministic `%PDF-1.4` Typesetter, Zenodo Staging & `ENG-NEXUS-4` Onboarding CLI
- **`papers/_pdf_preprint_renderer.py`:** Zero-dependency deterministic multi-page `%PDF-1.4` typesetter generating standalone preprints across all 5 November 23–24 packages (`xtop006_non_clifford_kraus_fez`, `wp2_27q_debiased_ot_fez`, `nonabelian_2bga_and_adv1`, `heron_r2_calibration_staleness_and_colored_nulls`, and `bt2t001_binary_tetrahedral_fez`).
- **`papers/publish_nov23_papers_to_zenodo.py`:** Default `--dry-run` `SYS-024` + `SHA256SUMS.txt` staging bundler (`papers/zenodo_staging_manifest_nov23.json`) with fail-closed `--execute` requiring `ZENODO_TOKEN` + `--confirm-publish`.
- **`scripts/nexus_onboard_preflight.py` (`ENG-NEXUS-4`):** Offline single-canary HQC reconciliation (`36.9000` HQC on `xtop005q_Q0_L3_ZZ`) and fail-closed `--preflight-live` hard-blocking paid Tier 3 targets (`H2-1`, `H2-2`, `Helios`).

### 3. Offline `Card B` (`OSIRIS-ECHO`) Stop/Go Gate & `Check #6` (`PHY-9`) Injection-Recovery (`0s` QPU)
- **`experiments/echo_placement/` (`Card B`):** Stabilizer-reflected barrier-independent non-cancellable 16-`CZ` echo probe across `K = 10` disjoint `T5` subgraphs on `ibm_fez` over `M = 16` independent calibration windows: `ECHO_RANKED` Spearman $\bar{\rho} = +0.9333$ vs stale `CALIBRATION_RANKED` $\bar{\rho} = +0.3871$ ($\Delta\rho = +0.5462$, 95% window-bootstrap CI `[+0.4594, +0.6288]`), passing the offline stop/go gate across unseen `BT2T`, Clifford, and non-Clifford mirror circuits.
- **`research/a_field_study/experiments/falsification_harness.py` (`PHY-9`):** Replaced zero-signal simulation `"REFUTED"` mislabeling with `BELOW_THRESHOLD_WITH_ZERO_INJECTED_SIGNAL` and `physical_claim_assessment: UNVERIFIED`, adding a 4,000-trial Monte Carlo injection-recovery sensitivity curve (`0.35%` / `0.25%` FPR at $0\sigma$, `49.7%` power at $3\sigma$, `99.83%` / `99.90%` power at $6\sigma$).

### 4. `v6.2.0` Zenodo DOI Lineage Synchronization & Python 3.14 Hygiene
- **`osiris_cli/orchestration_departments.py`:** `index_doi_lineage()` now discovers `papers/**/zenodo_receipt.json` (`82` indexed Zenodo DOIs including `10.5281/zenodo.24486928`, `.24486931`, `.24486938`, `.24486942`, `.24486953`, `.24486962`, and `24287505 -> 24486953` supersession); `materialize_workspace_reproducibility_capsules()` materializes `23` chain-verified RFC 8785 JCS capsules.
- **`osiris-benchmarks/science_bench/run_science_bench.py`:** Fixed raw f-string LaTeX escape sequence (`0` Python 3.14 `SyntaxWarning`s).
