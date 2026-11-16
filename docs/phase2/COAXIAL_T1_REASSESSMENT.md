# Coaxial $T_1$ Reassessment Without Retrofitting (`COAXIAL_T1_REASSESSMENT.md`)

**Phase 2 Section:** Section 8 (`REASSESS COAXIAL T1 WITHOUT RETROFITTING`)  
**Status:** `Implemented and tested` (`phase2/coaxial_t1/coaxial_t1_reassessment.py`, `tests/test_coaxial_t1_reassessment.py`, `docs/phase2/coaxial_t1_reassessment_results.json`)  
**Mechanism Classification:** **`INCONCLUSIVE`** (Collective-mode periodicity cannot be distinguished from smooth dielectric/Purcell backgrounds across existing processor frequency bands; negative $T_1$-vs-frequency trend is preserved as a robust empirical observation).

---

## 1. Source, Provenance, and Age of Calibration Measurements

Inspection of both `/home/osiris/coaxial_t1_pilot_results.json` (`sha256: ca282686...`, 21 V2 snapshots) and direct programmatic extraction across all Qiskit `FakeProvider` backends (`extract_v2_calibration_snapshots()`) established:
- **28 testable snapshots** (`20Q–127Q`, including all **11** `127`-qubit Eagle r3 processors: `FakeKyoto` & `FakeOsaka` [`2024-02-28`], `FakeBrisbane`, `FakeCusco`, `FakeKawasaki`, `FakeKyiv`, `FakeQuebec`, `FakeSherbrooke` [`2025-02-26`], `FakeBrussels` & `FakeStrasbourg` [`2026-04-17`], and `FakeWashingtonV2` [`2022-04-12`]) publish both per-qubit transition frequency $f_i \in [4.34, 5.58]\text{ GHz}$ and $T_{1,i}$.
- **Heron r2 Frequency Gap:** Bundled `156`-qubit Heron snapshots (`FakeFez`, `FakeMarrakesh`, `FakeKingston`) and `133`-qubit `FakeTorino` do **not** populate per-qubit frequencies in `Target.qubit_properties`, blocking cross-qubit $T_1(f)$ evaluation without live calibration retrieval.
- **Historical April 2026 Script Defect Preserved on Record:** `/home/osiris/coaxial_t1_experiment.py` (`sha256: 025d3636...`) fell back to `simulate_calibration_data(seed=42)` while labeling output `ibm_fez`; no synthetic data are used here.

---

## 2. Preservation of Observed Negative $T_1(f)$ Relationship

Across **27 of 28** testable snapshots (and **11 of 11** `127`-qubit Eagle processors), $T_1$ decreases with qubit frequency:
- **Fixed-effect inverse-variance pooled slope (`28` snapshots):** **$\hat{\beta}_{1,\text{FE}} = -56.26 \pm 6.07\,\mu\text{s/GHz}$ ($z = -9.26$, $p < 10^{-19}$)**.
- **DerSimonian–Laird random-effects pooled slope (`28` snapshots, accounting for between-chip heterogeneity $I^2 = 86.3\%$ and spatial autocorrelation):** **$\hat{\beta}_{1,\text{RE}} = -114.38 \pm 18.04\,\mu\text{s/GHz}$ ($z = -6.34$, $p < 10^{-9}$)**.
- On modern high-coherence `127`-qubit Eagle processors (`mean T1 = 194–304 µs`), individual slopes range from **`-131` to `-295 µs/GHz`** (standard surface dielectric loss $1/T_1 \propto \omega \, \tan\delta_{\text{TLS}}$ and Purcell filter roll-off).

---

## 3. Physical Electrical Lengths, Spectral Coverage & Severe Collinearity

Without retrofitting $c_{\text{eff}} = 1.1\times 10^8\text{ m/s}$ or hop spacing $1.2\text{ mm}$:

| Candidate Physical Structure | Length $L_{\text{eff}}$ | Standing-Wave Period $\Delta f = \frac{v_p}{2 L_{\text{eff}}}$ | Cycles in `0.58 GHz` Chip Band | Identifiable Beyond Smooth Background? |
|---|---:|---:|---:|---|
| 13-hop 27Q subgraph (historical April script) | `15.6 mm` | **`3.5256 GHz`** | **`0.165` cycles** | **No** (`quadratic VIF = 200–15,790`) |
| 26-hop 127Q Eagle graph diameter (`--hops graph`) | `31.2 mm` | **`1.7628 GHz`** | **`0.329` cycles** | **No** (`0/11` trend-adjusted permutation hits) |
| On-chip $\lambda/2$ readout / Purcell CPW (`~8 mm`) | `8.0 mm` | `6.875 GHz` | `0.084` cycles | **No** |
| Package-to-MXC cryogenic coaxial cable (`~0.50 m` PTFE) | `500 mm` | `0.200 GHz` | `2.90` cycles | *Identifiable in principle, not predicted by chip model* |

- **Why Plain Fixed-Phase Signs Flipped Between `13` and `26` Hops:** In `[4.4, 5.2] GHz`, $-\cos(2\pi f / 3.526\text{ GHz})$ has *positive* slope (correlating negatively with decreasing $T_1(f)$ on `11/11` Eagle chips), whereas $-\cos(2\pi f / 1.763\text{ GHz})$ has *negative* slope (correlating positively on `10/10` Eagle chips). Once `[1, f]` is residualized out (`coaxial_t1_test_v2.py`), **0 of 11** devices are significant at either length ($p \in [0.059, 0.917]$).
- **Minimum Detectable Modulation (`MDE_80%`):** After linear detrending at `3.5256 GHz`, $\mathrm{MDE}_{80\%}$ ranges from **`1.05×` to `20.9×` mean $T_1$** (e.g., `1,130 µs = 4.84× mean T1` on `FakeBrisbane`), exceeding physically positive $T_1$ amplitudes.

---

## 4. Four-Model BIC & 5-Fold Held-Out Cross-Validation + Layout Mimicry

Comparing **`M0_linear`**, **`M1_quadratic`**, **`M2_huber_local_loss`** (TLS dropout robust), and **`M3` (`3.526 GHz` / `1.763 GHz` collective modes)** across all 28 processors (`docs/phase2/coaxial_t1_reassessment_results.json`):
- **BIC Winner:** `M0_linear` wins on **`27 / 28`** processors (`M3` wins on `1 / 28`, consistent with chance).
- **5-Fold Held-Out Qubit CV RMSE:** `M3a` (`3.526 GHz` fixed-phase) improves held-out RMSE by $\ge 1.0\,\mu\text{s}$ over `M0_linear` on only **`1 / 28`** devices (and worsens CV RMSE on the majority of Eagle chips, e.g. `FakeBrisbane`: `M2_huber` `79.45 µs` < `M0_linear` `79.57 µs` < `M3a` `79.64 µs` < `M3c` `80.06 µs`).
- **Layout-Dependent Mimicry:** Heavy-hex frequency allocation assigns discrete frequency groups (`~4.6, 4.8, 5.0 GHz`) across bipartite sublattices and readout multiplexers while wafer-scale $T_1$ gradients run across physical qubit indices (e.g., `FakeBrisbane` qubit-index vs $T_1$ correlation **$r = -0.303, p = 0.00053$**), creating structured $T_1(f)$ curvature without any collective chip-bus standing wave.
- **Final Verdict:** Mechanism is classified **`INCONCLUSIVE`**. Detecting a collective bus mode would require flux-tunable single-qubit $T_1(\omega)$ spectroscopy sweeping individual qubits continuously across $\ge 1.5\text{ GHz}$ (`>1` full period), holding spatial position and local TLS environment fixed.
