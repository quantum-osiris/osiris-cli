# Paper 10: Maximal-Tree Gauge-Fixed Binary Tetrahedral ($2T \subset SU(2)$) Kogut–Susskind Plaquette Dynamics and Live Two-Window `OSIRIS-ECHO` Drift Catch on `ibm_fez`

- **Manuscript Directory:** `papers/bt2t003_kogut_susskind_plaquette/`
- **Author:** Devin Phillip Davis (Agile Defense Systems LLC, Lexington, Kentucky, USA)
- **Frozen Register Encoding:** `bt2t-enc-1` (`sha256 = 3e3b02d13539daaf431af39056d031f777a785a556eb9da21ef4ea6097dfc4c5`)

## Dual-Manifest Provenance Binding

1. **Theory & Exact Simulation Manifest (`SIMULATED` / `EXACT_STATEVECTOR`, `OSIRIS-BT2T-003`):**
   - Path: `../../experiments/bt2t_plaquette/manifest_bt2t003.json` (`34/34` exact checks PASS, `provider_job_id = null`, `0` QPU s, RFC 8785 canonical SHA-256 `a913c676901a0beb66743f6702424f0c1791146093946a2ba684210fbd618c0d`).
   - Maximal-tree gauge fixing collapses the unfixed 4-link single-plaquette Hilbert space from $|2T|^4 = 24^4 = 331,776$ group configurations (`20` qubits padded to $32^4 = 1,048,576$ or `25` qubits with a 5-qubit holonomy ancilla) down to $|2T| = 24$ physical holonomy states on `5` qubits (`32`-D register with `24` active `bt2t-enc-1` states, `8` strictly decoupled states, and `7`-dimensional conjugation-invariant physical Gauss subspace), matching unfixed 4-link evolution to `2.27e-15` across `20` random gauge superpositions.
   - Exact magnetic plaquette potential $V_B(U) = \frac{1}{2}(2 - \operatorname{Re}\chi_{2D}(U)) \in \{0, \frac{1}{2}, 1, \frac{3}{2}, 2\}$ across the `7` conjugacy classes (`C1..C7`) compiles via a 5-qubit Walsh–Hadamard Gray-code phase circuit to `16` `CZ` gates (all-to-all) or `19` core `CZ` gates on heavy-hex `T5` (`40` `CZ` total with Center/Null state prep/unprep vs `2,727` `CZ` for generic dense $32\times 32$ QFT-conjugated `UnitaryGate` synthesis = `68.18x` native `CZ` reduction on `ibm_fez`).
   - Second-order Strang splitting $S_2(\Delta t) = U_B(\Delta t/2) \widetilde{F}_{2T}^\dagger D_E(\Delta t) \widetilde{F}_{2T} U_B(\Delta t/2)$ compiles to `209` `CZ` (all-to-all) / `350` `CZ` (heavy-hex `T5`), achieving `104.33x` to `178.31x` fewer `CZ` gates than unfixed 20Q/25Q 4-link Trotterization (`21,804` to `62,407` `CZ`) and log-log second-order convergence slope `2.000008`.

2. **Live Hardware Execution & Two-Window `OSIRIS-ECHO` Drift-Catch Manifest (`HARDWARE_ATTESTED`, `OSIRIS-BT2T-003-FEZ`):**
   - Path: `../../experiments/bt2t003_fez/physical_bt2t003_fez_results.json` (`ibm_fez`, `provider_job_id = dc38nreq4uuc73a6g0k0`, `19` QPU s, `240` PUBs $\times$ `256` = `61,440` shots across `10` disjoint `T5` subgraphs = `50` physical transmons; write-ahead commit `82cfd46a`, `qpy_sha256 = 349a7ab40a1a55db6ca85dbfd27c24f75aa3e4c67c20accad82a7a41b2c87b9a`, `raw_counts_gz_sha256 = 4667adc71b3af9b7dd3c9dff1fde9d4d6acaf768597992bfbc39bbde2c0abeb3`).
   - **Character-Sector Trajectories on `#1` Echo-selected `pid = 0` (`[21, 22, 23, 16, 24]`):**
     - **Center sector ($\Delta\chi_{2D} = 4$, $P_{|+\rangle}(\tau) = \cos^2(2\tau)$):** Measured $P_{|+\rangle}(\tau) = [0.876302, 0.411458, 0.057292, 0.841146]$ (Pearson $r = 0.997576$), with $P_{|-\rangle}(\pi/4) = 0.850260$ (`85.03%` inversion into routed `'00010'`) vs `0.041667` for the `2,727-CZ` generic dense comparator (`+0.808594` = `+80.86` pp / `20.41x` advantage).
     - **Hurwitz half-frequency sector ($\Delta\chi_{2D} = 2$, $P_{\Phi^+}(\tau) = \cos^2(\tau)$):** Measured $P_{\Phi^+}(\tau) = [0.829427, 0.669271, 0.404948, 0.130208, 0.050781]$ (Pearson $r = 0.997940$), with $P_{\Phi^-}(\pi/2) = 0.796875$ (`79.69%` inversion into routed `'00100'`).
     - **Zero-differential-character null controls ($\Delta\chi_{2D} = 0$ at $\tau = \pi/4$):** Imaginary-$Q_8$ $|i\rangle$ survival `0.861979` (`'00001'`) and unused-8 padded subspace $|y=24\rangle$ survival `0.785156` (`'01100'`).
   - **Live Two-Window (`N_windows = 2`, `+3.763` h) `OSIRIS-ECHO` Intra-Calibration Drift Catch (`dc35e35s2omc738cvcug` Window 1 $\to$ `dc38nreq4uuc73a6g0k0` Window 2):**
     - Hub qubit **Q7** on `pid = 1` (`[5, 6, 7, 8, 17]`) experienced an unflagged `+31.12%` (`0.311198`) $|0\rangle \to |1\rangle$ SPAM/readout drift excursion (`IDENTITY_CTRL_0CZ` dropping `0.962240 -> 0.677083`, `ECHO_PROBE_16CZ` dropping `0.815104 -> 0.562500`).
     - Stale calibration remained frozen on `pid = 1` (`#1`), collapsing in Window 2 to Spearman $\rho_{\text{cal}} = +0.454545$ with `0.361002` top-1 selection regret, whereas live `ECHO_PROBE_16CZ` switched `#1` to `pid = 0`, achieving $\rho_{\text{echo}} = +0.948333$ ($\Delta\rho = +0.493787$) and `0.000000` top-1 selection regret across both windows (`2-window mean` $\bar{\rho}_{\text{echo}} = +0.834772$ vs $\bar{\rho}_{\text{cal}} = +0.618182$, $\Delta\bar{\rho} = +0.216591$).

## Reproducible Build & Verification

```bash
SOURCE_DATE_EPOCH=1700000000 PYTHONHASHSEED=0 python papers/bt2t003_kogut_susskind_plaquette/build_paper_assets.py
(cd papers/bt2t003_kogut_susskind_plaquette && sha256sum -c SHA256SUMS.txt)
pytest -q tests/test_bt2t003_paper.py
```
