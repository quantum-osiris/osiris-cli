# Expected Outcomes & Decision Matrix: `EXPERIMENT TESSERACT-MAP-02`

**Status:** `Simulated` (`docs/phase2/tesseract_map_02_simulation_results.json`, $B = 15$ blocks × $4{,}096$ shots on `FakeKingston` and `FakeMarrakesh` calibration models).

---

## 1. Calibrated Offline Simulation Predictions (`B = 15` Interleaved Blocks)

| Arm | 2Q Gates (`CZ`) | Pre-Readout Span | `FakeKingston` Mean $F_{\text{cl}}$ (±SD) | `FakeMarrakesh` Mean $F_{\text{cl}}$ (±SD) | Physical Mechanism Tested |
|---|---:|---:|---:|---:|---|
| `C1_STAR_FREE_OPT3` | 163 | `9,380 ns` | `0.0639 ± 0.0051` | `0.0171 ± 0.0080` | Unrewritten degree-20 star + SABRE 2D routing (matches archived `ibm_kingston` `0.0498`!) |
| `C3_STAR_BESTPATH_OPT1` | 147 | `7,300 ns` | `0.0980 ± 0.0081` | `0.1072 ± 0.0090` | Midpoint-pinned star without 2Q KAK consolidation |
| `C2_STAR_BESTPATH_OPT3` | 79 | `3,692 ns` | `0.2219 ± 0.0087` | `0.2594 ± 0.0104` | Midpoint-pinned star + `ConsolidateBlocks` (`CX+SWAP -> 2 CZ`) |
| `C6_CHAIN_MEDPATH_OPT3` | 39 | `2,176 ns` | `0.2178 ± 0.0088` | `0.2783 ± 0.0120` | 0-SWAP chain on median-error physical path |
| `C4_CHAIN_BESTPATH_OPT1` | 39 | `2,188 ns` | `0.3185 ± 0.0128` | `0.4048 ± 0.0118` | 0-SWAP chain on greedy low-error path (`opt=1`, virtual-Z control) |
| `C5_CHAIN_BESTPATH_OPT3` | 39 | `2,188 ns` | `0.3262 ± 0.0097` | `0.4054 ± 0.0100` | 0-SWAP chain on greedy low-error path (`opt=3`, identical physical schedule to `C4`) |
| **`C7_CHAIN_FREE_OPT3`** | **39** | **`2,176 ns`** | **`0.3847 ± 0.0104`** | **`0.4404 ± 0.0107`** | **Degree-reduced rewrite (`star->chain`) + exact Qiskit `VF2Layout` subgraph isomorphism** |

---

## 2. Key Engineering Insight Uncovered in Simulation

Offline compilation revealed two non-obvious compiler phenomena that transform `TESSERACT-MAP-02` into a concrete OSIRIS compiler product capability:
1. **Why stock Qiskit failed on `P0` (`C1`) yet excels on `C7`:** Qiskit `optimization_level=3` runs exact error-weighted subgraph isomorphism (`VF2Layout`; Nation & Treinish, *PRX Quantum* 4, 010327, 2023), **but only when the circuit's interaction graph embeds directly into the heavy-hex coupling map**. Because `star` has vertex degree 20 while heavy-hex has max degree 3, `VF2Layout` cannot embed `C1`, forcing fallback to `SabreSwap` (`163 CZ` gates). Rewriting the commuting CNOT fan-out into a degree-2 `chain` (`C7`) via `reduce_commuting_cx_fanouts` unlocks `VF2Layout`, which automatically discovers a **39-CZ, 0-SWAP subgraph** outperforming greedy DFS path search (`C5`). Note that the engineering value is specifically the **algebraic fan-out degree-reduction pre-pass feeding stock `VF2Layout`**, not a replacement for `VF2Layout`.
2. **KAK Consolidation Halves Star SWAP Cost on 1D Paths (`C2` vs `C3`):** Placing hubs `(0, 1)` at the center of a 1D path allows `opt_level=3`'s `ConsolidateBlocks` pass to fuse each `CX + SWAP` pair (naively $1 + 3 = 4$ CZs) into a single $SU(4)$ block requiring only **2 CZs** (`147 -> 79` CZs).
3. **`C5` vs `C4` Is a Built-In Virtual-$Z$ A/A Null Control:** Because `rz` updates are zero-duration frame shifts, `C5` (`depth 66`) and `C4` (`depth 86`) share the exact same physical gate schedule (`81 sx, 39 cz`, `4,772 ns` on `ibm_marrakesh`); any significant `|C5 - C4| > 1.5 pp` indicates residual unpaired drift rather than physical gate compression. Note also that offline predictions come from a calibrated gate + idle-$T_1/T_2$ + domain-wall surrogate (`calibrated_gate_idle_t1t2_domainwall_surrogate`), not full 40Q density-matrix simulation.

---

## 3. Diagnostic Interpretation Matrix for Hardware Outcomes

| Hardware Contrast Pattern | Causal Conclusion | OSIRIS Product Action |
|---|---|---|
| `C5 > C2` AND `C5 > C6` (both Holm $p < 0.05$) | **Both** SWAP elimination (`79 -> 39 CZ`) and physical path quality (`best` vs `median`) contribute significantly and additively. | Ship combined **Commuting-Fanout Degree Reducer + Calibration-Weighted VF2 Pass** in `osiris.compiler`. |
| `C7 >= C5 >> C1` | Degree-reducing logical rewrite is the primary bottleneck; once rewritten to max-degree $\le 2$, exact `VF2Layout` is optimal without custom greedy path heuristics. | Focus OSIRIS compiler pass on algebraic stabilizer/CNOT tree degree reduction feeding `VF2Layout`. |
| `C5 - C2` small while `C0_SPAM` flags 1–2 outlier qubits (`>20%` readout bias) | Historical `P2` vs `P0` gap was dominated by avoiding faulty readout/leakage qubits (`E-83`). | Enforce pre-flight SPAM/readout exclusion mask in `osiris.compiler` layout selection. |
| `C5 - C4 > +1.5 pp` (`p_Holm < 0.05`) | Single-qubit layer compression (`depth 86 -> 66` at fixed `39 CZ`) materially reduces idle $T_1/T_2$ decay. | Enforce `opt_level=3` 1Q Euler synthesis after fixed-layout routing. |
