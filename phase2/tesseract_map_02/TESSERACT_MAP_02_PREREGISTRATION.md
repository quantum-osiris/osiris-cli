# Pre-Registration: `EXPERIMENT TESSERACT-MAP-02`
## Controlled Decomposition of Degree-Reducing Circuit Compilation, Subgraph Selection, and Routing on 40-Qubit Heavy-Hex Arrays

**Experiment ID:** `TESSERACT-MAP-02`  
**Status:** `Implemented and tested` (offline code, equivalence proof, and calibrated 15-block simulations complete; QPU execution `Proposed only` — **halted at QPU submission boundary awaiting explicit owner approval**)  
**Target Hardware:** `ibm_marrakesh` or `ibm_fez` (156-qubit IBM Heron r2; `ibm_kingston` if accessible on authorized plan)  
**Baseline Anchor:** Archived rerun `dansntf8gn2s739n32e0` (`ibm_kingston`, 2026-09-20; `tesseract_rerun_2026-09-20.zip`, `sha256: cdebc558424a7cb970a76bb084242ba52bbd985f50ad96aff1316da666402b13`), where 4-string support improved from `5.305%` (`P0_replication`, $F_{\text{cl}} = 0.0498$) to `17.995%` (`P2_chain_mapped`, $F_{\text{cl}} = 0.1645$) across four simultaneously changed factors.

---

## 1. Experimental Factors, Structural Couplings, and Matched Contrasts

### 1.1 Factors Examined
1. **Logical circuit structure:** `star` (`cx(i % 2, i)`, hub degree 20) vs `chain` (`cx(2k-2, 2k); cx(2k-1, 2k+1)`, degree 2).
2. **Physical-qubit selection:** `sabre_free` / `VF2Layout` compiler selection vs `best_path` (low-error 40Q simple path) vs `median_path` (50th-percentile operational 40Q simple path).
3. **Routing strategy & 2Q gate count / depth:** `routing="sabre"` (SWAP-routed) vs `routing="none"` (contiguous 0-SWAP mapping).
4. **Transpiler optimization level:** `optimization_level = 1` vs `optimization_level = 3` (which enables 2-qubit KAK block consolidation `ConsolidateBlocks` and `VF2Layout` subgraph isomorphism).
5. **Hardware calibration & SPAM baseline:** Per-block `|0⟩^40` and `|1⟩^40` SPAM readout circuits (`C0_SPAM_ALL0`, `C0_SPAM_ALL1`) on the exact target physical qubits.

### 1.2 Why a Fully Crossed Factorial Design Is Structurally Infeasible (Coupled Factors)
On a planar heavy-hex coupling lattice (maximum vertex degree $3$), a degree-20 `star` graph is **not subgraph-isomorphic** to any 40-qubit hardware subset. Consequently, `(logical_structure="star", routing="none")` is physically impossible, coupling `logical_structure` + `routing` directly to `(twoq_gate_count, circuit_depth)`. We therefore isolate every identifiable mechanism via **7 matched test arms + 2 SPAM baseline controls** per interleaved block:

| Arm ID | Logical | Layout Policy | Routing | Opt Level | `FakeKingston` 2Q Gates | `FakeKingston` Depth | Role |
|---|---|---|---|---:|---:|---:|---|
| `C0_SPAM_ALL0` | $\|0\rangle^{\otimes 40}$ | `best_path` | `none` | 1 | 0 | 1 | Per-qubit $P(1\|0)$ readout baseline |
| `C0_SPAM_ALL1` | $\|1\rangle^{\otimes 40}$ | `best_path` | `none` | 1 | 0 | 2 | Per-qubit $P(0\|1)$ readout / $T_1$ baseline |
| `C1_STAR_FREE_OPT3` | `star` | `sabre_free` | `sabre` | 3 | 163 | 214 | Historical `P0` replication |
| `C2_STAR_BESTPATH_OPT3` | `star` | `best_path` | `sabre` | 3 | 79 | 91 | Fixed-qubit `star` at `opt=3` (KAK-consolidated SWAPs) |
| `C3_STAR_BESTPATH_OPT1` | `star` | `best_path` | `sabre` | 1 | 147 | 179 | Fixed-qubit `star` at `opt=1` (unconsolidated SWAPs) |
| `C4_CHAIN_BESTPATH_OPT1` | `chain` | `best_path` | `none` | 1 | 39 | 86 | Historical `P2` replication |
| `C5_CHAIN_BESTPATH_OPT3` | `chain` | `best_path` | `none` | 3 | 39 | 66 | Fixed-qubit `chain` at `opt=3` (0 SWAPs) |
| `C6_CHAIN_MEDPATH_OPT3` | `chain` | `median_path` | `none` | 3 | 39 | 66 | Matched-39-CZ qubit-quality contrast |
| `C7_CHAIN_FREE_OPT3` | `chain` | `sabre_free` (`VF2`) | `sabre` (0 SWAPs) | 3 | 39 | 66 | Pure logical rewrite handed to stock Qiskit `VF2Layout` |

### 1.3 Five Pre-Specified Identifiable Causal Contrasts
1. **$\Delta_{\text{route|opt3}} = Y(C5) - Y(C2)$:** Effect of degree-reducing logical rewrite + SWAP elimination (`79 -> 39` CZ, depth `91 -> 66`) on the **exact same 40 physical qubits** at `opt_level=3`.
2. **$\Delta_{\text{route|opt1}} = Y(C4) - Y(C3)$:** Effect of degree-reducing logical rewrite + SWAP elimination (`147 -> 39` CZ, depth `179 -> 86`) on the **exact same 40 physical qubits** at `opt_level=1`.
3. **$\Delta_{\text{qubits|39cz}} = Y(C5) - Y(C6)$:** Pure effect of physical-qubit calibration quality (`best_path` vs `median_path`) holding **39 CZ gates, 0 SWAPs, depth 66, and `opt=3` strictly constant**.
4. **$\Delta_{\text{opt|chain}} = Y(C5) - Y(C4)$ (Pre-Registered Virtual-$Z$ / A/A Drift Control):** On IBM Heron/Eagle (`rz` = zero-duration virtual-$Z$ NCO update), `C5` (`depth 66`) and `C4` (`depth 86`) compile to **identical physical microwave schedules (`81 sx, 39 cz`, `4,772 ns` on `ibm_marrakesh`)**, differing only by 40 virtual-`RZ` updates. Expected causal difference is $\Delta F_{\text{expected}} = 0.000$; any observed discrepancy directly bounds residual unpaired intra-block drift.
5. **$\Delta_{\text{path\_vs\_vf2}} = Y(C5) - Y(C7)$:** Tests whether custom greedy DFS path selection outperforms stock Qiskit error-weighted subgraph isomorphism (`VF2Layout`; Nation & Treinish, *PRX Quantum* 4, 010327, 2023) once the algebraic `star -> chain` fan-out rewrite reduces max vertex degree to $\le 2$.

---

## 2. Pre-Execution Logical Equivalence & Noise Asymmetry Proof

Verified in `phase2/tesseract_map_02/tesseract_map_02.py::verify_logical_equivalence`:
- **Exact Statevector Equivalence ($N = 12$):** $|\langle \psi_{\text{star}} | \psi_{\text{chain}} \rangle| = 1.000000000000000$; maximum output probability discrepancy across all $2^{12}$ basis states (with and without the inert pre-measurement `RZ`) is $< 10^{-15}$.
- **Exact $\mathrm{GF}(2)$ Propagation Equivalence ($N = 40$):** Every even wire $2k$ copies $q_0$ and every odd wire $2k+1$ copies $q_1$, yielding the exact 4-string ideal support $\mathcal{S}_4 = \{0^{40}, 1^{40}, (10)^{20}, (01)^{20}\}$ with probabilities $\{\frac{1}{2}\cos^2\theta, \frac{1}{2}\cos^2\theta, \frac{1}{2}\sin^2\theta, \frac{1}{2}\sin^2\theta\}$ ($\theta = 51.7^\circ$).
- **Non-Equivalent Mid-Circuit Fault Propagation (E-84 Control):** A single mid-circuit $X$ fault on wire $q_2$ flips **1 qubit** in `star` vs **19 downstream even qubits** in `chain` (domain-wall strings). Therefore, **exact-support endpoints** ($F_{\text{cl}}$ and $P(\mathcal{S}_4)$) are primary; Hamming-distance-1 metrics are diagnostic only, and `reduce_commuting_cx_fanouts` must be disabled inside QEC syndrome-extraction gadgets.

---

## 3. Endpoints, Controls, and Blocked Statistical Plan

- **Primary Endpoint:** Classical Bhattacharyya fidelity to the exact 4-string target distribution:
  $$F_{\text{cl}}(P_{\text{emp}}, P_{\text{ideal}}) = \left(\sum_{x \in \mathcal{S}_4} \sqrt{P_{\text{emp}}(x)\, P_{\text{ideal}}(x)}\right)^2$$
- **Secondary Endpoints:** (a) Four-string ideal-support probability $P(\mathcal{S}_4) = \sum_{x \in \mathcal{S}_4} P_{\text{emp}}(x)$; (b) Total variation distance $D_{\text{TV}}$; (c) Full-union SPAM readout diagnostics (`build_union_spam_calibration_circuits`, covering 100% of physical qubits active across `C1..C7`, not solely `best_path`).
- **Pre-Lock Calibration Separation (Preventing Winner's Curse):** Path scoring across the 400 candidate paths (`best_path`, `median_path`) and `VF2Layout` embedding are frozen on the **pre-submission calibration snapshot $t_{\text{select}}$** before locking the QPY manifest; hardware evaluation runs on subsequent execution blocks $t_{\text{eval}}$ with live execution-time `backend.properties()` archived alongside each block.
- **Mandatory Artifact Logging (Fixing E-83):** Every compiled circuit saves its exact 40-element `final_physical_layout` (`extract_final_physical_layout`), QPY binary, structural SHA-256 digest, 2Q gate count, depth, backend calibration snapshot JSON + calibration timestamp, randomized block order, raw counts, and IBM Job IDs before analysis.
- **Blocked Replication & Multiple-Testing Plan:**
  - **Replication unit:** One complete **randomized interleaved block** $b \in \{1, \dots, B\}$ containing all 9 circuits in freshly shuffled order at $S = 4{,}096$ shots/circuit (`36,864` shots/block). Individual shots are **never** treated as independent replications of the hardware condition.
  - **Sample Size Justification (`compute_required_blocks`):** Under empirical Heron r2 intra-session paired block-difference SD $\sigma_d = 0.014$ (`1.1 pp` drift + multinomial noise) and 5 Holm–Bonferroni contrasts ($\alpha_{\text{FWER}} = 0.05$, strictest $\alpha_1 = 0.010$), **$B = 15$ blocks** achieves **$85.8\%$ statistical power** to detect a minimum detectable effect $\mathrm{MDE} = 0.015$ (`1.5 pp` in $F_{\text{cl}}$).
  - **Inference:** Two-sided paired $t$-test across the $B = 15$ block differences ($df = 14$) with Holm–Bonferroni step-down FWER control at $\alpha = 0.05$, plus non-parametric Wilcoxon signed-rank sensitivity check.
