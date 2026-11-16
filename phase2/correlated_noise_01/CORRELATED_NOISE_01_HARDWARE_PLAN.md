# Hardware Execution & Resource Plan (`CORRELATED_NOISE_01_HARDWARE_PLAN.md`)

**Experiment ID:** `CORRELATED-NOISE-01`  
**Status:** `Proposed only` for hardware execution (**halted at QPU authorization gate; 0 RU spent**).

---

## 1. Two-Stage Gated Hardware Architecture (OpenPlan `< 600 s` Compliant)

To respect IBM Quantum OpenPlan's 10-minute (`600 s`) monthly window while testing both training and held-out couplers simultaneously without cross-job drift:

### Stage 1 — Multiplexed 10-Coupler Training + Held-Out Validation (`ibm_marrakesh`)
- **Programmatically Verified `>= 5`-Hop Disjoint Coupler Cohort (`select_disjoint_couplers_on_backend`):**
  All 10 couplers (`20` active qubits out of `156` on `ibm_marrakesh`) have **pairwise BFS graph distance $\ge 5$ hops** (eliminating 2-hop shared-spectator and shared-feedline collisions):
  - **Training Cohort ($G_{\text{train}} = 4$):** `(13, 14)`, `(1, 2)`, `(46, 47)`, `(98, 111)` (`CZ` errors `0.104%–0.170%`).
  - **Held-Out Validation Cohort ($G_{\text{test}} = 6$, locked prior to submission):** `(136, 143)`, `(62, 63)`, `(135, 139)`, `(54, 55)`, `(7, 8)`, `(127, 128)` (`CZ` errors `0.172%–0.241%`, providing $df = 5$ coupler-cluster degrees of freedom and $d \ge 6$ cross-board LO comparison).
- **Simultaneous PUB Multiplexing & Budget:**
  - Because all 10 couplers are $\ge 5$ hops apart, **all 4 training couplers and all 6 held-out couplers execute simultaneously in every circuit PUB** (zero extra circuits for held-out validation!).
  - **Compact OpenPlan Stage-1 Core (`18` conditions $\times$ avg `3.0` Pauli/readout circuits = `54` circuits/block):**
    - $B = 8$ randomized interleaved blocks $\times$ `54` PUBs $\times$ `4,200` shots = **`432` PUBs, `1,814,400` shots (`33,600` shots/basis/condition)** $\approx$ **`5.4 minutes` (`325 s < 600 s` OpenPlan cap)**!
  - **Full 27-Condition Extended Suite (Pay-As-You-Go / Dedicated Allocation Option):**
    - `27` conditions + `2` within-block A/A sentinels + $4\times 4$ readout calibration (`~66` PUBs/block $\times 8$ blocks = `528` PUBs, `2,217,600` shots $\approx$ **`6.7–8.5 minutes`**).

### Stage 2 — Cross-Chip Generalization Gate (`ibm_fez`, Conditional on Stage 1 Pass)
- Executed **only if** Stage 1 achieves both $\Delta\mathrm{BIC}_{\text{GLS}} \ge 8.0$ over `H0` and cluster-significant held-out coupler noisy RMSE improvement ($t_5$, $p < 0.05$).
- Tests frozen Stage 1 schedule policy on `6` `>= 5`-hop disjoint couplers on `ibm_fez` (`7` offsets $\times 3$ bases $\times 8$ blocks $\times 4{,}200$ shots = `168` multiplexed PUBs, `705,600` shots, **`~2.2–3.0 minutes` QPU time**).

---

## 2. Frozen Hardware Timing & ISA Transpilation Controls

1. **Frozen `4.0 ns` (`dt`) Timing Grid (`build_isa_staggered_dd_bell_circuit`):**
   - Evolution windows: $T_8 = \mathbf{8.192\,\mu\text{s}}$ (`2,048 dt`, slot spacing $\tau_{\text{slot}} = \mathbf{512.0\text{ ns}} = \mathbf{128\,\text{dt}}$ at $N = 16$) and $T_{16} = \mathbf{16.384\,\mu\text{s}}$ (`4,096 dt`).
   - Stagger offsets: $\Delta \in \{0, 16, 32, 48, 80, 160, 256\}\text{ ns} = \{0, 4, 8, 12, 20, 40, 64\}\,\text{dt}$ — every delay and pulse start is an **exact integer multiple of `dt = 4.0 ns`** with zero rounding residual.
   - Backend-specific finite 1Q `X/SX` durations are enforced explicitly: **`36.0 ns` (`9 dt`) on `ibm_marrakesh`** and **`24.0 ns` (`6 dt`) on `ibm_fez`**.
2. **Transpiler DD Collapse Prevention:** Following the forensic lesson of `dblr6lhkbkrc73co9h30`, all `delay(ticks, unit="dt")` slots are bracketed by barriers before `optimization_level=1` (`routing_method="none"`, `layout_method="trivial"`), and pre-submission QPY verification asserts exact `x`/`sx` and `delay` counts plus live `backend.properties()` snapshot archival.
3. **Authorization Gate:** Do **not** submit Stage 1 or Stage 2 without explicit PI approval of backend quota.
