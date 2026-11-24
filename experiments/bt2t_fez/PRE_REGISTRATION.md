# Write-Ahead Preregistration — `OSIRIS-BT2T-001-FEZ`

- **Protocol ID:** `OSIRIS-BT2T-001-FEZ` (`Card A` in `phase1/verification_queue.md`, Milestone 1 Hardware Stage)
- **Target Backend:** `ibm_fez` (156-qubit IBM Heron r2 processor; `pending_jobs = 0` at staging)
- **Pinned Physical 5-Transmon Heavy-Hex T-Junction (`T5`):** `FEZ_T5_LAYOUT = [21, 22, 23, 16, 24]`
  - Undirected physical edges: `(21, 22), (22, 23), (23, 16), (23, 24)` (`deg(23) = 3` hub; exact isomorphism `[0, 1, 2, 3, 4] -> [21, 22, 23, 16, 24]` of `verify.HEAVY_HEX_5Q`)
  - Selection rule (fixed before viewing any BT2T hardware counts): lowest calibration composite score `cz_sum + 0.5 * ro_mean + 5.0 * sx_mean = 0.01502` (`cz_sum = 0.00921`, `ro_mean = 0.0085`, `sx_mean = 0.00031`) among all 5-qubit T-junctions on `ibm_fez` with valid `T1 > 80 us` and `T2 > 50 us`.
- **Frozen Encoding:** `bt2t-enc-1` (`sha256 = 3e3b02d13539daaf431af39056d031f777a785a556eb9da21ef4ea6097dfc4c5`)
- **Independent Gate G1 Sign-Off:** `experiments/bt2t_inversion/G1_INDEPENDENT_REVIEW.md` (`PASS`)

---

## 1. Experimental Arms, Equal-Shot Rule, and Schedule

Three circuit arms are evaluated across **all 32 computational basis inputs** $x \in \{0,\dots,31\}$ (`24` valid $2T$ group elements $x \in \{0,\dots,23\}$ and `8` unused codewords $x \in \{24,\dots,31\}$):

1. **`STRUCTURED` (`72 CZ`, `151 SX`, `depth = 218` unitary core):**
   - Compiled from `bt2t.inversion_circuit_structured()` onto `HEAVY_HEX_5Q` (`optimization_level=3, seed_transpiler=20261122`).
   - Basis input $|x\rangle$ prepared via native `X` gates on `FEZ_T5_LAYOUT[init_idx[q]]` (`init_idx = [2, 0, 3, 1, 4]`) before a `barrier`, followed by the **identical** frozen `72-CZ` unitary core for all 32 inputs, `barrier`, and measurement of `FEZ_T5_LAYOUT[final_idx[q]]` (`final_idx = [4, 3, 1, 2, 0]`) into classical bit `c[q]`.
2. **`GENERIC` (`723 CZ`, `1560 SX`, `depth = 2073` unitary core):**
   - Compiled from `bt2t.inversion_circuit()` (11 disjoint transpositions) onto the **exact same** `HEAVY_HEX_5Q` coupling map and **exact same** 5 physical transmons `[21, 22, 23, 16, 24]` (`optimization_level=3, seed_transpiler=20261122`, `init_idx = [4, 1, 0, 3, 2]`, `final_idx = [1, 2, 0, 4, 3]`).
3. **`PREP_RO_CTRL` (`0 CZ`, preparation + readout SPAM control):**
   - Prepares $|x\rangle$ on `FEZ_T5_LAYOUT[ctrl_idx[q]]` (`ctrl_idx = [4, 3, 1, 2, 0]`, matching `STRUCTURED` readout assignment), `barrier`, and immediately measures into `c[q]` (`expected_output = x`).

### Equal-Shot Rule & Quota Budget
- **Randomized Blocks:** `N_BLOCKS = 3` temporally interleaved blocks (`SCHEDULE_SEED = 20261123`). Within each block $b \in \{0, 1, 2\}$, all `96` `(arm, input_x)` circuits appear in deterministic pseudo-random order (`288` total PUBs $\le 300 = \text{max\_experiments}$).
- **Equal-Shot Rule:** Every `(arm, input_x)` cell receives **exactly `96` shots per block**, i.e., **`288` shots total** across the 3 blocks (`zero` postselection; `100%` shot retention).
- **Total Shots:** `288 PUBs * 96 shots/PUB = 27,648` total shots (`9,216` shots per arm: `6,912` valid-24 shots + `2,304` unused-8 shots).
- **Predicted QPU Charge Time:** `~10.5–12.0 s` (`<= 15 s` ceiling, safely inside the `19 s` remaining `troykey.json` Open Plan quota).

---

## 2. Primary Metrics, Separation of Subspaces, and Unit of Inference

Per `Card A` (`phase1/verification_queue.md`), valid-group, unused-codeword, and SPAM-control accuracies are NEVER pooled together:

1. **Primary Metric (`P_valid24`, uniform weighting over $2T$):**
   $$
   \widehat{P}_{\text{valid-24}}(\text{arm}) = \frac{1}{24}\sum_{x=0}^{23} \widehat{P}_{\text{arm}}\big(c = \pi(x) \,\big|\, \text{in}=x\big),
   $$
   where $\pi(x) \in \{0,\dots,23\}$ is the exact group inverse in `bt2t-enc-1`.
2. **Secondary Metric A (`P_unused8`, uniform weighting over unused codewords):**
   $$
   \widehat{P}_{\text{unused-8}}(\text{arm}) = \frac{1}{8}\sum_{x=24}^{31} \widehat{P}_{\text{arm}}\big(c = x \,\big|\, \text{in}=x\big).
   $$
3. **Secondary Metric B (`L_leakage`, valid-set leakage into unused subspace):**
   $$
   \widehat{L}_{24\to\text{unused}}(\text{arm}) = \frac{1}{24}\sum_{x=0}^{23} \widehat{P}_{\text{arm}}\big(c \in \{24,\dots,31\} \,\big|\, \text{in}=x\big) \quad (\text{random uniform 5-bit baseline } = 8/32 = 0.25).
   $$
4. **Preparation + Readout Control (`PREP_RO_CTRL`):**
   - $\widehat{P}_{\text{valid-24}}(\text{PREP\_RO\_CTRL})$ and $\widehat{P}_{\text{unused-8}}(\text{PREP\_RO\_CTRL})$ isolate pure 5-bit state-preparation and measurement fidelity ($0$ `CZ` gates).

### Unit of Inference Discipline (Mandatory per `Card A`)
- **Cross-Window Scope (`N_windows = 1`):** All `288` PUBs execute in a single Qiskit Runtime job within one `ibm_fez` calibration window on `FEZ_T5_LAYOUT`. Individual shots ($N = 27,648$) are **NOT** treated as cross-calibration-window replicates; claims are explicitly scoped to this attested calibration window (`HARDWARE_ATTESTED`, `N_windows = 1`).
- **Within-Window Inferential Checks & Hypotheses:**
  - **`H1` (Structured vs Generic Synthesis Advantage on Valid $2T$):**
    $$
    \Delta \widehat{P}_{\text{valid-24}} = \widehat{P}_{\text{valid-24}}(\text{STRUCTURED}) - \widehat{P}_{\text{valid-24}}(\text{GENERIC}) > 0.25
    $$
    with **both**:
    1. All `3/3` interleaved blocks positive ($\Delta \widehat{P}_{\text{valid-24}}^{(b)} > 0.25$ for $b \in \{0, 1, 2\}$, paired block-$t$ one-sided $p < 0.01$ at $\text{df}=2$), AND
    2. Paired element-wise two-sided Wilcoxon signed-rank test across the $N_{\text{inputs}} = 24$ group elements $d_x = \widehat{P}_{\text{struct}}(\pi(x)|x) - \widehat{P}_{\text{gen}}(\pi(x)|x)$ significant at $p < 10^{-4}$.
  - **`H2` (Above-Chance Unused-Codeword Preservation & Subspace Confinement):**
    - $\widehat{P}_{\text{unused-8}}(\text{STRUCTURED}) > \widehat{P}_{\text{unused-8}}(\text{GENERIC}) + 0.20$ AND valid-set leakage $\widehat{L}_{24\to\text{unused}}(\text{STRUCTURED}) < 0.15$ (strictly below the $8/32 = 0.25$ uniform-depolarized baseline).
  - **`H3` (SPAM Baseline Integrity):**
    - $\widehat{P}_{\text{valid-24}}(\text{PREP\_RO\_CTRL}) \ge 0.90$ and $\widehat{P}_{\text{unused-8}}(\text{PREP\_RO\_CTRL}) \ge 0.90$, confirming 5-qubit SPAM errors do not confound the synthesis comparison.
  - **Falsifier:** Any failure of `H1` or `H3`, or any stage-0 noiseless mismatch on any of the `96` `(arm, input_x)` circuits, marks the corresponding hypothesis `REFUTED` / `HALTED`.
