# Competing Physical Models for the ~46 µs Revival (`TAU_REVIVAL_01_MODEL_COMPARISON.md`)

**Experiment ID:** `TAU-REVIVAL-01`  
**Status:** `Implemented and tested` (`phase2/tau_revival_01/tau_revival_models.py`, `tests/test_tau_revival_01.py`)

---

## 1. Candidate Mechanisms Under Test

No universal physical constant is presupposed. Four candidate classes are formulated with explicit quantitative equations:

### Mechanism 1 — Ordinary Pair-Dependent Interference / Detuning / Residual $ZZ$ (`ConventionalBeatModel`)
When single-qubit drive frequency $f_{\text{drive}}$ differs from dressed transition frequency $f_{\text{qubit}}$ by $\Delta f_0$ (or residual static coupling shifts a transition by $\zeta / 2\pi$), Ramsey / incomplete-tomography population observables oscillate at the beat period:
$$T_{\text{beat}} = \frac{1}{|\Delta f_{\text{eff}}|}, \qquad \Delta f_{\text{eff}} = \Delta f_0 + f_{\text{inj}}$$
- A characteristic period $T_{\text{beat}} \approx 46.1\,\mu\text{s}$ corresponds to an ordinary detuning or coupling shift of **$|\Delta f_0| \approx 21.69\text{ kHz}$** — squarely within standard transmon calibration drift and fixed-coupler residual $ZZ$ ranges (`5–50 kHz`, cf. `Bridge E4a` frequency jump diagnostics and `CLAIM_RECONCILIATION.md` `NB-06`).
- **Three Strict Predictions:**
  1. **Pair & Epoch Dependence:** $T_{\text{beat}}$ varies across physical pairs (`CV > 50%`, matching OSIRIS's own Feb 2026 `ZZ_UNIVERSAL` self-test showing `6.7–333 µs` periods!) and shifts across recalibration cycles.
  2. **Unit-Slope Virtual-Z Shift:** Adding a controlled software virtual-$Z$ ramp $\phi(\tau) = 2\pi f_{\text{inj}} \tau$ shifts the observed oscillation frequency linearly with **exact slope $\frac{d f_{\text{obs}}}{d f_{\text{inj}}} = +1.0$** (`T_obs = 1 / |\Delta f_0 + f_inj|`).
  3. **Hahn Echo Cancellation:** Inserting a balanced Hahn $\pi$-pulse at $\tau/2$ refocuses static detuning, eliminating the `~46 µs` peak into monotonic $T_2^{\text{echo}}$ decay.

### Mechanism 2 — Calibration / Sampling / Aliasing / Timestamp Artifact
- **Server-Clock Modulo Artifact (`TAU_PHASE`):** Computing $\phi_\tau = (t_{\text{unix}} \bmod 46\,\mu\text{s}) / 46\,\mu\text{s}$ bins job wall-clock start times rather than qubit evolution delays.
- **Coarse Delay-Grid Aliasing (`compute_aliased_frequency_khz`):** Sampling delays every $\Delta\tau = 10\,\mu\text{s}$ ($f_s = 100\text{ kHz}$, Nyquist $50\text{ kHz}$) folds *any* detuning $f_{\text{true}} = m \cdot 100\text{ kHz} \pm 21.7\text{ kHz}$ (e.g., $78.3\text{ kHz}$ or $121.7\text{ kHz}$) onto an apparent `46.1 µs` period.
- **Population-Only Estimator Artifact:** Measuring $P_{00} + P_{11}$ without $XX$ and $YY$ parity bases mistakesclassical $T_1$ relaxation toward $|00\rangle$ ($P_{00} \to 1 \implies P_{00}+P_{11} \to 1$!) or local 1Q rotations for entanglement revival (`K8_REVIVAL`, where full 3-basis tomography on `ibm_fez` Job `dbiobimne4hc73bf2am0` proved $\mathcal{F}_{\text{Bell}}(46\,\mu\text{s}) = 0.3438 < 0.50$ and $C_\perp = +0.0029 \pm 0.0221$).

### Mechanism 3 — Correlated Non-Markovian Bath Mode (`H3` with Narrowband TLS / Cavity Mode)
- A discrete environmental defect (coherent two-level system TLS at splitting $\Delta_{\text{TLS}}/2\pi \approx 21.7\text{ kHz}$) exchanges excitations coherently with a specific transmon, producing a genuine Jaynes–Cummings-like coherence revival on *that specific qubit*, which **persists under Hahn echo** if the TLS flips coherently, **but remains localized to that qubit** (not universal across pairs or chips).

### Mechanism 4 — Hypothesized Universal Physical Constant (`UniversalTau0Model`, $\tau_0 = \varphi^8 \times 1\,\mu\text{s} = 46.9787\,\mu\text{s}$)
- **Note on Parameter Count (`NB-06`):** Because $1\,\mu\text{s}$ is a human-chosen SI time unit, $\tau_0 = \varphi^8 \times 1\,\mu\text{s}$ contains an arbitrary unit scale choice and is not dimensionally parameter-free.
- **Strict Non-Retuning Prediction:** If $\tau_0 = 46.9787\,\mu\text{s}$ were a universal physical constant, full 3-basis Bell coherence $C_\perp(\tau) = \frac{1}{2}(\langle XX\rangle - \langle YY\rangle)$ MUST exhibit a statistically significant local maximum at $\tau = 46.98 \pm 1.0\,\mu\text{s}$ **simultaneously across multiple distinct qubit pairs, multiple calibration epochs, and multiple backends, with ZERO per-pair or per-device retuning of $\tau_0$**.

---

## 2. Direct Falsification Summary Table

| Test Observable | Conventional Detuning (`M1`) | Sampling / Readout Artifact (`M2`) | Local Coherent TLS (`M3`) | Universal $\tau_0 = 46.98\,\mu\text{s}$ (`M4`) |
|---|---|---|---|---|
| **Full 3-basis $C_\perp(46.98\,\mu\text{s})$** | Oscillates if basis rotated; $F_{\text{Bell}}$ tracks 2Q phase | **$C_\perp \approx 0$** ($P_{00}\uparrow$ from $T_1$ damping only) | Non-zero on 1 TLS-coupled qubit only | Non-zero $>0$ across **all** pairs (`Falsified` on `ibm_fez` `[0,1]`: $C_\perp = +0.0029 \pm 0.0221$) |
| **Pair-to-pair period variation** | **High** ($\mathrm{CV} > 50\%$, $T_i = 1/\|\Delta f_i\|$) | Grid-dependent | Present on TLS-defect qubit only | **Zero** ($\tau_{\text{peak}} \equiv 46.98\,\mu\text{s}$ on all pairs) |
| **Virtual-$Z$ slope $df_{\text{obs}}/df_{\text{inj}}$** | **`+1.00`** (shifts peak continuously) | Distorts aliased fold | Shifts detuned TLS hybridization | Peak locked at `46.98 µs` |
| **Sub-Nyquist dense sampling (`Δτ = 0.5 µs`)** | Resolves true $\Delta f_0$ without folding | **Eliminates** aliased peak | Preserves local TLS oscillation | Preserves `46.98 µs` peak |
| **Hahn spin-echo (`π` at `τ/2`)** | **Eliminates** static beat completely | Eliminates static frame error | Partial survival if resonant | Must survive if non-static memory |
