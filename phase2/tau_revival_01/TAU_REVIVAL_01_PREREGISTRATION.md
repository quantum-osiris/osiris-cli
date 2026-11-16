# Pre-Registration: `EXPERIMENT TAU-REVIVAL-01` (`TAU_REVIVAL_01_PREREGISTRATION.md`)

**Experiment ID:** `TAU-REVIVAL-01`  
**Status:** `Implemented and tested` offline (`phase2/tau_revival_01/tau_revival_models.py`, `tests/test_tau_revival_01.py`); hardware execution `Proposed only` (**0 QPU jobs submitted; note that `K8_REVIVAL` is already refuted on `ibm_fez` `[0,1]`, so any new QPU run is low-priority behind `CORRELATED-NOISE-01` and `TESSERACT-MAP-02`**).

---

## 1. Pre-Registered Falsification Protocol (If Owner Authorizes Confirmatory Run)

1. **Frozen Universal Target (Zero Retuning):**
   - $\tau_0 = \varphi^8 \times 1.0\,\mu\text{s} = \mathbf{46.9787\,\mu\text{s}}$ is locked identically across all pairs, dates, and processors. Retuning $\tau_0$ to `46.0`, `46.1`, `46.5`, or `60.0 µs` after inspecting data is strictly prohibited.
2. **Multi-Pair, Multi-Epoch Cohort:**
   - `6` spatially disjoint physical pairs spanning distinct calibration detunings on `ibm_marrakesh` (or `ibm_fez`), repeated across `2` separate calibration epochs ($\ge 24\text{ h}$ apart).
3. **Anti-Aliased Dense Delay Grid:**
   - Uniform spacing $\Delta\tau = 1.0\,\mu\text{s}$ over $\tau \in [0, 80]\,\mu\text{s}$ (`81` points; Nyquist frequency $f_{\text{Nyq}} = 500\text{ kHz}$, $>20\times$ above `21.7 kHz`, eliminating coarse-grid aliasing and observation-window edge truncation).
4. **Conjugate-Quadrature Pauli Tomography + Sublattice-Selective Echo + Signed Virtual-$Z$ Ramp:**
   - **Arm A (Phase-Invariant 5-Basis Bell Tomography):** Measures $\langle XX\rangle, \langle YY\rangle, \langle XY\rangle, \langle YX\rangle, \langle ZZ\rangle$ to reconstruct complex transverse coherence $C_{\text{complex}}(\tau) = \frac{1}{2}(\langle XX - YY\rangle + i\langle XY + YX\rangle)$ and frame-invariant magnitude $|C_\perp(\tau)| = |C_{\text{complex}}(\tau)|$, separating genuine entanglement revival from $T_1$ relaxation $P_{00}(\tau) \to 1$ and preventing virtual-$Z$ phase rotation from rotating $|\Phi^+\rangle \to |\Phi^-\rangle$ out of a single quadrature.
   - **Arm B (Signed Virtual-$Z$ Detuning Ramp):** Applies deterministic virtual-$Z$ phase $\phi_{\text{inj}}(\tau) = 2\pi f_{\text{inj}}\tau$ with $f_{\text{inj}} \in \{-15, -7.5, 0, +7.5, +15\}\text{ kHz}$ on $C_{\text{complex}}(\tau)$, fitting **signed** frequency $f_{\text{signed}} = \Delta f_0 + f_{\text{inj}}$ (`slope = +1.0` without $|\Delta f_0 + f_{\text{inj}}|$ V-folding regardless of sign or magnitude of $\Delta f_0$).
   - **Arm C (Echo Controls):** Runs (C1) single-qubit/simultaneous Hahn echo to refocus static 1Q detuning and (C2) **sublattice-selective echo** ($\pi$-pulse on qubit $u$ only at $\tau/2$, which anticommutes with $Z_u Z_v$, whereas simultaneous 2Q echo commutes with $Z_u Z_v$) to refocus residual pair $ZZ$ coupling.

---

## 2. Statistical Decision & Multiple-Testing Rules

- **Primary Endpoint:** Phase-invariant transverse coherence excess $|C_\perp(\tau_0)|$ at $\tau_0 = 47.0\,\mu\text{s}$ relative to smooth monotonic exponential-stretched interpolation fitted over $[10, 38]\,\mu\text{s} \cup [56, 80]\,\mu\text{s}$, tested across all 6 pairs with **Holm–Bonferroni FWER control** ($\alpha_{\text{FWER}} = 0.05$, calibrated via block-permutation null testing).
- **Decisive Falsification Criteria:**
  1. **Universal Revival Falsified (`M4` Rejected):** If $< 5/6$ pairs show a Holm-adjusted ($p_{\text{Holm}} < 0.05$) positive $|C_\perp|$ peak within $46.98 \pm 1.5\,\mu\text{s}$, OR if peak times vary across pairs (`Cochran's Q` $p < 0.05$).
  2. **Conventional Detuning / Residual $ZZ$ Confirmed (`M1` Accepted):** If signed oscillation frequency shifts linearly with $f_{\text{inj}}$ (`95% CI` of slope contains `+1.0` and excludes `0.0`) and vanishes under Arm C1 (1Q detuning) or Arm C2 (sublattice-selective $ZZ$ echo).
