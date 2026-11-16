# Competing Physical Theory & Observable Discrimination (`CORRELATED_NOISE_01_THEORY.md`)

**Experiment ID:** `CORRELATED-NOISE-01`  
**Status:** `Implemented and tested` (`phase2/correlated_noise_01/correlated_noise_models.py`, `phase2/correlated_noise_01/test_correlated_noise_models.py`)  
**Central Scientific Question:** *Can measurements on superconducting transmon arrays distinguish spatially correlated or temporally non-Markovian decoherence from conventional independent local noise, classical control-line drift, and static spectator crosstalk—and can the inferred structure predict improved control schedules on unseen held-out qubits?*

---

## 1. Competing Open-System Models (Non-Hierarchical Alternatives)

Consider an active qubit pair $(u, v)$ and neighboring spectator qubit(s) $s$ evolving over duration $T = 8.192\,\mu\text{s}$ (`2,048 dt` at `dt = 4.0 ns`) under periodic dynamical decoupling (DD) toggling waveforms $s_j(t) \in \{+1, -1\}$ ($j \in \{u, v, s\}$) with $N$ $\pi$-pulses, slot spacing $\tau_{\text{slot}} = T / N$ (`512.0 ns = 128 dt` at $N = 16$), and modulo-wrapped sublattice stagger offset $\Delta \in [0, \tau_{\text{slot}}/2]$ (`[0, 256] ns = [0, 64] dt`). Define the normalized toggling-frame exposure integrals:
$$\alpha_j = \frac{1}{T}\int_0^T s_j(t)\,dt = 0 \quad (N \text{ even}), \qquad \alpha_{ab}(\Delta) = \frac{1}{T}\int_0^T s_a(t)\,s_b(t)\,dt = 1 - \frac{2\Delta}{\tau_{\text{slot}}} \quad (\Delta \in [0, \tau_{\text{slot}}/2])$$
so $\alpha_{uv}(0) = +1$ (synchronous DD) and $\alpha_{uv}(\tau_{\text{slot}}/2) = 0$ (bipartite half-slot stagger; cf. heavy-hex staggered DD in Tripathi et al., *Phys. Rev. Applied* 18, 024068, 2022, and 2-qubit CPMG noise spectroscopy in Paz-Silva et al., *PRA* 95, 022121, 2017; Sung et al., *Nature Commun.* 12, 967, 2021).

### Model `H0` — Independent Local Markovian Lindblad Noise (`5` parameters)
$$\dot{\rho} = \sum_{j \in \{u,v,s\}} \left( \frac{1}{T_{1,j}} \mathcal{D}[\sigma_-^{(j)}](\rho) + \frac{\gamma_{\phi,j}}{2} \mathcal{D}[Z_j](\rho) \right)$$
- Pure single-qubit amplitude damping + white-noise dephasing; zero spatial covariance and zero bath memory. Under balanced periodic DD ($N \ge 1$), $|0\rangle \leftrightarrow |1\rangle$ inversion symmetrizes longitudinal $T_1$ parity decay equally across $|\Phi^+\rangle$ and $|\Psi^+\rangle$ ($P_{\text{even/odd}} = \frac{1}{2}(1 + e^{-T(1/T_{1,u}+1/T_{1,v})})$), while $d \ge 2$ pairs use $T=0$ SWAP-routed SPAM calibration so routing infidelity is never conflated with spatial decay.

### Model `H1` — Static Spectator / Residual $ZZ$ Hamiltonian (`6` parameters)
$$\dot{\rho} = -i [H_{ZZ}, \rho] + \mathcal{L}_{H0}(\rho), \qquad H_{ZZ} = \frac{\zeta_{uv}}{2} Z_u Z_v + \frac{\zeta_{us}}{2} Z_u Z_s + \frac{\zeta_{vs}}{2} Z_v Z_s \quad (\zeta = 2\pi f_{ZZ})$$
- **Key Physical Subtlety:** On an isolated pair $(u, v)$, $Z_u Z_v$ acts identically on $\{|00\rangle, |11\rangle\}$ ($+1$) and on $\{|01\rangle, |10\rangle\}$ ($-1$), commuting with both $|\Phi^+\rangle$ and $|\Psi^+\rangle$. Static Bell dephasing under `H1` is therefore driven by **spectator $ZZ$ couplings** ($\zeta_{us} Z_u Z_s, \zeta_{vs} Z_v Z_s$).
- On 1Q echo (`ramsey_echo_u`) with unpulsed spectators, balanced DD gives $\alpha_u = 0$ and refocuses static $ZZ$ completely. On 2Q Bell arms with active sublattice-B spectators pulsed alongside $v$, coherence follows the visibility mixture $(1 - p_s) + p_s \cos(2\pi f_{ZZ} T \alpha_{us}(\Delta))$, refocusing symmetrically for **both** $|\Phi^+\rangle$ and $|\Psi^+\rangle$ as $\Delta \to 256\text{ ns}$.

### Model `H2` — Spatially Correlated Markovian Collective Dephasing (`7` parameters)
$$\dot{\rho} = \mathcal{L}_{H0}(\rho) + \frac{1}{2}\sum_{a \neq b} \gamma_{\phi, ab} \left( Z_a \rho Z_b - \frac{1}{2}\{Z_a Z_b, \rho\} \right)$$
with spatial decay $\gamma_{\phi, ab}(d) = \rho_{\text{sp}} \sqrt{\gamma_{\phi,a}\gamma_{\phi,b}} e^{-(d-1)/\xi}$.
- **Multi-Qubit Lattice CPTP Positivity:** Beyond pairwise $|\rho_{\text{sp}}| \le 1$, full multi-qubit complete positivity on a degree-3 heavy-hex star requires Kossakowski matrix positive-semidefiniteness $K \succeq 0$, i.e. $3\rho_{\text{sp}}^2 - 2\rho_{\text{sp}} e^{-1/\xi} - 1 \le 0$ (`verify_heavy_hex_psd_covariance`, `clamp_heavy_hex_psd_rho`).
- **Decoherence-Free Subspace (DFS) Asymmetry:**
  $$\Gamma_{\Phi^\pm}(\Delta) = \gamma_{\phi,u} + \gamma_{\phi,v} + 2\alpha_{uv}(\Delta)\gamma_{\phi,uv}, \qquad \Gamma_{\Psi^\pm}(\Delta) = \gamma_{\phi,u} + \gamma_{\phi,v} - 2\alpha_{uv}(\Delta)\gamma_{\phi,uv}$$
  For $\gamma_{\phi,uv} > 0$, half-slot staggering ($\alpha_{uv}: 1 \to 0$) **improves $|\Phi^+\rangle$ AND degrades $|\Psi^+\rangle$ by equal and opposite rates**, while remaining completely invariant to DD pulse count $N$ at fixed $\alpha_{uv}$.

### Model `H3` — Temporally Non-Markovian Colored / Ornstein–Uhlenbeck Bath (`8` parameters)
Couples qubits to low-frequency Gaussian bath modes $\delta\omega_j(t)$ with spatiotemporal memory kernel:
$$K_{ab}(t_1 - t_2) = C_{ab}\,\sigma_a\sigma_b\, e^{-|t_1 - t_2|/\tau_c} \cos(2\pi f_{\text{bath}} (t_1 - t_2))$$
Coherence decay on $|\Phi^\pm\rangle$ ($+$) and $|\Psi^\pm\rangle$ ($-$) is evaluated via **exact closed-form piecewise analytical integration over the sorted pulse-flip knots** $0 = t_0 < t_1 < \dots < t_M = T$ (`_exact_piecewise_kernel_integrals`), eliminating Riemann-grid quadrature aliasing across sub-slot offsets $\{16, 32, 48, 80, 160, 256\}\text{ ns}$ to machine precision ($\sim 10^{-16}$):
$$\chi_\pm(T, N, \Delta) = \gamma_{w,uv} T + \frac{1}{2}\int_0^T dt_1 \int_0^T dt_2 \Big[ s_u(t_1)s_u(t_2)K_{uu} + s_v(t_1)s_v(t_2)K_{vv} \pm 2 s_u(t_1)s_v(t_2)K_{uv} \Big]$$
- **Scope Note:** `H3` models Gaussian colored/oscillatory baths; discrete two-level-system (TLS) random telegraph noise (RTN) introduces non-Gaussian higher cumulants beyond second-order filter integrals and is flagged via residual kurtosis diagnostics.

### Control Model `Apparatus` — Finite-Width Microwave Drive Collision (`6` parameters)
- Models RF crosstalk during overlapping finite-width single-qubit `X/SX` envelopes (`36.0 ns = 9 dt` on `ibm_marrakesh`; `24.0 ns = 6 dt` on `ibm_fez`): produces a sharp step recovery once $\Delta \ge w_p$ (`48 ns` on both backends, giving $S_{\text{step}} = \frac{F(48\text{ ns})-F(0)}{F(256\text{ ns})-F(0)} \ge 0.75$) and a flat plateau across $\Delta \in [48, 256]\text{ ns}$.

---

## 2. Eight-Axis Observable Discrimination Table

| Observable / Intervention | `H0` (Local Lindblad) | `H1` (Static Spectator $ZZ$) | `H2` (Correlated Markovian) | `H3` (Non-Markovian Memory) | `Apparatus` (`24/36 ns` Collision) |
|---|---|---|---|---|---|
| **1. $\|\Phi^+\rangle$ stagger gain ($\Delta: 0 \to 256\text{ ns}$)** | `0` | Positive (`>0`, smooth $\cos(\theta_{ZZ})$) | Positive (`>0`, linear in $\alpha_{uv}$) | Positive when $\tau_c \sim \tau_{\text{slot}}$; filtered when $\tau_c \gg \tau_{\text{slot}}$ | Step jump by $\Delta = 48\text{ ns}$ only |
| **2. $\|\Psi^+\rangle$ DFS basis contrast** | Equal to $\|\Phi^+\rangle$ under balanced DD | **Improves** identically to $\|\Phi^+\rangle$ | **Degrades** under stagger (DFS broken!) | Splits from $\|\Phi^+\rangle$ + $N$-dependent | Improves identically to $\|\Phi^+\rangle$ by $48\text{ ns}$ |
| **3. Ramsey $\to$ Echo $\to$ XY8$\times 4$ pulse density ($N$)** | Invariant ($T_2 = T_2^*$) | Refocuses static 1Q detuning & unpulsed $ZZ$ at $N \ge 1$ | **Invariant** to $N$ (white noise!) | **Strong monotonic gain** for $T/N < \tau_c$ | Degrades with $N$ only when $\Delta < w_p$ |
| **4. Idle-time curve shape $F(T)$** | Pure exponential | Oscillatory $\cos(\zeta T)$ fringes | Pure exponential (split rates) | Gaussian/stretched $e^{-(T/T_2^*)^2}$ at $T \ll \tau_c$ | Pure exponential |
| **5. Spectator activation ($\|0\rangle_s \to \|+\rangle_s$)** | `0` change | **Large drop** at $\Delta=0$, recovered at $256\text{ ns}$ | `0` change | `0` change | `0` change |
| **6. Fine offset sweep $\Delta \in [0, 256]\text{ ns}$ ($S_{48}$)** | Flat (`0`) | Smooth ($S_{48} < 0.35$) | Smooth ($S_{48} = 48/256 = 0.188$) | Smooth $\tau_c$-scale curve ($S_{48} < 0.35$) | **Sharp step ($S_{48} = 1.00 \ge 0.75$)** |
| **7. Graph distance $d(u,v) = 1, 2, 3$ (+ $d \ge 6$ LO control)** | Distance-invariant (after $T=0$ SWAP baseline) | Vanishes for $d \ge 2$ (`<0.1 kHz`) | On-chip decays as $e^{-(d-1)/\xi}$; shared-LO persists at $d \ge 6$ on same board | On-chip decays as $e^{-(d-1)/\xi}$; shared-LO separated by cross-board pairs | Vanishes for non-spectator $d \ge 2$ |
| **8. Held-out coupler transfer** | Local $T_1, T_2$ sufficient | Predicted from independent ZZ-Ramsey $\zeta$ | Predicted from pair correlation length $\xi$ | Predicted from bath $(\tau_c, \xi)$ + local 1Q calibration | Predicted from drive-line adjacency |
