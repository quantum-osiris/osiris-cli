# Pre-Registration: `OSIRIS-XTOP-006-FEZ` — Constant-Skeleton 7-Qubit Tunable Non-Clifford Hayden–Preskill Kraus Trajectory & Decoherence Discriminator on `ibm_fez`

- **Protocol ID:** `OSIRIS-XTOP-006-FEZ`
- **Target Processor:** `ibm_fez` (156Q IBM Heron r2, `open-instance` via governed `troykey.json`)
- **Pinned 7-Transmon Heavy-Hex T-Junction `[q0..q6]`:** `[99, 115, 95, 93, 94, 92, 79]`
  - Active 2Q edges (`28 CZ` total, `0` routing SWAPs during $U(\theta)\otimes U^*(\theta)$):
    `(115,99)` [5 CZ], `(99,95)` [6 CZ], `(95,94)` [4 CZ], `(94,93)` [6 CZ], `(93,92)` [6 CZ], `(93,79)` [1 CZ].
  - Measured qubits (`c[0]=q2(95), c[1]=q4(94), c[2]=q6(79)`): all $< 0.9\%$ listed readout error.
- **Zero-Confound Constant-Pulse-Duration Invariant:**
  Every circuit in `COHERENT`, `DECOH_CTRL`, and `AA_SENTINEL` compiles every inter-`CZ` 1Q unitary via the exact universal 2-`SX` decomposition $U \equiv R_z(\phi)\sqrt{X}R_z(\pi-\vartheta)\sqrt{X}R_z(\lambda+\pi)$, locking **exact `28 CZ` and constant `SX` count** across all $\theta \in \{0, \pi/8, \pi/4, 3\pi/8, \pi/2\}$. Only zero-duration virtual `Rz` angles vary with $\theta$.

## Preregistered Hypotheses & Kill Rules

1. **H1 — Continuous Non-Clifford Kraus Recovery Climb (`COHERENT`):**
   - Theoretical noiseless law: $P_{\Phi^+}(\theta) = (1 - \frac{1}{2}\sin^2\theta)^2$ (`1.000 -> 0.250`) and $F_{\text{6-state}}(\theta \mid \Phi^+) = \frac{1}{3} + \frac{1}{6 P_{\Phi^+}(\theta)}$ (`0.500 -> 1.000`, non-Clifford peak magic $M_2 = 4.000$ bits at $\theta = \pi/4$).
   - **Pass Criterion:** Raw post-selected 6-state fidelity $F_{\text{raw}}(\pi/2 \mid \Phi^+) > F_{\text{raw}}(0 \mid \Phi^+) + 0.20$ (`+20 pp`) with paired block-$t$ Holm-adjusted $p < 0.01$, and classical teleportation threshold $F_{\text{raw}}(\pi/2 \mid \Phi^+) > 2/3$ (`0.6667`) at 95% lower confidence limit.
   - **Kill Rule:** Refuted if $F_{\text{raw}}(\pi/2 \mid \Phi^+) \le 2/3$ or fails monotonicity across $\{0, \pi/4, \pi/2\}$.

2. **H2 — Landsman–Figgatt Scrambling-vs-Decoherence Discriminator (`COHERENT` vs `DECOH_CTRL`):**
   - `DECOH_CTRL` applies deterministic mid-scrambler Pauli-$Z$ phase twirls $(Z^{s_0}\otimes Z^{s_1}\otimes Z^{s_2})$ across the 4 blocks (`(1,1,0), (1,0,1), (0,1,1), (1,1,1)`) using zero extra `CZ` or `SX` gates, destroying off-diagonal $X/Y$ operator coherence (bounding noiseless `DECOH_CTRL` at the classical $2/3 = 0.6667$ ceiling at $\theta=\pi/2$ and $0.6427$ at $\theta=3\pi/8$) while preserving the exact `28-CZ`, `126-SX` pulse schedule.
   - **Pass Criterion:** $\Delta F_{\text{scram-decoh}}(\pi/2) > +0.15$ (`+15 pp`) and $\Delta F_{\text{scram-decoh}}(3\pi/8) > +0.10$ (`+10 pp`), both with paired block-$t$ Holm-adjusted $p < 0.01$.

3. **H3 — Yoshida–Kitaev Branching-Product Conservation Test:**
   - Report both raw and $8\times 8$ readout-mitigated branching product $\Pi(\theta) = P_{\Phi^+}(\theta) F_{\text{ent}}(\theta \mid \Phi^+)$ against the unitary ceiling $1/4 = 0.250$, quantifying exact hardware attenuation across constant-duration $\theta$.
