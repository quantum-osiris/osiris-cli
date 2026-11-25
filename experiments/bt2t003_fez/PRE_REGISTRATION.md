# Write-Ahead Preregistration — `OSIRIS-BT2T-003-FEZ` ($2T$ Magnetic Plaquette Trajectory & Window-2 Echo Pilot)

- **Protocol ID:** `OSIRIS-BT2T-003-FEZ`
- **Target Backend:** `ibm_fez` (156-qubit IBM Heron r2 processor)
- **Frozen Encoding & Theory Reference:** `bt2t-enc-1` (`sha256 = 3e3b02d1...`) and `experiments/bt2t_plaquette/manifest_bt2t003.json` (`34/34` exact Kogut-Susskind $2T \subset SU(2)$ checks).
- **Candidate Placements (`K = 10` vertex-disjoint `T5` heavy-hex subgraphs, `50` transmons):**
  - `pid=0`: `[21, 22, 23, 16, 24]` (`BT2T-001` baseline), `pid=1`: `[5, 6, 7, 8, 17]` (`ECHO-FEZ` Window-1 top-1),
    `pid=2`: `[9, 10, 11, 12, 18]`, `pid=3`: `[27, 28, 29, 30, 38]`, `pid=4`: `[31, 32, 33, 34, 39]`,
    `pid=5`: `[43, 42, 41, 36, 40]`, `pid=6`: `[25, 37, 45, 44, 46]`, `pid=7`: `[49, 50, 51, 52, 58]`,
    `pid=8`: `[63, 62, 61, 60, 76]`, `pid=9`: `[67, 66, 65, 64, 77]`.

## 1. Physics & Circuit Arms (`240` PUBs = `3` Randomized Blocks x `80` PUBs/Block, `256` Shots/PUB = `61,440` Shots)

### Part A — All `K = 10` Disjoint `T5` Placements (`10 x 6 = 60` PUBs/block = `180` PUBs)
1. `IDENTITY_CTRL_0CZ`: `0-CZ` 5-qubit SPAM return baseline (`|00000>`).
2. `ECHO_PROBE_16CZ`: Non-cancellable `16-CZ` stabilizer-reflected echo probe (`|00000>`).
3. `UB_CENTER_PI4_19CZ`: Full `19-CZ` structured $2T$ magnetic plaquette propagator $U_B(\tau=\pi/4, g_{\text{YM}}^2=1)$ acting on $Q_8$-center superposition $(|+1\rangle + |-1\rangle)/\sqrt{2} = |0000\rangle_{q_4..q_1}\otimes|+\rangle_{q_0}$ ($\Delta\chi_{2D} = 4 \implies \cos^2(2\tau) = 0.0$, exact inversion to $|0000\rangle\otimes|-\rangle_{q_0}$, measured in $Z_{4..1}\otimes X_0$ as target `|00001>` = `1`).
4. `UB_HURWITZ_PI2_19CZ`: Full `19-CZ` structured $2T$ magnetic plaquette propagator $U_B(\tau=\pi/2, g_{\text{YM}}^2=1)$ acting on Hurwitz coset superposition $(|c=1,q=0\rangle + |c=2,q=0\rangle)/\sqrt{2} = \frac{1}{\sqrt{2}}(|01\rangle+|10\rangle)_{q_4,q_3}\otimes|000\rangle_{q_2..q_0}$ ($\Delta\chi_{2D} = 2 \implies \cos^2(\tau) = 0.0$, exact Bell-basis inversion to target `|11000>` = `24`).
5. `CLIFFORD_30CZ`: Held-out `30-CZ` 5-qubit random Clifford survival circuit.
6. `MIRROR_24CZ`: Held-out `24-CZ` non-Clifford stabilizer-twisted peaked circuit (target `|11001>` = `25`).

### Part B — Character-Sector Ramsey Trajectory & Structured-vs-Generic Ablation on `pid=1` (`[5,6,7,8,17]`) vs `pid=0` (`[21,22,23,16,24]`) (`2 x 10 = 20` PUBs/block = `60` PUBs)
On both `pid=1` (Window-1 `ECHO` top-1) and `pid=0` (`BT2T-001` baseline), execute the full `19-CZ` structured $U_B(\tau)$ across:
- **Center sector ($\{+1,-1\}$, $\Delta\chi_{2D}=4$, $P_{|+\rangle}(\tau) = \cos^2(2\tau)$):** $\tau \in \{0, \pi/8, \pi/2\}$ (`UB_CENTER_T0`, `UB_CENTER_PI8`, `UB_CENTER_PI2`; combined with `UB_CENTER_PI4_19CZ` from Part A).
- **Hurwitz sector ($c=1$ vs $c=2$, $\Delta\chi_{2D}=2$, $P_{\Phi^+}(\tau) = \cos^2(\tau)$):** $\tau \in \{0, \pi/8, \pi/4, 3\pi/8\}$ (`UB_HURWITZ_T0`, `UB_HURWITZ_PI8`, `UB_HURWITZ_PI4`, `UB_HURWITZ_3PI8`; combined with `UB_HURWITZ_PI2_19CZ` from Part A).
- **Imaginary-$Q_8$ zero-frequency null control ($\{+i,-i\}$, $\chi_{2D}(\pm i)=0 \implies \Delta\chi_{2D}=0$, ideal $P_{|+\rangle} = 1.0000$):** `UB_IMAGQ8_PI4_19CZ` ($|0001\rangle_{q_4..q_1}\otimes|+\rangle_{q_0}$ evolved under the identical `19-CZ` $U_B(\pi/4)$ circuit).
- **Unused-8 decoupling control ($c=3$, $\Delta\chi=0$, ideal $P_{|+\rangle} = 1.0000$):** `UB_UNUSED8_PI4_19CZ` ($|1100\rangle_{q_4..q_1}\otimes|+\rangle_{q_0}$ evolved under the identical `19-CZ` $U_B(\pi/4)$ circuit).
- **Generic dense unitary synthesis comparator (`UB_CENTER_PI4_GENERIC`):** Dense `32x32` QSD compilation of $\operatorname{diag}(\exp(-i H_B \pi/4))$ with per-gate phase perturbations breaking diagonal commutation, contrasting generic `T5` compilation against structured `19-CZ` Walsh-Hadamard Gray-code synthesis.

## 2. Preregistered Hypotheses
- **`H1` ($2T$ Magnetic Plaquette Character-Frequency & Null-Sector Discrimination):** On `pid=1`, the `19-CZ` structured $U_B(\tau)$ trajectory reproduces the $\Delta\chi_{2D}=4$ (`Center` $\cos^2(2\tau)$: inversion at $\tau=\pi/4$ and revival at $\tau=\pi/2$) and $\Delta\chi_{2D}=2$ (`Hurwitz` $\cos^2(\tau)$: monotone decay to inversion at $\tau=\pi/2$) frequencies with Pearson $r > 0.95$ against exact $SU(2)$ character theory, while the $\Delta\chi_{2D}=0$ Imaginary-$Q_8$ and Unused-8 controls preserve $P_{|+\rangle}(\pi/4) > 0.70$ (`> 50 pp` above the inverted Center sector at the same $\tau=\pi/4$).
- **`H2` (ECHO Top-1 Placement Advantage & Structured Synthesis Compression):** `pid=1` (`[5,6,7,8,17]`) outperforms `pid=0` (`[21,22,23,16,24]`) on average 5-qubit magnetic plaquette & held-out survival across `3/3` blocks, and structured `19-CZ` $U_B(\pi/4)$ outperforms generic dense synthesis.
- **`H3` (Window-2 Co-Primary SPAM-Normalized Echo `ECHO_PROBE_16CZ / IDENTITY_CTRL_0CZ`):** Across all `K = 10` disjoint `T5` placements (`N_windows = 2` combined with `dc35e35s2omc738cvcug`), SPAM-normalized echo `ECHO / ID_0CZ` and/or raw `ECHO` achieves positive Spearman rank correlation $\rho > 0.70$ with held-out composite (`UB_CENTER_PI4_19CZ`, `UB_HURWITZ_PI2_19CZ`, `CLIFFORD_30CZ`, `MIRROR_24CZ`) and top-1 selection regret $< 0.03$.
