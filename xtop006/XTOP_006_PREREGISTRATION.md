# Preregistration: OSIRIS-XTOP-006 (Quantinuum H2-1 QCCD Design)

**Verified Tunable Non-Clifford Scrambling & `[[4,2,2]]` Codespace Storage Protection on Quantinuum `H2-1`**

- **Status:** Stage-0 exact Kraus verification & Stage-1 `H2-1` Pauli + QCCD memory-dephasing simulation complete (`evidence_class: "SIMULATED"`, `provider_job_id: null`). Zero hardware jobs have been executed.
- **Locked Artifacts:**
  - Manifest (RFC 8785 JCS): `xtop006/artifacts/xtop006_h2_manifest.json` (`SHA-256: 1af448f37e4f0637da84408f5ef5c36094646731b3c81ca16904c0756dda3858`)
  - Compiled 60-Circuit Bundle (RFC 8785 JCS): `xtop006/artifacts/xtop006_h2_1_compiled_circuits.json` (`SHA-256: ce8f2d1f51c970862763ef79a651b5c412fa5900ebed7e9ebddfd50851f95d75`)
  - Compiled Batch Digest: `855af6d68f2964d8392d9fd0819a02358faeeebdcb1c2468bc81b92890cb00f3`

---

## 1. Scientific Motivation & Problem Statement

OSIRIS-XTOP-005 established `[[4,2,2]]` error-detected Hayden–Preskill recovery on `H2-1` using purely **Clifford** transversal scramblers ($U_L$ built from transversal `CX`, `H^{\otimes 4}`, `S/Sdg`, and slot permutations), because non-Clifford gates are not transversal on `[[4,2,2]]`. However, genuine chaotic black-hole / many-body operator spreading requires **continuous-angle non-Clifford dynamics** and must separate coherent unitary operator growth from environmental decoherence, which mimics out-of-time-ordered correlator (OTOC) decay.

OSIRIS-XTOP-006 resolves both limitations on Quantinuum `H2-1` all-to-all QCCD hardware:

1. **Analytical Remnant Kraus Channel (`xtop006/kraus_channel.py`):**
   For any 3-qubit unitary $U(\theta) \in U(8)$ acting on $(q_0, q_1, q_2)$ and its entrywise complex conjugate $U^*(\theta)$ acting on $(q_3, q_5, q_4)$, with initial EPR pairs on $(q_1, q_5)$, $(q_2, q_4)$, and $(q_3, q_{\text{out}})$, Bell postselection of $(q_2, q_4)$ onto $|\Phi^+\rangle$ induces the 16 remnant $2 \times 2$ Kraus operators indexed by $r = (i, j, m, n) \in \{0, 1\}^4$:
   $$(M_r(\theta))_{x a} = \frac{1}{4}\sum_{b, c, k \in \{0, 1\}} U(\theta)_{i j k, a b c}\, U^*(\theta)_{n m k, x b c}, \qquad \mathcal{E}_{\Phi^+}(\rho_{\text{in}}) = \sum_{r \in \{0,1\}^4} M_r(\theta)\, \rho_{\text{in}}\, M_r(\theta)^\dagger.$$
   Across all four Bell outcomes $\beta \in \{\Phi^+, \Phi^-, \Psi^+, \Psi^-\}$, the 64 operators satisfy exact CPTP trace completeness $\sum_{\beta, r} M_{\beta, r}^\dagger M_{\beta, r} = I_2$.

2. **Explicit 1-Parameter Tunable Non-Clifford Scrambler $U(\theta)$:**
   $$U(\theta) = \widetilde{\mathrm{CX}}_{0 \to 2}(\theta)\, \widetilde{\mathrm{CX}}_{0 \to 1}(\theta)\, H^{\otimes 3}\, \widetilde{\mathrm{CX}}_{0 \to 2}(\theta)\, \widetilde{\mathrm{CX}}_{0 \to 1}(\theta)\, H^{\otimes 3},$$
   where $\widetilde{\mathrm{CX}}_{c \to t}(\theta) = (I_c \otimes H_t)\, \mathrm{diag}(1, 1, 1, e^{i 2\theta})\, (I_c \otimes H_t)$ compiles directly to a **single native `H2-1` `ZZPhase(-\theta/\pi)` gate** plus virtual `Rz(\theta/\pi)` and single-qubit `PhasedX` gates (`0` routing SWAPs).
   - At $\theta = 0$: $U(0) = I_8$ (zero scrambling; $P_{\Phi^+}(0) = 1$, $M_r(0) \not\propto I_2$, $\overline{F}(0|\Phi^+) = 0.500000$).
   - At $\theta \in \{\pi/8, \pi/4, 3\pi/8\}$: non-Clifford regime with strictly positive stabilizer Rényi-2 magic ($M_2 = 2.501, 4.000, 3.479$ bits) and exact analytical Yoshida–Kitaev trajectory:
     $$P_{\Phi^+}(\theta) = \left(1 - \frac{1}{2}\sin^2\theta\right)^2, \qquad F_{\text{ent}}(\theta|\Phi^+) = \frac{1}{4 P_{\Phi^+}(\theta)}, \qquad \overline{F}_{6\text{-state}}(\theta|\Phi^+) = \frac{1}{3} + \frac{1}{6 P_{\Phi^+}(\theta)}.$$
   - At $\theta = \pi/2$: maximal 3-qubit scrambler ($X_0 \mapsto X_0 X_1 X_2$, $Z_0 \mapsto Z_0 Z_1 Z_2$, $Y_0 \mapsto -Y_0 Y_1 Y_2$), achieving exact remnant decoupling $M_r(\pi/2) = c_r I_2$ ($\max_r \|M_r - c_r I_2\|_F < 10^{-16}$), $P_{\Phi^+}(\pi/2) = 0.250000$, and $\overline{F}(\pi/2|\Phi^+) = 1.000000$.

3. **Landsman–Figgatt Scrambling-vs-Decoherence Discriminator:**
   Pure decoherence without scrambling ($\theta = 0$, depolarizing rate $p \in [0, 0.75]$) drives $P_{\Phi^+}$ down from $1.000 \to 0.250$—indistinguishable from OTOC decay alone—yet keeps $\overline{F}(0, p|\Phi^+) = 0.500000$ pinned at the random-guess floor ($P_{\Phi^+} F_{\text{ent}} \to 0.0625$). Coherent scrambling drives $P_{\Phi^+} \to 0.250$ while raising $\overline{F}_{|\Phi^+} \to 0.9675$ under realistic noise ($\Delta F_{\text{verified}} = +0.4675$ above the decoherence ceiling).

---

## 2. Experimental Arms & QCCD Architecture (`xtop006/qccd_circuits.py`)

Both arms sweep $\theta \in \{0, \pi/8, \pi/4, 3\pi/8, \pi/2\}$ and the 6 Pauli eigenstates $|\psi\rangle \in \{|0\rangle, |1\rangle, |+\rangle, |-\rangle, |+i\rangle, |-i\rangle\}$ (`30` circuits/arm, `60` circuits total, compiled offline for `H2-1` with `optimisation_level=2`, `0` routing SWAPs):

| Arm | Label | Qubits | Architecture | Total $N_{1q}$ (phys / `Rz`) | Total $N_{2q}$ | Total $N_m$ | HQC @ 500 shots (cons / phys) |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| `P0` | `PHYSICAL_7Q` | 7 | Bare 7Q Landsman–Figgatt circuit; output $q_6$ idles unprotected in storage zone | 1,032 (711 / 321) | 312 | 300 | `715.2` / `683.1` |
| `E1` | `ENCODED_422_HYBRID` | 12 | Output + $|0\rangle_{L_2}$ reference co-encoded in `[[4,2,2]]` ($q_6..q_9$), verified prep ($q_{10}$) + flagged weight-4 $X/Z$ checks ($q_{10}, q_{11}$) | 1,922 (1,541 / 381) | 852 | 840 | `1,614.2` / `1,576.1` |
| **Total** | **All 60 circuits** | **7 / 12** | **Offline `H2-1` compiled (`0` SWAPs)** | **2,954 (2,252 / 702)** | **1,164** | **1,140** | **`2,330` (`2,329.4` / `2,259.2`)** |

---

## 3. Stage-0 & Stage-1 Quantitative Predictions (`SIMULATED`)

Noise parameters: `H2-1` published gate/SPAM magnitudes ($p_{1q} = 3\times 10^{-5}$, $p_{2q} = 10^{-3}$, $p_{\text{spam}} = 10^{-3}$) plus QCCD storage-zone memory dephasing $p_{\text{mem}} = 0.045$ ($70\%\,Z + 15\%\,X + 15\%\,Y$ on idling storage qubits during the scrambling/shuttling window), 4,000 shots/circuit, seed `20261123`:

| $\theta$ | Non-Clifford ($M_2$ bits) | Analytical $P_{\Phi^+}$ | Analytical $\overline{F}_{|\Phi^+}$ | `P0` (`PHYSICAL_7Q`) $\overline{F}_{|\Phi^+}$ | `E1` (`ENCODED_422`) $\overline{F}_{|\Phi^+}$ | `E1` $p_{\text{keep}}$ | Advantage $\Delta F = F_{E1} - F_{P0}$ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| $0$ | False (`0.000`) | `1.000000` | `0.500000` | `0.503762` | `0.497299` | `0.8136` | `-0.006463` |
| $\pi/8$ | **True** (`2.501`) | `0.858915` | `0.527377` | `0.526786` | `0.523815` | `0.8143` | `-0.002971` |
| $\pi/4$ | **True** (`4.000`) | `0.562500` | `0.629630` | `0.617719` | `0.622513` | `0.8143` | **`+0.004794`** |
| $3\pi/8$ | **True** (`3.479`) | `0.328585` | `0.840559` | `0.822971` | `0.832529` | `0.8138` | **`+0.009558`** |
| $\pi/2$ | False (`0.000`) | `0.250000` | `1.000000` | `0.960571` | **`0.986758`** | `0.8137` | **`+0.026187`** |

---

## 4. Preregistered Hypotheses & Governance Gate

1. **H1 (Verified Non-Clifford Scrambling):** At $\theta \in \{\pi/4, 3\pi/8, \pi/2\}$, both `P0` and `E1` exhibit $P_{\Phi^+}(\theta) < P_{\Phi^+}(0)$ jointly with $\overline{F}(\theta|\Phi^+) > 0.50$ ($\alpha = 0.01$, one-sided bootstrap), falsifying pure-decoherence OTOC decay.
2. **H2 (`[[4,2,2]]` QCCD Storage Protection):** Under extended QCCD storage-zone idling, error-detected `E1` achieves strictly higher post-selected recovery fidelity than bare `P0` at $\theta = \pi/2$ ($F_{E1}(\pi/2|\Phi^+) - F_{P0}(\pi/2|\Phi^+) > 0$, Stage-1 prediction `+2.62%`, reducing total infidelity by `66%`).
3. **Governance Rule:** All artifacts in this repository carry `evidence_class: "SIMULATED"` and `provider_job_id: null`. Execution on `H2-1E` or `H2-1` requires explicit operator approval and cryptographic Tier 3 token verification via `osiris nexus`.
