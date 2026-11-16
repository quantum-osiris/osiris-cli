# Mathematical & Statistical Erratum: 27-Qubit $W_2$ Robustness Manuscript (`w2_robustness_paper.tex`)

**Status:** `Implemented and tested` (analytical proofs + Qiskit/LP verification in `phase2/aeterna_w2/w2_circuit_reanalysis.py` & `tests/test_phase2_aeterna_w2.py`; raw hardware count JSONs for jobs `d554h...` not present in workspace)  
**Target Manuscript:** `/home/osiris/w2_robustness_paper.tex` (`sha256: b1386f271cb11ef821b14927ee499ec2b2c0f277b9be9458d0e427677c81b403`)  
**Related Ledger Claim:** `NB-11` (`Outcome diversity vs W2 robustness`) / `W2_ROBUSTNESS` (`osiris_cli/claims.py`)  
**Governing Decision:** **Quarantine physical/robustness conclusions (`QUARANTINED_PENDING_RECOMPUTATION`) and withhold journal submission until replaced by a debiased optimal-transport benchmark.**

---

## 1. Summary of Four Confirmed Mathematical Errors

| # | Location in `w2_robustness_paper.tex` | Original Claim / Equation | Mathematical Reality | Impact on Conclusions |
|---|---|---|---|---|
| **1** | Appendix A, Eq. (7) (L455–466) | $W_2^2(p,q) \approx \sum_{b,b'} \mathrm{cost}(b,b') p(b) q(b')$ *"equals exact $W_2^2$ when $p=q$ (giving zero, as desired)"* | Computes independent-sampling expectation $C(p,q) = \mathbb{E}_{B\sim p, B'\sim q}[(d_H(B,B')/Q)^2]$, **not** optimal transport $\inf_{\pi \in \Pi(p,q)}$. Violates $W_2(p,p)=0$: for uniform $p=q$ on $Q=27$ qubits, $\sqrt{C(p,p)} = \sqrt{\frac{1}{4} + \frac{1}{108}} = \mathbf{0.50918}$ ($\hat{R} = \mathbf{0.4908}$). | The reported $\theta_3 = 3.118$ value ($\text{med. } W_2 = 0.506, \hat{R} = 0.494$) equals the **self-dispersion of an identical uniform distribution** to within $0.003$. |
| **2** | Sec. II.B Eqs. (3)–(6) & Sec. III.C (L270–275), Sec. IV (L290–295) | Layer-0 single-qubit state predicts $P(|1\rangle) = \sin^2(\theta_1/2) = 0.760 \implies 0.760^{27} \approx 6\times 10^{-4}$, so observed $P = 0.0398$ proves *"scrambling layers generate constructive interference toward this attractor"*; at $\theta^\star = 2.618$, $P(|1\rangle) = 0.933$ *"near the equator"*. | Author **omitted the Layer-0 Hadamard gate $H_i$** in Eq. (3) when computing $0.760$ and $0.933$. Including $H_i$ gives $p_{\text{dom}}(\theta) = \frac{1 + \sin\theta}{2} = \sin^2(\theta/2 + \pi/4)$, yielding **$0.9270^{27} = 0.1291$ (`12.91%`)** at $\theta_1 = 2.118$, **$0.7500^{27} = 4.23\times 10^{-4}$** at $\theta^\star = 2.618$, and **$0.5118^{27} = 1.40\times 10^{-8}$** (the true equator!) at $\theta_3 = 3.118$. | Concentration at $\theta_1$ is **entirely prepared in Layer 0**; hardware $P = 0.0398$ is **3.25× attenuated** by NISQ gate/readout error relative to the $12.91\%$ ideal peak, **not** enhanced by constructive interference. |
| **3** | Sec. II.B Eqs. (4)–(6) (L147–156) | `CP(φ²)` (Layer 1) and `CX` + `RZ` (Layer 2) create *"phase relationships"* and *"constructive / destructive interference"* in computational-basis outcomes. | `CP(φ²)` is diagonal in the Z basis, `CX` is an invertible $\mathrm{GF}(2)$ basis permutation $\Pi$, and `RZ` is diagonal immediately preceding Z readout: $|\langle y | D_{RZ} U_{CX} D_{CP} |\psi_0\rangle|^2 = |\langle \Pi^{-1}(y) | \psi_0\rangle|^2$. Verified to **$4.44\times 10^{-16}$** against Qiskit `Statevector`. | **Zero quantum phase interference occurs after Layer 0.** Neither `CP(φ²)` nor `RZ` alters any ideal computational-basis probability. |
| **4** | Abstract (L45) & Sec. III.C (L261–264) | $P = 0.0398$ (`326 / 8,192` shots) represents a **"112× enhancement over the uniform-random baseline"** on 27 qubits. | True 27-qubit uniform probability is $2^{-27} = \mathbf{7.45058\times 10^{-9}}$ ($0.0398 / 2^{-27} = 5.34\times 10^6\times$). The number `112×` came from dividing $326/8192$ by $1/\overline{N_{\text{unique}}} = 1/2818.5$ (`0.039795 × 2818.5 = 112.16×`), conflating empirical sample support size with $2^{27}$. | Reported baseline factor is off by $4.76\times 10^4\times$ due to substituting observed unique-shot count for $2^{27}$. |

---

## 2. Exact Optimal-Transport Reanalysis & Why the Physical Interpretation Cannot Be Retained

Because raw shot-count JSON files for the six December 23, 2025 `ibm_fez` jobs (`d554h0rh...`–`d554hbhs...`) are not present in the local workspace, historical hardware scores are **not** fabricated or guessed; hardware observations remain classified `HARDWARE-OBSERVED / OWNER-REPORTED` while the $W_2$ conclusion is `QUARANTINED_PENDING_RECOMPUTATION`.

Using `phase2/aeterna_w2/w2_circuit_reanalysis.py` (`docs/phase2/w2_27q_erratum_reanalysis.json`), we evaluated **both** the defective marginal-product formula (Eq. 7) and **exact Kantorovich Linear Programming $W_2$** (`osiris_cli.metrics.exact_wasserstein_lp`) on finite-shot multinomial replicates drawn from the **exact noiseless 27-qubit circuit** ($\epsilon_{\text{noise}} = 0$, $S = 512$ shots/replicate):

| $\theta$ (rad) | Ideal Mode $p_{\text{dom}}^{27}$ | Exact LP $W_2(\hat{p}_1, \hat{p}_1)$ (Self) | Defective Eq. 7 $\sqrt{C(\hat{p}_1, \hat{p}_1)}$ (Self) | Noiseless Inter-Rep Exact LP $W_2(\hat{p}_1, \hat{p}_2)$ | Noiseless Exact $\hat{R}_{\text{true}}$ | Noiseless Defective $\hat{R}_{\text{Eq.7}}$ | Paper Hardware $\hat{R}_{\text{Eq.7}}$ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `2.118` | `0.129135` | `0.000000` | `0.310784` | `0.079970` | `0.920030` | `0.683829` | `0.754` |
| `2.618` ($\varphi^2$) | `0.000423` | `0.000000` | `0.472195` | `0.186195` | `0.813805` | `0.525341` | `0.587` |
| `3.118` | `1.40e-08` | `0.000000` | `0.508550` | `0.229126` | `0.770874` | `0.490901` | `0.494` |

### Critical Statistical Lesson (Why Corrected Numerical Ordering Does NOT Rescue the Mechanism)
Even when Eq. (7) is replaced by **exact Kantorovich LP $W_2$**, and even on an **ideal noiseless quantum computer** ($\epsilon_{\text{noise}} = 0$, where true population $W_2(p_\theta, p_\theta) = 0$ and true $\hat{R}(\theta) = 1$ identically for *every* $\theta$), two finite-shot multinomial draws $\hat{p}_1, \hat{p}_2$ exhibit $\hat{R}_{\text{true}} = 0.920 \to 0.814 \to 0.771$!

**Reason:** On a $Q=27$ hypercube ($d = 27$), empirical Wasserstein distance between two $S$-shot samples from the *same* high-entropy distribution converges at the slow minimax rate $O(S^{-1/Q})$ (Weed & Bach, 2019): two $4{,}096$-shot draws from a uniform distribution over $2^{27} = 1.34\times 10^8$ states share virtually zero bitstrings and must transport mass across typical nearest-neighbor Hamming distances $\sim 5\text{–}7$ bits ($\hat{W}_2 \approx 0.20\text{–}0.25$) purely from **finite-sample multinomial sparsity**, with zero hardware noise! Any future hardware-robustness estimator MUST subtract a **split-sample within-job multinomial null** (or use debiased two-sample Sinkhorn / MMD divergence) so finite-shot entropy bias is never misattributed to NISQ decoherence.
