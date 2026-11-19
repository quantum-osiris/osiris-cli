# NotebookLM Executive Briefing & 308-Row Studio Table — Provenance & Epistemic Audit

- **Document ID**: `KB-SYNTHESIS-2026-11-16-NLM`
- **Source Artifact**: NotebookLM Notebook *"Dual-Substrate ISIS-OSIRIS Mesh Architecture"* (250–316 aggregated sources + Studio Data Table export, reviewed 2026-11-16)
- **Epistemic Standing**: `SECONDARY_SYNTHESIS_UNVERIFIED`
- **Governance Rule (`SYS-024`)**: This synthesis is an index and research input—**not** independent verification of underlying experiments or software. Every claim extracted from it remains `UNVERIFIED` unless corroborated by primary records in `osiris_cli.claims.REGISTER`, SHA-256-verified workspace artifacts, or peer-reviewed DOIs.

---

## 1. Mandatory Corrections & Contradiction Reconciliations

| Briefing / Table Item | Audit Correction & Canonical Standing |
| :--- | :--- |
| **`4.2 × 10⁻¹⁸ W` described as "microwatt-scale"** | **INCORRECT (SI Unit Error — Off by $2.38 \times 10^{11}$)**. $4.2 \times 10^{-18}\text{ W} = \mathbf{4.2\text{ attowatts}}$ ($4.2\text{ aW}$), whereas $1\,\mu\text{W} = 10^{-6}\text{ W}$. Any prose describing $10^{-18}\text{ W}$ signals as microwatt-scale fails dimensional/unit review. |
| **Conflicting rows on $\theta_{\text{lock}} = 51.843^\circ$, $\Lambda_\Phi$, and $\tau_0 = 46\,\mu\text{s}$** | **QUARANTINED / REFUTED IN REGISTER**. NotebookLM's Data Table mixed unquarantined legacy drafts (marking `51.843°` and `82.27% Black Hole Preservation` as "VALIDATED") with later forensic audits. Canonical verdicts in `osiris_cli.claims.REGISTER` govern: `THETA_LOCK` = **REFUTED** (`ibm_fez` Job `dbiobimne4hc73bf2am0` Block C phase-compensated $R_z(\pm\theta)$ tomography: $z = +0.93, p = 0.3537$; $\arctan(14/11)$ is geometric, not a physical resonance); `LAMBDA_PHI` ($2.176435\times 10^{-8}\text{ s}^{-1}$) = **NOT_MEASURED** (Planck mass in kg relabelled as $\text{s}^{-1}$); `TAU_PHASE` ($46\,\mu\text{s}$ revival) = **REFUTED** (classical $T_1$ decay to $|00\rangle$, $C_\perp = +0.0029 \pm 0.0221$); Zenodo `18450505` = **REFUTED / ERRATUM** (quantum Hellinger distance misidentified as $W_2$ with inverted preservation metric). |
| **Blanket "CRSM categorically refuted" phrasing** | **HYPOTHESIS-SCOPED REQUIRED**. Each CRSM result must cite its specific hypothesis, dataset, test, and record: e.g., `NCLM-ARCH-1` (8 paired seeds: standard causal transformer 725,760 params beat CRSM 726,304 params by `+0.595 BPB` on held-out docs and `+0.728 BPB` on held-out chat) vs native autograd memory repair (`osiris/nclm/autograd.py` reducing peak RSS from `11,323.4 MB` to `294.17 MB`). |
| **"49 passing governance tests" assertion** | **HISTORICAL POINT-IN-TIME SNAPSHOT**. Refers to an earlier local REPL v1.3 run; current verification requires live test execution and current commit/file SHA-256 digests (`osiris_governance`, `osiris-cli`, `dnalang-core`, `quantum-ide`). |
| **Optimal T-count synthesis scaling** | **ASYMPTOTIC MATHEMATICAL BOUND ONLY**. Valid under ideal fault-tolerant Clifford+T assumptions (precision $\varepsilon$ and clean ancilla availability); does **not** establish near-term NISQ hardware advantage. |
| **250–316 source synthesis citations (`[cite: N]`)** | **UNTRACED EXPORT INDICES**. Exported NotebookLM indices (`[1]..[55]`, `[cite: ...]`) lack a complete attached bibliography mapping every index to a SHA-256 or DOI; claims citing only NotebookLM indices remain `UNVERIFIED_SYNTHESIS`. |
| **C4 prohibition of `curl`, `wget`, `chmod`** | **OSIRIS-SPECIFIC POLICY CHOICE**. Enforced within governed OSIRIS sandboxes to prevent unmediated egress/mode mutation, not a universal operating-system security law. |

---

## 2. Validated Technical Material & Primary Cross-References

### A. Quantum Circuit Synthesis Literature (Mathematical Bounds)
- **Gosset–Kothari–Wu (*Quantum*, 2026; `arXiv:2411.04790`)**: Proves constant-precision $n$-qubit quantum state preparation requires $\Theta(2^{n/2})$ T-gates when $O(2^{n/2})$ clean ancillae are available, matching information-theoretic lower bounds; precision $\varepsilon$ dependence scales additively/multiplicatively per ancilla regime.
- **Tan (*PRX Quantum*, Sept 2026; `arXiv:2509.25702`)**: Cosine-Sine Decomposition (CSD) $n$-qubit unitary synthesis achieves T-count $O(2^{4n/3} L^{2/3} + 2^n L)$ with $O(2^{2n/3} L^{1/3} + L)$ ancillae for $L = n + \log_2(1/\varepsilon)$, using a linear telescoping error budget $\delta = \varepsilon / 2^{n-k+1}$ so total operator-norm error satisfies $\le (2^{n-k+1}-1)\delta < \varepsilon$.
- **Low–Kliuchnikov–Schaeffer**: Space-time (T-count vs ancilla width) state preparation trade-off $\Theta(2^n / (m + n) + \log_2(1/\varepsilon))$.

### B. Audited Hardware Campaigns & Referee Noise Floors
- **STAG-XY8 Campaign (`ibm_marrakesh` & `ibm_fez`, OSIRIS v5.3.1 Referee Audit)**:
  - *Theory / 3-Engine Simulation*: Duty-cycle-balanced bipartite staggering ($\Delta t = \tau / 2$) bisects heavy-hex spectator $ZZ$ crosstalk ($T_+ = T_- \implies T_{\text{eff}} = 0$, $F_{\text{global}} = 1.0000$).
  - *Experiment E2 (`ibm_marrakesh`, Job `dblv7sm74qds73dr8ho0`, 294,912 shots)*: Fixed 80 ns staggering scored $+0.446 \pm 0.113\text{ pp}$ over synchronous XY8 across 4 held-out couplers; `Adaptive v1.0` trailed Fixed 80 ns by $-0.276\text{ pp}$ (falsified).
  - *Experiment E3 (`ibm_fez`, Job `dbm86158ft8s738ucqi0`, 82,944 shots)*: Cross-chip scalar transfer null ($\Delta \mathcal{F}_{\text{stag}} = +0.109 \pm 0.461\text{ pp}, p = 0.820$; matched-arm adaptive contrast $-0.25\text{ pp}$).
  - *Experiment E4 (`ibm_marrakesh`, Job `dbm9ctu74qds73drs9j0`)*: Pulse-overlap sweep across $\Delta \in \{0, 16, 32, 48, 80, 160\}\text{ ns}$ fit a flat line ($\chi^2 = 1.87, df = 5, p = 0.866$), resolving neither MPCA ($9.5\%$ step recovery vs $\ge 75\%$ pre-registered criterion) nor continuous filter refocusing at $N = 2,048$ shots/point.
  - *Accidental A/A Drift Floor & Power Law*: 7 identical-schedule arm pairs drifted by **$1.03\text{ to }1.11\text{ pp}$ RMS** ($\chi^2 = 11.09, df = 6, p = 0.086$). Inverse-variance pooled fixed-80 ns contrast across all 17 hardware cells: $\Delta \mathcal{F}_{\text{pooled}} = +0.22 \pm 0.18\text{ pp}$ ($Z = 1.19\sigma, p = 0.233$). Resolving a $0.50\text{ pp}$ effect at $2\sigma$ requires **$\ge 33,500$ shots per basis per arm**.
- **Hayden–Preskill Information Recovery (`OSIRIS-HP-001` through `HP-004`)**:
  - *HP-001 Stage 1 (`FakeMarrakesh`, Zenodo `10.5281/zenodo.23986363`)*: Calibrated simulation (`0 QPU s`), $+18.39\text{ pp}$ entangled recovery over separable control ($t_{19} = 8.58, p = 2.94 \times 10^{-8}$); Arm A zero-variance (`0.9694`) traced to missing transpiler barrier before $U^\dagger U$, fixed via explicit `barrier` in HP-002.
  - *HP-002 (`ibm_fez`, Job `dbnchbgqtr0c73acppe0`)*: Barrier-separated restricted 1-qubit recovery $\mathcal{F}_{\text{ent}} = 79.49 \pm 0.14\%$ vs $24.89\%$ separable control.
  - *HP-003 (`ibm_fez`, Job `dbndsvoou4as73b4cvi0`)*: 8-qubit 2-photon capacity $\overline{\mathcal{F}} = 75.49 \pm 0.18\%$ vs $24.86 \pm 0.84\%$ 1-photon sub-capacity.
  - *HP-004 (`HP004_JOINT_STABILIZER`, Jobs `dbti1fbsbsbs73cb2vu0` & `dbtj61bnlcms73ajc380`)*: Raw 9-basis joint Bell overlap `0.633423 ± 0.002634` (`ibm_marrakesh`, witnesses $\text{SN}(\rho) \ge 3$, $E_F \ge 0.8451$ ebits) and `0.730713 ± 0.002732` (`ibm_fez`, certifies $\text{SN}(\rho) \ge 3$ and $E_F \ge 1.1465 \pm 0.0086 > 1$ ebit via Terhal–Vollbrecht isotropic convex roof; exact 1-ebit threshold root `0.683900317` vs printed `0.683910` fifth-decimal typo).
- **qByte Simulator Audit (P-2 Patch)**: Fixed double-swap identity defect in `_apply_toffoli` and inverted bit indexing in `measure_qubit` (`21/21` regression tests passing against `qiskit.quantum_info.Statevector`).

---

## 3. Core Governed Scientific Research Question

> *"Can hardware-aware control and measurement distinguish spatially correlated or temporally non-Markovian decoherence from conventional local noise models—and can that distinction predict better quantum operations on previously unseen qubits?"*

This question is physically grounded, experimentally falsifiable under pre-registration, and scientifically informative under both positive transfer and null drift-bounded outcomes.
