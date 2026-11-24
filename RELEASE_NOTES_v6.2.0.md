# OSIRIS v6.2.0 Release Notes

**Release Date:** 2026-11-23  
**Tag:** `v6.2.0`  

OSIRIS `v6.2.0` consolidates all active feature, governance, quantum-decoding, live `ibm_fez` / `ibm_marrakesh` / `ibm_kingston` hardware-verification, and macro-locked reproducibility-capsule branches onto `main` with **zero file deletions (`0` removed paths vs `origin/main`) and zero capability regressions**, alongside **6 newly minted Zenodo preprints and data archives**.

---

## Newly Published Zenodo Preprints & Reproducibility Archives (`v6.2.0`)

| Slug | Title | Version | Zenodo DOI | Record URL |
| :--- | :--- | :---: | :--- | :--- |
| `heron_r2_unified` | *Calibration Staleness, Transient Readout Telegraphing, and Colored-Noise False Positives Across Three 156-Qubit Superconducting Quantum Processors* (42 QPU jobs, 1,851,392 shots) | `1.0.0` | `10.5281/zenodo.24486928` | https://zenodo.org/records/24486928 |
| `wp2_27q_ot` | *Eliminating Finite-Shot Hypercube Sparsity Bias in 27-Qubit Quantum Processor Stability Benchmarks via Split-Sample-Debiased Optimal Transport (`OSIRIS-WP2-27Q-FEZ`)* | `1.0.0` | `10.5281/zenodo.24486931` | https://zenodo.org/records/24486931 |
| `nonabelian_2bga_adv1` | *Left-Right Rank Duality in Non-Abelian Two-Block Group-Algebra Quantum LDPC Codes and a Three-Attack Classical-Hardness Pre-Gate for Looped Non-Mirror Peaked Circuits* | `1.0.0` | `10.5281/zenodo.24486938` | https://zenodo.org/records/24486938 |
| `xtop006_fez` | *Continuous-Angle Non-Clifford Hayden–Preskill Scrambling Trajectory and Stratified Decoherence Discrimination on a 156-Qubit Superconducting Processor (`OSIRIS-XTOP-006` & `XTOP-006B`)* | `1.0.0` | `10.5281/zenodo.24486942` | https://zenodo.org/records/24486942 |
| `hp004_v2_1` | *Dual-Processor Certification of Schmidt Number >= 3 and >1-Ebit Entanglement of Formation in Hayden–Preskill Recovery via Layout-Locked Nine-Basis Stabilizer Tomography on Heavy-Hex Superconducting Processors (`OSIRIS-HP-004` & `HP-004B`, v2.1)* (supersedes `10.5281/zenodo.24287505`) | `2.1.0` | `10.5281/zenodo.24486953` | https://zenodo.org/records/24486953 |
| `qdecode_heron_59` | *Reference-Verified Wire Deserialization, Little-Endian Bit-Order Disambiguation, and Finite-Shot Entropy Saturation Across a 59-Job (2–156 Qubit) Superconducting Quantum Corpus* | `1.0.0` | `10.5281/zenodo.24486962` | https://zenodo.org/records/24486962 |

---

## Key Capabilities & Improvements

### 1. IBM Executor `v0.2` Decoding, Qiskit Reference Parity & Fail-Closed Provenance (`osiris_cli.qdecode` v1.1.0, `osiris_cli.qscore`)
- **Unified pure-NumPy wire decoder (`osiris_cli/qdecode.py` v1.1.0):** Decodes both SamplerV2 `PrimitiveResult` (`.npy`-header-safe `BitArray` streams, eliminating the `128`-byte `np.frombuffer` shift with median `TVD = 0.0664` across `575` slices) and IBM Executor `QuantumProgramResult` (`schema_version` `v0.1`–`v2.0`, little-endian bit unpack with reversed Qiskit bit axis `matrix[..., ::-1]`, eliminating `0.0231`–`1.0000` unreversed bit-order TVD errors across `8` Heron `100,000`-shot jobs, plus `.zip` archive support).
- **Live dual-path Qiskit parity:** Automatically cross-checks decoded counts against Qiskit's `RuntimeDecoder` and `ExecutorResultDecoder` (`BitArray.from_bool_array(..., order="little")`), achieving `0.0000` TVD across all `59 / 59` archival jobs (`583` register slices) and `6` permanent `tests/fixtures/executor_v02_*.json` fixtures.
- **Finite-shot entropy scoring (`osiris_cli/qscore.py`):** Enforces $H_{\max} = \min(n, \log_2 N)$, Miller–Madow bias correction, plug-in delta-method standard errors, and circuit-referenced TVD / KL divergences, quarantining `24` unmeasured `ALL_ZERO` classical storage registers (`2`–`84` bits) and classifying `167` `SATURATED_ENTROPY` registers (`ratio >= 0.98`, including `127Q` `4,096`-shot runs at `12.0000 / 12.0000` bits = `100.00%` and `156Q` `ibm_fez` job `d77uq7jc6das739h8k50` at `16.505144 / 16.609640` bits = `99.37%`).
- **Fail-closed Phase 1 hardware ingestion (`quantum_discovery/phase1_analysis/`):** Enforces `SKIP_DIRS` (`venv/`, `fake_provider/`, `tests/`), removes synthetic fallback count generation, write-protects quarantined historical artifacts, and exports deterministic `HARDWARE_PROVENANCE_MANIFEST.json` records with SHA-256 payload digests.

### 2. Governed Execution Kernel & Deterministic Interpreter Alias Resolution (`fe/exec-alias-resolution`)
- Deterministic interpreter alias resolution (`python` / `python3` / venv binary mapping) across `osiris_cli/exec_policy.py`, `osiris_cli/policy/exec_policy.json`, and `osiris_cli/code_agent.py`.

### 3. 9-Dimension Evolution Engine & Multi-Model Department Orchestration (`feat/osiris-evolution-engine`)
- Live wiring of the 9-dimension OSIRIS Evolution Monitor (`osiris_cli/evolution_engine.py`) and multi-model virtual department, quantum-protocol, rigor, RFC 8785 JCS canonicalization, and blocker engines (`osiris_cli/orchestration_departments.py`) into `code_agent.py`, `knowledge.py`, `living.py`, `node_router.py`, `plan_gates.py`, `quantinuum_gate.py`, and `train.py`, with portable `Path.home()` resolution across `osiris_cli/ibm_credentials.py` and `osiris_cli/orchestration_departments.py`.

### 4. Quantum Error Correction, Debiased Optimal Transport & Heron r2 Reproducibility Packages
- **XTOP-006 / XTOP-006b & Quantinuum H2-1E Staging (`xtop006/`, `xtop005/nexus_h2_bundle/`, `experiments/xtop006_fez/`, `experiments/xtop006b_fez/`):** Tunable non-Clifford Kraus scrambling, 18-circuit H2 decisive TKET staging bundle, live `ibm_fez` 4-twirl discriminator runs (`dc1tqaanj2vs73dfrt40`, `dc23ouqnj2vs73dg3bog`), and governed `osiris_cli/ibm_credentials.py`.
- **WP2 Debiased Sinkhorn / Exact LP Optimal Transport & Live 27Q `ibm_fez` Benchmark (`experiments/wp2_debiased_sinkhorn/`, `experiments/wp2_27q_fez/`):** Sample-size-matched split-sample U-statistic estimators and live 27-qubit `ibm_fez` hardware execution archive (`dc1tqvinj2vs73dfrts0`).
- **Non-Abelian Odd-$k$ 2BGA qLDPC & ADV-1 Looped Peaked-Circuit Pre-Gate (`qldpc_nonabelian/`, `adv1/`):** Frobenius left-right rank duality proof, 16 odd-$k$ CSS codes (`d >= 4`), BP+OSD-CS benchmarks, and 3-attack classical ablation pre-gate with $M_6 = J_6 - I_6 \pmod 2$.
- **Macro-Locked REVTeX 4.2 Reproducibility Capsules & Standalone PDF 1.4 Preprints (`papers/`):**
  - `papers/heron_r2_calibration_staleness_and_colored_nulls/`
  - `papers/wp2_27q_debiased_ot_fez/`
  - `papers/nonabelian_2bga_and_adv1/`
  - `papers/xtop006_non_clifford_kraus_fez/`
  - `papers/hp004_joint_stabilizer_heron_r2_v2_1/`
  - `papers/qdecode_heron_provenance_and_finite_shot_entropy/`
  - `papers/zenodo_v620_deposits/` (`render_v620_preprints.py` & `publish_v620_papers_to_zenodo.py`)

### 5. Dual-Node Wireless ADB mDNS & Bluetooth PAN Synchronization
- Updated `.agents/skills/osiris-dual-node-sync/SKILL.md` and `scripts/bootstrap_osiris_agents.py` with `_adb-tls-connect._tcp` vs `_adb-tls-pairing._tcp` mDNS discovery, Bluetooth PAN (`bnep0`) fallback, and `--dry-run --itemize-changes` quarantine staging without `--delete`.

---

## Verification Summary
- **Implemented test items (`tests/`):** `3,273` (`3,272` collected + `1` module-level skip)
- **Full unified suite executed:** `3,246` executed (`3,246` passed, `0` failed; `27` runtime skips)
- **Targeted release, paper, decoder & SYS-024 firewall gate:** `247` collected, `247` executed, `247` passed (`0` failed)
- **Branch merge preservation check:** `0` deleted files across `origin/main..v6.2.0` (`git diff --diff-filter=D --name-only origin/main..v6.2.0` empty)
