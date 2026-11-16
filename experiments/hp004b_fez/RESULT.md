# OSIRIS-HP-004B Cross-Processor Replication Result (`ibm_fez`, Job `dbtj61bnlcms73ajc380`)

- **Status**: `SUPPORTED (RAW UNMITIGATED CROSS-PROCESSOR REPLICATION, 5/5 PRE-REGISTERED HYPOTHESES PASSED)`
- **Replicates**: `OSIRIS-HP-004` (`ibm_marrakesh`, Job `dbti1fbsbsbs73cb2vu0`, Zenodo DOI `10.5281/zenodo.24285577`)
- **Pre-Registration Commit (GitHub-Timestamped Prior to Job Creation)**: `0c5a5a39` (`staged_at_utc: 2026-11-16T16:01:38.695752+00:00` < IBM `created: 2026-11-16T16:44:53.592489Z`)
- **Processor**: `ibm_fez` (156-qubit IBM Heron r2, `open-instance` — independent cryostat and QPU from `ibm_marrakesh`)
- **Job ID**: `dbtj61bnlcms73ajc380` (`running: 17:06:28.822425Z`, `finished: 17:08:14.754884Z`, `circuits_execution_time_ns: 76,263,417,344`)
- **Frozen QPY SHA-256**: `6d7e5a6d17a002b5551eba35fb99b29661044477a9f6b91fb9dc51376a3025d9`
- **Pinned 8-Transmon Tree Layout (`routing_method="none"`)**: `[2, 24, 4, 3, 16, 23, 22, 21]` (mean internal `CZ` error `0.00179`)
- **Pinned 4-Transmon Readout (`c0..c3`, Identical Across All 286 Circuits)**: `[2, 16, 24, 23]` (`mean 4Q SPAM diagonal = 0.9731`)
- **Total Payload**: `286` circuits (`16` SPAM states + `6` randomized blocks × `5` arms × `9` Pauli bases, `seed=20261117`) × `1,024` shots = **`292,864` shots** (`resilience_level=0`, zero postselection, all `286` raw 4-bit histograms archived in `physical_hp004b_fez_results.json`)
- **QPU Accounting**: `79` QPU charge seconds (`1.308950 RU`) consumed (`160 / 600s` = `2.644492 / 10.0 RU` cumulative across `ibm_marrakesh` + `ibm_fez`); **`440 / 600` seconds (`7.355508 / 10.0 RU`) remaining**.

## Primary Raw Unmitigated Results (`ibm_fez`) & Cross-Processor Comparison (`ibm_marrakesh`)

| Arm | ISA CZs | `ibm_fez` Raw $F_1 \pm \sigma$ | `ibm_fez` Raw $F_2 \pm \sigma$ | **`ibm_fez` Raw 9-Basis $F_{\text{joint}} \pm \text{SE}_{\text{eff}}$** | 95% Shot-Boot CI | $Z_{\text{eff}}$ vs `0.5000` | Secondary SPAM $F_{\text{joint}}$ | `ibm_marrakesh` Raw $F_{\text{joint}}$ | Hypothesis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Arm A` (Barriered Inverse)** | `38` | `0.8174 ± 0.0021` | `0.8422 ± 0.0020` | **`0.7445 ± 0.0023`** | `[0.7400, 0.7491]` | `+105.62σ` | `0.7755` | `0.6591 ± 0.0026` | **H1 PASS** (`> 0.60`) |
| **`Arm B` (2-Photon Restricted)** | **`32`** | `0.8124 ± 0.0021` | `0.8213 ± 0.0021` | **`0.7307 ± 0.0027`** | **`[0.7261, 0.7353]`** | **`+84.45σ`** | `0.7611` | `0.6334 ± 0.0026` | **H2 PASS (`> 0.50` at `84.4σ`)** |
| **`Arm B-Ctrl` ($|0000\rangle_{BE}$)** | `30` | `0.4488 ± 0.0028` | `0.2554 ± 0.0032` | **`0.2053 ± 0.0034`** | `[0.2000, 0.2106]` | `-87.78σ` | `0.2127` | `0.1799 ± 0.0027` | **H3 PASS** (`∈ (0.125, 0.25 + 3 SE]`) |
| **`Arm B-Def` (1-Photon Sub-Cap.)** | `25` | `0.4428 ± 0.0029` | `0.2516 ± 0.0032` | **`0.1140 ± 0.0028`** | `[0.1087, 0.1195]` | `-137.72σ` | `0.1162` | `0.1086 ± 0.0035` | **H4 PASS** (`<= 0.125 + 3 SE`) |
| **`Arm B-Held` (Wrong $U_2$)** | `35` | `0.2438 ± 0.0032` | `0.2511 ± 0.0032` | **`0.0563 ± 0.0028`** | `[0.0510, 0.0614]` | `-158.96σ` | `0.0561` | `0.0634 ± 0.0027` | **H5 PASS** (`∈ [0.04, 0.09]`) |

## Key Scientific Findings

1. **Out-of-the-Box Cross-Processor Replication of Schmidt-Number $\ge 3$ (`> 1 Ebit`) Recovery**:
   On an independent Heron r2 processor (`ibm_fez`), raw unmitigated `Arm B` achieves $F_{\text{joint}} = 0.730713 \pm 0.002732$ across 6 interleaved blocks (`[0.7362, 0.7363, 0.7200, 0.7324, 0.7251, 0.7343]`, 6-block Student-$t_{5,\,0.975}$ 95% CI **`[0.7237, 0.7377]`**), exceeding the exact Terhal–Horodecki (`quant-ph/9911117`) `0.5000` Schmidt-rank-2 ceiling by **`+84.45σ`** (`5σ lower bound = 0.7171 > 0.5000`) while physical horizon remnants (`q4, q21`) and references (`q2, q24`) experienced **zero physical gates** during decoding.
2. **Simultaneous Crossing of Both Joint and Marginal Bounds on Lower-Noise Couplers**:
   Consistent with `ibm_fez`'s lower coupler error (`0.179%` mean `CZ` and `97.31%` 4Q SPAM diagonal vs `0.41%` `CZ` and `93.98%` SPAM diagonal on `ibm_marrakesh`), `Arm B` raw $F_{\text{joint}}$ increased by **`+0.0973`** (`0.6334 -> 0.7307`), and even its conservative 2-qubit marginal union lower bound $F_1 + F_2 - 1 = 0.8124 + 0.8213 - 1 = \mathbf{0.6337}$ crossed `0.5000` (`f_marginal_mean = 0.8169 > 0.7500`, `f_marginal_product = 0.6672 > 0.5625`).
3. **Quantitative Agreement with Pre-Registered Kraus Calibration Model Across All Negative Controls**:
   Pre-registered noisy Kraus predictions captured the physical negative controls within `~0.2%–1.2%` absolute fidelity (`Arm B-Ctrl`: predicted `0.2171` vs observed `0.2053`; `Arm B-Def`: predicted `0.1163` vs observed `0.1140`; `Arm B-Held`: predicted `0.0626` vs observed `0.0563`), replicating the full dyadic capacity ladder ($1 \to 1/4 \to 1/8 \to 1/16$) across two distinct quantum processors.
