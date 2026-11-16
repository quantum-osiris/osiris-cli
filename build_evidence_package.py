import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

repo = Path(__file__).resolve().parent
pkg_dir = repo / "osiris_quantum_hardware_v2.0_evidence_package"
pkg_dir.mkdir(parents=True, exist_ok=True)

# Create/preserve subdirectories without wiping existing anchored ledger artifacts
for sub in ("hp001", "hp002", "qaml001", "stag_xy8", "hp003", "manuscript", "ledger"):
    (pkg_dir / sub).mkdir(parents=True, exist_ok=True)

# 1. HP001
for f in [
    "hp001_phase2_isa_circuits.qpy",
    "job_receipt_hp001_phase2_fez.json",
    "physical_hp001_phase2_results.json",
    "sys021_hp001_phase2_grant.json",
    "audit_subsystem_boundaries.py",
]:
    src = repo / "experiments" / "hp001" / f
    if src.exists():
        shutil.copy2(src, pkg_dir / "hp001" / f)

# 2. HP002 (historical anchored harnesses/results copied verbatim)
for f in [
    "hp002_staged_circuits.qpy",
    "job_receipt_hp002.json",
    "physical_hp002_results.json",
    "sys021_hp002_grant.json",
    "block_44_ledger_receipt.json",
    "stage_hp002_harness.py",
    "harvest_hp002.py",
]:
    src = repo / "experiments" / "hp002" / f
    if src.exists():
        shutil.copy2(src, pkg_dir / "hp002" / f)

# 3. QAML001
for f in [
    "submitted_isa_circuits.qpy",
    "job_receipt_qaml_001_marrakesh.json",
    "physical_qpu_benchmark_results.json",
    "sys021_qaml_001_dispatch_grant.json",
    "block_42_ledger_receipt.json",
    "PRE_REGISTRATION.md",
]:
    src = repo / "experiments" / "qaml_001_adversarial_feature_map" / f
    if src.exists():
        shutil.copy2(src, pkg_dir / "qaml001" / f)

# 4. STAG_XY8
src_stag = repo / "osiris_qpu_stag_xy8_v1.0_evidence_package"
if src_stag.exists():
    for f in [
        "analysis_7edge.json",
        "job_receipts_7edge.json",
        "raw_results_7edge.json",
        "submitted_circuits.qpy",
        "EXPERIMENT_REPORT.md",
    ]:
        p = src_stag / f
        if p.exists():
            shutil.copy2(p, pkg_dir / "stag_xy8" / f)

# 5. HP003 (historical anchored harnesses/results copied verbatim + offline forensic re-audit)
for f in [
    "hp003_staged_circuits.qpy",
    "job_receipt_hp003.json",
    "physical_hp003_results.json",
    "sys021_hp003_grant.json",
    "block_48_ledger_receipt.json",
    "stage_hp003_harness.py",
    "harvest_hp003.py",
    "test_hp003_staged.py",
    "PRE_REGISTRATION.md",
    "mitigate_and_extrapolate_hp003.py",
    "mitigated_hp003_analysis.json",
]:
    src = repo / "experiments" / "hp003" / f
    if src.exists():
        shutil.copy2(src, pkg_dir / "hp003" / f)

# 6. Manuscript
shutil.copy2(
    repo / "papers" / "crsm_nature_physics_2026.tex",
    pkg_dir / "manuscript" / "crsm_nature_physics_2026.tex",
)

# 7. Ledger (preserve packaged ledger if external SQLite DB is not mounted)
db_path = repo.parent / "evidence_ledger.db"
if not db_path.exists():
    db_path = Path(r"C:\Users\Ron\evidence_ledger.db")
if db_path.exists():
    shutil.copy2(db_path, pkg_dir / "ledger" / "evidence_ledger.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT sequence_index, prev_block_hash, block_hash, timestamp, event_type, operator_id, "
        "payload_sha256, operation_schema_sha256, nonce, raw_grant_json "
        "FROM ledger_blocks ORDER BY sequence_index ASC"
    )
    cols = [
        "sequence_index",
        "prev_block_hash",
        "block_hash",
        "timestamp",
        "event_type",
        "operator_id",
        "payload_sha256",
        "operation_schema_sha256",
        "nonce",
        "raw_grant_json",
    ]
    blocks = [dict(zip(cols, r)) for r in cur.fetchall()]
    conn.close()
    with open(pkg_dir / "ledger" / f"ledger_blocks_1_to_{len(blocks)}.jsonl", "w", encoding="utf-8") as out_f:
        for b in blocks:
            out_f.write(json.dumps(b) + "\n")

report_content = """# OSIRIS Quantum Hardware Research Campaign: Unified Evidence Package v2.0 (with v2.1 Forensic Errata)
**Governing Standard:** SYS-021 (Zero Self-Supplied Authorization) / Gate 0 Conformance  
**Lead Investigator:** Devin Phillip Davis / Enki  
**Target Processors:** IBM Quantum Heron r2 (`ibm_fez`) and Heron r1 (`ibm_marrakesh`)  
**Package Generation Date:** November 7, 2026 (Forensic Errata Updated: November 16, 2026)  
**Ledger Anchor:** Blocks #1 through #48 (Cryptographic SHA-256 Hash Chain; Node Beta Permits via 32-Byte Symmetric HMAC-SHA256)  

---

## 1. Executive Summary & Hardware Milestone Determination

This archival evidence package consolidates physical quantum telemetry, compiled native ISA circuits (`.qpy`), dispatch grants, hardware acquisition receipts, and post-audit forensic errata across five hardware benchmarks on IBM Quantum processors. All uncertainties below use the conservative estimator $\\mathrm{SE}_{\\mathrm{eff}} = \\max(\\mathrm{SEM}_{\\mathrm{block,ddof=1}},\\, \\mathrm{SE}_{\\mathrm{shot}})$:

| Campaign Identifier | Physical QPU | Physical Subgraph | Physical Shots | Primary Validated Observable ($\\pm \\mathrm{SE}_{\\mathrm{eff}}$) | Milestone Determination | Status |
| :--- | :--- | :---: | :---: | :--- | :--- | :---: |
| **OSIRIS-HP-001** | `ibm_fez` (Heron r2) | 6 transmons | 49,152 | Subspace Population: **80.20 +/- 0.62%** | Restricted 1-qubit subspace population verified | **VERIFIED** |
| **OSIRIS-HP-002** | `ibm_fez` (Heron r2) | Unpinned (6–7Q across 2 regions) | 73,728 | 1-Pair Bell Fidelity: **79.49 +/- 0.37%** | 1-qubit entanglement verified ($Z_{\\mathrm{shot}} = 78.9\\sigma$, $t_2 = 172.0, p = 1.69 \\times 10^{-5}$); unpinned layouts & cross-stage optimization noted | **VERIFIED (WITH LAYOUT CAVEAT)** |
| **7-Edge Stagger DD** | `ibm_marrakesh` (Heron r1) | 14 transmons | 86,016 | State Fidelity: **0.9023 +/- 0.0069** | Edge (14, 15) +2.28 pp cross-talk gain; 86.0% ensemble win-rate ($p < 10^{-4}$) | **CERTIFIED** |
| **OSIRIS-QAML-001** | `ibm_marrakesh` (Heron r1) | 8 transmons | 81,920 | Adversarial Retention vs. CNN | Quarantined as inconclusive due to statistical power deficit at $N_{\\mathrm{test}}=5$ (20 pp step size) | **QUARANTINED** |
| **OSIRIS-HP-003** | `ibm_fez` (Heron r2) | Unpinned (disjoint 8Q regions) | 46,080 | Marginal Mean $\\bar{\\mathcal{F}} = \\mathbf{74.95 \\pm 0.41\\%}$; Joint Bell-LP $[0.4990, 0.7375]$ | Per-pair entanglement & 1-photon collapse ($F_1 = 42.42\\%$) verified; **$>1\\text{ ebit}$ joint overlap ($F_{\\mathrm{joint}} > 0.50$) uncertified under 3-basis marginals** | **PER-PAIR VERIFIED / JOINT INCONCLUSIVE** |

---

## 2. OSIRIS-HP-002 & OSIRIS-HP-003 Forensic Audit & v2.1 Errata

### 2.1 Exact Physical QPY Layout & Compiler Boundary Audit
Direct inspection of the compiled QPY binaries (`hp002_staged_circuits.qpy` and `hp003_staged_circuits.qpy`) established three methodological corrections to the initial v2.0 report:
1. **Disjoint Physical Subgraphs Across Arms (Retires `100.59%` Coherence-Ceiling Claim):** Neither `stage_hp002_harness.py` nor `stage_hp003_harness.py` passed `initial_layout` to `transpile(..., optimization_level=3)`. In **HP-003**, `Arm B` executed on bottom-chip transmons `{123, 136, 142, 143, 144, 145, 146, 147}` (readout `[147, 143, 123, 144]`), whereas positive-control `Arm A` executed on top-chip transmons `{2, 3, 4, 5, 16, 21, 22, 23}` (readout `[5, 4, 21, 16]`). Comparing `Arm B` directly against `Arm A` confounds algorithmic recovery with spatial $T_1/T_2$ and CZ calibration variation across disjoint processor regions; the prior `100.59%` / `100.6%` coherence-ceiling parity claim is formally retired.
2. **SPAM Calibration Subgraph Restriction:** Recorded `ibm_fez` readout errors `{146, 143, 123, 144}` match **only `Arm B-Def`** (`[146, 143, 123, 144]`) and Pair 2 of `Arm B` (`[123, 144]`). Re-audited analysis in `hp003/mitigated_hp003_analysis.json` withholds SPAM rescaling on uncalibrated readout transmons.
3. **HP-002 Stage Barriers & Block Ordering:** In HP-002, `qc.barrier` was placed only in `Arm A` (`U_1 U_1^\\dagger`), allowing `optimization_level=3` to co-optimize `Arm B` across the scrambling–decoding boundary from 23 CZs down to 17 CZs (`Arm A` routed across 7 transmons `{123, 136, 140..144}` at 18 CZs), and HP-002 blocks executed in fixed order rather than shuffled order.

### 2.2 Re-Audited Physical Telemetry on `ibm_fez` (Job `dbndsvoou4as73b4cvi0`, Block #48)
- **Total Physical Shots:** 46,080 shots across 45 circuits (1,024 shots/circuit, 3 randomized blocks = 3,072 shots/arm/basis in global bases `XX, YY, ZZ`).
- **Uncertainty & Significance Convention:** Reports $\\mathrm{SE}_{\\mathrm{eff}} = \\max(\\mathrm{SEM}_{\\mathrm{block,ddof=1}},\\, \\mathrm{SE}_{\\mathrm{shot}})$ alongside exact one-sided Student-$t$ ($\nu = 2$) tail probabilities, replacing $\\nu=2$ Gaussian $Z$-extrapolations ($p < 10^{-100}$) and sub-shot-noise 3-block SEMs.

| Experimental Arm | Readout Qubits `[c0..c3]` | $\\mathcal{F}_1(R_1, M_1')$ | $\\mathcal{F}_2(R_2, M_2')$ | Marginal Mean $\\bar{\\mathcal{F}} \\pm \\mathrm{SE}_{\\mathrm{eff}}$ | Tight 16-State Bell-LP $F_{\\mathrm{joint}}$ | Re-Audited Determination |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Arm A (Positive Control, Top-Chip)** | `[5, 4, 21, 16]` | $73.24 \\pm 0.59\\%$ | $75.78 \\pm 0.57\\%$ | **74.51 +/- 0.41%** | **`[0.4902, 0.7324]`** | Per-pair entangled; disjoint subgraph from Arm B |
| **Arm B (Restricted Recovery, Bottom-Chip)** | `[147, 143, 123, 144]` | $73.75 \\pm 0.59\\%$ | $76.16 \\pm 0.57\\%$ | **74.95 +/- 0.41%** | **`[0.4990, 0.7375]`** | **Per-pair entangled ($Z_{\\mathrm{shot}}=60.7\\sigma$, $p_{t_2}=9.4\\times 10^{-5}$); $>1\\text{ ebit}$ uncertified ($F_{\\min}=0.4990 \\le 0.50$)** |
| **Arm B-Def (Single Photon Sub-Capacity)** | `[146, 143, 123, 144]` | $42.42 \\pm 0.72\\%$ | $24.90 \\pm 0.78\\%$ | 33.66 +/- 0.53% | `[0.0000, 0.2490]` | **1-Photon Sub-Capacity Collapse Verified ($\\mathcal{W}_1 = +0.0758 \\ge 0$)** |
| **Arm B-Ctrl (Unentangled Early Radiation)** | `[20, 23, 4, 22]` | $26.19 \\pm 0.78\\%$ | $24.14 \\pm 0.78\\%$ | 25.16 +/- 0.55% | `[0.0000, 0.2414]` | Unentangled Radiation Floor (~25%) |
| **Arm B-Held (Held-Out Scrambler Seed 158)** | `[20, 22, 24, 23]` | $26.14 \\pm 0.78\\%$ | $25.28 \\pm 0.78\\%$ | 25.71 +/- 0.82% | `[0.0000, 0.2528]` | Scrambler Specificity Floor (~25%) |

### 2.3 Exact Multi-Qubit Entanglement & Capacity Interpretation
1. **Why 3-Basis Marginal Mean Cannot Certify $>1\\text{ ebit}$ Here:** A 1-ebit Schmidt-rank-2 mixture $\\rho_{\\mathrm{SR2}} = \\frac{1}{4}(|\\Phi^+\\rangle_1|00\\rangle_2 + |\\Phi^+\\rangle_1|11\\rangle_2 + |00\\rangle_1|\\Phi^+\\rangle_2 + |11\\rangle_1|\\Phi^+\\rangle_2)$ achieves $\\mathcal{F}_1 = \\mathcal{F}_2 = \\bar{\\mathcal{F}} = 75.00\\%$ and $\\mathcal{F}_1 \\mathcal{F}_2 = 56.25\\%$ while having true 4-qubit joint overlap $F_{\\mathrm{joint}} = 0.5000$ (at most 1 ebit). Over all valid 4-qubit density matrices consistent with HP-003's 6 measured marginal correlators, the exact tight Fréchet–Bell-LP lower bound is $F_{\\mathrm{joint}} \\ge \\max(0, \\mathcal{F}_1 + \\mathcal{F}_2 - 1) = 0.499023 < 0.5000$.
2. **Validated Empirical Contrasts:** Individual decoded pairs in Arm B exceed the 2-qubit separability bound ($0.50$), single-photon deprivation (Arm B-Def) collapses to $\\bar{\\mathcal{F}} = 33.66\\%$ ($\\Delta \\bar{\\mathcal{F}} = +41.29\\text{ pp}$), unentangled early radiation (Arm B-Ctrl) collapses to $25.16\\%$ ($\\Delta \\bar{\\mathcal{F}} = +49.79\\text{ pp}$), and held-out scrambler Seed 158 (Arm B-Held) collapses to $25.71\\%$ ($\\Delta \\bar{\\mathcal{F}} = +49.24\\text{ pp}$).
3. **Follow-On Protocol (`OSIRIS-HP-004`, Executed Nov 16, 2026 on `ibm_marrakesh`, Job `dbti1fbsbsbs73cb2vu0`):** Layout-locked, `routing_method='none'` 9-basis (`{X,Y,Z}_1 \\times {X,Y,Z}_2`) 16-stabilizer joint tomography on `[34, 52, 32, 33, 39, 53, 54, 55]` (`292,864` shots across 6 randomized blocks; see `experiments/hp004/RESULT.md`) subsequently certified raw unmitigated $F_{\\mathrm{joint}} = \\mathbf{0.6334 \\pm 0.0026}$ ($+50.64\\sigma$ above the `0.5000` 1-ebit Schmidt-rank-2 ceiling; 5/5 pre-registered hypotheses passed) while its 2-qubit marginals ($\\bar{\\mathcal{F}} = 0.7438$, $F_1 + F_2 - 1 = 0.4875$) confirmed why 3-basis marginal estimators (`HP-003`) are insufficient.

---

## 3. Methodological Quarantine: OSIRIS-QAML-001
The QAML-001 hardware benchmark evaluated adversarial accuracy at $N_{\\mathrm{test}} = 5$ samples, producing a discrete $\\pm 20\\text{ pp}$ granularity. Because the classical baseline boundary artifact biased the comparative metric, the claim of quantum advantage under PGD attack remains formally quarantined as inconclusive until high-sample validation ($N \\ge 50$) is conducted.

---

## 4. Cryptographic Provenance, Permit Algorithm & Line-Ending Canonicalization

The campaign is anchored across Blocks #1 through #48 in `evidence_ledger.db` via SHA-256 hash chaining:
- **Authority Permit Algorithm Clarification:** `authority_permit.signature` fields in `sys021_hp001_phase2_grant.json`, `sys021_hp002_grant.json`, and `sys021_hp003_grant.json` are **32-byte (64-hex-character) symmetric HMAC-SHA256 tags** minted by `NodeBetaGovernor` (`osiris_cli/node_beta_governance.py`), not 64-byte asymmetric Ed25519 signatures.
- **CRLF vs. LF Digest Note for Cross-Platform Verification:** Historical Blocks #44 (`physical_hp002_results.json`, `0be99120...`) and #47 (`stage_hp003_harness.py`, `68174cfd...`) computed SHA-256 over Windows CRLF (`\\r\\n`) byte streams; on POSIX LF (`\\n`) checkouts their byte digests are `b9a5...` / `9e38bc1b...` respectively, and normalize identically when CRLF/LF canonicalization is applied.
- **Key Ledger Sequence Anchors:**
  - **Block #35 / #38:** `SYS021_HP001_PHASE2_DISPATCH_GRANT`
  - **Block #39:** `SYS021_HP001_PHASE2_HARDWARE_ACQUISITION_RECEIPT` (`e78d2b78...`)
  - **Block #42:** `SYS021_QAML_001_HARDWARE_ACQUISITION_RECEIPT` (`5df1ba53...`, Quarantined)
  - **Block #43:** `SYS021_HP002_DISPATCH_GRANT` (`08d20448...`, `stage_hp002_harness.py` LF `72d6fed2...`)
  - **Block #44:** `SYS021_HP002_HARDWARE_ACQUISITION_RECEIPT` (Job `dbnchbgqtr0c73acppe0`, `e19fc278...`)
  - **Block #46:** `SYS021_HP003_HARDWARE_ACQUISITION_RECEIPT` (Job `dbncs05t7fqc73ajci00`, `11da5b59...`, Quarantined)
  - **Block #47:** `SYS021_HP003_DISPATCH_GRANT` (`74a0987e...`, `stage_hp003_harness.py` CRLF `68174cfd...` / LF `9e38bc1b...`)
  - **Block #48:** `SYS021_HP003_HARDWARE_ACQUISITION_RECEIPT` (Job `dbndsvoou4as73b4cvi0`, `3eee9491...`, `physical_hp003_results.json` LF `757442a2...`)
"""

with open(pkg_dir / "EXPERIMENT_REPORT_v2.0.md", "w", encoding="utf-8") as f:
    f.write(report_content.strip() + "\n")

lines = []
for p in sorted(pkg_dir.rglob("*")):
    if p.is_file() and p.name != "SHA256SUMS.txt":
        rel = p.relative_to(pkg_dir)
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{h}  {rel.as_posix()}")

with open(pkg_dir / "SHA256SUMS.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"Updated EXPERIMENT_REPORT_v2.0.md and SHA256SUMS.txt across {len(lines)} packaged files (ledger preserved).")
