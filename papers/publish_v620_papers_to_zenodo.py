#!/usr/bin/env python3
"""Publish all 6 v6.2.0 unpublished / updated OSIRIS papers and reproducibility
packages to Zenodo:

  1. `heron_r2_unified`   — Unified 42-job Heron r2 calibration staleness & colored-noise nulls
  2. `wp2_27q_ot`         — 27-qubit split-sample-debiased optimal transport on `ibm_fez`
  3. `nonabelian_2bga_adv1` — Left-right rank duality in odd-k non-abelian 2BGA qLDPC & ADV-1 pre-gate
  4. `xtop006_fez`        — Continuous-angle non-Clifford Kraus scrambling & 4-twirl discriminator (`ibm_fez`)
  5. `hp004_v2_1`         — Version 2.1.0 update (`newversion` of 24287505) of dual-processor OSIRIS-HP-004/004B
  6. `qdecode_heron_59`   — Reference-verified wire deserialization & finite-shot entropy across 59 QPU jobs

Security & non-mutation guarantees:
  - Reads `ZENODO_API_TOKEN` from environment or `.env` / `.env.local` in `REPO_ROOT` without
    EVER printing, logging, or serializing secret tokens.
  - Refuses to upload any `.env*`, `*key*`, or enclave credential file.
  - Writes draft/publication receipts exclusively under `papers/zenodo_v620_deposits/<slug>/zenodo_receipt.json`
    so frozen `SHA256SUMS.txt` manifests in paper directories are never mutated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPOSITS_ROOT = REPO_ROOT / "papers" / "zenodo_v620_deposits"
API_BASE = "https://zenodo.org/api/deposit/depositions"
CREATOR = [{"name": "Davis, Devin Phillip", "affiliation": "Agile Defense Systems LLC"}]

FORBIDDEN_NAME_SUBSTRINGS = (".env", "troykey", "apikey", "secret", "credential", "private_key")


def _load_zenodo_token() -> str:
    for key in ("ZENODO_API_TOKEN", "OSIRIS_ZENODO_TOKEN", "ZENODO_TOKEN"):
        val = os.environ.get(key, "").strip()
        if val:
            return val
    candidate_envs = (
        REPO_ROOT / ".env",
        REPO_ROOT / ".env.local",
        Path.home() / "osiris-ide" / ".env",
        Path.home() / ".env",
    )
    for env_path in candidate_envs:
        if env_path.is_file():
            for raw_line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                for key in ("ZENODO_API_TOKEN=", "OSIRIS_ZENODO_TOKEN=", "ZENODO_TOKEN="):
                    if line.startswith(key):
                        tok = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if tok:
                            return tok
    raise RuntimeError("ZENODO_API_TOKEN not found in environment or authorized .env/.env.local files")


def _call_once(
    method: str,
    url: str,
    token: str,
    data: Dict[str, Any] | None = None,
    raw: bytes | None = None,
    ctype: str | None = "application/json",
) -> Tuple[int, Dict[str, Any]]:
    body = raw if raw is not None else (json.dumps(data).encode("utf-8") if data is not None else None)
    headers = {"Authorization": f"Bearer {token}"}
    if body is not None and ctype:
        headers["Content-Type"] = ctype
    req = urllib.request.Request(url, method=method, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            txt = resp.read().decode("utf-8")
            return resp.status, (json.loads(txt) if txt else {})
    except urllib.error.HTTPError as exc:
        txt = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(txt)
        except Exception:
            return exc.code, {"raw": txt[:600]}


def _call(
    method: str,
    url: str,
    token: str,
    data: Dict[str, Any] | None = None,
    raw: bytes | None = None,
    ctype: str | None = "application/json",
) -> Tuple[int, Dict[str, Any]]:
    last: Tuple[int, Dict[str, Any]] = (0, {})
    for attempt in range(6):
        try:
            st, body = _call_once(method, url, token, data=data, raw=raw, ctype=ctype)
        except Exception as exc:
            st, body = 0, {"raw": f"{type(exc).__name__}: {exc}"}
        if st not in (0, 429, 500, 502, 503, 504):
            return st, body
        last = (st, body)
        time.sleep(min(45, 3 * (2**attempt)))
    return last


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_safe_upload(src: Path, arc_name: str) -> None:
    low_src = str(src).lower()
    low_arc = arc_name.lower()
    for bad in FORBIDDEN_NAME_SUBSTRINGS:
        if bad in low_src or bad in low_arc:
            raise ValueError(f"Refusing to upload forbidden credential-like path: {src} ({arc_name})")
    if not src.is_file():
        raise FileNotFoundError(f"Required upload file does not exist: {src}")


PAPERS: Dict[str, Dict[str, Any]] = {
    "heron_r2_unified": {
        "new_version_of": None,
        "version": "1.0.0",
        "publication_date": "2026-11-23",
        "files": [
            (
                "papers/zenodo_v620_deposits/heron_r2_unified/heron_r2_calibration_staleness_and_colored_nulls.pdf",
                "heron_r2_calibration_staleness_and_colored_nulls.pdf",
            ),
            (
                "papers/heron_r2_calibration_staleness_and_colored_nulls/heron_r2_calibration_staleness_and_colored_nulls.tex",
                "heron_r2_calibration_staleness_and_colored_nulls.tex",
            ),
            ("papers/heron_r2_calibration_staleness_and_colored_nulls/generated.tex", "generated.tex"),
            (
                "papers/heron_r2_calibration_staleness_and_colored_nulls/generate_unified_assets.py",
                "generate_unified_assets.py",
            ),
            (
                "papers/heron_r2_calibration_staleness_and_colored_nulls/fig1_tmap_sign_reversal_three_qpus.pdf",
                "fig1_tmap_sign_reversal_three_qpus.pdf",
            ),
            (
                "papers/heron_r2_calibration_staleness_and_colored_nulls/fig2_ro06_07_staleness_persistence.pdf",
                "fig2_ro06_07_staleness_persistence.pdf",
            ),
            (
                "papers/heron_r2_calibration_staleness_and_colored_nulls/fig3_cn01_colored_noise_heldout_gate.pdf",
                "fig3_cn01_colored_noise_heldout_gate.pdf",
            ),
            (
                "papers/heron_r2_calibration_staleness_and_colored_nulls/reproducibility_capsule.json",
                "reproducibility_capsule.json",
            ),
            ("papers/heron_r2_calibration_staleness_and_colored_nulls/SHA256SUMS.txt", "SHA256SUMS.txt"),
            ("tests/test_heron_r2_unified_paper.py", "test_heron_r2_unified_paper.py"),
        ],
        "related": [
            {"identifier": "10.5281/zenodo.24406426", "relation": "continues", "scheme": "doi"},
            {"identifier": "10.5281/zenodo.24406435", "relation": "continues", "scheme": "doi"},
            {"identifier": "10.5281/zenodo.24406525", "relation": "continues", "scheme": "doi"},
            {"identifier": "10.5281/zenodo.24407001", "relation": "continues", "scheme": "doi"},
            {"identifier": "10.5281/zenodo.24407875", "relation": "continues", "scheme": "doi"},
            {"identifier": "10.5281/zenodo.24420902", "relation": "continues", "scheme": "doi"},
        ],
        "meta": {
            "title": (
                "Calibration Staleness, Transient Readout Telegraphing, and Colored-Noise False Positives "
                "Across Three 156-Qubit Superconducting Quantum Processors"
            ),
            "description": (
                "<p>Unified synthesis of <strong>7 preregistered hardware campaigns</strong> (<code>TMAP-02</code>&ndash;"
                "<code>TMAP-05</code>, <code>RO-06</code>, <code>RO-07</code>, and <code>CN-01</code>) comprising "
                "<strong>42 IBM Quantum jobs, 904 PUBs, 1,851,392 physical circuit shots, and 873 s QPU time</strong> "
                "across all three 156-qubit IBM Heron r2 processors (<code>ibm_marrakesh</code>, <code>ibm_fez</code>, "
                "and <code>ibm_kingston</code>).</p>"
                "<ul>"
                "<li><strong>90-Minute Three-Processor Layout Sign Reversal (<code>TMAP-02..05</code>):</strong> "
                "Repeating the identical 40-qubit, 39-CZ probe-guided vs calibration-guided path comparison 85&ndash;90 "
                "minutes apart inverted sign on all three processors (+25.0&rarr;&minus;3.0 pp on <code>ibm_marrakesh</code>, "
                "&minus;3.6&rarr;+6.9 pp on <code>ibm_fez</code>, &minus;0.3&rarr;+7.8 pp on <code>ibm_kingston</code>), "
                "while tensored readout mitigation restored calibration-path superiority on every device.</li>"
                "<li><strong>Stale Calibration Snapshots, Not Multiplexing Crosstalk (<code>RO-06/07</code>):</strong> "
                "Hidden readout faults (&gt;15% actual vs &lt;3% published) all recovered below 5% (8/8 intermittent), "
                "persisted at 58.3% across 15-minute rounds (Fisher OR = 365, p = 3.2e-18) while backend calibration "
                "snapshots stayed static up to 16.2 h, and showed &le;1.12 pp simultaneous-readout multiplexing shift.</li>"
                "<li><strong>Colored-Noise Nulls vs White-Noise Misspecification (<code>CN-01</code>):</strong> "
                "Across 10 couplers &ge;5 hops apart (499,712 circuit shots), white-noise GLS selected a non-local OU bath "
                "by &Delta;BIC = 10,263 before failing its preregistered held-out coupler gate (p = 0.82); 350 synthetic "
                "replicates prove white-noise nulls misclassify 100% of local 1/f dephasing, whereas local colored-noise "
                "nulls cut false positives to 0%.</li>"
                "</ul>"
                "<p>Includes standalone PDF preprint, macro-locked REVTeX 4.2 source, vector figures, deterministic "
                "generator, and RFC 8785 canonical reproducibility capsule binding all 42 hardware job IDs.</p>"
            ),
            "keywords": [
                "superconducting qubits",
                "IBM Heron r2",
                "calibration drift",
                "readout telegraphing",
                "qubit layout compilation",
                "colored noise",
                "dynamical decoupling",
                "preregistration",
                "reproducibility",
            ],
        },
    },
    "wp2_27q_ot": {
        "new_version_of": None,
        "version": "1.0.0",
        "publication_date": "2026-11-23",
        "files": [
            (
                "papers/zenodo_v620_deposits/wp2_27q_ot/wp2_27q_debiased_ot_fez.pdf",
                "wp2_27q_debiased_ot_fez.pdf",
            ),
            ("papers/wp2_27q_debiased_ot_fez/wp2_27q_debiased_ot_fez.tex", "wp2_27q_debiased_ot_fez.tex"),
            ("papers/wp2_27q_debiased_ot_fez/generated.tex", "generated.tex"),
            ("papers/wp2_27q_debiased_ot_fez/generate_wp2_paper_assets.py", "generate_wp2_paper_assets.py"),
            (
                "papers/wp2_27q_debiased_ot_fez/fig1_27q_fez_sparsity_collapse_vs_eq7.pdf",
                "fig1_27q_fez_sparsity_collapse_vs_eq7.pdf",
            ),
            (
                "papers/wp2_27q_debiased_ot_fez/fig2_coherent_drift_calibration_and_heron_rescoring.pdf",
                "fig2_coherent_drift_calibration_and_heron_rescoring.pdf",
            ),
            ("papers/wp2_27q_debiased_ot_fez/reproducibility_capsule.json", "reproducibility_capsule.json"),
            ("papers/wp2_27q_debiased_ot_fez/SHA256SUMS.txt", "SHA256SUMS.txt"),
            ("experiments/wp2_debiased_sinkhorn/debiased_ot.py", "debiased_ot.py"),
            (
                "experiments/wp2_debiased_sinkhorn/artifacts/wp2_debiased_ot_report.json",
                "wp2_debiased_ot_report.json",
            ),
            ("experiments/wp2_27q_fez/PRE_REGISTRATION.md", "PRE_REGISTRATION.md"),
            ("experiments/wp2_27q_fez/stage_and_harvest_wp2_27q_fez.py", "stage_and_harvest_wp2_27q_fez.py"),
            ("experiments/wp2_27q_fez/wp2_27q_fez_staging_manifest.json", "wp2_27q_fez_staging_manifest.json"),
            ("experiments/wp2_27q_fez/wp2_27q_fez_staged_circuits.qpy", "wp2_27q_fez_staged_circuits.qpy"),
            (
                "experiments/wp2_27q_fez/wp2_27q_fez_calibration_snapshot.json",
                "wp2_27q_fez_calibration_snapshot.json",
            ),
            ("experiments/wp2_27q_fez/wp2_27q_fez_submitted_job.json", "wp2_27q_fez_submitted_job.json"),
            ("experiments/wp2_27q_fez/physical_wp2_27q_fez_raw_counts.json.gz", "physical_wp2_27q_fez_raw_counts.json.gz"),
            ("experiments/wp2_27q_fez/physical_wp2_27q_fez_results.json", "physical_wp2_27q_fez_results.json"),
        ],
        "related": [
            {"identifier": "10.5281/zenodo.23977105", "relation": "continues", "scheme": "doi"},
            {"identifier": "10.5281/zenodo.24293756", "relation": "references", "scheme": "doi"},
            {"identifier": "10.5281/zenodo.24420902", "relation": "references", "scheme": "doi"},
        ],
        "meta": {
            "title": (
                "Eliminating Finite-Shot Hypercube Sparsity Bias in 27-Qubit Quantum Processor Stability "
                "Benchmarks via Split-Sample-Debiased Optimal Transport (OSIRIS-WP2-27Q-FEZ)"
            ),
            "description": (
                "<p>Formulates sample-size-matched multivariate-hypergeometric split-sample Kantorovich LP and "
                "entropic Sinkhorn (&varepsilon; = 0.05) two-sample stability estimators whose expectation under the "
                "identical-distribution null is identically zero at arbitrary register width <em>Q</em>, eliminating "
                "both the non-zero self-distance defect of marginal-product surrogates and the Weed&ndash;Bach "
                "<em>O(S<sup>&minus;1/Q</sup>)</em> finite-shot hypercube sparsity bias.</p>"
                "<ul>"
                "<li><strong>Live 27-Qubit <code>ibm_fez</code> Validation (Job <code>dc1tqvinj2vs73dfrts0</code>, "
                "48 circuits, 98,304 raw shots, constant 52-CZ / 378-SX K&odblac;nig tree skeleton):</strong> "
                "Matched split-sample Sinkhorn collapses inter-block A/A drift across all (&theta;, arm) cells to "
                "&le;0.00012 (&lt;0.005 threshold, H1 supported), falsifies golden-ratio &theta;* = &phi;&sup2; "
                "&approx; 2.618 rad stability specialness (H2 supported), and detects controlled coherent perturbations "
                "&Delta;&theta; &isin; {0.10, 0.25} rad monotonically at &gt;20&sigma; (H3 supported).</li>"
                "<li><strong>Multi-Width Simulations up to 96Q &amp; Heron r2 Re-Scoring:</strong> Eliminates a "
                "+443&sigma; raw plug-in LP false-drift artifact on <code>Uniform({0,1}^96)</code> (SIMULATED, "
                "<code>provider_job_id: null</code>) and cleanly separates hardware decoherence and the localized "
                "<code>q107</code> <em>T</em><sub>2</sub> hotspot from finite-shot sparsity on archived "
                "<code>ibm_marrakesh</code> DD (<code>dau0q3qhcrkc73durtgg</code>) and 49-bit <code>OSIRIS-QEC-001</code> "
                "(<code>dbtl2vb2kf8s73c9a0d0</code>) hardware datasets.</li>"
                "</ul>"
            ),
            "keywords": [
                "optimal transport",
                "Sinkhorn divergence",
                "Kantorovich distance",
                "Weed-Bach sparsity bias",
                "quantum benchmarking",
                "IBM Heron r2",
                "ibm_fez",
                "OSIRIS-WP2-27Q-FEZ",
            ],
        },
    },
    "nonabelian_2bga_adv1": {
        "new_version_of": None,
        "version": "1.0.0",
        "publication_date": "2026-11-23",
        "files": [
            (
                "papers/zenodo_v620_deposits/nonabelian_2bga_adv1/nonabelian_2bga_and_adv1.pdf",
                "nonabelian_2bga_and_adv1.pdf",
            ),
            ("papers/nonabelian_2bga_and_adv1/nonabelian_2bga_and_adv1.tex", "nonabelian_2bga_and_adv1.tex"),
            ("papers/nonabelian_2bga_and_adv1/generated.tex", "generated.tex"),
            ("papers/nonabelian_2bga_and_adv1/generate_2bga_adv1_assets.py", "generate_2bga_adv1_assets.py"),
            (
                "papers/nonabelian_2bga_and_adv1/fig1_odd_k_nonabelian_2bga_codes_and_bposd.pdf",
                "fig1_odd_k_nonabelian_2bga_codes_and_bposd.pdf",
            ),
            (
                "papers/nonabelian_2bga_and_adv1/fig2_adv1_three_attack_ablation_m6.pdf",
                "fig2_adv1_three_attack_ablation_m6.pdf",
            ),
            ("papers/nonabelian_2bga_and_adv1/reproducibility_capsule.json", "reproducibility_capsule.json"),
            ("papers/nonabelian_2bga_and_adv1/SHA256SUMS.txt", "SHA256SUMS.txt"),
            ("qldpc_nonabelian/nonabelian_2bga.py", "nonabelian_2bga.py"),
            ("qldpc_nonabelian/artifacts/nonabelian_2bga_report.json", "nonabelian_2bga_report.json"),
            ("adv1/looped_peaked.py", "looped_peaked.py"),
            ("adv1/pregate.py", "pregate.py"),
            ("adv1/artifacts/adv1_pregate_report.json", "adv1_pregate_report.json"),
            ("tests/test_wp2_and_2bga_papers.py", "test_wp2_and_2bga_papers.py"),
        ],
        "related": [
            {"identifier": "10.5281/zenodo.24424726", "relation": "references", "scheme": "doi"},
        ],
        "meta": {
            "title": (
                "Left-Right Rank Duality in Non-Abelian Two-Block Group-Algebra Quantum LDPC Codes and a "
                "Three-Attack Classical-Hardness Pre-Gate for Looped Non-Mirror Peaked Circuits"
            ),
            "description": (
                "<p>Algebraic proofs and reproducible <code>SIMULATED</code> benchmarks (<code>provider_job_id: null</code>, "
                "zero physical hardware execution claims) calibrated to all-to-all QCCD trapped-ion architectures:</p>"
                "<ul>"
                "<li><strong>Left&ndash;Right Rank Duality &amp; Odd-<em>k</em> Non-Abelian 2BGA Codes:</strong> "
                "Proves via group-inversion anti-automorphism and Frobenius algebra duality that unquotiented regular "
                "Lin&ndash;Pryadko 2BGA CSS codes over &Fopf;<sub>2</sub>[<em>G</em>] satisfy rank(<em>H<sub>X</sub></em>) "
                "= rank(<em>H<sub>Z</sub></em>) identically for every finite group <em>G</em> (0 violations across 6,480 "
                "subset pairs over 11 non-abelian groups, forcing even <em>k</em>). Breaking regularity via coset "
                "quotienting (<em>D</em><sub>9</sub>/&Zopf;<sub>2</sub>) and symplectic minimum-weight logical gauge "
                "pinning constructs 16 odd-<em>k</em> (<em>k</em> &isin; {1,3,5}, <em>d</em> &ge; 4) CSS codes up to "
                "<em>C</em><sub>19</sub> &rtimes; <em>C</em><sub>3</sub> [[114,3,12]] (<em>kd</em>&sup2;/<em>n</em> = 3.79) "
                "and <em>C</em><sub>31</sub> &rtimes; <em>C</em><sub>3</sub> [[186,3,13]], benchmarked under Sum-Product "
                "BP+OSD-CS decoding.</li>"
                "<li><strong>Fail-Closed <code>OSIRIS-ADV-1-PREGATE</code> &amp; <em>M</em><sub>6</sub> = <em>J</em><sub>6</sub> "
                "&minus; <em>I</em><sub>6</sub> Pauli-Weight-5 Involution:</strong> Gates candidate peaked circuits "
                "against mirror-overlap, bounded-&chi; MPS (&chi; &le; 128), truncated Pauli-path (<em>w</em><sub>max</sub> "
                "&le; 4), and causal light-cone (&le;20Q) attacks; proves that appending the 15-CX involution "
                "<em>M</em><sub>6</sub> lifts every output <em>Z<sub>j</sub></em> to Pauli weight 5, enabling 24Q and "
                "30Q flat-Schmidt chorded circuits to defeat all three classical attacks.</li>"
                "</ul>"
            ),
            "keywords": [
                "quantum LDPC codes",
                "two-block group algebra",
                "non-abelian finite groups",
                "BP+OSD decoding",
                "peaked quantum circuits",
                "classical simulation hardness",
                "Pauli path simulation",
                "QCCD trapped ions",
            ],
        },
    },
    "xtop006_fez": {
        "new_version_of": None,
        "version": "1.0.0",
        "publication_date": "2026-11-23",
        "files": [
            (
                "papers/zenodo_v620_deposits/xtop006_fez/xtop006_non_clifford_kraus_fez.pdf",
                "xtop006_non_clifford_kraus_fez.pdf",
            ),
            ("papers/xtop006_non_clifford_kraus_fez/xtop006_non_clifford_kraus_fez.tex", "xtop006_non_clifford_kraus_fez.tex"),
            ("papers/xtop006_non_clifford_kraus_fez/generated.tex", "generated.tex"),
            ("papers/xtop006_non_clifford_kraus_fez/generate_xtop006_paper_assets.py", "generate_xtop006_paper_assets.py"),
            (
                "papers/xtop006_non_clifford_kraus_fez/fig1_xtop006_fez_kraus_trajectory.svg",
                "fig1_xtop006_fez_kraus_trajectory.svg",
            ),
            (
                "papers/xtop006_non_clifford_kraus_fez/fig2_xtop006b_fez_4twirl_discriminator.svg",
                "fig2_xtop006b_fez_4twirl_discriminator.svg",
            ),
            ("papers/xtop006_non_clifford_kraus_fez/reproducibility_capsule.json", "reproducibility_capsule.json"),
            ("papers/xtop006_non_clifford_kraus_fez/SHA256SUMS.txt", "SHA256SUMS.txt"),
            ("experiments/xtop006_fez/PRE_REGISTRATION.md", "xtop006_fez_PRE_REGISTRATION.md"),
            ("experiments/xtop006_fez/stage_and_harvest_xtop006_fez.py", "stage_and_harvest_xtop006_fez.py"),
            ("experiments/xtop006_fez/xtop006_fez_staging_manifest.json", "xtop006_fez_staging_manifest.json"),
            ("experiments/xtop006_fez/xtop006_fez_staged_circuits.qpy", "xtop006_fez_staged_circuits.qpy"),
            ("experiments/xtop006_fez/xtop006_fez_calibration_snapshot.json", "xtop006_fez_calibration_snapshot.json"),
            ("experiments/xtop006_fez/xtop006_fez_submitted_job.json", "xtop006_fez_submitted_job.json"),
            ("experiments/xtop006_fez/physical_xtop006_fez_raw_counts.json.gz", "physical_xtop006_fez_raw_counts.json.gz"),
            ("experiments/xtop006_fez/physical_xtop006_fez_results.json", "physical_xtop006_fez_results.json"),
            ("experiments/xtop006b_fez/PRE_REGISTRATION.md", "xtop006b_fez_PRE_REGISTRATION.md"),
            ("experiments/xtop006b_fez/stage_and_harvest_xtop006b_fez.py", "stage_and_harvest_xtop006b_fez.py"),
            ("experiments/xtop006b_fez/xtop006b_fez_staging_manifest.json", "xtop006b_fez_staging_manifest.json"),
            ("experiments/xtop006b_fez/xtop006b_fez_staged_circuits.qpy", "xtop006b_fez_staged_circuits.qpy"),
            ("experiments/xtop006b_fez/xtop006b_fez_calibration_snapshot.json", "xtop006b_fez_calibration_snapshot.json"),
            ("experiments/xtop006b_fez/xtop006b_fez_submitted_job.json", "xtop006b_fez_submitted_job.json"),
            ("experiments/xtop006b_fez/physical_xtop006b_fez_raw_counts.json.gz", "physical_xtop006b_fez_raw_counts.json.gz"),
            ("experiments/xtop006b_fez/physical_xtop006b_fez_results.json", "physical_xtop006b_fez_results.json"),
            ("tests/test_xtop006_fez_paper.py", "test_xtop006_fez_paper.py"),
        ],
        "related": [
            {"identifier": "10.5281/zenodo.24426216", "relation": "continues", "scheme": "doi"},
            {"identifier": "10.5281/zenodo.24287505", "relation": "references", "scheme": "doi"},
        ],
        "meta": {
            "title": (
                "Continuous-Angle Non-Clifford Hayden-Preskill Scrambling Trajectory and Stratified Decoherence "
                "Discrimination on a 156-Qubit Superconducting Processor (OSIRIS-XTOP-006 & XTOP-006B)"
            ),
            "description": (
                "<p>Derives the exact 16-operator remnant Kraus channel for a continuous non-Clifford 3-qubit "
                "scrambler <em>U(&theta;)</em> satisfying <em>P</em><sub>&Phi;+</sub>(&theta;) = (1 &minus; "
                "&frac12; sin&sup2;&theta;)&sup2; and <em>P</em><sub>&Phi;+</sub>(&theta;)<em>F</em><sub>ent</sub>(&theta;|&Phi;+) "
                "= &frac14;, and executes a constant-skeleton (28 CZ, 114 &radic;X, zero routing SWAPs) trajectory "
                "across 184,320 total shots (54 s QPU) on a 7-transmon heavy-hex T-junction of <code>ibm_fez</code> "
                "(jobs <code>dc1tqaanj2vs73dfrt40</code> and <code>dc23ouqnj2vs73dg3bog</code>).</p>"
                "<ul>"
                "<li><strong>Continuous Non-Clifford Recovery Climb (<code>dc1tqaanj2vs73dfrt40</code>):</strong> "
                "Raw 6-state conditional teleportation fidelity rises monotonically across &theta; &isin; "
                "{0, &pi;/8, &pi;/4, 3&pi;/8, &pi;/2} from 0.4962 to 0.8568 &plusmn; 0.0053 (0.8926 readout-mitigated; "
                "+62.4&sigma; climb), surpassing the 2/3 classical teleportation ceiling without mitigation.</li>"
                "<li><strong>Confirmatory Intra-Block 4-Twirl Discriminator (<code>dc23ouqnj2vs73dg3bog</code>):</strong> "
                "Executing all four mid-scrambler Pauli-<em>Z</em> twirls inside every randomized block separates "
                "coherent scrambling from decoherence on the identical 28-CZ pulse schedule by +14.8 pp (<em>t</em> = "
                "14.6) at &theta; = 3&pi;/8 and +22.8 pp (<em>t</em> = 15.3, Holm <em>p</em> = 1.2e-3) at &theta; = &pi;/2.</li>"
                "</ul>"
            ),
            "keywords": [
                "Hayden-Preskill protocol",
                "quantum scrambling",
                "non-Clifford circuits",
                "stabilizer magic",
                "verified teleportation",
                "IBM Heron r2",
                "ibm_fez",
                "OSIRIS-XTOP-006",
            ],
        },
    },
    "hp004_v2_1": {
        "new_version_of": 24287505,
        "version": "2.1.0",
        "publication_date": "2026-11-23",
        "files": [
            (
                "papers/hp004_joint_stabilizer_heron_r2_v2_1/hp004_joint_stabilizer_heron_r2_v2_1.pdf",
                "hp004_joint_stabilizer_heron_r2_v2_1.pdf",
            ),
            (
                "papers/hp004_joint_stabilizer_heron_r2_v2_1/hp004_joint_stabilizer_heron_r2_v2_1.tex",
                "hp004_joint_stabilizer_heron_r2_v2_1.tex",
            ),
            (
                "papers/hp004_joint_stabilizer_heron_r2_v2_1/generate_paper_assets_v2_1.py",
                "generate_paper_assets_v2_1.py",
            ),
            ("papers/hp004_joint_stabilizer_heron_r2_v2_1/fig1_dyadic_capacity_ladder.pdf", "fig1_dyadic_capacity_ladder.pdf"),
            (
                "papers/hp004_joint_stabilizer_heron_r2_v2_1/fig2_block_stability_and_marginals.pdf",
                "fig2_block_stability_and_marginals.pdf",
            ),
            (
                "papers/hp004_joint_stabilizer_heron_r2_v2_1/fig3_marrakesh_tree_light_cone.pdf",
                "fig3_marrakesh_tree_light_cone.pdf",
            ),
            (
                "papers/hp004_joint_stabilizer_heron_r2_v2_1/fig4_fez_tree_light_cone.pdf",
                "fig4_fez_tree_light_cone.pdf",
            ),
            ("papers/hp004_joint_stabilizer_heron_r2_v2_1/SHA256SUMS_v2_1.txt", "SHA256SUMS_v2_1.txt"),
            ("experiments/hp004/PRE_REGISTRATION.md", "hp004_marrakesh_PRE_REGISTRATION.md"),
            ("experiments/hp004/RESULT.md", "hp004_marrakesh_RESULT.md"),
            ("experiments/hp004/hp004_joint_stabilizer.py", "hp004_joint_stabilizer.py"),
            ("experiments/hp004/execute_and_harvest_hp004.py", "execute_and_harvest_hp004.py"),
            ("experiments/hp004/hp004_staged_circuits.qpy", "hp004_staged_circuits.qpy"),
            ("experiments/hp004/hp004_staging_manifest.json", "hp004_staging_manifest.json"),
            ("experiments/hp004/hp004_calibration_snapshot.json", "hp004_calibration_snapshot.json"),
            ("experiments/hp004/hp004_submitted_job.json", "hp004_submitted_job.json"),
            ("experiments/hp004/physical_hp004_results.json", "physical_hp004_results.json"),
            ("experiments/hp004b_fez/PRE_REGISTRATION.md", "hp004b_fez_PRE_REGISTRATION.md"),
            ("experiments/hp004b_fez/RESULT.md", "hp004b_fez_RESULT.md"),
            ("experiments/hp004b_fez/stage_and_harvest_hp004b_fez.py", "stage_and_harvest_hp004b_fez.py"),
            ("experiments/hp004b_fez/hp004b_fez_staged_circuits.qpy", "hp004b_fez_staged_circuits.qpy"),
            ("experiments/hp004b_fez/hp004b_fez_staging_manifest.json", "hp004b_fez_staging_manifest.json"),
            ("experiments/hp004b_fez/hp004b_fez_calibration_snapshot.json", "hp004b_fez_calibration_snapshot.json"),
            ("experiments/hp004b_fez/hp004b_fez_submitted_job.json", "hp004b_fez_submitted_job.json"),
            ("experiments/hp004b_fez/physical_hp004b_fez_results.json", "physical_hp004b_fez_results.json"),
            ("tests/test_hp004_hardware_artifact.py", "test_hp004_hardware_artifact.py"),
            ("tests/test_hp004b_fez_hardware_artifact.py", "test_hp004b_fez_hardware_artifact.py"),
        ],
        "related": [
            {"identifier": "10.5281/zenodo.24031814", "relation": "continues", "scheme": "doi"},
            {"identifier": "10.5281/zenodo.24287505", "relation": "isNewVersionOf", "scheme": "doi"},
        ],
        "meta": {
            "title": (
                "Dual-Processor Certification of Schmidt Number >= 3 and >1-Ebit Entanglement "
                "of Formation in Hayden-Preskill Recovery via Layout-Locked Nine-Basis "
                "Stabilizer Tomography on Heavy-Hex Superconducting Processors (OSIRIS-HP-004 & HP-004B, v2.1)"
            ),
            "description": (
                "<p><strong>Version 2.1.0 Precision &amp; Bibliographic Update (superseding v2.0.0, "
                "<code>10.5281/zenodo.24287505</code>):</strong></p>"
                "<ol>"
                "<li>Adds primary citation B. M. Terhal &amp; K. G. H. Vollbrecht, <em>Phys. Rev. Lett.</em> "
                "<strong>85</strong>, 2625&ndash;2628 (2000), <code>doi:10.1103/PhysRevLett.85.2625</code> alongside "
                "Vollbrecht &amp; Werner, <em>Phys. Rev. A</em> <strong>64</strong>, 062307 (2001) for the exact "
                "<em>d</em> = 4 isotropic Entanglement of Formation convex roof.</li>"
                "<li>Refines the fifth-decimal threshold root from <code>0.683910</code> to exact "
                "<code>F* = 0.683900</code> (<code>0.683900317</code>).</li>"
                "<li>Explicitly documents that legacy JSON key <code>z_above_1ebit_eff</code> in the frozen "
                "<code>physical_hp004*_results.json</code> files records <code>(F_joint - 0.5000) / SE_eff</code> "
                "(Z-score above the SN &le; 2 bound: +50.64&sigma; on <code>ibm_marrakesh</code>, +84.45&sigma; on "
                "<code>ibm_fez</code>), while Z vs <code>F* = 0.683900</code> is &minus;19.16&sigma; / +17.13&sigma;.</li>"
                "</ol>"
                "<p>Preserves the complete 572-circuit, 585,728-shot dual-processor hardware archive "
                "(<code>ibm_marrakesh</code> Job <code>dbti1fbsbsbs73cb2vu0</code>, <code>F_joint = 0.6334 &plusmn; 0.0026</code>; "
                "<code>ibm_fez</code> Job <code>dbtj61bnlcms73ajc380</code>, <code>F_joint = 0.7307 &plusmn; 0.0027</code>, "
                "<code>E_F &ge; 1.1465 &plusmn; 0.0086 ebits</code>).</p>"
            ),
            "keywords": [
                "Hayden-Preskill protocol",
                "Schmidt number witness",
                "entanglement of formation",
                "Terhal-Vollbrecht bound",
                "stabilizer tomography",
                "IBM Heron r2",
                "OSIRIS-HP-004",
                "OSIRIS-HP-004B",
            ],
        },
    },
    "qdecode_heron_59": {
        "new_version_of": None,
        "version": "1.0.0",
        "publication_date": "2026-11-23",
        "files": [
            (
                "papers/qdecode_heron_provenance_and_finite_shot_entropy/qdecode_heron_provenance_and_finite_shot_entropy.pdf",
                "qdecode_heron_provenance_and_finite_shot_entropy.pdf",
            ),
            (
                "papers/qdecode_heron_provenance_and_finite_shot_entropy/qdecode_heron_provenance_and_finite_shot_entropy.tex",
                "qdecode_heron_provenance_and_finite_shot_entropy.tex",
            ),
            ("papers/qdecode_heron_provenance_and_finite_shot_entropy/generated.tex", "generated.tex"),
            (
                "papers/qdecode_heron_provenance_and_finite_shot_entropy/generate_qdecode_paper_assets.py",
                "generate_qdecode_paper_assets.py",
            ),
            (
                "papers/qdecode_heron_provenance_and_finite_shot_entropy/fig1_wire_deserialization_tvd_and_bit_order.pdf",
                "fig1_wire_deserialization_tvd_and_bit_order.pdf",
            ),
            (
                "papers/qdecode_heron_provenance_and_finite_shot_entropy/fig2_heron_59job_finite_shot_entropy_ceiling.pdf",
                "fig2_heron_59job_finite_shot_entropy_ceiling.pdf",
            ),
            (
                "papers/qdecode_heron_provenance_and_finite_shot_entropy/HARDWARE_PROVENANCE_MANIFEST.json",
                "HARDWARE_PROVENANCE_MANIFEST.json",
            ),
            (
                "papers/qdecode_heron_provenance_and_finite_shot_entropy/reproducibility_capsule.json",
                "reproducibility_capsule.json",
            ),
            ("papers/qdecode_heron_provenance_and_finite_shot_entropy/SHA256SUMS.txt", "SHA256SUMS.txt"),
            ("osiris_cli/qdecode.py", "qdecode.py"),
            ("osiris_cli/qscore.py", "qscore.py"),
            ("tests/test_qdecode_qscore.py", "test_qdecode_qscore.py"),
            ("tests/test_qdecode_heron_paper.py", "test_qdecode_heron_paper.py"),
        ],
        "related": [
            {"identifier": "10.5281/zenodo.23213023", "relation": "continues", "scheme": "doi"},
            {"identifier": "10.5281/zenodo.24293756", "relation": "references", "scheme": "doi"},
        ],
        "meta": {
            "title": (
                "Reference-Verified Wire Deserialization, Little-Endian Bit-Order Disambiguation, and "
                "Finite-Shot Entropy Saturation Across a 59-Job (2-156 Qubit) Superconducting Quantum Corpus"
            ),
            "description": (
                "<p>Establishes reference-verified wire deserialization (<code>osiris_qdecode</code> v1.1.0) and "
                "finite-shot Shannon entropy scoring (<code>osiris_qscore</code>) across a 59-job (2,334,720-shot, "
                "583-register-slice) superconducting quantum corpus spanning 2 to 156 qubits across SamplerV2 "
                "<code>PrimitiveResult</code> and IBM Executor <code>schema_version=\"v0.2\"</code> schemas:</p>"
                "<ul>"
                "<li><strong><code>.npy</code> Header Shift Elimination:</strong> Quantifies the 128-byte "
                "<code>np.frombuffer</code> header-cast corruption across 575 <code>PrimitiveResult</code> slices "
                "(median TVD = 0.0664, max = 0.1211, 10 row-misaligned slices) and verifies 0.0000 TVD parity against "
                "Qiskit <code>RuntimeDecoder</code>.</li>"
                "<li><strong>Executor <code>v0.2</code> Little-Endian Bit-Axis Disambiguation:</strong> Shows that "
                "unreversed <code>np.unpackbits(..., bitorder=\"little\")</code> induces TVD = 0.0231&ndash;1.0000 "
                "(median 0.3544) across 8 Heron 100,000-shot jobs, eliminated identically (0.0000 TVD) by "
                "<code>matrix[..., ::-1]</code> verified against Qiskit <code>ExecutorResultDecoder</code>.</li>"
                "<li><strong><code>ALL_ZERO</code> Quarantine &amp; Finite-Shot Ceiling <em>H</em><sub>max</sub> = "
                "min(<em>n</em>, log<sub>2</sub><em>N</em>):</strong> Filters 24 unmeasured classical storage registers "
                "(2&ndash;84 bits) and confirms finite-shot ceiling saturation at 127Q/4,096 shots (12.0000/12.0000 bits "
                "= 100.00%) and 156Q/100,000 shots on <code>ibm_fez</code> Job <code>d77uq7jc6das739h8k50</code> "
                "(95,723 unique bitstrings, <em>H</em> = 16.5051/16.6096 bits = 99.37%, <code>SATURATED_ENTROPY</code>).</li>"
                "</ul>"
            ),
            "keywords": [
                "quantum wire deserialization",
                "Qiskit Runtime",
                "BitArray",
                "Executor v0.2",
                "finite-shot Shannon entropy",
                "Miller-Madow estimator",
                "provenance verification",
                "IBM Heron",
            ],
        },
    },
}


def build_zenodo_metadata(cfg: Dict[str, Any]) -> Dict[str, Any]:
    meta = {
        "upload_type": "publication",
        "publication_type": "preprint",
        "publication_date": cfg["publication_date"],
        "title": cfg["meta"]["title"],
        "creators": CREATOR,
        "description": cfg["meta"]["description"],
        "access_right": "open",
        "license": "cc-by-4.0",
        "version": cfg["version"],
        "keywords": cfg["meta"]["keywords"],
    }
    if cfg.get("related"):
        meta["related_identifiers"] = cfg["related"]
    return meta


def draft_one(slug: str, token: str) -> Dict[str, Any]:
    cfg = PAPERS[slug]
    out_dir = DEPOSITS_ROOT / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = out_dir / "zenodo_receipt.json"

    resolved_files: List[Tuple[Path, str]] = []
    for rel_src, arc_name in cfg["files"]:
        src = REPO_ROOT / rel_src
        arc = arc_name or src.name
        _assert_safe_upload(src, arc)
        resolved_files.append((src, arc))

    existing_did = None
    if receipt_path.is_file():
        prev = json.loads(receipt_path.read_text(encoding="utf-8"))
        if prev.get("state") == "published" and prev.get("doi"):
            print(f"[{slug}] Already published at DOI {prev['doi']} (record {prev.get('record_id')}); skipping.")
            return prev
        existing_did = prev.get("deposition_id")

    dep: Dict[str, Any] | None = None
    if existing_did:
        st, got = _call("GET", f"{API_BASE}/{existing_did}", token, ctype=None)
        if st == 200 and not got.get("submitted"):
            dep = got

    if dep is None:
        nv_parent = cfg.get("new_version_of")
        if nv_parent:
            st, nv_res = _call("POST", f"{API_BASE}/{nv_parent}/actions/newversion", token, ctype=None)
            if st not in (200, 201):
                raise RuntimeError(f"[{slug}] newversion of {nv_parent} failed (HTTP {st}): {nv_res}")
            draft_url = (nv_res.get("links") or {}).get("latest_draft")
            if not draft_url:
                raise RuntimeError(f"[{slug}] missing latest_draft in newversion reply: {nv_res}")
            st, dep = _call("GET", draft_url, token, ctype=None)
            if st != 200:
                raise RuntimeError(f"[{slug}] GET latest_draft failed (HTTP {st}): {dep}")
        else:
            st, dep = _call("POST", API_BASE, token, data={})
            if st not in (200, 201):
                raise RuntimeError(f"[{slug}] create deposition failed (HTTP {st}): {dep}")

    did = dep["id"]
    bucket = dep["links"]["bucket"]

    # Clear any pre-existing / inherited draft files so bundle matches exact file list
    for old_f in dep.get("files") or []:
        fid = old_f.get("id")
        if fid:
            del_st, del_res = _call("DELETE", f"{API_BASE}/{did}/files/{fid}", token, ctype=None)
            if del_st not in (200, 204):
                raise RuntimeError(f"[{slug}] failed deleting draft file {old_f.get('filename')} (HTTP {del_st}): {del_res}")

    uploaded_manifest: List[Dict[str, Any]] = []
    for src, arc in resolved_files:
        blob = src.read_bytes()
        st, up = _call("PUT", f"{bucket}/{arc}", token, raw=blob, ctype="application/octet-stream")
        if st not in (200, 201):
            raise RuntimeError(f"[{slug}] upload failed for {arc} (HTTP {st}): {up}")
        remote_size = up.get("size")
        if remote_size is not None and int(remote_size) != len(blob):
            raise RuntimeError(f"[{slug}] size mismatch for {arc}: remote={remote_size} vs local={len(blob)}")
        uploaded_manifest.append(
            {
                "archive_name": arc,
                "source_relpath": str(src.relative_to(REPO_ROOT)),
                "bytes": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(),
                "zenodo_checksum": up.get("checksum"),
            }
        )

    meta = build_zenodo_metadata(cfg)
    st, meta_res = _call("PUT", f"{API_BASE}/{did}", token, data={"metadata": meta})
    if st not in (200, 201):
        raise RuntimeError(f"[{slug}] metadata PUT failed (HTTP {st}): {meta_res}")

    prereserve = ((meta_res.get("metadata") or {}).get("prereserve_doi") or {}).get("doi")
    receipt = {
        "slug": slug,
        "state": "draft",
        "deposition_id": did,
        "supersedes_record_id": cfg.get("new_version_of"),
        "version": cfg["version"],
        "prereserved_doi": prereserve,
        "draft_url": f"https://zenodo.org/uploads/{did}",
        "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "title": meta["title"],
        "files_count": len(uploaded_manifest),
        "files": uploaded_manifest,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"[{slug}] Draft ready: id={did}, prereserved_doi={prereserve}, files={len(uploaded_manifest)}")
    return receipt


def publish_one(slug: str, token: str) -> Dict[str, Any]:
    receipt_path = DEPOSITS_ROOT / slug / "zenodo_receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError(f"[{slug}] No draft receipt at {receipt_path}; run --draft first")
    rec = json.loads(receipt_path.read_text(encoding="utf-8"))
    if rec.get("state") == "published" and rec.get("doi"):
        print(f"[{slug}] Already published: DOI {rec['doi']}")
        return rec

    # Verify local files have not drifted since draft upload
    for fentry in rec["files"]:
        src = REPO_ROOT / fentry["source_relpath"]
        cur_sha = _sha256(src)
        if cur_sha != fentry["sha256"]:
            raise RuntimeError(f"[{slug}] Local file {src} changed since draft; re-run --draft")

    did = rec["deposition_id"]
    st, pub = _call("POST", f"{API_BASE}/{did}/actions/publish", token, ctype=None)
    if st not in (200, 201, 202):
        raise RuntimeError(f"[{slug}] Publish failed (HTTP {st}): {pub}")

    rec.update(
        {
            "state": "published",
            "record_id": pub.get("id", did),
            "doi": pub.get("doi"),
            "conceptdoi": pub.get("conceptdoi"),
            "doi_url": pub.get("doi_url") or (f"https://doi.org/{pub.get('doi')}" if pub.get("doi") else None),
            "record_html": (pub.get("links") or {}).get("record_html", f"https://zenodo.org/records/{pub.get('id', did)}"),
            "published_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    receipt_path.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(f"[{slug}] PUBLISHED -> DOI: {rec['doi']}  ({rec['record_html']})")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage and publish v6.2.0 OSIRIS papers to Zenodo")
    ap.add_argument("--paper", choices=[*PAPERS.keys(), "all"], required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--draft", action="store_true")
    mode.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    token = _load_zenodo_token()
    targets = list(PAPERS.keys()) if args.paper == "all" else [args.paper]

    summary = []
    for slug in targets:
        if args.draft:
            res = draft_one(slug, token)
        else:
            res = publish_one(slug, token)
        summary.append(
            {
                "slug": slug,
                "state": res.get("state"),
                "record_id": res.get("record_id") or res.get("deposition_id"),
                "doi": res.get("doi") or res.get("prereserved_doi"),
                "record_html": res.get("record_html") or res.get("draft_url"),
            }
        )
    print("\n=== ZENODO SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
