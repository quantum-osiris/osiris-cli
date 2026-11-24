#!/usr/bin/env python3
"""Stage and (when explicitly authorized) upload the four Nov 23, 2026 OSIRIS manuscripts to Zenodo.

Default mode is ``--dry-run`` (0 network calls):
  1. Verifies ``SHA256SUMS.txt`` and ``reproducibility_capsule.json`` integrity for every manuscript.
  2. Runs the ``SYS-024`` Claims Firewall (``osiris_cli.claims.flagged_in_reply`` + banned-constant check)
     across raw LaTeX, macro-generated LaTeX, macro-expanded LaTeX, and Zenodo deposition metadata.
  3. Writes deterministic ``papers/zenodo_staging_manifest_nov23.json`` with complete Zenodo Deposition
     metadata, exact staged file inventory, and SHA-256 digests.

Live upload mode requires ``--execute`` AND ``ZENODO_TOKEN`` in the environment AND explicit operator
confirmation via ``--confirm-publish`` (or interactive TTY confirmation with ``--interactive``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from osiris_cli import claims, plan_gates  # noqa: E402

ZENODO_API_URL = "https://zenodo.org/api/deposit/depositions"
PUBLICATION_DATE = "2026-11-23"
STAGING_MANIFEST_SCHEMA = "osiris.zenodo_staging_manifest/1"
DEFAULT_STAGING_MANIFEST_PATH = ROOT / "papers" / "zenodo_staging_manifest_nov23.json"
CREATORS: List[Dict[str, str]] = [
    {"name": "Davis, Devin Phillip", "affiliation": "Agile Defense Systems LLC"}
]

HttpCallFn = Callable[
    [str, str, str, Optional[Dict[str, Any]], Optional[bytes], str],
    Tuple[int, Dict[str, Any]],
]


class ZenodoPublishError(RuntimeError):
    """Fail-closed error raised on integrity, SYS-024, credential, or confirmation violations."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"[{code}] {detail}")
        self.code = code
        self.detail = detail


PAPER_ALIASES: Dict[str, str] = {
    "xtop006": "xtop006_non_clifford_kraus_fez",
    "xtop006_non_clifford_kraus_fez": "xtop006_non_clifford_kraus_fez",
    "wp2": "wp2_27q_debiased_ot_fez",
    "wp2_27q_debiased_ot_fez": "wp2_27q_debiased_ot_fez",
    "2bga": "nonabelian_2bga_and_adv1",
    "nonabelian_2bga_and_adv1": "nonabelian_2bga_and_adv1",
    "heron_r2": "heron_r2_calibration_staleness_and_colored_nulls",
    "heron_r2_calibration_staleness_and_colored_nulls": (
        "heron_r2_calibration_staleness_and_colored_nulls"
    ),
    "bt2t": "bt2t001_binary_tetrahedral_fez",
    "bt2t001": "bt2t001_binary_tetrahedral_fez",
    "bt2t001_binary_tetrahedral_fez": "bt2t001_binary_tetrahedral_fez",
}

PAPER_ORDER: Tuple[str, ...] = (
    "xtop006_non_clifford_kraus_fez",
    "wp2_27q_debiased_ot_fez",
    "nonabelian_2bga_and_adv1",
    "heron_r2_calibration_staleness_and_colored_nulls",
    "bt2t001_binary_tetrahedral_fez",
)

PAPERS: Dict[str, Dict[str, Any]] = {
    "xtop006_non_clifford_kraus_fez": {
        "dir": "papers/xtop006_non_clifford_kraus_fez",
        "tex_file": "xtop006_non_clifford_kraus_fez.tex",
        "publication_type": "preprint",
        "version": "1.0.0",
        "related": [
            {
                "identifier": "10.5281/zenodo.24425416",
                "relation": "continues",
                "scheme": "doi",
            }
        ],
        "files": [
            ("papers/xtop006_non_clifford_kraus_fez/xtop006_non_clifford_kraus_fez.tex", None),
            ("papers/xtop006_non_clifford_kraus_fez/generated.tex", None),
            ("papers/xtop006_non_clifford_kraus_fez/generate_xtop006_paper_assets.py", None),
            ("papers/xtop006_non_clifford_kraus_fez/fig1_xtop006_fez_kraus_trajectory.svg", None),
            ("papers/xtop006_non_clifford_kraus_fez/fig2_xtop006b_fez_4twirl_discriminator.svg", None),
            ("papers/xtop006_non_clifford_kraus_fez/reproducibility_capsule.json", None),
            ("papers/xtop006_non_clifford_kraus_fez/SHA256SUMS.txt", None),
            ("experiments/xtop006_fez/PRE_REGISTRATION.md", "xtop006_fez_PRE_REGISTRATION.md"),
            ("experiments/xtop006_fez/stage_and_harvest_xtop006_fez.py", None),
            ("experiments/xtop006_fez/physical_xtop006_fez_results.json", None),
            ("experiments/xtop006_fez/physical_xtop006_fez_raw_counts.json.gz", None),
            ("experiments/xtop006_fez/xtop006_fez_staging_manifest.json", None),
            ("experiments/xtop006_fez/xtop006_fez_calibration_snapshot.json", None),
            ("experiments/xtop006_fez/xtop006_fez_submitted_job.json", None),
            ("experiments/xtop006b_fez/PRE_REGISTRATION.md", "xtop006b_fez_PRE_REGISTRATION.md"),
            ("experiments/xtop006b_fez/stage_and_harvest_xtop006b_fez.py", None),
            ("experiments/xtop006b_fez/physical_xtop006b_fez_results.json", None),
            ("experiments/xtop006b_fez/physical_xtop006b_fez_raw_counts.json.gz", None),
            ("experiments/xtop006b_fez/xtop006b_fez_staging_manifest.json", None),
            ("experiments/xtop006b_fez/xtop006b_fez_calibration_snapshot.json", None),
            ("experiments/xtop006b_fez/xtop006b_fez_submitted_job.json", None),
            ("xtop006/artifacts/xtop006_h2_manifest.json", None),
            ("xtop006/artifacts/xtop006_h2_1_compiled_circuits.json", None),
        ],
        "optional_files": [
            ("papers/xtop006_non_clifford_kraus_fez/xtop006_non_clifford_kraus_fez.pdf", None),
            ("papers/xtop006_non_clifford_kraus_fez/fig1_xtop006_fez_kraus_trajectory.pdf", None),
            ("papers/xtop006_non_clifford_kraus_fez/fig2_xtop006b_fez_4twirl_discriminator.pdf", None),
        ],
        "meta": {
            "title": (
                "Continuous-Angle Non-Clifford Hayden-Preskill Scrambling Trajectory and Stratified "
                "Decoherence Discrimination on a 156-Qubit Superconducting Processor (OSIRIS-XTOP-006)"
            ),
            "description": (
                "<p>Hardware-attested Hayden&ndash;Preskill conditional recovery experiment on a seven-transmon "
                "heavy-hex T-junction of the 156-qubit IBM Heron r2 processor <code>ibm_fez</code> "
                "(jobs <code>dc1tqaanj2vs73dfrt40</code> and <code>dc23ouqnj2vs73dg3bog</code>; 184,320 total shots, "
                "54 s QPU) measuring the exact 16-operator remnant Kraus trajectory for a continuous non-Clifford "
                "three-qubit scrambler <code>U(theta)</code>.</p><ul>"
                "<li>A universal two-sqrt(X) Euler synthesis locks an identical 28-CZ, 126-SX native gate skeleton "
                "with zero routing SWAPs across every angle theta, reducing continuous scrambling sweeps to pure "
                "virtual-Z frame updates.</li>"
                "<li>Six-state conditional recovery rises monotonically from F_6(0 | Phi+) = 0.4964 "
                "(P_Phi+ = 0.8913) through peak stabilizer magic theta = pi/4 (F_6 = 0.5998, P_Phi+ = 0.5198) to "
                "F_6(pi/2 | Phi+) = 0.8568 +/- 0.0043 raw (0.8771 readout-mitigated, P_Phi+ = 0.2454), exceeding "
                "theta = 0 by +0.3602 (t = 62.39, p = 4.54e-6).</li>"
                "<li>Confirmatory job <code>dc23ouqnj2vs73dg3bog</code> executes all four non-trivial mid-scrambler "
                "Pauli-Z twirls inside every randomized block, separating coherent scrambling from dephasing on the "
                "identical 28-CZ pulse schedule by +0.1481 +/- 0.0101 (t = 14.61) at 3pi/8 and +0.2283 +/- 0.0149 "
                "(t = 15.28, p_Holm = 6.09e-4) at pi/2.</li>"
                "<li>Includes an offline-compiled [[4,2,2]] codespace companion bundle staged for Quantinuum H2-1 "
                "(2,330 HQC).</li></ul>"
            ),
            "keywords": [
                "quantum information scrambling",
                "Hayden-Preskill protocol",
                "Yoshida-Kitaev decoding",
                "non-Clifford circuits",
                "Kraus channel",
                "IBM Heron r2",
                "ibm_fez",
                "preregistration",
                "OSIRIS-XTOP-006",
            ],
        },
    },
    "wp2_27q_debiased_ot_fez": {
        "dir": "papers/wp2_27q_debiased_ot_fez",
        "tex_file": "wp2_27q_debiased_ot_fez.tex",
        "publication_type": "preprint",
        "version": "1.0.0",
        "related": [
            {
                "identifier": "10.5281/zenodo.19864030",
                "relation": "continues",
                "scheme": "doi",
            },
            {
                "identifier": "10.5281/zenodo.23967473",
                "relation": "references",
                "scheme": "doi",
            },
        ],
        "files": [
            ("papers/wp2_27q_debiased_ot_fez/wp2_27q_debiased_ot_fez.tex", None),
            ("papers/wp2_27q_debiased_ot_fez/generated.tex", None),
            ("papers/wp2_27q_debiased_ot_fez/generate_wp2_paper_assets.py", None),
            ("papers/wp2_27q_debiased_ot_fez/fig1_27q_fez_sparsity_collapse_vs_eq7.svg", None),
            ("papers/wp2_27q_debiased_ot_fez/fig1_27q_fez_sparsity_collapse_vs_eq7.pdf", None),
            (
                "papers/wp2_27q_debiased_ot_fez/fig2_coherent_drift_calibration_and_heron_rescoring.svg",
                None,
            ),
            (
                "papers/wp2_27q_debiased_ot_fez/fig2_coherent_drift_calibration_and_heron_rescoring.pdf",
                None,
            ),
            ("papers/wp2_27q_debiased_ot_fez/reproducibility_capsule.json", None),
            ("papers/wp2_27q_debiased_ot_fez/SHA256SUMS.txt", None),
            ("experiments/wp2_27q_fez/PRE_REGISTRATION.md", None),
            ("experiments/wp2_27q_fez/stage_and_harvest_wp2_27q_fez.py", None),
            ("experiments/wp2_27q_fez/physical_wp2_27q_fez_results.json", None),
            ("experiments/wp2_27q_fez/physical_wp2_27q_fez_raw_counts.json.gz", None),
            ("experiments/wp2_27q_fez/wp2_27q_fez_staged_circuits.qpy", None),
            ("experiments/wp2_27q_fez/wp2_27q_fez_staging_manifest.json", None),
            ("experiments/wp2_27q_fez/wp2_27q_fez_calibration_snapshot.json", None),
            ("experiments/wp2_27q_fez/wp2_27q_fez_submitted_job.json", None),
            ("experiments/wp2_debiased_sinkhorn/debiased_ot.py", None),
            ("experiments/wp2_debiased_sinkhorn/artifacts/wp2_debiased_ot_report.json", None),
            ("experiments/osiris_advantage_marrakesh/dau0q3qhcrkc73durtgg.counts.json", None),
            ("experiments/qec001/physical_qec001_fez_raw_counts.json.gz", None),
        ],
        "optional_files": [
            ("papers/wp2_27q_debiased_ot_fez/wp2_27q_debiased_ot_fez.pdf", None),
        ],
        "meta": {
            "title": (
                "Eliminating Finite-Shot Hypercube Sparsity Bias in 27-Qubit Quantum Processor Stability "
                "Benchmarks via Split-Sample-Debiased Optimal Transport"
            ),
            "description": (
                "<p>Preregistered 27-qubit stability benchmark on the IBM Heron r2 processor "
                "<code>ibm_fez</code> (protocol <code>OSIRIS-WP2-27Q-FEZ</code>, job "
                "<code>dc1tqvinj2vs73dfrts0</code>; 48 ISA circuits, 98,304 raw shots, 28 s QPU, constant "
                "52-CZ / 378-SX heavy-hex tree with zero routing SWAPs) paired with multi-width SIMULATED "
                "hypercube benchmarks up to 96 qubits and hardware re-scoring of archived Heron r2 datasets.</p><ul>"
                "<li>Resolves both the refuted legacy Eq. (7) marginal-product self-cost artifact (0.50918 at Q = 27) "
                "and the Weed&ndash;Bach O(S^{-1/Q}) finite-shot sparsity artifact of raw plug-in Kantorovich "
                "optimal transport via exact multivariate-hypergeometric sample-size-matched two-by-two split-sample "
                "Kantorovich and Sinkhorn (epsilon = 0.05) U-statistics.</li>"
                "<li>Across theta in {2.118, 2.618, 3.118} rad on <code>ibm_fez</code>, matched split-sample "
                "Sinkhorn collapses inter-block A/A drift to &le; 0.000076 (&lt; 0.005 threshold, H1 supported), "
                "confirms no special stability advantage at the golden-ratio angle theta* = 2.618 rad (H2 supported), "
                "and detects controlled coherent perturbations Delta-theta = 0.45 rad monotonically at 13.76 sigma "
                "(H3 supported).</li>"
                "<li>On SIMULATED Uniform({0,1}^96), raw plug-in LP suffers a +855.48 sigma false-drift artifact "
                "whereas matched split-sample Sinkhorn yields z = -0.34 sigma; archived <code>ibm_marrakesh</code> "
                "(job <code>dau0q3qhcrkc73durtgg</code>) and 49-bit <code>OSIRIS-QEC-001</code> "
                "(job <code>dbtl2vb2kf8s73c9a0d0</code>) counts corroborate clean separation of hardware drift "
                "from sampling sparsity.</li></ul>"
            ),
            "keywords": [
                "optimal transport",
                "Sinkhorn divergence",
                "finite-sample debiasing",
                "quantum benchmarking",
                "IBM Heron r2",
                "ibm_fez",
                "preregistration",
                "OSIRIS-WP2-27Q-FEZ",
            ],
        },
    },
    "nonabelian_2bga_and_adv1": {
        "dir": "papers/nonabelian_2bga_and_adv1",
        "tex_file": "nonabelian_2bga_and_adv1.tex",
        "publication_type": "preprint",
        "version": "1.0.0",
        "related": [],
        "files": [
            ("papers/nonabelian_2bga_and_adv1/nonabelian_2bga_and_adv1.tex", None),
            ("papers/nonabelian_2bga_and_adv1/generated.tex", None),
            ("papers/nonabelian_2bga_and_adv1/generate_2bga_adv1_assets.py", None),
            (
                "papers/nonabelian_2bga_and_adv1/fig1_odd_k_nonabelian_2bga_codes_and_bposd.svg",
                None,
            ),
            (
                "papers/nonabelian_2bga_and_adv1/fig1_odd_k_nonabelian_2bga_codes_and_bposd.pdf",
                None,
            ),
            (
                "papers/nonabelian_2bga_and_adv1/fig2_adv1_three_attack_ablation_m6.svg",
                None,
            ),
            (
                "papers/nonabelian_2bga_and_adv1/fig2_adv1_three_attack_ablation_m6.pdf",
                None,
            ),
            ("papers/nonabelian_2bga_and_adv1/reproducibility_capsule.json", None),
            ("papers/nonabelian_2bga_and_adv1/SHA256SUMS.txt", None),
            ("qldpc_nonabelian/nonabelian_2bga.py", None),
            ("qldpc_nonabelian/artifacts/nonabelian_2bga_report.json", None),
            ("adv1/looped_peaked.py", None),
            ("adv1/pregate.py", None),
            ("adv1/artifacts/adv1_pregate_report.json", None),
        ],
        "optional_files": [
            ("papers/nonabelian_2bga_and_adv1/nonabelian_2bga_and_adv1.pdf", None),
        ],
        "meta": {
            "title": (
                "Left-Right Rank Duality in Non-Abelian Two-Block Group-Algebra Quantum LDPC Codes and a "
                "Three-Attack Classical-Hardness Pre-Gate for Looped Non-Mirror Peaked Circuits"
            ),
            "description": (
                "<p>Algebraic proofs and reproducible SIMULATED benchmarks (<code>provider_job_id: null</code>) "
                "calibrated to the 98-qubit Quantinuum <code>Helios-1</code> all-to-all QCCD architecture across "
                "non-abelian quantum LDPC coding and pre-submission peaked-circuit classical-hardness gating.</p><ul>"
                "<li>Proves via group-inversion anti-automorphism and Frobenius / Wedderburn&ndash;Artin duality "
                "that unquotiented regular two-block group-algebra (2BGA) CSS codes over F_2[G] satisfy "
                "rank(H_X) = rank(H_Z) identically for every finite group G, forcing even logical dimension k "
                "(0 violations across 6,480 subset pairs over 11 non-abelian groups).</li>"
                "<li>Breaking left-right regularity via subgroup-quotient cosets (including D_9/Z_2 with n = 21, "
                "k = 3) or symplectic minimum-weight gauge pinning yields 16 odd-k (k in {1, 3, 5}, d &ge; 4) "
                "non-abelian CSS codes&mdash;including exact D_9 [[36,3,6]], C_7 x| C_3 [[42,3,6]], and ISD-witnessed "
                "C_19 x| C_3 [[114,3,12]] (kd^2/n = 3.7895)&mdash;benchmarked under log-domain Sum-Product "
                "BP+OSD-CS decoding.</li>"
                "<li>Formulates the fail-closed ADV-1 pre-gate requiring candidate peaked circuits to defeat "
                "bounded-chi MPS (chi &le; 128), truncated Pauli-path (w_max &le; 4), and causal light-cone "
                "(&le; 20 qubits) attacks; appending the 15-CX involution M_6 = J_6 - I_6 (mod 2) eliminates "
                "weight-four Pauli leakage and yields 24-qubit and 30-qubit chorded circuits surviving all three "
                "classical attacks (predicted Helios-1 peaks 0.3908 and 0.3508).</li></ul>"
            ),
            "keywords": [
                "quantum LDPC codes",
                "non-abelian group algebra",
                "two-block group-algebra codes",
                "BP+OSD decoding",
                "peaked circuits",
                "classical hardness pre-gate",
                "Quantinuum Helios",
                "QCCD",
            ],
        },
    },
    "heron_r2_calibration_staleness_and_colored_nulls": {
        "dir": "papers/heron_r2_calibration_staleness_and_colored_nulls",
        "tex_file": "heron_r2_calibration_staleness_and_colored_nulls.tex",
        "publication_type": "article",
        "version": "1.0.0",
        "related": [
            {"identifier": "10.5281/zenodo.24406426", "relation": "compiles", "scheme": "doi"},
            {"identifier": "10.5281/zenodo.24406435", "relation": "compiles", "scheme": "doi"},
            {"identifier": "10.5281/zenodo.24406496", "relation": "compiles", "scheme": "doi"},
            {"identifier": "10.5281/zenodo.24406525", "relation": "compiles", "scheme": "doi"},
            {"identifier": "10.5281/zenodo.24406640", "relation": "compiles", "scheme": "doi"},
            {"identifier": "10.5281/zenodo.24407001", "relation": "compiles", "scheme": "doi"},
            {"identifier": "10.5281/zenodo.24407538", "relation": "compiles", "scheme": "doi"},
            {"identifier": "10.5281/zenodo.24407875", "relation": "compiles", "scheme": "doi"},
            {"identifier": "10.5281/zenodo.24407897", "relation": "compiles", "scheme": "doi"},
            {"identifier": "10.5281/zenodo.24419327", "relation": "compiles", "scheme": "doi"},
            {"identifier": "10.5281/zenodo.24420902", "relation": "compiles", "scheme": "doi"},
        ],
        "files": [
            (
                "papers/heron_r2_calibration_staleness_and_colored_nulls/"
                "heron_r2_calibration_staleness_and_colored_nulls.tex",
                None,
            ),
            ("papers/heron_r2_calibration_staleness_and_colored_nulls/generated.tex", None),
            (
                "papers/heron_r2_calibration_staleness_and_colored_nulls/generate_unified_assets.py",
                None,
            ),
            (
                "papers/heron_r2_calibration_staleness_and_colored_nulls/"
                "fig1_tmap_sign_reversal_three_qpus.pdf",
                None,
            ),
            (
                "papers/heron_r2_calibration_staleness_and_colored_nulls/"
                "fig1_tmap_sign_reversal_three_qpus.svg",
                None,
            ),
            (
                "papers/heron_r2_calibration_staleness_and_colored_nulls/"
                "fig2_ro06_07_staleness_persistence.pdf",
                None,
            ),
            (
                "papers/heron_r2_calibration_staleness_and_colored_nulls/"
                "fig2_ro06_07_staleness_persistence.svg",
                None,
            ),
            (
                "papers/heron_r2_calibration_staleness_and_colored_nulls/"
                "fig3_cn01_colored_noise_heldout_gate.pdf",
                None,
            ),
            (
                "papers/heron_r2_calibration_staleness_and_colored_nulls/"
                "fig3_cn01_colored_noise_heldout_gate.svg",
                None,
            ),
            (
                "papers/heron_r2_calibration_staleness_and_colored_nulls/reproducibility_capsule.json",
                None,
            ),
            ("papers/heron_r2_calibration_staleness_and_colored_nulls/SHA256SUMS.txt", None),
            ("docs/phase3/PHASE3_PREREG_AMENDMENTS.md", None),
        ],
        "optional_files": [
            (
                "papers/heron_r2_calibration_staleness_and_colored_nulls/"
                "heron_r2_calibration_staleness_and_colored_nulls.pdf",
                None,
            ),
        ],
        "meta": {
            "title": (
                "Calibration Staleness, Transient Readout Telegraphing, and Colored-Noise False Positives "
                "Across Three 156-Qubit Superconducting Quantum Processors"
            ),
            "description": (
                "<p>Unified synthesis of 7 preregistered hardware campaigns (42 IBM Quantum jobs, 1,062 PUBs, "
                "2,174,976 physical circuit shots, 673 s QPU) across all three 156-qubit IBM Heron r2 processors "
                "(<code>ibm_marrakesh</code>, <code>ibm_fez</code>, and <code>ibm_kingston</code>).</p><ul>"
                "<li>In 40-qubit compilation benchmarks at fixed 39-CZ depth 66 (<code>TMAP-02</code>&ndash;"
                "<code>TMAP-05</code>), calibration-ranked vs median path fidelity reversed from +10.54 pp to "
                "&minus;4.29 pp four days later due to an unflagged q46 readout excursion (35.0% actual vs 1.1% "
                "published), and repeating a pre-job readout-probe selector 82&ndash;85 minutes apart flipped "
                "the probe-vs-calibration sign on all three processors while tensored readout mitigation restored "
                "calibration-path superiority on all three.</li>"
                "<li>Longitudinal monitoring and isolation experiments (<code>RO-06</code> and <code>RO-07</code>, "
                "27 jobs, 516,096 shots) show hidden faults (&gt;15% actual vs &lt;3% listed) are transient "
                "(6 of 6 recovered below 5%), persist across 15-minute rounds at 87.4% (Fisher OR 1,095), and "
                "exhibit &le;1.16 pp multiplexing shift (&le;1.61 pp neighbor state-flip shift), tracing reversals "
                "to calibration snapshots static up to 16.0 h rather than readout crosstalk.</li>"
                "<li>Multiplexed DD spectroscopy across 10 couplers &ge;5 hops apart (<code>CN-01</code>, 499,712 "
                "shots) gave null mechanistic contrasts (C_DFS = +0.43 pp, p_Holm = 0.47) while white-noise-only "
                "GLS falsely favored a non-local OU bath by Delta-BIC = 10,263 before failing its preregistered "
                "6-coupler held-out gate (p = 0.82); a 550-replicate synthetic audit proves white-noise nulls "
                "misclassify 100% of local 1/f dephasing whereas local colored-noise nulls cut false non-local "
                "selection to 0%.</li></ul>"
            ),
            "keywords": [
                "quantum benchmarking",
                "calibration staleness",
                "readout error",
                "telegraph noise",
                "colored dephasing",
                "dynamical decoupling",
                "IBM Heron r2",
                "ibm_marrakesh",
                "ibm_fez",
                "ibm_kingston",
                "preregistration",
            ],
        },
    },
    "bt2t001_binary_tetrahedral_fez": {
        "dir": "papers/bt2t001_binary_tetrahedral_fez",
        "tex_file": "bt2t001_binary_tetrahedral_fez.tex",
        "publication_type": "preprint",
        "version": "1.0.0",
        "related": [
            {
                "identifier": "10.5281/zenodo.24420902",
                "relation": "references",
                "scheme": "doi",
            },
            {
                "identifier": "10.5281/zenodo.24406426",
                "relation": "references",
                "scheme": "doi",
            },
        ],
        "files": [
            ("papers/bt2t001_binary_tetrahedral_fez/bt2t001_binary_tetrahedral_fez.tex", None),
            ("papers/bt2t001_binary_tetrahedral_fez/generated.tex", None),
            ("papers/bt2t001_binary_tetrahedral_fez/generate_bt2t001_paper_assets.py", None),
            (
                "papers/bt2t001_binary_tetrahedral_fez/"
                "fig1_bt2t001_fez_synthesis_and_sectors.svg",
                None,
            ),
            (
                "papers/bt2t001_binary_tetrahedral_fez/"
                "fig1_bt2t001_fez_synthesis_and_sectors.pdf",
                None,
            ),
            (
                "papers/bt2t001_binary_tetrahedral_fez/"
                "fig2_bt2t001_fez_elementwise_and_spam_drift.svg",
                None,
            ),
            (
                "papers/bt2t001_binary_tetrahedral_fez/"
                "fig2_bt2t001_fez_elementwise_and_spam_drift.pdf",
                None,
            ),
            ("papers/bt2t001_binary_tetrahedral_fez/reproducibility_capsule.json", None),
            ("papers/bt2t001_binary_tetrahedral_fez/SHA256SUMS.txt", None),
            ("experiments/bt2t_fez/PRE_REGISTRATION.md", None),
            ("experiments/bt2t_fez/stage_and_harvest_bt2t_fez.py", None),
            ("experiments/bt2t_fez/physical_bt2t_fez_results.json", None),
            ("experiments/bt2t_fez/physical_bt2t_fez_raw_counts.json.gz", None),
            ("experiments/bt2t_fez/bt2t_fez_staged_circuits.qpy", None),
            ("experiments/bt2t_fez/bt2t_fez_staging_manifest.json", None),
            ("experiments/bt2t_fez/bt2t_fez_calibration_snapshot.json", None),
            ("experiments/bt2t_fez/bt2t_fez_submitted_job.json", None),
            ("experiments/bt2t_inversion/G1_INDEPENDENT_REVIEW.md", None),
            ("experiments/bt2t_inversion/bt2t.py", None),
            ("experiments/bt2t_inversion/verify.py", None),
            ("experiments/bt2t_inversion/manifest.py", "bt2t_inversion_manifest.py"),
            ("experiments/bt2t_inversion/manifest.json", "bt2t_inversion_manifest.json"),
        ],
        "optional_files": [
            ("papers/bt2t001_binary_tetrahedral_fez/bt2t001_binary_tetrahedral_fez.pdf", None),
        ],
        "meta": {
            "title": (
                "Exact Binary Tetrahedral Group Inversion on a 5-Qubit Heavy-Hex Superconducting "
                "Register: Coset-Parity Synthesis Compression and Unflagged Readout Staleness "
                "(OSIRIS-BT2T-001-FEZ)"
            ),
            "description": (
                "<p>Write-ahead preregistered benchmark (<code>OSIRIS-BT2T-001-FEZ</code>, "
                "<code>Card A</code>; independent gate <code>G1</code> signed off prior to "
                "submission) of exact non-abelian group inversion <code>g -&gt; g^-1</code> in "
                "the 24-element binary tetrahedral group <code>2T = Q_8 |x Z_3</code> embedded in "
                "a 5-qubit register (frozen encoding <code>bt2t-enc-1</code>) on a five-transmon "
                "heavy-hex T-junction <code>[21, 22, 23, 16, 24]</code> of the 156-qubit IBM "
                "Heron r2 processor <code>ibm_fez</code> (job <code>dc2ee3uq4uuc73a56o6g</code>; "
                "<code>HARDWARE_ATTESTED</code>, 288 ISA circuits, 27,648 equal-shot physical "
                "measurements across 3 interleaved randomized blocks, 11 s QPU, "
                "<code>N_windows = 1</code>).</p><ul>"
                "<li>Coset-parity Boolean factoring over <code>2T = Q_8 |x Z_3</code> compresses "
                "heavy-hex <code>CZ</code> count 10.04x from 723 <code>CZ</code> (1,560 "
                "<code>sqrt(X)</code>, unitary depth 2,073) in the generic 11-transposition "
                "reference to 72 <code>CZ</code> (151 <code>sqrt(X)</code>, unitary depth 218) "
                "for the identical 32x32 permutation unitary, and compresses local "
                "<code>pytket</code> <code>H2</code>-native compilation from 400 to 47 "
                "<code>ZZPhase</code> gates (8.51x).</li>"
                "<li>Uniform valid-24 raw inversion accuracy rises from 0.0661 +/- 0.0030 "
                "to 0.5337 +/- 0.0059 (0.6104 readout-mitigated, <code>cond(M_32) = "
                "1.4524</code>), yielding Delta P_valid24 = +0.4676 (paired block t = 73.23, "
                "one-sided p = 9.32e-5; 24/24 elementwise wins, Wilcoxon p = 1.19e-7; "
                "<code>H1 = SUPPORTED</code>) and 32/32 exact modal majority-vote recovery.</li>"
                "<li>Unused-codeword fixed-point preservation rises by +0.4818 (0.5595 vs 0.0777) "
                "while valid-to-unused subspace leakage drops to 0.0940 &lt; 0.15 below the 0.25 "
                "uniform random baseline (<code>H2 = SUPPORTED</code>).</li>"
                "<li>The interleaved 0-<code>CZ</code> SPAM control achieves 0.9076 &ge; 0.90 on "
                "unused-8 but 0.8643 &lt; 0.90 on valid-24, honestly refuting <code>H3</code> "
                "(<code>REFUTED</code>) and isolating an unflagged 14.91x live readout staleness "
                "excursion on leaf transmon <code>q24</code> (0.33% published vs 4.92% live) and "
                "a 2.70x excursion on <code>q21</code> (2.34% vs 6.33%).</li></ul>"
            ),
            "keywords": [
                "binary tetrahedral group",
                "non-abelian group primitives",
                "quantum circuit synthesis",
                "lattice gauge theory",
                "calibration staleness",
                "IBM Heron r2",
                "ibm_fez",
                "preregistration",
                "OSIRIS-BT2T-001-FEZ",
            ],
        },
    },
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_bytes(obj: Any) -> bytes:
    """Deterministic UTF-8 JSON canonicalization (RFC 8785 JCS compatible for string/int/bool/null trees)."""
    try:
        import rfc8785  # type: ignore[import-not-found]

        return rfc8785.dumps(obj)
    except ImportError:
        return json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")


def _strip_html(html_text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", html_text)
    cleaned = (
        cleaned.replace("&ndash;", "-")
        .replace("&mdash;", "--")
        .replace("&minus;", "-")
        .replace("&le;", "<=")
        .replace("&ge;", ">=")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&times;", "x")
    )
    return re.sub(r"\s+", " ", cleaned).strip()


def _parse_latex_macros(gen_tex: str) -> Dict[str, str]:
    macros: Dict[str, str] = {}
    pattern = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{")
    pos = 0
    while True:
        m = pattern.search(gen_tex, pos)
        if not m:
            break
        name = m.group(1)
        idx = m.end()
        depth = 1
        start = idx
        while idx < len(gen_tex) and depth > 0:
            ch = gen_tex[idx]
            if ch == "\\" and idx + 1 < len(gen_tex):
                idx += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            idx += 1
        if depth == 0:
            macros[name] = gen_tex[start : idx - 1]
        pos = idx
    return macros


def _expand_latex_macros(tex: str, macros: Mapping[str, str]) -> str:
    out = tex
    for name in sorted(macros.keys(), key=len, reverse=True):
        val = macros[name]
        out = re.sub(rf"\\{name}(?:\{{\}}|\b)", lambda _m, v=val: v, out)
    return out


def verify_paper_sha256sums(
    paper_key: str,
    cfg: Mapping[str, Any],
    root: Path = ROOT,
) -> Dict[str, Any]:
    """Verify ``SHA256SUMS.txt`` and ``reproducibility_capsule.json`` before staging."""
    paper_dir = root / str(cfg["dir"])
    sums_path = paper_dir / "SHA256SUMS.txt"
    if not sums_path.is_file():
        raise ZenodoPublishError(
            "MISSING_SHA256SUMS",
            f"{paper_key}: missing lockfile {sums_path}",
        )

    verified_entries: Dict[str, str] = {}
    for raw_line in sums_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        parts = stripped.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ZenodoPublishError(
                "MALFORMED_SHA256SUMS",
                f"{paper_key}: malformed SHA256SUMS line {raw_line!r}",
            )
        expected_sha, rel_ref = parts
        candidate = paper_dir / rel_ref if (paper_dir / rel_ref).is_file() else root / rel_ref
        if not candidate.is_file():
            raise ZenodoPublishError(
                "MISSING_LOCKED_FILE",
                f"{paper_key}: missing file {rel_ref!r} referenced in SHA256SUMS.txt",
            )
        actual_sha = sha256_file(candidate)
        if actual_sha != expected_sha:
            raise ZenodoPublishError(
                "SHA256_MISMATCH",
                f"{paper_key}: SHA-256 mismatch for {rel_ref!r} "
                f"(expected {expected_sha}, got {actual_sha})",
            )
        norm_rel = candidate.resolve().relative_to(root.resolve()).as_posix()
        verified_entries[norm_rel] = actual_sha

    if not verified_entries:
        raise ZenodoPublishError(
            "EMPTY_SHA256SUMS",
            f"{paper_key}: SHA256SUMS.txt contains zero entries",
        )

    capsule_path = paper_dir / "reproducibility_capsule.json"
    if not capsule_path.is_file():
        raise ZenodoPublishError(
            "MISSING_CAPSULE",
            f"{paper_key}: missing reproducibility_capsule.json at {capsule_path}",
        )
    capsule = json.loads(capsule_path.read_text(encoding="utf-8"))
    if capsule.get("schema") != "osiris.reproducibility_capsule/1":
        raise ZenodoPublishError(
            "INVALID_CAPSULE_SCHEMA",
            f"{paper_key}: unexpected capsule schema {capsule.get('schema')!r}",
        )
    for digest_key in ("rfc8785_canonical_sha256", "capsule_body_sha256"):
        if digest_key in capsule:
            body = {k: v for k, v in capsule.items() if k != digest_key}
            recomputed = sha256_bytes(canonical_json_bytes(body))
            if recomputed != capsule[digest_key]:
                raise ZenodoPublishError(
                    "CAPSULE_HASH_MISMATCH",
                    f"{paper_key}: {digest_key} mismatch (expected {capsule[digest_key]}, got {recomputed})",
                )

    return {
        "passed": True,
        "verified_count": len(verified_entries),
        "sha256sums_sha256": sha256_file(sums_path),
        "capsule_sha256": sha256_file(capsule_path),
    }


def verify_paper_sys024(
    paper_key: str,
    cfg: Mapping[str, Any],
    root: Path = ROOT,
) -> Dict[str, Any]:
    """Run SYS-024 ``claims.flagged_in_reply`` and banned-constant checks on manuscript & metadata."""
    paper_dir = root / str(cfg["dir"])
    tex_path = paper_dir / str(cfg["tex_file"])
    gen_path = paper_dir / "generated.tex"
    if not tex_path.is_file() or not gen_path.is_file():
        raise ZenodoPublishError(
            "MISSING_MANUSCRIPT_SOURCE",
            f"{paper_key}: missing {tex_path.name} or generated.tex in {paper_dir}",
        )

    raw_tex = tex_path.read_text(encoding="utf-8")
    gen_tex = gen_path.read_text(encoding="utf-8")
    macros = _parse_latex_macros(gen_tex)
    expanded_tex = _expand_latex_macros(raw_tex, macros)

    meta = cfg["meta"]
    meta_text = f"{meta['title']}\n{_strip_html(meta['description'])}\n" + ", ".join(meta["keywords"])

    surfaces: Tuple[Tuple[str, str], ...] = (
        ("raw_tex", raw_tex),
        ("generated_tex", gen_tex),
        ("expanded_tex", expanded_tex),
        ("zenodo_metadata", meta_text),
    )
    for surface_name, content in surfaces:
        flagged = claims.flagged_in_reply(content, [])
        if flagged:
            ids = [c.id for c in flagged]
            raise ZenodoPublishError(
                "SYS024_FLAGGED_CLAIM",
                f"{paper_key} ({surface_name}): SYS-024 flagged un-negated claim(s): {ids}",
            )
        for const_label, pattern in plan_gates._QUARANTINED_PLAN_PATTERNS:
            if pattern.search(content):
                raise ZenodoPublishError(
                    "SYS024_QUARANTINED_CONSTANT",
                    f"{paper_key} ({surface_name}): banned SYS-024 constant {const_label!r} matched",
                )

    return {
        "passed": True,
        "flagged_claims": [],
        "quarantined_constants": [],
        "surfaces_checked": [name for name, _ in surfaces],
    }


def build_deposition_metadata(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    meta = dict(cfg["meta"])
    pub_type = str(cfg.get("publication_type", "preprint"))
    if pub_type not in ("preprint", "article"):
        raise ZenodoPublishError(
            "INVALID_PUBLICATION_TYPE",
            f"Unsupported Zenodo publication_type {pub_type!r}",
        )
    related_ids: List[Dict[str, str]] = [
        {
            "identifier": "https://github.com/quantum-osiris/osiris-cli",
            "relation": "isSupplementedBy",
            "scheme": "url",
        }
    ] + [dict(item) for item in cfg.get("related", [])]
    return {
        "upload_type": "publication",
        "publication_type": pub_type,
        "publication_date": PUBLICATION_DATE,
        "creators": [dict(c) for c in CREATORS],
        "access_right": "open",
        "license": "cc-by-4.0",
        "version": str(cfg.get("version", "1.0.0")),
        "language": "eng",
        "title": str(meta["title"]),
        "description": str(meta["description"]),
        "keywords": list(meta["keywords"]),
        "related_identifiers": related_ids,
    }


def stage_paper_bundle(
    paper_key: str,
    cfg: Mapping[str, Any],
    root: Path = ROOT,
) -> Dict[str, Any]:
    """Verify SYS-024 + SHA256SUMS and return a deterministic Zenodo staging record for ``paper_key``."""
    sha_report = verify_paper_sha256sums(paper_key, cfg, root=root)
    sys024_report = verify_paper_sys024(paper_key, cfg, root=root)
    metadata = build_deposition_metadata(cfg)

    staged_files: List[Dict[str, Any]] = []
    seen_upload_names: set[str] = set()

    for rel_path, custom_name in cfg["files"]:
        full_path = root / rel_path
        if not full_path.is_file():
            raise ZenodoPublishError(
                "MISSING_STAGED_FILE",
                f"{paper_key}: required file missing at {rel_path}",
            )
        upload_name = custom_name or Path(rel_path).name
        if upload_name in seen_upload_names:
            raise ZenodoPublishError(
                "DUPLICATE_UPLOAD_NAME",
                f"{paper_key}: duplicate Zenodo filename {upload_name!r}",
            )
        seen_upload_names.add(upload_name)
        raw = full_path.read_bytes()
        staged_files.append(
            {
                "upload_name": upload_name,
                "source_relpath": Path(rel_path).as_posix(),
                "size_bytes": len(raw),
                "sha256": sha256_bytes(raw),
                "required": True,
            }
        )

    pdf_compiled = False
    for rel_path, custom_name in cfg.get("optional_files", []):
        full_path = root / rel_path
        if not full_path.is_file():
            continue
        upload_name = custom_name or Path(rel_path).name
        if upload_name in seen_upload_names:
            raise ZenodoPublishError(
                "DUPLICATE_UPLOAD_NAME",
                f"{paper_key}: duplicate optional Zenodo filename {upload_name!r}",
            )
        seen_upload_names.add(upload_name)
        raw = full_path.read_bytes()
        if upload_name == f"{paper_key}.pdf":
            pdf_compiled = True
        staged_files.append(
            {
                "upload_name": upload_name,
                "source_relpath": Path(rel_path).as_posix(),
                "size_bytes": len(raw),
                "sha256": sha256_bytes(raw),
                "required": False,
            }
        )

    bundle_core = {
        "paper_key": paper_key,
        "paper_dir": str(cfg["dir"]),
        "pdf_compiled": pdf_compiled,
        "sha256sums_check": sha_report,
        "sys024_check": sys024_report,
        "metadata": metadata,
        "files": staged_files,
        "total_files": len(staged_files),
        "total_bytes": sum(int(f["size_bytes"]) for f in staged_files),
    }
    bundle_core["bundle_sha256"] = sha256_bytes(canonical_json_bytes(bundle_core))
    return bundle_core


def resolve_selected_papers(paper_arg: str) -> List[str]:
    key = paper_arg.strip()
    if key == "all":
        return list(PAPER_ORDER)
    mapped = PAPER_ALIASES.get(key)
    if not mapped:
        valid = ", ".join(["all"] + list(PAPER_ORDER))
        raise ZenodoPublishError("UNKNOWN_PAPER", f"Unknown --paper {paper_arg!r} (valid: {valid})")
    return [mapped]


def build_staging_manifest(
    paper_keys: Sequence[str] = PAPER_ORDER,
    root: Path = ROOT,
    manifest_out: Optional[Path] = DEFAULT_STAGING_MANIFEST_PATH,
) -> Dict[str, Any]:
    """Build deterministic ``zenodo_staging_manifest_nov23.json`` with zero network I/O."""
    papers_payload: Dict[str, Any] = {}
    for key in paper_keys:
        if key not in PAPERS:
            raise ZenodoPublishError("UNKNOWN_PAPER", f"Unregistered paper key {key!r}")
        papers_payload[key] = stage_paper_bundle(key, PAPERS[key], root=root)

    manifest_body: Dict[str, Any] = {
        "schema": STAGING_MANIFEST_SCHEMA,
        "publication_date": PUBLICATION_DATE,
        "mode": "DRY_RUN_STAGED",
        "network_calls": 0,
        "paper_count": len(papers_payload),
        "paper_order": list(paper_keys),
        "papers": papers_payload,
    }
    manifest_sha = sha256_bytes(canonical_json_bytes(manifest_body))
    manifest = {**manifest_body, "manifest_sha256": manifest_sha}

    if manifest_out is not None:
        manifest_out = Path(manifest_out)
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        manifest_out.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return manifest


def _resolve_zenodo_token(env: Optional[Mapping[str, str]] = None) -> str:
    env_map = env if env is not None else os.environ
    for key in ("ZENODO_TOKEN", "ZENODO_API_TOKEN"):
        val = env_map.get(key, "").strip()
        if val:
            return val
    raise ZenodoPublishError(
        "MISSING_ZENODO_TOKEN",
        "--execute requires ZENODO_TOKEN in the environment (fail-closed; 0 network calls made)",
    )


def _confirm_operator_publish(confirm_flag: bool, interactive: bool = False) -> bool:
    if confirm_flag:
        return True
    if interactive and sys.stdin.isatty():
        try:
            ans = input("Upload staged Nov 23 Zenodo deposition(s)? Type 'CONFIRM' to proceed: ")
        except EOFError:
            return False
        return ans.strip() == "CONFIRM"
    return False


def _zenodo_http_call_once(
    method: str,
    url: str,
    token: str,
    data: Optional[Dict[str, Any]] = None,
    raw: Optional[bytes] = None,
    ctype: str = "application/json",
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
            return exc.code, {"raw": txt[:800]}


def _zenodo_http_call(
    method: str,
    url: str,
    token: str,
    data: Optional[Dict[str, Any]] = None,
    raw: Optional[bytes] = None,
    ctype: str = "application/json",
) -> Tuple[int, Dict[str, Any]]:
    last: Tuple[int, Dict[str, Any]] = (0, {})
    for attempt in range(5):
        try:
            st, body = _zenodo_http_call_once(method, url, token, data=data, raw=raw, ctype=ctype)
        except Exception as exc:
            st, body = 0, {"raw": f"{type(exc).__name__}: {exc}"}
        if st not in (0, 429, 500, 502, 503, 504):
            return st, body
        last = (st, body)
        time.sleep(min(30, 2 ** attempt))
    return last


def execute_zenodo_upload(
    paper_keys: Sequence[str] = PAPER_ORDER,
    root: Path = ROOT,
    manifest_out: Optional[Path] = DEFAULT_STAGING_MANIFEST_PATH,
    confirm_publish: bool = False,
    interactive: bool = False,
    finalize_publish: bool = False,
    receipt_dir: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    http_call: Optional[HttpCallFn] = None,
) -> Dict[str, Any]:
    """Upload verified Zenodo depositions (fail-closed without ZENODO_TOKEN and confirmation)."""
    # Gate 1: Credentials & explicit operator confirmation checked BEFORE any network call
    token = _resolve_zenodo_zenodo_token_checked(env)
    if not _confirm_operator_publish(confirm_publish, interactive=interactive):
        raise ZenodoPublishError(
            "MISSING_PUBLISH_CONFIRMATION",
            "--execute requires explicit --confirm-publish (or --interactive TTY 'CONFIRM'); "
            "0 network calls made",
        )

    # Gate 2: Full SYS-024 + SHA256SUMS + capsule verification before touching Zenodo API
    manifest = build_staging_manifest(paper_keys=paper_keys, root=root, manifest_out=manifest_out)
    caller = http_call if http_call is not None else _zenodo_http_call

    receipts: Dict[str, Any] = {}
    for key in paper_keys:
        staged = manifest["papers"][key]
        out_dir = Path(receipt_dir) if receipt_dir is not None else (root / str(PAPERS[key]["dir"]))
        receipt_path = out_dir / f"zenodo_receipt_{key}.json"

        st, created = caller("POST", ZENODO_API_URL, token, {}, None, "application/json")
        if st != 201:
            raise ZenodoPublishError("ZENODO_CREATE_FAILED", f"{key}: HTTP {st}: {created}")
        dep_id = created["id"]
        bucket = created["links"]["bucket"]

        st, meta_resp = caller(
            "PUT",
            f"{ZENODO_API_URL}/{dep_id}",
            token,
            {"metadata": staged["metadata"]},
            None,
            "application/json",
        )
        if st != 200:
            raise ZenodoPublishError("ZENODO_METADATA_FAILED", f"{key}: HTTP {st}: {meta_resp}")

        uploaded_hashes: Dict[str, str] = {}
        uploaded_sources: Dict[str, str] = {}
        for fentry in staged["files"]:
            src_path = root / fentry["source_relpath"]
            raw_bytes = src_path.read_bytes()
            if sha256_bytes(raw_bytes) != fentry["sha256"]:
                raise ZenodoPublishError(
                    "CONCURRENT_FILE_MUTATION",
                    f"{key}: local file {fentry['source_relpath']} mutated after staging check",
                )
            up_name = fentry["upload_name"]
            st, up_resp = caller(
                "PUT",
                f"{bucket}/{up_name}",
                token,
                None,
                raw_bytes,
                "application/octet-stream",
            )
            if st not in (200, 201) or int(up_resp.get("size", -1)) != len(raw_bytes):
                raise ZenodoPublishError(
                    "ZENODO_UPLOAD_FAILED",
                    f"{key}: upload {up_name} failed HTTP {st}: {up_resp}",
                )
            uploaded_hashes[up_name] = fentry["sha256"]
            uploaded_sources[up_name] = fentry["source_relpath"]

        state = "draft"
        doi = (
            meta_resp.get("metadata", {})
            .get("prereserve_doi", {})
            .get("doi")
        )
        record_html = created.get("links", {}).get("html")
        if finalize_publish:
            st, pub_resp = caller(
                "POST",
                f"{ZENODO_API_URL}/{dep_id}/actions/publish",
                token,
                None,
                None,
                "application/json",
            )
            if st != 202:
                raise ZenodoPublishError("ZENODO_PUBLISH_FAILED", f"{key}: HTTP {st}: {pub_resp}")
            state = "published"
            doi = pub_resp.get("doi", doi)
            record_html = pub_resp.get("links", {}).get("record_html", record_html)

        rec_obj = {
            "paper": key,
            "deposition_id": dep_id,
            "state": state,
            "doi": doi,
            "record_url": record_html,
            "publication_date": PUBLICATION_DATE,
            "bundle_sha256": staged["bundle_sha256"],
            "files": uploaded_hashes,
            "sources": uploaded_sources,
            "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(rec_obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        receipts[key] = rec_obj

    return {
        "mode": "EXECUTED",
        "manifest_sha256": manifest["manifest_sha256"],
        "receipts": receipts,
    }


def _resolve_zenodo_zenodo_token_checked(env: Optional[Mapping[str, str]] = None) -> str:
    return _resolve_zenodo_token(env=env)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage (--dry-run, default) or upload (--execute) the 4 Nov 23 OSIRIS Zenodo bundles"
    )
    parser.add_argument(
        "--paper",
        default="all",
        help="Manuscript key to stage/upload, or 'all' (default: all 4 Nov 23 papers)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Offline staging only: SYS-024 + SHA256SUMS checks + zenodo_staging_manifest_nov23.json (default)",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Upload to Zenodo API (fail-closed without ZENODO_TOKEN and --confirm-publish)",
    )
    parser.add_argument(
        "--confirm-publish",
        action="store_true",
        help="Explicit operator confirmation required alongside --execute",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Allow interactive TTY 'CONFIRM' prompt when --confirm-publish is omitted",
    )
    parser.add_argument(
        "--finalize-publish",
        action="store_true",
        help="With --execute: invoke /actions/publish after draft upload and size verification",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=DEFAULT_STAGING_MANIFEST_PATH,
        help="Output path for deterministic zenodo_staging_manifest_nov23.json",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        selected = resolve_selected_papers(args.paper)
        if not args.execute:
            manifest = build_staging_manifest(
                paper_keys=selected,
                root=ROOT,
                manifest_out=args.manifest_out,
            )
            summary = {
                "status": "DRY_RUN_VERIFIED",
                "network_calls": 0,
                "publication_date": manifest["publication_date"],
                "paper_count": manifest["paper_count"],
                "manifest_out": str(args.manifest_out),
                "manifest_sha256": manifest["manifest_sha256"],
                "papers": {
                    k: {
                        "publication_type": v["metadata"]["publication_type"],
                        "total_files": v["total_files"],
                        "bundle_sha256": v["bundle_sha256"],
                        "sys024_passed": v["sys024_check"]["passed"],
                        "sha256sums_passed": v["sha256sums_check"]["passed"],
                    }
                    for k, v in manifest["papers"].items()
                },
            }
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0

        res = execute_zenodo_upload(
            paper_keys=selected,
            root=ROOT,
            manifest_out=args.manifest_out,
            confirm_publish=args.confirm_publish,
            interactive=args.interactive,
            finalize_publish=args.finalize_publish,
        )
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0
    except ZenodoPublishError as exc:
        print(f"ZENODO_GATE_BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
