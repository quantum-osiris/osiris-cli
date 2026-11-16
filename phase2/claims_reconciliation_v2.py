"""
Phase 2 Section 1 & Section 2: State Restoration Inventory & Claims Ledger Repair
=================================================================================

Reads `/home/osiris/claims_ledger.jsonl` (44 Phase 1 rows, immutable) and produces:
  - `docs/phase2/PHASE2_STATE_INVENTORY.json`
  - `docs/phase2/claims_ledger_phase2_proposed.jsonl`
  - Verifies Correction A (NB-11 W2 separation), Correction B (NB-07 Bell state vs
    gate fidelity like-for-like comparison), Correction C (structured 5-field
    audit on every MEASURED & VERIFIED row), STAGGERED_DD structured points, and
    Tesseract namespace disambiguation, without modifying `/home/osiris/claims_ledger.jsonl`
    or `osiris_cli/claims.py`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from osiris_cli.workflows.review_coverage import (
    TESSERACT_NAMESPACE_REGISTRY,
    build_canonical_staggered_dd_record,
)

SECTION1_TARGETS: List[tuple[str, str]] = [
    ("CLAIM_RECONCILIATION.md", "/home/osiris/CLAIM_RECONCILIATION.md"),
    ("PHASE2_REPAIRS.md", "/home/osiris/PHASE2_REPAIRS.md"),
    ("claims_ledger.jsonl", "/home/osiris/claims_ledger.jsonl"),
    ("build_ledger.py", "/home/osiris/build_ledger.py"),
    ("audit_tesseract.py", "/home/osiris/audit_tesseract.py"),
    ("OSIRIS_Signal_Noise_Forensic_Audit_2026-11-16.md", "/home/osiris/OSIRIS_Signal_Noise_Forensic_Audit_2026-11-16.md"),
    ("tesseract_rerun_2026-09-20.zip", "/home/osiris/tesseract_rerun_2026-09-20.zip"),
    ("qbyte_substrate_engine_darpa - Copy.zip", "/home/osiris/qbyte_substrate_engine_darpa - Copy.zip"),
    ("qbyte_substrate_engine_darpa.zip", "/home/osiris/qbyte_substrate_engine_darpa.zip"),
    ("w2_robustness_paper.tex", "/home/osiris/w2_robustness_paper.tex"),
    ("coaxial_t1_experiment.py", "/home/osiris/coaxial_t1_experiment.py"),
    ("coaxial_t1_test_v2.py", "/home/osiris/coaxial_t1_test_v2.py"),
    ("coaxial_t1_pilot_results.json", "/home/osiris/coaxial_t1_pilot_results.json"),
    ("coaxial_t1_pilot_results_device_length.json", "/home/osiris/coaxial_t1_pilot_results_device_length.json"),
    ("aeterna-porta-v2-main.zip", "/home/osiris/aeterna-porta-v2-main.zip"),
    ("electrogravitic-unified-physics-main.zip", "/home/osiris/electrogravitic-unified-physics-main.zip"),
    ("FERMI-quantum-coherence-modulation-systems-main.zip", "/home/osiris/FERMI-quantum-coherence-modulation-systems-main.zip"),
    ("EXPERIMENT_E_FRONTIER_REPORT.md", "/home/osiris/osiris-assessment/work/osiris-cli-phase2/experiments/experiment_e/EXPERIMENT_E_FRONTIER_REPORT.md"),
    ("analysis_exp_e2_clustered.json", "/home/osiris/osiris-assessment/work/osiris-cli-phase2/experiments/experiment_e/analysis_exp_e2_clustered.json"),
    ("analysis_exp_e3_pilot.json", "/home/osiris/osiris-assessment/work/osiris-cli-phase2/experiments/experiment_e/analysis_exp_e3_pilot.json"),
    ("analysis_exp_e3_fez.json", "/home/osiris/osiris-assessment/work/osiris-cli-phase2/experiments/experiment_e/analysis_exp_e3_fez.json"),
    ("analysis_exp_e4.json", "/home/osiris/osiris-assessment/work/osiris-cli-phase2/experiments/experiment_e/analysis_exp_e4.json"),
    ("osiris_cli/claims.py", "/home/osiris/osiris-assessment/work/osiris-cli-phase2/osiris_cli/claims.py"),
]


LIKE_FOR_LIKE_BELL_STATE_BENCHMARKS: List[Dict[str, Any]] = [
    {
        "reference": "Benhelm et al., Nat. Phys. 4, 463 (2008)",
        "doi": "10.1038/nphys961",
        "platform": "40Ca+ trapped ions",
        "quantity_measured": "Bell-state preparation fidelity F_Bell",
        "state_prep": "100 us bichromatic Molmer-Sorensen entangling gate on 2 ions",
        "tomography_estimator": "Direct Z-basis population P(00)+P(11) + parity oscillation contrast",
        "spam_corrected": False,
        "fidelity": 0.993,
        "uncertainty": 0.001,
        "exceeds_0_9787_sigma": round((0.993 - 0.978701) / 0.001, 1),
    },
    {
        "reference": "Ballance et al., Phys. Rev. Lett. 117, 060504 (2016)",
        "doi": "10.1103/PhysRevLett.117.060504",
        "platform": "43Ca+ hyperfine clock ions",
        "quantity_measured": "Bell-state preparation fidelity (raw & SPAM-adjusted reported separately)",
        "state_prep": "Raman geometric-phase gate preparing Bell state",
        "tomography_estimator": "Population + parity fringe contrast (raw Bell fidelity ~0.9973(10) including 1.7e-3 SPAM; SPAM-deducted Bell fidelity 0.999(1))",
        "spam_corrected": False,
        "fidelity": 0.9973,
        "spam_adjusted_fidelity": 0.999,
        "uncertainty": 0.001,
        "exceeds_0_9787_sigma": round((0.9973 - 0.978701) / 0.001, 1),
    },
    {
        "reference": "Gaebler et al., Phys. Rev. Lett. 117, 060505 (2016)",
        "doi": "10.1103/PhysRevLett.117.060505",
        "platform": "9Be+ trapped ions",
        "quantity_measured": "Bell-state fidelity (raw & SPAM-adjusted)",
        "state_prep": "Deterministic Molmer-Sorensen Bell-state preparation",
        "tomography_estimator": "Population + parity oscillation (raw Bell fidelity 0.9955(5) including SPAM; SPAM-deducted 0.9991(4))",
        "spam_corrected": False,
        "fidelity": 0.9955,
        "uncertainty": 0.0005,
        "exceeds_0_9787_sigma": round((0.9955 - 0.978701) / 0.0005, 1),
    },
    {
        "reference": "Clark et al., Phys. Rev. Lett. 127, 130505 (2021)",
        "doi": "10.1103/PhysRevLett.127.130505",
        "platform": "40Ca+ trapped ions",
        "quantity_measured": "Raw Bell-state preparation fidelity (no SPAM subtraction)",
        "state_prep": "Laser-driven Molmer-Sorensen gate",
        "tomography_estimator": "Direct population + parity contrast without SPAM correction",
        "spam_corrected": False,
        "fidelity": 0.9994,
        "uncertainty": 0.0003,
        "exceeds_0_9787_sigma": round((0.9994 - 0.978701) / 0.0003, 1),
    },
    {
        "reference": "Ding et al., Phys. Rev. X 13, 031035 (2023)",
        "doi": "10.1103/PhysRevX.13.031035",
        "platform": "Fluxonium superconducting transmon-like circuit",
        "quantity_measured": "Bell-state preparation fidelity (superconducting hardware)",
        "state_prep": " Microwave-activated CZ preparing Bell state",
        "tomography_estimator": "Quantum state tomography (raw 0.985+, SPAM-mitigated 0.992+)",
        "spam_corrected": False,
        "fidelity": 0.985,
        "uncertainty": 0.002,
        "exceeds_0_9787_sigma": round((0.985 - 0.978701) / 0.002, 1),
    },
]


def build_state_inventory() -> Dict[str, Any]:
    items = []
    seen_hashes: Dict[str, str] = {}
    for label, path_str in SECTION1_TARGETS:
        p = Path(path_str)
        if p.exists() and p.is_file():
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            duplicate_of = seen_hashes.get(digest)
            if duplicate_of is None:
                seen_hashes[digest] = path_str
            items.append({
                "name": label,
                "path": path_str,
                "available": True,
                "size_bytes": p.stat().st_size,
                "sha256": digest,
                "duplicate_of": duplicate_of,
            })
        else:
            items.append({
                "name": label,
                "path": path_str,
                "available": False,
                "size_bytes": None,
                "sha256": None,
                "missing_dependency_note": (
                    "External audit file not attached to workspace; independent Phase 2 "
                    "work proceeds using primary raw archives and ledger records."
                ),
            })

    return {
        "phase": "OSIRIS Frontier Research — Phase 2",
        "base_commit": "97f5db21",
        "phase1_claim_count": 44,
        "all_dispositions_proposed": True,
        "credential_exposures_flagged_without_values": [
            {
                "path": "/home/osiris/apikey.json",
                "type": "IBM Quantum JSON API key file on local disk (162 B)",
                "action": "Never read or committed; excluded from all archives; recommend rotation if previously synced.",
            },
            {
                "path": "/home/osiris/apikey - Copy.json",
                "type": "Duplicate IBM Quantum JSON API key file on local disk (162 B, world-readable 0644)",
                "action": "Recommend deleting duplicate and restrictingapikey.json to chmod 0600 + key rotation.",
            },
            {
                "path": "/home/osiris/osiris-ide/.env",
                "type": "Local environment credential file (350 B)",
                "action": "Keep untracked; never copy into evidence bundles.",
            },
            {
                "path": "/home/osiris/osiris-assessment/legacy-mirrors/ENKI-420/dnalang-sovereign-sdk-v2.git (HEAD:apikey.json)",
                "type": "Committed JSON API key in legacy bare git mirror",
                "action": "CRITICAL: Rotate that historical IBM Quantum API key immediately if not already revoked.",
            },
        ],
        "artifacts": items,
    }


def build_evidence_audit_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Phase 2 Section 2 Correction C:
    Audit every MEASURED and VERIFIED (and all other) rows with 5 mandatory fields:
      1. direct_or_derived
      2. data_accessibility
      3. verification_domain
      4. independent_replication
      5. remaining_limitations
    """
    cid = row["id"]
    p_status = row["proposed_status"]

    if cid == "NB-11":
        return {
            "direct_or_derived": "derived_metric_from_hardware_counts",
            "data_accessibility": "owner_reported_summary_table_only (raw d554h... count JSONs not in workspace)",
            "verification_domain": "hardware_execution_reported; mathematical_metric_DEFECTIVE (Eq. 7 computes independent expectation C(p,q), not W2)",
            "independent_replication": "not_replicated; 3 angles x 2 replicates on single backend ibm_fez",
            "remaining_limitations": (
                "Eq. 7 violates W2(p,p)=0 (gives 0.509 on identical 27Q uniform distributions); "
                "even exact LP W2 on 4,096-shot 27Q histograms is dominated by O(S^-1/27) multinomial "
                "sparsity bias unless split-sample debiased."
            ),
        }
    if cid == "NB-07":
        return {
            "direct_or_derived": "derived_arithmetic_bound_compared_to_literature_measurements",
            "data_accessibility": "public_peer_reviewed_literature_plus_owner_table",
            "verification_domain": "mathematical_and_published_hardware_falsification (like-for-like Bell-state preparation fidelity, distinct from RB gate fidelity)",
            "independent_replication": "independently_falsified_by_multiple_groups (Benhelm 2008 0.993(1), Gaebler 2016 0.9955(5) raw, Clark 2021 0.9994(3) raw, Ding 2023 0.985(2))",
            "remaining_limitations": "0/189 IBM runs had max Bell fidelity 0.9773 and therefore had zero statistical power to test a 0.9787 ceiling.",
        }
    if cid in ("NB-15", "NB-17"):
        return {
            "direct_or_derived": "derived_survival_probability_from_hardware_counts",
            "data_accessibility": (
                "owner_reported_for_ibm_fez_zenodo_22855102_22870287 (raw counts not in local workspace); "
                "accessible_raw_data_for_ibm_marrakesh_job_dau0q3qhcrkc73durtgg_and_E0_E4"
            ),
            "verification_domain": "hardware_observed_on_specific_chains (NOT universal across couplers; E2/E3/E4 show coupler dependence and 1.03-1.11 pp A/A drift floor)",
            "independent_replication": "replicated_within_osiris_on_selected_chains; prior art exists in Tripathi et al., Phys. Rev. Applied 18, 024068 (2022)",
            "remaining_limitations": (
                "NB-15 (0.946/0.888/0.795) and NB-17 (0.944/0.906/0.832) lack local raw count JSONs for ibm_fez; "
                "durations (8, 16, 32 us) must never be conflated with 8 us treatment-vs-control; "
                "adaptive cross-chip transfer to ibm_fez (E3) was null (-0.80 +/- 0.60 pp, p=0.218)."
            ),
        }
    if cid in ("NB-38", "NB-39", "NB-40", "NB-41", "NB-42"):
        return {
            "direct_or_derived": "derived_analytic_identity",
            "data_accessibility": "fully_reproducible_from_first_principles_code (build_ledger.py)",
            "verification_domain": "mathematical_verification_only (NOT hardware verification and NOT new physics)",
            "independent_replication": "standard_textbook_linear_algebra_and_geometry",
            "remaining_limitations": "Establishes classical Euclidean/matrix identities only; confers zero support for physical CRSM or quantum hardware claims.",
        }
    if p_status == "MEASURED":
        return {
            "direct_or_derived": "derived_statistic_from_hardware_runs",
            "data_accessibility": "owner_reported_without_local_raw_shot_files",
            "verification_domain": "owner_reported_hardware_observation (not independently recomputed from raw histograms in this workspace)",
            "independent_replication": "unreplicated_outside_owner_corpus",
            "remaining_limitations": "Reclassified from unqualified MEASURED to HARDWARE-OBSERVED / OWNER-REPORTED until raw job counts and layouts are archived locally.",
        }
    if cid == "NB-19":
        return {
            "direct_or_derived": "derived_from_accessible_raw_hardware_counts (rerun job dansntf8gn2s739n32e0)",
            "data_accessibility": "accessible_raw_data (tesseract_rerun_2026-09-20.zip)",
            "verification_domain": "hardware_and_mathematical_verification (v1 10^6x refuted; rerun 5.305% star vs 17.995% chain verified from raw histograms)",
            "independent_replication": "single_rerun_job_confounded_across_4_factors (targeted by TESSERACT-MAP-02)",
            "remaining_limitations": "Layouts of P0 and P1 were not saved; routing, physical qubit selection, depth, and opt_level are confounded.",
        }
    if cid == "OS-T1":
        return {
            "direct_or_derived": "derived_regression_over_archived_calibration_snapshots",
            "data_accessibility": "accessible_calibration_snapshots (11 Eagle Fake* V2 snapshots with T1 and frequency)",
            "verification_domain": "statistical_reanalysis_of_historical_calibration_snapshots (exploratory)",
            "independent_replication": "null_after_linear_trend_adjustment_across_all_11_snapshots",
            "remaining_limitations": "Electrical length L_eff was not fixed from hardware drawings; 0.51-0.68 GHz device bands span <20% of a 3.526 GHz period.",
        }
    return {
        "direct_or_derived": row.get("evidence_class", "unverified"),
        "data_accessibility": row.get("evidence_in_hand", "none"),
        "verification_domain": "audit_classification_only",
        "independent_replication": "none",
        "remaining_limitations": row.get("reason", ""),
    }


def generate_phase2_claims_artifacts(repo_root: Path) -> Dict[str, Any]:
    src_ledger = Path("/home/osiris/claims_ledger.jsonl")
    raw_lines = [line for line in src_ledger.read_text().splitlines() if line.strip()]
    assert len(raw_lines) == 44, f"Expected 44 Phase 1 claims, got {len(raw_lines)}"

    out_dir = repo_root / "docs" / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)

    inventory = build_state_inventory()
    inv_json_path = out_dir / "PHASE2_STATE_INVENTORY.json"
    inv_json_path.write_text(json.dumps(inventory, indent=2) + "\n")

    proposed_rows: List[Dict[str, Any]] = []
    staggered_structured = build_canonical_staggered_dd_record().to_dict()

    for line in raw_lines:
        row = json.loads(line)
        cid = row["id"]
        updated = dict(row)
        updated["phase1_proposed_status"] = row["proposed_status"]
        updated["phase1_label"] = row["label"]
        updated["evidence_audit"] = build_evidence_audit_metadata(row)
        updated["approval_state"] = "PROPOSED_PENDING_OWNER_APPROVAL"

        # Correction A — NB-11: Separate hardware counts from defective W2 conclusion
        if cid == "NB-11":
            updated["historical_measurements_classification"] = "HARDWARE-OBSERVED / OWNER-REPORTED"
            updated["w2_robustness_conclusion_classification"] = "QUARANTINED_PENDING_RECOMPUTATION"
            updated["label"] = "QUARANTINED_PENDING_RECOMPUTATION"
            updated["proposed_status"] = "QUARANTINED_PENDING_RECOMPUTATION"
            updated["reason"] = (
                "Hardware execution of 6 jobs on ibm_fez (Dec 23, 2025) is classified "
                "HARDWARE-OBSERVED / OWNER-REPORTED (raw count files not in workspace). "
                "However, the W2 robustness conclusion is QUARANTINED_PENDING_RECOMPUTATION: "
                "Appendix A Eq. 7 of w2_robustness_paper.tex computes the independent-product "
                "expectation C(p,q) = sum c(x,y)p(x)q(y), which gives sqrt(C(p,p)) = 0.5092 "
                "(R_hat = 0.4908) on two identical 27Q uniform distributions instead of W2(p,p) = 0, "
                "and omits the Layer-0 Hadamard gate when claiming constructive interference."
            )

        # Correction B — NB-07: Separate 2Q gate fidelity from like-for-like Bell-state fidelity
        elif cid == "NB-07":
            updated["bounded_quantity"] = "Two-qubit Bell-state preparation fidelity F_Bell = <Phi+|rho_exp|Phi+>"
            updated["gate_vs_state_distinction"] = (
                "Randomized benchmarking (RB) average 2-qubit gate fidelity (e.g. Kandala et al. 2021 "
                "0.9977(2) CNOT; Ballance et al. 2016 RB gate metric) isolates single-gate error from "
                "state preparation and measurement (SPAM) and is NOT interchangeable with Bell-state "
                "preparation fidelity. Falsification of F_max = 1 - phi^-8 = 0.978701 must use "
                "like-for-like Bell-state preparation + tomography measurements."
            )
            updated["like_for_like_bell_experiments"] = LIKE_FOR_LIKE_BELL_STATE_BENCHMARKS
            updated["reason"] = (
                "1 - phi^-8 = 0.978701 bounds Bell-state preparation fidelity, which is distinct from "
                "RB two-qubit gate fidelity. Comparing strictly like-for-like Bell-state preparation "
                "experiments (population + parity oscillation tomography), Benhelm et al. (Nat. Phys. 4, "
                "463, 2008) measured raw Bell-state fidelity F_Bell = 0.993(1) without SPAM subtraction "
                "(14.3 sigma above 0.9787); Gaebler et al. (PRL 117, 060505, 2016) measured 0.9955(5) raw "
                "(0.9991(4) SPAM-corrected); Clark et al. (PRL 127, 130505, 2021) measured 0.9994(3) raw; "
                "and Ding et al. (PRX 13, 031035, 2023) exceeded 0.985 raw on superconducting fluxonium. "
                "Furthermore, 0/189 IBM runs (max F_Bell = 0.9773) never reached the 0.9787 regime and "
                "had zero statistical power to test a ceiling."
            )

        # Correction C — Qualify owner-reported MEASURED rows and math-only VERIFIED rows
        elif row["proposed_status"] == "MEASURED":
            updated["label"] = "HARDWARE-OBSERVED / OWNER-REPORTED"
            updated["proposed_status"] = "HARDWARE-OBSERVED / OWNER-REPORTED"
            if cid == "NB-15":
                updated["structured_record"] = staggered_structured
                updated["verification_distinction"] = "supported_according_to_owners_register_not_independently_recomputed_from_fez_raw_counts"

        elif row["proposed_status"] == "VERIFIED":
            updated["label"] = "MATHEMATICALLY_VERIFIED_STANDARD_IDENTITY"
            updated["proposed_status"] = "MATHEMATICALLY_VERIFIED_ONLY"

        if cid == "NB-19":
            updated["namespace_disambiguation"] = TESSERACT_NAMESPACE_REGISTRY

        proposed_rows.append(updated)

    # Append explicit external-decoder disambiguation record (non-OSIRIS claim) as row 45 metadata note?
    # Wait: keep exact 44 claim rows in `claims_ledger_phase2_proposed.jsonl` matching Phase 1's 44-claim
    # index 1-to-1, while embedding `namespace_disambiguation` on NB-19 and documenting external records cleanly!
    proposed_jsonl_path = out_dir / "claims_ledger_phase2_proposed.jsonl"
    proposed_jsonl_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in proposed_rows) + "\n"
    )

    return {
        "inventory_path": str(inv_json_path),
        "proposed_ledger_path": str(proposed_jsonl_path),
        "row_count": len(proposed_rows),
        "source_ledger_sha256": hashlib.sha256(src_ledger.read_bytes()).hexdigest(),
        "proposed_ledger_sha256": hashlib.sha256(proposed_jsonl_path.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(generate_phase2_claims_artifacts(root), indent=2))
