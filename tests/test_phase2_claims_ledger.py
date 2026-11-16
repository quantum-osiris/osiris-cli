"""Verification tests for Phase 2 Section 1 & Section 2 Claims Ledger deliverables."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_authoritative_claims_and_phase1_files_unchanged():
    """Verify neither `/home/osiris/claims_ledger.jsonl` nor `osiris_cli/claims.py` was modified."""
    baseline = json.loads((REPO_ROOT / "docs/phase2/PRE_PHASE2_BASELINE_HASHES.json").read_text())
    expected_claims_py = baseline["files"]["osiris_cli/claims.py"]["sha256"]
    actual_claims_py = hashlib.sha256((REPO_ROOT / "osiris_cli/claims.py").read_bytes()).hexdigest()
    assert actual_claims_py == expected_claims_py

    phase1_ledger = Path("/home/osiris/claims_ledger.jsonl")
    assert hashlib.sha256(phase1_ledger.read_bytes()).hexdigest() == (
        "567a1ffd6c425eab3b031d5191ecd3fbd1198b89233bbbe87f5c1ee0fa25fb1f"
    )


def test_phase2_proposed_claims_ledger_completeness():
    path = REPO_ROOT / "docs/phase2/claims_ledger_phase2_proposed.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) == 44

    by_id = {r["id"]: r for r in rows}

    # Correction A: NB-11 separation
    nb11 = by_id["NB-11"]
    assert nb11["historical_measurements_classification"] == "HARDWARE-OBSERVED / OWNER-REPORTED"
    assert nb11["w2_robustness_conclusion_classification"] == "QUARANTINED_PENDING_RECOMPUTATION"
    assert nb11["proposed_status"] == "QUARANTINED_PENDING_RECOMPUTATION"

    # Correction B: NB-07 like-for-like Bell-state experiments
    nb07 = by_id["NB-07"]
    assert "Bell-state preparation fidelity" in nb07["bounded_quantity"]
    raw_non_spam = [
        b for b in nb07["like_for_like_bell_experiments"]
        if not b["spam_corrected"] and b["fidelity"] > 0.978701
    ]
    assert len(raw_non_spam) >= 4

    # Correction C: All 44 rows carry the 5-field `evidence_audit` metadata and remain PROPOSED
    required_audit_keys = {
        "direct_or_derived",
        "data_accessibility",
        "verification_domain",
        "independent_replication",
        "remaining_limitations",
    }
    for r in rows:
        assert r["approval_state"] == "PROPOSED_PENDING_OWNER_APPROVAL"
        assert required_audit_keys.issubset(set(r["evidence_audit"].keys()))
        # No unqualified MEASURED or VERIFIED proposed_status remains
        assert r["proposed_status"] not in ("MEASURED", "VERIFIED")

    # Structured STAGGERED_DD points on NB-15 and namespace disambiguation on NB-19
    assert by_id["NB-15"]["structured_record"]["comparison"]["duration_us"] == 32.0
    assert "TESSERACT_COLOR_CODE_DECODER_EXT" in by_id["NB-19"]["namespace_disambiguation"]
