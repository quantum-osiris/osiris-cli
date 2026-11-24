"""Verification tests for Phase 1 claim_evidence_map.jsonl & verification_queue.md reconciliation."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CEM_PATH = REPO_ROOT / "phase1" / "claim_evidence_map.jsonl"
VQ_PATH = REPO_ROOT / "phase1" / "verification_queue.md"

REQUIRED_KEYS = {
    "claim_id",
    "proposition",
    "origin",
    "evidence_basis",
    "execution_class",
    "assessment",
    "scope",
    "links",
    "next_check",
    "reviewer",
    "reviewed",
}


def test_claim_evidence_map_jsonl_schema_and_reconciled_claims() -> None:
    lines = [ln for ln in CEM_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) >= 42
    by_id = {}
    for idx, ln in enumerate(lines, 1):
        row = json.loads(ln)
        assert REQUIRED_KEYS.issubset(row.keys()), f"Line {idx} missing keys"
        cid = row["claim_id"]
        assert cid not in by_id, f"Duplicate claim_id {cid}"
        by_id[cid] = row

    assert by_id["SCI-XTOP-1"]["assessment"] == "SUPPORTED"
    assert "+19.11 s" in by_id["SCI-XTOP-1"]["scope"]
    assert "remediated" in by_id["PHY-9"]["assessment"]
    assert "G1_INDEPENDENT_REVIEW.md" in json.dumps(by_id["BT2T-2"]["links"])
    assert by_id["ENG-NEXUS-4"]["assessment"] == "UNVERIFIED"

    for new_cid in ("BT2T-4", "BT2T-5", "ECHO-1", "QEC-CARDC-1"):
        assert new_cid in by_id, f"Missing {new_cid}"
        for link in by_id[new_cid]["links"]:
            art = link["artifact"].split("::")[0]
            assert (REPO_ROOT / art).exists(), f"Missing linked artifact {art} for {new_cid}"


def test_verification_queue_contains_2026_11_24_reconciliation_ledger() -> None:
    vq = VQ_PATH.read_text(encoding="utf-8")
    assert "## 5. Reconciliation & Execution Ledger (2026-11-24)" in vq
    for marker in (
        "G1_INDEPENDENT_REVIEW.md",
        "sci_xtop1_dc0uhdts2omc7389tmt0_provider_receipt.json",
        "dc2ee3uq4uuc73a56o6g",
        "OSIRIS-BT2T-002",
        "echo_stopgo_report.json",
        "card_c_qec_crossing_report.json",
        "STOP_NO_CROSSING_AT_TESTED_ROUNDS",
    ):
        assert marker in vq, f"Missing marker {marker!r} in verification_queue.md"
