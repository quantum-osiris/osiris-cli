"""Verification tests for Phase 2 & Phase 3 Owner-Approved Claims Register & Ledger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from osiris.compiler.dna_ir import (
    ClassicalRegister,
    IROperation,
    IROpType,
    QuantumCircuitIR,
    QuantumRegister,
)
from osiris.compiler.dna_runtime import QuantumRuntime, RuntimeConfig
from osiris_cli import claims

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_owner_approved_claims_register_updates():
    """Verify owner-approved Phase 3 transitions in `osiris_cli/claims.py`."""
    w2_art = claims.by_id("W2_ROBUSTNESS")
    w2_cnt = claims.by_id("W2_THETA_SWEEP_COUNTS")
    assert w2_art is not None and w2_art.verdict == "ARTIFACT"
    assert w2_cnt is not None and w2_cnt.verdict == "REPORTED"

    # Tesseract namespace disambiguation: external color-code decoder & TESSERACT-MAP-02 must NOT match ERROR_SUPPRESSION_1E6
    assert "ERROR_SUPPRESSION_1E6" not in {
        c.id for c in claims.match("Quantinuum Tesseract decoder on d=7 color-code syndromes")
    }
    assert "ERROR_SUPPRESSION_1E6" not in {
        c.id for c in claims.match("Benchmark TESSERACT-MAP-02 degree-2 chain compilation")
    }
    assert "ERROR_SUPPRESSION_1E6" in {
        c.id for c in claims.match("40-qubit tesseract circuit on ibm_torino")
    }

    # F_MAX cites raw like-for-like Bell-state experiments (Benhelm 2008, Clark 2021, Ding 2023)
    fmax_dois = {ref for kind, ref, _ in claims.by_id("F_MAX").evidence if kind == "doi"}
    assert "10.1103/PhysRevLett.127.130505" in fmax_dois
    assert "10.1103/PhysRevX.13.031035" in fmax_dois

    # SovereignAdapter executes genuine unitary Bell state (only 00 and 11, zero 01 or 10!)
    bell_ir = QuantumCircuitIR(
        name="bell_check",
        quantum_registers=[QuantumRegister("q", 2)],
        classical_registers=[ClassicalRegister("c", 2)],
        operations=[
            IROperation(IROpType.H, [0]),
            IROperation(IROpType.CX, [0, 1]),
            IROperation(IROpType.MEASURE, [0, 1], classical_bits=[0, 1]),
        ],
    )
    bell_ir.compute_metrics()
    res = QuantumRuntime(RuntimeConfig(backend_name="sovereign", shots=128)).execute(bell_ir)
    assert set(res.counts.keys()) <= {"00", "11"}
    assert sum(res.counts.values()) == 128


def test_phase2_and_phase3_claims_ledger_completeness():
    # Phase 1 backup hash preserved immutably if present on local host
    backup = Path("/home/osiris/claims_ledger.phase1_backup.jsonl")
    if backup.exists():
        assert hashlib.sha256(backup.read_bytes()).hexdigest() == (
            "567a1ffd6c425eab3b031d5191ecd3fbd1191ecd3fbd1198b89233bbbe87f5c1ee0fa25fb1f"[:64]
            if False
            else "567a1ffd6c425eab3b031d5191ecd3fbd1198b89233bbbe87f5c1ee0fa25fb1f"
        )

    path = REPO_ROOT / "docs/phase2/claims_ledger_phase2_proposed.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) == 44

    by_id = {r["id"]: r for r in rows}
    nb11 = by_id["NB-11"]
    assert nb11["historical_measurements_classification"] == "HARDWARE-OBSERVED / OWNER-REPORTED"
    assert nb11["w2_robustness_conclusion_classification"] == "QUARANTINED_PENDING_RECOMPUTATION"

    nb07 = by_id["NB-07"]
    raw_non_spam = [
        b for b in nb07["like_for_like_bell_experiments"]
        if not b["spam_corrected"] and b["fidelity"] > 0.978701
    ]
    assert len(raw_non_spam) >= 4

    required_audit_keys = {
        "direct_or_derived",
        "data_accessibility",
        "verification_domain",
        "independent_replication",
        "remaining_limitations",
    }
    for r in rows:
        assert required_audit_keys.issubset(set(r["evidence_audit"].keys()))
        assert r["proposed_status"] not in ("MEASURED", "VERIFIED")
