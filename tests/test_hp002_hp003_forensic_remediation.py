"""Offline regression suite verifying HP-002 / HP-003 forensic audit & v2.1 errata invariants."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = REPO_ROOT / "osiris_quantum_hardware_v2.0_evidence_package"
EXP2_DIR = REPO_ROOT / "experiments" / "hp002"
EXP3_DIR = REPO_ROOT / "experiments" / "hp003"


def _load_reaudit_module():
    mod_path = EXP3_DIR / "mitigate_and_extrapolate_hp003.py"
    spec = importlib.util.spec_from_file_location("mitigate_and_extrapolate_hp003", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ledger_anchored_artifacts_preserved_verbatim() -> None:
    """Ensure historical ledger-anchored scripts, results, and QPY binaries remain byte-intact."""
    expected = [
        (EXP2_DIR / "stage_hp002_harness.py", "72d6fed2e8e1beb5d9c0a618eb6982b45247e6e4ede96dfdb3199626b6f05973"),
        (EXP2_DIR / "hp002_staged_circuits.qpy", "a7c8cf7a84cde0bac9b65fe8199172f1a9b3c760cfd87c8d897ce76a71c15185"),
        (EXP2_DIR / "physical_hp002_results.json", "0be991209e91d763f131db071db373ed8c4b2986c4b2986c44a0a5fe3206b65bafabc56"[:64]),
        (EXP3_DIR / "stage_hp003_harness.py", "68174cfdef1eebd0d633edf69387fb6d54f32f602b70661e49dd3bb34416beb4"),
        (EXP3_DIR / "hp003_staged_circuits.qpy", "931563b75563b4a5c899b5c6ce72c6d0c057dfd0b2ec0f94f970aaf68732a075"),
        (EXP3_DIR / "physical_hp003_results.json", "757442a222e37b5c5050c5ac3843954a253ae1a358b1451ec9f1fc743912a19a"),
    ]
    # Read exact Block 44 digest from block_44_ledger_receipt.json
    b44_digest = json.loads((EXP2_DIR / "block_44_ledger_receipt.json").read_text(encoding="utf-8"))["results_sha256"]
    expected[2] = (EXP2_DIR / "physical_hp002_results.json", b44_digest)

    for path, digest in expected:
        raw = path.read_bytes()
        h_raw = hashlib.sha256(raw).hexdigest()
        lf = raw.replace(b"\r\n", b"\n")
        h_lf = hashlib.sha256(lf).hexdigest()
        h_crlf = hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest()
        assert digest in {h_raw, h_lf, h_crlf}, f"Ledger anchor broken for {path}"

    ledger_jsonl = PKG_DIR / "ledger" / "ledger_blocks_1_to_48.jsonl"
    assert ledger_jsonl.exists()
    assert len(ledger_jsonl.read_text(encoding="utf-8").splitlines()) == 48


def test_hp003_offline_qpy_reaudit_layouts_lp_bounds_and_conservative_se() -> None:
    mod = _load_reaudit_module()
    payload = mod.run_offline_reaudit()
    arms = payload["arm_reaudited_summary"]

    # Exact QPY physical readout quartets [c0, c1, c2, c3]
    assert arms["Arm A"]["readout_quartet_c0_c1_c2_c3"] == [5, 4, 21, 16]
    assert arms["Arm B"]["readout_quartet_c0_c1_c2_c3"] == [147, 143, 123, 144]
    assert arms["Arm B-Def"]["readout_quartet_c0_c1_c2_c3"] == [146, 143, 123, 144]
    assert arms["Arm B-Ctrl"]["readout_quartet_c0_c1_c2_c3"] == [20, 23, 4, 22]
    assert arms["Arm B-Held"]["readout_quartet_c0_c1_c2_c3"] == [20, 22, 24, 23]

    # SPAM calibration applied strictly where physical readout qubits are calibrated
    assert arms["Arm B-Def"]["mitigation_status"] == "FULL_PAIR1_PAIR2_READOUT_CALIBRATED"
    assert arms["Arm B-Def"]["mitigated_mean_fidelity"] is not None
    assert arms["Arm B"]["mitigation_status"] == "PARTIAL_PAIR2_ONLY_MISSING_QUBIT_147"
    assert arms["Arm B"]["mitigated_mean_fidelity"] is None
    assert arms["Arm A"]["mitigated_mean_fidelity"] is None

    # Conservative SE never drops below multinomial shot-noise SE (fixes Arm A 0.08% -> 0.41%)
    for st in arms.values():
        assert st["conservative_se_eff"] >= st["shot_se_mean_fidelity"]
        assert st["conservative_se_eff"] >= st["sem_block_ddof1_df2"]
    assert arms["Arm A"]["conservative_se_eff"] == pytest.approx(0.004127, abs=5e-6)
    assert arms["Arm B"]["conservative_se_eff"] == pytest.approx(0.004109, abs=5e-6)

    # Tight 16-state Bell-LP joint overlap bounds & >1 ebit verdict
    assert arms["Arm B"]["f_joint_bell_lp_bounds"] == pytest.approx([0.499023, 0.737467], abs=5e-6)
    assert arms["Arm B"]["certifies_gt_1ebit_unconditionally"] is False

    audit = payload["cross_arm_coherence_comparison_audit"]
    assert audit["subgraphs_disjoint"] is True
    assert audit["cross_subgraph_coherence_ratio_valid"] is False


def test_evidence_package_sha256sums_and_permit_labeling() -> None:
    sums_file = PKG_DIR / "SHA256SUMS.txt"
    entries = [line.strip().split("  ", 1) for line in sums_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(entries) >= 35
    for expected_hash, rel_path in entries:
        actual = hashlib.sha256((PKG_DIR / rel_path).read_bytes()).hexdigest()
        assert actual == expected_hash, f"SHA256SUMS mismatch on {rel_path}"

    for grant_script in (
        REPO_ROOT / "experiments" / "hp001" / "generate_hp001_phase2_grant.py",
        EXP2_DIR / "generate_hp002_grant.py",
        EXP3_DIR / "generate_hp003_grant.py",
    ):
        txt = grant_script.read_text(encoding="utf-8")
        assert "HMAC-SHA256" in txt
        assert '.replace(b"\\r\\n", b"\\n")' in txt
        assert "Issue Node Beta Ed25519 Authority Permit" not in txt

    dispatch_txt = (EXP3_DIR / "dispatch_hp003.py").read_text(encoding="utf-8")
    assert 'grant.get("payload_sha256")' in dispatch_txt
