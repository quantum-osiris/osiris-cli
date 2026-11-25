"""Verification suite for write-ahead staged & live HARDWARE_ATTESTED OSIRIS-ECHO-FEZ and OSIRIS-QEC-REP3-FEZ."""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ECHO_DIR = REPO_ROOT / "experiments" / "echo_fez_staged"
REP3_DIR = REPO_ROOT / "experiments" / "rep3_fez_staged"


def _load(path: Path, name: str):
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _parse_iso(ts: str) -> dt.datetime:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def test_rep3_fez_live_hardware_attested_results_and_write_ahead() -> None:
    man = json.loads((REP3_DIR / "rep3_fez_staging_manifest.json").read_text(encoding="utf-8"))
    sub = json.loads((REP3_DIR / "rep3_fez_submitted_job.json").read_text(encoding="utf-8"))
    res = json.loads((REP3_DIR / "physical_rep3_fez_results.json").read_text(encoding="utf-8"))
    raw_gz = REP3_DIR / "physical_rep3_fez_raw_counts.json.gz"

    assert res["execution_tier"] == "HARDWARE_ATTESTED"
    assert res["provider_job_id"] == "dc35d4inj2vs73dhmsog"
    assert sub["job_id"] == "dc35d4inj2vs73dhmsog"
    assert res["qpy_sha256"] == man["qpy_sha256"] == "f240d445acc36a69ab2a2604132fa8fc8a0dd5c45130cd8d641b6780fcecc83a"
    assert hashlib.sha256(raw_gz.read_bytes()).hexdigest() == res["raw_counts_gz_sha256"]

    # Write-ahead server-clock check (> 9 hours margin)
    staged_dt = _parse_iso(man["staged_utc"])
    created_dt = _parse_iso(res["provider_metadata"]["timestamps"]["created"])
    assert (created_dt - staged_dt).total_seconds() > 30000.0

    with gzip.open(raw_gz, "rt", encoding="utf-8") as f:
        raw_counts = json.load(f)
    assert len(raw_counts) == 96
    assert sum(sum(v.values()) for v in raw_counts.values()) == 49152

    # Both Arm 4a (Pauli-frame MWPM) and Arm 4b (physical feed-forward) cross below Arm 1 & Arm 2 at R=3 and R=5
    by_r = {int(row["rounds"]): row for row in res["rounds_summary"]}
    for r in (3, 5):
        row = by_r[r]
        assert row["arm4a_decoded_pauli_frame_p_fail"] < row["arm2_encoded_no_corr_p_fail"] < row["arm1_unencoded_p_fail"]
        assert row["arm4b_physical_feedforward_p_fail"] < row["arm2_encoded_no_corr_p_fail"] < row["arm1_unencoded_p_fail"]
    assert res["paired_block_tests"]["R5_arm2_minus_arm4b"]["p_one_sided"] < 0.01
    assert res["ro_cal_3q_assignment_fidelity"] > 0.97


def test_echo_fez_live_hardware_attested_results_and_write_ahead() -> None:
    man = json.loads((ECHO_DIR / "echo_fez_staging_manifest.json").read_text(encoding="utf-8"))
    sub = json.loads((ECHO_DIR / "echo_fez_submitted_job.json").read_text(encoding="utf-8"))
    res = json.loads((ECHO_DIR / "physical_echo_fez_results.json").read_text(encoding="utf-8"))
    raw_gz = ECHO_DIR / "physical_echo_fez_raw_counts.json.gz"

    assert res["execution_tier"] == "HARDWARE_ATTESTED"
    assert res["provider_job_id"] == "dc35e35s2omc738cvcug"
    assert sub["job_id"] == "dc35e35s2omc738cvcug"
    assert res["qpy_sha256"] == man["qpy_sha256"] == "538af9bec5fd1ab2536272e5b33c50021d4e87247cb6f84cbbb415df3b406011"
    assert hashlib.sha256(raw_gz.read_bytes()).hexdigest() == res["raw_counts_gz_sha256"]

    staged_dt = _parse_iso(man["staged_utc"])
    created_dt = _parse_iso(res["provider_metadata"]["timestamps"]["created"])
    assert (created_dt - staged_dt).total_seconds() > 30000.0

    with gzip.open(raw_gz, "rt", encoding="utf-8") as f:
        raw_counts = json.load(f)
    assert len(raw_counts) == 240
    assert sum(sum(v.values()) for v in raw_counts.values()) == 61440

    assert res["spearman_echo"] > 0.70
    assert res["spearman_echo"] > res["spearman_topology"] + 0.50
    assert res["verdicts"]["top1_heldout_composite_regret"] == 0.0
    assert res["verdicts"]["H2_single_window_echo_beats_calibration_spearman"].startswith("REFUTED")

    for folder in (ECHO_DIR, REP3_DIR):
        for p in folder.iterdir():
            if p.is_file() and p.suffix in {".json", ".md", ".py"}:
                txt = p.read_text(encoding="utf-8")
                assert "crn:v1:bluemix:" not in txt


def test_echo_fez_staged_manifest_qpy_and_synthetic_analysis() -> None:
    pytest.importorskip("qiskit")
    man_p = ECHO_DIR / "echo_fez_staging_manifest.json"
    qpy_p = ECHO_DIR / "echo_fez_staged_circuits.qpy"
    prereg_p = ECHO_DIR / "PRE_REGISTRATION.md"
    assert man_p.is_file() and qpy_p.is_file() and prereg_p.is_file()

    man = json.loads(man_p.read_text(encoding="utf-8"))
    assert man["execution_tier"] == "OFFLINE_STAGED"
    assert man["provider_job_id"] is None
    assert man["total_pubs"] == 240
    assert man["shots_per_pub"] == 256
    assert man["total_shots"] == 61440
    assert hashlib.sha256(qpy_p.read_bytes()).hexdigest() == man["qpy_sha256"]
    assert hashlib.sha256(prereg_p.read_bytes()).hexdigest() == man["preregistration_md_sha256"]

    echo_mod = _load(ECHO_DIR / "stage_and_harvest_echo_fez.py", "_test_echo_staged")
    fake_counts = {}
    for item in man["schedule"]:
        pid = int(item["placement_id"])
        exp_b = format(int(item["expected_y"]), "05b")
        good = 250 - 15 * pid
        bad_b = format(int(item["expected_y"]) ^ 1, "05b")
        fake_counts[str(item["pub_index"])] = {exp_b: good, bad_b: 256 - good}

    res = echo_mod.analyze_harvested(man, fake_counts, {"job_id": None}, "0" * 64)
    assert res["execution_tier"] == "SIMULATED"
    assert res["spearman_echo"] > 0.99


def test_rep3_fez_staged_manifest_qpy_noiseless_and_mwpm_decode() -> None:
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    man_p = REP3_DIR / "rep3_fez_staging_manifest.json"
    qpy_p = REP3_DIR / "rep3_fez_staged_circuits.qpy"
    prereg_p = REP3_DIR / "PRE_REGISTRATION.md"
    assert man_p.is_file() and qpy_p.is_file() and prereg_p.is_file()

    man = json.loads(man_p.read_text(encoding="utf-8"))
    assert man["execution_tier"] == "OFFLINE_STAGED"
    assert man["provider_job_id"] is None
    assert man["total_pubs"] == 96
    assert man["shots_per_pub"] == 512
    assert man["total_shots"] == 49152
    assert hashlib.sha256(qpy_p.read_bytes()).hexdigest() == man["qpy_sha256"]
    assert hashlib.sha256(prereg_p.read_bytes()).hexdigest() == man["preregistration_md_sha256"]

    rep3_mod = _load(REP3_DIR / "stage_and_harvest_rep3_fez.py", "_test_rep3_staged")
    ver = rep3_mod.verify_noiseless_and_single_fault_5q()
    assert ver["noiseless_all_24_cells_exact"] is True
    assert ver["arm4b_single_fault_feedforward_exact"] is True

    fake_counts = {}
    for item in man["schedule"]:
        arm = item["arm"]
        if arm == "RO_CAL":
            fake_counts[str(item["pub_index"])] = {"000": 512}
            continue
        r = int(item["rounds"])
        lb = int(item["logical_bit"])
        if arm == "ARM1_UNENCODED":
            fake_counts[str(item["pub_index"])] = {str(lb ^ 1): 50, str(lb): 462}
        elif arm == "ARM2_ENCODED_NO_CORR":
            fake_counts[str(item["pub_index"])] = {f"{lb}{lb}{lb}": 490, f"{lb^1}{lb^1}{lb^1}": 22}
        elif arm == "ARM4B_PHYSICAL_FEEDFORWARD":
            fake_counts[str(item["pub_index"])] = {
                f"{lb}{lb}{lb} " + " ".join(["00"] * r): 508,
                f"{lb^1}{lb^1}{lb^1} " + " ".join(["00"] * r): 4,
            }
        elif arm == "ARM3_4A_SYNDROME_STREAM":
            clean_key = f"{lb}{lb}{lb} " + " ".join(["00"] * r)
            d0_flipped_out = f"{lb}{lb}{lb^1}"
            fault_key = f"{d0_flipped_out} " + " ".join(["01"] * r)
            fake_counts[str(item["pub_index"])] = {clean_key: 450, fault_key: 62}

    res = rep3_mod.analyze_harvested(man, fake_counts, {"job_id": None}, "0" * 64)
    assert res["execution_tier"] == "SIMULATED"
    for row in res["rounds_summary"]:
        assert row["arm4a_decoded_pauli_frame_p_fail"] == 0.0
        assert abs(row["arm3_postselected_acceptance_rate"] - (450.0 / 512.0)) < 1e-5
        assert row["arm3_postselected_conditional_p_fail"] == 0.0
