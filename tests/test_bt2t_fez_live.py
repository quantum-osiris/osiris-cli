"""Offline verification tests for OSIRIS-BT2T-001-FEZ (Card A hardware execution & G1 review)."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BT2T_INV_DIR = REPO_ROOT / "experiments" / "bt2t_inversion"
BT2T_FEZ_DIR = REPO_ROOT / "experiments" / "bt2t_fez"


def _load_fez_runner():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    mod_name = "_test_bt2t_fez_runner"
    spec = importlib.util.spec_from_file_location(
        mod_name, BT2T_FEZ_DIR / "stage_and_harvest_bt2t_fez.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_g1_independent_review_and_preregistration_hashes_match_manifest():
    g1_path = BT2T_INV_DIR / "G1_INDEPENDENT_REVIEW.md"
    prereg_path = BT2T_FEZ_DIR / "PRE_REGISTRATION.md"
    man_path = BT2T_FEZ_DIR / "bt2t_fez_staging_manifest.json"
    qpy_path = BT2T_FEZ_DIR / "bt2t_fez_staged_circuits.qpy"

    assert g1_path.is_file()
    assert prereg_path.is_file()
    assert man_path.is_file()
    assert qpy_path.is_file()

    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    assert (
        hashlib.sha256(g1_path.read_bytes()).hexdigest()
        == manifest["g1_independent_review_sha256"]
    )
    assert (
        hashlib.sha256(prereg_path.read_bytes()).hexdigest()
        == manifest["preregistration_md_sha256"]
    )
    assert hashlib.sha256(qpy_path.read_bytes()).hexdigest() == manifest["qpy_sha256"]
    assert manifest["equal_shot_rule_enforced"] is True
    assert manifest["total_pubs"] == 288
    assert manifest["shots_per_pub"] == 96
    assert manifest["total_shots"] == 27648
    assert manifest["unit_of_inference"]["calibration_windows"] == 1


def test_t5_cores_noiseless_verification_and_gate_counts():
    pytest.importorskip("qiskit")
    runner = _load_fez_runner()
    ver = runner.compile_and_verify_t5_cores()["summary"]
    assert ver["encoding_version"] == "bt2t-enc-1"
    assert (
        ver["encoding_sha256"]
        == "3e3b02d13539daaf431af39056d031f777a785a556eb9da21ef4ea6097dfc4c5"
    )
    assert ver["all_96_cells_noiseless_verified"] is True
    assert ver["max_noiseless_basis_inaccuracy"] < 1e-11
    assert ver["arms"]["STRUCTURED"]["cz_count"] == 72
    assert ver["arms"]["GENERIC"]["cz_count"] == 723
    assert ver["arms"]["PREP_RO_CTRL"]["cz_count"] == 0


def test_harvested_bt2t_fez_results_reproducible_and_redacted():
    pytest.importorskip("qiskit")
    runner = _load_fez_runner()

    res_path = BT2T_FEZ_DIR / "physical_bt2t_fez_results.json"
    raw_gz_path = BT2T_FEZ_DIR / "physical_bt2t_fez_raw_counts.json.gz"
    job_path = BT2T_FEZ_DIR / "bt2t_fez_submitted_job.json"
    man_path = BT2T_FEZ_DIR / "bt2t_fez_staging_manifest.json"

    assert res_path.is_file()
    assert raw_gz_path.is_file()
    assert job_path.is_file()

    saved = json.loads(res_path.read_text(encoding="utf-8"))
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    job_meta = json.loads(job_path.read_text(encoding="utf-8"))
    raw_sha = hashlib.sha256(raw_gz_path.read_bytes()).hexdigest()

    assert saved["execution_tier"] == "HARDWARE_ATTESTED"
    assert saved["backend"] == "ibm_fez"
    assert saved["provider_job_id"] == job_meta["job_id"]
    assert saved["provider_job_id"] is not None and len(saved["provider_job_id"]) >= 10
    assert saved["raw_counts_gz_sha256"] == raw_sha
    assert saved["equal_shot_rule"]["total_shots"] == 27648
    assert saved["equal_shot_rule"]["shots_per_arm_input_cell"] == 288

    with gzip.open(raw_gz_path, "rb") as gz:
        raw_counts = json.loads(gz.read().decode("utf-8"))
    assert len(raw_counts) == 288
    assert sum(sum(c.values()) for c in raw_counts.values()) == 27648

    recomputed = runner.analyze_harvested(
        manifest, raw_counts, job_meta, raw_sha, saved.get("quota_accounting", {})
    )
    assert recomputed == saved

    # Secret / CRN redaction scan across all BT2T artifacts
    for folder in (BT2T_FEZ_DIR, BT2T_INV_DIR):
        for p in folder.iterdir():
            if p.is_file() and p.suffix in {".json", ".md", ".py"}:
                txt = p.read_text(encoding="utf-8")
                assert "crn:v1:bluemix:" not in txt, f"CRN leaked in {p}"
                assert '"apikey"' not in txt or p.name.endswith(".py"), f"Secret key field in {p}"
