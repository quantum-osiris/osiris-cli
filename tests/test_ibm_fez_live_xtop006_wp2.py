"""Verification suite for governed IBM credentials (`troykey.json`), `OSIRIS-XTOP-006-FEZ`
(`dc1tqaanj2vs73dfrt40`), `OSIRIS-WP2-27Q-FEZ` (`dc1tqvinj2vs73dfrts0`), and `SCI-XTOP-1`
provider timestamp receipt (`dc0uhdts2omc7389tmt0`).
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from osiris_cli.ibm_credentials import load_ibm_token_with_provenance

REPO_ROOT = Path(__file__).resolve().parents[1]
XTOP_DIR = REPO_ROOT / "experiments" / "xtop006_fez"
WP2_DIR = REPO_ROOT / "experiments" / "wp2_27q_fez"


def test_ibm_credentials_redacts_secrets(tmp_path: Path) -> None:
    fake_key = tmp_path / "troykey.json"
    secret = "SUPER_SECRET_IBM_API_TOKEN_DO_NOT_LEAK_12345"
    fake_key.write_text(
        json.dumps({"name": "osiris", "createdAt": "2026-11-20T11:28+0000", "apikey": secret}),
        encoding="utf-8",
    )
    tok, prov = load_ibm_token_with_provenance(explicit_path=fake_key)
    assert tok == secret
    serialized = json.dumps(prov)
    assert secret not in serialized
    assert prov["credential_basename"] == "troykey.json"
    assert len(prov["key_file_sha256_prefix16"]) == 16


def test_sci_xtop1_provider_receipt_redacted_and_valid() -> None:
    p = XTOP_DIR / "sci_xtop1_dc0uhdts2omc7389tmt0_provider_receipt.json"
    assert p.is_file()
    raw = p.read_text(encoding="utf-8")
    assert "crn:v1:" not in raw
    rec = json.loads(raw)
    assert rec["job_id"] == "dc0uhdts2omc7389tmt0"
    assert rec["backend"] == "ibm_marrakesh"
    assert rec["status"] == "DONE"
    assert rec["timestamps"]["created"] == "2026-11-21T18:53:43.172751Z"
    assert rec["usage"]["qpu_charge_time_seconds"] == 63


def test_xtop006_fez_hardware_attested_bundle_integrity() -> None:
    man = json.loads((XTOP_DIR / "xtop006_fez_staging_manifest.json").read_text(encoding="utf-8"))
    res = json.loads((XTOP_DIR / "physical_xtop006_fez_results.json").read_text(encoding="utf-8"))
    qpy_bytes = (XTOP_DIR / "xtop006_fez_staged_circuits.qpy").read_bytes()
    gz_path = XTOP_DIR / "physical_xtop006_fez_raw_counts.json.gz"

    assert hashlib.sha256(qpy_bytes).hexdigest() == man["qpy_sha256"] == res["qpy_sha256"]
    assert hashlib.sha256(gz_path.read_bytes()).hexdigest() == res["raw_counts_gz_sha256"]
    assert res["execution_tier"] == "HARDWARE_ATTESTED"
    assert res["provider_job_id"] == "dc1tqaanj2vs73dfrt40"
    assert res["constant_cz_count"] == 28
    assert res["constant_sx_count"] == 126
    assert "crn:v1:" not in json.dumps(res)

    traj = res["coherent_trajectory"]
    assert len(traj) == 5
    # Strict monotonic increase in F_6state and decrease in P_{Phi+} across all 5 theta points
    f_vals = [r["raw_f_6state"] for r in traj]
    p_vals = [r["raw_p_phi_plus"] for r in traj]
    assert f_vals == sorted(f_vals)
    assert p_vals == sorted(p_vals, reverse=True)
    assert res["hypotheses"]["H1_non_clifford_kraus_climb"]["verdict"] == "SUPPORTED"
    assert traj[-1]["raw_f_6state_95pct_lcl"] > (2.0 / 3.0)

    with gzip.open(gz_path, "rb") as gz:
        counts = json.loads(gz.read().decode("utf-8"))
    assert len(counts) == 232
    assert sum(sum(c.values()) for c in counts.values()) == 232 * 512


def test_wp2_27q_fez_hardware_attested_bundle_integrity() -> None:
    man = json.loads((WP2_DIR / "wp2_27q_fez_staging_manifest.json").read_text(encoding="utf-8"))
    res_path = WP2_DIR / "physical_wp2_27q_fez_results.json"
    if not res_path.is_file():
        pytest.skip("Analysis still running")
    res = json.loads(res_path.read_text(encoding="utf-8"))
    qpy_bytes = (WP2_DIR / "wp2_27q_fez_staged_circuits.qpy").read_bytes()
    gz_path = WP2_DIR / "physical_wp2_27q_fez_raw_counts.json.gz"

    assert hashlib.sha256(qpy_bytes).hexdigest() == man["qpy_sha256"] == res["qpy_sha256"]
    assert hashlib.sha256(gz_path.read_bytes()).hexdigest() == res["raw_counts_gz_sha256"]
    assert res["execution_tier"] == "HARDWARE_ATTESTED"
    assert res["provider_job_id"] == "dc1tqvinj2vs73dfrts0"
    assert res["constant_cz_count"] == 52
    assert "crn:v1:" not in json.dumps(res)

    with gzip.open(gz_path, "rb") as gz:
        counts = json.loads(gz.read().decode("utf-8"))
    assert len(counts) == 48
    assert sum(sum(c.values()) for c in counts.values()) == 48 * 2048
