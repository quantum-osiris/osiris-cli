"""CORRELATED-NOISE-01 Stage 1 on `ibm_marrakesh` (job `dbtp67dav06s73cfbsm0`): the archived files agree with
each other, carry no IBM Cloud account identifiers, and both analyses reproduce from the archived counts."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict

import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_ibm_runtime")  # CI does not install qiskit-ibm-runtime

from phase3.cn01_amendment_a3 import a3_inference  # noqa: E402
from phase3.execute_and_analyze_live import (  # noqa: E402
    analyze_correlated_noise_01_live_counts,
    cn01_fidelity_arrays_from_counts,
)

DOCS = Path(__file__).resolve().parents[1] / "docs" / "phase3"
JOB_ID = "dbtp67dav06s73cfbsm0"
LOCK_SHA = "38ebfe7f7c007633281039bca551af963e8335db54e6e6915a1e3f5be4a64fd8"
QPY_SHA = "158de03260c429c8170cd3e3b296ade8045881e385370f1774619344e57b9da1"


def _load(name: str) -> Any:
    return json.loads((DOCS / name).read_text())


def _utc(stamp: str) -> dt.datetime:
    return dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def _lock() -> Dict[str, Any]:
    path = DOCS / "correlated_noise_01_live_prereg_lock.json"
    lock = json.loads(path.read_text())
    lock["lock_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return lock


def _assert_close(got: Any, want: Any, where: str = "") -> None:
    """Exact for strings, bools, ints and structure; floats to 1e-6 relative (fits may differ across BLAS builds)."""
    if isinstance(want, dict):
        assert isinstance(got, dict) and set(got) == set(want), where
        for k in want:
            _assert_close(got[k], want[k], f"{where}.{k}")
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), where
        for i, (g, w) in enumerate(zip(got, want)):
            _assert_close(g, w, f"{where}[{i}]")
    elif isinstance(want, float) and not isinstance(got, bool):
        assert isinstance(got, (int, float)), where
        assert (math.isnan(got) and math.isnan(want)) or got == pytest.approx(want, rel=1e-6, abs=1e-9), where
    else:
        assert got == want, where


def test_artifacts_name_the_same_job_lock_and_circuits() -> None:
    assert hashlib.sha256((DOCS / "correlated_noise_01_live_prereg_lock.json").read_bytes()).hexdigest() == LOCK_SHA
    assert hashlib.sha256((DOCS / "correlated_noise_01_live_pubs.qpy").read_bytes()).hexdigest() == QPY_SHA
    receipt = _load("correlated_noise_01_submission_receipt.json")
    raw = _load("correlated_noise_01_live_raw_counts.json")
    usage = _load("correlated_noise_01_job_usage.json")
    locked = _load("correlated_noise_01_live_analysis.json")
    amended = _load("correlated_noise_01_live_analysis_A3.json")
    assert receipt["job_id"] == raw["job_id"] == usage["job_id"] == locked["job_id"] == amended["job_id"] == JOB_ID
    for record in (receipt, locked, amended):
        assert record["prereg_lock_sha256"] == LOCK_SHA
        assert record["qpy_sha256"] == QPY_SHA
    lock = _lock()
    assert len(raw["pub_counts"]) == lock["total_pubs"] == 244
    assert all(sum(c.values()) == lock["shots_per_pub"] for c in raw["pub_counts"])
    # The saved counts were compared PUB by PUB with IBM's stored result for this job (read-only re-query).
    assert usage["counts_cross_check"] == {**usage["counts_cross_check"], "ibm_result_pubs": 244, "saved_pubs": 244, "mismatched_pubs": 0}


def test_raw_counts_carry_no_account_identifiers() -> None:
    raw = _load("correlated_noise_01_live_raw_counts.json")
    assert not {"instance_id", "plan_id"} & set(raw["post_job_usage"])
    assert raw["redacted_fields"] == ["post_job_usage.instance_id", "post_job_usage.plan_id"]
    assert "crn:" not in (DOCS / "correlated_noise_01_live_raw_counts.json").read_text()


def test_recorded_times_are_in_order() -> None:
    """Lock, then A-3, then the job. A-3's `recorded_utc` is self-reported; the public record is the git history."""
    usage = _load("correlated_noise_01_job_usage.json")
    created = _utc(usage["metrics_timestamps"]["created"])
    assert _utc(_lock()["locked_utc"]) < _utc(_load("correlated_noise_01_A3_amendment.json")["recorded_utc"]) < created
    assert abs((_utc(_load("correlated_noise_01_submission_receipt.json")["submitted_utc"]) - created).total_seconds()) < 5
    assert created < _utc(usage["metrics_timestamps"]["finished"])


def test_locked_analysis_reproduces_from_the_archived_counts() -> None:
    raw = _load("correlated_noise_01_live_raw_counts.json")
    out = analyze_correlated_noise_01_live_counts(_lock(), raw["pub_counts"], raw["execution_time_calibration"], raw["job_id"])
    _assert_close(json.loads(json.dumps(out)), _load("correlated_noise_01_live_analysis.json"))


def test_a3_analysis_reproduces_from_the_archived_counts() -> None:
    lock = _lock()
    raw = _load("correlated_noise_01_live_raw_counts.json")
    amendment = _load("correlated_noise_01_A3_amendment.json")
    archived = _load("correlated_noise_01_live_analysis_A3.json")
    out = a3_inference(
        cn01_fidelity_arrays_from_counts(lock, raw["pub_counts"]), lock["coupler_cohort"],
        amendment["train_slots"], raw["execution_time_calibration"],
    )
    out = json.loads(json.dumps(out))
    _assert_close(out, {k: archived[k] for k in out})
    assert archived["analysis_amendment_sha256"] == hashlib.sha256(
        (DOCS / "correlated_noise_01_A3_amendment.json").read_bytes()
    ).hexdigest()
