"""`_job_usage_seconds` must wait for IBM to finalise a job's usage instead of recording 0."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_ibm_runtime")

from phase3.execute_and_analyze_live import _job_usage_seconds  # noqa: E402


class _FakeJob:
    def __init__(self, metrics_seq: List[Dict[str, Any]], usage_seq: List[float]):
        self._metrics = list(metrics_seq)
        self._usage = list(usage_seq)

    def metrics(self) -> Dict[str, Any]:
        return self._metrics.pop(0) if len(self._metrics) > 1 else self._metrics[0]

    def usage(self) -> float:
        return self._usage.pop(0) if len(self._usage) > 1 else self._usage[0]


def test_waits_for_finalised_usage() -> None:
    pending = {"usage": {"status": "pending"}}
    done = {"usage": {"status": "complete", "qpu_charge_time_seconds": 53}}
    job = _FakeJob([pending, pending, done], [0.0])
    assert _job_usage_seconds(job, wait_s=5.0, poll_s=0.0) == 53.0


def test_accepts_positive_job_usage() -> None:
    job = _FakeJob([{"usage": {}}], [0.0, 12.0])
    assert _job_usage_seconds(job, wait_s=5.0, poll_s=0.0) == 12.0


def test_legacy_metrics_key_and_timeout() -> None:
    job = _FakeJob([{"usage": {"quantum_seconds": 7, "status": "pending"}}], [0.0])
    assert _job_usage_seconds(job, wait_s=0.0, poll_s=0.0) == 7.0
    assert _job_usage_seconds(_FakeJob([{}], [0.0]), wait_s=0.0, poll_s=0.0) is None
