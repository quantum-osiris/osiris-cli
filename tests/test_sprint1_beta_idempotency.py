"""Node Beta request ids are idempotency keys."""

from __future__ import annotations

import hashlib
import json

import pytest

pytest.importorskip("dnalang")
pytest_plugins = ["tests.test_node_beta"]

from osiris_cli.substrate_guard import begin_experiment
from tests.test_node_beta import _body


def test_duplicate_delivery_runs_the_kernel_once(signed_client):
    post, session = signed_client
    calls = {"n": 0}
    original = session.runner

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    session.runner = counting
    challenge = "e" * 32
    request_id = "abcdef0123456789"
    code, body = post(_body(challenge, request_id))
    assert code == 200, body
    code2, body2 = post(_body(challenge, request_id))
    assert code2 == 200 and body2["replayed"] is True
    assert calls["n"] == 1


def test_crash_after_claim_does_not_run_the_kernel(signed_client):
    post, session = signed_client
    body = _body("f" * 32, "fedcba9876543210")
    raw = json.dumps(body, separators=(",", ":")).encode()
    begin_experiment(session.idempotency_path, body["osiris_request_id"], body["osiris_request_id"], hashlib.sha256(raw).hexdigest())
    calls = {"n": 0}
    original = session.runner

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    session.runner = counting
    code, response = post(body)
    assert code == 409
    assert response["error"] == "PRIOR_ATTEMPT_INCOMPLETE"
    assert calls["n"] == 0


def test_replay_after_success_returns_the_stored_body(tmp_path: Path, signed_client):
    post, session = signed_client
    challenge = "a1b2c3d4e5f67890"
    request_id = "0011223344556677"
    code, body = post(_body(challenge, request_id))
    assert code == 200 and body["replayed"] is False
    # A second delivery is exactly-once even though the idempotency row is acked.
    code2, body2 = post(_body(challenge, request_id))
    assert code2 == 200 and body2["replayed"] is True
    assert body2["digest_sha256"] == hashlib.sha256(challenge.encode()).hexdigest()
    rows = session.idempotency_path.read_text(encoding="utf-8")
    assert request_id in rows
    assert '"phase": "acked"' in rows or '"phase":"acked"' in rows
