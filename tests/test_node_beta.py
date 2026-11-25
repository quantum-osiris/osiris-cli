"""Phone-shaped client against the Node Beta acceptor. The digest comes from sha256sum."""
from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

pytest.importorskip("dnalang")

from osiris_cli.node_beta import BetaSession, make_server, resolve_instruction


@pytest.fixture()
def signed_client(tmp_path: Path):
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes_raw()
    session = BetaSession(
        peer_public=public,
        root=tmp_path / "root",
        ledger_path=tmp_path / "ledger.jsonl",
        journal_path=tmp_path / "journal.jsonl",
        artifacts_dir=tmp_path / "artifacts",
    )
    server = make_server(session, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]

    def post(body: dict, *, signature: bytes | None = None, timestamp: int | None = None,
             raw: bytes | None = None) -> tuple[int, dict]:
        payload = raw if raw is not None else json.dumps(body, separators=(",", ":")).encode()
        ts = str(int(time.time()) if timestamp is None else timestamp)
        sig = signature if signature is not None else private.sign(ts.encode() + b"." + payload)
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/execute",
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json",
                     "X-Signature-Ed25519": __import__("base64").b64encode(sig).decode(),
                     "X-Timestamp": ts},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode())

    yield post, session
    server.shutdown()


def _journal(session: BetaSession) -> str:
    path = session.journal.path
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _body(challenge: str = "a" * 32, request_id: str = "abc123abc123abcd") -> dict:
    return {
        "osiris_request_id": request_id,
        "instruction": "Compute the SHA-256 digest of the challenge on the workstation.",
        "challenge": challenge,
    }


def test_instruction_that_names_the_phone_is_refused():
    ok, reason, _info = resolve_instruction(
        "Compute the SHA-256 digest of the challenge on the phone.", "a" * 32)
    assert ok is False
    assert "phone" in reason


def test_unsigned_request_is_refused_and_does_not_run(signed_client):
    post, session = signed_client
    code, body = post(_body(), signature=b"\x00" * 64)
    assert code == 401
    assert body["ok"] is False
    assert _journal(session) == ""
    assert session.ledger.verify() is None


def test_sha256_challenge_runs_once_and_replays_the_receipt(signed_client):
    post, session = signed_client
    challenge = "c" * 32
    code, body = post(_body(challenge, "0123456789abcdef"))
    assert code == 200, body
    assert body["replayed"] is False
    assert body["kernel_decision"] == "ALLOW"
    assert body["digest_sha256"] == hashlib.sha256(challenge.encode()).hexdigest()
    assert body["peer_fingerprint_sha256"] == session.fingerprint

    code2, body2 = post(_body(challenge, "0123456789abcdef"))
    assert code2 == 200, body2
    assert body2["replayed"] is True
    assert body2["digest_sha256"] == body["digest_sha256"]
    assert body2["ledger_hash"] == body["ledger_hash"]
    journal_lines = [line for line in session.journal.path.read_text(encoding="utf-8").splitlines() if line]
    assert len(journal_lines) == 1
    assert session.ledger.verify() is None


def test_other_work_and_reused_id_do_not_execute(signed_client):
    post, session = signed_client
    bad = _body()
    bad["instruction"] = "Compute the SHA-256 digest of the challenge, then run pytest."
    code, body = post(bad)
    assert code == 422
    assert body["ok"] is False
    assert _journal(session) == ""

    code2, again = post(bad)
    assert code2 == 422
    assert again["replayed"] is True
    assert _journal(session) == ""

    other = _body(request_id="0123456789abcdef")
    other["challenge"] = "d" * 16
    code3, conflict = post(other | {"osiris_request_id": bad["osiris_request_id"]})
    assert code3 == 409
    assert conflict["error"] == "REQUEST_ID_REUSED"
