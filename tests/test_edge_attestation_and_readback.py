"""Beta-side checks for the bridge's A1.4-A1.6 additions, against a mock bridge that behaves like
osiris-edge-bridge `claude/a1-replay-attest-readback` (replay guard, signed attestation frame,
raw ledger readback). The real device has not run this code yet."""
from __future__ import annotations

import base64
import hashlib
import http.server
import json
import threading

import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519

from osiris_cli import edge_ipc_client as eic


def _android_canon(obj) -> str:
    # Independent of the client's implementation: json.dumps, then org.json's '/' escaping.
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).replace("/", "\\/")


class Bridge:
    def __init__(self, beta_pub: bytes):
        self.beta_pub = beta_pub
        self.alpha = ed25519.Ed25519PrivateKey.generate()
        self.alpha_pub = self.alpha.public_key().public_bytes_raw()
        self.seen: set = set()
        self.lines: list = []
        self.tamper = None
        self.append("service_boot", {"device": "samsung SM-S948U1", "soc": "SM8850 / canoe"})

    def append(self, kind, payload):
        prev = json.loads(self.lines[-1])["hash"] if self.lines else "0" * 64
        body = {"kind": kind, "ts": "2026-11-19T00:00:00Z", **payload}
        h = hashlib.sha256((prev + _android_canon(body)).encode()).hexdigest()
        self.lines.append(_android_canon({"prev": prev, **body, "hash": h}))
        return h

    def identity(self):
        return {"schema": "osiris.edge-identity.v1", "node_id": "node_alpha_s26_ultra",
                "public_key_b64": base64.b64encode(self.alpha_pub).decode(),
                "fingerprint_sha256": hashlib.sha256(self.alpha_pub).hexdigest()}


def _serve(b: Bridge):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/v1/node/identity":
                return self._json(200, b.identity())
            if self.path.startswith("/v1/ledger/rows"):
                q = dict(p.split("=") for p in self.path.split("?", 1)[1].split("&"))
                since, limit = int(q.get("since", 0)), int(q.get("limit", 500))
                head = json.loads(b.lines[-1])["hash"]
                return self._json(200, {"rows": b.lines[since:since + limit], "total_rows": len(b.lines),
                                        "head_hash": "f" * 64 if b.tamper == "head" else head, "chain_valid": True})
            self._json(404, {})

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            ts, sig = self.headers["X-Timestamp"], base64.b64decode(self.headers["X-Signature-Ed25519"])
            try:
                ed25519.Ed25519PublicKey.from_public_bytes(b.beta_pub).verify(sig, ts.encode() + b"." + body)
            except Exception:
                return self._json(401, {"error": "Ed25519 verification failed: SIGNATURE_VERIFICATION_FAILED"})
            digest = hashlib.sha256(ts.encode() + b"." + body).hexdigest()
            if digest in b.seen:
                b.append("ipc_rejected", {"reason": "REPLAYED_REQUEST"})
                return self._json(409, {"error": "Request refused: REPLAYED_REQUEST"})
            b.seen.add(digest)
            req = hashlib.sha256(body).hexdigest()
            b.append("ipc_intent_write_ahead", {"payload_sha256": req, "auth": "VERIFIED:node_beta:955752e776b3"})
            chunks = [json.dumps({"status": "Bell"}), json.dumps({"status": " state."})]
            resp = hashlib.sha256("".join(c + "\n" for c in chunks).encode()).hexdigest()
            row = b.append("ipc_intent_completed", {"payload_sha256": req, "chunks": 2, "auth": "VERIFIED"})
            att = {"schema": "osiris.edge-attestation.v1", "node_id": "node_alpha_s26_ultra",
                   "node_fingerprint_sha256": hashlib.sha256(b.alpha_pub).hexdigest(),
                   "request_sha256": "0" * 64 if b.tamper == "request" else req, "request_timestamp": ts,
                   "response_sha256": resp, "chunks": 2, "ledger_row_hash": row, "issued_at": 1}
            msg = b"OSIRIS-EDGE-ATTESTATION-V1." + json.dumps(att, sort_keys=True, separators=(",", ":")).encode()
            att["signature_b64"] = base64.b64encode(b.alpha.sign(msg)).decode()
            if b.tamper == "chunk":
                chunks[1] = json.dumps({"status": " cat."})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for c in chunks + [json.dumps({"osiris_attestation": att}).replace("/", "\\/")]:
                self.wfile.write(f"data: {c}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.delenv("OSIRIS_BRIDGE_URL", raising=False)
    monkeypatch.delenv("OSIRIS_GEMMA_ENDPOINT", raising=False)
    keys = tmp_path / "keys"
    b = Bridge(eic.get_public_key_bytes(keys))
    srv = _serve(b)
    yield b, f"http://127.0.0.1:{srv.server_address[1]}", keys
    srv.shutdown()


def _pin(b, url, keys):
    return eic.pin_node_identity(eic.fetch_node_identity(url), expected_fingerprint=b.identity()["fingerprint_sha256"][:16], key_dir=keys)


def _send(url, keys, **kw):
    return eic.send_osiris_intent("explain a Bell state", base_url=url, key_dir=keys, stream_stdout=False, **kw)


def test_unpinned_response_is_reported_as_unattested(env):
    b, url, keys = env
    r = _send(url, keys)
    assert r["ok"] and not r["response_attested"] and r["attestation_check"] == "NO_PINNED_NODE_IDENTITY"


def test_pinned_alpha_key_attests_the_response(env):
    b, url, keys = env
    _pin(b, url, keys)
    r = _send(url, keys)
    assert r["response_attested"], r["attestation_check"]
    assert [c["status"] for c in r["chunks"]] == ["Bell", " state."]  # attestation frame is not a chunk


@pytest.mark.parametrize("mode,reason", [("chunk", "RESPONSE_ALTERED_IN_TRANSIT"), ("request", "ATTESTATION_FOR_DIFFERENT_REQUEST")])
def test_altered_response_or_other_request_is_not_attested(env, mode, reason):
    b, url, keys = env
    _pin(b, url, keys)
    b.tamper = mode
    r = _send(url, keys)
    assert not r["response_attested"] and r["attestation_check"] == reason


def test_response_signed_by_a_different_key_is_not_attested(env, tmp_path):
    b, url, keys = env
    _pin(b, url, keys)
    b.alpha = ed25519.Ed25519PrivateKey.generate()
    b.alpha_pub = b.alpha.public_key().public_bytes_raw()
    assert _send(url, keys)["attestation_check"] == "ATTESTED_BY_UNPINNED_KEY"


def test_pinning_needs_the_operator_fingerprint_and_never_replaces_a_key(env, tmp_path):
    b, url, keys = env
    ident = eic.fetch_node_identity(url)
    with pytest.raises(ValueError, match="does not match"):
        eic.pin_node_identity(ident, expected_fingerprint="0" * 16, key_dir=keys)
    with pytest.raises(ValueError):
        eic.pin_node_identity(ident, expected_fingerprint=ident["fingerprint_sha256"][:8], key_dir=keys)
    first = _pin(b, url, keys)
    assert _pin(b, url, keys) == first  # same key again: unchanged
    other = ed25519.Ed25519PrivateKey.generate().public_key().public_bytes_raw()
    forged = {"public_key_b64": base64.b64encode(other).decode(), "fingerprint_sha256": hashlib.sha256(other).hexdigest()}
    with pytest.raises(ValueError, match="already pinned"):
        eic.pin_node_identity(forged, expected_fingerprint=forged["fingerprint_sha256"][:16], key_dir=keys)


def test_each_send_is_a_new_request_and_a_raw_resend_is_refused(env):
    b, url, keys = env
    r1, r2 = _send(url, keys), _send(url, keys)
    assert r1["ok"] and r2["ok"] and r1["request_id"] != r2["request_id"]
    body = json.dumps({"model": "m", "messages": [], "osiris_request_id": "fixed"}).encode()
    sig, ts = eic.sign_payload(body, key_dir=keys)
    import urllib.request, urllib.error
    def post():
        req = urllib.request.Request(url + "/v1/chat/completions", data=body, method="POST",
                                     headers={"X-Signature-Ed25519": sig, "X-Timestamp": ts, "Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(req, timeout=5).status
        except urllib.error.HTTPError as e:
            return e.code
    assert post() == 200 and post() == 409


def test_ledger_readback_recomputes_the_chain_and_finds_the_request(env):
    b, url, keys = env
    r = _send(url, keys)
    led = eic.fetch_ledger_rows(url)
    assert led["valid"] and led["head_matches_bridge"], led["problem"]
    rows = eic.ledger_rows_for_request(led, r["request_sha256"])
    assert [x["kind"] for x in rows] == ["ipc_intent_write_ahead", "ipc_intent_completed"]
    _pin(b, url, keys)
    r2 = _send(url, keys)
    led2 = eic.fetch_ledger_rows(url)
    done = [x for x in eic.ledger_rows_for_request(led2, r2["request_sha256"]) if x["kind"] == "ipc_intent_completed"]
    assert done and done[0]["hash"] == r2["attestation"]["ledger_row_hash"]  # response bound to its row


def test_ledger_readback_detects_edits_and_a_wrong_head(env):
    b, url, keys = env
    _send(url, keys)
    rows = list(b.lines)
    edited = json.loads(rows[1]); edited["auth"] = "VERIFIED:someone_else"
    rows[1] = _android_canon(edited)
    assert not eic.verify_ledger_lines(rows)["valid"]
    b.tamper = "head"
    led = eic.fetch_ledger_rows(url)
    assert led["valid"] and not led["head_matches_bridge"]


def test_canonical_json_matches_android_escaping():
    row = {"soc": "SM8850 / canoe", "q": 'a"b\\c', "n": 3, "t": True, "x": None, "u": "é"}
    assert eic.bridge_canonical_json(row) == _android_canon(row)
