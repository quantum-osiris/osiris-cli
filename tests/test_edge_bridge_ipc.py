"""Tests for OSIRIS Edge Bridge Ed25519 IPC signing, /healthz, /readyz, router, and key sync."""
from __future__ import annotations

import base64
import hashlib
import http.server
import json
import threading
import time
from pathlib import Path

from osiris_cli.edge_ipc_client import (
    check_bridge_health,
    check_ledger_readiness,
    enroll_edge_key,
    get_public_key_bytes,
    send_osiris_intent,
    sign_payload,
    verify_payload_signature,
)
from osiris_cli.edge_router import OSIRISCommandRouter
from osiris_cli.key_sync import DualSubstrateKeySync


class _MockEdgeBridgeState:
    def __init__(self) -> None:
        self.trusted_keys: list[bytes] = []
        self.ledger: list[dict] = []
        self.corrupt_chain = False

    def append_ledger(self, kind: str, payload: dict) -> str:
        prev = self.ledger[-1]["hash"] if self.ledger else "0" * 64
        body = {"kind": kind, "ts": "2026-11-16T00:00:00Z", **payload}
        canon = json.dumps(body, sort_keys=True, separators=(",", ":"))
        h = hashlib.sha256((prev + canon).encode("utf-8")).hexdigest()
        self.ledger.append({"prev": prev, **body, "hash": h})
        return h

    def verify_ledger(self) -> tuple[bool, str]:
        if self.corrupt_chain:
            return False, "tampered"
        prev = "0" * 64
        for row in self.ledger:
            body = {k: v for k, v in row.items() if k not in ("prev", "hash")}
            want = hashlib.sha256((prev + json.dumps(body, sort_keys=True, separators=(",", ":"))).encode()).hexdigest()
            if row["prev"] != prev or row["hash"] != want:
                return False, "hash mismatch"
            prev = row["hash"]
        return True, prev


def _start_mock_server(state: _MockEdgeBridgeState):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            if self.path == "/healthz":
                body = json.dumps(
                    {
                        "status": "ONLINE",
                        "device_model": "SM-S948U1",
                        "backend": "Snapdragon 8 Elite Test Kernel",
                        "enrolled_keys": len(state.trusted_keys),
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/readyz":
                ok, head = state.verify_ledger()
                code = 200 if ok else 503
                body = json.dumps(
                    {
                        "status": "READY" if ok else "CORRUPTED",
                        "chain_valid": ok,
                        "rows": len(state.ledger),
                        "head_hash": head,
                    }
                ).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            if self.path == "/v1/keys/enroll":
                obj = json.loads(raw.decode("utf-8"))
                pub = base64.b64decode(obj["public_key_b64"])
                state.trusted_keys.append(pub)
                lh = state.append_ledger("key_enrolled", {"node_id": obj["node_id"]})
                resp = json.dumps(
                    {
                        "status": "ENROLLED",
                        "node_id": obj["node_id"],
                        "fingerprint_sha256": hashlib.sha256(pub).hexdigest(),
                        "ledger_hash": lh,
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
            elif self.path == "/v1/chat/completions":
                sig = self.headers.get("X-Signature-Ed25519", "")
                ts = self.headers.get("X-Timestamp", "")
                verified = any(
                    verify_payload_signature(raw, sig, ts, pub)[0] for pub in state.trusted_keys
                )
                if not verified:
                    err = b'{"error":"SIGNATURE_VERIFICATION_FAILED"}'
                    self.send_response(401)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err)))
                    self.end_headers()
                    self.wfile.write(err)
                    return
                state.append_ledger("ipc_intent", {"bytes": len(raw)})
                sse = b'data: {"status": "ACKNOWLEDGED"}\n\ndata: [DONE]\n\n'
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(sse)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


def test_ed25519_enroll_healthz_readyz_and_signed_sse(tmp_path: Path):
    state = _MockEdgeBridgeState()
    state.append_ledger("boot", {"model": "SM-S948U1"})
    srv = _start_mock_server(state)
    base_url = f"http://127.0.0.1:{srv.server_address[1]}"

    try:
        # 1. Before enrollment, signed intent is rejected (401) because key is not enrolled
        rejected = send_osiris_intent(
            "Unauthorized probe", base_url=base_url, key_dir=tmp_path, stream_stdout=False
        )
        assert not rejected["ok"]
        assert rejected["http_code"] == 401

        # 2. Enroll via router
        assert OSIRISCommandRouter(["bridge", "enroll", "test_node"], base_url=base_url, key_dir=tmp_path).route() == 0
        assert OSIRISCommandRouter(["bridge", "status"], base_url=base_url, key_dir=tmp_path).route() == 0
        assert OSIRISCommandRouter(["bridge", "readyz"], base_url=base_url, key_dir=tmp_path).route() == 0

        # 3. Signed intent succeeds and streams ACKNOWLEDGED
        ok_res = send_osiris_intent(
            "Synthesize routing table", base_url=base_url, key_dir=tmp_path, stream_stdout=False
        )
        assert ok_res["ok"]
        assert ok_res["chunks"][0]["status"] == "ACKNOWLEDGED"

        # 4. Replay drift (> 300s) is rejected by signature verifier
        pub = get_public_key_bytes(tmp_path)
        sig, stale_ts = sign_payload(b"{}", key_dir=tmp_path, timestamp=int(time.time()) - 900)
        valid, reason = verify_payload_signature(b"{}", sig, stale_ts, pub)
        assert not valid and reason == "TIMESTAMP_DRIFT_EXCEEDED"

        # 5. Tampered ledger fails /readyz
        state.corrupt_chain = True
        ready_bad = check_ledger_readiness(base_url=base_url)
        assert ready_bad["status"] == "NOT_READY"
        assert not ready_bad["chain_valid"]

        # 6. DualSubstrateKeySync registers trusted keys & writes receipt
        state.corrupt_chain = False
        syncer = DualSubstrateKeySync(osiris_home=tmp_path, bridge_url=base_url)
        receipt = syncer.sync_via_adb(node_id="node_beta_test")
        assert receipt["edge_bridge_enrollment"]["status"] == "ENROLLED"
        assert (tmp_path / "trusted_edge_keys.json").exists()
        assert (tmp_path / "node_beta_sync_receipt.json").exists()
    finally:
        srv.shutdown()
