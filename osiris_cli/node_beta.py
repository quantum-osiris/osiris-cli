"""Node Beta acceptor for one phone-originated instruction.

The phone signs the raw body with Ed25519, the same ``timestamp + '.' + body``
message the edge bridge already verifies. This process checks that signature,
correlates ``osiris_request_id``, resolves the instruction, and only then asks
the execution kernel to run ``sha256sum``. A repeated id returns the recorded
result and does not run the kernel again. Anything that is not a SHA-256
challenge, or that the kernel does not ALLOW, is refused before a second try
can widen it.

Run:
    python -m osiris_cli.node_beta --port 8770 --peer-pubkey-hex <64 hex chars> \\
        --root DIR --ledger DIR/ledger.jsonl --journal DIR/exec_journal.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from dnalang.ledger import Ledger

from osiris_cli.edge_ipc_client import verify_payload_signature
from osiris_cli.exec_kernel import CommandRequest, ExecJournal, run
from osiris_cli.node_ids import NODE_BETA
from osiris_cli.node_router import route_task

_SHA256_WORD = re.compile(r"\bsha-?256\b", re.I)
_CHALLENGE = re.compile(r"[A-Za-z0-9]{16,128}")
_OTHER_WORK = re.compile(
    r"\b(delete|install|deploy|push|curl|wget|ssh|sudo|pytest|compile|build|train)\b",
    re.I,
)
_CLOSED = ("beta_execute_completed", "beta_execute_failed", "beta_execute_refused")


def peer_from_hex(text: str) -> bytes:
    raw = bytes.fromhex(text.strip())
    if len(raw) != 32:
        raise ValueError("peer public key must be 32 bytes")
    return raw


def resolve_instruction(instruction: str, challenge: str) -> Tuple[bool, str, Dict[str, Any]]:
    """Deterministic intent check. No model call. Fail closed."""
    route = route_task(instruction)
    info = route.as_dict()
    if not isinstance(instruction, str) or not instruction.strip():
        return False, "instruction is empty", info
    if len(instruction) > 400:
        return False, "instruction is too long", info
    if not route.routed or route.target_node != NODE_BETA:
        return False, route.reason, info
    if route.rule_id != "N4-default-workstation":
        return False, f"route {route.rule_id} is not the workstation default", info
    if not _SHA256_WORD.search(instruction) or "challenge" not in instruction.lower():
        return False, "instruction does not ask for the SHA-256 digest of the challenge", info
    if _OTHER_WORK.search(instruction):
        return False, "instruction asks for work other than the digest", info
    if not isinstance(challenge, str) or _CHALLENGE.fullmatch(challenge) is None:
        return False, "challenge must be 16 to 128 letters or digits", info
    return True, route.rule_id, info


def _digest_token(stdout: str) -> Optional[str]:
    token = stdout.split()[0] if stdout.split() else ""
    if re.fullmatch(r"[0-9a-f]{64}", token):
        return token
    return None


class BetaSession:
    def __init__(
        self,
        *,
        peer_public: bytes,
        root: Path,
        ledger_path: Path,
        journal_path: Path,
        artifacts_dir: Path,
        runner: Callable[..., Any] = run,
    ) -> None:
        self.peer_public = peer_public
        self.fingerprint = hashlib.sha256(peer_public).hexdigest()
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger(Path(ledger_path))
        self.journal = ExecJournal(Path(journal_path))
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.runner = runner
        self._lock = threading.Lock()

    def _rows(self, request_id: str) -> list:
        return [row for row in self.ledger if row.get("request_id") == request_id]

    def _closed(self, request_id: str) -> Optional[Dict[str, Any]]:
        found = [row for row in self._rows(request_id) if row.get("kind") in _CLOSED]
        return found[-1] if found else None

    def _result_from_row(self, row: Dict[str, Any], *, replayed: bool) -> Tuple[int, Dict[str, Any]]:
        kind = row.get("kind")
        body = {
            "ok": kind == "beta_execute_completed",
            "replayed": replayed,
            "request_id": row.get("request_id"),
            "digest_sha256": row.get("digest_sha256"),
            "kernel_decision": row.get("kernel_decision"),
            "ledger_hash": row.get("hash"),
            "peer_fingerprint_sha256": self.fingerprint,
            "error": row.get("error"),
        }
        code = 200 if body["ok"] else 422
        if kind == "beta_execute_refused":
            code = 422
        return code, body

    def handle(self, payload: bytes, signature_b64: str, timestamp: str) -> Tuple[int, Dict[str, Any]]:
        with self._lock:
            return self._handle_locked(payload, signature_b64, timestamp)

    def _handle_locked(self, payload: bytes, signature_b64: str, timestamp: str) -> Tuple[int, Dict[str, Any]]:
        ok, why = verify_payload_signature(payload, signature_b64 or "", timestamp or "", self.peer_public)
        if not ok:
            self.ledger.append("beta_rejected", {"reason": why})
            return 401, {"ok": False, "error": why, "peer_fingerprint_sha256": self.fingerprint}

        try:
            data = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.ledger.append("beta_rejected", {"reason": "BODY_NOT_JSON"})
            return 400, {"ok": False, "error": "BODY_NOT_JSON"}
        if not isinstance(data, dict):
            return 400, {"ok": False, "error": "BODY_NOT_OBJECT"}

        request_id = data.get("osiris_request_id")
        instruction = data.get("instruction")
        challenge = data.get("challenge")
        if not isinstance(request_id, str) or re.fullmatch(r"[0-9a-f]{16,64}", request_id) is None:
            return 400, {"ok": False, "error": "BAD_REQUEST_ID"}

        prior = self._closed(request_id)
        if prior is not None:
            same = prior.get("body_sha256") == hashlib.sha256(payload).hexdigest()
            if not same:
                self.ledger.append("beta_rejected", {"reason": "REQUEST_ID_REUSED", "request_id": request_id})
                return 409, {"ok": False, "error": "REQUEST_ID_REUSED", "request_id": request_id}
            self.ledger.append(
                "beta_execute_replayed",
                {"request_id": request_id, "original_hash": prior.get("hash"),
                 "digest_sha256": prior.get("digest_sha256")},
            )
            code, body = self._result_from_row(prior, replayed=True)
            body["ledger_hash"] = prior.get("hash")
            return code, body

        if any(row.get("kind") == "beta_execute_write_ahead" for row in self._rows(request_id)):
            return 409, {"ok": False, "error": "PRIOR_ATTEMPT_INCOMPLETE", "request_id": request_id}

        accepted, reason, route_info = resolve_instruction(
            instruction if isinstance(instruction, str) else "",
            challenge if isinstance(challenge, str) else "",
        )
        body_sha = hashlib.sha256(payload).hexdigest()
        if not accepted:
            row = self.ledger.append(
                "beta_execute_refused",
                {"request_id": request_id, "body_sha256": body_sha, "error": reason,
                 "route": route_info},
            )
            return 422, {"ok": False, "error": reason, "request_id": request_id,
                         "ledger_hash": row["hash"], "replayed": False,
                         "peer_fingerprint_sha256": self.fingerprint}

        self.ledger.append(
            "beta_execute_write_ahead",
            {"request_id": request_id, "body_sha256": body_sha, "route_rule": reason,
             "challenge": challenge, "instruction": instruction},
        )
        work = self.root / request_id
        work.mkdir(parents=True, exist_ok=False)
        challenge_path = work / "challenge.bin"
        challenge_path.write_bytes(challenge.encode("ascii"))
        try:
            result = self.runner(
                CommandRequest(
                    argv=("sha256sum", str(challenge_path)),
                    cwd=str(self.root),
                    origin="human",
                    root=str(self.root),
                    timeout_s=10,
                ),
                None,
                self.journal,
                artifacts_dir=self.artifacts_dir,
            )
        except Exception as exc:
            row = self.ledger.append(
                "beta_execute_failed",
                {"request_id": request_id, "body_sha256": body_sha, "error": f"kernel raised {exc.__class__.__name__}"},
            )
            return 422, {"ok": False, "error": "KERNEL_ERROR", "request_id": request_id, "ledger_hash": row["hash"]}

        digest = _digest_token(result.stdout_view or "")
        preimage = hashlib.sha256(challenge.encode("ascii")).hexdigest()
        mismatch = digest != preimage
        if result.decision != "ALLOW" or result.exit_code != 0 or digest is None or mismatch:
            error = "KERNEL_DIGEST_MISMATCH" if mismatch else f"KERNEL_{result.decision}"
            row = self.ledger.append(
                "beta_execute_failed",
                {"request_id": request_id, "body_sha256": body_sha, "error": error,
                 "kernel_decision": result.decision, "kernel_exit": result.exit_code,
                 "kernel_receipt": result.receipt_id},
            )
            return 422, {"ok": False, "error": error, "request_id": request_id,
                         "kernel_decision": result.decision, "ledger_hash": row["hash"], "replayed": False}

        row = self.ledger.append(
            "beta_execute_completed",
            {"request_id": request_id, "body_sha256": body_sha, "challenge": challenge,
             "instruction": instruction, "digest_sha256": digest, "kernel_decision": result.decision,
             "kernel_exit": result.exit_code, "kernel_receipt": result.receipt_id,
             "stdout_sha256": result.stdout_sha256, "route_rule": reason,
             "peer_fingerprint_sha256": self.fingerprint},
        )
        return 200, {
            "ok": True,
            "replayed": False,
            "request_id": request_id,
            "digest_sha256": digest,
            "kernel_decision": result.decision,
            "kernel_receipt": result.receipt_id,
            "ledger_hash": row["hash"],
            "route_rule": reason,
            "peer_fingerprint_sha256": self.fingerprint,
        }

    def ledger_ok(self) -> bool:
        return self.ledger.verify() is None


def make_server(session: BetaSession, host: str, port: int) -> ThreadingHTTPServer:
    holder = {"session": session}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _send(self, code: int, payload: Dict[str, Any]) -> None:
            raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/healthz":
                self._send(404, {"ok": False, "error": "NOT_FOUND"})
                return
            session_ = holder["session"]
            problem = session_.ledger.verify()
            self._send(200, {
                "ok": problem is None,
                "service": "node-beta",
                "ledger_ok": problem is None,
                "ledger_error": problem,
                "peer_fingerprint_sha256": session_.fingerprint,
            })

        def do_POST(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/v1/execute":
                self._send(404, {"ok": False, "error": "NOT_FOUND"})
                return
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length < 0 or length > 8192:
                self._send(400, {"ok": False, "error": "BODY_SIZE"})
                return
            body = self.rfile.read(length)
            code, payload = holder["session"].handle(
                body,
                self.headers.get("X-Signature-Ed25519", ""),
                self.headers.get("X-Timestamp", ""),
            )
            self._send(code, payload)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m osiris_cli.node_beta")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--peer-pubkey-hex", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.bind != "127.0.0.1":
        raise SystemExit("refusing to bind anywhere but 127.0.0.1")
    session = BetaSession(
        peer_public=peer_from_hex(args.peer_pubkey_hex),
        root=args.root,
        ledger_path=args.ledger,
        journal_path=args.journal,
        artifacts_dir=args.artifacts or (args.root / "artifacts"),
    )
    server = make_server(session, args.bind, args.port)
    print(f"node-beta 127.0.0.1:{args.port} peer {session.fingerprint}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
