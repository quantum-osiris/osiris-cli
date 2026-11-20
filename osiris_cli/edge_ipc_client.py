#!/usr/bin/env python3
"""
OSIRIS Termux / WSL-to-Android Edge Bridge IPC Client
=====================================================
Dispatches Ed25519-signed task intents (`X-Signature-Ed25519`, `X-Timestamp`),
queries `/healthz` liveness, verifies `/readyz` SHA-256 hash-chained ledger integrity,
and enrolls Ed25519 public keys with `com.osiris.edge.bridge` on 127.0.0.1:8080.

Key Separation Note (SYS-021 / F-21):
Uses `~/.osiris/edge_ed25519_private.key` + `~/.osiris/ed25519_public.pub` exclusively
for edge IPC transport authentication—strictly isolated from `node_beta_permit_hmac.key`
and `node_beta_operator.key`.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, List, Optional, Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
WIN_CURL_INTEROP = "/mnt/c/Windows/System32/curl.exe"


def _can_use_wsl_curl_bridge(url: str) -> bool:
    if not url.startswith("http://127.0.0.1:") and not url.startswith("http://localhost:"):
        return False
    if not os.path.exists(WIN_CURL_INTEROP):
        return False
    return os.path.exists("/proc/version") and "microsoft" in pathlib.Path("/proc/version").read_text(errors="ignore").lower()


def _wsl_curl_request(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    data: Optional[bytes] = None,
    timeout: float = 10.0,
) -> Tuple[int, str]:
    cmd: List[str] = [
        WIN_CURL_INTEROP,
        "-sS",
        "-X",
        method,
        "--max-time",
        str(max(1, int(timeout))),
        "-w",
        "\n__HTTP_CODE__:%{http_code}",
    ]
    for k, v in (headers or {}).items():
        cmd.extend(["-H", f"{k}: {v}"])
    if data is not None:
        cmd.extend(["--data-binary", "@-"])
    proc = subprocess.run(cmd + [url], input=data, capture_output=True, check=False)
    out = proc.stdout.decode("utf-8", errors="replace")
    if "\n__HTTP_CODE__:" in out:
        body, code_str = out.rsplit("\n__HTTP_CODE__:", 1)
        try:
            code = int(code_str.strip())
        except ValueError:
            code = 0
        if code > 0:
            return code, body
    raise urllib.error.URLError(proc.stderr.decode("utf-8", errors="replace").strip() or "WSL curl bridge failed")


def resolve_base_url(override: Optional[str] = None) -> str:
    if override:
        return override.rstrip("/")
    env_ep = os.environ.get("OSIRIS_GEMMA_ENDPOINT") or os.environ.get("OSIRIS_BRIDGE_URL")
    if env_ep:
        marker = "/v1/chat/completions"
        if env_ep.endswith(marker):
            return env_ep[: -len(marker)].rstrip("/")
        return env_ep.rstrip("/")
    return DEFAULT_BASE_URL


def default_osiris_dir() -> pathlib.Path:
    override = os.environ.get("OSIRIS_EDGE_KEY_DIR")
    if override:
        return pathlib.Path(override)
    return pathlib.Path.home() / ".osiris"


def get_private_key(key_dir: Optional[pathlib.Path] = None) -> ed25519.Ed25519PrivateKey:
    """Loads or initializes the dedicated Edge IPC Ed25519 keypair."""
    base = pathlib.Path(key_dir) if key_dir else default_osiris_dir()
    base.mkdir(parents=True, exist_ok=True)
    priv_path = base / "edge_ed25519_private.key"
    pub_path = base / "ed25519_public.pub"

    if priv_path.exists():
        raw = priv_path.read_bytes()
        priv = ed25519.Ed25519PrivateKey.from_private_bytes(raw)
    else:
        priv = ed25519.Ed25519PrivateKey.generate()
        priv_path.write_bytes(priv.private_bytes_raw())
        try:
            os.chmod(priv_path, 0o600)
        except OSError:
            pass

    pub_bytes = priv.public_key().public_bytes_raw()
    if not pub_path.exists() or pub_path.read_bytes() != pub_bytes:
        pub_path.write_bytes(pub_bytes)
    return priv


def get_public_key_bytes(key_dir: Optional[pathlib.Path] = None) -> bytes:
    priv = get_private_key(key_dir)
    return priv.public_key().public_bytes_raw()


def public_key_fingerprint(pub_bytes: bytes) -> str:
    return hashlib.sha256(pub_bytes).hexdigest()


def sign_payload(
    payload_bytes: bytes,
    *,
    key_dir: Optional[pathlib.Path] = None,
    timestamp: Optional[int] = None,
) -> Tuple[str, str]:
    priv_key = get_private_key(key_dir)
    ts_str = str(int(time.time()) if timestamp is None else int(timestamp))
    message = ts_str.encode("utf-8") + b"." + payload_bytes
    signature = priv_key.sign(message)
    sig_b64 = base64.b64encode(signature).decode("ascii")
    return sig_b64, ts_str


def verify_payload_signature(
    payload_bytes: bytes,
    signature_b64: str,
    timestamp_str: str,
    public_key_raw: bytes,
    *,
    max_drift_seconds: int = 300,
    now_ts: Optional[int] = None,
) -> Tuple[bool, str]:
    try:
        req_ts = int(timestamp_str)
    except ValueError:
        return False, "INVALID_TIMESTAMP"

    cur_ts = int(time.time()) if now_ts is None else int(now_ts)
    if abs(cur_ts - req_ts) > max_drift_seconds:
        return False, "TIMESTAMP_DRIFT_EXCEEDED"

    try:
        sig = base64.b64decode(signature_b64)
        pub = ed25519.Ed25519PublicKey.from_public_bytes(public_key_raw)
        pub.verify(sig, timestamp_str.encode("utf-8") + b"." + payload_bytes)
        return True, "VERIFIED"
    except (ValueError, InvalidSignature):
        return False, "SIGNATURE_VERIFICATION_FAILED"


def check_bridge_health(base_url: Optional[str] = None, timeout: float = 3.0) -> Dict[str, Any]:
    url = f"{resolve_base_url(base_url)}/healthz"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                return {"status": "ONLINE", "details": data}
    except Exception as exc:
        if _can_use_wsl_curl_bridge(url):
            try:
                code, body = _wsl_curl_request("GET", url, timeout=timeout)
                if code == 200:
                    return {"status": "ONLINE", "details": json.loads(body)}
            except Exception:
                pass
        return {"status": "OFFLINE", "error": str(exc)}
    return {"status": "OFFLINE", "error": "Unknown failure"}


def check_ledger_readiness(base_url: Optional[str] = None, timeout: float = 5.0) -> Dict[str, Any]:
    url = f"{resolve_base_url(base_url)}/readyz"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                return {"status": "READY", "chain_valid": bool(data.get("chain_valid", True)), "details": data}
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            pass
        return {"status": "NOT_READY", "chain_valid": False, "error": f"HTTP {exc.code}: {body or 'Ledger verification failed.'}"}
    except Exception as exc:
        if _can_use_wsl_curl_bridge(url):
            try:
                code, body = _wsl_curl_request("GET", url, timeout=timeout)
                if code == 200:
                    data = json.loads(body)
                    return {"status": "READY", "chain_valid": bool(data.get("chain_valid", True)), "details": data}
                return {"status": "NOT_READY", "chain_valid": False, "error": f"HTTP {code}: {body}"}
            except Exception:
                pass
        return {"status": "OFFLINE", "chain_valid": False, "error": str(exc)}
    return {"status": "OFFLINE", "chain_valid": False, "error": "Unknown failure"}


def enroll_edge_key(
    node_id: str = "node_beta_wsl_workstation",
    *,
    base_url: Optional[str] = None,
    key_dir: Optional[pathlib.Path] = None,
    timeout: float = 5.0,
) -> Dict[str, Any]:
    pub_bytes = get_public_key_bytes(key_dir)
    body = json.dumps(
        {
            "node_id": node_id,
            "public_key_b64": base64.b64encode(pub_bytes).decode("ascii"),
        }
    ).encode("utf-8")
    sig_b64, ts_str = sign_payload(body, key_dir=key_dir)
    url = f"{resolve_base_url(base_url)}/v1/keys/enroll"
    hdrs = {
        "Content-Type": "application/json",
        "X-Signature-Ed25519": sig_b64,
        "X-Timestamp": ts_str,
    }
    req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_txt = exc.read().decode("utf-8", errors="replace")
        return {"status": "ERROR", "http_code": exc.code, "error": err_txt}
    except Exception as exc:
        if _can_use_wsl_curl_bridge(url):
            try:
                code, resp_body = _wsl_curl_request("POST", url, headers=hdrs, data=body, timeout=timeout)
                if code == 200:
                    return json.loads(resp_body)
                return {"status": "ERROR", "http_code": code, "error": resp_body}
            except Exception:
                pass
        return {"status": "OFFLINE", "error": str(exc)}


ATTESTATION_SCHEMA = "osiris.edge-attestation.v1"
ATTESTATION_DOMAIN = b"OSIRIS-EDGE-ATTESTATION-V1."
PINNED_IDENTITY_FILE = "node_alpha_identity.json"


def _parse_sse(lines) -> Dict[str, Any]:
    """Split an SSE stream into content chunks and the bridge's attestation frame.

    ``response_sha256`` is computed the way the bridge computes it: SHA-256 over each content
    chunk's JSON text followed by a newline (the attestation frame itself is not included).
    """
    chunks: List[Any] = []
    raw: List[str] = []
    attestation = None
    attestation_error = None
    done = False
    for line in lines:
        decoded = line.decode("utf-8", errors="replace") if isinstance(line, bytes) else line
        decoded = decoded.strip()
        if not decoded.startswith("data: "):
            continue
        content = decoded[6:]
        if content == "[DONE]":
            done = True
            break
        try:
            chunk = json.loads(content)
        except json.JSONDecodeError:
            raw.append(content)
            chunks.append({"raw": content})
            continue
        if isinstance(chunk, dict) and "osiris_attestation" in chunk:
            attestation = chunk["osiris_attestation"]
            continue
        if isinstance(chunk, dict) and "osiris_attestation_error" in chunk:
            attestation_error = str(chunk["osiris_attestation_error"])
            continue
        raw.append(content)
        chunks.append(chunk)
    digest = hashlib.sha256()
    for c in raw:
        digest.update(c.encode("utf-8") + b"\n")
    return {"chunks": chunks, "done": done, "attestation": attestation,
            "attestation_error": attestation_error, "response_sha256": digest.hexdigest()}


def _attestation_message(att: Dict[str, Any]) -> bytes:
    fields = {k: v for k, v in att.items() if k != "signature_b64"}
    return ATTESTATION_DOMAIN + json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")


def verify_attestation(
    att: Optional[Dict[str, Any]],
    pinned: Optional[Dict[str, Any]],
    *,
    request_sha256: str,
    response_sha256: str,
    request_timestamp: str,
) -> Tuple[bool, str]:
    """Check a bridge response attestation against the Alpha key pinned on this machine."""
    if pinned is None:
        return False, "NO_PINNED_NODE_IDENTITY"
    if not isinstance(att, dict):
        return False, "RESPONSE_NOT_ATTESTED"
    if att.get("schema") != ATTESTATION_SCHEMA:
        return False, "UNKNOWN_ATTESTATION_SCHEMA"
    if att.get("node_fingerprint_sha256") != pinned.get("fingerprint_sha256"):
        return False, "ATTESTED_BY_UNPINNED_KEY"
    try:
        pub = ed25519.Ed25519PublicKey.from_public_bytes(base64.b64decode(pinned["public_key_b64"]))
        pub.verify(base64.b64decode(str(att.get("signature_b64", ""))), _attestation_message(att))
    except (ValueError, KeyError, InvalidSignature, TypeError):
        return False, "ATTESTATION_SIGNATURE_INVALID"
    if att.get("request_sha256") != request_sha256:
        return False, "ATTESTATION_FOR_DIFFERENT_REQUEST"
    if str(att.get("request_timestamp")) != str(request_timestamp):
        return False, "ATTESTATION_TIMESTAMP_MISMATCH"
    if att.get("response_sha256") != response_sha256:
        return False, "RESPONSE_ALTERED_IN_TRANSIT"
    return True, "ATTESTED"


def _get_json(url: str, timeout: float) -> Dict[str, Any]:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.URLError:
        if not _can_use_wsl_curl_bridge(url):
            raise
        code, body = _wsl_curl_request("GET", url, timeout=timeout)
        if code != 200:
            raise urllib.error.URLError(f"HTTP {code}: {body[:200]}")
        return json.loads(body)


def fetch_node_identity(base_url: Optional[str] = None, timeout: float = 5.0) -> Dict[str, Any]:
    """The bridge's public attestation key (``GET /v1/node/identity``). Not trusted until pinned."""
    ident = _get_json(f"{resolve_base_url(base_url)}/v1/node/identity", timeout)
    raw = base64.b64decode(ident.get("public_key_b64", ""))
    if len(raw) != 32 or hashlib.sha256(raw).hexdigest() != ident.get("fingerprint_sha256"):
        raise ValueError("bridge identity is malformed: public key and fingerprint disagree")
    return ident


def load_pinned_node_identity(key_dir: Optional[pathlib.Path] = None) -> Optional[Dict[str, Any]]:
    path = (pathlib.Path(key_dir) if key_dir else default_osiris_dir()) / PINNED_IDENTITY_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pin_node_identity(
    identity: Dict[str, Any],
    *,
    expected_fingerprint: str,
    key_dir: Optional[pathlib.Path] = None,
) -> Dict[str, Any]:
    """Pin Alpha's attestation key on this machine. The operator supplies ``expected_fingerprint``
    (read from the phone's screen or over USB); a different key is refused, and an existing pin
    is never replaced by a different one."""
    fp = str(identity.get("fingerprint_sha256", ""))
    raw = base64.b64decode(identity.get("public_key_b64", ""))
    if len(raw) != 32 or hashlib.sha256(raw).hexdigest() != fp:
        raise ValueError("identity is malformed: public key and fingerprint disagree")
    if not expected_fingerprint or not fp.startswith(expected_fingerprint.lower()) or len(expected_fingerprint) < 16:
        raise ValueError("the bridge's key does not match the fingerprint you expected (give at least 16 hex characters)")
    base = pathlib.Path(key_dir) if key_dir else default_osiris_dir()
    base.mkdir(parents=True, exist_ok=True)
    path = base / PINNED_IDENTITY_FILE
    existing = load_pinned_node_identity(base)
    if existing is not None:
        if existing.get("fingerprint_sha256") != fp:
            raise ValueError(f"a different Alpha key is already pinned in {path}; remove it yourself to re-pin")
        return existing
    record = {"node_id": identity.get("node_id"), "key_type": "Ed25519",
              "public_key_b64": identity["public_key_b64"], "fingerprint_sha256": fp,
              "pinned_at": int(time.time())}
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    return record


def _android_quote(s: str) -> str:
    """org.json JSONObject.quote as on Android: escapes quote, backslash and '/', short forms for
    \\b \\t \\n \\f \\r, \\uXXXX for other control characters; non-ASCII stays as is."""
    out = ['"']
    for ch in s:
        if ch in ('"', "\\", "/"):
            out.append("\\" + ch)
        elif ch == "\b":
            out.append("\\b")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\f":
            out.append("\\f")
        elif ch == "\r":
            out.append("\\r")
        elif ord(ch) < 0x20:
            out.append("\\u%04x" % ord(ch))
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def bridge_canonical_json(value: Any) -> str:
    """The bridge ledger's canonical JSON (LedgerChain.canonicalJson), reimplemented independently."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value == int(value) else repr(value)
    if isinstance(value, str):
        return _android_quote(value)
    if isinstance(value, dict):
        return "{" + ",".join(f"{_android_quote(k)}:{bridge_canonical_json(value[k])}" for k in sorted(value)) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(bridge_canonical_json(v) for v in value) + "]"
    return _android_quote(str(value))


GENESIS_HASH = "0" * 64


def verify_ledger_lines(lines: List[str], *, prev: str = GENESIS_HASH) -> Dict[str, Any]:
    """Recompute the bridge ledger's hash chain from raw lines. Independent of the bridge's own check."""
    rows = []
    for idx, line in enumerate(lines):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return {"valid": False, "rows": rows, "head_hash": prev, "problem": f"row {idx}: malformed JSON"}
        if obj.get("prev") != prev:
            return {"valid": False, "rows": rows, "head_hash": prev, "problem": f"row {idx}: prev-hash mismatch"}
        body = {k: v for k, v in obj.items() if k not in ("prev", "hash")}
        expected = hashlib.sha256((prev + bridge_canonical_json(body)).encode("utf-8")).hexdigest()
        if obj.get("hash") != expected:
            return {"valid": False, "rows": rows, "head_hash": prev, "problem": f"row {idx}: hash mismatch"}
        rows.append(obj)
        prev = expected
    return {"valid": True, "rows": rows, "head_hash": prev, "problem": None}


def fetch_ledger_rows(base_url: Optional[str] = None, *, timeout: float = 10.0, max_rows: int = 100_000) -> Dict[str, Any]:
    """Read the whole bridge ledger (``GET /v1/ledger/rows``, 500 rows per page) and recompute it here."""
    lines: List[str] = []
    reported: Dict[str, Any] = {}
    while len(lines) < max_rows:
        page = _get_json(f"{resolve_base_url(base_url)}/v1/ledger/rows?since={len(lines)}&limit=500", timeout)
        reported = page
        batch = [str(r) for r in page.get("rows", [])]
        lines.extend(batch)
        if not batch or len(lines) >= int(page.get("total_rows", 0)):
            break
    result = verify_ledger_lines(lines)
    result["reported_head_hash"] = reported.get("head_hash")
    result["reported_total_rows"] = reported.get("total_rows")
    result["head_matches_bridge"] = result["valid"] and result["head_hash"] == reported.get("head_hash")
    return result


def ledger_rows_for_request(ledger: Dict[str, Any], request_sha256: str) -> List[Dict[str, Any]]:
    return [r for r in ledger.get("rows", []) if r.get("payload_sha256") == request_sha256]


def send_osiris_intent(
    prompt: str,
    *,
    base_url: Optional[str] = None,
    key_dir: Optional[pathlib.Path] = None,
    model: str = "litert-gemma-3b",
    stream_stdout: bool = True,
    timeout: float = 30.0,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Send one signed prompt. Each call carries a fresh ``osiris_request_id`` in the signed body,
    so a deliberate repeat is a new request and a byte-identical resend is a replay the bridge
    refuses. The result reports whether the response was attested by the pinned Alpha key."""
    endpoint = f"{resolve_base_url(base_url)}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are the OSIRIS Edge Substrate Agent."},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "osiris_request_id": request_id or uuid.uuid4().hex,
    }

    data_bytes = json.dumps(payload).encode("utf-8")
    signature, timestamp = sign_payload(data_bytes, key_dir=key_dir)
    request_sha256 = hashlib.sha256(data_bytes).hexdigest()

    hdrs = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "X-Signature-Ed25519": signature,
        "X-Timestamp": timestamp,
    }
    req = urllib.request.Request(endpoint, data=data_bytes, headers=hdrs, method="POST")

    if stream_stdout:
        print(f"[*] Dispatching Ed25519-signed intent to Android Bridge at {endpoint}...")

    def finish(parsed: Dict[str, Any]) -> Dict[str, Any]:
        if stream_stdout:
            for chunk in parsed["chunks"]:
                piece = chunk.get("status", "") if isinstance(chunk, dict) else ""
                if piece:
                    sys.stdout.write(str(piece))
            sys.stdout.flush()
        ok_att, why_att = verify_attestation(
            parsed["attestation"], load_pinned_node_identity(key_dir),
            request_sha256=request_sha256, response_sha256=parsed["response_sha256"],
            request_timestamp=timestamp)
        if stream_stdout:
            print(f"\n[+] Signed IPC stream completed. Response origin: {why_att}.")
        return {"ok": True, "chunks": parsed["chunks"], "request_id": payload["osiris_request_id"],
                "request_sha256": request_sha256, "request_timestamp": timestamp,
                "response_sha256": parsed["response_sha256"], "attestation": parsed["attestation"],
                "attestation_error": parsed["attestation_error"],
                "response_attested": ok_att, "attestation_check": why_att}

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return finish(_parse_sse(response))
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        if stream_stdout:
            print(f"\n[-] IPC Bridge Rejected Request (HTTP {exc.code}): {err_body}")
        return {"ok": False, "http_code": exc.code, "error": err_body}
    except urllib.error.URLError as exc:
        if _can_use_wsl_curl_bridge(endpoint):
            try:
                code, resp_body = _wsl_curl_request("POST", endpoint, headers=hdrs, data=data_bytes, timeout=timeout)
                if code != 200:
                    if stream_stdout:
                        print(f"\n[-] IPC Bridge Rejected Request (HTTP {code}): {resp_body}")
                    return {"ok": False, "http_code": code, "error": resp_body}
                return finish(_parse_sse(resp_body.splitlines()))
            except Exception:
                pass
        if stream_stdout:
            print(f"\n[-] IPC Bridge Connection Failed: {exc.reason}. Ensure BridgeCoreService is running.")
        return {"ok": False, "error": str(exc.reason)}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python3 osiris_ipc_client.py "Your task intent here"')
        sys.exit(1)
    res = send_osiris_intent(" ".join(sys.argv[1:]))
    sys.exit(0 if res.get("ok") else 1)
