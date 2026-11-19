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


def send_osiris_intent(
    prompt: str,
    *,
    base_url: Optional[str] = None,
    key_dir: Optional[pathlib.Path] = None,
    model: str = "litert-gemma-3b",
    stream_stdout: bool = True,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    endpoint = f"{resolve_base_url(base_url)}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are the OSIRIS Edge Substrate Agent."},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
    }

    data_bytes = json.dumps(payload).encode("utf-8")
    signature, timestamp = sign_payload(data_bytes, key_dir=key_dir)

    hdrs = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "X-Signature-Ed25519": signature,
        "X-Timestamp": timestamp,
    }
    req = urllib.request.Request(endpoint, data=data_bytes, headers=hdrs, method="POST")

    if stream_stdout:
        print(f"[*] Dispatching Ed25519-signed intent to Android Bridge at {endpoint}...")

    chunks = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            for raw_line in response:
                decoded = raw_line.decode("utf-8", errors="replace").strip()
                if not decoded.startswith("data: "):
                    continue
                content = decoded[6:]
                if content == "[DONE]":
                    break
                try:
                    chunk = json.loads(content)
                    chunks.append(chunk)
                    text_piece = chunk.get("status", "")
                    if stream_stdout and text_piece:
                        sys.stdout.write(str(text_piece))
                        sys.stdout.flush()
                except json.JSONDecodeError:
                    chunks.append({"raw": content})
                    if stream_stdout:
                        sys.stdout.write(content)
                        sys.stdout.flush()
        if stream_stdout:
            print("\n[+] Signed IPC Stream completed successfully.")
        return {"ok": True, "chunks": chunks}
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
                for line in resp_body.splitlines():
                    decoded = line.strip()
                    if not decoded.startswith("data: "):
                        continue
                    content = decoded[6:]
                    if content == "[DONE]":
                        break
                    try:
                        chunk = json.loads(content)
                        chunks.append(chunk)
                        text_piece = chunk.get("status", "")
                        if stream_stdout and text_piece:
                            sys.stdout.write(str(text_piece))
                            sys.stdout.flush()
                    except json.JSONDecodeError:
                        chunks.append({"raw": content})
                if stream_stdout:
                    print("\n[+] Signed IPC Stream completed successfully.")
                return {"ok": True, "chunks": chunks}
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
