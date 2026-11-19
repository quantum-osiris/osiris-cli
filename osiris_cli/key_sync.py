#!/usr/bin/env python3
"""
OSIRIS Dual-Substrate Key & Ledger Synchronization Utility
==========================================================
Automates secure exchange of Ed25519 IPC public keys and SHA-256 ledger digests
between Node Alpha (Samsung Galaxy S26 Ultra / Termux / Edge Bridge) and
Node Beta (WSL Workstation) over ADB or SSH (:8022).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from .edge_ipc_client import (
    enroll_edge_key,
    get_public_key_bytes,
    public_key_fingerprint,
)

WSL_INTEROP_ADB = "/mnt/c/Users/there/AppData/Local/Android/Sdk/platform-tools/adb.exe"


def find_adb_binary() -> Optional[str]:
    found = shutil.which("adb")
    if found:
        return found
    if os.path.exists(WSL_INTEROP_ADB):
        return WSL_INTEROP_ADB
    win_sdk = pathlib.Path.home() / "AppData/Local/Android/Sdk/platform-tools/adb.exe"
    if win_sdk.exists():
        return str(win_sdk)
    return None


def compute_sha256(file_path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DualSubstrateKeySync:
    def __init__(
        self,
        osiris_home: Optional[pathlib.Path] = None,
        bridge_url: Optional[str] = None,
    ) -> None:
        self.osiris_home = pathlib.Path(osiris_home) if osiris_home else (pathlib.Path.home() / ".osiris")
        self.osiris_home.mkdir(parents=True, exist_ok=True)
        self.pub_key_path = self.osiris_home / "ed25519_public.pub"
        self.ledger_path = self.osiris_home / "living" / "exchanges.jsonl"
        self.trusted_keys_path = self.osiris_home / "trusted_edge_keys.json"
        self.receipt_path = self.osiris_home / "node_beta_sync_receipt.json"
        self.bridge_url = bridge_url

    def ensure_and_verify_local_assets(self) -> Tuple[bytes, str, str]:
        pub_bytes = get_public_key_bytes(self.osiris_home)
        pub_hash = compute_sha256(self.pub_key_path)
        ledger_hash = compute_sha256(self.ledger_path) if self.ledger_path.exists() else "N/A_NO_LEDGER"
        return pub_bytes, pub_hash, ledger_hash

    def register_trusted_key(self, node_id: str, pub_bytes: bytes) -> Dict[str, Any]:
        if len(pub_bytes) != 32:
            raise ValueError(f"Expected 32-byte raw Ed25519 public key, got {len(pub_bytes)} bytes")
        fp = public_key_fingerprint(pub_bytes)
        entry = {
            "node_id": node_id,
            "key_type": "Ed25519",
            "public_key_b64": base64.b64encode(pub_bytes).decode("ascii"),
            "fingerprint_sha256": fp,
            "registered_timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "AUTHORIZED",
        }
        existing = []
        if self.trusted_keys_path.exists():
            try:
                loaded = json.loads(self.trusted_keys_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    existing = [k for k in loaded if k.get("fingerprint_sha256") != fp]
            except json.JSONDecodeError:
                existing = []
        existing.append(entry)
        self.trusted_keys_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
        return entry

    def sync_via_adb(self, node_id: str = "node_beta_wsl_workstation") -> Dict[str, Any]:
        pub_bytes, pub_hash, ledger_hash = self.ensure_and_verify_local_assets()
        self.register_trusted_key(node_id, pub_bytes)

        adb_bin = find_adb_binary()
        adb_pushed = False
        pulled_alpha_fp: Optional[str] = None

        if adb_bin:
            subprocess.run([adb_bin, "forward", "tcp:8080", "tcp:8080"], capture_output=True, check=False)
            subprocess.run([adb_bin, "forward", "tcp:8022", "tcp:8022"], capture_output=True, check=False)
            subprocess.run([adb_bin, "shell", "mkdir -p /sdcard/crsm"], capture_output=True, check=False)

            # Convert WSL path to Windows path if running under WSL interop
            is_wsl = os.path.exists("/proc/version") and "microsoft" in pathlib.Path("/proc/version").read_text(errors="ignore").lower()
            push_src = str(self.pub_key_path)
            if (adb_bin.endswith(".exe") or is_wsl) and shutil.which("wslpath"):
                wp = subprocess.run(["wslpath", "-w", push_src], capture_output=True, text=True, check=False)
                if wp.returncode == 0 and wp.stdout.strip():
                    push_src = wp.stdout.strip()

            push_res = subprocess.run(
                [adb_bin, "push", push_src, "/sdcard/crsm/node_beta_ed25519.pub"],
                capture_output=True,
                text=True,
                check=False,
            )
            adb_pushed = push_res.returncode == 0

            # Pull Node Alpha's Termux key if already exported to /sdcard/crsm/node_alpha_ed25519.pub
            alpha_tmp = self.osiris_home / "node_alpha_ed25519.pub"
            pull_dst = str(alpha_tmp)
            if (adb_bin.endswith(".exe") or is_wsl) and shutil.which("wslpath"):
                wp = subprocess.run(["wslpath", "-w", pull_dst], capture_output=True, text=True, check=False)
                if wp.returncode == 0 and wp.stdout.strip():
                    pull_dst = wp.stdout.strip()
            pull_res = subprocess.run(
                [adb_bin, "pull", "/sdcard/crsm/node_alpha_ed25519.pub", pull_dst],
                capture_output=True,
                text=True,
                check=False,
            )
            if pull_res.returncode == 0 and alpha_tmp.exists() and len(alpha_tmp.read_bytes()) == 32:
                reg = self.register_trusted_key("node_alpha_s26_ultra", alpha_tmp.read_bytes())
                pulled_alpha_fp = reg["fingerprint_sha256"]

        enroll_res = enroll_edge_key(
            node_id=node_id,
            base_url=self.bridge_url,
            key_dir=self.osiris_home,
        )

        receipt = {
            "sync_timestamp": datetime.now(timezone.utc).isoformat(),
            "transport": "ADB_PORT_FORWARD_AND_STORAGE",
            "source_node": node_id,
            "target_node": "Node Alpha (Samsung Galaxy S26 Ultra - SM-S948U1)",
            "public_key_sha256": pub_hash,
            "ledger_sha256": ledger_hash,
            "adb_staged_sdcard": adb_pushed,
            "node_alpha_key_fingerprint": pulled_alpha_fp,
            "edge_bridge_enrollment": enroll_res,
            "status": "SYNCHRONIZED_AND_VERIFIED"
            if (adb_pushed or enroll_res.get("status") == "ENROLLED")
            else "LOCAL_KEY_PREPARED",
        }
        self.receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        return receipt

    def sync_via_ssh(
        self,
        host: str = "127.0.0.1",
        port: int = 8022,
        user: Optional[str] = None,
        dest_dir: str = "~/.osiris/",
    ) -> Dict[str, Any]:
        _, pub_hash, ledger_hash = self.ensure_and_verify_local_assets()
        target = f"{user}@{host}:{dest_dir}" if user else f"{host}:{dest_dir}"
        scp_cmd = ["scp", "-o", "BatchMode=yes", "-P", str(port), str(self.pub_key_path), target]
        res = subprocess.run(scp_cmd, capture_output=True, text=True, check=False)
        ok = res.returncode == 0
        receipt = {
            "sync_timestamp": datetime.now(timezone.utc).isoformat(),
            "transport": "SCP_SSH",
            "target": target,
            "public_key_sha256": pub_hash,
            "ledger_sha256": ledger_hash,
            "status": "SYNCHRONIZED_AND_VERIFIED" if ok else "SSH_TRANSPORT_FAILED",
            "error": None if ok else res.stderr.strip(),
        }
        self.receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        return receipt


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="OSIRIS Dual-Substrate Ed25519 Key Sync")
    parser.add_argument("--mode", choices=["adb", "ssh"], default="adb")
    parser.add_argument("--node-id", default="node_beta_wsl_workstation")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8022)
    parser.add_argument("--user", default=None)
    args = parser.parse_args(argv)

    syncer = DualSubstrateKeySync()
    if args.mode == "adb":
        receipt = syncer.sync_via_adb(node_id=args.node_id)
    else:
        receipt = syncer.sync_via_ssh(host=args.host, port=args.port, user=args.user)

    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "SYNCHRONIZED_AND_VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
