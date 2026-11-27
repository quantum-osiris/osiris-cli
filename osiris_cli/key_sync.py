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
import re
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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


_REMOTE_PATH = re.compile(r"^[A-Za-z0-9_./~-]+$")
Runner = Callable[..., Any]


def assert_safe_remote_path(path: str) -> str:
    """Reject shell metacharacters and parent traversal in a remote path."""
    if not isinstance(path, str) or not _REMOTE_PATH.fullmatch(path) or ".." in path.split("/"):
        raise ValueError(f"unsafe remote path {path!r}")
    return path


def adb_alpha_probe_commands(adb_bin: str, local_helper: pathlib.Path, remote_dir: str) -> List[List[str]]:
    """Push the versioned helper and run it by path. No ``python -c`` and no shell string."""
    from osiris_cli.substrate_guard import probe_alpha_state

    remote_dir = assert_safe_remote_path(remote_dir).rstrip("/")
    remote = assert_safe_remote_path(remote_dir + "/alpha_state_v1.py")
    plan = probe_alpha_state(interpreter="python3", helper=local_helper)
    if plan["shell"] or any(";" in part or "\n" in part or part == "-c" for part in plan["argv"]):
        raise ValueError("refusing an inline alpha probe")
    return [
        [adb_bin, "push", str(local_helper), remote],
        [adb_bin, "exec-out", "python3", remote],
    ]


def ssh_alpha_probe_commands(
    host: str, port: int, user: Optional[str], local_helper: pathlib.Path, remote_dir: str,
) -> List[List[str]]:
    from osiris_cli.substrate_guard import probe_alpha_state

    remote_dir = assert_safe_remote_path(remote_dir).rstrip("/")
    remote = assert_safe_remote_path(remote_dir + "/alpha_state_v1.py")
    plan = probe_alpha_state(interpreter="python3", helper=local_helper)
    if plan["shell"] or any(part == "-c" for part in plan["argv"]):
        raise ValueError("refusing an inline alpha probe")
    target = f"{user}@{host}" if user else host
    destination = f"{target}:{remote}"
    return [
        ["scp", "-o", "BatchMode=yes", "-P", str(port), str(local_helper), destination],
        ["ssh", "-o", "BatchMode=yes", "-p", str(port), target, "python3", remote],
    ]


def adb_ledger_pull_command(adb_bin: str, remote_ledger: str, staging: pathlib.Path, live_ledger: pathlib.Path) -> List[str]:
    remote_ledger = assert_safe_remote_path(remote_ledger)
    if staging.resolve() == live_ledger.resolve():
        raise ValueError("refusing to pull a remote ledger onto the live ledger path")
    return [adb_bin, "pull", remote_ledger, str(staging)]


def ssh_ledger_pull_command(
    host: str, port: int, user: Optional[str], remote_ledger: str, staging: pathlib.Path, live_ledger: pathlib.Path,
) -> List[str]:
    remote_ledger = assert_safe_remote_path(remote_ledger)
    if staging.resolve() == live_ledger.resolve():
        raise ValueError("refusing to pull a remote ledger onto the live ledger path")
    target = f"{user}@{host}" if user else host
    return ["scp", "-o", "BatchMode=yes", "-P", str(port), f"{target}:{remote_ledger}", str(staging)]


def accept_remote_evidence(local_ledger: pathlib.Path, payload: bytes) -> str:
    """Parse a pulled ledger and accept it only through the ancestry check."""
    from osiris_cli.substrate_guard import apply_return_chain

    text = payload.decode("utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return apply_return_chain(local_ledger, rows)


def _run_checked(runner: Runner, argv: Sequence[str]) -> Any:
    joined = " ".join(argv)
    if "python -c" in joined or "python3 -c" in joined or ";" in joined:
        raise ValueError("refusing a shell-quoted remote command")
    return runner(list(argv), capture_output=True, text=True, timeout=10, check=False)


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

    def _pull_remote_ledger(
        self,
        pull_argv: List[str],
        runner: Runner,
    ) -> str:
        from osiris_cli.substrate_guard import SubstrateGuardError

        staging = self.osiris_home / "incoming" / "exchanges.jsonl"
        staging.parent.mkdir(parents=True, exist_ok=True)
        if staging.resolve() == self.ledger_path.resolve():
            return "rejected:pull destination is the live ledger"
        result = _run_checked(runner, pull_argv)
        if getattr(result, "returncode", 1) != 0 or not staging.is_file():
            return "not_pulled"
        try:
            return accept_remote_evidence(self.ledger_path, staging.read_bytes())
        except (OSError, UnicodeError, json.JSONDecodeError, SubstrateGuardError, ValueError) as exc:
            reason = getattr(exc, "reason", str(exc))
            return f"rejected:{reason}"

    def sync_via_adb(
        self,
        node_id: str = "node_beta_wsl_workstation",
        runner: Optional[Runner] = None,
        remote_dir: str = ".osiris",
    ) -> Dict[str, Any]:
        pub_bytes, pub_hash, ledger_hash = self.ensure_and_verify_local_assets()
        self.register_trusted_key(node_id, pub_bytes)

        adb_bin = find_adb_binary()
        adb_pushed = False
        pulled_alpha_fp: Optional[str] = None

        transport = runner or subprocess.run
        alpha_probe: Dict[str, Any] = {"status": "not_run"}
        evidence_chain = "not_pulled"
        if adb_bin:
            transport([adb_bin, "forward", "tcp:8080", "tcp:8080"], capture_output=True, check=False)
            transport([adb_bin, "forward", "tcp:8022", "tcp:8022"], capture_output=True, check=False)
            transport([adb_bin, "shell", "mkdir", "-p", "/sdcard/crsm"], capture_output=True, check=False)

            # Convert WSL path to Windows path if running under WSL interop
            is_wsl = os.path.exists("/proc/version") and "microsoft" in pathlib.Path("/proc/version").read_text(errors="ignore").lower()
            push_src = str(self.pub_key_path)
            if (adb_bin.endswith(".exe") or is_wsl) and shutil.which("wslpath"):
                wp = subprocess.run(["wslpath", "-w", push_src], capture_output=True, text=True, check=False)
                if wp.returncode == 0 and wp.stdout.strip():
                    push_src = wp.stdout.strip()

            push_res = transport(
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
            pull_res = transport(
                [adb_bin, "pull", "/sdcard/crsm/node_alpha_ed25519.pub", pull_dst],
                capture_output=True,
                text=True,
                check=False,
            )
            if pull_res.returncode == 0 and alpha_tmp.exists() and len(alpha_tmp.read_bytes()) == 32:
                reg = self.register_trusted_key("node_alpha_s26_ultra", alpha_tmp.read_bytes())
                pulled_alpha_fp = reg["fingerprint_sha256"]

            helper = pathlib.Path(__file__).resolve().parent / "remote_helpers" / "alpha_state_v1.py"
            push_cmd, probe_cmd = adb_alpha_probe_commands(adb_bin, helper, remote_dir)
            push_helper = _run_checked(transport, push_cmd)
            probe_res = _run_checked(transport, probe_cmd) if getattr(push_helper, "returncode", 1) == 0 else None
            alpha_probe = {
                "status": "ok" if probe_res is not None and probe_res.returncode == 0 else "failed",
                "argv": probe_cmd,
            }
            staging = self.osiris_home / "incoming" / "exchanges.jsonl"
            pull_cmd = adb_ledger_pull_command(
                adb_bin, remote_dir.rstrip("/") + "/living/exchanges.jsonl", staging, self.ledger_path,
            )
            evidence_chain = self._pull_remote_ledger(pull_cmd, transport)

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
            "alpha_probe": alpha_probe,
            "evidence_chain": evidence_chain,
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
        runner: Optional[Runner] = None,
        remote_dir: str = ".osiris",
    ) -> Dict[str, Any]:
        _, pub_hash, ledger_hash = self.ensure_and_verify_local_assets()
        target = f"{user}@{host}:{dest_dir}" if user else f"{host}:{dest_dir}"
        scp_cmd = ["scp", "-o", "BatchMode=yes", "-P", str(port), str(self.pub_key_path), target]
        transport = runner or subprocess.run
        res = transport(scp_cmd, capture_output=True, text=True, timeout=10, check=False)
        ok = res.returncode == 0
        alpha_probe: Dict[str, Any] = {"status": "not_run"}
        evidence_chain = "not_pulled"
        if ok:
            helper = pathlib.Path(__file__).resolve().parent / "remote_helpers" / "alpha_state_v1.py"
            commands = ssh_alpha_probe_commands(host, port, user, helper, remote_dir)
            for command in commands[:-1]:
                _run_checked(transport, command)
            probe_res = _run_checked(transport, commands[-1])
            alpha_probe = {"status": "ok" if probe_res.returncode == 0 else "failed", "argv": commands[-1]}
            staging = self.osiris_home / "incoming" / "exchanges.jsonl"
            pull_cmd = ssh_ledger_pull_command(
                host, port, user, remote_dir.rstrip("/") + "/living/exchanges.jsonl", staging, self.ledger_path,
            )
            evidence_chain = self._pull_remote_ledger(pull_cmd, transport)
        receipt = {
            "sync_timestamp": datetime.now(timezone.utc).isoformat(),
            "transport": "SCP_SSH",
            "target": target,
            "public_key_sha256": pub_hash,
            "ledger_sha256": ledger_hash,
            "alpha_probe": alpha_probe,
            "evidence_chain": evidence_chain,
            "status": "SYNCHRONIZED_AND_VERIFIED" if ok else "SSH_TRANSPORT_FAILED",
            "error": None if ok else (res.stderr or "").strip(),
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
