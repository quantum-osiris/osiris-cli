"""
OSIRIS Edge Bridge Command Router (`osiris bridge status | readyz | enroll | intent`).
"""
from __future__ import annotations

import json
import pathlib
import socket
import sys
from typing import List, Optional

from .edge_ipc_client import (
    check_bridge_health,
    check_ledger_readiness,
    enroll_edge_key,
    send_osiris_intent,
)


class OSIRISCommandRouter:
    def __init__(
        self,
        args: List[str],
        *,
        base_url: Optional[str] = None,
        key_dir: Optional[pathlib.Path] = None,
    ) -> None:
        self.args = list(args)
        self.base_url = base_url
        self.key_dir = key_dir

    def route(self) -> int:
        if not self.args:
            print("OSIRIS Dual-Engine Substrate")
            print("Usage: osiris bridge <status|readyz|enroll|intent> [arguments]")
            return 1

        command = self.args[0].lower()

        if command == "intent":
            if len(self.args) < 2:
                print("[-] Error: Missing prompt intent string.")
                print('    Usage: osiris intent "Your architectural intent here"')
                return 1
            prompt_intent = " ".join(self.args[1:])
            res = send_osiris_intent(prompt_intent, base_url=self.base_url, key_dir=self.key_dir)
            return 0 if res.get("ok") else 1

        if command == "bridge":
            if len(self.args) < 2:
                print("[-] Error: Missing bridge subcommand.")
                print("    Usage: osiris bridge status")
                print("    Usage: osiris bridge readyz")
                print("    Usage: osiris bridge enroll [node_id]")
                print('    Usage: osiris bridge intent "Your task here"')
                return 1

            subcommand = self.args[1].lower()
            if subcommand == "status":
                print("[*] Polling local Android LiteRT-LM bridge health...")
                result = check_bridge_health(base_url=self.base_url)
                if result["status"] == "ONLINE":
                    print("[+] Bridge Status: ONLINE (Hardware Accelerated)")
                    print(f"    Details: {json.dumps(result['details'], indent=2)}")
                    return 0
                print("[-] Bridge Status: OFFLINE or Unresponsive")
                print(f"    Reason: {result.get('error', 'unknown')}")
                return 1

            if subcommand == "readyz":
                print("[*] Verifying SHA-256 hash-chained evidence ledger integrity on Android bridge...")
                result = check_ledger_readiness(base_url=self.base_url)
                if result["status"] == "READY" and result.get("chain_valid"):
                    print("[+] Ledger Status: VERIFIED (Chain Intact)")
                    print(f"    Details: {json.dumps(result['details'], indent=2)}")
                    return 0
                print("[-] Ledger Status: FAILED / NOT READY")
                print(f"    Reason: {result.get('error', 'Chain invalid')}")
                return 1

            if subcommand == "enroll":
                default_node = f"node_{socket.gethostname().lower()}"
                node_id = self.args[2] if len(self.args) >= 3 else default_node
                print(f"[*] Enrolling local Ed25519 public key ({node_id}) with Android bridge...")
                result = enroll_edge_key(node_id=node_id, base_url=self.base_url, key_dir=self.key_dir)
                if result.get("status") == "ENROLLED":
                    print(f"[+] Enrolled Key Fingerprint: {result['fingerprint_sha256']}")
                    print(f"    Ledger Anchor Hash:       {result['ledger_hash']}")
                    return 0
                print(f"[-] Enrollment failed: {result}")
                return 1

            if subcommand in ("intent", "run"):
                if len(self.args) < 3:
                    print('[-] Error: Missing prompt. Usage: osiris bridge intent "Your task here"')
                    return 1
                prompt_intent = " ".join(self.args[2:])
                res = send_osiris_intent(prompt_intent, base_url=self.base_url, key_dir=self.key_dir)
                return 0 if res.get("ok") else 1

            # Allow shorthand: `osiris bridge "Your architectural intent"`
            prompt_intent = " ".join(self.args[1:])
            res = send_osiris_intent(prompt_intent, base_url=self.base_url, key_dir=self.key_dir)
            return 0 if res.get("ok") else 1

        print(f"[-] Unknown command: {command}")
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    router = OSIRISCommandRouter(sys.argv[1:] if argv is None else argv)
    return router.route()
