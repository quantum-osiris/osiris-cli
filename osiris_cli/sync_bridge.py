"""
OSIRIS Dual-Node Mobile Dev Synchronization Subsystem
=====================================================
Runbook and status inspector for synchronizing OSIRIS repositories between:
  - Node Alpha: Mobile Edge (Samsung Galaxy S26 Ultra / Fold7 / Termux / PRoot debian, aarch64)
  - Node Beta:  Workstation / WSL (x86_64 / arm64)

Navigates Android UID Sandboxing, Scoped Storage noexec, and FUSE latency via:
  1. OpenSSH on port 8022 inside Termux private app sandbox.
  2. Android Wireless Debugging (pairing + adb forward tcp:8022 tcp:8022).
  3. Bluetooth PAN (Personal Area Network direct IP tethering).

Axiom: "Cross-node transfer is a data operation, not an authorization operation."
Transferring commits preserves cryptographic integrity but confers zero authority.
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
from typing import Any, Dict, Optional


class DualNodeSyncBridge:
    """Manages dev synchronization diagnostics and runbook between Node Alpha and Node Beta."""

    DEFAULT_SSH_PORT = 8022

    @classmethod
    def detect_substrate(cls) -> Dict[str, Any]:
        """Detect whether currently running on Node Alpha (edge) or Node Beta (workstation)."""
        uname = platform.uname()
        machine = uname.machine.lower()
        system = uname.system.lower()
        is_arm64 = machine in ("aarch64", "arm64")
        
        # Check for Termux or PRoot environment markers
        is_termux = "TERMUX_VERSION" in os.environ or "com.termux" in os.environ.get("PREFIX", "")
        is_proot = os.path.exists("/proc/version") and "proot" in open("/proc/version", "r", errors="ignore").read().lower()

        if is_arm64 and (is_termux or is_proot or os.path.exists("/data/data/com.termux")):
            node_id = "Node Alpha"
            node_role = "Mobile Edge Substrate (Termux / PRoot)"
        elif is_arm64:
            node_id = "Node Alpha (Edge ARM64)"
            node_role = "Edge Substrate"
        else:
            node_id = "Node Beta"
            node_role = "Workstation / WSL Host"

        return {
            "node_id": node_id,
            "node_role": node_role,
            "machine": machine,
            "system": system,
            "is_arm64": is_arm64,
            "is_termux": is_termux,
            "is_proot": is_proot,
        }

    @classmethod
    def check_ssh_listener(cls, host: str = "127.0.0.1", port: int = DEFAULT_SSH_PORT, timeout: float = 0.5) -> bool:
        """Non-privileged socket probe to test if OpenSSH is listening on port 8022."""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        try:
            res = s.connect_ex((host, port))
            return (res == 0)
        except Exception:
            return False
        finally:
            s.close()

    @classmethod
    def check_adb_status(cls) -> Dict[str, Any]:
        """Inspect local ADB availability and forwarded ports."""
        adb_bin = shutil.which("adb") or ("/mnt/c/Users/there/AppData/Local/Android/Sdk/platform-tools/adb.exe" if os.path.exists("/mnt/c/Users/there/AppData/Local/Android/Sdk/platform-tools/adb.exe") else None)
        if not adb_bin:
            return {"installed": False, "devices": [], "forwards": []}

        try:
            dev_res = subprocess.run([adb_bin, "devices"], capture_output=True, text=True, check=False)
            fwd_res = subprocess.run([adb_bin, "forward", "--list"], capture_output=True, text=True, check=False)
            devices = [line.strip() for line in (dev_res.stdout or "").splitlines()[1:] if line.strip()]
            forwards = [line.strip() for line in (fwd_res.stdout or "").splitlines() if line.strip()]
            return {
                "installed": True,
                "devices": devices,
                "forwards": forwards
            }
        except Exception as e:
            return {"installed": True, "error": str(e)}

    @classmethod
    def format_runbook(cls) -> str:
        """Generates step-by-step synchronization runbook for the user."""
        return """
\033[1;36m══════════════════════════════════════════════════════════════════════════
 OSIRIS DUAL-NODE DEV SYNCHRONIZATION RUNBOOK (BLUETOOTH PAN / WIRELESS ADB)
══════════════════════════════════════════════════════════════════════════\033[0m
\033[1;33mCore Axiom:\033[0m "Cross-node transfer is a data operation, not an authorization operation."
Transferring git commits preserves cryptographic digests without granting execution authority.

\033[1;37m[CHANNEL A: Android Wireless Debugging & ADB Port Forward]\033[0m
  1. On Node Alpha (Phone):
     • Settings -> Developer Options -> Wireless Debugging -> Enable.
     • Tap "Pair device with pairing code" -> Note IP:Port and 6-digit Code.
  2. On Node Beta (Workstation):
     • adb pair <IP>:<Pairing_Port>  (Enter pairing code)
     • adb connect <IP>:<Connect_Port>
     • adb forward tcp:8022 tcp:8022
  3. In Termux (Node Alpha):
     • pkg install -y openssh rsync
     • sshd -p 8022
  4. From Node Beta (Push or Pull):
     • Push to phone: rsync -avz --delete -e 'ssh -p 8022' ~/repos/osiris-cli/ localhost:~/src/repos/osiris-cli/
     • Pull to workstation: rsync -avz --delete -e 'ssh -p 8022' localhost:~/src/repos/osiris-cli/ ~/repos/osiris-cli/

\033[1;37m[CHANNEL B: Bluetooth PAN (Direct Mobile Tethering)]\033[0m
  1. Pair Fold7 and Workstation via Bluetooth.
  2. Enable "Bluetooth Tethering" on Fold7 (creates local network interface, e.g. bnep0).
  3. Workstation connects via Bluetooth Network Connection (obtains local subnet IP, e.g. 192.168.44.X).
  4. Node Beta executes rsync directly over the Bluetooth PAN IP to port 8022:
     • rsync -avz -e 'ssh -p 8022' ./ <Phone_PAN_IP>:~/src/repos/osiris-cli/

\033[1;37m[STORAGE CONSTRAINTS]\033[0m
  • Never sync git repositories into /sdcard/ (Scoped Storage mounts noexec and has FUSE latency).
  • Keep active repositories in Termux private app sandbox (~/src/ or PRoot /root/repos).
"""

    @classmethod
    def format_status_card(cls) -> str:
        """Format a live status card of the sync infrastructure."""
        sub = cls.detect_substrate()
        ssh_active = cls.check_ssh_listener()
        adb = cls.check_adb_status()

        lines = []
        lines.append("\033[1;36m[OSIRIS::DUAL-NODE SYNC & DEVICE BRIDGE]\033[0m")
        lines.append(f"  ├─ Detected Substrate : \033[1;32m{sub['node_id']}\033[0m ({sub['node_role']})")
        lines.append(f"  ├─ Architecture       : {sub['machine']} (ARM64 SIMD: {sub['is_arm64']})")
        ssh_str = "\033[1;32mLISTENING (Ready for rsync)\033[0m" if ssh_active else "\033[1;33mOFFLINE (Start with 'sshd -p 8022')\033[0m"
        lines.append(f"  ├─ SSH Daemon (:8022) : {ssh_str}")

        if adb.get("installed"):
            dev_cnt = len(adb.get("devices", []))
            fwd_cnt = len(adb.get("forwards", []))
            lines.append(f"  ├─ Local ADB Daemon   : ACTIVE ({dev_cnt} devices, {fwd_cnt} forwarded ports)")
        else:
            lines.append("  ├─ Local ADB Daemon   : NOT DETECTED (Normal inside PRoot container)")

        lines.append("  └─ Sync Runbook       : Type \033[1;37m/sync plan\033[0m for Bluetooth & Wireless ADB procedures")
        return "\n".join(lines)
