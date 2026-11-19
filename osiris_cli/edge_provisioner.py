"""Governed Multi-Substrate Edge Node Discovery, Provisioning & Verification Primitives.

Provides deterministic, non-interactive CLI & programmatic primitives (`osiris provision ...` /
`/provision`) across heterogeneous OSIRIS edge nodes:
  * `alpha_android_termux`   -- ARM64 Android / Termux (Snapdragon Oryon Prime cores, ADB, SSH :8022, Bridge :8080)
  * `beta_wsl_workstation`   -- x86_64/ARM64 Linux / WSL2 compute orchestrator
  * `gamma_win10_mobile_ppkg`-- Windows 10 Mobile / IoT Runtime Provisioning (`.ppkg` WIM + OMA-DM `.provxml`)
  * `delta_posix_ssh_edge`   -- Generic POSIX / Container SSH edge node

Safety & Governance Invariants:
  1. Cross-node transfer is a data operation, never an implicit authorization operation.
  2. Hardware USB mode state machine distinguishes `MTP_ONLINE` (`VID_03F0&PID_0155`) from
     `UEFI_FFU_FLASH_MODE` (`VID_045E&PID_062A` "Lightning Bolt + Gear"), refusing file pushes
     in FFU mode instead of hanging or invoking modal GUI COM dialogs (`InvokeVerb("delete")` is banned).
  3. Every staged bundle embeds a `dnalang` v0.4.0 Claims-Firewall-verified `.dna` organism,
     lowered `RuleSet` + `RegulatoryGraph` IR, a `dnalang.Ledger` genesis receipt, LF-only text files,
     and a complete SHA-256 `manifest.json` verifiable offline via `verify_provisioning_bundle()`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import uuid
import xml.sax.saxutils as saxutils
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from dnalang.ledger import Ledger

from osiris_cli.dnalang.edge_organism import (
    CompiledEdgeOrganism,
    EdgeOrganismRuntime,
    EdgeTelemetry,
    build_edge_organism_source,
    compile_governed_edge_organism,
)

WSL_POWERSHELL = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"


@dataclass(frozen=True)
class EdgeProfileSpec:
    profile_id: str
    short_alias: str
    node_tier: str
    title: str
    os_family: str
    default_horizon_s: int
    transports: Tuple[str, ...]


EDGE_PROFILES: Dict[str, EdgeProfileSpec] = {
    "alpha_android_termux": EdgeProfileSpec(
        profile_id="alpha_android_termux",
        short_alias="alpha",
        node_tier="NODE_ALPHA",
        title="Android / Termux Mobile Edge (Snapdragon ARM64, SSH :8022, Bridge :8080)",
        os_family="android_termux",
        default_horizon_s=300,
        transports=("adb", "ssh_8022", "edge_bridge_8080", "bluetooth_pan"),
    ),
    "beta_wsl_workstation": EdgeProfileSpec(
        profile_id="beta_wsl_workstation",
        short_alias="beta",
        node_tier="NODE_BETA",
        title="Linux / WSL2 Compute & Control Workstation",
        os_family="linux_wsl",
        default_horizon_s=1800,
        transports=("local_bus", "ollama_11434", "wsl_interop"),
    ),
    "gamma_win10_mobile_ppkg": EdgeProfileSpec(
        profile_id="gamma_win10_mobile_ppkg",
        short_alias="gamma",
        node_tier="NODE_GAMMA",
        title="Windows 10 Mobile / IoT Edge (.ppkg WIM + OMA-DM .provxml + MTP)",
        os_family="windows_10_mobile",
        default_horizon_s=300,
        transports=("usb_mtp", "ppkg_wim", "device_portal_https"),
    ),
    "delta_posix_ssh_edge": EdgeProfileSpec(
        profile_id="delta_posix_ssh_edge",
        short_alias="delta",
        node_tier="NODE_DELTA",
        title="Generic POSIX / Container SSH Edge Node",
        os_family="posix_ssh",
        default_horizon_s=600,
        transports=("ssh", "https"),
    ),
}

_ALIAS_MAP: Dict[str, str] = {
    **{k: k for k in EDGE_PROFILES},
    **{v.short_alias: k for k, v in EDGE_PROFILES.items()},
}


def resolve_edge_profile(name_or_alias: str) -> EdgeProfileSpec:
    key = _ALIAS_MAP.get((name_or_alias or "").strip().lower())
    if key is None:
        known = ", ".join(sorted(_ALIAS_MAP))
        raise ValueError(f"unknown edge profile {name_or_alias!r}; choose one of: {known}")
    return EDGE_PROFILES[key]


# ═══════════════════════════════════════════════════════════════════════════════
# NON-INTERACTIVE HARDWARE & USB MODE STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class UsbEndpointState:
    friendly_name: str
    instance_id: str
    mode: str  # "MTP_ONLINE" | "UEFI_FFU_FLASH_MODE" | "ADB_ONLINE" | "GENERIC_USB"
    push_allowed: bool
    advisory: str


def classify_usb_pnp_devices(raw_devices: Sequence[Dict[str, Any]]) -> List[UsbEndpointState]:
    """Classify Windows PnP USB device records into safe provisioning states."""
    classified: List[UsbEndpointState] = []
    for item in raw_devices:
        iid = str(item.get("InstanceId") or item.get("instance_id") or "").strip()
        fname = str(item.get("FriendlyName") or item.get("friendly_name") or "").strip()
        cls_name = str(item.get("Class") or item.get("class") or "").strip()
        iid_up = iid.upper()
        fn_low = fname.lower()

        if "VID_045E&PID_062A" in iid_up or "ffu" in fn_low or ("qualcomm msm device" in fn_low and "062A" in iid_up):
            classified.append(UsbEndpointState(
                friendly_name=fname or "QUALCOMM MSM DEVICE (UEFI FFU)",
                instance_id=iid,
                mode="UEFI_FFU_FLASH_MODE",
                push_allowed=False,
                advisory=(
                    "Device is in UEFI Firmware Flash Mode (Lightning Bolt + Gear). "
                    "MTP/AppX file push is BLOCKED. Hold Power + Volume Down for 10s to reboot into OS."
                ),
            ))
        elif (
            "VID_03F0&PID_0155" in iid_up
            or cls_name.upper() == "WPD"
            or "elite x3" in fn_low
            or "mtp" in fn_low
        ):
            classified.append(UsbEndpointState(
                friendly_name=fname or "MTP Portable Device",
                instance_id=iid,
                mode="MTP_ONLINE",
                push_allowed=True,
                advisory="MTP user-mode storage reachable; stage .ppkg without modal delete verbs.",
            ))
        elif (
            "ADB" in iid_up
            or "adb" in fn_low
            or "VID_04E8&PID_6860" in iid_up
            or "galaxy" in fn_low
        ):
            classified.append(UsbEndpointState(
                friendly_name=fname or "Android ADB Endpoint",
                instance_id=iid,
                mode="ADB_ONLINE",
                push_allowed=True,
                advisory="Android USB endpoint present; use `adb forward tcp:8022 tcp:8022` + Termux sshd.",
            ))
    return classified


def _probe_tcp_port(host: str, port: int, timeout: float = 0.25) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        return sock.connect_ex((host, port)) == 0
    except OSError:
        return False
    finally:
        sock.close()


def _query_wsl_windows_usb_pnp(timeout: float = 4.0) -> List[Dict[str, Any]]:
    """Non-interactive Windows PowerShell PnP query when running on WSL2 (`microsoft` kernel)."""
    if "microsoft" not in platform.release().lower():
        return []
    ps_bin = shutil.which("powershell.exe") or (WSL_POWERSHELL if os.path.exists(WSL_POWERSHELL) else None)
    if not ps_bin:
        return []
    ps_cmd = (
        "Get-PnpDevice -PresentOnly | Where-Object { $_.InstanceId -like 'USB\\*' } | "
        "Select-Object FriendlyName,InstanceId,Status,Class | ConvertTo-Json -Compress"
    )
    try:
        res = subprocess.run(
            [ps_bin, "-NoProfile", "-NonInteractive", "-Command", ps_cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            check=False,
        )
        raw = (res.stdout or "").strip()
        if not raw:
            return []
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
    except Exception:
        return []
    return []


def discover_edge_topology(
    *,
    pnp_provider: Optional[Callable[[], Sequence[Dict[str, Any]]]] = None,
    port_probe: Optional[Callable[[str, int], bool]] = None,
    adb_devices_provider: Optional[Callable[[], List[str]]] = None,
) -> Dict[str, Any]:
    """Non-interactively inspect local + attached edge transports and classify USB safety states."""
    probe = port_probe or _probe_tcp_port
    raw_pnp = list(pnp_provider()) if pnp_provider is not None else _query_wsl_windows_usb_pnp()
    usb_states = classify_usb_pnp_devices(raw_pnp)

    if adb_devices_provider is not None:
        adb_devs = list(adb_devices_provider())
    else:
        from osiris_cli.sync_bridge import DualNodeSyncBridge
        adb_info = DualNodeSyncBridge.check_adb_status()
        adb_devs = list(adb_info.get("devices") or [])

    from osiris_cli.s26_cartographer import detect_prime_cores

    ffu_active = any(u.mode == "UEFI_FFU_FLASH_MODE" for u in usb_states)
    return {
        "host_arch": platform.machine().lower(),
        "is_wsl2": "microsoft" in platform.release().lower(),
        "prime_cores": detect_prime_cores(),
        "ssh_8022_listening": bool(probe("127.0.0.1", 8022)),
        "edge_bridge_8080_listening": bool(probe("127.0.0.1", 8080)),
        "ollama_11434_listening": bool(probe("127.0.0.1", 11434)),
        "adb_devices": adb_devs,
        "usb_endpoints": [asdict(u) for u in usb_states],
        "uefi_ffu_flash_block": ffu_active,
        "wimlib_available": shutil.which("wimlib-imagex") is not None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DETERMINISTIC LF-CLEAN PROVISIONING BUNDLE STAGER & VERIFIER
# ═══════════════════════════════════════════════════════════════════════════════


def _write_lf_file(path: Path, content: str, executable: bool = False) -> Tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    data = normalized.encode("utf-8")
    path.write_bytes(data)
    if executable:
        try:
            os.chmod(path, 0o755)
        except OSError:
            pass
    return hashlib.sha256(data).hexdigest(), len(data)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_gamma_ppkg_tree(
    stage_root: Path,
    *,
    package_id: str,
    package_name: str,
    account_email: Optional[str] = None,
    cert_thumbprint: Optional[str] = None,
    cert_der_b64: Optional[str] = None,
) -> List[Path]:
    """Write unconditional Windows 10 Mobile / ICD WIM `.ppkg` XML payload files (LF-clean)."""
    written: List[Path] = []
    esc_name = saxutils.escape(package_name)
    esc_guid = saxutils.escape(package_id)

    common_xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<Elements Type="CommonSettings">\n'
        '  <Element Index="1">\n'
        '  </Element>\n'
        '</Elements>\n'
    )
    answer_xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<WindowsCustomizations>\n'
        '  <PackageConfig xmlns="urn:schemas-Microsoft-com:Windows-ICD-Package-Config.v1.0">\n'
        f'    <ID>{{{esc_guid}}}</ID>\n'
        f'    <Name>{esc_name}</Name>\n'
        '    <Version>1.0</Version>\n'
        '    <OwnerType>ITAdmin</OwnerType>\n'
        '    <Rank>0</Rank>\n'
        '    <Notes />\n'
        '  </PackageConfig>\n'
        '  <Settings xmlns="urn:schemas-microsoft-com:windows-provisioning">\n'
        '    <Customizations>\n'
        '      <Common>\n'
        '        <Policies>\n'
        '          <ApplicationManagement>\n'
        '            <AllowAllTrustedApps>Yes</AllowAllTrustedApps>\n'
        '            <AllowDeveloperUnlock>Yes</AllowDeveloperUnlock>\n'
        '          </ApplicationManagement>\n'
        '        </Policies>\n'
        '      </Common>\n'
        '    </Customizations>\n'
        '  </Settings>\n'
        '</WindowsCustomizations>\n'
    )
    multivar_xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<Elements Type="Multivariant">\n'
        '  <Element Index="0">\n'
        '  </Element>\n'
        '</Elements>\n'
    )

    prov_sets: List[Tuple[str, str, str]] = []
    app_mgmt_provxml = (
        '<?xml version="1.0" encoding="utf-8" standalone="yes"?>\n'
        '<wap-provisioningdoc>\n'
        '  <characteristic type="Policy">\n'
        '    <characteristic type="Config">\n'
        '      <characteristic type="ApplicationManagement">\n'
        '        <parm name="AllowAllTrustedApps" value="1" datatype="integer" />\n'
        '        <parm name="AllowDeveloperUnlock" value="1" datatype="integer" />\n'
        '      </characteristic>\n'
        '    </characteristic>\n'
        '  </characteristic>\n'
        '</wap-provisioningdoc>\n'
    )
    prov_sets.append((
        "e1010001-9053-4e21-b8f0-000000000001",
        "0__OSIRIS_ApplicationManagement_Policy.provxml",
        app_mgmt_provxml,
    ))

    if cert_thumbprint and cert_der_b64:
        clean_tp = re.sub(r"[^0-9A-Fa-f]", "", cert_thumbprint).upper()
        clean_b64 = "".join(cert_der_b64.split())
        cert_provxml = (
            '<?xml version="1.0" encoding="utf-8" standalone="yes"?>\n'
            '<wap-provisioningdoc>\n'
            '  <characteristic type="CertificateStore">\n'
            '    <characteristic type="Root">\n'
            '      <characteristic type="System">\n'
            f'        <characteristic type="{saxutils.escape(clean_tp)}">\n'
            f'          <parm name="EncodedCertificate" value="{saxutils.escape(clean_b64)}" datatype="string" />\n'
            '        </characteristic>\n'
            '      </characteristic>\n'
            '    </characteristic>\n'
            '  </characteristic>\n'
            '</wap-provisioningdoc>\n'
        )
        prov_sets.append((
            "e1010002-9053-4e21-b8f0-000000000002",
            "1__OSIRIS_Certificates_Root.provxml",
            cert_provxml,
        ))

    if account_email:
        esc_email = saxutils.escape(account_email.strip())
        acct_provxml = (
            '<?xml version="1.0" encoding="utf-8" standalone="yes"?>\n'
            '<wap-provisioningdoc>\n'
            '  <characteristic type="Email2">\n'
            '    <characteristic type="{D8A47C12-9053-4E21-B8F0-000000000003}">\n'
            f'      <parm name="SERVICENAME" value="OSIRIS Edge Account ({esc_email})" datatype="string" />\n'
            '      <parm name="SERVICETYPE" value="EAS" datatype="string" />\n'
            '      <parm name="INSERVER" value="eas.outlook.com" datatype="string" />\n'
            '      <parm name="OUTSERVER" value="eas.outlook.com" datatype="string" />\n'
            '      <parm name="DWNDAY" value="7" datatype="integer" />\n'
            f'      <parm name="AUTHNAME" value="{esc_email}" datatype="string" />\n'
            f'      <parm name="REPLYADDR" value="{esc_email}" datatype="string" />\n'
            '      <parm name="SSL" value="1" datatype="integer" />\n'
            '    </characteristic>\n'
            '  </characteristic>\n'
            '</wap-provisioningdoc>\n'
        )
        prov_sets.append((
            "e1010003-9053-4e21-b8f0-000000000003",
            "2__OSIRIS_Account_Designation.provxml",
            acct_provxml,
        ))

    cfg_lines = "\n".join(
        f'  <ConfigurationSet Type="provxml" SettingsGroup="{sg}" Data="$(_prov)\\RunTime\\{fname}" />'
        for sg, fname, _body in prov_sets
    )
    runtime_xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<ConfigurationSource Version="1.0">\n'
        f'{cfg_lines}\n'
        '</ConfigurationSource>\n'
    )

    files_map = {
        stage_root / "CommonSettings" / "CommonSettings.xml": common_xml,
        stage_root / "CommonSettings" / "1" / "AnswerFile.xml": answer_xml,
        stage_root / "Multivariant" / "Multivariant.xml": multivar_xml,
        stage_root / "Multivariant" / "0" / "customizations.xml": answer_xml,
        stage_root / "Multivariant" / "0" / "Prov" / "RunTime.xml": runtime_xml,
    }
    for _sg, fname, body in prov_sets:
        files_map[stage_root / "Multivariant" / "0" / "Prov" / "RunTime" / fname] = body

    for fpath, body in files_map.items():
        _write_lf_file(fpath, body)
        written.append(fpath)
    return written


def _build_posix_or_termux_bootstrap(
    stage_root: Path,
    profile: EdgeProfileSpec,
    node_pubkey_fp: Optional[str],
) -> List[Path]:
    is_termux = profile.os_family == "android_termux"
    repo_root_hint = "$HOME/src/repos" if is_termux else "$HOME/osiris-edge"
    script = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"# OSIRIS Non-Interactive Edge Bootstrap ({profile.profile_id})\n"
        "# Axiom: Cross-node transfer is a data operation, never an authorization grant.\n\n"
        f"EDGE_ROOT=\"{repo_root_hint}\"\n"
        "case \"$EDGE_ROOT\" in\n"
        "  /sdcard/*|/storage/emulated/*)\n"
        "    echo '[!] Refusing /sdcard Scoped Storage (noexec / FUSE latency).' >&2\n"
        "    exit 2\n"
        "    ;;\n"
        "esac\n"
        "mkdir -p \"$EDGE_ROOT\" \"$HOME/.osiris\"\n"
        f"echo 'Profile: {profile.profile_id} | Horizon: {profile.default_horizon_s}s'\n"
        + (f"echo 'Trusted Coordinator Key SHA-256: {node_pubkey_fp}'\n" if node_pubkey_fp else "")
        + ("if command -v sshd >/dev/null 2>&1; then pgrep -x sshd >/dev/null 2>&1 || sshd -p 8022; fi\n" if is_termux else "")
    )
    boot_path = stage_root / "bootstrap_edge_node.sh"
    _write_lf_file(boot_path, script, executable=True)
    return [boot_path]


def stage_provisioning_bundle(
    out_dir: str | Path,
    *,
    profile: str = "alpha_android_termux",
    dna_source: Optional[str] = None,
    account_email: Optional[str] = None,
    cert_thumbprint: Optional[str] = None,
    cert_der_b64: Optional[str] = None,
    key_dir: Optional[str | Path] = None,
    package_id: Optional[str] = None,
    capture_wim: bool = False,
    usb_devices: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compile, audit, and stage a verifiable Edge Node provisioning bundle."""
    prof = resolve_edge_profile(profile)
    if usb_devices is not None:
        states = classify_usb_pnp_devices(usb_devices)
        ffu_hits = [s for s in states if s.mode == "UEFI_FFU_FLASH_MODE"]
        if ffu_hits:
            raise RuntimeError(f"EDGE_USB_FFU_BLOCKED: {ffu_hits[0].advisory}")

    src = dna_source or build_edge_organism_source(
        profile_id=prof.profile_id,
        default_horizon_s=prof.default_horizon_s,
    )
    compiled: CompiledEdgeOrganism = compile_governed_edge_organism(src)

    bundle_root = Path(out_dir).resolve()
    bundle_root.mkdir(parents=True, exist_ok=True)
    org_dir = bundle_root / "organism"
    payload_dir = bundle_root / "payload"

    dna_path = org_dir / f"{prof.short_alias}_edge.dna"
    ir_path = org_dir / f"{prof.short_alias}_edge.ir.json"
    ledger_path = org_dir / "provisioning_ledger.jsonl"
    if ledger_path.exists():
        ledger_path.unlink()

    ir_payload = {
        "organism": compiled.name,
        "source_sha256": compiled.source_sha256,
        "meta": compiled.meta,
        "rules": compiled.rules.to_dict() if compiled.rules else None,
        "rules_sha256": compiled.rules_sha256,
        "regulation": compiled.regulation.to_dict() if compiled.regulation else None,
        "regulation_sha256": compiled.regulation_sha256,
        "circuit": compiled.circuit.to_dict() if compiled.circuit else None,
        "circuit_sha256": compiled.circuit_sha256,
    }
    _write_lf_file(dna_path, compiled.source)
    _write_lf_file(ir_path, json.dumps(ir_payload, indent=2, sort_keys=True) + "\n")

    pubkey_fp: Optional[str] = None
    if key_dir is not None:
        from osiris_cli.edge_ipc_client import get_public_key_bytes, public_key_fingerprint
        pub_raw = get_public_key_bytes(Path(key_dir))
        pubkey_fp = public_key_fingerprint(pub_raw)
        pub_b64_path = bundle_root / "coordinator_ed25519.pub.json"
        _write_lf_file(
            pub_b64_path,
            json.dumps({"fingerprint_sha256": pubkey_fp, "key_type": "Ed25519"}, indent=2) + "\n",
        )

    pkg_guid = (package_id or str(uuid.uuid5(uuid.NAMESPACE_DNS, f"osiris.edge.{prof.profile_id}"))).strip("{}")
    staged_paths: List[Path] = [dna_path, ir_path]
    if key_dir is not None:
        staged_paths.append(bundle_root / "coordinator_ed25519.pub.json")

    if prof.os_family == "windows_10_mobile":
        staged_paths.extend(_build_gamma_ppkg_tree(
            payload_dir,
            package_id=pkg_guid,
            package_name=f"OSIRIS.{prof.short_alias.capitalize()}.Provisioning",
            account_email=account_email,
            cert_thumbprint=cert_thumbprint,
            cert_der_b64=cert_der_b64,
        ))
    else:
        staged_paths.extend(_build_posix_or_termux_bootstrap(payload_dir, prof, pubkey_fp))

    ppkg_out: Optional[Path] = None
    if capture_wim and prof.os_family == "windows_10_mobile":
        wim_bin = shutil.which("wimlib-imagex")
        if not wim_bin:
            raise RuntimeError("wimlib-imagex is not installed in PATH; cannot capture .ppkg WIM archive")
        ppkg_out = bundle_root / "OSIRIS.Edge.Provisioning.ppkg"
        if ppkg_out.exists():
            ppkg_out.unlink()
        subprocess.run(
            [
                wim_bin, "capture", str(payload_dir), str(ppkg_out),
                f"OSIRIS.{prof.short_alias.capitalize()}.Provisioning",
                "OSIRIS Governed Edge Provisioning Package",
                "--compress=XPRESS", "--chunk-size=32768", "--no-acls", "--check",
                "--image-property", f"PACKAGEID={{{pkg_guid}}}",
                "--image-property", "VERSION=1.0",
                "--image-property", "ALTITUDE=5000",
            ],
            check=True,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        )
        staged_paths.append(ppkg_out)

    led = Ledger(str(ledger_path))
    ledger_entry = led.append("SUBSTRATE_PROVISION_STAGED", {
        "profile_id": prof.profile_id,
        "node_tier": prof.node_tier,
        "package_id": pkg_guid,
        "organism": compiled.name,
        "organism_sha256": compiled.source_sha256,
        "rules_sha256": compiled.rules_sha256,
        "regulation_sha256": compiled.regulation_sha256,
        "coordinator_pubkey_sha256": pubkey_fp,
        "ppkg_captured": ppkg_out is not None,
    })
    ledger_head = ledger_entry["hash"]
    staged_paths.append(ledger_path)

    from osiris_cli import exec_policy
    _pol, policy_sha = exec_policy.load_policy()

    files_inventory: Dict[str, str] = {}
    for p in sorted(staged_paths):
        rel = p.relative_to(bundle_root).as_posix()
        files_inventory[rel] = _sha256_file(p)

    manifest = {
        "schema": "osiris.edge.bundle/1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile_id": prof.profile_id,
        "node_tier": prof.node_tier,
        "package_id": pkg_guid,
        "exec_policy_sha256": policy_sha,
        "organism": compiled.summary_dict(),
        "ledger_head_sha256": ledger_head,
        "coordinator_pubkey_sha256": pubkey_fp,
        "files": files_inventory,
    }
    manifest_path = bundle_root / "manifest.json"
    _write_lf_file(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def verify_provisioning_bundle(bundle_dir: str | Path) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    """Offline verification of SHA-256 file inventory, LF cleanliness, Claims Firewall, and dnalang Ledger."""
    root = Path(bundle_dir).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return False, [f"missing manifest.json in {root}"], None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, [f"corrupt manifest.json: {exc}"], None

    errors: List[str] = []
    files_map = manifest.get("files")
    if not isinstance(files_map, dict) or not files_map:
        errors.append("manifest.json contains an empty 'files' inventory")
        return False, errors, manifest

    dna_rel: Optional[str] = None
    ledger_rel: Optional[str] = None
    ppkg_rel: Optional[str] = None

    for rel, expected_sha in sorted(files_map.items()):
        fpath = (root / rel).resolve()
        if not str(fpath).startswith(str(root) + os.sep):
            errors.append(f"path traversal in manifest entry: {rel!r}")
            continue
        if not fpath.is_file():
            errors.append(f"missing staged file: {rel}")
            continue
        actual_sha = _sha256_file(fpath)
        if actual_sha != expected_sha:
            errors.append(f"SHA-256 mismatch on {rel}: expected {expected_sha[:12]}, got {actual_sha[:12]}")
        if rel.endswith((".dna", ".xml", ".provxml", ".sh", ".json", ".jsonl")):
            raw_bytes = fpath.read_bytes()
            if b"\r\n" in raw_bytes:
                errors.append(f"CRLF line endings found in {rel} (LF required)")
        if rel.endswith(".dna"):
            dna_rel = rel
        elif rel.endswith("provisioning_ledger.jsonl"):
            ledger_rel = rel
        elif rel.endswith(".ppkg"):
            ppkg_rel = rel

    if dna_rel:
        try:
            compiled = compile_governed_edge_organism((root / dna_rel).read_text(encoding="utf-8"))
            expected_org_sha = (manifest.get("organism") or {}).get("source_sha256")
            if expected_org_sha and compiled.source_sha256 != expected_org_sha:
                errors.append("compiled .dna source_sha256 does not match manifest")
        except Exception as exc:
            errors.append(f".dna verification failed: {exc}")
    else:
        errors.append("bundle contains no .dna organism file")

    if ledger_rel:
        led = Ledger(str(root / ledger_rel))
        led_problem = led.verify()
        if led_problem is not None:
            errors.append(f"dnalang Ledger hash chain broken in {ledger_rel}: {led_problem}")
    else:
        errors.append("bundle contains no provisioning_ledger.jsonl")

    if ppkg_rel and shutil.which("wimlib-imagex"):
        res = subprocess.run(
            ["wimlib-imagex", "verify", str(root / ppkg_rel)],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            check=False,
        )
        if res.returncode != 0:
            errors.append(f"wimlib-imagex verify failed on {ppkg_rel}: {(res.stderr or res.stdout).strip()}")

    return len(errors) == 0, errors, manifest


def simulate_edge_administration(
    *,
    profile: str = "alpha_android_termux",
    dna_path: Optional[str | Path] = None,
    scenarios: Optional[Sequence[Tuple[str, EdgeTelemetry]]] = None,
    ledger_path: Optional[str | Path] = None,
) -> List[Dict[str, Any]]:
    """Run an adaptive `dna::}{::lang` edge organism across one or more telemetry scenarios."""
    prof = resolve_edge_profile(profile)
    src = (
        Path(dna_path).read_text(encoding="utf-8")
        if dna_path is not None
        else build_edge_organism_source(prof.profile_id, default_horizon_s=prof.default_horizon_s)
    )
    compiled = compile_governed_edge_organism(src)
    rt = EdgeOrganismRuntime(
        compiled,
        step_budget=64,
        ledger_path=str(ledger_path) if ledger_path is not None else None,
    )

    run_scenarios = list(scenarios) if scenarios is not None else [
        ("genesis_nominal", EdgeTelemetry(temp_c=33.0, battery_pct=92.0, link_rtt_ms=18.0)),
        ("thermal_battery_stress", EdgeTelemetry(temp_c=45.5, battery_pct=15.0, link_rtt_ms=640.0, network_up=False)),
        ("uefi_ffu_flash_guard", EdgeTelemetry(temp_c=35.0, battery_pct=80.0, usb_ffu_mode=1, transport_ready=False)),
        ("ledger_tamper_quarantine", EdgeTelemetry(temp_c=34.0, battery_pct=85.0, ledger_intact=0)),
    ]

    report: List[Dict[str, Any]] = []
    for label, telem in run_scenarios:
        state = rt.step(telem)
        report.append({
            "scenario": label,
            "register": telem.capability_register(),
            "metrics": telem.to_metrics(),
            "decision": state.to_dict(),
        })
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# CLI & REPL DISPATCHER (`osiris provision ...` / `/provision ...`)
# ═══════════════════════════════════════════════════════════════════════════════


def run_provision_command(argv: Sequence[str]) -> int:
    """Execute `osiris provision {status|stage|verify|simulate}` non-interactively."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="osiris provision",
        description="Governed multi-substrate Edge Node discovery, .dna provisioning, and simulation",
    )
    sub = parser.add_subparsers(dest="subcommand")

    p_status = sub.add_parser("status", help="Inspect attached edge transports, ADB/SSH, and USB safety states")
    p_status.add_argument("--json", action="store_true", help="Emit JSON status payload")

    p_stage = sub.add_parser("stage", help="Compile .dna organism and stage a verifiable provisioning bundle")
    p_stage.add_argument("--profile", default="alpha", help="Edge profile (alpha|beta|gamma|delta)")
    p_stage.add_argument("--out", required=True, help="Destination bundle directory")
    p_stage.add_argument("--dna", default=None, help="Optional custom .dna organism file path")
    p_stage.add_argument("--account", default=None, help="Optional Exchange ActiveSync email for gamma .ppkg")
    p_stage.add_argument("--key-dir", default=None, help="Optional Ed25519 key directory to enroll public fingerprint")
    p_stage.add_argument("--capture-wim", action="store_true", help="Capture .ppkg WIM archive via wimlib-imagex")

    p_verify = sub.add_parser("verify", help="Verify SHA-256 manifest, LF line endings, Claims Firewall, and Ledger")
    p_verify.add_argument("bundle_dir", help="Path to staged provisioning bundle directory")

    p_sim = sub.add_parser("simulate", help="Simulate adaptive dna::}{::lang edge administration across telemetry")
    p_sim.add_argument("--profile", default="alpha", help="Edge profile (alpha|beta|gamma|delta)")
    p_sim.add_argument("--dna", default=None, help="Optional custom .dna organism file path")
    p_sim.add_argument("--temp-c", type=float, default=None, help="Single-tick override: device temperature (C)")
    p_sim.add_argument("--battery-pct", type=float, default=None, help="Single-tick override: battery percentage")
    p_sim.add_argument("--rtt-ms", type=float, default=None, help="Single-tick override: link RTT in ms")
    p_sim.add_argument("--usb-ffu", action="store_true", help="Single-tick override: UEFI FFU Flash Mode active")
    p_sim.add_argument("--ledger-tampered", action="store_true", help="Single-tick override: broken ledger chain")
    p_sim.add_argument("--ledger-out", default=None, help="Optional dnalang.Ledger output path")

    if not argv:
        argv = ["status"]
    try:
        args = parser.parse_args(list(argv))
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2

    if args.subcommand == "status":
        topo = discover_edge_topology()
        if args.json:
            print(json.dumps(topo, indent=2, sort_keys=True))
            return 0
        print("[OSIRIS::EDGE PROVISIONING & TRANSPORT MATRIX]")
        print(f"  ├─ Host Arch / WSL2     : {topo['host_arch']} (WSL2={topo['is_wsl2']}, Prime Cores={topo['prime_cores']})")
        print(f"  ├─ SSH (:8022)          : {'LISTENING' if topo['ssh_8022_listening'] else 'OFFLINE'}")
        print(f"  ├─ Edge Bridge (:8080)  : {'LISTENING' if topo['edge_bridge_8080_listening'] else 'OFFLINE'}")
        print(f"  ├─ Local Ollama (:11434): {'LISTENING' if topo['ollama_11434_listening'] else 'OFFLINE'}")
        print(f"  ├─ ADB Devices          : {len(topo['adb_devices'])} attached")
        print(f"  ├─ wimlib-imagex        : {'AVAILABLE' if topo['wimlib_available'] else 'NOT INSTALLED'}")
        if topo["usb_endpoints"]:
            for ep in topo["usb_endpoints"]:
                flag = "BLOCKED" if not ep["push_allowed"] else "READY"
                print(f"  ├─ USB [{ep['mode']}] ({flag}): {ep['friendly_name']} -- {ep['advisory']}")
        else:
            print("  └─ USB Endpoints        : none enumerated")
        return 0

    if args.subcommand == "stage":
        try:
            custom_src = Path(args.dna).read_text(encoding="utf-8") if args.dna else None
            manifest = stage_provisioning_bundle(
                args.out,
                profile=args.profile,
                dna_source=custom_src,
                account_email=args.account,
                key_dir=args.key_dir,
                capture_wim=args.capture_wim,
            )
        except Exception as exc:
            print(f"[OSIRIS::PROVISION ERROR] {exc}")
            return 1
        org_info = manifest["organism"]
        print(
            f"[OSIRIS::PROVISION STAGED] profile={manifest['profile_id']} ({manifest['node_tier']})\n"
            f"  ├─ Bundle Dir      : {Path(args.out).resolve()}\n"
            f"  ├─ Organism        : {org_info['organism']} (sha256={org_info['source_sha256'][:16]})\n"
            f"  ├─ Ledger Head     : {manifest['ledger_head_sha256'][:16]}\n"
            f"  └─ Staged Files    : {len(manifest['files'])} verified LF-clean artifact(s)"
        )
        return 0

    if args.subcommand == "verify":
        ok, errs, manifest = verify_provisioning_bundle(args.bundle_dir)
        if ok and manifest is not None:
            print(
                f"[OSIRIS::PROVISION VERIFIED] PASS ({len(manifest['files'])} files, "
                f"organism={manifest['organism']['organism']}, ledger intact)"
            )
            return 0
        print("[OSIRIS::PROVISION VERIFIED] FAIL:")
        for e in errs:
            print(f"  - {e}")
        return 1

    if args.subcommand == "simulate":
        custom_scenarios = None
        if any(
            x is not None
            for x in (args.temp_c, args.battery_pct, args.rtt_ms)
        ) or args.usb_ffu or args.ledger_tampered:
            t = EdgeTelemetry(
                temp_c=args.temp_c if args.temp_c is not None else 34.0,
                battery_pct=args.battery_pct if args.battery_pct is not None else 85.0,
                link_rtt_ms=args.rtt_ms if args.rtt_ms is not None else 25.0,
                usb_ffu_mode=1 if args.usb_ffu else 0,
                transport_ready=not args.usb_ffu,
                network_up=(args.rtt_ms is None or args.rtt_ms <= 500.0),
                ledger_intact=0 if args.ledger_tampered else 1,
            )
            custom_scenarios = [("custom_probe", t)]
        try:
            rows = simulate_edge_administration(
                profile=args.profile,
                dna_path=args.dna,
                scenarios=custom_scenarios,
                ledger_path=args.ledger_out,
            )
        except Exception as exc:
            print(f"[OSIRIS::PROVISION SIMULATE ERROR] {exc}")
            return 1
        print(f"[OSIRIS::EDGE ORGANISM SIMULATION] ({len(rows)} scenario(s))")
        for r in rows:
            dec = r["decision"]
            sev = ",".join(dec["severed_capabilities"]) or "none"
            evts = ",".join(dec["emitted_events"]) or "none"
            print(
                f"  • {r['scenario']:<24} reg={r['register']} -> route={dec['active_route']} "
                f"| horizon={dec['horizon_s']:.0f}s | severed=[{sev}] | events=[{evts}]"
            )
        return 0

    parser.print_help()
    return 2
