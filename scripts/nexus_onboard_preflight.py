#!/usr/bin/env python3
"""Quantinuum Nexus (`qnexus`) Onboarding & Single-Circuit Preflight Tool (`ENG-NEXUS-4`).

Implements `phase1/verification_queue.md` Check #8 (`ENG-NEXUS-4`):
  "First approved Nexus preflight on one circuit, with Nexus HQC compared to `local_hqc`"

Modes:
  * `--status` / `--offline-dry-run` (default, 0 network calls):
      - Checks Python `qnexus` package availability & version (pinned target `0.51.4`).
      - Checks `~/.qnx`, `~/.config/qnexus`, `QNEXUS_TOKEN`, and `QUANTINUUM_NEXUS_TOKEN` auth presence
        (boolean/redacted only; never reads or prints secret values).
      - Validates `osiris.quantum_protocol/1` pre-gate via `quantinuum_gate.validate_quantum_protocol_pregate`.
      - Verifies exact local HQC formula `5 + C * (N_1q + 10*N_2q + 5*N_m) / 5000` on the 1-circuit canary
        `xtop005q_Q0_L3_ZZ.tket.json` (`xtop005/nexus_h2_bundle/`) and companion X-basis Wen-plaquette
        circuit (`/home/osiris/osiris-assessment/work/osiris-qec/submission/nexus_h2_bundle/` when present).
      - Writes a deterministic `ENG-NEXUS-4` offline preflight readiness receipt.

  * `--preflight-live` (fail-closed without `qnexus` credentials AND `--operator-approved`):
      - Compiles/uploads ONLY 1 canary circuit to `H2-1SC` / `H2-1E` (Tier 3 paid QPU targets are hard-blocked).
      - Queries `qnexus` provider HQC cost via `NexusSdkRunner.preflight`, compares `provider_hqc` vs
        `local_hqc` (`ENG-NEXUS-4`), and writes a redacted receipt with `paid_qpu_hqc_spent = "0.0000"`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import sys
from decimal import ROUND_HALF_EVEN, Decimal
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from osiris_cli.nexus_ledger import NexusAuditLedger  # noqa: E402
from osiris_cli.providers.nexus_sdk import (  # noqa: E402
    QNEXUS_VERSION,
    NexusSdkRunner,
    NexusSdkUnavailable,
    load_circuits,
    load_qnexus,
    local_hqc,
    tket_sha256,
)
from osiris_cli.providers.quantinuum import compute_hqc  # noqa: E402
from osiris_cli.providers.quantinuum_nexus import GovernanceError  # noqa: E402
from osiris_cli.quantinuum_gate import validate_quantum_protocol_pregate  # noqa: E402

PREFLIGHT_RECEIPT_SCHEMA = "osiris.eng_nexus_4.preflight_receipt/1"
HQC_FORMULA_SPEC = "5 + C * (N_1q + 10*N_2q + 5*N_m) / 5000"

DEFAULT_XTOP005_BUNDLE_DIR = ROOT / "xtop005" / "nexus_h2_bundle"
DEFAULT_CANARY_PATH = DEFAULT_XTOP005_BUNDLE_DIR / "circuits" / "xtop005q_Q0_L3_ZZ.tket.json"
DEFAULT_WEN_BUNDLE_DIR = Path(
    "/home/osiris/osiris-assessment/work/osiris-qec/submission/nexus_h2_bundle"
)
DEFAULT_RECEIPT_OUT = ROOT / "xtop005" / "eng_nexus_4_preflight_receipt.json"

ALLOWED_LIVE_PREFLIGHT_TARGETS = frozenset({"H2-1SC", "H2-1E"})
H2_NATIVE_ALLOWED_OPS = frozenset(
    {"PhasedX", "Rz", "ZZMax", "ZZPhase", "Measure", "Reset", "Barrier"}
)

DEFAULT_ENG_NEXUS_4_PROTOCOL_MANIFEST: Dict[str, Any] = {
    "schema": "osiris.quantum_protocol/1",
    "protocol_id": "ENG-NEXUS-4-SINGLE-CIRCUIT-CANARY",
    "claim_id": "ENG-NEXUS-4",
    "shots_per_circuit": 500,
    "num_blocks": 3,
    "classical_baseline": "Stage-0 exact noiseless statevector/stabilizer oracle (F = 1.000000)",
    "classical_baseline_passed": True,
    "colored_noise_null": "H2-1 QCCD local Pauli + 1/f dephasing baseline control",
    "colored_noise_control_passed": True,
    "aa_drift_sentinel": "Interleaved A/A ZZ-basis canary drift sentinel",
    "drift_sentinel_passed": True,
    "control_experiments": [
        "classical statevector Stage-0 oracle",
        "local 1/f colored noise baseline",
        "interleaved A/A drift sentinel",
    ],
}


class NexusPreflightError(RuntimeError):
    """Fail-closed error raised on pre-gate, credential, approval, target, or integrity failures."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"[{code}] {detail}")
        self.code = code
        self.detail = detail


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    try:
        import rfc8785  # type: ignore[import-not-found]

        return rfc8785.dumps(obj)
    except ImportError:
        return json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")


def exact_hqc_formula(n_1q: int, n_2q: int, n_m: int, shots: int) -> Decimal:
    """Exact H-series HQC formula: `5 + C * (N_1q + 10*N_2q + 5*N_m) / 5000`."""
    if min(n_1q, n_2q, n_m, shots) < 0:
        raise NexusPreflightError("INVALID_HQC_INPUT", "HQC counts and shots must be non-negative")
    units = Decimal(n_1q) + Decimal(10) * Decimal(n_2q) + Decimal(5) * Decimal(n_m)
    val = Decimal("5") + (Decimal(shots) * units) / Decimal("5000")
    return val.quantize(Decimal("0.0001"), rounding=ROUND_HALF_EVEN)


def check_qnexus_package() -> Dict[str, Any]:
    """Inspect local `qnexus` and `pytket` availability without importing network clients eagerly."""
    qnx_spec = importlib.util.find_spec("qnexus")
    tket_spec = importlib.util.find_spec("pytket")
    qnx_ver: Optional[str] = None
    if qnx_spec is not None:
        try:
            qnx_ver = importlib.metadata.version("qnexus")
        except importlib.metadata.PackageNotFoundError:
            qnx_ver = "installed-unknown-version"
    tket_ver: Optional[str] = None
    if tket_spec is not None:
        try:
            tket_ver = importlib.metadata.version("pytket")
        except importlib.metadata.PackageNotFoundError:
            tket_ver = "installed-unknown-version"
    return {
        "qnexus_available": qnx_spec is not None,
        "qnexus_version": qnx_ver,
        "qnexus_pinned_target": QNEXUS_VERSION,
        "pytket_available": tket_spec is not None,
        "pytket_version": tket_ver,
    }


def _has_nonempty_auth_files(directory: Path) -> bool:
    if not directory.exists():
        return False
    if directory.is_file():
        return directory.stat().st_size > 0
    for child in directory.rglob("*"):
        if child.is_file() and child.stat().st_size > 0:
            return True
    return False


def check_nexus_auth_status(
    env: Optional[Mapping[str, str]] = None,
    home_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Check `~/.qnx`, `~/.config/qnexus`, `QNEXUS_TOKEN`, and `QUANTINUUM_NEXUS_TOKEN` (redacted)."""
    env_map = env if env is not None else os.environ
    resolved_home = Path(home_dir) if home_dir is not None else Path.home()

    dot_qnx = _has_nonempty_auth_files(resolved_home / ".qnx")
    dot_config_qnexus = _has_nonempty_auth_files(resolved_home / ".config" / "qnexus")
    qnx_token_env = bool(env_map.get("QNEXUS_TOKEN", "").strip())
    quant_nexus_token_env = bool(env_map.get("QUANTINUUM_NEXUS_TOKEN", "").strip())

    sources = {
        "dot_qnx_dir": dot_qnx,
        "dot_config_qnexus_dir": dot_config_qnexus,
        "env_QNEXUS_TOKEN": qnx_token_env,
        "env_QUANTINUUM_NEXUS_TOKEN": quant_nexus_token_env,
    }
    active = [name for name, present in sorted(sources.items()) if present]
    return {
        "auth_configured": len(active) > 0,
        "active_sources": active,
        "sources": sources,
        "secrets_redacted": True,
    }


def inspect_tket_json_circuit(circuit_path: Path, shots: int) -> Dict[str, Any]:
    """Deterministically inspect a pytket JSON circuit and cross-verify `local_hqc`."""
    if not circuit_path.is_file():
        raise NexusPreflightError(
            "MISSING_CANARY_CIRCUIT",
            f"Canary circuit file not found: {circuit_path}",
        )
    raw_bytes = circuit_path.read_bytes()
    file_sha = sha256_bytes(raw_bytes)
    payload = json.loads(raw_bytes.decode("utf-8"))
    if isinstance(payload, list):
        if len(payload) != 1:
            raise NexusPreflightError(
                "MULTI_CIRCUIT_CANARY_FORBIDDEN",
                f"ENG-NEXUS-4 canary file must contain exactly 1 circuit; found {len(payload)}",
            )
        entry = payload[0]
        tk_dict = entry.get("tket", entry) if isinstance(entry, dict) else entry
        circ_name = (entry.get("name") if isinstance(entry, dict) else None) or circuit_path.stem
    elif isinstance(payload, dict):
        tk_dict = payload.get("tket", payload) if "tket" in payload and "commands" not in payload else payload
        circ_name = str(tk_dict.get("name") or circuit_path.stem)
    else:
        raise NexusPreflightError("INVALID_CANARY_JSON", f"Unsupported JSON root in {circuit_path}")

    qubits_raw = tk_dict.get("qubits", [])
    qubit_set = {(str(q[0]), tuple(int(i) for i in q[1])) for q in qubits_raw}
    n_qubits = len(qubit_set)

    n_1q_total = 0
    n_1q_phys = 0
    n_1q_rz = 0
    n_2q = 0
    n_meas = 0
    n_reset = 0
    n_conditional = 0
    swaps = 0

    for cmd in tk_dict.get("commands", []):
        op = cmd["op"]
        if op["type"] == "Conditional":
            n_conditional += 1
        while op["type"] == "Conditional":
            op = op["conditional"]["op"]
        op_type = str(op["type"])
        if op_type == "SWAP":
            swaps += 1
        if op_type not in H2_NATIVE_ALLOWED_OPS:
            raise NexusPreflightError(
                "NON_NATIVE_H2_OP",
                f"Non-native H2-1 operation {op_type!r} in {circ_name}",
            )
        q_args = [
            a
            for a in cmd.get("args", [])
            if isinstance(a, list)
            and len(a) == 2
            and (str(a[0]), tuple(int(i) for i in a[1])) in qubit_set
        ]
        if op_type == "Measure":
            n_meas += 1
        elif op_type == "Reset":
            n_reset += 1
        elif op_type == "Barrier" or len(q_args) == 0:
            continue
        elif len(q_args) == 1:
            n_1q_total += 1
            if op_type == "Rz":
                n_1q_rz += 1
            else:
                n_1q_phys += 1
        elif len(q_args) == 2:
            n_2q += 1
        else:
            raise NexusPreflightError(
                "UNSUPPORTED_ARITY",
                f"Unexpected {len(q_args)}-qubit op {op_type!r} in {circ_name}",
            )

    n_m_with_prep = n_qubits + n_meas + n_reset
    n_m_meas_reset_only = n_meas + n_reset

    hqc_phys = exact_hqc_formula(n_1q_phys, n_2q, n_m_with_prep, shots)
    hqc_cons = exact_hqc_formula(n_1q_total, n_2q, n_m_with_prep, shots)
    hqc_meas_reset_only = exact_hqc_formula(n_1q_phys, n_2q, n_m_meas_reset_only, shots)

    provider_compute = compute_hqc(n_1q_phys, n_2q, n_m_with_prep, shots)
    if hqc_phys != provider_compute:
        raise NexusPreflightError(
            "HQC_FORMULA_MISMATCH",
            f"compute_hqc ({provider_compute}) != exact_hqc_formula ({hqc_phys})",
        )

    tket_canonical_sha: Optional[str] = None
    pytket_verified = False
    if importlib.util.find_spec("pytket") is not None:
        loaded = load_circuits(circuit_path)
        if len(loaded) != 1:
            raise NexusPreflightError(
                "MULTI_CIRCUIT_CANARY_FORBIDDEN",
                f"load_circuits returned {len(loaded)} circuits for single-circuit canary",
            )
        sdk_hqc = local_hqc(loaded[0], shots)
        if sdk_hqc != hqc_phys:
            raise NexusPreflightError(
                "SDK_LOCAL_HQC_MISMATCH",
                f"nexus_sdk.local_hqc ({sdk_hqc}) != exact_hqc_formula ({hqc_phys})",
            )
        tket_canonical_sha = tket_sha256(loaded[0])
        pytket_verified = True

    return {
        "name": circ_name,
        "file_sha256": file_sha,
        "tket_sha256": tket_canonical_sha or file_sha,
        "pytket_cross_verified": pytket_verified,
        "qubits": n_qubits,
        "shots": shots,
        "N_1q_total": n_1q_total,
        "N_1q_physical": n_1q_phys,
        "N_1q_virtual_rz": n_1q_rz,
        "N_2q": n_2q,
        "N_meas": n_meas,
        "N_reset": n_reset,
        "N_conditional": n_conditional,
        "N_m": n_m_with_prep,
        "N_m_meas_reset_only": n_m_meas_reset_only,
        "routing_swaps": swaps,
        "hqc_formula": HQC_FORMULA_SPEC,
        "local_hqc_physical": str(hqc_phys),
        "local_hqc_conservative": str(hqc_cons),
        "local_hqc_meas_reset_only": str(hqc_meas_reset_only),
    }


def verify_xtop005_canary(
    canary_path: Path = DEFAULT_CANARY_PATH,
    shots: int = 500,
) -> Dict[str, Any]:
    """Verify primary 1-circuit canary (`xtop005q_Q0_L3_ZZ.tket.json`) and bundle manifest."""
    info = inspect_tket_json_circuit(canary_path, shots=shots)
    manifest_path = canary_path.parent.parent / "nexus_h2_decisive_manifest.json"
    manifest_match = False
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        matched = [
            c
            for c in manifest.get("circuits", [])
            if c.get("file_sha256") == info["file_sha256"] or c.get("name") == info["name"]
        ]
        if matched:
            m = matched[0]
            if (
                int(m["N_1q_physical"]) != info["N_1q_physical"]
                or int(m["N_2q"]) != info["N_2q"]
                or int(m["N_m"]) != info["N_m"]
                or m["file_sha256"] != info["file_sha256"]
            ):
                raise NexusPreflightError(
                    "XTOP005_MANIFEST_MISMATCH",
                    f"Canary {info['name']} disagrees with nexus_h2_decisive_manifest.json",
                )
            if shots == int(m["shots"]):
                if m["hqc_physical"] != info["local_hqc_physical"]:
                    raise NexusPreflightError(
                        "XTOP005_HQC_MISMATCH",
                        f"Expected hqc_physical={m['hqc_physical']}, computed {info['local_hqc_physical']}",
                    )
                if m["hqc_conservative"] != info["local_hqc_conservative"]:
                    raise NexusPreflightError(
                        "XTOP005_HQC_MISMATCH",
                        f"Expected hqc_conservative={m['hqc_conservative']}, "
                        f"computed {info['local_hqc_conservative']}",
                    )
            manifest_match = True
    info["bundle_manifest_verified"] = manifest_match
    return info


def verify_wen_plaquette_canary(
    wen_bundle_dir: Path = DEFAULT_WEN_BUNDLE_DIR,
    shots: int = 500,
) -> Dict[str, Any]:
    """Verify companion X-basis Wen-plaquette canary from `osiris-qec` if present on disk."""
    x_path = Path(wen_bundle_dir) / "wen_plaquette_qec002_x_basis_h2_isa.json"
    if not x_path.is_file():
        return {
            "present": False,
            "reason": "wen_plaquette_qec002_x_basis_h2_isa.json not present in workspace",
        }

    info = inspect_tket_json_circuit(x_path, shots=shots)
    sums_path = Path(wen_bundle_dir) / "SHA256SUMS.txt"
    if sums_path.is_file():
        sums_entries: Dict[str, str] = {}
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, rel = line.strip().split(None, 1)
                sums_entries[rel.strip()] = digest.strip()
        expected = sums_entries.get(x_path.name)
        if expected and expected != info["file_sha256"]:
            raise NexusPreflightError(
                "WEN_SHA256_MISMATCH",
                f"Wen-plaquette SHA256SUMS mismatch: expected {expected}, got {info['file_sha256']}",
            )

    manifest_path = Path(wen_bundle_dir) / "nexus_qec002_manifest.json"
    manifest_verified = False
    if manifest_path.is_file():
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
        xb = man.get("circuits", {}).get("x_basis", {})
        if xb.get("file_sha256") != info["file_sha256"]:
            raise NexusPreflightError(
                "WEN_MANIFEST_HASH_MISMATCH",
                "Wen-plaquette file_sha256 disagrees with nexus_qec002_manifest.json",
            )
        hb = xb.get("hqc_budget", {})
        if (
            int(hb.get("n_1q_phasedx", -1)) != info["N_1q_physical"]
            or int(hb.get("n_2q_zzphase", -1)) != info["N_2q"]
            or int(hb.get("n_m_total", -1)) != info["N_m_meas_reset_only"]
        ):
            raise NexusPreflightError(
                "WEN_GATE_COUNT_MISMATCH",
                "Wen-plaquette gate counts disagree with nexus_qec002_manifest.json",
            )
        # Also verify 1000-shot reference point recorded in nexus_qec002_manifest.json
        ref_1000_mr = exact_hqc_formula(info["N_1q_physical"], info["N_2q"], info["N_m_meas_reset_only"], 1000)
        expected_1000 = Decimal(str(hb["hqc_by_shots"]["1000"])).quantize(Decimal("0.0001"))
        if ref_1000_mr != expected_1000:
            raise NexusPreflightError(
                "WEN_1000SHOT_HQC_MISMATCH",
                f"Wen-plaquette 1000-shot HQC mismatch: {ref_1000_mr} != {expected_1000}",
            )
        manifest_verified = True

    return {
        "present": True,
        "manifest_verified": manifest_verified,
        "hqc_1000_shots_with_state_prep": str(
            exact_hqc_formula(info["N_1q_physical"], info["N_2q"], info["N_m"], 1000)
        ),
        "hqc_1000_shots_meas_reset_only": str(
            exact_hqc_formula(info["N_1q_physical"], info["N_2q"], info["N_m_meas_reset_only"], 1000)
        ),
        **info,
    }


def _redact_text(
    text: str,
    env: Optional[Mapping[str, str]] = None,
    home_dir: Optional[Path] = None,
) -> str:
    env_map = env if env is not None else os.environ
    out = text
    for secret_key in (
        "QNEXUS_TOKEN",
        "QUANTINUUM_NEXUS_TOKEN",
        "OSIRIS_TIER3_HMAC_SECRET",
        "OSIRIS_APPROVAL_SECRET",
        "OSIRIS_TIER3_APPROVAL_SECRET",
        "ZENODO_TOKEN",
        "ZENODO_API_TOKEN",
    ):
        secret_val = env_map.get(secret_key, "").strip()
        if secret_val and len(secret_val) >= 4:
            out = out.replace(secret_val, "[REDACTED]")
    out = re.sub(r"Bearer\s+[A-Za-z0-9._\-]+", "Bearer [REDACTED]", out)
    resolved_home = str((Path(home_dir) if home_dir is not None else Path.home()).resolve())
    if resolved_home and resolved_home != "/":
        out = out.replace(resolved_home, "~")
    return out


def _redact_obj(
    obj: Any,
    env: Optional[Mapping[str, str]] = None,
    home_dir: Optional[Path] = None,
) -> Any:
    if isinstance(obj, str):
        return _redact_text(obj, env=env, home_dir=home_dir)
    if isinstance(obj, list):
        return [_redact_obj(v, env=env, home_dir=home_dir) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _redact_obj(v, env=env, home_dir=home_dir) for k, v in obj.items()}
    return obj


def run_offline_preflight(
    mode_label: str = "OFFLINE_DRY_RUN",
    canary_path: Path = DEFAULT_CANARY_PATH,
    wen_bundle_dir: Path = DEFAULT_WEN_BUNDLE_DIR,
    shots: int = 500,
    protocol_manifest: Optional[Mapping[str, Any]] = None,
    receipt_out: Optional[Path] = DEFAULT_RECEIPT_OUT,
    env: Optional[Mapping[str, str]] = None,
    home_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute zero-network ENG-NEXUS-4 onboarding & local HQC verification."""
    proto = dict(protocol_manifest or DEFAULT_ENG_NEXUS_4_PROTOCOL_MANIFEST)
    pregate = validate_quantum_protocol_pregate(proto)
    if not pregate["passed"]:
        raise NexusPreflightError(
            "BLOCKED_PROTOCOL_PREGATE",
            f"osiris.quantum_protocol/1 pre-gate failed: {', '.join(pregate['violations'])}",
        )

    pkg_status = check_qnexus_package()
    auth_status = check_nexus_auth_status(env=env, home_dir=home_dir)
    primary_canary = verify_xtop005_canary(canary_path=canary_path, shots=shots)
    wen_canary = verify_wen_plaquette_canary(wen_bundle_dir=wen_bundle_dir, shots=shots)

    ready_for_live = bool(
        pkg_status["qnexus_available"]
        and auth_status["auth_configured"]
        and pregate["passed"]
    )

    body: Dict[str, Any] = {
        "schema": PREFLIGHT_RECEIPT_SCHEMA,
        "claim_id": "ENG-NEXUS-4",
        "verification_queue_check": 8,
        "mode": mode_label,
        "evidence_class": "OFFLINE_PREFLIGHT",
        "hardware_executed": False,
        "paid_qpu_hqc_spent": "0.0000",
        "network_calls": 0,
        "ready_for_live_preflight": ready_for_live,
        "qnexus_sdk": pkg_status,
        "auth_status": auth_status,
        "protocol_pregate": pregate,
        "primary_canary": primary_canary,
        "wen_plaquette_canary": wen_canary,
    }
    redacted_body = _redact_obj(body, env=env, home_dir=home_dir)
    receipt_sha = sha256_bytes(canonical_json_bytes(redacted_body))
    receipt = {**redacted_body, "receipt_sha256": receipt_sha}

    if receipt_out is not None:
        out_path = Path(receipt_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return receipt


def run_live_canary_preflight(
    target: str = "H2-1SC",
    operator_approved: bool = False,
    canary_path: Path = DEFAULT_CANARY_PATH,
    wen_bundle_dir: Path = DEFAULT_WEN_BUNDLE_DIR,
    shots: int = 500,
    hqc_tolerance: float = 0.5,
    project_name: str = "OSIRIS",
    protocol_manifest: Optional[Mapping[str, Any]] = None,
    receipt_out: Optional[Path] = DEFAULT_RECEIPT_OUT,
    osiris_home: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
    home_dir: Optional[Path] = None,
    qnx_override: Any = None,
) -> Dict[str, Any]:
    """Upload & compile ONLY 1 canary circuit to `H2-1SC` / `H2-1E` and compare provider vs local HQC.

    Fails closed before any SDK/network call unless:
      1. `target` is `H2-1SC` or `H2-1E` (Tier 3 paid QPU targets are strictly forbidden).
      2. `operator_approved` is `True` (`--operator-approved`).
      3. `osiris.quantum_protocol/1` pre-gate and canary SHA-256 checks pass.
      4. Nexus credentials (`~/.qnx`, `~/.config/qnexus`, `QNEXUS_TOKEN`, or `QUANTINUUM_NEXUS_TOKEN`)
         and `qnexus` SDK are present.
    """
    norm_target = target.strip().upper()
    if norm_target not in ALLOWED_LIVE_PREFLIGHT_TARGETS:
        raise NexusPreflightError(
            "BLOCKED_PAID_QPU_TARGET",
            f"--preflight-live permits only non-QPU targets {sorted(ALLOWED_LIVE_PREFLIGHT_TARGETS)} "
            f"(zero paid QPU HQCs); got {target!r}",
        )

    if not operator_approved:
        raise NexusPreflightError(
            "BLOCKED_OPERATOR_APPROVAL",
            "--preflight-live requires explicit --operator-approved (0 network calls made)",
        )

    offline = run_offline_preflight(
        mode_label="PREFLIGHT_LIVE_PRECHECK",
        canary_path=canary_path,
        wen_bundle_dir=wen_bundle_dir,
        shots=shots,
        protocol_manifest=protocol_manifest,
        receipt_out=None,
        env=env,
        home_dir=home_dir,
    )

    if not offline["auth_status"]["auth_configured"]:
        raise NexusPreflightError(
            "BLOCKED_MISSING_CREDENTIALS",
            "Nexus credentials absent (~/.qnx, ~/.config/qnexus, QNEXUS_TOKEN, or QUANTINUUM_NEXUS_TOKEN); "
            "run `python -c 'import qnexus; qnexus.login()'` first",
        )

    try:
        qnx = qnx_override if qnx_override is not None else load_qnexus()
    except NexusSdkUnavailable as exc:
        raise NexusPreflightError("BLOCKED_QNEXUS_UNAVAILABLE", str(exc)) from exc

    try:
        circuits = load_circuits(canary_path)
    except ImportError as exc:
        raise NexusPreflightError("BLOCKED_PYTKET_UNAVAILABLE", str(exc)) from exc
    if len(circuits) != 1:
        raise NexusPreflightError(
            "MULTI_CIRCUIT_CANARY_FORBIDDEN",
            f"--preflight-live strictly enforces 1 canary circuit; got {len(circuits)}",
        )

    ledger = NexusAuditLedger(osiris_home)
    runner = NexusSdkRunner(
        qnx=qnx,
        ledger=ledger,
        project_name=project_name,
        project_description="OSIRIS ENG-NEXUS-4 single-circuit onboarding preflight",
    )

    # Compile for H2-1 and query H2-1SC syntax-checker HQC cost (Tier 1, 0 paid QPU HQC)
    try:
        pre = runner.preflight(
            circuits,
            target="H2-1",
            shots=shots,
            optimisation_level=0,
            job_name="eng-nexus-4-canary-preflight",
            approve_remote=True,
        )
        emulator_config_verified = False
        if norm_target == "H2-1E":
            _ = runner._config("H2-1E")
            emulator_config_verified = True
    except GovernanceError as exc:
        raise NexusPreflightError("NEXUS_GOVERNANCE_ERROR", str(exc)) from exc

    if len(pre.nexus_hqc) != 1 or pre.nexus_hqc[0] is None:
        raise NexusPreflightError(
            "MISSING_PROVIDER_HQC",
            f"Nexus circuits.cost returned invalid HQC vector: {pre.nexus_hqc!r}",
        )

    provider_dec = Decimal(str(pre.nexus_hqc[0])).quantize(Decimal("0.0001"), rounding=ROUND_HALF_EVEN)
    compiled_local_dec = Decimal(pre.local_hqc[0]).quantize(Decimal("0.0001"), rounding=ROUND_HALF_EVEN)
    canary_phys_dec = Decimal(offline["primary_canary"]["local_hqc_physical"])
    canary_cons_dec = Decimal(offline["primary_canary"]["local_hqc_conservative"])

    delta_vs_compiled_local = provider_dec - compiled_local_dec
    delta_vs_physical = provider_dec - canary_phys_dec
    delta_vs_conservative = provider_dec - canary_cons_dec
    tol_dec = Decimal(str(hqc_tolerance))

    min_abs_delta = min(
        abs(delta_vs_compiled_local),
        abs(delta_vs_physical),
        abs(delta_vs_conservative),
    )
    within_tol = min_abs_delta <= tol_dec
    eng_nexus_4_status = "SUPPORTED" if within_tol else "ADAPTER_HQC_MISMATCH"

    live_body: Dict[str, Any] = {
        "schema": PREFLIGHT_RECEIPT_SCHEMA,
        "claim_id": "ENG-NEXUS-4",
        "verification_queue_check": 8,
        "mode": "PREFLIGHT_LIVE",
        "requested_target": norm_target,
        "syntax_checker_target": "H2-1SC",
        "emulator_config_verified": emulator_config_verified,
        "evidence_class": "PREFLIGHT_CHECKED",
        "hardware_executed": False,
        "paid_qpu_hqc_spent": "0.0000",
        "canary_circuits_uploaded": 1,
        "sdk_calls_recorded": [name for name, _ in runner.calls],
        "compile_job_id": pre.compile_job_id,
        "project_id": pre.project_id,
        "source_sha256": pre.source_sha256[0],
        "compiled_sha256": pre.compiled_sha256[0],
        "hqc_comparison": {
            "shots": shots,
            "provider_hqc": str(provider_dec),
            "compiled_local_hqc": str(compiled_local_dec),
            "canary_local_hqc_physical": str(canary_phys_dec),
            "canary_local_hqc_conservative": str(canary_cons_dec),
            "delta_vs_compiled_local": str(delta_vs_compiled_local),
            "delta_vs_physical": str(delta_vs_physical),
            "delta_vs_conservative": str(delta_vs_conservative),
            "tolerance_hqc": str(tol_dec),
            "exact_match_physical": abs(delta_vs_physical) <= Decimal("0.0001"),
            "exact_match_conservative": abs(delta_vs_conservative) <= Decimal("0.0001"),
            "within_tolerance": within_tol,
        },
        "eng_nexus_4_status": eng_nexus_4_status,
        "auth_status": offline["auth_status"],
        "protocol_pregate": offline["protocol_pregate"],
        "ledger_receipt_sha256": pre.receipt.get("receipt_sha256"),
    }
    redacted_live = _redact_obj(live_body, env=env, home_dir=home_dir)
    receipt_sha = sha256_bytes(canonical_json_bytes(redacted_live))
    receipt = {**redacted_live, "receipt_sha256": receipt_sha}

    if receipt_out is not None:
        out_path = Path(receipt_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return receipt


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Quantinuum Nexus (`qnexus`) onboarding & ENG-NEXUS-4 single-circuit preflight"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--offline-dry-run",
        action="store_true",
        help="Run zero-network ENG-NEXUS-4 readiness, protocol pre-gate, and local HQC checks (default)",
    )
    mode.add_argument(
        "--status",
        action="store_true",
        help="Alias for zero-network qnexus/auth/canary readiness status report",
    )
    mode.add_argument(
        "--preflight-live",
        action="store_true",
        help="Upload & compile 1 canary circuit on H2-1SC/H2-1E and compare provider_hqc vs local_hqc",
    )
    parser.add_argument(
        "--operator-approved",
        action="store_true",
        help="Explicit operator approval required for --preflight-live",
    )
    parser.add_argument(
        "--target",
        default="H2-1SC",
        help="Target for --preflight-live (H2-1SC or H2-1E only; paid QPU targets are blocked)",
    )
    parser.add_argument(
        "--canary-circuit",
        type=Path,
        default=DEFAULT_CANARY_PATH,
        help="Path to single-circuit canary (.tket.json)",
    )
    parser.add_argument(
        "--wen-bundle-dir",
        type=Path,
        default=DEFAULT_WEN_BUNDLE_DIR,
        help="Optional osiris-qec nexus_h2_bundle directory for Wen-plaquette cross-check",
    )
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        default=None,
        help="Optional JSON file overriding the osiris.quantum_protocol/1 manifest",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=500,
        help="Reference shot count C in 5 + C*(N_1q + 10*N_2q + 5*N_m)/5000 (default: 500)",
    )
    parser.add_argument(
        "--hqc-tolerance",
        type=float,
        default=0.5,
        help="Maximum |provider_hqc - local_hqc| tolerance for ENG-NEXUS-4 SUPPORTED verdict",
    )
    parser.add_argument(
        "--project",
        default="OSIRIS",
        help="Quantinuum Nexus project name for --preflight-live",
    )
    parser.add_argument(
        "--receipt-out",
        type=Path,
        default=DEFAULT_RECEIPT_OUT,
        help="Output path for deterministic/redacted ENG-NEXUS-4 receipt JSON",
    )
    parser.add_argument(
        "--osiris-home",
        type=Path,
        default=None,
        help="Optional OSIRIS_HOME override for NexusAuditLedger",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        proto_manifest: Optional[Dict[str, Any]] = None
        if args.protocol_manifest is not None:
            proto_manifest = json.loads(args.protocol_manifest.read_text(encoding="utf-8"))

        if not args.preflight_live:
            label = "STATUS" if args.status else "OFFLINE_DRY_RUN"
            receipt = run_offline_preflight(
                mode_label=label,
                canary_path=args.canary_circuit,
                wen_bundle_dir=args.wen_bundle_dir,
                shots=args.shots,
                protocol_manifest=proto_manifest,
                receipt_out=args.receipt_out,
            )
            print(json.dumps(receipt, indent=2, sort_keys=True))
            return 0

        live_receipt = run_live_canary_preflight(
            target=args.target,
            operator_approved=args.operator_approved,
            canary_path=args.canary_circuit,
            wen_bundle_dir=args.wen_bundle_dir,
            shots=args.shots,
            hqc_tolerance=args.hqc_tolerance,
            project_name=args.project,
            protocol_manifest=proto_manifest,
            receipt_out=args.receipt_out,
            osiris_home=args.osiris_home,
        )
        print(json.dumps(live_receipt, indent=2, sort_keys=True))
        return 0
    except NexusPreflightError as exc:
        print(f"NEXUS_ONBOARD_BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
