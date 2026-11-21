"""Local cryptographic audit ledger & verification receipt uploader for `NexusDisposition`.

Persists `NexusDisposition` records into an append-only `exec-canon-1` SHA-256 hash chain
at `$OSIRIS_HOME/quantum/nexus_dispositions.jsonl` (`0o600`, zero JSON floats) and supports
deterministic offline verification plus opt-in upload to Quantum Agent Lab
(`https://www.quantumagentlab.com/verify/<receipt_sha256>`).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .exec_kernel import ExecJournal
from .providers.quantinuum_nexus import GovernanceError, NexusDisposition
from .quantinuum_gate import _verify_canon_chain_file
from .quantum_readiness import _canon_safe, default_osiris_home

DISPOSITION_LEDGER_SCHEMA = "osiris.nexus_disposition_entry/1"
DEFAULT_VERIFY_PORTAL_BASE = "https://www.quantumagentlab.com"
PathArg = Optional[Union[os.PathLike, str]]


def default_nexus_dispositions_path(osiris_home: PathArg = None) -> Path:
    if osiris_home is None:
        override = os.environ.get("OSIRIS_NEXUS_DISPOSITIONS_LOG")
        if override:
            return Path(override).expanduser().resolve()
    return default_osiris_home(osiris_home) / "quantum" / "nexus_dispositions.jsonl"


class NexusAuditLedger:
    """Append-only `exec-canon-1` hash-chained ledger for `NexusDisposition` receipts."""

    def __init__(self, osiris_home: PathArg = None) -> None:
        self.osiris_home = default_osiris_home(osiris_home)
        self.path = default_nexus_dispositions_path(self.osiris_home)

    def append_disposition(self, disposition: NexusDisposition) -> Dict[str, Any]:
        raw = disposition.to_ledger_dict()
        if raw["disposition"] == "HARDWARE_ATTESTED" and (
            raw["tier"] != 3 or raw["hardware"] is not True
        ):
            raise GovernanceError(
                "Refusing to persist HARDWARE_ATTESTED disposition unless tier==3 and hardware==True."
            )
        if raw["tier"] in (1, 2) and (
            raw["hardware"] is True or raw["disposition"] == "HARDWARE_ATTESTED"
        ):
            raise GovernanceError(
                "Tier 1/2 simulated disposition cannot record hardware=True or HARDWARE_ATTESTED."
            )

        canon_body: Dict[str, Any] = _canon_safe(raw)
        canon_body["schema"] = DISPOSITION_LEDGER_SCHEMA
        canon_body["recorded_at_unix"] = int(time.time())

        journal = ExecJournal(self.path)
        chain_hash = journal.append(canon_body)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass
        return {
            "receipt_sha256": raw["receipt_sha256"],
            "chain_hash": chain_hash,
            "ledger_path": str(self.path),
            "verify_url": f"{DEFAULT_VERIFY_PORTAL_BASE}/verify/{raw['receipt_sha256']}",
        }

    def list_entries(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        entries: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                s = line.strip()
                if s:
                    entries.append(json.loads(s))
        return entries

    def find_receipt(self, identifier: str) -> Dict[str, Any]:
        """Locate a persisted receipt body by `receipt_sha256` prefix, `chain_hash`, or `job_id`."""
        needle = identifier.strip()
        if not needle:
            raise ValueError("Empty receipt identifier")
        for rec in reversed(self.list_entries()):
            body = rec.get("body", {})
            r_sha = str(body.get("receipt_sha256", ""))
            c_hash = str(rec.get("hash", ""))
            job_id = str(body.get("job_id", ""))
            if (
                r_sha == needle
                or (len(needle) >= 8 and r_sha.startswith(needle))
                or c_hash == needle
                or (len(needle) >= 8 and c_hash.startswith(needle))
                or job_id == needle
            ):
                return dict(body)
        raise KeyError(f"No Nexus disposition receipt matching {identifier!r} in {self.path}")

    def verify(self) -> Dict[str, Any]:
        return _verify_canon_chain_file(self.path, check_quantum_integrity=True)


def compute_receipt_sha256(receipt_payload: Dict[str, Any]) -> str:
    """Recompute the canonical `receipt_sha256` over a ledger payload (excluding metadata keys)."""
    core_keys = (
        "calibration_snapshot",
        "compiled_sha256",
        "compiler_options",
        "counts",
        "diagnostics",
        "disposition",
        "estimated_hqc",
        "hardware",
        "job_id",
        "shots",
        "source_sha256",
        "status",
        "target",
        "tier",
    )
    normalized: Dict[str, Any] = {}
    for k in core_keys:
        if k not in receipt_payload:
            raise ValueError(f"Receipt payload missing required field {k!r}")
        val = receipt_payload[k]
        if k == "estimated_hqc":
            val = float(val)
        normalized[k] = val
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def upload_disposition_receipt(
    receipt: Union[NexusDisposition, Dict[str, Any]],
    *,
    portal_base_url: Optional[str] = None,
    api_token: Optional[str] = None,
    timeout_s: float = 15.0,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Verify and upload a `NexusDisposition` receipt to Quantum Agent Lab (`/api/verify`)."""
    payload = (
        receipt.to_ledger_dict() if isinstance(receipt, NexusDisposition) else dict(receipt)
    )
    recomputed_sha = compute_receipt_sha256(payload)
    recorded_sha = str(payload.get("receipt_sha256") or recomputed_sha)
    if recorded_sha != recomputed_sha:
        raise GovernanceError(
            f"Receipt SHA-256 integrity mismatch before upload "
            f"(recorded={recorded_sha}, recomputed={recomputed_sha})."
        )
    payload["receipt_sha256"] = recomputed_sha

    tier = int(payload["tier"])
    hw = bool(payload["hardware"])
    disp = str(payload["disposition"])
    if disp == "HARDWARE_ATTESTED" and (tier != 3 or not hw):
        raise GovernanceError("Refusing to upload spoofed HARDWARE_ATTESTED receipt.")

    base = (
        portal_base_url
        or os.environ.get("QUANTUM_AGENT_LAB_URL")
        or DEFAULT_VERIFY_PORTAL_BASE
    ).rstrip("/")
    verify_url = f"{base}/verify/{recomputed_sha}"
    endpoint = f"{base}/api/verify"

    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "receipt_sha256": recomputed_sha,
            "endpoint": endpoint,
            "verify_url": verify_url,
            "payload": payload,
        }

    token = (api_token or os.environ.get("QUANTUM_AGENT_LAB_TOKEN") or "").strip()
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "osiris-cli/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, sort_keys=True).encode("utf-8"),
        method="POST",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            remote = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:512]
        raise RuntimeError(f"Receipt upload HTTP {exc.code} on {endpoint}: {detail}") from exc

    return {
        "ok": True,
        "dry_run": False,
        "receipt_sha256": recomputed_sha,
        "verify_url": str(remote.get("verify_url", verify_url)),
        "remote_response": remote,
    }
