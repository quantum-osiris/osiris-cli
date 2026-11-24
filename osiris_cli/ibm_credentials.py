"""Governed IBM Quantum credential & service loader (`troykey.json` enclave primary).

Resolves IBM Quantum API credentials in strict priority order without ever logging,
printing, or serializing raw API keys or instance CRNs into public artifacts:
  1. `OSIRIS_IBM_KEY_FILE` environment override (if set and readable)
  2. `/home/osiris/.osiris/enclave/troykey.json` (`0600` enclave primary)
  3. `/home/osiris/troykey.json` (legacy fallback path)
  4. `/home/osiris/.osiris/enclave/apikey.json` / `/home/osiris/apikey.json`
  5. `IBM_QUANTUM_TOKEN` environment variable
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ENCLAVE_TROYKEY = Path("/home/osiris/.osiris/enclave/troykey.json")
LEGACY_TROYKEY = Path("/home/osiris/troykey.json")
ENCLAVE_APIKEY = Path("/home/osiris/.osiris/enclave/apikey.json")
LEGACY_APIKEY = Path("/home/osiris/apikey.json")


def load_ibm_token_with_provenance(explicit_path: Optional[Path] = None) -> Tuple[str, Dict[str, Any]]:
    """Return `(token_secret, safe_redacted_provenance_dict)`.

    The returned provenance dictionary contains ONLY the credential label, key-file
    basename, `createdAt` metadata if present, and a salted/unsalted 12-hex-char
    SHA-256 fingerprint of the key file bytes — NEVER the secret token or CRN.
    """
    candidates: list[Tuple[str, Path]] = []
    if explicit_path is not None:
        candidates.append(("explicit_path", Path(explicit_path)))
    env_file = os.environ.get("OSIRIS_IBM_KEY_FILE", "").strip()
    if env_file:
        candidates.append(("OSIRIS_IBM_KEY_FILE", Path(env_file)))
    candidates.extend(
        [
            ("enclave_troykey", ENCLAVE_TROYKEY),
            ("legacy_troykey", LEGACY_TROYKEY),
            ("enclave_apikey", ENCLAVE_APIKEY),
            ("legacy_apikey", LEGACY_APIKEY),
        ]
    )

    for source_label, path in candidates:
        if path.is_file():
            raw = path.read_bytes()
            data = json.loads(raw.decode("utf-8"))
            tok = str(data.get("apikey") or data.get("token") or "").strip()
            if tok:
                file_fp = hashlib.sha256(raw).hexdigest()[:16]
                return tok, {
                    "credential_source": source_label,
                    "credential_basename": path.name,
                    "key_created_at": data.get("createdAt"),
                    "key_file_sha256_prefix16": file_fp,
                }

    env_tok = os.environ.get("IBM_QUANTUM_TOKEN", "").strip()
    if env_tok:
        return env_tok, {
            "credential_source": "IBM_QUANTUM_TOKEN_env",
            "credential_basename": None,
            "key_created_at": None,
            "key_file_sha256_prefix16": hashlib.sha256(env_tok.encode("utf-8")).hexdigest()[:16],
        }

    raise RuntimeError(
        "No IBM Quantum credential found in ~/.osiris/enclave/troykey.json or IBM_QUANTUM_TOKEN"
    )


def connect_ibm_runtime_service(
    instance: str = "open-instance",
    channel: str = "ibm_quantum_platform",
    explicit_path: Optional[Path] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Connect `QiskitRuntimeService` using `troykey.json` and return `(service, safe_provenance)`."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    token, prov = load_ibm_token_with_provenance(explicit_path=explicit_path)
    svc = QiskitRuntimeService(channel=channel, token=token, instance=instance)
    prov_out = dict(prov)
    prov_out["channel"] = channel
    prov_out["instance_alias"] = instance
    return svc, prov_out


def redact_ibm_job_metadata(job: Any) -> Dict[str, Any]:
    """Extract provider-attested IBM job metadata with CRN / account identifiers stripped."""
    metrics = job.metrics() if hasattr(job, "metrics") else {}
    if not isinstance(metrics, dict):
        metrics = {}
    usage = metrics.get("usage")
    safe_usage = None
    if isinstance(usage, dict):
        safe_usage = {
            "qpu_charge_time_seconds": usage.get("qpu_charge_time_seconds") or usage.get("quantum_seconds"),
            "resource_units": usage.get("value"),
            "status": usage.get("status"),
        }
    return {
        "job_id": job.job_id(),
        "backend": job.backend().name if job.backend() else None,
        "status": str(job.status()),
        "creation_date": job.creation_date.isoformat() if getattr(job, "creation_date", None) else None,
        "timestamps": metrics.get("timestamps"),
        "usage": safe_usage,
        "qiskit_version": metrics.get("qiskit_version"),
    }
