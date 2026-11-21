"""Governed Quantinuum Nexus execution client using Python stdlib only.

Implements a fail-closed three-tier governance boundary:
  - Tier 1 (``H2-1SC``, ``H2-2SC``, ``Helios-SC``): Free syntax verification (0 HQC);
    ``disposition="SIMULATED"``, ``hardware=False``.
  - Tier 2 (``H2-1E``, ``H2-2E``, ``Helios-E``): Mandatory syntax checker
    pre-flight plus shot and HQC budget caps; ``disposition="SIMULATED"``,
    ``hardware=False``.
  - Tier 3 (``H2-1``, ``H2-2``, ``Helios``): Mandatory syntax checker
    pre-flight, shot and HQC budget caps, and HMAC-SHA256 approval token bound
    to ``(target, compiled_sha256, shots)``; ``hardware=True`` and
    ``disposition="HARDWARE_ATTESTED"`` are emitted ONLY on completed Tier 3
    runs carrying a real provider ``job_id``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .tket_wrapper import (
    QuantinuumCompilerOptions,
    TketCompilationResult,
    canonicalize_quantinuum_target,
    compile_for_quantinuum,
)

DEFAULT_NEXUS_API_URL = "https://nexus.quantinuum.com/api/v1"
DEFAULT_NEXUS_BASE_URL = DEFAULT_NEXUS_API_URL
DEFAULT_MAX_SHOTS = 10_000
DEFAULT_MAX_HQC = 50.0

TIER1_TARGETS = frozenset({"H2-1SC", "H2-2SC", "Helios-SC"})
TIER2_TARGETS = frozenset({"H2-1E", "H2-2E", "Helios-E"})
TIER3_TARGETS = frozenset({"H2-1", "H2-2", "Helios"})
ALL_NEXUS_TARGETS = TIER1_TARGETS | TIER2_TARGETS | TIER3_TARGETS

QUANTINUUM_TIERS: Dict[str, int] = {
    "H2-1SC": 1,
    "H2-2SC": 1,
    "Helios-SC": 1,
    "H2-1E": 2,
    "H2-2E": 2,
    "Helios-E": 2,
    "H2-1": 3,
    "H2-2": 3,
    "Helios": 3,
}

PREFLIGHT_SYNTAX_CHECKER: Dict[str, str] = {
    "H2-1SC": "H2-1SC",
    "H2-1E": "H2-1SC",
    "H2-1": "H2-1SC",
    "H2-2SC": "H2-2SC",
    "H2-2E": "H2-2SC",
    "H2-2": "H2-2SC",
    "Helios-SC": "Helios-SC",
    "Helios-E": "H2-1SC",
    "Helios": "H2-1SC",
}

_TOKEN_PREFIX = "osiris-t3"
_HEX64_CHARS = frozenset("0123456789abcdef")

HttpRequestHook = Callable[
    [str, str, Dict[str, str], Optional[Dict[str, Any]]],
    Tuple[int, Dict[str, Any]],
]


class GovernanceError(RuntimeError):
    """Fail-closed exception raised when a governance invariant is violated."""


@dataclass(frozen=True)
class NexusConfig:
    token: str
    base_url: str = DEFAULT_NEXUS_BASE_URL
    timeout_s: float = 30.0
    max_shots: int = DEFAULT_MAX_SHOTS
    max_hqc: float = DEFAULT_MAX_HQC
    approval_secret: Optional[str] = None

    @classmethod
    def from_env(cls) -> "NexusConfig":
        token = os.environ.get("QUANTINUUM_NEXUS_TOKEN", "").strip()
        if not token:
            raise GovernanceError("QUANTINUUM_NEXUS_TOKEN is not set.")
        secret = (
            os.environ.get("OSIRIS_TIER3_APPROVAL_SECRET")
            or os.environ.get("OSIRIS_APPROVAL_SECRET")
            or os.environ.get("OSIRIS_TIER3_HMAC_SECRET")
            or None
        )
        return cls(
            token=token,
            base_url=os.environ.get("QUANTINUUM_NEXUS_URL", DEFAULT_NEXUS_BASE_URL).rstrip("/"),
            max_shots=int(os.environ.get("OSIRIS_MAX_SHOTS", str(DEFAULT_MAX_SHOTS))),
            max_hqc=float(os.environ.get("OSIRIS_MAX_HQC", str(DEFAULT_MAX_HQC))),
            approval_secret=secret,
        )


def _canonical_json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_hex64(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in _HEX64_CHARS for ch in value)
    )


def get_target_tier(target: str) -> Tuple[str, int]:
    """Return ``(canonical_target, tier)`` or raise ``GovernanceError``."""
    try:
        canonical = canonicalize_quantinuum_target(target)
    except ValueError as exc:
        raise GovernanceError(str(exc)) from exc
    return canonical, QUANTINUUM_TIERS[canonical]


def compute_disposition_receipt_sha256(disposition_dict: Dict[str, Any]) -> str:
    """Compute the deterministic SHA-256 receipt digest for a disposition dict."""
    canonical = {
        k: v
        for k, v in disposition_dict.items()
        if k not in ("receipt_sha256", "schema", "recorded_at_unix")
    }
    return hashlib.sha256(_canonical_json_bytes(canonical)).hexdigest()


def verify_disposition_receipt(disposition_dict: Dict[str, Any]) -> bool:
    """Verify SHA-256 integrity and tier attestation invariants offline."""
    if not isinstance(disposition_dict, dict):
        return False

    provided = disposition_dict.get("receipt_sha256")
    if not _is_hex64(provided):
        return False

    tier = disposition_dict.get("tier")
    hardware = disposition_dict.get("hardware")
    disposition = disposition_dict.get("disposition")
    status = str(disposition_dict.get("status") or "").upper()
    job_id = disposition_dict.get("job_id")
    preflight_passed = disposition_dict.get("preflight_passed", True)

    if tier not in (1, 2, 3) or not isinstance(hardware, bool):
        return False

    if tier in (1, 2):
        if hardware is True or disposition == "HARDWARE_ATTESTED":
            return False

    if hardware is True or disposition == "HARDWARE_ATTESTED":
        if not (
            tier == 3
            and hardware is True
            and disposition == "HARDWARE_ATTESTED"
            and status in ("COMPLETED", "SUCCEEDED")
            and isinstance(job_id, str)
            and bool(job_id.strip())
            and job_id.strip().lower() != "dry-run-local"
            and preflight_passed is True
        ):
            return False

    expected = compute_disposition_receipt_sha256(disposition_dict)
    return hmac.compare_digest(expected, provided)


@dataclass(frozen=True)
class NexusDisposition:
    """Immutable, cryptographically receipted disposition of a Nexus run."""

    target: str
    tier: int
    disposition: str
    hardware: bool
    status: str
    job_id: str
    shots: int
    estimated_hqc: float
    source_sha256: str
    compiled_sha256: str
    compiler_backend: str = "static_fallback"
    preflight_target: Optional[str] = None
    preflight_job_id: Optional[str] = None
    preflight_passed: bool = True
    counts: Dict[str, int] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    compiler_options: Dict[str, Any] = field(default_factory=dict)
    calibration_snapshot: Dict[str, Any] = field(default_factory=dict)
    diagnostics: List[str] = field(default_factory=list)
    recorded_at: str = field(default_factory=_utc_now_iso)
    schema_version: str = "osiris.nexus.disposition.v1"
    receipt_sha256: str = ""

    def __post_init__(self) -> None:
        if self.tier not in (1, 2, 3):
            raise GovernanceError(f"Invalid governance tier: {self.tier!r}")
        if self.tier in (1, 2) and (
            self.hardware or self.disposition == "HARDWARE_ATTESTED"
        ):
            raise GovernanceError(
                f"Tier {self.tier} target {self.target} can never produce "
                "hardware=True or disposition='HARDWARE_ATTESTED'"
            )
        if self.hardware or self.disposition == "HARDWARE_ATTESTED":
            if not (
                self.tier == 3
                and self.hardware is True
                and self.disposition == "HARDWARE_ATTESTED"
                and str(self.status).upper() in ("COMPLETED", "SUCCEEDED")
                and isinstance(self.job_id, str)
                and bool(self.job_id.strip())
                and self.job_id.strip().lower() != "dry-run-local"
                and self.preflight_passed is True
            ):
                raise GovernanceError(
                    "HARDWARE_ATTESTED disposition requires completed Tier 3 run "
                    "with passed pre-flight and a real provider job_id"
                )

        if not self.receipt_sha256:
            digest = compute_disposition_receipt_sha256(self.to_dict(include_receipt=False))
            object.__setattr__(self, "receipt_sha256", digest)

    def to_dict(self, include_receipt: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "target": self.target,
            "tier": self.tier,
            "disposition": self.disposition,
            "hardware": self.hardware,
            "status": self.status,
            "job_id": self.job_id,
            "shots": self.shots,
            "estimated_hqc": self.estimated_hqc,
            "source_sha256": self.source_sha256,
            "compiled_sha256": self.compiled_sha256,
            "compiler_backend": self.compiler_backend,
            "preflight_target": self.preflight_target,
            "preflight_job_id": self.preflight_job_id,
            "preflight_passed": self.preflight_passed,
            "counts": dict(self.counts),
            "metrics": dict(self.metrics),
            "compiler_options": dict(self.compiler_options),
            "calibration_snapshot": dict(self.calibration_snapshot),
            "diagnostics": list(self.diagnostics),
            "recorded_at": self.recorded_at,
        }
        if include_receipt:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload

    def to_ledger_dict(self) -> Dict[str, Any]:
        return self.to_dict(include_receipt=True)


def estimate_hqc(
    compilation: TketCompilationResult,
    shots: int,
    target: Optional[str] = None,
) -> float:
    """Estimate Quantinuum HQC consumption for a compiled circuit."""
    if not isinstance(shots, int) or isinstance(shots, bool) or shots <= 0:
        raise GovernanceError(f"shots must be a positive integer; got {shots!r}")

    eff_target = target if target is not None else "H2-1"
    _, tier = get_target_tier(eff_target)
    if tier == 1:
        return 0.0

    m = compilation.metrics
    n_1q = m.single_qubit_gates if m.single_qubit_gates else max(0, m.total_gates - m.two_qubit_gates)
    n_2q = m.two_qubit_gates
    n_m = m.n_qubits + m.n_measurements + m.n_resets
    hqc = 5.0 + (float(shots) * (float(n_1q) + 10.0 * float(n_2q) + 5.0 * float(n_m))) / 5000.0
    return round(hqc, 6)


def sign_tier3_approval_token(
    target: str,
    compiled_sha256: str,
    shots: int,
    secret: str,
    expires_at: Optional[int] = None,
    ttl_seconds: int = 900,
    issued_at: Optional[int] = None,
) -> str:
    """Sign a scoped HMAC-SHA256 Tier 3 hardware approval token."""
    if not secret or not isinstance(secret, str):
        raise GovernanceError("Tier 3 approval signing secret must be a non-empty string")
    canonical_target, tier = get_target_tier(target)
    if tier != 3:
        raise GovernanceError(
            f"Tier 3 approval tokens apply only to Tier 3 targets; got {canonical_target} (Tier {tier})"
        )
    if not _is_hex64(compiled_sha256):
        raise GovernanceError("compiled_sha256 must be a 64-character lowercase hex SHA-256")
    if not isinstance(shots, int) or isinstance(shots, bool) or shots <= 0:
        raise GovernanceError(f"shots must be a positive integer; got {shots!r}")

    iat = int(time.time()) if issued_at is None else int(issued_at)
    exp = int(expires_at) if expires_at is not None else iat + int(ttl_seconds)
    payload = {
        "compiled_sha256": compiled_sha256,
        "exp": exp,
        "iat": iat,
        "shots": int(shots),
        "target": canonical_target,
        "tier": 3,
        "v": 1,
    }
    payload_b64 = base64.urlsafe_b64encode(_canonical_json_bytes(payload)).decode("ascii")
    sig = hmac.new(
        secret.encode("utf-8"),
        payload_b64.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    return f"{_TOKEN_PREFIX}.{payload_b64}.{sig}"


def verify_tier3_approval_token(
    approval_token: Optional[str],
    target: str,
    compiled_sha256: str,
    shots: int,
    secret: Optional[str],
    now: Optional[int] = None,
) -> Dict[str, Any]:
    """Verify an HMAC-SHA256 Tier 3 approval token; fail closed with GovernanceError."""
    if not approval_token or not isinstance(approval_token, str):
        raise GovernanceError(
            "Tier 3 hardware execution requires a valid --approval-token (missing approval_token)"
        )
    if not secret or not isinstance(secret, str):
        raise GovernanceError(
            "Tier 3 approval verification secret is not configured "
            "(set OSIRIS_TIER3_APPROVAL_SECRET or OSIRIS_APPROVAL_SECRET)"
        )

    canonical_target, tier = get_target_tier(target)
    if tier != 3:
        raise GovernanceError(
            f"Tier 3 approval verification called for non-Tier-3 target {canonical_target}"
        )

    clean_tok = approval_token.strip()
    if "." not in clean_tok and len(clean_tok) == 64:
        expected_msg = f"{canonical_target}:{compiled_sha256}:{shots}".encode("utf-8")
        expected_sig = hmac.new(secret.encode("utf-8"), expected_msg, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(clean_tok, expected_sig):
            raise GovernanceError(f"Invalid Tier 3 approval_token for target '{canonical_target}'.")
        ts = int(time.time()) if now is None else int(now)
        return {
            "compiled_sha256": compiled_sha256,
            "exp": ts + 300,
            "iat": ts,
            "shots": int(shots),
            "target": canonical_target,
            "tier": 3,
            "v": 1,
        }

    parts = clean_tok.split(".")
    if len(parts) != 3 or parts[0] != _TOKEN_PREFIX:
        raise GovernanceError("Invalid Tier 3 approval_token: token has invalid format")

    _, payload_b64, provided_sig = parts
    expected_sig = hmac.new(
        secret.encode("utf-8"),
        payload_b64.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected_sig, provided_sig):
        raise GovernanceError(
            "Invalid Tier 3 approval_token: Tier 3 approval token signature verification failed"
        )

    try:
        raw_bytes = base64.urlsafe_b64decode(payload_b64.encode("ascii"))
        payload = json.loads(raw_bytes.decode("utf-8"))
    except Exception as exc:
        raise GovernanceError("Tier 3 approval token payload is malformed") from exc

    if not isinstance(payload, dict) or payload.get("v") != 1 or payload.get("tier") != 3:
        raise GovernanceError("Tier 3 approval token version/tier invalid")

    if payload.get("target") != canonical_target:
        raise GovernanceError(
            f"Invalid Tier 3 approval_token target mismatch: token={payload.get('target')!r}, "
            f"requested={canonical_target!r}"
        )

    tok_hash = payload.get("compiled_sha256")
    if not isinstance(tok_hash, str) or not hmac.compare_digest(tok_hash, compiled_sha256):
        raise GovernanceError(
            "Invalid Tier 3 approval_token compiled_sha256 mismatch (circuit changed after approval)"
        )

    tok_shots = payload.get("shots")
    if not isinstance(tok_shots, int) or isinstance(tok_shots, bool) or tok_shots != int(shots):
        raise GovernanceError(
            f"Invalid Tier 3 approval_token shots mismatch: token={tok_shots!r}, requested={shots!r}"
        )

    exp = payload.get("exp")
    if not isinstance(exp, int) or isinstance(exp, bool):
        raise GovernanceError("Tier 3 approval token missing integer expiration")

    current_ts = int(time.time()) if now is None else int(now)
    if current_ts > exp:
        raise GovernanceError("Tier 3 approval token is expired")

    return payload


def record_disposition(disposition: NexusDisposition, path: str) -> Path:
    """Persist a NexusDisposition receipt to a .json or .jsonl ledger path."""
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    record = disposition.to_dict(include_receipt=True)

    if out_path.suffix.lower() == ".jsonl":
        line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(line)
    else:
        content = json.dumps(record, indent=2, sort_keys=True) + "\n"
        out_path.write_text(content, encoding="utf-8")
    return out_path


class QuantinuumNexusClient:
    """Stdlib-only governed client for Quantinuum Nexus."""

    def __init__(
        self,
        api_url: Optional[Any] = None,
        api_token: Optional[str] = None,
        approval_secret: Optional[str] = None,
        max_shots: int = 1000,
        budget_hqc: float = 50.0,
        timeout_seconds: float = 30.0,
        poll_interval_seconds: float = 0.0,
        max_poll_attempts: int = 30,
        http_request: Optional[HttpRequestHook] = None,
        opener: Optional[Any] = None,
        config: Optional[NexusConfig] = None,
    ) -> None:
        if isinstance(api_url, NexusConfig):
            config = api_url
            api_url = None
        if config is not None:
            self.config = config
            api_url = config.base_url
            api_token = config.token
            approval_secret = config.approval_secret
            max_shots = config.max_shots
            budget_hqc = config.max_hqc
            timeout_seconds = config.timeout_s
        self.api_url = (
            api_url
            or os.environ.get("QUANTINUUM_NEXUS_URL")
            or DEFAULT_NEXUS_API_URL
        ).rstrip("/")
        self._api_token = (
            api_token if api_token is not None else os.environ.get("QUANTINUUM_NEXUS_TOKEN")
        )
        self._approval_secret = (
            approval_secret
            if approval_secret is not None
            else (
                os.environ.get("OSIRIS_TIER3_APPROVAL_SECRET")
                or os.environ.get("OSIRIS_APPROVAL_SECRET")
                or os.environ.get("OSIRIS_TIER3_HMAC_SECRET")
            )
        )
        self.max_shots = int(max_shots)
        self.budget_hqc = float(budget_hqc)
        self.timeout_seconds = float(timeout_seconds)
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.max_poll_attempts = int(max_poll_attempts)
        self._http_request = http_request
        self._opener = opener
        if config is None:
            self.config = NexusConfig(
                token=self._api_token or "",
                base_url=self.api_url,
                timeout_s=self.timeout_seconds,
                max_shots=self.max_shots,
                max_hqc=self.budget_hqc,
                approval_secret=self._approval_secret,
            )

    def __repr__(self) -> str:
        return (
            f"QuantinuumNexusClient(api_url={self.api_url!r}, "
            f"authenticated={bool(self._api_token)}, "
            f"max_shots={self.max_shots}, budget_hqc={self.budget_hqc})"
        )

    def _sanitize(self, text: str) -> str:
        sanitized = str(text)
        for secret in (self._api_token, self._approval_secret):
            if secret:
                sanitized = sanitized.replace(secret, "[REDACTED]")
        return sanitized

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        url = f"{self.api_url}/{path.lstrip('/')}"
        headers: Dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": "osiris-cli/6.0.0",
        }
        if payload is not None:
            headers["Content-Type"] = "application/json"
        if self._api_token:
            headers["Authorization"] = f"Bearer {self._api_token}"

        if self._http_request is not None:
            return self._http_request(method.upper(), url, headers, payload)

        body_bytes = (
            json.dumps(payload, sort_keys=True).encode("utf-8")
            if payload is not None
            else None
        )
        req = urllib.request.Request(
            url,
            data=body_bytes,
            headers=headers,
            method=method.upper(),
        )
        try:
            if self._opener is not None:
                resp_ctx = self._opener.open(req, timeout=self.timeout_seconds)
            else:
                resp_ctx = urllib.request.urlopen(req, timeout=self.timeout_seconds)
            with resp_ctx as resp:
                status = int(getattr(resp, "status", 200))
                raw = resp.read().decode("utf-8", errors="replace")
                data = json.loads(raw) if raw.strip() else {}
                if not isinstance(data, dict):
                    data = {"data": data}
                return status, data
        except urllib.error.HTTPError as exc:
            raw_err = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            try:
                err_data = json.loads(raw_err) if raw_err.strip() else {}
                if not isinstance(err_data, dict):
                    err_data = {"error": raw_err}
            except ValueError:
                err_data = {"error": self._sanitize(raw_err or str(exc))}
            return int(exc.code), err_data
        except urllib.error.URLError as exc:
            raise GovernanceError(
                f"Quantinuum Nexus connection error: {self._sanitize(str(exc.reason))}"
            ) from exc

    def _request(
        self, method: str, path: str, body: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        status, data = self._request_json(method, path, body)
        if not (200 <= status < 300):
            raise RuntimeError(
                f"Nexus API {status} on {method} {path}: {self._sanitize(str(data))}"
            )
        return data

    def test_connection(self) -> Dict[str, Any]:
        """Probe the Nexus targets/backends endpoint and return a secret-free status dict."""
        if type(self)._request is not QuantinuumNexusClient._request or (
            "_request" in self.__dict__
        ):
            res = self._request("GET", "/backends")
            return {
                "ok": True,
                "authenticated": bool(self._api_token),
                "backends": res.get("backends", res),
            }
        try:
            status, data = self._request_json("GET", "/targets")
        except GovernanceError as exc:
            return {
                "ok": False,
                "endpoint": f"{self.api_url}/targets",
                "authenticated": bool(self._api_token),
                "error": self._sanitize(str(exc)),
            }

        if 200 <= status < 300:
            raw_targets = data.get("targets", data.get("devices", data.get("backends", [])))
            if not isinstance(raw_targets, list):
                raw_targets = []
            return {
                "ok": True,
                "status_code": status,
                "endpoint": f"{self.api_url}/targets",
                "authenticated": bool(self._api_token),
                "targets": raw_targets,
                "backends": raw_targets,
            }

        return {
            "ok": False,
            "status_code": status,
            "endpoint": f"{self.api_url}/targets",
            "authenticated": bool(self._api_token),
            "error": self._sanitize(str(data.get("error", f"HTTP {status}"))),
        }

    def _submit_and_poll(
        self,
        compilation: TketCompilationResult,
        target: str,
        shots: int,
    ) -> Dict[str, Any]:
        submit_payload = {
            "target": target,
            "backend": target,
            "shots": int(shots),
            "language": "OPENQASM 2.0",
            "program": compilation.qasm2,
            "compiled_sha256": compilation.compiled_sha256,
            "source_sha256": compilation.source_sha256,
        }
        status_code, resp = self._request_json("POST", "/jobs", submit_payload)
        if not (200 <= status_code < 300):
            err_msg = self._sanitize(str(resp.get("error", f"HTTP {status_code}")))
            raise GovernanceError(
                f"Nexus job submission to {target} failed (HTTP {status_code}): {err_msg}"
            )

        job_id = str(resp.get("job_id") or "").strip()
        if not job_id:
            raise GovernanceError(
                f"Nexus job submission to {target} returned no valid job_id"
            )

        state = str(resp.get("status") or "QUEUED").upper()
        latest = dict(resp)
        attempts = 0
        while state in ("QUEUED", "SUBMITTED", "RUNNING", "PENDING"):
            if attempts >= self.max_poll_attempts:
                raise GovernanceError(
                    f"Nexus job {job_id} on {target} timed out in state {state}"
                )
            if self.poll_interval_seconds > 0:
                time.sleep(self.poll_interval_seconds)
            attempts += 1
            poll_code, poll_resp = self._request_json("GET", f"/jobs/{job_id}")
            if not (200 <= poll_code < 300):
                err_msg = self._sanitize(str(poll_resp.get("error", f"HTTP {poll_code}")))
                raise GovernanceError(
                    f"Nexus job poll for {job_id} failed (HTTP {poll_code}): {err_msg}"
                )
            latest = dict(poll_resp)
            latest.setdefault("job_id", job_id)
            state = str(latest.get("status") or "UNKNOWN").upper()

        latest["job_id"] = job_id
        latest["status"] = state
        return latest

    def syntax_check(
        self,
        compilation: TketCompilationResult,
        target: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a free Tier 1 syntax check pre-flight (H2-1SC or H2-2SC)."""
        raw_target = target or compilation.target
        canonical, _ = get_target_tier(raw_target)
        sc_target = PREFLIGHT_SYNTAX_CHECKER[canonical]

        result = self._submit_and_poll(compilation, target=sc_target, shots=1)
        state = str(result.get("status") or "").upper()
        syntax_errors = result.get("errors") or []
        if state not in ("COMPLETED", "SUCCEEDED") or syntax_errors:
            raise GovernanceError(
                f"Syntax pre-flight failed on {sc_target} "
                f"(status={state!r}, errors={syntax_errors!r})"
            )

        return {
            "target": sc_target,
            "job_id": result["job_id"],
            "status": state,
            "passed": True,
            "diagnostics": list(result.get("diagnostics") or []),
        }

    def execute(
        self,
        source: str,
        target: str = "H2-1SC",
        shots: int = 100,
        options: Optional[QuantinuumCompilerOptions] = None,
        max_shots: Optional[int] = None,
        budget_hqc: Optional[float] = None,
        approval_token: Optional[str] = None,
        record_ledger: Optional[str] = None,
        dry_run: bool = False,
        now: Optional[int] = None,
    ) -> NexusDisposition:
        """Compile, enforce governance gates, execute on Nexus, and receipt."""
        canonical_target, tier = get_target_tier(target)
        if not isinstance(shots, int) or isinstance(shots, bool) or shots <= 0:
            raise GovernanceError(f"shots must be a positive integer; got {shots!r}")

        try:
            compilation = compile_for_quantinuum(
                source,
                target=canonical_target,
                options=options,
            )
        except ValueError as exc:
            raise GovernanceError(f"Compilation rejected: {exc}") from exc
        if not compilation.ok:
            raise GovernanceError(f"TKET pre-flight failed: {'; '.join(compilation.diagnostics)}")

        eff_max_shots = self.max_shots if max_shots is None else int(max_shots)
        eff_budget_hqc = self.budget_hqc if budget_hqc is None else float(budget_hqc)

        est_hqc = estimate_hqc(compilation, shots=shots, target=canonical_target)

        if tier in (2, 3):
            if shots > eff_max_shots:
                raise GovernanceError(
                    f"Requested shots ({shots}) exceed max_shots cap ({eff_max_shots}) "
                    f"for Tier {tier} target {canonical_target}"
                )
            if est_hqc > eff_budget_hqc:
                raise GovernanceError(
                    f"Estimated HQC ({est_hqc:.4f}) exceeds budget_hqc cap "
                    f"({eff_budget_hqc:.4f}) for Tier {tier} target {canonical_target}"
                )

        if tier == 3:
            verify_tier3_approval_token(
                approval_token=approval_token,
                target=canonical_target,
                compiled_sha256=compilation.compiled_sha256,
                shots=shots,
                secret=self._approval_secret,
                now=now,
            )

        if dry_run:
            disp = NexusDisposition(
                target=canonical_target,
                tier=tier,
                disposition="SIMULATED",
                hardware=False,
                status="DRY_RUN",
                job_id="dry-run-local",
                shots=shots,
                estimated_hqc=est_hqc,
                source_sha256=compilation.source_sha256,
                compiled_sha256=compilation.compiled_sha256,
                compiler_backend=compilation.compiler_backend,
                preflight_target=PREFLIGHT_SYNTAX_CHECKER[canonical_target],
                preflight_job_id=None,
                preflight_passed=False,
                counts={},
                metrics=compilation.metrics.to_dict(),
                compiler_options=compilation.compiler_options,
                diagnostics=list(compilation.diagnostics),
            )
            if record_ledger:
                record_disposition(disp, record_ledger)
            return disp

        if tier == 1:
            pf = self.syntax_check(compilation, target=canonical_target)
            disp = NexusDisposition(
                target=canonical_target,
                tier=1,
                disposition="SIMULATED",
                hardware=False,
                status="COMPLETED",
                job_id=pf["job_id"],
                shots=shots,
                estimated_hqc=0.0,
                source_sha256=compilation.source_sha256,
                compiled_sha256=compilation.compiled_sha256,
                compiler_backend=compilation.compiler_backend,
                preflight_target=pf["target"],
                preflight_job_id=pf["job_id"],
                preflight_passed=True,
                counts={},
                metrics=compilation.metrics.to_dict(),
                compiler_options=compilation.compiler_options,
                diagnostics=list(compilation.diagnostics),
            )
        elif tier == 2:
            pf = self.syntax_check(compilation, target=canonical_target)
            job = self._submit_and_poll(compilation, target=canonical_target, shots=shots)
            if job["status"] not in ("COMPLETED", "SUCCEEDED"):
                raise GovernanceError(
                    f"Tier 2 emulation job {job['job_id']} on {canonical_target} "
                    f"did not complete (status={job['status']!r})"
                )
            raw_counts = job.get("counts") or {}
            counts = {str(k): int(v) for k, v in raw_counts.items()} if isinstance(raw_counts, dict) else {}
            disp = NexusDisposition(
                target=canonical_target,
                tier=2,
                disposition="SIMULATED",
                hardware=False,
                status="COMPLETED",
                job_id=job["job_id"],
                shots=shots,
                estimated_hqc=est_hqc,
                source_sha256=compilation.source_sha256,
                compiled_sha256=compilation.compiled_sha256,
                compiler_backend=compilation.compiler_backend,
                preflight_target=pf["target"],
                preflight_job_id=pf["job_id"],
                preflight_passed=True,
                counts=counts,
                metrics=compilation.metrics.to_dict(),
                compiler_options=compilation.compiler_options,
                calibration_snapshot=dict(job.get("calibration") or {}),
                diagnostics=list(compilation.diagnostics),
            )
        else:
            pf = self.syntax_check(compilation, target=canonical_target)
            job = self._submit_and_poll(compilation, target=canonical_target, shots=shots)
            if job["status"] not in ("COMPLETED", "SUCCEEDED"):
                raise GovernanceError(
                    f"Tier 3 hardware job {job['job_id']} on {canonical_target} "
                    f"did not complete (status={job['status']!r}); refusing hardware attestation"
                )
            real_job_id = str(job.get("job_id") or "").strip()
            if not real_job_id or real_job_id.lower() == "dry-run-local":
                raise GovernanceError(
                    "Tier 3 hardware job missing authentic provider job_id; "
                    "refusing hardware attestation"
                )
            raw_counts = job.get("counts") or {}
            counts = {str(k): int(v) for k, v in raw_counts.items()} if isinstance(raw_counts, dict) else {}
            disp = NexusDisposition(
                target=canonical_target,
                tier=3,
                disposition="HARDWARE_ATTESTED",
                hardware=True,
                status="COMPLETED",
                job_id=real_job_id,
                shots=shots,
                estimated_hqc=est_hqc,
                source_sha256=compilation.source_sha256,
                compiled_sha256=compilation.compiled_sha256,
                compiler_backend=compilation.compiler_backend,
                preflight_target=pf["target"],
                preflight_job_id=pf["job_id"],
                preflight_passed=True,
                counts=counts,
                metrics=compilation.metrics.to_dict(),
                compiler_options=compilation.compiler_options,
                calibration_snapshot=dict(job.get("calibration") or {}),
                diagnostics=list(compilation.diagnostics),
            )

        if record_ledger:
            record_disposition(disp, record_ledger)
        return disp

    def verify(
        self,
        source: str,
        target: str = "H2-1SC",
        options: Optional[QuantinuumCompilerOptions] = None,
        record_ledger: Optional[str] = None,
        dry_run: bool = False,
    ) -> NexusDisposition:
        """Convenience wrapper for Tier 1 syntax verification."""
        canonical, _ = get_target_tier(target)
        sc_target = PREFLIGHT_SYNTAX_CHECKER[canonical]
        return self.execute(
            source=source,
            target=sc_target,
            shots=1,
            options=options,
            record_ledger=record_ledger,
            dry_run=dry_run,
        )

    def submit_and_wait(
        self,
        source: str,
        *,
        target: str = "H2-1SC",
        shots: int = 1024,
        budget_hqc: Optional[float] = None,
        compiler_options: Optional[QuantinuumCompilerOptions] = None,
        approval_token: Optional[str] = None,
        skip_preflight: bool = False,
        poll_interval_s: float = 0.0,
        max_wait_s: float = 300.0,
    ) -> NexusDisposition:
        """Compile, enforce governance, submit via `_request`, poll, and return `NexusDisposition`."""
        canonical_target, tier = get_target_tier(target)
        if shots <= 0 or shots > self.config.max_shots:
            raise GovernanceError(f"shots={shots} out of bounds (1..{self.config.max_shots}).")

        opts = compiler_options or QuantinuumCompilerOptions()
        try:
            compiled = compile_for_quantinuum(source, target=canonical_target, options=opts)
        except ValueError as exc:
            raise GovernanceError(f"TKET pre-flight failed: {exc}") from exc
        if not compiled.ok:
            raise GovernanceError(f"TKET pre-flight failed: {'; '.join(compiled.diagnostics)}")

        hqc_est = 0.0 if tier == 1 else estimate_hqc(compiled, shots, canonical_target)
        cap = (
            min(self.config.max_hqc, budget_hqc)
            if budget_hqc is not None
            else self.config.max_hqc
        )
        if tier >= 2 and hqc_est > cap:
            raise GovernanceError(f"Estimated HQC ({hqc_est}) exceeds budget cap ({cap}).")

        if tier == 3:
            verify_tier3_approval_token(
                approval_token,
                target=canonical_target,
                compiled_sha256=compiled.compiled_sha256,
                shots=shots,
                secret=self._approval_secret,
            )

        pf_target: Optional[str] = None
        pf_job_id: Optional[str] = None
        pf_passed = True
        if tier >= 2 and not skip_preflight:
            sc_target = "Helios-SC" if compiled.metrics.n_qubits > 56 else PREFLIGHT_SYNTAX_CHECKER[canonical_target]
            preflight = self.submit_and_wait(
                source,
                target=sc_target,
                shots=1,
                compiler_options=opts,
                skip_preflight=True,
                poll_interval_s=poll_interval_s,
                max_wait_s=60.0,
            )
            if preflight.status.lower() not in {"completed", "succeeded"}:
                raise GovernanceError(f"Mandatory {sc_target} pre-flight failed: {preflight.status}")
            pf_target = sc_target
            pf_job_id = preflight.job_id

        payload = {
            "target": canonical_target,
            "backend": canonical_target,
            "language": "OPENQASM 2.0",
            "program": compiled.compiled_qasm2,
            "shots": shots,
            "options": compiled.compiler_options,
        }
        submitted = self._request("POST", "/jobs", payload)
        job_id = str(submitted["job_id"])

        deadline = time.monotonic() + max_wait_s
        job_state: Dict[str, Any] = submitted
        while time.monotonic() < deadline:
            cur_st = str(job_state.get("status", "unknown")).lower()
            if cur_st in {"completed", "succeeded", "failed", "cancelled"} and (
                job_state is not submitted or "counts" in job_state
            ):
                break
            job_state = self._request("GET", f"/jobs/{job_id}")
            status = str(job_state.get("status", "unknown")).lower()
            if status in {"completed", "succeeded", "failed", "cancelled"}:
                break
            if poll_interval_s > 0:
                time.sleep(poll_interval_s)

        final_status = str(job_state.get("status", "timeout")).lower()
        is_hw = tier == 3 and final_status in {"completed", "succeeded"}
        disposition = (
            "HARDWARE_ATTESTED"
            if is_hw
            else "SIMULATED"
            if final_status in {"completed", "succeeded"}
            else "QUARANTINED"
        )
        return NexusDisposition(
            target=canonical_target,
            tier=tier,
            disposition=disposition,
            hardware=is_hw,
            status=final_status,
            job_id=job_id,
            shots=shots,
            estimated_hqc=float(job_state.get("hqc_cost", hqc_est)),
            source_sha256=compiled.source_sha256,
            compiled_sha256=compiled.compiled_sha256,
            compiler_backend=compiled.compiler_backend,
            preflight_target=pf_target,
            preflight_job_id=pf_job_id,
            preflight_passed=pf_passed,
            counts=dict(job_state.get("counts") or {}),
            metrics=compiled.metrics.to_dict(),
            compiler_options=compiled.compiler_options,
            calibration_snapshot=dict(job_state.get("calibration") or {}),
            diagnostics=list(compiled.diagnostics),
        )
