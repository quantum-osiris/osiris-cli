"""Governed Quantinuum Nexus submission client for osiris-cli.

Enforces Tier 1–3 governance:
- Tier 1 (H2-1SC, H2-2SC, Helios-SC): Free syntax check -> SIMULATED
- Tier 2 (H2-1E, H2-2E, Helios-E): Emulation with HQC/shot cap -> SIMULATED
- Tier 3 (H2-1, H2-2, Helios): Live QPU requiring signed approval_token + pre-flight -> HARDWARE_ATTESTED
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

from .tket_wrapper import (
    QuantinuumCompilerOptions,
    QuantinuumTarget,
    TketCompilationResult,
    compile_for_quantinuum,
)

TIER1_TARGETS = frozenset({"H2-1SC", "H2-2SC", "Helios-SC"})
TIER2_TARGETS = frozenset({"H2-1E", "H2-2E", "Helios-E"})
TIER3_TARGETS = frozenset({"H2-1", "H2-2", "Helios"})
ALL_NEXUS_TARGETS = TIER1_TARGETS | TIER2_TARGETS | TIER3_TARGETS

DEFAULT_NEXUS_BASE_URL = "https://nexus.quantinuum.com/api/v1"
DEFAULT_MAX_SHOTS = 10_000
DEFAULT_MAX_HQC = 50.0


class GovernanceError(RuntimeError):
    """Raised when a job violates OSIRIS Tier 1–3 governance or budget rules."""


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
            os.environ.get("OSIRIS_APPROVAL_SECRET")
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


@dataclass(frozen=True)
class NexusDisposition:
    """Immutable receipt persisted to the OSIRIS evidence ledger."""

    job_id: str
    target: QuantinuumTarget
    tier: Literal[1, 2, 3]
    status: str
    hardware: bool
    disposition: Literal["SIMULATED", "HARDWARE_ATTESTED", "QUARANTINED"]
    shots: int
    estimated_hqc: float
    source_sha256: str
    compiled_sha256: str
    compiler_options: Dict[str, Any]
    counts: Dict[str, int] = field(default_factory=dict)
    calibration_snapshot: Dict[str, Any] = field(default_factory=dict)
    diagnostics: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.disposition == "HARDWARE_ATTESTED" and (self.tier != 3 or not self.hardware):
            raise GovernanceError(
                "Unspoofable integrity violation: HARDWARE_ATTESTED requires tier=3 and hardware=True."
            )
        if self.hardware and (self.tier != 3 or self.disposition != "HARDWARE_ATTESTED"):
            raise GovernanceError(
                "Unspoofable integrity violation: hardware=True requires tier=3 and HARDWARE_ATTESTED."
            )

    def to_ledger_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
        return payload


def estimate_hqc(compilation: TketCompilationResult, shots: int) -> float:
    """Deterministic HQC upper-bound estimate: 5 + C * (N_1q + 10*N_2q + 5*N_m) / 5000."""
    m = compilation.metrics
    n_1q = max(0, m.total_gates - m.two_qubit_gates)
    n_2q = m.two_qubit_gates
    n_m = max(1, m.n_cbits)
    return round(5.0 + (shots * (n_1q + 10 * n_2q + 5 * n_m)) / 5000.0, 4)


def sign_tier3_approval_token(
    *,
    target: QuantinuumTarget,
    compiled_sha256: str,
    shots: int,
    secret: str,
) -> str:
    """Mint an HMAC-SHA256 Tier 3 approval token bound to `(target, compiled_sha256, shots)`."""
    if target not in TIER3_TARGETS:
        raise GovernanceError(f"sign_tier3_approval_token expects a Tier 3 target, got '{target}'.")
    if not secret or not secret.strip():
        raise GovernanceError("OSIRIS_APPROVAL_SECRET must be non-empty to sign a Tier 3 token.")
    msg = f"{target}:{compiled_sha256}:{shots}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def verify_tier3_approval_token(
    approval_token: Optional[str],
    *,
    target: QuantinuumTarget,
    compiled_sha256: str,
    shots: int,
    secret: Optional[str],
) -> None:
    """Fail closed unless the HMAC-SHA256 Tier 3 approval token matches the exact job."""
    if not approval_token or not secret:
        raise GovernanceError(
            f"Tier 3 target '{target}' requires a valid signed approval_token and OSIRIS_APPROVAL_SECRET."
        )
    expected_msg = f"{target}:{compiled_sha256}:{shots}".encode("utf-8")
    expected_sig = hmac.new(secret.encode("utf-8"), expected_msg, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(approval_token.strip(), expected_sig):
        raise GovernanceError(f"Invalid Tier 3 approval_token for target '{target}'.")


class QuantinuumNexusClient:
    """Minimal, zero-dependency HTTP client for Quantinuum Nexus with OSIRIS governance."""

    def __init__(self, config: NexusConfig) -> None:
        self.config = config

    def _request(
        self, method: str, path: str, body: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        url = f"{self.config.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.config.token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "osiris-cli/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:512]
            raise RuntimeError(f"Nexus API {exc.code} on {method} {path}: {detail}") from exc

    def test_connection(self) -> Dict[str, Any]:
        """Verify Nexus credentials and return sanitized backend list."""
        res = self._request("GET", "/backends")
        return {"ok": True, "backends": res.get("backends", res)}

    def submit_and_wait(
        self,
        source: str,
        *,
        target: QuantinuumTarget = "H2-1SC",
        shots: int = 1024,
        budget_hqc: Optional[float] = None,
        compiler_options: Optional[QuantinuumCompilerOptions] = None,
        approval_token: Optional[str] = None,
        skip_preflight: bool = False,
        poll_interval_s: float = 2.0,
        max_wait_s: float = 300.0,
    ) -> NexusDisposition:
        """Compile, enforce governance, submit to Nexus, poll, and return an immutable receipt."""
        if target not in ALL_NEXUS_TARGETS:
            raise GovernanceError(f"Unsupported Quantinuum Nexus target '{target}'.")
        if shots <= 0 or shots > self.config.max_shots:
            raise GovernanceError(f"shots={shots} out of bounds (1..{self.config.max_shots}).")

        opts = compiler_options or QuantinuumCompilerOptions()
        compiled = compile_for_quantinuum(source, target=target, options=opts)
        if not compiled.ok:
            raise GovernanceError(f"TKET pre-flight failed: {'; '.join(compiled.diagnostics)}")

        tier: Literal[1, 2, 3] = (
            1 if target in TIER1_TARGETS else 2 if target in TIER2_TARGETS else 3
        )
        hqc_est = 0.0 if tier == 1 else estimate_hqc(compiled, shots)
        cap = min(self.config.max_hqc, budget_hqc) if budget_hqc is not None else self.config.max_hqc

        if tier >= 2 and hqc_est > cap:
            raise GovernanceError(f"Estimated HQC ({hqc_est}) exceeds budget cap ({cap}).")

        if tier == 3:
            verify_tier3_approval_token(
                approval_token,
                target=target,
                compiled_sha256=compiled.compiled_sha256,
                shots=shots,
                secret=self.config.approval_secret,
            )

        # Mandatory free H2-1SC (or Helios-SC for >56q) syntax check before any paid Tier 2/3 job
        if tier >= 2 and not skip_preflight:
            sc_target: QuantinuumTarget = (
                "Helios-SC" if compiled.metrics.n_qubits > 56 else "H2-1SC"
            )
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

        payload = {
            "backend": target,
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
            status_now = str(job_state.get("status", "unknown")).lower()
            if status_now in {"completed", "succeeded", "failed", "cancelled"} and (
                job_state is not submitted or "counts" in job_state
            ):
                break
            job_state = self._request("GET", f"/jobs/{job_id}")
            status = str(job_state.get("status", "unknown")).lower()
            if status in {"completed", "succeeded", "failed", "cancelled"}:
                break
            time.sleep(poll_interval_s)

        final_status = str(job_state.get("status", "timeout")).lower()
        is_hw = tier == 3 and final_status in {"completed", "succeeded"}
        disposition: Literal["SIMULATED", "HARDWARE_ATTESTED", "QUARANTINED"] = (
            "HARDWARE_ATTESTED"
            if is_hw
            else "SIMULATED"
            if final_status in {"completed", "succeeded"}
            else "QUARANTINED"
        )

        return NexusDisposition(
            job_id=job_id,
            target=target,
            tier=tier,
            status=final_status,
            hardware=is_hw,
            disposition=disposition,
            shots=shots,
            estimated_hqc=float(job_state.get("hqc_cost", hqc_est)),
            source_sha256=compiled.source_sha256,
            compiled_sha256=compiled.compiled_sha256,
            compiler_options=compiled.compiler_options,
            counts=dict(job_state.get("counts", {})),
            calibration_snapshot=dict(job_state.get("calibration", {})),
            diagnostics=list(compiled.diagnostics),
        )
