"""Quantinuum Nexus Gatekeeper, Zero-Waste Tier 3 HQC Budget Guard & Receipt Ledger.

Core invariant: *Models propose; deterministic code decides, measures, and records.*
  1. `propose_quantinuum_circuit`: generates a QCCD candidate circuit (with optional local
     Ollama proposer) and validates it deterministically before returning.
  2. `verify_quantinuum_circuit`: runs `H2-1SC` / `Helios-SC` static QCCD gatekeeping,
     exact `Decimal` HQC cost calculation, optional local QCCD trajectory noise emulation,
     and `exec-canon-1` SHA-256 receipt recording (`SIMULATED`, `hardware=False`).
  3. `mint_tier3_token` / `verify_tier3_token` / `run_tier3_governed`: cryptographic Tier 3
     approval guard bound to `(circuit_sha256, target, max_shots, budget_hqc)` (as well as
     `OSIRIS_APPROVAL_SECRET` HMAC tokens over `target:compiled_sha256:shots`) so an AI
     loop can never burn HQCs on unapproved or mutated circuits.
  4. `verify_nexus_ledger` / `verify_all_quantum_ledgers`: offline tamper-evident audit of
     `$OSIRIS_HOME/quantum/nexus_receipts.jsonl`, `nexus_dispositions.jsonl`, and sibling ledgers.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import urllib.request
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .exec_kernel import CANON_VERSION, ExecJournal, canon
from .providers.local_qccd import (
    ParsedCircuit,
    QCCDCircuitError,
    QCCDEmulationResult,
    QCCDNoiseModel,
    emulate_qccd,
    parse_quantum_source,
)
from .providers.quantinuum import (
    QuantinuumTargetSpec,
    compute_hqc,
    optional_tket_check,
    resolve_quantinuum_target,
)
from .providers.quantinuum_nexus import (
    GovernanceError,
    NexusConfig,
    QuantinuumNexusClient,
    verify_tier3_approval_token,
)
from .providers.tket_wrapper import compile_for_quantinuum
from .quantum_readiness import _canon_safe, default_osiris_home, default_readiness_receipts_path

NEXUS_RECEIPT_SCHEMA = "osiris.quantinuum_nexus_receipt/1"
TIER3_TOKEN_SCHEMA = "osiris.tier3_approval/1"
QUANTUM_PROTOCOL_SCHEMA = "osiris.quantum_protocol/1"
PREGATE_SCHEMA = "osiris.quantum_protocol_pregate/1"
_GENESIS = "0" * 64

_CLASSICAL_BASELINE_TOKENS = (
    "mps",
    "tensor",
    "pauli",
    "bp+osd",
    "bposd",
    "mwpm",
    "blossom",
    "statevector",
    "light-cone",
    "classical",
)
_COLORED_NOISE_TOKENS = (
    "1/f",
    "colored",
    "coloured",
    "ou ",
    "ornstein",
    "local-noise",
    "local noise",
    "non-markovian",
    "correlated-noise",
)
_DRIFT_CONTROL_TOKENS = (
    "a/a",
    "a-a",
    "sentinel",
    "drift",
    "interleaved",
)

DISPOSITIONS = frozenset({"SIMULATED", "PREVIEW_COMPLETE", "HARDWARE_ATTESTED", "QUARANTINED"})
PathArg = Optional[Union[os.PathLike, str]]


class QuantinuumGateError(ValueError):
    """Raised when a pre-flight check, budget gate, approval token, or ledger verification fails."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.detail = message


def default_nexus_receipts_path(osiris_home: PathArg = None) -> Path:
    if osiris_home is None:
        override = os.environ.get("OSIRIS_NEXUS_RECEIPTS_LOG")
        if override:
            return Path(override).expanduser().resolve()
    return default_osiris_home(osiris_home) / "quantum" / "nexus_receipts.jsonl"


def _ed25519_keypair_path(osiris_home: PathArg = None) -> Path:
    return default_osiris_home(osiris_home) / "quantum" / "tier3_operator_ed25519.pem"


def _load_or_create_ed25519_key(osiris_home: PathArg = None) -> Ed25519PrivateKey:
    key_path = _ed25519_keypair_path(osiris_home)
    if key_path.exists():
        pem = key_path.read_bytes()
        loaded = serialization.load_pem_private_key(pem, password=None)
        if not isinstance(loaded, Ed25519PrivateKey):
            raise QuantinuumGateError("KEY_ERROR", f"Invalid operator key type at {key_path}")
        return loaded
    key_path.parent.mkdir(parents=True, exist_ok=True)
    priv = Ed25519PrivateKey.generate()
    pem_bytes = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    fd = os.open(str(key_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(pem_bytes)
    try:
        os.chmod(key_path, 0o600)
    except OSError:
        pass
    return priv


def append_nexus_receipt(body: Dict[str, Any], osiris_home: PathArg = None) -> Tuple[str, Path]:
    """Append a zero-float `exec-canon-1` receipt to `nexus_receipts.jsonl` (`0o600`)."""
    disp = str(body.get("disposition", ""))
    if disp not in DISPOSITIONS:
        raise QuantinuumGateError("INVALID_DISPOSITION", f"Unsupported disposition {disp!r}")
    tier = int(body.get("tier", 1))
    if tier in (1, 2) and (disp == "HARDWARE_ATTESTED" or body.get("hardware") is True):
        raise QuantinuumGateError(
            "SPOOFED_HARDWARE_DISPOSITION",
            "Tier 1/2 syntax-checker or emulator runs can never emit HARDWARE_ATTESTED or hardware=True",
        )
    if body.get("hardware") is True and disp != "HARDWARE_ATTESTED":
        raise QuantinuumGateError(
            "SPOOFED_HARDWARE_DISPOSITION",
            "hardware=True is permitted exclusively with disposition=HARDWARE_ATTESTED",
        )

    path = default_nexus_receipts_path(osiris_home)
    clean_body = _canon_safe(body)
    journal = ExecJournal(path)
    digest = journal.append(clean_body)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return digest, path


# ---------------------------------------------------------------------------
# 1. Deterministic Candidate Proposal (`osiris propose --target quantinuum`)
# ---------------------------------------------------------------------------

_TEMPLATE_BITFLIP_5Q = """OPENQASM 2.0;
include "qelib1.inc";

// Proposed for Quantinuum QCCD: 3-qubit bit-flip QEC + 2 reusable syndrome ancillas
qreg q[5];
creg syn[2];
creg syn_r2[2];
creg data[3];

x q[0];
cx q[0], q[1];
cx q[0], q[2];
barrier q[0], q[1], q[2];

// Simulated single-qubit fault on q[0]
x q[0];
barrier q[0], q[1], q[2];

// Mid-circuit syndrome extraction + optical-pumping ancilla reset
cx q[0], q[3];
cx q[1], q[3];
cx q[1], q[4];
cx q[2], q[4];
measure q[3] -> syn[0];
measure q[4] -> syn[1];
reset q[3];
reset q[4];

// Real-time classical feedforward correction
if(syn == 1) x q[0];
if(syn == 2) x q[2];
if(syn == 3) x q[1];

// Round 2 verification reusing reset ancillas
cx q[0], q[3];
cx q[1], q[3];
cx q[1], q[4];
cx q[2], q[4];
measure q[3] -> syn_r2[0];
measure q[4] -> syn_r2[1];
reset q[3];
reset q[4];

measure q[0] -> data[0];
measure q[1] -> data[1];
measure q[2] -> data[2];
"""

_TEMPLATE_CODE422 = """OPENQASM 2.0;
include "qelib1.inc";

// Proposed for Quantinuum QCCD: [[4,2,2]] code with non-local weight-4 checks & ancilla reset
qreg q[6];
creg syn_r1[2];
creg syn_r2[2];
creg out[4];

h q[0];
cx q[0], q[1];
cx q[0], q[2];
cx q[0], q[3];
x q[0];
x q[1];
barrier q[0], q[1], q[2], q[3];

h q[4];
cx q[4], q[0];
cx q[4], q[1];
cx q[4], q[2];
cx q[4], q[3];
h q[4];

cx q[0], q[5];
cx q[1], q[5];
cx q[2], q[5];
cx q[3], q[5];

measure q[4] -> syn_r1[0];
measure q[5] -> syn_r1[1];
reset q[4];
reset q[5];

h q[4];
cx q[4], q[0];
cx q[4], q[1];
cx q[4], q[2];
cx q[4], q[3];
h q[4];

cx q[0], q[5];
cx q[1], q[5];
cx q[2], q[5];
cx q[3], q[5];

measure q[4] -> syn_r2[0];
measure q[5] -> syn_r2[1];
reset q[4];
reset q[5];

measure q[0] -> out[0];
measure q[1] -> out[1];
measure q[2] -> out[2];
measure q[3] -> out[3];
"""

_TEMPLATE_TELEPORT_FEEDFORWARD = """OPENQASM 2.0;
include "qelib1.inc";

// Proposed for Quantinuum QCCD: Mid-circuit state teleportation with reset & Pauli feedforward
qreg q[3];
creg m0[1];
creg m1[1];
creg out[1];

rx(pi/3) q[0];
h q[1];
cx q[1], q[2];

cx q[0], q[1];
h q[0];
measure q[0] -> m0[0];
measure q[1] -> m1[0];
reset q[0];
reset q[1];

if(m1 == 1) x q[2];
if(m0 == 1) z q[2];
rx(-pi/3) q[2];
measure q[2];
measure q[2] -> out[0];
""".replace("measure q[2];\n", "")


def _try_ollama_propose(prompt: str, model: str) -> Optional[str]:
    payload = json.dumps(
        {
            "model": model,
            "prompt": (
                "Output ONLY valid OpenQASM 2.0 code (starting with `OPENQASM 2.0;`) for "
                "Quantinuum H2 QCCD with mid-circuit measurement, `reset`, and `if(...)` "
                f"feedforward matching this specification: {prompt}"
            ),
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=8.0) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        text = str(raw.get("response", "")).strip()
        if "```" in text:
            for block in text.split("```"):
                cleaned = block.strip()
                if cleaned.startswith("qasm") or cleaned.startswith("openqasm"):
                    cleaned = cleaned.split("\n", 1)[1].strip() if "\n" in cleaned else ""
                if "OPENQASM" in cleaned:
                    return cleaned + "\n"
        if "OPENQASM" in text:
            idx = text.index("OPENQASM")
            return text[idx:].strip() + "\n"
    except Exception:
        return None
    return None


@dataclass(frozen=True)
class ProposalResult:
    prompt: str
    target: str
    proposer_source: str
    qasm_source: str
    circuit_sha256: str
    preverified: bool


def propose_quantinuum_circuit(
    prompt: str,
    target: str = "quantinuum",
    model: Optional[str] = None,
) -> ProposalResult:
    """Propose an OpenQASM 2.0 circuit for Quantinuum QCCD and gatekeep it before return."""
    spec = resolve_quantinuum_target(target)
    proposer_source = "deterministic_qccd_synthesizer"
    candidate: Optional[str] = None

    if model:
        llm_qasm = _try_ollama_propose(prompt, model=model)
        if llm_qasm is not None:
            try:
                parsed_llm = parse_quantum_source(llm_qasm)
                if parsed_llm.n_qubits <= spec.max_qubits:
                    candidate = llm_qasm
                    proposer_source = f"ollama/{model}"
            except QCCDCircuitError:
                candidate = None

    if candidate is None:
        low = prompt.lower()
        if any(tok in low for tok in ("4,2,2", "422", "error-detect", "detect")):
            candidate = _TEMPLATE_CODE422
        elif any(tok in low for tok in ("teleport", "bell")):
            candidate = _TEMPLATE_TELEPORT_FEEDFORWARD
        else:
            candidate = _TEMPLATE_BITFLIP_5Q

    parsed = parse_quantum_source(candidate)
    if parsed.n_qubits > spec.max_qubits:
        raise QuantinuumGateError(
            "QUBIT_LIMIT_EXCEEDED",
            f"Proposed circuit uses {parsed.n_qubits} qubits > {spec.max_qubits} on {spec.name}",
        )
    return ProposalResult(
        prompt=prompt,
        target=spec.name,
        proposer_source=proposer_source,
        qasm_source=candidate,
        circuit_sha256=parsed.circuit_sha256,
        preverified=True,
    )


# ---------------------------------------------------------------------------
# 2. Static `H2-1SC` Gatekeeper + Local QCCD Emulation (`osiris verify`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerificationReport:
    passed: bool
    target: str
    tier: int
    disposition: str
    hardware: bool
    source_format: str
    circuit_sha256: str
    n_qubits: int
    max_qubits: int
    single_qubit_gates: int
    physical_1q_gates: int
    virtual_z_gates: int
    two_qubit_gates: int
    measurements: int
    mid_circuit_measurements: int
    resets: int
    feedforward_conditionals: int
    routing_swaps: int
    shots: int
    hqc_estimate: Decimal
    tket_check: Dict[str, Any]
    emulation: Optional[QCCDEmulationResult]
    receipt_hash: Optional[str]
    receipt_path: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema": NEXUS_RECEIPT_SCHEMA,
            "passed": self.passed,
            "verification_status": "VERIFIED" if self.passed else "FAILED",
            "target": self.target,
            "tier": self.tier,
            "disposition": self.disposition,
            "hardware": False,
            "source_format": self.source_format,
            "circuit_sha256": self.circuit_sha256,
            "n_qubits": self.n_qubits,
            "max_qubits": self.max_qubits,
            "single_qubit_gates": self.single_qubit_gates,
            "physical_1q_gates": self.physical_1q_gates,
            "virtual_z_gates": self.virtual_z_gates,
            "two_qubit_gates": self.two_qubit_gates,
            "measurements": self.measurements,
            "mid_circuit_measurements": self.mid_circuit_measurements,
            "resets": self.resets,
            "feedforward_conditionals": self.feedforward_conditionals,
            "routing_swaps": self.routing_swaps,
            "shots": self.shots,
            "hqc_estimate": str(self.hqc_estimate),
            "tket_check": self.tket_check,
        }
        if self.emulation is not None:
            payload["emulation"] = {
                "noise_model": self.emulation.noise_model,
                "shots": self.emulation.shots,
                "seed": self.emulation.seed,
                "raw_counts": self.emulation.raw_counts,
                "raw_counts_sha256": self.emulation.raw_counts_sha256,
                "register_counts": self.emulation.register_counts,
                "dominant_bitstring": self.emulation.dominant_bitstring,
                "dominant_probability": str(
                    Decimal(str(round(self.emulation.dominant_probability, 6)))
                ),
                "syndrome_zero_rate": (
                    str(Decimal(str(round(self.emulation.syndrome_zero_rate, 6))))
                    if self.emulation.syndrome_zero_rate is not None
                    else None
                ),
            }
        if self.receipt_hash is not None:
            payload["receipt_hash"] = self.receipt_hash
            payload["receipt_path"] = self.receipt_path
        return payload


def verify_quantinuum_circuit(
    source: str,
    target: str = "H2-1SC",
    emulate: Optional[str] = None,
    shots: int = 1024,
    seed: int = 7,
    record_ledger: bool = False,
    osiris_home: PathArg = None,
    filename: str = "",
) -> VerificationReport:
    """Deterministically validate a circuit for Quantinuum QCCD and optionally emulate it."""
    spec = resolve_quantinuum_target(target)
    max_shots = int(os.environ.get("OSIRIS_MAX_SHOTS", "10000"))
    if shots <= 0 or shots > max_shots:
        raise QuantinuumGateError(
            "BLOCKED_SHOT_LIMIT",
            f"Requested shots={shots} outside governed interval [1, {max_shots}] (OSIRIS_MAX_SHOTS)",
        )

    try:
        parsed: ParsedCircuit = parse_quantum_source(source, filename=filename)
    except QCCDCircuitError as exc:
        raise QuantinuumGateError("BLOCKED_SYNTAX_PREFLIGHT", str(exc)) from exc

    if parsed.n_qubits > spec.max_qubits:
        raise QuantinuumGateError(
            "BLOCKED_QUBIT_CEILING",
            f"Circuit declares {parsed.n_qubits} qubits > {spec.name} limit ({spec.max_qubits})",
        )

    n_m = parsed.active_qubits + parsed.measurements + parsed.resets
    hqc_est = compute_hqc(parsed.physical_1q_gates, parsed.two_qubit_gates, n_m, shots)
    tk_info = optional_tket_check(source, parsed.source_format)

    run_emu = emulate is not None or spec.role == "emulator"
    emu_result: Optional[QCCDEmulationResult] = None
    if run_emu:
        mode = (emulate or "qccd").strip().lower()
        if mode == "ideal":
            nmodel = QCCDNoiseModel.ideal()
        elif mode in ("qccd", "selene", "h2", "helios"):
            nmodel = spec.noise_model()
        else:
            raise QuantinuumGateError(
                "INVALID_EMULATOR",
                f"Unsupported --emulate mode {emulate!r} (expected 'qccd' or 'ideal')",
            )
        try:
            emu_result = emulate_qccd(
                parsed,
                target=spec.name,
                noise_model=nmodel,
                shots=shots,
                seed=seed,
            )
        except QCCDCircuitError as exc:
            raise QuantinuumGateError("EMULATION_ERROR", str(exc)) from exc

    effective_tier = 2 if (spec.tier >= 2 or emu_result is not None) else 1
    report = VerificationReport(
        passed=True,
        target=spec.name,
        tier=effective_tier,
        disposition="SIMULATED",
        hardware=False,
        source_format=parsed.source_format,
        circuit_sha256=parsed.circuit_sha256,
        n_qubits=parsed.n_qubits,
        max_qubits=spec.max_qubits,
        single_qubit_gates=parsed.single_qubit_gates,
        physical_1q_gates=parsed.physical_1q_gates,
        virtual_z_gates=parsed.virtual_z_gates,
        two_qubit_gates=parsed.two_qubit_gates,
        measurements=parsed.measurements,
        mid_circuit_measurements=parsed.mid_circuit_measurements,
        resets=parsed.resets,
        feedforward_conditionals=parsed.feedforward_conditionals,
        routing_swaps=0,
        shots=shots,
        hqc_estimate=hqc_est,
        tket_check=tk_info,
        emulation=emu_result,
        receipt_hash=None,
        receipt_path=None,
    )

    if record_ledger:
        receipt_body = report.to_dict()
        receipt_body["recorded_at_unix"] = int(time.time())
        r_hash, r_path = append_nexus_receipt(receipt_body, osiris_home=osiris_home)
        report = VerificationReport(
            **{
                **report.__dict__,
                "receipt_hash": r_hash,
                "receipt_path": str(r_path),
            }
        )
    return report


# ---------------------------------------------------------------------------
# 3. Tier 3 Zero-Waste HQC Budget & Signed Approval Guard (`osiris run`)
# ---------------------------------------------------------------------------


def mint_tier3_token(
    circuit_sha256: str,
    target: str,
    max_shots: int,
    max_hqc: Union[Decimal, str, int, float],
    ttl_seconds: int = 3600,
    osiris_home: PathArg = None,
) -> str:
    """Mint an operator-signed Tier 3 token bound to `(circuit_sha256, target, max_shots, max_hqc)`."""
    spec = resolve_quantinuum_target(target)
    if not spec.is_live_hardware:
        raise QuantinuumGateError(
            "INVALID_TIER3_TARGET",
            f"Tier 3 approval tokens apply to live QPU targets (H2-1, H2-2, Helios), got {spec.name}",
        )
    hqc_dec = Decimal(str(max_hqc))
    now = int(time.time())
    payload_obj = {
        "schema": TIER3_TOKEN_SCHEMA,
        "circuit_sha256": circuit_sha256.lower(),
        "target": spec.name,
        "max_shots": int(max_shots),
        "max_hqc": str(hqc_dec),
        "iat": now,
        "exp": now + int(ttl_seconds),
    }
    payload_bytes = canon(payload_obj).encode("utf-8")
    payload_b64 = base64.urlsafe_b64encode(payload_bytes).decode("ascii").rstrip("=")

    hmac_secret = os.environ.get("OSIRIS_TIER3_HMAC_SECRET", "").strip()
    if hmac_secret:
        sig_hex = hmac.new(hmac_secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
        return f"osiris-t3-hmac1.{payload_b64}.{sig_hex}"

    priv = _load_or_create_ed25519_key(osiris_home)
    sig = priv.sign(payload_bytes)
    pub_raw = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    pub_b64 = base64.urlsafe_b64encode(pub_raw).decode("ascii").rstrip("=")
    return f"osiris-t3-ed25519.{payload_b64}.{pub_b64}.{sig.hex()}"


def _b64url_decode(segment: str) -> bytes:
    pad = "=" * ((4 - (len(segment) % 4)) % 4)
    return base64.urlsafe_b64decode(segment + pad)


def verify_tier3_token(
    token: str,
    circuit_sha256: str,
    target_spec: QuantinuumTargetSpec,
    shots: int,
    hqc_estimate: Decimal,
    osiris_home: PathArg = None,
    compiled_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify cryptographic signature and scope of a Tier 3 approval token."""
    raw = (token or "").strip()
    if not raw:
        raise QuantinuumGateError(
            "BLOCKED_APPROVAL_TOKEN",
            "Tier 3 live QPU target requires a non-empty signed --approval-token",
        )

    # Support compact 64-hex HMAC tokens minted via `OSIRIS_APPROVAL_SECRET` (`quantinuum_nexus.py`)
    if "." not in raw and len(raw) == 64 and compiled_sha256:
        secret = (
            os.environ.get("OSIRIS_APPROVAL_SECRET")
            or os.environ.get("OSIRIS_TIER3_HMAC_SECRET")
            or ""
        ).strip()
        try:
            verify_tier3_approval_token(
                raw,
                target=target_spec.name,  # type: ignore[arg-type]
                compiled_sha256=compiled_sha256,
                shots=shots,
                secret=secret or None,
            )
            return {
                "schema": "osiris.tier3_hmac_compact/1",
                "compiled_sha256": compiled_sha256,
                "target": target_spec.name,
                "max_shots": shots,
                "exp": int(time.time()) + 300,
            }
        except GovernanceError as exc:
            raise QuantinuumGateError("BLOCKED_APPROVAL_TOKEN", str(exc)) from exc

    parts = raw.split(".")
    try:
        if len(parts) == 3 and parts[0] == "osiris-t3-hmac1":
            secret = os.environ.get("OSIRIS_TIER3_HMAC_SECRET", "").strip()
            if not secret:
                raise QuantinuumGateError(
                    "BLOCKED_APPROVAL_TOKEN",
                    "HMAC Tier 3 token supplied, but OSIRIS_TIER3_HMAC_SECRET is not configured",
                )
            payload_bytes = _b64url_decode(parts[1])
            expected_sig = hmac.new(
                secret.encode("utf-8"), payload_bytes, hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(expected_sig, parts[2]):
                raise QuantinuumGateError(
                    "BLOCKED_APPROVAL_TOKEN", "Invalid HMAC signature on Tier 3 approval token"
                )
        elif len(parts) == 4 and parts[0] == "osiris-t3-ed25519":
            payload_bytes = _b64url_decode(parts[1])
            token_pub_bytes = _b64url_decode(parts[2])
            sig_bytes = bytes.fromhex(parts[3])
            local_priv = _load_or_create_ed25519_key(osiris_home)
            expected_pub_bytes = local_priv.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
            if not hmac.compare_digest(token_pub_bytes, expected_pub_bytes):
                raise QuantinuumGateError(
                    "BLOCKED_APPROVAL_TOKEN",
                    "Tier 3 Ed25519 token was not signed by this workspace's operator key",
                )
            pub = Ed25519PublicKey.from_public_bytes(expected_pub_bytes)
            try:
                pub.verify(sig_bytes, payload_bytes)
            except InvalidSignature as exc:
                raise QuantinuumGateError(
                    "BLOCKED_APPROVAL_TOKEN", "Ed25519 signature verification failed"
                ) from exc
        else:
            raise QuantinuumGateError(
                "BLOCKED_APPROVAL_TOKEN",
                "Unrecognized Tier 3 approval token format",
            )
        claims = json.loads(payload_bytes.decode("utf-8"))
    except QuantinuumGateError:
        raise
    except Exception as exc:
        raise QuantinuumGateError(
            "BLOCKED_APPROVAL_TOKEN", f"Malformed Tier 3 approval token: {exc}"
        ) from exc

    now = int(time.time())
    if int(claims.get("exp", 0)) < now:
        raise QuantinuumGateError("BLOCKED_APPROVAL_TOKEN", "Tier 3 approval token has expired")
    if str(claims.get("circuit_sha256", "")).lower() != circuit_sha256.lower():
        raise QuantinuumGateError(
            "BLOCKED_APPROVAL_TOKEN",
            "Token circuit_sha256 mismatch (circuit was modified after approval)",
        )
    if str(claims.get("target", "")) != target_spec.name:
        raise QuantinuumGateError(
            "BLOCKED_APPROVAL_TOKEN",
            f"Token approved for target {claims.get('target')!r}, not {target_spec.name!r}",
        )
    if shots > int(claims.get("max_shots", 0)):
        raise QuantinuumGateError(
            "BLOCKED_APPROVAL_TOKEN",
            f"Requested shots={shots} exceeds token max_shots={claims.get('max_shots')}",
        )
    token_hqc_cap = Decimal(str(claims.get("max_hqc", "0")))
    if hqc_estimate > token_hqc_cap:
        raise QuantinuumGateError(
            "BLOCKED_HQC_BUDGET",
            f"Estimated HQC={hqc_estimate} exceeds token max_hqc={token_hqc_cap}",
        )
    return claims


@dataclass(frozen=True)
class GovernedRunResult:
    status: str
    target: str
    tier: int
    disposition: str
    hardware: bool
    circuit_sha256: str
    shots: int
    hqc_estimate: Decimal
    budget_hqc: Decimal
    preflight_checker: str
    receipt_hash: str
    receipt_path: str
    nexus_job_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": NEXUS_RECEIPT_SCHEMA,
            "status": self.status,
            "target": self.target,
            "tier": self.tier,
            "disposition": self.disposition,
            "hardware": self.hardware,
            "circuit_sha256": self.circuit_sha256,
            "shots": self.shots,
            "hqc_estimate": str(self.hqc_estimate),
            "budget_hqc": str(self.budget_hqc),
            "preflight_checker": self.preflight_checker,
            "nexus_job_id": self.nexus_job_id,
            "receipt_hash": self.receipt_hash,
            "receipt_path": self.receipt_path,
        }


def validate_quantum_protocol_pregate(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Enforce ``osiris.quantum_protocol/1``, classical-attack baseline, and colored-noise/drift
    controls before live Tier 3 HQC budget allocation (WK-RW-002)."""
    from .plan_gates import check_no_quarantined_constants

    if not isinstance(manifest, Mapping):
        return {
            "classical_baseline_ok": False,
            "colored_noise_and_drift_ok": False,
            "manifest_sha256": "",
            "passed": False,
            "protocol_schema": "",
            "schema": PREGATE_SCHEMA,
            "violations": ["INVALID_PROTOCOL_SCHEMA"],
        }

    violations: List[str] = []
    proto_schema = str(manifest.get("schema") or "")
    if proto_schema != QUANTUM_PROTOCOL_SCHEMA:
        violations.append("INVALID_PROTOCOL_SCHEMA")

    if check_no_quarantined_constants(dict(manifest)):
        violations.append("SYS024_QUARANTINED_CONSTANT")

    shots_val = int(manifest.get("shots_per_circuit", 0) or 0)
    blocks_val = int(manifest.get("num_blocks", 0) or 0)
    if shots_val < 100 or blocks_val < 3:
        violations.append("INSUFFICIENT_REPLICATION")

    controls_raw = manifest.get("control_experiments") or manifest.get("controls") or ()
    if isinstance(controls_raw, str):
        controls_list = [controls_raw]
    elif isinstance(controls_raw, (list, tuple)):
        controls_list = [str(item) for item in controls_raw]
    else:
        controls_list = []
    controls_blob = " ".join(controls_list).lower()

    cb_flag = manifest.get("classical_baseline_passed")
    cb_field = str(manifest.get("classical_baseline") or "").strip().lower()
    if cb_flag is False:
        classical_ok = False
        violations.append("CLASSICAL_BASELINE_FAILED")
    else:
        classical_ok = bool(
            cb_flag is True
            or cb_field
            or any(tok in controls_blob for tok in _CLASSICAL_BASELINE_TOKENS)
        )
        if not classical_ok:
            violations.append("MISSING_CLASSICAL_BASELINE")

    cn_flag = manifest.get("colored_noise_control_passed")
    drift_flag = manifest.get("drift_sentinel_passed")
    cn_field = str(manifest.get("colored_noise_null") or "").strip().lower()
    drift_field = str(manifest.get("aa_drift_sentinel") or "").strip().lower()
    if cn_flag is False or drift_flag is False:
        controls_ok = False
        violations.append("COLORED_NOISE_OR_DRIFT_CONTROL_FAILED")
    else:
        has_colored = bool(
            cn_flag is True
            or cn_field
            or any(tok in controls_blob for tok in _COLORED_NOISE_TOKENS)
        )
        has_drift = bool(
            drift_flag is True
            or drift_field
            or any(tok in controls_blob for tok in _DRIFT_CONTROL_TOKENS)
        )
        controls_ok = has_colored and has_drift
        if not controls_ok:
            violations.append("MISSING_COLORED_NOISE_OR_DRIFT_CONTROL")

    manifest_sha = hashlib.sha256(canon(_canon_safe(dict(manifest))).encode("utf-8")).hexdigest()
    return {
        "classical_baseline_ok": classical_ok,
        "colored_noise_and_drift_ok": controls_ok,
        "manifest_sha256": manifest_sha,
        "passed": len(violations) == 0,
        "protocol_schema": proto_schema,
        "schema": PREGATE_SCHEMA,
        "violations": violations,
    }


def run_tier3_governed(
    source: str,
    target: str = "H2-1",
    shots: int = 500,
    budget_hqc: Union[Decimal, str, int, float] = Decimal("25"),
    approval_token: str = "",
    osiris_home: PathArg = None,
    filename: str = "",
    protocol_manifest: Optional[Mapping[str, Any]] = None,
) -> GovernedRunResult:
    """Execute governed Tier 3 Quantinuum Nexus workflow with mandatory fail-closed preflight."""
    if protocol_manifest is not None:
        pregate = validate_quantum_protocol_pregate(protocol_manifest)
        if not pregate["passed"]:
            raise QuantinuumGateError(
                "BLOCKED_PROTOCOL_PREGATE",
                f"Quantum protocol pre-gate failed: {', '.join(pregate['violations'])}",
            )
    spec = resolve_quantinuum_target(target)
    try:
        budget_dec = Decimal(str(budget_hqc))
    except InvalidOperation as exc:
        raise QuantinuumGateError("BLOCKED_HQC_BUDGET", f"Invalid --budget-hqc {budget_hqc!r}") from exc
    if budget_dec <= Decimal("0"):
        raise QuantinuumGateError("BLOCKED_HQC_BUDGET", "--budget-hqc must be > 0")

    env_max_hqc = Decimal(os.environ.get("OSIRIS_MAX_HQC", "500"))
    effective_cap = min(budget_dec, env_max_hqc)

    # Gate 1: Mandatory deterministic Syntax-Checker pre-flight (H2-1SC / Helios-SC)
    checker_name = "Helios-SC" if spec.family == "Helios" else "H2-1SC"
    preflight = verify_quantinuum_circuit(
        source,
        target=checker_name,
        emulate=None,
        shots=shots,
        record_ledger=False,
        osiris_home=osiris_home,
        filename=filename,
    )
    compiled = compile_for_quantinuum(source, target=spec.name)  # type: ignore[arg-type]

    # Gate 2: Hard HQC budget cap check BEFORE checking credentials or network
    if preflight.hqc_estimate > effective_cap:
        raise QuantinuumGateError(
            "BLOCKED_HQC_BUDGET",
            f"Estimated HQC cost {preflight.hqc_estimate} exceeds cap {effective_cap} "
            f"(budget={budget_dec}, OSIRIS_MAX_HQC={env_max_hqc})",
        )

    # Gate 3: Cryptographic Tier 3 operator approval token check
    if spec.is_live_hardware:
        token_claims = verify_tier3_token(
            token=approval_token,
            circuit_sha256=preflight.circuit_sha256,
            target_spec=spec,
            shots=shots,
            hqc_estimate=preflight.hqc_estimate,
            osiris_home=osiris_home,
            compiled_sha256=compiled.compiled_sha256,
        )
    else:
        token_claims = {"tier": spec.tier, "note": "non_hardware_target"}

    # Gate 4: Live Nexus submission remains fail-closed unless explicitly enabled + token present
    allow_live = os.environ.get("OSIRIS_ALLOW_LIVE_NEXUS_SUBMIT", "").strip() == "1"
    nexus_token = os.environ.get("QUANTINUUM_NEXUS_TOKEN", "").strip()

    if not (allow_live and nexus_token):
        receipt_body = {
            "schema": NEXUS_RECEIPT_SCHEMA,
            "status": "QNEXUS_LIVE_SUBMIT_DISABLED",
            "target": spec.name,
            "tier": spec.tier,
            "disposition": "PREVIEW_COMPLETE",
            "hardware": False,
            "circuit_sha256": preflight.circuit_sha256,
            "compiled_sha256": compiled.compiled_sha256,
            "shots": shots,
            "hqc_estimate": str(preflight.hqc_estimate),
            "budget_hqc": str(effective_cap),
            "preflight_checker": checker_name,
            "approval_verified": bool(spec.is_live_hardware),
            "approval_exp": int(token_claims.get("exp", 0)),
            "recorded_at_unix": int(time.time()),
        }
        r_hash, r_path = append_nexus_receipt(receipt_body, osiris_home=osiris_home)
        return GovernedRunResult(
            status="QNEXUS_LIVE_SUBMIT_DISABLED",
            target=spec.name,
            tier=spec.tier,
            disposition="PREVIEW_COMPLETE",
            hardware=False,
            circuit_sha256=preflight.circuit_sha256,
            shots=shots,
            hqc_estimate=preflight.hqc_estimate,
            budget_hqc=effective_cap,
            preflight_checker=checker_name,
            receipt_hash=r_hash,
            receipt_path=str(r_path),
            nexus_job_id=None,
        )

    # Dispatch via zero-dependency governed HTTP client (`QuantinuumNexusClient`)
    from .nexus_ledger import NexusAuditLedger

    cfg = NexusConfig.from_env()
    hmac_tok = approval_token
    if "." in hmac_tok and cfg.approval_secret:
        from .providers.quantinuum_nexus import sign_tier3_approval_token

        hmac_tok = sign_tier3_approval_token(
            target=spec.name,  # type: ignore[arg-type]
            compiled_sha256=compiled.compiled_sha256,
            shots=shots,
            secret=cfg.approval_secret,
        )
    client = QuantinuumNexusClient(cfg)
    try:
        disp = client.submit_and_wait(
            source,
            target=spec.name,  # type: ignore[arg-type]
            shots=shots,
            budget_hqc=float(effective_cap),
            approval_token=hmac_tok,
        )
    except (GovernanceError, RuntimeError) as exc:
        raise QuantinuumGateError("NEXUS_SUBMIT_FAILED", str(exc)) from exc

    persisted = NexusAuditLedger(osiris_home).append_disposition(disp)
    return GovernedRunResult(
        status=disp.status.upper(),
        target=disp.target,
        tier=disp.tier,
        disposition=disp.disposition,
        hardware=disp.hardware,
        circuit_sha256=disp.source_sha256,
        shots=disp.shots,
        hqc_estimate=Decimal(str(disp.estimated_hqc)),
        budget_hqc=effective_cap,
        preflight_checker=checker_name,
        receipt_hash=persisted["chain_hash"],
        receipt_path=persisted["ledger_path"],
        nexus_job_id=disp.job_id,
    )


# ---------------------------------------------------------------------------
# 4. Offline Cryptographic Receipt Chain Verifier (`osiris ledger verify`)
# ---------------------------------------------------------------------------


def _verify_canon_chain_file(path: Path, check_quantum_integrity: bool = False) -> Dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "valid": True,
            "entries": 0,
            "tail_hash": _GENESIS,
            "errors": [],
        }

    prev = _GENESIS
    expected_seq = 1
    errors: List[str] = []
    entries = 0

    def _forbid_float(num_str: str) -> None:
        raise ValueError(f"float literal {num_str!r} forbidden in exec-canon-1 ledger")

    with open(path, "r", encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            entries += 1
            try:
                rec = json.loads(stripped, parse_float=_forbid_float)
            except Exception as exc:
                errors.append(f"line {lineno}: invalid JSON or forbidden float ({exc})")
                break

            if rec.get("canon_version") != CANON_VERSION:
                errors.append(f"line {lineno}: unexpected canon_version {rec.get('canon_version')!r}")
                break
            if int(rec.get("seq", -1)) != expected_seq:
                errors.append(f"line {lineno}: seq={rec.get('seq')} != expected {expected_seq}")
                break
            if rec.get("prev") != prev:
                errors.append(f"line {lineno}: prev hash chain break")
                break
            body = rec.get("body")
            if not isinstance(body, dict):
                errors.append(f"line {lineno}: body is not a JSON object")
                break
            recomputed = hashlib.sha256((prev + canon(body)).encode("utf-8")).hexdigest()
            if rec.get("hash") != recomputed:
                errors.append(
                    f"line {lineno}: SHA-256 mismatch (recorded={rec.get('hash')}, expected={recomputed})"
                )
                break

            if check_quantum_integrity:
                tier = int(body.get("tier", 1))
                disp = str(body.get("disposition", body.get("evidence_class", "SIMULATED")))
                hw = bool(body.get("hardware", False))
                if tier in (1, 2) and (hw or disp == "HARDWARE_ATTESTED"):
                    errors.append(
                        f"line {lineno}: integrity violation (Tier {tier} entry claims hardware attestation)"
                    )
                    break

            prev = recomputed
            expected_seq += 1

    return {
        "path": str(path),
        "exists": True,
        "valid": len(errors) == 0,
        "entries": entries,
        "tail_hash": prev,
        "errors": errors,
    }


def verify_nexus_ledger(osiris_home: PathArg = None) -> Dict[str, Any]:
    return _verify_canon_chain_file(
        default_nexus_receipts_path(osiris_home), check_quantum_integrity=True
    )


def verify_all_quantum_ledgers(osiris_home: PathArg = None) -> Dict[str, Any]:
    """Verify Nexus receipts, Nexus dispositions, Wave 4 readiness receipts, and exec journal."""
    home = default_osiris_home(osiris_home)
    nexus_res = _verify_canon_chain_file(
        default_nexus_receipts_path(home), check_quantum_integrity=True
    )
    dispositions_res = _verify_canon_chain_file(
        home / "quantum" / "nexus_dispositions.jsonl", check_quantum_integrity=True
    )
    readiness_res = _verify_canon_chain_file(
        default_readiness_receipts_path(home), check_quantum_integrity=True
    )
    exec_res = _verify_canon_chain_file(home / "exec_journal.jsonl", check_quantum_integrity=False)
    ledgers = {
        "nexus_receipts": nexus_res,
        "nexus_dispositions": dispositions_res,
        "readiness_receipts": readiness_res,
        "exec_journal": exec_res,
    }
    overall_ok = all(item["valid"] for item in ledgers.values())
    return {
        "osiris_home": str(home),
        "valid": overall_ok,
        "ledgers": ledgers,
    }
