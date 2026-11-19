"""
osiris_cli/pqc_enclave.py
OSIRIS 6.0 Pillar 4: Post-Quantum Cryptographic (PQC) Enclave Engine
Implements NIST FIPS 204 (ML-DSA-65) & Ed25519 Hybrid Operator Grants.

Security invariants:
1. Local signing only: `generate_hybrid_grant` signs internally with the enclave's private
   keys and never accepts caller-supplied `ed25519_sig_hex` or `pqc_sig_hex` strings.
2. Readable operator approval: `request_approved_grant` displays the exact requested action,
   scope, summary, payload digest, signer fingerprint, and TTL, and signs only after explicit
   operator approval.
3. Explicit trust root: `verify_grant_integrity` verifies signatures strictly against
   explicitly trusted operator public keys (`trusted_public_keys` / `trusted_operator_keys.json`),
   never establishing trust from public keys embedded inside an untrusted grant.
4. Full-payload canonical binding, expiry window (`issued_at`..`expires_at`), and single-use
   replay protection (`consumed_grants.jsonl` + in-memory set).
5. Private keys stay local (`~/.osiris/enclave`, mode `0700` dir / `0600` files, `O_EXCL`
   creation so existing keys are never silently overwritten, and never printed or logged).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from osiris_cli import chain_lock

try:  # cryptography >= 47 provides FIPS 204 ML-DSA; Ed25519 has been available since 2.6
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives.asymmetric.mldsa import (
        MLDSA65PrivateKey,
        MLDSA65PublicKey,
    )
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )
    _CRYPTO_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without cryptography
    _CRYPTO_AVAILABLE = False


DEFAULT_OPERATOR_ID = "Devin Phillip Davis / Enki"
DEFAULT_GRANT_TTL_S = 300.0
MAX_CLOCK_SKEW_S = 5.0
_HEX64 = frozenset("0123456789abcdef")


class ApprovalDeniedError(RuntimeError):
    """Raised when the operator declines or omits approval for a requested grant."""


class GrantVerificationError(RuntimeError):
    """Raised when a grant fails trust-root, signature, payload, scope, expiry, or replay checks."""


class PQCAlgorithm(str, Enum):
    ED25519_CLASSICAL = "Ed25519"
    ML_DSA_44 = "ML-DSA-44"
    ML_DSA_65 = "ML-DSA-65"
    ML_DSA_87 = "ML-DSA-87"
    HYBRID_ED25519_ML_DSA_65 = "Hybrid-Ed25519+ML-DSA-65"


@dataclass(frozen=True)
class FIPS204Parameters:
    """NIST FIPS 204 Module-Lattice-Based Digital Signature Parameters."""

    algorithm: PQCAlgorithm
    n: int = 256
    q: int = 8380417
    k: int = 6
    l: int = 5
    public_key_bytes: int = 1952
    signature_bytes: int = 3309


def default_enclave_dir() -> str:
    env_dir = os.environ.get("OSIRIS_ENCLAVE_DIR", "").strip()
    if env_dir:
        return os.path.abspath(os.path.expanduser(env_dir))
    return os.path.join(os.path.expanduser("~"), ".osiris", "enclave")


def _canon_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in _HEX64 for c in value)


def _normalize_scope(scope: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(scope, str):
        s = scope.strip()
        if not s:
            raise ValueError("grant scope must be non-empty")
        return [s]
    if isinstance(scope, Sequence):
        items = [str(x).strip() for x in scope if str(x).strip()]
        if not items or len(items) != len(scope):
            raise ValueError("grant scope sequence must be non-empty with non-blank items")
        return items
    raise ValueError(f"unsupported grant scope type: {type(scope).__name__}")


def compute_key_fingerprint(ed25519_pub_hex: str, ml_dsa_pub_hex: str) -> str:
    """Compute canonical SHA-256 fingerprint over raw Ed25519 (32B) + ML-DSA-65 (1952B) public keys."""
    ed_bytes = bytes.fromhex(ed25519_pub_hex)
    ml_bytes = bytes.fromhex(ml_dsa_pub_hex)
    if len(ed_bytes) != 32:
        raise ValueError(f"Ed25519 public key must be 32 bytes (got {len(ed_bytes)})")
    if len(ml_bytes) != 1952:
        raise ValueError(f"ML-DSA-65 public key must be 1952 bytes (got {len(ml_bytes)})")
    digest = hashlib.sha256(ed_bytes + b":" + ml_bytes).hexdigest()
    return f"sha256:{digest}"


def _write_exclusive_file(path: str, data: bytes, mode: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, mode)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def _check_private_key_permissions(path: str) -> None:
    if os.name != "posix":
        return
    st = os.stat(path)
    if (stat.S_IMODE(st.st_mode) & 0o077) != 0:
        raise RuntimeError(
            f"Insecure permissions {oct(stat.S_IMODE(st.st_mode))} on private key {path}; "
            "expected owner-only 0600"
        )


def load_trusted_public_keys(trusted_keys_path: str) -> List[Dict[str, str]]:
    """Load explicitly enrolled operator public keys from `trusted_operator_keys.json`."""
    if not os.path.isfile(trusted_keys_path):
        raise RuntimeError(f"Trusted operator key store not found at {trusted_keys_path}")
    with open(trusted_keys_path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    keys = doc.get("trusted_keys") if isinstance(doc, dict) else None
    if not isinstance(keys, list) or not keys:
        raise RuntimeError(f"No trusted operator keys enrolled in {trusted_keys_path}")
    validated: List[Dict[str, str]] = []
    for entry in keys:
        if not isinstance(entry, dict):
            raise RuntimeError("Malformed entry in trusted operator key store")
        ed_hex = str(entry.get("ed25519_public_key", ""))
        ml_hex = str(entry.get("ml_dsa_public_key", ""))
        fp = compute_key_fingerprint(ed_hex, ml_hex)
        rec_fp = entry.get("key_fingerprint")
        if rec_fp and rec_fp != fp:
            raise RuntimeError(f"Key fingerprint mismatch in {trusted_keys_path}: {rec_fp} != {fp}")
        validated.append(
            {
                "operator_id": str(entry.get("operator_id", DEFAULT_OPERATOR_ID)),
                "key_fingerprint": fp,
                "ed25519_public_key": ed_hex,
                "ml_dsa_public_key": ml_hex,
            }
        )
    return validated


def format_approval_summary(
    *,
    action: str,
    scope: Union[str, Sequence[str]],
    summary: str,
    payload_sha256: str,
    operator_id: str,
    key_fingerprint: str,
    ttl_seconds: float = DEFAULT_GRANT_TTL_S,
) -> str:
    """Format a readable local operator approval card (no signature hex required from user)."""
    scope_list = _normalize_scope(scope)
    scope_text = " :: ".join(scope_list)
    return "\n".join(
        [
            "┌─ OSIRIS LOCAL OPERATOR GRANT REQUEST ──────────────────────────────",
            f"│ Action:      {action}",
            f"│ Scope:       {scope_text}",
            f"│ Summary:     {summary}",
            f"│ Digest:      {payload_sha256}",
            f"│ Signer:      {operator_id} ({key_fingerprint[:23]}…) · TTL {int(ttl_seconds)}s",
            "└────────────────────────────────────────────────────────────────────",
        ]
    )


class PQCEnclaveGate:
    """
    Hybrid Post-Quantum Signing & Verification Gate (Ed25519 + NIST FIPS 204 ML-DSA-65).

    - Signing uses local private keys only (`verifier_only=False`).
    - Verification uses explicitly enrolled `_trusted_keys` only — never untrusted keys
      carried inside a grant's `pqc_envelope`.
    """

    ML_DSA_65_PARAMS = FIPS204Parameters(algorithm=PQCAlgorithm.ML_DSA_65)

    def __init__(
        self,
        operator_id: str = DEFAULT_OPERATOR_ID,
        *,
        ed25519_private_key: Optional["Ed25519PrivateKey"] = None,
        mldsa65_private_key: Optional["MLDSA65PrivateKey"] = None,
        trusted_public_keys: Optional[Sequence[Dict[str, str]]] = None,
        consumed_ledger_path: Optional[str] = None,
        verifier_only: bool = False,
    ):
        if not _CRYPTO_AVAILABLE:
            raise RuntimeError(
                "pyca/cryptography >= 47 is required for real FIPS 204 ML-DSA-65 "
                "signatures; refusing to run with placeholder hashes"
            )
        self.operator_id = operator_id
        self.verifier_only = bool(verifier_only)
        self.consumed_ledger_path = consumed_ledger_path
        self._consumed_grants: Dict[str, str] = {}  # grant_id -> consumer_id
        self._consumed_nonces: Dict[str, str] = {}  # nonce -> consumer_id

        if self.verifier_only:
            if ed25519_private_key is not None or mldsa65_private_key is not None:
                raise ValueError("verifier_only=True cannot be combined with private keys")
            self._ed25519_private_key = None
            self._mldsa65_private_key = None
            self.key_fingerprint = "verifier-only"
        else:
            self._ed25519_private_key = ed25519_private_key or Ed25519PrivateKey.generate()
            self._mldsa65_private_key = mldsa65_private_key or MLDSA65PrivateKey.generate()
            pubs = self.public_keys_hex()
            self.key_fingerprint = compute_key_fingerprint(
                pubs["ed25519_public_key"], pubs["ml_dsa_public_key"]
            )

        self._trusted_keys: Dict[str, Tuple["Ed25519PublicKey", "MLDSA65PublicKey", Dict[str, str]]] = {}
        if trusted_public_keys is not None:
            for item in trusted_public_keys:
                ed_hex = str(item["ed25519_public_key"])
                ml_hex = str(item["ml_dsa_public_key"])
                fp = compute_key_fingerprint(ed_hex, ml_hex)
                if item.get("key_fingerprint") and item["key_fingerprint"] != fp:
                    raise ValueError(f"Trusted key fingerprint mismatch: {item['key_fingerprint']} != {fp}")
                ed_pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(ed_hex))
                ml_pub = MLDSA65PublicKey.from_public_bytes(bytes.fromhex(ml_hex))
                self._trusted_keys[fp] = (
                    ed_pub,
                    ml_pub,
                    {
                        "operator_id": str(item.get("operator_id", self.operator_id)),
                        "key_fingerprint": fp,
                        "ed25519_public_key": ed_hex,
                        "ml_dsa_public_key": ml_hex,
                    },
                )
        elif not self.verifier_only:
            pubs = self.public_keys_hex()
            assert self._ed25519_private_key is not None and self._mldsa65_private_key is not None
            self._trusted_keys[self.key_fingerprint] = (
                self._ed25519_private_key.public_key(),
                self._mldsa65_private_key.public_key(),
                {
                    "operator_id": self.operator_id,
                    "key_fingerprint": self.key_fingerprint,
                    **pubs,
                },
            )

    def __repr__(self) -> str:
        return (
            f"<PQCEnclaveGate operator_id={self.operator_id!r} "
            f"key_fingerprint={self.key_fingerprint!r} "
            f"trusted_keys={len(self._trusted_keys)}>"
        )

    # ------------------------------------------------------------------
    # Persistent local enclave bootstrap & pure verifier construction
    # ------------------------------------------------------------------

    @classmethod
    def load_or_init_local(
        cls,
        enclave_dir: Optional[str] = None,
        *,
        operator_id: str = DEFAULT_OPERATOR_ID,
        allow_init: bool = True,
    ) -> "PQCEnclaveGate":
        """Load the operator's local signing keys from `enclave_dir` (`~/.osiris/enclave`),
        initializing once on first use when `allow_init=True`. Never overwrites existing keys."""
        if not _CRYPTO_AVAILABLE:
            raise RuntimeError("pyca/cryptography >= 47 is required")
        base = os.path.abspath(os.path.expanduser(enclave_dir or default_enclave_dir()))
        os.makedirs(base, mode=0o700, exist_ok=True)
        if os.name == "posix":
            try:
                os.chmod(base, 0o700)
            except OSError:
                pass

        ed_path = os.path.join(base, "operator_ed25519.key")
        ml_path = os.path.join(base, "operator_mldsa65.key")
        trusted_path = os.path.join(base, "trusted_operator_keys.json")
        consumed_path = os.path.join(base, "consumed_grants.jsonl")

        ed_exists = os.path.exists(ed_path)
        ml_exists = os.path.exists(ml_path)
        if ed_exists != ml_exists:
            raise RuntimeError(
                f"Incomplete local operator keypair in {base}; refusing to overwrite existing key material"
            )

        if not ed_exists:
            if not allow_init:
                raise RuntimeError(f"No local operator signing keys initialized in {base}")
            ed_priv = Ed25519PrivateKey.generate()
            ml_priv = MLDSA65PrivateKey.generate()
            ed_seed = ed_priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
            ml_seed = ml_priv.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
            _write_exclusive_file(ed_path, ed_seed, 0o600)
            _write_exclusive_file(ml_path, ml_seed, 0o600)
        else:
            _check_private_key_permissions(ed_path)
            _check_private_key_permissions(ml_path)
            with open(ed_path, "rb") as fh:
                ed_seed = fh.read()
            with open(ml_path, "rb") as fh:
                ml_seed = fh.read()
            if len(ed_seed) != 32 or len(ml_seed) != 32:
                raise RuntimeError(
                    f"Corrupt local operator key seed length in {base}; refusing to replace existing keys"
                )
            ed_priv = Ed25519PrivateKey.from_private_bytes(ed_seed)
            ml_priv = MLDSA65PrivateKey.from_seed_bytes(ml_seed)

        ed_pub_hex = ed_priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
        ml_pub_hex = ml_priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
        fp = compute_key_fingerprint(ed_pub_hex, ml_pub_hex)

        # Initialize trust root ONLY if absent; never silently overwrite an existing trust store.
        if not os.path.exists(trusted_path):
            trust_doc = {
                "schema_version": 1,
                "trusted_keys": [
                    {
                        "operator_id": operator_id,
                        "key_fingerprint": fp,
                        "ed25519_public_key": ed_pub_hex,
                        "ml_dsa_public_key": ml_pub_hex,
                    }
                ],
            }
            _write_exclusive_file(
                trusted_path,
                (json.dumps(trust_doc, indent=2, sort_keys=True) + "\n").encode("utf-8"),
                0o644,
            )

        trusted_keys = load_trusted_public_keys(trusted_path)
        return cls(
            operator_id=operator_id,
            ed25519_private_key=ed_priv,
            mldsa65_private_key=ml_priv,
            trusted_public_keys=trusted_keys,
            consumed_ledger_path=consumed_path,
        )

    @classmethod
    def verifier_from_enclave_dir(
        cls,
        enclave_dir: Optional[str] = None,
        *,
        trusted_keys_path: Optional[str] = None,
        operator_id: str = DEFAULT_OPERATOR_ID,
    ) -> "PQCEnclaveGate":
        """Create a pure verifier (zero private keys loaded) pinned strictly to `trusted_operator_keys.json`."""
        base = os.path.abspath(os.path.expanduser(enclave_dir or default_enclave_dir()))
        tpath = trusted_keys_path or os.path.join(base, "trusted_operator_keys.json")
        cpath = os.path.join(base, "consumed_grants.jsonl")
        trusted_keys = load_trusted_public_keys(tpath)
        return cls(
            operator_id=operator_id,
            trusted_public_keys=trusted_keys,
            consumed_ledger_path=cpath,
            verifier_only=True,
        )

    # ------------------------------------------------------------------
    # Digests / canonical signing body
    # ------------------------------------------------------------------

    @staticmethod
    def compute_payload_digest(code_or_bytes: bytes | str) -> str:
        """Deterministic SHA-256 payload digest (SYS-021)."""
        if isinstance(code_or_bytes, str):
            code_or_bytes = code_or_bytes.encode("utf-8")
        return hashlib.sha256(code_or_bytes).hexdigest()

    @staticmethod
    def compute_shake256_prehash(payload_sha256: str, context: str = "OSIRIS_6.0_PQC") -> str:
        """Computes NIST FIPS 204 SHAKE-256 512-bit pre-hash."""
        data = f"{context}:{payload_sha256}".encode("utf-8")
        return hashlib.shake_256(data).hexdigest(64)

    @staticmethod
    def _grant_core_from_fields(
        *,
        sequence_index: int,
        prev_block_hash: str,
        issued_at: float,
        expires_at: float,
        event_type: str,
        operator_id: str,
        scope: List[str],
        payload_sha256: str,
        operation_schema_sha256: str,
        nonce: str,
        key_fingerprint: str,
    ) -> Dict[str, Any]:
        return {
            "event_type": event_type,
            "expires_at": expires_at,
            "issued_at": issued_at,
            "key_fingerprint": key_fingerprint,
            "nonce": nonce,
            "operation_schema_sha256": operation_schema_sha256,
            "operator_id": operator_id,
            "payload_sha256": payload_sha256,
            "prev_block_hash": prev_block_hash,
            "scope": scope,
            "sequence_index": sequence_index,
        }

    @staticmethod
    def signing_message(grant_core: Dict[str, Any], grant_id: str, prehash: str) -> bytes:
        """Deterministic canonical byte string signed by both Ed25519 and ML-DSA-65."""
        envelope = {
            "domain": "OSIRIS-HYBRID-GRANT-V2",
            "fips204_prehash": prehash,
            "grant_core": grant_core,
            "grant_id": grant_id,
        }
        return _canon_json(envelope).encode("utf-8")

    def public_keys_hex(self) -> Dict[str, str]:
        if self.verifier_only or self._ed25519_private_key is None or self._mldsa65_private_key is None:
            raise RuntimeError("Verifier-only PQCEnclaveGate has no local signing keypair")
        return {
            "ed25519_public_key": self._ed25519_private_key.public_key()
            .public_bytes(Encoding.Raw, PublicFormat.Raw)
            .hex(),
            "ml_dsa_public_key": self._mldsa65_private_key.public_key()
            .public_bytes(Encoding.Raw, PublicFormat.Raw)
            .hex(),
        }

    def trusted_key_records(self) -> List[Dict[str, str]]:
        return [dict(meta) for _, _, meta in self._trusted_keys.values()]

    # ------------------------------------------------------------------
    # Grant generation, readable approval workflow, and verification
    # ------------------------------------------------------------------

    def generate_hybrid_grant(
        self,
        sequence_index: int,
        prev_block_hash: str,
        event_type: str,
        payload_sha256: str,
        operation_schema: Dict[str, Any],
        *,
        scope: Union[str, Sequence[str]] = "default",
        ttl_seconds: float = DEFAULT_GRANT_TTL_S,
        issued_at: Optional[float] = None,
        approval_mode: str = "direct",
    ) -> Dict[str, Any]:
        """Sign a hybrid Ed25519 + ML-DSA-65 grant using the enclave's local private keys."""
        if self.verifier_only or self._ed25519_private_key is None or self._mldsa65_private_key is None:
            raise RuntimeError("Verifier-only PQCEnclaveGate cannot sign grants")
        if not _is_hex64(payload_sha256):
            raise ValueError("payload_sha256 must be a 64-character lowercase SHA-256 hex digest")
        if not isinstance(operation_schema, dict):
            raise ValueError("operation_schema must be a dict")
        if not isinstance(ttl_seconds, (int, float)) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")

        norm_scope = _normalize_scope(scope)
        ts = round(float(time.time() if issued_at is None else issued_at), 6)
        exp = round(ts + float(ttl_seconds), 6)
        nonce = f"OSIRIS-PQC-{secrets.token_hex(16)}"
        op_schema_sha = hashlib.sha256(_canon_json(operation_schema).encode("utf-8")).hexdigest()

        core = self._grant_core_from_fields(
            sequence_index=int(sequence_index),
            prev_block_hash=str(prev_block_hash),
            issued_at=ts,
            expires_at=exp,
            event_type=str(event_type),
            operator_id=self.operator_id,
            scope=norm_scope,
            payload_sha256=payload_sha256,
            operation_schema_sha256=op_schema_sha,
            nonce=nonce,
            key_fingerprint=self.key_fingerprint,
        )
        grant_id = "grant-" + hashlib.sha256(_canon_json(core).encode("utf-8")).hexdigest()[:32]
        prehash = self.compute_shake256_prehash(payload_sha256, context=str(event_type))
        msg = self.signing_message(core, grant_id, prehash)

        ed25519_signature = self._ed25519_private_key.sign(msg).hex()
        ml_dsa_signature = self._mldsa65_private_key.sign(msg).hex()
        pubs = self.public_keys_hex()

        return {
            "grant_id": grant_id,
            "sequence_index": int(sequence_index),
            "prev_block_hash": str(prev_block_hash),
            "timestamp": ts,
            "issued_at": ts,
            "expires_at": exp,
            "event_type": str(event_type),
            "operator_id": self.operator_id,
            "scope": norm_scope,
            "approval_mode": str(approval_mode),
            "key_fingerprint": self.key_fingerprint,
            "payload_sha256": payload_sha256,
            "operation_schema_sha256": op_schema_sha,
            "operation_schema": operation_schema,
            "nonce": nonce,
            "pqc_envelope": {
                "algorithm": PQCAlgorithm.HYBRID_ED25519_ML_DSA_65.value,
                "specification": "NIST FIPS 204",
                "fips204_prehash": prehash,
                "key_fingerprint": self.key_fingerprint,
                "ed25519_public_key": pubs["ed25519_public_key"],
                "ml_dsa_public_key": pubs["ml_dsa_public_key"],
                "ed25519_signature": ed25519_signature,
                "ml_dsa_signature": ml_dsa_signature,
                "lattice_parameters": {
                    "ring_degree": self.ML_DSA_65_PARAMS.n,
                    "modulus": self.ML_DSA_65_PARAMS.q,
                    "matrix_dimensions": f"{self.ML_DSA_65_PARAMS.k}x{self.ML_DSA_65_PARAMS.l}",
                },
            },
        }

    def request_approved_grant(
        self,
        *,
        action: str,
        scope: Union[str, Sequence[str]],
        summary: str,
        payload_sha256: str,
        operation_schema: Dict[str, Any],
        confirm: Callable[[str], bool],
        out: Callable[[str], None] = lambda s: None,
        prompt: Optional[str] = None,
        approved_by_flag: bool = False,
        sequence_index: int = 1,
        prev_block_hash: str = "0" * 64,
        ttl_seconds: float = DEFAULT_GRANT_TTL_S,
        issued_at: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Show human-readable action & scope, obtain explicit operator approval, and sign locally."""
        card = format_approval_summary(
            action=action,
            scope=scope,
            summary=summary,
            payload_sha256=payload_sha256,
            operator_id=self.operator_id,
            key_fingerprint=self.key_fingerprint,
            ttl_seconds=ttl_seconds,
        )
        out(card)
        if approved_by_flag:
            approval_mode = "operator-flag"
        else:
            ask = prompt or f"Approve `{action}` for digest {payload_sha256[:16]}…? [y/N] "
            if not bool(confirm(ask)):
                raise ApprovalDeniedError("operator did not approve requested action")
            approval_mode = "interactive"

        grant = self.generate_hybrid_grant(
            sequence_index=sequence_index,
            prev_block_hash=prev_block_hash,
            event_type=action,
            payload_sha256=payload_sha256,
            operation_schema=operation_schema,
            scope=scope,
            ttl_seconds=ttl_seconds,
            issued_at=issued_at,
            approval_mode=approval_mode,
        )
        ok, reason = self.verify_grant_integrity(
            grant,
            expected_payload_sha256=payload_sha256,
            expected_scope=scope,
            expected_event_type=action,
            now=issued_at,
            consume=False,
        )
        if not ok:
            raise GrantVerificationError(f"Locally signed grant failed trusted-key self-check: {reason}")
        return grant

    def _check_and_record_consumption(
        self,
        *,
        grant_id: str,
        nonce: str,
        payload_sha256: str,
        consumer_id: Optional[str],
        record_now: bool,
    ) -> Optional[str]:
        cid = consumer_id or f"once:{secrets.token_hex(8)}"

        prior_g = self._consumed_grants.get(grant_id)
        if prior_g is not None and (consumer_id is None or prior_g != consumer_id):
            return f"Grant replay detected: {grant_id} already consumed"
        prior_n = self._consumed_nonces.get(nonce)
        if prior_n is not None and (consumer_id is None or prior_n != consumer_id):
            return f"Grant nonce replay detected: {nonce} already consumed"

        if self.consumed_ledger_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.consumed_ledger_path)), exist_ok=True)
            try:
                with chain_lock.locked_tail(self.consumed_ledger_path) as (fh, last):
                    fh.seek(0)
                    already_recorded_for_cid = False
                    prev_hash = last["hash"] if isinstance(last, dict) and "hash" in last else "0" * 64
                    for raw_line in fh.read().splitlines():
                        if not raw_line.strip():
                            continue
                        try:
                            row = json.loads(raw_line.decode("utf-8"))
                        except ValueError:
                            return "Consumed-grants ledger is corrupted"
                        row_gid = row.get("grant_id")
                        row_nonce = row.get("nonce")
                        row_cid = row.get("consumer_id")
                        if row_gid == grant_id or row_nonce == nonce:
                            if consumer_id is not None and row_cid == consumer_id:
                                already_recorded_for_cid = True
                            else:
                                return f"Grant replay detected: {grant_id} already consumed"
                    if record_now and not already_recorded_for_cid:
                        body = {
                            "consumed_at": round(time.time(), 6),
                            "consumer_id": cid,
                            "grant_id": grant_id,
                            "nonce": nonce,
                            "payload_sha256": payload_sha256,
                            "prev": prev_hash,
                        }
                        entry = {
                            **body,
                            "hash": hashlib.sha256(_canon_json(body).encode("utf-8")).hexdigest(),
                        }
                        fh.seek(0, os.SEEK_END)
                        fh.write((_canon_json(entry) + "\n").encode("utf-8"))
                        fh.flush()
                        os.fsync(fh.fileno())
            except (OSError, chain_lock.ChainTailError, chain_lock.ChainLockTimeout) as exc:
                return f"Consumed-grants ledger error: {exc}"

        if record_now:
            self._consumed_grants[grant_id] = cid
            self._consumed_nonces[nonce] = cid
        return None

    def verify_grant_integrity(
        self,
        grant: Dict[str, Any],
        *,
        expected_payload_sha256: Optional[str] = None,
        expected_scope: Optional[Union[str, Sequence[str]]] = None,
        expected_event_type: Optional[str] = None,
        now: Optional[float] = None,
        consume: bool = False,
        check_replay: Optional[bool] = None,
        consumer_id: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Fail-closed cryptographic & policy verification against explicitly trusted public keys ONLY.
        """
        if not isinstance(grant, dict):
            return False, "Grant must be a dictionary"

        required_fields = (
            "grant_id",
            "sequence_index",
            "prev_block_hash",
            "issued_at",
            "expires_at",
            "event_type",
            "operator_id",
            "scope",
            "key_fingerprint",
            "payload_sha256",
            "operation_schema_sha256",
            "operation_schema",
            "nonce",
            "pqc_envelope",
        )
        for field_name in required_fields:
            if field_name not in grant:
                return False, f"Missing required grant field: {field_name}"

        if grant["operator_id"] != self.operator_id:
            return False, f"Unauthorized operator: {grant['operator_id']} != {self.operator_id}"

        env = grant["pqc_envelope"]
        if not isinstance(env, dict):
            return False, "Malformed pqc_envelope"
        if env.get("specification") != "NIST FIPS 204":
            return False, "Unsupported PQC specification; must be NIST FIPS 204"

        expected_prehash = self.compute_shake256_prehash(
            str(grant["payload_sha256"]), context=str(grant["event_type"])
        )
        if env.get("fips204_prehash") != expected_prehash:
            return False, "FIPS 204 SHAKE-256 prehash mismatch"

        if not env.get("ed25519_signature") or not env.get("ml_dsa_signature"):
            return False, "Incomplete hybrid cryptographic signatures"

        # Verify operation_schema digest and expected payload / event / scope bindings
        if not isinstance(grant["operation_schema"], dict):
            return False, "Grant operation_schema must be a dict"
        try:
            computed_schema_sha = hashlib.sha256(_canon_json(grant["operation_schema"]).encode("utf-8")).hexdigest()
        except (TypeError, ValueError) as exc:
            return False, f"Unserializable operation_schema: {exc}"
        if grant["operation_schema_sha256"] != computed_schema_sha:
            return False, "Operation schema SHA-256 mismatch (tampered operation_schema)"

        if expected_payload_sha256 is not None and grant["payload_sha256"] != expected_payload_sha256:
            return (
                False,
                f"Grant payload_sha256 mismatch: {grant['payload_sha256'][:16]}… != {expected_payload_sha256[:16]}…",
            )
        if expected_event_type is not None and grant["event_type"] != expected_event_type:
            return False, f"Grant event_type mismatch: {grant['event_type']!r} != {expected_event_type!r}"

        try:
            grant_scope = _normalize_scope(grant["scope"])
        except ValueError as exc:
            return False, f"Invalid grant scope: {exc}"
        if expected_scope is not None:
            exp_scope_norm = _normalize_scope(expected_scope)
            if grant_scope != exp_scope_norm:
                return False, f"Grant scope mismatch: {grant_scope!r} != {exp_scope_norm!r}"

        # Verify expiry window
        try:
            issued_at = float(grant["issued_at"])
            expires_at = float(grant["expires_at"])
        except (TypeError, ValueError):
            return False, "Invalid issued_at / expires_at timestamps"
        if expires_at <= issued_at:
            return False, "Invalid grant lifetime (expires_at <= issued_at)"
        check_now = float(time.time() if now is None else now)
        if check_now < issued_at - MAX_CLOCK_SKEW_S:
            return False, f"Grant is not yet valid (issued_at={issued_at}, now={check_now})"
        if check_now > expires_at:
            return False, f"Grant has expired (expires_at={expires_at}, now={check_now})"

        # Establish trust STRICTLY from self._trusted_keys (never from grant['pqc_envelope'])
        if not self._trusted_keys:
            return False, "No trusted operator public keys enrolled"
        key_fp = str(grant["key_fingerprint"])
        trusted_entry = self._trusted_keys.get(key_fp)
        if trusted_entry is None:
            return False, f"Cryptographic signature verification failed: untrusted signer key {key_fp}"
        ed_pub, ml_pub, trusted_meta = trusted_entry

        # Reject if envelope public keys were swapped or disagree with the pinned trusted key
        if env.get("ed25519_public_key") != trusted_meta["ed25519_public_key"]:
            return False, "Cryptographic signature verification failed: ed25519_public_key mismatch against trusted key"
        if env.get("ml_dsa_public_key") != trusted_meta["ml_dsa_public_key"]:
            return False, "Cryptographic signature verification failed: ml_dsa_public_key mismatch against trusted key"

        # Reconstruct canonical grant core & grant_id and verify BOTH signatures
        try:
            core = self._grant_core_from_fields(
                sequence_index=int(grant["sequence_index"]),
                prev_block_hash=str(grant["prev_block_hash"]),
                issued_at=issued_at,
                expires_at=expires_at,
                event_type=str(grant["event_type"]),
                operator_id=str(grant["operator_id"]),
                scope=grant_scope,
                payload_sha256=str(grant["payload_sha256"]),
                operation_schema_sha256=str(grant["operation_schema_sha256"]),
                nonce=str(grant["nonce"]),
                key_fingerprint=key_fp,
            )
            expected_grant_id = "grant-" + hashlib.sha256(_canon_json(core).encode("utf-8")).hexdigest()[:32]
            if grant["grant_id"] != expected_grant_id:
                return False, "Cryptographic signature verification failed: grant_id core digest mismatch"
            msg = self.signing_message(core, expected_grant_id, expected_prehash)
            ed_sig = bytes.fromhex(str(env["ed25519_signature"]))
            ml_sig = bytes.fromhex(str(env["ml_dsa_signature"]))
            ed_pub.verify(ed_sig, msg)
            ml_pub.verify(ml_sig, msg)
        except Exception as exc:
            return False, f"Cryptographic signature verification failed: {type(exc).__name__}"

        # Replay / single-use check (always enforced when consume=True or check_replay=True)
        enforce_replay = bool(consume) if check_replay is None else (bool(consume) or bool(check_replay))
        if enforce_replay:
            replay_err = self._check_and_record_consumption(
                grant_id=str(grant["grant_id"]),
                nonce=str(grant["nonce"]),
                payload_sha256=str(grant["payload_sha256"]),
                consumer_id=consumer_id,
                record_now=consume,
            )
            if replay_err is not None:
                return False, replay_err

        return True, "VERIFIED_HYBRID_PQC_OK"
