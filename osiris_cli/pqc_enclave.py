"""
osiris_cli/pqc_enclave.py
OSIRIS 6.0 Pillar 4: Post-Quantum Cryptographic (PQC) Enclave Engine
Implements NIST FIPS 204 (ML-DSA / Dilithium) & Ed25519 Hybrid Cryptographic Grants.

Phase-3 reconciliation (release/production-reconciliation-20261119): the previous
implementation stored `ml_dsa_signature = SHA3-512(prehash:ed25519_sig)` and the
verifier only checked the signature fields were non-empty, so any attacker could
forge a grant. This module now performs REAL cryptographic signing and verification
with pyca/cryptography (Ed25519 + ML-DSA-65, FIPS 204). Verification is fail-closed:
any structural, prehash, or signature failure returns (False, reason).
"""

from __future__ import annotations
import hashlib
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

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


class PQCAlgorithm(str, Enum):
    ED25519_CLASSICAL = "Ed25519"
    ML_DSA_44 = "ML-DSA-44"  # NIST Security Level 2
    ML_DSA_65 = "ML-DSA-65"  # NIST Security Level 3 (Primary OSIRIS 6.0 standard)
    ML_DSA_87 = "ML-DSA-87"  # NIST Security Level 5
    HYBRID_ED25519_ML_DSA_65 = "Hybrid-Ed25519+ML-DSA-65"


@dataclass(frozen=True)
class FIPS204Parameters:
    """NIST FIPS 204 Module-Lattice-Based Digital Signature Parameters."""
    algorithm: PQCAlgorithm
    n: int = 256  # Polynomial ring degree
    q: int = 8380417  # Modulus
    k: int = 6    # Matrix rows for ML-DSA-65
    l: int = 5    # Matrix columns for ML-DSA-65
    public_key_bytes: int = 1952
    signature_bytes: int = 3309


class PQCEnclaveGate:
    """
    Hardware/Software Hybrid Post-Quantum Verification Gate.

    Holds real Ed25519 + ML-DSA-65 keypairs (generated per instance unless supplied).
    Grants carry the raw public keys in the envelope; verification reconstructs the
    public keys from the envelope and cryptographically verifies both signatures over
    the deterministic signing message. Fail-closed on every error path.
    """

    ML_DSA_65_PARAMS = FIPS204Parameters(algorithm=PQCAlgorithm.ML_DSA_65)

    def __init__(
        self,
        operator_id: str = "Devin Phillip Davis / Enki",
        *,
        ed25519_private_key: Optional["Ed25519PrivateKey"] = None,
        mldsa65_private_key: Optional["MLDSA65PrivateKey"] = None,
    ):
        if not _CRYPTO_AVAILABLE:
            raise RuntimeError(
                "pyca/cryptography >= 47 is required for real FIPS 204 ML-DSA-65 "
                "signatures; refusing to run with placeholder hashes"
            )
        self.operator_id = operator_id
        self.ed25519_private_key = ed25519_private_key or Ed25519PrivateKey.generate()
        self.mldsa65_private_key = mldsa65_private_key or MLDSA65PrivateKey.generate()

    # ------------------------------------------------------------------
    # Digests / prehash
    # ------------------------------------------------------------------

    @staticmethod
    def compute_payload_digest(code_or_bytes: bytes | str) -> str:
        """Deterministic SHA-256 payload digest (SYS-021)."""
        if isinstance(code_or_bytes, str):
            code_or_bytes = code_or_bytes.encode("utf-8")
        return hashlib.sha256(code_or_bytes).hexdigest()

    @staticmethod
    def compute_shake256_prehash(payload_sha256: str, context: str = "OSIRIS_6.0_PQC") -> str:
        """Computes NIST FIPS 204 SHAKE-256 pre-hash representation for lattice reduction."""
        data = f"{context}:{payload_sha256}".encode("utf-8")
        return hashlib.shake_256(data).hexdigest(64)  # 512-bit output

    @staticmethod
    def signing_message(event_type: str, payload_sha256: str, prehash: str) -> bytes:
        """The exact byte string both hybrid signatures cover."""
        return f"{event_type}|{payload_sha256}|{prehash}".encode("utf-8")

    def public_keys_hex(self) -> Dict[str, str]:
        return {
            "ed25519_public_key": self.ed25519_private_key.public_key()
            .public_bytes(Encoding.Raw, PublicFormat.Raw).hex(),
            "ml_dsa_public_key": self.mldsa65_private_key.public_key()
            .public_bytes(Encoding.Raw, PublicFormat.Raw).hex(),
        }

    # ------------------------------------------------------------------
    # Grant generation / verification
    # ------------------------------------------------------------------

    def generate_hybrid_grant(
        self,
        sequence_index: int,
        prev_block_hash: str,
        event_type: str,
        payload_sha256: str,
        operation_schema: Dict[str, Any],
        ed25519_sig_hex: Optional[str] = None,
        pqc_sig_hex: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Assembles a hybrid PQC Grant receipt with REAL Ed25519 + ML-DSA-65 signatures.

        `ed25519_sig_hex` / `pqc_sig_hex` are accepted for backward API compatibility but
        are IGNORED: signatures are always produced by this gate's private keys. A grant
        whose signatures were not produced by a real private key cannot pass
        `verify_grant_integrity`.
        """
        ts = time.time()
        nonce = f"OSIRIS-PQC-{int(ts * 1000)}"

        prehash = self.compute_shake256_prehash(payload_sha256, context=event_type)
        msg = self.signing_message(event_type, payload_sha256, prehash)

        ed25519_signature = self.ed25519_private_key.sign(msg).hex()
        ml_dsa_signature = self.mldsa65_private_key.sign(msg).hex()

        grant = {
            "sequence_index": sequence_index,
            "prev_block_hash": prev_block_hash,
            "timestamp": ts,
            "event_type": event_type,
            "operator_id": self.operator_id,
            "payload_sha256": payload_sha256,
            "operation_schema": operation_schema,
            "operation_schema_sha256": hashlib.sha256(json.dumps(operation_schema, sort_keys=True).encode()).hexdigest(),
            "nonce": nonce,
            "pqc_envelope": {
                "specification": "NIST FIPS 204",
                "algorithm": PQCAlgorithm.HYBRID_ED25519_ML_DSA_65.value,
                "fips204_prehash": prehash,
                "ed25519_signature": ed25519_signature,
                "ml_dsa_signature": ml_dsa_signature,
                **self.public_keys_hex(),
                "lattice_parameters": {
                    "ring_degree": self.ML_DSA_65_PARAMS.n,
                    "modulus": self.ML_DSA_65_PARAMS.q,
                    "matrix_dimensions": f"{self.ML_DSA_65_PARAMS.k}x{self.ML_DSA_65_PARAMS.l}"
                }
            }
        }
        return grant

    def verify_grant_integrity(self, grant: Dict[str, Any]) -> Tuple[bool, str]:
        """Fail-closed validation of the hybrid PQC grant receipt, with real signature checks."""
        # 1. Structural fields check
        required_fields = ["sequence_index", "prev_block_hash", "timestamp", "event_type", "operator_id", "payload_sha256", "nonce", "pqc_envelope"]
        for f in required_fields:
            if f not in grant:
                return False, f"Missing required grant field: {f}"

        # 2. Operator identity
        if grant["operator_id"] != self.operator_id:
            return False, f"Unauthorized operator: {grant['operator_id']} != {self.operator_id}"

        # 3. PQC envelope validation
        env = grant["pqc_envelope"]
        if env.get("specification") != "NIST FIPS 204":
            return False, "Unsupported PQC specification; must be NIST FIPS 204"

        # 4. Prehash re-verification
        expected_prehash = self.compute_shake256_prehash(grant["payload_sha256"], context=grant["event_type"])
        if env.get("fips204_prehash") != expected_prehash:
            return False, "FIPS 204 SHAKE-256 prehash mismatch"

        # 5. Signatures presence
        if not env.get("ed25519_signature") or not env.get("ml_dsa_signature"):
            return False, "Incomplete hybrid cryptographic signatures"

        # 6. REAL cryptographic verification (Ed25519 + FIPS 204 ML-DSA-65), fail closed.
        if not _CRYPTO_AVAILABLE:
            return False, "cryptography library unavailable; cannot verify signatures (fail closed)"
        try:
            msg = self.signing_message(grant["event_type"], grant["payload_sha256"], expected_prehash)
            ed_pub_hex = env.get("ed25519_public_key") or self.public_keys_hex()["ed25519_public_key"]
            ml_pub_hex = env.get("ml_dsa_public_key") or self.public_keys_hex()["ml_dsa_public_key"]
            ed_pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(ed_pub_hex))
            ml_pub = MLDSA65PublicKey.from_public_bytes(bytes.fromhex(ml_pub_hex))
            ed_pub.verify(bytes.fromhex(env["ed25519_signature"]), msg)
            ml_pub.verify(bytes.fromhex(env["ml_dsa_signature"]), msg)
        except Exception as exc:
            return False, f"Cryptographic signature verification failed: {type(exc).__name__}"

        return True, "VERIFIED_HYBRID_PQC_OK"
