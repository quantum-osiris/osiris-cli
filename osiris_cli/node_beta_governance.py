#!/usr/bin/env python3
"""
OSIRIS Node Beta Standalone Governance Adapter
===============================================
Signs and verifies HMAC AuthorityPermits on Node Beta (WSL2) without a physical JavaCard or
NFC terminal.

Complies with:
- SYS-021: Zero Self-Supplied Authorization (Payload-bound cryptographic receipts)
- SYS-022: Provenance Tracking
- SYS-023: Merkle Ledger Integrity

Authority model (F-16)
----------------------
This module never creates, rotates or enrolls a key. A process that could mint its own key
could also approve its own permits, and SYS-021 forbids that. Two things must therefore come
from the owner, outside any OSIRIS process:

1. **The permit key.** A typed key file: the line ``OSIRIS-PERMIT-HMAC-KEY-V1`` followed by a
   line of 64 or more hex digits (32+ random bytes). It must be a regular file (not a symlink)
   that the current user owns, with mode 0600 or stricter. The default path is
   ``~/.osiris/keystore/node_beta_permit_hmac.key``; ``$OSIRIS_OPERATOR_KEY_FILE`` overrides it.
2. **Its enrollment.** The key's fingerprint, ``python -m osiris_cli.node_beta_governance
   fingerprint``, must appear in ``governance/operator_keys``. That file is changed only by a
   reviewed commit.

If either is missing, malformed or does not match, every issue fails and every verify returns
DENY. Once both are enrolled, a permit verifies only if every byte it covers (issuer, scope,
target digest, nonce, validity window and policy version) is unchanged and it is within its
validity window.

Key separation (F-21): ``node_beta_operator.key`` is the raw Ed25519 signing seed that the QPU
dispatch scripts read. It is never accepted here, under any path, and a raw (untyped) key file
is refused, so a signing seed cannot double as the shared permit secret. The typed file is not
32 bytes long, so it cannot be loaded as an Ed25519 seed either.

Import resolution (F-22): the permit contracts come only from the installed
``osiris_governance`` package. This module does not add a sibling checkout to ``sys.path``; if
the package is not installed, every issue fails and every verify is DENY.

Limits: HMAC is symmetric, so any process that can read the key can both issue and verify.
Same-user isolation therefore rests on the key file's permissions and on who may change
``governance/operator_keys``. For production execution and GitHub writes, the authority is the
owner's ssh-signed, single-use grant (``osiris_cli.grants``), not this module.
"""

from __future__ import annotations

import hashlib
import os
import stat
import sys
from pathlib import Path
from typing import FrozenSet, Optional, Tuple

# The permit contracts come from the installed osiris_governance only. sys.path is not changed
# here: a sibling checkout must never take precedence over the installed authority code (F-22).
try:
    from osiris_governance import contracts as _contracts
    from osiris_governance.contracts import PermitAuthority, AuthorityPermit, PROMOTION_SCOPE, VERIFICATION_SCOPE
    GOVERNANCE_ORIGIN: Optional[str] = getattr(_contracts, "__file__", None)
except ImportError:
    PermitAuthority = None
    AuthorityPermit = None
    GOVERNANCE_ORIGIN = None
    PROMOTION_SCOPE = "PROMOTION_AUTHORITY_VERIFIED"
    VERIFICATION_SCOPE = "VERIFICATION_AUTHORITY_VERIFIED"

KEY_DIR = Path.home() / ".osiris" / "keystore"
KEY_FILE = KEY_DIR / "node_beta_permit_hmac.key"
SIGNING_SEED_NAME = "node_beta_operator.key"   # raw Ed25519 seed used by the QPU dispatch scripts; never an HMAC key
KEY_FILE_HEADER = "OSIRIS-PERMIT-HMAC-KEY-V1"
MAX_KEY_FILE_BYTES = 4096
KEY_FILE_ENV = "OSIRIS_OPERATOR_KEY_FILE"
TRUST_ROOT = Path(__file__).resolve().parents[1] / "governance" / "operator_keys"
DEFAULT_ISSUER = "Devin Phillip Davis / Node Beta (Standalone)"
MIN_KEY_BYTES = 32
_FINGERPRINT_DOMAIN = b"osiris-node-beta-operator-key/v1\x00"


class OperatorAuthorityError(Exception):
    """No usable owner-provisioned, enrolled operator key. ``reason`` is a stable code."""

    def __init__(self, reason: str, message: str):
        super().__init__(f"{reason}: {message}")
        self.reason = reason


def operator_key_fingerprint(key: bytes) -> str:
    """The enrollment fingerprint of a key: a domain-separated SHA-256. The key cannot be recovered from it."""
    return "sha256:" + hashlib.sha256(_FINGERPRINT_DOMAIN + key).hexdigest()


def enrolled_fingerprints(trust_root: Optional[os.PathLike] = None) -> FrozenSet[str]:
    """Fingerprints enrolled by a reviewed commit. A missing or unreadable trust root enrolls nothing."""
    try:
        text = Path(trust_root or TRUST_ROOT).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return frozenset()
    out = set()
    for line in text.splitlines():
        token = line.split("#", 1)[0].strip()
        if token:
            out.add(token.split()[0])
    return frozenset(out)


def _key_path(path: Optional[os.PathLike]) -> Path:
    if path is not None:
        return Path(path)
    env = os.environ.get(KEY_FILE_ENV)
    return Path(env) if env else KEY_FILE


def _decode_key_file(p: Path, raw: bytes) -> bytes:
    """The key bytes of a typed permit-key file. Anything else, including a raw 32-byte seed, is refused."""
    wrong = OperatorAuthorityError("OPERATOR_KEY_WRONG_TYPE",
                                   f"{p} is not a {KEY_FILE_HEADER} key file; raw key or seed files are not accepted")
    if len(raw) > MAX_KEY_FILE_BYTES:
        raise wrong
    try:
        lines = raw.decode("ascii").split("\n")
    except UnicodeDecodeError:
        raise wrong
    if lines and lines[-1] == "":
        lines.pop()
    if len(lines) != 2 or lines[0] != KEY_FILE_HEADER:
        raise wrong
    hexkey = lines[1]
    if len(hexkey) % 2 or any(c not in "0123456789abcdef" for c in hexkey):
        raise wrong
    return bytes.fromhex(hexkey)


def read_operator_key(path: Optional[os.PathLike] = None) -> bytes:
    """Read an owner-provisioned key without creating, repairing or re-permissioning anything."""
    p = _key_path(path)
    if p.name == SIGNING_SEED_NAME:
        raise OperatorAuthorityError("OPERATOR_KEY_RESERVED_FOR_SIGNING",
                                     f"{p.name} is the Ed25519 signing seed; the permit HMAC key must be a separate file")
    provision = (f"the owner must provision it outside OSIRIS (a {KEY_FILE_HEADER} file, mode 0600) "
                 "and enroll its fingerprint in governance/operator_keys by a reviewed commit")
    try:
        st = os.lstat(p)
    except FileNotFoundError:
        raise OperatorAuthorityError("OPERATOR_KEY_MISSING", f"no operator key at {p}; {provision}")
    except OSError as e:
        raise OperatorAuthorityError("OPERATOR_KEY_UNREADABLE", f"{p}: {e.strerror}")
    if stat.S_ISLNK(st.st_mode):
        raise OperatorAuthorityError("OPERATOR_KEY_SYMLINK", f"{p} is a symlink; refusing to follow it")
    if not stat.S_ISREG(st.st_mode):
        raise OperatorAuthorityError("OPERATOR_KEY_NOT_REGULAR_FILE", f"{p} is not a regular file")
    if os.name == "posix":
        if st.st_uid != os.geteuid():
            raise OperatorAuthorityError("OPERATOR_KEY_WRONG_OWNER", f"{p} is not owned by the current user")
        if st.st_mode & 0o077:
            raise OperatorAuthorityError("OPERATOR_KEY_PERMISSIONS",
                                         f"{p} has mode {stat.S_IMODE(st.st_mode):04o}; it must be 0600 or stricter")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(p, flags)
    except OSError as e:
        raise OperatorAuthorityError("OPERATOR_KEY_UNREADABLE", f"{p}: {e.strerror}")
    try:
        fst = os.fstat(fd)
        if (fst.st_dev, fst.st_ino) != (st.st_dev, st.st_ino):
            raise OperatorAuthorityError("OPERATOR_KEY_REPLACED", f"{p} changed while it was being opened")
        with os.fdopen(fd, "rb") as fh:
            fd = -1
            raw = fh.read(MAX_KEY_FILE_BYTES + 1)
    finally:
        if fd != -1:
            os.close(fd)
    key = _decode_key_file(p, raw)
    if len(key) < MIN_KEY_BYTES:
        raise OperatorAuthorityError("OPERATOR_KEY_TOO_SHORT",
                                     f"{p} holds {len(key)} bytes; at least {MIN_KEY_BYTES} are required")
    return key


def load_operator_key(path: Optional[os.PathLike] = None, *, trust_root: Optional[os.PathLike] = None) -> bytes:
    """The operator key, only if the owner provisioned it and its fingerprint is enrolled."""
    key = read_operator_key(path)
    fingerprint = operator_key_fingerprint(key)
    if fingerprint not in enrolled_fingerprints(trust_root):
        raise OperatorAuthorityError("OPERATOR_KEY_NOT_ENROLLED",
                                     f"key {fingerprint[:23]}… is not enrolled in {trust_root or TRUST_ROOT}")
    return key


class NodeBetaGovernor:
    """Local governance signer for Node Beta standalone operations.

    Construction fails with OperatorAuthorityError unless the governance package is importable and
    the operator key is owner-provisioned and enrolled."""

    def __init__(self, issuer: str = DEFAULT_ISSUER, *, key_path: Optional[os.PathLike] = None,
                 trust_root: Optional[os.PathLike] = None):
        if PermitAuthority is None:
            raise OperatorAuthorityError("GOVERNANCE_MODULE_MISSING", "osiris_governance is not importable")
        self.issuer = issuer
        key = load_operator_key(key_path, trust_root=trust_root)
        self.fingerprint = operator_key_fingerprint(key)
        self.authority = PermitAuthority(key=key)

    def issue_grant(self, target_digest: str, scope: str = PROMOTION_SCOPE, ttl_seconds: int = 86400) -> AuthorityPermit:
        """Issue an AuthorityPermit for a given payload SHA-256 digest."""
        return self.authority.issue(
            issuer=self.issuer,
            scope=scope,
            target_digest=target_digest,
            ttl_seconds=ttl_seconds
        )

    def verify_grant(self, permit: AuthorityPermit) -> Tuple[bool, Optional[str]]:
        """Verify that an AuthorityPermit is authentic and current."""
        if AuthorityPermit is None or not isinstance(permit, AuthorityPermit):
            return False, "PERMIT_MALFORMED"
        reason = self.authority.check(permit=permit)
        return (reason is None), reason


def sign_payload_grant(target_digest: str, scope: str = PROMOTION_SCOPE, ttl_seconds: int = 86400) -> AuthorityPermit:
    """Issue an AuthorityPermit with the enrolled Node Beta key. Raises OperatorAuthorityError without one."""
    gov = NodeBetaGovernor()
    return gov.issue_grant(target_digest=target_digest, scope=scope, ttl_seconds=ttl_seconds)


def check_payload_grant(permit: AuthorityPermit, expected_digest: Optional[str] = None,
                        expected_scope: Optional[str] = None, **governor_kwargs) -> Tuple[bool, str]:
    """(True, "OK") or (False, reason). Every failure, including a missing or unenrolled key, is a DENY."""
    try:
        gov = NodeBetaGovernor(**governor_kwargs)
    except OperatorAuthorityError as e:
        return False, e.reason
    valid, reason = gov.verify_grant(permit)
    if not valid:
        return False, reason or "PERMIT_INVALID"
    if expected_digest is not None and permit.target_digest != expected_digest:
        return False, "PERMIT_TARGET_MISMATCH"
    if expected_scope is not None and permit.scope != expected_scope:
        return False, "PERMIT_SCOPE_MISMATCH"
    return True, "OK"


def verify_payload_grant(permit: AuthorityPermit, expected_digest: Optional[str] = None,
                         expected_scope: Optional[str] = None) -> bool:
    """Convenience helper to verify an AuthorityPermit. Fails closed."""
    return check_payload_grant(permit, expected_digest, expected_scope)[0]


def _main(argv) -> int:
    if argv[:1] == ["fingerprint"]:
        try:
            key = read_operator_key(argv[1] if len(argv) > 1 else None)
        except OperatorAuthorityError as e:
            print(f"refused: {e}", file=sys.stderr)
            return 1
        fp = operator_key_fingerprint(key)
        state = "enrolled" if fp in enrolled_fingerprints() else "NOT enrolled"
        print(f"{fp}  ({state} in {TRUST_ROOT})")
        return 0
    if argv[:1] == ["self-test"]:
        try:
            gov = NodeBetaGovernor()
        except OperatorAuthorityError as e:
            print(f"refused: {e}", file=sys.stderr)
            return 1
        grant = gov.issue_grant("e21d8231cf3c53efcebb76391104cd6b67193527", ttl_seconds=60)
        valid, err = gov.verify_grant(grant)
        print(f"NodeBetaGovernor self-test: valid={valid}, err={err}")
        return 0 if valid else 1
    print("usage: python -m osiris_cli.node_beta_governance {fingerprint [KEYFILE] | self-test}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
