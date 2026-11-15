"""F-16: no OSIRIS process can create, enroll or self-supply the Node Beta operator key.

Each test runs with HOME and the key-path override pointed at tmp_path. Test keys are made
here, in tmp_path, and enrolled only in a tmp_path trust root."""

import dataclasses
import hashlib
import os
import secrets
import stat
from pathlib import Path

import pytest

pytest.importorskip("osiris_governance.contracts")

from osiris_cli import node_beta_governance as nbg
from osiris_governance.contracts import AuthorityPermit, PermitAuthority

DIGEST = "ab" * 32
SCOPE = "OSIRIS_TEST_SCOPE_V1"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """No test can see or touch a real keystore."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv(nbg.KEY_FILE_ENV, raising=False)
    monkeypatch.setattr(nbg, "KEY_DIR", home / ".osiris" / "keystore")
    monkeypatch.setattr(nbg, "KEY_FILE", home / ".osiris" / "keystore" / "node_beta_operator.key")
    monkeypatch.setattr(nbg, "TRUST_ROOT", tmp_path / "empty_operator_keys")
    return home


def provision(directory: Path, name="operator.key", mode=0o600, key=None) -> Path:
    """What the owner does outside OSIRIS."""
    path = directory / name
    path.write_bytes(key if key is not None else secrets.token_bytes(32))
    os.chmod(path, mode)
    return path


def enroll(trust_root: Path, *keys: Path) -> Path:
    lines = ["# test trust root"] + [nbg.operator_key_fingerprint(k.read_bytes()) for k in keys]
    trust_root.write_text("\n".join(lines) + "\n")
    return trust_root


@pytest.fixture
def owner(tmp_path):
    key = provision(tmp_path)
    root = enroll(tmp_path / "operator_keys", key)
    return nbg.NodeBetaGovernor(key_path=key, trust_root=root), key, root


# --- no self-authorization -------------------------------------------------------------------

def test_module_has_no_key_creating_entry_point():
    assert not hasattr(nbg, "get_or_create_operator_key")
    source = Path(nbg.__file__).read_text()
    for forbidden in ("urandom", "token_bytes", "write_bytes", "chmod", "mkdir"):
        assert forbidden not in source, forbidden


def test_missing_key_refuses_and_creates_nothing(isolated):
    with pytest.raises(nbg.OperatorAuthorityError) as e:
        nbg.NodeBetaGovernor()
    assert e.value.reason == "OPERATOR_KEY_MISSING"
    assert not (isolated / ".osiris").exists()


def test_sign_payload_grant_without_owner_key_cannot_issue(isolated):
    with pytest.raises(nbg.OperatorAuthorityError):
        nbg.sign_payload_grant(DIGEST)
    assert not (isolated / ".osiris").exists()


def test_untrusted_process_minting_its_own_key_cannot_self_authorize(tmp_path, monkeypatch, isolated):
    # An untrusted process writes a well-formed 0600 key at the default path itself...
    nbg.KEY_DIR.mkdir(parents=True)
    provision(nbg.KEY_DIR, nbg.KEY_FILE.name)
    with pytest.raises(nbg.OperatorAuthorityError) as e:
        nbg.NodeBetaGovernor()
    assert e.value.reason == "OPERATOR_KEY_NOT_ENROLLED"
    # ...or points the override at one: it is still unenrolled.
    monkeypatch.setenv(nbg.KEY_FILE_ENV, str(provision(tmp_path, "rogue.key")))
    with pytest.raises(nbg.OperatorAuthorityError) as e:
        nbg.sign_payload_grant(DIGEST)
    assert e.value.reason == "OPERATOR_KEY_NOT_ENROLLED"


def test_self_minted_permit_is_denied_by_the_enrolled_verifier(owner):
    gov, _, _ = owner
    rogue = PermitAuthority(key=secrets.token_bytes(32)).issue(
        issuer=nbg.DEFAULT_ISSUER, scope=SCOPE, target_digest=DIGEST)
    assert gov.verify_grant(rogue) == (False, "PERMIT_SIGNATURE_INVALID")


# --- unknown or unauthorized signing keys ----------------------------------------------------

def test_unknown_key_is_rejected_even_when_well_formed(tmp_path, owner):
    _, _, root = owner
    stranger = provision(tmp_path, "stranger.key")
    with pytest.raises(nbg.OperatorAuthorityError) as e:
        nbg.NodeBetaGovernor(key_path=stranger, trust_root=root)
    assert e.value.reason == "OPERATOR_KEY_NOT_ENROLLED"


def test_empty_or_missing_trust_root_enrolls_nothing(tmp_path):
    key = provision(tmp_path)
    empty = tmp_path / "empty"
    empty.write_text("# nothing enrolled\n\n")
    for root in (empty, tmp_path / "absent"):
        with pytest.raises(nbg.OperatorAuthorityError) as e:
            nbg.NodeBetaGovernor(key_path=key, trust_root=root)
        assert e.value.reason == "OPERATOR_KEY_NOT_ENROLLED"


def test_shipped_trust_root_enrolls_no_key():
    assert nbg.enrolled_fingerprints(Path(nbg.__file__).resolve().parents[1] / "governance" / "operator_keys") == frozenset()


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
@pytest.mark.parametrize("mode", [0o640, 0o604, 0o644, 0o666])
def test_group_or_world_readable_key_is_refused(tmp_path, mode):
    key = provision(tmp_path, mode=mode)
    root = enroll(tmp_path / "root", key)
    with pytest.raises(nbg.OperatorAuthorityError) as e:
        nbg.NodeBetaGovernor(key_path=key, trust_root=root)
    assert e.value.reason == "OPERATOR_KEY_PERMISSIONS"
    assert stat.S_IMODE(os.stat(key).st_mode) == mode  # not "repaired"


def test_symlinked_key_is_refused(tmp_path):
    key = provision(tmp_path)
    root = enroll(tmp_path / "root", key)
    link = tmp_path / "link.key"
    link.symlink_to(key)
    with pytest.raises(nbg.OperatorAuthorityError) as e:
        nbg.NodeBetaGovernor(key_path=link, trust_root=root)
    assert e.value.reason == "OPERATOR_KEY_SYMLINK"


def test_short_key_and_directory_are_refused(tmp_path):
    short = provision(tmp_path, key=b"x" * 31)
    with pytest.raises(nbg.OperatorAuthorityError) as e:
        nbg.read_operator_key(short)
    assert e.value.reason == "OPERATOR_KEY_TOO_SHORT"
    with pytest.raises(nbg.OperatorAuthorityError) as e:
        nbg.read_operator_key(tmp_path)
    assert e.value.reason == "OPERATOR_KEY_NOT_REGULAR_FILE"


# --- missing owner signature -> DENY ---------------------------------------------------------

def test_verify_without_an_owner_key_is_deny_not_an_error(owner, monkeypatch):
    gov, _, _ = owner
    permit = gov.issue_grant(DIGEST, SCOPE)
    # The default (unprovisioned) environment: verification fails closed.
    assert nbg.check_payload_grant(permit, DIGEST) == (False, "OPERATOR_KEY_MISSING")
    assert nbg.verify_payload_grant(permit, DIGEST) is False


@pytest.mark.parametrize("signature", ["", "0" * 64])
def test_unsigned_or_blank_signature_is_deny(owner, signature):
    gov, _, _ = owner
    permit = dataclasses.replace(gov.issue_grant(DIGEST, SCOPE), signature=signature)
    assert gov.verify_grant(permit) == (False, "PERMIT_SIGNATURE_INVALID")


@pytest.mark.parametrize("not_a_permit", [None, {}, "permit", 0])
def test_non_permit_is_deny(owner, not_a_permit):
    gov, _, _ = owner
    assert gov.verify_grant(not_a_permit) == (False, "PERMIT_MALFORMED")


def test_governance_module_missing_is_deny(owner, monkeypatch):
    gov, key, root = owner
    permit = gov.issue_grant(DIGEST, SCOPE)
    monkeypatch.setattr(nbg, "PermitAuthority", None)
    assert nbg.check_payload_grant(permit, key_path=key, trust_root=root) == (False, "GOVERNANCE_MODULE_MISSING")


# --- altered authorization payloads ----------------------------------------------------------

@pytest.mark.parametrize("field,value", [
    ("target_digest", "cd" * 32),
    ("scope", "PROMOTION_AUTHORITY_VERIFIED"),
    ("issuer", "someone else"),
    ("nonce", "00" * 16),
    ("issued_at_utc", "2000-01-01T00:00:00+00:00"),
    ("expires_at_utc", "2999-01-01T00:00:00+00:00"),
    ("policy_version", "tampered"),
])
def test_any_altered_field_is_rejected(owner, field, value):
    gov, _, _ = owner
    permit = gov.issue_grant(DIGEST, SCOPE)
    assert gov.verify_grant(permit) == (True, None)
    altered = dataclasses.replace(permit, **{field: value})
    assert gov.verify_grant(altered) == (False, "PERMIT_SIGNATURE_INVALID")


def test_expected_digest_and_scope_bind(owner):
    gov, key, root = owner
    permit = gov.issue_grant(DIGEST, SCOPE)
    kw = {"key_path": key, "trust_root": root}
    assert nbg.check_payload_grant(permit, DIGEST, SCOPE, **kw) == (True, "OK")
    assert nbg.check_payload_grant(permit, "cd" * 32, **kw) == (False, "PERMIT_TARGET_MISMATCH")
    assert nbg.check_payload_grant(permit, DIGEST, "OTHER", **kw) == (False, "PERMIT_SCOPE_MISMATCH")


def test_expired_permit_is_rejected(owner):
    gov, _, _ = owner
    permit = gov.issue_grant(DIGEST, SCOPE, ttl_seconds=-1)
    assert gov.verify_grant(permit) == (False, "PERMIT_EXPIRED_OR_NOT_YET_VALID")


# --- the legitimate, owner-provisioned path still works --------------------------------------

def test_owner_provisioned_enrolled_key_issues_and_verifies(owner, monkeypatch):
    gov, key, root = owner
    permit = gov.issue_grant(DIGEST, SCOPE)
    assert isinstance(permit, AuthorityPermit)
    assert gov.verify_grant(permit) == (True, None)
    # The module-level helpers, as the experiment scripts call them, via the path override.
    monkeypatch.setenv(nbg.KEY_FILE_ENV, str(key))
    monkeypatch.setattr(nbg, "TRUST_ROOT", root)
    p2 = nbg.sign_payload_grant(DIGEST, scope=SCOPE)
    assert nbg.verify_payload_grant(p2, DIGEST, SCOPE) is True
    assert oct(os.stat(key).st_mode & 0o777) == "0o600"


def test_rotation_requires_reenrollment(tmp_path, owner):
    gov, key, root = owner
    old = gov.issue_grant(DIGEST, SCOPE)
    new_key = provision(tmp_path, "new.key")
    enroll(root, new_key)  # the reviewed commit replaces the old fingerprint
    new_gov = nbg.NodeBetaGovernor(key_path=new_key, trust_root=root)
    assert new_gov.verify_grant(old) == (False, "PERMIT_SIGNATURE_INVALID")
    with pytest.raises(nbg.OperatorAuthorityError):
        nbg.NodeBetaGovernor(key_path=key, trust_root=root)


def test_fingerprint_cli_reads_only(tmp_path, capsys, isolated):
    key = provision(tmp_path)
    before = key.read_bytes()
    assert nbg._main(["fingerprint", str(key)]) == 0
    out = capsys.readouterr().out
    assert out.startswith(nbg.operator_key_fingerprint(before)) and "NOT enrolled" in out
    assert key.read_bytes() == before
    assert nbg._main(["fingerprint"]) == 1  # default path absent
    assert nbg._main(["self-test"]) == 1
    assert not (isolated / ".osiris").exists()


def test_fingerprint_is_domain_separated_and_does_not_contain_the_key():
    key = secrets.token_bytes(32)
    fp = nbg.operator_key_fingerprint(key)
    assert key.hex() not in fp and fp != "sha256:" + hashlib.sha256(key).hexdigest()
