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
    monkeypatch.setattr(nbg, "KEY_FILE", home / ".osiris" / "keystore" / "node_beta_permit_hmac.key")
    monkeypatch.setattr(nbg, "TRUST_ROOT", tmp_path / "empty_operator_keys")
    return home


def provision(directory: Path, name="operator.key", mode=0o600, key=None) -> Path:
    """What the owner does outside OSIRIS."""
    path = directory / name
    key = key if key is not None else secrets.token_bytes(32)
    path.write_text(f"{nbg.KEY_FILE_HEADER}\n{key.hex()}\n")
    os.chmod(path, mode)
    return path


def key_bytes(path: Path) -> bytes:
    return bytes.fromhex(path.read_text().split("\n")[1])


def enroll(trust_root: Path, *keys: Path) -> Path:
    lines = ["# test trust root"] + [nbg.operator_key_fingerprint(key_bytes(k)) for k in keys]
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
    assert out.startswith(nbg.operator_key_fingerprint(key_bytes(key))) and "NOT enrolled" in out
    assert key.read_bytes() == before
    assert nbg._main(["fingerprint"]) == 1  # default path absent
    assert nbg._main(["self-test"]) == 1
    assert not (isolated / ".osiris").exists()


def test_fingerprint_is_domain_separated_and_does_not_contain_the_key():
    key = secrets.token_bytes(32)
    fp = nbg.operator_key_fingerprint(key)
    assert key.hex() not in fp and fp != "sha256:" + hashlib.sha256(key).hexdigest()


# --- F-21: the signing seed and the permit HMAC key are different keys ------------------------

def test_default_permit_key_is_not_the_signing_seed_file():
    import importlib
    fresh = importlib.reload(importlib.import_module("osiris_cli.node_beta_governance"))
    assert fresh.KEY_FILE.name == "node_beta_permit_hmac.key" != fresh.SIGNING_SEED_NAME
    assert fresh.SIGNING_SEED_NAME == "node_beta_operator.key"


def test_signing_seed_filename_is_refused_whatever_it_holds(tmp_path, monkeypatch):
    # Even a correctly typed, enrolled key is refused under the Ed25519 seed's name...
    seed_named = provision(tmp_path, nbg.SIGNING_SEED_NAME)
    root = enroll(tmp_path / "root", seed_named)
    for attempt in (lambda: nbg.NodeBetaGovernor(key_path=seed_named, trust_root=root),
                    lambda: nbg.read_operator_key(seed_named)):
        with pytest.raises(nbg.OperatorAuthorityError) as e:
            attempt()
        assert e.value.reason == "OPERATOR_KEY_RESERVED_FOR_SIGNING"
    # ...and through the environment override.
    monkeypatch.setenv(nbg.KEY_FILE_ENV, str(seed_named))
    monkeypatch.setattr(nbg, "TRUST_ROOT", root)
    with pytest.raises(nbg.OperatorAuthorityError) as e:
        nbg.sign_payload_grant(DIGEST)
    assert e.value.reason == "OPERATOR_KEY_RESERVED_FOR_SIGNING"


def test_raw_ed25519_seed_is_refused_as_a_permit_key(tmp_path):
    seed = secrets.token_bytes(32)                      # what the QPU dispatch scripts read
    raw = tmp_path / "copied_seed.key"
    raw.write_bytes(seed)
    os.chmod(raw, 0o600)
    root = tmp_path / "root"
    root.write_text(nbg.operator_key_fingerprint(seed) + "\n")   # even if someone enrolled it
    with pytest.raises(nbg.OperatorAuthorityError) as e:
        nbg.NodeBetaGovernor(key_path=raw, trust_root=root)
    assert e.value.reason == "OPERATOR_KEY_WRONG_TYPE"


@pytest.mark.parametrize("content", [
    b"", b"OSIRIS-PERMIT-HMAC-KEY-V1\n", b"OSIRIS-PERMIT-HMAC-KEY-V2\n" + b"ab" * 32 + b"\n",
    b"OSIRIS-PERMIT-HMAC-KEY-V1\n" + b"AB" * 32 + b"\n",          # uppercase hex
    b"OSIRIS-PERMIT-HMAC-KEY-V1\n" + b"ab" * 32 + b"\nextra\n",
    b"OSIRIS-PERMIT-HMAC-KEY-V1\r\n" + b"ab" * 32 + b"\r\n",
    b"ab" * 32, b"x" * 5000,
])
def test_malformed_key_files_are_refused(tmp_path, content):
    path = tmp_path / "k"
    path.write_bytes(content)
    os.chmod(path, 0o600)
    with pytest.raises(nbg.OperatorAuthorityError) as e:
        nbg.read_operator_key(path)
    assert e.value.reason == "OPERATOR_KEY_WRONG_TYPE"


def test_permit_key_file_cannot_be_loaded_as_an_ed25519_seed(tmp_path):
    ed = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    path = provision(tmp_path)
    assert len(path.read_bytes()) != 32
    with pytest.raises(ValueError):
        ed.Ed25519PrivateKey.from_private_bytes(path.read_bytes())


# --- F-22: the permit contracts come from the installed package, not a sibling checkout -------

def _run(code, cwd=None, env=None):
    import subprocess, sys
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=cwd,
                          env={**os.environ, **(env or {})}, timeout=120)


def test_import_does_not_change_sys_path():
    r = _run("import sys; before = list(sys.path); import osiris_cli.node_beta_governance; "
             "assert sys.path == before, [p for p in sys.path if p not in before]")
    assert r.returncode == 0, r.stderr


def test_module_source_does_not_touch_sys_path():
    import ast
    tree = ast.parse(Path(nbg.__file__).read_text())
    touched = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in ("path", "meta_path", "modules")
               and isinstance(n.value, ast.Name) and n.value.id == "sys"]
    assert touched == []


def test_loaded_contracts_are_the_installed_package():
    import importlib.util
    spec = importlib.util.find_spec("osiris_governance.contracts")
    assert nbg.GOVERNANCE_ORIGIN == spec.origin
    assert nbg.PermitAuthority is PermitAuthority


def test_planted_sibling_checkout_is_not_loaded(tmp_path):
    """A copy of the module with a hostile ../osiris-governance/src next to it, as F-22 described."""
    import importlib.util, shutil
    installed = importlib.util.find_spec("osiris_governance.contracts").origin
    checkout = tmp_path / "checkout" / "osiris_cli"
    checkout.mkdir(parents=True)
    shutil.copy(nbg.__file__, checkout / "node_beta_governance.py")
    planted = tmp_path / "osiris-governance" / "src" / "osiris_governance"
    planted.mkdir(parents=True)
    (planted / "__init__.py").write_text("")
    marker = tmp_path / "planted_was_imported"
    (planted / "contracts.py").write_text(
        f"open({str(marker)!r}, 'w').close()\n"
        "class PermitAuthority:\n    def __init__(self, key): pass\n"
        "    def check(self, permit, now=None): return None\n"
        "class AuthorityPermit: pass\nPROMOTION_SCOPE = VERIFICATION_SCOPE = 'x'\n")
    code = ("import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location('planted_nbg', {str(checkout / 'node_beta_governance.py')!r})\n"
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            "print(m.GOVERNANCE_ORIGIN)\n")
    r = _run(code, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == installed
    assert not marker.exists()


def test_without_the_installed_package_everything_is_denied(tmp_path):
    """Hide osiris_governance: issue raises, verify is DENY, even with a sibling checkout present."""
    code = ("import sys\n"
            "class Block:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'osiris_governance' or name.startswith('osiris_governance.'):\n"
            "            raise ImportError(name)\n"
            "sys.meta_path.insert(0, Block())\n"
            "import osiris_cli.node_beta_governance as m\n"
            "assert m.PermitAuthority is None and m.GOVERNANCE_ORIGIN is None\n"
            "assert m.check_payload_grant(object()) == (False, 'GOVERNANCE_MODULE_MISSING')\n"
            "try:\n    m.sign_payload_grant('ab' * 32)\n"
            "except m.OperatorAuthorityError as e:\n    assert e.reason == 'GOVERNANCE_MODULE_MISSING'\n"
            "else:\n    raise SystemExit('issued without the governance package')\n")
    r = _run(code)
    assert r.returncode == 0, r.stderr
