"""Shared test support: a throwaway owner key in tmp_path that signs test grants only, and a mock
provider that writes a real module and a real test for the governed engine.

Not a test module (no ``test_`` prefix). Keys made here never leave the test's tmp_path."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from osiris_cli import grants
from osiris_cli.orchestration.controller import GovernanceController
from osiris_cli.orchestration.schemas import SignedGrant

needs_ssh_keygen = pytest.mark.skipif(shutil.which("ssh-keygen") is None,
                                      reason="needs OpenSSH ssh-keygen")


def synced_clock():
    return True, "test clock"


def make_owner(directory):
    key = directory / "owner_key"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "owner", "-f", str(key)],
                   check=True)
    kind, blob = (directory / "owner_key.pub").read_text().split()[:2]
    signers = directory / "allowed_signers"
    signers.write_text(f'owner namespaces="osiris-grant" {kind} {blob}\n')
    return SimpleNamespace(key=key, signers=signers, ledger=directory / "authorization_ledger.jsonl",
                           dir=directory)


def signed_grant(owner, digest, scope=GovernanceController.APPLY_SCOPE, name="grant.json"):
    """grant.json for ``digest`` signed now by the test owner, as a SignedGrant reference."""
    path = owner.dir / name
    path.write_bytes(grants.draft(digest, scope))
    (owner.dir / (name + ".sig")).unlink(missing_ok=True)
    subprocess.run(["ssh-keygen", "-q", "-Y", "sign", "-f", str(owner.key), "-n", "osiris-grant", str(path)],
                   check=True, capture_output=True)
    return SignedGrant(grant_path=str(path), allowed_signers=str(owner.signers),
                       ledger_path=str(owner.ledger))


def real_code_responder(request):
    """Mock provider output: real code and a real test for the implementation schema, the
    provider's usual deterministic mock for every other schema."""
    props = (request.response_schema or {}).get("properties", {})
    if "module_code" in props:
        m = re.search(r"Module name: (\w+)", request.prompt)
        name = m.group(1) if m else "candidate"
        return json.dumps({
            "module_code": "def add_values(a, b):\n    return a + b\n",
            "test_code": f"import {name}\n\n\ndef test_add_values():\n    assert {name}.add_values(2, 3) == 5\n",
            "changes_summary": "add_values with a test",
        })
    out = {}
    for k, v in props.items():
        t = v.get("type", "string")
        out[k] = f"mock_{k}" if t in ("string", "STRING") else 1 if t in ("integer", "INTEGER") else \
            True if t in ("boolean", "BOOLEAN") else None
    return json.dumps(out if props else {"status": "ok", "mock": True})
