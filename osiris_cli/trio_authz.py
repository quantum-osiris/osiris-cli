"""Signed, single-use, payload-bound authorization for ``/trio run``.

This reuses the existing SYS-021 grant machinery (``osiris_cli.grants``: ssh-keygen ed25519
signature against ``governance/allowed_signers``, NTP-synchronized clock, expiry window, and
GRANTED+CONSUMED written atomically under the ledger's file lock, with revocation). It adds no
competing mechanism. What this module adds is the payload: ONE digest over everything the owner is
authorizing, so the grant cannot be moved to a different execution:

  candidate digest, per-agent validated spec digests (the "genome"), the plan digest (repo tree,
  patch, agents, budget), requested capabilities, resource budgets, the sandbox policy and the
  authorization-policy version.

Changing any of those changes the bundle digest, and ``grants.authorize`` refuses a grant signed for
another digest. Fail closed everywhere: no sandbox -> nothing consumed, nothing run; unreadable or
broken ledger, no ssh-keygen, unsynchronized clock -> refused.

Receipts are a hash-chained JSONL under the run's --out directory. They carry grant id, key
fingerprint, digests, decisions and outcomes, never grant bytes, signatures or key material.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from typing import Any, Dict, Optional, Sequence

SCOPE = "trio.run.execute"
POLICY_VERSION = "trio-authz/1"
GENESIS = "0" * 64


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()


def sandbox_policy(limits: Any) -> Dict[str, Any]:
    """What the sandbox will enforce for this run. Part of the authorized payload."""
    return {"backend": "bwrap", "unshare": "all", "network": False, "home_visible": False,
            "writable": "workdir-only", "env": "scrubbed", "fallback": "none",
            "limits": {"wall_s": limits.wall_s, "cpu_s": limits.cpu_s, "mem_mb": limits.mem_mb,
                       "procs": limits.procs, "fsize_mb": limits.fsize_mb, "nofile": limits.nofile,
                       "out_bytes": limits.out_bytes}}


def make_bundle(*, plan_digest: str, candidate_digest: str, agents: Sequence[Dict[str, Any]],
                budget: Dict[str, Any], limits: Any) -> Dict[str, Any]:
    from osiris_cli.orchestration.agent_gene import policy_digest
    caps = sorted({c for a in agents for c in a["capabilities"]})
    return {"policy_version": POLICY_VERSION, "scope": SCOPE, "plan_digest": plan_digest,
            "candidate_digest": candidate_digest,
            "genome": [{"role": a["role"], "spec_digest": a["digest"]} for a in agents],
            "capabilities": caps, "budget": budget, "sandbox": sandbox_policy(limits),
            "capability_policy_digest": policy_digest()}


def bundle_digest(bundle: Dict[str, Any]) -> str:
    return _digest(bundle)


class Receipts:
    """Append-only hash-chained receipts. Not the canonical evidence ledgers."""

    def __init__(self, path: str):
        self.path = path

    def _tail(self) -> str:
        prev = GENESIS
        if os.path.isfile(self.path):
            for line in open(self.path, encoding="utf-8"):
                if line.strip():
                    prev = json.loads(line)["hash"]
        return prev

    def append(self, kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        row = {"kind": kind, "payload": payload, "prev": self._tail(),
               "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "node": platform.node()}
        row["hash"] = _digest(row)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return row

    @staticmethod
    def verify(path: str) -> bool:
        prev = GENESIS
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            body = {k: r[k] for k in r if k != "hash"}
            if r["prev"] != prev or r["hash"] != _digest(body):
                return False
            prev = r["hash"]
        return True


def authorize_execution(bundle: Dict[str, Any], grant_path: str, *, allowed_signers: str,
                        ledger_path: str, receipts: Receipts, signature_path: Optional[str] = None,
                        **kw: Any):
    """Verify and atomically consume the grant for exactly this bundle. Returns grants.Decision.
    ``kw`` passes ``now`` / ``clock`` through (tests only). Every outcome is receipted."""
    from osiris_cli import grants
    bd = bundle_digest(bundle)
    try:
        d = grants.authorize(grant_path, bd, SCOPE, signature_path=signature_path,
                             allowed_signers=allowed_signers, ledger_path=ledger_path, **kw)
    except Exception as e:                      # fail closed on anything unexpected
        d = grants.Decision(False, f"authorization error ({type(e).__name__}: {e})")
    rec = d.record or {}
    receipts.append("AUTHORIZATION_DECISION", {
        "authorized": bool(d.authorized), "reason": d.reason, "bundle_digest": bd, "scope": SCOPE,
        "policy_version": POLICY_VERSION, "grant_id": rec.get("grant_id"),
        "key_fingerprint": rec.get("key_fingerprint"), "principal": rec.get("principal"),
        "candidate_digest": bundle["candidate_digest"], "plan_digest": bundle["plan_digest"],
        "capabilities": bundle["capabilities"]})
    return d
