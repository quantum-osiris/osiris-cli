"""
OSIRIS Governance Controller
============================
Enforces deterministic policy invariants:
  - SYS-021: Payload-Bound Authorization. A write is authorized only by an owner-signed grant
    (grants.authorize) bound to the exact action digest: target realpath, pre-image, post-image,
    test code and execution-policy SHA-256. A plain ExecutionGrant dataclass authorizes nothing.
  - REQ-AUTH-4: the write also needs a PASS VerificationReport for the same proposal digest.
  - SYS-025: one shared deny-by-default AST firewall (ast_firewall) and governance files
    protected by repo-relative path (governance_scope).
Deterministic code decides; models propose.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Callable, List, Optional, Tuple

from . import ast_firewall, governance_scope
from .governance_scope import IMMUTABLE_GOVERNANCE_DIRS, IMMUTABLE_GOVERNANCE_FILES
from .schemas import CodeProposal, SignedGrant, VerificationReport

# Kept for callers that import them from here.
BANNED_CALL_NAMES = set(ast_firewall.BANNED_BUILTINS)

__all__ = ["GovernanceController", "ASTFirewallVisitor", "IMMUTABLE_GOVERNANCE_FILES",
           "IMMUTABLE_GOVERNANCE_DIRS", "BANNED_CALL_NAMES"]


class ASTFirewallVisitor:
    """Compatibility wrapper over the shared firewall: ``visit(tree)`` then read ``violations``."""

    def __init__(self) -> None:
        self.violations: List[str] = []

    def visit(self, tree) -> None:
        self.violations.extend(ast_firewall.controller_message(v) for v in ast_firewall.scan_tree(tree))


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class GovernanceController:
    """Deterministic policy gatekeeper."""

    APPLY_SCOPE = "governed_engine.apply"
    ACTION_DIGEST_VERSION = "governed-write-1"

    @staticmethod
    def inspect_ast(source_code: str) -> Tuple[bool, List[str]]:
        """Static AST security analysis. Returns (is_clean, violations)."""
        ok, found = ast_firewall.check(source_code)
        return ok, [ast_firewall.controller_message(v) for v in found]

    @staticmethod
    def verify_target_scope(target_file: str, allowed_paths: Optional[List[str]] = None) -> Tuple[bool, str]:
        """Check the target is not a governance file and resides in the allowed scope.

        Governance files are matched by repo-relative path after resolving symlinks
        (governance_scope). With no scope given, the scope is the working directory -- the same
        safe root the REPL's own write check uses -- never "anywhere".
        """
        roots = list(allowed_paths) if allowed_paths else ["."]
        why = governance_scope.protected_reason(target_file, roots)
        if why:
            return False, f"Target {why} (SYS-025 violation)"
        resolved = os.path.realpath(target_file)
        for p in roots:
            root = os.path.realpath(p)
            if resolved == root or resolved.startswith(root.rstrip(os.sep) + os.sep):
                return True, "Scope valid"
        return False, f"Target '{target_file}' outside permitted scope {roots}"

    @staticmethod
    def action_digest(proposal: CodeProposal) -> str:
        """The digest the owner signs for this write: realpath, pre-image, post-image, tests, policy.

        A one-byte change to the code, the tests, the file on disk, or the execution policy after
        signing gives a different digest, and the grant no longer matches.
        """
        from osiris_cli.exec_policy import load_policy

        target = os.path.realpath(proposal.target_file)
        try:
            with open(target, "rb") as fh:
                pre = _sha256_hex(fh.read())
        except FileNotFoundError:
            pre = None
        body = {
            "v": GovernanceController.ACTION_DIGEST_VERSION,
            "target_realpath": target,
            "pre_image_sha256": pre,
            "post_image_sha256": _sha256_hex(proposal.module_code.encode("utf-8")),
            "test_sha256": _sha256_hex(proposal.test_code.encode("utf-8")),
            "policy_sha256": load_policy()[1],
        }
        return _sha256_hex(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8"))

    @staticmethod
    def authorize_execution(
        proposal: CodeProposal,
        grant: Any,
        allowed_paths: Optional[List[str]] = None,
        verification: Optional[VerificationReport] = None,
        *,
        clock: Optional[Callable[[], Tuple[bool, str]]] = None,
        now: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Authorize one write under SYS-021, REQ-AUTH-4 and SYS-025.
        Fails closed on: a target outside scope or a governance file; banned AST constructs; a
        missing grant; any grant that is not a SignedGrant (dataclass grants are records, not
        authority); a missing, failing or mismatched verification report; and any refusal by
        grants.authorize (signature, digest, scope, clock, revocation, already consumed).
        """
        scope_ok, scope_msg = GovernanceController.verify_target_scope(proposal.target_file, allowed_paths)
        if not scope_ok:
            return False, f"AUTHORIZATION_REJECTED: {scope_msg}"

        for label, source in (("module", proposal.module_code), ("test", proposal.test_code)):
            ast_ok, ast_violations = GovernanceController.inspect_ast(source)
            if not ast_ok:
                return False, f"AUTHORIZATION_REJECTED (SYS-025): Banned constructs in {label} {ast_violations}"

        if grant is None:
            return False, "AUTHORIZATION_REJECTED (SYS-021): Missing execution grant"
        if not isinstance(grant, SignedGrant):
            return False, (f"AUTHORIZATION_REJECTED (REQ-AUTH-1): a {type(grant).__name__} is not an "
                           "authority; only an owner-signed grant verified by grants.authorize is")

        digest = proposal.payload_digest()
        if not isinstance(verification, VerificationReport):
            return False, "AUTHORIZATION_REJECTED (REQ-AUTH-4): no verification report for this proposal"
        if verification.proposal_digest != digest:
            return False, ("AUTHORIZATION_REJECTED (REQ-AUTH-4): the verification report is for a "
                           "different proposal digest")
        if not (verification.is_passing and verification.acceptance_test_pass):
            return False, (f"AUTHORIZATION_REJECTED (REQ-AUTH-4): verification verdict is "
                           f"{verification.overall_verdict}, not a passing test run")

        from osiris_cli import grants

        kwargs = {}
        if grant.signature_path:
            kwargs["signature_path"] = grant.signature_path
        if grant.allowed_signers:
            kwargs["allowed_signers"] = grant.allowed_signers
        if grant.ledger_path:
            kwargs["ledger_path"] = grant.ledger_path
        if clock is not None:
            kwargs["clock"] = clock
        if now is not None:
            kwargs["now"] = now
        try:
            action = GovernanceController.action_digest(proposal)
        except Exception as exc:  # policy unreadable, target unreadable: refuse
            return False, f"AUTHORIZATION_REJECTED (SYS-021): action digest unavailable ({exc})"
        decision = grants.authorize(grant.grant_path, action, GovernanceController.APPLY_SCOPE, **kwargs)
        if not decision.authorized:
            return False, f"AUTHORIZATION_REJECTED (SYS-021): {decision.reason}"
        return True, "AUTHORIZED"
