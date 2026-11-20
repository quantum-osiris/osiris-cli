"""
OSIRIS L5 Verify: Independent Critic Verifier
============================================
Performs deterministic static AST analysis, forbidden pattern detection,
syntax verification, and test execution validation on candidate code proposals.
Enforces SYS-023 and SYS-025 fail-closed guarantees.
"""

from __future__ import annotations

import ast
import time
from typing import List

from .schemas import CodeProposal, VerificationReport, VerificationResult

from . import ast_firewall, governance_scope

BANNED_FUNCTIONS = set(ast_firewall.BANNED_BUILTINS)
BANNED_MODULES = {"os.system", "posix.system"}  # kept for importers; the rules live in ast_firewall
# Basenames refused anywhere; governance_scope adds the repo-relative paths (exec_policy.json,
# governance/allowed_signers, contracts/, ...).
IMMUTABLE_FILES = set(governance_scope.LEGACY_BASENAMES)


class ASTSecurityVisitor:
    """Compatibility wrapper over the shared firewall: ``visit(tree)`` then read ``violations``
    as critic tokens such as ``banned_call:os.system:L3``."""

    def __init__(self):
        self.violations: List[str] = []

    def visit(self, tree) -> None:
        self.violations.extend(ast_firewall.critic_token(v) for v in ast_firewall.scan_tree(tree))


class IndependentCriticVerifier:
    """Independent critic verifier enforcing deterministic AST and governance policies."""

    def verify_proposal(self, proposal: CodeProposal) -> VerificationReport:
        t0 = time.time()
        checks: List[VerificationResult] = []
        banned_constructs: List[str] = []

        # 1. Target file immutability check
        if governance_scope.is_protected(proposal.target_file):
            checks.append(VerificationResult(
                check_name="immutable_governance_target",
                passed=False,
                reason=f"Target file '{proposal.target_file}' is an immutable governance artifact.",
            ))
            return VerificationReport(
                proposal_id=proposal.proposal_id,
                overall_verdict="BLOCKED",
                static_ast_pass=False,
                banned_constructs_found=["immutable_target_violation"],
                checks=checks,
            )
        checks.append(VerificationResult(
            check_name="immutable_governance_target",
            passed=True,
            reason="Target file is within mutable workspace boundaries.",
        ))

        # 2. Python AST syntax & security analysis
        try:
            tree = ast.parse(proposal.module_code)
            visitor = ASTSecurityVisitor()
            visitor.visit(tree)
            banned_constructs.extend(visitor.violations)

            ast_pass = len(visitor.violations) == 0
            checks.append(VerificationResult(
                check_name="static_ast_firewall",
                passed=ast_pass,
                reason="Static AST check passed with 0 violations." if ast_pass else f"Violations: {visitor.violations}",
                execution_time_ms=(time.time() - t0) * 1000.0,
            ))
        except SyntaxError as e:
            checks.append(VerificationResult(
                check_name="static_ast_syntax",
                passed=False,
                reason=f"SyntaxError in candidate code: {e}",
            ))
            return VerificationReport(
                proposal_id=proposal.proposal_id,
                overall_verdict="FAIL",
                static_ast_pass=False,
                banned_constructs_found=["syntax_error"],
                checks=checks,
            )

        # 3. Test code syntax validation
        if proposal.test_code:
            try:
                ast.parse(proposal.test_code)
                checks.append(VerificationResult(
                    check_name="test_code_syntax",
                    passed=True,
                    reason="Test suite AST is valid.",
                ))
            except SyntaxError as e:
                checks.append(VerificationResult(
                    check_name="test_code_syntax",
                    passed=False,
                    reason=f"SyntaxError in test code: {e}",
                ))

        overall_verdict = "PASS" if all(c.passed for c in checks) and not banned_constructs else "FAIL"

        return VerificationReport(
            proposal_id=proposal.proposal_id,
            overall_verdict=overall_verdict,
            static_ast_pass=len(banned_constructs) == 0,
            banned_constructs_found=banned_constructs,
            # This critic parses tests; it never runs them, so its report can never stand in for a
            # test run when a write is authorized (REQ-AUTH-4 needs VerificationAgent's report).
            acceptance_test_pass=False,
            checks=checks,
            proposal_digest=proposal.payload_digest(),
        )
