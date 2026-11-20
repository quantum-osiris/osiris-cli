"""
OSIRIS Governed Agent Platform Schemas
======================================
Typed data structures for multi-agent execution, task decomposition,
code proposals, verification reports, and payload-bound grants.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaskSpec:
    task_id: str
    title: str
    spec: str
    target_files: List[str] = field(default_factory=list)
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    acceptance_criteria: List[str] = field(default_factory=list)
    timeout_seconds: float = 60.0
    resource_budget: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntentProposal:
    intent_id: str
    raw_prompt: str
    intent_type: str  # "code_modification", "research", "benchmark", "status", "query"
    domain: str       # "quantum", "compiler", "governance", "general"
    confidence: float
    target_scope: List[str] = field(default_factory=list)
    risk_level: str = "low"  # "low", "medium", "high"
    provenance: str = "human_operator"


@dataclass
class DecomposedPlan:
    plan_id: str
    intent_id: str
    tasks: List[TaskSpec] = field(default_factory=list)
    architectural_rationale: str = ""
    required_grants: List[str] = field(default_factory=list)


@dataclass
class CodeProposal:
    proposal_id: str
    task_id: str
    target_file: str
    module_code: str
    test_code: str
    changes_summary: str = ""
    rollback_strategy: str = ""
    created_at: float = field(default_factory=time.time)

    def payload_digest(self) -> str:
        """Compute canonical SHA-256 digest of proposed code (SYS-021)."""
        blob = json.dumps({
            "target_file": self.target_file,
            "module_code": self.module_code,
            "test_code": self.test_code,
        }, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


@dataclass
class VerificationResult:
    check_name: str
    passed: bool
    reason: str = ""
    logs: str = ""
    execution_time_ms: float = 0.0


@dataclass
class VerificationReport:
    proposal_id: str
    overall_verdict: str  # "PASS", "FAIL", "QUARANTINED"
    static_ast_pass: bool = False
    banned_constructs_found: List[str] = field(default_factory=list)
    acceptance_test_pass: bool = False
    checks: List[VerificationResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    # The CodeProposal.payload_digest() this report is about. A write is authorized only with a
    # PASS report whose digest equals the proposal being written (REQ-AUTH-4).
    proposal_digest: str = ""
    tests_collected: int = 0

    @property
    def is_passing(self) -> bool:
        return self.overall_verdict == "PASS" and self.static_ast_pass and not self.banned_constructs_found


@dataclass
class ExecutionGrant:
    """A record of an approval that already happened elsewhere. It is NOT an authority.

    Anyone can construct this dataclass, so GovernanceController.authorize_execution refuses it
    (REQ-AUTH-1). Writes are authorized only by a SignedGrant verified with grants.authorize;
    commands only by the execution kernel's own grant.
    """

    grant_id: str
    payload_digest: str  # Bound strictly to H_payload per SYS-021
    authorized_by: str
    authorized_at: float
    expires_at: float
    scope: List[str] = field(default_factory=list)
    revoked: bool = False

    def is_valid(self, candidate_digest: str) -> bool:
        """Validate payload binding and expiration."""
        if self.revoked:
            return False
        if time.time() > self.expires_at:
            return False
        return self.payload_digest == candidate_digest


@dataclass(frozen=True)
class SignedGrant:
    """Where to find an owner-signed grant (grant.json + its ssh signature) for one write.

    The owner signs ``GovernanceController.action_digest(proposal)`` under scope
    ``GovernanceController.APPLY_SCOPE``; grants.authorize checks the signature against
    allowed_signers, the clock window, revocation, and consumes the grant once.
    """

    grant_path: str
    signature_path: Optional[str] = None
    allowed_signers: Optional[str] = None
    ledger_path: Optional[str] = None


@dataclass
class AuditReceipt:
    receipt_id: str
    grant_id: str
    payload_digest: str
    action: str
    timestamp: str
    status: str
    ledger_hash: str
