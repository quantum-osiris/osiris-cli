"""
OSIRIS Governed Multi-Agent Execution Engine
============================================
Coordinates the sequential, bounded multi-agent pipeline:
  Goal -> IntentProposal -> DecomposedPlan -> CodeProposal ->
  Independent Verification -> Governance Authorization (SYS-021/SYS-025) ->
  Execution -> Audit Recording.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from osiris_cli.providers import BaseModelProvider, get_active_provider
from .agents import (
    ArchitectAgent,
    ImplementationAgent,
    IntentAgent,
    MemoryLineageAgent,
    ResearchAgent,
    VerificationAgent,
)
from .controller import GovernanceController
from .schemas import (
    AuditReceipt,
    CodeProposal,
    DecomposedPlan,
    ExecutionGrant,
    IntentProposal,
    TaskSpec,
    VerificationReport,
)

logger = logging.getLogger("OSIRIS_GOVERNED_ENGINE")


class GovernedAgentEngine:
    """End-to-end governed multi-agent execution pipeline."""

    def __init__(
        self,
        provider: Optional[BaseModelProvider] = None,
        ledger_file: Optional[str] = None,
        allowed_paths: Optional[List[str]] = None,
    ):
        self.provider = provider or get_active_provider()
        self.intent_agent = IntentAgent(self.provider)
        self.architect_agent = ArchitectAgent(self.provider)
        self.implementation_agent = ImplementationAgent(self.provider)
        self.verification_agent = VerificationAgent()
        self.research_agent = ResearchAgent()
        self.memory_agent = MemoryLineageAgent(ledger_file)
        self.allowed_paths = allowed_paths or ["."]
        self._proposals: Dict[Tuple[str, str], CodeProposal] = {}

    def run_sprint_step(
        self,
        goal: str,
        target_file: str,
        grant: Optional[ExecutionGrant] = None,
        proposal: Optional[CodeProposal] = None,
    ) -> Dict[str, Any]:
        """
        Execute one complete governed cycle.
        If grant is provided and matches the proposed digest, applies change;
        otherwise stops at VERIFIED stage ready for authorization.
        """
        # 1. Intent interpretation
        intent = self.intent_agent.interpret(goal)

        # 2. Architectural task decomposition
        plan = self.architect_agent.plan(intent, target_file=target_file)
        task = plan.tasks[0]

        # 3. Implementation synthesis (reuse operator-reviewed proposal bound to grant when present)
        if proposal is None and grant is not None:
            proposal = self._proposals.get((target_file, grant.payload_digest))
        if proposal is None:
            proposal = self.implementation_agent.synthesize(task)
            self._proposals[(target_file, proposal.payload_digest())] = proposal

        # 4. Independent verification (AST firewall + compilation)
        verification = self.verification_agent.verify(proposal)

        # 5. Authorization decision (GovernanceController)
        authorized = False
        auth_message = "NOT_REQUESTED"
        if grant is not None:
            authorized, auth_message = GovernanceController.authorize_execution(
                proposal, grant, allowed_paths=self.allowed_paths
            )

        # 6. Governed Application (apply change only if authorized)
        applied = False
        apply_message = "SKIPPED_NOT_AUTHORIZED"
        if authorized:
            try:
                target_path = os.path.realpath(proposal.target_file)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(proposal.module_code)
                applied = True
                apply_message = f"APPLIED: Changes written to {target_path}"
            except Exception as exc:
                applied = False
                apply_message = f"APPLICATION_FAILED: {exc}"

        # 7. Audit ledger recording
        receipt_hash = self.memory_agent.record_event(
            event_type="governed_agent_step",
            payload={
                "intent_id": intent.intent_id,
                "plan_id": plan.plan_id,
                "proposal_id": proposal.proposal_id,
                "target_file": proposal.target_file,
                "payload_digest": proposal.payload_digest(),
                "verdict": verification.overall_verdict,
                "authorized": authorized,
                "auth_message": auth_message,
                "applied": applied,
                "apply_message": apply_message,
            },
        )

        return {
            "intent": intent,
            "plan": plan,
            "proposal": proposal,
            "verification": verification,
            "authorized": authorized,
            "auth_message": auth_message,
            "applied": applied,
            "apply_message": apply_message,
            "payload_digest": proposal.payload_digest(),
            "ledger_receipt_hash": receipt_hash,
        }
