#!/usr/bin/env python3
"""
osiris_cli/gap_analyzer.py
L4 -> L5 Deterministic Capability-Gap Analyzer for the OSIRIS Sovereign Architecture.
Enforces SYS-021 payload-bound fingerprinting and epistemic/context hygiene.

Advisory only: a READY report clears a step for the SYS-021 gateway
(orchestration.controller.GovernanceController.authorize_execution); it authorizes nothing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from osiris_cli.node_ids import NODE_ALPHA, NODE_BETA, canonical_node, canonical_set


class GapKind(str, Enum):
    INTENT_AMBIGUITY = "INTENT_AMBIGUITY"
    MISSING_INPUT = "MISSING_INPUT"
    MISSING_CONTEXT = "MISSING_CONTEXT"
    STALE_CONTEXT = "STALE_CONTEXT"
    SNAPSHOT_MISMATCH = "SNAPSHOT_MISMATCH"
    CAPABILITY_UNAVAILABLE = "CAPABILITY_UNAVAILABLE"
    EXECUTION_UNAVAILABLE = "EXECUTION_UNAVAILABLE"
    POLICY_UNEVALUATED = "POLICY_UNEVALUATED"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    POLICY_DENIED = "POLICY_DENIED"
    VERIFICATION_UNAVAILABLE = "VERIFICATION_UNAVAILABLE"
    CONTRADICTORY_EVIDENCE = "CONTRADICTORY_EVIDENCE"


class Disposition(str, Enum):
    READY = "READY"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    NEEDS_CONTEXT = "NEEDS_CONTEXT"
    NEEDS_POLICY = "NEEDS_POLICY"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    BLOCKED = "BLOCKED"


class PolicyState(str, Enum):
    UNEVALUATED = "UNEVALUATED"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    DENIED = "DENIED"
    ALLOWED = "ALLOWED"


@dataclass(frozen=True)
class ContextRequirement:
    key: str
    max_age_seconds: Optional[float] = None
    require_snapshot_match: bool = True


@dataclass(frozen=True)
class ContextFact:
    key: str
    value: Any
    observed_at_epoch: float
    snapshot_id: str
    source_hash: str = ""


@dataclass(frozen=True)
class CapabilitySpec:
    capability_id: str
    required_inputs: Tuple[str, ...] = ()
    context_requirements: Tuple[ContextRequirement, ...] = ()
    supported_targets: Tuple[str, ...] = (NODE_ALPHA, NODE_BETA)
    required_verifiers: Tuple[str, ...] = ()
    enabled: bool = True


@dataclass(frozen=True)
class PolicyAssessment:
    state: PolicyState
    bound_action_fingerprint: Optional[str] = None
    policy_rule_id: str = "SYS-021"
    reason: str = ""


@dataclass(frozen=True)
class ProposedStep:
    step_id: str
    capability_id: str
    target_node: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    workspace_snapshot_id: str = ""
    intent_confidence: float = 1.0
    min_confidence_threshold: float = 0.70


@dataclass(frozen=True)
class AnalysisInput:
    step: ProposedStep
    current_epoch: float
    active_snapshot_id: str
    registered_capabilities: Mapping[str, CapabilitySpec]
    available_targets: FrozenSet[str]
    available_verifiers: FrozenSet[str]
    context_facts: Tuple[ContextFact, ...]
    policy: Optional[PolicyAssessment] = None


@dataclass(frozen=True)
class Gap:
    kind: GapKind
    subject: str
    detail: str
    remediation: str


@dataclass(frozen=True)
class GapReport:
    step_id: str
    action_fingerprint: str
    disposition: Disposition
    gaps: Tuple[Gap, ...]

    @property
    def is_ready(self) -> bool:
        return self.disposition == Disposition.READY and len(self.gaps) == 0


def compute_action_fingerprint(step: ProposedStep) -> str:
    """Computes a canonical SHA-256 digest binding capability, target, params, and snapshot."""
    canonical_payload = {
        "capability_id": step.capability_id,
        "parameters": dict(sorted(step.parameters.items())),
        "target_node": step.target_node,
        "workspace_snapshot_id": step.workspace_snapshot_id,
    }
    encoded = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class CapabilityGapAnalyzer:
    """
    Pure, deterministic L4 -> L5 analyzer.
    Evaluates proposed actions against capability specs, context freshness,
    snapshot alignment, epistemic consistency, verifiers, and SYS-021 policy state.
    """

    _GAP_ORDER: Tuple[GapKind, ...] = (
        GapKind.POLICY_DENIED,
        GapKind.CONTRADICTORY_EVIDENCE,
        GapKind.CAPABILITY_UNAVAILABLE,
        GapKind.EXECUTION_UNAVAILABLE,
        GapKind.VERIFICATION_UNAVAILABLE,
        GapKind.INTENT_AMBIGUITY,
        GapKind.MISSING_INPUT,
        GapKind.MISSING_CONTEXT,
        GapKind.STALE_CONTEXT,
        GapKind.SNAPSHOT_MISMATCH,
        GapKind.POLICY_UNEVALUATED,
        GapKind.APPROVAL_REQUIRED,
    )

    @classmethod
    def analyze(cls, inp: AnalysisInput) -> GapReport:
        step = inp.step
        fingerprint = compute_action_fingerprint(step)
        gaps: List[Gap] = []

        # 1. Intent Ambiguity Check (written so a NaN confidence fails closed)
        if not step.intent_confidence >= step.min_confidence_threshold:
            gaps.append(
                Gap(
                    kind=GapKind.INTENT_AMBIGUITY,
                    subject=step.step_id,
                    detail=f"Confidence {step.intent_confidence:.2f} below threshold {step.min_confidence_threshold:.2f}",
                    remediation="Request user disambiguation before scheduling step.",
                )
            )

        # 2. Epistemic Hygiene: Detect contradictory facts for the same key at the active snapshot
        facts_by_key: Dict[str, List[ContextFact]] = {}
        for fact in inp.context_facts:
            facts_by_key.setdefault(fact.key, []).append(fact)

        for key, key_facts in sorted(facts_by_key.items()):
            distinct_vals = {
                json.dumps(f.value, sort_keys=True, default=str)
                for f in key_facts
                if f.snapshot_id == inp.active_snapshot_id
            }
            if len(distinct_vals) > 1:
                gaps.append(
                    Gap(
                        kind=GapKind.CONTRADICTORY_EVIDENCE,
                        subject=key,
                        detail=f"Conflicting values observed for context key '{key}' ({len(distinct_vals)} distinct states)",
                        remediation=f"Reconcile contradictory evidence for '{key}' before execution.",
                    )
                )

        # 3. Proposal vs Active Workspace Snapshot Check
        if not step.workspace_snapshot_id:
            gaps.append(
                Gap(
                    kind=GapKind.SNAPSHOT_MISMATCH,
                    subject="workspace_snapshot_id",
                    detail="Step is not bound to a workspace snapshot; an approval of it would survive any commit.",
                    remediation="Bind the step to the active snapshot and regenerate the action proposal.",
                )
            )
        elif step.workspace_snapshot_id != inp.active_snapshot_id:
            gaps.append(
                Gap(
                    kind=GapKind.SNAPSHOT_MISMATCH,
                    subject="workspace_snapshot_id",
                    detail=f"Step planned against '{step.workspace_snapshot_id}', active is '{inp.active_snapshot_id}'",
                    remediation="Re-hydrate context snapshot and regenerate action proposal.",
                )
            )

        # 4. Capability Registration & Availability
        cap_spec = inp.registered_capabilities.get(step.capability_id)
        if cap_spec is None or not cap_spec.enabled:
            gaps.append(
                Gap(
                    kind=GapKind.CAPABILITY_UNAVAILABLE,
                    subject=step.capability_id,
                    detail=f"Capability '{step.capability_id}' is unregistered or disabled.",
                    remediation="Register or enable the capability in the sovereign skill registry.",
                )
            )
        else:
            # 5. Required Input Parameters Check
            for req_input in cap_spec.required_inputs:
                val = step.parameters.get(req_input)
                if val is None or (isinstance(val, str) and not val.strip()):
                    gaps.append(
                        Gap(
                            kind=GapKind.MISSING_INPUT,
                            subject=req_input,
                            detail=f"Required parameter '{req_input}' is missing or empty.",
                            remediation=f"Supply parameter '{req_input}' in ProposedStep.",
                        )
                    )

            # 6. Execution Target Availability
            if (
                canonical_node(step.target_node) is None
                or canonical_node(step.target_node) not in canonical_set(cap_spec.supported_targets)
                or canonical_node(step.target_node) not in canonical_set(inp.available_targets)
            ):
                gaps.append(
                    Gap(
                        kind=GapKind.EXECUTION_UNAVAILABLE,
                        subject=step.target_node,
                        detail=f"Target node '{step.target_node}' is offline or unsupported by '{step.capability_id}'.",
                        remediation="Route step to an online, supported substrate node.",
                    )
                )

            # 7. Verifier Coverage Check
            for verifier in cap_spec.required_verifiers:
                if verifier not in inp.available_verifiers:
                    gaps.append(
                        Gap(
                            kind=GapKind.VERIFICATION_UNAVAILABLE,
                            subject=verifier,
                            detail=f"Required verifier '{verifier}' is unavailable.",
                            remediation=f"Bring verifier '{verifier}' online before scheduling.",
                        )
                    )

            # 8. Context Completeness, Freshness & Fact Snapshot Match
            for creq in cap_spec.context_requirements:
                matching = facts_by_key.get(creq.key, [])
                if not matching:
                    gaps.append(
                        Gap(
                            kind=GapKind.MISSING_CONTEXT,
                            subject=creq.key,
                            detail=f"Required context fact '{creq.key}' is missing.",
                            remediation=f"Collect context fact '{creq.key}' prior to scheduling.",
                        )
                    )
                    continue

                latest_fact = max(matching, key=lambda f: f.observed_at_epoch)
                if creq.max_age_seconds is not None:
                    age = inp.current_epoch - latest_fact.observed_at_epoch
                    if age > creq.max_age_seconds:
                        gaps.append(
                            Gap(
                                kind=GapKind.STALE_CONTEXT,
                                subject=creq.key,
                                detail=f"Context '{creq.key}' age {age:.1f}s exceeds max_age {creq.max_age_seconds:.1f}s.",
                                remediation=f"Refresh context fact '{creq.key}'.",
                            )
                        )

                if creq.require_snapshot_match and latest_fact.snapshot_id != inp.active_snapshot_id:
                    gaps.append(
                        Gap(
                            kind=GapKind.SNAPSHOT_MISMATCH,
                            subject=creq.key,
                            detail=f"Fact '{creq.key}' bound to '{latest_fact.snapshot_id}' != '{inp.active_snapshot_id}'.",
                            remediation=f"Re-observe '{creq.key}' at snapshot '{inp.active_snapshot_id}'.",
                        )
                    )

        # 9. Policy & Payload-Bound Authorization Check (SYS-021)
        if inp.policy is None or inp.policy.state == PolicyState.UNEVALUATED:
            gaps.append(
                Gap(
                    kind=GapKind.POLICY_UNEVALUATED,
                    subject="SYS-021",
                    detail="Policy assessment has not been evaluated for this step.",
                    remediation="Evaluate step through the SYS-021 policy gateway.",
                )
            )
        elif inp.policy.state == PolicyState.DENIED:
            gaps.append(
                Gap(
                    kind=GapKind.POLICY_DENIED,
                    subject=inp.policy.policy_rule_id,
                    detail=inp.policy.reason or "Action explicitly denied by governance policy.",
                    remediation="Abort action; policy denial is terminal.",
                )
            )
        elif inp.policy.state == PolicyState.APPROVAL_REQUIRED:
            gaps.append(
                Gap(
                    kind=GapKind.APPROVAL_REQUIRED,
                    subject=inp.policy.policy_rule_id,
                    detail=inp.policy.reason or "Human operator approval required.",
                    remediation=f"Obtain operator signature bound to fingerprint {fingerprint[:12]}.",
                )
            )
        elif inp.policy.state == PolicyState.ALLOWED:
            # Verify payload-bound fingerprint matches approval
            if inp.policy.bound_action_fingerprint != fingerprint:
                gaps.append(
                    Gap(
                        kind=GapKind.POLICY_UNEVALUATED,
                        subject="SYS-021::FINGERPRINT_MISMATCH",
                        detail="Approval fingerprint does not match current ProposedStep payload.",
                        remediation="Re-evaluate policy and bind approval to current action_fingerprint.",
                    )
                )

        # Sort gaps deterministically
        order_index = {k: idx for idx, k in enumerate(cls._GAP_ORDER)}
        sorted_gaps = tuple(sorted(gaps, key=lambda g: (order_index[g.kind], g.subject, g.detail)))
        disposition = cls._resolve_disposition(sorted_gaps)

        return GapReport(
            step_id=step.step_id,
            action_fingerprint=fingerprint,
            disposition=disposition,
            gaps=sorted_gaps,
        )

    @staticmethod
    def _resolve_disposition(gaps: Sequence[Gap]) -> Disposition:
        if not gaps:
            return Disposition.READY

        kinds = {g.kind for g in gaps}
        if kinds & {
            GapKind.POLICY_DENIED,
            GapKind.CONTRADICTORY_EVIDENCE,
            GapKind.CAPABILITY_UNAVAILABLE,
            GapKind.EXECUTION_UNAVAILABLE,
            GapKind.VERIFICATION_UNAVAILABLE,
        }:
            return Disposition.BLOCKED
        if kinds & {GapKind.INTENT_AMBIGUITY, GapKind.MISSING_INPUT}:
            return Disposition.NEEDS_CLARIFICATION
        if kinds & {GapKind.MISSING_CONTEXT, GapKind.STALE_CONTEXT, GapKind.SNAPSHOT_MISMATCH}:
            return Disposition.NEEDS_CONTEXT
        if GapKind.POLICY_UNEVALUATED in kinds:
            return Disposition.NEEDS_POLICY
        if GapKind.APPROVAL_REQUIRED in kinds:
            return Disposition.WAITING_APPROVAL
        return Disposition.BLOCKED
