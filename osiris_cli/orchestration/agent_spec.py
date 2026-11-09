"""Agent contract: identity, capabilities, budgets, state machine, receipts.

OSIRIS stays the control plane. An agent here is a declaration (AgentSpec) plus a run
(AgentRun) whose state transitions are checked and whose result is an AgentReceipt
carrying input/output digests. Nothing in this module calls a model or touches disk.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List


class AgentState(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    REJECTED = "REJECTED"        # evidence contradicts the output
    QUARANTINED = "QUARANTINED"  # output not supported by evidence; kept, not trusted


_TRANSITIONS: Dict[AgentState, FrozenSet[AgentState]] = {
    AgentState.CREATED: frozenset({AgentState.RUNNING, AgentState.FAILED}),
    AgentState.RUNNING: frozenset({AgentState.DONE, AgentState.FAILED,
                                  AgentState.REJECTED, AgentState.QUARANTINED}),
    AgentState.DONE: frozenset({AgentState.REJECTED, AgentState.QUARANTINED}),
    AgentState.FAILED: frozenset(),
    AgentState.REJECTED: frozenset(),
    AgentState.QUARANTINED: frozenset({AgentState.REJECTED}),
}


class IllegalTransition(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


def digest(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    version: str
    role: str                                   # "research" | "developer" | "critic"
    capabilities: FrozenSet[str]                # CapabilityGovernor names, e.g. READ_ONLY_ANALYSIS
    tool_permissions: FrozenSet[str] = frozenset()
    provider: str = "none"                      # provider registry name, or "none" if deterministic
    model: str = ""
    input_schema: str = ""
    output_schema: str = ""
    max_tokens: int = 4096
    deadline_s: float = 120.0
    evidence_required: bool = True

    def allows(self, capability: str) -> bool:
        return capability in self.capabilities


@dataclass
class RunBudget:
    """Bounds on one orchestrated run: agents spawned, nesting depth, wall time."""
    max_agents: int = 3
    max_depth: int = 1
    deadline_s: float = 300.0
    started: float = field(default_factory=time.monotonic)
    agents_spawned: int = 0

    def charge_agent(self, depth: int = 0) -> None:
        if depth > self.max_depth:
            raise BudgetExceeded(f"depth {depth} > max_depth {self.max_depth}")
        if self.agents_spawned >= self.max_agents:
            raise BudgetExceeded(f"agent budget {self.max_agents} exhausted")
        self.check_time()
        self.agents_spawned += 1

    def remaining_s(self) -> float:
        return self.deadline_s - (time.monotonic() - self.started)

    def check_time(self) -> None:
        if self.remaining_s() <= 0:
            raise BudgetExceeded(f"run deadline {self.deadline_s}s exceeded")


@dataclass
class AgentReceipt:
    run_id: str
    agent_id: str
    version: str
    role: str
    state: str
    input_digest: str
    output_digest: str
    transitions: List[Dict[str, Any]]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AgentRun:
    def __init__(self, spec: AgentSpec, run_id: str, budget: RunBudget, depth: int = 0):
        budget.charge_agent(depth)
        self.spec, self.run_id, self.budget = spec, run_id, budget
        self.state = AgentState.CREATED
        self.transitions: List[Dict[str, Any]] = [{"to": self.state.value, "t": time.time()}]
        self.notes: List[str] = []
        self._in = ""
        self._out = ""

    def to(self, new: AgentState, note: str = "") -> None:
        if new not in _TRANSITIONS[self.state]:
            raise IllegalTransition(f"{self.spec.agent_id}: {self.state.value} -> {new.value}")
        self.state = new
        self.transitions.append({"to": new.value, "t": time.time()})
        if note:
            self.notes.append(note)

    def start(self, inputs: Any) -> None:
        self._in = digest(inputs)
        self.to(AgentState.RUNNING)

    def finish(self, outputs: Any) -> None:
        self._out = digest(outputs)
        self.to(AgentState.DONE)

    def receipt(self) -> AgentReceipt:
        return AgentReceipt(self.run_id, self.spec.agent_id, self.spec.version, self.spec.role,
                            self.state.value, self._in, self._out, list(self.transitions),
                            list(self.notes))


def new_run_id() -> str:
    return "trio_" + uuid.uuid4().hex[:16]
