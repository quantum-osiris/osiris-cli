"""One run ID per task, carrying what the task may do and how much it may spend.

Before this, a task's limits lived wherever it ran: /code had a step count, a sprint story had
k attempts, nothing had a wall clock, and no record said which limits a run had been given.
A ``TaskRunContext`` is minted by code when a task starts and is the single statement of:

  * who it is            -- ``run_id``, the kind of task, the SHA-256 of the operator's words;
  * what it costs wrong  -- the risk tier and cost class ``intent_router.assess`` gave those words;
  * what it may do       -- ``Permissions``: which tools, whether writes may be proposed, whether
                            commands may be asked for;
  * what it may spend    -- ``Budget``: model steps, tool calls, tokens, wall-clock seconds,
                            and how many failures in a row.

No model output is read to build one, and nothing in a context grants an external or hardware
action: those fields exist so a record can say they were refused. Permissions are a ceiling. A
proposed write still needs the operator's approval; a command still needs its own.

The context is frozen and carries the SHA-256 of its own canonical form; ``from_dict`` rejects
one whose content no longer hashes to it. ``BudgetMeter`` enforces the budget and returns a
``Stop`` with one of two statuses:

  ``STOPPED_BUDGET``  a cap was reached (steps, tool calls, tokens, wall clock, repeated failure);
  ``BLOCKED``         the run asked for something its context does not permit, or the context is
                      not the one that was issued.

``TaskLedger`` records each run's opening (the whole context, and the head of the L0 exchange
log at that moment, which links the task to intake without changing the exchange log) and its
close (status, stop reason, what was spent) in a hash-chained JSONL file written under
``chain_lock``. It uses the exchange log's hash rule, so ``probes.verify_chain`` checks it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

SCHEMA = "osiris.task_context/1"
LEDGER_SCHEMA = "osiris.task_ledger/1"
DEFAULT_LEDGER = os.path.join(os.path.expanduser("~"), ".osiris", "tasks", "task_runs.jsonl")
EXCHANGE_LOG = os.path.join(os.path.expanduser("~"), ".osiris", "living", "exchanges.jsonl")
GENESIS = "0" * 64

STOPPED_BUDGET = "STOPPED_BUDGET"
BLOCKED = "BLOCKED"

RISK_TIERS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
COST_CLASSES = ("read_only", "workspace_write", "external_action", "hardware_action")
KINDS = ("code", "sprint_story")

READ_TOOLS = ("grep_code", "list_dir", "read_file")
WRITE_TOOL = "write_file"
COMMAND_TOOL = "run_command"
KNOWN_TOOLS = frozenset(READ_TOOLS) | {WRITE_TOOL, COMMAND_TOOL}


class TaskContextError(ValueError):
    """A context that cannot be built, or is not the one that was issued."""


class TaskLedgerError(RuntimeError):
    """The task ledger could not be written or does not verify; the task must not start."""


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TaskContextError(f"budget {name} must be a whole number >= 1, got {value!r}")
    return value


@dataclass(frozen=True)
class Budget:
    """Hard caps for one task. Every field is required and finite: there is no 'unlimited'."""
    max_steps: int               # model turns (a sprint story: candidates)
    max_tool_calls: int          # tool calls the runner executes or refuses
    max_writes: int              # distinct files a run may propose
    max_tokens: int              # prompt + reply tokens, as reported by the backend or estimated
    max_wall_s: float            # seconds from the run's first step
    max_repeated_failures: int   # failures in a row before the run is stopped

    def __post_init__(self) -> None:
        for name in ("max_steps", "max_tool_calls", "max_writes", "max_tokens", "max_repeated_failures"):
            _positive_int(name, getattr(self, name))
        wall = self.max_wall_s
        if isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall <= 0:
            raise TaskContextError(f"budget max_wall_s must be a finite number of seconds > 0, got {wall!r}")
        object.__setattr__(self, "max_wall_s", float(wall))


@dataclass(frozen=True)
class Permissions:
    """What a task may ask for. A ceiling, never an approval."""
    tools: Tuple[str, ...] = ()
    propose_writes: bool = False
    run_commands: bool = False
    external_actions: bool = False   # push, publish, deploy, send: no context grants these
    hardware_actions: bool = False   # QPU submission, device flashing: no context grants these

    def __post_init__(self) -> None:
        tools = tuple(sorted(set(self.tools)))
        unknown = [t for t in tools if t not in KNOWN_TOOLS]
        if unknown:
            raise TaskContextError(f"unknown tool(s) in permissions: {unknown}")
        for name in ("propose_writes", "run_commands", "external_actions", "hardware_actions"):
            if not isinstance(getattr(self, name), bool):
                raise TaskContextError(f"permission {name} must be true or false")
        if self.external_actions or self.hardware_actions:
            raise TaskContextError("no task context may grant an external or hardware action")
        if (WRITE_TOOL in tools) != self.propose_writes:
            raise TaskContextError("write_file is permitted exactly when writes may be proposed")
        if (COMMAND_TOOL in tools) != self.run_commands:
            raise TaskContextError("run_command is permitted exactly when commands may be asked for")
        object.__setattr__(self, "tools", tools)

    def allows_tool(self, name: str) -> bool:
        return name in self.tools


def permissions(*, read: bool = True, propose_writes: bool = False, run_commands: bool = False) -> Permissions:
    """Permissions from three plain facts, so the tool list cannot disagree with them."""
    tools = list(READ_TOOLS) if read else []
    if propose_writes:
        tools.append(WRITE_TOOL)
    if run_commands:
        tools.append(COMMAND_TOOL)
    return Permissions(tools=tuple(tools), propose_writes=propose_writes, run_commands=run_commands)


# Budgets by risk tier. A riskier request gets a shorter leash, not a longer one: fewer
# unattended steps and less time before a person looks. MEDIUM is what /code allowed before
# contexts existed (8 model turns); its wall clock covers 8 turns at Node Alpha's 300 s horizon
# with room for approved commands. Callers may tighten a budget, never loosen past these.
TIER_BUDGETS: Dict[str, Dict[str, Any]] = {
    "LOW": dict(max_steps=8, max_tool_calls=24, max_tokens=120_000, max_wall_s=1800.0, max_repeated_failures=4),
    "MEDIUM": dict(max_steps=8, max_tool_calls=32, max_tokens=160_000, max_wall_s=3600.0, max_repeated_failures=4),
    "HIGH": dict(max_steps=6, max_tool_calls=18, max_tokens=90_000, max_wall_s=1800.0, max_repeated_failures=3),
    "CRITICAL": dict(max_steps=4, max_tool_calls=8, max_tokens=40_000, max_wall_s=900.0, max_repeated_failures=2),
}


def budget_for(tier: str, *, max_writes: int = 1, **tighten: Any) -> Budget:
    """The tier's budget, with any cap in ``tighten`` lowered (a value above the tier's cap is
    ignored: the tier is the ceiling). An unknown tier is refused."""
    if tier not in TIER_BUDGETS:
        raise TaskContextError(f"unknown risk tier {tier!r}")
    caps = dict(TIER_BUDGETS[tier])
    for name, value in tighten.items():
        if name not in caps:
            raise TaskContextError(f"unknown budget field {name!r}")
        if value is not None:
            caps[name] = min(caps[name], value)
    return Budget(max_writes=max_writes, **caps)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(body: Dict[str, Any]) -> str:
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def new_run_id(kind: str) -> str:
    return f"task-{kind.replace('_', '-')}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True)
class TaskRunContext:
    run_id: str
    kind: str                    # one of KINDS
    created_at: str              # UTC, ISO 8601
    text_sha256: str             # SHA-256 of the operator's words for this task
    intent_class: str            # intent_router class of those words
    risk_tier: str               # one of RISK_TIERS
    cost_class: str              # one of COST_CLASSES
    risk_reasons: Tuple[str, ...]
    permissions: Permissions
    budget: Budget
    parent_run_id: Optional[str] = None      # the run this one belongs to (e.g. a code run id)
    intake: Dict[str, Any] = field(default_factory=dict)   # {"exchange_head": <hash or None>, ...}
    context_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.run_id or any(c in self.run_id for c in "/\\\n\r\t "):
            raise TaskContextError(f"bad run id {self.run_id!r}")
        if self.kind not in KINDS:
            raise TaskContextError(f"unknown task kind {self.kind!r}")
        if self.risk_tier not in RISK_TIERS:
            raise TaskContextError(f"unknown risk tier {self.risk_tier!r}")
        if self.cost_class not in COST_CLASSES:
            raise TaskContextError(f"unknown cost class {self.cost_class!r}")
        if not (isinstance(self.text_sha256, str) and len(self.text_sha256) == 64
                and all(c in "0123456789abcdef" for c in self.text_sha256)):
            raise TaskContextError("text_sha256 must be a SHA-256 hex digest")
        if not isinstance(self.permissions, Permissions) or not isinstance(self.budget, Budget):
            raise TaskContextError("permissions and budget must be Permissions and Budget")
        object.__setattr__(self, "risk_reasons", tuple(str(r) for r in self.risk_reasons))
        object.__setattr__(self, "intake", json.loads(_canonical(dict(self.intake))))
        digest = sha256_text(_canonical(self._body()))
        if self.context_sha256 and self.context_sha256 != digest:
            raise TaskContextError(f"context {self.run_id} does not hash to its recorded digest; "
                                   f"it is not the context that was issued")
        object.__setattr__(self, "context_sha256", digest)

    def _body(self) -> Dict[str, Any]:
        return {
            "schema": SCHEMA, "run_id": self.run_id, "kind": self.kind, "created_at": self.created_at,
            "text_sha256": self.text_sha256, "intent_class": self.intent_class,
            "risk_tier": self.risk_tier, "cost_class": self.cost_class,
            "risk_reasons": list(self.risk_reasons),
            "permissions": dict(asdict(self.permissions), tools=list(self.permissions.tools)),
            "budget": asdict(self.budget),
            "parent_run_id": self.parent_run_id, "intake": self.intake,
        }

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._body(), context_sha256=self.context_sha256)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskRunContext":
        """Rebuild a recorded context. Fails closed: a missing field, an unknown schema or a body
        that does not hash to ``context_sha256`` raises TaskContextError."""
        if not isinstance(data, dict) or data.get("schema") != SCHEMA:
            raise TaskContextError(f"not a {SCHEMA} record")
        if not data.get("context_sha256"):
            raise TaskContextError("recorded context has no digest")
        try:
            perms = dict(data["permissions"])
            return cls(
                run_id=data["run_id"], kind=data["kind"], created_at=data["created_at"],
                text_sha256=data["text_sha256"], intent_class=data["intent_class"],
                risk_tier=data["risk_tier"], cost_class=data["cost_class"],
                risk_reasons=tuple(data["risk_reasons"]),
                permissions=Permissions(**dict(perms, tools=tuple(perms["tools"]))),
                budget=Budget(**data["budget"]),
                parent_run_id=data.get("parent_run_id"), intake=dict(data.get("intake") or {}),
                context_sha256=data["context_sha256"],
            )
        except (KeyError, TypeError) as exc:
            raise TaskContextError(f"recorded context is incomplete or malformed ({exc})") from None

    def meter(self, clock: Callable[[], float] = time.monotonic) -> "BudgetMeter":
        return BudgetMeter(self, clock=clock)

    def summary(self) -> str:
        b, p = self.budget, self.permissions
        may = ", ".join(p.tools) or "no tools"
        return (f"{self.run_id} · risk {self.risk_tier} ({self.cost_class.replace('_', ' ')}) · may use: {may} · "
                f"budget: {b.max_steps} steps, {b.max_tool_calls} tool calls, {b.max_tokens:,} tokens, "
                f"{b.max_wall_s:.0f}s, stop after {b.max_repeated_failures} failures in a row")


def new_context(text: str, *, kind: str, perms: Permissions, max_writes: int = 1,
                parent_run_id: Optional[str] = None, exchange_log: Optional[str] = EXCHANGE_LOG,
                assess: Optional[Callable[[str], Any]] = None, now: Optional[str] = None,
                run_id: Optional[str] = None, **tighten: Any) -> TaskRunContext:
    """Mint the context for a task that is about to start.

    The risk tier and cost class come from ``intent_router.assess`` on the operator's words (code
    rules, no model). A task of these kinds proposes local changes, so its cost class is never
    below workspace_write. The budget is the tier's, tightened by ``tighten``. ``perms`` comes
    from what the operator asked for on the command line, never from the text."""
    if assess is None:
        try:
            import intent_router
            assess = intent_router.assess
        except ImportError:
            assess = _unassessed
    a = assess(text)
    cost = a.cost_class if COST_CLASSES.index(a.cost_class) >= COST_CLASSES.index("workspace_write") \
        else "workspace_write"
    tier = a.tier if RISK_TIERS.index(a.tier) >= RISK_TIERS.index("MEDIUM") else "MEDIUM"
    reasons = tuple(a.reasons) + ((f"a {kind.replace('_', ' ')} task proposes local changes",)
                                  if (cost, tier) != (a.cost_class, a.tier) else ())
    return TaskRunContext(
        run_id=run_id or new_run_id(kind), kind=kind,
        created_at=now or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        text_sha256=sha256_text(text), intent_class=a.cls, risk_tier=tier, cost_class=cost,
        risk_reasons=reasons, permissions=perms, budget=budget_for(tier, max_writes=max_writes, **tighten),
        parent_run_id=parent_run_id, intake={"exchange_head": exchange_head(exchange_log)},
    )


@dataclass(frozen=True)
class _Unassessed:
    cls: str = "unknown"
    cost_class: str = "workspace_write"
    tier: str = "CRITICAL"
    reasons: Tuple[str, ...] = ("the risk rules (intent_router) could not be loaded: treated as CRITICAL",)


def _unassessed(text: str) -> _Unassessed:
    """When the rules cannot be loaded the task still gets a context -- the tightest one."""
    return _Unassessed()


def exchange_head(path: Optional[str]) -> Optional[str]:
    """The hash of the last entry in the L0 exchange log, read without changing it. None when
    there is no log yet or its last line cannot be read (recorded as None, not guessed)."""
    if not path or not os.path.exists(path):
        return None
    from osiris_cli import chain_lock
    try:
        with open(path, "rb") as f:
            last = chain_lock.read_tail(f)
    except (OSError, chain_lock.ChainTailError):
        return None
    return last.get("hash") if isinstance(last, dict) else None


# ── enforcement ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Stop:
    status: str    # STOPPED_BUDGET | BLOCKED
    #: STOPPED_BUDGET: max_steps | max_tool_calls | max_tokens | max_wall_s | repeated_failure
    #: BLOCKED:        not_permitted | context_invalid | ledger_unavailable
    reason: str
    detail: str

    def __str__(self) -> str:
        return f"{self.status} ({self.reason}): {self.detail}"


class BudgetMeter:
    """Counts what a run spends against its context's budget. Once stopped, it stays stopped:
    every later check returns the same Stop, so a caller cannot spend past a cap by asking again."""

    def __init__(self, context: TaskRunContext, clock: Callable[[], float] = time.monotonic):
        self.context = context
        self._clock = clock
        self._started: Optional[float] = None
        self.steps = 0
        self.tool_calls = 0
        self.tokens = 0
        self.tokens_estimated = False
        self.failures_in_a_row = 0
        self.stop: Optional[Stop] = None

    # -- internals --
    def _halt(self, status: str, reason: str, detail: str) -> Stop:
        if self.stop is None:
            self.stop = Stop(status, reason, f"run {self.context.run_id}: {detail}")
        return self.stop

    def elapsed_s(self) -> float:
        return 0.0 if self._started is None else max(0.0, self._clock() - self._started)

    def _standing(self) -> Optional[Stop]:
        """Caps that hold at any moment: the clock, tokens, failures in a row."""
        if self.stop is not None:
            return self.stop
        b = self.context.budget
        if self.elapsed_s() >= b.max_wall_s:
            return self._halt(STOPPED_BUDGET, "max_wall_s",
                              f"{self.elapsed_s():.0f}s elapsed of {b.max_wall_s:.0f}s allowed")
        if self.tokens >= b.max_tokens:
            return self._halt(STOPPED_BUDGET, "max_tokens",
                              f"{self.tokens:,} tokens used of {b.max_tokens:,} allowed"
                              + (" (estimated from text length)" if self.tokens_estimated else ""))
        if self.failures_in_a_row >= b.max_repeated_failures:
            return self._halt(STOPPED_BUDGET, "repeated_failure",
                              f"{self.failures_in_a_row} failures in a row (limit {b.max_repeated_failures})")
        return None

    # -- the caller's checkpoints --
    def check(self) -> Optional[Stop]:
        return self._standing()

    def begin_step(self) -> Optional[Stop]:
        """Before each model turn (or sprint candidate). Charges the step when it may start."""
        if self._started is None:
            self._started = self._clock()
        stop = self._standing()
        if stop is not None:
            return stop
        b = self.context.budget
        if self.steps >= b.max_steps:
            return self._halt(STOPPED_BUDGET, "max_steps", f"{self.steps} of {b.max_steps} steps used, no answer")
        self.steps += 1
        return None

    def charge_tokens(self, reported: Optional[int], text: str = "") -> None:
        """Tokens for one model turn: the backend's own count when it gave one, otherwise one
        token per 4 UTF-8 bytes of the reply, rounded up, and flagged as an estimate."""
        if isinstance(reported, int) and not isinstance(reported, bool) and reported > 0:
            self.tokens += reported
        else:
            self.tokens += (len(text.encode("utf-8")) + 3) // 4
            self.tokens_estimated = True

    def begin_tool_call(self) -> Optional[Stop]:
        """Before each tool call. Charges the call when it may run."""
        stop = self._standing()
        if stop is not None:
            return stop
        b = self.context.budget
        if self.tool_calls >= b.max_tool_calls:
            return self._halt(STOPPED_BUDGET, "max_tool_calls",
                              f"{self.tool_calls} of {b.max_tool_calls} tool calls used")
        self.tool_calls += 1
        return None

    def note(self, ok: bool) -> None:
        """The outcome of a tool call or candidate: a success ends a run of failures."""
        self.failures_in_a_row = 0 if ok else self.failures_in_a_row + 1

    def block(self, detail: str) -> Stop:
        """The run asked for something its context does not permit."""
        return self._halt(BLOCKED, "not_permitted", detail)

    def usage(self) -> Dict[str, Any]:
        b = self.context.budget
        return {
            "steps": self.steps, "max_steps": b.max_steps,
            "tool_calls": self.tool_calls, "max_tool_calls": b.max_tool_calls,
            "tokens": self.tokens, "max_tokens": b.max_tokens, "tokens_estimated": self.tokens_estimated,
            "wall_s": round(self.elapsed_s(), 3), "max_wall_s": b.max_wall_s,
            "failures_in_a_row": self.failures_in_a_row, "max_repeated_failures": b.max_repeated_failures,
        }


def require(context: TaskRunContext, *, propose_writes: bool = False, run_commands: bool = False,
            max_writes: int = 1) -> Optional[Stop]:
    """BLOCKED when a run is being started with more than its context permits; None otherwise.
    Checked before any model is called."""
    asked = []
    if propose_writes and not context.permissions.propose_writes:
        asked.append("to propose writes")
    if run_commands and not context.permissions.run_commands:
        asked.append("to run commands")
    if max_writes > context.budget.max_writes:
        asked.append(f"up to {max_writes} writes (budget: {context.budget.max_writes})")
    if not asked:
        return None
    return Stop(BLOCKED, "not_permitted",
                f"run {context.run_id}: started {' and '.join(asked)}, which its context does not permit")


# ── the record ───────────────────────────────────────────────────────────────────────────────

class TaskLedger:
    """Hash-chained JSONL of task runs: one ``task_open`` and one ``task_close`` per run."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or DEFAULT_LEDGER

    def _append(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        from osiris_cli import chain_lock

        def build(last: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            entry["prev"] = last["hash"] if last else GENESIS
            entry["hash"] = sha256_text(json.dumps(entry, sort_keys=True, ensure_ascii=False))
            return entry

        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            return chain_lock.append_chained(self.path, build, ensure_ascii=False)
        except (OSError, chain_lock.ChainTailError, chain_lock.ChainLockTimeout) as exc:
            raise TaskLedgerError(f"task ledger {self.path} could not be written "
                                  f"({type(exc).__name__}: {exc})") from None

    def open(self, context: TaskRunContext) -> Dict[str, Any]:
        """Record that a run was issued this context. Raises TaskLedgerError if it cannot be
        recorded, or if this run id was already opened: the task must not start."""
        if any(e.get("run_id") == context.run_id for e in self.entries()):
            raise TaskLedgerError(f"run {context.run_id} is already in the task ledger")
        return self._append({"schema": LEDGER_SCHEMA, "kind": "task_open",
                             "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             "run_id": context.run_id, "context": context.to_dict()})

    def close(self, context: TaskRunContext, status: str, *, stop: Optional[Stop] = None,
              usage: Optional[Dict[str, Any]] = None, detail: str = "") -> Dict[str, Any]:
        return self._append({"schema": LEDGER_SCHEMA, "kind": "task_close",
                             "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             "run_id": context.run_id, "context_sha256": context.context_sha256,
                             "status": status, "stop_reason": stop.reason if stop else None,
                             "detail": (stop.detail if stop else detail)[:500], "usage": usage or {}})

    def entries(self) -> Iterable[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        out = []
        try:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.append(json.loads(line))
        except (OSError, ValueError) as exc:
            raise TaskLedgerError(f"task ledger {self.path} cannot be read ({exc})") from None
        return out

    def verify(self) -> Dict[str, Any]:
        """The exchange log's check (probes.verify_chain), plus: every recorded context still
        hashes to its digest, and every close names an opened run and that run's context."""
        from osiris_cli.probes import verify_chain
        result = dict(verify_chain(self.path))
        if not result["intact"]:
            return result
        opened: Dict[str, str] = {}
        for i, e in enumerate(self.entries()):
            if e.get("kind") == "task_open":
                try:
                    ctx = TaskRunContext.from_dict(e.get("context"))
                except TaskContextError as exc:
                    return dict(result, intact=False, status="TAMPERED", exit_code=1, broken_at=i, error=str(exc))
                opened[ctx.run_id] = ctx.context_sha256
            elif e.get("kind") == "task_close":
                if opened.get(e.get("run_id")) != e.get("context_sha256"):
                    return dict(result, intact=False, status="TAMPERED", exit_code=1, broken_at=i,
                                error=f"entry {i} closes a run that was not opened with that context")
        return result

    def context(self, run_id: str) -> TaskRunContext:
        for e in self.entries():
            if e.get("kind") == "task_open" and e.get("run_id") == run_id:
                return TaskRunContext.from_dict(e.get("context"))
        raise TaskContextError(f"no task run {run_id} in {self.path}")

    def open_runs(self) -> Tuple[str, ...]:
        """Runs opened and never closed (a crash, or still running)."""
        state: Dict[str, bool] = {}
        for e in self.entries():
            if e.get("kind") == "task_open":
                state[e["run_id"]] = True
            elif e.get("kind") == "task_close":
                state[e.get("run_id")] = False
        return tuple(r for r, is_open in state.items() if is_open)
