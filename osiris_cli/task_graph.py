"""OSIRIS DAG Task Graph & Server-Owned State Machine (§5, §6, §7, REQ-GRAPH-1..3).

A task is a directed acyclic graph (DAG) of nodes. Every node has a policy-classified tier,
an action digest, explicit acceptance criteria, an optional human/signed grant, disk-hashed
artifacts, and a separate-process verifier check (`osiris_cli.task_verifier`).

States (§5):
    DRAFT -> AWAITING_APPROVAL -> AUTHORIZED -> EXECUTING -> VERIFYING -> VERIFIED
    Any before VERIFIED -> FAILED
    DRAFT, AWAITING_APPROVAL, AUTHORIZED -> BLOCKED
    Any before EXECUTING -> CANCELLED

Nothing returns from a terminal stop (VERIFIED, FAILED, BLOCKED, CANCELLED). Every state
transition appends one canonical `exec-canon-1` receipt to `~/.osiris/task_receipts.jsonl`.
"""

from __future__ import annotations

from collections import deque
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
import uuid

from .exec_kernel import (
    ExecJournal,
    ExecutionGrant,
    _default_state_home,
    _last_line,
    canon,
    default_artifacts_dir,
    default_journal_path,
    mint_grant,
    run as kernel_run,
    verify_grant,
)
from .exec_policy import (
    AGENT_ORIGINS,
    CommandRequest,
    ORIGINS,
    classify,
    load_policy,
    scrubbed_env,
)
from .orchestration.governance_scope import protected_reason
from .task_receipts import TaskReceiptLedger, build_transition_body, default_task_receipts_path
from .task_verifier import run_separate_verifier

__all__ = [
    "ALL_STATES",
    "DEFAULT_MAX_NODES",
    "TERMINAL_STATES",
    "TRANSITIONS",
    "ApprovalRefusedError",
    "ChainIntegrityError",
    "TaskGraphEngine",
    "TaskGraphError",
    "compute_node_action_digest",
    "default_task_graphs_dir",
]

ALL_STATES = frozenset(
    {
        "DRAFT",
        "AWAITING_APPROVAL",
        "AUTHORIZED",
        "EXECUTING",
        "VERIFYING",
        "VERIFIED",
        "FAILED",
        "BLOCKED",
        "CANCELLED",
    }
)
TERMINAL_STATES = frozenset({"VERIFIED", "FAILED", "BLOCKED", "CANCELLED"})
TRANSITIONS: Mapping[str, frozenset[str]] = {
    "DRAFT": frozenset({"AWAITING_APPROVAL", "AUTHORIZED", "BLOCKED", "CANCELLED", "FAILED"}),
    "AWAITING_APPROVAL": frozenset({"AUTHORIZED", "BLOCKED", "CANCELLED", "FAILED"}),
    "AUTHORIZED": frozenset({"EXECUTING", "BLOCKED", "CANCELLED", "FAILED"}),
    "EXECUTING": frozenset({"VERIFYING", "FAILED"}),
    "VERIFYING": frozenset({"VERIFIED", "FAILED"}),
    "VERIFIED": frozenset(),
    "FAILED": frozenset(),
    "BLOCKED": frozenset(),
    "CANCELLED": frozenset(),
}

DEFAULT_MAX_NODES = 20
DEFAULT_TOKEN_BUDGET = 16384
DEFAULT_WALLTIME_MS = 300_000


class TaskGraphError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


class ChainIntegrityError(TaskGraphError):
    def __init__(self, message: str):
        super().__init__("BROKEN_RECEIPT_CHAIN", message, status=409)


class ApprovalRefusedError(TaskGraphError):
    def __init__(self, code: str, message: str, status: int = 403):
        super().__init__(code, message, status=status)


def default_task_graphs_dir() -> Path:
    """Resolve ~/.osiris/task_graphs lazily without touching disk on import."""
    override = os.environ.get("OSIRIS_TASK_GRAPHS_DIR")
    if override:
        return Path(override).expanduser()
    return _default_state_home() / "task_graphs"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file_if_exists(path: Path) -> str:
    if not path.is_file():
        return "ABSENT"
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_node_action_digest(node: Mapping[str, Any], policy_sha256: str) -> str:
    """Compute canonical action digest bound by approval/grant (REQ-AUTH-2)."""
    kind = str(node.get("kind") or "")
    if kind == "run_command":
        payload = {
            "kind": "run_command",
            "argv": [str(tok) for tok in (node.get("argv") or [])],
            "cwd": str(node.get("cwd") or ""),
            "policy_sha256": str(policy_sha256),
        }
    elif kind == "write_file":
        content = str(node.get("content") or "")
        payload = {
            "kind": "write_file",
            "target_path": str(node.get("target_path") or ""),
            "pre_image_sha256": str(node.get("pre_image_sha256") or "ABSENT"),
            "post_image_sha256": _sha256_bytes(content.encode("utf-8")),
            "policy_sha256": str(policy_sha256),
        }
    elif kind == "simulate":
        payload = {
            "kind": "simulate",
            "source_sha256": _sha256_bytes(str(node.get("source") or "").encode("utf-8")),
            "shots": int(node.get("shots") or 2000),
            "seed": int(node.get("seed") or 1234),
            "policy_sha256": str(policy_sha256),
        }
    else:
        raise TaskGraphError("INVALID_NODE_KIND", f"unsupported node kind: {kind!r}", 400)
    return _sha256_bytes(canon(payload).encode("utf-8"))


def _toposort_check(nodes: Sequence[Mapping[str, Any]]) -> List[str]:
    """Validate DAG acyclicity and dependency existence (REQ-GRAPH-1). Returns topo order."""
    ids = [str(n["node_id"]) for n in nodes]
    if len(ids) != len(set(ids)):
        raise TaskGraphError("DUPLICATE_NODE_ID", "duplicate node_id in task graph", 400)
    id_set = set(ids)
    indegree: Dict[str, int] = {nid: 0 for nid in ids}
    children: Dict[str, List[str]] = {nid: [] for nid in ids}

    for n in nodes:
        nid = str(n["node_id"])
        deps = n.get("depends_on") or []
        if not isinstance(deps, list):
            raise TaskGraphError("INVALID_DEPENDS_ON", f"node {nid}: depends_on must be a list", 400)
        for dep in deps:
            dep_id = str(dep)
            if dep_id == nid:
                raise TaskGraphError("GRAPH_CYCLE", f"node {nid} cannot depend on itself", 400)
            if dep_id not in id_set:
                raise TaskGraphError("UNKNOWN_DEPENDENCY", f"node {nid} depends on unknown node {dep_id!r}", 400)
            indegree[nid] += 1
            children[dep_id].append(nid)

    q = deque([nid for nid in ids if indegree[nid] == 0])
    order: List[str] = []
    while q:
        cur = q.popleft()
        order.append(cur)
        for nxt in children[cur]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                q.append(nxt)

    if len(order) != len(ids):
        raise TaskGraphError("GRAPH_CYCLE", "task graph contains a dependency cycle", 400)
    return order


class TaskGraphEngine:
    """Server-owned DAG task orchestrator, receipt chain manager, and verifier dispatcher."""

    def __init__(
        self,
        *,
        graphs_dir: Optional[object] = None,
        receipts_path: Optional[object] = None,
        exec_journal_path: Optional[object] = None,
        artifacts_dir: Optional[object] = None,
        referenced_ledgers: Optional[Mapping[str, object]] = None,
        default_workdir: Optional[str] = None,
        slice_runner: Optional[Callable[..., Dict[str, Any]]] = None,
    ) -> None:
        self.graphs_dir = Path(graphs_dir) if graphs_dir is not None else default_task_graphs_dir()
        self.receipts = TaskReceiptLedger(receipts_path if receipts_path is not None else default_task_receipts_path())
        self.exec_journal = ExecJournal(exec_journal_path if exec_journal_path is not None else default_journal_path())
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else default_artifacts_dir()
        self._extra_refs: Dict[str, Path] = {}
        if referenced_ledgers:
            for k, v in referenced_ledgers.items():
                if v is not None:
                    self._extra_refs[k] = Path(v)
        self.default_workdir = os.path.realpath(default_workdir) if default_workdir else os.path.realpath(os.getcwd())
        self._slice_runner = slice_runner
        self._lock = threading.Lock()
        # Live in-memory session grants indexed by (graph_id, node_id) so kernel grants stay process-bound
        self._session_grants: Dict[Tuple[str, str], Tuple[ExecutionGrant, List[str], str]] = {}

    def _referenced_ledgers(self) -> Dict[str, Path]:
        refs: Dict[str, Path] = {"exec_journal": self.exec_journal.path}
        refs.update(self._extra_refs)
        return refs

    def verify_chains(self) -> Tuple[bool, int, str]:
        """Verify task_receipts.jsonl and every referenced chain (REQ-EV-4)."""
        return self.receipts.verify(referenced_ledgers=self._referenced_ledgers())

    def _require_intact_chains(self) -> None:
        ok, _, err = self.verify_chains()
        if not ok:
            raise ChainIntegrityError(f"refusing task operation while receipt chain is broken: {err}")

    def _graph_path(self, graph_id: str) -> Path:
        if not isinstance(graph_id, str) or not graph_id.startswith("graph-") or len(graph_id) != 18:
            raise TaskGraphError("BAD_GRAPH_ID", "graph_id must look like graph-<12 hex>", 400)
        suffix = graph_id[6:]
        if not all(c in "0123456789abcdef" for c in suffix):
            raise TaskGraphError("BAD_GRAPH_ID", "graph_id must look like graph-<12 hex>", 400)
        return self.graphs_dir / f"{graph_id}.json"

    def _save_graph(self, graph: Dict[str, Any]) -> None:
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
        path = self._graph_path(graph["graph_id"])
        tmp = path.with_suffix(".json.tmp")
        data = json.dumps(graph, sort_keys=True, indent=2) + "\n"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    def load_graph(self, graph_id: str) -> Dict[str, Any]:
        path = self._graph_path(graph_id)
        if not path.is_file():
            raise TaskGraphError("NO_GRAPH", f"no task graph {graph_id}", 404)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise TaskGraphError("CORRUPT_GRAPH", f"could not read graph {graph_id}: {exc}", 500) from exc

    def _transition(
        self,
        graph: Dict[str, Any],
        node: Dict[str, Any],
        target_state: str,
        *,
        actor_role: str,
        actor_origin: str,
        reason: str,
        session_id: Optional[str] = None,
        grant: Optional[Dict[str, Any]] = None,
        refs: Optional[Mapping[str, Optional[str]]] = None,
        verification: Optional[Dict[str, Any]] = None,
        evidence_class: str = "COMPUTED",
    ) -> Dict[str, Any]:
        current = node.get("state")
        if current is not None:
            allowed = TRANSITIONS.get(current, frozenset())
            if target_state not in allowed:
                raise TaskGraphError(
                    "INVALID_STATE_TRANSITION",
                    f"node {node['node_id']}: transition {current} -> {target_state} is not permitted",
                    409,
                )
        now_ms = int(time.time() * 1000)
        body = build_transition_body(
            graph_id=graph["graph_id"],
            node_id=node["node_id"],
            state_before=current,
            state_after=target_state,
            actor_role=actor_role,
            actor_origin=actor_origin,
            action_digest=node["action_digest"],
            reason=reason,
            session_id=session_id,
            grant=grant,
            refs=refs,
            verification=verification,
            evidence_class=evidence_class,
            hardware=False,
            recorded_ms=now_ms,
        )
        row = self.receipts.append(body)
        node["state"] = target_state
        node["updated_ms"] = now_ms
        node["last_reason"] = reason
        node.setdefault("receipt_hashes", []).append(row["hash"])
        graph["updated_ms"] = now_ms
        return row

    def _compute_graph_status(self, graph: Dict[str, Any]) -> str:
        states = [n["state"] for n in graph["nodes"]]
        if all(s == "VERIFIED" for s in states):
            return "VERIFIED"
        if any(s == "FAILED" for s in states):
            return "FAILED"
        if any(s == "BLOCKED" for s in states):
            return "BLOCKED"
        if all(s == "CANCELLED" for s in states):
            return "CANCELLED"
        if any(s in ("EXECUTING", "VERIFYING") for s in states):
            return "EXECUTING"
        if any(s == "AWAITING_APPROVAL" for s in states):
            return "AWAITING_APPROVAL"
        if any(s == "AUTHORIZED" for s in states):
            return "AUTHORIZED"
        return "DRAFT"

    def _find_node(self, graph: Dict[str, Any], node_id: str) -> Dict[str, Any]:
        for node in graph["nodes"]:
            if node["node_id"] == node_id:
                return node
        raise TaskGraphError("NO_NODE", f"graph {graph['graph_id']} has no node {node_id!r}", 404)

    def _public_graph(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Return safe serialized view (strip write contents from summaries, keep digests)."""
        nodes_out = []
        for n in graph["nodes"]:
            item = {k: v for k, v in n.items() if k != "content"}
            nodes_out.append(item)
        return {
            "graph_id": graph["graph_id"],
            "objective": graph["objective"],
            "status": self._compute_graph_status(graph),
            "workdir": graph["workdir"],
            "created_ms": graph["created_ms"],
            "updated_ms": graph["updated_ms"],
            "budgets": graph["budgets"],
            "nodes": nodes_out,
        }

    def list_graphs(self) -> Dict[str, Any]:
        ok, count, err = self.verify_chains()
        rows: List[Dict[str, Any]] = []
        if self.graphs_dir.is_dir():
            for name in os.listdir(self.graphs_dir):
                if not name.endswith(".json"):
                    continue
                try:
                    g = json.loads((self.graphs_dir / name).read_text(encoding="utf-8"))
                except Exception:
                    continue
                nodes = g.get("nodes") or []
                state_counts: Dict[str, int] = {}
                for n in nodes:
                    st = str(n.get("state") or "DRAFT")
                    state_counts[st] = state_counts.get(st, 0) + 1
                rows.append(
                    {
                        "graph_id": g["graph_id"],
                        "objective": g.get("objective", "")[:240],
                        "status": self._compute_graph_status(g),
                        "node_count": len(nodes),
                        "state_counts": state_counts,
                        "created_ms": g.get("created_ms", 0),
                        "updated_ms": g.get("updated_ms", 0),
                    }
                )
        rows.sort(key=lambda r: (r["updated_ms"], r["graph_id"]), reverse=True)
        return {
            "chain_valid": ok,
            "chain_error": err or None,
            "receipt_count": count,
            "count": len(rows),
            "graphs": rows[:100],
        }

    def get_graph(self, graph_id: str) -> Dict[str, Any]:
        ok, _, err = self.verify_chains()
        graph = self.load_graph(graph_id)
        receipts = self.receipts.rows(graph_id=graph_id, limit=200)
        return {
            "graph": self._public_graph(graph),
            "chain_valid": ok,
            "chain_error": err or None,
            "receipts": receipts,
        }

    def create_graph(
        self,
        *,
        objective: str,
        nodes: Sequence[Mapping[str, Any]],
        workdir: Optional[str] = None,
        origin: str = "model",
        max_nodes: int = DEFAULT_MAX_NODES,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        walltime_ms: int = DEFAULT_WALLTIME_MS,
        client_request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Decompose and open a DAG task graph (REQ-GRAPH-1, REQ-GEN-3..4, REQ-EV-1, REQ-EV-4)."""
        with self._lock:
            self._require_intact_chains()

            if not isinstance(objective, str) or not objective.strip():
                raise TaskGraphError("INVALID_INPUT", "objective must be a non-empty string", 400)
            if origin not in ORIGINS:
                raise TaskGraphError("INVALID_ORIGIN", f"unknown origin {origin!r}", 400)
            if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)) or len(nodes) == 0:
                raise TaskGraphError("INVALID_INPUT", "nodes must be a non-empty list", 400)
            if not isinstance(max_nodes, int) or isinstance(max_nodes, bool) or max_nodes < 1:
                raise TaskGraphError("INVALID_INPUT", "max_nodes must be a positive integer", 400)
            if len(nodes) > max_nodes:
                raise TaskGraphError(
                    "NODE_CAP_EXCEEDED",
                    f"task graph has {len(nodes)} nodes, exceeding max_nodes={max_nodes}",
                    400,
                )

            # Idempotency check on client_request_id across existing graphs (REQ-API-4)
            if client_request_id:
                existing = self._find_graph_by_request_id(client_request_id)
                if existing is not None:
                    ok, _, err = self.verify_chains()
                    return {
                        "graph": self._public_graph(existing),
                        "chain_valid": ok,
                        "chain_error": err or None,
                        "idempotent_replay": True,
                    }

            resolved_workdir = os.path.realpath(workdir) if workdir else self.default_workdir
            if not os.path.isdir(resolved_workdir):
                raise TaskGraphError("INVALID_WORKDIR", f"workdir is not a directory: {resolved_workdir}", 400)

            _, policy_sha = load_policy()
            graph_id = "graph-" + uuid.uuid4().hex[:12]
            now_ms = int(time.time() * 1000)

            # First validate raw node structure and DAG acyclicity
            prepared_nodes: List[Dict[str, Any]] = []
            for idx, raw in enumerate(nodes, 1):
                if not isinstance(raw, Mapping):
                    raise TaskGraphError("INVALID_NODE", f"node {idx} must be an object", 400)
                if raw.get("command") is not None:
                    raise TaskGraphError(
                        "RAW_COMMAND_REFUSED",
                        "raw shell command strings are refused (REQ-PLAN-1); pass argv: list[str]",
                        400,
                    )
                node_id = str(raw.get("node_id") or f"node-{idx}").strip()
                if not node_id:
                    raise TaskGraphError("INVALID_NODE", f"node {idx} missing node_id", 400)
                title = str(raw.get("title") or f"Step {idx}").strip()[:200]
                kind = str(raw.get("kind") or "run_command").strip().lower()
                deps = [str(d) for d in (raw.get("depends_on") or [])]
                criteria = raw.get("acceptance_criteria")
                if not isinstance(criteria, list) or not criteria or not all(isinstance(c, str) and c.strip() for c in criteria):
                    raise TaskGraphError(
                        "MISSING_ACCEPTANCE_CRITERIA",
                        f"node {node_id} must declare non-empty acceptance_criteria (§6 step 5)",
                        400,
                    )

                node_entry: Dict[str, Any] = {
                    "node_id": node_id,
                    "title": title,
                    "kind": kind,
                    "depends_on": deps,
                    "origin": origin,
                    "acceptance_criteria": [c.strip() for c in criteria],
                    "state": None,
                    "tier": "FORBIDDEN",
                    "policy_decision": "DENY",
                    "policy_reasons": [],
                    "policy_sha256": policy_sha,
                    "action_digest": "",
                    "approved_action_digest": None,
                    "grant": None,
                    "artifact_path": None,
                    "artifact_sha256": None,
                    "exec_journal_hash": None,
                    "verification": None,
                    "receipt_hashes": [],
                    "created_ms": now_ms,
                    "updated_ms": now_ms,
                    "last_reason": "",
                }

                if kind == "run_command":
                    raw_argv = raw.get("argv")
                    if not isinstance(raw_argv, list) or not raw_argv or not all(isinstance(t, str) and t.strip() for t in raw_argv):
                        raise TaskGraphError(
                            "INVALID_ARGV",
                            f"node {node_id}: run_command requires non-empty argv: list[str]",
                            400,
                        )
                    raw_cwd = str(raw.get("cwd") or ".").strip()
                    cand_cwd = raw_cwd if os.path.isabs(raw_cwd) else os.path.join(resolved_workdir, raw_cwd)
                    node_cwd = os.path.realpath(cand_cwd)
                    # REQ-EXEC-2 / G-7: Tier comes ONLY from exec_policy.classify(argv), NEVER from prompt/title!
                    cls = classify(
                        CommandRequest(
                            argv=tuple(raw_argv),
                            cwd=node_cwd,
                            origin=origin,
                            root=resolved_workdir,
                        )
                    )
                    node_entry["argv"] = list(cls.normalized_argv) if cls.decision != "DENY" else list(raw_argv)
                    node_entry["cwd"] = node_cwd
                    node_entry["tier"] = cls.tier
                    node_entry["policy_decision"] = cls.decision
                    node_entry["policy_reasons"] = list(cls.reasons)
                    node_entry["policy_sha256"] = cls.policy_sha256
                elif kind == "write_file":
                    rel_target = str(raw.get("target_path") or "").strip()
                    if not rel_target:
                        raise TaskGraphError("INVALID_PATH", f"node {node_id}: write_file requires target_path", 400)
                    cand_path = rel_target if os.path.isabs(rel_target) else os.path.join(resolved_workdir, rel_target)
                    real_target = os.path.realpath(cand_path)
                    inside_root = real_target == resolved_workdir or real_target.startswith(resolved_workdir + os.sep)
                    content = raw.get("content")
                    if not isinstance(content, str):
                        raise TaskGraphError("INVALID_CONTENT", f"node {node_id}: write_file requires string content", 400)
                    node_entry["target_path"] = real_target
                    node_entry["content"] = content
                    node_entry["pre_image_sha256"] = _sha256_file_if_exists(Path(real_target))
                    node_entry["post_image_sha256"] = _sha256_bytes(content.encode("utf-8"))

                    prot = protected_reason(real_target, roots=[resolved_workdir])
                    if not inside_root:
                        node_entry["tier"] = "FORBIDDEN"
                        node_entry["policy_decision"] = "DENY"
                        node_entry["policy_reasons"] = [f"target_path outside workdir root: {rel_target}"]
                    elif prot is not None and origin in AGENT_ORIGINS:
                        node_entry["tier"] = "FORBIDDEN"
                        node_entry["policy_decision"] = "DENY"
                        node_entry["policy_reasons"] = [prot]
                    else:
                        node_entry["tier"] = "MUTATE"
                        node_entry["policy_decision"] = "ASK"
                        node_entry["policy_reasons"] = [
                            prot if prot is not None else "file mutation requires explicit human approval"
                        ]
                        if prot is not None:
                            node_entry["requires_signed_grant"] = True
                elif kind == "simulate":
                    source = raw.get("source")
                    if not isinstance(source, str) or not source.strip():
                        raise TaskGraphError("INVALID_SOURCE", f"node {node_id}: simulate requires DNA-Lang source", 400)
                    shots = int(raw.get("shots") or 1000)
                    seed = int(raw.get("seed") or 1234)
                    node_entry["source"] = source
                    node_entry["shots"] = shots
                    node_entry["seed"] = seed
                    node_entry["tier"] = "VERIFY"
                    node_entry["policy_decision"] = "ALLOW"
                    node_entry["policy_reasons"] = ["authoritative local simulator (SIMULATED, hardware=false)"]
                else:
                    raise TaskGraphError("INVALID_NODE_KIND", f"node {node_id}: unsupported kind {kind!r}", 400)

                node_entry["action_digest"] = compute_node_action_digest(node_entry, node_entry["policy_sha256"])
                prepared_nodes.append(node_entry)

            # Enforce DAG acyclicity before recording any state or receipt (REQ-GRAPH-1)
            _toposort_check(prepared_nodes)

            graph: Dict[str, Any] = {
                "graph_id": graph_id,
                "objective": objective.strip()[:2000],
                "workdir": resolved_workdir,
                "created_ms": now_ms,
                "updated_ms": now_ms,
                "budgets": {
                    "max_nodes": max_nodes,
                    "review_loops_max": 2,
                    "token_budget": int(token_budget),
                    "walltime_ms": int(walltime_ms),
                },
                "client_request_ids": [client_request_id] if client_request_id else [],
                "processed_requests": {},
                "nodes": prepared_nodes,
            }

            # Record DRAFT then initial policy-driven transition (ALLOW->AUTHORIZED, ASK->AWAITING_APPROVAL, DENY->BLOCKED)
            for node in graph["nodes"]:
                self._transition(
                    graph,
                    node,
                    "DRAFT",
                    actor_role="orchestrator",
                    actor_origin="system",
                    reason="decomposed into task graph DAG",
                )
                dec = node["policy_decision"]
                if dec == "ALLOW":
                    self._transition(
                        graph,
                        node,
                        "AUTHORIZED",
                        actor_role="orchestrator",
                        actor_origin="system",
                        reason=f"policy decision ALLOW ({node['tier']}); no grant needed",
                    )
                elif dec == "ASK":
                    self._transition(
                        graph,
                        node,
                        "AWAITING_APPROVAL",
                        actor_role="orchestrator",
                        actor_origin="system",
                        reason=f"policy decision ASK ({node['tier']}): {'; '.join(node['policy_reasons'])}",
                    )
                else:
                    self._transition(
                        graph,
                        node,
                        "BLOCKED",
                        actor_role="orchestrator",
                        actor_origin="system",
                        reason=f"policy decision DENY ({node['tier']}): {'; '.join(node['policy_reasons'])}",
                    )

            self._save_graph(graph)
            ok, _, err = self.verify_chains()
            return {
                "graph": self._public_graph(graph),
                "chain_valid": ok,
                "chain_error": err or None,
                "idempotent_replay": False,
            }

    def _find_graph_by_request_id(self, client_request_id: str) -> Optional[Dict[str, Any]]:
        if not self.graphs_dir.is_dir():
            return None
        for name in os.listdir(self.graphs_dir):
            if not name.endswith(".json"):
                continue
            try:
                g = json.loads((self.graphs_dir / name).read_text(encoding="utf-8"))
            except Exception:
                continue
            if client_request_id in (g.get("client_request_ids") or []):
                return g
        return None

    def approve_node(
        self,
        graph_id: str,
        node_id: str,
        *,
        action_digest: str,
        actor_origin: str = "human",
        session_id: Optional[str] = None,
        client_request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Approve a node in AWAITING_APPROVAL and mint a bound kernel session grant (REQ-AUTH-1..5)."""
        with self._lock:
            self._require_intact_chains()
            graph = self.load_graph(graph_id)
            node = self._find_node(graph, node_id)

            # REQ-API-4: Idempotent replay of client_request_id changes nothing
            if client_request_id and client_request_id in (graph.get("processed_requests") or {}):
                ok, _, err = self.verify_chains()
                return {
                    "graph": self._public_graph(graph),
                    "node": {k: v for k, v in node.items() if k != "content"},
                    "chain_valid": ok,
                    "chain_error": err or None,
                    "idempotent_replay": True,
                }

            # REQ-AUTH-5: Agent origins (model, skill, claude_code) can NEVER confirm or approve!
            if actor_origin in AGENT_ORIGINS or actor_origin != "human":
                self.receipts.append(
                    build_transition_body(
                        graph_id=graph_id,
                        node_id=node_id,
                        state_before=node["state"],
                        state_after=node["state"],
                        actor_role="operator",
                        actor_origin=str(actor_origin),
                        action_digest=node["action_digest"],
                        reason="REFUSED: agent origin cannot confirm or approve a task node (REQ-AUTH-5)",
                        session_id=session_id,
                    )
                )
                raise ApprovalRefusedError(
                    "AGENT_CANNOT_CONFIRM",
                    f"origin {actor_origin!r} cannot approve node {node_id} (REQ-AUTH-5)",
                    status=403,
                )

            if not session_id or not str(session_id).strip():
                raise ApprovalRefusedError("SESSION_ID_REQUIRED", "human approval requires a live session_id", 400)

            if node["state"] != "AWAITING_APPROVAL":
                raise TaskGraphError(
                    "NODE_NOT_AWAITING_APPROVAL",
                    f"node {node_id} is in state {node['state']}, not AWAITING_APPROVAL",
                    409,
                )

            if node.get("requires_signed_grant"):
                raise ApprovalRefusedError(
                    "SIGNED_GRANT_REQUIRED",
                    f"node {node_id} targets a protected governance path and requires an external signed grant (REQ-AUTH-3)",
                    403,
                )

            # Recompute digest from current node definition and require exact match (REQ-AUTH-2)
            current_digest = compute_node_action_digest(node, node["policy_sha256"])
            if not isinstance(action_digest, str) or action_digest != current_digest:
                self.receipts.append(
                    build_transition_body(
                        graph_id=graph_id,
                        node_id=node_id,
                        state_before=node["state"],
                        state_after=node["state"],
                        actor_role="operator",
                        actor_origin="human",
                        action_digest=current_digest,
                        reason="REFUSED: approval action_digest does not match current node definition (REQ-AUTH-2)",
                        session_id=session_id,
                    )
                )
                raise ApprovalRefusedError(
                    "ACTION_DIGEST_MISMATCH",
                    "action_digest does not match the current node definition",
                    409,
                )

            # Mint a real bound grant via exec_kernel.mint_grant (REQ-AUTH-1, REQ-AUTH-2)
            env_keys = sorted(scrubbed_env(os.environ))
            if node["kind"] == "run_command":
                kernel_grant = mint_grant(tuple(node["argv"]), node["cwd"], env_keys, node["policy_sha256"])
            else:
                # Bind write_file grant digest to realpath + pre/post hashes + policy
                kernel_grant = mint_grant(
                    (node["kind"], node["target_path"], node["pre_image_sha256"], node["post_image_sha256"]),
                    graph["workdir"],
                    env_keys,
                    node["policy_sha256"],
                )

            self._session_grants[(graph_id, node_id)] = (kernel_grant, env_keys, node["policy_sha256"])
            grant_meta = {
                "kind": "session",
                "grant_id": kernel_grant.grant_id,
                "expires_ms": kernel_grant.expires_ms,
                "session_id": str(session_id).strip(),
            }
            node["approved_action_digest"] = current_digest
            node["grant"] = grant_meta

            self._transition(
                graph,
                node,
                "AUTHORIZED",
                actor_role="operator",
                actor_origin="human",
                session_id=str(session_id).strip(),
                grant={"kind": "session", "grant_id": kernel_grant.grant_id},
                reason=f"operator approved action_digest={current_digest[:12]}",
            )
            if client_request_id:
                graph.setdefault("processed_requests", {})[client_request_id] = {
                    "action": "approve",
                    "node_id": node_id,
                    "grant_id": kernel_grant.grant_id,
                }
            self._save_graph(graph)
            ok, _, err = self.verify_chains()
            return {
                "graph": self._public_graph(graph),
                "node": {k: v for k, v in node.items() if k != "content"},
                "chain_valid": ok,
                "chain_error": err or None,
                "idempotent_replay": False,
            }

    def cancel_node(
        self,
        graph_id: str,
        node_id: Optional[str] = None,
        *,
        session_id: Optional[str] = None,
        reason: str = "cancelled by operator",
        client_request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Cancel one unstarted node (or all unstarted nodes in the graph)."""
        with self._lock:
            self._require_intact_chains()
            graph = self.load_graph(graph_id)
            if client_request_id and client_request_id in (graph.get("processed_requests") or {}):
                ok, _, err = self.verify_chains()
                return {
                    "graph": self._public_graph(graph),
                    "chain_valid": ok,
                    "chain_error": err or None,
                    "idempotent_replay": True,
                }
            targets = [self._find_node(graph, node_id)] if node_id else [
                n for n in graph["nodes"] if n["state"] in ("DRAFT", "AWAITING_APPROVAL", "AUTHORIZED")
            ]
            if node_id and targets[0]["state"] not in ("DRAFT", "AWAITING_APPROVAL", "AUTHORIZED"):
                raise TaskGraphError(
                    "CANNOT_CANCEL",
                    f"node {node_id} in state {targets[0]['state']} cannot be cancelled",
                    409,
                )
            for node in targets:
                self._session_grants.pop((graph_id, node["node_id"]), None)
                self._transition(
                    graph,
                    node,
                    "CANCELLED",
                    actor_role="operator",
                    actor_origin="human",
                    session_id=session_id,
                    reason=reason,
                )
            if client_request_id:
                graph.setdefault("processed_requests", {})[client_request_id] = {
                    "action": "cancel",
                    "node_id": node_id,
                }
            self._save_graph(graph)
            ok, _, err = self.verify_chains()
            return {
                "graph": self._public_graph(graph),
                "chain_valid": ok,
                "chain_error": err or None,
                "idempotent_replay": False,
            }

    def _cascade_stop_dependents(self, graph: Dict[str, Any], failed_node_id: str) -> None:
        """Stop unstarted downstream nodes when a prerequisite fails (REQ-EXEC-5)."""
        stopped = {failed_node_id}
        changed = True
        while changed:
            changed = False
            for node in graph["nodes"]:
                if node["state"] in ("DRAFT", "AWAITING_APPROVAL", "AUTHORIZED"):
                    if any(dep in stopped for dep in (node.get("depends_on") or [])):
                        self._session_grants.pop((graph["graph_id"], node["node_id"]), None)
                        self._transition(
                            graph,
                            node,
                            "BLOCKED",
                            actor_role="orchestrator",
                            actor_origin="system",
                            reason=f"prerequisite node {failed_node_id} did not reach VERIFIED",
                        )
                        stopped.add(node["node_id"])
                        changed = True

    def execute_node(
        self,
        graph_id: str,
        node_id: str,
        *,
        client_request_id: Optional[str] = None,
        pre_verify_hook: Optional[Callable[[Path], None]] = None,
    ) -> Dict[str, Any]:
        """Run one AUTHORIZED node through exec_kernel + separate-process verifier."""
        with self._lock:
            self._require_intact_chains()
            graph = self.load_graph(graph_id)
            node = self._find_node(graph, node_id)

            if client_request_id and client_request_id in (graph.get("processed_requests") or {}):
                ok, _, err = self.verify_chains()
                return {
                    "graph": self._public_graph(graph),
                    "node": {k: v for k, v in node.items() if k != "content"},
                    "chain_valid": ok,
                    "chain_error": err or None,
                    "idempotent_replay": True,
                }

            if node["state"] != "AUTHORIZED":
                raise TaskGraphError(
                    "NODE_NOT_AUTHORIZED",
                    f"node {node_id} is in state {node['state']}, not AUTHORIZED",
                    409,
                )

            # REQ-GRAPH-2: A node starts ONLY when every dependency is VERIFIED.
            for dep_id in node.get("depends_on") or []:
                dep = self._find_node(graph, dep_id)
                if dep["state"] in ("FAILED", "BLOCKED", "CANCELLED"):
                    self._transition(
                        graph,
                        node,
                        "BLOCKED",
                        actor_role="orchestrator",
                        actor_origin="system",
                        reason=f"dependency {dep_id} is in terminal state {dep['state']}",
                    )
                    self._save_graph(graph)
                    raise TaskGraphError(
                        "DEPENDENCY_FAILED",
                        f"node {node_id} blocked because dependency {dep_id} is {dep['state']}",
                        409,
                    )
                if dep["state"] != "VERIFIED":
                    raise TaskGraphError(
                        "DEPENDENCY_NOT_VERIFIED",
                        f"node {node_id} cannot start until dependency {dep_id} is VERIFIED (currently {dep['state']})",
                        409,
                    )

            # REQ-AUTH-2: Re-verify action digest & grant immediately before execution!
            recomputed_digest = compute_node_action_digest(node, node["policy_sha256"])
            stored_grant_entry = self._session_grants.pop((graph_id, node_id), None)
            grant_ref: Optional[Dict[str, Any]] = None

            if node["policy_decision"] != "ALLOW":
                approved_digest = node.get("approved_action_digest")
                if not approved_digest or recomputed_digest != approved_digest or recomputed_digest != node["action_digest"]:
                    self._transition(
                        graph,
                        node,
                        "FAILED",
                        actor_role="orchestrator",
                        actor_origin="system",
                        reason="ACTION_DIGEST_MISMATCH: node changed after approval (REQ-AUTH-2)",
                    )
                    self._cascade_stop_dependents(graph, node_id)
                    self._save_graph(graph)
                    ok, _, err = self.verify_chains()
                    return {
                        "graph": self._public_graph(graph),
                        "node": {k: v for k, v in node.items() if k != "content"},
                        "chain_valid": ok,
                        "chain_error": err or None,
                        "idempotent_replay": False,
                    }

                if stored_grant_entry is None:
                    self._transition(
                        graph,
                        node,
                        "FAILED",
                        actor_role="orchestrator",
                        actor_origin="system",
                        reason="GRANT_MISSING: no active session grant bound to this node",
                    )
                    self._cascade_stop_dependents(graph, node_id)
                    self._save_graph(graph)
                    ok, _, err = self.verify_chains()
                    return {
                        "graph": self._public_graph(graph),
                        "node": {k: v for k, v in node.items() if k != "content"},
                        "chain_valid": ok,
                        "chain_error": err or None,
                        "idempotent_replay": False,
                    }

                kernel_grant, env_keys, pol_sha = stored_grant_entry
                if node["kind"] == "run_command":
                    grant_ok = verify_grant(kernel_grant, tuple(node["argv"]), node["cwd"], env_keys, pol_sha)
                else:
                    grant_ok = verify_grant(
                        kernel_grant,
                        (node["kind"], node["target_path"], node["pre_image_sha256"], node["post_image_sha256"]),
                        graph["workdir"],
                        env_keys,
                        pol_sha,
                    )
                if not grant_ok:
                    self._transition(
                        graph,
                        node,
                        "FAILED",
                        actor_role="orchestrator",
                        actor_origin="system",
                        reason="GRANT_VERIFICATION_FAILED: grant expired or mismatched action (REQ-AUTH-2)",
                    )
                    self._cascade_stop_dependents(graph, node_id)
                    self._save_graph(graph)
                    ok, _, err = self.verify_chains()
                    return {
                        "graph": self._public_graph(graph),
                        "node": {k: v for k, v in node.items() if k != "content"},
                        "chain_valid": ok,
                        "chain_error": err or None,
                        "idempotent_replay": False,
                    }
                grant_ref = {"kind": "session", "grant_id": kernel_grant.grant_id}
            else:
                # Even ALLOW nodes refuse silent post-creation tampering
                if recomputed_digest != node["action_digest"]:
                    self._transition(
                        graph,
                        node,
                        "FAILED",
                        actor_role="orchestrator",
                        actor_origin="system",
                        reason="ACTION_DIGEST_MISMATCH: ALLOW node mutated after creation",
                    )
                    self._cascade_stop_dependents(graph, node_id)
                    self._save_graph(graph)
                    ok, _, err = self.verify_chains()
                    return {
                        "graph": self._public_graph(graph),
                        "node": {k: v for k, v in node.items() if k != "content"},
                        "chain_valid": ok,
                        "chain_error": err or None,
                        "idempotent_replay": False,
                    }

            # AUTHORIZED -> EXECUTING
            self._transition(
                graph,
                node,
                "EXECUTING",
                actor_role="orchestrator",
                actor_origin="system",
                grant=grant_ref,
                reason="grant verified and dependencies VERIFIED; starting adapter",
            )
            self._save_graph(graph)

            artifact_path: Optional[Path] = None
            artifact_sha256: Optional[str] = None
            exec_journal_hash: Optional[str] = None
            observed_exit_code: Optional[int] = None
            evidence_class = "COMPUTED"

            try:
                if node["kind"] == "run_command":
                    req = CommandRequest(
                        argv=tuple(node["argv"]),
                        cwd=node["cwd"],
                        origin=node["origin"],
                        root=graph["workdir"],
                    )
                    expected_argv = tuple(node["argv"])
                    expected_pol = node["policy_sha256"]
                    confirm_fn = (
                        None
                        if node["policy_decision"] == "ALLOW"
                        else (lambda _prompt, cls: cls.normalized_argv == expected_argv and cls.policy_sha256 == expected_pol)
                    )
                    cmd_res = kernel_run(
                        req,
                        confirm=confirm_fn,
                        journal=self.exec_journal,
                        artifacts_dir=self.artifacts_dir,
                    )
                    last_exec = _last_line(self.exec_journal.path)
                    if last_exec is not None:
                        exec_journal_hash = json.loads(last_exec.decode("utf-8")).get("hash")
                    observed_exit_code = cmd_res.exit_code
                    if cmd_res.decision in ("DENY", "ASK_DECLINED", "ERROR", "TIMEOUT") or not cmd_res.artifact_path:
                        self._transition(
                            graph,
                            node,
                            "FAILED",
                            actor_role="adapter",
                            actor_origin="system",
                            grant=grant_ref,
                            refs={"exec_journal_hash": exec_journal_hash, "artifact_sha256": cmd_res.artifact_sha256},
                            reason=f"kernel execution decision={cmd_res.decision}: {'; '.join(cmd_res.reasons)}",
                        )
                        self._cascade_stop_dependents(graph, node_id)
                        self._save_graph(graph)
                        ok, _, err = self.verify_chains()
                        return {
                            "graph": self._public_graph(graph),
                            "node": {k: v for k, v in node.items() if k != "content"},
                            "chain_valid": ok,
                            "chain_error": err or None,
                            "idempotent_replay": False,
                        }
                    artifact_path = Path(cmd_res.artifact_path)
                    artifact_sha256 = cmd_res.artifact_sha256
                elif node["kind"] == "write_file":
                    target = Path(node["target_path"])
                    current_pre = _sha256_file_if_exists(target)
                    if current_pre != node["pre_image_sha256"]:
                        self._transition(
                            graph,
                            node,
                            "FAILED",
                            actor_role="adapter",
                            actor_origin="system",
                            grant=grant_ref,
                            reason="PRE_IMAGE_MISMATCH: target file changed on disk since approval",
                        )
                        self._cascade_stop_dependents(graph, node_id)
                        self._save_graph(graph)
                        ok, _, err = self.verify_chains()
                        return {
                            "graph": self._public_graph(graph),
                            "node": {k: v for k, v in node.items() if k != "content"},
                            "chain_valid": ok,
                            "chain_error": err or None,
                            "idempotent_replay": False,
                        }
                    target.parent.mkdir(parents=True, exist_ok=True)
                    tmp_target = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.tmp")
                    raw_bytes = str(node["content"]).encode("utf-8")
                    fd = os.open(tmp_target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                    with os.fdopen(fd, "wb") as fh:
                        fh.write(raw_bytes)
                        fh.flush()
                        os.fsync(fh.fileno())
                    os.replace(tmp_target, target)
                    artifact_path = target
                    artifact_sha256 = _sha256_bytes(raw_bytes)
                    observed_exit_code = 0
                elif node["kind"] == "simulate":
                    from .vertical_slice import run_slice

                    runner = self._slice_runner or run_slice
                    sim_result = runner(
                        node["source"],
                        shots=node["shots"],
                        seed=node["seed"],
                    )
                    self.artifacts_dir.mkdir(parents=True, exist_ok=True)
                    sim_path = self.artifacts_dir / f"sim-{graph_id}-{node_id}.json"
                    raw_sim = (json.dumps(sim_result, sort_keys=True, indent=2) + "\n").encode("utf-8")
                    fd = os.open(sim_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                    with os.fdopen(fd, "wb") as fh:
                        fh.write(raw_sim)
                        fh.flush()
                        os.fsync(fh.fileno())
                    artifact_path = sim_path
                    artifact_sha256 = _sha256_bytes(raw_sim)
                    observed_exit_code = 0 if sim_result.get("status") == "COMPLETED" else 1
                    evidence_class = "SIMULATED"
            except Exception as exc:
                self._transition(
                    graph,
                    node,
                    "FAILED",
                    actor_role="adapter",
                    actor_origin="system",
                    grant=grant_ref,
                    refs={"exec_journal_hash": exec_journal_hash},
                    reason=f"adapter error: {type(exc).__name__}: {exc}",
                )
                self._cascade_stop_dependents(graph, node_id)
                self._save_graph(graph)
                ok, _, err = self.verify_chains()
                return {
                    "graph": self._public_graph(graph),
                    "node": {k: v for k, v in node.items() if k != "content"},
                    "chain_valid": ok,
                    "chain_error": err or None,
                    "idempotent_replay": False,
                }

            node["artifact_path"] = str(artifact_path) if artifact_path else None
            node["artifact_sha256"] = artifact_sha256
            node["exec_journal_hash"] = exec_journal_hash

            # EXECUTING -> VERIFYING
            self._transition(
                graph,
                node,
                "VERIFYING",
                actor_role="adapter",
                actor_origin="system",
                grant=grant_ref,
                refs={"exec_journal_hash": exec_journal_hash, "artifact_sha256": artifact_sha256},
                evidence_class=evidence_class,
                reason="execution finished and disk artifact hashed; invoking separate verifier",
            )
            self._save_graph(graph)

            # Optional test hook between execute and verify (used to test disk tampering, REQ-VER-4)
            if pre_verify_hook is not None and artifact_path is not None:
                pre_verify_hook(artifact_path)

            # Separate OS process verifier (REQ-VER-4, G-8)
            ver_report, ver_exec_hash = run_separate_verifier(
                workdir=graph["workdir"],
                artifacts=(
                    [{"path": str(artifact_path), "expected_sha256": str(artifact_sha256)}]
                    if artifact_path and artifact_sha256
                    else []
                ),
                acceptance_criteria=list(node["acceptance_criteria"]),
                exit_code=observed_exit_code,
                exec_journal=self.exec_journal,
                artifacts_dir=self.artifacts_dir,
            )
            node["verification"] = ver_report

            ver_summary = {
                "integrity": ver_report.get("integrity", "FAIL"),
                "acceptance": ver_report.get("acceptance", "FAIL"),
                "verifier_version": ver_report.get("verifier_version", "unknown"),
                "code": ver_report.get("code", "FAIL"),
            }
            if ver_report.get("integrity") == "PASS" and ver_report.get("acceptance") == "PASS":
                self._transition(
                    graph,
                    node,
                    "VERIFIED",
                    actor_role="verifier",
                    actor_origin="system",
                    grant=grant_ref,
                    refs={
                        "exec_journal_hash": ver_exec_hash or exec_journal_hash,
                        "artifact_sha256": artifact_sha256,
                    },
                    verification=ver_summary,
                    evidence_class=evidence_class,
                    reason="separate verifier passed both integrity and acceptance",
                )
            else:
                self._transition(
                    graph,
                    node,
                    "FAILED",
                    actor_role="verifier",
                    actor_origin="system",
                    grant=grant_ref,
                    refs={
                        "exec_journal_hash": ver_exec_hash or exec_journal_hash,
                        "artifact_sha256": artifact_sha256,
                    },
                    verification=ver_summary,
                    evidence_class=evidence_class,
                    reason=f"verifier rejected ({ver_report.get('code')}): integrity={ver_report.get('integrity')}, acceptance={ver_report.get('acceptance')}",
                )
                self._cascade_stop_dependents(graph, node_id)

            if client_request_id:
                graph.setdefault("processed_requests", {})[client_request_id] = {
                    "action": "execute",
                    "node_id": node_id,
                    "final_state": node["state"],
                }
            self._save_graph(graph)
            ok, _, err = self.verify_chains()
            return {
                "graph": self._public_graph(graph),
                "node": {k: v for k, v in node.items() if k != "content"},
                "chain_valid": ok,
                "chain_error": err or None,
                "idempotent_replay": False,
            }

    def events(self, *, graph_id: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
        ok, count, err = self.verify_chains()
        rows = self.receipts.rows(graph_id=graph_id, limit=limit)
        return {
            "chain_valid": ok,
            "chain_error": err or None,
            "count": count,
            "tip_hash": self.receipts.tip_hash(),
            "rows": rows,
        }
