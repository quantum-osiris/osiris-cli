"""RFC-0001 Phase 0: an OSIRIS-side adapter from agent-shaped key-value genes to ``AgentSpec``.

No grammar, parser or compiler change. dna::}{::lang (dnalang-core) already parses
``gene NAME { kind: "agent" ... }`` and preserves unknown fields, but it neither validates nor
hashes them (docs/rfc/RFC-0001-agent-gene.md, finding 3). So validation and identity live here:

  * closed schema: unknown keys, wrong types, duplicate keys/ids, out-of-range budgets -> refused;
  * explicit capability allowlist (mirror of the governor's table, see CAPABILITY_TIERS) and a
    per-role ceiling; WORKSPACE_WRITE is not expressible in a gene at all (needs a human grant);
  * canonical JSON + sha256 over EVERY authorization-relevant field, the schema version and the
    capability-policy digest, so changing any of them changes the digest;
  * an ``AgentSpec`` is produced only from fields that passed validation. Source text is never
    evaluated; nothing here touches disk, the network or ~/.osiris (except reading the .dna text
    the caller passes in).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .agent_spec import AgentSpec

SCHEMA = "osiris.agent-gene/1"

# Mirror of CapabilityGovernor.CAPABILITY_TIERS (osiris/agents/gateway/governor.py:32 in the home
# tree; that module is not part of this repository). tier >= HUMAN_APPROVAL_FLOOR needs a human grant.
CAPABILITY_TIERS: Dict[str, int] = {
    "READ_ONLY_ANALYSIS": 0, "RETURN_STRUCTURED_PROPOSAL": 0,
    "REPLAY_FIXTURE": 1, "SESSION_STATE_WRITE": 1,
    "WORKSPACE_WRITE": 3, "SUBPROCESS_EXEC": 4,
}
HUMAN_APPROVAL_FLOOR = 3

# What a gene of each role may ever hold. WORKSPACE_WRITE appears nowhere: not grantable via a gene.
ROLE_CEILING: Dict[str, frozenset] = {
    "research": frozenset({"READ_ONLY_ANALYSIS", "RETURN_STRUCTURED_PROPOSAL"}),
    "developer": frozenset({"READ_ONLY_ANALYSIS", "RETURN_STRUCTURED_PROPOSAL"}),
    "critic": frozenset({"READ_ONLY_ANALYSIS", "RETURN_STRUCTURED_PROPOSAL", "REPLAY_FIXTURE",
                         "SUBPROCESS_EXEC"}),
}
PROVIDERS = frozenset({"none", "ollama", "gemma", "gemini", "local_nclm"})
LIMITS = {"max_tokens": (1, 32768), "deadline_s": (1, 3600)}

_KEYS = {"kind", "role", "version", "capabilities", "tool_permissions", "provider", "model",
         "max_tokens", "deadline_s", "evidence_required", "input_schema", "output_schema", "id"}
_REQUIRED = {"kind", "role", "version", "capabilities"}
_VERSION = re.compile(r"^\d{1,4}\.\d{1,4}(\.\d{1,4})?$")
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,62}$")
_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,47}$")
_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,79}$")


class AgentGeneError(ValueError):
    def __init__(self, problems: List[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


def policy_digest() -> str:
    return hashlib.sha256(json.dumps({"tiers": CAPABILITY_TIERS, "floor": HUMAN_APPROVAL_FLOOR,
                                      "ceilings": {k: sorted(v) for k, v in ROLE_CEILING.items()},
                                      "providers": sorted(PROVIDERS), "limits": LIMITS},
                                     sort_keys=True).encode()).hexdigest()


def _num(v: Any) -> Optional[float]:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (isinstance(v, float) and not math.isfinite(v)):
        return None
    return v


def _str_list(v: Any, what: str, pat: Optional["re.Pattern[str]"], problems: List[str],
              max_len: int = 16) -> List[str]:
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        problems.append(f"{what} must be a list of strings")
        return []
    if len(v) > max_len:
        problems.append(f"{what} has more than {max_len} entries")
    if len(set(v)) != len(v):
        problems.append(f"{what} has duplicate entries")
    if pat is not None:
        for x in v:
            if not pat.match(x):
                problems.append(f"{what} entry {x!r} is malformed")
    return list(v)


@dataclass(frozen=True)
class ValidatedAgentGene:
    gene_name: str
    spec: AgentSpec
    requires_human_approval: bool
    canonical_json: str
    digest: str


def canonical_fields(spec: AgentSpec) -> Dict[str, Any]:
    """Every field that bears on what this agent may do, normalized."""
    def norm(x: Any) -> Any:
        return int(x) if isinstance(x, float) and x == int(x) else x
    return {
        "schema": SCHEMA, "policy": policy_digest(),
        "agent_id": spec.agent_id, "version": spec.version, "role": spec.role,
        "capabilities": sorted(spec.capabilities), "tool_permissions": sorted(spec.tool_permissions),
        "provider": spec.provider, "model": spec.model,
        "input_schema": spec.input_schema, "output_schema": spec.output_schema,
        "max_tokens": norm(spec.max_tokens), "deadline_s": norm(spec.deadline_s),
        "evidence_required": spec.evidence_required,
        "requires_human_approval": any(CAPABILITY_TIERS.get(c, 99) >= HUMAN_APPROVAL_FLOOR
                                       for c in spec.capabilities),
    }


def spec_digest(spec: AgentSpec) -> Tuple[str, str]:
    """(canonical_json, sha256). Works on any AgentSpec, so hand-built and gene-built specs compare."""
    blob = json.dumps(canonical_fields(spec), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)
    return blob, hashlib.sha256(blob.encode("utf-8")).hexdigest()


def validate_agent_gene(name: str, fields: Mapping[str, Any]) -> ValidatedAgentGene:
    problems: List[str] = []
    if not isinstance(name, str) or not _NAME.match(name):
        problems.append(f"gene name {name!r} is malformed")
    if not isinstance(fields, Mapping):
        raise AgentGeneError(["fields must be a mapping"])
    unknown = sorted(set(fields) - _KEYS)
    if unknown:
        problems.append(f"unknown field(s): {unknown}")
    missing = sorted(_REQUIRED - set(fields))
    if missing:
        problems.append(f"missing required field(s): {missing}")
    if fields.get("kind") != "agent":
        problems.append('kind must be exactly "agent"')

    role = fields.get("role")
    if role not in ROLE_CEILING:
        problems.append(f"role must be one of {sorted(ROLE_CEILING)}")
    version = fields.get("version")
    if not isinstance(version, str) or not _VERSION.match(version):
        problems.append("version must be a string like '1.0' or '1.2.3'")

    caps = _str_list(fields.get("capabilities"), "capabilities", None, problems, max_len=len(CAPABILITY_TIERS))
    if not caps:
        problems.append("capabilities must not be empty")
    for c in caps:
        if c not in CAPABILITY_TIERS:
            problems.append(f"unknown capability {c!r} (rejected: not in the allowlist)")
        elif role in ROLE_CEILING and c not in ROLE_CEILING[role]:
            why = ("needs a human grant and cannot be declared in a gene" if CAPABILITY_TIERS[c] >= HUMAN_APPROVAL_FLOOR
                   and c not in ROLE_CEILING.get(role, ()) else "exceeds the ceiling for this role")
            problems.append(f"capability {c} {why} (role {role})")

    tools = _str_list(fields.get("tool_permissions", []), "tool_permissions", _TOKEN, problems)
    provider = fields.get("provider", "none")
    if provider not in PROVIDERS:
        problems.append(f"provider must be one of {sorted(PROVIDERS)}")
    model = fields.get("model", "")
    if not isinstance(model, str) or (model and not _MODEL.match(model)):
        problems.append("model is malformed")

    budgets: Dict[str, Any] = {}
    for key, default in (("max_tokens", 4096), ("deadline_s", 120)):
        v = fields.get(key, default)
        lo, hi = LIMITS[key]
        n = _num(v)
        if n is None or not lo <= n <= hi:
            problems.append(f"{key} must be a number in [{lo}, {hi}]")
        elif key == "max_tokens" and n != int(n):
            problems.append("max_tokens must be a whole number")
        else:
            budgets[key] = int(n) if key == "max_tokens" else n
    evidence = fields.get("evidence_required", True)
    if not isinstance(evidence, bool):
        problems.append("evidence_required must be a boolean (true/false), not text or a number")
    for key in ("input_schema", "output_schema"):
        v = fields.get(key, "")
        if not isinstance(v, str) or (v and not _NAME.match(v)):
            problems.append(f"{key} is malformed")
    agent_id = fields.get("id", name)
    if not isinstance(agent_id, str) or not _NAME.match(agent_id):
        problems.append("id is malformed")

    if problems:
        raise AgentGeneError(problems)
    spec = AgentSpec(
        agent_id=agent_id, version=version, role=role, capabilities=frozenset(caps),
        tool_permissions=frozenset(tools), provider=provider, model=model,
        input_schema=fields.get("input_schema", ""), output_schema=fields.get("output_schema", ""),
        max_tokens=budgets["max_tokens"], deadline_s=float(budgets["deadline_s"]), evidence_required=evidence)
    blob, dg = spec_digest(spec)
    return ValidatedAgentGene(name, spec, json.loads(blob)["requires_human_approval"], blob, dg)


def agent_genes_from_dna(text: str) -> List[ValidatedAgentGene]:
    """Parse dnalang-core (dialect B) text and validate every ``kind: "agent"`` gene.
    Non-agent genes are ignored. Duplicate keys inside a gene (which the language resolves
    silently, last wins) and duplicate gene ids are errors. Raises ImportError if dnalang-core
    is unavailable; the source is parsed, never executed."""
    from dnalang import parser as dp   # dnalang-core, optional

    class _Strict(dp.Parser):
        dups: List[str] = []

        def kvgene(self):          # same shape as dnalang.parser.Parser.kvgene, plus a duplicate-key check
            pos = self.cur.pos
            name = self.expect("IDENT").text
            self.expect("SYM", "{")
            fields: Dict[str, Any] = {}
            while not self.accept("SYM", "}"):
                key = self.key()
                self.expect("SYM", ":")
                value = self.value()
                if key in fields:
                    self.dups.append(f"{name}.{key}")
                fields[key] = value
                self.accept("SYM", ",")
            return dp.A.KVGene(name, fields, pos)

    p = _Strict(dp.tokenize(text))
    p.dups = []
    ast = p.organism()
    if p.dups:
        raise AgentGeneError([f"duplicate key {d} (the language would silently keep the last)" for d in p.dups])
    out, seen = [], set()
    problems: List[str] = []
    for g in list(ast.kv_genes):
        if g.fields.get("kind") != "agent":
            continue
        if g.name in seen:
            problems.append(f"duplicate agent gene {g.name}")
        seen.add(g.name)
        try:
            out.append(validate_agent_gene(g.name, g.fields))
        except AgentGeneError as e:
            problems += [f"{g.name}: {x}" for x in e.problems]
    if problems:
        raise AgentGeneError(problems)
    return out
