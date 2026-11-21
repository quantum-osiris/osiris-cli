"""Wave 3 Plan Artifact v1 builder, validator & measurable change-size gates (G-10..G-12).

Enforces:
  * ``REQ-PLAN-1`` / ``G-12``: Canonical ``plan_v1.json`` schema with tokenized ``argv: list[str]``
    and ``cwd: str`` on command steps; raw shell ``command`` strings and shell operators are refused.
  * ``REQ-PLAN-2`` / ``G-10``: Four named, measurable, independently fireable change-size gates
    (``max_files_touched``, ``max_lines_changed``, ``protected_paths_matched``,
    ``highest_tier_allowed``), replacing non-firing CRSM formulas.
  * ``REQ-PLAN-3`` / ``G-11``: Zero quarantined/refuted constants (``theta_lock``, ``51.843``,
    ``PHI_THRESHOLD``, ``0.7734``, ``F_max=0.97871``, ``Lambda_Phi=2.176435e-08``) and zero
    filesystem, network, or process side effects at module import time.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

from . import contracts_validator
from .exec_kernel import ExecJournal, canon
from .exec_policy import CommandRequest, PolicyError, classify
from .orchestration import governance_scope

PLAN_VERSION = "1.0"
RECEIPT_SCHEMA = "osiris.plan_gate_receipt/1"
PathArg = Optional[Union[os.PathLike, str]]

SHELL_META_TOKENS = frozenset({
    ";", "|", "||", "&&", "&", ">", ">>", "<", "<<", "<<<", "2>", "2>&1", "&>", ">|", "`",
})

TIER_ORDER: Dict[str, int] = {
    "READ": 1,
    "VERIFY": 2,
    "MUTATE": 3,
    "NETWORK": 4,
    "PRIVILEGED_ADMIN": 5,
    "FORBIDDEN": 5,
}

# Quarantined / refuted constants banned from planner rules, thresholds, and plan artifacts (REQ-PLAN-3).
_QUARANTINED_PLAN_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    ("theta_lock=51.843", re.compile(r"(?:\btheta_lock\b|(?<![\d.])51\.843\d*(?![\d]))", re.IGNORECASE)),
    ("PHI_THRESHOLD=0.7734", re.compile(r"(?:\bphi_threshold\b|(?<![\d.])0\.7734\d*(?![\d]))", re.IGNORECASE)),
    ("F_max=0.97871", re.compile(r"(?:\bf_max\b|(?<![\d.])0\.97871\d*(?![\d]))", re.IGNORECASE)),
    ("Lambda_Phi=2.176435e-08", re.compile(r"(?:\blambda_phi\b|2\.176435e-0?8)", re.IGNORECASE)),
)


class PlanValidationError(ValueError):
    """Raised when a Plan Artifact v1 violates schema, argv, confinement, or SYS-024 rules."""


def default_osiris_home(osiris_home: PathArg = None) -> Path:
    raw = osiris_home or os.environ.get("OSIRIS_HOME")
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".osiris").resolve()


def default_plan_receipts_path(osiris_home: PathArg = None) -> Path:
    if osiris_home is None:
        override = os.environ.get("OSIRIS_PLAN_RECEIPTS_LOG")
        if override:
            return Path(override).expanduser().resolve()
    return default_osiris_home(osiris_home) / "plans" / "plan_receipts.jsonl"


@dataclass(frozen=True)
class PlanGateThresholds:
    max_files_touched: int = 10
    max_lines_changed: int = 500
    protected_paths_matched: int = 0
    highest_tier_allowed: str = "MUTATE"

    def __post_init__(self) -> None:
        if isinstance(self.max_files_touched, bool) or self.max_files_touched < 0:
            raise PlanValidationError("max_files_touched must be an integer >= 0")
        if isinstance(self.max_lines_changed, bool) or self.max_lines_changed < 0:
            raise PlanValidationError("max_lines_changed must be an integer >= 0")
        if isinstance(self.protected_paths_matched, bool) or self.protected_paths_matched < 0:
            raise PlanValidationError("protected_paths_matched must be an integer >= 0")
        if self.highest_tier_allowed not in TIER_ORDER:
            raise PlanValidationError(
                f"highest_tier_allowed must be one of {sorted(TIER_ORDER)}, got {self.highest_tier_allowed!r}"
            )


@dataclass(frozen=True)
class PlanAuditResult:
    valid: bool
    gates_passed: bool
    errors: Tuple[str, ...]
    gates: Dict[str, Any]
    plan_sha256: str


def check_no_quarantined_constants(payload: Any, path: str = "plan") -> List[str]:
    """Recursively scan plan structures and reject quarantined CRSM/refuted constants (REQ-PLAN-3)."""
    issues: List[str] = []
    if isinstance(payload, dict):
        for k, v in payload.items():
            key_str = str(k)
            for label, rx in _QUARANTINED_PLAN_PATTERNS:
                if rx.search(key_str):
                    issues.append(f"{path}.{key_str}: quarantined constant {label} refused (REQ-PLAN-3)")
            issues.extend(check_no_quarantined_constants(v, f"{path}.{key_str}"))
    elif isinstance(payload, (list, tuple)):
        for idx, item in enumerate(payload):
            issues.extend(check_no_quarantined_constants(item, f"{path}[{idx}]"))
    elif isinstance(payload, str):
        for label, rx in _QUARANTINED_PLAN_PATTERNS:
            if rx.search(payload):
                issues.append(f"{path}: quarantined constant {label} refused (REQ-PLAN-3)")
    elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
        rendered = repr(payload)
        for label, rx in _QUARANTINED_PLAN_PATTERNS:
            if rx.search(rendered):
                issues.append(f"{path}: quarantined numeric constant {label} refused (REQ-PLAN-3)")
    return issues


def normalize_argv(argv_or_cmd: Any) -> List[str]:
    """Validate tokenized ``argv: list[str]``; refuse raw command strings & shell operators (G-12)."""
    if isinstance(argv_or_cmd, (str, bytes)):
        raise PlanValidationError("Raw command strings are refused (REQ-PLAN-1 / G-12); pass argv: list[str].")
    if not isinstance(argv_or_cmd, Sequence):
        raise PlanValidationError("Command step requires argv: list[str].")
    tokens: List[str] = []
    for raw_tok in argv_or_cmd:
        if not isinstance(raw_tok, str):
            raise PlanValidationError(f"argv token must be a string, got {type(raw_tok).__name__}")
        tok = raw_tok.strip()
        if not tok:
            raise PlanValidationError("argv tokens must be non-empty strings")
        if "\n" in raw_tok or "\r" in raw_tok or "\x00" in raw_tok:
            raise PlanValidationError("Control/newline characters refused in argv token")
        if tok in SHELL_META_TOKENS or "$(" in tok or "`" in tok or ">(" in tok or "<(" in tok:
            raise PlanValidationError(f"Shell operator or substitution refused in argv token: {tok!r}")
        tokens.append(tok)
    if not tokens:
        raise PlanValidationError("argv must contain at least 1 token")
    return tokens


def normalize_rel_path(raw_path: str) -> str:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise PlanValidationError("Plan path must be a non-empty string")
    cleaned = raw_path.strip().replace("\\", "/")
    if "\x00" in cleaned:
        raise PlanValidationError("NUL byte refused in plan path")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    norm = os.path.normpath(cleaned).replace("\\", "/")
    if norm.startswith("..") or os.path.isabs(cleaned):
        raise PlanValidationError(f"Plan path must be workspace-relative without '..': {raw_path!r}")
    return norm


def normalize_cwd(raw_cwd: Any, *, root: PathArg = None) -> str:
    if not isinstance(raw_cwd, str) or not raw_cwd.strip():
        raise PlanValidationError("Command step requires a non-empty cwd string (REQ-PLAN-1)")
    cleaned = raw_cwd.strip()
    if "\x00" in cleaned:
        raise PlanValidationError("NUL byte refused in cwd")
    if cleaned in (".", "./"):
        return "."
    if os.path.isabs(cleaned):
        if root is not None:
            real_root = os.path.realpath(str(root))
            real_cwd = os.path.realpath(cleaned)
            if not (real_cwd == real_root or real_cwd.startswith(real_root.rstrip(os.sep) + os.sep)):
                raise PlanValidationError(f"Absolute cwd {cleaned!r} escapes workspace root")
        return cleaned
    norm = os.path.normpath(cleaned).replace("\\", "/")
    if norm.startswith(".."):
        raise PlanValidationError(f"Relative cwd {raw_cwd!r} cannot traverse above workspace root")
    return norm


def is_protected_plan_path(rel_or_path: str, *, root: PathArg = None) -> bool:
    """True when ``rel_or_path`` matches any immutable governance file or directory (G-16 / REQ-SEC-6)."""
    cleaned = rel_or_path.strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    norm = os.path.normpath(cleaned).replace("\\", "/").lstrip("/")
    if norm.startswith(".."):
        return True
    if norm in governance_scope.IMMUTABLE_GOVERNANCE_FILES:
        return True
    if os.path.basename(norm).lower() in governance_scope.LEGACY_BASENAMES:
        return True
    for gdir in governance_scope.IMMUTABLE_GOVERNANCE_DIRS:
        if norm == gdir or norm.startswith(gdir + "/"):
            return True
    if norm.startswith("osiris/contracts/") or norm == "osiris/contracts":
        return True
    if root is not None:
        base = os.path.realpath(str(root))
        candidate = os.path.join(base, norm)
        if governance_scope.is_protected(candidate, roots=[base]):
            return True
    return False


def count_diff_changed_lines(diff_text: str) -> int:
    """Count added + removed lines in a unified diff (excluding ``+++``/``---``/``@@`` headers)."""
    changed = 0
    for line in (diff_text or "").splitlines():
        if line.startswith(("+++", "---", "@@")):
            continue
        if line.startswith(("+", "-")):
            changed += 1
    return changed


def classify_step_tier(step: Mapping[str, Any], *, root: PathArg = None) -> str:
    """Authoritative step tier from concrete step type & ``exec_policy.classify(argv)`` (G-7)."""
    if "command" in step:
        raise PlanValidationError("Step contains forbidden 'command' string property (G-12); use argv + cwd")

    stype = str(step.get("type") or "").strip()
    ws_root = os.path.realpath(str(root)) if root is not None else os.getcwd()

    if stype in ("run_command", "verify"):
        if "argv" not in step:
            if stype == "verify":
                return "VERIFY"
            raise PlanValidationError("run_command step requires tokenized 'argv' list")
        argv = normalize_argv(step["argv"])
        cwd_str = normalize_cwd(step.get("cwd", "."), root=ws_root)
        abs_cwd = cwd_str if os.path.isabs(cwd_str) else os.path.realpath(os.path.join(ws_root, cwd_str))
        try:
            cls = classify(CommandRequest(argv=tuple(argv), cwd=abs_cwd, origin="model", root=ws_root))
            tier = cls.tier
            if cls.decision == "DENY" and tier not in ("NETWORK", "FORBIDDEN"):
                prog = os.path.basename(argv[0])
                if prog in ("sudo", "su", "chmod", "chown", "dd", "mkfs", "mount"):
                    return "PRIVILEGED_ADMIN"
                return "FORBIDDEN"
            if tier == "FORBIDDEN":
                prog = os.path.basename(argv[0])
                if prog in ("sudo", "su", "chmod", "chown", "dd", "mkfs", "mount"):
                    return "PRIVILEGED_ADMIN"
            return tier
        except PolicyError:
            return "FORBIDDEN"

    if stype in ("edit_file", "create_file", "delete_file", "apply_patch"):
        return "MUTATE"
    raise PlanValidationError(f"Unsupported plan step type {stype!r}")


def evaluate_plan_gates(
    steps: Sequence[Mapping[str, Any]],
    thresholds: PlanGateThresholds = PlanGateThresholds(),
    *,
    root: PathArg = None,
) -> Dict[str, Any]:
    """Evaluate all 4 measurable change-size gates against ``steps`` (G-10 / REQ-PLAN-2)."""
    if not steps:
        raise PlanValidationError("Plan requires at least 1 step")

    touched_files: Set[str] = set()
    protected_hits: Set[str] = set()
    total_lines_changed = 0
    highest_tier = "READ"

    for idx, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise PlanValidationError(f"steps[{idx}] must be an object")
        if "command" in step:
            raise PlanValidationError(f"steps[{idx}]: raw 'command' property refused (G-12)")

        stype = str(step.get("type") or "")
        if stype in ("run_command", "verify") and "argv" in step:
            normalize_argv(step["argv"])
            normalize_cwd(step.get("cwd"), root=root)

        raw_single = step.get("path")
        if isinstance(raw_single, str) and raw_single.strip():
            cleaned = raw_single.strip().replace("\\", "/")
            if is_protected_plan_path(cleaned, root=root):
                protected_hits.add(cleaned)
            norm = normalize_rel_path(cleaned)
            touched_files.add(norm)

        raw_multi = step.get("paths")
        if isinstance(raw_multi, Sequence) and not isinstance(raw_multi, (str, bytes)):
            for p in raw_multi:
                if isinstance(p, str) and p.strip():
                    cleaned = p.strip().replace("\\", "/")
                    if is_protected_plan_path(cleaned, root=root):
                        protected_hits.add(cleaned)
                    norm = normalize_rel_path(cleaned)
                    touched_files.add(norm)

        lc = step.get("lines_changed")
        if lc is not None:
            if isinstance(lc, bool) or not isinstance(lc, int) or lc < 0:
                raise PlanValidationError(f"steps[{idx}].lines_changed must be an integer >= 0")
            total_lines_changed += lc
        elif isinstance(step.get("diff"), str) and step["diff"]:
            total_lines_changed += count_diff_changed_lines(step["diff"])

        tier = classify_step_tier(step, root=root)
        if TIER_ORDER[tier] > TIER_ORDER[highest_tier]:
            highest_tier = tier

    files_count = len(touched_files)
    prot_count = len(protected_hits)
    files_ok = files_count <= thresholds.max_files_touched
    lines_ok = total_lines_changed <= thresholds.max_lines_changed
    prot_ok = prot_count <= thresholds.protected_paths_matched
    tier_ok = TIER_ORDER[highest_tier] <= TIER_ORDER[thresholds.highest_tier_allowed]

    checks = [
        {
            "name": "max_files_touched",
            "measured": files_count,
            "threshold": thresholds.max_files_touched,
            "passed": files_ok,
            "reason": (
                f"{files_count} file(s) <= cap {thresholds.max_files_touched}"
                if files_ok
                else f"Touched {files_count} files, exceeding cap {thresholds.max_files_touched}"
            ),
        },
        {
            "name": "max_lines_changed",
            "measured": total_lines_changed,
            "threshold": thresholds.max_lines_changed,
            "passed": lines_ok,
            "reason": (
                f"{total_lines_changed} line(s) <= cap {thresholds.max_lines_changed}"
                if lines_ok
                else f"Changed {total_lines_changed} lines, exceeding cap {thresholds.max_lines_changed}"
            ),
        },
        {
            "name": "protected_paths_matched",
            "measured": prot_count,
            "threshold": thresholds.protected_paths_matched,
            "passed": prot_ok,
            "reason": (
                "0 protected governance paths targeted"
                if prot_ok and prot_count == 0
                else (
                    f"{prot_count} protected path(s) <= cap {thresholds.protected_paths_matched}"
                    if prot_ok
                    else f"Targeted protected governance path(s): {', '.join(sorted(protected_hits))}"
                )
            ),
        },
        {
            "name": "highest_tier_allowed",
            "measured": highest_tier,
            "threshold": thresholds.highest_tier_allowed,
            "passed": tier_ok,
            "reason": (
                f"Highest tier {highest_tier} <= allowed {thresholds.highest_tier_allowed}"
                if tier_ok
                else f"Step tier {highest_tier} exceeds allowed ceiling {thresholds.highest_tier_allowed}"
            ),
        },
    ]
    return {
        "passed": bool(files_ok and lines_ok and prot_ok and tier_ok),
        "checks": checks,
    }


def build_plan_v1(
    *,
    intent: str,
    steps: Sequence[Mapping[str, Any]],
    domain: str = "code",
    execution_level: str = "modify",
    execution_mode: str = "dry_run",
    confirmed: bool = False,
    preconditions: Sequence[str] = (),
    postconditions: Sequence[str] = (),
    thresholds: PlanGateThresholds = PlanGateThresholds(),
    plan_id: Optional[str] = None,
    intent_id: Optional[str] = None,
    annotations: Optional[Dict[str, Any]] = None,
    root: PathArg = None,
) -> Dict[str, Any]:
    """Construct a canonical ``plan_v1.json`` artifact and evaluate all 4 measurable gates."""
    gates = evaluate_plan_gates(steps, thresholds=thresholds, root=root)
    normalized_steps: List[Dict[str, Any]] = []
    for raw_step in steps:
        st = dict(raw_step)
        if st.get("type") in ("run_command", "verify") and "argv" in st:
            st["argv"] = normalize_argv(st["argv"])
            st["cwd"] = normalize_cwd(st.get("cwd", "."), root=root)
        normalized_steps.append(st)

    planner_digest = hashlib.sha256(
        json.dumps(normalized_steps, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    plan: Dict[str, Any] = {
        "plan_version": PLAN_VERSION,
        "execution_mode": execution_mode,
        "meta": {
            "plan_id": plan_id or f"plan-{uuid.uuid4().hex[:12]}",
            "intent_id": intent_id or f"intent-{hashlib.sha256(intent.encode('utf-8')).hexdigest()[:12]}",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "planner_hash": planner_digest,
            "intent": str(intent)[:4000],
        },
        "authority": {
            "domain": domain,
            "execution_level": execution_level,
            "confirmed": bool(confirmed),
        },
        "constraints": {
            "write_allowed": execution_level in ("modify", "deploy"),
            "network_allowed": False,
        },
        "steps": normalized_steps,
        "validation": {
            "preconditions": [str(x) for x in preconditions],
            "postconditions": [str(x) for x in postconditions],
        },
        "gates": gates,
    }
    if annotations is not None:
        plan["annotations"] = dict(annotations)

    quarantined_issues = check_no_quarantined_constants(plan)
    if quarantined_issues:
        raise PlanValidationError("; ".join(quarantined_issues))

    ok, refusal = contracts_validator.validate_contract(
        "plan_v1.json", plan, require_execution_mode_on_plan=True
    )
    if not ok:
        raise PlanValidationError(f"Generated plan failed plan_v1.json schema: {refusal}")
    return plan


def validate_plan_v1(
    plan: Any,
    *,
    thresholds: PlanGateThresholds = PlanGateThresholds(),
    root: PathArg = None,
) -> PlanAuditResult:
    """Validate an arbitrary dict against ``plan_v1.json``, SYS-024 constant ban, and 4 gates."""
    errors: List[str] = []
    if not isinstance(plan, dict):
        return PlanAuditResult(False, False, ("plan must be a JSON object",), {"passed": False, "checks": []}, "")

    errors.extend(check_no_quarantined_constants(plan))
    ok_schema, refusal = contracts_validator.validate_contract("plan_v1.json", plan)
    if not ok_schema and refusal:
        for v in (refusal.get("error", {}).get("violating_fields") or ()):
            errors.append(f"{v.get('field')}: {v.get('issue')}")

    gates: Dict[str, Any] = {"passed": False, "checks": []}
    steps = plan.get("steps")
    if isinstance(steps, Sequence) and not isinstance(steps, (str, bytes)):
        try:
            gates = evaluate_plan_gates(steps, thresholds=thresholds, root=root)
        except PlanValidationError as exc:
            errors.append(str(exc))
    else:
        errors.append("steps must be a non-empty array")

    # Also verify embedded gates block if caller supplied one that claims passed=True falsely
    supplied_gates = plan.get("gates")
    if isinstance(supplied_gates, dict) and supplied_gates.get("passed") is True and not gates["passed"]:
        errors.append("plan.gates.passed is True, but re-evaluated change-size gates failed")

    canonical_bytes = json.dumps(plan, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    plan_sha = hashlib.sha256(canonical_bytes).hexdigest()
    valid = len(errors) == 0
    return PlanAuditResult(
        valid=valid,
        gates_passed=bool(valid and gates.get("passed")),
        errors=tuple(errors),
        gates=gates,
        plan_sha256=plan_sha,
    )


def record_plan_receipt(
    plan: Mapping[str, Any],
    *,
    thresholds: PlanGateThresholds = PlanGateThresholds(),
    osiris_home: PathArg = None,
    root: PathArg = None,
) -> Tuple[PlanAuditResult, str]:
    """Audit ``plan`` and append an ``exec-canon-1`` receipt under ``$OSIRIS_HOME/plans/``."""
    audit = validate_plan_v1(dict(plan), thresholds=thresholds, root=root)
    path = default_plan_receipts_path(osiris_home)
    journal = ExecJournal(path)
    meta = plan.get("meta") if isinstance(plan.get("meta"), Mapping) else {}
    body = {
        "schema": RECEIPT_SCHEMA,
        "plan_id": str(meta.get("plan_id") or ""),
        "plan_sha256": audit.plan_sha256,
        "valid": audit.valid,
        "gates_passed": audit.gates_passed,
        "errors": list(audit.errors),
        "gates": audit.gates,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    canon(body)
    digest = journal.append(body)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return audit, digest


def default_pytest_verify_step(target_paths: Sequence[str] = (), cwd: str = ".") -> Dict[str, Any]:
    """Build a canonical ``verify`` step invoking ``pytest`` via tokenized ``argv``."""
    argv = [sys.executable, "-m", "pytest", "-q"]
    argv.extend(str(p) for p in target_paths if str(p).strip())
    return {
        "id": "step-verify-pytest",
        "type": "verify",
        "description": "Run hermetic pytest verification via tokenized argv",
        "argv": argv,
        "cwd": normalize_cwd(cwd),
    }
