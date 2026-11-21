"""L4 Propose: multi-candidate proposal pipeline (baseline, alternative, conservative).

Generates and evaluates 3 distinct candidate strategies side by side before ``/apply``:

  * ``baseline``      direct, idiomatic implementation satisfying the specification
  * ``alternative``   clean structural/algorithmic alternative (data-driven / decomposed)
  * ``conservative``  minimal-diff surgical change preserving existing structure + guards

Every candidate passes through the identical 6-stage deterministic verification pipeline
(``REQ-VER-1..4``, ``SYS-025``):
  1. Workspace confinement & immutable governance scope guard (``G-16`` / ``REQ-SEC-6``)
  2. Python syntax & existing public contract preservation (``production_synth``)
  3. Zero-stub placeholder gate (``payload_gate``)
  4. Unified deny-by-default AST firewall on module & test (``ast_firewall.check``)
  5. Non-trivial test-shape check (imports target, real non-constant assertions)
  6. Isolated temp-workspace ``pytest`` execution via ``exec_kernel.run`` (``VERIFY`` tier)
     plus Plan Artifact v1 measurable change-size gates (``plan_gates``).

Renders a compact side-by-side comparison table strictly ``< 100`` columns wide (clamped
cleanly down to 74 cols for Node Alpha Termux viewports) and records an ``exec-canon-1``
comparison receipt under ``$OSIRIS_HOME/proposals/multi_candidate.jsonl``.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import os
import re
import shlex
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import payload_gate
from . import exec_kernel, plan_gates, production_synth, quarantine
from .exec_policy import CommandRequest
from .orchestration import ast_firewall, governance_scope
from .orchestration.agents import check_test_shape

STRATEGIES: Tuple[str, str, str] = ("baseline", "alternative", "conservative")
RECEIPT_SCHEMA = "osiris.multi_candidate/1"
PathArg = Optional[Union[os.PathLike, str]]

STRATEGY_DIRECTIVES: Dict[str, str] = {
    "baseline": (
        "STRATEGY: BASELINE — Write the clearest, most direct idiomatic Python implementation "
        "that satisfies every specification requirement and preserves public contracts."
    ),
    "alternative": (
        "STRATEGY: ALTERNATIVE — Implement a distinct structural design (e.g., pure helper "
        "decomposition, explicit state/table dispatch, or functional pipeline) optimizing "
        "modularity and edge-case testability while preserving public contracts."
    ),
    "conservative": (
        "STRATEGY: CONSERVATIVE — Produce the smallest, lowest-risk surgical diff touching the "
        "minimum lines necessary, preserving existing control flow and adding strict defensive "
        "input validation."
    ),
}

_PYTEST_COUNTS_RE = re.compile(r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed)")


def default_osiris_home(osiris_home: PathArg = None) -> Path:
    raw = osiris_home or os.environ.get("OSIRIS_HOME")
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".osiris").resolve()


def default_multi_candidate_log(osiris_home: PathArg = None) -> Path:
    if osiris_home is None:
        override = os.environ.get("OSIRIS_MULTI_CANDIDATE_LOG")
        if override:
            return Path(override).expanduser().resolve()
    return default_osiris_home(osiris_home) / "proposals" / "multi_candidate.jsonl"


@dataclass(frozen=True)
class CandidateEvaluation:
    strategy: str
    target_rel_path: str
    module_source: str
    test_source: str
    verified: bool
    stage_failed: str  # "" when verified
    reason: str
    ast_clean: bool
    stub_clean: bool
    contract_clean: bool
    test_shape_clean: bool
    plan_gates_passed: bool
    plan_gates_passed_count: int
    tests_passed: int
    tests_failed: int
    tests_skipped: int
    lines_added: int
    lines_removed: int
    lines_changed: int
    diff_text: str
    module_sha256: str
    test_sha256: str
    exec_receipt_id: str
    plan_v1: Dict[str, Any]

    def to_receipt_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "target_rel_path": self.target_rel_path,
            "verified": self.verified,
            "stage_failed": self.stage_failed,
            "reason": self.reason[:240],
            "ast_clean": self.ast_clean,
            "stub_clean": self.stub_clean,
            "contract_clean": self.contract_clean,
            "test_shape_clean": self.test_shape_clean,
            "plan_gates_passed": self.plan_gates_passed,
            "plan_gates_passed_count": self.plan_gates_passed_count,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "tests_skipped": self.tests_skipped,
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed,
            "lines_changed": self.lines_changed,
            "module_sha256": self.module_sha256,
            "test_sha256": self.test_sha256,
            "exec_receipt_id": self.exec_receipt_id,
            "plan_id": str(self.plan_v1.get("meta", {}).get("plan_id", "")),
        }


@dataclass(frozen=True)
class MultiCandidateBundle:
    bundle_id: str
    target_rel_path: str
    goal: str
    candidates: Tuple[CandidateEvaluation, ...]
    selected_strategy: Optional[str]
    selection_reason: str
    receipt_hash: str = ""

    def by_strategy(self, strategy: str) -> Optional[CandidateEvaluation]:
        key = strategy.strip().lower()
        for c in self.candidates:
            if c.strategy == key:
                return c
        return None

    @property
    def selected(self) -> Optional[CandidateEvaluation]:
        if not self.selected_strategy:
            return None
        return self.by_strategy(self.selected_strategy)


def _compute_diff_stats(existing_source: Optional[str], new_source: str, rel_path: str) -> Tuple[int, int, str]:
    old_lines = (existing_source or "").splitlines(keepends=True)
    new_lines = (new_source or "").splitlines(keepends=True)
    diff_lines = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
        )
    )
    added = sum(1 for ln in diff_lines if ln.startswith("+") and not ln.startswith("+++"))
    removed = sum(1 for ln in diff_lines if ln.startswith("-") and not ln.startswith("---"))
    return added, removed, "".join(diff_lines)


def _parse_pytest_summary(output: str) -> Tuple[int, int, int]:
    passed = failed = skipped = 0
    summary_lines = [ln for ln in (output or "").splitlines() if "==" in ln or "passed" in ln or "failed" in ln]
    target_line = summary_lines[-1] if summary_lines else (output or "")
    for cnt_s, kind in _PYTEST_COUNTS_RE.findall(target_line):
        n = int(cnt_s)
        if kind == "passed":
            passed += n
        elif kind in ("failed", "error", "errors"):
            failed += n
        elif kind == "skipped":
            skipped += n
    return passed, failed, skipped


def _stage_workspace_copy(root: Path, target_rel: str) -> Path:
    """Create a lightweight isolated workspace copy so candidate tests cannot mutate the repo."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="osiris-mc-verify-"))
    skip_Top = {".git", ".venv", "venv", "node_modules", ".pytest_cache", "__pycache__", ".next", "results"}
    # Copy top-level .py files and package directories needed for imports without huge caches.
    for entry in root.iterdir():
        if entry.name in skip_Top or entry.name.startswith(".osiris"):
            continue
        dest = tmp_dir / entry.name
        if entry.is_symlink():
            continue
        if entry.is_file() and entry.suffix in (".py", ".toml", ".cfg", ".ini", ".json", ".yaml", ".yml"):
            shutil.copy2(entry, dest)
        elif entry.is_dir() and entry.name in ("osiris_cli", "osiris", "contracts"):
            shutil.copytree(
                entry,
                dest,
                symlinks=False,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"),
            )
    # Ensure parent directory of target_rel exists inside tmp_dir
    target_parent = (tmp_dir / target_rel).parent
    target_parent.mkdir(parents=True, exist_ok=True)
    # Copy target file's sibling modules if target lives in a subdirectory not copied above
    src_parent = (root / target_rel).parent
    if src_parent.is_dir() and src_parent != root and not (tmp_dir / target_rel).parent.samefile(target_parent if target_parent.exists() else tmp_dir):
        pass
    if src_parent.is_dir() and src_parent != root:
        rel_parent = src_parent.relative_to(root)
        dst_parent = tmp_dir / rel_parent
        if not dst_parent.exists():
            shutil.copytree(
                src_parent,
                dst_parent,
                symlinks=False,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
    return tmp_dir


def evaluate_strategy_candidate(
    *,
    strategy: str,
    root: PathArg,
    target_rel_path: str,
    goal: str,
    module_source: str,
    test_source: str,
    existing_source: Optional[str] = None,
    thresholds: plan_gates.PlanGateThresholds = plan_gates.PlanGateThresholds(),
    exec_journal: Optional[exec_kernel.ExecJournal] = None,
    exec_artifacts_dir: PathArg = None,
    timeout_s: float = 60.0,
) -> CandidateEvaluation:
    """Run all 6 deterministic verification gates + Plan v1 gates on one candidate."""
    ws_root = Path(root or os.getcwd()).expanduser().resolve()
    clean_strategy = str(strategy).strip().lower()
    mod_src = (module_source or "").replace("\r\n", "\n").replace("\r", "\n")
    tst_src = (test_source or "").replace("\r\n", "\n").replace("\r", "\n")
    mod_sha = hashlib.sha256(mod_src.encode("utf-8")).hexdigest()
    tst_sha = hashlib.sha256(tst_src.encode("utf-8")).hexdigest()

    try:
        rel_norm = plan_gates.normalize_rel_path(target_rel_path)
    except plan_gates.PlanValidationError as exc:
        rel_norm = str(target_rel_path).strip()
        empty_plan: Dict[str, Any] = {"meta": {"plan_id": ""}, "gates": {"passed": False, "checks": []}}
        return CandidateEvaluation(
            strategy=clean_strategy, target_rel_path=rel_norm, module_source=mod_src, test_source=tst_src,
            verified=False, stage_failed="confinement", reason=str(exc),
            ast_clean=False, stub_clean=False, contract_clean=False, test_shape_clean=False,
            plan_gates_passed=False, plan_gates_passed_count=0,
            tests_passed=0, tests_failed=0, tests_skipped=0,
            lines_added=0, lines_removed=0, lines_changed=0, diff_text="",
            module_sha256=mod_sha, test_sha256=tst_sha, exec_receipt_id="", plan_v1=empty_plan,
        )

    if existing_source is None and (ws_root / rel_norm).is_file():
        existing_source = (ws_root / rel_norm).read_text(encoding="utf-8", errors="replace")

    added, removed, diff_txt = _compute_diff_stats(existing_source, mod_src, rel_norm)
    lines_changed = added + removed

    step_kind = "edit_file" if existing_source is not None else "create_file"
    steps = [
        {
            "id": f"step-1-{clean_strategy}",
            "type": step_kind,
            "description": f"{clean_strategy} candidate update to {rel_norm}",
            "path": rel_norm,
            "lines_changed": lines_changed,
        },
        plan_gates.default_pytest_verify_step(["test_candidate_eval.py"], cwd="."),
    ]
    try:
        plan_doc = plan_gates.build_plan_v1(
            intent=f"[{clean_strategy}] {goal}",
            steps=steps,
            execution_level="modify",
            execution_mode="dry_run",
            confirmed=False,
            preconditions=[f"workspace confined to {ws_root.name}"],
            postconditions=["AST firewall clean", "zero stubs", "sandboxed pytest green"],
            thresholds=thresholds,
            root=ws_root,
        )
    except plan_gates.PlanValidationError as exc:
        empty_plan = {"meta": {"plan_id": ""}, "gates": {"passed": False, "checks": []}}
        return CandidateEvaluation(
            strategy=clean_strategy, target_rel_path=rel_norm, module_source=mod_src, test_source=tst_src,
            verified=False, stage_failed="plan_gates", reason=str(exc),
            ast_clean=False, stub_clean=False, contract_clean=False, test_shape_clean=False,
            plan_gates_passed=False, plan_gates_passed_count=0,
            tests_passed=0, tests_failed=0, tests_skipped=0,
            lines_added=added, lines_removed=removed, lines_changed=lines_changed, diff_text=diff_txt,
            module_sha256=mod_sha, test_sha256=tst_sha, exec_receipt_id="", plan_v1=empty_plan,
        )

    checks_list = plan_doc["gates"]["checks"]
    gates_passed_cnt = sum(1 for c in checks_list if c.get("passed"))
    plan_ok = bool(plan_doc["gates"]["passed"])

    # Stage 1: Confinement & governance scope guard
    prot_reason = governance_scope.protected_reason(str(ws_root / rel_norm), roots=[str(ws_root)])
    if prot_reason or plan_gates.is_protected_plan_path(rel_norm, root=ws_root):
        return CandidateEvaluation(
            strategy=clean_strategy, target_rel_path=rel_norm, module_source=mod_src, test_source=tst_src,
            verified=False, stage_failed="confinement",
            reason=prot_reason or f"refusing protected governance target {rel_norm}",
            ast_clean=False, stub_clean=False, contract_clean=False, test_shape_clean=False,
            plan_gates_passed=plan_ok, plan_gates_passed_count=gates_passed_cnt,
            tests_passed=0, tests_failed=0, tests_skipped=0,
            lines_added=added, lines_removed=removed, lines_changed=lines_changed, diff_text=diff_txt,
            module_sha256=mod_sha, test_sha256=tst_sha, exec_receipt_id="", plan_v1=plan_doc,
        )

    if not mod_src.strip():
        return CandidateEvaluation(
            strategy=clean_strategy, target_rel_path=rel_norm, module_source=mod_src, test_source=tst_src,
            verified=False, stage_failed="syntax", reason="empty candidate module source",
            ast_clean=False, stub_clean=False, contract_clean=False, test_shape_clean=False,
            plan_gates_passed=plan_ok, plan_gates_passed_count=gates_passed_cnt,
            tests_passed=0, tests_failed=0, tests_skipped=0,
            lines_added=added, lines_removed=removed, lines_changed=lines_changed, diff_text=diff_txt,
            module_sha256=mod_sha, test_sha256=tst_sha, exec_receipt_id="", plan_v1=plan_doc,
        )

    # Stage 2: Syntax & public contract preservation
    try:
        ast.parse(mod_src, filename=rel_norm)
    except SyntaxError as exc:
        return CandidateEvaluation(
            strategy=clean_strategy, target_rel_path=rel_norm, module_source=mod_src, test_source=tst_src,
            verified=False, stage_failed="syntax", reason=f"SyntaxError: {exc.msg} (line {exc.lineno})",
            ast_clean=False, stub_clean=False, contract_clean=False, test_shape_clean=False,
            plan_gates_passed=plan_ok, plan_gates_passed_count=gates_passed_cnt,
            tests_passed=0, tests_failed=0, tests_skipped=0,
            lines_added=added, lines_removed=removed, lines_changed=lines_changed, diff_text=diff_txt,
            module_sha256=mod_sha, test_sha256=tst_sha, exec_receipt_id="", plan_v1=plan_doc,
        )

    contract_clean = True
    if existing_source:
        try:
            contract = production_synth.extract_module_contract(existing_source)
            broken = production_synth.verify_contract_preservation(contract, mod_src)
            if broken:
                contract_clean = False
                return CandidateEvaluation(
                    strategy=clean_strategy, target_rel_path=rel_norm, module_source=mod_src, test_source=tst_src,
                    verified=False, stage_failed="contract", reason="; ".join(broken[:3]),
                    ast_clean=False, stub_clean=False, contract_clean=False, test_shape_clean=False,
                    plan_gates_passed=plan_ok, plan_gates_passed_count=gates_passed_cnt,
                    tests_passed=0, tests_failed=0, tests_skipped=0,
                    lines_added=added, lines_removed=removed, lines_changed=lines_changed, diff_text=diff_txt,
                    module_sha256=mod_sha, test_sha256=tst_sha, exec_receipt_id="", plan_v1=plan_doc,
                )
        except SyntaxError:
            contract_clean = True

    # Stage 3: SYS-025 zero-stub placeholder gate
    stub_issue = payload_gate.check_payload(mod_src)
    stub_clean = stub_issue is None
    if not stub_clean:
        return CandidateEvaluation(
            strategy=clean_strategy, target_rel_path=rel_norm, module_source=mod_src, test_source=tst_src,
            verified=False, stage_failed="placeholder_gate", reason=str(stub_issue),
            ast_clean=False, stub_clean=False, contract_clean=contract_clean, test_shape_clean=False,
            plan_gates_passed=plan_ok, plan_gates_passed_count=gates_passed_cnt,
            tests_passed=0, tests_failed=0, tests_skipped=0,
            lines_added=added, lines_removed=removed, lines_changed=lines_changed, diff_text=diff_txt,
            module_sha256=mod_sha, test_sha256=tst_sha, exec_receipt_id="", plan_v1=plan_doc,
        )

    # Stage 4: Unified deny-by-default AST firewall on BOTH module & test (SYS-025 / REQ-VER-1)
    mod_fw_ok, mod_violations = ast_firewall.check(mod_src)
    tst_fw_ok, tst_violations = ast_firewall.check(tst_src) if tst_src.strip() else (False, [])
    ast_clean = bool(mod_fw_ok and tst_fw_ok)
    if not mod_fw_ok:
        msgs = "; ".join(ast_firewall.controller_message(v) for v in mod_violations[:3])
        return CandidateEvaluation(
            strategy=clean_strategy, target_rel_path=rel_norm, module_source=mod_src, test_source=tst_src,
            verified=False, stage_failed="ast_firewall", reason=msgs or "AST firewall rejected candidate module",
            ast_clean=False, stub_clean=True, contract_clean=contract_clean, test_shape_clean=False,
            plan_gates_passed=plan_ok, plan_gates_passed_count=gates_passed_cnt,
            tests_passed=0, tests_failed=0, tests_skipped=0,
            lines_added=added, lines_removed=removed, lines_changed=lines_changed, diff_text=diff_txt,
            module_sha256=mod_sha, test_sha256=tst_sha, exec_receipt_id="", plan_v1=plan_doc,
        )
    if not tst_fw_ok:
        msgs = (
            "; ".join(ast_firewall.controller_message(v) for v in tst_violations[:3])
            if tst_violations
            else "empty or unparseable candidate test source"
        )
        return CandidateEvaluation(
            strategy=clean_strategy, target_rel_path=rel_norm, module_source=mod_src, test_source=tst_src,
            verified=False, stage_failed="ast_firewall", reason=f"test AST firewall: {msgs}",
            ast_clean=False, stub_clean=True, contract_clean=contract_clean, test_shape_clean=False,
            plan_gates_passed=plan_ok, plan_gates_passed_count=gates_passed_cnt,
            tests_passed=0, tests_failed=0, tests_skipped=0,
            lines_added=added, lines_removed=removed, lines_changed=lines_changed, diff_text=diff_txt,
            module_sha256=mod_sha, test_sha256=tst_sha, exec_receipt_id="", plan_v1=plan_doc,
        )

    # Stage 5: Non-trivial test shape check (REQ-VER-3)
    mod_stem = Path(rel_norm).stem
    pkg_top = Path(rel_norm).parts[0] if len(Path(rel_norm).parts) > 1 else mod_stem
    shape_issues = check_test_shape(tst_src, mod_stem)
    if shape_issues and pkg_top != mod_stem:
        alt_issues = check_test_shape(tst_src, pkg_top)
        if not alt_issues:
            shape_issues = []
    if shape_issues:
        return CandidateEvaluation(
            strategy=clean_strategy, target_rel_path=rel_norm, module_source=mod_src, test_source=tst_src,
            verified=False, stage_failed="test_shape", reason="; ".join(shape_issues),
            ast_clean=True, stub_clean=True, contract_clean=contract_clean, test_shape_clean=False,
            plan_gates_passed=plan_ok, plan_gates_passed_count=gates_passed_cnt,
            tests_passed=0, tests_failed=0, tests_skipped=0,
            lines_added=added, lines_removed=removed, lines_changed=lines_changed, diff_text=diff_txt,
            module_sha256=mod_sha, test_sha256=tst_sha, exec_receipt_id="", plan_v1=plan_doc,
        )

    # Check Plan v1 measurable change-size gates before running code
    if not plan_ok:
        failed_reasons = [c["reason"] for c in checks_list if not c.get("passed")]
        return CandidateEvaluation(
            strategy=clean_strategy, target_rel_path=rel_norm, module_source=mod_src, test_source=tst_src,
            verified=False, stage_failed="plan_gates", reason="; ".join(failed_reasons),
            ast_clean=True, stub_clean=True, contract_clean=contract_clean, test_shape_clean=True,
            plan_gates_passed=False, plan_gates_passed_count=gates_passed_cnt,
            tests_passed=0, tests_failed=0, tests_skipped=0,
            lines_added=added, lines_removed=removed, lines_changed=lines_changed, diff_text=diff_txt,
            module_sha256=mod_sha, test_sha256=tst_sha, exec_receipt_id="", plan_v1=plan_doc,
        )

    # Stage 6: Hermetic temp-workspace pytest execution via exec_kernel.run (REQ-VER-2..4)
    sandbox_dir = _stage_workspace_copy(ws_root, rel_norm)
    try:
        cand_mod_file = sandbox_dir / rel_norm
        cand_mod_file.parent.mkdir(parents=True, exist_ok=True)
        cand_mod_file.write_text(mod_src, encoding="utf-8")
        cand_test_file = sandbox_dir / "test_candidate_eval.py"
        cand_test_file.write_text(tst_src, encoding="utf-8")

        journal = exec_journal or exec_kernel.ExecJournal()
        req = CommandRequest(
            argv=(sys.executable, "-m", "pytest", "-q", "--tb=short", "test_candidate_eval.py"),
            cwd=str(sandbox_dir),
            origin="model",
            root=str(sandbox_dir),
            timeout_s=timeout_s,
        )
        res = exec_kernel.run(
            req,
            confirm=lambda _p, cls: getattr(cls, 'tier', '') in ('READ', 'VERIFY'),
            journal=journal,
            artifacts_dir=Path(exec_artifacts_dir) if exec_artifacts_dir else None,
        )
        combined_out = (res.stdout_view or "") + ("\n" + res.stderr_view if res.stderr_view else "")
        p_cnt, f_cnt, s_cnt = _parse_pytest_summary(combined_out)

        if res.timed_out:
            return CandidateEvaluation(
                strategy=clean_strategy, target_rel_path=rel_norm, module_source=mod_src, test_source=tst_src,
                verified=False, stage_failed="pytest", reason=f"pytest timed out after {timeout_s:.0f}s",
                ast_clean=True, stub_clean=True, contract_clean=contract_clean, test_shape_clean=True,
                plan_gates_passed=True, plan_gates_passed_count=gates_passed_cnt,
                tests_passed=p_cnt, tests_failed=f_cnt, tests_skipped=s_cnt,
                lines_added=added, lines_removed=removed, lines_changed=lines_changed, diff_text=diff_txt,
                module_sha256=mod_sha, test_sha256=tst_sha, exec_receipt_id=res.receipt_id, plan_v1=plan_doc,
            )
        if res.exit_code != 0 or f_cnt > 0:
            tail = " ".join(ln.strip() for ln in combined_out.splitlines()[-4:] if ln.strip())[:200]
            return CandidateEvaluation(
                strategy=clean_strategy, target_rel_path=rel_norm, module_source=mod_src, test_source=tst_src,
                verified=False, stage_failed="pytest", reason=tail or f"pytest exit_code={res.exit_code}",
                ast_clean=True, stub_clean=True, contract_clean=contract_clean, test_shape_clean=True,
                plan_gates_passed=True, plan_gates_passed_count=gates_passed_cnt,
                tests_passed=p_cnt, tests_failed=f_cnt, tests_skipped=s_cnt,
                lines_added=added, lines_removed=removed, lines_changed=lines_changed, diff_text=diff_txt,
                module_sha256=mod_sha, test_sha256=tst_sha, exec_receipt_id=res.receipt_id, plan_v1=plan_doc,
            )
        if p_cnt < 1:
            return CandidateEvaluation(
                strategy=clean_strategy, target_rel_path=rel_norm, module_source=mod_src, test_source=tst_src,
                verified=False, stage_failed="pytest", reason="0 passing tests collected/executed",
                ast_clean=True, stub_clean=True, contract_clean=contract_clean, test_shape_clean=True,
                plan_gates_passed=True, plan_gates_passed_count=gates_passed_cnt,
                tests_passed=0, tests_failed=f_cnt, tests_skipped=s_cnt,
                lines_added=added, lines_removed=removed, lines_changed=lines_changed, diff_text=diff_txt,
                module_sha256=mod_sha, test_sha256=tst_sha, exec_receipt_id=res.receipt_id, plan_v1=plan_doc,
            )
        if s_cnt > 0:
            return CandidateEvaluation(
                strategy=clean_strategy, target_rel_path=rel_norm, module_source=mod_src, test_source=tst_src,
                verified=False, stage_failed="pytest",
                reason=f"candidate test suite skipped {s_cnt} test(s); full execution required",
                ast_clean=True, stub_clean=True, contract_clean=contract_clean, test_shape_clean=True,
                plan_gates_passed=True, plan_gates_passed_count=gates_passed_cnt,
                tests_passed=p_cnt, tests_failed=f_cnt, tests_skipped=s_cnt,
                lines_added=added, lines_removed=removed, lines_changed=lines_changed, diff_text=diff_txt,
                module_sha256=mod_sha, test_sha256=tst_sha, exec_receipt_id=res.receipt_id, plan_v1=plan_doc,
            )

        return CandidateEvaluation(
            strategy=clean_strategy, target_rel_path=rel_norm, module_source=mod_src, test_source=tst_src,
            verified=True, stage_failed="", reason="all 6 verification & plan gates passed",
            ast_clean=True, stub_clean=True, contract_clean=contract_clean, test_shape_clean=True,
            plan_gates_passed=True, plan_gates_passed_count=gates_passed_cnt,
            tests_passed=p_cnt, tests_failed=0, tests_skipped=0,
            lines_added=added, lines_removed=removed, lines_changed=lines_changed, diff_text=diff_txt,
            module_sha256=mod_sha, test_sha256=tst_sha, exec_receipt_id=res.receipt_id, plan_v1=plan_doc,
        )
    finally:
        shutil.rmtree(sandbox_dir, ignore_errors=True)


def build_strategy_prompt(
    *,
    strategy: str,
    target_rel_path: str,
    goal: str,
    existing_source: Optional[str] = None,
) -> str:
    """Build the structured synthesis prompt for ``baseline``, ``alternative``, or ``conservative``."""
    directive = STRATEGY_DIRECTIVES.get(strategy, STRATEGY_DIRECTIVES["baseline"])
    mod_name = Path(target_rel_path).stem
    contract = None
    if existing_source:
        try:
            contract = production_synth.extract_module_contract(existing_source)
        except SyntaxError:
            contract = None
    contract_block = production_synth.format_contract_prompt_block(goal, contract)
    current_block = (
        f"\nCURRENT CONTENT OF {target_rel_path}:\n{existing_source}\n"
        if existing_source is not None
        else ""
    )
    fmt_block = production_synth.delimited_instructions(target_rel_path, mod_name)
    return (
        f"You are OSIRIS L4 Multi-Candidate Synthesizer.\n"
        f"{directive}\n\n"
        f"SPECIFICATION:\n{goal}\n\n"
        f"{contract_block}"
        f"{current_block}\n"
        f"{fmt_block}\n"
    )


def select_recommended_strategy(
    candidates: Sequence[CandidateEvaluation],
    *,
    prefer: str = "baseline",
    quarantine_queue: Optional[quarantine.QuarantineQueue] = None,
) -> Tuple[Optional[str], str]:
    """Deterministically select the recommended verified candidate before ``/apply``."""
    q = quarantine_queue or quarantine.QuarantineQueue()
    eligible: List[CandidateEvaluation] = []
    for c in candidates:
        if not c.verified:
            continue
        qv = q.check_promotion_allowed(candidate_digest=c.module_sha256, target_path=c.target_rel_path)
        if qv.allowed:
            eligible.append(c)

    if not eligible:
        return None, "0 of 3 candidates passed verification & quarantine gates"

    for c in eligible:
        if c.strategy == prefer:
            return c.strategy, f"{c.strategy} passed all gates ({c.tests_passed}P, +{c.lines_added}/-{c.lines_removed})"

    # Otherwise prefer highest passing test count, then smallest diff (conservative tie-breaker).
    ranked = sorted(eligible, key=lambda c: (-c.tests_passed, c.lines_changed, STRATEGIES.index(c.strategy) if c.strategy in STRATEGIES else 99))
    best = ranked[0]
    return best.strategy, f"{best.strategy} selected ({best.tests_passed}P, +{best.lines_added}/-{best.lines_removed})"


def propose_three_candidates(
    *,
    root: PathArg,
    target_rel_path: str,
    goal: str,
    proposer_fn: Optional[Callable[[str, str], Union[str, Tuple[str, str], Mapping[str, str]]]] = None,
    strategy_payloads: Optional[Mapping[str, Tuple[str, str]]] = None,
    existing_source: Optional[str] = None,
    prefer: str = "baseline",
    thresholds: plan_gates.PlanGateThresholds = plan_gates.PlanGateThresholds(),
    osiris_home: PathArg = None,
    exec_journal: Optional[exec_kernel.ExecJournal] = None,
    exec_artifacts_dir: PathArg = None,
    persist_receipt: bool = True,
) -> MultiCandidateBundle:
    """Generate and verify ``baseline``, ``alternative``, and ``conservative`` candidates side by side."""
    ws_root = Path(root or os.getcwd()).expanduser().resolve()
    rel_norm = plan_gates.normalize_rel_path(target_rel_path)
    if existing_source is None and (ws_root / rel_norm).is_file():
        existing_source = (ws_root / rel_norm).read_text(encoding="utf-8", errors="replace")

    evaluated: List[CandidateEvaluation] = []
    for strat in STRATEGIES:
        mod_src = ""
        tst_src = ""
        if strategy_payloads and strat in strategy_payloads:
            mod_src, tst_src = strategy_payloads[strat]
        elif proposer_fn is not None:
            prompt = build_strategy_prompt(
                strategy=strat,
                target_rel_path=rel_norm,
                goal=goal,
                existing_source=existing_source,
            )
            raw_reply = proposer_fn(strat, prompt)
            if isinstance(raw_reply, tuple) and len(raw_reply) == 2:
                mod_src, tst_src = str(raw_reply[0]), str(raw_reply[1])
            elif isinstance(raw_reply, Mapping):
                mod_src = str(raw_reply.get("module") or "")
                tst_src = str(raw_reply.get("test") or "")
            else:
                parsed = production_synth.parse_delimited_or_json_proposal(
                    str(raw_reply or ""), default_path=rel_norm
                )
                if parsed:
                    mod_src = str(parsed.get("module") or "")
                    tst_src = str(parsed.get("test") or "")

        ev = evaluate_strategy_candidate(
            strategy=strat,
            root=ws_root,
            target_rel_path=rel_norm,
            goal=goal,
            module_source=mod_src,
            test_source=tst_src,
            existing_source=existing_source,
            thresholds=thresholds,
            exec_journal=exec_journal,
            exec_artifacts_dir=exec_artifacts_dir,
        )
        evaluated.append(ev)

    q_path = quarantine.default_quarantine_path(osiris_home) if osiris_home else None
    q_queue = quarantine.QuarantineQueue(q_path)
    selected_strat, sel_reason = select_recommended_strategy(
        evaluated, prefer=prefer, quarantine_queue=q_queue
    )
    bundle_id = f"mc-{uuid.uuid4().hex[:12]}"
    rhash = ""
    if persist_receipt:
        log_path = default_multi_candidate_log(osiris_home)
        journal = exec_kernel.ExecJournal(log_path)
        body = {
            "schema": RECEIPT_SCHEMA,
            "bundle_id": bundle_id,
            "target_rel_path": rel_norm,
            "goal": str(goal)[:500],
            "selected_strategy": selected_strategy_or_none(selected_strat),
            "selection_reason": sel_reason,
            "candidates": [c.to_receipt_dict() for c in evaluated],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        exec_kernel.canon(body)
        rhash = journal.append(body)
        try:
            os.chmod(log_path, 0o600)
        except OSError:
            pass

    return MultiCandidateBundle(
        bundle_id=bundle_id,
        target_rel_path=rel_norm,
        goal=goal,
        candidates=tuple(evaluated),
        selected_strategy=selected_strat,
        selection_reason=sel_reason,
        receipt_hash=rhash,
    )


def selected_strategy_or_none(val: Optional[str]) -> str:
    return str(val) if val else "NONE"


def switch_selected_strategy(bundle: MultiCandidateBundle, strategy: str) -> MultiCandidateBundle:
    """Switch the active candidate for ``/apply`` to another verified strategy."""
    clean = strategy.strip().lower()
    cand = bundle.by_strategy(clean)
    if cand is None:
        raise ValueError(f"Unknown strategy {strategy!r}; choose from {', '.join(STRATEGIES)}")
    if not cand.verified:
        raise ValueError(
            f"Cannot select unverified strategy {clean!r} (failed at {cand.stage_failed}: {cand.reason})"
        )
    return MultiCandidateBundle(
        bundle_id=bundle.bundle_id,
        target_rel_path=bundle.target_rel_path,
        goal=bundle.goal,
        candidates=bundle.candidates,
        selected_strategy=clean,
        selection_reason=f"operator selected {clean} ({cand.tests_passed}P, +{cand.lines_added}/-{cand.lines_removed})",
        receipt_hash=bundle.receipt_hash,
    )


def _cell(text: str, col_w: int) -> str:
    s = str(text)
    if len(s) <= col_w:
        return s.ljust(col_w)
    return s[: max(1, col_w - 1)] + "…"


def render_side_by_side(bundle: MultiCandidateBundle, *, width: int = 88) -> str:
    """Render compact side-by-side 3-strategy comparison table (<100 cols; Node Alpha safe)."""
    eff_w = max(68, min(width, 96))
    label_w = 11
    # 3 columns + label + 3 separators (" | " = 3 chars * 3 = 9 chars)
    col_w = max(14, (eff_w - label_w - 9) // 3)

    by_s = {c.strategy: c for c in bundle.candidates}
    ordered = [by_s.get(s) for s in STRATEGIES]

    def row(label: str, vals: Sequence[str]) -> str:
        cells = [_cell(v, col_w) for v in vals]
        return f"{_cell(label, label_w)} | {cells[0]} | {cells[1]} | {cells[2]}"

    sep = f"{'-' * label_w}-+-{'-' * col_w}-+-{'-' * col_w}-+-{'-' * col_w}"

    verdicts: List[str] = []
    ast_stubs: List[str] = []
    pytests: List[str] = []
    diffs: List[str] = []
    pgates: List[str] = []
    digests: List[str] = []

    for s, c in zip(STRATEGIES, ordered):
        if c is None:
            verdicts.append("MISSING")
            ast_stubs.append("--")
            pytests.append("--")
            diffs.append("--")
            pgates.append("--")
            digests.append("--")
            continue
        star = " *" if bundle.selected_strategy == s else ""
        verdicts.append(f"PASS{star}" if c.verified else f"FAIL:{c.stage_failed}")
        a_tag = "ok" if c.ast_clean else "NO"
        s_tag = "ok" if c.stub_clean else "NO"
        ast_stubs.append(f"ast:{a_tag} stb:{s_tag}")
        if c.tests_passed or c.tests_failed or c.tests_skipped:
            pytests.append(f"{c.tests_passed}P/{c.tests_failed}F/{c.tests_skipped}S")
        else:
            pytests.append("--")
        diffs.append(f"+{c.lines_added}/-{c.lines_removed} ({c.lines_changed}L)")
        pgates.append(f"{c.plan_gates_passed_count}/4 {'OK' if c.plan_gates_passed else 'FAIL'}")
        digests.append(c.module_sha256[:12])

    header_title = f"L4 CANDIDATES · {bundle.target_rel_path} ({bundle.bundle_id})"[:eff_w]
    lines = [
        header_title,
        row("Strategy", STRATEGIES),
        sep,
        row("Verdict", verdicts),
        row("AST / Stub", ast_stubs),
        row("Pytest", pytests),
        row("Diff (+/-)", diffs),
        row("Plan Gates", pgates),
        row("SHA-256", digests),
    ]
    for c in ordered:
        if c is not None and not c.verified:
            fail_line = f"  ! {c.strategy}: [{c.stage_failed}] {c.reason}"
            lines.append(fail_line[:eff_w])
    sel_footer = (
        f"Selected for /apply: {bundle.selected_strategy} · {bundle.selection_reason}"
        if bundle.selected_strategy
        else f"Selected for /apply: NONE · {bundle.selection_reason}"
    )
    lines.append(sel_footer[:eff_w])
    return "\n".join(lines)


# Session holder for interactive CLI inspection (/candidates) before /apply
_ACTIVE_BUNDLE: Optional[MultiCandidateBundle] = None


def get_active_bundle() -> Optional[MultiCandidateBundle]:
    return _ACTIVE_BUNDLE


def set_active_bundle(bundle: Optional[MultiCandidateBundle]) -> None:
    global _ACTIVE_BUNDLE
    _ACTIVE_BUNDLE = bundle


def run_command(
    args: str = "",
    *,
    bundle: Optional[MultiCandidateBundle] = None,
    width: int = 88,
    on_select: Optional[Callable[[CandidateEvaluation], None]] = None,
) -> str:
    """CLI handler for ``/candidates [table | diff <strategy> | plan <strategy> | select <strategy>]``."""
    active = bundle or _ACTIVE_BUNDLE
    try:
        tokens = shlex.split(args or "")
    except ValueError:
        tokens = (args or "").strip().split()

    if active is None:
        return (
            "No active L4 multi-candidate comparison in this session.\n"
            "Use /propose <path> :: <goal> (or /sprint execute --compare) to evaluate "
            "baseline, alternative, and conservative candidates side by side before /apply."
        )

    sub = tokens[0].lower() if tokens else "table"
    if sub in ("table", "list"):
        return render_side_by_side(active, width=width)
    if sub == "diff" and len(tokens) >= 2:
        cand = active.by_strategy(tokens[1])
        if cand is None:
            return f"Unknown strategy {tokens[1]!r}; choose baseline, alternative, or conservative."
        return cand.diff_text or f"(no diff lines for {cand.strategy})"
    if sub == "plan" and len(tokens) >= 2:
        cand = active.by_strategy(tokens[1])
        if cand is None:
            return f"Unknown strategy {tokens[1]!r}; choose baseline, alternative, or conservative."
        return json.dumps(cand.plan_v1, indent=2, sort_keys=True)
    if sub == "select" and len(tokens) >= 2:
        try:
            updated = switch_selected_strategy(active, tokens[1])
        except ValueError as exc:
            return f"Selection refused: {exc}"
        if bundle is None:
            set_active_bundle(updated)
        if on_select is not None and updated.selected is not None:
            on_select(updated.selected)
        return render_side_by_side(updated, width=width)
    return "Usage: /candidates [table | diff <strategy> | plan <strategy> | select <strategy>]"
