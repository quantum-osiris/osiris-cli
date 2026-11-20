"""
OSIRIS Multi-Agent Logical Roles
================================
Implements distinct logical roles using the Gemma-native provider substrate:
  1. Intent Agent: Parses goals into typed intent proposals.
  2. Architect Agent: Decomposes intents into bounded task specs.
  3. Implementation Agent: Proposes candidate patches and tests.
  4. Research Agent: Formulates falsifiable hypotheses with citations.
  5. Verification Agent: Runs AST firewall and isolated unit checks.
  6. Benchmark Agent: Evaluates candidates against acceptance suites.
  7. Memory & Lineage Agent: Persists verified audit records into the ledger.
"""

from __future__ import annotations

import ast
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from osiris_cli.providers import BaseModelProvider, ProviderRequest, get_active_provider
from .controller import GovernanceController
from .schemas import (
    CodeProposal,
    DecomposedPlan,
    ExecutionGrant,
    IntentProposal,
    TaskSpec,
    VerificationReport,
    VerificationResult,
)


class IntentAgent:
    """Interprets human input and produces a typed IntentProposal."""

    def __init__(self, provider: Optional[BaseModelProvider] = None):
        self.provider = provider or get_active_provider()

    def interpret(self, raw_input: str) -> IntentProposal:
        # Fast deterministic classification for common keywords
        raw_lower = raw_input.strip().lower()
        intent_type = "query"
        domain = "general"
        confidence = 0.85
        risk = "low"

        if raw_lower.startswith("/") or "def " in raw_input or "fix" in raw_lower or "implement" in raw_lower:
            intent_type = "code_modification"
            risk = "medium"
        elif "quantum" in raw_lower or "circuit" in raw_lower or "qvm" in raw_lower or "qubit" in raw_lower:
            intent_type = "research"
            domain = "quantum"
        elif "bench" in raw_lower or "eval" in raw_lower or "test" in raw_lower:
            intent_type = "benchmark"
            confidence = 0.90
        elif "status" in raw_lower or "health" in raw_lower:
            intent_type = "status"

        return IntentProposal(
            intent_id=str(uuid.uuid4())[:8],
            raw_prompt=raw_input,
            intent_type=intent_type,
            domain=domain,
            confidence=confidence,
            target_scope=[],
            risk_level=risk,
            provenance="human_operator",
        )


class ArchitectAgent:
    """Decomposes an IntentProposal into a bounded, testable DecomposedPlan."""

    def __init__(self, provider: Optional[BaseModelProvider] = None):
        self.provider = provider or get_active_provider()

    def plan(self, intent: IntentProposal, target_file: str = "candidate.py") -> DecomposedPlan:
        task = TaskSpec(
            task_id=f"task_{intent.intent_id}",
            title=f"Execute: {intent.raw_prompt[:40]}",
            spec=intent.raw_prompt,
            target_files=[target_file],
            acceptance_criteria=["Module parses without syntax errors", "Candidate passes its unit test"],
            timeout_seconds=60.0,
        )
        return DecomposedPlan(
            plan_id=str(uuid.uuid4())[:8],
            intent_id=intent.intent_id,
            tasks=[task],
            architectural_rationale="Bounded single-module task decomposition with discrete acceptance criteria.",
        )


class ImplementationAgent:
    """Proposes code patches and tests conforming to structured proposal schemas."""

    def __init__(self, provider: Optional[BaseModelProvider] = None):
        self.provider = provider or get_active_provider()

    def synthesize(self, task: TaskSpec) -> CodeProposal:
        target_file = task.target_files[0] if task.target_files else "candidate.py"
        module_name = os.path.splitext(os.path.basename(target_file))[0]

        schema = {
            "type": "object",
            "properties": {
                "module_code": {"type": "string"},
                "test_code": {"type": "string"},
                "changes_summary": {"type": "string"},
            },
            "required": ["module_code", "test_code"],
        }

        req = ProviderRequest(
            prompt=f"Implement task: {task.spec}\nModule name: {module_name}\nWrite clean Python code and a unit test.",
            system="You are OSIRIS Implementation Agent. Write production Python code and tests.",
            response_schema=schema,
            max_tokens=1024,
        )

        resp = self.provider.generate(req)
        module_code = ""
        test_code = ""
        summary = ""

        if resp.structured_output and isinstance(resp.structured_output, dict):
            module_code = resp.structured_output.get("module_code", "")
            test_code = resp.structured_output.get("test_code", "")
            summary = resp.structured_output.get("changes_summary", "")

        # Fallback if raw text or mock
        if not module_code:
            module_code = (
                f"# Generated by OSIRIS Implementation Agent\n"
                f"def solve():\n"
                f"    \"\"\"{task.title}\"\"\"\n"
                f"    return True\n"
            )
            test_code = (
                f"import {module_name}\n\n"
                f"def test_solve():\n"
                f"    assert {module_name}.solve() is True\n"
            )
            summary = f"Synthesized module {module_name} and unit test"

        return CodeProposal(
            proposal_id=str(uuid.uuid4())[:8],
            task_id=task.task_id,
            target_file=target_file,
            module_code=module_code,
            test_code=test_code,
            changes_summary=summary,
            rollback_strategy="Revert file from git working tree",
        )


_PYTEST_COUNTS = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed)")


def check_test_shape(test_code: str, module_name: str) -> List[str]:
    """Why a test file cannot count as evidence for ``module_name`` (empty list: none found).

    Refused: no test function; never importing the target module; no assertion that is not a
    constant (``assert True``, ``assert 1``); and calls that never name anything imported from the
    target. These are static facts about the test text; running it is a separate check.
    """
    try:
        tree = ast.parse(test_code)
    except SyntaxError as exc:
        return [f"test code does not parse: {exc}"]
    problems: List[str] = []
    tests = [n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test")]
    if not tests:
        problems.append("no test functions (names starting with 'test')")
    bound: set = set()
    imports_target = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == module_name or a.name.split(".")[0] == module_name:
                    imports_target = True
                    bound.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module \
                and node.module.split(".")[0] == module_name:
            imports_target = True
            bound.update(a.asname or a.name for a in node.names)
    if not imports_target:
        problems.append(f"tests never import the target module '{module_name}'")
    asserts = [n for n in ast.walk(tree) if isinstance(n, ast.Assert)]
    raises = [n for n in ast.walk(tree) if isinstance(n, ast.With) and any(
        isinstance(i.context_expr, ast.Call) and isinstance(i.context_expr.func, ast.Attribute)
        and i.context_expr.func.attr == "raises" for i in n.items)]
    real = [a for a in asserts if not isinstance(a.test, ast.Constant)]
    if not real and not raises:
        problems.append("only constant assertions (e.g. 'assert True'), or none")
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    if imports_target and not (bound & used):
        problems.append(f"tests import '{module_name}' but never use it")
    return problems


class VerificationAgent:
    """Independently verifies candidate code proposals: AST firewall, test shape, then a real
    pytest run of the proposal's tests in a temporary workspace through the execution kernel.

    The run is classified by exec_policy (VERIFY tier), uses the kernel's scrubbed environment and
    process-group kill, and leaves one hash-chained receipt. The only command this agent approves
    is its own fixed pytest invocation. PASS needs exit code 0 and at least one collected test.
    """

    def __init__(self, journal_path: Optional[str] = None, artifacts_dir: Optional[str] = None,
                 timeout_s: float = 120.0):
        self.journal_path = journal_path
        self.artifacts_dir = artifacts_dir
        self.timeout_s = timeout_s

    def _run_tests(self, proposal: CodeProposal, module_name: str) -> Tuple[bool, int, str]:
        import shutil
        import sys
        import tempfile

        from osiris_cli import exec_kernel
        from osiris_cli.exec_policy import CommandRequest

        work = tempfile.mkdtemp(prefix="osiris-verify-")
        try:
            with open(os.path.join(work, module_name + ".py"), "w", encoding="utf-8", newline="\n") as fh:
                fh.write(proposal.module_code)
            test_name = f"test_{module_name}_proposal.py"
            with open(os.path.join(work, test_name), "w", encoding="utf-8", newline="\n") as fh:
                fh.write(proposal.test_code)
            argv = (sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", test_name)
            req = CommandRequest(argv=argv, cwd=work, origin="model", root=work, timeout_s=self.timeout_s)

            def approve_own_command(_prompt, cls) -> bool:
                return cls.tier == "VERIFY" and tuple(cls.normalized_argv)[-1:] == (test_name,)

            result = exec_kernel.run(req, approve_own_command, exec_kernel.ExecJournal(self.journal_path),
                                     artifacts_dir=self.artifacts_dir)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        out = result.stdout_view + "\n" + result.stderr_view
        counts = {kind.rstrip("s"): int(n) for n, kind in _PYTEST_COUNTS.findall(out)}
        collected = sum(counts.get(k, 0) for k in ("passed", "failed", "error", "skipped", "xfailed", "xpassed"))
        if result.decision not in ("ALLOW", "ASK_APPROVED"):
            return False, collected, f"test run {result.decision}: {'; '.join(result.reasons)}"
        if result.exit_code != 0:
            why = "no tests collected" if result.exit_code == 5 else f"pytest exit code {result.exit_code}"
            return False, collected, f"{why}; receipt {result.receipt_id}\n{out[-1500:]}"
        if counts.get("passed", 0) == 0:
            return False, collected, f"no test passed; receipt {result.receipt_id}"
        return True, collected, f"{counts.get('passed', 0)} passed; receipt {result.receipt_id}"

    def verify(self, proposal: CodeProposal) -> VerificationReport:
        t0 = time.monotonic()
        results: List[VerificationResult] = []
        module_name = os.path.splitext(os.path.basename(proposal.target_file))[0] or "candidate"

        # 1. AST Firewall Check (SYS-025)
        ast_ok, ast_violations = GovernanceController.inspect_ast(proposal.module_code)
        results.append(VerificationResult(
            check_name="ast_firewall_module", passed=ast_ok,
            reason="Clean AST" if ast_ok else f"Violations: {ast_violations}",
            execution_time_ms=(time.monotonic() - t0) * 1000))
        test_ast_ok, test_violations = GovernanceController.inspect_ast(proposal.test_code)
        results.append(VerificationResult(
            check_name="ast_firewall_test", passed=test_ast_ok,
            reason="Clean AST" if test_ast_ok else f"Violations: {test_violations}",
            execution_time_ms=(time.monotonic() - t0) * 1000))
        all_ast_ok = ast_ok and test_ast_ok
        banned = ast_violations + test_violations

        # 2. Test shape: must exercise the target, with a real assertion
        shape = check_test_shape(proposal.test_code, module_name) if all_ast_ok else ["AST firewall failed"]
        results.append(VerificationResult(
            check_name="test_shape", passed=not shape,
            reason="Tests import and exercise the target" if not shape else "; ".join(shape)))

        # 3. Real test run in a temporary workspace, through the execution kernel
        test_passed, collected, detail = False, 0, "not run: earlier checks failed"
        if all_ast_ok and not shape:
            try:
                test_passed, collected, detail = self._run_tests(proposal, module_name)
            except Exception as exc:  # the kernel or the filesystem failed: that is a FAIL
                test_passed, detail = False, f"test run could not start: {exc}"
        results.append(VerificationResult(
            check_name="sandboxed_pytest", passed=test_passed, reason=detail,
            execution_time_ms=(time.monotonic() - t0) * 1000))

        overall = "PASS" if (all_ast_ok and not shape and test_passed) else "FAIL"
        return VerificationReport(
            proposal_id=proposal.proposal_id,
            overall_verdict=overall,
            static_ast_pass=all_ast_ok,
            banned_constructs_found=banned,
            acceptance_test_pass=test_passed,
            checks=results,
            timestamp=time.time(),
            proposal_digest=proposal.payload_digest(),
            tests_collected=collected,
        )


class ResearchAgent:
    """Ingests sources and structures falsifiable hypotheses with bibliographic metadata."""

    def extract_hypothesis(self, title: str, claims: List[str], metric: str, refutation: str) -> Dict[str, Any]:
        return {
            "hypothesis_id": str(uuid.uuid4())[:8],
            "title": title,
            "claims": claims,
            "eval_metric": metric,
            "refutation_condition": refutation,
            "status": "UNTESTED",
            "created_at": time.time(),
        }


class MemoryLineageAgent:
    """Records verified proposals and runs in the genome ledger through GenomeLedger.append, so
    they share its entry format ({prev, ts, kind, payload, hash}) and its locks with /apply and
    sprint runs, and the chain stays verifiable."""

    def __init__(self, ledger_file: Optional[str] = None):
        self.ledger_file = ledger_file or os.path.expanduser("~/.osiris/genome_ledger.jsonl")

    def record_event(self, event_type: str, payload: Dict[str, Any]) -> str:
        """Append one entry and return its hash. Raises, writing nothing, if the ledger cannot be
        appended to (dnalang-core's ledger.py not found, or an unreadable tail)."""
        import genome_ledger

        return genome_ledger.GenomeLedger(self.ledger_file).append(event_type, payload)["hash"]
