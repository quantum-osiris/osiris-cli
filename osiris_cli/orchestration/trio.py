"""Research -> Developer -> Critic: the first governed three-agent slice.

Task shape: a small Python repo has failing tests; propose a tested correction.

  * The repo is never written. Every run works on a throwaway copy (the sandbox).
  * Test results come only from ``run_pytest`` (no shell, scrubbed env, wall timeout);
    an agent's own statement that tests passed is just a sentence for the firewall.
  * The baseline (unpatched) run is frozen first; the Critic re-derives its own verdict
    from fresh runs plus the diff and never reads the Developer's conclusions as facts.
  * The best outcome is PENDING_HUMAN_APPROVAL. Nothing here applies, merges or promotes.
  * Candidates stay UNMEASURED until the runner has measured them; rejected and quarantined
    candidates are kept (patch + evidence) under the run directory.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

from .agent_spec import (AgentRun, AgentSpec, AgentState, BudgetExceeded, RunBudget, digest,
                         new_run_id)
from .controller import GovernanceController
from .evidence_firewall import ExecutionEvidence, FirewallResult, check_report

_IGNORE = shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".venv", "venv", "node_modules",
                                 ".pytest_cache")

RESEARCH = AgentSpec("research", "1.0", "research",
                     frozenset({"READ_ONLY_ANALYSIS", "RETURN_STRUCTURED_PROPOSAL"}),
                     input_schema="RepoTask", output_schema="ResearchFindings", evidence_required=True)
DEVELOPER = AgentSpec("developer", "1.0", "developer",
                      frozenset({"RETURN_STRUCTURED_PROPOSAL"}),  # no WORKSPACE_WRITE: it only proposes
                      input_schema="ResearchFindings", output_schema="DeveloperOutput")
CRITIC = AgentSpec("critic", "1.0", "critic",
                   frozenset({"READ_ONLY_ANALYSIS", "REPLAY_FIXTURE", "SUBPROCESS_EXEC"}),
                   input_schema="CandidatePatch", output_schema="CriticVerdict")


# ── sandbox + runner ─────────────────────────────────────────────────────────

def make_sandbox(repo: str, parent: str) -> str:
    dst = os.path.join(parent, "sandbox_" + hashlib.sha1(os.urandom(8)).hexdigest()[:8])
    shutil.copytree(repo, dst, ignore=_IGNORE)
    return dst


def tree_digest(root: str) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts:
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def run_pytest(workdir: str, timeout: float = 60.0) -> Tuple[ExecutionEvidence, Set[str], Set[str]]:
    """Run pytest in workdir. Returns (evidence, passed test ids, failed test ids)."""
    argv = (sys.executable, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider", "--tb=short")
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": workdir,
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": workdir, "PYTHONHASHSEED": "0"}
    try:
        p = subprocess.run(argv, cwd=workdir, env=env, capture_output=True, text=True,
                           timeout=timeout, shell=False)
        out, rc = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = ((e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes)
                   else (e.stdout or "")) + "\n[TIMEOUT]", 124
    passed = set(re.findall(r"^PASSED (\S+)", out, re.M))
    failed = set(re.findall(r"^FAILED (\S+)", out, re.M))
    errors = len(re.findall(r"^ERROR ", out, re.M))
    return (ExecutionEvidence(argv, rc, len(passed), len(failed), errors,
                              hashlib.sha256(out.encode()).hexdigest(), out[-1500:],
                              tree_digest(workdir)), passed, failed)


# ── Research ─────────────────────────────────────────────────────────────────

@dataclass
class SourceFinding:
    path: str
    line: int
    sha256: str
    snippet: str
    why: str


@dataclass
class ResearchFindings:
    failing_tests: List[str]
    sources: List[SourceFinding]
    baseline_stdout_digest: str
    uncertainty: str = ("heuristic: sources are traceback frames plus modules the failing tests "
                        "import; the defect may live elsewhere")


_FRAME = re.compile(r"^(\S+\.py):(\d+): ", re.M)


def _imported_repo_sources(root: str, test_rel: str) -> List[Tuple[str, int]]:
    """Repo-local non-test modules a test file imports -> [(relpath, line of the imported def)]."""
    import ast
    try:
        tree = ast.parse(Path(root, test_rel).read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return []
    out: List[Tuple[str, int]] = []
    for node in ast.walk(tree):
        mods = ([node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level
                else [a.name for a in node.names] if isinstance(node, ast.Import) else [])
        for m in mods:
            rel = m.replace(".", "/") + ".py"
            if not Path(root, rel).is_file():
                continue
            line = 1
            if isinstance(node, ast.ImportFrom):
                wanted = {a.name for a in node.names}
                for d in ast.walk(ast.parse(Path(root, rel).read_text(encoding="utf-8", errors="replace"))):
                    if isinstance(d, (ast.FunctionDef, ast.ClassDef)) and d.name in wanted:
                        line = d.lineno
                        break
            out.append((rel, line))
    return out


def research(repo_sandbox: str, baseline: ExecutionEvidence, failing: Set[str]) -> ResearchFindings:
    """Implicated sources = (a) non-test frames in the baseline traceback, (b) repo-local modules
    imported by the failing tests' files. Both are deterministic and recorded with a sha256."""
    cands: List[Tuple[str, int, str]] = [(rel.lstrip("./"), int(ln), "frame in baseline failure traceback")
                                         for rel, ln in _FRAME.findall(baseline.stdout_tail)]
    for test_file in sorted({t.split("::")[0] for t in failing}):
        cands += [(r, ln, f"imported by failing test file {test_file}")
                  for r, ln in _imported_repo_sources(repo_sandbox, test_file)]
    seen: Dict[str, SourceFinding] = {}
    for rel, i, why in cands:
        p = Path(repo_sandbox, rel)
        base = os.path.basename(rel)
        if base.startswith("test_") or base.endswith("_test.py") or not p.is_file() or rel in seen:
            continue
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        seen[rel] = SourceFinding(rel, i, hashlib.sha256(p.read_bytes()).hexdigest(),
                                  "\n".join(lines[max(0, i - 4): i + 3]), why)
    return ResearchFindings(sorted(failing), list(seen.values()), baseline.stdout_digest)


# ── Developer ────────────────────────────────────────────────────────────────

@dataclass
class DeveloperOutput:
    patched_files: Dict[str, str]   # repo-relative path -> full new content
    report: str                     # free text; goes through the evidence firewall


ProposeFn = Callable[[ResearchFindings, str], DeveloperOutput]


def provider_propose_fn(provider) -> ProposeFn:
    """Adapt an existing OSIRIS provider (osiris_cli.providers) as the Developer's generator."""
    from osiris_cli.providers import ProviderRequest

    schema = {"type": "object", "required": ["patched_files", "report"],
              "properties": {"patched_files": {"type": "object"}, "report": {"type": "string"}}}

    def fn(f: ResearchFindings, repo_sandbox: str) -> DeveloperOutput:
        ctx = "\n\n".join(f"# {s.path} (line {s.line})\n{Path(repo_sandbox, s.path).read_text()}"
                          for s in f.sources)
        r = provider.generate(ProviderRequest(
            prompt=f"Failing tests: {f.failing_tests}\n\n{ctx}\n\nReturn the minimal corrected files.",
            system="You are the OSIRIS Developer agent. Return JSON only. Do not claim tests passed; "
                   "the runner measures that.", response_schema=schema, max_tokens=2048))
        so = r.structured_output if isinstance(r.structured_output, dict) else {}
        return DeveloperOutput(dict(so.get("patched_files") or {}), str(so.get("report") or ""))
    return fn


def _safe_rel(repo_sandbox: str, rel: str) -> Optional[str]:
    if os.path.isabs(rel) or ".." in Path(rel).parts:
        return None
    full = os.path.realpath(os.path.join(repo_sandbox, rel))
    root = os.path.realpath(repo_sandbox)
    return full if full.startswith(root + os.sep) else None


def apply_to_sandbox(repo_sandbox: str, out: DeveloperOutput) -> Tuple[Dict[str, str], List[str]]:
    """Write patched files into the sandbox. Returns (unified diffs by path, refusals)."""
    diffs, refused = {}, []
    for rel, new in out.patched_files.items():
        full = _safe_rel(repo_sandbox, rel)
        base = os.path.basename(rel)
        if full is None:
            refused.append(f"{rel}: path escapes the repo")
            continue
        if base.startswith("test_") or base.endswith("_test.py") or rel.split("/")[0] == "tests":
            refused.append(f"{rel}: the Developer may not edit or add tests")
            continue
        ok, why = GovernanceController.inspect_ast(new)
        if not ok:
            refused.append(f"{rel}: AST firewall: {why}")
            continue
        old = Path(full).read_text() if os.path.isfile(full) else ""
        Path(full).parent.mkdir(parents=True, exist_ok=True)
        Path(full).write_text(new)
        diffs[rel] = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                                  f"a/{rel}", f"b/{rel}"))
    return diffs, refused


# ── Critic ───────────────────────────────────────────────────────────────────

@dataclass
class CriticVerdict:
    approve: bool
    reasons: List[str]
    fail_to_pass: List[str]
    pass_to_fail: List[str]
    supplementary: str = ""   # optional model critique; never authoritative


def critic_review(repo: str, diffs: Dict[str, str], patched: Dict[str, str], refused: List[str],
                  f: ResearchFindings,
                  baseline_passed: Set[str], dev_report: str, timeout: float,
                  work_parent: str, supplementary_fn: Optional[Callable[[str], str]] = None
                  ) -> Tuple[CriticVerdict, ExecutionEvidence, FirewallResult]:
    """Independent of the Developer: fresh sandbox, fresh patch application from the diff-bound
    content, its own test run, its own firewall pass over the Developer's report."""
    reasons: List[str] = []
    if refused:
        reasons += [f"refused edit: {r}" for r in refused]
    if not diffs:
        reasons.append("empty patch")
    named = {s.path for s in f.sources}
    stray = sorted(set(diffs) - named)
    if stray:
        reasons.append(f"patch touches files research did not implicate: {stray}")
    if sum(d.count("\n+") + d.count("\n-") for d in diffs.values()) > 60:
        reasons.append("patch is larger than the minimal-change bound (60 changed lines)")

    sb = make_sandbox(repo, work_parent)
    for rel in diffs:  # exactly the files that survived the Developer-side checks
        full = _safe_rel(sb, rel)
        if full:
            Path(full).write_text(patched[rel])
    ev, passed, failed = run_pytest(sb, timeout)
    fail_to_pass = sorted(set(f.failing_tests) & passed)
    pass_to_fail = sorted(baseline_passed - passed)
    if not ev.green:
        reasons.append(f"candidate suite not green (rc={ev.returncode}, failed={sorted(failed)})")
    if pass_to_fail:
        reasons.append(f"regressions vs frozen baseline: {pass_to_fail}")
    if set(f.failing_tests) - set(fail_to_pass):
        reasons.append(f"baseline failures still failing: {sorted(set(f.failing_tests) - passed)}")

    fw = check_report(dev_report, ev)
    if fw.decision != "ACCEPT":
        reasons.append(f"developer report failed the evidence firewall ({fw.decision})")
    sup = ""
    if supplementary_fn:
        try:
            sup = str(supplementary_fn("\n".join(diffs.values())))[:2000]
        except Exception as e:  # supplementary only
            sup = f"[supplementary critic error: {e}]"
    return CriticVerdict(not reasons, reasons, fail_to_pass, pass_to_fail, sup), ev, fw


# ── Orchestrator ─────────────────────────────────────────────────────────────

@dataclass
class TrioResult:
    run_id: str
    outcome: str                  # REJECTED | QUARANTINED | PENDING_HUMAN_APPROVAL | NO_DEFECT | FAILED
    candidate_status: str         # UNMEASURED | MEASURED
    run_dir: str
    receipts: List[dict] = field(default_factory=list)
    critic: Optional[CriticVerdict] = None
    firewall: Optional[FirewallResult] = None
    baseline: Optional[ExecutionEvidence] = None
    candidate: Optional[ExecutionEvidence] = None
    reasons: List[str] = field(default_factory=list)


def _append_chained(path: str, kind: str, payload: dict) -> str:
    prev = "0" * 64
    if os.path.isfile(path):
        with open(path, "rb") as fh:
            lines = fh.read().splitlines()
        if lines:
            prev = json.loads(lines[-1])["hash"]
    body = {"prev": prev, "ts": time.time(), "kind": kind, "payload": payload}
    body["hash"] = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(body, sort_keys=True, default=str) + "\n")
    return body["hash"]


def verify_chain(path: str) -> bool:
    prev = "0" * 64
    for line in open(path, encoding="utf-8"):
        rec = json.loads(line)
        h = rec.pop("hash")
        if rec["prev"] != prev or hashlib.sha256(
                json.dumps(rec, sort_keys=True, default=str).encode()).hexdigest() != h:
            return False
        prev = h
    return True


def run_trio(repo: str, propose_fn: ProposeFn, out_dir: str, *, timeout: float = 60.0,
             budget: Optional[RunBudget] = None,
             supplementary_fn: Optional[Callable[[str], str]] = None) -> TrioResult:
    """Run the slice. ``repo`` is only read. ``out_dir`` receives the run record (and must not be
    inside ``repo``); the receipt chain there is its own file, not the live genome ledger."""
    repo, out_dir = os.path.realpath(repo), os.path.realpath(out_dir)
    if out_dir == repo or out_dir.startswith(repo + os.sep):
        raise ValueError("out_dir must be outside the repo under test")
    run_id = new_run_id()
    run_dir = os.path.join(out_dir, run_id)
    os.makedirs(run_dir)
    budget = budget or RunBudget()
    repo_before = tree_digest(repo)
    res = TrioResult(run_id, "FAILED", "UNMEASURED", run_dir)
    runs: List[AgentRun] = []
    work = tempfile.mkdtemp(prefix="osiris_trio_")
    chain = os.path.join(out_dir, "receipts.jsonl")
    try:
        # frozen baseline
        base_sb = make_sandbox(repo, work)
        res.baseline, base_pass, base_fail = run_pytest(base_sb, timeout)
        if res.baseline.green:
            res.outcome, res.reasons = "NO_DEFECT", ["baseline suite is already green"]
            return res

        r = AgentRun(RESEARCH, run_id, budget)

        runs.append(r)
        r.start({"repo_digest": repo_before})
        findings = research(base_sb, res.baseline, base_fail)
        r.finish(asdict(findings))
        if not findings.sources:
            res.outcome, res.reasons = "FAILED", ["research found no implicated source file"]
            return res

        d = AgentRun(DEVELOPER, run_id, budget)

        runs.append(d)
        d.start(asdict(findings))
        dev_sb = make_sandbox(repo, work)
        out = propose_fn(findings, dev_sb)
        diffs, refused = apply_to_sandbox(dev_sb, out)
        d.finish({"files": sorted(out.patched_files), "report": out.report})
        for rel, df in diffs.items():
            Path(run_dir, "candidate_" + rel.replace("/", "__") + ".diff").write_text(df)

        c = AgentRun(CRITIC, run_id, budget)

        runs.append(c)
        c.start({"diffs": digest(diffs), "refused": refused})
        verdict, res.candidate, fw = critic_review(
            repo, diffs, out.patched_files, refused, findings, base_pass, out.report, timeout, work, supplementary_fn)
        res.critic, res.firewall, res.candidate_status = verdict, fw, "MEASURED"
        c.finish({"approve": verdict.approve, "reasons": verdict.reasons})

        if fw.decision == "REJECT":
            res.outcome = "REJECTED"
        elif fw.decision == "QUARANTINE":
            res.outcome = "QUARANTINED"
        elif not verdict.approve:
            res.outcome = "REJECTED"
        else:
            res.outcome = "PENDING_HUMAN_APPROVAL"
        res.reasons = list(verdict.reasons) + [f"{x.verdict}: {x.reason} <- {x.sentence!r}"
                                               for x in fw.findings if x.verdict != "SUPPORTED"]
        if res.outcome == "REJECTED":
            d.to(AgentState.REJECTED, "; ".join(res.reasons)[:300])
        elif res.outcome == "QUARANTINED":
            d.to(AgentState.QUARANTINED, "; ".join(res.reasons)[:300])
    except BudgetExceeded as e:
        res.outcome, res.reasons = "FAILED", [f"budget: {e}"]
    finally:
        repo_after = tree_digest(repo)
        if repo_after != repo_before:           # would be a bug in this module; fail loudly
            res.outcome, res.reasons = "FAILED", res.reasons + ["REPO MODIFIED DURING RUN"]
        res.receipts = [x.receipt().to_dict() for x in runs]
        record = {"run_id": run_id, "outcome": res.outcome, "candidate_status": res.candidate_status,
                  "repo_digest_before": repo_before, "repo_digest_after": repo_after,
                  "baseline": asdict(res.baseline) if res.baseline else None,
                  "candidate": asdict(res.candidate) if res.candidate else None,
                  "reasons": res.reasons, "agent_receipts": res.receipts,
                  "approval": "PENDING" if res.outcome == "PENDING_HUMAN_APPROVAL" else "NOT_APPLICABLE"}
        Path(run_dir, "record.json").write_text(json.dumps(record, indent=2, sort_keys=True, default=str))
        _append_chained(chain, "trio_run", {"run_id": run_id, "outcome": res.outcome,
                                            "record_digest": digest(record)})
        shutil.rmtree(work, ignore_errors=True)
    return res
