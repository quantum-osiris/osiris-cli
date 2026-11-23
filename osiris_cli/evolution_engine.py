"""OSIRIS Evolution Engine: closed-loop 6-dimension telemetry & reproducible improvement generator.

Continuously monitors OSIRIS across six operational dimensions using existing
append-only ``exec-canon-1`` ledgers (read-only by default):

  1. ``planning``             DAG / Plan-v1 utilization & G-10..G-12 gate compliance
  2. ``reasoning``            Per-model Brier, ECE & calibration-adjusted utility routing
  3. ``memory``               Context truncation rate & epistemic memory stratification
  4. ``tool_use``             Subprocess/timeout failure clusters & precondition-hashed patching
  5. ``autonomy``             Hash-chain health, earned autonomy tier & decoupled subsystem promotion
  6. ``knowledge_retention``  Claim-provenance grounding ratio & SYS-024 training-corpus filter

Whenever a metric breaches its registered threshold, ``generate_improvements()`` emits a
deterministic, hash-addressed ``ImprovementSpec`` structured into the four mandatory
reproducible-research sections:

  1. ``rationale``                Measured baseline telemetry, breached threshold & root cause
  2. ``architecture``             Target module boundaries, schemas & SYS-021/024/025 invariants
  3. ``implementation_approach``  Validated ``plan_v1`` artifact (``plan_gates.validate_plan_v1``)
  4. ``examples``                 Runnable Python / CLI invocation & deterministic assertions

Persisted cycle receipts use ``ExecJournal`` (``exec-canon-1``, ``0600``, ``flock`` + ``fsync``)
with fixed-precision 6-decimal strings so floats never enter canonical hash chains.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from . import calibration, claim_provenance, claims, failure_clustering, plan_gates, sandbox
from .exec_kernel import ExecJournal, canon
from .journal import OperationalMerkleLedger
from .promotion_gate import Predicate, PredicateResult, PromotionCriteria, PromotionEvidence

SCHEMA = "osiris.evolution_cycle/1"
DIMENSIONS: Tuple[str, ...] = (
    "planning",
    "reasoning",
    "memory",
    "tool_use",
    "autonomy",
    "knowledge_retention",
    "research_workflows",
    "agent_collaboration",
    "experiment_management",
)

PathArg = Optional[Union[os.PathLike, str]]

_REFUTATION_CONTEXT_RE = re.compile(
    r"\b(refut(?:e|ed|es|ing|ation)|falsif(?:y|ied|ies|ication)|errat(?:a|um)|"
    r"withdrawn|retracted|artifact|quarantin(?:e|ed)|null\s+result|not\s+supported|"
    r"unsupported|disproved|invalid)\b",
    re.IGNORECASE,
)


def default_osiris_home(osiris_home: PathArg = None) -> Path:
    if osiris_home is not None:
        return Path(osiris_home).expanduser().resolve()
    env = os.environ.get("OSIRIS_HOME", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".osiris").resolve()


def default_evolution_ledger_path(osiris_home: PathArg = None) -> Path:
    return default_osiris_home(osiris_home) / "evolution" / "evolution_ledger.jsonl"


def _dec(val: Union[Decimal, int, str]) -> Decimal:
    return Decimal(str(val))


def _fmt_ratio(num: int, den: int) -> str:
    if den <= 0:
        return calibration.format_decimal(0)
    return calibration.format_decimal(_dec(num) / _dec(den))


def _count_jsonl_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


@dataclass(frozen=True)
class ModelUtilityScore:
    model_id: str
    samples: int
    accuracy_str: str
    brier_str: str
    ece_str: str
    overconfidence_str: str
    utility_str: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy_str,
            "brier": self.brier_str,
            "ece": self.ece_str,
            "model_id": self.model_id,
            "overconfidence": self.overconfidence_str,
            "samples": self.samples,
            "utility": self.utility_str,
        }


@dataclass(frozen=True)
class DimensionStatus:
    dimension: str
    healthy: bool
    headline_metric_str: str
    threshold_str: str
    summary: str
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "headline_metric": self.headline_metric_str,
            "healthy": self.healthy,
            "metrics": self.metrics,
            "summary": self.summary,
            "threshold": self.threshold_str,
        }


@dataclass(frozen=True)
class EvolutionSnapshot:
    osiris_home: str
    captured_at_utc: str
    all_chains_intact: bool
    dimensions: Tuple[DimensionStatus, ...]
    model_rankings: Tuple[ModelUtilityScore, ...]

    def by_dimension(self, name: str) -> DimensionStatus:
        for d in self.dimensions:
            if d.dimension == name:
                return d
        raise KeyError(f"Unknown evolution dimension: {name}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "all_chains_intact": self.all_chains_intact,
            "captured_at_utc": self.captured_at_utc,
            "dimensions": [d.to_dict() for d in self.dimensions],
            "model_rankings": [m.to_dict() for m in self.model_rankings],
            "osiris_home": self.osiris_home,
            "schema": SCHEMA,
        }

    def digest(self) -> str:
        return hashlib.sha256(canon(self.to_dict()).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ImprovementSpec:
    proposal_id: str
    dimension: str
    title: str
    rationale: str
    architecture: Dict[str, Any]
    implementation_approach: Dict[str, Any]
    examples: Dict[str, str]
    plan_gate_passed: bool
    spec_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture,
            "dimension": self.dimension,
            "examples": self.examples,
            "implementation_approach": self.implementation_approach,
            "plan_gate_passed": self.plan_gate_passed,
            "proposal_id": self.proposal_id,
            "rationale": self.rationale,
            "spec_sha256": self.spec_sha256,
            "title": self.title,
        }


# ---------------------------------------------------------------------------
# Operational Primitives Used by the 6 Dimensions
# ---------------------------------------------------------------------------


def compute_model_utility(report: calibration.CalibrationReport) -> ModelUtilityScore:
    """Rank models by U(m) = accuracy - ECE - 0.5 * Brier (exact Decimal arithmetic)."""
    acc = _dec(report.mean_accuracy_str)
    ece = _dec(report.ece_str)
    brier = _dec(report.brier_score_str)
    utility = acc - ece - (_dec("0.5") * brier)
    return ModelUtilityScore(
        model_id=report.model_filter or "all",
        samples=report.num_samples,
        accuracy_str=report.mean_accuracy_str,
        brier_str=report.brier_score_str,
        ece_str=report.ece_str,
        overconfidence_str=report.overconfidence_str,
        utility_str=calibration.format_decimal(utility),
    )


def select_calibrated_model(
    candidate_models: Sequence[str],
    *,
    calibration_path: PathArg = None,
    min_samples: int = 3,
    max_ece: Decimal = Decimal("0.150000"),
) -> Tuple[str, Optional[ModelUtilityScore]]:
    """Select the highest-utility calibrated model, penalizing overconfident failures."""
    if not candidate_models:
        raise ValueError("candidate_models must be non-empty")
    cpath = (
        Path(calibration_path)
        if calibration_path is not None
        else calibration.default_calibration_path()
    )
    if not cpath.is_file():
        return candidate_models[0], None

    ledger = calibration.CalibrationLedger(cpath)
    obs = ledger.observations()
    scored: List[Tuple[Decimal, int, str, ModelUtilityScore]] = []
    for idx, model_id in enumerate(candidate_models):
        m_obs = [o for o in obs if o.model_id == model_id]
        if len(m_obs) < min_samples:
            continue
        rep = calibration.compute_calibration(m_obs, model_filter=model_id)
        score = compute_model_utility(rep)
        if _dec(score.ece_str) <= max_ece:
            scored.append((_dec(score.utility_str), -idx, model_id, score))

    if not scored:
        return candidate_models[0], None
    scored.sort(reverse=True)
    _, _, winner, best_score = scored[0]
    return winner, best_score


def stratify_memory_context(
    *,
    recent_turns: Sequence[Mapping[str, str]],
    grounded_facts: Sequence[str],
    quarantined_warnings: Sequence[str],
    max_tokens: int = 1200,
) -> Dict[str, Any]:
    """Build a token-bounded, epistemically stratified context window for 4B-7B models."""
    if max_tokens < 64:
        raise ValueError("max_tokens must be >= 64")

    def est_tok(txt: str) -> int:
        return max(1, (len(txt) + 3) // 4)

    guard_budget = max_tokens // 5
    fact_budget = (max_tokens * 2) // 5
    turn_budget = max_tokens - guard_budget - fact_budget

    kept_guards: List[str] = []
    used_guard = 0
    for g in quarantined_warnings:
        t = est_tok(g)
        if used_guard + t <= guard_budget:
            kept_guards.append(g)
            used_guard += t

    kept_facts: List[str] = []
    used_fact = 0
    for f in grounded_facts:
        t = est_tok(f)
        if used_fact + t <= fact_budget:
            kept_facts.append(f)
            used_fact += t

    kept_turns: List[Dict[str, str]] = []
    used_turn = 0
    for turn in reversed(list(recent_turns)):
        role = str(turn.get("role", "user"))
        content = str(turn.get("content", ""))
        t = est_tok(content)
        if used_turn + t <= turn_budget:
            kept_turns.append({"content": content, "role": role})
            used_turn += t
    kept_turns.reverse()

    total_used = used_guard + used_fact + used_turn
    return {
        "dropped_turns": len(recent_turns) - len(kept_turns),
        "episodic_turns": kept_turns,
        "grounded_semantic_facts": kept_facts,
        "max_tokens": max_tokens,
        "sys024_guardrails": kept_guards,
        "tokens_used": total_used,
    }


def apply_preconditioned_patch(
    path: Union[str, Path],
    *,
    expected_sha256: str,
    old_string: str,
    new_string: str,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Verify SHA-256 precondition & unique hunk match before atomic replacement."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Target file does not exist: {target}")
    if not old_string:
        raise ValueError("old_string must be non-empty")

    raw_bytes = target.read_bytes()
    actual_sha = hashlib.sha256(raw_bytes).hexdigest()
    if actual_sha != expected_sha256:
        raise ValueError(
            f"Precondition SHA-256 mismatch on {target.name}: "
            f"expected {expected_sha256[:12]}, got {actual_sha[:12]}"
        )

    text = raw_bytes.decode("utf-8")
    occurrences = text.count(old_string)
    if occurrences == 0:
        raise ValueError(f"Target hunk not found in {target.name}")
    if occurrences > 1:
        raise ValueError(f"Ambiguous hunk ({occurrences} matches) in {target.name}")

    updated = text.replace(old_string, new_string, 1)
    after_bytes = updated.encode("utf-8")
    after_sha = hashlib.sha256(after_bytes).hexdigest()

    if not dry_run:
        tmp = target.with_suffix(target.suffix + ".osiris_patch_tmp")
        tmp.write_bytes(after_bytes)
        os.replace(tmp, target)

    return {
        "after_sha256": after_sha,
        "applied": not dry_run,
        "before_sha256": actual_sha,
        "bytes_delta": len(after_bytes) - len(raw_bytes),
        "path": str(target),
    }


@dataclass(frozen=True)
class SubsystemPromotionVerdict:
    candidate_id: str
    subsystem_scope: str
    eligible: bool
    results: Tuple[PredicateResult, ...]
    speaking_gate_required: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "eligible": self.eligible,
            "results": [
                {"detail": r.detail, "passed": r.passed, "predicate": r.predicate.value}
                for r in self.results
            ],
            "speaking_gate_required": self.speaking_gate_required,
            "subsystem_scope": self.subsystem_scope,
        }


def evaluate_subsystem_promotion(
    evidence: PromotionEvidence,
    criteria: PromotionCriteria,
    journal: OperationalMerkleLedger,
    *,
    subsystem_scope: str = "engineering",
) -> SubsystemPromotionVerdict:
    """Evaluate candidate eligibility without deadlocking non-NCLM code/routing upgrades on BPB."""
    from .promotion_gate import evaluate_promotion

    base_verdict = evaluate_promotion(evidence, criteria, journal)
    requires_speaking = subsystem_scope == "nclm_core_voice"
    filtered: List[PredicateResult] = []
    for res in base_verdict.results:
        if res.predicate == Predicate.SPEAKING and not requires_speaking:
            continue
        filtered.append(res)
    eligible = all(r.passed for r in filtered)
    return SubsystemPromotionVerdict(
        candidate_id=evidence.candidate_id,
        subsystem_scope=subsystem_scope,
        eligible=eligible,
        results=tuple(filtered),
        speaking_gate_required=requires_speaking,
    )


_BLOCKED_TRAINING_VERDICTS = frozenset(
    {"REFUTED", "ARTIFACT", "NO_EVIDENCE", "QUARANTINED", "NOT_MEASURED"}
)


def filter_training_doc_sys024(text: str) -> Tuple[bool, Tuple[str, ...]]:
    """Block un-refuted SYS-024 REFUTED/ARTIFACT claims from entering continual training corpora."""
    matched = [c for c in claims.match(text) if c.verdict in _BLOCKED_TRAINING_VERDICTS]
    if not matched:
        return True, ()

    unnegated_ids: List[str] = []
    for clm in matched:
        # Check whether each matching sentence explicitly frames the claim as refuted/erratum
        contaminated = False
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
            if any(re.search(pat, sentence, re.IGNORECASE) for pat in clm.patterns):
                if not _REFUTATION_CONTEXT_RE.search(sentence):
                    contaminated = True
                    break
        if contaminated:
            unnegated_ids.append(clm.id)

    if unnegated_ids:
        return False, tuple(sorted(set(unnegated_ids)))
    return True, ()


def compile_and_audit_agent_plan(
    intent: str,
    target_files: List[str],
    pytest_targets: Optional[List[str]] = None,
    record_receipt: bool = False,
    plan_ledger_path: Optional[str] = None,
    *,
    root: PathArg = None,
) -> Dict[str, Any]:
    """Build a canonical ``plan_v1`` artifact for ``/code`` agent tasks, audit G-10..G-12,
    and optionally persist an ``exec-canon-1`` receipt via ``plan_gates.record_plan_receipt()``."""
    steps: List[Dict[str, Any]] = []
    for idx, rel_file in enumerate(target_files):
        steps.append(
            {
                "id": f"step-patch-{idx + 1}",
                "type": "apply_patch",
                "description": f"Apply preconditioned change to {rel_file}",
                "path": str(rel_file),
            }
        )
    if pytest_targets is not None or not steps:
        steps.append(plan_gates.default_pytest_verify_step(pytest_targets or ()))

    plan = plan_gates.build_plan_v1(
        intent=intent,
        steps=steps,
        domain="code",
        execution_level="modify" if target_files else "read",
        execution_mode="dry_run",
        confirmed=False,
        preconditions=["ledger_chains_intact", "clean_git_worktree"],
        postconditions=["pytest_zero_failures", "zero_sys024_violations"],
        root=root,
    )
    receipt_hash: Optional[str] = None
    resolved_ledger: Optional[str] = None
    if record_receipt:
        target_ledger = (
            Path(plan_ledger_path).expanduser().resolve()
            if plan_ledger_path is not None
            else plan_gates.default_plan_receipts_path()
        )
        audit, receipt_hash = plan_gates.record_plan_receipt(
            plan,
            ledger_path=target_ledger,
            root=root,
        )
        resolved_ledger = str(target_ledger)
    else:
        audit = plan_gates.validate_plan_v1(plan, root=root)

    return {
        "errors": list(audit.errors),
        "gates": audit.gates,
        "gates_passed": audit.gates_passed,
        "ledger_path": resolved_ledger,
        "plan": plan,
        "plan_id": str(plan["meta"]["plan_id"]),
        "plan_sha256": audit.plan_sha256,
        "receipt_hash": receipt_hash,
        "valid": audit.valid,
    }


# ---------------------------------------------------------------------------
# 6-Dimension Telemetry Monitor & 4-Step Improvement Synthesis
# ---------------------------------------------------------------------------


def monitor_evolution(*, osiris_home: PathArg = None) -> EvolutionSnapshot:
    """Inspect all six operational dimensions strictly in read-only mode."""
    home = default_osiris_home(osiris_home)
    ts_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # 1. Failure clusters & ledger chain health
    fc_report = failure_clustering.analyze_failure_clusters(osiris_home=home)

    # 2. Planning telemetry
    plan_rows = _count_jsonl_rows(plan_gates.default_plan_receipts_path(home))
    task_run_rows = _count_jsonl_rows(home / "tasks" / "task_runs.jsonl")
    task_receipt_rows = _count_jsonl_rows(home / "task_receipts.jsonl")
    exec_rows = _count_jsonl_rows(home / "exec_journal.jsonl")
    structured_runs = plan_rows + task_run_rows + task_receipt_rows
    denom_plan = max(1, structured_runs + exec_rows)
    dag_cov_str = _fmt_ratio(structured_runs, denom_plan)
    planning_ok = structured_runs > 0 and _dec(dag_cov_str) >= _dec("0.200000")
    planning_status = DimensionStatus(
        dimension="planning",
        healthy=planning_ok,
        headline_metric_str=dag_cov_str,
        threshold_str="0.200000",
        summary=(
            f"structured_plan_or_dag_rows={structured_runs} vs raw_exec_rows={exec_rows} "
            f"(coverage={dag_cov_str})"
        ),
        metrics={
            "dag_coverage": dag_cov_str,
            "exec_journal_rows": exec_rows,
            "plan_receipt_rows": plan_rows,
            "task_receipt_rows": task_receipt_rows,
            "task_run_rows": task_run_rows,
        },
    )

    # 3. Reasoning telemetry (Calibration & Model Rankings)
    cal_path = calibration.default_calibration_path(home)
    rankings: List[ModelUtilityScore] = []
    if cal_path.is_file():
        cal_ledger = calibration.CalibrationLedger(cal_path)
        obs = cal_ledger.observations()
        overall_cal = calibration.compute_calibration(obs)
        models = sorted({o.model_id for o in obs if o.model_id})
        for m in models:
            m_rep = calibration.compute_calibration(
                [o for o in obs if o.model_id == m], model_filter=m
            )
            rankings.append(compute_model_utility(m_rep))
        rankings.sort(key=lambda r: _dec(r.utility_str), reverse=True)
        ece_str = overall_cal.ece_str
        acc_str = overall_cal.mean_accuracy_str
        brier_str = overall_cal.brier_score_str
        n_cal = overall_cal.num_samples
    else:
        ece_str = "1.000000"
        acc_str = "0.000000"
        brier_str = "1.000000"
        n_cal = 0

    max_model_ece = max((_dec(r.ece_str) for r in rankings), default=_dec(ece_str))
    reasoning_ok = n_cal >= 10 and max_model_ece <= _dec("0.100000") and _dec(acc_str) >= _dec("0.500000")
    reasoning_status = DimensionStatus(
        dimension="reasoning",
        healthy=reasoning_ok,
        headline_metric_str=calibration.format_decimal(max_model_ece),
        threshold_str="0.100000",
        summary=(
            f"n={n_cal} overall_acc={acc_str} overall_ece={ece_str} "
            f"worst_model_ece={calibration.format_decimal(max_model_ece)}"
        ),
        metrics={
            "brier_score": brier_str,
            "mean_accuracy": acc_str,
            "overall_ece": ece_str,
            "scored_predictions": n_cal,
            "worst_model_ece": calibration.format_decimal(max_model_ece),
        },
    )

    # 4. Memory telemetry (Exchanges & Incomplete generation / truncation)
    exchanges_path = home / "living" / "exchanges.jsonl"
    exchange_rows = _count_jsonl_rows(exchanges_path)
    incomplete_count = sum(
        c.count for c in fc_report.clusters if c.error_class == "IncompleteReply"
    )
    trunc_rate_str = _fmt_ratio(incomplete_count, max(1, exchange_rows))
    memory_ok = incomplete_count == 0 and exchange_rows > 0
    memory_status = DimensionStatus(
        dimension="memory",
        healthy=memory_ok,
        headline_metric_str=trunc_rate_str,
        threshold_str="0.000000",
        summary=(
            f"exchanges={exchange_rows} incomplete_replies={incomplete_count} "
            f"truncation_rate={trunc_rate_str}"
        ),
        metrics={
            "exchange_rows": exchange_rows,
            "incomplete_reply_count": incomplete_count,
            "truncation_rate": trunc_rate_str,
        },
    )

    # 5. Tool Use telemetry (Subprocess failures, timeouts & bwrap sandbox)
    subproc_failures = sum(
        c.count
        for c in fc_report.clusters
        if c.error_class in ("SubprocessExitFailure", "ExecTimeout", "PolicyError", "Denied")
    )
    fail_rate_str = _fmt_ratio(subproc_failures, max(1, exec_rows))
    bwrap_ok = bool(sandbox.available())
    tool_ok = subproc_failures == 0 and bwrap_ok
    tool_status = DimensionStatus(
        dimension="tool_use",
        healthy=tool_ok,
        headline_metric_str=fail_rate_str,
        threshold_str="0.050000",
        summary=(
            f"tool_failures={subproc_failures}/{max(1, exec_rows)} ({fail_rate_str}) "
            f"bwrap_sandbox={bwrap_ok}"
        ),
        metrics={
            "bwrap_available": bwrap_ok,
            "failure_clusters": len(fc_report.clusters),
            "subprocess_and_policy_failures": subproc_failures,
            "tool_failure_rate": fail_rate_str,
        },
    )

    # 6. Autonomy telemetry (Ledger chain integrity & promotion coupling check)
    autonomy_tier = "A1_GOVERNED_ASSIST" if fc_report.all_chains_intact else "A0_QUARANTINE_HOLD"
    # Note: promotion_gate.py couples all promotions to Predicate.SPEAKING unless decoupled
    autonomy_ok = fc_report.all_chains_intact and task_run_rows > 0
    autonomy_metric = "1.000000" if autonomy_ok else "0.000000"
    autonomy_status = DimensionStatus(
        dimension="autonomy",
        healthy=autonomy_ok,
        headline_metric_str=autonomy_metric,
        threshold_str="1.000000",
        summary=(
            f"tier={autonomy_tier} all_chains_intact={fc_report.all_chains_intact} "
            f"resumable_task_runs={task_run_rows}"
        ),
        metrics={
            "all_chains_intact": fc_report.all_chains_intact,
            "autonomy_tier": autonomy_tier,
            "resumable_task_runs": task_run_rows,
        },
    )

    # 7. Knowledge Retention telemetry (Claim provenance grounding vs refuted leakage)
    prov_path = claim_provenance.default_provenance_path(home)
    if prov_path.is_file():
        pledger = claim_provenance.ProvenanceLedger(prov_path)
        p_entries = pledger.entries()
        total_claims = len(p_entries)
        grounded_cnt = sum(
            1
            for e in p_entries
            if e.get("sys024_verdict") in ("GROUNDED", "SUPPORTED", "VERIFIED", "PILOT")
        )
        flagged_cnt = sum(
            1
            for e in p_entries
            if e.get("sys024_verdict") in ("REFUTED", "ARTIFACT", "NOT_MEASURED", "UNVERIFIED", "QUARANTINED")
        )
    else:
        total_claims = 0
        grounded_cnt = 0
        flagged_cnt = 0

    leakage_str = _fmt_ratio(flagged_cnt, max(1, total_claims))
    grounding_str = _fmt_ratio(grounded_cnt, max(1, total_claims))
    knowledge_ok = total_claims > 0 and _dec(leakage_str) <= _dec("0.050000")
    knowledge_status = DimensionStatus(
        dimension="knowledge_retention",
        healthy=knowledge_ok,
        headline_metric_str=leakage_str,
        threshold_str="0.050000",
        summary=(
            f"claims={total_claims} grounded_share={grounding_str} "
            f"flagged_or_unverified_share={leakage_str}"
        ),
        metrics={
            "flagged_or_unverified_claims": flagged_cnt,
            "flagged_share": leakage_str,
            "grounded_claims": grounded_cnt,
            "grounded_share": grounding_str,
            "total_claims": total_claims,
        },
    )

    # 8. Research Workflows, Agent Collaboration & Experiment Management dimensions
    doi_lineage_path = home / "evidence" / "doi_lineage.json"
    rw_ok = doi_lineage_path.is_file() and knowledge_ok
    rw_status = DimensionStatus(
        dimension="research_workflows",
        healthy=rw_ok,
        headline_metric_str="1.000000" if rw_ok else "0.000000",
        threshold_str="1.000000",
        summary=(
            f"doi_lineage_indexed={doi_lineage_path.is_file()} "
            f"claim_grounding_healthy={knowledge_ok}"
        ),
        metrics={
            "claim_grounding_healthy": knowledge_ok,
            "doi_lineage_indexed": doi_lineage_path.is_file(),
        },
    )

    collab_receipts = _count_jsonl_rows(home / "proposals" / "multi_candidate.jsonl")
    ac_ok = collab_receipts > 0 and task_receipt_rows > 0
    ac_status = DimensionStatus(
        dimension="agent_collaboration",
        healthy=ac_ok,
        headline_metric_str="1.000000" if ac_ok else "0.000000",
        threshold_str="1.000000",
        summary=(
            f"multi_candidate_receipts={collab_receipts} "
            f"typed_task_receipts={task_receipt_rows}"
        ),
        metrics={
            "multi_candidate_receipts": collab_receipts,
            "typed_task_receipts": task_receipt_rows,
        },
    )

    nexus_rows = _count_jsonl_rows(home / "nexus_ledger.jsonl")
    capsules_rows = _count_jsonl_rows(home / "evidence" / "reproducibility_capsules.jsonl")
    em_total = nexus_rows + capsules_rows
    em_ok = em_total > 0 and fc_report.all_chains_intact
    em_status = DimensionStatus(
        dimension="experiment_management",
        healthy=em_ok,
        headline_metric_str="1.000000" if em_ok else "0.000000",
        threshold_str="1.000000",
        summary=(
            f"registered_capsules_or_nexus_rows={em_total} "
            f"all_chains_intact={fc_report.all_chains_intact}"
        ),
        metrics={
            "all_chains_intact": fc_report.all_chains_intact,
            "nexus_ledger_rows": nexus_rows,
            "reproducibility_capsule_rows": capsules_rows,
        },
    )

    return EvolutionSnapshot(
        osiris_home=str(home),
        captured_at_utc=ts_utc,
        all_chains_intact=fc_report.all_chains_intact,
        dimensions=(
            planning_status,
            reasoning_status,
            memory_status,
            tool_status,
            autonomy_status,
            knowledge_status,
            rw_status,
            ac_status,
            em_status,
        ),
        model_rankings=tuple(rankings),
    )


def _make_spec(
    *,
    proposal_id: str,
    dimension: str,
    title: str,
    rationale: str,
    architecture: Dict[str, Any],
    plan_intent: str,
    plan_steps: Sequence[Mapping[str, Any]],
    examples: Dict[str, str],
    root: PathArg = None,
) -> ImprovementSpec:
    plan_doc = plan_gates.build_plan_v1(
        intent=plan_intent,
        steps=plan_steps,
        domain="code",
        execution_level="modify",
        execution_mode="dry_run",
        confirmed=False,
        preconditions=["ledger_chains_intact", "clean_git_worktree"],
        postconditions=["pytest_zero_failures", "zero_sys024_violations"],
        root=root,
    )
    audit = plan_gates.validate_plan_v1(plan_doc, root=root)
    gate_ok = bool(audit.valid and audit.gates_passed)
    body = {
        "architecture": architecture,
        "dimension": dimension,
        "examples": examples,
        "implementation_approach": plan_doc,
        "plan_gate_passed": gate_ok,
        "proposal_id": proposal_id,
        "rationale": rationale,
        "title": title,
    }
    sha = hashlib.sha256(canon(body).encode("utf-8")).hexdigest()
    return ImprovementSpec(
        proposal_id=proposal_id,
        dimension=dimension,
        title=title,
        rationale=rationale,
        architecture=architecture,
        implementation_approach=plan_doc,
        examples=examples,
        plan_gate_passed=gate_ok,
        spec_sha256=sha,
    )


def generate_improvements(
    snapshot: EvolutionSnapshot, *, root: PathArg = None
) -> Tuple[ImprovementSpec, ...]:
    """Generate 4-part reproducible improvement specs for every unhealthy dimension."""
    specs: List[ImprovementSpec] = []

    p = snapshot.by_dimension("planning")
    if not p.healthy:
        specs.append(
            _make_spec(
                proposal_id="EVO-PLAN-001",
                dimension="planning",
                title="Unify GapAnalyzer, PlanGate v1, and TaskGraphEngine before /code dispatch",
                rationale=(
                    f"Measured DAG/plan coverage is {p.headline_metric_str} (threshold >= {p.threshold_str}; "
                    f"summary: {p.summary}). Multi-step coding tasks bypass TaskGraphEngine and "
                    "CapabilityGapAnalyzer, executing as un-checkpointed flat loops."
                ),
                architecture={
                    "components": [
                        "osiris_cli.gap_analyzer.CapabilityGapAnalyzer",
                        "osiris_cli.plan_gates.validate_plan_v1",
                        "osiris_cli.task_graph.TaskGraphEngine",
                    ],
                    "data_flow": "UserIntent -> GapReport -> plan_v1.json (G-10..G-12) -> TaskGraph DAG -> TaskReceiptLedger",
                    "invariants": ["REQ-PLAN-1..3", "SYS-021 payload-bound action digest"],
                },
                plan_intent="Compile multi-step /code requests through PlanGate v1 into TaskGraphEngine",
                plan_steps=[
                    plan_gates.default_pytest_verify_step(["tests/test_evolution_engine.py"]),
                ],
                examples={
                    "cli": "python -m osiris_cli.evolution_engine --dimension planning",
                    "python": (
                        "from osiris_cli import plan_gates\n"
                        "plan = plan_gates.build_plan_v1(intent='Run unit suite', "
                        "steps=[plan_gates.default_pytest_verify_step(['tests/'])])\n"
                        "res = plan_gates.validate_plan_v1(plan)\n"
                        "assert res.valid and res.gates_passed"
                    ),
                },
                root=root,
            )
        )

    r = snapshot.by_dimension("reasoning")
    if not r.healthy:
        top_model = snapshot.model_rankings[0].model_id if snapshot.model_rankings else "unranked"
        specs.append(
            _make_spec(
                proposal_id="EVO-REASON-001",
                dimension="reasoning",
                title="Calibration-weighted model routing & automatic multi-candidate escalation",
                rationale=(
                    f"Worst-model ECE is {r.headline_metric_str} (threshold <= {r.threshold_str}; "
                    f"{r.summary}). Miscalibrated models emit high stated confidence on failing "
                    f"tasks while calibration.jsonl was write-only; top utility model is '{top_model}'."
                ),
                architecture={
                    "components": [
                        "osiris_cli.calibration.CalibrationLedger",
                        "osiris_cli.evolution_engine.select_calibrated_model",
                        "osiris_cli.multi_candidate.propose_three_candidates",
                        "osiris_cli.code_critic.critique",
                    ],
                    "data_flow": "calibration.jsonl -> U(m) = acc - ECE - 0.5*Brier -> route to argmax U(m); if U(m) < 0.25 escalate to 3-strategy multi_candidate + cross-model critic",
                    "invariants": ["Critic model != author model", "Zero float receipts (6-dp Decimal)"],
                },
                plan_intent="Route reasoning tasks by historical Brier/ECE utility and trigger multi-candidate critic",
                plan_steps=[
                    plan_gates.default_pytest_verify_step(["tests/test_evolution_engine.py"]),
                ],
                examples={
                    "cli": "python -m osiris_cli.calibration --summary",
                    "python": (
                        "from osiris_cli.evolution_engine import select_calibrated_model\n"
                        "winner, score = select_calibrated_model(['ollama/qwen2.5-coder:7b', 'ollama/gemma4:e4b'])\n"
                        "assert winner == 'ollama/gemma4:e4b'"
                    ),
                },
                root=root,
            )
        )

    m = snapshot.by_dimension("memory")
    if not m.healthy:
        specs.append(
            _make_spec(
                proposal_id="EVO-MEM-001",
                dimension="memory",
                title="Epistemically stratified, token-budgeted working memory window",
                rationale=(
                    f"Measured reply truncation rate is {m.headline_metric_str} ({m.summary}). "
                    "Unstratified chronological history crowds out output token headroom on local "
                    "4B-7B models and mixes unverified chat turns with verified facts."
                ),
                architecture={
                    "components": [
                        "osiris_cli.evolution_engine.stratify_memory_context",
                        "osiris_cli.living",
                        "osiris_cli.knowledge.Knowledge",
                    ],
                    "data_flow": "Reserve 20% SYS-024 guardrails + 40% BM25F/RRF grounded citations + 40% recent episodic turns within hard token ceiling",
                    "invariants": ["Byte-stable system prefix for KV cache reuse", "Explicit head/turn pruning"],
                },
                plan_intent="Enforce stratified token budgets across guardrails, grounded facts, and episodic turns",
                plan_steps=[
                    plan_gates.default_pytest_verify_step(["tests/test_evolution_engine.py"]),
                ],
                examples={
                    "cli": "python -m osiris_cli.evolution_engine --dimension memory",
                    "python": (
                        "from osiris_cli.evolution_engine import stratify_memory_context\n"
                        "ctx = stratify_memory_context(recent_turns=[{'role': 'user', 'content': 'check QEC-001'}], "
                        "grounded_facts=['OSIRIS-QEC-001: 393,216 shots on ibm_fez'], quarantined_warnings=[], max_tokens=256)\n"
                        "assert ctx['tokens_used'] <= 256"
                    ),
                },
                root=root,
            )
        )

    t = snapshot.by_dimension("tool_use")
    if not t.healthy:
        specs.append(
            _make_spec(
                proposal_id="EVO-TOOL-001",
                dimension="tool_use",
                title="Precondition-hashed surgical write_patch & bwrap-confined test runner",
                rationale=(
                    f"Observed tool/subprocess failure rate is {t.headline_metric_str} ({t.summary}). "
                    "Full-file write_file overwrites and ad-hoc interpreter invocations cause avoidable "
                    "SubprocessExitFailure and ExecTimeout clusters."
                ),
                architecture={
                    "components": [
                        "osiris_cli.evolution_engine.apply_preconditioned_patch",
                        "osiris_cli.exec_kernel.run",
                        "osiris_cli.sandbox",
                    ],
                    "data_flow": "Verify expected_sha256 + unique hunk -> stage atomic patch in isolated worktree -> run pytest inside bwrap sandbox -> commit or rollback",
                    "invariants": ["shell=False argv only", "Atomic os.replace with precondition digest"],
                },
                plan_intent="Add SHA-256 preconditioned hunk patching and isolated sandbox verification",
                plan_steps=[
                    plan_gates.default_pytest_verify_step(["tests/test_evolution_engine.py"]),
                ],
                examples={
                    "cli": "python -m osiris_cli.exec_kernel --explain -- pytest -q",
                    "python": (
                        "from osiris_cli.evolution_engine import apply_preconditioned_patch\n"
                        "res = apply_preconditioned_patch(p, expected_sha256=sha, old_string='a = 1', new_string='a = 2', dry_run=True)\n"
                        "assert res['applied'] is False and res['before_sha256'] != res['after_sha256']"
                    ),
                },
                root=root,
            )
        )

    a = snapshot.by_dimension("autonomy")
    if not a.healthy:
        specs.append(
            _make_spec(
                proposal_id="EVO-AUTO-001",
                dimension="autonomy",
                title="Decouple engineering subsystem promotion from NCLM byte-LM speaking gate",
                rationale=(
                    f"Autonomy readiness metric is {a.headline_metric_str} ({a.summary}). "
                    "promotion_gate.py previously required Predicate.SPEAKING (NCLM <= 2.0 BPB) for "
                    "all candidates, deadlocking verified routing/tool/skill upgrades while NCLM remains an experiment."
                ),
                architecture={
                    "components": [
                        "osiris_cli.evolution_engine.evaluate_subsystem_promotion",
                        "osiris_cli.promotion_gate.evaluate_promotion",
                        "osiris_cli.session_approval.SessionScope",
                    ],
                    "data_flow": "Scope=='nclm_core_voice' enforces all 4 predicates (incl. SPEAKING); Scope=='engineering' enforces QUALITY + STABILITY + INTEGRITY under operator sign-off",
                    "invariants": ["Zero self-promotion (operator approves)", "Full OperationalMerkleLedger verification"],
                },
                plan_intent="Scope Predicate.SPEAKING strictly to nclm_core_voice while gating engineering upgrades on QUALITY+STABILITY+INTEGRITY",
                plan_steps=[
                    plan_gates.default_pytest_verify_step(["tests/test_evolution_engine.py"]),
                ],
                examples={
                    "cli": "python -m osiris_cli.task_receipts --verify",
                    "python": (
                        "from osiris_cli.evolution_engine import evaluate_subsystem_promotion\n"
                        "verdict = evaluate_subsystem_promotion(ev, crit, journal, subsystem_scope='engineering')\n"
                        "assert verdict.speaking_gate_required is False"
                    ),
                },
                root=root,
            )
        )

    k = snapshot.by_dimension("knowledge_retention")
    if not k.healthy:
        specs.append(
            _make_spec(
                proposal_id="EVO-KNOW-001",
                dimension="knowledge_retention",
                title="Enforce SYS-024 refuted-claim quarantine filter on continual training corpora",
                rationale=(
                    f"Flagged/unverified claim share in provenance ledger is {k.headline_metric_str} "
                    f"(threshold <= {k.threshold_str}; {k.summary}). train.py previously sampled "
                    "archive_roots without screening out un-negated SYS-024 REFUTED/ARTIFACT claims."
                ),
                architecture={
                    "components": [
                        "osiris_cli.evolution_engine.filter_training_doc_sys024",
                        "osiris_cli.claims.flagged_in_reply",
                        "osiris_cli.claim_provenance.ProvenanceLedger",
                        "osiris_cli.train",
                    ],
                    "data_flow": "Candidate doc/exchange -> claims.flagged_in_reply -> block unless sentence carries explicit refutation/erratum framing -> freeze clean training manifest",
                    "invariants": ["SYS-024 quarantine preservation", "Errata & negative results remain learnable"],
                },
                plan_intent="Filter un-negated SYS-024 refuted claims out of train.py corpus construction while retaining explicit errata",
                plan_steps=[
                    plan_gates.default_pytest_verify_step(["tests/test_evolution_engine.py"]),
                ],
                examples={
                    "cli": "python -m osiris_cli.claim_provenance --unverified",
                    "python": (
                        "from osiris_cli.evolution_engine import filter_training_doc_sys024\n"
                        "ok_bad, ids = filter_training_doc_sys024('Universal lock at 51.843 degrees')\n"
                        "ok_err, _ = filter_training_doc_sys024('Erratum: 51.843 degrees was refuted on ibm_fez')\n"
                        "assert not ok_bad and ok_err"
                    ),
                },
                root=root,
            )
        )

    return tuple(specs)


def record_evolution_cycle(
    snapshot: EvolutionSnapshot,
    proposals: Sequence[ImprovementSpec],
    *,
    ledger_path: PathArg = None,
    osiris_home: PathArg = None,
) -> str:
    """Append one canonical float-free evolution receipt to evolution_ledger.jsonl."""
    path = (
        Path(ledger_path)
        if ledger_path is not None
        else default_evolution_ledger_path(osiris_home)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    journal = ExecJournal(path)
    body = {
        "all_chains_intact": snapshot.all_chains_intact,
        "captured_at_utc": snapshot.captured_at_utc,
        "healthy_count": sum(1 for d in snapshot.dimensions if d.healthy),
        "proposal_digests": [
            {"dimension": p.dimension, "proposal_id": p.proposal_id, "spec_sha256": p.spec_sha256}
            for p in proposals
        ],
        "schema": SCHEMA,
        "snapshot_sha256": snapshot.digest(),
        "total_dimensions": len(snapshot.dimensions),
    }
    return journal.append(body)


def render_markdown_report(
    snapshot: EvolutionSnapshot, proposals: Sequence[ImprovementSpec]
) -> str:
    lines = [
        "# OSIRIS Evolution Engine — 6-Dimension Telemetry & Improvement Report",
        "",
        f"- **OSIRIS Home:** `{snapshot.osiris_home}`",
        f"- **Captured UTC:** `{snapshot.captured_at_utc}`",
        f"- **Ledger Chains Intact:** `{snapshot.all_chains_intact}`",
        f"- **Snapshot SHA-256:** `{snapshot.digest()}`",
        "",
        "## Monitored Dimensions",
        "",
        "| Dimension | Status | Metric | Threshold | Summary |",
        "|---|---|---|---|---|",
    ]
    for d in snapshot.dimensions:
        badge = "PASS" if d.healthy else "NEEDS_IMPROVEMENT"
        lines.append(
            f"| `{d.dimension}` | **{badge}** | `{d.headline_metric_str}` | `{d.threshold_str}` | {d.summary} |"
        )

    if proposals:
        lines.extend(["", "## Generated Reproducible Improvements", ""])
        for p in proposals:
            lines.extend(
                [
                    f"### {p.proposal_id} (`{p.dimension}`): {p.title}",
                    f"- **Spec SHA-256:** `{p.spec_sha256}` · **Plan Gate v1 Passed:** `{p.plan_gate_passed}`",
                    f"1. **Rationale:** {p.rationale}",
                    f"2. **Architecture:** `{json.dumps(p.architecture, sort_keys=True)}`",
                    f"3. **Implementation Approach (`plan_v1`):** `plan_id={p.implementation_approach.get('plan_id')}` · `steps={len(p.implementation_approach.get('steps', []))}`",
                    f"4. **Example:**\n```python\n{p.examples['python']}\n```",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="OSIRIS 6-Dimension Evolution Engine")
    parser.add_argument("--osiris-home", default=None, help="Path to .osiris state directory")
    parser.add_argument("--dimension", choices=DIMENSIONS, default=None, help="Filter to one dimension")
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON")
    parser.add_argument("--markdown", action="store_true", help="Emit Markdown report")
    parser.add_argument("--record", action="store_true", help="Append receipt to evolution_ledger.jsonl")
    args = parser.parse_args(argv)

    snap = monitor_evolution(osiris_home=args.osiris_home)
    props = generate_improvements(snap)
    if args.dimension:
        props = tuple(p for p in props if p.dimension == args.dimension)

    receipt_hash: Optional[str] = None
    if args.record:
        receipt_hash = record_evolution_cycle(snap, props, osiris_home=args.osiris_home)

    if args.json:
        payload: Dict[str, Any] = {
            "proposals": [p.to_dict() for p in props],
            "snapshot": snap.to_dict(),
        }
        if receipt_hash:
            payload["receipt_hash"] = receipt_hash
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_markdown_report(snap, props), end="")
        if receipt_hash:
            print(f"\nRecorded evolution cycle receipt: {receipt_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
