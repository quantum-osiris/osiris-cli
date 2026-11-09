"""Turn a finished Research -> Developer -> Critic run into a governed evolutionary candidate.

What this adds over ``trio.run_trio``: a candidate *identity*, a measured-fitness receipt pair
(frozen baseline = champion, candidate), the existing ``PromotionPolicy``, an explicit
authorization step, and an append-only lineage record for EVERY outcome (rejected and
quarantined candidates keep their reasons and evidence references).

Hard rules, each enforced in code and covered by a test:
  * An agent's recommendation (Critic ``approve``) is necessary, never sufficient.
  * Promotion needs: measured candidate + firewall ACCEPT + Critic approve + PromotionPolicy
    pass + an authorizer that accepts this exact candidate digest. No authorizer -> at best
    VERIFIED_PENDING_AUTHORIZATION.
  * "Promoted" means a PROMOTED row in the lineage ledger. The patch is never applied to the repo.
  * The only fitness is ``frozen_suite_pass_rate`` = passed / (passed + failed + errors) as the
    runner measured it. It is a single deterministic measurement (re-run for reproducibility by
    the Critic); no uncertainty is estimated and none is claimed.
  * The lineage ledger path is explicit. Paths under ~/.osiris are refused unless allow_live=True.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from osiris_cli.dnalang.receipts import EvaluationReceipt, PromotionPolicy

from .agent_spec import digest
from .evidence_firewall import ExecutionEvidence
from .trio import TrioResult

SCOPE = "trio.candidate.promote"
METRIC = "frozen_suite_pass_rate"
EVALUATOR = ("osiris_cli.orchestration.trio.run_pytest", "1")

Authorizer = Callable[[str, str], Tuple[bool, str]]   # (candidate_digest, scope) -> (ok, detail)


def pass_rate(ev: ExecutionEvidence) -> Optional[float]:
    total = ev.passed + ev.failed + ev.errors
    return None if total == 0 else ev.passed / total


def candidate_digest(files: Dict[str, str]) -> str:
    return digest({p: hashlib.sha256(c.encode("utf-8")).hexdigest() for p, c in sorted(files.items())})


def grants_authorizer(grant_path: str, *, signature_path: Optional[str] = None,
                      allowed_signers: str, ledger_path: str, **kw: Any) -> Authorizer:
    """Adapt the existing signed, single-use SYS-021 grant (``osiris_cli.grants.authorize``).
    ledger_path and allowed_signers are required here so no caller falls into the live defaults."""
    from osiris_cli import grants

    def fn(cand_digest: str, scope: str) -> Tuple[bool, str]:
        d = grants.authorize(grant_path, cand_digest, scope, signature_path=signature_path,
                             allowed_signers=allowed_signers, ledger_path=ledger_path, **kw)
        return bool(d.authorized), d.reason
    return fn


@dataclass
class CandidateDecision:
    candidate_id: str
    candidate_digest: str
    status: str            # UNMEASURED | QUARANTINED | REJECTED | VERIFIED_PENDING_AUTHORIZATION | PROMOTED
    reasons: List[str]
    evidence_refs: Dict[str, Any]
    ledger_hash: str = ""
    baseline_fitness: Optional[float] = None
    candidate_fitness: Optional[float] = None
    receipts: Dict[str, Any] = field(default_factory=dict)


def _refuse_live(path: str, allow_live: bool) -> None:
    live = os.path.realpath(os.path.expanduser("~/.osiris"))
    p = os.path.realpath(path)
    if not allow_live and (p == live or p.startswith(live + os.sep)):
        raise ValueError(f"{path} is live OSIRIS state; pass allow_live=True to write there")


def _file_sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _evidence_refs(res: TrioResult) -> Dict[str, Any]:
    rec = os.path.join(res.run_dir, "record.json")
    cand = os.path.join(res.run_dir, "candidate.json")
    refs: Dict[str, Any] = {
        "run_id": res.run_id, "run_dir": res.run_dir,
        "record_sha256": _file_sha(rec) if os.path.isfile(rec) else None,
        "candidate_json_sha256": _file_sha(cand) if os.path.isfile(cand) else None,
        "baseline_stdout_sha256": res.baseline.stdout_digest if res.baseline else None,
        "candidate_stdout_sha256": res.candidate.stdout_digest if res.candidate else None,
        "baseline_tree_sha256": res.baseline.workdir_digest if res.baseline else None,
        "agent_receipts": [{k: r[k] for k in ("agent_id", "version", "state", "input_digest",
                                              "output_digest")} for r in res.receipts],
    }
    return refs


def _criticism(res: TrioResult) -> Dict[str, Any]:
    c, fw = res.critic, res.firewall
    return {
        "approve": c.approve if c else None, "reasons": list(c.reasons) if c else [],
        "fail_to_pass": list(c.fail_to_pass) if c else [],
        "pass_to_fail": list(c.pass_to_fail) if c else [],
        "supplementary_is_non_authoritative": c.supplementary if c else "",
        "firewall_decision": fw.decision if fw else None,
        "firewall_findings": [[f.verdict, f.kind, f.reason, f.sentence] for f in (fw.findings if fw else [])],
        "run_reasons": list(res.reasons),
    }


def evaluate_candidate(res: TrioResult, lineage_ledger: str, *,
                       authorizer: Optional[Authorizer] = None,
                       policy: Optional[PromotionPolicy] = None,
                       allow_live: bool = False) -> CandidateDecision:
    """Decide and record. Always appends exactly one lineage row, whatever the outcome."""
    _refuse_live(lineage_ledger, allow_live)
    import genome_ledger

    files = res.proposed_files if res.outcome in ("REJECTED", "QUARANTINED") else (
        res.patched_files or res.proposed_files)
    cdigest = candidate_digest(files)
    cid = f"cand_{cdigest[:12]}"
    refs = _evidence_refs(res)
    dec = CandidateDecision(cid, cdigest, "UNMEASURED", [], refs)
    kind = "CANDIDATE_UNMEASURED"

    measured = (res.candidate is not None and res.candidate_status == "MEASURED"
                and res.baseline is not None and res.critic is not None and res.firewall is not None)
    if not measured:
        dec.reasons = list(res.reasons) or ["candidate was never measured by the runner"]
    elif res.outcome == "QUARANTINED" or res.firewall.decision == "QUARANTINE":
        dec.status, kind = "QUARANTINED", "CANDIDATE_QUARANTINED"
        dec.reasons = list(res.reasons)
    elif res.outcome != "PENDING_HUMAN_APPROVAL" or res.firewall.decision != "ACCEPT" \
            or not res.critic.approve or not res.candidate.green:
        dec.status, kind = "REJECTED", "CANDIDATE_REJECTED"
        dec.reasons = list(res.reasons) or ["failed a verification precondition"]
    else:
        base_rate, cand_rate = pass_rate(res.baseline), pass_rate(res.candidate)
        dec.baseline_fitness, dec.candidate_fitness = base_rate, cand_rate
        common = dict(evaluator_id=EVALUATOR[0], evaluator_version=EVALUATOR[1], metric_name=METRIC,
                      uncertainty=0.0, parent_id=None)
        champ = EvaluationReceipt(
            candidate_id="baseline:" + (res.baseline.workdir_digest or "")[:12],
            candidate_digest=res.baseline.workdir_digest, fitness_score=base_rate or 0.0, passed=False,
            input_artifacts_hash=res.baseline.stdout_digest,
            details={"role": "frozen_baseline", "uncertainty_estimated": False}, **common)
        cand = EvaluationReceipt(
            candidate_id=cid, candidate_digest=cdigest, fitness_score=cand_rate or 0.0, passed=True,
            input_artifacts_hash=res.candidate.stdout_digest,
            details={"role": "candidate", "uncertainty_estimated": False,
                     "reproduced_twice": True}, **{**common, "parent_id": champ.candidate_id})
        dec.receipts = {"baseline": champ.to_dict(), "candidate": cand.to_dict()}
        ok, why = (policy or PromotionPolicy()).evaluate_promotion(cand, champ, cdigest)
        why = why.replace("PROMOTED:", "PromotionPolicy passed:", 1) if ok else why
        if not ok:
            dec.status, kind, dec.reasons = "REJECTED", "CANDIDATE_REJECTED", [why]
        elif authorizer is None:
            dec.status, kind = "VERIFIED_PENDING_AUTHORIZATION", "CANDIDATE_VERIFIED_PENDING_AUTHORIZATION"
            dec.reasons = [why, "no explicit authorization presented"]
        else:
            allowed, detail = authorizer(cdigest, SCOPE)
            if not allowed:
                dec.status, kind = "VERIFIED_PENDING_AUTHORIZATION", "CANDIDATE_VERIFIED_PENDING_AUTHORIZATION"
                dec.reasons = [why, f"authorization refused: {detail}"]
            else:
                dec.status, kind = "PROMOTED", "CANDIDATE_PROMOTED"
                dec.reasons = [why, f"authorized: {detail}"]

    payload = {"candidate_id": cid, "candidate_digest": cdigest, "status": dec.status,
               "reasons": dec.reasons, "criticism": _criticism(res), "evidence_refs": refs,
               "baseline_fitness": dec.baseline_fitness, "candidate_fitness": dec.candidate_fitness,
               "metric": METRIC, "receipts": dec.receipts,
               "applied_to_repo": False}
    dec.ledger_hash = genome_ledger.GenomeLedger(lineage_ledger).append(kind, payload)["hash"]
    return dec


def lineage(path: str) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in open(path, encoding="utf-8")] if os.path.isfile(path) else []
