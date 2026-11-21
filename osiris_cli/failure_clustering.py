"""L8 Ledger: read-only failure-pattern clustering across OSIRIS ledgers (REQ-EV-4).

Deterministically scans ``living/exchanges.jsonl``, ``genome_ledger.jsonl``,
``tasks/task_runs.jsonl``, ``exec_journal.jsonl``, and immutable run records
(``runs/*/run.json``, ``code/runs/*.json``) strictly in read-only mode (zero file
writes or repairs), verifies hash-chain integrity on every hash-chained ledger, and
clusters recurring failure signatures by ``(error_class, module, root_cause_tag)``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from . import probes

PathArg = Optional[Union[os.PathLike, str]]

_PY_EXC_RE = re.compile(
    r"\b(SyntaxError|IndentationError|AssertionError|ImportError|ModuleNotFoundError|"
    r"AttributeError|TypeError|ValueError|KeyError|IndexError|RuntimeError|TimeoutError|"
    r"PermissionError|FileNotFoundError|OSError|ConnectionError|PolicyError|Denied|"
    r"QuarantineError|TaskContextError|TaskLedgerError)\b"
)
_MODULE_PATH_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_/.-]*\.py)\b")
_VOLATILE_HEX_RE = re.compile(r"\b[0-9a-f]{12,64}\b", re.IGNORECASE)
_VOLATILE_NUM_RE = re.compile(r"\bline\s+\d+\b", re.IGNORECASE)


def default_osiris_home(osiris_home: PathArg = None) -> Path:
    raw = osiris_home or os.environ.get("OSIRIS_HOME")
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".osiris").resolve()


@dataclass(frozen=True)
class LedgerIntegrityStatus:
    name: str
    path: str
    exists: bool
    intact: bool
    status: str  # INTACT | EMPTY | TAMPERED | UNREADABLE
    entries_checked: int
    detail: str = ""


@dataclass(frozen=True)
class FailureCluster:
    cluster_id: str
    error_class: str
    module: str
    root_cause_tag: str
    count: int
    first_seen: str
    last_seen: str
    sources: Tuple[str, ...]
    sample_message: str


@dataclass(frozen=True)
class FailureClusterReport:
    osiris_home: str
    ledger_integrity: Tuple[LedgerIntegrityStatus, ...]
    all_chains_intact: bool
    total_failures: int
    clusters: Tuple[FailureCluster, ...]


@dataclass(frozen=True)
class _RawFailure:
    error_class: str
    module: str
    root_cause_tag: str
    ts: str
    source: str
    message: str


def _normalize_module(text: str, fallback: str) -> str:
    m = _MODULE_PATH_RE.search(text or "")
    if m:
        raw = m.group(1).replace("\\", "/")
        parts = [p for p in raw.split("/") if p and p not in ("tmp", "var", "home")]
        if len(parts) >= 2 and parts[-2] in ("osiris_cli", "orchestration", "tests", "providers"):
            return f"{parts[-2]}/{parts[-1]}"
        return parts[-1]
    return fallback


def _classify_message(msg: str, default_class: str, default_module: str, default_tag: str) -> Tuple[str, str, str]:
    text = (msg or "").strip()
    low = text.lower()
    exc_m = _PY_EXC_RE.search(text)
    err_cls = exc_m.group(1) if exc_m else default_class
    mod = _normalize_module(text, default_module)

    if "stopped_budget" in low or "budget" in low and ("cap" in low or "exceeded" in low or "exhaust" in low):
        tag = "budget_exhausted"
    elif "ast" in low and ("firewall" in low or "forbidden" in low or "disallowed" in low):
        tag = "ast_firewall_violation"
    elif "placeholder" in low or "stub function" in low:
        tag = "placeholder_stub_rejected"
    elif "phantom" in low or "grounding audit" in low or "unverified_number" in low:
        tag = "grounding_audit_failure"
    elif "not legit" in low or "flagged_claim" in low or "quarantined_constant" in low or "sys-024" in low:
        tag = "sys024_claim_violation"
    elif "timeout" in low or "timed out" in low or "horizon" in low:
        tag = "execution_timeout"
    elif "execution policy refus" in low or "policy_deny" in low or err_cls in ("PolicyError", "Denied"):
        tag = "exec_policy_denied"
    elif "grant" in low and ("invalid" in low or "expired" in low or "digest" in low or "refused" in low):
        tag = "grant_verification_failed"
    elif "quarantine" in low:
        tag = "candidate_quarantined"
    elif "skipped" in low and ("increased" in low or "pass_rate" in low or "0 collected" in low):
        tag = "test_skip_or_empty_suite"
    elif "unreachable" in low or "connection" in low or "ollama" in low and "fail" in low:
        tag = "provider_unreachable"
    elif err_cls == "SyntaxError":
        tag = "syntax_error"
    elif err_cls == "AssertionError":
        tag = "test_assertion_failure"
    elif err_cls in ("ImportError", "ModuleNotFoundError"):
        tag = "missing_dependency_or_import"
    else:
        tag = default_tag

    return err_cls, mod, tag


def _clean_sample(msg: str) -> str:
    collapsed = " ".join((msg or "").split())
    collapsed = _VOLATILE_HEX_RE.sub("<hex>", collapsed)
    collapsed = _VOLATILE_NUM_RE.sub("line <n>", collapsed)
    return collapsed[:160]


def _check_chained_ledger(name: str, path: Path, *, use_full_exchange_verify: bool = False) -> LedgerIntegrityStatus:
    if not path.exists():
        return LedgerIntegrityStatus(
            name=name,
            path=str(path),
            exists=False,
            intact=True,
            status="EMPTY",
            entries_checked=0,
            detail="not created yet",
        )
    if use_full_exchange_verify:
        res = probes.verify_chain(str(path))
        intact = bool(res.get("intact"))
        status = str(res.get("status", "TAMPERED"))
        entries = int(res.get("entries", 0) or 0)
        err = str(res.get("error") or "")
        tail = probes.verify_chain_tail(path, tail_lines=20)
        if intact and not tail.intact:
            intact = False
            status = "TAMPERED"
            err = f"tail check failed ({tail.status})"
        return LedgerIntegrityStatus(
            name=name,
            path=str(path),
            exists=True,
            intact=intact,
            status=status,
            entries_checked=entries,
            detail=err,
        )

    tail = probes.verify_chain_tail(path, tail_lines=50)
    return LedgerIntegrityStatus(
        name=name,
        path=str(path),
        exists=True,
        intact=bool(tail.intact),
        status=str(tail.status),
        entries_checked=int(tail.inspected_count),
        detail="" if tail.intact else f"tail verification status={tail.status}",
    )


def _scan_exchanges(path: Path) -> List[_RawFailure]:
    out: List[_RawFailure] = []
    if not path.is_file():
        return out
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except ValueError:
                    continue
                if not isinstance(row, dict):
                    continue
                ts = str(row.get("t", ""))
                ga = row.get("grounding_audit")
                if isinstance(ga, dict) and ga.get("passed") is False:
                    reasons: List[str] = []
                    if ga.get("phantom_files"):
                        reasons.append("phantom_files=" + ",".join(str(x) for x in ga["phantom_files"]))
                    if ga.get("unverified_numbers"):
                        reasons.append("unverified_numbers=" + ",".join(str(x) for x in ga["unverified_numbers"]))
                    if ga.get("contradictions"):
                        reasons.append("contradictions=" + ",".join(str(x) for x in ga["contradictions"]))
                    msg = "; ".join(reasons) or "grounding audit failed"
                    out.append(_RawFailure("GroundingAuditFailure", "living.py", "grounding_audit_failure", ts, "exchanges.jsonl", msg))
                flagged = row.get("flagged_claims")
                if isinstance(flagged, list) and flagged:
                    msg = "flagged claims in reply: " + ", ".join(str(x) for x in flagged)
                    out.append(_RawFailure("SYS024ClaimFlag", "claims.py", "sys024_claim_violation", ts, "exchanges.jsonl", msg))
                inc = row.get("incomplete")
                if inc:
                    out.append(_RawFailure("IncompleteReply", "living.py", "incomplete_generation", ts, "exchanges.jsonl", str(inc)))
                if row.get("interrupted") is True:
                    reply_txt = str(row.get("reply", ""))
                    cls, mod, tag = _classify_message(reply_txt, "InterruptedTurn", "living.py", "interrupted_turn")
                    out.append(_RawFailure(cls, mod, tag, ts, "exchanges.jsonl", reply_txt or "turn interrupted"))
    except OSError:
        return out
    return out


def _scan_genome_ledger(path: Path) -> List[_RawFailure]:
    out: List[_RawFailure] = []
    if not path.is_file():
        return out
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except ValueError:
                    continue
                if not isinstance(row, dict):
                    continue
                ts = str(row.get("ts") or row.get("timestamp") or row.get("t") or "")
                kind = str(row.get("kind") or row.get("event") or row.get("type") or "")
                payload = row.get("payload") if isinstance(row.get("payload"), dict) else row
                if isinstance(payload.get("payload"), dict):
                    payload = payload["payload"]
                status = str(payload.get("status") or payload.get("outcome") or "")
                if kind in ("CANDIDATE_REJECTED", "CANDIDATE_QUARANTINED", "CANDIDATE_UNMEASURED") or status in (
                    "REJECTED",
                    "QUARANTINED",
                    "UNMEASURED",
                    "FAILED",
                    "BLOCKED",
                    "STOPPED_BUDGET",
                ):
                    reasons = payload.get("reasons") or payload.get("error") or status or kind
                    msg = "; ".join(str(x) for x in reasons) if isinstance(reasons, list) else str(reasons)
                    default_cls = "CandidateQuarantined" if "QUARANTINE" in (kind + status) else "CandidateRejected"
                    default_tag = "candidate_quarantined" if "QUARANTINE" in (kind + status) else "candidate_rejected"
                    cls, mod, tag = _classify_message(msg, default_cls, "evolution_bridge.py", default_tag)
                    out.append(_RawFailure(cls, mod, tag, ts, "genome_ledger.jsonl", msg))
    except OSError:
        return out
    return out


def _scan_task_and_exec_ledgers(task_ledger: Path, exec_journal: Path) -> List[_RawFailure]:
    out: List[_RawFailure] = []
    if task_ledger.is_file():
        try:
            with open(task_ledger, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        row = json.loads(raw)
                    except ValueError:
                        continue
                    status = str(row.get("status") or "")
                    if status in ("STOPPED_BUDGET", "BLOCKED", "FAILED", "ERROR"):
                        ts = str(row.get("ts") or row.get("t") or "")
                        reason = str(row.get("stop_reason") or row.get("reason") or status)
                        cls, mod, tag = _classify_message(reason, status, "task_context.py", status.lower())
                        out.append(_RawFailure(cls, mod, tag, ts, "task_runs.jsonl", reason))
        except OSError:
            pass

    if exec_journal.is_file():
        try:
            with open(exec_journal, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        row = json.loads(raw)
                    except ValueError:
                        continue
                    body = row.get("body") if isinstance(row.get("body"), dict) else {}
                    decision = str(body.get("decision") or "")
                    exit_code = body.get("exit_code")
                    if decision in ("DENY", "TIMEOUT", "ERROR") or (
                        isinstance(exit_code, int) and exit_code not in (0,) and decision in ("ALLOW", "ASK_APPROVED")
                    ):
                        ts = str(body.get("ts") or body.get("started_ms") or "")
                        reasons = body.get("reasons") or ()
                        argv = body.get("argv") or ()
                        argv_head = str(argv[0]) if argv else "cmd"
                        msg = "; ".join(str(r) for r in reasons) or f"{argv_head} exit_code={exit_code}"
                        default_cls = "ExecPolicyDeny" if decision == "DENY" else (
                            "ExecTimeout" if decision == "TIMEOUT" else "SubprocessExitFailure"
                        )
                        cls, mod, tag = _classify_message(msg, default_cls, "exec_kernel.py", "command_failure")
                        out.append(_RawFailure(cls, mod, tag, ts, "exec_journal.jsonl", msg))
        except OSError:
            pass
    return out


def _scan_run_records(runs_dir: Path, code_runs_dir: Path) -> List[_RawFailure]:
    out: List[_RawFailure] = []
    if runs_dir.is_dir():
        for rec_file in sorted(runs_dir.glob("*/run.json")):
            try:
                data = json.loads(rec_file.read_text(encoding="utf-8", errors="replace"))
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            ts = str(data.get("started_at") or data.get("ts") or "")
            target = str(data.get("target") or "sprint_manager.py")
            attempts = data.get("attempts") or ()
            for att in attempts:
                if not isinstance(att, dict):
                    continue
                verdict = str(att.get("verdict") or "")
                if verdict.lower() in ("pass", "passed", "ok"):
                    continue
                stage = str(att.get("stage") or "verify")
                reason = str(att.get("reason") or verdict or stage)
                cls, mod, tag = _classify_message(reason, f"SprintAttempt_{stage}", _normalize_module(target, "sprint_manager.py"), stage.lower())
                out.append(_RawFailure(cls, mod, tag, ts, "runs/run.json", reason))

    if code_runs_dir.is_dir():
        for cfile in sorted(code_runs_dir.glob("*.json")):
            try:
                data = json.loads(cfile.read_text(encoding="utf-8", errors="replace"))
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            status = str(data.get("status") or "")
            if status in ("SUCCESS", "PASSED", "OK", ""):
                continue
            ts = str(data.get("finished_at") or data.get("started_at") or "")
            err = str(data.get("error") or data.get("stop_reason") or status)
            cls, mod, tag = _classify_message(err, f"CodeRun_{status}", "code_agent.py", status.lower())
            out.append(_RawFailure(cls, mod, tag, ts, "code/runs", err))
    return out


def analyze_failure_clusters(
    *,
    osiris_home: PathArg = None,
    exchanges_path: PathArg = None,
    genome_ledger_path: PathArg = None,
    runs_dir: PathArg = None,
    code_runs_dir: PathArg = None,
    task_ledger_path: PathArg = None,
    exec_journal_path: PathArg = None,
) -> FailureClusterReport:
    """Read-only multi-ledger integrity verification + deterministic failure clustering."""
    home = default_osiris_home(osiris_home)
    ex_p = Path(exchanges_path).expanduser().resolve() if exchanges_path is not None else (home / "living" / "exchanges.jsonl")
    gl_p = Path(genome_ledger_path).expanduser().resolve() if genome_ledger_path is not None else (home / "genome_ledger.jsonl")
    rn_d = Path(runs_dir).expanduser().resolve() if runs_dir is not None else (home / "runs")
    cr_d = Path(code_runs_dir).expanduser().resolve() if code_runs_dir is not None else (home / "code" / "runs")
    tl_p = Path(task_ledger_path).expanduser().resolve() if task_ledger_path is not None else (home / "tasks" / "task_runs.jsonl")
    ej_p = Path(exec_journal_path).expanduser().resolve() if exec_journal_path is not None else (home / "exec_journal.jsonl")

    integrity = (
        _check_chained_ledger("exchanges.jsonl", ex_p, use_full_exchange_verify=True),
        _check_chained_ledger("genome_ledger.jsonl", gl_p, use_full_exchange_verify=False),
        _check_chained_ledger("task_runs.jsonl", tl_p, use_full_exchange_verify=True),
        _check_chained_ledger("exec_journal.jsonl", ej_p, use_full_exchange_verify=False),
    )
    all_intact = all(st.intact for st in integrity)

    raw_failures: List[_RawFailure] = []
    raw_failures.extend(_scan_exchanges(ex_p))
    raw_failures.extend(_scan_genome_ledger(gl_p))
    raw_failures.extend(_scan_task_and_exec_ledgers(tl_p, ej_p))
    raw_failures.extend(_scan_run_records(rn_d, cr_d))

    grouped: Dict[Tuple[str, str, str], List[_RawFailure]] = {}
    for item in raw_failures:
        sig = (item.error_class, item.module, item.root_cause_tag)
        grouped.setdefault(sig, []).append(item)

    clusters: List[FailureCluster] = []
    for (err_cls, mod, tag), items in grouped.items():
        cid = "fcl-" + hashlib.sha256(f"{err_cls}|{mod}|{tag}".encode("utf-8")).hexdigest()[:10]
        timestamps = sorted(it.ts for it in items if it.ts)
        first_seen = timestamps[0] if timestamps else ""
        last_seen = timestamps[-1] if timestamps else ""
        sources = tuple(sorted({it.source for it in items}))
        sample = _clean_sample(items[-1].message)
        clusters.append(
            FailureCluster(
                cluster_id=cid,
                error_class=err_cls,
                module=mod,
                root_cause_tag=tag,
                count=len(items),
                first_seen=first_seen,
                last_seen=last_seen,
                sources=sources,
                sample_message=sample,
            )
        )

    clusters.sort(key=lambda c: (-c.count, c.last_seen, c.error_class, c.module, c.root_cause_tag))
    # Place highest count first, and among ties newest last_seen first.
    clusters.sort(key=lambda c: c.last_seen, reverse=True)
    clusters.sort(key=lambda c: c.count, reverse=True)

    return FailureClusterReport(
        osiris_home=str(home),
        ledger_integrity=integrity,
        all_chains_intact=all_intact,
        total_failures=len(raw_failures),
        clusters=tuple(clusters),
    )


def format_clusters(report: FailureClusterReport, *, limit: int = 15, width: int = 92) -> str:
    """Render compact (<100 col) ledger integrity + recurring failure clusters."""
    cap = max(48, min(width, 96))
    chain_summary = ", ".join(f"{st.name}={st.status}" for st in report.ledger_integrity)
    verdict = "INTACT" if report.all_chains_intact else "TAMPERED"
    lines = [
        f"L8 FAILURE CLUSTERS · chains {verdict} · {report.total_failures} failure event(s) -> "
        f"{len(report.clusters)} cluster(s)"[:cap],
        f"  integrity: {chain_summary}"[:cap],
    ]
    for st in report.ledger_integrity:
        if not st.intact and st.detail:
            lines.append(f"  ! {st.name}: {st.detail}"[:cap])
    if not report.clusters:
        lines.append("  no recurring failure patterns recorded across ledgers.")
        return "\n".join(lines)

    for cl in report.clusters[:max(1, limit)]:
        seen = f"last={cl.last_seen}" if cl.last_seen else "last=n/a"
        src = ",".join(cl.sources)
        head = f"  [{cl.count:>3}x] {cl.error_class} · {cl.module} · {cl.root_cause_tag} ({seen})"
        lines.append(head[:cap])
        sub = f"        sources={src} · {cl.sample_message}"
        lines.append(sub[:cap])
    return "\n".join(lines)


def run_command(args: str = "", *, osiris_home: PathArg = None) -> str:
    """CLI handler for ``/clusters [limit]`` and ``/runs clusters``."""
    parts = (args or "").strip().split()
    if parts and parts[0].lower() == "clusters":
        parts = parts[1:]
    limit = 15
    if parts:
        try:
            limit = max(1, min(100, int(parts[0])))
        except ValueError:
            limit = 15
    rep = analyze_failure_clusters(osiris_home=osiris_home)
    return format_clusters(rep, limit=limit)
