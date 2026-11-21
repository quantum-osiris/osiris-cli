"""L7 Promote: fail-closed persistent quarantine queue for inconclusive candidates.

Candidates that neither cleanly pass promotion gates nor cleanly fail (e.g., epistemic
firewall ``QUARANTINE``, unmeasured/interrupted runs, ambiguous skip/stability evidence)
are recorded in an append-only ``exec-canon-1`` SHA-256 hash-chained JSONL queue under
``$OSIRIS_HOME/promote/quarantine_queue.jsonl`` (mode ``0600``, ``flock`` + ``fsync``).

Invariant (fail-closed):
  * Any candidate whose ``candidate_digest`` (or overlapping ``target_path``) has an
    unresolved ``QUARANTINED`` entry — or a ``REJECTED`` resolution — is blocked from
    promotion until a human operator explicitly resolves it via CLI (``/quarantine``).
  * If the quarantine queue file itself fails hash-chain verification, promotion is
    refused across the board (``FAIL_CLOSED_LEDGER_TAMPERED``).
  * Agent origins (``model``, ``skill``, ``claude_code``) may never resolve quarantine items.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .exec_kernel import AGENT_ORIGINS, ExecJournal, canon

SCHEMA = "osiris.quarantine_queue/1"
VALID_RESOLUTIONS = ("APPROVED", "REJECTED")
PathArg = Optional[Union[os.PathLike, str]]


class QuarantineError(ValueError):
    """Invalid quarantine transition, unauthorized origin, or corrupted ledger."""


def default_osiris_home(osiris_home: PathArg = None) -> Path:
    raw = osiris_home or os.environ.get("OSIRIS_HOME")
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".osiris").resolve()


def default_quarantine_path(osiris_home: PathArg = None) -> Path:
    if osiris_home is None:
        override = os.environ.get("OSIRIS_QUARANTINE_QUEUE")
        if override:
            return Path(override).expanduser().resolve()
    return default_osiris_home(osiris_home) / "promote" / "quarantine_queue.jsonl"


def _canon_safe_value(val: Any) -> Any:
    """Convert floats to decimal strings recursively so ``exec-canon-1`` never sees floats."""
    if isinstance(val, float):
        if not (val == val and abs(val) != float("inf")):
            return "NaN"
        return str(Decimal(str(val)))
    if isinstance(val, Decimal):
        return str(val)
    if isinstance(val, dict):
        return {str(k): _canon_safe_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_canon_safe_value(v) for v in val]
    return val


@dataclass(frozen=True)
class QuarantineRecord:
    quarantine_id: str
    candidate_id: str
    candidate_digest: str
    status: str  # QUARANTINED | APPROVED | REJECTED
    reasons: Tuple[str, ...]
    target_paths: Tuple[str, ...]
    run_id: str
    source: str
    enqueued_at: str
    resolved_at: str = ""
    resolved_by: str = ""
    resolution_note: str = ""
    last_receipt_hash: str = ""


@dataclass(frozen=True)
class QuarantineGateVerdict:
    allowed: bool
    status: str  # CLEAR | RELEASED | BLOCKED_QUARANTINED | BLOCKED_REJECTED | FAIL_CLOSED_LEDGER_TAMPERED
    reason: str
    quarantine_id: str = ""


class QuarantineQueue:
    """Append-only ``exec-canon-1`` hash-chained quarantine queue under ``$OSIRIS_HOME``."""

    def __init__(self, path: PathArg = None) -> None:
        self.path = Path(path).expanduser().resolve() if path is not None else default_quarantine_path()
        self._journal = ExecJournal(self.path)

    def _append_body(self, body: Dict[str, Any]) -> str:
        canon(body)
        digest = self._journal.append(body)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass
        return digest

    def verify(self) -> Tuple[bool, int, str]:
        return self._journal.verify()

    def verify_tail(self, tail_lines: int = 20) -> Any:
        from . import probes
        return probes.verify_chain_tail(self.path, tail_lines=tail_lines)

    def enqueue(
        self,
        candidate_id: str,
        candidate_digest: str,
        reasons: Sequence[str],
        *,
        target_paths: Iterable[str] = (),
        run_id: str = "",
        source: str = "evolution_bridge",
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Tuple[QuarantineRecord, str]:
        """Place an inconclusive candidate into ``QUARANTINED`` state."""
        if not candidate_digest or not str(candidate_digest).strip():
            raise QuarantineError("candidate_digest is required to quarantine a candidate")
        clean_digest = str(candidate_digest).strip()
        clean_cid = str(candidate_id).strip() or f"cand_{clean_digest[:12]}"
        qid = "quar-" + hashlib.sha256(clean_digest.encode("utf-8")).hexdigest()[:12]
        reason_list = [str(r) for r in reasons if str(r).strip()] or ["inconclusive candidate outcome"]
        norm_paths = tuple(sorted({str(p).replace(os.sep, "/").strip() for p in target_paths if str(p).strip()}))
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        body: Dict[str, Any] = {
            "schema": SCHEMA,
            "event": "ENQUEUED",
            "quarantine_id": qid,
            "candidate_id": clean_cid,
            "candidate_digest": clean_digest,
            "status": "QUARANTINED",
            "reasons": reason_list,
            "target_paths": list(norm_paths),
            "run_id": str(run_id),
            "source": str(source),
            "metrics": _canon_safe_value(metrics or {}),
            "ts": ts,
        }
        rhash = self._append_body(body)
        rec = QuarantineRecord(
            quarantine_id=qid,
            candidate_id=clean_cid,
            candidate_digest=clean_digest,
            status="QUARANTINED",
            reasons=tuple(reason_list),
            target_paths=norm_paths,
            run_id=str(run_id),
            source=str(source),
            enqueued_at=ts,
            last_receipt_hash=rhash,
        )
        return rec, rhash

    def resolve(
        self,
        quarantine_id: str,
        resolution: str,
        *,
        note: str,
        origin: str = "human",
        reviewer: str = "operator",
    ) -> Tuple[QuarantineRecord, str]:
        """Explicit human resolution (``APPROVED`` releases quarantine hold; ``REJECTED`` seals block)."""
        if origin in AGENT_ORIGINS or origin != "human":
            raise QuarantineError(
                f"origin {origin!r} cannot resolve quarantine entries; human operator required"
            )
        res_upper = str(resolution).strip().upper()
        if res_upper == "APPROVE":
            res_upper = "APPROVED"
        elif res_upper == "REJECT":
            res_upper = "REJECTED"
        if res_upper not in VALID_RESOLUTIONS:
            raise QuarantineError(f"resolution must be APPROVED or REJECTED, got {resolution!r}")
        clean_note = str(note or "").strip()
        if not clean_note:
            raise QuarantineError("explicit resolution note/reason is required")

        state_map = self._replay()
        qid = str(quarantine_id).strip()
        if qid not in state_map:
            # Allow lookup by candidate_id or candidate_digest prefix if unambiguous.
            matches = [
                r for r in state_map.values()
                if r.candidate_id == qid or r.candidate_digest.startswith(qid) or r.quarantine_id.startswith(qid)
            ]
            if len(matches) != 1:
                raise QuarantineError(f"unknown or ambiguous quarantine identifier {quarantine_id!r}")
            current = matches[0]
            qid = current.quarantine_id
        else:
            current = state_map[qid]

        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        body = {
            "schema": SCHEMA,
            "event": "RESOLVED",
            "quarantine_id": qid,
            "candidate_id": current.candidate_id,
            "candidate_digest": current.candidate_digest,
            "status": res_upper,
            "reviewer": str(reviewer),
            "origin": "human",
            "note": clean_note,
            "ts": ts,
        }
        rhash = self._append_body(body)
        updated = QuarantineRecord(
            quarantine_id=qid,
            candidate_id=current.candidate_id,
            candidate_digest=current.candidate_digest,
            status=res_upper,
            reasons=current.reasons,
            target_paths=current.target_paths,
            run_id=current.run_id,
            source=current.source,
            enqueued_at=current.enqueued_at,
            resolved_at=ts,
            resolved_by=str(reviewer),
            resolution_note=clean_note,
            last_receipt_hash=rhash,
        )
        return updated, rhash

    def _replay(self) -> Dict[str, QuarantineRecord]:
        if not self.path.is_file():
            return {}
        ok, _cnt, err = self.verify()
        if not ok:
            raise QuarantineError(f"quarantine queue integrity check failed: {err}")

        items: Dict[str, QuarantineRecord] = {}
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                rhash = str(row.get("hash", ""))
                body = row.get("body") or {}
                ev = body.get("event")
                qid = str(body.get("quarantine_id", ""))
                if not qid:
                    continue
                if ev == "ENQUEUED":
                    items[qid] = QuarantineRecord(
                        quarantine_id=qid,
                        candidate_id=str(body.get("candidate_id", "")),
                        candidate_digest=str(body.get("candidate_digest", "")),
                        status="QUARANTINED",
                        reasons=tuple(str(x) for x in (body.get("reasons") or ())),
                        target_paths=tuple(str(x) for x in (body.get("target_paths") or ())),
                        run_id=str(body.get("run_id", "")),
                        source=str(body.get("source", "")),
                        enqueued_at=str(body.get("ts", "")),
                        last_receipt_hash=rhash,
                    )
                elif ev == "RESOLVED" and qid in items:
                    prev = items[qid]
                    items[qid] = QuarantineRecord(
                        quarantine_id=qid,
                        candidate_id=prev.candidate_id,
                        candidate_digest=prev.candidate_digest,
                        status=str(body.get("status", "REJECTED")),
                        reasons=prev.reasons,
                        target_paths=prev.target_paths,
                        run_id=prev.run_id,
                        source=prev.source,
                        enqueued_at=prev.enqueued_at,
                        resolved_at=str(body.get("ts", "")),
                        resolved_by=str(body.get("reviewer", "")),
                        resolution_note=str(body.get("note", "")),
                        last_receipt_hash=rhash,
                    )
        return items

    def list_items(self, *, active_only: bool = False) -> List[QuarantineRecord]:
        items = list(self._replay().values())
        if active_only:
            items = [it for it in items if it.status == "QUARANTINED"]
        return sorted(items, key=lambda r: (r.enqueued_at, r.quarantine_id))

    def get(self, identifier: str) -> Optional[QuarantineRecord]:
        ident = str(identifier).strip()
        state_map = self._replay()
        if ident in state_map:
            return state_map[ident]
        matches = [
            r for r in state_map.values()
            if r.candidate_id == ident or r.candidate_digest.startswith(ident) or r.quarantine_id.startswith(ident)
        ]
        return matches[0] if len(matches) == 1 else None

    def check_promotion_allowed(
        self,
        *,
        candidate_digest: str = "",
        target_path: str = "",
    ) -> QuarantineGateVerdict:
        """Fail-closed promotion check: blocks unresolved or rejected quarantined candidates."""
        if self.path.exists():
            ok, _cnt, err = self.verify()
            if not ok:
                return QuarantineGateVerdict(
                    allowed=False,
                    status="FAIL_CLOSED_LEDGER_TAMPERED",
                    reason=f"quarantine queue hash chain broken ({err}); promotion blocked fail-closed",
                )
        try:
            items = self._replay()
        except QuarantineError as exc:
            return QuarantineGateVerdict(
                allowed=False,
                status="FAIL_CLOSED_LEDGER_TAMPERED",
                reason=str(exc),
            )

        clean_digest = str(candidate_digest or "").strip()
        norm_target = str(target_path or "").replace(os.sep, "/").strip()

        if clean_digest:
            for rec in items.values():
                if rec.candidate_digest == clean_digest:
                    if rec.status == "QUARANTINED":
                        return QuarantineGateVerdict(
                            allowed=False,
                            status="BLOCKED_QUARANTINED",
                            reason=(
                                f"candidate {rec.candidate_id} is quarantined ({rec.quarantine_id}: "
                                f"{'; '.join(rec.reasons)}); resolve via /quarantine resolve {rec.quarantine_id} ..."
                            ),
                            quarantine_id=rec.quarantine_id,
                        )
                    if rec.status == "REJECTED":
                        return QuarantineGateVerdict(
                            allowed=False,
                            status="BLOCKED_REJECTED",
                            reason=(
                                f"candidate {rec.candidate_id} was rejected in quarantine review "
                                f"({rec.quarantine_id}: {rec.resolution_note})"
                            ),
                            quarantine_id=rec.quarantine_id,
                        )
                    if rec.status == "APPROVED":
                        return QuarantineGateVerdict(
                            allowed=True,
                            status="RELEASED",
                            reason=f"released from quarantine {rec.quarantine_id}: {rec.resolution_note}",
                            quarantine_id=rec.quarantine_id,
                        )

        if norm_target:
            base_target = os.path.basename(norm_target)
            for rec in items.values():
                if rec.status != "QUARANTINED":
                    continue
                for tp in rec.target_paths:
                    if tp == norm_target or (tp and norm_target.endswith("/" + tp)) or (
                        base_target and tp == base_target and "/" not in norm_target
                    ):
                        return QuarantineGateVerdict(
                            allowed=False,
                            status="BLOCKED_QUARANTINED",
                            reason=(
                                f"target {norm_target} has unresolved quarantined candidate "
                                f"{rec.quarantine_id} ({rec.candidate_id}); resolve via /quarantine first"
                            ),
                            quarantine_id=rec.quarantine_id,
                        )

        return QuarantineGateVerdict(allowed=True, status="CLEAR", reason="no blocking quarantine record")


def format_queue(path: PathArg = None, *, include_resolved: bool = False, width: int = 92) -> str:
    """Render compact (<100 col) quarantine queue status."""
    q = QuarantineQueue(path)
    ok, row_count, err = q.verify()
    if not ok:
        return f"L7 QUARANTINE QUEUE · TAMPERED ({err}) -- promotion FAIL-CLOSED"
    all_items = q.list_items(active_only=False)
    active = [it for it in all_items if it.status == "QUARANTINED"]
    shown = all_items if include_resolved else active
    cap = max(48, min(width, 96))
    lines = [
        f"L7 QUARANTINE QUEUE · chain INTACT ({row_count} events · "
        f"{len(active)} active, {len(all_items) - len(active)} resolved)"
    ]
    if not shown:
        lines.append("  queue clear: 0 unresolved quarantined candidates.")
        return "\n".join(lines)
    for rec in shown:
        targets = ",".join(rec.target_paths) or "(unspecified)"
        head = f"  [{rec.status}] {rec.quarantine_id} · {rec.candidate_id} · targets={targets}"
        lines.append(head[:cap])
        why = "; ".join(rec.reasons)
        lines.append(("    reason: " + why)[:cap])
        if rec.status != "QUARANTINED":
            lines.append((f"    resolved by {rec.resolved_by}: {rec.resolution_note}")[:cap])
    return "\n".join(lines)


def run_command(
    args: str = "",
    *,
    osiris_home: PathArg = None,
    origin: str = "human",
) -> str:
    """CLI handler for ``/quarantine [list|all|show <id>|resolve <id> approve|reject <note>|verify]``."""
    try:
        tokens = shlex.split(args or "")
    except ValueError:
        tokens = (args or "").strip().split()

    q = QuarantineQueue(default_quarantine_path(osiris_home))
    sub = tokens[0].lower() if tokens else "list"

    if sub in ("list", "active"):
        return format_queue(q.path, include_resolved=False)
    if sub == "all":
        return format_queue(q.path, include_resolved=True)
    if sub == "verify":
        ok, count, err = q.verify()
        tail = q.verify_tail()
        status = "INTACT" if (ok and tail.intact) else "TAMPERED"
        return f"L7 quarantine chain {status}: {count} events, tail={tail.status}" + (f" ({err})" if err else "")
    if sub == "show" and len(tokens) >= 2:
        rec = q.get(tokens[1])
        if rec is None:
            return f"Quarantine item {tokens[1]!r} not found."
        lines = [
            f"Quarantine ID:    {rec.quarantine_id}",
            f"Candidate:        {rec.candidate_id} (digest {rec.candidate_digest[:16]}...)",
            f"Status:           {rec.status}",
            f"Target paths:     {', '.join(rec.target_paths) or '(none)'}",
            f"Run ID / Source:  {rec.run_id or '(none)'} / {rec.source}",
            f"Enqueued at:      {rec.enqueued_at}",
            f"Reasons:          {'; '.join(rec.reasons)}",
        ]
        if rec.resolved_at:
            lines.append(f"Resolved at:      {rec.resolved_at} by {rec.resolved_by}")
            lines.append(f"Resolution note:  {rec.resolution_note}")
        return "\n".join(lines)
    if sub == "resolve":
        if len(tokens) < 4:
            return "Usage: /quarantine resolve <quarantine_id> <approve|reject> <reason>"
        qid, action = tokens[1], tokens[2]
        note = " ".join(tokens[3:]).strip()
        try:
            updated, rhash = q.resolve(qid, action, note=note, origin=origin)
        except QuarantineError as exc:
            return f"Quarantine resolution refused: {exc}"
        return (
            f"Quarantine {updated.quarantine_id} -> {updated.status} "
            f"(receipt {rhash[:12]}; note: {updated.resolution_note})"
        )
    return (
        "Usage: /quarantine [list | all | show <id> | resolve <id> <approve|reject> <reason> | verify]"
    )
