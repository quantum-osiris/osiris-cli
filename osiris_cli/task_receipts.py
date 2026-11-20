"""OSIRIS Task Receipt Chain (~/.osiris/task_receipts.jsonl, §11 / REQ-EV-1..7).

Append-only, hash-chained JSONL recording every state transition of every task-graph node,
using the execution kernel's exact canonical form (`exec-canon-1`: sorted keys, compact
separators, floats rejected) with file mode 0600, exclusive file lock and fsync per append.

Rather than migrating or rewriting the six existing hash chains (G-13), each task receipt
carries a `refs` object pointing into referenced chains by SHA-256 hash (`exec_journal_hash`,
`genome_ledger_hash`, `authorization_ledger_hash`, `operational_journal_hash`, `artifact_sha256`).
`verify()` walks both this chain and every configured referenced chain; any tampered byte in
any chain blocks new task graphs from starting (REQ-EV-4).

Usage:
    python -m osiris_cli.task_receipts --verify [--path ~/.osiris/task_receipts.jsonl]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .exec_kernel import (
    CANON_VERSION,
    ExecJournal,
    _GENESIS,
    _default_state_home,
    _file_lock,
    _last_line,
    _link_hash,
    canon,
)

__all__ = [
    "CANON_VERSION",
    "EVIDENCE_CLASSES",
    "REF_LEDGER_KEYS",
    "TaskReceiptLedger",
    "build_transition_body",
    "default_task_receipts_path",
    "verify_referenced_chain",
]

EVIDENCE_CLASSES = frozenset(
    {"SIMULATED", "HARDWARE_ATTESTED", "PREVIEW", "SYNDROME_SIMULATED", "STATIC_RECORD", "COMPUTED"}
)

REF_LEDGER_KEYS = {
    "exec_journal": "exec_journal_hash",
    "genome_ledger": "genome_ledger_hash",
    "authorization_ledger": "authorization_ledger_hash",
    "operational_journal": "operational_journal_hash",
}


def default_task_receipts_path() -> Path:
    """Resolve ~/.osiris/task_receipts.jsonl lazily without touching disk on import."""
    override = os.environ.get("OSIRIS_TASK_RECEIPTS")
    if override:
        return Path(override).expanduser()
    return _default_state_home() / "task_receipts.jsonl"


def build_transition_body(
    *,
    graph_id: str,
    node_id: str,
    state_before: Optional[str],
    state_after: str,
    actor_role: str,
    actor_origin: str,
    action_digest: str,
    reason: str,
    session_id: Optional[str] = None,
    model_route: Optional[Dict[str, Any]] = None,
    grant: Optional[Dict[str, Any]] = None,
    refs: Optional[Mapping[str, Optional[str]]] = None,
    verification: Optional[Dict[str, Any]] = None,
    evidence_class: str = "COMPUTED",
    hardware: bool = False,
    supersedes: Optional[str] = None,
    receipt_id: Optional[str] = None,
    recorded_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a canonical float-free task-receipt body compliant with exec-canon-1."""
    if evidence_class not in EVIDENCE_CLASSES:
        raise ValueError(f"unsupported evidence_class: {evidence_class!r}")
    if evidence_class != "HARDWARE_ATTESTED" and hardware is not False:
        raise ValueError(f"evidence_class={evidence_class!r} requires hardware=False")

    clean_refs = {
        "exec_journal_hash": None,
        "genome_ledger_hash": None,
        "authorization_ledger_hash": None,
        "operational_journal_hash": None,
        "artifact_sha256": None,
    }
    if refs:
        for key in clean_refs:
            val = refs.get(key)
            if val is not None:
                clean_refs[key] = str(val)

    body: Dict[str, Any] = {
        "kind": "task-receipt",
        "receipt_version": "1.0",
        "receipt_id": receipt_id or ("trcpt-" + uuid.uuid4().hex[:16]),
        "actor": {
            "role": str(actor_role),
            "origin": str(actor_origin),
            "session_id": str(session_id) if session_id is not None else None,
            "model_route": model_route,
        },
        "task": {
            "graph_id": str(graph_id),
            "node_id": str(node_id),
            "state_before": str(state_before) if state_before is not None else None,
            "state_after": str(state_after),
        },
        "action_digest": str(action_digest),
        "grant": dict(grant) if grant is not None else None,
        "refs": clean_refs,
        "verification": dict(verification) if verification is not None else None,
        "evidence_class": evidence_class,
        "hardware": bool(hardware),
        "timestamps": {
            "recorded_ms": int(recorded_ms if recorded_ms is not None else time.time() * 1000),
        },
        "reason": str(reason)[:500],
        "supersedes": str(supersedes) if supersedes is not None else None,
    }
    # Fail immediately if any float sneaked into nested dicts (REQ-EV-2)
    canon(body)
    return body


def verify_referenced_chain(name: str, path: Path) -> Tuple[bool, set[str], str]:
    """Verify one referenced hash-chained ledger and return (ok, set_of_hashes, error)."""
    if not path.exists():
        return True, set(), ""
    try:
        raw_lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except OSError as exc:
        return False, set(), f"{name}: unreadable ({exc})"

    if not raw_lines:
        return True, set(), ""

    hashes: set[str] = set()

    if name == "exec_journal":
        ok, _, err = ExecJournal(path).verify()
        if not ok:
            return False, set(), f"exec_journal: {err}"
        for ln in raw_lines:
            entry = json.loads(ln)
            h = entry.get("hash")
            if isinstance(h, str):
                hashes.add(h)
        return True, hashes, ""

    if name == "operational_journal":
        from .journal import OperationalMerkleLedger

        err = OperationalMerkleLedger(path).verify()
        if err is not None:
            return False, set(), f"operational_journal: {err}"
        for ln in raw_lines:
            entry = json.loads(ln)
            h = entry.get("hash")
            if isinstance(h, str):
                hashes.add(h)
        return True, hashes, ""

    # Generic JSONL hash chain check (genome_ledger, authorization_ledger, slice_receipts):
    # Supports both exec-canon-1 entries and prev/hash JSONL chains (trio_authz / grants).
    first = None
    try:
        first = json.loads(raw_lines[0])
    except ValueError:
        return False, set(), f"{name}: line 1 is not JSON"

    if isinstance(first, dict) and first.get("canon_version") == CANON_VERSION:
        ok, _, err = ExecJournal(path).verify()
        if not ok:
            return False, set(), f"{name}: {err}"
        for ln in raw_lines:
            h = json.loads(ln).get("hash")
            if isinstance(h, str):
                hashes.add(h)
        return True, hashes, ""

    if isinstance(first, dict) and "prev" in first and "hash" in first:
        from .trio_authz import Receipts

        if not Receipts.verify(str(path)):
            return False, set(), f"{name}: hash chain verification failed"
        for ln in raw_lines:
            h = json.loads(ln).get("hash")
            if isinstance(h, str):
                hashes.add(h)
        return True, hashes, ""

    # Fallback: check OperationalMerkleLedger style or grants ledger prev_hash/entry_hash
    prev_seen = _GENESIS
    for idx, ln in enumerate(raw_lines, 1):
        try:
            row = json.loads(ln)
        except ValueError:
            return False, set(), f"{name}: line {idx} is not JSON"
        if not isinstance(row, dict):
            return False, set(), f"{name}: line {idx} is not a JSON object"
        h = row.get("hash") or row.get("entry_hash") or row.get("receipt_hash")
        p = row.get("prev") if "prev" in row else row.get("prev_hash")
        if p is not None and p != prev_seen:
            return False, set(), f"{name}: line {idx} prev link mismatch"
        if isinstance(h, str):
            hashes.add(h)
            prev_seen = h
        else:
            return False, set(), f"{name}: line {idx} missing hash field"
    return True, hashes, ""


class TaskReceiptLedger:
    """Append-only, 0600, lock+fsync, exec-canon-1 task receipt ledger (REQ-EV-1..7)."""

    def __init__(self, path: Optional[object] = None) -> None:
        self.path = Path(path) if path is not None else default_task_receipts_path()
        self._thread_lock = threading.Lock()

    def append(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Append one transition receipt atomically under lock + fsync. Rejects floats."""
        canon(body)  # Validate float-free canonical JSON before taking locks
        with self._thread_lock, _file_lock(self.path):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            last = _last_line(self.path)
            if last is None:
                prev, seq = _GENESIS, 0
            else:
                try:
                    entry = json.loads(last.decode("utf-8"))
                    prev, seq = entry["hash"], int(entry["seq"])
                except Exception as exc:
                    raise ValueError(f"corrupt task receipt chain tail: {exc}") from exc
            digest = _link_hash(prev, body)
            row = {
                "canon_version": CANON_VERSION,
                "seq": seq + 1,
                "prev": prev,
                "body": body,
                "hash": digest,
            }
            line = json.dumps(
                row, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            )
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass
            with os.fdopen(fd, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return row

    def rows(self, *, graph_id: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Read receipt rows (optionally filtered to one graph_id)."""
        if not self.path.exists():
            return []
        out: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                except ValueError:
                    continue
                if graph_id is not None:
                    task_info = (entry.get("body") or {}).get("task") or {}
                    if task_info.get("graph_id") != graph_id:
                        continue
                out.append(entry)
        if limit is not None and limit > 0:
            return out[-limit:]
        return out

    def tip_hash(self) -> str:
        last = _last_line(self.path)
        if last is None:
            return _GENESIS
        try:
            return str(json.loads(last.decode("utf-8"))["hash"])
        except Exception:
            return ""

    def verify(
        self,
        referenced_ledgers: Optional[Mapping[str, object]] = None,
    ) -> Tuple[bool, int, str]:
        """Walk this chain and every referenced chain. Returns (ok, checked_count, error)."""
        seen_entries: List[Dict[str, Any]] = []
        if self.path.exists():
            prev, count = _GENESIS, 0
            try:
                with open(self.path, "r", encoding="utf-8") as handle:
                    for lineno, raw in enumerate(handle, 1):
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            entry = json.loads(raw)
                        except ValueError:
                            return False, count, f"task_receipts line {lineno}: not JSON"
                        if not isinstance(entry, dict):
                            return False, count, f"task_receipts line {lineno}: not a JSON object"
                        if entry.get("canon_version") != CANON_VERSION:
                            return False, count, f"task_receipts line {lineno}: unknown canon_version"
                        if entry.get("prev") != prev or entry.get("seq") != count + 1:
                            return False, count, f"task_receipts line {lineno}: broken link or sequence"
                        try:
                            want = _link_hash(prev, entry["body"])
                        except (KeyError, ValueError) as exc:
                            return False, count, f"task_receipts line {lineno}: {exc}"
                        if want != entry.get("hash"):
                            return False, count, f"task_receipts line {lineno}: hash mismatch"
                        prev = want
                        count += 1
                        seen_entries.append(entry)
            except OSError as exc:
                return False, 0, f"task_receipts unreadable: {exc}"
        else:
            count = 0

        # Verify every configured referenced ledger and cross-check any referenced hashes (REQ-EV-4)
        ref_hashes_by_name: Dict[str, set[str]] = {}
        if referenced_ledgers:
            for name, ref_path in referenced_ledgers.items():
                if ref_path is None:
                    continue
                ok, found_hashes, err = verify_referenced_chain(name, Path(ref_path))
                if not ok:
                    return False, count, err
                ref_hashes_by_name[name] = found_hashes

            for idx, entry in enumerate(seen_entries, 1):
                refs = (entry.get("body") or {}).get("refs") or {}
                for ledger_name, ref_key in REF_LEDGER_KEYS.items():
                    wanted = refs.get(ref_key)
                    if wanted and ledger_name in ref_hashes_by_name:
                        if wanted not in ref_hashes_by_name[ledger_name]:
                            return (
                                False,
                                count,
                                f"task_receipts line {idx}: dangling {ref_key}={wanted[:12]} not found in {ledger_name}",
                            )

        return True, count, ""


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m osiris_cli.task_receipts", description=__doc__.split("\n")[0])
    ap.add_argument("--path", default=None, help="Path to task_receipts.jsonl")
    ap.add_argument("--exec-journal", default=None, help="Optional exec_journal.jsonl to cross-verify")
    ap.add_argument("--verify", action="store_true", help="Verify the receipt chain and exit 0/1")
    args = ap.parse_args(argv)

    ledger = TaskReceiptLedger(args.path)
    refs = {"exec_journal": args.exec_journal} if args.exec_journal else None
    ok, count, err = ledger.verify(referenced_ledgers=refs)
    print(json.dumps({"ok": ok, "count": count, "tip_hash": ledger.tip_hash(), "error": err}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
