"""Fail-closed dual-substrate guards for Sprint 0.

These checks are deterministic. They do not run a GPU, a QPU, or a network
publish. A simulation result stays simulation evidence.

The historical defects this module closes were not present as named functions
on ``main`` at ``c1f0444``. The unsafe shapes are rejected here, and the
replacement behavior is what callers must use:

* remote Alpha probes are a versioned helper file, argv-only;
* an incoming evidence chain replaces the local chain only when it extends it;
* an idempotency key that has already executed is not executed again;
* a manifest is renamed into quarantine before it is verified or deleted;
* missing git commit info fails closed;
* success telemetry must name the current experiment and both stages;
* mock Zenodo fixtures, including the fixture confidence 0.6, stay out of
  production evidence.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import secrets
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from osiris_cli import chain_lock
from osiris_cli.remote_helpers import HELPER_VERSION
from osiris_cli.remote_helpers.alpha_state_v1 import HELPER_VERSION as ALPHA_HELPER_VERSION

GENESIS = "0" * 64
STAGES = ("dispatch", "result")
MOCK_FIXTURE_CONFIDENCE = 0.6
_MOCK_CLASSES = frozenset({"MOCK", "FIXTURE"})
_MOCK_SOURCES = frozenset({"mock", "fixture"})


class SubstrateGuardError(ValueError):
    """A dual-substrate guard refused the operation."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _entry_hash(body: Mapping[str, Any]) -> str:
    clean = {k: v for k, v in body.items() if k != "hash"}
    raw = json.dumps(clean, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def verify_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Require a linear SHA-256 chain. A fork, a gap, or a bad sequence is refused.

    The digest matches ``osiris_cli.probes.verify_chain``: hash is excluded from
    the canonical JSON, keys are sorted, and ``prev`` must be the prior hash.
    """
    prev = GENESIS
    seen = {GENESIS}
    verified: List[Dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or not isinstance(raw.get("hash"), str):
            raise SubstrateGuardError(f"entry {index} has no hash")
        row = dict(raw)
        recorded = row["hash"]
        if _entry_hash(row) != recorded:
            raise SubstrateGuardError(f"entry {index} hash mismatch")
        link = row.get("prev")
        if link not in seen:
            raise SubstrateGuardError(f"entry {index} links to no ancestor")
        if link != prev:
            raise SubstrateGuardError(f"entry {index} diverges from the previous hash")
        seq = row.get("seq")
        if not isinstance(seq, int) or seq != index + 1:
            raise SubstrateGuardError(f"entry {index} sequence {seq!r} is not {index + 1}")
        seen.add(recorded)
        prev = recorded
        verified.append(row)
    return verified


def load_chain(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return verify_rows(rows)


def _next_entry(prev: Optional[Mapping[str, Any]], fields: Mapping[str, Any]) -> Dict[str, Any]:
    entry = dict(fields)
    entry["prev"] = prev["hash"] if prev else GENESIS
    entry["seq"] = (int(prev["seq"]) + 1) if prev else 1
    entry["hash"] = _entry_hash(entry)
    return entry


def append_entry(path: Path, fields: Mapping[str, Any]) -> Dict[str, Any]:
    """Append one chained entry under the ledger lock. ``fields`` must not set chain keys."""
    forbidden = {"hash", "prev", "seq"} & set(fields)
    if forbidden:
        raise SubstrateGuardError(f"caller must not set chain keys {sorted(forbidden)}")

    def build_checked(last: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if last is not None and (last.get("hash") != _entry_hash(last) or not isinstance(last.get("seq"), int)):
            raise SubstrateGuardError("ledger tail is not a valid chained entry")
        return _next_entry(last, fields)

    path.parent.mkdir(parents=True, exist_ok=True)
    return chain_lock.append_chained(str(path), build_checked)


def bundled_alpha_helper() -> Path:
    return Path(__file__).resolve().parent / "remote_helpers" / "alpha_state_v1.py"


def remote_python_defect(source: str) -> Optional[str]:
    """Name the historical remote-Python defect, if ``source`` still has it.

    A compound ``for`` after a semicolon does not parse. Shell-quoting that
    source with ``json.dumps`` is the second half of the same defect and is
    reported by :func:`shell_quotes_multiline_source`.
    """
    try:
        ast.parse(source)
    except SyntaxError as exc:
        if ";" in source and "for " in source:
            return f"compound for-statement follows a semicolon ({exc.msg})"
        return f"remote python does not parse ({exc.msg})"
    return None


def shell_quotes_multiline_source(command: str, source: str) -> bool:
    """True when ``command`` carries ``source`` via ``json.dumps`` and ``source`` is multiline."""
    return "\n" in source and json.dumps(source) in command


def probe_alpha_state(interpreter: str = "python3", helper: Optional[Path] = None) -> Dict[str, Any]:
    """Build an argv-only probe of Node Alpha. Never a shell string.

    The helper is the versioned file shipped in this package. Its ``for`` loop
    is a real statement. The argv does not contain that source, a newline, or
    a ``json.dumps`` quoting of it.
    """
    path = Path(helper) if helper is not None else bundled_alpha_helper()
    source = path.read_text(encoding="utf-8")
    defect = remote_python_defect(source)
    if defect:
        raise SubstrateGuardError(defect)
    if HELPER_VERSION != ALPHA_HELPER_VERSION or f'HELPER_VERSION = "{HELPER_VERSION}"' not in source:
        raise SubstrateGuardError("alpha helper version does not match the package contract")
    argv = [interpreter, str(path)]
    rendered = " ".join(argv)
    if shell_quotes_multiline_source(rendered, source) or "\n" in rendered or ";" in rendered:
        raise SubstrateGuardError("probe command must not shell-quote the helper")
    return {
        "helper_version": HELPER_VERSION,
        "argv": argv,
        "shell": False,
        "encoding": "versioned-file",
        "helper_path": str(path),
    }


def accept_return_chain(
    local: Sequence[Mapping[str, Any]],
    incoming: Sequence[Mapping[str, Any]],
) -> str:
    """Decide whether Alpha may replace its chain with ``incoming``.

    Returns ``identical`` (no write), ``extend`` (incoming continues the local
    tip), or ``replace`` (local is empty). An older tip or a chain that does
    not descend from the local tip raises :class:`SubstrateGuardError`.
    """
    local_rows = verify_rows(local) if local else []
    incoming_rows = verify_rows(list(incoming))
    if not incoming_rows:
        raise SubstrateGuardError("incoming chain is empty")
    if not local_rows:
        return "replace"
    local_hashes = [row["hash"] for row in local_rows]
    incoming_hashes = [row["hash"] for row in incoming_rows]
    local_head = local_hashes[-1]
    incoming_head = incoming_hashes[-1]
    if incoming_head == local_head and incoming_hashes == local_hashes:
        return "identical"
    if local_head in incoming_hashes and incoming_hashes[: len(local_hashes)] == local_hashes:
        if len(incoming_rows) > len(local_rows) and incoming_rows[-1]["seq"] > local_rows[-1]["seq"]:
            return "extend"
        raise SubstrateGuardError("incoming chain does not extend the local tip")
    if incoming_head in local_hashes:
        raise SubstrateGuardError("older chain: incoming tip is an ancestor of the local tip")
    raise SubstrateGuardError("divergent chain: local tip is not an ancestor of the incoming tip")


def apply_return_chain(local_path: Path, incoming: Sequence[Mapping[str, Any]]) -> str:
    """Replace ``local_path`` only when :func:`accept_return_chain` allows it."""
    local_rows = load_chain(local_path)
    decision = accept_return_chain(local_rows, incoming)
    if decision == "identical":
        return decision
    payload = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in verify_rows(list(incoming)))
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = local_path.with_name(local_path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, local_path)
    return decision


def _rows_for_key(rows: Sequence[Mapping[str, Any]], idempotency_key: str) -> List[Dict[str, Any]]:
    return [dict(row) for row in rows if row.get("idempotency_key") == idempotency_key]


def begin_experiment(
    path: Path,
    experiment_id: str,
    idempotency_key: str,
    payload_sha256: str,
) -> Dict[str, Any]:
    """Claim ``idempotency_key`` once.

    A key that already has an ``executed`` or ``acked`` row returns that result
    and does not ask the caller to run the experiment again. A key that was
    started and not finished stays in progress; it is not started a second time.
    """
    if not experiment_id or not idempotency_key or len(payload_sha256) != 64:
        raise SubstrateGuardError("experiment_id, idempotency_key, and a 64-hex payload hash are required")
    path.parent.mkdir(parents=True, exist_ok=True)
    with chain_lock.locked_tail(str(path)) as (handle, _last):
        handle.seek(0)
        raw = handle.read().decode("utf-8")
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        verified = verify_rows(rows) if rows else []
        prior = _rows_for_key(verified, idempotency_key)
        phases = [row.get("phase") for row in prior]
        if prior and prior[0].get("experiment_id") != experiment_id:
            raise SubstrateGuardError("idempotency key is already bound to another experiment")
        if prior and prior[0].get("payload_sha256") != payload_sha256:
            raise SubstrateGuardError("idempotency key was already used for a different payload")
        if "acked" in phases or "executed" in phases:
            done = [row for row in prior if row.get("phase") in ("executed", "acked")][-1]
            return {"execute": False, "replayed": True, "phase": done.get("phase"), "result": done.get("result")}
        if "started" in phases:
            return {"execute": False, "replayed": False, "phase": "started", "result": None}
        entry = _next_entry(
            verified[-1] if verified else None,
            {
                "experiment_id": experiment_id,
                "idempotency_key": idempotency_key,
                "payload_sha256": payload_sha256,
                "phase": "started",
            },
        )
        chain_lock.write_line(handle, entry, sort_keys=True, ensure_ascii=False)
        return {"execute": True, "replayed": False, "phase": "started", "result": None}


def record_execution(
    path: Path,
    experiment_id: str,
    idempotency_key: str,
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    """Persist the execution result before the caller acknowledges it."""
    rows = load_chain(path)
    prior = _rows_for_key(rows, idempotency_key)
    if not prior or prior[-1].get("phase") != "started" or prior[-1].get("experiment_id") != experiment_id:
        raise SubstrateGuardError("execution requires a started row for this experiment and key")
    return append_entry(
        path,
        {
            "experiment_id": experiment_id,
            "idempotency_key": idempotency_key,
            "payload_sha256": prior[-1]["payload_sha256"],
            "phase": "executed",
            "result": dict(result),
        },
    )


def acknowledge(path: Path, experiment_id: str, idempotency_key: str) -> Dict[str, Any]:
    rows = load_chain(path)
    prior = _rows_for_key(rows, idempotency_key)
    executed = [row for row in prior if row.get("phase") == "executed" and row.get("experiment_id") == experiment_id]
    if not executed:
        raise SubstrateGuardError("ack requires an executed row for this experiment and key")
    return append_entry(
        path,
        {
            "experiment_id": experiment_id,
            "idempotency_key": idempotency_key,
            "payload_sha256": executed[-1]["payload_sha256"],
            "phase": "acked",
            "result": executed[-1].get("result"),
        },
    )


def retire_manifest(manifest: Path, quarantine_dir: Path, verify: Callable[[Path], bool]) -> str:
    """Rename ``manifest`` into quarantine, verify that copy, then delete it.

    ``os.replace`` is the check. A file that appears at the original path after
    the rename is a different file and is left alone. Verification failure
    leaves the renamed copy in quarantine.
    """
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    destination = quarantine_dir / f"{manifest.name}.{secrets.token_hex(8)}"
    try:
        os.replace(manifest, destination)
    except FileNotFoundError:
        return "absent"
    if not verify(destination):
        return "quarantined"
    destination.unlink()
    return "deleted"


def assert_strict_git_alignment(local_commit: Optional[str], remote_commit: Optional[str]) -> None:
    """Fail closed when either commit id is missing or the two ids differ."""
    if not isinstance(local_commit, str) or not local_commit.strip():
        raise SubstrateGuardError("strict git alignment failed closed: local commit info is missing")
    if not isinstance(remote_commit, str) or not remote_commit.strip():
        raise SubstrateGuardError("strict git alignment failed closed: remote commit info is missing")
    if local_commit.strip() != remote_commit.strip():
        raise SubstrateGuardError("strict git alignment failed closed: commit ids differ")


def verify_success_telemetry(entries: Iterable[Mapping[str, Any]], experiment_id: str) -> None:
    """Success requires both stages, and only rows for ``experiment_id`` count."""
    if not experiment_id:
        raise SubstrateGuardError("success verification requires an experiment id")
    stages = {
        row.get("stage")
        for row in entries
        if row.get("experiment_id") == experiment_id and row.get("stage") in STAGES
    }
    missing = [stage for stage in STAGES if stage not in stages]
    if missing:
        raise SubstrateGuardError(
            f"success verification requires telemetry stages {list(STAGES)} for experiment {experiment_id}; "
            f"missing {missing}"
        )


def classify_zenodo_evidence(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Quarantine mock or fixture Zenodo records, including fixture confidence 0.6.

    A record with no mock or fixture marker is not assigned a confidence and is
    not treated as production proof. Admission here only means it is not a
    known mock fixture.
    """
    evidence_class = str(record.get("evidence_class") or "").upper()
    source = str(record.get("source") or "").lower()
    mockish = bool(record.get("mock") or record.get("fixture") or evidence_class in _MOCK_CLASSES or source in _MOCK_SOURCES)
    confidence = record.get("confidence")
    if mockish:
        marked = confidence == MOCK_FIXTURE_CONFIDENCE
        return {
            "admitted": False,
            "quarantine": True,
            "reason": (
                "mock Zenodo fixture"
                + (f" with fixture confidence {MOCK_FIXTURE_CONFIDENCE}" if marked else "")
                + " is not production evidence"
            ),
            "confidence": confidence if "confidence" in record else None,
        }
    return {
        "admitted": True,
        "quarantine": False,
        "reason": "record is not marked as a mock or fixture",
        "confidence": confidence if "confidence" in record else None,
    }


def evidence_view(result: Any) -> Dict[str, Any]:
    """Read evidence markers that are actually present. Do not invent values."""
    if isinstance(result, Mapping):
        return dict(result)
    view: Dict[str, Any] = {}
    for key in ("confidence", "mock", "fixture", "evidence_class", "source", "doi"):
        if hasattr(result, key):
            view[key] = getattr(result, key)
    return view
