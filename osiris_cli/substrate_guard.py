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
_HEX_CHARS = frozenset("0123456789abcdefABCDEF")


def _is_sha256_hex(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in _HEX_CHARS for ch in value)


def _read_verified_handle(handle: Any) -> List[Dict[str, Any]]:
    handle.seek(0)
    raw = handle.read().decode("utf-8")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return verify_rows(rows) if rows else []


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        wrote = os.write(fd, view)
        if wrote <= 0:
            raise SubstrateGuardError("temp ledger write wrote no bytes")
        view = view[wrote:]


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


def _durable_write(fd: int, payload: bytes) -> None:
    """Write ``payload`` and make those bytes durable before returning."""
    view = payload
    offset = 0
    while view:
        wrote = os.pwrite(fd, view, offset)
        if wrote <= 0:
            raise SubstrateGuardError("ledger rewrite wrote no bytes")
        view = view[wrote:]
        offset += wrote
    os.ftruncate(fd, offset)
    os.fsync(fd)


def _fd_matches(fd: int, payload: bytes) -> bool:
    """True when ``fd`` is exactly ``payload`` after a durable write."""
    try:
        size = os.lseek(fd, 0, os.SEEK_END)
    except OSError:
        return False
    if size != len(payload):
        return False
    try:
        return os.pread(fd, len(payload), 0) == payload
    except OSError:
        return False


def _try_restore(fd: int, original: bytes) -> bool:
    """Put ``original`` back. False when the live inode still cannot be trusted."""
    try:
        _durable_write(fd, original)
    except (OSError, SubstrateGuardError):
        return False
    return _fd_matches(fd, original)


def _fsync_dir(directory: Path) -> None:
    fd = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def apply_return_chain(local_path: Path, incoming: Sequence[Mapping[str, Any]]) -> str:
    """Replace ``local_path`` only when :func:`accept_return_chain` allows it.

    The read, the ancestry decision, and the rewrite share one ``chain_lock``
    on the ledger inode. ``append_chained`` takes that same lock, so an append
    cannot land between the check and the write. The file is rewritten in
    place: replacing the path would leave the lock on the unlinked inode and
    let a newer tip through. A unique temp file is fsynced first so a crash
    during the rewrite still has the accepted bytes beside the ledger.
    """
    incoming_rows = verify_rows(list(incoming))
    if not incoming_rows:
        raise SubstrateGuardError("incoming chain is empty")
    payload = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in incoming_rows
    ).encode("utf-8")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with chain_lock.locked_tail(str(local_path)) as (handle, _last):
        handle.seek(0)
        original = handle.read()
        text = original.decode("utf-8")
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        local_rows = verify_rows(rows) if rows else []
        decision = accept_return_chain(local_rows, incoming_rows)
        if decision == "identical":
            return decision
        temporary = local_path.with_name(local_path.name + "." + secrets.token_hex(8) + ".tmp")
        fd = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            _write_all(fd, payload)
            os.fsync(fd)
        except BaseException:
            os.close(fd)
            temporary.unlink(missing_ok=True)
            raise
        else:
            os.close(fd)
        _fsync_dir(local_path.parent)
        # The lock fd is O_APPEND. On Linux pwrite appends anyway, so the
        # rewrite goes through a second descriptor on the same inode.
        # The temp file is removed only after that inode matches either the
        # accepted payload or the bytes we read under this lock. A failed
        # rewrite must not delete the only intact copy.
        rewrite = os.open(str(local_path), os.O_RDWR)
        live_intact = False
        try:
            try:
                _durable_write(rewrite, payload)
                if not _fd_matches(rewrite, payload):
                    raise SubstrateGuardError("ledger rewrite did not match the accepted bytes")
                live_intact = True
            except (OSError, SubstrateGuardError) as exc:
                if _try_restore(rewrite, original):
                    live_intact = True
                    raise SubstrateGuardError(
                        "ledger rewrite failed; the previous chain was restored"
                    ) from exc
                raise SubstrateGuardError(
                    "ledger rewrite failed and the previous chain could not be restored; "
                    f"intact accepted bytes remain at {temporary}"
                ) from exc
        finally:
            os.close(rewrite)
            if live_intact:
                temporary.unlink(missing_ok=True)
            _fsync_dir(local_path.parent)
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
    if not experiment_id or not idempotency_key or not _is_sha256_hex(payload_sha256):
        raise SubstrateGuardError("experiment_id, idempotency_key, and a 64-hex payload hash are required")
    path.parent.mkdir(parents=True, exist_ok=True)
    with chain_lock.locked_tail(str(path)) as (handle, _last):
        verified = _read_verified_handle(handle)
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
    if not path.exists():
        raise SubstrateGuardError("execution requires a started row for this experiment and key")
    with chain_lock.locked_tail(str(path)) as (handle, _last):
        verified = _read_verified_handle(handle)
        prior = _rows_for_key(verified, idempotency_key)
        if not prior or prior[-1].get("phase") != "started" or prior[-1].get("experiment_id") != experiment_id:
            raise SubstrateGuardError("execution requires a started row for this experiment and key")
        entry = _next_entry(
            verified[-1],
            {
                "experiment_id": experiment_id,
                "idempotency_key": idempotency_key,
                "payload_sha256": prior[-1]["payload_sha256"],
                "phase": "executed",
                "result": dict(result),
            },
        )
        chain_lock.write_line(handle, entry, sort_keys=True, ensure_ascii=False)
        return entry


def acknowledge(path: Path, experiment_id: str, idempotency_key: str) -> Dict[str, Any]:
    if not path.exists():
        raise SubstrateGuardError("ack requires an executed row for this experiment and key")
    with chain_lock.locked_tail(str(path)) as (handle, _last):
        verified = _read_verified_handle(handle)
        prior = _rows_for_key(verified, idempotency_key)
        if any(row.get("phase") == "acked" and row.get("experiment_id") == experiment_id for row in prior):
            raise SubstrateGuardError("experiment and key are already acknowledged")
        executed = [row for row in prior if row.get("phase") == "executed" and row.get("experiment_id") == experiment_id]
        if not executed:
            raise SubstrateGuardError("ack requires an executed row for this experiment and key")
        entry = _next_entry(
            verified[-1],
            {
                "experiment_id": experiment_id,
                "idempotency_key": idempotency_key,
                "payload_sha256": executed[-1]["payload_sha256"],
                "phase": "acked",
                "result": executed[-1].get("result"),
            },
        )
        chain_lock.write_line(handle, entry, sort_keys=True, ensure_ascii=False)
        return entry


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
