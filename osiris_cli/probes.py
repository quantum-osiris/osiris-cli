"""Read-only checks, SafePathGuard, and SHA-256 pre-answer probes for OSIRIS v6.0.0.

Code chooses and runs every check deterministically before model generation
("Models propose; deterministic code decides, measures, and records"). Every
probe is strictly read-only, blocks credentials/secrets and workspace escapes,
and emits content-hashed provenance receipts.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

HOME = os.path.expanduser("~")
REPOS = ("osiris-cli", "organism_sim", "dnalang-core", "bridge")
MAX_FILE_CHARS = 2500
MAX_FILE_CHECKS = 2
RESULTS_DIR = os.path.join("organism_sim", "results")
MAX_VERDICT_CHARS = 1500

# How people name the pre-registered experiments -> the results-file prefix.
EXPERIMENT_ALIASES = (
    (re.compile(r"\bexperiment[\s_-]*zero\b", re.I), "expzero"),
    (re.compile(r"\bexperiment[\s_-]*one\b", re.I), "expone"),
    (re.compile(r"\bsubstrate[\s_-]*opt[\s_-]*([12])\b", re.I), r"substrate_opt\1"),
)
EXPERIMENT_RE = re.compile(
    r"\b(m[1-7][a-c]?|expzero|expone|substrate_opt[12]|relay(?:_cusum)?|drift|signalling)\b",
    re.I,
)

# Never read, whatever is asked: credentials and private keys.
SECRET_RE = re.compile(
    r"(^|/)(\.env[^/]*|[^/]*\.env|\.ssh|\.gnupg|\.aws|\.config/gcloud|\.netrc|"
    r"[^/]*(token|secret|credential|passw|apikey|api_key)[^/]*|id_[a-z0-9]+|[^/]*\.pem|[^/]*\.key)$",
    re.I,
)

FORBIDDEN_SECRET_RE = re.compile(
    r"(^\.env($|\.)|\.env$|apikey\.json$|credentials|secret|token|passw|api_key|"
    r"^id_rsa|^id_ed25519|\.pem$|\.key$|\.p12$|\.pfx$|"
    r"^\.netrc$|^\.npmrc$|^\.pypirc$|^\.ssh$|^\.gnupg$|^authorized_keys$)",
    re.IGNORECASE,
)

ALLOWED_PROBES = frozenset(
    {"git", "ledger", "trainer", "system", "evidence", "read_file", "sibling_repo"}
)

GIT_KEYWORDS = ("commit", "commits", "diff", "branch", "git", "head", "untracked", "dirty")
LEDGER_KEYWORDS = (
    "chain",
    "ledger",
    "fork",
    "hash",
    "orphan",
    "orphans",
    "exchanges.jsonl",
    "genome_ledger",
    "provenance",
    "tamper",
)
TRAINER_KEYWORDS = (
    "trainer",
    "nclm",
    "loss",
    "losses",
    "bpb",
    "bits/byte",
    "speaking gate",
    "held-out",
    "checkpoint",
    "distill",
)
SYSTEM_KEYWORDS = (
    "ram",
    "memory",
    "cpu",
    "thermal",
    "power",
    "battery",
    "substrate",
    "termux",
    "wsl",
    "node alpha",
    "node beta",
    "aarch64",
    "x86_64",
)
SIBLING_REPOS = ("dnalang-core", "organism_sim")

# Captures workspace paths (e.g. osiris_cli/living.py, ../../.env, apikey.json, id_ed25519)
CANDIDATE_PATH_RE = re.compile(
    r"(?:^|[\s\"'`(\[=])"
    r"("
    r"(?:\.\.?/)+[A-Za-z0-9_./-]+|"
    r"/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+|"
    r"\.(?:env(?:\.[A-Za-z0-9_.-]+)?|netrc|npmrc|pypirc)|"
    r"(?:[A-Za-z0-9_.-]+/)*(?:apikey\.json|id_rsa[A-Za-z0-9_.-]*|id_ed25519[A-Za-z0-9_.-]*)|"
    r"[A-Za-z0-9_./-]+\.(?:py|md|json|jsonl|dna|toml|yaml|yml|sh|txt|cfg|ini|qasm|pem|key)"
    r")"
    r"(?=$|[\s\"'`)\],.:;!?])"
)

HEX_HASH_RE = re.compile(r"^[0-9a-fA-F]{12,128}$")


@dataclass(frozen=True)
class SubstrateInfo:
    node_id: str
    arch: str
    is_termux: bool
    horizon_sec: int
    free_ram_mb: int
    cpu_load_1m: float
    power_status: str

    @property
    def node_role(self) -> str:
        return self.node_id

    @property
    def horizon_s(self) -> float:
        return float(self.horizon_sec)


@dataclass(frozen=True)
class ChainTailReport:
    ledger_path: str
    exists: bool
    intact: bool
    inspected_count: int
    tail_hash: str
    detail: str
    status: str = ""
    crlf_detected: bool = False
    orphan_runs: int = 0
    error: Optional[str] = None

    @property
    def resolved_path(self) -> str:
        return self.ledger_path

    @property
    def tail_head(self) -> str:
        return self.tail_hash[:12]

    def _effective_status(self) -> str:
        if self.status:
            return self.status
        if not self.exists:
            return "EMPTY"
        if self.intact:
            return "INTACT" if self.inspected_count > 0 else "EMPTY"
        return "TAMPERED"

    def _as_compat_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "ledger_path": self.ledger_path,
            "resolved_path": self.ledger_path,
            "exists": self.exists,
            "intact": self.intact,
            "status": self._effective_status(),
            "inspected_count": self.inspected_count,
            "tail_count": self.inspected_count,
            "tail_hash": self.tail_hash,
            "tail_head": self.tail_head,
            "head": self.tail_hash,
            "crlf_detected": self.crlf_detected,
            "orphan_runs": self.orphan_runs,
            "detail": self.detail,
        }
        if self.error:
            out["error"] = self.error
        return out

    def __getitem__(self, key: str) -> Any:
        return self._as_compat_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._as_compat_dict().get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self._as_compat_dict()

    def keys(self):
        return self._as_compat_dict().keys()


@dataclass(frozen=True)
class ProbePlan:
    run_git: bool
    run_ledger: bool
    run_trainer: bool
    run_system: bool
    sibling_targets: tuple[str, ...]
    file_targets: tuple[str, ...]


@dataclass(frozen=True)
class ProbeReceipt:
    probe: str
    target: str
    exit_code: int
    output_sha256: str
    summary: str
    blocked: bool = False
    duration_ms: float = 0.0

    def to_ledger_dict(self) -> dict[str, object]:
        """Serialize compact cryptographic receipt for exchanges.jsonl."""
        return {
            "probe": self.probe,
            "target": self.target,
            "exit_code": self.exit_code,
            "sha256": self.output_sha256,
            "blocked": self.blocked,
            "bytes": len(self.summary.encode("utf-8")),
            "duration_ms": self.duration_ms,
        }


def _make_receipt(
    probe: str,
    target: str,
    exit_code: int,
    summary: str,
    blocked: bool = False,
    duration_ms: float = 0.0,
) -> ProbeReceipt:
    if probe not in ALLOWED_PROBES:
        raise ValueError(f"Unsupported probe type: {probe!r}")
    normalized = summary.replace("\r\n", "\n").replace("\r", "\n")
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return ProbeReceipt(
        probe=probe,
        target=target,
        exit_code=exit_code,
        output_sha256=digest,
        summary=normalized,
        blocked=blocked,
        duration_ms=round(float(duration_ms), 3),
    )


def _run(args: List[str], timeout: float = 10.0) -> str:
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or r.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"(failed: {type(e).__name__})"


def resolve_osiris_home(
    explicit_home: Path | str | None = None,
    *,
    environ: Optional[Dict[str, str]] = None,
    home: Path | str | None = None,
) -> Path:
    """Resolve active OSIRIS state directory via $OSIRIS_HOME or ~/.osiris."""
    if explicit_home is not None:
        return Path(explicit_home).expanduser().resolve()
    env_map = os.environ if environ is None else environ
    env_val = str(env_map.get("OSIRIS_HOME", "")).strip()
    if env_val:
        return Path(env_val).expanduser().resolve()
    base_home = Path(home).expanduser().resolve() if home is not None else Path.home().resolve()
    return (base_home / ".osiris").resolve()


def resolve_sibling_repo(
    repo_name: str,
    workspace_root: Path | str | None = None,
    osiris_home: Path | str | None = None,
    *,
    repo_root: Path | str | None = None,
    home: Path | str | None = None,
) -> Path | None:
    """Locate sibling repositories (e.g. dnalang-core) across dual-substrate layouts."""
    clean_name = repo_name.strip().strip("/")
    if not clean_name or "/" in clean_name or ".." in clean_name:
        return None

    home_dir = Path(home).expanduser().resolve() if home is not None else Path.home().resolve()
    active_osiris = resolve_osiris_home(osiris_home, home=home_dir)
    ws_arg = workspace_root if workspace_root is not None else repo_root
    ws = Path(ws_arg).expanduser().resolve() if ws_arg else Path.cwd().resolve()

    candidates = [
        home_dir / "osiris-repos" / clean_name,
        ws.parent / clean_name,
        active_osiris / "src" / clean_name,
        home_dir / "osiris-assessment" / "work" / clean_name,
        home_dir / clean_name,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return None


class SafePathGuard:
    """Confines read-only file inspection strictly inside canonical workspace roots."""

    def __init__(
        self,
        workspace_root: Path | str,
        extra_allowed_roots: Sequence[Path | str] = (),
        max_read_bytes: int = 4096,
    ) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.allowed_roots = (
            self.workspace_root,
            *(Path(r).expanduser().resolve() for r in extra_allowed_roots),
        )
        self.max_read_bytes = max(256, max_read_bytes)

    @staticmethod
    def is_forbidden_segment(segment: str) -> bool:
        return bool(FORBIDDEN_SECRET_RE.search(segment))

    def _find_containing_root(self, resolved_path: Path) -> Path | None:
        for root in self.allowed_roots:
            if resolved_path.is_relative_to(root):
                return root
        return None

    def resolve(self, raw_path: str | Path) -> Path:
        """Validate and canonicalize raw_path; raise PermissionError on any violation."""
        raw_str = str(raw_path).strip()
        if not raw_str or "\x00" in raw_str:
            raise PermissionError("Empty or null-byte path rejected by SafePathGuard")
        if "\\" in raw_str:
            raise PermissionError(f"Backslash path rejected by SafePathGuard: {raw_str!r}")
        if raw_str.startswith("/") or raw_str.startswith("~"):
            raise PermissionError(f"Non-relative path rejected by SafePathGuard: {raw_str!r}")

        rel_obj = Path(raw_str)
        for part in rel_obj.parts:
            if part == "..":
                raise PermissionError(f"Directory traversal blocked by SafePathGuard: {raw_str!r}")
            if self.is_forbidden_segment(part):
                raise PermissionError(f"Credential/secret path blocked by SafePathGuard: {raw_str!r}")

        candidate = self.workspace_root / rel_obj
        resolved = candidate.resolve(strict=False)

        containing_root = self._find_containing_root(resolved)
        if containing_root is None:
            raise PermissionError(
                f"Path or symlink escapes workspace ({self.workspace_root}): {raw_str!r}"
            )

        rel_resolved = resolved.relative_to(containing_root)
        for part in rel_resolved.parts:
            if self.is_forbidden_segment(part):
                raise PermissionError(
                    f"Symlink target resolves to forbidden secret ({part}): {raw_str!r}"
                )

        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(f"Workspace file not found: {raw_str}")

        return resolved

    def read_text_bounded(self, raw_path: str | Path) -> tuple[Path, str]:
        """Safely read up to max_read_bytes of UTF-8 text from a verified workspace file."""
        safe_file = self.resolve(raw_path)
        with safe_file.open("rb") as fh:
            chunk = fh.read(self.max_read_bytes)
        if b"\x00" in chunk:
            raise PermissionError(f"Binary file read blocked by SafePathGuard: {safe_file.name}")
        text = chunk.decode("utf-8", errors="replace")
        return safe_file, text


def verify_chain(path: str) -> Dict[str, object]:
    """Recompute the exchange log's hash chain (the same rule Osiris._record writes).

    Every entry must hash to its own content, and its `prev` must be the genesis hash or the hash of an earlier
    entry; otherwise the chain is broken there (an edited, inserted-with-a-false-link or removed entry). A `prev`
    that names an earlier entry other than the one just before is a fork: two writers appended from different heads
    (2026-10-07, before the writer read the file's last entry). Forks are listed, never hidden; they are not breaks,
    because no entry was altered and every link resolves.

    The result names the file it read (``resolved_path``, so a console reading the wrong data home shows it) and a
    ``status`` with an ``exit_code`` (E-24): INTACT 0; EMPTY 0, no log or no entries yet; TAMPERED 1, an entry fails
    its hash or links nowhere; UNREADABLE 2, the file or a line of it cannot be read or parsed."""
    genesis = "0" * 64
    resolved = os.path.abspath(os.path.expanduser(path))
    n, prev, forks, seen = 0, genesis, [], {genesis}

    def result(status: str, broken_at: Optional[int] = None, error: Optional[str] = None) -> Dict[str, object]:
        return {
            "resolved_path": resolved,
            "status": status,
            "intact": status in ("INTACT", "EMPTY"),
            "exit_code": {"INTACT": 0, "EMPTY": 0, "TAMPERED": 1, "UNREADABLE": 2}[status],
            "entries": n,
            "broken_at": broken_at,
            "forks": forks,
            "head": prev[:12],
            "error": error,
        }

    if not os.path.exists(resolved):
        return result("EMPTY")
    try:
        with open(resolved, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except ValueError as e:
                    return result("UNREADABLE", i, f"line {i + 1} is not JSON ({e})")
                if not isinstance(entry, dict):
                    return result("TAMPERED", i, f"line {i + 1} is not an entry")
                h = entry.pop("hash", None)
                body = json.dumps(entry, sort_keys=True, ensure_ascii=False)
                if hashlib.sha256(body.encode()).hexdigest() != h:
                    return result("TAMPERED", i, f"entry {i} does not hash to its recorded hash")
                if entry.get("prev") not in seen:
                    return result("TAMPERED", i, f"entry {i} links to no earlier entry")
                if entry.get("prev") != prev:
                    forks.append(i)
                seen.add(h)
                prev, n = h, n + 1
    except (OSError, UnicodeDecodeError) as e:
        return result("UNREADABLE", None, f"{type(e).__name__}: {e}")
    return result("INTACT" if n else "EMPTY")


def compute_chained_entry_hash(prev_hash: str, payload: dict[str, object]) -> str:
    """Compute canonical SHA-256 entry hash over prev_hash + canonical JSON payload."""
    clean_payload = {
        k: v for k, v in payload.items() if k not in ("hash", "entry_hash", "sha256")
    }
    clean_payload["prev_hash"] = prev_hash
    canonical = json.dumps(clean_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _extract_entry_hashes(record: dict[str, object]) -> tuple[str | None, str | None]:
    curr = None
    for key in ("entry_hash", "hash", "sha256", "record_hash"):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            curr = val.strip()
            break

    prev = None
    for key in ("prev", "prev_hash", "previous_hash", "parent_hash"):
        val = record.get(key)
        if isinstance(val, str):
            prev = val.strip()
            break

    return prev, curr


def _verify_row_self_hash(row: dict[str, object], prev_h: str | None, curr_h: str) -> bool:
    """Verify row self-digest across living.py, dnalang/genome_ledger, and canonical_v1 schemas."""
    if row.get("hash_mode") == "canonical_v1" and prev_h is not None:
        return compute_chained_entry_hash(prev_h, row) == curr_h
    if "hash" in row and "prev" in row:
        # 1. living.py exchanges.jsonl formula: sha256(json.dumps(entry_without_hash, sort_keys=True))
        entry_no_hash = {k: v for k, v in row.items() if k != "hash"}
        living_body = json.dumps(entry_no_hash, sort_keys=True, ensure_ascii=False)
        if hashlib.sha256(living_body.encode("utf-8")).hexdigest() == curr_h:
            return True
        # 2. dnalang-core / genome_ledger formula: sha256(prev + compact canonical json without prev/hash)
        if prev_h is not None:
            dnalang_body = {k: v for k, v in row.items() if k not in ("prev", "hash")}
            compact = json.dumps(dnalang_body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            if hashlib.sha256((prev_h + compact).encode("utf-8")).hexdigest() == curr_h:
                return True
        return False
    return True


def _read_tail_lines(path: Path, max_lines: int = 8, chunk_bytes: int = 65536) -> list[str]:
    """Read last max_lines from a JSONL file in O(1) seek time."""
    file_size = path.stat().st_size
    if file_size == 0:
        return []

    read_len = min(file_size, max(8192, chunk_bytes, max_lines * 8192))
    with path.open("rb") as fh:
        fh.seek(file_size - read_len, os.SEEK_SET)
        raw = fh.read(read_len)

    if b"\r" in raw:
        raise ValueError(f"CRLF byte detected in ledger {path.name}")

    text = raw.decode("utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if file_size > read_len and len(lines) > 1:
        lines = lines[1:]
    return lines[-max_lines:]


def verify_chain_tail(
    ledger_path: Path | str,
    tail_lines: int = 20,
    *,
    tail_window: int | None = None,
    osiris_home: Path | str | None = None,
) -> ChainTailReport:
    """Fast O(1) reverse-seek verification of the last N entries of a JSONL ledger.

    Returns a ChainTailReport supporting both attribute access (.intact, .inspected_count,
    .tail_hash, .crlf_detected, .orphan_runs, .tail_head) and legacy dict indexing
    (["intact"], ["tail_count"], ["head"], ["status"], ["crlf_detected"]).
    Compatible with living.py exchanges.jsonl ("prev"/"hash"), dnalang-core ledgers,
    and canonical_v1 ("prev_hash"/"entry_hash") rows. Rejects CRLF bytes fail-closed.
    """
    window = tail_window if tail_window is not None else tail_lines
    resolved = Path(ledger_path).expanduser().resolve()
    orphans = _count_orphan_runs(Path(osiris_home).expanduser().resolve())[0] if osiris_home is not None else 0
    if not resolved.exists():
        return ChainTailReport(
            ledger_path=str(resolved),
            exists=False,
            intact=True,
            inspected_count=0,
            tail_hash="0" * 12,
            detail="not initialized yet",
            status="EMPTY",
            crlf_detected=False,
            orphan_runs=orphans,
        )
    if not resolved.is_file():
        return ChainTailReport(
            ledger_path=str(resolved),
            exists=False,
            intact=False,
            inspected_count=0,
            tail_hash="INVALID",
            detail="ledger path is not a regular file",
            status="UNREADABLE",
            crlf_detected=False,
            orphan_runs=orphans,
            error="ledger path is not a regular file",
        )

    try:
        lines = _read_tail_lines(resolved, max_lines=max(2, int(window)))
    except ValueError as exc:
        is_crlf = "CRLF" in str(exc)
        return ChainTailReport(
            ledger_path=str(resolved),
            exists=True,
            intact=False,
            inspected_count=0,
            tail_hash="CORRUPT",
            detail=f"tail read error: {exc}",
            status="TAMPERED" if is_crlf else "UNREADABLE",
            crlf_detected=is_crlf,
            orphan_runs=orphans,
            error=str(exc),
        )
    except (OSError, UnicodeDecodeError) as exc:
        return ChainTailReport(
            ledger_path=str(resolved),
            exists=True,
            intact=False,
            inspected_count=0,
            tail_hash="CORRUPT",
            detail=f"tail read error: {exc}",
            status="UNREADABLE",
            crlf_detected=False,
            orphan_runs=orphans,
            error=str(exc),
        )

    if not lines:
        return ChainTailReport(
            ledger_path=str(resolved),
            exists=True,
            intact=True,
            inspected_count=0,
            tail_hash="0" * 12,
            detail="0 entries",
            status="EMPTY",
            crlf_detected=False,
            orphan_runs=orphans,
        )

    parsed: list[dict[str, object]] = []
    for idx, line in enumerate(lines):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            return ChainTailReport(
                ledger_path=str(resolved),
                exists=True,
                intact=False,
                inspected_count=idx,
                tail_hash="CORRUPT_JSON",
                detail=f"malformed JSON in tail line {idx + 1}: {exc.msg}",
                status="UNREADABLE",
                crlf_detected=False,
                orphan_runs=orphans,
                error=f"malformed JSON in tail line {idx + 1}: {exc.msg}",
            )
        if not isinstance(obj, dict):
            return ChainTailReport(
                ledger_path=str(resolved),
                exists=True,
                intact=False,
                inspected_count=idx,
                tail_hash="INVALID_ROW",
                detail="JSONL entry is not an object",
                status="TAMPERED",
                crlf_detected=False,
                orphan_runs=orphans,
                error="JSONL entry is not an object",
            )
        parsed.append(obj)

    prev_seen_hash: str | None = None
    for idx, row in enumerate(parsed):
        prev_h, curr_h = _extract_entry_hashes(row)
        if curr_h is None or not HEX_HASH_RE.match(curr_h):
            return ChainTailReport(
                ledger_path=str(resolved),
                exists=True,
                intact=False,
                inspected_count=idx + 1,
                tail_hash="MISSING_HASH",
                detail=f"missing or malformed entry hash at tail offset {idx}",
                status="TAMPERED",
                crlf_detected=False,
                orphan_runs=orphans,
                error=f"missing or malformed entry hash at tail offset {idx}",
            )

        if not _verify_row_self_hash(row, prev_h, curr_h):
            return ChainTailReport(
                ledger_path=str(resolved),
                exists=True,
                intact=False,
                inspected_count=idx + 1,
                tail_hash=curr_h[:12],
                detail=f"entry hash mismatch at tail offset {idx}",
                status="TAMPERED",
                crlf_detected=False,
                orphan_runs=orphans,
                error=f"entry hash mismatch at tail offset {idx}",
            )

        if idx > 0 and prev_seen_hash is not None:
            if prev_h != prev_seen_hash:
                return ChainTailReport(
                    ledger_path=str(resolved),
                    exists=True,
                    intact=False,
                    inspected_count=idx + 1,
                    tail_hash=curr_h[:12],
                    detail=(
                        f"chain fork/break at tail offset {idx}: "
                        f"prev={str(prev_h)[:12]} != {prev_seen_hash[:12]}"
                    ),
                    status="TAMPERED",
                    crlf_detected=False,
                    orphan_runs=orphans,
                    error=f"chain fork/break at tail offset {idx}",
                )
        prev_seen_hash = curr_h

    final_hash = prev_seen_hash or ("0" * 12)
    return ChainTailReport(
        ledger_path=str(resolved),
        exists=True,
        intact=True,
        inspected_count=len(parsed),
        tail_hash=final_hash[:12],
        detail=f"intact ({len(parsed)} tail entries verified, head={final_hash[:12]})",
        status="INTACT",
        crlf_detected=False,
        orphan_runs=orphans,
    )


def verify_chain_tail_report(
    path: Path | str,
    tail_lines: int = 20,
    *,
    osiris_home: Path | str | None = None,
) -> ChainTailReport:
    """Typed O(1) tail verification alias preserving capability-reconciliation-s1 callers."""
    return verify_chain_tail(path, tail_lines=tail_lines, osiris_home=osiris_home)


def safe_path(raw: str, base: str = HOME) -> Optional[str]:
    """A readable text file under the base directory that is not a secret."""
    if not raw or "\x00" in raw or "\\" in raw:
        return None
    p = os.path.realpath(os.path.expanduser(raw if raw.startswith(("~", "/")) else os.path.join(base, raw)))
    base_real = os.path.realpath(base)
    if not (p == base_real or p.startswith(base_real.rstrip("/") + "/")):
        return None
    rel = os.path.relpath(p, base_real)
    parts = rel.split(os.sep)
    if any(
        SECRET_RE.search("/".join(parts[: i + 1])) or FORBIDDEN_SECRET_RE.search(parts[i])
        for i in range(len(parts))
    ):
        return None
    return p if os.path.isfile(p) else None


def _read_free_ram_mb(meminfo_path: Path = Path("/proc/meminfo")) -> int:
    try:
        if meminfo_path.is_file():
            content = meminfo_path.read_text(encoding="utf-8", errors="replace")
            for key in ("MemAvailable:", "MemFree:"):
                for line in content.splitlines():
                    if line.startswith(key):
                        parts = line.split()
                        if len(parts) >= 2 and parts[1].isdigit():
                            return int(parts[1]) // 1024
    except OSError:
        pass

    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if isinstance(pages, int) and isinstance(page_size, int) and pages > 0 and page_size > 0:
            return (pages * page_size) // (1024 * 1024)
    except (AttributeError, OSError, ValueError):
        pass

    return 0


def _read_cpu_load_1m() -> float:
    try:
        return round(float(os.getloadavg()[0]), 2)
    except (AttributeError, OSError):
        return 0.0


def _probe_power_supply(power_dir: Path = Path("/sys/class/power_supply")) -> str:
    try:
        if not power_dir.exists():
            return "unreadable_attended_only"
        entries = list(power_dir.iterdir())
        if not entries:
            return "unreadable_attended_only"
        for entry in entries:
            status_file = entry / "status"
            if status_file.is_file():
                val = status_file.read_text(encoding="utf-8", errors="replace").strip()
                if val:
                    return f"readable:{val.lower()}"
        return "unreadable_attended_only"
    except OSError:
        return "unreadable_attended_only"


def probe_substrate(
    power_dir: Path = Path("/sys/class/power_supply"),
    meminfo_path: Path = Path("/proc/meminfo"),
    *,
    power_supply_dir: Path | str | None = None,
) -> SubstrateInfo:
    """Detect Node Alpha (Termux ARM64) vs Node Beta (Workstation x86_64) deterministically."""
    eff_power = Path(power_supply_dir) if power_supply_dir is not None else Path(power_dir)
    arch = platform.machine().lower() or "unknown"
    prefix = os.environ.get("PREFIX", "")
    is_termux = "com.termux" in prefix or arch in ("aarch64", "arm64")
    node_id = "NODE_ALPHA_EDGE" if is_termux else "NODE_BETA_WORKSTATION"
    horizon_sec = 300 if is_termux else 1800

    return SubstrateInfo(
        node_id=node_id,
        arch=arch,
        is_termux=is_termux,
        horizon_sec=horizon_sec,
        free_ram_mb=_read_free_ram_mb(meminfo_path=meminfo_path),
        cpu_load_1m=_read_cpu_load_1m(),
        power_status=_probe_power_supply(power_dir=eff_power),
    )


def _count_orphan_runs(osiris_home: Path) -> tuple[int, list[str]]:
    """Detect started run records lacking a terminal end/verdict marker."""
    runs_dir = osiris_home / "runs"
    if not runs_dir.is_dir():
        return 0, []

    orphans: list[str] = []
    for item in sorted(runs_dir.glob("*.json")):
        try:
            data = json.loads(item.read_text(encoding="utf-8", errors="replace"))
            if isinstance(data, dict):
                has_end = any(
                    data.get(k) is not None
                    for k in ("ended_at", "end_ts", "verdict", "completed", "exit_code")
                )
                status = str(data.get("status", "")).lower()
                if not has_end and status not in ("completed", "passed", "failed", "aborted"):
                    orphans.append(item.name)
        except (OSError, json.JSONDecodeError):
            orphans.append(item.name)

    return len(orphans), orphans


def probe_git(workspace_root: Path | str | None = None) -> ProbeReceipt:
    """Execute read-only git status, branch, HEAD SHA, and recent log probe."""
    ws = Path(workspace_root).expanduser().resolve() if workspace_root else Path.cwd().resolve()

    def _git(*args: str) -> tuple[int, str]:
        try:
            proc = subprocess.run(
                ["git", "-C", str(ws), *args],
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
            out = (proc.stdout if proc.returncode == 0 else proc.stderr).strip()
            return proc.returncode, out
        except (OSError, subprocess.SubprocessError) as exc:
            return 1, str(exc)

    rc_sha, sha = _git("rev-parse", "--short", "HEAD")
    rc_branch, branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    if rc_sha != 0 and rc_branch != 0:
        return _make_receipt(
            "git",
            str(ws),
            1,
            f"git probe failed in {ws}: {sha or branch or 'not a git repository'}",
        )

    _, status_out = _git("status", "--short")
    _, log_out = _git("log", "--oneline", "-n", "3")
    dirty_lines = [ln for ln in status_out.splitlines() if ln.strip()]
    clean_tag = "CLEAN" if not dirty_lines else f"DIRTY ({len(dirty_lines)} paths)"

    summary_lines = [
        f"repo: {ws}",
        f"head: {branch or 'HEAD'} @ {sha or 'UNBORN'} [{clean_tag}]",
    ]
    if dirty_lines:
        summary_lines.append("modified: " + "; ".join(dirty_lines[:8]))
    if log_out:
        summary_lines.append("recent_commits:\n" + log_out)

    return _make_receipt("git", "HEAD", 0, "\n".join(summary_lines))


def probe_ledger(osiris_home: Path | str | None = None) -> ProbeReceipt:
    """Verify canonical OSIRIS JSONL ledgers in O(1) tail time and check orphan runs."""
    home = resolve_osiris_home(osiris_home)
    ledgers = (
        ("exchanges", home / "living" / "exchanges.jsonl"),
        ("genome", home / "genome_ledger.jsonl"),
        ("retrieval", home / "retrieval" / "ledger.jsonl"),
    )

    lines: list[str] = [f"osiris_home: {home}"]
    all_ok = True
    for label, path in ledgers:
        report = verify_chain_tail(path)
        if not report.intact:
            all_ok = False
        state = "INTACT" if report.intact else "BROKEN"
        lines.append(f"{label}_ledger: {state} ({report.detail})")

    orphan_count, orphan_names = _count_orphan_runs(home)
    if orphan_count > 0:
        all_ok = False
        lines.append(f"orphan_runs: {orphan_count} ({', '.join(orphan_names[:5])})")
    else:
        lines.append("orphan_runs: 0")

    return _make_receipt("ledger", str(home), 0 if all_ok else 1, "\n".join(lines))


def probe_trainer(osiris_home: Path | str | None = None) -> ProbeReceipt:
    """Inspect NCLM trainer state, step count, and SYS-026 held-out Speaking Gate metrics."""
    home = resolve_osiris_home(osiris_home)
    living_dir = home / "living"

    state_candidates = (
        living_dir / "trainer_state.json",
        living_dir / "nclm_state.json",
        living_dir / "stats.json",
        home / "trainer_state.json",
    )
    trainer_meta: dict[str, object] = {}
    state_source = "none"
    for cand in state_candidates:
        if cand.is_file():
            try:
                loaded = json.loads(cand.read_text(encoding="utf-8", errors="replace"))
                if isinstance(loaded, dict):
                    trainer_meta = loaded
                    state_source = cand.name
                    break
            except (OSError, json.JSONDecodeError):
                continue

    exchanges_path = living_dir / "exchanges.jsonl"
    total_exchanges = 0
    held_out_count = 0
    if exchanges_path.is_file():
        try:
            with exchanges_path.open("rb") as fh:
                for idx, raw_line in enumerate(fh, start=1):
                    if raw_line.strip():
                        total_exchanges += 1
                        if idx % 5 == 0:
                            held_out_count += 1
        except OSError:
            pass

    step = trainer_meta.get("step", 0)
    held_out_bpb = trainer_meta.get("held_out_bpb", None)
    unigram_bpb = trainer_meta.get("unigram_bpb", None)

    gate_open = (
        isinstance(held_out_bpb, (int, float))
        and isinstance(unigram_bpb, (int, float))
        and held_out_count >= 30
        and float(held_out_bpb) <= 2.0
        and float(held_out_bpb) < float(unigram_bpb)
    )
    gate_str = "OPEN" if gate_open else "CLOSED (requires <= 2.0 bpb on >= 30 held-out & < unigram)"

    summary = (
        f"trainer_state_source: {state_source}\n"
        f"nclm_step: {step}\n"
        f"exchanges_recorded: {total_exchanges} (held_out_every_5th={held_out_count})\n"
        f"held_out_bpb: {held_out_bpb if held_out_bpb is not None else 'unmeasured'} "
        f"(unigram={unigram_bpb if unigram_bpb is not None else 'unmeasured'})\n"
        f"sys026_speaking_gate: {gate_str}"
    )
    return _make_receipt("trainer", "osiris.nclm", 0, summary)


def probe_system(
    workspace_root: Path | str | None = None,
    osiris_home: Path | str | None = None,
) -> ProbeReceipt:
    """Report substrate profile, RAM/CPU telemetry, power sensor status, and sibling repos."""
    sub = probe_substrate()
    home = resolve_osiris_home(osiris_home)
    ws = Path(workspace_root).expanduser().resolve() if workspace_root else Path.cwd().resolve()

    siblings: list[str] = []
    for repo_name in SIBLING_REPOS:
        found = resolve_sibling_repo(repo_name, workspace_root=ws, osiris_home=home)
        siblings.append(f"{repo_name}={found if found else 'NOT_FOUND'}")

    summary = (
        f"substrate: {sub.node_id} ({sub.arch}, termux={sub.is_termux}, horizon={sub.horizon_sec}s)\n"
        f"python: {platform.python_version()}\n"
        f"telemetry: free_ram_mb={sub.free_ram_mb}, cpu_load_1m={sub.cpu_load_1m}, power={sub.power_status}\n"
        f"osiris_home: {home}\n"
        f"siblings: {', '.join(siblings)}"
    )
    return _make_receipt("system", sub.node_id, 0, summary)


def probe_evidence(
    osiris_home: Path | str | None = None,
    workspace_root: Path | str | None = None,
) -> ProbeReceipt:
    """Audit L3 read-only evidence assets: notes, kept facts, and claims module."""
    home = resolve_osiris_home(osiris_home)
    ws = Path(workspace_root).expanduser().resolve() if workspace_root else Path.cwd().resolve()

    notes_dir = home / "notes"
    note_count = len(list(notes_dir.glob("*.md"))) if notes_dir.is_dir() else 0

    facts_file = home / "living" / "facts.jsonl"
    fact_count = 0
    if facts_file.is_file():
        try:
            fact_count = sum(
                1
                for ln in facts_file.read_text(encoding="utf-8", errors="replace").splitlines()
                if ln.strip()
            )
        except OSError:
            fact_count = 0

    claims_file = ws / "osiris_cli" / "claims.py"
    claims_present = claims_file.is_file()

    summary = (
        f"notes_md_count: {note_count}\n"
        f"kept_facts_count: {fact_count}\n"
        f"claims_module_present: {claims_present}"
    )
    return _make_receipt("evidence", str(home), 0, summary)


def probe_sibling_repository(
    repo_name: str,
    workspace_root: Path | str | None = None,
    osiris_home: Path | str | None = None,
) -> ProbeReceipt:
    """Inspect sibling package (dnalang-core / organism_sim) availability and HEAD commit."""
    resolved = resolve_sibling_repo(repo_name, workspace_root=workspace_root, osiris_home=osiris_home)
    if resolved is None:
        return _make_receipt(
            "sibling_repo",
            repo_name,
            1,
            f"sibling repo {repo_name!r} not found in ~/osiris-repos, workspace parent, or $OSIRIS_HOME/src",
        )

    ledger_py = resolved / "dnalang" / "ledger.py"
    git_receipt = probe_git(resolved)
    summary = (
        f"sibling_repo: {repo_name} -> {resolved}\n"
        f"dnalang_ledger_py: {ledger_py.is_file()}\n"
        f"{git_receipt.summary}"
    )
    return _make_receipt("sibling_repo", repo_name, 0, summary)


def extract_candidate_paths(user_msg: str, max_files: int = 5) -> tuple[str, ...]:
    """Deterministically extract candidate file/secret/traversal mentions from prompt text."""
    scrubbed = re.sub(r"https?://\S+", " ", user_msg)
    scrubbed = re.sub(r"\bdoi\s*:\s*10\.\d{4,9}/\S+", " ", scrubbed, flags=re.I)
    seen: list[str] = []
    for match in CANDIDATE_PATH_RE.finditer(scrubbed):
        token = match.group(1).strip()
        if not token or token in ("exchanges.jsonl", "genome_ledger.jsonl"):
            continue
        if token not in seen:
            seen.append(token)
        if len(seen) >= max_files:
            break
    return tuple(seen)


def plan_probes(user_msg: str, max_files: int = 5) -> ProbePlan:
    """Phase 1: Zero-LLM deterministic intent-to-probe planner (< 5 ms)."""
    lower = user_msg.lower()
    words = set(re.findall(r"[a-z0-9_./-]+", lower))

    run_git = any(k in words for k in GIT_KEYWORDS)
    run_ledger = any(k in lower if "." in k else k in words for k in LEDGER_KEYWORDS)
    run_trainer = any(
        k in lower if ("/" in k or " " in k or "-" in k) else k in words
        for k in TRAINER_KEYWORDS
    )
    run_system = any(k in lower if " " in k else k in words for k in SYSTEM_KEYWORDS)

    siblings = tuple(
        name for name in SIBLING_REPOS if name in lower or name.replace("-", "_") in lower
    )
    files = extract_candidate_paths(user_msg, max_files=max_files)

    return ProbePlan(
        run_git=run_git,
        run_ledger=run_ledger,
        run_trainer=run_trainer,
        run_system=run_system,
        sibling_targets=siblings,
        file_targets=files,
    )


def plan_and_run_probes(
    user_msg: str,
    workspace_root: Path | str | None = None,
    osiris_home: Path | str | None = None,
    max_file_bytes: int = 4096,
    max_files: int = 5,
) -> list[ProbeReceipt]:
    """Phase 1 + Phase 2: Plan deterministic read-only probes and return SHA-256 receipts."""
    ws = Path(workspace_root).expanduser().resolve() if workspace_root else Path.cwd().resolve()
    home = resolve_osiris_home(osiris_home)
    guard = SafePathGuard(ws, max_read_bytes=max_file_bytes)
    plan = plan_probes(user_msg, max_files=max_files)

    receipts: list[ProbeReceipt] = []
    if plan.run_git:
        receipts.append(probe_git(ws))
    if plan.run_ledger:
        receipts.append(probe_ledger(home))
    if plan.run_trainer:
        receipts.append(probe_trainer(home))
    if plan.run_system:
        receipts.append(probe_system(workspace_root=ws, osiris_home=home))

    for sib in plan.sibling_targets:
        receipts.append(probe_sibling_repository(sib, workspace_root=ws, osiris_home=home))

    for raw_target in plan.file_targets:
        try:
            safe_file, content = guard.read_text_bounded(raw_target)
            rel_label = str(safe_file.relative_to(ws))
            receipts.append(_make_receipt("read_file", rel_label, 0, content, blocked=False))
        except PermissionError as exc:
            receipts.append(
                _make_receipt(
                    "read_file",
                    raw_target,
                    13,
                    f"[SAFEPATHGUARD BLOCKED] {exc}",
                    blocked=True,
                )
            )
        except FileNotFoundError:
            continue

    return receipts


def run_check_command(
    subcommand: str = "all",
    target: str | None = None,
    workspace_root: Path | str | None = None,
    osiris_home: Path | str | None = None,
) -> list[ProbeReceipt]:
    """Deterministic handler for `/check [trainer|git|ledger|system|evidence] [path]`."""
    ws = Path(workspace_root).expanduser().resolve() if workspace_root else Path.cwd().resolve()
    home = resolve_osiris_home(osiris_home)
    sub = (subcommand or "all").strip().lower()

    if sub in ("all", ""):
        results = [
            probe_ledger(home),
            probe_git(ws),
            probe_trainer(home),
            probe_system(workspace_root=ws, osiris_home=home),
            probe_evidence(osiris_home=home, workspace_root=ws),
        ]
    elif sub == "git":
        results = [probe_git(ws)]
    elif sub == "ledger":
        results = [probe_ledger(home)]
    elif sub == "trainer":
        results = [probe_trainer(home)]
    elif sub == "system":
        results = [probe_system(workspace_root=ws, osiris_home=home)]
    elif sub == "evidence":
        results = [probe_evidence(osiris_home=home, workspace_root=ws)]
    else:
        target = subcommand
        results = []

    if target:
        guard = SafePathGuard(ws)
        try:
            safe_file, content = guard.read_text_bounded(target)
            rel_label = str(safe_file.relative_to(ws))
            results.append(_make_receipt("read_file", rel_label, 0, content, blocked=False))
        except PermissionError as exc:
            results.append(
                _make_receipt(
                    "read_file",
                    target,
                    13,
                    f"[SAFEPATHGUARD BLOCKED] {exc}",
                    blocked=True,
                )
            )
        except FileNotFoundError as exc:
            results.append(_make_receipt("read_file", target, 2, str(exc), blocked=False))

    return results


def format_probe_receipts_for_prompt(
    receipts: Sequence[ProbeReceipt],
    max_chars: int = 3200,
) -> str:
    """Format verified probe receipts into a bounded grounding block for mentor prompts."""
    if not receipts:
        return ""

    blocks: list[str] = ["--- DETERMINISTIC PRE-ANSWER PROBES (CODE-VERIFIED) ---"]
    used = len(blocks[0])

    for rec in receipts:
        header = (
            f"[PROBE={rec.probe} TARGET={rec.target} RC={rec.exit_code} "
            f"SHA256={rec.output_sha256[:12]} BLOCKED={rec.blocked}]"
        )
        remaining = max_chars - used - len(header) - 4
        if remaining <= 32:
            break
        snippet = rec.summary if len(rec.summary) <= remaining else rec.summary[: remaining - 3] + "..."
        section = f"{header}\n{snippet}"
        blocks.append(section)
        used += len(section) + 2

    return "\n\n".join(blocks)


def receipts_from_legacy_results(results: Sequence[Dict[str, Any]]) -> list[dict[str, object]]:
    """Build compact SHA-256 ledger receipt dicts from legacy `Probes.run()` outputs."""
    out: list[dict[str, object]] = []
    for r in results:
        name = str(r.get("name", "unknown"))
        cmd = str(r.get("cmd", name))
        body = str(r.get("out", ""))
        blocked = "not read:" in body or "SAFEPATHGUARD BLOCKED" in body
        rc = int(r["exit_code"]) if "exit_code" in r else (13 if blocked else 0)
        target = cmd[5:].strip() if name == "file" and cmd.startswith("read ") else cmd
        out.append(
            {
                "probe": "read_file" if name == "file" else name,
                "target": target,
                "exit_code": rc,
                "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                "blocked": blocked,
                "bytes": len(body.encode("utf-8")),
            }
        )
    return out


class Probes:
    def __init__(
        self,
        living_home: str,
        base: str = HOME,
        trainer_status: Optional[Callable[[], List[str]]] = None,
    ):
        self.living_home = living_home
        self.base = base
        self.trainer_status = trainer_status

    # -- the probes --------------------------------------------------------

    def trainer(self, _msg: str = "") -> Dict[str, str]:
        lines = self.trainer_status() if self.trainer_status else ["(trainer status unavailable)"]
        return {"name": "trainer", "cmd": "osiris train --status", "out": "\n".join(lines)}

    def ledger(self, _msg: str = "") -> Dict[str, Any]:
        """The exchange log's chain, naming the file read: two data homes (PRoot's ~/.osiris and Termux's) each
        hold a log, and a console reading the other one is the E-24 failure."""
        resolved_path = os.path.abspath(os.path.expanduser(os.path.join(self.living_home, "exchanges.jsonl")))
        v = verify_chain(resolved_path)
        verdict_str = "PASS" if v["intact"] else f"FAIL ({v['status']})"
        out_lines = [
            f"Data Home:    {resolved_path}",
            f"Status:       {v['status']}",
            f"Integrity:    {verdict_str} (exit code {v['exit_code']})",
            f"Entries:      {v['entries']} verified",
            f"Chain Head:   {v['head']}",
            f"Forks:        {len(v['forks'])}",
        ]
        if v.get("broken_at") is not None:
            out_lines.append(f"Broken At:    entry {v['broken_at']} ({v.get('error', '')})")
        elif v.get("error"):
            out_lines.append(f"Error:        {v['error']}")
        if v["status"] == "EMPTY":
            out_lines.append(
                "Note:         no exchange log here yet; a console that has talked before is "
                "reading another data home"
            )
        return {
            "name": "ledger",
            "cmd": f"verify {resolved_path} hash chain",
            "out": "\n".join(out_lines),
            "resolved_path": resolved_path,
            "status": str(v["status"]),
            "intact": bool(v["intact"]),
            "exit_code": int(v["exit_code"]),
        }

    def _resolve_subpath(self, rel: str, want_git: bool = False, want_dir: bool = False) -> Optional[str]:
        roots = (
            self.base,
            os.path.join(self.base, "osiris-repos"),
            os.path.join(self.base, "osiris-assessment", "work"),
        )
        for root in roots:
            cand = os.path.join(root, rel)
            if want_git and os.path.exists(os.path.join(cand, ".git")):
                return cand
            if want_dir and os.path.isdir(cand):
                return cand
        return None

    def git(self, _msg: str = "") -> Dict[str, str]:
        parts = []
        for repo in REPOS:
            d = self._resolve_subpath(repo, want_git=True)
            if d is None:
                continue
            branch = _run(["git", "-C", d, "branch", "--show-current"])
            log = _run(["git", "-C", d, "log", "--oneline", "-3", "--format=%h %ad %s", "--date=short"])
            dirty = _run(["git", "-C", d, "status", "--short", "--untracked-files=no"])
            n = len([x for x in dirty.splitlines() if x.strip()])
            parts.append(f"{repo} ({branch}, {n} modified tracked files):\n{log}")
        return {
            "name": "git",
            "cmd": "git log --oneline -3; git status --short (per repo)",
            "out": "\n\n".join(parts) or "(no repos found)",
        }

    def system(self, _msg: str = "") -> Dict[str, str]:
        free_lines = _run(["free", "-h"]).splitlines()
        out = [_run(["uptime"]), free_lines[1] if len(free_lines) > 1 else ""]
        try:
            import urllib.request

            base = os.environ.get("OSIRIS_OLLAMA", "http://localhost:11434").rstrip("/")
            with urllib.request.urlopen(base + "/api/ps", timeout=2.0) as r:
                loaded = [m["name"] for m in json.load(r).get("models", [])]
            out.append("ollama models in memory: " + (", ".join(loaded) or "none"))
        except (OSError, ValueError, KeyError):
            out.append("ollama: not reachable")
        return {"name": "system", "cmd": "uptime; free -h; ollama ps", "out": "\n".join(x for x in out if x)}

    def evidence(self, msg: str) -> Dict[str, str]:
        """The pre-registered results behind the scorecard: each named experiment's
        results file, its sha256, and the verdict block exactly as the run wrote it.
        With no experiment named, one line per evaluation file."""
        d = self._resolve_subpath(RESULTS_DIR, want_dir=True) or os.path.join(self.base, RESULTS_DIR)
        try:
            names = sorted(n for n in os.listdir(d) if n.endswith(".json"))
        except OSError:
            return {"name": "evidence", "cmd": f"ls {RESULTS_DIR}", "out": "(no results directory)"}
        text = msg
        for rx, repl in EXPERIMENT_ALIASES:
            text = rx.sub(repl, text)
        wanted = {m.lower() for m in EXPERIMENT_RE.findall(text)}
        picked = [n for n in names if n.split("_eval")[0].split("_sweep")[0].split("_tuning")[0] in wanted]
        if not wanted:
            picked = [n for n in names if "_eval" in n]
        elif not picked:
            return {
                "name": "evidence",
                "cmd": f"ls {RESULTS_DIR}",
                "out": f"(no results file for {', '.join(sorted(wanted))}: not run, or not recorded)",
            }
        parts = []
        for n in picked:
            with open(os.path.join(d, n), "rb") as f:
                raw = f.read()
            head = f"{RESULTS_DIR}/{n} sha256:{hashlib.sha256(raw).hexdigest()[:16]}"
            try:
                verdict = json.loads(raw).get("verdict") if raw.lstrip()[:1] == b"{" else None
            except ValueError:
                verdict = None
            if not wanted:
                if isinstance(verdict, dict):
                    verdict = next(
                        (verdict[k] for k in ("outcome", "verdict", "pass", "PASS") if k in verdict),
                        None,
                    )
                if isinstance(verdict, bool):
                    verdict = "PASS" if verdict else "FAIL"
                parts.append(head + (f"  verdict: {str(verdict)[:100]}" if verdict is not None else ""))
                continue
            body = json.dumps(verdict, separators=(",", ":")) if verdict is not None else "(no verdict block)"
            more = "…(truncated)" if len(body) > MAX_VERDICT_CHARS else ""
            parts.append(f"{head}\nverdict: {body[:MAX_VERDICT_CHARS]}{more}")
        cmd = f"sha256sum + verdict of {RESULTS_DIR}/" + ("{" + ",".join(picked) + "}" if wanted else "*_eval*.json")
        return {"name": "evidence", "cmd": cmd, "out": ("\n\n" if wanted else "\n").join(parts)}

    def files(self, msg: str) -> List[Dict[str, str]]:
        found, seen = [], set()
        doi_or_host = re.compile(r"^(?:10\.\d{4,9}/|(?:https?:/)?[\w.\-]+\.(?:org|com|io|net|edu|gov|ai)/)", re.I)
        scrubbed = re.sub(r"https?://\S+", " ", msg)
        scrubbed = re.sub(r"\bdoi\s*:\s*10\.\d{4,9}/\S+", " ", scrubbed, flags=re.I)
        for raw in re.findall(
            r"(?<![\w:/.-])(?:(?:~/|/)?[\w.\-]+(?:/[\w.\-]+)+\.[A-Za-z][A-Za-z0-9]{1,7}|[\w\-]+\.(?:md|txt|json|py|dna|toml|yaml|yml))\b",
            scrubbed,
        ):
            if raw in seen or doi_or_host.match(raw):
                continue
            seen.add(raw)
            if len(found) >= MAX_FILE_CHECKS:
                break
            p = safe_path(raw, self.base)
            if not p and "/" not in raw:
                continue
            if p is None:
                found.append(
                    {
                        "name": "file",
                        "cmd": f"read {raw}",
                        "out": "(not read: missing, outside home, or a credentials file)",
                    }
                )
                continue
            with open(p, encoding="utf-8", errors="replace") as f:
                text = f.read(MAX_FILE_CHARS + 1)
            more = "\n…(truncated)" if len(text) > MAX_FILE_CHARS else ""
            found.append(
                {
                    "name": "file",
                    "cmd": f"read {os.path.relpath(p, self.base)}",
                    "out": text[:MAX_FILE_CHARS] + more,
                }
            )
        return found

    # -- routing -----------------------------------------------------------

    ROUTES = (
        ("trainer", re.compile(r"\b(train(ing|er)?|distill\w*|checkpoint|step|bpb|bits.per.byte|gate|learn(ing|ed)?)\b", re.I)),
        ("git", re.compile(r"\b(git|commits?|committed|branch|diff|changes?|changed|repo(sitory|s)?|pushed)\b", re.I)),
        ("ledger", re.compile(r"\b(ledger|hash.?chain|exchanges?|log intact|tamper\w*|audit)\b", re.I)),
        (
            "evidence",
            re.compile(
                r"\b(m[1-7][a-c]?|ablation|pre-?regist\w*|scorecard|evidence|results?|"
                r"experiment\w*|substrate[\s_-]*opt\w*|relay|drift|signalling|annecs)\b",
                re.I,
            ),
        ),
        ("system", re.compile(r"\b(cpu|ram|memory usage|load|slow|ollama|system|machine|resources?)\b", re.I)),
    )
    SELF_RE = re.compile(r"\b(about yourself|who are you|what are you|status|how are you doing|what do you know)\b", re.I)

    def select(self, msg: str) -> List[str]:
        names = [n for n, rx in self.ROUTES if rx.search(msg)]
        if self.SELF_RE.search(msg) and "trainer" not in names:
            names.insert(0, "trainer")
        return names

    def run(self, msg: str, names: Optional[List[str]] = None) -> List[Dict[str, str]]:
        results = []
        for name in (names if names is not None else self.select(msg)):
            try:
                results.append(getattr(self, name)(msg))
            except Exception as e:  # noqa: BLE001 - a failed probe is reported, never invented
                results.append({"name": name, "cmd": name, "out": f"(probe failed: {type(e).__name__})"})
        try:
            results.extend(self.files(msg))
        except OSError:
            pass
        return results

    @staticmethod
    def format(results: List[Dict[str, str]]) -> str:
        return "\n\n".join(f"$ {r['cmd']}\n{r['out']}" for r in results)
