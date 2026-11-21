"""L3 Evidence: claim-to-source provenance ledger per reply.

Every substantive claim made in an OSIRIS reply is paired with its deterministic
provenance record alongside the exchange log:

  * ``claim_id``        registered claim ID from ``osiris_cli.claims`` or ``clm-<sha256[:12]>``
  * ``claim_text``      normalized claim statement extracted from the reply
  * ``source_ref``      ``file:<path>@sha256:<12hex>``, ``doi:<id>``, ``sha256:<hex>``, or ``UNVERIFIED``
  * ``sys024_verdict``  registered SYS-024 verdict, ``QUARANTINED``, ``GROUNDED``, or ``UNVERIFIED``
  * ``run_id``          governing task/exchange run identifier

Persisted in an append-only ``exec-canon-1`` SHA-256 hash-chained JSONL ledger under
``$OSIRIS_HOME/evidence/claim_provenance.jsonl`` (mode ``0600``, ``flock`` + ``fsync``),
verifiable in O(1) via ``probes.verify_chain_tail``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .exec_kernel import ExecJournal, canon

SCHEMA = "osiris.claim_provenance/1"
PathArg = Optional[Union[os.PathLike, str]]

# Quarantined / refuted constants (SYS-024 / REQ-PLAN-3) that must never pass as grounded physics.
_QUARANTINED_CONSTANTS: Tuple[Tuple[str, re.Pattern], ...] = (
    ("SYS024_THETA_LOCK", re.compile(r"\b(?:theta_lock|51\.843\d*)\b", re.IGNORECASE)),
    ("SYS024_PHI_THRESHOLD", re.compile(r"\b(?:phi_threshold|0\.7734\d*)\b", re.IGNORECASE)),
    ("SYS024_F_MAX", re.compile(r"\b(?:f_max\s*=\s*0\.97871|0\.97871\d*)\b", re.IGNORECASE)),
    ("SYS024_LAMBDA_PHI", re.compile(r"\b(?:lambda_phi|2\.176435e-0?8)\b", re.IGNORECASE)),
)

_FENCED_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b")
_SHA_RE = re.compile(r"\b(?:sha256:)?([0-9a-f]{12,64})\b", re.IGNORECASE)
_METRIC_ASSERTION_RE = re.compile(
    r"(?:\b\d+(?:\.\d+)?\s*(?:%|ms|s|us|µs|μs|ghz|mhz|khz|ev|mev|gev|kelvin|k|bits?/byte|bpb|"
    r"passed|failed|qubits?|shots?|x\b|sigma|σ)\b|"
    r"\b(?:fidelity|p-value|brier|ece|ndcg|accuracy|error rate|pass rate|latency)\b[^.\n]{0,40}\b\d)",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def default_osiris_home(osiris_home: PathArg = None) -> Path:
    """Resolve ``$OSIRIS_HOME`` dynamically without hardcoded user directories."""
    raw = osiris_home or os.environ.get("OSIRIS_HOME")
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".osiris").resolve()


def default_provenance_path(osiris_home: PathArg = None) -> Path:
    """Return canonical path to ``claim_provenance.jsonl`` under ``$OSIRIS_HOME``."""
    if osiris_home is None:
        override = os.environ.get("OSIRIS_PROVENANCE_LOG")
        if override:
            return Path(override).expanduser().resolve()
    return default_osiris_home(osiris_home) / "evidence" / "claim_provenance.jsonl"


@dataclass(frozen=True)
class ProvenanceRecord:
    claim_id: str
    claim_text: str
    source_ref: str
    sys024_verdict: str
    run_id: str
    exchange_hash: str = ""
    voice: str = ""
    ts: str = ""

    def to_body(self) -> Dict[str, Any]:
        body = {
            "schema": SCHEMA,
            "claim_id": str(self.claim_id),
            "claim_text": str(self.claim_text),
            "source_ref": str(self.source_ref) if self.source_ref else "UNVERIFIED",
            "sys024_verdict": str(self.sys024_verdict),
            "run_id": str(self.run_id),
            "exchange_hash": str(self.exchange_hash),
            "voice": str(self.voice),
            "ts": str(self.ts) or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        canon(body)  # enforces exec-canon-1 (no floats)
        return body


class ProvenanceLedger:
    """Append-only ``exec-canon-1`` hash-chained JSONL store for L3 claim provenance."""

    def __init__(self, path: PathArg = None) -> None:
        self.path = Path(path).expanduser().resolve() if path is not None else default_provenance_path()
        self._journal = ExecJournal(self.path)

    def append(self, record: ProvenanceRecord) -> str:
        digest = self._journal.append(record.to_body())
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass
        return digest

    def append_many(self, records: Sequence[ProvenanceRecord]) -> List[str]:
        return [self.append(rec) for rec in records]

    def verify(self) -> Tuple[bool, int, str]:
        return self._journal.verify()

    def verify_tail(self, tail_lines: int = 20) -> Any:
        from . import probes
        return probes.verify_chain_tail(self.path, tail_lines=tail_lines)

    def entries(self) -> List[Dict[str, Any]]:
        if not self.path.is_file():
            return []
        rows: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                body = dict(row.get("body") or {})
                body["hash"] = row.get("hash", "")
                body["prev"] = row.get("prev", "")
                body["seq"] = row.get("seq", 0)
                rows.append(body)
        return rows


def _file_source_ref(ref_path: str, base_dirs: Sequence[Path]) -> str:
    clean = ref_path.strip().strip("`'\"()[]{}")
    for base in base_dirs:
        cand = (base / clean).resolve() if not os.path.isabs(clean) else Path(clean).resolve()
        if cand.is_file():
            try:
                digest = hashlib.sha256(cand.read_bytes()).hexdigest()[:12]
                rel = os.path.relpath(cand, base) if str(cand).startswith(str(base)) else cand.name
                rel_posix = rel.replace(os.sep, "/")
                if not rel_posix.startswith(".."):
                    return f"file:{rel_posix}@sha256:{digest}"
                return f"file:{cand.name}@sha256:{digest}"
            except OSError:
                return f"file:{clean}"
    return f"file:{clean}"


def _claim_evidence_ref(claim: Any, base_dirs: Sequence[Path]) -> str:
    ev = getattr(claim, "evidence", ()) or ()
    if not ev:
        return "UNVERIFIED"
    kind, ref, _note = ev[0]
    if kind == "doi":
        return f"doi:{ref}"
    if kind == "file":
        return _file_source_ref(str(ref), base_dirs)
    return f"{kind}:{ref}" if ref else "UNVERIFIED"


def _candidate_sentences(reply: str) -> List[str]:
    stripped = _FENCED_BLOCK_RE.sub(" ", reply or "")
    out: List[str] = []
    for chunk in _SENTENCE_SPLIT_RE.split(stripped):
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)]|#+)\s*", "", chunk).strip()
        if len(line) < 12 or line.endswith("?"):
            continue
        out.append(line)
    return out


def extract_reply_claims(
    reply: str,
    *,
    run_id: str = "",
    exchange_hash: str = "",
    voice: str = "",
    retrieved_files: Iterable[str] = (),
    probe_receipts: Sequence[Dict[str, Any]] = (),
    on_record: Sequence[str] = (),
    grounding_audit: Optional[Any] = None,
    workspace_root: PathArg = None,
    ts: Optional[str] = None,
) -> List[ProvenanceRecord]:
    """Deterministically extract claims from ``reply`` and bind each to a source or ``UNVERIFIED``."""
    if not reply or not reply.strip():
        return []

    from . import claims as claims_mod

    stamp = ts or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    eff_run_id = run_id or (f"exch-{exchange_hash[:12]}" if exchange_hash else "exch-adhoc")

    base_dirs: List[Path] = []
    if workspace_root:
        base_dirs.append(Path(workspace_root).expanduser().resolve())
    repo_root = Path(__file__).resolve().parent.parent
    if repo_root not in base_dirs:
        base_dirs.append(repo_root)

    retrieved_list = [str(f) for f in retrieved_files if f]
    phantom_set = set(getattr(grounding_audit, "phantom_files", ()) or ())
    unverified_nums = tuple(getattr(grounding_audit, "unverified_numbers", ()) or ())
    contradictions = tuple(getattr(grounding_audit, "contradictions", ()) or ())

    receipt_hashes: List[str] = []
    for pr in probe_receipts or ():
        if isinstance(pr, dict):
            for k in ("receipt_id", "hash", "sha256", "stdout_sha256"):
                val = pr.get(k)
                if isinstance(val, str) and len(val) >= 12:
                    receipt_hashes.append(val)
                    break

    records: List[ProvenanceRecord] = []
    seen_keys: set = set()

    sentences = _candidate_sentences(reply)
    for sent in sentences:
        clipped = sent[:300]
        # 1. Check registered SYS-024 claims first.
        matched = claims_mod.match(sent)
        if matched:
            for clm in matched:
                key = (clm.id, clipped)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                records.append(
                    ProvenanceRecord(
                        claim_id=clm.id,
                        claim_text=clipped,
                        source_ref=_claim_evidence_ref(clm, base_dirs),
                        sys024_verdict=clm.verdict,
                        run_id=eff_run_id,
                        exchange_hash=exchange_hash,
                        voice=voice,
                        ts=stamp,
                    )
                )
            continue

        # 2. Check quarantined constants (SYS-024 / REQ-PLAN-3).
        q_hit: Optional[str] = None
        for q_id, q_re in _QUARANTINED_CONSTANTS:
            if q_re.search(sent):
                q_hit = q_id
                break
        if q_hit is not None:
            key = (q_hit, clipped)
            if key not in seen_keys:
                seen_keys.add(key)
                records.append(
                    ProvenanceRecord(
                        claim_id=q_hit,
                        claim_text=clipped,
                        source_ref="SYS-024:quarantined_constant",
                        sys024_verdict="QUARANTINED",
                        run_id=eff_run_id,
                        exchange_hash=exchange_hash,
                        voice=voice,
                        ts=stamp,
                    )
                )
            continue

        # 3. Check explicit DOI, retrieved file citation, SHA receipt, or quantitative assertion.
        cid = "clm-" + hashlib.sha256(clipped.encode("utf-8")).hexdigest()[:12]
        doi_m = _DOI_RE.search(sent)
        cited_file: Optional[str] = None
        cited_phantom = False
        for pf in phantom_set:
            if pf and pf in sent:
                cited_phantom = True
                break
        if not cited_phantom:
            for rf in retrieved_list:
                base_rf = os.path.basename(rf)
                if rf in sent or (base_rf and len(base_rf) >= 4 and base_rf in sent):
                    cited_file = rf
                    break

        has_bad_num = any(str(un) in sent for un in unverified_nums if un)
        has_contradiction = any(str(ct) in sent for ct in contradictions if ct)

        if cited_phantom or has_bad_num or has_contradiction:
            source_ref = "UNVERIFIED"
            verdict = "UNVERIFIED"
        elif doi_m:
            source_ref = f"doi:{doi_m.group(1).rstrip('.,;)')}"
            verdict = "GROUNDED"
        elif cited_file:
            source_ref = _file_source_ref(cited_file, base_dirs)
            verdict = "GROUNDED"
        elif receipt_hashes and _SHA_RE.search(sent):
            sha_m = _SHA_RE.search(sent)
            assert sha_m is not None
            source_ref = f"sha256:{sha_m.group(1).lower()}"
            verdict = "GROUNDED"
        elif on_record and any(rec_line[:24] in sent for rec_line in on_record if len(rec_line) >= 8):
            digest = hashlib.sha256("\n".join(on_record).encode("utf-8")).hexdigest()[:12]
            source_ref = f"sha256:{digest}"
            verdict = "GROUNDED"
        elif _METRIC_ASSERTION_RE.search(sent):
            if receipt_hashes:
                source_ref = f"sha256:{receipt_hashes[0]}"
                verdict = "GROUNDED"
            elif retrieved_list and grounding_audit is not None and getattr(grounding_audit, "passed", False):
                source_ref = _file_source_ref(retrieved_list[0], base_dirs)
                verdict = "GROUNDED"
            else:
                source_ref = "UNVERIFIED"
                verdict = "UNVERIFIED"
        else:
            continue

        key = (cid, clipped)
        if key not in seen_keys:
            seen_keys.add(key)
            records.append(
                ProvenanceRecord(
                    claim_id=cid,
                    claim_text=clipped,
                    source_ref=source_ref,
                    sys024_verdict=verdict,
                    run_id=eff_run_id,
                    exchange_hash=exchange_hash,
                    voice=voice,
                    ts=stamp,
                )
            )

    if not records and sentences:
        lead = sentences[0][:300]
        cid = "clm-" + hashlib.sha256(lead.encode("utf-8")).hexdigest()[:12]
        if retrieved_list and (grounding_audit is None or getattr(grounding_audit, "passed", True)):
            sref = _file_source_ref(retrieved_list[0], base_dirs)
            verd = "GROUNDED"
        elif receipt_hashes:
            sref = f"sha256:{receipt_hashes[0]}"
            verd = "GROUNDED"
        elif voice == "code":
            sref = f"sha256:{exchange_hash}" if exchange_hash else "code:deterministic"
            verd = "GROUNDED"
        else:
            sref = "UNVERIFIED"
            verd = "UNVERIFIED"
        records.append(
            ProvenanceRecord(
                claim_id=cid,
                claim_text=lead,
                source_ref=sref,
                sys024_verdict=verd,
                run_id=eff_run_id,
                exchange_hash=exchange_hash,
                voice=voice,
                ts=stamp,
            )
        )

    return records


def record_reply_provenance(
    reply: str,
    *,
    ledger_path: PathArg = None,
    osiris_home: PathArg = None,
    run_id: str = "",
    exchange_hash: str = "",
    voice: str = "",
    retrieved_files: Iterable[str] = (),
    probe_receipts: Sequence[Dict[str, Any]] = (),
    on_record: Sequence[str] = (),
    grounding_audit: Optional[Any] = None,
    workspace_root: PathArg = None,
) -> List[Tuple[ProvenanceRecord, str]]:
    """Extract and append structured provenance rows for ``reply``; return ``(record, hash)`` pairs."""
    target = Path(ledger_path) if ledger_path is not None else default_provenance_path(osiris_home)
    ledger = ProvenanceLedger(target)
    claims_found = extract_reply_claims(
        reply,
        run_id=run_id,
        exchange_hash=exchange_hash,
        voice=voice,
        retrieved_files=retrieved_files,
        probe_receipts=probe_receipts,
        on_record=on_record,
        grounding_audit=grounding_audit,
        workspace_root=workspace_root,
    )
    return [(rec, ledger.append(rec)) for rec in claims_found]


def format_report(
    path: PathArg = None,
    *,
    limit: int = 12,
    unverified_only: bool = False,
    width: int = 92,
) -> str:
    """Render a compact (<100 col safe) L3 provenance summary and recent claim entries."""
    ledger = ProvenanceLedger(path)
    ok, total, err = ledger.verify()
    chain_note = f"INTACT ({total} entries)" if ok else f"TAMPERED at {total}: {err}"
    lines = [f"L3 CLAIM PROVENANCE · {chain_note}"]
    if not ok:
        return "\n".join(lines)
    rows = ledger.entries()
    if not rows:
        lines.append("  no reply provenance entries recorded yet.")
        return "\n".join(lines)

    counts: Dict[str, int] = {}
    for r in rows:
        v = str(r.get("sys024_verdict", "UNVERIFIED"))
        counts[v] = counts.get(v, 0) + 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    lines.append(f"  verdicts: {summary}")

    filtered = [r for r in rows if not unverified_only or r.get("sys024_verdict") == "UNVERIFIED"]
    shown = filtered[-max(1, limit):]
    cap = max(48, min(width, 96))
    for r in shown:
        cid = str(r.get("claim_id", "?"))[:18]
        verdict = str(r.get("sys024_verdict", "?"))
        src = str(r.get("source_ref", "UNVERIFIED"))
        if len(src) > 30:
            src = src[:27] + "..."
        head = f"  [{verdict}] {cid} -> {src}"
        lines.append(head[:cap])
        txt = str(r.get("claim_text", "")).replace("\n", " ")
        if txt:
            lines.append(("    " + txt)[:cap])
    return "\n".join(lines)


def run_command(args: str = "", *, osiris_home: PathArg = None) -> str:
    """CLI handler for ``/provenance [verify|unverified|tail N]``."""
    parts = (args or "").strip().split()
    path = default_provenance_path(osiris_home)
    if parts and parts[0].lower() == "verify":
        ledger = ProvenanceLedger(path)
        ok, count, err = ledger.verify()
        tail = ledger.verify_tail()
        status = "INTACT" if (ok and tail.intact) else "TAMPERED"
        detail = f" ({err})" if err else ""
        return f"L3 provenance chain {status}: {count} entries, tail={tail.status}{detail}"
    if parts and parts[0].lower() == "unverified":
        return format_report(path, unverified_only=True)
    limit = 12
    if len(parts) >= 2 and parts[0].lower() == "tail":
        try:
            limit = max(1, min(100, int(parts[1])))
        except ValueError:
            limit = 12
    return format_report(path, limit=limit)
