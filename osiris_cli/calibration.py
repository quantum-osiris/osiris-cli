"""L6 Benchmark: deterministic confidence calibration (Brier score, ECE, reliability bins).

Evaluates stated confidence in ``[0, 1]`` against measured binary correctness ``{0, 1}``
from hidden tests or verified verdicts. All persisted numbers are serialized as exact
fixed-precision decimal strings (``ROUND_HALF_EVEN``, 6 decimal places) inside an
append-only ``exec-canon-1`` SHA-256 hash-chained JSONL ledger under
``$OSIRIS_HOME/benchmarks/calibration.jsonl`` (mode ``0600``, ``flock`` + ``fsync``),
so floats never enter canonical receipts.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

from .exec_kernel import ExecJournal, canon

SCHEMA = "osiris.calibration/1"
_QUANT = Decimal("0.000001")
_ZERO = Decimal("0")
_ONE = Decimal("1")
PathArg = Optional[Union[os.PathLike, str]]
NumArg = Union[Decimal, float, int, str]


class CalibrationError(ValueError):
    """Invalid probability, bin count, or corrupted calibration record."""


def default_osiris_home(osiris_home: PathArg = None) -> Path:
    raw = osiris_home or os.environ.get("OSIRIS_HOME")
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".osiris").resolve()


def default_calibration_path(osiris_home: PathArg = None) -> Path:
    if osiris_home is None:
        override = os.environ.get("OSIRIS_CALIBRATION_LOG")
        if override:
            return Path(override).expanduser().resolve()
    return default_osiris_home(osiris_home) / "benchmarks" / "calibration.jsonl"


def format_decimal(value: Union[Decimal, int, str]) -> str:
    """Format a numeric value into a 6-decimal canonical string without float imprecision."""
    if isinstance(value, bool):
        raise CalibrationError("boolean is not a valid decimal probability")
    try:
        dec = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise CalibrationError(f"invalid decimal value {value!r}: {exc}") from exc
    if not dec.is_finite():
        raise CalibrationError(f"non-finite decimal value {value!r}")
    return str(dec.quantize(_QUANT, rounding=ROUND_HALF_EVEN))


def parse_probability(value: NumArg) -> Decimal:
    """Parse a stated confidence in ``[0, 1]`` into an exact ``Decimal``."""
    if isinstance(value, bool):
        raise CalibrationError("stated_confidence cannot be a boolean")
    try:
        dec = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise CalibrationError(f"invalid stated_confidence {value!r}: {exc}") from exc
    if not dec.is_finite() or dec < _ZERO or dec > _ONE:
        raise CalibrationError(f"stated_confidence must be in [0, 1], got {value!r}")
    return dec.quantize(_QUANT, rounding=ROUND_HALF_EVEN)


@dataclass(frozen=True)
class CalibrationObservation:
    prediction_id: str
    model_id: str
    task_id: str
    stated_confidence_str: str
    measured_correct: int  # 1 or 0
    run_id: str = ""
    evidence_ref: str = ""
    ts: str = ""

    @property
    def confidence(self) -> Decimal:
        return parse_probability(self.stated_confidence_str)


@dataclass(frozen=True)
class ReliabilityBin:
    bin_index: int
    lower_str: str
    upper_str: str
    count: int
    correct_count: int
    mean_confidence_str: str
    empirical_accuracy_str: str
    abs_gap_str: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bin_index": self.bin_index,
            "lower": self.lower_str,
            "upper": self.upper_str,
            "count": self.count,
            "correct_count": self.correct_count,
            "mean_confidence": self.mean_confidence_str,
            "empirical_accuracy": self.empirical_accuracy_str,
            "abs_gap": self.abs_gap_str,
        }


@dataclass(frozen=True)
class CalibrationReport:
    model_filter: str
    num_samples: int
    num_bins: int
    brier_score_str: str
    ece_str: str
    mean_confidence_str: str
    mean_accuracy_str: str
    overconfidence_str: str
    bins: Tuple[ReliabilityBin, ...]
    intact: bool = True

    @property
    def brier_score(self) -> Decimal:
        return Decimal(self.brier_score_str)

    @property
    def ece(self) -> Decimal:
        return Decimal(self.ece_str)

    def to_receipt_body(self) -> Dict[str, Any]:
        body = {
            "schema": SCHEMA,
            "kind": "CALIBRATION_RECEIPT",
            "model_filter": self.model_filter or "ALL",
            "num_samples": self.num_samples,
            "num_bins": self.num_bins,
            "brier_score": self.brier_score_str,
            "ece": self.ece_str,
            "mean_confidence": self.mean_confidence_str,
            "mean_accuracy": self.mean_accuracy_str,
            "overconfidence": self.overconfidence_str,
            "bins": [b.to_dict() for b in self.bins],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        canon(body)  # guarantees zero floats (exec-canon-1)
        return body


def compute_calibration(
    observations: Sequence[Union[CalibrationObservation, Tuple[NumArg, Union[bool, int]]]],
    *,
    num_bins: int = 10,
    model_filter: str = "",
) -> CalibrationReport:
    """Deterministically compute Brier score, ECE, and equal-width reliability bins."""
    if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins < 1 or num_bins > 100:
        raise CalibrationError(f"num_bins must be an integer in [1, 100], got {num_bins!r}")

    parsed: List[Tuple[Decimal, int]] = []
    for item in observations:
        if isinstance(item, CalibrationObservation):
            if model_filter and item.model_id != model_filter:
                continue
            p = item.confidence
            y = int(item.measured_correct)
        elif isinstance(item, tuple) and len(item) == 2:
            raw_p, raw_y = item
            p = parse_probability(raw_p)
            if isinstance(raw_y, bool):
                y = 1 if raw_y else 0
            elif isinstance(raw_y, int) and raw_y in (0, 1):
                y = raw_y
            else:
                raise CalibrationError(f"outcome must be bool or 0/1, got {raw_y!r}")
        else:
            raise CalibrationError(f"unsupported observation item: {item!r}")
        if y not in (0, 1):
            raise CalibrationError(f"measured_correct must be 0 or 1, got {y!r}")
        parsed.append((p, y))

    m_dec = Decimal(num_bins)
    bin_buckets: List[List[Tuple[Decimal, int]]] = [[] for _ in range(num_bins)]
    for p, y in parsed:
        # [0, 1/M], (1/M, 2/M], ..., ((M-1)/M, 1] so p=0 -> bin 0 and p=1 -> bin M-1.
        if p == _ZERO:
            idx = 0
        else:
            scaled = p * m_dec
            idx = int(scaled.to_integral_value(rounding="ROUND_CEILING")) - 1
            idx = max(0, min(num_bins - 1, idx))
        bin_buckets[idx].append((p, y))

    n_total = len(parsed)
    bins_out: List[ReliabilityBin] = []
    ece_acc = _ZERO
    for b_idx in range(num_bins):
        lower = Decimal(b_idx) / m_dec
        upper = Decimal(b_idx + 1) / m_dec
        bucket = bin_buckets[b_idx]
        cnt = len(bucket)
        if cnt == 0:
            bins_out.append(
                ReliabilityBin(
                    bin_index=b_idx,
                    lower_str=format_decimal(lower),
                    upper_str=format_decimal(upper),
                    count=0,
                    correct_count=0,
                    mean_confidence_str=format_decimal(_ZERO),
                    empirical_accuracy_str=format_decimal(_ZERO),
                    abs_gap_str=format_decimal(_ZERO),
                )
            )
            continue
        cnt_dec = Decimal(cnt)
        corr_cnt = sum(y for _p, y in bucket)
        mean_conf = sum((p for p, _y in bucket), _ZERO) / cnt_dec
        emp_acc = Decimal(corr_cnt) / cnt_dec
        gap = abs(emp_acc - mean_conf)
        if n_total > 0:
            ece_acc += (cnt_dec / Decimal(n_total)) * gap
        bins_out.append(
            ReliabilityBin(
                bin_index=b_idx,
                lower_str=format_decimal(lower),
                upper_str=format_decimal(upper),
                count=cnt,
                correct_count=corr_cnt,
                mean_confidence_str=format_decimal(mean_conf),
                empirical_accuracy_str=format_decimal(emp_acc),
                abs_gap_str=format_decimal(gap),
            )
        )

    if n_total == 0:
        return CalibrationReport(
            model_filter=model_filter or "ALL",
            num_samples=0,
            num_bins=num_bins,
            brier_score_str=format_decimal(_ZERO),
            ece_str=format_decimal(_ZERO),
            mean_confidence_str=format_decimal(_ZERO),
            mean_accuracy_str=format_decimal(_ZERO),
            overconfidence_str=format_decimal(_ZERO),
            bins=tuple(bins_out),
            intact=True,
        )

    n_dec = Decimal(n_total)
    brier = sum(((p - Decimal(y)) ** 2 for p, y in parsed), _ZERO) / n_dec
    overall_conf = sum((p for p, _y in parsed), _ZERO) / n_dec
    overall_acc = Decimal(sum(y for _p, y in parsed)) / n_dec
    overconf = overall_conf - overall_acc
    return CalibrationReport(
        model_filter=model_filter or "ALL",
        num_samples=n_total,
        num_bins=num_bins,
        brier_score_str=format_decimal(brier),
        ece_str=format_decimal(ece_acc),
        mean_confidence_str=format_decimal(overall_conf),
        mean_accuracy_str=format_decimal(overall_acc),
        overconfidence_str=format_decimal(overconf),
        bins=tuple(bins_out),
        intact=True,
    )


class CalibrationLedger:
    """Append-only ``exec-canon-1`` hash-chained JSONL store for predictions, outcomes & receipts."""

    def __init__(self, path: PathArg = None) -> None:
        self.path = Path(path).expanduser().resolve() if path is not None else default_calibration_path()
        self._journal = ExecJournal(self.path)

    def _append_body(self, body: Dict[str, Any]) -> str:
        canon(body)
        digest = self._journal.append(body)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass
        return digest

    def record_prediction(
        self,
        model_id: str,
        task_id: str,
        stated_confidence: NumArg,
        *,
        prediction_id: Optional[str] = None,
        run_id: str = "",
    ) -> Tuple[str, str]:
        """Log a stated confidence before outcome is known. Returns ``(prediction_id, receipt_hash)``."""
        pid = prediction_id or f"pred-{uuid.uuid4().hex[:12]}"
        conf_str = format_decimal(parse_probability(stated_confidence))
        body = {
            "schema": SCHEMA,
            "kind": "PREDICTION",
            "prediction_id": str(pid),
            "model_id": str(model_id),
            "task_id": str(task_id),
            "stated_confidence": conf_str,
            "run_id": str(run_id),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        return pid, self._append_body(body)

    def record_outcome(
        self,
        prediction_id: str,
        correct: Union[bool, int],
        *,
        evidence_ref: str = "",
        run_id: str = "",
    ) -> str:
        """Bind a measured hidden-test/verdict outcome (0 or 1) to an earlier ``prediction_id``."""
        if isinstance(correct, bool):
            y = 1 if correct else 0
        elif isinstance(correct, int) and correct in (0, 1):
            y = correct
        else:
            raise CalibrationError(f"correct must be bool or 0/1, got {correct!r}")
        body = {
            "schema": SCHEMA,
            "kind": "OUTCOME",
            "prediction_id": str(prediction_id),
            "measured_correct": y,
            "evidence_ref": str(evidence_ref),
            "run_id": str(run_id),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        return self._append_body(body)

    def record_observation(
        self,
        model_id: str,
        task_id: str,
        stated_confidence: NumArg,
        correct: Union[bool, int],
        *,
        prediction_id: Optional[str] = None,
        run_id: str = "",
        evidence_ref: str = "",
    ) -> str:
        """Append a paired prediction + measured outcome observation in one canonical row."""
        pid = prediction_id or f"obs-{uuid.uuid4().hex[:12]}"
        conf_str = format_decimal(parse_probability(stated_confidence))
        if isinstance(correct, bool):
            y = 1 if correct else 0
        elif isinstance(correct, int) and correct in (0, 1):
            y = correct
        else:
            raise CalibrationError(f"correct must be bool or 0/1, got {correct!r}")
        body = {
            "schema": SCHEMA,
            "kind": "OBSERVATION",
            "prediction_id": str(pid),
            "model_id": str(model_id),
            "task_id": str(task_id),
            "stated_confidence": conf_str,
            "measured_correct": y,
            "run_id": str(run_id),
            "evidence_ref": str(evidence_ref),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        return self._append_body(body)

    def write_receipt(self, report: CalibrationReport) -> str:
        """Persist a computed calibration report receipt (all metrics stored as decimal strings)."""
        return self._append_body(report.to_receipt_body())

    def verify(self) -> Tuple[bool, int, str]:
        return self._journal.verify()

    def verify_tail(self, tail_lines: int = 20) -> Any:
        from . import probes
        return probes.verify_chain_tail(self.path, tail_lines=tail_lines)

    def observations(self, *, model_filter: str = "") -> List[CalibrationObservation]:
        """Reconstruct matched observations from ``OBSERVATION`` and ``PREDICTION``+``OUTCOME`` rows."""
        if not self.path.is_file():
            return []
        ok, _cnt, err = self.verify()
        if not ok:
            raise CalibrationError(f"calibration ledger failed integrity check: {err}")

        preds: Dict[str, Dict[str, Any]] = {}
        outcomes: Dict[str, Dict[str, Any]] = {}
        ordered_obs: List[CalibrationObservation] = []

        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                body = row.get("body") or {}
                kind = body.get("kind")
                if kind == "OBSERVATION":
                    obs = CalibrationObservation(
                        prediction_id=str(body["prediction_id"]),
                        model_id=str(body["model_id"]),
                        task_id=str(body["task_id"]),
                        stated_confidence_str=format_decimal(parse_probability(body["stated_confidence"])),
                        measured_correct=int(body["measured_correct"]),
                        run_id=str(body.get("run_id", "")),
                        evidence_ref=str(body.get("evidence_ref", "")),
                        ts=str(body.get("ts", "")),
                    )
                    if not model_filter or obs.model_id == model_filter:
                        ordered_obs.append(obs)
                elif kind == "PREDICTION":
                    preds[str(body["prediction_id"])] = body
                elif kind == "OUTCOME":
                    outcomes[str(body["prediction_id"])] = body

        for pid, pbody in preds.items():
            if pid not in outcomes:
                continue
            obody = outcomes[pid]
            obs = CalibrationObservation(
                prediction_id=pid,
                model_id=str(pbody["model_id"]),
                task_id=str(pbody["task_id"]),
                stated_confidence_str=format_decimal(parse_probability(pbody["stated_confidence"])),
                measured_correct=int(obody["measured_correct"]),
                run_id=str(obody.get("run_id") or pbody.get("run_id") or ""),
                evidence_ref=str(obody.get("evidence_ref", "")),
                ts=str(obody.get("ts") or pbody.get("ts") or ""),
            )
            if not model_filter or obs.model_id == model_filter:
                ordered_obs.append(obs)

        return ordered_obs

    def evaluate(
        self,
        *,
        model_filter: str = "",
        num_bins: int = 10,
        persist_receipt: bool = False,
    ) -> Tuple[CalibrationReport, Optional[str]]:
        obs = self.observations(model_filter=model_filter)
        report = compute_calibration(obs, num_bins=num_bins, model_filter=model_filter)
        receipt_hash = self.write_receipt(report) if (persist_receipt and obs) else None
        return report, receipt_hash


def _bench_prediction_id(c: Dict[str, Any]) -> str:
    run_id = str(c.get("run") or c.get("entry") or "run")
    task_id = str(c.get("task") or "task")
    attempt_n = c.get("n") if c.get("n") is not None else 1
    return f"bench-{run_id}-{task_id}-{attempt_n}"


def no_output_prediction_ids(cands: Sequence[Dict[str, Any]]) -> FrozenSet[str]:
    """Prediction IDs of bench attempts that produced no output (timeouts, empty or malformed replies).

    Used at read time so rows a ledger recorded before such attempts were skipped on ingest are
    left out of the scores too. The append-only ledger itself is never rewritten.
    """
    return frozenset(_bench_prediction_id(c) for c in cands if isinstance(c, dict) and c.get("no_output"))


def sync_from_bench_candidates(
    cands: Sequence[Dict[str, Any]],
    *,
    path: PathArg = None,
) -> int:
    """Idempotently ingest verified `protege.candidates(...)` records into `CalibrationLedger`.

    Maps each candidate's self-test gate (`own_ok`) as empirical self-verification confidence
    (`1.000000` if `own_ok` else `0.000000`) against `hidden_ok` (`1` or `0`), skipping
    infrastructure/artifact `integrity` exclusions and `no_output` attempts (a timeout or an empty
    or malformed reply). A model that produced nothing made no prediction: scoring it as
    "confidence 0, wrong" would give it a perfect Brier score for failing. Those attempts stay
    counted as failures in the bench scorecard; they are availability failures, not calibration
    evidence. Returns count of newly appended rows.
    """
    valid = [c for c in cands if isinstance(c, dict) and not c.get("integrity") and not c.get("no_output")]
    if not valid:
        return 0

    ledger = CalibrationLedger(path)
    existing_ids = {o.prediction_id for o in ledger.observations()}
    added = 0
    for c in valid:
        run_id = str(c.get("run") or c.get("entry") or "run")
        task_id = str(c.get("task") or "task")
        pid = _bench_prediction_id(c)
        if pid in existing_ids:
            continue
        mentor = str(c.get("mentor") or "unknown")
        conf = "1.000000" if c.get("own_ok") else "0.000000"
        correct = 1 if c.get("hidden_ok") else 0
        sha_ref = str(c.get("module_sha256") or c.get("entry") or "")[:16]
        ledger.record_observation(
            mentor,
            task_id,
            conf,
            correct,
            prediction_id=pid,
            run_id=run_id,
            evidence_ref=f"bench_self_test:{sha_ref}",
        )
        existing_ids.add(pid)
        added += 1
    return added


def load_bench_candidates(*, bench_dir: PathArg = None) -> List[Dict[str, Any]]:
    """Comparable `protege` candidates from `~/.osiris/bench/results.ledger.jsonl` (empty if unavailable)."""
    try:
        import protege
    except ImportError:
        return []
    bdir = str(Path(bench_dir).expanduser().resolve()) if bench_dir is not None else protege.BENCH_DIR
    entries, problem = protege.load_ledger(bdir)
    if problem or not entries:
        return []
    cands, _skipped = protege.candidates(
        entries,
        protege.current_suite(),
        protege.current_runner(),
        bench_dir=bdir,
    )
    return list(cands)


def sync_bench_ledger(
    *,
    bench_dir: PathArg = None,
    path: PathArg = None,
) -> int:
    """Load `~/.osiris/bench/results.ledger.jsonl` via `protege` and sync comparable candidates."""
    return sync_from_bench_candidates(load_bench_candidates(bench_dir=bench_dir), path=path)


def _drop_excluded(obs: List[CalibrationObservation], exclude_ids: FrozenSet[str]):
    kept = [o for o in obs if o.prediction_id not in exclude_ids]
    return kept, len(obs) - len(kept)


def _excluded_note(n: int) -> str:
    return (f" · {n} attempt(s) with no output left out (availability failures, not predictions)"
            if n else "")


def format_summary(
    path: PathArg = None,
    *,
    model_filter: str = "",
    exclude_ids: FrozenSet[str] = frozenset(),
) -> str:
    """Compact one-block summary surfaced in ``/mentors`` and ``/bench`` (<100 cols)."""
    ledger = CalibrationLedger(path)
    ok, _cnt, err = ledger.verify()
    if not ok:
        return f"Calibration ledger TAMPERED ({err}) -- inspect /bench calibration"
    obs, n_excluded = _drop_excluded(ledger.observations(model_filter=model_filter), exclude_ids)
    if not obs:
        return ("Calibration: 0 confidence-tagged predictions recorded (use /bench calibration)"
                + _excluded_note(n_excluded))
    rep = compute_calibration(obs, num_bins=10, model_filter=model_filter)
    models = sorted({o.model_id for o in obs})
    by_model: List[str] = []
    for m in models[:4]:
        m_rep = compute_calibration(obs, num_bins=10, model_filter=m)
        by_model.append(f"{m} n={m_rep.num_samples} Brier={m_rep.brier_score_str} ECE={m_rep.ece_str}")
    head = (
        f"Calibration ({rep.num_samples} scored): Brier={rep.brier_score_str} · "
        f"ECE={rep.ece_str} · conf={rep.mean_confidence_str} vs acc={rep.mean_accuracy_str}"
        + _excluded_note(n_excluded)
    )
    if len(models) > 1 or (models and not model_filter):
        return head + "\n  " + " | ".join(by_model)
    return head


def format_detailed(
    path: PathArg = None,
    *,
    model_filter: str = "",
    num_bins: int = 10,
    exclude_ids: FrozenSet[str] = frozenset(),
) -> str:
    """Full reliability-bin table (<100 cols) for ``/bench calibration`` and ``/calibration``."""
    ledger = CalibrationLedger(path)
    ok, total_rows, err = ledger.verify()
    if not ok:
        return f"L6 CALIBRATION · TAMPERED ({err})"
    obs, n_excluded = _drop_excluded(ledger.observations(model_filter=model_filter), exclude_ids)
    rep = compute_calibration(obs, num_bins=num_bins, model_filter=model_filter)
    label = model_filter or "all models"
    lines = [
        f"L6 CALIBRATION ({label}) · chain INTACT ({total_rows} rows, {rep.num_samples} scored)",
        f"  Brier score: {rep.brier_score_str}   ECE ({rep.num_bins} bins): {rep.ece_str}",
        f"  Mean conf:   {rep.mean_confidence_str}   Empirical acc: {rep.mean_accuracy_str}   "
        f"Overconf: {rep.overconfidence_str}",
    ]
    if n_excluded:
        lines.append(" " + _excluded_note(n_excluded).lstrip(" ·").rstrip())
    if rep.num_samples == 0:
        lines.append("  No paired confidence/outcome observations recorded yet.")
        return "\n".join(lines)
    lines.append("  Bin          Count  Correct  MeanConf   EmpAcc     |Gap|")
    for b in rep.bins:
        if b.count == 0:
            continue
        bracket = "[" if b.bin_index == 0 else "("
        rng = f"{bracket}{b.lower_str[:4]},{b.upper_str[:4]}]"
        lines.append(
            f"  {rng:<12} {b.count:>5}  {b.correct_count:>7}  "
            f"{b.mean_confidence_str:<9}  {b.empirical_accuracy_str:<9}  {b.abs_gap_str}"
        )
    return "\n".join(lines)


def run_command(args: str = "", *, osiris_home: PathArg = None) -> str:
    """CLI handler for ``/calibration [model_id|sync|verify]`` and ``/bench calibration``."""
    parts = (args or "").strip().split()
    from_bench_cmd = bool(parts and parts[0].lower() == "calibration")
    if from_bench_cmd:
        parts = parts[1:]
    path = default_calibration_path(osiris_home)
    bdir = (default_osiris_home(osiris_home) / "bench") if osiris_home is not None else None
    cands = load_bench_candidates(bench_dir=bdir)
    exclude = no_output_prediction_ids(cands)
    if parts and parts[0].lower() == "sync":
        added = sync_from_bench_candidates(cands, path=path)
        return (f"Synced {added} new benchmark attempt(s) into calibration ledger.\n"
                + format_detailed(path, exclude_ids=exclude))
    if from_bench_cmd and osiris_home is None and "OSIRIS_CALIBRATION_LOG" not in os.environ:
        sync_from_bench_candidates(cands, path=path)
    if parts and parts[0].lower() == "verify":
        ledger = CalibrationLedger(path)
        ok, count, err = ledger.verify()
        tail = ledger.verify_tail()
        status = "INTACT" if (ok and tail.intact) else "TAMPERED"
        return f"L6 calibration chain {status}: {count} entries, tail={tail.status}" + (f" ({err})" if err else "")
    model_filter = parts[0] if parts else ""
    return format_detailed(path, model_filter=model_filter, exclude_ids=exclude)
