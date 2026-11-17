"""
Workflow A: Offline Color-Code Syndrome Analysis & Paired Decoder Evaluation
============================================================================

Implements the bounded offline decoder-evaluation governance contract for
`arXiv:2610.24077v1` (Quantinuum color-code study, distance d in {3, 5, 7}, r=20).

Enforced OSIRIS invariants:
  1. Complete memory records (`SyndromeRecord`) are atomic units; rounds from the
     same record can NEVER be split across train/tuning/test partitions.
  2. Splits are grouped by `acquisition_batch` with `test_set_locked=True`; any
     overlap in record IDs or acquisition batches between train/tune and test
     raises `ValueError`.
  3. Both baseline and candidate decoders are evaluated on the exact same test
     records; abstentions, timeouts, and decoder crashes are preserved and
     counted conservatively as failures (never silently dropped).
  4. Logical failure counts (n_F, n) and binomial Clopper-Pearson / Wilson bounds
     are reported BEFORE converting to per-round rates.
  5. The per-round conversion:
         LER_hat = (n_F / n) / (r + 0.5)
     is explicitly tagged `estimator_kind="authors_half_round_estimator"` rather
     than an independently measured per-round physical error frequency.
  6. Execution mode (`offline_replay`, `streaming_emulation`, `hardware_feedback`)
     and CPU/accelerator model are mandatory provenance fields; hardware QPU
     submission and automatic claim promotion are hard-disabled (`False`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest

VALID_MODES = frozenset({"offline_replay", "streaming_emulation", "hardware_feedback"})


@dataclass(frozen=True)
class SyndromeRecord:
    record_id: str
    acquisition_batch: str
    code_distance: int
    rounds: int
    detectors_by_round: Tuple[Tuple[int, ...], ...]
    true_logical_observable: int

    def __post_init__(self):
        if not self.record_id or not self.acquisition_batch:
            raise ValueError("record_id and acquisition_batch must be non-empty")
        if self.code_distance < 3 or self.code_distance % 2 == 0:
            raise ValueError(f"Invalid color-code distance: {self.code_distance}")
        if self.rounds <= 0:
            raise ValueError(f"rounds must be positive, got {self.rounds}")
        if len(self.detectors_by_round) != self.rounds:
            raise ValueError(
                f"Record {self.record_id} declares {self.rounds} rounds but has "
                f"{len(self.detectors_by_round)} round slices (partial/split record forbidden)"
            )
        if self.true_logical_observable not in (0, 1):
            raise ValueError("true_logical_observable must be 0 or 1")

    @property
    def syndrome_weight(self) -> int:
        return sum(sum(1 for b in r if b != 0) for r in self.detectors_by_round)


@dataclass(frozen=True)
class DecoderPrediction:
    record_id: str
    predicted_logical_observable: Optional[int]
    latency_us: float
    abstained: bool = False
    error: Optional[str] = None
    peak_memory_kib: float = 0.0

    def is_effective_failure(
        self,
        true_observable: int,
        deadline_us: Optional[float] = None,
        count_deadline_miss_as_failure: bool = False,
    ) -> bool:
        """
        Preserve failures and abstentions rather than excluding them:
        an abstention, decoder exception, missing prediction, wrong logical bit,
        or (when `count_deadline_miss_as_failure=True`) a deadline overrun counts
        as an effective logical failure.
        """
        if self.abstained or self.error is not None or self.predicted_logical_observable is None:
            return True
        if count_deadline_miss_as_failure and deadline_us is not None and self.latency_us > deadline_us:
            return True
        return int(self.predicted_logical_observable) != int(true_observable)


def validate_locked_group_split(
    train_records: Sequence[SyndromeRecord],
    tune_records: Sequence[SyndromeRecord],
    test_records: Sequence[SyndromeRecord],
    test_set_locked: bool = True,
) -> Dict[str, int]:
    """Verify zero record-ID and zero acquisition-batch leakage into the locked test split."""
    if not test_set_locked:
        raise ValueError("test_set_locked must be True before evaluating decoders")
    if not test_records:
        raise ValueError("test_records must be non-empty")

    dev_ids = {r.record_id for r in train_records} | {r.record_id for r in tune_records}
    test_ids = {r.record_id for r in test_records}
    if len(test_ids) != len(test_records):
        raise ValueError("Duplicate record_id detected in test_records")
    overlap_ids = dev_ids & test_ids
    if overlap_ids:
        raise ValueError(f"Record-level train/tune -> test leakage: {sorted(overlap_ids)[:5]}")

    dev_batches = {r.acquisition_batch for r in train_records} | {r.acquisition_batch for r in tune_records}
    test_batches = {r.acquisition_batch for r in test_records}
    overlap_batches = dev_batches & test_batches
    if overlap_batches:
        raise ValueError(
            f"Acquisition-batch leakage between dev and test splits: {sorted(overlap_batches)}"
        )

    return {
        "n_train": len(train_records),
        "n_tune": len(tune_records),
        "n_test": len(test_records),
        "n_test_batches": len(test_batches),
    }


def authors_per_round_ler_estimator(
    n_failures: int, n_records: int, rounds: int = 20, confidence: float = 0.95
) -> Dict[str, Any]:
    """
    Compute the arXiv:2610.24077v1 per-round estimator:
        LER_hat = (n_F / n) / (r + 1/2)
    alongside exact Clopper-Pearson confidence bounds on per-record failure probability,
    explicitly labeled as the authors' estimator (not an independent per-round measurement).
    """
    if n_records <= 0 or n_failures < 0 or n_failures > n_records:
        raise ValueError(f"Invalid failure/record counts: n_F={n_failures}, n={n_records}")
    if rounds <= 0:
        raise ValueError(f"rounds must be positive, got {rounds}")

    p_hat = n_failures / float(n_records)
    ci = binomtest(n_failures, n_records).proportion_ci(confidence_level=confidence, method="exact")
    denom = float(rounds) + 0.5
    return {
        "n_failures": int(n_failures),
        "n_records": int(n_records),
        "rounds": int(rounds),
        "effective_rounds_denominator": denom,
        "logical_failure_prob_per_record": p_hat,
        "logical_failure_prob_ci95_exact": (float(ci.low), float(ci.high)),
        "per_round_ler_estimate": p_hat / denom,
        "per_round_ler_ci95_exact": (float(ci.low) / denom, float(ci.high) / denom),
        "estimator_kind": "authors_half_round_estimator_arxiv_2610_24077v1",
        "sparse_failure_warning": n_failures < 10,
    }


def evaluate_paired_decoders(
    test_records: Sequence[SyndromeRecord],
    baseline_predictions: Sequence[DecoderPrediction],
    candidate_predictions: Sequence[DecoderPrediction],
    *,
    processor_model: str,
    execution_mode: str = "offline_replay",
    deadline_us: Optional[float] = None,
    count_deadline_miss_as_failure: bool = False,
) -> Dict[str, Any]:
    """
    Paired evaluation and disagreement audit of baseline vs candidate decoders on
    identical held-out `SyndromeRecord` items.
    """
    if execution_mode not in VALID_MODES:
        raise ValueError(f"execution_mode must be one of {sorted(VALID_MODES)}, got {execution_mode!r}")
    if not processor_model.strip():
        raise ValueError("processor_model must be recorded for latency reproducibility")
    if not test_records:
        raise ValueError("test_records must be non-empty")

    rec_map = {r.record_id: r for r in test_records}
    base_map = {p.record_id: p for p in baseline_predictions}
    cand_map = {p.record_id: p for p in candidate_predictions}

    if set(rec_map.keys()) != set(base_map.keys()) or set(rec_map.keys()) != set(cand_map.keys()):
        raise ValueError(
            "Both baseline and candidate decoders must supply predictions for the "
            "exact same set of test record IDs (silent record exclusion is prohibited)."
        )

    rounds_set = {r.rounds for r in test_records}
    if len(rounds_set) != 1:
        raise ValueError(f"Mixed round counts in single evaluation cohort: {sorted(rounds_set)}")
    rounds = next(iter(rounds_set))

    n_base_fail = 0
    n_cand_fail = 0
    n_base_abstain = 0
    n_cand_abstain = 0
    # McNemar 2x2 discordance counts:
    # base_fail_cand_pass = candidate rescues baseline failure
    # base_pass_cand_fail = candidate introduces new failure
    base_fail_cand_pass: List[str] = []
    base_pass_cand_fail: List[str] = []
    disagreement_weights: List[int] = []
    batch_stats: Dict[str, Dict[str, int]] = {}

    base_lat = np.empty(len(test_records), dtype=np.float64)
    cand_lat = np.empty(len(test_records), dtype=np.float64)
    base_mem = 0.0
    cand_mem = 0.0

    for idx, rec in enumerate(test_records):
        bp = base_map[rec.record_id]
        cp = cand_map[rec.record_id]
        bf = bp.is_effective_failure(
            rec.true_logical_observable,
            deadline_us=deadline_us,
            count_deadline_miss_as_failure=count_deadline_miss_as_failure,
        )
        cf = cp.is_effective_failure(
            rec.true_logical_observable,
            deadline_us=deadline_us,
            count_deadline_miss_as_failure=count_deadline_miss_as_failure,
        )

        n_base_fail += int(bf)
        n_cand_fail += int(cf)
        n_base_abstain += int(bp.abstained or bp.error is not None)
        n_cand_abstain += int(cp.abstained or cp.error is not None)

        b_entry = batch_stats.setdefault(
            rec.acquisition_batch, {"n": 0, "baseline_failures": 0, "candidate_failures": 0}
        )
        b_entry["n"] += 1
        b_entry["baseline_failures"] += int(bf)
        b_entry["candidate_failures"] += int(cf)

        if bf and not cf:
            base_fail_cand_pass.append(rec.record_id)
            disagreement_weights.append(rec.syndrome_weight)
        elif not bf and cf:
            base_pass_cand_fail.append(rec.record_id)
            disagreement_weights.append(rec.syndrome_weight)

        base_lat[idx] = bp.latency_us
        cand_lat[idx] = cp.latency_us
        base_mem = max(base_mem, bp.peak_memory_kib)
        cand_mem = max(cand_mem, cp.peak_memory_kib)

    n_discordant = len(base_fail_cand_pass) + len(base_pass_cand_fail)
    if n_discordant > 0:
        mcnemar_p = float(binomtest(len(base_fail_cand_pass), n_discordant, p=0.5).pvalue)
    else:
        mcnemar_p = 1.0

    def _latency_summary(lat: np.ndarray, peak_mem: float) -> Dict[str, Any]:
        miss = float(np.mean(lat > deadline_us)) if deadline_us is not None else None
        return {
            "median_latency_us": float(np.median(lat)),
            "p99_latency_us": float(np.percentile(lat, 99)),
            "peak_memory_kib": float(peak_mem),
            "deadline_us": deadline_us,
            "deadline_miss_fraction": miss,
        }

    # Acquisition-batch cluster-robust standard error on paired failure-rate difference (cand - base)
    n_total = len(test_records)
    g_batches = len(batch_stats)
    if g_batches >= 2:
        overall_diff = (n_cand_fail - n_base_fail) / float(n_total)
        # Cluster-sum residual variance estimator across G acquisition batches:
        # Var_CR(diff_hat) = [G / (G - 1)] * sum_g ((d_g - n_g * diff_hat) / N)^2
        sq_cluster_resids = [
            ((st["candidate_failures"] - st["baseline_failures"]) - st["n"] * overall_diff) ** 2
            for st in batch_stats.values()
        ]
        cluster_var = (g_batches / float(g_batches - 1)) * (sum(sq_cluster_resids) / float(n_total ** 2))
        batch_cluster_se_diff: Optional[float] = float(np.sqrt(max(0.0, cluster_var)))
    else:
        batch_cluster_se_diff = None

    return {
        "execution_mode": execution_mode,
        "processor_model": processor_model,
        "permissions": {
            "hardware_submission": False,
            "learning_from_test_set": False,
            "automatic_claim_promotion": False,
        },
        "baseline": {
            **authors_per_round_ler_estimator(n_base_fail, len(test_records), rounds=rounds),
            "abstentions_or_errors": n_base_abstain,
            **_latency_summary(base_lat, base_mem),
        },
        "candidate": {
            **authors_per_round_ler_estimator(n_cand_fail, len(test_records), rounds=rounds),
            "abstentions_or_errors": n_cand_abstain,
            **_latency_summary(cand_lat, cand_mem),
        },
        "paired_disagreement_audit": {
            "rescued_by_candidate_ids": base_fail_cand_pass,
            "regressed_by_candidate_ids": base_pass_cand_fail,
            "n_rescued": len(base_fail_cand_pass),
            "n_regressed": len(base_pass_cand_fail),
            "exact_mcnemar_two_sided_p": mcnemar_p,
            "batch_cluster_se_failure_rate_diff": batch_cluster_se_diff,
            "n_acquisition_batches": g_batches,
            "mean_disagreement_syndrome_weight": (
                float(np.mean(disagreement_weights)) if disagreement_weights else None
            ),
            "by_acquisition_batch": batch_stats,
        },
    }
