#!/usr/bin/env python3
"""
Anomaly Detector: Circuit-aware & finite-shot-normalized evaluation of QPU counts.

Provenance note (claims register ``ENTROPY_SUPPRESSION_Z29``):
Fixed multi-qubit toy baselines and unconditioned entropy-suppression scoring have
been removed:
  1. ``ALL_ZERO`` unmeasured registers are rejected before scoring.
  2. Finite-shot ceilings ``min(num_bits, log2(shots))`` classify near-ceiling runs
     as ``SATURATED_ENTROPY``, never as entropy suppression.
  3. Physical state-retention / divergence scoring requires an explicit per-circuit
     reference distribution (ideal target or matched control counts) and reports
     ``delta_H``, ``TVD``, and ``KL`` with empirical plug-in standard errors.
"""

from __future__ import annotations

import csv
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
from scipy import stats

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from osiris_cli import qdecode, qscore

logger = logging.getLogger(__name__)

QUARANTINED_ARTIFACTS = frozenset({"anomalies_week4.csv"})


@dataclass
class AnomalyFlag:
    """Scored register classification or circuit-baseline deviation."""
    circuit_id: str
    metric_name: str
    observed_value: float
    baseline_mean: float
    baseline_std: float
    z_score: float
    percentile: float
    permutation_p_value: float
    is_significant: bool
    anomaly_type: str
    description: str
    tvd: Optional[float] = None
    kl_bits: Optional[float] = None
    delta_entropy: Optional[float] = None
    flags: List[str] = field(default_factory=list)


class AnomalyDetector:
    """Evaluate decoded hardware registers with finite-shot and circuit-specific baselines."""

    def __init__(self):
        self.flags: List[AnomalyFlag] = []
        self.skipped_all_zero: List[str] = []

    def classify_finite_shot_registers(
        self,
        register_counts: Mapping[str, Mapping[str, int]],
        qubit_widths: Optional[Mapping[str, int]] = None,
        saturation_threshold: float = qscore.SATURATION_RATIO_THRESHOLD,
    ) -> List[AnomalyFlag]:
        """Classify hardware registers against ``min(n, log2(shots))`` after filtering ``ALL_ZERO``."""
        results: List[AnomalyFlag] = []
        for cid, counts in register_counts.items():
            if qdecode.is_all_zero_register(dict(counts)):
                self.skipped_all_zero.append(cid)
                continue
            width = qubit_widths.get(cid) if qubit_widths else None
            es = qscore.score_entropy(counts, num_bits=width, saturation_threshold=saturation_threshold)
            if qscore.ALL_ZERO in es.flags:
                self.skipped_all_zero.append(cid)
                continue
            if qscore.SATURATED_ENTROPY in es.flags:
                se = es.stderr if es.stderr > 0 else float("nan")
                z_val = (es.shannon_entropy - es.ceiling) / es.stderr if es.stderr > 0 else 0.0
                p_val = float(2.0 * stats.norm.sf(abs(z_val))) if es.stderr > 0 else 1.0
                flag = AnomalyFlag(
                    circuit_id=cid,
                    metric_name="entropy_ceiling_ratio",
                    observed_value=es.shannon_entropy,
                    baseline_mean=es.ceiling,
                    baseline_std=es.stderr,
                    z_score=float(z_val),
                    percentile=float(100.0 * es.ceiling_ratio),
                    permutation_p_value=p_val,
                    is_significant=False,
                    anomaly_type=qscore.SATURATED_ENTROPY,
                    description=(
                        f"Finite-shot entropy saturation: H={es.shannon_entropy:.4f} bits / "
                        f"ceiling={es.ceiling:.4f} bits ({100.0 * es.ceiling_ratio:.2f}%)"
                    ),
                    delta_entropy=float(es.ceiling - es.shannon_entropy),
                    flags=list(es.flags),
                )
                results.append(flag)
                self.flags.append(flag)
        return results

    def compare_against_circuit_baseline(
        self,
        circuit_id: str,
        observed_counts: Mapping[str, int],
        *,
        ideal_probs: Optional[Mapping[str, float]] = None,
        control_counts: Optional[Mapping[str, int]] = None,
        num_bits: Optional[int] = None,
        alpha: float = 0.01,
    ) -> Optional[AnomalyFlag]:
        """Compare observed counts against a circuit-specific ideal distribution or matched control.

        Refuses to invent a fixed multi-qubit baseline: at least one of ``ideal_probs``
        or ``control_counts`` must be provided. ``ALL_ZERO`` registers return ``None``.
        """
        if ideal_probs is None and control_counts is None:
            raise ValueError(
                "Circuit-specific baseline required: supply ideal_probs or control_counts "
                "(fixed qubit-count baselines are invalid)."
            )
        if qdecode.is_all_zero_register(dict(observed_counts)):
            self.skipped_all_zero.append(circuit_id)
            return None

        obs = qscore.score_entropy(observed_counts, num_bits=num_bits)
        if qscore.ALL_ZERO in obs.flags:
            self.skipped_all_zero.append(circuit_id)
            return None

        if control_counts is not None:
            ctrl = qscore.score_entropy(control_counts, num_bits=obs.num_bits)
            ctrl_total = sum(control_counts.values())
            ref_probs = {k: v / ctrl_total for k, v in control_counts.items() if v > 0}
            ref_h = ctrl.shannon_entropy
            joint_se = math.hypot(obs.stderr, ctrl.stderr)
        else:
            assert ideal_probs is not None
            ref_probs = dict(ideal_probs)
            ref_h = float(-sum(p * math.log2(p) for p in ref_probs.values() if p > 0))
            joint_se = obs.stderr

        delta_h = float(ref_h - obs.shannon_entropy)
        tvd = qscore.distribution_tvd(observed_counts, ref_probs)
        kl = qscore.distribution_kl(observed_counts, ref_probs)
        z_score = (obs.shannon_entropy - ref_h) / joint_se if joint_se > 0 else 0.0
        p_val = float(2.0 * stats.norm.sf(abs(z_score))) if joint_se > 0 else 1.0
        sig = p_val < alpha

        kind = "control_entropy_reduction" if (sig and delta_h > 0) else (
            "control_entropy_growth" if (sig and delta_h < 0) else "within_circuit_baseline"
        )
        flag = AnomalyFlag(
            circuit_id=circuit_id,
            metric_name="circuit_baseline_delta_h",
            observed_value=obs.shannon_entropy,
            baseline_mean=ref_h,
            baseline_std=joint_se,
            z_score=float(z_score),
            percentile=float(100.0 * obs.ceiling_ratio),
            permutation_p_value=p_val,
            is_significant=sig,
            anomaly_type=kind,
            description=(
                f"delta_H={delta_h:+.4f} bits vs circuit reference "
                f"(TVD={tvd:.4f}, KL={kl:.4f} bits)"
            ),
            tvd=tvd,
            kl_bits=kl,
            delta_entropy=delta_h,
            flags=list(obs.flags),
        )
        self.flags.append(flag)
        return flag

    def detect_outliers(
        self,
        metric_values: Dict[str, float],
        metric_name: str,
        threshold_zscore: float = 2.5,
    ) -> List[AnomalyFlag]:
        """Empirical cohort Z-score outlier check with Monte-Carlo rank p-value."""
        if len(metric_values) < 3:
            return []
        values = np.array(list(metric_values.values()), dtype=float)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        if std <= 0:
            return []

        anomalies: List[AnomalyFlag] = []
        for cid, value in metric_values.items():
            z_score = (value - mean) / std
            if abs(z_score) > threshold_zscore:
                percentile = float(100.0 * np.mean(values <= value))
                p_val = float(2.0 * stats.norm.sf(abs(z_score)))
                anomaly_type = f"high_{metric_name}" if z_score > 0 else f"low_{metric_name}"
                flag = AnomalyFlag(
                    circuit_id=cid,
                    metric_name=metric_name,
                    observed_value=float(value),
                    baseline_mean=mean,
                    baseline_std=std,
                    z_score=float(z_score),
                    percentile=percentile,
                    permutation_p_value=p_val,
                    is_significant=p_val < 0.01,
                    anomaly_type=anomaly_type,
                    description=f"Empirical cohort outlier in {metric_name} (z={z_score:+.2f})",
                )
                anomalies.append(flag)
                self.flags.append(flag)
        return anomalies

    def detect_entropy_suppression(self, *args: Any, **kwargs: Any) -> List[AnomalyFlag]:
        """Fail closed: unconditioned fixed-baseline entropy suppression is a refuted artifact."""
        raise RuntimeError(
            "detect_entropy_suppression with fixed baselines has been removed "
            "(claims register ENTROPY_SUPPRESSION_Z29). Use classify_finite_shot_registers() "
            "and compare_against_circuit_baseline() with explicit ideal/control distributions."
        )

    def export_flags(self, output_path: str = "hardware_entropy_report.csv"):
        """Export scored hardware flags to CSV, refusing to clobber quarantined artifacts."""
        out = Path(output_path)
        if out.name in QUARANTINED_ARTIFACTS:
            raise ValueError(
                f"Refusing to overwrite quarantined historical artifact {out.name!r}; "
                "write hardware reports to 'hardware_entropy_report.csv' instead."
            )

        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "circuit_id", "metric_name", "observed_value", "baseline_mean",
                    "baseline_std", "z_score", "percentile", "p_value", "significant",
                    "anomaly_type", "tvd", "kl_bits", "delta_entropy", "flags", "description",
                ],
            )
            writer.writeheader()
            for flag in self.flags:
                writer.writerow({
                    "circuit_id": flag.circuit_id,
                    "metric_name": flag.metric_name,
                    "observed_value": f"{flag.observed_value:.6f}",
                    "baseline_mean": f"{flag.baseline_mean:.6f}",
                    "baseline_std": f"{flag.baseline_std:.6f}",
                    "z_score": f"{flag.z_score:.2f}",
                    "percentile": f"{flag.percentile:.2f}",
                    "p_value": f"{flag.permutation_p_value:.4f}",
                    "significant": "Yes" if flag.is_significant else "No",
                    "anomaly_type": flag.anomaly_type,
                    "tvd": "" if flag.tvd is None else f"{flag.tvd:.6f}",
                    "kl_bits": "" if flag.kl_bits is None else f"{flag.kl_bits:.6f}",
                    "delta_entropy": "" if flag.delta_entropy is None else f"{flag.delta_entropy:+.6f}",
                    "flags": ",".join(flag.flags),
                    "description": flag.description,
                })

        logger.info(f"Exported {len(self.flags)} hardware records to {out}")
