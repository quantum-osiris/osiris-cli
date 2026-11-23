#!/usr/bin/env python3
"""
Single System Analyzer: Compute finite-shot-normalized information-theoretic metrics.

Entropy normalization uses the finite-shot ceiling ``min(num_qubits, log2(shots))``
from ``osiris_cli.qscore.entropy_ceiling`` rather than raw ``num_qubits``, preventing
finite-sample saturation (e.g. 156 qubits at 100,000 shots -> 16.51 / 16.61 bits)
from being misclassified as entropy suppression.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from osiris_cli import qscore

logger = logging.getLogger(__name__)


@dataclass
class EntropyMetrics:
    """Entropy and purity metrics for a single measured register."""
    circuit_id: str
    shannon_entropy: float
    entropy_per_qubit: float
    max_entropy: float          # theoretical infinite-shot bound: num_qubits
    entropy_ceiling: float      # observable finite-shot bound: min(num_qubits, log2(shots))
    entropy_normalized: float   # shannon_entropy / entropy_ceiling
    miller_madow_entropy: float
    purity: float
    mixedness: float
    predictability: float
    num_qubits: int
    num_bitstrings_observed: int
    shots: int
    flags: List[str] = field(default_factory=list)


@dataclass
class StatisticalSignificance:
    """Statistical comparison of observed entropy against a finite-shot baseline."""
    circuit_id: str
    observed_entropy: float
    baseline_entropy: float
    z_score: float
    p_value: float
    is_significant: bool
    effect_size: float


class SingleSystemAnalyzer:
    """Analyze finite-shot entropy and collision purity of measured quantum registers."""

    def __init__(self):
        self.results: Dict[str, EntropyMetrics] = {}

    def compute_shannon_entropy(
        self,
        counts: Dict[str, int],
        num_qubits: int,
    ) -> EntropyMetrics:
        """Compute Shannon entropy normalized against ``min(num_qubits, log2(shots))``."""
        total_shots = sum(counts.values())
        if total_shots <= 0:
            raise ValueError("counts must contain at least one shot")

        probabilities = np.array([c / total_shots for c in counts.values() if c > 0], dtype=float)
        escore = qscore.score_entropy(counts, num_bits=num_qubits)

        purity = float(np.sum(probabilities ** 2))
        predictability = float(np.max(probabilities))
        entropy_per_qubit = escore.shannon_entropy / num_qubits if num_qubits > 0 else 0.0

        return EntropyMetrics(
            circuit_id="",
            shannon_entropy=escore.shannon_entropy,
            entropy_per_qubit=float(entropy_per_qubit),
            max_entropy=float(num_qubits),
            entropy_ceiling=escore.ceiling,
            entropy_normalized=escore.ceiling_ratio,
            miller_madow_entropy=escore.miller_madow_entropy,
            purity=purity,
            mixedness=float(1.0 - purity),
            predictability=predictability,
            num_qubits=num_qubits,
            num_bitstrings_observed=escore.unique_outcomes,
            shots=total_shots,
            flags=list(escore.flags),
        )

    def estimate_purity(self, counts: Dict[str, int]) -> float:
        """Estimate computational-basis collision purity ``sum_x p_x^2``."""
        total_shots = sum(counts.values())
        probabilities = np.array([c / total_shots for c in counts.values() if c > 0], dtype=float)
        return float(np.sum(probabilities ** 2))

    def test_purity_against_baseline(
        self,
        observed_counts: Dict[str, int],
        num_qubits: int,
        baseline_entropy: Optional[float] = None,
    ) -> StatisticalSignificance:
        """Compare observed entropy against an explicit or finite-shot ceiling baseline."""
        total_shots = sum(observed_counts.values())
        escore = qscore.score_entropy(observed_counts, num_bits=num_qubits)
        ref_entropy = escore.ceiling if baseline_entropy is None else float(baseline_entropy)

        se = escore.stderr
        z_score = (escore.shannon_entropy - ref_entropy) / se if se > 0 else 0.0
        p_value = float(2.0 * stats.norm.sf(abs(z_score))) if se > 0 else 1.0

        return StatisticalSignificance(
            circuit_id="",
            observed_entropy=escore.shannon_entropy,
            baseline_entropy=ref_entropy,
            z_score=float(z_score),
            p_value=p_value,
            is_significant=p_value < 0.01,
            effect_size=float(escore.shannon_entropy - ref_entropy),
        )

    def entropy_trend_analysis(
        self,
        circuit_data_sequence: List[Dict[str, int]],
        num_qubits: int,
    ) -> Dict[str, Any]:
        entropies = [
            self.compute_shannon_entropy(counts, num_qubits).shannon_entropy
            for counts in circuit_data_sequence
        ]
        arr = np.array(entropies, dtype=float)
        indices = np.arange(len(arr))
        slope, _, r_value, _, _ = stats.linregress(indices, arr)
        trend = "increasing" if slope > 0.01 else ("decreasing" if slope < -0.01 else "flat")
        return {
            "entropies": [float(e) for e in arr],
            "trend": trend,
            "slope": float(slope),
            "correlation_with_index": float(r_value),
            "variance": float(np.var(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }

    def accessible_information(
        self,
        counts_dict: Dict[str, Dict[str, int]],
        num_qubits: int,
    ) -> float:
        all_counts: Dict[str, int] = {}
        for counts in counts_dict.values():
            for bitstring, count in counts.items():
                all_counts[bitstring] = all_counts.get(bitstring, 0) + count

        global_entropy = self.compute_shannon_entropy(all_counts, num_qubits).shannon_entropy
        total_outcomes = sum(all_counts.values())
        conditional_entropy = 0.0
        for counts in counts_dict.values():
            setting_weight = sum(counts.values()) / total_outcomes
            conditional_entropy += setting_weight * self.compute_shannon_entropy(counts, num_qubits).shannon_entropy
        return float(max(0.0, global_entropy - conditional_entropy))

    def batch_analyze(
        self,
        circuit_data: Dict[str, Dict[str, int]],
        qubit_counts: Dict[str, int],
    ) -> List[EntropyMetrics]:
        results: List[EntropyMetrics] = []
        for circuit_id, counts in circuit_data.items():
            num_qubits = qubit_counts.get(circuit_id, 0)
            if num_qubits > 0 and counts:
                metrics = self.compute_shannon_entropy(counts, num_qubits)
                metrics.circuit_id = circuit_id
                results.append(metrics)
                self.results[circuit_id] = metrics
        return results

    def export_results(self, output_path: str):
        data = {
            cid: {
                "shannon_entropy": m.shannon_entropy,
                "miller_madow_entropy": m.miller_madow_entropy,
                "entropy_per_qubit": m.entropy_per_qubit,
                "max_entropy": m.max_entropy,
                "entropy_ceiling": m.entropy_ceiling,
                "entropy_normalized": m.entropy_normalized,
                "purity": m.purity,
                "mixedness": m.mixedness,
                "predictability": m.predictability,
                "num_qubits": m.num_qubits,
                "num_bitstrings_observed": m.num_bitstrings_observed,
                "shots": m.shots,
                "flags": m.flags,
            }
            for cid, m in self.results.items()
        }
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        logger.info(f"Exported {len(self.results)} entropy analyses to {output_path}")
