#!/usr/bin/env python3
"""
Phase 1 Execution Script: Decode and analyze genuine IBM Quantum hardware results.

Provenance guardrails (claims register ``ENTROPY_SUPPRESSION_Z29``):
  - All QPU results are ingested via ``QuantumDataLoader`` -> ``osiris_cli.qdecode``.
  - ``ALL_ZERO`` unmeasured classical registers are discarded before scoring.
  - Synthetic fallback count and placeholder circuit generation have been removed:
    ``Phase1Executor`` fails closed with ``RuntimeError`` if no valid measured
    hardware circuits load.
  - Repo-root ``anomalies_week4.csv`` is preserved untouched as a quarantined artifact;
    hardware reports write to ``hardware_entropy_report.csv``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from anomaly_detector import AnomalyDetector
from correlation_analyzer import CorrelationAnalyzer
from quantum_data_loader import QuantumDataLoader
from single_system_analyzer import SingleSystemAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class Phase1Executor:
    """Orchestrate fail-closed Phase 1 analysis of decoded QPU hardware data."""

    def __init__(
        self,
        workspace_root: Optional[str | Path] = None,
        report_path: str = "hardware_entropy_report.csv",
    ):
        self.workspace_root = str(Path(workspace_root) if workspace_root is not None else _REPO_ROOT)
        self.report_path = report_path
        self.data_loader = QuantumDataLoader(self.workspace_root)
        self.entropy_analyzer = SingleSystemAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.circuits: Dict[str, Dict] = {}

    def audit_phase(self):
        """Discover decodable QPU result files (excluding venv / fake_provider)."""
        audit = self.data_loader.audit_filesystem(max_results=10000)
        print("\n--- Filesystem Audit Results ---")
        print(f"JSON files scanned: {audit['json_files']}")
        print(f"QASM files found: {audit['qasm_files']}")
        print(f"Decodable QPU job candidates: {audit['ibm_result_files']}")
        print(f"Total data size: {audit['total_size_gb']:.2f} GB")
        return audit

    def load_phase(self, num_to_load: Optional[int] = None) -> int:
        """Decode QPU job files via ``osiris_cli.qdecode`` and filter ``ALL_ZERO`` registers."""
        audit = self.data_loader.audit_filesystem()
        candidates = audit["potential_job_results"]
        if num_to_load is not None:
            candidates = candidates[:num_to_load]

        for result_path in candidates:
            self.data_loader.load_job_result(result_path)

        for cid, meta in self.data_loader.inventory.items():
            mdata = self.data_loader.measurement_data.get(cid)
            if mdata is None or not mdata.counts or meta.num_qubits <= 0:
                continue
            self.circuits[cid] = {
                "metadata": meta,
                "counts": mdata.counts,
                "analysis": {},
            }

        if not self.circuits:
            raise RuntimeError("No valid measured hardware circuits loaded.")

        logger.info(
            f"Loaded {len(self.circuits)} valid measured registers "
            f"(filtered {len(self.data_loader.filtered_all_zero)} ALL_ZERO registers)"
        )
        return len(self.circuits)

    def entropy_analysis_phase(self, output_path: Optional[str] = "entropy_analysis_results.json"):
        """Compute finite-shot-normalized Shannon entropy and collision purity on real counts."""
        if not self.circuits:
            raise RuntimeError("No valid measured hardware circuits loaded.")

        entropy_ratios: Dict[str, float] = {}
        purity_values: Dict[str, float] = {}

        for circuit_id, circuit_data in self.circuits.items():
            metadata = circuit_data["metadata"]
            counts = circuit_data.get("counts")
            if not counts:
                raise RuntimeError(f"Circuit {circuit_id!r} has no hardware counts; refusing synthetic fallback.")

            metrics = self.entropy_analyzer.compute_shannon_entropy(counts, metadata.num_qubits)
            metrics.circuit_id = circuit_id
            self.entropy_analyzer.results[circuit_id] = metrics

            entropy_ratios[circuit_id] = metrics.entropy_normalized
            purity_values[circuit_id] = metrics.purity

            circuit_data["analysis"]["entropy"] = {
                "shannon_entropy": metrics.shannon_entropy,
                "miller_madow_entropy": metrics.miller_madow_entropy,
                "entropy_per_qubit": metrics.entropy_per_qubit,
                "entropy_ceiling": metrics.entropy_ceiling,
                "entropy_normalized": metrics.entropy_normalized,
                "purity": metrics.purity,
                "mixedness": metrics.mixedness,
                "flags": metrics.flags,
            }

        if output_path:
            self.entropy_analyzer.export_results(output_path)

        return entropy_ratios, purity_values

    def correlation_analysis_phase(self, max_qubits_pairwise: int = 16):
        """Compute pairwise mutual information on bounded-width measured registers."""
        mi_results = {}
        for circuit_id, circuit_data in self.circuits.items():
            metadata = circuit_data["metadata"]
            counts = circuit_data.get("counts")
            if not counts:
                raise RuntimeError(f"Circuit {circuit_id!r} has no hardware counts.")
            if metadata.num_qubits < 2 or metadata.num_qubits > max_qubits_pairwise:
                continue

            mi_matrix = self.correlation_analyzer.all_pairs_mutual_information(counts, metadata.num_qubits)
            light_cone = self.correlation_analyzer.light_cone_analysis(counts, metadata.num_qubits)
            wass = self.correlation_analyzer.wasserstein_local_global(counts, metadata.num_qubits)
            circuit_data["analysis"]["correlations"] = {
                "mi_matrix": mi_matrix,
                "light_cone": [(lc.distance, lc.avg_correlation, lc.num_pairs) for lc in light_cone],
                "wasserstein_w2": wass.local_to_global_w2,
            }
            mi_results[circuit_id] = mi_matrix
        return mi_results

    def anomaly_detection_phase(self, entropy_ratios: Dict[str, float]):
        """Classify finite-shot saturation and empirical ratio outliers without toy baselines."""
        counts_map = {cid: cdata["counts"] for cid, cdata in self.circuits.items()}
        widths_map = {cid: cdata["metadata"].num_qubits for cid, cdata in self.circuits.items()}

        saturated = self.anomaly_detector.classify_finite_shot_registers(counts_map, widths_map)
        ratio_outliers = self.anomaly_detector.detect_outliers(
            entropy_ratios,
            metric_name="entropy_ceiling_ratio",
            threshold_zscore=2.5,
        )
        if self.report_path:
            self.anomaly_detector.export_flags(self.report_path)
        return ratio_outliers, saturated

    def full_analysis(self, run_correlations: bool = False):
        """Run fail-closed hardware audit, decode, and finite-shot entropy evaluation."""
        self.load_phase()
        entropy_ratios, _ = self.entropy_analysis_phase()
        if run_correlations:
            self.correlation_analysis_phase()
        outliers, saturated = self.anomaly_detection_phase(entropy_ratios)

        print("\n" + "=" * 60)
        print("PHASE 1 HARDWARE ENTROPY SUMMARY")
        print("=" * 60)
        print(f"Valid measured registers analyzed: {len(self.circuits)}")
        print(f"Filtered ALL_ZERO registers: {len(self.data_loader.filtered_all_zero)}")
        print(f"Finite-shot saturated registers (SATURATED_ENTROPY): {len(saturated)}")
        print(f"Empirical ceiling-ratio outliers: {len(outliers)}")
        print(f"Report written to: {self.report_path}")
        return outliers, saturated


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Decode and score QPU hardware results")
    parser.add_argument("--audit", action="store_true", help="Audit filesystem only")
    parser.add_argument("--load-sample", type=int, metavar="N", help="Load first N job files")
    parser.add_argument("--full-analysis", action="store_true", help="Run full hardware analysis")
    parser.add_argument("--output", default="hardware_entropy_report.csv", help="Output CSV path")
    args = parser.parse_args()

    executor = Phase1Executor(report_path=args.output)
    if args.audit:
        executor.audit_phase()
    elif args.load_sample:
        executor.load_phase(num_to_load=args.load_sample)
    else:
        executor.full_analysis()


if __name__ == "__main__":
    main()
