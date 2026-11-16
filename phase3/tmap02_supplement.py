"""
TESSERACT-MAP-02: supplementary computations on the live analysis (not part of the locked pipeline)
===================================================================================================

The preregistration (`phase2/tesseract_map_02/TESSERACT_MAP_02_PREREGISTRATION.md`, section 3) specifies
"a two-sided paired t-test across the block differences with Holm-Bonferroni step-down FWER control at
alpha = 0.05, plus a non-parametric Wilcoxon signed-rank sensitivity check". The locked analysis in
`phase3/execute_and_analyze_live.py` produced the t-tests but not the Wilcoxon check. This module adds it
from the per-block values in `tesseract_map_02_live_analysis.json`, and re-derives each paired t-test from
the same values to confirm the block pairing.

Usage: python -m phase3.tmap02_supplement
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
from scipy import stats

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "phase3"
ENDPOINTS = {
    "classical_fidelity": ("paired_contrasts_classical_fidelity", "block_classical_fidelities"),
    "four_string_support": ("paired_contrasts_four_string_support", "block_four_string_supports"),
}


def _holm(p_values: Mapping[str, float]) -> Dict[str, float]:
    adjusted: Dict[str, float] = {}
    running = 0.0
    names = sorted(p_values, key=lambda k: p_values[k])
    for rank, name in enumerate(names):
        running = max(running, min(1.0, p_values[name] * (len(names) - rank)))
        adjusted[name] = running
    return adjusted


def wilcoxon_sensitivity(analysis: Mapping[str, Any]) -> Dict[str, Any]:
    arms = analysis["arm_summaries"]
    out: Dict[str, Any] = {}
    for endpoint, (section, block_key) in ENDPOINTS.items():
        rows: List[Dict[str, Any]] = []
        for c in analysis[section]["contrasts"]:
            a = np.asarray(arms[c["arm_a"]][block_key], dtype=np.float64)
            b = np.asarray(arms[c["arm_b"]][block_key], dtype=np.float64)
            d = a - b
            abs_d = np.abs(d)
            exact = bool(np.all(d != 0.0) and np.unique(abs_d).size == abs_d.size)
            w = stats.wilcoxon(d, alternative="two-sided", method="exact" if exact else "approx")
            t = stats.ttest_1samp(d, 0.0)
            rows.append(
                {
                    "contrast": c["contrast"],
                    "arm_a": c["arm_a"],
                    "arm_b": c["arm_b"],
                    "n_blocks": int(d.size),
                    "n_positive_differences": int(np.sum(d > 0)),
                    "wilcoxon_statistic": float(w.statistic),
                    "wilcoxon_method": "exact" if exact else "normal approximation (ties or zeros)",
                    "wilcoxon_p_two_sided": float(w.pvalue),
                    # Re-derived from the 5-decimal block values; should agree with the locked output.
                    "rederived_mean_diff": float(np.mean(d)),
                    "rederived_se_diff": float(np.std(d, ddof=1) / math.sqrt(d.size)),
                    "rederived_t_p_two_sided": float(t.pvalue),
                    "locked_mean_diff": c["mean_diff"],
                    "locked_p_two_sided_unadjusted": c["p_two_sided_unadjusted"],
                    "locked_significant_holm_0_05": c["significant_holm_0_05"],
                }
            )
        adjusted = _holm({r["contrast"]: r["wilcoxon_p_two_sided"] for r in rows})
        for r in rows:
            r["wilcoxon_p_holm_adjusted"] = adjusted[r["contrast"]]
            r["wilcoxon_significant_holm_0_05"] = adjusted[r["contrast"]] < 0.05
            r["agrees_with_locked_t_test"] = r["wilcoxon_significant_holm_0_05"] == r["locked_significant_holm_0_05"]
        out[endpoint] = rows
    return out


def main() -> None:
    analysis_path = OUT_DIR / "tesseract_map_02_live_analysis.json"
    analysis = json.loads(analysis_path.read_text())
    payload = {
        "experiment_id": "TESSERACT-MAP-02",
        "job_id": analysis["job_id"],
        "source": analysis_path.name,
        "status": "Supplementary (preregistered Wilcoxon sensitivity check; computed after the data)",
        "alpha_fwer": 0.05,
        "wilcoxon_signed_rank_holm": wilcoxon_sensitivity(analysis),
    }
    (OUT_DIR / "tesseract_map_02_live_sensitivity.json").write_text(json.dumps(payload, indent=2) + "\n")
    for endpoint, rows in payload["wilcoxon_signed_rank_holm"].items():
        for r in rows:
            print(
                f"{endpoint:20s} {r['contrast']:34s} n+={r['n_positive_differences']:2d}/{r['n_blocks']} "
                f"W={r['wilcoxon_statistic']:5.1f} p={r['wilcoxon_p_two_sided']:.5f} "
                f"p_holm={r['wilcoxon_p_holm_adjusted']:.5f} sig={r['wilcoxon_significant_holm_0_05']} "
                f"agrees={r['agrees_with_locked_t_test']} | t re-derived p={r['rederived_t_p_two_sided']:.3g} "
                f"(locked {r['locked_p_two_sided_unadjusted']:.3g}), mean {r['rederived_mean_diff']:+.5f} "
                f"(locked {r['locked_mean_diff']:+.5f})"
            )


if __name__ == "__main__":
    main()
