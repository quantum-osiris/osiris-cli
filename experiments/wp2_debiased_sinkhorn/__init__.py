"""WP2 & Direction 7.2: Split-sample-debiased Kantorovich and Sinkhorn optimal transport."""

from .debiased_ot import (
    DebiasedLPResult,
    DebiasedSinkhornResult,
    EntropicOTResult,
    defective_marginal_product_cost,
    entropic_sinkhorn_ot,
    exact_kantorovich_w2_sq,
    hamming_distance_int,
    normalized_squared_hamming_cost,
    run_wp2_benchmark_and_hardware_rescoring,
    split_sample_sinkhorn_divergence,
    split_sample_wasserstein_lp,
    split_shot_histogram,
)

__all__ = [
    "DebiasedLPResult",
    "DebiasedSinkhornResult",
    "EntropicOTResult",
    "defective_marginal_product_cost",
    "entropic_sinkhorn_ot",
    "exact_kantorovich_w2_sq",
    "hamming_distance_int",
    "normalized_squared_hamming_cost",
    "run_wp2_benchmark_and_hardware_rescoring",
    "split_sample_sinkhorn_divergence",
    "split_sample_wasserstein_lp",
    "split_shot_histogram",
]
