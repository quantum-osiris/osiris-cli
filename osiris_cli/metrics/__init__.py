"""Verified probability and optimal-transport metrics for OSIRIS."""

from .wasserstein_verified import (
    OTResult,
    SinkhornResult,
    debiased_empirical_wasserstein,
    defective_independent_cost,
    exact_wasserstein_1d,
    exact_wasserstein_lp,
    hamming_cost_matrix,
    sinkhorn_wasserstein_bounded,
    sliced_wasserstein_hypercube,
    validate_probability_measure,
)

__all__ = [
    "OTResult",
    "SinkhornResult",
    "debiased_empirical_wasserstein",
    "defective_independent_cost",
    "exact_wasserstein_1d",
    "exact_wasserstein_lp",
    "hamming_cost_matrix",
    "sinkhorn_wasserstein_bounded",
    "sliced_wasserstein_hypercube",
    "validate_probability_measure",
]
