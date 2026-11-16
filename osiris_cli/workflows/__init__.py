"""Bounded offline research workflow validators and structured claim ingestion."""

from .color_code_eval import (
    DecoderPrediction,
    SyndromeRecord,
    authors_per_round_ler_estimator,
    evaluate_paired_decoders,
    validate_locked_group_split,
)
from .nmr_budget import (
    NMRErrorDiagnostics,
    NMRResourceConfig,
    compute_attempted_shot_budget,
    evaluate_nmr_spectral_window_error,
    transform_spectrum_to_measure,
)
from .review_coverage import (
    ReviewCoverage,
    StaggeredDDPoint,
    StaggeredDDStructuredRecord,
    build_canonical_staggered_dd_record,
    disambiguate_tesseract_identifier,
)

__all__ = [
    "DecoderPrediction",
    "SyndromeRecord",
    "authors_per_round_ler_estimator",
    "evaluate_paired_decoders",
    "validate_locked_group_split",
    "NMRErrorDiagnostics",
    "NMRResourceConfig",
    "compute_attempted_shot_budget",
    "evaluate_nmr_spectral_window_error",
    "transform_spectrum_to_measure",
    "ReviewCoverage",
    "StaggeredDDPoint",
    "StaggeredDDStructuredRecord",
    "build_canonical_staggered_dd_record",
    "disambiguate_tesseract_identifier",
]
