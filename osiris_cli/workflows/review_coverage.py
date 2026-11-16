"""
Structured Claim Ingestion & Section-Addressed Document Review Coverage
======================================================================

Fixes three governance/ingestion failure modes identified in Phase 2:
  1. Duration-vs-comparator conflation on `STAGGERED_DD` (preventing `0.946` at
     8 us and `0.888` at 16 us from being misread as treatment vs control at 8 us).
  2. Identifier collision between the literature A*-search color-code `Tesseract`
     decoder (`arXiv:2610.24077v1`) and OSIRIS's refuted 40-qubit `TESSERACT`
     circuit (`ERROR_SUPPRESSION_1E6`).
  3. Partial-document inference during manuscript review: enforces explicit
     `sections_read` vs `sections_not_read` and `external_unreviewed_claims`
     tracking so unread sections are never summarized by speculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class StaggeredDDPoint:
    duration_us: float
    estimate: float


@dataclass(frozen=True)
class StaggeredDDStructuredRecord:
    claim_id: str
    source_status: str
    verification_status: str
    platform: str
    sequence: str
    observable: str
    points: Tuple[StaggeredDDPoint, ...]
    comparator_duration_us: float
    simultaneous_approx: float
    no_dd_approx: float

    def __post_init__(self):
        durations = [p.duration_us for p in self.points]
        if len(set(durations)) != len(durations):
            raise ValueError(f"Duplicate duration_us entries in points: {durations}")
        if self.comparator_duration_us not in durations:
            raise ValueError(
                f"comparator_duration_us={self.comparator_duration_us} must match a measured duration point"
            )

    def contrast_vs_simultaneous_at_comparator(self) -> Tuple[float, float]:
        """Return (comparator_duration_us, staggered_minus_simultaneous) ONLY at matched duration."""
        matched = next(p for p in self.points if p.duration_us == self.comparator_duration_us)
        return self.comparator_duration_us, matched.estimate - self.simultaneous_approx

    def to_dict(self) -> Dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "source_status": self.source_status,
            "verification_status": self.verification_status,
            "platform": self.platform,
            "sequence": self.sequence,
            "observable": self.observable,
            "points": [
                {"duration_us": p.duration_us, "estimate": p.estimate} for p in self.points
            ],
            "comparison": {
                "duration_us": self.comparator_duration_us,
                "simultaneous_approx": self.simultaneous_approx,
                "no_dd_approx": self.no_dd_approx,
            },
        }


def build_canonical_staggered_dd_record() -> StaggeredDDStructuredRecord:
    """Return the corrected structured record for `STAGGERED_DD` (`ibm_fez` XY4)."""
    return StaggeredDDStructuredRecord(
        claim_id="STAGGERED_DD",
        source_status="user_supplied_register",
        verification_status="supported_according_to_owners_register_pending_independent_raw_audit",
        platform="ibm_fez",
        sequence="staggered_xy4",
        observable="P_plus",
        points=(
            StaggeredDDPoint(duration_us=8.0, estimate=0.946),
            StaggeredDDPoint(duration_us=16.0, estimate=0.888),
            StaggeredDDPoint(duration_us=32.0, estimate=0.795),
        ),
        comparator_duration_us=32.0,
        simultaneous_approx=0.59,
        no_dd_approx=0.55,
    )


TESSERACT_NAMESPACE_REGISTRY: Dict[str, Dict[str, str]] = {
    "TESSERACT_COLOR_CODE_DECODER_EXT": {
        "domain": "external_literature_qec_decoder",
        "reference": "arXiv:2610.24077v1",
        "description": "External A*-search syndrome decoder used in the Quantinuum color-code study.",
        "linked_osiris_claim": "NONE (must never inherit or lend evidence to ERROR_SUPPRESSION_1E6)",
        "standing": "EXTERNAL_LITERATURE_ARTIFACT",
    },
    "ERROR_SUPPRESSION_1E6": {
        "domain": "osiris_historical_40q_circuit",
        "reference": "doi:10.5281/zenodo.22870279",
        "description": "OSIRIS historical 40-qubit 'tesseract' circuit claiming 10^6x gain (refuted).",
        "linked_osiris_claim": "ERROR_SUPPRESSION_1E6",
        "standing": "REFUTED",
    },
    "TESSERACT_MAP_02": {
        "domain": "osiris_phase2_compilation_experiment",
        "reference": "experiments/tesseract_map_02/",
        "description": "Controlled compilation/routing study isolating chain vs star 4-string support.",
        "linked_osiris_claim": "PROPOSED_PREREGISTERED_BENCHMARK",
        "standing": "PROPOSED_ONLY",
    },
}


def disambiguate_tesseract_identifier(context: str) -> str:
    """Resolve ambiguous 'tesseract' mentions into strictly disjoint namespace IDs."""
    low = context.lower()
    if any(tok in low for tok in ("2610.24077", "color-code", "color code", "syndrome", "a*")):
        return "TESSERACT_COLOR_CODE_DECODER_EXT"
    if any(tok in low for tok in ("map-02", "map_02", "17.995", "5.305", "chain vs star", "chain-mapped")):
        return "TESSERACT_MAP_02"
    return "ERROR_SUPPRESSION_1E6"


@dataclass(frozen=True)
class ReviewCoverage:
    document_id: str
    declared_sections: Tuple[str, ...]
    sections_read: Tuple[str, ...]
    source_versions: Tuple[str, ...]
    independently_checked_claims: Tuple[str, ...] = ()
    unresolved_claims: Tuple[str, ...] = ()
    external_unreviewed_claims: Tuple[str, ...] = ()

    def __post_init__(self):
        declared = set(self.declared_sections)
        read = set(self.sections_read)
        unknown = read - declared
        if unknown:
            raise ValueError(f"sections_read contains undeclared sections: {sorted(unknown)}")
        if not self.source_versions:
            raise ValueError("source_versions must be non-empty")

    @property
    def sections_not_read(self) -> Tuple[str, ...]:
        read = set(self.sections_read)
        return tuple(s for s in self.declared_sections if s not in read)

    def assert_section_was_read(self, section: str) -> None:
        """Prevent summarizing or asserting findings about an unread section."""
        if section not in self.sections_read:
            raise PermissionError(
                f"Section {section!r} is marked UNREAD in {self.document_id}; "
                "inference over unread sections is prohibited."
            )

    def to_dict(self) -> Dict[str, object]:
        return {
            "document_id": self.document_id,
            "review_coverage": {
                "sections_read": list(self.sections_read),
                "sections_not_read": list(self.sections_not_read),
                "source_versions": list(self.source_versions),
                "independently_checked_claims": list(self.independently_checked_claims),
                "unresolved_claims": list(self.unresolved_claims),
                "external_unreviewed_claims": list(self.external_unreviewed_claims),
            },
        }
