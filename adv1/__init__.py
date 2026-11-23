"""Direction 6: ADV-1 Classical-Hardness Pre-Gate for Looped Non-Mirror Peaked Circuits."""

from .looped_peaked import (
    LoopedPeakedCircuit,
    build_adv0_mirror_control,
    build_looped_chorded_gadget_peaked,
    build_looped_commuting_phase_peaked,
    verify_non_mirror_signature,
)
from .pregate import (
    PregateVerdict,
    attack_bounded_mps,
    attack_causal_light_cone,
    attack_truncated_pauli_path,
    evaluate_adv1_pregate,
    run_adv1_pregate_suite,
)

__all__ = [
    "LoopedPeakedCircuit",
    "PregateVerdict",
    "attack_bounded_mps",
    "attack_causal_light_cone",
    "attack_truncated_pauli_path",
    "build_adv0_mirror_control",
    "build_looped_chorded_gadget_peaked",
    "build_looped_commuting_phase_peaked",
    "evaluate_adv1_pregate",
    "run_adv1_pregate_suite",
    "verify_non_mirror_signature",
]
