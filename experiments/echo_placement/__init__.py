"""Card B (OSIRIS-ECHO): Offline noisy-simulation placement-ranking stop/go gate."""

from .echo_placement_gate import (
    DEFAULT_SEED,
    build_echo_probe_and_controls,
    build_heldout_reference_circuits,
    heron_r2_156q_coupling_edges,
    run_echo_placement_stopgo_gate,
    select_t5_candidate_placements,
    simulate_t5_noisy_probabilities,
)

__all__ = [
    "DEFAULT_SEED",
    "build_echo_probe_and_controls",
    "build_heldout_reference_circuits",
    "heron_r2_156q_coupling_edges",
    "run_echo_placement_stopgo_gate",
    "select_t5_candidate_placements",
    "simulate_t5_noisy_probabilities",
]
