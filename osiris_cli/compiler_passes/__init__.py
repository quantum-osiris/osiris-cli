"""Hardware-aware compiler optimization passes for OSIRIS Quantum Systems."""

from .fanout_degree_reducer import compile_with_degree_reduction, reduce_commuting_cx_fanouts

__all__ = ["compile_with_degree_reduction", "reduce_commuting_cx_fanouts"]
