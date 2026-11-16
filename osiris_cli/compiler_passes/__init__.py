"""Hardware-aware compiler optimization passes for OSIRIS Quantum Systems."""

from .fanout_degree_reducer import reduce_commuting_cx_fanouts

__all__ = ["reduce_commuting_cx_fanouts"]
