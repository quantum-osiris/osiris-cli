"""One set of node identifiers for every OSIRIS module (Sprint A1.3).

The canonical names are the ones ``osiris.substrate.node_type()`` reports (they come from
``osiris_hardware_mesh``). Older names used by the capability-gap analyzer and the edge
provisioner are accepted as aliases, so existing callers and recorded action fingerprints keep
working. ``tests/test_node_ids_and_router.py`` fails if any module drifts from this list.
"""
from __future__ import annotations

from typing import Iterable, Optional, Set

NODE_ALPHA = "NODE_ALPHA_MOBILE"      # the phone: on-device model, Edge Bridge
NODE_BETA = "NODE_BETA_COMPUTE"       # the workstation: WSL2, GPU, QPU clients, test suites
NODE_GENERIC = "NODE_GENERIC_EDGE"

CANONICAL = (NODE_ALPHA, NODE_BETA, NODE_GENERIC)

ALIASES = {
    "NODE_ALPHA_EDGE": NODE_ALPHA,
    "NODE_ALPHA": NODE_ALPHA,
    "NODE_BETA_WORKSTATION": NODE_BETA,
    "NODE_BETA": NODE_BETA,
}


def canonical_node(name: object) -> Optional[str]:
    """The canonical id for ``name``, or None when it is not a known node."""
    s = str(name or "").strip().upper()
    if s in CANONICAL:
        return s
    return ALIASES.get(s)


def canonical_set(names: Iterable[object]) -> Set[str]:
    return {c for c in (canonical_node(n) for n in names) if c is not None}
