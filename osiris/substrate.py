"""OSIRIS Dual-Substrate Invariant: the one place an execution horizon is resolved.

Two horizons, never conflated:

* batch / compute -- ``execution_timeout_s()``: adapters, orchestrator jobs, QPU dispatch.
  Node Alpha (ARM64 / Termux edge) 300.0 s, Node Beta (x86_64 / WSL compute) 1800.0 s,
  read from ``osiris_hardware_mesh``'s profile.
* interactive -- ``interactive_timeout_s()``: REPL children and model tool calls. 120.0 s on
  both nodes, so a stuck child can never hold the REPL for the full batch horizon.

Only the mesh's profile is read (every mesh version has it); without the mesh, ``platform``
and ``os`` probes decide. Nothing here hardcodes a path or an architecture.
"""

from __future__ import annotations

import hashlib
import math
import os
import platform
from typing import Any, Dict, Optional

from osiris_cli.node_ids import NODE_ALPHA, NODE_BETA, NODE_GENERIC

ALPHA_TIMEOUT_S = 300.0
BETA_TIMEOUT_S = 1800.0
INTERACTIVE_TIMEOUT_S = 120.0
INTERACTIVE_TIMEOUT_ENV = "OSIRIS_INTERACTIVE_TIMEOUT_S"

#: Keys every telemetry frame carries on every node, so NCLM feedback consumers never
#: branch on which substrate produced the frame.
TELEMETRY_KEYS = ("node_id", "node_type", "horizon_s", "w2_distance", "shannon_entropy")

try:
    import osiris_hardware_mesh as _mesh
except ImportError:  # pragma: no cover - partial installs only
    _mesh = None


def _profile():
    getter = getattr(_mesh, "get_current_profile", None)
    try:
        return getter() if getter else None
    except Exception:  # noqa: BLE001 - a broken probe must not take the REPL down
        return None


def mesh_available() -> bool:
    return _profile() is not None


def _arch() -> str:
    return platform.machine().lower()


def _fallback_alpha() -> bool:
    return bool(os.environ.get("TERMUX_VERSION") or os.path.exists("/data/data/com.termux")
                or _arch() in ("aarch64", "arm64"))


def node_type() -> str:
    """NODE_ALPHA_MOBILE, NODE_BETA_COMPUTE or NODE_GENERIC_EDGE."""
    p = _profile()
    if p is not None:
        return p.node_type.value
    if _fallback_alpha():
        return NODE_ALPHA
    if _arch() in ("x86_64", "amd64"):
        return NODE_BETA
    return NODE_GENERIC


def node_id() -> str:
    """Stable, non-identifying node id: a digest of hostname and architecture."""
    raw = f"{platform.node()}|{_arch()}".encode("utf-8")
    short = node_type().replace("NODE_", "").split("_")[0].lower()
    return f"{short}-{hashlib.sha256(raw).hexdigest()[:12]}"


def execution_timeout_s(override: Optional[float] = None) -> float:
    """Batch / compute horizon in seconds; ``override`` wins when given."""
    if override is not None:
        return float(override)
    p = _profile()
    if p is not None:
        return float(p.execution_timeout_s)
    return BETA_TIMEOUT_S if node_type() == NODE_BETA else ALPHA_TIMEOUT_S


def interactive_timeout_s(override: Optional[float] = None) -> float:
    """Interactive horizon: ``override``, else ``$OSIRIS_INTERACTIVE_TIMEOUT_S``, else 120 s.

    A malformed, non-finite or non-positive value falls back to the default rather than
    disabling the timeout, and the result never exceeds the batch horizon.
    """
    value = override
    if value is None:
        raw = os.environ.get(INTERACTIVE_TIMEOUT_ENV, "").strip()
        if raw:
            try:
                value = float(raw)
            except ValueError:
                value = None
    try:
        value = float(value) if value is not None else INTERACTIVE_TIMEOUT_S
    except (TypeError, ValueError):
        value = INTERACTIVE_TIMEOUT_S
    if not math.isfinite(value) or value <= 0:
        value = INTERACTIVE_TIMEOUT_S
    return min(value, execution_timeout_s())


def _metric(name: str, value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")
    return value


def get_substrate_context(
    w2_distance: Optional[float] = None,
    shannon_entropy: Optional[float] = None,
) -> Dict[str, Any]:
    """Telemetry frame with the same keys on every node. ``w2_distance`` and
    ``shannon_entropy`` are per-run measurements, ``None`` until a caller supplies them;
    they are never synthesized here."""
    return {
        "node_id": node_id(),
        "node_type": node_type(),
        "machine_arch": _arch(),
        "cpu_cores": os.cpu_count() or 1,
        "horizon_s": execution_timeout_s(),
        "interactive_horizon_s": interactive_timeout_s(),
        "w2_distance": _metric("w2_distance", w2_distance),
        "shannon_entropy": _metric("shannon_entropy", shannon_entropy),
        "mesh_available": mesh_available(),
    }


__all__ = [
    "ALPHA_TIMEOUT_S", "BETA_TIMEOUT_S", "INTERACTIVE_TIMEOUT_ENV", "INTERACTIVE_TIMEOUT_S",
    "TELEMETRY_KEYS", "execution_timeout_s", "get_substrate_context", "interactive_timeout_s",
    "mesh_available", "node_id", "node_type",
]
