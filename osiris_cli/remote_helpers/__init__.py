"""Versioned remote helpers executed on Node Alpha as files, never as shell-quoted source."""

from __future__ import annotations

HELPER_VERSION = "1"


def helper_version() -> str:
    """Return the remote-helper contract version shipped with this package."""
    return HELPER_VERSION
