#!/usr/bin/env python3
"""Verify and upload an OSIRIS Quantinuum Nexus disposition receipt to Quantum Agent Lab."""

from __future__ import annotations

import sys

from osiris_cli.quantinuum_cli import cmd_ledger


def main() -> int:
    return cmd_ledger(["upload"] + sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
