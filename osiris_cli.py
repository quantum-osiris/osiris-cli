#!/usr/bin/env python3
"""
OSIRIS Sovereign Universal Entry Point
Unified REPL: Dual-Engine Living Language Console + 11D CRSM Substrate
"""
import sys
import os

# Ensure repo and submodules are in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from osiris_cli.osiris_repl import main  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() in ('bridge', 'intent'):
        from osiris_cli.edge_router import main as _edge_router_main
        raise SystemExit(_edge_router_main(sys.argv[1:]))
    main()
