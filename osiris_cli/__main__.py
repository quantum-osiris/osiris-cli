import sys
if len(sys.argv) > 1 and sys.argv[1].lower() in ('bridge', 'intent'):
    from osiris_cli.edge_router import main as _edge_router_main
    raise SystemExit(_edge_router_main(sys.argv[1:]))
"""
OSIRIS Sovereign Truth Architecture - Universal CLI Module Entry Point
"""

from osiris_cli.osiris_repl import main

if __name__ == "__main__":
    main()
