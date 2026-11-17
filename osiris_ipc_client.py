#!/usr/bin/env python3
from osiris_cli.edge_ipc_client import *  # noqa: F401,F403
import sys
from osiris_cli.edge_ipc_client import send_osiris_intent

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 osiris_ipc_client.py "Your task intent here"')
        raise SystemExit(1)
    res = send_osiris_intent(' '.join(sys.argv[1:]))
    raise SystemExit(0 if res.get('ok') else 1)
