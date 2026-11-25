"""Node Alpha state helper, contract version 1.

This file is the remote program. The workstation invokes it as an argv list
(``python3 <path>``). It is not embedded in ``python -c`` and it is not
shell-quoted. A ``for`` loop here is its own statement.
"""

from __future__ import annotations

import json
import os
import platform

HELPER_VERSION = "1"
_ENV_KEYS = ("TERMUX_VERSION", "PREFIX", "OSIRIS_HOME")


def collect_state() -> dict:
    present = []
    for key in _ENV_KEYS:
        if os.environ.get(key):
            present.append(key)
    return {
        "helper_version": HELPER_VERSION,
        "machine": platform.machine(),
        "system": platform.system(),
        "env_present": present,
    }


def main() -> None:
    print(json.dumps(collect_state(), sort_keys=True))


if __name__ == "__main__":
    main()
