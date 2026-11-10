#!/usr/bin/env python3
"""
bench_ledger_append.py
Append latency, verify time and tamper detection for the operational journal
(osiris_cli/journal.py: hash = SHA-256(prev || canonical(entry)), file-locked appends).

Everything runs on a temporary journal; the tracked journal is never touched.

Tamper cases, each applied to a fresh copy of the benchmark journal:
  edit_field      change one payload field in the middle entry
  edit_hash       replace the middle entry's stored hash
  delete_middle   remove the middle entry
  swap_adjacent   exchange two neighboring entries
  truncate_tail   remove the last 10 entries

A hash chain alone cannot detect truncate_tail: what remains is a valid chain. Detecting it needs the
tip (last hash and count) recorded somewhere the attacker cannot rewrite, e.g. signed by an operator
token. The benchmark reports that case as it is rather than hiding it.

Usage:
  python benchmarks/bench_ledger_append.py [--n 2000] [--out benchmarks/results_ledger_append.json]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from osiris_cli.journal import OperationalMerkleLedger  # noqa: E402


def _pct(sorted_ms, q):
    return sorted_ms[min(len(sorted_ms) - 1, int(q * len(sorted_ms)))]


def _tamper(lines, case):
    out = list(lines)
    mid = len(out) // 2
    if case == "edit_field":
        e = json.loads(out[mid])
        e["i"] = -1  # append() stores payload fields at the top level of the entry
        out[mid] = json.dumps(e)
    elif case == "edit_hash":
        e = json.loads(out[mid])
        e["hash"] = "f" * 64
        out[mid] = json.dumps(e)
    elif case == "delete_middle":
        del out[mid]
    elif case == "swap_adjacent":
        out[mid], out[mid + 1] = out[mid + 1], out[mid]
    elif case == "truncate_tail":
        out = out[:-10]
    else:
        raise ValueError(case)
    return out


def run(n: int) -> dict:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "bench.jsonl"
        ledger = OperationalMerkleLedger(path)
        lat = []
        for i in range(n):
            t0 = time.perf_counter()
            ledger.append("bench", {"i": i, "payload_sha256": "0" * 64})
            lat.append((time.perf_counter() - t0) * 1000.0)
        lat.sort()
        t0 = time.perf_counter()
        clean = ledger.verify()
        verify_ms = (time.perf_counter() - t0) * 1000.0

        lines = path.read_text(encoding="utf-8").splitlines()
        tamper = {}
        for case in ("edit_field", "edit_hash", "delete_middle", "swap_adjacent", "truncate_tail"):
            copy = Path(d) / f"{case}.jsonl"
            copy.write_text("\n".join(_tamper(lines, case)) + "\n", encoding="utf-8")
            fault = OperationalMerkleLedger(copy).verify()
            tamper[case] = {"detected": fault is not None, "fault": fault}

    return {
        "benchmark": "ledger_append",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": {"machine": platform.machine(), "system": platform.system(),
                 "release": platform.release(), "cpus": os.cpu_count(),
                 "python": platform.python_version()},
        "n_appends": n,
        "append_ms": {"median": round(statistics.median(lat), 4), "p95": round(_pct(lat, 0.95), 4),
                      "p99": round(_pct(lat, 0.99), 4), "max": round(lat[-1], 4)},
        "verify_ms": round(verify_ms, 2),
        "verify_clean": clean is None,
        "tamper": tamper,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default=None, help="write the result JSON here")
    args = ap.parse_args(argv)
    result = run(args.n)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0 if result["verify_clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
