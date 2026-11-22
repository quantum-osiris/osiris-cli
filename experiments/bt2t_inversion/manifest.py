#!/usr/bin/env python3
"""Offline manifest for OSIRIS-BT2T-001 milestone 1 (inversion). No provider calls, no network.

    python experiments/bt2t_inversion/manifest.py --out experiments/bt2t_inversion/manifest.json

Records: experiment ID; source commit and dirty-tree state; dependency versions; the frozen encoding and
inversion map; logical circuit (OpenQASM 2 and 3) with SHA-256; gate counts; local proxy compilations
(labelled PROXY: these are local Qiskit transpiler runs, not provider output); and the verification summary.
A hash identifies bytes. It does not establish when anything was registered or what hardware ran it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bt2t  # noqa: E402
import verify  # noqa: E402


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=HERE, check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def _version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "not installed"


def build_manifest() -> dict:
    from qiskit import qasm2, qasm3
    qc = bt2t.inversion_circuit()
    q2, q3 = qasm2.dumps(qc), qasm3.dumps(qc)
    qs = bt2t.inversion_circuit_structured()
    qs2 = qasm2.dumps(qs)
    sha = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()  # noqa: E731
    report = verify.run_all()
    return {
        "experiment_id": "OSIRIS-BT2T-001",
        "milestone": "M1: exact inversion on one 5-qubit register (offline)",
        "execution_class": "NO_EXECUTION (exact linear algebra and local proxy compilation only)",
        "preregistration": None,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"commit": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain", "--", str(HERE))),
                   "path": "experiments/bt2t_inversion"},
        "dependencies": {"python": sys.version.split()[0],
                         **{p: _version(p) for p in ("qiskit", "numpy", "pytket")}},
        "encoding": bt2t.encoding_record(),
        "logical_circuit": {"qasm2_sha256": sha(q2), "qasm3_sha256": sha(q3),
                            "gate_counts": dict(qc.count_ops()), "depth": qc.depth(), "qasm2": q2,
                            "role": "generic transposition reference"},
        "structured_circuit": {"qasm2_sha256": sha(qs2), "gate_counts": dict(qs.count_ops()),
                               "depth": qs.depth(), "qasm2": qs2,
                               "role": "candidate synthesis; must equal the reference exactly"},
        "proxy_compilations": report["compilations"],
        "verification": {k: v for k, v in report.items() if k != "compilations"},
        "next_gate": "G1 passed only if every verification check is true; hardware comparison stays BLOCKED "
                     "until a preregistration and explicit approval exist.",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    m = build_manifest()
    args.out.write_text(json.dumps(m, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    ok = all(m["verification"]["checks"].values())
    print(f"wrote {args.out}; all checks pass: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
