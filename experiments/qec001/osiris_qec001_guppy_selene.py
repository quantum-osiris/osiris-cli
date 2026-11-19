"""osiris_qec001_guppy_selene.py — standalone OSIRIS-QEC-001 analyzer + Quantinuum port.

Recovered into tracked code during release/production-reconciliation-20261119
(was chat-only; see llm-chat-dev-logs/disjointed-neeeds-resolution.txt).

Two layers:
1. Deterministic analyzer (numpy + networkx only): re-decodes the archived raw
   ibm_fez bitstrings with space-time Blossom MWPM and reports logical error per
   syndrome cycle epsilon_L(d, R) and suppression factors Lambda_{d->d+2}.
2. Optional Quantinuum port: `build_guppy_repetition_program()` compiles the same
   multi-round repetition circuit to guppylang/HUGR, and `selene_classical_replay()`
   feeds archived counts through Selene's Classical Replay. Both lazy-import the
   Quantinuum packages and raise a clear RuntimeError when they are not installed
   (they are optional; the analyzer never depends on them).

Usage:
    python osiris_qec001_guppy_selene.py            # analyze archived counts
    python osiris_qec001_guppy_selene.py --guppy    # additionally build the guppy program
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent


def _load_scaling_module():
    import sys
    spec = importlib.util.spec_from_file_location("qec001_repetition_scaling", HERE / "qec001_repetition_scaling.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses in the module require sys.modules registration
    spec.loader.exec_module(mod)
    return mod


SN_CEILING_SCHMIDT_LE_2 = 0.5  # Terhal-Horodecki: F <= k/d with k=2, d=4


def analyze_archived_counts(raw_gz: Path | None = None, cal_json: Path | None = None) -> List[Dict[str, Any]]:
    """Re-decode archived OSIRIS-QEC-001 raw counts; return one row per (basis, d, R)."""
    rs = _load_scaling_module()
    raw_gz = raw_gz or HERE / "physical_qec001_fez_raw_counts.json.gz"
    cal_json = cal_json or HERE / "qec001_fez_calibration_snapshot.json"
    priors = {int(k[1:]): v for k, v in json.load(open(cal_json))["decoder_priors"].items()}
    records = json.load(gzip.open(raw_gz))

    cells = defaultdict(list)
    for e in records:
        cells[(e["code_basis"], e["distance"], e["rounds"])].append(e)

    rows = []
    for (basis, d, R), entries in sorted(cells.items()):
        pr = priors[d]
        dec = rs.build_mwpm_decoder(d, R, p_data=pr["p_data"], p_meas=pr["p_meas"], p_hook=pr["p_hook"])
        errors = 0
        shots = 0
        cache: Dict[bytes, int] = {}
        for e in entries:
            for bitstr, cnt in e["raw_counts"].items():
                syn, dat = rs.parse_bitstring(bitstr, d, R)
                det = rs.compute_detectors(syn, dat).ravel()
                key = det.tobytes()
                flip = cache.get(key)
                if flip is None:
                    pred = rs.decode_shot_mwpm(dec, syn, dat)
                    flip = pred ^ int(dat[0])
                    cache[key] = flip
                if (int(dat[0]) ^ flip) != e["logical_state"]:
                    errors += cnt
                shots += cnt
        p_logical = errors / shots
        eps = rs.logical_error_per_cycle(p_logical, R)
        rows.append({"basis": basis, "d": d, "R": R, "shots": shots,
                     "p_logical": p_logical, "epsilon_L": eps,
                     "cz_count": entries[0]["cz_count"]})
    return rows


def suppression_factors(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Lambda_{d->d+2} per (basis, R) from analyzer rows."""
    out = []
    by = {(r["basis"], r["R"], r["d"]): r for r in rows}
    for (basis, R, d), r in sorted(by.items()):
        hi = by.get((basis, R, d + 2))
        if hi and r["epsilon_L"] > 0:
            out.append({"basis": basis, "R": R, "d_from": d, "d_to": d + 2,
                        "Lambda": r["epsilon_L"] / hi["epsilon_L"]})
    return out


# ---------------------------------------------------------------------------
# Optional Quantinuum guppylang / Selene layer (lazy imports, fail loud)
# ---------------------------------------------------------------------------

def build_guppy_repetition_program(distance: int, rounds: int):
    """Compile the multi-round 1D repetition memory to guppylang (HUGR).

    Raises RuntimeError with an install hint when guppylang is not installed.
    """
    try:
        import guppylang  # noqa: F401
        from guppylang import guppy  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "guppylang is not installed; install Quantinuum's guppylang package "
            "to build the HUGR program (the numpy analyzer above does not need it)"
        ) from exc
    from guppylang import guppy
    from guppylang.std.builtins import qubit
    from guppylang.std.quantum import h, cx, measure, reset  # type: ignore

    @guppy
    def repetition_memory(data: tuple[qubit, ...], anc: tuple[qubit, ...], rounds: int) -> None:
        for _ in range(rounds):
            for a in anc:
                h(a)
                for d in data:
                    cx(a, d)
                h(a)
                m = measure(a)
                reset(a)

    return repetition_memory


def selene_classical_replay(counts_by_setting: Dict[str, Dict[str, int]]):
    """Replay archived counts through Selene's Classical Replay simulator."""
    try:
        from selene_sim import ClassicalReplay  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "selene-sim is not installed; `pip install selene-sim` to replay "
            "archived counts without QPU access"
        ) from exc
    return ClassicalReplay(counts_by_setting)


def main(argv: List[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--guppy", action="store_true", help="also build the guppylang program (needs guppylang)")
    args = ap.parse_args(argv)

    rows = analyze_archived_counts()
    print(f"{'basis':<13}{'d':>3}{'R':>3}{'cz':>5}{'shots':>8}{'eps_L':>10}")
    for r in rows:
        print(f"{r['basis']:<13}{r['d']:>3}{r['R']:>3}{r['cz_count']:>5}{r['shots']:>8}{r['epsilon_L']:>10.5f}")
    print("\nLambda_{d->d+2}:")
    for s in suppression_factors(rows):
        print(f"  {s['basis']:<13} R={s['R']} d={s['d_from']}->{s['d_to']}: {s['Lambda']:.2f}x")

    if args.guppy:
        prog = build_guppy_repetition_program(3, 3)
        print("\nguppy program built:", prog)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
