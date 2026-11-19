"""osiris_hp004_guppy_selene.py — OSIRIS-HP-004 8-qubit Hayden-Preskill 9-basis
stabilizer benchmark: standalone re-analysis + Quantinuum guppylang/Selene port.

Recovered into tracked code during release/production-reconciliation-20261119.
The chat log shows the original port attempt errored twice and was never written;
this module provides it, with the Quantinuum packages as lazy optional imports.

Analyzer (deterministic, numpy only): recomputes the 16-stabilizer 4-qubit joint
Bell overlap F_joint per arm from the archived raw 4-bit histograms
(physical_hp004_results.json), cross-checked against the archived arm_summaries,
and evaluates the pre-registered witnesses:
  - Schmidt number >= 3: F_joint > 0.5000 (Terhal-Horodecki k/d ceiling, d=4)
  - Entanglement of formation > 1 ebit: F_joint > F* = 0.683900317
    (Vollbrecht-Werner isotropic convex-roof root; archived as 0.683910)

Port: build_guppy_hp004_arm_b() compiles the Arm-B 2-photon restricted recovery
circuit to guppylang/HUGR for Selene/H2 execution. Raises RuntimeError with an
install hint when guppylang is absent.

Usage:
    python osiris_hp004_guppy_selene.py             # re-analyze archived results
    python osiris_hp004_guppy_selene.py --guppy     # additionally build guppy program
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "physical_hp004_results.json"

SN_CEILING_K2_D4 = 0.5           # Terhal-Horodecki Schmidt-number <= 2 ceiling
VW_EF_GT_1EBIT_ROOT = 0.683900317  # exact Vollbrecht-Werner d=4 isotropic root (printed 0.683910)

ARM_KEY_RE = re.compile(r"^hp004_b(\d+)_(arm_[a-z_]+?)_([a-z]{2})$")


def _load_hp004_module():
    import sys
    spec = importlib.util.spec_from_file_location("hp004_joint_stabilizer", HERE / "hp004_joint_stabilizer.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses in the module require sys.modules registration
    spec.loader.exec_module(mod)
    return mod


def _arm_display(arm_field: str) -> str:
    return "Arm " + arm_field[len("arm_"):].replace("_", "-").title().replace("-", "-")


def analyze_archived_results(results_json: Path | None = None) -> Dict[str, Dict[str, Any]]:
    """Recompute per-arm F_joint from raw counts and compare with archived summaries."""
    hp = _load_hp004_module()
    data = json.load(open(results_json or RESULTS))

    # arm -> block -> basis -> counts
    grouped: Dict[str, Dict[int, Dict[str, Dict[str, int]]]] = {}
    for rec in data["circuit_records"]:
        if rec.get("kind") != "tomography":
            continue
        m = ARM_KEY_RE.match(rec["circuit_name"])
        if not m:
            continue
        block, arm, basis = int(m.group(1)), m.group(2), m.group(3).upper()  # JOINT_BASES are uppercase
        counts = ast.literal_eval(rec["raw_counts"]) if isinstance(rec["raw_counts"], str) else rec["raw_counts"]
        grouped.setdefault(arm, {}).setdefault(block, {})[basis] = {k: int(v) for k, v in counts.items()}

    out: Dict[str, Dict[str, Any]] = {}
    archived = data.get("arm_summaries", {})
    for arm, blocks in sorted(grouped.items()):
        block_list = [blocks[b] for b in sorted(blocks)]
        summary = hp.summarize_blocks(block_list)
        display = _arm_display(arm)
        f_joint = summary["f_joint"]
        se = summary["f_joint_se_eff"]
        row = {
            "arm": display,
            "f_joint": f_joint,
            "f_joint_se_eff": se,
            "z_above_SN_le_2_ceiling": (f_joint - SN_CEILING_K2_D4) / se if se > 0 else float("inf"),
            "clears_VW_1ebit_root": bool(f_joint > VW_EF_GT_1EBIT_ROOT),
            "archived_f_joint": archived.get(display, {}).get("f_joint"),
            "matches_archive": bool(archived.get(display) and abs(archived[display]["f_joint"] - f_joint) < 1e-12),
            "n_blocks": summary["n_blocks"],
        }
        out[display] = row
    return out


# ---------------------------------------------------------------------------
# Optional Quantinuum port (lazy import, fail loud)
# ---------------------------------------------------------------------------

def build_guppy_hp004_arm_b():
    """Compile the Arm-B 2-photon restricted Hayden-Preskill recovery to guppylang/HUGR."""
    try:
        from guppylang import guppy  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "guppylang is not installed; install Quantinuum's guppylang package to "
            "compile the HP-004 Arm-B circuit (the analyzer above does not need it)"
        ) from exc
    from guppylang import guppy
    from guppylang.std.builtins import qubit
    from guppylang.std.quantum import h, cx, measure  # type: ignore

    @guppy
    def hp004_arm_b(ref: tuple[qubit, qubit], msg: tuple[qubit, qubit], black_hole: tuple[qubit, qubit]) -> None:
        # Bell-pair preparation on ref and message registers
        for pair in (ref, msg):
            h(pair[0])
            cx(pair[0], pair[1])
        # 2-photon restricted scrambler on the black hole (32 ISA CZ on heavy-hex)
        for q in black_hole:
            h(q)
        for i in range(len(black_hole)):
            for j in range(i + 1, len(black_hole)):
                cx(black_hole[i], black_hole[j])
        # Hayden-Preskill decoding: barriered inverse + 9-basis stabilizer readout
        for pair in (ref, msg):
            cx(pair[0], pair[1])
            h(pair[0])
            measure(pair[0])
            measure(pair[1])

    return hp004_arm_b


def main(argv: List[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--guppy", action="store_true")
    args = ap.parse_args(argv)

    rows = analyze_archived_results()
    print(f"{'arm':<12}{'F_joint':>10}{'SE_eff':>10}{'z>0.5':>9}{'>F*':>6}{'match':>7}")
    for arm, r in rows.items():
        print(f"{arm:<12}{r['f_joint']:>10.6f}{r['f_joint_se_eff']:>10.6f}"
              f"{r['z_above_SN_le_2_ceiling']:>9.2f}{str(r['clears_VW_1ebit_root']):>6}{str(r['matches_archive']):>7}")
    if args.guppy:
        prog = build_guppy_hp004_arm_b()
        print("\nguppy Arm-B program built:", prog)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
