"""Deterministic tests for the recovered QEC-001 / HP-004 guppy+selene port modules
(release/production-reconciliation-20261119). The numpy/networkx analyzer paths run
against the archived hardware artifacts; the guppylang/Selene paths are exercised
only when those optional packages are installed, otherwise they must raise a clear
RuntimeError (never silently skip the port).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parent.parent / "experiments"
QEC001 = EXP / "qec001" / "osiris_qec001_guppy_selene.py"
HP004 = EXP / "hp004" / "osiris_hp004_guppy_selene.py"


def _load(path: Path, name: str):
    import sys
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


qec_mod = _load(QEC001, "osiris_qec001_guppy_selene")
hp_mod = _load(HP004, "osiris_hp004_guppy_selene")

HAS_GUPPY = importlib.util.find_spec("guppylang") is not None
HAS_SELENE = importlib.util.find_spec("selene_sim") is not None


def test_qec001_archived_scaling_reproduced():
    rows = qec_mod.analyze_archived_counts()
    assert len(rows) == 24  # 2 bases x 3 distances x 4 rounds
    cell = next(r for r in rows if r["basis"] == "bit_flip_z" and r["d"] == 7 and r["R"] == 7)
    # RESULT.md: epsilon_L(d=7, R=7, bit_flip_z) = 0.00195 +- 0.00018
    assert cell["epsilon_L"] == pytest.approx(0.00195, abs=5e-4)
    assert cell["shots"] == 16384 and cell["cz_count"] == 84
    lambdas = {(s["basis"], s["R"], s["d_from"]): s["Lambda"] for s in qec_mod.suppression_factors(rows)}
    # RESULT.md R=7 bit_flip_z: Lambda 3->5 = 3.46x, 5->7 = 1.08x
    assert lambdas[("bit_flip_z", 7, 3)] == pytest.approx(3.46, abs=0.2)
    assert lambdas[("bit_flip_z", 7, 5)] == pytest.approx(1.08, abs=0.2)


def test_hp004_arm_summaries_reproduced_from_raw_counts():
    rows = hp_mod.analyze_archived_results()
    arm_b = rows["Arm B"]
    # Archived: F_joint = 0.6334228515625, z = +50.64 above the 0.5000 SN<=2 ceiling
    assert arm_b["f_joint"] == pytest.approx(0.6334228515625, abs=1e-12)
    assert arm_b["matches_archive"] is True
    assert arm_b["z_above_SN_le_2_ceiling"] > 50.0
    assert arm_b["clears_VW_1ebit_root"] is False  # 0.6334 < 0.683900317 on ibm_marrakesh
    arm_a = rows["Arm A"]
    assert arm_a["f_joint"] == pytest.approx(0.6590847439236112, abs=1e-9)
    arm_held = rows["Arm B-Held"]
    assert arm_held["f_joint"] == pytest.approx(0.0625, abs=0.01)  # maximally mixed floor


def test_hp004_tampered_counts_detected():
    rows = hp_mod.analyze_archived_results()
    import json
    data = json.load(open(hp_mod.RESULTS))
    import ast, re
    for rec in data["circuit_records"]:
        if rec.get("kind") == "tomography" and re.search(r"_arm_b_[a-z]{2}$", rec["circuit_name"]):
            raw = rec["raw_counts"]
            counts = raw if isinstance(raw, dict) else ast.literal_eval(raw)
            k = next(iter(counts))
            counts[k] = counts[k] + 500  # inject fake shots
            rec["raw_counts"] = repr(counts)
            break
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        tmp = f.name
    try:
        tampered = hp_mod.analyze_archived_results(Path(tmp))
        assert abs(tampered["Arm B"]["f_joint"] - rows["Arm B"]["f_joint"]) > 1e-6
    finally:
        os.unlink(tmp)


def test_hp004_bad_bitstring_fails_closed():
    hp = hp_mod._load_hp004_module()
    with pytest.raises(ValueError):
        hp.pair_parities_from_counts({"012": 5})  # not 4-bit binary


@pytest.mark.skipif(HAS_GUPPY, reason="guppylang installed; testing real build instead")
def test_guppy_port_raises_clear_error_when_absent():
    with pytest.raises(RuntimeError, match="guppylang"):
        hp_mod.build_guppy_hp004_arm_b()
    with pytest.raises(RuntimeError, match="guppylang"):
        qec_mod.build_guppy_repetition_program(3, 3)


@pytest.mark.skipif(not HAS_GUPPY, reason="guppylang not installed in this environment")
def test_guppy_program_builds_when_installed():
    prog = hp_mod.build_guppy_hp004_arm_b()
    assert prog is not None


@pytest.mark.skipif(HAS_SELENE, reason="selene-sim installed; testing real replay instead")
def test_selene_replay_raises_clear_error_when_absent():
    with pytest.raises(RuntimeError, match="selene-sim"):
        qec_mod.selene_classical_replay({"XXX": {"000": 10}})
