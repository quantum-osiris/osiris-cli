"""Independent forensic verification of archived OSIRIS-HP-004 hardware evidence (`Job dbti1fbsbsbs73cb2vu0`)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("qiskit")

from qiskit import qpy  # noqa: E402

_EXP_DIR = Path(__file__).resolve().parent.parent / "experiments" / "hp004"
_MOD_SPEC = importlib.util.spec_from_file_location("hp004_joint_stabilizer", _EXP_DIR / "hp004_joint_stabilizer.py")
assert _MOD_SPEC is not None and _MOD_SPEC.loader is not None
_HP004 = importlib.util.module_from_spec(_MOD_SPEC)
_MOD_SPEC.loader.exec_module(_HP004)


def test_hp004_qpy_checksum_and_physical_light_cones() -> None:
    manifest = json.loads((_EXP_DIR / "hp004_staging_manifest.json").read_text(encoding="utf-8"))
    results = json.loads((_EXP_DIR / "physical_hp004_results.json").read_text(encoding="utf-8"))
    qpy_bytes = (_EXP_DIR / "hp004_staged_circuits.qpy").read_bytes()
    qpy_sha = hashlib.sha256(qpy_bytes).hexdigest()

    assert qpy_sha == "c3daaa48d89a74cec39932c310365b3eb23660ac4922219586543f8fe473d02c"
    assert manifest["qpy_sha256"] == qpy_sha
    assert results["qpy_sha256"] == qpy_sha
    assert results["job_id"] == "dbti1fbsbsbs73cb2vu0"
    assert results["backend"] == "ibm_marrakesh"

    with open(_EXP_DIR / "hp004_staged_circuits.qpy", "rb") as fh:
        circuits = qpy.load(fh)
    assert len(circuits) == 286

    phys_tree = {34, 52, 32, 33, 39, 53, 54, 55}
    for qc, meta in zip(circuits, manifest["circuit_metadata"]):
        used = {qc.find_bit(q).index for inst in qc.data for q in inst.qubits}
        assert used.issubset(phys_tree)
        meas = tuple(
            qc.find_bit(i.qubits[0]).index
            for i in sorted((x for x in qc.data if x.operation.name == "measure"), key=lambda x: qc.find_bit(x.clbits[0]).index)
        )
        assert meas == (34, 39, 52, 53)
        if meta["kind"] == "tomography":
            bars = [idx for idx, inst in enumerate(qc.data) if inst.operation.name == "barrier"]
            dec_phys = {qc.find_bit(q).index for inst in qc.data[bars[1] + 1 : bars[2]] for q in inst.qubits}
            if meta["arm"] in ("Arm B", "Arm B-Ctrl", "Arm B-Held"):
                assert dec_phys.issubset({33, 39, 53, 54})
            elif meta["arm"] == "Arm B-Def":
                assert dec_phys.issubset({33, 39, 53})


def test_hp004_raw_counts_reproduce_all_reported_fidelities_and_hypotheses() -> None:
    results = json.loads((_EXP_DIR / "physical_hp004_results.json").read_text(encoding="utf-8"))
    records = results["circuit_records"]
    assert len(records) == 286
    assert sum(r["shots"] for r in records) == 292_864

    blocks_by_arm = {arm: [{} for _ in range(6)] for arm in _HP004.ARMS}
    for rec in records:
        counts = rec["raw_counts"]
        assert sum(counts.values()) == 1024
        if rec["kind"] == "tomography":
            blocks_by_arm[rec["arm"]][rec["block_index"]][rec["basis"]] = counts

    for arm in _HP004.ARMS:
        recomputed = _HP004.summarize_blocks(blocks_by_arm[arm])
        reported = results["arm_summaries"][arm]
        assert recomputed["f_joint"] == pytest.approx(reported["f_joint"], abs=1e-12)
        assert recomputed["f1_marginal"] == pytest.approx(reported["f1_marginal"], abs=1e-12)
        assert recomputed["f2_marginal"] == pytest.approx(reported["f2_marginal"], abs=1e-12)
        assert recomputed["f_joint_se_eff"] == pytest.approx(reported["f_joint_se_eff"], abs=1e-12)
        assert recomputed["z_above_1ebit_eff"] == pytest.approx(reported["z_above_1ebit_eff"], abs=1e-9)

    arm_b = results["arm_summaries"]["Arm B"]
    assert arm_b["f_joint"] == pytest.approx(0.6334228515625, abs=1e-9)
    assert arm_b["z_above_1ebit_eff"] > 50.0
    for h_name, h_val in results["preregistered_hypotheses"].items():
        assert h_val["passed"] is True, h_name
