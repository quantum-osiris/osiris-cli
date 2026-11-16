"""Independent forensic verification of archived OSIRIS-HP-004B cross-processor evidence (`ibm_fez`, Job `dbtj61bnlcms73ajc380`)."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("qiskit")

from qiskit import qpy  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HP004_DIR = _REPO_ROOT / "experiments" / "hp004"
_FEZ_DIR = _REPO_ROOT / "experiments" / "hp004b_fez"

_MOD_SPEC = importlib.util.spec_from_file_location(
    "hp004_joint_stabilizer", _HP004_DIR / "hp004_joint_stabilizer.py"
)
assert _MOD_SPEC is not None and _MOD_SPEC.loader is not None
_HP004 = importlib.util.module_from_spec(_MOD_SPEC)
_MOD_SPEC.loader.exec_module(_HP004)


def test_hp004b_fez_preregistration_checksum_and_physical_light_cones() -> None:
    manifest = json.loads((_FEZ_DIR / "hp004b_fez_staging_manifest.json").read_text(encoding="utf-8"))
    submitted = json.loads((_FEZ_DIR / "hp004b_fez_submitted_job.json").read_text(encoding="utf-8"))
    results = json.loads((_FEZ_DIR / "physical_hp004b_fez_results.json").read_text(encoding="utf-8"))

    qpy_bytes = (_FEZ_DIR / "hp004b_fez_staged_circuits.qpy").read_bytes()
    qpy_sha = hashlib.sha256(qpy_bytes).hexdigest()

    assert qpy_sha == "6d7e5a6d17a002b5551eba35fb99b29661044477a9f6b91fb9dc51376a3025d9"
    assert manifest["qpy_sha256"] == qpy_sha
    assert submitted["qpy_sha256"] == qpy_sha
    assert results["qpy_sha256"] == qpy_sha
    assert results["job_id"] == "dbtj61bnlcms73ajc380"
    assert results["backend"] == "ibm_fez"

    # Verify offline staging preceded IBM Quantum job creation
    t_staged = dt.datetime.fromisoformat(manifest["staged_at_utc"].replace("Z", "+00:00"))
    t_created = dt.datetime.fromisoformat(
        results["job_metrics"]["timestamps"]["created"].replace("Z", "+00:00")
    )
    assert t_staged < t_created

    with open(_FEZ_DIR / "hp004b_fez_staged_circuits.qpy", "rb") as fh:
        circuits = qpy.load(fh)
    assert len(circuits) == 286

    # Pinned tree on ibm_fez: w0=2(R2), w1=24(R1), w2=4(C1), w3=3(E1), w4=16(D1), w5=23(E2), w6=22(D2), w7=21(C2)
    phys_tree = {2, 24, 4, 3, 16, 23, 22, 21}
    for qc, meta in zip(circuits, manifest["circuit_metadata"]):
        used = {qc.find_bit(q).index for inst in qc.data for q in inst.qubits}
        assert used.issubset(phys_tree)
        meas = tuple(
            qc.find_bit(i.qubits[0]).index
            for i in sorted(
                (x for x in qc.data if x.operation.name == "measure"),
                key=lambda x: qc.find_bit(x.clbits[0]).index,
            )
        )
        assert meas == (2, 16, 24, 23)
        if meta["kind"] == "tomography":
            bars = [idx for idx, inst in enumerate(qc.data) if inst.operation.name == "barrier"]
            dec_phys = {qc.find_bit(q).index for inst in qc.data[bars[1] + 1 : bars[2]] for q in inst.qubits}
            assert dec_phys.isdisjoint({2, 24})
            if meta["arm"] in ("Arm B", "Arm B-Ctrl", "Arm B-Held"):
                assert dec_phys.issubset({3, 16, 23, 22})
            elif meta["arm"] == "Arm B-Def":
                assert dec_phys.issubset({3, 16, 23})


def test_hp004b_fez_raw_counts_reproduce_all_fidelities_and_hypotheses() -> None:
    results = json.loads((_FEZ_DIR / "physical_hp004b_fez_results.json").read_text(encoding="utf-8"))
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
    assert arm_b["f_joint"] == pytest.approx(0.730712890625, abs=1e-12)
    assert arm_b["f_union_lower_bound"] > 0.63
    assert arm_b["z_above_1ebit_eff"] > 84.0
    for h_name, h_val in results["preregistered_hypotheses"].items():
        assert h_val["passed"] is True, h_name
