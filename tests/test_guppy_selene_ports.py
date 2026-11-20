"""Verification tests for OSIRIS-QEC-001 (24-cell + ClassicalReplay) & OSIRIS-HP-004 Guppy/Selene ports."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PREP_DIR = _REPO_ROOT / "experiments" / "qcup_prep"


def _load_mod(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _PREP_DIR / filename)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_qec001_all_24_archived_cells_and_phase_flip_d7_lambda57() -> None:
    qec_port = _load_mod("osiris_qec001_guppy_selene", "osiris_qec001_guppy_selene.py")
    assert len(qec_port.IBM_FEZ_QEC001_CELLS) == 24

    report = qec_port.analyze_archived_qec001()
    bf_scale = report["scaling"]["bit_flip_z"]
    pf_scale = report["scaling"]["phase_flip_x"]

    # Exact archived ibm_fez R=7 suppression factors (no double-R division!)
    assert abs(bf_scale["Lambda_3_to_5_eff_R7"] - 3.4615) < 0.002
    assert abs(bf_scale["Lambda_5_to_7_eff_R7"] - 1.0825) < 0.002
    assert abs(pf_scale["Lambda_3_to_5_eff_R7"] - 3.8881) < 0.002
    assert abs(pf_scale["Lambda_5_to_7_eff_R7"] - 2.2339) < 0.002
    assert abs(pf_scale["Lambda_3_to_7_eff_R7"] - 8.6855) < 0.002
    assert bf_scale["sub_threshold_5_to_7"] is True
    assert pf_scale["sub_threshold_5_to_7"] is True


def test_qec001_classical_replay_reproduces_archived_ibm_fez_bitstrings() -> None:
    qec_port = _load_mod("osiris_qec001_guppy_selene", "osiris_qec001_guppy_selene.py")
    replay = qec_port.run_port_sweep(mode="replay")

    assert replay["evidence_class"] == "HARDWARE_REPLAY"
    assert replay["source_job_id"] == "dbtl2vb2kf8s73c9a0d0"
    assert len(replay["replayed_cells"]) == 24
    assert len(replay["receipt_sha256"]) == 64

    by_key = {(c["sector"], c["distance"], c["rounds"]): c for c in replay["replayed_cells"]}
    # Verify exact Majority Vote & strong MWPM suppression on phase_flip_x d=3,5,7 at R=7
    pf3 = by_key[("phase_flip_x", 3, 7)]
    pf5 = by_key[("phase_flip_x", 5, 7)]
    pf7 = by_key[("phase_flip_x", 7, 7)]
    assert abs(pf7["pl_majority_vote"] - 0.088684) < 1e-5
    assert pf3["pl_mwpm_cal"] > pf5["pl_mwpm_cal"] > pf7["pl_mwpm_cal"]
    assert pf7["pl_mwpm_cal"] < 0.025


def test_qec001_builtin_q107_hotspot_sweep_and_nclm_telemetry() -> None:
    qec_port = _load_mod("osiris_qec001_guppy_selene", "osiris_qec001_guppy_selene.py")
    sim = qec_port.run_port_sweep(shots=1024, mode="builtin", hotspot_mult=10.0, seed=42)
    assert sim["evidence_class"] == "SIMULATED"
    assert sim["hardware"] is False
    assert len(sim["emulated_cells"]) == 24
    r7_pf = [
        c for c in sim["emulated_cells"] if c["sector"] == "phase_flip_x" and c["rounds"] == 7
    ]
    assert len(r7_pf) == 3
    # MWPM beats majority vote at d=7 R=7 under q107 hotspot noise
    d7_cell = [c for c in r7_pf if c["distance"] == 7][0]
    d3_cell = [c for c in r7_pf if c["distance"] == 3][0]
    assert d7_cell["pl_mwpm_cal"] <= d7_cell["pl_majority_vote"]
    assert d7_cell["pl_mwpm_cal"] < d3_cell["pl_mwpm_cal"]
    assert d7_cell["shannon_syndrome_bits"] > 0.0


def test_hp004_8q_9basis_port_arms_and_negative_controls() -> None:
    hp_port = _load_mod("osiris_hp004_guppy_selene", "osiris_hp004_guppy_selene.py")
    # 1. Exact noiseless check verifies theoretical fixed points
    ideal = hp_port.run_hp004_port_benchmark(
        shots_per_basis=512, p_2q=0.0, p_meas=0.0, mode="builtin", seed=7
    )
    assert ideal["arms"]["Arm A"]["f_joint_qccd_emulated"] == 1.0
    assert ideal["arms"]["Arm B"]["f_joint_qccd_emulated"] == 1.0
    assert abs(ideal["arms"]["Arm B-Def"]["f_joint_qccd_emulated"] - 0.25) < 0.04
    assert abs(ideal["arms"]["Arm B-Ctrl"]["f_joint_qccd_emulated"] - 0.25) < 0.04
    assert abs(ideal["arms"]["Arm B-Held"]["f_joint_qccd_emulated"] - 0.0625) < 0.03

    # 2. Quantinuum QCCD noise regime (p_2q=0.00079, p_meas=0.0015)
    qccd = hp_port.run_hp004_port_benchmark(
        shots_per_basis=512, p_2q=0.00079, p_meas=0.0015, mode="builtin", seed=19
    )
    assert qccd["evidence_class"] == "SIMULATED"
    assert qccd["arms"]["Arm B"]["qccd_2q_gates"] == 20
    assert qccd["arms"]["Arm B"]["heavy_hex_isa_cz_gates"] == 32
    assert qccd["arms"]["Arm B"]["f_joint_qccd_emulated"] > 0.96
    assert qccd["arms"]["Arm B"]["exceeds_vollbrecht_werner_1ebit_0p68391"] is True
    assert qccd["arms"]["Arm B-Def"]["exceeds_k3_entanglement_0p50"] is False
    assert qccd["arms"]["Arm B-Held"]["exceeds_k3_entanglement_0p50"] is False
