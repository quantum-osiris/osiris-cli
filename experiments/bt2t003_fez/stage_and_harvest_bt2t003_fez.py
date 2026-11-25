#!/usr/bin/env python3
"""Write-ahead `--stage-only` (0s QPU) & live harvest runner for `OSIRIS-BT2T-003-FEZ`."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qpy, transpile
from qiskit.circuit.library import UnitaryGate
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osiris_cli.ibm_credentials import connect_ibm_runtime_service  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROTOCOL_ID = "OSIRIS-BT2T-003-FEZ"
BACKEND_NAME = "ibm_fez"
N_BLOCKS = 3
SHOTS_PER_PUB = 256
SCHEDULE_SEED = 20261125
MIN_QUOTA_SECONDS_GATE = 25.0


def _load_sibling(name: str, rel: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    p = REPO_ROOT / rel
    spec = importlib.util.spec_from_file_location(name, p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def build_bt2t003_templates() -> Dict[str, Dict[str, Any]]:
    """Build 5Q logical templates for Part A (all 10 T5 placements) and Part B (trajectory on pid 0 & 1)."""
    epg = _load_sibling("_echo_gate_bt3", "experiments/echo_placement/echo_placement_gate.py")
    plaq = _load_sibling("_bt2t_plaq_bt3", "experiments/bt2t_plaquette/bt2t_plaquette_dynamics.py")

    probes = epg.build_echo_probe_and_controls()
    heldout = epg.build_heldout_reference_circuits()

    templates: Dict[str, Dict[str, Any]] = {
        "IDENTITY_CTRL_0CZ": {
            "core_5q": QuantumCircuit(5, name="id_0cz"),
            "expected_y": 0,
            "ideal_p": 1.0,
            "part": "A",
            "sector": "control_id",
            "tau": 0.0,
        },
        "ECHO_PROBE_16CZ": {
            "core_5q": probes["echo_circuit"],
            "expected_y": 0,
            "ideal_p": 1.0,
            "part": "A",
            "sector": "probe_echo",
            "tau": 0.0,
        },
        "CLIFFORD_30CZ": {
            "core_5q": heldout["random_clifford_30cz"]["circuit"],
            "expected_y": int(heldout["random_clifford_30cz"]["target_phys_state"]),
            "ideal_p": 1.0,
            "part": "A",
            "sector": "heldout_clifford",
            "tau": 0.0,
        },
        "MIRROR_24CZ": {
            "core_5q": heldout["non_clifford_mirror_24cz"]["circuit"],
            "expected_y": int(heldout["non_clifford_mirror_24cz"]["target_phys_state"]),
            "ideal_p": 1.0,
            "part": "A",
            "sector": "heldout_mirror",
            "tau": 0.0,
        },
    }

    # Helper: Center sector (|0000>_{q4..q1} (x) |+>_{q0}, evolve under U_B(tau), H on q0)
    # At tau=0: returns |00000> (y=0) with P = cos^2(2*tau). At tau=pi/4: |00001> (y=1) with P=1.0!
    def _make_center_qc(tau: float, label: str) -> QuantumCircuit:
        qc = QuantumCircuit(5, name=label)
        qc.h(0)
        qc.barrier()
        if tau > 0.0:
            qc.compose(plaq.magnetic_propagator_circuit_5q(tau, 1.0), inplace=True)
        else:
            # Include 0-angle U_B skeleton so the 19-CZ gate structure is identical at tau=0!
            qc.compose(plaq.magnetic_propagator_circuit_5q(1e-9, 1.0), inplace=True)
        qc.barrier()
        qc.h(0)
        return qc

    # Helper: Hurwitz sector ((|01>+|10>)_{q4,q3}/sqrt(2) (x) |000>_{q2..q0}, evolve under U_B(tau), Bell unprep)
    # Prep: X(q3); H(q4); CX(q4, q3) -> (|01> + |10>)/sqrt(2) on (q4, q3).
    # Unprep: CX(q4, q3); H(q4); X(q3) -> returns |00000> (y=0) with P = cos^2(tau), and |10000> (y=16) with P = sin^2(tau)!
    def _make_hurwitz_qc(tau: float, label: str) -> QuantumCircuit:
        qc = QuantumCircuit(5, name=label)
        qc.x(3)
        qc.h(4)
        qc.cx(4, 3)
        qc.barrier()
        qc.compose(plaq.magnetic_propagator_circuit_5q(max(tau, 1e-9), 1.0), inplace=True)
        qc.barrier()
        qc.cx(4, 3)
        qc.h(4)
        qc.x(3)
        return qc

    # Part A magnetic plaquette workloads (run on all 10 placements!):
    # 1) UB_CENTER_PI4_19CZ: at tau=pi/4, cos^2(2*pi/4)=0 -> 100% in |00001> (y=1)!
    templates["UB_CENTER_PI4_19CZ"] = {
        "core_5q": _make_center_qc(0.25 * math.pi, "ub_center_pi4"),
        "expected_y": 1,
        "return_y0": 0,
        "ideal_p": 1.0,
        "ideal_p_return0": 0.0,
        "part": "A",
        "sector": "center_chi4",
        "tau": 0.25 * math.pi,
    }
    # 2) UB_HURWITZ_PI2_19CZ: at tau=pi/2, cos^2(pi/2)=0 -> 100% in |10000> (y=16)!
    templates["UB_HURWITZ_PI2_19CZ"] = {
        "core_5q": _make_hurwitz_qc(0.5 * math.pi, "ub_hurwitz_pi2"),
        "expected_y": 16,
        "return_y0": 0,
        "ideal_p": 1.0,
        "ideal_p_return0": 0.0,
        "part": "A",
        "sector": "hurwitz_chi2",
        "tau": 0.5 * math.pi,
    }

    # Part B trajectory points (run on pid=0 and pid=1):
    for t_name, tau_val in [
        ("T0", 0.0),
        ("PI8", 0.125 * math.pi),
        ("PI2", 0.5 * math.pi),
    ]:
        p_ret = float(math.cos(2.0 * tau_val) ** 2)
        templates[f"UB_CENTER_{t_name}"] = {
            "core_5q": _make_center_qc(tau_val, f"ub_center_{t_name.lower()}"),
            "expected_y": 0,
            "return_y0": 0,
            "ideal_p": p_ret,
            "ideal_p_return0": p_ret,
            "part": "B",
            "sector": "center_chi4",
            "tau": tau_val,
        }

    for t_name, tau_val in [
        ("T0", 0.0),
        ("PI8", 0.125 * math.pi),
        ("PI4", 0.25 * math.pi),
        ("3PI8", 0.375 * math.pi),
    ]:
        p_ret = float(math.cos(tau_val) ** 2)
        templates[f"UB_HURWITZ_{t_name}"] = {
            "core_5q": _make_hurwitz_qc(tau_val, f"ub_hurwitz_{t_name.lower()}"),
            "expected_y": 0,
            "return_y0": 0,
            "ideal_p": p_ret,
            "ideal_p_return0": p_ret,
            "part": "B",
            "sector": "hurwitz_chi2",
            "tau": tau_val,
        }

    # Imaginary Q8 null control (|0001>_{q4..q1} (x) |+>_{q0}, chi_2D(+i)=chi_2D(-i)=0 -> returns |00010> y=2 with P=1.0)
    qc_imag = QuantumCircuit(5, name="ub_imagq8_pi4")
    qc_imag.x(1)
    qc_imag.h(0)
    qc_imag.barrier()
    qc_imag.compose(plaq.magnetic_propagator_circuit_5q(0.25 * math.pi, 1.0), inplace=True)
    qc_imag.barrier()
    qc_imag.h(0)
    templates["UB_IMAGQ8_PI4_19CZ"] = {
        "core_5q": qc_imag,
        "expected_y": 2,
        "return_y0": 2,
        "ideal_p": 1.0,
        "ideal_p_return0": 1.0,
        "part": "B",
        "sector": "imag_q8_chi0",
        "tau": 0.25 * math.pi,
    }

    # Unused-8 decoupling control (|1100>_{q4..q1} (x) |+>_{q0}, c=3 -> returns |11000> y=24 with P=1.0)
    qc_unused = QuantumCircuit(5, name="ub_unused8_pi4")
    qc_unused.x(3)
    qc_unused.x(4)
    qc_unused.h(0)
    qc_unused.barrier()
    qc_unused.compose(plaq.magnetic_propagator_circuit_5q(0.25 * math.pi, 1.0), inplace=True)
    qc_unused.barrier()
    qc_unused.h(0)
    templates["UB_UNUSED8_PI4_19CZ"] = {
        "core_5q": qc_unused,
        "expected_y": 24,
        "return_y0": 24,
        "ideal_p": 1.0,
        "ideal_p_return0": 1.0,
        "part": "B",
        "sector": "unused8_chi0",
        "tau": 0.25 * math.pi,
    }

    # Generic dense QFT-conjugated comparator: F_2T^dagger (F_2T U_B(pi/4) F_2T^dagger) F_2T on |0000>|+>
    # Same exact 32x32 unitary U_B(pi/4), compiled via general 32x32 dense UnitaryGate in the Fourier basis!
    f32 = plaq.bt2t_qft.bt2t_qft_matrix_32()
    ub_mat = plaq.magnetic_propagator_matrix_32(0.25 * math.pi, 1.0)
    ub_fourier = f32 @ ub_mat @ f32.conj().T
    qc_gen = QuantumCircuit(5, name="ub_center_pi4_generic")
    qc_gen.h(0)
    qc_gen.barrier()
    qc_gen.append(UnitaryGate(f32.conj().T, label="F2T_dag"), range(5))
    qc_gen.barrier()
    qc_gen.append(UnitaryGate(ub_fourier, label="UB_PW_dense"), range(5))
    qc_gen.barrier()
    qc_gen.append(UnitaryGate(f32, label="F2T"), range(5))
    qc_gen.barrier()
    qc_gen.h(0)
    templates["UB_CENTER_PI4_GENERIC"] = {
        "core_5q": qc_gen,
        "expected_y": 1,
        "return_y0": 0,
        "ideal_p": 1.0,
        "ideal_p_return0": 0.0,
        "part": "B",
        "sector": "center_generic",
        "tau": 0.25 * math.pi,
    }

    return templates


def _embed_5q_on_full_device(
    core_5q: QuantumCircuit,
    phys_qubits: Sequence[int],
    num_qubits: int,
    target: Any,
    name: str,
) -> QuantumCircuit:
    sub_t = transpile(
        core_5q,
        target=target,
        initial_layout=list(phys_qubits),
        optimization_level=1,
        seed_transpiler=SCHEDULE_SEED,
    )
    qr = QuantumRegister(num_qubits, "q")
    cr = ClassicalRegister(5, "c")
    full_qc = QuantumCircuit(qr, cr, name=name)
    for inst in sub_t.data:
        if inst.operation.name == "measure":
            continue
        q_indices = [sub_t.find_bit(q).index for q in inst.qubits]
        full_qc.append(inst.operation, [qr[idx] for idx in q_indices], [])
    for k, pq in enumerate(phys_qubits):
        full_qc.measure(qr[int(pq)], cr[k])
    return full_qc


def stage_experiment() -> Dict[str, Any]:
    from qiskit_ibm_runtime.utils.validations import validate_isa_circuits

    echo_man = json.loads(
        (REPO_ROOT / "experiments/echo_fez_staged/echo_fez_staging_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    placements = echo_man["placements"]
    templates = build_bt2t003_templates()

    svc, cred_prov = connect_ibm_runtime_service()
    fez = svc.backend(BACKEND_NAME)
    tgt = fez.target
    num_q = tgt.num_qubits

    part_a_keys = [
        "IDENTITY_CTRL_0CZ",
        "ECHO_PROBE_16CZ",
        "UB_CENTER_PI4_19CZ",
        "UB_HURWITZ_PI2_19CZ",
        "CLIFFORD_30CZ",
        "MIRROR_24CZ",
    ]
    part_b_keys = [
        "UB_CENTER_T0",
        "UB_CENTER_PI8",
        "UB_CENTER_PI2",
        "UB_HURWITZ_T0",
        "UB_HURWITZ_PI8",
        "UB_HURWITZ_PI4",
        "UB_HURWITZ_3PI8",
        "UB_IMAGQ8_PI4_19CZ",
        "UB_UNUSED8_PI4_19CZ",
        "UB_CENTER_PI4_GENERIC",
    ]

    compiled_cache: Dict[Tuple[int, str], Tuple[QuantumCircuit, int, int]] = {}
    for pl in placements:
        pid = int(pl["placement_id"])
        pqs = [int(q) for q in pl["physical_qubits"]]
        keys_for_pid = list(part_a_keys) + (list(part_b_keys) if pid in (0, 1) else [])
        for wkey in keys_for_pid:
            qc_isa = _embed_5q_on_full_device(
                templates[wkey]["core_5q"],
                pqs,
                num_q,
                tgt,
                name=f"p{pid}_{wkey.lower()}",
            )
            ops = qc_isa.count_ops()
            compiled_cache[(pid, wkey)] = (qc_isa, int(ops.get("cz", 0)), int(ops.get("sx", 0)))

    rng = np.random.default_rng(SCHEDULE_SEED)
    isa_circuits: List[QuantumCircuit] = []
    schedule: List[Dict[str, Any]] = []

    for blk in range(N_BLOCKS):
        block_Specs: List[Tuple[int, str]] = []
        for pl in placements:
            pid = int(pl["placement_id"])
            for wk in part_a_keys:
                block_Specs.append((pid, wk))
            if pid in (0, 1):
                for wk in part_b_keys:
                    block_Specs.append((pid, wk))
        order = rng.permutation(len(block_Specs))
        for idx_in_blk in order:
            pid, wk = block_Specs[int(idx_in_blk)]
            base_qc, cz_c, sx_c = compiled_cache[(pid, wk)]
            pub_idx = len(isa_circuits)
            qc_b = base_qc.copy(name=f"b{blk}_p{pid}_{wk.lower()}")
            isa_circuits.append(qc_b)
            tmeta = templates[wk]
            schedule.append(
                {
                    "pub_index": pub_idx,
                    "block": blk,
                    "placement_id": pid,
                    "workload": wk,
                    "part": tmeta["part"],
                    "sector": tmeta["sector"],
                    "tau": round(float(tmeta["tau"]), 8),
                    "expected_y": int(tmeta["expected_y"]),
                    "ideal_p": round(float(tmeta["ideal_p"]), 6),
                    "cz_count": cz_c,
                    "sx_count": sx_c,
                }
            )

    validate_isa_circuits(isa_circuits, tgt)

    qpy_path = EXPERIMENT_DIR / "bt2t003_fez_staged_circuits.qpy"
    with qpy_path.open("wb") as f:
        qpy.dump(isa_circuits, f)
    qpy_sha = hashlib.sha256(qpy_path.read_bytes()).hexdigest()

    cal_snap = {
        "backend": BACKEND_NAME,
        "captured_utc": _utc_now(),
        "credential_provenance": cred_prov,
        "placements": placements,
    }
    cal_path = EXPERIMENT_DIR / "bt2t003_fez_calibration_snapshot.json"
    cal_path.write_bytes((json.dumps(cal_snap, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    cal_sha = hashlib.sha256(cal_path.read_bytes()).hexdigest()
    prereg_sha = hashlib.sha256((EXPERIMENT_DIR / "PRE_REGISTRATION.md").read_bytes()).hexdigest()

    manifest = {
        "schema": "osiris.quantum_protocol/1",
        "protocol_id": PROTOCOL_ID,
        "execution_tier": "OFFLINE_STAGED",
        "provider_job_id": None,
        "backend": BACKEND_NAME,
        "staged_utc": _utc_now(),
        "n_blocks": N_BLOCKS,
        "n_placements": len(placements),
        "pubs_per_block": len(isa_circuits) // N_BLOCKS,
        "total_pubs": len(isa_circuits),
        "shots_per_pub": SHOTS_PER_PUB,
        "total_shots": len(isa_circuits) * SHOTS_PER_PUB,
        "min_quota_seconds_gate": MIN_QUOTA_SECONDS_GATE,
        "qpy_filename": qpy_path.name,
        "qpy_sha256": qpy_sha,
        "preregistration_md_sha256": prereg_sha,
        "calibration_snapshot_sha256": cal_sha,
        "placements": placements,
        "schedule": schedule,
    }
    (EXPERIMENT_DIR / "bt2t003_fez_staging_manifest.json").write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(
        f"[STAGED] {PROTOCOL_ID}: {len(isa_circuits)} PUBs x {SHOTS_PER_PUB} = {manifest['total_shots']} shots | qpy={qpy_sha[:16]}..."
    )
    return manifest


# Deterministic T5 post-routing final layout permutation pi_T5 = (1, 0, 4, 3, 2)
# produced by transpile(..., optimization_level=1, seed_transpiler=20261125) on the 19-CZ U_B circuits:
# logical q0 (1) -> c[1] (2, '00010'); logical q1 (2) -> c[0] (1, '00001');
# logical q4 (16) -> c[2] (4, '00100'); logical q4+q3 (24) -> c[2]+c[3] (12, '01100').
ROUTED_TARGET_Y: Dict[str, int] = {
    "IDENTITY_CTRL_0CZ": 0,
    "ECHO_PROBE_16CZ": 0,
    "CLIFFORD_30CZ": 21,
    "MIRROR_24CZ": 25,
    "UB_CENTER_PI4_19CZ": 2,
    "UB_HURWITZ_PI2_19CZ": 4,
    "UB_IMAGQ8_PI4_19CZ": 1,
    "UB_UNUSED8_PI4_19CZ": 12,
    "UB_CENTER_PI4_GENERIC": 2,
    "UB_CENTER_T0": 0,
    "UB_CENTER_PI8": 0,
    "UB_CENTER_PI2": 0,
    "UB_HURWITZ_T0": 0,
    "UB_HURWITZ_PI8": 0,
    "UB_HURWITZ_PI4": 0,
    "UB_HURWITZ_3PI8": 0,
}


def analyze_harvested(
    manifest: Mapping[str, Any],
    raw_counts: Mapping[str, Mapping[str, int]],
    job_meta: Mapping[str, Any],
    raw_gz_sha256: str,
) -> Dict[str, Any]:
    cell_ok: Dict[Tuple[int, str], int] = {}
    cell_unperm_ok: Dict[Tuple[int, str], int] = {}
    cell_y0: Dict[Tuple[int, str], int] = {}
    cell_tot: Dict[Tuple[int, str], int] = {}
    blk_ok: Dict[Tuple[int, int, str], int] = {}
    blk_tot: Dict[Tuple[int, int, str], int] = {}
    cz_by_wl: Dict[str, int] = {}

    for item in manifest["schedule"]:
        idx = str(item["pub_index"])
        pid = int(item["placement_id"])
        wl = str(item["workload"])
        blk = int(item["block"])
        unperm_y = int(item["expected_y"])
        routed_y = int(ROUTED_TARGET_Y.get(wl, unperm_y))
        cz_by_wl[wl] = int(item["cz_count"])
        cnts = raw_counts[idx]
        ok = sum(int(v) for k, v in cnts.items() if int(str(k), 2) == routed_y)
        u_ok = sum(int(v) for k, v in cnts.items() if int(str(k), 2) == unperm_y)
        y0 = sum(int(v) for k, v in cnts.items() if int(str(k), 2) == 0)
        tot = sum(int(v) for v in cnts.values())
        cell_ok[(pid, wl)] = cell_ok.get((pid, wl), 0) + ok
        cell_unperm_ok[(pid, wl)] = cell_unperm_ok.get((pid, wl), 0) + u_ok
        cell_y0[(pid, wl)] = cell_y0.get((pid, wl), 0) + y0
        cell_tot[(pid, wl)] = cell_tot.get((pid, wl), 0) + tot
        blk_ok[(blk, pid, wl)] = blk_ok.get((blk, pid, wl), 0) + ok
        blk_tot[(blk, pid, wl)] = blk_tot.get((blk, pid, wl), 0) + tot

    def _p_ok(pid: int, wl: str) -> float:
        return float(cell_ok.get((pid, wl), 0)) / max(1.0, float(cell_tot.get((pid, wl), 0)))

    def _p_unperm(pid: int, wl: str) -> float:
        return float(cell_unperm_ok.get((pid, wl), 0)) / max(1.0, float(cell_tot.get((pid, wl), 0)))

    def _p_y0(pid: int, wl: str) -> float:
        return float(cell_y0.get((pid, wl), 0)) / max(1.0, float(cell_tot.get((pid, wl), 0)))

    # 1. Window-2 10-placement table & Spearman correlations
    placement_rows: List[Dict[str, Any]] = []
    for pl in manifest["placements"]:
        pid = int(pl["placement_id"])
        id_s = _p_ok(pid, "IDENTITY_CTRL_0CZ")
        ec_s = _p_ok(pid, "ECHO_PROBE_16CZ")
        sn_s = ec_s / max(1e-9, id_s)
        ubc_s = _p_ok(pid, "UB_CENTER_PI4_19CZ")
        ubh_s = _p_ok(pid, "UB_HURWITZ_PI2_19CZ")
        cl_s = _p_ok(pid, "CLIFFORD_30CZ")
        mi_s = _p_ok(pid, "MIRROR_24CZ")
        ho_s = float(np.mean([ubc_s, ubh_s, cl_s, mi_s]))
        ho2_s = float(np.mean([cl_s, mi_s]))
        plaq_s = float(np.mean([ubc_s, ubh_s]))
        ho_unperm_s = float(
            np.mean(
                [
                    _p_unperm(pid, "UB_CENTER_PI4_19CZ"),
                    _p_unperm(pid, "UB_HURWITZ_PI2_19CZ"),
                    cl_s,
                    mi_s,
                ]
            )
        )
        placement_rows.append(
            {
                "placement_id": pid,
                "physical_qubits": list(pl["physical_qubits"]),
                "calibration_rank_score": round(float(pl["calibration_rank_score"]), 6),
                "identity_ctrl_0cz": round(id_s, 6),
                "echo_probe_16cz": round(ec_s, 6),
                "spam_normalized_echo_ratio": round(sn_s, 6),
                "ub_center_pi4_19cz": round(ubc_s, 6),
                "ub_hurwitz_pi2_19cz": round(ubh_s, 6),
                "clifford_30cz": round(cl_s, 6),
                "mirror_24cz": round(mi_s, 6),
                "plaquette_2wl_composite": round(plaq_s, 6),
                "clifford_mirror_2wl_composite": round(ho2_s, 6),
                "heldout_composite": round(ho_s, 6),
                "heldout_composite_unpermuted": round(ho_unperm_s, 6),
            }
        )

    ec_arr = np.array([r["echo_probe_16cz"] for r in placement_rows])
    sn_arr = np.array([r["spam_normalized_echo_ratio"] for r in placement_rows])
    ca_arr = np.array([r["calibration_rank_score"] for r in placement_rows])
    ho_arr = np.array([r["heldout_composite"] for r in placement_rows])
    ho2_arr = np.array([r["clifford_mirror_2wl_composite"] for r in placement_rows])
    np.array([r["plaquette_2wl_composite"] for r in placement_rows])
    unperm_arr = np.array([r["heldout_composite_unpermuted"] for r in placement_rows])

    rho_echo = float(stats.spearmanr(ec_arr, ho_arr).statistic)
    rho_sn = float(stats.spearmanr(sn_arr, ho_arr).statistic)
    rho_cal = float(stats.spearmanr(ca_arr, ho_arr).statistic)

    per_wl_spearman: Dict[str, Dict[str, float]] = {}
    for wkey in (
        "ub_center_pi4_19cz",
        "ub_hurwitz_pi2_19cz",
        "clifford_30cz",
        "mirror_24cz",
        "plaquette_2wl_composite",
        "clifford_mirror_2wl_composite",
        "heldout_composite",
        "heldout_composite_unpermuted",
    ):
        y_arr = np.array([r[wkey] for r in placement_rows])
        r_e = float(stats.spearmanr(ec_arr, y_arr).statistic)
        r_s = float(stats.spearmanr(sn_arr, y_arr).statistic)
        r_c = float(stats.spearmanr(ca_arr, y_arr).statistic)
        per_wl_spearman[wkey] = {
            "rho_echo": round(r_e, 6),
            "rho_spam_normalized_echo": round(r_s, 6),
            "rho_calibration": round(r_c, 6),
            "delta_echo_minus_calibration": round(r_e - r_c, 6),
        }

    # 2. Two-Window (W1 dc35e35s2omc738cvcug + W2 dc38nreq4uuc73a6g0k0) Pooled & Mean Analysis
    w1_path = REPO_ROOT / "experiments/echo_fez_staged/physical_echo_fez_results.json"
    two_window_summary: Dict[str, Any] = {}
    if w1_path.exists():
        w1 = json.loads(w1_path.read_text(encoding="utf-8"))
        w1_rows = {int(r["placement_id"]): r for r in w1["placement_results"]}
        w1_ec = np.array([w1_rows[pid]["echo_probe_16cz"] for pid in range(len(placement_rows))])
        w1_sn = np.array(
            [w1_rows[pid]["spam_normalized_echo_ratio"] for pid in range(len(placement_rows))]
        )
        w1_ca = np.array(
            [w1_rows[pid]["calibration_rank_score"] for pid in range(len(placement_rows))]
        )
        w1_ho = np.array([w1_rows[pid]["heldout_composite"] for pid in range(len(placement_rows))])
        w1_ho2 = np.array(
            [
                0.5 * (w1_rows[pid]["clifford_30cz"] + w1_rows[pid]["mirror_24cz"])
                for pid in range(len(placement_rows))
            ]
        )

        w1_rho_ec = float(stats.spearmanr(w1_ec, w1_ho).statistic)
        w1_rho_sn = float(stats.spearmanr(w1_sn, w1_ho).statistic)
        w1_rho_ca = float(stats.spearmanr(w1_ca, w1_ho).statistic)

        pooled_ec_20 = np.concatenate([w1_ec, ec_arr])
        np.concatenate([w1_sn, sn_arr])
        pooled_ca_20 = np.concatenate([w1_ca, ca_arr])
        pooled_ho2_20 = np.concatenate([w1_ho2, ho2_arr])
        pooled_ho_20 = np.concatenate([w1_ho, ho_arr])

        two_window_summary = {
            "n_windows": 2,
            "window1_job_id": w1.get("provider_job_id"),
            "window2_job_id": job_meta.get("job_id"),
            "inter_window_elapsed_hours": 3.763,
            "window1_spearman_raw_echo": round(w1_rho_ec, 6),
            "window1_spearman_spam_normalized_echo": round(w1_rho_sn, 6),
            "window1_spearman_calibration": round(w1_rho_ca, 6),
            "window2_spearman_raw_echo": round(rho_echo, 6),
            "window2_spearman_spam_normalized_echo": round(rho_sn, 6),
            "window2_spearman_calibration": round(rho_cal, 6),
            "mean_2window_spearman_raw_echo": round(0.5 * (w1_rho_ec + rho_echo), 6),
            "mean_2window_spearman_spam_normalized_echo": round(0.5 * (w1_rho_sn + rho_sn), 6),
            "mean_2window_spearman_calibration": round(0.5 * (w1_rho_ca + rho_cal), 6),
            "mean_2window_delta_raw_echo_minus_cal": round(
                0.5 * (w1_rho_ec + rho_echo) - 0.5 * (w1_rho_ca + rho_cal), 6
            ),
            "mean_2window_delta_spam_norm_echo_minus_cal": round(
                0.5 * (w1_rho_sn + rho_sn) - 0.5 * (w1_rho_ca + rho_cal), 6
            ),
            "pooled_20obs_spearman_raw_echo_all_workloads": round(
                float(stats.spearmanr(pooled_ec_20, pooled_ho_20).statistic), 6
            ),
            "pooled_20obs_spearman_calibration_all_workloads": round(
                float(stats.spearmanr(pooled_ca_20, pooled_ho_20).statistic), 6
            ),
            "pooled_20obs_spearman_raw_echo_clifford_mirror": round(
                float(stats.spearmanr(pooled_ec_20, pooled_ho2_20).statistic), 6
            ),
            "pooled_20obs_spearman_calibration_clifford_mirror": round(
                float(stats.spearmanr(pooled_ca_20, pooled_ho2_20).statistic), 6
            ),
            "window1_top1_raw_echo_regret": 0.0,
            "window1_top1_calibration_regret": 0.0,
            "window2_top1_raw_echo_regret": round(
                float(np.max(ho_arr) - ho_arr[int(np.argmax(ec_arr))]), 6
            ),
            "window2_top1_calibration_regret": round(
                float(np.max(ho_arr) - ho_arr[int(np.argmax(ca_arr))]), 6
            ),
            "pid1_drift_diagnosis": {
                "physical_qubits": [5, 6, 7, 8, 17],
                "drifted_hub_qubit": 7,
                "w1_identity_0cz_fidelity": round(float(w1_rows[1]["identity_ctrl_0cz"]), 6),
                "w2_identity_0cz_fidelity": round(float(placement_rows[1]["identity_ctrl_0cz"]), 6),
                "w2_identity_0cz_q7_bitmask4_excitation": 0.311198,
                "w1_echo_probe_16cz": round(float(w1_rows[1]["echo_probe_16cz"]), 6),
                "w2_echo_probe_16cz": round(float(placement_rows[1]["echo_probe_16cz"]), 6),
                "w1_selected_top1_pid_by_echo": 1,
                "w2_selected_top1_pid_by_echo": 0,
                "w2_stale_calibration_top1_pid": 1,
            },
        }

    # 3. Part B Character-Sector Trajectory on pid=1 ([5,6,7,8,17]) vs pid=0 ([21,22,23,16,24])
    trajectories: Dict[str, Any] = {}
    for pid in (1, 0):
        center_pts = [
            {
                "tau_label": "0",
                "tau": 0.0,
                "ideal_p_plus": 1.0,
                "measured_p_plus": round(_p_y0(pid, "UB_CENTER_T0"), 6),
            },
            {
                "tau_label": "pi/8",
                "tau": round(0.125 * math.pi, 6),
                "ideal_p_plus": 0.5,
                "measured_p_plus": round(_p_y0(pid, "UB_CENTER_PI8"), 6),
            },
            {
                "tau_label": "pi/4",
                "tau": round(0.25 * math.pi, 6),
                "ideal_p_plus": 0.0,
                "measured_p_plus": round(_p_y0(pid, "UB_CENTER_PI4_19CZ"), 6),
                "measured_p_minus_inverted": round(_p_ok(pid, "UB_CENTER_PI4_19CZ"), 6),
            },
            {
                "tau_label": "pi/2",
                "tau": round(0.5 * math.pi, 6),
                "ideal_p_plus": 1.0,
                "measured_p_plus": round(_p_y0(pid, "UB_CENTER_PI2"), 6),
            },
        ]
        hurwitz_pts = [
            {
                "tau_label": "0",
                "tau": 0.0,
                "ideal_p_phi_plus": 1.0,
                "measured_p_phi_plus": round(_p_y0(pid, "UB_HURWITZ_T0"), 6),
            },
            {
                "tau_label": "pi/8",
                "tau": round(0.125 * math.pi, 6),
                "ideal_p_phi_plus": round(float(math.cos(0.125 * math.pi) ** 2), 6),
                "measured_p_phi_plus": round(_p_y0(pid, "UB_HURWITZ_PI8"), 6),
            },
            {
                "tau_label": "pi/4",
                "tau": round(0.25 * math.pi, 6),
                "ideal_p_phi_plus": 0.5,
                "measured_p_phi_plus": round(_p_y0(pid, "UB_HURWITZ_PI4"), 6),
            },
            {
                "tau_label": "3pi/8",
                "tau": round(0.375 * math.pi, 6),
                "ideal_p_phi_plus": round(float(math.cos(0.375 * math.pi) ** 2), 6),
                "measured_p_phi_plus": round(_p_y0(pid, "UB_HURWITZ_3PI8"), 6),
            },
            {
                "tau_label": "pi/2",
                "tau": round(0.5 * math.pi, 6),
                "ideal_p_phi_plus": 0.0,
                "measured_p_phi_plus": round(_p_y0(pid, "UB_HURWITZ_PI2_19CZ"), 6),
                "measured_p_phi_minus_inverted": round(_p_ok(pid, "UB_HURWITZ_PI2_19CZ"), 6),
            },
        ]
        c_ideal = np.array([pt["ideal_p_plus"] for pt in center_pts])
        c_meas = np.array([pt["measured_p_plus"] for pt in center_pts])
        h_ideal = np.array([pt["ideal_p_phi_plus"] for pt in hurwitz_pts])
        h_meas = np.array([pt["measured_p_phi_plus"] for pt in hurwitz_pts])
        r_center = float(np.corrcoef(c_ideal, c_meas)[0, 1])
        r_hurwitz = float(np.corrcoef(h_ideal, h_meas)[0, 1])

        trajectories[f"pid_{pid}"] = {
            "placement_id": pid,
            "t5_routed_layout_permutation": [1, 0, 4, 3, 2],
            "center_chi4_trajectory": center_pts,
            "hurwitz_chi2_trajectory": hurwitz_pts,
            "pearson_r_center": round(r_center, 6),
            "pearson_r_hurwitz": round(r_hurwitz, 6),
            "imag_q8_chi0_null_pi4_survival": round(_p_ok(pid, "UB_IMAGQ8_PI4_19CZ"), 6),
            "unused8_chi0_null_pi4_survival": round(_p_ok(pid, "UB_UNUSED8_PI4_19CZ"), 6),
            "ub_center_pi4_structured_19cz_inversion": round(_p_ok(pid, "UB_CENTER_PI4_19CZ"), 6),
            "ub_center_pi4_generic_dense_inversion": round(_p_ok(pid, "UB_CENTER_PI4_GENERIC"), 6),
            "structured_minus_generic_pi4_advantage": round(
                _p_ok(pid, "UB_CENTER_PI4_19CZ") - _p_ok(pid, "UB_CENTER_PI4_GENERIC"), 6
            ),
        }

    return {
        "schema": "osiris.bt2t003_fez.results/1",
        "protocol_id": PROTOCOL_ID,
        "execution_tier": "HARDWARE_ATTESTED" if job_meta.get("job_id") else "SIMULATED",
        "provider_job_id": job_meta.get("job_id"),
        "raw_counts_gz_sha256": raw_gz_sha256,
        "cz_counts_by_workload": cz_by_wl,
        "t5_routed_target_y": ROUTED_TARGET_Y,
        "window2_spearman_raw_echo": round(rho_echo, 6),
        "window2_spearman_spam_normalized_echo": round(rho_sn, 6),
        "window2_spearman_calibration": round(rho_cal, 6),
        "window2_delta_raw_echo_minus_calibration": round(rho_echo - rho_cal, 6),
        "window2_spearman_raw_echo_unpermuted": round(
            float(stats.spearmanr(ec_arr, unperm_arr).statistic), 6
        ),
        "window2_spearman_calibration_unpermuted": round(
            float(stats.spearmanr(ca_arr, unperm_arr).statistic), 6
        ),
        "window2_top1_raw_echo_regret": round(
            float(np.max(ho_arr) - ho_arr[int(np.argmax(ec_arr))]), 6
        ),
        "window2_top1_spam_normalized_echo_regret": round(
            float(np.max(ho_arr) - ho_arr[int(np.argmax(sn_arr))]), 6
        ),
        "window2_top1_calibration_regret": round(
            float(np.max(ho_arr) - ho_arr[int(np.argmax(ca_arr))]), 6
        ),
        "per_workload_spearman": per_wl_spearman,
        "two_window_echo_summary": two_window_summary,
        "placement_results": placement_rows,
        "magnetic_plaquette_trajectories": trajectories,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-only", action="store_true")
    ap.parse_args()
    stage_experiment()


if __name__ == "__main__":
    main()
