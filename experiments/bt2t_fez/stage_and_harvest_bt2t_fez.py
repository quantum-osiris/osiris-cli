#!/usr/bin/env python3
"""
stage_and_harvest_bt2t_fez.py — Preregistered 5-qubit Binary Tetrahedral Group (|2T| = 24)
exact inversion synthesis & SPAM benchmark (`OSIRIS-BT2T-001-FEZ`, `Card A`) on `ibm_fez`
(`FEZ_T5_LAYOUT = [21, 22, 23, 16, 24]`).

Default `--stage-only` spends 0 QPU seconds:
  1. Compiles `bt2t.inversion_circuit_structured()` (`72 CZ`) and `bt2t.inversion_circuit()` (`723 CZ`)
     onto `HEAVY_HEX_5Q` (`optimization_level=3, seed_transpiler=20261122`).
  2. Verifies exact 32x32 unitary equivalence (`Operator.from_circuit`) AND exact 32/32 deterministic
     computational-basis statevector + SABRE layout readout decoding across all 96 `(arm, input_x)` cells.
  3. Validates all 288 156Q ISA circuits (`3` blocks x `96` PUBs) via `validate_isa_circuits(..., fez.target)`.
  4. Freezes `bt2t_fez_staged_circuits.qpy` + SHA-256 `bt2t_fez_staging_manifest.json` BEFORE submission.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
from qiskit import QuantumCircuit, qpy, transpile
from qiskit.quantum_info import Operator, Statevector
from qiskit.transpiler import CouplingMap
from qiskit_ibm_runtime import SamplerV2
from qiskit_ibm_runtime.utils.validations import validate_isa_circuits
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osiris_cli.ibm_credentials import connect_ibm_runtime_service, redact_ibm_job_metadata  # noqa: E402

BT2T_DIR = REPO_ROOT / "experiments" / "bt2t_inversion"


def _load_bt2t_module(mod_name: str) -> Any:
    if str(BT2T_DIR) not in sys.path:
        sys.path.insert(0, str(BT2T_DIR))
    full_name = f"_bt2t_pkg_{mod_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, BT2T_DIR / f"{mod_name}.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {mod_name} from {BT2T_DIR}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


bt2t = _load_bt2t_module("bt2t")
bt2t_verify = _load_bt2t_module("verify")

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROTOCOL_ID = "OSIRIS-BT2T-001-FEZ"
BACKEND_NAME = "ibm_fez"
# Pinned 5-transmon heavy-hex T-junction isomorphic to HEAVY_HEX_5Q [(0,1),(1,2),(2,3),(2,4)]:
#   21 -- 22 -- 23(hub) -- {16, 24}
FEZ_T5_LAYOUT: Tuple[int, int, int, int, int] = (21, 22, 23, 16, 24)

TRANSPILER_SEED = 20261122
SCHEDULE_SEED = 20261123
N_BLOCKS = 3
SHOTS_PER_PUB = 96  # Equal-shot rule: 3 blocks x 96 shots/PUB = 288 shots per (arm, input_x); 27,648 total shots
ARMS: Tuple[str, str, str] = ("STRUCTURED", "GENERIC", "PREP_RO_CTRL")


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def compile_and_verify_t5_cores() -> Dict[str, Any]:
    """Compile STRUCTURED and GENERIC onto T5 and verify 32/32 exact basis + phase behavior."""
    cmap = CouplingMap(bt2t_verify.HEAVY_HEX_5Q)
    perm = bt2t.inversion_permutation()
    ref_op = Operator(bt2t.inversion_circuit())
    enc_rec = bt2t.encoding_record()
    assert enc_rec["sha256"] == "3e3b02d13539daaf431af39056d031f777a785a556eb9da21ef4ea6097dfc4c5"

    compiled: Dict[str, Dict[str, Any]] = {}
    for arm_name, logical_qc in (
        ("STRUCTURED", bt2t.inversion_circuit_structured()),
        ("GENERIC", bt2t.inversion_circuit()),
    ):
        t5 = transpile(
            logical_qc,
            basis_gates=["rz", "sx", "x", "cz"],
            coupling_map=cmap,
            optimization_level=3,
            seed_transpiler=TRANSPILER_SEED,
        )
        assert Operator.from_circuit(t5).equiv(ref_op), f"{arm_name} failed Operator.from_circuit equivalence"
        init_idx = [int(i) for i in t5.layout.initial_index_layout()]
        final_idx = [int(i) for i in t5.layout.final_index_layout()]

        core_5q = QuantumCircuit(5, name=f"{arm_name.lower()}_core5q")
        for inst in t5.data:
            q_indices = [t5.find_bit(q).index for q in inst.qubits]
            core_5q.append(inst.operation, q_indices)

        ops = {str(k): int(v) for k, v in core_5q.count_ops().items()}
        compiled[arm_name] = {
            "core_5q": core_5q,
            "init_idx": init_idx,
            "final_idx": final_idx,
            "ops": ops,
            "cz_count": int(ops.get("cz", 0)),
            "sx_count": int(ops.get("sx", 0)),
            "depth": int(core_5q.depth()),
            "expected_map": list(perm),
        }

    # PREP_RO_CTRL uses the exact final readout assignment of STRUCTURED so bit-wise SPAM aligns directly
    ctrl_idx = list(compiled["STRUCTURED"]["final_idx"])
    compiled["PREP_RO_CTRL"] = {
        "core_5q": QuantumCircuit(5, name="prep_ro_ctrl_core5q"),
        "init_idx": ctrl_idx,
        "final_idx": ctrl_idx,
        "ops": {},
        "cz_count": 0,
        "sx_count": 0,
        "depth": 0,
        "expected_map": list(range(32)),
    }

    max_inaccuracy = 0.0
    for arm_name in ARMS:
        spec = compiled[arm_name]
        init_idx = spec["init_idx"]
        final_idx = spec["final_idx"]
        exp_map = spec["expected_map"]
        for x_in in range(32):
            qc_test = QuantumCircuit(5)
            for lq in range(5):
                if (x_in >> lq) & 1:
                    qc_test.x(init_idx[lq])
            qc_test.compose(spec["core_5q"], inplace=True)
            probs = np.abs(Statevector(qc_test).data) ** 2
            t5_state = int(np.argmax(probs))
            inacc = float(1.0 - probs[t5_state])
            max_inaccuracy = max(max_inaccuracy, inacc)
            decoded_y = sum(((t5_state >> final_idx[lq]) & 1) << lq for lq in range(5))
            if decoded_y != exp_map[x_in] or inacc > 1e-11:
                raise RuntimeError(
                    f"Stage-0 noiseless verification failed on {arm_name} x={x_in}: "
                    f"decoded={decoded_y}, expected={exp_map[x_in]}, inacc={inacc}"
                )

    assert compiled["STRUCTURED"]["cz_count"] == 72, f"Expected 72 CZ, got {compiled['STRUCTURED']['cz_count']}"
    assert compiled["GENERIC"]["cz_count"] == 723, f"Expected 723 CZ, got {compiled['GENERIC']['cz_count']}"

    return {
        "compiled": compiled,
        "summary": {
            "encoding_version": enc_rec["encoding_version"],
            "encoding_sha256": enc_rec["sha256"],
            "all_96_cells_noiseless_verified": True,
            "max_noiseless_basis_inaccuracy": max_inaccuracy,
            "arms": {
                k: {
                    "cz_count": v["cz_count"],
                    "sx_count": v["sx_count"],
                    "core_depth": v["depth"],
                    "init_idx_logical_to_t5": v["init_idx"],
                    "final_idx_logical_to_t5": v["final_idx"],
                    "ops": v["ops"],
                }
                for k, v in compiled.items()
            },
        },
    }


def lift_to_fez_isa(
    arm_spec: Mapping[str, Any],
    x_in: int,
    tgt: Any,
    name: str,
) -> QuantumCircuit:
    """Lift a 5Q `(arm, x_in)` circuit onto `ibm_fez` (`FEZ_T5_LAYOUT`) in strict ISA form."""
    qc = QuantumCircuit(tgt.num_qubits, 5, name=name)
    init_idx = arm_spec["init_idx"]
    final_idx = arm_spec["final_idx"]
    core_5q: QuantumCircuit = arm_spec["core_5q"]

    for lq in range(5):
        if (x_in >> lq) & 1:
            qc.x(FEZ_T5_LAYOUT[init_idx[lq]])

    qc.barrier(*FEZ_T5_LAYOUT)

    for inst in core_5q.data:
        op_name = inst.operation.name
        t5_qs = [core_5q.find_bit(q).index for q in inst.qubits]
        phys_qs = [FEZ_T5_LAYOUT[idx] for idx in t5_qs]
        if op_name == "cz":
            pa, pb = phys_qs[0], phys_qs[1]
            if (pa, pb) in tgt["cz"]:
                qc.cz(pa, pb)
            elif (pb, pa) in tgt["cz"]:
                qc.cz(pb, pa)
            else:
                raise RuntimeError(f"Physical edge ({pa}, {pb}) missing from ibm_fez target['cz']")
        else:
            qc.append(inst.operation, phys_qs)

    if arm_spec["cz_count"] > 0:
        qc.barrier(*FEZ_T5_LAYOUT)

    for lq in range(5):
        qc.measure(FEZ_T5_LAYOUT[final_idx[lq]], lq)

    return qc


def stage_experiment() -> Dict[str, Any]:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    ver = compile_and_verify_t5_cores()
    compiled = ver["compiled"]

    svc, cred_prov = connect_ibm_runtime_service()
    fez = svc.backend(BACKEND_NAME)
    tgt = fez.target
    props = fez.properties()
    st = fez.status()

    rng = np.random.default_rng(SCHEDULE_SEED)
    isa_circuits: List[QuantumCircuit] = []
    schedule_meta: List[Dict[str, Any]] = []

    for blk in range(N_BLOCKS):
        block_items: List[Tuple[QuantumCircuit, Dict[str, Any]]] = []
        for arm_name in ARMS:
            spec = compiled[arm_name]
            exp_map = spec["expected_map"]
            for x_in in range(32):
                subspace = "valid_2T" if x_in < 24 else "unused_codeword"
                cname = f"b{blk}_{arm_name.lower()}_x{x_in:02d}"
                qc_isa = lift_to_fez_isa(spec, x_in, tgt, cname)
                block_items.append(
                    (
                        qc_isa,
                        {
                            "block": blk,
                            "arm": arm_name,
                            "input_x": x_in,
                            "input_bits_c4_to_c0": format(x_in, "05b"),
                            "expected_y": int(exp_map[x_in]),
                            "expected_bits_c4_to_c0": format(int(exp_map[x_in]), "05b"),
                            "subspace": subspace,
                        },
                    )
                )

        order = rng.permutation(len(block_items))
        for idx in order:
            qc_i, meta_i = block_items[int(idx)]
            meta_out = dict(meta_i)
            meta_out["pub_index"] = len(isa_circuits)
            meta_out["circuit_name"] = qc_i.name
            meta_out["cz_count"] = int(qc_i.count_ops().get("cz", 0))
            meta_out["sx_count"] = int(qc_i.count_ops().get("sx", 0))
            isa_circuits.append(qc_i)
            schedule_meta.append(meta_out)

    validate_isa_circuits(isa_circuits, tgt)

    qpy_path = EXPERIMENT_DIR / "bt2t_fez_staged_circuits.qpy"
    with qpy_path.open("wb") as f:
        qpy.dump(isa_circuits, f)
    qpy_sha256 = hashlib.sha256(qpy_path.read_bytes()).hexdigest()

    cal_snap = {
        "backend": BACKEND_NAME,
        "captured_utc": _utc_now(),
        "pending_jobs_at_staging": int(st.pending_jobs),
        "last_update_date": str(getattr(props, "last_update_date", None)),
        "fez_t5_layout": list(FEZ_T5_LAYOUT),
        "qubits": {
            str(pq): {
                "t1_us": round(float(props.t1(pq)) * 1e6, 3),
                "t2_us": round(float(props.t2(pq)) * 1e6, 3),
                "readout_error": round(float(props.readout_error(pq)), 6),
                "sx_error": round(float(props.gate_error("sx", [pq])), 6),
            }
            for pq in FEZ_T5_LAYOUT
        },
        "active_cz_edges": {
            f"{FEZ_T5_LAYOUT[a]}-{FEZ_T5_LAYOUT[b]}": round(
                float(
                    (
                        tgt["cz"][(FEZ_T5_LAYOUT[a], FEZ_T5_LAYOUT[b])]
                        if (FEZ_T5_LAYOUT[a], FEZ_T5_LAYOUT[b]) in tgt["cz"]
                        else tgt["cz"][(FEZ_T5_LAYOUT[b], FEZ_T5_LAYOUT[a])]
                    ).error
                    or 0.0
                ),
                6,
            )
            for a, b in ((0, 1), (1, 2), (2, 3), (2, 4))
        },
    }
    cal_path = EXPERIMENT_DIR / "bt2t_fez_calibration_snapshot.json"
    cal_path.write_text(json.dumps(cal_snap, indent=2, sort_keys=True), encoding="utf-8")

    prereg_sha = hashlib.sha256((EXPERIMENT_DIR / "PRE_REGISTRATION.md").read_bytes()).hexdigest()
    g1_sha = hashlib.sha256(
        (REPO_ROOT / "experiments" / "bt2t_inversion" / "G1_INDEPENDENT_REVIEW.md").read_bytes()
    ).hexdigest()

    manifest = {
        "schema": "osiris.quantum_protocol/1",
        "protocol_id": PROTOCOL_ID,
        "staged_utc": _utc_now(),
        "backend": BACKEND_NAME,
        "credential_provenance": cred_prov,
        "fez_t5_layout": list(FEZ_T5_LAYOUT),
        "n_blocks": N_BLOCKS,
        "pubs_per_block": 96,
        "total_pubs": len(isa_circuits),
        "shots_per_pub": SHOTS_PER_PUB,
        "shots_per_arm_input_cell": N_BLOCKS * SHOTS_PER_PUB,
        "equal_shot_rule_enforced": True,
        "total_shots": len(isa_circuits) * SHOTS_PER_PUB,
        "unit_of_inference": {
            "calibration_windows": 1,
            "scope": "within_single_calibration_window_hardware_attestation",
            "within_window_blocks": N_BLOCKS,
            "paired_valid_group_elements": 24,
        },
        "qpy_filename": qpy_path.name,
        "qpy_sha256": qpy_sha256,
        "preregistration_md_sha256": prereg_sha,
        "g1_independent_review_sha256": g1_sha,
        "calibration_snapshot_sha256": hashlib.sha256(cal_path.read_bytes()).hexdigest(),
        "noiseless_and_compilation_verification": ver["summary"],
        "schedule": schedule_meta,
    }
    man_path = EXPERIMENT_DIR / "bt2t_fez_staging_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[STAGED] {PROTOCOL_ID}: {len(isa_circuits)} PUBs x {SHOTS_PER_PUB} shots = "
        f"{len(isa_circuits) * SHOTS_PER_PUB} shots | qpy_sha256={qpy_sha256}"
    )
    return manifest


def _extract_counts_5bit(pub_result: Any) -> Dict[str, int]:
    """Extract 5-bit string ('c4c3c2c1c0') counts cleanly via Qiskit BitArray."""
    data_bin = pub_result.data
    for attr in ("c", "meas"):
        if hasattr(data_bin, attr):
            return {str(k): int(v) for k, v in getattr(data_bin, attr).get_counts().items()}
    for field_name in dir(data_bin):
        if not field_name.startswith("_"):
            val = getattr(data_bin, field_name)
            if hasattr(val, "get_counts"):
                return {str(k): int(v) for k, v in val.get_counts().items()}
    raise RuntimeError("Could not find BitArray counts in pub_result.data")


def _simplex_project(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the 32-element probability simplex."""
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, len(v) + 1)
    cond = u - cssv / ind > 0
    rho = int(ind[cond][-1])
    theta = float(cssv[cond][-1] / rho)
    return np.maximum(v - theta, 0.0)


def analyze_harvested(
    manifest: Mapping[str, Any],
    raw_counts_by_pub: Mapping[str, Mapping[str, int]],
    job_meta: Mapping[str, Any],
    raw_gz_sha256: str,
    quota_snapshot: Mapping[str, Any],
) -> Dict[str, Any]:
    sched = manifest["schedule"]

    counts_grid: Dict[Tuple[int, str, int], np.ndarray] = {}
    exp_y_by_arm_x: Dict[Tuple[str, int], int] = {}
    for item in sched:
        blk = int(item["block"])
        arm = str(item["arm"])
        x_in = int(item["input_x"])
        exp_y = int(item["expected_y"])
        exp_y_by_arm_x[(arm, x_in)] = exp_y
        vec = np.zeros(32, dtype=np.float64)
        for bstr, cnt in raw_counts_by_pub[str(item["pub_index"])].items():
            idx = int(bstr.replace(" ", "").zfill(5)[-5:], 2)
            vec[idx] += float(cnt)
        counts_grid[(blk, arm, x_in)] = vec

    ro_mat = np.zeros((32, 32), dtype=np.float64)
    for x_true in range(32):
        col = sum(counts_grid[(b, "PREP_RO_CTRL", x_true)] for b in range(N_BLOCKS))
        tot = float(np.sum(col))
        assert int(tot) == N_BLOCKS * SHOTS_PER_PUB, f"Equal-shot violation on PREP_RO_CTRL x={x_true}: {tot}"
        ro_mat[:, x_true] = col / tot
    ro_cond = float(np.linalg.cond(ro_mat))
    ro_pinv = np.linalg.pinv(ro_mat)

    arm_summaries: Dict[str, Any] = {}
    per_input_tables: Dict[str, List[Dict[str, Any]]] = {}
    block_valid24_by_arm: Dict[str, List[float]] = {a: [] for a in ARMS}
    block_unused8_by_arm: Dict[str, List[float]] = {a: [] for a in ARMS}

    for arm in ARMS:
        for b in range(N_BLOCKS):
            v24_probs = []
            u8_probs = []
            for x_in in range(32):
                vec = counts_grid[(b, arm, x_in)]
                tot = float(np.sum(vec))
                assert int(tot) == SHOTS_PER_PUB, f"Equal-shot violation: {arm} b={b} x={x_in} got {tot}"
                p_ok = float(vec[exp_y_by_arm_x[(arm, x_in)]] / tot)
                if x_in < 24:
                    v24_probs.append(p_ok)
                else:
                    u8_probs.append(p_ok)
            block_valid24_by_arm[arm].append(float(np.mean(v24_probs)))
            block_unused8_by_arm[arm].append(float(np.mean(u8_probs)))

        rows: List[Dict[str, Any]] = []
        valid24_raw: List[float] = []
        valid24_ro_mit: List[float] = []
        valid24_leakage: List[float] = []
        unused8_raw: List[float] = []
        unused8_ro_mit: List[float] = []

        q8_fixed_raw: List[float] = []
        q8_imag_raw: List[float] = []
        hurwitz16_raw: List[float] = []

        for x_in in range(32):
            pooled = sum(counts_grid[(b, arm, x_in)] for b in range(N_BLOCKS))
            tot = float(np.sum(pooled))
            p_emp = pooled / tot
            exp_y = exp_y_by_arm_x[(arm, x_in)]
            p_ok_raw = float(p_emp[exp_y])
            p_mit_vec = _simplex_project(ro_pinv @ p_emp)
            p_ok_mit = float(p_mit_vec[exp_y])
            leak_unused = float(np.sum(p_emp[24:32])) if x_in < 24 else float(np.sum(p_emp[0:24]))
            top_obs = int(np.argmax(pooled))

            rows.append(
                {
                    "input_x": x_in,
                    "subspace": "valid_2T" if x_in < 24 else "unused_codeword",
                    "expected_y": exp_y,
                    "shots": int(tot),
                    "correct_shots": int(pooled[exp_y]),
                    "p_correct_raw": round(p_ok_raw, 6),
                    "p_correct_ro_mitigated_secondary": round(p_ok_mit, 6),
                    "subspace_leakage_prob": round(leak_unused, 6),
                    "modal_output_y": top_obs,
                    "modal_matches_expected": bool(top_obs == exp_y),
                }
            )
            if x_in < 24:
                valid24_raw.append(p_ok_raw)
                valid24_ro_mit.append(p_ok_mit)
                valid24_leakage.append(leak_unused)
                if x_in < 2:
                    q8_fixed_raw.append(p_ok_raw)
                elif x_in < 8:
                    q8_imag_raw.append(p_ok_raw)
                else:
                    hurwitz16_raw.append(p_ok_raw)
            else:
                unused8_raw.append(p_ok_raw)
                unused8_ro_mit.append(p_ok_mit)

        n_cell = float(N_BLOCKS * SHOTS_PER_PUB)
        se_v24_shot = float(math.sqrt(sum(p * (1.0 - p) / n_cell for p in valid24_raw)) / 24.0)
        se_u8_shot = float(math.sqrt(sum(p * (1.0 - p) / n_cell for p in unused8_raw)) / 8.0)
        blk_v24 = block_valid24_by_arm[arm]
        blk_u8 = block_unused8_by_arm[arm]

        arm_summaries[arm] = {
            "cz_count": manifest["noiseless_and_compilation_verification"]["arms"][arm]["cz_count"],
            "sx_count": manifest["noiseless_and_compilation_verification"]["arms"][arm]["sx_count"],
            "core_depth": manifest["noiseless_and_compilation_verification"]["arms"][arm]["core_depth"],
            "valid_24_uniform": {
                "p_correct_raw": round(float(np.mean(valid24_raw)), 6),
                "within_window_sampling_se": round(se_v24_shot, 6),
                "block_values": [round(v, 6) for v in blk_v24],
                "block_sd": round(float(np.std(blk_v24, ddof=1)), 6),
                "p_correct_ro_mitigated_secondary": round(float(np.mean(valid24_ro_mit)), 6),
                "modal_majority_vote_exact_matches_out_of_24": sum(1 for r in rows[:24] if r["modal_matches_expected"]),
                "leakage_to_unused_8": round(float(np.mean(valid24_leakage)), 6),
                "sector_breakdown_raw": {
                    "q8_center_pm1_2_inputs": round(float(np.mean(q8_fixed_raw)), 6),
                    "q8_imag_ijk_6_inputs": round(float(np.mean(q8_imag_raw)), 6),
                    "hurwitz_16_inputs": round(float(np.mean(hurwitz16_raw)), 6),
                },
            },
            "unused_8_uniform": {
                "p_fixed_raw": round(float(np.mean(unused8_raw)), 6),
                "within_window_sampling_se": round(se_u8_shot, 6),
                "block_values": [round(v, 6) for v in blk_u8],
                "block_sd": round(float(np.std(blk_u8, ddof=1)), 6),
                "p_fixed_ro_mitigated_secondary": round(float(np.mean(unused8_ro_mit)), 6),
                "modal_majority_vote_exact_matches_out_of_8": sum(1 for r in rows[24:] if r["modal_matches_expected"]),
                "leakage_to_valid_24": round(float(np.mean([r["subspace_leakage_prob"] for r in rows[24:]])), 6),
            },
        }
        per_input_tables[arm] = rows

    blk_diffs_v24 = np.array(block_valid24_by_arm["STRUCTURED"]) - np.array(block_valid24_by_arm["GENERIC"])
    t_blk, p_blk_two = stats.ttest_1samp(blk_diffs_v24, popmean=0.0)
    p_blk_one = float(p_blk_two / 2.0) if t_blk > 0 else 1.0

    elem_diffs_24 = np.array(
        [
            per_input_tables["STRUCTURED"][x]["p_correct_raw"] - per_input_tables["GENERIC"][x]["p_correct_raw"]
            for x in range(24)
        ],
        dtype=np.float64,
    )
    n_elem_wins = int(np.sum(elem_diffs_24 > 0.0))
    wilcox_res = stats.wilcoxon(elem_diffs_24, alternative="two-sided", mode="exact")
    p_wilcox_two = float(wilcox_res.pvalue)

    delta_v24 = float(
        arm_summaries["STRUCTURED"]["valid_24_uniform"]["p_correct_raw"]
        - arm_summaries["GENERIC"]["valid_24_uniform"]["p_correct_raw"]
    )
    delta_u8 = float(
        arm_summaries["STRUCTURED"]["unused_8_uniform"]["p_fixed_raw"]
        - arm_summaries["GENERIC"]["unused_8_uniform"]["p_fixed_raw"]
    )
    leak_struct = float(arm_summaries["STRUCTURED"]["valid_24_uniform"]["leakage_to_unused_8"])
    spam_v24 = float(arm_summaries["PREP_RO_CTRL"]["valid_24_uniform"]["p_correct_raw"])
    spam_u8 = float(arm_summaries["PREP_RO_CTRL"]["unused_8_uniform"]["p_fixed_raw"])

    h1_pass = (
        delta_v24 > 0.25
        and bool(np.all(blk_diffs_v24 > 0.25))
        and p_blk_one < 0.01
        and p_wilcox_two < 1e-4
    )
    h2_pass = delta_u8 > 0.20 and leak_struct < 0.15
    h3_pass = spam_v24 >= 0.90 and spam_u8 >= 0.90

    return {
        "schema": "osiris.bt2t_fez.results/1",
        "protocol_id": PROTOCOL_ID,
        "execution_tier": "HARDWARE_ATTESTED",
        "backend": BACKEND_NAME,
        "provider_job_id": job_meta["job_id"],
        "job_metadata": job_meta,
        "quota_accounting": quota_snapshot,
        "raw_counts_gz_sha256": raw_gz_sha256,
        "qpy_sha256": manifest["qpy_sha256"],
        "encoding_version": "bt2t-enc-1",
        "encoding_sha256": manifest["noiseless_and_compilation_verification"]["encoding_sha256"],
        "fez_t5_layout": manifest["fez_t5_layout"],
        "equal_shot_rule": {
            "enforced": True,
            "n_blocks": N_BLOCKS,
            "shots_per_pub": SHOTS_PER_PUB,
            "shots_per_arm_input_cell": N_BLOCKS * SHOTS_PER_PUB,
            "total_shots": manifest["total_shots"],
            "postselection_loss": 0.0,
        },
        "unit_of_inference": manifest["unit_of_inference"],
        "readout_assignment_32x32_cond": round(ro_cond, 4),
        "arms": arm_summaries,
        "contrasts": {
            "structured_minus_generic_valid24_raw": round(delta_v24, 6),
            "block_deltas_valid24": [round(float(d), 6) for d in blk_diffs_v24],
            "paired_block_t_stat_df2": round(float(t_blk), 4),
            "paired_block_p_one_sided": round(p_blk_one, 8),
            "elementwise_wins_out_of_24": n_elem_wins,
            "elementwise_min_delta_24": round(float(np.min(elem_diffs_24)), 6),
            "elementwise_max_delta_24": round(float(np.max(elem_diffs_24)), 6),
            "wilcoxon_24_elements_p_two_sided": p_wilcox_two,
            "structured_minus_generic_unused8_raw": round(delta_u8, 6),
            "uniform_random_32state_chance": 1.0 / 32.0,
            "uniform_random_unused_leakage_baseline": 8.0 / 32.0,
        },
        "hypotheses": {
            "H1_structured_vs_generic_valid24": {
                "delta_p_valid24": round(delta_v24, 6),
                "all_3_blocks_gt_0_25": bool(np.all(blk_diffs_v24 > 0.25)),
                "paired_block_p_one_sided": round(p_blk_one, 8),
                "wilcoxon_p_two_sided": p_wilcox_two,
                "verdict": "SUPPORTED" if h1_pass else "REFUTED",
            },
            "H2_unused_codeword_preservation_and_confinement": {
                "delta_p_unused8": round(delta_u8, 6),
                "structured_valid24_leakage_to_unused": round(leak_struct, 6),
                "verdict": "SUPPORTED" if h2_pass else "REFUTED",
            },
            "H3_spam_control_integrity": {
                "prep_ro_ctrl_valid24": round(spam_v24, 6),
                "prep_ro_ctrl_unused8": round(spam_u8, 6),
                "verdict": "SUPPORTED" if h3_pass else "REFUTED",
            },
        },
        "per_input_tables": per_input_tables,
    }


def submit_frozen_and_harvest() -> Dict[str, Any]:
    man_path = EXPERIMENT_DIR / "bt2t_fez_staging_manifest.json"
    qpy_path = EXPERIMENT_DIR / "bt2t_fez_staged_circuits.qpy"
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    actual_sha = hashlib.sha256(qpy_path.read_bytes()).hexdigest()
    if actual_sha != manifest["qpy_sha256"]:
        raise RuntimeError(f"QPY SHA-256 mismatch: {actual_sha} != {manifest['qpy_sha256']}")

    with qpy_path.open("rb") as f:
        isa_circuits = qpy.load(f)

    svc, _ = connect_ibm_runtime_service()
    usage_before = svc.usage()
    rem_before = float(usage_before.get("usage_remaining_seconds", 0))
    cons_before = float(usage_before.get("usage_consumed_seconds", 0))
    if rem_before < 14.0:
        raise RuntimeError(f"Insufficient QPU quota remaining ({rem_before}s < 14.0s safety gate)")

    fez = svc.backend(BACKEND_NAME)
    pending = int(fez.status().pending_jobs)
    sampler = SamplerV2(mode=fez)
    sampler.options.default_shots = SHOTS_PER_PUB

    print(
        f"[SUBMIT] Submitting {len(isa_circuits)} ISA PUBs x {SHOTS_PER_PUB} shots = "
        f"{len(isa_circuits) * SHOTS_PER_PUB} shots to {BACKEND_NAME} "
        f"(pending={pending}, quota_remaining={rem_before}s)..."
    )
    job = sampler.run(isa_circuits)
    job_id = job.job_id()
    print(f"[SUBMITTED] provider_job_id = {job_id}")
    (EXPERIMENT_DIR / "bt2t_fez_submitted_job.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "submitted_utc": _utc_now(),
                "backend": BACKEND_NAME,
                "pending_jobs_at_submission": pending,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    t0 = time.time()
    while True:
        st = str(job.status())
        print(f"  [{time.time() - t0:5.1f}s] job {job_id} status = {st}")
        if st in ("DONE", "ERROR", "CANCELLED"):
            break
        time.sleep(3.0)

    if str(job.status()) != "DONE":
        raise RuntimeError(f"Job {job_id} ended with status {job.status()}")

    res = job.result()
    raw_counts: Dict[str, Dict[str, int]] = {}
    for idx in range(len(isa_circuits)):
        raw_counts[str(idx)] = _extract_counts_5bit(res[idx])

    raw_gz_path = EXPERIMENT_DIR / "physical_bt2t_fez_raw_counts.json.gz"
    raw_bytes = json.dumps(raw_counts, sort_keys=True).encode("utf-8")
    with gzip.open(raw_gz_path, "wb") as gz:
        gz.write(raw_bytes)
    raw_gz_sha = hashlib.sha256(raw_gz_path.read_bytes()).hexdigest()

    job_meta = redact_ibm_job_metadata(job)
    (EXPERIMENT_DIR / "bt2t_fez_submitted_job.json").write_text(
        json.dumps(job_meta, indent=2, sort_keys=True), encoding="utf-8"
    )

    usage_after = svc.usage()
    quota_snapshot = {
        "usage_limit_seconds": usage_after.get("usage_limit_seconds"),
        "usage_consumed_seconds_before": cons_before,
        "usage_remaining_seconds_before": rem_before,
        "usage_consumed_seconds_after": usage_after.get("usage_consumed_seconds"),
        "usage_remaining_seconds_after": usage_after.get("usage_remaining_seconds"),
    }

    analysis = analyze_harvested(manifest, raw_counts, job_meta, raw_gz_sha, quota_snapshot)
    out_path = EXPERIMENT_DIR / "physical_bt2t_fez_results.json"
    out_path.write_text(json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[HARVESTED] {PROTOCOL_ID} job={job_id} "
        f"QPU_s={job_meta.get('usage', {}).get('qpu_charge_time_seconds')} "
        f"rem_after={quota_snapshot['usage_remaining_seconds_after']}s"
    )
    return analysis


def analyze_saved() -> Dict[str, Any]:
    manifest = json.loads((EXPERIMENT_DIR / "bt2t_fez_staging_manifest.json").read_text(encoding="utf-8"))
    job_meta = json.loads((EXPERIMENT_DIR / "bt2t_fez_submitted_job.json").read_text(encoding="utf-8"))
    raw_gz_path = EXPERIMENT_DIR / "physical_bt2t_fez_raw_counts.json.gz"
    raw_gz_sha = hashlib.sha256(raw_gz_path.read_bytes()).hexdigest()
    with gzip.open(raw_gz_path, "rb") as gz:
        raw_counts = json.loads(gz.read().decode("utf-8"))
    prev_res_path = EXPERIMENT_DIR / "physical_bt2t_fez_results.json"
    quota_snap = {}
    if prev_res_path.exists():
        quota_snap = json.loads(prev_res_path.read_text(encoding="utf-8")).get("quota_accounting", {})
    analysis = analyze_harvested(manifest, raw_counts, job_meta, raw_gz_sha, quota_snap)
    prev_res_path.write_text(json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")
    return analysis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-only", action="store_true")
    ap.add_argument("--submit-frozen-and-harvest", action="store_true")
    ap.add_argument("--analyze-saved", action="store_true")
    args = ap.parse_args()
    if args.submit_frozen_and_harvest:
        submit_frozen_and_harvest()
    elif args.analyze_saved:
        analyze_saved()
    else:
        stage_experiment()


if __name__ == "__main__":
    main()
