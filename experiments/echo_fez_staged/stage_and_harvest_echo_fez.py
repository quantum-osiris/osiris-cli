#!/usr/bin/env python3
"""
stage_and_harvest_echo_fez.py — Write-ahead `--stage-only` (0s QPU) & harvest runner for
`OSIRIS-ECHO-FEZ` (`Card B` 10-placement `T5` echo-ranking pilot on `ibm_fez`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, qpy
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osiris_cli.ibm_credentials import connect_ibm_runtime_service, redact_ibm_job_metadata  # noqa: E402


def _load_echo_gate() -> Any:
    mod_name = "_echo_placement_gate_mod"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    p = REPO_ROOT / "experiments" / "echo_placement" / "echo_placement_gate.py"
    spec = importlib.util.spec_from_file_location(mod_name, p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROTOCOL_ID = "OSIRIS-ECHO-FEZ"
BACKEND_NAME = "ibm_fez"
N_BLOCKS = 3
SHOTS_PER_PUB = 256
SCHEDULE_SEED = 20261124
BT2T_STRATIFIED_INPUTS: Tuple[int, int, int] = (0, 2, 8)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def build_per_placement_templates() -> Dict[str, Dict[str, Any]]:
    """Build all 8 5-qubit T5 template circuits and verify 100% noiseless target probabilities."""
    epg = _load_echo_gate()
    probes = epg.build_echo_probe_and_controls()
    heldout = epg.build_heldout_reference_circuits()
    assert probes["audit"]["non_cancelled_cz_verified"] is True

    templates: Dict[str, Dict[str, Any]] = {
        "ECHO_PROBE_16CZ": {
            "core_5q": probes["echo_circuit"],
            "init_x_t5": [],
            "final_idx": [0, 1, 2, 3, 4],
            "expected_y": 0,
            "cz_count": int(probes["echo_circuit"].count_ops().get("cz", 0)),
            "role": "probe",
        },
        "IDENTITY_CTRL_0CZ": {
            "core_5q": QuantumCircuit(5, name="id_0cz"),
            "init_x_t5": [],
            "final_idx": [0, 1, 2, 3, 4],
            "expected_y": 0,
            "cz_count": 0,
            "role": "control_identity",
        },
        "OVERROT_CTRL_016": {
            "core_5q": probes["perturbed_circuits"][0.16],
            "init_x_t5": [],
            "final_idx": [0, 1, 2, 3, 4],
            "expected_y": 0,
            "cz_count": int(probes["perturbed_circuits"][0.16].count_ops().get("cz", 0)),
            "role": "control_overrot",
        },
        "CLIFFORD_30CZ": {
            "core_5q": heldout["random_clifford_30cz"]["circuit"],
            "init_x_t5": [],
            "final_idx": [0, 1, 2, 3, 4],
            "expected_y": int(heldout["random_clifford_30cz"]["target_phys_state"]),
            "cz_count": int(heldout["random_clifford_30cz"]["cz_count"]),
            "role": "heldout",
        },
        "MIRROR_24CZ": {
            "core_5q": heldout["non_clifford_mirror_24cz"]["circuit"],
            "init_x_t5": [],
            "final_idx": [0, 1, 2, 3, 4],
            "expected_y": int(heldout["non_clifford_mirror_24cz"]["target_phys_state"]),
            "cz_count": int(heldout["non_clifford_mirror_24cz"]["cz_count"]),
            "role": "heldout",
        },
    }

    bt_spec = heldout["bt2t_72cz"]
    bt_core = QuantumCircuit(5, name="bt2t_core5q")
    for inst in bt_spec["circuit"].data:
        qs = [bt_spec["circuit"].find_bit(q).index for q in inst.qubits]
        bt_core.append(inst.operation, qs)
    for x_in in BT2T_STRATIFIED_INPUTS:
        init_t5 = [bt_spec["init_idx"][lq] for lq in range(5) if (x_in >> lq) & 1]
        templates[f"BT2T_72CZ_x{x_in:02d}"] = {
            "core_5q": bt_core,
            "init_x_t5": init_t5,
            "final_idx": list(bt_spec["final_idx"]),
            "expected_y": int(bt_spec["inversion_perm"][x_in]),
            "cz_count": int(bt_spec["cz_count"]),
            "role": "heldout",
        }
    return templates


def _lift_template_to_isa(
    tmpl: Mapping[str, Any],
    phys_t5: Sequence[int],
    tgt: Any,
    name: str,
) -> QuantumCircuit:
    qc = QuantumCircuit(tgt.num_qubits, 5, name=name)
    for t5_q in tmpl["init_x_t5"]:
        qc.x(phys_t5[int(t5_q)])
    qc.barrier(*phys_t5)
    core: QuantumCircuit = tmpl["core_5q"]
    for inst in core.data:
        op_name = inst.operation.name
        if op_name == "barrier":
            qs = [phys_t5[core.find_bit(q).index] for q in inst.qubits]
            qc.barrier(*qs)
            continue
        qs = [phys_t5[core.find_bit(q).index] for q in inst.qubits]
        if op_name == "cz":
            pa, pb = qs[0], qs[1]
            if (pa, pb) in tgt["cz"]:
                qc.cz(pa, pb)
            elif (pb, pa) in tgt["cz"]:
                qc.cz(pb, pa)
            else:
                raise RuntimeError(f"Missing CZ edge ({pa}, {pb}) on {BACKEND_NAME}")
        else:
            qc.append(inst.operation, qs)
    qc.barrier(*phys_t5)
    final_idx = tmpl["final_idx"]
    for lq in range(5):
        qc.measure(phys_t5[int(final_idx[lq])], lq)
    return qc


def stage_experiment() -> Dict[str, Any]:
    from qiskit_ibm_runtime.utils.validations import validate_isa_circuits

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    templates = build_per_placement_templates()
    stopgo = json.loads(
        (REPO_ROOT / "experiments" / "echo_placement" / "artifacts" / "echo_stopgo_report.json").read_text(
            encoding="utf-8"
        )
    )
    placements = stopgo["design"]["candidate_placements"]

    svc, cred_prov = connect_ibm_runtime_service()
    fez = svc.backend(BACKEND_NAME)
    tgt = fez.target
    props = fez.properties()

    cal_scores: Dict[int, float] = {}
    cal_snap_placements: List[Dict[str, Any]] = []
    for p in placements:
        pid = int(p["placement_id"])
        qs = [int(q) for q in p["physical_qubits"]]
        cz_sum = 0.0
        for a, b in ((0, 1), (1, 2), (2, 3), (2, 4)):
            pa, pb = qs[a], qs[b]
            inst = tgt["cz"].get((pa, pb)) or tgt["cz"].get((pb, pa))
            cz_sum += float(inst.error or 0.0)
        ro_mean = float(np.mean([props.readout_error(q) for q in qs]))
        sx_mean = float(np.mean([props.gate_error("sx", [q]) for q in qs]))
        cal_fidelity_score = float(-(16.0 * (cz_sum / 4.0) + 5.0 * ro_mean + 20.0 * sx_mean))
        cal_scores[pid] = cal_fidelity_score
        cal_snap_placements.append(
            {
                "placement_id": pid,
                "physical_qubits": qs,
                "cz_error_sum": round(cz_sum, 6),
                "readout_error_mean": round(ro_mean, 6),
                "sx_error_mean": round(sx_mean, 6),
                "calibration_rank_score": round(cal_fidelity_score, 6),
                "topology_degree_score": float(p["topology_degree_score"]),
            }
        )

    rng = np.random.default_rng(SCHEDULE_SEED)
    isa_circuits: List[QuantumCircuit] = []
    schedule: List[Dict[str, Any]] = []

    for blk in range(N_BLOCKS):
        blk_items: List[Tuple[QuantumCircuit, Dict[str, Any]]] = []
        for p in placements:
            pid = int(p["placement_id"])
            qs = [int(q) for q in p["physical_qubits"]]
            for w_key, tmpl in templates.items():
                cname = f"b{blk}_p{pid}_{w_key.lower()}"
                qc_isa = _lift_template_to_isa(tmpl, qs, tgt, cname)
                blk_items.append(
                    (
                        qc_isa,
                        {
                            "block": blk,
                            "placement_id": pid,
                            "physical_qubits": qs,
                            "workload": w_key,
                            "role": tmpl["role"],
                            "expected_y": int(tmpl["expected_y"]),
                            "cz_count": int(tmpl["cz_count"]),
                        },
                    )
                )
        for idx in rng.permutation(len(blk_items)):
            qc_i, meta_i = blk_items[int(idx)]
            m = dict(meta_i)
            m["pub_index"] = len(isa_circuits)
            m["circuit_name"] = qc_i.name
            isa_circuits.append(qc_i)
            schedule.append(m)

    validate_isa_circuits(isa_circuits, tgt)

    qpy_path = EXPERIMENT_DIR / "echo_fez_staged_circuits.qpy"
    with qpy_path.open("wb") as f:
        qpy.dump(isa_circuits, f)
    qpy_sha = hashlib.sha256(qpy_path.read_bytes()).hexdigest()

    cal_path = EXPERIMENT_DIR / "echo_fez_calibration_snapshot.json"
    cal_path.write_text(
        json.dumps(
            {
                "backend": BACKEND_NAME,
                "captured_utc": _utc_now(),
                "last_update_date": str(getattr(props, "last_update_date", None)),
                "placements": cal_snap_placements,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    prereg_sha = hashlib.sha256((EXPERIMENT_DIR / "PRE_REGISTRATION.md").read_bytes()).hexdigest()
    manifest = {
        "schema": "osiris.quantum_protocol/1",
        "protocol_id": PROTOCOL_ID,
        "execution_tier": "OFFLINE_STAGED",
        "provider_job_id": None,
        "staged_utc": _utc_now(),
        "backend": BACKEND_NAME,
        "credential_provenance": cred_prov,
        "n_placements": len(placements),
        "n_blocks": N_BLOCKS,
        "pubs_per_block": len(placements) * len(templates),
        "total_pubs": len(isa_circuits),
        "shots_per_pub": SHOTS_PER_PUB,
        "total_shots": len(isa_circuits) * SHOTS_PER_PUB,
        "min_quota_seconds_gate": 25.0,
        "qpy_filename": qpy_path.name,
        "qpy_sha256": qpy_sha,
        "preregistration_md_sha256": prereg_sha,
        "calibration_snapshot_sha256": hashlib.sha256(cal_path.read_bytes()).hexdigest(),
        "placements": cal_snap_placements,
        "schedule": schedule,
    }
    (EXPERIMENT_DIR / "echo_fez_staging_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[STAGED] {PROTOCOL_ID}: {len(isa_circuits)} PUBs x {SHOTS_PER_PUB} = {manifest['total_shots']} shots | qpy={qpy_sha[:16]}...")
    return manifest


def analyze_harvested(
    manifest: Mapping[str, Any],
    raw_counts_by_pub: Mapping[str, Mapping[str, int]],
    job_meta: Mapping[str, Any],
    raw_gz_sha256: str,
) -> Dict[str, Any]:
    cell_ok: Dict[Tuple[int, str], int] = {}
    cell_tot: Dict[Tuple[int, str], int] = {}
    for item in manifest["schedule"]:
        pid = int(item["placement_id"])
        wl = str(item["workload"])
        exp_y = int(item["expected_y"])
        for bstr, cnt in raw_counts_by_pub[str(item["pub_index"])].items():
            obs = int(bstr.replace(" ", "").zfill(5)[-5:], 2)
            cell_tot[(pid, wl)] = cell_tot.get((pid, wl), 0) + int(cnt)
            if obs == exp_y:
                cell_ok[(pid, wl)] = cell_ok.get((pid, wl), 0) + int(cnt)

    def _p(pid: int, wl: str) -> float:
        return float(cell_ok.get((pid, wl), 0)) / max(1.0, float(cell_tot.get((pid, wl), 0)))

    pids = [int(p["placement_id"]) for p in manifest["placements"]]
    echo_scores = np.array([_p(pid, "ECHO_PROBE_16CZ") for pid in pids])
    id_scores = np.array([_p(pid, "IDENTITY_CTRL_0CZ") for pid in pids])
    over_scores = np.array([_p(pid, "OVERROT_CTRL_016") for pid in pids])
    cal_scores = np.array([float(p["calibration_rank_score"]) for p in manifest["placements"]])
    topo_scores = np.array([float(p["topology_degree_score"]) for p in manifest["placements"]])

    heldout_fids = np.array(
        [
            float(
                np.mean(
                    [
                        _p(pid, "CLIFFORD_30CZ"),
                        _p(pid, "MIRROR_24CZ"),
                        np.mean([_p(pid, f"BT2T_72CZ_x{x:02d}") for x in BT2T_STRATIFIED_INPUTS]),
                    ]
                )
            )
            for pid in pids
        ]
    )
    rho_echo = float(stats.spearmanr(echo_scores, heldout_fids).statistic)
    rho_cal = float(stats.spearmanr(cal_scores, heldout_fids).statistic)
    rho_topo = float(stats.spearmanr(topo_scores, heldout_fids).statistic)

    return {
        "schema": "osiris.echo_fez.results/1",
        "protocol_id": PROTOCOL_ID,
        "execution_tier": "HARDWARE_ATTESTED" if job_meta.get("job_id") else "SIMULATED",
        "provider_job_id": job_meta.get("job_id"),
        "raw_counts_gz_sha256": raw_gz_sha256,
        "spearman_echo": round(rho_echo, 6),
        "spearman_calibration": round(rho_cal, 6),
        "spearman_topology": round(rho_topo, 6),
        "delta_spearman_echo_minus_best_comparator": round(rho_echo - max(rho_cal, rho_topo), 6),
        "mean_identity_return": round(float(np.mean(id_scores)), 6),
        "mean_echo_return": round(float(np.mean(echo_scores)), 6),
        "mean_overrot_return": round(float(np.mean(over_scores)), 6),
    }


def submit_frozen_and_harvest() -> Dict[str, Any]:
    from qiskit_ibm_runtime import SamplerV2

    manifest = json.loads((EXPERIMENT_DIR / "echo_fez_staging_manifest.json").read_text(encoding="utf-8"))
    qpy_path = EXPERIMENT_DIR / "echo_fez_staged_circuits.qpy"
    if hashlib.sha256(qpy_path.read_bytes()).hexdigest() != manifest["qpy_sha256"]:
        raise RuntimeError("QPY SHA-256 mismatch")
    with qpy_path.open("rb") as f:
        isa_circuits = qpy.load(f)

    svc, _ = connect_ibm_runtime_service()
    rem = float(svc.usage().get("usage_remaining_seconds", 0))
    if rem < float(manifest["min_quota_seconds_gate"]):
        raise RuntimeError(f"Quota gate blocked submission: {rem}s < {manifest['min_quota_seconds_gate']}s")

    fez = svc.backend(BACKEND_NAME)
    sampler = SamplerV2(mode=fez)
    sampler.options.default_shots = SHOTS_PER_PUB
    job = sampler.run(isa_circuits)
    while str(job.status()) not in ("DONE", "ERROR", "CANCELLED"):
        time.sleep(3.0)
    if str(job.status()) != "DONE":
        raise RuntimeError(f"Job {job.job_id()} failed: {job.status()}")
    res = job.result()
    raw_counts = {str(i): {str(k): int(v) for k, v in res[i].data.c.get_counts().items()} for i in range(len(isa_circuits))}
    raw_gz = EXPERIMENT_DIR / "physical_echo_fez_raw_counts.json.gz"
    with gzip.open(raw_gz, "wb") as gz:
        gz.write(json.dumps(raw_counts, sort_keys=True).encode("utf-8"))
    jmeta = redact_ibm_job_metadata(job)
    out = analyze_harvested(manifest, raw_counts, jmeta, hashlib.sha256(raw_gz.read_bytes()).hexdigest())
    (EXPERIMENT_DIR / "physical_echo_fez_results.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-only", action="store_true")
    ap.add_argument("--submit-frozen-and-harvest", action="store_true")
    args = ap.parse_args()
    if args.submit_frozen_and_harvest:
        submit_frozen_and_harvest()
    else:
        stage_experiment()


if __name__ == "__main__":
    main()
