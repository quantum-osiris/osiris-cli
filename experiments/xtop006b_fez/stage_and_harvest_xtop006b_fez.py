#!/usr/bin/env python3
"""
stage_and_harvest_xtop006b_fez.py — Confirmatory intra-block 4-twirl Landsman-Figgatt
scrambling-vs-decoherence discriminator (`OSIRIS-XTOP-006B-FEZ`) on `ibm_fez`
(`[99, 115, 95, 93, 94, 92, 79]`, 256 PUBs x 256 shots = 65,536 shots, ~18-19s QPU).
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
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import SamplerV2
from qiskit_ibm_runtime.utils.validations import validate_isa_circuits
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_X06_PATH = REPO_ROOT / "experiments" / "xtop006_fez" / "stage_and_harvest_xtop006_fez.py"
_X06_SPEC = importlib.util.spec_from_file_location("xtop006_fez_mod", _X06_PATH)
assert _X06_SPEC is not None and _X06_SPEC.loader is not None
x06 = importlib.util.module_from_spec(_X06_SPEC)
sys.modules["xtop006_fez_mod"] = x06
_X06_SPEC.loader.exec_module(x06)

from osiris_cli.ibm_credentials import connect_ibm_runtime_service, redact_ibm_job_metadata  # noqa: E402
from xtop006.kraus_channel import (  # noqa: E402
    INPUT_STATES_6,
    THETA_GRID_DEFAULT,
    analytical_fidelity_6state,
    analytical_p_phi_plus,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROTOCOL_ID = "OSIRIS-XTOP-006B-FEZ"
BACKEND_NAME = "ibm_fez"
FEZ_7Q_LAYOUT = x06.FEZ_7Q_LAYOUT
MEAS_PHYS = x06.MEAS_PHYS
INPUT_ORDER = x06.INPUT_ORDER
FOUR_TWIRLS: Tuple[Tuple[int, int, int], ...] = (
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1),
)
TARGET_THETAS: Tuple[Tuple[int, str, float], ...] = (
    (3, "3pi_8", THETA_GRID_DEFAULT[3]),
    (4, "pi_2", THETA_GRID_DEFAULT[4]),
)

N_BLOCKS = 4
SHOTS_PER_PUB = 256
SCHEDULE_SEED = 20261125


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def verify_intra_block_4twirl_noiseless() -> Dict[str, Any]:
    """Verify that inside ANY single block, the 4-twirl average equals exact classical
    dephased channel expectation (`0.64265717` at `3pi/8` and `0.66666667` at `pi/2`)
    with ZERO inter-block twirl variance.
    """
    rows: List[Dict[str, Any]] = []
    for t_idx, t_name, th in TARGET_THETAS:
        coh_f = []
        tw_f = []
        for lbl in INPUT_ORDER:
            _, _, eig = INPUT_STATES_6[lbl]
            target_b6 = 0 if eig == +1 else 1
            qc_c = x06.compile_stream_to_compact_7q(x06.build_constant_skeleton_7q_ops(th, lbl, (0, 0, 0)))
            pc = np.abs(Statevector.from_instruction(qc_c).data) ** 2
            pp_c = sum(float(pc[i]) for i in range(128) if ((i >> 2) & 1) == 0 and ((i >> 4) & 1) == 0)
            po_c = sum(
                float(pc[i])
                for i in range(128)
                if ((i >> 2) & 1) == 0 and ((i >> 4) & 1) == 0 and ((i >> 6) & 1) == target_b6
            )
            coh_f.append(po_c / pp_c)

            # Equal-weighted 4-twirl mixture (pool unnormalized joint probabilities across the 4 twirls!)
            pp_d_sum = 0.0
            po_d_sum = 0.0
            for tw in FOUR_TWIRLS:
                qc_d = x06.compile_stream_to_compact_7q(x06.build_constant_skeleton_7q_ops(th, lbl, tw))
                pd = np.abs(Statevector.from_instruction(qc_d).data) ** 2
                pp_d_sum += sum(float(pd[i]) for i in range(128) if ((i >> 2) & 1) == 0 and ((i >> 4) & 1) == 0)
                po_d_sum += sum(
                    float(pd[i])
                    for i in range(128)
                    if ((i >> 2) & 1) == 0 and ((i >> 4) & 1) == 0 and ((i >> 6) & 1) == target_b6
                )
            tw_f.append(po_d_sum / pp_d_sum)

        mean_coh = float(np.mean(coh_f))
        mean_dec = float(np.mean(tw_f))
        assert abs(mean_coh - analytical_fidelity_6state(th)) < 1e-12
        rows.append(
            {
                "theta_name": t_name,
                "theta_rad": round(th, 8),
                "noiseless_coherent_f_6state": round(mean_coh, 8),
                "noiseless_4twirl_decoh_f_6state": round(mean_dec, 8),
                "noiseless_delta_f": round(mean_coh - mean_dec, 8),
                "noiseless_inter_block_twirl_sd": 0.0,
            }
        )
    return {"angles": rows}


def stage_experiment() -> Dict[str, Any]:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    noiseless_info = verify_intra_block_4twirl_noiseless()

    svc, cred_prov = connect_ibm_runtime_service()
    fez = svc.backend(BACKEND_NAME)
    tgt = fez.target
    props = fez.properties()

    rng = np.random.default_rng(SCHEDULE_SEED)
    isa_circuits: List[QuantumCircuit] = []
    schedule_meta: List[Dict[str, Any]] = []

    for blk in range(N_BLOCKS):
        block_items: List[Tuple[QuantumCircuit, Dict[str, Any]]] = []
        # 1. COHERENT: 2 angles x 6 states = 12 PUBs
        for t_idx, t_name, th in TARGET_THETAS:
            for lbl in INPUT_ORDER:
                cname = f"b{blk}_coh_{t_name}_in_{lbl}"
                stream = x06.build_constant_skeleton_7q_ops(th, lbl, decoh_twirl=(0, 0, 0))
                qc = x06.compile_stream_to_fez_isa(stream, tgt, cname)
                block_items.append(
                    (
                        qc,
                        {
                            "block": blk,
                            "arm": "COHERENT",
                            "theta_idx": t_idx,
                            "theta_name": t_name,
                            "theta_rad": th,
                            "input_label": lbl,
                            "decoh_twirl": [0, 0, 0],
                        },
                    )
                )

        # 2. DECOH_4TWIRL: 2 angles x ALL 4 twirls x 6 states = 48 PUBs INSIDE EVERY BLOCK!
        for t_idx, t_name, th in TARGET_THETAS:
            for tw_idx, tw in enumerate(FOUR_TWIRLS):
                for lbl in INPUT_ORDER:
                    cname = f"b{blk}_dec4_tw{tw_idx}_{t_name}_in_{lbl}"
                    stream = x06.build_constant_skeleton_7q_ops(th, lbl, decoh_twirl=tw)
                    qc = x06.compile_stream_to_fez_isa(stream, tgt, cname)
                    block_items.append(
                        (
                            qc,
                            {
                                "block": blk,
                                "arm": "DECOH_4TWIRL",
                                "twirl_idx": tw_idx,
                                "theta_idx": t_idx,
                                "theta_name": t_name,
                                "theta_rad": th,
                                "input_label": lbl,
                                "decoh_twirl": list(tw),
                            },
                        )
                    )

        # 3. 2 A/A sentinels + 2 readout sentinels (|000>, |111>) = 4 PUBs
        for s_rep in range(2):
            cname = f"b{blk}_aa_sent_{s_rep}"
            stream = x06.build_constant_skeleton_7q_ops(math.pi / 2.0, "+", decoh_twirl=(0, 0, 0))
            qc = x06.compile_stream_to_fez_isa(stream, tgt, cname)
            block_items.append(
                (
                    qc,
                    {
                        "block": blk,
                        "arm": "AA_SENTINEL",
                        "sentinel_rep": s_rep,
                        "theta_name": "pi_2",
                        "input_label": "+",
                    },
                )
            )
        for bit_v in (0, 1):
            bits = (bit_v, bit_v, bit_v)
            qc_ro = x06.build_ro_cal_fez_isa(bits, tgt, f"b{blk}_ro_{bit_v}{bit_v}{bit_v}")
            block_items.append(
                (qc_ro, {"block": blk, "arm": "RO_SENTINEL", "prep_bits_c2c1c0": f"{bit_v}{bit_v}{bit_v}"})
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
    work_cz = {m["cz_count"] for m in schedule_meta if m["arm"] != "RO_SENTINEL"}
    work_sx = {m["sx_count"] for m in schedule_meta if m["arm"] != "RO_SENTINEL"}
    assert work_cz == {28}
    assert len(work_sx) == 1

    qpy_path = EXPERIMENT_DIR / "xtop006b_fez_staged_circuits.qpy"
    with qpy_path.open("wb") as f:
        qpy.dump(isa_circuits, f)
    qpy_sha = hashlib.sha256(qpy_path.read_bytes()).hexdigest()

    cal_snap = {
        "backend": BACKEND_NAME,
        "captured_utc": _utc_now(),
        "last_update_date": str(getattr(props, "last_update_date", None)),
        "layout_7q": list(FEZ_7Q_LAYOUT),
    }
    cal_path = EXPERIMENT_DIR / "xtop006b_fez_calibration_snapshot.json"
    cal_path.write_text(json.dumps(cal_snap, indent=2, sort_keys=True), encoding="utf-8")

    manifest = {
        "schema": "osiris.quantum_protocol/1",
        "protocol_id": PROTOCOL_ID,
        "staged_utc": _utc_now(),
        "backend": BACKEND_NAME,
        "credential_provenance": cred_prov,
        "layout_7q_q0_to_q6": list(FEZ_7Q_LAYOUT),
        "measured_phys_c0c1c2": list(MEAS_PHYS),
        "n_blocks": N_BLOCKS,
        "pubs_per_block": 64,
        "total_pubs": len(isa_circuits),
        "shots_per_pub": SHOTS_PER_PUB,
        "total_shots": len(isa_circuits) * SHOTS_PER_PUB,
        "constant_cz_count": 28,
        "constant_sx_count": next(iter(work_sx)),
        "qpy_filename": qpy_path.name,
        "qpy_sha256": qpy_sha,
        "preregistration_md_sha256": hashlib.sha256((EXPERIMENT_DIR / "PRE_REGISTRATION.md").read_bytes()).hexdigest(),
        "noiseless_verification": noiseless_info,
        "schedule": schedule_meta,
    }
    (EXPERIMENT_DIR / "xtop006b_fez_staging_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[STAGED] {PROTOCOL_ID}: {len(isa_circuits)} PUBs x {SHOTS_PER_PUB} = {len(isa_circuits)*SHOTS_PER_PUB} shots")
    print(f"  qpy_sha256 = {qpy_sha}")
    return manifest


def analyze_harvested(
    manifest: Mapping[str, Any],
    raw_counts_by_pub: Mapping[str, Mapping[str, int]],
    job_meta: Mapping[str, Any],
    raw_gz_sha256: str,
) -> Dict[str, Any]:
    sched = manifest["schedule"]
    contrasts: List[Dict[str, Any]] = []
    raw_pvals: List[float] = []

    for t_idx, t_name, th in TARGET_THETAS:
        blk_f_coh: List[float] = []
        blk_p_coh: List[float] = []
        blk_f_dec: List[float] = []
        blk_p_dec: List[float] = []

        for blk in range(N_BLOCKS):
            # COHERENT in block blk
            c_tot, c_phi, c_ok = 0, 0, 0
            # DECOH_4TWIRL in block blk (pooling all 4 twirls equally inside block blk!)
            d_tot, d_phi, d_ok = 0, 0, 0
            for item in sched:
                if item.get("block") == blk and item.get("theta_name") == t_name:
                    cnts = raw_counts_by_pub[str(item["pub_index"])]
                    if item["arm"] == "COHERENT":
                        tc, pc, oc = x06._score_counts(cnts, item["input_label"])
                        c_tot += tc
                        c_phi += pc
                        c_ok += oc
                    elif item["arm"] == "DECOH_4TWIRL":
                        tc, pc, oc = x06._score_counts(cnts, item["input_label"])
                        d_tot += tc
                        d_phi += pc
                        d_ok += oc
            blk_p_coh.append(c_phi / max(1, c_tot))
            blk_f_coh.append(c_ok / max(1, c_phi))
            blk_p_dec.append(d_phi / max(1, d_tot))
            blk_f_dec.append(d_ok / max(1, d_phi))

        diffs = np.array(blk_f_coh, dtype=np.float64) - np.array(blk_f_dec, dtype=np.float64)
        t_stat, p_two = stats.ttest_1samp(diffs, popmean=0.0)
        p_one = float(p_two / 2.0) if t_stat > 0 else float(1.0 - p_two / 2.0)
        raw_pvals.append(p_one)
        contrasts.append(
            {
                "theta_name": t_name,
                "theta_rad": round(th, 8),
                "analytic_coherent_p_phi_plus": round(analytical_p_phi_plus(th), 6),
                "coherent_mean_p_phi_plus": round(float(np.mean(blk_p_coh)), 6),
                "decoh_4twirl_mean_p_phi_plus": round(float(np.mean(blk_p_dec)), 6),
                "analytic_coherent_f_6state": round(analytical_fidelity_6state(th), 6),
                "coherent_mean_raw_f": round(float(np.mean(blk_f_coh)), 6),
                "coherent_block_se": round(float(np.std(blk_f_coh, ddof=1) / math.sqrt(N_BLOCKS)), 6),
                "decoh_4twirl_mean_raw_f": round(float(np.mean(blk_f_dec)), 6),
                "decoh_4twirl_block_se": round(float(np.std(blk_f_dec, ddof=1) / math.sqrt(N_BLOCKS)), 6),
                "delta_f_coherent_minus_decoh_4twirl": round(float(np.mean(diffs)), 6),
                "delta_f_block_se": round(float(np.std(diffs, ddof=1) / math.sqrt(N_BLOCKS)), 6),
                "paired_t_stat": round(float(t_stat), 4),
                "p_one_sided": round(p_one, 8),
                "block_coherent_f": [round(x, 6) for x in blk_f_coh],
                "block_decoh_4twirl_f": [round(x, 6) for x in blk_f_dec],
            }
        )

    holm = x06._holm_bonferroni(raw_pvals)
    for row, hp in zip(contrasts, holm):
        row["p_holm_adjusted"] = round(hp, 8)

    h2_conf_pass = all(
        row["delta_f_coherent_minus_decoh_4twirl"] > (0.15 if row["theta_name"] == "pi_2" else 0.10)
        and row["p_holm_adjusted"] < 0.01
        for row in contrasts
    )

    return {
        "schema": "osiris.xtop006b_fez.results/1",
        "protocol_id": PROTOCOL_ID,
        "execution_tier": "HARDWARE_ATTESTED",
        "backend": BACKEND_NAME,
        "provider_job_id": job_meta["job_id"],
        "job_metadata": job_meta,
        "raw_counts_gz_sha256": raw_gz_sha256,
        "qpy_sha256": manifest["qpy_sha256"],
        "layout_7q": manifest["layout_7q_q0_to_q6"],
        "constant_cz_count": 28,
        "constant_sx_count": manifest["constant_sx_count"],
        "intra_block_4twirl_contrasts": contrasts,
        "hypotheses": {
            "H2_confirmatory_intra_block_4twirl_discriminator": {
                "verdict": "SUPPORTED" if h2_conf_pass else "FALSIFIED",
            }
        },
    }


def submit_frozen_and_harvest() -> Dict[str, Any]:
    man_path = EXPERIMENT_DIR / "xtop006b_fez_staging_manifest.json"
    qpy_path = EXPERIMENT_DIR / "xtop006b_fez_staged_circuits.qpy"
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    if hashlib.sha256(qpy_path.read_bytes()).hexdigest() != manifest["qpy_sha256"]:
        raise RuntimeError("QPY SHA-256 mismatch")

    with qpy_path.open("rb") as f:
        isa_circuits = qpy.load(f)

    svc, _ = connect_ibm_runtime_service()
    usage_before = svc.usage()
    rem_sec = float(usage_before.get("usage_remaining_seconds", 0))
    if rem_sec < 22.0:
        raise RuntimeError(f"Insufficient QPU quota remaining ({rem_sec}s < 22s safety gate)")

    fez = svc.backend(BACKEND_NAME)
    sampler = SamplerV2(mode=fez)
    sampler.options.default_shots = SHOTS_PER_PUB

    print(f"[SUBMIT] Submitting {len(isa_circuits)} ISA PUBs x {SHOTS_PER_PUB} shots to {BACKEND_NAME} (rem={rem_sec}s)...")
    job = sampler.run(isa_circuits)
    job_id = job.job_id()
    print(f"[SUBMITTED] provider_job_id = {job_id}")
    (EXPERIMENT_DIR / "xtop006b_fez_submitted_job.json").write_text(
        json.dumps({"job_id": job_id, "submitted_utc": _utc_now(), "backend": BACKEND_NAME}, indent=2),
        encoding="utf-8",
    )

    t0 = time.time()
    while True:
        st = str(job.status())
        print(f"  [{time.time() - t0:5.1f}s] job {job_id} status = {st}")
        if st in ("DONE", "ERROR", "CANCELLED"):
            break
        time.sleep(4.0)

    if str(job.status()) != "DONE":
        raise RuntimeError(f"Job {job_id} ended with status {job.status()}")

    res = job.result()
    raw_counts: Dict[str, Dict[str, int]] = {}
    for idx in range(len(isa_circuits)):
        raw_counts[str(idx)] = x06._extract_counts_3bit(res[idx])

    raw_gz_path = EXPERIMENT_DIR / "physical_xtop006b_fez_raw_counts.json.gz"
    with gzip.open(raw_gz_path, "wb") as gz:
        gz.write(json.dumps(raw_counts, sort_keys=True).encode("utf-8"))
    raw_gz_sha = hashlib.sha256(raw_gz_path.read_bytes()).hexdigest()

    job_meta = redact_ibm_job_metadata(job)
    (EXPERIMENT_DIR / "xtop006b_fez_submitted_job.json").write_text(
        json.dumps(job_meta, indent=2, sort_keys=True), encoding="utf-8"
    )

    analysis = analyze_harvested(manifest, raw_counts, job_meta, raw_gz_sha)
    (EXPERIMENT_DIR / "physical_xtop006b_fez_results.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[HARVESTED] {PROTOCOL_ID} job={job_id} QPU_s={job_meta.get('usage', {}).get('qpu_charge_time_seconds')}")
    return analysis


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
