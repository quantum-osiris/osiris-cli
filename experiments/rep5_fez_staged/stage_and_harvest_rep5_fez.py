#!/usr/bin/env python3
"""Write-ahead `--stage-only` (0s QPU) & harvest runner for `OSIRIS-QEC-REP5-FEZ` (`[[5,1,5]]` vs `[[3,1,3]]`)."""

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
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qpy

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osiris_cli.ibm_credentials import connect_ibm_runtime_service, redact_ibm_job_metadata  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROTOCOL_ID = "OSIRIS-QEC-REP5-FEZ"
BACKEND_NAME = "ibm_fez"
FEZ_REP5_CHAIN: Tuple[int, ...] = (120, 121, 122, 123, 136, 143, 142, 141, 140)
ROUNDS_LIST: Tuple[int, int, int] = (1, 3, 5)
ROUND_DT = 904
N_BLOCKS = 2
SHOTS_PER_PUB = 512
SCHEDULE_SEED = 20261126
MIN_QUOTA_SECONDS_GATE = 25.0
RO_CAL_STATES: Tuple[int, ...] = (0, 31, 10, 21, 4, 27)


def _load_multidist() -> Any:
    name = "_qec_multidist_rep5"
    if name in sys.modules:
        return sys.modules[name]
    p = REPO_ROOT / "experiments/qec_card_c/qec_multidistance_scaling.py"
    spec = importlib.util.spec_from_file_location(name, p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _emit_h_isa(qc: QuantumCircuit, q: int) -> None:
    qc.rz(0.5 * math.pi, q)
    qc.sx(q)
    qc.rz(0.5 * math.pi, q)


def _emit_cx_isa(qc: QuantumCircuit, ctrl: int, targ: int, cz_pairs: Any = None) -> None:
    _emit_h_isa(qc, targ)
    if cz_pairs is None or (ctrl, targ) in cz_pairs:
        qc.cz(ctrl, targ)
    elif (targ, ctrl) in cz_pairs:
        qc.cz(targ, ctrl)
    else:
        raise RuntimeError(f"Missing CZ edge ({ctrl}, {targ})")
    _emit_h_isa(qc, targ)


def build_rep5_or_rep3_circuit(
    arm: str,
    rounds: int,
    logical_bit: int,
    num_qubits: int,
    chain9: Sequence[int],
    cz_pairs: Any = None,
    name: str = "rep5",
) -> QuantumCircuit:
    d_all = [int(chain9[0]), int(chain9[2]), int(chain9[4]), int(chain9[6]), int(chain9[8])]
    a_all = [int(chain9[1]), int(chain9[3]), int(chain9[5]), int(chain9[7])]
    d_sub3 = [d_all[1], d_all[2], d_all[3]]
    a_sub3 = [a_all[1], a_all[2]]
    qr = QuantumRegister(num_qubits, "q")

    if arm == "RO_CAL_5D":
        out_r = ClassicalRegister(5, "out")
        qc = QuantumCircuit(qr, out_r, name=name)
        for k, dq in enumerate(d_all):
            if (logical_bit >> k) & 1:
                qc.x(qr[dq])
        for k, dq in enumerate(d_all):
            qc.measure(qr[dq], out_r[k])
        return qc

    if arm == "ARM1_UNENCODED":
        out_r = ClassicalRegister(1, "out")
        qc = QuantumCircuit(qr, out_r, name=name)
        if logical_bit == 1:
            qc.x(qr[d_all[2]])
        for _ in range(rounds):
            qc.delay(ROUND_DT, qr[d_all[2]], unit="dt")
        qc.measure(qr[d_all[2]], out_r[0])
        return qc

    if arm in ("ARM2_REP3_NOCORR", "ARM2_REP5_NOCORR"):
        d_use = d_sub3 if arm == "ARM2_REP3_NOCORR" else d_all
        out_r = ClassicalRegister(len(d_use), "out")
        qc = QuantumCircuit(qr, out_r, name=name)
        if logical_bit == 1:
            for dq in d_use:
                qc.x(qr[dq])
        for _ in range(rounds):
            for dq in d_use:
                qc.delay(ROUND_DT, qr[dq], unit="dt")
        for k, dq in enumerate(d_use):
            qc.measure(qr[dq], out_r[k])
        return qc

    # ARM4A_REP3_MWPM or ARM4A_REP5_MWPM
    is_d5 = arm == "ARM4A_REP5_MWPM"
    d_use = d_all if is_d5 else d_sub3
    a_use = a_all if is_d5 else a_sub3
    syn_regs = [ClassicalRegister(len(a_use), f"s{r_idx}") for r_idx in range(rounds)]
    out_r = ClassicalRegister(len(d_use), "out")
    qc = QuantumCircuit(qr, *syn_regs, out_r, name=name)
    if logical_bit == 1:
        for dq in d_use:
            qc.x(qr[dq])
    for r_idx in range(rounds):
        for k, aq in enumerate(a_use):
            _emit_cx_isa(qc, d_use[k], aq, cz_pairs)
        for k, aq in enumerate(a_use):
            _emit_cx_isa(qc, d_use[k + 1], aq, cz_pairs)
        for k, aq in enumerate(a_use):
            qc.measure(qr[aq], syn_regs[r_idx][k])
            qc.reset(qr[aq])
    for k, dq in enumerate(d_use):
        qc.measure(qr[dq], out_r[k])
    return qc


def stage_experiment() -> Dict[str, Any]:
    from qiskit_ibm_runtime.utils.validations import validate_isa_circuits

    svc, cred_prov = connect_ibm_runtime_service()
    fez = svc.backend(BACKEND_NAME)
    tgt = fez.target
    num_q = tgt.num_qubits
    cz_pairs = set(tgt.build_coupling_map().get_edges())

    rng = np.random.default_rng(SCHEDULE_SEED)
    isa_circuits: List[QuantumCircuit] = []
    schedule: List[Dict[str, Any]] = []
    arms = (
        "ARM1_UNENCODED",
        "ARM2_REP3_NOCORR",
        "ARM2_REP5_NOCORR",
        "ARM4A_REP3_MWPM",
        "ARM4A_REP5_MWPM",
    )

    for blk in range(N_BLOCKS):
        specs: List[Dict[str, Any]] = []
        for st_val in RO_CAL_STATES:
            specs.append({"arm": "RO_CAL_5D", "rounds": 0, "logical_bit": int(st_val)})
        for r in ROUNDS_LIST:
            for lb in (0, 1):
                for arm in arms:
                    specs.append({"arm": arm, "rounds": int(r), "logical_bit": int(lb)})
        order = rng.permutation(len(specs))
        for idx_in_blk in order:
            sp = specs[int(idx_in_blk)]
            pub_idx = len(isa_circuits)
            cname = f"b{blk}_{sp['arm'].lower()}_r{sp['rounds']}_l{sp['logical_bit']}"
            qc = build_rep5_or_rep3_circuit(
                sp["arm"],
                sp["rounds"],
                sp["logical_bit"],
                num_q,
                FEZ_REP5_CHAIN,
                cz_pairs=cz_pairs,
                name=cname,
            )
            isa_circuits.append(qc)
            schedule.append(
                {
                    "pub_index": pub_idx,
                    "block": blk,
                    "arm": sp["arm"],
                    "rounds": sp["rounds"],
                    "logical_bit": sp["logical_bit"],
                    "circuit_name": cname,
                }
            )

    validate_isa_circuits(isa_circuits, tgt)

    qpy_path = EXPERIMENT_DIR / "rep5_fez_staged_circuits.qpy"
    with qpy_path.open("wb") as f:
        qpy.dump(isa_circuits, f)
    qpy_sha = hashlib.sha256(qpy_path.read_bytes()).hexdigest()

    cal_snap = {
        "backend": BACKEND_NAME,
        "captured_utc": _utc_now(),
        "chain_9q": list(FEZ_REP5_CHAIN),
        "credential_provenance": cred_prov,
    }
    cal_path = EXPERIMENT_DIR / "rep5_fez_calibration_snapshot.json"
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
        "chain_9q": list(FEZ_REP5_CHAIN),
        "round_dt": ROUND_DT,
        "n_blocks": N_BLOCKS,
        "pubs_per_block": len(isa_circuits) // N_BLOCKS,
        "total_pubs": len(isa_circuits),
        "shots_per_pub": SHOTS_PER_PUB,
        "total_shots": len(isa_circuits) * SHOTS_PER_PUB,
        "min_quota_seconds_gate": MIN_QUOTA_SECONDS_GATE,
        "qpy_filename": qpy_path.name,
        "qpy_sha256": qpy_sha,
        "preregistration_md_sha256": prereg_sha,
        "calibration_snapshot_sha256": cal_sha,
        "schedule": schedule,
    }
    (EXPERIMENT_DIR / "rep5_fez_staging_manifest.json").write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(
        f"[STAGED] {PROTOCOL_ID}: {len(isa_circuits)} PUBs x {SHOTS_PER_PUB} = {manifest['total_shots']} shots | qpy={qpy_sha[:16]}..."
    )
    return manifest


def _extract_composite_bitstrings(pub_result: Any, reg_names_rev: Sequence[str]) -> Dict[str, int]:
    """Join multi-register BitArrays per shot into 'out s{R-1} ... s0' count dictionary."""
    data_bin = pub_result.data
    bitstrings_by_reg = [getattr(data_bin, rn).get_bitstrings() for rn in reg_names_rev]
    n_shots = len(bitstrings_by_reg[0])
    counts: Dict[str, int] = {}
    for s_idx in range(n_shots):
        key = " ".join(col[s_idx] for col in bitstrings_by_reg)
        counts[key] = counts.get(key, 0) + 1
    return counts


def analyze_harvested(
    manifest: Mapping[str, Any],
    raw_counts_by_pub: Mapping[str, Mapping[str, int]],
    job_meta: Mapping[str, Any],
    raw_gz_sha256: str,
) -> Dict[str, Any]:
    md = _load_multidist()
    fez_prof = md.gate.build_published_target_profiles()["FEZ_PUBLISHED"]
    dec3 = {
        r: md.PolynomialMWPMDecoder(md.build_repetition_dem(3, r, fez_prof, "Z"))
        for r in ROUNDS_LIST
    }
    dec5 = {
        r: md.PolynomialMWPMDecoder(md.build_repetition_dem(5, r, fez_prof, "Z"))
        for r in ROUNDS_LIST
    }

    agg: Dict[Tuple[int, str], Dict[str, float]] = {}
    blk_agg: Dict[Tuple[int, int, str], Dict[str, float]] = {}
    ro_cal_ok = 0.0
    ro_cal_tot = 0.0
    ro_cal_per_qubit_err = np.zeros(5, dtype=float)
    ro_cal_per_qubit_tot = np.zeros(5, dtype=float)
    streak_stats: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}

    for item in manifest["schedule"]:
        arm = str(item["arm"])
        blk = int(item["block"])
        r = int(item["rounds"])
        lb = int(item["logical_bit"])
        cnts = raw_counts_by_pub[str(item["pub_index"])]

        if arm == "RO_CAL_5D":
            for bs, c in cnts.items():
                out_s = str(bs).split()[0].zfill(5)
                val = int(out_s, 2)
                ro_cal_tot += float(c)
                if val == lb:
                    ro_cal_ok += float(c)
                for k in range(5):
                    exp_b = (lb >> k) & 1
                    got_b = (val >> k) & 1
                    ro_cal_per_qubit_tot[k] += float(c)
                    if got_b != exp_b:
                        ro_cal_per_qubit_err[k] += float(c)
            continue

        bucket = agg.setdefault((r, arm), {"fail": 0.0, "tot": 0.0})
        bb = blk_agg.setdefault((blk, r, arm), {"fail": 0.0, "tot": 0.0})

        if arm in ("ARM1_UNENCODED", "ARM2_REP3_NOCORR", "ARM2_REP5_NOCORR"):
            for bs, c in cnts.items():
                out_s = str(bs).split()[0]
                if arm == "ARM1_UNENCODED":
                    pred = int(out_s[-1])
                else:
                    pred = 1 if out_s.count("1") > (len(out_s) // 2) else 0
                bucket["tot"] += float(c)
                bb["tot"] += float(c)
                if pred != lb:
                    bucket["fail"] += float(c)
                    bb["fail"] += float(c)
        else:
            is_d5 = arm == "ARM4A_REP5_MWPM"
            d_val = 5 if is_d5 else 3
            n_anc = d_val - 1
            dec = dec5[r] if is_d5 else dec3[r]
            post_arm = "ARM3_REP5_POSTSELECTED" if is_d5 else "ARM3_REP3_POSTSELECTED"
            b_post = agg.setdefault((r, post_arm), {"fail": 0.0, "acc": 0.0, "tot": 0.0})
            s_entry = streak_stats.setdefault(
                (arm, r),
                {
                    "d1": np.zeros(n_anc, dtype=float),
                    "d11": np.zeros(n_anc, dtype=float),
                    "shots": np.zeros(1, dtype=float),
                },
            )
            for bs, c in cnts.items():
                parts = str(bs).split()
                out_s = parts[0].zfill(d_val)
                data_bits = [int(out_s[d_val - 1 - k]) for k in range(d_val)]
                syn_parts = parts[1:][::-1]
                meas = np.zeros((r, n_anc), dtype=np.int8)
                for r_idx, sp in enumerate(syn_parts):
                    spz = sp.zfill(n_anc)
                    for k in range(n_anc):
                        meas[r_idx, k] = int(spz[n_anc - 1 - k])
                final_syn = np.array(
                    [data_bits[k] ^ data_bits[k + 1] for k in range(n_anc)], dtype=np.int8
                )
                dets = np.zeros((r + 1, n_anc), dtype=np.int8)
                dets[0] = meas[0]
                for r_idx in range(1, r):
                    dets[r_idx] = meas[r_idx] ^ meas[r_idx - 1]
                dets[r] = final_syn ^ meas[r - 1]

                s_entry["shots"][0] += float(c)
                s_entry["d1"] += float(c) * np.mean(dets, axis=0)
                if r >= 1:
                    s_entry["d11"] += float(c) * np.mean(dets[:-1, :] * dets[1:, :], axis=0)

                det_flat = dets.reshape(-1)
                active = [idx for idx, val in enumerate(det_flat) if int(val) == 1]
                raw_logical = 0
                for b_v in data_bits:
                    raw_logical ^= b_v
                corr = int(dec.decode_nodes(active))
                pred = raw_logical ^ corr
                bucket["tot"] += float(c)
                bb["tot"] += float(c)
                if pred != lb:
                    bucket["fail"] += float(c)
                    bb["fail"] += float(c)

                b_post["tot"] += float(c)
                if len(active) == 0:
                    b_post["acc"] += float(c)
                    maj_pred = 1 if sum(data_bits) > (d_val // 2) else 0
                    if maj_pred != lb:
                        b_post["fail"] += float(c)

    rows_out: List[Dict[str, Any]] = []
    for r in ROUNDS_LIST:
        p1 = agg[(r, "ARM1_UNENCODED")]["fail"] / max(1.0, agg[(r, "ARM1_UNENCODED")]["tot"])
        p2_3 = agg[(r, "ARM2_REP3_NOCORR")]["fail"] / max(1.0, agg[(r, "ARM2_REP3_NOCORR")]["tot"])
        p2_5 = agg[(r, "ARM2_REP5_NOCORR")]["fail"] / max(1.0, agg[(r, "ARM2_REP5_NOCORR")]["tot"])
        p4_3 = agg[(r, "ARM4A_REP3_MWPM")]["fail"] / max(1.0, agg[(r, "ARM4A_REP3_MWPM")]["tot"])
        p4_5 = agg[(r, "ARM4A_REP5_MWPM")]["fail"] / max(1.0, agg[(r, "ARM4A_REP5_MWPM")]["tot"])
        bp3 = agg[(r, "ARM3_REP3_POSTSELECTED")]
        bp5 = agg[(r, "ARM3_REP5_POSTSELECTED")]

        blk_p3 = [
            blk_agg[(b, r, "ARM4A_REP3_MWPM")]["fail"]
            / max(1.0, blk_agg[(b, r, "ARM4A_REP3_MWPM")]["tot"])
            for b in range(int(manifest["n_blocks"]))
        ]
        blk_p5 = [
            blk_agg[(b, r, "ARM4A_REP5_MWPM")]["fail"]
            / max(1.0, blk_agg[(b, r, "ARM4A_REP5_MWPM")]["tot"])
            for b in range(int(manifest["n_blocks"]))
        ]

        s5 = streak_stats[("ARM4A_REP5_MWPM", r)]
        p_d1_5 = s5["d1"] / max(1.0, float(s5["shots"][0]))
        p_d11_5 = s5["d11"] / max(1.0, float(s5["shots"][0]))
        streak_ratios_5 = [
            round(float(p_d11_5[k] / max(1e-9, p_d1_5[k] ** 2)), 4) for k in range(4)
        ]

        rows_out.append(
            {
                "rounds": r,
                "arm1_unencoded_p_fail": round(p1, 6),
                "arm2_rep3_nocorr_p_fail": round(p2_3, 6),
                "arm2_rep5_nocorr_p_fail": round(p2_5, 6),
                "arm4a_rep3_mwpm_p_fail": round(p4_3, 6),
                "arm4a_rep5_mwpm_p_fail": round(p4_5, 6),
                "lambda_3_to_5_mwpm": round(p4_3 / max(1e-6, p4_5), 4),
                "lambda_3_to_5_nocorr": round(p2_3 / max(1e-6, p2_5), 4),
                "rep5_crossing_below_rep3_mwpm_pp": round((p4_3 - p4_5) * 100.0, 4),
                "rep5_crossing_below_rep5_nocorr_pp": round((p2_5 - p4_5) * 100.0, 4),
                "arm3_rep3_postselected_p_fail": round(bp3["fail"] / max(1.0, bp3["acc"]), 6),
                "arm3_rep3_acceptance_rate": round(bp3["acc"] / max(1.0, bp3["tot"]), 6),
                "arm3_rep5_postselected_p_fail": round(bp5["fail"] / max(1.0, bp5["acc"]), 6),
                "arm3_rep5_acceptance_rate": round(bp5["acc"] / max(1.0, bp5["tot"]), 6),
                "block_arm4a_rep3_p_fail": [round(x, 6) for x in blk_p3],
                "block_arm4a_rep5_p_fail": [round(x, 6) for x in blk_p5],
                "rep5_ancilla_streak_ratios_over_bernoulli": streak_ratios_5,
            }
        )

    data_qs = [120, 122, 136, 142, 140]
    ro_per_q = {
        str(data_qs[k]): round(
            float(ro_cal_per_qubit_err[k] / max(1.0, ro_cal_per_qubit_tot[k])), 6
        )
        for k in range(5)
    }

    # Linear slope per round from R=1 to R=5: (p(R=5) - p(R=1)) / 4.0
    r1_row, r3_row, r5_row = rows_out[0], rows_out[1], rows_out[2]
    slope_unenc = (r5_row["arm1_unencoded_p_fail"] - r1_row["arm1_unencoded_p_fail"]) / 4.0
    slope_rep3_nc = (r5_row["arm2_rep3_nocorr_p_fail"] - r1_row["arm2_rep3_nocorr_p_fail"]) / 4.0
    slope_rep5_nc = (r5_row["arm2_rep5_nocorr_p_fail"] - r1_row["arm2_rep5_nocorr_p_fail"]) / 4.0
    slope_rep3_mwpm = (r5_row["arm4a_rep3_mwpm_p_fail"] - r1_row["arm4a_rep3_mwpm_p_fail"]) / 4.0
    slope_rep5_mwpm = (r5_row["arm4a_rep5_mwpm_p_fail"] - r1_row["arm4a_rep5_mwpm_p_fail"]) / 4.0
    mean_lambda = float(np.mean([row["lambda_3_to_5_mwpm"] for row in rows_out]))

    return {
        "schema": "osiris.rep5_fez.results/1",
        "protocol_id": PROTOCOL_ID,
        "execution_tier": "HARDWARE_ATTESTED" if job_meta.get("job_id") else "SIMULATED",
        "provider_job_id": job_meta.get("job_id"),
        "backend": BACKEND_NAME,
        "chain_9q": list(FEZ_REP5_CHAIN),
        "qpy_sha256": manifest["qpy_sha256"],
        "preregistration_md_sha256": manifest["preregistration_md_sha256"],
        "raw_counts_gz_sha256": raw_gz_sha256,
        "ro_cal_5d_assignment_fidelity": round(ro_cal_ok / max(1.0, ro_cal_tot), 6),
        "ro_cal_5d_per_data_qubit_error": ro_per_q,
        "mean_lambda_3_to_5_mwpm": round(mean_lambda, 4),
        "peak_lambda_3_to_5_mwpm_r3": r3_row["lambda_3_to_5_mwpm"],
        "slope_per_round": {
            "arm1_unencoded": round(slope_unenc, 6),
            "arm2_rep3_nocorr": round(slope_rep3_nc, 6),
            "arm2_rep5_nocorr": round(slope_rep5_nc, 6),
            "arm4a_rep3_mwpm": round(slope_rep3_mwpm, 6),
            "arm4a_rep5_mwpm": round(slope_rep5_mwpm, 6),
            "lambda_3_to_5_slope_ratio": round(slope_rep3_mwpm / max(1e-9, slope_rep5_mwpm), 4),
        },
        "rounds_summary": rows_out,
    }


def submit_frozen_and_harvest() -> Dict[str, Any]:
    from qiskit_ibm_runtime import SamplerV2

    manifest = json.loads(
        (EXPERIMENT_DIR / "rep5_fez_staging_manifest.json").read_text(encoding="utf-8")
    )
    qpy_path = EXPERIMENT_DIR / "rep5_fez_staged_circuits.qpy"
    if hashlib.sha256(qpy_path.read_bytes()).hexdigest() != manifest["qpy_sha256"]:
        raise RuntimeError("QPY SHA-256 mismatch against write-ahead manifest!")
    with qpy_path.open("rb") as f:
        isa_circuits = qpy.load(f)

    svc, cred_prov = connect_ibm_runtime_service()
    usage = svc.usage()
    rem = float(usage.get("usage_remaining_seconds", 0.0))
    if rem < float(manifest["min_quota_seconds_gate"]):
        raise RuntimeError(
            f"Quota gate blocked submission: {rem}s < {manifest['min_quota_seconds_gate']}s"
        )

    fez = svc.backend(BACKEND_NAME)
    sampler = SamplerV2(mode=fez)
    sampler.options.default_shots = SHOTS_PER_PUB
    job = sampler.run(isa_circuits)
    jid = str(job.job_id())
    print(f"[SUBMITTED] {PROTOCOL_ID} job_id={jid} on {BACKEND_NAME} (quota_before={rem}s)")

    while str(job.status()) not in ("DONE", "ERROR", "CANCELLED"):
        time.sleep(3.0)
    if str(job.status()) != "DONE":
        raise RuntimeError(f"Job {jid} failed with status={job.status()}")

    res = job.result()
    raw_counts: Dict[str, Dict[str, int]] = {}
    for idx, qc_i in enumerate(isa_circuits):
        reg_rev = [creg.name for creg in reversed(qc_i.cregs)]
        raw_counts[str(idx)] = _extract_composite_bitstrings(res[idx], reg_rev)

    raw_gz = EXPERIMENT_DIR / "physical_rep5_fez_raw_counts.json.gz"
    raw_bytes = (json.dumps(raw_counts, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with raw_gz.open("wb") as raw_fh:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_fh, mtime=0) as gz:
            gz.write(raw_bytes)
    raw_sha = hashlib.sha256(raw_gz.read_bytes()).hexdigest()

    usage_after = svc.usage()
    jmeta = redact_ibm_job_metadata(job)
    jmeta["job_id"] = jid
    jmeta["backend"] = BACKEND_NAME
    jmeta["total_pubs"] = len(isa_circuits)
    jmeta["shots_per_pub"] = SHOTS_PER_PUB
    jmeta["total_shots"] = len(isa_circuits) * SHOTS_PER_PUB
    jmeta["credential_provenance"] = cred_prov
    jmeta["quota_remaining_before_s"] = rem
    jmeta["quota_remaining_after_s"] = float(usage_after.get("usage_remaining_seconds", 0.0))
    (EXPERIMENT_DIR / "rep5_fez_submitted_job.json").write_bytes(
        (json.dumps(jmeta, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )

    out = analyze_harvested(manifest, raw_counts, jmeta, raw_sha)
    (EXPERIMENT_DIR / "physical_rep5_fez_results.json").write_bytes(
        (json.dumps(out, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(
        f"[HARVESTED] {PROTOCOL_ID} job_id={jid} | quota_after={jmeta['quota_remaining_after_s']}s"
    )
    print(json.dumps(out, indent=2))
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
