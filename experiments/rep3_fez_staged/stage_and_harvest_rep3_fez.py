#!/usr/bin/env python3
"""
stage_and_harvest_rep3_fez.py — Write-ahead `--stage-only` (0s QPU) & harvest runner for
`OSIRIS-QEC-REP3-FEZ` (`Card C` 4-arm `[[3,1,3]]` Pauli-frame vs physical feed-forward on `ibm_fez`).
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
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qpy

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osiris_cli.ibm_credentials import connect_ibm_runtime_service, redact_ibm_job_metadata  # noqa: E402


def _load_card_c_gate() -> Any:
    mod_name = "_qec_card_c_gate_mod"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    p = REPO_ROOT / "experiments" / "qec_card_c" / "qec_crossing_gate.py"
    spec = importlib.util.spec_from_file_location(mod_name, p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROTOCOL_ID = "OSIRIS-QEC-REP3-FEZ"
BACKEND_NAME = "ibm_fez"
# Pinned [d0, a0, d1, a1, d2] chain on ibm_fez:
FEZ_REP3_CHAIN: Tuple[int, int, int, int, int] = (58, 71, 70, 69, 78)
ROUNDS_LIST: Tuple[int, int, int] = (1, 3, 5)
ROUND_DT = 904  # 904 * 4 ns = 3.616 us matched syndrome + measure + reset cycle duration
N_BLOCKS = 3
SHOTS_PER_PUB = 512
SCHEDULE_SEED = 20261125


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _emit_h_isa(qc: QuantumCircuit, q: int) -> None:
    """Exact H gate in Heron r2 native RZ-SX-RZ basis: RZ(pi/2) SX RZ(pi/2)."""
    qc.rz(0.5 * math.pi, q)
    qc.sx(q)
    qc.rz(0.5 * math.pi, q)


def _emit_cx_isa(qc: QuantumCircuit, ctrl: int, targ: int, cz_pairs: Any = None) -> None:
    """Exact CX(ctrl -> targ) = (I x H) CZ (I x H) in Heron r2 native gates."""
    _emit_h_isa(qc, targ)
    if cz_pairs is None or (ctrl, targ) in cz_pairs:
        qc.cz(ctrl, targ)
    elif (targ, ctrl) in cz_pairs:
        qc.cz(targ, ctrl)
    else:
        raise RuntimeError(f"Missing CZ edge ({ctrl}, {targ})")
    _emit_h_isa(qc, targ)


def build_rep3_circuit(
    arm: str,
    rounds: int,
    logical_bit: int,
    num_qubits: int,
    chain: Sequence[int],
    cz_pairs: Any = None,
    name: str = "rep3",
) -> QuantumCircuit:
    """Construct strict Heron r2 ISA circuit for each `Card C` arm."""
    d0, a0, d1, a1, d2 = (int(chain[0]), int(chain[1]), int(chain[2]), int(chain[3]), int(chain[4]))
    qr = QuantumRegister(num_qubits, "q")

    if arm == "ARM1_UNENCODED":
        out_r = ClassicalRegister(1, "out")
        qc = QuantumCircuit(qr, out_r, name=name)
        if logical_bit == 1:
            qc.x(d1)
        qc.barrier(d1)
        for _ in range(rounds):
            qc.delay(ROUND_DT, d1, unit="dt")
        qc.barrier(d1)
        qc.measure(d1, out_r[0])
        return qc

    if arm == "ARM2_ENCODED_NO_CORR":
        out_r = ClassicalRegister(3, "out")
        qc = QuantumCircuit(qr, out_r, name=name)
        if logical_bit == 1:
            qc.x(d0)
            qc.x(d1)
            qc.x(d2)
        qc.barrier(d0, d1, d2)
        for _ in range(rounds):
            qc.delay(ROUND_DT, d0, unit="dt")
            qc.delay(ROUND_DT, d1, unit="dt")
            qc.delay(ROUND_DT, d2, unit="dt")
        qc.barrier(d0, d1, d2)
        qc.measure(d0, out_r[0])
        qc.measure(d1, out_r[1])
        qc.measure(d2, out_r[2])
        return qc

    if arm in ("ARM3_4A_SYNDROME_STREAM", "ARM4B_PHYSICAL_FEEDFORWARD"):
        syn_regs = [ClassicalRegister(2, f"s{r}") for r in range(rounds)]
        out_r = ClassicalRegister(3, "out")
        qc = QuantumCircuit(qr, *syn_regs, out_r, name=name)
        if logical_bit == 1:
            qc.x(d0)
            qc.x(d1)
            qc.x(d2)
        qc.barrier(d0, a0, d1, a1, d2)
        for r in range(rounds):
            # Step 1: CX(d0->a0) and CX(d1->a1) in parallel
            _emit_cx_isa(qc, d0, a0, cz_pairs)
            _emit_cx_isa(qc, d1, a1, cz_pairs)
            qc.barrier(d0, a0, d1, a1, d2)
            # Step 2: CX(d1->a0) and CX(d2->a1) in parallel
            _emit_cx_isa(qc, d1, a0, cz_pairs)
            _emit_cx_isa(qc, d2, a1, cz_pairs)
            qc.barrier(d0, a0, d1, a1, d2)
            # Mid-circuit syndrome measurement + active reset
            qc.measure(a0, syn_regs[r][0])
            qc.measure(a1, syn_regs[r][1])
            qc.reset(a0)
            qc.reset(a1)
            if arm == "ARM4B_PHYSICAL_FEEDFORWARD":
                # syn_regs[r] bits: bit0 = Z0 Z1, bit1 = Z1 Z2
                # 0b01 (1) -> d0 flipped; 0b11 (3) -> d1 flipped; 0b10 (2) -> d2 flipped
                with qc.if_test((syn_regs[r], 1)):
                    qc.x(d0)
                with qc.if_test((syn_regs[r], 3)):
                    qc.x(d1)
                with qc.if_test((syn_regs[r], 2)):
                    qc.x(d2)
            qc.barrier(d0, a0, d1, a1, d2)
        qc.measure(d0, out_r[0])
        qc.measure(d1, out_r[1])
        qc.measure(d2, out_r[2])
        return qc

    raise ValueError(f"Unknown arm: {arm}")


def verify_noiseless_and_single_fault_5q() -> Dict[str, Any]:
    """Offline Aer verification of compact 5Q [0, 1, 2, 3, 4] circuits before 156Q ISA staging."""
    from qiskit_aer import AerSimulator

    sim = AerSimulator()
    compact_chain = (0, 1, 2, 3, 4)

    for r in ROUNDS_LIST:
        for lbit in (0, 1):
            for arm in (
                "ARM1_UNENCODED",
                "ARM2_ENCODED_NO_CORR",
                "ARM3_4A_SYNDROME_STREAM",
                "ARM4B_PHYSICAL_FEEDFORWARD",
            ):
                qc5 = build_rep3_circuit(arm, r, lbit, 5, compact_chain, cz_pairs=None, name=f"chk_{arm}_{r}_{lbit}")
                res = sim.run(qc5, shots=64).result().get_counts()
                exp_out = str(lbit) if arm == "ARM1_UNENCODED" else (f"{lbit}{lbit}{lbit}")
                for k in res:
                    parts = k.split()
                    out_bits = parts[0]  # Last classical register `out` is leftmost in Qiskit count strings
                    assert out_bits == exp_out, f"Noiseless mismatch on {arm} R={r} b={lbit}: got {k}"
                    if arm in ("ARM3_4A_SYNDROME_STREAM", "ARM4B_PHYSICAL_FEEDFORWARD"):
                        assert all(p == "00" for p in parts[1:]), f"Non-zero noiseless syndrome: {k}"

    # Positive single-fault feed-forward test: inject X on d0, d1, or d2 before syndrome extraction in ARM4B
    for fault_q in (0, 2, 4):
        for lbit in (0, 1):
            qr = QuantumRegister(5, "q")
            sr = ClassicalRegister(2, "s0")
            out_r = ClassicalRegister(3, "out")
            qc_f = QuantumCircuit(qr, sr, out_r)
            if lbit == 1:
                qc_f.x([0, 2, 4])
            qc_f.x(fault_q)  # Injected single bit-flip fault!
            _emit_cx_isa(qc_f, 0, 1)
            _emit_cx_isa(qc_f, 2, 3)
            _emit_cx_isa(qc_f, 2, 1)
            _emit_cx_isa(qc_f, 4, 3)
            qc_f.measure(1, sr[0])
            qc_f.measure(3, sr[1])
            qc_f.reset([1, 3])
            with qc_f.if_test((sr, 1)):
                qc_f.x(0)
            with qc_f.if_test((sr, 3)):
                qc_f.x(2)
            with qc_f.if_test((sr, 2)):
                qc_f.x(4)
            qc_f.measure([0, 2, 4], [out_r[0], out_r[1], out_r[2]])
            cnts = sim.run(qc_f, shots=32).result().get_counts()
            for k in cnts:
                assert k.split()[0] == f"{lbit}{lbit}{lbit}", f"ARM4B feed-forward failed on fault_q={fault_q}: {k}"

    return {"noiseless_all_24_cells_exact": True, "arm4b_single_fault_feedforward_exact": True}


def stage_experiment() -> Dict[str, Any]:
    from qiskit_ibm_runtime.utils.validations import validate_isa_circuits

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    ver = verify_noiseless_and_single_fault_5q()

    svc, cred_prov = connect_ibm_runtime_service()
    fez = svc.backend(BACKEND_NAME)
    tgt = fez.target
    props = fez.properties()
    cz_pairs = tgt["cz"]

    rng = np.random.default_rng(SCHEDULE_SEED)
    isa_circuits: List[QuantumCircuit] = []
    schedule: List[Dict[str, Any]] = []
    d0, a0, d1, a1, d2 = FEZ_REP3_CHAIN

    for blk in range(N_BLOCKS):
        blk_items: List[Tuple[QuantumCircuit, Dict[str, Any]]] = []
        for r in ROUNDS_LIST:
            for lbit in (0, 1):
                for arm in (
                    "ARM1_UNENCODED",
                    "ARM2_ENCODED_NO_CORR",
                    "ARM3_4A_SYNDROME_STREAM",
                    "ARM4B_PHYSICAL_FEEDFORWARD",
                ):
                    cname = f"b{blk}_{arm.lower()}_r{r}_l{lbit}"
                    qc_isa = build_rep3_circuit(arm, r, lbit, tgt.num_qubits, FEZ_REP3_CHAIN, cz_pairs, cname)
                    blk_items.append(
                        (
                            qc_isa,
                            {
                                "block": blk,
                                "arm": arm,
                                "rounds": r,
                                "logical_bit": lbit,
                            },
                        )
                    )
        for ro_state in range(8):
            cname = f"b{blk}_ro_cal_{ro_state:03b}"
            qr = QuantumRegister(tgt.num_qubits, "q")
            out_r = ClassicalRegister(3, "out")
            qc_ro = QuantumCircuit(qr, out_r, name=cname)
            for idx_b, pq in enumerate((d0, d1, d2)):
                if (ro_state >> idx_b) & 1:
                    qc_ro.x(pq)
            qc_ro.barrier(d0, d1, d2)
            qc_ro.measure(d0, out_r[0])
            qc_ro.measure(d1, out_r[1])
            qc_ro.measure(d2, out_r[2])
            blk_items.append((qc_ro, {"block": blk, "arm": "RO_CAL", "ro_state": ro_state}))

        for idx in rng.permutation(len(blk_items)):
            qc_i, meta_i = blk_items[int(idx)]
            m = dict(meta_i)
            m["pub_index"] = len(isa_circuits)
            m["circuit_name"] = qc_i.name
            isa_circuits.append(qc_i)
            schedule.append(m)

    validate_isa_circuits(isa_circuits, tgt)

    qpy_path = EXPERIMENT_DIR / "rep3_fez_staged_circuits.qpy"
    with qpy_path.open("wb") as f:
        qpy.dump(isa_circuits, f)
    qpy_sha = hashlib.sha256(qpy_path.read_bytes()).hexdigest()

    cal_snap = {
        "backend": BACKEND_NAME,
        "captured_utc": _utc_now(),
        "last_update_date": str(getattr(props, "last_update_date", None)),
        "fez_rep3_chain_d0_a0_d1_a1_d2": list(FEZ_REP3_CHAIN),
        "round_dt": ROUND_DT,
        "dt_seconds": float(tgt.dt),
        "qubits": {
            str(q): {
                "t1_us": round(float(props.t1(q)) * 1e6, 3),
                "t2_us": round(float(props.t2(q)) * 1e6, 3),
                "readout_error": round(float(props.readout_error(q)), 6),
            }
            for q in FEZ_REP3_CHAIN
        },
    }
    cal_path = EXPERIMENT_DIR / "rep3_fez_calibration_snapshot.json"
    cal_path.write_text(json.dumps(cal_snap, indent=2, sort_keys=True), encoding="utf-8")

    prereg_sha = hashlib.sha256((EXPERIMENT_DIR / "PRE_REGISTRATION.md").read_bytes()).hexdigest()
    manifest = {
        "schema": "osiris.quantum_protocol/1",
        "protocol_id": PROTOCOL_ID,
        "execution_tier": "OFFLINE_STAGED",
        "provider_job_id": None,
        "staged_utc": _utc_now(),
        "backend": BACKEND_NAME,
        "credential_provenance": cred_prov,
        "fez_rep3_chain": list(FEZ_REP3_CHAIN),
        "rounds_list": list(ROUNDS_LIST),
        "round_dt": ROUND_DT,
        "n_blocks": N_BLOCKS,
        "pubs_per_block": 32,
        "total_pubs": len(isa_circuits),
        "shots_per_pub": SHOTS_PER_PUB,
        "total_shots": len(isa_circuits) * SHOTS_PER_PUB,
        "min_quota_seconds_gate": 25.0,
        "qpy_filename": qpy_path.name,
        "qpy_sha256": qpy_sha,
        "preregistration_md_sha256": prereg_sha,
        "calibration_snapshot_sha256": hashlib.sha256(cal_path.read_bytes()).hexdigest(),
        "stage0_verification": ver,
        "schedule": schedule,
    }
    (EXPERIMENT_DIR / "rep3_fez_staging_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[STAGED] {PROTOCOL_ID}: {len(isa_circuits)} PUBs x {SHOTS_PER_PUB} = {manifest['total_shots']} shots | qpy={qpy_sha[:16]}...")
    return manifest


def analyze_harvested(
    manifest: Mapping[str, Any],
    raw_counts_by_pub: Mapping[str, Mapping[str, Tuple[Mapping[str, int], ...]] | Mapping[str, int]],
    job_meta: Mapping[str, Any],
    raw_gz_sha256: str,
) -> Dict[str, Any]:
    """Decode all 4 `Card C` arms (separating Arm 3 postselected + acceptance from Arm 4a MWPM & Arm 4b FF)."""
    cc = _load_card_c_gate()
    fez_prof = cc.build_published_target_profiles()["FEZ_PUBLISHED"]
    decoders = {
        r: cc.auditable_mwpm_decoder(cc.build_repetition_d3_dem(rounds=r, profile=fez_prof, basis="Z"))
        for r in ROUNDS_LIST
    }

    agg: Dict[Tuple[int, str], Dict[str, float]] = {}
    for item in manifest["schedule"]:
        arm = str(item["arm"])
        if arm == "RO_CAL":
            continue
        r = int(item["rounds"])
        lbit = int(item["logical_bit"])
        cnts = raw_counts_by_pub[str(item["pub_index"])]

        if arm in ("ARM1_UNENCODED", "ARM2_ENCODED_NO_CORR", "ARM4B_PHYSICAL_FEEDFORWARD"):
            bucket = agg.setdefault((r, arm), {"fail": 0.0, "tot": 0.0})
            for bitstr, cnt in cnts.items():
                out_str = str(bitstr).split()[0]
                if arm == "ARM1_UNENCODED":
                    pred = int(out_str[-1])
                else:
                    pred = 1 if out_str.count("1") >= 2 else 0
                bucket["tot"] += float(cnt)
                if pred != lbit:
                    bucket["fail"] += float(cnt)
        elif arm == "ARM3_4A_SYNDROME_STREAM":
            b3 = agg.setdefault((r, "ARM3_POSTSELECTED"), {"fail": 0.0, "accepted": 0.0, "tot": 0.0})
            b4a = agg.setdefault((r, "ARM4A_DECODED_PAULI_FRAME"), {"fail": 0.0, "tot": 0.0})
            dec = decoders[r]
            for bitstr, cnt in cnts.items():
                # Space-separated classical registers in reverse declaration order: "out s{R-1} ... s0"
                parts = str(bitstr).split()
                out_str = parts[0].zfill(3)
                d2_b, d1_b, d0_b = int(out_str[0]), int(out_str[1]), int(out_str[2])
                syn_parts = parts[1:][::-1]  # s0 .. s{R-1}, each "b_a1 b_a0"
                meas = np.zeros((r, 2), dtype=np.int8)
                for r_idx, sp in enumerate(syn_parts):
                    spz = sp.zfill(2)
                    meas[r_idx, 0] = int(spz[-1])  # a0 = Z0 Z1
                    meas[r_idx, 1] = int(spz[-2])  # a1 = Z1 Z2
                # Space-time detectors: D_0 = m_0, D_r = m_r ^ m_{r-1}, D_R = final_parity ^ m_{R-1}
                final_syn = np.array([d0_b ^ d1_b, d1_b ^ d2_b], dtype=np.int8)
                dets = np.zeros((r + 1, 2), dtype=np.int8)
                dets[0] = meas[0]
                for r_idx in range(1, r):
                    dets[r_idx] = meas[r_idx] ^ meas[r_idx - 1]
                dets[r] = final_syn ^ meas[r - 1]
                det_flat = dets.reshape(-1)
                syn_mask = sum(int(det_flat[k]) << k for k in range(len(det_flat)))
                raw_logical = d0_b ^ d1_b ^ d2_b
                corr_flip = int(dec.decode_mask(syn_mask))
                pred_4a = raw_logical ^ corr_flip

                b4a["tot"] += float(cnt)
                if pred_4a != lbit:
                    b4a["fail"] += float(cnt)

                b3["tot"] += float(cnt)
                if int(np.sum(det_flat)) == 0:
                    b3["accepted"] += float(cnt)
                    pred_3 = 1 if (d0_b + d1_b + d2_b) >= 2 else 0
                    if pred_3 != lbit:
                        b3["fail"] += float(cnt)

    rows_out: List[Dict[str, Any]] = []
    for r in ROUNDS_LIST:
        row: Dict[str, Any] = {"rounds": r}
        for k_arm in (
            "ARM1_UNENCODED",
            "ARM2_ENCODED_NO_CORR",
            "ARM4A_DECODED_PAULI_FRAME",
            "ARM4B_PHYSICAL_FEEDFORWARD",
        ):
            b = agg[(r, k_arm)]
            row[f"{k_arm.lower()}_p_fail"] = round(b["fail"] / max(1.0, b["tot"]), 6)
        b3 = agg[(r, "ARM3_POSTSELECTED")]
        row["arm3_postselected_acceptance_rate"] = round(b3["accepted"] / max(1.0, b3["tot"]), 6)
        row["arm3_postselected_conditional_p_fail"] = round(b3["fail"] / max(1.0, b3["accepted"]), 6)
        rows_out.append(row)

    return {
        "schema": "osiris.rep3_fez.results/1",
        "protocol_id": PROTOCOL_ID,
        "execution_tier": "HARDWARE_ATTESTED" if job_meta.get("job_id") else "SIMULATED",
        "provider_job_id": job_meta.get("job_id"),
        "raw_counts_gz_sha256": raw_gz_sha256,
        "rounds_summary": rows_out,
    }


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


def submit_frozen_and_harvest() -> Dict[str, Any]:
    from qiskit_ibm_runtime import SamplerV2

    manifest = json.loads((EXPERIMENT_DIR / "rep3_fez_staging_manifest.json").read_text(encoding="utf-8"))
    qpy_path = EXPERIMENT_DIR / "rep3_fez_staged_circuits.qpy"
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
    raw_counts: Dict[str, Dict[str, int]] = {}
    for idx, qc_i in enumerate(isa_circuits):
        reg_rev = [creg.name for creg in reversed(qc_i.cregs)]
        raw_counts[str(idx)] = _extract_composite_bitstrings(res[idx], reg_rev)

    raw_gz = EXPERIMENT_DIR / "physical_rep3_fez_raw_counts.json.gz"
    with gzip.open(raw_gz, "wb") as gz:
        gz.write(json.dumps(raw_counts, sort_keys=True).encode("utf-8"))
    jmeta = redact_ibm_job_metadata(job)
    out = analyze_harvested(manifest, raw_counts, jmeta, hashlib.sha256(raw_gz.read_bytes()).hexdigest())
    (EXPERIMENT_DIR / "physical_rep3_fez_results.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
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
