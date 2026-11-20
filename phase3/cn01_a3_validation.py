"""
CORRELATED-NOISE-01 amendment A-3: validation on synthetic counts (no hardware data)
===================================================================================

Synthesises complete Stage-1 count sets in the locked schedule's exact bit layout (244 PUBs, 10
multiplexed couplers, 2,048 shots) from known per-coupler truths, then runs both the locked
analysis (`analyze_correlated_noise_01_live_counts`) and the A-3 analysis (`a3_inference`) on each.

Truths. Four have no spatial correlation, so any correlated or non-Markovian-correlated claim made
on them is a false positive:
  white                     H0 (independent white dephasing)
  local_1f                  H0_1f (independent 1/f plus a white floor)
  local_1f_ramsey_detuned   H0_1f plus a +/-15 kHz frame detuning on the Ramsey arms
  local_ou                  H3 with spatial_corr = 0 (independent single-timescale memory)
Two are correlated with a correlation time shorter than the DD16 slot (512 ns), to measure power:
  correlated_h2             H2 (correlated Markovian), rho in [0.5, 0.7]
  correlated_h3_fast        H3 with tau_c in [0.25, 0.35] us, spatial_corr in [0.5, 0.7]
One is correlated but slow, reported without a power criterion:
  correlated_h3_slow        H3 with tau_c in [1, 4] us, spatial_corr in [0.5, 0.7]
DD16 refocuses noise slower than its slot, so slow correlated noise leaves almost no stagger or
Phi+/Psi+ signature: at these strengths the expected C_DFS is below 0.1 pp, against a design noise
floor of a few tenths of a pp. No analysis of this design can detect it.
All truths use the hardware pulse placement, per-coupler T1 from the lock-time calibration with 15%
jitter, per-qubit readout error, a per-(coupler, block) drift offset and per-PUB jitter.

Acceptance criteria, fixed before the full run:
  * On every local truth, A-3 makes any non-local claim (correlated dephasing, correlated memory,
    spectator ZZ, pulse collision) in at most 5% of replicates.
  * On each fast correlated truth, A-3 reports correlated dephasing in at least 80% of replicates.
The locked analysis's rates are reported for comparison. Disclosure: the alternative truths were
revised once before the full run. A smoke test of three datasets, followed by a calculation of each
truth's expected C_DFS, showed that the first correlated-H3 truth (tau_c 1-4 us) was undetectable
by design. It is kept as `correlated_h3_slow`, and the two fast truths carry the power criterion.
The fast-H3 strength range (white-equivalent rate sigma^2 * tau_c of 6-15 kHz) was chosen from the
same calculation, so that its expected C_DFS (about 0.8-3 pp) is comparable to the H2 truth's (1-4 pp).

Usage: python -m phase3.cn01_a3_validation [--null-replicates 100] [--alt-replicates 50] [--workers 16]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from phase2.correlated_noise_01.correlated_noise_models import (
    H0IndependentLocalModel,
    NoiseModelBase,
)
from phase3.cn01_amendment_a3 import (
    H0Local1fHardwareTiming,
    H2HardwareTiming,
    H3HardwareTiming,
    a3_inference,
    a3_random_split,
)
from phase3.execute_and_analyze_live import (
    _build_cn01_stage1_probe_conditions,
    analyze_correlated_noise_01_live_counts,
    cn01_fidelity_arrays_from_counts,
)

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs" / "phase3"
CODE_FILES: Tuple[str, ...] = (
    "phase3/cn01_amendment_a3.py",
    "phase3/cn01_a3_validation.py",
    "phase3/execute_and_analyze_live.py",
    "phase2/correlated_noise_01/correlated_noise_models.py",
    "phase2/correlated_noise_01/simulate_correlated_noise.py",
)
NULL_TRUTHS: Tuple[str, ...] = ("white", "local_1f", "local_1f_ramsey_detuned", "local_ou")
ALT_TRUTHS: Tuple[str, ...] = ("correlated_h2", "correlated_h3_fast")
DESCRIPTIVE_TRUTHS: Tuple[str, ...] = ("correlated_h3_slow",)
BLOCK_DRIFT_SD_F = 0.003   # per (coupler, block) offset on every fidelity in that block
PUB_JITTER_SD_E = 0.004    # per PUB jitter on each Pauli expectation (half of it on 1Q P(0))
MAX_NULL_CLAIM_RATE = 0.05
MIN_POWER = 0.80


def _seed(*parts: Any) -> int:
    return int(hashlib.sha256(":".join(str(p) for p in ("CN01-A3:validation",) + parts).encode()).hexdigest()[:16], 16)


def _assignment(p10: float, p01: float) -> np.ndarray:
    """Single-qubit assignment matrix, columns = prepared 0/1, rows = measured 0/1."""
    return np.array([[1.0 - p10, p01], [p10, 1.0 - p01]])


def draw_truths(truth: str, cohort: Sequence[Mapping[str, Any]], cal: Mapping[str, Any], rng: np.random.Generator
                ) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    q = cal.get("qubits", {})
    for c in cohort:
        u, v = c["coupler"]
        t1u = float(q.get(str(u), {}).get("t1_s") or 170e-6) * float(rng.lognormal(0.0, 0.15))
        t1v = float(q.get(str(v), {}).get("t1_s") or 170e-6) * float(rng.lognormal(0.0, 0.15))
        a_u = _assignment(rng.uniform(0.005, 0.03), rng.uniform(0.01, 0.05))
        a_v = _assignment(rng.uniform(0.005, 0.03), rng.uniform(0.01, 0.05))
        m2 = np.kron(a_u, a_v)  # index 2*b_u + b_v, as the analysis orders it
        f0 = float(np.clip(np.mean(np.diag(m2)) * (1.0 - 2.0 * float(c["twoq_error"])) - rng.uniform(0.0, 0.01),
                           0.85, 0.995))
        gw = lambda: float(rng.uniform(200.0, 1500.0))  # noqa: E731
        model: NoiseModelBase
        if truth == "white":
            model = H0IndependentLocalModel(t1u, t1v, float(rng.uniform(1500, 5000)), float(rng.uniform(1500, 5000)), f0)
        elif truth in ("local_1f", "local_1f_ramsey_detuned"):
            model = H0Local1fHardwareTiming(
                t1_u_s=t1u, t1_v_s=t1v, gamma_white_u_hz=gw(), gamma_white_v_hz=gw(),
                sigma_ou_u_rad_s=0.0, sigma_ou_v_rad_s=0.0, spatial_corr=0.0, f0_spam=f0,
                amp_1f_u_rad_s=float(rng.uniform(30e3, 90e3)), amp_1f_v_rad_s=float(rng.uniform(30e3, 90e3)))
        elif truth in ("local_ou", "correlated_h3_slow"):
            model = H3HardwareTiming(
                t1_u_s=t1u, t1_v_s=t1v, gamma_white_u_hz=gw(), gamma_white_v_hz=gw(),
                sigma_ou_u_rad_s=float(rng.uniform(30e3, 80e3)), sigma_ou_v_rad_s=float(rng.uniform(30e3, 80e3)),
                tau_c_s=float(math.exp(rng.uniform(math.log(1e-6), math.log(4e-6)))),
                spatial_corr=0.0 if truth == "local_ou" else float(rng.uniform(0.5, 0.7)), f0_spam=f0)
        elif truth == "correlated_h3_fast":
            # Correlation time below the DD16 slot; sigma set from a white-equivalent rate
            # sigma^2 * tau_c of 6-15 kHz, which gives an expected C_DFS comparable to the H2 truth.
            tau = float(rng.uniform(0.25e-6, 0.35e-6))
            sig = lambda: float(math.sqrt(rng.uniform(6000.0, 15000.0) / tau))  # noqa: E731
            model = H3HardwareTiming(
                t1_u_s=t1u, t1_v_s=t1v, gamma_white_u_hz=gw(), gamma_white_v_hz=gw(),
                sigma_ou_u_rad_s=sig(), sigma_ou_v_rad_s=sig(), tau_c_s=tau,
                spatial_corr=float(rng.uniform(0.5, 0.7)), f0_spam=f0)
        elif truth == "correlated_h2":
            model = H2HardwareTiming(
                base=H0IndependentLocalModel(t1u, t1v, float(rng.uniform(1500, 5000)), float(rng.uniform(1500, 5000)), f0),
                rho_spatial=float(rng.uniform(0.5, 0.7)), spatial_length_hops=2.0, enforce_lattice_psd=True)
        else:
            raise ValueError(f"Unknown truth {truth!r}")
        detune_hz = float(rng.uniform(-15e3, 15e3)) if truth == "local_1f_ramsey_detuned" else 0.0
        out.append({"model": model, "t1_u": t1u, "t1_v": t1v, "f0": f0, "a_u": a_u, "a_v": a_v,
                    "ramsey_detuning_hz": detune_hz})
    return out


def _counts_from_ints(ints: np.ndarray, n_bits: int) -> Dict[str, int]:
    vals, cnts = np.unique(ints, return_counts=True)
    return {format(int(x), f"0{n_bits}b"): int(n) for x, n in zip(vals, cnts)}


def synthesize_counts(lock: Mapping[str, Any], truths: Sequence[Mapping[str, Any]], rng: np.random.Generator
                      ) -> List[Dict[str, int]]:
    """One complete count set for the locked schedule."""
    suite = {c.condition_id: c for c in _build_cn01_stage1_probe_conditions()}
    n_slots = len(truths)
    n_blocks = int(lock["n_blocks"])
    shots = int(lock["shots_per_pub"])
    drift = rng.normal(0.0, BLOCK_DRIFT_SD_F, size=(n_slots, n_blocks))
    # Expected values are computed once per (coupler, condition).
    f_pred = [{cid: float(t["model"].predict(c)) for cid, c in suite.items()} for t in truths]
    counts: List[Dict[str, int]] = []
    shift_2q = np.array([2 * s for s in range(n_slots)], dtype=np.int64)
    for rec in lock["pub_schedule"]:
        kind, cid, b = rec["kind"], rec["condition_id"], int(rec["block"])
        if kind == "2q_spam":
            bu_prep, bv_prep = int(rec["prep"][0]), int(rec["prep"][1])
            probs = np.array([
                np.kron(t["a_u"][:, bu_prep], t["a_v"][:, bv_prep]) for t in truths])  # (n_slots, 4)
        elif kind == "1q":
            cond = suite[cid]
            p0 = np.empty(n_slots)
            for s, t in enumerate(truths):
                val = f_pred[s][cid]
                if cond.n_dd_pulses == 0 and t["ramsey_detuning_hz"]:
                    val = 0.5 + (val - 0.5) * math.cos(2.0 * math.pi * t["ramsey_detuning_hz"] * cond.total_time_s)
                p0[s] = val + drift[s, b] + rng.normal(0.0, 0.5 * PUB_JITTER_SD_E)
            p0 = np.clip(p0, 0.0, 1.0)
            bits = (rng.random((n_slots, shots)) >= p0[:, None]).astype(np.int64)
            ints = np.sum(bits << np.arange(n_slots, dtype=np.int64)[:, None], axis=0)
            counts.append(_counts_from_ints(ints, n_slots))
            continue
        else:  # 2q_bell
            ref_id = "phi_dd16_d0ns" if cid.endswith("_aa_sentinel") else cid
            cond = suite[ref_id]
            is_phi = rec["bell"] == "phi_plus"
            probs = np.empty((n_slots, 4))
            for s, t in enumerate(truths):
                f_val = f_pred[s][ref_id] + drift[s, b]
                alpha = (4.0 * t["f0"] - 1.0) / 3.0
                pop = 0.5 * (1.0 + math.exp(-cond.total_time_s / t["t1_u"]) * math.exp(-cond.total_time_s / t["t1_v"]))
                z = alpha * (2.0 * pop - 1.0) * (1.0 if is_phi else -1.0)
                x = (4.0 * f_val - 1.0 - z) / 2.0 if is_phi else (4.0 * f_val - 1.0 + z) / 2.0
                y = -x if is_phi else x
                e = {"XX": x, "YY": y, "ZZ": z}[rec["basis"]] + rng.normal(0.0, PUB_JITTER_SD_E)
                e = float(np.clip(e, -1.0, 1.0))
                probs[s] = [(1 + e) / 4, (1 - e) / 4, (1 - e) / 4, (1 + e) / 4]
        cdf = np.cumsum(probs, axis=1)
        cdf[:, -1] = 1.0
        draws = rng.random((n_slots, shots))
        k = np.array([np.searchsorted(cdf[s], draws[s], side="right") for s in range(n_slots)], dtype=np.int64)
        bu, bv = k >> 1, k & 1
        ints = np.sum((bu << shift_2q[:, None]) | (bv << (shift_2q[:, None] + 1)), axis=0)
        counts.append(_counts_from_ints(ints, 2 * n_slots))
    return counts


def _locked_summary(res: Mapping[str, Any]) -> Dict[str, Any]:
    sel = res["gls_model_selection_on_training_cohort"]
    ho = res["held_out_6coupler_cluster_validation_df5"]
    mech = res["mechanistic_contrasts_10couplers_holm"]

    def sig_pos(name: str) -> bool:
        return bool(mech[name]["significant_holm_0_05"] and mech[name]["mean"] > 0)

    selected = sel["selected_model"]
    transfer = bool(ho["transfer_beats_h0_p_0_05"])
    return {
        "selected_model": selected,
        "delta_bic_over_h0": sel["delta_bic_over_h0"],
        "transfer_beats_h0": transfer,
        "c_mem_sig_pos": sig_pos("C_memory_dd32_minus_echo1_16us"),
        "c_dfs_sig_pos": sig_pos("C_DFS_asymmetry"),
        # Hardware plan's Stage-2 gate: delta-BIC >= 8 over H0 and held-out improvement over H0.
        "stage2_gate": bool(selected != "H0" and transfer),
        "h3_memory_claim": bool(selected == "H3" and sig_pos("C_memory_dd32_minus_echo1_16us") and transfer),
        "correlated_dephasing_claim": bool(selected in ("H2", "H3") and sig_pos("C_DFS_asymmetry") and transfer),
    }


def _a3_summary(res: Mapping[str, Any]) -> Dict[str, Any]:
    d = res["decisions"]
    claims = ("correlated_dephasing_supported", "non_markovian_correlated_memory_supported",
              "spectator_zz_supported", "pulse_collision_apparatus_supported")
    ho = res["held_out_transfer"]
    dfs = res["mechanistic_contrasts_holm"]["C_DFS_asymmetry"]
    selected = res["model_selection"]["selected_model"]
    return {
        "selected_model": selected,
        "best_local_model": res["model_selection"]["best_local_model"],
        "delta_bic_over_best_local": res["model_selection"]["delta_bic_over_best_local"],
        **{k: bool(d[k]) for k in d},
        "any_nonlocal_claim": bool(any(d[k] for k in claims)),
        # Components, to show where the correlated-dephasing decision is won or lost.
        "in_sample_correlated_evidence": bool(
            selected in ("H2", "H3") and dfs["significant_holm_0_05"] and dfs["mean"] > 0),
        "held_out_primary_pass": bool(
            ho["primary_stagger_change_best_local_minus_candidate"]["candidate_beats_best_local_p_0_05"]),
        "held_out_absolute_pass": bool(
            ho["secondary_absolute_best_local_minus_candidate"]["candidate_beats_best_local_p_0_05"]),
    }


_CTX: Dict[str, Any] = {}


def _init(lock: Mapping[str, Any], cal: Mapping[str, Any], train: Sequence[int]) -> None:
    _CTX.update(lock=lock, cal=cal, train=list(train))


def run_one(job: Tuple[str, int]) -> Dict[str, Any]:
    truth, rep = job
    lock, cal = _CTX["lock"], _CTX["cal"]
    rng = np.random.default_rng(_seed(truth, rep))
    t0 = time.perf_counter()
    truths = draw_truths(truth, lock["coupler_cohort"], cal, rng)
    counts = synthesize_counts(lock, truths, rng)
    locked = analyze_correlated_noise_01_live_counts(lock, counts, cal, f"SYNTHETIC-{truth}-{rep}")
    arrays = cn01_fidelity_arrays_from_counts(lock, counts)
    a3 = a3_inference(arrays, lock["coupler_cohort"], _CTX["train"], cal)
    conds = {c.condition_id: c for c in _build_cn01_stage1_probe_conditions()}

    def expected(t: Mapping[str, Any], a: str, b: str) -> float:
        return float(t["model"].predict(conds[a]) - t["model"].predict(conds[b]))

    dfs = [expected(t, "phi_dd16_d256ns", "phi_dd16_d0ns") - expected(t, "psi_dd16_d256ns", "psi_dd16_d0ns") for t in truths]
    mem = [expected(t, "echo_dd32_16us", "echo1_16us") for t in truths]
    return {"truth": truth, "replicate": rep, "seconds": round(time.perf_counter() - t0, 2),
            "expected_c_dfs_pp": round(100 * float(np.mean(dfs)), 4),
            "expected_c_mem_pp": round(100 * float(np.mean(mem)), 4),
            "aa_rms_pp": locked["live_aa_sentinel_rms_pp_raw"],
            "locked": _locked_summary(locked), "a3": _a3_summary(a3)}


def _wilson(k: int, n: int, z: float = 1.959964) -> List[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, mid - half), 4), round(min(1.0, mid + half), 4)]


def aggregate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for truth in NULL_TRUTHS + ALT_TRUTHS + DESCRIPTIVE_TRUTHS:
        sub = [r for r in rows if r["truth"] == truth]
        if not sub:
            continue
        n = len(sub)
        entry: Dict[str, Any] = {
            "n_replicates": n,
            "median_aa_rms_pp": float(np.median([r["aa_rms_pp"] for r in sub])),
            "expected_c_dfs_pp_median_min_max": [float(np.median([r["expected_c_dfs_pp"] for r in sub])),
                                                 float(min(r["expected_c_dfs_pp"] for r in sub)),
                                                 float(max(r["expected_c_dfs_pp"] for r in sub))],
            "expected_c_mem_pp_median": float(np.median([r["expected_c_mem_pp"] for r in sub])),
        }
        for side in ("locked", "a3"):
            keys = [k for k, v in sub[0][side].items() if isinstance(v, bool)]
            entry[side] = {
                "selected_model_counts": {m: sum(r[side]["selected_model"] == m for r in sub)
                                          for m in sorted({r[side]["selected_model"] for r in sub})},
                "rates": {k: {"rate": round(sum(r[side][k] for r in sub) / n, 4),
                              "wilson95": _wilson(sum(r[side][k] for r in sub), n)} for k in keys},
            }
        out[truth] = entry
    return out


def evaluate_criteria(agg: Mapping[str, Any]) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    for truth in NULL_TRUTHS:
        if truth in agg:
            r = agg[truth]["a3"]["rates"]["any_nonlocal_claim"]["rate"]
            checks[f"{truth}: A-3 non-local claim rate <= {MAX_NULL_CLAIM_RATE}"] = {"rate": r, "pass": r <= MAX_NULL_CLAIM_RATE}
    for truth in ALT_TRUTHS:
        if truth in agg:
            r = agg[truth]["a3"]["rates"]["correlated_dephasing_supported"]["rate"]
            checks[f"{truth}: A-3 correlated-dephasing power >= {MIN_POWER}"] = {"rate": r, "pass": r >= MIN_POWER}
    return {"checks": checks, "all_pass": all(c["pass"] for c in checks.values())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--null-replicates", type=int, default=100)
    ap.add_argument("--alt-replicates", type=int, default=50)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=str(DOCS / "correlated_noise_01_A3_validation.json"))
    args = ap.parse_args()

    lock = json.loads((DOCS / "correlated_noise_01_live_prereg_lock.json").read_text())
    cal = json.loads((DOCS / "ibm_marrakesh_prelock_calibration.json").read_text())
    train = a3_random_split(lock["qpy_sha256"])
    code_sha256 = {rel: hashlib.sha256((REPO / rel).read_bytes()).hexdigest() for rel in CODE_FILES}
    jobs = [(t, i) for t in NULL_TRUTHS for i in range(args.null_replicates)]
    jobs += [(t, i) for t in ALT_TRUTHS + DESCRIPTIVE_TRUTHS for i in range(args.alt_replicates)]
    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(lock, cal, train)) as pool:
        rows = []
        for i, row in enumerate(pool.imap_unordered(run_one, jobs, chunksize=1), 1):
            rows.append(row)
            if i % 16 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] {time.time() - t0:.0f} s", flush=True)
    rows.sort(key=lambda r: (r["truth"], r["replicate"]))
    agg = aggregate(rows)
    payload = {
        "amendment_id": "CN01-A3",
        "status": "Simulated (synthetic counts; no hardware data)",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "lock_qpy_sha256": lock["qpy_sha256"],
        "a3_train_slots": train,
        "code_sha256": code_sha256,
        "settings": {"null_replicates": args.null_replicates, "alt_replicates": args.alt_replicates,
                     "shots_per_pub": lock["shots_per_pub"], "n_blocks": lock["n_blocks"],
                     "block_drift_sd_f": BLOCK_DRIFT_SD_F, "pub_jitter_sd_e": PUB_JITTER_SD_E,
                     "seed_rule": "numpy default_rng(int(sha256('CN01-A3:validation:<truth>:<replicate>')[:16], 16))"},
        "acceptance": evaluate_criteria(agg),
        "summary": agg,
        "replicates": rows,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps(payload["acceptance"], indent=1))
    for truth, e in agg.items():
        print(truth, "locked:", {k: v["rate"] for k, v in e["locked"]["rates"].items()},
              e["locked"]["selected_model_counts"])
        print(" " * len(truth), "A-3:   ", {k: v["rate"] for k, v in e["a3"]["rates"].items()},
              e["a3"]["selected_model_counts"])


if __name__ == "__main__":
    main()
