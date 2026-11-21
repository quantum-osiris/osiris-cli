"""TESSERACT-RO-07: are the readout faults IBM's calibration misses context-dependent?

Follows TESSERACT-RO-06. RO-06 found qubits whose all-qubit probe readout error is far above IBM's reported
value. RO-07 asks whether such a qubit's error depends on its measurement context:
  (a) what else is measured at the same time (multiplexed readout of all qubits vs only the targets), and
  (b) the state of every other qubit (all others |0> vs all others |1>).
Strong context dependence would explain why per-qubit calibration and all-qubit probes disagree.

Selection (rule fixed before data), from RO-06's LAST completed round on each backend (no qubit may be
coupled to another selected qubit; greedy in the stated order):
  DISAGREE  up to 3 qubits with the largest (e_probe - e_ibm), requiring e_probe > 0.05 and
            e_probe - e_ibm > 0.03: readout worse than IBM reports
  AGREE     up to 2 qubits with e_probe > 0.15 and |e_probe - e_ibm| < 0.03: bad readout that IBM reports
  CONTROLS  the 2 lowest-e_probe qubits
'Targets' = DISAGREE + AGREE.

Per backend, one job: 6 circuit types x REPEATS interleaved, 2,048 shots.
  ISO0 / ISO1     targets+controls prepared 0 / 1; only they are measured
  ALL0 / ALL1     every qubit prepared 0 / 1; every qubit measured (the RO-06 probe context)
  FLIP0 / FLIP1   targets+controls prepared 0 / 1, every OTHER qubit prepared 1 / 0; every qubit measured

For each measured qubit q and prepared bit b, per repeat r: e_b = P(read != b).
Per qubit: e = mean of e_0 and e_1 in each context.
  multiplexing effect  D_mux  = e(ALL)  - e(ISO)
  neighbour effect     D_nbr  = e(FLIP) - e(ALL)
Each has a paired t over REPEATS (df = REPEATS - 1) and a 95% CI.
Preregistered criteria:
  R1 (multiplexing matters)    on >= 2 devices, at least half the DISAGREE qubits have |D_mux| > 0.05 with the
                                CI excluding 0
  R2 (neighbour state matters)  same criterion for D_nbr
  R2b (mechanism)               DISAGREE qubits show larger |D_mux| + |D_nbr| than AGREE qubits (pooled; reported)
  R3 (sanity)                  every control has |D_mux| < 0.01 and |D_nbr| < 0.01
  R4 (reported)                D_mux and D_nbr for every target, with their p(1|0) and p(0|1) parts
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from phase3 import readout_drift_06 as ro6
from phase3.tesseract_map_04 import SECONDS_PER_PUB, _service

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "phase7"
BACKENDS = ["ibm_marrakesh", "ibm_fez", "ibm_kingston"]
REPEATS = 6
SHOTS = 2048
N_TARGETS = 4
N_CONTROLS = 2
SEED = 20261121
TYPES = ["ISO0", "ISO1", "ALL0", "ALL1", "FLIP0", "FLIP1"]


def f(b: str, what: str) -> Path:
    return OUT_DIR / f"ro07_{b}_{what}"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def select(b: str, edges) -> Dict[str, Any]:
    last = max(r for r in range(ro6.ROUNDS) if ro6.f(b, r, "counts").exists())
    ep, _er, ei, t, jid = ro6._round_errors(b, last)
    adj = {q: set() for q in ep}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    chosen: List[int] = []

    def free(q):
        return q not in chosen and all(q not in adj[c] for c in chosen)
    disagree = []
    for q in sorted(ep, key=lambda q: -(ep[q] - (ei[q] if ei[q] is not None else 1.0))):
        if len(disagree) >= 3:
            break
        if ei[q] is not None and ep[q] > 0.05 and ep[q] - ei[q] > 0.03 and free(q):
            disagree.append(q)
            chosen.append(q)
    agree = []
    for q in sorted(ep, key=lambda q: -ep[q]):
        if len(agree) >= 2:
            break
        if ei[q] is not None and ep[q] > 0.15 and abs(ep[q] - ei[q]) < 0.03 and free(q):
            agree.append(q)
            chosen.append(q)
    targets = disagree + agree
    controls: List[int] = []
    for q in sorted(ep, key=lambda q: ep[q]):
        if len(controls) >= N_CONTROLS:
            break
        if free(q):
            controls.append(q)
            chosen.append(q)
    return {"ro06_round": last, "ro06_job_id": jid, "ro06_time": t, "targets": targets, "controls": controls,
            "disagree": disagree, "agree": agree,
            "e_probe": {str(q): ep[q] for q in targets + controls}, "e_ibm": {str(q): ei[q] for q in targets + controls}}


def build_and_run(b: str) -> Dict[str, Any]:
    from qiskit import QuantumCircuit, qpy, transpile
    from qiskit_ibm_runtime import SamplerV2
    from phase3.live_hardware_runner import extract_counts_from_pub_result, snapshot_backend_calibration

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_p = f(b, "raw_counts.json")
    if raw_p.exists():
        return {"skipped": "already run"}
    svc = _service()
    be = svc.backend(b)
    n = be.num_qubits
    edges = [tuple(e) for e in be.coupling_map.get_edges()]
    lock_p = f(b, "lock.json")
    if lock_p.exists():
        L = json.loads(lock_p.read_text())
    else:
        sel = select(b, edges)
        S = sel["targets"] + sel["controls"]
        others = [q for q in range(n) if q not in S]

        def circ(kind: str):
            if kind.startswith("ISO"):
                qc = QuantumCircuit(n, len(S), name=f"ro07_{kind}")
                if kind == "ISO1":
                    qc.x(S)
                qc.measure(S, list(range(len(S))))
            else:
                qc = QuantumCircuit(n, n, name=f"ro07_{kind}")
                ones = {"ALL0": [], "ALL1": list(range(n)), "FLIP0": others, "FLIP1": S}[kind]
                if ones:
                    qc.x(ones)
                qc.measure(list(range(n)), list(range(n)))
            return transpile(qc, backend=be, initial_layout=list(range(n)), optimization_level=0)

        base = {k: circ(k) for k in TYPES}
        rng = np.random.default_rng(SEED)
        sched, pubs = [], []
        for r in range(REPEATS):
            order = list(TYPES)
            rng.shuffle(order)
            for k in order:
                sched.append({"repeat": r, "type": k})
                pubs.append(base[k].copy(name=f"ro07_r{r}_{k}"))
        buf = io.BytesIO()
        qpy.dump(pubs, buf)
        f(b, "pubs.qpy").write_bytes(buf.getvalue())
        cal_p = f(b, "calibration.json")
        cal_p.write_text(json.dumps(snapshot_backend_calibration(be), indent=1) + "\n")
        L = {"experiment_id": "TESSERACT-RO-07", "backend": b, "locked_utc": datetime.now(timezone.utc).isoformat(),
             "selection": sel, "measured_set": S, "schedule": sched, "repeats": REPEATS, "shots": SHOTS,
             "qpy_sha256": _sha(f(b, "pubs.qpy")), "calibration_sha256": _sha(cal_p), "code_sha256": _sha(Path(__file__)),
             "estimated_qpu_seconds": round(SECONDS_PER_PUB * len(pubs), 1)}
        lock_p.write_text(json.dumps(L, indent=1) + "\n")
    with f(b, "pubs.qpy").open("rb") as fh:
        pubs = qpy.load(fh)
    if _sha(f(b, "pubs.qpy")) != L["qpy_sha256"]:
        raise SystemExit("QPY mismatch")
    rec_p = f(b, "receipt.json")
    if rec_p.exists():
        job = svc.job(json.loads(rec_p.read_text())["job_id"])
    else:
        rem = float(svc.usage().get("usage_remaining_seconds", 0) or 0)
        if rem < 1.3 * L["estimated_qpu_seconds"]:
            raise SystemExit(f"{rem:.0f} s quota left; not submitting")
        job = SamplerV2(mode=be).run(pubs, shots=SHOTS)
        rec_p.write_text(json.dumps({"job_id": job.job_id(), "submitted_utc": datetime.now(timezone.utc).isoformat(),
                                     "lock_sha256": _sha(lock_p), "quota_remaining_before_s": rem}, indent=1) + "\n")
    res = job.result()
    ts = {}
    try:
        ts = {k: str(v) for k, v in (job.metrics().get("timestamps") or {}).items()}
    except Exception:
        pass
    usage = None
    try:
        usage = job.usage()
    except Exception:
        pass
    raw_p.write_text(json.dumps({"job_id": job.job_id(), "job_timestamps": ts, "job_usage_seconds": usage,
                                 "pub_counts": [extract_counts_from_pub_result(res[i]) for i in range(len(pubs))]}) + "\n")
    return {"job_id": job.job_id(), "usage_s": usage, "targets": L["selection"]["targets"], "controls": L["selection"]["controls"]}


def _p_read(counts: Dict[str, int], clbit: int, bit: int) -> float:
    tot = sum(counts.values())
    wrong = sum(v for s, v in counts.items() if s.replace(" ", "")[-1 - clbit] != str(bit))
    return wrong / tot


def analyze() -> Dict[str, Any]:
    from scipy import stats
    out: Dict[str, Any] = {"experiment_id": "TESSERACT-RO-07", "devices": {}}
    tally = {"R1": 0, "R2": 0}
    r3_ok = True
    for b in BACKENDS:
        if not f(b, "raw_counts.json").exists():
            continue
        L = json.loads(f(b, "lock.json").read_text())
        raw = json.loads(f(b, "raw_counts.json").read_text())
        S = L["measured_set"]
        e = {q: {ctx: [[None, None] for _ in range(REPEATS)] for ctx in ("ISO", "ALL", "FLIP")} for q in S}
        for row, c in zip(L["schedule"], raw["pub_counts"]):
            ctx, bit = row["type"][:-1], int(row["type"][-1])
            for i, q in enumerate(S):
                clbit = i if ctx == "ISO" else q
                e[q][ctx][row["repeat"]][bit] = _p_read(c, clbit, bit)
        per = {}
        for q in S:
            m = {ctx: np.array([0.5 * (a + c) for a, c in e[q][ctx]]) for ctx in e[q]}
            res = {}
            for name, x, y in (("D_mux", m["ALL"], m["ISO"]), ("D_nbr", m["FLIP"], m["ALL"])):
                d = x - y
                se = d.std(ddof=1) / np.sqrt(len(d))
                tcrit = stats.t.ppf(0.975, len(d) - 1)
                p = float(stats.ttest_1samp(d, 0.0).pvalue) if se > 0 else (0.0 if d.mean() != 0 else 1.0)
                res[name] = {"mean": float(d.mean()), "ci95": [float(d.mean() - tcrit * se), float(d.mean() + tcrit * se)], "p": p}
            res["e"] = {ctx: float(m[ctx].mean()) for ctx in m}
            res["p10_p01"] = {ctx: [float(np.mean([a for a, _ in e[q][ctx]])), float(np.mean([c for _, c in e[q][ctx]]))] for ctx in e[q]}
            res["role"] = "disagree" if q in L["selection"]["disagree"] else ("agree" if q in L["selection"]["agree"] else "control")
            per[str(q)] = res
        tg = [str(q) for q in L["selection"]["disagree"]]

        def sig(r, k):
            lo, hi = r[k]["ci95"]
            return abs(r[k]["mean"]) > 0.05 and (lo > 0 or hi < 0)
        r1 = sum(sig(per[q], "D_mux") for q in tg) >= len(tg) / 2 if tg else False
        r2 = sum(sig(per[q], "D_nbr") for q in tg) >= len(tg) / 2 if tg else False
        tally["R1"] += int(r1)
        tally["R2"] += int(r2)
        for q in L["selection"]["controls"]:
            r = per[str(q)]
            r3_ok &= abs(r["D_mux"]["mean"]) < 0.01 and abs(r["D_nbr"]["mean"]) < 0.01
        out["devices"][b] = {"job_id": raw["job_id"], "selection": L["selection"], "per_qubit": per,
                             "R1_majority_of_targets": r1, "R2_majority_of_targets": r2}
    out["R1_supported"] = tally["R1"] >= 2
    out["R2_supported"] = tally["R2"] >= 2
    out["R3_controls_clean"] = bool(r3_ok)
    out["device_tally"] = tally
    (OUT_DIR / "ro07_analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "analyze"])
    ap.add_argument("backend", nargs="?")
    a = ap.parse_args()
    if a.step == "run":
        for b in ([a.backend] if a.backend else BACKENDS):
            print(b, json.dumps(build_and_run(b), default=str), flush=True)
    else:
        r = analyze()
        print(json.dumps({k: v for k, v in r.items() if k != "devices"}, indent=1))
        for b, d in r["devices"].items():
            for q, v in d["per_qubit"].items():
                print(b, q, v["role"], {k: round(x, 4) for k, x in v["e"].items()},
                      "mux", round(v["D_mux"]["mean"], 4), "nbr", round(v["D_nbr"]["mean"], 4))


if __name__ == "__main__":
    main()
