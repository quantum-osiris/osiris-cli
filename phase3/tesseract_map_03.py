"""TESSERACT-MAP-03: replication, readout attribution and rewrite x path interaction.

Follow-up to TESSERACT-MAP-02 (job dbto408n26qs739tnrj0, Zenodo 10.5281/zenodo.24406426).
Reuses MAP-02's frozen physical paths, so arms A1/A2/A3/A6 replay MAP-02's C5/C6/C7/C2 layouts
on the same qubits four days later.

Arms (all optimisation level 3, 40 qubits, theta = 51.7 deg):
  A1_CHAIN_BEST      chain on MAP-02 best path, no routing            (replays C5)
  A2_CHAIN_MED       chain on MAP-02 median path, no routing          (replays C6)
  A3_CHAIN_C7REPLAY  chain on MAP-02 C7 layout, no routing            (replays C7's qubits)
  A4_CHAIN_STOCK     chain, free layout, stock level-3 pipeline today (fresh C7)
  A5_STAR_MED        star pinned to MAP-02 median path, SABRE routed  (new: missing 2x2 cell)
  A6_STAR_BEST       star pinned to MAP-02 best path, SABRE routed    (replays C2)
  U0 / U1            union |0>/|1> readout calibration on every qubit any arm uses

Commands:
  python -m phase3.tesseract_map_03 build      # compile, lock, write QPY + lock (needs IBM backend access, no QPU)
  python -m phase3.tesseract_map_03 dryrun     # noiseless MPS simulation of the locked QPY + synthetic analysis check
  python -m phase3.tesseract_map_03 submit     # guarded: QPY must match lock, quota >= 1.3x estimate, never twice
  python -m phase3.tesseract_map_03 analyze    # analysis of saved raw counts
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy import stats

from phase2.tesseract_map_02.tesseract_map_02 import (
    THETA_DEFAULT_DEG,
    build_chain_circuit,
    build_star_circuit,
    chain_logical_to_physical_layout,
    extract_final_physical_layout,
    ideal_four_string_distribution,
    score_tesseract_counts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "phase3"
STEM = "tesseract_map_03"
LOCK_PATH = OUT_DIR / f"{STEM}_live_prereg_lock.json"
QPY_PATH = OUT_DIR / f"{STEM}_live_pubs.qpy"
RECEIPT_PATH = OUT_DIR / f"{STEM}_submission_receipt.json"
RAW_PATH = OUT_DIR / f"{STEM}_live_raw_counts.json"
ANALYSIS_PATH = OUT_DIR / f"{STEM}_live_analysis.json"
MAP02_LOCK = OUT_DIR / "tesseract_map_02_live_prereg_lock.json"
MAP02_LOCK_SHA = "5f4a04c45933e046c0fff0574a426fd3a876ec5f19d808d69514374c2f931f1f"

N_QUBITS = 40
N_BLOCKS = 10
SHOTS = 2048
SEED = 20261120
BACKEND = "ibm_marrakesh"
# MAP-02 was charged 53 s for 90 PUBs of the same size; scaled to this PUB count with a 5% margin.
SECONDS_PER_PUB = 53.0 / 90.0 * 1.05
QUOTA_SAFETY_FACTOR = 1.3

TEST_ARMS = ["A1_CHAIN_BEST", "A2_CHAIN_MED", "A3_CHAIN_C7REPLAY", "A4_CHAIN_STOCK", "A5_STAR_MED", "A6_STAR_BEST"]
SPAM_ARMS = ["U0_UNION_ALL0", "U1_UNION_ALL1"]

# Preregistered contrasts, in pp of F_cl. Each is a per-block linear combination of arm scores.
# (name, {arm: weight}, endpoint)  endpoint "raw" or "mit" (tensored readout-mitigated F_cl);
# "mixed" combines both and lists the endpoint per term.
CONTRASTS: List[Tuple[str, List[Tuple[str, str, float]]]] = [
    ("K1_path_gap_replication", [("raw", "A1_CHAIN_BEST", 1.0), ("raw", "A2_CHAIN_MED", -1.0)]),
    ("K2_readout_share_of_path_gap", [("raw", "A1_CHAIN_BEST", 1.0), ("raw", "A2_CHAIN_MED", -1.0),
                                      ("mit", "A1_CHAIN_BEST", -1.0), ("mit", "A2_CHAIN_MED", 1.0)]),
    ("K3_rewrite_x_path_interaction", [("raw", "A1_CHAIN_BEST", 1.0), ("raw", "A2_CHAIN_MED", -1.0),
                                       ("raw", "A6_STAR_BEST", -1.0), ("raw", "A5_STAR_MED", 1.0)]),
    ("K4_best_vs_C7_replay", [("raw", "A1_CHAIN_BEST", 1.0), ("raw", "A3_CHAIN_C7REPLAY", -1.0)]),
    ("K5_best_vs_stock_today", [("raw", "A1_CHAIN_BEST", 1.0), ("raw", "A4_CHAIN_STOCK", -1.0)]),
]


# ---------------------------------------------------------------------------
# Readout mitigation for the four target strings
# ---------------------------------------------------------------------------

def per_qubit_readout(spam0: Sequence[Mapping[str, int]], spam1: Sequence[Mapping[str, int]],
                      union: Sequence[int]) -> Dict[int, Tuple[float, float]]:
    """Pooled p(1|0), p(0|1) per physical qubit from union |0>/|1> counts (little-endian bitstrings)."""
    k = len(union)
    n10 = np.zeros(k)
    n01 = np.zeros(k)
    t0 = t1 = 0.0
    for c in spam0:
        t0 += sum(c.values())
        for bs, n in c.items():
            for i in range(k):
                if bs[k - 1 - i] == "1":
                    n10[i] += n
    for c in spam1:
        t1 += sum(c.values())
        for bs, n in c.items():
            for i in range(k):
                if bs[k - 1 - i] == "0":
                    n01[i] += n
    return {int(q): (float(n10[i] / t0), float(n01[i] / t1)) for i, q in enumerate(union)}


def mitigated_target_probs(counts: Mapping[str, int], layout: Sequence[int],
                           readout: Mapping[int, Tuple[float, float]],
                           targets: Sequence[str]) -> Dict[str, float]:
    """Tensored single-qubit readout inversion, evaluated only on `targets`.

    q(x) = sum_y p_obs(y) prod_i Minv_i[x_i, y_i], with M_i = [[1-e0, e1], [e0, 1-e1]] (rows: measured,
    columns: prepared), e0 = p(1|0), e1 = p(0|1) of the physical qubit holding logical bit i.
    Bitstrings are Qiskit little-endian: logical bit i is character n-1-i.
    """
    n = len(layout)
    minv = []
    for i in range(n):
        e0, e1 = readout[int(layout[i])]
        det = 1.0 - e0 - e1
        minv.append(np.array([[1.0 - e1, -e1], [-e0, 1.0 - e0]]) / det)
    tot = float(sum(counts.values()))
    out = {}
    for x in targets:
        xb = [int(x[n - 1 - i]) for i in range(n)]
        acc = 0.0
        for y, c in counts.items():
            prod = c / tot
            for i in range(n):
                prod *= minv[i][xb[i], int(y[n - 1 - i])]
            acc += prod
        out[x] = acc
    return out


def mitigated_fcl(counts: Mapping[str, int], layout: Sequence[int],
                  readout: Mapping[int, Tuple[float, float]]) -> float:
    ideal = ideal_four_string_distribution(N_QUBITS, THETA_DEFAULT_DEG)
    q = mitigated_target_probs(counts, layout, readout, list(ideal))
    return float(sum(math.sqrt(max(q[x], 0.0) * p) for x, p in ideal.items()) ** 2)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def holm(pvals: Mapping[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, float] = {}
    run = 0.0
    for r, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - r) * p))
        out[k] = run
    return out


def evaluate_contrasts(block_raw: Mapping[str, Sequence[float]], block_mit: Mapping[str, Sequence[float]],
                       alpha: float = 0.05) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    praw: Dict[str, float] = {}
    pw: Dict[str, float] = {}
    for name, terms in CONTRASTS:
        n = len(block_raw[terms[0][1]])
        d = np.zeros(n)
        for ep, arm, w in terms:
            src = block_raw if ep == "raw" else block_mit
            d += w * 100.0 * np.asarray(src[arm], dtype=float)
        mean = float(d.mean())
        se = float(d.std(ddof=1) / math.sqrt(n))
        tcrit = float(stats.t.ppf(0.975, n - 1))
        t = mean / se if se > 0 else float("inf")
        p = float(2 * stats.t.sf(abs(t), n - 1)) if se > 0 else 0.0
        try:
            pwil = float(stats.wilcoxon(d, method="exact").pvalue)
        except ValueError:
            pwil = 1.0
        praw[name] = p
        pw[name] = pwil
        res[name] = {"mean_pp": mean, "se_pp": se, "ci95_pp": [mean - tcrit * se, mean + tcrit * se], "t": t,
                     "df": n - 1, "p_two_sided": p, "wilcoxon_p_two_sided": pwil,
                     "block_values_pp": [round(float(x), 4) for x in d]}
    hp = holm(praw)
    hw = holm(pw)
    for k in res:
        res[k]["p_holm"] = hp[k]
        res[k]["wilcoxon_p_holm"] = hw[k]
        res[k]["significant_holm"] = hp[k] < alpha
    return res


def analyze(lock: Mapping[str, Any], pub_counts: Sequence[Mapping[str, int]], job_id: str) -> Dict[str, Any]:
    union = [int(q) for q in lock["union_active_physical_qubits"]]
    sched = lock["pub_schedule"]
    spam0 = [c for r, c in zip(sched, pub_counts) if r["arm_id"] == "U0_UNION_ALL0"]
    spam1 = [c for r, c in zip(sched, pub_counts) if r["arm_id"] == "U1_UNION_ALL1"]
    ro = per_qubit_readout(spam0, spam1, union)
    nb = int(lock["n_blocks"])
    raw = {a: [0.0] * nb for a in TEST_ARMS}
    mit = {a: [0.0] * nb for a in TEST_ARMS}
    sup = {a: [0.0] * nb for a in TEST_ARMS}
    for r, c in zip(sched, pub_counts):
        a = r["arm_id"]
        if a not in TEST_ARMS:
            continue
        b = int(r["block"])
        sc = score_tesseract_counts(c, N_QUBITS, THETA_DEFAULT_DEG)
        raw[a][b] = sc["classical_fidelity"]
        sup[a][b] = sc["four_string_support"]
        mit[a][b] = mitigated_fcl(c, lock["arm_manifests"][a]["final_physical_layout"], ro)
    arms = {}
    for a in TEST_ARMS:
        lay = lock["arm_manifests"][a]["final_physical_layout"]
        errs = [0.5 * sum(ro[q]) for q in lay]
        arms[a] = {
            "twoq_gates": lock["arm_manifests"][a]["twoq_gate_count"],
            "depth": lock["arm_manifests"][a]["circuit_depth"],
            "mean_fcl_raw": float(np.mean(raw[a])), "sd_fcl_raw": float(np.std(raw[a], ddof=1)),
            "mean_fcl_mit": float(np.mean(mit[a])), "sd_fcl_mit": float(np.std(mit[a], ddof=1)),
            "mean_support_raw": float(np.mean(sup[a])),
            "mean_layout_readout_err": float(np.mean(errs)), "max_layout_readout_err": float(np.max(errs)),
            "worst_layout_qubit": int(lay[int(np.argmax(errs))]),
            "block_fcl_raw": [round(x, 6) for x in raw[a]], "block_fcl_mit": [round(x, 6) for x in mit[a]],
        }
    con = evaluate_contrasts(raw, mit)
    k1 = con["K1_path_gap_replication"]["mean_pp"]
    k2 = con["K2_readout_share_of_path_gap"]["mean_pp"]
    # eta: share of the raw path gap removed by readout mitigation, with a block bootstrap CI
    rng = np.random.default_rng(SEED)
    g_raw = 100 * (np.array(raw["A1_CHAIN_BEST"]) - np.array(raw["A2_CHAIN_MED"]))
    g_mit = 100 * (np.array(mit["A1_CHAIN_BEST"]) - np.array(mit["A2_CHAIN_MED"]))
    boots = []
    for _ in range(10000):
        idx = rng.integers(0, nb, nb)
        gr = g_raw[idx].mean()
        if abs(gr) > 1e-9:
            boots.append(1.0 - g_mit[idx].mean() / gr)
    eta = 1.0 - (k1 - k2) / k1 if abs(k1) > 1e-9 else float("nan")
    return {
        "experiment_id": "TESSERACT-MAP-03",
        "job_id": job_id,
        "backend": lock["backend_name"],
        "prereg_lock_sha256": lock.get("lock_sha256"),
        "qpy_sha256": lock["qpy_sha256"],
        "n_blocks": nb, "shots_per_pub": lock["shots_per_pub"],
        "union_mean_readout_err": float(np.mean([0.5 * (a + b) for a, b in ro.values()])),
        "per_qubit_readout": {str(q): {"p1_given_0": round(a, 5), "p0_given_1": round(b, 5)} for q, (a, b) in ro.items()},
        "arms": arms,
        "contrasts": con,
        "eta_readout_share": {"estimate": eta, "bootstrap95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                              "note": "Descriptive: 1 - (mitigated path gap / raw path gap). Not a preregistered test."},
    }


# ---------------------------------------------------------------------------
# Build and lock
# ---------------------------------------------------------------------------

def _service():
    import os
    from qiskit_ibm_runtime import QiskitRuntimeService
    tok = os.environ.get("QISKIT_IBM_TOKEN") or os.environ.get("IBM_QUANTUM_TOKEN")
    if tok:
        return QiskitRuntimeService(channel="ibm_quantum_platform", token=tok)
    return QiskitRuntimeService()


def _digest(qc) -> str:
    rep = "|".join(f"{i.operation.name}:{[qc.find_bit(q).index for q in i.qubits]}:{[float(p) for p in i.operation.params]}" for i in qc.data)
    return hashlib.sha256(rep.encode()).hexdigest()


def build(backend: Any) -> Dict[str, Any]:
    from qiskit import QuantumCircuit, qpy, transpile
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from phase3.live_hardware_runner import snapshot_backend_calibration, verify_isa_circuit

    if LOCK_PATH.exists() or QPY_PATH.exists():
        raise SystemExit("MAP-03 lock already exists; refusing to overwrite")
    if hashlib.sha256(MAP02_LOCK.read_bytes()).hexdigest() != MAP02_LOCK_SHA:
        raise SystemExit("MAP-02 lock does not match its published hash")
    m2 = json.loads(MAP02_LOCK.read_text())
    best = chain_logical_to_physical_layout(m2["path_summary"]["best_path"])
    med = chain_logical_to_physical_layout(m2["path_summary"]["median_path"])
    c7 = m2["arm_manifests"]["C7_CHAIN_FREE_OPT3"]["final_physical_layout"]
    assert best == m2["arm_manifests"]["C5_CHAIN_BESTPATH_OPT3"]["final_physical_layout"]
    assert med == m2["arm_manifests"]["C6_CHAIN_MEDPATH_OPT3"]["final_physical_layout"]

    chain = lambda nm: build_chain_circuit(N_QUBITS, THETA_DEFAULT_DEG, name=nm)
    star = lambda nm: build_star_circuit(N_QUBITS, THETA_DEFAULT_DEG, include_inert_rz=False, name=nm)

    def pinned(qc, layout, routing):
        pm = generate_preset_pass_manager(optimization_level=3, backend=backend, initial_layout=list(layout),
                                          layout_method="trivial", routing_method=routing, seed_transpiler=SEED)
        return pm.run(qc)

    specs = {
        "A1_CHAIN_BEST": ("chain", "map02_best_path", lambda: pinned(chain("A1"), best, "none")),
        "A2_CHAIN_MED": ("chain", "map02_median_path", lambda: pinned(chain("A2"), med, "none")),
        "A3_CHAIN_C7REPLAY": ("chain", "map02_c7_layout", lambda: pinned(chain("A3"), c7, "none")),
        "A4_CHAIN_STOCK": ("chain", "stock_level3_today", lambda: transpile(chain("A4"), backend=backend, optimization_level=3, seed_transpiler=SEED)),
        "A5_STAR_MED": ("star", "map02_median_path", lambda: pinned(star("A5"), med, "sabre")),
        "A6_STAR_BEST": ("star", "map02_best_path", lambda: pinned(star("A6"), best, "sabre")),
    }
    arms: Dict[str, Any] = {}
    manifests: Dict[str, Any] = {}
    for arm, (logical, policy, make) in specs.items():
        t = make()
        verify_isa_circuit(t, backend)
        arms[arm] = t
        manifests[arm] = {
            "logical_structure": logical, "layout_policy": policy, "optimization_level": 3,
            "twoq_gate_count": sum(1 for i in t.data if i.operation.num_qubits == 2),
            "circuit_depth": t.depth(), "ops_breakdown": {k: int(v) for k, v in t.count_ops().items()},
            "final_physical_layout": extract_final_physical_layout(t, N_QUBITS), "circuit_sha256": _digest(t),
        }
    for a in ("A1_CHAIN_BEST", "A2_CHAIN_MED", "A3_CHAIN_C7REPLAY"):
        assert manifests[a]["twoq_gate_count"] == 39, (a, manifests[a]["twoq_gate_count"])
    assert manifests["A1_CHAIN_BEST"]["final_physical_layout"] == best
    assert manifests["A2_CHAIN_MED"]["final_physical_layout"] == med
    assert manifests["A3_CHAIN_C7REPLAY"]["final_physical_layout"] == list(c7)

    union = sorted({q for m in manifests.values() for q in m["final_physical_layout"]})
    k = len(union)
    u0 = QuantumCircuit(k, k, name="U0")
    u0.measure(range(k), range(k))
    u1 = QuantumCircuit(k, k, name="U1")
    u1.x(range(k))
    u1.measure(range(k), range(k))
    arms["U0_UNION_ALL0"] = transpile(u0, backend=backend, initial_layout=union, optimization_level=0, seed_transpiler=SEED)
    arms["U1_UNION_ALL1"] = transpile(u1, backend=backend, initial_layout=union, optimization_level=1, seed_transpiler=SEED)
    for a in SPAM_ARMS:
        verify_isa_circuit(arms[a], backend)

    rng = np.random.default_rng(SEED)
    keys = TEST_ARMS + SPAM_ARMS
    pubs, sched = [], []
    for b in range(N_BLOCKS):
        order = list(keys)
        rng.shuffle(order)
        for pos, a in enumerate(order):
            sched.append({"pub_idx": len(pubs), "block": b, "position_in_block": pos, "arm_id": a})
            pubs.append(arms[a].copy(name=f"tmap03_b{b:02d}_p{pos}_{a}"))
    buf = io.BytesIO()
    qpy.dump(pubs, buf)
    QPY_PATH.write_bytes(buf.getvalue())
    cal = snapshot_backend_calibration(backend)
    (OUT_DIR / f"{STEM}_prelock_calibration.json").write_text(json.dumps(cal, indent=1) + "\n")
    lock = {
        "experiment_id": "TESSERACT-MAP-03",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "backend_name": backend.name, "n_blocks": N_BLOCKS, "shots_per_pub": SHOTS,
        "total_pubs": len(pubs), "total_shots": len(pubs) * SHOTS,
        "estimated_qpu_seconds": round(SECONDS_PER_PUB * len(pubs), 1),
        "qpy_file": QPY_PATH.name, "qpy_sha256": hashlib.sha256(QPY_PATH.read_bytes()).hexdigest(),
        "prelock_calibration_sha256": hashlib.sha256((OUT_DIR / f"{STEM}_prelock_calibration.json").read_bytes()).hexdigest(),
        "map02_lock_sha256": MAP02_LOCK_SHA, "theta_deg": THETA_DEFAULT_DEG, "seed": SEED,
        "union_active_physical_qubits": union,
        "arm_manifests": manifests,
        "contrasts": [{"name": n, "terms": [{"endpoint": e, "arm": a, "weight": w} for e, a, w in t]} for n, t in CONTRASTS],
        "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pub_schedule": sched,
    }
    LOCK_PATH.write_text(json.dumps(lock, indent=1) + "\n")
    return lock


# ---------------------------------------------------------------------------
# Dry run: noiseless MPS check of the locked circuits + synthetic analysis check
# ---------------------------------------------------------------------------

def dryrun() -> Dict[str, Any]:
    from qiskit import qpy
    from qiskit_aer import AerSimulator

    lock = json.loads(LOCK_PATH.read_text())
    if hashlib.sha256(QPY_PATH.read_bytes()).hexdigest() != lock["qpy_sha256"]:
        raise SystemExit("QPY does not match lock")
    with QPY_PATH.open("rb") as f:
        pubs = qpy.load(f)
    sim = AerSimulator(method="matrix_product_state")
    ideal = ideal_four_string_distribution(N_QUBITS, THETA_DEFAULT_DEG)
    out: Dict[str, Any] = {"noiseless_fcl_block0": {}}
    for r, qc in zip(lock["pub_schedule"], pubs):
        if r["block"] != 0:
            continue
        # strip unused idle qubits for the simulator by simulating the full-width ISA circuit
        counts = sim.run(qc, shots=2048, seed_simulator=SEED).result().get_counts()
        counts = {k.replace(" ", ""): v for k, v in counts.items()}
        if r["arm_id"] in TEST_ARMS:
            out["noiseless_fcl_block0"][r["arm_id"]] = score_tesseract_counts(counts, N_QUBITS, THETA_DEFAULT_DEG)["classical_fidelity"]
        else:
            k = len(lock["union_active_physical_qubits"])
            want = ("0" if r["arm_id"] == "U0_UNION_ALL0" else "1") * k
            out.setdefault("spam_ok", {})[r["arm_id"]] = counts.get(want, 0) == 2048
    # synthetic: ideal samples with injected readout flips from MAP-02's measured rates
    m2a = json.loads((OUT_DIR / "tesseract_map_02_live_analysis.json").read_text())["per_qubit_spam_calibration"]
    rates = {int(q): (v["p1_given_0"], v["p0_given_1"]) for q, v in m2a.items()}
    rng = np.random.default_rng(SEED)
    strings = list(ideal)
    probs = np.array([ideal[s] for s in strings])
    synth: Dict[str, Any] = {}
    for arm in ("A1_CHAIN_BEST", "A2_CHAIN_MED"):
        lay = lock["arm_manifests"][arm]["final_physical_layout"]
        rr = {q: rates.get(q, (0.01, 0.02)) for q in lay}
        cnt: Dict[str, int] = {}
        for s in rng.choice(strings, size=20000, p=probs):
            bits = list(s)
            for i in range(N_QUBITS):
                q = lay[i]
                pos = N_QUBITS - 1 - i
                e0, e1 = rr[q]
                if bits[pos] == "0" and rng.random() < e0:
                    bits[pos] = "1"
                elif bits[pos] == "1" and rng.random() < e1:
                    bits[pos] = "0"
            y = "".join(bits)
            cnt[y] = cnt.get(y, 0) + 1
        synth[arm] = {"raw_fcl": score_tesseract_counts(cnt, N_QUBITS, THETA_DEFAULT_DEG)["classical_fidelity"],
                      "mitigated_fcl": mitigated_fcl(cnt, lay, rr)}
    out["synthetic_readout_only"] = synth
    out["checks"] = {
        "all_arms_noiseless_fcl_above_0.95": all(v > 0.95 for v in out["noiseless_fcl_block0"].values()),
        "spam_circuits_exact": all(out.get("spam_ok", {}).values()),
        "mitigation_recovers_readout_only_fcl_above_0.97": all(v["mitigated_fcl"] > 0.97 for v in synth.values()),
    }
    (OUT_DIR / f"{STEM}_dryrun.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


# ---------------------------------------------------------------------------
# Submit / analyze
# ---------------------------------------------------------------------------

def submit() -> None:
    from qiskit import qpy
    from qiskit_ibm_runtime import SamplerV2
    from phase3.execute_and_analyze_live import _job_usage_seconds, _public_usage
    from phase3.live_hardware_runner import extract_counts_from_pub_result, snapshot_backend_calibration

    lock = json.loads(LOCK_PATH.read_text())
    lock_sha = hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest()
    svc = _service()
    backend = svc.backend(lock["backend_name"])
    if RAW_PATH.exists():
        print("raw counts already saved; nothing to submit")
        return
    if RECEIPT_PATH.exists():
        jid = json.loads(RECEIPT_PATH.read_text())["job_id"]
        job = svc.job(jid)
        print("resuming", jid)
    else:
        if hashlib.sha256(QPY_PATH.read_bytes()).hexdigest() != lock["qpy_sha256"]:
            raise SystemExit("QPY does not match lock; not submitting")
        remaining = float(svc.usage().get("usage_remaining_seconds", 0) or 0)
        need = QUOTA_SAFETY_FACTOR * float(lock["estimated_qpu_seconds"])
        if remaining < need:
            raise SystemExit(f"{remaining:.0f} s quota left, need {need:.0f} s; not submitting")
        with QPY_PATH.open("rb") as f:
            pubs = qpy.load(f)
        job = SamplerV2(mode=backend).run(pubs, shots=int(lock["shots_per_pub"]))
        jid = job.job_id()
        RECEIPT_PATH.write_text(json.dumps({
            "experiment_id": "TESSERACT-MAP-03", "job_id": jid, "backend": backend.name,
            "submitted_utc": datetime.now(timezone.utc).isoformat(), "prereg_lock_sha256": lock_sha,
            "qpy_sha256": lock["qpy_sha256"], "quota_remaining_before_s": remaining}, indent=2) + "\n")
        print("submitted", jid, flush=True)
    res = job.result()
    usage = _job_usage_seconds(job)
    RAW_PATH.write_text(json.dumps({
        "job_id": jid, "completed_utc": datetime.now(timezone.utc).isoformat(), "job_usage_seconds": usage,
        "estimated_qpu_seconds": lock["estimated_qpu_seconds"], "post_job_usage": _public_usage(svc.usage()),
        "execution_time_calibration": snapshot_backend_calibration(backend),
        "pub_counts": [extract_counts_from_pub_result(res[i]) for i in range(len(lock["pub_schedule"]))],
    }) + "\n")
    print("done", jid, usage, "s")


def run_analysis() -> Dict[str, Any]:
    lock = json.loads(LOCK_PATH.read_text())
    lock["lock_sha256"] = hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest()
    raw = json.loads(RAW_PATH.read_text())
    out = analyze(lock, raw["pub_counts"], raw["job_id"])
    out["job_usage_seconds"] = raw.get("job_usage_seconds")
    ANALYSIS_PATH.write_text(json.dumps(out, indent=1) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "dryrun", "submit", "analyze"])
    a = ap.parse_args()
    if a.step == "build":
        lock = build(_service().backend(BACKEND))
        print(json.dumps({k: lock[k] for k in ("total_pubs", "estimated_qpu_seconds", "qpy_sha256")}),
              {k: (v["twoq_gate_count"], v["circuit_depth"]) for k, v in lock["arm_manifests"].items()})
    elif a.step == "dryrun":
        print(json.dumps(dryrun(), indent=1))
    elif a.step == "submit":
        submit()
    else:
        r = run_analysis()
        print(json.dumps({k: {kk: r["contrasts"][k][kk] for kk in ("mean_pp", "ci95_pp", "p_holm")} for k in r["contrasts"]}, indent=1))


if __name__ == "__main__":
    main()
