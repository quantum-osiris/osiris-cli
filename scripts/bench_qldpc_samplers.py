"""Run the frozen sampler comparison (docs/qldpc/SAMPLER_PROTOCOL.md). Read-only on the challenge repo."""
import collections
import json
import os
import random
import sys
import time
sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "/home/enki/github-backup/qldpc-challenge"
sys.path[:0] = [ROOT, REPO + "/research/kit", REPO + "/verify"]
import bb, css, surrogate  # noqa: E401
from osiris_cli import qldpc_samplers as qs, qldpc_adapter as qa
OUT = os.path.expanduser("~/research-archive/qldpc/sampler-benchmark-20261109")
os.makedirs(OUT, exist_ok=True)
ARCH = os.path.expanduser("~/research-archive/qldpc/campaigns/osiris-bb-w6-unrestricted-n60-126-20261109/run/evidence.jsonl")
archived = set()
for l in open(ARCH):
    r = json.loads(l)
    if r["kind"] == "SCREEN":
        p = r["payload"]["params"]
        archived.add(qs.canonical(p["l"], p["m"], [tuple(t) for t in p["A"]], [tuple(t) for t in p["B"]]))
SEEDS = {"A_uniform": [1101, 1102, 1103, 1104, 1105], "B_rank_aware": [1201, 1202, 1203, 1204, 1205],
         "C_structure_guided": [1301, 1302, 1303, 1304, 1305]}
ev = qa.Evidence(OUT + "/evidence.jsonl")
ev.append("BENCH_START", {"protocol": "docs/qldpc/SAMPLER_PROTOCOL.md", "seeds": SEEDS, "environment": qa.environment(),
                          "archived_structures": len(archived)})
t0 = time.process_time(); qs.StructureSampler._zd  # noqa
pre = {}
for cls in qs.SAMPLERS:
    if cls is qs.StructureSampler:
        t = time.process_time(); s = cls(0); pre[cls.name] = round(time.process_time() - t, 2)
    else:
        pre[cls.name] = 0.0
ev.append("PRECOMPUTE", {"cpu_seconds": pre, "note": "C's zero-divisor tables, built once"})
for cls in qs.SAMPLERS:
    for seed in SEEDS[cls.name]:
        if (qa.mem_available_mb() or 9999) < 1500:
            raise SystemExit("RAM guard")
        smp = cls(seed)
        t = time.process_time()
        cands = []
        for _ in range(400):
            cands.append(smp.next())
        gen_cpu = time.process_time() - t
        keys = [qs.canonical(*c) for c in cands]
        ks = [qs.k_of(*c) for c in cands]
        uniq = set(keys)
        kpos_unique = {k_ for k_, kk in zip(keys, ks) if kk > 0}
        rng = random.Random(seed)
        valid = 0; chk = rng.sample(cands, 20)
        for ell, m, A, Bt in chk:
            HX, HZ = bb.build_bb(ell, m, A, Bt)
            valid += int(css.verify_css(HX, HZ) and int(max(HX.sum(1).max(), HZ.sum(1).max())) == 6 and 60 <= 2*ell*m <= 126)
        sub = [c for c, kk in zip(cands, ks) if kk > 0][:40]
        ubs = []; t2 = time.process_time()
        for ell, m, A, Bt in sub:
            HX, HZ = bb.build_bb(ell, m, A, Bt)
            ubs.append(min(int(surrogate.distance_rand(HX, HZ, trials=200, seed=s)) for s in (101, 211, 307)))
        scr_cpu = time.process_time() - t2
        ev.append("REPLICATE", {
            "sampler": cls.name, "seed": seed, "candidates": 400, "proposals": smp.proposals,
            "frac_k_pos_among_candidates": sum(1 for x in ks if x > 0) / 400,
            "frac_k_pos_among_proposals": sum(1 for x in ks if x > 0) / smp.proposals,
            "k_distribution": dict(sorted(collections.Counter(ks).items())),
            "unique_structures": len(uniq), "within_replicate_duplicate_rate": 1 - len(uniq) / 400,
            "unique_k_pos": len(kpos_unique), "unique_k_pos_in_archive": len(kpos_unique & archived),
            "valid_construction_checked": [valid, 20], "generation_cpu_s": round(gen_cpu, 3),
            "cpu_s_per_unique_k_pos": round(gen_cpu / max(1, len(kpos_unique)), 4),
            "surrogate_screened": len(ubs), "surrogate_min_ub": ubs, "surrogate_cpu_s": round(scr_cpu, 2),
            "n_ge6": sum(1 for u in ubs if u >= 6)})
ev.append("BENCH_END", {"wall_cpu_s": round(time.process_time() - t0, 1)})
print("done", OUT)
