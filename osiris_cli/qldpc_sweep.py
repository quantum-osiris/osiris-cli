"""Bounded, resumable, sequential sweep for an externally defined qLDPC campaign.

Builds on ``qldpc_adapter`` (repo pinning, the official gate subprocess, hash-chained evidence).
What it adds: (1) loading a campaign defined OUTSIDE the challenge repo by materialising it into a
throw-away ``git clone --local`` of that repo, validated by the repo's own ``load_campaign`` with the
repo's own schema; (2) a deterministic candidate sampler; (3) a three-seed surrogate screen used for
ranking only; (4) a gate stage with the user's stopping rules; (5) resource guards.

Everything is derived from the evidence chain, so a stopped run resumes without redoing work and
without re-running (or re-seeding) any official evaluation.
"""

from __future__ import annotations

import calendar
import copy
import json
import os
import resource
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from osiris_cli import qldpc_adapter as qa

SAMPLER_SEED = 101                  # documented in EXECUTION_INTERPRETATION; not a validation seed
MAX_SCREENED = 400
MAX_GATE = 12
CONSEC_NO_PROGRESS = 8
MAX_ACCEPTED = 3
WALL_BUDGET_S = 6 * 3600
MIN_FREE_RAM_MB = 1500
MIN_FREE_DISK_MB = 2048
MAX_DISK_USED_FRAC = 0.95


class SweepStop(Exception):
    pass


# ── campaign loading ─────────────────────────────────────────────────────────

_NOTE_KEYS = ("validation", "prior", "forbidden")


def materialize_campaign(proposal: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Return (schema-valid campaign object, list of every transformation made).

    Parameters are never changed. Fields the repo's schema has no slot for are carried verbatim in
    ``notes``; the only substantive edit is status -> active, which is the owner's recorded approval."""
    src = proposal["campaign"]
    c = copy.deepcopy(src)
    changes: List[str] = []
    carried: Dict[str, Any] = {}
    if c.get("status") != "active":
        changes.append("status: %r -> 'active' (owner approval given in conversation)" % c.get("status"))
        c["status"] = "active"
    for k in _NOTE_KEYS:
        if k in c:
            carried[k] = c.pop(k)
            changes.append("moved %r into notes (schema has no such field)" % k)
    obj = c.get("objective", {})
    if obj.get("metric") not in ("kd2_over_n", "geometric_efficiency", "distance", "logical_qubits",
                                 "frontier_entries"):
        carried["objective_original"] = copy.deepcopy(obj)
        c["objective"] = {"metric": "frontier_entries", "direction": "maximize"}
        changes.append("objective replaced by schema-valid frontier_entries/maximize; original in notes")
    cons = c.get("constraints", {})
    if "css" in cons:
        carried["constraints_css"] = cons.pop("css")
        changes.append("moved constraints.css into notes")
    meth = c.get("methods", {})
    scr = meth.get("screening", {})
    if "use" in scr:
        carried["screening_use"] = scr.pop("use")
        changes.append("moved methods.screening.use into notes")
    if "search_space" in meth:
        carried["search_space"] = meth.pop("search_space")
        changes.append("moved methods.search_space into notes")
    bud = c.get("budget", {})
    for k in [k for k in bud if k not in ("cpu_hours", "gpu_hours", "walltime_hours", "candidates_screened")]:
        carried["budget_" + k] = bud.pop(k)
        changes.append("moved budget.%s into notes (enforced by the runner)" % k)
    c["notes"] = ("Materialised from the frozen proposal docs/qldpc/PROPOSED_bb-w6-unrestricted-n60-126.json. "
                  "Fields with no schema slot, carried verbatim: " + json.dumps(carried, sort_keys=True))
    return {"campaign": c}, changes


def make_checkout(src_repo: str, dest: str, campaign_obj: Dict[str, Any]) -> str:
    """Clone the challenge repo locally into ``dest`` (reads ``src_repo`` only), then add the campaign file
    there. Returns the campaign path relative to the clone. The source repo is never written."""
    if os.path.exists(dest):
        raise qa.AdapterError("checkout destination already exists: " + dest)
    subprocess.run(["git", "clone", "--local", "--no-hardlinks", "-q", src_repo, dest], check=True)
    cid = campaign_obj["campaign"]["id"]
    rel = os.path.join("research", "campaigns", cid, "campaign.json")
    os.makedirs(os.path.join(dest, os.path.dirname(rel)), exist_ok=True)
    with open(os.path.join(dest, rel), "w", encoding="utf-8") as f:
        json.dump(campaign_obj, f, indent=2, sort_keys=True)
        f.write("\n")
    return rel


# ── candidate sampling (deterministic) ───────────────────────────────────────

def _canon_terms(ell: int, m: int, terms: Sequence[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    best = None
    for da, db in terms:                       # translate so each term in turn sits at the origin
        t = tuple(sorted(((a - da) % ell, (b - db) % m) for a, b in terms))
        if best is None or t < best:
            best = t
    return best


def canonical_key(ell: int, m: int, A, B) -> Tuple:
    """Multiplying A or B by a unit monomial, swapping A<->B, and (ell,m) ordering give equivalent codes."""
    a, b = _canon_terms(ell, m, A), _canon_terms(ell, m, B)
    return (ell, m) + tuple(sorted((a, b)))


def sample_specs(count: int = MAX_SCREENED, n_range: Tuple[int, int] = (60, 126),
                 seed: int = SAMPLER_SEED) -> List[qa.CandidateSpec]:
    import random
    rng = random.Random(seed)
    lms = [(ell, m) for ell in range(2, 64) for m in range(ell, 64) if n_range[0] <= 2 * ell * m <= n_range[1]]
    out: List[qa.CandidateSpec] = []
    seen = set()
    guard = 0
    while len(out) < count and guard < count * 200:
        guard += 1
        ell, m = rng.choice(lms)
        cells = [(i, j) for i in range(ell) for j in range(m)]
        A, B = rng.sample(cells, 3), rng.sample(cells, 3)
        key = canonical_key(ell, m, A, B)
        if key in seen:
            continue
        seen.add(key)
        a, b = key[2], key[3]
        out.append(qa.bb_spec(ell, m, a, b, "BB ell=%d m=%d" % (ell, m)))
    return out


# ── resource guards ──────────────────────────────────────────────────────────

def disk_state(path: str) -> Tuple[int, float]:
    u = shutil.disk_usage(path)
    return u.free >> 20, u.used / u.total


def guard(out_dir: str, *, mem: Callable = qa.mem_available_mb, disk: Callable = disk_state) -> None:
    m = mem()
    if m is not None and m < MIN_FREE_RAM_MB:
        raise SweepStop("available RAM %d MB < %d MB" % (m, MIN_FREE_RAM_MB))
    free_mb, frac = disk(out_dir)
    if free_mb < MIN_FREE_DISK_MB or frac > MAX_DISK_USED_FRAC:
        raise SweepStop("disk nearly full (free %d MB, used %.0f%%)" % (free_mb, frac * 100))


# ── evidence replay ──────────────────────────────────────────────────────────

def _rows(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    return [json.loads(ell) for ell in open(path, encoding="utf-8") if ell.strip()]


def _rank_key(r: Dict[str, Any]):
    d, k, n = r["screen_min_ub"], r["k"], r["n"]
    return (-(k * d * d / n), r["spec_key"])


def select_for_gate(screen_rows: Sequence[Dict[str, Any]], min_distance: int, limit: int) -> List[Dict[str, Any]]:
    ok = [r for r in screen_rows if r.get("screen_min_ub") and r["screen_min_ub"] >= min_distance and r.get("k", 0) > 0]
    return sorted(ok, key=_rank_key)[:limit]


@dataclass
class SweepConfig:
    out_dir: str
    staging_root: str
    screen_trials: int = 200
    screen_seeds: Tuple[int, ...] = (101, 211, 307)
    max_screened: int = MAX_SCREENED
    max_gate: int = MAX_GATE
    wall_budget_s: float = WALL_BUDGET_S
    authors: Tuple[str, ...] = ("osiris-dnalang",)
    screen_fn: Optional[Callable] = None     # tests only
    gate_fn: Optional[Callable] = None       # tests only; production always uses the repo's gate
    mem_fn: Callable = qa.mem_available_mb
    disk_fn: Callable = disk_state
    clock: Callable[[], float] = time.time


def _is_progress(verdict: Optional[Dict[str, Any]], rc: Optional[int], validator_sha: str) -> bool:
    if not verdict or rc != 0 or verdict.get("passed") is not True:
        return False
    if (verdict.get("validator") or {}).get("source_sha256") != validator_sha:
        return False
    return ((verdict.get("gates") or {}).get("dedup") or {}).get("exact_duplicate_of") is None


def run_sweep(repo: qa.ChallengeRepo, direction: qa.Direction, specs: Sequence[qa.CandidateSpec],
              cfg: SweepConfig) -> Dict[str, Any]:
    qa._refuse_paths(cfg.out_dir, repo)
    if len(specs) > min(cfg.max_screened, direction.max_candidates):
        raise qa.AdapterError("%d candidates exceeds the bound %d" % (len(specs), min(cfg.max_screened,
                                                                                    direction.max_candidates)))
    os.makedirs(cfg.out_dir, exist_ok=True)
    ev = qa.Evidence(os.path.join(cfg.out_dir, "evidence.jsonl"))
    rows = _rows(ev.path)
    if not rows:
        ev.append("RUN_START", {
            "repo": repo.snapshot(), "environment": qa.environment(),
            "direction": {"campaign_id": direction.campaign_id, "campaign_sha256": direction.campaign_sha256,
                          "campaign_status": direction.status},
            "limits": {"max_screened": cfg.max_screened, "max_gate": cfg.max_gate,
                       "wall_budget_s": cfg.wall_budget_s, "gate_mem_mb": qa.GATE_MEM_MB, "concurrency": 1,
                       "consecutive_no_progress": CONSEC_NO_PROGRESS, "max_accepted": MAX_ACCEPTED},
            "screen": {"trials": cfg.screen_trials, "seeds": list(cfg.screen_seeds), "sampler_seed": SAMPLER_SEED},
            "candidate_list_digest": qa._digest([s.key() for s in specs])})
        rows = _rows(ev.path)
    else:
        ev.append("RESUME", {"existing_rows": len(rows), "candidate_list_digest": qa._digest([s.key() for s in specs])})
        rows = _rows(ev.path)
    t_start = calendar.timegm(time.strptime(rows[0]["ts"], "%Y-%m-%dT%H:%M:%SZ"))
    kit = dict(repo.kit())
    import css as _css                      # repo's research/kit/css.py (path set by repo.kit())
    kit["css"] = _css
    stop_reason = None

    def over_wall() -> bool:
        return cfg.clock() - t_start > cfg.wall_budget_s

    try:
        # stage 1: screen (ranking signal only)
        done = {r["payload"]["spec_key"] for r in rows if r["kind"] == "SCREEN"}
        for spec in specs:
            if spec.key() in done:
                continue
            guard(cfg.out_dir, mem=cfg.mem_fn, disk=cfg.disk_fn)
            if over_wall():
                raise SweepStop("wall-clock budget reached during screening")
            p = {"spec_key": spec.key(), "family": spec.family, "params": spec.params, "status": "SCREENED"}
            t0 = time.monotonic()
            try:
                HX, HZ = qa.construct(kit, spec)
                p["n"] = int(HX.shape[1])
                p["k"] = int(kit["css"].compute_k(HX, HZ))
                p["check_weight"] = int(max(HX.sum(axis=1).max(), HZ.sum(axis=1).max()))
                if not direction.n_range[0] <= p["n"] <= direction.n_range[1]:
                    p["status"] = "NOT_PERMITTED"
                elif direction.max_check_weight and p["check_weight"] > direction.max_check_weight:
                    p["status"] = "NOT_PERMITTED"
                elif p["k"] <= 0:
                    p["status"] = "NO_LOGICAL_QUBITS"
                else:
                    ubs = {}
                    for s in cfg.screen_seeds:
                        ubs[str(s)] = int((cfg.screen_fn or (lambda a, b, t, sd: int(
                            kit["surrogate"].distance_rand(a, b, trials=t, seed=sd))))(HX, HZ, cfg.screen_trials, s))
                    p["screen_ub_by_seed"] = ubs
                    p["screen_min_ub"] = min(ubs.values())
                    p["screen_note"] = "surrogate upper bounds; ranking signal only, never a distance"
            except Exception as e:
                p["status"], p["error"] = "UNPACKAGEABLE", "%s: %s" % (type(e).__name__, e)
            p["screen_seconds"] = round(time.monotonic() - t0, 3)
            ev.append("SCREEN", p)
        rows = _rows(ev.path)
        screen_rows = [r["payload"] for r in rows if r["kind"] == "SCREEN"]
        # stage 2: official gate, sequential, stop rules
        ranked = select_for_gate(screen_rows, direction.min_distance or 0, cfg.max_gate)
        by_key = {s.key(): s for s in specs}
        gated = [r["payload"] for r in rows if r["kind"] == "GATE"]
        consecutive = 0
        accepted = 0
        for g in gated:
            if g["progress"]:
                accepted, consecutive = accepted + 1, 0
            else:
                consecutive += 1
        for item in ranked:
            if item["spec_key"] in {g["spec_key"] for g in gated}:
                continue
            if len(gated) >= cfg.max_gate:
                stop_reason = "gate budget exhausted"
                break
            if accepted >= MAX_ACCEPTED:
                stop_reason = "three accepted candidates"
                break
            if consecutive >= CONSEC_NO_PROGRESS:
                stop_reason = "eight consecutive gate evaluations without progress"
                break
            guard(cfg.out_dir, mem=cfg.mem_fn, disk=cfg.disk_fn)
            if over_wall():
                raise SweepStop("wall-clock budget reached before gate")
            spec = by_key[item["spec_key"]]
            HX, HZ = qa.construct(kit, spec)
            g: Dict[str, Any] = {"spec_key": spec.key(), "params": spec.params, "n": item["n"], "k": item["k"],
                                 "screen_ub_by_seed": item["screen_ub_by_seed"], "screen_min_ub": item["screen_min_ub"],
                                 "gate_ordinal": len(gated) + 1}
            try:
                doc = kit["submit"].make_submission(
                    HX, HZ, name="[[%d,%d,?]] %s (OSIRIS sweep)" % (item["n"], item["k"], spec.label),
                    construction="%s %s; OSIRIS qLDPC sweep, surrogate ranking only (%d trials, seeds %s)" % (
                        spec.family, json.dumps(spec.params, sort_keys=True), cfg.screen_trials,
                        list(cfg.screen_seeds)),
                    authors=list(cfg.authors), family=spec.family, confidence="upper_bound",
                    trials=cfg.screen_trials)
                g["candidate_digest"] = kit["coordination"].content_digest(doc)
                pending = os.path.join(cfg.out_dir, "candidates", spec.key() + ".json")
                os.makedirs(os.path.dirname(pending), exist_ok=True)
                kit["submit"].save_submission(doc, pending, on_collision="overwrite")
                t0 = time.monotonic()
                verdict, rc, err = (cfg.gate_fn or qa.run_official_gate)(repo, pending)
                g["gate_seconds"] = round(time.monotonic() - t0, 2)
                g["peak_rss_mb_gate_children_so_far"] = round(
                    resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1)
                g.update({"gate_exit_code": rc, "gate_stderr_tail": err, "gate_verdict": verdict})
                v = verdict or {}
                g["validator_source_sha256"] = (v.get("validator") or {}).get("source_sha256")
                g["validator_seed"] = ((v.get("gates") or {}).get("refute") or {}).get("seed")
                g["refuted"] = ((v.get("gates") or {}).get("refute") or {}).get("refuted")
                g["exact_duplicate_of"] = ((v.get("gates") or {}).get("dedup") or {}).get("exact_duplicate_of")
                g["board_advancing"] = ((v.get("gates") or {}).get("novelty") or {}).get("board_advancing")
                g["literature_novelty"] = ((v.get("gates") or {}).get("novelty") or {}).get("literature_novelty")
                g["progress"] = _is_progress(verdict, rc, repo.validator_sha256)
                g["status"] = ("VALIDATED" if g["progress"] else "GATE_ERROR" if verdict is None or rc not in (0, 1)
                               else "REFUTED" if g["refuted"] else "REJECTED")
                if g["progress"]:
                    stage = kit["coordination"].staging_dir(root=cfg.staging_root)
                    sp = kit["coordination"].unique_path(
                        os.path.join(stage, "%d-%d-%d.json" % (doc["n"], doc["k"], doc["distance"]["d"])), doc)
                    kit["submit"].save_submission(doc, sp)
                    g["staged_path"] = sp
            except Exception as e:                       # recorded, never retried
                g.update({"status": "UNPACKAGEABLE", "progress": False, "error": "%s: %s" % (type(e).__name__, e)})
            ev.append("GATE", g)
            gated.append(g)
            if g["progress"]:
                accepted, consecutive = accepted + 1, 0
            else:
                consecutive += 1
        else:
            stop_reason = stop_reason or ("gate candidates exhausted (%d eligible by surrogate)" % len(ranked))
        if stop_reason is None:
            stop_reason = "completed"
    except SweepStop as e:
        stop_reason = "SAFE_STOP: %s" % e
    rows = _rows(ev.path)
    gates = [r["payload"] for r in rows if r["kind"] == "GATE"]
    screens = [r["payload"] for r in rows if r["kind"] == "SCREEN"]
    counts: Dict[str, int] = {}
    for g in gates:
        counts[g["status"]] = counts.get(g["status"], 0) + 1
    scount: Dict[str, int] = {}
    for s in screens:
        scount[s["status"]] = scount.get(s["status"], 0) + 1
    summary = {"stop_reason": stop_reason, "screened": len(screens), "screen_status_counts": scount,
               "gate_evaluations": len(gates), "gate_status_counts": counts,
               "accepted": [g.get("staged_path") for g in gates if g.get("progress")],
               "board_advancing": [g["candidate_digest"] for g in gates if g.get("progress") and g.get("board_advancing")],
               "repo_rev": repo.git_rev,
               "claims": "accepted = official gate passed:true (upper-bound distance; literature novelty unverified); "
                         "nothing submitted, committed or published"}
    ev.append("RUN_END", summary)
    summary["evidence"] = ev.path
    return summary
