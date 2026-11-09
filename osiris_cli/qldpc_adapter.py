"""Bounded, governed adapter to the qLDPC challenge's autoresearch loop.

Authority: the challenge repository (``research/QUICKSTART.md``, ``AGENTS.md``, its campaign
definitions) and its trusted gate ``verify/validate_candidate.py``. This module is OSIRIS-side glue
only. It never edits, copies-and-patches or reimplements anything under ``verify/``; it never
measures a distance itself (distances come from the repo's surrogate, as RANKING signals, or from the
gate); and it never writes to the challenge repo except the stage-only directory the repo's own
workflow prescribes (``research/candidates/<run>/``, gitignored), and only for candidates the gate
passed. No git, no network, no hardware, no publishing, no ``~/.osiris``.

Pipeline per candidate:  construct (kit) -> screen (surrogate, ranking only) -> package
(``submit.make_submission``, confidence ``upper_bound``) -> official gate (subprocess of the repo's
own ``validate_candidate.py``) -> classify:

  SCREENED_OUT   never sent to the gate (surrogate rank below the cut, or budget reached)
  UNPACKAGEABLE  the kit refused to build a submission (e.g. k = 0, CSS violation)
  REJECTED       gate ran, ``passed`` false (structure/duplicate); evidence only
  REFUTED        gate ran, its own refutation found a lighter logical; evidence only
  VALIDATED      gate returned ``passed: true``; staged for HUMAN review. Not a discovery, not
                 novel (literature novelty is UNVERIFIED), distance is an upper bound.

Every outcome is a row in a hash-chained ``evidence.jsonl`` carrying the complete verdict, candidate
digest, repo revision/state, validator hash and execution environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

DEFAULT_REPO = os.environ.get("OSIRIS_QLDPC_REPO", os.path.expanduser("~/github-backup/qldpc-challenge"))
GENESIS = "0" * 64
HONEST_LABELS = ["distance is an upper bound (witness-backed)", "literature novelty UNVERIFIED",
                 "staged for human review; not a board entry"]
GATE_TIMEOUT_S = 600.0
GATE_MEM_MB = 3072
MIN_AVAILABLE_MB = 1500


class AdapterError(RuntimeError):
    pass


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _sha_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _git(root: str, *args: str) -> str:
    r = subprocess.run(["git", "-C", root] + list(args), capture_output=True, text=True, timeout=30)
    return r.stdout.rstrip("\n") if r.returncode == 0 else ""   # keep porcelain's leading status column


def mem_available_mb() -> Optional[int]:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


# ── the challenge repository ────────────────────────────────────────────────

@dataclass
class ChallengeRepo:
    root: str
    git_rev: str = ""
    git_dirty: List[str] = field(default_factory=list)
    validator_sha256: str = ""
    integrity_ok: bool = False
    integrity_output: str = ""
    require_git: bool = True

    @classmethod
    def open(cls, root: Optional[str] = None, *, require_git: bool = True,
             run_integrity: bool = True) -> "ChallengeRepo":
        root = os.path.realpath(root or DEFAULT_REPO)
        for rel in ("verify/validate_candidate.py", "research/kit/submit.py", "research/QUICKSTART.md",
                    "codes", "schema"):
            if not os.path.exists(os.path.join(root, rel)):
                raise AdapterError(f"not a qLDPC challenge checkout: {rel} missing under {root}")
        r = cls(root, require_git=require_git)
        r.git_rev = _git(root, "rev-parse", "HEAD").strip()
        r.git_dirty = [ln for ln in _git(root, "status", "--porcelain").splitlines() if ln.strip()]
        if require_git and not r.git_rev:
            raise AdapterError(f"{root} is not a git checkout; cannot establish its revision")
        if any(ln[3:].strip().startswith("verify/") for ln in r.git_dirty):
            raise AdapterError("verify/ has local modifications; refusing to use a modified trusted gate")
        r.validator_sha256 = _sha_file(os.path.join(root, "verify", "validate_candidate.py"))
        if run_integrity:
            p = subprocess.run([sys.executable, "-I", os.path.join(root, "verify", "check_validator_integrity.py"),
                                "--root", root], capture_output=True, text=True, timeout=120,
                               env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"})
            r.integrity_ok, r.integrity_output = p.returncode == 0, (p.stdout + p.stderr).strip()[-300:]
            if not r.integrity_ok:
                raise AdapterError("validator integrity check failed: " + r.integrity_output)
        return r

    def snapshot(self) -> Dict[str, Any]:
        return {"root_name": os.path.basename(self.root), "git_rev": self.git_rev,
                "working_tree_dirty_paths": self.git_dirty[:50], "validator_sha256": self.validator_sha256,
                "validator_integrity_ok": self.integrity_ok}

    def kit(self):
        """Import the repo's own kit modules (read-only use). Switching repos purges the previous
        repo's modules so one checkout's code is never served for another."""
        global _KIT_ROOT
        if _KIT_ROOT != self.root:
            for name, mod in list(sys.modules.items()):
                f = getattr(mod, "__file__", None) or ""
                if _KIT_ROOT and f.startswith(_KIT_ROOT + os.sep):
                    del sys.modules[name]
            sys.path[:] = [p for p in sys.path if not (_KIT_ROOT and p.startswith(_KIT_ROOT + os.sep))]
            _KIT_ROOT = self.root
        for p in (os.path.join(self.root, "research", "kit"), os.path.join(self.root, "verify")):
            if p not in sys.path:
                sys.path.insert(0, p)
        sys.dont_write_bytecode = True
        import bb, campaign, coordination, submit, surrogate   # noqa: E401  (repo modules, on sys.path above)
        return {"bb": bb, "campaign": campaign, "coordination": coordination, "submit": submit,
                "surrogate": surrogate}


_KIT_ROOT: Optional[str] = None


def environment() -> Dict[str, Any]:
    env: Dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform(),
                           "machine": platform.machine(), "cpu_count": os.cpu_count(),
                           "mem_available_mb": mem_available_mb()}
    for m in ("numpy", "scipy", "jsonschema"):
        try:
            from importlib import metadata
            env[m] = metadata.version(m)
        except Exception:
            env[m] = None
    return env


# ── direction (from the repo's own campaign definitions) ────────────────────

@dataclass(frozen=True)
class Direction:
    campaign_id: str
    families: Tuple[str, ...]
    n_range: Tuple[int, int]
    max_check_weight: Optional[int]
    min_distance: Optional[int]
    max_candidates: int
    trials_per_candidate: int
    campaign_sha256: str
    status: str


def load_direction(repo: ChallengeRepo, campaign_json: str) -> Direction:
    """The permitted research direction, read through the repo's own validating loader."""
    path = campaign_json if os.path.isabs(campaign_json) else os.path.join(repo.root, campaign_json)
    path = os.path.realpath(path)
    if not path.startswith(repo.root + os.sep):
        raise AdapterError("campaign must live inside the challenge repo")
    repo.kit()["campaign"].load_campaign(path)   # validates against the repo schema
    raw = json.load(open(path))["campaign"]
    con, meth = raw.get("constraints", {}), raw.get("methods", {})
    n = con.get("n") or [1, 10**9]
    return Direction(raw["id"], tuple(meth.get("families", [])), (int(n[0]), int(n[1])),
                     con.get("max_check_weight"), con.get("min_distance"),
                     int(raw.get("budget", {}).get("candidates_screened", 1)),
                     int(meth.get("screening", {}).get("trials_per_candidate", 200)), _sha_file(path),
                     raw.get("status", "?"))


# ── candidates ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CandidateSpec:
    family: str
    params: Dict[str, Any]
    label: str = ""

    def key(self) -> str:
        return _digest({"family": self.family, "params": self.params})[:16]


def bb_spec(ell: int, m: int, a_terms: Sequence[Tuple[int, int]], b_terms: Sequence[Tuple[int, int]],
            label: str = "") -> CandidateSpec:
    return CandidateSpec("bivariate-bicycle", {"l": ell, "m": m, "A": [list(t) for t in a_terms],
                                               "B": [list(t) for t in b_terms]}, label)


def construct(kit: Dict[str, Any], spec: CandidateSpec):
    if spec.family != "bivariate-bicycle":
        raise AdapterError(f"no constructor registered for family {spec.family!r}")
    p = spec.params
    return kit["bb"].build_bb(p["l"], p["m"], [tuple(t) for t in p["A"]], [tuple(t) for t in p["B"]])


@dataclass
class Outcome:
    spec: CandidateSpec
    status: str
    reason: str = ""
    screen_distance_ub: Optional[int] = None
    candidate_digest: str = ""
    verdict: Optional[Dict[str, Any]] = None
    gate_exit: Optional[int] = None
    staged_path: Optional[str] = None
    n: Optional[int] = None
    k: Optional[int] = None


# ── the official gate ────────────────────────────────────────────────────────

def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (GATE_MEM_MB << 20,) * 2)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.setsid()


def run_official_gate(repo: ChallengeRepo, doc_path: str, *, timeout: float = GATE_TIMEOUT_S
                      ) -> Tuple[Optional[Dict[str, Any]], Optional[int], str]:
    """Run the repo's own ``verify/validate_candidate.py`` unmodified, on a file. Returns
    (verdict dict | None, exit code, stderr tail). The verdict is whatever the gate printed."""
    env = {"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", "HOME": os.path.dirname(doc_path)}
    try:
        p = subprocess.run([sys.executable, "-I", os.path.join(repo.root, "verify", "validate_candidate.py"),
                            doc_path], capture_output=True, text=True, timeout=timeout, env=env,
                           cwd=repo.root, preexec_fn=_limits)
    except subprocess.TimeoutExpired:
        return None, None, "gate timed out"
    try:
        verdict = json.loads(p.stdout)
    except ValueError:
        verdict = None
    return verdict, p.returncode, p.stderr.strip()[-400:]


# ── evidence ─────────────────────────────────────────────────────────────────

class Evidence:
    """Append-only hash-chained JSONL (this run's own file; not any canonical ledger)."""

    def __init__(self, path: str):
        self.path = path

    def _tail(self) -> str:
        prev = GENESIS
        if os.path.isfile(self.path):
            for ln in open(self.path, encoding="utf-8"):
                if ln.strip():
                    prev = json.loads(ln)["hash"]
        return prev

    def append(self, kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        row = {"kind": kind, "payload": payload, "prev": self._tail(),
               "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        row["hash"] = _digest(row)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return row

    @staticmethod
    def verify(path: str) -> bool:
        prev = GENESIS
        for ln in open(path, encoding="utf-8"):
            if ln.strip():
                r = json.loads(ln)
                if r["prev"] != prev or r["hash"] != _digest({k: v for k, v in r.items() if k != "hash"}):
                    return False
                prev = r["hash"]
        return True


# ── the bounded run ──────────────────────────────────────────────────────────

@dataclass
class RunConfig:
    out_dir: str                       # evidence + failed candidates (explicit, never defaulted)
    staging_root: Optional[str] = None  # where VALIDATED candidates go; None -> repo research/candidates
    screen_trials: int = 200
    top_k: int = 2                     # at most this many ranked survivors reach the gate
    max_candidates: int = 16
    wall_budget_s: float = 900.0
    authors: Tuple[str, ...] = ("osiris-dnalang",)
    gate: Optional[Callable[[ChallengeRepo, str], Tuple[Optional[dict], Optional[int], str]]] = None
    # ``gate`` exists only so tests can simulate gate outputs; production always uses the repo's gate.


def _refuse_paths(out_dir: str, repo: ChallengeRepo) -> None:
    live = os.path.realpath(os.path.expanduser("~/.osiris"))
    o = os.path.realpath(out_dir)
    if o == live or o.startswith(live + os.sep):
        raise AdapterError("refusing to write into live OSIRIS state (~/.osiris)")
    verify = os.path.join(repo.root, "verify")
    if o == verify or o.startswith(verify + os.sep):
        raise AdapterError("refusing to write under verify/")
    if o == repo.root or (o.startswith(repo.root + os.sep) and
                          not o.startswith(os.path.join(repo.root, "research", "candidates") + os.sep)):
        raise AdapterError("out_dir must be outside the challenge repo (staging is separate and stage-only)")


def run_campaign(repo: ChallengeRepo, direction: Direction, specs: Sequence[CandidateSpec], cfg: RunConfig
                 ) -> Dict[str, Any]:
    _refuse_paths(cfg.out_dir, repo)
    if not specs:
        raise AdapterError("no candidates supplied")
    if len(specs) > min(cfg.max_candidates, direction.max_candidates):
        raise AdapterError(f"{len(specs)} candidates exceeds the bound "
                           f"{min(cfg.max_candidates, direction.max_candidates)} (config/campaign budget)")
    avail = mem_available_mb()
    if avail is not None and avail < MIN_AVAILABLE_MB:
        raise AdapterError(f"only {avail} MB RAM available (< {MIN_AVAILABLE_MB}); not starting")
    os.makedirs(cfg.out_dir, exist_ok=True)
    kit = repo.kit()
    ev = Evidence(os.path.join(cfg.out_dir, "evidence.jsonl"))
    t0 = time.monotonic()
    meta = {"repo": repo.snapshot(), "environment": environment(),
            "direction": {"campaign_id": direction.campaign_id, "campaign_sha256": direction.campaign_sha256,
                          "families": list(direction.families), "n_range": list(direction.n_range),
                          "max_check_weight": direction.max_check_weight, "campaign_status": direction.status},
            "config": {"screen_trials": cfg.screen_trials, "top_k": cfg.top_k,
                       "wall_budget_s": cfg.wall_budget_s}}
    ev.append("RUN_START", meta)

    outcomes: List[Outcome] = []
    built: List[Tuple[Outcome, Any, Any]] = []
    # 1. construct + screen (ranking only)
    for spec in specs:
        o = Outcome(spec, "SCREENED_OUT")
        outcomes.append(o)
        if spec.family not in direction.families:
            o.status, o.reason = "NOT_PERMITTED", f"family {spec.family!r} is not in the campaign's families"
            continue
        if time.monotonic() - t0 > cfg.wall_budget_s:
            o.reason = "wall-clock budget reached before screening"
            continue
        try:
            HX, HZ = construct(kit, spec)
            n = int(HX.shape[1])
            if not direction.n_range[0] <= n <= direction.n_range[1]:
                o.status, o.reason = "NOT_PERMITTED", f"n={n} outside campaign range {list(direction.n_range)}"
                o.n = n
                continue
            w = int(max(HX.sum(axis=1).max(), HZ.sum(axis=1).max()))
            if direction.max_check_weight is not None and w > direction.max_check_weight:
                o.status, o.reason = "NOT_PERMITTED", f"check weight {w} > {direction.max_check_weight}"
                continue
            o.n = n
            o.screen_distance_ub = int(kit["surrogate"].distance_rand(HX, HZ, trials=cfg.screen_trials))
        except Exception as e:                      # a malformed candidate is evidence, not a crash
            o.status, o.reason = "UNPACKAGEABLE", f"construction/screen failed: {type(e).__name__}: {e}"
            continue
        built.append((o, HX, HZ))
    # rank by the surrogate; it only orders, it never claims
    built.sort(key=lambda t: -(t[0].screen_distance_ub or 0))
    for rank, (o, HX, HZ) in enumerate(built):
        if rank >= cfg.top_k or (direction.min_distance and (o.screen_distance_ub or 0) < direction.min_distance):
            o.reason = ("surrogate upper bound below campaign min_distance" if rank < cfg.top_k
                        else f"surrogate rank {rank + 1} beyond top_k={cfg.top_k}")
            continue
        if time.monotonic() - t0 > cfg.wall_budget_s:
            o.reason = "wall-clock budget reached before the gate"
            continue
        # 2. package (witness embedded), 3. gate, 4. classify
        try:
            doc = kit["submit"].make_submission(
                HX, HZ, name=f"[[{o.n},?,{o.screen_distance_ub}]] {o.spec.label or o.spec.family} (adapter)",
                construction=f"{o.spec.family} {json.dumps(o.spec.params, sort_keys=True)}; OSIRIS qLDPC adapter, "
                             f"surrogate screen {cfg.screen_trials} trials (ranking only)",
                authors=list(cfg.authors), family=o.spec.family, confidence="upper_bound",
                trials=cfg.screen_trials)
        except Exception as e:
            o.status, o.reason = "UNPACKAGEABLE", f"{type(e).__name__}: {e}"
            continue
        o.n, o.k = int(doc["n"]), int(doc["k"])
        o.candidate_digest = kit["coordination"].content_digest(doc)
        pending = os.path.join(cfg.out_dir, "candidates", f"{o.spec.key()}.json")
        os.makedirs(os.path.dirname(pending), exist_ok=True)
        kit["submit"].save_submission(doc, pending, on_collision="overwrite")   # witness kept in evidence dir
        verdict, rc, err = (cfg.gate or run_official_gate)(repo, pending)
        o.verdict, o.gate_exit = verdict, rc
        if verdict is None or rc not in (0, 1):
            o.status, o.reason = "GATE_ERROR", f"gate produced no verdict (exit {rc}): {err}"
        elif verdict.get("passed") is True and rc == 0:
            if (verdict.get("validator") or {}).get("source_sha256") != repo.validator_sha256:
                o.status, o.reason = "GATE_ERROR", "verdict's validator hash does not match the checked-out gate"
            else:
                stage = kit["coordination"].staging_dir(root=cfg.staging_root) if cfg.staging_root \
                    else kit["coordination"].staging_dir()
                o.staged_path = os.path.join(stage, f"{doc['n']}-{doc['k']}-{doc['distance']['d']}.json")
                o.staged_path = kit["coordination"].unique_path(o.staged_path, doc)
                kit["submit"].save_submission(doc, o.staged_path)
                o.status, o.reason = "VALIDATED", "gate passed:true (upper bound; novelty unverified; human review)"
        elif (verdict.get("gates", {}).get("refute") or {}).get("refuted"):
            o.status, o.reason = "REFUTED", (verdict["gates"]["refute"].get("detail") or "refuted by the gate")
        else:
            o.status, o.reason = "REJECTED", "; ".join(verdict.get("labels", [])[:2]) or "gate passed:false"

    for o in outcomes:
        ev.append("CANDIDATE", {
            "status": o.status, "reason": o.reason, "family": o.spec.family, "params": o.spec.params,
            "spec_key": o.spec.key(), "n": o.n, "k": o.k, "screen_distance_upper_bound": o.screen_distance_ub,
            "screen_note": "surrogate value is a ranking signal only", "candidate_digest": o.candidate_digest,
            "gate_exit_code": o.gate_exit, "gate_verdict": o.verdict, "staged_path": o.staged_path,
            "labels": HONEST_LABELS if o.status == "VALIDATED" else ["NOT a discovery", "evidence only"]})
    counts: Dict[str, int] = {}
    for o in outcomes:
        counts[o.status] = counts.get(o.status, 0) + 1
    summary = {"counts": counts, "validated": [o.staged_path for o in outcomes if o.status == "VALIDATED"],
               "wall_s": round(time.monotonic() - t0, 2), "repo_rev": repo.git_rev,
               "claims": "VALIDATED means the repo's gate returned passed:true; distances are upper bounds; "
                         "literature novelty unverified; nothing was submitted, committed or published"}
    ev.append("RUN_END", summary)
    summary["evidence"] = ev.path
    return summary
