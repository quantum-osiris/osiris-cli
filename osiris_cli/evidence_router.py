"""Evidence-Weighted, Role-Specialized Deterministic Model Router (`osiris_cli/evidence_router.py`).

Unifies OSIRIS's model selection across Architect (`L4`), Code Synthesizer (`/propose`,
`run_synergy_pipeline`, `/code`), Independent Critic (`L5`), and Conversational Voice (`L0`/`L4`)
using real measured local evidence rather than stale static model lists:

  1. Installed runtime inventory (`Ollama /api/tags` + `gemini_bridge` cloud availability).
  2. Role affinity + parameter-scale awareness across modern model families (`qwen2.5-coder`,
     `qwen3.5`, `gemma4`, `qwen3`, `deepseek-coder`, `gemma3`, `qwen2.5`, `llama3`, `smollm2`).
  3. Verified benchmark scorecard (`protege.py`): Wilson 95% lower bound, task-specific
     pass standing (`backlog`, `claims`, `gaps`, etc.), `oracle@k` verified passes, and
     automatic demotion of models with `>= 3` scored candidates and `0` passes.
  4. Confidence calibration (`osiris_cli/calibration.py`): penalizes empirical false-confidence
     overconfidence (`own_ok=True` while `hidden_ok=False`).
  5. Empirical latency (`~/.osiris/living/stats.json` median `first_word` seconds) & substrate
     RAM envelope (`probes.probe_substrate()` — protecting Termux / `< 3.5 GiB` edge nodes from
     `>= 7B` OOM thrashing).
  6. Cross-family L5 critic diversity: `select_for_role("critic", avoid_model=...)` prefers an
     independent model family from the proposer so generator and critic do not share blind spots.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

PathArg = Optional[Union[os.PathLike, str]]

ROLES: Tuple[str, ...] = ("architect", "code", "critic", "voice")

# Family role priors in [0.0, 1.0]: (prefix, architect_prior, code_prior, critic_prior, voice_prior)
_FAMILY_PRIORS: Tuple[Tuple[str, float, float, float, float], ...] = (
    ("qwen2.5-coder", 0.76, 0.95, 0.80, 0.74),
    ("deepseek-coder", 0.72, 0.92, 0.78, 0.65),
    ("qwen3.5", 0.92, 0.90, 0.92, 0.84),
    ("gemma4", 0.91, 0.88, 0.91, 0.86),
    ("qwen3", 0.88, 0.87, 0.89, 0.80),
    ("qwen2.5", 0.84, 0.83, 0.84, 0.82),
    ("gemma3", 0.80, 0.76, 0.80, 0.85),
    ("gemma-3", 0.80, 0.76, 0.80, 0.85),
    ("gemma2", 0.75, 0.72, 0.75, 0.80),
    ("llama3", 0.74, 0.72, 0.74, 0.78),
    ("smollm2", 0.52, 0.50, 0.50, 0.72),
)

_PARAM_BILLIONS_RE = re.compile(r":(?:e)?(\d+(?:\.\d+)?)b\b", re.IGNORECASE)
_PARAM_MILLIONS_RE = re.compile(r":(\d+)m\b", re.IGNORECASE)


@dataclass(frozen=True)
class ModelCandidateScore:
    model: str
    provider: str  # ollama | gemini
    family: str
    param_b: float
    role: str
    total_score: float
    role_prior: float
    bench_bonus: float
    calibration_adj: float
    latency_adj: float
    substrate_adj: float
    verified_passes: int
    scored_candidates: int
    wilson_lb: float
    overconfidence: Optional[float]
    median_latency_s: Optional[float]
    demoted: bool
    reasons: Tuple[str, ...]


def model_family(model_name: str) -> str:
    low = (model_name or "").strip().lower()
    base = low.split(":", 1)[0]
    for prefix in ("qwen2.5-coder", "deepseek-coder", "qwen3.5", "gemma4", "qwen3", "qwen2.5", "gemma3", "gemma-3", "gemma2", "llama3", "smollm2"):
        if base.startswith(prefix):
            return prefix.replace("gemma-3", "gemma3")
    return base or "unknown"


def estimate_param_billions(model_name: str) -> float:
    low = (model_name or "").lower()
    m_b = _PARAM_BILLIONS_RE.search(low)
    if m_b:
        return float(m_b.group(1))
    m_m = _PARAM_MILLIONS_RE.search(low)
    if m_m:
        return float(m_m.group(1)) / 1000.0
    if "7b" in low:
        return 7.0
    if "8b" in low:
        return 8.0
    if "9b" in low:
        return 9.0
    if "4b" in low:
        return 4.0
    if "1b" in low:
        return 1.0
    return 4.0


def _base_role_prior(model_name: str, role: str) -> float:
    low = (model_name or "").lower()
    role_idx = {"architect": 1, "code": 2, "synthesizer": 2, "critic": 3, "voice": 4, "fast": 4}.get(role, 1)
    prior = 0.68
    for prefix, p_arch, p_code, p_crit, p_voice in _FAMILY_PRIORS:
        if low.startswith(prefix):
            prior = (p_arch, p_code, p_crit, p_voice)[role_idx - 1]
            break

    params = estimate_param_billions(model_name)
    # Parameter-scale bonus on reasoning/code/critic vs compactness bonus on voice/fast
    if role in ("architect", "code", "synthesizer", "critic"):
        if params >= 8.0:
            prior += 0.05
        elif params >= 4.0:
            prior += 0.03
        elif params < 2.0:
            prior -= 0.10
    else:
        if 1.5 <= params <= 5.0:
            prior += 0.04
        elif params >= 8.5:
            prior -= 0.04
    return round(max(0.1, min(1.0, prior)), 4)


def _load_bench_scorecard(bench_dir: Optional[str] = None) -> Dict[str, Any]:
    try:
        import protege

        bdir = bench_dir or protege.BENCH_DIR
        entries, problem = protege.load_ledger(bdir)
        if problem or not entries:
            return {}
        cands, _ = protege.candidates(
            entries, protege.current_suite(), protege.current_runner(), bench_dir=bdir
        )
        return protege.scorecard(cands) or {}
    except Exception:
        return {}


def _load_calibration_by_model(osiris_home: PathArg = None) -> Dict[str, Tuple[int, float, float]]:
    """Return `mentor_id -> (num_samples, brier_score, overconfidence)`."""
    try:
        from . import calibration

        led = calibration.CalibrationLedger(calibration.default_calibration_path(osiris_home))
        ok, _cnt, _err = led.verify()
        if not ok:
            return {}
        obs = led.observations()
        if not obs:
            return {}
        out: Dict[str, Tuple[int, float, float]] = {}
        for mid in sorted({o.model_id for o in obs}):
            rep = calibration.compute_calibration(obs, model_filter=mid)
            if rep.num_samples > 0:
                out[mid] = (
                    rep.num_samples,
                    float(rep.brier_score_str),
                    float(rep.overconfidence_str),
                )
        return out
    except Exception:
        return {}


def _load_median_latencies(living_home: PathArg = None) -> Dict[str, float]:
    base = (
        Path(living_home).expanduser().resolve()
        if living_home is not None
        else (Path.home() / ".osiris" / "living")
    )
    stats_file = base / "stats.json"
    if not stats_file.is_file():
        return {}
    try:
        data = json.loads(stats_file.read_text(encoding="utf-8"))
        fw = data.get("first_word") or {}
        out: Dict[str, float] = {}
        for mname, samples in fw.items():
            if isinstance(samples, list) and len(samples) >= 2:
                vals = sorted(float(x) for x in samples if isinstance(x, (int, float)))
                if vals:
                    out[str(mname)] = round(vals[len(vals) // 2], 2)
        return out
    except Exception:
        return {}


def _infer_task_bucket(task_text: str) -> str:
    low = (task_text or "").lower()
    if any(w in low for w in ("claim", "unsourced", "number", "legit", "provenance", "citation")):
        return "claims"
    if any(w in low for w in ("backlog", "duplicate", "idempoten", "sprint", "story")):
        return "backlog"
    if any(w in low for w in ("gap", "extract", "regex", "truncat", "parse")):
        return "gaps"
    return ""


def rank_models(
    role: str = "code",
    *,
    installed: Optional[Sequence[str]] = None,
    task_text: str = "",
    avoid_model: Optional[str] = None,
    is_edge_low_ram: Optional[bool] = None,
    bench_dir: Optional[str] = None,
    osiris_home: PathArg = None,
    living_home: PathArg = None,
) -> List[ModelCandidateScore]:
    """Deterministically score and rank installed local models for `role` + `task_text`."""
    norm_role = role.strip().lower()
    if norm_role == "synthesizer":
        norm_role = "code"
    elif norm_role == "fast":
        norm_role = "voice"

    if installed is None:
        from .providers.router import discover_ollama_models

        _ok, discovered = discover_ollama_models(timeout=1.2)
        model_list = list(discovered)
    else:
        model_list = [str(m) for m in installed if m]

    if not model_list:
        return []

    if is_edge_low_ram is None:
        try:
            from .probes import probe_substrate

            sub = probe_substrate()
            is_edge_low_ram = bool(sub.is_termux or sub.free_ram_mb < 3584)
        except Exception:
            is_edge_low_ram = False

    card = _load_bench_scorecard(bench_dir=bench_dir)
    cal_map = _load_calibration_by_model(osiris_home=osiris_home)
    lat_map = _load_median_latencies(living_home=living_home)
    task_bucket = _infer_task_bucket(task_text)
    avoid_fam = model_family(avoid_model) if avoid_model else ""

    scored: List[ModelCandidateScore] = []
    for mname in dict.fromkeys(model_list):
        fam = model_family(mname)
        params = estimate_param_billions(mname)
        prior = _base_role_prior(mname, norm_role)
        reasons: List[str] = [f"prior={prior:.2f} ({fam}, ~{params:g}B)"]

        mentor_key = f"ollama/{mname}"
        mstats = card.get(mentor_key) or {}
        n_cands = int(mstats.get("candidates", 0))
        n_ver = int(mstats.get("verified", 0))
        w_lb = 0.0
        bench_bonus = 0.0
        demoted = False

        if n_cands > 0:
            try:
                import protege

                w_lb = round(float(protege.wilson_lower(n_ver, n_cands)), 4)
            except Exception:
                w_lb = round(n_ver / n_cands, 4)
            if n_ver > 0:
                pass_rate = n_ver / n_cands
                bench_bonus += 0.30 * w_lb + 0.22 * pass_rate
                reasons.append(f"bench={n_ver}/{n_cands} (Wilson_lb={w_lb:.2f}, +{bench_bonus:.2f})")
                if task_bucket:
                    tstats = (mstats.get("by_task") or {}).get(task_bucket) or {}
                    if tstats.get("verified", 0) > 0:
                        bench_bonus += 0.12
                        reasons.append(f"task[{task_bucket}] verified (+0.12)")
            elif n_cands >= 3:
                demoted = True
                bench_bonus -= 0.32
                reasons.append(f"demoted: 0/{n_cands} verified passes (-0.32)")

        cal_adj = 0.0
        overconf_val: Optional[float] = None
        if mentor_key in cal_map:
            cal_n, brier, overconf = cal_map[mentor_key]
            overconf_val = round(overconf, 4)
            if cal_n >= 3:
                if overconf > 0.15:
                    cal_adj = -round(0.18 * overconf, 4)
                    reasons.append(f"false-conf penalty ({overconf:+.2f} overconf -> {cal_adj:+.2f})")
                elif brier <= 0.15:
                    cal_adj = 0.05
                    reasons.append(f"well-calibrated (Brier={brier:.2f} -> +0.05)")

        lat_adj = 0.0
        med_lat = lat_map.get(mname)
        if med_lat is not None:
            if norm_role == "voice":
                if med_lat <= 5.0:
                    lat_adj = 0.10
                    reasons.append(f"fast {med_lat:.1f}s (+0.10)")
                elif med_lat >= 30.0:
                    lat_adj = -0.08
                    reasons.append(f"slow {med_lat:.1f}s (-0.08)")
            elif med_lat <= 4.0:
                lat_adj = 0.04
                reasons.append(f"responsive {med_lat:.1f}s (+0.04)")

        sub_adj = 0.0
        if is_edge_low_ram:
            if params >= 6.5:
                sub_adj -= 0.35
                reasons.append("edge low-RAM guard (>=6.5B -> -0.35)")
            elif params <= 4.5:
                sub_adj += 0.10
                reasons.append("edge compact fit (+0.10)")

        # Cross-family diversity bonus for Independent L5 Critic
        if norm_role == "critic" and avoid_fam and len(model_list) >= 2:
            if avoid_model and mname == avoid_model:
                sub_adj -= 0.25
                reasons.append("avoid same proposer model (-0.25)")
            elif fam != avoid_fam:
                sub_adj += 0.12
                reasons.append(f"cross-family critic vs {avoid_fam} (+0.12)")

        total = round(prior + bench_bonus + cal_adj + lat_adj + sub_adj, 4)
        scored.append(
            ModelCandidateScore(
                model=mname,
                provider="ollama",
                family=fam,
                param_b=params,
                role=norm_role,
                total_score=total,
                role_prior=prior,
                bench_bonus=round(bench_bonus, 4),
                calibration_adj=cal_adj,
                latency_adj=lat_adj,
                substrate_adj=round(sub_adj, 4),
                verified_passes=n_ver,
                scored_candidates=n_cands,
                wilson_lb=w_lb,
                overconfidence=overconf_val,
                median_latency_s=med_lat,
                demoted=demoted,
                reasons=tuple(reasons),
            )
        )

    scored.sort(
        key=lambda c: (
            not c.demoted,
            c.total_score,
            c.verified_passes,
            -c.param_b if is_edge_low_ram else c.param_b,
            c.model,
        ),
        reverse=True,
    )
    return scored


def select_for_role(
    role: str,
    *,
    installed: Optional[Sequence[str]] = None,
    task_text: str = "",
    avoid_model: Optional[str] = None,
    bench_dir: Optional[str] = None,
    osiris_home: PathArg = None,
) -> Optional[ModelCandidateScore]:
    ranked = rank_models(
        role,
        installed=installed,
        task_text=task_text,
        avoid_model=avoid_model,
        bench_dir=bench_dir,
        osiris_home=osiris_home,
    )
    return ranked[0] if ranked else None


def resolve_role_matrix(
    *,
    installed: Optional[Sequence[str]] = None,
    task_text: str = "",
    bench_dir: Optional[str] = None,
    osiris_home: PathArg = None,
) -> Dict[str, Optional[ModelCandidateScore]]:
    """Compute coordinated assignments across `architect`, `code`, `critic` (cross-family), and `voice`."""
    arch = select_for_role("architect", installed=installed, task_text=task_text, bench_dir=bench_dir, osiris_home=osiris_home)
    code = select_for_role("code", installed=installed, task_text=task_text, bench_dir=bench_dir, osiris_home=osiris_home)
    critic = select_for_role(
        "critic",
        installed=installed,
        task_text=task_text,
        avoid_model=code.model if code else None,
        bench_dir=bench_dir,
        osiris_home=osiris_home,
    )
    voice = select_for_role("voice", installed=installed, task_text=task_text, bench_dir=bench_dir, osiris_home=osiris_home)
    return {"architect": arch, "code": code, "critic": critic, "voice": voice}


def format_routing_overview(
    *,
    task_text: str = "",
    role_filter: str = "",
    installed: Optional[Sequence[str]] = None,
    bench_dir: Optional[str] = None,
    osiris_home: PathArg = None,
) -> str:
    matrix = resolve_role_matrix(
        installed=installed, task_text=task_text, bench_dir=bench_dir, osiris_home=osiris_home
    )
    if all(v is None for v in matrix.values()):
        return "MODEL ROUTER · 0 local Ollama models reachable (start `ollama serve` or configure Gemini)."

    cloud_status = "unconfigured"
    try:
        import gemini_bridge

        if gemini_bridge.is_configured():
            burst_on = os.environ.get("OSIRIS_CLOUD_BURST", "0").strip() == "1"
            cloud_status = "configured (burst=ON)" if burst_on else "configured (local-first; set OSIRIS_CLOUD_BURST=1 for auto-burst)"
    except Exception:
        pass

    lines = [
        "EVIDENCE-WEIGHTED MODEL ROUTER" + (f" · task={task_text[:48]!r}" if task_text else ""),
        f"  cloud provider (gemini): {cloud_status}",
        "  active role matrix:",
    ]
    for rname in ("architect", "code", "critic", "voice"):
        pick = matrix.get(rname)
        if pick is None:
            continue
        why = "; ".join(pick.reasons[:3])
        lines.append(f"    {rname:<10} -> {pick.model:<20} score={pick.total_score:5.2f}  ({why})")

    target_role = role_filter if role_filter in ROLES else "code"
    ranked = rank_models(
        target_role,
        installed=installed,
        task_text=task_text,
        avoid_model=(matrix["code"].model if target_role == "critic" and matrix.get("code") else None),
        bench_dir=bench_dir,
        osiris_home=osiris_home,
    )
    lines.append(f"  ranked `{target_role}` fallback chain ({len(ranked)} installed):")
    for idx, c in enumerate(ranked, 1):
        bench_str = f"{c.verified_passes}/{c.scored_candidates}" if c.scored_candidates else "unmeasured"
        lat_str = f"{c.median_latency_s:.1f}s" if c.median_latency_s is not None else "n/a"
        over_str = f"{c.overconfidence:+.2f}" if c.overconfidence is not None else "n/a"
        flag = " [DEMOTED]" if c.demoted else ""
        lines.append(
            f"    {idx}. {c.model:<20} score={c.total_score:5.2f} · bench={bench_str:<6} · "
            f"overconf={over_str:<6} · lat={lat_str}{flag}"
        )
    return "\n".join(lines)


def run_command(args: str = "", *, osiris_home: PathArg = None) -> str:
    """CLI handler for `/models` and `/route [architect|code|critic|voice] [task...]`."""
    try:
        tokens = shlex.split(args or "")
    except ValueError:
        tokens = (args or "").strip().split()

    role_filter = ""
    if tokens and tokens[0].lower() in ("architect", "code", "synthesizer", "critic", "voice", "fast"):
        head = tokens[0].lower()
        role_filter = "code" if head == "synthesizer" else ("voice" if head == "fast" else head)
        tokens = tokens[1:]

    task_text = " ".join(tokens).strip()
    return format_routing_overview(
        task_text=task_text,
        role_filter=role_filter,
        osiris_home=osiris_home,
    )
