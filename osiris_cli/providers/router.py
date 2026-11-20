"""Telemetry-governed deterministic MentorRouter for OSIRIS v6.0.0 (L4 / L6 / L7).

Enforces:
- "Models propose; deterministic code decides, measures, and records."
- Dynamic model discovery against installed Ollama tags with explicit legacy alias
  normalization (never silently substituting an unavailable model).
- Hardware-aware routing across Node Alpha (Termux ARM64 / low RAM) and Node Beta
  (x86_64 workstation), gating cloud bursting behind explicit authorization + quota.
- Continual-learning protection: every 5th exchange (exchange_idx % 5 == 0) is strictly
  held out, and degraded low-RAM fallbacks set allow_distillation=False so noisy replies
  never poison osiris.nclm.
- Non-blocking shadow bits/byte scoring helper for held-out progression tracking.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from osiris_cli.probes import probe_substrate

SOVEREIGN_LOCAL_MODELS: Tuple[str, ...] = (
    "qwen2.5:7b",
    "qwen2.5-coder:7b",
    "qwen3:8b",
    "qwen2.5:3b",
)

EDGE_LIGHTWEIGHT_MODELS: Tuple[str, ...] = (
    "gemma3:1b",
    "gemma2:2b",
    "qwen2.5:1.5b",
    "gemma3:4b",
    "llama3.2:1b",
    "llama3.2:3b",
    "smollm2:360m",
)

CLOUD_BURST_MODEL = "gemini-2.5-flash"

LEGACY_TAG_ALIASES: Dict[str, str] = {
    "gemma-3-1b": "gemma3:1b",
    "gemma-3-4b": "gemma3:4b",
    "gemma-3-270m": "gemma3:270m",
    "gemma-2-2b": "gemma2:2b",
}

EDGE_MIN_RAM_MB = 3584
HIGH_TOKEN_THRESHOLD = 1400
HIGH_CPU_LOAD_1M = 4.0
HELDOUT_EVERY = 5


@dataclass(frozen=True)
class SystemTelemetry:
    free_ram_mb: int
    cpu_load_1m: float
    is_termux_edge: bool
    ollama_reachable: bool
    cloud_quota_ok: bool
    cloud_authorized: bool = False
    node_id: str = "NODE_BETA_WORKSTATION"


@dataclass(frozen=True)
class RouteDecision:
    mentor_id: Optional[str]
    provider: str
    reason: str
    allow_distillation: bool
    is_held_out_turn: bool
    degraded_fallback: bool = False
    unavailable_reason: Optional[str] = None

    @property
    def tier(self) -> Optional[str]:
        """Alias for mentor_id for callers using tier terminology."""
        return self.mentor_id

    @property
    def available(self) -> bool:
        return self.mentor_id is not None

    def to_ledger_dict(
        self,
        core_shadow_bpb: Optional[float] = None,
        unigram_bpb: Optional[float] = None,
    ) -> Dict[str, Any]:
        rec: Dict[str, Any] = {
            "mentor_id": self.mentor_id,
            "provider": self.provider,
            "reason": self.reason,
            "distill_eligible": self.allow_distillation,
            "held_out": self.is_held_out_turn,
            "degraded_fallback": self.degraded_fallback,
        }
        if self.unavailable_reason:
            rec["unavailable_reason"] = self.unavailable_reason
        if core_shadow_bpb is not None:
            rec["core_shadow_bpb"] = round(float(core_shadow_bpb), 4)
        if unigram_bpb is not None:
            rec["unigram_bpb"] = round(float(unigram_bpb), 4)
        return rec


def normalize_ollama_tag(model: str) -> str:
    """Normalize legacy hyphenated tags (e.g. gemma-3-1b -> gemma3:1b)."""
    cleaned = (model or "").strip()
    if not cleaned:
        return ""
    return LEGACY_TAG_ALIASES.get(cleaned.lower(), cleaned)


def match_installed_model(target: str, installed_models: Sequence[str]) -> Optional[str]:
    """Resolve `target` against `installed_models` via exact match, `:latest`, or explicit legacy alias.

    Never prefix-matches bare model families (e.g. 'gemma3' does not match 'gemma3:1b').
    """
    if not target or not installed_models:
        return None
    installed_set = {m.strip() for m in installed_models if m and m.strip()}
    if target in installed_set:
        return target
    if f"{target}:latest" in installed_set:
        return f"{target}:latest"
    canonical = normalize_ollama_tag(target)
    if canonical != target:
        if canonical in installed_set:
            return canonical
        if f"{canonical}:latest" in installed_set:
            return f"{canonical}:latest"
    return None


def discover_ollama_models(
    ollama_base: Optional[str] = None,
    timeout: float = 1.5,
) -> Tuple[bool, List[str]]:
    """Query local Ollama `/api/tags` and return `(reachable, sorted_model_names)`."""
    base = (ollama_base or os.environ.get("OSIRIS_OLLAMA") or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").strip().rstrip("/")
    if "://" not in base:
        base = "http://" + base
    try:
        req = urllib.request.Request(f"{base}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        models = sorted(
            str(m["name"])
            for m in data.get("models", [])
            if isinstance(m, dict) and m.get("name")
        )
        return True, models
    except Exception:  # noqa: BLE001
        return False, []


def collect_system_telemetry(
    ollama_base: Optional[str] = None,
    cloud_authorized: Optional[bool] = None,
    cloud_quota_ok: Optional[bool] = None,
) -> Tuple[SystemTelemetry, List[str]]:
    """Gather deterministic hardware + Ollama + cloud-policy telemetry."""
    sub = probe_substrate()
    ollama_ok, installed = discover_ollama_models(ollama_base=ollama_base)
    if cloud_authorized is None:
        cloud_authorized = os.environ.get("OSIRIS_CLOUD_BURST", "0").strip() == "1"
    if cloud_quota_ok is None:
        try:
            import gemini_bridge

            cloud_quota_ok = bool(gemini_bridge.is_configured())
        except Exception:  # noqa: BLE001
            cloud_quota_ok = False
    hw = SystemTelemetry(
        free_ram_mb=sub.free_ram_mb,
        cpu_load_1m=sub.cpu_load_1m,
        is_termux_edge=sub.is_termux,
        ollama_reachable=ollama_ok,
        cloud_quota_ok=bool(cloud_quota_ok),
        cloud_authorized=bool(cloud_authorized),
        node_id=sub.node_id,
    )
    return hw, installed


def _first_installed(preferred: Iterable[str], installed: Sequence[str]) -> Optional[str]:
    for cand in preferred:
        matched = match_installed_model(cand, installed)
        if matched is not None:
            return matched
    return None


def select_mentor(
    exchange_idx: int,
    prompt_tokens: int,
    requires_multi_hop: bool,
    hw_telemetry: SystemTelemetry,
    installed_ollama_models: Sequence[str] = (),
    manual_override: Optional[str] = None,
) -> RouteDecision:
    """Deterministically select a mentor model and distillation eligibility.

    Never silently substitutes an unavailable manual override or pretends an
    uninstalled model is active.
    """
    held_out = (int(exchange_idx) % HELDOUT_EVERY == 0)
    installed = [m for m in installed_ollama_models if m]

    # 0. Explicit manual override (/mentor NAME): honor or fail closed (no silent substitution).
    if manual_override:
        override = manual_override.strip()
        if override.lower().startswith("gemini"):
            if hw_telemetry.cloud_quota_ok:
                return RouteDecision(
                    mentor_id=override,
                    provider="gemini",
                    reason="manual_override",
                    allow_distillation=not held_out,
                    is_held_out_turn=held_out,
                    degraded_fallback=False,
                )
            return RouteDecision(
                mentor_id=None,
                provider="gemini",
                reason="manual_override_unavailable",
                allow_distillation=False,
                is_held_out_turn=held_out,
                degraded_fallback=False,
                unavailable_reason=f"Requested cloud mentor '{override}' is unavailable (cloud quota/config unavailable).",
            )
        if not hw_telemetry.ollama_reachable:
            return RouteDecision(
                mentor_id=None,
                provider="ollama",
                reason="manual_override_daemon_offline",
                allow_distillation=False,
                is_held_out_turn=held_out,
                degraded_fallback=False,
                unavailable_reason=f"Requested local mentor '{override}' is unavailable: Ollama daemon is offline.",
            )
        matched = match_installed_model(override, installed)
        if matched is None:
            return RouteDecision(
                mentor_id=None,
                provider="ollama",
                reason="manual_override_not_installed",
                allow_distillation=False,
                is_held_out_turn=held_out,
                degraded_fallback=False,
                unavailable_reason=(
                    f"Requested Ollama mentor '{override}' is not installed "
                    f"(installed: {', '.join(sorted(installed)) or 'none'})."
                ),
            )
        is_degraded = (hw_telemetry.is_termux_edge or hw_telemetry.free_ram_mb < EDGE_MIN_RAM_MB) and matched not in SOVEREIGN_LOCAL_MODELS
        return RouteDecision(
            mentor_id=matched,
            provider="ollama",
            reason="manual_override",
            allow_distillation=(not held_out) and (not is_degraded),
            is_held_out_turn=held_out,
            degraded_fallback=is_degraded,
        )

    # 1. Node Alpha / Low-RAM Edge Guard (< 3.5 GiB free or Termux)
    if hw_telemetry.is_termux_edge or hw_telemetry.free_ram_mb < EDGE_MIN_RAM_MB:
        if requires_multi_hop and hw_telemetry.cloud_authorized and hw_telemetry.cloud_quota_ok:
            return RouteDecision(
                mentor_id=CLOUD_BURST_MODEL,
                provider="gemini",
                reason="edge_complex_burst",
                allow_distillation=not held_out,
                is_held_out_turn=held_out,
                degraded_fallback=False,
            )
        if hw_telemetry.ollama_reachable and installed:
            edge_pick = _first_installed(EDGE_LIGHTWEIGHT_MODELS, installed) or sorted(installed)[0]
            return RouteDecision(
                mentor_id=edge_pick,
                provider="ollama",
                reason="edge_low_ram",
                allow_distillation=False,
                is_held_out_turn=held_out,
                degraded_fallback=True,
            )
        if hw_telemetry.cloud_authorized and hw_telemetry.cloud_quota_ok:
            return RouteDecision(
                mentor_id=CLOUD_BURST_MODEL,
                provider="gemini",
                reason="edge_cloud_fallback_authorized",
                allow_distillation=not held_out,
                is_held_out_turn=held_out,
                degraded_fallback=False,
            )
        return RouteDecision(
            mentor_id=None,
            provider="none",
            reason="edge_no_reachable_mentor",
            allow_distillation=False,
            is_held_out_turn=held_out,
            degraded_fallback=False,
            unavailable_reason="No installed edge Ollama model reachable and cloud burst is not authorized/available.",
        )

    # 2. Node Beta Workstation: load-shed cloud burst if authorized + overloaded + heavy query
    if (
        (prompt_tokens > HIGH_TOKEN_THRESHOLD or requires_multi_hop)
        and hw_telemetry.cpu_load_1m > HIGH_CPU_LOAD_1M
        and hw_telemetry.cloud_authorized
        and hw_telemetry.cloud_quota_ok
    ):
        return RouteDecision(
            mentor_id=CLOUD_BURST_MODEL,
            provider="gemini",
            reason="load_shed_burst",
            allow_distillation=not held_out,
            is_held_out_turn=held_out,
            degraded_fallback=False,
        )

    # 3. Node Beta Sovereign Local Default (7B/3B+)
    if hw_telemetry.ollama_reachable and installed:
        sovereign_pick = _first_installed(SOVEREIGN_LOCAL_MODELS, installed)
        if sovereign_pick is not None:
            return RouteDecision(
                mentor_id=sovereign_pick,
                provider="ollama",
                reason="sovereign_default",
                allow_distillation=not held_out,
                is_held_out_turn=held_out,
                degraded_fallback=False,
            )
        # 4. Degraded Local Fallback (distillation blocked so small-model replies don't poison core)
        small_pick = _first_installed(EDGE_LIGHTWEIGHT_MODELS, installed) or sorted(installed)[0]
        return RouteDecision(
            mentor_id=small_pick,
            provider="ollama",
            reason="degraded_fallback",
            allow_distillation=False,
            is_held_out_turn=held_out,
            degraded_fallback=True,
        )

    # 5. Ollama unreachable: authorized cloud fallback or explicit fail-closed
    if hw_telemetry.cloud_authorized and hw_telemetry.cloud_quota_ok:
        return RouteDecision(
            mentor_id=CLOUD_BURST_MODEL,
            provider="gemini",
            reason="cloud_fallback_authorized",
            allow_distillation=not held_out,
            is_held_out_turn=held_out,
            degraded_fallback=False,
        )
    return RouteDecision(
        mentor_id=None,
        provider="none",
        reason="no_reachable_mentor",
        allow_distillation=False,
        is_held_out_turn=held_out,
        degraded_fallback=False,
        unavailable_reason="Ollama is unreachable (or has no models installed) and cloud burst is not authorized/available.",
    )


class MentorRouter:
    """Stateful telemetry-aware mentor router with failure tracking and async shadow scoring."""

    def __init__(
        self,
        telemetry_provider: Optional[Callable[[], Tuple[SystemTelemetry, List[str]]]] = None,
    ) -> None:
        self._telemetry_provider = telemetry_provider or collect_system_telemetry
        self._failed_models: Dict[str, str] = {}

    def record_failure(self, mentor_id: str, error: str) -> None:
        """Record a runtime provider failure so subsequent fallback selection avoids it honestly."""
        if mentor_id:
            self._failed_models[mentor_id] = str(error)

    def clear_failures(self) -> None:
        self._failed_models.clear()

    def route(
        self,
        exchange_idx: int,
        prompt_tokens: int,
        requires_multi_hop: bool = False,
        manual_override: Optional[str] = None,
        hw_override: Optional[SystemTelemetry] = None,
        installed_override: Optional[Sequence[str]] = None,
    ) -> RouteDecision:
        if hw_override is not None:
            hw = hw_override
            installed = list(installed_override or ())
        else:
            hw, installed = self._telemetry_provider()

        healthy_installed = [m for m in installed if m not in self._failed_models]
        if manual_override:
            # Explicit manual override always checks full installed set without silent substitution.
            return select_mentor(
                exchange_idx=exchange_idx,
                prompt_tokens=prompt_tokens,
                requires_multi_hop=requires_multi_hop,
                hw_telemetry=hw,
                installed_ollama_models=installed,
                manual_override=manual_override,
            )

        decision = select_mentor(
            exchange_idx=exchange_idx,
            prompt_tokens=prompt_tokens,
            requires_multi_hop=requires_multi_hop,
            hw_telemetry=hw,
            installed_ollama_models=healthy_installed,
            manual_override=None,
        )
        if self._failed_models and decision.mentor_id is not None:
            failed_note = ",".join(sorted(self._failed_models))
            return RouteDecision(
                mentor_id=decision.mentor_id,
                provider=decision.provider,
                reason=f"{decision.reason}_after_failure({failed_note})",
                allow_distillation=False if decision.degraded_fallback else decision.allow_distillation,
                is_held_out_turn=decision.is_held_out_turn,
                degraded_fallback=decision.degraded_fallback,
            )
        return decision

    @staticmethod
    async def score_shadow_bpb_async(
        core: Any,
        text: str,
        unigram_fn: Optional[Callable[[str], float]] = None,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Non-blocking background shadow bpb scoring via asyncio.to_thread without touching weights."""
        if core is None or not text:
            return None, None
        core_bpb = await asyncio.to_thread(core.bits_per_byte, text)
        uni_bpb = await asyncio.to_thread(unigram_fn, text) if callable(unigram_fn) else None
        return (
            round(float(core_bpb), 4) if core_bpb is not None else None,
            round(float(uni_bpb), 4) if uni_bpb is not None else None,
        )
