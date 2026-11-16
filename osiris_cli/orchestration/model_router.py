"""OSIRIS chooses which model answers a role, and records which one actually did.

A route policy lists, per role, the candidates in order. A candidate is a runtime and a model:

    ollama:gemma3:4b      a local model served by Ollama on this machine
    gemini                Google's hosted model, through ``osiris_cli.gemini_gateway``

The rules are code, not a prompt:

  * A cloud candidate is eligible only when the caller passed ``cloud_allowed=True``. Without it
    the request stays on this machine, and an unavailable local model is reported as unavailable.
    It never turns into a cloud call.
  * A local model must be installed under exactly the requested name. A missing model is
    ``MODEL_UNAVAILABLE``; no other installed model is substituted for it.
  * The next candidate is tried only because the policy lists it. Every candidate that was
    skipped or failed is recorded with its reason.
  * The route record keeps the requested model and the resolved model apart, with the local
    model's digest, the runtime, and whether the request left the device.

This reuses the existing provider contract (``providers.base``), the Ollama adapter
(``providers.ollama_compat``) and the Gemini gateway (redaction, token budget, call ledger).
It selects a model. It grants nothing: no route gives a model authority to execute.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

POLICY_VERSION = "osiris-model-routes/v1"

# Starting choices for a laptop with one 8 GB GPU: one small local model for proposing and
# briefing, one local reasoning model for review. They are defaults, not benchmark winners.
DEFAULT_ROUTES: Dict[str, List[str]] = {
    "AURA": ["ollama:gemma3:4b", "gemini"],
    "AIDEN": ["ollama:qwen3:8b", "gemini"],
    "IRIS": ["ollama:gemma3:4b", "gemini"],
}

OLLAMA_TIMEOUT = 300.0
OLLAMA_CONTEXT = 8192   # tokens; the council's review and brief prompts carry the earlier roles' text


class RouteUnavailable(Exception):
    """No eligible candidate produced text. ``route`` holds what was tried and why."""

    def __init__(self, message: str, route: Dict[str, Any]):
        super().__init__(message)
        self.route = route


def load_routes(path: Optional[str] = None) -> Dict[str, List[str]]:
    """The route table: ``OSIRIS_MODEL_ROUTES`` (a JSON file ``{"ROLE": ["candidate", ...]}``)
    when set, else the defaults. A malformed file is an error, not a silent return to defaults."""
    path = path or os.environ.get("OSIRIS_MODEL_ROUTES", "")
    if not path:
        return {k: list(v) for k, v in DEFAULT_ROUTES.items()}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or not all(
            isinstance(k, str) and isinstance(v, list) and v and all(isinstance(c, str) for c in v)
            for k, v in data.items()):
        raise ValueError(f"{path}: expected {{role: [candidate, ...]}}")
    for cands in data.values():
        for c in cands:
            parse_candidate(c)
    return data


def parse_candidate(candidate: str) -> Tuple[str, Optional[str]]:
    """``ollama:<model>`` -> ("ollama", model); ``gemini`` or ``gemini:<model>`` -> ("gemini", model)."""
    runtime, _, model = candidate.partition(":")
    if runtime == "ollama" and model:
        return "ollama", model
    if runtime == "gemini":
        return "gemini", model or None
    raise ValueError(f"unknown route candidate {candidate!r}")


def ollama_installed(host: Optional[str] = None, timeout: float = 3.0) -> Dict[str, str]:
    """Installed Ollama models, name -> digest. Raises when the service does not answer."""
    from ..providers.ollama_compat import ollama_base_url
    with urllib.request.urlopen(f"{ollama_base_url(host)}/api/tags", timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return {m.get("name", ""): m.get("digest", "") for m in data.get("models", [])}


def _call_ollama(model: str, system: str, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
    from ..providers.base import Message, ProviderRequest
    from ..providers.ollama_compat import OllamaCompatibilityProvider
    provider = OllamaCompatibilityProvider(model=model, timeout=OLLAMA_TIMEOUT)
    resp = provider.chat(ProviderRequest(messages=[Message("user", prompt)], model=model, system=system,
                                         temperature=temperature, max_tokens=max_tokens,
                                         timeout=OLLAMA_TIMEOUT, think=False, num_ctx=OLLAMA_CONTEXT))
    if resp.error:
        raise RuntimeError(resp.error)
    return {"text": resp.content, "model": resp.model, "is_mock": bool(getattr(resp, "is_mock", False))}


def _call_gemini(model: Optional[str], system: str, prompt: str, max_tokens: int, temperature: float,
                 purpose: str) -> Dict[str, Any]:
    from .. import gemini_gateway
    out = gemini_gateway.generate(prompt, purpose=purpose, model=model, system=system,
                                  max_output_tokens=max_tokens, temperature=temperature)
    return {"text": out["text"], "model": f"{out['backend']}:{out.get('model_version') or model or 'gemini'}"}


def route(role: str, system: str, prompt: str, *, cloud_allowed: bool = False,
          routes: Optional[Dict[str, List[str]]] = None, max_tokens: int = 1500, temperature: float = 0.3,
          installed: Optional[Callable[[], Dict[str, str]]] = None,
          call_ollama: Optional[Callable[..., Dict[str, Any]]] = None,
          call_gemini: Optional[Callable[..., Dict[str, Any]]] = None) -> Tuple[str, Dict[str, Any]]:
    """Run the first eligible candidate for ``role`` that answers. Returns (text, route record);
    raises RouteUnavailable (carrying the record) when none does."""
    routes = load_routes() if routes is None else routes
    candidates = routes.get(role)
    if not candidates:
        raise RouteUnavailable(f"no route is configured for role {role}",
                               {"policy": POLICY_VERSION, "role": role, "status": "NO_ROUTE",
                                "cloud_allowed": bool(cloud_allowed), "attempts": []})
    installed = installed or ollama_installed
    call_ollama = call_ollama or _call_ollama
    call_gemini = call_gemini or _call_gemini
    attempts: List[Dict[str, Any]] = []
    local: Optional[Dict[str, str]] = None
    local_error: Optional[str] = None

    for index, candidate in enumerate(candidates):
        runtime, model = parse_candidate(candidate)
        attempt: Dict[str, Any] = {"candidate": candidate, "runtime": runtime, "requested_model": model}
        attempts.append(attempt)
        if runtime == "gemini":
            if not cloud_allowed:
                attempt.update(status="NOT_PERMITTED", reason="cloud use was not allowed for this request")
                continue
            try:
                out = call_gemini(model, system, prompt, max_tokens, temperature, f"agents.task/{role}")
            except Exception as e:  # noqa: BLE001 - recorded, then the next listed candidate
                attempt.update(status="FAILED", reason=f"{type(e).__name__}: {str(e)[:200]}")
                continue
            digest = None
        else:
            if local is None and local_error is None:
                try:
                    local = installed()
                except Exception as e:  # noqa: BLE001
                    local_error = f"Ollama is not reachable: {type(e).__name__}: {str(e)[:120]}"
            if local is None:
                attempt.update(status="RUNTIME_UNAVAILABLE", reason=local_error)
                continue
            if model not in local:
                attempt.update(status="MODEL_UNAVAILABLE",
                               reason=f"{model} is not installed in Ollama; no other model was substituted")
                continue
            try:
                out = call_ollama(model, system, prompt, max_tokens, temperature)
            except Exception as e:  # noqa: BLE001
                attempt.update(status="FAILED", reason=f"{type(e).__name__}: {str(e)[:200]}")
                continue
            if out.get("model") != model:
                attempt.update(status="FAILED",
                               reason=f"the runtime answered as {out.get('model')!r}, not the requested {model!r}")
                continue
            digest = local[model] or None
        if out.get("is_mock") is True:
            attempt.update(status="FAILED", reason="mock output cannot masquerade as live model output")
            continue
        text = (out.get("text") or "").strip()
        if not text:
            attempt.update(status="FAILED", reason="the model returned no text")
            continue
        if "[gemma:mock" in text.lower() or "deterministic mock response" in text.lower():
            attempt.update(status="FAILED", reason="mock output cannot masquerade as live model output")
            continue
        attempt.update(status="ANSWERED")
        return text, {
            "policy": POLICY_VERSION, "role": role, "status": "ANSWERED", "cloud_allowed": bool(cloud_allowed),
            "runtime": runtime, "requested_model": model, "resolved_model": out.get("model"),
            "resolved_digest": digest, "left_device": runtime == "gemini",
            "fallback_used": index > 0,
            "reason": "first listed candidate" if index == 0 else
                      f"candidate {index + 1} in the route policy; earlier candidates are in attempts",
            "attempts": attempts,
        }

    record = {"policy": POLICY_VERSION, "role": role, "status": "MODEL_UNAVAILABLE",
              "cloud_allowed": bool(cloud_allowed), "runtime": None, "requested_model": parse_candidate(candidates[0])[1],
              "resolved_model": None, "resolved_digest": None, "left_device": False, "fallback_used": False,
              "attempts": attempts}
    raise RouteUnavailable("; ".join(f"{a['candidate']}: {a['status']} ({a.get('reason')})" for a in attempts), record)
