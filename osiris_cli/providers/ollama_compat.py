"""
Ollama Compatibility Provider
=============================
Optional backwards-compatibility adapter preserving existing Ollama paths
(local Qwen/DeepSeek) as secondary backends without making Ollama an
architectural prerequisite for the OSIRIS core.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Iterator, Optional

from .base import (
    BaseModelProvider,
    ExecutionPlacement,
    FinishReason,
    HealthStatus,
    ProviderCapabilities,
    ProviderHealth,
    ProviderRequest,
    ProviderResponse,
    ProviderStreamChunk,
)

DEFAULT_OLLAMA_URL = "http://localhost:11434"


def ollama_base_url(raw: Optional[str] = None) -> str:
    """A client URL from OLLAMA_HOST, which follows the server's convention: it may be a bare
    host, host:port, or a bind address such as 0.0.0.0 (listen everywhere) -- none of which is
    a URL a client can call as-is."""
    raw = (os.environ.get("OLLAMA_HOST", "") if raw is None else raw).strip().rstrip("/")
    if not raw:
        return DEFAULT_OLLAMA_URL
    if "://" not in raw:
        raw = "http://" + raw
    scheme, rest = raw.split("://", 1)
    hostport, sep, path = rest.partition("/")
    if hostport.startswith("["):
        host, _, port = hostport[1:].partition("]")
        port = port.lstrip(":")
        host = f"[{host}]"
    else:
        host, _, port = hostport.partition(":")
    if host in ("", "0.0.0.0", "[::]"):
        host = "127.0.0.1"
    port = port or ("443" if scheme == "https" else "11434")
    return f"{scheme}://{host}:{port}" + (f"/{path}" if sep and path else "")


def _diagnose_ollama_exception(prefix: str, host: str, model: str, exc: Exception) -> str:
    """Classify Ollama HTTP/transport errors without hiding 404 model-missing vs endpoint vs daemon faults."""
    if isinstance(exc, urllib.error.HTTPError):
        raw_body = ""
        try:
            raw_body = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            raw_body = ""
        detail = raw_body
        if raw_body:
            try:
                parsed = json.loads(raw_body)
                if isinstance(parsed, dict) and isinstance(parsed.get("error"), str):
                    detail = parsed["error"].strip()
            except Exception:
                pass
        if exc.code == 404:
            low = detail.lower()
            if "not found" in low and ("model" in low or model.lower() in low):
                return (
                    f"{prefix}: HTTP 404 MODEL_NOT_INSTALLED: model {model!r} is not installed on Ollama "
                    f"at {host} ({detail})"
                )
            return (
                f"{prefix}: HTTP 404 ENDPOINT_INCOMPATIBLE at {host} "
                f"({detail or exc.reason})"
            )
        return f"{prefix}: HTTP {exc.code} at {host} ({detail or exc.reason})"
    if isinstance(exc, urllib.error.URLError):
        return f"{prefix}: DAEMON_UNAVAILABLE at {host} ({exc.reason})"
    return f"{prefix}: {exc}"


class OllamaCompatibilityProvider(BaseModelProvider):
    """Wraps Ollama HTTP API conforming to BaseModelProvider."""

    def __init__(self, host: Optional[str] = None, model: str = "qwen2.5:1.5b", timeout: float = 60.0):
        self.host = ollama_base_url(host)
        self.selected_model = model
        self.timeout = timeout
        self.cancelled_requests: set[str] = set()

    def list_installed_models(self, timeout: float = 3.0) -> list[dict]:
        """Query /api/tags and return raw model manifest dicts from the running Ollama daemon."""
        req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=min(timeout, self.timeout)) as resp:
            if resp.getcode() != 200:
                return []
            data = json.loads(resp.read().decode("utf-8"))
        return [m for m in data.get("models", []) if isinstance(m, dict) and m.get("name")]

    def installed_model_names(self, timeout: float = 3.0) -> list[str]:
        """Return installed model names from /api/tags, or [] if the daemon is offline."""
        try:
            return [str(m["name"]) for m in self.list_installed_models(timeout=timeout)]
        except Exception:
            return []

    def discover_capabilities(self, timeout: float = 3.0) -> ProviderCapabilities:
        """Probe /api/tags to report actual installed models, digests, context length, and tool support."""
        try:
            entries = self.list_installed_models(timeout=timeout)
        except Exception as exc:
            return ProviderCapabilities(
                provider_name="ollama_compat",
                supports_streaming=True,
                supports_structured_output=True,
                supports_tool_calling=True,
                supports_token_count=True,
                supported_models=[],
                max_context_length=4096,
                default_placement=ExecutionPlacement.WORKSTATION,
                metadata={"host": self.host, "reachable": False, "error": str(exc)},
            )
        names = [str(m["name"]) for m in entries]
        digests = {str(m["name"]): str(m.get("digest", "")) for m in entries}
        ctx_lengths = [
            int(m.get("details", {}).get("context_length", 0) or 0)
            for m in entries
            if isinstance(m.get("details"), dict)
        ]
        has_tools = any(
            "tools" in (m.get("capabilities") or [])
            for m in entries
        ) or bool(names)
        return ProviderCapabilities(
            provider_name="ollama_compat",
            supports_streaming=True,
            supports_structured_output=True,
            supports_tool_calling=has_tools,
            supports_token_count=True,
            supported_models=names,
            max_context_length=max(ctx_lengths) if any(ctx_lengths) else 8192,
            default_placement=ExecutionPlacement.WORKSTATION,
            metadata={"host": self.host, "reachable": True, "installed_digests": digests},
        )

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_name="ollama_compat",
            supports_streaming=True,
            supports_structured_output=True,
            supports_tool_calling=True,
            supports_token_count=True,
            supported_models=["qwen3:8b", "gemma3:4b", "qwen2.5-coder:7b", "qwen2.5:1.5b"],
            max_context_length=8192,
            default_placement=ExecutionPlacement.WORKSTATION,
            metadata={"host": self.host},
        )

    def health(self) -> ProviderHealth:
        t0 = time.monotonic()
        url = f"{self.host}/api/tags"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=min(3.0, self.timeout)) as resp:
                if resp.getcode() == 200:
                    data = json.loads(resp.read().decode())
                    models = [m.get("name", "") for m in data.get("models", [])]
                    return ProviderHealth(
                        status=HealthStatus.HEALTHY,
                        available=True,
                        provider_name="ollama_compat",
                        selected_model=self.selected_model,
                        message=f"Ollama reachable at {self.host} with {len(models)} model(s)",
                        details={"installed_models": models},
                        latency_ms=(time.monotonic() - t0) * 1000,
                    )
                return ProviderHealth(
                    status=HealthStatus.DEGRADED,
                    available=False,
                    provider_name="ollama_compat",
                    message=f"Ollama returned HTTP {resp.getcode()}",
                )
        except urllib.error.URLError as e:
            return ProviderHealth(
                status=HealthStatus.UNAVAILABLE,
                available=False,
                provider_name="ollama_compat",
                message=f"Ollama service unreachable at {self.host} ({e.reason})",
            )
        except Exception as e:
            return ProviderHealth(
                status=HealthStatus.UNAVAILABLE,
                available=False,
                provider_name="ollama_compat",
                message=f"Ollama check error: {e}",
            )

    def cancel(self, request_id: str) -> bool:
        self.cancelled_requests.add(request_id)
        return True

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        t0 = time.monotonic()
        if request.request_id in self.cancelled_requests:
            return ProviderResponse(
                content="", model=self.selected_model, provider="ollama_compat",
                request_id=request.request_id, finish_reason=FinishReason.CANCELLED, error="Cancelled"
            )

        model = request.model or self.selected_model
        payload = {
            "model": model,
            "prompt": request.get_full_prompt(),
            "stream": False,
            "options": {
                "num_predict": request.max_tokens,
                "temperature": request.temperature,
            },
        }
        if request.system:
            payload["system"] = request.system
        if request.response_schema:
            payload["format"] = request.response_schema

        body = json.dumps(payload).encode()
        req = urllib.request.Request(f"{self.host}/api/generate", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=request.timeout) as resp:
                data = json.loads(resp.read().decode())
                content = data.get("response", "")
                duration = (time.monotonic() - t0) * 1000.0
                eval_count = data.get("eval_count", 0)

                structured_out = None
                if request.response_schema and content:
                    try:
                        structured_out = json.loads(content)
                    except Exception:
                        pass

                return ProviderResponse(
                    content=content,
                    model=model,
                    provider="ollama_compat",
                    request_id=request.request_id,
                    finish_reason=FinishReason.STOP if data.get("done") else FinishReason.LENGTH,
                    eval_count=eval_count,
                    prompt_eval_count=data.get("prompt_eval_count", 0),
                    total_duration_ms=duration,
                    structured_output=structured_out,
                )
        except Exception as e:
            return ProviderResponse(
                content="",
                model=model,
                provider="ollama_compat",
                request_id=request.request_id,
                finish_reason=FinishReason.ERROR,
                error=_diagnose_ollama_exception("Ollama call failed", self.host, model, e),
                total_duration_ms=(time.monotonic() - t0) * 1000.0,
            )

    def chat(self, request: ProviderRequest) -> ProviderResponse:
        """Multi-turn /api/chat. ``request.tools`` are offered to the model and any native
        tool calls come back in ``ProviderResponse.tool_calls``. A message's metadata may
        carry ``tool_calls`` (assistant turns) or ``name`` (tool results). Never raises."""
        t0 = time.monotonic()
        model = request.model or self.selected_model
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        for m in request.messages:
            entry = {"role": m.role, "content": m.content}
            entry.update({k: v for k, v in (m.metadata or {}).items() if k in ("tool_calls", "name")})
            messages.append(entry)
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": request.max_tokens, "temperature": request.temperature},
        }
        if request.tools:
            payload["tools"] = request.tools
        if request.think is not None:
            payload["think"] = request.think
        if request.num_ctx:
            payload["options"]["num_ctx"] = int(request.num_ctx)

        req = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=request.timeout) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:  # noqa: BLE001 - errors are returned, never raised
            return ProviderResponse(
                content="", model=model, provider="ollama_compat", request_id=request.request_id,
                finish_reason=FinishReason.ERROR,
                error=_diagnose_ollama_exception("Ollama chat failed", self.host, model, e),
                total_duration_ms=(time.monotonic() - t0) * 1000.0,
            )
        message = data.get("message") or {}
        return ProviderResponse(
            content=message.get("content", "") or "",
            model=data.get("model") or model,
            provider="ollama_compat",
            request_id=request.request_id,
            finish_reason=FinishReason.STOP if data.get("done") else FinishReason.LENGTH,
            eval_count=data.get("eval_count", 0),
            prompt_eval_count=data.get("prompt_eval_count", 0),
            total_duration_ms=(time.monotonic() - t0) * 1000.0,
            tool_calls=message.get("tool_calls") or None,
        )

    def stream(self, request: ProviderRequest) -> Iterator[ProviderStreamChunk]:
        resp = self.generate(request)
        yield ProviderStreamChunk(
            text=resp.content,
            is_final=True,
            chunk_index=0,
            request_id=request.request_id,
            finish_reason=resp.finish_reason,
        )
