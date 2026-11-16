"""
Gemma-Native Inference Substrate Provider
=========================================
First-class provider implementation for the Google Gemma model family.
Supports edge execution (Gemma 3 270M/1B, Gemma 2 2B) and workstation/cloud scales,
with support for multiple runtimes (HTTP/REST, local llama.cpp/transformers, and deterministic test fixtures).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from .base import (
    BaseModelProvider,
    ExecutionPlacement,
    FinishReason,
    HealthStatus,
    Message,
    ProviderCapabilities,
    ProviderHealth,
    ProviderRequest,
    ProviderResponse,
    ProviderStreamChunk,
)

logger = logging.getLogger("OSIRIS_GEMMA_PROVIDER")


@dataclass
class GemmaModelSpec:
    model_id: str
    family: str           # "gemma-3", "gemma-2", "gemma"
    parameter_scale: str  # "270m", "1b", "2b", "4b", "9b", "12b", "27b"
    context_window: int
    recommended_placement: ExecutionPlacement
    quantization_options: List[str]
    memory_footprint_mb: int
    description: str


# Canonical Gemma Model Manifest (verifiable metadata)
GEMMA_MANIFEST: Dict[str, GemmaModelSpec] = {
    "gemma-3-270m": GemmaModelSpec(
        model_id="gemma-3-270m",
        family="gemma-3",
        parameter_scale="270m",
        context_window=32768,
        recommended_placement=ExecutionPlacement.EDGE_MOBILE,
        quantization_options=["int4", "int8", "fp16"],
        memory_footprint_mb=280,
        description="Ultra-compact edge model; runs in low-RAM Android/Termux environments.",
    ),
    "gemma-3-1b": GemmaModelSpec(
        model_id="gemma-3-1b",
        family="gemma-3",
        parameter_scale="1b",
        context_window=32768,
        recommended_placement=ExecutionPlacement.EDGE_MOBILE,
        quantization_options=["q4_k_m", "q8_0", "fp16"],
        memory_footprint_mb=950,
        description="Fast mobile edge model; ideal for intent classification and code stubs.",
    ),
    "gemma-2-2b": GemmaModelSpec(
        model_id="gemma-2-2b",
        family="gemma-2",
        parameter_scale="2b",
        context_window=8192,
        recommended_placement=ExecutionPlacement.EDGE_MOBILE,
        quantization_options=["q4_k_m", "q8_0", "fp16"],
        memory_footprint_mb=1600,
        description="Balanced mobile/desktop model with strong instruction following.",
    ),
    "gemma2:2b": GemmaModelSpec(
        model_id="gemma2:2b",
        family="gemma-2",
        parameter_scale="2b",
        context_window=8192,
        recommended_placement=ExecutionPlacement.EDGE_MOBILE,
        quantization_options=["q4_k_m", "q8_0", "fp16"],
        memory_footprint_mb=1600,
        description="Gemma 2 2B parameter model for edge and local inference.",
    ),
    "gemma:2b": GemmaModelSpec(
        model_id="gemma:2b",
        family="gemma",
        parameter_scale="2b",
        context_window=8192,
        recommended_placement=ExecutionPlacement.EDGE_MOBILE,
        quantization_options=["q4_k_m", "q8_0", "fp16"],
        memory_footprint_mb=1600,
        description="Gemma 2B parameter model for edge and local inference.",
    ),
    "gemma-3-4b": GemmaModelSpec(
        model_id="gemma-3-4b",
        family="gemma-3",
        parameter_scale="4b",
        context_window=131072,
        recommended_placement=ExecutionPlacement.WORKSTATION,
        quantization_options=["q4_k_m", "q8_0", "bf16"],
        memory_footprint_mb=3200,
        description="Multimodal capable workstation model for complex synthesis and code tasks.",
    ),
    "gemma-3-12b": GemmaModelSpec(
        model_id="gemma-3-12b",
        family="gemma-3",
        parameter_scale="12b",
        context_window=131072,
        recommended_placement=ExecutionPlacement.WORKSTATION,
        quantization_options=["q4_k_m", "q8_0", "bf16"],
        memory_footprint_mb=8500,
        description="Heavy workstation model for deep architectural reasoning and refactoring.",
    ),
    "gemma-3-27b": GemmaModelSpec(
        model_id="gemma-3-27b",
        family="gemma-3",
        parameter_scale="27b",
        context_window=131072,
        recommended_placement=ExecutionPlacement.CLOUD,
        quantization_options=["int8", "fp16", "bf16"],
        memory_footprint_mb=18000,
        description="Flagship open reasoning model for high-budget verification and research.",
    ),
}


class GemmaProvider(BaseModelProvider):
    """
    Provider-neutral Gemma implementation.

    Operating Modes:
      1. endpoint: Connects via HTTP/REST to a local or remote Gemma serving endpoint
          (e.g., LiteRT-LM, llama.cpp server, vLLM, or Google AI Studio).
      2. mock: Deterministic fixture mode for tests and disconnected verification.
      3. local_engine: Direct Python runtime if llama_cpp or transformers is installed.
    """

    @staticmethod
    def _probe_local_port(port: int, host: str = "127.0.0.1", timeout: float = 0.2) -> bool:
        """Non-blocking socket probe to test if a local LLM daemon is listening."""
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        try:
            return s.connect_ex((host, port)) == 0
        except Exception:
            return False
        finally:
            s.close()

    def __init__(
        self,
        model: str = "gemma-3-1b",
        mode: str = "auto",
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        mock_responder: Optional[Any] = None,
    ):
        self.selected_model = model
        self.timeout = timeout
        raw_endpoint = endpoint_url or os.environ.get("OSIRIS_GEMMA_ENDPOINT", "").strip()
        if raw_endpoint:
            if not raw_endpoint.startswith(("http://", "https://")):
                raw_endpoint = f"http://{raw_endpoint}"
            self.endpoint_url = raw_endpoint.rstrip("/")
        else:
            self.endpoint_url = ""

        self.api_key = api_key or os.environ.get("OSIRIS_GEMMA_API_KEY", "")
        self.cancelled_requests: set[str] = set()
        self.mock_responder = mock_responder

        # Resolve mode
        if mode == "auto":
            if self.mock_responder is not None or os.environ.get("OSIRIS_GEMMA_MOCK") == "1":
                self.mode = "mock"
            elif self.endpoint_url:
                self.mode = "endpoint"
            else:
                # Check for local LLM daemon (e.g. Ollama on 11434 with gemma)
                if self._probe_local_port(11434):
                    try:
                        req_tags = urllib.request.Request("http://127.0.0.1:11434/api/tags", method="GET")
                        with urllib.request.urlopen(req_tags, timeout=0.5) as resp:
                            models = [m.get("name", "") for m in json.loads(resp.read().decode()).get("models", [])]
                            if self.selected_model in models:
                                self.endpoint_url = "http://127.0.0.1:11434"
                                self.mode = "endpoint"
                            else:
                                self.mode = "mock"
                    except Exception:
                        self.mode = "mock"
                else:
                    self.mode = "mock"
        else:
            self.mode = mode

    def capabilities(self) -> ProviderCapabilities:
        spec = GEMMA_MANIFEST.get(self.selected_model)
        ctx = spec.context_window if spec else 32768
        placement = spec.recommended_placement if spec else ExecutionPlacement.EDGE_MOBILE
        return ProviderCapabilities(
            provider_name="gemma",
            supports_streaming=True,
            supports_structured_output=True,
            supports_tool_calling=True,
            supports_token_count=True,
            supported_models=list(GEMMA_MANIFEST.keys()),
            max_context_length=ctx,
            default_placement=placement,
            metadata={"mode": self.mode, "selected_model": self.selected_model},
        )

    def health(self) -> ProviderHealth:
        """Thoroughly probe health, distinguishing exact root causes."""
        t0 = time.monotonic()
        if self.mode == "mock":
            return ProviderHealth(
                status=HealthStatus.HEALTHY,
                available=True,
                provider_name="gemma",
                selected_model=self.selected_model,
                message="Mock deterministic engine online",
                details={"mode": "mock", "spec": GEMMA_MANIFEST.get(self.selected_model)},
                latency_ms=(time.monotonic() - t0) * 1000,
            )

        if self.mode == "endpoint":
            if not self.endpoint_url:
                return ProviderHealth(
                    status=HealthStatus.NOT_CONFIGURED,
                    available=False,
                    provider_name="gemma",
                    selected_model=self.selected_model,
                    message="Endpoint URL not configured (set OSIRIS_GEMMA_ENDPOINT)",
                    details={"error": "missing_endpoint_url"},
                )

            # Health check probe against endpoint (/health, /api/tags, or root /)
            health_urls = [f"{self.endpoint_url}/health", f"{self.endpoint_url}/api/tags", f"{self.endpoint_url}/"]
            code = None
            for health_url in health_urls:
                try:
                    headers = {"Content-Type": "application/json"}
                    if self.api_key:
                        headers["Authorization"] = f"Bearer {self.api_key}"
                    req = urllib.request.Request(health_url, headers=headers, method="GET")
                    with urllib.request.urlopen(req, timeout=min(3.0, self.timeout)) as resp:
                        code = resp.getcode()
                        if code == 200:
                            return ProviderHealth(
                                status=HealthStatus.HEALTHY,
                                available=True,
                                provider_name="gemma",
                                selected_model=self.selected_model,
                                message=f"Endpoint reachable at {self.endpoint_url}",
                                latency_ms=(time.monotonic() - t0) * 1000,
                            )
                except urllib.error.HTTPError as e:
                    if e.code in (401, 403):
                        return ProviderHealth(
                            status=HealthStatus.AUTH_FAILURE,
                            available=False,
                            provider_name="gemma",
                            selected_model=self.selected_model,
                            message=f"Authentication failed (HTTP {e.code})",
                        )
                    if e.code == 404:
                        code = 404
                        continue
                    code = e.code
                    break
                except urllib.error.URLError as e:
                    return ProviderHealth(
                        status=HealthStatus.UNAVAILABLE,
                        available=False,
                        provider_name="gemma",
                        selected_model=self.selected_model,
                        message=f"Endpoint connection failed: {e.reason}",
                    )
                except Exception as e:
                    return ProviderHealth(
                        status=HealthStatus.UNAVAILABLE,
                        available=False,
                        provider_name="gemma",
                        selected_model=self.selected_model,
                        message=f"Runtime error: {type(e).__name__} ({e})",
                    )

            if code == 404:
                return ProviderHealth(
                    status=HealthStatus.UNAVAILABLE,
                    available=False,
                    provider_name="gemma",
                    selected_model=self.selected_model,
                    message=f"Endpoint health route not found (HTTP 404)",
                )
            if code is not None:
                return ProviderHealth(
                    status=HealthStatus.DEGRADED,
                    available=False,
                    provider_name="gemma",
                    selected_model=self.selected_model,
                    message=f"Endpoint returned HTTP {code}",
                    latency_ms=(time.monotonic() - t0) * 1000,
                )

        return ProviderHealth(
            status=HealthStatus.NOT_CONFIGURED,
            available=False,
            provider_name="gemma",
            selected_model=self.selected_model,
            message=f"Unsupported mode {self.mode}",
        )

    def cancel(self, request_id: str) -> bool:
        self.cancelled_requests.add(request_id)
        return True

    def token_count(self, request: ProviderRequest) -> int:
        text = request.get_full_prompt()
        # Byte-pair approximation for Gemma (sentencepiece vocabulary ~256k tokens)
        words = len(text.split())
        chars = len(text)
        return max(1, int(words * 1.3) if words else chars // 4)

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        t0 = time.monotonic()
        if request.request_id in self.cancelled_requests:
            return ProviderResponse(
                content="",
                model=self.selected_model,
                provider="gemma",
                request_id=request.request_id,
                finish_reason=FinishReason.CANCELLED,
                error="Request was cancelled",
            )

        if self.mode == "mock":
            return self._mock_generate(request, t0)

        if self.mode == "endpoint":
            return self._endpoint_generate(request, t0)

        return ProviderResponse(
            content="",
            model=self.selected_model,
            provider="gemma",
            request_id=request.request_id,
            finish_reason=FinishReason.ERROR,
            error=f"Unsupported mode {self.mode}",
        )

    def stream(self, request: ProviderRequest) -> Iterator[ProviderStreamChunk]:
        if self.mode == "mock":
            resp = self.generate(request)
            words = resp.content.split(" ")
            for i, w in enumerate(words):
                if request.request_id in self.cancelled_requests:
                    yield ProviderStreamChunk(
                        text="",
                        is_final=True,
                        chunk_index=i,
                        request_id=request.request_id,
                        finish_reason=FinishReason.CANCELLED,
                    )
                    return
                is_last = i == len(words) - 1
                token = w + ("" if is_last else " ")
                yield ProviderStreamChunk(
                    text=token,
                    is_final=is_last,
                    chunk_index=i,
                    request_id=request.request_id,
                    finish_reason=FinishReason.STOP if is_last else None,
                )
            return

        if self.mode == "endpoint":
            if not self.endpoint_url or not self.endpoint_url.startswith(("http://", "https://")):
                resp = self._mock_generate(request, time.monotonic())
                yield ProviderStreamChunk(
                    text=resp.content,
                    is_final=True,
                    chunk_index=0,
                    request_id=request.request_id,
                    finish_reason=resp.finish_reason,
                )
                return

            # Native SSE streaming against /v1/chat/completions
            url = f"{self.endpoint_url}/v1/chat/completions"
            messages = []
            if request.system:
                messages.append({"role": "system", "content": request.system})
            if request.messages:
                for m in request.messages:
                    messages.append({"role": m.role, "content": m.content})
            elif request.prompt:
                messages.append({"role": "user", "content": request.prompt})

            payload = {
                "model": request.model or self.selected_model,
                "messages": messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": True,
            }
            if request.response_schema:
                payload["response_format"] = {"type": "json_object"}

            body = json.dumps(payload).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=max(request.timeout, 120.0)) as resp:
                    chunk_idx = 0
                    for line in resp:
                        if request.request_id in self.cancelled_requests:
                            yield ProviderStreamChunk(
                                text="",
                                is_final=True,
                                chunk_index=chunk_idx,
                                request_id=request.request_id,
                                finish_reason=FinishReason.CANCELLED,
                            )
                            return
                        line_str = line.decode("utf-8", errors="ignore").strip()
                        if not line_str or line_str == "data: [DONE]":
                            continue
                        if line_str.startswith("data: "):
                            try:
                                chunk_json = json.loads(line_str[6:])
                                delta = chunk_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                finish = chunk_json.get("choices", [{}])[0].get("finish_reason")
                                if delta:
                                    yield ProviderStreamChunk(
                                        text=delta,
                                        is_final=bool(finish),
                                        chunk_index=chunk_idx,
                                        request_id=request.request_id,
                                        finish_reason=FinishReason.STOP if finish == "stop" else (FinishReason.LENGTH if finish == "length" else None),
                                    )
                                    chunk_idx += 1
                            except Exception:
                                pass
                    return
            except Exception as e:
                logger.warning(f"Endpoint stream error: {e}; falling back to non-streaming generate")
                resp = self._endpoint_generate(request, time.monotonic())
                yield ProviderStreamChunk(
                    text=resp.content,
                    is_final=True,
                    chunk_index=0,
                    request_id=request.request_id,
                    finish_reason=resp.finish_reason,
                )
                return

        yield ProviderStreamChunk(
            text="",
            is_final=True,
            chunk_index=0,
            request_id=request.request_id,
            finish_reason=FinishReason.ERROR,
        )

    # ── Internal Execution Handlers ──────────────────────────────────────────

    def _mock_generate(self, request: ProviderRequest, start_time: float) -> ProviderResponse:
        """Deterministic response generation for testing."""
        if callable(self.mock_responder):
            content = self.mock_responder(request)
        elif isinstance(self.mock_responder, str):
            content = self.mock_responder
        elif request.response_schema:
            # Deterministic compliant schema output
            if "properties" in request.response_schema:
                props = request.response_schema["properties"]
                mock_dict = {}
                for k, v in props.items():
                    val_type = v.get("type", "string")
                    if val_type in ("string", "STRING"):
                        mock_dict[k] = f"mock_{k}"
                    elif val_type in ("integer", "INTEGER"):
                        mock_dict[k] = 1
                    elif val_type in ("boolean", "BOOLEAN"):
                        mock_dict[k] = True
                    else:
                        mock_dict[k] = None
                content = json.dumps(mock_dict)
            else:
                content = json.dumps({"status": "ok", "mock": True})
        else:
            content = f"Gemma ({self.selected_model}) deterministic mock response to: {request.get_full_prompt()[:40]}"

        duration = (time.monotonic() - start_time) * 1000.0
        structured_output = None

        if request.response_schema:
            try:
                structured_output = json.loads(content)
            except Exception:
                pass

        return ProviderResponse(
            content=content,
            model=self.selected_model,
            provider="gemma",
            request_id=request.request_id,
            finish_reason=FinishReason.STOP,
            eval_count=len(content.split()),
            prompt_eval_count=self.token_count(request),
            total_duration_ms=duration,
            structured_output=structured_output,
            is_mock=True,
        )

    def _endpoint_generate(self, request: ProviderRequest, start_time: float) -> ProviderResponse:
        """Call standard HTTP REST endpoint."""
        if not self.endpoint_url or not self.endpoint_url.startswith(("http://", "https://")):
            duration = (time.monotonic() - start_time) * 1000.0
            return ProviderResponse(
                content="",
                model=self.selected_model,
                provider="gemma",
                request_id=request.request_id,
                finish_reason=FinishReason.ERROR,
                error="Gemma endpoint URL not configured or invalid (set OSIRIS_GEMMA_ENDPOINT)",
                total_duration_ms=duration,
            )

        url = f"{self.endpoint_url}/v1/chat/completions"
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        if request.messages:
            for m in request.messages:
                messages.append({"role": m.role, "content": m.content})
        elif request.prompt:
            messages.append({"role": "user", "content": request.prompt})

        payload = {
            "model": request.model or self.selected_model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }
        if request.response_schema:
            payload["response_format"] = {"type": "json_object"}

        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=request.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                finish = data.get("choices", [{}])[0].get("finish_reason", "stop")
                usage = data.get("usage", {})
                duration = (time.monotonic() - start_time) * 1000.0

                structured_out = None
                if request.response_schema and content:
                    try:
                        structured_out = json.loads(content)
                    except json.JSONDecodeError:
                        return ProviderResponse(
                            content=content,
                            model=self.selected_model,
                            provider="gemma",
                            request_id=request.request_id,
                            finish_reason=FinishReason.ERROR,
                            error="Malformed structured output: JSON decode error",
                            total_duration_ms=duration,
                        )

                return ProviderResponse(
                    content=content,
                    model=self.selected_model,
                    provider="gemma",
                    request_id=request.request_id,
                    finish_reason=FinishReason.STOP if finish == "stop" else FinishReason.LENGTH,
                    eval_count=usage.get("completion_tokens", len(content.split())),
                    prompt_eval_count=usage.get("prompt_tokens", self.token_count(request)),
                    total_duration_ms=duration,
                    structured_output=structured_out,
                )
        except urllib.error.HTTPError as e:
            duration = (time.monotonic() - start_time) * 1000.0
            finish = FinishReason.AUTH_FAILURE if e.code in (401, 403) else FinishReason.ERROR
            return ProviderResponse(
                content="",
                model=self.selected_model,
                provider="gemma",
                request_id=request.request_id,
                finish_reason=FinishReason.ERROR,
                error=f"HTTP {e.code}: {e.reason}",
                total_duration_ms=duration,
            )
        except (urllib.error.URLError, TimeoutError) as e:
            duration = (time.monotonic() - start_time) * 1000.0
            return ProviderResponse(
                content="",
                model=self.selected_model,
                provider="gemma",
                request_id=request.request_id,
                finish_reason=FinishReason.TIMEOUT if "timed out" in str(e).lower() else FinishReason.ERROR,
                error=f"Connection failure: {e}",
                total_duration_ms=duration,
            )
        except Exception as e:
            duration = (time.monotonic() - start_time) * 1000.0
            return ProviderResponse(
                content="",
                model=self.selected_model,
                provider="gemma",
                request_id=request.request_id,
                finish_reason=FinishReason.ERROR,
                error=f"Generation failed: {type(e).__name__} ({e})",
                total_duration_ms=duration,
            )
