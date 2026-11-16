"""
OSIRIS Provider-Neutral Model Interface
=======================================
Defines stable contracts for model providers (Gemma, Ollama, Gemini, etc.)
ensuring OSIRIS core does not depend on a single model vendor, inference server,
or proprietary runtime.

Contracts:
  - generate(request) -> ProviderResponse
  - stream(request) -> Iterator[ProviderStreamChunk]
  - health() -> ProviderHealth
  - capabilities() -> ProviderCapabilities
  - token_count(request) -> int
  - cancel(request_id) -> bool
"""

from __future__ import annotations

import abc
import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


class HealthStatus(str, enum.Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    NOT_CONFIGURED = "not_configured"
    MISSING_DEPENDENCY = "missing_dependency"
    AUTH_FAILURE = "auth_failure"


class FinishReason(str, enum.Enum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALL = "tool_call"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ExecutionPlacement(str, enum.Enum):
    EDGE_MOBILE = "edge_mobile"       # Node Alpha (Termux / Android / low-RAM)
    WORKSTATION = "workstation"       # Node Beta (WSL / Linux x86_64 / desktop)
    CLOUD = "cloud"                   # Cloud endpoints / managed APIs


@dataclass
class Message:
    role: str  # "system", "user", "assistant", "tool"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderCapabilities:
    provider_name: str
    supports_streaming: bool = True
    supports_structured_output: bool = True
    supports_tool_calling: bool = False
    supports_token_count: bool = True
    supported_models: List[str] = field(default_factory=list)
    max_context_length: int = 8192
    default_placement: ExecutionPlacement = ExecutionPlacement.EDGE_MOBILE
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderHealth:
    status: HealthStatus
    available: bool
    provider_name: str
    selected_model: str = ""
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass
class ProviderRequest:
    prompt: str = ""
    messages: List[Message] = field(default_factory=list)
    model: Optional[str] = None
    system: str = ""
    temperature: float = 0.2
    max_tokens: int = 512
    stop_sequences: List[str] = field(default_factory=list)
    response_schema: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timeout: float = 60.0
    stream: bool = False
    num_ctx: Optional[int] = None  # Ollama context window in tokens; None keeps the server default
    think: Optional[bool] = None  # Ollama reasoning models: False asks for the answer without a thinking pass

    def get_full_prompt(self) -> str:
        """Helper to get flattened prompt string if messages are used."""
        if self.prompt:
            return self.prompt
        parts = []
        if self.system:
            parts.append(f"System: {self.system}")
        for m in self.messages:
            parts.append(f"{m.role.capitalize()}: {m.content}")
        return "\n\n".join(parts)


@dataclass
class ProviderResponse:
    content: str
    model: str
    provider: str
    request_id: str
    finish_reason: FinishReason = FinishReason.STOP
    eval_count: int = 0
    prompt_eval_count: int = 0
    total_duration_ms: float = 0.0
    structured_output: Optional[Any] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    error: str = ""
    is_mock: bool = False

    @property
    def tokens_per_second(self) -> float:
        if self.total_duration_ms > 0 and self.eval_count > 0:
            return self.eval_count / (self.total_duration_ms / 1000.0)
        return 0.0

    @property
    def is_success(self) -> bool:
        return not self.error and self.finish_reason in (FinishReason.STOP, FinishReason.LENGTH, FinishReason.TOOL_CALL)


@dataclass
class ProviderStreamChunk:
    text: str
    is_final: bool = False
    chunk_index: int = 0
    request_id: str = ""
    finish_reason: Optional[FinishReason] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseModelProvider(abc.ABC):
    """Abstract Base Class for OSIRIS Model Inference Substrates."""

    @abc.abstractmethod
    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Execute non-streaming text generation."""
        pass

    @abc.abstractmethod
    def stream(self, request: ProviderRequest) -> Iterator[ProviderStreamChunk]:
        """Stream generation tokens/chunks."""
        pass

    @abc.abstractmethod
    def health(self) -> ProviderHealth:
        """Check provider health, distinguishing missing dependencies, invalid config,
        auth failures, unavailable models, and runtime errors."""
        pass

    @abc.abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Return provider capabilities, model support, and execution placement."""
        pass

    def token_count(self, request: ProviderRequest) -> int:
        """Estimate or compute token count for a request. Default is char-heuristic fallback."""
        text = request.get_full_prompt()
        return max(1, len(text) // 4)

    def cancel(self, request_id: str) -> bool:
        """Cancel a running request. Subclasses may implement async/threaded cancellation."""
        return False
