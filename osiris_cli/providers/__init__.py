"""
OSIRIS Provider-Neutral Inference Layer
=======================================
"""

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
from .gemma import GEMMA_MANIFEST, GemmaModelSpec, GemmaProvider
from .gemini_compat import GeminiCompatibilityProvider
from .ollama_compat import OllamaCompatibilityProvider
from .mentor_adapter import ProviderMentorAdapter
from .router import (
    MentorRouter,
    RouteDecision,
    SystemTelemetry,
    collect_system_telemetry,
    discover_ollama_models,
    match_installed_model,
    normalize_ollama_tag,
    select_mentor,
)
from .registry import (
    create_provider,
    get_active_provider,
    get_available_providers,
    register_provider,
    reset_active_provider,
)

__all__ = [
    "BaseModelProvider",
    "ExecutionPlacement",
    "FinishReason",
    "HealthStatus",
    "Message",
    "ProviderCapabilities",
    "ProviderHealth",
    "ProviderRequest",
    "ProviderResponse",
    "ProviderStreamChunk",
    "GEMMA_MANIFEST",
    "GemmaModelSpec",
    "GemmaProvider",
    "GeminiCompatibilityProvider",
    "OllamaCompatibilityProvider",
    "ProviderMentorAdapter",
    "MentorRouter",
    "RouteDecision",
    "SystemTelemetry",
    "collect_system_telemetry",
    "discover_ollama_models",
    "match_installed_model",
    "normalize_ollama_tag",
    "select_mentor",
    "create_provider",
    "get_active_provider",
    "get_available_providers",
    "register_provider",
    "reset_active_provider",
]

from .local_qccd import (
    ParsedCircuit,
    QCCDCircuitError,
    QCCDEmulationResult,
    QCCDNoiseModel,
    emulate_qccd,
    parse_guppy_ast,
    parse_openqasm,
    parse_quantum_source,
)
from .quantinuum import (
    QUANTINUUM_TARGETS,
    QuantinuumTargetSpec,
    TopologyBenchmarkReport,
    TopologyComparisonRow,
    compare_topologies,
    compute_hqc,
    resolve_quantinuum_target,
)

__all__.extend([
    "ParsedCircuit",
    "QCCDCircuitError",
    "QCCDEmulationResult",
    "QCCDNoiseModel",
    "QUANTINUUM_TARGETS",
    "QuantinuumTargetSpec",
    "TopologyBenchmarkReport",
    "TopologyComparisonRow",
    "compare_topologies",
    "compute_hqc",
    "emulate_qccd",
    "parse_guppy_ast",
    "parse_openqasm",
    "parse_quantum_source",
    "resolve_quantinuum_target",
])

from .quantinuum_nexus import (
    GovernanceError,
    NexusConfig,
    NexusDisposition,
    QuantinuumNexusClient,
    estimate_hqc,
    sign_tier3_approval_token,
    verify_tier3_approval_token,
)
from .tket_wrapper import (
    QuantinuumCompilerOptions,
    QuantinuumTarget,
    TketCircuitMetrics,
    TketCompilationResult,
    compile_for_quantinuum,
)

__all__.extend([
    "GovernanceError",
    "NexusConfig",
    "NexusDisposition",
    "QuantinuumCompilerOptions",
    "QuantinuumNexusClient",
    "QuantinuumTarget",
    "TketCircuitMetrics",
    "TketCompilationResult",
    "compile_for_quantinuum",
    "estimate_hqc",
    "sign_tier3_approval_token",
    "verify_tier3_approval_token",
])
