"""Natural-language task -> node (Sprint A1.2). Deterministic and explainable.

The router only *proposes* a target. The capability-gap analyzer then accepts or rejects it
(is the node online, does it support the capability, is policy evaluated), and the SYS-021
gateway authorizes nothing the analyzer did not clear. The router never executes anything.

Rules, in order (the first that applies decides):

1. A task that names both nodes is ambiguous: no route; ask which node.
2. Work that only the workstation can do (QPU/quantum jobs, simulation, test suites, builds,
   training, benchmarks, GPU work) goes to Node Beta. If the task explicitly names the phone, it
   is refused with the reason, rather than silently moved.
3. A task that names the phone/edge/on-device model and asks for model inference (explain,
   answer, summarise, classify, translate, chat) goes to Node Alpha's ``edge_llm_inference``.
   Naming the phone without an inference request is refused: the phone runs nothing else.
4. Otherwise the default is Node Beta's ``workstation_task``, recorded as the default rule.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from osiris_cli.node_ids import NODE_ALPHA, NODE_BETA

#: capability -> nodes that provide it. The analyzer's CapabilitySpec registry must agree.
CAPABILITY_NODES: Dict[str, Tuple[str, ...]] = {
    "edge_llm_inference": (NODE_ALPHA,),
    "workstation_task": (NODE_BETA,),
    "workstation_compute": (NODE_BETA,),
}

_ALPHA_TERMS = re.compile(
    r"\b(mobile (?:edge|node|device)|phone|handset|on[- ]device|edge (?:node|model|bridge|device)|"
    r"node alpha|alpha node|s26)\b", re.I)
_BETA_TERMS = re.compile(r"\b(workstation|laptop|desktop|wsl2?|node beta|beta node)\b", re.I)
_INFERENCE = re.compile(
    r"\b(explain|answer|summari[sz]e|summary|classify|translate|chat|describe|infer(?:ence)?|"
    r"ask the (?:on[- ]device |edge )?model|use the (?:on[- ]device |edge |local )?model)\b", re.I)
_COMPUTE_ONLY = re.compile(
    r"\b(qpu|ibm[_ ]?\w*|qiskit|quantum (?:job|circuit|hardware)|simulat\w*|pytest|test suite|run (?:the )?tests|"
    r"compile|build|train(?:ing)?|fine[- ]tune|benchmark\w*|gpu|cuda)\b", re.I)


@dataclass(frozen=True)
class RouteDecision:
    routed: bool
    rule_id: str
    reason: str
    capability_id: Optional[str] = None
    target_node: Optional[str] = None
    matched: Dict[str, Tuple[str, ...]] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {"routed": self.routed, "rule_id": self.rule_id, "reason": self.reason,
                "capability_id": self.capability_id, "target_node": self.target_node,
                "matched": {k: list(v) for k, v in self.matched.items()}}


def _found(rx: "re.Pattern[str]", text: str) -> Tuple[str, ...]:
    return tuple(sorted({m.group(0).lower() for m in rx.finditer(text)}))


def route_task(text: str) -> RouteDecision:
    t = " ".join(str(text or "").split())
    if not t:
        return RouteDecision(False, "N0-empty", "the task is empty")
    alpha, beta = _found(_ALPHA_TERMS, t), _found(_BETA_TERMS, t)
    infer, compute = _found(_INFERENCE, t), _found(_COMPUTE_ONLY, t)
    matched = {k: v for k, v in {"alpha": alpha, "beta": beta, "inference": infer, "compute": compute}.items() if v}

    if alpha and beta:
        return RouteDecision(False, "N1-both-nodes-named",
                             f"the task names both the phone ({', '.join(alpha)}) and the workstation "
                             f"({', '.join(beta)}); say which node should do it", matched=matched)
    if compute:
        if alpha:
            return RouteDecision(False, "N2-compute-not-on-alpha",
                                 f"{', '.join(compute)} needs the workstation; the phone only runs its "
                                 "on-device model. Remove the phone from the task to run it on Node Beta",
                                 matched=matched)
        return RouteDecision(True, "N2-workstation-compute", f"{', '.join(compute)} runs on the workstation",
                             "workstation_compute", NODE_BETA, matched)
    if alpha:
        if infer:
            return RouteDecision(True, "N3-edge-inference",
                                 f"the task asks the phone's model to {infer[0]}",
                                 "edge_llm_inference", NODE_ALPHA, matched)
        return RouteDecision(False, "N3-alpha-without-inference",
                             "the task names the phone but asks for nothing its model can do "
                             "(explain, answer, summarise, classify, translate)", matched=matched)
    return RouteDecision(True, "N4-default-workstation", "no node was named; the workstation is the default",
                         "workstation_task", NODE_BETA, matched)


_MULTI_CANDIDATE_UTILITY_FLOOR = Decimal("0.250000")


def route_task_calibrated(
    text: str,
    candidate_models: Sequence[str] = ("ollama/gemma4:e4b", "ollama/qwen2.5-coder:7b"),
    calibration_path: Optional[Union[str, os.PathLike]] = None,
) -> Dict[str, Any]:
    """Combine deterministic substrate routing with calibration-weighted model selection."""
    from osiris_cli import evolution_engine

    decision = route_task(text)
    selected_model, score = evolution_engine.select_calibrated_model(
        candidate_models,
        calibration_path=calibration_path,
    )
    escalate = (
        True
        if score is None
        else Decimal(score.utility_str) < _MULTI_CANDIDATE_UTILITY_FLOOR
    )
    return {
        "escalate_multi_candidate": escalate,
        "model_score": score.to_dict() if score is not None else None,
        "route": decision.as_dict(),
        "selected_model": selected_model,
    }
