"""OSIRIS orchestrates AURA, AIDEN and IRIS on one task.

    objective -> AURA proposes -> AIDEN reviews -> IRIS writes the operator brief -> OSIRIS decides

The three roles are language-model calls. They hold no authority and cannot run anything:

  * AURA  proposes a plan for the objective.
  * AIDEN reviews AURA's proposal and ends with one line ``VERDICT: PROCEED | REVISE | STOP``.
  * IRIS  integrates both into a short brief for the operator. It adds no claims of its own.

OSIRIS is ordinary code. It runs the roles in order, puts every output through the evidence
firewall (which flags statements that tests passed, that something was executed or measured, or
that restate a claim the register records as refuted), reads AIDEN's verdict line, and derives
the decision. The best possible decision is PENDING_HUMAN_APPROVAL: nothing here approves,
executes, merges or submits. Each task ends with one row on a hash-chained receipt file that
holds digests, the firewall findings and the decision.

The model call is injected (``model_fn``), so tests run without a network. The default is the
model router (``model_router``): each role runs on a local Ollama model, and Google's hosted
model is used only when the request says ``allow_cloud`` and only through
``osiris_cli.gemini_gateway``. Each role's route record (requested and resolved model, runtime,
whether the request left the device, every candidate tried) is returned and receipted.

"OSIRIS decides" means the workflow disposition of a proposal, nothing more. No decision here
authorizes execution; anything that acts goes through the separate approval gates.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..trio_authz import Receipts
from .evidence_firewall import check_report

ROLES = ("AURA", "AIDEN", "IRIS")
RECEIPT_KIND = "agents.task"
EVIDENCE_CLASS = "MODEL_OUTPUT"
MAX_OBJECTIVE, MAX_CONTEXT, MAX_OUTPUT = 4000, 12000, 6000

DECISIONS = ("PENDING_HUMAN_APPROVAL", "REVISION_REQUESTED", "HELD_BY_REVIEW", "QUARANTINED", "REJECTED",
             "FAILED")

_RULES = (
    "You are one role inside OSIRIS, a governed research and engineering workspace for DNA-Lang.\n"
    "You cannot execute, approve, verify, deploy or submit anything, and you must not say that you did.\n"
    "Do not state that a test passed, that something ran or that a value was measured.\n"
    "Only the simulator is connected; no quantum hardware is available.\n"
    "Do not present a claim as true when the claims register below records it as refuted, an artifact, "
    "not measured or without evidence.\n"
    "Reference the claims register only when the objective or proposal touches a listed claim; do not recite unrelated claim IDs.\n"
    "Plain text only. Be concrete and brief."
)

_ROLE_TEXT = {
    "AURA": ("You are AURA, the proposing role. Write a proposal for the objective: the approach, three to "
             "six numbered steps, what evidence each step would produce, and the main risk. Every item is a "
             "proposal, not a result."),
    "AIDEN": ("You are AIDEN, the reviewing role. Review AURA's proposal: what is unsupported, what could go "
              "wrong, and what evidence or receipt would settle it. Do not add new ideas of your own. "
              "End with exactly one line: 'VERDICT: PROCEED', 'VERDICT: REVISE' or 'VERDICT: STOP'."),
    "IRIS": ("You are IRIS, the integrating role. Write the brief for the human operator in at most 150 "
             "words: the objective in one line, what AURA proposes, AIDEN's objections, and the single next "
             "action the operator could approve. Add no claims that are not in AURA's or AIDEN's text."),
}

_VERDICT = re.compile(r"^\s*VERDICT:\s*(PROCEED|REVISE|STOP)\s*$", re.I | re.M)

# (role, system prompt, user prompt) -> (text, provider label) or (text, provider label, route record)
ModelFn = Callable[[str, str, str], Tuple[Any, ...]]


class CouncilError(Exception):
    """Bad input. Model failures are not raised: they end the task with decision FAILED."""


class ReceiptError(Exception):
    """The task ran but its receipt could not be written. No result is returned without a receipt."""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _system(role: str, claims: List[str]) -> str:
    register = ("Claims register (verdict per claim):\n" + "\n".join(claims[:120])) if claims else \
        "Claims register: not available. Treat every scientific claim as unverified."
    return f"{_RULES}\n\n{_ROLE_TEXT[role]}\n\n{register}"


def gateway_model_fn(role: str, system: str, prompt: str) -> Tuple[str, str]:
    """The default model call: OSIRIS's own Gemini gateway (redaction, token budget, call ledger)."""
    from .. import gemini_gateway
    out = gemini_gateway.generate(prompt, purpose=f"agents.task/{role}", system=system,
                                  max_output_tokens=4096, temperature=0.3)
    return out["text"], f"{out['backend']}:{out.get('model_version') or 'gemini'}"


def routed_model_fn(cloud_allowed: bool = False) -> ModelFn:
    """The default model call: the router picks the model for the role and reports what answered."""
    def call(role: str, system: str, prompt: str) -> Tuple[str, str, Dict[str, Any]]:
        from . import model_router
        text, route = model_router.route(role, system, prompt, cloud_allowed=cloud_allowed)
        return text, f"{route['runtime']}:{route['resolved_model']}", route
    return call


def claim_lines() -> List[str]:
    from .. import claims
    return [f"{c.id}: {c.verdict}" for c in claims.REGISTER]


_MOCK_PROVIDER_RE = re.compile(r"(?:^|[:+/_\-\s])mock(?:$|[:+/_\-\s])", re.I)


def _is_mock_output(provider: Any, route: Optional[Dict[str, Any]], text: str) -> bool:
    if isinstance(route, dict) and (route.get("is_mock") is True or str(route.get("runtime", "")).lower() == "mock"):
        return True
    if _MOCK_PROVIDER_RE.search(str(provider or "")):
        return True
    low = (text or "").lower()
    return "[gemma:mock" in low or "deterministic mock response" in low


def parse_aiden_verdict(text: str) -> Tuple[Optional[str], bool, List[str]]:
    """Return (effective_verdict, conflicted, all_verdicts). Conflicting verdicts never resolve to PROCEED."""
    found = [v.upper() for v in _VERDICT.findall(text or "")]
    if not found:
        return None, False, []
    distinct = set(found)
    if len(distinct) == 1:
        return found[-1], False, found
    conservative = "STOP" if "STOP" in distinct else "REVISE"
    return conservative, True, found


def aiden_verdict(text: str) -> Optional[str]:
    """Conservative verdict from AIDEN's review: None if missing, never PROCEED when lines conflict."""
    verdict, _, _ = parse_aiden_verdict(text)
    return verdict


def decide(firewall: Dict[str, str], verdict: Optional[str], *, verdict_conflicted: bool = False) -> Tuple[str, str]:
    """The decision and its reason, from the firewall results and AIDEN's verdict line only."""
    for role in ROLES:
        if firewall.get(role) == "REJECT":
            return "REJECTED", f"the evidence firewall rejected {role}'s text"
    for role in ROLES:
        if firewall.get(role) == "QUARANTINE":
            return "QUARANTINED", f"{role}'s text is ambiguous, quoted without rejection, or unsupported by evidence"
    for role in ROLES:
        state = firewall.get(role)
        if state != "ACCEPT":
            return "QUARANTINED", f"{role} has unknown or missing firewall state {state!r}"
    if verdict_conflicted:
        if verdict == "STOP":
            return "HELD_BY_REVIEW", "AIDEN gave conflicting verdict lines including STOP"
        return "REVISION_REQUESTED", "AIDEN gave conflicting verdict lines"
    if verdict == "STOP":
        return "HELD_BY_REVIEW", "AIDEN's verdict is STOP"
    if verdict == "REVISE":
        return "REVISION_REQUESTED", "AIDEN's verdict is REVISE"
    if verdict == "PROCEED":
        return "PENDING_HUMAN_APPROVAL", "no firewall finding and AIDEN's verdict is PROCEED; the operator decides"
    return "REVISION_REQUESTED", "AIDEN gave no verdict line"


def run_task(objective: Any, context: Any = None, *, receipts_path: str, model_fn: Optional[ModelFn] = None,
             claims: Optional[List[str]] = None, allow_cloud: Any = False) -> Dict[str, Any]:
    if not isinstance(objective, str) or not objective.strip():
        raise CouncilError("objective must be a non-empty string")
    objective = objective.strip()
    if len(objective) > MAX_OBJECTIVE:
        raise CouncilError(f"objective is longer than {MAX_OBJECTIVE} characters")
    if context is None:
        context = ""
    if not isinstance(context, str) or len(context) > MAX_CONTEXT:
        raise CouncilError(f"context must be a string of at most {MAX_CONTEXT} characters")
    context = context.strip()
    if not isinstance(allow_cloud, bool):
        raise CouncilError("allow_cloud must be true or false")

    model_fn = model_fn or routed_model_fn(allow_cloud)
    claims = claim_lines() if claims is None else claims
    task_id = "task-" + secrets.token_hex(6)
    events: List[Dict[str, Any]] = []

    def emit(actor: str, event: str, detail: str) -> None:
        events.append({"seq": len(events) + 1, "actor": actor, "event": event, "detail": detail})

    emit("OSIRIS", "TASK_ACCEPTED", f"{task_id}: objective {_sha(objective)[:16]}, {len(claims)} claims in context, "
                                    f"cloud {'allowed' if allow_cloud else 'not allowed'}")

    outputs: Dict[str, Dict[str, Any]] = {}
    firewall: Dict[str, str] = {}
    failure: Optional[str] = None
    failed_routes: Dict[str, Any] = {}
    base = f"Objective:\n{objective}" + (f"\n\nContext from the workspace:\n{context}" if context else "")
    prompts = {
        "AURA": lambda: base,
        "AIDEN": lambda: f"{base}\n\nAURA's proposal:\n{outputs['AURA']['text']}",
        "IRIS": lambda: (f"{base}\n\nAURA's proposal:\n{outputs['AURA']['text']}\n\n"
                         f"AIDEN's review:\n{outputs['AIDEN']['text']}"),
    }
    stage = {"AURA": "PROPOSAL", "AIDEN": "REVIEW", "IRIS": "BRIEF"}

    for role in ROLES:
        route: Optional[Dict[str, Any]] = None
        try:
            reply = model_fn(role, _system(role, claims), prompts[role]())
            if not isinstance(reply, (tuple, list)) or len(reply) < 2:
                failure = f"{role}: malformed model reply ({type(reply).__name__})"
                failed_routes[role] = {"status": "MALFORMED_REPLY", "type": type(reply).__name__}
                emit("OSIRIS", "MODEL_UNAVAILABLE", failure)
                break
            text, provider = reply[0], reply[1]
            route = reply[2] if len(reply) > 2 and isinstance(reply[2], dict) else None
        except Exception as e:  # noqa: BLE001 - any model failure ends the task; no key material is in e
            failure = f"{role}: {type(e).__name__}: {str(e)[:400]}"
            failed_routes[role] = getattr(e, "route", None) or {"status": "EXCEPTION", "error": type(e).__name__}
            emit("OSIRIS", "MODEL_UNAVAILABLE", failure)
            break
        if not isinstance(text, str):
            failure = f"{role}: the model call returned {type(text).__name__}, not text"
            failed_routes[role] = route or {"status": "MALFORMED_OUTPUT", "type": type(text).__name__}
            emit("OSIRIS", "MODEL_UNAVAILABLE", failure)
            break
        text = (text or "").strip()[:MAX_OUTPUT]
        if not text:
            failure = f"{role}: the model returned no text"
            failed_routes[role] = route or {"status": "EMPTY_OUTPUT"}
            emit("OSIRIS", "MODEL_UNAVAILABLE", failure)
            break
        if _is_mock_output(provider, route, text):
            failure = f"{role}: mock model output refused; cannot masquerade as live model output"
            failed_routes[role] = route or {"status": "MOCK_OUTPUT_REFUSED", "provider": str(provider)[:80]}
            emit("OSIRIS", "MODEL_UNAVAILABLE", failure)
            break
        result = check_report(text)
        firewall[role] = result.decision
        findings = [{"kind": f.kind, "verdict": f.verdict, "reason": f.reason, "sentence": f.sentence[:240]}
                    for f in result.findings]
        outputs[role] = {"role": role, "text": text, "provider": str(provider)[:80], "text_sha256": _sha(text),
                         "firewall": result.decision, "findings": findings, "route": route}
        where = "" if route is None else (" (left this device)" if route.get("left_device") else " (on this device)")
        emit(role, stage[role], f"{len(text)} characters via {str(provider)[:80]}{where}")
        emit("OSIRIS", "FIREWALL_" + result.decision, f"{role}: {len(findings)} finding(s)")

    verdict, verdict_conflicted, all_verdicts = parse_aiden_verdict(outputs["AIDEN"]["text"]) if "AIDEN" in outputs else (None, False, [])
    if failure:
        decision, reason = "FAILED", failure
    else:
        detail = f"conflicting ({', '.join(all_verdicts)}) -> {verdict}" if verdict_conflicted else (verdict or "none given")
        emit("AIDEN", "VERDICT", detail)
        decision, reason = decide(firewall, verdict, verdict_conflicted=verdict_conflicted)
    emit("OSIRIS", "DECISION_" + decision, reason)

    try:
        os.makedirs(os.path.dirname(os.path.abspath(receipts_path)), mode=0o700, exist_ok=True)
        row = Receipts(receipts_path).append(RECEIPT_KIND, {
            "task_id": task_id, "objective_sha256": _sha(objective), "context_sha256": _sha(context),
            "evidence_class": EVIDENCE_CLASS, "decision": decision, "reason": reason, "aiden_verdict": verdict,
            "allow_cloud": allow_cloud,
            "roles": [{"role": r, "provider": outputs[r]["provider"], "text_sha256": outputs[r]["text_sha256"],
                       "firewall": outputs[r]["firewall"], "findings": len(outputs[r]["findings"]),
                       "route": outputs[r]["route"]}
                      for r in ROLES if r in outputs],
            "failed_routes": failed_routes,
            "events": len(events) + 1, "executed": False,
        })
        chain_valid = Receipts.verify(receipts_path)
    except Exception as e:  # noqa: BLE001 - an unreceipted task is not reported as a result
        raise ReceiptError(f"the task receipt could not be written ({type(e).__name__}); no result is reported") from e
    emit("OSIRIS", "RECEIPT_RECORDED", f"{RECEIPT_KIND} {row['hash'][:16]}")

    return {
        "schema": "osiris-agent-task/v1", "task_id": task_id, "objective": objective,
        "evidence_class": EVIDENCE_CLASS, "orchestrator": "osiris_cli.orchestration.council",
        "decision": decision, "reason": reason, "aiden_verdict": verdict, "allow_cloud": allow_cloud,
        "failed_routes": failed_routes,
        "authority": {"execution": "NONE", "executed": False,
                      "note": "Model text is not an approval or evidence. The operator decides what happens next."},
        "agents": [outputs[r] for r in ROLES if r in outputs],
        "events": events,
        "receipt": {"kind": row["kind"], "hash": row["hash"], "prev": row["prev"],
                    "chain_valid": chain_valid},
    }


def task_receipts(receipts_path: str) -> Dict[str, Any]:
    import json
    rows: List[Dict[str, Any]] = []
    valid = True
    if os.path.isfile(receipts_path):
        valid = Receipts.verify(receipts_path)
        with open(receipts_path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    return {"path": receipts_path, "chain_valid": valid, "count": len(rows), "rows": rows[-50:]}
