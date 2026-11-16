"""Bounded editor assistance, answered by a model that OSIRIS's router selects.

Four actions only. Each returns text, and ``propose`` / ``example`` may also return a complete
replacement source. Nothing here changes a file or runs anything: a proposal is data for the
editor to show as a diff, and it is compiled here only to report whether the real compiler
accepts it. The route record says which model answered and whether the request left the machine.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Optional

ACTIONS = ("explain_selection", "explain_diagnostic", "propose_change", "generate_example")
MAX_SOURCE = 16_000
MAX_TEXT = 2_000
ROLE = "EDITOR"

GUIDE = """You assist with dna::}{::lang 0.4 circuit genes. The ONLY supported syntax is:

organism NAME {
  meta { key: "value" }                      // optional, metadata only
  gene NAME(param: angle) on a, b {          // params optional; ports are qubit names
    helix a;            // H          bond a, b;   // CX
    twist(theta) a;     // RZ         fold(theta) a; // RY     splice(theta) a; // RX
    x a; y a; z a; s a; t a; sx a; cz a, b; swap a, b;
    wait(100ns) a;  echo(X) a;  barrier a, b;
    cleave a -> ca;     // measure qubit a into classical bit ca
  }
  genome { NAME(0.5) @ [0, 1]; }             // instantiate a gene on qubit indices
}

Comments are // or /* */. '#' comments, Python statements, EVOLUTION / PHENOTYPE / CHROMOSOME
blocks and print() are NOT part of the language. Do not claim a program was run, and do not
state measurement results, fidelities or hardware behaviour."""


class AssistError(ValueError):
    pass


def _text(value: Any, name: str, limit: int, required: bool = True) -> str:
    if value is None and not required:
        return ""
    if not isinstance(value, str) or (required and not value.strip()):
        raise AssistError(f"{name} is required")
    if len(value) > limit:
        raise AssistError(f"{name} is longer than {limit} characters")
    return value


def extract_source(text: str) -> Optional[str]:
    """The last fenced block that starts with ``organism``; None when there is none."""
    blocks = re.findall(r"```[A-Za-z0-9_:+{}-]*\n(.*?)```", text, flags=re.S)
    for block in reversed(blocks):
        if re.match(r"\s*(//[^\n]*\n|/\*.*?\*/|\s)*organism\b", block, flags=re.S | re.I):
            return block.strip() + "\n"
    return None


def assist(body: Dict[str, Any], *, route_fn: Optional[Callable[..., Any]] = None,
           compile_fn: Optional[Callable[[str], Dict[str, Any]]] = None) -> Dict[str, Any]:
    from osiris_cli.orchestration import model_router
    action = body.get("action")
    if action not in ACTIONS:
        raise AssistError("action must be one of " + ", ".join(ACTIONS))
    source = _text(body.get("source"), "source", MAX_SOURCE, required=action != "generate_example")
    if action == "explain_selection":
        selection = _text(body.get("selection"), "selection", MAX_TEXT)
        prompt = (f"Program:\n```\n{source}\n```\nExplain what this selected part does, in at most 120 words:\n"
                  f"```\n{selection}\n```")
    elif action == "explain_diagnostic":
        diagnostic = _text(body.get("diagnostic"), "diagnostic", MAX_TEXT)
        prompt = (f"Program:\n```\n{source}\n```\nThe compiler reported:\n{diagnostic}\n"
                  "Explain the cause in at most 120 words and say what to change. Do not rewrite the whole program.")
    elif action == "propose_change":
        instruction = _text(body.get("instruction"), "instruction", MAX_TEXT)
        prompt = (f"Program:\n```\n{source}\n```\nRequested change: {instruction}\n"
                  "Reply with one sentence describing the change, then the COMPLETE revised program in one "
                  "fenced code block that starts with the word organism.")
    else:
        instruction = _text(body.get("instruction"), "instruction", MAX_TEXT)
        prompt = (f"Write a small example program for: {instruction}\n"
                  "Reply with one sentence, then the COMPLETE program in one fenced code block that starts with "
                  "the word organism.")
    cloud = body.get("allow_cloud") is True          # only a literal true lets the request leave this machine
    routes = model_router.load_routes()
    table = {ROLE: routes.get(ROLE) or routes.get("AURA") or []}
    try:
        text, record = (route_fn or model_router.route)(ROLE, GUIDE, prompt, cloud_allowed=cloud, routes=table,
                                                        max_tokens=900, temperature=0.2)
    except model_router.RouteUnavailable as e:
        return {"status": "UNAVAILABLE", "action": action, "text": None, "proposed_source": None,
                "proposal_compiles": None, "route": e.route, "applied": False}
    out: Dict[str, Any] = {"status": "ANSWERED", "action": action, "text": text[:8000], "proposed_source": None,
                           "proposal_compiles": None, "proposal_diagnostics": None, "route": record, "applied": False}
    if action in ("propose_change", "generate_example"):
        proposed = extract_source(text)
        if proposed is not None:
            if compile_fn is None:
                from osiris_cli.vertical_slice import compile_source as compile_fn  # type: ignore[no-redef]
            comp = compile_fn(proposed)
            out.update(proposed_source=proposed, proposal_compiles=bool(comp["ok"]),
                       proposal_diagnostics=comp["diagnostics"])
    return out
