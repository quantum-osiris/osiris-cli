"""Headless governed coding task: model proposes, operator approves, the runner verifies.

    python -m osiris_cli.code_task --root REPO --accept "python -m pytest -q" "fix the failing test"

Stages, each visible in the JSON summary and in the receipt chain:

1. the /code tool loop runs (reads, proposed writes, commands the operator approves);
2. every proposed write is shown as a diff and applied only on approval (interactive y/N,
   or ``--approve-writes``, which is recorded as such);
3. the runner, not the model, runs the ``--accept`` command and checks the applied files'
   hashes (``CodeAgent.verify``).

Exit codes: 0 VERIFIED; 1 FAILED or BLOCKED; 2 model or daemon unavailable; 3 finished
without an acceptance command (UNVERIFIED: nothing proves the work).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional

from osiris_cli.code_agent import CodeAgent, Denied

DEFAULT_RECEIPTS = os.path.join(os.path.expanduser("~"), ".osiris", "code", "receipts.jsonl")


def _stdin_confirm(prompt: str) -> bool:
    sys.stderr.write(prompt)
    sys.stderr.flush()
    line = sys.stdin.readline()
    return line.strip().lower() in ("y", "yes")


def run_task(task: str, *, root: str, accept: Optional[str], model: Optional[str] = None,
             multi: bool = False, approve_writes: bool = False, allow_commands: bool = False,
             require_change: bool = True, receipts: Optional[str] = DEFAULT_RECEIPTS,
             accept_timeout: float = 600.0, chat: Optional[Callable[..., Dict[str, Any]]] = None,
             confirm: Callable[[str], bool] = _stdin_confirm,
             out: Callable[[str], None] = lambda s: print(s, file=sys.stderr)) -> Dict[str, Any]:
    """Run one task through all three stages and return the summary (also the exit decision)."""
    agent = CodeAgent(
        root=root, model=model, chat=chat,
        confirm=(confirm if allow_commands else (lambda prompt: False)),
        propose_write=lambda path, content: True,  # staged in the agent; nothing is written here
        has_pending_write=lambda: False,
        out=out, max_writes=4 if multi else 1, receipts_path=receipts,
        write_nudges=1 if require_change else 0,
    )
    result = agent.run(task)
    summary: Dict[str, Any] = {
        "run_id": result.run_id, "status": result.status, "error": result.error,
        "requested_model": result.requested_model, "resolved_model": result.resolved_model,
        "resolved_digest": result.resolved_digest, "run_receipt": result.receipt_hash,
        "answer": result.answer, "unbacked_claims": result.unbacked_claims,
        "tool_calls": [{k: d.get(k) for k in ("tool", "decision", "reason", "path", "exit_code")
                        if d.get(k) is not None} for d in result.decisions],
        "applied": [], "verification": None,
    }
    if result.status in ("UNAVAILABLE", "MODEL_UNAVAILABLE"):
        summary["disposition"] = result.status
        return summary

    proposals: List[Dict[str, Any]] = list(agent._pending_proposals.values())
    if proposals:
        for p in proposals:
            out(f"\n--- proposed change to {p['rel_path']} ---\n{p['diff'] or '(no textual change)'}")

        def approve(props: List[Dict[str, Any]]) -> bool:
            if approve_writes:
                return True
            return confirm(f"Apply {len(props)} change(s) to {agent.root}? [y/N] ")

        try:
            summary["applied"] = agent.apply_all_pending_writes(approve)
        except Denied as exc:
            summary["applied"] = [{"decision": "DENIED", "reason": str(exc)}]
        # A decline is final for a headless task: the proposal is withdrawn, not left pending.
        agent._pending_proposals.clear()
        agent._pending_proposal = None
        summary["write_approval"] = "operator-flag" if approve_writes else "interactive"

    if not accept:
        summary["disposition"] = "UNVERIFIED"
        summary["verification"] = {"decision": "UNVERIFIED",
                                   "reasons": ["no --accept command: nothing checks this work"]}
        return summary
    v = agent.verify(accept, require_change=require_change, timeout=accept_timeout)
    summary["verification"] = v
    summary["disposition"] = v["decision"]
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m osiris_cli.code_task", description=__doc__.split("\n\n")[0])
    ap.add_argument("task", help="what to do (quote it)")
    ap.add_argument("--root", default=os.getcwd(), help="working directory the tools are confined to")
    ap.add_argument("--accept", help="acceptance command the runner executes after applying, e.g. "
                                     "'python -m pytest -q tests/test_x.py' (one program, no shell)")
    ap.add_argument("--model", help="pin an installed Ollama model (no substitution)")
    ap.add_argument("--multi", action="store_true", help="allow up to 4 staged file writes")
    ap.add_argument("--approve-writes", action="store_true",
                    help="apply proposed writes without a prompt (recorded as operator-flag approval)")
    ap.add_argument("--allow-commands", action="store_true",
                    help="let the model request commands (each still needs a y/N on stdin)")
    ap.add_argument("--no-change-ok", action="store_true", help="VERIFIED is possible without an applied write")
    ap.add_argument("--receipts", default=DEFAULT_RECEIPTS, help="hash-chained receipt file")
    ap.add_argument("--accept-timeout", type=float, default=600.0)
    a = ap.parse_args(argv)
    summary = run_task(a.task, root=a.root, accept=a.accept, model=a.model, multi=a.multi,
                       approve_writes=a.approve_writes, allow_commands=a.allow_commands,
                       require_change=not a.no_change_ok, receipts=a.receipts,
                       accept_timeout=a.accept_timeout)
    print(json.dumps(summary, indent=2, default=str))
    return {"VERIFIED": 0, "UNAVAILABLE": 2, "MODEL_UNAVAILABLE": 2, "UNVERIFIED": 3}.get(
        summary["disposition"], 1)


if __name__ == "__main__":
    sys.exit(main())
