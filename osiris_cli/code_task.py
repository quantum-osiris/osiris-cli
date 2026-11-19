"""Headless governed coding task: model proposes, critic screens, operator approves, runner verifies.

    python -m osiris_cli.code_task --root REPO --accept "python -m pytest -q" "fix the failing test"
    python -m osiris_cli.code_task --list | --status RUN | --resume RUN | --cancel RUN

Stages, each a state in the durable run record (``osiris_cli.code_runs``) and receipted:

1. CONTEXT_READY: source snapshot (git HEAD, status digest);
2. RUNNING: the /code tool loop (reads, staged writes, commands only with ``--allow-commands``
   plus a y/N each); a model that answers with a prose diff gets one recorded RETRYING nudge;
3. critic (``osiris_cli.code_critic``): protected paths, test edits, AST firewall, optional
   second-model review. It can only block, before anything is written;
4. AWAITING_APPROVAL: each diff plus a readable action/scope summary is shown; on approval
   (y/N on stdin, or ``--approve-writes``), the local enclave signs a hybrid Ed25519 + ML-DSA-65
   grant bound to the exact root, file paths, and before/after hashes. ``apply_idempotent``
   verifies that grant against explicitly trusted operator public keys before any write;
5. VERIFYING: the runner, not the model, runs the ``--accept`` command and checks hashes.

Exit codes: 0 SUCCEEDED (VERIFIED); 1 FAILED, BLOCKED or CANCELLED; 2 model or daemon
unavailable; 3 no acceptance command (UNVERIFIED: nothing proves the work).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from osiris_cli import code_critic
from osiris_cli.code_agent import CodeAgent, append_receipt, atomic_write, edit_claims, file_sha256, verify_applied
from osiris_cli.code_runs import DEFAULT_DIR, TERMINAL, RunError, RunStore, source_snapshot
from osiris_cli.pqc_enclave import (
    ApprovalDeniedError,
    GrantVerificationError,
    PQCEnclaveGate,
    default_enclave_dir,
)

DEFAULT_RECEIPTS = os.path.join(os.path.expanduser("~"), ".osiris", "code", "receipts.jsonl")
NO_ACCEPT = "no --accept command: nothing checks this work"


def _stdin_confirm(prompt: str) -> bool:
    sys.stderr.write(prompt)
    sys.stderr.flush()
    return sys.stdin.readline().strip().lower() in ("y", "yes")


def _err(s: str) -> None:
    print(s, file=sys.stderr)


class StaleTarget(Exception):
    pass


def _enclave_dir_for_store(store: RunStore, explicit_dir: Optional[str] = None) -> str:
    if explicit_dir:
        return os.path.abspath(os.path.expanduser(explicit_dir))
    if os.path.realpath(store.dir) == os.path.realpath(DEFAULT_DIR):
        return default_enclave_dir()
    return os.path.join(store.dir, "_enclave")


def proposals_grant_spec(
    root: str, proposals: Sequence[Dict[str, Any]]
) -> Tuple[str, List[str], Dict[str, Any], str]:
    """Build deterministic `(payload_sha256, scope, operation_schema, summary)` for staged writes."""
    real_root = os.path.realpath(root)
    ordered = sorted(proposals, key=lambda p: str(p["rel_path"]))
    files_spec = [
        {
            "after_sha256": p["after_sha256"],
            "before_sha256": p["before_sha256"],
            "existed_at_proposal": bool(p["existed_at_proposal"]),
            "rel_path": p["rel_path"],
        }
        for p in ordered
    ]
    schema: Dict[str, Any] = {
        "action": "code.apply_writes",
        "files": files_spec,
        "root": real_root,
    }
    canon = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    payload_sha256 = hashlib.sha256(canon.encode("utf-8")).hexdigest()
    scope = ["code.apply_writes", real_root, *[p["rel_path"] for p in ordered]]
    file_parts = ", ".join(
        f"{p['rel_path']} ({(p['before_sha256'] or 'new')[:10]}->{str(p['after_sha256'])[:10]})"
        for p in ordered
    )
    summary = f"Apply {len(ordered)} staged file change(s): {file_parts}"
    return payload_sha256, scope, schema, summary


def apply_idempotent(
    run: Dict[str, Any],
    store: RunStore,
    *,
    enclave_dir: Optional[str] = None,
    trusted_keys_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Verify the signed operator approval grant against trusted public keys, then apply
    the run's approved proposals at most once."""
    grant = run.get("approval_grant")
    if not isinstance(grant, dict):
        raise GrantVerificationError("missing signed operator approval grant")

    opts = run.get("options") or {}
    eff_enclave_dir = _enclave_dir_for_store(store, enclave_dir or opts.get("enclave_dir"))
    eff_trusted = trusted_keys_path or opts.get("trusted_keys_path")
    try:
        verifier = PQCEnclaveGate.verifier_from_enclave_dir(
            eff_enclave_dir, trusted_keys_path=eff_trusted
        )
    except Exception as exc:
        raise GrantVerificationError(str(exc)) from exc

    expected_sha, expected_scope, _schema, _summary = proposals_grant_spec(
        run["root"], run["proposals"]
    )
    ok, reason = verifier.verify_grant_integrity(
        grant,
        expected_payload_sha256=expected_sha,
        expected_scope=expected_scope,
        expected_event_type="code.apply_writes",
        consume=True,
        consumer_id=run["run_id"],
    )
    if not ok:
        raise GrantVerificationError(reason)

    root, receipts = run["root"], run.get("receipts_path")
    plan = []
    for p in run["proposals"]:
        disk = file_sha256(os.path.join(root, p["rel_path"]))
        if disk == p["after_sha256"]:
            plan.append((p, False))
        elif disk == p["before_sha256"] and (disk is not None or not p["existed_at_proposal"]):
            plan.append((p, True))
        else:
            raise StaleTarget(
                f"{p['rel_path']} changed since the proposal "
                f"(proposal-time {str(p['before_sha256'])[:12]}, now {str(disk)[:12]})"
            )
    for p, write in plan:
        target = os.path.join(root, p["rel_path"])
        if write:
            atomic_write(target, p["content"].encode("utf-8"))
        rec = {
            "tool": "apply_write",
            "decision": "GRANTED",
            "run_id": run["run_id"],
            "path": p["rel_path"],
            "before_sha256": p["before_sha256"],
            "after_sha256": file_sha256(target),
            "approval": run.get("write_approval"),
            "grant_id": grant["grant_id"],
            "key_fingerprint": grant["key_fingerprint"],
            "already_applied": not write,
        }
        prior = next((a for a in run.get("applied", []) if a["path"] == p["rel_path"]), None)
        if prior and prior.get("receipt_hash"):
            rec["receipt_hash"] = prior["receipt_hash"]
        elif receipts:
            rec["receipt_hash"] = append_receipt(receipts, "code.agent.apply_write", rec)
        run["applied"] = [a for a in run.get("applied", []) if a["path"] != p["rel_path"]] + [rec]
        store.save(run)  # each write is on record before the next one starts
    return run["applied"]


def _advance(
    store: RunStore,
    run: Dict[str, Any],
    *,
    approve_writes: bool,
    confirm: Callable[[str], bool],
    out: Callable[[str], None],
    enclave_dir: Optional[str] = None,
    trusted_keys_path: Optional[str] = None,
) -> Dict[str, Any]:
    opts = run["options"]
    eff_enclave_dir = _enclave_dir_for_store(store, enclave_dir or opts.get("enclave_dir"))
    eff_trusted = trusted_keys_path or opts.get("trusted_keys_path")

    while run["state"] not in TERMINAL:
        if store.cancel_requested(run["run_id"]):
            store.transition(run, "CANCELLED", "cancel requested")
            break
        state = run["state"]
        if state in ("CREATED", "CONTEXT_READY", "RUNNING"):
            store.transition(
                run,
                "FAILED",
                f"interrupted in {state}; writes are only staged before "
                "AWAITING_APPROVAL, so nothing was written; start a new run",
            )
        elif state == "AWAITING_APPROVAL":
            for p in run["proposals"]:
                out(f"\n--- proposed change to {p['rel_path']} ---\n{p['diff'] or '(no textual change)'}")
            if not isinstance(run.get("approval_grant"), dict):
                payload_sha, scope, schema, summary_text = proposals_grant_spec(
                    run["root"], run["proposals"]
                )
                try:
                    signer = PQCEnclaveGate.load_or_init_local(eff_enclave_dir)
                    grant = signer.request_approved_grant(
                        action="code.apply_writes",
                        scope=scope,
                        summary=summary_text,
                        payload_sha256=payload_sha,
                        operation_schema=schema,
                        confirm=confirm,
                        out=out,
                        prompt=f"Apply {len(run['proposals'])} change(s) to {run['root']}? [y/N] ",
                        approved_by_flag=approve_writes,
                    )
                except ApprovalDeniedError:
                    run["write_approval"] = None
                    run["applied"] = [
                        {
                            "tool": "apply_write",
                            "decision": "DENIED",
                            "path": p["rel_path"],
                            "reason": "operator declined",
                        }
                        for p in run["proposals"]
                    ]
                    run["verification"] = {"decision": "FAILED", "reasons": ["operator declined the change"]}
                    store.transition(run, "FAILED", "operator declined the change; nothing written")
                    break
                except (GrantVerificationError, RuntimeError, ValueError) as exc:
                    run["verification"] = {"decision": "BLOCKED", "reasons": [f"grant signing refused: {exc}"]}
                    store.transition(run, "BLOCKED", f"grant signing refused: {exc}; nothing written")
                    break

                run["approval_grant"] = grant
                run["write_approval"] = grant.get("approval_mode")
                store.save(run)

            try:
                apply_idempotent(
                    run,
                    store,
                    enclave_dir=eff_enclave_dir,
                    trusted_keys_path=eff_trusted,
                )
            except StaleTarget as exc:
                store.transition(run, "BLOCKED", f"stale target: {exc}; nothing written")
                break
            except GrantVerificationError as exc:
                run["verification"] = {"decision": "BLOCKED", "reasons": [f"grant verification failed: {exc}"]}
                store.transition(run, "BLOCKED", f"grant verification failed: {exc}; nothing written")
                break
            store.transition(
                run,
                "VERIFYING",
                "applied "
                + ", ".join(
                    f"{a['path']}{' (already applied)' if a['already_applied'] else ''}"
                    for a in run["applied"]
                ),
            )
        elif state == "VERIFYING":
            if not run.get("accept"):
                run["verification"] = {"decision": "UNVERIFIED", "reasons": [NO_ACCEPT]}
                store.transition(run, "BLOCKED", NO_ACCEPT)
                break
            applied = [a for a in run.get("applied", []) if a.get("decision") == "GRANTED"]
            v = verify_applied(
                run["root"],
                applied,
                run["accept"],
                require_change=opts["require_change"],
                timeout=opts["accept_timeout"],
                run_id=run["run_id"],
                unbacked=[] if applied else edit_claims(run.get("answer", "")),
            )
            if run.get("receipts_path"):
                v["receipt_hash"] = append_receipt(
                    run["receipts_path"],
                    "code.agent.verify",
                    {k: x for k, x in v.items() if k != "output_tail"},
                )
            run["verification"] = v
            if store.cancel_requested(run["run_id"]):
                store.transition(
                    run,
                    "CANCELLED",
                    f"cancel arrived during verification ({v['decision']}); result not promoted",
                )
            elif v["decision"] == "VERIFIED":
                store.transition(run, "SUCCEEDED", f"acceptance exited 0 in {v.get('elapsed_s')}s")
            else:
                store.transition(run, v["decision"], "; ".join(v["reasons"]))
    return run


def summarize(run: Dict[str, Any]) -> Dict[str, Any]:
    state = run["state"]
    v = run.get("verification") or None
    disposition = {"SUCCEEDED": "VERIFIED"}.get(state, state)
    if state == "BLOCKED" and v and v.get("decision") == "UNVERIFIED":
        disposition = "UNVERIFIED"
    model = run.get("model") or {}
    grant = run.get("approval_grant") if isinstance(run.get("approval_grant"), dict) else {}
    return {
        "run_id": run["run_id"],
        "state": state,
        "disposition": disposition,
        "status": run.get("status"),
        "error": run.get("error", ""),
        "requested_model": model.get("requested"),
        "resolved_model": model.get("resolved"),
        "resolved_digest": model.get("digest"),
        "run_receipt": run.get("run_receipt"),
        "answer": run.get("answer", ""),
        "unbacked_claims": run.get("unbacked_claims", []),
        "tool_calls": run.get("tool_calls", []),
        "critic": run.get("critic"),
        "applied": run.get("applied", []),
        "write_approval": run.get("write_approval"),
        "grant_id": grant.get("grant_id"),
        "key_fingerprint": grant.get("key_fingerprint"),
        "verification": v,
        "source": run.get("source"),
        "history": run["history"],
    }


def run_task(
    task: str,
    *,
    root: str,
    accept: Optional[str],
    model: Optional[str] = None,
    multi: bool = False,
    approve_writes: bool = False,
    allow_commands: bool = False,
    require_change: bool = True,
    receipts: Optional[str] = DEFAULT_RECEIPTS,
    accept_timeout: float = 600.0,
    chat: Optional[Callable[..., Dict[str, Any]]] = None,
    confirm: Callable[[str], bool] = _stdin_confirm,
    out: Callable[[str], None] = _err,
    runs_dir: Optional[str] = None,
    store: Optional[RunStore] = None,
    critic_model: Optional[str] = None,
    allow_test_edits: bool = False,
    critic_chat: Optional[Callable[..., Dict[str, Any]]] = None,
    critic_installed: Optional[Dict[str, str]] = None,
    stop_after: Optional[str] = None,
    enclave_dir: Optional[str] = None,
    trusted_keys_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one task through every stage and return its summary.

    ``stop_after`` ends this process's work at that state, as a crash would (tests).
    """
    store = store or RunStore(runs_dir or DEFAULT_DIR)
    eff_enclave_dir = _enclave_dir_for_store(store, enclave_dir)
    cmd_enclave = PQCEnclaveGate.load_or_init_local(eff_enclave_dir) if allow_commands else None
    agent = CodeAgent(
        root=root,
        model=model,
        chat=chat,
        confirm=(confirm if allow_commands else (lambda prompt: False)),
        propose_write=lambda path, content: True,  # staged in the agent; nothing is written here
        has_pending_write=lambda: False,
        out=out,
        max_writes=4 if multi else 1,
        receipts_path=receipts,
        write_nudges=1 if require_change else 0,
        enclave=cmd_enclave,
    )
    run_id = f"code-{uuid.uuid4().hex[:12]}"
    run = store.create(
        run_id,
        task=task,
        task_sha256=hashlib.sha256(task.encode("utf-8")).hexdigest(),
        root=agent.root,
        accept=accept,
        receipts_path=receipts,
        model={"requested": model},
        options={
            "require_change": require_change,
            "accept_timeout": accept_timeout,
            "multi": multi,
            "allow_commands": allow_commands,
            "critic_model": critic_model,
            "allow_test_edits": allow_test_edits,
            "enclave_dir": eff_enclave_dir,
            "trusted_keys_path": trusted_keys_path,
        },
    )
    with store.locked(run_id):
        run["source"] = source_snapshot(agent.root)
        store.transition(run, "CONTEXT_READY", f"git HEAD {run['source']['git_head']}")
        if store.cancel_requested(run_id):
            store.transition(run, "CANCELLED", "cancel requested")
            return summarize(run)
        store.transition(run, "RUNNING", f"model {agent.model}")
        if stop_after == "RUNNING":
            return summarize(run)
        result = agent.run(task, run_id=run_id)
        run.update(
            status=result.status,
            error=result.error,
            answer=result.answer,
            unbacked_claims=result.unbacked_claims,
            run_receipt=result.receipt_hash,
            model={
                "requested": result.requested_model,
                "resolved": result.resolved_model,
                "digest": result.resolved_digest,
            },
            tool_calls=[
                {
                    k: d.get(k)
                    for k in ("tool", "decision", "reason", "path", "exit_code", "grant_id")
                    if d.get(k) is not None
                }
                for d in result.decisions
            ],
        )
        for d in result.decisions:
            if d.get("decision") == "RETRY":
                store.event(run, "RETRYING", d.get("reason", ""))
        if result.status in ("UNAVAILABLE", "MODEL_UNAVAILABLE"):
            store.transition(run, "UNAVAILABLE", result.error)
            return summarize(run)
        if result.status != "SUCCESS":
            store.transition(run, "FAILED", f"tool loop ended {result.status}: {result.error}")
            return summarize(run)
        run["proposals"] = [
            {
                k: p[k]
                for k in (
                    "rel_path",
                    "existed_at_proposal",
                    "before_sha256",
                    "after_sha256",
                    "content",
                    "diff",
                )
            }
            for p in agent._pending_proposals.values()
        ]
        if run["proposals"]:
            for_critic = [
                dict(
                    p,
                    before_content=(
                        open(
                            os.path.join(agent.root, p["rel_path"]),
                            encoding="utf-8",
                            errors="replace",
                        ).read()
                        if p["existed_at_proposal"]
                        else ""
                    ),
                )
                for p in run["proposals"]
            ]
            run["critic"] = code_critic.critique(
                task,
                for_critic,
                accept=accept,
                allow_test_edits=allow_test_edits,
                critic_model=critic_model,
                author_model=result.resolved_model,
                chat=critic_chat,
                installed=critic_installed,
            )
            if run["critic"]["verdict"] == "BLOCK":
                store.transition(
                    run,
                    "BLOCKED",
                    "critic: "
                    + "; ".join(f["reason"] for f in run["critic"]["findings"])
                    + "; nothing written",
                )
                return summarize(run)
            store.transition(run, "AWAITING_APPROVAL", f"{len(run['proposals'])} proposal(s), critic PASS")
        else:
            store.transition(run, "VERIFYING", "no proposals")
        if stop_after == run["state"]:
            return summarize(run)
        return summarize(
            _advance(
                store,
                run,
                approve_writes=approve_writes,
                confirm=confirm,
                out=out,
                enclave_dir=eff_enclave_dir,
                trusted_keys_path=trusted_keys_path,
            )
        )


def resume(
    run_id: str,
    *,
    store: RunStore,
    approve_writes: bool = False,
    confirm: Callable[[str], bool] = _stdin_confirm,
    out: Callable[[str], None] = _err,
    enclave_dir: Optional[str] = None,
    trusted_keys_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Continue a run from its last committed state, with the options it was created with."""
    with store.locked(run_id):
        run = store.load(run_id)
        if run["state"] in TERMINAL:
            raise RunError(f"run {run_id} is already {run['state']}; nothing to resume")
        store.event(run, "RESUMED", f"from {run['state']}")
        return summarize(
            _advance(
                store,
                run,
                approve_writes=approve_writes,
                confirm=confirm,
                out=out,
                enclave_dir=enclave_dir,
                trusted_keys_path=trusted_keys_path,
            )
        )


def exit_code(summary: Dict[str, Any]) -> int:
    return {"VERIFIED": 0, "UNAVAILABLE": 2, "UNVERIFIED": 3}.get(summary["disposition"], 1)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m osiris_cli.code_task", description=__doc__.split("\n\n")[0])
    ap.add_argument("task", nargs="?", help="what to do (quote it)")
    ap.add_argument("--root", default=os.getcwd(), help="working directory the tools are confined to")
    ap.add_argument(
        "--accept",
        help="acceptance command the runner executes after applying, e.g. "
        "'python -m pytest -q tests/test_x.py' (one program, no shell)",
    )
    ap.add_argument("--model", help="pin an installed Ollama model (no substitution)")
    ap.add_argument(
        "--critic-model", help="second, different Ollama model that reviews the diff (can only block)"
    )
    ap.add_argument("--allow-test-edits", action="store_true", help="let the patch change tests")
    ap.add_argument("--multi", action="store_true", help="allow up to 4 staged file writes")
    ap.add_argument(
        "--approve-writes",
        action="store_true",
        help="apply proposed writes without a prompt (recorded as operator-flag approval)",
    )
    ap.add_argument(
        "--allow-commands",
        action="store_true",
        help="let the model request commands (each still needs a y/N on stdin)",
    )
    ap.add_argument("--no-change-ok", action="store_true", help="VERIFIED is possible without an applied write")
    ap.add_argument("--receipts", default=DEFAULT_RECEIPTS, help="hash-chained receipt file")
    ap.add_argument("--runs-dir", default=DEFAULT_DIR, help="durable run records")
    ap.add_argument("--enclave-dir", default=None, help="local operator PQC enclave key directory")
    ap.add_argument("--accept-timeout", type=float, default=600.0)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--resume", metavar="RUN_ID")
    g.add_argument("--cancel", metavar="RUN_ID")
    g.add_argument("--status", metavar="RUN_ID")
    g.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    store = RunStore(a.runs_dir)
    try:
        if a.list:
            print(json.dumps(store.list(), indent=2))
            return 0
        if a.status:
            print(json.dumps(summarize(store.load(a.status)), indent=2, default=str))
            return 0
        if a.cancel:
            store.request_cancel(a.cancel)
            print(json.dumps({"run_id": a.cancel, "cancel_requested": True}))
            return 0
        if a.resume:
            summary = resume(
                a.resume, store=store, approve_writes=a.approve_writes, enclave_dir=a.enclave_dir
            )
        else:
            if not a.task:
                ap.error("a task is required (or --list/--status/--resume/--cancel)")
            summary = run_task(
                a.task,
                root=a.root,
                accept=a.accept,
                model=a.model,
                multi=a.multi,
                approve_writes=a.approve_writes,
                allow_commands=a.allow_commands,
                require_change=not a.no_change_ok,
                receipts=a.receipts,
                accept_timeout=a.accept_timeout,
                store=store,
                critic_model=a.critic_model,
                allow_test_edits=a.allow_test_edits,
                enclave_dir=a.enclave_dir,
            )
    except RunError as exc:
        _err(f"[code_task] {exc}")
        return 1
    print(json.dumps(summary, indent=2, default=str))
    return exit_code(summary)


if __name__ == "__main__":
    sys.exit(main())
