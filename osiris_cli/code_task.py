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

Every run is issued a task context (``osiris_cli.task_context``) before the tool loop starts:
its run ID, the risk tier and cost class of the task's words, which tools the loop may use, and
its budget. The loop is stopped at any cap (STOPPED_BUDGET) and nothing it staged is applied.

Exit codes: 0 SUCCEEDED (VERIFIED); 1 FAILED, BLOCKED or CANCELLED; 2 model or daemon
unavailable; 3 no acceptance command (UNVERIFIED: nothing proves the work); 4 STOPPED_BUDGET.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from osiris_cli import code_critic, session_approval, task_context
from osiris_cli.code_agent import CodeAgent, append_receipt, atomic_write, edit_claims, file_sha256, verify_applied
from osiris_cli.code_runs import DEFAULT_DIR, TERMINAL, RunError, RunStore, source_snapshot
from osiris_cli.pqc_enclave import (
    KeySetupRequired,
    load_or_onboard,
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


def _task_ledger_for_store(store: RunStore, explicit: Optional[str] = None) -> str:
    """The task ledger beside the run store: the shared one for the default store, a private one
    for any other (so a test or a second store never writes ~/.osiris/tasks)."""
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    if os.path.realpath(store.dir) == os.path.realpath(DEFAULT_DIR):
        return task_context.DEFAULT_LEDGER
    return os.path.join(store.dir, "_tasks", "task_runs.jsonl")


def _enclave_dir_for_store(store: RunStore, explicit_dir: Optional[str] = None) -> str:
    if explicit_dir:
        return os.path.abspath(os.path.expanduser(explicit_dir))
    if os.path.realpath(store.dir) == os.path.realpath(DEFAULT_DIR):
        return default_enclave_dir()
    return os.path.join(store.dir, "_enclave")


def _validate_rel_path(rel_path: Any) -> str:
    if not isinstance(rel_path, str) or not rel_path or "\x00" in rel_path:
        raise GrantVerificationError(f"invalid target path {rel_path!r}")
    norm_slash = rel_path.replace("\\", "/")
    if norm_slash.startswith("/") or (len(norm_slash) > 1 and norm_slash[1] == ":"):
        raise GrantVerificationError(f"target path {rel_path!r} must be workspace-relative")
    parts = [seg for seg in norm_slash.split("/") if seg]
    if not parts or any(seg in (".", "..") for seg in parts):
        raise GrantVerificationError(f"target path {rel_path!r} escapes or is non-canonical")
    return "/".join(parts)


def _resolve_confined_target(root: str, rel_path: Any) -> str:
    """Resolve `rel_path` strictly inside `root`, rejecting path traversal, symlinks, and directories."""
    clean_rel = _validate_rel_path(rel_path)
    real_root = os.path.realpath(root)
    cur = real_root
    parts = clean_rel.split("/")
    for idx, part in enumerate(parts):
        cur = os.path.join(cur, part)
        is_last = idx == len(parts) - 1
        try:
            st = os.lstat(cur)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(st.st_mode):
            raise GrantVerificationError(f"target path {rel_path!r} traverses or points to a symlink; refusing write")
        if is_last:
            if stat.S_ISDIR(st.st_mode):
                raise GrantVerificationError(f"target path {rel_path!r} is a directory; refusing write")
            if not stat.S_ISREG(st.st_mode):
                raise GrantVerificationError(f"target path {rel_path!r} is not a regular file; refusing write")
        elif not stat.S_ISDIR(st.st_mode):
            raise GrantVerificationError(f"parent component of {rel_path!r} is not a directory")
    candidate = os.path.normpath(os.path.join(real_root, *parts))
    if not candidate.startswith(real_root.rstrip(os.sep) + os.sep):
        raise GrantVerificationError(f"target path {rel_path!r} resolves outside workspace root")
    return candidate


def _regular_file_sha256(path: str) -> Optional[str]:
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise GrantVerificationError(f"target {path!r} is a symlink or non-regular file")
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _existing_apply_receipt_hash(
    receipts_path: Optional[str], run_id: str, rel_path: str, after_sha256: Optional[str]
) -> Optional[str]:
    if not receipts_path or not os.path.isfile(receipts_path):
        return None
    try:
        with open(receipts_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("kind") != "code.agent.apply_write":
                    continue
                payload = row.get("payload") or {}
                if (
                    payload.get("run_id") == run_id
                    and payload.get("path") == rel_path
                    and payload.get("after_sha256") == after_sha256
                    and isinstance(row.get("hash"), str)
                ):
                    return row["hash"]
    except (OSError, ValueError):
        return None
    return None


def proposals_grant_spec(
    root: str, proposals: Sequence[Dict[str, Any]]
) -> Tuple[str, List[str], Dict[str, Any], str]:
    """Build deterministic `(payload_sha256, scope, operation_schema, summary)` for staged writes."""
    real_root = os.path.realpath(root)
    ordered = sorted(proposals, key=lambda p: str(p["rel_path"]))
    seen_paths = set()
    for p in ordered:
        rel = _validate_rel_path(p.get("rel_path"))
        if rel in seen_paths:
            raise GrantVerificationError(f"duplicate proposal path: {rel}")
        seen_paths.add(rel)
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


def _session_write_grant(signer, run, session_grant, payload_sha, scope, schema, out):
    """A child grant for this run's writes when the session covers every file, else None
    (the caller then asks the operator about this change)."""
    ok, why, sscope = session_approval.verify_session_grant(signer, session_grant)
    covered = (session_approval.covers_paths(sscope, run["root"], [p["rel_path"] for p in run["proposals"]])
               if ok else (False, why))
    if not covered[0]:
        out(f"  [session] this change is not covered ({covered[1]}); asking you")
        return None
    try:
        grant = session_approval.issue_child_grant(
            signer, session_grant, action="code.apply_writes", scope=scope,
            payload_sha256=payload_sha, operation_schema=schema, covered=covered)
    except GrantVerificationError as exc:
        out(f"  [session] not covered ({exc}); asking you")
        return None
    out(f"  [session] applying {len(run['proposals'])} change(s) inside your session approval")
    return grant


def apply_idempotent(
    run: Dict[str, Any],
    store: RunStore,
    *,
    enclave_dir: Optional[str] = None,
    trusted_keys_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Verify the signed operator approval grant against pinned trusted public keys, then
    sequentially apply the run's approved proposals via per-file atomic replacement with
    idempotent same-run resume (not an atomic multi-file transaction)."""
    grant = run.get("approval_grant")
    if not isinstance(grant, dict):
        raise GrantVerificationError("missing signed operator approval grant")

    # Never read enclave_dir or trusted_keys_path from mutable run['options'].
    eff_enclave_dir = _enclave_dir_for_store(store, enclave_dir)
    eff_trusted = trusted_keys_path
    try:
        verifier = PQCEnclaveGate.verifier_from_enclave_dir(
            eff_enclave_dir, trusted_keys_path=eff_trusted
        )
    except Exception as exc:
        raise GrantVerificationError(str(exc)) from exc

    root, receipts = run["root"], run.get("receipts_path")
    expected_sha, expected_scope, _schema, _summary = proposals_grant_spec(
        root, run["proposals"]
    )
    # Verify actual staged content bytes match signed after_sha256 and paths stay confined
    # (no traversal, no symlinks, no directory collisions) BEFORE consuming the grant.
    resolved_targets: List[Tuple[Dict[str, Any], str, bytes]] = []
    for p in run["proposals"]:
        content = p.get("content")
        if not isinstance(content, str):
            raise GrantVerificationError(f"proposal for {p.get('rel_path')!r} missing string content")
        raw_bytes = content.encode("utf-8")
        actual_after = hashlib.sha256(raw_bytes).hexdigest()
        if actual_after != p.get("after_sha256"):
            raise GrantVerificationError(
                f"proposal content SHA-256 mismatch for {p.get('rel_path')!r}: "
                f"{actual_after[:16]}… != {str(p.get('after_sha256'))[:16]}…"
            )
        target = _resolve_confined_target(root, p["rel_path"])
        resolved_targets.append((p, target, raw_bytes))

    # Verify session parent link BEFORE consuming the child grant so an invalid link never spends it.
    ok_link, link_reason = session_approval.verify_child_link(
        verifier, grant, run.get("session_grant")
    )
    if not ok_link:
        raise GrantVerificationError(link_reason)

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

    plan = []
    for p, target, raw_bytes in resolved_targets:
        disk = _regular_file_sha256(target)
        if disk == p["after_sha256"]:
            plan.append((p, target, raw_bytes, False))
        elif disk == p["before_sha256"] and (disk is not None or not p["existed_at_proposal"]):
            plan.append((p, target, raw_bytes, True))
        else:
            raise StaleTarget(
                f"{p['rel_path']} changed since the proposal "
                f"(proposal-time {str(p['before_sha256'])[:12]}, now {str(disk)[:12]})"
            )
    for p, target, raw_bytes, write in plan:
        if write:
            # Re-verify confinement and pre-write digest immediately before replacement.
            target = _resolve_confined_target(root, p["rel_path"])
            disk_now = _regular_file_sha256(target)
            if disk_now != p["before_sha256"] or (disk_now is None and p["existed_at_proposal"]):
                raise StaleTarget(
                    f"{p['rel_path']} changed between validation and write "
                    f"(expected {str(p['before_sha256'])[:12]}, now {str(disk_now)[:12]})"
                )
            atomic_write(target, raw_bytes)
        after_on_disk = _regular_file_sha256(target)
        rec = {
            "tool": "apply_write",
            "decision": "GRANTED",
            "run_id": run["run_id"],
            "path": p["rel_path"],
            "before_sha256": p["before_sha256"],
            "after_sha256": after_on_disk,
            "approval": run.get("write_approval"),
            "grant_id": grant["grant_id"],
            "key_fingerprint": grant["key_fingerprint"],
            "approval_mode": grant.get("approval_mode"),
            "session_grant_id": (run.get("session_grant") or {}).get("grant_id")
            if grant.get("approval_mode") == "session" else None,
            "already_applied": not write,
        }
        prior = next((a for a in run.get("applied", []) if a["path"] == p["rel_path"]), None)
        if prior and prior.get("receipt_hash"):
            rec["receipt_hash"] = prior["receipt_hash"]
        elif receipts:
            existing_hash = _existing_apply_receipt_hash(
                receipts, run["run_id"], p["rel_path"], after_on_disk
            )
            if existing_hash:
                rec["receipt_hash"] = existing_hash
            else:
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
    # Derive enclave & trust store strictly from caller args / store default, never run['options'].
    eff_enclave_dir = _enclave_dir_for_store(store, enclave_dir)
    eff_trusted = trusted_keys_path

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
                    signer = load_or_onboard(eff_enclave_dir, confirm=confirm, out=out)
                    grant = None
                    session_grant = run.get("session_grant")
                    if session_grant is not None:
                        grant = _session_write_grant(signer, run, session_grant, payload_sha, scope, schema, out)
                    if grant is None:
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
                except KeySetupRequired as exc:
                    run["verification"] = {"decision": "BLOCKED", "reasons": [str(exc)]}
                    store.transition(run, "BLOCKED", "no signing key set up; nothing written")
                    out(f"\n{exc}")
                    break
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
        "task_run": run.get("task_run"),
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
    session: Optional[session_approval.SessionScope] = None,
    context: Optional[task_context.TaskRunContext] = None,
    task_ledger: Optional[str] = None,
    clock: Optional[Callable[[], float]] = None,
) -> Dict[str, Any]:
    """Run one task through every stage and return its summary.

    ``context`` is the run's task context. When none is given one is minted here, by code, from
    the task's words (risk tier, cost class) and the operator's flags (what the loop may do); it
    is never read from the model. The run takes the context's run ID. ``task_ledger`` is the
    path the run is opened and closed in; by default the ledger beside the run store.

    ``session`` is an operator-approved scope (see ``osiris_cli.session_approval``): the
    operator approves it once at the start, and writes and commands inside it then need no
    further prompt. Anything outside it is asked about as before.

    ``stop_after`` ends this process's work at that state, as a crash would (tests).
    """
    store = store or RunStore(runs_dir or DEFAULT_DIR)
    eff_enclave_dir = _enclave_dir_for_store(store, enclave_dir)
    cmd_enclave = None
    session_grant = None
    if allow_commands or session is not None:
        cmd_enclave = load_or_onboard(eff_enclave_dir, confirm=confirm, out=out)
    if session is not None:
        # The one approval for the whole session. Declining it raises ApprovalDeniedError.
        session_grant = session_approval.request_session_grant(cmd_enclave, session, confirm=confirm, out=out)
    max_writes = 4 if multi else 1
    if context is None:
        context = task_context.new_context(
            task, kind="code", run_id=f"code-{uuid.uuid4().hex[:12]}", max_writes=max_writes,
            perms=task_context.permissions(propose_writes=True, run_commands=allow_commands))
    run_id = context.run_id
    ledger_path = _task_ledger_for_store(store, task_ledger)
    agent = CodeAgent(
        root=root,
        model=model,
        chat=chat,
        confirm=(confirm if allow_commands else (lambda prompt: False)),
        propose_write=lambda path, content: True,  # staged in the agent; nothing is written here
        has_pending_write=lambda: False,
        out=out,
        max_writes=max_writes,
        receipts_path=receipts,
        write_nudges=1 if require_change else 0,
        enclave=cmd_enclave,
        session_grant=session_grant if allow_commands else None,
        context=context,
        task_ledger=task_context.TaskLedger(ledger_path),
        **({"clock": clock} if clock is not None else {}),
    )
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
    if session_grant is not None:
        run["session_grant"] = session_grant
    run["task_run"] = {"context": context.to_dict(), "ledger": ledger_path, "stop_reason": None, "usage": {}}
    store.save(run)
    out(f"  task run {context.summary()}")
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
                    for k in ("tool", "decision", "reason", "path", "exit_code", "grant_id", "approval_mode")
                    if d.get(k) is not None
                }
                for d in result.decisions
            ],
        )
        run["task_run"].update(stop_reason=result.stop_reason, usage=result.usage)
        for d in result.decisions:
            if d.get("decision") == "RETRY":
                store.event(run, "RETRYING", d.get("reason", ""))
        if result.status == task_context.STOPPED_BUDGET:
            store.transition(run, "STOPPED_BUDGET", f"{result.error}; nothing written")
            return summarize(run)
        if result.status == task_context.BLOCKED:
            store.transition(run, "BLOCKED", f"{result.error}; nothing written")
            return summarize(run)
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
    return {"VERIFIED": 0, "UNAVAILABLE": 2, "UNVERIFIED": 3, "STOPPED_BUDGET": 4}.get(summary["disposition"], 1)


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
    ap.add_argument(
        "--session-edit", action="append", default=[], metavar="GLOB",
        help="session approval: files the agent may change without asking, relative to --root "
        "(repeatable, e.g. 'src/*.py'). You approve the whole session once at the start.",
    )
    ap.add_argument(
        "--session-command", action="append", default=[], metavar="CMD",
        help="session approval: an exact test/build command that may run without asking "
        "(repeatable, e.g. 'python -m pytest -q'). Shells, network, installs, deletes, "
        "pushes and deploys are refused.",
    )
    ap.add_argument("--session-minutes", type=int, default=session_approval.DEFAULT_SESSION_MINUTES)
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
            session = None
            if a.session_edit or a.session_command:
                try:
                    session = session_approval.SessionScope.build(
                        a.root, a.session_edit, a.session_command, a.session_minutes)
                except session_approval.SessionScopeError as exc:
                    _err(f"[code_task] session not started: {exc}")
                    return 2
            try:
                summary = _run_cli_task(a, store, session)
            except KeySetupRequired as exc:
                _err(f"[code_task] {exc}")
                return 2
            except ApprovalDeniedError:
                _err("[code_task] session not approved; nothing ran")
                return 1
    except RunError as exc:
        _err(f"[code_task] {exc}")
        return 1
    print(json.dumps(summary, indent=2, default=str))
    return exit_code(summary)


def _run_cli_task(a: argparse.Namespace, store: RunStore, session: Optional[session_approval.SessionScope]):
    return run_task(
        a.task,
        root=a.root,
        accept=a.accept,
        model=a.model,
        multi=a.multi,
        approve_writes=a.approve_writes,
        allow_commands=a.allow_commands or bool(session and session.commands),
        require_change=not a.no_change_ok,
        receipts=a.receipts,
        accept_timeout=a.accept_timeout,
        store=store,
        critic_model=a.critic_model,
        allow_test_edits=a.allow_test_edits,
        enclave_dir=a.enclave_dir,
        session=session,
            )


if __name__ == "__main__":
    sys.exit(main())
