"""Focused end-to-end tests for the local operator grant-signing workflow.

Covers:
1. `PQCEnclaveGate.generate_hybrid_grant` rejects manual `ed25519_sig_hex` / `pqc_sig_hex`
   arguments (`TypeError`) instead of silently ignoring them.
2. Readable local approval workflow in `code_task` and `CodeAgent`: displays action, scope,
   summary, digest, signer fingerprint, and TTL; asks `[y/N]` without asking the operator to
   create or paste signature hex; signs locally and verifies against `trusted_operator_keys.json`.
3. Missing or declined operator approval fails closed with zero grant minted and zero writes.
4. Changed payload, changed `operation_schema`, or changed `scope` after signing fails closed.
5. Wrong / untrusted signer fails closed even when the untrusted grant embeds its own valid
   public keys and valid signatures inside `pqc_envelope`.
6. Malformed or bit-flipped `Ed25519` / `ML-DSA-65` signatures fail closed.
7. Expiry (`now > expires_at`) and future-dated (`now < issued_at`) grants fail closed.
8. Single-use replay protection blocks reusing a consumed grant across runs.
9. Local private keys keep `0600` permissions, are never silently overwritten on reload or
   partial corruption, and never appear in `repr()`, console output, run records, or receipts.
"""
from __future__ import annotations

import copy
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from osiris_cli.code_agent import CodeAgent
from osiris_cli.code_runs import RunStore
from osiris_cli.code_task import apply_idempotent, proposals_grant_spec, resume, run_task
from osiris_cli.pqc_enclave import (
    ApprovalDeniedError,
    GrantVerificationError,
    PQCEnclaveGate,
)
from osiris_cli.trio_authz import Receipts

BUGGY = "def add(a: int, b: int) -> int:\n    return a - b\n"
FIXED = "def add(a: int, b: int) -> int:\n    return a + b\n"
TEST_SRC = "from calc import add\n\ndef test_add():\n    assert add(4, 5) == 9\n"


KEY_DIRS = ("enclave", "runs/_enclave", "agent_enclave")

@pytest.fixture(autouse=True)
def _signing_key_set_up(tmp_path):
    """Keys are no longer created on first use; set them up the explicit way, as an operator would."""
    from osiris_cli.pqc_enclave import setup_local_keys
    for d in KEY_DIRS:
        setup_local_keys(str(tmp_path / d), confirm=lambda prompt: True)


def _call(name: str, **args):
    return {"content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def _scripted(*steps, final: str = "I proposed updating calc.py."):
    items = list(steps) + [{"content": final}]
    return lambda messages, tools, model: items.pop(0) if items else {"content": final}


@pytest.fixture
def disposable_repo(tmp_path: Path, monkeypatch) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text(BUGGY, encoding="utf-8")
    (repo / "test_calc.py").write_text(TEST_SRC, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"], cwd=repo, check=True)
    monkeypatch.delenv("OSIRIS_TEXT_TOOL_CALLS", raising=False)
    return repo


# ---------------------------------------------------------------------------
# 1. Misleading ed25519_sig_hex / pqc_sig_hex arguments removed
# ---------------------------------------------------------------------------
def test_generate_hybrid_grant_rejects_manual_sig_hex_arguments():
    gate = PQCEnclaveGate()
    digest = gate.compute_payload_digest("payload")
    with pytest.raises(TypeError, match="ed25519_sig_hex"):
        gate.generate_hybrid_grant(  # type: ignore[call-arg]
            sequence_index=1,
            prev_block_hash="0" * 64,
            event_type="PROBE",
            payload_sha256=digest,
            operation_schema={"probe": 1},
            ed25519_sig_hex="00" * 32,
        )
    with pytest.raises(TypeError, match="pqc_sig_hex"):
        gate.generate_hybrid_grant(  # type: ignore[call-arg]
            sequence_index=1,
            prev_block_hash="0" * 64,
            event_type="PROBE",
            payload_sha256=digest,
            operation_schema={"probe": 1},
            pqc_sig_hex="00" * 32,
        )


# ---------------------------------------------------------------------------
# 2. Readable operator approval -> local signing -> trusted verification
# ---------------------------------------------------------------------------
def test_readable_interactive_approval_workflow_succeeds_end_to_end(disposable_repo: Path, tmp_path: Path):
    enclave_dir = tmp_path / "enclave"
    runs_dir = tmp_path / "runs"
    receipts_path = tmp_path / "receipts.jsonl"
    shown_lines: list[str] = []
    prompts_seen: list[str] = []

    accept_cmd = f"{sys.executable} -m pytest -q -p no:cacheprovider test_calc.py"
    summary = run_task(
        "fix add in calc.py",
        root=str(disposable_repo),
        accept=accept_cmd,
        approve_writes=False,  # real interactive confirm workflow, not flag bypass
        confirm=lambda prompt: (prompts_seen.append(prompt) or True),
        out=shown_lines.append,
        runs_dir=str(runs_dir),
        enclave_dir=str(enclave_dir),
        receipts=str(receipts_path),
        chat=_scripted(_call("read_file", path="calc.py"), _call("write_file", path="calc.py", content=FIXED)),
    )

    assert summary["disposition"] == "VERIFIED", summary
    assert summary["state"] == "SUCCEEDED"
    assert summary["write_approval"] == "interactive"
    assert summary["grant_id"] and summary["grant_id"].startswith("grant-")
    assert summary["key_fingerprint"] and summary["key_fingerprint"].startswith("sha256:")
    assert (disposable_repo / "calc.py").read_text(encoding="utf-8") == FIXED

    # Verify readable action & scope card was displayed and never asked for signature hex
    rendered = "\n".join(shown_lines)
    assert "OSIRIS LOCAL OPERATOR GRANT REQUEST" in rendered
    assert "Action:      code.apply_writes" in rendered
    assert "calc.py" in rendered
    assert "ed25519_sig_hex" not in rendered
    assert len(prompts_seen) == 1
    assert "ed25519_sig_hex" not in prompts_seen[0]

    # Verify pure verifier (zero private keys) independently validates the recorded grant
    store = RunStore(str(runs_dir))
    saved_run = store.load(summary["run_id"])
    verifier = PQCEnclaveGate.verifier_from_enclave_dir(str(enclave_dir))
    assert verifier.verifier_only is True
    ok, reason = verifier.verify_grant_integrity(saved_run["approval_grant"], consume=False)
    assert (ok, reason) == (True, "VERIFIED_HYBRID_PQC_OK")
    assert Receipts.verify(str(receipts_path))


# ---------------------------------------------------------------------------
# 3. Missing / declined operator approval fails closed
# ---------------------------------------------------------------------------
def test_missing_or_declined_approval_mints_no_grant_and_writes_nothing(disposable_repo: Path, tmp_path: Path):
    enclave_dir = tmp_path / "enclave"
    runs_dir = tmp_path / "runs"
    receipts_path = tmp_path / "receipts.jsonl"
    accept_cmd = f"{sys.executable} -m pytest -q -p no:cacheprovider test_calc.py"

    summary = run_task(
        "fix add",
        root=str(disposable_repo),
        accept=accept_cmd,
        approve_writes=False,
        confirm=lambda _prompt: False,
        out=lambda _s: None,
        runs_dir=str(runs_dir),
        enclave_dir=str(enclave_dir),
        receipts=str(receipts_path),
        chat=_scripted(_call("write_file", path="calc.py", content=FIXED)),
    )

    assert summary["disposition"] == "FAILED"
    assert summary["grant_id"] is None
    assert (disposable_repo / "calc.py").read_text(encoding="utf-8") == BUGGY
    saved_run = RunStore(str(runs_dir)).load(summary["run_id"])
    assert "approval_grant" not in saved_run

    # Also verify direct apply_idempotent without an approval_grant raises GrantVerificationError
    saved_run["state"] = "AWAITING_APPROVAL"
    with pytest.raises(GrantVerificationError, match="missing signed operator approval grant"):
        apply_idempotent(saved_run, RunStore(str(runs_dir)), enclave_dir=str(enclave_dir))


# ---------------------------------------------------------------------------
# 4. Changed payload, schema, or scope after signing fails closed
# ---------------------------------------------------------------------------
def test_changed_payload_or_scope_or_schema_fails_closed(disposable_repo: Path, tmp_path: Path):
    enclave_dir = tmp_path / "enclave"
    runs_dir = tmp_path / "runs"
    store = RunStore(str(runs_dir))
    accept_cmd = f"{sys.executable} -m pytest -q -p no:cacheprovider test_calc.py"

    # Pause at AWAITING_APPROVAL, mint a valid grant, then mutate the proposal before apply
    paused = run_task(
        "fix add",
        root=str(disposable_repo),
        accept=accept_cmd,
        stop_after="AWAITING_APPROVAL",
        runs_dir=str(runs_dir),
        enclave_dir=str(enclave_dir),
        receipts=str(tmp_path / "receipts.jsonl"),
        out=lambda _s: None,
        chat=_scripted(_call("write_file", path="calc.py", content=FIXED)),
    )
    run_rec = store.load(paused["run_id"])
    signer = PQCEnclaveGate.load_or_init_local(str(enclave_dir))
    payload_sha, scope, schema, summary_txt = proposals_grant_spec(run_rec["root"], run_rec["proposals"])
    grant = signer.request_approved_grant(
        action="code.apply_writes",
        scope=scope,
        summary=summary_txt,
        payload_sha256=payload_sha,
        operation_schema=schema,
        confirm=lambda _p: True,
    )

    # (a) Tamper staged content / after_sha256 after grant was signed -> resume must BLOCK
    tampered_run = copy.deepcopy(run_rec)
    tampered_run["approval_grant"] = copy.deepcopy(grant)
    tampered_run["proposals"][0]["content"] = "def add(a, b):\n    return 999\n"
    tampered_run["proposals"][0]["after_sha256"] = signer.compute_payload_digest(
        tampered_run["proposals"][0]["content"]
    )
    store.save(tampered_run)

    res = resume(paused["run_id"], store=store, enclave_dir=str(enclave_dir), out=lambda _s: None)
    assert res["state"] == "BLOCKED"
    assert any("payload_sha256 mismatch" in r for r in res["verification"]["reasons"])
    assert (disposable_repo / "calc.py").read_text(encoding="utf-8") == BUGGY

    # (b) Tamper operation_schema or scope directly -> verifier rejects
    verifier = PQCEnclaveGate.verifier_from_enclave_dir(str(enclave_dir))
    bad_schema_grant = copy.deepcopy(grant)
    bad_schema_grant["operation_schema"]["extra"] = "injected"
    ok_schema, msg_schema = verifier.verify_grant_integrity(bad_schema_grant)
    assert ok_schema is False and "Operation schema SHA-256 mismatch" in msg_schema

    ok_scope, msg_scope = verifier.verify_grant_integrity(
        grant,
        expected_scope=["code.apply_writes", str(disposable_repo), "other_file.py"],
    )
    assert ok_scope is False and "Grant scope mismatch" in msg_scope


# ---------------------------------------------------------------------------
# 5. Wrong / untrusted signer fails even when embedding its own public keys
# ---------------------------------------------------------------------------
def test_wrong_signer_rejected_even_with_self_consistent_envelope_keys(disposable_repo: Path, tmp_path: Path):
    enclave_dir = tmp_path / "enclave"
    runs_dir = tmp_path / "runs"
    store = RunStore(str(runs_dir))
    accept_cmd = f"{sys.executable} -m pytest -q -p no:cacheprovider test_calc.py"

    # Initialize legitimate trusted operator key store
    legit_signer = PQCEnclaveGate.load_or_init_local(str(enclave_dir))
    verifier = PQCEnclaveGate.verifier_from_enclave_dir(str(enclave_dir))

    # Rogue signer with the SAME operator_id signs a self-consistent grant carrying its own public keys
    rogue_signer = PQCEnclaveGate(operator_id=legit_signer.operator_id)
    paused = run_task(
        "fix add",
        root=str(disposable_repo),
        accept=accept_cmd,
        stop_after="AWAITING_APPROVAL",
        runs_dir=str(runs_dir),
        enclave_dir=str(enclave_dir),
        receipts=str(tmp_path / "receipts.jsonl"),
        out=lambda _s: None,
        chat=_scripted(_call("write_file", path="calc.py", content=FIXED)),
    )
    run_rec = store.load(paused["run_id"])
    payload_sha, scope, schema, _ = proposals_grant_spec(run_rec["root"], run_rec["proposals"])
    rogue_grant = rogue_signer.generate_hybrid_grant(
        sequence_index=1,
        prev_block_hash="0" * 64,
        event_type="code.apply_writes",
        payload_sha256=payload_sha,
        operation_schema=schema,
        scope=scope,
    )

    # Direct verifier check against trusted_operator_keys.json MUST reject rogue grant
    ok, reason = verifier.verify_grant_integrity(rogue_grant)
    assert ok is False
    assert "untrusted signer key" in reason

    # Injecting rogue_grant into the run record MUST block apply_idempotent with zero file writes
    run_rec["approval_grant"] = rogue_grant
    store.save(run_rec)
    res = resume(paused["run_id"], store=store, enclave_dir=str(enclave_dir), out=lambda _s: None)
    assert res["state"] == "BLOCKED"
    assert "untrusted signer key" in res["verification"]["reasons"][0]
    assert (disposable_repo / "calc.py").read_text(encoding="utf-8") == BUGGY


# ---------------------------------------------------------------------------
# 6. Malformed signatures, expiry window, and replay protection
# ---------------------------------------------------------------------------
def test_malformed_signature_expiry_and_replay_protection(tmp_path: Path):
    enclave_dir = tmp_path / "enclave"
    signer = PQCEnclaveGate.load_or_init_local(str(enclave_dir))
    verifier = PQCEnclaveGate.verifier_from_enclave_dir(str(enclave_dir))
    digest = signer.compute_payload_digest("important-payload")

    grant = signer.generate_hybrid_grant(
        sequence_index=1,
        prev_block_hash="0" * 64,
        event_type="code.apply_writes",
        payload_sha256=digest,
        operation_schema={"action": "code.apply_writes"},
        scope=["code.apply_writes", "/tmp/ws"],
        ttl_seconds=60.0,
        issued_at=1000.0,
    )

    # Malformed non-hex signature fails closed
    bad_hex = copy.deepcopy(grant)
    bad_hex["pqc_envelope"]["ed25519_signature"] = "not-valid-hex!!"
    ok_mal, msg_mal = verifier.verify_grant_integrity(bad_hex, now=1010.0)
    assert ok_mal is False and "signature verification failed" in msg_mal

    # Expired grant fails closed
    ok_exp, msg_exp = verifier.verify_grant_integrity(grant, now=1061.0)
    assert ok_exp is False and "expired" in msg_exp.lower()

    # Future-dated grant beyond clock skew fails closed
    ok_fut, msg_fut = verifier.verify_grant_integrity(grant, now=900.0)
    assert ok_fut is False and "not yet valid" in msg_fut.lower()

    # First consumption within validity window succeeds
    ok_first, msg_first = verifier.verify_grant_integrity(
        grant, now=1020.0, consume=True, consumer_id="run-alpha"
    )
    assert (ok_first, msg_first) == (True, "VERIFIED_HYBRID_PQC_OK")

    # Replaying the consumed grant under a new consumer (even in a fresh verifier process!) fails
    fresh_verifier = PQCEnclaveGate.verifier_from_enclave_dir(str(enclave_dir))
    ok_replay, msg_replay = fresh_verifier.verify_grant_integrity(
        grant, now=1025.0, consume=True, consumer_id="run-beta"
    )
    assert ok_replay is False and "replay detected" in msg_replay.lower()


# ---------------------------------------------------------------------------
# 7. Private key locality, permissions, non-overwrite, and non-leakage
# ---------------------------------------------------------------------------
def test_private_keys_stay_local_keep_0600_never_overwrite_and_never_leak(
    disposable_repo: Path, tmp_path: Path
):
    enclave_dir = tmp_path / "enclave"
    runs_dir = tmp_path / "runs"
    receipts_path = tmp_path / "receipts.jsonl"
    logs_out: list[str] = []

    g1 = PQCEnclaveGate.load_or_init_local(str(enclave_dir))
    ed_key_file = enclave_dir / "operator_ed25519.key"
    ml_key_file = enclave_dir / "operator_mldsa65.key"
    trusted_file = enclave_dir / "trusted_operator_keys.json"

    assert ed_key_file.is_file() and ml_key_file.is_file() and trusted_file.is_file()
    if os.name == "posix":
        assert stat.S_IMODE(ed_key_file.stat().st_mode) == 0o600
        assert stat.S_IMODE(ml_key_file.stat().st_mode) == 0o600

    ed_bytes_before = ed_key_file.read_bytes()
    ml_bytes_before = ml_key_file.read_bytes()

    # Reloading must preserve exact existing keys without overwriting
    g2 = PQCEnclaveGate.load_or_init_local(str(enclave_dir))
    assert g2.key_fingerprint == g1.key_fingerprint
    assert ed_key_file.read_bytes() == ed_bytes_before
    assert ml_key_file.read_bytes() == ml_bytes_before

    # Run full task and ensure private seed hex never appears in repr, output, run JSON, or receipts
    accept_cmd = f"{sys.executable} -m pytest -q -p no:cacheprovider test_calc.py"
    summary = run_task(
        "fix add",
        root=str(disposable_repo),
        accept=accept_cmd,
        approve_writes=True,
        out=logs_out.append,
        runs_dir=str(runs_dir),
        enclave_dir=str(enclave_dir),
        receipts=str(receipts_path),
        chat=_scripted(_call("write_file", path="calc.py", content=FIXED)),
    )
    assert summary["disposition"] == "VERIFIED"

    ed_secret_hex = ed_bytes_before.hex()
    ml_secret_hex = ml_bytes_before.hex()
    run_json_text = (runs_dir / f"{summary['run_id']}.json").read_text(encoding="utf-8")
    receipts_text = receipts_path.read_text(encoding="utf-8")
    console_text = "\n".join(logs_out) + "\n" + repr(g1)

    for blob_name, blob in (
        ("repr/console", console_text),
        ("run_record", run_json_text),
        ("receipts", receipts_text),
        ("trusted_operator_keys.json", trusted_file.read_text(encoding="utf-8")),
    ):
        assert ed_secret_hex not in blob, f"Ed25519 private seed leaked into {blob_name}"
        assert ml_secret_hex not in blob, f"ML-DSA-65 private seed leaked into {blob_name}"

    # Partial key state (one key missing) must fail closed instead of overwriting the remaining key
    partial_dir = tmp_path / "partial_enclave"
    partial_dir.mkdir()
    (partial_dir / "operator_ed25519.key").write_bytes(ed_bytes_before)
    os.chmod(partial_dir / "operator_ed25519.key", 0o600)
    with pytest.raises(RuntimeError, match="Incomplete local operator keypair"):
        PQCEnclaveGate.load_or_init_local(str(partial_dir))


# ---------------------------------------------------------------------------
# 8. CodeAgent._run_command and apply_all_pending_writes use signed local grants
# ---------------------------------------------------------------------------
def test_code_agent_command_and_batch_write_mint_and_verify_hybrid_grants(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "hello.py").write_text("X = 1\n", encoding="utf-8")

    enclave = PQCEnclaveGate.load_or_init_local(str(tmp_path / "agent_enclave"))
    prompts: list[str] = []
    outputs: list[str] = []

    agent = CodeAgent(
        root=str(ws),
        chat=_scripted(
            _call("write_file", path="hello.py", content="X = 2\n"),
            _call("run_command", command=f"{sys.executable} -V"),
        ),
        confirm=lambda p: (prompts.append(p) or True),
        propose_write=lambda _p, _c: True,
        has_pending_write=lambda: False,
        out=outputs.append,
        enclave=enclave,
    )
    res = agent.run("update hello and run check")
    assert res.status == "SUCCESS"

    cmd_decisions = [d for d in res.decisions if d["tool"] == "run_command"]
    assert len(cmd_decisions) == 1
    assert cmd_decisions[0]["decision"] == "GRANTED"
    assert cmd_decisions[0]["grant_id"].startswith("grant-")
    assert cmd_decisions[0]["key_fingerprint"] == enclave.key_fingerprint

    applied = agent.apply_all_pending_writes(lambda _props: True)
    assert len(applied) == 1
    assert applied[0]["decision"] == "GRANTED"
    assert applied[0]["grant_id"].startswith("grant-")
    assert applied[0]["key_fingerprint"] == enclave.key_fingerprint
    assert (ws / "hello.py").read_text(encoding="utf-8") == "X = 2\n"


# ---------------------------------------------------------------------------
# 9. Hardening regression tests (F-01 through F-13)
# ---------------------------------------------------------------------------
def test_f01_apply_idempotent_rejects_content_mismatch_against_after_sha256(
    disposable_repo: Path, tmp_path: Path
):
    enclave_dir = tmp_path / "f01_enclave"
    signer = PQCEnclaveGate.load_or_init_local(str(enclave_dir), allow_init=True)
    store = RunStore(str(tmp_path / "f01_runs"))

    old_sha = signer.compute_payload_digest(BUGGY)
    good_after = signer.compute_payload_digest(FIXED)
    props = [
        {
            "rel_path": "calc.py",
            "existed_at_proposal": True,
            "before_sha256": old_sha,
            "after_sha256": good_after,
            "content": FIXED,
            "diff": "",
        }
    ]
    psha, pscope, pschema, psumm = proposals_grant_spec(str(disposable_repo), props)
    grant = signer.request_approved_grant(
        action="code.apply_writes",
        scope=pscope,
        summary=psumm,
        payload_sha256=psha,
        operation_schema=pschema,
        confirm=lambda _p: True,
    )
    tampered_props = copy.deepcopy(props)
    tampered_props[0]["content"] = "def add(a, b):\n    return -999\n"
    run = store.create(
        "code-f01",
        root=str(disposable_repo),
        proposals=tampered_props,
        approval_grant=grant,
        options={},
    )
    with pytest.raises(GrantVerificationError, match="content SHA-256 mismatch"):
        apply_idempotent(run, store, enclave_dir=str(enclave_dir))
    assert (disposable_repo / "calc.py").read_text(encoding="utf-8") == BUGGY


def test_f02_apply_idempotent_ignores_run_options_trust_root_redirection(
    disposable_repo: Path, tmp_path: Path
):
    legit_dir = tmp_path / "f02_legit"
    rogue_dir = tmp_path / "f02_rogue"
    PQCEnclaveGate.load_or_init_local(str(legit_dir), allow_init=True)
    rogue = PQCEnclaveGate.load_or_init_local(str(rogue_dir), allow_init=True)
    store = RunStore(str(tmp_path / "f02_runs"))

    props = [
        {
            "rel_path": "calc.py",
            "existed_at_proposal": True,
            "before_sha256": rogue.compute_payload_digest(BUGGY),
            "after_sha256": rogue.compute_payload_digest(FIXED),
            "content": FIXED,
            "diff": "",
        }
    ]
    psha, pscope, pschema, _ = proposals_grant_spec(str(disposable_repo), props)
    rogue_grant = rogue.generate_hybrid_grant(
        1, "0" * 64, "code.apply_writes", psha, pschema, scope=pscope
    )
    run = store.create(
        "code-f02",
        root=str(disposable_repo),
        proposals=props,
        approval_grant=rogue_grant,
        options={
            "enclave_dir": str(rogue_dir),
            "trusted_keys_path": str(rogue_dir / "trusted_operator_keys.json"),
        },
    )
    with pytest.raises(GrantVerificationError, match="untrusted signer key"):
        apply_idempotent(run, store, enclave_dir=str(legit_dir))
    assert (disposable_repo / "calc.py").read_text(encoding="utf-8") == BUGGY


def test_f03_apply_idempotent_blocks_path_traversal_and_symlink_escape(
    disposable_repo: Path, tmp_path: Path
):
    enclave_dir = tmp_path / "f03_enclave"
    signer = PQCEnclaveGate.load_or_init_local(str(enclave_dir), allow_init=True)
    store = RunStore(str(tmp_path / "f03_runs"))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("SECRET = 1\n", encoding="utf-8")

    # (a) Path traversal rejected
    trav_props = [
        {
            "rel_path": "../outside/secret.py",
            "existed_at_proposal": True,
            "before_sha256": signer.compute_payload_digest("SECRET = 1\n"),
            "after_sha256": signer.compute_payload_digest("PWN = 1\n"),
            "content": "PWN = 1\n",
            "diff": "",
        }
    ]
    with pytest.raises((GrantVerificationError, ValueError)):
        tsha, tscope, tschema, tsumm = proposals_grant_spec(str(disposable_repo), trav_props)
        tgrant = signer.request_approved_grant(
            action="code.apply_writes",
            scope=tscope,
            summary=tsumm,
            payload_sha256=tsha,
            operation_schema=tschema,
            confirm=lambda _p: True,
        )
        run_t = store.create(
            "code-f03-trav",
            root=str(disposable_repo),
            proposals=trav_props,
            approval_grant=tgrant,
            options={},
        )
        apply_idempotent(run_t, store, enclave_dir=str(enclave_dir))
    assert (outside / "secret.py").read_text(encoding="utf-8") == "SECRET = 1\n"

    # (b) Post-approval parent directory replaced by symlink pointing outside workspace
    sub = disposable_repo / "sub"
    sub.mkdir()
    (sub / "target.py").write_text("SAFE = 1\n", encoding="utf-8")
    (outside / "target.py").write_text("SAFE = 1\n", encoding="utf-8")
    sprops = [
        {
            "rel_path": "sub/target.py",
            "existed_at_proposal": True,
            "before_sha256": signer.compute_payload_digest("SAFE = 1\n"),
            "after_sha256": signer.compute_payload_digest("ESCAPED = 1\n"),
            "content": "ESCAPED = 1\n",
            "diff": "",
        }
    ]
    ssha, sscope, sschema, ssumm = proposals_grant_spec(str(disposable_repo), sprops)
    sgrant = signer.request_approved_grant(
        action="code.apply_writes",
        scope=sscope,
        summary=ssumm,
        payload_sha256=ssha,
        operation_schema=sschema,
        confirm=lambda _p: True,
    )
    (sub / "target.py").unlink()
    sub.rmdir()
    os.symlink(str(outside), str(sub))
    run_s = store.create(
        "code-f03-sym",
        root=str(disposable_repo),
        proposals=sprops,
        approval_grant=sgrant,
        options={},
    )
    with pytest.raises(GrantVerificationError, match="symlink|outside"):
        apply_idempotent(run_s, store, enclave_dir=str(enclave_dir))
    assert (outside / "target.py").read_text(encoding="utf-8") == "SAFE = 1\n"


def test_f04_f05_no_silent_enrollment_and_no_ephemeral_code_agent_enclave(
    tmp_path: Path, monkeypatch
):
    # Ordinary load_or_init_local() without allow_init=True must refuse to mint or enroll keys
    empty_dir = tmp_path / "uninit_enclave"
    with pytest.raises(RuntimeError):
        PQCEnclaveGate.load_or_init_local(str(empty_dir))
    assert not (empty_dir / "operator_ed25519.key").exists()
    assert not (empty_dir / "trusted_operator_keys.json").exists()

    # Existing keys without trusted_operator_keys.json must NOT silently re-enroll when allow_init=False
    keys_only = tmp_path / "keys_only_enclave"
    PQCEnclaveGate.load_or_init_local(str(keys_only), allow_init=True)
    (keys_only / "trusted_operator_keys.json").unlink()
    with pytest.raises(RuntimeError):
        PQCEnclaveGate.load_or_init_local(str(keys_only), allow_init=False)
    assert not (keys_only / "trusted_operator_keys.json").exists()

    # Existing trust store without private keys must refuse even allow_init=True
    trust_only = tmp_path / "trust_only_enclave"
    PQCEnclaveGate.load_or_init_local(str(trust_only), allow_init=True)
    (trust_only / "operator_ed25519.key").unlink()
    (trust_only / "operator_mldsa65.key").unlink()
    with pytest.raises(RuntimeError):
        PQCEnclaveGate.load_or_init_local(str(trust_only), allow_init=True)
    assert not (trust_only / "operator_ed25519.key").exists()

    # CodeAgent._get_enclave() must NOT mint an ephemeral self-trusting keypair when unconfigured
    from osiris_cli.code_agent import Denied

    ws = tmp_path / "ws_uninit"
    ws.mkdir()
    monkeypatch.setenv("OSIRIS_ENCLAVE_DIR", str(tmp_path / "missing_enclave"))
    agent = CodeAgent(root=str(ws), chat=_scripted(), out=lambda _s: None)
    with pytest.raises(Denied):
        agent._get_enclave()


def test_f06_publish_v600_requires_enrolled_operator_approval_before_publish(tmp_path: Path):
    import publish_v600_to_zenodo as pub

    assert hasattr(pub, "authorize_v600_release")
    z = tmp_path / "bundle.zip"
    t = tmp_path / "bundle.tar.gz"
    z.write_bytes(b"zip-bytes")
    t.write_bytes(b"tar-bytes")

    # Missing enclave fails closed
    with pytest.raises(RuntimeError):
        pub.authorize_v600_release(
            z, t, enclave_dir=str(tmp_path / "no_enclave"), confirm=lambda _p: True, out=lambda _s: None
        )

    enc = tmp_path / "pub_enclave"
    PQCEnclaveGate.load_or_init_local(str(enc), allow_init=True)

    # Declined operator approval fails closed
    with pytest.raises(ApprovalDeniedError):
        pub.authorize_v600_release(
            z, t, enclave_dir=str(enc), confirm=lambda _p: False, out=lambda _s: None
        )

    # Approved grant verifies and consumes once against pinned trusted keys
    grant = pub.authorize_v600_release(
        z, t, enclave_dir=str(enc), confirm=lambda _p: True, out=lambda _s: None
    )
    verifier = PQCEnclaveGate.verifier_from_enclave_dir(str(enc))
    assert verifier.verify_grant_integrity(grant, consume=False)[0] is True
    assert verifier.verify_grant_integrity(grant, consume=True)[0] is False  # already consumed


def test_f07_f11_algorithm_downgrade_nonfinite_coercions_and_envelope_integrity(tmp_path: Path):
    enc = tmp_path / "f07_enclave"
    signer = PQCEnclaveGate.load_or_init_local(str(enc), allow_init=True)
    verifier = PQCEnclaveGate.verifier_from_enclave_dir(str(enc))
    d = signer.compute_payload_digest("strict-payload")
    grant = signer.generate_hybrid_grant(
        sequence_index=7,
        prev_block_hash="b" * 64,
        event_type="code.apply_writes",
        payload_sha256=d,
        operation_schema={"action": "code.apply_writes"},
        scope=["code.apply_writes", "/tmp/ws"],
        ttl_seconds=60.0,
        issued_at=1000.0,
    )

    # Missing or downgraded algorithm fails closed
    for mut in (
        lambda g: g["pqc_envelope"].pop("algorithm"),
        lambda g: g["pqc_envelope"].__setitem__("algorithm", "Ed25519"),
        lambda g: g["pqc_envelope"].__setitem__("algorithm", "ML-DSA-65"),
        lambda g: g["pqc_envelope"].__setitem__("algorithm", "none"),
        lambda g: g["pqc_envelope"].__setitem__("key_fingerprint", "sha256:" + "f" * 64),
        lambda g: g["pqc_envelope"]["lattice_parameters"].__setitem__("modulus", 1),
        lambda g: g.__setitem__("sequence_index", "7"),
        lambda g: g.__setitem__("sequence_index", 7.9),
        lambda g: g.__setitem__("issued_at", "1000.0"),
        lambda g: g.__setitem__("expires_at", "1060.0"),
        lambda g: g.__setitem__("timestamp", 9999999.0),
        lambda g: g.__setitem__("scope", [" code.apply_writes ", "/tmp/ws"]),
    ):
        bad = copy.deepcopy(grant)
        mut(bad)
        assert verifier.verify_grant_integrity(bad, now=1010.0)[0] is False

    # Non-finite now fails closed
    assert verifier.verify_grant_integrity(grant, now=float("nan"))[0] is False
    assert verifier.verify_grant_integrity(grant, now=float("inf"))[0] is False

    # Boolean / non-finite ttl_seconds rejected at generation
    for bad_ttl in (True, False, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            signer.generate_hybrid_grant(1, "0" * 64, "E", d, {}, ttl_seconds=bad_ttl)  # type: ignore[arg-type]

    # Trusted key metadata operator_id mismatch rejected
    tk = json.loads((enc / "trusted_operator_keys.json").read_text(encoding="utf-8"))
    tk["trusted_keys"][0]["operator_id"] = "Other Enrolled Operator"
    alt = tmp_path / "alt_trusted.json"
    alt.write_text(json.dumps(tk), encoding="utf-8")
    alt_v = PQCEnclaveGate.verifier_from_enclave_dir(str(enc), trusted_keys_path=str(alt))
    assert alt_v.verify_grant_integrity(grant, now=1010.0)[0] is False


def test_f08_f09_code_agent_post_approval_stale_target_and_script_toctou(tmp_path: Path):
    from osiris_cli.code_agent import Denied

    enc = tmp_path / "f08_enclave"
    signer = PQCEnclaveGate.load_or_init_local(str(enc), allow_init=True)

    # (a) apply_pending_write rejects target edited DURING approval callback
    ws1 = tmp_path / "ws1"
    ws1.mkdir()
    (ws1 / "f.py").write_text("V = 1\n", encoding="utf-8")
    ca1 = CodeAgent(
        root=str(ws1),
        chat=_scripted(),
        propose_write=lambda _p, _c: True,
        has_pending_write=lambda: False,
        out=lambda _s: None,
        enclave=signer,
    )
    ca1._write_file("f.py", "V = 2\n")
    with pytest.raises(Denied, match="changed on disk"):
        ca1.apply_pending_write(
            approve=lambda _prop: ((ws1 / "f.py").write_text("CONCURRENT = 99\n", encoding="utf-8") or True)
        )
    assert (ws1 / "f.py").read_text(encoding="utf-8") == "CONCURRENT = 99\n"

    # (b) apply_all_pending_writes rejects target edited DURING approval callback
    ws2 = tmp_path / "ws2"
    ws2.mkdir()
    (ws2 / "f.py").write_text("V = 1\n", encoding="utf-8")
    ca2 = CodeAgent(
        root=str(ws2),
        chat=_scripted(),
        propose_write=lambda _p, _c: True,
        has_pending_write=lambda: False,
        out=lambda _s: None,
        enclave=signer,
        max_writes=4,
    )
    ca2._write_file("f.py", "V = 2\n")
    with pytest.raises(Denied, match="changed on disk"):
        ca2.apply_all_pending_writes(
            approve=lambda _props: ((ws2 / "f.py").write_text("CONCURRENT_BATCH = 99\n", encoding="utf-8") or True)
        )
    assert (ws2 / "f.py").read_text(encoding="utf-8") == "CONCURRENT_BATCH = 99\n"

    # (c) _run_command rejects workspace script mutated between approval and launch
    ws3 = tmp_path / "ws3"
    ws3.mkdir()
    script = ws3 / "run.py"
    script.write_text("print('BENIGN')\n", encoding="utf-8")
    ca3 = CodeAgent(
        root=str(ws3),
        chat=_scripted(),
        confirm=lambda _p: (script.write_text("print('SWAPPED')\n", encoding="utf-8") or True),
        out=lambda _s: None,
        enclave=signer,
    )
    with pytest.raises(Denied, match="execution grant verification failed"):
        ca3._run_command(f"{sys.executable} run.py")


def test_f10_consumed_ledger_verifies_hash_chain_and_handles_corruption_and_race(tmp_path: Path):
    enc = tmp_path / "f10_enclave"
    signer = PQCEnclaveGate.load_or_init_local(str(enc), allow_init=True)
    verifier = PQCEnclaveGate.verifier_from_enclave_dir(str(enc))
    d = signer.compute_payload_digest("ledger-payload")
    g1 = signer.generate_hybrid_grant(1, "0" * 64, "E", d, {"i": 1}, issued_at=1000.0)
    g2 = signer.generate_hybrid_grant(2, "0" * 64, "E", d, {"i": 2}, issued_at=1000.0)
    g3 = signer.generate_hybrid_grant(3, "0" * 64, "E", d, {"i": 3}, issued_at=1000.0)
    assert verifier.verify_grant_integrity(g1, now=1010.0, consume=True, consumer_id="r1")[0] is True
    assert verifier.verify_grant_integrity(g2, now=1010.0, consume=True, consumer_id="r2")[0] is True

    ledger = enc / "consumed_grants.jsonl"
    orig_lines = ledger.read_text(encoding="utf-8").splitlines()

    # Reordered rows must fail closed
    ledger.write_text("\n".join(reversed(orig_lines)) + "\n", encoding="utf-8")
    assert (
        PQCEnclaveGate.verifier_from_enclave_dir(str(enc)).verify_grant_integrity(
            g3, now=1010.0, consume=True, consumer_id="r3"
        )[0]
        is False
    )

    # Tampered earlier row erasing g1 must fail closed (preventing replay of g1)
    rows = [json.loads(line) for line in orig_lines]
    rows[0]["grant_id"] = "grant-erased"
    rows[0]["nonce"] = "nonce-erased"
    ledger.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert (
        PQCEnclaveGate.verifier_from_enclave_dir(str(enc)).verify_grant_integrity(
            g1, now=1010.0, consume=True, consumer_id="r-attacker"
        )[0]
        is False
    )

    # Non-dict JSON row must fail closed without uncaught AttributeError
    ledger.write_text("[]\n" + orig_lines[-1] + "\n", encoding="utf-8")
    ok_nd, why_nd = PQCEnclaveGate.verifier_from_enclave_dir(str(enc)).verify_grant_integrity(
        g3, now=1010.0, consume=True
    )
    assert ok_nd is False and "corrupted" in why_nd.lower()

    # Multi-process race: 6 independent processes competing for one grant -> exactly 1 succeeds
    race_enc = tmp_path / "f10_race"
    rsig = PQCEnclaveGate.load_or_init_local(str(race_enc), allow_init=True)
    rgrant = rsig.generate_hybrid_grant(1, "0" * 64, "E", d, {"race": 1}, ttl_seconds=60.0)
    gfile = tmp_path / "race_grant.json"
    gfile.write_text(json.dumps(rgrant), encoding="utf-8")
    worker = (
        "import json, sys, time\n"
        "from osiris_cli.pqc_enclave import PQCEnclaveGate\n"
        "v = PQCEnclaveGate.verifier_from_enclave_dir(sys.argv[1])\n"
        "g = json.load(open(sys.argv[2]))\n"
        "t0 = float(sys.argv[4])\n"
        "while time.time() < t0:\n"
        "    time.sleep(0.002)\n"
        "ok, why = v.verify_grant_integrity(g, consume=True, consumer_id=sys.argv[3])\n"
        "print(json.dumps({'ok': ok, 'why': why}))\n"
    )
    import time as _time

    t0 = str(_time.time() + 0.35)
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", worker, str(race_enc), str(gfile), f"p-{i}", t0],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for i in range(6)
    ]
    outs = [json.loads(p.communicate(timeout=20)[0].strip()) for p in procs]
    assert sum(1 for o in outs if o["ok"]) == 1
    assert len([ln for ln in (race_enc / "consumed_grants.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]) == 1


def test_f13_atomic_write_cleanup_and_crash_resume_no_duplicate_receipt(tmp_path: Path):
    from osiris_cli.code_agent import append_receipt, atomic_write

    # (a) atomic_write cleans up orphan temp file when target is a directory
    d_target = tmp_path / "is_a_dir"
    d_target.mkdir()
    with pytest.raises(OSError):
        atomic_write(str(d_target), b"hello")
    assert not [p for p in tmp_path.iterdir() if ".osiris-tmp-" in p.name]

    # (b) Crash after receipt append before store.save(run) must not duplicate receipt on resume
    enc = tmp_path / "f13_enclave"
    signer = PQCEnclaveGate.load_or_init_local(str(enc), allow_init=True)
    repo = tmp_path / "f13_repo"
    repo.mkdir()
    (repo / "f1.py").write_text("F1 = 1\n", encoding="utf-8")
    (repo / "f2.py").write_text("F2 = 1\n", encoding="utf-8")
    rcpt = tmp_path / "f13_receipts.jsonl"
    store = RunStore(str(tmp_path / "f13_runs"))
    f1_after = signer.compute_payload_digest("F1 = 2\n")
    props = [
        {
            "rel_path": "f1.py",
            "existed_at_proposal": True,
            "before_sha256": signer.compute_payload_digest("F1 = 1\n"),
            "after_sha256": f1_after,
            "content": "F1 = 2\n",
            "diff": "",
        },
        {
            "rel_path": "f2.py",
            "existed_at_proposal": True,
            "before_sha256": signer.compute_payload_digest("F2 = 1\n"),
            "after_sha256": signer.compute_payload_digest("F2 = 2\n"),
            "content": "F2 = 2\n",
            "diff": "",
        },
    ]
    csha, cscope, cschema, csumm = proposals_grant_spec(str(repo), props)
    cgrant = signer.request_approved_grant(
        action="code.apply_writes",
        scope=cscope,
        summary=csumm,
        payload_sha256=csha,
        operation_schema=cschema,
        confirm=lambda _p: True,
    )
    # Simulate crash after writing f1.py and appending its receipt, before store.save(run)
    (repo / "f1.py").write_text("F1 = 2\n", encoding="utf-8")
    append_receipt(
        str(rcpt),
        "code.agent.apply_write",
        {
            "tool": "apply_write",
            "decision": "GRANTED",
            "run_id": "code-f13",
            "path": "f1.py",
            "before_sha256": props[0]["before_sha256"],
            "after_sha256": f1_after,
            "grant_id": cgrant["grant_id"],
            "key_fingerprint": cgrant["key_fingerprint"],
            "already_applied": False,
        },
    )
    (repo / "accept_ok.py").write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
    run = store.create(
        "code-f13",
        root=str(repo),
        proposals=props,
        approval_grant=cgrant,
        receipts_path=str(rcpt),
        options={"require_change": True, "accept_timeout": 60.0},
        accept=f"{sys.executable} accept_ok.py",
    )
    run["state"] = "AWAITING_APPROVAL"
    store.save(run)
    res = resume("code-f13", store=store, enclave_dir=str(enc), out=lambda _s: None)
    assert res["state"] == "SUCCEEDED"
    assert Receipts.verify(str(rcpt)) is True
    f1_rows = [
        json.loads(ln)
        for ln in rcpt.read_text(encoding="utf-8").splitlines()
        if ln.strip()
        and json.loads(ln)["kind"] == "code.agent.apply_write"
        and json.loads(ln)["payload"]["path"] == "f1.py"
    ]
    assert len(f1_rows) == 1


def test_f15_session_link_checked_before_consumption_and_covers_scope_and_symlinks(
    disposable_repo: Path, tmp_path: Path
):
    from osiris_cli import session_approval as sa

    enc = tmp_path / "f15_enclave"
    signer = PQCEnclaveGate.load_or_init_local(str(enc), allow_init=True)
    scope = sa.SessionScope.build(str(disposable_repo), ["calc.py", "conf/*"], [])

    # Case-insensitive and *.env exclusions + symlink exclusion
    assert not sa.covers_paths(scope, str(disposable_repo), ["conf/.ENV"])[0]
    assert not sa.covers_paths(scope, str(disposable_repo), ["conf/prod.env"])[0]
    (disposable_repo / "conf").mkdir()
    outside_f = tmp_path / "ext.py"
    outside_f.write_text("X = 1\n", encoding="utf-8")
    os.symlink(str(outside_f), str(disposable_repo / "conf" / "link.py"))
    assert not sa.covers_paths(scope, str(disposable_repo), ["conf/link.py"])[0]

    # Child link checked BEFORE consumption in apply_idempotent (missing parent does not spend grant)
    sg = sa.request_session_grant(signer, scope, confirm=lambda _p: True)
    props = [
        {
            "rel_path": "calc.py",
            "existed_at_proposal": True,
            "before_sha256": signer.compute_payload_digest(BUGGY),
            "after_sha256": signer.compute_payload_digest(FIXED),
            "content": FIXED,
            "diff": "",
        }
    ]
    psha, pscope, pschema, _ = proposals_grant_spec(str(disposable_repo), props)
    child = sa.issue_child_grant(
        signer,
        sg,
        action="code.apply_writes",
        scope=pscope,
        payload_sha256=psha,
        operation_schema=pschema,
        covered=sa.covers_paths(scope, str(disposable_repo), ["calc.py"]),
    )
    store = RunStore(str(tmp_path / "f15_runs"))
    run = store.create(
        "code-f15",
        root=str(disposable_repo),
        proposals=props,
        approval_grant=child,
        session_grant=None,  # missing parent session grant
        options={},
    )
    with pytest.raises(GrantVerificationError, match="parent approval is missing"):
        apply_idempotent(run, store, enclave_dir=str(enc))
    # Confirm child grant was NOT consumed by the failed link check
    v = PQCEnclaveGate.verifier_from_enclave_dir(str(enc))
    assert v.verify_grant_integrity(child, consume=False, check_replay=True)[0] is True
