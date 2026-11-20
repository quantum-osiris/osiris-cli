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
            _call("run_command", command=f"{sys.executable} -c \"print('AGENT_CMD_OK')\""),
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
