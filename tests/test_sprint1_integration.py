"""Sprint 1: sync transports, historical ledgers, trusted imports, isolation, simulated evidence."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from osiris.runtime.contracts import RuntimeCapability
from osiris.runtime.isolation import IS_SANDBOX, RlimitSubprocessBackend
from osiris.runtime.worker import IsolationUnavailable, WorkerRequest, WorkerRunner
from osiris_cli import key_sync
from osiris_cli.orchestration import ast_firewall
from osiris_cli.osiris_repl import execute_mutate, execute_tetra1d
from osiris_cli.paths import trusted_module_origin
from osiris_cli.key_sync import accept_remote_evidence as accept_pulled
from osiris_cli.substrate_guard import (
    SubstrateGuardError,
    _entry_hash,
    apply_return_chain,
    classify_zenodo_evidence,
    execution_evidence_class,
    load_chain,
)
from osiris_auto_discovery import ExperimentResult
from osiris_zenodo_publisher import PublishingWorkflow


def _living_row(prev: str, kind: str) -> dict:
    entry = {"kind": kind, "prev": prev}
    entry["hash"] = _entry_hash(entry)
    return entry


def test_living_exchange_chain_without_seq_can_extend(tmp_path: Path) -> None:
    first = _living_row("0" * 64, "a")
    second = _living_row(first["hash"], "b")
    third = _living_row(second["hash"], "d")
    local = tmp_path / "exchanges.jsonl"
    local.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in (first, second)), encoding="utf-8")
    assert apply_return_chain(local, [first, second]) == "identical"
    assert apply_return_chain(local, [first, second, third]) == "extend"
    assert [row["kind"] for row in load_chain(local)] == ["a", "b", "d"]
    assert "seq" not in load_chain(local)[0]
    with pytest.raises(SubstrateGuardError, match="older chain"):
        apply_return_chain(local, [first])


def test_planted_home_module_is_not_executed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    marker = tmp_path / "executed"
    (home / "osiris_tetra1d_benchmark.py").write_text(
        f"open({str(marker)!r}, 'w').write('ran')\n"
        "def run_tetra1d_benchmark(**kwargs):\n"
        "    return kwargs\n",
        encoding="utf-8",
    )
    (home / "osiris_autonomous_mutation.py").write_text(
        f"open({str(marker)!r}, 'a').write('mut')\n"
        "def run_autonomous_mutation_loop(**kwargs):\n"
        "    return kwargs\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    sys.path.insert(0, str(home))
    try:
        assert trusted_module_origin("osiris_tetra1d_benchmark") is None
        assert execute_tetra1d()["error"] == "untrusted_artifact_refused"
        assert execute_mutate()["error"] == "untrusted_artifact_refused"
        assert not marker.exists()
    finally:
        sys.path.remove(str(home))


def test_discovery_qvm_result_is_quarantined_from_zenodo() -> None:
    result = ExperimentResult(
        name="local-qvm",
        config={"backend": "ibm_fez"},
        backend="ibm_fez",
        job_ids=["qvm_abc123", "fallback_deadbeef"],
        hypothesis="unused",
        falsifiable=True,
        passes_significance=True,
        p_value=0.01,
        effect_size=1.0,
        confidence_interval=(0.1, 0.2),
    )
    assert execution_evidence_class(result.job_ids) == "SIMULATED"
    verdict = classify_zenodo_evidence(
        {"backend": result.backend, "job_ids": result.job_ids, "evidence_class": "SIMULATED"}
    )
    assert verdict["admitted"] is False
    assert "simulated" in verdict["reason"]
    published = PublishingWorkflow("not-a-token").publish_result(result, "report")
    assert published["status"] == "QUARANTINED"
    assert ast_firewall.IS_SANDBOX is False


def _completed(argv, stdout: str = ""):
    return type("Result", (), {"returncode": 0, "stdout": stdout, "stderr": ""})()


def test_adb_and_ssh_use_the_helper_and_do_not_overwrite_a_newer_chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(key_sync, "find_adb_binary", lambda: "adb")
    monkeypatch.setattr(key_sync, "get_public_key_bytes", lambda key_dir=None: b"k" * 32)
    monkeypatch.setattr(key_sync, "enroll_edge_key", lambda **kwargs: {"status": "ENROLLED"})
    first = _living_row("0" * 64, "local")
    newer = _living_row(first["hash"], "newer")
    older_only = json.dumps(first, sort_keys=True) + "\n"
    calls = []

    def runner(argv, **kwargs):
        calls.append(list(argv))
        text = " ".join(argv)
        assert "python -c" not in text and "python3 -c" not in text
        assert ";" not in text
        if (len(argv) > 1 and argv[1] == "pull") or (argv and argv[0] == "scp" and "incoming" in argv[-1]):
            dest = Path(argv[-1])
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(older_only, encoding="utf-8")
        if argv[0] == "ssh":
            assert argv[-2:] == ["python3", ".osiris/alpha_state_v1.py"]
        return _completed(argv, stdout='{"helper_version":"1"}\n')

    syncer = key_sync.DualSubstrateKeySync(osiris_home=tmp_path)
    syncer.pub_key_path.write_bytes(b"k" * 32)
    local = syncer.ledger_path
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in (first, newer)), encoding="utf-8")
    before = local.read_bytes()
    receipt = syncer.sync_via_adb(runner=runner, remote_dir=".osiris")
    assert receipt["alpha_probe"]["status"] == "ok"
    assert receipt["alpha_probe"]["argv"][-2:] == ["python3", ".osiris/alpha_state_v1.py"]
    assert receipt["evidence_chain"].startswith("rejected:")
    assert local.read_bytes() == before
    assert any(cmd[:2] == ["adb", "exec-out"] for cmd in calls)

    calls.clear()
    ssh_receipt = syncer.sync_via_ssh(host="phone.invalid", port=8022, runner=runner, remote_dir=".osiris")
    assert ssh_receipt["alpha_probe"]["argv"][-2:] == ["python3", ".osiris/alpha_state_v1.py"]
    assert ssh_receipt["evidence_chain"].startswith("rejected:")
    assert local.read_bytes() == before
    pulled = accept_pulled(tmp_path / "empty.jsonl", older_only.encode("utf-8"))
    assert pulled == "replace"


def test_rlimit_backend_runs_a_child_and_fails_closed_when_unverified() -> None:
    assert IS_SANDBOX is False
    assert ast_firewall.IS_SANDBOX is False
    backend = RlimitSubprocessBackend()
    assert backend.verified is True
    runner = WorkerRunner(backend)
    import time
    cap = RuntimeCapability(
        run_id="run-1",
        task_id="task-1",
        expires_at_utc=time.time() + 60,
        allowed_commands=("echo",),
        allowed_environment=(),
        max_cpu_seconds=5,
        max_memory_mb=256,
        max_disk_mb=32,
        max_processes=8,
    )
    result = runner.run(WorkerRequest("request-1", "echo", ("sprint1",)), cap)
    assert result.status == "completed"
    assert "sprint1" in result.output
    with pytest.raises(IsolationUnavailable):
        WorkerRunner().run(WorkerRequest("request-1", "echo"), cap)
    blind = RlimitSubprocessBackend(probe=lambda: False)
    assert blind.verified is False
    with pytest.raises(IsolationUnavailable):
        blind(WorkerRequest("request-1", "echo"), cap)
