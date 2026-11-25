"""Sprint 0 P0/P1 guards. Each case is the defect, then the fail-closed behavior."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from osiris_cli import chain_lock, probes
from osiris_cli.orchestration import ast_firewall
from osiris_cli.osiris_repl import execute_mutate, execute_tetra1d
from osiris_cli.substrate_guard import (
    SubstrateGuardError,
    accept_return_chain,
    acknowledge,
    append_entry,
    apply_return_chain,
    assert_strict_git_alignment,
    begin_experiment,
    classify_zenodo_evidence,
    load_chain,
    probe_alpha_state,
    record_execution,
    remote_python_defect,
    retire_manifest,
    shell_quotes_multiline_source,
    verify_success_telemetry,
)
from osiris_zenodo_publisher import PublishingWorkflow


def _chain(tmp_path: Path, name: str, bodies: list[dict]) -> list[dict]:
    path = tmp_path / name
    for body in bodies:
        append_entry(path, body)
    return load_chain(path)


def test_historical_remote_python_does_not_parse() -> None:
    bad = "import os; for name in os.listdir('.'):\n    print(name)\n"
    with pytest.raises(SyntaxError):
        ast.parse(bad)
    assert remote_python_defect(bad) is not None
    assert "semicolon" in remote_python_defect(bad)


def test_probe_alpha_state_is_a_versioned_file_not_a_shell_quote() -> None:
    plan = probe_alpha_state(interpreter=sys.executable)
    assert plan["shell"] is False
    assert plan["helper_version"] == "1"
    assert plan["encoding"] == "versioned-file"
    source = Path(plan["helper_path"]).read_text(encoding="utf-8")
    ast.parse(source)
    rendered = " ".join(plan["argv"])
    assert shell_quotes_multiline_source(rendered, source) is False
    assert "\n" not in rendered and ";" not in rendered
    assert json.dumps(source) not in rendered
    completed = subprocess.run(plan["argv"], capture_output=True, text=True, check=False)
    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["helper_version"] == "1"
    assert isinstance(payload["env_present"], list)


def test_ast_firewall_does_not_import_untrusted_modules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = tmp_path / "planted.py"
    artifact.write_text("SENTINEL = 'executed'\n", encoding="utf-8")
    monkeypatch.delitem(sys.modules, "planted", raising=False)

    def explode(*_args, **_kwargs):
        raise AssertionError("untrusted module was loaded")

    monkeypatch.setattr("importlib.util.spec_from_file_location", explode)
    assert ast_firewall.IS_SANDBOX is False
    refused = ast_firewall.refuse_untrusted_dynamic_import(artifact)
    assert refused["error"] == "untrusted_artifact_refused"
    assert "static policy" in refused["reason"]
    assert "planted" not in sys.modules
    tetra = execute_tetra1d()
    mutate = execute_mutate()
    assert tetra["error"] == "untrusted_artifact_refused"
    assert mutate["error"] == "untrusted_artifact_refused"


def test_return_sync_rejects_older_and_divergent_chains(tmp_path: Path) -> None:
    local = _chain(tmp_path, "local.jsonl", [{"kind": "a"}, {"kind": "b"}])
    older = local[:1]
    divergent = _chain(tmp_path, "other.jsonl", [{"kind": "c"}])
    extended = _chain(tmp_path, "ext.jsonl", [{"kind": "a"}, {"kind": "b"}, {"kind": "d"}])
    with pytest.raises(SubstrateGuardError, match="older chain"):
        accept_return_chain(local, older)
    with pytest.raises(SubstrateGuardError, match="divergent"):
        accept_return_chain(local, divergent)
    assert accept_return_chain(local, local) == "identical"
    assert accept_return_chain(local, extended) == "extend"
    target = tmp_path / "alpha.jsonl"
    target.write_text(Path(tmp_path / "local.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    before = target.read_bytes()
    assert apply_return_chain(target, local) == "identical"
    assert target.read_bytes() == before
    assert apply_return_chain(target, extended) == "extend"
    assert [row["kind"] for row in load_chain(target)] == ["a", "b", "d"]
    report = probes.verify_chain(str(target))
    assert report["status"] == "INTACT"
    assert report["entries"] == 3
    assert list(target.parent.glob(target.name + ".*.tmp")) == []


def test_apply_waits_for_the_chain_lock_and_keeps_a_newer_tip(tmp_path: Path) -> None:
    """An append that commits before the rewrite must still be on the ledger."""
    target = tmp_path / "alpha.jsonl"
    local = _chain(tmp_path, "local.jsonl", [{"kind": "a"}, {"kind": "b"}])
    target.write_text((tmp_path / "local.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    extended = _chain(tmp_path, "ext.jsonl", [{"kind": "a"}, {"kind": "b"}, {"kind": "d"}])
    held = threading.Event()
    release = threading.Event()

    def append_under_lock() -> None:
        with chain_lock.locked_tail(str(target)) as (handle, last):
            entry = {
                "kind": "c",
                "prev": last["hash"],
                "seq": int(last["seq"]) + 1,
            }
            from osiris_cli.substrate_guard import _entry_hash

            entry["hash"] = _entry_hash(entry)
            chain_lock.write_line(handle, entry, sort_keys=True, ensure_ascii=False)
            held.set()
            assert release.wait(5)

    worker = threading.Thread(target=append_under_lock)
    worker.start()
    assert held.wait(5)
    outcome: dict = {}

    def apply() -> None:
        try:
            outcome["decision"] = apply_return_chain(target, extended)
        except SubstrateGuardError as exc:
            outcome["error"] = str(exc)

    applier = threading.Thread(target=apply)
    applier.start()
    applier.join(0.2)
    assert applier.is_alive()
    release.set()
    applier.join(5)
    worker.join(5)
    assert "error" in outcome
    assert "does not extend" in outcome["error"] or "divergent" in outcome["error"]
    assert [row["kind"] for row in load_chain(target)] == ["a", "b", "c"]
    assert local[0]["hash"] == load_chain(target)[0]["hash"]


def test_executed_experiment_is_not_run_again_before_ack(tmp_path: Path) -> None:
    ledger = tmp_path / "experiments.jsonl"
    payload = "ab" * 32
    first = begin_experiment(ledger, "exp-1", "key-1", payload)
    assert first["execute"] is True
    record_execution(ledger, "exp-1", "key-1", {"backend": "simulated", "shots": 0})
    # Process died after execution and before ack.
    second = begin_experiment(ledger, "exp-1", "key-1", payload)
    assert second["execute"] is False
    assert second["replayed"] is True
    assert second["result"]["backend"] == "simulated"
    acknowledge(ledger, "exp-1", "key-1")
    third = begin_experiment(ledger, "exp-1", "key-1", payload)
    assert third["execute"] is False and third["phase"] == "acked"
    with pytest.raises(SubstrateGuardError, match="different payload"):
        begin_experiment(ledger, "exp-1", "key-1", "cd" * 32)


def test_manifest_rename_then_verify_does_not_delete_a_replacement(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"generation":1}\n', encoding="utf-8")
    quarantine = tmp_path / "quarantine"

    def verify(path: Path) -> bool:
        manifest.write_text('{"generation":2}\n', encoding="utf-8")
        return path.read_text(encoding="utf-8") == '{"generation":1}\n'

    assert retire_manifest(manifest, quarantine, verify) == "deleted"
    assert manifest.read_text(encoding="utf-8") == '{"generation":2}\n'
    assert list(quarantine.iterdir()) == []

    manifest.write_text('{"generation":3}\n', encoding="utf-8")
    assert retire_manifest(manifest, quarantine, lambda path: False) == "quarantined"
    assert not manifest.exists()
    held = list(quarantine.iterdir())
    assert len(held) == 1
    assert held[0].read_text(encoding="utf-8") == '{"generation":3}\n'


def test_strict_git_alignment_fails_closed_when_commit_info_is_missing(tmp_path: Path) -> None:
    with pytest.raises(SubstrateGuardError, match="local commit"):
        assert_strict_git_alignment(None, "abc")
    with pytest.raises(SubstrateGuardError, match="remote commit"):
        assert_strict_git_alignment("abc", " ")
    with pytest.raises(SubstrateGuardError, match="differ"):
        assert_strict_git_alignment("abc", "def")
    assert_strict_git_alignment("abc", "abc")
    missing = probes.probe_git(tmp_path)
    assert missing.exit_code == 1
    assert "failed closed" in missing.summary
    here = probes.probe_git(Path(__file__).resolve().parents[1])
    assert here.exit_code == 0
    assert "head:" in here.summary


def test_success_telemetry_requires_this_experiment_and_both_stages() -> None:
    other = [
        {"experiment_id": "exp-other", "stage": "dispatch"},
        {"experiment_id": "exp-other", "stage": "result"},
        {"experiment_id": "exp-1", "stage": "dispatch"},
    ]
    with pytest.raises(SubstrateGuardError, match="missing"):
        verify_success_telemetry(other, "exp-1")
    verify_success_telemetry(other + [{"experiment_id": "exp-1", "stage": "result"}], "exp-1")


def test_mock_zenodo_fixture_confidence_is_quarantined() -> None:
    fixture = {"evidence_class": "MOCK", "source": "fixture", "confidence": 0.6, "doi": None}
    verdict = classify_zenodo_evidence(fixture)
    assert verdict["admitted"] is False and verdict["quarantine"] is True
    assert "0.6" in verdict["reason"]
    unmarked = classify_zenodo_evidence({"title": "note without a mock marker"})
    assert unmarked["admitted"] is True and unmarked["confidence"] is None

    class MockResult:
        mock = True
        fixture = True
        evidence_class = "FIXTURE"
        source = "fixture"
        confidence = 0.6

    published = PublishingWorkflow("not-a-token").publish_result(MockResult(), "report")
    assert published["status"] == "QUARANTINED"
