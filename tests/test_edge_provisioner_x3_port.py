"""The agent-council Elite x3 provisioner, ported into the phase-2 edge_provisioner (2026-11-19):
machine-specific locations come from the environment, flashing needs --confirm, and the old
`/provision` / `/sync provision` subcommand names still reach the same features."""

import json

import pytest

from osiris_cli import edge_provisioner as ep
from osiris_cli import osiris_repl


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("OSIRIS_X3_PROV_DIR", "OSIRIS_X3_FLASH_PS1", "OSIRIS_X3_OPERATOR_ACCOUNT", "OSIRIS_PPKG_SCRIPT"):
        monkeypatch.delenv(k, raising=False)


def test_unconfigured_artifacts_report_missing_not_a_personal_path():
    rep = ep.AdaptiveEdgeProvisioner.verify_artifacts()
    assert rep["x3_provisioning_dir"] is None
    assert not any(rep[k] for k in ("ppkg_exists", "appx_exists", "cer_exists", "dnalang_ledger_exists"))
    assert "not configured" in ep._x3_operator_account()


def test_configured_dir_is_used_for_artifacts(tmp_path, monkeypatch):
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / ep.X3_ARTIFACTS["appx"]).write_bytes(b"appx")
    monkeypatch.setenv("OSIRIS_X3_PROV_DIR", str(tmp_path))
    monkeypatch.setenv("OSIRIS_X3_OPERATOR_ACCOUNT", "operator@example.invalid")
    rep = ep.AdaptiveEdgeProvisioner.verify_artifacts()
    assert rep["appx_exists"] and rep["appx_bytes"] == 4 and rep["x3_provisioning_dir"] == str(tmp_path)
    assert ep._x3_operator_account() == "operator@example.invalid"


def test_flashing_an_ffu_image_requires_confirm(monkeypatch, capsys):
    ran = []
    monkeypatch.setattr(ep.subprocess, "run", lambda *a, **k: ran.append(a) or None)
    assert ep.run_provision_command(["flash", "image.ffu"]) == 2
    assert ran == [] and "--confirm" in capsys.readouterr().out


def test_flash_without_a_configured_script_fails_closed(monkeypatch):
    ran = []
    monkeypatch.setattr(ep.subprocess, "run", lambda *a, **k: ran.append(a) or None)
    assert ep.run_provision_command(["flash", "image.ffu", "--confirm"]) == 1
    assert ran == []


def test_missing_ppkg_pipeline_fails_closed(monkeypatch):
    monkeypatch.setattr(ep, "_local_tool", lambda name: None)
    assert ep.run_provision_command(["build"]) == 1


def test_artifacts_subcommand_prints_json(capsys):
    assert ep.run_provision_command(["artifacts"]) == 0
    assert "ppkg_exists" in json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("legacy,expected", [
    (["status"], ["x3"]), (["discover"], ["x3"]), (["deploy"], ["build"]), (["ppkg"], ["build"]),
    (["ffu", "x.ffu"], ["flash", "x.ffu"]), (["verify"], ["artifacts"]),
    (["verify", "/tmp/bundle"], ["verify", "/tmp/bundle"]), (["stage", "--out", "d"], ["stage", "--out", "d"]),
])
def test_legacy_subcommand_names_are_translated(monkeypatch, legacy, expected):
    seen = []
    monkeypatch.setattr(ep, "run_provision_command", lambda argv: seen.append(list(argv)) or 0)
    osiris_repl.execute_provision(legacy)
    assert seen == [expected]
