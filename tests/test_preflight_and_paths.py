"""Stage 2 offline tests: unified Ollama preflight, canonical tag resolution,
Termux split-layout dnalang-core path discovery, trial terminal end-entry guard,
and truthful L4 /architecture state."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

import bench_trial
import genome_ledger
import osiris_bench
import osiris_termux_console as otc
import protege
from osiris_cli import architecture as arch
from osiris_cli.living import Osiris

FIXTURE_LEDGER = Path(__file__).parent / "fixtures" / "dnalang_ledger_v0_2_0.py"


def _put_ledger(root: Path) -> str:
    target = root / "dnalang-core" / "dnalang" / "ledger.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FIXTURE_LEDGER, target)
    return str(target)


def test_canonical_ollama_tag_resolution_and_no_bare_prefix_match() -> None:
    installed = {"gemma3:1b", "llama3:latest"}
    assert otc.canonicalize_ollama_tag("gemma-3-1b") == "gemma3:1b"
    assert otc.canonicalize_ollama_tag("gemma-2-2b") == "gemma2:2b"
    assert otc._resolve_installed_ollama_tag("gemma-3-1b", installed) == "gemma3:1b"
    assert otc._resolve_installed_ollama_tag("gemma3:1b", installed) == "gemma3:1b"
    assert otc._resolve_installed_ollama_tag("llama3", installed) == "llama3:latest"
    # Bare family names and uninstalled tags must still fail closed.
    assert otc._resolve_installed_ollama_tag("gemma3", installed) is None
    assert otc._resolve_installed_ollama_tag("qwen2.5-coder:7b", installed) is None


def test_suggest_blocks_offline_ollama_and_normalizes_legacy_tag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(otc, "query_model", lambda prompt, model, **kw: (calls.append((model, prompt)), "SUGGESTION 1: STATUS -- inspect state")[1])

    # 1. Offline daemon fails closed without calling query_model
    monkeypatch.setattr(otc, "_ollama_models", lambda: None)
    otc._suggest("/suggest")
    assert calls == []
    assert "Ollama daemon is offline" in capsys.readouterr().out

    # 2. Legacy CHAT_MODEL='gemma-3-1b' resolves cleanly to installed 'gemma3:1b'
    monkeypatch.setattr(otc, "CHAT_MODEL", "gemma-3-1b")
    monkeypatch.setattr(otc, "_ollama_models", lambda: {"gemma3:1b"})
    otc._suggest("/suggest")
    assert len(calls) == 1
    assert calls[0][0] == "gemma3:1b"
    assert "Full status" in capsys.readouterr().out


def test_next_best_picks_installed_node_alpha_model_instead_of_uninstalled_7b(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ledger = tmp_path / "dnalang" / "ledger.py"
    ledger.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURE_LEDGER, ledger)
    monkeypatch.setattr(genome_ledger, "LEDGER_SRC", str(ledger))
    monkeypatch.setattr(otc, "_pending_write", {"path": None})
    monkeypatch.setattr(otc.gaps, "drafts", lambda: [])
    monkeypatch.setattr(otc.sprint_manager, "summary", lambda: ({}, []))
    monkeypatch.setattr(otc, "_ignite_eligible", lambda g: False)
    monkeypatch.setattr(protege, "load_ledger", lambda bench_dir=None: ([], "no benchmark yet"))
    monkeypatch.setattr(otc, "_ollama_models", lambda: {"gemma3:1b"})

    nb = otc._next_best()
    assert nb["cmd"] == "/bench gemma3:1b"
    assert "gemma3:1b" in nb["label"]
    assert "qwen2.5-coder:7b" not in nb["label"]


def test_bench_trial_preflight_blocks_missing_model_and_closes_aborted_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "bench"
    monkeypatch.setattr(osiris_bench, "_load_osiris", lambda: type("O", (), {"_sandbox_probe": staticmethod(lambda: None), "CODE_MODEL": "qwen2.5-coder:7b"})())
    monkeypatch.setattr(osiris_bench, "ollama_preflight", lambda model, fetch=None: f"Ollama model '{model}' is not installed")
    rc = bench_trial.main(["--strategy", "invariant-first-v1", "--backend", "ollama", "--model", "absent:7b", "--tasks", "backlog", "--out", str(out_dir)])
    assert rc == 1
    assert not (out_dir / "results.ledger.jsonl").exists()
    assert "not installed" in capsys.readouterr().err

    # Unexpected failure mid-run writes a terminal end entry (0 orphan runs).
    ledger_src = _put_ledger(tmp_path / "repos")
    monkeypatch.setattr(osiris_bench, "LEDGER_SRC", ledger_src)
    monkeypatch.setattr(osiris_bench, "ollama_preflight", lambda model, fetch=None: None)
    monkeypatch.setattr(osiris_bench, "run_suite", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("simulated mid-run crash")))

    with pytest.raises(RuntimeError, match="simulated mid-run crash"):
        bench_trial.main(["--strategy", "invariant-first-v1", "--backend", "ollama", "--model", "gemma3:1b", "--tasks", "backlog", "--out", str(out_dir)])

    entries, problem = protege.load_ledger(str(out_dir))
    assert problem is None
    assert [e["kind"] for e in entries] == ["bench_run_start", "bench_run_end"]
    assert protege.run_status(entries)["unterminated"] == 0


def test_architecture_l4_reports_installed_vs_uninstalled_models_truthfully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger_path = tmp_path / "genome_ledger.jsonl"
    monkeypatch.setattr(genome_ledger, "DEFAULT_PATH", str(ledger_path))
    monkeypatch.setattr(genome_ledger, "LEDGER_SRC", str(FIXTURE_LEDGER))
    monkeypatch.setattr(genome_ledger, "_ledger_mod", None)

    class NodeAlphaConsole:
        CHAT_MODEL = "gemma-3-1b"
        _pending_write = {"path": None}

        class research:
            @staticmethod
            def sources():
                return []

        @staticmethod
        def _next_best():
            return {"label": "Benchmark gemma3:1b on 3 tasks"}

        @staticmethod
        def _check_ollama_reachable(timeout=0.5):
            return True, "http://localhost:11434"

        @staticmethod
        def _ollama_models():
            return {"gemma3:1b"}

        @staticmethod
        def _resolve_installed_ollama_tag(tag, installed):
            return otc._resolve_installed_ollama_tag(tag, installed)

        @staticmethod
        def _synth_chain():
            return "qwen2.5-coder:7b"

        @staticmethod
        def _integrity():
            return "verified", "bench evidence 9 files verified"

    class EdgeMentor:
        base = "http://localhost:11434"

        def model(self):
            return "gemma3:1b"

        def fast_model(self):
            return None

    living_obj = Osiris(core=None, mentor=EdgeMentor(), home=str(tmp_path / "living"), out=lambda s: None, background=False, tty=False)
    state = arch.live_state(living_obj, NodeAlphaConsole)
    kind, line = state["L4"]
    assert kind == "review"
    assert "qwen2.5-coder:7b [not installed]" in line
    assert "installed: gemma3:1b" in line
    assert line.endswith("voice gemma3:1b")


def test_termux_split_layout_and_runner_sha256_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "termux_home"
    repos = home / "osiris-repos"
    (repos / "osiris-cli").mkdir(parents=True)
    expected = _put_ledger(repos)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("OSIRIS_HOME", str(home / ".osiris"))
    monkeypatch.delenv("OSIRIS_DNALANG_LEDGER", raising=False)
    monkeypatch.setattr(genome_ledger, "HOME_DIR", str(repos))
    monkeypatch.setattr(genome_ledger.importlib.util, "find_spec", lambda name, package=None: None)
    assert genome_ledger._find_ledger_src() == expected
    assert osiris_bench.runner_sha256() == "af49b2bd7707c36ff94cc0f87d307846294afefb5a35cb808f8524d41e53264c"
