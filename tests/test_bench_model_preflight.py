"""osiris_bench --backend ollama must refuse to start (and write nothing to the
ledger) when the requested model is not installed: Ollama answers every call
for a missing model with HTTP 404, which used to be recorded as a "complete"
round of coding failures (run bench-20261116T150659Z-55b952)."""

import json
import os
import sys

import pytest

BIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BIN)

import osiris_bench  # noqa: E402

TAGS = json.dumps({"models": [{"name": "qwen2.5-coder:7b"}, {"name": "gemma3:4b"},
                              {"name": "llama3:latest"}]}).encode()


def _tags():
    return TAGS


def test_installed_model_passes():
    assert osiris_bench.ollama_preflight("qwen2.5-coder:7b", fetch=_tags) is None


def test_latest_alias_passes():
    assert osiris_bench.ollama_preflight("llama3", fetch=_tags) is None


def test_absent_model_is_diagnosed_with_installed_names():
    msg = osiris_bench.ollama_preflight("qwen2.5-coder:3b", fetch=_tags)
    assert "not installed" in msg and "HTTP 404" in msg
    assert "qwen2.5-coder:7b" in msg and "gemma3:4b" in msg


def test_prefix_is_not_a_match():
    # "qwen2.5-coder" must not silently resolve to "qwen2.5-coder:7b".
    assert osiris_bench.ollama_preflight("qwen2.5-coder", fetch=_tags) is not None


def test_unreachable_daemon_is_diagnosed():
    def down():
        raise OSError("[Errno 111] Connection refused")
    msg = osiris_bench.ollama_preflight("gemma3:4b", fetch=down)
    assert "unreachable" in msg


@pytest.mark.parametrize("body", [b"not json", b"{}", b'{"models": 3}'])
def test_unreadable_model_list_fails_closed(body):
    assert osiris_bench.ollama_preflight("gemma3:4b", fetch=lambda: body) is not None


def test_no_model_selected_fails_closed():
    assert osiris_bench.ollama_preflight(None, fetch=_tags) is not None


def test_main_refuses_and_writes_no_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(osiris_bench, "ollama_preflight",
                        lambda model, fetch=None: f"Ollama model '{model}' is not installed")
    monkeypatch.setattr(osiris_bench, "_load_osiris",
                        lambda: type("O", (), {"_sandbox_probe": staticmethod(lambda: None)})())
    out = tmp_path / "bench"
    rc = osiris_bench.main(["--backend", "ollama", "--model", "absent:1b",
                            "--tasks", "backlog", "--out", str(out)])
    assert rc == 1
    assert not (out / "results.ledger.jsonl").exists()


def test_previous_runner_hash_is_registered_compatible():
    import protege
    assert "e48927f381297e3d05f7f0e9c42226548c905998390939293bdf45fb720d8f96" in protege.COMPATIBLE_RUNNERS
