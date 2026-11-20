"""WP3 tests: benchmark task-spec audit + RTX 5050 Node-Beta harness.

* The three `modify` task specs (backlog / claims / gaps) now state their
  contracts explicitly in the prompt the mentor sees, without touching the
  hidden acceptance tests.
* scripts/bench_rtx5050_multimentor.py is substrate-guarded (Node Beta
  x86_64 only), sets the 8 GB VRAM Ollama env, verifies installed tags, and
  never alters active routing.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TASKS = ROOT / "bench_tasks"
SCRIPT = ROOT / "scripts" / "bench_rtx5050_multimentor.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("bench_rtx5050_multimentor", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── task-spec audit: contracts stated in the prompt, hidden tests untouched ──

class TestTaskSpecContracts:
    def _spec(self, name):
        return json.loads((TASKS / name / "task.json").read_text(encoding="utf-8"))["spec"]

    def test_backlog_spec_states_idempotency_and_existing_id(self):
        spec = self._spec("backlog")
        assert "idempotent" in spec
        assert "EXISTING story's id" in spec
        assert "must not store a duplicate" in spec

    def test_claims_spec_states_whole_number_and_cycle_safe_resolution(self):
        spec = self._spec("claims")
        assert "WHOLE number" in spec
        assert "3 inside 13" in spec
        assert "non-recursive" in spec and "cyclic" in spec

    def test_gaps_spec_states_no_length_limit(self):
        spec = self._spec("gaps")
        assert "NO length limit" in spec
        assert "never truncated" in spec
        assert "decimal point" in spec

    def test_hidden_acceptance_tests_untouched(self):
        # The audit may tighten prompts only; acceptance/reference/known_bad
        # must still carry their original assertions.
        acc = (TASKS / "backlog" / "acceptance.py").read_text(encoding="utf-8")
        assert "adding an open story twice must return the existing id" in acc
        acc = (TASKS / "claims" / "acceptance.py").read_text(encoding="utf-8")
        assert "3 is not supported by 13" in acc
        acc = (TASKS / "gaps" / "acceptance.py").read_text(encoding="utf-8")
        assert "a long gap must come out whole" in acc

    def test_suite_hash_changed_and_recorded(self):
        # Tightening task.json necessarily changes suite_sha256; the new hash
        # is what future runs must register (old 09037df63614... runs are not
        # comparable to the tightened suite).
        import osiris_bench
        assert osiris_bench.suite_sha256() != "09037df636145b1e" + "0" * 0 or True  # full hash below
        full = osiris_bench.suite_sha256()
        assert full.startswith("125a27de6d2fe733"), full


# ── RTX 5050 harness ─────────────────────────────────────────────────────────

class TestRtx5050Harness:
    def test_substrate_guard_refuses_non_beta(self, monkeypatch, capsys):
        mod = _load_script()
        monkeypatch.setattr(mod, "is_node_beta", lambda: (False, "substrate is 'aarch64', not Node Beta"))
        assert mod.preflight() == 2
        out = capsys.readouterr().out
        assert "Refusing" in out and "x86_64" in out

    def test_substrate_guard_accepts_beta(self, monkeypatch, capsys):
        mod = _load_script()
        monkeypatch.setattr(mod, "is_node_beta", lambda: (True, "osiris_hardware_mesh: NODE_BETA_COMPUTE"))
        monkeypatch.setattr(mod, "ollama_installed_models",
                            lambda timeout=2.0: ["qwen3.5:9b", "qwen3.5:4b", "gemma4:e4b", "qwen2.5-coder:7b"])
        assert mod.preflight() == 0
        out = capsys.readouterr().out
        assert mod.RTX_ENV["OLLAMA_FLASH_ATTENTION"] == "1"
        assert mod.RTX_ENV["OLLAMA_KV_CACHE_TYPE"] == "q8_0"
        assert "All expected mentor tags present" in out

    def test_drift_reported_honestly_when_tag_missing(self, monkeypatch, capsys):
        mod = _load_script()
        monkeypatch.setattr(mod, "is_node_beta", lambda: (True, "x86_64"))
        monkeypatch.setattr(mod, "ollama_installed_models",
                            lambda timeout=2.0: ["qwen2.5-coder:7b", "qwen3:8b"])
        assert mod.preflight() == 0  # proceeds, but reports the drift
        out = capsys.readouterr().out
        assert "gemma4:e4b" in out and "ollama pull gemma4:e4b" in out

    def test_unreachable_daemon_fails_closed(self, monkeypatch, capsys):
        mod = _load_script()
        monkeypatch.setattr(mod, "is_node_beta", lambda: (True, "x86_64"))
        monkeypatch.setattr(mod, "ollama_installed_models", lambda timeout=2.0: None)
        assert mod.preflight() == 2
        assert "unreachable" in capsys.readouterr().out

    def test_bench_argv_is_advisory_only(self, monkeypatch, capsys):
        mod = _load_script()
        monkeypatch.setattr(mod, "ollama_installed_models",
                            lambda timeout=2.0: ["qwen3.5:9b"])
        calls = []
        monkeypatch.setattr(mod, "run_bench", lambda m, k=1: calls.append((m, k)) or 0)
        monkeypatch.setattr(mod, "preflight", lambda: 0)
        monkeypatch.setattr(sys, "argv", ["bench_rtx5050_multimentor.py", "--models", "qwen3.5:9b,ghost:1b"])
        assert mod.main() == 0
        assert calls == [("qwen3.5:9b", 1)]  # ghost:1b not installed -> skipped
        assert "Skipping (not installed): ghost:1b" in capsys.readouterr().out

    def test_expected_model_set(self):
        mod = _load_script()
        assert set(mod.EXPECTED_MODELS) == {"qwen3.5:9b", "qwen3.5:4b", "gemma4:e4b", "qwen2.5-coder:7b"}
        assert mod.QUICK_TASKS == ("gaps", "claims", "backlog")
