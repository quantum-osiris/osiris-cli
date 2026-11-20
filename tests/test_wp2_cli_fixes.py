"""WP2 regression tests (osiris-cli v6.0.0 WP2, rebased onto origin/main).

Post-rebase behavior under test:

* PR #45's ledger discovery (genome_ledger.ledger_src_candidates /
  ledger_src_problem / _find_ledger_src) -- dynamic, injectable, fail-closed
  when dnalang-core's ledger.py is nowhere. There is deliberately exactly one
  ledger-discovery implementation; osiris_cli/paths.py carries none.
* bare '/sprint' dispatches to a usage screen instead of "Unknown command";
* '/runs --orphans' read-only orphan inspector (console + REPL routing);
* PR #45's shared bench gate (_bench_blocker / _bench_refused) covers G, N,
  /bench, the preflight card and /trial, refuses before any start-of-run
  ledger write when the daemon is offline, the model tag is missing, or the
  ledger recorder is unavailable -- and '/bench ... now' cannot bypass it;
* legacy 'gemma-3-1b' Ollama tag normalized to canonical 'gemma3:1b' in the
  console's model defaults (GemmaProvider manifest id untouched).
"""

import inspect
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

genome_ledger = pytest.importorskip("genome_ledger")
from osiris_cli import osiris_repl

otc = osiris_repl.otc


# ── PR #45 ledger discovery (the single resolver) ────────────────────────────

class TestLedgerSrcCandidates:
    def test_env_override_is_first_candidate(self):
        c = genome_ledger.ledger_src_candidates(environ={"OSIRIS_DNALANG_LEDGER": "/x/ledger.py"},
                                                user_home="/home/u", checkout_parent="/work",
                                                package_dirs=[])
        assert c[0] == "/x/ledger.py"

    def test_osiris_home_src_follows_dnalang_ledger_override(self):
        c = genome_ledger.ledger_src_candidates(environ={"OSIRIS_HOME": "/oh"},
                                                user_home="/home/u", checkout_parent="/work",
                                                package_dirs=[])
        assert c[0] == os.path.join("/oh", "src", "dnalang-core", "dnalang", "ledger.py")
        c2 = genome_ledger.ledger_src_candidates(environ={"OSIRIS_HOME": "/oh",
                                                          "OSIRIS_DNALANG_LEDGER": "/x/ledger.py"},
                                                 user_home="/home/u", checkout_parent="/work",
                                                 package_dirs=[])
        assert c2[0] == "/x/ledger.py" and c2[1] == c[0]

    def test_legacy_order_preserved_then_new_locations(self):
        c = genome_ledger.ledger_src_candidates(environ={}, user_home="/home/u",
                                                checkout_parent="/work", package_dirs=["/pkg/dnalang"])
        tail = os.path.join("dnalang-core", "dnalang", "ledger.py")
        assert c == [
            os.path.join("/work", ".osiris", "src", tail),       # 3. legacy
            os.path.join("/home/u", tail),                        # 4. legacy
            "/pkg/dnalang/ledger.py",                             # 5. installed package
            os.path.join("/home/u", ".osiris", "src", tail),       # 6.
            os.path.join("/work", tail),                          # 7. sibling checkout
            os.path.join("/home/u", "osiris-repos", tail),        # 8. Node Alpha /update layout
        ]

    def test_dedup(self):
        c = genome_ledger.ledger_src_candidates(environ={"OSIRIS_HOME": "/home/u"},
                                                user_home="/home/u", checkout_parent="/work",
                                                package_dirs=[])
        assert len(c) == len(set(c))

    def test_find_picks_first_existing_node_alpha(self, tmp_path):
        # Node Alpha: ~/osiris-repos/dnalang-core/dnalang/ledger.py exists
        home = tmp_path / "home"
        target = home / "osiris-repos" / "dnalang-core" / "dnalang" / "ledger.py"
        target.parent.mkdir(parents=True)
        target.write_text("# ledger\n", encoding="utf-8")
        got = genome_ledger._find_ledger_src(environ={}, user_home=str(home),
                                             checkout_parent=str(tmp_path / "work"),
                                             package_dirs=[])
        assert got == str(target)

    def test_find_picks_sibling_checkout_node_beta(self, tmp_path):
        work = tmp_path / "work"
        target = work / "dnalang-core" / "dnalang" / "ledger.py"
        target.parent.mkdir(parents=True)
        target.write_text("# ledger\n", encoding="utf-8")
        got = genome_ledger._find_ledger_src(environ={}, user_home=str(tmp_path / "home"),
                                             checkout_parent=str(work), package_dirs=[])
        assert got == str(target)

    def test_find_falls_back_to_first_candidate_when_none_exist(self, tmp_path):
        got = genome_ledger._find_ledger_src(environ={}, user_home=str(tmp_path / "home"),
                                             checkout_parent=str(tmp_path / "work"), package_dirs=[])
        assert got == os.path.join(str(tmp_path / "work"), ".osiris", "src",
                                   "dnalang-core", "dnalang", "ledger.py")

    def test_ledger_src_problem_is_fail_closed_and_names_everywhere(self, monkeypatch, tmp_path):
        monkeypatch.setattr(genome_ledger, "LEDGER_SRC", str(tmp_path / "nowhere" / "ledger.py"))
        problem = genome_ledger.ledger_src_problem()
        assert problem is not None and "dnalang-core ledger not found" in problem
        assert "OSIRIS_DNALANG_LEDGER" in problem and "osiris-repos" in problem

    def test_no_competing_resolver_in_paths(self):
        from osiris_cli import paths
        src = inspect.getsource(paths)
        assert "_resolve_dnalang_ledger" not in src and "_resolve_osiris_home" not in src


# ── bare /sprint ─────────────────────────────────────────────────────────────

@pytest.mark.skipif(otc is None, reason="osiris_termux_console not importable on this platform")
class TestBareSprint:
    def test_bare_sprint_dispatches_to_usage(self, monkeypatch):
        called = []
        monkeypatch.setattr(otc, "_sprint_usage", lambda: called.append("usage"))
        assert otc.run_command("/sprint") is True
        assert called == ["usage"]

    def test_sprint_subcommands_still_route(self, monkeypatch):
        seen = []
        monkeypatch.setattr(otc, "_sprint_plan", lambda c: seen.append(("plan", c)))
        monkeypatch.setattr(otc, "_sprint_execute", lambda c: seen.append(("execute", c)))
        monkeypatch.setattr(otc, "_sprint_review", lambda: seen.append(("review",)))
        assert otc.run_command("/sprint plan") is True
        assert otc.run_command("/sprint execute #3") is True
        assert otc.run_command("/sprint review") is True
        assert seen == [("plan", "/sprint plan"), ("execute", "/sprint execute #3"), ("review",)]

    def test_repl_bare_sprint_not_unknown(self, monkeypatch):
        monkeypatch.setattr(otc, "_sprint_usage", lambda: None)
        unknown = []
        monkeypatch.setattr(otc, "_unknown_command", lambda t: unknown.append(t))
        osiris_repl.dispatch_command(osiris_repl.SESSION, "/sprint")
        assert unknown == []

    def test_usage_screen_lists_subcommands(self, monkeypatch, capsys):
        monkeypatch.setattr(otc.sprint_manager, "summary",
                            lambda: ({"backlog": 2, "active": 1}, []))
        otc._sprint_usage()
        out = capsys.readouterr().out
        assert "/sprint plan" in out and "/sprint execute" in out and "/sprint review" in out
        assert "active 1" in out


# ── /runs --orphans ──────────────────────────────────────────────────────────

@pytest.mark.skipif(otc is None, reason="osiris_termux_console not importable on this platform")
class TestRunsOrphans:
    def _ledger_entries(self, monkeypatch, tmp_path):
        monkeypatch.setattr(genome_ledger, "DEFAULT_PATH", str(tmp_path / "no-ledger.jsonl"))
        monkeypatch.setattr(otc.genome_ledger, "DEFAULT_PATH", str(tmp_path / "no-ledger.jsonl"))
        monkeypatch.setattr(otc.run_record, "RUNS_DIR", str(tmp_path / "runs"))
        return tmp_path / "runs"

    def test_console_routes_orphans_flag(self, monkeypatch):
        called = []
        monkeypatch.setattr(otc, "_runs_orphans", lambda: called.append(1))
        assert otc.run_command("/runs --orphans") is True
        assert called == [1]

    def test_repl_routes_orphans_flag(self, monkeypatch):
        called = []
        monkeypatch.setattr(otc, "_runs_orphans", lambda: called.append(1))
        osiris_repl.dispatch_command(osiris_repl.SESSION, "/runs --orphans")
        assert called == [1]

    def test_plain_runs_still_stats(self, monkeypatch):
        called = []
        monkeypatch.setattr(otc, "_runs_stats", lambda: called.append("stats"))
        monkeypatch.setattr(otc, "_runs_orphans", lambda: called.append("orphans"))
        assert otc.run_command("/runs") is True
        assert called == ["stats"]

    def test_orphans_reported_read_only(self, monkeypatch, capsys, tmp_path):
        runs = self._ledger_entries(monkeypatch, tmp_path)
        orphan = runs / "run-abc123"
        orphan.mkdir(parents=True)
        (orphan / "run.json").write_text("{}", encoding="utf-8")
        empty = runs / "run-def456"
        empty.mkdir()
        otc._runs_orphans()
        out = capsys.readouterr().out
        assert "run-abc123" in out and "run-def456" in out
        assert "ledger append failed" in out          # record exists, no ledger vouch
        assert "record write never landed" in out      # empty directory
        assert not (tmp_path / "no-ledger.jsonl").exists()  # nothing written

    def test_no_orphans_message(self, monkeypatch, capsys, tmp_path):
        self._ledger_entries(monkeypatch, tmp_path)
        otc._runs_orphans()
        assert "none" in capsys.readouterr().out

    def test_orphans_flag_is_read_only_command(self):
        assert "/runs --orphans" in otc.READ_ONLY_COMMANDS
        assert otc._is_read_only("/runs --orphans")


# ── PR #45 shared bench gate: _bench_blocker / _bench_refused ────────────────

@pytest.mark.skipif(otc is None, reason="osiris_termux_console not importable on this platform")
class TestBenchBlocker:
    def test_daemon_offline_blocks(self, monkeypatch):
        monkeypatch.setattr(otc, "_ollama_models", lambda: None)
        monkeypatch.setattr(otc.genome_ledger, "ledger_src_problem", lambda: None)
        title, text = otc._bench_blocker("ollama", "qwen2.5-coder:7b")
        assert title == "BACKEND UNREACHABLE" and "offline" in text

    def test_missing_model_blocks_with_installed_list(self, monkeypatch):
        monkeypatch.setattr(otc, "_ollama_models", lambda: {"qwen2.5-coder:7b", "gemma3:4b"})
        monkeypatch.setattr(otc.genome_ledger, "ledger_src_problem", lambda: None)
        title, text = otc._bench_blocker("ollama", "nope:9b")
        assert title == "MODEL NOT INSTALLED"
        assert "qwen2.5-coder:7b" in text and "nope:9b" in text and "ollama pull nope:9b" in text

    def test_installed_model_passes_model_check(self, monkeypatch):
        monkeypatch.setattr(otc, "_ollama_models", lambda: {"qwen2.5-coder:7b"})
        monkeypatch.setattr(otc.genome_ledger, "ledger_src_problem", lambda: None)
        assert otc._bench_blocker("ollama", "qwen2.5-coder:7b") is None

    def test_latest_alias_passes(self, monkeypatch):
        monkeypatch.setattr(otc, "_ollama_models", lambda: {"qwen2.5-coder:latest"})
        monkeypatch.setattr(otc.genome_ledger, "ledger_src_problem", lambda: None)
        assert otc._bench_blocker("ollama", "qwen2.5-coder") is None

    def test_a_different_tag_is_not_the_latest_alias(self, monkeypatch):
        monkeypatch.setattr(otc, "_ollama_models", lambda: {"qwen2.5-coder:latest"})
        monkeypatch.setattr(otc.genome_ledger, "ledger_src_problem", lambda: None)
        assert otc._bench_blocker("ollama", "qwen2.5-coder:7b")[0] == "MODEL NOT INSTALLED"

    def test_missing_ledger_recorder_blocks_even_for_installed_model(self, monkeypatch):
        # fail-closed: a round that cannot be recorded must not start
        monkeypatch.setattr(otc, "_ollama_models", lambda: {"qwen2.5-coder:7b"})
        monkeypatch.setattr(otc.genome_ledger, "ledger_src_problem",
                            lambda: "dnalang-core ledger not found at /x (also looked in: ...)")
        title, text = otc._bench_blocker("ollama", "qwen2.5-coder:7b")
        assert title == "RECORDER UNAVAILABLE" and "cannot be recorded" in text

    def test_non_ollama_backend_still_checks_recorder(self, monkeypatch):
        monkeypatch.setattr(otc.genome_ledger, "ledger_src_problem", lambda: "ledger missing")
        title, _ = otc._bench_blocker("gemini", None)
        assert title == "RECORDER UNAVAILABLE"

    def test_bench_refuses_before_preflight_and_run(self, monkeypatch, capsys):
        monkeypatch.setattr(otc, "_ollama_models", lambda: {"qwen2.5-coder:7b"})
        monkeypatch.setattr(otc.genome_ledger, "ledger_src_problem", lambda: None)
        preflight, run = [], []
        monkeypatch.setattr(otc, "_bench_preflight", lambda *a: preflight.append(a))
        monkeypatch.setattr(otc, "_bench_run", lambda *a: run.append(a) or (1, "x"))
        monkeypatch.setattr(otc, "_bench_argv",
                            lambda c: (["osiris_bench.py", "--events", "--backend", "ollama",
                                        "--model", "ghost:1b", "--k", "1"], "ghost:1b"))
        otc._bench("/bench ghost:1b")
        out = capsys.readouterr().out
        assert "MODEL NOT INSTALLED" in out
        assert preflight == [] and run == []   # refused before card AND before any run/ledger write

    def test_bench_now_cannot_bypass_the_gate(self, monkeypatch, capsys):
        # '/bench ghost:1b now' skips the preflight card but NOT the blocker
        monkeypatch.setattr(otc, "_ollama_models", lambda: {"qwen2.5-coder:7b"})
        monkeypatch.setattr(otc.genome_ledger, "ledger_src_problem", lambda: None)
        run = []
        monkeypatch.setattr(otc, "_bench_run", lambda *a: run.append(a) or (1, "x"))
        monkeypatch.setattr(otc, "_bench_argv",
                            lambda c: (["osiris_bench.py", "--events", "--backend", "ollama",
                                        "--model", "ghost:1b", "--k", "1"], "ghost:1b"))
        otc._bench("/bench ghost:1b now")
        assert "MODEL NOT INSTALLED" in capsys.readouterr().out
        assert run == []

    def test_preflight_card_uses_same_gate(self, monkeypatch, capsys):
        monkeypatch.setattr(otc, "_ollama_models", lambda: None)
        monkeypatch.setattr(otc.genome_ledger, "ledger_src_problem", lambda: None)
        argv = ["osiris_bench.py", "--events", "--backend", "ollama", "--model", "qwen2.5-coder:7b", "--k", "1"]
        otc._bench_preflight("/bench", argv, "label", otc.osiris_ui.Canvas())
        assert "BACKEND UNREACHABLE" in capsys.readouterr().out

    def test_trial_goes_through_the_same_gate(self):
        src = inspect.getsource(otc)
        trial = src[src.index("def _trial"):]
        trial = trial[:trial.index("\ndef ")] if "\ndef " in trial else trial
        assert "_bench_refused" in trial
        assert "_bench_blocker" in src

    def test_no_leftover_mentor_model_guard(self):
        assert not hasattr(otc, "_mentor_model_guard")


# ── canonical Ollama tag ─────────────────────────────────────────────────────

@pytest.mark.skipif(otc is None, reason="osiris_termux_console not importable on this platform")
class TestCanonicalModelTag:
    def test_console_defaults_use_gemma3_1b(self):
        src = inspect.getsource(otc)
        assert 'os.environ.get("OSIRIS_CHAT_MODEL", "gemma3:1b")' in src
        assert 'os.environ.get("OSIRIS_CODE_MODEL", "gemma3:1b")' in src

    def test_no_legacy_ollama_tag_left_in_console(self):
        assert '"gemma-3-1b"' not in inspect.getsource(otc)

    def test_gemma_provider_manifest_untouched(self):
        from osiris_cli.providers import gemma
        assert "gemma-3-1b" in gemma.GEMMA_MANIFEST
