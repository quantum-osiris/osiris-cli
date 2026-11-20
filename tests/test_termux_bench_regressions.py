"""Two mentor-round defects seen on Termux (Node Alpha, 2026-11-19), and the gate that covers both.

1. dnalang-core's ledger.py was looked for only at <parent of the checkout>/.osiris/src,
   ~/dnalang-core and an installed package. On the phone the checkout is
   ~/osiris-repos/osiris-cli, so the first of those was ~/osiris-repos/.osiris/..., which never
   exists, while dnalang-core sat beside it at ~/osiris-repos/dnalang-core:

       [!] Recording failed: dnalang-core ledger not found at
           /data/data/com.termux/files/home/osiris-repos/.osiris/src/dnalang-core/dnalang/ledger.py

2. A round could be started that could not be recorded, and /trial started a round without the
   model-installed check /bench makes. Every way of starting a round now goes through one gate.

Nothing here reaches Ollama or a real ~/.osiris: model lists are injected, every path is tmp_path.
"""

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

BIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BIN)

import genome_ledger  # noqa: E402
import osiris_bench  # noqa: E402
import osiris_termux_console as otc  # noqa: E402
import protege  # noqa: E402

FIXTURE_LEDGER = Path(__file__).parent / "fixtures" / "dnalang_ledger_v0_2_0.py"
TAIL = os.path.join("dnalang-core", "dnalang", "ledger.py")


def _put_ledger(root: Path) -> str:
    """A real dnalang ledger.py (the pinned copy) at <root>/dnalang-core/dnalang/ledger.py."""
    target = root / "dnalang-core" / "dnalang" / "ledger.py"
    target.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURE_LEDGER, target)
    return str(target)


@pytest.fixture
def termux(tmp_path, monkeypatch):
    """The phone's layout: ~/osiris-repos/osiris-cli, no installed dnalang package, no overrides."""
    home = tmp_path / "home"
    repos = home / "osiris-repos"
    (repos / "osiris-cli").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("OSIRIS_DNALANG_LEDGER", raising=False)
    monkeypatch.delenv("OSIRIS_HOME", raising=False)
    monkeypatch.setattr(genome_ledger, "HOME_DIR", str(repos))
    monkeypatch.setattr(genome_ledger.importlib.util, "find_spec", lambda name, package=None: None)
    return home, repos


# ── 1. where dnalang-core's ledger.py is looked for ──────────────────────────────────────────────

def test_the_sibling_checkout_under_osiris_repos_is_found(termux):
    home, repos = termux
    expected = _put_ledger(repos)
    assert genome_ledger._find_ledger_src() == expected
    assert "osiris-repos/.osiris" not in genome_ledger._find_ledger_src()


def test_home_osiris_repos_is_found_when_the_checkout_is_elsewhere(termux, tmp_path, monkeypatch):
    home, repos = termux
    monkeypatch.setattr(genome_ledger, "HOME_DIR", str(tmp_path / "somewhere-else"))
    expected = _put_ledger(repos)
    assert genome_ledger._find_ledger_src() == expected


def test_the_state_dir_is_found_wherever_the_checkout_is(termux):
    home, _repos = termux
    expected = _put_ledger(home / ".osiris" / "src")
    assert genome_ledger._find_ledger_src() == expected


def test_osiris_home_names_the_state_dir_and_wins_over_the_checkouts(termux, tmp_path, monkeypatch):
    _home, repos = termux
    _put_ledger(repos)
    state = tmp_path / "state"
    expected = _put_ledger(state / "src")
    monkeypatch.setenv("OSIRIS_HOME", str(state))
    assert genome_ledger._find_ledger_src() == expected


def test_an_explicit_ledger_file_wins_over_everything(termux, tmp_path, monkeypatch):
    home, repos = termux
    _put_ledger(repos)
    _put_ledger(home / ".osiris" / "src")
    explicit = tmp_path / "explicit" / "ledger.py"
    explicit.parent.mkdir()
    shutil.copyfile(FIXTURE_LEDGER, explicit)
    monkeypatch.setenv("OSIRIS_DNALANG_LEDGER", str(explicit))
    monkeypatch.setenv("OSIRIS_HOME", str(home / ".osiris"))
    assert genome_ledger._find_ledger_src() == str(explicit)


def test_a_machine_that_already_resolved_a_ledger_resolves_the_same_one(termux, tmp_path):
    """The three places looked in before keep their order and stay ahead of the new ones, so the
    genome ledger of a working install is not suddenly verified with a different ledger.py."""
    home, repos = termux
    package = tmp_path / "site-packages" / "dnalang"
    new_places = [os.path.join(str(home), ".osiris", "src", TAIL), os.path.join(str(repos), TAIL),
                  os.path.join(str(home), "osiris-repos", TAIL)]
    order = genome_ledger.ledger_src_candidates({}, str(home), str(repos), [str(package)])
    old = [os.path.join(str(repos), ".osiris", "src", TAIL), os.path.join(str(home), TAIL),
           os.path.join(str(package), "ledger.py")]
    assert order[:3] == old
    assert [c for c in order if c in new_places] == list(dict.fromkeys(new_places))
    assert all(order.index(o) < order.index(n) for o in old for n in set(new_places))
    # ... and with a file in an old place and a new place, the old place still wins.
    _put_ledger(repos)
    legacy = _put_ledger(repos / ".osiris" / "src")
    assert genome_ledger._find_ledger_src({}, str(home), str(repos), []) == legacy


def test_candidates_read_no_disk_and_do_not_repeat(tmp_path):
    home = str(tmp_path / "h")
    order = genome_ledger.ledger_src_candidates({}, home, os.path.join(home, "osiris-repos"), [])
    assert len(order) == len(set(order))
    assert not os.path.exists(home)


def test_when_nothing_is_found_the_problem_names_every_place_looked_in(termux, monkeypatch):
    home, repos = termux
    missing = genome_ledger._find_ledger_src()
    assert not os.path.exists(missing)
    monkeypatch.setattr(genome_ledger, "LEDGER_SRC", missing)
    problem = genome_ledger.ledger_src_problem()
    assert problem and "dnalang-core ledger not found" in problem
    for place in (os.path.join(str(repos), TAIL), os.path.join(str(home), ".osiris", "src", TAIL),
                  "OSIRIS_DNALANG_LEDGER"):
        assert place in problem
    monkeypatch.setattr(genome_ledger, "LEDGER_SRC", _put_ledger(repos))
    assert genome_ledger.ledger_src_problem() is None


def test_a_round_is_recorded_start_to_end_with_the_resolved_ledger(termux, tmp_path, monkeypatch):
    """The failure itself, end to end: on the phone's layout a run log opens, and closes."""
    _home, repos = termux
    _put_ledger(repos)
    monkeypatch.setattr(osiris_bench, "LEDGER_SRC", genome_ledger._find_ledger_src())
    out = tmp_path / "bench"
    log = osiris_bench.RunLog(str(out), [{"id": "backlog"}], 1, "ollama", "gemma3:1b")
    log.end({}, None)
    kinds = [json.loads(line)["kind"] for line in (out / "results.ledger.jsonl").read_text().splitlines()]
    assert kinds == ["bench_run_start", "bench_run_end"]
    entries, problem = protege.load_ledger(str(out))
    assert problem is None and len(entries) == 2


def test_a_missing_recorder_writes_no_run_entry_at_all(tmp_path, monkeypatch):
    """What the phone's failed round left behind: nothing. The ledger source is checked before the
    start entry is appended, so there is no unclosed run to clean up. (Holds before this fix too;
    pinned here because the message printed at the time said 'run left without an end entry'.)"""
    monkeypatch.setattr(osiris_bench, "LEDGER_SRC", str(tmp_path / "missing" / "ledger.py"))
    monkeypatch.setattr(osiris_bench, "ollama_preflight", lambda model, fetch=None: None)
    monkeypatch.setattr(osiris_bench, "_load_osiris",
                        lambda: type("O", (), {"_sandbox_probe": staticmethod(lambda: None)})())
    out = tmp_path / "bench"
    rc = osiris_bench.main(["--backend", "ollama", "--model", "gemma3:1b", "--tasks", "backlog",
                            "--out", str(out)])
    assert rc == 1
    assert not out.exists()


# ── 2. one gate for every way a round starts ─────────────────────────────────────────────────────

@pytest.fixture
def console(monkeypatch, tmp_path):
    """The console with a recorder present, no round able to really run, and an empty menu."""
    ledger = tmp_path / "recorder" / "ledger.py"
    ledger.parent.mkdir()
    shutil.copyfile(FIXTURE_LEDGER, ledger)
    monkeypatch.setattr(genome_ledger, "LEDGER_SRC", str(ledger))
    started = []
    monkeypatch.setattr(otc, "_bench_run", lambda argv, label, ui: (started.append(argv), (1, "stub"))[1])
    monkeypatch.setattr(otc, "_notify", lambda *a, **k: None)
    monkeypatch.setattr(otc, "_ask", lambda *a, **k: pytest.fail("asked to confirm a round that must not start"))
    monkeypatch.setattr(otc, "_pending_suggestions", {})
    return started


def test_the_gate_passes_only_an_installed_model_with_a_recorder(console, monkeypatch, tmp_path):
    monkeypatch.setattr(otc, "_ollama_models", lambda: {"gemma3:1b", "llama3:latest"})
    assert otc._bench_blocker("ollama", "gemma3:1b") is None
    assert otc._bench_blocker("ollama", "llama3") is None            # the :latest alias
    assert otc._bench_blocker("ollama", "qwen2.5-coder:7b")[0] == "MODEL NOT INSTALLED"
    assert otc._bench_blocker("ollama", "gemma3")[0] == "MODEL NOT INSTALLED"   # a prefix is no match
    assert otc._bench_blocker("ollama", None)[0] == "MODEL NOT INSTALLED"
    assert otc._bench_blocker("ollama", "")[0] == "MODEL NOT INSTALLED"
    monkeypatch.setattr(otc, "_ollama_models", lambda: None)
    assert otc._bench_blocker("ollama", "gemma3:1b")[0] == "BACKEND UNREACHABLE"
    monkeypatch.setattr(otc, "_ollama_models", lambda: set())
    assert otc._bench_blocker("ollama", "gemma3:1b")[0] == "MODEL NOT INSTALLED"
    monkeypatch.setattr(otc, "_ollama_models", lambda: {"gemma3:1b"})
    monkeypatch.setattr(genome_ledger, "LEDGER_SRC", str(tmp_path / "nowhere" / "ledger.py"))
    assert otc._bench_blocker("ollama", "gemma3:1b")[0] == "RECORDER UNAVAILABLE"
    assert otc._bench_blocker("gemini", None)[0] == "RECORDER UNAVAILABLE"


def test_hotkey_g_goes_through_the_same_command_and_gate_as_n(console, monkeypatch, capsys):
    """G is '/bench' -- the command N gives when no model is measured -- so it cannot skip a check
    N makes. Dispatched the way the REPL does it, with a model that is not installed."""
    from osiris_cli import osiris_repl
    if osiris_repl.otc is not otc:
        pytest.skip("the REPL did not load this console module")
    monkeypatch.setattr(otc, "_ollama_models", lambda: {"gemma3:1b"})
    letters = {key: cmd for key, _label, cmd in otc._home_letters()}
    assert letters["G"] == "/bench"
    state = osiris_repl.OsirisReplState()
    for typed in ("G", "g", "/bench", "/bench now", f"/bench {otc.BENCH_DEFAULT_MODEL}",
                  f"/bench {otc.BENCH_DEFAULT_MODEL} now", "/bench all now"):
        osiris_repl.dispatch_command(state, typed)
        out = capsys.readouterr().out
        assert "MODEL NOT INSTALLED" in out and "gemma3:1b" in out, typed
        assert "PREFLIGHT" not in out, typed
        assert "1" not in otc._pending_suggestions, typed             # no "[1] Start the round" armed
    assert console == []


def test_a_round_that_cannot_be_recorded_is_not_offered_or_started(console, monkeypatch, tmp_path, capsys):
    """The phone's sequence: the model check passed, [1] started the round, and recording failed
    at once. With no recorder the preflight card is not shown and 'now' starts nothing."""
    monkeypatch.setattr(otc, "_ollama_models", lambda: {otc.BENCH_DEFAULT_MODEL})
    monkeypatch.setattr(genome_ledger, "LEDGER_SRC", str(tmp_path / "osiris-repos" / ".osiris" / "ledger.py"))
    for typed in ("/bench", "/bench now", "/bench gemini now"):
        otc._bench(typed)
        out = capsys.readouterr().out
        assert "RECORDER UNAVAILABLE" in out and "Nothing was run or written" in out, typed
        assert "PREFLIGHT" not in out, typed
        assert "1" not in otc._pending_suggestions, typed
    assert console == []


def test_with_the_model_and_a_recorder_the_round_is_offered_then_starts(console, monkeypatch, capsys):
    monkeypatch.setattr(otc, "_ollama_models", lambda: {otc.BENCH_DEFAULT_MODEL})
    monkeypatch.setattr(protege, "load_ledger", lambda bench_dir=None: ([], "no ledger yet"))
    monkeypatch.setattr(otc, "_mentors", lambda *a, **k: None)
    otc._bench("/bench")
    assert "MENTOR ROUND · PREFLIGHT" in capsys.readouterr().out
    assert otc._pending_suggestions["1"] == "/bench now"
    assert console == []
    otc._bench(otc._pending_suggestions["1"])
    assert len(console) == 1 and console[0][2:4] == ["--backend", "ollama"]


def _control_round(model):
    """A complete baseline round /trial accepts as the control for `model`."""
    start = {"kind": "bench_run_start", "run_id": "bench-control", "backend": "ollama", "model": model, "k": 1,
             "planned": sorted(otc.BENCH_QUICK_TASKS), "suite_sha256": protege.current_suite(),
             "runner_sha256": protege.current_runner()}
    end = {"kind": "bench_run_end", "run_id": "bench-control", "status": "complete",
           "summary": {"delivered": 1, "tasks": 3, "false_confidence": 0}}
    return [start, end]


def test_a_trial_round_makes_the_same_checks_as_a_mentor_round(console, monkeypatch, tmp_path, capsys):
    """/trial ... now went straight to the round: with a control on record it started for a model
    that had since been removed, and for a round that could not be recorded."""
    model = otc.BENCH_DEFAULT_MODEL
    monkeypatch.setattr(protege, "load_ledger", lambda bench_dir=None: (_control_round(model), None))
    monkeypatch.setattr(otc, "_trial_result", lambda *a, **k: None)

    monkeypatch.setattr(otc, "_ollama_models", lambda: {"gemma3:1b"})
    for typed in ("/trial", "/trial now", f"/trial invariant-first-v1 {model} now"):
        otc._trial(typed)
        out = capsys.readouterr().out
        assert "MODEL NOT INSTALLED" in out, typed
        assert "1" not in otc._pending_suggestions, typed
    assert console == []

    monkeypatch.setattr(otc, "_ollama_models", lambda: None)
    otc._trial("/trial now")
    assert "BACKEND UNREACHABLE" in capsys.readouterr().out and console == []

    monkeypatch.setattr(otc, "_ollama_models", lambda: {model})
    real = genome_ledger.LEDGER_SRC
    monkeypatch.setattr(genome_ledger, "LEDGER_SRC", str(tmp_path / "nowhere" / "ledger.py"))
    otc._trial("/trial now")
    assert "RECORDER UNAVAILABLE" in capsys.readouterr().out and console == []

    monkeypatch.setattr(genome_ledger, "LEDGER_SRC", real)
    otc._trial("/trial now")                                           # everything in place: it runs
    assert len(console) == 1 and "--strategy" in console[0]


def test_reading_a_trial_needs_no_backend(console, monkeypatch, capsys):
    """'/trial last' and '/trial text' only read; they are not gated on Ollama."""
    monkeypatch.setattr(otc, "_ollama_models", lambda: None)
    monkeypatch.setattr(otc, "_trial_result", lambda *a, **k: print("LAST RESULT"))
    otc._trial("/trial last")
    assert "LAST RESULT" in capsys.readouterr().out
    otc._trial("/trial text")
    assert "BACKEND UNREACHABLE" not in capsys.readouterr().out


def test_the_runner_the_scorecard_compares_by_is_untouched():
    """osiris_bench.py is part of runner_sha256; this fix does not edit it, so recorded rounds stay
    comparable. The pin is the audited one in tests/test_edge_provisioning_and_e0_e3.py."""
    assert osiris_bench.runner_sha256() == "af49b2bd7707c36ff94cc0f87d307846294afefb5a35cb808f8524d41e53264c"
