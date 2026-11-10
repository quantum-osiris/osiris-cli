"""
test_repl_dispatch.py
Automated test suite verifying Living Language Console hotkey dispatch,
bridge functions, and exception resilience in osiris_repl.py and osiris_termux_console.py.

Combines unittest bridge test suite with parametrized console command routing assertions.
"""

import inspect
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from osiris_cli import osiris_repl
from osiris_cli.osiris_repl import dispatch_command, OsirisReplState

otc = osiris_repl.otc


@unittest.skipIf(otc is None, "osiris_termux_console not importable on this platform")
class TestReplDispatchBridge(unittest.TestCase):
    def setUp(self):
        self.state = OsirisReplState()

    def test_otc_bridge_methods_exist(self):
        """Verify all 7 previously missing bridge methods exist and are callable on otc."""
        bridge_methods = [
            '_intent',
            '_runs',
            '_apply',
            '_discard',
            '_organism',
            '_sprint',
            '_nclm',
        ]
        for name in bridge_methods:
            self.assertTrue(hasattr(otc, name), f"Missing method {name} on osiris_termux_console")
            self.assertTrue(callable(getattr(otc, name)), f"Attribute {name} is not callable")

    def test_intent_bridge(self):
        """Test _intent bridge handles status and confidence queries without crashing."""
        with patch.object(otc, '_intent_status') as mock_status:
            otc._intent()
            mock_status.assert_called_once()

        with patch.object(otc, '_intent_status') as mock_status:
            otc._intent("/intent status")
            mock_status.assert_called_once()

    def test_runs_bridge(self):
        """Test _runs bridge differentiates show vs stats."""
        with patch.object(otc, '_runs_stats') as mock_stats, \
             patch.object(otc, '_run_show') as mock_show:
            otc._runs("/runs")
            mock_stats.assert_called_once()
            mock_show.assert_not_called()

        with patch.object(otc, '_runs_stats') as mock_stats, \
             patch.object(otc, '_run_show') as mock_show:
            otc._runs("/run show")
            mock_show.assert_called_once_with("/run show")
            mock_stats.assert_not_called()

    def test_apply_and_discard_bridge(self):
        """Test _apply and _discard call their respective write management functions."""
        with patch.object(otc, '_apply_pending_write') as mock_apply:
            otc._apply()
            mock_apply.assert_called_once()

        with patch.object(otc, '_discard_pending_write') as mock_discard:
            otc._discard()
            mock_discard.assert_called_once()

    def test_organism_bridge(self):
        """Test _organism routes alerts vs status."""
        with patch.object(otc, '_organism_status') as mock_status, \
             patch.object(otc, '_organism_alerts') as mock_alerts:
            otc._organism("/organism")
            mock_status.assert_called_once()
            mock_alerts.assert_not_called()

        with patch.object(otc, '_organism_status') as mock_status, \
             patch.object(otc, '_organism_alerts') as mock_alerts:
            otc._organism("/organism alerts 5")
            mock_alerts.assert_called_once_with(n=5)
            mock_status.assert_not_called()

    def test_sprint_bridge(self):
        """Test _sprint routes plan, execute, and review."""
        with patch.object(otc, '_sprint_plan') as mock_plan:
            otc._sprint("/sprint plan")
            mock_plan.assert_called_once_with("/sprint plan")

        with patch.object(otc, '_sprint_execute') as mock_exec:
            otc._sprint("/sprint execute #1")
            mock_exec.assert_called_once_with("/sprint execute #1")

        with patch.object(otc, '_sprint_review') as mock_review:
            otc._sprint("/sprint review")
            mock_review.assert_called_once()

    def test_nclm_bridge(self):
        """Test _nclm routes to _organism_status."""
        with patch.object(otc, '_organism_status') as mock_status:
            otc._nclm("/nclm")
            mock_status.assert_called_once()

    def test_mentors_signature_tolerance(self):
        """Test _mentors accepts optional argument without TypeError."""
        with patch('protege.load_ledger', return_value=([], None)), \
             patch('protege.candidates', return_value=([], [])):
            try:
                otc._mentors()
                otc._mentors("/mentors")
            except TypeError as te:
                self.fail(f"_mentors raised TypeError: {te}")

    def test_facts_signature_tolerance(self):
        """Test _facts accepts optional argument without TypeError."""
        try:
            otc._facts()
            otc._facts("/facts")
        except TypeError as te:
            self.fail(f"_facts raised TypeError: {te}")

    def test_consensus_signature_tolerance(self):
        """Test _consensus accepts string, integer, or default without TypeError."""
        try:
            otc._consensus()
            otc._consensus(10)
            otc._consensus("/consensus 5")
            otc._consensus("/consensus")
        except TypeError as te:
            self.fail(f"_consensus raised TypeError: {te}")

    def test_hotkey_b_intent_status_dispatch(self):
        """Direct regression test for hotkey B crash."""
        otc._pending_suggestions['B'] = '/intent status'
        with patch.object(otc, '_intent_status') as mock_intent:
            dispatch_command(self.state, 'B')
            mock_intent.assert_called_once()

    def test_all_home_letter_hotkeys_dispatch(self):
        """Verify all letter hotkeys registered in _home_letters() dispatch without exceptions."""
        letters = otc._home_letters()
        for key, label, cmd in letters:
            otc._pending_suggestions[key] = cmd
            with patch.object(otc, '_gap', return_value=None), \
                 patch.object(otc, '_experiment', return_value=None), \
                 patch.object(otc, '_bench', return_value=None), \
                 patch.object(otc, '_ask', return_value="0"):
                try:
                    dispatch_command(self.state, key)
                except Exception as e:
                    self.fail(f"Hotkey [{key}] -> '{cmd}' raised exception: {type(e).__name__}: {e}")

    def test_repl_fault_isolation(self):
        """Verify that an unexpected exception in otc does not crash dispatch_command."""
        with patch.object(otc, '_intent_status', side_effect=RuntimeError("simulated unexpected failure")):
            try:
                dispatch_command(self.state, "/intent status")
            except Exception as e:
                self.fail(f"dispatch_command failed to isolate exception: {e}")


# ---------------------------------------------------------------------------
# Pytest-compatible parametrized console routing tests
# ---------------------------------------------------------------------------
try:
    import pytest
except ImportError:
    pytest = None

if pytest is not None:
    pytestmark = pytest.mark.skipif(otc is None, reason="osiris_termux_console not importable")

    REPL_SRC = Path(osiris_repl.__file__).read_text(encoding="utf-8")

    def test_every_console_call_in_the_repl_exists_and_binds():
        for name, args in sorted(set(re.findall(r"otc\.(\w+)\(([^)]*)\)", REPL_SRC))):
            fn = getattr(otc, name, None)
            assert fn is not None, f"otc.{name} does not exist"
            if "=" not in args:
                n = len([a for a in args.split(",") if a.strip()])
                inspect.signature(fn).bind(*([None] * n))

    @pytest.fixture
    def calls(monkeypatch):
        seen = []

        def rec(name):
            return lambda *a, **k: seen.append((name, a))

        for name in ("_mentors", "_experiment", "_experiments", "_gap", "_gaps", "_facts", "_consensus",
                     "_plan", "_runs_stats", "_run_show", "_organism_status", "_intent_status",
                     "_sprint_review", "_home", "_unknown_command", "_apply_pending_write",
                     "_discard_pending_write"):
            monkeypatch.setattr(otc, name, rec(name))
        monkeypatch.setattr(otc, "_pending_suggestions", {})
        monkeypatch.setattr(otc, "_home_letters", lambda pending=None: [
            ("C", "Engine 3 (NCLM): training status", "/organism status"),
            ("F", "Mentors: which model the evidence favors", "/mentors"),
            ("J", "Draft a quantum / research experiment brief", "/experiment")])
        monkeypatch.setattr(osiris_repl, "handle_livlm_prompt", lambda state, text: seen.append(("model", text)))
        monkeypatch.setattr(osiris_repl, "execute_osiris_status", lambda: seen.append(("nclm_status", ())))
        return seen

    def run(line):
        osiris_repl.dispatch_command(osiris_repl.SESSION, line)

    @pytest.mark.parametrize("line, expected", [
        ("f", ("_mentors", ())),
        ("/mentors", ("_mentors", ())),
        ("j", ("_experiment", ("",))),
        ("/experiment does DD help at 16 qubits", ("_experiment", (" does DD help at 16 qubits",))),
        ("/experiments", ("_experiments", ())),
        ("/gap add a quarantine queue", ("_gap", (" add a quarantine queue",))),
        ("/gaps", ("_gaps", ("/gaps",))),
        ("/facts", ("_facts", ())),
        ("/consensus", ("_consensus", ())),
        ("/plan a bell test", ("_plan", (" a bell test",))),
        ("/runs", ("_runs_stats", ())),
        ("/run show 3", ("_run_show", ("/run show 3",))),
        ("c", ("_organism_status", ())),
        ("/intent", ("_intent_status", ())),
        ("/nclm", ("nclm_status", ())),
        ("/sprint review", ("_sprint_review", ())),
    ])
    def test_console_commands_reach_the_right_function_with_the_right_argument(calls, line, expected):
        run(line)
        assert calls == [expected]

    def test_apply_and_discard_with_nothing_pending_say_so(calls, monkeypatch, capsys):
        monkeypatch.setattr(otc, "_pending_write", {"path": None})
        run("/apply")
        run("/discard")
        assert calls == [] and capsys.readouterr().out.count("nothing pending") == 2

    def test_keys_and_typos_never_reach_the_model(calls, capsys):
        run("q")            # a letter no menu uses
        run("7")
        run("/stauts")
        assert ("model", "q") not in calls and ("model", "7") not in calls
        assert calls == [("_unknown_command", ("/stauts",))]
        assert capsys.readouterr().out.count("No menu item") == 2

    def test_a_bare_command_word_is_not_sent_to_the_model(calls, capsys):
        run("code")
        run("status")
        run("publish")
        out = capsys.readouterr().out
        assert not [c for c in calls if c[0] == "model"]
        assert "looks like the command /code" in out and "/code explain" in out
        assert "[1] Run /status (read-only)" in out
        assert "Type /publish to run it" in out and "[1] Run /publish" not in out
        assert otc._pending_suggestions == {"1": "/status"}          # publish is never offered

    def test_sentences_and_unknown_words_still_go_to_osiris(calls):
        run("code review my parser please")
        run("hello")
        assert calls == [("model", "code review my parser please"), ("model", "hello")]

    def test_repl_commands_set_matches_the_dispatch_source():
        found = set(re.findall(r'cmd (?:==|in) \(?"(/[a-z0-9_-]+)"', REPL_SRC))
        assert found <= osiris_repl.REPL_COMMANDS | osiris_repl.CONSOLE_COMMANDS, \
            found - osiris_repl.REPL_COMMANDS - osiris_repl.CONSOLE_COMMANDS

    def test_charger_note_only_on_the_phone(monkeypatch):
        import osiris.substrate as sub
        monkeypatch.setattr(sub, "node_type", lambda: "NODE_BETA_COMPUTE")
        assert otc._charger_note() == []
        monkeypatch.setattr(sub, "node_type", lambda: "NODE_ALPHA_MOBILE")
        assert "charger" in " ".join(otc._charger_note())

    def test_plain_text_still_goes_to_osiris(calls):
        run("hello osiris")
        run("/home/enki/log.txt line 1\nline 2")          # a pasted log is conversation
        assert calls == [("model", "hello osiris"), ("model", "/home/enki/log.txt line 1\nline 2")]

    def test_a_suggestion_carrying_the_self_modify_override_is_not_run(calls, monkeypatch, capsys):
        monkeypatch.setattr(otc, "_pending_suggestions", {"1": "/sprint execute --allow-self-modify"})
        run("1")
        assert calls == [] and "only counts when you type it yourself" in capsys.readouterr().out


if __name__ == '__main__':
    unittest.main()
