"""Wave 5 Verification Suite:
1. Benchmark -> Calibration ledger auto-sync (`sync_from_bench_candidates`, `sync_bench_ledger`,
   `/calibration sync`, and `_mentors()` integration) with idempotency, integrity-drop filtering,
   exact `0.666667` Brier/ECE on the 2/3 false-confidence round, and `exec-canon-1` zero-float
   JSONL verification.
2. L4/L5 False-Confidence Hardening in `osiris_cli/production_synth.py`:
   - Catches `self.stories = {}` (`KeyError: -1` negative list indexing regression) on `Backlog`.
   - Catches unguarded `int(token)` on mixed prose `'parity telemetry reached 0.9728'` on `claims`.
   - Catches `.{0,160}` regex length cap & mid-sentence decimal `0.97` splitting on `gaps`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import osiris_termux_console as otc
from osiris_cli import calibration, probes, production_synth


def _run_combined_in_temp(tmp_path: Path, mod_name: str, mod_src: str, combined_test_src: str) -> tuple[bool, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / f"{mod_name}.py").write_text(mod_src, encoding="utf-8")
    test_file = tmp_path / "test_cand.py"
    test_file.write_text(
        combined_test_src + "\nif __name__ == '__main__':\n    test_proposal()\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(test_file)],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=15,
    )
    return proc.returncode == 0, (proc.stdout + "\n" + proc.stderr)


def test_bench_candidates_sync_to_calibration_idempotently_and_skip_integrity_drops(tmp_path: Path) -> None:
    cal_path = tmp_path / "benchmarks" / "calibration.jsonl"
    # Mirrors the exact Termux qwen2.5-coder:7b round (2 own-pass/hidden-fail + 1 own-fail/hidden-fail)
    # plus 1 unreachable integrity drop that must NOT be scored.
    cands = [
        {
            "entry": "aaa111" * 10 + "abcd",
            "run": "bench-20261121T084500Z-termux",
            "task": "backlog",
            "mentor": "ollama/qwen2.5-coder:7b",
            "n": 1,
            "own_ok": True,
            "hidden_ok": False,
            "module_sha256": "1" * 64,
            "integrity": False,
        },
        {
            "entry": "bbb222" * 10 + "abcd",
            "run": "bench-20261121T084500Z-termux",
            "task": "claims",
            "mentor": "ollama/qwen2.5-coder:7b",
            "n": 1,
            "own_ok": True,
            "hidden_ok": False,
            "module_sha256": "2" * 64,
            "integrity": False,
        },
        {
            "entry": "ccc333" * 10 + "abcd",
            "run": "bench-20261121T084500Z-termux",
            "task": "gaps",
            "mentor": "ollama/qwen2.5-coder:7b",
            "n": 1,
            "own_ok": False,
            "hidden_ok": False,
            "module_sha256": "3" * 64,
            "integrity": False,
        },
        {
            "entry": "ddd444" * 10 + "abcd",
            "run": "bench-20261121T084500Z-termux",
            "task": "dropped_task",
            "mentor": "ollama/qwen2.5-coder:7b",
            "n": 1,
            "own_ok": False,
            "hidden_ok": False,
            "module_sha256": None,
            "integrity": True,
        },
    ]

    added_first = calibration.sync_from_bench_candidates(cands, path=cal_path)
    assert added_first == 3
    # Second sync must be 100% idempotent (0 duplicate observations)
    added_second = calibration.sync_from_bench_candidates(cands, path=cal_path)
    assert added_second == 0

    led = calibration.CalibrationLedger(cal_path)
    obs = led.observations()
    assert len(obs) == 3
    rep = calibration.compute_calibration(obs)
    assert rep.brier_score_str == "0.666667"
    assert rep.ece_str == "0.666667"
    assert rep.overconfidence_str == "0.666667"

    summary = calibration.format_summary(cal_path)
    assert "Brier=0.666667" in summary and "ECE=0.666667" in summary
    assert all(len(ln) <= 96 for ln in summary.splitlines())

    # Verify strict exec-canon-1 zero-float invariant
    for raw_line in cal_path.read_text(encoding="utf-8").splitlines():
        parsed = json.loads(
            raw_line,
            parse_float=lambda s: (_ for _ in ()).throw(AssertionError(f"float in calibration JSONL: {s}")),
        )
        assert parsed["canon_version"] == "exec-canon-1"
    assert probes.verify_chain_tail(cal_path).intact


def test_mentors_auto_syncs_bench_candidates_into_calibration(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("OSIRIS_HOME", str(tmp_path))
    fake_entries = [{"kind": "bench_run_start", "run_id": "r1"}]
    fake_cands = [
        {
            "entry": "e" * 64,
            "run": "r1",
            "task": "backlog",
            "mentor": "ollama/qwen2.5-coder:7b",
            "n": 1,
            "own_ok": True,
            "hidden_ok": True,
            "module_sha256": "a" * 64,
            "no_output": False,
            "integrity": False,
        }
    ]
    with patch("protege.load_ledger", return_value=(fake_entries, None)), \
         patch("protege.candidates", return_value=(fake_cands, 0)):
        otc._mentors()
    out = capsys.readouterr().out
    assert "Calibration (1 scored" in out
    assert "Brier=0.000000" in out


def test_l5_critic_catches_backlog_dict_keyerror_minus_one_false_confidence(tmp_path: Path) -> None:
    existing = (
        "class Backlog:\n"
        "    def __init__(self):\n"
        "        self.stories = []\n"
        "        self.next_id = 1\n"
        "    def add(self, text, source='manual'):\n"
        "        s = {'id': self.next_id, 'text': text, 'source': source, 'status': 'backlog'}\n"
        "        self.stories.append(s)\n"
        "        self.next_id += 1\n"
        "        return s['id']\n"
    )
    # Candidate uses a dict {1: ...} (or custom dict subclass) where stories[-1] raises KeyError: -1,
    # while passing its own naive test!
    buggy_cand = (
        "class _StoryMap(dict):\n"
        "    pass\n"
        "class Backlog:\n"
        "    def __init__(self):\n"
        "        self.stories = _StoryMap()\n"
        "        self.next_id = 1\n"
        "    def add(self, text, source='manual'):\n"
        "        for s in self.stories.values():\n"
        "            if s['text'] == text:\n"
        "                return s['id']\n"
        "        sid = self.next_id\n"
        "        self.stories[sid] = {'id': sid, 'text': text, 'source': source, 'status': 'backlog'}\n"
        "        self.next_id += 1\n"
        "        return sid\n"
    )
    own_test = (
        "import backlog\n"
        "def test_proposal():\n"
        "    b = backlog.Backlog()\n"
        "    assert b.add('x') == b.add('x')\n"
    )
    contract = production_synth.extract_module_contract(existing)
    critic = production_synth.synthesize_independent_critic_tests(
        "backlog",
        "Make add() idempotent for duplicate open stories and preserve existing behavior.",
        contract=contract,
        existing_source=existing,
    )
    combined, err = production_synth.combine_verification_tests(own_test, critic)
    assert err is None and combined is not None
    ok, output = _run_combined_in_temp(tmp_path / "b1", "backlog", buggy_cand, combined)
    assert ok is False, f"Buggy dict-backed Backlog should fail L5 critic, got: {output}"


def test_l5_critic_catches_claims_int_parity_parse_crash_false_confidence(tmp_path: Path) -> None:
    existing = (
        "def check_claim(text, evidence):\n"
        "    problems = []\n"
        "    if not evidence or not evidence.strip():\n"
        "        problems.append('no evidence cited')\n"
        "    return problems\n"
    )
    # Candidate splits tokens and calls int() on any token containing digits or crashes on 'parity'
    buggy_claims = (
        "def check_claim(text, evidence):\n"
        "    if not evidence.strip():\n"
        "        return ['no evidence cited']\n"
        "    ev_nums = [w.rstrip('.') for w in evidence.split()]\n"
        "    out = []\n"
        "    for tok in text.split():\n"
        "        if tok == 'parity':\n"
        "            int(tok)\n"
        "        if any(ch.isdigit() for ch in tok) and tok not in ev_nums:\n"
        "            out.append(f'unsourced {tok}')\n"
        "    return out\n"
    )
    own_test = (
        "import claims\n"
        "def test_proposal():\n"
        "    assert claims.check_claim('0.9728', '0.9722') != []\n"
    )
    contract = production_synth.extract_module_contract(existing)
    critic = production_synth.synthesize_independent_critic_tests(
        "claims",
        "Catch made-up unsourced numbers in claims compared against cited evidence.",
        contract=contract,
        existing_source=existing,
    )
    combined, err = production_synth.combine_verification_tests(own_test, critic)
    assert err is None and combined is not None
    ok, _out = _run_combined_in_temp(tmp_path / "c1", "claims", buggy_claims, combined)
    assert ok is False


def test_l5_critic_and_contract_gate_catch_truncated_or_decimal_split_gaps(tmp_path: Path) -> None:
    existing = (
        "import re\n"
        "_GAP_RE = re.compile(r'capability gap[:\\s]*(.{0,160})', re.IGNORECASE)\n"
        "def extract_gaps(prompts):\n"
        "    return [m.group(1).strip() for t in prompts for m in [_GAP_RE.search(t)] if m]\n"
    )
    contract = production_synth.extract_module_contract(existing)
    assert contract.bounded_regex_caps == (".{0,160}",)

    still_capped = existing.replace("160", "200")
    # Both .{0,200} truncation AND naive '.' decimal splitting pass their own naive test,
    # and BOTH are defeated deterministically by the Independent L5 Critic!
    decimal_broken = (
        "import re\n"
        "def extract_gaps(prompts):\n"
        "    out = []\n"
        "    for p in prompts:\n"
        "        m = re.search(r'capability gap[:\\s]*(.*)', p, re.I)\n"
        "        if m:\n"
        "            out.append(m.group(1).split('.')[0].strip())\n"
        "    return out\n"
    )
    own_test = (
        "import gaps\n"
        "def test_proposal():\n"
        "    assert gaps.extract_gaps(['capability gap: simple item. Next.']) == ['simple item']\n"
    )
    critic = production_synth.synthesize_independent_critic_tests(
        "gaps",
        "Extract capability gap descriptions whole without truncating long lines or splitting decimals.",
        contract=contract,
        existing_source=existing,
    )
    combined, err = production_synth.combine_verification_tests(own_test, critic)
    assert err is None and combined is not None
    ok_cap, out_cap = _run_combined_in_temp(tmp_path / "g0", "gaps", still_capped, combined)
    assert ok_cap is False and "AssertionError" in out_cap
    ok, out_dec = _run_combined_in_temp(tmp_path / "g1", "gaps", decimal_broken, combined)
    assert ok is False and "AssertionError" in out_dec

    # 3. Clean reference implementation passes both static contract gate and L5 critic!
    good_gaps = (
        "import re\n"
        "_GAP_RE = re.compile(r'capability gap[:\\s]*([^\\n]+)', re.I)\n"
        "_SENT_END = re.compile(r'\\.(?=\\s+[A-Z]|\\s*$)')\n"
        "def extract_gaps(prompts):\n"
        "    out = []\n"
        "    for p in prompts:\n"
        "        m = _GAP_RE.search(p)\n"
        "        if m:\n"
        "            seg = _SENT_END.split(m.group(1), maxsplit=1)[0].strip()\n"
        "            if seg:\n"
        "                out.append(seg)\n"
        "    return out\n"
    )
    assert production_synth.verify_contract_preservation(contract, good_gaps) == []
    ok_good, out_good = _run_combined_in_temp(tmp_path / "g2", "gaps", good_gaps, combined)
    assert ok_good is True, out_good
