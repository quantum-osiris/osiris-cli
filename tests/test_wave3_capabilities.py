"""Wave 3 regression & capability suite covering L4 multi-candidate & Plan v1 gates:

  * [G-12 / REQ-PLAN-1] Tokenized argv + cwd on Plan Artifact v1; raw command strings
                        and shell operators deterministically refused.
  * [G-10 / REQ-PLAN-2] Four named, measurable change-size gates (max_files_touched,
                        max_lines_changed, protected_paths_matched, highest_tier_allowed),
                        each independently tested to prove it fires when exceeded.
  * [G-11 / REQ-PLAN-3] Zero quarantined CRSM constants (theta_lock=51.843, 0.7734,
                        0.97871, 2.176435e-08) & zero import-time side effects.
  * [L4 Propose]        Baseline, alternative, and conservative candidates evaluated
                        side by side through identical SYS-025 AST firewall + payload_gate
                        + check_test_shape + exec_kernel sandboxed pytest, rendered in a
                        compact <100-col (and 72-col Node Alpha) comparison table before /apply.
"""

from __future__ import annotations

import ast
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

import osiris_ui
import payload_gate
from osiris_cli import architecture, contracts_validator, multi_candidate, plan_gates, probes, quarantine
from osiris_cli.exec_kernel import ExecJournal


# ── Hygiene: Python 3.9 AST, payload_gate zero-stub, zero import side effects ─

@pytest.mark.parametrize(
    "rel_path",
    [
        "osiris_cli/plan_gates.py",
        "osiris_cli/multi_candidate.py",
    ],
)
def test_wave3_modules_pass_py39_ast_and_zero_stub_payload_gate(rel_path: str) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src = (repo_root / rel_path).read_text(encoding="utf-8")
    assert "\r" not in src, f"CRLF found in {rel_path}"
    ast.parse(src, filename=rel_path, feature_version=(3, 9))
    stub_err = payload_gate.check_payload(src)
    assert stub_err is None, f"{rel_path} failed payload_gate: {stub_err}"


def test_g11_import_has_zero_filesystem_side_effects(tmp_path: Path) -> None:
    fake_home = tmp_path / "isolated_home"
    fake_home.mkdir()
    repo_root = Path(__file__).resolve().parent.parent
    env = dict(os.environ, HOME=str(fake_home), OSIRIS_HOME=str(fake_home / ".osiris"), PYTHONPATH=str(repo_root))
    proc = subprocess.run(
        [sys.executable, "-c", "import osiris_cli.plan_gates, osiris_cli.multi_candidate"],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.returncode == 0
    assert list(fake_home.iterdir()) == [], f"Import created unexpected state in HOME: {list(fake_home.iterdir())}"


# ── G-12 / REQ-PLAN-1: Tokenized argv + cwd; raw command strings refused ──────

@pytest.mark.parametrize(
    "bad_step",
    [
        {"id": "s1", "type": "run_command", "command": "pytest -q", "cwd": "."},
        {"id": "s1", "type": "run_command", "argv": "pytest -q", "cwd": "."},
        {"id": "s1", "type": "run_command", "argv": [], "cwd": "."},
        {"id": "s1", "type": "run_command", "argv": ["pytest", ";", "rm", "-rf", "."], "cwd": "."},
        {"id": "s1", "type": "run_command", "argv": ["sh", "-c", "$(id)"], "cwd": "."},
        {"id": "s1", "type": "run_command", "argv": ["pytest", "`whoami`"], "cwd": "."},
        {"id": "s1", "type": "run_command", "argv": ["pytest", "-q"], "cwd": "../../etc"},
    ],
)
def test_g12_req_plan1_refuses_raw_command_shell_operators_and_escaping_cwd(
    bad_step: dict, tmp_path: Path
) -> None:
    with pytest.raises(plan_gates.PlanValidationError):
        plan_gates.build_plan_v1(intent="run tests", steps=[bad_step], root=tmp_path)


def test_g12_valid_plan_v1_passes_canonical_contract_and_records_receipt(tmp_path: Path) -> None:
    steps = [
        {"id": "s1", "type": "edit_file", "path": "pkg/math_ops.py", "lines_changed": 24},
        plan_gates.default_pytest_verify_step(["tests/test_math_ops.py"], cwd="."),
    ]
    plan = plan_gates.build_plan_v1(
        intent="Add clamped division helper",
        steps=steps,
        preconditions=["clean checkout"],
        postconditions=["pytest green"],
        root=tmp_path,
    )
    ok, refusal = contracts_validator.validate_contract(
        "plan_v1.json", plan, require_execution_mode_on_plan=True
    )
    assert ok is True and refusal is None
    assert plan["gates"]["passed"] is True
    assert len(plan["gates"]["checks"]) == 4

    audit, rhash = plan_gates.record_plan_receipt(plan, osiris_home=tmp_path, root=tmp_path)
    assert audit.valid is True and audit.gates_passed is True and len(rhash) == 64
    receipt_path = plan_gates.default_plan_receipts_path(tmp_path)
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
    assert probes.verify_chain_tail(receipt_path).intact


# ── G-10 / REQ-PLAN-2: Prove EACH of the 4 measurable change-size gates fires ─

def test_g10_gate1_max_files_touched_fires_when_exceeded(tmp_path: Path) -> None:
    steps = [
        {"id": f"s{i}", "type": "edit_file", "path": f"pkg/mod_{i}.py", "lines_changed": 5}
        for i in range(4)
    ]
    thresholds = plan_gates.PlanGateThresholds(max_files_touched=3)
    gates = plan_gates.evaluate_plan_gates(steps, thresholds=thresholds, root=tmp_path)
    assert gates["passed"] is False
    by_name = {c["name"]: c for c in gates["checks"]}
    assert by_name["max_files_touched"]["passed"] is False
    assert by_name["max_files_touched"]["measured"] == 4
    assert by_name["max_lines_changed"]["passed"] is True
    assert by_name["protected_paths_matched"]["passed"] is True
    assert by_name["highest_tier_allowed"]["passed"] is True


def test_g10_gate2_max_lines_changed_fires_when_exceeded(tmp_path: Path) -> None:
    steps = [{"id": "s1", "type": "edit_file", "path": "pkg/big.py", "lines_changed": 120}]
    thresholds = plan_gates.PlanGateThresholds(max_lines_changed=100)
    gates = plan_gates.evaluate_plan_gates(steps, thresholds=thresholds, root=tmp_path)
    assert gates["passed"] is False
    by_name = {c["name"]: c for c in gates["checks"]}
    assert by_name["max_lines_changed"]["passed"] is False
    assert by_name["max_lines_changed"]["measured"] == 120
    assert by_name["max_files_touched"]["passed"] is True


@pytest.mark.parametrize(
    "protected_path",
    [
        "osiris_cli/policy/exec_policy.json",
        "governance/allowed_signers",
        "GEMINI.md",
        "AGENTS.md",
        "contracts/plan_v1.json",
        ".osiris_compliance/audit.json",
    ],
)
def test_g10_gate3_protected_paths_matched_fires_on_governance_targets(
    protected_path: str, tmp_path: Path
) -> None:
    steps = [{"id": "s1", "type": "edit_file", "path": protected_path, "lines_changed": 2}]
    gates = plan_gates.evaluate_plan_gates(steps, root=tmp_path)
    assert gates["passed"] is False
    by_name = {c["name"]: c for c in gates["checks"]}
    assert by_name["protected_paths_matched"]["passed"] is False
    assert by_name["protected_paths_matched"]["measured"] == 1


def test_g10_gate4_highest_tier_allowed_fires_on_network_or_forbidden_argv(tmp_path: Path) -> None:
    steps = [
        {
            "id": "s-net",
            "type": "run_command",
            "argv": ["curl", "https://example.com/payload.sh"],
            "cwd": ".",
        }
    ]
    thresholds = plan_gates.PlanGateThresholds(highest_tier_allowed="MUTATE")
    gates = plan_gates.evaluate_plan_gates(steps, thresholds=thresholds, root=tmp_path)
    assert gates["passed"] is False
    by_name = {c["name"]: c for c in gates["checks"]}
    assert by_name["highest_tier_allowed"]["passed"] is False
    assert by_name["highest_tier_allowed"]["measured"] in ("NETWORK", "FORBIDDEN")


# ── G-11 / REQ-PLAN-3: Quarantined constants refused anywhere in a plan ───────

@pytest.mark.parametrize(
    "bad_note",
    [
        "Require theta_lock = 51.843 resonance gate",
        "Enforce PHI_THRESHOLD >= 0.7734 before apply",
        "Bound fidelity at F_max = 0.97871",
        "Scale timeout by Lambda_Phi = 2.176435e-08",
    ],
)
def test_g11_req_plan3_refuses_quarantined_constants_in_plan(bad_note: str, tmp_path: Path) -> None:
    steps = [{"id": "s1", "type": "edit_file", "path": "pkg/a.py", "lines_changed": 4}]
    with pytest.raises(plan_gates.PlanValidationError, match="REQ-PLAN-3"):
        plan_gates.build_plan_v1(
            intent="update module",
            steps=steps,
            postconditions=[bad_note],
            root=tmp_path,
        )


# ── [L4 Propose] Multi-candidate pipeline (baseline, alternative, conservative) ─

def test_l4_multi_candidate_verifies_three_strategies_blocks_bad_ast_or_stubs_and_renders_narrow_table(
    tmp_path: Path,
) -> None:
    ws = tmp_path / "repo"
    ws.mkdir()
    (ws / "stats_util.py").write_text(
        "def clamped_ratio(num: float, den: float) -> float:\n"
        "    return num / den\n",
        encoding="utf-8",
    )

    # Case A: baseline & conservative pass; alternative smuggles subprocess.run (caught by AST firewall!)
    payloads_mixed = {
        "baseline": (
            "def clamped_ratio(num: float, den: float) -> float:\n"
            "    if den == 0:\n"
            "        return 0.0\n"
            "    val = num / den\n"
            "    if val < 0.0:\n"
            "        return 0.0\n"
            "    if val > 1.0:\n"
            "        return 1.0\n"
            "    return val\n",
            "from stats_util import clamped_ratio\n\n"
            "def test_clamped_ratio_bounds_and_zero_den():\n"
            "    assert clamped_ratio(1, 2) == 0.5\n"
            "    assert clamped_ratio(5, 0) == 0.0\n"
            "    assert clamped_ratio(9, 2) == 1.0\n"
            "    assert clamped_ratio(-3, 2) == 0.0\n",
        ),
        "alternative": (
            "from subprocess import run as _sneaky\n"
            "def clamped_ratio(num: float, den: float) -> float:\n"
            "    _sneaky(['echo', 'hi'])\n"
            "    return 0.0 if den == 0 else min(1.0, max(0.0, num / den))\n",
            "from stats_util import clamped_ratio\n\n"
            "def test_alt():\n"
            "    assert clamped_ratio(1, 2) == 0.5\n",
        ),
        "conservative": (
            "def clamped_ratio(num: float, den: float) -> float:\n"
            "    return 0.0 if den == 0 else max(0.0, min(1.0, num / den))\n",
            "from stats_util import clamped_ratio\n\n"
            "def test_conservative_clamp():\n"
            "    assert clamped_ratio(2, 4) == 0.5\n"
            "    assert clamped_ratio(1, 0) == 0.0\n"
            "    assert clamped_ratio(10, 3) == 1.0\n",
        ),
    }

    journal = ExecJournal(tmp_path / "exec_journal.jsonl")
    bundle = multi_candidate.propose_three_candidates(
        root=ws,
        target_rel_path="stats_util.py",
        goal="Guard zero denominator and clamp output into [0.0, 1.0]",
        strategy_payloads=payloads_mixed,
        osiris_home=tmp_path / "osiris_home",
        exec_journal=journal,
        exec_artifacts_dir=tmp_path / "artifacts",
    )

    base_ev = bundle.by_strategy("baseline")
    alt_ev = bundle.by_strategy("alternative")
    cons_ev = bundle.by_strategy("conservative")
    assert base_ev is not None and base_ev.verified is True and base_ev.exec_receipt_id
    assert alt_ev is not None and alt_ev.verified is False and alt_ev.stage_failed == "ast_firewall"
    assert cons_ev is not None and cons_ev.verified is True and cons_ev.lines_changed < base_ev.lines_changed
    assert bundle.selected_strategy == "baseline"

    # Verify compact side-by-side table fits both 88-col and 72-col Node Alpha mobile viewports!
    table_88 = multi_candidate.render_side_by_side(bundle, width=88)
    assert "baseline" in table_88 and "alternative" in table_88 and "conservative" in table_88
    assert "FAIL:ast_firewall" in table_88 or "FAIL:ast_f" in table_88
    assert all(len(line) <= 88 for line in table_88.splitlines())

    table_72 = multi_candidate.render_side_by_side(bundle, width=72)
    assert all(len(line) <= 72 for line in table_72.splitlines())

    # Operator cannot switch to unverified 'alternative', CAN switch to verified 'conservative'
    multi_candidate.set_active_bundle(bundle)
    refused_msg = multi_candidate.run_command("select alternative")
    assert "Selection refused" in refused_msg

    staged = []
    switched_out = multi_candidate.run_command(
        "select conservative", on_select=lambda ev: staged.append(ev)
    )
    assert "Selected for /apply: conservative" in switched_out
    assert len(staged) == 1 and staged[0].strategy == "conservative"

    # Receipt log is mode 0600, exec-canon-1, and passes verify_chain_tail in O(1)
    mc_log = multi_candidate.default_multi_candidate_log(tmp_path / "osiris_home")
    assert stat.S_IMODE(mc_log.stat().st_mode) == 0o600
    assert probes.verify_chain_tail(mc_log).intact
    ok_j, cnt_j, _ = journal.verify()
    assert ok_j is True and cnt_j == 2  # baseline + conservative ran pytest; alternative blocked pre-exec!


def test_l4_multi_candidate_rejects_placeholder_stubs_and_assert_true_tests(tmp_path: Path) -> None:
    ws = tmp_path / "repo"
    ws.mkdir()
    payloads_bad = {
        "baseline": (
            "def solve(x: int) -> int:\n    pass\n",
            "from mod import solve\ndef test_s():\n    assert solve(1) == 1\n",
        ),
        "alternative": (
            "def solve(x: int) -> int:\n    return x * 2\n",
            "from mod import solve\ndef test_trivial():\n    _ = solve\n    assert True\n",
        ),
        "conservative": (
            "def solve(x: int) -> int:\n    return x + 1\n",
            "from mod import solve\ndef test_wrong():\n    assert solve(2) == 99\n",
        ),
    }
    bundle = multi_candidate.propose_three_candidates(
        root=ws,
        target_rel_path="mod.py",
        goal="Implement solve(x)",
        strategy_payloads=payloads_bad,
        osiris_home=tmp_path / "osiris_home",
        exec_journal=ExecJournal(tmp_path / "ej.jsonl"),
        exec_artifacts_dir=tmp_path / "art",
    )
    assert bundle.selected_strategy is None
    assert bundle.by_strategy("baseline").stage_failed == "placeholder_gate"
    assert bundle.by_strategy("alternative").stage_failed == "test_shape"
    assert bundle.by_strategy("conservative").stage_failed == "pytest"


def test_l4_multi_candidate_respects_quarantine_gate(tmp_path: Path) -> None:
    ws = tmp_path / "repo"
    ws.mkdir()
    osiris_home = tmp_path / "ohome"
    mod_code = "def add_one(x: int) -> int:\n    return x + 1\n"
    tst_code = "from adder import add_one\ndef test_add_one():\n    assert add_one(3) == 4\n"

    # Quarantine baseline's exact module digest ahead of time
    import hashlib
    q = quarantine.QuarantineQueue(quarantine.default_quarantine_path(osiris_home))
    q.enqueue("cand_blocked", hashlib.sha256(mod_code.encode("utf-8")).hexdigest(), ["under manual review"])

    alt_code = "def add_one(x: int) -> int:\n    val = int(x)\n    return val + 1\n"
    bundle = multi_candidate.propose_three_candidates(
        root=ws,
        target_rel_path="adder.py",
        goal="Implement add_one",
        strategy_payloads={
            "baseline": (mod_code, tst_code),
            "alternative": (alt_code, tst_code),
            "conservative": (mod_code, tst_code),
        },
        osiris_home=osiris_home,
        exec_journal=ExecJournal(tmp_path / "ej.jsonl"),
        exec_artifacts_dir=tmp_path / "art",
    )
    # Baseline & conservative match quarantined digest -> recommendation falls back to verified 'alternative'!
    assert bundle.selected_strategy == "alternative"


# ── [/architecture] All 9 pipeline layers L0-L8 now have 0 missing items! ─────

def test_wave3_architecture_marks_all_nine_layers_l0_through_l8_built() -> None:
    for ly in architecture.LAYERS:
        assert ly.missing == (), f"Layer {ly.key} ({ly.name}) still has missing items: {ly.missing}"
    # Only the 3 Wave 4 Physics & Quantum research items remain in SUBSYSTEMS
    remaining = [item for _name, _built, missing in architecture.SUBSYSTEMS for item in missing]
    assert len(remaining) == 3
    state = architecture.live_state(None, None)
    rendered = architecture.full(osiris_ui.Canvas("plain", 80), state)
    assert "osiris_cli/multi_candidate.py" in rendered
