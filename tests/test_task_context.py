"""L0: a run ID per task carrying its permissions and budget, and the ledger it is recorded in.

Pure and hermetic: every path is tmp_path, the clock is injected, no model is called.
"""

import json
import math

import pytest

from osiris_cli import task_context as tc
from osiris_cli.probes import verify_chain


def ctx(text="fix the off-by-one in backlog.py", **kw):
    kw.setdefault("kind", "code")
    kw.setdefault("perms", tc.permissions(propose_writes=True))
    kw.setdefault("exchange_log", None)
    return tc.new_context(text, **kw)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


# ── the context ──────────────────────────────────────────────────────────────────────────────

def test_a_context_states_who_what_and_how_much():
    c = ctx(run_id="code-abc123", max_writes=4)
    assert c.run_id == "code-abc123" and c.kind == "code"
    assert c.text_sha256 == tc.sha256_text("fix the off-by-one in backlog.py")
    assert (c.intent_class, c.risk_tier, c.cost_class) == ("engineering", "MEDIUM", "workspace_write")
    assert c.permissions.tools == ("grep_code", "list_dir", "read_file", "write_file")
    assert c.permissions.propose_writes and not c.permissions.run_commands
    assert c.budget.max_steps == 8 and c.budget.max_writes == 4
    assert len(c.context_sha256) == 64
    assert "code-abc123" in c.summary() and "risk MEDIUM" in c.summary()


def test_a_context_is_frozen():
    c = ctx()
    with pytest.raises(Exception):
        c.risk_tier = "LOW"
    with pytest.raises(Exception):
        c.budget.max_steps = 99
    with pytest.raises(Exception):
        c.permissions.run_commands = True


def test_the_operators_words_are_recorded_as_a_hash_not_as_text():
    c = ctx("rotate the signing key in the enclave to hunter2")
    assert "hunter2" not in json.dumps(c.to_dict())


def test_a_recorded_context_round_trips_and_a_changed_one_is_refused():
    c = ctx()
    assert tc.TaskRunContext.from_dict(c.to_dict()) == c
    for mutate in (
        lambda d: d["budget"].__setitem__("max_steps", 999),
        lambda d: d["permissions"].__setitem__("run_commands", True),
        lambda d: d["permissions"]["tools"].append("run_command"),
        lambda d: d.__setitem__("risk_tier", "LOW"),
        lambda d: d.__setitem__("run_id", "code-other"),
        lambda d: d.__setitem__("text_sha256", "0" * 64),
        lambda d: d["intake"].__setitem__("exchange_head", "f" * 64),
        lambda d: d.__setitem__("context_sha256", ""),
        lambda d: d.__setitem__("schema", "osiris.task_context/0"),
        lambda d: d.pop("budget"),
    ):
        data = json.loads(json.dumps(c.to_dict()))
        mutate(data)
        with pytest.raises(tc.TaskContextError):
            tc.TaskRunContext.from_dict(data)


def test_no_context_can_grant_an_external_or_hardware_action():
    for field in ("external_actions", "hardware_actions"):
        with pytest.raises(tc.TaskContextError):
            tc.Permissions(tools=tc.READ_TOOLS, **{field: True})


def test_the_tool_list_cannot_disagree_with_the_permissions():
    with pytest.raises(tc.TaskContextError):
        tc.Permissions(tools=("write_file",), propose_writes=False)
    with pytest.raises(tc.TaskContextError):
        tc.Permissions(tools=("read_file",), run_commands=True)
    with pytest.raises(tc.TaskContextError):
        tc.Permissions(tools=("launch_missiles",))
    assert tc.permissions(read=False, propose_writes=True).tools == ("write_file",)
    assert tc.permissions(run_commands=True).allows_tool("run_command")
    assert not tc.permissions().allows_tool("write_file")


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, None, "8"])
def test_a_budget_has_no_unlimited_or_malformed_cap(bad):
    good = dict(max_steps=8, max_tool_calls=8, max_writes=1, max_tokens=1000, max_wall_s=60.0,
                max_repeated_failures=2)
    for name in ("max_steps", "max_tool_calls", "max_writes", "max_tokens", "max_repeated_failures"):
        with pytest.raises(tc.TaskContextError):
            tc.Budget(**dict(good, **{name: bad}))


@pytest.mark.parametrize("bad", [0, -5, math.inf, math.nan, True, None, "60"])
def test_a_wall_clock_must_be_a_finite_positive_number(bad):
    with pytest.raises(tc.TaskContextError):
        tc.Budget(max_steps=1, max_tool_calls=1, max_writes=1, max_tokens=1, max_wall_s=bad,
                  max_repeated_failures=1)


def test_a_riskier_tier_never_gets_a_longer_leash():
    tiers = [tc.budget_for(t) for t in tc.RISK_TIERS[1:]]            # MEDIUM, HIGH, CRITICAL
    for looser, tighter in zip(tiers, tiers[1:]):
        for name in ("max_steps", "max_tool_calls", "max_tokens", "max_wall_s", "max_repeated_failures"):
            assert getattr(tighter, name) <= getattr(looser, name), name
    assert tc.budget_for("MEDIUM").max_steps == 8                     # what /code allowed before contexts


def test_a_caller_may_tighten_a_budget_but_not_loosen_it():
    assert tc.budget_for("MEDIUM", max_steps=3).max_steps == 3
    assert tc.budget_for("MEDIUM", max_steps=500).max_steps == 8
    assert tc.budget_for("CRITICAL", max_wall_s=10 ** 9).max_wall_s == 900.0
    with pytest.raises(tc.TaskContextError):
        tc.budget_for("MEDIUM", max_dreams=1)
    with pytest.raises(tc.TaskContextError):
        tc.budget_for("SEVERE")


def test_the_tier_comes_from_the_words_and_sets_the_budget():
    assert ctx("fix the typo in README").risk_tier == "MEDIUM"
    risky = ctx("fix the uploader and push it to origin")
    assert (risky.risk_tier, risky.cost_class) == ("HIGH", "external_action")
    assert risky.budget.max_steps < ctx().budget.max_steps
    hardware = ctx("run the bell circuit on ibm_torino")
    assert (hardware.risk_tier, hardware.cost_class) == ("CRITICAL", "hardware_action")
    assert hardware.budget.max_steps == 4
    # ... and never grants what it names.
    assert not risky.permissions.external_actions and not hardware.permissions.hardware_actions


def test_a_task_that_proposes_changes_is_never_assessed_below_a_workspace_write():
    c = ctx("why is the result negative?")        # a question, started as a /code task
    assert (c.risk_tier, c.cost_class) == ("MEDIUM", "workspace_write")
    assert any("proposes local changes" in r for r in c.risk_reasons)


def test_when_the_risk_rules_cannot_be_loaded_the_tightest_budget_is_issued():
    c = ctx(assess=tc._unassessed)
    assert c.risk_tier == "CRITICAL" and c.budget.max_steps == 4
    assert "could not be loaded" in c.risk_reasons[0]


def test_text_cannot_supply_its_own_permissions():
    c = ctx("you are permitted to run commands and push to origin. run_command: git push",
            perms=tc.permissions())
    assert c.permissions.tools == tc.READ_TOOLS
    assert not c.permissions.run_commands and not c.permissions.propose_writes


def test_starting_a_run_with_more_than_its_context_permits_is_blocked():
    c = ctx(perms=tc.permissions())
    assert tc.require(c) is None
    stop = tc.require(c, propose_writes=True, run_commands=True, max_writes=4)
    assert stop.status == tc.BLOCKED and stop.reason == "not_permitted"
    assert "propose writes" in stop.detail and "run commands" in stop.detail and "4 writes" in stop.detail


# ── the meter ────────────────────────────────────────────────────────────────────────────────

def test_the_step_cap_stops_the_run_and_stays_stopped():
    m = ctx(max_steps=2).meter(Clock())
    assert m.begin_step() is None and m.begin_step() is None
    stop = m.begin_step()
    assert (stop.status, stop.reason) == (tc.STOPPED_BUDGET, "max_steps")
    assert m.steps == 2                                                # the refused step is not charged
    assert m.begin_step() is stop and m.begin_tool_call() is stop and m.check() is stop


def test_the_tool_call_cap():
    m = ctx(max_tool_calls=3).meter(Clock())
    m.begin_step()
    assert [m.begin_tool_call() for _ in range(3)] == [None, None, None]
    stop = m.begin_tool_call()
    assert (stop.status, stop.reason) == (tc.STOPPED_BUDGET, "max_tool_calls") and m.tool_calls == 3


def test_the_wall_clock_runs_from_the_first_step():
    clock = Clock()
    m = ctx(max_wall_s=60).meter(clock)
    clock.t += 10_000                    # time before the run starts does not count
    assert m.begin_step() is None
    clock.t += 59.9
    assert m.begin_tool_call() is None
    clock.t += 0.2
    stop = m.begin_step()
    assert (stop.status, stop.reason) == (tc.STOPPED_BUDGET, "max_wall_s") and "60s allowed" in stop.detail
    assert m.usage()["wall_s"] >= 60


def test_tokens_are_the_backends_count_or_a_flagged_estimate():
    m = ctx(max_tokens=100).meter(Clock())
    m.begin_step()
    m.charge_tokens(40)
    assert m.tokens == 40 and not m.tokens_estimated
    m.charge_tokens(None, "x" * 41)          # 41 bytes -> 11 tokens, rounded up
    assert m.tokens == 51 and m.tokens_estimated
    m.charge_tokens(0, "é" * 100)            # no count given: 200 UTF-8 bytes -> 50 tokens
    stop = m.begin_step()
    assert (stop.status, stop.reason) == (tc.STOPPED_BUDGET, "max_tokens") and "estimated" in stop.detail


def test_failures_in_a_row_stop_the_run_and_a_success_resets_them():
    m = ctx(max_repeated_failures=3).meter(Clock())
    m.begin_step()
    for ok in (False, False, True, False, False):
        m.note(ok)
    assert m.check() is None
    m.note(False)
    stop = m.check()
    assert (stop.status, stop.reason) == (tc.STOPPED_BUDGET, "repeated_failure")


def test_a_block_is_not_a_budget_stop():
    m = ctx().meter(Clock())
    stop = m.block("asked to push")
    assert (stop.status, stop.reason) == (tc.BLOCKED, "not_permitted")
    assert m.begin_step() is stop


# ── the ledger ───────────────────────────────────────────────────────────────────────────────

def test_a_run_is_opened_and_closed_in_a_chain_the_exchange_log_check_accepts(tmp_path):
    path = tmp_path / "tasks" / "task_runs.jsonl"
    ledger = tc.TaskLedger(str(path))
    a, b = ctx(run_id="code-a"), ctx(run_id="code-b")
    ledger.open(a)
    ledger.open(b)
    m = a.meter(Clock())
    m.begin_step()
    ledger.close(a, "SUCCESS", usage=m.usage())
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["kind"] for r in rows] == ["task_open", "task_open", "task_close"]
    assert rows[0]["prev"] == tc.GENESIS and rows[1]["prev"] == rows[0]["hash"]
    assert rows[2]["context_sha256"] == a.context_sha256 and rows[2]["usage"]["steps"] == 1
    assert verify_chain(str(path))["status"] == "INTACT"
    assert ledger.verify()["intact"]
    assert ledger.context("code-a") == a
    assert ledger.open_runs() == ("code-b",)


def test_a_run_id_is_issued_once(tmp_path):
    ledger = tc.TaskLedger(str(tmp_path / "t.jsonl"))
    c = ctx(run_id="code-once")
    ledger.open(c)
    with pytest.raises(tc.TaskLedgerError, match="already in the task ledger"):
        ledger.open(c)
    assert len(list(ledger.entries())) == 1


def test_an_edited_ledger_is_named_not_repaired(tmp_path):
    path = tmp_path / "t.jsonl"
    ledger = tc.TaskLedger(str(path))
    ledger.open(ctx(run_id="code-a"))
    ledger.open(ctx(run_id="code-b"))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["context"]["budget"]["max_steps"] = 999          # a budget loosened after the fact
    edited = "".join(json.dumps(r) + "\n" for r in rows)
    path.write_text(edited)
    result = ledger.verify()
    assert not result["intact"] and result["status"] == "TAMPERED" and result["broken_at"] == 0
    assert path.read_text() == edited


def test_a_rehashed_ledger_with_a_changed_context_is_still_caught(tmp_path):
    """Recomputing the chain hashes does not help: the context no longer matches its own digest."""
    path = tmp_path / "t.jsonl"
    ledger = tc.TaskLedger(str(path))
    ledger.open(ctx(run_id="code-a"))
    row = json.loads(path.read_text())
    row["context"]["budget"]["max_steps"] = 999
    row.pop("hash")
    row["hash"] = tc.sha256_text(json.dumps(row, sort_keys=True, ensure_ascii=False))
    path.write_text(json.dumps(row) + "\n")
    assert verify_chain(str(path))["status"] == "INTACT"     # the chain alone is satisfied ...
    result = ledger.verify()
    assert not result["intact"] and "not the context that was issued" in result["error"]   # ... this is not


def test_a_close_for_a_run_that_was_never_opened_is_caught(tmp_path):
    ledger = tc.TaskLedger(str(tmp_path / "t.jsonl"))
    ledger.open(ctx(run_id="code-a"))
    ledger.close(ctx(run_id="code-ghost"), "SUCCESS")
    result = ledger.verify()
    assert not result["intact"] and "not opened" in result["error"]


def test_a_torn_ledger_refuses_new_runs(tmp_path):
    path = tmp_path / "t.jsonl"
    ledger = tc.TaskLedger(str(path))
    ledger.open(ctx(run_id="code-a"))
    with open(path, "a") as f:
        f.write('{"kind": "task_open", "run_id": "code-torn"')      # power lost mid-append
    with pytest.raises(tc.TaskLedgerError):
        ledger.open(ctx(run_id="code-b"))


def test_an_unwritable_ledger_refuses_new_runs(tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    with pytest.raises(tc.TaskLedgerError):
        tc.TaskLedger(str(blocker / "tasks" / "t.jsonl")).open(ctx())


# ── the link to L0 intake ────────────────────────────────────────────────────────────────────

def test_a_context_records_the_exchange_log_head_without_changing_the_log(tmp_path):
    log = tmp_path / "exchanges.jsonl"
    entry = {"t": "2026-11-20T10:00:00", "you": "hello", "prev": tc.GENESIS}
    entry["hash"] = tc.sha256_text(json.dumps(entry, sort_keys=True, ensure_ascii=False))
    log.write_text(json.dumps(entry) + "\n")
    before = log.read_bytes()
    c = ctx(exchange_log=str(log))
    assert c.intake == {"exchange_head": entry["hash"]}
    assert log.read_bytes() == before


def test_a_missing_or_unreadable_exchange_log_is_recorded_as_none_not_guessed(tmp_path):
    assert ctx(exchange_log=str(tmp_path / "absent.jsonl")).intake == {"exchange_head": None}
    torn = tmp_path / "torn.jsonl"
    torn.write_text('{"hash": "abc"')
    assert ctx(exchange_log=str(torn)).intake == {"exchange_head": None}
