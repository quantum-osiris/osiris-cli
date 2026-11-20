"""Phase C tests: DAG task state machine, exec-canon-1 task receipts, separate verifier, and IDE API 1.7.0.

Covers §5, §6, §7, §9, §11, and §13 requirements:
  - REQ-GRAPH-1..3 (DAG cycle/cap validation, VERIFIED dependency barrier, server-owned state)
  - REQ-EXEC-1..2, REQ-EXEC-5 (kernel execution, argv-driven tier, atomic stop cascade)
  - REQ-AUTH-1..2, REQ-AUTH-5 (bound grants, 1-byte tamper refusal, agent cannot confirm)
  - REQ-VER-4 / G-8 (separate OS process verifier, integrity vs acceptance split, disk tamper -> INTEGRITY_FAIL)
  - REQ-EV-1..4, REQ-EV-7 (exec-canon-1, 0600 permissions, cross-ledger refs, broken chain blocks new tasks)
  - REQ-API-1..4 (IDE API 1.7.0 contract conformance, idempotent client_request_id, REST aliases)
  - REQ-SEC-6 / G-16 (protected governance paths denied for agent write nodes)
"""

from __future__ import annotations

import http.client
import json
import os
from pathlib import Path
import sys
import threading

import pytest

from osiris_cli import ide_api
from osiris_cli.task_graph import (
    ApprovalRefusedError,
    ChainIntegrityError,
    TaskGraphEngine,
    TaskGraphError,
)
from osiris_cli.task_receipts import TaskReceiptLedger, build_transition_body


@pytest.fixture
def engine_env(tmp_path):
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    (workdir / "hello.txt").write_text("hello phase c\n", encoding="utf-8")
    (workdir / "sample.py").write_text("def square(x):\n    return x * x\n", encoding="utf-8")
    (workdir / "test_sample.py").write_text(
        "from sample import square\n\ndef test_square():\n    assert square(4) == 16\n",
        encoding="utf-8",
    )
    eng = TaskGraphEngine(
        graphs_dir=tmp_path / "task_graphs",
        receipts_path=tmp_path / "task_receipts.jsonl",
        exec_journal_path=tmp_path / "exec_journal.jsonl",
        artifacts_dir=tmp_path / "runs",
        default_workdir=str(workdir),
    )
    return {"engine": eng, "workdir": workdir, "tmp": tmp_path}


def test_dag_cycle_and_node_cap_rejected(engine_env):
    eng = engine_env["engine"]
    # Cycle A -> B -> A rejected at decompose time (REQ-GRAPH-1)
    with pytest.raises(TaskGraphError) as exc_cycle:
        eng.create_graph(
            objective="Cyclic graph",
            nodes=[
                {"node_id": "a", "kind": "run_command", "argv": ["ls"], "depends_on": ["b"], "acceptance_criteria": ["exit_code_zero"]},
                {"node_id": "b", "kind": "run_command", "argv": ["ls"], "depends_on": ["a"], "acceptance_criteria": ["exit_code_zero"]},
            ],
        )
    assert exc_cycle.value.code == "GRAPH_CYCLE"

    # Node cap exceeded rejected (REQ-GRAPH-1, REQ-GEN-4)
    with pytest.raises(TaskGraphError) as exc_cap:
        eng.create_graph(
            objective="Too many nodes",
            max_nodes=2,
            nodes=[
                {"node_id": f"n{i}", "kind": "run_command", "argv": ["ls"], "acceptance_criteria": ["exit_code_zero"]}
                for i in range(3)
            ],
        )
    assert exc_cap.value.code == "NODE_CAP_EXCEEDED"

    # Raw shell command string rejected (REQ-PLAN-1 / G-12)
    with pytest.raises(TaskGraphError) as exc_cmd:
        eng.create_graph(
            objective="Raw shell",
            nodes=[{"node_id": "n1", "kind": "run_command", "command": "ls -la", "acceptance_criteria": ["exit_code_zero"]}],
        )
    assert exc_cmd.value.code == "RAW_COMMAND_REFUSED"

    # Missing acceptance criteria rejected (§6 step 5)
    with pytest.raises(TaskGraphError) as exc_crit:
        eng.create_graph(
            objective="No criteria",
            nodes=[{"node_id": "n1", "kind": "run_command", "argv": ["ls"], "acceptance_criteria": []}],
        )
    assert exc_crit.value.code == "MISSING_ACCEPTANCE_CRITERIA"


def test_tier_comes_from_argv_not_prompt(engine_env):
    """REQ-EXEC-2 / G-7: 'show status' with argv=['rm', 'x'] is classified MUTATE from argv, never READ."""
    eng = engine_env["engine"]
    # For origin="model", rm is MUTATE + agent_deny -> DENY -> BLOCKED
    created_model = eng.create_graph(
        objective="show status",
        origin="model",
        nodes=[
            {
                "node_id": "n1",
                "title": "show status",
                "kind": "run_command",
                "argv": ["rm", "hello.txt"],
                "cwd": ".",
                "acceptance_criteria": ["exit_code_zero"],
            }
        ],
    )
    node_model = created_model["graph"]["nodes"][0]
    assert node_model["tier"] == "MUTATE"
    assert node_model["policy_decision"] == "DENY"
    assert node_model["state"] == "BLOCKED"

    # For origin="human", rm is MUTATE -> ASK -> AWAITING_APPROVAL (never READ/ALLOW)
    created_human = eng.create_graph(
        objective="show status",
        origin="human",
        nodes=[
            {
                "node_id": "n1",
                "title": "show status",
                "kind": "run_command",
                "argv": ["rm", "hello.txt"],
                "cwd": ".",
                "acceptance_criteria": ["exit_code_zero"],
            }
        ],
    )
    node_human = created_human["graph"]["nodes"][0]
    assert node_human["tier"] == "MUTATE"
    assert node_human["policy_decision"] == "ASK"
    assert node_human["state"] == "AWAITING_APPROVAL"


def test_allow_node_auto_authorizes_and_verifies_in_separate_process(engine_env):
    """ALLOW node moves DRAFT -> AUTHORIZED, executes via kernel, and verifies in a separate PID (REQ-VER-4)."""
    eng = engine_env["engine"]
    created = eng.create_graph(
        objective="Read workspace file",
        origin="human",
        nodes=[
            {
                "node_id": "n1",
                "title": "Read hello.txt",
                "kind": "run_command",
                "argv": ["cat", "hello.txt"],
                "cwd": ".",
                "acceptance_criteria": ["exit_code_zero", "stdout_contains:hello phase c"],
            }
        ],
    )
    gid = created["graph"]["graph_id"]
    assert created["graph"]["nodes"][0]["state"] == "AUTHORIZED"

    res = eng.execute_node(gid, "n1")
    node = res["node"]
    assert node["state"] == "VERIFIED"
    assert node["verification"]["integrity"] == "PASS"
    assert node["verification"]["acceptance"] == "PASS"
    assert node["verification"]["verifier_pid"] != os.getpid()
    assert res["chain_valid"] is True

    # Every transition produced a receipt: DRAFT -> AUTHORIZED -> EXECUTING -> VERIFYING -> VERIFIED
    events = eng.events(graph_id=gid)["rows"]
    states = [r["body"]["task"]["state_after"] for r in events]
    assert states == ["DRAFT", "AUTHORIZED", "EXECUTING", "VERIFYING", "VERIFIED"]
    assert events[-1]["body"]["refs"]["exec_journal_hash"] is not None
    assert events[-1]["body"]["refs"]["artifact_sha256"] == node["artifact_sha256"]


def test_dependency_must_be_verified_before_start(engine_env):
    """REQ-GRAPH-2: A dependent node cannot execute until every dependency is VERIFIED."""
    eng = engine_env["engine"]
    created = eng.create_graph(
        objective="Chained read tasks",
        origin="human",
        nodes=[
            {
                "node_id": "n1",
                "kind": "run_command",
                "argv": ["cat", "hello.txt"],
                "cwd": ".",
                "acceptance_criteria": ["exit_code_zero"],
            },
            {
                "node_id": "n2",
                "kind": "run_command",
                "argv": ["ls"],
                "cwd": ".",
                "depends_on": ["n1"],
                "acceptance_criteria": ["exit_code_zero"],
            },
        ],
    )
    gid = created["graph"]["graph_id"]
    with pytest.raises(TaskGraphError) as exc:
        eng.execute_node(gid, "n2")
    assert exc.value.code == "DEPENDENCY_NOT_VERIFIED"

    assert eng.execute_node(gid, "n1")["node"]["state"] == "VERIFIED"
    assert eng.execute_node(gid, "n2")["node"]["state"] == "VERIFIED"
    assert eng.get_graph(gid)["graph"]["status"] == "VERIFIED"


def test_agent_origin_cannot_confirm(engine_env):
    """REQ-AUTH-5: A confirm/approval from any agent origin (model, skill, claude_code) is refused and receipted."""
    eng = engine_env["engine"]
    created = eng.create_graph(
        objective="Write helper note",
        origin="model",
        nodes=[
            {
                "node_id": "n1",
                "kind": "write_file",
                "target_path": "note.txt",
                "content": "phase c note\n",
                "acceptance_criteria": ["file_exists:note.txt"],
            }
        ],
    )
    gid = created["graph"]["graph_id"]
    node = created["graph"]["nodes"][0]
    assert node["state"] == "AWAITING_APPROVAL"

    for bad_origin in ("model", "skill", "claude_code"):
        with pytest.raises(ApprovalRefusedError) as exc:
            eng.approve_node(
                gid,
                "n1",
                action_digest=node["action_digest"],
                actor_origin=bad_origin,
                session_id="sess-1",
            )
        assert exc.value.code == "AGENT_CANNOT_CONFIRM"

    assert eng.get_graph(gid)["graph"]["nodes"][0]["state"] == "AWAITING_APPROVAL"


def test_one_byte_change_after_approval_is_refused(engine_env):
    """REQ-AUTH-2: Changing one byte of argv or file content after approval is refused and logged."""
    eng = engine_env["engine"]
    created = eng.create_graph(
        objective="Write controlled file",
        origin="model",
        nodes=[
            {
                "node_id": "n1",
                "kind": "write_file",
                "target_path": "output.txt",
                "content": "approved-v1\n",
                "acceptance_criteria": ["file_exists:output.txt"],
            },
            {
                "node_id": "n2",
                "kind": "run_command",
                "argv": ["cat", "output.txt"],
                "depends_on": ["n1"],
                "acceptance_criteria": ["exit_code_zero"],
            },
        ],
    )
    gid = created["graph"]["graph_id"]
    n1 = created["graph"]["nodes"][0]
    eng.approve_node(
        gid,
        "n1",
        action_digest=n1["action_digest"],
        actor_origin="human",
        session_id="human-session-42",
    )

    # Mutate 1 byte of the stored node payload on disk after approval!
    gpath = eng.graphs_dir / f"{gid}.json"
    raw = json.loads(gpath.read_text(encoding="utf-8"))
    raw["nodes"][0]["content"] = "approved-v2\n"
    gpath.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    res = eng.execute_node(gid, "n1")
    assert res["node"]["state"] == "FAILED"
    assert "ACTION_DIGEST_MISMATCH" in res["node"]["last_reason"]
    assert not (engine_env["workdir"] / "output.txt").exists()
    # Downstream node cascaded to BLOCKED (REQ-EXEC-5)
    assert res["graph"]["nodes"][1]["state"] == "BLOCKED"


def test_artifact_tampered_on_disk_fails_integrity(engine_env):
    """REQ-VER-4 / G-8: A 1-byte disk tamper between execute and verify returns INTEGRITY_FAIL and FAILED."""
    eng = engine_env["engine"]
    created = eng.create_graph(
        objective="Check tamper detection",
        origin="human",
        nodes=[
            {
                "node_id": "n1",
                "kind": "run_command",
                "argv": ["cat", "hello.txt"],
                "cwd": ".",
                "acceptance_criteria": ["exit_code_zero", "stdout_contains:hello phase c"],
            },
            {
                "node_id": "n2",
                "kind": "run_command",
                "argv": ["ls"],
                "depends_on": ["n1"],
                "acceptance_criteria": ["exit_code_zero"],
            },
        ],
    )
    gid = created["graph"]["graph_id"]

    def flip_one_byte(artifact_path: Path) -> None:
        raw = artifact_path.read_bytes()
        artifact_path.write_bytes(raw + b" ")

    res = eng.execute_node(gid, "n1", pre_verify_hook=flip_one_byte)
    node = res["node"]
    assert node["state"] == "FAILED"
    assert node["verification"]["integrity"] == "FAIL"
    assert node["verification"]["code"] == "INTEGRITY_FAIL"
    # Acceptance still ran independently and saw the text, proving integrity & acceptance are separate!
    assert node["verification"]["acceptance"] == "PASS"
    assert res["graph"]["nodes"][1]["state"] == "BLOCKED"


def test_acceptance_failure_separated_from_integrity(engine_env):
    """REQ-VER-4: Intact artifact failing acceptance produces integrity=PASS, acceptance=FAIL, state=FAILED."""
    eng = engine_env["engine"]
    created = eng.create_graph(
        objective="Expect missing marker",
        origin="human",
        nodes=[
            {
                "node_id": "n1",
                "kind": "run_command",
                "argv": ["cat", "hello.txt"],
                "cwd": ".",
                "acceptance_criteria": ["exit_code_zero", "stdout_contains:NEVER_PRESENT_TOKEN"],
            }
        ],
    )
    gid = created["graph"]["graph_id"]
    res = eng.execute_node(gid, "n1")
    node = res["node"]
    assert node["state"] == "FAILED"
    assert node["verification"]["integrity"] == "PASS"
    assert node["verification"]["acceptance"] == "FAIL"
    assert node["verification"]["code"] == "ACCEPTANCE_FAIL"


def test_broken_chain_blocks_new_tasks(engine_env):
    """REQ-EV-4: Flipping 1 byte in task_receipts.jsonl or referenced exec_journal.jsonl blocks new tasks."""
    eng = engine_env["engine"]
    g1 = eng.create_graph(
        objective="Initial clean task",
        origin="human",
        nodes=[{"node_id": "n1", "kind": "run_command", "argv": ["cat", "hello.txt"], "acceptance_criteria": ["exit_code_zero"]}],
    )
    assert eng.execute_node(g1["graph"]["graph_id"], "n1")["node"]["state"] == "VERIFIED"

    # 1. Tamper 1 byte in referenced exec_journal.jsonl -> blocks new task graph!
    ej_path = eng.exec_journal.path
    orig_ej = ej_path.read_text(encoding="utf-8")
    ej_lines = orig_ej.splitlines()
    row = json.loads(ej_lines[0])
    row["body"]["cwd"] = "/tampered/path"
    ej_lines[0] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    ej_path.write_text("\n".join(ej_lines) + "\n", encoding="utf-8")

    ok, _, err = eng.verify_chains()
    assert ok is False and "exec_journal" in err
    with pytest.raises(ChainIntegrityError):
        eng.create_graph(
            objective="Should be blocked by broken exec_journal",
            origin="human",
            nodes=[{"node_id": "n1", "kind": "run_command", "argv": ["ls"], "acceptance_criteria": ["exit_code_zero"]}],
        )

    # Restore exec_journal, then tamper 1 byte in task_receipts.jsonl -> blocks new task graph!
    ej_path.write_text(orig_ej, encoding="utf-8")
    assert eng.verify_chains()[0] is True

    tr_path = eng.receipts.path
    tr_lines = tr_path.read_text(encoding="utf-8").splitlines()
    trow = json.loads(tr_lines[0])
    trow["body"]["reason"] = "tampered reason"
    tr_lines[0] = json.dumps(trow, sort_keys=True, separators=(",", ":"))
    tr_path.write_text("\n".join(tr_lines) + "\n", encoding="utf-8")

    with pytest.raises(ChainIntegrityError):
        eng.create_graph(
            objective="Should be blocked by broken task_receipts",
            origin="human",
            nodes=[{"node_id": "n1", "kind": "run_command", "argv": ["ls"], "acceptance_criteria": ["exit_code_zero"]}],
        )


def test_task_receipts_file_mode_0600_and_exec_canon_1_rejects_floats(tmp_path):
    """REQ-EV-2 & REQ-EV-3: 0600 permissions, exec-canon-1 format, floats rejected."""
    ledger = TaskReceiptLedger(tmp_path / "task_receipts.jsonl")
    with pytest.raises(ValueError, match="floats"):
        ledger.append(
            {
                "kind": "task-receipt",
                "metric": 0.95,
            }
        )
    body = build_transition_body(
        graph_id="graph-000000000001",
        node_id="n1",
        state_before=None,
        state_after="DRAFT",
        actor_role="orchestrator",
        actor_origin="system",
        action_digest="a" * 64,
        reason="created",
    )
    row = ledger.append(body)
    assert row["canon_version"] == "exec-canon-1"
    assert oct(os.stat(ledger.path).st_mode & 0o777) == "0o600"
    ok, count, err = ledger.verify()
    assert ok is True and count == 1 and err == ""


def test_protected_governance_path_blocked_for_agent_write_node(engine_env):
    """REQ-SEC-6 / G-16: Agent write targeting exec_policy.json or allowed_signers is BLOCKED."""
    eng = engine_env["engine"]
    created = eng.create_graph(
        objective="Attempt governance edit",
        origin="model",
        nodes=[
            {
                "node_id": "g1",
                "kind": "write_file",
                "target_path": "osiris_cli/policy/exec_policy.json",
                "content": "{}\n",
                "acceptance_criteria": ["file_exists:osiris_cli/policy/exec_policy.json"],
            },
            {
                "node_id": "g2",
                "kind": "write_file",
                "target_path": "governance/allowed_signers",
                "content": "rogue key\n",
                "acceptance_criteria": ["file_exists:governance/allowed_signers"],
            },
        ],
    )
    for node in created["graph"]["nodes"]:
        assert node["tier"] == "FORBIDDEN"
        assert node["policy_decision"] == "DENY"
        assert node["state"] == "BLOCKED"


def test_idempotent_client_request_id_replay_changes_nothing(engine_env):
    """REQ-API-4: Replaying client_request_id on create or approve is idempotent."""
    eng = engine_env["engine"]
    first = eng.create_graph(
        objective="Idempotent graph",
        origin="model",
        client_request_id="req-create-001",
        nodes=[
            {
                "node_id": "w1",
                "kind": "write_file",
                "target_path": "idemp.txt",
                "content": "hello\n",
                "acceptance_criteria": ["file_exists:idemp.txt"],
            }
        ],
    )
    replay = eng.create_graph(
        objective="Idempotent graph",
        origin="model",
        client_request_id="req-create-001",
        nodes=[
            {
                "node_id": "w1",
                "kind": "write_file",
                "target_path": "idemp.txt",
                "content": "hello\n",
                "acceptance_criteria": ["file_exists:idemp.txt"],
            }
        ],
    )
    assert replay["idempotent_replay"] is True
    assert replay["graph"]["graph_id"] == first["graph"]["graph_id"]

    gid = first["graph"]["graph_id"]
    digest = first["graph"]["nodes"][0]["action_digest"]
    a1 = eng.approve_node(gid, "w1", action_digest=digest, actor_origin="human", session_id="s1", client_request_id="req-app-1")
    count_before = eng.events(graph_id=gid)["count"]
    a2 = eng.approve_node(gid, "w1", action_digest=digest, actor_origin="human", session_id="s1", client_request_id="req-app-1")
    assert a1["idempotent_replay"] is False
    assert a2["idempotent_replay"] is True
    assert eng.events(graph_id=gid)["count"] == count_before


def test_ide_api_task_graph_routes_match_contract_v1_7_0_and_rest_aliases(tmp_path):
    """End-to-end HTTP test of /api/v1/tasks/* routes and REST aliases against osiris-ide-api-v1.json 1.7.0."""
    jsonschema = pytest.importorskip("jsonschema")
    contract = json.loads(Path(ide_api.CONTRACT_PATH).read_text(encoding="utf-8"))
    assert contract["api_version"] == "1.7.0" and ide_api.API_VERSION == "1.7.0"

    workdir = tmp_path / "repo"
    workdir.mkdir()
    (workdir / "data.txt").write_text("phase-c-verified\n", encoding="utf-8")

    backend = ide_api.Backend(
        gateway_factory=lambda: None,
        slice_receipts=str(tmp_path / "slice.jsonl"),
        examples_dir="",
        task_graphs_dir=str(tmp_path / "graphs"),
        task_receipts=str(tmp_path / "task_receipts.jsonl"),
        exec_journal=str(tmp_path / "exec_journal.jsonl"),
        task_artifacts_dir=str(tmp_path / "runs"),
        task_workdir=str(workdir),
    )
    httpd, token = ide_api.serve(backend, port=0)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        port = httpd.server_address[1]

        def req(method, path, body=None):
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
            h = {"Host": f"127.0.0.1:{port}", "X-OSIRIS-Token": token}
            raw = None
            if body is not None:
                raw = json.dumps(body).encode("utf-8")
                h["Content-Type"] = "application/json"
            c.request(method, path, body=raw, headers=h)
            r = c.getresponse()
            data = json.loads(r.read().decode("utf-8"))
            c.close()
            return r.status, data

        # 1. Create graph with one ALLOW node and one ASK node
        st, created = req(
            "POST",
            "/api/v1/tasks/graph",
            {
                "objective": "Verify Phase C HTTP workflow",
                "origin": "human",
                "client_request_id": "http-create-1",
                "nodes": [
                    {
                        "node_id": "read-1",
                        "title": "Read data.txt",
                        "kind": "run_command",
                        "argv": ["cat", "data.txt"],
                        "cwd": ".",
                        "acceptance_criteria": ["exit_code_zero", "stdout_contains:phase-c-verified"],
                    },
                    {
                        "node_id": "write-2",
                        "title": "Write report",
                        "kind": "write_file",
                        "target_path": "report.txt",
                        "content": "verified\n",
                        "depends_on": ["read-1"],
                        "acceptance_criteria": ["file_exists:report.txt"],
                    },
                ],
            },
        )
        assert st == 200, created
        jsonschema.validate(created, contract["routes"]["POST /api/v1/tasks/graph"]["response"])
        gid = created["graph"]["graph_id"]

        # 2. List graphs
        st, listed = req("GET", "/api/v1/tasks/graphs")
        assert st == 200 and listed["count"] == 1 and listed["chain_valid"] is True
        jsonschema.validate(listed, contract["routes"]["GET /api/v1/tasks/graphs"]["response"])

        # 3. Get graph via both POST /api/v1/tasks/graph/get and REST alias GET /api/v1/tasks/graph/<id>
        st, detail = req("POST", "/api/v1/tasks/graph/get", {"graph_id": gid})
        assert st == 200 and detail["graph"]["graph_id"] == gid
        jsonschema.validate(detail, contract["routes"]["POST /api/v1/tasks/graph/get"]["response"])

        st_alias, detail_alias = req("GET", f"/api/v1/tasks/graph/{gid}")
        assert st_alias == 200 and detail_alias["graph"]["graph_id"] == gid

        # 4. Execute ALLOW node read-1 -> VERIFIED in separate verifier process
        st, ex1 = req("POST", "/api/v1/tasks/node/run", {"graph_id": gid, "node_id": "read-1", "client_request_id": "http-ex-1"})
        assert st == 200 and ex1["node"]["state"] == "VERIFIED"
        jsonschema.validate(ex1, contract["routes"]["POST /api/v1/tasks/node/run"]["response"])

        # 5. Approve ASK node write-2 via REST alias POST /api/v1/tasks/node/<id>/approve
        digest_w2 = detail["graph"]["nodes"][1]["action_digest"]
        st, app2 = req(
            "POST",
            "/api/v1/tasks/node/write-2/approve",
            {
                "graph_id": gid,
                "action_digest": digest_w2,
                "session_id": "operator-ui-session-1",
                "client_request_id": "http-app-2",
            },
        )
        assert st == 200 and app2["node"]["state"] == "AUTHORIZED"
        jsonschema.validate(app2, contract["routes"]["POST /api/v1/tasks/node/approve"]["response"])

        # 6. Execute approved node write-2 -> VERIFIED
        st, ex2 = req("POST", "/api/v1/tasks/node/run", {"graph_id": gid, "node_id": "write-2", "client_request_id": "http-ex-2"})
        assert st == 200 and ex2["node"]["state"] == "VERIFIED"
        assert ex2["graph"]["status"] == "VERIFIED"

        # 7. Events endpoint reports intact chain
        st, ev = req("GET", "/api/v1/tasks/events")
        assert st == 200 and ev["chain_valid"] is True and ev["count"] >= 9
        jsonschema.validate(ev, contract["routes"]["GET /api/v1/tasks/events"]["response"])
    finally:
        httpd.shutdown()
        httpd.server_close()
