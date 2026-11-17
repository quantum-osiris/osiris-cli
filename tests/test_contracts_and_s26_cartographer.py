"""Tests for canonical OSIRIS v1 contracts, multi-header auth, and governed S26 Ultra cartographer."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from osiris_cli import contracts_validator as cv
from osiris_cli import ide_api, s26_cartographer


def test_canonical_contracts_enforce_fail_closed_invariants():
    # 1. plan_v1.json: valid vs missing execution_mode when required
    valid_plan = {
        "plan_version": "1.0",
        "execution_mode": "simulation",
        "meta": {
            "plan_id": "plan-001",
            "intent_id": "intent-001",
            "created_at": "2026-11-16T23:00:00Z",
            "planner_hash": "a" * 64,
            "confidence": 0.9,
            "intent": "Verify bell circuit",
        },
        "authority": {"domain": "dnalang.io", "execution_level": "read", "confirmed": False},
        "constraints": {"write_allowed": False, "network_allowed": False, "quantum_domain": "read-only"},
        "steps": [{"id": "s1", "type": "verify", "target_engine": "dnalang-core"}],
        "validation": {"preconditions": ["ast_valid"], "postconditions": ["receipt_minted"]},
    }
    ok, err = cv.validate_contract("plan_v1.json", valid_plan, require_execution_mode_on_plan=True)
    assert ok is True and err is None

    no_mode = {k: v for k, v in valid_plan.items() if k != "execution_mode"}
    ok2, err2 = cv.validate_contract("plan_v1.json", no_mode, require_execution_mode_on_plan=True)
    assert ok2 is False and err2["error_code"] == "SCHEMA_VALIDATION_FAILED" and err2["status"] == "REFUSED"
    assert any(v["field"] == "execution_mode" for v in err2["violating_fields"])

    # 2. execution_v1.json
    valid_exec = {
        "execution_version": "1.0",
        "job_id": "job-001",
        "source": "organism bell { }",
        "source_format": "dnalang",
        "execution_mode": "simulation",
        "target_profile": "aer_statevector",
        "shots": 1024,
        "timeout_seconds": 60,
        "governance_permit_id": "permit-001",
    }
    assert cv.validate_contract("execution_v1.json", valid_exec) == (True, None)

    # 3. receipt_v1.json: simulated run MUST NOT claim hardware=True
    sim_receipt = {
        "receipt_version": "1.0",
        "receipt_id": "rcpt-001",
        "actor": {"id": "devin", "kind": "user"},
        "execution_parameters": {
            "input_sha256": "1" * 64,
            "output_sha256": "2" * 64,
            "target_profile": "organism_sim",
        },
        "timestamps": {
            "received_at": "2026-11-16T23:00:00Z",
            "dispatched_at": "2026-11-16T23:00:01Z",
            "completed_at": "2026-11-16T23:00:02Z",
        },
        "state_mutations": [],
        "evidence_class": "SIMULATED",
        "hardware": False,
        "chain_valid": True,
        "prev_hash": "0" * 64,
        "hash": "3" * 64,
    }
    assert cv.validate_contract("receipt_v1.json", sim_receipt) == (True, None)
    forged_hw = {**sim_receipt, "hardware": True}
    ok_hw, err_hw = cv.validate_contract("receipt_v1.json", forged_hw)
    assert ok_hw is False and err_hw["error_code"] == "SCHEMA_VALIDATION_FAILED"

    # 4. agent_task_v1.json: AURA proposal, Aiden review, Governance refusal
    aura = {
        "type": "proposal",
        "agent": "AURA",
        "department": "Engineering",
        "summary": "Bind notebook run route",
        "options": [{"option_id": "opt-1", "description": "Primary adapter", "parameters": {}}],
    }
    aiden = {
        "type": "review",
        "agent": "Aiden",
        "department": "Security",
        "risks": ["Unverified hardware target"],
        "issues": [],
        "recommendations": ["Run in simulation first"],
    }
    refusal = {
        "type": "refusal",
        "department": "Governance",
        "status": "REFUSED",
        "blockers": ["MISSING_GATE0_HUMAN_AUTHORIZATION"],
        "required_remediations": ["Sign with RELEASE_FROZENS-1"],
    }
    for obj in (aura, aiden, refusal):
        assert cv.validate_contract("agent_task_v1.json", obj) == (True, None)

    # 5. Governed terminal command allowlist vs arbitrary shell refusal
    assert cv.validate_governed_terminal_command("claims") == (True, None)
    assert cv.validate_governed_terminal_command("cat bell.dna") == (True, None)
    ok_sh, err_sh = cv.validate_governed_terminal_command("ls -la")
    assert ok_sh is False and err_sh["error_code"] == "UNAUTHORIZED_COMMAND"
    ok_rm, err_rm = cv.validate_governed_terminal_command("rm -rf /")
    assert ok_rm is False and err_rm["status"] == "REFUSED"


def test_ide_api_accepts_x_osiris_token_x_session_token_and_bearer_auth(tmp_path):
    backend = ide_api.default_backend(
        examples_dir=str(tmp_path),
        slice_receipts=str(tmp_path / "slice.jsonl"),
        github_receipts=str(tmp_path / "gh.jsonl"),
    )
    httpd, tok = ide_api.serve(backend, host="127.0.0.1", port=0)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        url = f"http://127.0.0.1:{port}/api/v1/meta"
        for hdr in (
            {"X-OSIRIS-Token": tok},
            {"X-Session-Token": tok},
            {"Authorization": f"Bearer {tok}"},
        ):
            req = urllib.request.Request(url, headers=hdr)
            with urllib.request.urlopen(req, timeout=5) as resp:
                assert resp.status == 200
                body = json.loads(resp.read().decode("utf-8"))
                assert body["service"] == "osiris-ide-api"

        # Missing or wrong credentials across all three headers must fail closed with 401
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(urllib.request.Request(url), timeout=5)
        assert exc_info.value.code == 401
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_s26_cartographer_prime_cores_ast_gate_hash_chain_csv_and_table(tmp_path):
    # 1. Simulate Snapdragon 8 Elite (2+6 Oryon: cpu0..5 @ 3.53 GHz, cpu6..7 @ 4.74 GHz)
    sysfs = tmp_path / "cpu"
    for cid in range(8):
        cpudir = sysfs / f"cpu{cid}" / "cpufreq"
        cpudir.mkdir(parents=True)
        freq = 4742400 if cid in (6, 7) else 3532800
        (cpudir / "cpuinfo_max_freq").write_text(f"{freq}\n", encoding="utf-8")
    assert s26_cartographer.detect_prime_cores(str(sysfs)) == [6, 7]

    # 2. Scan workspace with clean Python (including open()), unsafe Python, and .dna file
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "clean_reader.py").write_text(
        "def load_text(p: str) -> str:\n    with open(p, 'r', encoding='utf-8') as f:\n        return f.read()\n",
        encoding="utf-8",
    )
    (ws / "unsafe_exec.py").write_text("import os\nos.system('uname -a')\n", encoding="utf-8")
    (ws / "sample.dna").write_text("organism bell { }\n", encoding="utf-8")

    ledger = tmp_path / "semantic_ledger.jsonl"
    res = s26_cartographer.run_cartography(ws, ledger, max_workers=2)
    assert res["processed"] == 3 and res["approved"] == 2 and res["rejected"] == 1

    # Verify zero refuted CRSM numerology in ledger output
    raw_ledger = ledger.read_text(encoding="utf-8")
    assert "51.843" not in raw_ledger and "1.7975" not in raw_ledger and "su2_trace" not in raw_ledger

    valid, count, _ = s26_cartographer.verify_semantic_ledger(ledger)
    assert valid is True and count == 3

    summary = s26_cartographer.summarize_ledger(ledger)
    assert summary["chain_valid"] is True and summary["approved_count"] == 2 and summary["rejected_count"] == 1

    csv_out = tmp_path / "export.csv"
    assert s26_cartographer.export_ledger_csv(ledger, csv_out) == 3
    table_all = s26_cartographer.render_ascii_table(csv_out)
    assert "APPROVED_READ_ONLY" in table_all and "REJECTED_POLICY_VIOLATION" in table_all
    table_rej = s26_cartographer.render_ascii_table(csv_out, status_filter="REJECTED_POLICY_VIOLATION")
    assert "REJECTED_POLICY_VIOLATION" in table_rej and "APPROVED_READ_ONLY" not in table_rej

    # Tamper detection on hash chain
    lines = raw_ledger.splitlines()
    tampered = json.loads(lines[0])
    tampered["ast_nodes"] = 9999
    lines[0] = json.dumps(tampered)
    ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
    valid_after, broken_idx, _ = s26_cartographer.verify_semantic_ledger(ledger)
    assert valid_after is False and broken_idx == 0
