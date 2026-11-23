"""Comprehensive unit & integration tests for OSIRIS Evolution Engine live production wiring
(EVO-PLAN-001..EVO-KNOW-001 & WK-RW-001, WK-RW-002, WK-AC-001, WK-EM-001)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from osiris_cli import (
    calibration,
    code_agent,
    evolution_engine,
    knowledge,
    living,
    node_router,
    orchestration_departments,
    plan_gates,
    quantinuum_gate,
    task_context,
    train,
)
from osiris_cli.node_ids import NODE_ALPHA, NODE_BETA

# ---------------------------------------------------------------------------
# 1. EVO-PLAN-001: compile_and_audit_agent_plan
# ---------------------------------------------------------------------------


def test_evo_plan_001_compile_and_audit_agent_plan_gates_and_receipt(tmp_path: Path) -> None:
    ledger = tmp_path / "plans" / "plan_receipts.jsonl"
    res = code_agent.compile_and_audit_agent_plan(
        intent="Wire surgical write_patch into CodeAgent",
        target_files=["osiris_cli/code_agent.py", "tests/test_evolution_live_wiring.py"],
        pytest_targets=["tests/test_evolution_live_wiring.py"],
        record_receipt=True,
        plan_ledger_path=str(ledger),
        root=tmp_path,
    )
    assert res["valid"] is True
    assert res["gates_passed"] is True
    assert res["plan"]["plan_version"] == "1.0"
    assert len(res["plan_sha256"]) == 64
    assert res["receipt_hash"] is not None and len(res["receipt_hash"]) == 64
    assert ledger.is_file()
    direct = evolution_engine.compile_and_audit_agent_plan(
        intent="Verify read-only plan",
        target_files=[],
        pytest_targets=["tests/test_evolution_live_wiring.py"],
        record_receipt=False,
        root=tmp_path,
    )
    assert direct["valid"] is True and direct["gates_passed"] is True

    # Touching protected governance contract path fails G-10 protected_paths_matched gate
    prot = code_agent.compile_and_audit_agent_plan(
        intent="Attempt protected contract edit",
        target_files=["osiris/contracts/plan_v1.json"],
        pytest_targets=["tests/test_evolution_live_wiring.py"],
        record_receipt=False,
        root=tmp_path,
    )
    assert prot["gates_passed"] is False

    # Quarantined SYS-024 constant in intent fails closed (G-11)
    with pytest.raises(plan_gates.PlanValidationError):
        code_agent.compile_and_audit_agent_plan(
            intent="Set planner threshold to theta_lock = 51.843",
            target_files=["osiris_cli/code_agent.py"],
        )


# ---------------------------------------------------------------------------
# 2. EVO-REASON-001: route_task_calibrated
# ---------------------------------------------------------------------------


def test_evo_reason_001_route_task_calibrated_preserves_route_and_escalates(tmp_path: Path) -> None:
    # Existing route_task contract unchanged
    raw_dec = node_router.route_task("Ask the phone to summarise this paragraph")
    assert raw_dec.routed is True
    assert raw_dec.target_node == NODE_ALPHA

    cal_path = tmp_path / "calibration.jsonl"
    ledger = calibration.CalibrationLedger(cal_path)
    # Low-utility calibrated model: acc = 0.20, conf = 0.20 -> ECE = 0, Brier = 0.16 -> U = 0.12 < 0.25
    for idx, outcome in enumerate([1, 0, 0, 0, 0]):
        ledger.record_observation("ollama/gemma4:e4b", f"q-{idx}", "0.200000", outcome)
    # Miscalibrated model: conf = 1.0, acc = 0.0 -> ECE = 1.0 > 0.15
    for idx in range(5):
        ledger.record_observation("ollama/qwen2.5-coder:7b", f"c-{idx}", "1.000000", 0)

    low_res = node_router.route_task_calibrated(
        "run pytest on osiris_cli",
        candidate_models=("ollama/qwen2.5-coder:7b", "ollama/gemma4:e4b"),
        calibration_path=str(cal_path),
    )
    assert low_res["route"]["target_node"] == NODE_BETA
    assert low_res["route"]["capability_id"] == "workstation_compute"
    assert low_res["selected_model"] == "ollama/gemma4:e4b"
    assert low_res["escalate_multi_candidate"] is True

    # High-utility model: acc = 0.80, conf = 0.80 -> ECE = 0.0, Brier = 0.16 -> U = 0.72 >= 0.25
    cal_high = tmp_path / "calibration_high.jsonl"
    ledger_hi = calibration.CalibrationLedger(cal_high)
    for idx, outcome in enumerate([1, 1, 1, 1, 0]):
        ledger_hi.record_observation("ollama/gemma4:e4b", f"h-{idx}", "0.800000", outcome)

    hi_res = node_router.route_task_calibrated(
        "run pytest on osiris_cli",
        candidate_models=("ollama/gemma4:e4b", "ollama/qwen2.5-coder:7b"),
        calibration_path=str(cal_high),
    )
    assert hi_res["selected_model"] == "ollama/gemma4:e4b"
    assert hi_res["escalate_multi_candidate"] is False


# ---------------------------------------------------------------------------
# 3. EVO-MEM-001: assemble_stratified_working_memory
# ---------------------------------------------------------------------------


def test_evo_mem_001_assemble_stratified_working_memory() -> None:
    turns = [{"role": "user", "content": f"turn-{i} " + ("z" * 160)} for i in range(12)]
    facts = ["OSIRIS-QEC-001 (ibm_fez, job dbtl2vb2kf8s73c9a0d0): 393,216 shots, Lambda_3->7=8.69x"]
    guards = ["SYS-024: Never present refuted theta_lock = 51.843 deg as physical."]

    mem = living.assemble_stratified_working_memory(
        recent_turns=turns,
        grounded_facts=facts,
        quarantined_warnings=guards,
        max_tokens=256,
    )
    assert mem["tokens_used"] <= 256
    assert mem["dropped_turns"] > 0
    assert mem["output_headroom_tokens"] == living.MENTOR_MAX_TOKENS
    assert mem["messages"][0]["role"] == "system"
    assert "[SYS-024 Guardrails]" in mem["messages"][0]["content"]
    assert "[Grounded Facts]" in mem["messages"][0]["content"]
    assert mem["episodic_turns"][-1]["content"].startswith("turn-11")


# ---------------------------------------------------------------------------
# 4. EVO-TOOL-001: CodeAgent surgical write_patch
# ---------------------------------------------------------------------------


def test_evo_tool_001_code_agent_write_patch_confinement_precondition_and_apply(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "router_cfg.py"
    target.write_text("HORIZON_S = 420\nMODE = 'strict'\n", encoding="utf-8")
    good_sha = hashlib.sha256(target.read_bytes()).hexdigest()
    (tmp_path / "outside.py").write_text("SECRET = 1\n", encoding="utf-8")

    assert "write_patch" in code_agent.TOOL_NAMES

    # 1. Escape outside workspace & wrong SHA precondition must both be DENIED without touching disk
    turns_bad = iter(
        [
            {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write_patch",
                            "arguments": {
                                "path": "../outside.py",
                                "expected_sha256": good_sha,
                                "old_string": "SECRET = 1",
                                "new_string": "SECRET = 2",
                            },
                        }
                    },
                    {
                        "function": {
                            "name": "write_patch",
                            "arguments": {
                                "path": "router_cfg.py",
                                "expected_sha256": "0" * 64,
                                "old_string": "HORIZON_S = 420",
                                "new_string": "HORIZON_S = 300",
                            },
                        }
                    },
                ],
            },
            {"content": "Refused invalid patches."},
        ]
    )
    bad_agent = code_agent.CodeAgent(
        root=str(ws),
        chat=lambda m, t, mod: next(turns_bad),
        propose_write=lambda p, c: True,
        has_pending_write=lambda: False,
        out=lambda _: None,
    )
    r_bad = bad_agent.run("Try bad patches")
    assert r_bad.status == "SUCCESS"
    assert [d["decision"] for d in r_bad.decisions] == ["DENIED", "DENIED"]
    assert target.read_text(encoding="utf-8") == "HORIZON_S = 420\nMODE = 'strict'\n"

    # 2. Valid governed write_patch under TaskRunContext stages diff and applies cleanly on operator approval
    ctx = task_context.new_context(
        "Update HORIZON_S to 300",
        kind="code",
        run_id="code-patch001",
        perms=task_context.permissions(propose_writes=True),
        exchange_log=None,
    )
    turns_ok = iter(
        [
            {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write_patch",
                            "arguments": {
                                "path": "router_cfg.py",
                                "expected_sha256": good_sha,
                                "old_string": "HORIZON_S = 420",
                                "new_string": "HORIZON_S = 300",
                            },
                        }
                    }
                ],
            },
            {"content": "Proposed surgical patch."},
        ]
    )
    ok_agent = code_agent.CodeAgent(
        root=str(ws),
        chat=lambda m, t, mod: next(turns_ok),
        propose_write=lambda p, c: True,
        has_pending_write=lambda: False,
        context=ctx,
        out=lambda _: None,
    )
    r_ok = ok_agent.run("Update HORIZON_S to 300", run_id="code-patch001")
    assert r_ok.status == "SUCCESS"
    assert r_ok.decisions[0]["decision"] == "GRANTED"
    # Still untouched on disk until operator /apply
    assert target.read_text(encoding="utf-8") == "HORIZON_S = 420\nMODE = 'strict'\n"

    applied = ok_agent.apply_pending_write(approve=lambda prop: "-HORIZON_S = 420" in prop["diff"])
    assert applied["decision"] == "GRANTED"
    assert target.read_text(encoding="utf-8") == "HORIZON_S = 300\nMODE = 'strict'\n"


# ---------------------------------------------------------------------------
# 5. EVO-KNOW-001: SYS-024 quarantine filter in train.py
# ---------------------------------------------------------------------------


def test_evo_know_001_train_corpus_and_chat_lessons_sys024_filter(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "clean.md").write_text(
        "Staggered dynamical decoupling preserves coherence on heavy-hex chains. " * 4,
        encoding="utf-8",
    )
    (docs_dir / "contaminated.md").write_text(
        "Coherence locks universally at theta_lock = 51.843 degrees across all processors. " * 3,
        encoding="utf-8",
    )
    (docs_dir / "erratum.md").write_text(
        "Erratum (10.5281/zenodo.23832283): theta_lock = 51.843 degrees was refuted on ibm_fez. " * 3,
        encoding="utf-8",
    )

    built = train.build_corpus(["docs"], [".md"], str(tmp_path), enforce_sys024=True)
    kept_paths = {d.path for d in built["docs"]}
    skipped_map = dict(built["skipped"])
    assert kept_paths == {"docs/clean.md", "docs/erratum.md"}
    assert "docs/contaminated.md" in skipped_map
    assert "THETA_LOCK" in skipped_map["docs/contaminated.md"]

    # chat_lessons catches un-negated THETA_LOCK (even though THETA_LOCK has asserts=()) while keeping errata
    ex_log = tmp_path / "exchanges.jsonl"
    rows = [
        {
            "user": "tell me about resonance",
            "reply": "Coherence locks at theta_lock = 51.843 degrees.",
            "learnable": True,
            "heldout": False,
            "voice": "mentor:x",
            "hash": "bad_theta_hash",
        },
        {
            "user": "what happened to 51.843?",
            "reply": "Erratum: theta_lock = 51.843 degrees was refuted by 3-basis tomography.",
            "learnable": True,
            "heldout": False,
            "voice": "mentor:x",
            "hash": "good_erratum_hash",
        },
    ]
    ex_log.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    lessons = train.chat_lessons(str(ex_log))
    assert lessons["flagged"] == ["bad_theta_hash"]
    assert len(lessons["train"]) == 1
    assert "refuted" in lessons["train"][0]


# ---------------------------------------------------------------------------
# 6. WK-RW-001: DOI Lineage Indexer & Knowledge Erratum Warnings
# ---------------------------------------------------------------------------


def test_wk_rw_001_doi_lineage_indexer_and_knowledge_warnings(tmp_path: Path) -> None:
    records_src = tmp_path / "records.json"
    records_src.write_text(
        json.dumps(
            [
                {
                    "id": "24031814",
                    "date": "2026-11-07",
                    "title": "OSIRIS Unified Quantum Hardware Evidence Package v2.0",
                    "cat": "research",
                    "phase": "test",
                },
                {
                    "id": "24287505",
                    "date": "2026-11-16",
                    "title": "Dual-Processor Hayden-Preskill Certification (HP-004 & HP-004B v2)",
                    "cat": "research",
                    "phase": "test",
                },
            ]
        ),
        encoding="utf-8",
    )
    lineage_out = tmp_path / "doi_lineage.json"
    lineage = orchestration_departments.index_doi_lineage(
        records_path=records_src,
        output_path=lineage_out,
    )
    assert lineage["schema"] == "osiris.doi_lineage/1"
    assert len(lineage["lineage_sha256"]) == 64
    for expected_id in ("24031814", "23832282", "23832283", "23213023", "23977105", "23986321", "22870279"):
        doi_key = f"10.5281/zenodo.{expected_id}"
        assert doi_key in lineage["records"]
        assert lineage["records"][doi_key]["status"] != "ACTIVE"
        assert lineage["records"][doi_key]["warning"]

    # Active current record has no warning
    assert knowledge.Knowledge.lookup_doi_warnings(
        "Validated in 10.5281/zenodo.24287505.", lineage_path=str(lineage_out)
    ) == []

    # Superseded / staged-erratum DOI emits warning and annotates format_notes()
    warns = knowledge.Knowledge.lookup_doi_warnings(
        "Earlier table in 10.5281/zenodo.24031814 and 10.5281/zenodo.19656600.",
        lineage_path=str(lineage_out),
    )
    assert len(warns) == 2
    formatted = knowledge.Knowledge.format_notes(
        [{"source": "docs/audit.md", "heading": "History", "text": "See 10.5281/zenodo.24031814 for v2.0."}],
        lineage_path=str(lineage_out),
    )
    assert "[DOI LINEAGE WARNING — 10.5281/zenodo.24031814 (ERRATUM_STAGED)]" in formatted


# ---------------------------------------------------------------------------
# 7. WK-RW-002: Quantum Protocol Pre-Gate in quantinuum_gate.py
# ---------------------------------------------------------------------------


def test_wk_rw_002_validate_quantum_protocol_pregate_and_tier3_hook(tmp_path: Path) -> None:
    good_proto = orchestration_departments.generate_quantum_experiment_protocol(
        idea_id="OSIRIS-XTOP-006",
        title="Non-Clifford Codespace Scrambling on H2-1",
        target_backend="H2-1",
        primary_observable="Post-selected Bell fidelity",
        null_value_str="0.975000",
        target_effect_str="0.014000",
        shots_per_circuit=500,
        num_blocks=6,
    )
    ok_res = quantinuum_gate.validate_quantum_protocol_pregate(good_proto)
    assert ok_res["passed"] is True
    assert ok_res["classical_baseline_ok"] is True
    assert ok_res["colored_noise_and_drift_ok"] is True

    bad_proto = {
        "schema": "osiris.quantum_protocol/1",
        "shots_per_circuit": 500,
        "num_blocks": 6,
        "classical_baseline_passed": False,
        "control_experiments": ["White depolarizing noise only"],
    }
    bad_res = quantinuum_gate.validate_quantum_protocol_pregate(bad_proto)
    assert bad_res["passed"] is False
    assert "CLASSICAL_BASELINE_FAILED" in bad_res["violations"]
    assert "MISSING_COLORED_NOISE_OR_DRIFT_CONTROL" in bad_res["violations"]

    qasm_src = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\nh q[0];\ncx q[0], q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n'
    with pytest.raises(quantinuum_gate.QuantinuumGateError) as exc_info:
        quantinuum_gate.run_tier3_governed(
            qasm_src,
            target="H2-1",
            shots=500,
            budget_hqc="25",
            osiris_home=tmp_path / "ohome",
            protocol_manifest=bad_proto,
        )
    assert exc_info.value.code == "BLOCKED_PROTOCOL_PREGATE"


# ---------------------------------------------------------------------------
# 8. WK-AC-001 & WK-EM-001: Department Handoffs & Reproducibility Capsules
# ---------------------------------------------------------------------------


def test_wk_ac_001_and_wk_em_001_handoff_and_capsule_materialization(tmp_path: Path) -> None:
    artifact = tmp_path / "verified_patch.json"
    artifact.write_text('{"ok": true}\n', encoding="utf-8")

    # Mutating department without worktree isolation or same critic==author must fail closed
    with pytest.raises(ValueError, match="worktree_isolated=True"):
        orchestration_departments.build_department_handoff_receipt(
            from_department="Engineering Department",
            to_department="Infrastructure Department",
            artifact_path=artifact,
            verifier_exit_code=0,
            author_model="ollama/gemma4:e4b",
            critic_model="ollama/qwen2.5-coder:7b",
            worktree_isolated=False,
        )
    with pytest.raises(ValueError, match="critic_model != author_model"):
        orchestration_departments.build_department_handoff_receipt(
            from_department="Engineering Department",
            to_department="Infrastructure Department",
            artifact_path=artifact,
            verifier_exit_code=0,
            author_model="ollama/gemma4:e4b",
            critic_model="ollama/gemma4:e4b",
            worktree_isolated=True,
        )

    handoff = orchestration_departments.build_department_handoff_receipt(
        from_department="Engineering Department",
        to_department="Infrastructure Department",
        artifact_path=artifact,
        verifier_exit_code=0,
        author_model="ollama/gemma4:e4b",
        critic_model="ollama/qwen2.5-coder:7b",
        worktree_isolated=True,
    )
    assert handoff["artifact_sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert len(handoff["handoff_sha256"]) == 64

    # Synthetic workspace materialization & RFC 8785 JCS chain verification
    fake_work = tmp_path / "work"
    hp4 = fake_work / "osiris-cli" / "experiments" / "hp004"
    hp4.mkdir(parents=True)
    (hp4 / "PRE_REGISTRATION.md").write_text("# HP-004 Prereg\n", encoding="utf-8")
    (hp4 / "hp004_staged_circuits.qpy").write_bytes(b"QPY-ISA-BYTES")
    (hp4 / "physical_hp004_results.json").write_text('{"counts": {"0000": 1024}}\n', encoding="utf-8")
    (hp4 / "hp004_submitted_job.json").write_text(
        json.dumps({"backend": "ibm_marrakesh", "job_id": "dbti1fbsbsbs73cb2vu0", "total_shots": 292864}),
        encoding="utf-8",
    )
    (hp4 / "zenodo_hp004_publication_receipt.json").write_text(
        json.dumps({"doi": "10.5281/zenodo.24285577"}),
        encoding="utf-8",
    )

    qec_bundle = fake_work / "osiris-qec" / "submission" / "nexus_h2_bundle"
    qec_bundle.mkdir(parents=True)
    (qec_bundle / "nexus_qec002_manifest.json").write_text('{"protocol_id": "OSIRIS-QEC-002"}\n', encoding="utf-8")
    (qec_bundle / "wen_plaquette_qec002_x_basis_h2_isa.json").write_text('{"gates": []}\n', encoding="utf-8")

    out_jsonl = tmp_path / "reproducibility_capsules.jsonl"
    mat = orchestration_departments.materialize_workspace_reproducibility_capsules(
        work_root=fake_work,
        output_path=out_jsonl,
    )
    assert mat["capsule_count"] == 2
    assert mat["hardware_attested_count"] == 1
    assert mat["offline_or_simulated_count"] == 1

    chain = orchestration_departments.verify_reproducibility_capsules_chain(out_jsonl)
    assert chain["valid"] is True
    assert chain["count"] == 2
    assert chain["tail_sha256"] == mat["tail_capsule_sha256"]
