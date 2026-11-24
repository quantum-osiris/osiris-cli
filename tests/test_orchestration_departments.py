"""Unit & integration tests for osiris_cli.orchestration_departments."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osiris_cli import calibration, orchestration_departments


def test_identify_workflow_weaknesses_covers_all_three_domains() -> None:
    weaknesses = orchestration_departments.identify_workflow_weaknesses()
    domains = {w.domain for w in weaknesses}
    assert domains == {"research_workflows", "agent_collaboration", "experiment_management"}
    for w in weaknesses:
        assert w.root_cause
        assert w.architectural_improvement
        assert len(w.migration_strategy) >= 3
        assert len(w.success_metrics) >= 2


def test_assign_virtual_departments_enforces_cross_model_critic_and_calibration(
    tmp_path: Path,
) -> None:
    cal_file = tmp_path / "calibration.jsonl"
    ledger = calibration.CalibrationLedger(cal_file)
    for i in range(4):
        ledger.record_observation("ollama/miscalibrated:7b", f"t-{i}", "1.000000", 0)
        ledger.record_observation("ollama/calibrated:4b", f"u-{i}", "0.500000", i % 2)

    plan = orchestration_departments.assign_virtual_departments(
        "Execute OSIRIS-XTOP-005 on Quantinuum H2-1E",
        candidate_models=("ollama/miscalibrated:7b", "ollama/calibrated:4b"),
        calibration_path=cal_file,
    )
    assert plan["calibrated_primary_model"] == "ollama/calibrated:4b"
    assert plan["cross_model_critic"] != plan["calibrated_primary_model"]
    dept_names = [d["department"] for d in plan["departments"]]
    assert dept_names == [
        "Research Department",
        "Engineering Department",
        "Infrastructure Department",
        "Publication Department",
        "Business Development Department",
    ]
    assert len(plan["orchestration_sha256"]) == 64


def test_generate_quantum_experiment_protocol_answers_four_skeptic_questions() -> None:
    proto = orchestration_departments.generate_quantum_experiment_protocol(
        idea_id="OSIRIS-XTOP-006",
        title="Tunable Non-Clifford Scrambling in [[4,2,2]] Codespace",
        target_backend="Quantinuum H2-1",
        primary_observable="Post-selected Bell recovery fidelity F(theta | Phi+)",
        null_value_str="0.975000",
        target_effect_str="0.014000",
        shots_per_circuit=500,
        num_blocks=6,
    )
    sk = proto["skeptic_interrogation"]
    assert set(sk.keys()) == {
        "what_convinces_skeptic",
        "what_evidence_is_missing",
        "what_experiment_produces_it",
        "what_publication_becomes_possible",
    }
    assert len(proto["control_experiments"]) == 4
    assert len(proto["failure_criteria"]) == 3
    assert len(proto["protocol_sha256"]) == 64


def test_audit_scientific_rigor_flags_defects_and_passes_valid_study() -> None:
    flawed = {
        "title": "Universal resonance at theta_lock = 51.843 degrees",
        "total_shots": 128,
        "num_blocks": 1,
        "evidence_class": "SIMULATED",
        "provider_job_id": "fake_job_123",
    }
    bad_audit = orchestration_departments.audit_scientific_rigor(flawed)
    assert bad_audit["passed"] is False
    assert set(bad_audit["flag_codes"]) == {
        "UNSUPPORTED_CLAIM",
        "WEAK_EVIDENCE",
        "MISSING_CONTROLS",
        "MISSING_BASELINES",
        "MISSING_STATISTICAL_ANALYSIS",
        "PROVENANCE_VIOLATION",
    }

    rigorous = {
        "title": "Layout-locked 9-basis Hayden-Preskill recovery on ibm_fez",
        "total_shots": 292864,
        "num_blocks": 6,
        "controls": ["Arm B-Ctrl unentangled", "A/A drift sentinel"],
        "classical_baseline": "Exact statevector + Terhal-Vollbrecht gamma_4(F) bound",
        "statistical_test": "6-block Student-t + 4000-resample bootstrap",
        "confidence_interval": "[0.7237, 0.7377]",
        "evidence_class": "HARDWARE_ATTESTED",
        "provider_job_id": "dbtj61bnlcms73ajc380",
        "raw_counts_sha256": "a" * 64,
    }
    good_audit = orchestration_departments.audit_scientific_rigor(rigorous)
    assert good_audit["passed"] is True
    assert good_audit["flags"] == []


def test_rfc8785_reproducibility_capsule_enforces_sim_vs_hardware_boundary() -> None:
    with pytest.raises(ValueError, match="SIMULATED capsule cannot carry provider_job_id"):
        orchestration_departments.build_rfc8785_reproducibility_capsule(
            experiment_id="OSIRIS-QEC-002",
            evidence_class="SIMULATED",
            prereg_sha256="1" * 64,
            circuit_isa_sha256="2" * 64,
            provider="Quantinuum",
            backend_id="H2-1E",
            shots=1000,
            provider_job_id="bad_id",
        )

    sim_cap = orchestration_departments.build_rfc8785_reproducibility_capsule(
        experiment_id="OSIRIS-QEC-002",
        evidence_class="OFFLINE_COMPILED",
        prereg_sha256="1" * 64,
        circuit_isa_sha256="2" * 64,
        provider="Quantinuum",
        backend_id="H2-1",
        shots=1000,
        zenodo_doi="10.5281/zenodo.24424726",
    )
    assert sim_cap["provider_job_id"] is None
    assert len(sim_cap["capsule_sha256"]) == 64


def test_blockers_opportunities_and_cli_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ohome = tmp_path / ".osiris"
    ohome.mkdir()
    rc = orchestration_departments.main(["--osiris-home", str(ohome)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    cats = {b["category"] for b in payload["blockers"]}
    assert cats == set(orchestration_departments.BLOCKER_CATEGORIES)
    assert len(payload["opportunities"]) >= 4
    assert len(payload["publication_plan"]) >= 3
    assert "durable_system_built" in payload["self_improvement"]


def test_index_doi_lineage_discovers_v620_zenodo_receipts_and_supersession(tmp_path: Path) -> None:
    fake_work = tmp_path / "work"
    dep_dir = fake_work / "osiris-cli" / "papers" / "zenodo_v620_deposits" / "hp004_v2_1"
    dep_dir.mkdir(parents=True)
    (dep_dir / "zenodo_receipt.json").write_text(
        json.dumps(
            {
                "slug": "hp004_v2_1",
                "deposition_id": 24486953,
                "supersedes_record_id": 24287505,
                "prereserved_doi": "10.5281/zenodo.24486953",
                "updated_utc": "2026-11-24T01:11:57+00:00",
                "title": "Dual-Processor Hayden-Preskill Certification v2.1",
            }
        ),
        encoding="utf-8",
    )
    qdec_dir = fake_work / "osiris-cli" / "papers" / "zenodo_v620_deposits" / "qdecode_heron_59"
    qdec_dir.mkdir(parents=True)
    (qdec_dir / "zenodo_receipt.json").write_text(
        json.dumps(
            {
                "slug": "qdecode_heron_59",
                "deposition_id": 24486962,
                "prereserved_doi": "10.5281/zenodo.24486962",
                "updated_utc": "2026-11-24T01:12:10+00:00",
                "title": "Reference-Verified Wire Deserialization 59-Job Corpus",
            }
        ),
        encoding="utf-8",
    )
    out_json = tmp_path / "doi_lineage.json"
    res = orchestration_departments.index_doi_lineage(
        records_path=tmp_path / "missing_records.json",
        output_path=out_json,
        work_root=fake_work,
    )
    recs = res["records"]
    assert recs["10.5281/zenodo.24486953"]["status"] == "ACTIVE"
    assert recs["10.5281/zenodo.24486953"]["supersedes"] == ["10.5281/zenodo.24287505"]
    assert recs["10.5281/zenodo.24287505"]["status"] == "SUPERSEDED"
    assert recs["10.5281/zenodo.24287505"]["superseded_by"] == "10.5281/zenodo.24486953"
    assert recs["10.5281/zenodo.24486962"]["status"] == "ACTIVE"


def test_run_science_bench_compiles_without_syntax_warning() -> None:
    import py_compile
    import warnings

    repo_root = Path(__file__).resolve().parents[1]
    sb_file = repo_root / "osiris-benchmarks" / "science_bench" / "run_science_bench.py"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SyntaxWarning)
        py_compile.compile(str(sb_file), doraise=True)
    syntax_warns = [w for w in caught if issubclass(w.category, SyntaxWarning)]
    assert syntax_warns == []
