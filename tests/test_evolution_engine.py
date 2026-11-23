"""Deterministic unit & integration tests for osiris_cli.evolution_engine."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from osiris_cli import calibration, claim_provenance, evolution_engine
from osiris_cli.journal import OperationalMerkleLedger
from osiris_cli.promotion_gate import PromotionCriteria, PromotionEvidence


def test_select_calibrated_model_penalizes_overconfident_failures(tmp_path: Path) -> None:
    cal_file = tmp_path / "calibration.jsonl"
    ledger = calibration.CalibrationLedger(cal_file)
    # Model A: overconfident 0% accuracy (matches observed qwen2.5-coder:7b failure mode)
    for i in range(5):
        ledger.record_observation(
            "ollama/overconfident:7b",
            f"t-{i}",
            "1.000000",
            0,
            prediction_id=f"a-{i}",
        )
    # Model B: calibrated, 60% accuracy with matching stated confidence
    for i, ok in enumerate([1, 1, 1, 0, 0]):
        ledger.record_observation(
            "ollama/calibrated:4b",
            f"t-{i}",
            "0.600000",
            ok,
            prediction_id=f"b-{i}",
        )

    winner, score = evolution_engine.select_calibrated_model(
        ["ollama/overconfident:7b", "ollama/calibrated:4b"],
        calibration_path=cal_file,
        min_samples=3,
    )
    assert winner == "ollama/calibrated:4b"
    assert score is not None
    assert score.ece_str == "0.000000"


def test_stratify_memory_context_respects_token_ceiling_and_keeps_recent() -> None:
    turns = [{"role": "user", "content": f"message-{i} " + ("x" * 120)} for i in range(10)]
    facts = ["OSIRIS-HP-004B certified E_F >= 1.147 ebits on ibm_fez (10.5281/zenodo.24287505)"]
    guards = ["SYS-024: Do not cite refuted 136-bit negentropy sampling artifact."]

    ctx = evolution_engine.stratify_memory_context(
        recent_turns=turns,
        grounded_facts=facts,
        quarantined_warnings=guards,
        max_tokens=200,
    )
    assert ctx["tokens_used"] <= 200
    assert ctx["dropped_turns"] > 0
    assert ctx["episodic_turns"][-1]["content"].startswith("message-9")
    assert ctx["grounded_semantic_facts"] == facts
    assert ctx["sys024_guardrails"] == guards


def test_apply_preconditioned_patch_enforces_sha_and_uniqueness(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text("x = 1\ny = 2\n", encoding="utf-8")
    good_sha = hashlib.sha256(target.read_bytes()).hexdigest()

    # Wrong SHA fails closed
    with pytest.raises(ValueError, match="Precondition SHA-256 mismatch"):
        evolution_engine.apply_preconditioned_patch(
            target,
            expected_sha256="0" * 64,
            old_string="x = 1",
            new_string="x = 10",
        )

    # Dry-run verifies without modifying file on disk
    dry = evolution_engine.apply_preconditioned_patch(
        target,
        expected_sha256=good_sha,
        old_string="x = 1",
        new_string="x = 10",
        dry_run=True,
    )
    assert dry["applied"] is False
    assert target.read_text(encoding="utf-8") == "x = 1\ny = 2\n"

    # Live atomic replacement succeeds and updates digest
    live = evolution_engine.apply_preconditioned_patch(
        target,
        expected_sha256=good_sha,
        old_string="x = 1",
        new_string="x = 10",
        dry_run=False,
    )
    assert live["applied"] is True
    assert target.read_text(encoding="utf-8") == "x = 10\ny = 2\n"
    assert hashlib.sha256(target.read_bytes()).hexdigest() == live["after_sha256"]


def test_evaluate_subsystem_promotion_decouples_engineering_from_nclm_speaking(tmp_path: Path) -> None:
    jpath = tmp_path / "op_journal.jsonl"
    journal = OperationalMerkleLedger(jpath)
    journal.append("init", {"msg": "boot"})

    criteria = PromotionCriteria(
        alpha=0.05,
        w2_max=0.25,
        entropy_floor_bits=2.0,
        registration_ref="prereg:EVO-AUTO-001",
    )
    # Speaking gate is CLOSED (bpb=4.20 > 2.0), but quality, stability, and integrity all pass
    evidence = PromotionEvidence(
        candidate_id="cand-routing-v2",
        candidate_digest="a" * 64,
        delta_q=0.15,
        p_value=0.002,
        w2_drift=0.04,
        entropy_bits=3.5,
        ast_violations=0,
        speaking_gate={"held_out": 30, "mean_bpb": 4.20, "open": False, "unigram_bpb": 4.88},
    )

    eng_verdict = evolution_engine.evaluate_subsystem_promotion(
        evidence, criteria, journal, subsystem_scope="engineering"
    )
    assert eng_verdict.eligible is True
    assert eng_verdict.speaking_gate_required is False

    voice_verdict = evolution_engine.evaluate_subsystem_promotion(
        evidence, criteria, journal, subsystem_scope="nclm_core_voice"
    )
    assert voice_verdict.eligible is False
    assert voice_verdict.speaking_gate_required is True


def test_filter_training_doc_sys024_blocks_unrefuted_and_allows_errata() -> None:
    bad_doc = "Our processor locks coherence at theta_lock = 51.843 degrees universally."
    ok_bad, ids_bad = evolution_engine.filter_training_doc_sys024(bad_doc)
    assert ok_bad is False
    assert "THETA_LOCK" in ids_bad

    erratum_doc = (
        "Erratum (10.5281/zenodo.23832283): theta_lock = 51.843 degrees was refuted "
        "by 3-basis Bell tomography on ibm_fez (p = 0.3537)."
    )
    ok_err, ids_err = evolution_engine.filter_training_doc_sys024(erratum_doc)
    assert ok_err is True
    assert ids_err == ()


def test_monitor_and_generate_four_part_improvements_and_record_cycle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ohome = tmp_path / ".osiris"
    (ohome / "benchmarks").mkdir(parents=True)
    (ohome / "evidence").mkdir(parents=True)

    # Seed miscalibrated observation + refuted provenance entry
    cal = calibration.CalibrationLedger(ohome / "benchmarks" / "calibration.jsonl")
    cal.record_observation(
        "ollama/test:7b",
        "claims-1",
        "1.000000",
        0,
        prediction_id="p1",
    )
    prov = claim_provenance.ProvenanceLedger(ohome / "evidence" / "claim_provenance.jsonl")
    prov.append(
        claim_provenance.ProvenanceRecord(
            claim_id="THETA_LOCK",
            claim_text="theta_lock = 51.843",
            source_ref="doi:10.5281/zenodo.18781261",
            sys024_verdict="REFUTED",
            run_id="r1",
        )
    )

    snap = evolution_engine.monitor_evolution(osiris_home=ohome)
    assert len(snap.dimensions) == 6
    assert {d.dimension for d in snap.dimensions} == set(evolution_engine.DIMENSIONS)

    props = evolution_engine.generate_improvements(snap, root=tmp_path)
    assert len(props) >= 4
    for p in props:
        assert p.plan_gate_passed is True
        assert p.rationale
        assert "components" in p.architecture
        assert p.implementation_approach.get("plan_version") == "1.0"
        assert "python" in p.examples and "cli" in p.examples

    rc = evolution_engine.main(["--osiris-home", str(ohome), "--json", "--record"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert "receipt_hash" in out
    assert (ohome / "evolution" / "evolution_ledger.jsonl").is_file()
