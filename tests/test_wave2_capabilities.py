"""Wave 2 regression & capability suite covering L3, L6, L7, and L8:

  * [L3 Evidence]   osiris_cli/claim_provenance.py + living.Osiris integration
  * [L6 Benchmark]  osiris_cli/calibration.py (Brier, ECE, reliability bins, decimal strings)
  * [L7 Promote]    osiris_cli/quarantine.py + evolution_bridge fail-closed gate
  * [L8 Ledger]     osiris_cli/failure_clustering.py (strictly read-only + chain integrity)
  * [/architecture] L3, L6, L7, L8 marked built with zero stubs (payload_gate)

All tests run strictly offline inside ``tmp_path`` without touching real ``~/.osiris``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
from decimal import Decimal
from pathlib import Path

import pytest

import osiris_ui
import payload_gate
from osiris_cli import (
    architecture,
    calibration,
    claim_provenance,
    failure_clustering,
    probes,
    quarantine,
)
from osiris_cli.exec_kernel import ExecJournal
from osiris_cli.living import GroundingAudit, Osiris


# ── Hygiene & Invariants (Python 3.9 AST, payload_gate, zero floats) ─────────

@pytest.mark.parametrize(
    "rel_path",
    [
        "osiris_cli/claim_provenance.py",
        "osiris_cli/calibration.py",
        "osiris_cli/quarantine.py",
        "osiris_cli/failure_clustering.py",
    ],
)
def test_wave2_modules_pass_py39_ast_and_zero_stub_payload_gate(rel_path: str) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src = (repo_root / rel_path).read_text(encoding="utf-8")
    assert "\r" not in src, f"CRLF found in {rel_path}"
    ast.parse(src, filename=rel_path, feature_version=(3, 9))
    stub_err = payload_gate.check_payload(src)
    assert stub_err is None, f"{rel_path} failed payload_gate: {stub_err}"


# ── [L3 Evidence] Claim-to-source provenance per reply ───────────────────────

def test_l3_provenance_classifies_sys024_quarantined_grounded_and_unverified(tmp_path: Path) -> None:
    doc = tmp_path / "notes.md"
    doc.write_text("Benchmark run achieved 94% pass rate on local fixture.\n", encoding="utf-8")
    doc_sha12 = hashlib.sha256(doc.read_bytes()).hexdigest()[:12]

    reply = (
        "1. The τ-phase coherence anomaly oscillates at 46 µs across shots.\n"
        "2. Setting theta_lock = 51.843 stabilizes the manifold.\n"
        "6. Locking lambda_phi = 2.176435e-08 guarantees coherence.\n"
        "3. According to notes.md, the fixture run achieved 94% pass rate.\n"
        "4. Published baseline DOI 10.5281/zenodo.18781261 documents the self-test.\n"
        "5. Our cluster latency dropped by 73 ms across 16 qubits without any source.\n"
    )
    audit = GroundingAudit(
        passed=False,
        phantom_files=(),
        unverified_numbers=("73 ms",),
        contradictions=(),
    )
    ledger_path = tmp_path / "evidence" / "claim_provenance.jsonl"
    written = claim_provenance.record_reply_provenance(
        reply,
        ledger_path=ledger_path,
        run_id="task-wave2-l3",
        exchange_hash="a" * 64,
        voice="mentor:qwen2.5-coder:7b",
        retrieved_files=[str(doc)],
        grounding_audit=audit,
        workspace_root=tmp_path,
    )
    assert len(written) >= 5
    mode = stat.S_IMODE(ledger_path.stat().st_mode)
    assert mode == 0o600

    by_verdict = {rec.sys024_verdict: rec for rec, _h in written}
    # 1. Registered SYS-024 TAU_PHASE -> ARTIFACT
    assert any(rec.claim_id == "TAU_PHASE" and rec.sys024_verdict == "ARTIFACT" for rec, _ in written)
    # 2. Registered THETA_LOCK -> REFUTED, and quarantined constant lambda_phi -> REFUTED/QUARANTINED
    assert any(rec.claim_id == "THETA_LOCK" and rec.sys024_verdict == "REFUTED" for rec, _ in written)
    assert any(rec.sys024_verdict in ("QUARANTINED", "REFUTED", "NOT_MEASURED") and "2.176435e-08" in rec.claim_text for rec, _ in written)
    # 3 & 4. Cited retrieved file + DOI -> GROUNDED with file@sha256 and doi:
    grounded_refs = [rec.source_ref for rec, _ in written if rec.sys024_verdict == "GROUNDED"]
    assert f"file:notes.md@sha256:{doc_sha12}" in grounded_refs
    assert "doi:10.5281/zenodo.18781261" in grounded_refs
    # 5. Unverified metric -> UNVERIFIED
    assert any(rec.source_ref == "UNVERIFIED" and rec.sys024_verdict == "UNVERIFIED" for rec, _ in written)

    # Verify full chain + O(1) verify_chain_tail
    pledger = claim_provenance.ProvenanceLedger(ledger_path)
    ok, count, err = pledger.verify()
    assert ok and count == len(written) and err == ""
    tail = probes.verify_chain_tail(ledger_path, tail_lines=10)
    assert tail.intact and tail.inspected_count == len(written)

    # CLI report & tamper detection
    out = claim_provenance.run_command("", osiris_home=tmp_path)
    assert f"L3 CLAIM PROVENANCE · INTACT ({len(written)} entries)" in out
    assert all(len(line) <= 96 for line in out.splitlines())

    raw = ledger_path.read_text(encoding="utf-8")
    ledger_path.write_text(raw.replace("UNVERIFIED", "GROUNDED", 1), encoding="utf-8")
    ok_after, _, _ = pledger.verify()
    assert ok_after is False
    assert probes.verify_chain_tail(ledger_path, tail_lines=10).intact is False


def test_l3_living_converse_writes_provenance_alongside_exchange_log(tmp_path: Path) -> None:
    class ScriptedMentor:
        base = "scripted://"

        def model(self) -> str:
            return "test-mentor:7b"

        def fast_model(self):
            return None

        def stream(self, messages, model=None):
            yield "Latency reached 42 ms during the synthetic run."

    liv = Osiris(
        core=None,
        mentor=ScriptedMentor(),
        home=str(tmp_path / "living"),
        out=lambda _s: None,
        background=False,
        tty=False,
    )
    reply = liv.converse("How fast was the synthetic run?", learnable=False)
    assert "42 ms" in reply
    prov_file = tmp_path / "evidence" / "claim_provenance.jsonl"
    assert prov_file.is_file()
    rows = claim_provenance.ProvenanceLedger(prov_file).entries()
    assert len(rows) >= 1
    assert rows[-1]["exchange_hash"] == liv.stats["last_head"]
    assert rows[-1]["run_id"] == f"exch-{liv.stats['last_head'][:12]}"
    assert probes.verify_chain_tail(prov_file).intact


# ── [L6 Benchmark] Calibration (Brier, ECE, reliability bins, decimal strings) ─

def test_l6_calibration_brier_ece_bins_and_decimal_string_receipts(tmp_path: Path) -> None:
    # Perfect calibration: (1.0, 1) and (0.0, 0) -> Brier=0, ECE=0
    rep_perf = calibration.compute_calibration([("1.0", 1), ("0.0", 0)], num_bins=10)
    assert rep_perf.brier_score_str == "0.000000"
    assert rep_perf.ece_str == "0.000000"
    assert rep_perf.bins[0].count == 1 and rep_perf.bins[9].count == 1

    # Maximally wrong overconfident: (1.0, 0) -> Brier=1, ECE=1
    rep_worst = calibration.compute_calibration([("1.0", 0)], num_bins=10)
    assert rep_worst.brier_score_str == "1.000000"
    assert rep_worst.ece_str == "1.000000"

    # Analytical multi-bin check:
    # Bin 7 (0.7, 0.8]: two samples p=0.8, outcomes [1, 0] -> mean_conf=0.8, acc=0.5, gap=0.3
    # Bin 1 (0.1, 0.2]: two samples p=0.2, outcomes [0, 0] -> mean_conf=0.2, acc=0.0, gap=0.2
    # ECE = (2/4)*0.3 + (2/4)*0.2 = 0.250000
    # Brier = ((0.8-1)^2 + (0.8-0)^2 + (0.2-0)^2 + (0.2-0)^2) / 4
    #       = (0.04 + 0.64 + 0.04 + 0.04) / 4 = 0.76 / 4 = 0.190000
    cal_path = tmp_path / "benchmarks" / "calibration.jsonl"
    led = calibration.CalibrationLedger(cal_path)
    pid1, _ = led.record_prediction("ollama/qwen2.5-coder:7b", "task_a", "0.8", run_id="r1")
    led.record_outcome(pid1, True, evidence_ref="sha256:" + "1" * 64, run_id="r1")
    led.record_observation("ollama/qwen2.5-coder:7b", "task_b", Decimal("0.8"), False, run_id="r1")
    led.record_observation("ollama/qwen2.5-coder:7b", "task_c", "0.2", 0, run_id="r1")
    led.record_observation("ollama/qwen2.5-coder:7b", "task_d", "0.2", 0, run_id="r1")
    # Unmatched prediction (no outcome yet) must not corrupt or enter scored denominator
    led.record_prediction("ollama/qwen2.5-coder:7b", "task_pending", "0.9", run_id="r2")

    rep, rhash = led.evaluate(model_filter="ollama/qwen2.5-coder:7b", num_bins=10, persist_receipt=True)
    assert rhash is not None
    assert rep.num_samples == 4
    assert rep.brier_score_str == "0.190000"
    assert rep.ece_str == "0.250000"
    assert rep.mean_confidence_str == "0.500000"
    assert rep.mean_accuracy_str == "0.250000"
    assert rep.overconfidence_str == "0.250000"

    # Verify zero floats exist anywhere in the persisted JSONL file (exec-canon-1)
    assert stat.S_IMODE(cal_path.stat().st_mode) == 0o600
    for raw_line in cal_path.read_text(encoding="utf-8").splitlines():
        parsed = json.loads(raw_line, parse_float=lambda s: (_ for _ in ()).throw(AssertionError(f"float in JSONL: {s}")))
        assert parsed["canon_version"] == "exec-canon-1"

    assert probes.verify_chain_tail(cal_path).intact
    detailed = calibration.run_command("ollama/qwen2.5-coder:7b", osiris_home=tmp_path)
    assert "Brier score: 0.190000" in detailed and "ECE (10 bins): 0.250000" in detailed
    assert all(len(line) <= 96 for line in detailed.splitlines())

    with pytest.raises(calibration.CalibrationError):
        calibration.parse_probability("1.05")
    with pytest.raises(calibration.CalibrationError):
        calibration.parse_probability("-0.01")
    with pytest.raises(calibration.CalibrationError):
        calibration.parse_probability(True)


# ── [L7 Promote] Fail-closed quarantine queue ────────────────────────────────

def test_l7_quarantine_blocks_promotion_forbids_agent_resolve_and_fails_closed_on_tamper(tmp_path: Path) -> None:
    q_path = tmp_path / "promote" / "quarantine_queue.jsonl"
    q = quarantine.QuarantineQueue(q_path)

    # Clean empty state permits non-quarantined candidates
    assert q.check_promotion_allowed(candidate_digest="d" * 64, target_path="osiris_cli/foo.py").allowed is True

    rec, _ = q.enqueue(
        "cand_deadbeef1234",
        "d" * 64,
        ["epistemic firewall QUARANTINE: unbacked speed claim"],
        target_paths=["osiris_cli/foo.py"],
        run_id="run-q1",
        metrics={"candidate_fitness": 0.75},
    )
    assert stat.S_IMODE(q_path.stat().st_mode) == 0o600
    assert probes.verify_chain_tail(q_path).intact

    # Blocked by digest AND by overlapping target_path
    v_digest = q.check_promotion_allowed(candidate_digest="d" * 64)
    assert v_digest.allowed is False and v_digest.status == "BLOCKED_QUARANTINED"
    v_path = q.check_promotion_allowed(candidate_digest="e" * 64, target_path="osiris_cli/foo.py")
    assert v_path.allowed is False and v_path.status == "BLOCKED_QUARANTINED"

    # Agent origins cannot resolve quarantine items (REQ-AUTH-5)
    for bad_origin in ("model", "skill", "claude_code"):
        with pytest.raises(quarantine.QuarantineError):
            q.resolve(rec.quarantine_id, "APPROVED", note="auto-approve", origin=bad_origin)

    # Human CLI resolution -> APPROVED releases quarantine hold
    cli_out = quarantine.run_command(
        f"resolve {rec.quarantine_id} approve 'verified benchmark log manually'",
        osiris_home=tmp_path,
        origin="human",
    )
    assert "-> APPROVED" in cli_out
    v_released = q.check_promotion_allowed(candidate_digest="d" * 64, target_path="osiris_cli/foo.py")
    assert v_released.allowed is True and v_released.status == "RELEASED"

    # Enqueue a second item and resolve REJECTED -> digest stays blocked permanently, target_path unblocks
    rec2, _ = q.enqueue("cand_bad2", "f" * 64, ["flaky timeout"], target_paths=["osiris_cli/bar.py"])
    q.resolve(rec2.quarantine_id, "REJECTED", note="confirmed regression", origin="human")
    assert q.check_promotion_allowed(candidate_digest="f" * 64).allowed is False
    assert q.check_promotion_allowed(candidate_digest="f" * 64).status == "BLOCKED_REJECTED"
    assert q.check_promotion_allowed(candidate_digest="a" * 64, target_path="osiris_cli/bar.py").allowed is True

    # Tamper 1 byte in quarantine_queue.jsonl -> fail-closed across ALL candidates
    corrupted = q_path.read_text(encoding="utf-8").replace("flaky timeout", "clean pass   ", 1)
    q_path.write_text(corrupted, encoding="utf-8")
    v_tamper = q.check_promotion_allowed(candidate_digest="9" * 64, target_path="unrelated.py")
    assert v_tamper.allowed is False and v_tamper.status == "FAIL_CLOSED_LEDGER_TAMPERED"


def test_l7_evolution_bridge_enqueues_inconclusive_and_blocks_until_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import genome_ledger
    from osiris_cli.orchestration import evolution_bridge, trio
    from osiris_cli.orchestration.evidence_firewall import ExecutionEvidence

    fixture_ledger = Path(__file__).parent / "fixtures" / "dnalang_ledger_v0_2_0.py"
    if not os.path.exists(genome_ledger.LEDGER_SRC) and fixture_ledger.is_file():
        monkeypatch.setattr(genome_ledger, "LEDGER_SRC", str(fixture_ledger))
        monkeypatch.setattr(genome_ledger, "_ledger_mod", None)

    lineage = tmp_path / "lineage.jsonl"
    q_file = tmp_path / "quarantine_queue.jsonl"
    run_dir = tmp_path / "trio_run"
    run_dir.mkdir()

    ev_base = ExecutionEvidence(
        argv=("pytest",), returncode=1, passed=1, failed=1, errors=0,
        stdout_digest="1" * 64, stdout_tail="2 passed", workdir_digest="b" * 64,
    )
    ev_cand = ExecutionEvidence(
        argv=("pytest",), returncode=0, passed=3, failed=0, errors=0,
        stdout_digest="2" * 64, stdout_tail="3 passed", workdir_digest="c" * 64,
    )
    proposed = {"mod.py": "def compute(x):\n    return x + 1\n"}

    # Step 1: Epistemic firewall returns QUARANTINE -> candidate is auto-enqueued in quarantine_queue.jsonl
    res_inconclusive = trio.TrioResult(
        run_id="trio-q1",
        run_dir=str(run_dir),
        outcome="QUARANTINED",
        reasons=["firewall quarantined ambiguous claim"],
        proposed_files=proposed,
        patched_files=proposed,
        baseline=ev_base,
        candidate=ev_cand,
        candidate_status="MEASURED",
        critic=trio.CriticVerdict(approve=True, reasons=(), fail_to_pass=("t1",), pass_to_fail=(), supplementary=""),
        firewall=trio.FirewallResult(),
        receipts=[],
    )
    dec1 = evolution_bridge.evaluate_candidate(
        res_inconclusive, str(lineage), quarantine_ledger=str(q_file)
    )
    assert dec1.status == "QUARANTINED"
    q = quarantine.QuarantineQueue(q_file)
    active = q.list_items(active_only=True)
    assert len(active) == 1 and active[0].candidate_digest == dec1.candidate_digest

    # Step 2: Re-evaluating the exact same candidate_digest (even with ACCEPT + authorizer=True) is BLOCKED
    res_ready = trio.TrioResult(
        run_id="trio-q2",
        run_dir=str(run_dir),
        outcome="PENDING_HUMAN_APPROVAL",
        reasons=[],
        proposed_files=proposed,
        patched_files=proposed,
        baseline=ev_base,
        candidate=ev_cand,
        candidate_status="MEASURED",
        critic=trio.CriticVerdict(approve=True, reasons=(), fail_to_pass=("t1",), pass_to_fail=(), supplementary=""),
        firewall=trio.FirewallResult(),
        receipts=[],
    )
    dec_blocked = evolution_bridge.evaluate_candidate(
        res_ready,
        str(lineage),
        authorizer=lambda _d, _s: (True, "owner-grant-ok"),
        quarantine_ledger=str(q_file),
    )
    assert dec_blocked.status == "QUARANTINED"
    assert "quarantined" in dec_blocked.reasons[0]

    # Step 3: Explicit human resolution releases the hold -> promotion now succeeds with authorizer
    q.resolve(active[0].quarantine_id, "APPROVED", note="human reviewed diff & tests", origin="human")
    dec_promoted = evolution_bridge.evaluate_candidate(
        res_ready,
        str(lineage),
        authorizer=lambda _d, _s: (True, "owner-grant-ok"),
        quarantine_ledger=str(q_file),
    )
    assert dec_promoted.status == "PROMOTED"


# ── [L8 Ledger] Read-only failure-pattern clustering & chain integrity ───────

def test_l8_failure_clustering_is_strictly_read_only_verifies_chains_and_clusters(tmp_path: Path) -> None:
    home = tmp_path / "osiris_home"
    (home / "living").mkdir(parents=True)
    (home / "tasks").mkdir(parents=True)
    (home / "runs" / "run-001").mkdir(parents=True)
    (home / "code" / "runs").mkdir(parents=True)

    # 1. Valid hash-chained exchanges.jsonl with 2 grounding audit failures
    ex_path = home / "living" / "exchanges.jsonl"
    prev = "0" * 64
    ex_rows = []
    for idx, ts in enumerate(("2026-11-20T10:00:00", "2026-11-20T11:00:00")):
        entry = {
            "prev": prev,
            "t": ts,
            "voice": "mentor:qwen",
            "user": "q",
            "reply": "r",
            "index": idx,
            "grounding_audit": {
                "passed": False,
                "phantom_files": ["ghost.py"],
                "unverified_numbers": [],
                "contradictions": [],
            },
        }
        body = json.dumps(entry, sort_keys=True, ensure_ascii=False)
        h = hashlib.sha256(body.encode("utf-8")).hexdigest()
        entry["hash"] = h
        prev = h
        ex_rows.append(json.dumps(entry, sort_keys=True))
    ex_path.write_text("\n".join(ex_rows) + "\n", encoding="utf-8")

    # 2. Valid exec-canon-1 exec_journal.jsonl with 3 policy denials
    ej_path = home / "exec_journal.jsonl"
    ej = ExecJournal(ej_path)
    for idx in range(3):
        ej.append({
            "argv": ["rm", "-rf", f"x{idx}"],
            "decision": "DENY",
            "exit_code": None,
            "reasons": ["execution policy refused destructive rm inosiris_cli/code_agent.py"],
            "ts": f"2026-11-20T12:0{idx}:00Z",
        })

    # 3. Sprint run record & /code run record
    (home / "runs" / "run-001" / "run.json").write_text(
        json.dumps({
            "started_at": "2026-11-20T13:00:00Z",
            "target": "osiris_cli/sprint_manager.py",
            "attempts": [
                {"stage": "ast_firewall", "verdict": "fail", "reason": "AST firewall disallowed subprocess import"},
                {"stage": "verify", "verdict": "pass", "reason": "ok"},
            ],
        }),
        encoding="utf-8",
    )
    (home / "code" / "runs" / "code-abc123def456.json").write_text(
        json.dumps({
            "status": "STOPPED_BUDGET",
            "finished_at": "2026-11-20T14:00:00Z",
            "stop_reason": "STOPPED_BUDGET: budget cap exceeded in osiris_cli/code_agent.py",
        }),
        encoding="utf-8",
    )

    # Snapshot exact bytes & mtimes BEFORE clustering (REQ-EV-4 read-only proof)
    tracked_files = [
        ex_path,
        ej_path,
        home / "runs" / "run-001" / "run.json",
        home / "code" / "runs" / "code-abc123def456.json",
    ]
    before_snapshots = {
        p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in tracked_files
    }

    report = failure_clustering.analyze_failure_clusters(osiris_home=home)
    after_snapshots = {
        p: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in tracked_files
    }
    assert before_snapshots == after_snapshots, "failure_clustering mutated a ledger or run file!"

    assert report.all_chains_intact is True
    assert report.total_failures == 7
    assert len(report.clusters) == 4
    # Top cluster is the 3x policy denial, second is 2x grounding audit failure
    assert report.clusters[0].count == 3 and report.clusters[0].root_cause_tag == "exec_policy_denied"
    assert report.clusters[1].count == 2 and report.clusters[1].root_cause_tag == "grounding_audit_failure"

    rendered = failure_clustering.format_clusters(report)
    assert "chains INTACT" in rendered
    assert all(len(line) <= 96 for line in rendered.splitlines())

    # Tamper exchanges.jsonl -> report flags TAMPERED without modifying/repairing the file
    tampered_bytes = ex_path.read_text(encoding="utf-8").replace("ghost.py", "other.py", 1)
    ex_path.write_text(tampered_bytes, encoding="utf-8")
    tampered_sha = hashlib.sha256(ex_path.read_bytes()).hexdigest()

    bad_rep = failure_clustering.analyze_failure_clusters(osiris_home=home)
    assert bad_rep.all_chains_intact is False
    ex_status = [s for s in bad_rep.ledger_integrity if s.name == "exchanges.jsonl"][0]
    assert ex_status.intact is False and ex_status.status == "TAMPERED"
    assert hashlib.sha256(ex_path.read_bytes()).hexdigest() == tampered_sha


# ── [/architecture] Wave 2 registry honesty check ────────────────────────────

def test_wave2_architecture_marks_l3_l6_l7_l8_built_and_keeps_wave3_wave4_honest() -> None:
    by_key = {ly.key: ly for ly in architecture.LAYERS}
    for key in ("L0", "L1", "L2", "L3", "L5", "L6", "L7", "L8"):
        assert by_key[key].missing == (), f"{key} still has missing items: {by_key[key].missing}"
    # Wave 3 (L4) remains honestly listed as not built until Wave 3 runs
    assert by_key["L4"].missing == ("baseline, alternative and conservative candidates side by side",)

    state = architecture.live_state(None, None)
    text_narrow = architecture.full(osiris_ui.Canvas("plain", 80), state)
    assert "osiris_cli/claim_provenance.py" in text_narrow
    assert "osiris_cli/calibration.py" in text_narrow
    assert "osiris_cli/quarantine.py" in text_narrow
    assert "osiris_cli/failure_clustering.py" in text_narrow
