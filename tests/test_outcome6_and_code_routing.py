"""Offline verification tests for:
1. Outcome 6 (`osiris_cli.reproducible_eval` — PWS Section 1.2.6 protocol verifier,
   hash-chained ledger integrity, null-outcome nonperformance protection, and annex overclaim checks).
2. Claims register entries (`QKERNEL_GENOMICS_BENCHMARK`, `TESSERACT_MAP_02_COMPILATION`,
   `CORRELATED_NOISE_01_STAGE1`) and `verify_register` numerical provenance against
   `docs/phase3/PWS_OUTCOME6_AND_PHASE3_CONSOLIDATION.md`.
3. REPL code-authoring paste routing (`write the code for "<paste>"`, `/code "<paste>"`, `/eval-audit`).
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch

from osiris_cli import claims, code_agent, living, reproducible_eval
from osiris_cli.osiris_repl import OsirisReplState, dispatch_command, handle_livlm_prompt

REPO_ROOT = Path(__file__).resolve().parents[1]


def _append_ledger_entry(lines: list[str], prev: str, kind: str, payload: dict) -> str:
    body = {"ts": "2026-11-16T23:00:00Z", "kind": kind, **payload}
    canon_body = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256((prev + canon_body).encode("utf-8")).hexdigest()
    full = {"prev": prev, **body, "hash": digest}
    lines.append(json.dumps(full, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return digest


def _build_qkernel_fixture(run_dir: Path, *, tamper_margin: bool = False) -> Path:
    (run_dir / "circuits").mkdir(parents=True, exist_ok=True)
    (run_dir / "kernels").mkdir(parents=True, exist_ok=True)
    (run_dir / "circuits" / "fidelity_product.qasm").write_text("OPENQASM 3.0;\n", encoding="utf-8")
    (run_dir / "kernels" / "q_product_exact_fold0.npz").write_bytes(b"npz-placeholder")

    cfg = {
        "n_components": 6,
        "reps": 2,
        "outer_folds": 2,
        "inner_folds": 2,
        "margin": 0.02,
    }
    verdict = "H0 not rejected: no evidence that the quantum kernel beats the best classical model"
    lines: list[str] = []
    prev = reproducible_eval.GENESIS_HASH
    prev = _append_ledger_entry(
        lines,
        prev,
        "qkernel.preregister",
        {
            "dataset": "SYNTHETIC Balding-Nichols, 4 populations x 50, 2000 SNPs, F_ST=0.004",
            "dataset_sha256": "a" * 64,
            "config": cfg,
            "H0": "best quantum balanced accuracy - best classical balanced accuracy <= 0.02",
            "decision_rule": "reject H0 only if the paired-bootstrap 95% CI lower bound exceeds the margin",
        },
    )
    prev = _append_ledger_entry(
        lines,
        prev,
        "qkernel.circuit",
        {"family": "product", "agree": True, "n_qubits": 6, "depth": 13, "two_qubit_gates": 0},
    )
    prev = _append_ledger_entry(
        lines,
        prev,
        "qkernel.fold",
        {
            "fold": 0,
            "model": "q_product_exact",
            "train_ids_sha256": "1" * 64,
            "kernel_file": "kernels/q_product_exact_fold0.npz",
            "test_balanced_accuracy": 0.55,
        },
    )
    prev = _append_ledger_entry(
        lines,
        prev,
        "qkernel.fold",
        {
            "fold": 1,
            "model": "q_product_exact",
            "train_ids_sha256": "2" * 64,
            "test_balanced_accuracy": 0.54,
        },
    )
    _append_ledger_entry(
        lines,
        prev,
        "qkernel.result",
        {"verdict": verdict, "comparison": {"margin": 0.02}},
    )
    (run_dir / "ledger.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    results = {
        "dataset": "SYNTHETIC Balding-Nichols, 4 populations x 50, 2000 SNPs, F_ST=0.004",
        "dataset_sha256": "a" * 64,
        "n": 200,
        "config": cfg,
        "verdict": verdict,
        "models": {
            "logistic": {"balanced_accuracy": 0.595, "quantum": False},
            "q_product_exact": {"balanced_accuracy": 0.545, "quantum": True},
        },
        "comparison": {
            "best_classical": "logistic",
            "best_classical_ba": 0.595,
            "best_quantum": "q_product_exact",
            "best_quantum_ba": 0.545,
            "diff": -0.050,
            "ci95": [-0.117, 0.018],
            "margin": -0.20 if tamper_margin else 0.02,
        },
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (run_dir / "report.md").write_text(
        "# Quantum-kernel genomics benchmark: SYNTHETIC Balding-Nichols\n\n"
        "- Hardware was not used. Simulator evaluation only.\n",
        encoding="utf-8",
    )
    return run_dir


def test_outcome6_null_qkernel_package_completes_protocol_without_nonperformance(tmp_path):
    pkg = _build_qkernel_fixture(tmp_path / "v2-synthetic-fst0.004")
    rep = reproducible_eval.evaluate_qkernel_package(pkg)
    assert rep.protocol_completed is True
    assert rep.criterion_met is False
    # Failure to beat classical baselines is NOT contractor nonperformance when protocol is complete
    assert rep.contractor_nonperformance is False
    assert rep.data_origin == "SYNTHETIC"
    assert rep.execution_plane == "SIMULATOR"
    assert rep.candidate_entry is not None
    assert rep.candidate_entry.record_standing == "CANDIDATE_REPORTED"
    md = reproducible_eval.render_candidate_evidence_markdown(rep.candidate_entry)
    assert "CANDIDATE_REPORTED" in md
    assert "protocol_completed=True" in md and "criterion_met=False" in md


def test_outcome6_detects_posthoc_acceptance_rule_tampering_and_ledger_corruption(tmp_path):
    tampered = _build_qkernel_fixture(tmp_path / "tampered-rule", tamper_margin=True)
    rep = reproducible_eval.evaluate_qkernel_package(tampered)
    assert rep.protocol_completed is False
    assert rep.contractor_nonperformance is True
    assert any("acceptance rule" in i for i in rep.issues)

    # Corrupting a ledger line breaks 1.2.6-4 hash-chain verification
    corrupted_ledger = tmp_path / "tampered-rule" / "ledger.jsonl"
    corrupted_ledger.write_text(
        corrupted_ledger.read_text(encoding="utf-8").replace("0.55", "0.99"),
        encoding="utf-8",
    )
    ok, _, msg = reproducible_eval.verify_hash_chained_ledger(corrupted_ledger)
    assert ok is False and "hash mismatch" in msg


def test_outcome6_phase3_hardware_package_audit(tmp_path):
    lock_path = REPO_ROOT / "docs" / "phase3" / "correlated_noise_01_live_prereg_lock.json"
    lock_bytes = lock_path.read_bytes()
    lock_sha = hashlib.sha256(lock_bytes).hexdigest()
    lock_data = json.loads(lock_bytes.decode("utf-8"))

    analysis_path = tmp_path / "correlated_noise_01_live_analysis.json"
    analysis_path.write_text(
        json.dumps(
            {
                "experiment_id": "CORRELATED-NOISE-01-STAGE1",
                "status": "Hardware-observed",
                "job_id": "dbtp67dav06s73cfbsm0",
                "backend": "ibm_marrakesh",
                "n_blocks": 5,
                "prereg_lock_sha256": lock_sha,
                "qpy_sha256": lock_data["qpy_sha256"],
                "gls_model_selection_on_training_cohort": {
                    "selected_model": "H3",
                    "fits": {"H0": {}, "H1": {}, "H2": {}, "H3": {}, "Apparatus": {}},
                },
                "held_out_6coupler_cluster_validation_df5": {
                    "n_held_out_couplers": 6,
                    "p_two_sided_df5": 0.8214,
                    "transfer_beats_h0_p_0_05": False,
                },
            }
        ),
        encoding="utf-8",
    )
    a3_path = tmp_path / "correlated_noise_01_live_analysis_A3.json"
    a3_path.write_text(json.dumps({"model_selection": {"selected_model": "H0"}}), encoding="utf-8")

    rep = reproducible_eval.evaluate_phase3_hardware_package(
        lock_path, analysis_path, a3_analysis_path=a3_path
    )
    assert rep.protocol_completed is True
    assert rep.criterion_met is False
    assert rep.contractor_nonperformance is False
    assert rep.execution_plane == "PHYSICAL_HARDWARE"
    assert rep.data_origin == "REAL_OBSERVATIONAL"
    assert rep.candidate_entry is not None
    matrix = reproducible_eval.render_side_by_side_matrix([rep.candidate_entry])
    assert "PHYSICAL_HARDWARE" in matrix and "CORRELATED-NOISE-01-STAGE1" in matrix


def test_outcome6_annex_language_guardrails():
    good = (
        "Implemented and executed a simulator-based quantum-kernel evaluation package comparing "
        "three circuit feature maps against classical classifiers on two synthetic genotype datasets. "
        "Neither setting satisfied the recorded quantum-advantage criterion. No physical quantum-computing "
        "validation was performed."
    )
    assert reproducible_eval.check_annex_language(good) == []

    bad = (
        "This study demonstrated quantum advantage for genomics, demonstrated real-world population "
        "classification, provided physical quantum-computing validation, proves that entanglement "
        "provides no benefit, and confirms that kernel concentration caused the lower accuracy."
    )
    flags = reproducible_eval.check_annex_language(bad)
    assert len(flags) == 5


def test_new_claims_register_entries_and_firewall_provenance():
    ev_doc = (REPO_ROOT / "docs" / "phase3" / "PWS_OUTCOME6_AND_PHASE3_CONSOLIDATION.md").read_text(
        encoding="utf-8"
    )
    for cid, expected_verdict in (
        ("QKERNEL_GENOMICS_BENCHMARK", "NULL"),
        ("TESSERACT_MAP_02_COMPILATION", "REPORTED"),
        ("CORRELATED_NOISE_01_STAGE1", "NULL"),
    ):
        claim = claims.by_id(cid)
        assert claim is not None and claim.verdict == expected_verdict
        assert claims.verify_register(cid, ev_doc) == {"statement": [], "scoped_wording": []}, cid

    # Ensure TESSERACT-MAP-02 still never triggers ERROR_SUPPRESSION_1E6
    hits = {c.id for c in claims.match("TESSERACT-MAP-02 job dbto408n26qs739tnrj0")}
    assert "TESSERACT_MAP_02_COMPILATION" in hits
    assert "ERROR_SUPPRESSION_1E6" not in hits


def test_repl_code_directive_paste_and_quoted_spec_normalization(monkeypatch, tmp_path):
    long_spec = (
        "Candidate evidence entry: Quantum-kernel genomics evaluation\n"
        + ("Specification line detailing comparative baseline protocol.\n" * 30)
    )
    pasted_request = f'write the code for "\n{long_spec}\n"'

    # 1. living.asks_something recognizes code-authoring requests (never says 'You didn't say what to do')
    assert living.asks_something(pasted_request) is True
    extracted = living.extract_code_request(pasted_request)
    assert extracted is not None
    assert extracted.lower().startswith("write the code for:")
    assert "Candidate evidence entry" in extracted

    # Ordinary prose starting with 'code review ...' does NOT hijack to extract_code_request
    assert living.extract_code_request("code review my parser please") is None

    # 2. handle_livlm_prompt routes explicit 'write the code for "<paste>"' to code_agent.run_code_command
    routed: list[str] = []
    monkeypatch.setattr(code_agent, "run_code_command", lambda state, rest: routed.append(rest))
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        handle_livlm_prompt(OsirisReplState(), pasted_request)
    assert len(routed) == 1
    assert "Candidate evidence entry" in routed[0]
    assert "routing to governed /code loop" in buf.getvalue()

    # 3. /code "<multi-line spec with no leading verb>" frames task for tool inspection + write_file
    framed = code_agent.normalize_code_task(f'"\n{long_spec}\n"')
    assert framed.startswith("[Task from pasted specification:")
    assert "write_file" in framed
    # Short tasks pass through untouched
    assert code_agent.normalize_code_task("list the tests") == "list the tests"

    # 4. /eval-audit command is wired in REPL dispatch
    pkg = _build_qkernel_fixture(tmp_path / "qk-repl")
    audit_buf = io.StringIO()
    with patch("sys.stdout", audit_buf):
        dispatch_command(OsirisReplState(), f"/eval-audit qkernel {pkg}")
    assert "protocol_completed=True" in audit_buf.getvalue()
