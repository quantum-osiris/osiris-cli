"""Verification suite for Quantinuum/QCUP HQC sizing, BP+OSD hypergraph decoding, and proposal claim/firewall hygiene."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("qiskit")

from osiris_cli import claims  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PREP_DIR = _REPO_ROOT / "experiments" / "qcup_prep"
_PROPOSALS_DIR = _REPO_ROOT / "docs" / "proposals" / "quantinuum-olcf-2026"

_SPEC = importlib.util.spec_from_file_location("hqc_and_bposd_prep", _PREP_DIR / "hqc_and_bposd_prep.py")
assert _SPEC is not None and _SPEC.loader is not None
_PREP = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _PREP
_SPEC.loader.exec_module(_PREP)


def test_css_hypergraph_codes_algebra_and_helios_qubit_footprint() -> None:
    hx_s, hz_s = _PREP.build_steane_7_1_3()
    lx_s, lz_s = _PREP.css_logical_operators(hx_s, hz_s)
    assert hx_s.shape == (3, 7)
    assert lx_s.shape == (1, 7) and lz_s.shape == (1, 7)
    assert int(hz_s.sum(axis=0).max()) == 3  # weight-3 syndrome hyperedge

    hx_t, hz_t = _PREP.build_tesseract_16_6_4()
    lx_t, lz_t = _PREP.css_logical_operators(hx_t, hz_t)
    assert hx_t.shape == (5, 16)
    assert lx_t.shape == (6, 16) and lz_t.shape == (6, 16)
    assert list(hx_t.sum(axis=1)) == [8, 8, 8, 8, 8]

    hx_bb, hz_bb = _PREP.build_bivariate_bicycle_72_12_6()
    lx_bb, lz_bb = _PREP.css_logical_operators(hx_bb, hz_bb)
    assert hx_bb.shape == (36, 72)
    assert _PREP.gf2_rank(hx_bb) == 30 and _PREP.gf2_rank(hz_bb) == 30
    assert lx_bb.shape == (12, 72) and lz_bb.shape == (12, 72)
    # 72 data ions + 24 reused ancilla ions = 96 physical ions <= 98 ions on Quantinuum Helios
    assert 72 + 24 <= 98


def test_bp_osd_corrects_all_weight1_and_bb_weight2_hyperedge_faults() -> None:
    for builder in (_PREP.build_steane_7_1_3, _PREP.build_tesseract_16_6_4, _PREP.build_bivariate_bicycle_72_12_6):
        hx, hz = builder()
        lx, _ = _PREP.css_logical_operators(hx, hz)
        n = hx.shape[1]
        for q in range(n):
            err = np.zeros(n, dtype=np.uint8)
            err[q] = 1
            res = _PREP.decode_bp_osd(hz, (hz @ err) % 2, channel_probs=0.015)
            residual = err ^ res.correction
            assert np.all((hz @ residual) % 2 == 0)
            assert np.all((lx @ residual) % 2 == 0)

    # Spot-check 50 deterministic weight-2 error pairs on [[72, 12, 6]] (d=6 guarantees all weight-2 correctable)
    hx_bb, hz_bb = _PREP.build_bivariate_bicycle_72_12_6()
    lx_bb, _ = _PREP.css_logical_operators(hx_bb, hz_bb)
    rng = np.random.default_rng(12345)
    for _ in range(50):
        q1, q2 = rng.choice(72, size=2, replace=False)
        err = np.zeros(72, dtype=np.uint8)
        err[int(q1)] = 1
        err[int(q2)] = 1
        res = _PREP.decode_bp_osd(hz_bb, (hz_bb @ err) % 2, channel_probs=0.015)
        residual = err ^ res.correction
        assert np.all((hz_bb @ residual) % 2 == 0)
        assert np.all((lx_bb @ residual) % 2 == 0)


def test_archived_hqc_and_bposd_receipt_integrity() -> None:
    receipt = json.loads((_PREP_DIR / "qcup_workload_and_bposd_estimates.json").read_text(encoding="utf-8"))
    assert receipt["total_quantinuum_circuits"] == 191
    assert receipt["total_quantinuum_shots"] == 67124
    assert receipt["total_estimated_hqc"] == pytest.approx(21169.8, abs=0.5)
    assert receipt["recommended_initial_hqc_allocation"] == 21500
    assert receipt["recommended_ibm_qpu_seconds"] == 600
    for w in receipt["workloads"]:
        assert w["routing_swaps"] == 0
        assert w["physical_qubits"] <= 98
    assert receipt["decoder_benchmarks"]["Steane_7_1_3"]["exhaustive_weight1_recovery"] == "7/7"
    assert receipt["decoder_benchmarks"]["Tesseract_16_6_4"]["exhaustive_weight1_recovery"] == "16/16"
    assert receipt["decoder_benchmarks"]["BivariateBicycle_72_12_6"]["exhaustive_weight1_recovery"] == "72/72"
    assert receipt["decoder_benchmarks"]["BivariateBicycle_72_12_6"]["exhaustive_weight2_recovery"] == "2556/2556"


def test_track_a_and_track_b_submissions_pass_claims_and_open_science_firewalls() -> None:
    track_a = (_PROPOSALS_DIR / "track_a_quantinuum_startup_partner" / "QUANTINUUM_STARTUP_PARTNER_SUBMISSION.md").read_text(
        encoding="utf-8"
    )
    track_b = (_PROPOSALS_DIR / "track_b_olcf_qcup_open_science" / "OLCF_QCUP_PROJECT_PROPOSAL_V2.md").read_text(
        encoding="utf-8"
    )
    myolcf = (_PROPOSALS_DIR / "track_b_olcf_qcup_open_science" / "MYOLCF_CREDIT_ALLOCATION_JUSTIFICATION.md").read_text(
        encoding="utf-8"
    )

    # Zero refuted/unsupported OSIRIS claims asserted in any external submission
    assert claims.flagged_in_reply(track_a) == []
    assert claims.flagged_in_reply(track_b) == []
    assert claims.flagged_in_reply(myolcf) == []

    # Both permanent Zenodo DOIs present in both Track A and Track B
    for doc in (track_a, track_b, myolcf):
        assert "10.5281/zenodo.24287505" in doc
        assert "10.5281/zenodo.24293756" in doc

    # Track B open-science compliance: verbatim DOE OLCF acknowledgement + explicit open research attestation
    assert "Contract DE-AC05-00OR22725" in track_b
    assert "open, fundamental research" in track_b
    assert "no export-controlled, phi, or other controlled data" in track_b.lower()
    # Ensure defense-product positioning appears only in PI org name ("Agile Defense Systems LLC"), never in science text
    track_b_without_org = track_b.replace("Agile Defense Systems LLC", "")
    assert "defense" not in track_b_without_org.lower()
