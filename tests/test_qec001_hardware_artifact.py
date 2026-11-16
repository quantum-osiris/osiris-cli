"""Forensic verification of archived OSIRIS-QEC-001 physical hardware dataset (`ibm_fez`, Job `dbtl2vb2kf8s73c9a0d0`)."""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("qiskit")

from qiskit import qpy  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
_QEC_DIR = _REPO_ROOT / "experiments" / "qec001"

_SPEC = importlib.util.spec_from_file_location(
    "qec001_repetition_scaling", _QEC_DIR / "qec001_repetition_scaling.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_QEC = importlib.util.module_from_spec(_SPEC)
sys.modules["qec001_repetition_scaling"] = _QEC
_SPEC.loader.exec_module(_QEC)


def test_qec001_preregistration_checksums_and_concentric_subchain_isolation() -> None:
    manifest = json.loads((_QEC_DIR / "qec001_fez_staging_manifest.json").read_text(encoding="utf-8"))
    submitted = json.loads((_QEC_DIR / "qec001_fez_submitted_job.json").read_text(encoding="utf-8"))
    results = json.loads((_QEC_DIR / "physical_qec001_fez_results.json").read_text(encoding="utf-8"))

    qpy_sha = hashlib.sha256((_QEC_DIR / "qec001_fez_staged_circuits.qpy").read_bytes()).hexdigest()
    raw_gz_sha = hashlib.sha256((_QEC_DIR / "physical_qec001_fez_raw_counts.json.gz").read_bytes()).hexdigest()

    assert qpy_sha == "357a29ad2746e2be94522702516dce3e04319894c1dc981596c9a3640041c056"
    assert raw_gz_sha == "cdae697c364e2aad2085cc3585039d079baf4b3d1bb619dcb6178647b75646c1"
    assert manifest["qpy_sha256"] == qpy_sha
    assert submitted["qpy_sha256"] == qpy_sha
    assert results["qpy_sha256"] == qpy_sha
    assert results["raw_counts_archive_sha256"] == raw_gz_sha
    assert results["job_id"] == "dbtl2vb2kf8s73c9a0d0"
    assert results["backend"] == "ibm_fez"

    t_staged = dt.datetime.fromisoformat(manifest["staged_at_utc"].replace("Z", "+00:00"))
    t_created = dt.datetime.fromisoformat(
        results["job_metrics"]["timestamps"]["created"].replace("Z", "+00:00")
    )
    assert t_staged < t_created

    with open(_QEC_DIR / "qec001_fez_staged_circuits.qpy", "rb") as fh:
        circuits = qpy.load(fh)
    assert len(circuits) == 192

    for qc, meta in zip(circuits, manifest["circuit_metadata"]):
        expected_sub = set(meta["pinned_subchain"])
        used = {qc.find_bit(qb).index for inst in qc.data for qb in inst.qubits}
        assert used == expected_sub


def test_qec001_raw_counts_reproduce_mwpm_suppression_and_hypotheses() -> None:
    results = json.loads((_QEC_DIR / "physical_qec001_fez_results.json").read_text(encoding="utf-8"))
    cal_snap = json.loads((_QEC_DIR / "qec001_fez_calibration_snapshot.json").read_text(encoding="utf-8"))
    with gzip.open(_QEC_DIR / "physical_qec001_fez_raw_counts.json.gz", "rt", encoding="utf-8") as gzf:
        raw_records = json.load(gzf)

    assert len(raw_records) == 192
    assert sum(int(r["shots"]) for r in raw_records) == 393_216

    # Recompute exact Blossom MWPM on the R=7 endpoints (d=3, 5, 7 for both bit_flip_z and phase_flip_x)
    # directly from the archived raw hardware bitstrings and verify 1e-12 match to reported results.
    priors = cal_snap["decoder_priors"]
    decoders = {
        d: _QEC.build_mwpm_decoder(
            distance=d,
            rounds=7,
            p_data=float(priors[f"d{d}"]["p_data"]),
            p_meas=float(priors[f"d{d}"]["p_meas"]),
            p_hook=float(priors[f"d{d}"]["p_hook"]),
        )
        for d in (3, 5, 7)
    }

    r7_records = [r for r in raw_records if int(r["rounds"]) == 7]
    assert len(r7_records) == 48

    accum = {(b, d): {"shots": 0, "mwpm_err": 0, "maj_err": 0} for b in _QEC.CODE_BASES for d in (3, 5, 7)}
    for rec in r7_records:
        b = str(rec["code_basis"])
        d = int(rec["distance"])
        s = int(rec["logical_state"])
        ev = _QEC.evaluate_repetition_counts(rec["raw_counts"], distance=d, rounds=7, logical_state=s, decoder=decoders[d])
        accum[(b, d)]["shots"] += int(rec["shots"])
        accum[(b, d)]["mwpm_err"] += int(round(ev["mwpm_logical_errors"]))
        accum[(b, d)]["maj_err"] += int(round(ev["p_logical_majority"] * int(rec["shots"])))

    for (b, d), vals in accum.items():
        rep = results["grid_summary"][b][f"d{d}"]["R7"]
        assert vals["shots"] == 16_384
        assert vals["mwpm_err"] == rep["mwpm_logical_errors"]
        assert vals["maj_err"] == rep["majority_logical_errors"]
        assert (vals["mwpm_err"] / 16_384.0) == pytest.approx(rep["p_logical_mwpm"], abs=1e-12)

    # Verify monotonic sub-threshold suppression across all 8 (basis, R) curves
    for b in _QEC.CODE_BASES:
        for rk in ("R1", "R3", "R5", "R7"):
            sup = results["suppression_observed"][b][rk]
            assert sup["monotonic_suppression"] is True
            assert sup["lambda_3_to_5"] > 3.0
            assert sup["lambda_3_to_7"] > 3.7

    hyps = results["preregistered_hypotheses"]
    assert hyps["QEC_H1_bit_flip_subthreshold_monotonic_suppression"]["passed"] is True
    assert hyps["QEC_H2_phase_flip_midcircuit_dephasing_diagnostic"]["subthreshold_passed"] is True
    # QEC-H3 preserved un-doctored: MWPM beat majority by 3x-7.2x, while phase_flip_x R=7 defect rate ~0.127 > 0.12
    assert hyps["QEC_H3_spacetime_mwpm_advantage_and_defect_ceiling"]["passed"] is False
