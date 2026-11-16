#!/usr/bin/env python3
"""
publish_qec001_to_zenodo.py — Uploads and publishes the standalone OSIRIS-QEC-001
manuscript, vector figures, frozen QPY circuits, calibration snapshot, and complete
192-circuit (393,216-shot) raw bitstring archive (`ibm_fez` Job `dbtl2vb2kf8s73c9a0d0`)
to Zenodo.

Reads `ZENODO_API_TOKEN` from `~/osiris-ide/.env` or environment without printing secrets.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Tuple

PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parent.parent
QEC_DIR = REPO_ROOT / "experiments" / "qec001"
RECEIPT_OUT = QEC_DIR / "zenodo_qec001_publication_receipt.json"
API_BASE = "https://zenodo.org/api/deposit/depositions"


def _get_token() -> str:
    for key in ("ZENODO_API_TOKEN", "OSIRIS_ZENODO_TOKEN", "ZENODO_TOKEN"):
        val = os.environ.get(key, "").strip()
        if val:
            return val
    for p in (Path("/home/osiris/osiris-ide/.env"), Path.home() / ".env", REPO_ROOT / ".env"):
        if p.is_file():
            for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
                line = raw.strip()
                if line.startswith(("ZENODO_API_TOKEN=", "OSIRIS_ZENODO_TOKEN=", "ZENODO_TOKEN=")):
                    tok = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if tok:
                        return tok
    raise RuntimeError("Zenodo token not found in environment or ~/osiris-ide/.env")


def _call(
    method: str,
    url: str,
    token: str,
    data: Dict[str, Any] | None = None,
    raw: bytes | None = None,
    ctype: str | None = "application/json",
) -> Tuple[int, Dict[str, Any]]:
    body = raw if raw is not None else (json.dumps(data).encode("utf-8") if data is not None else None)
    headers = {"Authorization": f"Bearer {token}"}
    if ctype and body is not None:
        headers["Content-Type"] = ctype
    req = urllib.request.Request(url, method=method, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            txt = resp.read().decode("utf-8")
            return resp.status, (json.loads(txt) if txt else {})
    except urllib.error.HTTPError as exc:
        txt = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(txt)
        except Exception:
            return exc.code, {"raw": txt[:500]}


def build_metadata() -> Dict[str, Any]:
    return {
        "upload_type": "publication",
        "publication_type": "preprint",
        "publication_date": "2026-11-16",
        "title": (
            "Sub-Threshold Multi-Round Repetition-Code Logical Error Scaling and Real-Time "
            "Suppression of a Localized T2 Hotspot Across Code Distances d in {3, 5, 7} on a "
            "Heavy-Hex Superconducting Processor (OSIRIS-QEC-001)"
        ),
        "creators": [
            {
                "name": "Davis, Devin Phillip",
                "affiliation": "Agile Defense Systems LLC",
            }
        ],
        "description": (
            "<p>Complete standalone manuscript, vector figures, prospectively pre-registered specification, "
            "compiled Qiskit QPY binary (SHA-256 <code>357a29ad2746e2be94522702516dce3e04319894c1dc981596c9a3640041c056</code>), "
            "pre-submission calibration snapshot, Edmonds Blossom MWPM decoder, and verbatim 192-circuit raw "
            "bitstring archive (<code>physical_qec001_fez_raw_counts.json.gz</code>, SHA-256 "
            "<code>cdae697c364e2aad2085cc3585039d079baf4b3d1bb619dcb6178647b75646c1</code>) for benchmark "
            "<strong>OSIRIS-QEC-001</strong> executed on the 156-qubit IBM Heron r2 superconducting processor "
            "<code>ibm_fez</code> (IBM Runtime Job <code>dbtl2vb2kf8s73c9a0d0</code>, 192 dynamic mid-circuit "
            "<code>measure</code>+<code>reset</code> circuits &times; 2,048 shots = <strong>393,216 shots</strong>, "
            "<code>resilience_level=0</code>, zero postselection, pre-submission calibration decoder priors).</p>"
            "<ul>"
            "<li><strong>Monotonic Sub-Threshold Suppression Across All 8 <code>(basis, R)</code> Curves:</strong> "
            "Across concentric subchains <code>d &isin; {3, 5, 7}</code> (5, 9, and 13 transmons on "
            "<code>[111,110,109,108,107,106,105,117,125,124,123,122,121]</code>) and syndrome cycles "
            "<code>R &isin; {1, 3, 5, 7}</code> (16,384 pooled shots/cell), logical error per cycle satisfies "
            "<code>&epsilon;_L(3) &gt; &epsilon;_L(5) &gt; &epsilon;_L(7)</code> everywhere. At <code>R = 7</code>, "
            "<code>bit_flip_z</code> drops from <code>0.00731 &plusmn; 0.00026</code> (<code>d=3</code>) to "
            "<code>0.00211 &plusmn; 0.00025</code> (<code>d=5</code>) and <code>0.00195 &plusmn; 0.00018</code> "
            "(<code>d=7</code>; <code>&Lambda;_{3&rarr;7} = 3.75&times;</code>, <strong>QEC-H1 PASS</strong>).</li>"
            "<li><strong>Real-Time Error Correction of a Live <code>q107</code> <code>T2 = 18.17 &mu;s</code> Hotspot:</strong> "
            "Pre-submission calibration captured a localized coherence hotspot on data transmon <code>q107</code> "
            "inside the <code>d=3</code> core (<code>T2 = 18.17 &mu;s</code>, giving <code>p_Z = 8.38%</code> thermal "
            "phase-flip probability per <code>3.344 &mu;s</code> mid-circuit <code>measure</code>+<code>reset</code> "
            "window, vs <code>T2 &isin; [45.2, 176.8] &mu;s</code> on surrounding <code>d=5, 7</code> data qubits). "
            "Expanding <code>d = 3 &rarr; 5 &rarr; 7</code> actively corrected <code>q107</code> across all 7 cycles, "
            "suppressing <code>phase_flip_x</code> logical error per cycle from <code>0.02301 &plusmn; 0.00051</code> "
            "(<code>d=3</code>) to <code>0.00592 &plusmn; 0.00025</code> (<code>d=5</code>, <code>3.89&times;</code>) "
            "and <code>0.00265 &plusmn; 0.00020</code> (<code>d=7</code>, <code>&Lambda;_{5&rarr;7} = 2.23&times;</code>, "
            "<strong><code>&Lambda;_{3&rarr;7} = 8.69&times;</code></strong>; <strong>QEC-H2 PASS</strong>).</li>"
            "<li><strong>Space-Time Blossom MWPM Gain &amp; Un-Doctored <code>QEC-H3</code> Ceiling:</strong> At "
            "<code>R = 7</code>, space-time MWPM reduced logical errors over final-round majority vote by "
            "<strong>7.22&times;</strong> (<code>bit_flip_z d=7</code>: 221 vs 1,596 errors) and "
            "<strong>4.86&times;</strong> (<code>phase_flip_x d=7</code>: 299 vs 1,453 errors), while "
            "<code>phase_flip_x</code> <code>R = 7</code> detector defect density reached <code>0.1265&ndash;0.1284</code> "
            "vs the pre-registered <code>0.1200</code> second conjunct of <code>QEC-H3</code> (preserved verbatim).</li>"
            "<li><strong>Scope Note:</strong> 1D repetition codes protect one Pauli sector (<code>Z</code> or <code>X</code>) "
            "per circuit run within standard open-system quantum mechanics, rather than simultaneous 2D logical qubits.</li>"
            "</ul>"
        ),
        "access_right": "open",
        "license": "cc-by-4.0",
        "version": "1.0.0",
        "keywords": [
            "quantum error correction",
            "repetition code",
            "mid-circuit measurement",
            "minimum-weight perfect matching",
            "superconducting qubits",
            "IBM Heron r2",
            "pre-registered quantum experiment",
            "OSIRIS-QEC-001",
        ],
        "related_identifiers": [
            {
                "identifier": "10.5281/zenodo.24287505",
                "relation": "isSupplementedBy",
                "scheme": "doi",
            }
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-only", action="store_true")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if not args.draft_only and not args.publish:
        parser.error("Specify --draft-only or --publish")

    files = [
        PAPER_DIR / "qec001_syndrome_scaling_heron_r2.pdf",
        PAPER_DIR / "qec001_syndrome_scaling_heron_r2.tex",
        PAPER_DIR / "fig1_logical_error_scaling.pdf",
        PAPER_DIR / "fig2_mwpm_vs_majority_and_defects.pdf",
        PAPER_DIR / "fig3_fez_chain_q107_t2_hotspot.pdf",
        PAPER_DIR / "SHA256SUMS.txt",
        QEC_DIR / "PRE_REGISTRATION.md",
        QEC_DIR / "RESULT.md",
        QEC_DIR / "qec001_repetition_scaling.py",
        QEC_DIR / "stage_and_harvest_qec001.py",
        QEC_DIR / "qec001_fez_staged_circuits.qpy",
        QEC_DIR / "qec001_fez_staging_manifest.json",
        QEC_DIR / "qec001_fez_calibration_snapshot.json",
        QEC_DIR / "qec001_fez_submitted_job.json",
        QEC_DIR / "physical_qec001_fez_results.json",
        QEC_DIR / "physical_qec001_fez_raw_counts.json.gz",
        REPO_ROOT / "tests" / "test_qec001_hardware_artifact.py",
    ]
    for f in files:
        if not f.is_file():
            raise FileNotFoundError(f"Missing required artifact file: {f}")

    token = _get_token()
    st, dep = _call("POST", API_BASE, token, data={})
    if st not in (200, 201):
        raise RuntimeError(f"Failed to create Zenodo deposition (HTTP {st}): {dep}")

    did = dep["id"]
    bucket = dep["links"]["bucket"]

    st, meta_res = _call("PUT", f"{API_BASE}/{did}", token, data={"metadata": build_metadata()})
    if st not in (200, 201):
        _call("DELETE", f"{API_BASE}/{did}", token, ctype=None)
        raise RuntimeError(f"Failed to set Zenodo metadata (HTTP {st}): {meta_res}")

    uploaded = []
    for f in files:
        raw = f.read_bytes()
        st, up_res = _call("PUT", f"{bucket}/{f.name}", token, raw=raw, ctype="application/octet-stream")
        if st not in (200, 201):
            raise RuntimeError(f"Upload failed for {f.name} (HTTP {st}): {up_res}")
        uploaded.append({"name": f.name, "bytes": len(raw), "checksum": up_res.get("checksum")})

    if args.draft_only:
        prereserve = (meta_res.get("metadata") or {}).get("prereserve_doi") or {}
        print(
            json.dumps(
                {
                    "state": "draft",
                    "deposition_id": did,
                    "prereserved_doi": prereserve.get("doi"),
                    "draft_url": f"https://zenodo.org/deposit/{did}",
                    "uploaded_files_count": len(uploaded),
                },
                indent=2,
            )
        )
        return 0

    st, pub = _call("POST", f"{API_BASE}/{did}/actions/publish", token, ctype=None)
    if st not in (200, 201, 202):
        raise RuntimeError(f"Zenodo publish failed (HTTP {st}): {pub}")

    receipt = {
        "benchmark_id": "OSIRIS-QEC-001",
        "job_id": "dbtl2vb2kf8s73c9a0d0",
        "record_id": pub.get("id"),
        "doi": pub.get("doi"),
        "conceptdoi": pub.get("conceptdoi"),
        "doi_url": pub.get("doi_url"),
        "record_html": (pub.get("links") or {}).get("record_html", f"https://zenodo.org/records/{pub.get('id')}"),
        "state": pub.get("state"),
        "submitted": pub.get("submitted"),
        "created": pub.get("created"),
        "uploaded_files": uploaded,
    }
    RECEIPT_OUT.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
