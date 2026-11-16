#!/usr/bin/env python3
"""
publish_hp004_to_zenodo.py — Uploads and publishes the self-contained OSIRIS-HP-004
manuscript, vector figures, frozen QPY circuits, and complete 286-circuit raw bitstring
archive (`ibm_marrakesh` Job `dbti1fbsbsbs73cb2vu0`) to Zenodo.

Reads `ZENODO_API_TOKEN` from `~/osiris-ide/.env` or environment without printing secrets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Tuple

PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parent.parent
HP004_DIR = REPO_ROOT / "experiments" / "hp004"
RECEIPT_OUT = HP004_DIR / "zenodo_hp004_publication_receipt.json"
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
    ctype: str = "application/json",
) -> Tuple[int, Dict[str, Any]]:
    body = raw if raw is not None else (json.dumps(data).encode("utf-8") if data is not None else None)
    headers = {"Authorization": f"Bearer {token}"}
    if ctype:
        headers["Content-Type"] = ctype
    req = urllib.request.Request(url, method=method, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
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
            "Unconditional Certification of Multi-Qubit Hayden-Preskill Channel Capacity "
            "Beyond One Ebit via Layout-Locked Nine-Basis Stabilizer Tomography on a "
            "Heavy-Hex Superconducting Processor (OSIRIS-HP-004)"
        ),
        "creators": [
            {
                "name": "Davis, Devin Phillip",
                "affiliation": "Agile Defense Systems LLC",
            }
        ],
        "description": (
            "<p>Complete manuscript, vector figures, prospectively pre-registered specification, "
            "compiled Qiskit QPY binary (SHA-256 <code>c3daaa48d89a74cec39932c310365b3eb23660ac4922219586543f8fe473d02c</code>), "
            "calibration snapshot, and full 286-circuit raw 4-bit bitstring dataset for benchmark "
            "<strong>OSIRIS-HP-004</strong> executed on the 156-qubit IBM Heron r2 superconducting processor "
            "<code>ibm_marrakesh</code> (IBM Runtime Job <code>dbti1fbsbsbs73cb2vu0</code>, 286 circuits &times; 1,024 shots = "
            "292,864 shots, <code>resilience_level=0</code>, zero postselection).</p>"
            "<p><strong>Key Findings:</strong></p>"
            "<ul>"
            "<li><strong>Exact 1-Ebit Certification:</strong> Measuring all 9 two-pair Pauli settings "
            "<code>{X,Y,Z}_1 &times; {X,Y,Z}_2</code> (all 16 four-qubit Bell stabilizers) on the pinned 8-transmon "
            "heavy-hex tree <code>[34, 52, 32, 33, 39, 53, 54, 55]</code> with <code>routing_method='none'</code>, "
            "raw unmitigated restricted two-photon recovery (<code>Arm B</code>, 32 CZ gates; zero physical gates on "
            "horizon remnants <code>q32, q55</code> or references <code>q34, q52</code> during decoding) achieves "
            "<code>F_joint = 0.6334 &plusmn; 0.0026</code> (95% bootstrap CI <code>[0.6284, 0.6385]</code>), exceeding "
            "the exact <code>0.5000</code> one-ebit Schmidt-rank-2 ceiling by <strong>+50.64&sigma;</strong> "
            "(<code>0.6921</code> under secondary same-job 16-state readout calibration).</li>"
            "<li><strong>Empirical Necessity of Nine-Basis Tomography:</strong> On the identical raw bitstrings, "
            "two-qubit marginals yield <code>F_mean = 0.7438 &lt; 0.7500</code>, <code>F1*F2 = 0.5528 &lt; 0.5625</code>, "
            "and union bound <code>F1+F2-1 = 0.4875 &lt; 0.5000</code>, demonstrating that 3-basis marginal protocols "
            "cannot certify &gt;1 ebit even when true joint overlap is <code>0.6334</code>.</li>"
            "<li><strong>Dyadic Capacity Ladder (5/5 Pre-Registered Hypotheses Passed):</strong> Barriered unitary inverse "
            "<code>Arm A = 0.6591 &plusmn; 0.0026</code>; unentangled 0-ebit classical correlation "
            "<code>Arm B-Ctrl = 0.1799 &plusmn; 0.0027</code>; single-photon sub-capacity collapse "
            "<code>Arm B-Def = 0.1086 &plusmn; 0.0035</code>; mismatched scrambler specificity "
            "<code>Arm B-Held = 0.0634 &plusmn; 0.0027</code> (&approx; 1/16).</li>"
            "</ul>"
            "<p><strong>Scope Note:</strong> The scrambler and restricted decoders are exact 4-qubit Clifford unitaries "
            "(classically simulable via Gottesman-Knill); this dataset certifies physical multi-qubit quantum channel "
            "capacity (&gt;1 ebit) and subsystem decoupling on heavy-hex hardware rather than super-polynomial quantum "
            "computational advantage.</p>"
        ),
        "access_right": "open",
        "license": "cc-by-4.0",
        "version": "1.0.0",
        "keywords": [
            "Hayden-Preskill protocol",
            "quantum channel capacity",
            "stabilizer tomography",
            "superconducting qubits",
            "IBM Heron r2",
            "pre-registered quantum experiment",
            "OSIRIS-HP-004",
        ],
        "related_identifiers": [
            {
                "identifier": "10.5281/zenodo.24031814",
                "relation": "continues",
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
        PAPER_DIR / "hp004_joint_stabilizer_heron_r2.pdf",
        PAPER_DIR / "hp004_joint_stabilizer_heron_r2.tex",
        PAPER_DIR / "fig1_dyadic_capacity_ladder.pdf",
        PAPER_DIR / "fig2_block_stability_and_marginals.pdf",
        PAPER_DIR / "fig3_marrakesh_tree_light_cone.pdf",
        PAPER_DIR / "SHA256SUMS.txt",
        HP004_DIR / "PRE_REGISTRATION.md",
        HP004_DIR / "RESULT.md",
        HP004_DIR / "physical_hp004_results.json",
        HP004_DIR / "hp004_staged_circuits.qpy",
        HP004_DIR / "hp004_staging_manifest.json",
        HP004_DIR / "hp004_calibration_snapshot.json",
        HP004_DIR / "hp004_submitted_job.json",
        HP004_DIR / "hp004_joint_stabilizer.py",
        HP004_DIR / "execute_and_harvest_hp004.py",
        REPO_ROOT / "tests" / "test_hp004_hardware_artifact.py",
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
        _call("DELETE", f"{API_BASE}/{did}", token)
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
        out = {
            "state": "draft",
            "deposition_id": did,
            "prereserved_doi": prereserve.get("doi"),
            "draft_url": f"https://zenodo.org/deposit/{did}",
            "uploaded_files": uploaded,
        }
        print(json.dumps(out, indent=2))
        return 0

    st, pub = _call("POST", f"{API_BASE}/{did}/actions/publish", token)
    if st not in (200, 201, 202):
        raise RuntimeError(f"Zenodo publish failed (HTTP {st}): {pub}")

    receipt = {
        "benchmark_id": "OSIRIS-HP-004",
        "record_id": pub.get("id"),
        "doi": pub.get("doi"),
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
