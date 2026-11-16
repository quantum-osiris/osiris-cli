#!/usr/bin/env python3
"""
publish_hp004_to_zenodo.py — Uploads and publishes Version 2.0.0 of the dual-processor
OSIRIS-HP-004 / OSIRIS-HP-004B manuscript, vector figures, frozen QPY circuits, and
complete 572-circuit (585,728-shot) raw bitstring archive (`ibm_marrakesh` Job
`dbti1fbsbsbs73cb2vu0` + `ibm_fez` Job `dbtj61bnlcms73ajc380`) to Zenodo.

Reads `ZENODO_API_TOKEN` from `~/osiris-ide/.env` or environment without printing secrets.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parent.parent
HP004_DIR = REPO_ROOT / "experiments" / "hp004"
FEZ_DIR = REPO_ROOT / "experiments" / "hp004b_fez"
RECEIPT_V1 = HP004_DIR / "zenodo_hp004_publication_receipt.json"
RECEIPT_V2 = FEZ_DIR / "zenodo_hp004_v2_publication_receipt.json"
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


def build_metadata_v2() -> Dict[str, Any]:
    return {
        "upload_type": "publication",
        "publication_type": "preprint",
        "publication_date": "2026-11-16",
        "title": (
            "Dual-Processor Certification of Schmidt Number >= 3 and >1-Ebit Entanglement "
            "of Formation in Hayden-Preskill Recovery via Layout-Locked Nine-Basis "
            "Stabilizer Tomography on Heavy-Hex Superconducting Processors (OSIRIS-HP-004 & HP-004B, v2.0)"
        ),
        "creators": [
            {
                "name": "Davis, Devin Phillip",
                "affiliation": "Agile Defense Systems LLC",
            }
        ],
        "description": (
            "<p><strong>Version 2.0.0 Update &amp; Mathematical Clarification:</strong> "
            "This revision adds the completed cross-processor physical hardware replication "
            "<strong>OSIRIS-HP-004B</strong> on <code>ibm_fez</code> (IBM Runtime Job <code>dbtj61bnlcms73ajc380</code>, "
            "pre-registered QPY SHA-256 <code>6d7e5a6d17a002b5551eba35fb99b29661044477a9f6b91fb9dc51376a3025d9</code>, "
            "286 circuits &times; 1,024 shots = 292,864 shots) alongside <strong>OSIRIS-HP-004</strong> on "
            "<code>ibm_marrakesh</code> (Job <code>dbti1fbsbsbs73cb2vu0</code>, 292,864 shots), totaling "
            "<strong>585,728 raw unmitigated shots</strong> across two independent 156-qubit IBM Heron r2 cryostats. "
            "Version 2.0 also explicitly separates the <strong>Terhal&ndash;Horodecki (2000, quant-ph/9911117) "
            "Schmidt-number &ge; 3 witness</strong> (<code>F_joint &gt; 0.5000</code>) from the "
            "<strong>Vollbrecht&ndash;Werner (2001, quant-ph/0010095) Entanglement of Formation &gt; 1 ebit "
            "convex-roof threshold</strong> (<code>F_joint &gt; F* = 0.683910</code> for <code>d = 4</code>), correcting "
            "v1.0.0's loose conflation of <code>F_joint &gt; 0.5000</code> with <code>&gt; 1 ebit</code>.</p>"
            "<ul>"
            "<li><strong><code>ibm_marrakesh</code> (Job <code>dbti1fbsbsbs73cb2vu0</code>, Tree <code>[34,52,32,33,39,53,54,55]</code>):</strong> "
            "Raw unmitigated <code>Arm B</code> (32 CZs; zero physical gates on horizon remnants <code>q32, q55</code> or "
            "references <code>q34, q52</code> during decoding) achieves <code>F_joint = 0.6334 &plusmn; 0.0026</code> "
            "(6-block 95% Student-<em>t</em> CI <code>[0.6267, 0.6402]</code>), certifying <strong>Schmidt number "
            "SN(&rho;) &ge; 3 at +50.64&sigma;</strong> and <code>E_F(&rho;) &ge; 0.8451 &plusmn; 0.0080 ebits</code> "
            "(while 3-basis marginals yield <code>F1 + F2 - 1 = 0.4875 &lt; 0.5000</code>).</li>"
            "<li><strong><code>ibm_fez</code> Cross-Processor Replication (Job <code>dbtj61bnlcms73ajc380</code>, Tree <code>[2,24,4,3,16,23,22,21]</code>):</strong> "
            "Raw unmitigated <code>Arm B</code> (32 CZs; zero gates on remnants <code>q4, q21</code> or references "
            "<code>q2, q24</code> during decoding) achieves <strong><code>F_joint = 0.7307 &plusmn; 0.0027</code></strong> "
            "(6-block 95% Student-<em>t</em> CI <code>[0.7237, 0.7377]</code>; <code>0.7611</code> under secondary SPAM "
            "inversion), simultaneously certifying <strong>SN(&rho;) &ge; 3 at +84.45&sigma;</strong> AND "
            "<strong>Entanglement of Formation <code>E_F(&rho;) &ge; 1.1465 &plusmn; 0.0086 ebits &gt; 1.0000 ebit</code> "
            "at +17.13&sigma;</strong> (5&sigma; lower bound <code>E_F &ge; 1.1035 ebits</code>). Negative controls "
            "(<code>0.2053, 0.1140, 0.0563</code>) match pre-registered Kraus predictions (<code>0.2171, 0.1163, 0.0626</code>) "
            "within 0.2%&ndash;1.2%.</li>"
            "<li><strong>Scope Boundary:</strong> All scramblers and decoders are exact 4-qubit Clifford circuits "
            "(Gottesman&ndash;Knill simulable) testing standard open-system quantum channel decoupling and entanglement "
            "dimensionality on heavy-hex hardware, making zero claims of super-polynomial computational advantage or "
            "departures from standard quantum mechanics.</li>"
            "</ul>"
        ),
        "access_right": "open",
        "license": "cc-by-4.0",
        "version": "2.0.0",
        "keywords": [
            "Hayden-Preskill protocol",
            "Schmidt number witness",
            "entanglement of formation",
            "stabilizer tomography",
            "superconducting qubits",
            "IBM Heron r2",
            "cross-processor replication",
            "OSIRIS-HP-004",
            "OSIRIS-HP-004B",
        ],
        "related_identifiers": [
            {
                "identifier": "10.5281/zenodo.24031814",
                "relation": "continues",
                "scheme": "doi",
            }
        ],
    }


def get_bundle_files() -> List[Tuple[Path, str]]:
    return [
        (PAPER_DIR / "hp004_joint_stabilizer_heron_r2.pdf", "hp004_joint_stabilizer_heron_r2.pdf"),
        (PAPER_DIR / "hp004_joint_stabilizer_heron_r2.tex", "hp004_joint_stabilizer_heron_r2.tex"),
        (PAPER_DIR / "fig1_dyadic_capacity_ladder.pdf", "fig1_dyadic_capacity_ladder.pdf"),
        (PAPER_DIR / "fig2_block_stability_and_marginals.pdf", "fig2_block_stability_and_marginals.pdf"),
        (PAPER_DIR / "fig3_marrakesh_tree_light_cone.pdf", "fig3_marrakesh_tree_light_cone.pdf"),
        (PAPER_DIR / "fig4_fez_tree_light_cone.pdf", "fig4_fez_tree_light_cone.pdf"),
        (PAPER_DIR / "SHA256SUMS.txt", "SHA256SUMS.txt"),
        (HP004_DIR / "PRE_REGISTRATION.md", "hp004_marrakesh_PRE_REGISTRATION.md"),
        (HP004_DIR / "RESULT.md", "hp004_marrakesh_RESULT.md"),
        (HP004_DIR / "hp004_joint_stabilizer.py", "hp004_joint_stabilizer.py"),
        (HP004_DIR / "execute_and_harvest_hp004.py", "execute_and_harvest_hp004.py"),
        (HP004_DIR / "hp004_staged_circuits.qpy", "hp004_staged_circuits.qpy"),
        (HP004_DIR / "hp004_staging_manifest.json", "hp004_staging_manifest.json"),
        (HP004_DIR / "hp004_calibration_snapshot.json", "hp004_calibration_snapshot.json"),
        (HP004_DIR / "hp004_submitted_job.json", "hp004_submitted_job.json"),
        (HP004_DIR / "physical_hp004_results.json", "physical_hp004_results.json"),
        (FEZ_DIR / "PRE_REGISTRATION.md", "hp004b_fez_PRE_REGISTRATION.md"),
        (FEZ_DIR / "RESULT.md", "hp004b_fez_RESULT.md"),
        (FEZ_DIR / "stage_and_harvest_hp004b_fez.py", "stage_and_harvest_hp004b_fez.py"),
        (FEZ_DIR / "hp004b_fez_staged_circuits.qpy", "hp004b_fez_staged_circuits.qpy"),
        (FEZ_DIR / "hp004b_fez_staging_manifest.json", "hp004b_fez_staging_manifest.json"),
        (FEZ_DIR / "hp004b_fez_calibration_snapshot.json", "hp004b_fez_calibration_snapshot.json"),
        (FEZ_DIR / "hp004b_fez_submitted_job.json", "hp004b_fez_submitted_job.json"),
        (FEZ_DIR / "physical_hp004b_fez_results.json", "physical_hp004b_fez_results.json"),
        (REPO_ROOT / "tests" / "test_hp004_hardware_artifact.py", "test_hp004_hardware_artifact.py"),
        (REPO_ROOT / "tests" / "test_hp004b_fez_hardware_artifact.py", "test_hp004b_fez_hardware_artifact.py"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-version-of", type=int, default=24285577)
    parser.add_argument("--draft-only", action="store_true")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if not args.draft_only and not args.publish:
        parser.error("Specify --draft-only or --publish")

    files = get_bundle_files()
    for src, _ in files:
        if not src.is_file():
            raise FileNotFoundError(f"Missing required artifact file: {src}")

    token = _get_token()

    # Create a new version draft from the existing published record
    st, nv_res = _call("POST", f"{API_BASE}/{args.new_version_of}/actions/newversion", token, ctype=None)
    if st not in (200, 201):
        raise RuntimeError(f"Failed to create new version of {args.new_version_of} (HTTP {st}): {nv_res}")

    latest_draft_url = (nv_res.get("links") or {}).get("latest_draft")
    if not latest_draft_url:
        raise RuntimeError(f"Missing latest_draft link in newversion response: {nv_res}")

    st, draft = _call("GET", latest_draft_url, token, ctype=None)
    if st != 200:
        raise RuntimeError(f"Failed to fetch latest_draft (HTTP {st}): {draft}")

    did = draft["id"]
    bucket = draft["links"]["bucket"]

    # Remove inherited v1 files so the v2 bundle has clean, explicit filenames matching SHA256SUMS.txt
    for old_f in draft.get("files") or []:
        fid = old_f.get("id")
        if fid:
            del_st, del_res = _call("DELETE", f"{API_BASE}/{did}/files/{fid}", token, ctype=None)
            if del_st not in (200, 204):
                raise RuntimeError(f"Failed deleting inherited file {old_f.get('filename')} (HTTP {del_st}): {del_res}")

    uploaded = []
    for src, arc_name in files:
        raw = src.read_bytes()
        st, up_res = _call("PUT", f"{bucket}/{arc_name}", token, raw=raw, ctype="application/octet-stream")
        if st not in (200, 201):
            raise RuntimeError(f"Upload failed for {arc_name} (HTTP {st}): {up_res}")
        uploaded.append({"name": arc_name, "bytes": len(raw), "checksum": up_res.get("checksum")})

    st, meta_res = _call("PUT", f"{API_BASE}/{did}", token, data={"metadata": build_metadata_v2()})
    if st not in (200, 201):
        raise RuntimeError(f"Failed to update v2 metadata (HTTP {st}): {meta_res}")

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
        raise RuntimeError(f"Zenodo v2 publish failed (HTTP {st}): {pub}")

    receipt = {
        "benchmark_ids": ["OSIRIS-HP-004", "OSIRIS-HP-004B"],
        "supersedes_record_id": args.new_version_of,
        "version": "2.0.0",
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
    RECEIPT_V2.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
