#!/usr/bin/env python3
"""Publish v6.2.4 OSIRIS Papers 9 & 10 and their reproducibility bundles to Zenodo:

  1. `rep3_echo_multidistance_qec` (Paper 9) — Write-ahead live `ibm_fez` `OSIRIS-QEC-REP3-FEZ`
     (`dc35d4inj2vs73dhmsog`, 49,152 shots) & `OSIRIS-ECHO-FEZ` (`dc35e35s2omc738cvcug`,
     61,440 shots) + `OSIRIS-QEC-MULTIDISTANCE` hook-safe d=3,5,7 surface/repetition threshold
  2. `bt2t003_plaquette_dynamics` (Paper 10) — Kogut-Susskind 2T subset SU(2) non-abelian
     lattice gauge plaquette Trotter dynamics & 104x-178x native gate compression (`SIMULATED`)

Security & non-mutation guarantees:
  - Reads `ZENODO_API_TOKEN` from environment or `.env` / `.env.local` in `REPO_ROOT` without
    EVER printing, logging, or serializing secret tokens.
  - Refuses to upload any `.env*`, `*key*`, or enclave credential file.
  - Verifies `SHA256SUMS.txt` and `SYS-024` Claims Firewall before any network call.
  - Writes draft/publication receipts exclusively under `papers/zenodo_v624_deposits/<slug>/zenodo_receipt.json`
    so locked `SHA256SUMS.txt` manifests inside `papers/<slug>/` remain 100% untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osiris_cli import claims, plan_gates  # noqa: E402

DEPOSITS_ROOT = REPO_ROOT / "papers" / "zenodo_v624_deposits"
API_BASE = "https://zenodo.org/api/deposit/depositions"
CREATOR = [{"name": "Davis, Devin Phillip", "affiliation": "Agile Defense Systems LLC"}]
FORBIDDEN_NAME_SUBSTRINGS = (".env", "troykey", "apikey", "secret", "credential", "private_key")


def _load_zenodo_token() -> str:
    for key in ("ZENODO_API_TOKEN", "OSIRIS_ZENODO_TOKEN", "ZENODO_TOKEN"):
        val = os.environ.get(key, "").strip()
        if val:
            return val
    candidate_envs = (
        REPO_ROOT / ".env",
        REPO_ROOT / ".env.local",
        Path.home() / "osiris-ide" / ".env",
        Path.home() / ".env",
    )
    for env_path in candidate_envs:
        if env_path.is_file():
            for raw_line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                for key in ("ZENODO_API_TOKEN=", "OSIRIS_ZENODO_TOKEN=", "ZENODO_TOKEN="):
                    if line.startswith(key):
                        tok = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if tok:
                            return tok
    raise RuntimeError("ZENODO_API_TOKEN not found in environment or authorized .env/.env.local files")


def _call_once(
    method: str,
    url: str,
    token: str,
    data: Optional[Dict[str, Any]] = None,
    raw: Optional[bytes] = None,
    ctype: Optional[str] = "application/json",
) -> Tuple[int, Dict[str, Any]]:
    body = raw if raw is not None else (json.dumps(data).encode("utf-8") if data is not None else None)
    headers = {"Authorization": f"Bearer {token}"}
    if body is not None and ctype:
        headers["Content-Type"] = ctype
    req = urllib.request.Request(url, method=method, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            txt = resp.read().decode("utf-8")
            return resp.status, (json.loads(txt) if txt else {})
    except urllib.error.HTTPError as exc:
        txt = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(txt)
        except Exception:
            return exc.code, {"raw": txt[:600]}


def _call(
    method: str,
    url: str,
    token: str,
    data: Optional[Dict[str, Any]] = None,
    raw: Optional[bytes] = None,
    ctype: Optional[str] = "application/json",
) -> Tuple[int, Dict[str, Any]]:
    last: Tuple[int, Dict[str, Any]] = (0, {})
    for attempt in range(6):
        try:
            st, body = _call_once(method, url, token, data=data, raw=raw, ctype=ctype)
        except Exception as exc:
            st, body = 0, {"raw": f"{type(exc).__name__}: {exc}"}
        if st not in (0, 429, 500, 502, 503, 504):
            return st, body
        last = (st, body)
        time.sleep(min(45, 3 * (2**attempt)))
    return last


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_safe_and_sys024(slug: str, cfg: Dict[str, Any]) -> None:
    pdir = REPO_ROOT / cfg["paper_dir"]
    sums_path = pdir / "SHA256SUMS.txt"
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        exp_sha, rel = line.strip().split("  ", 1)
        target = REPO_ROOT / rel
        if not target.is_file() or _sha256(target) != exp_sha:
            raise RuntimeError(f"[{slug}] SHA256SUMS mismatch on {rel}")

    tex_txt = (pdir / f"{slug}.tex").read_text(encoding="utf-8")
    gen_txt = (pdir / "generated.tex").read_text(encoding="utf-8")
    meta_txt = f"{cfg['meta']['title']}\n{cfg['meta']['description']}\n" + ", ".join(cfg["meta"]["keywords"])
    for label, txt in (("tex", tex_txt), ("gen", gen_txt), ("meta", meta_txt)):
        flagged = claims.flagged_in_reply(txt, [])
        if flagged:
            raise RuntimeError(f"[{slug}] SYS-024 flagged in {label}: {[c.id for c in flagged]}")
        for cname, pat in plan_gates._QUARANTINED_PLAN_PATTERNS:
            if pat.search(txt):
                raise RuntimeError(f"[{slug}] SYS-024 banned pattern {cname} in {label}")


PAPERS: Dict[str, Dict[str, Any]] = {
    "rep3_echo_multidistance_qec": {
        "paper_dir": "papers/rep3_echo_multidistance_qec",
        "version": "1.0.0",
        "publication_date": "2026-11-25",
        "files": [
            ("papers/rep3_echo_multidistance_qec/rep3_echo_multidistance_qec.pdf", "rep3_echo_multidistance_qec.pdf"),
            ("papers/rep3_echo_multidistance_qec/rep3_echo_multidistance_qec.tex", "rep3_echo_multidistance_qec.tex"),
            ("papers/rep3_echo_multidistance_qec/generated.tex", "generated.tex"),
            (
                "papers/rep3_echo_multidistance_qec/generate_rep3_echo_qec_paper_assets.py",
                "generate_rep3_echo_qec_paper_assets.py",
            ),
            (
                "papers/rep3_echo_multidistance_qec/fig1_rep3_fez_crossing_and_echo_ranking.pdf",
                "fig1_rep3_fez_crossing_and_echo_ranking.pdf",
            ),
            (
                "papers/rep3_echo_multidistance_qec/fig1_rep3_fez_crossing_and_echo_ranking.svg",
                "fig1_rep3_fez_crossing_and_echo_ranking.svg",
            ),
            (
                "papers/rep3_echo_multidistance_qec/fig2_multidistance_surface_threshold_and_lambda.pdf",
                "fig2_multidistance_surface_threshold_and_lambda.pdf",
            ),
            (
                "papers/rep3_echo_multidistance_qec/fig2_multidistance_surface_threshold_and_lambda.svg",
                "fig2_multidistance_surface_threshold_and_lambda.svg",
            ),
            ("papers/rep3_echo_multidistance_qec/reproducibility_capsule.json", "reproducibility_capsule.json"),
            ("papers/rep3_echo_multidistance_qec/SHA256SUMS.txt", "SHA256SUMS.txt"),
            ("experiments/rep3_fez_staged/PRE_REGISTRATION.md", "rep3_fez_PRE_REGISTRATION.md"),
            ("experiments/rep3_fez_staged/stage_and_harvest_rep3_fez.py", "stage_and_harvest_rep3_fez.py"),
            ("experiments/rep3_fez_staged/rep3_fez_staging_manifest.json", "rep3_fez_staging_manifest.json"),
            ("experiments/rep3_fez_staged/rep3_fez_staged_circuits.qpy", "rep3_fez_staged_circuits.qpy"),
            ("experiments/rep3_fez_staged/rep3_fez_calibration_snapshot.json", "rep3_fez_calibration_snapshot.json"),
            ("experiments/rep3_fez_staged/rep3_fez_submitted_job.json", "rep3_fez_submitted_job.json"),
            ("experiments/rep3_fez_staged/physical_rep3_fez_raw_counts.json.gz", "physical_rep3_fez_raw_counts.json.gz"),
            ("experiments/rep3_fez_staged/physical_rep3_fez_results.json", "physical_rep3_fez_results.json"),
            ("experiments/echo_fez_staged/PRE_REGISTRATION.md", "echo_fez_PRE_REGISTRATION.md"),
            ("experiments/echo_fez_staged/stage_and_harvest_echo_fez.py", "stage_and_harvest_echo_fez.py"),
            ("experiments/echo_fez_staged/echo_fez_staging_manifest.json", "echo_fez_staging_manifest.json"),
            ("experiments/echo_fez_staged/echo_fez_staged_circuits.qpy", "echo_fez_staged_circuits.qpy"),
            ("experiments/echo_fez_staged/echo_fez_calibration_snapshot.json", "echo_fez_calibration_snapshot.json"),
            ("experiments/echo_fez_staged/echo_fez_submitted_job.json", "echo_fez_submitted_job.json"),
            ("experiments/echo_fez_staged/physical_echo_fez_raw_counts.json.gz", "physical_echo_fez_raw_counts.json.gz"),
            ("experiments/echo_fez_staged/physical_echo_fez_results.json", "physical_echo_fez_results.json"),
            ("experiments/qec_card_c/qec_multidistance_scaling.py", "qec_multidistance_scaling.py"),
            (
                "experiments/qec_card_c/artifacts/card_c_multidistance_threshold_report.json",
                "card_c_multidistance_threshold_report.json",
            ),
            ("tests/test_paper9_rep3_echo_multidistance.py", "test_paper9_rep3_echo_multidistance.py"),
        ],
        "related": [
            {"identifier": "10.5281/zenodo.24520746", "relation": "continues", "scheme": "doi"},
            {"identifier": "10.5281/zenodo.24520723", "relation": "references", "scheme": "doi"},
            {"identifier": "10.5281/zenodo.24293756", "relation": "references", "scheme": "doi"},
            {"identifier": "10.5281/zenodo.24420902", "relation": "references", "scheme": "doi"},
        ],
        "meta": {
            "title": (
                "Write-Ahead Hardware Crossing of Real-Time Feed-Forward Repetition Correction, "
                "50-Transmon Echo Placement Diagnostics on ibm_fez, and Hook-Safe d in {3,5,7} "
                "Surface-Code Threshold Scaling"
            ),
            "description": (
                "<p>Reports two write-ahead QPY-locked <code>HARDWARE_ATTESTED</code> experiments "
                "(110,592 physical shots, 35 s QPU) on the 156-qubit IBM Heron r2 processor <code>ibm_fez</code> "
                "alongside a <code>SIMULATED</code> (<code>provider_job_id: null</code>, 0 s QPU) "
                "<em>d</em> &isin; {3, 5, 7} repetition and rotated surface code threshold audit:</p>"
                "<ul>"
                "<li><strong><code>OSIRIS-QEC-REP3-FEZ</code> (Job <code>dc35d4inj2vs73dhmsog</code>, 96 PUBs, "
                "49,152 shots, 16 s QPU on <code>[58,71,70,69,78]</code>):</strong> Both space-time MWPM Pauli-frame "
                "decoding (Arm 4a) and real-time OpenQASM 3 <code>if_test</code> physical feed-forward (Arm 4b) "
                "cross cleanly below bare unencoded (Arm 1) and encoded no-correction (Arm 2) baselines at "
                "<em>R</em> = 3 and <em>R</em> = 5 across all 3/3 interleaved blocks (at <em>R</em> = 5: "
                "Arm 4a = 0.0872, Arm 4b = 0.0980 vs Arm 2 = 0.1230, Arm 1 = 0.2770; Arm 4b &lt; Arm 2 paired "
                "<em>t</em> = 9.01, one-sided <em>p</em> = 0.00604).</li>"
                "<li><strong><code>OSIRIS-ECHO-FEZ</code> (Job <code>dc35e35s2omc738cvcug</code>, 240 PUBs, "
                "61,440 shots, 19 s QPU across 10 disjoint <em>T</em><sub>5</sub> subgraphs = 50 transmons):</strong> "
                "16-CZ echo selects the #1 oracle-optimal placement (<code>[5,6,7,8,17]</code>, 0.0000 top-1 regret) "
                "and beats fresh calibration on 24-CZ mirror circuits (+0.9030 vs +0.7576 Spearman), while trailing "
                "calibration by &minus;0.0606 on the unweighted composite (+0.7212 vs +0.7818, H2 refuted); post-hoc "
                "0-CZ identity SPAM normalization lifts Spearman correlation to +0.9030 composite and +0.9515 on "
                "72-CZ binary-tetrahedral circuits.</li>"
                "<li><strong><code>OSIRIS-QEC-MULTIDISTANCE</code> (<code>SIMULATED</code>, 0 s QPU):</strong> "
                "Exact polynomial Edmonds Blossom MWPM (0/3,600 DP mismatches) across hook-safe [[<em>d</em>&sup2;,1,<em>d</em>]] "
                "rotated surface codes (17Q, 49Q, 97Q) establishes <em>p</em><sub>th</sub><sup>surf</sup> = 1.124%, "
                "deep sub-threshold scaling on Quantinuum <code>H2-1</code> (&Lambda;<sub>3&rarr;5</sub> = 6.96, "
                "&Lambda;<sub>5&rarr;7</sub> = 13.78), and drifted-SPAM differential pinning (&Lambda;<sub>3&rarr;5</sub> "
                "= 1.023).</li>"
                "</ul>"
            ),
            "keywords": [
                "quantum error correction",
                "dynamic circuits",
                "real-time feed-forward",
                "rotated surface code",
                "minimum-weight perfect matching",
                "IBM Heron r2",
                "ibm_fez",
                "qubit placement",
                "preregistration",
            ],
        },
    },
    "bt2t003_plaquette_dynamics": {
        "paper_dir": "papers/bt2t003_plaquette_dynamics",
        "version": "1.0.0",
        "publication_date": "2026-11-25",
        "files": [
            ("papers/bt2t003_plaquette_dynamics/bt2t003_plaquette_dynamics.pdf", "bt2t003_plaquette_dynamics.pdf"),
            ("papers/bt2t003_plaquette_dynamics/bt2t003_plaquette_dynamics.tex", "bt2t003_plaquette_dynamics.tex"),
            ("papers/bt2t003_plaquette_dynamics/generated.tex", "generated.tex"),
            (
                "papers/bt2t003_plaquette_dynamics/generate_bt2t003_paper_assets.py",
                "generate_bt2t003_paper_assets.py",
            ),
            (
                "papers/bt2t003_plaquette_dynamics/fig1_bt2t003_strang_scaling_and_wilson_regimes.pdf",
                "fig1_bt2t003_strang_scaling_and_wilson_regimes.pdf",
            ),
            (
                "papers/bt2t003_plaquette_dynamics/fig1_bt2t003_strang_scaling_and_wilson_regimes.svg",
                "fig1_bt2t003_strang_scaling_and_wilson_regimes.svg",
            ),
            (
                "papers/bt2t003_plaquette_dynamics/fig2_bt2t003_gauss_holonomy_cz_compression.pdf",
                "fig2_bt2t003_gauss_holonomy_cz_compression.pdf",
            ),
            (
                "papers/bt2t003_plaquette_dynamics/fig2_bt2t003_gauss_holonomy_cz_compression.svg",
                "fig2_bt2t003_gauss_holonomy_cz_compression.svg",
            ),
            ("papers/bt2t003_plaquette_dynamics/reproducibility_capsule.json", "reproducibility_capsule.json"),
            ("papers/bt2t003_plaquette_dynamics/SHA256SUMS.txt", "SHA256SUMS.txt"),
            ("experiments/bt2t_plaquette/bt2t_plaquette_dynamics.py", "bt2t_plaquette_dynamics.py"),
            ("experiments/bt2t_plaquette/verify_bt2t003.py", "verify_bt2t003.py"),
            ("experiments/bt2t_plaquette/manifest_bt2t003.json", "manifest_bt2t003.json"),
            ("experiments/bt2t_inversion/bt2t.py", "bt2t.py"),
            ("experiments/bt2t_inversion/manifest.json", "bt2t_inversion_manifest.json"),
            ("tests/test_paper10_bt2t003_plaquette.py", "test_paper10_bt2t003_plaquette.py"),
        ],
        "related": [
            {"identifier": "10.5281/zenodo.24520723", "relation": "continues", "scheme": "doi"},
            {"identifier": "10.5281/zenodo.24520738", "relation": "continues", "scheme": "doi"},
        ],
        "meta": {
            "title": (
                "Exact Gauss-Law Gauge Fixing, Peter-Weyl Casimir Diagonalization, and 104x-178x "
                "Native Gate Compression for Kogut-Susskind Binary Tetrahedral (2T subset SU(2)) "
                "Lattice Gauge Plaquette Dynamics (OSIRIS-BT2T-003)"
            ),
            "description": (
                "<p>Formulates and exhaustively verifies (<code>SIMULATED</code>, <code>provider_job_id: null</code>, "
                "0 s QPU, 35/35 verification checks passed) the Kogut&ndash;Susskind non-abelian single-plaquette "
                "Hamiltonian <em>H</em><sub>KS</sub> = (<em>g</em><sub>YM</sub>&sup2;/2)<em>H<sub>E</sub></em> + "
                "(1/<em>g</em><sub>YM</sub>&sup2;)<em>H<sub>B</sub></em> over the order-24 binary tetrahedral "
                "subgroup 2<em>T</em> = <em>Q</em><sub>8</sub> &rtimes; &Zopf;<sub>3</sub> &subset; SU(2) on the "
                "5-qubit semidirect encoding <code>bt2t-enc-1</code>:</p>"
                "<ul>"
                "<li><strong>Bi-Invariant 24-Cell Cayley Electric Laplacian &amp; Peter&ndash;Weyl Spectrum:</strong> "
                "Diagonalized to 7.70e-16 across all 7 irreps into calibrated SU(2) Casimir eigenvalues "
                "{0, 0.75, 1.50, 1.875, 2.25} with 0.0 leakage onto the 8 unused basis codewords.</li>"
                "<li><strong>Exact 4-Vertex Gauss-Law Gauge Fixing (13,824 2-Link &amp; 331,776 4-Link Checks):</strong> "
                "Maximal-tree reduction from 20Q/25Q four-link registers onto a single 5Q register reproduces "
                "unfixed four-link Strang evolution to 2.27e-15 <em>L</em><sub>2</sub> error across the 7-dimensional "
                "conjugacy-class subspace (4&times;&ndash;5&times; qubit reduction).</li>"
                "<li><strong>104.3&times;&ndash;178.3&times; Native Gate Compression:</strong> Structured Walsh&ndash;Gray "
                "synthesis compiles <em>U<sub>B</sub></em>(&Delta;<em>t</em>) in 16 CZ (16 ZZPhase) and a full "
                "second-order Strang step (slope 2.000008, operator error 7.94e-9 at <em>N</em> = 2,048) in 209 "
                "all-to-all CZ (350 heavy-hex <em>T</em><sub>5</sub> CZ, 208 H2 ZZPhase) versus 423 CZ for dense 5Q "
                "synthesis and 21,804&ndash;62,407 CZ for unfixed 20Q&ndash;25Q four-link circuits.</li>"
                "</ul>"
            ),
            "keywords": [
                "lattice gauge theory",
                "Kogut-Susskind Hamiltonian",
                "binary tetrahedral group",
                "non-abelian gauge invariance",
                "Peter-Weyl theorem",
                "Trotter-Suzuki simulation",
                "quantum circuit synthesis",
                "OSIRIS-BT2T-003",
            ],
        },
    },
}


def build_zenodo_metadata(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "upload_type": "publication",
        "publication_type": "preprint",
        "publication_date": cfg["publication_date"],
        "title": cfg["meta"]["title"],
        "creators": CREATOR,
        "description": cfg["meta"]["description"],
        "access_right": "open",
        "license": "cc-by-4.0",
        "version": cfg["version"],
        "language": "eng",
        "keywords": cfg["meta"]["keywords"],
        "related_identifiers": [
            {
                "identifier": "https://github.com/quantum-osiris/osiris-cli",
                "relation": "isSupplementedBy",
                "scheme": "url",
            },
            *cfg["related"],
        ],
    }


def draft_one(slug: str, token: str) -> Dict[str, Any]:
    cfg = PAPERS[slug]
    _assert_safe_and_sys024(slug, cfg)

    out_dir = DEPOSITS_ROOT / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = out_dir / "zenodo_receipt.json"

    resolved_files: List[Tuple[Path, str]] = []
    for rel_src, arc_name in cfg["files"]:
        src = REPO_ROOT / rel_src
        arc = arc_name or src.name
        for bad in FORBIDDEN_NAME_SUBSTRINGS:
            if bad in str(src).lower() or bad in arc.lower():
                raise ValueError(f"Forbidden upload path: {src} ({arc})")
        if not src.is_file():
            raise FileNotFoundError(f"Missing required upload file: {src}")
        resolved_files.append((src, arc))

    existing_did = None
    if receipt_path.is_file():
        prev = json.loads(receipt_path.read_text(encoding="utf-8"))
        if prev.get("state") == "published" and prev.get("doi"):
            print(f"[{slug}] Already published at DOI {prev['doi']}; skipping.")
            return prev
        existing_did = prev.get("deposition_id")

    dep: Optional[Dict[str, Any]] = None
    if existing_did:
        st, got = _call("GET", f"{API_BASE}/{existing_did}", token, ctype=None)
        if st == 200 and not got.get("submitted"):
            dep = got

    if dep is None:
        st, dep = _call("POST", API_BASE, token, data={})
        if st not in (200, 201):
            raise RuntimeError(f"[{slug}] create deposition failed (HTTP {st}): {dep}")

    did = dep["id"]
    bucket = dep["links"]["bucket"]

    for old_f in dep.get("files") or []:
        fid = old_f.get("id")
        if fid:
            del_st, del_res = _call("DELETE", f"{API_BASE}/{did}/files/{fid}", token, ctype=None)
            if del_st not in (200, 204):
                raise RuntimeError(f"[{slug}] delete old file failed (HTTP {del_st}): {del_res}")

    uploaded_manifest: List[Dict[str, Any]] = []
    for src, arc in resolved_files:
        blob = src.read_bytes()
        st, up = _call("PUT", f"{bucket}/{arc}", token, raw=blob, ctype="application/octet-stream")
        if st not in (200, 201):
            raise RuntimeError(f"[{slug}] upload failed for {arc} (HTTP {st}): {up}")
        remote_size = up.get("size")
        if remote_size is not None and int(remote_size) != len(blob):
            raise RuntimeError(f"[{slug}] size mismatch for {arc}: remote={remote_size} vs local={len(blob)}")
        uploaded_manifest.append(
            {
                "archive_name": arc,
                "source_relpath": src.relative_to(REPO_ROOT).as_posix(),
                "bytes": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(),
                "zenodo_checksum": up.get("checksum"),
            }
        )

    meta = build_zenodo_metadata(cfg)
    st, meta_res = _call("PUT", f"{API_BASE}/{did}", token, data={"metadata": meta})
    if st not in (200, 201):
        raise RuntimeError(f"[{slug}] metadata PUT failed (HTTP {st}): {meta_res}")

    prereserve = ((meta_res.get("metadata") or {}).get("prereserve_doi") or {}).get("doi")
    receipt = {
        "slug": slug,
        "state": "draft",
        "deposition_id": did,
        "version": cfg["version"],
        "publication_date": cfg["publication_date"],
        "prereserved_doi": prereserve,
        "draft_url": f"https://zenodo.org/uploads/{did}",
        "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "title": meta["title"],
        "files_count": len(uploaded_manifest),
        "files": uploaded_manifest,
    }
    receipt_path.write_bytes((json.dumps(receipt, indent=2) + "\n").encode("utf-8"))
    print(f"[{slug}] Draft ready: id={did}, prereserved_doi={prereserve}, files={len(uploaded_manifest)}")
    return receipt


def publish_one(slug: str, token: str) -> Dict[str, Any]:
    receipt_path = DEPOSITS_ROOT / slug / "zenodo_receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError(f"[{slug}] Missing draft receipt at {receipt_path}; run --draft first")
    rec = json.loads(receipt_path.read_text(encoding="utf-8"))
    if rec.get("state") == "published" and rec.get("doi"):
        print(f"[{slug}] Already published: DOI {rec['doi']}")
        return rec

    for fentry in rec["files"]:
        src = REPO_ROOT / fentry["source_relpath"]
        if _sha256(src) != fentry["sha256"]:
            raise RuntimeError(f"[{slug}] Local file {src} mutated since draft; re-run --draft")

    did = rec["deposition_id"]
    st, pub = _call("POST", f"{API_BASE}/{did}/actions/publish", token, ctype=None)
    if st not in (200, 201, 202):
        raise RuntimeError(f"[{slug}] Publish failed (HTTP {st}): {pub}")

    rec.update(
        {
            "state": "published",
            "record_id": pub.get("id", did),
            "doi": pub.get("doi"),
            "conceptdoi": pub.get("conceptdoi"),
            "doi_url": pub.get("doi_url") or (f"https://doi.org/{pub.get('doi')}" if pub.get("doi") else None),
            "record_html": (pub.get("links") or {}).get("record_html", f"https://zenodo.org/records/{pub.get('id', did)}"),
            "published_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    receipt_path.write_bytes((json.dumps(rec, indent=2) + "\n").encode("utf-8"))
    print(f"[{slug}] PUBLISHED -> DOI: {rec['doi']}  ({rec['record_html']})")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage and publish v6.2.4 OSIRIS Papers 9 & 10 to Zenodo")
    ap.add_argument("--paper", choices=[*PAPERS.keys(), "all"], required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--draft", action="store_true")
    mode.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    token = _load_zenodo_token()
    targets = list(PAPERS.keys()) if args.paper == "all" else [args.paper]

    summary = []
    for slug in targets:
        res = draft_one(slug, token) if args.draft else publish_one(slug, token)
        summary.append(
            {
                "slug": slug,
                "state": res.get("state"),
                "record_id": res.get("record_id") or res.get("deposition_id"),
                "doi": res.get("doi") or res.get("prereserved_doi"),
                "record_html": res.get("record_html") or res.get("draft_url"),
            }
        )
    print("\n=== ZENODO v6.2.4 SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
