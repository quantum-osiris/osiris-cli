#!/usr/bin/env python3
"""Upload the TESSERACT-MAP-02 and CORRELATED-NOISE-01 papers and their data to Zenodo.

    python papers/publish_phase3_papers_to_zenodo.py --paper tmap02|cn01 --draft     # create/refresh a draft
    python papers/publish_phase3_papers_to_zenodo.py --paper tmap02|cn01 --publish   # publish that draft

The draft id is kept in papers/<dir>/zenodo_receipt.json, so --publish publishes exactly the draft
that was reviewed. Reads ZENODO_API_TOKEN from the environment or ~/osiris-ide/.env and never
prints it. Each upload's size is checked against Zenodo's reply and its SHA-256 is recorded
in the receipt; --publish refuses if any local file changed since the draft.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "https://zenodo.org/api/deposit/depositions"
P3 = ROOT / "docs" / "phase3"
CREATOR = [{"name": "Davis, Devin Phillip", "affiliation": "Agile Defense Systems LLC"}]


def _token() -> str:
    for k in ("ZENODO_API_TOKEN", "ZENODO_TOKEN"):
        if os.environ.get(k, "").strip():
            return os.environ[k].strip()
    env = Path.home() / "osiris-ide" / ".env"
    for line in env.read_text().splitlines():
        if line.startswith("ZENODO_API_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("ZENODO_API_TOKEN not found")


def _call(method: str, url: str, tok: str, data=None, raw: bytes | None = None, ctype="application/json"):
    body = raw if raw is not None else (json.dumps(data).encode() if data is not None else None)
    h = {"Authorization": f"Bearer {tok}"}
    if body is not None and ctype:
        h["Content-Type"] = ctype
    req = urllib.request.Request(url, method=method, data=body, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            t = r.read().decode()
            return r.status, (json.loads(t) if t else {})
    except urllib.error.HTTPError as e:
        t = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(t)
        except Exception:
            return e.code, {"raw": t[:800]}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


PAPERS = {
    "tmap02": {
        "dir": ROOT / "papers" / "tmap02_layout_sensitivity_marrakesh",
        "files": [
            ("papers/tmap02_layout_sensitivity_marrakesh/tmap02_layout_sensitivity_marrakesh.pdf", None),
            ("papers/tmap02_layout_sensitivity_marrakesh/tmap02_layout_sensitivity_marrakesh.tex", None),
            ("papers/tmap02_layout_sensitivity_marrakesh/generated.tex", None),
            ("papers/tmap02_layout_sensitivity_marrakesh/generate_tables.py", None),
            ("phase2/tesseract_map_02/TESSERACT_MAP_02_PREREGISTRATION.md", None),
            ("phase2/tesseract_map_02/TESSERACT_MAP_02_EXPECTED_OUTCOMES.md", None),
            ("phase2/tesseract_map_02/TESSERACT_MAP_02_RESOURCE_ESTIMATE.md", None),
            ("phase2/tesseract_map_02/tesseract_map_02.py", None),
            ("docs/phase3/PHASE3_PREREG_AMENDMENTS.md", None),
            ("docs/phase3/TESSERACT_MAP_02_RESULTS.md", None),
            ("docs/phase3/tesseract_map_02_live_prereg_lock.json", None),
            ("docs/phase3/tesseract_map_02_live_pubs.qpy", None),
            ("docs/phase3/ibm_marrakesh_prelock_calibration.json", None),
            ("docs/phase3/tesseract_map_02_submission_receipt.json", None),
            ("docs/phase3/tesseract_map_02_live_raw_counts.json", None),
            ("docs/phase3/tesseract_map_02_live_analysis.json", None),
            ("docs/phase3/tesseract_map_02_live_sensitivity.json", None),
            ("docs/phase3/tesseract_map_02_job_usage.json", None),
            ("phase3/execute_and_analyze_live.py", None),
            ("phase3/tmap02_supplement.py", None),
        ],
        "meta": {
            "title": ("Same gate count, half the fidelity: a preregistered decomposition of layout, routing "
                      "and optimisation-level effects on a 40-qubit circuit on ibm_marrakesh (TESSERACT-MAP-02)"),
            "description": (
                "<p>Preregistered, blocked hardware experiment on the 156-qubit IBM Heron r2 processor "
                "<code>ibm_marrakesh</code> (job <code>dbto408n26qs739tnrj0</code>; 10 blocks &times; 9 circuits "
                "&times; 2,048 shots; 53 s QPU) separating four compilation factors for a 40-qubit circuit with an "
                "exact four-string output distribution.</p><ul>"
                "<li>Rewriting a degree-20 CNOT star into an equivalent degree-2 chain and removing all SWAPs on the "
                "same 40 qubits raised classical fidelity by +14.27 pp (opt 3) and +12.47 pp (opt 1).</li>"
                "<li>At a fixed 39 CZ and depth 66, a calibration-ranked best path beat a median path by +10.54 pp "
                "(95% CI [+9.95, +11.12]), halving the fidelity.</li>"
                "<li>A greedy calibration-weighted path beat Qiskit's stock level-3 layout by +5.70 pp, the reverse "
                "of a FakeMarrakesh simulation made before the run.</li>"
                "<li>A virtual-Z A/A control was null (&minus;0.24 pp, 95% CI [&minus;0.88, +0.41]).</li>"
                "<li>Union readout calibration found 22.3% and 12.1% assignment error on two qubits IBM listed "
                "near 1%; both lie on the losing layouts.</li></ul>"
                "<p>All effect contrasts survive Holm correction and an exact Wilcoxon check. One job on one device; "
                "these are compilation and device-heterogeneity effects only. The record contains the paper, LaTeX "
                "source, preregistration and amendment A-1, the lock file (SHA-256 5f4a04c4&hellip;) and QPY "
                "(409dc736&hellip;) pushed before submission, raw counts for all 90 PUBs, analysis outputs and the "
                "scripts that regenerate every number.</p>"
            ),
            "keywords": ["quantum compilation", "qubit layout", "SWAP routing", "IBM Heron r2", "ibm_marrakesh",
                         "preregistration", "readout error", "VF2Layout", "TESSERACT-MAP-02"],
        },
    },
    "cn01": {
        "dir": ROOT / "papers" / "cn01_correlated_noise_null_marrakesh",
        "files": [
            ("papers/cn01_correlated_noise_null_marrakesh/cn01_correlated_noise_null_marrakesh.pdf", None),
            ("papers/cn01_correlated_noise_null_marrakesh/cn01_correlated_noise_null_marrakesh.tex", None),
            ("papers/cn01_correlated_noise_null_marrakesh/generated.tex", None),
            ("papers/cn01_correlated_noise_null_marrakesh/generate_tables.py", None),
            ("phase2/correlated_noise_01/CORRELATED_NOISE_01_PREREGISTRATION.md", None),
            ("phase2/correlated_noise_01/CORRELATED_NOISE_01_THEORY.md", None),
            ("phase2/correlated_noise_01/CORRELATED_NOISE_01_IDENTIFIABILITY.md", None),
            ("phase2/correlated_noise_01/CORRELATED_NOISE_01_HARDWARE_PLAN.md", None),
            ("phase2/correlated_noise_01/correlated_noise_models.py", None),
            ("phase2/correlated_noise_01/simulate_correlated_noise.py", None),
            ("docs/phase3/PHASE3_PREREG_AMENDMENTS.md", None),
            ("docs/phase3/CORRELATED_NOISE_01_RESULTS.md", None),
            ("docs/phase3/correlated_noise_01_live_prereg_lock.json", None),
            ("docs/phase3/correlated_noise_01_live_pubs.qpy", None),
            ("docs/phase3/ibm_marrakesh_prelock_calibration.json", None),
            ("docs/phase3/correlated_noise_01_A3_amendment.json", None),
            ("docs/phase3/correlated_noise_01_A3_validation.json", None),
            ("docs/phase3/correlated_noise_01_submission_receipt.json", None),
            ("docs/phase3/correlated_noise_01_live_raw_counts.json", None),
            ("docs/phase3/correlated_noise_01_live_analysis.json", None),
            ("docs/phase3/correlated_noise_01_live_analysis_A3.json", None),
            ("docs/phase3/correlated_noise_01_A3_bound_diagnostic.json", None),
            ("docs/phase3/correlated_noise_01_job_usage.json", None),
            ("phase3/execute_and_analyze_live.py", None),
            ("phase3/cn01_amendment_a3.py", None),
            ("phase3/cn01_a3_validation.py", None),
            ("phase3/cn01_a3_bound_diagnostic.py", None),
        ],
        "meta": {
            "title": ("No evidence of spatially correlated or non-Markovian dephasing across ten couplers of "
                      "ibm_marrakesh, and why a white-noise null is not enough to look for it (CORRELATED-NOISE-01 Stage 1)"),
            "description": (
                "<p>Stage 1 of a preregistered hardware test for non-local dephasing on the 156-qubit IBM Heron r2 "
                "processor <code>ibm_marrakesh</code> (job <code>dbtp67dav06s73cfbsm0</code>; 244 PUBs, 499,712 "
                "circuit shots, 139 s QPU). Ten couplers &ge;5 hops apart were measured simultaneously: Bell states "
                "under DD with 0&ndash;256 ns sublattice staggers, &Phi;+ vs &Psi;+, active vs idle spectators, echo "
                "and DD pulse-count series, Ramsey arms.</p><ul>"
                "<li>No &Phi;+/&Psi;+ asymmetry (C_DFS = +0.43 pp, 95% CI [&minus;0.13, +1.00]), no spectator effect, "
                "no memory contrast, no pulse-collision step.</li>"
                "<li>The analysis as locked selects a non-Markovian model over white noise by &Delta;BIC = 10,263 but "
                "fails its preregistered held-out test (p = 0.82): the transferable memory claim is falsified.</li>"
                "<li>Synthetic validation shows that analysis selects the memory model on 100% of purely local 1/f or "
                "OU noise; an amended analysis with local coloured-noise nulls (hashed before the job, published after "
                "it, as disclosed) selects the local model. Verdict under both: local noise only.</li></ul>"
                "<p>Bounds fast correlated dephasing at about 1 pp; says nothing about correlations slower than about "
                "1 &micro;s. The record contains the paper, LaTeX source, preregistration, theory, amendments A-2/A-3 "
                "with validation, the lock (38ebfe7f&hellip;) and QPY (158de032&hellip;), raw counts, both analyses and "
                "the scripts that regenerate every number.</p>"
            ),
            "keywords": ["correlated noise", "non-Markovian dephasing", "dynamical decoupling", "Bell states",
                         "IBM Heron r2", "ibm_marrakesh", "preregistration", "null result", "model selection",
                         "CORRELATED-NOISE-01"],
        },
    },
    "tmap03pre": {
        "dir": ROOT / "papers" / "tmap03_prereg",
        "related": [{"identifier": "10.5281/zenodo.24406426", "relation": "continues", "scheme": "doi"}],
        "files": [
            ("phase3/TESSERACT_MAP_03_PREREGISTRATION.md", None),
            ("docs/phase3/tesseract_map_03_live_prereg_lock.json", None),
            ("docs/phase3/tesseract_map_03_live_pubs.qpy", None),
            ("docs/phase3/tesseract_map_03_prelock_calibration.json", None),
            ("docs/phase3/tesseract_map_03_dryrun.json", None),
            ("phase3/tesseract_map_03.py", None),
            ("phase2/tesseract_map_02/tesseract_map_02.py", None),
            ("phase3/live_hardware_runner.py", None),
            ("phase3/execute_and_analyze_live.py", None),
        ],
        "meta": {
            "title": ("Preregistration: TESSERACT-MAP-03, replication, readout attribution and rewrite x path "
                      "interaction for a 40-qubit circuit on ibm_marrakesh"),
            "description": (
                "<p>Preregistration, published before submission, of a follow-up to TESSERACT-MAP-02 "
                "(doi:10.5281/zenodo.24406426). One job on <code>ibm_marrakesh</code>: 10 blocks &times; 8 circuits "
                "&times; 2,048 shots (80 PUBs, est. 49.5 s QPU). Arms replay MAP-02's best-path, median-path and "
                "stock-layout qubits, add today's stock layout, and add the missing star/median-path cell; union "
                "readout calibration on all 81 qubits.</p><p>Five Holm-corrected contrasts: K1 path-gap replication; "
                "K2 readout share of the gap (raw minus tensored readout-mitigated); K3 rewrite &times; path interaction; "
                "K4 best path vs MAP-02 stock qubits; K5 best path vs today's stock layout. Contains the "
                "preregistration, the lock (SHA-256 ea7e8765&hellip;), the compiled QPY (f15167c4&hellip;), the "
                "lock-time calibration, a noiseless simulation and synthetic-mitigation dry run, and the build, "
                "analysis and submission code (20beb4e3&hellip;).</p>"
            ),
            "keywords": ["preregistration", "quantum compilation", "qubit layout", "readout error mitigation",
                         "IBM Heron r2", "ibm_marrakesh", "TESSERACT-MAP-03"],
        },
    },
    "tmap03": {
        "dir": ROOT / "papers" / "tmap03_readout_attribution_marrakesh",
        "related": [{"identifier": "10.5281/zenodo.24406496", "relation": "isSupplementTo", "scheme": "doi"},
                    {"identifier": "10.5281/zenodo.24406426", "relation": "continues", "scheme": "doi"}],
        "files": [
            ("papers/tmap03_readout_attribution_marrakesh/tmap03_readout_attribution_marrakesh.pdf", None),
            ("papers/tmap03_readout_attribution_marrakesh/tmap03_readout_attribution_marrakesh.tex", None),
            ("papers/tmap03_readout_attribution_marrakesh/generated.tex", None),
            ("papers/tmap03_readout_attribution_marrakesh/generate_tables.py", None),
            ("phase3/TESSERACT_MAP_03_PREREGISTRATION.md", None),
            ("docs/phase3/tesseract_map_03_live_prereg_lock.json", None),
            ("docs/phase3/tesseract_map_03_live_pubs.qpy", None),
            ("docs/phase3/tesseract_map_03_prelock_calibration.json", None),
            ("docs/phase3/tesseract_map_03_dryrun.json", None),
            ("docs/phase3/tesseract_map_03_submission_receipt.json", None),
            ("docs/phase3/tesseract_map_03_live_raw_counts.json", None),
            ("docs/phase3/tesseract_map_03_live_analysis.json", None),
            ("phase3/tesseract_map_03.py", None),
        ],
        "meta": {
            "title": ("Four days later, the best qubits were the worst: a preregistered replication on ibm_marrakesh "
                      "in which readout faults invisible to the published calibration reversed a 10 pp layout effect "
                      "(TESSERACT-MAP-03)"),
            "description": (
                "<p>Preregistered follow-up (doi:10.5281/zenodo.24406496, published before submission) to "
                "TESSERACT-MAP-02 (doi:10.5281/zenodo.24406426). Job <code>dc0ajv5s2omc738915k0</code> on "
                "<code>ibm_marrakesh</code>, 80 PUBs, 47 s QPU, same physical qubits as MAP-02 four days later.</p><ul>"
                "<li>The 10.5 pp best-vs-median path gap did not replicate: it reversed to &minus;4.29 pp "
                "(95% CI [&minus;5.24, &minus;3.33], Holm p = 6.4e-6).</li>"
                "<li>Cause: qubit 46 on the best path read |0&gt; as 1 in 54% of shots while IBM's calibration "
                "listed ~1.1% readout error; q31 and q37 also degraded.</li>"
                "<li>After tensored readout mitigation the best path leads again by +17.0 pp.</li>"
                "<li>The path effect is not additive across chain and routed star (interaction &minus;5.64 pp).</li>"
                "<li>Today's stock level-3 layout beat the MAP-02 best path by 1.75 pp.</li></ul>"
                "<p>Layout rankings built on published calibration can be overturned within days by single-qubit "
                "readout faults; a per-job union readout check catches them. Record contains the paper, source, "
                "preregistration, lock, QPY, calibrations, raw counts, analysis and code.</p>"
            ),
            "keywords": ["quantum compilation", "qubit layout", "readout error", "replication", "calibration drift",
                         "IBM Heron r2", "ibm_marrakesh", "preregistration", "TESSERACT-MAP-03"],
        },
    },
}


def build_manifest(cfg) -> Path:
    lines = []
    for rel, _ in cfg["files"]:
        p = ROOT / rel
        if not p.is_file():
            raise SystemExit(f"missing {rel}")
        lines.append(f"{sha(p)}  {rel}")
    out = cfg["dir"] / "SHA256SUMS.txt"
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", choices=sorted(PAPERS), required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--draft", action="store_true")
    g.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    cfg = PAPERS[a.paper]
    receipt_path = cfg["dir"] / "zenodo_receipt.json"
    tok = _token()

    if a.publish:
        rec = json.loads(receipt_path.read_text())
        dep = rec["deposition_id"]
        # refuse to publish if any local file changed since the draft was uploaded
        for name, h in rec["files"].items():
            src = ROOT / rec["sources"][name]
            if sha(src) != h:
                raise SystemExit(f"{name} changed since the draft was made; re-run --draft first")
        st, r = _call("POST", f"{API}/{dep}/actions/publish", tok)
        if st != 202:
            raise SystemExit(f"publish failed HTTP {st}: {r}")
        rec.update(state="published", doi=r.get("doi"), record_url=r.get("links", {}).get("record_html"),
                   published_utc=datetime.now(timezone.utc).isoformat())
        receipt_path.write_text(json.dumps(rec, indent=2) + "\n")
        print("published", r.get("doi"), r.get("links", {}).get("record_html"))
        return 0

    manifest = build_manifest(cfg)
    files = [(ROOT / rel, Path(rel).name) for rel, _ in cfg["files"]] + [(manifest, "SHA256SUMS.txt")]
    names = [n for _, n in files]
    assert len(set(names)) == len(names), "duplicate upload names"

    if receipt_path.is_file() and json.loads(receipt_path.read_text()).get("state") == "draft":
        dep = json.loads(receipt_path.read_text())["deposition_id"]
        st, d = _call("GET", f"{API}/{dep}", tok)
        for fobj in d.get("files", []):
            _call("DELETE", f"{API}/{dep}/files/{fobj['id']}", tok)
    else:
        st, d = _call("POST", API, tok, data={})
        if st != 201:
            raise SystemExit(f"create failed HTTP {st}: {d}")
        dep = d["id"]
    st, d = _call("GET", f"{API}/{dep}", tok)
    bucket = d["links"]["bucket"]

    meta = {
        "upload_type": "publication",
        "publication_type": "preprint",
        "publication_date": "2026-11-20",
        "creators": CREATOR,
        "access_right": "open",
        "license": "cc-by-4.0",
        "version": "1.0.0",
        "language": "eng",
        **cfg["meta"],
        "related_identifiers": [
            {"identifier": "https://github.com/quantum-osiris/osiris-cli", "relation": "isSupplementedBy", "scheme": "url"},
        ] + cfg.get("related", []),
    }
    st, r = _call("PUT", f"{API}/{dep}", tok, data={"metadata": meta})
    if st != 200:
        raise SystemExit(f"metadata failed HTTP {st}: {r}")

    rec = {"paper": a.paper, "deposition_id": dep, "state": "draft", "files": {}, "sources": {},
           "draft_html": d["links"].get("html"), "prereserved_doi": r.get("metadata", {}).get("prereserve_doi", {}).get("doi"),
           "drafted_utc": datetime.now(timezone.utc).isoformat()}
    for src, name in files:
        raw = src.read_bytes()
        st, up = _call("PUT", f"{bucket}/{name}", tok, raw=raw, ctype="application/octet-stream")
        if st not in (200, 201) or up.get("size") != len(raw):
            raise SystemExit(f"upload {name} failed HTTP {st}: {up}")
        rec["files"][name] = hashlib.sha256(raw).hexdigest()
        rec["sources"][name] = str(src.relative_to(ROOT))
    receipt_path.write_text(json.dumps(rec, indent=2) + "\n")
    print("draft", dep, rec["prereserved_doi"], len(files), "files", rec["draft_html"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
