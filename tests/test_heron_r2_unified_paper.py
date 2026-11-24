"""Verification suite for the unified Heron r2 calibration-staleness and colored-nulls paper."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from osiris_cli import claims, exec_kernel, plan_gates

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = REPO_ROOT / "papers" / "heron_r2_calibration_staleness_and_colored_nulls"
SCRIPT_PATH = PAPER_DIR / "generate_unified_assets.py"
TEX_PATH = PAPER_DIR / "heron_r2_calibration_staleness_and_colored_nulls.tex"
GEN_TEX_PATH = PAPER_DIR / "generated.tex"
CAPSULE_PATH = PAPER_DIR / "reproducibility_capsule.json"
SUMS_PATH = PAPER_DIR / "SHA256SUMS.txt"

FIGURE_STEMS = (
    "fig1_tmap_sign_reversal_three_qpus",
    "fig2_ro06_07_staleness_persistence",
    "fig3_cn01_colored_noise_heldout_gate",
)

REQUIRED_TABLE_LABELS = (
    "tab:campaign_registry",
    "tab:tmap_sign_reversal",
    "tab:ro_staleness",
    "tab:cn01_heldout_nulls",
)

STANDARD_LATEX_COMMANDS = frozenset(
    {
        "Alpha",
        "Beta",
        "Chi",
        "Delta",
        "Epsilon",
        "Eta",
        "Gamma",
        "Kappa",
        "L",
        "Lambda",
        "Mu",
        "Nu",
        "Omega",
        "Phi",
        "Pi",
        "Psi",
        "Rho",
        "Sigma",
        "Tau",
        "Theta",
        "Upsilon",
        "Xi",
        "Zeta",
        "acknowledgments",
        "addlinespace",
        "affiliation",
        "alpha",
        "approx",
        "author",
        "begin",
        "beta",
        "bibitem",
        "bigl",
        "bigr",
        "bottomrule",
        "caption",
        "centering",
        "chi",
        "circ",
        "cite",
        "cmidrule",
        "cos",
        "date",
        "documentclass",
        "dots",
        "emph",
        "end",
        "eqref",
        "frac",
        "gamma",
        "ge",
        "hline",
        "in",
        "includegraphics",
        "input",
        "item",
        "label",
        "langle",
        "ldots",
        "le",
        "ln",
        "longrightarrow",
        "lvert",
        "maketitle",
        "mathcal",
        "mathrm",
        "midrule",
        "mu",
        "multicolumn",
        "newcommand",
        "noindent",
        "nonumber",
        "otimes",
        "paragraph",
        "pi",
        "pm",
        "qquad",
        "quad",
        "rangle",
        "ref",
        "rho",
        "ruledtabular",
        "rvert",
        "section",
        "sin",
        "sqrt",
        "subsection",
        "sum",
        "tau",
        "text",
        "textbf",
        "texttt",
        "textwidth",
        "theta",
        "times",
        "title",
        "to",
        "toprule",
        "usepackage",
    }
)


def _strip_latex_comments(text: str) -> str:
    out_lines: list[str] = []
    for line in text.splitlines():
        idx = 0
        while True:
            pos = line.find("%", idx)
            if pos == -1:
                out_lines.append(line)
                break
            backslashes = 0
            k = pos - 1
            while k >= 0 and line[k] == "\\":
                backslashes += 1
                k -= 1
            if backslashes % 2 == 0:
                out_lines.append(line[:pos])
                break
            idx = pos + 1
    return "\n".join(out_lines)


def _check_balanced_braces(text: str, name: str) -> None:
    cleaned = _strip_latex_comments(text)
    depth = 0
    i = 0
    n = len(cleaned)
    while i < n:
        ch = cleaned[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            assert depth >= 0, f"Unmatched closing brace in {name} near offset {i}"
        i += 1
    assert depth == 0, f"Unbalanced braces in {name}: final depth={depth}"


def _check_balanced_environments(text: str, name: str) -> None:
    cleaned = _strip_latex_comments(text)
    stack: list[str] = []
    for match in re.finditer(r"\\(begin|end)\{([^}]+)\}", cleaned):
        kind, env = match.group(1), match.group(2)
        if kind == "begin":
            stack.append(env)
        else:
            assert stack, f"Unexpected \\end{{{env}}} with empty stack in {name}"
            top = stack.pop()
            assert top == env, f"Mismatched environment in {name}: \\begin{{{top}}} vs \\end{{{env}}}"
    assert not stack, f"Unclosed environments in {name}: {stack}"


def test_deterministic_regeneration_byte_for_byte() -> None:
    tracked_files = [
        GEN_TEX_PATH,
        CAPSULE_PATH,
        SUMS_PATH,
        TEX_PATH,
        SCRIPT_PATH,
        *(PAPER_DIR / f"{stem}.svg" for stem in FIGURE_STEMS),
        *(PAPER_DIR / f"{stem}.pdf" for stem in FIGURE_STEMS),
    ]
    for path in tracked_files:
        assert path.is_file(), f"Missing required deliverable: {path}"

    before_bytes = {path.name: path.read_bytes() for path in tracked_files}

    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Asset generator failed:\n{proc.stderr}\n{proc.stdout}"

    after_bytes = {path.name: path.read_bytes() for path in tracked_files}
    assert before_bytes == after_bytes, "Re-running generate_unified_assets.py mutated tracked deliverables"

    # Verify SHA256SUMS.txt matches every file in PAPER_DIR
    sum_entries: dict[str, str] = {}
    for line in SUMS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, fname = line.strip().split("  ", 1)
            sum_entries[fname] = digest

    actual_files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in PAPER_DIR.iterdir()
        if p.is_file() and p.name != "SHA256SUMS.txt"
    }
    assert sum_entries == actual_files


def test_latex_macros_syntax_and_zero_placeholders() -> None:
    tex = TEX_PATH.read_text(encoding="utf-8")
    gen = GEN_TEX_PATH.read_text(encoding="utf-8")

    for bad_token in ("TODO", "TBD", "FIXME", "XXX"):
        assert bad_token not in tex, f"Placeholder {bad_token} found in manuscript"
        assert bad_token not in gen, f"Placeholder {bad_token} found in generated.tex"

    _check_balanced_braces(gen, "generated.tex")
    _check_balanced_braces(tex, TEX_PATH.name)
    _check_balanced_environments(tex, TEX_PATH.name)

    defined_macros = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", gen))
    assert len(defined_macros) >= 120

    # Ensure zero \newcommand definitions appear inline inside the main manuscript
    assert not re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", _strip_latex_comments(tex))

    cleaned_tex = _strip_latex_comments(tex)
    used_commands = set(re.findall(r"\\([A-Za-z]+)", cleaned_tex))
    undefined = sorted(used_commands - defined_macros - STANDARD_LATEX_COMMANDS)
    assert not undefined, f"Undefined LaTeX macros in {TEX_PATH.name}: {undefined}"

    for label in REQUIRED_TABLE_LABELS:
        assert f"\\label{{{label}}}" in cleaned_tex, f"Missing required table label {label}"

    for stem in FIGURE_STEMS:
        svg_txt = (PAPER_DIR / f"{stem}.svg").read_text(encoding="utf-8")
        pdf_bytes = (PAPER_DIR / f"{stem}.pdf").read_bytes()
        assert svg_txt.startswith("<svg ") and svg_txt.strip().endswith("</svg>")
        assert pdf_bytes.startswith(b"%PDF-1.4") and pdf_bytes.rstrip().endswith(b"%%EOF")
        assert f"{stem}.pdf" in cleaned_tex

    # Verify zero un-macroed floating-point empirical numbers in manuscript body prose
    body = cleaned_tex.split(r"\begin{thebibliography}")[0]
    body_no_tt = re.sub(r"\\texttt\{[^}]*\}", "", body)
    raw_floats = re.findall(r"\b\d+\.\d+\b", body_no_tt)
    assert not raw_floats, f"Hand-typed floating-point numbers found in manuscript prose: {raw_floats}"


def test_claims_firewall_and_sys024_quarantine() -> None:
    combined_text = (
        TEX_PATH.read_text(encoding="utf-8")
        + "\n"
        + GEN_TEX_PATH.read_text(encoding="utf-8")
    )

    # Zero un-negated refuted/quarantined claims in claims.flagged_in_reply
    flagged = claims.flagged_in_reply(combined_text)
    assert not flagged, f"SYS-024 flagged claims in manuscript: {[c.id for c in flagged]}"

    # Zero SYS-024 refuted / artifact / not-measured / ruled-out / overclaim / no-evidence hits in claims.match
    quarantined_verdicts = {
        "REFUTED",
        "ARTIFACT",
        "NOT_MEASURED",
        "RULED_OUT",
        "OVERCLAIM",
        "NO_EVIDENCE",
    }
    matched = claims.match(combined_text)
    bad_matches = [c for c in matched if c.verdict in quarantined_verdicts]
    assert not bad_matches, (
        f"Quarantined SYS-024 claims matched in manuscript: "
        f"{[(c.id, c.verdict) for c in bad_matches]}"
    )

    # Zero plan_gates REQ-PLAN-3 / SYS-024 banned constants
    for label, pattern in plan_gates._QUARANTINED_PLAN_PATTERNS:
        assert not pattern.search(combined_text), f"Banned SYS-024 constant {label} matched"


def test_reproducibility_capsule_and_rfc8785_attestation() -> None:
    capsule = json.loads(CAPSULE_PATH.read_text(encoding="utf-8"))
    assert capsule["schema"] == "osiris.reproducibility_capsule/1"
    assert capsule["execution_tier"] == "HARDWARE_ATTESTED"

    recorded_digest = capsule["rfc8785_canonical_sha256"]
    assert re.fullmatch(r"[0-9a-f]{64}", recorded_digest)

    body = {k: v for k, v in capsule.items() if k != "rfc8785_canonical_sha256"}
    canonical_str = exec_kernel.canon(body)
    recomputed_digest = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
    assert recorded_digest == recomputed_digest

    jobs = capsule["verified_hardware_jobs"]
    assert len(jobs) == 42
    seen_ids: set[str] = set()
    for job in jobs:
        jid = job["job_id"]
        assert isinstance(jid, str) and len(jid.strip()) >= 10
        assert jid not in seen_ids
        seen_ids.add(jid)
        assert job["execution_tier"] == "HARDWARE_ATTESTED"
        assert job["backend"] in {"ibm_marrakesh", "ibm_fez", "ibm_kingston"}
        assert job["pubs"] > 0 and job["shots"] == job["pubs"] * 2048
        assert job["qpu_seconds"] > 0

    totals = capsule["aggregate_totals"]
    assert totals["jobs"] == len(jobs)
    assert totals["pubs"] == sum(j["pubs"] for j in jobs)
    assert totals["total_shots"] == sum(j["shots"] for j in jobs)
    assert totals["total_qpu_seconds"] == sum(j["qpu_seconds"] for j in jobs)

    dois = {z["doi"] for z in capsule["zenodo_records"]}
    for required_doi in (
        "10.5281/zenodo.24406435",
        "10.5281/zenodo.24406426",
        "10.5281/zenodo.24406525",
        "10.5281/zenodo.24407001",
        "10.5281/zenodo.24407875",
        "10.5281/zenodo.24420902",
    ):
        assert required_doi in dois

    for src in capsule["source_json_artifacts"]:
        src_path = REPO_ROOT / src["path"]
        assert src_path.is_file(), f"Missing capsule source JSON: {src_path}"
        assert hashlib.sha256(src_path.read_bytes()).hexdigest() == src["sha256"]
