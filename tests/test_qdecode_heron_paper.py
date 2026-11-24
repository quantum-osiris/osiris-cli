"""Verification suite for `papers/qdecode_heron_provenance_and_finite_shot_entropy/`."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from osiris_cli import claims, exec_kernel, plan_gates

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = REPO_ROOT / "papers" / "qdecode_heron_provenance_and_finite_shot_entropy"
SCRIPT_PATH = PAPER_DIR / "generate_qdecode_paper_assets.py"
TEX_PATH = PAPER_DIR / "qdecode_heron_provenance_and_finite_shot_entropy.tex"
PDF_PATH = PAPER_DIR / "qdecode_heron_provenance_and_finite_shot_entropy.pdf"
GEN_TEX_PATH = PAPER_DIR / "generated.tex"
MANIFEST_PATH = PAPER_DIR / "HARDWARE_PROVENANCE_MANIFEST.json"
CAPSULE_PATH = PAPER_DIR / "reproducibility_capsule.json"
SUMS_PATH = PAPER_DIR / "SHA256SUMS.txt"

FIGURE_STEMS = (
    "fig1_wire_deserialization_tvd_and_bit_order",
    "fig2_heron_59job_finite_shot_entropy_ceiling",
)

REQUIRED_TABLE_LABELS = (
    "tab:corpus_summary",
    "tab:executor_parity",
    "tab:scale_benchmarks",
)

STANDARD_LATEX_COMMANDS = frozenset(
    {
        "Rightarrow",
        "acknowledgments",
        "affiliation",
        "author",
        "begin",
        "bibitem",
        "bigl",
        "bigr",
        "caption",
        "centering",
        "cite",
        "date",
        "documentclass",
        "emph",
        "end",
        "eqref",
        "frac",
        "ge",
        "gg",
        "hat",
        "hline",
        "in",
        "includegraphics",
        "input",
        "item",
        "label",
        "lceil",
        "ldots",
        "le",
        "left",
        "leftrightarrow",
        "ll",
        "ln",
        "log",
        "lvert",
        "maketitle",
        "mathrm",
        "max",
        "min",
        "bmod",
        "neq",
        "newcommand",
        "notin",
        "paragraph",
        "pm",
        "quad",
        "rceil",
        "ref",
        "right",
        "ruledtabular",
        "rvert",
        "section",
        "sqrt",
        "sum",
        "text",
        "textbf",
        "textbackslash",
        "texttt",
        "textwidth",
        "times",
        "title",
        "usepackage",
        "equiv",
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


def test_qdecode_paper_deterministic_regeneration_byte_for_byte() -> None:
    tracked_files = [
        GEN_TEX_PATH,
        MANIFEST_PATH,
        CAPSULE_PATH,
        SUMS_PATH,
        TEX_PATH,
        PDF_PATH,
        SCRIPT_PATH,
        *(PAPER_DIR / f"{stem}.svg" for stem in FIGURE_STEMS),
        *(PAPER_DIR / f"{stem}.pdf" for stem in FIGURE_STEMS),
    ]
    for path in tracked_files:
        assert path.is_file(), f"Missing required deliverable: {path}"

    import importlib.util

    if importlib.util.find_spec("qiskit_ibm_runtime") is not None:
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
        assert before_bytes == after_bytes, "Re-running generate_qdecode_paper_assets.py mutated tracked deliverables"

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


def test_qdecode_paper_latex_macros_syntax_and_zero_prose_floats() -> None:
    tex = TEX_PATH.read_text(encoding="utf-8")
    gen = GEN_TEX_PATH.read_text(encoding="utf-8")

    for bad_token in ("TODO", "TBD", "FIXME", "XXX"):
        assert bad_token not in tex
        assert bad_token not in gen

    _check_balanced_braces(gen, "generated.tex")
    _check_balanced_braces(tex, TEX_PATH.name)
    _check_balanced_environments(tex, TEX_PATH.name)

    defined_macros = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", gen))
    assert len(defined_macros) >= 65

    assert not re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", _strip_latex_comments(tex))

    cleaned_tex = _strip_latex_comments(tex)
    used_commands = set(re.findall(r"\\([A-Za-z]+)", cleaned_tex))
    undefined = sorted(used_commands - defined_macros - STANDARD_LATEX_COMMANDS)
    assert not undefined, f"Undefined LaTeX macros in {TEX_PATH.name}: {undefined}"

    for label in REQUIRED_TABLE_LABELS:
        assert f"\\label{{{label}}}" in cleaned_tex

    preprint_bytes = PDF_PATH.read_bytes()
    assert preprint_bytes.startswith(b"%PDF-1.4") and preprint_bytes.rstrip().endswith(b"%%EOF")
    assert len(preprint_bytes) > 15_000

    for stem in FIGURE_STEMS:
        svg_txt = (PAPER_DIR / f"{stem}.svg").read_text(encoding="utf-8")
        pdf_bytes = (PAPER_DIR / f"{stem}.pdf").read_bytes()
        assert svg_txt.startswith("<svg ") and svg_txt.strip().endswith("</svg>")
        assert pdf_bytes.startswith(b"%PDF-1.4") and pdf_bytes.rstrip().endswith(b"%%EOF")
        assert f"{stem}.pdf" in cleaned_tex

    body = cleaned_tex.split(r"\begin{thebibliography}")[0]
    body_no_tt = re.sub(r"\\texttt\{[^}]*\}", "", body)
    raw_floats = re.findall(r"\b\d+\.\d+\b", body_no_tt)
    assert not raw_floats, f"Hand-typed floating-point numbers found in manuscript prose: {raw_floats}"


def test_qdecode_paper_claims_firewall_and_sys024_clean() -> None:
    combined_text = TEX_PATH.read_text(encoding="utf-8") + "\n" + GEN_TEX_PATH.read_text(encoding="utf-8")

    flagged = claims.flagged_in_reply(combined_text)
    assert not flagged, f"SYS-024 flagged claims in manuscript: {[c.id for c in flagged]}"

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
    assert not bad_matches, f"Quarantined SYS-024 claims matched: {[(c.id, c.verdict) for c in bad_matches]}"

    for label, pattern in plan_gates._QUARANTINED_PLAN_PATTERNS:
        assert not pattern.search(combined_text), f"Banned SYS-024 constant {label} matched"


def test_qdecode_paper_capsule_and_manifest_invariants() -> None:
    capsule = json.loads(CAPSULE_PATH.read_text(encoding="utf-8"))
    assert capsule["schema"] == "osiris.reproducibility_capsule/1"
    assert capsule["execution_tier"] == "HARDWARE_ATTESTED"

    recorded_digest = capsule["rfc8785_canonical_sha256"]
    body = {k: v for k, v in capsule.items() if k != "rfc8785_canonical_sha256"}
    recomputed = hashlib.sha256(exec_kernel.canon(body).encode("utf-8")).hexdigest()
    assert recorded_digest == recomputed

    totals = capsule["corpus_totals"]
    assert totals["total_jobs"] == 59
    assert totals["primitive_result_jobs"] == 51
    assert totals["executor_v02_jobs"] == 8
    assert totals["valid_measured_registers"] == 559
    assert totals["filtered_all_zero_registers"] == 24
    assert totals["qiskit_reference_disagreements"] == 0

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["total_jobs_verified"] == 59
    assert manifest["total_measured_registers"] == 559
    assert manifest["filtered_all_zero_registers"] == 24
    assert len(manifest["registers"]) == 559
