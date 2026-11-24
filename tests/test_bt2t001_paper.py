"""Verification suite for `papers/bt2t001_binary_tetrahedral_fez/` (`OSIRIS-BT2T-001-FEZ`).

Verifies:
  1. Byte-for-byte deterministic regeneration of macros, dual SVG/PDF figures, standalone
     `%PDF-1.4` preprint (`>= 4` pages), `reproducibility_capsule.json`, and `SHA256SUMS.txt`
  2. Zero hardcoded empirical floats in `bt2t001_binary_tetrahedral_fez.tex` prose or tables,
     balanced braces/environments, and 100% macro completeness
  3. `SYS-024` Claims Firewall (`claims.flagged_in_reply`, `claims.match`, `plan_gates`,
     and `claim_provenance`) across raw TeX, generated TeX, macro-expanded TeX, and PDF text
  4. CRN / secret / token absence across all manuscript and experiment files
  5. Clean `sha256sum -c` verification across all 22 locked paper and hardware artifacts
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from osiris_cli import claim_provenance, claims, exec_kernel, plan_gates
from tests.test_standalone_paper_pdfs import _audit_pdf14_bytes, _extract_pdf_text

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = REPO_ROOT / "papers" / "bt2t001_binary_tetrahedral_fez"
FEZ_DIR = REPO_ROOT / "experiments" / "bt2t_fez"
INV_DIR = REPO_ROOT / "experiments" / "bt2t_inversion"

SCRIPT_PATH = PAPER_DIR / "generate_bt2t001_paper_assets.py"
TEX_PATH = PAPER_DIR / "bt2t001_binary_tetrahedral_fez.tex"
PDF_PATH = PAPER_DIR / "bt2t001_binary_tetrahedral_fez.pdf"
GEN_TEX_PATH = PAPER_DIR / "generated.tex"
CAPSULE_PATH = PAPER_DIR / "reproducibility_capsule.json"
SUMS_PATH = PAPER_DIR / "SHA256SUMS.txt"

FIGURE_STEMS = (
    "fig1_bt2t001_fez_synthesis_and_sectors",
    "fig2_bt2t001_fez_elementwise_and_spam_drift",
)

REQUIRED_LABELS = (
    "tab:bt2t_compilation",
    "tab:bt2t_hardware",
    "tab:bt2t_spam",
    "fig:bt2t_sectors",
    "fig:bt2t_elementwise_spam",
    "eq:su2_map",
    "eq:boolean_factoring",
    "eq:sabre_layouts",
    "eq:metrics",
    "eq:marginal_product",
)

STANDARD_LATEX_COMMANDS = frozenset(
    {
        "Bigl",
        "Bigr",
        "Delta",
        "Pi",
        "acknowledgments",
        "affiliation",
        "author",
        "bar",
        "begin",
        "bibitem",
        "bigl",
        "bigr",
        "caption",
        "centering",
        "cite",
        "cong",
        "cup",
        "dagger",
        "date",
        "det",
        "documentclass",
        "dots",
        "emph",
        "end",
        "epsilon",
        "frac",
        "ge",
        "hline",
        "in",
        "includegraphics",
        "infty",
        "input",
        "item",
        "label",
        "land",
        "le",
        "leftarrow",
        "lor",
        "maketitle",
        "mapsto",
        "mathbb",
        "mathbf",
        "mathrm",
        "mid",
        "mu",
        "neg",
        "neq",
        "nonumber",
        "oplus",
        "otimes",
        "paragraph",
        "pi",
        "pm",
        "prod",
        "qquad",
        "quad",
        "rangle",
        "ref",
        "rtimes",
        "ruledtabular",
        "section",
        "setminus",
        "sigma",
        "sqrt",
        "subsection",
        "subset",
        "sum",
        "text",
        "textbf",
        "texttt",
        "textwidth",
        "tfrac",
        "times",
        "title",
        "to",
        "triangleleft",
        "usepackage",
        "widehat",
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


def _parse_macros(gen_tex: str) -> dict[str, str]:
    macros: dict[str, str] = {}
    pattern = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{")
    pos = 0
    while True:
        m = pattern.search(gen_tex, pos)
        if not m:
            break
        name = m.group(1)
        idx = m.end()
        depth = 1
        start = idx
        while idx < len(gen_tex) and depth > 0:
            ch = gen_tex[idx]
            if ch == "\\" and idx + 1 < len(gen_tex):
                idx += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            idx += 1
        if depth == 0:
            macros[name] = gen_tex[start : idx - 1]
        pos = idx
    return macros


def _expand_macros(tex: str, macros: dict[str, str]) -> str:
    out = tex
    for name in sorted(macros.keys(), key=len, reverse=True):
        val = macros[name]
        out = re.sub(rf"\\{name}(?:\{{\}}|\b)", lambda _m, v=val: v, out)
    return out


def test_bt2t001_paper_deterministic_regeneration_and_sha256sums() -> None:
    tracked_files = [
        GEN_TEX_PATH,
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
    assert before_bytes == after_bytes, "Re-running generate_bt2t001_paper_assets.py mutated tracked files"

    sha256sum_bin = shutil.which("sha256sum")
    assert sha256sum_bin is not None
    chk = subprocess.run(
        [sha256sum_bin, "-c", str(SUMS_PATH.relative_to(REPO_ROOT))],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert chk.returncode == 0, f"sha256sum -c failed:\n{chk.stderr}\n{chk.stdout}"
    verified_lines = [ln for ln in chk.stdout.splitlines() if ln.strip()]
    assert len(verified_lines) == 22


def test_bt2t001_latex_macros_zero_prose_floats_and_structure() -> None:
    tex = TEX_PATH.read_text(encoding="utf-8")
    gen = GEN_TEX_PATH.read_text(encoding="utf-8")

    for bad_token in ("TODO", "TBD", "FIXME", "XXX"):
        assert bad_token not in tex
        assert bad_token not in gen

    _check_balanced_braces(gen, "generated.tex")
    _check_balanced_braces(tex, TEX_PATH.name)
    _check_balanced_environments(tex, TEX_PATH.name)

    macros = _parse_macros(gen)
    defined_macros = set(macros.keys())
    assert len(defined_macros) >= 150
    assert all(name.startswith("bt") and name.isalpha() for name in defined_macros)

    cleaned_tex = _strip_latex_comments(tex)
    assert not re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", cleaned_tex)

    used_commands = set(re.findall(r"\\([A-Za-z]+)", cleaned_tex))
    undefined = sorted(used_commands - defined_macros - STANDARD_LATEX_COMMANDS)
    assert not undefined, f"Undefined LaTeX commands in {TEX_PATH.name}: {undefined}"

    for label in REQUIRED_LABELS:
        assert f"\\label{{{label}}}" in cleaned_tex

    for stem in FIGURE_STEMS:
        svg_txt = (PAPER_DIR / f"{stem}.svg").read_text(encoding="utf-8")
        fig_pdf = (PAPER_DIR / f"{stem}.pdf").read_bytes()
        assert svg_txt.startswith("<svg ") and svg_txt.strip().endswith("</svg>")
        fig_pages, _ = _audit_pdf14_bytes(fig_pdf, f"{stem}.pdf")
        assert fig_pages == 1
        assert f"{stem}.pdf" in cleaned_tex

    # Strictly zero empirical floats anywhere in manuscript prose or tables (outside bibliography)
    body = cleaned_tex.split(r"\begin{thebibliography}")[0]
    raw_floats = re.findall(r"\b\d+\.\d+\b", body)
    assert not raw_floats, f"Hardcoded floating-point literals found in {TEX_PATH.name}: {raw_floats}"

    # Verify key empirical macros match hardware artifacts
    assert macros["btJobId"] == r"\texttt{dc2ee3uq4uuc73a56o6g}"
    assert macros["btStructCz"] == "72"
    assert macros["btGenCz"] == "723"
    assert macros["btCtrlCz"] == "0"
    assert macros["btQpuSec"] == "11"
    assert macros["btRoCond"] == "1.4524"
    assert macros["btQTwentyFourCalPct"] == r"0.33\%"
    assert macros["btQTwentyFourLivePct"] == r"4.92\%"
    assert macros["btQTwentyFourStaleX"] == r"14.91\times"
    assert macros["btQTwentyOneCalPct"] == r"2.34\%"
    assert macros["btQTwentyOneLivePct"] == r"6.33\%"
    assert macros["btQTwentyOneStaleX"] == r"2.70\times"
    assert macros["btHOneVerdict"] == r"\texttt{SUPPORTED}"
    assert macros["btHTwoVerdict"] == r"\texttt{SUPPORTED}"
    assert macros["btHThreeVerdict"] == r"\texttt{REFUTED}"


def test_bt2t001_standalone_pdf_sys024_capsule_and_secret_redaction() -> None:
    pdf_bytes = PDF_PATH.read_bytes()
    page_count, streams = _audit_pdf14_bytes(pdf_bytes, PDF_PATH.name)
    assert page_count >= 4, f"Expected >= 4 pages in {PDF_PATH.name}, got {page_count}"
    pdf_text = _extract_pdf_text(streams)

    required_pdf_anchors = (
        "HARDWARE_ATTESTED",
        "dc2ee3uq4uuc73a56o6g",
        "bt2t-enc-1",
        "N_windows = 1",
        "Card A",
        "27,648",
        "0.5337",
        "0.0661",
        "0.8643",
        "0.9076",
        "73.23",
        "14.91x",
        "2.70x",
        "1.4524",
        "SUPPORTED",
        "REFUTED",
    )
    for anchor in required_pdf_anchors:
        assert anchor in pdf_text, f"Missing anchor {anchor!r} in {PDF_PATH.name}"

    raw_tex = TEX_PATH.read_text(encoding="utf-8")
    gen_tex = GEN_TEX_PATH.read_text(encoding="utf-8")
    expanded_tex = _expand_macros(raw_tex, _parse_macros(gen_tex))

    quarantined_verdicts = {
        "REFUTED",
        "ARTIFACT",
        "NOT_MEASURED",
        "RULED_OUT",
        "OVERCLAIM",
        "NO_EVIDENCE",
    }
    for surface_name, surface_text in (
        ("raw_tex", raw_tex),
        ("generated_tex", gen_tex),
        ("expanded_tex", expanded_tex),
        ("pdf_text", pdf_text),
    ):
        flagged = claims.flagged_in_reply(surface_text, [])
        assert not flagged, f"SYS-024 flagged in {surface_name}: {[c.id for c in flagged]}"
        matched = claims.match(surface_text)
        bad_matches = [c for c in matched if c.verdict in quarantined_verdicts]
        assert not bad_matches, f"Quarantined claim in {surface_name}: {[(c.id, c.verdict) for c in bad_matches]}"
        for p_id, p_re in plan_gates._QUARANTINED_PLAN_PATTERNS:
            assert not p_re.search(surface_text), f"Banned constant {p_id} in {surface_name}"
        for q_id, q_re in claim_provenance._QUARANTINED_CONSTANTS:
            assert not q_re.search(surface_text), f"Quarantined constant {q_id} in {surface_name}"

    capsule = json.loads(CAPSULE_PATH.read_text(encoding="utf-8"))
    assert capsule["schema"] == "osiris.reproducibility_capsule/1"
    assert capsule["execution_tier"] == "HARDWARE_ATTESTED"
    assert capsule["backend"] == "ibm_fez"
    assert capsule["provider_job_id"] == "dc2ee3uq4uuc73a56o6g"
    assert capsule["combined_shots"] == 27648
    assert capsule["combined_qpu_seconds"] == 11
    assert capsule["unit_of_inference"]["calibration_windows"] == 1
    assert capsule["hypothesis_verdicts"] == {
        "H1_structured_vs_generic_valid24": "SUPPORTED",
        "H2_unused_codeword_preservation_and_confinement": "SUPPORTED",
        "H3_spam_control_integrity": "REFUTED",
    }
    body = {k: v for k, v in capsule.items() if k != "rfc8785_canonical_sha256"}
    recomputed = hashlib.sha256(exec_kernel.canon(body).encode("utf-8")).hexdigest()
    assert capsule["rfc8785_canonical_sha256"] == recomputed

    for rel_path, expected_sha in capsule["source_artifacts"].items():
        actual_sha = hashlib.sha256((REPO_ROOT / rel_path).read_bytes()).hexdigest()
        assert actual_sha == expected_sha, f"Source artifact hash mismatch for {rel_path}"

    # CRN / secret absence scan across paper and experiment directories
    for folder in (PAPER_DIR, FEZ_DIR, INV_DIR):
        for p in folder.iterdir():
            if p.is_file() and p.suffix in {".tex", ".json", ".md", ".svg", ".txt", ".py"}:
                txt = p.read_text(encoding="utf-8")
                assert "crn:v1:bluemix:" not in txt, f"CRN leaked in {p}"
                assert "ZENODO_TOKEN=" not in txt, f"Zenodo token assignment leaked in {p}"
                if p.suffix != ".py":
                    assert '"apikey"' not in txt, f"Secret apikey field leaked in {p}"
