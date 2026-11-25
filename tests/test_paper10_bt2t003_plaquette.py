"""Verification suite for Paper 10 (`papers/bt2t003_plaquette_dynamics/`).

Verifies:
  1. Byte-for-byte deterministic regeneration of macros, dual SVG/PDF figures, standalone
     `%PDF-1.4` preprint (`>= 4` pages), `reproducibility_capsule.json`, and `SHA256SUMS.txt`
  2. Clean offline fallback regeneration without `qiskit`, `qiskit_ibm_runtime`, `pytket`,
     or `rfc8785` installed
  3. Zero hardcoded empirical floats in `bt2t003_plaquette_dynamics.tex`, balanced braces
     and environments, and 100% `\\Btp...` macro completeness
  4. `SYS-024` Claims Firewall across raw TeX, generated TeX, macro-expanded TeX, and PDF text
  5. Strict `SIMULATED` provenance (`provider_job_id is None`, `0` QPU seconds) and
     `sha256sum -c` verification across all 15 locked files.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from osiris_cli import claim_provenance, claims, plan_gates
from tests.test_standalone_paper_pdfs import _audit_pdf14_bytes, _extract_pdf_text

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = REPO_ROOT / "papers" / "bt2t003_plaquette_dynamics"
SCRIPT_PATH = PAPER_DIR / "generate_bt2t003_paper_assets.py"
TEX_PATH = PAPER_DIR / "bt2t003_plaquette_dynamics.tex"
PDF_PATH = PAPER_DIR / "bt2t003_plaquette_dynamics.pdf"
GEN_TEX_PATH = PAPER_DIR / "generated.tex"
CAPSULE_PATH = PAPER_DIR / "reproducibility_capsule.json"
SUMS_PATH = PAPER_DIR / "SHA256SUMS.txt"

FIGURE_STEMS = (
    "fig1_bt2t003_strang_scaling_and_wilson_regimes",
    "fig2_bt2t003_gauss_holonomy_cz_compression",
)

REQUIRED_LABELS = (
    "tab:btp_irreps",
    "tab:btp_regimes",
    "tab:btp_synthesis",
    "fig:btp_trotter",
    "fig:btp_cz",
    "eq:h_ks",
    "eq:gauss_vertices",
)

STANDARD_LATEX_COMMANDS = frozenset(
    {
        "Box",
        "Delta",
        "affiliation",
        "align",
        "author",
        "begin",
        "bibitem",
        "caption",
        "centering",
        "chi",
        "cite",
        "cong",
        "dagger",
        "date",
        "dim",
        "documentclass",
        "emph",
        "end",
        "exp",
        "frac",
        "hline",
        "in",
        "includegraphics",
        "input",
        "label",
        "lambda",
        "langle",
        "ldots",
        "le",
        "lor",
        "maketitle",
        "mathbb",
        "mathbf",
        "mathcal",
        "mathrm",
        "nonumber",
        "oplus",
        "otimes",
        "pm",
        "psi",
        "rangle",
        "ref",
        "rho",
        "rtimes",
        "ruledtabular",
        "section",
        "subset",
        "sum",
        "textbf",
        "texttt",
        "textwidth",
        "times",
        "title",
        "to",
        "usepackage",
        "widehat",
    }
)


def _tracked_paper_files() -> list[Path]:
    return [
        GEN_TEX_PATH,
        CAPSULE_PATH,
        SUMS_PATH,
        TEX_PATH,
        PDF_PATH,
        SCRIPT_PATH,
        *(PAPER_DIR / f"{stem}.svg" for stem in FIGURE_STEMS),
        *(PAPER_DIR / f"{stem}.pdf" for stem in FIGURE_STEMS),
    ]


def _strip_latex_comments(text: str) -> str:
    out_lines: list[str] = []
    for line in text.splitlines():
        idx = 0
        while True:
            pos = line.find("%", idx)
            if pos == -1:
                out_lines.append(line)
                break
            bs = 0
            k = pos - 1
            while k >= 0 and line[k] == "\\":
                bs += 1
                k -= 1
            if bs % 2 == 0:
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
            assert stack, f"Unexpected \\end{{{env}}} in {name}"
            top = stack.pop()
            assert top == env, f"Mismatched env in {name}: {top} vs {env}"
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


def test_paper10_deterministic_regeneration_and_sha256sums() -> None:
    tracked_files = _tracked_paper_files()
    for path in tracked_files:
        assert path.is_file(), f"Missing deliverable: {path}"

    before_bytes = {path.name: path.read_bytes() for path in tracked_files}
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Generator failed:\n{proc.stderr}\n{proc.stdout}"
    after_bytes = {path.name: path.read_bytes() for path in tracked_files}
    assert before_bytes == after_bytes

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
    assert len([ln for ln in chk.stdout.splitlines() if ln.strip()]) == 15


def test_paper10_offline_regeneration_without_optional_sdks_or_rfc8785() -> None:
    tracked_files = _tracked_paper_files()
    before_bytes = {path.name: path.read_bytes() for path in tracked_files}

    blocked_runner = (
        "import importlib.abc, runpy, sys\n"
        "BLOCKED = {'qiskit', 'qiskit_aer', 'qiskit_ibm_runtime', 'pytket', 'rfc8785'}\n"
        "class _BlockFinder(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path, target=None):\n"
        "        if fullname.split('.')[0] in BLOCKED:\n"
        "            raise ModuleNotFoundError(f'Blocked {fullname}')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _BlockFinder())\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        f"runpy.run_path({str(SCRIPT_PATH)!r}, run_name='__main__')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", blocked_runner],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Offline fallback failed:\n{proc.stderr}\n{proc.stdout}"
    after_bytes = {path.name: path.read_bytes() for path in tracked_files}
    assert before_bytes == after_bytes


def test_paper10_latex_macros_pdf_structure_and_sys024_firewall() -> None:
    tex = TEX_PATH.read_text(encoding="utf-8")
    gen = GEN_TEX_PATH.read_text(encoding="utf-8")

    for bad in ("TODO", "TBD", "FIXME", "XXX", "crn:v1:bluemix:"):
        assert bad not in tex and bad not in gen

    _check_balanced_braces(gen, "generated.tex")
    _check_balanced_braces(tex, TEX_PATH.name)
    _check_balanced_environments(tex, TEX_PATH.name)

    macros = _parse_macros(gen)
    assert len(macros) >= 55
    cleaned_tex = _strip_latex_comments(tex)
    assert not re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", cleaned_tex)

    used_commands = set(re.findall(r"\\([A-Za-z]+)", cleaned_tex))
    undefined = sorted(used_commands - set(macros.keys()) - STANDARD_LATEX_COMMANDS)
    assert not undefined, f"Undefined LaTeX macros: {undefined}"

    for lbl in REQUIRED_LABELS:
        assert f"\\label{{{lbl}}}" in cleaned_tex

    body = cleaned_tex.split(r"\begin{thebibliography}")[0]
    body_no_tt = re.sub(r"\\texttt\{[^}]*\}", "", body)
    raw_floats = re.findall(r"\b\d+\.\d+\b", body_no_tt)
    assert not raw_floats, f"Un-macroed floats in manuscript prose: {raw_floats}"

    pages, streams = _audit_pdf14_bytes(PDF_PATH.read_bytes(), PDF_PATH.name)
    assert pages >= 4
    pdf_txt = _extract_pdf_text(streams)
    for anchor in ("SIMULATED", "provider_job_id: null", "bt2t-enc-1", "331,776", "209", "350"):
        assert anchor in pdf_txt

    expanded = _expand_macros(tex, macros)
    quarantined_verdicts = {"REFUTED", "ARTIFACT", "NOT_MEASURED", "RULED_OUT", "OVERCLAIM", "NO_EVIDENCE"}
    for surf_name, surf_txt in [("raw", tex), ("gen", gen), ("expanded", expanded), ("pdf", pdf_txt)]:
        assert claims.flagged_in_reply(surf_txt) == [], f"Flagged claim in {surf_name}"
        bad_m = [c for c in claims.match(surf_txt) if c.verdict in quarantined_verdicts]
        assert not bad_m, f"Quarantined claim matched in {surf_name}: {[(c.id, c.verdict) for c in bad_m]}"
        for q_id, q_re in claim_provenance._QUARANTINED_CONSTANTS:
            assert not q_re.search(surf_txt), f"{q_id} matched in {surf_name}"
        for p_id, p_re in plan_gates._QUARANTINED_PLAN_PATTERNS:
            assert not p_re.search(surf_txt), f"{p_id} matched in {surf_name}"

    cap = json.loads(CAPSULE_PATH.read_text(encoding="utf-8"))
    assert cap["schema"] == "osiris.reproducibility_capsule/1"
    assert cap["execution_tier"] == "SIMULATED"
    assert cap["provider_job_id"] is None
    assert cap["qpu_seconds"] == 0
    from osiris_cli.orchestration_departments import jcs_bytes

    body_cap = {k: v for k, v in cap.items() if k != "rfc8785_canonical_sha256"}
    assert hashlib.sha256(jcs_bytes(body_cap)).hexdigest() == cap["rfc8785_canonical_sha256"]
