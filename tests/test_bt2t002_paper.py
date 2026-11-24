"""Verification suite for `papers/bt2t002_mul_and_nonabelian_qft/` (`OSIRIS-BT2T-002`).

Verifies:
  1. Byte-for-byte deterministic regeneration of macros, dual SVG/PDF figures, standalone
     `%PDF-1.4` preprint (`>= 4` pages), `reproducibility_capsule.json`, and `SHA256SUMS.txt`
  2. Clean offline fallback regeneration without `qiskit`, `qiskit_ibm_runtime`, `pytket`,
     or `rfc8785` installed (via `osiris_cli.orchestration_departments.jcs_bytes`)
  3. Zero hardcoded empirical floats in `bt2t002_mul_and_nonabelian_qft.tex` prose or tables,
     balanced braces/environments, and 100% `\btm...` macro completeness
  4. `SYS-024` Claims Firewall (`claims.flagged_in_reply`, `claims.match`, `plan_gates`,
     and `claim_provenance`) across raw TeX, generated TeX, macro-expanded TeX, and PDF text
  5. `SIMULATED` / `OFFLINE_COMPILED` provenance (`provider_job_id is None`, `0` QPU seconds)
     and CRN / secret / token absence across all manuscript and experiment files
  6. Clean `sha256sum -c` verification across all 17 locked paper and source artifacts
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
PAPER_DIR = REPO_ROOT / "papers" / "bt2t002_mul_and_nonabelian_qft"
PRIM_DIR = REPO_ROOT / "experiments" / "bt2t_primitives"

SCRIPT_PATH = PAPER_DIR / "generate_bt2t002_paper_assets.py"
TEX_PATH = PAPER_DIR / "bt2t002_mul_and_nonabelian_qft.tex"
PDF_PATH = PAPER_DIR / "bt2t002_mul_and_nonabelian_qft.pdf"
GEN_TEX_PATH = PAPER_DIR / "generated.tex"
CAPSULE_PATH = PAPER_DIR / "reproducibility_capsule.json"
SUMS_PATH = PAPER_DIR / "SHA256SUMS.txt"

FIGURE_STEMS = (
    "fig1_bt2t002_semidirect_mul_compression",
    "fig2_bt2t002_peter_weyl_qft_and_hsp",
)

REQUIRED_LABELS = (
    "tab:btm_compilation",
    "tab:btm_irreps",
    "tab:btm_hsp",
    "tab:btm_feasibility",
    "fig:btm_mul_compression",
    "fig:btm_qft_hsp",
    "eq:extended_mul",
    "eq:ad_omega_linear",
    "eq:semidirect_law",
    "eq:q8_cocycle",
    "eq:mobius_gate",
    "eq:mul_unitary_exact",
    "eq:plancherel_dims",
    "eq:peter_weyl_qft",
    "eq:block_diag",
    "eq:frobenius_hsp",
)

STANDARD_LATEX_COMMANDS = frozenset(
    {
        "Bigl",
        "Bigr",
        "Delta",
        "Phi",
        "Pi",
        "acknowledgments",
        "affiliation",
        "alpha",
        "approx",
        "author",
        "bar",
        "begin",
        "bibitem",
        "bigoplus",
        "bigl",
        "bigr",
        "bmod",
        "caption",
        "cases",
        "cdot",
        "centering",
        "chi",
        "circ",
        "cite",
        "cong",
        "cup",
        "dagger",
        "date",
        "det",
        "documentclass",
        "dots",
        "emph",
        "emptyset",
        "end",
        "epsilon",
        "eqref",
        "equiv",
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
        "langle",
        "lceil",
        "le",
        "leftarrow",
        "ll",
        "log",
        "lor",
        "maketitle",
        "mapsto",
        "mathbb",
        "mathbf",
        "mathrm",
        "max",
        "mid",
        "mu",
        "neg",
        "neq",
        "nonumber",
        "o",
        "omega",
        "oplus",
        "otimes",
        "overline",
        "paragraph",
        "pi",
        "pm",
        "pmod",
        "prod",
        "qquad",
        "quad",
        "rangle",
        "rceil",
        "ref",
        "rho",
        "rtimes",
        "ruledtabular",
        "section",
        "setminus",
        "sigma",
        "sqcup",
        "sqrt",
        "subset",
        "subsection",
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
        "widetilde",
        "zeta",
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


def test_bt2t002_paper_deterministic_regeneration_and_sha256sums() -> None:
    tracked_files = _tracked_paper_files()
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
    assert before_bytes == after_bytes, "Re-running generate_bt2t002_paper_assets.py mutated tracked files"

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
    assert len(verified_lines) == 17


def test_bt2t002_offline_regeneration_without_optional_sdks_or_rfc8785() -> None:
    tracked_files = _tracked_paper_files()
    before_bytes = {path.name: path.read_bytes() for path in tracked_files}

    blocked_runner = (
        "import importlib.abc, runpy, sys\n"
        "BLOCKED = {'qiskit', 'qiskit_aer', 'qiskit_ibm_runtime', 'pytket', 'rfc8785'}\n"
        "class _BlockFinder(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path, target=None):\n"
        "        if fullname.split('.')[0] in BLOCKED:\n"
        "            raise ModuleNotFoundError(f'Blocked {fullname} for offline test')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _BlockFinder())\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        f"runpy.run_path({str(SCRIPT_PATH)!r}, run_name='__main__')\n"
    )
    proc_offline = subprocess.run(
        [sys.executable, "-c", blocked_runner],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_offline.returncode == 0, (
        f"Offline fallback regeneration failed:\n{proc_offline.stderr}\n{proc_offline.stdout}"
    )
    after_offline_bytes = {path.name: path.read_bytes() for path in tracked_files}
    assert before_bytes == after_offline_bytes, (
        "Offline fallback without qiskit/qiskit_ibm_runtime/pytket/rfc8785 mutated tracked files"
    )


def test_bt2t002_latex_macros_zero_prose_floats_and_structure() -> None:
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
    assert all(name.startswith("btm") and name.isalpha() for name in defined_macros)

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

    # Verify locked Milestone 2 & Milestone 1 macros match manifest_bt2t002.json
    assert macros["btmExecTier"] == r"\texttt{SIMULATED}"
    assert macros["btmCompileTier"] == r"\texttt{OFFLINE\_COMPILED}"
    assert macros["btmJobId"] == r"\texttt{null}"
    assert macros["btmQpuSec"] == "0"
    assert macros["btmMulBasisStates"] == "1,024"
    assert macros["btmValidPairs"] == "576"
    assert macros["btmUnusedPairs"] == "448"
    assert macros["btmMulStructCcx"] == "270"
    assert macros["btmMulStructCx"] == "35"
    assert macros["btmMulStructX"] == "18"
    assert macros["btmMulStructCswap"] == "14"
    assert macros["btmMulStructDepth"] == "300"
    assert macros["btmMulGenMcx"] == "2,040"
    assert macros["btmMulGenCx"] == "1,656"
    assert macros["btmMulGenX"] == "1,064"
    assert macros["btmMulGenCswap"] == "144"
    assert macros["btmMulGenDepth"] == "3,760"
    assert macros["btmMulStructOpErr"] == "0.0"
    assert macros["btmMulAtaStructCz"] == "1,739"
    assert macros["btmMulAtaGenCz"] == "33,042"
    assert macros["btmMulAtaCzRatio"] == r"19.00\times"
    assert macros["btmMulHhStructCz"] == "3,662"
    assert macros["btmMulHhGenCz"] == "78,035"
    assert macros["btmMulHhCzRatio"] == r"21.31\times"
    assert macros["btmQftUnitarityErr"] == r"9.80 \times 10^{-16}"
    assert macros["btmIrrepHomMaxErr"] == r"7.03 \times 10^{-16}"
    assert macros["btmLeftOffBlockErr"] == r"3.11 \times 10^{-16}"
    assert macros["btmQeightMaxForbLeak"] == r"1.62 \times 10^{-33}"
    assert macros["btmQftAtaCz"] == "423"
    assert macros["btmQftTfCz"] == "613"
    assert macros["btmQftTkZz"] == "423"
    assert macros["btmMoneAtaRatio"] == r"8.33\times"
    assert macros["btmMoneFezRatio"] == r"10.04\times"


def test_bt2t002_standalone_pdf14_structure_and_anchors() -> None:
    pdf_bytes = PDF_PATH.read_bytes()
    assert len(pdf_bytes) > 20_000
    page_count, streams = _audit_pdf14_bytes(pdf_bytes, PDF_PATH.name)
    assert page_count >= 4, f"Expected >= 4 pages in {PDF_PATH.name}, got {page_count}"
    assert b"/MediaBox [0 0 612.00 792.00]" in pdf_bytes
    pdf_text = _extract_pdf_text(streams)

    for page_idx in range(1, page_count + 1):
        assert f"Page {page_idx} of {page_count}" in pdf_text

    required_pdf_anchors = (
        "Devin Phillip Davis",
        "Agile Defense Systems",
        "SIMULATED",
        "OFFLINE_COMPILED",
        "PROXY",
        "provider_job_id: null",
        "bt2t-enc-1",
        "1,024",
        "576",
        "448",
        "1,739",
        "33,042",
        "19.00x",
        "3,662",
        "78,035",
        "21.31x",
        "423",
        "613",
        "9.80e-16",
        "7.03e-16",
        "3.11e-16",
        "1.62e-33",
        "1/3",
        "QCCD",
        "dc2ee3uq4uuc73a56o6g",
    )
    for anchor in required_pdf_anchors:
        assert anchor in pdf_text, f"Missing anchor {anchor!r} in {PDF_PATH.name}"


def test_bt2t002_sys024_claims_firewall_across_all_surfaces() -> None:
    raw_tex = TEX_PATH.read_text(encoding="utf-8")
    gen_tex = GEN_TEX_PATH.read_text(encoding="utf-8")
    expanded_tex = _expand_macros(raw_tex, _parse_macros(gen_tex))
    _, streams = _audit_pdf14_bytes(PDF_PATH.read_bytes(), PDF_PATH.name)
    pdf_text = _extract_pdf_text(streams)

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


def test_bt2t002_simulated_capsule_provenance_and_secret_redaction() -> None:
    capsule = json.loads(CAPSULE_PATH.read_text(encoding="utf-8"))
    assert capsule["schema"] == "osiris.reproducibility_capsule/1"
    assert capsule["paper_id"] == "OSIRIS-BT2T-002-PAPER"
    assert capsule["experiment_id"] == "OSIRIS-BT2T-002"
    assert capsule["execution_tier"] == "SIMULATED"
    assert capsule["compilation_tier"] == "OFFLINE_COMPILED"
    assert capsule["evidence_class"] == "SIMULATED"
    assert capsule["hardware_executed"] is False
    assert capsule["provider_job_id"] is None
    assert capsule["qpu_seconds"] == 0
    assert capsule["all_checks_passed"] is True
    assert capsule["checks_passed_count"] == 31
    assert capsule["checks_total_count"] == 31

    body = {k: v for k, v in capsule.items() if k != "rfc8785_canonical_sha256"}
    recomputed = hashlib.sha256(exec_kernel.canon(body).encode("utf-8")).hexdigest()
    assert capsule["rfc8785_canonical_sha256"] == recomputed

    for rel_path, expected_sha in capsule["source_artifacts"].items():
        actual_sha = hashlib.sha256((REPO_ROOT / rel_path).read_bytes()).hexdigest()
        assert actual_sha == expected_sha, f"Source artifact hash mismatch for {rel_path}"

    for folder in (PAPER_DIR, PRIM_DIR):
        for p in folder.iterdir():
            if p.is_file() and p.suffix in {".tex", ".json", ".md", ".svg", ".txt", ".py"}:
                txt = p.read_text(encoding="utf-8")
                assert "crn:v1:bluemix:" not in txt, f"CRN leaked in {p}"
                assert "ZENODO_TOKEN=" not in txt, f"Zenodo token assignment leaked in {p}"
                if p.suffix != ".py":
                    assert '"apikey"' not in txt, f"Secret apikey field leaked in {p}"
