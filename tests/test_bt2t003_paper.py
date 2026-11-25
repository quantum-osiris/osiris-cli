r"""Verification suite for Paper 10: `papers/bt2t003_kogut_susskind_plaquette/`
(`OSIRIS-BT2T-003` + `OSIRIS-BT2T-003-FEZ`).

Verifies:
  1. Byte-for-byte deterministic regeneration (`SOURCE_DATE_EPOCH=1700000000`, `PYTHONHASHSEED=0`)
     of `generated_macros.tex`, 4 LaTeX tables, 3 dual `{pdf,png}` figures, standalone multi-page
     `%PDF-1.4` preprint (`CreationDate=2023-11-14T22:13:20Z`), and `SHA256SUMS.txt`
  2. Clean `sha256sum -c SHA256SUMS.txt` across all 18 locked deliverables and upstream manifests
     (`../../experiments/bt2t_plaquette/manifest_bt2t003.json` and
     `../../experiments/bt2t003_fez/physical_bt2t003_fez_results.json`)
  3. Clean offline regeneration without `qiskit`, `qiskit_ibm_runtime`, `pytket`, or `rfc8785`
  4. Zero hardcoded floating-point literals in `main.tex` or `tables/*.tex`, balanced braces and
     environments, and 100% `\BTPlaq...` macro completeness and exact numeric binding
  5. Multipage `%PDF-1.4` structure, `/CreationDate (2023-11-14T22:13:20Z)`, PNG header/chunk
     integrity, and presence of all `SIMULATED` and `HARDWARE_ATTESTED` anchors
  6. `SYS-024` Claims Firewall (`claims.flagged_in_reply`, `claims.match`, `plan_gates`,
     `claim_provenance`) and secret/token/CRN redaction across all surfaces
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

from osiris_cli import claim_provenance, claims, plan_gates
from tests.test_standalone_paper_pdfs import _audit_pdf14_bytes, _extract_pdf_text

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = REPO_ROOT / "papers" / "bt2t003_kogut_susskind_plaquette"
TABLES_DIR = PAPER_DIR / "tables"
FIGURES_DIR = PAPER_DIR / "figures"

MANIFEST_BT2T003_PATH = REPO_ROOT / "experiments" / "bt2t_plaquette" / "manifest_bt2t003.json"
FEZ_RESULTS_PATH = REPO_ROOT / "experiments" / "bt2t003_fez" / "physical_bt2t003_fez_results.json"

SCRIPT_PATH = PAPER_DIR / "build_paper_assets.py"
TEX_PATH = PAPER_DIR / "main.tex"
BIB_PATH = PAPER_DIR / "references.bib"
README_PATH = PAPER_DIR / "README.md"
MACROS_PATH = PAPER_DIR / "generated_macros.tex"
PREPRINT_PATH = PAPER_DIR / "bt2t003_kogut_susskind_plaquette_preprint.pdf"
ARCHIVE_PREPRINT_PATH = Path(
    "/home/osiris/osiris-assessment/work/preprints_2026-11-23/bt2t003_kogut_susskind_plaquette_preprint.pdf"
)
SUMS_PATH = PAPER_DIR / "SHA256SUMS.txt"

TABLE_FILES = (
    "table1_conjugacy_magnetic_potential.tex",
    "table2_strang_scaling_and_gate_compression.tex",
    "table3_fez_10placement_echo_and_plaquette.tex",
    "table4_fez_character_sector_trajectories.tex",
)

FIGURE_STEMS = (
    "fig1_magnetic_potential_and_strang_scaling",
    "fig2_fez_character_sector_trajectories",
    "fig3_two_window_echo_drift_catch",
)

REQUIRED_LABELS = (
    "tab:bt2t003_conjugacy_magnetic_potential",
    "tab:bt2t003_strang_scaling_and_gate_compression",
    "tab:bt2t003_fez_10placement_echo_and_plaquette",
    "tab:bt2t003_fez_character_sector_trajectories",
    "fig:bt2t003_magnetic_potential_and_strang_scaling",
    "fig:bt2t003_fez_character_sector_trajectories",
    "fig:bt2t003_two_window_echo_drift_catch",
    "eq:plaquette_holonomy",
    "eq:magnetic_potential",
    "eq:electric_kinetic_peter_weyl",
    "eq:strang_splitting",
    "eq:character_ramsey_frequency",
    "eq:top1_regret_and_spearman",
)

STANDARD_LATEX_COMMANDS = frozenset(
    {
        "Delta",
        "Phi",
        "Pi",
        "Re",
        "Tr",
        "acknowledgments",
        "affiliation",
        "author",
        "bar",
        "begin",
        "bibliography",
        "bigl",
        "bigr",
        "caption",
        "cases",
        "lceil",
        "centering",
        "chi",
        "cite",
        "cong",
        "cos",
        "dagger",
        "date",
        "documentclass",
        "dots",
        "emph",
        "end",
        "exp",
        "frac",
        "gamma",
        "hat",
        "hline",
        "hook",
        "hookrightarrow",
        "implies",
        "in",
        "includegraphics",
        "input",
        "item",
        "label",
        "lambda",
        "langle",
        "le",
        "left",
        "log",
        "maketitle",
        "mapsto",
        "mathbb",
        "mathbf",
        "mathcal",
        "mathrm",
        "max",
        "omega",
        "oplus",
        "operatorname",
        "otimes",
        "pi",
        "pm",
        "qquad",
        "quad",
        "rangle",
        "rceil",
        "ref",
        "rho",
        "right",
        "rtimes",
        "ruledtabular",
        "section",
        "sqrt",
        "subset",
        "subsection",
        "sum",
        "tau",
        "text",
        "textbf",
        "texttt",
        "textwidth",
        "tfrac",
        "times",
        "title",
        "to",
        "usepackage",
        "widetilde",
    }
)


def _tracked_paper_files() -> list[Path]:
    return [
        README_PATH,
        PREPRINT_PATH,
        SCRIPT_PATH,
        MACROS_PATH,
        TEX_PATH,
        BIB_PATH,
        SUMS_PATH,
        *(TABLES_DIR / fn for fn in TABLE_FILES),
        *(FIGURES_DIR / f"{stem}.pdf" for stem in FIGURE_STEMS),
        *(FIGURES_DIR / f"{stem}.png" for stem in FIGURE_STEMS),
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
            assert top == env, (
                f"Mismatched environment in {name}: \\begin{{{top}}} vs \\end{{{env}}}"
            )
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


def test_bt2t003_deterministic_regeneration_and_sha256sums() -> None:
    tracked = _tracked_paper_files()
    for p in tracked:
        assert p.is_file(), f"Missing required deliverable: {p}"

    before = {p.relative_to(PAPER_DIR).as_posix(): p.read_bytes() for p in tracked}

    env = dict(os.environ)
    env["SOURCE_DATE_EPOCH"] = "1700000000"
    env["PYTHONHASHSEED"] = "0"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"build_paper_assets.py failed:\n{proc.stderr}\n{proc.stdout}"

    after = {p.relative_to(PAPER_DIR).as_posix(): p.read_bytes() for p in tracked}
    assert before == after, "Re-running build_paper_assets.py mutated tracked deliverables"

    if ARCHIVE_PREPRINT_PATH.is_file():
        assert ARCHIVE_PREPRINT_PATH.read_bytes() == PREPRINT_PATH.read_bytes()

    sha256sum_bin = shutil.which("sha256sum")
    assert sha256sum_bin is not None
    chk = subprocess.run(
        [sha256sum_bin, "-c", "SHA256SUMS.txt"],
        cwd=str(PAPER_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    assert chk.returncode == 0, f"sha256sum -c failed:\n{chk.stderr}\n{chk.stdout}"
    verified_lines = [ln for ln in chk.stdout.splitlines() if ln.strip()]
    assert len(verified_lines) == 18


def test_bt2t003_offline_regeneration_without_optional_sdks() -> None:
    tracked = _tracked_paper_files()
    before = {p.relative_to(PAPER_DIR).as_posix(): p.read_bytes() for p in tracked}

    blocked_runner = (
        "import importlib.abc, runpy, sys\n"
        "BLOCKED = {'qiskit', 'qiskit_aer', 'qiskit_ibm_runtime', 'pytket', 'rfc8785', 'matplotlib'}\n"
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
        f"Offline regeneration failed:\n{proc_offline.stderr}\n{proc_offline.stdout}"
    )
    after = {p.relative_to(PAPER_DIR).as_posix(): p.read_bytes() for p in tracked}
    assert before == after, "Offline fallback regeneration mutated tracked files"


def test_bt2t003_macros_tables_zero_prose_floats_and_manifest_binding() -> None:
    manifest = json.loads(MANIFEST_BT2T003_PATH.read_text(encoding="utf-8"))
    fez_results = json.loads(FEZ_RESULTS_PATH.read_text(encoding="utf-8"))

    tex = TEX_PATH.read_text(encoding="utf-8")
    gen = MACROS_PATH.read_text(encoding="utf-8")
    bib = BIB_PATH.read_text(encoding="utf-8")

    for bad_token in ("TODO", "TBD", "FIXME", "XXX"):
        assert bad_token not in tex
        assert bad_token not in gen
        assert bad_token not in bib

    _check_balanced_braces(gen, "generated_macros.tex")
    _check_balanced_braces(tex, "main.tex")
    _check_balanced_environments(tex, "main.tex")

    macros = _parse_macros(gen)
    defined_macros = set(macros.keys())
    assert len(defined_macros) >= 250
    assert all(name.startswith("BTPlaq") and name.isalpha() for name in defined_macros)

    cleaned_main = _strip_latex_comments(tex)
    assert not re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", cleaned_main)

    # Zero hardcoded floating-point literals in main.tex or any generated table .tex
    raw_floats_main = re.findall(r"\b\d+\.\d+\b", cleaned_main)
    assert not raw_floats_main, f"Hardcoded floats in main.tex: {raw_floats_main}"

    combined_cleaned = cleaned_main
    for t_fn in TABLE_FILES:
        t_path = TABLES_DIR / t_fn
        t_txt = t_path.read_text(encoding="utf-8")
        _check_balanced_braces(t_txt, t_fn)
        _check_balanced_environments(t_txt, t_fn)
        cleaned_t = _strip_latex_comments(t_txt)
        raw_floats_t = re.findall(r"\b\d+\.\d+\b", cleaned_t)
        assert not raw_floats_t, f"Hardcoded floats in {t_fn}: {raw_floats_t}"
        assert f"\\input{{tables/{t_fn}}}" in cleaned_main
        combined_cleaned += "\n" + cleaned_t

    used_commands = set(re.findall(r"\\([A-Za-z]+)", combined_cleaned))
    undefined = sorted(used_commands - defined_macros - STANDARD_LATEX_COMMANDS)
    assert not undefined, f"Undefined LaTeX commands in main.tex + tables: {undefined}"

    for label in REQUIRED_LABELS:
        assert f"\\label{{{label}}}" in combined_cleaned, f"Missing label {label}"

    # Verify BibTeX keys cited in main.tex exist in references.bib
    cited_keys: set[str] = set()
    for m in re.finditer(r"\\cite\{([^}]+)\}", cleaned_main):
        for k in m.group(1).split(","):
            cited_keys.add(k.strip())
    assert len(cited_keys) >= 6
    for k in cited_keys:
        assert re.search(rf"@\w+\{{{re.escape(k)},", bib), f"Missing BibTeX entry for {k}"

    # Verify exact numeric bindings to manifest_bt2t003.json and physical_bt2t003_fez_results.json
    assert manifest["all_checks_passed"] is True
    assert macros["BTPlaqChecksPassed"] == "34"
    assert macros["BTPlaqChecksTotal"] == "34"
    assert macros["BTPlaqSimExperimentId"] == r"\texttt{OSIRIS-BT2T-003}"
    assert macros["BTPlaqSimExecTier"] == r"\texttt{SIMULATED}"
    assert macros["BTPlaqSimJobId"] == r"\texttt{null}"
    assert macros["BTPlaqFourLinkGroupConfigs"] == "331,776"
    assert macros["BTPlaqMaxGaugeDiscrepancy"] == r"2.27 \times 10^{-15}"
    assert macros["BTPlaqUbAtaCz"] == "16"
    assert macros["BTPlaqUbTfCoreCz"] == "19"
    assert macros["BTPlaqFezCzCenterStruct"] == "40"
    assert macros["BTPlaqFezCzCenterGeneric"] == "2,727"
    assert macros["BTPlaqFezCzReductionRatio"] == r"68.18\times"
    assert macros["BTPlaqStrangAtaCz"] == "209"
    assert macros["BTPlaqStrangTfCz"] == "350"
    assert macros["BTPlaqFourLinkInplaceAtaCz"] == "21,804"
    assert macros["BTPlaqFourLinkAncillaHhCz"] == "62,407"
    assert macros["BTPlaqCompTwentyQAtaRatio"] == r"104.33\times"
    assert macros["BTPlaqCompTwentyFiveQHhRatio"] == r"178.31\times"
    assert macros["BTPlaqStrangSlopeGOne"] == "2.000008"

    assert fez_results["execution_tier"] == "HARDWARE_ATTESTED"
    assert macros["BTPlaqFezJobId"] == r"\texttt{dc38nreq4uuc73a6g0k0}"
    assert macros["BTPlaqFezWindowOneJobId"] == r"\texttt{dc35e35s2omc738cvcug}"
    assert macros["BTPlaqFezTotalPubs"] == "240"
    assert macros["BTPlaqFezShotsPerPub"] == "256"
    assert macros["BTPlaqFezTotalShots"] == "61,440"
    assert macros["BTPlaqFezNumPlacements"] == "10"
    assert macros["BTPlaqFezTotalTransmons"] == "50"
    assert macros["BTPlaqFezWriteAheadCommit"] == r"\texttt{82cfd46a}"
    assert macros["BTPlaqFezQpyShaFull"] == (
        r"\texttt{349a7ab40a1a55db6ca85dbfd27c24f75aa3e4c67c20accad82a7a41b2c87b9a}"
    )
    assert macros["BTPlaqFezRawCountsShaFull"] == (
        r"\texttt{4667adc71b3af9b7dd3c9dff1fde9d4d6acaf768597992bfbc39bbde2c0abeb3}"
    )

    assert macros["BTPlaqPidZeroCenterTZero"] == "0.876302"
    assert macros["BTPlaqPidZeroCenterPiEight"] == "0.411458"
    assert macros["BTPlaqPidZeroCenterPiFourPlus"] == "0.057292"
    assert macros["BTPlaqPidZeroCenterPiTwo"] == "0.841146"
    assert macros["BTPlaqPidZeroPearsonCenter"] == "0.997576"
    assert macros["BTPlaqPidZeroCenterPiFourMinus"] == "0.850260"
    assert macros["BTPlaqPidZeroCenterGenericMinus"] == "0.041667"
    assert macros["BTPlaqPidZeroCenterStructAdv"] == "0.808594"
    assert macros["BTPlaqPidZeroCenterStructFidelityRatio"] == r"20.41\times"

    assert macros["BTPlaqPidZeroHurwitzTZero"] == "0.829427"
    assert macros["BTPlaqPidZeroHurwitzPiEight"] == "0.669271"
    assert macros["BTPlaqPidZeroHurwitzPiFour"] == "0.404948"
    assert macros["BTPlaqPidZeroHurwitzThreePiEight"] == "0.130208"
    assert macros["BTPlaqPidZeroHurwitzPiTwoPlus"] == "0.050781"
    assert macros["BTPlaqPidZeroPearsonHurwitz"] == "0.997940"
    assert macros["BTPlaqPidZeroHurwitzPiTwoMinus"] == "0.796875"
    assert macros["BTPlaqPidZeroImagNullSurv"] == "0.861979"
    assert macros["BTPlaqPidZeroUnusedNullSurv"] == "0.785156"

    assert macros["BTPlaqPidOnePearsonCenter"] == "0.999283"
    assert macros["BTPlaqPidOnePearsonHurwitz"] == "0.977581"
    assert macros["BTPlaqEchoElapsedHours"] == "3.763"
    assert macros["BTPlaqDriftedHubQubit"] == "7"
    assert macros["BTPlaqQSevenExcitation"] == "0.311198"
    assert macros["BTPlaqQSevenExcitationPct"] == r"31.12\%"
    assert macros["BTPlaqWOnePidOneIdFid"] == "0.962240"
    assert macros["BTPlaqWTwoPidOneIdFid"] == "0.677083"
    assert macros["BTPlaqWOnePidOneEcho"] == "0.815104"
    assert macros["BTPlaqWTwoPidOneEcho"] == "0.562500"
    assert macros["BTPlaqWTwoSpearmanCal"] == "0.454545"
    assert macros["BTPlaqWTwoSpearmanEcho"] == "0.948333"
    assert macros["BTPlaqWTwoDeltaSpearman"] == "0.493787"
    assert macros["BTPlaqWTwoRegretCal"] == "0.361002"
    assert macros["BTPlaqWTwoRegretEcho"] == "0.000000"
    assert macros["BTPlaqTwoWinMeanSpearmanEcho"] == "0.834772"
    assert macros["BTPlaqTwoWinMeanSpearmanCal"] == "0.618182"
    assert macros["BTPlaqTwoWinMeanDeltaSpearman"] == "0.216591"


def test_bt2t003_figures_and_standalone_pdf14_preprint() -> None:
    for stem in FIGURE_STEMS:
        pdf_p = FIGURES_DIR / f"{stem}.pdf"
        png_p = FIGURES_DIR / f"{stem}.png"
        assert pdf_p.is_file() and pdf_p.stat().st_size > 2_000
        assert png_p.is_file() and png_p.stat().st_size > 5_000

        fig_pages, _ = _audit_pdf14_bytes(pdf_p.read_bytes(), f"{stem}.pdf")
        assert fig_pages == 1
        assert b"/CreationDate (2023-11-14T22:13:20Z)" in pdf_p.read_bytes()

        png_bytes = png_p.read_bytes()
        assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        assert (
            b"IHDR" in png_bytes and b"IDAT" in png_bytes and png_bytes.endswith(b"IEND\xaeB`\x82")
        )
        assert b"2023-11-14T22:13:20Z" in png_bytes
        width, height = struct.unpack(">II", png_bytes[16:24])
        assert width == 1152 and height == 648

    pdf_bytes = PREPRINT_PATH.read_bytes()
    assert len(pdf_bytes) > 40_000
    assert b"/CreationDate (2023-11-14T22:13:20Z)" in pdf_bytes
    assert b"/MediaBox [0 0 612.00 792.00]" in pdf_bytes

    page_count, streams = _audit_pdf14_bytes(pdf_bytes, PREPRINT_PATH.name)
    assert page_count >= 5, f"Expected >= 5 pages in {PREPRINT_PATH.name}, got {page_count}"
    pdf_text = _extract_pdf_text(streams)

    for page_idx in range(1, page_count + 1):
        assert f"Page {page_idx} of {page_count}" in pdf_text

    required_pdf_anchors = (
        "Devin Phillip Davis",
        "Agile Defense Systems",
        "OSIRIS-BT2T-003",
        "OSIRIS-BT2T-003-FEZ",
        "SIMULATED",
        "EXACT_STATEVECTOR",
        "HARDWARE_ATTESTED",
        "provider_job_id: null",
        "dc38nreq4uuc73a6g0k0",
        "dc35e35s2omc738cvcug",
        "82cfd46a",
        "349a7ab40a1a55db",
        "4667adc71b3af9b7",
        "bt2t-enc-1",
        "331,776",
        "2.27e-15",
        "16 CZ",
        "19 CZ",
        "40 CZ",
        "2,727",
        "68.18x",
        "209 CZ",
        "350 CZ",
        "104.33x",
        "178.31x",
        "2.000008",
        "61,440",
        "0.876302",
        "0.411458",
        "0.057292",
        "0.841146",
        "0.997576",
        "0.850260",
        "0.041667",
        "0.808594",
        "20.41x",
        "0.829427",
        "0.669271",
        "0.404948",
        "0.130208",
        "0.050781",
        "0.997940",
        "0.796875",
        "0.861979",
        "0.785156",
        "0.999283",
        "0.977581",
        "3.763",
        "31.12%",
        "0.962240",
        "0.677083",
        "0.815104",
        "0.562500",
        "0.454545",
        "0.948333",
        "0.493787",
        "0.361002",
        "0.000000",
        "0.834772",
        "0.618182",
        "0.216591",
        "2023-11-14T22:13:20Z",
    )
    for anchor in required_pdf_anchors:
        assert anchor in pdf_text, f"Missing required anchor {anchor!r} in {PREPRINT_PATH.name}"


def test_bt2t003_sys024_claims_firewall_and_secret_redaction() -> None:
    raw_tex = TEX_PATH.read_text(encoding="utf-8")
    gen_tex = MACROS_PATH.read_text(encoding="utf-8")
    readme_txt = README_PATH.read_text(encoding="utf-8")
    tables_txt = "\n".join((TABLES_DIR / fn).read_text(encoding="utf-8") for fn in TABLE_FILES)
    macros = _parse_macros(gen_tex)
    expanded_tex = _expand_macros(raw_tex + "\n" + tables_txt, macros)
    _, streams = _audit_pdf14_bytes(PREPRINT_PATH.read_bytes(), PREPRINT_PATH.name)
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
        ("main_tex", raw_tex),
        ("generated_macros_tex", gen_tex),
        ("tables_tex", tables_txt),
        ("expanded_tex", expanded_tex),
        ("readme_md", readme_txt),
        ("preprint_pdf_text", pdf_text),
    ):
        flagged = claims.flagged_in_reply(surface_text, [])
        assert not flagged, f"SYS-024 flagged in {surface_name}: {[c.id for c in flagged]}"
        matched = claims.match(surface_text)
        bad_matches = [c for c in matched if c.verdict in quarantined_verdicts]
        assert not bad_matches, (
            f"Quarantined claim in {surface_name}: {[(c.id, c.verdict) for c in bad_matches]}"
        )
        for p_id, p_re in plan_gates._QUARANTINED_PLAN_PATTERNS:
            assert not p_re.search(surface_text), f"Banned constant {p_id} in {surface_name}"
        for q_id, q_re in claim_provenance._QUARANTINED_CONSTANTS:
            assert not q_re.search(surface_text), f"Quarantined constant {q_id} in {surface_name}"

    for p in _tracked_paper_files():
        if p.suffix in {".tex", ".md", ".bib", ".txt", ".py"}:
            txt = p.read_text(encoding="utf-8")
            assert "crn:v1:bluemix:" not in txt, f"CRN leaked in {p}"
            assert "ZENODO_TOKEN=" not in txt, f"Zenodo token leaked in {p}"
            if p.suffix != ".py":
                assert '"apikey"' not in txt, f"Secret apikey field leaked in {p}"
