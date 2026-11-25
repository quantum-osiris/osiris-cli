"""Verification suite for Paper 9 (`papers/rep3_echo_fez_and_multidistance_qec/`).

Verifies:
  1. Byte-for-byte deterministic regeneration of macros (`generated.tex`), dual SVG/PDF 1.4
     figures, standalone multi-page `%PDF-1.4` preprint (`>= 5` pages), RFC 8785 JCS
     `reproducibility_capsule.json`, and `SHA256SUMS.txt` -- including offline fallback
     when `qiskit`, `pytket`, and `rfc8785` are blocked in `sys.modules`.
  2. Cryptographic write-ahead server-clock margins (`> 9.7 h`), QPY SHA-256 hashes,
     raw count `.json.gz` hashes, and exact empirical agreement across:
     - `OSIRIS-QEC-REP3-FEZ` (`dc35d4inj2vs73dhmsog`, `HARDWARE_ATTESTED`, `16s` QPU, `49,152` shots)
     - `OSIRIS-ECHO-FEZ` (`dc35e35s2omc738cvcug`, `HARDWARE_ATTESTED`, `19s` QPU, `61,440` shots)
     - `OSIRIS-QEC-MULTIDISTANCE` (`SIMULATED`, `provider_job_id: null`, `0s` QPU, `d in {3, 5, 7}`)
  3. Zero hardcoded empirical floats in `rep3_echo_fez_and_multidistance_qec.tex` prose or
     tables, balanced braces/environments, and 100% macro completeness (`>= 150` `\rem*` macros).
  4. `SYS-024` Claims Firewall across `raw_tex`, `generated_tex`, `expanded_tex`, and `pdf_text`,
     plus CRN/secret absence across paper and experiment directories.
  5. Clean `sha256sum -c SHA256SUMS.txt` verification across all 27 locked artifacts.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osiris_cli import claim_provenance, claims, exec_kernel, plan_gates  # noqa: E402
from tests.test_standalone_paper_pdfs import _audit_pdf14_bytes, _extract_pdf_text  # noqa: E402

PAPER_DIR = REPO_ROOT / "papers" / "rep3_echo_fez_and_multidistance_qec"
ASSET_SCRIPT_PATH = PAPER_DIR / "generate_rep3_echo_multidist_paper_assets.py"
TEX_PATH = PAPER_DIR / "rep3_echo_fez_and_multidistance_qec.tex"
PDF_PATH = PAPER_DIR / "rep3_echo_fez_and_multidistance_qec.pdf"
GEN_TEX_PATH = PAPER_DIR / "generated.tex"
CAPSULE_PATH = PAPER_DIR / "reproducibility_capsule.json"
SUMS_PATH = PAPER_DIR / "SHA256SUMS.txt"

REP3_DIR = REPO_ROOT / "experiments" / "rep3_fez_staged"
ECHO_DIR = REPO_ROOT / "experiments" / "echo_fez_staged"
MULTIDIST_DIR = REPO_ROOT / "experiments" / "qec_card_c"

FIGURE_STEMS = (
    "fig1_rep3_fez_live_crossing_and_echo_10t5",
    "fig2_multidistance_d3_d5_d7_threshold_and_lambda",
)

REQUIRED_LABELS = (
    "tab:rep3_fez_crossing",
    "tab:echo_fez_placements",
    "tab:multidistance_scaling",
    "fig:rep3_echo_fez_live",
    "fig:multidistance_threshold_lambda",
    "eq:rep3_feedforward_streaks",
    "eq:echo_spam_norm",
    "eq:multidistance_lambda",
)

STANDARD_LATEX_COMMANDS = frozenset(
    {
        "Bigl",
        "Bigr",
        "Delta",
        "Lambda",
        "Pr",
        "Xi",
        "affiliation",
        "align",
        "author",
        "bar",
        "begin",
        "bibitem",
        "bigl",
        "bigr",
        "caption",
        "centering",
        "cite",
        "dagger",
        "date",
        "documentclass",
        "emph",
        "end",
        "epsilon",
        "eqref",
        "frac",
        "ge",
        "hat",
        "hline",
        "in",
        "includegraphics",
        "input",
        "item",
        "label",
        "land",
        "langle",
        "le",
        "maketitle",
        "mathbb",
        "mathcal",
        "mathrm",
        "max",
        "min",
        "mu",
        "neq",
        "nonumber",
        "oplus",
        "otimes",
        "pm",
        "quad",
        "qquad",
        "rangle",
        "ref",
        "rho",
        "ruledtabular",
        "section",
        "sqrt",
        "subsection",
        "text",
        "textbf",
        "texttt",
        "textwidth",
        "tfrac",
        "times",
        "title",
        "to",
        "usepackage",
        "Rightarrow",
    }
)


def _load_paper_module() -> Any:
    spec = importlib.util.spec_from_file_location("_test_p9_assets", ASSET_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None, f"Cannot load {ASSET_SCRIPT_PATH}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_test_p9_assets"] = mod
    spec.loader.exec_module(mod)
    return mod


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


def test_paper9_deterministic_regeneration_offline_fallback_and_sha256sums(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracked_files = [
        GEN_TEX_PATH,
        CAPSULE_PATH,
        SUMS_PATH,
        TEX_PATH,
        PDF_PATH,
        ASSET_SCRIPT_PATH,
        *(PAPER_DIR / f"{stem}.svg" for stem in FIGURE_STEMS),
        *(PAPER_DIR / f"{stem}.pdf" for stem in FIGURE_STEMS),
    ]
    for path in tracked_files:
        assert path.is_file(), f"Missing required Paper 9 deliverable: {path}"

    before_bytes = {path.name: path.read_bytes() for path in tracked_files}

    proc = subprocess.run(
        [sys.executable, str(ASSET_SCRIPT_PATH)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Paper 9 generator failed:\n{proc.stderr}\n{proc.stdout}"

    after_bytes = {path.name: path.read_bytes() for path in tracked_files}
    assert before_bytes == after_bytes, (
        "Re-running generate_rep3_echo_multidist_paper_assets.py mutated tracked deliverables"
    )

    # Verify offline execution when qiskit, pytket, and rfc8785 are all blocked in sys.modules
    paper_mod = _load_paper_module()
    capsule = json.loads(CAPSULE_PATH.read_text(encoding="utf-8"))
    capsule_body = {k: v for k, v in capsule.items() if k != "rfc8785_canonical_sha256"}
    sha_default = hashlib.sha256(paper_mod.jcs_bytes(capsule_body)).hexdigest()
    sha_kernel = hashlib.sha256(exec_kernel.canon(capsule_body).encode("utf-8")).hexdigest()

    for blocked_pkg in ("qiskit", "pytket", "rfc8785"):
        monkeypatch.setitem(sys.modules, blocked_pkg, None)

    sha_fallback = hashlib.sha256(paper_mod.jcs_bytes(capsule_body)).hexdigest()
    assert capsule["rfc8785_canonical_sha256"] == sha_default == sha_kernel == sha_fallback

    regen_capsule = paper_mod.generate_all()
    assert regen_capsule["rfc8785_canonical_sha256"] == capsule["rfc8785_canonical_sha256"]
    assert {path.name: path.read_bytes() for path in tracked_files} == before_bytes

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
    assert len(verified_lines) == 27


def test_paper9_live_rep3_and_echo_fez_upstream_invariants() -> None:
    rep3_res = json.loads((REP3_DIR / "physical_rep3_fez_results.json").read_text(encoding="utf-8"))
    echo_res = json.loads((ECHO_DIR / "physical_echo_fez_results.json").read_text(encoding="utf-8"))

    assert rep3_res["execution_tier"] == "HARDWARE_ATTESTED"
    assert rep3_res["provider_job_id"] == "dc35d4inj2vs73dhmsog"
    assert rep3_res["total_pubs"] == 96
    assert rep3_res["total_shots"] == 49152
    assert rep3_res["provider_metadata"]["usage"]["qpu_charge_time_seconds"] == 16
    for r_key in ("R3_arm2_minus_arm4a", "R3_arm2_minus_arm4b", "R5_arm2_minus_arm4a", "R5_arm2_minus_arm4b"):
        assert rep3_res["paired_block_tests"][r_key]["block_wins"] == 3
        assert rep3_res["paired_block_tests"][r_key]["delta_mean"] > 0.0

    assert echo_res["execution_tier"] == "HARDWARE_ATTESTED"
    assert echo_res["provider_job_id"] == "dc35e35s2omc738cvcug"
    assert echo_res["total_pubs"] == 240
    assert echo_res["total_shots"] == 61440
    assert echo_res["provider_metadata"]["usage"]["qpu_charge_time_seconds"] == 19
    assert len(echo_res["placement_results"]) == 10
    all_qubits = [q for p in echo_res["placement_results"] for q in p["physical_qubits"]]
    assert len(all_qubits) == 50 and len(set(all_qubits)) == 50
    assert echo_res["verdicts"]["top1_placement_selected_pid"] == 1
    assert echo_res["verdicts"]["top1_heldout_composite_regret"] == 0.0


def test_paper9_multidistance_d3_d5_d7_upstream_invariants() -> None:
    multi_rep = json.loads(
        (MULTIDIST_DIR / "artifacts" / "card_c_multidistance_threshold_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert multi_rep["execution_tier"] == "SIMULATED"
    assert multi_rep["provider_job_id"] is None
    assert multi_rep["qpu_seconds"] == 0
    assert multi_rep["hardware_execution_authorized"] is False
    assert multi_rep["model_documentation"]["distances_evaluated"] == [3, 5, 7]
    vc = multi_rep["verification_controls"]
    assert vc["fault_distance_and_hook_safety_audit"]["all_distance_and_t_fault_audits_passed"] is True
    assert vc["polynomial_mwpm_vs_bitmask_dp_equivalence"]["all_exact_100pct_match"] is True
    assert vc["polynomial_mwpm_vs_bitmask_dp_equivalence"]["total_multi_defect_syndromes_tested"] == 3600


def test_paper9_latex_macros_zero_prose_floats_and_upstream_bindings() -> None:
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
    assert len(defined_macros) >= 150, f"Expected >= 150 macros, got {len(defined_macros)}"
    assert all(name.startswith("rem") and name.isalpha() for name in defined_macros)

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

    # Strictly ZERO floating-point literals anywhere in manuscript prose or tables
    body = cleaned_tex.split(r"\begin{thebibliography}")[0]
    raw_floats = re.findall(r"\b\d+\.\d+\b", body)
    assert not raw_floats, (
        f"Hardcoded floating-point literals found in {TEX_PATH.name}: {raw_floats}"
    )

    # Verify key empirical macros match the three upstream JSON reports
    assert macros["remRepJobId"] == r"\texttt{dc35d4inj2vs73dhmsog}"
    assert macros["remEchoJobId"] == r"\texttt{dc35e35s2omc738cvcug}"
    assert macros["remTotalPhysicalShots"] == "110,592"
    assert macros["remTotalQpuSec"] == "35"
    assert float(macros["remRepWriteAheadHours"]) > 9.7
    assert float(macros["remEchoWriteAheadHours"]) > 9.7
    assert macros["remRepQpyShaPrefix"].startswith(r"\texttt{f240d445")
    assert macros["remEchoQpyShaPrefix"].startswith(r"\texttt{538af9be")
    assert macros["remRepRawGzShaPrefix"].startswith(r"\texttt{2af0e736")
    assert macros["remEchoRawGzShaPrefix"].startswith(r"\texttt{db19dfe6")

    assert macros["remRepArmOneROne"] == "0.1055"
    assert macros["remRepArmOneRThree"] == "0.2253"
    assert macros["remRepArmOneRFive"] == "0.2770"
    assert macros["remRepArmTwoROne"] == "0.0098"
    assert macros["remRepArmTwoRThree"] == "0.0713"
    assert macros["remRepArmTwoRFive"] == "0.1230"
    assert macros["remRepArmFourAROne"] == "0.0124"
    assert macros["remRepArmFourARThree"] == "0.0540"
    assert macros["remRepArmFourARFive"] == "0.0872"
    assert macros["remRepArmFourBROne"] == "0.0260"
    assert macros["remRepArmFourBRThree"] == "0.0628"
    assert macros["remRepArmFourBRFive"] == "0.0980"
    assert macros["remRepRFiveTFourB"] == "9.0131"
    assert macros["remRepRFivePFourB"] == "0.006043"
    assert macros["remRepRoCalFid"] == "0.983317"

    assert macros["remEchoTopOneRegret"] == "0.0000"
    assert macros["remEchoPidOneComposite"] == "0.6594"
    assert macros["remEchoMirrorRho"] == "0.9030"
    assert macros["remEchoMirrorPEcho"] == "0.000344"
    assert macros["remEchoCompRho"] == "0.7212"
    assert macros["remEchoCompCalRho"] == "0.7818"
    assert macros["remEchoCompDeltaRho"] == "-0.0606"
    assert macros["remEchoNormCompRho"] == "0.9030"
    assert macros["remEchoNormBtRho"] == "0.9515"

    assert macros["remMultiSurfPthPct"] == r"1.124\%"
    assert macros["remMultiRepPthPct"] == r"13.81\%"
    assert macros["remMultiDrfSurfDiffLamThreeFive"] == "1.023"
    assert macros["remMultiHtwoSurfLamThreeFive"] == "6.96"
    assert macros["remMultiHtwoSurfLamFiveSeven"] == "13.78"


def test_paper9_standalone_pdf14_multipage_sys024_and_secret_hygiene() -> None:
    pdf_bytes = PDF_PATH.read_bytes()
    page_count, streams = _audit_pdf14_bytes(pdf_bytes, PDF_PATH.name)
    assert page_count >= 5, f"Expected >= 5 pages in {PDF_PATH.name}, got {page_count}"
    assert b"/MediaBox [0 0 612.00 792.00]" in pdf_bytes
    pdf_text = _extract_pdf_text(streams)

    for page_idx in range(1, page_count + 1):
        assert f"Page {page_idx} of {page_count}" in pdf_text

    required_pdf_anchors = (
        "HARDWARE_ATTESTED",
        "SIMULATED",
        "provider_job_id: null",
        "dc35d4inj2vs73dhmsog",
        "dc35e35s2omc738cvcug",
        "110,592",
        "49,152",
        "61,440",
        "0.0872",
        "0.0980",
        "9.0131",
        "0.006043",
        "0.983317",
        "0.6594",
        "+0.9030",
        "-0.0606",
        "+0.9515",
        "1.124%",
        "13.81%",
        "1.023",
        "6.96",
        "13.78",
        "Devin Phillip Davis",
    )
    for anchor in required_pdf_anchors:
        assert anchor in pdf_text, f"Missing empirical anchor {anchor!r} in {PDF_PATH.name}"

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
        assert not bad_matches, (
            f"Quarantined claim in {surface_name}: {[(c.id, c.verdict) for c in bad_matches]}"
        )
        for p_id, p_re in plan_gates._QUARANTINED_PLAN_PATTERNS:
            assert not p_re.search(surface_text), f"Banned constant {p_id} in {surface_name}"
        for q_id, q_re in claim_provenance._QUARANTINED_CONSTANTS:
            assert not q_re.search(surface_text), f"Quarantined constant {q_id} in {surface_name}"

    capsule = json.loads(CAPSULE_PATH.read_text(encoding="utf-8"))
    assert capsule["schema"] == "osiris.reproducibility_capsule/1"
    assert capsule["combined_physical_shots"] == 110592
    assert capsule["combined_physical_pubs"] == 336
    assert capsule["combined_qpu_seconds"] == 35
    assert capsule["hardware_jobs"]["OSIRIS-QEC-REP3-FEZ"]["provider_job_id"] == "dc35d4inj2vs73dhmsog"
    assert capsule["hardware_jobs"]["OSIRIS-ECHO-FEZ"]["provider_job_id"] == "dc35e35s2omc738cvcug"
    assert capsule["simulated_companion"]["execution_tier"] == "SIMULATED"
    assert capsule["simulated_companion"]["provider_job_id"] is None
    assert capsule["simulated_companion"]["qpu_seconds"] == 0

    for rel_path, expected_sha in capsule["source_artifacts"].items():
        actual_sha = hashlib.sha256((REPO_ROOT / rel_path).read_bytes()).hexdigest()
        assert actual_sha == expected_sha, f"Source artifact hash mismatch for {rel_path}"

    for folder in (PAPER_DIR, REP3_DIR, ECHO_DIR, MULTIDIST_DIR):
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix in {".tex", ".json", ".md", ".svg", ".txt", ".py"}:
                txt = p.read_text(encoding="utf-8")
                assert "crn:v1:bluemix:" not in txt, f"CRN leaked in {p}"
                assert "ZENODO_TOKEN=" not in txt, f"Zenodo token leaked in {p}"
                if p.suffix != ".py":
                    assert '"apikey"' not in txt, f"Secret apikey field leaked in {p}"
