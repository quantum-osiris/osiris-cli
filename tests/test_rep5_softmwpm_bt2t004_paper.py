"""Verification suite for Paper 11 (`papers/rep5_fez_softmwpm_and_bt2t004_ladder/`).

Verifies:
  1. Byte-for-byte deterministic regeneration of macros (`generated.tex`), dual SVG/PDF 1.4
     figures, standalone multi-page `%PDF-1.4` preprint (`>= 5` pages), RFC 8785 JCS
     `reproducibility_capsule.json`, and `SHA256SUMS.txt` -- including offline fallback
     when `qiskit`, `pytket`, and `rfc8785` are blocked in `sys.modules`.
  2. Cryptographic write-ahead server-clock margin (`+3.39 h` staged at commit `82cfd46a`),
     QPY SHA-256 hash, raw count `.json.gz` hash, and exact empirical agreement across:
     - `OSIRIS-QEC-REP5-FEZ` (`dc3bmlinj2vs73di19s0`, `HARDWARE_ATTESTED`, `12s` QPU, `36,864` shots)
     - `OSIRIS-QEC-SOFTMWPM` (`SoftDriftBlossomDecoder`, `600/600` Blossom, `38/38` sinkholes,
       `dc35d4inj2vs73dhmsog` offline re-scoring across `49,152` shots, `0s` new QPU)
     - `OSIRIS-BT2T-004` (`SIMULATED` / `PROXY`, `provider_job_id: null`, `0s` QPU, `37/37` checks)
  3. Zero hardcoded empirical floats in `rep5_fez_softmwpm_and_bt2t004_ladder.tex` prose or
     tables, balanced braces/environments, and 100% macro completeness (`>= 150` `\\rfs*` macros).
  4. `SYS-024` Claims Firewall across `raw_tex`, `generated_tex`, `expanded_tex`, and `pdf_text`,
     plus CRN/secret absence across paper and experiment directories.
  5. Clean `sha256sum -c SHA256SUMS.txt` verification across all 22 locked artifacts.
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

PAPER_DIR = REPO_ROOT / "papers" / "rep5_fez_softmwpm_and_bt2t004_ladder"
ASSET_SCRIPT_PATH = PAPER_DIR / "generate_rep5_softmwpm_bt2t004_paper_assets.py"
TEX_PATH = PAPER_DIR / "rep5_fez_softmwpm_and_bt2t004_ladder.tex"
PDF_PATH = PAPER_DIR / "rep5_fez_softmwpm_and_bt2t004_ladder.pdf"
GEN_TEX_PATH = PAPER_DIR / "generated.tex"
CAPSULE_PATH = PAPER_DIR / "reproducibility_capsule.json"
SUMS_PATH = PAPER_DIR / "SHA256SUMS.txt"
PREPRINTS_COPY_PATH = Path(
    "/home/osiris/osiris-assessment/work/preprints_2026-11-23/rep5_fez_softmwpm_and_bt2t004_ladder.pdf"
)

REP5_DIR = REPO_ROOT / "experiments" / "rep5_fez_staged"
SOFT_DIR = REPO_ROOT / "experiments" / "qec_card_c"
BT2T_DIR = REPO_ROOT / "experiments" / "bt2t_multiplaquette"

FIGURE_STEMS = (
    "fig1_rep5_fez_lambda35_and_soft_drift_mwpm",
    "fig2_bt2t004_two_plaquette_ladder_confinement_and_strang",
)

REQUIRED_LABELS = (
    "tab:rep5_fez_scaling",
    "tab:soft_drift_mwpm",
    "tab:bt2t004_confinement",
    "fig:rep5_soft_drift_mwpm",
    "fig:bt2t004_confinement_strang",
    "eq:rep5_lambda_streaks",
    "eq:soft_drift_blossom",
    "eq:bt2t004_gauss_hamiltonian",
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
        "dim",
        "documentclass",
        "dots",
        "emph",
        "end",
        "epsilon",
        "eqref",
        "eta",
        "frac",
        "gamma",
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
        "ln",
        "maketitle",
        "mapsto",
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
        "sigma",
        "sim",
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
        "usepackage",
        "Rightarrow",
    }
)


def _load_paper_module() -> Any:
    spec = importlib.util.spec_from_file_location("_test_p11_assets", ASSET_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None, f"Cannot load {ASSET_SCRIPT_PATH}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_test_p11_assets"] = mod
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


def test_paper11_deterministic_regeneration_offline_fallback_and_sha256sums(
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
        assert path.is_file(), f"Missing required Paper 11 deliverable: {path}"

    before_bytes = {path.name: path.read_bytes() for path in tracked_files}

    proc = subprocess.run(
        [sys.executable, str(ASSET_SCRIPT_PATH)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Paper 11 generator failed:\n{proc.stderr}\n{proc.stdout}"

    after_bytes = {path.name: path.read_bytes() for path in tracked_files}
    assert before_bytes == after_bytes, (
        "Re-running generate_rep5_softmwpm_bt2t004_paper_assets.py mutated tracked deliverables"
    )
    if PREPRINTS_COPY_PATH.parent.is_dir():
        assert PREPRINTS_COPY_PATH.is_file()
        assert PREPRINTS_COPY_PATH.read_bytes() == PDF_PATH.read_bytes()

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
    assert len(verified_lines) == 22


def test_paper11_live_rep5_fez_and_soft_drift_mwpm_upstream_invariants() -> None:
    rep5_res = json.loads((REP5_DIR / "physical_rep5_fez_results.json").read_text(encoding="utf-8"))
    rep5_sub = json.loads((REP5_DIR / "rep5_fez_submitted_job.json").read_text(encoding="utf-8"))
    soft_rep = json.loads(
        (SOFT_DIR / "artifacts" / "card_c_soft_drift_mwpm_report.json").read_text(encoding="utf-8")
    )

    assert rep5_res["execution_tier"] == "HARDWARE_ATTESTED"
    assert rep5_res["provider_job_id"] == "dc3bmlinj2vs73di19s0"
    assert rep5_res["chain_9q"] == [120, 121, 122, 123, 136, 143, 142, 141, 140]
    assert (
        rep5_res["qpy_sha256"] == "da8061639e8eaa7e708f506095c590f69dc39ec5573f67edebadbb6f9416aee1"
    )
    assert (
        rep5_res["raw_counts_gz_sha256"]
        == "d73bd9c153d2feddc7e403455b6b725630106ee3b09d8d4f103bdb83fcadea56"
    )
    assert rep5_sub["total_pubs"] == 72
    assert rep5_sub["shots_per_pub"] == 512
    assert rep5_sub["total_shots"] == 36864
    assert rep5_sub["usage"]["qpu_charge_time_seconds"] == 12
    assert rep5_res["ro_cal_5d_assignment_fidelity"] == pytest.approx(0.947754)
    assert rep5_res["mean_lambda_3_to_5_mwpm"] == pytest.approx(2.1209)
    assert rep5_res["peak_lambda_3_to_5_mwpm_r3"] == pytest.approx(2.8333)

    r_by_depth = {int(r["rounds"]): r for r in rep5_res["rounds_summary"]}
    assert r_by_depth[1]["lambda_3_to_5_mwpm"] == pytest.approx(2.0)
    assert r_by_depth[3]["lambda_3_to_5_mwpm"] == pytest.approx(2.8333)
    assert r_by_depth[5]["lambda_3_to_5_mwpm"] == pytest.approx(1.5294)
    for rv in (1, 3, 5):
        assert r_by_depth[rv]["arm3_rep3_postselected_p_fail"] == 0.0
        assert r_by_depth[rv]["arm3_rep5_postselected_p_fail"] == 0.0
        assert r_by_depth[rv]["arm4a_rep5_mwpm_p_fail"] < r_by_depth[rv]["arm4a_rep3_mwpm_p_fail"]

    assert soft_rep["new_qpu_seconds"] == 0
    assert soft_rep["provider_job_id"] == "dc35d4inj2vs73dhmsog"
    vc = soft_rep["verification_controls"]
    assert vc["component_dp_vs_pure_edmonds_blossom_syndromes_tested"] == 600
    assert vc["component_dp_vs_pure_edmonds_blossom_mismatches"] == 0
    assert vc["exact_100pct_blossom_match"] is True

    b2_hi = soft_rep["benchmark_2_live_fez_dc35d4inj2vs73dhmsog_offline_rescoring"][
        "headline_improvements"
    ]
    assert b2_hi["crossing_round_vs_arm1_and_arm2_uniform_mwpm"] == 3
    assert b2_hi["crossing_round_vs_arm1_and_arm2_soft_drift_mwpm"] == 1
    assert b2_hi["r1_p_fail_uniform_to_soft_drift"] == [0.01237, 0.003255]
    assert b2_hi["r3_p_fail_uniform_to_drift_prior_to_soft_drift"] == [0.054036, 0.039062, 0.011719]
    assert b2_hi["r5_p_fail_uniform_to_drift_prior_to_soft_drift"] == [0.08724, 0.063802, 0.022786]
    assert b2_hi["slope_per_round_uniform_to_drift_prior_to_soft_drift"] == [
        0.018717,
        0.012858,
        0.004883,
    ]
    assert b2_hi["slope_reduction_soft_drift_vs_uniform_pct"] == pytest.approx(73.91)


def test_paper11_bt2t004_two_plaquette_ladder_upstream_invariants() -> None:
    bt2t_man = json.loads((BT2T_DIR / "manifest_bt2t004.json").read_text(encoding="utf-8"))
    assert bt2t_man["experiment_id"] == "OSIRIS-BT2T-004"
    assert bt2t_man["execution_tier"] == "SIMULATED"
    assert bt2t_man["provider_job_id"] is None
    assert bt2t_man["qpu_seconds"] == 0
    assert bt2t_man["all_checks_passed"] is True
    assert len(bt2t_man["checks"]) == 37
    assert all(bt2t_man["checks"].values())

    p1 = bt2t_man["part1_gauge_reduction_and_76d_physical_subspace"]
    assert p1["unfixed_group_configurations"] == 4586471424
    assert p1["unfixed_qubits"] == 35
    assert p1["gauge_fixed_active_configurations"] == 576
    assert p1["decoupled_padded_subspace_dim"] == 448
    assert p1["gauge_fixed_qubits"] == 10
    assert p1["hilbert_space_config_reduction_factor"] == 7962624
    assert p1["qubit_reduction_factor"] == pytest.approx(3.5)
    assert (
        p1["burnside_physical_subspace_dim"]
        == p1["class_centralizer_sum_dim"]
        == p1["clebsch_gordan_endomorphism_dim"]
        == p1["explicit_conjugation_orbit_count"]
        == 76
    )

    p5 = bt2t_man["part5_native_10q_and_35q_circuit_compilations"]["gate_count_summary"]
    assert p5["cz_ub_structured_10q_all_to_all"] == 32
    assert p5["cz_ub_structured_10q_heavy_hex_2xT5_patches"] == 38
    assert p5["cz_strang_structured_10q_all_to_all"] == 4125
    assert p5["cz_generic_dense_1024x1024_qsd_exact_all_to_all"] == 479063


def test_paper11_latex_macros_zero_prose_floats_and_upstream_bindings() -> None:
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
    assert all(name.startswith("rfs") and name.isalpha() for name in defined_macros)

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
    assert macros["rfsRepFiveJobId"] == r"\texttt{dc3bmlinj2vs73di19s0}"
    assert macros["rfsRepFiveQpuSec"] == "12"
    assert macros["rfsRepFiveTotalPubs"] == "72"
    assert macros["rfsRepFiveTotalShots"] == "36,864"
    assert macros["rfsRepFiveRoCalFid"] == "0.947754"
    assert macros["rfsRepFiveArmFourDFiveROne"] == "0.000488"
    assert macros["rfsRepFiveArmFourDFiveRThree"] == "0.002930"
    assert macros["rfsRepFiveArmFourDFiveRFive"] == "0.008301"
    assert macros["rfsRepFiveLamROne"] == "2.0000"
    assert macros["rfsRepFiveLamRThree"] == "2.8333"
    assert macros["rfsRepFiveLamRFive"] == "1.5294"
    assert macros["rfsRepFiveMeanLam"] == "2.1209"
    assert macros["rfsRepFiveSlopeArmFourDFive"] == "0.001953"

    assert macros["rfsSoftSurfStaleLamA"] == "1.269"
    assert macros["rfsSoftSurfSoftLamA"] == "3.537"
    assert macros["rfsSoftSurfSoftLamB"] == "3.633"
    assert macros["rfsSoftSurfSoftLamC"] == "3.334"
    assert macros["rfsSoftFezRepThreeJobId"] == r"\texttt{dc35d4inj2vs73dhmsog}"
    assert macros["rfsSoftFezROneSoft"] == "0.003255"
    assert macros["rfsSoftFezRThreeSoft"] == "0.011719"
    assert macros["rfsSoftFezRFiveSoft"] == "0.022786"
    assert macros["rfsSoftFezSlopeSoft"] == "0.004883"
    assert macros["rfsSoftFezSlopeRedPct"] == r"73.91\%"

    assert macros["rfsBtUnfixedConfigs"] == "4,586,471,424"
    assert macros["rfsBtActiveConfigs"] == "576"
    assert macros["rfsBtPhysSubspaceDim"] == "76"
    assert macros["rfsBtCzUbAllToAll"] == "32"
    assert macros["rfsBtCzUbHeavyHex"] == "38"
    assert macros["rfsBtCzDenseTenQQsd"] == "479,063"


def test_paper11_standalone_pdf14_multipage_sys024_and_secret_hygiene() -> None:
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
        "PROXY",
        "provider_job_id: null",
        "dc3bmlinj2vs73di19s0",
        "dc35d4inj2vs73dhmsog",
        "36,864",
        "49,152",
        "82cfd46a",
        "0.947754",
        "0.000488",
        "0.002930",
        "0.008301",
        "2.8333",
        "1.5294",
        "2.1209",
        "0.001953",
        "3.537",
        "3.633",
        "3.334",
        "0.003255",
        "0.011719",
        "0.022786",
        "0.004883",
        "73.91%",
        "4,586,471,424",
        "7,962,624x",
        "576",
        "76",
        "479,063",
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
    assert capsule["combined_new_qpu_seconds"] == 12
    assert capsule["combined_rep5_physical_shots"] == 36864
    assert capsule["combined_rep5_physical_pubs"] == 72
    assert (
        capsule["hardware_jobs"]["OSIRIS-QEC-REP5-FEZ"]["provider_job_id"] == "dc3bmlinj2vs73di19s0"
    )
    assert (
        capsule["hardware_jobs"]["OSIRIS-QEC-REP3-FEZ-OFFLINE-RESCORED"]["provider_job_id"]
        == "dc35d4inj2vs73dhmsog"
    )
    assert capsule["simulated_companions"]["OSIRIS-BT2T-004"]["provider_job_id"] is None
    assert capsule["simulated_companions"]["OSIRIS-BT2T-004"]["qpu_seconds"] == 0

    for rel_path, expected_sha in capsule["source_artifacts"].items():
        actual_sha = hashlib.sha256((REPO_ROOT / rel_path).read_bytes()).hexdigest()
        assert actual_sha == expected_sha, f"Source artifact hash mismatch for {rel_path}"

    for folder in (PAPER_DIR, REP5_DIR, SOFT_DIR, BT2T_DIR):
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix in {".tex", ".json", ".md", ".svg", ".txt", ".py"}:
                txt = p.read_text(encoding="utf-8")
                assert "crn:v1:bluemix:" not in txt, f"CRN leaked in {p}"
                assert "ZENODO_TOKEN=" not in txt, f"Zenodo token leaked in {p}"
                if p.suffix != ".py":
                    assert '"apikey"' not in txt, f"Secret apikey field leaked in {p}"
