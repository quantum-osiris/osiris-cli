"""Verification suite for Card C 5-Qutrit `|2>` Leakage / `LRU` Gate & Paper 8
(`papers/echo_and_card_c_qec_gates/`).

Verifies:
  1. Exact CPTP completeness (`< 1e-14`) of 3-level & 9-level qutrit Kraus channels and
     exact equivalence (`< 1e-14`) between full `243 x 243` (`3^5 = 243`-state) 5-qutrit
     density-matrix evolution and `125`-dimensional real-Hermitian quantum instruments.
  2. Exact agreement (`0.0` rounded difference; `< 1e-14` raw branch sum) at `p_leak = 0`
     with the 2-level Card C crossing report across `FEZ_PUBLISHED`, `FEZ_DRIFTED_SPAM`,
     and `H2_1_PUBLISHED`.
  3. Physical monotonicity of qutrit `|2>` leakage (`NO_LRU > ANCILLA_LRU_ONLY >
     FULL_DATA_AND_ANCILLA_LRU` at `p_leak_cz = 1.2e-3` and `5e-4`, multi-round ancilla
     streak elimination, data `|2>` suppression, crossing preservation, and `H2-1` Raman
     comparison).
  4. Zero hardcoded empirical floats in `echo_and_card_c_qec_gates.tex`, 100% macro
     completeness, balanced braces/environments, standalone `%PDF-1.4` `>= 4` pages &
     `xref` integrity, `SYS-024` cleanliness, graceful `jcs_bytes` fallback when
     `rfc8785` is absent, `sha256sum -c SHA256SUMS.txt`, and byte-for-byte deterministic
     regeneration.
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

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from osiris_cli import claim_provenance, claims, exec_kernel, plan_gates  # noqa: E402
from tests.test_standalone_paper_pdfs import _audit_pdf14_bytes, _extract_pdf_text  # noqa: E402

CROSSING_PY = REPO_ROOT / "experiments" / "qec_card_c" / "qec_crossing_gate.py"
CROSSING_REPORT_PATH = (
    REPO_ROOT / "experiments" / "qec_card_c" / "artifacts" / "card_c_qec_crossing_report.json"
)
LRU_PY = REPO_ROOT / "experiments" / "qec_card_c" / "qec_leakage_lru.py"
LRU_REPORT_PATH = (
    REPO_ROOT
    / "experiments"
    / "qec_card_c"
    / "artifacts"
    / "card_c_qutrit_leakage_lru_report.json"
)
ECHO_REPORT_PATH = (
    REPO_ROOT / "experiments" / "echo_placement" / "artifacts" / "echo_stopgo_report.json"
)

PAPER_DIR = REPO_ROOT / "papers" / "echo_and_card_c_qec_gates"
ASSET_SCRIPT_PATH = PAPER_DIR / "generate_echo_qec_paper_assets.py"
TEX_PATH = PAPER_DIR / "echo_and_card_c_qec_gates.tex"
PDF_PATH = PAPER_DIR / "echo_and_card_c_qec_gates.pdf"
GEN_TEX_PATH = PAPER_DIR / "generated.tex"
CAPSULE_PATH = PAPER_DIR / "reproducibility_capsule.json"
SUMS_PATH = PAPER_DIR / "SHA256SUMS.txt"

FIGURE_STEMS = (
    "fig1_echo_placement_spearman_and_regret",
    "fig2_card_c_crossing_and_qutrit_lru",
)

REQUIRED_LABELS = (
    "tab:echo_ranking",
    "tab:cardc_crossing",
    "tab:qutrit_lru",
    "fig:echo_spearman_regret",
    "fig:cardc_crossing_lru",
    "eq:echo_probe",
    "eq:eps_l_crossing",
    "eq:qutrit_kraus",
)

STANDARD_LATEX_COMMANDS = frozenset(
    {
        "Bigl",
        "Bigr",
        "Delta",
        "Gamma",
        "mathcal",
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
        "delta",
        "dim",
        "documentclass",
        "emph",
        "end",
        "epsilon",
        "frac",
        "ge",
        "hline",
        "in",
        "includegraphics",
        "input",
        "item",
        "label",
        "lambda",
        "langle",
        "le",
        "leftrightarrow",
        "maketitle",
        "mapsto",
        "mathbb",
        "mathrm",
        "min",
        "mu",
        "neq",
        "nonumber",
        "oplus",
        "otimes",
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
        "theta",
        "times",
        "title",
        "to",
        "usepackage",
    }
)


def _load_module(mod_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None and spec.loader is not None, f"Cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


gate_mod = _load_module("_test_p8_qec_gate", CROSSING_PY)
lru_mod = _load_module("_test_p8_qec_lru", LRU_PY)
paper_mod = _load_module("_test_p8_assets", ASSET_SCRIPT_PATH)


@pytest.fixture(scope="module")
def lru_report() -> dict[str, Any]:
    assert LRU_REPORT_PATH.is_file(), f"Missing qutrit LRU artifact: {LRU_REPORT_PATH}"
    return json.loads(LRU_REPORT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def crossing_report() -> dict[str, Any]:
    assert CROSSING_REPORT_PATH.is_file(), f"Missing Card C artifact: {CROSSING_REPORT_PATH}"
    return json.loads(CROSSING_REPORT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def echo_report() -> dict[str, Any]:
    assert ECHO_REPORT_PATH.is_file(), f"Missing Card B artifact: {ECHO_REPORT_PATH}"
    return json.loads(ECHO_REPORT_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# LaTeX Helper Parsers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 1. Qutrit CPTP Completeness, 243-State Equivalence & Zero-Leakage Baseline
# ---------------------------------------------------------------------------


def test_qutrit_cptp_channels_and_243_vs_125_exact_equivalence() -> None:
    fez = gate_mod.build_published_target_profiles()["FEZ_PUBLISHED"]
    params = lru_mod.default_lru_params_for_profile(fez, p_leak_2q=1.2e-3)

    assert (
        lru_mod.cptp_completeness_error(
            lru_mod.build_qutrit_cz_cx_kraus(fez.delta_theta_rad, fez.p_2q, 1.2e-3)
        )
        < 1e-14
    )
    assert (
        lru_mod.cptp_completeness_error(
            lru_mod.build_qutrit_t1_t2_seep_kraus(
                fez.t_round_us, fez.t1_us, fez.t2_us, params.p_seep_round
            )
        )
        < 1e-14
    )
    assert (
        lru_mod.cptp_completeness_error(
            lru_mod.build_data_dqlr_kraus(
                1.0 - params.lru_infidelity,
                params.p_lru_1q,
                params.dt_lru_us,
                fez.t1_us,
                fez.t2_us,
            )
        )
        < 1e-14
    )
    assert (
        lru_mod.cptp_completeness_error(
            lru_mod.build_qutrit_depol1_kraus(gate_mod._rx(fez.delta_theta_rad) @ gate_mod.H2, fez.p_1q)
        )
        < 1e-14
    )
    assert (
        lru_mod.cptp_completeness_error(lru_mod.build_qutrit_zz_corr_kraus(fez.p_corr)) < 1e-14
    )

    audit = lru_mod._audit_243x243_vs_liouville125(fez)
    assert audit["max_cptp_error"] < 1e-14
    assert audit["full_243x243_vs_liouville125_max_abs_diff"] < 1e-14
    for arch in lru_mod.LRU_ARCHITECTURES:
        assert audit["full_243x243_vs_liouville125_max_abs_diff_by_arch"][arch] < 1e-14


def test_qutrit_zero_leakage_exact_agreement_with_card_c_baseline(
    lru_report: dict[str, Any],
    crossing_report: dict[str, Any],
) -> None:
    fez = gate_mod.build_published_target_profiles()["FEZ_PUBLISHED"]
    sb_2level = gate_mod._build_rep3_kraus_superoperators(fez)
    params_zero = lru_mod.default_lru_params_for_profile(fez, p_leak_2q=0.0)
    bundle_qutrit = lru_mod.build_qutrit_lru_superoperators(fez, params_zero, "NO_LRU")

    for r_val in (1, 2, 3):
        dem = gate_mod.build_repetition_d3_dem(rounds=r_val, profile=fez, basis="X")
        dec = gate_mod.auditable_mwpm_decoder(dem)
        # Chronological exact 2-level branch sum over sb_2level["m_ops"]
        br_2level = {0: sb_2level["rho_init_vec"].copy()}
        for _ in range(r_val):
            br_2level = {
                (h << 2) | m: sb_2level["m_ops"][m] @ v
                for h, v in br_2level.items()
                for m in range(4)
            }
        p_2level = 0.0
        for h, v in br_2level.items():
            tp = np.maximum(0.0, np.real(sb_2level["meas_linear"] @ v))
            m_seq = [(h >> (2 * (r_val - 1 - r))) & 3 for r in range(r_val)]
            bdm = 0
            for r in range(r_val):
                diff = m_seq[r] ^ (0 if r == 0 else m_seq[r - 1])
                if (diff >> 1) & 1:
                    bdm |= 1 << (2 * r)
                if diff & 1:
                    bdm |= 1 << (2 * r + 1)
            ml0, ml1 = (m_seq[-1] >> 1) & 1, m_seq[-1] & 1
            for d_out in range(8):
                d0, d1, d2 = (d_out >> 2) & 1, (d_out >> 1) & 1, d_out & 1
                fm = bdm | ((ml0 ^ d0 ^ d1) << (2 * r_val)) | ((ml1 ^ d1 ^ d2) << (2 * r_val + 1))
                if dec.decode_mask(fm) != (d0 ^ d1 ^ d2):
                    p_2level += float(tp[d_out])
        p_qutrit = lru_mod.exact_qutrit_branch_sum_arm4a(r_val, bundle_qutrit, dec)
        assert abs(p_2level - p_qutrit) < 1e-14

    for pid in ("FEZ_PUBLISHED", "FEZ_DRIFTED_SPAM", "H2_1_PUBLISHED"):
        c_zero = lru_report["cells"][f"{pid}__pleak_0.00000__NO_LRU"]
        ref_arm4a = crossing_report["cells"][f"repetition_d3_5q__{pid}"]["unconditional_arms"][
            "arm4a_decoded_pauli_frame"
        ]
        assert c_zero["p_fail_by_round"] == ref_arm4a["p_fail_by_round"]
        assert c_zero["fit"] == ref_arm4a["fit"]
        assert c_zero["delta_eps_L_slope_vs_zero_leak"] == 0.0
        assert (
            lru_report["verification_controls"]["zero_leakage_exact_card_c_agreement"][pid][
                "max_p_fail_diff_vs_card_c_report"
            ]
            == 0.0
        )


# ---------------------------------------------------------------------------
# 2. Physical Monotonicity of Qutrit |2> Leakage & LRU Mitigation
# ---------------------------------------------------------------------------


def test_qutrit_leakage_physical_monotonicity_streaks_and_ion_comparison(
    lru_report: dict[str, Any],
) -> None:
    cells = lru_report["cells"]

    # 1. Monotonic degradation of NO_LRU across p_leak_cz in [0.0, 2e-4, 5e-4, 1.2e-3]
    no_lru_slopes = [
        cells[f"FEZ_PUBLISHED__pleak_{plk:.5f}__NO_LRU"]["fit"]["epsilon_L_slope_fit"]
        for plk in (0.0, 0.0002, 0.0005, 0.0012)
    ]
    assert all(
        no_lru_slopes[i] < no_lru_slopes[i + 1] for i in range(len(no_lru_slopes) - 1)
    ), f"Expected strict NO_LRU degradation with p_leak_cz, got {no_lru_slopes}"

    # 2. Strict architecture ordering NO_LRU > ANCILLA_LRU_ONLY > FULL_DATA_AND_ANCILLA_LRU
    #    at p_leak_cz = 1.2e-3 and 5e-4 on FEZ_PUBLISHED, 1.2e-3 on FEZ_DRIFTED_SPAM,
    #    and p_leak_zz = 5e-5 on H2_1_PUBLISHED
    for pid, plk in (
        ("FEZ_PUBLISHED", 0.0012),
        ("FEZ_PUBLISHED", 0.0005),
        ("FEZ_DRIFTED_SPAM", 0.0012),
        ("H2_1_PUBLISHED", 0.00005),
    ):
        c_no = cells[f"{pid}__pleak_{plk:.5f}__NO_LRU"]
        c_an = cells[f"{pid}__pleak_{plk:.5f}__ANCILLA_LRU_ONLY"]
        c_fu = cells[f"{pid}__pleak_{plk:.5f}__FULL_DATA_AND_ANCILLA_LRU"]
        s_no = c_no["fit"]["epsilon_L_slope_fit"]
        s_an = c_an["fit"]["epsilon_L_slope_fit"]
        s_fu = c_fu["fit"]["epsilon_L_slope_fit"]
        assert s_no > s_an > s_fu, f"Ordering violated for {pid} p_leak={plk}: {s_no}, {s_an}, {s_fu}"
        assert c_fu["crossing_vs_card_c_baselines"]["verdict"] == "PASS_CROSSING_OBSERVED"
        assert c_fu["crossing_vs_card_c_baselines"]["r_cross_vs_both_sustained"] == 1

    # 3. Leakage streak & population diagnostics at p_leak_cz = 1.2e-3 on FEZ_PUBLISHED
    fez_no = cells["FEZ_PUBLISHED__pleak_0.00120__NO_LRU"]
    fez_an = cells["FEZ_PUBLISHED__pleak_0.00120__ANCILLA_LRU_ONLY"]
    fez_fu = cells["FEZ_PUBLISHED__pleak_0.00120__FULL_DATA_AND_ANCILLA_LRU"]

    assert fez_no["leakage_diagnostics"]["ancilla_streak_ge2_fraction_at_max_round"] > 0.01
    assert fez_an["leakage_diagnostics"]["ancilla_streak_ge2_fraction_at_max_round"] < 1e-4
    assert fez_fu["leakage_diagnostics"]["ancilla_streak_ge2_fraction_at_max_round"] < 1e-4

    # Under NO_LRU, multi-round leakage bends per-round eps_L(R) upward from R=3 to R=9
    assert (
        fez_no["fit"]["per_round_epsilon_L"]["9"] > fez_no["fit"]["per_round_epsilon_L"]["3"]
    )
    # Under FULL_LRU, per-round eps_L(R) decreases from R=3 to R=9 and data |2> is pinned < 1e-5
    assert (
        fez_fu["fit"]["per_round_epsilon_L"]["9"] < fez_fu["fit"]["per_round_epsilon_L"]["3"]
    )
    assert fez_an["leakage_diagnostics"]["mean_data_qutrit_p2_by_round"]["9"] > 0.005
    assert fez_fu["leakage_diagnostics"]["mean_data_qutrit_p2_by_round"]["9"] < 1e-5


def test_qutrit_artifact_governance_and_deterministic_regeneration(
    lru_report: dict[str, Any],
    tmp_path: Path,
) -> None:
    assert lru_report["schema"] == "osiris.qec_card_c.qutrit_leakage_lru_report/1"
    assert lru_report["execution_tier"] == "SIMULATED"
    assert lru_report["provider_job_id"] is None
    assert lru_report["qpu_seconds"] == 0
    assert lru_report["hardware_execution_authorized"] is False
    assert lru_report["stop_go_decision"]["verdict"] == "PASS_QUTRIT_LEAKAGE_GAP_CLOSED_WITH_LRU"

    raw_saved = LRU_REPORT_PATH.read_text(encoding="utf-8")
    assert raw_saved == json.dumps(lru_report, indent=2, sort_keys=True) + "\n"

    out_a = tmp_path / "lru_a.json"
    out_b = tmp_path / "lru_b.json"
    lru_mod.run_card_c_qutrit_leakage_lru_gate(
        output_path=out_a,
        seed=20261124,
        rounds_list=(1, 3, 5),
        n_shots=512,
        n_boot=50,
    )
    lru_mod.run_card_c_qutrit_leakage_lru_gate(
        output_path=out_b,
        seed=20261124,
        rounds_list=(1, 3, 5),
        n_shots=512,
        n_boot=50,
    )
    assert out_a.read_bytes() == out_b.read_bytes()


# ---------------------------------------------------------------------------
# 3. Paper 8 Asset Generator, JCS Fallback, LaTeX Zero-Floats, PDF & SYS-024
# ---------------------------------------------------------------------------


def test_paper8_deterministic_regeneration_jcs_fallback_and_sha256sums(
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
        assert path.is_file(), f"Missing required Paper 8 deliverable: {path}"

    before_bytes = {path.name: path.read_bytes() for path in tracked_files}

    proc = subprocess.run(
        [sys.executable, str(ASSET_SCRIPT_PATH)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"Paper 8 generator failed:\n{proc.stderr}\n{proc.stdout}"

    after_bytes = {path.name: path.read_bytes() for path in tracked_files}
    assert before_bytes == after_bytes, (
        "Re-running generate_echo_qec_paper_assets.py mutated tracked deliverables"
    )

    # Verify graceful pure-Python JCS fallback produces identical canonical SHA-256 without rfc8785
    capsule = json.loads(CAPSULE_PATH.read_text(encoding="utf-8"))
    capsule_body = {k: v for k, v in capsule.items() if k != "rfc8785_canonical_sha256"}
    sha_with_default = hashlib.sha256(paper_mod.jcs_bytes(capsule_body)).hexdigest()
    sha_exec_kernel = hashlib.sha256(exec_kernel.canon(capsule_body).encode("utf-8")).hexdigest()
    monkeypatch.setitem(sys.modules, "rfc8785", None)
    sha_fallback = hashlib.sha256(paper_mod.jcs_bytes(capsule_body)).hexdigest()
    assert capsule["rfc8785_canonical_sha256"] == sha_with_default == sha_exec_kernel == sha_fallback

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
    assert len(verified_lines) == 15


def test_paper8_latex_macros_zero_prose_floats_and_structure(
    echo_report: dict[str, Any],
    crossing_report: dict[str, Any],
    lru_report: dict[str, Any],
) -> None:
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
    assert len(defined_macros) >= 130
    assert all(name.startswith("ecq") and name.isalpha() for name in defined_macros)

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

    # Cross-check empirical macros against the three source reports
    rk = echo_report["ranking_comparison"]["rankers"]
    assert macros["ecqExecTier"] == r"\texttt{SIMULATED}"
    assert macros["ecqJobId"] == r"\texttt{null}"
    assert macros["ecqQpuSec"] == "0"
    assert macros["ecqEchoSpearman"] == f"{rk['ECHO_RANKED']['mean_spearman']:.4f}"
    assert macros["ecqCalSpearman"] == f"{rk['CALIBRATION_RANKED']['mean_spearman']:.4f}"
    assert (
        macros["ecqDeltaSpearman"]
        == f"{echo_report['ranking_comparison']['delta_spearman_point']:.4f}"
    )
    assert (
        macros["ecqFezRepArmFourASlope"]
        == f"{crossing_report['cells']['repetition_d3_5q__FEZ_PUBLISHED']['unconditional_arms']['arm4a_decoded_pauli_frame']['fit']['epsilon_L_slope_fit']:.6f}"
    )
    assert (
        macros["ecqLruNoHighSlope"]
        == f"{lru_report['cells']['FEZ_PUBLISHED__pleak_0.00120__NO_LRU']['fit']['epsilon_L_slope_fit']:.6f}"
    )
    assert (
        macros["ecqLruAncHighSlope"]
        == f"{lru_report['cells']['FEZ_PUBLISHED__pleak_0.00120__ANCILLA_LRU_ONLY']['fit']['epsilon_L_slope_fit']:.6f}"
    )
    assert (
        macros["ecqLruFullHighSlope"]
        == f"{lru_report['cells']['FEZ_PUBLISHED__pleak_0.00120__FULL_DATA_AND_ANCILLA_LRU']['fit']['epsilon_L_slope_fit']:.6f}"
    )


def test_paper8_standalone_pdf14_xref_sys024_and_secret_hygiene(
    lru_report: dict[str, Any],
) -> None:
    pdf_bytes = PDF_PATH.read_bytes()
    page_count, streams = _audit_pdf14_bytes(pdf_bytes, PDF_PATH.name)
    assert page_count >= 4, f"Expected >= 4 pages in {PDF_PATH.name}, got {page_count}"
    assert b"/MediaBox [0 0 612.00 792.00]" in pdf_bytes
    pdf_text = _extract_pdf_text(streams)

    for page_idx in range(1, page_count + 1):
        assert f"Page {page_idx} of {page_count}" in pdf_text

    required_pdf_anchors = (
        "SIMULATED",
        "provider_job_id: null",
        "Card B",
        "Card C",
        "0.9333",
        "0.3871",
        "+0.5462",
        "0.0043",
        "STOP_NO_CROSSING_AT_TESTED_ROUNDS",
        "0.001896",
        "0.001159",
        "0.000955",
        "mechanism_inferred",
        "Devin Phillip Davis",
    )
    for anchor in required_pdf_anchors:
        assert anchor in pdf_text, f"Missing empirical anchor {anchor!r} in {PDF_PATH.name}"

    raw_tex = TEX_PATH.read_text(encoding="utf-8")
    gen_tex = GEN_TEX_PATH.read_text(encoding="utf-8")
    expanded_tex = _expand_macros(raw_tex, _parse_macros(gen_tex))
    lru_json_str = json.dumps(lru_report, sort_keys=True)

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
        ("lru_report_json", lru_json_str),
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
    assert capsule["execution_tier"] == "SIMULATED"
    assert capsule["provider_job_id"] is None
    assert capsule["qpu_seconds"] == 0
    assert capsule["hardware_execution_authorized"] is False
    for rel_path, expected_sha in capsule["source_artifacts"].items():
        actual_sha = hashlib.sha256((REPO_ROOT / rel_path).read_bytes()).hexdigest()
        assert actual_sha == expected_sha, f"Source artifact hash mismatch for {rel_path}"

    for folder in (
        PAPER_DIR,
        REPO_ROOT / "experiments" / "qec_card_c",
        REPO_ROOT / "experiments" / "echo_placement",
    ):
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix in {".tex", ".json", ".md", ".svg", ".txt", ".py"}:
                txt = p.read_text(encoding="utf-8")
                assert "crn:v1:bluemix:" not in txt, f"CRN leaked in {p}"
                assert "ZENODO_TOKEN=" not in txt, f"Zenodo token leaked in {p}"
                if p.suffix != ".py":
                    assert '"apikey"' not in txt, f"Secret apikey field leaked in {p}"


def test_qiskit_unitary_embedding_crosscheck_when_available() -> None:
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Operator

    op_h = Operator.from_label("H").data
    u3 = lru_mod.embed_2q_to_3q_1q(op_h)
    assert u3.shape == (3, 3)
    assert np.allclose(u3[:2, :2], op_h)
    assert u3[2, 2] == pytest.approx(1.0)
    assert np.allclose(u3.conj().T @ u3, np.eye(3))
