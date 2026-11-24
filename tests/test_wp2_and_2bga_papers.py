"""Verification suite for WP2 27Q Debiased OT and Non-Abelian 2BGA / ADV-1 REVTeX 4.2 packages.

Covers:
  1. Required files, REVTeX 4.2 structure, author/affiliation, sections, and vector SVG/PDF figures
  2. Byte-for-byte deterministic asset regeneration for both manuscript packages
  3. Verbatim number firewall (zero un-macroed prose floats) and 1-to-1 macro completeness
  4. Balanced LaTeX braces, inline math delimiters, and \\begin/\\end environment stacks
  5. SYS-024 firewall: claims.flagged_in_reply() == [], zero quarantined constants, and zero
     un-negated FLAG_STANDINGS / NO_EVIDENCE sentence hits under claims.match()
  6. RFC 8785 JCS canonicalization, provenance tier binding, and SHA256SUMS.txt integrity
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Mapping, Set

import pytest

try:
    import rfc8785
except ImportError:
    from osiris_cli import orchestration_departments as _od

    class _Rfc8785Fallback:
        @staticmethod
        def dumps(obj):
            return _od.jcs_bytes(obj)

    import sys as _sys

    rfc8785 = _Rfc8785Fallback()  # type: ignore[assignment]
    _sys.modules.setdefault("rfc8785", rfc8785)  # type: ignore[arg-type]

from osiris_cli import claim_provenance, claims
from papers.nonabelian_2bga_and_adv1 import generate_2bga_adv1_assets as gen_2bga
from papers.wp2_27q_debiased_ot_fez import generate_wp2_paper_assets as gen_wp2

REPO_ROOT = Path(__file__).resolve().parents[1]
WP2_DIR = REPO_ROOT / "papers" / "wp2_27q_debiased_ot_fez"
BGA_DIR = REPO_ROOT / "papers" / "nonabelian_2bga_and_adv1"

STANDARD_LATEX_CAP_COMMANDS: Set[str] = {
    "Delta",
    "Gamma",
    "Lambda",
    "Omega",
    "Phi",
    "Pi",
    "Psi",
    "Sigma",
    "Theta",
    "Upsilon",
    "Xi",
    "LaTeX",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _strip_comments(tex: str) -> str:
    out_lines = []
    for line in tex.splitlines():
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


def _parse_generated_macros(gen_tex: str) -> Dict[str, str]:
    clean = _strip_comments(gen_tex)
    macros: Dict[str, str] = {}
    pos = 0
    prefix = "\\newcommand{\\"
    while True:
        start = clean.find(prefix, pos)
        if start == -1:
            break
        name_start = start + len(prefix)
        name_end = clean.find("}", name_start)
        assert name_end != -1, f"Unclosed macro name at offset {start}"
        name = clean[name_start:name_end]
        assert re.fullmatch(r"[A-Za-z]+", name), f"Invalid macro name: {name!r}"
        assert name not in macros, f"Duplicate macro definition: {name!r}"

        brace_start = clean.find("{", name_end + 1)
        assert brace_start != -1, f"Missing body brace for macro {name}"
        depth = 0
        i = brace_start
        while i < len(clean):
            ch = clean[i]
            if ch == "\\" and i + 1 < len(clean):
                if clean[i + 1] in ("\\", "{", "}"):
                    i += 2
                    continue
                i += 1
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        assert depth == 0, f"Unbalanced braces in macro body for {name}"
        macros[name] = clean[brace_start + 1 : i]
        pos = i + 1
    return macros


def _expand_macros(tex: str, macros: Mapping[str, str]) -> str:
    expanded = tex.replace("\\input{generated.tex}", "")
    for name in sorted(macros.keys(), key=len, reverse=True):
        val = macros[name]
        expanded = re.sub(rf"\\{name}\{{\}}", lambda _m, v=val: v, expanded)
        expanded = re.sub(rf"\\{name}(?![A-Za-z])", lambda _m, v=val: v, expanded)
    return expanded


def _assert_balanced_braces(text: str, label: str) -> None:
    clean = _strip_comments(text)
    depth = 0
    i = 0
    while i < len(clean):
        ch = clean[i]
        if ch == "\\" and i + 1 < len(clean):
            if clean[i + 1] in ("\\", "{", "}"):
                i += 2
                continue
            i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            assert depth >= 0, f"{label}: negative brace depth near offset {i}"
        i += 1
    assert depth == 0, f"{label}: unclosed braces remaining (depth={depth})"


def _assert_balanced_environments_and_math(tex: str, label: str) -> None:
    clean = _strip_comments(tex)
    stack = []
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", clean):
        kind, env = m.group(1), m.group(2)
        if kind == "begin":
            stack.append(env)
        else:
            assert stack, f"{label}: unexpected \\end{{{env}}} with empty stack"
            top = stack.pop()
            assert top == env, f"{label}: mismatched \\begin{{{top}}} vs \\end{{{env}}}"
    assert not stack, f"{label}: unclosed environments: {stack}"

    # Remove escaped \$ and display $$ blocks, then verify inline $ count is even
    no_esc_dollar = re.sub(r"\\\$", "", clean)
    no_display = re.sub(r"\$\$", "", no_esc_dollar)
    dollar_count = no_display.count("$")
    assert dollar_count % 2 == 0, f"{label}: odd number of inline '$' delimiters ({dollar_count})"


def test_package1_files_and_revtex_structure() -> None:
    required = [
        "generate_wp2_paper_assets.py",
        "wp2_27q_debiased_ot_fez.tex",
        "generated.tex",
        "fig1_27q_fez_sparsity_collapse_vs_eq7.svg",
        "fig1_27q_fez_sparsity_collapse_vs_eq7.pdf",
        "fig2_coherent_drift_calibration_and_heron_rescoring.svg",
        "fig2_coherent_drift_calibration_and_heron_rescoring.pdf",
        "reproducibility_capsule.json",
        "SHA256SUMS.txt",
    ]
    for fname in required:
        p = WP2_DIR / fname
        assert p.is_file() and p.stat().st_size > 0, f"Missing or empty WP2 file: {fname}"

    tex = (WP2_DIR / "wp2_27q_debiased_ot_fez.tex").read_text(encoding="utf-8")
    assert "{revtex4-2}" in tex
    assert (
        "\\title{Eliminating Finite-Shot Hypercube Sparsity Bias in 27-Qubit Quantum Processor "
        "Stability Benchmarks via Split-Sample-Debiased Optimal Transport}"
    ) in tex
    assert "\\author{Devin Phillip Davis}" in tex
    assert "\\affiliation{Agile Defense Systems LLC}" in tex
    assert "\\input{generated.tex}" in tex
    for sec in (
        "\\section{Introduction and the Weed--Bach $O(S^{-1/Q})$ Trap}",
        "\\section{Split-Sample-Debiased Kantorovich and Sinkhorn Estimators}",
        "\\section{Constant-Skeleton 27-Qubit \\texttt{ibm\\_fez} Experiment and Resolution of \\texttt{NB-11}}",
        "\\section{Multi-Processor Re-Scoring on \\texttt{ibm\\_marrakesh} and 49-Bit \\texttt{OSIRIS-QEC-001}}",
        "\\section{Conclusion and Data Availability}",
    ):
        assert sec in tex

    for stem in (
        "fig1_27q_fez_sparsity_collapse_vs_eq7",
        "fig2_coherent_drift_calibration_and_heron_rescoring",
    ):
        svg = (WP2_DIR / f"{stem}.svg").read_text(encoding="utf-8")
        pdf = (WP2_DIR / f"{stem}.pdf").read_bytes()
        assert svg.startswith("<svg ") and svg.strip().endswith("</svg>")
        assert pdf.startswith(b"%PDF-1.4") and pdf.strip().endswith(b"%%EOF")


def test_package2_files_and_revtex_structure() -> None:
    required = [
        "generate_2bga_adv1_assets.py",
        "nonabelian_2bga_and_adv1.tex",
        "generated.tex",
        "fig1_odd_k_nonabelian_2bga_codes_and_bposd.svg",
        "fig1_odd_k_nonabelian_2bga_codes_and_bposd.pdf",
        "fig2_adv1_three_attack_ablation_m6.svg",
        "fig2_adv1_three_attack_ablation_m6.pdf",
        "reproducibility_capsule.json",
        "SHA256SUMS.txt",
    ]
    for fname in required:
        p = BGA_DIR / fname
        assert p.is_file() and p.stat().st_size > 0, f"Missing or empty 2BGA/ADV-1 file: {fname}"

    tex = (BGA_DIR / "nonabelian_2bga_and_adv1.tex").read_text(encoding="utf-8")
    assert "{revtex4-2}" in tex
    assert (
        "\\title{Left-Right Rank Duality in Non-Abelian Two-Block Group-Algebra Quantum LDPC Codes "
        "and a Three-Attack Classical-Hardness Pre-Gate for Looped Non-Mirror Peaked Circuits}"
    ) in tex
    assert "\\author{Devin Phillip Davis}" in tex
    assert "\\affiliation{Agile Defense Systems LLC}" in tex
    assert "\\input{generated.tex}" in tex
    assert "[[42,3,6]]" in tex
    assert "[[36,3,6]]" in tex
    assert "[[114,3,12]]" in tex
    assert "M_6 = J_6 - I_6" in tex

    for stem in (
        "fig1_odd_k_nonabelian_2bga_codes_and_bposd",
        "fig2_adv1_three_attack_ablation_m6",
    ):
        svg = (BGA_DIR / f"{stem}.svg").read_text(encoding="utf-8")
        pdf = (BGA_DIR / f"{stem}.pdf").read_bytes()
        assert svg.startswith("<svg ") and svg.strip().endswith("</svg>")
        assert pdf.startswith(b"%PDF-1.4") and pdf.strip().endswith(b"%%EOF")


@pytest.mark.parametrize(
    ("pkg_dir", "regen_fn", "tracked_files"),
    [
        (
            WP2_DIR,
            gen_wp2.generate_all,
            [
                "generated.tex",
                "fig1_27q_fez_sparsity_collapse_vs_eq7.svg",
                "fig1_27q_fez_sparsity_collapse_vs_eq7.pdf",
                "fig2_coherent_drift_calibration_and_heron_rescoring.svg",
                "fig2_coherent_drift_calibration_and_heron_rescoring.pdf",
                "reproducibility_capsule.json",
                "SHA256SUMS.txt",
            ],
        ),
        (
            BGA_DIR,
            gen_2bga.generate_all,
            [
                "generated.tex",
                "fig1_odd_k_nonabelian_2bga_codes_and_bposd.svg",
                "fig1_odd_k_nonabelian_2bga_codes_and_bposd.pdf",
                "fig2_adv1_three_attack_ablation_m6.svg",
                "fig2_adv1_three_attack_ablation_m6.pdf",
                "reproducibility_capsule.json",
                "SHA256SUMS.txt",
            ],
        ),
    ],
)
def test_byte_for_byte_deterministic_regeneration(pkg_dir, regen_fn, tracked_files) -> None:
    before = {fn: (pkg_dir / fn).read_bytes() for fn in tracked_files}
    regen_fn()
    after = {fn: (pkg_dir / fn).read_bytes() for fn in tracked_files}
    for fn in tracked_files:
        assert after[fn] == before[fn], f"Non-deterministic regeneration in {pkg_dir.name}/{fn}"


@pytest.mark.parametrize(
    ("pkg_dir", "tex_fn"),
    [
        (WP2_DIR, "wp2_27q_debiased_ot_fez.tex"),
        (BGA_DIR, "nonabelian_2bga_and_adv1.tex"),
    ],
)
def test_verbatim_number_firewall_and_macro_completeness(pkg_dir: Path, tex_fn: str) -> None:
    tex = (pkg_dir / tex_fn).read_text(encoding="utf-8")
    gen = (pkg_dir / "generated.tex").read_text(encoding="utf-8")
    macros = _parse_generated_macros(gen)
    assert len(macros) >= 50

    body_no_bib = _strip_comments(tex).split("\\begin{thebibliography}")[0]
    unmacroed_floats = re.findall(r"\b\d+\.\d+\b", body_no_bib)
    assert unmacroed_floats == [], f"Un-macroed prose floats in {tex_fn}: {unmacroed_floats}"

    used = {
        m
        for m in re.findall(r"\\([A-Z][A-Za-z]+)(?![A-Za-z])", body_no_bib)
        if m not in STANDARD_LATEX_CAP_COMMANDS
    }
    defined = set(macros.keys())
    assert used - defined == set(), f"Undefined macros in {tex_fn}: {sorted(used - defined)}"
    assert defined - used == set(), f"Unused macros in {pkg_dir.name}/generated.tex: {sorted(defined - used)}"


def test_generated_macros_match_verified_json_artifacts() -> None:
    wp2_macros = _parse_generated_macros((WP2_DIR / "generated.tex").read_text(encoding="utf-8"))
    fez = json.loads(
        (REPO_ROOT / "experiments/wp2_27q_fez/physical_wp2_27q_fez_results.json").read_text(
            encoding="utf-8"
        )
    )
    assert wp2_macros["WpFezJobId"] == "dc1tqvinj2vs73dfrts0"
    assert wp2_macros["WpMarrJobId"] == "dau0q3qhcrkc73durtgg"
    assert wp2_macros["WpQecJobId"] == "dbtl2vb2kf8s73c9a0d0"
    assert wp2_macros["WpFezTotalShots"] == "98{,}304"
    assert wp2_macros["WpFezCzCount"] == str(fez["constant_cz_count"])
    assert wp2_macros["WpFezSxCount"] == str(fez["constant_sx_count"])
    assert wp2_macros["WpDriftLargeZ"] == f"{fez['controlled_drift_calibration_arm_b']['z_score_0p45_vs_aa_null']:.2f}"

    bga_macros = _parse_generated_macros((BGA_DIR / "generated.tex").read_text(encoding="utf-8"))
    qldpc = json.loads(
        (REPO_ROOT / "qldpc_nonabelian/artifacts/nonabelian_2bga_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert bga_macros["TwoBgaPairsChecked"] == "6{,}480"
    assert bga_macros["TwoBgaRankViolations"] == "0"
    assert bga_macros["TwoBgaOddKCount"] == str(qldpc["odd_k_d_ge_4_discoveries_count"])
    assert bga_macros["AdvOnePermittedCount"] == "2"


@pytest.mark.parametrize(
    ("pkg_dir", "tex_fn"),
    [
        (WP2_DIR, "wp2_27q_debiased_ot_fez.tex"),
        (BGA_DIR, "nonabelian_2bga_and_adv1.tex"),
    ],
)
def test_balanced_latex_braces_and_environments(pkg_dir: Path, tex_fn: str) -> None:
    tex = (pkg_dir / tex_fn).read_text(encoding="utf-8")
    gen = (pkg_dir / "generated.tex").read_text(encoding="utf-8")
    macros = _parse_generated_macros(gen)
    expanded = _expand_macros(tex, macros)

    _assert_balanced_braces(gen, f"{pkg_dir.name}/generated.tex")
    _assert_balanced_braces(tex, f"{pkg_dir.name}/{tex_fn}")
    _assert_balanced_braces(expanded, f"{pkg_dir.name}/{tex_fn} [expanded]")
    _assert_balanced_environments_and_math(tex, f"{pkg_dir.name}/{tex_fn}")
    _assert_balanced_environments_and_math(expanded, f"{pkg_dir.name}/{tex_fn} [expanded]")


@pytest.mark.parametrize(
    ("pkg_dir", "tex_fn"),
    [
        (WP2_DIR, "wp2_27q_debiased_ot_fez.tex"),
        (BGA_DIR, "nonabelian_2bga_and_adv1.tex"),
    ],
)
def test_zero_sys024_flagged_claims_and_quarantined_constants(pkg_dir: Path, tex_fn: str) -> None:
    tex = (pkg_dir / tex_fn).read_text(encoding="utf-8")
    gen = (pkg_dir / "generated.tex").read_text(encoding="utf-8")
    macros = _parse_generated_macros(gen)
    expanded = _expand_macros(tex, macros)

    assert claims.flagged_in_reply(tex) == []
    assert claims.flagged_in_reply(expanded) == []

    for q_id, q_re in claim_provenance._QUARANTINED_CONSTANTS:
        assert not q_re.search(tex), f"Quarantined constant {q_id} found in {tex_fn}"
        assert not q_re.search(gen), f"Quarantined constant {q_id} found in {pkg_dir.name}/generated.tex"
        assert not q_re.search(expanded), f"Quarantined constant {q_id} found in expanded {tex_fn}"

    for doc_label, doc_body in (
        ("raw_body", _strip_comments(tex).split("\\begin{thebibliography}")[0]),
        ("expanded_body", _strip_comments(expanded).split("\\begin{thebibliography}")[0]),
    ):
        for m in claims._SENTENCE.finditer(doc_body):
            sent = m.group(0).strip()
            hits = [
                c
                for c in claims.match(sent)
                if c.standing() in claims.FLAG_STANDINGS or c.verdict == "NO_EVIDENCE"
            ]
            if hits:
                assert claims._REPORTS_VERDICT.search(sent), (
                    f"Un-negated SYS-024 claim {[c.id for c in hits]} in {pkg_dir.name} "
                    f"({doc_label}): {sent!r}"
                )


def test_rfc8785_capsules_and_sha256sums_integrity() -> None:
    # 1. WP2 capsule (HARDWARE_ATTESTED)
    wp2_raw = (WP2_DIR / "reproducibility_capsule.json").read_bytes()
    wp2_cap = json.loads(wp2_raw.decode("utf-8"))
    assert wp2_raw == rfc8785.dumps(wp2_cap)
    wp2_body = {k: v for k, v in wp2_cap.items() if k != "capsule_body_sha256"}
    assert wp2_cap["capsule_body_sha256"] == _sha256_bytes(rfc8785.dumps(wp2_body))
    assert wp2_cap["evidence_class"] == "HARDWARE_ATTESTED"
    assert wp2_cap["hardware_executed"] is True
    assert wp2_cap["provider_job_id"] == "dc1tqvinj2vs73dfrts0"
    assert wp2_cap["provider_job_ids"] == [
        "dc1tqvinj2vs73dfrts0",
        "dau0q3qhcrkc73durtgg",
        "dbtl2vb2kf8s73c9a0d0",
    ]
    assert wp2_cap["simulated_companion_tier"]["evidence_class"] == "SIMULATED"
    assert wp2_cap["simulated_companion_tier"]["provider_job_id"] is None

    # 2. 2BGA + ADV-1 capsule (SIMULATED)
    bga_raw = (BGA_DIR / "reproducibility_capsule.json").read_bytes()
    bga_cap = json.loads(bga_raw.decode("utf-8"))
    assert bga_raw == rfc8785.dumps(bga_cap)
    bga_body = {k: v for k, v in bga_cap.items() if k != "capsule_body_sha256"}
    assert bga_cap["capsule_body_sha256"] == _sha256_bytes(rfc8785.dumps(bga_body))
    assert bga_cap["evidence_class"] == "SIMULATED"
    assert bga_cap["hardware_executed"] is False
    assert bga_cap["provider_job_id"] is None

    # 3. Verify every digest inside both capsules and both SHA256SUMS.txt manifests
    for pkg_dir, cap in ((WP2_DIR, wp2_cap), (BGA_DIR, bga_cap)):
        for section in ("input_artifacts", "generated_assets"):
            for rel, expected_sha in cap[section].items():
                actual_sha = _sha256_bytes((REPO_ROOT / rel).read_bytes())
                assert actual_sha == expected_sha, f"Digest mismatch in {pkg_dir.name} capsule: {rel}"

        sha_lines = [
            ln.strip()
            for ln in (pkg_dir / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        assert len(sha_lines) >= 12
        for ln in sha_lines:
            expected_sha, rel = ln.split("  ", 1)
            assert re.fullmatch(r"[0-9a-f]{64}", expected_sha)
            target = REPO_ROOT / rel
            assert target.is_file(), f"Missing file in {pkg_dir.name}/SHA256SUMS.txt: {rel}"
            assert _sha256_bytes(target.read_bytes()) == expected_sha, f"SHA256 mismatch for {rel}"
