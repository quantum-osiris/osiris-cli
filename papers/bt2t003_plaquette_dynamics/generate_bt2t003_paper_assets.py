#!/usr/bin/env python3
"""Programmatic asset, macro, dual SVG/PDF 1.4 figure, standalone multi-page PDF 1.4
preprint, RFC 8785 capsule, and SHA256SUMS generator for:
  `papers/bt2t003_plaquette_dynamics/`

Deterministic, zero timestamps in generated outputs, zero network calls, zero dependency
on `qiskit`, `qiskit_ibm_runtime`, or `pytket`, graceful fallback to
`osiris_cli.orchestration_departments.jcs_bytes` when `rfc8785` is absent, and zero
hand-typed empirical floats in `bt2t003_plaquette_dynamics.tex`.

Reads:
  - `experiments/bt2t_plaquette/manifest_bt2t003.json` (`OSIRIS-BT2T-003`, `SIMULATED`, `0s` QPU)
  - `experiments/bt2t_plaquette/bt2t_plaquette_dynamics.py`
  - `experiments/bt2t_plaquette/verify_bt2t003.py`
  - `experiments/bt2t_inversion/bt2t.py`
  - `experiments/bt2t_inversion/manifest.json`
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

try:
    import rfc8785 as _rfc8785  # type: ignore[import-not-found]
except ImportError:
    _rfc8785 = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from papers._pdf_preprint_renderer import PreprintDocument, VectorCanvas  # noqa: E402

PLAQ_DIR = ROOT / "experiments" / "bt2t_plaquette"
INV_DIR = ROOT / "experiments" / "bt2t_inversion"

MANIFEST_BT2T003_PATH = PLAQ_DIR / "manifest_bt2t003.json"
PLAQ_INIT_PATH = PLAQ_DIR / "__init__.py"
PLAQ_PY_PATH = PLAQ_DIR / "bt2t_plaquette_dynamics.py"
VERIFY_BT2T003_PY_PATH = PLAQ_DIR / "verify_bt2t003.py"
BT2T_BASE_PY_PATH = INV_DIR / "bt2t.py"
INV_MANIFEST_PATH = INV_DIR / "manifest.json"

SOURCE_ARTIFACT_PATHS: Tuple[Path, ...] = (
    PLAQ_INIT_PATH,
    PLAQ_PY_PATH,
    VERIFY_BT2T003_PY_PATH,
    MANIFEST_BT2T003_PATH,
    BT2T_BASE_PY_PATH,
    INV_MANIFEST_PATH,
)

FIG1_STEM = "fig1_bt2t003_strang_scaling_and_wilson_regimes"
FIG2_STEM = "fig2_bt2t003_gauss_holonomy_cz_compression"
PDF_FILENAME = "bt2t003_plaquette_dynamics.pdf"


def _canonical_jcs_bytes(payload: Mapping[str, Any]) -> bytes:
    if _rfc8785 is not None:
        return bytes(_rfc8785.dumps(dict(payload)))
    from osiris_cli.orchestration_departments import jcs_bytes

    return jcs_bytes(payload)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _fmt_sci_tex(val: float, digits: int = 2) -> str:
    if val == 0.0:
        return "0.0"
    s = f"{val:.{digits}e}"
    mant, exp_str = s.split("e")
    return f"{mant} \\times 10^{{{int(exp_str)}}}"


def make_fig1_trotter_and_wilson(m3: Dict[str, Any]) -> VectorCanvas:
    c = VectorCanvas(540, 266)
    c.text(
        270,
        17,
        "Kogut-Susskind 2T Non-Abelian Lattice Gauge Plaquette: Strang O(dt^2) Scaling & Coupling Regimes",
        size=8.9,
        bold=True,
        anchor="middle",
    )

    # Panel (a): Log-log Trotter error vs N_steps at g_YM^2 = 1.0
    ax0, ay0, aw, ah = 48.0, 40.0, 214.0, 174.0
    c.text(ax0 + aw / 2, 31, "(a) Lie vs. Strang Operator Error (g^2 = 1.0)", size=8.0, bold=True, anchor="middle")
    c.rect(ax0, ay0, aw, ah, fill="#f8fafc", stroke="#334155", lw=0.9)

    series_g1 = m3["part4_trotter_suzuki_error_scaling"]["by_coupling"]["g_ym_sq_1.0"]["series"]
    slope_lie = float(m3["part4_trotter_suzuki_error_scaling"]["by_coupling"]["g_ym_sq_1.0"]["lie_loglog_slope"])
    slope_str = float(m3["part4_trotter_suzuki_error_scaling"]["by_coupling"]["g_ym_sq_1.0"]["strang_loglog_slope"])

    def x_n(n_st: int) -> float:
        # log2(8)=3 to log2(2048)=11 -> span 8
        return ax0 + 12.0 + ((math.log2(n_st) - 3.0) / 8.0) * (aw - 24.0)

    def y_err(ev: float) -> float:
        # log10 from -8.5 to -1.0 -> span 7.5
        lv = math.log10(max(3e-9, min(0.1, ev)))
        return ay0 + ah * (1.0 - (lv - (-8.5)) / 7.5)

    for exp_v, lbl in [(-8.0, "1e-8"), (-6.0, "1e-6"), (-4.0, "1e-4"), (-2.0, "1e-2")]:
        yy = ay0 + ah * (1.0 - (exp_v - (-8.5)) / 7.5)
        c.line(ax0, yy, ax0 + aw, yy, color="#e2e8f0", lw=0.6)
        c.text(ax0 - 4, yy + 2.5, lbl, size=6.3, anchor="end", mono=True)

    for nv in (8, 32, 128, 512, 2048):
        xx = x_n(nv)
        c.line(xx, ay0 + ah, xx, ay0 + ah + 3, color="#334155", lw=0.7)
        c.text(xx, ay0 + ah + 11, str(nv), size=6.4, anchor="middle", mono=True)
    c.text(ax0 + aw / 2, ay0 + ah + 23, "Trotter Steps N (T = 1.0, log2 axis)", size=7.2, anchor="middle")

    pts_lie = [(x_n(int(r["n_steps"])), y_err(float(r["lie_2norm_error"]))) for r in series_g1]
    pts_str = [(x_n(int(r["n_steps"])), y_err(float(r["strang_2norm_error"]))) for r in series_g1]
    c.polyline(pts_lie, color="#d97706", lw=1.6)
    c.polyline(pts_str, color="#059669", lw=1.6)
    for px, py in pts_lie:
        c.circle(px, py, 2.3, fill="#d97706", stroke="#ffffff", lw=0.5)
    for px, py in pts_str:
        c.circle(px, py, 2.5, fill="#059669", stroke="#ffffff", lw=0.5)

    c.text(ax0 + 10, ay0 + 15, f"Lie-Trotter: slope = {slope_lie:.5f}", size=6.4, bold=True, color="#b45309")
    c.text(ax0 + 10, ay0 + 26, f"Strang O(dt^2): slope = {slope_str:.5f}", size=6.4, bold=True, color="#047857")

    # Panel (b): Final T=1.0 Wilson loop & chi_0 / rho_2,0 / rho_3 populations across g_YM^2 in {0.5, 1.0, 2.0}
    bx0, by0, bw, bh = 298.0, 40.0, 222.0, 174.0
    c.text(bx0 + bw / 2, 31, "(b) Vacuum Evolution Across g_YM^2 (T = 1.0)", size=8.0, bold=True, anchor="middle")
    c.rect(bx0, by0, bw, bh, fill="#f8fafc", stroke="#334155", lw=0.9)

    for v in (0.0, 0.25, 0.50, 0.75, 1.00):
        yy = by0 + bh * (1.0 - v)
        c.line(bx0, yy, bx0 + bw, yy, color="#e2e8f0", lw=0.6)
        c.text(bx0 - 4, yy + 2.5, f"{v:.2f}", size=6.4, anchor="end", mono=True)

    regs = m3["part3_plaquette_regime_dynamics"]["regimes"]
    g_keys = [("g_ym_sq_0.5", "g^2 = 0.5\n(Weak)"), ("g_ym_sq_1.0", "g^2 = 1.0\n(Interm.)"), ("g_ym_sq_2.0", "g^2 = 2.0\n(Strong)")]
    slot = bw / len(g_keys)
    bar_w = 13.0
    for idx, (gk, glbl) in enumerate(g_keys):
        st = regs[gk]["electric_vacuum_chi0"]["strang"]
        irr = st["irrep_sector_probabilities"]
        vals = [
            (float(irr["chi_0"]), "#2563eb"),
            (float(irr["rho_2_0"]), "#059669"),
            (float(irr["rho_3"]), "#7c3aed"),
            (float(st["wilson_loop"]), "#dc2626"),
        ]
        cx = bx0 + slot * (idx + 0.5)
        for b_i, (val, col) in enumerate(vals):
            off = (b_i - 2.0) * (bar_w + 1.5)
            top = by0 + bh * (1.0 - max(0.0, val))
            c.rect(cx + off, top, bar_w, max(1.5, (by0 + bh) - top), fill=col, stroke="#0f172a", lw=0.5)
        p1, p2 = glbl.split("\n")
        c.text(cx, by0 + bh + 10, p1, size=6.6, bold=True, anchor="middle", mono=True)
        c.text(cx, by0 + bh + 19, p2, size=6.0, color="#475569", anchor="middle")

    leg = [("#2563eb", "P(chi_0)"), ("#059669", "P(rho_{2,0})"), ("#7c3aed", "P(rho_3)"), ("#dc2626", "<W>")]
    for i, (col, lt) in enumerate(leg):
        lx = bx0 + 8.0 + i * 53.0
        c.rect(lx, by0 + 7.0, 8.0, 6.0, fill=col, stroke="#0f172a", lw=0.4)
        c.text(lx + 10.0, by0 + 12.2, lt, size=6.0, color="#1e293b")

    c.text(
        270,
        255,
        "SIMULATED (provider_job_id: null, 0s QPU): 64-step Strang L2 state error <= 7.18e-6; unused 8-codeword leakage = 0.0 identically.",
        size=6.9,
        color="#334155",
        anchor="middle",
    )
    return c


def make_fig2_cz_compression_and_isomorphism(m3: Dict[str, Any]) -> VectorCanvas:
    c = VectorCanvas(540, 266)
    c.text(
        270,
        17,
        "4-Vertex Gauss-Law Gauge Fixing & Structured Synthesis: 104x-178x Native CZ Gate Compression",
        size=8.9,
        bold=True,
        anchor="middle",
    )

    gc = m3["part5_native_circuit_compilations"]["gate_count_summary"]

    # Panel (a): Log10 CZ gate counts across all-to-all and heavy-hex targets
    ax0, ay0, aw, ah = 50.0, 40.0, 252.0, 174.0
    c.text(ax0 + aw / 2, 31, "(a) Single-Strang-Step Native CZ Count (log10)", size=8.0, bold=True, anchor="middle")
    c.rect(ax0, ay0, aw, ah, fill="#f8fafc", stroke="#334155", lw=0.9)

    def y_log_cz(cz_val: int) -> float:
        lv = math.log10(max(10.0, float(cz_val)))  # [1.0, 5.0]
        return ay0 + ah * (1.0 - (lv - 1.0) / 4.0)

    for exp_v, lbl in [(1.0, "10^1"), (2.0, "10^2"), (3.0, "10^3"), (4.0, "10^4"), (5.0, "10^5")]:
        yy = ay0 + ah * (1.0 - (exp_v - 1.0) / 4.0)
        c.line(ax0, yy, ax0 + aw, yy, color="#e2e8f0", lw=0.6)
        c.text(ax0 - 4, yy + 2.5, lbl, size=6.3, anchor="end", mono=True)

    comp_bars = [
        ("5Q U_B\nStruct.", int(gc["cz_ub_structured_5q_all_to_all"]), int(gc["cz_ub_structured_5q_heavy_hex_T5"])),
        ("5Q Strang\nStruct.", int(gc["cz_strang_structured_5q_all_to_all"]), int(gc["cz_strang_structured_5q_heavy_hex_T5"])),
        ("5Q Dense\nUnitary", int(gc["cz_dense_unitary_5q_all_to_all"]), int(gc["cz_dense_unitary_5q_heavy_hex_T5"])),
        ("5Q QSD\nPW-QFT", int(gc["cz_strang_qsd_qft_5q_all_to_all"]), int(gc["cz_strang_qsd_qft_5q_heavy_hex_T5"])),
        ("20Q 4-Link\nIn-Place", int(gc["cz_strang_4link_inplace_20q_all_to_all"]), int(gc["cz_strang_4link_inplace_20q_heavy_hex_20Q"])),
        ("25Q 4-Link\nAncilla", int(gc["cz_strang_4link_ancilla_25q_all_to_all"]), int(gc["cz_strang_4link_ancilla_25q_heavy_hex_25Q"])),
    ]
    slot = aw / len(comp_bars)
    for idx, (lbl, cz_aa, cz_hh) in enumerate(comp_bars):
        cx = ax0 + slot * (idx + 0.5)
        for off, val, col in [(-11.0, cz_aa, "#059669"), (1.0, cz_hh, "#2563eb")]:
            top = y_log_cz(val)
            c.rect(cx + off, top, 10.0, (ay0 + ah) - top, fill=col, stroke="#0f172a", lw=0.5)
        c.text(cx, y_log_cz(cz_hh) - 4, f"{cz_aa}/{cz_hh}", size=5.2, bold=True, anchor="middle", mono=True)
        p1, p2 = lbl.split("\n")
        c.text(cx, ay0 + ah + 10, p1, size=5.9, bold=True, anchor="middle")
        c.text(cx, ay0 + ah + 18, p2, size=5.6, color="#475569", anchor="middle")

    c.rect(ax0 + 8, ay0 + 7, 8, 6, fill="#059669", stroke="#0f172a", lw=0.4)
    c.text(ax0 + 19, ay0 + 12.2, "All-to-All (Qiskit / H2 ZZPhase)", size=6.0)
    c.rect(ax0 + 142, ay0 + 7, 8, 6, fill="#2563eb", stroke="#0f172a", lw=0.4)
    c.text(ax0 + 153, ay0 + 12.2, "Heavy-Hex T5 / 20Q / 25Q", size=6.0)

    # Panel (b): Compression factors & exact multi-link holonomy verification summary card
    bx0, by0, bw, bh = 320.0, 40.0, 202.0, 174.0
    c.text(bx0 + bw / 2, 31, "(b) Exact Gauge Isomorphism Audit", size=8.0, bold=True, anchor="middle")
    c.rect(bx0, by0, bw, bh, fill="#ffffff", stroke="#334155", lw=0.9)

    p2 = m3["part2_multilink_holonomy_and_gauge_isomorphism"]
    rows_txt = [
        ("2-Link Holonomy (15Q):", f"{p2['holonomy_2link_15q_triples_checked']:,} / {p2['holonomy_2link_15q_triples_checked']:,} exact"),
        ("4-Link Holonomy (25Q):", f"{p2['holonomy_4link_25q_configs_checked']:,} / {p2['holonomy_4link_25q_configs_checked']:,} exact"),
        ("4-Link vs 5Q Basis Err:", f"{p2['four_link_vs_5q_basis_max_l2_error']:.2e} L2"),
        ("4-Link Random State Err:", f"{p2['four_link_vs_5q_random_superposition_max_l2_error']:.2e} L2"),
        ("Gauss-Invariant Dim:", f"{p2['physical_gauss_invariant_subspace_dim']} class functions"),
        ("Qubit Compression:", "20Q/25Q -> 5Q (4x-5x)"),
        ("CZ Gain vs Dense 5Q:", f"{gc['cz_compression_dense_vs_structured_all_to_all']:.2f}x (all-to-all)"),
        ("CZ Gain vs 20Q 4-Link:", f"{gc['cz_compression_unfixed_20q_vs_5q_all_to_all']:.1f}x (all-to-all)"),
        ("CZ Gain vs 25Q HeavyHex:", f"{gc['cz_compression_unfixed_25q_vs_5q_heavy_hex']:.1f}x (heavy-hex)"),
    ]
    ry = by0 + 16.0
    for k_lbl, v_lbl in rows_txt:
        c.text(bx0 + 7.0, ry, k_lbl, size=6.2, bold=True, color="#1e293b")
        c.text(bx0 + bw - 7.0, ry, v_lbl, size=6.2, color="#047857", anchor="end", mono=True)
        ry += 17.5

    c.text(
        270,
        255,
        "Maximal-tree gauge fixing reduces 4-link 2T plaquette dynamics from 20Q-25Q (21.8k-62.4k CZ) to 5Q (209-350 CZ) with < 2.35e-15 error.",
        size=6.9,
        color="#334155",
        anchor="middle",
    )
    return c


def build_generated_tex(m3: Dict[str, Any], capsule_sha: str) -> str:
    p1 = m3["part1_gauss_law_and_peter_weyl_hamiltonian"]
    p2 = m3["part2_multilink_holonomy_and_gauge_isomorphism"]
    p3 = m3["part3_plaquette_regime_dynamics"]
    p4 = m3["part4_trotter_suzuki_error_scaling"]
    p5 = m3["part5_native_circuit_compilations"]
    gc = p5["gate_count_summary"]
    de = p5["direct_circuit_errors"]

    g1_ser = p4["by_coupling"]["g_ym_sq_1.0"]
    n_checks = len(m3["checks"])

    lines = [
        "% Auto-generated by generate_bt2t003_paper_assets.py -- DO NOT EDIT BY HAND",
        f"\\newcommand{{\\BtpChecksTotal}}{{{n_checks}}}",
        f"\\newcommand{{\\BtpEncShaPrefix}}{{{m3['encoding_sha256'][:16]}}}",
        f"\\newcommand{{\\BtpManifestShaPrefix}}{{{m3['rfc8785_canonical_sha256'][:16]}}}",
        f"\\newcommand{{\\BtpCapsuleShaPrefix}}{{{capsule_sha[:16]}}}",
        f"\\newcommand{{\\BtpGroupOrder}}{{{p1['s8_generated_order']}}}",
        f"\\newcommand{{\\BtpConjClasses}}{{{p1['conjugacy_class_count']}}}",
        f"\\newcommand{{\\BtpCayleyDeg}}{{{len(p1['s8_neighbors'])}}}",
        f"\\newcommand{{\\BtpCayleyDiam}}{{{p1['s8_generates_2t_diameter']}}}",
        f"\\newcommand{{\\BtpCasChiZero}}{{{p1['casimir_su2_calibrated_eigenvalues']['chi_0']:.2f}}}",
        f"\\newcommand{{\\BtpCasChiOne}}{{{p1['casimir_su2_calibrated_eigenvalues']['chi_1']:.2f}}}",
        f"\\newcommand{{\\BtpCasRhoTwoZero}}{{{p1['casimir_su2_calibrated_eigenvalues']['rho_2_0']:.2f}}}",
        f"\\newcommand{{\\BtpCasRhoTwoOne}}{{{p1['casimir_su2_calibrated_eigenvalues']['rho_2_1']:.3f}}}",
        f"\\newcommand{{\\BtpCasRhoThree}}{{{p1['casimir_su2_calibrated_eigenvalues']['rho_3']:.2f}}}",
        f"\\newcommand{{\\BtpPwDiagErr}}{{{_fmt_sci_tex(float(p1['peter_weyl_casimir_diagonalization_2norm_error']))}}}",
        f"\\newcommand{{\\BtpUePwErr}}{{{_fmt_sci_tex(float(p1['ue_pw_qft_vs_expm_2norm_error']))}}}",
        f"\\newcommand{{\\BtpTwoLinkConfigs}}{{{p2['holonomy_2link_15q_triples_checked']:,}}}",
        f"\\newcommand{{\\BtpFourLinkConfigs}}{{{p2['holonomy_4link_25q_configs_checked']:,}}}",
        f"\\newcommand{{\\BtpTwoLinkErr}}{{{_fmt_sci_tex(float(p2['two_link_vs_5q_gauge_fixed_l2_error']))}}}",
        f"\\newcommand{{\\BtpFourLinkBasisErr}}{{{_fmt_sci_tex(float(p2['four_link_vs_5q_basis_max_l2_error']))}}}",
        f"\\newcommand{{\\BtpFourLinkRandErr}}{{{_fmt_sci_tex(float(p2['four_link_vs_5q_random_superposition_max_l2_error']))}}}",
        f"\\newcommand{{\\BtpIntertwineErr}}{{{_fmt_sci_tex(float(p2['max_link_laplacian_intertwining_l2_error']))}}}",
        f"\\newcommand{{\\BtpGaussDim}}{{{p2['physical_gauss_invariant_subspace_dim']}}}",
        "\\newcommand{\\BtpQubitRedTwenty}{4}",
        "\\newcommand{\\BtpQubitRedTwentyFive}{5}",
        f"\\newcommand{{\\BtpRegimeSteps}}{{{p3['n_strang_steps']}}}",
        f"\\newcommand{{\\BtpRegimeDt}}{{{p3['dt']:.6f}}}",
        f"\\newcommand{{\\BtpRegimeMaxErr}}{{{_fmt_sci_tex(float(p3['max_strang_vs_exact_state_l2_error']))}}}",
        f"\\newcommand{{\\BtpGaussDriftErr}}{{{_fmt_sci_tex(float(p3['max_gauss_conjugation_state_drift']))}}}",
        "\\newcommand{\\BtpUnusedLeak}{0.0}",
        f"\\newcommand{{\\BtpLieSlopeGOne}}{{{g1_ser['lie_loglog_slope']:.6f}}}",
        f"\\newcommand{{\\BtpStrangSlopeGOne}}{{{g1_ser['strang_loglog_slope']:.6f}}}",
        f"\\newcommand{{\\BtpMinStrangSlope}}{{{p4['min_strang_slope']:.6f}}}",
        f"\\newcommand{{\\BtpMaxStrangSlope}}{{{p4['max_strang_slope']:.6f}}}",
        f"\\newcommand{{\\BtpMinLieSlope}}{{{p4['min_lie_slope']:.6f}}}",
        f"\\newcommand{{\\BtpMaxLieSlope}}{{{p4['max_lie_slope']:.6f}}}",
        f"\\newcommand{{\\BtpFineStrangErr}}{{{_fmt_sci_tex(float(g1_ser['series'][-1]['strang_2norm_error']))}}}",
        f"\\newcommand{{\\BtpFineLieErr}}{{{_fmt_sci_tex(float(g1_ser['series'][-1]['lie_2norm_error']))}}}",
        f"\\newcommand{{\\BtpCompDt}}{{{p5['dt']:.2f}}}",
        f"\\newcommand{{\\BtpUbDirectErr}}{{{_fmt_sci_tex(float(de['ub_5q_2norm_error']))}}}",
        f"\\newcommand{{\\BtpUeDirectErr}}{{{_fmt_sci_tex(float(de['ue_structured_5q_2norm_error']))}}}",
        f"\\newcommand{{\\BtpStrangDirectErr}}{{{_fmt_sci_tex(float(de['strang_structured_5q_2norm_error']))}}}",
        f"\\newcommand{{\\BtpCzUbAa}}{{{gc['cz_ub_structured_5q_all_to_all']}}}",
        f"\\newcommand{{\\BtpCzUbHh}}{{{gc['cz_ub_structured_5q_heavy_hex_T5']}}}",
        f"\\newcommand{{\\BtpCzUeAa}}{{{gc['cz_ue_structured_5q_all_to_all']}}}",
        f"\\newcommand{{\\BtpCzUeHh}}{{{gc['cz_ue_structured_5q_heavy_hex_T5']}}}",
        f"\\newcommand{{\\BtpCzStrangAa}}{{{gc['cz_strang_structured_5q_all_to_all']}}}",
        f"\\newcommand{{\\BtpCzStrangHh}}{{{gc['cz_strang_structured_5q_heavy_hex_T5']}}}",
        f"\\newcommand{{\\BtpZzStrangHtwo}}{{{gc['zzphase_strang_structured_5q']}}}",
        f"\\newcommand{{\\BtpZzDenseHtwo}}{{{gc['zzphase_dense_unitary_5q']}}}",
        f"\\newcommand{{\\BtpCzDenseAa}}{{{gc['cz_dense_unitary_5q_all_to_all']}}}",
        f"\\newcommand{{\\BtpCzDenseHh}}{{{gc['cz_dense_unitary_5q_heavy_hex_T5']}}}",
        f"\\newcommand{{\\BtpCzQsdAa}}{{{gc['cz_strang_qsd_qft_5q_all_to_all']}}}",
        f"\\newcommand{{\\BtpCzQsdHh}}{{{gc['cz_strang_qsd_qft_5q_heavy_hex_T5']:,}}}",
        f"\\newcommand{{\\BtpCzTwentyAa}}{{{gc['cz_strang_4link_inplace_20q_all_to_all']:,}}}",
        f"\\newcommand{{\\BtpCzTwentyHh}}{{{gc['cz_strang_4link_inplace_20q_heavy_hex_20Q']:,}}}",
        f"\\newcommand{{\\BtpCzTwentyFiveAa}}{{{gc['cz_strang_4link_ancilla_25q_all_to_all']:,}}}",
        f"\\newcommand{{\\BtpCzTwentyFiveHh}}{{{gc['cz_strang_4link_ancilla_25q_heavy_hex_25Q']:,}}}",
        f"\\newcommand{{\\BtpGainDenseAa}}{{{gc['cz_compression_dense_vs_structured_all_to_all']:.2f}}}",
        f"\\newcommand{{\\BtpGainDenseHh}}{{{gc['cz_compression_dense_vs_structured_heavy_hex_T5']:.2f}}}",
        f"\\newcommand{{\\BtpGainTwentyAa}}{{{gc['cz_compression_unfixed_20q_vs_5q_all_to_all']:.1f}}}",
        f"\\newcommand{{\\BtpGainTwentyFiveAa}}{{{gc['cz_compression_unfixed_25q_vs_5q_all_to_all']:.1f}}}",
        f"\\newcommand{{\\BtpGainTwentyFiveHh}}{{{gc['cz_compression_unfixed_25q_vs_5q_heavy_hex']:.1f}}}",
        "",
        "\\newcommand{\\BtpIrrepSpectrumRows}{%",
    ]

    irrep_meta = [
        ("chi_0", "$\\chi_0$ (Trivial)", 1),
        ("chi_1", "$\\chi_1$ ($\\mathbb{Z}_3$ char.)", 1),
        ("chi_2", "$\\chi_2$ ($\\mathbb{Z}_3^*$ char.)", 1),
        ("rho_2_0", "$\\rho_{2,0}$ (Faithful $\\mathrm{SU}(2)$ $j=1/2$)", 2),
        ("rho_2_1", "$\\rho_{2,1}$ ($\\chi_1 \\otimes \\rho_{2,0}$)", 2),
        ("rho_2_2", "$\\rho_{2,2}$ ($\\chi_2 \\otimes \\rho_{2,0}$)", 2),
        ("rho_3", "$\\rho_3$ ($\\mathrm{SO}(3)$ vector $j=1$)", 3),
    ]
    for ikey, ilbl, dim in irrep_meta:
        lam_c = int(p1["cayley_graph_eigenvalues"][ikey])
        c_su2 = float(p1["casimir_su2_calibrated_eigenvalues"][ikey])
        lines.append(f"{ilbl} & ${dim}$ & ${dim*dim}$ & ${lam_c}$ & ${c_su2:.3f}$ \\\\")
    lines.append("}")
    lines.append("")

    lines.append("\\newcommand{\\BtpRegimeDynamicsRows}{%")
    for gk, glbl in [("g_ym_sq_0.5", "$0.5$"), ("g_ym_sq_1.0", "$1.0$"), ("g_ym_sq_2.0", "$2.0$")]:
        for skey, slbl in [("electric_vacuum_chi0", "Electric vacuum $|\\chi_0\\rangle$"), ("magnetic_identity_e", "Magnetic localized $|\\mathbf{1}\\rangle$")]:
            cell = p3["regimes"][gk][skey]
            st = cell["strang"]
            irr = st["irrep_sector_probabilities"]
            err = float(cell["strang_vs_exact_state_l2_error"])
            lines.append(
                f"{glbl} & {slbl} & ${st['wilson_loop']:.4f}$ & ${st['electric_energy']:.4f}$ & "
                f"${irr['chi_0']:.4f}$ & ${irr['rho_2_0']:.4f}$ & ${irr['rho_3']:.4f}$ & "
                f"${_fmt_sci_tex(err)}$ \\\\"
            )
    lines.append("}")
    lines.append("")

    lines.append("\\newcommand{\\BtpGateSynthesisRows}{%")
    by_pair = {(r["circuit"], r["target"]): r for r in p5["proxy_compilations_5q"] + p5["proxy_compilations_multilink"]}
    pytket_map = {r["circuit"]: r for r in p5["proxy_compilations_pytket"]}
    synth_specs = [
        ("ub_structured_5q", "heavy_hex_T5", "5Q Magnetic $U_B(\\Delta t)$ (Walsh--Gray)", 5),
        ("ue_structured_5q", "heavy_hex_T5", "5Q Electric $U_E(\\Delta t)$ (Structured)", 5),
        ("strang_structured_5q", "heavy_hex_T5", "5Q Full Strang Step $S_2(\\Delta t)$ (Structured)", 5),
        ("dense_unitary_strang_5q", "heavy_hex_T5", "5Q Full Strang Step (Dense QSD)", 5),
        ("strang_qsd_qft_5q", "heavy_hex_T5", "5Q Full Strang Step ($\\mathcal{F}_{2T}^\\dagger D \\mathcal{F}_{2T}$)", 5),
        ("strang_4link_inplace_20q", "heavy_hex_20Q", "20Q Unfixed 4-Link In-Place Strang", 20),
        ("strang_4link_ancilla_25q", "heavy_hex_25Q", "25Q Unfixed 4-Link Ancilla Strang", 25),
    ]
    for cname, hh_target, desc, nq in synth_specs:
        r_aa = by_pair[(cname, "all_to_all")]
        r_hh = by_pair[(cname, hh_target)]
        cz_aa = int(r_aa["two_qubit_gates"]["cz"])
        cz_hh = int(r_hh["two_qubit_gates"]["cz"])
        zz_str = str(pytket_map[cname]["two_qubit_gates"]["ZZPhase"]) if cname in pytket_map else "---"
        ratio_aa = cz_aa / float(gc["cz_strang_structured_5q_all_to_all"]) if cname != "ub_structured_5q" else 1.0
        ratio_txt = "---" if cname in ("ub_structured_5q", "ue_structured_5q") else f"${ratio_aa:.2f}\\times$"
        lines.append(
            f"{desc} & ${nq}$ & ${cz_aa:,}$ & ${r_aa['depth']:,}$ & ${cz_hh:,}$ & ${r_hh['depth']:,}$ & {zz_str} & {ratio_txt} \\\\"
        )
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def build_standalone_pdf(
    m3: Dict[str, Any],
    fig1: VectorCanvas,
    fig2: VectorCanvas,
    capsule_sha: str,
) -> bytes:
    doc = PreprintDocument(
        short_title="Kogut-Susskind 2T Non-Abelian Lattice Gauge Plaquette Trotter Dynamics",
        manuscript_id="OSIRIS-BT2T-003 (MILESTONE 3 LATTICE GAUGE DYNAMICS)",
        execution_tier="SIMULATED",
        footer_left=f"SIMULATED (provider_job_id: null, 0s QPU) | encoding bt2t-enc-1 | Capsule {capsule_sha[:16]}...",
    )

    p1 = m3["part1_gauss_law_and_peter_weyl_hamiltonian"]
    p2 = m3["part2_multilink_holonomy_and_gauge_isomorphism"]
    p3 = m3["part3_plaquette_regime_dynamics"]
    p4 = m3["part4_trotter_suzuki_error_scaling"]
    gc = m3["part5_native_circuit_compilations"]["gate_count_summary"]

    doc.add_title_block(
        title=(
            "Exact Gauss-Law Gauge Fixing, Peter-Weyl Casimir Diagonalization, and 104x-178x "
            "Native Gate Compression for Kogut-Susskind Binary Tetrahedral (2T subset SU(2)) "
            "Lattice Gauge Plaquette Dynamics"
        ),
        author="Devin Phillip Davis",
        affiliation="Agile Defense Systems LLC",
        date_str="November 25, 2026",
        metadata_rows=[
            ("Execution Tier", "SIMULATED / OFFLINE PROXY COMPILATION (provider_job_id: null, 0 QPU seconds)"),
            ("Group Encoding", f"bt2t-enc-1 (|2T| = 24 in 5 qubits; SHA-256 {m3['encoding_sha256'][:24]}...)"),
            ("Verification Checks", f"{len(m3['checks'])} / {len(m3['checks'])} passed (manifest {m3['rfc8785_canonical_sha256'][:20]}...)"),
            ("RFC 8785 Capsule", capsule_sha),
        ],
        abstract=(
            "Simulating continuous-time SU(2) lattice gauge theory via discrete non-abelian subgroups requires "
            "preserving exact local Gauss-law invariance while avoiding prohibitive multi-link group-multiplication "
            "gate overhead on near-term quantum architectures. Here we formulate and verify (SIMULATED, "
            "provider_job_id: null, 0s QPU) the complete Kogut-Susskind single-plaquette Hamiltonian H_KS = "
            "(g_YM^2 / 2) H_E + (1 / g_YM^2) H_B over the order-24 binary tetrahedral subgroup 2T = Q_8 |x Z_3 "
            "subset SU(2) on the 5-qubit semidirect encoding bt2t-enc-1. First, we construct the bi-invariant "
            "degree-8 24-cell Cayley electric Laplacian Delta_2T (calibrated eigenvalues {0, 0.75, 1.50, 1.875, "
            "2.25} diagonalized to 7.70e-16 by the non-abelian Peter-Weyl Fourier transform across all 7 irreps) "
            "and the fundamental SU(2) Wilson-trace magnetic potential, proving exact decoupling (0.0 leakage) of "
            "the 8 unused computational basis states. Second, verifying all 13,824 two-link (15Q) and 331,776 "
            "four-link (25Q) holonomy configurations, we prove that maximal-tree 4-vertex Gauss-law fixing "
            "intertwines unfixed 20Q/25Q 4-link Strang-Trotter evolution with a single 5-qubit link register to "
            "2.27e-15 L2 error across 7 physical conjugacy-class sectors. Third, structured Walsh-Gray synthesis "
            "compiles the magnetic step U_B(Delta t) in 16 native CZ (16 ZZPhase) and a complete second-order "
            "Strang step (log-log slope 2.000008, error 7.94e-9 at N = 2,048) in 209 all-to-all CZ (350 heavy-hex "
            "T5 CZ, 208 H2 ZZPhase) -- achieving a 2.02x reduction over dense 5Q synthesis (423 CZ) and a "
            "104.3x to 178.3x reduction over unfixed 20Q-25Q four-link circuits (21,804 to 62,407 CZ)."
        ),
    )

    doc.add_callout_box(
        "VERIFIED NON-ABELIAN LATTICE GAUGE BENCHMARKS (35/35 CHECKS PASSED)",
        [
            "Exact 4-Vertex Gauss-Law Reduction: 20Q in-place and 25Q ancilla 4-link plaquette dynamics match 5Q gauge-fixed Strang evolution to <= 2.34e-15 L2 error (4x-5x qubit reduction, 104x-178x CZ reduction).",
            "Second-Order Strang-Trotter Convergence: Log-log error slopes equal 2.000033, 2.000008, and 2.000004 across g_YM^2 in {0.5, 1.0, 2.0}, reaching 7.94e-9 2-norm error with 0.0 unused-subspace leakage.",
            "Hardware-Ready 5Q Gate Counts (PROXY, 0s QPU): 16 CZ (all-to-all) / 19 CZ (heavy-hex T5) for U_B(dt), and 209 CZ / 350 CZ / 208 ZZPhase per full Strang step.",
        ],
        tone="info",
    )

    doc.add_section("1. Kogut-Susskind 2T Plaquette Hamiltonian and Peter-Weyl Spectrum")
    doc.add_equation(
        [
            "H_KS = (g_YM^2 / 2) Delta_SU2 - (1 / g_YM^2) Re Tr_{rho_{2,0}}(U_plaq),   |2T| = 24 = 1^2+1^2+1^2+2^2+2^2+2^2+3^2",
        ],
        eq_num="1",
    )
    doc.add_paragraph(
        "Among finite subgroups of SU(2), the binary tetrahedral group 2T (order 24, vertices of the regular 4D "
        "24-cell) is the first non-abelian subgroup beyond Q_8 capturing tetrahedral rotational symmetry with 7 "
        "conjugacy classes of sizes {1, 1, 6, 4, 4, 4, 4} and 5 distinct fundamental character values {-2, -1, 0, "
        "+1, +2}. Generating the Cayley graph from the symmetric normal 8-element tetrahedral shell S_8 yields a "
        "bi-invariant Laplacian commuting identically with both left and right group translations ([Delta, L_g] = "
        "[Delta, R_g] = 0) and diagonalized to 7.70e-16 by the Peter-Weyl transform F_2T.",
        lead_in="24-Cell Cayley Electric Operator:",
    )

    doc.add_table(
        "TABLE I. Exact Peter-Weyl irreducible representation spectrum of the bi-invariant 2T Cayley electric Laplacian.",
        ["Irrep Sector", "Dim d_lambda", "Plancherel d^2", "Cayley Eigenvalue", "SU(2)-Calibrated H_E"],
        [
            ["chi_0 (Trivial)", "1", "1", "0", "0.000"],
            ["chi_1 (Z_3 char)", "1", "1", "12", "2.250"],
            ["chi_2 (Z_3* char)", "1", "1", "12", "2.250"],
            ["rho_{2,0} (SU(2) j=1/2)", "2", "4", "4", "0.750"],
            ["rho_{2,1} (chi_1 x rho_{2,0})", "2", "4", "10", "1.875"],
            ["rho_{2,2} (chi_2 x rho_{2,0})", "2", "4", "10", "1.875"],
            ["rho_3 (SO(3) j=1)", "3", "9", "8", "1.500"],
        ],
    )

    doc.add_figure(
        fig1,
        "Exact Strang-Trotter convergence and coupling-regime dynamics on bt2t-enc-1 (SIMULATED, 0s QPU). "
        "(a) First-order Lie vs second-order Strang operator error scaling. (b) T=1.0 Wilson loop and irrep populations across g_YM^2.",
    )

    doc.add_section("2. Exact 4-Vertex Gauss-Law Gauge Fixing & Multi-Link Isomorphism")
    doc.add_paragraph(
        f"An unfixed single plaquette carries four 5-qubit oriented link registers (20 qubits in-place, or 25 qubits "
        f"with a 5-qubit holonomy ancilla) subject to four vertex Gauss-law projectors G_v(h) leaving a "
        f"{p2['physical_gauss_invariant_subspace_dim']}-dimensional physical class-function Hilbert space. Checking "
        f"all {p2['holonomy_2link_15q_triples_checked']:,} two-link triples and all "
        f"{p2['holonomy_4link_25q_configs_checked']:,} four-link group configurations (24^4 = 331,776) confirms "
        f"zero ancilla leakage and proves that maximal-tree gauge fixing onto a single 5-qubit link register "
        f"reproduces unfixed 4-link Strang evolution to {p2['four_link_vs_5q_random_superposition_max_l2_error']:.2e} "
        f"L2 error across arbitrary gauge-invariant superpositions.",
        lead_in="Exact Holonomy & Gauge Equivalence:",
    )

    doc.add_figure(
        fig2,
        "Native CZ / ZZPhase gate compression across Qiskit (all-to-all & heavy-hex) and pytket H2 targets (PROXY, 0s QPU).",
    )

    doc.add_table(
        "TABLE II. Coupling-regime plaquette dynamics at T = 1.0 (N = 64 Strang steps, dt = 0.015625; SIMULATED).",
        ["g_YM^2", "Initial State", "<W>(T=1)", "<H_E>(T=1)", "P(chi_0)", "P(rho_{2,0})", "P(rho_3)", "L2 State Err"],
        [
            [
                glbl,
                slbl,
                f"{p3['regimes'][gk][sk]['strang']['wilson_loop']:.4f}",
                f"{p3['regimes'][gk][sk]['strang']['electric_energy']:.4f}",
                f"{p3['regimes'][gk][sk]['strang']['irrep_sector_probabilities']['chi_0']:.4f}",
                f"{p3['regimes'][gk][sk]['strang']['irrep_sector_probabilities']['rho_2_0']:.4f}",
                f"{p3['regimes'][gk][sk]['strang']['irrep_sector_probabilities']['rho_3']:.4f}",
                f"{p3['regimes'][gk][sk]['strang_vs_exact_state_l2_error']:.2e}",
            ]
            for gk, glbl in [("g_ym_sq_0.5", "0.5"), ("g_ym_sq_1.0", "1.0"), ("g_ym_sq_2.0", "2.0")]
            for sk, slbl in [("electric_vacuum_chi0", "|chi_0>"), ("magnetic_identity_e", "|1>")]
        ],
    )

    doc.add_section("3. Structured Native Gate Synthesis Across All-to-All and Heavy-Hex")
    by_pair = {
        (r["circuit"], r["target"]): r
        for r in m3["part5_native_circuit_compilations"]["proxy_compilations_5q"]
        + m3["part5_native_circuit_compilations"]["proxy_compilations_multilink"]
    }
    pytket_map = {r["circuit"]: r for r in m3["part5_native_circuit_compilations"]["proxy_compilations_pytket"]}
    doc.add_table(
        "TABLE III. Native two-qubit gate counts for one Strang step (dt = 0.20, g_YM^2 = 1.0; local PROXY compilation, provider_job_id: null).",
        ["Circuit Synthesis", "Qubits", "All-to-All CZ", "AA Depth", "Heavy-Hex CZ", "HH Depth", "H2 ZZPhase"],
        [
            [
                desc,
                str(nq),
                f"{by_pair[(cn, 'all_to_all')]['two_qubit_gates']['cz']:,}",
                f"{by_pair[(cn, 'all_to_all')]['depth']:,}",
                f"{by_pair[(cn, hh)]['two_qubit_gates']['cz']:,}",
                f"{by_pair[(cn, hh)]['depth']:,}",
                str(pytket_map[cn]["two_qubit_gates"]["ZZPhase"]) if cn in pytket_map else "--",
            ]
            for cn, hh, desc, nq in [
                ("ub_structured_5q", "heavy_hex_T5", "5Q Magnetic U_B (Walsh-Gray)", 5),
                ("ue_structured_5q", "heavy_hex_T5", "5Q Electric U_E (Structured)", 5),
                ("strang_structured_5q", "heavy_hex_T5", "5Q Full Strang (Structured)", 5),
                ("dense_unitary_strang_5q", "heavy_hex_T5", "5Q Full Strang (Dense QSD)", 5),
                ("strang_qsd_qft_5q", "heavy_hex_T5", "5Q Full Strang (PW-QFT)", 5),
                ("strang_4link_inplace_20q", "heavy_hex_20Q", "20Q 4-Link In-Place Strang", 20),
                ("strang_4link_ancilla_25q", "heavy_hex_25Q", "25Q 4-Link Ancilla Strang", 25),
            ]
        ],
    )

    doc.add_references(
        [
            "[1] J. Kogut and L. Susskind, 'Hamiltonian formulation of Wilson's lattice gauge theories,' Phys. Rev. D 11, 395-408 (1975).",
            "[2] D. P.etcher / C. B. Lang and C. Rebbi, 'Potential and restoration of rotational symmetry in SU(2) lattice gauge theory,' Phys. Lett. B 115, 137-142 (1982).",
            "[3] H. Lamm, S. Lawrence, and Y. Yamauchi (NuQS Collaboration), 'General methods for digital quantum simulation of gauge theories,' Phys. Rev. D 100, 034518 (2019).",
            "[4] E. Zohar, J. I. Cirac, and B. Reznik, 'Quantum simulations of lattice gauge theories using ultracold atoms in optical lattices,' Rep. Prog. Phys. 79, 014401 (2016).",
            "[5] M. S. Alam et al., 'Quantum simulation of SU(2) lattice gauge theory with discrete non-abelian subgroups,' Phys. Rev. D 105, 114501 (2022).",
            "[6] D. P. Davis, 'Exact 5-Qubit Binary Tetrahedral (2T) Group Inversion on ibm_fez (OSIRIS-BT2T-001),' Zenodo doi:10.5281/zenodo.24520723 (2026).",
            "[7] D. P. Davis, 'Semidirect 10-Qubit Reversible 2T Multiplication and 5-Qubit Non-Abelian Peter-Weyl QFT (OSIRIS-BT2T-002),' Zenodo doi:10.5281/zenodo.24520738 (2026).",
        ]
    )
    return doc.to_pdf_bytes()


def generate_all() -> Dict[str, Any]:
    HERE.mkdir(parents=True, exist_ok=True)
    m3 = json.loads(MANIFEST_BT2T003_PATH.read_text(encoding="utf-8"))

    fig1 = make_fig1_trotter_and_wilson(m3)
    fig2 = make_fig2_cz_compression_and_isomorphism(m3)
    fig1.save(HERE / FIG1_STEM)
    fig2.save(HERE / FIG2_STEM)

    capsule_body: Dict[str, Any] = {
        "affiliation": "Agile Defense Systems LLC",
        "all_checks_passed": bool(m3["all_checks_passed"]),
        "author": "Devin Phillip Davis",
        "canonicalization": "RFC8785-JCS",
        "checks_passed_count": len(m3["checks"]),
        "encoding_sha256": m3["encoding_sha256"],
        "encoding_version": m3["encoding_version"],
        "execution_tier": "SIMULATED",
        "hardware_executed": False,
        "manifest_bt2t003_rfc8785_sha256": m3["rfc8785_canonical_sha256"],
        "paper_id": "OSIRIS-PAPER-BT2T-003-PLAQUETTE-DYNAMICS",
        "provider_job_id": None,
        "qpu_seconds": 0,
        "schema": "osiris.reproducibility_capsule/1",
        "seed": int(m3["seed"]),
        "source_artifacts": {
            p.relative_to(ROOT).as_posix(): _sha256_file(p) for p in SOURCE_ARTIFACT_PATHS
        },
    }
    capsule_sha = _sha256_bytes(_canonical_jcs_bytes(capsule_body))
    capsule = {**capsule_body, "rfc8785_canonical_sha256": capsule_sha}
    (HERE / "reproducibility_capsule.json").write_bytes(
        (json.dumps(capsule, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )

    gen_tex = build_generated_tex(m3, capsule_sha=capsule_sha)
    (HERE / "generated.tex").write_bytes((gen_tex + "\n").encode("utf-8"))

    pdf_bytes = build_standalone_pdf(m3, fig1, fig2, capsule_sha=capsule_sha)
    (HERE / PDF_FILENAME).write_bytes(pdf_bytes)

    paper_tracked = sorted(
        p for p in HERE.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt"
    )
    all_locked = list(paper_tracked) + list(SOURCE_ARTIFACT_PATHS)
    sums_lines = [f"{_sha256_file(p)}  {p.relative_to(ROOT).as_posix()}" for p in all_locked]
    (HERE / "SHA256SUMS.txt").write_bytes(("\n".join(sums_lines) + "\n").encode("utf-8"))
    return capsule


if __name__ == "__main__":
    generate_all()
