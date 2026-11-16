#!/usr/bin/env python3
"""
generate_paper_assets_v2_1.py — Non-mutating Version 2.1 vector SVG + PDF 1.4 generator
for OSIRIS-HP-004 / HP-004B (`ibm_marrakesh` + `ibm_fez`).

Preserves the frozen Zenodo `10.5281/zenodo.24287505` v2.0 directory untouched while
generating the v2.1 manuscript revision (`hp004_joint_stabilizer_heron_r2_v2_1.pdf`,
SVG/PDF figures, and `SHA256SUMS_v2_1.txt`) incorporating:
  1. Primary citation B. M. Terhal & K. G. H. Vollbrecht, Phys. Rev. Lett. 85, 2625 (2000),
     doi:10.1103/PhysRevLett.85.2625 alongside Vollbrecht & Werner, Phys. Rev. A 64, 062307 (2001).
  2. Exact 1-ebit threshold root F* = 0.683900 (0.683900317, correcting v2.0's 0.683910 typo).
  3. Explicit schema documentation for legacy JSON field `z_above_1ebit_eff` (Z above 0.5000 SN<=2 bound).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

V21_DIR = Path(__file__).resolve().parent
REPO_ROOT = V21_DIR.parent.parent
V20_DIR = REPO_ROOT / "papers" / "hp004_joint_stabilizer_heron_r2"
HP004_DIR = REPO_ROOT / "experiments" / "hp004"
FEZ_DIR = REPO_ROOT / "experiments" / "hp004b_fez"

sys.path.insert(0, str(V20_DIR))
import generate_paper_assets as v20  # noqa: E402

F_STAR_EXACT = 0.683900317


def make_fig1_v2_1(res_m: Dict, res_f: Dict) -> v20.VectorCanvas:
    c = v20.VectorCanvas(540, 345)
    c.text(270, 20, "Dual-Processor 9-Basis 4Q Joint Bell Overlap (ibm_marrakesh & ibm_fez)", size=10.8, bold=True, anchor="middle")
    c.text(270, 34, "572 circuits x 1,024 shots = 585,728 raw shots | routing_method='none' | Zero postselection", size=8.3, color="#4b5563", anchor="middle")

    x0, y0, pw, ph = 62.0, 48.0, 452.0, 232.0
    y_half = y0 + ph * (1.0 - 0.5)
    y_ef = y0 + ph * (1.0 - F_STAR_EXACT)
    c.rect(x0, y_half, pw, y0 + ph - y_half, fill="#fef2f2")
    c.rect(x0, y_ef, pw, y_half - y_ef, fill="#fffbeb")
    c.rect(x0, y0, pw, y_ef - y0, fill="#f0fdf4")
    c.rect(x0, y0, pw, ph, fill=None, stroke="#374151", lw=1.0)

    rungs = [
        (1.0000, "1.0000 (16/16 Ideal 2-Bell Target)", "#15803d"),
        (F_STAR_EXACT, "0.683900 (Terhal-Vollbrecht E_F > 1.0000 Ebit)", "#0369a1"),
        (0.5000, "0.5000 (Terhal-Horodecki SN >= 3 Witness)", "#b91c1c"),
        (0.2500, "0.2500 (4/16 Separable Ceiling)", "#92400e"),
        (0.1250, "0.1250 (2/16 1-Photon Ceiling)", "#6b7280"),
        (0.0625, "0.0625 (1/16 Mixed Floor)", "#9ca3af"),
    ]
    for val, lbl, col in rungs:
        yy = y0 + ph * (1.0 - val)
        c.line(x0, yy, x0 + pw, yy, color=col, lw=1.2 if val in (0.5, F_STAR_EXACT) else 0.75, dashed=True)
        c.text(x0 - 5, yy + 3, f"{val:.2f}", size=7.6, color="#111827", anchor="end", mono=True)
        if val == 0.1250:
            c.text(x0 + pw * 0.60, yy - 2.5, "2/16 Ceiling", size=6.4, color=col, anchor="middle")
        elif val == 0.0625:
            c.text(x0 + pw * 0.60, yy - 2.5, "1/16 Floor", size=6.4, color=col, anchor="middle")
        else:
            c.text(x0 + pw - 4, yy - 2.5, lbl, size=6.8, color=col, bold=(val in (0.5, F_STAR_EXACT)), anchor="end")

    arms = [
        ("Arm A", "Arm A\n(Inverse 38 CZ)"),
        ("Arm B", "Arm B\n(Restricted 32 CZ)"),
        ("Arm B-Ctrl", "Arm B-Ctrl\n(|0000>_BE 30 CZ)"),
        ("Arm B-Def", "Arm B-Def\n(1-Photon 25 CZ)"),
        ("Arm B-Held", "Arm B-Held\n(Wrong U2 35 CZ)"),
    ]
    slot = pw / len(arms)
    for idx, (key, label) in enumerate(arms):
        sm = res_m["arm_summaries"][key]
        sf = res_f["arm_summaries"][key]
        fm, sem = float(sm["f_joint"]), float(sm["f_joint_se_eff"])
        ff, sef = float(sf["f_joint"]), float(sf["f_joint_se_eff"])
        cx = x0 + slot * (idx + 0.5)
        bw = 23.0

        bym = y0 + ph * (1.0 - fm)
        c.rect(cx - bw - 2, bym, bw, max(1.5, (y0 + ph) - bym), fill="#3b82f6", stroke="#111827", lw=0.8)
        etm = y0 + ph * (1.0 - min(1.0, fm + 5.0 * sem))
        ebm = y0 + ph * (1.0 - max(0.0, fm - 5.0 * sem))
        c.line(cx - 0.5 * bw - 2, etm, cx - 0.5 * bw - 2, ebm, color="#111827", lw=1.1)

        byf = y0 + ph * (1.0 - ff)
        c.rect(cx + 2, byf, bw, max(1.5, (y0 + ph) - byf), fill="#059669", stroke="#111827", lw=0.8)
        etf = y0 + ph * (1.0 - min(1.0, ff + 5.0 * sef))
        ebf = y0 + ph * (1.0 - max(0.0, ff - 5.0 * sef))
        c.line(cx + 0.5 * bw + 2, etf, cx + 0.5 * bw + 2, ebf, color="#111827", lw=1.1)

        c.text(cx - 0.5 * bw - 3, bym - 5, f"{fm:.3f}", size=6.8, bold=True, color="#1d4ed8", anchor="middle", mono=True)
        c.text(cx + 0.5 * bw + 3, byf - 5, f"{ff:.3f}", size=6.8, bold=True, color="#047857", anchor="middle", mono=True)
        p1, p2 = label.split("\n")
        c.text(cx, y0 + ph + 14, p1, size=8.2, bold=True, anchor="middle")
        c.text(cx, y0 + ph + 24, p2, size=7.2, color="#4b5563", anchor="middle")

    c.rect(66, 320, 12, 8, fill="#3b82f6", stroke="#111827", lw=0.7)
    c.text(82, 327, "ibm_marrakesh (Job dbti1fbsbsbs73cb2vu0, Raw +/- 5 SE)", size=7.7)
    c.rect(314, 320, 12, 8, fill="#059669", stroke="#111827", lw=0.7)
    c.text(330, 327, "ibm_fez (Job dbtj61bnlcms73ajc380, Raw +/- 5 SE)", size=7.7)
    return c


def build_preprint_pdf_v2_1(
    res_m: Dict,
    res_f: Dict,
    fig1: v20.VectorCanvas,
    fig2: v20.VectorCanvas,
    fig3: v20.VectorCanvas,
    fig4: v20.VectorCanvas,
) -> bytes:
    W, H = 612.0, 792.0

    def wrap_paragraph(text: str, max_chars: int = 94) -> List[str]:
        words = text.split()
        lines: List[str] = []
        cur = ""
        for w in words:
            if not cur:
                cur = w
            elif len(cur) + 1 + len(w) <= max_chars:
                cur += " " + w
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    def draw_lines(cmds: List[str], x: float, y: float, lines: Sequence[str], font: str = "/F4", size: float = 10.0, leading: float = 13.2) -> float:
        for ln in lines:
            esc = v20._pdf_escape(ln)
            cmds.append(f"BT {font} {size:.1f} Tf 0.07 0.09 0.15 rg {x:.2f} {H - y:.2f} Td ({esc}) Tj ET")
            y += leading
        return y

    p1: List[str] = ["1 1 1 rg 0 0 612 792 re f"]
    y = 46.0
    y = draw_lines(
        p1, 52, y,
        [
            "Dual-Processor Certification of Schmidt Number >= 3 and >1-Ebit Entanglement",
            "of Formation in Hayden-Preskill Recovery via Layout-Locked Nine-Basis",
            "Stabilizer Tomography on Heavy-Hex Superconducting Processors (v2.1)",
        ],
        font="/F5", size=13.2, leading=16.8,
    )
    y += 4
    y = draw_lines(
        p1, 52, y,
        [
            "Devin Phillip Davis  |  Agile Defense Systems LLC  |  November 16, 2026 (Version 2.1; v2.0 DOI 10.5281/zenodo.24287505)",
            "OSIRIS-HP-004 (ibm_marrakesh Job dbti1fbsbsbs73cb2vu0) & OSIRIS-HP-004B (ibm_fez Job dbtj61bnlcms73ajc380)",
            f"572 Circuits x 1,024 Shots = 585,728 Raw Shots  |  Marrakesh QPY: {res_m['qpy_sha256'][:24]}...",
            f"Fez Pre-Registered QPY (commit 0c5a5a39): {res_f['qpy_sha256']}",
        ],
        font="/F3", size=7.4, leading=10.5,
    )
    y += 7
    p1.append(f"0.7 w 0.6 0.6 0.6 RG 52 {H - y:.2f} m 560 {H - y:.2f} l S")
    y += 12
    y = draw_lines(p1, 52, y, ["ABSTRACT"], font="/F5", size=10.0, leading=13.0)
    abstract = (
        "Multi-qubit Hayden-Preskill recovery experiments face three distinct certification pitfalls: (1) 3-basis "
        "marginal tomography (XXXX, YYYY, ZZZZ) cannot exclude a Schmidt-number-2 mixture achieving F1 = F2 = 0.7500 "
        "and F1*F2 = 0.5625 with joint Bell overlap F_joint = 0.5000; (2) unconstrained compilation can insert SWAPs "
        "across quarantined horizon qubits; and (3) exceeding the F_joint = 0.5000 Terhal-Horodecki ceiling certifies "
        "Schmidt number SN(rho) >= 3 rather than Entanglement of Formation E_F(rho) > 1 ebit, whose exact d=4 "
        "Terhal-Vollbrecht / Vollbrecht-Werner convex-roof threshold is F* = 0.683900 (0.683900317). Here we report two "
        "prospectively pre-registered, layout-locked 8-qubit experiments on independent 156-qubit IBM Heron r2 processors: "
        "OSIRIS-HP-004 on ibm_marrakesh (Job dbti1fbsbsbs73cb2vu0) and OSIRIS-HP-004B on ibm_fez (Job dbtj61bnlcms73ajc380), "
        "totaling 585,728 raw shots (resilience_level=0, zero postselection). On ibm_marrakesh, raw Arm B (32 CZs, zero "
        "gates on horizon remnants or references during decoding) achieves F_joint = 0.6334 +/- 0.0026 (+50.64 sigma above "
        "0.5000, certifying SN >= 3 and E_F >= 0.8451 ebits, while F1+F2-1 = 0.4875 < 0.50). On lower-noise ibm_fez, "
        "raw Arm B achieves F_joint = 0.7307 +/- 0.0027 (6-block 95% t-CI [0.7237, 0.7377]), simultaneously certifying "
        "SN >= 3 (+84.45 sigma) AND E_F(rho) >= 1.1465 +/- 0.0086 ebits > 1.0000 ebit (+17.13 sigma above F* = 0.683900)."
    )
    y = draw_lines(p1, 52, y, wrap_paragraph(abstract, 96), font="/F4", size=9.2, leading=12.1)
    y += 9
    y = draw_lines(p1, 52, y, ["1. SCHMIDT NUMBER >= 3 WITNESS VS. ENTANGLEMENT OF FORMATION > 1 EBIT (V2.1 REVISION)"], font="/F5", size=9.9, leading=13.2)
    sec1 = (
        "Proposition 1 (Terhal & Horodecki, PRA 61, 040301, 2000): Because |Psi_4> = |Phi+>_1 (x) |Phi+>_2 has four "
        "equal Schmidt coefficients 1/2 across (R2 R1):(M1' M2'), every state of Schmidt number SN(rho) <= 2 obeys "
        "F_joint(rho) <= 2/4 = 0.5000. Proposition 2 (Terhal & Vollbrecht, PRL 85, 2625, 2000; Vollbrecht & Werner, "
        "PRA 64, 062307, 2001): For F in [0.25, 0.75], the exact d=4 isotropic convex-roof lower bound is "
        "E_F(rho) >= H_2(p0, (1-p0)/3, (1-p0)/3, (1-p0)/3) with p0 = ((sqrt(F)+sqrt(3(1-F)))/2)^2, crossing 1.0000 ebit "
        "at exact root F* = 0.683900 (0.683900317; refining v2.0's fifth-decimal 0.683910). Version 2.1 adds the primary "
        "Terhal-Vollbrecht citation [5] and notes that legacy JSON field z_above_1ebit_eff in Zenodo 24287505 computes "
        "(F_joint - 0.5000)/SE_eff (+50.64 on Marrakesh, +84.45 on Fez), whereas Z vs F* = 0.683900 is -19.16 / +17.13."
    )
    y = draw_lines(p1, 52, y, wrap_paragraph(sec1, 96), font="/F4", size=9.2, leading=12.1)

    p1.append("q 0.90 0 0 0.90 62 34 cm")
    p1.extend(fig3.pdf)
    p1.append("Q")

    p2: List[str] = ["1 1 1 rg 0 0 612 792 re f"]
    p2.append("q 0.92 0 0 0.92 58 532 cm")
    p2.extend(fig4.pdf)
    p2.append("Q")
    p2.append("q 0.92 0 0 0.92 58 175 cm")
    p2.extend(fig1.pdf)
    p2.append("Q")
    draw_lines(
        p2, 52, 636,
        wrap_paragraph(
            "Figure 1 (top): Isomorphic 8-transmon tree on ibm_fez [2, 24, 4, 3, 16, 23, 22, 21] (mean CZ error 0.179%, "
            "4Q SPAM diagonal 97.31%). Figure 2 (bottom): Dual-processor 9-basis joint Bell overlap across all 5 arms "
            "(10/10 pre-registered hypotheses passed across 585,728 total shots; F* = 0.683900).",
            96,
        ),
        font="/F6", size=9.0, leading=12.0,
    )

    p3: List[str] = ["1 1 1 rg 0 0 612 792 re f"]
    p3.append("q 0.92 0 0 0.92 58 478 cm")
    p3.extend(fig2.pdf)
    p3.append("Q")

    ty = 335.0
    ty = draw_lines(p3, 46, ty, ["TABLE I. Dual-Processor Hardware Results (OSIRIS-HP-004 ibm_marrakesh & OSIRIS-HP-004B ibm_fez)"], font="/F5", size=9.2, leading=13.0)
    rows = [
        "Processor      Arm (CZs)    Raw F1     Raw F2     F1+F2-1   Raw F_joint +/- SE   95% Boot CI       Z vs 0.50   E_F(ebit)  SPAM   Gate",
        "-------------------------------------------------------------------------------------------------------------------------------------",
    ]
    for proc_lbl, res in [("ibm_marrakesh", res_m), ("ibm_fez      ", res_f)]:
        for arm, tag in [("Arm A", "H1 PASS"), ("Arm B", "H2 PASS"), ("Arm B-Ctrl", "H3 PASS"), ("Arm B-Def", "H4 PASS"), ("Arm B-Held", "H5 PASS")]:
            s = res["arm_summaries"][arm]
            cz = {"Arm A": 38, "Arm B": 32, "Arm B-Ctrl": 30, "Arm B-Def": 25, "Arm B-Held": 35}[arm]
            ci = s["f_joint_shot_boot_95ci"]
            sec = s["secondary_spam_sensitivity"]["f_joint_spam_mitigated"]
            ef = v20.ef_lower_bound_d4(float(s["f_joint"]))
            short = arm.replace("Arm ", "")
            rows.append(
                f"{proc_lbl}  {short:7s}({cz:2d})  {s['f1_marginal']:.4f}     {s['f2_marginal']:.4f}     {s['f_union_lower_bound']:+.4f}   "
                f"{s['f_joint']:.4f}+/-{s['f_joint_se_eff']:.4f}   [{ci[0]:.4f},{ci[1]:.4f}]  {s['z_above_1ebit_eff']:+7.2f}s   >={ef:.4f}   {sec:.4f} {tag}"
            )
        rows.append("-------------------------------------------------------------------------------------------------------------------------------------")
    draw_lines(p3, 42, ty, rows, font="/F3", size=6.2, leading=10.5)

    p4: List[str] = ["1 1 1 rg 0 0 612 792 re f"]
    y4 = 52.0
    y4 = draw_lines(p4, 52, y4, ["2. CROSS-PROCESSOR REPRODUCIBILITY, KRAUS COMPARISON, AND SCOPE BOUNDARIES"], font="/F5", size=10.4, leading=14.0)
    bm = res_m["arm_summaries"]["Arm B"]
    bf = res_f["arm_summaries"]["Arm B"]
    bm_s = ", ".join(f"{v:.4f}" for v in bm["block_f_joints"])
    bf_s = ", ".join(f"{v:.4f}" for v in bf["block_f_joints"])
    sec2 = (
        f"1. Pre-Registration & QPU Accounting: OSIRIS-HP-004 (ibm_marrakesh, commit fa9ef46d at 15:26:18Z < job created "
        f"15:26:53Z, 81 s = 1.3355 RU) measured Arm B blocks [{bm_s}], giving F_joint = 0.633423 +/- 0.002634 (95% "
        f"Student-t CI [0.6267, 0.6402], E_F >= 0.8451 ebits, 95% LCL >= 0.8248 ebits). OSIRIS-HP-004B (ibm_fez, commit "
        f"0c5a5a39 staged at 16:01:38Z < job created 16:44:53Z, 79 s = 1.3090 RU) measured Arm B blocks [{bf_s}], giving "
        f"F_joint = 0.730713 +/- 0.002732 (95% Student-t CI [0.7237, 0.7377], E_F >= 1.1465 ebits, 95% LCL >= 1.1244 ebits)."
    )
    y4 = draw_lines(p4, 52, y4, wrap_paragraph(sec2, 96), font="/F4", size=9.5, leading=13.0)
    y4 += 7
    lim = (
        "2. Scientific Scope, Witness Caveats & Schema Invariant: (a) Both U1 (seed=111, 7 CZs) and V_dec (10 CZs) are "
        "exact 4-qubit Clifford circuits efficiently simulable via Gottesman-Knill; these experiments certify physical "
        "entanglement dimensionality (SN >= 3) and convex-roof entanglement of formation (E_F >= 1.1465 ebits on ibm_fez) "
        "conditional on target-state mapping and faithful Pauli basis settings -- NOT a direct von Neumann entropy "
        "measurement, new physical law, or super-polynomial quantum advantage. (b) Frozen Zenodo 10.5281/zenodo.24287505 "
        "files remain untouched; its JSON field z_above_1ebit_eff denotes Z above 0.5000 (SN <= 2 ceiling)."
    )
    y4 = draw_lines(p4, 52, y4, wrap_paragraph(lim, 96), font="/F4", size=9.5, leading=13.0)
    y4 += 10
    y4 = draw_lines(p4, 52, y4, ["3. DETERMINISTIC REPRODUCTION & REFERENCES"], font="/F5", size=10.4, leading=14.0)
    draw_lines(
        p4, 52, y4,
        [
            "All 572 raw 4-bit histograms and frozen QPY circuits are verified to 1e-12 precision via pytest:",
            "  pytest tests/test_hp004_joint_stabilizer.py \\",
            "         tests/test_hp004_hardware_artifact.py \\",
            "         tests/test_hp004b_fez_hardware_artifact.py",
            "",
            "References:",
            "  [1] P. Hayden and J. Preskill, J. High Energy Phys. 2007(09), 120 (2007).",
            "  [2] B. Yoshida and A. Kitaev, arXiv:1710.03363 (2017).",
            "  [3] K. A. Landsman et al., Nature 567, 61-65 (2019).",
            "  [4] B. M. Terhal and P. Horodecki, Phys. Rev. A 61, 040301(R) (2000); quant-ph/9911117.",
            "  [5] B. M. Terhal and K. G. H. Vollbrecht, Phys. Rev. Lett. 85, 2625-2628 (2000); doi:10.1103/PhysRevLett.85.2625.",
            "  [6] K. G. H. Vollbrecht and R. F. Werner, Phys. Rev. A 64, 062307 (2001); quant-ph/0010095.",
        ],
        font="/F3", size=8.0, leading=11.8,
    )

    return v20.build_pdf_document([
        (W, H, ("\n".join(p1) + "\n").encode("latin-1", errors="replace")),
        (W, H, ("\n".join(p2) + "\n").encode("latin-1", errors="replace")),
        (W, H, ("\n".join(p3) + "\n").encode("latin-1", errors="replace")),
        (W, H, ("\n".join(p4) + "\n").encode("latin-1", errors="replace")),
    ])


def main() -> None:
    res_m = json.loads((HP004_DIR / "physical_hp004_results.json").read_text(encoding="utf-8"))
    cal_m = json.loads((HP004_DIR / "hp004_calibration_snapshot.json").read_text(encoding="utf-8"))
    res_f = json.loads((FEZ_DIR / "physical_hp004b_fez_results.json").read_text(encoding="utf-8"))
    cal_f = json.loads((FEZ_DIR / "hp004b_fez_calibration_snapshot.json").read_text(encoding="utf-8"))

    mean_m = 100.0 * sum(v["cz_error"] for v in cal_m["edge_calibration"].values()) / len(cal_m["edge_calibration"])
    mean_f = 100.0 * sum(v["cz_error"] for v in cal_f["edge_calibration"].values()) / len(cal_f["edge_calibration"])

    fig1 = make_fig1_v2_1(res_m, res_f)
    fig2 = v20.make_fig2_blocks_and_marginals(res_m, res_f)
    fig3 = v20._make_tree_canvas(
        cal_m,
        "Pinned Heavy-Hex 8-Transmon Tree & Light-Cone Isolation (ibm_marrakesh)",
        cal_m["pinned_tree_layout"],
        mean_m,
    )
    fig4 = v20._make_tree_canvas(
        cal_f,
        "Pinned Heavy-Hex 8-Transmon Tree & Light-Cone Isolation (ibm_fez)",
        cal_f["pinned_tree_layout"],
        mean_f,
    )

    fig1.save(V21_DIR / "fig1_dyadic_capacity_ladder")
    fig2.save(V21_DIR / "fig2_block_stability_and_marginals")
    fig3.save(V21_DIR / "fig3_marrakesh_tree_light_cone")
    fig4.save(V21_DIR / "fig4_fez_tree_light_cone")

    pdf_bytes = build_preprint_pdf_v2_1(res_m, res_f, fig1, fig2, fig3, fig4)
    pdf_path = V21_DIR / "hp004_joint_stabilizer_heron_r2_v2_1.pdf"
    pdf_path.write_bytes(pdf_bytes)

    v21_files = [
        pdf_path,
        V21_DIR / "hp004_joint_stabilizer_heron_r2_v2_1.tex",
        V21_DIR / "generate_paper_assets_v2_1.py",
        V21_DIR / "fig1_dyadic_capacity_ladder.pdf",
        V21_DIR / "fig2_block_stability_and_marginals.pdf",
        V21_DIR / "fig3_marrakesh_tree_light_cone.pdf",
        V21_DIR / "fig4_fez_tree_light_cone.pdf",
    ]
    sums_lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in v21_files]
    (V21_DIR / "SHA256SUMS_v2_1.txt").write_text("\n".join(sums_lines) + "\n", encoding="utf-8")
    print(f"Generated v2.1 4-page PDF ({len(pdf_bytes):,} bytes), SVG/PDF figures, and SHA256SUMS_v2_1.txt.")


if __name__ == "__main__":
    main()
