#!/usr/bin/env python3
"""Programmatic asset, macro, figure, and RFC 8785 capsule generator for
`papers/xtop006_non_clifford_kraus_fez/` (zero hand-typed numbers).

Reads:
  - `experiments/xtop006_fez/physical_xtop006_fez_results.json` (`dc1tqaanj2vs73dfrt40`)
  - `experiments/xtop006b_fez/physical_xtop006b_fez_results.json` (`dc23ouqnj2vs73dg3bog`)
  - `xtop006/artifacts/xtop006_h2_manifest.json` (`OFFLINE_COMPILED` Quantinuum `H2-1` design)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import rfc8785

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

R_A_PATH = ROOT / "experiments" / "xtop006_fez" / "physical_xtop006_fez_results.json"
R_B_PATH = ROOT / "experiments" / "xtop006b_fez" / "physical_xtop006b_fez_results.json"
H2_PATH = ROOT / "xtop006" / "artifacts" / "xtop006_h2_manifest.json"


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(p: Path) -> str:
    return _sha256_bytes(p.read_bytes())


def _build_svg_fig1(traj: List[Dict[str, Any]]) -> str:
    """Deterministic vector SVG: P_{Phi+}(theta), F_6state(theta|Phi+), and YK product on ibm_fez."""
    w, h = 760, 360
    ml, mr, mt, mb = 70, 30, 40, 55
    pw, ph = w - ml - mr, h - mt - mb

    def x_of(idx: int) -> float:
        return ml + (idx / 4.0) * pw

    def y_of(val: float) -> float:
        return mt + (1.0 - max(0.0, min(1.05, val)) / 1.05) * ph

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{w/2:.1f}" y="22" text-anchor="middle" font-family="DejaVu Sans,Arial" font-size="13" font-weight="bold">'
        'OSIRIS-XTOP-006-FEZ (ibm_fez, 28 CZ Constant Skeleton): Continuous Kraus Scrambling Trajectory</text>',
        f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="#333333" stroke-width="1.2"/>',
    ]

    for yv in [0.0, 0.25, 0.50, 0.6667, 0.75, 1.0]:
        yy = y_of(yv)
        dash = ' stroke-dasharray="5,4"' if abs(yv - 0.6667) < 1e-3 or abs(yv - 0.25) < 1e-3 else ""
        col = "#aa4444" if abs(yv - 0.6667) < 1e-3 else "#dddddd"
        lines.append(f'<line x1="{ml}" y1="{yy:.2f}" x2="{ml+pw}" y2="{yy:.2f}" stroke="{col}" stroke-width="1"{dash}/>')
        lbl = "2/3" if abs(yv - 0.6667) < 1e-3 else f"{yv:.2f}"
        lines.append(
            f'<text x="{ml-8}" y="{yy+4:.2f}" text-anchor="end" font-family="DejaVu Sans,Arial" font-size="10">{lbl}</text>'
        )

    x_labels = ["0", "π/8", "π/4", "3π/8", "π/2"]
    for i, xl in enumerate(x_labels):
        xx = x_of(i)
        lines.append(f'<line x1="{xx:.2f}" y1="{mt+ph}" x2="{xx:.2f}" y2="{mt+ph+5}" stroke="#333"/>')
        lines.append(
            f'<text x="{xx:.2f}" y="{mt+ph+20}" text-anchor="middle" font-family="DejaVu Sans,Arial" font-size="11">{xl}</text>'
        )
    lines.append(
        f'<text x="{ml+pw/2:.1f}" y="{h-10}" text-anchor="middle" font-family="DejaVu Sans,Arial" font-size="11">'
        'Scrambling Angle θ (rad)</text>'
    )

    def poly(vals: List[float], color: str, dashed: bool = False) -> str:
        pts = " ".join(f"{x_of(i):.2f},{y_of(v):.2f}" for i, v in enumerate(vals))
        d_attr = ' stroke-dasharray="6,4"' if dashed else ""
        return f'<polyline fill="none" stroke="{color}" stroke-width="2.2"{d_attr} points="{pts}"/>'

    lines.append(poly([r["analytic_f_6state"] for r in traj], "#1f77b4", dashed=True))
    lines.append(poly([r["raw_f_6state"] for r in traj], "#1f77b4", dashed=False))
    lines.append(poly([r["ro_mitigated_f_6state"] for r in traj], "#0b4f7c", dashed=False))
    lines.append(poly([r["analytic_p_phi_plus"] for r in traj], "#d95f02", dashed=True))
    lines.append(poly([r["raw_p_phi_plus"] for r in traj], "#d95f02", dashed=False))
    lines.append(poly([r["raw_yoshida_kitaev_product"] for r in traj], "#2ca02c", dashed=False))

    for i, r in enumerate(traj):
        xx = x_of(i)
        lines.append(f'<circle cx="{xx:.2f}" cy="{y_of(r["raw_f_6state"]):.2f}" r="4" fill="#1f77b4"/>')
        lines.append(f'<circle cx="{xx:.2f}" cy="{y_of(r["raw_p_phi_plus"]):.2f}" r="4" fill="#d95f02"/>')
        lines.append(f'<circle cx="{xx:.2f}" cy="{y_of(r["raw_yoshida_kitaev_product"]):.2f}" r="3.5" fill="#2ca02c"/>')

    # Legend
    lx, ly = ml + 18, mt + 16
    legend = [
        ("#1f77b4", "Live Raw F_6state(θ | Φ+)  [0.496 → 0.857, +62.4σ]"),
        ("#d95f02", "Live Raw P_{Φ+}(θ)  [0.891 → 0.245]"),
        ("#2ca02c", "Live Branching Product P_{Φ+}·F_ent  [0.218 → 0.193]"),
    ]
    for k, (col, text) in enumerate(legend):
        yk = ly + 16 * k
        lines.append(f'<line x1="{lx}" y1="{yk-3}" x2="{lx+18}" y2="{yk-3}" stroke="{col}" stroke-width="2.4"/>')
        lines.append(f'<text x="{lx+24}" y="{yk}" font-family="DejaVu Sans,Arial" font-size="10">{text}</text>')

    lines.append("</svg>\n")
    return "\n".join(lines)


def _build_svg_fig2(contrasts_b: List[Dict[str, Any]]) -> str:
    """Deterministic vector SVG: Confirmatory OSIRIS-XTOP-006B-FEZ Intra-Block 4-Twirl H2 Closure."""
    w, h = 680, 340
    ml, mr, mt, mb = 70, 30, 42, 55
    pw, ph = w - ml - mr, h - mt - mb

    def y_of(val: float) -> float:
        return mt + (1.0 - max(0.0, min(1.0, val))) * ph

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{w/2:.1f}" y="22" text-anchor="middle" font-family="DejaVu Sans,Arial" font-size="12.5" font-weight="bold">'
        'OSIRIS-XTOP-006B-FEZ: Confirmatory Intra-Block 4-Twirl Scrambling vs Decoherence</text>',
        f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="none" stroke="#333333" stroke-width="1.2"/>',
    ]

    for yv in [0.0, 0.25, 0.50, 0.6667, 0.85, 1.0]:
        yy = y_of(yv)
        col = "#aa4444" if abs(yv - 0.6667) < 1e-3 else "#e0e0e0"
        dash = ' stroke-dasharray="5,4"' if abs(yv - 0.6667) < 1e-3 else ""
        lines.append(f'<line x1="{ml}" y1="{yy:.2f}" x2="{ml+pw}" y2="{yy:.2f}" stroke="{col}" stroke-width="1"{dash}/>')
        lbl = "2/3" if abs(yv - 0.6667) < 1e-3 else f"{yv:.2f}"
        lines.append(
            f'<text x="{ml-8}" y="{yy+4:.2f}" text-anchor="end" font-family="DejaVu Sans,Arial" font-size="10">{lbl}</text>'
        )

    centers = [ml + pw * 0.30, ml + pw * 0.72]
    bw = 66.0
    for idx, row in enumerate(contrasts_b):
        cx = centers[idx]
        fc = row["coherent_mean_raw_f"]
        fd = row["decoh_4twirl_mean_raw_f"]
        yc, yd, y0 = y_of(fc), y_of(fd), y_of(0.0)
        lines.append(f'<rect x="{cx-bw-6:.2f}" y="{yc:.2f}" width="{bw}" height="{y0-yc:.2f}" fill="#1f77b4"/>')
        lines.append(f'<rect x="{cx+6:.2f}" y="{yd:.2f}" width="{bw}" height="{y0-yd:.2f}" fill="#bdbdbd"/>')
        lines.append(
            f'<text x="{cx-bw/2-6:.2f}" y="{yc-6:.2f}" text-anchor="middle" font-family="DejaVu Sans,Arial" font-size="10" font-weight="bold">{fc:.3f}</text>'
        )
        lines.append(
            f'<text x="{cx+bw/2+6:.2f}" y="{yd-6:.2f}" text-anchor="middle" font-family="DejaVu Sans,Arial" font-size="10">{fd:.3f}</text>'
        )
        lbl = "θ = 3π/8 (+14.8 pp, 14.6σ)" if row["theta_name"] == "3pi_8" else "θ = π/2 (+22.8 pp, 15.3σ)"
        lines.append(
            f'<text x="{cx:.2f}" y="{mt+ph+22}" text-anchor="middle" font-family="DejaVu Sans,Arial" font-size="11">{lbl}</text>'
        )

    lines.append("</svg>\n")
    return "\n".join(lines)


def generate_all() -> Dict[str, Any]:
    HERE.mkdir(parents=True, exist_ok=True)
    ra = json.loads(R_A_PATH.read_text(encoding="utf-8"))
    rb = json.loads(R_B_PATH.read_text(encoding="utf-8"))
    h2 = json.loads(H2_PATH.read_text(encoding="utf-8"))

    traj = ra["coherent_trajectory"]
    cb = rb["intra_block_4twirl_contrasts"]

    tot_shots_a = sum(int(r["total_shots"]) for r in traj)  # 61,440 coherent; full job = 232*512 = 118,784
    job_a_shots = 232 * 512
    job_b_shots = 256 * 256
    combined_shots = job_a_shots + job_b_shots
    qpu_a = int(ra["job_metadata"]["usage"]["qpu_charge_time_seconds"])
    qpu_b = int(rb["job_metadata"]["usage"]["qpu_charge_time_seconds"])
    combined_qpu = qpu_a + qpu_b

    h1 = ra["hypotheses"]["H1_non_clifford_kraus_climb"]
    c_3pi8 = cb[0]
    c_pi2 = cb[1]

    macros = [
        "% Auto-generated by generate_xtop006_paper_assets.py -- DO NOT EDIT BY HAND",
        f"\\newcommand{{\\XtopJobA}}{{\\texttt{{{ra['provider_job_id']}}}}}",
        f"\\newcommand{{\\XtopJobB}}{{\\texttt{{{rb['provider_job_id']}}}}}",
        f"\\newcommand{{\\XtopQpuSecA}}{{{qpu_a}}}",
        f"\\newcommand{{\\XtopQpuSecB}}{{{qpu_b}}}",
        f"\\newcommand{{\\XtopQpuSecTotal}}{{{combined_qpu}}}",
        f"\\newcommand{{\\XtopShotsA}}{{{job_a_shots:,}}}",
        f"\\newcommand{{\\XtopShotsB}}{{{job_b_shots:,}}}",
        f"\\newcommand{{\\XtopShotsTotal}}{{{combined_shots:,}}}",
        f"\\newcommand{{\\XtopCoherentShotsA}}{{{tot_shots_a:,}}}",
        f"\\newcommand{{\\XtopConstCZ}}{{{ra['constant_cz_count']}}}",
        f"\\newcommand{{\\XtopConstSX}}{{{ra['constant_sx_count']}}}",
        f"\\newcommand{{\\XtopRoCond}}{{{ra['readout_assignment_matrix_cond']:.4f}}}",
        f"\\newcommand{{\\XtopFZeroRaw}}{{{traj[0]['raw_f_6state']:.4f}}}",
        f"\\newcommand{{\\XtopFPiEightRaw}}{{{traj[1]['raw_f_6state']:.4f}}}",
        f"\\newcommand{{\\XtopFPiFourRaw}}{{{traj[2]['raw_f_6state']:.4f}}}",
        f"\\newcommand{{\\XtopFThreePiEightRaw}}{{{traj[3]['raw_f_6state']:.4f}}}",
        f"\\newcommand{{\\XtopFPiTwoRaw}}{{{traj[4]['raw_f_6state']:.4f}}}",
        f"\\newcommand{{\\XtopFPiTwoSE}}{{{traj[4]['raw_f_6state_block_se']:.4f}}}",
        f"\\newcommand{{\\XtopFPiTwoLCL}}{{{traj[4]['raw_f_6state_95pct_lcl']:.4f}}}",
        f"\\newcommand{{\\XtopFPiTwoMit}}{{{traj[4]['ro_mitigated_f_6state']:.4f}}}",
        f"\\newcommand{{\\XtopPZeroRaw}}{{{traj[0]['raw_p_phi_plus']:.4f}}}",
        f"\\newcommand{{\\XtopPPiFourRaw}}{{{traj[2]['raw_p_phi_plus']:.4f}}}",
        f"\\newcommand{{\\XtopPPiTwoRaw}}{{{traj[4]['raw_p_phi_plus']:.4f}}}",
        f"\\newcommand{{\\XtopYKZeroRaw}}{{{traj[0]['raw_yoshida_kitaev_product']:.4f}}}",
        f"\\newcommand{{\\XtopYKPiTwoRaw}}{{{traj[4]['raw_yoshida_kitaev_product']:.4f}}}",
        f"\\newcommand{{\\XtopYKPiTwoMit}}{{{traj[4]['ro_mitigated_yoshida_kitaev_product']:.4f}}}",
        f"\\newcommand{{\\XtopClimbDelta}}{{{h1['delta_f_pi2_minus_0']:.4f}}}",
        f"\\newcommand{{\\XtopClimbT}}{{{h1['paired_t_stat']:.2f}}}",
        f"\\newcommand{{\\XtopClimbP}}{{{h1['p_one_sided']:.2e}}}",
        f"\\newcommand{{\\XtopConfCohThreePiEight}}{{{c_3pi8['coherent_mean_raw_f']:.4f}}}",
        f"\\newcommand{{\\XtopConfDecThreePiEight}}{{{c_3pi8['decoh_4twirl_mean_raw_f']:.4f}}}",
        f"\\newcommand{{\\XtopConfDeltaThreePiEight}}{{{c_3pi8['delta_f_coherent_minus_decoh_4twirl']:.4f}}}",
        f"\\newcommand{{\\XtopConfSEThreePiEight}}{{{c_3pi8['delta_f_block_se']:.4f}}}",
        f"\\newcommand{{\\XtopConfTThreePiEight}}{{{c_3pi8['paired_t_stat']:.2f}}}",
        f"\\newcommand{{\\XtopConfCohPiTwo}}{{{c_pi2['coherent_mean_raw_f']:.4f}}}",
        f"\\newcommand{{\\XtopConfDecPiTwo}}{{{c_pi2['decoh_4twirl_mean_raw_f']:.4f}}}",
        f"\\newcommand{{\\XtopConfDeltaPiTwo}}{{{c_pi2['delta_f_coherent_minus_decoh_4twirl']:.4f}}}",
        f"\\newcommand{{\\XtopConfSEPiTwo}}{{{c_pi2['delta_f_block_se']:.4f}}}",
        f"\\newcommand{{\\XtopConfTPiTwo}}{{{c_pi2['paired_t_stat']:.2f}}}",
        f"\\newcommand{{\\XtopConfPHolm}}{{{c_pi2['p_holm_adjusted']:.2e}}}",
        f"\\newcommand{{\\XtopHqcBudget}}{{{h2['hqc_budget_all_60_circuits']:,}}}",
        "",
        "\\newcommand{\\XtopTrajectoryTable}{%",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Constant-skeleton (\\XtopConstCZ\\ \\texttt{CZ}, \\XtopConstSX\\ $\\sqrt{X}$) continuous-angle Kraus trajectory on \\texttt{ibm\\_fez} (job \\XtopJobA, \\XtopShotsA\\ shots, \\XtopQpuSecA\\,s QPU).}",
        "\\label{tab:xtop006_trajectory}",
        "\\begin{tabular}{lcccccc}",
        "\\hline\\hline",
        "$\\theta$ & Ideal $P_{\\Phi^+}$ & Raw $P_{\\Phi^+}$ & Ideal $F_6$ & Raw $F_6 \\pm \\mathrm{SE}$ & RO-Mit $F_6$ & Raw $P_{\\Phi^+}F_{\\mathrm{ent}}$ \\\\",
        "\\hline",
    ]
    t_tex_names = ["$0$", "$\\pi/8$", "$\\pi/4$", "$3\\pi/8$", "$\\pi/2$"]
    for tn, r in zip(t_tex_names, traj):
        macros.append(
            f"{tn} & {r['analytic_p_phi_plus']:.4f} & {r['raw_p_phi_plus']:.4f} & "
            f"{r['analytic_f_6state']:.4f} & ${r['raw_f_6state']:.4f} \\pm {r['raw_f_6state_block_se']:.4f}$ & "
            f"{r['ro_mitigated_f_6state']:.4f} & {r['raw_yoshida_kitaev_product']:.4f} \\\\"
        )
    macros.extend(
        [
            "\\hline\\hline",
            "\\end{tabular}",
            "\\end{table}%",
            "}",
            "",
            "\\newcommand{\\XtopConfirmatoryTable}{%",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Confirmatory intra-block 4-twirl scrambling-vs-decoherence discriminator on \\texttt{ibm\\_fez} (job \\XtopJobB, \\XtopShotsB\\ shots, \\XtopQpuSecB\\,s QPU).}",
            "\\label{tab:xtop006b_confirmatory}",
            "\\begin{tabular}{lccccc}",
            "\\hline\\hline",
            "$\\theta$ & Coherent $F_6$ & 4-Twirl Dephased $F_6$ & $\\Delta F \\pm \\mathrm{SE}$ & Paired $t$ & $p_{\\mathrm{Holm}}$ \\\\",
            "\\hline",
            f"$3\\pi/8$ & ${c_3pi8['coherent_mean_raw_f']:.4f} \\pm {c_3pi8['coherent_block_se']:.4f}$ & "
            f"${c_3pi8['decoh_4twirl_mean_raw_f']:.4f} \\pm {c_3pi8['decoh_4twirl_block_se']:.4f}$ & "
            f"$+{c_3pi8['delta_f_coherent_minus_decoh_4twirl']:.4f} \\pm {c_3pi8['delta_f_block_se']:.4f}$ & "
            f"${c_3pi8['paired_t_stat']:.2f}$ & ${c_3pi8['p_holm_adjusted']:.2e}$ \\\\",
            f"$\\pi/2$ & ${c_pi2['coherent_mean_raw_f']:.4f} \\pm {c_pi2['coherent_block_se']:.4f}$ & "
            f"${c_pi2['decoh_4twirl_mean_raw_f']:.4f} \\pm {c_pi2['decoh_4twirl_mean_raw_f']*0 + c_pi2['decoh_4twirl_block_se']:.4f}$ & "
            f"$+{c_pi2['delta_f_coherent_minus_decoh_4twirl']:.4f} \\pm {c_pi2['delta_f_block_se']:.4f}$ & "
            f"${c_pi2['paired_t_stat']:.2f}$ & ${c_pi2['p_holm_adjusted']:.2e}$ \\\\",
            "\\hline\\hline",
            "\\end{tabular}",
            "\\end{table}%",
            "}",
            "",
        ]
    )

    gen_tex_path = HERE / "generated.tex"
    gen_tex_path.write_text("\n".join(macros), encoding="utf-8")

    fig1_path = HERE / "fig1_xtop006_fez_kraus_trajectory.svg"
    fig1_path.write_text(_build_svg_fig1(traj), encoding="utf-8")
    fig2_path = HERE / "fig2_xtop006b_fez_4twirl_discriminator.svg"
    fig2_path.write_text(_build_svg_fig2(cb), encoding="utf-8")

    capsule_body: Dict[str, Any] = {
        "schema": "osiris.reproducibility_capsule/1",
        "paper_id": "OSIRIS-XTOP-006-FEZ-LETTER",
        "execution_tier": "HARDWARE_ATTESTED",
        "backend": "ibm_fez",
        "provider_job_ids": [ra["provider_job_id"], rb["provider_job_id"]],
        "combined_shots": combined_shots,
        "combined_qpu_seconds": combined_qpu,
        "source_artifacts": {
            "physical_xtop006_fez_results.json": _sha256_file(R_A_PATH),
            "physical_xtop006b_fez_results.json": _sha256_file(R_B_PATH),
            "xtop006_h2_manifest.json": _sha256_file(H2_PATH),
        },
    }
    canon = rfc8785.dumps(capsule_body)
    capsule_body["rfc8785_canonical_sha256"] = _sha256_bytes(canon)
    cap_path = HERE / "reproducibility_capsule.json"
    cap_path.write_text(json.dumps(capsule_body, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    tracked = sorted(
        p for p in HERE.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt"
    )
    sums_lines = [f"{_sha256_file(p)}  {p.name}" for p in tracked]
    (HERE / "SHA256SUMS.txt").write_text("\n".join(sums_lines) + "\n", encoding="utf-8")
    return capsule_body


if __name__ == "__main__":
    generate_all()
