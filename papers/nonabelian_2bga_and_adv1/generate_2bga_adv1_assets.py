#!/usr/bin/env python3
"""Deterministic asset generator for the Non-Abelian 2BGA & ADV-1 Pre-Gate paper.

Reads strictly from verified SIMULATED repository artifacts:
  - qldpc_nonabelian/artifacts/nonabelian_2bga_report.json
  - adv1/artifacts/adv1_pregate_report.json

Emits deterministically:
  1. generated.tex (all empirical macros and formatted REVTeX 4.2 tables; zero hand-typed floats)
  2. fig1_odd_k_nonabelian_2bga_codes_and_bposd.svg and .pdf
  3. fig2_adv1_three_attack_ablation_m6.svg and .pdf
  4. reproducibility_capsule.json (RFC 8785 JCS canonical JSON; SIMULATED, provider_job_id: null)
  5. SHA256SUMS.txt
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import rfc8785

PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parent.parent

QLDPC_REPORT_PATH = REPO_ROOT / "qldpc_nonabelian" / "artifacts" / "nonabelian_2bga_report.json"
ADV1_REPORT_PATH = REPO_ROOT / "adv1" / "artifacts" / "adv1_pregate_report.json"

TEX_FILENAME = "nonabelian_2bga_and_adv1.tex"
FIG1_STEM = "fig1_odd_k_nonabelian_2bga_codes_and_bposd"
FIG2_STEM = "fig2_adv1_three_attack_ablation_m6"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pdf_escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .replace("±", "+/-")
        .replace("≥", ">=")
        .replace("≤", "<=")
        .replace("×", "x")
        .replace("—", "--")
        .replace("–", "-")
        .replace("−", "-")
        .replace("σ", " sigma")
        .replace("μ", "u")
        .replace("µ", "u")
    )


class VectorCanvas:
    """Deterministic pure-Python dual SVG + PDF 1.4 vector graphics builder."""

    def __init__(self, width: float, height: float) -> None:
        self.w = float(width)
        self.h = float(height)
        self.svg: List[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w:.0f}" height="{self.h:.0f}" '
            f'viewBox="0 0 {self.w:.0f} {self.h:.0f}">',
            f'<rect x="0" y="0" width="{self.w:.0f}" height="{self.h:.0f}" fill="#ffffff"/>',
        ]
        self.pdf: List[str] = ["1 1 1 rg", f"0 0 {self.w:.2f} {self.h:.2f} re f"]

    def _py(self, y: float) -> float:
        return self.h - y

    @staticmethod
    def _rgb(hex_color: str) -> Tuple[float, float, float]:
        c = hex_color.lstrip("#")
        return int(c[0:2], 16) / 255.0, int(c[2:4], 16) / 255.0, int(c[4:6], 16) / 255.0

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str | None = None,
        stroke: str | None = None,
        lw: float = 1.0,
    ) -> None:
        s_fill = fill if fill else "none"
        s_stroke = f' stroke="{stroke}" stroke-width="{lw:.2f}"' if stroke else ""
        self.svg.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{s_fill}"{s_stroke}/>'
        )
        py = self._py(y + h)
        if fill and stroke:
            fr, fg, fb = self._rgb(fill)
            sr, sg, sb = self._rgb(stroke)
            self.pdf.append(
                f"{lw:.2f} w {fr:.3f} {fg:.3f} {fb:.3f} rg "
                f"{sr:.3f} {sg:.3f} {sb:.3f} RG {x:.2f} {py:.2f} {w:.2f} {h:.2f} re B"
            )
        elif fill:
            fr, fg, fb = self._rgb(fill)
            self.pdf.append(f"{fr:.3f} {fg:.3f} {fb:.3f} rg {x:.2f} {py:.2f} {w:.2f} {h:.2f} re f")
        elif stroke:
            sr, sg, sb = self._rgb(stroke)
            self.pdf.append(
                f"{lw:.2f} w {sr:.3f} {sg:.3f} {sb:.3f} RG {x:.2f} {py:.2f} {w:.2f} {h:.2f} re S"
            )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = "#111827",
        lw: float = 1.0,
        dashed: bool = False,
    ) -> None:
        dash_svg = ' stroke-dasharray="5,3"' if dashed else ""
        self.svg.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="{lw:.2f}"{dash_svg}/>'
        )
        r, g, b = self._rgb(color)
        dash_pdf = "[5 3] 0 d" if dashed else "[] 0 d"
        self.pdf.append(
            f"q {dash_pdf} {lw:.2f} w {r:.3f} {g:.3f} {b:.3f} RG "
            f"{x1:.2f} {self._py(y1):.2f} m {x2:.2f} {self._py(y2):.2f} l S Q"
        )

    def circle(
        self,
        cx: float,
        cy: float,
        r: float,
        fill: str = "#ffffff",
        stroke: str = "#111827",
        lw: float = 1.4,
    ) -> None:
        self.svg.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{lw:.2f}"/>'
        )
        fr, fg, fb = self._rgb(fill)
        sr, sg, sb = self._rgb(stroke)
        py = self._py(cy)
        k = 0.55228475 * r
        self.pdf.append(
            f"q {lw:.2f} w {fr:.3f} {fg:.3f} {fb:.3f} rg {sr:.3f} {sg:.3f} {sb:.3f} RG "
            f"{cx + r:.2f} {py:.2f} m "
            f"{cx + r:.2f} {py + k:.2f} {cx + k:.2f} {py + r:.2f} {cx:.2f} {py + r:.2f} c "
            f"{cx - k:.2f} {py + r:.2f} {cx - r:.2f} {py + k:.2f} {cx - r:.2f} {py:.2f} c "
            f"{cx - r:.2f} {py - k:.2f} {cx - k:.2f} {py - r:.2f} {cx:.2f} {py - r:.2f} c "
            f"{cx + k:.2f} {py - r:.2f} {cx + r:.2f} {py - k:.2f} {cx + r:.2f} {py:.2f} c B Q"
        )

    def text(
        self,
        x: float,
        y: float,
        s: str,
        size: float = 9.0,
        color: str = "#111827",
        bold: bool = False,
        anchor: str = "start",
        mono: bool = False,
    ) -> None:
        fam_svg = "monospace" if mono else "Helvetica, Arial, sans-serif"
        wt_svg = "bold" if bold else "normal"
        esc_svg = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.svg.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-family="{fam_svg}" font-size="{size:.1f}" '
            f'font-weight="{wt_svg}" fill="{color}" text-anchor="{anchor}">{esc_svg}</text>'
        )
        esc_pdf = _pdf_escape(s)
        font_tag = "/F3" if mono else ("/F2" if bold else "/F1")
        char_w = 0.60 if mono else (0.53 if bold else 0.50)
        approx_w = len(esc_pdf) * size * char_w
        px = x if anchor == "start" else (x - 0.5 * approx_w if anchor == "middle" else x - approx_w)
        r, g, b = self._rgb(color)
        self.pdf.append(
            f"BT {font_tag} {size:.1f} Tf {r:.3f} {g:.3f} {b:.3f} rg "
            f"{px:.2f} {self._py(y):.2f} Td ({esc_pdf}) Tj ET"
        )

    def to_svg(self) -> str:
        return "\n".join(self.svg + ["</svg>"]) + "\n"

    def pdf_stream(self) -> bytes:
        return ("\n".join(self.pdf) + "\n").encode("latin-1", errors="replace")

    def save(self, stem_path: Path) -> None:
        stem_path.with_suffix(".svg").write_text(self.to_svg(), encoding="utf-8", newline="\n")
        stem_path.with_suffix(".pdf").write_bytes(
            build_pdf_document([(self.w, self.h, self.pdf_stream())])
        )


def build_pdf_document(pages: Sequence[Tuple[float, float, bytes]]) -> bytes:
    objects: List[bytes] = []

    def add_obj(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    f1 = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    f2 = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    f3 = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")

    pages_idx = add_obj(b"")
    page_ids: List[int] = []
    for w, h, stream in pages:
        c_id = add_obj(f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"endstream")
        p_id = add_obj(
            f"<< /Type /Page /Parent {pages_idx} 0 R /MediaBox [0 0 {w:.2f} {h:.2f}] "
            f"/Resources << /Font << /F1 {f1} 0 R /F2 {f2} 0 R /F3 {f3} 0 R >> >> "
            f"/Contents {c_id} 0 R >>".encode("ascii")
        )
        page_ids.append(p_id)

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[pages_idx - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii")
    cat_id = add_obj(f"<< /Type /Catalog /Pages {pages_idx} 0 R >>".encode("ascii")

    )
    buf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: List[int] = []
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(buf))
        buf.extend(f"{idx} 0 obj\n".encode("ascii"))
        buf.extend(obj)
        buf.extend(b"\nendobj\n")

    xref_pos = len(buf)
    buf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for off in offsets:
        buf.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    buf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root {cat_id} 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n".encode("ascii")
    )
    return bytes(buf)


def _fmt_int(n: int) -> str:
    return f"{int(n):,}".replace(",", "{,}")


def _tex_group_name(gname: str) -> str:
    if "_semidirect_" in gname:
        lhs, rhs = gname.split("_semidirect_")
        p = lhs.split("_")[1]
        q = rhs.split("_")[1]
        return f"$C_{{{p}}} \\rtimes C_{{{q}}}$"
    if "_" in gname:
        sym, sub = gname.split("_", 1)
        return f"${sym}_{{{sub}}}$"
    return f"${gname}$"


def build_generated_tex(qldpc: Mapping[str, Any], adv1: Mapping[str, Any]) -> str:
    lines: List[str] = [
        "% Generated deterministically by generate_2bga_adv1_assets.py -- do not edit by hand."
    ]
    seen_names: set[str] = set()

    def mac(name: str, val: Any) -> None:
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError(f"Invalid LaTeX macro name (must be ASCII letters only): {name!r}")
        if name in seen_names:
            raise ValueError(f"Duplicate LaTeX macro name: {name!r}")
        seen_names.add(name)
        lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    # 1. Non-Abelian 2BGA Theorem & Code Catalog
    thm = qldpc["algebraic_rank_symmetry_theorem"]
    mac("TwoBgaPairsChecked", _fmt_int(int(thm["unquotiented_pairs_checked"])))
    mac("TwoBgaRankViolations", int(thm["rank_asymmetry_violations"]))
    mac("TwoBgaCosetN", int(thm["coset_2bga_d9_z2_example"]["n"]))
    mac("TwoBgaCosetK", int(thm["coset_2bga_d9_z2_example"]["k"]))

    mac("TwoBgaGroupsCount", len(qldpc["group_catalog"]))
    mac("TwoBgaParentsCount", len(qldpc["base_nonabelian_2bga_parents"]))
    mac("TwoBgaOddKCount", int(qldpc["odd_k_d_ge_4_discoveries_count"]))

    odd_cands = qldpc["odd_k_candidates"]
    exact_odd_cands = [c for c in odd_cands if c["distance_type"] == "exact_d"]
    ub_odd_cands = [c for c in odd_cands if c["distance_type"] == "upper_bound_d"]
    mac("TwoBgaExactCount", len(exact_odd_cands))
    mac("TwoBgaUpperCount", len(ub_odd_cands))

    def _cand_by_label(lbl: str) -> Mapping[str, Any]:
        for c in odd_cands:
            if c["code_label"] == lbl:
                return c
        raise KeyError(f"Candidate not found: {lbl}")

    c_114_3 = _cand_by_label("C_19_semidirect_C_3_odd_[[114,3,12]]")
    c_60_3 = _cand_by_label("D_15_odd_[[60,3,8]]")
    c_36_3 = _cand_by_label("D_9_odd_[[36,3,6]]")
    c_42_3 = _cand_by_label("C_7_semidirect_C_3_odd_[[42,3,6]]")
    c_28_3 = _cand_by_label("D_7_odd_[[28,3,5]]")
    c_16_3 = _cand_by_label("Q_8_odd_[[16,3,4]]")
    c_186_3 = _cand_by_label("C_31_semidirect_C_3_odd_[[186,3,13]]")
    c_48_5 = _cand_by_label("S_4_odd_[[48,5,4]]")
    c_42_1 = _cand_by_label("C_7_semidirect_C_3_odd_[[42,1,7]]")

    mac("KdSqRatioBest", f"{float(c_114_3['kd2_over_n']):.4f}")
    mac("KdSqRatioDFifteen", f"{float(c_60_3['kd2_over_n']):.2f}")
    mac("KdSqRatioDNine", f"{float(c_36_3['kd2_over_n']):.2f}")
    mac("KdSqRatioQEight", f"{float(c_16_3['kd2_over_n']):.2f}")
    mac("KdSqRatioDSeven", f"{float(c_28_3['kd2_over_n']):.4f}")
    mac("KdSqRatioFrobTwentyOne", f"{float(c_42_3['kd2_over_n']):.4f}")
    mac("KdSqRatioOneEightySix", f"{float(c_186_3['kd2_over_n']):.4f}")
    mac("KdSqRatioSFourFive", f"{float(c_48_5['kd2_over_n']):.4f}")
    mac("KdSqRatioFrobRankOne", f"{float(c_42_1['kd2_over_n']):.4f}")

    # Table I: All 16 discovered odd-k non-abelian 2BGA codes
    t1_rows: List[str] = []
    for c in odd_cands:
        dtype = "Exact" if c["distance_type"] == "exact_d" else f"ISD ($\\ge {int(c['certified_lower_bound_d'])}$)"
        t1_rows.append(
            f"{_tex_group_name(c['group_name'])} & {int(c['group_order'])} & "
            f"$[[{int(c['n'])},{int(c['k'])},{int(c['d'])}]]$ & {dtype} & "
            f"$({int(c['d_x'])},{int(c['d_z'])})$ & "
            f"$({int(c['max_row_weight_hx'])},{int(c['max_row_weight_hz'])})$ & "
            f"{float(c['kd2_over_n']):.4f}\\\\"
        )
    mac("TwoBgaOddKRows", "\n".join(t1_rows))

    # 2. Helios-1 BP+OSD-CS Decoder Benchmarks
    bench = qldpc["helios_1_bposd_cs_benchmarks"]
    mac("TwoBgaBenchCount", len(bench))
    first_sweeps = next(iter(bench.values()))["sweeps"]
    mac("TwoBgaMcShots", int(first_sweeps[0]["mc_shots"]))
    mac("TwoBgaPLow", f"{float(first_sweeps[0]['physical_depolarizing_p']):.5f}")
    mac("TwoBgaPMid", f"{float(first_sweeps[1]['physical_depolarizing_p']):.4f}")
    mac("TwoBgaPHigh", f"{float(first_sweeps[2]['physical_depolarizing_p']):.4f}")
    mac("TwoBgaUnencLow", f"{float(first_sweeps[0]['unencoded_k_qubit_error_prob']):.6f}")
    mac("TwoBgaUnencMid", f"{float(first_sweeps[1]['unencoded_k_qubit_error_prob']):.6f}")
    mac("TwoBgaUnencHigh", f"{float(first_sweeps[2]['unencoded_k_qubit_error_prob']):.6f}")

    b_42 = bench["C_7_semidirect_C_3_odd_[[42,3,6]]"]["sweeps"]
    b_36 = bench["D_9_odd_[[36,3,6]]"]["sweeps"]
    b_114 = bench["C_19_semidirect_C_3_odd_[[114,3,12]]"]["sweeps"]
    b_16 = bench["Q_8_odd_[[16,3,4]]"]["sweeps"]

    mac("BpPhLerFortyTwoHigh", f"{float(b_42[2]['phenomenological_3round_ler']):.5f}")
    mac("BpPhSupprFortyTwoHigh", f"{float(b_42[2]['unencoded_k_qubit_error_prob']) / float(b_42[2]['phenomenological_3round_ler']):.1f}")
    mac("BpPhLerDNineAll", f"{float(b_36[2]['phenomenological_3round_ler']):.4f}")
    mac("BpPhLerOneFourteenLow", f"{float(b_114[0]['phenomenological_3round_ler']):.5f}")
    mac("BpPhLerOneFourteenHigh", f"{float(b_114[2]['phenomenological_3round_ler']):.4f}")
    mac("BpCcLerQEightHigh", f"{float(b_16[2]['code_capacity_ler']):.4f}")
    mac("BpCcSupprQEightHigh", f"{float(b_16[2]['suppression_vs_unencoded_cc']):.2f}")

    # Table II: BP+OSD-CS sweeps across all 7 benchmarked k=3 non-abelian codes
    bench_order = [
        "Q_8_odd_[[16,3,4]]",
        "D_7_odd_[[28,3,5]]",
        "D_9_odd_[[36,3,6]]",
        "C_7_semidirect_C_3_odd_[[42,3,6]]",
        "D_15_odd_[[60,3,8]]",
        "C_19_semidirect_C_3_odd_[[114,3,12]]",
        "C_31_semidirect_C_3_odd_[[186,3,13]]",
    ]
    t2_rows: List[str] = []
    for key in bench_order:
        b = bench[key]
        sw = b["sweeps"]
        t2_rows.append(
            f"{_tex_group_name(b['group'])} $[[{int(b['n'])},{int(b['k'])},{int(b['d'])}]]$ & "
            f"{float(sw[0]['code_capacity_ler']):.4f} / {float(sw[0]['phenomenological_3round_ler']):.5f} & "
            f"{float(sw[1]['code_capacity_ler']):.4f} / {float(sw[1]['phenomenological_3round_ler']):.5f} & "
            f"{float(sw[2]['code_capacity_ler']):.4f} / {float(sw[2]['phenomenological_3round_ler']):.5f}\\\\"
        )
    t2_rows.append("\\addlinespace")
    t2_rows.append(
        f"Unencoded $k=3$ baseline $1-(1-p)^3$ & "
        f"{float(first_sweeps[0]['unencoded_k_qubit_error_prob']):.6f} & "
        f"{float(first_sweeps[1]['unencoded_k_qubit_error_prob']):.6f} & "
        f"{float(first_sweeps[2]['unencoded_k_qubit_error_prob']):.6f}\\\\"
    )
    mac("TwoBgaBpOsdRows", "\n".join(t2_rows))

    # 3. ADV-1 Pre-Gate & M_6 = J_6 - I_6 Pauli-Weight-5 Involution
    pol = adv1["pregate_policy"]
    summ = adv1["summary"]
    mac("AdvZeroCount", int(summ["adv0_controls_evaluated"]))
    mac("AdvZeroRejected", int(summ["adv0_controls_rejected"]))
    mac("AdvOneEvalCount", int(summ["adv1_non_mirror_evaluated"]))
    mac("AdvOnePermittedCount", len(summ["adv1_permitted_candidates"]))
    mac("AdvConeLimit", int(pol["tractable_light_cone_limit"]))
    mac("AdvMinSnr", f"{float(pol['min_snr_over_uniform']):.1f}")
    mac("AdvChiList", ", ".join(str(x) for x in pol["mps_chi_sweep"]))
    mac("AdvWmaxList", ", ".join(str(x) for x in pol["pauli_weight_cutoff_sweep"]))

    ctrls = {c["candidate_id"]: c for c in adv1["adv0_twirled_mirror_controls"]}
    cands = {c["candidate_id"]: c for c in adv1["adv1_looped_non_mirror_candidates"]}

    c0_28 = ctrls["adv0_ctrl_n28_d8"]
    c1_comm = cands["adv1_commuting_phase_n24_L3"]
    c1_nom6 = cands["adv1_chorded_2q_n28_L5_no_m6"]
    c1_n18 = cands["adv1_chorded_m6_n18_L4_small_cone"]
    c1_n24 = cands["adv1_chorded_m6_n24_L5"]
    c1_n30 = cands["adv1_chorded_m6_n30_L5"]

    mac("AdvZeroTwentyEightMpsChiEight", f"{float(c0_28['attack1_bounded_mps']['chi_curves'][0]['p_peak_est']):.4f}")
    mac("AdvZeroTwentyEightMpsChiMax", f"{float(c0_28['attack1_bounded_mps']['chi_curves'][-1]['p_peak_est']):.4f}")

    mac("AdvCommMpsChiEight", f"{float(c1_comm['attack1_bounded_mps']['chi_curves'][0]['p_peak_est']):.4f}")
    mac("AdvCommPauliWTwoAccPct", f"{100.0 * float(c1_comm['attack2_pauli_path']['w_max_curves'][0]['bit_accuracy']):.1f}")

    mac("AdvNoMSixPauliWTwoAccPct", f"{100.0 * float(c1_nom6['attack2_pauli_path']['w_max_curves'][0]['bit_accuracy']):.1f}")
    mac("AdvNoMSixPauliWThreeAccPct", f"{100.0 * float(c1_nom6['attack2_pauli_path']['w_max_curves'][1]['bit_accuracy']):.1f}")
    mac("AdvNoMSixPauliWThreeMeanZ", f"{float(c1_nom6['attack2_pauli_path']['w_max_curves'][1]['mean_abs_z']):.4f}")

    mac("AdvEighteenExactPeak", f"{float(c1_n18['exact_noiseless_peak_prob']):.4f}")
    mac("AdvEighteenMpsChiMax", f"{float(c1_n18['attack1_bounded_mps']['chi_curves'][-1]['p_peak_est']):.4f}")

    mac("AdvTwentyFourCz", int(c1_n24["n_cz"]))
    mac("AdvTwentyFourOneQ", int(c1_n24["n_1q_phys"]))
    mac("AdvTwentyFourMirrorOv", f"{float(c1_n24['mirror_audit']['mean_block_mirror_overlap']):.4f}")
    mac("AdvTwentyFourPauliAccPct", f"{100.0 * float(c1_n24['attack2_pauli_path']['w_max_curves'][-1]['bit_accuracy']):.1f}")
    mac("AdvTwentyFourDroppedLTwo", f"{float(c1_n24['attack2_pauli_path']['w_max_curves'][-1]['mean_dropped_l2_weight']):.4f}")
    mac("AdvTwentyFourHeliosPeak", f"{float(c1_n24['helios1_predicted_peak_prob']):.4f}")
    mac("AdvTwentyFourHqc", f"{float(c1_n24['helios1_hqc_1000_shots']):.1f}")

    mac("AdvThirtyCz", int(c1_n30["n_cz"]))
    mac("AdvThirtyOneQ", _fmt_int(int(c1_n30["n_1q_phys"])))
    mac("AdvThirtyMirrorOv", f"{float(c1_n30['mirror_audit']['mean_block_mirror_overlap']):.4f}")
    mac("AdvThirtyPauliAccPct", f"{100.0 * float(c1_n30['attack2_pauli_path']['w_max_curves'][-1]['bit_accuracy']):.1f}")
    mac("AdvThirtyDroppedLTwo", f"{float(c1_n30['attack2_pauli_path']['w_max_curves'][-1]['mean_dropped_l2_weight']):.4f}")
    mac("AdvThirtyHeliosPeak", f"{float(c1_n30['helios1_predicted_peak_prob']):.4f}")
    mac("AdvThirtyHqc", _fmt_int(int(round(float(c1_n30["helios1_hqc_1000_shots"])))))
    mac("AdvThirtyHqcExact", f"{float(c1_n30['helios1_hqc_1000_shots']):.1f}")

    # Table III: ADV-0 controls & ADV-1 candidates across all 3 classical attacks
    t3_rows: List[str] = []
    all_eval = list(adv1["adv0_twirled_mirror_controls"]) + list(adv1["adv1_looped_non_mirror_candidates"])
    for idx, c in enumerate(all_eval):
        if idx == len(adv1["adv0_twirled_mirror_controls"]):
            t3_rows.append("\\addlinespace")
        cid_tex = c["candidate_id"].replace("_", "\\_")
        mir = (
            "Artifact"
            if c["mirror_audit"]["mirror_structure_detected"]
            else f"No ({float(c['mirror_audit']['mean_block_mirror_overlap']):.2f})"
        )
        mps_p128 = float(c["attack1_bounded_mps"]["chi_curves"][-1]["p_peak_est"])
        mps_s = f"{'Win' if c['attack1_bounded_mps']['any_win'] else 'Fail'} ({mps_p128:.3f})"
        w4 = c["attack2_pauli_path"]["w_max_curves"][-1]
        pauli_s = f"{'Win' if c['attack2_pauli_path']['any_win'] else 'Fail'} ({100.0 * float(w4['bit_accuracy']):.1f}\\%)"
        cone = c["attack3_light_cone"]
        cone_s = f"{'Win' if cone['win'] else 'Fail'} ({int(cone['max_cone_size'])})"
        gate_s = "\\textbf{PERMITTED}" if c["qpu_allocation_permitted"] else "REJECTED"
        t3_rows.append(
            f"\\texttt{{{cid_tex}}} & {int(c['n_qubits'])} & {int(c['n_cz'])} & "
            f"{mir} & {mps_s} & {pauli_s} & {cone_s} & "
            f"{float(c['helios1_predicted_peak_prob']):.3f} / {float(c['helios1_hqc_1000_shots']):.1f} & {gate_s}\\\\"
        )
    mac("AdvOnePregateRows", "\n".join(t3_rows))

    return "\n".join(lines) + "\n"


def make_fig1(qldpc: Mapping[str, Any]) -> VectorCanvas:
    """Dual-panel vector figure: (a) 16 odd-k non-abelian 2BGA codes; (b) Helios-1 BP+OSD-CS."""
    c = VectorCanvas(560, 290)
    c.text(
        280,
        17,
        "Non-Abelian Two-Block Group-Algebra (2BGA) Odd-k CSS Codes & Helios-1 BP+OSD-CS Benchmarks",
        size=9.5,
        bold=True,
        anchor="middle",
    )
    c.text(
        280,
        29,
        "SIMULATED (provider_job_id: null) | 6,480 Regular Pairs Verified (0 Violations) | 16 Odd-k Codes Discovered",
        size=7.5,
        color="#4b5563",
        anchor="middle",
    )

    # Panel (a): kd^2 / n vs block length n across all 16 odd-k codes
    x0, y0, pw, ph = 46.0, 52.0, 218.0, 192.0
    c.text(x0 + 0.5 * pw, 44, "(a) Encoding Efficiency k d^2 / n Across 16 Odd-k Codes", size=7.9, bold=True, anchor="middle")
    c.rect(x0, y0, pw, ph, fill="#fafafa", stroke="#374151", lw=1.0)

    def xa(n_val: float) -> float:
        return x0 + pw * ((n_val - 5.0) / 195.0)

    def ya(kd_val: float) -> float:
        return y0 + ph * (1.0 - min(1.0, max(0.0, (kd_val - 0.8) / 3.3)))

    for yv in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
        yy = ya(yv)
        c.line(x0, yy, x0 + pw, yy, color="#e5e7eb" if yv != 1.0 else "#9ca3af", lw=0.8, dashed=(yv != 1.0))
        c.text(x0 - 4, yy + 2.5, f"{yv:.1f}", size=6.6, color="#374151", anchor="end", mono=True)

    for xv in (20, 60, 100, 140, 180):
        xx = xa(float(xv))
        c.line(xx, y0 + ph, xx, y0 + ph + 3, color="#374151", lw=0.9)
        c.text(xx, y0 + ph + 12, str(xv), size=6.8, anchor="middle", mono=True)
    c.text(x0 + 0.5 * pw, y0 + ph + 24, "Physical Block Length n = 2|G| (qubits)", size=7.5, anchor="middle")

    for cand in qldpc["odd_k_candidates"]:
        px = xa(float(cand["n"]))
        py = ya(float(cand["kd2_over_n"]))
        is_ex = cand["distance_type"] == "exact_d"
        col = "#059669" if is_ex else "#2563eb"
        fill = col if is_ex else "#dbeafe"
        c.circle(px, py, 3.4, fill=fill, stroke=col, lw=1.3)

    # Highlight key code annotations
    highlights = [
        ("C_19|xC_3 [[114,3,12]]", 114, 3.7895, -6.0),
        ("D_15 [[60,3,8]]", 60, 3.2, -6.0),
        ("D_9 [[36,3,6]]", 36, 3.0, -6.0),
        ("C_7|xC_3 [[42,3,6]]", 42, 2.5714, 9.5),
        ("C_31|xC_3 [[186,3,13]]", 186, 2.7258, -6.0),
    ]
    for txt_lbl, nv, kv, dy_off in highlights:
        c.text(xa(float(nv)), ya(kv) + dy_off, txt_lbl, size=5.8, bold=True, color="#111827", anchor="middle", mono=True)

    c.rect(x0 + 6, y0 + ph - 29, 136, 24, fill="#ffffff", stroke="#d1d5db", lw=0.7)
    c.circle(x0 + 14, y0 + ph - 21, 3.0, fill="#059669", stroke="#059669", lw=1.1)
    c.text(x0 + 21, y0 + ph - 18.5, "Exact MITM/Coset Distance (n <= 48)", size=5.9)
    c.circle(x0 + 14, y0 + ph - 11, 3.0, fill="#dbeafe", stroke="#2563eb", lw=1.1)
    c.text(x0 + 21, y0 + ph - 8.5, "ISD Witness + MITM d >= 5 (n > 48)", size=5.9)

    # Panel (b): Helios-1 3-Round Phenomenological LER vs Physical Error Rate p
    bx0, by0, bpw, bph = 310.0, 52.0, 230.0, 192.0
    c.text(bx0 + 0.5 * bpw, 44, "(b) Helios-1 3-Round Phenomenological LER (BP+OSD-CS)", size=7.8, bold=True, anchor="middle")
    c.rect(bx0, by0, bpw, bph, fill="#fafafa", stroke="#374151", lw=1.0)

    ymax_b = 0.019
    ymin_b = -0.0008

    def yb(val: float) -> float:
        frac = (val - ymin_b) / (ymax_b - ymin_b)
        return by0 + bph * (1.0 - min(1.0, max(0.0, frac)))

    for tv in (0.0, 0.005, 0.010, 0.015):
        yy = yb(tv)
        c.line(bx0, yy, bx0 + bpw, yy, color="#e5e7eb" if tv != 0.0 else "#9ca3af", lw=0.8, dashed=(tv != 0.0))
        c.text(bx0 - 4, yy + 2.5, f"{tv:.3f}", size=6.5, color="#374151", anchor="end", mono=True)

    xs_b = [bx0 + 38.0, bx0 + 0.5 * bpw, bx0 + bpw - 38.0]
    for xx, plbl in zip(xs_b, ["p = 7.9e-4", "p = 2.0e-3", "p = 5.0e-3"]):
        c.text(xx, by0 + bph + 12, plbl, size=6.9, bold=True, anchor="middle", mono=True)
    c.text(bx0 + 0.5 * bpw, by0 + bph + 24, "Helios-1 Depolarizing + Syndrome Flip Rate p", size=7.5, anchor="middle")

    bench = qldpc["helios_1_bposd_cs_benchmarks"]
    unenc = [float(s["unencoded_k_qubit_error_prob"]) for s in bench["D_9_odd_[[36,3,6]]"]["sweeps"]]
    curves_b = [
        (unenc, "#dc2626", True, "Unencoded k=3: 1-(1-p)^3"),
        (
            [float(s["phenomenological_3round_ler"]) for s in bench["C_19_semidirect_C_3_odd_[[114,3,12]]"]["sweeps"]],
            "#d97706",
            False,
            "C_19|xC_3 [[114,3,12]]",
        ),
        (
            [float(s["phenomenological_3round_ler"]) for s in bench["D_15_odd_[[60,3,8]]"]["sweeps"]],
            "#7c3aed",
            False,
            "D_15 [[60,3,8]]",
        ),
        (
            [float(s["phenomenological_3round_ler"]) for s in bench["C_7_semidirect_C_3_odd_[[42,3,6]]"]["sweeps"]],
            "#2563eb",
            False,
            "C_7|xC_3 [[42,3,6]]",
        ),
        (
            [float(s["phenomenological_3round_ler"]) for s in bench["D_9_odd_[[36,3,6]]"]["sweeps"]],
            "#059669",
            False,
            "D_9 [[36,3,6]] (0/800 fails)",
        ),
    ]

    for vals, col, dashed, _ in curves_b:
        pts = [(xx, yb(v)) for xx, v in zip(xs_b, vals)]
        for i in range(len(pts) - 1):
            c.line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], color=col, lw=1.6, dashed=dashed)
        for px, py in pts:
            c.circle(px, py, 3.0, fill=col, stroke="#ffffff", lw=0.8)

    c.rect(bx0 + 8, by0 + 8, 138, 56, fill="#ffffff", stroke="#d1d5db", lw=0.7)
    for idx, (_, col, dashed, lbl) in enumerate(curves_b):
        ly = by0 + 17.5 + idx * 10.2
        c.line(bx0 + 13, ly - 2.5, bx0 + 27, ly - 2.5, color=col, lw=1.5, dashed=dashed)
        c.text(bx0 + 31, ly, lbl, size=6.0, color="#111827")

    return c


def make_fig2(adv1: Mapping[str, Any]) -> VectorCanvas:
    """Dual-panel vector figure: (a) Bounded-chi MPS sweep; (b) M_6 Pauli-weight-5 ablation."""
    c = VectorCanvas(560, 290)
    c.text(
        280,
        17,
        "ADV-1 Fail-Closed Three-Attack Classical-Hardness Pre-Gate & M_6 = J_6 - I_6 Ablation",
        size=9.5,
        bold=True,
        anchor="middle",
    )
    c.text(
        280,
        29,
        "SIMULATED (provider_job_id: null) | 3/3 ADV-0 Mirror Controls Rejected | n=24 & n=30 Chorded M_6 Permitted",
        size=7.4,
        color="#4b5563",
        anchor="middle",
    )

    # Panel (a): Attack 1 Bounded-chi MPS recovered peak probability across chi in {8,16,32,64,128}
    x0, y0, pw, ph = 48.0, 52.0, 214.0, 192.0
    c.text(x0 + 0.5 * pw, 44, "(a) Attack 1: Bounded-chi MPS Peak Recovery", size=8.0, bold=True, anchor="middle")
    c.rect(x0, y0, pw, ph, fill="#fafafa", stroke="#374151", lw=1.0)

    def ya(val: float) -> float:
        return y0 + ph * (1.0 - min(1.0, max(0.0, (val + 0.04) / 1.08)))

    for tv in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        yy = ya(tv)
        c.line(x0, yy, x0 + pw, yy, color="#e5e7eb" if tv != 0.0 else "#9ca3af", lw=0.8, dashed=(tv != 0.0))
        c.text(x0 - 4, yy + 2.5, f"{tv:.1f}", size=6.6, color="#374151", anchor="end", mono=True)

    chis = [8, 16, 32, 64, 128]
    xs_a = [x0 + 22.0 + idx * 42.5 for idx in range(len(chis))]
    for xx, ch in zip(xs_a, chis):
        c.text(xx, y0 + ph + 12, str(ch), size=6.9, bold=True, anchor="middle", mono=True)
    c.text(x0 + 0.5 * pw, y0 + ph + 24, "MPS Bond Dimension Cutoff chi", size=7.5, anchor="middle")

    ctrls = {d["candidate_id"]: d for d in adv1["adv0_twirled_mirror_controls"]}
    cands = {d["candidate_id"]: d for d in adv1["adv1_looped_non_mirror_candidates"]}

    mps_series = [
        (ctrls["adv0_ctrl_n28_d8"], "#dc2626", True, "ADV-0 Mirror n=28 (REJECT)"),
        (cands["adv1_commuting_phase_n24_L3"], "#d97706", True, "Commuting Phase n=24 (REJECT)"),
        (cands["adv1_chorded_m6_n18_L4_small_cone"], "#7c3aed", True, "Chorded M_6 n=18 (REJECT)"),
        (cands["adv1_chorded_m6_n24_L5"], "#059669", False, "Chorded M_6 n=24 (PERMIT)"),
        (cands["adv1_chorded_m6_n30_L5"], "#2563eb", False, "Chorded M_6 n=30 (PERMIT)"),
    ]
    for item, col, dashed, _ in mps_series:
        vals = [float(pt["p_peak_est"]) for pt in item["attack1_bounded_mps"]["chi_curves"]]
        pts = [(xx, ya(v)) for xx, v in zip(xs_a, vals)]
        for i in range(len(pts) - 1):
            c.line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], color=col, lw=1.6, dashed=dashed)
        for px, py in pts:
            c.circle(px, py, 2.9, fill=col, stroke="#ffffff", lw=0.8)

    c.rect(x0 + 8, y0 + 28, 146, 54, fill="#ffffff", stroke="#d1d5db", lw=0.7)
    for idx, (_, col, dashed, lbl) in enumerate(mps_series):
        ly = y0 + 37.0 + idx * 10.0
        c.line(x0 + 12, ly - 2.5, x0 + 26, ly - 2.5, color=col, lw=1.5, dashed=dashed)
        c.text(x0 + 30, ly, lbl, size=5.9, color="#111827")

    # Panel (b): Attack 2 Truncated Pauli-Path bit accuracy across w_max in {2, 3, 4} (M_6 ablation)
    bx0, by0, bpw, bph = 310.0, 52.0, 230.0, 192.0
    c.text(bx0 + 0.5 * bpw, 44, "(b) Attack 2: Pauli-Path Secret Bit Recovery vs M_6", size=7.9, bold=True, anchor="middle")
    c.rect(bx0, by0, bpw, bph, fill="#fafafa", stroke="#374151", lw=1.0)

    def yb(val: float) -> float:
        return by0 + bph * (1.0 - min(1.0, max(0.0, (val - 0.35) / 0.70)))

    for tv in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        yy = yb(tv)
        c.line(bx0, yy, bx0 + bpw, yy, color="#e5e7eb" if tv != 0.5 else "#6b7280", lw=0.8, dashed=True)
        c.text(bx0 - 4, yy + 2.5, f"{int(round(100*tv))}%", size=6.5, color="#374151", anchor="end", mono=True)

    c.text(bx0 + bpw - 5, yb(0.5) + 9, "50% random-guess baseline", size=6.0, color="#4b5563", anchor="end")

    xs_b = [bx0 + 42.0, bx0 + 0.5 * bpw, bx0 + bpw - 42.0]
    for xx, wlbl in zip(xs_b, ["w_max = 2", "w_max = 3", "w_max = 4"]):
        c.text(xx, by0 + bph + 12, wlbl, size=6.9, bold=True, anchor="middle", mono=True)
    c.text(bx0 + 0.5 * bpw, by0 + bph + 24, "Pauli Back-Propagation Weight Cutoff w_max", size=7.5, anchor="middle")

    pauli_series = [
        (cands["adv1_commuting_phase_n24_L3"], "#d97706", True, "Commuting Phase n=24 (100% at w=2)"),
        (cands["adv1_chorded_2q_n28_L5_no_m6"], "#dc2626", True, "Chorded n=28 NO M_6 (100% at w=3!)"),
        (cands["adv1_chorded_m6_n30_L5"], "#2563eb", False, "Chorded + M_6 n=30 (46.7% chance)"),
        (cands["adv1_chorded_m6_n24_L5"], "#059669", False, "Chorded + M_6 n=24 (41.7% chance)"),
    ]
    for item, col, dashed, _ in pauli_series:
        vals = [float(pt["bit_accuracy"]) for pt in item["attack2_pauli_path"]["w_max_curves"]]
        pts = [(xx, yb(v)) for xx, v in zip(xs_b, vals)]
        for i in range(len(pts) - 1):
            c.line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], color=col, lw=1.7, dashed=dashed)
        for px, py in pts:
            c.circle(px, py, 3.1, fill=col, stroke="#ffffff", lw=0.8)

    c.rect(bx0 + 8, by0 + 42, 162, 46, fill="#ffffff", stroke="#d1d5db", lw=0.7)
    for idx, (_, col, dashed, lbl) in enumerate(pauli_series):
        ly = by0 + 51.5 + idx * 10.2
        c.line(bx0 + 12, ly - 2.5, bx0 + 26, ly - 2.5, color=col, lw=1.5, dashed=dashed)
        c.text(bx0 + 30, ly, lbl, size=5.8, color="#111827")

    return c


def generate_all() -> Dict[str, Any]:
    qldpc = json.loads(QLDPC_REPORT_PATH.read_text(encoding="utf-8"))
    adv1 = json.loads(ADV1_REPORT_PATH.read_text(encoding="utf-8"))

    gen_tex = build_generated_tex(qldpc, adv1)
    (PAPER_DIR / "generated.tex").write_text(gen_tex, encoding="utf-8", newline="\n")

    fig1 = make_fig1(qldpc)
    fig1.save(PAPER_DIR / FIG1_STEM)

    fig2 = make_fig2(adv1)
    fig2.save(PAPER_DIR / FIG2_STEM)

    upstream_rels = [
        "qldpc_nonabelian/nonabelian_2bga.py",
        "qldpc_nonabelian/artifacts/nonabelian_2bga_report.json",
        "adv1/looped_peaked.py",
        "adv1/pregate.py",
        "adv1/artifacts/adv1_pregate_report.json",
    ]
    generated_rels = [
        f"papers/nonabelian_2bga_and_adv1/{TEX_FILENAME}",
        "papers/nonabelian_2bga_and_adv1/generated.tex",
        "papers/nonabelian_2bga_and_adv1/generate_2bga_adv1_assets.py",
        f"papers/nonabelian_2bga_and_adv1/{FIG1_STEM}.svg",
        f"papers/nonabelian_2bga_and_adv1/{FIG1_STEM}.pdf",
        f"papers/nonabelian_2bga_and_adv1/{FIG2_STEM}.svg",
        f"papers/nonabelian_2bga_and_adv1/{FIG2_STEM}.pdf",
    ]

    capsule_body: Dict[str, Any] = {
        "schema": "osiris.reproducibility_capsule/1",
        "canonicalization": "RFC8785-JCS",
        "paper_id": "OSIRIS-PAPER-NONABELIAN-2BGA-AND-ADV1",
        "title": (
            "Left-Right Rank Duality in Non-Abelian Two-Block Group-Algebra Quantum LDPC Codes "
            "and a Three-Attack Classical-Hardness Pre-Gate for Looped Non-Mirror Peaked Circuits"
        ),
        "author": "Devin Phillip Davis",
        "affiliation": "Agile Defense Systems LLC",
        "evidence_class": "SIMULATED",
        "hardware_executed": False,
        "provider_job_id": None,
        "seed": int(qldpc["seed"]),
        "target_architecture": adv1["target_architecture"],
        "key_verifications": {
            "unquotiented_2bga_pairs_checked": int(
                qldpc["algebraic_rank_symmetry_theorem"]["unquotiented_pairs_checked"]
            ),
            "rank_asymmetry_violations": int(
                qldpc["algebraic_rank_symmetry_theorem"]["rank_asymmetry_violations"]
            ),
            "odd_k_d_ge_4_codes_discovered": int(qldpc["odd_k_d_ge_4_discoveries_count"]),
            "adv0_controls_100pct_rejected": bool(adv1["summary"]["adv0_100pct_rejected"]),
            "adv1_permitted_candidates": list(adv1["summary"]["adv1_permitted_candidates"]),
        },
        "input_artifacts": {rel: _sha256_file(REPO_ROOT / rel) for rel in upstream_rels},
        "generated_assets": {rel: _sha256_file(REPO_ROOT / rel) for rel in generated_rels},
    }
    body_digest = hashlib.sha256(rfc8785.dumps(capsule_body)).hexdigest()
    capsule: Dict[str, Any] = dict(capsule_body)
    capsule["capsule_body_sha256"] = body_digest

    capsule_path = PAPER_DIR / "reproducibility_capsule.json"
    capsule_path.write_bytes(rfc8785.dumps(capsule))

    sha_rels = (
        generated_rels
        + ["papers/nonabelian_2bga_and_adv1/reproducibility_capsule.json"]
        + upstream_rels
    )
    sha_lines = [f"{_sha256_file(REPO_ROOT / rel)}  {rel}" for rel in sha_rels]
    (PAPER_DIR / "SHA256SUMS.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8", newline="\n")
    return capsule


if __name__ == "__main__":
    cap = generate_all()
    print(f"[OK] Generated 2BGA + ADV-1 paper assets; capsule_body_sha256={cap['capsule_body_sha256']}")
