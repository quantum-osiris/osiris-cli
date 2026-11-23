#!/usr/bin/env python3
"""Deterministic asset generator for the WP2 27-qubit debiased optimal-transport paper.

Reads strictly from verified repository artifacts:
  - experiments/wp2_27q_fez/physical_wp2_27q_fez_results.json (HARDWARE_ATTESTED, dc1tqvinj2vs73dfrts0)
  - experiments/wp2_27q_fez/wp2_27q_fez_staging_manifest.json
  - experiments/wp2_27q_fez/wp2_27q_fez_calibration_snapshot.json
  - experiments/wp2_debiased_sinkhorn/artifacts/wp2_debiased_ot_report.json

Emits deterministically:
  1. generated.tex (all empirical macros and formatted REVTeX 4.2 tables; zero hand-typed floats)
  2. fig1_27q_fez_sparsity_collapse_vs_eq7.svg and .pdf
  3. fig2_coherent_drift_calibration_and_heron_rescoring.svg and .pdf
  4. reproducibility_capsule.json (RFC 8785 JCS canonical JSON)
  5. SHA256SUMS.txt
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import rfc8785

PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parent.parent

FEZ_DIR = REPO_ROOT / "experiments" / "wp2_27q_fez"
OT_REPORT_PATH = (
    REPO_ROOT / "experiments" / "wp2_debiased_sinkhorn" / "artifacts" / "wp2_debiased_ot_report.json"
)
TEX_FILENAME = "wp2_27q_debiased_ot_fez.tex"
FIG1_STEM = "fig1_27q_fez_sparsity_collapse_vs_eq7"
FIG2_STEM = "fig2_coherent_drift_calibration_and_heron_rescoring"


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
    cat_id = add_obj(f"<< /Type /Catalog /Pages {pages_idx} 0 R >>".encode("ascii"))

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


def _find_sweep_cell(
    cells: Sequence[Mapping[str, Any]], family: str, theta: float, eps: float
) -> Mapping[str, Any]:
    for c in cells:
        if (
            c["circuit_family"] == family
            and math.isclose(float(c["theta_rad"]), theta, abs_tol=1e-6)
            and math.isclose(float(c["eps_depol"]), eps, abs_tol=1e-6)
        ):
            return c
    raise KeyError(f"Sweep cell not found: family={family}, theta={theta}, eps={eps}")


def build_generated_tex(
    fez: Mapping[str, Any],
    manifest: Mapping[str, Any],
    cal: Mapping[str, Any],
    ot_rep: Mapping[str, Any],
) -> str:
    lines: List[str] = [
        "% Generated deterministically by generate_wp2_paper_assets.py -- do not edit by hand."
    ]
    seen_names: set[str] = set()

    def mac(name: str, val: Any) -> None:
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError(f"Invalid LaTeX macro name (must be ASCII letters only): {name!r}")
        if name in seen_names:
            raise ValueError(f"Duplicate LaTeX macro name: {name!r}")
        seen_names.add(name)
        lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    # Protocol & hardware execution identifiers
    mac("WpFezJobId", fez["provider_job_id"])
    mac("WpFezBackend", fez["backend"].replace("_", "\\_"))
    mac("WpFezQubits", int(manifest["n_qubits"]))
    mac("WpFezEdges", int(manifest["n_qubits"]) - 1)
    mac("WpFezBlocks", int(manifest["n_blocks"]))
    mac("WpFezPubsPerBlock", int(manifest["pubs_per_block"]))
    mac("WpFezTotalPubs", int(manifest["total_pubs"]))
    mac("WpFezShotsPerPub", _fmt_int(int(manifest["shots_per_pub"])))
    mac("WpFezTotalShots", _fmt_int(int(manifest["total_shots"])))
    mac("WpFezPooledShots", _fmt_int(int(fez["arm_theta_summaries"]["NB11_REPAIRED_27Q"][0]["pooled_shots"])))
    mac("WpFezCzCount", int(fez["constant_cz_count"]))
    mac("WpFezSxCount", int(fez["constant_sx_count"]))
    mac("WpFezQpuSeconds", int(fez["job_metadata"]["usage"]["qpu_charge_time_seconds"]))

    km = cal["konig_matching_sizes"]
    mac("WpKonigSizes", f"{km[0]}, {km[1]}, \\text{{and }} {km[2]}")
    mac("WpCalMeanRoPct", f"{100.0 * float(cal['mean_readout_error']):.2f}")
    mac("WpCalMaxRoPct", f"{100.0 * float(cal['max_readout_error']):.2f}")
    mac("WpEmpMeanRoPct", f"{100.0 * float(fez['readout_sentinel_27q']['mean_empirical_ro_error']):.2f}")
    mac("WpEmpMaxRoPct", f"{100.0 * float(fez['readout_sentinel_27q']['max_empirical_ro_error']):.2f}")
    mac("WpHiddenRoFaults", len(fez["readout_sentinel_27q"]["hidden_fault_qubits_gt_10pct"]))
    mac("WpLayoutList", ", ".join(str(q) for q in fez["layout_27q"]))

    # Estimator & preregistration parameters
    eps_sk = float(ot_rep["epsilon_sinkhorn_default"])
    mac("WpSinkhornEps", f"{eps_sk:.2f}")
    q27 = int(manifest["n_qubits"])
    eq7_unif_27q = math.sqrt(0.25 + 1.0 / (4.0 * q27))
    mac("WpEqSevenUnifTwentySeven", f"{eq7_unif_27q:.5f}")

    arm_a = fez["arm_theta_summaries"]["NB11_REPAIRED_27Q"]
    arm_b = fez["arm_theta_summaries"]["ENTANGLED_BRICKWALL_27Q"]

    mac("WpThetaOne", f"{float(arm_a[0]['theta_rad']):.3f}")
    mac("WpThetaStar", f"{float(arm_a[1]['theta_rad']):.3f}")
    mac("WpThetaThree", f"{float(arm_a[2]['theta_rad']):.3f}")

    # Arm A (NB11_REPAIRED_27Q) prose scalars
    mac("WpArmAEntOne", f"{float(arm_a[0]['empirical_shannon_entropy_bits']):.2f}")
    mac("WpArmAEntStar", f"{float(arm_a[1]['empirical_shannon_entropy_bits']):.2f}")
    mac("WpArmAEntThree", f"{float(arm_a[2]['empirical_shannon_entropy_bits']):.2f}")
    mac("WpArmAUniqOne", _fmt_int(int(arm_a[0]["unique_bitstrings_8192"])))
    mac("WpArmAUniqStar", _fmt_int(int(arm_a[1]["unique_bitstrings_8192"])))
    mac("WpArmAUniqThree", _fmt_int(int(arm_a[2]["unique_bitstrings_8192"])))
    mac("WpArmAEqSevenWOne", f"{float(arm_a[0]['mean_defective_eq7_w2']):.4f}")
    mac("WpArmAEqSevenWStar", f"{float(arm_a[1]['mean_defective_eq7_w2']):.4f}")
    mac("WpArmAEqSevenWThree", f"{float(arm_a[2]['mean_defective_eq7_w2']):.4f}")
    mac("WpArmAEqSevenROne", f"{float(arm_a[0]['mean_defective_eq7_R_hat']):.4f}")
    mac("WpArmAEqSevenRStar", f"{float(arm_a[1]['mean_defective_eq7_R_hat']):.4f}")
    mac("WpArmAEqSevenRThree", f"{float(arm_a[2]['mean_defective_eq7_R_hat']):.4f}")
    mac("WpArmARawWOne", f"{float(arm_a[0]['mean_raw_plugin_lp_w2']):.4f}")
    mac("WpArmARawWStar", f"{float(arm_a[1]['mean_raw_plugin_lp_w2']):.4f}")
    mac("WpArmARawWThree", f"{float(arm_a[2]['mean_raw_plugin_lp_w2']):.4f}")
    mac("WpArmARawROne", f"{float(arm_a[0]['mean_raw_plugin_R_hat']):.4f}")
    mac("WpArmARawRStar", f"{float(arm_a[1]['mean_raw_plugin_R_hat']):.4f}")
    mac("WpArmARawRThree", f"{float(arm_a[2]['mean_raw_plugin_R_hat']):.4f}")
    mac("WpArmADebSkOne", f"{float(arm_a[0]['mean_matched_debiased_sinkhorn_S_eps']):+.6f}")
    mac("WpArmADebSkStar", f"{float(arm_a[1]['mean_matched_debiased_sinkhorn_S_eps']):+.6f}")
    mac("WpArmADebSkThree", f"{float(arm_a[2]['mean_matched_debiased_sinkhorn_S_eps']):+.6f}")
    mac("WpArmADebLpOne", f"{float(arm_a[0]['mean_matched_debiased_lp_w2_sq']):+.6f}")
    mac("WpArmADebLpStar", f"{float(arm_a[1]['mean_matched_debiased_lp_w2_sq']):+.6f}")
    mac("WpArmADebLpThree", f"{float(arm_a[2]['mean_matched_debiased_lp_w2_sq']):+.6f}")

    # Arm B (ENTANGLED_BRICKWALL_27Q) prose scalars
    mac("WpArmBEqSevenWOne", f"{float(arm_b[0]['mean_defective_eq7_w2']):.4f}")
    mac("WpArmBEqSevenWStar", f"{float(arm_b[1]['mean_defective_eq7_w2']):.4f}")
    mac("WpArmBEqSevenWThree", f"{float(arm_b[2]['mean_defective_eq7_w2']):.4f}")
    mac("WpArmBRawWOne", f"{float(arm_b[0]['mean_raw_plugin_lp_w2']):.4f}")
    mac("WpArmBRawWStar", f"{float(arm_b[1]['mean_raw_plugin_lp_w2']):.4f}")
    mac("WpArmBRawWThree", f"{float(arm_b[2]['mean_raw_plugin_lp_w2']):.4f}")
    mac("WpArmBDebSkOne", f"{float(arm_b[0]['mean_matched_debiased_sinkhorn_S_eps']):+.6f}")
    mac("WpArmBDebSkStar", f"{float(arm_b[1]['mean_matched_debiased_sinkhorn_S_eps']):+.6f}")
    mac("WpArmBDebSkThree", f"{float(arm_b[2]['mean_matched_debiased_sinkhorn_S_eps']):+.6f}")

    # Hypotheses H1, H2, H3 & controlled drift calibration
    h1 = fez["hypotheses"]["H1_weed_bach_sparsity_bias_eliminated"]
    h3 = fez["hypotheses"]["H3_monotonic_coherent_drift_detection_gt_10sigma"]
    cal_b = fez["controlled_drift_calibration_arm_b"]
    mac("WpHOneThreshold", f"{float(h1['threshold']):.3f}")
    mac("WpHOneMaxAbs", f"{float(h1['max_abs_interblock_matched_S_eps']):.6f}")
    mac("WpDeltaSmall", "0.20")
    mac("WpDeltaLarge", "0.45")
    mac("WpSigmaThreshold", "10")
    mac("WpDriftNullMean", f"{float(cal_b['intra_block_aa_null_matched_S_eps_mean']):+.6f}")
    mac("WpDriftNullSe", f"{float(cal_b['intra_block_aa_null_se']):.6f}")
    mac("WpDriftSmallMean", f"{float(cal_b['delta_theta_0p20_matched_S_eps_mean']):+.6f}")
    mac("WpDriftSmallSe", f"{float(cal_b['delta_theta_0p20_se']):.6f}")
    mac("WpDriftLargeMean", f"{float(cal_b['delta_theta_0p45_matched_S_eps_mean']):+.6f}")
    mac("WpDriftLargeSe", f"{float(cal_b['delta_theta_0p45_se']):.6f}")
    mac("WpDriftLargeZ", f"{float(h3['z_score_0p45']):.2f}")

    # Table I rows: 27Q ibm_fez inter-block A/A across Arm A and Arm B
    t1_rows: List[str] = []
    for arm_lbl, entries in (
        ("Arm A (\\texttt{NB11\\_REPAIRED})", arm_a),
        ("Arm B (\\texttt{BRICKWALL})", arm_b),
    ):
        for e in entries:
            raw_sk = 0.5 * (
                float(e["pair_b0_b1"]["raw_sinkhorn_ot_eps"])
                + float(e["pair_b2_b3"]["raw_sinkhorn_ot_eps"])
            )
            t1_rows.append(
                f"{arm_lbl} & {float(e['theta_rad']):.3f} & "
                f"{_fmt_int(int(e['unique_bitstrings_8192']))} & "
                f"{float(e['empirical_shannon_entropy_bits']):.2f} & "
                f"{float(e['mean_defective_eq7_w2']):.4f} & "
                f"{float(e['mean_raw_plugin_lp_w2']):.4f} & "
                f"{raw_sk:.4f} & "
                f"{float(e['mean_matched_debiased_lp_w2_sq']):+.6f} & "
                f"{float(e['mean_matched_debiased_sinkhorn_S_eps']):+.6f}\\\\"
            )
        t1_rows.append("\\addlinespace")
    mac("WpFezSweepRows", "\n".join(t1_rows[:-1]))

    # Part A Simulated High-Entropy & 2^96 Uniform Stress Test
    pa = ot_rep["part_a_simulated_high_entropy_benchmark"]
    u96 = pa["uniform_2_pow_96_stress_test"]
    ngate = pa["null_unbiasedness_gate"]
    dgate = pa["drift_detection_gate_eps_0_05"]
    mac("WpSimShots", int(pa["shots_per_replicate"]))
    mac("WpSimNullThreshold", f"{float(ngate['threshold_abs_mean']):.3f}")
    mac("WpSimMaxAbsMatchedSk", f"{float(ngate['max_abs_mean_debiased_sinkhorn_matched_eps0']):.6f}")
    mac("WpSimMaxAbsMatchedLp", f"{float(ngate['max_abs_mean_debiased_lp_matched_eps0']):.6f}")
    mac("WpSimMinDriftZ", f"{float(dgate['min_z_score_across_all_widths_and_thetas']):.2f}")
    mac("WpUnifNinetySixRawLp", f"{float(u96['raw_plugin_lp_w2_sq']['mean']):.5f}")
    mac("WpUnifNinetySixRawZ", f"{float(u96['raw_plugin_lp_w2_sq']['z_score']):.2f}")
    mac("WpUnifNinetySixDebLp", f"{float(u96['debiased_lp_w2_sq_matched']['mean']):+.6f}")
    mac("WpUnifNinetySixDebSk", f"{float(u96['debiased_sinkhorn_matched']['mean']):+.6f}")
    mac("WpUnifNinetySixDebSkZ", f"{float(u96['debiased_sinkhorn_matched']['z_score']):+.2f}")

    # Table II rows: Simulated 27Q / 50Q / 96Q + Uniform({0,1}^96) + live Arm B drift
    cells = pa["sweep_cells"]
    t2_rows: List[str] = []
    sim_specs = [
        ("\\texttt{SIM} 27Q \\texttt{NB-11}", "NB-11_27Q_repaired", 2.618, 0.0),
        ("\\texttt{SIM} 27Q \\texttt{NB-11}", "NB-11_27Q_repaired", 3.118, 0.0),
        ("\\texttt{SIM} 27Q \\texttt{NB-11}", "NB-11_27Q_repaired", 2.618, 0.05),
        ("\\texttt{SIM} 50Q \\texttt{H2-1}", "WP2a_H2_1_50Q", 3.118, 0.0),
        ("\\texttt{SIM} 50Q \\texttt{H2-1}", "WP2a_H2_1_50Q", 2.618, 0.05),
        ("\\texttt{SIM} 96Q \\texttt{Helios-1}", "WP2b_Helios_1_96Q", 3.118, 0.0),
        ("\\texttt{SIM} 96Q \\texttt{Helios-1}", "WP2b_Helios_1_96Q", 2.618, 0.05),
    ]
    for lbl, fam, th, ep in sim_specs:
        c = _find_sweep_cell(cells, fam, th, ep)
        t2_rows.append(
            f"{lbl} ($\\theta={th:.3f},\\,\\epsilon={ep:.2f}$) & "
            f"{int(c['n_qubits'])} & {float(c['ideal_shannon_entropy_bits']):.2f} & "
            f"{float(c['raw_plugin_lp_w2_sq']['mean']):.5f} & "
            f"{float(c['debiased_lp_w2_sq_matched']['mean']):+.6f} & "
            f"{float(c['debiased_sinkhorn_matched']['mean']):+.6f} & "
            f"{float(c['debiased_sinkhorn_matched']['z_score']):+.2f}\\\\"
        )
    t2_rows.append(
        f"\\texttt{{SIM}} $\\mathrm{{Unif}}(\\{{0,1\\}}^{{96}})$ null & "
        f"96 & 96.00 & {float(u96['raw_plugin_lp_w2_sq']['mean']):.5f} & "
        f"{float(u96['debiased_lp_w2_sq_matched']['mean']):+.6f} & "
        f"{float(u96['debiased_sinkhorn_matched']['mean']):+.6f} & "
        f"{float(u96['debiased_sinkhorn_matched']['z_score']):+.2f}\\\\"
    )
    t2_rows.append("\\addlinespace")
    t2_rows.append(
        f"\\texttt{{HW}} \\texttt{{ibm\\_fez}} Arm B A/A ($\\Delta\\theta=0$) & "
        f"27 & {float(arm_b[1]['empirical_shannon_entropy_bits']):.2f} & -- & -- & "
        f"{float(cal_b['intra_block_aa_null_matched_S_eps_mean']):+.6f} & "
        f"{float(cal_b['intra_block_aa_null_matched_S_eps_mean']) / float(cal_b['intra_block_aa_null_se']):+.2f}\\\\"
    )
    z020 = float(cal_b["delta_theta_0p20_matched_S_eps_mean"]) / float(cal_b["delta_theta_0p20_se"])
    t2_rows.append(
        f"\\texttt{{HW}} \\texttt{{ibm\\_fez}} Arm B ($\\Delta\\theta=0.20$) & "
        f"27 & -- & -- & -- & "
        f"{float(cal_b['delta_theta_0p20_matched_S_eps_mean']):+.6f} & "
        f"{z020:+.2f}\\\\"
    )
    t2_rows.append(
        f"\\texttt{{HW}} \\texttt{{ibm\\_fez}} Arm B ($\\Delta\\theta=0.45$) & "
        f"27 & -- & -- & -- & "
        f"{float(cal_b['delta_theta_0p45_matched_S_eps_mean']):+.6f} & "
        f"+{float(h3['z_score_0p45']):.2f}\\\\"
    )
    mac("WpDriftAndSimRows", "\n".join(t2_rows))

    # Part B Hardware Re-Scoring: ibm_marrakesh & OSIRIS-QEC-001
    pb = ot_rep["part_b_hardware_attested_heron_r2_rescoring"]
    marr = pb["marrakesh_dd_rescoring"]
    qec = pb["fez_qec001_rescoring"]

    mac("WpMarrJobId", marr["provider_job_id"])
    mac("WpMarrQubits", int(marr["n_qubits"]))
    mac("WpMarrCircuits", int(marr["total_circuits"]))
    mac("WpMarrTotalShots", _fmt_int(int(marr["total_shots"])))

    mac("WpQecJobId", qec["provider_job_id"])
    mac("WpQecCircuits", int(qec["total_circuits"]))
    mac("WpQecTotalShots", _fmt_int(int(qec["total_shots"])))

    marr_aa = marr["aa_and_split_half_self_consistency"]
    max_marr_aa_sk = max(abs(float(x["inter_rep_debiased_sinkhorn"])) for x in marr_aa)
    max_marr_aa_eq7 = max(float(x["inter_rep_defective_eq7_cost"]) for x in marr_aa)
    mac("WpMarrMaxAaAbsSk", f"{max_marr_aa_sk:.6f}")
    mac("WpMarrMaxAaEqSeven", f"{max_marr_aa_eq7:.4f}")

    def _marr_contrast(t_us: float, a: str, b: str) -> Mapping[str, Any]:
        for row in marr["dd_cross_condition_contrasts"]:
            if (
                math.isclose(float(row["T_us"]), t_us, abs_tol=1e-6)
                and row["arm_a"] == a
                and row["arm_b"] == b
            ):
                return row
        raise KeyError(f"Marrakesh contrast not found: {t_us}, {a}, {b}")

    mc_16_stag_none = _marr_contrast(16.0, "xy8_stag", "none")
    mc_16_stag_xy4s = _marr_contrast(16.0, "xy8_stag", "xy4x2_stag")
    mc_32_stag_none = _marr_contrast(32.0, "xy8_stag", "none")
    mc_32_stag_xy4 = _marr_contrast(32.0, "xy8_stag", "xy4x2")
    mc_32_stag_xy4s = _marr_contrast(32.0, "xy8_stag", "xy4x2_stag")

    mac("WpMarrSixteenStagNoneSk", f"{float(mc_16_stag_none['debiased_sinkhorn_matched']):.5f}")
    mac("WpMarrSixteenStagPairSk", f"{float(mc_16_stag_xy4s['debiased_sinkhorn_matched']):+.6f}")
    mac("WpMarrThirtyTwoStagNoneSk", f"{float(mc_32_stag_none['debiased_sinkhorn_matched']):.5f}")
    mac("WpMarrThirtyTwoStagStdSk", f"{float(mc_32_stag_xy4['debiased_sinkhorn_matched']):.5f}")
    mac("WpMarrThirtyTwoStagPairSk", f"{float(mc_32_stag_xy4s['debiased_sinkhorn_matched']):+.6f}")

    qec_evals = qec["evaluations"]
    qec_z_49_aa = next(
        x
        for x in qec_evals
        if x["comparison_type"] == "inter_block_AA_full_register"
        and x["code_basis"] == "bit_flip_z"
        and int(x["register_bits"]) == 49
    )
    qec_x_49_aa = next(
        x
        for x in qec_evals
        if x["comparison_type"] == "inter_block_AA_full_register"
        and x["code_basis"] == "phase_flip_x"
        and int(x["register_bits"]) == 49
    )
    qec_49_hot = next(
        x for x in qec_evals if x["comparison_type"] == "spectator_t2_hotspot_basis_contrast_full_49bit"
    )
    qec_z_r7 = next(
        x
        for x in qec_evals
        if x["comparison_type"] == "round_scaling_7bit_data_marginal"
        and x["code_basis"] == "bit_flip_z"
        and int(x["target_rounds"]) == 7
    )
    qec_x_r3 = next(
        x
        for x in qec_evals
        if x["comparison_type"] == "round_scaling_7bit_data_marginal"
        and x["code_basis"] == "phase_flip_x"
        and int(x["target_rounds"]) == 3
    )
    qec_x_r5 = next(
        x
        for x in qec_evals
        if x["comparison_type"] == "round_scaling_7bit_data_marginal"
        and x["code_basis"] == "phase_flip_x"
        and int(x["target_rounds"]) == 5
    )
    qec_x_r7 = next(
        x
        for x in qec_evals
        if x["comparison_type"] == "round_scaling_7bit_data_marginal"
        and x["code_basis"] == "phase_flip_x"
        and int(x["target_rounds"]) == 7
    )

    mac("WpQecZFortyNineEqSeven", f"{float(qec_z_49_aa['defective_eq7_cost']):.5f}")
    mac("WpQecZFortyNineSk", f"{float(qec_z_49_aa['debiased_sinkhorn_matched']):+.6f}")
    mac("WpQecXFortyNineEqSeven", f"{float(qec_x_49_aa['defective_eq7_cost']):.5f}")
    mac("WpQecXFortyNineRawLp", f"{float(qec_x_49_aa['raw_lp_w2_sq']):.5f}")
    mac("WpQecXFortyNineSk", f"{float(qec_x_49_aa['debiased_sinkhorn_matched']):+.6f}")
    mac("WpQecHotspotSk", f"{float(qec_49_hot['debiased_sinkhorn_matched']):.5f}")
    mac("WpQecHotspotLp", f"{float(qec_49_hot['debiased_lp_w2_sq_matched']):.5f}")
    mac("WpQecZRSevenSk", f"{float(qec_z_r7['debiased_sinkhorn']):+.6f}")
    mac("WpQecXRThreeSk", f"{float(qec_x_r3['debiased_sinkhorn']):.5f}")
    mac("WpQecXRFiveSk", f"{float(qec_x_r5['debiased_sinkhorn']):.5f}")
    mac("WpQecXRSevenSk", f"{float(qec_x_r7['debiased_sinkhorn']):.5f}")

    # Table III rows: Multi-processor hardware rescoring
    marr_none_32_aa = next(
        x for x in marr_aa if math.isclose(float(x["T_us"]), 32.0) and x["condition"] == "none"
    )
    marr_stag_32_aa = next(
        x for x in marr_aa if math.isclose(float(x["T_us"]), 32.0) and x["condition"] == "xy8_stag"
    )
    t3_rows = [
        f"\\texttt{{ibm\\_marrakesh}} $32\\,\\mu\\mathrm{{s}}$ \\texttt{{xy8\\_stag}} A/A & "
        f"8 & 512 & {float(marr_stag_32_aa['inter_rep_defective_eq7_cost']):.5f} & "
        f"{float(marr_stag_32_aa['inter_rep_raw_lp_w2_sq']):.5f} & "
        f"{float(marr_stag_32_aa['inter_rep_debiased_lp_w2_sq_matched']):+.6f} & "
        f"{float(marr_stag_32_aa['inter_rep_debiased_sinkhorn']):+.6f}\\\\",
        f"\\texttt{{ibm\\_marrakesh}} $32\\,\\mu\\mathrm{{s}}$ \\texttt{{none}} A/A & "
        f"8 & 512 & {float(marr_none_32_aa['inter_rep_defective_eq7_cost']):.5f} & "
        f"{float(marr_none_32_aa['inter_rep_raw_lp_w2_sq']):.5f} & "
        f"{float(marr_none_32_aa['inter_rep_debiased_lp_w2_sq_matched']):+.6f} & "
        f"{float(marr_none_32_aa['inter_rep_debiased_sinkhorn']):+.6f}\\\\",
        f"\\texttt{{ibm\\_marrakesh}} $16\\,\\mu\\mathrm{{s}}$ \\texttt{{xy8\\_stag}} vs \\texttt{{xy4x2\\_stag}} & "
        f"8 & 1{{,}}024 & {float(mc_16_stag_xy4s['defective_eq7_cost']):.5f} & "
        f"{float(mc_16_stag_xy4s['raw_lp_w2_sq']):.5f} & "
        f"{float(mc_16_stag_xy4s['debiased_lp_w2_sq_matched']):+.6f} & "
        f"{float(mc_16_stag_xy4s['debiased_sinkhorn_matched']):+.6f}\\\\",
        f"\\texttt{{ibm\\_marrakesh}} $16\\,\\mu\\mathrm{{s}}$ \\texttt{{xy8\\_stag}} vs \\texttt{{none}} & "
        f"8 & 1{{,}}024 & {float(mc_16_stag_none['defective_eq7_cost']):.5f} & "
        f"{float(mc_16_stag_none['raw_lp_w2_sq']):.5f} & "
        f"{float(mc_16_stag_none['debiased_lp_w2_sq_matched']):+.6f} & "
        f"{float(mc_16_stag_none['debiased_sinkhorn_matched']):+.6f}\\\\",
        f"\\texttt{{ibm\\_marrakesh}} $32\\,\\mu\\mathrm{{s}}$ \\texttt{{xy8\\_stag}} vs \\texttt{{xy4x2}} & "
        f"8 & 1{{,}}024 & {float(mc_32_stag_xy4['defective_eq7_cost']):.5f} & "
        f"{float(mc_32_stag_xy4['raw_lp_w2_sq']):.5f} & "
        f"{float(mc_32_stag_xy4['debiased_lp_w2_sq_matched']):+.6f} & "
        f"{float(mc_32_stag_xy4['debiased_sinkhorn_matched']):+.6f}\\\\",
        f"\\texttt{{ibm\\_marrakesh}} $32\\,\\mu\\mathrm{{s}}$ \\texttt{{xy8\\_stag}} vs \\texttt{{none}} & "
        f"8 & 1{{,}}024 & {float(mc_32_stag_none['defective_eq7_cost']):.5f} & "
        f"{float(mc_32_stag_none['raw_lp_w2_sq']):.5f} & "
        f"{float(mc_32_stag_none['debiased_lp_w2_sq_matched']):+.6f} & "
        f"{float(mc_32_stag_none['debiased_sinkhorn_matched']):+.6f}\\\\",
        "\\addlinespace",
        f"\\texttt{{ibm\\_fez}} \\texttt{{QEC-001}} $Z$-sector $d=7,R=7$ A/A & "
        f"49 & 2{{,}}048 & {float(qec_z_49_aa['defective_eq7_cost']):.5f} & "
        f"{float(qec_z_49_aa['raw_lp_w2_sq']):.5f} & "
        f"{float(qec_z_49_aa['debiased_lp_w2_sq_matched']):+.6f} & "
        f"{float(qec_z_49_aa['debiased_sinkhorn_matched']):+.6f}\\\\",
        f"\\texttt{{ibm\\_fez}} \\texttt{{QEC-001}} $X$-sector $d=7,R=7$ A/A & "
        f"49 & 2{{,}}048 & {float(qec_x_49_aa['defective_eq7_cost']):.5f} & "
        f"{float(qec_x_49_aa['raw_lp_w2_sq']):.5f} & "
        f"{float(qec_x_49_aa['debiased_lp_w2_sq_matched']):+.6f} & "
        f"{float(qec_x_49_aa['debiased_sinkhorn_matched']):+.6f}\\\\",
        f"\\texttt{{ibm\\_fez}} \\texttt{{QEC-001}} $Z$ vs $X$ ($q_{{107}}$ $T_2$ hotspot) & "
        f"49 & 2{{,}}048 & {float(qec_49_hot['defective_eq7_cost']):.5f} & "
        f"{float(qec_49_hot['raw_lp_w2_sq']):.5f} & "
        f"{float(qec_49_hot['debiased_lp_w2_sq_matched']):+.6f} & "
        f"{float(qec_49_hot['debiased_sinkhorn_matched']):+.6f}\\\\",
        f"\\texttt{{ibm\\_fez}} \\texttt{{QEC-001}} $Z$-data $R=1\\to 7$ marginal & "
        f"7 & 2{{,}}048 & -- & {float(qec_z_r7['raw_lp_w2_sq']):.5f} & "
        f"{float(qec_z_r7['debiased_lp_w2_sq']):+.6f} & "
        f"{float(qec_z_r7['debiased_sinkhorn']):+.6f}\\\\",
        f"\\texttt{{ibm\\_fez}} \\texttt{{QEC-001}} $X$-data $R=1\\to 7$ marginal & "
        f"7 & 2{{,}}048 & -- & {float(qec_x_r7['raw_lp_w2_sq']):.5f} & "
        f"{float(qec_x_r7['debiased_lp_w2_sq']):+.6f} & "
        f"{float(qec_x_r7['debiased_sinkhorn']):+.6f}\\\\",
    ]
    mac("WpHeronRescoreRows", "\n".join(t3_rows))

    return "\n".join(lines) + "\n"


def make_fig1(fez: Mapping[str, Any], ot_rep: Mapping[str, Any]) -> VectorCanvas:
    """Dual-panel vector figure: (a) Live 27Q ibm_fez A/A sparsity collapse; (b) 27Q/50Q/96Q scaling."""
    c = VectorCanvas(560, 290)
    c.text(
        280,
        17,
        "OSIRIS-WP2-27Q-FEZ (ibm_fez, Job dc1tqvinj2vs73dfrts0) & Multi-Width Hypercube Scaling",
        size=9.8,
        bold=True,
        anchor="middle",
    )
    c.text(
        280,
        29,
        "Elimination of Defective Eq. (7) Self-Distance & Weed-Bach O(S^{-1/Q}) Sparsity Bias via Split-Sample OT",
        size=7.6,
        color="#4b5563",
        anchor="middle",
    )

    # Panel (a): Live 27Q ibm_fez Arm A & Arm B A/A across theta
    x0, y0, pw, ph = 48.0, 52.0, 214.0, 192.0
    c.text(
        x0 + 0.5 * pw,
        44,
        "(a) Live 27Q ibm_fez Inter-Block A/A (52 CZ, 98,304 Shots)",
        size=8.0,
        bold=True,
        anchor="middle",
    )
    c.rect(x0, y0, pw, ph, fill="#fafafa", stroke="#374151", lw=1.0)

    ymax_a = 0.52
    ymin_a = -0.02

    def ya(val: float) -> float:
        frac = (val - ymin_a) / (ymax_a - ymin_a)
        return y0 + ph * (1.0 - min(1.0, max(0.0, frac)))

    for tv in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        yy = ya(tv)
        c.line(
            x0,
            yy,
            x0 + pw,
            yy,
            color="#d1d5db" if tv != 0.0 else "#6b7280",
            lw=0.8,
            dashed=(tv != 0.0),
        )
        c.text(x0 - 4, yy + 2.5, f"{tv:.1f}", size=6.8, color="#374151", anchor="end", mono=True)

    xs_a = [x0 + 36.0, x0 + 0.5 * pw, x0 + pw - 36.0]
    th_lbls = ["2.118", "2.618 (phi^2)", "3.118"]
    for xx, lbl in zip(xs_a, th_lbls):
        c.text(xx, y0 + ph + 13, lbl, size=7.2, bold=True, anchor="middle", mono=True)
    c.text(x0 + 0.5 * pw, y0 + ph + 25, "Preparation Angle theta (rad)", size=7.6, anchor="middle")

    arm_a = fez["arm_theta_summaries"]["NB11_REPAIRED_27Q"]
    arm_b = fez["arm_theta_summaries"]["ENTANGLED_BRICKWALL_27Q"]

    series_a = [
        ([float(e["mean_defective_eq7_w2"]) for e in arm_b], "#dc2626", True, "Eq.(7) Arm B"),
        ([float(e["mean_defective_eq7_w2"]) for e in arm_a], "#ef4444", False, "Eq.(7) Arm A"),
        ([float(e["mean_raw_plugin_lp_w2"]) for e in arm_a], "#d97706", False, "Raw LP W2 Arm A"),
        (
            [float(e["mean_matched_debiased_sinkhorn_S_eps"]) for e in arm_a],
            "#059669",
            False,
            "Debiased S_eps Arm A",
        ),
        (
            [float(e["mean_matched_debiased_sinkhorn_S_eps"]) for e in arm_b],
            "#2563eb",
            True,
            "Debiased S_eps Arm B",
        ),
    ]
    for vals, col, dashed, _ in series_a:
        pts = [(xx, ya(v)) for xx, v in zip(xs_a, vals)]
        for i in range(len(pts) - 1):
            c.line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], color=col, lw=1.6, dashed=dashed)
        for px, py in pts:
            c.circle(px, py, 3.0, fill=col, stroke="#ffffff", lw=0.8)

    leg_y = y0 + 12.0
    for idx, (_, col, dashed, lbl) in enumerate(series_a):
        ly = leg_y + idx * 10.5
        c.line(x0 + 8, ly - 2.5, x0 + 24, ly - 2.5, color=col, lw=1.5, dashed=dashed)
        c.text(x0 + 28, ly, lbl, size=6.3, color="#111827")

    # Panel (b): Multi-width SIMULATED noiseless A/A (27Q, 50Q, 96Q, Uniform 2^96; S=512)
    bx0, by0, bpw, bph = 314.0, 52.0, 224.0, 192.0
    c.text(
        bx0 + 0.5 * bpw,
        44,
        "(b) Noiseless A/A Sparsity Bias Across Width Q (S=512)",
        size=8.0,
        bold=True,
        anchor="middle",
    )
    c.rect(bx0, by0, bpw, bph, fill="#fafafa", stroke="#374151", lw=1.0)

    pa = ot_rep["part_a_simulated_high_entropy_benchmark"]
    cells = pa["sweep_cells"]
    u96 = pa["uniform_2_pow_96_stress_test"]
    b_items = [
        (
            "27Q",
            float(_find_sweep_cell(cells, "NB-11_27Q_repaired", 3.118, 0.0)["raw_plugin_lp_w2_sq"]["mean"]),
            float(_find_sweep_cell(cells, "NB-11_27Q_repaired", 3.118, 0.0)["debiased_sinkhorn_matched"]["mean"]),
        ),
        (
            "50Q",
            float(_find_sweep_cell(cells, "WP2a_H2_1_50Q", 3.118, 0.0)["raw_plugin_lp_w2_sq"]["mean"]),
            float(_find_sweep_cell(cells, "WP2a_H2_1_50Q", 3.118, 0.0)["debiased_sinkhorn_matched"]["mean"]),
        ),
        (
            "96Q",
            float(_find_sweep_cell(cells, "WP2b_Helios_1_96Q", 3.118, 0.0)["raw_plugin_lp_w2_sq"]["mean"]),
            float(_find_sweep_cell(cells, "WP2b_Helios_1_96Q", 3.118, 0.0)["debiased_sinkhorn_matched"]["mean"]),
        ),
        (
            "Unif 2^96",
            float(u96["raw_plugin_lp_w2_sq"]["mean"]),
            float(u96["debiased_sinkhorn_matched"]["mean"]),
        ),
    ]

    ymax_b = 0.145
    ymin_b = -0.008

    def yb(val: float) -> float:
        frac = (val - ymin_b) / (ymax_b - ymin_b)
        return by0 + bph * (1.0 - min(1.0, max(0.0, frac)))

    for tv in (0.0, 0.03, 0.06, 0.09, 0.12):
        yy = yb(tv)
        c.line(
            bx0,
            yy,
            bx0 + bpw,
            yy,
            color="#d1d5db" if tv != 0.0 else "#6b7280",
            lw=0.8,
            dashed=(tv != 0.0),
        )
        c.text(bx0 - 4, yy + 2.5, f"{tv:.2f}", size=6.8, color="#374151", anchor="end", mono=True)

    y_zero_b = yb(0.0)
    for idx, (lbl, raw_v, deb_v) in enumerate(b_items):
        cx = bx0 + 30.0 + idx * 54.0
        ry = yb(raw_v)
        c.rect(cx - 14.0, ry, 13.0, max(1.0, y_zero_b - ry), fill="#d97706", stroke="#92400e", lw=0.8)
        c.text(cx - 7.5, ry - 3.5, f"{raw_v:.3f}", size=6.0, color="#92400e", anchor="middle", mono=True)
        dy = yb(deb_v)
        c.circle(cx + 7.5, dy, 3.2, fill="#059669", stroke="#064e3b", lw=0.9)
        c.text(cx + 7.5, dy - 5.0, "0.000", size=5.8, color="#065f46", anchor="middle", mono=True)
        c.text(cx, by0 + bph + 13, lbl, size=7.1, bold=True, anchor="middle", mono=True)

    c.text(
        bx0 + 0.5 * bpw,
        by0 + bph + 25,
        "SIMULATED Width Benchmark (theta=3.118 / Uniform)",
        size=7.6,
        anchor="middle",
    )
    c.rect(bx0 + 8, by0 + 8, 138, 26, fill="#ffffff", stroke="#d1d5db", lw=0.7)
    c.rect(bx0 + 13, by0 + 13, 9, 7, fill="#d97706", stroke="#92400e", lw=0.7)
    c.text(bx0 + 26, by0 + 19, "Raw Plug-in LP Cost (z=+855 sigma)", size=6.2)
    c.circle(bx0 + 17.5, by0 + 28, 2.8, fill="#059669", stroke="#064e3b", lw=0.8)
    c.text(bx0 + 26, by0 + 30.5, "Matched Debiased S_eps (z=-0.34 sigma)", size=6.2)

    return c


def make_fig2(fez: Mapping[str, Any], ot_rep: Mapping[str, Any]) -> VectorCanvas:
    """Dual-panel vector figure: (a) 27Q Coherent-drift calibration; (b) Heron r2 re-scoring."""
    c = VectorCanvas(560, 290)
    c.text(
        280,
        17,
        "Controlled Coherent-Drift Calibration & Multi-Processor Heron r2 Hardware Re-Scoring",
        size=9.8,
        bold=True,
        anchor="middle",
    )
    c.text(
        280,
        29,
        "HARDWARE_ATTESTED: ibm_fez (dc1tqvinj2vs73dfrts0, dbtl2vb2kf8s73c9a0d0) & ibm_marrakesh (dau0q3qhcrkc73durtgg)",
        size=7.4,
        color="#4b5563",
        anchor="middle",
    )

    # Panel (a): Controlled drift calibration on 27Q Arm B
    x0, y0, pw, ph = 52.0, 52.0, 206.0, 192.0
    c.text(
        x0 + 0.5 * pw,
        44,
        "(a) 27Q ibm_fez Arm B Coherent Drift (H3: 13.76 sigma)",
        size=8.0,
        bold=True,
        anchor="middle",
    )
    c.rect(x0, y0, pw, ph, fill="#fafafa", stroke="#374151", lw=1.0)

    cal_b = fez["controlled_drift_calibration_arm_b"]
    pts_a = [
        (
            "0.00 (A/A)",
            float(cal_b["intra_block_aa_null_matched_S_eps_mean"]),
            float(cal_b["intra_block_aa_null_se"]),
        ),
        (
            "+0.20 rad",
            float(cal_b["delta_theta_0p20_matched_S_eps_mean"]),
            float(cal_b["delta_theta_0p20_se"]),
        ),
        (
            "+0.45 rad",
            float(cal_b["delta_theta_0p45_matched_S_eps_mean"]),
            float(cal_b["delta_theta_0p45_se"]),
        ),
    ]
    ymax_a = 0.0042
    ymin_a = -0.0003

    def ya(val: float) -> float:
        frac = (val - ymin_a) / (ymax_a - ymin_a)
        return y0 + ph * (1.0 - min(1.0, max(0.0, frac)))

    for tv in (0.0, 0.001, 0.002, 0.003, 0.004):
        yy = ya(tv)
        c.line(
            x0,
            yy,
            x0 + pw,
            yy,
            color="#d1d5db" if tv != 0.0 else "#6b7280",
            lw=0.8,
            dashed=(tv != 0.0),
        )
        c.text(x0 - 4, yy + 2.5, f"{tv:.3f}", size=6.6, color="#374151", anchor="end", mono=True)

    xs_a = [x0 + 36.0, x0 + 0.5 * pw, x0 + pw - 36.0]
    coords_a = [(xx, ya(m)) for xx, (_, m, _) in zip(xs_a, pts_a)]
    for i in range(len(coords_a) - 1):
        c.line(coords_a[i][0], coords_a[i][1], coords_a[i + 1][0], coords_a[i + 1][1], color="#2563eb", lw=1.8)

    for xx, (lbl, m, se) in zip(xs_a, pts_a):
        ym = ya(m)
        y_hi = ya(m + se)
        y_lo = ya(m - se)
        c.line(xx, y_hi, xx, y_lo, color="#1d4ed8", lw=1.3)
        c.line(xx - 4, y_hi, xx + 4, y_hi, color="#1d4ed8", lw=1.3)
        c.line(xx - 4, y_lo, xx + 4, y_lo, color="#1d4ed8", lw=1.3)
        c.circle(xx, ym, 3.5, fill="#2563eb", stroke="#ffffff", lw=0.9)
        c.text(xx, ym - 7.5, f"{m:+.5f}", size=6.3, color="#1e3a8a", anchor="middle", mono=True)
        c.text(xx, y0 + ph + 13, lbl, size=7.1, bold=True, anchor="middle", mono=True)
    c.text(x0 + 0.5 * pw, y0 + ph + 25, "Perturbation Delta_theta from theta* = 2.618 rad", size=7.5, anchor="middle")

    # Panel (b): Multi-processor Heron r2 re-scoring (ibm_marrakesh DD & ibm_fez QEC-001)
    bx0, by0, bpw, bph = 308.0, 52.0, 232.0, 192.0
    c.text(
        bx0 + 0.5 * bpw,
        44,
        "(b) Archived Heron r2 Re-Scoring: Eq.(7) vs Debiased S_eps",
        size=7.8,
        bold=True,
        anchor="middle",
    )
    c.rect(bx0, by0, bpw, bph, fill="#fafafa", stroke="#374151", lw=1.0)

    pb = ot_rep["part_b_hardware_attested_heron_r2_rescoring"]
    marr = pb["marrakesh_dd_rescoring"]["dd_cross_condition_contrasts"]
    qec_evals = pb["fez_qec001_rescoring"]["evaluations"]

    m_pair_32 = next(
        x
        for x in marr
        if math.isclose(float(x["T_us"]), 32.0) and x["arm_a"] == "xy8_stag" and x["arm_b"] == "xy4x2_stag"
    )
    m_none_32 = next(
        x
        for x in marr
        if math.isclose(float(x["T_us"]), 32.0) and x["arm_a"] == "xy8_stag" and x["arm_b"] == "none"
    )
    q_aa_49 = next(
        x
        for x in qec_evals
        if x["comparison_type"] == "inter_block_AA_full_register"
        and x["code_basis"] == "phase_flip_x"
        and int(x["register_bits"]) == 49
    )
    q_hot_49 = next(
        x for x in qec_evals if x["comparison_type"] == "spectator_t2_hotspot_basis_contrast_full_49bit"
    )

    bars_b = [
        (
            "Marr 32us\nstags A/B",
            float(m_pair_32["defective_eq7_cost"]),
            float(m_pair_32["debiased_sinkhorn_matched"]),
        ),
        (
            "Marr 32us\nstag/none",
            float(m_none_32["defective_eq7_cost"]),
            float(m_none_32["debiased_sinkhorn_matched"]),
        ),
        (
            "QEC 49b\nX-sec A/A",
            float(q_aa_49["defective_eq7_cost"]),
            float(q_aa_49["debiased_sinkhorn_matched"]),
        ),
        (
            "QEC 49b\nZ vs X T2",
            float(q_hot_49["defective_eq7_cost"]),
            float(q_hot_49["debiased_sinkhorn_matched"]),
        ),
    ]

    ymax_b = 0.24
    ymin_b = -0.01

    def yb(val: float) -> float:
        frac = (val - ymin_b) / (ymax_b - ymin_b)
        return by0 + bph * (1.0 - min(1.0, max(0.0, frac)))

    for tv in (0.0, 0.05, 0.10, 0.15, 0.20):
        yy = yb(tv)
        c.line(
            bx0,
            yy,
            bx0 + bpw,
            yy,
            color="#d1d5db" if tv != 0.0 else "#6b7280",
            lw=0.8,
            dashed=(tv != 0.0),
        )
        c.text(bx0 - 4, yy + 2.5, f"{tv:.2f}", size=6.6, color="#374151", anchor="end", mono=True)

    yz = yb(0.0)
    for idx, (lbl, eq7_v, sk_v) in enumerate(bars_b):
        cx = bx0 + 31.0 + idx * 56.0
        ye = yb(eq7_v)
        ys = yb(max(0.0, sk_v))
        c.rect(cx - 14.0, ye, 12.5, max(1.0, yz - ye), fill="#ef4444", stroke="#991b1b", lw=0.8)
        c.rect(cx + 1.5, min(yz, ys), 12.5, max(1.5, abs(yz - ys)), fill="#059669", stroke="#065f46", lw=0.8)
        c.text(cx + 7.5, min(yz, ys) - 3.5, f"{sk_v:.4f}", size=5.7, color="#065f46", anchor="middle", mono=True)
        parts = lbl.split("\n")
        c.text(cx, by0 + bph + 11, parts[0], size=6.5, bold=True, anchor="middle", mono=True)
        c.text(cx, by0 + bph + 20, parts[1], size=6.5, anchor="middle", mono=True)

    c.rect(bx0 + 8, by0 + 8, 126, 25, fill="#ffffff", stroke="#d1d5db", lw=0.7)
    c.rect(bx0 + 13, by0 + 12, 9, 7, fill="#ef4444", stroke="#991b1b", lw=0.7)
    c.text(bx0 + 26, by0 + 18, "Defective Eq.(7) Cost", size=6.2)
    c.rect(bx0 + 13, by0 + 22, 9, 7, fill="#059669", stroke="#065f46", lw=0.7)
    c.text(bx0 + 26, by0 + 28, "Matched Debiased S_eps", size=6.2)

    return c


def generate_all() -> Dict[str, Any]:
    fez_path = FEZ_DIR / "physical_wp2_27q_fez_results.json"
    manifest_path = FEZ_DIR / "wp2_27q_fez_staging_manifest.json"
    cal_path = FEZ_DIR / "wp2_27q_fez_calibration_snapshot.json"

    fez = json.loads(fez_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cal = json.loads(cal_path.read_text(encoding="utf-8"))
    ot_rep = json.loads(OT_REPORT_PATH.read_text(encoding="utf-8"))

    gen_tex = build_generated_tex(fez, manifest, cal, ot_rep)
    gen_tex_path = PAPER_DIR / "generated.tex"
    gen_tex_path.write_text(gen_tex, encoding="utf-8", newline="\n")

    fig1 = make_fig1(fez, ot_rep)
    fig1.save(PAPER_DIR / FIG1_STEM)

    fig2 = make_fig2(fez, ot_rep)
    fig2.save(PAPER_DIR / FIG2_STEM)

    upstream_rels = [
        "experiments/wp2_27q_fez/PRE_REGISTRATION.md",
        "experiments/wp2_27q_fez/stage_and_harvest_wp2_27q_fez.py",
        "experiments/wp2_27q_fez/physical_wp2_27q_fez_results.json",
        "experiments/wp2_27q_fez/physical_wp2_27q_fez_raw_counts.json.gz",
        "experiments/wp2_27q_fez/wp2_27q_fez_staged_circuits.qpy",
        "experiments/wp2_27q_fez/wp2_27q_fez_staging_manifest.json",
        "experiments/wp2_27q_fez/wp2_27q_fez_calibration_snapshot.json",
        "experiments/wp2_27q_fez/wp2_27q_fez_submitted_job.json",
        "experiments/wp2_debiased_sinkhorn/debiased_ot.py",
        "experiments/wp2_debiased_sinkhorn/artifacts/wp2_debiased_ot_report.json",
        "experiments/osiris_advantage_marrakesh/dau0q3qhcrkc73durtgg.counts.json",
        "experiments/qec001/physical_qec001_fez_raw_counts.json.gz",
    ]
    generated_rels = [
        f"papers/wp2_27q_debiased_ot_fez/{TEX_FILENAME}",
        "papers/wp2_27q_debiased_ot_fez/generated.tex",
        "papers/wp2_27q_debiased_ot_fez/generate_wp2_paper_assets.py",
        f"papers/wp2_27q_debiased_ot_fez/{FIG1_STEM}.svg",
        f"papers/wp2_27q_debiased_ot_fez/{FIG1_STEM}.pdf",
        f"papers/wp2_27q_debiased_ot_fez/{FIG2_STEM}.svg",
        f"papers/wp2_27q_debiased_ot_fez/{FIG2_STEM}.pdf",
    ]

    capsule_body: Dict[str, Any] = {
        "schema": "osiris.reproducibility_capsule/1",
        "canonicalization": "RFC8785-JCS",
        "paper_id": "OSIRIS-PAPER-WP2-27Q-DEBIASED-OT-FEZ",
        "title": (
            "Eliminating Finite-Shot Hypercube Sparsity Bias in 27-Qubit Quantum Processor "
            "Stability Benchmarks via Split-Sample-Debiased Optimal Transport"
        ),
        "author": "Devin Phillip Davis",
        "affiliation": "Agile Defense Systems LLC",
        "evidence_class": "HARDWARE_ATTESTED",
        "hardware_executed": True,
        "provider_job_id": fez["provider_job_id"],
        "provider_job_ids": [
            fez["provider_job_id"],
            ot_rep["part_b_hardware_attested_heron_r2_rescoring"]["marrakesh_dd_rescoring"]["provider_job_id"],
            ot_rep["part_b_hardware_attested_heron_r2_rescoring"]["fez_qec001_rescoring"]["provider_job_id"],
        ],
        "simulated_companion_tier": {
            "evidence_class": "SIMULATED",
            "provider_job_id": None,
            "scope": "Part A multi-width 27Q/50Q/96Q and Uniform({0,1}^96) hypercube scaling benchmarks",
        },
        "backends": ["ibm_fez", "ibm_marrakesh"],
        "total_hardware_shots_bound": (
            int(manifest["total_shots"])
            + int(ot_rep["part_b_hardware_attested_heron_r2_rescoring"]["marrakesh_dd_rescoring"]["total_shots"])
            + int(ot_rep["part_b_hardware_attested_heron_r2_rescoring"]["fez_qec001_rescoring"]["total_shots"])
        ),
        "hypotheses_verdicts": {
            "H1_weed_bach_sparsity_bias_eliminated": fez["hypotheses"][
                "H1_weed_bach_sparsity_bias_eliminated"
            ]["verdict"],
            "H2_golden_ratio_phi_sq_no_special_stability": fez["hypotheses"][
                "H2_golden_ratio_phi_sq_no_special_stability"
            ]["verdict"],
            "H3_monotonic_coherent_drift_detection_gt_10sigma": fez["hypotheses"][
                "H3_monotonic_coherent_drift_detection_gt_10sigma"
            ]["verdict"],
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
        + ["papers/wp2_27q_debiased_ot_fez/reproducibility_capsule.json"]
        + upstream_rels
    )
    sha_lines = [f"{_sha256_file(REPO_ROOT / rel)}  {rel}" for rel in sha_rels]
    (PAPER_DIR / "SHA256SUMS.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8", newline="\n")
    return capsule


if __name__ == "__main__":
    cap = generate_all()
    print(f"[OK] Generated WP2 paper assets; capsule_body_sha256={cap['capsule_body_sha256']}")
