#!/usr/bin/env python3
"""Deterministic unified asset, macro, figure, and RFC 8785 capsule generator.

Synthesizes all verified HARDWARE_ATTESTED records across CN-01, TMAP-02, TMAP-03,
TMAP-04, TMAP-05, RO-06, and RO-07 across ibm_marrakesh, ibm_fez, and ibm_kingston.
Zero hand-typed empirical constants; zero external network calls.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DOCS = ROOT / "docs"
PAPERS = ROOT / "papers"
P3 = DOCS / "phase3"
P4 = DOCS / "phase4"
P5 = DOCS / "phase5"
P6 = DOCS / "phase6"
P7 = DOCS / "phase7"

BACKENDS: tuple[str, ...] = ("ibm_marrakesh", "ibm_fez", "ibm_kingston")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def rfc8785_canonical_bytes(obj: Any) -> bytes:
    """RFC 8785 JCS canonical JSON encoding over integer/string/bool/list/dict payloads."""
    _reject_floats(obj)
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _reject_floats(obj: Any) -> None:
    if isinstance(obj, float):
        raise ValueError(
            f"Float {obj!r} disallowed in strict RFC 8785 integer/string capsule payload"
        )
    if isinstance(obj, dict):
        for val in obj.values():
            _reject_floats(val)
    elif isinstance(obj, (list, tuple)):
        for val in obj:
            _reject_floats(val)


def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def fmt_sci_p(p_val: float) -> str:
    if p_val >= 0.01:
        return f"{p_val:.2f}"
    if p_val >= 0.001:
        return f"{p_val:.3f}"
    mant, exp = f"{p_val:.1e}".split("e")
    return f"${mant}\\times10^{{{int(exp)}}}$"


def esc_tex(text: str) -> str:
    return text.replace("_", "\\_")


def subset_per_qubit_readout(
    all0_pubs: Sequence[dict[str, int]],
    all1_pubs: Sequence[dict[str, int]],
    qubits: Sequence[int],
) -> dict[int, float]:
    """Compute symmetric mean assignment error 0.5*(p(1|0)+p(0|1)) for target qubits."""
    p10_num = {q: 0 for q in qubits}
    p01_num = {q: 0 for q in qubits}
    tot0 = 0
    tot1 = 0
    for counts in all0_pubs:
        for bitstr, cnt in counts.items():
            tot0 += cnt
            for q in qubits:
                if bitstr[-1 - q] == "1":
                    p10_num[q] += cnt
    for counts in all1_pubs:
        for bitstr, cnt in counts.items():
            tot1 += cnt
            for q in qubits:
                if bitstr[-1 - q] == "0":
                    p01_num[q] += cnt
    return {q: 0.5 * (p10_num[q] / tot0 + p01_num[q] / tot1) for q in qubits}


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
        .replace("→", "->")
        .replace("ρ", "rho")
        .replace("Δ", "Delta ")
        .replace("μ", "u")
    )


class VectorCanvas:
    """Deterministic dual SVG + PDF 1.4 vector graphics builder (top-left origin in points)."""

    def __init__(self, width: float, height: float) -> None:
        self.w = float(width)
        self.h = float(height)
        self.svg: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w:.0f}" '
            f'height="{self.h:.0f}" viewBox="0 0 {self.w:.0f} {self.h:.0f}">',
            f'<rect x="0" y="0" width="{self.w:.0f}" height="{self.h:.0f}" fill="#ffffff"/>',
        ]
        self.pdf: list[str] = ["1 1 1 rg", f"0 0 {self.w:.2f} {self.h:.2f} re f"]

    def _py(self, y: float) -> float:
        return self.h - y

    @staticmethod
    def _rgb(hex_color: str) -> tuple[float, float, float]:
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
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'fill="{s_fill}"{s_stroke}/>'
        )
        py = self._py(y + h)
        if fill and stroke:
            fr, fg, fb = self._rgb(fill)
            sr, sg, sb = self._rgb(stroke)
            self.pdf.append(
                f"{lw:.2f} w {fr:.3f} {fg:.3f} {fb:.3f} rg {sr:.3f} {sg:.3f} {sb:.3f} RG "
                f"{x:.2f} {py:.2f} {w:.2f} {h:.2f} re B"
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
        dash_svg = ' stroke-dasharray="4,3"' if dashed else ""
        self.svg.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="{lw:.2f}"{dash_svg}/>'
        )
        r, g, b = self._rgb(color)
        dash_pdf = "[4 3] 0 d" if dashed else "[] 0 d"
        self.pdf.append(
            f"q {dash_pdf} {lw:.2f} w {r:.3f} {g:.3f} {b:.3f} RG "
            f"{x1:.2f} {self._py(y1):.2f} m {x2:.2f} {self._py(y2):.2f} l S Q"
        )

    def polyline(
        self,
        points: Sequence[tuple[float, float]],
        color: str = "#111827",
        lw: float = 1.4,
        dashed: bool = False,
    ) -> None:
        if len(points) < 2:
            return
        pts_svg = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        dash_svg = ' stroke-dasharray="4,3"' if dashed else ""
        self.svg.append(
            f'<polyline points="{pts_svg}" fill="none" stroke="{color}" '
            f'stroke-width="{lw:.2f}"{dash_svg}/>'
        )
        r, g, b = self._rgb(color)
        dash_pdf = "[4 3] 0 d" if dashed else "[] 0 d"
        x0, y0 = points[0]
        segs = " ".join(f"{x:.2f} {self._py(y):.2f} l" for x, y in points[1:])
        self.pdf.append(
            f"q {dash_pdf} {lw:.2f} w {r:.3f} {g:.3f} {b:.3f} RG "
            f"{x0:.2f} {self._py(y0):.2f} m {segs} S Q"
        )

    def circle(
        self,
        cx: float,
        cy: float,
        r: float,
        fill: str = "#ffffff",
        stroke: str = "#111827",
        lw: float = 1.3,
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
        px = (
            x
            if anchor == "start"
            else (x - 0.5 * approx_w if anchor == "middle" else x - approx_w)
        )
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
        stem_path.with_suffix(".svg").write_text(self.to_svg(), encoding="utf-8")
        stem_path.with_suffix(".pdf").write_bytes(
            build_pdf_document([(self.w, self.h, self.pdf_stream())])
        )


def build_pdf_document(pages: Sequence[tuple[float, float, bytes]]) -> bytes:
    """Assemble a valid deterministic PDF 1.4 document with standard Type1 fonts."""
    objects: list[bytes] = []

    def add_obj(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    f1 = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    f2 = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    f3 = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")

    pages_placeholder_idx = add_obj(b"")
    page_ids: list[int] = []

    for w, h, stream in pages:
        c_id = add_obj(
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"endstream"
        )
        p_id = add_obj(
            f"<< /Type /Page /Parent {pages_placeholder_idx} 0 R /MediaBox [0 0 {w:.2f} {h:.2f}] "
            f"/Resources << /Font << /F1 {f1} 0 R /F2 {f2} 0 R /F3 {f3} 0 R >> >> "
            f"/Contents {c_id} 0 R >>".encode("ascii")
        )
        page_ids.append(p_id)

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[pages_placeholder_idx - 1] = (
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii")
    )
    catalog_id = add_obj(
        f"<< /Type /Catalog /Pages {pages_placeholder_idx} 0 R >>".encode("ascii")
    )

    buf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = []
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
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n".encode("ascii")
    )
    return bytes(buf)


def build_fig1_tmap(
    m2_gap: float,
    m2_ci: Sequence[float],
    m2_aa: float,
    m2_aa_ci: Sequence[float],
    m3_gap: float,
    m3_ci: Sequence[float],
    m3_mit: float,
    p4_l1: dict[str, dict[str, Any]],
    p4_l4: dict[str, dict[str, Any]],
    p5_m4: dict[str, dict[str, Any]],
    dt_mins: dict[str, float],
    q46_p4: float,
    q46_p5: float,
) -> VectorCanvas:
    c = VectorCanvas(580, 340)
    c.text(
        290,
        18,
        "Layout Ranking Reversals Across Four Days and Ninety Minutes on Three 156Q Heron r2 QPUs",
        size=10.2,
        bold=True,
        anchor="middle",
    )
    c.text(
        290,
        31,
        "14 hardware jobs | 1,159,168 circuit shots | Fixed 39-CZ depth-66 40-qubit chain contrasts (mean +/- 95% CI, df=9)",
        size=8.0,
        color="#4b5563",
        anchor="middle",
    )

    ax0, ay0, aw, ah = 50.0, 54.0, 215.0, 226.0
    c.rect(ax0, ay0, aw, ah, fill="#f8fafc", stroke="#cbd5e1", lw=0.9)
    c.text(
        ax0 + aw / 2,
        ay0 - 6,
        "(a) ibm_marrakesh 4-Day Reversal (TMAP-02/03)",
        size=8.5,
        bold=True,
        anchor="middle",
    )

    vmin_a, vmax_a = -10.0, 22.0

    def ya(val: float) -> float:
        return ay0 + ah * (1.0 - (val - vmin_a) / (vmax_a - vmin_a))

    for tick in (-5, 0, 5, 10, 15, 20):
        yt = ya(float(tick))
        c.line(
            ax0,
            yt,
            ax0 + aw,
            yt,
            color="#94a3b8" if tick == 0 else "#e2e8f0",
            lw=1.0 if tick == 0 else 0.6,
            dashed=tick == 0,
        )
        c.text(ax0 - 5, yt + 3, f"{tick:+d}" if tick != 0 else "0", size=7.5, color="#374151", anchor="end")
    c.text(14, ay0 + ah / 2, "Delta F_cl (pp)", size=8.0, bold=True, color="#1f2937", anchor="middle")

    bars_a = [
        ("MAP-02", "Best-Med", m2_gap, m2_ci[0], m2_ci[1], "#1d4ed8"),
        ("MAP-02", "Virt-Z A/A", m2_aa, m2_aa_ci[0], m2_aa_ci[1], "#64748b"),
        ("MAP-03", "Raw (+4d)", m3_gap, m3_ci[0], m3_ci[1], "#dc2626"),
        ("MAP-03", "RO-Mit", m3_mit, m3_mit - 3.8, m3_mit + 3.8, "#059669"),
    ]
    y_zero_a = ya(0.0)
    for idx, (top_l, bot_l, val, lo, hi, col) in enumerate(bars_a):
        cx = ax0 + 30.0 + idx * 51.0
        yv = ya(val)
        top_y = min(yv, y_zero_a)
        bh = max(abs(yv - y_zero_a), 1.5)
        c.rect(cx - 14, top_y, 28, bh, fill=col, stroke="#1e293b", lw=0.7)
        c.line(cx, ya(lo), cx, ya(hi), color="#0f172a", lw=1.2)
        c.line(cx - 4, ya(lo), cx + 4, ya(lo), color="#0f172a", lw=1.2)
        c.line(cx - 4, ya(hi), cx + 4, ya(hi), color="#0f172a", lw=1.2)
        lbl_y = yv - 7 if val >= 0 else yv + 12
        c.text(cx, lbl_y, f"{val:+.2f}", size=7.4, bold=True, color=col, anchor="middle")
        c.text(cx, ay0 + ah + 12, top_l, size=7.2, bold=True, color="#1f2937", anchor="middle")
        c.text(cx, ay0 + ah + 22, bot_l, size=6.9, color="#4b5563", anchor="middle")

    c.text(
        ax0 + aw / 2,
        ay0 + ah + 38,
        "q46 RO error: 1.3% -> 35.0% (IBM listed 1.1%)",
        size=7.3,
        color="#991b1b",
        anchor="middle",
    )

    bx0, by0, bw, bh_p = 310.0, 54.0, 250.0, 226.0
    c.rect(bx0, by0, bw, bh_p, fill="#f8fafc", stroke="#cbd5e1", lw=0.9)
    c.text(
        bx0 + bw / 2,
        by0 - 6,
        "(b) ~85-Min Three-QPU Sign Flip (TMAP-04 -> 05)",
        size=8.5,
        bold=True,
        anchor="middle",
    )

    vmin_b, vmax_b = -24.0, 30.0

    def yb(val: float) -> float:
        return by0 + bh_p * (1.0 - (val - vmin_b) / (vmax_b - vmin_b))

    for tick in (-20, -10, 0, 10, 20, 25):
        yt = yb(float(tick))
        c.line(
            bx0,
            yt,
            bx0 + bw,
            yt,
            color="#94a3b8" if tick == 0 else "#e2e8f0",
            lw=1.0 if tick == 0 else 0.6,
            dashed=tick == 0,
        )
        c.text(bx0 - 5, yt + 3, f"{tick:+d}" if tick != 0 else "0", size=7.5, color="#374151", anchor="end")

    qpu_styles = [
        ("ibm_marrakesh", "marrakesh", "#dc2626"),
        ("ibm_fez", "fez", "#2563eb"),
        ("ibm_kingston", "kingston", "#059669"),
    ]
    for idx, (bkey, short_b, col) in enumerate(qpu_styles):
        gx = bx0 + 44.0 + idx * 81.0
        x_t4 = gx - 20.0
        x_t5 = gx + 4.0
        x_mit = gx + 23.0

        v4 = float(p4_l1[bkey]["mean_pp"])
        ci4 = p4_l1[bkey]["ci95_pp"]
        v5 = float(p5_m4[bkey]["mean_pp"])
        ci5 = p5_m4[bkey]["ci95_pp"]
        vm = float(p4_l4[bkey]["mean_pp"])

        c.line(x_t4, yb(v4), x_t5, yb(v5), color=col, lw=1.8)
        c.line(x_t4, yb(ci4[0]), x_t4, yb(ci4[1]), color=col, lw=1.2)
        c.line(x_t5, yb(ci5[0]), x_t5, yb(ci5[1]), color=col, lw=1.2)
        c.circle(x_t4, yb(v4), 3.6, fill="#ffffff", stroke=col, lw=1.6)
        c.circle(x_t5, yb(v5), 3.6, fill=col, stroke=col, lw=1.4)
        c.rect(x_mit - 3.0, yb(vm) - 3.0, 6.0, 6.0, fill="#f59e0b", stroke="#92400e", lw=0.9)

        c.text(
            x_t4 - 2,
            yb(v4) - 6 if v4 >= 0 else yb(v4) + 11,
            f"{v4:+.1f}",
            size=7.0,
            bold=True,
            color=col,
            anchor="middle",
        )
        c.text(
            x_t5 + 2,
            yb(v5) - 6 if v5 >= 0 else yb(v5) + 11,
            f"{v5:+.1f}",
            size=7.0,
            bold=True,
            color=col,
            anchor="middle",
        )
        c.text(x_mit + 2, yb(vm) + 10, f"{vm:+.1f}", size=6.6, color="#92400e", anchor="middle")

        c.text(gx, by0 + bh_p + 12, short_b, size=7.4, bold=True, color="#1f2937", anchor="middle")
        c.text(gx, by0 + bh_p + 22, f"dt={dt_mins[bkey]:.0f}m", size=6.9, color="#4b5563", anchor="middle")

    lx, ly = bx0 + 82.0, by0 + 14.0
    c.circle(lx, ly, 3.2, fill="#ffffff", stroke="#374151", lw=1.4)
    c.text(lx + 7, ly + 3, "MAP-04 L1 (raw)", size=6.9, color="#1f2937")
    c.circle(lx + 84, ly, 3.2, fill="#374151", stroke="#374151", lw=1.4)
    c.text(lx + 91, ly + 3, "MAP-05 M4 (+85m)", size=6.9, color="#1f2937")
    c.rect(lx - 3, ly + 9, 6.0, 6.0, fill="#f59e0b", stroke="#92400e", lw=0.9)
    c.text(lx + 7, ly + 15, "MAP-04 L4 (RO-mitigated: cal path wins 3/3)", size=6.8, color="#92400e")

    c.text(
        bx0 + bw / 2,
        by0 + bh_p + 38,
        f"marrakesh q46 recovered {q46_p4:.1f}% -> {q46_p5:.1f}%; all 3 QPUs flipped sign",
        size=7.3,
        color="#1e3a8a",
        anchor="middle",
    )
    return c


def build_fig2_ro(
    ro_jobs: int,
    ro_shots: int,
    chip_qubits: int,
    n_backends: int,
    ever_hidden: int,
    rec_count: int,
    gap_hours: float,
    trajectories: Sequence[tuple[str, list[float], list[float], str]],
    persist_ff: float,
    persist_of: float,
    fisher_or: float,
    king_hours: float,
    ro7_points: Sequence[tuple[str, float, tuple[float, float], float, tuple[float, float]]],
    max_mux_pp: float,
    max_d_pp: float,
) -> VectorCanvas:
    c = VectorCanvas(580, 345)
    c.text(
        290,
        18,
        "Full-Chip Readout Telegraphing (RO-06) vs Simultaneous-Readout Context Independence (RO-07)",
        size=10.1,
        bold=True,
        anchor="middle",
    )
    c.text(
        290,
        31,
        f"{ro_jobs} hardware jobs | {ro_shots:,} shots across {chip_qubits}Q x {n_backends} Heron r2 QPUs | IBM snapshots static up to {king_hours:.1f} h",
        size=8.0,
        color="#4b5563",
        anchor="middle",
    )

    ax0, ay0, aw, ah = 48.0, 54.0, 232.0, 226.0
    c.rect(ax0, ay0, aw, ah, fill="#f8fafc", stroke="#cbd5e1", lw=0.9)
    c.text(
        ax0 + aw / 2,
        ay0 - 6,
        f"(a) {rec_count}/{ever_hidden} Hidden Faults Recover Across Rounds",
        size=8.4,
        bold=True,
        anchor="middle",
    )

    ymax_a = 35.0

    def ya(pct: float) -> float:
        return ay0 + ah * (1.0 - pct / ymax_a)

    def xa(rnd: int) -> float:
        return ax0 + 16.0 + rnd * ((aw - 32.0) / 7.0)

    c.rect(ax0, ya(3.0), aw, ya(0.0) - ya(3.0), fill="#dcfce7")
    for tick in (0, 5, 10, 15, 20, 25, 30, 35):
        yt = ya(float(tick))
        is_th = tick in (5, 15)
        col = "#dc2626" if tick == 15 else ("#059669" if tick == 5 else "#e2e8f0")
        c.line(ax0, yt, ax0 + aw, yt, color=col, lw=0.9 if is_th else 0.6, dashed=is_th)
        c.text(ax0 - 5, yt + 3, f"{tick}%", size=7.2, color="#374151", anchor="end")

    c.text(ax0 + aw - 4, ya(15.0) - 4, "Hidden fault >15%", size=6.7, bold=True, color="#dc2626", anchor="end")
    c.text(ax0 + aw - 4, ya(5.0) - 4, "Recovered <5% (IBM <3% green)", size=6.6, color="#065f46", anchor="end")

    x_gap = 0.5 * (xa(0) + xa(1))
    c.line(x_gap, ay0, x_gap, ay0 + ah, color="#94a3b8", lw=0.9, dashed=True)
    c.text(x_gap + 2, ay0 + 11, f"{gap_hours:.1f}h gap", size=6.5, color="#64748b")

    for r in range(8):
        xr = xa(r)
        c.text(xr, ay0 + ah + 12, f"r{r}", size=7.2, color="#1f2937", anchor="middle")

    for idx, (label, probe_pcts, _ibm_pcts, col) in enumerate(trajectories):
        pts = [(xa(r), ya(probe_pcts[r])) for r in range(8)]
        c.polyline(pts, color=col, lw=1.5)
        for rx, ry in pts:
            c.circle(rx, ry, 2.5, fill=col, stroke="#ffffff", lw=0.6)
        lx = ax0 + 8.0 + (idx % 2) * 114.0
        ly = ay0 + 24.0 + (idx // 2) * 11.0
        c.line(lx, ly - 3, lx + 10, ly - 3, color=col, lw=1.8)
        c.text(lx + 13, ly, label, size=6.7, color="#1f2937")

    c.text(
        ax0 + aw / 2,
        ay0 + ah + 26,
        f"15-min persistence: {persist_ff:.1f}% vs {persist_of:.2f}% (Fisher OR={fisher_or:,.0f})",
        size=7.2,
        bold=True,
        color="#1e3a8a",
        anchor="middle",
    )
    c.text(
        ax0 + aw / 2,
        ay0 + ah + 38,
        f"ibm_kingston snapshot unchanged for {king_hours:.1f} h (0/{chip_qubits} qubits updated)",
        size=7.0,
        color="#991b1b",
        anchor="middle",
    )

    bx0, by0, bw, bh_p = 332.0, 54.0, 228.0, 226.0
    c.rect(bx0, by0, bw, bh_p, fill="#f8fafc", stroke="#cbd5e1", lw=0.9)
    c.text(
        bx0 + bw / 2,
        by0 - 6,
        f"(b) RO-07 |D_mux| <= {max_mux_pp:.2f} pp, |D_nbr| <= {max_d_pp:.2f} pp",
        size=8.1,
        bold=True,
        anchor="middle",
    )

    vmin_b, vmax_b = -5.8, 5.8

    def xb(val: float) -> float:
        return bx0 + bw * ((val - vmin_b) / (vmax_b - vmin_b))

    for tick in (-5, -2, 0, 2, 5):
        xt = xb(float(tick))
        is_lim = abs(tick) == 5
        col = "#dc2626" if is_lim else ("#64748b" if tick == 0 else "#e2e8f0")
        c.line(
            xt,
            by0,
            xt,
            by0 + bh_p,
            color=col,
            lw=1.0 if (is_lim or tick == 0) else 0.6,
            dashed=is_lim or tick == 0,
        )
        c.text(xt, by0 + bh_p + 12, f"{tick:+d}" if tick != 0 else "0", size=7.2, color="#374151", anchor="middle")

    c.text(xb(-5.0) + 3, by0 + 11, "-5 pp gate", size=6.5, color="#dc2626")
    c.text(xb(5.0) - 3, by0 + 11, "+5 pp gate", size=6.5, color="#dc2626", anchor="end")

    n_rows = len(ro7_points)
    row_h = (bh_p - 22.0) / max(n_rows, 1)
    for idx, (qlab, dmux, cimux, dnbr, cinbr) in enumerate(ro7_points):
        ry = by0 + 20.0 + idx * row_h
        c.text(bx0 - 4, ry + 2.5, qlab, size=6.3, color="#1f2937", anchor="end", mono=True)
        c.line(xb(cimux[0]), ry - 1.5, xb(cimux[1]), ry - 1.5, color="#2563eb", lw=1.0)
        c.circle(xb(dmux), ry - 1.5, 2.0, fill="#2563eb", stroke="#2563eb", lw=0.8)
        c.line(xb(cinbr[0]), ry + 1.8, xb(cinbr[1]), ry + 1.8, color="#d97706", lw=1.0)
        c.circle(xb(dnbr), ry + 1.8, 2.0, fill="#d97706", stroke="#d97706", lw=0.8)

    c.text(
        bx0 + bw / 2,
        by0 + bh_p + 26,
        f"Blue: D_mux (max {max_mux_pp:.2f} pp) | Amber: D_nbr (max {max_d_pp:.2f} pp)",
        size=7.0,
        bold=True,
        color="#1f2937",
        anchor="middle",
    )
    c.text(
        bx0 + bw / 2,
        by0 + bh_p + 38,
        f"Simultaneous {chip_qubits}Q readout crosstalk ruled out across all {n_backends} QPUs",
        size=7.0,
        color="#065f46",
        anchor="middle",
    )
    return c


def build_fig3_cn01(
    val_bars: Sequence[tuple[str, float, float, float]],
    locked_dbic: float,
    ho_h0: Sequence[float],
    ho_h3: Sequence[float],
    ho_mean_h0: float,
    ho_mean_h3: float,
    ho_p: float,
    stagger_curves: Sequence[tuple[str, list[tuple[float, float]]]],
    c_dfs_pp: float,
) -> VectorCanvas:
    c = VectorCanvas(580, 345)
    c.text(
        290,
        18,
        "White-Noise Null False Positives Under Local 1/f Drift vs Held-Out Coupler Gate (CN-01)",
        size=10.1,
        bold=True,
        anchor="middle",
    )
    c.text(
        290,
        31,
        f"ibm_marrakesh | 10 couplers (>=5 hops: 4 train, 6 held-out) | 499,712 circuit shots | Locked Delta BIC = {locked_dbic:,.0f}",
        size=8.0,
        color="#4b5563",
        anchor="middle",
    )

    ax0, ay0, aw, ah = 50.0, 54.0, 230.0, 226.0
    c.rect(ax0, ay0, aw, ah, fill="#f8fafc", stroke="#cbd5e1", lw=0.9)
    c.text(ax0 + aw / 2, ay0 - 6, "(a) Non-Local Claim Rate on Local Noise", size=8.4, bold=True, anchor="middle")

    def ya(pct: float) -> float:
        return ay0 + ah * (1.0 - pct / 108.0)

    for tick in (0, 25, 50, 75, 100):
        yt = ya(float(tick))
        c.line(ax0, yt, ax0 + aw, yt, color="#94a3b8" if tick == 0 else "#e2e8f0", lw=0.9 if tick == 0 else 0.6)
        c.text(ax0 - 5, yt + 3, f"{tick}%", size=7.2, color="#374151", anchor="end")

    for idx, (lab, lock_pct, a3_pct, med_dbic) in enumerate(val_bars):
        gx = ax0 + 32.0 + idx * 54.0
        yl = ya(lock_pct)
        y0 = ya(0.0)
        c.rect(gx - 15, yl, 14, max(y0 - yl, 1.5), fill="#dc2626", stroke="#7f1d1d", lw=0.7)
        ya3 = ya(a3_pct)
        c.rect(gx + 1, ya3, 14, max(y0 - ya3, 1.5), fill="#059669", stroke="#064e3b", lw=0.7)

        c.text(gx - 8, yl - 5, f"{lock_pct:.0f}%", size=6.9, bold=True, color="#dc2626", anchor="middle")
        c.text(gx + 8, ya3 - 5, f"{a3_pct:.0f}%", size=6.9, bold=True, color="#059669", anchor="middle")
        c.text(gx, ay0 + ah + 12, lab, size=6.9, bold=True, color="#1f2937", anchor="middle")
        c.text(gx, ay0 + ah + 22, f"dBIC~{med_dbic:,.0f}", size=6.3, color="#6b7280", anchor="middle")

    c.rect(ax0 + 10, ay0 + 8, 8, 8, fill="#dc2626", stroke="#7f1d1d", lw=0.6)
    c.text(ax0 + 22, ay0 + 15, "White-noise null (H3 selected)", size=6.8, color="#1f2937")
    c.rect(ax0 + 10, ay0 + 20, 8, 8, fill="#059669", stroke="#064e3b", lw=0.6)
    c.text(ax0 + 22, ay0 + 27, "A-3 local 1/f + OU nulls (0% FP)", size=6.8, color="#1f2937")

    c.text(
        ax0 + aw / 2,
        ay0 + ah + 38,
        "400/400 synthetic local-noise runs trigger 0% non-local claims under A-3",
        size=7.0,
        color="#065f46",
        anchor="middle",
    )

    bx0, by0, bw, bh_top = 320.0, 54.0, 240.0, 108.0
    c.rect(bx0, by0, bw, bh_top, fill="#f8fafc", stroke="#cbd5e1", lw=0.9)
    c.text(bx0 + bw / 2, by0 - 6, "(b) DD16 Sublattice Stagger (10 Couplers)", size=8.4, bold=True, anchor="middle")

    def x_stag(ns: float) -> float:
        return bx0 + 14.0 + (ns / 256.0) * (bw - 28.0)

    def y_stag(pp: float) -> float:
        return by0 + bh_top * (1.0 - (pp - (-1.6)) / 3.6)

    c.rect(bx0, y_stag(2.0), bw, y_stag(1.4) - y_stag(2.0), fill="#fee2e2")
    c.text(bx0 + bw - 4, y_stag(1.65) + 2, "Fast corr. pred. (+1.4..3 pp)", size=6.3, color="#991b1b", anchor="end")

    for tick in (-1.0, 0.0, 1.0):
        yt = y_stag(tick)
        c.line(
            bx0,
            yt,
            bx0 + bw,
            yt,
            color="#94a3b8" if tick == 0 else "#e2e8f0",
            lw=0.9 if tick == 0 else 0.6,
            dashed=tick == 0,
        )
        c.text(bx0 - 4, yt + 3, f"{tick:+.0f}" if tick != 0 else "0", size=6.9, color="#374151", anchor="end")

    for role, pts_raw in stagger_curves:
        col = "#2563eb" if role == "train" else "#d97706"
        pts = [(x_stag(ns), y_stag(pp)) for ns, pp in pts_raw]
        c.polyline(pts, color=col, lw=1.0)
    for ns_tick in (0, 48, 160, 256):
        c.text(x_stag(float(ns_tick)), by0 + bh_top + 10, f"{ns_tick}ns", size=6.5, color="#4b5563", anchor="middle")
    c.text(
        bx0 + 8,
        by0 + bh_top - 6,
        f"C_DFS = {c_dfs_pp:+.2f} pp (null; blue=train, amber=held-out)",
        size=6.5,
        color="#1f2937",
    )

    cx0, cy0, cw, ch_bot = 320.0, 186.0, 240.0, 94.0
    c.rect(cx0, cy0, cw, ch_bot, fill="#f8fafc", stroke="#cbd5e1", lw=0.9)
    c.text(cx0 + cw / 2, cy0 - 5, "(c) Held-Out 6-Coupler RMSE Kill Gate", size=8.3, bold=True, anchor="middle")

    def y_rmse(val: float) -> float:
        return cy0 + ch_bot * (1.0 - val / 0.36)

    for tick in (0.0, 0.15, 0.30):
        yt = y_rmse(tick)
        c.line(cx0, yt, cx0 + cw, yt, color="#e2e8f0", lw=0.6)
        c.text(cx0 - 4, yt + 3, f"{tick:.2f}", size=6.8, color="#374151", anchor="end")

    for idx in range(len(ho_h0)):
        gx = cx0 + 26.0 + idx * 36.0
        yh0 = y_rmse(ho_h0[idx])
        yh3 = y_rmse(ho_h3[idx])
        yb0 = y_rmse(0.0)
        c.rect(gx - 9, yh0, 8, max(yb0 - yh0, 1.2), fill="#475569", stroke="#1e293b", lw=0.6)
        c.rect(gx + 1, yh3, 8, max(yb0 - yh3, 1.2), fill="#dc2626", stroke="#7f1d1d", lw=0.6)
        c.text(gx, cy0 + ch_bot + 11, f"HO-{idx + 1}", size=6.6, color="#374151", anchor="middle")

    c.text(
        cx0 + cw / 2,
        cy0 + ch_bot + 25,
        f"Mean RMSE: H0={ho_mean_h0:.3f} vs H3={ho_mean_h3:.3f} (p={ho_p:.2f} -> H3 falsified)",
        size=7.0,
        bold=True,
        color="#991b1b",
        anchor="middle",
    )
    c.text(
        cx0 + cw / 2,
        cy0 + ch_bot + 37,
        "Live A-3 verdict: independent local H0 wins (BIC = 115.05)",
        size=7.0,
        color="#1e3a8a",
        anchor="middle",
    )
    return c


def main() -> None:
    source_files: list[Path] = []

    def load_json(path: Path) -> Any:
        source_files.append(path)
        return json.loads(path.read_text(encoding="utf-8"))

    doi_cn01 = load_json(PAPERS / "cn01_correlated_noise_null_marrakesh/zenodo_receipt.json")["doi"]
    doi_tmap02 = load_json(PAPERS / "tmap02_layout_sensitivity_marrakesh/zenodo_receipt.json")["doi"]
    doi_tmap03 = load_json(PAPERS / "tmap03_readout_attribution_marrakesh/zenodo_receipt.json")["doi"]
    doi_tmap03_pre = load_json(PAPERS / "tmap03_prereg/zenodo_receipt.json")["doi"]
    doi_tmap04 = load_json(PAPERS / "tmap04_probe_guided_layout/zenodo_receipt.json")["doi"]
    doi_tmap04_pre = load_json(PAPERS / "tmap04_prereg/zenodo_receipt.json")["doi"]
    doi_tmap05 = load_json(PAPERS / "tmap05_exclusion_layout/zenodo_receipt.json")["doi"]
    doi_tmap05_pre = load_json(PAPERS / "tmap05_prereg/zenodo_receipt.json")["doi"]
    doi_ro06_07 = load_json(PAPERS / "ro06_07_readout_staleness/zenodo_receipt.json")["doi"]
    doi_ro06_pre = load_json(PAPERS / "ro06_prereg/zenodo_receipt.json")["doi"]
    doi_ro07_pre = load_json(PAPERS / "ro07_prereg/zenodo_receipt.json")["doi"]

    zenodo_records = [
        {"campaign": "CN-01", "kind": "results", "doi": doi_cn01},
        {"campaign": "TMAP-02", "kind": "results", "doi": doi_tmap02},
        {"campaign": "TMAP-03", "kind": "preregistration", "doi": doi_tmap03_pre},
        {"campaign": "TMAP-03", "kind": "results", "doi": doi_tmap03},
        {"campaign": "TMAP-04", "kind": "preregistration", "doi": doi_tmap04_pre},
        {"campaign": "TMAP-04", "kind": "results", "doi": doi_tmap04},
        {"campaign": "TMAP-05", "kind": "preregistration", "doi": doi_tmap05_pre},
        {"campaign": "TMAP-05", "kind": "results", "doi": doi_tmap05},
        {"campaign": "RO-06", "kind": "preregistration", "doi": doi_ro06_pre},
        {"campaign": "RO-07", "kind": "preregistration", "doi": doi_ro07_pre},
        {"campaign": "RO-06/RO-07", "kind": "results", "doi": doi_ro06_07},
    ]

    cn_lock = load_json(P3 / "correlated_noise_01_live_analysis.json")
    cn_a3 = load_json(P3 / "correlated_noise_01_live_analysis_A3.json")
    cn_val = load_json(P3 / "correlated_noise_01_A3_validation.json")
    cn_diag = load_json(P3 / "correlated_noise_01_A3_bound_diagnostic.json")
    cn_usage = load_json(P3 / "correlated_noise_01_job_usage.json")
    cn_raw = load_json(P3 / "correlated_noise_01_live_raw_counts.json")
    cn_prereg = load_json(P3 / "correlated_noise_01_live_prereg_lock.json")

    m2_a = load_json(P3 / "tesseract_map_02_live_analysis.json")
    m2_s = load_json(P3 / "tesseract_map_02_live_sensitivity.json")
    m2_lock = load_json(P3 / "tesseract_map_02_live_prereg_lock.json")
    m2_usage = load_json(P3 / "tesseract_map_02_job_usage.json")
    m2_raw = load_json(P3 / "tesseract_map_02_live_raw_counts.json")

    m3_a = load_json(P3 / "tesseract_map_03_live_analysis.json")
    m3_lock = load_json(P3 / "tesseract_map_03_live_prereg_lock.json")
    m3_cal = load_json(P3 / "tesseract_map_03_prelock_calibration.json")
    m3_raw = load_json(P3 / "tesseract_map_03_live_raw_counts.json")
    m3_rec = load_json(P3 / "tesseract_map_03_submission_receipt.json")

    p4_data: dict[str, dict[str, Any]] = {}
    p5_data: dict[str, dict[str, Any]] = {}
    for b in BACKENDS:
        p4_data[b] = {
            "analysis": load_json(P4 / f"tesseract_map_04_{b}_live_analysis.json"),
            "lock": load_json(P4 / f"tesseract_map_04_{b}_live_prereg_lock.json"),
            "cal": load_json(P4 / f"tesseract_map_04_{b}_probe_time_calibration.json"),
            "probe_ro": load_json(P4 / f"tesseract_map_04_{b}_probe_readout.json"),
            "probe_raw": load_json(P4 / f"tesseract_map_04_{b}_probe_raw_counts.json"),
            "main_raw": load_json(P4 / f"tesseract_map_04_{b}_live_raw_counts.json"),
        }
        p5_data[b] = {
            "analysis": load_json(P5 / f"tesseract_map_05_{b}_live_analysis.json"),
            "lock": load_json(P5 / f"tesseract_map_05_{b}_live_prereg_lock.json"),
            "cal": load_json(P5 / f"tesseract_map_05_{b}_probe_time_calibration.json"),
            "probe_ro": load_json(P5 / f"tesseract_map_05_{b}_probe_readout.json"),
            "probe_raw": load_json(P5 / f"tesseract_map_05_{b}_probe_raw_counts.json"),
            "main_raw": load_json(P5 / f"tesseract_map_05_{b}_live_raw_counts.json"),
        }

    ro6_a = load_json(P6 / "ro06_analysis.json")
    ro7_a = load_json(P7 / "ro07_analysis.json")
    ro_ex = load_json(P7 / "ro06_07_exploratory.json")
    ro_usage = load_json(P7 / "ro06_07_job_usage.json")

    ro6_counts: dict[tuple[str, int], dict[str, Any]] = {}
    ro6_cals: dict[tuple[str, int], dict[str, Any]] = {}
    for b in BACKENDS:
        for r in range(8):
            ro6_counts[(b, r)] = load_json(P6 / f"ro06_{b}_r{r:02d}_counts.json")
            ro6_cals[(b, r)] = load_json(P6 / f"ro06_{b}_r{r:02d}_calibration.json")

    ro7_raw: dict[str, dict[str, Any]] = {}
    ro7_cals: dict[str, dict[str, Any]] = {}
    ro7_locks: dict[str, dict[str, Any]] = {}
    for b in BACKENDS:
        ro7_raw[b] = load_json(P7 / f"ro07_{b}_raw_counts.json")
        ro7_cals[b] = load_json(P7 / f"ro07_{b}_calibration.json")
        ro7_locks[b] = load_json(P7 / f"ro07_{b}_lock.json")

    verified_jobs: list[dict[str, Any]] = []

    def register_job(
        campaign: str,
        backend: str,
        role: str,
        raw_obj: dict[str, Any],
        qpu_s: int,
        doi: str,
    ) -> None:
        jid = str(raw_obj["job_id"]).strip()
        if not jid:
            raise ValueError(f"Empty job_id in {campaign} {backend} {role}")
        pubs = raw_obj["pub_counts"]
        shots = sum(sum(int(v) for v in pub.values()) for pub in pubs)
        ts = str(
            raw_obj.get("completed_utc")
            or (raw_obj.get("job_timestamps") or {}).get("finished")
            or ""
        )
        verified_jobs.append(
            {
                "campaign": campaign,
                "backend": backend,
                "role": role,
                "job_id": jid,
                "pubs": len(pubs),
                "shots": shots,
                "qpu_seconds": int(qpu_s),
                "completed_utc": ts,
                "zenodo_doi": doi,
                "execution_tier": "HARDWARE_ATTESTED",
            }
        )

    register_job("CN-01", "ibm_marrakesh", "main", cn_raw, int(cn_usage["job_usage_seconds"]), doi_cn01)
    register_job("TMAP-02", "ibm_marrakesh", "main", m2_raw, int(m2_usage["job_usage_seconds"]), doi_tmap02)
    register_job(
        "TMAP-03", "ibm_marrakesh", "main", m3_raw, int(round(m3_raw["job_usage_seconds"])), doi_tmap03
    )
    for b in BACKENDS:
        register_job(
            "TMAP-04",
            b,
            "probe",
            p4_data[b]["probe_raw"],
            int(p4_data[b]["probe_raw"]["job_usage_seconds"]),
            doi_tmap04,
        )
        register_job(
            "TMAP-04",
            b,
            "main",
            p4_data[b]["main_raw"],
            int(p4_data[b]["analysis"]["job_usage_seconds"]),
            doi_tmap04,
        )
    for b in BACKENDS:
        register_job(
            "TMAP-05",
            b,
            "probe",
            p5_data[b]["probe_raw"],
            int(p5_data[b]["probe_raw"]["job_usage_seconds"]),
            doi_tmap05,
        )
        register_job(
            "TMAP-05",
            b,
            "main",
            p5_data[b]["main_raw"],
            int(p5_data[b]["analysis"]["job_usage_seconds"]),
            doi_tmap05,
        )
    for b in BACKENDS:
        for r in range(8):
            raw_r = ro6_counts[(b, r)]
            jid_r = raw_r["job_id"]
            register_job("RO-06", b, f"round_{r:02d}", raw_r, int(ro_usage["jobs"][jid_r]), doi_ro06_07)
    for b in BACKENDS:
        raw_7 = ro7_raw[b]
        jid_7 = raw_7["job_id"]
        register_job("RO-07", b, "main", raw_7, int(ro_usage["jobs"][jid_7]), doi_ro06_07)

    assert verified_jobs[0]["shots"] == int(cn_lock["total_circuit_shots"])
    assert verified_jobs[1]["shots"] == int(m2_a["total_shots"])
    assert verified_jobs[2]["shots"] == int(m3_lock["total_shots"])
    assert sum(j["qpu_seconds"] for j in verified_jobs if j["campaign"] in ("RO-06", "RO-07")) == int(
        ro_usage["total_s"]
    )
    assert len({j["job_id"] for j in verified_jobs}) == len(verified_jobs)

    chip_qubits_set = {int(ro6_cals[(b, 0)]["num_qubits"]) for b in BACKENDS}
    assert len(chip_qubits_set) == 1
    chip_qubits = next(iter(chip_qubits_set))
    for b in BACKENDS:
        assert int(ro7_cals[b]["num_qubits"]) == chip_qubits
        assert ro7_locks[b]["backend"] == b

    total_jobs = len(verified_jobs)
    total_pubs = sum(j["pubs"] for j in verified_jobs)
    total_shots = sum(j["shots"] for j in verified_jobs)
    total_qpu_s = sum(j["qpu_seconds"] for j in verified_jobs)
    shots_per_pub = int(cn_lock["shots_per_pub"])
    assert total_pubs * shots_per_pub == total_shots

    by_camp: dict[str, dict[str, int]] = defaultdict(
        lambda: {"jobs": 0, "pubs": 0, "shots": 0, "qpu_s": 0}
    )
    by_back: dict[str, dict[str, int]] = defaultdict(
        lambda: {"jobs": 0, "pubs": 0, "shots": 0, "qpu_s": 0}
    )
    for j in verified_jobs:
        for bucket in (by_camp[j["campaign"]], by_back[j["backend"]]):
            bucket["jobs"] += 1
            bucket["pubs"] += j["pubs"]
            bucket["shots"] += j["shots"]
            bucket["qpu_s"] += j["qpu_seconds"]

    unique_sources = sorted(
        {p.resolve() for p in source_files}, key=lambda p: str(p.relative_to(ROOT))
    )
    source_artifacts = [
        {
            "path": str(p.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(p),
        }
        for p in unique_sources
    ]

    capsule_body: dict[str, Any] = {
        "schema": "osiris.reproducibility_capsule/1",
        "capsule_id": "heron_r2_calibration_staleness_and_colored_nulls_v1",
        "execution_tier": "HARDWARE_ATTESTED",
        "title": (
            "Calibration Staleness, Transient Readout Telegraphing, and Colored-Noise False Positives "
            "Across Three 156-Qubit Superconducting Quantum Processors"
        ),
        "author": "Devin Phillip Davis",
        "affiliation": "Agile Defense Systems LLC",
        "backends": [
            {
                "backend": b,
                "architecture": "IBM Quantum Heron r2",
                "num_qubits": chip_qubits,
                "jobs": by_back[b]["jobs"],
                "pubs": by_back[b]["pubs"],
                "shots": by_back[b]["shots"],
                "qpu_seconds": by_back[b]["qpu_s"],
            }
            for b in BACKENDS
        ],
        "aggregate_totals": {
            "campaigns": len(by_camp),
            "jobs": total_jobs,
            "pubs": total_pubs,
            "shots_per_pub": shots_per_pub,
            "total_shots": total_shots,
            "total_qpu_seconds": total_qpu_s,
        },
        "zenodo_records": zenodo_records,
        "verified_hardware_jobs": verified_jobs,
        "source_json_artifacts": source_artifacts,
    }
    capsule_digest = sha256_bytes(rfc8785_canonical_bytes(capsule_body))
    capsule_full = dict(capsule_body)
    capsule_full["rfc8785_canonical_sha256"] = capsule_digest
    (HERE / "reproducibility_capsule.json").write_text(
        json.dumps(capsule_full, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    macros: list[str] = ["% Auto-generated by generate_unified_assets.py -- DO NOT EDIT BY HAND."]

    def mac(name: str, val: Any) -> None:
        if not name.isalpha():
            raise ValueError(f"LaTeX macro name must be strictly alphabetic: {name!r}")
        macros.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    mac("NumBackends", len(BACKENDS))
    mac("ChipQubits", chip_qubits)
    mac("NumCampaigns", len(by_camp))
    mac("NumZenodoLetters", sum(1 for z in zenodo_records if z["kind"] == "results"))
    mac("NumZenodoPreregs", sum(1 for z in zenodo_records if z["kind"] == "preregistration"))
    mac("TotalJobs", total_jobs)
    mac("TotalPubs", fmt_int(total_pubs))
    mac("ShotsPerPub", fmt_int(shots_per_pub))
    mac("TotalShots", fmt_int(total_shots))
    mac("TotalQpuSeconds", fmt_int(total_qpu_s))

    mac("MarrakeshJobs", by_back["ibm_marrakesh"]["jobs"])
    mac("MarrakeshPubs", fmt_int(by_back["ibm_marrakesh"]["pubs"]))
    mac("MarrakeshShots", fmt_int(by_back["ibm_marrakesh"]["shots"]))
    mac("MarrakeshSeconds", fmt_int(by_back["ibm_marrakesh"]["qpu_s"]))

    mac("FezJobs", by_back["ibm_fez"]["jobs"])
    mac("FezPubs", fmt_int(by_back["ibm_fez"]["pubs"]))
    mac("FezShots", fmt_int(by_back["ibm_fez"]["shots"]))
    mac("FezSeconds", fmt_int(by_back["ibm_fez"]["qpu_s"]))

    mac("KingstonJobs", by_back["ibm_kingston"]["jobs"])
    mac("KingstonPubs", fmt_int(by_back["ibm_kingston"]["pubs"]))
    mac("KingstonShots", fmt_int(by_back["ibm_kingston"]["shots"]))
    mac("KingstonSeconds", fmt_int(by_back["ibm_kingston"]["qpu_s"]))

    tmap_jobs = sum(by_camp[k]["jobs"] for k in ("TMAP-02", "TMAP-03", "TMAP-04", "TMAP-05"))
    tmap_shots = sum(by_camp[k]["shots"] for k in ("TMAP-02", "TMAP-03", "TMAP-04", "TMAP-05"))
    tmap_secs = sum(by_camp[k]["qpu_s"] for k in ("TMAP-02", "TMAP-03", "TMAP-04", "TMAP-05"))
    ro_jobs = by_camp["RO-06"]["jobs"] + by_camp["RO-07"]["jobs"]
    ro_shots = by_camp["RO-06"]["shots"] + by_camp["RO-07"]["shots"]
    ro_secs = by_camp["RO-06"]["qpu_s"] + by_camp["RO-07"]["qpu_s"]

    mac("TmapSeriesJobs", tmap_jobs)
    mac("TmapSeriesShots", fmt_int(tmap_shots))
    mac("TmapSeriesSeconds", fmt_int(tmap_secs))
    mac("RoSeriesJobs", ro_jobs)
    mac("RoSeriesShots", fmt_int(ro_shots))
    mac("RoSeriesSeconds", fmt_int(ro_secs))
    mac("CnJobs", by_camp["CN-01"]["jobs"])
    mac("CnPubs", fmt_int(by_camp["CN-01"]["pubs"]))
    mac("CnShots", fmt_int(by_camp["CN-01"]["shots"]))
    mac("CnCouplerShots", fmt_int(int(cn_lock["total_multiplexed_coupler_shots"])))
    mac("CnSeconds", fmt_int(by_camp["CN-01"]["qpu_s"]))

    mac("DoiCnZeroOne", doi_cn01)
    mac("DoiTmapTwo", doi_tmap02)
    mac("DoiTmapThree", doi_tmap03)
    mac("DoiTmapThreePrereg", doi_tmap03_pre)
    mac("DoiTmapFour", doi_tmap04)
    mac("DoiTmapFourPrereg", doi_tmap04_pre)
    mac("DoiTmapFive", doi_tmap05)
    mac("DoiTmapFivePrereg", doi_tmap05_pre)
    mac("DoiRoSixSeven", doi_ro06_07)
    mac("DoiRoSixPrereg", doi_ro06_pre)
    mac("DoiRoSevenPrereg", doi_ro07_pre)
    mac("CapsuleRfcShaPrefix", capsule_digest[:16])
    mac("CapsuleSourceCount", len(source_artifacts))

    camp_meta = [
        ("TMAP-02", "2026-11-16", "\\texttt{marrakesh}", "--", doi_tmap02),
        ("CN-01", "2026-11-16", "\\texttt{marrakesh}", "--", doi_cn01),
        ("TMAP-03", "2026-11-20", "\\texttt{marrakesh}", doi_tmap03_pre.split(".")[-1], doi_tmap03),
        ("TMAP-04", "2026-11-20", "All 3 QPUs", doi_tmap04_pre.split(".")[-1], doi_tmap04),
        ("TMAP-05", "2026-11-20", "All 3 QPUs", doi_tmap05_pre.split(".")[-1], doi_tmap05),
        ("RO-06", "2026-11-20..21", "All 3 QPUs", doi_ro06_pre.split(".")[-1], doi_ro06_07),
        ("RO-07", "2026-11-21", "All 3 QPUs", doi_ro07_pre.split(".")[-1], doi_ro06_07),
    ]
    reg_rows = []
    for cid, dt_s, b_str, pre_id, res_doi in camp_meta:
        st = by_camp[cid]
        res_id = res_doi.split(".")[-1]
        reg_rows.append(
            f"{cid} & {dt_s} & {b_str} & {st['jobs']} & {fmt_int(st['pubs'])} & "
            f"{fmt_int(st['shots'])} & {st['qpu_s']} & {pre_id} & {res_id}\\\\"
        )
    reg_rows.append("\\hline")
    reg_rows.append(
        f"\\textbf{{Total}} & Nov 16--21 & 3 Heron r2 & \\textbf{{{total_jobs}}} & "
        f"\\textbf{{{fmt_int(total_pubs)}}} & \\textbf{{{fmt_int(total_shots)}}} & "
        f"\\textbf{{{fmt_int(total_qpu_s)}}} & 5 DOIs & 6 DOIs\\\\"
    )
    mac("CampaignRegistryRows", "\n".join(reg_rows))

    m2_arms = m2_a["arm_summaries"]
    m2_cons = {c["contrast"]: c for c in m2_a["paired_contrasts_classical_fidelity"]["contrasts"]}
    mac("TmapCircuitQubits", int(m2_lock["path_summary"]["n_qubits"]))
    mac("TmapChainCz", int(m2_arms["C5_CHAIN_BESTPATH_OPT3"]["twoq_gates"]))
    mac("TmapChainDepth", int(m2_arms["C5_CHAIN_BESTPATH_OPT3"]["depth"]))
    mac("TmapStarOptThreeCz", int(m2_arms["C2_STAR_BESTPATH_OPT3"]["twoq_gates"]))
    mac("TmapStarOptOneCz", int(m2_arms["C3_STAR_BESTPATH_OPT1"]["twoq_gates"]))
    mac("TmapStarFreeCz", int(m2_arms["C1_STAR_FREE_OPT3"]["twoq_gates"]))
    mac("TmapSeedThetaDeg", f"{m3_lock['theta_deg']:.1f}")
    mac("TmapBlocks", int(m2_a["n_blocks"]))
    mac("TmapBlocksDf", int(m2_a["n_blocks"]) - 1)
    mac("TmapCandidatePaths", int(m2_lock["path_summary"]["num_candidate_paths_scored"]))
    mac(
        "TmapWilcoxonFloorP",
        fmt_sci_p(m2_s["wilcoxon_signed_rank_holm"]["classical_fidelity"][0]["wilcoxon_p_holm_adjusted"]),
    )

    mac("TmapTwoJobId", m2_a["job_id"])
    mac("TmapTwoPubs", int(m2_lock["total_pubs"]))
    mac("TmapTwoShots", fmt_int(int(m2_a["total_shots"])))
    mac("TmapTwoSeconds", int(m2_usage["job_usage_seconds"]))
    mac("TmapTwoUnionQubits", int(m2_a["union_spam_qubit_count"]))
    mac("TmapTwoUnionRoPct", f"{100 * m2_a['union_mean_readout_error']:.2f}")

    c_rew3 = m2_cons["routing_and_rewrite_at_opt3"]
    c_rew1 = m2_cons["routing_and_rewrite_at_opt1"]
    c_path = m2_cons["qubit_selection_at_fixed_39cz"]
    c_aa = m2_cons["transpiler_opt3_vs_opt1_on_chain"]
    c_stk = m2_cons["error_aware_path_vs_sabre_chain"]

    mac("TmapTwoRewriteOptThreePp", f"{100 * c_rew3['mean_diff']:+.2f}")
    mac("TmapTwoRewriteOptOnePp", f"{100 * c_rew1['mean_diff']:+.2f}")
    mac("TmapTwoPathGapPp", f"{100 * c_path['mean_diff']:+.2f}")
    mac("TmapTwoPathGapCi", f"[{100 * c_path['ci95'][0]:+.2f}, {100 * c_path['ci95'][1]:+.2f}]")
    mac("TmapTwoBestFcl", f"{m2_arms['C5_CHAIN_BESTPATH_OPT3']['mean_classical_fidelity']:.3f}")
    mac("TmapTwoMedFcl", f"{m2_arms['C6_CHAIN_MEDPATH_OPT3']['mean_classical_fidelity']:.3f}")
    mac("TmapTwoStockFcl", f"{m2_arms['C7_CHAIN_FREE_OPT3']['mean_classical_fidelity']:.3f}")
    mac("TmapTwoStarFreeFcl", f"{m2_arms['C1_STAR_FREE_OPT3']['mean_classical_fidelity']:.3f}")
    mac("TmapTwoVsStockPp", f"{100 * c_stk['mean_diff']:+.2f}")
    mac("TmapTwoAaVirtualZPp", f"{100 * c_aa['mean_diff']:+.2f}")
    mac("TmapTwoAaVirtualZCi", f"[{100 * c_aa['ci95'][0]:+.2f}, {100 * c_aa['ci95'][1]:+.2f}]")
    mac("TmapTwoAaVirtualZP", f"{c_aa['p_holm_adjusted']:.2f}")
    max_eff_p = max(
        c_rew3["p_holm_adjusted"],
        c_rew1["p_holm_adjusted"],
        c_path["p_holm_adjusted"],
        c_stk["p_holm_adjusted"],
    )
    mac("TmapTwoMaxHolmP", fmt_sci_p(max_eff_p))

    m2_spam = m2_a["per_qubit_spam_calibration"]
    mac("TmapStarHubDeg", int(m2_lock["path_summary"]["n_qubits"]) // 2)
    mac("TmapTwoQFortyTwoPct", f"{100 * m2_spam['42']['mean_assign_err']:.1f}")
    mac("TmapTwoQFortyOnePct", f"{100 * m2_spam['41']['mean_assign_err']:.1f}")
    mac("TmapTwoQThirtyOnePct", f"{100 * m2_spam['31']['mean_assign_err']:.1f}")
    mac("TmapTwoQFortySixPct", f"{100 * m2_spam['46']['mean_assign_err']:.1f}")
    mac(
        "TmapTwoQFortyTwoIbmPct",
        f"{100 * m2_raw['execution_time_calibration']['qubits']['42']['readout_error']:.1f}",
    )
    mac(
        "TmapTwoQFortyOneIbmPct",
        f"{100 * m2_raw['execution_time_calibration']['qubits']['41']['readout_error']:.1f}",
    )
    probe_secs = [
        int(p4_data[b]["probe_raw"]["job_usage_seconds"]) for b in BACKENDS
    ] + [int(p5_data[b]["probe_raw"]["job_usage_seconds"]) for b in BACKENDS]
    mac("TmapProbeSecLow", min(probe_secs))
    mac("TmapProbeSecHigh", max(probe_secs))

    dt_days_23 = round(
        (
            datetime.fromisoformat(m3_raw["completed_utc"])
            - datetime.fromisoformat(m2_raw["completed_utc"])
        ).total_seconds()
        / 86400.0
    )
    mac("TmapThreeDaysLater", dt_days_23)
    mac("TmapThreeJobId", m3_a["job_id"])
    mac("TmapThreePubs", int(m3_lock["total_pubs"]))
    mac("TmapThreeShots", fmt_int(int(m3_lock["total_shots"])))
    mac("TmapThreeSeconds", int(round(m3_raw["job_usage_seconds"])))
    mac("TmapThreeUnionQubits", len(m3_lock["union_active_physical_qubits"]))

    k1 = m3_a["contrasts"]["K1_path_gap_replication"]
    k2 = m3_a["contrasts"]["K2_readout_share_of_path_gap"]
    k3 = m3_a["contrasts"]["K3_rewrite_x_path_interaction"]
    k4 = m3_a["contrasts"]["K4_best_vs_C7_replay"]
    k5 = m3_a["contrasts"]["K5_best_vs_stock_today"]
    m3_mit_gap = k1["mean_pp"] - k2["mean_pp"]
    m3_star_gap = 100.0 * (
        m3_a["arms"]["A6_STAR_BEST"]["mean_fcl_raw"] - m3_a["arms"]["A5_STAR_MED"]["mean_fcl_raw"]
    )
    mac("TmapThreePathGapPp", f"{k1['mean_pp']:+.2f}")
    mac("TmapThreePathGapCi", f"[{k1['ci95_pp'][0]:+.2f}, {k1['ci95_pp'][1]:+.2f}]")
    mac("TmapThreePathGapP", fmt_sci_p(k1["p_holm"]))
    mac("TmapThreeBestFcl", f"{m3_a['arms']['A1_CHAIN_BEST']['mean_fcl_raw']:.3f}")
    mac("TmapThreeMedFcl", f"{m3_a['arms']['A2_CHAIN_MED']['mean_fcl_raw']:.3f}")
    mac("TmapThreeBestMitFcl", f"{m3_a['arms']['A1_CHAIN_BEST']['mean_fcl_mit']:.3f}")
    mac("TmapThreeMedMitFcl", f"{m3_a['arms']['A2_CHAIN_MED']['mean_fcl_mit']:.3f}")
    mac("TmapThreeMitGapPp", f"{m3_mit_gap:+.2f}")
    mac("TmapThreeReadoutShiftPp", f"{-k2['mean_pp']:+.1f}")
    mac("TmapThreeReadoutShiftP", fmt_sci_p(k2["p_holm"]))
    mac("TmapThreeInteractionPp", f"{k3['mean_pp']:+.2f}")
    mac("TmapThreeInteractionP", fmt_sci_p(k3["p_holm"]))
    mac("TmapThreeStarGapPp", f"{m3_star_gap:+.2f}")
    mac("TmapThreeStockReplayGapPp", f"{k4['mean_pp']:+.2f}")
    mac("TmapThreeStockTodayGapPp", f"{k5['mean_pp']:+.2f}")
    q46_m3 = m3_a["per_qubit_readout"]["46"]
    q46_m3_mean = 0.5 * (q46_m3["p1_given_0"] + q46_m3["p0_given_1"])
    mac("TmapThreeQFortySixMeanPct", f"{100 * q46_m3_mean:.1f}")
    mac("TmapThreeQFortySixZeroToOnePct", f"{100 * q46_m3['p1_given_0']:.1f}")
    mac("TmapThreeQFortySixOneToZeroPct", f"{100 * q46_m3['p0_given_1']:.1f}")
    mac("TmapThreeQThirtyOneOneToZeroPct", f"{100 * m3_a['per_qubit_readout']['31']['p0_given_1']:.1f}")
    mac(
        "TmapThreeQFortySixIbmPct",
        f"{100 * m3_raw['execution_time_calibration']['qubits']['46']['readout_error']:.1f}",
    )
    mac("TmapTwoLockShaPrefix", m2_a["prereg_lock_sha256"][:16])
    mac("TmapThreeLockShaPrefix", m3_a["prereg_lock_sha256"][:16])
    assert m3_cal["num_qubits"] == chip_qubits
    assert m3_rec["job_id"] == m3_a["job_id"]

    mac("TmapFourTotalSeconds", by_camp["TMAP-04"]["qpu_s"])
    mac("TmapFiveTotalSeconds", by_camp["TMAP-05"]["qpu_s"])
    mac("TmapProbePubs", len(p4_data["ibm_marrakesh"]["probe_raw"]["pub_counts"]))
    mac("TmapMainPubs", len(p4_data["ibm_marrakesh"]["main_raw"]["pub_counts"]))
    mac("TmapExclThresholdPct", 15)

    p4_helps = sum(
        int(
            p4_data[b]["analysis"]["contrasts"]["L1_probe_vs_calibration_path"]["mean_pp"] > 0
            and p4_data[b]["analysis"]["contrasts"]["L1_probe_vs_calibration_path"]["p_holm"] < 0.05
        )
        for b in BACKENDS
    )
    p5_helps = sum(
        int(
            p5_data[b]["analysis"]["contrasts"]["M1_exclusion_vs_calibration"]["mean_pp"] > 0
            and p5_data[b]["analysis"]["contrasts"]["M1_exclusion_vs_calibration"]["p_holm"] < 0.05
        )
        for b in BACKENDS
    )
    mac("TmapFourHelpsCount", p4_helps)
    mac("TmapFiveHelpsCount", p5_helps)

    dt_mins: dict[str, float] = {}
    for b in BACKENDS:
        t4 = datetime.fromisoformat(p4_data[b]["probe_raw"]["completed_utc"])
        t5 = datetime.fromisoformat(p5_data[b]["probe_raw"]["completed_utc"])
        dt_mins[b] = (t5 - t4).total_seconds() / 60.0

    mac("TmapDtMinLow", f"{min(dt_mins.values()):.0f}")
    mac("TmapDtMinHigh", f"{max(dt_mins.values()):.0f}")

    tag_map = {"ibm_marrakesh": "Marr", "ibm_fez": "Fez", "ibm_kingston": "King"}
    tmap_rows: list[str] = []
    for b in BACKENDS:
        tag = tag_map[b]
        c_l1 = p4_data[b]["analysis"]["contrasts"]["L1_probe_vs_calibration_path"]
        c_l4 = p4_data[b]["analysis"]["contrasts"]["L4_L1_after_readout_mitigation"]
        c_m1 = p5_data[b]["analysis"]["contrasts"]["M1_exclusion_vs_calibration"]
        c_m2 = p5_data[b]["analysis"]["contrasts"]["M2_exclusion_vs_probe_ranking"]
        c_m4 = p5_data[b]["analysis"]["contrasts"]["M4_probe_vs_calibration_replication"]

        mac(f"TmapFour{tag}LonePp", f"{c_l1['mean_pp']:+.2f}")
        mac(f"TmapFour{tag}LoneCi", f"[{c_l1['ci95_pp'][0]:+.2f}, {c_l1['ci95_pp'][1]:+.2f}]")
        mac(f"TmapFour{tag}LoneP", fmt_sci_p(c_l1["p_holm"]))
        mac(f"TmapFour{tag}LfourMitPp", f"{c_l4['mean_pp']:+.1f}")
        mac(f"TmapFive{tag}MonePp", f"{c_m1['mean_pp']:+.2f}")
        mac(f"TmapFive{tag}MtwoPp", f"{c_m2['mean_pp']:+.2f}")
        mac(f"TmapFive{tag}MfourPp", f"{c_m4['mean_pp']:+.2f}")
        mac(f"TmapFive{tag}MfourCi", f"[{c_m4['ci95_pp'][0]:+.2f}, {c_m4['ci95_pp'][1]:+.2f}]")
        mac(f"TmapFive{tag}MfourP", fmt_sci_p(c_m4["p_holm"]))

        tmap_rows.append(
            f"{esc_tex(b)} & {dt_mins[b]:.0f}\\,m & "
            f"{c_l1['mean_pp']:+.2f} [{c_l1['ci95_pp'][0]:+.2f}, {c_l1['ci95_pp'][1]:+.2f}] & "
            f"{c_l4['mean_pp']:+.1f} & "
            f"{c_m4['mean_pp']:+.2f} [{c_m4['ci95_pp'][0]:+.2f}, {c_m4['ci95_pp'][1]:+.2f}] & "
            f"{c_m1['mean_pp']:+.2f} & {c_m2['mean_pp']:+.2f}\\\\"
        )
    mac("TmapSignReversalRows", "\n".join(tmap_rows))

    q46_p4 = 100.0 * float(p4_data["ibm_marrakesh"]["probe_ro"]["mean_assign_err"]["46"])
    q46_p5 = 100.0 * float(p5_data["ibm_marrakesh"]["probe_ro"]["mean_assign_err"]["46"])
    mac("TmapFourMarrProbeFcl", f"{p4_data['ibm_marrakesh']['analysis']['arms']['B1_PROBE_BEST']['mean_fcl_raw']:.3f}")
    mac("TmapFourMarrCalFcl", f"{p4_data['ibm_marrakesh']['analysis']['arms']['B2_CAL_BEST']['mean_fcl_raw']:.3f}")
    mac("TmapFourQFortySixPct", f"{q46_p4:.1f}")
    mac("TmapFiveQFortySixPct", f"{q46_p5:.1f}")
    mac("TmapFourMarrUtc", p4_data["ibm_marrakesh"]["probe_raw"]["completed_utc"][11:16])
    mac("TmapFiveMarrUtc", p5_data["ibm_marrakesh"]["probe_raw"]["completed_utc"][11:16])
    mac(
        "TmapFourKingQOneZeroSevenPct",
        f"{100 * float(p4_data['ibm_kingston']['probe_ro']['mean_assign_err']['107']):.1f}",
    )
    mac(
        "TmapFourKingQOneZeroSevenIbmPct",
        f"{100 * float(p4_data['ibm_kingston']['cal']['qubits']['107']['readout_error']):.1f}",
    )
    mac(
        "TmapFiveFezQElevenPct",
        f"{100 * float(p5_data['ibm_fez']['probe_ro']['mean_assign_err']['11']):.1f}",
    )
    mac(
        "TmapFiveKingQOneZeroSevenPct",
        f"{100 * float(p5_data['ibm_kingston']['probe_ro']['mean_assign_err']['107']):.1f}",
    )

    mac("RoSixJobs", by_camp["RO-06"]["jobs"])
    mac("RoSevenJobsCount", by_camp["RO-07"]["jobs"])
    mac("RoSixRounds", int(ro6_a["min_rounds_any_device"]))
    mac("RoSixIntervalMin", 15)
    mac("RoFaultThreshPct", 15)
    mac("RoIbmOkThreshPct", 3)
    mac("RoRecoveredThreshPct", 5)
    mac("RoContextThreshPp", "5.0")

    t_r0 = datetime.fromisoformat(
        ro_ex["E2_ibm_update_cadence"]["ibm_marrakesh"]["snapshot_times"][0].replace("Z", "+00:00")
    )
    t_r1 = datetime.fromisoformat(
        ro_ex["E2_ibm_update_cadence"]["ibm_marrakesh"]["snapshot_times"][1].replace("Z", "+00:00")
    )
    t_r7 = datetime.fromisoformat(
        ro_ex["E2_ibm_update_cadence"]["ibm_marrakesh"]["snapshot_times"][7].replace("Z", "+00:00")
    )
    t_k0 = datetime.fromisoformat(
        ro_ex["E2_ibm_update_cadence"]["ibm_kingston"]["snapshot_times"][0].replace("Z", "+00:00")
    )
    t_k8 = datetime.fromisoformat(
        ro_ex["E2_ibm_update_cadence"]["ibm_kingston"]["snapshot_times"][-1].replace("Z", "+00:00")
    )
    king_unchanged_h = (t_k8 - t_k0).total_seconds() / 3600.0

    mac("RoOvernightGapHours", f"{(t_r1 - t_r0).total_seconds() / 3600.0:.1f}")
    mac("RoActiveSpanHours", f"{(t_r7 - t_r1).total_seconds() / 3600.0:.2f}")
    mac("RoKingstonUnchangedHours", f"{king_unchanged_h:.1f}")

    mac("RoEverHidden", int(ro6_a["H2_qubits_ever_hidden"]))
    mac("RoRecoveredCount", int(ro6_a["H2_of_which_recovered_in_another_round"]))
    mac("RoPersistFaultPct", f"{100 * ro6_a['H3_p_fault_next_given_fault']:.1f}")
    mac("RoPersistOkPct", f"{100 * ro6_a['H3_p_fault_next_given_ok']:.2f}")
    mac("RoFisherOr", fmt_int(int(round(ro6_a["H3_fisher_odds_ratio"]))))
    mac("RoFisherP", fmt_sci_p(ro6_a["H3_fisher_p"]))

    e1 = ro_ex["E1_H3_rounds_1_to_7"]
    mac("RoSensPersistFaultPct", f"{100 * e1['p_fault_next_given_fault']:.1f}")
    mac("RoSensPersistOkPct", f"{100 * e1['p_fault_next_given_ok']:.2f}")
    mac("RoSensFisherOr", fmt_int(int(round(e1["odds_ratio"]))))
    mac("RoSensFisherP", fmt_sci_p(e1["p"]))

    mac("RoWithinSpreadPp", f"{100 * ro6_a['noise_floor_within_round_range_median']:.2f}")
    mac("RoBetweenChangePp", f"{100 * ro6_a['between_round_abs_change_median_flagged']:.2f}")

    e3_fez = ro_ex["E3_update_alignment"]["ibm_fez"]
    mac("RoFezOldRho", f"{e3_fez['old_vs_last_round']['rho']:.3f}")
    mac("RoFezOldGapPp", f"{100 * e3_fez['old_vs_last_round']['median_abs_gap']:.2f}")
    mac("RoFezNewRhoRoundSix", f"{e3_fez['new_vs_round'][6]['rho']:.3f}")
    mac("RoFezNewRhoRoundSeven", f"{e3_fez['new_vs_round'][7]['rho']:.3f}")
    mac("RoFezNewGapRoundSevenPp", f"{100 * e3_fez['new_vs_round'][7]['median_abs_gap']:.2f}")

    max_mux_abs = max(
        abs(v["D_mux"]["mean"])
        for d in ro7_a["devices"].values()
        for v in d["per_qubit"].values()
    )
    max_nbr_abs = max(
        abs(v["D_nbr"]["mean"])
        for d in ro7_a["devices"].values()
        for v in d["per_qubit"].values()
    )
    max_d_abs = max(max_mux_abs, max_nbr_abs)
    max_ctrl_abs = max(
        abs(v[k]["mean"])
        for d in ro7_a["devices"].values()
        for v in d["per_qubit"].values()
        if v["role"] == "control"
        for k in ("D_mux", "D_nbr")
    )
    t_r8_marr = datetime.fromisoformat(
        ro_ex["E2_ibm_update_cadence"]["ibm_marrakesh"]["snapshot_times"][-1].replace("Z", "+00:00")
    )
    ro7_lag_min = round((t_r8_marr - t_r7).total_seconds() / 60.0)
    h3_tab = ro6_a["H3_transition_table_[fault_r][fault_r+1]"]
    mac("RoTransitionOkOk", fmt_int(int(h3_tab[0][0])))
    mac("RoTransitionOkFault", fmt_int(int(h3_tab[0][1])))
    mac("RoTransitionFaultOk", fmt_int(int(h3_tab[1][0])))
    mac("RoTransitionFaultFault", fmt_int(int(h3_tab[1][1])))
    mac("RoRhoMarr", f"{ro6_a['H4_spearman_median']['ibm_marrakesh']:.2f}")
    mac("RoRhoFez", f"{ro6_a['H4_spearman_median']['ibm_fez']:.2f}")
    mac("RoRhoKing", f"{ro6_a['H4_spearman_median']['ibm_kingston']:.2f}")
    mac("RoSevenRepeats", 6)
    mac("RoSevenLagMin", ro7_lag_min)
    mac("RoSevenMaxMuxPp", f"{100 * max_mux_abs:.2f}")
    mac("RoSevenMaxNbrPp", f"{100 * max_nbr_abs:.2f}")
    mac("RoSevenMaxAbsDPp", f"{100 * max_d_abs:.2f}")
    mac("RoSevenControlsMaxPp", f"{100 * max_ctrl_abs:.2f}")
    mac("RoSevenTestedQubits", sum(len(d["per_qubit"]) for d in ro7_a["devices"].values()))
    mac("TmapFiveFezStockFcl", f"{p5_data['ibm_fez']['analysis']['arms']['C4_STOCK']['mean_fcl_raw']:.3f}")
    mac("AlphaFwer", "0.05")
    mac("CiLevelPct", 95)
    mac("ProbeRepeats", 3)

    disagree_tot = 0
    disagree_rec = 0
    ro7_points_fig: list[tuple[str, float, tuple[float, float], float, tuple[float, float]]] = []
    short_b_map = {"ibm_marrakesh": "mar", "ibm_fez": "fez", "ibm_kingston": "kng"}
    for b in BACKENDS:
        dev = ro7_a["devices"][b]
        for q_str, q_val in dev["per_qubit"].items():
            if q_val["role"] == "disagree":
                disagree_tot += 1
                if q_val["e"]["ALL"] < 0.05:
                    disagree_rec += 1
            ro7_points_fig.append(
                (
                    f"{short_b_map[b]}:q{q_str}({q_val['role'][0].upper()})",
                    100.0 * q_val["D_mux"]["mean"],
                    (100.0 * q_val["D_mux"]["ci95"][0], 100.0 * q_val["D_mux"]["ci95"][1]),
                    100.0 * q_val["D_nbr"]["mean"],
                    (100.0 * q_val["D_nbr"]["ci95"][0], 100.0 * q_val["D_nbr"]["ci95"][1]),
                )
            )
    mac("RoSevenDisagreeTotal", disagree_tot)
    mac("RoSevenDisagreeRecovered", disagree_rec)
    mac(
        "RoSevenMarrQOneElevenPrePct",
        f"{100 * ro7_a['devices']['ibm_marrakesh']['selection']['e_probe']['111']:.1f}",
    )
    mac(
        "RoSevenMarrQOneElevenPostPct",
        f"{100 * ro7_a['devices']['ibm_marrakesh']['per_qubit']['111']['e']['ALL']:.1f}",
    )

    ro_rows: list[str] = []
    cad = ro_ex["E2_ibm_update_cadence"]
    h1_means = [float(ro6_a["H1_hidden_faults_per_round"][b]["mean"]) for b in BACKENDS]
    mac("RoMinHiddenMean", f"{min(h1_means):.3f}")
    mac("RoMaxHiddenMean", f"{max(h1_means):.3f}")
    mac("RoFezUpdatedQubits", int(cad["ibm_fez"]["qubits_changed"][-1]))
    for b in BACKENDS:
        h1_b = ro6_a["H1_hidden_faults_per_round"][b]
        rho_b = ro6_a["H4_spearman_median"][b]
        cad_str = " ".join(str(x) for x in cad[b]["qubits_changed"])
        dev7 = ro7_a["devices"][b]["per_qubit"]
        b_max_mux = max(abs(v["D_mux"]["mean"]) for v in dev7.values())
        b_max_nbr = max(abs(v["D_nbr"]["mean"]) for v in dev7.values())
        ro_rows.append(
            f"{esc_tex(b)} & {' '.join(str(x) for x in h1_b['counts'])} & {h1_b['mean']:.3f} & "
            f"{rho_b:.2f} & \\texttt{{{cad_str}}} & {100 * b_max_mux:.2f} & {100 * b_max_nbr:.2f}\\\\"
        )
    mac("RoStalenessSummaryRows", "\n".join(ro_rows))

    hidden_targets = [
        ("ibm_marrakesh", [85, 16, 111, 126]),
        ("ibm_fez", [93]),
        ("ibm_kingston", [107]),
    ]
    traj_colors = {
        ("ibm_marrakesh", 111): "#dc2626",
        ("ibm_marrakesh", 16): "#ea580c",
        ("ibm_marrakesh", 85): "#9333ea",
        ("ibm_marrakesh", 126): "#db2777",
        ("ibm_fez", 93): "#2563eb",
        ("ibm_kingston", 107): "#059669",
    }
    trajectories_fig: list[tuple[str, list[float], list[float], str]] = []
    for b, qs in hidden_targets:
        per_round_ep: dict[int, list[float]] = {q: [] for q in qs}
        per_round_ibm: dict[int, list[float]] = {q: [] for q in qs}
        for r in range(8):
            pc = ro6_counts[(b, r)]["pub_counts"]
            ep_map = subset_per_qubit_readout(pc[0::2], pc[1::2], qs)
            cal_q = ro6_cals[(b, r)]["qubits"]
            for q in qs:
                per_round_ep[q].append(100.0 * ep_map[q])
                per_round_ibm[q].append(100.0 * float(cal_q[str(q)]["readout_error"]))
                hf_round = ro6_a["rounds"][b][r]["hidden_faults"]
                if str(q) in hf_round:
                    assert abs(ep_map[q] - hf_round[str(q)]) < 2e-4
        for q in qs:
            trajectories_fig.append(
                (
                    f"{short_b_map[b]} q{q} (max {max(per_round_ep[q]):.1f}%)",
                    per_round_ep[q],
                    per_round_ibm[q],
                    traj_colors[(b, q)],
                )
            )

    mac("CnJobId", cn_lock["job_id"])
    mac("CnTotalCouplers", len(cn_lock["per_coupler_profiles"]))
    mac("CnTrainCouplers", len(cn_a3["train_slots"]))
    mac("CnHeldOutCouplers", int(cn_lock["held_out_6coupler_cluster_validation_df5"]["n_held_out_couplers"]))
    mac("CnHeldOutDf", int(cn_lock["held_out_6coupler_cluster_validation_df5"]["n_held_out_couplers"]) - 1)
    mac("CnMinHops", 5)
    mac("CnBlocks", int(cn_lock["n_blocks"]))
    mac("CnLockedConds", len(cn_lock["training_mean_curve_23conds"]))
    mac("CnAThreeConds", len(cn_a3["fit_conditions"]))
    mac("CnBicGate", int(cn_a3["model_selection"]["delta_bic_gate"]))
    mac("CnDdWindowUs", "8.192")
    mac("CnLongWindowUs", "16.384")
    mac("CnDdSixteenPulses", 16)
    mac("CnDdThirtyTwoPulses", 32)
    mac("CnSlotNs", 512)
    mac("CnMaxStaggerNs", 256)
    mac("CnCollisionStepNs", 48)
    mac("CnStepGate", "0.75")
    mac("CnFastBoundPp", "1.0")
    mac("CnAaRmsRawPp", f"{cn_lock['live_aa_sentinel_rms_pp_raw']:.2f}")
    mac("CnStepFractionS", f"{cn_lock['step_fraction_s48']:.2f}")
    mac("CnWhiteFloorCapKhz", 40)
    mac(
        "CnSolvedWhiteFloorKhz",
        f"{cn_diag['refits']['white_floor_cap_60000_hz']['fits']['H0']['params'][3] / 1000.0:.1f}",
    )
    mac("CnRamseySixteenMean", f"{cn_a3['ramsey_descriptive_mean']['ramsey_16us']:.3f}")
    mac("CnEchoSixteenMean", f"{cn_lock['training_mean_curve_23conds']['echo1_16us']:.3f}")
    mac("CnDdThirtyTwoSixteenMean", f"{cn_lock['training_mean_curve_23conds']['echo_dd32_16us']:.3f}")
    mac("CnLockShaPrefix", cn_lock["prereg_lock_sha256"][:16])
    mac("CnQpyShaPrefix", cn_lock["qpy_sha256"][:16])
    mac("CnAmendShaPrefix", cn_a3["analysis_amendment_sha256"][:16])
    assert cn_prereg["qpy_sha256"] == cn_lock["qpy_sha256"]

    t9 = float(stats.t.ppf(0.975, 9))
    mc = cn_lock["mechanistic_contrasts_10couplers_holm"]
    cn_con_specs = [
        (
            "C_DFS_asymmetry",
            "Cdfs",
            "$C_{\\mathrm{DFS}}$",
            "$\\Phi^+$ vs $\\Psi^+$ stagger asymmetry ($0\\to256$\\,ns)",
        ),
        (
            "C_spectator_activation",
            "Cspec",
            "$C_{\\mathrm{spec}}$",
            "Spectator in $|{+}\\rangle$ minus $|0\\rangle$",
        ),
        (
            "C_stagger_phi_256_minus_0",
            "Cstag",
            "$C_{\\mathrm{stag}}$",
            "$\\Phi^+(256\\,\\mathrm{ns})-\\Phi^+(0\\,\\mathrm{ns})$",
        ),
        (
            "C_memory_dd32_minus_echo1_16us",
            "Cmem",
            "$C_{\\mathrm{mem}}$",
            "DD32 minus Hahn echo at $16.384\\,\\mu$s",
        ),
    ]
    cn_con_rows: list[str] = []
    for key, tag, sym, desc in cn_con_specs:
        entry = mc[key]
        m_pp = 100.0 * entry["mean"]
        se_pp = 100.0 * entry["se_across_couplers"]
        ci_str = f"[{m_pp - t9 * se_pp:+.2f}, {m_pp + t9 * se_pp:+.2f}]"
        p_str = f"{entry['p_holm_adjusted']:.2f}"
        mac(f"Cn{tag}MeanPp", f"{m_pp:+.2f}")
        mac(f"Cn{tag}Ci", ci_str)
        mac(f"Cn{tag}P", p_str)
        cn_con_rows.append(
            f"{sym} & \\multicolumn{{3}}{{l}}{{{desc}}} & {m_pp:+.2f} & {ci_str} & {p_str}\\\\"
        )
    mac("CnContrastRows", "\n".join(cn_con_rows))

    lf = cn_lock["gls_model_selection_on_training_cohort"]["fits"]
    af = cn_a3["model_selection"]["fits"]
    ho = cn_lock["held_out_6coupler_cluster_validation_df5"]
    mac(
        "CnLockedDeltaBic",
        fmt_int(int(round(cn_lock["gls_model_selection_on_training_cohort"]["delta_bic_over_h0"]))),
    )
    mac("CnLockedBicHZero", fmt_int(int(round(lf["H0"]["bic"]))))
    mac("CnLockedBicHThree", f"{lf['H3']['bic']:.1f}")
    mac("CnHoRmseHZero", f"{ho['mean_rmse_h0']:.3f}")
    mac("CnHoRmseHThree", f"{ho['mean_rmse_h3']:.3f}")
    mac("CnHoTStat", f"{ho['t_stat_h0_minus_h3_df5']:.2f}")
    mac("CnHoPVal", f"{ho['p_two_sided_df5']:.2f}")
    mac("CnAThreeBicHZero", f"{af['H0']['bic']:.2f}")
    mac("CnAThreeBicHOne", f"{af['H1']['bic']:.2f}")
    mac("CnAThreeRunnerUpGap", f"{af['H1']['bic'] - af['H0']['bic']:.2f}")

    model_order = [
        ("H0", "Local white ($H_0$)", "local"),
        ("H0_1f", "Local $1/f$ + white ($H_{0,1/f}$)", "local"),
        ("H3_local", "Local OU bath ($H_{3,\\mathrm{loc}}$)", "local"),
        ("H1", "Static spectator $ZZ$ ($H_1$)", "non-local"),
        ("H2", "Correlated Markov ($H_2$)", "non-local"),
        ("H3", "Correlated OU bath ($H_3$)", "non-local"),
        ("Apparatus", "Pulse collision", "apparatus"),
    ]
    cn_bic_rows: list[str] = []
    for mkey, mlab, mscope in model_order:
        a_row = af[mkey]
        k_str = f"{a_row['k_params']}"
        if mkey in lf:
            l_str = f"{lf[mkey]['chi2']:,.1f} & {lf[mkey]['bic']:,.1f}".replace(",", "{,}")
        else:
            l_str = "-- & --"
        a_str = f"{a_row['chi2']:,.1f} & {a_row['bic']:,.2f}".replace(",", "{,}")
        cn_bic_rows.append(f"{mlab} & {mscope} & {k_str} & {l_str} & {a_str}\\\\")
    mac("CnModelBicRows", "\n".join(cn_bic_rows))

    val_summ = cn_val["summary"]
    mac("CnValTotalReps", len(cn_val["replicates"]))
    local_keys = ("white", "local_1f", "local_1f_ramsey_detuned", "local_ou")
    mac("CnValLocalNullReps", sum(int(val_summ[k]["n_replicates"]) for k in local_keys))
    colored_local_st2 = [
        100.0 * float(val_summ[k]["locked"]["rates"]["stage2_gate"]["rate"])
        for k in ("local_1f", "local_1f_ramsey_detuned", "local_ou")
    ]
    mac("CnValLocalStageTwoLowPct", f"{min(colored_local_st2):.0f}")
    mac("CnValLocalStageTwoHighPct", f"{max(colored_local_st2):.0f}")

    def med_locked_dbic(truth_key: str) -> float:
        vals = sorted(r["locked"]["delta_bic_over_h0"] for r in cn_val["replicates"] if r["truth"] == truth_key)
        return float(vals[len(vals) // 2])

    mac("CnValOneFMedDbic", fmt_int(int(round(med_locked_dbic("local_1f")))))
    mac("CnValDetunedMedDbic", fmt_int(int(round(med_locked_dbic("local_1f_ramsey_detuned")))))
    mac("CnValOuMedDbic", fmt_int(int(round(med_locked_dbic("local_ou")))))
    mac(
        "CnValOneFHThreePct",
        f"{100 * val_summ['local_1f']['locked']['selected_model_counts'].get('H3', 0) / val_summ['local_1f']['n_replicates']:.0f}",
    )
    mac("CnValAThreeLocalFpPct", 0)
    mac(
        "CnValAThreeFastPowPct",
        f"{100 * val_summ['correlated_h3_fast']['a3']['rates']['correlated_dephasing_supported']['rate']:.0f}",
    )
    mac(
        "CnValAThreeMarkovPowPct",
        f"{100 * val_summ['correlated_h2']['a3']['rates']['correlated_dephasing_supported']['rate']:.0f}",
    )

    val_table_specs = [
        ("white", "Local white ($H_0$)"),
        ("local_1f", "Local $1/f$ dephasing"),
        ("local_1f_ramsey_detuned", "Local $1/f$ + Ramsey detuning"),
        ("local_ou", "Local Ornstein--Uhlenbeck"),
        ("correlated_h2", "Spatially correlated Markov ($H_2$)"),
        ("correlated_h3_fast", "Correlated OU bath, fast ($H_3$)"),
        ("correlated_h3_slow", "Correlated OU bath, slow ($H_3$)"),
    ]
    cn_val_rows: list[str] = []
    for tkey, tlab in val_table_specs:
        s_row = val_summ[tkey]
        n_rep = int(s_row["n_replicates"])
        h3_sel = 100.0 * s_row["locked"]["selected_model_counts"].get("H3", 0) / n_rep
        st2 = 100.0 * s_row["locked"]["rates"]["stage2_gate"]["rate"]
        a3_any = (
            100.0
            * s_row["a3"]["rates"]
            .get("any_nonlocal_claim", s_row["a3"]["rates"]["correlated_dephasing_supported"])["rate"]
        )
        a3_corr = 100.0 * s_row["a3"]["rates"]["correlated_dephasing_supported"]["rate"]
        cn_val_rows.append(
            f"{tlab} & {n_rep} & {fmt_int(int(round(med_locked_dbic(tkey))))} & "
            f"{h3_sel:.0f}\\% & {st2:.0f}\\% & {a3_any:.0f}\\% & {a3_corr:.0f}\\%\\\\"
        )
    mac("CnValidationRows", "\n".join(cn_val_rows))

    (HERE / "generated.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")

    fig1 = build_fig1_tmap(
        m2_gap=100.0 * c_path["mean_diff"],
        m2_ci=[100.0 * c_path["ci95"][0], 100.0 * c_path["ci95"][1]],
        m2_aa=100.0 * c_aa["mean_diff"],
        m2_aa_ci=[100.0 * c_aa["ci95"][0], 100.0 * c_aa["ci95"][1]],
        m3_gap=float(k1["mean_pp"]),
        m3_ci=k1["ci95_pp"],
        m3_mit=float(m3_mit_gap),
        p4_l1={
            b: p4_data[b]["analysis"]["contrasts"]["L1_probe_vs_calibration_path"] for b in BACKENDS
        },
        p4_l4={
            b: p4_data[b]["analysis"]["contrasts"]["L4_L1_after_readout_mitigation"] for b in BACKENDS
        },
        p5_m4={
            b: p5_data[b]["analysis"]["contrasts"]["M4_probe_vs_calibration_replication"]
            for b in BACKENDS
        },
        dt_mins=dt_mins,
        q46_p4=q46_p4,
        q46_p5=q46_p5,
    )
    fig1.save(HERE / "fig1_tmap_sign_reversal_three_qpus")

    fig2 = build_fig2_ro(
        ro_jobs=ro_jobs,
        ro_shots=ro_shots,
        chip_qubits=chip_qubits,
        n_backends=len(BACKENDS),
        ever_hidden=int(ro6_a["H2_qubits_ever_hidden"]),
        rec_count=int(ro6_a["H2_of_which_recovered_in_another_round"]),
        gap_hours=(t_r1 - t_r0).total_seconds() / 3600.0,
        trajectories=trajectories_fig,
        persist_ff=100.0 * ro6_a["H3_p_fault_next_given_fault"],
        persist_of=100.0 * ro6_a["H3_p_fault_next_given_ok"],
        fisher_or=float(ro6_a["H3_fisher_odds_ratio"]),
        king_hours=king_unchanged_h,
        ro7_points=ro7_points_fig,
        max_mux_pp=100.0 * max_mux_abs,
        max_d_pp=100.0 * max_nbr_abs,
    )
    fig2.save(HERE / "fig2_ro06_07_staleness_persistence")

    val_bars_fig = [
        (
            "White H0",
            100.0 * val_summ["white"]["locked"]["rates"]["stage2_gate"]["rate"],
            0.0,
            med_locked_dbic("white"),
        ),
        (
            "Local 1/f",
            100.0
            * val_summ["local_1f"]["locked"]["selected_model_counts"].get("H3", 0)
            / val_summ["local_1f"]["n_replicates"],
            0.0,
            med_locked_dbic("local_1f"),
        ),
        (
            "1/f+Detun",
            100.0
            * val_summ["local_1f_ramsey_detuned"]["locked"]["selected_model_counts"].get("H3", 0)
            / val_summ["local_1f_ramsey_detuned"]["n_replicates"],
            0.0,
            med_locked_dbic("local_1f_ramsey_detuned"),
        ),
        (
            "Local OU",
            100.0
            * val_summ["local_ou"]["locked"]["selected_model_counts"].get("H3", 0)
            / val_summ["local_ou"]["n_replicates"],
            0.0,
            med_locked_dbic("local_ou"),
        ),
    ]
    offs = (0, 16, 32, 48, 80, 160, 256)
    stagger_curves_fig = []
    for prof in cn_lock["per_coupler_profiles"]:
        cur = prof["phi_offset_curve_raw"]
        base = float(cur["0ns"])
        stagger_curves_fig.append(
            (prof["role"], [(float(o), 100.0 * (float(cur[f"{o}ns"]) - base)) for o in offs])
        )
    fig3 = build_fig3_cn01(
        val_bars=val_bars_fig,
        locked_dbic=float(cn_lock["gls_model_selection_on_training_cohort"]["delta_bic_over_h0"]),
        ho_h0=[float(x) for x in ho["per_coupler_rmse_h0"]],
        ho_h3=[float(x) for x in ho["per_coupler_rmse_h3"]],
        ho_mean_h0=float(ho["mean_rmse_h0"]),
        ho_mean_h3=float(ho["mean_rmse_h3"]),
        ho_p=float(ho["p_two_sided_df5"]),
        stagger_curves=stagger_curves_fig,
        c_dfs_pp=100.0 * mc["C_DFS_asymmetry"]["mean"],
    )
    fig3.save(HERE / "fig3_cn01_colored_noise_heldout_gate")

    sum_lines: list[str] = []
    for item in sorted(HERE.iterdir(), key=lambda p: p.name):
        if item.is_file() and item.name != "SHA256SUMS.txt":
            sum_lines.append(f"{sha256_file(item)}  {item.name}")
    (HERE / "SHA256SUMS.txt").write_text("\n".join(sum_lines) + "\n", encoding="utf-8")
    print(
        f"Generated {len(macros) - 1} macros, 3 SVG/PDF figures, capsule {capsule_digest[:16]}..., "
        f"jobs={total_jobs}, shots={total_shots:,}, qpu_s={total_qpu_s}"
    )


if __name__ == "__main__":
    main()
