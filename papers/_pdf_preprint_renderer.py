#!/usr/bin/env python3
"""Deterministic pure-Python multi-page PDF 1.4 academic preprint typesetter.

Provides:
  - `VectorCanvas`: Dual SVG + PDF 1.4 vector graphics builder (top-left origin in points).
  - `build_pdf_document`: Deterministic PDF 1.4 assembler with standard Type1 fonts and valid xref.
  - `PreprintDocument`: Multi-page US-Letter (612 x 792 pt) academic preprint layout engine
    supporting title/provenance banners, wrapped paragraphs with bold lead-ins, bold section
    and subsection headings, displayed equations, formatted multi-page data tables, highlighted
    summary callout boxes, embedded vector figures, and running page headers/footers.

Zero external plotting or TeX dependencies; zero timestamps or non-deterministic state.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any, List, Sequence, Tuple

_UNICODE_ASCII_MAP: dict[str, str] = {
    "±": "+/-",
    "∓": "-/+",
    "≥": ">=",
    "≤": "<=",
    "≫": ">>",
    "≪": "<<",
    "×": "x",
    "÷": "/",
    "—": "--",
    "–": "-",
    "−": "-",
    "→": "->",
    "←": "<-",
    "↔": "<->",
    "↦": "|->",
    "⇒": "=>",
    "≈": "~",
    "≃": "~",
    "≠": "!=",
    "≡": "==",
    "∈": "in",
    "∉": "not in",
    "⊂": "subset",
    "⊆": "subseteq",
    "⊕": "(+)",
    "⊗": "(x)",
    "⋉": "|x",
    "·": "*",
    "•": "*",
    "…": "...",
    "°": " deg",
    "′": "'",
    "″": "''",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "⟨": "<",
    "⟩": ">",
    "†": "+",
    "⋆": "*",
    "∞": "inf",
    "√": "sqrt",
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "eps",
    "ϵ": "eps",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "ϑ": "vartheta",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "u",
    "µ": "u",
    "ν": "nu",
    "ξ": "xi",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "τ": "tau",
    "υ": "upsilon",
    "φ": "phi",
    "ϕ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
    "Γ": "Gamma",
    "Δ": "Delta",
    "Θ": "Theta",
    "Λ": "Lambda",
    "Ξ": "Xi",
    "Π": "Pi",
    "Σ": "Sigma",
    "Υ": "Upsilon",
    "Φ": "Phi",
    "Ψ": "Psi",
    "Ω": "Omega",
    "Ł": "L",
    "ł": "l",
}


def pdf_sanitize(text: str) -> str:
    """Normalize arbitrary scientific Unicode text to clean printable 7-bit ASCII."""
    out: list[str] = []
    for ch in text:
        if ch in _UNICODE_ASCII_MAP:
            out.append(_UNICODE_ASCII_MAP[ch])
        elif 32 <= ord(ch) <= 126:
            out.append(ch)
        elif ch in ("\n", "\t"):
            out.append(" ")
        else:
            norm = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode("ascii")
            out.append(norm if norm else "?")
    return "".join(out)


def pdf_escape(text: str) -> str:
    """Sanitize and escape a string for a PDF literal string `(...) Tj`."""
    clean = pdf_sanitize(text)
    return clean.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _hex_rgb(hex_color: str) -> Tuple[float, float, float]:
    c = hex_color.lstrip("#")
    return int(c[0:2], 16) / 255.0, int(c[2:4], 16) / 255.0, int(c[4:6], 16) / 255.0


def estimate_text_width(text: str, font: str, size: float) -> float:
    """Estimate rendered width in points for PDF Type1 standard fonts `/F1`..`/F7`."""
    clean = pdf_sanitize(text)
    if font in ("/F3", "/F7"):
        return 0.600 * size * len(clean)

    is_sans = font in ("/F1", "/F2")
    is_bold = font in ("/F2", "/F5")
    total_em = 0.0
    for ch in clean:
        if ch in " il.,:;!|'`":
            w = 0.278
        elif ch in 'frt()[]{}-/\\"':
            w = 0.345
        elif ch in "0123456789":
            w = 0.556 if is_sans else 0.505
        elif ch in "mw":
            w = 0.778
        elif ch in "abcdeghjknopqsuvxyz":
            w = 0.525 if is_sans else 0.485
        elif ch in "IJ":
            w = 0.380
        elif ch in "MW":
            w = 0.890
        elif "A" <= ch <= "Z":
            w = 0.685
        else:
            w = 0.580
        total_em += w
    if is_bold:
        total_em *= 1.055
    return total_em * size


def wrap_text_to_width(text: str, font: str, size: float, max_width: float) -> List[str]:
    """Deterministically wrap plain text so every line fits within `max_width` points."""
    clean = " ".join(pdf_sanitize(text).split())
    if not clean:
        return [""]
    eff_max = max(24.0, max_width * 0.985)
    words = clean.split(" ")
    lines: List[str] = []
    cur = ""

    def _split_long_token(tok: str) -> List[str]:
        parts: List[str] = []
        buf = ""
        for ch in tok:
            cand = buf + ch
            if buf and estimate_text_width(cand, font, size) > eff_max:
                parts.append(buf)
                buf = ch
            else:
                buf = cand
        if buf:
            parts.append(buf)
        return parts

    for word in words:
        if estimate_text_width(word, font, size) > eff_max:
            if cur:
                lines.append(cur)
                cur = ""
            pieces = _split_long_token(word)
            lines.extend(pieces[:-1])
            cur = pieces[-1]
            continue
        cand = word if not cur else f"{cur} {word}"
        if not cur or estimate_text_width(cand, font, size) <= eff_max:
            cur = cand
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


class VectorCanvas:
    """Deterministic dual SVG + PDF 1.4 vector graphics builder (top-left origin in points)."""

    def __init__(self, width: float, height: float) -> None:
        self.w = float(width)
        self.h = float(height)
        self.svg: List[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w:.0f}" '
            f'height="{self.h:.0f}" viewBox="0 0 {self.w:.0f} {self.h:.0f}">',
            f'<rect x="0" y="0" width="{self.w:.0f}" height="{self.h:.0f}" fill="#ffffff"/>',
        ]
        self.pdf: List[str] = ["1 1 1 rg", f"0 0 {self.w:.2f} {self.h:.2f} re f"]

    def _py(self, y: float) -> float:
        return self.h - y

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
            fr, fg, fb = _hex_rgb(fill)
            sr, sg, sb = _hex_rgb(stroke)
            self.pdf.append(
                f"{lw:.2f} w {fr:.3f} {fg:.3f} {fb:.3f} rg "
                f"{sr:.3f} {sg:.3f} {sb:.3f} RG {x:.2f} {py:.2f} {w:.2f} {h:.2f} re B"
            )
        elif fill:
            fr, fg, fb = _hex_rgb(fill)
            self.pdf.append(f"{fr:.3f} {fg:.3f} {fb:.3f} rg {x:.2f} {py:.2f} {w:.2f} {h:.2f} re f")
        elif stroke:
            sr, sg, sb = _hex_rgb(stroke)
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
        r, g, b = _hex_rgb(color)
        dash_pdf = "[5 3] 0 d" if dashed else "[] 0 d"
        self.pdf.append(
            f"q {dash_pdf} {lw:.2f} w {r:.3f} {g:.3f} {b:.3f} RG "
            f"{x1:.2f} {self._py(y1):.2f} m {x2:.2f} {self._py(y2):.2f} l S Q"
        )

    def polyline(
        self,
        points: Sequence[Tuple[float, float]],
        color: str = "#111827",
        lw: float = 1.6,
        dashed: bool = False,
    ) -> None:
        if len(points) < 2:
            return
        pts_svg = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        dash_svg = ' stroke-dasharray="5,3"' if dashed else ""
        self.svg.append(
            f'<polyline points="{pts_svg}" fill="none" stroke="{color}" '
            f'stroke-width="{lw:.2f}"{dash_svg}/>'
        )
        r, g, b = _hex_rgb(color)
        dash_pdf = "[5 3] 0 d" if dashed else "[] 0 d"
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
        fr, fg, fb = _hex_rgb(fill)
        sr, sg, sb = _hex_rgb(stroke)
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
        font_tag = "/F7" if (mono and bold) else ("/F3" if mono else ("/F2" if bold else "/F1"))
        approx_w = estimate_text_width(s, font_tag, size)
        px = (
            x
            if anchor == "start"
            else (x - 0.5 * approx_w if anchor == "middle" else x - approx_w)
        )
        r, g, b = _hex_rgb(color)
        esc_pdf = pdf_escape(s)
        self.pdf.append(
            f"BT {font_tag} {size:.1f} Tf {r:.3f} {g:.3f} {b:.3f} rg "
            f"{px:.2f} {self._py(y):.2f} Td ({esc_pdf}) Tj ET"
        )

    def to_svg(self) -> str:
        return "\n".join(self.svg + ["</svg>"]) + "\n"

    def pdf_stream(self) -> bytes:
        return ("\n".join(self.pdf) + "\n").encode("ascii")

    def save(self, stem_path: Path) -> None:
        stem_path.with_suffix(".svg").write_bytes(self.to_svg().encode("utf-8"))
        stem_path.with_suffix(".pdf").write_bytes(
            build_pdf_document([(self.w, self.h, self.pdf_stream())])
        )


def build_pdf_document(
    pages: Sequence[Tuple[float, float, bytes]],
    *,
    title: str | None = None,
    author: str | None = None,
) -> bytes:
    """Assemble a valid deterministic multi-page PDF 1.4 document with standard Type1 fonts."""
    if not pages:
        raise ValueError("PDF document must contain at least one page")

    objects: List[bytes] = []

    def add_obj(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    f1 = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    f2 = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    f3 = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")
    f4 = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman >>")
    f5 = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Bold >>")
    f6 = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Italic >>")
    f7 = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier-Bold >>")

    pages_placeholder_idx = add_obj(b"")
    page_ids: List[int] = []

    font_res = (
        f"/F1 {f1} 0 R /F2 {f2} 0 R /F3 {f3} 0 R /F4 {f4} 0 R "
        f"/F5 {f5} 0 R /F6 {f6} 0 R /F7 {f7} 0 R"
    )
    for w, h, stream in pages:
        c_id = add_obj(
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"endstream"
        )
        p_id = add_obj(
            f"<< /Type /Page /Parent {pages_placeholder_idx} 0 R /MediaBox [0 0 {w:.2f} {h:.2f}] "
            f"/Resources << /Font << {font_res} >> >> "
            f"/Contents {c_id} 0 R >>".encode("ascii")
        )
        page_ids.append(p_id)

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects[pages_placeholder_idx - 1] = (
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii")
    )
    catalog_id = add_obj(f"<< /Type /Catalog /Pages {pages_placeholder_idx} 0 R >>".encode("ascii"))

    info_id: int | None = None
    if title or author:
        info_parts = ["/Producer (OSIRIS Deterministic PDF 1.4 Preprint Renderer)"]
        if title:
            info_parts.append(f"/Title ({pdf_escape(title)})")
        if author:
            info_parts.append(f"/Author ({pdf_escape(author)})")
        info_id = add_obj(f"<< {' '.join(info_parts)} >>".encode("ascii"))

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

    info_ref = f" /Info {info_id} 0 R" if info_id is not None else ""
    buf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R{info_ref} >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n".encode("ascii")
    )
    return bytes(buf)


class PreprintDocument:
    """Multi-page US-Letter (612 x 792 pt) academic preprint PDF 1.4 typesetter."""

    def __init__(
        self,
        *,
        short_title: str,
        manuscript_id: str,
        execution_tier: str,
        footer_left: str,
        author: str = "Devin Phillip Davis",
        page_width: float = 612.0,
        page_height: float = 792.0,
        margin_x: float = 48.0,
    ) -> None:
        self.short_title = pdf_sanitize(short_title)
        self.manuscript_id = pdf_sanitize(manuscript_id)
        self.execution_tier = pdf_sanitize(execution_tier)
        self.footer_left = pdf_sanitize(footer_left)
        self.author = pdf_sanitize(author)
        self.full_title = self.short_title

        self.w = float(page_width)
        self.h = float(page_height)
        self.margin_x = float(margin_x)
        self.content_w = self.w - 2.0 * self.margin_x
        self.top_y = 54.0
        self.bottom_y = 740.0

        self._pages: List[List[str]] = [["1 1 1 rg", f"0 0 {self.w:.2f} {self.h:.2f} re f"]]
        self.y = 42.0
        self._fig_counter = 0

    @property
    def page_count(self) -> int:
        return len(self._pages)

    def _cur(self) -> List[str]:
        return self._pages[-1]

    def _py(self, y: float) -> float:
        return self.h - y

    def new_page(self) -> None:
        self._pages.append(["1 1 1 rg", f"0 0 {self.w:.2f} {self.h:.2f} re f"])
        self.y = self.top_y

    def ensure_space(self, needed_h: float) -> None:
        if self.y + needed_h > self.bottom_y and self.y > self.top_y + 2.0:
            self.new_page()

    def add_spacer(self, dy: float) -> None:
        self.y = min(self.bottom_y, self.y + dy)

    def _draw_rect(
        self,
        cmds: List[str],
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str | None = None,
        stroke: str | None = None,
        lw: float = 0.8,
    ) -> None:
        py = self._py(y + h)
        if fill and stroke:
            fr, fg, fb = _hex_rgb(fill)
            sr, sg, sb = _hex_rgb(stroke)
            cmds.append(
                f"q {lw:.2f} w {fr:.3f} {fg:.3f} {fb:.3f} rg {sr:.3f} {sg:.3f} {sb:.3f} RG "
                f"{x:.2f} {py:.2f} {w:.2f} {h:.2f} re B Q"
            )
        elif fill:
            fr, fg, fb = _hex_rgb(fill)
            cmds.append(f"q {fr:.3f} {fg:.3f} {fb:.3f} rg {x:.2f} {py:.2f} {w:.2f} {h:.2f} re f Q")
        elif stroke:
            sr, sg, sb = _hex_rgb(stroke)
            cmds.append(
                f"q {lw:.2f} w {sr:.3f} {sg:.3f} {sb:.3f} RG {x:.2f} {py:.2f} {w:.2f} {h:.2f} re S Q"
            )

    def _draw_line(
        self,
        cmds: List[str],
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = "#cbd5e1",
        lw: float = 0.7,
    ) -> None:
        r, g, b = _hex_rgb(color)
        cmds.append(
            f"q [] 0 d {lw:.2f} w {r:.3f} {g:.3f} {b:.3f} RG "
            f"{x1:.2f} {self._py(y1):.2f} m {x2:.2f} {self._py(y2):.2f} l S Q"
        )

    def _draw_text(
        self,
        cmds: List[str],
        x: float,
        y: float,
        s: str,
        *,
        font: str = "/F4",
        size: float = 9.5,
        color: str = "#111827",
        anchor: str = "start",
    ) -> None:
        w = estimate_text_width(s, font, size)
        px = x if anchor == "start" else (x - 0.5 * w if anchor == "middle" else x - w)
        r, g, b = _hex_rgb(color)
        esc = pdf_escape(s)
        cmds.append(
            f"BT {font} {size:.1f} Tf {r:.3f} {g:.3f} {b:.3f} rg "
            f"{px:.2f} {self._py(y):.2f} Td ({esc}) Tj ET"
        )

    def _wrap_styled_runs(
        self,
        lead_in: str | None,
        text: str,
        *,
        bold_font: str,
        body_font: str,
        size: float,
        max_width: float,
    ) -> List[List[Tuple[str, str]]]:
        """Wrap optional bold `lead_in` + regular `text` into lines of `(font, span_str)` runs."""
        tokens: List[Tuple[str, str]] = []
        if lead_in:
            for w in pdf_sanitize(lead_in).split():
                tokens.append((bold_font, w))
        for w in pdf_sanitize(text).split():
            tokens.append((body_font, w))
        if not tokens:
            return []

        eff_max = max(24.0, max_width * 0.985)
        lines: List[List[Tuple[str, str]]] = []
        cur_line: List[Tuple[str, str]] = []
        cur_w = 0.0
        space_w = estimate_text_width(" ", body_font, size)

        for fnt, tok in tokens:
            tw = estimate_text_width(tok, fnt, size)
            if not cur_line:
                cur_line.append((fnt, tok))
                cur_w = tw
            elif cur_w + space_w + tw <= eff_max:
                cur_line.append((fnt, tok))
                cur_w += space_w + tw
            else:
                lines.append(cur_line)
                cur_line = [(fnt, tok)]
                cur_w = tw
        if cur_line:
            lines.append(cur_line)

        merged_lines: List[List[Tuple[str, str]]] = []
        for ln in lines:
            runs: List[Tuple[str, str]] = []
            for fnt, tok in ln:
                if runs and runs[-1][0] == fnt:
                    runs[-1] = (fnt, f"{runs[-1][1]} {tok}")
                elif runs:
                    runs[-1] = (runs[-1][0], f"{runs[-1][1]} ")
                    runs.append((fnt, tok))
                else:
                    runs.append((fnt, tok))
            merged_lines.append(runs)
        return merged_lines

    def add_title_block(
        self,
        *,
        title: str,
        author: str,
        affiliation: str,
        date_str: str,
        metadata_rows: Sequence[Tuple[str, str]],
        abstract: str,
    ) -> None:
        self.full_title = pdf_sanitize(title)
        cmds = self._cur()

        # Top preprint banner strip
        is_hw = "HARDWARE" in self.execution_tier.upper()
        badge_bg = "#ecfdf5" if is_hw else "#eff6ff"
        badge_fg = "#065f46" if is_hw else "#1e40af"
        badge_bd = "#6ee7b7" if is_hw else "#93c5fd"
        self._draw_rect(cmds, self.margin_x, self.y - 9.0, self.content_w, 16.0, fill="#f8fafc", stroke="#e2e8f0", lw=0.6)
        self._draw_text(
            cmds,
            self.margin_x + 7.0,
            self.y + 2.0,
            f"OSIRIS ARCHIVAL PREPRINT  |  {self.manuscript_id}",
            font="/F2",
            size=7.4,
            color="#334155",
        )
        badge_txt = f"TIER: {self.execution_tier}  |  {date_str}"
        bw = estimate_text_width(badge_txt, "/F2", 7.2) + 12.0
        bx = self.margin_x + self.content_w - bw - 3.0
        self._draw_rect(cmds, bx, self.y - 7.0, bw, 12.0, fill=badge_bg, stroke=badge_bd, lw=0.6)
        self._draw_text(cmds, bx + 6.0, self.y + 1.8, badge_txt, font="/F2", size=7.2, color=badge_fg)
        self.y += 22.0

        # Title
        title_lines = wrap_text_to_width(title, "/F5", 13.4, self.content_w)
        for ln in title_lines:
            self._draw_text(cmds, self.margin_x, self.y, ln, font="/F5", size=13.4, color="#0f172a")
            self.y += 16.8
        self.y += 2.0

        # Author & Affiliation
        self._draw_text(cmds, self.margin_x, self.y, author, font="/F5", size=10.0, color="#1e293b")
        self.y += 12.5
        self._draw_text(
            cmds,
            self.margin_x,
            self.y,
            f"{affiliation}  |  {date_str}",
            font="/F6",
            size=9.0,
            color="#475569",
        )
        self.y += 10.0

        # Provenance metadata card
        if metadata_rows:
            prepared_meta: List[Tuple[str, float, List[str]]] = []
            total_meta_lines = 0
            for k_lbl, v_lbl in metadata_rows:
                k_clean = f"{k_lbl}: "
                kw = estimate_text_width(k_clean, "/F2", 7.4)
                max_v_w = self.content_w - 16.0 - kw
                v_wrapped = wrap_text_to_width(v_lbl, "/F3", 7.2, max_v_w)
                prepared_meta.append((k_clean, kw, v_wrapped))
                total_meta_lines += len(v_wrapped)
            box_h = 8.0 + total_meta_lines * 10.4
            self._draw_rect(
                cmds, self.margin_x, self.y, self.content_w, box_h, fill="#f8fafc", stroke="#cbd5e1", lw=0.7
            )
            self._draw_rect(cmds, self.margin_x, self.y, 2.8, box_h, fill="#1d4ed8")
            my = self.y + 10.0
            for k_clean, kw, v_wrapped in prepared_meta:
                self._draw_text(cmds, self.margin_x + 9.0, my, k_clean, font="/F2", size=7.4, color="#1e293b")
                for v_ln in v_wrapped:
                    self._draw_text(
                        cmds, self.margin_x + 9.0 + kw, my, v_ln, font="/F3", size=7.2, color="#334155"
                    )
                    my += 10.4
            self.y += box_h + 10.0

        # Abstract block
        abs_margin = 10.0
        abs_w = self.content_w - 2.0 * abs_margin
        abs_lines = wrap_text_to_width(abstract, "/F4", 9.0, abs_w)
        abs_h = 20.0 + len(abs_lines) * 11.8
        self._draw_rect(
            cmds, self.margin_x, self.y, self.content_w, abs_h, fill="#fcfcfd", stroke="#cbd5e1", lw=0.6
        )
        self._draw_text(
            cmds, self.margin_x + abs_margin, self.y + 11.5, "ABSTRACT", font="/F2", size=8.4, color="#0f172a"
        )
        ay = self.y + 23.0
        for ln in abs_lines:
            self._draw_text(cmds, self.margin_x + abs_margin, ay, ln, font="/F4", size=9.0, color="#1f2937")
            ay += 11.8
        self.y += abs_h + 12.0

    def add_section(self, heading: str) -> None:
        self.ensure_space(38.0)
        if self.y > self.top_y + 4.0:
            self.y += 5.0
        cmds = self._cur()
        lines = wrap_text_to_width(heading, "/F2", 10.4, self.content_w)
        for ln in lines:
            self._draw_text(cmds, self.margin_x, self.y, ln, font="/F2", size=10.4, color="#0f172a")
            self.y += 13.2
        rule_y = self.y - 8.2
        self._draw_line(cmds, self.margin_x, rule_y, self.margin_x + self.content_w, rule_y, color="#cbd5e1", lw=0.7)
        self._draw_line(cmds, self.margin_x, rule_y, self.margin_x + 58.0, rule_y, color="#1d4ed8", lw=1.4)
        self.y += 2.5

    def add_subsection(self, heading: str) -> None:
        self.ensure_space(28.0)
        if self.y > self.top_y + 2.0:
            self.y += 2.5
        lines = wrap_text_to_width(heading, "/F5", 9.7, self.content_w)
        for ln in lines:
            self._draw_text(self._cur(), self.margin_x, self.y, ln, font="/F5", size=9.7, color="#1e293b")
            self.y += 12.4
        self.y += 1.0

    def add_paragraph(
        self,
        text: str,
        *,
        lead_in: str | None = None,
        font: str = "/F4",
        size: float = 9.4,
        leading: float = 12.5,
        indent: float = 0.0,
        space_after: float = 5.0,
        color: str = "#111827",
    ) -> None:
        bold_font = "/F5" if font in ("/F4", "/F5", "/F6") else "/F2"
        lines = self._wrap_styled_runs(
            lead_in, text, bold_font=bold_font, body_font=font, size=size, max_width=self.content_w - indent
        )
        for runs in lines:
            if self.y + leading > self.bottom_y:
                self.new_page()
            cx = self.margin_x + indent
            cmds = self._cur()
            for fnt, span in runs:
                self._draw_text(cmds, cx, self.y, span, font=fnt, size=size, color=color)
                cx += estimate_text_width(span, fnt, size)
            self.y += leading
        self.y = min(self.bottom_y, self.y + space_after)

    def add_bullets(
        self,
        items: Sequence[Tuple[str | None, str] | str],
        *,
        size: float = 9.2,
        leading: float = 12.2,
    ) -> None:
        for item in items:
            if isinstance(item, tuple):
                lead, body = item
            else:
                lead, body = None, item
            runs_lines = self._wrap_styled_runs(
                lead, body, bold_font="/F5", body_font="/F4", size=size, max_width=self.content_w - 15.0
            )
            for idx, runs in enumerate(runs_lines):
                if self.y + leading > self.bottom_y:
                    self.new_page()
                cmds = self._cur()
                if idx == 0:
                    self._draw_rect(cmds, self.margin_x + 5.0, self.y - 4.8, 3.2, 3.2, fill="#1d4ed8")
                cx = self.margin_x + 15.0
                for fnt, span in runs:
                    self._draw_text(cmds, cx, self.y, span, font=fnt, size=size, color="#111827")
                    cx += estimate_text_width(span, fnt, size)
                self.y += leading
            self.y = min(self.bottom_y, self.y + 2.5)
        self.y = min(self.bottom_y, self.y + 2.5)

    def add_equation(self, eq_lines: Sequence[str], *, eq_num: str | None = None) -> None:
        box_h = 8.0 + len(eq_lines) * 12.2
        self.ensure_space(box_h + 6.0)
        cmds = self._cur()
        self._draw_rect(
            cmds, self.margin_x + 14.0, self.y - 3.0, self.content_w - 28.0, box_h, fill="#f8fafc", stroke="#e2e8f0", lw=0.5
        )
        ey = self.y + 7.5
        for idx, eq in enumerate(eq_lines):
            self._draw_text(
                cmds,
                self.margin_x + 0.5 * self.content_w,
                ey,
                eq,
                font="/F3",
                size=8.2,
                color="#0f172a",
                anchor="middle",
            )
            if eq_num and idx == len(eq_lines) - 1:
                self._draw_text(
                    cmds,
                    self.margin_x + self.content_w - 20.0,
                    ey,
                    f"({eq_num})",
                    font="/F4",
                    size=8.6,
                    color="#475569",
                    anchor="end",
                )
            ey += 12.2
        self.y += box_h + 6.0

    def add_callout_box(
        self,
        title: str,
        items: Sequence[str],
        *,
        tone: str = "info",
    ) -> None:
        palette = {
            "info": ("#eff6ff", "#bfdbfe", "#1d4ed8", "#1e3a8a"),
            "success": ("#ecfdf5", "#a7f3d0", "#047857", "#064e3b"),
            "warn": ("#fffbeb", "#fde68a", "#b45309", "#78350f"),
        }
        bg, bd, bar, title_col = palette.get(tone, palette["info"])

        inner_w = self.content_w - 22.0
        title_lines = wrap_text_to_width(title, "/F2", 8.8, inner_w)
        wrapped_items: List[List[str]] = [wrap_text_to_width(it, "/F4", 8.8, inner_w - 9.0) for it in items]
        total_item_lines = sum(len(w) for w in wrapped_items)
        box_h = 11.0 + len(title_lines) * 11.5 + total_item_lines * 11.4 + max(0, len(items) - 1) * 2.0

        self.ensure_space(box_h + 8.0)
        cmds = self._cur()
        self._draw_rect(cmds, self.margin_x, self.y, self.content_w, box_h, fill=bg, stroke=bd, lw=0.8)
        self._draw_rect(cmds, self.margin_x, self.y, 3.2, box_h, fill=bar)

        cy = self.y + 10.8
        for tln in title_lines:
            self._draw_text(cmds, self.margin_x + 10.0, cy, tln, font="/F2", size=8.8, color=title_col)
            cy += 11.5
        cy += 1.0
        for item_lines in wrapped_items:
            for j, iln in enumerate(item_lines):
                if j == 0:
                    self._draw_rect(cmds, self.margin_x + 11.0, cy - 4.4, 2.8, 2.8, fill=bar)
                self._draw_text(cmds, self.margin_x + 19.0, cy, iln, font="/F4", size=8.8, color="#1f2937")
                cy += 11.4
            cy += 2.0
        self.y += box_h + 8.0

    def add_table(
        self,
        caption: str,
        headers: Sequence[str],
        rows: Sequence[Sequence[str]],
        *,
        col_widths: Sequence[float] | None = None,
        alignments: Sequence[str] | None = None,
        font_size: float = 7.2,
        mono_cols: Sequence[bool] | None = None,
    ) -> None:
        ncols = len(headers)
        if ncols == 0:
            return
        aligns = list(alignments) if alignments else ["left"] + ["right"] * (ncols - 1)
        monos = list(mono_cols) if mono_cols else [False] + [True] * (ncols - 1)

        if col_widths is not None:
            tot = sum(col_widths)
            widths = [self.content_w * (w / tot) for w in col_widths]
        else:
            raw_w: List[float] = []
            for j in range(ncols):
                f_cell = "/F3" if monos[j] else "/F1"
                max_cell = estimate_text_width(headers[j], "/F2", font_size)
                for r in rows:
                    if j < len(r):
                        max_cell = max(max_cell, estimate_text_width(r[j], f_cell, font_size))
                raw_w.append(min(220.0, max(34.0, max_cell + 10.0)))
            tot = sum(raw_w)
            widths = [self.content_w * (w / tot) for w in raw_w]

        cap_lines = wrap_text_to_width(caption, "/F5", 8.4, self.content_w)
        hdr_wrapped = [wrap_text_to_width(headers[j], "/F2", font_size, widths[j] - 6.0) for j in range(ncols)]
        hdr_lines = max(len(hw) for hw in hdr_wrapped)
        row_lead = font_size + 2.6
        hdr_h = hdr_lines * row_lead + 5.0

        # Ensure caption + header + at least 2 rows fit together
        self.ensure_space(len(cap_lines) * 11.0 + hdr_h + 28.0)

        for cln in cap_lines:
            self._draw_text(self._cur(), self.margin_x, self.y, cln, font="/F5", size=8.4, color="#0f172a")
            self.y += 11.0
        self.y += 1.0

        def _render_header() -> None:
            cmds = self._cur()
            self._draw_rect(cmds, self.margin_x, self.y, self.content_w, hdr_h, fill="#e2e8f0")
            self._draw_line(cmds, self.margin_x, self.y, self.margin_x + self.content_w, self.y, color="#0f172a", lw=1.0)
            cx = self.margin_x
            for j in range(ncols):
                al = aligns[j]
                tx = (
                    cx + 3.0
                    if al == "left"
                    else (cx + 0.5 * widths[j] if al == "center" else cx + widths[j] - 3.0)
                )
                anch = "start" if al == "left" else ("middle" if al == "center" else "end")
                for k, hln in enumerate(hdr_wrapped[j]):
                    self._draw_text(
                        cmds,
                        tx,
                        self.y + font_size + 1.8 + k * row_lead,
                        hln,
                        font="/F2",
                        size=font_size,
                        color="#0f172a",
                        anchor=anch,
                    )
                cx += widths[j]
            self.y += hdr_h
            self._draw_line(cmds, self.margin_x, self.y, self.margin_x + self.content_w, self.y, color="#334155", lw=0.8)

        _render_header()

        for r_idx, row in enumerate(rows):
            cell_wrapped: List[List[str]] = []
            for j in range(ncols):
                val = row[j] if j < len(row) else ""
                f_cell = "/F3" if monos[j] else "/F1"
                cell_wrapped.append(wrap_text_to_width(val, f_cell, font_size, widths[j] - 6.0))
            n_sub = max(len(cw) for cw in cell_wrapped)
            rh = n_sub * row_lead + 4.2

            if self.y + rh > self.bottom_y:
                self._draw_line(
                    self._cur(), self.margin_x, self.y, self.margin_x + self.content_w, self.y, color="#0f172a", lw=0.9
                )
                self.new_page()
                self._draw_text(
                    self._cur(),
                    self.margin_x,
                    self.y,
                    f"{cap_lines[0]} (continued)",
                    font="/F6",
                    size=8.2,
                    color="#475569",
                )
                self.y += 10.5
                _render_header()

            cmds = self._cur()
            if r_idx % 2 == 1:
                self._draw_rect(cmds, self.margin_x, self.y, self.content_w, rh, fill="#f8fafc")
            cx = self.margin_x
            for j in range(ncols):
                al = aligns[j]
                f_cell = "/F3" if monos[j] else "/F1"
                tx = (
                    cx + 3.0
                    if al == "left"
                    else (cx + 0.5 * widths[j] if al == "center" else cx + widths[j] - 3.0)
                )
                anch = "start" if al == "left" else ("middle" if al == "center" else "end")
                for k, sln in enumerate(cell_wrapped[j]):
                    self._draw_text(
                        cmds,
                        tx,
                        self.y + font_size + 1.4 + k * row_lead,
                        sln,
                        font=f_cell,
                        size=font_size,
                        color="#1e293b",
                        anchor=anch,
                    )
                cx += widths[j]
            self.y += rh
            self._draw_line(cmds, self.margin_x, self.y, self.margin_x + self.content_w, self.y, color="#e2e8f0", lw=0.4)

        self._draw_line(
            self._cur(), self.margin_x, self.y, self.margin_x + self.content_w, self.y, color="#0f172a", lw=1.0
        )
        self.y = min(self.bottom_y, self.y + 8.5)

    def add_figure(self, canvas: Any, caption: str, *, max_height: float = 278.0) -> None:
        self._fig_counter += 1
        scale = min(self.content_w / float(canvas.w), max_height / float(canvas.h), 1.0)
        fig_w = float(canvas.w) * scale
        fig_h = float(canvas.h) * scale

        lead = f"Figure {self._fig_counter}."
        cap_runs = self._wrap_styled_runs(
            lead, caption, bold_font="/F5", body_font="/F6", size=8.5, max_width=self.content_w
        )
        cap_h = len(cap_runs) * 11.2

        self.ensure_space(fig_h + cap_h + 14.0)
        cmds = self._cur()
        x_left = self.margin_x + 0.5 * (self.content_w - fig_w)
        ty = self.h - (self.y + fig_h)

        cmds.append(f"q {scale:.4f} 0 0 {scale:.4f} {x_left:.2f} {ty:.2f} cm")
        cmds.extend(canvas.pdf)
        cmds.append("Q")
        self._draw_rect(cmds, x_left, self.y, fig_w, fig_h, stroke="#cbd5e1", lw=0.6)

        self.y += fig_h + 10.0
        for runs in cap_runs:
            cx = self.margin_x
            for fnt, span in runs:
                self._draw_text(cmds, cx, self.y, span, font=fnt, size=8.5, color="#334155")
                cx += estimate_text_width(span, fnt, 8.5)
            self.y += 11.2
        self.y = min(self.bottom_y, self.y + 6.0)

    def add_references(self, refs: Sequence[str]) -> None:
        self.add_section("References")
        for ref in refs:
            lines = wrap_text_to_width(ref, "/F4", 8.2, self.content_w - 14.0)
            for idx, ln in enumerate(lines):
                if self.y + 10.6 > self.bottom_y:
                    self.new_page()
                x = self.margin_x if idx == 0 else self.margin_x + 14.0
                self._draw_text(self._cur(), x, self.y, ln, font="/F4", size=8.2, color="#334155")
                self.y += 10.6
            self.y = min(self.bottom_y, self.y + 1.8)

    def to_pdf_bytes(self) -> bytes:
        total = len(self._pages)
        finalized_pages: List[Tuple[float, float, bytes]] = []
        for idx, page_cmds in enumerate(self._pages, start=1):
            cmds = list(page_cmds)
            if idx > 1:
                self._draw_text(
                    cmds, self.margin_x, 30.0, self.short_title, font="/F6", size=8.0, color="#64748b"
                )
                self._draw_text(
                    cmds,
                    self.margin_x + self.content_w,
                    30.0,
                    f"{self.author}  |  {self.manuscript_id}",
                    font="/F1",
                    size=7.5,
                    color="#64748b",
                    anchor="end",
                )
                self._draw_line(
                    cmds, self.margin_x, 36.0, self.margin_x + self.content_w, 36.0, color="#cbd5e1", lw=0.6
                )

            self._draw_line(
                cmds, self.margin_x, 750.0, self.margin_x + self.content_w, 750.0, color="#cbd5e1", lw=0.6
            )
            self._draw_text(
                cmds, self.margin_x, 763.0, self.footer_left, font="/F3", size=7.0, color="#64748b"
            )
            self._draw_text(
                cmds,
                self.margin_x + self.content_w,
                763.0,
                f"Page {idx} of {total}",
                font="/F2",
                size=7.6,
                color="#334155",
                anchor="end",
            )
            stream_bytes = ("\n".join(cmds) + "\n").encode("ascii")
            finalized_pages.append((self.w, self.h, stream_bytes))

        return build_pdf_document(finalized_pages, title=self.full_title, author=self.author)
