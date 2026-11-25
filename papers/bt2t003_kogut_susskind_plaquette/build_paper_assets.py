#!/usr/bin/env python3
"""Deterministic asset, macro, LaTeX table, dual PDF-1.4/PNG figure, standalone
multi-page PDF 1.4 preprint, and SHA256SUMS generator for
`papers/bt2t003_kogut_susskind_plaquette/`.

Enforces `SOURCE_DATE_EPOCH=1700000000` (`CreationDate=2023-11-14T22:13:20Z`) and
`PYTHONHASHSEED=0`. Binds two deterministic upstream manifests:
  1. `experiments/bt2t_plaquette/manifest_bt2t003.json` (`OSIRIS-BT2T-003`, `SIMULATED`)
  2. `experiments/bt2t003_fez/physical_bt2t003_fez_results.json` (`OSIRIS-BT2T-003-FEZ`,
     `HARDWARE_ATTESTED`, `ibm_fez` job `dc38nreq4uuc73a6g0k0` + Window-1 job `dc35e35s2omc738cvcug`)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import sys
import zlib
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault("SOURCE_DATE_EPOCH", "1700000000")
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from papers._pdf_preprint_renderer import (  # noqa: E402
    PreprintDocument,
    VectorCanvas,
    build_pdf_document,
    pdf_sanitize,
)

CREATION_DATE_ISO = "2023-11-14T22:13:20Z"

PLAQ_DIR = ROOT / "experiments" / "bt2t_plaquette"
FEZ_DIR = ROOT / "experiments" / "bt2t003_fez"

MANIFEST_BT2T003_PATH = PLAQ_DIR / "manifest_bt2t003.json"
FEZ_RESULTS_PATH = FEZ_DIR / "physical_bt2t003_fez_results.json"
FEZ_SUBMITTED_JOB_PATH = FEZ_DIR / "bt2t003_fez_submitted_job.json"

TABLES_DIR = HERE / "tables"
FIGURES_DIR = HERE / "figures"
PREPRINT_FILENAME = "bt2t003_kogut_susskind_plaquette_preprint.pdf"
PREPRINTS_ARCHIVE_DIR = Path("/home/osiris/osiris-assessment/work/preprints_2026-11-23")

FIG1_STEM = "fig1_magnetic_potential_and_strang_scaling"
FIG2_STEM = "fig2_fez_character_sector_trajectories"
FIG3_STEM = "fig3_two_window_echo_drift_catch"

DIGIT_WORDS: Tuple[str, ...] = (
    "Zero",
    "One",
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
)

# Compact 5x7 ASCII bitmap font for deterministic PNG rasterization
_FONT_5X7: Dict[str, Tuple[int, int, int, int, int, int, int]] = {
    " ": (0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00),
    "!": (0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x04),
    '"': (0x0A, 0x0A, 0x0A, 0x00, 0x00, 0x00, 0x00),
    "#": (0x0A, 0x1F, 0x0A, 0x0A, 0x1F, 0x0A, 0x00),
    "$": (0x04, 0x0F, 0x14, 0x0E, 0x05, 0x1E, 0x04),
    "%": (0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03),
    "&": (0x0C, 0x12, 0x14, 0x08, 0x15, 0x12, 0x0D),
    "'": (0x04, 0x04, 0x08, 0x00, 0x00, 0x00, 0x00),
    "(": (0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02),
    ")": (0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08),
    "*": (0x00, 0x04, 0x15, 0x0E, 0x15, 0x04, 0x00),
    "+": (0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00),
    ",": (0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x08),
    "-": (0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00),
    ".": (0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C),
    "/": (0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10),
    "0": (0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E),
    "1": (0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "2": (0x0E, 0x11, 0x01, 0x06, 0x08, 0x10, 0x1F),
    "3": (0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E),
    "4": (0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02),
    "5": (0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E),
    "6": (0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E),
    "7": (0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08),
    "8": (0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E),
    "9": (0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C),
    ":": (0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00),
    ";": (0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x04, 0x08),
    "<": (0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02),
    "=": (0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00),
    ">": (0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08),
    "?": (0x0E, 0x11, 0x01, 0x02, 0x04, 0x00, 0x04),
    "@": (0x0E, 0x11, 0x17, 0x15, 0x17, 0x10, 0x0E),
    "A": (0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11),
    "B": (0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E),
    "C": (0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E),
    "D": (0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C),
    "E": (0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F),
    "F": (0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10),
    "G": (0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F),
    "H": (0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11),
    "I": (0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "J": (0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C),
    "K": (0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11),
    "L": (0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F),
    "M": (0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11),
    "N": (0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11),
    "O": (0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E),
    "P": (0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10),
    "Q": (0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D),
    "R": (0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11),
    "S": (0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E),
    "T": (0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04),
    "U": (0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E),
    "V": (0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04),
    "W": (0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A),
    "X": (0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11),
    "Y": (0x11, 0x11, 0x11, 0x0A, 0x04, 0x04, 0x04),
    "Z": (0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F),
    "[": (0x0E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0E),
    "\\": (0x10, 0x08, 0x08, 0x04, 0x02, 0x02, 0x01),
    "]": (0x0E, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0E),
    "^": (0x04, 0x0A, 0x11, 0x00, 0x00, 0x00, 0x00),
    "_": (0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F),
    "`": (0x08, 0x04, 0x02, 0x00, 0x00, 0x00, 0x00),
    "a": (0x00, 0x00, 0x0E, 0x01, 0x0F, 0x11, 0x0F),
    "b": (0x10, 0x10, 0x1E, 0x11, 0x11, 0x11, 0x1E),
    "c": (0x00, 0x00, 0x0E, 0x10, 0x10, 0x11, 0x0E),
    "d": (0x01, 0x01, 0x0F, 0x11, 0x11, 0x11, 0x0F),
    "e": (0x00, 0x00, 0x0E, 0x11, 0x1F, 0x10, 0x0E),
    "f": (0x06, 0x09, 0x08, 0x1C, 0x08, 0x08, 0x08),
    "g": (0x00, 0x0F, 0x11, 0x11, 0x0F, 0x01, 0x0E),
    "h": (0x10, 0x10, 0x1E, 0x11, 0x11, 0x11, 0x11),
    "i": (0x04, 0x00, 0x0C, 0x04, 0x04, 0x04, 0x0E),
    "j": (0x02, 0x00, 0x06, 0x02, 0x02, 0x12, 0x0C),
    "k": (0x10, 0x10, 0x12, 0x14, 0x18, 0x14, 0x12),
    "l": (0x0C, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E),
    "m": (0x00, 0x00, 0x1A, 0x15, 0x15, 0x11, 0x11),
    "n": (0x00, 0x00, 0x1E, 0x11, 0x11, 0x11, 0x11),
    "o": (0x00, 0x00, 0x0E, 0x11, 0x11, 0x11, 0x0E),
    "p": (0x00, 0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10),
    "q": (0x00, 0x0F, 0x11, 0x11, 0x0F, 0x01, 0x01),
    "r": (0x00, 0x00, 0x16, 0x19, 0x10, 0x10, 0x10),
    "s": (0x00, 0x00, 0x0E, 0x10, 0x0E, 0x01, 0x1E),
    "t": (0x08, 0x08, 0x1C, 0x08, 0x08, 0x09, 0x06),
    "u": (0x00, 0x00, 0x11, 0x11, 0x11, 0x13, 0x0D),
    "v": (0x00, 0x00, 0x11, 0x11, 0x11, 0x0A, 0x04),
    "w": (0x00, 0x00, 0x11, 0x11, 0x15, 0x15, 0x0A),
    "x": (0x00, 0x00, 0x11, 0x0A, 0x04, 0x0A, 0x11),
    "y": (0x00, 0x11, 0x11, 0x11, 0x0F, 0x01, 0x0E),
    "z": (0x00, 0x00, 0x1F, 0x02, 0x04, 0x08, 0x1F),
    "{": (0x02, 0x04, 0x04, 0x08, 0x04, 0x04, 0x02),
    "|": (0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04),
    "}": (0x08, 0x04, 0x04, 0x02, 0x04, 0x04, 0x08),
    "~": (0x00, 0x00, 0x08, 0x15, 0x02, 0x00, 0x00),
}


def _hex_to_u8(hex_color: str) -> Tuple[int, int, int]:
    c = hex_color.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def _with_pdf_creation_date(pdf_bytes: bytes, creation_date: str = CREATION_DATE_ISO) -> bytes:
    """Inject deterministic `/CreationDate (2023-11-14T22:13:20Z)` into the PDF 1.4 `/Info` dictionary."""
    tag = f" /CreationDate ({creation_date}) >>\nendobj\nxref\n".encode("ascii")
    target = b" >>\nendobj\nxref\n"
    if target not in pdf_bytes:
        return pdf_bytes
    idx = pdf_bytes.rfind(target)
    delta = len(tag) - len(target)
    updated = pdf_bytes[:idx] + tag + pdf_bytes[idx + len(target) :]
    m = re.search(rb"startxref\n(\d+)\n%%EOF\n\Z", updated)
    if m is None:
        return pdf_bytes
    old_xref = int(m.group(1))
    new_tail = f"startxref\n{old_xref + delta}\n%%EOF\n".encode("ascii")
    return updated[: m.start()] + new_tail


def _encode_png_rgb(img: np.ndarray) -> bytes:
    """Encode an `(H, W, 3)` `uint8` array into a deterministic PNG with fixed Creation Time metadata."""
    height, width, _ = img.shape

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    text_meta = b"Creation Time\x00" + CREATION_DATE_ISO.encode("ascii")
    # Prepend filter byte 0 to each scanline
    filter_col = np.zeros((height, 1), dtype=np.uint8)
    raw_rows = np.hstack([filter_col, img.reshape(height, width * 3)]).tobytes()
    idat = zlib.compress(raw_rows, level=9)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"tEXt", text_meta)
        + _chunk(b"IDAT", idat)
        + _chunk(b"IEND", b"")
    )


class DualPdfPngCanvas(VectorCanvas):
    """Deterministic vector PDF-1.4 + raster PNG figure canvas (2x supersampled raster grid)."""

    def __init__(self, width: float, height: float, scale: int = 2) -> None:
        super().__init__(width, height)
        self.scale = int(scale)
        self.pw = int(round(self.w * self.scale))
        self.ph = int(round(self.h * self.scale))
        self.img = np.full((self.ph, self.pw, 3), 255, dtype=np.uint8)

    def _clip(self, x: int, y: int) -> Tuple[int, int]:
        return max(0, min(self.pw - 1, x)), max(0, min(self.ph - 1, y))

    def _raster_line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        rgb: Tuple[int, int, int],
        lw: float,
        dashed: bool = False,
    ) -> None:
        s = self.scale
        p1x, p1y = x1 * s, y1 * s
        p2x, p2y = x2 * s, y2 * s
        dist = math.hypot(p2x - p1x, p2y - p1y)
        steps = max(1, int(math.ceil(dist * 1.5)))
        rad = max(0, int(round(0.5 * lw * s)))
        for i in range(steps + 1):
            if dashed and ((i // max(1, 3 * s)) % 2 == 1):
                continue
            t = i / steps
            cx = int(round(p1x + t * (p2x - p1x)))
            cy = int(round(p1y + t * (p2y - p1y)))
            x_lo, y_lo = self._clip(cx - rad, cy - rad)
            x_hi, y_hi = self._clip(cx + rad, cy + rad)
            self.img[y_lo : y_hi + 1, x_lo : x_hi + 1] = rgb

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
        super().rect(x, y, w, h, fill=fill, stroke=stroke, lw=lw)
        s = self.scale
        x0, y0 = self._clip(int(round(x * s)), int(round(y * s)))
        x1, y1 = self._clip(int(round((x + w) * s)), int(round((y + h) * s)))
        if fill:
            self.img[y0 : y1 + 1, x0 : x1 + 1] = _hex_to_u8(fill)
        if stroke:
            rgb = _hex_to_u8(stroke)
            self._raster_line(x, y, x + w, y, rgb, lw)
            self._raster_line(x + w, y, x + w, y + h, rgb, lw)
            self._raster_line(x + w, y + h, x, y + h, rgb, lw)
            self._raster_line(x, y + h, x, y, rgb, lw)

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
        super().line(x1, y1, x2, y2, color=color, lw=lw, dashed=dashed)
        self._raster_line(x1, y1, x2, y2, _hex_to_u8(color), lw, dashed=dashed)

    def polyline(
        self,
        points: Sequence[Tuple[float, float]],
        color: str = "#111827",
        lw: float = 1.6,
        dashed: bool = False,
    ) -> None:
        super().polyline(points, color=color, lw=lw, dashed=dashed)
        rgb = _hex_to_u8(color)
        for i in range(len(points) - 1):
            self._raster_line(
                points[i][0],
                points[i][1],
                points[i + 1][0],
                points[i + 1][1],
                rgb,
                lw,
                dashed=dashed,
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
        super().circle(cx, cy, r, fill=fill, stroke=stroke, lw=lw)
        s = self.scale
        pcx, pcy, pr = cx * s, cy * s, r * s
        x0, y0 = self._clip(int(math.floor(pcx - pr - 2)), int(math.floor(pcy - pr - 2)))
        x1, y1 = self._clip(int(math.ceil(pcx + pr + 2)), int(math.ceil(pcy + pr + 2)))
        yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
        dist = np.sqrt((xx - pcx) ** 2 + (yy - pcy) ** 2)
        sub = self.img[y0 : y1 + 1, x0 : x1 + 1]
        if fill:
            sub[dist <= pr] = _hex_to_u8(fill)
        if stroke:
            half_w = max(0.7, 0.5 * lw * s)
            sub[np.abs(dist - pr) <= half_w] = _hex_to_u8(stroke)

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
        super().text(x, y, s, size=size, color=color, bold=bold, anchor=anchor, mono=mono)
        clean = pdf_sanitize(s)
        if not clean:
            return
        rgb = _hex_to_u8(color)
        px_scale = max(1, int(round((size * self.scale) / 7.2)))
        char_adv = 6 * px_scale
        total_w = len(clean) * char_adv
        base_x = int(round(x * self.scale))
        if anchor == "middle":
            base_x -= total_w // 2
        elif anchor == "end":
            base_x -= total_w
        base_y = int(round(y * self.scale)) - 6 * px_scale
        for idx, ch in enumerate(clean):
            glyph = _FONT_5X7.get(ch, _FONT_5X7.get(ch.upper(), _FONT_5X7["?"]))
            gx0 = base_x + idx * char_adv
            for row_i, row_bits in enumerate(glyph):
                if row_bits == 0:
                    continue
                py0 = base_y + row_i * px_scale
                for col_i in range(5):
                    if (row_bits >> (4 - col_i)) & 1:
                        px0 = gx0 + col_i * px_scale
                        x_lo, y_lo = self._clip(px0, py0)
                        x_hi, y_hi = self._clip(
                            px0 + px_scale - 1 + (1 if bold else 0),
                            py0 + px_scale - 1,
                        )
                        self.img[y_lo : y_hi + 1, x_lo : x_hi + 1] = rgb

    def save_pdf_and_png(self, stem_path: Path, title: str) -> None:
        pdf_raw = build_pdf_document(
            [(self.w, self.h, self.pdf_stream())],
            title=title,
            author="Devin Phillip Davis",
        )
        stem_path.with_suffix(".pdf").write_bytes(_with_pdf_creation_date(pdf_raw))
        stem_path.with_suffix(".png").write_bytes(_encode_png_rgb(self.img))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fmt_sci_tex(val: float, digits: int = 2) -> str:
    if val == 0.0:
        return "0.0"
    s = f"{val:.{digits}e}"
    mant, exp_str = s.split("e")
    return f"{mant} \\times 10^{{{int(exp_str)}}}"


def _fmt_sci_ascii(val: float, digits: int = 2) -> str:
    if val == 0.0:
        return "0.0"
    return f"{val:.{digits}e}"


def _find_proxy(
    compilations: Sequence[Mapping[str, Any]],
    *,
    circuit: str,
    target: str | None = None,
) -> Mapping[str, Any]:
    for entry in compilations:
        if entry.get("circuit") != circuit:
            continue
        if target is not None and entry.get("target") != target:
            continue
        return entry
    raise KeyError(f"Missing proxy compilation: {circuit=} {target=}")


def _competition_ranks_desc(values: Sequence[float]) -> List[int]:
    """Return 1-based descending ranks (1 = highest score)."""
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    ranks = [0] * len(values)
    for rk, idx in enumerate(order, start=1):
        ranks[idx] = rk
    return ranks


def _build_fig1(manifest: Mapping[str, Any], fez_results: Mapping[str, Any]) -> DualPdfPngCanvas:
    """Fig. 1: (a) 2T magnetic potential V_B(U) & Peter-Weyl Casimir spectrum; (b) Lie vs Strang scaling & CZ compression."""
    w, h = 576.0, 324.0
    c = DualPdfPngCanvas(w, h)

    p4 = manifest["part4_trotter_suzuki_error_scaling"]
    p5 = manifest["part5_native_circuit_compilations"]
    gcs = p5["gate_count_summary"]
    g1_series = p4["by_coupling"]["g_ym_sq_1.0"]["series"]
    lie_slope = float(p4["by_coupling"]["g_ym_sq_1.0"]["lie_loglog_slope"])
    strang_slope = float(p4["by_coupling"]["g_ym_sq_1.0"]["strang_loglog_slope"])
    fez_cz = fez_results["cz_counts_by_workload"]

    c.text(
        w / 2.0,
        15.0,
        "OSIRIS-BT2T-003 (SIMULATED / EXACT_STATEVECTOR): 2T Plaquette Spectrum, Strang O(dt^2) Scaling & CZ Compression",
        size=8.7,
        bold=True,
        anchor="middle",
    )
    c.text(
        w / 2.0,
        26.5,
        f"bt2t-enc-1 (34/34 Checks)  |  4-Link 24^4=331,776 -> 5Q 24-State Holonomy (<= 2.27e-15)  |  Strang Slope = {strang_slope:.6f}",
        size=6.8,
        color="#475569",
        anchor="middle",
    )

    # Panel (a): 7 Conjugacy Classes V_B(U) = (2 - Re chi_2D(U))/2 & 7 Peter-Weyl Casimir Irreps
    pa_x, pa_y, pa_w, pa_h = 34.0, 44.0, 244.0, 212.0
    c.text(
        pa_x,
        pa_y - 5.5,
        "(a) 7 Conjugacy Classes V_B(U) & 7 Peter-Weyl Irreps c_2(rho)",
        size=7.4,
        bold=True,
        color="#0f172a",
    )
    c.rect(pa_x, pa_y, pa_w, pa_h, fill="#fafafa", stroke="#334155", lw=0.9)

    classes_meta = [
        ("C1 {+1}", 1, 2.0, 0.0, "chi_0 (1D)", 0, 0.000, "#10b981"),
        ("C4 Hurw+", 4, 1.0, 0.5, "rho_2,0 (2D)", 2, 0.750, "#2563eb"),
        ("C5 Hurw+", 4, 1.0, 0.5, "rho_3 (3D)", 6, 1.500, "#7c3aed"),
        ("C3 {+-i,j,k}", 6, 0.0, 1.0, "rho_2,1 (2D)", 2, 1.875, "#2563eb"),
        ("C6 Hurw-", 4, -1.0, 1.5, "rho_2,2 (2D)", 2, 1.875, "#2563eb"),
        ("C7 Hurw-", 4, -1.0, 1.5, "chi_1 (1D)", 0, 2.250, "#10b981"),
        ("C2 {-1}", 1, -2.0, 2.0, "chi_2 (1D)", 0, 2.250, "#10b981"),
    ]
    row_h = (pa_h - 24.0) / 7.0
    c.rect(pa_x + 0.6, pa_y + 0.6, pa_w - 1.2, 20.0, fill="#e2e8f0")
    c.text(pa_x + 5.0, pa_y + 13.5, "Class (|C|) : V_B", size=5.9, bold=True, color="#0f172a")
    c.text(
        pa_x + 126.0, pa_y + 13.5, "Irrep (d) : c_2 [d(d-1)]", size=5.9, bold=True, color="#0f172a"
    )
    c.line(pa_x, pa_y + 20.6, pa_x + pa_w, pa_y + 20.6, color="#334155", lw=0.7)
    c.line(pa_x + 120.0, pa_y, pa_x + 120.0, pa_y + pa_h, color="#cbd5e1", lw=0.7)

    for idx, (c_lbl, c_sz, _chi, vb, ir_lbl, dd1, c2_val, col) in enumerate(classes_meta):
        ry = pa_y + 21.5 + idx * row_h
        if idx > 0:
            c.line(pa_x, ry, pa_x + pa_w, ry, color="#e2e8f0", lw=0.5)
        c.text(
            pa_x + 4.0,
            ry + 11.0,
            f"{c_lbl} ({c_sz})",
            size=5.6,
            bold=True,
            color="#0f172a",
            mono=True,
        )
        bar_w_vb = max(3.0, 48.0 * (vb / 2.0))
        c.rect(pa_x + 4.0, ry + 14.5, bar_w_vb, 5.2, fill="#0284c7")
        c.text(
            pa_x + 56.0, ry + 19.2, f"V_B={vb:.2f}", size=5.3, color="#0369a1", bold=True, mono=True
        )

        c.text(pa_x + 124.0, ry + 11.0, ir_lbl, size=5.6, bold=True, color=col, mono=True)
        bar_w_c2 = max(3.0, 52.0 * (c2_val / 2.25))
        c.rect(pa_x + 124.0, ry + 14.5, bar_w_c2, 5.2, fill=col)
        c.text(
            pa_x + 180.0,
            ry + 19.2,
            f"{c2_val:.3f} [{dd1}]",
            size=5.2,
            color="#1e293b",
            bold=True,
            mono=True,
        )

    # Panel (b): Log-Log Lie-Trotter vs Strang Error Scaling (N = 8..2048, g_YM^2 = 1.0)
    pb_x, pb_y, pb_w, pb_h = 304.0, 44.0, 256.0, 126.0
    c.text(
        pb_x,
        pb_y - 5.5,
        "(b) Trotter Scaling ||U_Trot - exp(-i H_KS t)||_2 (g_YM^2 = 1.0)",
        size=7.3,
        bold=True,
        color="#0f172a",
    )
    c.rect(pb_x, pb_y, pb_w, pb_h, fill="#ffffff", stroke="#334155", lw=0.9)

    def _map_pt(n_steps: int, err: float) -> Tuple[float, float]:
        lx = (math.log2(n_steps) - 3.0) / 8.0  # log2(8)=3 .. log2(2048)=11
        ly = (math.log10(max(err, 1e-10)) - (-9.0)) / 8.0  # 1e-9 .. 1e-1
        px = pb_x + 28.0 + lx * (pb_w - 40.0)
        py = pb_y + pb_h - 18.0 - ly * (pb_h - 28.0)
        return px, py

    for exp_y in (-2, -4, -6, -8):
        _, gy = _map_pt(8, 10.0**exp_y)
        c.line(pb_x + 24.0, gy, pb_x + pb_w - 8.0, gy, color="#e2e8f0", lw=0.5, dashed=True)
        c.text(
            pb_x + 21.0, gy + 2.5, f"1e{exp_y}", size=5.2, color="#475569", anchor="end", mono=True
        )

    for n_tick in (8, 32, 128, 512, 2048):
        gx, _ = _map_pt(n_tick, 1e-5)
        c.line(gx, pb_y + 8.0, gx, pb_y + pb_h - 16.0, color="#f1f5f9", lw=0.5)
        c.text(
            gx,
            pb_y + pb_h - 6.0,
            str(n_tick),
            size=5.2,
            color="#334155",
            anchor="middle",
            mono=True,
        )

    lie_pts = [_map_pt(int(row["n_steps"]), float(row["lie_2norm_error"])) for row in g1_series]
    str_pts = [_map_pt(int(row["n_steps"]), float(row["strang_2norm_error"])) for row in g1_series]
    c.polyline(lie_pts, color="#dc2626", lw=1.5, dashed=True)
    c.polyline(str_pts, color="#2563eb", lw=1.7)
    for px, py in lie_pts:
        c.circle(px, py, 2.2, fill="#fecaca", stroke="#dc2626", lw=1.0)
    for px, py in str_pts:
        c.circle(px, py, 2.4, fill="#bfdbfe", stroke="#1d4ed8", lw=1.1)

    c.text(
        pb_x + 34.0,
        pb_y + 18.0,
        f"1st-Order Lie: slope = {lie_slope:.6f}",
        size=5.6,
        bold=True,
        color="#b91c1c",
        mono=True,
    )
    c.text(
        pb_x + 34.0,
        pb_y + 30.0,
        f"2nd-Order Strang: slope = {strang_slope:.6f}",
        size=5.6,
        bold=True,
        color="#1d4ed8",
        mono=True,
    )
    c.text(
        pb_x + 34.0,
        pb_y + 41.0,
        "N=64: 8.14e-6 | N=2048: 7.94e-9",
        size=5.3,
        color="#0f172a",
        mono=True,
    )

    # Panel (c): Native CZ Compression Summary (5Q Gauge-Fixed vs Dense & Unfixed 4-Link)
    pc_x, pc_y, pc_w, pc_h = 304.0, 184.0, 256.0, 72.0
    c.rect(pc_x, pc_y, pc_w, pc_h, fill="#f8fafc", stroke="#334155", lw=0.8)
    c.text(
        pc_x + 6.0,
        pc_y + 11.5,
        "(c) Native CZ Gate Compression (Qiskit opt 3)",
        size=6.6,
        bold=True,
        color="#0f172a",
    )
    bars = [
        (
            "U_B 5Q Struct vs Fez Dense",
            int(fez_cz["UB_CENTER_PI4_19CZ"]),
            int(fez_cz["UB_CENTER_PI4_GENERIC"]),
            "68.18x (19 core)",
        ),
        (
            "S_2(dt) 5Q vs 20Q 4-Link (All)",
            int(gcs["cz_strang_structured_5q_all_to_all"]),
            int(gcs["cz_strang_4link_inplace_20q_all_to_all"]),
            "104.33x CZ",
        ),
        (
            "S_2(dt) 5Q vs 25Q 4-Link (HH)",
            int(gcs["cz_strang_structured_5q_heavy_hex_T5"]),
            int(gcs["cz_strang_4link_ancilla_25q_heavy_hex_25Q"]),
            "178.31x CZ",
        ),
    ]
    for b_i, (b_lbl, cz_s, cz_g, ratio_s) in enumerate(bars):
        by = pc_y + 22.0 + b_i * 16.5
        c.text(pc_x + 6.0, by + 6.5, b_lbl, size=5.2, bold=True, color="#1e293b")
        c.text(
            pc_x + pc_w - 6.0,
            by + 6.5,
            f"{cz_s} vs {cz_g:,} CZ ({ratio_s})",
            size=5.2,
            bold=True,
            color="#047857",
            anchor="end",
            mono=True,
        )

    # Bottom banner
    c.rect(34.0, 268.0, w - 50.0, 44.0, fill="#eff6ff", stroke="#93c5fd", lw=0.8)
    c.text(
        42.0,
        280.5,
        "Maximal-Tree Gauge Fixing & Walsh-Hadamard Plaquette Synthesis (OSIRIS-BT2T-003 Exact Theory Verification):",
        size=6.3,
        bold=True,
        color="#1e3a8a",
    )
    c.text(
        42.0,
        292.0,
        "* Unfixed 4-link state space |2T|^4 = 331,776 (20Q/25Q) collapses to 5Q |2T| = 24 holonomy (7-D Gauss subspace, max L2 discrepancy 2.27e-15).",
        size=5.8,
        color="#1e293b",
    )
    c.text(
        42.0,
        303.0,
        "* Diagonal U_B(dt) requires 16 CZ (all-to-all) / 19 CZ (T5 core; 40 CZ with prep vs 2,727 CZ generic dense = 68.18x compression); 0.0 unused-8 leakage.",
        size=5.8,
        bold=True,
        color="#065f46",
    )
    return c


def _build_fig2(fez_results: Mapping[str, Any]) -> DualPdfPngCanvas:
    """Fig. 2: Hardware-attested 2T magnetic plaquette character-sector trajectories on ibm_fez (job dc38nreq4uuc73a6g0k0)."""
    w, h = 576.0, 324.0
    c = DualPdfPngCanvas(w, h)

    traj0 = fez_results["magnetic_plaquette_trajectories"]["pid_0"]
    traj1 = fez_results["magnetic_plaquette_trajectories"]["pid_1"]
    job_id = str(fez_results["provider_job_id"])

    c.text(
        w / 2.0,
        15.0,
        f"OSIRIS-BT2T-003-FEZ (HARDWARE_ATTESTED, ibm_fez job {job_id}, 61,440 Shots): 2T Plaquette Trajectories",
        size=8.6,
        bold=True,
        anchor="middle",
    )
    c.text(
        w / 2.0,
        26.5,
        f"Echo #1 pid=0 [21,22,23,16,24]: Center r={traj0['pearson_r_center']:.6f}, Hurwitz r={traj0['pearson_r_hurwitz']:.6f}  |  40-CZ Struct 0.8503 vs 2,727-CZ Dense 0.0417 (+80.86 pp)",
        size=6.7,
        color="#065f46",
        bold=True,
        anchor="middle",
    )

    # Panel (a): Center Sector (Delta chi_2D = 4, P_|+>(tau) = cos^2(2*tau))
    pa_x, pa_y, pa_w, pa_h = 34.0, 44.0, 248.0, 198.0
    c.text(
        pa_x,
        pa_y - 5.5,
        "(a) Center Sector (Delta chi_2D = 4): P_|+>(tau) = cos^2(2*tau)",
        size=7.2,
        bold=True,
        color="#0f172a",
    )
    c.rect(pa_x, pa_y, pa_w, pa_h, fill="#ffffff", stroke="#334155", lw=0.9)

    def _map_a(tau_norm: float, prob: float) -> Tuple[float, float]:
        # tau_norm in [0, 1] where 1 corresponds to pi/2
        px = pa_x + 28.0 + tau_norm * (pa_w - 42.0)
        py = pa_y + pa_h - 22.0 - prob * (pa_h - 36.0)
        return px, py

    for p_tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        _, gy = _map_a(0.0, p_tick)
        c.line(pa_x + 24.0, gy, pa_x + pa_w - 10.0, gy, color="#e2e8f0", lw=0.5, dashed=True)
        c.text(
            pa_x + 21.0,
            gy + 2.5,
            f"{p_tick:.2f}",
            size=5.2,
            color="#475569",
            anchor="end",
            mono=True,
        )

    tau_norms_center = [0.0, 0.25, 0.5, 1.0]
    tau_labels_center = ["0", "pi/8", "pi/4", "pi/2"]
    for tn, tl in zip(tau_norms_center, tau_labels_center):
        gx, _ = _map_a(tn, 0.0)
        c.line(gx, pa_y + 10.0, gx, pa_y + pa_h - 20.0, color="#f1f5f9", lw=0.5)
        c.text(gx, pa_y + pa_h - 8.0, tl, size=5.4, color="#334155", anchor="middle", mono=True)

    # Ideal curve cos^2(2*tau)
    ideal_a = [
        _map_a(i / 40.0, math.cos(2.0 * (i / 40.0) * (math.pi / 2.0)) ** 2) for i in range(41)
    ]
    c.polyline(ideal_a, color="#94a3b8", lw=1.3, dashed=True)

    p0_c = [float(pt["measured_p_plus"]) for pt in traj0["center_chi4_trajectory"]]
    p1_c = [float(pt["measured_p_plus"]) for pt in traj1["center_chi4_trajectory"]]
    pts0_c = [_map_a(tn, val) for tn, val in zip(tau_norms_center, p0_c)]
    pts1_c = [_map_a(tn, val) for tn, val in zip(tau_norms_center, p1_c)]
    c.polyline(pts0_c, color="#059669", lw=1.8)
    c.polyline(pts1_c, color="#d97706", lw=1.5, dashed=True)
    for (px, py), val in zip(pts0_c, p0_c):
        c.circle(px, py, 3.0, fill="#a7f3d0", stroke="#047857", lw=1.2)
        c.text(px + 3.0, py - 5.0, f"{val:.4f}", size=5.1, bold=True, color="#065f46", mono=True)
    for (px, py), val in zip(pts1_c, p1_c):
        c.circle(px, py, 2.5, fill="#fde68a", stroke="#b45309", lw=1.1)

    # Null controls & inversion points at tau = pi/4 (tau_norm = 0.5)
    inv0_x, inv0_y = _map_a(0.5, float(traj0["ub_center_pi4_structured_19cz_inversion"]))
    gen0_x, gen0_y = _map_a(0.5, float(traj0["ub_center_pi4_generic_dense_inversion"]))
    im0_x, im0_y = _map_a(0.5, float(traj0["imag_q8_chi0_null_pi4_survival"]))
    un0_x, un0_y = _map_a(0.5, float(traj0["unused8_chi0_null_pi4_survival"]))
    c.circle(inv0_x, inv0_y, 3.3, fill="#bfdbfe", stroke="#1d4ed8", lw=1.3)
    c.text(
        inv0_x + 5.0,
        inv0_y + 2.0,
        "P_|-|=0.8503 (40 CZ)",
        size=5.1,
        bold=True,
        color="#1d4ed8",
        mono=True,
    )
    c.circle(gen0_x, gen0_y, 3.0, fill="#fecaca", stroke="#dc2626", lw=1.2)
    c.text(
        gen0_x + 5.0,
        gen0_y - 2.0,
        "Dense 2727 CZ: 0.0417",
        size=5.0,
        bold=True,
        color="#b91c1c",
        mono=True,
    )
    c.circle(im0_x, im0_y, 2.7, fill="#ddd6fe", stroke="#6d28d9", lw=1.1)
    c.text(
        im0_x - 4.0,
        im0_y - 4.0,
        "Imag-Q8 null: 0.8620",
        size=5.0,
        bold=True,
        color="#5b21b6",
        anchor="end",
        mono=True,
    )
    c.circle(un0_x, un0_y, 2.7, fill="#fbcfe8", stroke="#be185d", lw=1.1)
    c.text(
        un0_x - 4.0,
        un0_y + 8.0,
        "Unused-8 null: 0.7852",
        size=5.0,
        bold=True,
        color="#9d174d",
        anchor="end",
        mono=True,
    )

    # Panel (b): Hurwitz Half-Frequency Sector (Delta chi_2D = 2, P_Phi+(tau) = cos^2(tau))
    pb_x, pb_y, pb_w, pb_h = 304.0, 44.0, 256.0, 198.0
    c.text(
        pb_x,
        pb_y - 5.5,
        "(b) Hurwitz Sector (Delta chi_2D = 2): P_Phi+(tau) = cos^2(tau)",
        size=7.2,
        bold=True,
        color="#0f172a",
    )
    c.rect(pb_x, pb_y, pb_w, pb_h, fill="#ffffff", stroke="#334155", lw=0.9)

    def _map_b(tau_norm: float, prob: float) -> Tuple[float, float]:
        px = pb_x + 28.0 + tau_norm * (pb_w - 42.0)
        py = pb_y + pb_h - 22.0 - prob * (pb_h - 36.0)
        return px, py

    for p_tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        _, gy = _map_b(0.0, p_tick)
        c.line(pb_x + 24.0, gy, pb_x + pb_w - 10.0, gy, color="#e2e8f0", lw=0.5, dashed=True)
        c.text(
            pb_x + 21.0,
            gy + 2.5,
            f"{p_tick:.2f}",
            size=5.2,
            color="#475569",
            anchor="end",
            mono=True,
        )

    tau_norms_hurw = [0.0, 0.25, 0.5, 0.75, 1.0]
    tau_labels_hurw = ["0", "pi/8", "pi/4", "3pi/8", "pi/2"]
    for tn, tl in zip(tau_norms_hurw, tau_labels_hurw):
        gx, _ = _map_b(tn, 0.0)
        c.line(gx, pb_y + 10.0, gx, pb_y + pb_h - 20.0, color="#f1f5f9", lw=0.5)
        c.text(gx, pb_y + pb_h - 8.0, tl, size=5.4, color="#334155", anchor="middle", mono=True)

    ideal_b = [_map_b(i / 40.0, math.cos((i / 40.0) * (math.pi / 2.0)) ** 2) for i in range(41)]
    c.polyline(ideal_b, color="#94a3b8", lw=1.3, dashed=True)

    p0_h = [float(pt["measured_p_phi_plus"]) for pt in traj0["hurwitz_chi2_trajectory"]]
    p1_h = [float(pt["measured_p_phi_plus"]) for pt in traj1["hurwitz_chi2_trajectory"]]
    pts0_h = [_map_b(tn, val) for tn, val in zip(tau_norms_hurw, p0_h)]
    pts1_h = [_map_b(tn, val) for tn, val in zip(tau_norms_hurw, p1_h)]
    c.polyline(pts0_h, color="#2563eb", lw=1.8)
    c.polyline(pts1_h, color="#d97706", lw=1.5, dashed=True)
    for (px, py), val in zip(pts0_h, p0_h):
        c.circle(px, py, 3.0, fill="#bfdbfe", stroke="#1d4ed8", lw=1.2)
        c.text(px + 2.0, py - 5.0, f"{val:.4f}", size=5.0, bold=True, color="#1e3a8a", mono=True)
    for (px, py), val in zip(pts1_h, p1_h):
        c.circle(px, py, 2.5, fill="#fde68a", stroke="#b45309", lw=1.1)

    h_inv_p0 = float(traj0["hurwitz_chi2_trajectory"][-1]["measured_p_phi_minus_inverted"])
    hinv_x, hinv_y = _map_b(1.0, h_inv_p0)
    c.circle(hinv_x, hinv_y, 3.3, fill="#a7f3d0", stroke="#047857", lw=1.3)
    c.text(
        hinv_x - 5.0,
        hinv_y + 2.0,
        f"P_Phi-(pi/2)={h_inv_p0:.4f}",
        size=5.1,
        bold=True,
        color="#065f46",
        anchor="end",
        mono=True,
    )

    c.rect(pb_x + 88.0, pb_y + 12.0, 158.0, 36.0, fill="#f8fafc", stroke="#cbd5e1", lw=0.6)
    c.text(
        pb_x + 93.0,
        pb_y + 22.0,
        f"pid=0 (Echo #1): r = {traj0['pearson_r_hurwitz']:.6f}",
        size=5.4,
        bold=True,
        color="#1d4ed8",
        mono=True,
    )
    c.text(
        pb_x + 93.0,
        pb_y + 32.0,
        f"pid=1 (Q7 drift): r = {traj1['pearson_r_hurwitz']:.6f}",
        size=5.4,
        bold=True,
        color="#b45309",
        mono=True,
    )
    c.text(
        pb_x + 93.0, pb_y + 42.0, "Dashed gray: exact SU(2) cos^2(tau)", size=5.1, color="#475569"
    )

    # Bottom summary box
    c.rect(34.0, 254.0, w - 48.0, 58.0, fill="#ecfdf5", stroke="#6ee7b7", lw=0.8)
    c.text(
        42.0,
        266.5,
        "First Hardware-Attested Non-Abelian 2T subset SU(2) Magnetic Plaquette Character-Frequency Discrimination (ibm_fez):",
        size=6.3,
        bold=True,
        color="#065f46",
    )
    c.text(
        42.0,
        278.5,
        "* Center Sector (Delta chi_2D=4, 40 CZ): P_|+>(tau) = [0.876302, 0.411458, 0.057292, 0.841146] (r=0.997576); P_|-|(pi/4) = 0.850260 ('00010').",
        size=5.7,
        color="#1e293b",
    )
    c.text(
        42.0,
        290.0,
        "* Hurwitz Sector (Delta chi_2D=2, 48 CZ): P_Phi+(tau) = [0.829427, 0.669271, 0.404948, 0.130208, 0.050781] (r=0.997940); P_Phi-(pi/2) = 0.796875 ('00100').",
        size=5.7,
        color="#1e293b",
    )
    c.text(
        42.0,
        301.5,
        "* Null Controls (Delta chi_2D=0 at pi/4): Imaginary-Q8 |i> = 0.861979 ('00001'), Unused-8 |y=24> = 0.785156 ('01100'); 68.18x CZ reduction yields 20.41x fidelity.",
        size=5.7,
        bold=True,
        color="#047857",
    )
    return c


def _build_fig3(fez_results: Mapping[str, Any]) -> DualPdfPngCanvas:
    """Fig. 3: Two-Window OSIRIS-ECHO Intra-Calibration Drift Catch across 10 disjoint T5 placements (50 transmons)."""
    w, h = 576.0, 324.0
    c = DualPdfPngCanvas(w, h)

    placements = fez_results["placement_results"]
    tw = fez_results["two_window_echo_summary"]
    drift = tw["pid1_drift_diagnosis"]
    pws = fez_results["per_workload_spearman"]

    c.text(
        w / 2.0,
        15.0,
        "OSIRIS-BT2T-003-FEZ: Live Two-Window (N_windows = 2, +3.763 h) OSIRIS-ECHO Intra-Calibration Drift Catch",
        size=8.6,
        bold=True,
        anchor="middle",
    )
    c.text(
        w / 2.0,
        26.5,
        f"Win-1 dc35e35s2omc738cvcug -> Win-2 dc38nreq4uuc73a6g0k0  |  rho_echo = +{tw['window2_spearman_raw_echo']:.6f} vs rho_cal = +{tw['window2_spearman_calibration']:.6f} (Delta = +{fez_results['window2_delta_raw_echo_minus_calibration']:.6f})",
        size=6.7,
        color="#1e3a8a",
        bold=True,
        anchor="middle",
    )

    # Panel (a): Window-2 10-Placement Scatter/Bar Comparison (Echo vs Held-Out Composite vs Stale Cal Rank)
    pa_x, pa_y, pa_w, pa_h = 34.0, 44.0, 276.0, 200.0
    c.text(
        pa_x,
        pa_y - 5.5,
        "(a) Window-2 10 Disjoint T5 Placements (50 Transmons): Echo vs Held-Out",
        size=7.1,
        bold=True,
        color="#0f172a",
    )
    c.rect(pa_x, pa_y, pa_w, pa_h, fill="#fafafa", stroke="#334155", lw=0.9)

    cal_scores = [float(p["calibration_rank_score"]) for p in placements]
    echo_scores = [float(p["echo_probe_16cz"]) for p in placements]
    cal_ranks = _competition_ranks_desc(cal_scores)
    echo_ranks = _competition_ranks_desc(echo_scores)

    c.rect(pa_x + 0.6, pa_y + 0.6, pa_w - 1.2, 18.0, fill="#e2e8f0")
    c.text(
        pa_x + 4.0,
        pa_y + 12.0,
        "PID (Cal# -> Echo#)",
        size=5.5,
        bold=True,
        color="#0f172a",
        mono=True,
    )
    c.text(
        pa_x + 92.0,
        pa_y + 12.0,
        "ECHO_16CZ / Plaq_2WL / HeldOut_4WL",
        size=5.5,
        bold=True,
        color="#0f172a",
    )
    c.line(pa_x, pa_y + 18.6, pa_x + pa_w, pa_y + 18.6, color="#334155", lw=0.7)

    row_h = (pa_h - 20.0) / 10.0
    for idx, p in enumerate(placements):
        pid = int(p["placement_id"])
        ry = pa_y + 19.5 + idx * row_h
        if idx > 0:
            c.line(pa_x, ry, pa_x + pa_w, ry, color="#e2e8f0", lw=0.4)
        if pid == 0:
            c.rect(pa_x + 0.8, ry + 0.4, pa_w - 1.6, row_h - 0.8, fill="#ecfdf5")
        elif pid == 1:
            c.rect(pa_x + 0.8, ry + 0.4, pa_w - 1.6, row_h - 0.8, fill="#fff1f2")

        lbl_col = "#065f46" if pid == 0 else ("#be123c" if pid == 1 else "#0f172a")
        c.text(
            pa_x + 4.0,
            ry + 11.8,
            f"p{pid} (#{cal_ranks[idx]}->#{echo_ranks[idx]})",
            size=5.4,
            bold=(pid in (0, 1)),
            color=lbl_col,
            mono=True,
        )

        e_val = float(p["echo_probe_16cz"])
        pl_val = float(p["plaquette_2wl_composite"])
        ho_val = float(p["heldout_composite"])

        bx0 = pa_x + 84.0
        max_bw = 100.0
        c.rect(bx0, ry + 3.0, max(2.0, max_bw * e_val), 3.6, fill="#2563eb")
        c.rect(bx0, ry + 7.4, max(2.0, max_bw * pl_val), 3.6, fill="#059669")
        c.rect(bx0, ry + 11.8, max(2.0, max_bw * ho_val), 3.6, fill="#7c3aed")
        c.text(
            pa_x + pa_w - 4.0,
            ry + 11.8,
            f"{e_val:.3f}/{pl_val:.3f}/{ho_val:.3f}",
            size=5.1,
            bold=(pid in (0, 1)),
            color=lbl_col,
            anchor="end",
            mono=True,
        )

    # Panel (b): Hub Transmon Q7 Drift Diagnosis & Two-Window Regret / Spearman Comparison
    pb_x, pb_y, pb_w, pb_h = 324.0, 44.0, 236.0, 200.0
    c.text(
        pb_x,
        pb_y - 5.5,
        "(b) Hub Q7 Drift Catch & Two-Window Spearman / Regret",
        size=7.1,
        bold=True,
        color="#0f172a",
    )
    c.rect(pb_x, pb_y, pb_w, pb_h, fill="#ffffff", stroke="#334155", lw=0.9)

    # Top sub-box: pid=1 Q7 collapse between W1 and W2
    c.rect(pb_x + 6.0, pb_y + 8.0, pb_w - 12.0, 74.0, fill="#fff1f2", stroke="#fda4af", lw=0.7)
    c.text(
        pb_x + 11.0,
        pb_y + 19.0,
        "Drifted pid=1 [5,6,7,8,17] Hub Q7 Excursion (+3.763 h):",
        size=5.8,
        bold=True,
        color="#9f1239",
    )
    c.text(
        pb_x + 11.0,
        pb_y + 31.0,
        f"* IDENTITY_CTRL_0CZ: {drift['w1_identity_0cz_fidelity']:.6f} (Win-1) -> {drift['w2_identity_0cz_fidelity']:.6f} (Win-2)",
        size=5.3,
        color="#1e293b",
        mono=True,
    )
    c.text(
        pb_x + 11.0,
        pb_y + 42.0,
        f"* ECHO_PROBE_16CZ:   {drift['w1_echo_probe_16cz']:.6f} (Win-1) -> {drift['w2_echo_probe_16cz']:.6f} (Win-2)",
        size=5.3,
        color="#1e293b",
        mono=True,
    )
    c.text(
        pb_x + 11.0,
        pb_y + 53.0,
        f"* Q7 |0>->|1> Bitmask-4 Excitation: +{100.0 * float(drift['w2_identity_0cz_q7_bitmask4_excitation']):.2f}% ({drift['w2_identity_0cz_q7_bitmask4_excitation']:.6f})",
        size=5.3,
        bold=True,
        color="#be123c",
        mono=True,
    )
    c.text(
        pb_x + 11.0,
        pb_y + 64.0,
        "* Stale Cal keeps pid=1 at #1 (regret 0.361002 = 36.10 pp)!",
        size=5.3,
        bold=True,
        color="#9f1239",
    )
    c.text(
        pb_x + 11.0,
        pb_y + 75.0,
        "* Live ECHO switches #1 -> pid=0 (regret 0.000000 in Win-1 & Win-2)!",
        size=5.3,
        bold=True,
        color="#047857",
    )

    # Bottom sub-box: Per-workload & 2-window Spearman table
    c.rect(pb_x + 6.0, pb_y + 90.0, pb_w - 12.0, 102.0, fill="#f8fafc", stroke="#cbd5e1", lw=0.7)
    c.text(
        pb_x + 11.0,
        pb_y + 101.5,
        "Spearman rho Across 10 Placements (Echo vs Stale Cal):",
        size=5.7,
        bold=True,
        color="#0f172a",
    )
    sp_rows = [
        (
            "Win-2 Held-Out 4WL Composite",
            float(tw["window2_spearman_raw_echo"]),
            float(tw["window2_spearman_calibration"]),
            float(fez_results["window2_delta_raw_echo_minus_calibration"]),
        ),
        (
            "Win-2 CLIFFORD_30CZ (30 CZ)",
            float(pws["clifford_30cz"]["rho_echo"]),
            float(pws["clifford_30cz"]["rho_calibration"]),
            float(pws["clifford_30cz"]["delta_echo_minus_calibration"]),
        ),
        (
            "Win-2 MIRROR_24CZ (24 CZ)",
            float(pws["mirror_24cz"]["rho_echo"]),
            float(pws["mirror_24cz"]["rho_calibration"]),
            float(pws["mirror_24cz"]["delta_echo_minus_calibration"]),
        ),
        (
            "Win-2 UB_HURWITZ_PI2 (48 CZ)",
            float(pws["ub_hurwitz_pi2_19cz"]["rho_echo"]),
            float(pws["ub_hurwitz_pi2_19cz"]["rho_calibration"]),
            float(pws["ub_hurwitz_pi2_19cz"]["delta_echo_minus_calibration"]),
        ),
        (
            "Win-2 UB_CENTER_PI4 (40 CZ)",
            float(pws["ub_center_pi4_19cz"]["rho_echo"]),
            float(pws["ub_center_pi4_19cz"]["rho_calibration"]),
            float(pws["ub_center_pi4_19cz"]["delta_echo_minus_calibration"]),
        ),
        (
            "Win-2 Plaquette 2WL Composite",
            float(pws["plaquette_2wl_composite"]["rho_echo"]),
            float(pws["plaquette_2wl_composite"]["rho_calibration"]),
            float(pws["plaquette_2wl_composite"]["delta_echo_minus_calibration"]),
        ),
        (
            "2-Window Mean (Win-1 + Win-2)",
            float(tw["mean_2window_spearman_raw_echo"]),
            float(tw["mean_2window_spearman_calibration"]),
            float(tw["mean_2window_delta_raw_echo_minus_cal"]),
        ),
    ]
    for s_i, (s_lbl, r_e, r_c, d_r) in enumerate(sp_rows):
        sy = pb_y + 114.0 + s_i * 11.2
        c.text(pb_x + 11.0, sy, s_lbl, size=5.0, bold=(s_i in (0, 6)), color="#1e293b")
        c.text(
            pb_x + pb_w - 10.0,
            sy,
            f"{r_e:+.3f} vs {r_c:+.3f} ({d_r:+.3f})",
            size=5.0,
            bold=(s_i in (0, 6)),
            color="#047857" if d_r > 0.3 else "#1d4ed8",
            anchor="end",
            mono=True,
        )

    # Bottom summary banner
    c.rect(34.0, 254.0, w - 48.0, 58.0, fill="#eff6ff", stroke="#93c5fd", lw=0.8)
    c.text(
        42.0,
        266.5,
        "Live Two-Window OSIRIS-ECHO Drift Catch Summary (50 Physical Transmons on ibm_fez, 20 Placement-Window Observations):",
        size=6.2,
        bold=True,
        color="#1e3a8a",
    )
    c.text(
        42.0,
        278.5,
        "* Window 1 (dc35e35s2omc738cvcug, 03:33Z): pid=1 ranked #1 by both Calibration and Echo (ID_0CZ=0.962240, ECHO_16CZ=0.815104, regret=0.000000).",
        size=5.7,
        color="#1e293b",
    )
    c.text(
        42.0,
        290.0,
        "* Window 2 (dc38nreq4uuc73a6g0k0, 07:19Z, +3.763 h): Hub Q7 on pid=1 suffered +31.12% |0>->|1> drift; stale calibration collapsed to rho=+0.454545 (regret 0.361002).",
        size=5.7,
        color="#1e293b",
    )
    c.text(
        42.0,
        301.5,
        "* Live ECHO_PROBE_16CZ caught the Q7 excursion in real time, promoted pid=0 to #1, and achieved rho=+0.948333 (Delta rho=+0.493787) with 0.000000 top-1 regret!",
        size=5.7,
        bold=True,
        color="#047857",
    )
    return c


def _build_preprint_pdf(
    manifest: Mapping[str, Any],
    fez_results: Mapping[str, Any],
    fez_job: Mapping[str, Any],
    fig1: DualPdfPngCanvas,
    fig2: DualPdfPngCanvas,
    fig3: DualPdfPngCanvas,
) -> bytes:
    """Render deterministic >=5-page standalone US-Letter PDF 1.4 preprint with CreationDate=2023-11-14T22:13:20Z."""
    p2 = manifest["part2_multilink_holonomy_and_gauge_isomorphism"]
    p4 = manifest["part4_trotter_suzuki_error_scaling"]
    placements = fez_results["placement_results"]
    tw = fez_results["two_window_echo_summary"]

    job_id = str(fez_results["provider_job_id"])
    w1_job_id = str(tw["window1_job_id"])
    qpy_sha = str(fez_job["qpy_sha256"])
    raw_gz_sha = str(fez_results["raw_counts_gz_sha256"])
    wa_commit = str(fez_job["write_ahead_commit_sha"])[:8]

    doc = PreprintDocument(
        short_title="Kogut-Susskind 2T Plaquette & Live Two-Window ibm_fez Echo Drift Catch (OSIRIS-BT2T-003)",
        manuscript_id="OSIRIS-BT2T-003 / OSIRIS-BT2T-003-FEZ",
        execution_tier="SIMULATED + HARDWARE_ATTESTED",
        footer_left=f"SIMULATED (manifest_bt2t003) + HARDWARE_ATTESTED (ibm_fez {job_id})  |  CreationDate={CREATION_DATE_ISO}",
    )

    abstract = (
        "Digital quantum simulation of (2+1)D non-abelian SU(2) Kogut-Susskind lattice Yang-Mills theory over the "
        "binary tetrahedral gauge group 2T == SL(2, F_3) (|2T| = 24) confronts severe register and entangling-gate "
        "overhead when unfixed four-link plaquette Hilbert spaces (|2T|^4 = 24^4 = 331,776 group configurations "
        "embedded in 20 qubits of dimension 32^4 = 1,048,576, or 25 qubits with a 5-qubit holonomy ancilla) are "
        "evolved directly. Under frozen 5-qubit Hurwitz quaternion encoding bt2t-enc-1 (SHA-256 "
        f"{manifest['encoding_sha256'][:16]}...), we report a dual-manifest study combining exhaustive offline "
        "algebraic and Trotter-Suzuki verification (OSIRIS-BT2T-003, TIER: SIMULATED / EXACT_STATEVECTOR, 34/34 "
        "checks, provider_job_id: null) with write-ahead hardware execution across 50 physical transmons on the "
        f"156-qubit IBM Heron r2 processor ibm_fez (OSIRIS-BT2T-003-FEZ, TIER: HARDWARE_ATTESTED, provider_job_id: "
        f"{job_id}, 240 PUBs x 256 = 61,440 shots across K = 10 disjoint T5 heavy-hex subgraphs, staged at commit "
        f"{wa_commit}, qpy_sha256 = {qpy_sha[:16]}..., raw_counts_gz_sha256 = {raw_gz_sha[:16]}...). First, maximal-tree "
        "gauge fixing collapses the 4-link single-plaquette Hilbert space from 331,776 configurations (20Q/25Q) to "
        "24 physical holonomy states on 5 qubits (7-dimensional conjugation-invariant Gauss subspace, 8 strictly "
        f"decoupled unused states), matching unfixed 4-link evolution to {p2['four_link_vs_5q_random_superposition_max_l2_error']:.2e}. "
        "The magnetic plaquette potential V_B(U) = (2 - Re chi_2D(U))/2 in {0, 1/2, 1, 3/2, 2} across the 7 conjugacy "
        "classes C1..C7 (paired with dyadic Peter-Weyl Casimir eigenvalues {0, 3/4, 3/2, 15/8, 9/4} and kinetic sector "
        "weights d_rho(d_rho - 1) in {0, 2, 6}) compiles via a 5-qubit Walsh-Hadamard Gray-code phase circuit to 16 CZ "
        "gates (all-to-all) or 19 core CZ gates on heavy-hex T5 (40 CZ including state prep/unprep vs 2,727 CZ for "
        "generic dense 32x32 QFT-conjugated UnitaryGate synthesis: a 68.18x native CZ reduction on ibm_fez). Combined "
        "with the 5-qubit Peter-Weyl electric propagator U_E(dt) (181 CZ all-to-all / 312 CZ T5), second-order Strang "
        f"splitting S_2(dt) requires 209 CZ (all-to-all) / 350 CZ (T5) -- a 104.33x to 178.31x CZ compression over "
        f"unfixed 20Q/25Q 4-link Trotterization (21,804 to 62,407 CZ) -- with empirical O(dt^2) log-log slope "
        f"{p4['by_coupling']['g_ym_sq_1.0']['strang_loglog_slope']:.6f}. Second, on Echo-selected top-1 placement "
        "pid = 0 ([21, 22, 23, 16, 24]) on ibm_fez, the 19-CZ core magnetic propagator resolves both non-abelian "
        "character frequencies: Center sector (Delta chi_2D = 4, ideal cos^2(2*tau)) yields P_|+>(tau) = [0.876302, "
        "0.411458, 0.057292, 0.841146] (Pearson r = 0.997576) with P_|-|(pi/4) = 0.850260 ('00010') vs 0.041667 for "
        "the 2,727-CZ generic dense comparator (+0.808594 = 80.86 pp / 20.41x advantage); Hurwitz half-frequency "
        "sector (Delta chi_2D = 2, ideal cos^2(tau)) yields P_Phi+(tau) = [0.829427, 0.669271, 0.404948, 0.130208, "
        "0.050781] (Pearson r = 0.997940) with P_Phi-(pi/2) = 0.796875 ('00100'); and zero-differential-character "
        "controls (Delta chi_2D = 0 at tau = pi/4) preserve 0.861979 on imaginary-Q8 |i> ('00001') and 0.785156 on "
        "unused-8 |y=24> ('01100') (while Q7-drifted pid = 1 [5, 6, 7, 8, 17] retains Center r = 0.999283 and Hurwitz r = 0.977581 at reduced envelope). Third, across a two-window (+3.763 h) campaign spanning Window 1 (job "
        f"{w1_job_id}) and Window 2 (job {job_id}), hub transmon Q7 of pid = 1 ([5, 6, 7, 8, 17]) suffered an "
        "unflagged +31.12% |0>->|1> SPAM/readout drift excursion (IDENTITY_CTRL_0CZ dropping 0.962240 -> 0.677083, "
        "ECHO_PROBE_16CZ dropping 0.815104 -> 0.562500). Stale backend calibration kept pid = 1 ranked #1, collapsing "
        "to Spearman rho_cal = +0.454545 with 0.361002 top-1 regret, whereas live ECHO_PROBE_16CZ promoted pid = 0 "
        "to #1 and achieved rho_echo = +0.948333 (Delta rho = +0.493787) with 0.000000 top-1 regret in both windows."
    )

    doc.add_title_block(
        title=(
            "Maximal-Tree Gauge-Fixed Binary Tetrahedral (2T subset SU(2)) Kogut-Susskind Plaquette "
            "Dynamics, 68.18x Walsh-Hadamard Magnetic Gate Compression, and Live Two-Window "
            "OSIRIS-ECHO Intra-Calibration Drift Catch Across 50 Transmons on ibm_fez"
        ),
        author="Devin Phillip Davis",
        affiliation="Agile Defense Systems LLC, Lexington, Kentucky, USA",
        date_str="November 25, 2026",
        metadata_rows=[
            (
                "Simulated Theory Manifest",
                f"OSIRIS-BT2T-003 | TIER: SIMULATED / EXACT_STATEVECTOR | provider_job_id: null | 0 QPU s | 34/34 checks | RFC 8785 SHA-256: {manifest['rfc8785_canonical_sha256'][:16]}...",
            ),
            (
                "Hardware Execution Manifest",
                f"OSIRIS-BT2T-003-FEZ | TIER: HARDWARE_ATTESTED | ibm_fez job {job_id} (19 QPU s, 240 PUBs x 256 = 61,440 shots, K=10 T5 subgraphs = 50 transmons) + Win-1 job {w1_job_id}",
            ),
            (
                "Write-Ahead Governance",
                f"Commit {wa_commit} | QPY SHA-256: {qpy_sha[:16]}... | Raw Counts GZ SHA-256: {raw_gz_sha[:16]}... | CreationDate={CREATION_DATE_ISO}",
            ),
            (
                "Frozen Encoding",
                f"bt2t-enc-1 ({manifest['encoding_sha256']})",
            ),
        ],
        abstract=abstract,
    )

    doc.add_callout_box(
        "DUAL-PROVENANCE BOUNDARY: SIMULATED EXACT STATEVECTOR (OSIRIS-BT2T-003) VS. HARDWARE_ATTESTED IBM_FEZ (OSIRIS-BT2T-003-FEZ)",
        [
            "Theory & Proxy Compilation Boundary (SIMULATED, provider_job_id: null): Exact 4-link |2T|^4 = 331,776 to 5Q |2T| = 24 maximal-tree gauge isomorphism (max L2 discrepancy 2.27e-15), 7-class magnetic potential V_B(U) in {0, 1/2, 3/4, 1, 5/4, 3/2, 2} / dyadic Casimir spectrum c_2(rho) in {0, 3/4, 3/2, 15/8, 9/4}, and 2nd-order Strang splitting S_2(dt) (209 CZ all-to-all / 350 CZ T5 vs 21,804-62,407 CZ unfixed 4-link = 104.33x-178.31x compression, slope 2.000008) are offline linear-algebra and local PROXY compilation results.",
            f"Hardware Magnetic Plaquette Trajectories (HARDWARE_ATTESTED, ibm_fez job {job_id}, 61,440 shots): Only the 16-CZ / 19-CZ Walsh-Hadamard magnetic plaquette propagator U_B(tau) (40 CZ with Center/Null prep, 48 CZ with Hurwitz Bell prep) and 2,727-CZ generic dense comparator (68.18x CZ reduction) were executed on ibm_fez. Full 350-CZ Strang steps S_2(dt) remain beyond unmitigated heavy-hex coherence and are scoped for all-to-all QCCD / fault-tolerant hardware.",
            f"Live Two-Window OSIRIS-ECHO Intra-Calibration Drift Catch (N_windows = 2, +3.763 h): Between Window 1 ({w1_job_id}) and Window 2 ({job_id}), hub qubit Q7 on pid=1 ([5,6,7,8,17]) experienced an unflagged +31.12% |0>->|1> excitation excursion. Stale calibration remained frozen on pid=1 (rho_cal = +0.454545, top-1 regret 0.361002), whereas live ECHO_PROBE_16CZ switched top-1 to pid=0 ([21,22,23,16,24]), achieving rho_echo = +0.948333 (Delta rho = +0.493787) and 0.000000 top-1 regret across both windows.",
        ],
        tone="success",
    )

    doc.add_section("1. Maximal-Tree Gauge Fixing and 5-Qubit Kogut-Susskind Plaquette Hamiltonian")
    doc.add_paragraph(
        "On a single elementary square plaquette with four oriented non-abelian gauge links (g_1, g_2, g_3, g_4) in "
        "2T^4 (|2T| = 24, |2T|^4 = 331,776 basis configurations), encoding each link into a 5-qubit bt2t-enc-1 "
        "register requires 20 data qubits (2^20 = 1,048,576 states) for in-place holonomy evaluation or 25 qubits "
        "(2^25 = 33,554,432 states) with a 5-qubit holonomy workspace register. Imposing local Gauss-law invariance "
        "at three of the four vertices via maximal-tree gauge fixing transforms (g_1, g_2, g_3, g_4) -> (e, e, e, U) "
        "where U = g_1 g_2 g_3^{-1} g_4^{-1} in 2T is the basepoint plaquette holonomy, subject to residual basepoint "
        "conjugation Ad_h|U> = |h U h^{-1}> = L_h R_h|U> for all h in 2T. This collapses the register width by 4.0x "
        "(20Q -> 5Q) or 5.0x (25Q -> 5Q) down to a single 5-qubit holonomy register (24 active group states 0..23, "
        "8 decoupled unused states 24..31, and a 7-dimensional conjugation-invariant physical Gauss subspace spanned "
        "by the 7 conjugacy classes C1..C7 of 2T)."
    )
    doc.add_equation(
        [
            "H_KS = H_E + H_B = (g_YM^2 / 2) Delta_{2T} - (1 / g_YM^2) sum_{U in 2T} Re chi_{2D}(U) |U><U|",
            "V_B(U) = (2 - Re chi_{2D}(U)) / 2 in {0, 1/2, 1, 3/2, 2},   c_2(rho) in {0, 3/4, 3/2, 15/8, 9/4}",
            "T_E = sum_{rho} d_rho (d_rho - 1) Pi_rho in {0 (3x1D A_3), 2 (3x2D E_{1,2,3}), 6 (1x3D T)}",
            "S_2(dt) = U_B(dt/2) F_tilde_{2T}^+ D_E(dt) F_tilde_{2T} U_B(dt/2),   ||S_2(dt) - U_4link(dt)||_2 <= 2.27e-15",
        ],
        eq_num="1",
    )

    doc.add_figure(
        fig1,
        "Exact 2T single-plaquette magnetic potential V_B(U) across the 7 conjugacy classes C1..C7 paired with the "
        "7 Peter-Weyl Casimir irreps, log-log 1st-order Lie-Trotter (slope 1.000003) vs 2nd-order Strang splitting "
        "(slope 2.000008) error scaling across N = 8..2048 steps, and native CZ gate compression (68.18x for U_B on "
        "ibm_fez and 104.33x-178.31x for gauge-fixed 5Q S_2(dt) vs unfixed 20Q/25Q 4-link Trotterization).",
        max_height=256.0,
    )

    t1_rows = [
        [
            "C1 = {+1}",
            "1",
            "0 (c=0, +1)",
            "+2.0",
            "0.0 (0)",
            "-2.0",
            "chi_0 (A_0, d=1)",
            "0",
            "0",
            "0.000 (0)",
        ],
        [
            "C2 = {-1}",
            "1",
            "1 (c=0, -1)",
            "-2.0",
            "2.0 (2)",
            "+2.0",
            "chi_1 (A_1, d=1)",
            "0",
            "12",
            "2.250 (9/4)",
        ],
        [
            "C3 = {+-i,+-j,+-k}",
            "6",
            "2..7 (c=0, Q8)",
            "0.0",
            "1.0 (1)",
            "0.0",
            "chi_2 (A_2, d=1)",
            "0",
            "12",
            "2.250 (9/4)",
        ],
        [
            "C4 (Hurwitz+ c=1)",
            "4",
            "8,11,13,14",
            "+1.0",
            "0.5 (1/2)",
            "-1.0",
            "rho_2,0 (E_1, d=2)",
            "2",
            "4",
            "0.750 (3/4)",
        ],
        [
            "C5 (Hurwitz+ c=2)",
            "4",
            "9,10,12,15",
            "+1.0",
            "0.5 (1/2)",
            "-1.0",
            "rho_2,1 (E_2, d=2)",
            "2",
            "10",
            "1.875 (15/8)",
        ],
        [
            "C6 (Hurwitz- c=2)",
            "4",
            "16,19,21,22",
            "-1.0",
            "1.5 (3/2)",
            "+1.0",
            "rho_2,2 (E_3, d=2)",
            "2",
            "10",
            "1.875 (15/8)",
        ],
        [
            "C7 (Hurwitz- c=1)",
            "4",
            "17,18,20,23",
            "-1.0",
            "1.5 (3/2)",
            "+1.0",
            "rho_3 (T, d=3)",
            "6",
            "8",
            "1.500 (3/2)",
        ],
        [
            "Unused-8 Padded",
            "8",
            "24..31 (c=3)",
            "0.0 (dec)",
            "-- (I_8)",
            "0.0",
            "Decoupled I_8",
            "0",
            "0",
            "0.000 (0)",
        ],
    ]
    doc.add_table(
        "TABLE I. Conjugacy Classes C1..C7 of 2T subset SU(2), Magnetic Potential V_B(U), and Dual Peter-Weyl Irreps (SIMULATED, OSIRIS-BT2T-003).",
        [
            "Class",
            "|C|",
            "bt2t-enc-1 Indices",
            "chi_2D",
            "V_B(U)",
            "H_B(1)",
            "Dual Irrep",
            "d(d-1)",
            "lam_Cay",
            "c_2(rho)",
        ],
        t1_rows,
        font_size=6.6,
    )

    t2_rows = [
        [
            "U_B(dt) 5Q Walsh-Hadamard",
            "5Q",
            "16 CZ (d=77)",
            "19 CZ (d=88)",
            "16 ZZ (d=80)",
            "1.78e-15",
            "Core magnetic phase gate",
        ],
        [
            "U_B(pi/4) Center + Prep",
            "5Q",
            "--",
            "40 CZ (ibm_fez)",
            "--",
            "9.45e-15",
            "68.18x vs 2,727-CZ Dense",
        ],
        [
            "U_B(pi/2) Hurwitz + Prep",
            "5Q",
            "--",
            "48 CZ (ibm_fez)",
            "--",
            "9.45e-15",
            "Bell coset Ramsey arm",
        ],
        [
            "U_B(pi/4) Generic Dense",
            "5Q",
            "--",
            "2,727 CZ (fez)",
            "--",
            "<=1e-13",
            "QFT-conjugated UnitaryGate",
        ],
        [
            "U_E(dt) 5Q Structured",
            "5Q",
            "181 CZ (d=647)",
            "312 CZ (d=937)",
            "--",
            "2.25e-15",
            "Casimir Peter-Weyl kinetic",
        ],
        [
            "S_2(dt) 5Q Structured",
            "5Q",
            "209 CZ (d=790)",
            "350 CZ (d=1,080)",
            "208 ZZ (d=867)",
            "2.44e-15",
            "1.00x (5Q gauge-fixed ref)",
        ],
        [
            "S_2(dt) 5Q Dense Unitary",
            "5Q",
            "423 CZ (d=2,029)",
            "613 CZ (d=2,543)",
            "423 ZZ (d=1,995)",
            "8.20e-14",
            "2.02x (All) / 1.75x (T5)",
        ],
        [
            "S_2(dt) 5Q QSD-QFT",
            "5Q",
            "906 CZ (d=4,294)",
            "1,400 CZ (d=5,473)",
            "--",
            "9.46e-14",
            "4.33x (All) / 4.00x (T5)",
        ],
        [
            "S_2(dt) 4-Link In-Place",
            "20Q",
            "21,804 CZ (68k)",
            "47,291 CZ (107k)",
            "--",
            "2.34e-15",
            "104.33x (All) / 135.12x (HH)",
        ],
        [
            "S_2(dt) 4-Link Ancilla",
            "25Q",
            "28,735 CZ (89k)",
            "62,407 CZ (142k)",
            "--",
            "2.27e-15",
            "137.49x (All) / 178.31x (HH)",
        ],
    ]
    doc.add_table(
        "TABLE II. Native Gate Compression (5Q Gauge-Fixed vs Dense & Unfixed 20Q/25Q 4-Link) and Strang Scaling (SIMULATED / PROXY).",
        [
            "Circuit Architecture",
            "Width",
            "All-to-All CZ",
            "Heavy-Hex CZ",
            "pytket ZZPhase",
            "2-Norm Err",
            "Compression / Role",
        ],
        t2_rows,
        font_size=6.5,
    )

    doc.add_section("2. Hardware-Attested 2T Magnetic Plaquette Trajectories on ibm_fez")
    doc.add_paragraph(
        f"Under write-ahead preregistration OSIRIS-BT2T-003-FEZ (commit {wa_commit}, QPY SHA-256 {qpy_sha[:16]}...), "
        f"we executed 240 PUBs (3 randomized blocks x 80 PUBs/block x 256 shots/PUB = 61,440 shots, job {job_id}, "
        "19 QPU s) across K = 10 vertex-disjoint 5-qubit T5 subgraphs (50 transmons) on ibm_fez. Under the frozen "
        "encoding bt2t-enc-1, all 16 Hurwitz units (indices 8..23, cosets c in {1, 2}) satisfy chi_{2D}(g) = "
        "(Z_4 - Z_3)/2 with zero entangling gates, while the Q_8 center {+1, -1} (indices 0, 1) is isolated on "
        "|0000>_{q4..q1} (x) Z_0. Consequently, U_B(tau) reduces to a 16-CNOT Walsh-Hadamard Gray-code expansion of "
        "(I - Z_4)(I - Z_3)(I - Z_2)(I - Z_1)Z_0/8 routed onto T5 in 19 core CZ gates (40 total CZ with Center/Null "
        "state preparation and measurement; 48 total CZ with Hurwitz Bell preparation and unpreparation), compared "
        "against 2,727 CZ gates for generic dense 32x32 QFT-conjugated UnitaryGate synthesis (68.18x CZ reduction)."
    )

    doc.add_figure(
        fig2,
        f"Hardware-attested 2T magnetic plaquette character-sector trajectories on ibm_fez (job {job_id}, 61,440 "
        "shots) comparing Echo-selected #1 placement pid = 0 ([21, 22, 23, 16, 24]) against Q7-drifted pid = 1 "
        "([5, 6, 7, 8, 17]): (a) Center sector (Delta chi_2D = 4, P_|+>(tau) = cos^2(2*tau), r = 0.997576 on pid=0) "
        "with 0.850260 inversion vs 0.041667 for 2,727-CZ generic dense compilation (+80.86 pp / 20.41x) and "
        "Delta chi_2D = 0 null controls (0.861979 on |i>, 0.785156 on unused-8 |y=24>); (b) Hurwitz half-frequency "
        "sector (Delta chi_2D = 2, P_Phi+(tau) = cos^2(tau), r = 0.997940 on pid=0) with 0.796875 inversion at pi/2.",
        max_height=256.0,
    )

    t4_rows = [
        [
            "Center (Delta chi=4)",
            "tau = 0",
            "40",
            "0 ('00000')",
            "1.000000",
            "0.876302",
            "0.548177",
            "Full-frequency start",
        ],
        [
            "Center (Delta chi=4)",
            "tau = pi/8",
            "40",
            "0 ('00000')",
            "0.500000",
            "0.411458",
            "0.279948",
            "Half-way quadrature",
        ],
        [
            "Center (Delta chi=4)",
            "tau = pi/4 (P_|+>)",
            "40",
            "0 ('00000')",
            "0.000000",
            "0.057292",
            "0.045573",
            "Depleted initial state",
        ],
        [
            "Center Inv (Struct)",
            "tau = pi/4 (P_|-|)",
            "40",
            "2 ('00010')",
            "1.000000",
            "0.850260",
            "0.539062",
            "+80.86 pp / 20.41x vs Dense",
        ],
        [
            "Center Inv (Generic)",
            "tau = pi/4 (P_|-|)",
            "2,727",
            "2 ('00010')",
            "1.000000",
            "0.041667",
            "0.046875",
            "68.18x more CZ -> depolarized",
        ],
        [
            "Center (Delta chi=4)",
            "tau = pi/2",
            "40",
            "0 ('00000')",
            "1.000000",
            "0.841146",
            "0.553385",
            "Full revival (r=0.997576)",
        ],
        [
            "Hurwitz (Delta chi=2)",
            "tau = 0",
            "48",
            "0 ('00000')",
            "1.000000",
            "0.829427",
            "0.460938",
            "Half-frequency Bell start",
        ],
        [
            "Hurwitz (Delta chi=2)",
            "tau = pi/8",
            "48",
            "0 ('00000')",
            "0.853553",
            "0.669271",
            "0.393229",
            "cos^2(pi/8) = 0.853553",
        ],
        [
            "Hurwitz (Delta chi=2)",
            "tau = pi/4",
            "48",
            "0 ('00000')",
            "0.500000",
            "0.404948",
            "0.290365",
            "cos^2(pi/4) = 0.500000",
        ],
        [
            "Hurwitz (Delta chi=2)",
            "tau = 3pi/8",
            "48",
            "0 ('00000')",
            "0.146447",
            "0.130208",
            "0.239583",
            "cos^2(3pi/8) = 0.146447",
        ],
        [
            "Hurwitz (Delta chi=2)",
            "tau = pi/2 (P_Phi+)",
            "48",
            "0 ('00000')",
            "0.000000",
            "0.050781",
            "0.221354",
            "Depleted Bell+ (r=0.997940)",
        ],
        [
            "Hurwitz Inv (Struct)",
            "tau = pi/2 (P_Phi-)",
            "48",
            "4 ('00100')",
            "1.000000",
            "0.796875",
            "0.361979",
            "79.69% Bell- inversion",
        ],
        [
            "Imag-Q8 Null (chi=0)",
            "tau = pi/4 (|i>)",
            "40",
            "1 ('00001')",
            "1.000000",
            "0.861979",
            "0.524740",
            "+80.47 pp above Center P_|+>",
        ],
        [
            "Unused-8 Null (chi=0)",
            "tau = pi/4 (|y=24>)",
            "40",
            "12 ('01100')",
            "1.000000",
            "0.785156",
            "0.380208",
            "Decoupled I_8 subspace holds",
        ],
    ]
    doc.add_table(
        f"TABLE IV. Hardware-Attested 2T Magnetic Plaquette Trajectories on ibm_fez (job {job_id}): Echo #1 pid=0 vs Drifted pid=1.",
        [
            "Sector / Arm",
            "Point",
            "CZ",
            "Routed y",
            "Ideal P",
            "pid=0 (Echo #1)",
            "pid=1 (Drifted)",
            "Physical Interpretation",
        ],
        t4_rows,
        font_size=6.3,
    )

    doc.add_section(
        "3. Live Two-Window OSIRIS-ECHO Intra-Calibration Drift Catch Across 50 Transmons"
    )
    doc.add_paragraph(
        f"Comparing Window 1 (job {w1_job_id}, 03:33Z) and Window 2 (job {job_id}, 07:19Z, elapsed +3.763 hours) "
        "across the identical K = 10 disjoint T5 placements (50 transmons) provides a direct hardware demonstration "
        "of real-time intra-calibration drift detection by OSIRIS-ECHO. In Window 1, pid = 1 ([5, 6, 7, 8, 17]) was "
        "correctly ranked #1 by both backend calibration and ECHO_PROBE_16CZ (IDENTITY_CTRL_0CZ = 0.962240, "
        "ECHO_PROBE_16CZ = 0.815104, top-1 regret = 0.000000). Over the subsequent 3.763 hours, hub transmon Q7 of "
        "pid = 1 suffered an unflagged +31.12% (0.311198) |0>->|1> readout/SPAM excitation excursion (bitmask 4), "
        "collapsing pid = 1 IDENTITY_CTRL_0CZ to 0.677083 and ECHO_PROBE_16CZ to 0.562500. Because published "
        "calibration properties were not refreshed between windows, calibration_rank_score (-0.103649) kept pid = 1 "
        "at rank #1, degrading to Spearman rho_cal = +0.454545 with 0.361002 top-1 selection regret (0.811523 - "
        "0.450521). Live ECHO_PROBE_16CZ detected the Q7 degradation in situ (demoting pid = 1 to rank #8), switched "
        "#1 selection to pid = 0 ([21, 22, 23, 16, 24], ECHO_PROBE_16CZ = 0.858073, heldout_composite = 0.811523, "
        "plaquette_2wl_composite = 0.823568), and achieved Spearman rho_echo = +0.948333 (Delta rho = +0.493787 over "
        "calibration) with 0.000000 top-1 selection regret across both windows (2-window mean rho_echo = +0.834772 "
        "vs rho_cal = +0.618182, Delta = +0.216591)."
    )

    doc.add_figure(
        fig3,
        f"Two-window OSIRIS-ECHO intra-calibration drift catch across K = 10 disjoint T5 placements (50 physical "
        f"transmons on ibm_fez, Window 1 {w1_job_id} -> Window 2 {job_id}, +3.763 h): (a) Window-2 placement profile "
        "contrasting stale calibration rank against live ECHO_PROBE_16CZ, 2-workload Plaquette composite, and "
        "4-workload Held-Out composite; (b) Hub Q7 +31.12% excitation diagnosis on pid = 1 and per-workload Spearman "
        "rank correlations showing rho_echo = +0.948333 vs rho_cal = +0.454545 (Delta rho = +0.493787) and 0.000000 "
        "top-1 selection regret.",
        max_height=256.0,
    )

    cal_scores = [float(p["calibration_rank_score"]) for p in placements]
    echo_scores = [float(p["echo_probe_16cz"]) for p in placements]
    cal_ranks = _competition_ranks_desc(cal_scores)
    echo_ranks = _competition_ranks_desc(echo_scores)

    t3_rows = []
    for idx, p in enumerate(placements):
        pid = int(p["placement_id"])
        q_str = "[" + ",".join(str(q) for q in p["physical_qubits"]) + "]"
        t3_rows.append(
            [
                f"pid={pid} {q_str}",
                f"{p['calibration_rank_score']:.4f} (#{cal_ranks[idx]})",
                f"{p['identity_ctrl_0cz']:.4f}",
                f"{p['echo_probe_16cz']:.4f} (#{echo_ranks[idx]})",
                f"{p['spam_normalized_echo_ratio']:.4f}",
                f"{p['ub_center_pi4_19cz']:.4f}",
                f"{p['ub_hurwitz_pi2_19cz']:.4f}",
                f"{p['plaquette_2wl_composite']:.4f}",
                f"{p['clifford_30cz']:.4f}",
                f"{p['mirror_24cz']:.4f}",
                f"{p['heldout_composite']:.4f}",
            ]
        )
    doc.add_table(
        f"TABLE III. All K=10 Disjoint T5 Placements (50 Transmons) on ibm_fez (Window 2 Job {job_id}, 61,440 Shots).",
        [
            "Placement [Qubits]",
            "Cal Score (#)",
            "ID_0CZ",
            "ECHO_16CZ (#)",
            "NormEcho",
            "UB_Cnt",
            "UB_Hrw",
            "Plaq_2W",
            "Clf_30",
            "Mir_24",
            "Held_4W",
        ],
        t3_rows,
        font_size=6.0,
    )

    doc.add_references(
        [
            "[1] J. Kogut and L. Susskind, Hamiltonian formulation of Wilson's lattice gauge theories, Phys. Rev. D 11, 395 (1975).",
            "[2] D. Horn, M. Karliner, and E. Katznelson, Phase structure of the Z(2) and binary-tetrahedral lattice gauge theories, Phys. Lett. B 87, 113 (1979).",
            "[3] H. Lamm, S. Lawrence, and Y. Yamauchi (NuQS Collaboration), General methods for digital quantum simulation of gauge theories, Phys. Rev. D 100, 034518 (2019).",
            "[4] A. Barenco et al., Elementary gates for quantum computation, Phys. Rev. A 52, 3457 (1995).",
            "[5] G. Strang, On the construction and comparison of difference schemes, SIAM J. Numer. Anal. 5, 506 (1968).",
            "[6] D. P. Davis, Hardware-attested binary tetrahedral (2T) group inversion on ibm_fez (OSIRIS-BT2T-001-FEZ, job dc2ee3uq4uuc73a56o6g) and zero-ancilla 10Q semidirect multiplication / 7-irrep Peter-Weyl QFT (OSIRIS-BT2T-002), Agile Defense Systems Technical Reports (2026).",
            f"[7] D. P. Davis, Write-ahead preregistration and raw execution archive for OSIRIS-BT2T-003 and OSIRIS-BT2T-003-FEZ (ibm_fez jobs {w1_job_id} and {job_id}), Agile Defense Systems Technical Report (2026).",
        ]
    )

    return _with_pdf_creation_date(doc.to_pdf_bytes(), CREATION_DATE_ISO)


def generate_all() -> Dict[str, Any]:
    """Deterministically generate all macros, tables, figures, standalone preprint PDF, and SHA256SUMS.txt."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST_BT2T003_PATH.read_text(encoding="utf-8"))
    fez_results = json.loads(FEZ_RESULTS_PATH.read_text(encoding="utf-8"))
    fez_job = json.loads(FEZ_SUBMITTED_JOB_PATH.read_text(encoding="utf-8"))

    p1 = manifest["part1_gauss_law_and_peter_weyl_hamiltonian"]
    p2 = manifest["part2_multilink_holonomy_and_gauge_isomorphism"]
    p3 = manifest["part3_plaquette_regime_dynamics"]
    p4 = manifest["part4_trotter_suzuki_error_scaling"]
    p5 = manifest["part5_native_circuit_compilations"]
    gcs = p5["gate_count_summary"]
    dce = p5["direct_circuit_errors"]
    p5_prox = p5["proxy_compilations_5q"]
    p5_ml = p5["proxy_compilations_multilink"]
    p5_tk = p5["proxy_compilations_pytket"]

    fez_cz = fez_results["cz_counts_by_workload"]
    traj0 = fez_results["magnetic_plaquette_trajectories"]["pid_0"]
    traj1 = fez_results["magnetic_plaquette_trajectories"]["pid_1"]
    placements = fez_results["placement_results"]
    tw = fez_results["two_window_echo_summary"]
    drift = tw["pid1_drift_diagnosis"]
    pws = fez_results["per_workload_spearman"]
    routed_y = fez_results["t5_routed_target_y"]

    ub_ata = _find_proxy(p5_prox, circuit="ub_structured_5q", target="all_to_all")
    ub_t5 = _find_proxy(p5_prox, circuit="ub_structured_5q", target="heavy_hex_T5")
    ue_ata = _find_proxy(p5_prox, circuit="ue_structured_5q", target="all_to_all")
    ue_t5 = _find_proxy(p5_prox, circuit="ue_structured_5q", target="heavy_hex_T5")
    st_ata = _find_proxy(p5_prox, circuit="strang_structured_5q", target="all_to_all")
    st_t5 = _find_proxy(p5_prox, circuit="strang_structured_5q", target="heavy_hex_T5")
    qsd_ata = _find_proxy(p5_prox, circuit="strang_qsd_qft_5q", target="all_to_all")
    qsd_t5 = _find_proxy(p5_prox, circuit="strang_qsd_qft_5q", target="heavy_hex_T5")
    den_ata = _find_proxy(p5_prox, circuit="dense_unitary_strang_5q", target="all_to_all")
    den_t5 = _find_proxy(p5_prox, circuit="dense_unitary_strang_5q", target="heavy_hex_T5")

    ml20_ata = _find_proxy(p5_ml, circuit="strang_4link_inplace_20q", target="all_to_all")
    ml20_hh = _find_proxy(p5_ml, circuit="strang_4link_inplace_20q", target="heavy_hex_20Q")
    ml25_ata = _find_proxy(p5_ml, circuit="strang_4link_ancilla_25q", target="all_to_all")
    ml25_hh = _find_proxy(p5_ml, circuit="strang_4link_ancilla_25q", target="heavy_hex_25Q")

    tk_ub = _find_proxy(p5_tk, circuit="ub_structured_5q")
    tk_st = _find_proxy(p5_tk, circuit="strang_structured_5q")
    tk_den = _find_proxy(p5_tk, circuit="dense_unitary_exact_5q")

    cz_fez_struct = int(fez_cz["UB_CENTER_PI4_19CZ"])
    cz_fez_dense = int(fez_cz["UB_CENTER_PI4_GENERIC"])
    cz_fez_hurwitz = int(fez_cz["UB_HURWITZ_PI2_19CZ"])
    cz_fez_ratio = round(cz_fez_dense / float(cz_fez_struct) + 1e-12, 2)
    cz_fez_core_ratio = cz_fez_dense / float(gcs["cz_ub_structured_5q_heavy_hex_T5"])

    inv_p0_struct = float(traj0["ub_center_pi4_structured_19cz_inversion"])
    inv_p0_dense = float(traj0["ub_center_pi4_generic_dense_inversion"])
    inv_p0_adv = float(traj0["structured_minus_generic_pi4_advantage"])
    inv_p0_ratio = inv_p0_struct / inv_p0_dense

    inv_p1_struct = float(traj1["ub_center_pi4_structured_19cz_inversion"])
    inv_p1_dense = float(traj1["ub_center_pi4_generic_dense_inversion"])
    inv_p1_adv = float(traj1["structured_minus_generic_pi4_advantage"])
    inv_p1_ratio = inv_p1_struct / inv_p1_dense

    hh20_ratio = int(gcs["cz_strang_4link_inplace_20q_heavy_hex_20Q"]) / float(
        gcs["cz_strang_structured_5q_heavy_hex_T5"]
    )

    checks_total = len(manifest["checks"])
    checks_passed = sum(1 for v in manifest["checks"].values() if v)

    macros: List[str] = [
        "% Auto-generated by build_paper_assets.py -- DO NOT EDIT BY HAND",
        f"\\newcommand{{\\BTPlaqSimExperimentId}}{{\\texttt{{{manifest['experiment_id']}}}}}",
        f"\\newcommand{{\\BTPlaqSimExecTier}}{{\\texttt{{{manifest['execution_tier']}}}}}",
        "\\newcommand{\\BTPlaqSimExactTier}{\\texttt{EXACT\\_STATEVECTOR}}",
        "\\newcommand{\\BTPlaqProxyTier}{\\texttt{PROXY}}",
        "\\newcommand{\\BTPlaqSimJobId}{\\texttt{null}}",
        f"\\newcommand{{\\BTPlaqSimQpuSec}}{{{manifest['qpu_seconds']}}}",
        f"\\newcommand{{\\BTPlaqSimSeed}}{{{manifest['seed']}}}",
        f"\\newcommand{{\\BTPlaqEncVer}}{{\\texttt{{{manifest['encoding_version']}}}}}",
        f"\\newcommand{{\\BTPlaqEncShaPrefix}}{{\\texttt{{{manifest['encoding_sha256'][:16]}}}}}",
        f"\\newcommand{{\\BTPlaqEncShaFull}}{{\\texttt{{{manifest['encoding_sha256']}}}}}",
        f"\\newcommand{{\\BTPlaqSimCanonicalShaPrefix}}{{\\texttt{{{manifest['rfc8785_canonical_sha256'][:16]}}}}}",
        f"\\newcommand{{\\BTPlaqSimCanonicalShaFull}}{{\\texttt{{{manifest['rfc8785_canonical_sha256']}}}}}",
        f"\\newcommand{{\\BTPlaqChecksPassed}}{{{checks_passed}}}",
        f"\\newcommand{{\\BTPlaqChecksTotal}}{{{checks_total}}}",
        "\\newcommand{\\BTPlaqGroupOrder}{24}",
        "\\newcommand{\\BTPlaqSingleLinkQubits}{5}",
        "\\newcommand{\\BTPlaqSingleLinkStates}{32}",
        "\\newcommand{\\BTPlaqUnusedStates}{8}",
        "\\newcommand{\\BTPlaqFourLinkGroupConfigs}{331,776}",
        "\\newcommand{\\BTPlaqFourLinkGroupConfigsRaw}{331776}",
        "\\newcommand{\\BTPlaqTwoLinkGroupTriples}{13,824}",
        "\\newcommand{\\BTPlaqFourLinkInplaceQubits}{20}",
        "\\newcommand{\\BTPlaqFourLinkInplaceStates}{1,048,576}",
        "\\newcommand{\\BTPlaqFourLinkAncillaQubits}{25}",
        "\\newcommand{\\BTPlaqFourLinkAncillaStates}{33,554,432}",
        "\\newcommand{\\BTPlaqQubitReductionInplace}{4.0\\times}",
        "\\newcommand{\\BTPlaqQubitReductionAncilla}{5.0\\times}",
        f"\\newcommand{{\\BTPlaqPhysGaussSubspaceDim}}{{{p2['physical_gauss_invariant_subspace_dim']}}}",
        "\\newcommand{\\BTPlaqRandomGaugeConfigsChecked}{20}",
        f"\\newcommand{{\\BTPlaqMaxGaugeDiscrepancy}}{{{_fmt_sci_tex(float(p2['four_link_vs_5q_random_superposition_max_l2_error']))}}}",
        f"\\newcommand{{\\BTPlaqMaxGaugeDiscrepancyAscii}}{{{_fmt_sci_ascii(float(p2['four_link_vs_5q_random_superposition_max_l2_error']))}}}",
        f"\\newcommand{{\\BTPlaqFourLinkBasisMaxErr}}{{{_fmt_sci_tex(float(p2['four_link_vs_5q_basis_max_l2_error']))}}}",
        f"\\newcommand{{\\BTPlaqTwoLinkGaugeFixedErr}}{{{_fmt_sci_tex(float(p2['two_link_vs_5q_gauge_fixed_l2_error']))}}}",
        f"\\newcommand{{\\BTPlaqLaplacianIntertwineErr}}{{{_fmt_sci_tex(float(p2['max_link_laplacian_intertwining_l2_error']))}}}",
        f"\\newcommand{{\\BTPlaqIsometryGramErr}}{{{_fmt_sci_tex(float(p2['isometry_7x7_gram_max_abs_error']))}}}",
        "\\newcommand{\\BTPlaqFourVertexGaussErr}{0.0}",
        f"\\newcommand{{\\BTPlaqConjugacyClassCount}}{{{p1['conjugacy_class_count']}}}",
        f"\\newcommand{{\\BTPlaqCayleyOrder}}{{{p1['s8_generated_order']}}}",
        f"\\newcommand{{\\BTPlaqCayleyDiameter}}{{{p1['s8_generates_2t_diameter']}}}",
        "\\newcommand{\\BTPlaqCayleyDegree}{8}",
        f"\\newcommand{{\\BTPlaqPeterWeylCasimirDiagErr}}{{{_fmt_sci_tex(float(p1['peter_weyl_casimir_diagonalization_2norm_error']))}}}",
        f"\\newcommand{{\\BTPlaqPeterWeylOffDiagErr}}{{{_fmt_sci_tex(float(p1['peter_weyl_off_diagonal_max_abs']))}}}",
        f"\\newcommand{{\\BTPlaqUeQftExpmErr}}{{{_fmt_sci_tex(float(p1['ue_pw_qft_vs_expm_2norm_error']))}}}",
        "\\newcommand{\\BTPlaqUbMatrixExpmErr}{0.0}",
        "\\newcommand{\\BTPlaqUnusedCouplingNorm}{0.0}",
        "\\newcommand{\\BTPlaqUnusedPropagatorErr}{0.0}",
        "\\newcommand{\\BTPlaqChiClassOne}{+2.0}",
        "\\newcommand{\\BTPlaqChiClassTwo}{-2.0}",
        "\\newcommand{\\BTPlaqChiClassThree}{0.0}",
        "\\newcommand{\\BTPlaqChiClassFour}{+1.0}",
        "\\newcommand{\\BTPlaqChiClassFive}{+1.0}",
        "\\newcommand{\\BTPlaqChiClassSix}{-1.0}",
        "\\newcommand{\\BTPlaqChiClassSeven}{-1.0}",
        "\\newcommand{\\BTPlaqVbClassOne}{0.0}",
        "\\newcommand{\\BTPlaqVbClassTwo}{2.0}",
        "\\newcommand{\\BTPlaqVbClassThree}{1.0}",
        "\\newcommand{\\BTPlaqVbClassFour}{0.5}",
        "\\newcommand{\\BTPlaqVbClassFive}{0.5}",
        "\\newcommand{\\BTPlaqVbClassSix}{1.5}",
        "\\newcommand{\\BTPlaqVbClassSeven}{1.5}",
        "\\newcommand{\\BTPlaqHbClassOne}{-2.0}",
        "\\newcommand{\\BTPlaqHbClassTwo}{+2.0}",
        "\\newcommand{\\BTPlaqHbClassThree}{0.0}",
        "\\newcommand{\\BTPlaqHbClassFour}{-1.0}",
        "\\newcommand{\\BTPlaqHbClassFive}{-1.0}",
        "\\newcommand{\\BTPlaqHbClassSix}{+1.0}",
        "\\newcommand{\\BTPlaqHbClassSeven}{+1.0}",
        "\\newcommand{\\BTPlaqCasimirChiZero}{0.000}",
        "\\newcommand{\\BTPlaqCasimirChiOne}{2.250}",
        "\\newcommand{\\BTPlaqCasimirChiTwo}{2.250}",
        "\\newcommand{\\BTPlaqCasimirRhoTwoZero}{0.750}",
        "\\newcommand{\\BTPlaqCasimirRhoTwoOne}{1.875}",
        "\\newcommand{\\BTPlaqCasimirRhoTwoTwo}{1.875}",
        "\\newcommand{\\BTPlaqCasimirRhoThree}{1.500}",
        "\\newcommand{\\BTPlaqKinDimOneD}{0}",
        "\\newcommand{\\BTPlaqKinDimTwoD}{2}",
        "\\newcommand{\\BTPlaqKinDimThreeD}{6}",
        f"\\newcommand{{\\BTPlaqRegimeSteps}}{{{p3['n_strang_steps']}}}",
        f"\\newcommand{{\\BTPlaqRegimeDt}}{{{p3['dt']:.6f}}}",
        f"\\newcommand{{\\BTPlaqRegimeTFinal}}{{{p3['t_final']:.1f}}}",
        f"\\newcommand{{\\BTPlaqRegimeMaxStateErr}}{{{_fmt_sci_tex(float(p3['max_strang_vs_exact_state_l2_error']))}}}",
        f"\\newcommand{{\\BTPlaqRegimeMaxGaussDrift}}{{{_fmt_sci_tex(float(p3['max_gauss_conjugation_state_drift']))}}}",
        "\\newcommand{\\BTPlaqRegimeMaxUnusedLeak}{0.0}",
        f"\\newcommand{{\\BTPlaqWilsonWeakVacExact}}{{{p3['regimes']['g_ym_sq_0.5']['electric_vacuum_chi0']['exact']['wilson_loop']:.6f}}}",
        f"\\newcommand{{\\BTPlaqWilsonWeakVacStrang}}{{{p3['regimes']['g_ym_sq_0.5']['electric_vacuum_chi0']['strang']['wilson_loop']:.6f}}}",
        f"\\newcommand{{\\BTPlaqWilsonIntVacExact}}{{{p3['regimes']['g_ym_sq_1.0']['electric_vacuum_chi0']['exact']['wilson_loop']:.6f}}}",
        f"\\newcommand{{\\BTPlaqWilsonIntVacStrang}}{{{p3['regimes']['g_ym_sq_1.0']['electric_vacuum_chi0']['strang']['wilson_loop']:.6f}}}",
        f"\\newcommand{{\\BTPlaqWilsonStrongVacExact}}{{{p3['regimes']['g_ym_sq_2.0']['electric_vacuum_chi0']['exact']['wilson_loop']:.6f}}}",
        f"\\newcommand{{\\BTPlaqWilsonStrongVacStrang}}{{{p3['regimes']['g_ym_sq_2.0']['electric_vacuum_chi0']['strang']['wilson_loop']:.6f}}}",
        f"\\newcommand{{\\BTPlaqWilsonIntMagExact}}{{{p3['regimes']['g_ym_sq_1.0']['magnetic_identity_e']['exact']['wilson_loop']:.6f}}}",
        f"\\newcommand{{\\BTPlaqWilsonIntMagStrang}}{{{p3['regimes']['g_ym_sq_1.0']['magnetic_identity_e']['strang']['wilson_loop']:.6f}}}",
        f"\\newcommand{{\\BTPlaqLieSlopeGOne}}{{{p4['by_coupling']['g_ym_sq_1.0']['lie_loglog_slope']:.6f}}}",
        f"\\newcommand{{\\BTPlaqStrangSlopeGOne}}{{{p4['by_coupling']['g_ym_sq_1.0']['strang_loglog_slope']:.6f}}}",
        f"\\newcommand{{\\BTPlaqLieSlopeGHalf}}{{{p4['by_coupling']['g_ym_sq_0.5']['lie_loglog_slope']:.6f}}}",
        f"\\newcommand{{\\BTPlaqStrangSlopeGHalf}}{{{p4['by_coupling']['g_ym_sq_0.5']['strang_loglog_slope']:.6f}}}",
        f"\\newcommand{{\\BTPlaqLieSlopeGTwo}}{{{p4['by_coupling']['g_ym_sq_2.0']['lie_loglog_slope']:.6f}}}",
        f"\\newcommand{{\\BTPlaqStrangSlopeGTwo}}{{{p4['by_coupling']['g_ym_sq_2.0']['strang_loglog_slope']:.6f}}}",
        f"\\newcommand{{\\BTPlaqMaxFineStrangErr}}{{{_fmt_sci_tex(float(p4['max_fine_step_strang_error_n2048']))}}}",
        f"\\newcommand{{\\BTPlaqMaxModStrangErr}}{{{_fmt_sci_tex(float(p4['max_moderate_step_strang_error_n64']))}}}",
    ]

    step_word_map = {
        8: "Eight",
        16: "Sixteen",
        32: "ThirtyTwo",
        64: "SixtyFour",
        128: "OneTwoEight",
        256: "TwoFiveSix",
        512: "FiveOneTwo",
        1024: "OneZeroTwoFour",
        2048: "TwoZeroFourEight",
    }
    for row in p4["by_coupling"]["g_ym_sq_1.0"]["series"]:
        ns = int(row["n_steps"])
        sw = step_word_map[ns]
        macros.append(f"\\newcommand{{\\BTPlaqDtStep{sw}}}{{{float(row['dt']):.6f}}}")
        macros.append(
            f"\\newcommand{{\\BTPlaqLieErrGOneStep{sw}}}{{{_fmt_sci_tex(float(row['lie_2norm_error']))}}}"
        )
        macros.append(
            f"\\newcommand{{\\BTPlaqStrangErrGOneStep{sw}}}{{{_fmt_sci_tex(float(row['strang_2norm_error']))}}}"
        )

    macros.extend(
        [
            f"\\newcommand{{\\BTPlaqCompileDt}}{{{p5['dt']:.1f}}}",
            f"\\newcommand{{\\BTPlaqCompileGymSq}}{{{p5['g_ym_sq']:.1f}}}",
            f"\\newcommand{{\\BTPlaqUbAtaCz}}{{{gcs['cz_ub_structured_5q_all_to_all']}}}",
            f"\\newcommand{{\\BTPlaqUbAtaDepth}}{{{ub_ata['depth']}}}",
            f"\\newcommand{{\\BTPlaqUbAtaSize}}{{{ub_ata['size']}}}",
            f"\\newcommand{{\\BTPlaqUbTfCoreCz}}{{{gcs['cz_ub_structured_5q_heavy_hex_T5']}}}",
            f"\\newcommand{{\\BTPlaqUbTfCoreDepth}}{{{ub_t5['depth']}}}",
            f"\\newcommand{{\\BTPlaqUbTfCoreSize}}{{{ub_t5['size']}}}",
            f"\\newcommand{{\\BTPlaqUbTkZz}}{{{gcs['zzphase_ub_structured_5q']}}}",
            f"\\newcommand{{\\BTPlaqUbTkDepth}}{{{tk_ub['depth']}}}",
            f"\\newcommand{{\\BTPlaqUbDirectErr}}{{{_fmt_sci_tex(float(dce['ub_5q_2norm_error']))}}}",
            f"\\newcommand{{\\BTPlaqUeAtaCz}}{{{gcs['cz_ue_structured_5q_all_to_all']}}}",
            f"\\newcommand{{\\BTPlaqUeAtaDepth}}{{{ue_ata['depth']}}}",
            f"\\newcommand{{\\BTPlaqUeAtaSize}}{{{ue_ata['size']}}}",
            f"\\newcommand{{\\BTPlaqUeTfCz}}{{{gcs['cz_ue_structured_5q_heavy_hex_T5']}}}",
            f"\\newcommand{{\\BTPlaqUeTfDepth}}{{{ue_t5['depth']}}}",
            f"\\newcommand{{\\BTPlaqUeTfSize}}{{{ue_t5['size']}}}",
            f"\\newcommand{{\\BTPlaqUeDirectErr}}{{{_fmt_sci_tex(float(dce['ue_structured_5q_2norm_error']))}}}",
            f"\\newcommand{{\\BTPlaqStrangAtaCz}}{{{gcs['cz_strang_structured_5q_all_to_all']}}}",
            f"\\newcommand{{\\BTPlaqStrangAtaDepth}}{{{st_ata['depth']}}}",
            f"\\newcommand{{\\BTPlaqStrangAtaSize}}{{{st_ata['size']}}}",
            f"\\newcommand{{\\BTPlaqStrangTfCz}}{{{gcs['cz_strang_structured_5q_heavy_hex_T5']}}}",
            f"\\newcommand{{\\BTPlaqStrangTfDepth}}{{{st_t5['depth']:,}}}",
            f"\\newcommand{{\\BTPlaqStrangTfSize}}{{{st_t5['size']:,}}}",
            f"\\newcommand{{\\BTPlaqStrangTkZz}}{{{gcs['zzphase_strang_structured_5q']}}}",
            f"\\newcommand{{\\BTPlaqStrangTkDepth}}{{{tk_st['depth']}}}",
            f"\\newcommand{{\\BTPlaqStrangDirectErr}}{{{_fmt_sci_tex(float(dce['strang_structured_5q_2norm_error']))}}}",
            f"\\newcommand{{\\BTPlaqDenseFiveQAtaCz}}{{{gcs['cz_dense_unitary_5q_all_to_all']}}}",
            f"\\newcommand{{\\BTPlaqDenseFiveQAtaDepth}}{{{den_ata['depth']:,}}}",
            f"\\newcommand{{\\BTPlaqDenseFiveQTfCz}}{{{gcs['cz_dense_unitary_5q_heavy_hex_T5']}}}",
            f"\\newcommand{{\\BTPlaqDenseFiveQTfDepth}}{{{den_t5['depth']:,}}}",
            f"\\newcommand{{\\BTPlaqDenseFiveQTkZz}}{{{gcs['zzphase_dense_unitary_5q']}}}",
            f"\\newcommand{{\\BTPlaqDenseFiveQTkDepth}}{{{tk_den['depth']:,}}}",
            f"\\newcommand{{\\BTPlaqDenseVsStructAtaRatio}}{{{gcs['cz_compression_dense_vs_structured_all_to_all']:.2f}\\times}}",
            f"\\newcommand{{\\BTPlaqDenseVsStructTfRatio}}{{{gcs['cz_compression_dense_vs_structured_heavy_hex_T5']:.2f}\\times}}",
            f"\\newcommand{{\\BTPlaqQsdQftAtaCz}}{{{gcs['cz_strang_qsd_qft_5q_all_to_all']}}}",
            f"\\newcommand{{\\BTPlaqQsdQftAtaDepth}}{{{qsd_ata['depth']:,}}}",
            f"\\newcommand{{\\BTPlaqQsdQftTfCz}}{{{gcs['cz_strang_qsd_qft_5q_heavy_hex_T5']:,}}}",
            f"\\newcommand{{\\BTPlaqQsdQftTfDepth}}{{{qsd_t5['depth']:,}}}",
            f"\\newcommand{{\\BTPlaqFourLinkInplaceAtaCz}}{{{gcs['cz_strang_4link_inplace_20q_all_to_all']:,}}}",
            f"\\newcommand{{\\BTPlaqFourLinkInplaceAtaDepth}}{{{ml20_ata['depth']:,}}}",
            f"\\newcommand{{\\BTPlaqFourLinkInplaceHhCz}}{{{gcs['cz_strang_4link_inplace_20q_heavy_hex_20Q']:,}}}",
            f"\\newcommand{{\\BTPlaqFourLinkInplaceHhDepth}}{{{ml20_hh['depth']:,}}}",
            f"\\newcommand{{\\BTPlaqFourLinkAncillaAtaCz}}{{{gcs['cz_strang_4link_ancilla_25q_all_to_all']:,}}}",
            f"\\newcommand{{\\BTPlaqFourLinkAncillaAtaDepth}}{{{ml25_ata['depth']:,}}}",
            f"\\newcommand{{\\BTPlaqFourLinkAncillaHhCz}}{{{gcs['cz_strang_4link_ancilla_25q_heavy_hex_25Q']:,}}}",
            f"\\newcommand{{\\BTPlaqFourLinkAncillaHhDepth}}{{{ml25_hh['depth']:,}}}",
            f"\\newcommand{{\\BTPlaqCompTwentyQAtaRatio}}{{{gcs['cz_compression_unfixed_20q_vs_5q_all_to_all']:.2f}\\times}}",
            f"\\newcommand{{\\BTPlaqCompTwentyQTaOneDec}}{{{gcs['cz_compression_unfixed_20q_vs_5q_all_to_all']:.1f}\\times}}",
            f"\\newcommand{{\\BTPlaqCompTwentyQHhRatio}}{{{hh20_ratio:.2f}\\times}}",
            f"\\newcommand{{\\BTPlaqCompTwentyFiveQAtaRatio}}{{{gcs['cz_compression_unfixed_25q_vs_5q_all_to_all']:.2f}\\times}}",
            f"\\newcommand{{\\BTPlaqCompTwentyFiveQHhRatio}}{{{gcs['cz_compression_unfixed_25q_vs_5q_heavy_hex']:.2f}\\times}}",
            f"\\newcommand{{\\BTPlaqCompTwentyFiveQHhOneDec}}{{{gcs['cz_compression_unfixed_25q_vs_5q_heavy_hex']:.1f}\\times}}",
            f"\\newcommand{{\\BTPlaqFezProtocolId}}{{\\texttt{{{fez_results['protocol_id']}}}}}",
            f"\\newcommand{{\\BTPlaqFezExecTier}}{{\\texttt{{{fez_results['execution_tier']}}}}}",
            "\\newcommand{\\BTPlaqFezBackend}{\\texttt{ibm\\_fez}}",
            f"\\newcommand{{\\BTPlaqFezJobId}}{{\\texttt{{{fez_results['provider_job_id']}}}}}",
            f"\\newcommand{{\\BTPlaqFezWindowOneJobId}}{{\\texttt{{{tw['window1_job_id']}}}}}",
            f"\\newcommand{{\\BTPlaqFezQpuSec}}{{{fez_job['provider_metadata']['usage']['qpu_charge_time_seconds']}}}",
            f"\\newcommand{{\\BTPlaqFezTotalPubs}}{{{fez_job['total_pubs']}}}",
            f"\\newcommand{{\\BTPlaqFezShotsPerPub}}{{{fez_job['shots_per_pub']}}}",
            f"\\newcommand{{\\BTPlaqFezTotalShots}}{{{fez_job['total_shots']:,}}}",
            f"\\newcommand{{\\BTPlaqFezTotalShotsRaw}}{{{fez_job['total_shots']}}}",
            "\\newcommand{\\BTPlaqFezBlocks}{3}",
            "\\newcommand{\\BTPlaqFezPubsPerBlock}{80}",
            "\\newcommand{\\BTPlaqFezNumPlacements}{10}",
            "\\newcommand{\\BTPlaqFezTotalTransmons}{50}",
            f"\\newcommand{{\\BTPlaqFezWriteAheadCommit}}{{\\texttt{{{fez_job['write_ahead_commit_sha'][:8]}}}}}",
            f"\\newcommand{{\\BTPlaqFezWriteAheadCommitFull}}{{\\texttt{{{fez_job['write_ahead_commit_sha']}}}}}",
            f"\\newcommand{{\\BTPlaqFezQpyShaPrefix}}{{\\texttt{{{fez_job['qpy_sha256'][:16]}}}}}",
            f"\\newcommand{{\\BTPlaqFezQpyShaFull}}{{\\texttt{{{fez_job['qpy_sha256']}}}}}",
            f"\\newcommand{{\\BTPlaqFezRawCountsShaPrefix}}{{\\texttt{{{fez_results['raw_counts_gz_sha256'][:16]}}}}}",
            f"\\newcommand{{\\BTPlaqFezRawCountsShaFull}}{{\\texttt{{{fez_results['raw_counts_gz_sha256']}}}}}",
            f"\\newcommand{{\\BTPlaqFezPreregShaPrefix}}{{\\texttt{{{fez_job['preregistration_md_sha256'][:16]}}}}}",
            f"\\newcommand{{\\BTPlaqFezCzIdentity}}{{{fez_cz['IDENTITY_CTRL_0CZ']}}}",
            f"\\newcommand{{\\BTPlaqFezCzEchoProbe}}{{{fez_cz['ECHO_PROBE_16CZ']}}}",
            f"\\newcommand{{\\BTPlaqFezCzCenterStruct}}{{{cz_fez_struct}}}",
            f"\\newcommand{{\\BTPlaqFezCzCenterGeneric}}{{{cz_fez_dense:,}}}",
            f"\\newcommand{{\\BTPlaqFezCzCenterGenericRaw}}{{{cz_fez_dense}}}",
            f"\\newcommand{{\\BTPlaqFezCzHurwitzStruct}}{{{cz_fez_hurwitz}}}",
            f"\\newcommand{{\\BTPlaqFezCzImagNull}}{{{fez_cz['UB_IMAGQ8_PI4_19CZ']}}}",
            f"\\newcommand{{\\BTPlaqFezCzUnusedNull}}{{{fez_cz['UB_UNUSED8_PI4_19CZ']}}}",
            f"\\newcommand{{\\BTPlaqFezCzClifford}}{{{fez_cz['CLIFFORD_30CZ']}}}",
            f"\\newcommand{{\\BTPlaqFezCzMirror}}{{{fez_cz['MIRROR_24CZ']}}}",
            f"\\newcommand{{\\BTPlaqFezCzReductionRatio}}{{{cz_fez_ratio:.2f}\\times}}",
            f"\\newcommand{{\\BTPlaqFezCoreCzReductionRatio}}{{{cz_fez_core_ratio:.2f}\\times}}",
            f"\\newcommand{{\\BTPlaqRoutedYCenterInv}}{{{routed_y['UB_CENTER_PI4_19CZ']}}}",
            "\\newcommand{\\BTPlaqRoutedBitsCenterInv}{\\texttt{00010}}",
            f"\\newcommand{{\\BTPlaqRoutedYHurwitzInv}}{{{routed_y['UB_HURWITZ_PI2_19CZ']}}}",
            "\\newcommand{\\BTPlaqRoutedBitsHurwitzInv}{\\texttt{00100}}",
            f"\\newcommand{{\\BTPlaqRoutedYImagNull}}{{{routed_y['UB_IMAGQ8_PI4_19CZ']}}}",
            "\\newcommand{\\BTPlaqRoutedBitsImagNull}{\\texttt{00001}}",
            f"\\newcommand{{\\BTPlaqRoutedYUnusedNull}}{{{routed_y['UB_UNUSED8_PI4_19CZ']}}}",
            "\\newcommand{\\BTPlaqRoutedBitsUnusedNull}{\\texttt{01100}}",
            f"\\newcommand{{\\BTPlaqRoutedYClifford}}{{{routed_y['CLIFFORD_30CZ']}}}",
            f"\\newcommand{{\\BTPlaqRoutedYMirror}}{{{routed_y['MIRROR_24CZ']}}}",
            "\\newcommand{\\BTPlaqIdealOne}{1.000000}",
            "\\newcommand{\\BTPlaqIdealHalf}{0.500000}",
            "\\newcommand{\\BTPlaqIdealZero}{0.000000}",
            f"\\newcommand{{\\BTPlaqIdealCosSqPiEight}}{{{traj0['hurwitz_chi2_trajectory'][1]['ideal_p_phi_plus']:.6f}}}",
            f"\\newcommand{{\\BTPlaqIdealCosSqThreePiEight}}{{{traj0['hurwitz_chi2_trajectory'][3]['ideal_p_phi_plus']:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidZeroCenterTZero}}{{{traj0['center_chi4_trajectory'][0]['measured_p_plus']:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidZeroCenterPiEight}}{{{traj0['center_chi4_trajectory'][1]['measured_p_plus']:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidZeroCenterPiFourPlus}}{{{traj0['center_chi4_trajectory'][2]['measured_p_plus']:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidZeroCenterPiFourMinus}}{{{inv_p0_struct:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidZeroCenterPiFourMinusPct}}{{{100.0 * inv_p0_struct:.2f}\\%}}",
            f"\\newcommand{{\\BTPlaqPidZeroCenterPiTwo}}{{{traj0['center_chi4_trajectory'][3]['measured_p_plus']:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidZeroPearsonCenter}}{{{traj0['pearson_r_center']:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidZeroCenterGenericMinus}}{{{inv_p0_dense:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidZeroCenterStructAdv}}{{{inv_p0_adv:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidZeroCenterStructAdvPp}}{{{100.0 * inv_p0_adv:.2f}}}",
            f"\\newcommand{{\\BTPlaqPidZeroCenterStructFidelityRatio}}{{{inv_p0_ratio:.2f}\\times}}",
            f"\\newcommand{{\\BTPlaqPidZeroHurwitzTZero}}{{{traj0['hurwitz_chi2_trajectory'][0]['measured_p_phi_plus']:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidZeroHurwitzPiEight}}{{{traj0['hurwitz_chi2_trajectory'][1]['measured_p_phi_plus']:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidZeroHurwitzPiFour}}{{{traj0['hurwitz_chi2_trajectory'][2]['measured_p_phi_plus']:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidZeroHurwitzThreePiEight}}{{{traj0['hurwitz_chi2_trajectory'][3]['measured_p_phi_plus']:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidZeroHurwitzPiTwoPlus}}{{{traj0['hurwitz_chi2_trajectory'][4]['measured_p_phi_plus']:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidZeroHurwitzPiTwoMinus}}{{{traj0['hurwitz_chi2_trajectory'][4]['measured_p_phi_minus_inverted']:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidZeroHurwitzPiTwoMinusPct}}{{{100.0 * float(traj0['hurwitz_chi2_trajectory'][4]['measured_p_phi_minus_inverted']):.2f}\\%}}",
            f"\\newcommand{{\\BTPlaqPidZeroPearsonHurwitz}}{{{traj0['pearson_r_hurwitz']:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidZeroImagNullSurv}}{{{traj0['imag_q8_chi0_null_pi4_survival']:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidZeroUnusedNullSurv}}{{{traj0['unused8_chi0_null_pi4_survival']:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidOneCenterTZero}}{{{traj1['center_chi4_trajectory'][0]['measured_p_plus']:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidOneCenterPiEight}}{{{traj1['center_chi4_trajectory'][1]['measured_p_plus']:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidOneCenterPiFourPlus}}{{{traj1['center_chi4_trajectory'][2]['measured_p_plus']:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidOneCenterPiFourMinus}}{{{inv_p1_struct:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidOneCenterPiTwo}}{{{traj1['center_chi4_trajectory'][3]['measured_p_plus']:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidOnePearsonCenter}}{{{traj1['pearson_r_center']:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidOneCenterGenericMinus}}{{{inv_p1_dense:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidOneCenterStructAdv}}{{{inv_p1_adv:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidOneCenterStructFidelityRatio}}{{{inv_p1_ratio:.2f}\\times}}",
            f"\\newcommand{{\\BTPlaqPidOneHurwitzTZero}}{{{traj1['hurwitz_chi2_trajectory'][0]['measured_p_phi_plus']:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidOneHurwitzPiEight}}{{{traj1['hurwitz_chi2_trajectory'][1]['measured_p_phi_plus']:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidOneHurwitzPiFour}}{{{traj1['hurwitz_chi2_trajectory'][2]['measured_p_phi_plus']:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidOneHurwitzThreePiEight}}{{{traj1['hurwitz_chi2_trajectory'][3]['measured_p_phi_plus']:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidOneHurwitzPiTwoPlus}}{{{traj1['hurwitz_chi2_trajectory'][4]['measured_p_phi_plus']:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidOneHurwitzPiTwoMinus}}{{{traj1['hurwitz_chi2_trajectory'][4]['measured_p_phi_minus_inverted']:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidOnePearsonHurwitz}}{{{traj1['pearson_r_hurwitz']:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidOneImagNullSurv}}{{{traj1['imag_q8_chi0_null_pi4_survival']:.6f}}}",
            f"\\newcommand{{\\BTPlaqPidOneUnusedNullSurv}}{{{traj1['unused8_chi0_null_pi4_survival']:.6f}}}",
            f"\\newcommand{{\\BTPlaqEchoNumWindows}}{{{tw['n_windows']}}}",
            f"\\newcommand{{\\BTPlaqEchoElapsedHours}}{{{tw['inter_window_elapsed_hours']:.3f}}}",
            f"\\newcommand{{\\BTPlaqDriftedHubQubit}}{{{drift['drifted_hub_qubit']}}}",
            f"\\newcommand{{\\BTPlaqWOnePidOneIdFid}}{{{drift['w1_identity_0cz_fidelity']:.6f}}}",
            f"\\newcommand{{\\BTPlaqWTwoPidOneIdFid}}{{{drift['w2_identity_0cz_fidelity']:.6f}}}",
            f"\\newcommand{{\\BTPlaqWOnePidOneEcho}}{{{drift['w1_echo_probe_16cz']:.6f}}}",
            f"\\newcommand{{\\BTPlaqWTwoPidOneEcho}}{{{drift['w2_echo_probe_16cz']:.6f}}}",
            f"\\newcommand{{\\BTPlaqQSevenExcitation}}{{{drift['w2_identity_0cz_q7_bitmask4_excitation']:.6f}}}",
            f"\\newcommand{{\\BTPlaqQSevenExcitationPct}}{{{100.0 * float(drift['w2_identity_0cz_q7_bitmask4_excitation']):.2f}\\%}}",
            f"\\newcommand{{\\BTPlaqWOneSpearmanCal}}{{{tw['window1_spearman_calibration']:.6f}}}",
            f"\\newcommand{{\\BTPlaqWOneSpearmanEcho}}{{{tw['window1_spearman_raw_echo']:.6f}}}",
            f"\\newcommand{{\\BTPlaqWOneSpearmanNormEcho}}{{{tw['window1_spearman_spam_normalized_echo']:.6f}}}",
            "\\newcommand{\\BTPlaqWOneRegretCal}{0.000000}",
            "\\newcommand{\\BTPlaqWOneRegretEcho}{0.000000}",
            f"\\newcommand{{\\BTPlaqWTwoSpearmanCal}}{{{tw['window2_spearman_calibration']:.6f}}}",
            f"\\newcommand{{\\BTPlaqWTwoSpearmanEcho}}{{{tw['window2_spearman_raw_echo']:.6f}}}",
            f"\\newcommand{{\\BTPlaqWTwoSpearmanNormEcho}}{{{tw['window2_spearman_spam_normalized_echo']:.6f}}}",
            f"\\newcommand{{\\BTPlaqWTwoDeltaSpearman}}{{{fez_results['window2_delta_raw_echo_minus_calibration']:.6f}}}",
            f"\\newcommand{{\\BTPlaqWTwoRegretCal}}{{{tw['window2_top1_calibration_regret']:.6f}}}",
            f"\\newcommand{{\\BTPlaqWTwoRegretCalPp}}{{{100.0 * float(tw['window2_top1_calibration_regret']):.2f}}}",
            "\\newcommand{\\BTPlaqWTwoRegretEcho}{0.000000}",
            f"\\newcommand{{\\BTPlaqWTwoRegretNormEcho}}{{{fez_results['window2_top1_spam_normalized_echo_regret']:.6f}}}",
            f"\\newcommand{{\\BTPlaqTwoWinMeanSpearmanCal}}{{{tw['mean_2window_spearman_calibration']:.6f}}}",
            f"\\newcommand{{\\BTPlaqTwoWinMeanSpearmanEcho}}{{{tw['mean_2window_spearman_raw_echo']:.6f}}}",
            f"\\newcommand{{\\BTPlaqTwoWinMeanSpearmanNormEcho}}{{{tw['mean_2window_spearman_spam_normalized_echo']:.6f}}}",
            f"\\newcommand{{\\BTPlaqTwoWinMeanDeltaSpearman}}{{{tw['mean_2window_delta_raw_echo_minus_cal']:.6f}}}",
            f"\\newcommand{{\\BTPlaqPooledSpearmanCalAll}}{{{tw['pooled_20obs_spearman_calibration_all_workloads']:.6f}}}",
            f"\\newcommand{{\\BTPlaqPooledSpearmanEchoAll}}{{{tw['pooled_20obs_spearman_raw_echo_all_workloads']:.6f}}}",
            f"\\newcommand{{\\BTPlaqPooledSpearmanCalClfMir}}{{{tw['pooled_20obs_spearman_calibration_clifford_mirror']:.6f}}}",
            f"\\newcommand{{\\BTPlaqPooledSpearmanEchoClfMir}}{{{tw['pooled_20obs_spearman_raw_echo_clifford_mirror']:.6f}}}",
            f"\\newcommand{{\\BTPlaqRhoCalClifford}}{{{pws['clifford_30cz']['rho_calibration']:.3f}}}",
            f"\\newcommand{{\\BTPlaqRhoEchoClifford}}{{{pws['clifford_30cz']['rho_echo']:.3f}}}",
            f"\\newcommand{{\\BTPlaqDeltaRhoClifford}}{{{pws['clifford_30cz']['delta_echo_minus_calibration']:.3f}}}",
            f"\\newcommand{{\\BTPlaqRhoCalMirror}}{{{pws['mirror_24cz']['rho_calibration']:.6f}}}",
            f"\\newcommand{{\\BTPlaqRhoEchoMirror}}{{{pws['mirror_24cz']['rho_echo']:.6f}}}",
            f"\\newcommand{{\\BTPlaqDeltaRhoMirror}}{{{pws['mirror_24cz']['delta_echo_minus_calibration']:.6f}}}",
            f"\\newcommand{{\\BTPlaqRhoCalClfMir}}{{{pws['clifford_mirror_2wl_composite']['rho_calibration']:.6f}}}",
            f"\\newcommand{{\\BTPlaqRhoEchoClfMir}}{{{pws['clifford_mirror_2wl_composite']['rho_echo']:.6f}}}",
            f"\\newcommand{{\\BTPlaqDeltaRhoClfMir}}{{{pws['clifford_mirror_2wl_composite']['delta_echo_minus_calibration']:.6f}}}",
            f"\\newcommand{{\\BTPlaqRhoCalCenter}}{{{pws['ub_center_pi4_19cz']['rho_calibration']:.6f}}}",
            f"\\newcommand{{\\BTPlaqRhoEchoCenter}}{{{pws['ub_center_pi4_19cz']['rho_echo']:.6f}}}",
            f"\\newcommand{{\\BTPlaqDeltaRhoCenter}}{{{pws['ub_center_pi4_19cz']['delta_echo_minus_calibration']:.6f}}}",
            f"\\newcommand{{\\BTPlaqRhoCalHurwitz}}{{{pws['ub_hurwitz_pi2_19cz']['rho_calibration']:.6f}}}",
            f"\\newcommand{{\\BTPlaqRhoEchoHurwitz}}{{{pws['ub_hurwitz_pi2_19cz']['rho_echo']:.6f}}}",
            f"\\newcommand{{\\BTPlaqDeltaRhoHurwitz}}{{{pws['ub_hurwitz_pi2_19cz']['delta_echo_minus_calibration']:.6f}}}",
            f"\\newcommand{{\\BTPlaqRhoCalPlaqTwoW}}{{{pws['plaquette_2wl_composite']['rho_calibration']:.6f}}}",
            f"\\newcommand{{\\BTPlaqRhoEchoPlaqTwoW}}{{{pws['plaquette_2wl_composite']['rho_echo']:.6f}}}",
            f"\\newcommand{{\\BTPlaqDeltaRhoPlaqTwoW}}{{{pws['plaquette_2wl_composite']['delta_echo_minus_calibration']:.6f}}}",
        ]
    )

    cal_scores = [float(p["calibration_rank_score"]) for p in placements]
    echo_scores = [float(p["echo_probe_16cz"]) for p in placements]
    cal_ranks = _competition_ranks_desc(cal_scores)
    echo_ranks = _competition_ranks_desc(echo_scores)

    for idx, p in enumerate(placements):
        pw = DIGIT_WORDS[int(p["placement_id"])]
        q_str = "[" + ", ".join(str(q) for q in p["physical_qubits"]) + "]"
        macros.extend(
            [
                f"\\newcommand{{\\BTPlaqPid{pw}Qubits}}{{\\texttt{{{q_str}}}}}",
                f"\\newcommand{{\\BTPlaqPid{pw}CalScore}}{{{p['calibration_rank_score']:.6f}}}",
                f"\\newcommand{{\\BTPlaqPid{pw}CalRank}}{{{cal_ranks[idx]}}}",
                f"\\newcommand{{\\BTPlaqPid{pw}IdZeroCz}}{{{p['identity_ctrl_0cz']:.6f}}}",
                f"\\newcommand{{\\BTPlaqPid{pw}EchoSixteenCz}}{{{p['echo_probe_16cz']:.6f}}}",
                f"\\newcommand{{\\BTPlaqPid{pw}EchoRank}}{{{echo_ranks[idx]}}}",
                f"\\newcommand{{\\BTPlaqPid{pw}NormEcho}}{{{p['spam_normalized_echo_ratio']:.6f}}}",
                f"\\newcommand{{\\BTPlaqPid{pw}UbCenter}}{{{p['ub_center_pi4_19cz']:.6f}}}",
                f"\\newcommand{{\\BTPlaqPid{pw}UbHurwitz}}{{{p['ub_hurwitz_pi2_19cz']:.6f}}}",
                f"\\newcommand{{\\BTPlaqPid{pw}PlaqComp}}{{{p['plaquette_2wl_composite']:.6f}}}",
                f"\\newcommand{{\\BTPlaqPid{pw}Clifford}}{{{p['clifford_30cz']:.6f}}}",
                f"\\newcommand{{\\BTPlaqPid{pw}Mirror}}{{{p['mirror_24cz']:.6f}}}",
                f"\\newcommand{{\\BTPlaqPid{pw}ClfMirComp}}{{{p['clifford_mirror_2wl_composite']:.6f}}}",
                f"\\newcommand{{\\BTPlaqPid{pw}HeldoutComp}}{{{p['heldout_composite']:.6f}}}",
            ]
        )

    (HERE / "generated_macros.tex").write_bytes(("\n".join(macros) + "\n").encode("utf-8"))

    t1_tex = "\n".join(
        [
            "% Auto-generated by build_paper_assets.py -- DO NOT EDIT BY HAND",
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Exact conjugacy classes $C_1,\\dots,C_7$ of the binary tetrahedral gauge group $2T \\subset SU(2)$ ($|2T| = \\BTPlaqGroupOrder$) under frozen five-qubit Hurwitz encoding \\BTPlaqEncVer, fundamental character $\\chi_{2D}(U) = \\operatorname{Re}\\operatorname{Tr}(\\rho_{2,0}(U))$, shifted magnetic plaquette potential $V_B(U) = \\tfrac{1}{2}(2 - \\operatorname{Re}\\chi_{2D}(U))$, magnetic Hamiltonian $H_B(U)$ ($g_{\\mathrm{YM}}^2 = 1$), and dual Peter--Weyl irreducible representations $\\rho$ with dimension $d_\\rho$, representation-dimension electric eigenvalue $d_\\rho(d_\\rho - 1)$, $24$-cell Cayley graph Laplacian eigenvalue $\\lambda_{\\mathrm{Cayley}}$, and continuum-calibrated $SU(2)$ quadratic Casimir eigenvalue $c_2(\\rho) = \\tfrac{3}{16}\\lambda_{\\mathrm{Cayley}}$ (tier \\BTPlaqSimExecTier\\ / \\BTPlaqSimExactTier, \\BTPlaqSimExperimentId).}",
            "\\label{tab:bt2t003_conjugacy_magnetic_potential}",
            "\\begin{ruledtabular}",
            "\\begin{tabular}{lccccclccc}",
            "Class $C_k$ & $|C_k|$ & \\BTPlaqEncVer\\ Indices (Coset $c$) & $\\chi_{2D}(U)$ & $V_B(U)$ & $H_B(U)$ & Dual Irrep $\\rho$ & $d_\\rho(d_\\rho-1)$ & $\\lambda_{\\mathrm{Cayley}}$ & $c_2(\\rho)$ \\\\",
            "\\hline",
            "$C_1 = \\{+\\mathbf{1}\\}$ & 1 & $0$ ($c=0$, $+\\mathbf{1}$) & $\\BTPlaqChiClassOne$ & $\\BTPlaqVbClassOne$ ($0$) & $\\BTPlaqHbClassOne$ & $\\chi_0$ ($A_0$, $d=1$) & \\BTPlaqKinDimOneD & 0 & $\\BTPlaqCasimirChiZero$ ($0$) \\\\",
            "$C_2 = \\{-\\mathbf{1}\\}$ & 1 & $1$ ($c=0$, $-\\mathbf{1}$) & $\\BTPlaqChiClassTwo$ & $\\BTPlaqVbClassTwo$ ($2$) & $\\BTPlaqHbClassTwo$ & $\\chi_1$ ($A_1$, $d=1$) & \\BTPlaqKinDimOneD & 12 & $\\BTPlaqCasimirChiOne$ ($9/4$) \\\\",
            "$C_3 = \\{\\pm\\mathbf{i},\\pm\\mathbf{j},\\pm\\mathbf{k}\\}$ & 6 & $2\\dots 7$ ($c=0$, $Q_8$) & $\\BTPlaqChiClassThree$ & $\\BTPlaqVbClassThree$ ($1$) & $\\BTPlaqHbClassThree$ & $\\chi_2$ ($A_2$, $d=1$) & \\BTPlaqKinDimOneD & 12 & $\\BTPlaqCasimirChiTwo$ ($9/4$) \\\\",
            "$C_4$ (Hurwitz $+$, $c=1$) & 4 & $8,11,13,14$ ($q_H \\omega$) & $\\BTPlaqChiClassFour$ & $\\BTPlaqVbClassFour$ ($1/2$) & $\\BTPlaqHbClassFour$ & $\\rho_{2,0}$ ($E_1$, $d=2$) & \\BTPlaqKinDimTwoD & 4 & $\\BTPlaqCasimirRhoTwoZero$ ($3/4$) \\\\",
            "$C_5$ (Hurwitz $+$, $c=2$) & 4 & $9,10,12,15$ ($q_H \\omega^2$) & $\\BTPlaqChiClassFive$ & $\\BTPlaqVbClassFive$ ($1/2$) & $\\BTPlaqHbClassFive$ & $\\rho_{2,1}$ ($E_2$, $d=2$) & \\BTPlaqKinDimTwoD & 10 & $\\BTPlaqCasimirRhoTwoOne$ ($15/8$) \\\\",
            "$C_6$ (Hurwitz $-$, $c=2$) & 4 & $16,19,21,22$ ($-q_H \\omega^2$) & $\\BTPlaqChiClassSix$ & $\\BTPlaqVbClassSix$ ($3/2$) & $\\BTPlaqHbClassSix$ & $\\rho_{2,2}$ ($E_3$, $d=2$) & \\BTPlaqKinDimTwoD & 10 & $\\BTPlaqCasimirRhoTwoTwo$ ($15/8$) \\\\",
            "$C_7$ (Hurwitz $-$, $c=1$) & 4 & $17,18,20,23$ ($-q_H \\omega$) & $\\BTPlaqChiClassSeven$ & $\\BTPlaqVbClassSeven$ ($3/2$) & $\\BTPlaqHbClassSeven$ & $\\rho_3$ ($T$, $d=3$) & \\BTPlaqKinDimThreeD & 8 & $\\BTPlaqCasimirRhoThree$ ($3/2$) \\\\",
            "\\hline",
            "Unused-$8$ Subspace & 8 & $24\\dots 31$ ($c=3$ padded) & $\\BTPlaqChiClassThree$ & Decoupled $I_8$ & $\\BTPlaqHbClassThree$ & Identity Sector $I_8$ & 0 & 0 & $\\BTPlaqCasimirChiZero$ ($0$) \\\\",
            "\\end{tabular}",
            "\\end{ruledtabular}",
            "\\end{table*}",
            "",
        ]
    )
    (TABLES_DIR / "table1_conjugacy_magnetic_potential.tex").write_bytes(t1_tex.encode("utf-8"))

    t2_tex = "\n".join(
        [
            "% Auto-generated by build_paper_assets.py -- DO NOT EDIT BY HAND",
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Native two-qubit gate compression across five-qubit maximal-tree gauge-fixed structured compilation, generic dense $32\\times 32$ QFT-conjugated synthesis, and unfixed four-link ($20$-qubit in-place and $25$-qubit holonomy-ancilla) Trotterization, together with first-order Lie--Trotter ($O(\\Delta t)$) versus second-order Strang splitting ($O(\\Delta t^2)$) spectral-norm error scaling across $N \\in \\{8,\\dots,2048\\}$ steps to $t_{\\mathrm{final}} = \\BTPlaqRegimeTFinal$ at $g_{\\mathrm{YM}}^2 = \\BTPlaqCompileGymSq$ (tier \\BTPlaqSimExecTier\\ / \\BTPlaqProxyTier, \\BTPlaqSimExperimentId).}",
            "\\label{tab:bt2t003_strang_scaling_and_gate_compression}",
            "\\begin{ruledtabular}",
            "\\begin{tabular}{lcccccc}",
            "Circuit / Trotter Step Architecture & Register Width & All-to-All \\texttt{CZ} (Depth) & Heavy-Hex \\texttt{CZ} (Depth) & \\texttt{pytket} \\texttt{ZZPhase} (Depth) & Unitary / Trotter $2$-Norm Error & Compression Ratio / Scaling \\\\",
            "\\hline",
            "$U_B(\\Delta t)$ 5Q Walsh--Hadamard Core & 5Q ($32$-D) & \\BTPlaqUbAtaCz\\ (\\BTPlaqUbAtaDepth) & \\BTPlaqUbTfCoreCz\\ (\\BTPlaqUbTfCoreDepth) & \\BTPlaqUbTkZz\\ (\\BTPlaqUbTkDepth) & $\\BTPlaqUbDirectErr$ & Core diagonal magnetic phase \\\\",
            "$U_B(\\pi/4)$ Center Arm + State Prep & 5Q (\\texttt{T5}) & --- & \\BTPlaqFezCzCenterStruct\\ (\\BTPlaqFezBackend) & --- & $\\BTPlaqUbDirectErr$ & \\BTPlaqFezCzReductionRatio\\ vs \\BTPlaqFezCzCenterGeneric\\ \\texttt{CZ} (\\BTPlaqFezCoreCzReductionRatio\\ core) \\\\",
            "$U_B(\\pi/2)$ Hurwitz Bell Arm + Prep & 5Q (\\texttt{T5}) & --- & \\BTPlaqFezCzHurwitzStruct\\ (\\BTPlaqFezBackend) & --- & $\\BTPlaqUbDirectErr$ & Half-frequency $\\Delta\\chi_{2D}=2$ arm \\\\",
            "$U_B(\\pi/4)$ Generic Dense QFT-Conj. & 5Q (\\texttt{T5}) & --- & \\BTPlaqFezCzCenterGeneric\\ (\\BTPlaqFezBackend) & --- & $\\le 10^{-13}$ & Baseline dense $32\\times 32$ \\texttt{UnitaryGate} \\\\",
            "$U_E(\\Delta t)$ 5Q Casimir Peter--Weyl & 5Q ($32$-D) & \\BTPlaqUeAtaCz\\ (\\BTPlaqUeAtaDepth) & \\BTPlaqUeTfCz\\ (\\BTPlaqUeTfDepth) & --- & $\\BTPlaqUeDirectErr$ & Non-abelian electric kinetic step \\\\",
            "$S_2(\\Delta t)$ 5Q Structured Strang & 5Q ($32$-D) & \\BTPlaqStrangAtaCz\\ (\\BTPlaqStrangAtaDepth) & \\BTPlaqStrangTfCz\\ (\\BTPlaqStrangTfDepth) & \\BTPlaqStrangTkZz\\ (\\BTPlaqStrangTkDepth) & $\\BTPlaqStrangDirectErr$ & $1\\times$ (gauge-fixed 5Q reference) \\\\",
            "$S_2(\\Delta t)$ 5Q Dense \\texttt{UnitaryGate} & 5Q ($32$-D) & \\BTPlaqDenseFiveQAtaCz\\ (\\BTPlaqDenseFiveQAtaDepth) & \\BTPlaqDenseFiveQTfCz\\ (\\BTPlaqDenseFiveQTfDepth) & \\BTPlaqDenseFiveQTkZz\\ (\\BTPlaqDenseFiveQTkDepth) & $\\le 10^{-13}$ & \\BTPlaqDenseVsStructAtaRatio\\ (All) / \\BTPlaqDenseVsStructTfRatio\\ (\\texttt{T5}) \\\\",
            "$S_2(\\Delta t)$ 5Q Canonical QSD-QFT & 5Q ($32$-D) & \\BTPlaqQsdQftAtaCz\\ (\\BTPlaqQsdQftAtaDepth) & \\BTPlaqQsdQftTfCz\\ (\\BTPlaqQsdQftTfDepth) & --- & $\\le 10^{-13}$ & $4\\times$ more \\texttt{CZ} than structured 5Q \\\\",
            "$S_2(\\Delta t)$ 4-Link Unfixed In-Place & 20Q ($32^4$-D) & \\BTPlaqFourLinkInplaceAtaCz\\ (\\BTPlaqFourLinkInplaceAtaDepth) & \\BTPlaqFourLinkInplaceHhCz\\ (\\BTPlaqFourLinkInplaceHhDepth) & --- & $\\BTPlaqFourLinkBasisMaxErr$ & \\BTPlaqCompTwentyQAtaRatio\\ (All) / \\BTPlaqCompTwentyQHhRatio\\ (HH) \\\\",
            "$S_2(\\Delta t)$ 4-Link Unfixed Ancilla & 25Q ($32^5$-D) & \\BTPlaqFourLinkAncillaAtaCz\\ (\\BTPlaqFourLinkAncillaAtaDepth) & \\BTPlaqFourLinkAncillaHhCz\\ (\\BTPlaqFourLinkAncillaHhDepth) & --- & $\\BTPlaqMaxGaugeDiscrepancy$ & \\BTPlaqCompTwentyFiveQAtaRatio\\ (All) / \\BTPlaqCompTwentyFiveQHhRatio\\ (HH) \\\\",
            "\\hline",
            "Trotter Convergence ($N=8$, $\\Delta t=\\BTPlaqDtStepEight$) & 5Q ($g_{\\mathrm{YM}}^2=\\BTPlaqCompileGymSq$) & Lie: $\\BTPlaqLieErrGOneStepEight$ & Strang: $\\BTPlaqStrangErrGOneStepEight$ & --- & --- & Coarse-step baseline \\\\",
            "Trotter Convergence ($N=32$, $\\Delta t=\\BTPlaqDtStepThirtyTwo$) & 5Q ($g_{\\mathrm{YM}}^2=\\BTPlaqCompileGymSq$) & Lie: $\\BTPlaqLieErrGOneStepThirtyTwo$ & Strang: $\\BTPlaqStrangErrGOneStepThirtyTwo$ & --- & --- & $16\\times$ Strang reduction vs $N=8$ \\\\",
            "Trotter Convergence ($N=64$, $\\Delta t=\\BTPlaqDtStepSixtyFour$) & 5Q ($g_{\\mathrm{YM}}^2=\\BTPlaqCompileGymSq$) & Lie: $\\BTPlaqLieErrGOneStepSixtyFour$ & Strang: $\\BTPlaqStrangErrGOneStepSixtyFour$ & --- & State $L_2$: $\\BTPlaqRegimeMaxStateErr$ & Gauss drift $\\BTPlaqRegimeMaxGaussDrift$ \\\\",
            "Trotter Convergence ($N=256$, $\\Delta t=\\BTPlaqDtStepTwoFiveSix$) & 5Q ($g_{\\mathrm{YM}}^2=\\BTPlaqCompileGymSq$) & Lie: $\\BTPlaqLieErrGOneStepTwoFiveSix$ & Strang: $\\BTPlaqStrangErrGOneStepTwoFiveSix$ & --- & --- & Sub-$10^{-6}$ spectral error \\\\",
            "Trotter Convergence ($N=2048$, $\\Delta t=\\BTPlaqDtStepTwoZeroFourEight$) & 5Q ($g_{\\mathrm{YM}}^2=\\BTPlaqCompileGymSq$) & Lie: $\\BTPlaqLieErrGOneStepTwoZeroFourEight$ & Strang: $\\BTPlaqStrangErrGOneStepTwoZeroFourEight$ & --- & --- & Log-log slopes: $\\BTPlaqLieSlopeGOne$ vs $\\BTPlaqStrangSlopeGOne$ \\\\",
            "\\end{tabular}",
            "\\end{ruledtabular}",
            "\\end{table*}",
            "",
        ]
    )
    (TABLES_DIR / "table2_strang_scaling_and_gate_compression.tex").write_bytes(
        t2_tex.encode("utf-8")
    )

    t3_lines = [
        "% Auto-generated by build_paper_assets.py -- DO NOT EDIT BY HAND",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{All $K = \\BTPlaqFezNumPlacements$ vertex-disjoint five-qubit \\texttt{T5} heavy-hex placements ($\\BTPlaqFezTotalTransmons$ physical transmons) on \\BTPlaqFezBackend\\ under Window-2 job \\BTPlaqFezJobId\\ ($\\BTPlaqFezTotalPubs$ PUBs $\\times \\BTPlaqFezShotsPerPub = \\BTPlaqFezTotalShots$ shots, tier \\BTPlaqFezExecTier, protocol \\BTPlaqFezProtocolId) paired with Window-1 job \\BTPlaqFezWindowOneJobId\\ ($+\\BTPlaqEchoElapsedHours$\\,h elapsed). Between windows, hub qubit Q\\BTPlaqDriftedHubQubit\\ of \\texttt{pid=1} suffered an unflagged $+\\BTPlaqQSevenExcitationPct$ $|0\\rangle\\to|1\\rangle$ SPAM/readout drift excursion (\\texttt{IDENTITY\\_CTRL\\_0CZ} dropping $\\BTPlaqWOnePidOneIdFid \\to \\BTPlaqWTwoPidOneIdFid$, \\texttt{ECHO\\_PROBE\\_16CZ} dropping $\\BTPlaqWOnePidOneEcho \\to \\BTPlaqWTwoPidOneEcho$). Stale calibration ranked \\texttt{pid=1} at \\#1 ($\\rho_{\\mathrm{cal}} = +\\BTPlaqWTwoSpearmanCal$, top-1 regret $\\BTPlaqWTwoRegretCal$), whereas live \\texttt{ECHO\\_PROBE\\_16CZ} promoted \\texttt{pid=0} to \\#1 ($\\rho_{\\mathrm{echo}} = +\\BTPlaqWTwoSpearmanEcho$, $\\Delta\\rho = +\\BTPlaqWTwoDeltaSpearman$, top-1 regret $\\BTPlaqWTwoRegretEcho$ across both windows).}",
        "\\label{tab:bt2t003_fez_10placement_echo_and_plaquette}",
        "\\begin{ruledtabular}",
        "\\begin{tabular}{lcccccccccc}",
        "Placement & Cal Score (Rank) & \\texttt{ID\\_0CZ} & \\texttt{ECHO\\_16CZ} (Rank) & Norm Echo & $U_B^{\\mathrm{cnt}}(\\pi/4)$ & $U_B^{\\mathrm{hrw}}(\\pi/2)$ & Plaq 2WL & \\texttt{CLF\\_30CZ} & \\texttt{MIR\\_24CZ} & Held-Out 4WL \\\\",
        "\\hline",
    ]
    for p_idx in range(10):
        pw = DIGIT_WORDS[p_idx]
        t3_lines.append(
            f"\\texttt{{pid={p_idx}}} \\BTPlaqPid{pw}Qubits & $\\BTPlaqPid{pw}CalScore$ (\\#\\BTPlaqPid{pw}CalRank) & "
            f"$\\BTPlaqPid{pw}IdZeroCz$ & $\\BTPlaqPid{pw}EchoSixteenCz$ (\\#\\BTPlaqPid{pw}EchoRank) & "
            f"$\\BTPlaqPid{pw}NormEcho$ & $\\BTPlaqPid{pw}UbCenter$ & $\\BTPlaqPid{pw}UbHurwitz$ & "
            f"$\\BTPlaqPid{pw}PlaqComp$ & $\\BTPlaqPid{pw}Clifford$ & $\\BTPlaqPid{pw}Mirror$ & "
            f"$\\BTPlaqPid{pw}HeldoutComp$ \\\\"
        )
    t3_lines.extend(
        [
            "\\hline",
            "Window-2 Spearman $\\rho_{\\mathrm{cal}}$ & --- & --- & --- & --- & $+\\BTPlaqRhoCalCenter$ & $+\\BTPlaqRhoCalHurwitz$ & $+\\BTPlaqRhoCalPlaqTwoW$ & $+\\BTPlaqRhoCalClifford$ & $+\\BTPlaqRhoCalMirror$ & $+\\BTPlaqWTwoSpearmanCal$ \\\\",
            "Window-2 Spearman $\\rho_{\\mathrm{echo}}$ & --- & --- & --- & --- & $+\\BTPlaqRhoEchoCenter$ & $+\\BTPlaqRhoEchoHurwitz$ & $+\\BTPlaqRhoEchoPlaqTwoW$ & $+\\BTPlaqRhoEchoClifford$ & $+\\BTPlaqRhoEchoMirror$ & $+\\BTPlaqWTwoSpearmanEcho$ \\\\",
            "Advantage $\\Delta\\rho = \\rho_{\\mathrm{echo}} - \\rho_{\\mathrm{cal}}$ & --- & --- & --- & --- & $+\\BTPlaqDeltaRhoCenter$ & $+\\BTPlaqDeltaRhoHurwitz$ & $+\\BTPlaqDeltaRhoPlaqTwoW$ & $+\\BTPlaqDeltaRhoClifford$ & $+\\BTPlaqDeltaRhoMirror$ & $+\\BTPlaqWTwoDeltaSpearman$ \\\\",
            "2-Window Mean ($\\bar{\\rho}_{\\mathrm{echo}}$ vs $\\bar{\\rho}_{\\mathrm{cal}}$) & $\\Delta\\bar{\\rho} = +\\BTPlaqTwoWinMeanDeltaSpearman$ & Win-1 Regret: & $\\BTPlaqWOneRegretEcho$ / $\\BTPlaqWOneRegretCal$ & Win-2 Regret: & $\\BTPlaqWTwoRegretEcho$ (Echo) & vs & $\\BTPlaqWTwoRegretCal$ (Cal) & $\\bar{\\rho}_{\\mathrm{cal}} = +\\BTPlaqTwoWinMeanSpearmanCal$ & --- & $\\bar{\\rho}_{\\mathrm{echo}} = +\\BTPlaqTwoWinMeanSpearmanEcho$ \\\\",
            "\\end{tabular}",
            "\\end{ruledtabular}",
            "\\end{table*}",
            "",
        ]
    )
    (TABLES_DIR / "table3_fez_10placement_echo_and_plaquette.tex").write_bytes(
        "\n".join(t3_lines).encode("utf-8")
    )

    t4_tex = "\n".join(
        [
            "% Auto-generated by build_paper_assets.py -- DO NOT EDIT BY HAND",
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Hardware-attested non-abelian $2T \\subset SU(2)$ magnetic plaquette character-sector trajectories and structured-versus-generic compilation ablation on \\BTPlaqFezBackend\\ (job \\BTPlaqFezJobId, $\\BTPlaqFezTotalShots$ shots, tier \\BTPlaqFezExecTier), comparing \\#1 Echo-selected \\texttt{pid=0} (\\BTPlaqPidZeroQubits) against Q\\BTPlaqDriftedHubQubit-drifted stale-calibration \\#1 \\texttt{pid=1} (\\BTPlaqPidOneQubits).}",
            "\\label{tab:bt2t003_fez_character_sector_trajectories}",
            "\\begin{ruledtabular}",
            "\\begin{tabular}{lcccccc}",
            "Character Sector / Experimental Arm & Evolution Angle $\\tau$ & Total \\texttt{T5} \\texttt{CZ} & Routed Target $y$ & Ideal $SU(2)$ Theory & Measured \\texttt{pid=0} (Echo \\#1) & Measured \\texttt{pid=1} (Drifted) \\\\",
            "\\hline",
            "Center ($\\Delta\\chi_{2D}=4$, $P_{|+\\rangle}=\\cos^2 2\\tau$) & $\\tau = 0$ & \\BTPlaqFezCzCenterStruct & $0$ (\\texttt{00000}) & $\\BTPlaqIdealOne$ & $\\BTPlaqPidZeroCenterTZero$ & $\\BTPlaqPidOneCenterTZero$ \\\\",
            "Center ($\\Delta\\chi_{2D}=4$, $P_{|+\\rangle}=\\cos^2 2\\tau$) & $\\tau = \\pi/8$ & \\BTPlaqFezCzCenterStruct & $0$ (\\texttt{00000}) & $\\BTPlaqIdealHalf$ & $\\BTPlaqPidZeroCenterPiEight$ & $\\BTPlaqPidOneCenterPiEight$ \\\\",
            "Center ($\\Delta\\chi_{2D}=4$, $P_{|+\\rangle}=\\cos^2 2\\tau$) & $\\tau = \\pi/4$ ($P_{|+\\rangle}$) & \\BTPlaqFezCzCenterStruct & $0$ (\\texttt{00000}) & $\\BTPlaqIdealZero$ & $\\BTPlaqPidZeroCenterPiFourPlus$ & $\\BTPlaqPidOneCenterPiFourPlus$ \\\\",
            "Center Inversion (Structured \\BTPlaqUbTfCoreCz-\\texttt{CZ} Core) & $\\tau = \\pi/4$ ($P_{|-\\rangle}$) & \\BTPlaqFezCzCenterStruct & $\\BTPlaqRoutedYCenterInv$ (\\BTPlaqRoutedBitsCenterInv) & $\\BTPlaqIdealOne$ & $\\mathbf{\\BTPlaqPidZeroCenterPiFourMinus}$ (\\BTPlaqPidZeroCenterPiFourMinusPct) & $\\BTPlaqPidOneCenterPiFourMinus$ \\\\",
            "Center Inversion (Generic Dense QFT-Conj.) & $\\tau = \\pi/4$ ($P_{|-\\rangle}$) & \\BTPlaqFezCzCenterGeneric & $\\BTPlaqRoutedYCenterInv$ (\\BTPlaqRoutedBitsCenterInv) & $\\BTPlaqIdealOne$ & $\\BTPlaqPidZeroCenterGenericMinus$ ($+\\BTPlaqPidZeroCenterStructAdv$ adv.) & $\\BTPlaqPidOneCenterGenericMinus$ ($+\\BTPlaqPidOneCenterStructAdv$ adv.) \\\\",
            "Center ($\\Delta\\chi_{2D}=4$, $P_{|+\\rangle}=\\cos^2 2\\tau$) & $\\tau = \\pi/2$ (Revival) & \\BTPlaqFezCzCenterStruct & $0$ (\\texttt{00000}) & $\\BTPlaqIdealOne$ & $\\BTPlaqPidZeroCenterPiTwo$ ($r=\\BTPlaqPidZeroPearsonCenter$) & $\\BTPlaqPidOneCenterPiTwo$ ($r=\\BTPlaqPidOnePearsonCenter$) \\\\",
            "\\hline",
            "Hurwitz ($\\Delta\\chi_{2D}=2$, $P_{\\Phi^+}=\\cos^2 \\tau$) & $\\tau = 0$ & \\BTPlaqFezCzHurwitzStruct & $0$ (\\texttt{00000}) & $\\BTPlaqIdealOne$ & $\\BTPlaqPidZeroHurwitzTZero$ & $\\BTPlaqPidOneHurwitzTZero$ \\\\",
            "Hurwitz ($\\Delta\\chi_{2D}=2$, $P_{\\Phi^+}=\\cos^2 \\tau$) & $\\tau = \\pi/8$ & \\BTPlaqFezCzHurwitzStruct & $0$ (\\texttt{00000}) & $\\BTPlaqIdealCosSqPiEight$ & $\\BTPlaqPidZeroHurwitzPiEight$ & $\\BTPlaqPidOneHurwitzPiEight$ \\\\",
            "Hurwitz ($\\Delta\\chi_{2D}=2$, $P_{\\Phi^+}=\\cos^2 \\tau$) & $\\tau = \\pi/4$ & \\BTPlaqFezCzHurwitzStruct & $0$ (\\texttt{00000}) & $\\BTPlaqIdealHalf$ & $\\BTPlaqPidZeroHurwitzPiFour$ & $\\BTPlaqPidOneHurwitzPiFour$ \\\\",
            "Hurwitz ($\\Delta\\chi_{2D}=2$, $P_{\\Phi^+}=\\cos^2 \\tau$) & $\\tau = 3\\pi/8$ & \\BTPlaqFezCzHurwitzStruct & $0$ (\\texttt{00000}) & $\\BTPlaqIdealCosSqThreePiEight$ & $\\BTPlaqPidZeroHurwitzThreePiEight$ & $\\BTPlaqPidOneHurwitzThreePiEight$ \\\\",
            "Hurwitz ($\\Delta\\chi_{2D}=2$, $P_{\\Phi^+}=\\cos^2 \\tau$) & $\\tau = \\pi/2$ ($P_{\\Phi^+}$) & \\BTPlaqFezCzHurwitzStruct & $0$ (\\texttt{00000}) & $\\BTPlaqIdealZero$ & $\\BTPlaqPidZeroHurwitzPiTwoPlus$ ($r=\\BTPlaqPidZeroPearsonHurwitz$) & $\\BTPlaqPidOneHurwitzPiTwoPlus$ ($r=\\BTPlaqPidOnePearsonHurwitz$) \\\\",
            "Hurwitz Bell Inversion ($P_{\\Phi^-}$) & $\\tau = \\pi/2$ ($P_{\\Phi^-}$) & \\BTPlaqFezCzHurwitzStruct & $\\BTPlaqRoutedYHurwitzInv$ (\\BTPlaqRoutedBitsHurwitzInv) & $\\BTPlaqIdealOne$ & $\\mathbf{\\BTPlaqPidZeroHurwitzPiTwoMinus}$ (\\BTPlaqPidZeroHurwitzPiTwoMinusPct) & $\\BTPlaqPidOneHurwitzPiTwoMinus$ \\\\",
            "\\hline",
            "Imaginary-$Q_8$ Null Control ($\\Delta\\chi_{2D}=0$, $|\\mathbf{i}\\rangle$) & $\\tau = \\pi/4$ & \\BTPlaqFezCzImagNull & $\\BTPlaqRoutedYImagNull$ (\\BTPlaqRoutedBitsImagNull) & $\\BTPlaqIdealOne$ & $\\mathbf{\\BTPlaqPidZeroImagNullSurv}$ & $\\BTPlaqPidOneImagNullSurv$ \\\\",
            "Unused-$8$ Decoupling Control ($\\Delta\\chi_{2D}=0$, $|y=24\\rangle$) & $\\tau = \\pi/4$ & \\BTPlaqFezCzUnusedNull & $\\BTPlaqRoutedYUnusedNull$ (\\BTPlaqRoutedBitsUnusedNull) & $\\BTPlaqIdealOne$ & $\\mathbf{\\BTPlaqPidZeroUnusedNullSurv}$ & $\\BTPlaqPidOneUnusedNullSurv$ \\\\",
            "\\end{tabular}",
            "\\end{ruledtabular}",
            "\\end{table*}",
            "",
        ]
    )
    (TABLES_DIR / "table4_fez_character_sector_trajectories.tex").write_bytes(
        t4_tex.encode("utf-8")
    )

    fig1 = _build_fig1(manifest, fez_results)
    fig1.save_pdf_and_png(FIGURES_DIR / FIG1_STEM, title=FIG1_STEM)

    fig2 = _build_fig2(fez_results)
    fig2.save_pdf_and_png(FIGURES_DIR / FIG2_STEM, title=FIG2_STEM)

    fig3 = _build_fig3(fez_results)
    fig3.save_pdf_and_png(FIGURES_DIR / FIG3_STEM, title=FIG3_STEM)

    preprint_bytes = _build_preprint_pdf(manifest, fez_results, fez_job, fig1, fig2, fig3)
    preprint_path = HERE / PREPRINT_FILENAME
    preprint_path.write_bytes(preprint_bytes)
    if PREPRINTS_ARCHIVE_DIR.is_dir():
        (PREPRINTS_ARCHIVE_DIR / PREPRINT_FILENAME).write_bytes(preprint_bytes)

    (HERE / "generated.tex").write_bytes((HERE / "generated_macros.tex").read_bytes())
    capsule_body = {
        "schema": "osiris.reproducibility_capsule/1",
        "paper_id": "OSIRIS-PAPER-BT2T003-PLAQUETTE-2026",
        "protocol_ids": ["OSIRIS-BT2T-003", "OSIRIS-BT2T-003-FEZ"],
        "execution_tiers": {
            "OSIRIS-BT2T-003": "SIMULATED",
            "OSIRIS-BT2T-003-FEZ": "HARDWARE_ATTESTED",
        },
        "provider_job_ids": {
            "OSIRIS-BT2T-003": None,
            "OSIRIS-BT2T-003-FEZ": str(fez_results["provider_job_id"]),
            "OSIRIS-ECHO-FEZ-WINDOW1": str(fez_results["two_window_echo_summary"]["window1_job_id"]),
        },
        "backend": "ibm_fez",
        "upstream_manifests": {
            "experiments/bt2t_plaquette/manifest_bt2t003.json": _sha256_file((HERE / "../../experiments/bt2t_plaquette/manifest_bt2t003.json").resolve()),
            "experiments/bt2t003_fez/physical_bt2t003_fez_results.json": _sha256_file((HERE / "../../experiments/bt2t003_fez/physical_bt2t003_fez_results.json").resolve()),
            "experiments/bt2t003_fez/physical_bt2t003_fez_raw_counts.json.gz": str(fez_results["raw_counts_gz_sha256"]),
        },
        "preprint_pdf_sha256": hashlib.sha256(preprint_bytes).hexdigest(),
    }
    canon_bytes = json.dumps(
        capsule_body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    capsule = dict(capsule_body)
    capsule["rfc8785_canonical_sha256"] = hashlib.sha256(canon_bytes).hexdigest()
    (HERE / "reproducibility_capsule.json").write_bytes(
        (json.dumps(capsule, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )

    local_rels = [
        "README.md",
        PREPRINT_FILENAME,
        "build_paper_assets.py",
        f"figures/{FIG1_STEM}.pdf",
        f"figures/{FIG1_STEM}.png",
        f"figures/{FIG2_STEM}.pdf",
        f"figures/{FIG2_STEM}.png",
        f"figures/{FIG3_STEM}.pdf",
        f"figures/{FIG3_STEM}.png",
        "generated.tex",
        "generated_macros.tex",
        "main.tex",
        "references.bib",
        "reproducibility_capsule.json",
        "tables/table1_conjugacy_magnetic_potential.tex",
        "tables/table2_strang_scaling_and_gate_compression.tex",
        "tables/table3_fez_10placement_echo_and_plaquette.tex",
        "tables/table4_fez_character_sector_trajectories.tex",
    ]
    upstream_rels = [
        "../../experiments/bt2t_plaquette/manifest_bt2t003.json",
        "../../experiments/bt2t003_fez/physical_bt2t003_fez_results.json",
    ]
    sums_lines: List[str] = []
    for rel in local_rels + upstream_rels:
        target_p = (HERE / rel).resolve()
        if target_p.is_file():
            sums_lines.append(f"{_sha256_file(target_p)}  {rel}")
    (HERE / "SHA256SUMS.txt").write_bytes(("\n".join(sums_lines) + "\n").encode("utf-8"))

    return {
        "macros_count": sum(1 for ln in macros if ln.startswith("\\newcommand")),
        "preprint_bytes": len(preprint_bytes),
        "sums_count": len(sums_lines),
    }


if __name__ == "__main__":
    generate_all()
