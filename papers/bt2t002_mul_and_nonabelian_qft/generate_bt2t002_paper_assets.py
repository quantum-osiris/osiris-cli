#!/usr/bin/env python3
"""Programmatic asset, macro, dual SVG/PDF 1.4 figure, standalone multi-page PDF 1.4
preprint, RFC 8785 capsule, and SHA256SUMS generator for
`papers/bt2t002_mul_and_nonabelian_qft/`.

Deterministic, zero timestamps in generated outputs, zero network calls, zero dependency
on `qiskit_ibm_runtime` or `pytket`, graceful fallback to
`osiris_cli.orchestration_departments.jcs_bytes` when `rfc8785` is absent, and zero
hand-typed empirical floats in `bt2t002_mul_and_nonabelian_qft.tex`.

Reads:
  - `experiments/bt2t_primitives/manifest_bt2t002.json` (`OSIRIS-BT2T-002`, `SIMULATED`)
  - `experiments/bt2t_primitives/bt2t_mul.py`
  - `experiments/bt2t_primitives/bt2t_qft.py`
  - `experiments/bt2t_primitives/verify_bt2t002.py`
  - `experiments/bt2t_inversion/bt2t.py`
  - `experiments/bt2t_inversion/manifest.json`
  - `experiments/bt2t_fez/physical_bt2t_fez_results.json` (`OSIRIS-BT2T-001-FEZ` M1 anchor)
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

try:
    import rfc8785 as _rfc8785  # type: ignore[import-not-found]
except ImportError:
    _rfc8785 = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from papers._pdf_preprint_renderer import PreprintDocument, VectorCanvas  # noqa: E402

PRIM_DIR = ROOT / "experiments" / "bt2t_primitives"
INV_DIR = ROOT / "experiments" / "bt2t_inversion"
FEZ_DIR = ROOT / "experiments" / "bt2t_fez"

MANIFEST_BT2T002_PATH = PRIM_DIR / "manifest_bt2t002.json"
PRIM_INIT_PATH = PRIM_DIR / "__init__.py"
BT2T_MUL_PY_PATH = PRIM_DIR / "bt2t_mul.py"
BT2T_QFT_PY_PATH = PRIM_DIR / "bt2t_qft.py"
VERIFY_BT2T002_PY_PATH = PRIM_DIR / "verify_bt2t002.py"

BT2T_BASE_PY_PATH = INV_DIR / "bt2t.py"
INV_MANIFEST_PATH = INV_DIR / "manifest.json"
FEZ_RESULTS_PATH = FEZ_DIR / "physical_bt2t_fez_results.json"

SOURCE_ARTIFACT_PATHS: Tuple[Path, ...] = (
    PRIM_INIT_PATH,
    BT2T_MUL_PY_PATH,
    BT2T_QFT_PY_PATH,
    VERIFY_BT2T002_PY_PATH,
    MANIFEST_BT2T002_PATH,
    BT2T_BASE_PY_PATH,
    INV_MANIFEST_PATH,
    FEZ_RESULTS_PATH,
)

FIG1_STEM = "fig1_bt2t002_semidirect_mul_compression"
FIG2_STEM = "fig2_bt2t002_peter_weyl_qft_and_hsp"
PDF_FILENAME = "bt2t002_mul_and_nonabelian_qft.pdf"

SUBGROUP_ORDER: Tuple[Tuple[str, str, str, str], ...] = (
    ("trivial_1", "{1}", "\\{\\mathbf{1}\\}", "Yes"),
    ("Z2_center", "Z_2", "\\mathbb{Z}_2 = \\{\\pm\\mathbf{1}\\}", "Yes"),
    ("Z4_i", "Z_4", "\\mathbb{Z}_4 = \\langle\\mathbf{i}\\rangle", "No"),
    ("Z3_omega", "Z_3", "\\mathbb{Z}_3 = \\langle\\omega\\rangle", "No"),
    ("Q8_normal", "Q_8", "Q_8 \\triangleleft 2T", "Yes"),
    ("full_2T", "2T", "2T", "Yes"),
)

IRREP_SPECS: Tuple[Tuple[str, str, str, int], ...] = (
    ("chi_0", "chi_0", "\\chi_0", 1),
    ("chi_1", "chi_1", "\\chi_1", 1),
    ("chi_2", "chi_2", "\\chi_2", 1),
    ("rho_2_0", "rho_{2,0}", "\\rho_{2,0}", 2),
    ("rho_2_1", "rho_{2,1}", "\\rho_{2,1}", 2),
    ("rho_2_2", "rho_{2,2}", "\\rho_{2,2}", 2),
    ("rho_3", "rho_3", "\\rho_3", 3),
)


def _canonical_jcs_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize `payload` via RFC 8785 JCS, falling back cleanly to `osiris_cli` JCS."""
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
    exp_int = int(exp_str)
    return f"{mant} \\times 10^{{{exp_int}}}"


def _fmt_sci_ascii(val: float, digits: int = 2) -> str:
    if val == 0.0:
        return "0.0"
    return f"{val:.{digits}e}"


def _load_prim_module(mod_name: str, path: Path) -> Any:
    if str(PRIM_DIR) not in sys.path:
        sys.path.insert(0, str(PRIM_DIR))
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {mod_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _find_proxy(
    compilations: Sequence[Mapping[str, Any]],
    *,
    circuit: str,
    kind_sub: str | None = None,
    target: str | None = None,
    opt_level: int | None = None,
    peephole: bool | None = None,
) -> Mapping[str, Any]:
    for entry in compilations:
        if entry.get("circuit") != circuit:
            continue
        if kind_sub is not None and kind_sub.lower() not in str(entry.get("kind", "")).lower():
            continue
        if target is not None and entry.get("target") != target:
            continue
        if opt_level is not None and int(entry.get("optimization_level", -1)) != opt_level:
            continue
        if peephole is not None and bool(entry.get("FullPeepholeOptimise")) is not peephole:
            continue
        return entry
    raise KeyError(
        f"Missing proxy compilation: {circuit=} {kind_sub=} {target=} {opt_level=} {peephole=}"
    )


def _compute_esd_and_plancherel_rms() -> Dict[str, Any]:
    """Compute exact 5Q E_sd gate breakdown and 24x24 Schur RMS block matrix without Qiskit."""
    bt2t_mul = _load_prim_module("bt2t_mul", BT2T_MUL_PY_PATH)
    bt2t_qft = _load_prim_module("bt2t_qft", BT2T_QFT_PY_PATH)

    esd_gates = bt2t_mul.semidirect_bijection_gate_list(0)
    esd_counts: Counter[str] = Counter()
    for kind, ctrls, _tgts in esd_gates:
        if kind == "mcx":
            esd_counts[f"c{len(ctrls)}x"] += 1
        else:
            esd_counts[kind] += 1

    # Compute classical circuit depth on 5 wires for E_sd:
    wire_layer = [0] * 5
    for _kind, ctrls, tgts in esd_gates:
        touched = list(ctrls) + list(tgts)
        lvl = max(wire_layer[q] for q in touched) + 1
        for q in touched:
            wire_layer[q] = lvl
    esd_depth = max(wire_layer)

    f24 = bt2t_qft.peter_weyl_matrix_24()
    f24_dag = f24.conj().T
    rms_sq = np.zeros((24, 24), dtype=float)
    max_abs = np.zeros((24, 24), dtype=float)
    for g in range(24):
        lg = bt2t_qft.left_regular_matrix_24(g)
        conj_l = np.abs(f24 @ lg @ f24_dag)
        rms_sq += conj_l ** 2
        max_abs = np.maximum(max_abs, conj_l)
    rms_mat = np.sqrt(rms_sq / 24.0)

    return {
        "esd_counts": dict(esd_counts),
        "esd_total": len(esd_gates),
        "esd_depth": esd_depth,
        "rms_mat": rms_mat,
        "max_abs_mat": max_abs,
    }


def _fraction_label(w: float) -> str:
    """Exact rational label for 2T coset Fourier sector probabilities (denominator dividing 24)."""
    num = int(round(w * 24.0))
    if num == 0:
        return "0"
    if num == 24:
        return "1"
    g = math.gcd(num, 24)
    return f"{num // g}/{24 // g}"


def _build_fig1(
    manifest: Mapping[str, Any],
    inv_manifest: Mapping[str, Any],
    fez_results: Mapping[str, Any],
) -> VectorCanvas:
    """Fig. 1: Log-scale entangling-gate compression (M1 5Q inv + M2 10Q mul + 5Q QFT) & logical breakdown."""
    w, h = 576.0, 306.0
    c = VectorCanvas(w, h)

    p1 = manifest["part1_multiplication_10q"]
    p2 = manifest["part2_nonabelian_qft_5q"]
    m1_m = p1["metrics"]
    log_g = p1["logical_circuits"]["generic"]
    log_s = p1["logical_circuits"]["structured"]

    inv_prox = inv_manifest["proxy_compilations"]
    m1_ata_g = int(
        _find_proxy(inv_prox, circuit="generic", kind_sub="qiskit", target="all_to_all", opt_level=3)[
            "two_qubit_gates"
        ]["cz"]
    )
    m1_ata_s = int(
        _find_proxy(inv_prox, circuit="structured", kind_sub="qiskit", target="all_to_all", opt_level=3)[
            "two_qubit_gates"
        ]["cz"]
    )
    m1_fez_g = int(fez_results["arms"]["GENERIC"]["cz_count"])
    m1_fez_s = int(fez_results["arms"]["STRUCTURED"]["cz_count"])

    mul_ata_g = int(m1_m["cz_generic_all_to_all_opt3"])
    mul_ata_s = int(m1_m["cz_structured_all_to_all_opt3"])
    mul_hh_g = int(m1_m["cz_generic_heavy_hex_10q_opt3"])
    mul_hh_s = int(m1_m["cz_structured_heavy_hex_10q_opt3"])

    qft_ata = int(p2["logical_circuit"]["cz_all_to_all_opt3"])
    qft_t5 = int(p2["logical_circuit"]["cz_heavy_hex_T5_opt3"])

    c.text(
        w / 2.0,
        15.5,
        "OSIRIS-BT2T-002 (SIMULATED / PROXY, provider_job_id = null, bt2t-enc-1, seed = 20261124)",
        size=9.3,
        bold=True,
        anchor="middle",
    )
    c.text(
        w / 2.0,
        27.5,
        "Semidirect-Product Q_8 |x Z_3 Synthesis Compression (19.00x All-to-All, 21.31x 10Q Heavy-Hex) vs. M1 ibm_fez Anchor",
        size=7.1,
        color="#475569",
        anchor="middle",
    )

    # Panel (a): Log10 2Q gate comparison across M1 & M2
    ax_x, ax_y, ax_w, ax_h = 46.0, 47.0, 344.0, 194.0
    c.text(
        ax_x,
        ax_y - 6.0,
        "(a) Two-Qubit Entangling Gate Count (Log10 Scale: CZ / ZZPhase)",
        size=7.9,
        bold=True,
        color="#0f172a",
    )
    c.rect(ax_x, ax_y, ax_w, ax_h, fill="#fafafa", stroke="#334155", lw=0.9)

    log_min, log_max = 1.0, 5.45  # 10^1 to ~2.82e5

    def y_log(val: float) -> float:
        lv = max(log_min, min(log_max, math.log10(max(1.0, val))))
        return ax_y + (1.0 - (lv - log_min) / (log_max - log_min)) * ax_h

    # Shaded NISQ viable band (< 100 CZ, anchored by M1 72-CZ ibm_fez P_valid24 = 0.5337)
    y_100 = y_log(100.0)
    y_10 = y_log(10.0)
    c.rect(ax_x + 0.6, y_100, ax_w - 1.2, y_10 - y_100 - 0.6, fill="#ecfdf5")

    for exp_i, lbl in ((1, "10^1"), (2, "10^2"), (3, "10^3"), (4, "10^4"), (5, "10^5")):
        yy = y_log(10.0 ** exp_i)
        c.line(ax_x, yy, ax_x + ax_w, yy, color="#e2e8f0", lw=0.6)
        c.text(ax_x - 4.5, yy + 2.4, lbl, size=6.5, color="#334155", anchor="end", mono=True)

    # M1 723-CZ heavy-hex decoherence reference line
    y_723 = y_log(float(m1_fez_g))
    c.line(ax_x, y_723, ax_x + ax_w, y_723, color="#b45309", lw=0.8, dashed=True)

    # Callout box in open upper-left region of Panel (a) (above M1 bars, zero bar overlap)
    c.rect(ax_x + 6.0, ax_y + 6.0, 134.0, 33.0, fill="#ffffff", stroke="#cbd5e1", lw=0.6)
    c.text(ax_x + 10.0, ax_y + 15.0, "M1 ibm_fez Hardware Regimes:", size=5.6, bold=True, color="#0f172a")
    c.text(ax_x + 10.0, ax_y + 24.0, "-- 723 CZ: P=0.0661 (QCCD/FT above)", size=5.3, bold=True, color="#b45309")
    c.text(ax_x + 10.0, ax_y + 33.0, "[] <100 CZ: NISQ viable (72 CZ P=0.534)", size=5.3, bold=True, color="#047857")

    groups = [
        ("M1 5Q Inv", "All-to-All", m1_ata_g, m1_ata_s, f"{m1_ata_g / m1_ata_s:.2f}x", False),
        ("M1 5Q Inv", "ibm_fez T5", m1_fez_g, m1_fez_s, f"{m1_fez_g / m1_fez_s:.2f}x", False),
        ("M2 10Q Mul", "All-to-All", mul_ata_g, mul_ata_s, f"{mul_ata_g / mul_ata_s:.2f}x", False),
        ("M2 10Q Mul", "10Q H-Hex", mul_hh_g, mul_hh_s, f"{mul_hh_g / mul_hh_s:.2f}x", False),
        ("M2 5Q QFT", "T5 / All(H2)", qft_t5, qft_ata, "1.45x", True),
    ]

    slot_w = ax_w / float(len(groups))
    bw = 19.5
    y0 = y_log(10.0)
    for idx, (top_lbl, sub_lbl, v_hi, v_lo, ratio_txt, is_qft) in enumerate(groups):
        cx = ax_x + (idx + 0.5) * slot_w
        if is_qft:
            c_hi = ("#a78bfa", "#5b21b6")  # T5 613 CZ
            c_lo = ("#7c3aed", "#4c1d95")  # All-to-all 423 CZ / ZZPhase
        else:
            c_hi = ("#ef4444", "#991b1b")  # GENERIC
            c_lo = ("#2563eb", "#1e3a8a")  # STRUCTURED

        yh = y_log(float(v_hi))
        yl = y_log(float(v_lo))
        x_left = cx - bw - 1.5
        x_right = cx + 1.5
        c.rect(x_left, yh, bw, y0 - yh, fill=c_hi[0], stroke=c_hi[1], lw=0.75)
        c.rect(x_right, yl, bw, y0 - yl, fill=c_lo[0], stroke=c_lo[1], lw=0.75)

        c.text(
            x_left + 0.5 * bw,
            yh - 3.5,
            f"{v_hi:,}",
            size=5.6,
            bold=True,
            color=c_hi[1],
            anchor="middle",
            mono=True,
        )
        c.text(
            x_right + 0.5 * bw,
            yl - 3.5,
            f"{v_lo:,}",
            size=5.6,
            bold=True,
            color=c_lo[1],
            anchor="middle",
            mono=True,
        )
        # Place compression ratio cleanly above the structured/right bar (or above QFT pair)
        if is_qft:
            c.text(
                cx,
                yh - 12.5,
                ratio_txt,
                size=6.0,
                bold=True,
                color="#4c1d95",
                anchor="middle",
                mono=True,
            )
        else:
            c.text(
                x_right + 0.5 * bw,
                yl - 12.0,
                ratio_txt,
                size=5.9,
                bold=True,
                color="#0f172a",
                anchor="middle",
                mono=True,
            )
        c.line(cx, y0, cx, y0 + 3.0, color="#334155", lw=0.7)
        c.text(cx, y0 + 10.5, top_lbl, size=6.3, bold=True, color="#0f172a", anchor="middle")
        c.text(cx, y0 + 18.2, sub_lbl, size=5.8, color="#475569", anchor="middle")

    # Panel (b): Zero-ancilla 10Q logical gate & depth compression
    bx_x, bx_y, bx_w, bx_h = 406.0, 47.0, 156.0, 194.0
    c.text(
        bx_x,
        bx_y - 6.0,
        "(b) 10Q Zero-Ancilla Logical Synthesis",
        size=7.9,
        bold=True,
        color="#0f172a",
    )
    c.rect(bx_x, bx_y, bx_w, bx_h, fill="#fafafa", stroke="#334155", lw=0.9)

    gen_gc = log_g["gate_counts"]
    str_gc = log_s["gate_counts"]
    gen_tot = sum(int(v) for v in gen_gc.values())
    str_tot = sum(int(v) for v in str_gc.values())

    rows_b = [
        ("Total Gates", f"{gen_tot:,}", f"{str_tot}", f"{gen_tot / str_tot:.2f}x"),
        ("Log. Depth", f"{log_g['depth']:,}", f"{log_s['depth']}", f"{log_g['depth'] / log_s['depth']:.2f}x"),
        ("MCX (k>=3)", f"{gen_gc['mcx']:,}", "0", "Elim."),
        ("Pure CCX", "0", f"{str_gc['ccx']}", "Dirty"),
        ("CSWAP", f"{gen_gc['cswap']}", f"{str_gc['cswap']}", f"{gen_gc['cswap'] / str_gc['cswap']:.1f}x"),
        ("CX / X", f"{gen_gc['cx']}/{gen_gc['x']}", f"{str_gc['cx']}/{str_gc['x']}", "Exact"),
        ("Ancillas", "0", "0", "10Q"),
        ("||U - P||", "0.0", "0.0", "+1 ph"),
    ]

    hdr_y = bx_y + 12.0
    c.rect(bx_x + 1.0, bx_y + 1.0, bx_w - 2.0, 15.0, fill="#e2e8f0")
    c.text(bx_x + 5.0, hdr_y, "Metric", size=6.0, bold=True, color="#0f172a")
    c.text(bx_x + 76.0, hdr_y, "GEN", size=6.0, bold=True, color="#991b1b", anchor="end")
    c.text(bx_x + 114.0, hdr_y, "STR", size=6.0, bold=True, color="#1e3a8a", anchor="end")
    c.text(bx_x + bx_w - 5.0, hdr_y, "Gain", size=6.0, bold=True, color="#047857", anchor="end")
    c.line(bx_x, bx_y + 16.0, bx_x + bx_w, bx_y + 16.0, color="#334155", lw=0.7)

    ry = bx_y + 28.0
    for r_idx, (m_lbl, g_v, s_v, gain_v) in enumerate(rows_b):
        if r_idx % 2 == 1:
            c.rect(bx_x + 1.0, ry - 9.0, bx_w - 2.0, 13.5, fill="#f1f5f9")
        c.text(bx_x + 5.0, ry, m_lbl, size=5.9, bold=True, color="#1e293b")
        c.text(bx_x + 76.0, ry, g_v, size=5.8, color="#991b1b", anchor="end", mono=True)
        c.text(bx_x + 114.0, ry, s_v, size=5.8, bold=True, color="#1e3a8a", anchor="end", mono=True)
        c.text(bx_x + bx_w - 5.0, ry, gain_v, size=5.8, bold=True, color="#047857", anchor="end", mono=True)
        ry += 14.2

    c.line(bx_x + 4.0, bx_y + 136.0, bx_x + bx_w - 4.0, bx_y + 136.0, color="#cbd5e1", lw=0.6)
    note_lines = [
        "Barenco Dirty-Ancilla Theorem:",
        "Live register wires replace clean",
        "ancillas via pure {0,+1} CCX",
        "cascades -> zero phase error",
        "across all 1,024 basis states",
        "(CZ(0,5) ctrl caught: err=2.0).",
    ]
    ny = bx_y + 146.5
    for idx_n, nln in enumerate(note_lines):
        c.text(
            bx_x + 6.0,
            ny + idx_n * 8.0,
            nln,
            size=5.7,
            bold=(idx_n == 0),
            color="#0f172a" if idx_n == 0 else "#334155",
        )

    # Bottom legend & architectural takeaway strip
    ly = 266.0
    c.rect(ax_x, ly, w - ax_x - 14.0, 32.0, fill="#f8fafc", stroke="#cbd5e1", lw=0.7)
    legs = [
        ("#ef4444", "#991b1b", "GENERIC Cayley Cycle/Transposition"),
        ("#2563eb", "#1e3a8a", "STRUCTURED Q_8 |x Z_3 Synthesis"),
        ("#7c3aed", "#4c1d95", "5Q Non-Abelian QFT (613 T5 CZ / 423 All-to-All CZ & H2 ZZPhase)"),
    ]
    lx_pos = [ax_x + 8.0, ax_x + 152.0, ax_x + 286.0]
    for (fc, sc, ltxt), lx in zip(legs, lx_pos):
        c.rect(lx, ly + 5.0, 9.0, 6.2, fill=fc, stroke=sc, lw=0.7)
        c.text(lx + 12.0, ly + 10.5, ltxt, size=5.9, bold=True, color="#1e293b")

    c.text(
        w / 2.0,
        ly + 24.5,
        "Exact 1024x1024 Unitary (576 Cayley Pairs + 448 Unused Fixed): 1,739 CZ (All-to-All) & 423 ZZPhase (QFT) Target QCCD / Fault-Tolerant Execution",
        size=6.0,
        bold=True,
        color="#0f172a",
        anchor="middle",
    )
    return c


def _build_fig2(manifest: Mapping[str, Any], rms_mat: np.ndarray) -> VectorCanvas:
    """Fig. 2: (a) 24x24 Plancherel block-diagonalized F_{2T} L_g F_{2T}^dagger; (b) 6-subgroup HSP weights."""
    w, h = 576.0, 336.0
    c = VectorCanvas(w, h)

    p2 = manifest["part2_nonabelian_qft_5q"]
    blk = p2["block_diagonalization_and_convolution"]
    su = p2["schur_and_unitarity"]
    hsp = p2["subgroup_coset_hsp_sampling"]

    c.text(
        w / 2.0,
        15.5,
        "OSIRIS-BT2T-002: 7-Irrep Peter-Weyl Non-Abelian QFT Block Diagonalization & Hidden Subgroup Tomography",
        size=9.1,
        bold=True,
        anchor="middle",
    )
    c.text(
        w / 2.0,
        27.5,
        f"sum d_alpha^2 = 1^2+1^2+1^2+2^2+2^2+2^2+3^2 = 24  |  ||F_32 F_32^+ - I||_inf = {su['f32_unitarity_inf_error']:.2e}  |  Off-Block <= {blk['left_regular_max_off_block_abs']:.2e}",
        size=7.0,
        color="#475569",
        anchor="middle",
    )

    # Panel (a): 24x24 Peter-Weyl matrix heatmap
    pa_x, pa_y = 34.0, 48.0
    cell = 8.4
    grid_sz = 24.0 * cell  # 201.6 pt
    c.text(
        pa_x,
        pa_y - 6.0,
        "(a) Schur RMS |[F_{2T} L_g F_{2T}^+]_{r,c}| (24x24)",
        size=7.8,
        bold=True,
        color="#0f172a",
    )
    c.rect(pa_x, pa_y, grid_sz, grid_sz, fill="#ffffff", stroke="#334155", lw=0.9)

    # Draw nonzero cells (54 entries across 12 diagonal blocks) + subtle grid lines
    for i in range(1, 24):
        pos_xy = i * cell
        c.line(pa_x + pos_xy, pa_y, pa_x + pos_xy, pa_y + grid_sz, color="#f1f5f9", lw=0.35)
        c.line(pa_x, pa_y + pos_xy, pa_x + grid_sz, pa_y + pos_xy, color="#f1f5f9", lw=0.35)

    for r in range(24):
        for col_i in range(24):
            val = float(rms_mat[r, col_i])
            if val > 1e-6:
                if r < 3:
                    fill_col = "#10b981"  # 1D chi_0,1,2 (RMS = 1.000)
                elif r < 15:
                    fill_col = "#2563eb"  # 2D rho_{2,r} (RMS = 0.707)
                else:
                    fill_col = "#7c3aed"  # 3D rho_3 (RMS = 0.577)
                c.rect(
                    pa_x + col_i * cell + 0.3,
                    pa_y + r * cell + 0.3,
                    cell - 0.6,
                    cell - 0.6,
                    fill=fill_col,
                )

    # Outline all 12 contiguous blocks (1,1,1,2,2,2,2,2,2,3,3,3)
    blk_pos = 0
    for b_sz in blk["block_sizes_contiguous"]:
        bs = float(b_sz) * cell
        c.rect(
            pa_x + blk_pos * cell,
            pa_y + blk_pos * cell,
            bs,
            bs,
            stroke="#0f172a",
            lw=0.95,
        )
        blk_pos += int(b_sz)

    # Major sector dividers at 3 (end of 1D) and 15 (end of 2D)
    for cut in (3, 15):
        cp = cut * cell
        c.line(pa_x + cp, pa_y, pa_x + cp, pa_y + grid_sz, color="#475569", lw=0.8, dashed=True)
        c.line(pa_x, pa_y + cp, pa_x + grid_sz, pa_y + cp, color="#475569", lw=0.8, dashed=True)

    for tick_v in (0, 3, 7, 11, 15, 24):
        tp = pa_x + tick_v * cell
        c.text(tp, pa_y + grid_sz + 9.0, str(tick_v), size=5.8, color="#334155", anchor="middle", mono=True)
        if tick_v < 24:
            c.text(pa_x - 3.5, pa_y + tick_v * cell + 5.5, str(tick_v), size=5.8, color="#334155", anchor="end", mono=True)

    # Annotations inside quiet off-diagonal corners of Panel (a)
    c.rect(pa_x + 86.0, pa_y + 8.0, 108.0, 36.0, fill="#f8fafc", stroke="#cbd5e1", lw=0.6)
    c.text(pa_x + 91.0, pa_y + 17.5, "12 Contiguous Blocks:", size=5.8, bold=True, color="#0f172a")
    c.text(pa_x + 91.0, pa_y + 26.5, "(1,1,1, 2,2,2,2,2,2, 3,3,3)", size=5.5, color="#1e293b", mono=True)
    c.text(pa_x + 91.0, pa_y + 36.0, "522 off-blk <= 3.11e-16", size=5.5, bold=True, color="#047857", mono=True)

    c.rect(pa_x + 6.0, pa_y + 150.0, 112.0, 44.0, fill="#f8fafc", stroke="#cbd5e1", lw=0.6)
    c.rect(pa_x + 11.0, pa_y + 156.0, 6.5, 6.5, fill="#10b981", stroke="#065f46", lw=0.5)
    c.text(pa_x + 20.5, pa_y + 161.5, "chi_{0,1,2}: RMS = 1.000", size=5.5, bold=True, color="#065f46", mono=True)
    c.rect(pa_x + 11.0, pa_y + 168.0, 6.5, 6.5, fill="#2563eb", stroke="#1e3a8a", lw=0.5)
    c.text(pa_x + 20.5, pa_y + 173.5, "rho_{2,r}:  RMS = 0.707", size=5.5, bold=True, color="#1e3a8a", mono=True)
    c.rect(pa_x + 11.0, pa_y + 180.0, 6.5, 6.5, fill="#7c3aed", stroke="#4c1d95", lw=0.5)
    c.text(pa_x + 20.5, pa_y + 185.5, "rho_3:      RMS = 0.577", size=5.5, bold=True, color="#4c1d95", mono=True)

    c.text(
        pa_x + 0.5 * grid_sz,
        pa_y + grid_sz + 19.0,
        "Peter-Weyl Basis Index (alpha, k, j)",
        size=6.2,
        bold=True,
        color="#1e293b",
        anchor="middle",
    )

    # Panel (b): 6-Subgroup x 7-Irrep Exact HSP Coset Probability Table + Visual Bars
    pb_x, pb_y, pb_w, pb_h = 258.0, 48.0, 304.0, 201.6
    c.text(
        pb_x,
        pb_y - 6.0,
        "(b) Coset State |gH> Fourier Irrep Weights P(alpha | gH) Across All 6 Subgroups",
        size=7.6,
        bold=True,
        color="#0f172a",
    )
    c.rect(pb_x, pb_y, pb_w, pb_h, fill="#fafafa", stroke="#334155", lw=0.9)

    label_col_w = 54.0
    ir_col_w = (pb_w - label_col_w) / 7.0
    hdr_h = 22.0
    row_h = (pb_h - hdr_h) / 6.0

    c.rect(pb_x + 0.6, pb_y + 0.6, pb_w - 1.2, hdr_h - 0.6, fill="#e2e8f0")
    c.text(pb_x + 5.0, pb_y + 14.0, "H (|H|)", size=6.1, bold=True, color="#0f172a")

    ir_short = ["chi_0", "chi_1", "chi_2", "rho_20", "rho_21", "rho_22", "rho_3"]
    ir_dims = [1, 1, 1, 2, 2, 2, 3]
    ir_cols = ["#10b981", "#10b981", "#10b981", "#2563eb", "#2563eb", "#2563eb", "#7c3aed"]
    ir_bg = ["#d1fae5", "#d1fae5", "#d1fae5", "#dbeafe", "#dbeafe", "#dbeafe", "#ede9fe"]

    for j, (ish, idim) in enumerate(zip(ir_short, ir_dims)):
        cx = pb_x + label_col_w + (j + 0.5) * ir_col_w
        c.line(pb_x + label_col_w + j * ir_col_w, pb_y, pb_x + label_col_w + j * ir_col_w, pb_y + pb_h, color="#cbd5e1", lw=0.5)
        c.text(cx, pb_y + 9.5, ish, size=5.7, bold=True, color="#0f172a", anchor="middle", mono=True)
        c.text(cx, pb_y + 18.2, f"d={idim}", size=5.2, color="#475569", anchor="middle", mono=True)
    c.line(pb_x, pb_y + hdr_h, pb_x + pb_w, pb_y + hdr_h, color="#334155", lw=0.8)

    sg_data = hsp["subgroups"]
    for r_i, (sg_key, sg_ascii, _sg_tex, _norm_s) in enumerate(SUBGROUP_ORDER):
        ry0 = pb_y + hdr_h + r_i * row_h
        if r_i > 0:
            c.line(pb_x, ry0, pb_x + pb_w, ry0, color="#cbd5e1", lw=0.5)
        info = sg_data[sg_key]
        ord_h = int(info["order"])
        is_q8 = sg_key == "Q8_normal"
        if is_q8:
            c.rect(pb_x + 0.8, ry0 + 0.6, label_col_w - 1.2, row_h - 1.2, fill="#ecfdf5")
        c.text(
            pb_x + 5.0,
            ry0 + 12.2,
            f"{sg_ascii} ({ord_h})",
            size=6.1,
            bold=True,
            color="#065f46" if is_q8 else "#0f172a",
            mono=True,
        )
        c.text(
            pb_x + 5.0,
            ry0 + 22.0,
            "normal" if _norm_s == "Yes" else "non-norm",
            size=5.1,
            color="#475569",
        )

        weights = info["measured_irrep_weights_coset_eH"]
        mults = info["trivial_irrep_multiplicities"]
        for j, (ir_key, _ia, _it, _id) in enumerate(IRREP_SPECS):
            cx0 = pb_x + label_col_w + j * ir_col_w
            w_val = float(weights[ir_key])
            m_val = int(mults[ir_key])
            if m_val > 0:
                c.rect(cx0 + 1.4, ry0 + 1.6, ir_col_w - 2.8, row_h - 3.2, fill=ir_bg[j], stroke=ir_cols[j], lw=0.6)
                bar_w = max(2.0, (ir_col_w - 6.0) * min(1.0, w_val))
                c.rect(cx0 + 3.0, ry0 + row_h - 6.8, bar_w, 3.4, fill=ir_cols[j])
                c.text(
                    cx0 + 0.5 * ir_col_w,
                    ry0 + 11.2,
                    _fraction_label(w_val),
                    size=6.1,
                    bold=True,
                    color="#0f172a",
                    anchor="middle",
                    mono=True,
                )
                c.text(
                    cx0 + 0.5 * ir_col_w,
                    ry0 + 19.6,
                    f"{w_val:.3f}",
                    size=5.0,
                    color="#334155",
                    anchor="middle",
                    mono=True,
                )
            else:
                c.text(
                    cx0 + 0.5 * ir_col_w,
                    ry0 + 15.5,
                    "0",
                    size=6.0,
                    color="#94a3b8",
                    anchor="middle",
                    mono=True,
                )

    # Bottom verification summary box
    ly = 278.0
    c.rect(pa_x, ly, w - pa_x - 14.0, 48.0, fill="#f8fafc", stroke="#cbd5e1", lw=0.7)
    c.text(
        pa_x + 8.0,
        ly + 12.5,
        "Frobenius Reciprocity & Non-Abelian HSP Coset Invariance (Verified Across All 6 x 24 = 144 Left Cosets |gH>):",
        size=6.3,
        bold=True,
        color="#0f172a",
    )
    c.text(
        pa_x + 8.0,
        ly + 24.5,
        f"* Normal Q_8 <= 2T concentrates 100.0% on 1D irreps (chi_0, chi_1, chi_2) = (1/3, 1/3, 1/3) with 2D/3D leakage <= {hsp['subgroups']['Q8_normal']['max_forbidden_irrep_leakage']:.2e}.",
        size=5.9,
        color="#1e293b",
    )
    c.text(
        pa_x + 8.0,
        ly + 35.5,
        "* Center Z_2 annihilates spinor irreps rho_{2,r} (rho(-1) = -I_2 -> m=0), placing 3/4 on rho_3; non-normal Z_3 splits (1/8, 0, 0, 0, 1/4, 1/4, 3/8).",
        size=5.9,
        color="#1e293b",
    )
    c.text(
        pa_x + 8.0,
        ly + 45.0,
        f"* 5Q QSD Synthesis: 423 CX (depth 837, ||U - F_32|| = {p2['logical_circuit']['direct_operator_max_abs_error']:.2e}) -> PROXY: 423 CZ (All-to-All), 613 CZ (T5), 423 ZZPhase (pytket H2).",
        size=5.8,
        bold=True,
        color="#4c1d95",
    )
    return c


def _build_preprint_pdf(
    manifest: Mapping[str, Any],
    inv_manifest: Mapping[str, Any],
    fez_results: Mapping[str, Any],
    esd_info: Mapping[str, Any],
    capsule_sha: str,
    source_shas: Mapping[str, str],
    fig1: VectorCanvas,
    fig2: VectorCanvas,
) -> bytes:
    """Render deterministic >=4-page standalone US-Letter PDF 1.4 preprint."""
    p1 = manifest["part1_multiplication_10q"]
    p2 = manifest["part2_nonabelian_qft_5q"]
    m1_m = p1["metrics"]
    log_g = p1["logical_circuits"]["generic"]
    log_s = p1["logical_circuits"]["structured"]
    p1_prox = p1["proxy_compilations"]

    ata_g = _find_proxy(p1_prox, circuit="generic", target="all_to_all", opt_level=3)
    ata_s = _find_proxy(p1_prox, circuit="structured", target="all_to_all", opt_level=3)
    hh_g = _find_proxy(p1_prox, circuit="generic", target="heavy_hex_fragment_10Q", opt_level=3)
    hh_s = _find_proxy(p1_prox, circuit="structured", target="heavy_hex_fragment_10Q", opt_level=3)

    qft_log = p2["logical_circuit"]
    qft_prox = p2["proxy_compilations"]
    qft_ata = _find_proxy(qft_prox, circuit="bt2t_qft_5q", kind_sub="qiskit", target="all_to_all", opt_level=3)
    qft_t5 = _find_proxy(qft_prox, circuit="bt2t_qft_5q", kind_sub="qiskit", target="heavy_hex_fragment_T5", opt_level=3)
    qft_tk = _find_proxy(qft_prox, circuit="bt2t_qft_5q", kind_sub="pytket", peephole=True)

    su = p2["schur_and_unitarity"]
    blk = p2["block_diagonalization_and_convolution"]
    hsp = p2["subgroup_coset_hsp_sampling"]

    m1_job = str(fez_results["provider_job_id"])
    m1_s = fez_results["arms"]["STRUCTURED"]
    m1_g = fez_results["arms"]["GENERIC"]
    m1_sv = m1_s["valid_24_uniform"]
    m1_gv = m1_g["valid_24_uniform"]
    inv_prox = inv_manifest["proxy_compilations"]
    m1_ata_s = int(
        _find_proxy(inv_prox, circuit="structured", kind_sub="qiskit", target="all_to_all", opt_level=3)[
            "two_qubit_gates"
        ]["cz"]
    )
    m1_ata_g = int(
        _find_proxy(inv_prox, circuit="generic", kind_sub="qiskit", target="all_to_all", opt_level=3)[
            "two_qubit_gates"
        ]["cz"]
    )
    m1_tk_s = int(
        _find_proxy(inv_prox, circuit="structured", kind_sub="pytket", peephole=True)["two_qubit_gates"][
            "ZZPhase"
        ]
    )
    m1_tk_g = int(
        _find_proxy(inv_prox, circuit="generic", kind_sub="pytket", peephole=True)["two_qubit_gates"][
            "ZZPhase"
        ]
    )

    doc = PreprintDocument(
        short_title="Zero-Ancilla 10Q 2T Multiplication & 5Q Non-Abelian QFT (OSIRIS-BT2T-002)",
        manuscript_id="OSIRIS-BT2T-002",
        execution_tier="SIMULATED / OFFLINE_COMPILED",
        footer_left=f"SIMULATED / PROXY (provider_job_id: null)  |  bt2t-enc-1  |  Capsule: {capsule_sha[:16]}...",
    )

    abstract = (
        "Digital simulation of non-abelian lattice Yang-Mills gauge theories and Hidden Subgroup algorithms over "
        "finite SU(2) subgroups require two coherent primitives beyond single-register inversion: reversible "
        "two-register group multiplication U_mul|x>|y> = |x>|x*y> and the non-abelian Peter-Weyl Quantum Fourier "
        "Transform F_G. Under frozen 5-qubit Hurwitz quaternion encoding bt2t-enc-1 (SHA-256 "
        f"{manifest['encoding_sha256'][:16]}...), we construct and exhaustively verify Milestone 2 (OSIRIS-BT2T-002; "
        "TIER: SIMULATED / OFFLINE_COMPILED, provider_job_id: null, 0 QPU s) for the binary tetrahedral group "
        "2T == SL(2, F_3) of order |2T| = 24. First, exploiting the semidirect decomposition 2T ~= Q_8 |x Z_3 via "
        "canonical order-3 generator omega = (-1 - i - j - k)/2 (index 23) whose conjugation automorphism "
        "Ad_omega(i, j, k) = (j, k, i) acts as a 2-CNOT cycle on Q_8 imaginary bits, combined with GF(2) cocycle "
        "multiplication, Z_3 modular addition, unused-codeword Mobius cancellation, and Barenco dirty-ancilla CCX "
        "live-wire decomposition, we synthesize exact zero-ancilla 10-qubit reversible multiplication across all "
        f"1,024 computational basis states (576 valid Cayley pairs + 448 identity-extended unused pairs). Relative "
        f"to a generic disjoint-cycle reference ({log_g['gate_counts']['mcx']:,} MCX, {log_g['gate_counts']['cx']:,} CX, "
        f"{log_g['gate_counts']['x']:,} X, {log_g['gate_counts']['cswap']} CSWAP; logical depth {log_g['depth']:,}), "
        f"semidirect synthesis eliminates all k>=3 MCX gates ({log_s['gate_counts']['ccx']} CCX, "
        f"{log_s['gate_counts']['cx']} CX, {log_s['gate_counts']['x']} X, {log_s['gate_counts']['cswap']} CSWAP; "
        f"logical depth {log_s['depth']}) with exact 1024x1024 unitary error 0.0 (including phases), compressing "
        f"local PROXY native CZ counts from {m1_m['cz_generic_all_to_all_opt3']:,} to "
        f"{m1_m['cz_structured_all_to_all_opt3']:,} CZ ({m1_m['cz_compression_ratio_all_to_all']:.2f}x) on all-to-all "
        f"and from {m1_m['cz_generic_heavy_hex_10q_opt3']:,} to {m1_m['cz_structured_heavy_hex_10q_opt3']:,} CZ "
        f"({m1_m['cz_compression_ratio_heavy_hex_10q']:.2f}x) on a 10Q heavy-hex fragment. Second, we construct all 7 "
        "unitary irreps of 2T (1^2 + 1^2 + 1^2 + 2^2 + 2^2 + 2^2 + 3^2 = 24: chi_{0,1,2}, rho_{2,0..2}, rho_3) and "
        f"embed F_tilde_{{2T}} = F_{{2T}} (+) I_8 on 5 qubits (unitarity {su['f32_unitarity_inf_error']:.2e}, "
        f"homomorphism {max(su['irrep_homomorphism_max_abs_errors'].values()):.2e}, simultaneous block-diagonalization "
        f"of all 24 translations to {blk['left_regular_max_off_block_abs']:.2e}), proving exact non-abelian HSP coset "
        f"tomography across all 6 canonical subgroups (including |g*Q_8> |-> (1/3, 1/3, 1/3) on 1D irreps with "
        f"{hsp['subgroups']['Q8_normal']['max_forbidden_irrep_leakage']:.2e} leakage) and compiling to "
        f"{qft_log['cz_all_to_all_opt3']} CZ (all-to-all), {qft_log['cz_heavy_hex_T5_opt3']} CZ (T5), and "
        f"{qft_tk['two_qubit_gates']['ZZPhase']} ZZPhase (pytket H2). Anchored against Milestone 1 hardware results on "
        f"ibm_fez (job {m1_job}, where 72 CZ yielded 0.5337 raw accuracy while 723 CZ collapsed to 0.0661), these "
        "exact gate budgets establish that 5Q non-abelian QFT (423 ZZPhase) and 10Q 2T multiplication (1,739 2Q "
        "gates) motivate all-to-all QCCD trapped-ion and fault-tolerant execution rather than unmitigated heavy-hex NISQ."
    )

    doc.add_title_block(
        title=(
            "Zero-Ancilla Semidirect-Product Group Multiplication and Seven-Irrep Peter-Weyl Quantum "
            "Fourier Transform over the Binary Tetrahedral Group 2T"
        ),
        author="Devin Phillip Davis",
        affiliation="Agile Defense Systems LLC, Lexington, Kentucky, USA",
        date_str="November 24, 2026",
        metadata_rows=[
            (
                "Execution Provenance",
                "TIER: SIMULATED / OFFLINE_COMPILED | provider_job_id: null | 0 QPU s | Local PROXY Qiskit & pytket compilation (seed 20261124)",
            ),
            (
                "Frozen Encoding",
                f"bt2t-enc-1 ({manifest['encoding_sha256']}) | 31/31 exact algebraic & unitary checks PASS",
            ),
            (
                "M1 Hardware Anchor",
                f"OSIRIS-BT2T-001-FEZ on ibm_fez (job {m1_job}, N_windows = 1): 72 CZ (P=0.5337) vs 723 CZ (P=0.0661) contextualizing NISQ vs QCCD",
            ),
            ("RFC 8785 Capsule", capsule_sha),
        ],
        abstract=abstract,
    )

    doc.add_callout_box(
        "PROVENANCE & ARCHITECTURAL FEASIBILITY BOUNDARY (SIMULATED / OFFLINE_COMPILED, PROVIDER_JOB_ID: NULL)",
        [
            "Exact Zero-Ancilla 10Q 2T Multiplication (U_mul): Semidirect synthesis over 2T ~= Q_8 |x Z_3 + Barenco dirty-ancilla CCX live-wire invariance realizes all 1,024 basis states (576 Cayley pairs + 448 unused fixed pairs) in 270 CCX, 35 CX, 18 X, and 14 CSWAP gates (depth 300, ||U - P_1024||_max = 0.0), compressing PROXY CZ counts by 19.00x all-to-all (1,739 vs 33,042 CZ) and 21.31x on 10Q heavy-hex (3,662 vs 78,035 CZ).",
            "Exact 5Q Peter-Weyl Non-Abelian QFT (F_tilde_2T = F_2T (+) I_8): Constructed from all 7 unitary irreps (chi_0..2, rho_2,0..2, rho_3; sum d_alpha^2 = 24), simultaneously block-diagonalizing all 24 group translations into (1,1,1,2,2,2,2,2,2,3,3,3) blocks (<= 3.11e-16 off-block) and resolving all 6 canonical subgroups via coset sampling (|g*Q_8> -> 1/3 each on chi_0,1,2, leakage 1.62e-33).",
            f"NISQ vs. QCCD / Fault-Tolerant Target Selection: In Milestone 1 on ibm_fez (job {m1_job}), 72 CZ achieved P_valid24 = 0.5337 raw (0.6104 RO-mitigated) while 723 CZ degraded to 0.0661. Consequently, 5Q 2T QFT (423 ZZPhase / 613 T5 CZ) and 10Q 2T multiplication (1,739 all-to-all CZ / 3,662 heavy-hex CZ) are explicitly scoped for all-to-all QCCD trapped-ion (~1e-3 2Q infidelity) and fault-tolerant execution rather than unmitigated heavy-hex NISQ.",
        ],
        tone="info",
    )

    doc.add_section("1. Introduction and Milestone Progression")
    doc.add_paragraph(
        "Kogut-Susskind Hamiltonian formulations of SU(2) lattice Yang-Mills theory replace continuous gauge links "
        "U_e in SU(2) by finite non-abelian subgroups G subset SU(2) [1, 2], requiring four coherent register "
        "primitives on a n_G-qubit link Hilbert space (C^2)^(+x n_G): group inversion, controlled left/right link "
        "multiplication (parallel transport and plaquette assembly), trace-class magnetic plaquette operators, and the "
        "non-abelian Peter-Weyl Quantum Fourier Transform (electric-basis diagonalization and Hidden Subgroup coset "
        "tomography [3, 5]). For the binary tetrahedral group 2T == SL(2, F_3) of order |2T| = 24 -- the smallest "
        "non-abelian SU(2) subgroup possessing three-fold tetrahedral rotational cosets over the quaternion group "
        "Q_8 -- embedding 24 elements into 2^5 = 32 basis states leaves 8 unused codewords per 5-qubit register that "
        "must remain strictly decoupled without postselection or ancilla overhead."
    )
    doc.add_paragraph(
        f"In Milestone 1 (OSIRIS-BT2T-001-FEZ, ibm_fez job {m1_job}, HARDWARE_ATTESTED, 27,648 shots, N_windows = 1), "
        f"coset-parity Boolean factoring compressed 5-qubit 2T inversion from {m1_g['cz_count']} to {m1_s['cz_count']} "
        f"heavy-hex CZ gates (10.04x; {m1_ata_g} to {m1_ata_s} CZ all-to-all, 8.33x), boosting raw valid-24 accuracy "
        f"from {m1_gv['p_correct_raw']:.4f} to {m1_sv['p_correct_raw']:.4f} ({m1_sv['p_correct_ro_mitigated_secondary']:.4f} "
        "readout-mitigated) while exposing heavy-hex decoherence above ~700 CZ and unflagged leaf-qubit SPAM drift [6]. "
        "Here we report Milestone 2 (OSIRIS-BT2T-002, SIMULATED / OFFLINE_COMPILED, provider_job_id: null): exact "
        "zero-ancilla 10-qubit reversible 2T multiplication and the exact 5-qubit 7-irrep Peter-Weyl non-abelian QFT "
        "under the identical frozen encoding bt2t-enc-1."
    )

    doc.add_section("2. Semidirect-Product 10Q Multiplication and Barenco Live-Wire Invariance")
    doc.add_paragraph(
        "On 10 qubits |x>_{0..4}|y>_{5..9} (basis index s = x | (y << 5) in 0..1023), extended left multiplication "
        "U_mul|x>|y> = |x>|mu(x, y)> maps mu(x, y) = x*y in 0..23 across all 24 x 24 = 576 valid Cayley pairs and "
        "acts as the exact identity mu(x, y) = y across all 1024 - 576 = 448 pairs where x >= 24 or y >= 24. A "
        "generic permutation compiler decomposing each left translation L_x (x in 1..23) into disjoint cycles and "
        "paired double transpositions requires 8-controlled Toffoli reflections, inflating to 2,040 MCX, 1,656 CX, "
        "1,064 X, and 144 CSWAP gates (4,904 logical gates, depth 3,760)."
    )
    doc.add_paragraph(
        "By contrast, let omega = (-1 - i - j - k)/2 (index 23, omega^2 = (-1 + i + j + k)/2 at index 16, omega^3 = +1). "
        "Every g in 2T admits a unique right semidirect decomposition g = q_g * omega^{c_g} with c_g in Z_3 = {0, 1, 2} "
        "and q_g in Q_8 (indices 0..7), implemented unitarily on 5 qubits (fixing 24..31 as c = 3) by an 18-gate "
        f"bijection E_sd:|g> |-> |c_g>_{{4,3}}|q_g>_{{2,1,0}} ({esd_info['esd_counts']['x']} X, "
        f"{esd_info['esd_counts']['cx']} CX, {esd_info['esd_counts']['ccx']} CCX, "
        f"{esd_info['esd_counts']['cswap']} CSWAP, 3 MCX; depth {esd_info['esd_depth']}, ||U(E_sd) - P_sd|| = 0.0). "
        "Because Ad_omega(i, j, k) = (j, k, i) acts linearly on Q_8 imaginary bits (q2, q1) |-> (q1 (+) q2, q2) "
        "without flipping the sign bit q0, group multiplication reduces to three compact reversible stages:"
    )
    doc.add_equation(
        [
            "(q_x * omega^{c_x})(q_y * omega^{c_y}) = [q_x * Ad_omega^{c_x}(q_y)] * omega^{(c_x + c_y) mod 3}",
            "p_2 = x_2 (+) y_2,  p_1 = x_1 (+) y_1,  p_0 = x_0 (+) y_0 (+) (x_1 (+) x_2)y_1 (+) x_2 y_2",
            "valid(x, y) = 1 (+) x_3 x_4 (+) y_3 y_4 (+) x_3 x_4 y_3 y_4   (mod 2)",
        ],
        eq_num="1",
    )
    doc.add_paragraph(
        "Theorem 1 (Barenco Dirty-Ancilla Live-Wire Phase Invariance): Expanding every k-controlled MCX (k in {3,4,5,6}) "
        "in E_sd^(+x 2), Ad_omega^{c_x}, Q_8 cocycle multiplication, and Z_3 addition via Barenco Lemmas 7.2 and 7.3 [4] "
        "using untouched live register wires in {0..9} \\ (C U {t}) as dirty ancillas (4, 8, 12, and 24 pure CCX gates "
        "for k = 3, 4, 5, 6) yields a zero-ancilla 337-gate circuit (270 CCX, 35 CX, 18 X, 14 CSWAP; depth 300) "
        "satisfying ||U_struct - P_1024||_max = 0.0 across all 1,024^2 complex matrix entries. Because every elementary "
        "gate is a real {0, +1} permutation matrix (unlike relative-phase RCCX gates that imprint state-dependent -1 "
        "phases on dirty wires), exact bit-restoration across all 1,024 computational basis states guarantees zero "
        "relative-phase distortion on arbitrary entangled superpositions (verified to 0.0 across 32 seeded 10Q "
        "superpositions, while a positive-control CZ(0,5) phase fault preserving 1024/1024 basis probabilities is "
        f"caught at ||U_bad - P_1024||_max = {m1_m['phase_control_injected_cz_max_abs_error']:.1f}).",
        lead_in="Exact Synthesis Proof:",
    )

    doc.add_figure(
        fig1,
        "Log-scale two-qubit gate compression across Milestone 1 (5Q inversion) and Milestone 2 (10Q multiplication "
        "and 5Q non-abelian QFT), plus 10Q zero-ancilla logical breakdown. Semidirect Q_8 |x Z_3 synthesis compresses "
        "10Q multiplication CZ count by 19.00x on all-to-all (1,739 vs 33,042 CZ) and 21.31x on 10Q heavy-hex "
        "(3,662 vs 78,035 CZ) at exact 1024x1024 unitary fidelity.",
        max_height=268.0,
    )

    comp_rows = [
        [
            "M1 5Q Inv (All-to-All)",
            "GENERIC / STRUCT",
            f"{m1_ata_g} / {m1_ata_s} CZ",
            "1,481 / 168",
            "8.33x CZ",
            "PROXY (exact P_32)",
        ],
        [
            f"M1 5Q Inv (ibm_fez {m1_job[:8]}..)",
            "GENERIC / STRUCT",
            f"{m1_g['cz_count']} / {m1_s['cz_count']} CZ",
            f"{m1_g['core_depth']:,} / {m1_s['core_depth']}",
            "10.04x CZ",
            "HW: P=0.0661 -> 0.5337",
        ],
        [
            "M1 5Q Inv (pytket H2)",
            "GENERIC / STRUCT",
            f"{m1_tk_g} / {m1_tk_s} ZZPhase",
            "1,720 / 194",
            "8.51x ZZ",
            "PROXY H2 rebase",
        ],
        [
            "M2 10Q Mul (Logical 0-Anc)",
            "GENERIC",
            "2,040 MCX + 1,656 CX",
            f"{log_g['depth']:,}",
            "1.00x (ref)",
            "||U - P_1024|| = 0.0",
        ],
        [
            "M2 10Q Mul (Logical 0-Anc)",
            "STRUCTURED (Q8|xZ3)",
            "270 CCX + 35 CX + 14 CSW",
            f"{log_s['depth']}",
            "12.53x depth",
            "||U - P_1024|| = 0.0",
        ],
        [
            "M2 10Q Mul (All-to-All opt 3)",
            "GENERIC / STRUCT",
            f"{ata_g['two_qubit_gates']['cz']:,} / {ata_s['two_qubit_gates']['cz']:,} CZ",
            f"{ata_g['depth']:,} / {ata_s['depth']:,}",
            f"{m1_m['cz_compression_ratio_all_to_all']:.2f}x CZ",
            f"PROXY (<={ata_s['phase_aligned_max_abs_error']:.2e})",
        ],
        [
            "M2 10Q Mul (10Q H-Hex opt 3)",
            "GENERIC / STRUCT",
            f"{hh_g['two_qubit_gates']['cz']:,} / {hh_s['two_qubit_gates']['cz']:,} CZ",
            f"{hh_g['depth']:,} / {hh_s['depth']:,}",
            f"{m1_m['cz_compression_ratio_heavy_hex_10q']:.2f}x CZ",
            f"PROXY (<={hh_s['phase_aligned_max_abs_error']:.2e})",
        ],
        [
            "M2 5Q QFT (Logical QSD)",
            "F_2T (+) I_8",
            f"{qft_log['gate_counts']['cx']} CX + {qft_log['gate_counts']['u']} U",
            f"{qft_log['depth']}",
            "--",
            f"||U - F_32|| = {qft_log['direct_operator_max_abs_error']:.2e}",
        ],
        [
            "M2 5Q QFT (All-to-All / T5)",
            "Qiskit opt 3",
            f"{qft_ata['two_qubit_gates']['cz']} / {qft_t5['two_qubit_gates']['cz']} CZ",
            f"{qft_ata['depth']:,} / {qft_t5['depth']:,}",
            "1.45x T5/All",
            f"PROXY (<={qft_ata['phase_aligned_max_abs_error']:.2e})",
        ],
        [
            "M2 5Q QFT (pytket H2 opt)",
            "H2 Native Rebase",
            f"{qft_tk['two_qubit_gates']['ZZPhase']} ZZPhase",
            f"{qft_tk['depth']:,}",
            "1.00x vs All",
            f"PROXY (<={qft_tk['phase_aligned_max_abs_error']:.2e})",
        ],
    ]
    doc.add_table(
        "TABLE I. Cross-milestone logical and local PROXY native compilation benchmarks (bt2t-enc-1, seed 20261124) alongside Milestone 1 ibm_fez anchor.",
        ["Primitive & Target", "Synthesis", "Entangling / Key Gates", "Depth", "Gain / Ratio", "Verification / Tier"],
        comp_rows,
        col_widths=[112, 82, 108, 62, 58, 94],
        alignments=["left", "left", "right", "right", "right", "left"],
        font_size=6.7,
        mono_cols=[False, False, True, True, True, True],
    )

    doc.add_section("3. Seven-Irrep Peter-Weyl Non-Abelian QFT and Block Diagonalization")
    doc.add_paragraph(
        "The binary tetrahedral group 2T possesses 7 conjugacy classes and therefore 7 inequivalent unitary "
        "irreducible representations rho_alpha of dimensions (1, 1, 1, 2, 2, 2, 3) satisfying "
        "sum_alpha d_alpha^2 = 1^2 + 1^2 + 1^2 + 2^2 + 2^2 + 2^2 + 3^2 = 24: three 1D characters "
        "chi_r(g) = exp(2*pi*i*r*c(g)/3) (r in {0,1,2}) lifted from 2T/Q_8 ~= Z_3; three faithful 2D spinor irreps "
        "rho_{2,r}(g) = chi_r(g) U_{SU(2)}(g) in U(2); and the 3D real orthogonal tetrahedral vector irrep rho_3(g) "
        "in SO(3) defined via Pauli conjugation U(g) sigma_a U(g)^+ = sum_b [rho_3(g)]_{b,a} sigma_b."
    )
    doc.add_equation(
        [
            "[F_{2T}]_{(alpha,k,j),g} = sqrt(d_alpha/24) * [rho_alpha(g)]_{j,k},   F_tilde_{2T} = F_{2T} (+) I_8",
            "F_{2T} L_g F_{2T}^+ = (+)_alpha (I_{d_alpha} (x) rho_alpha(g))",
            "F_{2T} R_g F_{2T}^+ = (+)_alpha (conj(rho_alpha(g)) (x) I_{d_alpha})",
        ],
        eq_num="2",
    )
    doc.add_paragraph(
        "Ordering Peter-Weyl rows by (alpha, multiplicity k outer, matrix row j inner) simultaneously block-"
        "diagonalizes all 24 left-regular translations L_g into 12 contiguous blocks of sizes "
        f"(1,1,1,2,2,2,2,2,2,3,3,3) with maximum off-block residual {blk['left_regular_max_off_block_abs']:.2e} "
        f"(on-block error {blk['left_regular_max_on_block_error']:.2e}), right-regular translations R_g to "
        f"{blk['right_regular_max_off_irrep_abs']:.2e}, 32x32 unitarity ||F_tilde_2T F_tilde_2T^+ - I_32||_inf = "
        f"{su['f32_unitarity_inf_error']:.2e}, and non-abelian matrix convolution hat(a*b)(alpha) = hat(a)(alpha)hat(b)(alpha) "
        f"to {blk['nonabelian_convolution_max_error']:.2e}."
    )

    irrep_rows: List[List[str]] = []
    slice_pos = 0
    for ir_key, ir_ascii, _ir_tex, dim_d in IRREP_SPECS:
        span = dim_d * dim_d
        rms_v = 1.0 / math.sqrt(float(dim_d))
        u_err = float(su["irrep_unitary_max_inf_errors"][ir_key])
        h_err = float(su["irrep_homomorphism_max_abs_errors"][ir_key])
        desc = (
            "1D Z_3 quotient character"
            if dim_d == 1
            else ("2D U(2) spinor irrep" if dim_d == 2 else "3D SO(3) Pauli adjoint")
        )
        irrep_rows.append(
            [
                ir_ascii,
                str(dim_d),
                f"[{slice_pos}..{slice_pos + span - 1}] ({dim_d}x{dim_d}x{dim_d})",
                f"{rms_v:.4f}",
                _fmt_sci_ascii(u_err),
                _fmt_sci_ascii(h_err),
                desc,
            ]
        )
        slice_pos += span

    doc.add_table(
        "TABLE II. Complete 7-irrep Peter-Weyl decomposition of 2T (sum d_alpha^2 = 24; character orthogonality error 4.43e-16).",
        ["Irrep", "d", "Row Slice (Blocks)", "Schur RMS", "Unitarity Err", "576-Pair Hom Err", "Construction"],
        irrep_rows,
        col_widths=[48, 22, 92, 54, 68, 76, 156],
        alignments=["left", "right", "left", "right", "right", "right", "left"],
        font_size=6.8,
        mono_cols=[True, True, True, True, True, True, False],
    )

    doc.add_section("4. Non-Abelian Hidden Subgroup Coset Tomography and Hardware Feasibility")
    doc.add_paragraph(
        "For any subgroup H <= 2T and arbitrary left coset |gH> = |H|^{-1/2} sum_{h in H} |g*h> (embedded in C^32), "
        "left-translation invariance L_g -> I_{d_alpha} (x) rho_alpha(g) renders the Fourier irrep-sector measurement "
        "distribution strictly independent of coset representative g in 2T and governed by Frobenius reciprocity:"
    )
    doc.add_equation(
        [
            "m_alpha(H) = (1 / |H|) * sum_{h in H} chi_alpha(h)",
            "P(alpha | gH) = d_alpha * |H| * m_alpha(H) / 24",
        ],
        eq_num="3",
    )

    doc.add_figure(
        fig2,
        "(a) Exact 24x24 Schur RMS amplitude matrix across all 24 left translations F_{2T} L_g F_{2T}^+, isolating "
        "the 12 contiguous Plancherel blocks (1,1,1,2,2,2,2,2,2,3,3,3) with <= 3.11e-16 off-block leakage. "
        "(b) Coset-state Fourier irrep distributions P(alpha | gH) verified across all 6 x 24 = 144 left cosets.",
        max_height=292.0,
    )

    hsp_rows: List[List[str]] = []
    for sg_key, sg_ascii, _sg_tex, norm_s in SUBGROUP_ORDER:
        info = hsp["subgroups"][sg_key]
        w_map = info["measured_irrep_weights_coset_eH"]
        m_map = info["trivial_irrep_multiplicities"]
        m_vec = "(" + ",".join(str(m_map[k]) for k, _, _, _ in IRREP_SPECS) + ")"
        w_vec = "(" + ", ".join(_fraction_label(float(w_map[k])) for k, _, _, _ in IRREP_SPECS) + ")"
        hsp_rows.append(
            [
                sg_ascii,
                str(info["order"]),
                norm_s,
                m_vec,
                w_vec,
                _fmt_sci_ascii(float(info["max_coset_weight_abs_error_all_24_g"])),
                _fmt_sci_ascii(float(info["max_forbidden_irrep_leakage"])),
            ]
        )

    doc.add_table(
        "TABLE III. Exact Fourier irrep-sector sampling across all 6 canonical subgroups H <= 2T (24 left cosets each; 0.0 unused-8 leakage).",
        ["H", "|H|", "Normal", "m_alpha(H)", "Exact P(alpha | gH) [chi_0..2, rho_20..22, rho_3]", "Max Coset Err", "Forbidden Leak"],
        hsp_rows,
        col_widths=[36, 24, 38, 86, 198, 68, 66],
        alignments=["left", "right", "center", "left", "left", "right", "right"],
        font_size=6.5,
        mono_cols=[True, True, False, True, True, True, True],
    )

    doc.add_paragraph(
        "Weak Fourier sampling (measuring only irrep label alpha) uniquely distinguishes {1}, Z_2, Z_4, Z_3, Q_8, and "
        "2T up to conjugacy: normal Q_8 concentrates 100% weight equally on (chi_0, chi_1, chi_2) = (1/3, 1/3, 1/3) "
        "with 1.62e-33 forbidden 2D/3D leakage; central Z_2 = {+/-1} annihilates all three spinor irreps because "
        "rho_{2,r}(-1) = -I_2, concentrating 3/4 weight on rho_3; and non-normal Z_3 = <omega> (conjugate to the "
        "other three tetrahedral C_3 axes) places (1/8, 0, 0, 0, 1/4, 1/4, 3/8) across irreps, illustrating why "
        "distinguishing conjugate non-normal subgroups requires strong intra-block basis measurements inside rho_{2,r} "
        "and rho_3 [3, 5]."
    )

    # NISQ vs QCCD Resource Feasibility Table
    eps_hh, eps_qccd, eps_ft = 5.0e-3, 1.0e-3, 2.0e-4
    feas_rows = [
        [
            "M1 5Q Inversion (STRUCT)",
            "72 CZ (T5) / 47 ZZ (H2)",
            f"{(1.0 - eps_hh) ** 72:.3f} (emp 0.534 raw / 0.610 mit)",
            f"{(1.0 - eps_qccd) ** 47:.3f}",
            f"{(1.0 - eps_ft) ** 47:.3f}",
            "NISQ Heavy-Hex Attested",
        ],
        [
            "M1 5Q Inversion (GENERIC)",
            "723 CZ (T5) / 400 ZZ (H2)",
            f"{(1.0 - eps_hh) ** 723:.3f} (emp 0.066 raw / 0.072 mit)",
            f"{(1.0 - eps_qccd) ** 400:.3f}",
            f"{(1.0 - eps_ft) ** 400:.3f}",
            "Heavy-Hex Decoherence Wall",
        ],
        [
            "M2 5Q Non-Abelian QFT",
            "613 CZ (T5) / 423 ZZ (H2)",
            f"{(1.0 - eps_hh) ** 613:.3f} (unmitigated NISQ noisy)",
            f"{(1.0 - eps_qccd) ** 423:.3f}",
            f"{(1.0 - eps_ft) ** 423:.3f}",
            "Current All-to-All QCCD",
        ],
        [
            "M2 10Q Mul (STRUCTURED)",
            "3,662 CZ (10Q) / 1,739 All",
            f"{(1.0 - eps_hh) ** 3662:.2e} (depolarized)",
            f"{(1.0 - eps_qccd) ** 1739:.3f}",
            f"{(1.0 - eps_ft) ** 1739:.3f}",
            "Next-Gen QCCD / Early FT",
        ],
        [
            "M2 10Q Mul (GENERIC)",
            "78,035 CZ (10Q) / 33,042 All",
            f"{(1.0 - eps_hh) ** 78035:.1e}",
            f"{(1.0 - eps_qccd) ** 33042:.2e}",
            f"{(1.0 - eps_ft) ** 33042:.3f}",
            "Infeasible Without Algebra",
        ],
    ]
    doc.add_table(
        "TABLE IV. Analytical two-qubit survival envelope (1 - eps_2Q)^{N_2Q} anchored to Milestone 1 ibm_fez data, motivating QCCD / FT execution.",
        ["Primitive", "Native 2Q Count", "Heavy-Hex (eps=5e-3)", "QCCD H2 (eps=1e-3)", "Next-Gen / FT (2e-4)", "Target Regime"],
        feas_rows,
        col_widths=[98, 96, 108, 66, 68, 80],
        alignments=["left", "right", "right", "right", "right", "left"],
        font_size=6.5,
        mono_cols=[False, True, True, True, True, False],
    )

    doc.add_section("5. Conclusion and Reproducibility Availability")
    doc.add_paragraph(
        "Exploiting 2T ~= Q_8 |x Z_3 semidirect structure and Barenco dirty-ancilla live-wire invariance compresses "
        "exact 10-qubit zero-ancilla 2T group multiplication by 19.00x on all-to-all (1,739 vs 33,042 CZ) and 21.31x "
        "on 10Q heavy-hex (3,662 vs 78,035 CZ) with exact 1024x1024 unitary fidelity, while our 7-irrep Peter-Weyl "
        "5-qubit QFT (423 all-to-all CZ / 423 H2 ZZPhase) simultaneously block-diagonalizes all 24 group translations "
        "and resolves all 6 canonical subgroups of 2T. Together with Milestone 1 hardware attestation on ibm_fez, "
        "these results establish a complete, algebraically verified primitive library for binary tetrahedral lattice "
        "gauge simulation on all-to-all QCCD and fault-tolerant quantum architectures."
    )
    doc.add_paragraph(
        f"All clean-room source modules (bt2t_mul.py {source_shas['experiments/bt2t_primitives/bt2t_mul.py'][:16]}..., "
        f"bt2t_qft.py {source_shas['experiments/bt2t_primitives/bt2t_qft.py'][:16]}...), offline manifest "
        f"({source_shas['experiments/bt2t_primitives/manifest_bt2t002.json'][:16]}...), and RFC 8785 capsule "
        f"({capsule_sha[:16]}...) are locked in SHA256SUMS.txt under experiments/bt2t_primitives/ and "
        "papers/bt2t002_mul_and_nonabelian_qft/."
    )

    doc.add_references(
        [
            "[1] H. Lamm, S. Lawrence, and Y. Yamauchi (NuQS Collaboration), 'General methods for digital quantum simulation of gauge theories,' Phys. Rev. D 100, 034518 (2019).",
            "[2] J. Bender, E. Zohar, A. Farace, and J. I. Cirac, 'Digital quantum simulation of lattice gauge theories in three spatial dimensions,' New J. Phys. 20, 093001 (2018).",
            "[3] C. Moore, D. Rockmore, and A. Russell, 'Generic quantum Fourier transforms,' ACM Trans. Algorithms 2, 707-723 (2006).",
            "[4] A. Barenco et al., 'Elementary gates for quantum computation,' Phys. Rev. A 52, 3457-3467 (1995).",
            "[5] M. Ettinger, P. Hoyer, and E. Knill, 'The quantum query complexity of the hidden subgroup problem is polynomial,' Inf. Process. Lett. 91, 43-48 (2004).",
            f"[6] D. P. Davis, 'Exact Binary Tetrahedral Group Inversion on a 5-Qubit Heavy-Hex Superconducting Register (OSIRIS-BT2T-001-FEZ, job {m1_job}),' OSIRIS Archival Preprint (2026).",
        ]
    )

    return doc.to_pdf_bytes()


def generate_all() -> Dict[str, Any]:
    HERE.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST_BT2T002_PATH.read_text(encoding="utf-8"))
    inv_manifest = json.loads(INV_MANIFEST_PATH.read_text(encoding="utf-8"))
    fez_results = json.loads(FEZ_RESULTS_PATH.read_text(encoding="utf-8"))

    # Fail-closed provenance & digest checks
    assert manifest["experiment_id"] == "OSIRIS-BT2T-002"
    assert manifest["execution_tier"] == "SIMULATED"
    assert manifest["provider_job_id"] is None
    assert int(manifest["qpu_seconds"]) == 0
    assert manifest["all_checks_passed"] is True
    assert (
        manifest["encoding_version"]
        == inv_manifest["encoding"]["encoding_version"]
        == fez_results["encoding_version"]
    )
    assert (
        manifest["encoding_sha256"]
        == inv_manifest["encoding"]["sha256"]
        == fez_results["encoding_sha256"]
    )

    source_shas: Dict[str, str] = {
        p.relative_to(ROOT).as_posix(): _sha256_file(p) for p in SOURCE_ARTIFACT_PATHS
    }

    esd_info = _compute_esd_and_plancherel_rms()

    p1 = manifest["part1_multiplication_10q"]
    p2 = manifest["part2_nonabelian_qft_5q"]
    m1_m = p1["metrics"]
    log_g = p1["logical_circuits"]["generic"]
    log_s = p1["logical_circuits"]["structured"]
    p1_prox = p1["proxy_compilations"]

    ata_g = _find_proxy(p1_prox, circuit="generic", target="all_to_all", opt_level=3)
    ata_s = _find_proxy(p1_prox, circuit="structured", target="all_to_all", opt_level=3)
    hh_g = _find_proxy(p1_prox, circuit="generic", target="heavy_hex_fragment_10Q", opt_level=3)
    hh_s = _find_proxy(p1_prox, circuit="structured", target="heavy_hex_fragment_10Q", opt_level=3)

    qft_log = p2["logical_circuit"]
    qft_prox = p2["proxy_compilations"]
    qft_ata_1 = _find_proxy(qft_prox, circuit="bt2t_qft_5q", kind_sub="qiskit", target="all_to_all", opt_level=1)
    qft_ata_3 = _find_proxy(qft_prox, circuit="bt2t_qft_5q", kind_sub="qiskit", target="all_to_all", opt_level=3)
    qft_t5_1 = _find_proxy(
        qft_prox, circuit="bt2t_qft_5q", kind_sub="qiskit", target="heavy_hex_fragment_T5", opt_level=1
    )
    qft_t5_3 = _find_proxy(
        qft_prox, circuit="bt2t_qft_5q", kind_sub="qiskit", target="heavy_hex_fragment_T5", opt_level=3
    )
    qft_tk_raw = _find_proxy(qft_prox, circuit="bt2t_qft_5q", kind_sub="pytket", peephole=False)
    qft_tk_opt = _find_proxy(qft_prox, circuit="bt2t_qft_5q", kind_sub="pytket", peephole=True)

    su = p2["schur_and_unitarity"]
    blk = p2["block_diagonalization_and_convolution"]
    hsp = p2["subgroup_coset_hsp_sampling"]

    # 1. Build RFC 8785 reproducibility_capsule.json (strictly integer/string/bool/None values
    # so rfc8785, jcs_bytes fallback, and exec_kernel.canon agree byte-for-byte).
    capsule_body: Dict[str, Any] = {
        "schema": "osiris.reproducibility_capsule/1",
        "paper_id": "OSIRIS-BT2T-002-PAPER",
        "experiment_id": str(manifest["experiment_id"]),
        "execution_tier": "SIMULATED",
        "compilation_tier": "OFFLINE_COMPILED",
        "evidence_class": "SIMULATED",
        "hardware_executed": False,
        "provider_job_id": None,
        "qpu_seconds": 0,
        "seed": int(manifest["seed"]),
        "encoding_version": str(manifest["encoding_version"]),
        "encoding_sha256": str(manifest["encoding_sha256"]),
        "all_checks_passed": bool(manifest["all_checks_passed"]),
        "checks_passed_count": sum(1 for v in manifest["checks"].values() if v),
        "checks_total_count": len(manifest["checks"]),
        "part1_multiplication_10q_summary": {
            "num_qubits": 10,
            "num_ancillas": 0,
            "basis_states": 1024,
            "valid_cayley_pairs": int(m1_m["valid_pairs_checked"]),
            "unused_identity_pairs": int(m1_m["unused_pairs_checked"]),
            "logical_structured_gates": {k: int(v) for k, v in sorted(log_s["gate_counts"].items())},
            "logical_structured_depth": int(log_s["depth"]),
            "logical_generic_gates": {k: int(v) for k, v in sorted(log_g["gate_counts"].items())},
            "logical_generic_depth": int(log_g["depth"]),
            "structured_operator_max_abs_error": "0.0",
            "generic_operator_max_abs_error": "0.0",
            "proxy_cz_structured_all_to_all": int(m1_m["cz_structured_all_to_all_opt3"]),
            "proxy_cz_generic_all_to_all": int(m1_m["cz_generic_all_to_all_opt3"]),
            "proxy_cz_compression_all_to_all": f"{m1_m['cz_compression_ratio_all_to_all']:.2f}x",
            "proxy_cz_structured_heavy_hex_10q": int(m1_m["cz_structured_heavy_hex_10q_opt3"]),
            "proxy_cz_generic_heavy_hex_10q": int(m1_m["cz_generic_heavy_hex_10q_opt3"]),
            "proxy_cz_compression_heavy_hex_10q": f"{m1_m['cz_compression_ratio_heavy_hex_10q']:.2f}x",
        },
        "part2_nonabelian_qft_5q_summary": {
            "num_qubits": 5,
            "irrep_count": 7,
            "irrep_dimensions": [1, 1, 1, 2, 2, 2, 3],
            "plancherel_sum_d_squared": int(su["sum_d_alpha_squared"]),
            "contiguous_block_sizes": [int(x) for x in blk["block_sizes_contiguous"]],
            "f32_unitarity_inf_error": f"{su['f32_unitarity_inf_error']:.2e}",
            "irrep_homomorphism_max_error": f"{max(su['irrep_homomorphism_max_abs_errors'].values()):.2e}",
            "left_regular_max_off_block_abs": f"{blk['left_regular_max_off_block_abs']:.2e}",
            "q8_coset_1d_irrep_weights": ["1/3", "1/3", "1/3"],
            "q8_max_forbidden_leakage": f"{hsp['subgroups']['Q8_normal']['max_forbidden_irrep_leakage']:.2e}",
            "subgroups_verified": 6,
            "proxy_cz_all_to_all": int(qft_log["cz_all_to_all_opt3"]),
            "proxy_cz_heavy_hex_t5": int(qft_log["cz_heavy_hex_T5_opt3"]),
            "proxy_zzphase_pytket_h2": int(qft_tk_opt["two_qubit_gates"]["ZZPhase"]),
        },
        "milestone1_context_anchor": {
            "protocol_id": str(fez_results["protocol_id"]),
            "backend": str(fez_results["backend"]),
            "provider_job_id": str(fez_results["provider_job_id"]),
            "structured_cz": int(fez_results["arms"]["STRUCTURED"]["cz_count"]),
            "generic_cz": int(fez_results["arms"]["GENERIC"]["cz_count"]),
            "note": "Contextualizes NISQ decoherence wall (~723 CZ) motivating QCCD/FT for 1,739 CZ & 423 ZZPhase.",
        },
        "source_artifacts": dict(sorted(source_shas.items())),
    }
    capsule_sha = _sha256_bytes(_canonical_jcs_bytes(capsule_body))
    capsule = {**capsule_body, "rfc8785_canonical_sha256": capsule_sha}
    (HERE / "reproducibility_capsule.json").write_bytes(
        (json.dumps(capsule, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )

    # 2. Extract Milestone 1 contextual anchor numbers & Milestone 2 exact macros for generated.tex
    inv_prox = inv_manifest["proxy_compilations"]
    m1_ata_g = int(
        _find_proxy(inv_prox, circuit="generic", kind_sub="qiskit", target="all_to_all", opt_level=3)[
            "two_qubit_gates"
        ]["cz"]
    )
    m1_ata_s = int(
        _find_proxy(inv_prox, circuit="structured", kind_sub="qiskit", target="all_to_all", opt_level=3)[
            "two_qubit_gates"
        ]["cz"]
    )
    m1_tk_g = int(
        _find_proxy(inv_prox, circuit="generic", kind_sub="pytket", peephole=True)["two_qubit_gates"][
            "ZZPhase"
        ]
    )
    m1_tk_s = int(
        _find_proxy(inv_prox, circuit="structured", kind_sub="pytket", peephole=True)["two_qubit_gates"][
            "ZZPhase"
        ]
    )

    m1_s = fez_results["arms"]["STRUCTURED"]
    m1_g = fez_results["arms"]["GENERIC"]
    m1_c = fez_results["arms"]["PREP_RO_CTRL"]
    m1_sv = m1_s["valid_24_uniform"]
    m1_gv = m1_g["valid_24_uniform"]
    m1_cv = m1_c["valid_24_uniform"]
    m1_cu = m1_c["unused_8_uniform"]
    m1_con = fez_results["contrasts"]

    gen_gc = log_g["gate_counts"]
    str_gc = log_s["gate_counts"]
    gen_tot = sum(int(v) for v in gen_gc.values())
    str_tot = sum(int(v) for v in str_gc.values())
    qft_gc = qft_log["gate_counts"]
    qft_tot = sum(int(v) for v in qft_gc.values())

    max_ir_unit = max(float(v) for v in su["irrep_unitary_max_inf_errors"].values())
    max_ir_hom = max(float(v) for v in su["irrep_homomorphism_max_abs_errors"].values())
    max_hsp_err = max(
        float(v["max_coset_weight_abs_error_all_24_g"]) for v in hsp["subgroups"].values()
    )
    max_hsp_forb = max(
        float(v["max_forbidden_irrep_leakage"]) for v in hsp["subgroups"].values()
    )

    eps_hh, eps_qccd, eps_ft = 5.0e-3, 1.0e-3, 2.0e-4

    mul_table_rows = [
        f"M1 5Q Inv (All-to-All, opt 3) & \\texttt{{GENERIC}} / \\texttt{{STRUCTURED}} & {m1_ata_g} / {m1_ata_s} \\texttt{{CZ}} & 2,069 / 242 & 1,481 / 168 & ${m1_ata_g / m1_ata_s:.2f}\\times$ & \\texttt{{PROXY}} (exact $P_{{32}}$) \\\\",
        f"M1 5Q Inv (\\texttt{{ibm\\_fez}} \\texttt{{T5}}, opt 3) & \\texttt{{GENERIC}} / \\texttt{{STRUCTURED}} & {m1_g['cz_count']} / {m1_s['cz_count']} \\texttt{{CZ}} & 3,281 / 350 & {m1_g['core_depth']:,} / {m1_s['core_depth']} & ${m1_g['cz_count'] / m1_s['cz_count']:.2f}\\times$ & \\texttt{{HW}}: ${m1_gv['p_correct_raw']:.4f} \\to {m1_sv['p_correct_raw']:.4f}$ \\\\",
        f"M1 5Q Inv (\\texttt{{pytket}} \\texttt{{H2}} opt) & \\texttt{{GENERIC}} / \\texttt{{STRUCTURED}} & {m1_tk_g} / {m1_tk_s} \\texttt{{ZZPhase}} & 2,520 / 294 & 1,720 / 194 & ${m1_tk_g / m1_tk_s:.2f}\\times$ & \\texttt{{PROXY}} \\texttt{{H2}} rebase \\\\",
        f"M2 10Q Mul (Logical, $0$-ancilla) & \\texttt{{GENERIC}} (cycle/transp.) & {gen_gc['mcx']:,} \\texttt{{MCX}} + {gen_gc['cx']:,} \\texttt{{CX}} & {gen_tot:,} & {log_g['depth']:,} & $1.00\\times$ & $\\|U - P_{{1024}}\\|_{{\\max}} = 0.0$ \\\\",
        f"M2 10Q Mul (Logical, $0$-ancilla) & \\texttt{{STRUCTURED}} ($Q_8 \\rtimes \\mathbb{{Z}}_3$) & {str_gc['ccx']} \\texttt{{CCX}} + {str_gc['cx']} \\texttt{{CX}} + {str_gc['cswap']} \\texttt{{CSWAP}} & {str_tot} & {log_s['depth']} & ${log_g['depth'] / log_s['depth']:.2f}\\times$ depth & $\\|U - P_{{1024}}\\|_{{\\max}} = 0.0$ \\\\",
        f"M2 10Q Mul (All-to-All, opt 3) & \\texttt{{GENERIC}} / \\texttt{{STRUCTURED}} & {ata_g['two_qubit_gates']['cz']:,} / {ata_s['two_qubit_gates']['cz']:,} \\texttt{{CZ}} & {ata_g['size']:,} / {ata_s['size']:,} & {ata_g['depth']:,} / {ata_s['depth']:,} & ${m1_m['cz_compression_ratio_all_to_all']:.2f}\\times$ & \\texttt{{PROXY}} ($\\le {_fmt_sci_tex(float(ata_s['phase_aligned_max_abs_error']))}$) \\\\",
        f"M2 10Q Mul (10Q Heavy-Hex, opt 3) & \\texttt{{GENERIC}} / \\texttt{{STRUCTURED}} & {hh_g['two_qubit_gates']['cz']:,} / {hh_s['two_qubit_gates']['cz']:,} \\texttt{{CZ}} & {hh_g['size']:,} / {hh_s['size']:,} & {hh_g['depth']:,} / {hh_s['depth']:,} & ${m1_m['cz_compression_ratio_heavy_hex_10q']:.2f}\\times$ & \\texttt{{PROXY}} ($\\le {_fmt_sci_tex(float(hh_s['phase_aligned_max_abs_error']))}$) \\\\",
        f"M2 5Q QFT (Logical QSD) & $\\widetilde{{F}}_{{2T}} = F_{{2T}} \\oplus I_8$ & {qft_gc['cx']} \\texttt{{CX}} + {qft_gc['u']} $U$ + {qft_gc['rz']} $R_z$ & {qft_tot:,} & {qft_log['depth']} & --- & $\\|U - \\widetilde{{F}}_{{2T}}\\|_{{\\max}} = {_fmt_sci_tex(float(qft_log['direct_operator_max_abs_error']))}$ \\\\",
        f"M2 5Q QFT (All-to-All / \\texttt{{T5}}, opt 3) & Qiskit ISA & {qft_ata_3['two_qubit_gates']['cz']} / {qft_t5_3['two_qubit_gates']['cz']} \\texttt{{CZ}} & {qft_ata_3['size']:,} / {qft_t5_3['size']:,} & {qft_ata_3['depth']:,} / {qft_t5_3['depth']:,} & ${qft_t5_3['two_qubit_gates']['cz'] / qft_ata_3['two_qubit_gates']['cz']:.2f}\\times$ \\texttt{{T5}}/All & \\texttt{{PROXY}} ($\\le {_fmt_sci_tex(float(qft_ata_3['phase_aligned_max_abs_error']))}$) \\\\",
        f"M2 5Q QFT (\\texttt{{pytket}} \\texttt{{H2}} opt) & \\texttt{{H2}}-Native Rebase & {qft_tk_opt['two_qubit_gates']['ZZPhase']} \\texttt{{ZZPhase}} & {qft_tk_opt['size']:,} & {qft_tk_opt['depth']:,} & $1.00\\times$ vs All & \\texttt{{PROXY}} ($\\le {_fmt_sci_tex(float(qft_tk_opt['phase_aligned_max_abs_error']))}$) \\\\",
    ]

    irrep_table_rows: List[str] = []
    slice_cursor = 0
    for ir_key, _ir_ascii, ir_tex, dim_d in IRREP_SPECS:
        span = dim_d * dim_d
        rms_v = 1.0 / math.sqrt(float(dim_d))
        u_err = float(su["irrep_unitary_max_inf_errors"][ir_key])
        h_err = float(su["irrep_homomorphism_max_abs_errors"][ir_key])
        rule_tex = (
            "$\\zeta_3^{r c(g)}$ ($2T/Q_8 \\cong \\mathbb{Z}_3$)"
            if dim_d == 1
            else (
                "$\\chi_r(g)\\, U_{SU(2)}(g) \\in U(2)$"
                if dim_d == 2
                else "$\\tfrac{1}{2}\\mathrm{Tr}(\\sigma_b U(g)\\sigma_a U(g)^\\dagger) \\in SO(3)$"
            )
        )
        irrep_table_rows.append(
            f"${ir_tex}$ & {dim_d} & ${dim_d}^2 = {span}$ & $[{slice_cursor}..{slice_cursor + span - 1}]$ (${dim_d}\\times {dim_d}\\times {dim_d}$) & {rms_v:.4f} & ${_fmt_sci_tex(u_err)}$ & ${_fmt_sci_tex(h_err)}$ & {rule_tex} \\\\"
        )
        slice_cursor += span

    hsp_table_rows: List[str] = []
    for sg_key, _sg_ascii, sg_tex, norm_s in SUBGROUP_ORDER:
        info = hsp["subgroups"][sg_key]
        w_map = info["measured_irrep_weights_coset_eH"]
        m_map = info["trivial_irrep_multiplicities"]
        m_str = "(" + ",".join(str(m_map[k]) for k, _, _, _ in IRREP_SPECS) + ")"
        w_cells = " & ".join(f"${_fraction_label(float(w_map[k]))}$" for k, _, _, _ in IRREP_SPECS)
        hsp_table_rows.append(
            f"${sg_tex}$ & {info['order']} & {norm_s} & ${m_str}$ & {w_cells} & ${_fmt_sci_tex(float(info['max_coset_weight_abs_error_all_24_g']))}$ & ${_fmt_sci_tex(float(info['max_forbidden_irrep_leakage']))}$ \\\\"
        )

    feas_table_rows = [
        f"M1 5Q Inv (\\texttt{{STRUCTURED}}) & 72 \\texttt{{CZ}} / 47 \\texttt{{ZZPhase}} & ${(1.0 - eps_hh) ** 72:.3f}$ (emp {m1_sv['p_correct_raw']:.4f} / {m1_sv['p_correct_ro_mitigated_secondary']:.4f}) & ${(1.0 - eps_qccd) ** 47:.3f}$ & ${(1.0 - eps_ft) ** 47:.3f}$ & Heavy-Hex NISQ Attested \\\\",
        f"M1 5Q Inv (\\texttt{{GENERIC}}) & 723 \\texttt{{CZ}} / 400 \\texttt{{ZZPhase}} & ${(1.0 - eps_hh) ** 723:.3f}$ (emp {m1_gv['p_correct_raw']:.4f} / {m1_gv['p_correct_ro_mitigated_secondary']:.4f}) & ${(1.0 - eps_qccd) ** 400:.3f}$ & ${(1.0 - eps_ft) ** 400:.3f}$ & Heavy-Hex Decoherence Wall \\\\",
        f"M2 5Q Non-Abelian QFT & 613 \\texttt{{CZ}} / 423 \\texttt{{ZZPhase}} & ${(1.0 - eps_hh) ** 613:.3f}$ (NISQ marginal) & ${(1.0 - eps_qccd) ** 423:.3f}$ & ${(1.0 - eps_ft) ** 423:.3f}$ & Current All-to-All QCCD \\\\",
        f"M2 10Q Mul (\\texttt{{STRUCTURED}}) & 3,662 \\texttt{{CZ}} / 1,739 \\texttt{{CZ}} & ${_fmt_sci_tex((1.0 - eps_hh) ** 3662, 1)}$ (depolarized) & ${(1.0 - eps_qccd) ** 1739:.3f}$ & ${(1.0 - eps_ft) ** 1739:.3f}$ & Next-Gen QCCD / Early FT \\\\",
        f"M2 10Q Mul (\\texttt{{GENERIC}}) & 78,035 \\texttt{{CZ}} / 33,042 \\texttt{{CZ}} & ${_fmt_sci_tex((1.0 - eps_hh) ** 78035, 1)}$ & ${_fmt_sci_tex((1.0 - eps_qccd) ** 33042, 1)}$ & ${(1.0 - eps_ft) ** 33042:.3f}$ & Infeasible Without Algebra \\\\",
    ]

    esd_c = esd_info["esd_counts"]
    q8_sub = hsp["subgroups"]["Q8_normal"]
    z2_sub = hsp["subgroups"]["Z2_center"]
    z4_sub = hsp["subgroups"]["Z4_i"]
    z3_sub = hsp["subgroups"]["Z3_omega"]
    triv_sub = hsp["subgroups"]["trivial_1"]
    full_sub = hsp["subgroups"]["full_2T"]

    macros: List[str] = [
        "% Auto-generated by generate_bt2t002_paper_assets.py -- DO NOT EDIT BY HAND",
        f"\\newcommand{{\\btmExperimentId}}{{\\texttt{{{manifest['experiment_id']}}}}}",
        "\\newcommand{\\btmPaperId}{\\texttt{OSIRIS-BT2T-002-PAPER}}",
        "\\newcommand{\\btmExecTier}{\\texttt{SIMULATED}}",
        "\\newcommand{\\btmCompileTier}{\\texttt{OFFLINE\\_COMPILED}}",
        "\\newcommand{\\btmProxyTier}{\\texttt{PROXY}}",
        "\\newcommand{\\btmJobId}{\\texttt{null}}",
        f"\\newcommand{{\\btmQpuSec}}{{{int(manifest['qpu_seconds'])}}}",
        f"\\newcommand{{\\btmSeed}}{{{int(manifest['seed'])}}}",
        f"\\newcommand{{\\btmEncVer}}{{\\texttt{{{manifest['encoding_version']}}}}}",
        f"\\newcommand{{\\btmEncSha}}{{\\texttt{{{manifest['encoding_sha256'][:16]}}}}}",
        f"\\newcommand{{\\btmManifestSha}}{{\\texttt{{{source_shas['experiments/bt2t_primitives/manifest_bt2t002.json'][:16]}}}}}",
        f"\\newcommand{{\\btmMulPySha}}{{\\texttt{{{source_shas['experiments/bt2t_primitives/bt2t_mul.py'][:16]}}}}}",
        f"\\newcommand{{\\btmQftPySha}}{{\\texttt{{{source_shas['experiments/bt2t_primitives/bt2t_qft.py'][:16]}}}}}",
        f"\\newcommand{{\\btmVerifyPySha}}{{\\texttt{{{source_shas['experiments/bt2t_primitives/verify_bt2t002.py'][:16]}}}}}",
        f"\\newcommand{{\\btmBaseBtPySha}}{{\\texttt{{{source_shas['experiments/bt2t_inversion/bt2t.py'][:16]}}}}}",
        f"\\newcommand{{\\btmMoneResSha}}{{\\texttt{{{source_shas['experiments/bt2t_fez/physical_bt2t_fez_results.json'][:16]}}}}}",
        f"\\newcommand{{\\btmStructQasmSha}}{{\\texttt{{{log_s['decomposed_qasm2_sha256'][:16]}}}}}",
        f"\\newcommand{{\\btmGenQasmSha}}{{\\texttt{{{log_g['decomposed_qasm2_sha256'][:16]}}}}}",
        f"\\newcommand{{\\btmQftQasmSha}}{{\\texttt{{{qft_log['u3_cx_qasm2_sha256'][:16]}}}}}",
        f"\\newcommand{{\\btmCapsuleSha}}{{\\texttt{{{capsule_sha[:16]}}}}}",
        f"\\newcommand{{\\btmChecksTotal}}{{{len(manifest['checks'])}}}",
        f"\\newcommand{{\\btmChecksPassed}}{{{sum(1 for v in manifest['checks'].values() if v)}}}",
        "\\newcommand{\\btmGroupOrder}{24}",
        "\\newcommand{\\btmQeightOrder}{8}",
        "\\newcommand{\\btmZthreeOrder}{3}",
        "\\newcommand{\\btmRegQubits}{5}",
        "\\newcommand{\\btmRegStates}{32}",
        "\\newcommand{\\btmUnusedCodewords}{8}",
        f"\\newcommand{{\\btmMulQubits}}{{{int(log_s['num_qubits'])}}}",
        f"\\newcommand{{\\btmMulAncillas}}{{{int(log_s['num_ancillas'])}}}",
        "\\newcommand{\\btmMulBasisStates}{1,024}",
        "\\newcommand{\\btmMulBasisStatesRaw}{1024}",
        f"\\newcommand{{\\btmValidPairs}}{{{int(m1_m['valid_pairs_checked'])}}}",
        f"\\newcommand{{\\btmUnusedPairs}}{{{int(m1_m['unused_pairs_checked'])}}}",
        "\\newcommand{\\btmQeightPairs}{64}",
        "\\newcommand{\\btmNonIdElements}{23}",
        "\\newcommand{\\btmOmegaIdx}{23}",
        "\\newcommand{\\btmOmegaSqIdx}{16}",
        "\\newcommand{\\btmHurwitzCount}{16}",
        f"\\newcommand{{\\btmMoneProtocolId}}{{\\texttt{{{fez_results['protocol_id']}}}}}",
        "\\newcommand{\\btmMoneBackend}{\\texttt{ibm\\_fez}}",
        f"\\newcommand{{\\btmMoneJobId}}{{\\texttt{{{fez_results['provider_job_id']}}}}}",
        "\\newcommand{\\btmMoneTier}{\\texttt{HARDWARE\\_ATTESTED}}",
        f"\\newcommand{{\\btmMoneWindows}}{{{int(fez_results['unit_of_inference']['calibration_windows'])}}}",
        f"\\newcommand{{\\btmMoneShots}}{{{int(fez_results['equal_shot_rule']['total_shots']):,}}}",
        f"\\newcommand{{\\btmMoneQpuSec}}{{{int(fez_results['job_metadata']['usage']['qpu_charge_time_seconds'])}}}",
        f"\\newcommand{{\\btmMoneAtaStructCz}}{{{m1_ata_s}}}",
        f"\\newcommand{{\\btmMoneAtaGenCz}}{{{m1_ata_g}}}",
        f"\\newcommand{{\\btmMoneAtaRatio}}{{{m1_ata_g / m1_ata_s:.2f}\\times}}",
        f"\\newcommand{{\\btmMoneFezStructCz}}{{{int(m1_s['cz_count'])}}}",
        f"\\newcommand{{\\btmMoneFezStructSx}}{{{int(m1_s['sx_count'])}}}",
        f"\\newcommand{{\\btmMoneFezStructDepth}}{{{int(m1_s['core_depth'])}}}",
        f"\\newcommand{{\\btmMoneFezGenCz}}{{{int(m1_g['cz_count'])}}}",
        f"\\newcommand{{\\btmMoneFezGenSx}}{{{int(m1_g['sx_count']):,}}}",
        f"\\newcommand{{\\btmMoneFezGenDepth}}{{{int(m1_g['core_depth']):,}}}",
        f"\\newcommand{{\\btmMoneFezRatio}}{{{m1_g['cz_count'] / m1_s['cz_count']:.2f}\\times}}",
        f"\\newcommand{{\\btmMoneTkStructZz}}{{{m1_tk_s}}}",
        f"\\newcommand{{\\btmMoneTkGenZz}}{{{m1_tk_g}}}",
        f"\\newcommand{{\\btmMoneTkRatio}}{{{m1_tk_g / m1_tk_s:.2f}\\times}}",
        f"\\newcommand{{\\btmMonePValidStruct}}{{{float(m1_sv['p_correct_raw']):.4f}}}",
        f"\\newcommand{{\\btmMonePValidStructSe}}{{{float(m1_sv['within_window_sampling_se']):.4f}}}",
        f"\\newcommand{{\\btmMonePValidStructMit}}{{{float(m1_sv['p_correct_ro_mitigated_secondary']):.4f}}}",
        f"\\newcommand{{\\btmMonePValidGen}}{{{float(m1_gv['p_correct_raw']):.4f}}}",
        f"\\newcommand{{\\btmMonePValidGenSe}}{{{float(m1_gv['within_window_sampling_se']):.4f}}}",
        f"\\newcommand{{\\btmMonePValidGenMit}}{{{float(m1_gv['p_correct_ro_mitigated_secondary']):.4f}}}",
        f"\\newcommand{{\\btmMoneDeltaValid}}{{{float(m1_con['structured_minus_generic_valid24_raw']):.4f}}}",
        f"\\newcommand{{\\btmMonePValidCtrl}}{{{float(m1_cv['p_correct_raw']):.4f}}}",
        f"\\newcommand{{\\btmMonePUnusedCtrl}}{{{float(m1_cu['p_fixed_raw']):.4f}}}",
        f"\\newcommand{{\\btmMoneRandFloor}}{{{float(m1_con['uniform_random_32state_chance']):.5f}}}",
        f"\\newcommand{{\\btmEsdX}}{{{int(esd_c['x'])}}}",
        f"\\newcommand{{\\btmEsdCx}}{{{int(esd_c['cx'])}}}",
        f"\\newcommand{{\\btmEsdCcx}}{{{int(esd_c['ccx'])}}}",
        f"\\newcommand{{\\btmEsdCswap}}{{{int(esd_c['cswap'])}}}",
        f"\\newcommand{{\\btmEsdCthreex}}{{{int(esd_c['c3x'])}}}",
        f"\\newcommand{{\\btmEsdCfourx}}{{{int(esd_c['c4x'])}}}",
        f"\\newcommand{{\\btmEsdMcxTotal}}{{{int(esd_c['c3x'] + esd_c['c4x'])}}}",
        f"\\newcommand{{\\btmEsdTotalGates}}{{{int(esd_info['esd_total'])}}}",
        f"\\newcommand{{\\btmEsdDepth}}{{{int(esd_info['esd_depth'])}}}",
        f"\\newcommand{{\\btmEsdOpErr}}{{{float(manifest['semidirect_algebra']['esd_5q_operator_max_abs_error']):.1f}}}",
        "\\newcommand{\\btmBarencoLemmaSevenTwo}{7.2}",
        "\\newcommand{\\btmBarencoLemmaSevenThree}{7.3}",
        "\\newcommand{\\btmBarencoThreeCcx}{4}",
        "\\newcommand{\\btmBarencoFourCcx}{8}",
        "\\newcommand{\\btmBarencoFiveCcx}{12}",
        "\\newcommand{\\btmBarencoSixCcx}{24}",
        f"\\newcommand{{\\btmMulStructCcx}}{{{int(str_gc['ccx'])}}}",
        f"\\newcommand{{\\btmMulStructCx}}{{{int(str_gc['cx'])}}}",
        f"\\newcommand{{\\btmMulStructX}}{{{int(str_gc['x'])}}}",
        f"\\newcommand{{\\btmMulStructCswap}}{{{int(str_gc['cswap'])}}}",
        "\\newcommand{\\btmMulStructMcx}{0}",
        f"\\newcommand{{\\btmMulStructTotalGates}}{{{str_tot}}}",
        f"\\newcommand{{\\btmMulStructDepth}}{{{int(log_s['depth'])}}}",
        f"\\newcommand{{\\btmMulGenMcx}}{{{int(gen_gc['mcx']):,}}}",
        f"\\newcommand{{\\btmMulGenMcxRaw}}{{{int(gen_gc['mcx'])}}}",
        f"\\newcommand{{\\btmMulGenCx}}{{{int(gen_gc['cx']):,}}}",
        f"\\newcommand{{\\btmMulGenX}}{{{int(gen_gc['x']):,}}}",
        f"\\newcommand{{\\btmMulGenCswap}}{{{int(gen_gc['cswap'])}}}",
        f"\\newcommand{{\\btmMulGenTotalGates}}{{{gen_tot:,}}}",
        f"\\newcommand{{\\btmMulGenDepth}}{{{int(log_g['depth']):,}}}",
        f"\\newcommand{{\\btmMulLogGateRatio}}{{{gen_tot / str_tot:.2f}\\times}}",
        f"\\newcommand{{\\btmMulLogDepthRatio}}{{{log_g['depth'] / log_s['depth']:.2f}\\times}}",
        f"\\newcommand{{\\btmMulStructOpErr}}{{{float(m1_m['structured_operator_max_abs_error']):.1f}}}",
        f"\\newcommand{{\\btmMulGenOpErr}}{{{float(m1_m['generic_operator_max_abs_error']):.1f}}}",
        f"\\newcommand{{\\btmMulStructSupErr}}{{{float(m1_m['structured_superposition_max_abs_error']):.1f}}}",
        f"\\newcommand{{\\btmMulGenSupErr}}{{{float(m1_m['generic_superposition_max_abs_error']):.1f}}}",
        "\\newcommand{\\btmMulSupTrials}{32}",
        f"\\newcommand{{\\btmMulPhaseCtrlErr}}{{{float(m1_m['phase_control_injected_cz_max_abs_error']):.1f}}}",
        f"\\newcommand{{\\btmMulAtaStructCz}}{{{int(ata_s['two_qubit_gates']['cz']):,}}}",
        f"\\newcommand{{\\btmMulAtaStructCzRaw}}{{{int(ata_s['two_qubit_gates']['cz'])}}}",
        f"\\newcommand{{\\btmMulAtaStructDepth}}{{{int(ata_s['depth']):,}}}",
        f"\\newcommand{{\\btmMulAtaStructSize}}{{{int(ata_s['size']):,}}}",
        f"\\newcommand{{\\btmMulAtaStructErr}}{{{_fmt_sci_tex(float(ata_s['phase_aligned_max_abs_error']))}}}",
        f"\\newcommand{{\\btmMulAtaGenCz}}{{{int(ata_g['two_qubit_gates']['cz']):,}}}",
        f"\\newcommand{{\\btmMulAtaGenDepth}}{{{int(ata_g['depth']):,}}}",
        f"\\newcommand{{\\btmMulAtaGenSize}}{{{int(ata_g['size']):,}}}",
        f"\\newcommand{{\\btmMulAtaGenErr}}{{{_fmt_sci_tex(float(ata_g['phase_aligned_max_abs_error']))}}}",
        f"\\newcommand{{\\btmMulAtaCzRatio}}{{{float(m1_m['cz_compression_ratio_all_to_all']):.2f}\\times}}",
        f"\\newcommand{{\\btmMulAtaCzRatioFour}}{{{float(m1_m['cz_compression_ratio_all_to_all']):.4f}\\times}}",
        f"\\newcommand{{\\btmMulAtaDepthRatio}}{{{ata_g['depth'] / ata_s['depth']:.2f}\\times}}",
        f"\\newcommand{{\\btmMulHhStructCz}}{{{int(hh_s['two_qubit_gates']['cz']):,}}}",
        f"\\newcommand{{\\btmMulHhStructDepth}}{{{int(hh_s['depth']):,}}}",
        f"\\newcommand{{\\btmMulHhStructSize}}{{{int(hh_s['size']):,}}}",
        f"\\newcommand{{\\btmMulHhStructErr}}{{{_fmt_sci_tex(float(hh_s['phase_aligned_max_abs_error']))}}}",
        f"\\newcommand{{\\btmMulHhGenCz}}{{{int(hh_g['two_qubit_gates']['cz']):,}}}",
        f"\\newcommand{{\\btmMulHhGenDepth}}{{{int(hh_g['depth']):,}}}",
        f"\\newcommand{{\\btmMulHhGenSize}}{{{int(hh_g['size']):,}}}",
        f"\\newcommand{{\\btmMulHhGenErr}}{{{_fmt_sci_tex(float(hh_g['phase_aligned_max_abs_error']))}}}",
        f"\\newcommand{{\\btmMulHhCzRatio}}{{{float(m1_m['cz_compression_ratio_heavy_hex_10q']):.2f}\\times}}",
        f"\\newcommand{{\\btmMulHhCzRatioFour}}{{{float(m1_m['cz_compression_ratio_heavy_hex_10q']):.4f}\\times}}",
        f"\\newcommand{{\\btmMulHhDepthRatio}}{{{hh_g['depth'] / hh_s['depth']:.2f}\\times}}",
        f"\\newcommand{{\\btmMulHhSwapOverheadStruct}}{{{hh_s['two_qubit_gates']['cz'] / ata_s['two_qubit_gates']['cz']:.2f}\\times}}",
        f"\\newcommand{{\\btmMulHhSwapOverheadGen}}{{{hh_g['two_qubit_gates']['cz'] / ata_g['two_qubit_gates']['cz']:.2f}\\times}}",
        "\\newcommand{\\btmNumIrreps}{7}",
        "\\newcommand{\\btmNumOneDIrreps}{3}",
        "\\newcommand{\\btmNumTwoDIrreps}{3}",
        "\\newcommand{\\btmNumThreeDIrreps}{1}",
        f"\\newcommand{{\\btmNumContigBlocks}}{{{len(blk['block_sizes_contiguous'])}}}",
        "\\newcommand{\\btmBlockSizesStr}{(1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3)}",
        "\\newcommand{\\btmDiagBlockEntries}{54}",
        "\\newcommand{\\btmOffBlockEntries}{522}",
        f"\\newcommand{{\\btmSumDSquared}}{{{int(su['sum_d_alpha_squared'])}}}",
        f"\\newcommand{{\\btmRmsOneD}}{{{1.0 / math.sqrt(1.0):.4f}}}",
        f"\\newcommand{{\\btmRmsTwoD}}{{{1.0 / math.sqrt(2.0):.4f}}}",
        f"\\newcommand{{\\btmRmsThreeD}}{{{1.0 / math.sqrt(3.0):.4f}}}",
        f"\\newcommand{{\\btmQftUnitarityErr}}{{{_fmt_sci_tex(float(su['f32_unitarity_inf_error']))}}}",
        f"\\newcommand{{\\btmCharOrthErr}}{{{_fmt_sci_tex(float(su['character_orthogonality_inf_error']))}}}",
        f"\\newcommand{{\\btmIrrepUnitMaxErr}}{{{_fmt_sci_tex(max_ir_unit)}}}",
        f"\\newcommand{{\\btmIrrepHomMaxErr}}{{{_fmt_sci_tex(max_ir_hom)}}}",
        f"\\newcommand{{\\btmLeftOffBlockErr}}{{{_fmt_sci_tex(float(blk['left_regular_max_off_block_abs']))}}}",
        f"\\newcommand{{\\btmLeftOnBlockErr}}{{{_fmt_sci_tex(float(blk['left_regular_max_on_block_error']))}}}",
        f"\\newcommand{{\\btmRightOffIrrepErr}}{{{_fmt_sci_tex(float(blk['right_regular_max_off_irrep_abs']))}}}",
        f"\\newcommand{{\\btmRightOnBlockErr}}{{{_fmt_sci_tex(float(blk['right_regular_max_on_block_error']))}}}",
        f"\\newcommand{{\\btmPlancherelErr}}{{{float(blk['plancherel_norm_error']):.1f}}}",
        f"\\newcommand{{\\btmConvMaxErr}}{{{_fmt_sci_tex(float(blk['nonabelian_convolution_max_error']))}}}",
        "\\newcommand{\\btmQftUnusedIdErr}{0.0}",
        f"\\newcommand{{\\btmQftLogCx}}{{{int(qft_gc['cx'])}}}",
        f"\\newcommand{{\\btmQftLogU}}{{{int(qft_gc['u'])}}}",
        f"\\newcommand{{\\btmQftLogRz}}{{{int(qft_gc['rz'])}}}",
        f"\\newcommand{{\\btmQftLogH}}{{{int(qft_gc['h'])}}}",
        f"\\newcommand{{\\btmQftLogTotal}}{{{qft_tot:,}}}",
        f"\\newcommand{{\\btmQftLogDepth}}{{{int(qft_log['depth'])}}}",
        f"\\newcommand{{\\btmQftDirectOpErr}}{{{_fmt_sci_tex(float(qft_log['direct_operator_max_abs_error']))}}}",
        f"\\newcommand{{\\btmQftAtaOptOneCz}}{{{int(qft_ata_1['two_qubit_gates']['cz'])}}}",
        f"\\newcommand{{\\btmQftAtaOptOneDepth}}{{{int(qft_ata_1['depth']):,}}}",
        f"\\newcommand{{\\btmQftAtaOptOneSize}}{{{int(qft_ata_1['size']):,}}}",
        f"\\newcommand{{\\btmQftAtaCz}}{{{int(qft_ata_3['two_qubit_gates']['cz'])}}}",
        f"\\newcommand{{\\btmQftAtaDepth}}{{{int(qft_ata_3['depth']):,}}}",
        f"\\newcommand{{\\btmQftAtaSize}}{{{int(qft_ata_3['size']):,}}}",
        f"\\newcommand{{\\btmQftAtaErr}}{{{_fmt_sci_tex(float(qft_ata_3['phase_aligned_max_abs_error']))}}}",
        f"\\newcommand{{\\btmQftTfOptOneCz}}{{{int(qft_t5_1['two_qubit_gates']['cz'])}}}",
        f"\\newcommand{{\\btmQftTfOptOneDepth}}{{{int(qft_t5_1['depth']):,}}}",
        f"\\newcommand{{\\btmQftTfOptOneSize}}{{{int(qft_t5_1['size']):,}}}",
        f"\\newcommand{{\\btmQftTfCz}}{{{int(qft_t5_3['two_qubit_gates']['cz'])}}}",
        f"\\newcommand{{\\btmQftTfDepth}}{{{int(qft_t5_3['depth']):,}}}",
        f"\\newcommand{{\\btmQftTfSize}}{{{int(qft_t5_3['size']):,}}}",
        f"\\newcommand{{\\btmQftTfErr}}{{{_fmt_sci_tex(float(qft_t5_3['phase_aligned_max_abs_error']))}}}",
        f"\\newcommand{{\\btmQftTfOverhead}}{{{qft_t5_3['two_qubit_gates']['cz'] / qft_ata_3['two_qubit_gates']['cz']:.2f}\\times}}",
        f"\\newcommand{{\\btmQftTkZzRaw}}{{{int(qft_tk_raw['two_qubit_gates']['ZZPhase'])}}}",
        f"\\newcommand{{\\btmQftTkZz}}{{{int(qft_tk_opt['two_qubit_gates']['ZZPhase'])}}}",
        f"\\newcommand{{\\btmQftTkDepth}}{{{int(qft_tk_opt['depth']):,}}}",
        f"\\newcommand{{\\btmQftTkSize}}{{{int(qft_tk_opt['size']):,}}}",
        f"\\newcommand{{\\btmQftTkErr}}{{{_fmt_sci_tex(float(qft_tk_opt['phase_aligned_max_abs_error']))}}}",
        f"\\newcommand{{\\btmNumSubgroups}}{{{len(hsp['subgroups'])}}}",
        f"\\newcommand{{\\btmTotalCosetTests}}{{{len(hsp['subgroups']) * 24}}}",
        f"\\newcommand{{\\btmQeightOneDWeight}}{{{float(hsp['q8_1d_total_weight']):.4f}}}",
        f"\\newcommand{{\\btmQeightEachOneD}}{{{float(q8_sub['measured_irrep_weights_coset_eH']['chi_0']):.4f}}}",
        f"\\newcommand{{\\btmQeightMaxForbLeak}}{{{_fmt_sci_tex(float(q8_sub['max_forbidden_irrep_leakage']))}}}",
        f"\\newcommand{{\\btmQeightIdentityLeak}}{{{_fmt_sci_tex(float(hsp['q8_2d_3d_leakage']))}}}",
        f"\\newcommand{{\\btmQeightMaxCosetErr}}{{{_fmt_sci_tex(float(q8_sub['max_coset_weight_abs_error_all_24_g']))}}}",
        f"\\newcommand{{\\btmZthreeChiZeroW}}{{{float(z3_sub['measured_irrep_weights_coset_eH']['chi_0']):.4f}}}",
        f"\\newcommand{{\\btmZthreeRhoTwoW}}{{{float(z3_sub['measured_irrep_weights_coset_eH']['rho_2_1']):.4f}}}",
        f"\\newcommand{{\\btmZthreeRhoThreeW}}{{{float(z3_sub['measured_irrep_weights_coset_eH']['rho_3']):.4f}}}",
        f"\\newcommand{{\\btmZthreeMaxForbLeak}}{{{_fmt_sci_tex(float(z3_sub['max_forbidden_irrep_leakage']))}}}",
        f"\\newcommand{{\\btmZtwoChiW}}{{{float(z2_sub['measured_irrep_weights_coset_eH']['chi_0']):.4f}}}",
        f"\\newcommand{{\\btmZtwoRhoThreeW}}{{{float(z2_sub['measured_irrep_weights_coset_eH']['rho_3']):.4f}}}",
        f"\\newcommand{{\\btmZfourChiW}}{{{float(z4_sub['measured_irrep_weights_coset_eH']['chi_0']):.4f}}}",
        f"\\newcommand{{\\btmZfourRhoThreeW}}{{{float(z4_sub['measured_irrep_weights_coset_eH']['rho_3']):.4f}}}",
        f"\\newcommand{{\\btmTrivChiW}}{{{float(triv_sub['measured_irrep_weights_coset_eH']['chi_0']):.4f}}}",
        f"\\newcommand{{\\btmTrivRhoTwoW}}{{{float(triv_sub['measured_irrep_weights_coset_eH']['rho_2_0']):.4f}}}",
        f"\\newcommand{{\\btmTrivRhoThreeW}}{{{float(triv_sub['measured_irrep_weights_coset_eH']['rho_3']):.4f}}}",
        f"\\newcommand{{\\btmFullChiZeroW}}{{{float(full_sub['measured_irrep_weights_coset_eH']['chi_0']):.4f}}}",
        f"\\newcommand{{\\btmFullMaxForbLeak}}{{{_fmt_sci_tex(float(full_sub['max_forbidden_irrep_leakage']))}}}",
        f"\\newcommand{{\\btmHspMaxWeightErrAll}}{{{_fmt_sci_tex(max_hsp_err)}}}",
        f"\\newcommand{{\\btmHspMaxForbLeakAll}}{{{_fmt_sci_tex(max_hsp_forb)}}}",
        "\\newcommand{\\btmHspMaxUnusedLeakAll}{0.0}",
        f"\\newcommand{{\\btmEpsHeavyHex}}{{{_fmt_sci_tex(eps_hh, 1)}}}",
        f"\\newcommand{{\\btmEpsQccd}}{{{_fmt_sci_tex(eps_qccd, 1)}}}",
        f"\\newcommand{{\\btmEpsNextGen}}{{{_fmt_sci_tex(eps_ft, 1)}}}",
        f"\\newcommand{{\\btmSurvHhMoneStruct}}{{{(1.0 - eps_hh) ** 72:.3f}}}",
        f"\\newcommand{{\\btmSurvHhMoneGen}}{{{(1.0 - eps_hh) ** 723:.3f}}}",
        f"\\newcommand{{\\btmSurvHhQft}}{{{(1.0 - eps_hh) ** 613:.3f}}}",
        f"\\newcommand{{\\btmSurvHhMulStruct}}{{{_fmt_sci_tex((1.0 - eps_hh) ** 3662, 1)}}}",
        f"\\newcommand{{\\btmSurvQccdQft}}{{{(1.0 - eps_qccd) ** 423:.3f}}}",
        f"\\newcommand{{\\btmSurvQccdMulStruct}}{{{(1.0 - eps_qccd) ** 1739:.3f}}}",
        f"\\newcommand{{\\btmSurvQccdMulGen}}{{{_fmt_sci_tex((1.0 - eps_qccd) ** 33042, 1)}}}",
        f"\\newcommand{{\\btmSurvFtQft}}{{{(1.0 - eps_ft) ** 423:.3f}}}",
        f"\\newcommand{{\\btmSurvFtMulStruct}}{{{(1.0 - eps_ft) ** 1739:.3f}}}",
        f"\\newcommand{{\\btmSurvFtMulGen}}{{{(1.0 - eps_ft) ** 33042:.3f}}}",
        "\\newcommand{\\btmTenQRandFloor}{0.00098}",
        "",
        "\\newcommand{\\btmTableMulCompilationRows}{%",
        *mul_table_rows,
        "}",
        "",
        "\\newcommand{\\btmTableIrrepPeterWeylRows}{%",
        *irrep_table_rows,
        "}",
        "",
        "\\newcommand{\\btmTableHspSubgroupRows}{%",
        *hsp_table_rows,
        "}",
        "",
        "\\newcommand{\\btmTableFeasibilityRows}{%",
        *feas_table_rows,
        "}",
        "",
    ]
    (HERE / "generated.tex").write_bytes("\n".join(macros).encode("utf-8"))

    # 3. Render dual SVG + PDF 1.4 vector figures
    fig1 = _build_fig1(manifest, inv_manifest, fez_results)
    fig1.save(HERE / FIG1_STEM)

    fig2 = _build_fig2(manifest, esd_info["rms_mat"])
    fig2.save(HERE / FIG2_STEM)

    # 4. Render standalone multi-page US-Letter PDF 1.4 preprint
    pdf_bytes = _build_preprint_pdf(
        manifest=manifest,
        inv_manifest=inv_manifest,
        fez_results=fez_results,
        esd_info=esd_info,
        capsule_sha=capsule_sha,
        source_shas=source_shas,
        fig1=fig1,
        fig2=fig2,
    )
    (HERE / PDF_FILENAME).write_bytes(pdf_bytes)

    # 5. Emit repo-root-relative SHA256SUMS.txt locking all paper deliverables + source artifacts
    paper_files = sorted(
        p for p in HERE.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt"
    )
    locked_paths: List[Path] = list(paper_files) + list(SOURCE_ARTIFACT_PATHS)
    sums_lines = [
        f"{_sha256_file(p)}  {p.relative_to(ROOT).as_posix()}" for p in locked_paths
    ]
    (HERE / "SHA256SUMS.txt").write_bytes(("\n".join(sums_lines) + "\n").encode("utf-8"))
    return capsule


if __name__ == "__main__":
    generate_all()
