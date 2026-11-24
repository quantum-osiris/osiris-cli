"""Verification suite for standalone multi-page PDF 1.4 preprints across all 4 new REVTeX packages.

Verifies:
  1. Shared pure-Python `papers/_pdf_preprint_renderer.py` layout primitives and multi-page pagination
  2. Valid `%PDF-1.4` header, exact `xref` byte offsets, `/Catalog` -> `/Pages` tree, `%%EOF` trailer,
     stream `/Length` integrity, balanced `q`/`Q` and `BT`/`ET` operators, and `>= 3` pages per manuscript PDF
  3. Presence of verified hardware/simulation anchors and zero SYS-024 quarantined claims/constants
  4. Byte-for-byte determinism when re-executing each paper asset generator
  5. Clean `sha256sum -c SHA256SUMS.txt` verification across all 4 paper directories
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pytest
from osiris_cli import claim_provenance, claims, plan_gates
from papers._pdf_preprint_renderer import (
    PreprintDocument,
    VectorCanvas,
    pdf_escape,
    pdf_sanitize,
    wrap_text_to_width,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPERS_DIR = REPO_ROOT / "papers"

PAPER_SPECS: Sequence[Tuple[str, str, str, Tuple[str, ...], Tuple[str, ...]]] = (
    (
        "xtop006_non_clifford_kraus_fez",
        "generate_xtop006_paper_assets.py",
        "xtop006_non_clifford_kraus_fez.pdf",
        (
            "fig1_xtop006_fez_kraus_trajectory.pdf",
            "fig2_xtop006b_fez_4twirl_discriminator.pdf",
        ),
        (
            "dc1tqaanj2vs73dfrt40",
            "dc23ouqnj2vs73dg3bog",
            "184,320",
            "0.8568",
            "62.39",
        ),
    ),
    (
        "wp2_27q_debiased_ot_fez",
        "generate_wp2_paper_assets.py",
        "wp2_27q_debiased_ot_fez.pdf",
        (
            "fig1_27q_fez_sparsity_collapse_vs_eq7.pdf",
            "fig2_coherent_drift_calibration_and_heron_rescoring.pdf",
        ),
        (
            "dc1tqvinj2vs73dfrts0",
            "dau0q3qhcrkc73durtgg",
            "dbtl2vb2kf8s73c9a0d0",
            "98,304",
            "13.76",
        ),
    ),
    (
        "nonabelian_2bga_and_adv1",
        "generate_2bga_adv1_assets.py",
        "nonabelian_2bga_and_adv1.pdf",
        (
            "fig1_odd_k_nonabelian_2bga_codes_and_bposd.pdf",
            "fig2_adv1_three_attack_ablation_m6.pdf",
        ),
        (
            "SIMULATED",
            "provider_job_id: null",
            "6,480",
            "[[114,3,12]]",
            "adv1_chorded_m6_n30_L5",
        ),
    ),
    (
        "heron_r2_calibration_staleness_and_colored_nulls",
        "generate_unified_assets.py",
        "heron_r2_calibration_staleness_and_colored_nulls.pdf",
        (
            "fig1_tmap_sign_reversal_three_qpus.pdf",
            "fig2_ro06_07_staleness_persistence.pdf",
            "fig3_cn01_colored_noise_heldout_gate.pdf",
        ),
        (
            "2,174,976",
            "ibm_marrakesh",
            "ibm_fez",
            "ibm_kingston",
            "10.5281/zenodo.24420902",
        ),
    ),
)


def _audit_pdf14_bytes(pdf_bytes: bytes, label: str) -> Tuple[int, List[bytes]]:
    """Validate PDF 1.4 header, exact xref offsets, page tree, stream lengths, and operator stacks."""
    assert pdf_bytes.startswith(b"%PDF-1.4\n"), f"{label}: missing %PDF-1.4 header"
    assert pdf_bytes.endswith(b"%%EOF\n"), f"{label}: missing %%EOF trailer"

    m_startxref = re.search(rb"startxref\n(\d+)\n%%EOF\n\Z", pdf_bytes)
    assert m_startxref is not None, f"{label}: malformed startxref footer"
    xref_pos = int(m_startxref.group(1))
    assert pdf_bytes[xref_pos : xref_pos + 5] == b"xref\n", f"{label}: startxref {xref_pos} does not point to xref"

    tail = pdf_bytes[xref_pos:].decode("ascii")
    m_xref = re.match(
        r"xref\n0 (\d+)\n((?:\d{10} \d{5} [fn] \n)+)trailer\n<< /Size (\d+) /Root (\d+) 0 R(?: /Info \d+ 0 R)? >>\n",
        tail,
    )
    assert m_xref is not None, f"{label}: malformed xref/trailer block"
    xref_count = int(m_xref.group(1))
    trailer_size = int(m_xref.group(3))
    root_id = int(m_xref.group(4))
    assert xref_count == trailer_size, f"{label}: xref count {xref_count} != trailer /Size {trailer_size}"

    entries = m_xref.group(2).splitlines()
    assert len(entries) == xref_count
    assert entries[0] == "0000000000 65535 f "

    objects: Dict[int, bytes] = {}
    for obj_id, entry in enumerate(entries[1:], start=1):
        m_ent = re.fullmatch(r"(\d{10}) 00000 n ", entry)
        assert m_ent is not None, f"{label}: malformed xref entry {obj_id}: {entry!r}"
        off = int(m_ent.group(1))
        header = f"{obj_id} 0 obj\n".encode("ascii")
        assert pdf_bytes[off : off + len(header)] == header, (
            f"{label}: xref offset {off} does not match '{obj_id} 0 obj'"
        )
        end_pos = pdf_bytes.find(b"\nendobj\n", off + len(header))
        assert end_pos != -1, f"{label}: missing endobj for object {obj_id}"
        objects[obj_id] = pdf_bytes[off + len(header) : end_pos]

    cat_body = objects[root_id].decode("ascii")
    m_cat = re.fullmatch(r"<< /Type /Catalog /Pages (\d+) 0 R >>", cat_body)
    assert m_cat is not None, f"{label}: malformed /Catalog object: {cat_body!r}"
    pages_id = int(m_cat.group(1))

    pages_body = objects[pages_id].decode("ascii")
    m_pages = re.fullmatch(r"<< /Type /Pages /Kids \[([^\]]+)\] /Count (\d+) >>", pages_body)
    assert m_pages is not None, f"{label}: malformed /Pages object: {pages_body!r}"
    kid_ids = [int(tok.split()[0]) for tok in re.findall(r"\d+ 0 R", m_pages.group(1))]
    page_count = int(m_pages.group(2))
    assert page_count == len(kid_ids), f"{label}: /Count {page_count} != len(/Kids) {len(kid_ids)}"

    page_streams: List[bytes] = []
    for p_idx, pid in enumerate(kid_ids, start=1):
        p_body = objects[pid].decode("ascii")
        m_page = re.search(r"/Type /Page /Parent (\d+) 0 R /MediaBox \[0 0 ([0-9.]+) ([0-9.]+)\]", p_body)
        assert m_page is not None and int(m_page.group(1)) == pages_id, f"{label}: invalid /Page {pid}"
        m_cont = re.search(r"/Contents (\d+) 0 R", p_body)
        assert m_cont is not None, f"{label}: missing /Contents on page {p_idx}"
        c_raw = objects[int(m_cont.group(1))]
        m_stream = re.match(rb"<< /Length (\d+) >>\nstream\n", c_raw)
        assert m_stream is not None and c_raw.endswith(b"endstream"), f"{label}: invalid stream on page {p_idx}"
        decl_len = int(m_stream.group(1))
        stream_bytes = c_raw[m_stream.end() : -len(b"endstream")]
        assert len(stream_bytes) == decl_len, (
            f"{label} p{p_idx}: declared /Length {decl_len} != actual {len(stream_bytes)}"
        )

        # Strip literal strings before checking balanced q/Q and BT/ET operators
        no_literals = re.sub(r"\((?:\\.|[^\\()])*\)", "()", stream_bytes.decode("ascii"))
        q_push = len(re.findall(r"(?m)(?:^|\s)q(?:\s|$)", no_literals))
        q_pop = len(re.findall(r"(?m)(?:^|\s)Q(?:\s|$)", no_literals))
        bt_cnt = len(re.findall(r"\bBT\b", no_literals))
        et_cnt = len(re.findall(r"\bET\b", no_literals))
        assert q_push == q_pop, f"{label} p{p_idx}: unbalanced q/Q ({q_push} vs {q_pop})"
        assert bt_cnt == et_cnt, f"{label} p{p_idx}: unbalanced BT/ET ({bt_cnt} vs {et_cnt})"
        page_streams.append(stream_bytes)

    return page_count, page_streams


def _extract_pdf_text(streams: Sequence[bytes]) -> str:
    chunks: List[str] = []
    for raw in streams:
        for m in re.finditer(rb"\(((?:\\.|[^\\()])*)\) Tj", raw):
            s = (
                m.group(1)
                .decode("ascii")
                .replace("\\(", "(")
                .replace("\\)", ")")
                .replace("\\\\", "\\")
            )
            chunks.append(s)
    return " ".join(chunks)


def test_shared_pdf_preprint_renderer_primitives() -> None:
    assert pdf_sanitize("±1.5σ × Δθ → π/2") == "+/-1.5sigma x Delta theta -> pi/2".replace(
        "Delta theta", "Deltatheta"
    )
    assert pdf_escape("f(x) = a\\b") == "f\\(x\\) = a\\\\b"

    wrapped = wrap_text_to_width(" ".join(["quantum"] * 30), "/F4", 10.0, 180.0)
    assert len(wrapped) >= 5

    fig = VectorCanvas(400, 180)
    fig.rect(20, 20, 360, 140, fill="#fafafa", stroke="#111827")
    fig.polyline([(30, 140), (200, 70), (370, 40)], color="#2563eb")
    fig.circle(200, 70, 3.0, fill="#2563eb")
    fig.text(200, 15, "Unit Test Plot", bold=True, anchor="middle")

    doc = PreprintDocument(
        short_title="Unit Test Preprint",
        manuscript_id="TEST-001",
        execution_tier="SIMULATED",
        footer_left="Unit Test Footer",
    )
    doc.add_title_block(
        title="Deterministic Pure-Python Multi-Page Preprint Renderer Test",
        author="Devin Phillip Davis",
        affiliation="Agile Defense Systems LLC",
        date_str="November 23, 2026",
        metadata_rows=[("Tier", "SIMULATED"), ("Hash", "a" * 64)],
        abstract=" ".join(["Deterministic abstract verification sentence."] * 12),
    )
    doc.add_callout_box("SUMMARY CALLOUT", ["Point A verified.", "Point B verified."], tone="success")
    doc.add_section("1. Methods and Multi-Page Table Continuation")
    doc.add_paragraph("Body text following bold lead-in.", lead_in="Key Result:")
    doc.add_equation(["E[D_matched(X, Y)] = 0"], eq_num="1")
    doc.add_figure(fig, "Embedded vector figure caption.")
    doc.add_table(
        "TABLE I. Multi-row pagination stress test.",
        ["Row ID", "Metric A", "Metric B"],
        [[f"row_{i:02d}", f"{i * 0.125:.4f}", f"{1.0 / (i + 1):.5f}"] for i in range(55)],
    )
    doc.add_references(["[1] Reference entry one."])

    pdf_a = doc.to_pdf_bytes()
    pdf_b = doc.to_pdf_bytes()
    assert pdf_a == pdf_b
    pages, streams = _audit_pdf14_bytes(pdf_a, "unit_test_doc")
    assert pages >= 3
    txt = _extract_pdf_text(streams)
    assert f"Page 1 of {pages}" in txt and f"Page {pages} of {pages}" in txt


@pytest.mark.parametrize(
    ("pkg_name", "_script_fn", "pdf_fn", "fig_pdfs", "anchors"),
    PAPER_SPECS,
)
def test_manuscript_and_figure_pdfs_valid_structure_and_anchors(
    pkg_name: str,
    _script_fn: str,
    pdf_fn: str,
    fig_pdfs: Tuple[str, ...],
    anchors: Tuple[str, ...],
) -> None:
    pkg_dir = PAPERS_DIR / pkg_name
    manuscript_path = pkg_dir / pdf_fn
    assert manuscript_path.is_file() and manuscript_path.stat().st_size > 20_000

    page_count, streams = _audit_pdf14_bytes(manuscript_path.read_bytes(), f"{pkg_name}/{pdf_fn}")
    assert page_count >= 3, f"{pkg_name}/{pdf_fn} has {page_count} pages; expected >= 3"

    # Verify US-Letter MediaBox on manuscript
    assert b"/MediaBox [0 0 612.00 792.00]" in manuscript_path.read_bytes()

    text = _extract_pdf_text(streams)
    for page_idx in range(1, page_count + 1):
        assert f"Page {page_idx} of {page_count}" in text
    assert "Devin Phillip Davis" in text
    assert "Agile Defense Systems" in text
    for anchor in anchors:
        assert anchor in text, f"Missing empirical anchor {anchor!r} in {pkg_name}/{pdf_fn}"

    for bad_tok in ("TODO", "TBD", "FIXME", "XXX"):
        assert bad_tok not in text, f"Placeholder {bad_tok} found in {pkg_name}/{pdf_fn}"

    assert claims.flagged_in_reply(text) == []
    for q_id, q_re in claim_provenance._QUARANTINED_CONSTANTS:
        assert not q_re.search(text), f"Quarantined constant {q_id} matched in {pkg_name}/{pdf_fn}"
    for p_id, p_re in plan_gates._QUARANTINED_PLAN_PATTERNS:
        assert not p_re.search(text), f"Banned SYS-024 pattern {p_id} matched in {pkg_name}/{pdf_fn}"

    for fig_fn in fig_pdfs:
        fig_path = pkg_dir / fig_fn
        assert fig_path.is_file() and fig_path.stat().st_size > 1_000
        fig_pages, _ = _audit_pdf14_bytes(fig_path.read_bytes(), f"{pkg_name}/{fig_fn}")
        assert fig_pages == 1


@pytest.mark.parametrize(
    ("pkg_name", "script_fn", "pdf_fn", "fig_pdfs", "_anchors"),
    PAPER_SPECS,
)
def test_byte_for_byte_determinism_and_sha256sum_cli(
    pkg_name: str,
    script_fn: str,
    pdf_fn: str,
    fig_pdfs: Tuple[str, ...],
    _anchors: Tuple[str, ...],
) -> None:
    pkg_dir = PAPERS_DIR / pkg_name
    tracked_names = [
        pdf_fn,
        *fig_pdfs,
        "generated.tex",
        "reproducibility_capsule.json",
        "SHA256SUMS.txt",
    ]
    before = {name: (pkg_dir / name).read_bytes() for name in tracked_names}

    proc = subprocess.run(
        [sys.executable, str(pkg_dir / script_fn)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"{script_fn} failed:\n{proc.stderr}\n{proc.stdout}"

    after = {name: (pkg_dir / name).read_bytes() for name in tracked_names}
    for name in tracked_names:
        assert after[name] == before[name], f"Non-deterministic output in {pkg_name}/{name}"

    sums_path = pkg_dir / "SHA256SUMS.txt"
    sums_text = sums_path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in sums_text.splitlines() if ln.strip()]
    assert lines, f"Empty SHA256SUMS.txt in {pkg_name}"

    listed_targets = [ln.split("  ", 1)[1] for ln in lines]
    listed_basenames = {Path(t).name for t in listed_targets}
    assert pdf_fn in listed_basenames, f"{pdf_fn} missing from {pkg_name}/SHA256SUMS.txt"
    for fig_fn in fig_pdfs:
        assert fig_fn in listed_basenames, f"{fig_fn} missing from {pkg_name}/SHA256SUMS.txt"

    sha256sum_bin = shutil.which("sha256sum")
    assert sha256sum_bin is not None, "sha256sum binary required for manifest verification"
    use_repo_root = "/" in listed_targets[0]
    check_cwd = REPO_ROOT if use_repo_root else pkg_dir
    check_arg = str(sums_path.relative_to(REPO_ROOT)) if use_repo_root else "SHA256SUMS.txt"

    chk = subprocess.run(
        [sha256sum_bin, "-c", check_arg],
        cwd=str(check_cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    assert chk.returncode == 0, (
        f"sha256sum -c failed for {pkg_name}:\n{chk.stderr}\n{chk.stdout}"
    )
