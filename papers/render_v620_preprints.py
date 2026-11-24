#!/usr/bin/env python3
"""Deterministic standalone PDF 1.4 preprint renderer for the four v6.2.0 REVTeX
manuscript packages that ship `.tex` + `generated.tex` without an in-directory `.pdf`.

Outputs are written strictly under `papers/zenodo_v620_deposits/<slug>/<stem>.pdf` so
that existing `papers/<dir>/` file sets and `SHA256SUMS.txt` manifests remain 100%
untouched and byte-identical.

Parses `\newcommand` macro definitions from `generated.tex` and full prose, equations,
and tables from each manuscript `.tex`, converting LaTeX markup to clean academic
typography across multi-page US-Letter (612 x 792 pt) PDF 1.4 documents.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from papers.hp004_joint_stabilizer_heron_r2.generate_paper_assets import (  # noqa: E402
    _pdf_escape,
    build_pdf_document,
)

DEPOSITS_ROOT = REPO_ROOT / "papers" / "zenodo_v620_deposits"

MANUSCRIPTS = {
    "heron_r2_unified": (
        REPO_ROOT / "papers" / "heron_r2_calibration_staleness_and_colored_nulls",
        "heron_r2_calibration_staleness_and_colored_nulls",
    ),
    "wp2_27q_ot": (
        REPO_ROOT / "papers" / "wp2_27q_debiased_ot_fez",
        "wp2_27q_debiased_ot_fez",
    ),
    "nonabelian_2bga_adv1": (
        REPO_ROOT / "papers" / "nonabelian_2bga_and_adv1",
        "nonabelian_2bga_and_adv1",
    ),
    "xtop006_fez": (
        REPO_ROOT / "papers" / "xtop006_non_clifford_kraus_fez",
        "xtop006_non_clifford_kraus_fez",
    ),
}


def _strip_comments(tex: str) -> str:
    out: List[str] = []
    for line in tex.splitlines():
        idx = 0
        while True:
            pos = line.find("%", idx)
            if pos == -1:
                out.append(line)
                break
            bs = 0
            k = pos - 1
            while k >= 0 and line[k] == "\\":
                bs += 1
                k -= 1
            if bs % 2 == 0:
                out.append(line[:pos])
                break
            idx = pos + 1
    return "\n".join(out)


def parse_generated_macros(gen_tex: str) -> Dict[str, str]:
    """Extract all `\\newcommand{\\Name}{Body}` pairs with balanced-brace scanning."""
    cleaned = _strip_comments(gen_tex)
    macros: Dict[str, str] = {}
    i = 0
    n = len(cleaned)
    prefix = "\\newcommand{"
    while i < n:
        pos = cleaned.find(prefix, i)
        if pos == -1:
            break
        j = pos + len(prefix)
        end_name = cleaned.find("}", j)
        if end_name == -1:
            break
        cmd_name = cleaned[j:end_name].strip()
        k = end_name + 1
        while k < n and cleaned[k].isspace():
            k += 1
        if k >= n or cleaned[k] != "{":
            i = k
            continue
        depth = 1
        p = k + 1
        while p < n and depth > 0:
            if cleaned[p] == "\\":
                p += 2
                continue
            if cleaned[p] == "{":
                depth += 1
            elif cleaned[p] == "}":
                depth -= 1
            p += 1
        body = cleaned[k + 1 : p - 1]
        macros[cmd_name] = body.strip()
        i = p
    return macros


def expand_macros(tex: str, macros: Dict[str, str]) -> str:
    """Iteratively expand generated macros (longest command name first)."""
    ordered = sorted(macros.items(), key=lambda kv: -len(kv[0]))
    out = tex
    for _ in range(4):
        changed = False
        for cmd, val in ordered:
            # Replace \Cmd{} and \Cmd\ and bare \Cmd followed by non-letter
            pat = re.compile(re.escape(cmd) + r"(?:\{\}|\\(?=\s)|(?![A-Za-z]))")
            new_out = pat.sub(lambda _m, v=val: v, out)
            if new_out != out:
                out = new_out
                changed = True
        if not changed:
            break
    return out


_GREEK_AND_MATH = {
    r"\alpha": "alpha",
    r"\beta": "beta",
    r"\gamma": "gamma",
    r"\delta": "delta",
    r"\Delta": "Delta ",
    r"\varepsilon": "eps",
    r"\epsilon": "eps",
    r"\theta": "theta",
    r"\vartheta": "theta",
    r"\varphi": "phi",
    r"\phi": "phi",
    r"\Phi": "Phi",
    r"\psi": "psi",
    r"\Psi": "Psi",
    r"\rho": "rho",
    r"\sigma": "sigma",
    r"\tau": "tau",
    r"\chi": "chi",
    r"\pi": "pi",
    r"\mu": "u",
    r"\pm": " +/- ",
    r"\mp": " -/+ ",
    r"\times": " x ",
    r"\cdot": "*",
    r"\le": " <= ",
    r"\leq": " <= ",
    r"\ge": " >= ",
    r"\geq": " >= ",
    r"\neq": " != ",
    r"\ne": " != ",
    r"\approx": " ~ ",
    r"\equiv": " == ",
    r"\sim": " ~ ",
    r"\ll": " << ",
    r"\gg": " >> ",
    r"\in": " in ",
    r"\notin": " not-in ",
    r"\subseteq": " subseteq ",
    r"\setminus": " \\ ",
    r"\to": " -> ",
    r"\rightarrow": " -> ",
    r"\longrightarrow": " --> ",
    r"\mapsto": " |-> ",
    r"\leftrightarrow": " <-> ",
    r"\otimes": " (x) ",
    r"\oplus": " (+) ",
    r"\rtimes": " |x ",
    r"\mid": " | ",
    r"\ldots": "...",
    r"\dots": "...",
    r"\infty": "inf",
    r"\dagger": "+",
    r"\top": "^T",
    r"\star": "*",
    r"\circ": " deg",
    r"\langle": "<",
    r"\rangle": ">",
    r"\lvert": "|",
    r"\rvert": "|",
    r"\|": "||",
    r"\_": "_",
    r"\%": "%",
    r"\&": "&",
    r"\#": "#",
    r"\,": " ",
    r"\;": " ",
    r"\!": "",
    r"\quad": "  ",
    r"\qquad": "    ",
    r"~": " ",
    r"---": " -- ",
    r"--": "-",
    r"``": '"',
    r"''": '"',
}


def clean_latex_inline(text: str) -> str:
    s = text
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    s = re.sub(r"\\cite\{([^}]*)\}", r"[\1]", s)
    s = re.sub(r"\\eqref\{([^}]*)\}", r"(\1)", s)
    s = re.sub(r"\\ref\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\tfrac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    for _ in range(3):
        s = re.sub(r"\\(?:texttt|textbf|emph|textrm|mathrm|mathcal|mathbb|bm)\{([^{}]*)\}", r"\1", s)
        s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    for k, v in _GREEK_AND_MATH.items():
        s = s.replace(k, v)
    s = re.sub(r"\\(?:left|right|bigl|bigr|Bigl|Bigr|big|Big)\b", "", s)
    s = s.replace("$", "")
    s = s.replace("{,}", ",")
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\\([A-Za-z]+)", r"\1", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def wrap_line(text: str, width: int) -> List[str]:
    words = text.split()
    if not words:
        return [""]
    lines: List[str] = []
    cur = words[0]
    for w in words[1:]:
        if len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def render_manuscript_to_pdf(paper_dir: Path, stem: str) -> bytes:
    tex_raw = (paper_dir / f"{stem}.tex").read_text(encoding="utf-8")
    gen_raw = (paper_dir / "generated.tex").read_text(encoding="utf-8")
    macros = parse_generated_macros(gen_raw)
    expanded = expand_macros(_strip_comments(tex_raw), macros)

    title_m = re.search(r"\\title\{((?:[^{}]|\{[^{}]*\})*)\}", expanded, re.S)
    author_m = re.search(r"\\author\{([^}]*)\}", expanded)
    affil_m = re.search(r"\\affiliation\{([^}]*)\}", expanded)
    date_m = re.search(r"\\date\{([^}]*)\}", expanded)
    abs_m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", expanded, re.S)

    title = clean_latex_inline(title_m.group(1)) if title_m else stem
    author = clean_latex_inline(author_m.group(1)) if author_m else "Devin Phillip Davis"
    affil = clean_latex_inline(affil_m.group(1)) if affil_m else "Agile Defense Systems LLC"
    pub_date = clean_latex_inline(date_m.group(1)) if date_m else "November 2026"
    abstract = clean_latex_inline(abs_m.group(1)) if abs_m else ""

    # Extract body after \maketitle (or \end{abstract}) up to \end{document}
    body_start = expanded.find(r"\maketitle")
    if body_start != -1:
        body = expanded[body_start + len(r"\maketitle") :]
    elif abs_m:
        body = expanded[abs_m.end() :]
    else:
        body = expanded
    end_doc = body.find(r"\end{document}")
    if end_doc != -1:
        body = body[:end_doc]

    # Convert structured blocks (tables, figures, equations, bibliography, sections, lists) into styled line items
    # Each item: (style, text) where style in {"H1", "H2", "BODY", "MONO", "CAPTION", "RULE", "SPACER"}
    blocks: List[Tuple[str, str]] = []

    # Separate bibliography if present
    bib_text = ""
    bib_m = re.search(r"\\begin\{thebibliography\}\{[^}]*\}(.*?)\\end\{thebibliography\}", body, re.S)
    if bib_m:
        bib_text = bib_m.group(1)
        body = body[: bib_m.start()] + body[bib_m.end() :]

    # Tokenize environments (table*, table, figure*, figure, equation, align) vs normal paragraphs
    env_pat = re.compile(
        r"\\begin\{(table\*?|figure\*?|equation\*?|align\*?|theorem|proposition|proof|acknowledgments)\}(.*?)\\end\{\1\}",
        re.S,
    )

    pos = 0
    sec_num = 0
    sub_num = 0

    def _emit_prose_chunk(chunk: str) -> None:
        nonlocal sec_num, sub_num
        # Split on section/subsection/paragraph headers
        header_pat = re.compile(r"\\(section\*?|subsection\*?|paragraph)\{((?:[^{}]|\{[^{}]*\})*)\}")
        cpos = 0
        for hm in header_pat.finditer(chunk):
            pre = chunk[cpos : hm.start()].strip()
            if pre:
                _emit_paragraphs(pre)
            kind = hm.group(1)
            htitle = clean_latex_inline(hm.group(2))
            if kind == "section":
                sec_num += 1
                sub_num = 0
                blocks.append(("SPACER", ""))
                blocks.append(("H1", f"{sec_num}. {htitle.upper()}"))
            elif kind == "section*":
                blocks.append(("SPACER", ""))
                blocks.append(("H1", htitle.upper()))
            elif kind.startswith("subsection"):
                sub_num += 1
                blocks.append(("SPACER", ""))
                blocks.append(("H2", f"{sec_num}.{sub_num} {htitle}"))
            else:
                blocks.append(("H2", htitle))
            cpos = hm.end()
        tail = chunk[cpos:].strip()
        if tail:
            _emit_paragraphs(tail)

    def _emit_paragraphs(txt: str) -> None:
        txt = re.sub(r"\\(?:begin|end)\{(?:itemize|enumerate)\}", "\n", txt)
        txt = re.sub(r"\\item\b", "\n  * ", txt)
        for para in re.split(r"\n\s*\n", txt):
            cleaned = clean_latex_inline(para)
            if cleaned:
                blocks.append(("BODY", cleaned))

    for em in env_pat.finditer(body):
        pre = body[pos : em.start()]
        if pre.strip():
            _emit_prose_chunk(pre)
        env_name = em.group(1)
        env_body = em.group(2)
        if env_name.startswith("table"):
            cap_m = re.search(r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}", env_body, re.S)
            blocks.append(("SPACER", ""))
            if cap_m:
                blocks.append(("CAPTION", "TABLE: " + clean_latex_inline(cap_m.group(1))))
            blocks.append(("RULE", ""))
            tab_m = re.search(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", env_body, re.S)
            if tab_m:
                raw_rows = tab_m.group(1).split(r"\\")
                for rr in raw_rows:
                    rr_clean = re.sub(r"\\(?:hline|toprule|midrule|bottomrule|addlinespace)\b", "", rr).strip()
                    if not rr_clean:
                        continue
                    cols = [clean_latex_inline(c) for c in rr_clean.split("&")]
                    blocks.append(("MONO", " | ".join(cols)))
            blocks.append(("RULE", ""))
            blocks.append(("SPACER", ""))
        elif env_name.startswith("figure"):
            img_m = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", env_body)
            cap_m = re.search(r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}", env_body, re.S)
            blocks.append(("SPACER", ""))
            fig_ref = f"[Companion Vector Figure: {img_m.group(1)}] " if img_m else "[Figure] "
            cap_txt = clean_latex_inline(cap_m.group(1)) if cap_m else ""
            blocks.append(("CAPTION", fig_ref + cap_txt))
            blocks.append(("SPACER", ""))
        elif env_name in ("theorem", "proposition", "proof", "acknowledgments"):
            blocks.append(("H2", env_name.upper()))
            _emit_paragraphs(env_body)
        else:
            # equation / align
            eq_lines = [clean_latex_inline(x) for x in env_body.split(r"\\")]
            for el in eq_lines:
                if el:
                    blocks.append(("MONO", "    " + el))
        pos = em.end()

    if body[pos:].strip():
        _emit_prose_chunk(body[pos:])

    if bib_text.strip():
        blocks.append(("SPACER", ""))
        blocks.append(("H1", "REFERENCES"))
        items = re.split(r"\\bibitem\{([^}]*)\}", bib_text)
        # items = [pre, key1, text1, key2, text2, ...]
        for idx in range(1, len(items) - 1, 2):
            bkey = items[idx].strip()
            bval = clean_latex_inline(items[idx + 1])
            blocks.append(("CAPTION", f"[{bkey}] {bval}"))

    # Paginate cleanly onto US-Letter (612 x 792 pt)
    W, H = 612.0, 792.0
    TOP_Y = 50.0
    BOT_Y = 744.0
    pages_cmds: List[List[str]] = []
    cur_cmds: List[str] = ["1 1 1 rg 0 0 612 792 re f"]
    y = TOP_Y

    def _new_page() -> None:
        nonlocal cur_cmds, y
        pno = len(pages_cmds) + 1
        cur_cmds.append(f"BT /F3 8.0 Tf 0.40 0.42 0.46 rg 292.00 28.00 Td (- {pno} -) Tj ET")
        pages_cmds.append(cur_cmds)
        cur_cmds = ["1 1 1 rg 0 0 612 792 re f"]
        y = TOP_Y

    def _ensure(need: float) -> None:
        if y + need > BOT_Y:
            _new_page()

    def _put(line: str, font: str, size: float, leading: float, x: float = 50.0, rgb: str = "0.07 0.09 0.15") -> None:
        nonlocal y
        _ensure(leading)
        esc = _pdf_escape(line)
        cur_cmds.append(f"BT {font} {size:.1f} Tf {rgb} rg {x:.2f} {H - y:.2f} Td ({esc}) Tj ET")
        y += leading

    # Title block
    for tline in wrap_line(title, 74):
        _put(tline, "/F5", 13.0, 16.6)
    y += 4
    _put(f"{author}  |  {affil}  |  {pub_date}", "/F3", 7.8, 11.2, rgb="0.22 0.25 0.32")
    _put(f"OSIRIS v6.2.0 Reproducibility Archive  |  Source: papers/{paper_dir.name}/{stem}.tex", "/F3", 7.4, 11.0, rgb="0.28 0.32 0.38")
    y += 4
    cur_cmds.append(f"0.7 w 0.60 0.60 0.60 RG 50 {H - y:.2f} m 562 {H - y:.2f} l S")
    y += 12

    if abstract:
        _put("ABSTRACT", "/F5", 9.8, 13.0)
        for aline in wrap_line(abstract, 96):
            _put(aline, "/F4", 9.1, 12.0, x=54.0)
        y += 8
        cur_cmds.append(f"0.5 w 0.75 0.75 0.75 RG 50 {H - y:.2f} m 562 {H - y:.2f} l S")
        y += 10

    for style, content in blocks:
        if style == "SPACER":
            y += 5.0
        elif style == "RULE":
            _ensure(8.0)
            cur_cmds.append(f"0.5 w 0.65 0.68 0.72 RG 50 {H - y:.2f} m 562 {H - y:.2f} l S")
            y += 6.0
        elif style == "H1":
            _ensure(26.0)
            for hl in wrap_line(content, 82):
                _put(hl, "/F5", 10.2, 13.8)
        elif style == "H2":
            _ensure(22.0)
            for hl in wrap_line(content, 88):
                _put(hl, "/F5", 9.4, 12.6)
        elif style == "MONO":
            for ml in wrap_line(content, 118):
                _put(ml, "/F3", 6.8, 9.6, x=52.0, rgb="0.10 0.14 0.22")
        elif style == "CAPTION":
            for cl in wrap_line(content, 98):
                _put(cl, "/F6", 8.6, 11.4, x=52.0, rgb="0.18 0.22 0.28")
            y += 2.0
        else:
            for bl in wrap_line(content, 96):
                _put(bl, "/F4", 9.3, 12.2)
            y += 3.5

    pno = len(pages_cmds) + 1
    cur_cmds.append(f"BT /F3 8.0 Tf 0.40 0.42 0.46 rg 292.00 28.00 Td (- {pno} -) Tj ET")
    pages_cmds.append(cur_cmds)

    pdf_pages = [(W, H, ("\n".join(cmds) + "\n").encode("latin-1", errors="replace")) for cmds in pages_cmds]
    return build_pdf_document(pdf_pages)


def render_all() -> Dict[str, Dict[str, object]]:
    summary: Dict[str, Dict[str, object]] = {}
    for slug, (pdir, stem) in MANUSCRIPTS.items():
        out_dir = DEPOSITS_ROOT / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_bytes = render_manuscript_to_pdf(pdir, stem)
        out_pdf = out_dir / f"{stem}.pdf"
        out_pdf.write_bytes(pdf_bytes)
        sha = hashlib.sha256(pdf_bytes).hexdigest()
        summary[slug] = {
            "pdf_path": str(out_pdf.relative_to(REPO_ROOT)),
            "bytes": len(pdf_bytes),
            "sha256": sha,
        }
    return summary


if __name__ == "__main__":
    res = render_all()
    for slug, info in res.items():
        print(f"{slug:22s} -> {info['pdf_path']} ({info['bytes']:,} bytes, sha256={str(info['sha256'])[:16]}...)")
