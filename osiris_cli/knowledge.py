"""What OSIRIS knows about its own project, read from Devin's files.

The mentor model has no idea what OSIRIS is or what has been measured; asked
"what do you know" it can only produce pleasantries (2026-09-29 transcript).
This module gives it two things, both read from disk, never written by a model:

  brief     a short, stable summary: the organism_sim scorecard verdicts (a
            generated, test-guarded table) and one line per project. Goes in
            the system prompt, which stays byte-identical between turns so
            Ollama can reuse its cached prefix.
  lookup    per-message retrieval (pure-Python BM25F over heading + body fields
            of the sources below, bounded by a 1,200-token retrieval budget,
            with optional Reciprocal Rank Fusion when dense olmo3-embed-7b
            rankings are available on Node Beta), returned with file paths so
            answers can cite where a fact came from.

Sources are ~/.osiris/living/knowledge.json {"sources": [...]} when present,
otherwise DEFAULT_SOURCES. The osiris-cli README is left out on purpose: it
frames staggered DD as "quantum advantage", which the audits do not support.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

HOME = os.path.expanduser("~")
LIVING_HOME = os.path.join(HOME, ".osiris", "living")

DEFAULT_SOURCES = [
    "docs/HONEST_ASSESSMENT.md",
    "docs/FORENSIC_AUDIT.md",
    "docs/NISQ_PATHWAYS.md",
    "docs/FLYWHEEL_ASSESSMENT.md",
    "docs/DNALANG_SALVAGE.md",
    "organism_sim/README.md",
    "organism_sim/docs/PROGRAM.md",
    "dnalang-core/README.md",
    "bridge/README.md",
    "osiris-cli/docs/research/NOTEBOOKLM_SYNTHESIS_AUDIT_2026_11_16.md",
]
SCORECARD_SOURCE = "organism_sim/README.md"

PROJECTS = """\
- organism_sim: rule-evolving ALife agents (XCS classifier systems) with pre-registered experiments.
- dnalang-core: dnalang 0.2, a from-scratch rewrite of the dna::}{::lang circuit language (Qiskit, OpenQASM 3, hash-chained ledger).
- bridge: an organism controller over dnalang's dynamical-decoupling search space, compared budget-matched with a GA.
- osiris-cli: this console; your core (osiris.nclm) trains on every conversation and speaks only after passing a held-out gate.
- Hardware: IBM Heron runs of GHZ witnesses and staggered dynamical decoupling, logged in a write-ahead ledger."""

CHUNK_CHARS = 900
RETRIEVAL_TOKEN_BUDGET = 1200
CHARS_PER_TOKEN = 4
BM25F_K1 = 1.5
BM25F_HEADING_WEIGHT = 2.5
BM25F_BODY_WEIGHT = 1.0
BM25F_B_HEADING = 0.4
BM25F_B_BODY = 0.75
RRF_K = 60
DENSE_MIN_FREE_RAM_MB = 12288

STOP = frozenset("""a an and are as at be but by can did do does for from has have how i if in is it its
me my of on or so that the their them then there these this to was we were what when where which who
why will with you your about tell know yourself osiris""".split())


def _tokens(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9][a-z0-9_\-]{1,}", text.lower()) if w not in STOP]


def estimate_tokens(text: str) -> int:
    """Deterministic lightweight token count estimate (~4 chars/token, min 1 for non-empty)."""
    if not text:
        return 0
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


def dense_retrieval_allowed(
    free_ram_mb: Optional[int] = None,
    is_termux: Optional[bool] = None,
) -> bool:
    """Guard heavy olmo3-embed-7b dense retrieval away from Termux/edge and low-RAM hosts."""
    if free_ram_mb is None or is_termux is None:
        from osiris_cli.probes import probe_substrate

        sub = probe_substrate()
        if is_termux is None:
            is_termux = sub.is_termux
        if free_ram_mb is None:
            free_ram_mb = sub.free_ram_mb
    if is_termux:
        return False
    return int(free_ram_mb) >= DENSE_MIN_FREE_RAM_MB


def reciprocal_rank_fusion(
    bm25_indices: Sequence[int],
    dense_indices: Sequence[int],
    k_rrf: int = RRF_K,
) -> List[Tuple[float, int]]:
    """Fuse lexical (BM25F) and dense rankings via RRF(d) = 1/(k+r_bm25) + 1/(k+r_dense).

    Returns sorted list of (fused_score, chunk_index) highest score first with
    deterministic tie-breaking by chunk_index.
    """
    scores: Dict[int, float] = {}
    for rank, idx in enumerate(bm25_indices, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k_rrf + rank)
    for rank, idx in enumerate(dense_indices, start=1):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k_rrf + rank)
    return sorted(((score, idx) for idx, score in scores.items()), key=lambda item: (-item[0], item[1]))


def _sources(living_home: str, base: str) -> List[str]:
    try:
        with open(os.path.join(living_home, "knowledge.json"), encoding="utf-8") as f:
            listed = json.load(f).get("sources") or DEFAULT_SOURCES
    except (OSError, ValueError):
        listed = DEFAULT_SOURCES
    resolved = []
    roots = (
        base,
        os.path.join(base, "osiris-repos"),
        os.path.join(base, "osiris-assessment", "work"),
        os.path.join(base, "osiris-assessment"),
    )
    for p in listed:
        if os.path.isabs(p):
            resolved.append(p)
            continue
        chosen = os.path.join(base, p)
        for r in roots:
            cand = os.path.join(r, p)
            if os.path.isfile(cand):
                chosen = cand
                break
        resolved.append(chosen)
    return resolved


def chunk_markdown(text: str) -> List[tuple]:
    """(heading, text) pieces of at most ~CHUNK_CHARS, split on headings then paragraphs."""
    out, heading, buf = [], "", ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para or para.startswith("<!--"):
            continue
        m = re.match(r"(#{1,4})\s+(.*)", para)
        if m and "\n" not in para:
            if buf:
                out.append((heading, buf))
            heading, buf = m.group(2).strip(), ""
            continue
        if buf and len(buf) + len(para) > CHUNK_CHARS:
            out.append((heading, buf))
            buf = ""
        buf = (buf + "\n\n" + para).strip() if buf else para[: CHUNK_CHARS * 2]
    if buf:
        out.append((heading, buf))
    return out


def scorecard_verdicts(readme_text: str) -> List[str]:
    """'experiment: VERDICT' for each row between the scorecard markers."""
    m = re.search(r"<!-- scorecard:start -->(.*?)<!-- scorecard:end -->", readme_text, re.S)
    if not m:
        return []
    rows = []
    for line in m.group(1).splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and cells[0] not in ("experiment", "") and not set(cells[0]) <= set("-"):
            rows.append(f"{cells[0]}: {cells[1]}")
    return rows


NOTE_PREFIX = "approved-note:"
_DOI_REF_RE = re.compile(
    r"(?:10\.5281/zenodo\.|zenodo(?:\.org/records?/|[\s.:/_-]+)|(?:\bzenodo\s+)?\brecord\s+)(\d{7,9})\b",
    re.IGNORECASE,
)


def _default_doi_lineage_path() -> str:
    env_home = os.environ.get("OSIRIS_HOME", "").strip()
    base_home = os.path.expanduser(env_home) if env_home else os.path.join(os.path.expanduser("~"), ".osiris")
    return os.path.join(base_home, "evidence", "doi_lineage.json")


def active_notes(living_home: str) -> List[dict]:
    """Retrieval notes Devin approved through paste learning (notes.jsonl), status
    "active" only: /unlearn rewrites a note's status to "tombstoned" in place, so a
    withdrawn note drops out on the next load. Malformed lines are skipped."""
    out = []
    try:
        with open(os.path.join(living_home, "notes.jsonl"), encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if (
                    isinstance(rec, dict)
                    and rec.get("status") == "active"
                    and rec.get("id")
                    and isinstance(rec.get("content"), str)
                ):
                    out.append(rec)
    except OSError:
        pass
    return out


class Knowledge:
    def __init__(
        self,
        living_home: str = LIVING_HOME,
        base: str = HOME,
        sources: Optional[List[str]] = None,
        notes: Optional[bool] = None,
    ):
        self.base = base
        paths = sources if sources is not None else _sources(living_home, base)
        self.texts: Dict[str, str] = {}
        for p in paths:
            try:
                with open(p, encoding="utf-8", errors="replace") as f:
                    self.texts[os.path.relpath(p, base)] = f.read()
            except OSError:
                continue
        self.chunks = [(src, h, t) for src, text in self.texts.items() for h, t in chunk_markdown(text)]
        # Approved notes join the lookup index only. They stay out of self.texts, so
        # brief() and on_record() (scorecard verdicts) are unchanged. Default: on unless
        # an explicit source list was given.
        self.note_ids: List[str] = []
        if notes if notes is not None else sources is None:
            for rec in active_notes(living_home):
                src = NOTE_PREFIX + rec["id"]
                title = str(rec.get("title") or "").strip()
                self.note_ids.append(rec["id"])
                self.chunks += [(src, h or title, t) for h, t in chunk_markdown(rec["content"])]

        # Field-separated BM25F statistics (heading boosted over body) + combined TF for compatibility.
        self._heading_tf = [Counter(_tokens(h)) for _, h, _ in self.chunks]
        self._body_tf = [Counter(_tokens(t)) for _, _, t in self.chunks]
        self._heading_len = [sum(tf.values()) for tf in self._heading_tf]
        self._body_len = [sum(tf.values()) for tf in self._body_tf]
        nonzero_h = [ln for ln in self._heading_len if ln > 0]
        nonzero_b = [ln for ln in self._body_len if ln > 0]
        self._heading_avg = (sum(nonzero_h) / len(nonzero_h)) if nonzero_h else 1.0
        self._body_avg = (sum(nonzero_b) / len(nonzero_b)) if nonzero_b else 1.0

        self._tf = [h_tf + b_tf for h_tf, b_tf in zip(self._heading_tf, self._body_tf)]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = (sum(self._len) / len(self._len)) if self._len else 1.0
        df = Counter(w for tf in self._tf for w in tf)
        n = len(self.chunks)
        self._idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def brief(self) -> str:
        parts = ["Devin's projects:\n" + PROJECTS]
        for src in sorted(self.texts, key=lambda k: not k.endswith(SCORECARD_SOURCE)):
            verdicts = scorecard_verdicts(self.texts[src])
            if verdicts:
                parts.append(
                    f"Pre-registered results so far (from {src}, generated from "
                    "results/ and test-guarded):\n" + "\n".join("- " + v for v in verdicts)
                )
                break
        return "\n\n".join(parts)

    def on_record(self, text: str) -> List[str]:
        """Scorecard rows for the experiments a message names ("what did M3b find?"
        -> "m3b: open-ended tasks, 12-16 bit, PARAMS16: PASS"), so code states the
        recorded verdict and a model paraphrase is not the only account of it.
        A row matches on its name before the first ':' as whole words, ignoring
        case and '-' vs ' ' ("substrate opt 2" finds "substrate-opt-2")."""
        flat = " " + re.sub(r"[-_]", " ", text.lower()) + " "
        rows = []
        for src in sorted(self.texts, key=lambda k: not k.endswith(SCORECARD_SOURCE)):
            for row in scorecard_verdicts(self.texts[src]):
                name = re.sub(r"[-_]", " ", row.split(":")[0].lower()).strip()
                if name and re.search(r"(?<![\w])" + re.escape(name) + r"(?![\w])", flat):
                    rows.append(row)
            if rows:
                return [f"{r}  ({src})" for r in rows]
        return []

    def _bm25f_scored(self, query: str) -> List[Tuple[float, int]]:
        q = set(_tokens(query))
        if not q or not self.chunks:
            return []
        scored: List[Tuple[float, int]] = []
        for i in range(len(self.chunks)):
            h_tf = self._heading_tf[i]
            b_tf = self._body_tf[i]
            h_norm = 1.0 - BM25F_B_HEADING + BM25F_B_HEADING * (self._heading_len[i] / self._heading_avg)
            b_norm = 1.0 - BM25F_B_BODY + BM25F_B_BODY * (self._body_len[i] / self._body_avg)
            s = 0.0
            for w in q:
                idf = self._idf.get(w)
                if idf is None:
                    continue
                fh = h_tf.get(w, 0)
                fb = b_tf.get(w, 0)
                if fh or fb:
                    pseudo_tf = (BM25F_HEADING_WEIGHT * fh / max(h_norm, 1e-6)) + (
                        BM25F_BODY_WEIGHT * fb / max(b_norm, 1e-6)
                    )
                    s += idf * pseudo_tf * (BM25F_K1 + 1.0) / (pseudo_tf + BM25F_K1)
            if s > 0.0:
                scored.append((s, i))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return scored

    def lookup(
        self,
        query: str,
        k: int = 3,
        per_source: int = 2,
        *,
        max_tokens: int = RETRIEVAL_TOKEN_BUDGET,
        dense_scores: Optional[Dict[int, float]] = None,
    ) -> List[dict]:
        """Retrieve up to `k` chunks ranked by BM25F (or BM25F + RRF when `dense_scores` is given),
        strictly bounded within `max_tokens` (default 1,200 tokens)."""
        bm25_scored = self._bm25f_scored(query)
        if dense_scores:
            bm25_order = [idx for _, idx in bm25_scored]
            dense_order = [
                idx
                for idx, _ in sorted(
                    ((i, float(sc)) for i, sc in dense_scores.items() if 0 <= i < len(self.chunks)),
                    key=lambda kv: (-kv[1], kv[0]),
                )
            ]
            ranked = reciprocal_rank_fusion(bm25_order, dense_order)
            score_precision = 4
        else:
            ranked = bm25_scored
            score_precision = 2

        if not ranked:
            return []

        out: List[dict] = []
        used = Counter()
        budget_used = 0
        for s, i in ranked:
            src, h, t = self.chunks[i]
            if used[src] >= per_source:
                continue
            chunk_tokens = estimate_tokens(h + "\n" + t)
            if max_tokens > 0 and out and budget_used + chunk_tokens > max_tokens:
                continue
            text_out = t
            if max_tokens > 0 and not out and chunk_tokens > max_tokens:
                max_chars = max(64, max_tokens * CHARS_PER_TOKEN - len(h) - 8)
                text_out = t[:max_chars].rstrip() + "..."
                chunk_tokens = estimate_tokens(h + "\n" + text_out)
            used[src] += 1
            budget_used += chunk_tokens
            out.append({"source": src, "heading": h, "text": text_out, "score": round(s, score_precision)})
            if len(out) >= k:
                break
        return out

    def lookup_hybrid(
        self,
        query: str,
        k: int = 3,
        per_source: int = 2,
        *,
        max_tokens: int = RETRIEVAL_TOKEN_BUDGET,
        dense_ranker: Optional[Callable[[str, Sequence[Tuple[str, str, str]]], Dict[int, float]]] = None,
        free_ram_mb: Optional[int] = None,
        is_termux: Optional[bool] = None,
    ) -> List[dict]:
        """Run BM25F, fusing optional dense rankings via RRF only when substrate RAM guards allow."""
        dense_scores: Optional[Dict[int, float]] = None
        if dense_ranker is not None and dense_retrieval_allowed(free_ram_mb=free_ram_mb, is_termux=is_termux):
            try:
                dense_scores = dense_ranker(query, self.chunks)
            except Exception:  # noqa: BLE001 - degrade gracefully to pure BM25F
                dense_scores = None
        return self.lookup(
            query,
            k=k,
            per_source=per_source,
            max_tokens=max_tokens,
            dense_scores=dense_scores,
        )

    @staticmethod
    def lookup_doi_warnings(text: str, lineage_path: Optional[str] = None) -> List[str]:
        """Return deterministic erratum/supersession warnings for any Zenodo DOIs cited in ``text``."""
        if not text:
            return []
        target_path = lineage_path or _default_doi_lineage_path()
        if not os.path.isfile(target_path):
            return []
        try:
            with open(target_path, encoding="utf-8") as fh:
                lineage_doc = json.load(fh)
        except (OSError, ValueError):
            return []
        records = lineage_doc.get("records") if isinstance(lineage_doc, dict) else None
        if not isinstance(records, dict):
            return []

        seen_ids: List[str] = []
        for match_obj in _DOI_REF_RE.finditer(text):
            rec_id = match_obj.group(1)
            if rec_id and rec_id not in seen_ids:
                seen_ids.append(rec_id)

        warnings: List[str] = []
        for rec_id in seen_ids:
            doi_key = f"10.5281/zenodo.{rec_id}"
            entry = records.get(doi_key) or records.get(rec_id)
            if not isinstance(entry, dict):
                continue
            status = str(entry.get("status") or "ACTIVE")
            warn_msg = str(entry.get("warning") or "").strip()
            if status != "ACTIVE" and warn_msg:
                warnings.append(f"[DOI LINEAGE WARNING — {doi_key} ({status})]: {warn_msg}")
        return warnings

    @staticmethod
    def format_notes(hits: List[dict], lineage_path: Optional[str] = None) -> str:
        def label(h: dict) -> str:
            tag = (
                "  (approved by Devin; user-provided, unverified -- not a measurement record)"
                if h["source"].startswith(NOTE_PREFIX)
                else ""
            )
            return f"[{h['source']}" + (f" § {h['heading']}" if h["heading"] else "") + "]" + tag

        rendered: List[str] = []
        for h in hits:
            body_text = str(h.get("text") or "")
            doi_warnings = Knowledge.lookup_doi_warnings(body_text, lineage_path=lineage_path)
            if doi_warnings:
                rendered.append(label(h) + "\n" + "\n".join(doi_warnings) + "\n" + body_text)
            else:
                rendered.append(label(h) + "\n" + body_text)
        return "\n\n".join(rendered)
