"""Stage 4 offline tests: BM25F heading-weighted retrieval, 1,200-token budget cap,
Reciprocal Rank Fusion (RRF), RAM/Termux dense guard, and post-generation grounding auditor."""

from __future__ import annotations

import json
from pathlib import Path

from osiris_cli.knowledge import (
    Knowledge,
    RETRIEVAL_TOKEN_BUDGET,
    dense_retrieval_allowed,
    estimate_tokens,
    reciprocal_rank_fusion,
)
from osiris_cli.living import GroundingAudit, Osiris, audit_mentor_reply
from osiris_cli.probes import Probes, verify_chain, verify_chain_tail


class _ScriptedMentor:
    base = "scripted://"

    def __init__(self, reply: str, model: str = "qwen2.5:7b") -> None:
        self._reply = reply
        self._model = model
        self.last_done = {"done_reason": "stop"}

    def model(self) -> str:
        return self._model

    def stream(self, messages, model=None):
        yield self._reply


class _FakeCore:
    name = "fake.nclm"

    def __init__(self) -> None:
        self.learned: list[str] = []
        self._step = 10

    def step(self) -> int:
        return self._step

    def bits_per_byte(self, text: str) -> float:
        return 4.2

    def learn(self, lesson: str, steps: int) -> float:
        self.learned.append(lesson)
        self._step += steps
        return 3.0


def test_bm25f_heading_boost_and_multi_term_recall_beats_naive_substring(tmp_path: Path) -> None:
    doc1 = tmp_path / "docs" / "decoders.md"
    doc2 = tmp_path / "docs" / "noise.md"
    doc1.parent.mkdir(parents=True, exist_ok=True)
    doc1.write_text(
        "# qLDPC BP-OSD Decoding Latency\n\n"
        "Hardware FPGA decoders achieve bounded syndrome turnaround under surface-code schedules.\n\n"
        "## Surface Code Thresholds\n\n"
        "Baseline distance-5 lattices tolerate physical error rates near one percent.\n",
        encoding="utf-8",
    )
    doc2.write_text(
        "# General Notes\n\n"
        "Latency was briefly mentioned here in passing without decoder details.\n",
        encoding="utf-8",
    )

    kb = Knowledge(base=str(tmp_path), sources=[str(doc1), str(doc2)], notes=False)
    hits = kb.lookup("qLDPC decoding latency FPGA", k=3)
    assert hits
    assert hits[0]["source"] == "docs/decoders.md"
    assert "qLDPC" in hits[0]["heading"]


def test_bm25f_enforces_1200_token_budget_cap(tmp_path: Path) -> None:
    doc = tmp_path / "long_notes.md"
    sections = []
    for idx in range(10):
        body = f"quantum syndrome decoder metric section {idx} " * 45
        sections.append(f"## Decoder Section {idx}\n\n{body}\n")
    doc.write_text("\n".join(sections), encoding="utf-8")

    kb = Knowledge(base=str(tmp_path), sources=[str(doc)], notes=False)
    hits = kb.lookup("quantum syndrome decoder metric", k=10, per_source=10, max_tokens=200)
    assert hits
    total_tokens = sum(estimate_tokens(h["heading"] + "\n" + h["text"]) for h in hits)
    assert total_tokens <= 200
    assert RETRIEVAL_TOKEN_BUDGET == 1200


def test_reciprocal_rank_fusion_and_termux_ram_dense_guard(tmp_path: Path) -> None:
    fused = reciprocal_rank_fusion([0, 1, 2], [2, 0, 1], k_rrf=60)
    # Index 0: 1/61 + 1/62; Index 2: 1/63 + 1/61 -> 0 ranks above 2, both above 1 (1/62 + 1/63)
    assert [idx for _, idx in fused] == [0, 2, 1]

    assert dense_retrieval_allowed(free_ram_mb=16384, is_termux=True) is False
    assert dense_retrieval_allowed(free_ram_mb=4096, is_termux=False) is False
    assert dense_retrieval_allowed(free_ram_mb=16384, is_termux=False) is True

    doc = tmp_path / "notes.md"
    doc.write_text(
        "# Alpha Topic\n\nQuantum error mitigation baseline.\n\n"
        "# Beta Topic\n\nQuantum error mitigation with dense semantic match.\n",
        encoding="utf-8",
    )
    kb = Knowledge(base=str(tmp_path), sources=[str(doc)], notes=False)

    ranker_called: list[bool] = []

    def dense_ranker(query, chunks):
        ranker_called.append(True)
        return {1: 0.99, 0: 0.10}

    # On Node Alpha (Termux), dense ranker MUST NOT be called.
    kb.lookup_hybrid("quantum mitigation", dense_ranker=dense_ranker, free_ram_mb=16384, is_termux=True)
    assert ranker_called == []

    # On Node Beta with >= 12 GiB RAM, dense ranker fuses via RRF.
    hybrid_hits = kb.lookup_hybrid("quantum mitigation", dense_ranker=dense_ranker, free_ram_mb=16384, is_termux=False)
    assert ranker_called == [True]
    assert len(hybrid_hits) == 2


def test_audit_mentor_reply_catches_phantom_citations_unbacked_metrics_and_contradictions() -> None:
    corpus = "Measured ledger baseline: 4.96 bits/byte from docs/HONEST_ASSESSMENT.md.\n"
    retrieved = {"docs/HONEST_ASSESSMENT.md"}

    ok = audit_mentor_reply(
        "Per `docs/HONEST_ASSESSMENT.md`, baseline is 4.96 bits/byte and m3b passed.",
        retrieved_files=retrieved,
        context_corpus_text=corpus,
        on_record=["m3b: open-ended tasks: PASS  (organism_sim/README.md)"],
    )
    assert isinstance(ok, GroundingAudit)
    assert ok.passed is True
    assert ok.phantom_files == ()
    assert ok.unverified_numbers == ()
    assert ok.contradictions == ()

    bad = audit_mentor_reply(
        "According to [source: docs/FABRICATED.md] and `osiris_cli/ghost.py`, "
        "we achieved 1.2345 bits/byte with nDCG@10 = 0.9876, and m3b failed.",
        retrieved_files=retrieved,
        context_corpus_text=corpus,
        on_record=["m3b: open-ended tasks: PASS  (organism_sim/README.md)"],
    )
    assert bad.passed is False
    assert set(bad.phantom_files) == {"docs/FABRICATED.md", "osiris_cli/ghost.py"}
    assert any("1.2345" in m for m in bad.unverified_numbers)
    assert any("0.9876" in m for m in bad.unverified_numbers)
    assert any("m3b" in c for c in bad.contradictions)
    assert "[GOVERNANCE WARNING]" in bad.warning_banner()


def test_living_converse_blocks_training_on_ungrounded_reply_and_preserves_chain(tmp_path: Path) -> None:
    core = _FakeCore()
    out_buf: list[str] = []
    # Exchange index 0 is held out; make turn 0 clean so turn 1 is a train-eligible exchange.
    clean_mentor = _ScriptedMentor("All systems nominal.")
    o = Osiris(
        core=core,
        mentor=clean_mentor,
        home=str(tmp_path / "living"),
        out=out_buf.append,
        background=False,
        tty=False,
        probes=Probes(str(tmp_path / "living"), base=str(tmp_path)),
    )
    o.converse("status check")

    # Turn 1 (index=1, non-heldout): mentor fabricates citation + precision bpb metric
    hallucinating_mentor = _ScriptedMentor("Per `docs/PHANTOM_FILE.md`, loss hit 0.4215 bits/byte.")
    o.mentor = hallucinating_mentor
    o.converse("what is our git status?")

    rendered = "".join(out_buf)
    assert "[GOVERNANCE WARNING]" in rendered
    assert "docs/PHANTOM_FILE.md" in rendered
    assert "0.4215 bits/byte" in rendered
    # Core must NOT train on the ungrounded reply.
    assert core.learned == []

    rows = [json.loads(ln) for ln in Path(o.log_path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) == 2
    bad_row = rows[1]
    assert bad_row["distill_eligible"] is False
    assert bad_row["learnable"] is False
    assert bad_row["grounding_audit"]["passed"] is False
    assert "docs/PHANTOM_FILE.md" in bad_row["grounding_audit"]["phantom_files"]
    assert "probe_receipts" in bad_row and len(bad_row["probe_receipts"]) >= 1

    # Hash chain must remain 100% intact for both verify_chain and O(1) verify_chain_tail
    assert verify_chain(o.log_path)["intact"] is True
    assert verify_chain_tail(o.log_path).intact is True
