"""Regression tests for `/review` chunking, probe isolation, DOI filtering, and Penteract claim precision."""

from __future__ import annotations

import json
from typing import List

from osiris_cli import claims, living, osiris_repl, probes
from osiris_cli.osiris_repl import OsirisReplState, dispatch_command


class _StubCore:
    name = "stub-core"
    learned: List[str] = []

    def step(self) -> int:
        return 45

    def training_locked(self) -> bool:
        return False


class _RecordingProbes:
    def __init__(self) -> None:
        self.calls: List[str] = []

    def run(self, text: str):
        self.calls.append(text)
        return [{"name": "trainer", "cmd": "osiris train --status", "out": "step 45"}]

    @staticmethod
    def format(checks) -> str:
        return "\n".join(c["out"] for c in checks)


class _SelectiveTokenCapMentor:
    """Simulates hitting the 2048-token cap on chunk #1 while succeeding cleanly on chunk #2."""

    local_review_allowed = True

    def __init__(self) -> None:
        self.calls = 0
        self.last_done = {}

    def model(self) -> str:
        return "qwen2.5-coder:7b"

    def fast_model(self):
        return None

    def stream(self, messages, model=None):
        self.calls += 1
        if self.calls == 1:
            self.last_done = {"done_reason": "length"}
            yield "Truncated at 2048-token reply limit."
        else:
            self.last_done = {}
            yield f"Clean review of call {self.calls}."


def test_unfenced_python_and_bash_comments_do_not_explode_review_sections() -> None:
    pasted = (
        "## 1. Benchmark Fit Summary\n"
        "Monotonic suppression observed across d=3,5,7.\n\n"
        "Python\n"
        "#!/usr/bin/env python3\n"
        "from __future__ import annotations\n"
        "import math\n\n"
        "# ── 1. Dual-Substrate Mesh Probe ───────────────────────\n"
        "def probe_substrate() -> dict:\n"
        "    # Inline helper comment\n"
        "    return {'ok': True}\n\n"
        "# ── 2. MWPM Decoder ────────────────────────────────────\n"
        "class Decoder:\n"
        "    pass\n\n"
        "if __name__ == '__main__':\n"
        "    probe_substrate()\n\n"
        "Bash\n"
        "# Immediate run on WSL2:\n"
        "python3 script.py --shots 4096\n\n"
        "## 2. Physical Conclusions\n"
        "Space-time matching suppresses measurement flips.\n"
    )
    chunks = living.document_review_chunks(pasted)
    titles = [c["title"] for c in chunks]
    assert titles == ["## 1. Benchmark Fit Summary", "## 2. Physical Conclusions"]
    # Verify lossless preservation of all source bytes across chunks
    reconstructed = "".join("".join(c["parts"]) for c in chunks)
    assert reconstructed == pasted


def test_internal_review_skips_workspace_probes_and_continues_past_token_cap(tmp_path) -> None:
    rec_probes = _RecordingProbes()
    mentor = _SelectiveTokenCapMentor()
    out: List[str] = []
    o = living.Osiris(
        core=_StubCore(),
        mentor=mentor,
        home=str(tmp_path),
        out=out.append,
        tty=False,
        probes=rec_probes,
    )
    o.history.append({"role": "user", "content": "prior chat turn"})
    doc = (
        "## Section Alpha\n"
        + ("Alpha claim text mentioning git commits and training step.\n" * 65)
        + "\n## Section Beta\n"
        + ("Beta claim text mentioning 10.5281/zenodo.19813149.\n" * 65)
    )
    o.converse(doc, input_transport="bracketed_paste")
    rec_probes.calls.clear()

    res = o.review_last_paste()
    cov = res["coverage"]["review_coverage"]
    # Probes must NEVER execute on quoted untrusted review segments
    assert rec_probes.calls == []
    # Section 1 hit repeat/token truncation -> marked incomplete honestly, while Section 2 STILL ran!
    assert mentor.calls == 4
    assert res["status"] == "PARTIAL"
    assert cov["sections_read"] == ["2: ## Section Beta"]
    assert cov["sections_not_read"] == ["1: ## Section Alpha"]
    # Live chat history preserved (not overwritten by internal review chunks)
    assert any("prior chat turn" in m["content"] for m in o.history)


def test_review_section_range_and_resume_support(tmp_path) -> None:
    calls = []

    class _SimpleMentor:
        local_review_allowed = True

        def model(self):
            return "qwen2.5-coder:7b"

        def fast_model(self):
            return None

        def stream(self, messages, model=None):
            calls.append(messages[-1]["content"])
            yield "Reviewed section."

        last_done = {}

    o = living.Osiris(
        core=_StubCore(),
        mentor=_SimpleMentor(),
        home=str(tmp_path),
        out=lambda _s: None,
        tty=False,
    )
    doc = (
        "## Sec 1\n" + ("Text one.\n" * 350)
        + "\n## Sec 2\nText two.\n"
        + "\n## Sec 3\nText three.\n"
    )
    o.converse(doc, input_transport="bracketed_paste")

    # Review only section 1-2 first
    r1 = o.review_last_paste(section_spec="1-2")
    assert r1["status"] == "PARTIAL"
    assert r1["coverage"]["review_coverage"]["sections_read"] == ["1: ## Sec 1", "2: ## Sec 2"]
    first_call_count = len(calls)

    # Resume reviews only remaining section 3 and reaches COMPLETE
    r2 = o.review_last_paste(resume=True)
    assert r2["status"] == "COMPLETE"
    assert len(calls) == first_call_count + 1
    assert set(r2["coverage"]["review_coverage"]["sections_read"]) == {
        "1: ## Sec 1",
        "2: ## Sec 2",
        "3: ## Sec 3",
    }


def test_repl_review_command_accepts_range_and_resume(monkeypatch) -> None:
    recorded = []

    class _FakeLiving:
        def review_last_paste(self, section_spec=None, resume=False):
            recorded.append((section_spec, resume))

    monkeypatch.setattr(osiris_repl, "get_living", _FakeLiving)
    dispatch_command(OsirisReplState(), "/review 2-5")
    dispatch_command(OsirisReplState(), "/review --resume")
    assert recorded == [("2-5", False), (None, True)]


def test_probes_files_ignores_zenodo_dois_and_urls(tmp_path) -> None:
    p = probes.Probes(living_home=str(tmp_path), base=str(tmp_path))
    msg = (
        "See https://doi.org/10.5281/zenodo.19813149 and 10.5281/zenodo.24293756 "
        "and doi:10.5281/zenodo.19654953 for archived QEC-001 records."
    )
    assert p.files(msg) == []


def test_penteract_cosmology_claim_does_not_false_fire_on_koide_lepton_mass() -> None:
    koide_text = (
        "Zero-parameter prediction: m_tau = 1776.9690 MeV from K = 2/3 exactly "
        "with Brannen ratio r = sqrt(2)."
    )
    hit_ids = {c.id for c in claims.match(koide_text)}
    assert "PENTERACT_COSMOLOGY" not in hit_ids

    cosmo_text = "Zero-parameter predictions of omega_lambda, n_s and r = 0.00298 in cosmology."
    cosmo_ids = {c.id for c in claims.match(cosmo_text)}
    assert "PENTERACT_COSMOLOGY" in cosmo_ids
