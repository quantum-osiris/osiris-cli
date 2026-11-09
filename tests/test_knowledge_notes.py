"""Approved paste-learning notes (notes.jsonl) must reach Knowledge.lookup, and so the
mentor's per-message context. Regression for note-9a9bd331 (2026-11-09): the note was
stored and authorized but never retrieved, because Knowledge only indexed markdown files."""

import json

from osiris_cli.knowledge import NOTE_PREFIX, Knowledge, active_notes


def _write_notes(living, records):
    living.mkdir(parents=True, exist_ok=True)
    with open(living / "notes.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write((r if isinstance(r, str) else json.dumps(r)) + "\n")


def _note(nid, status="active", content="claim-level evidence gate between propose and verify"):
    return {"id": nid, "title": "Evidence-Grounded Response Audit", "content": content,
            "status": status, "source_type": "bracketed_paste"}


def test_active_note_is_retrieved_with_label(tmp_path):
    living = tmp_path / "living"
    _write_notes(living, [_note("note-aaaa1111")])
    kb = Knowledge(living_home=str(living), base=str(tmp_path))
    hits = kb.lookup("what evidence gate did we decide on?")
    assert hits and hits[0]["source"] == NOTE_PREFIX + "note-aaaa1111"
    text = kb.format_notes(hits)
    assert "note-aaaa1111" in text and "unverified" in text


def test_tombstoned_and_malformed_notes_are_excluded(tmp_path):
    living = tmp_path / "living"
    _write_notes(living, [_note("note-dead0000", status="tombstoned"), "{not json",
                          {"id": "note-nocontent", "status": "active"}])
    assert active_notes(str(living)) == []
    assert Knowledge(living_home=str(living), base=str(tmp_path)).lookup("evidence gate") == []


def test_explicit_sources_do_not_pull_in_notes(tmp_path):
    living = tmp_path / "living"
    _write_notes(living, [_note("note-aaaa1111")])
    kb = Knowledge(living_home=str(living), base=str(tmp_path), sources=[])
    assert kb.lookup("evidence gate") == [] and kb.note_ids == []
    kb = Knowledge(living_home=str(living), base=str(tmp_path), sources=[], notes=True)
    assert kb.note_ids == ["note-aaaa1111"]


def test_notes_do_not_change_brief_or_on_record(tmp_path):
    living = tmp_path / "living"
    _write_notes(living, [_note("note-aaaa1111", content="m3b: FAIL")])
    with_notes = Knowledge(living_home=str(living), base=str(tmp_path))
    without = Knowledge(living_home=str(living), base=str(tmp_path), notes=False)
    assert with_notes.brief() == without.brief()
    assert with_notes.on_record("what did m3b find?") == []


def test_missing_notes_file_is_fine(tmp_path):
    kb = Knowledge(living_home=str(tmp_path / "nope"), base=str(tmp_path))
    assert kb.note_ids == []
