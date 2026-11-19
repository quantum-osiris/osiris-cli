#!/usr/bin/env python3
"""
prompt_strategies.py -- versioned prompt strategies for benchmark trials.

A strategy is a fixed preamble placed before a bench task's prompt, to test a
hypothesis about HOW to ask a mentor (e.g. "listing invariants first reduces
false confidence"). It changes only what the model is asked: the task suite,
the hidden tests and the runner (osiris_bench.py and the files runner_sha256
covers) are untouched, so a trial round stays comparable with a baseline round
of the same model -- same suite, same runner, one declared difference, whose
text is hashed (strategy_sha256) into the round's start entry.

A strategy is text in this file, reviewed like code: never model output, never
set from a paste. Changing a strategy's text changes its hash; give it a new
id (…-v2) rather than editing one that has results.
"""
import hashlib

STRATEGIES = {
    "invariant-first-v1": {
        "about": "list the behaviour the code must keep (repeats, bad input, sources) before writing it, "
                 "and make the test assert each of those",
        "text": (
            "BEFORE YOU WRITE ANY CODE, work out -- silently, without printing it -- the invariants this module "
            "must keep:\n"
            "1. What must stay true when the same operation is done twice (idempotency, duplicates).\n"
            "2. How invalid, empty and boundary inputs must be handled.\n"
            "3. Which values must carry or report their source, and what must never be invented.\n"
            "4. What an easy test would miss.\n"
            "Then write the module so it keeps every one of them, and write the test so it asserts every one "
            "of them -- not only the easy case. Output exactly the format required below, nothing else.\n\n"
        ),
    },
    "contract-critic-v1": {
        "about": "preserve existing AST attribute/return contracts on modify tasks, remove fixed-length regex "
                 "truncation caps, and test duplicate calls + >250-char boundary inputs",
        "text": (
            "BEFORE WRITING CODE, silently verify these 5 production contracts:\n"
            "1. API & Container Preservation: If modifying existing code, keep every public symbol, parameter "
            "name, and `__init__` attribute type unchanged (e.g. if `self.stories` is a `list`, keep it a `list` "
            "-- NEVER replace it with a `dict`).\n"
            "2. Non-None Idempotent Returns: If an existing method returns an ID/value (e.g. `return story['id']`), "
            "duplicate/repeated calls MUST return the existing item's ID/value -- NEVER `return None`.\n"
            "3. Zero Arbitrary Length Caps: Remove fixed regex ceilings like `.{0,160}` so lines > 250 chars "
            "match cleanly up to newline or end-of-string.\n"
            "4. Embedded Token & Failure Boundaries: Match decimals/identifiers before punctuation (e.g. `0.97.`), "
            "and keep permanent auth failures sticky until explicit reset.\n"
            "5. Adversarial Test Suite: In `test_proposal()`, assert normal inputs, duplicate calls (checking exact "
            "return value and container `isinstance`), empty/invalid inputs, and a >250-character boundary input.\n"
            "Output ONLY the exact format requested below, with no markdown fences inside JSON strings.\n\n"
        ),
    },
    "contract-critic-v2": {
        "about": "compact (<80 lines) JSON-safe modules, whole-token number grounding reported by value, "
                 "idempotent existing-ID return + list preservation, and decimal-safe sentence extraction",
        "text": (
            "CRITICAL OUTPUT & CONTRACT RULES (keep both `module` and `test` concise, under 70 lines each, "
            "with zero markdown fences inside JSON strings so the JSON object closes cleanly):\n"
            "1. Modify Preservation: Keep every existing class/function signature and `__init__` container type "
            "(if `self.stories` is a `list`, keep it a `list` of dicts -- NEVER convert it to a `dict`).\n"
            "2. Duplicate / Idempotent Return: When avoiding duplicate open items, check existing open items "
            "and `return existing['id']` (NEVER `return None`, and do not increment `next_id` on duplicates).\n"
            "3. Unsourced Number Grounding: When checking claims against evidence, extract standalone numbers "
            "with `re.findall(r'(?<!\\w)\\d+(?:\\.\\d+)?(?!\\w)', ...)` so `3` does NOT match `13` and `0.9722.` "
            "matches `0.9722`; for each number in the claim absent from the evidence set, append a problem string "
            "containing that exact number value (e.g. `f'unsourced number: {num}'`), while keeping existing checks.\n"
            "4. Unbounded Sentence Extraction: Never use fixed regex caps like `.{0,160}`; match up to `\\n` and "
            "end at a sentence period `\\.(?!\\d)(?:\\s|$)` so >250-char descriptions stay whole and `0.97` is not split.\n"
            "5. Compact Adversarial `test_proposal()`: Write 6-10 focused `assert` lines testing duplicates, "
            "unsourced vs sourced numbers (`3` vs `13`), and a >220-char string.\n\n"
        ),
    },
}


def get(sid):
    """The strategy dict, or KeyError naming the known ids."""
    if sid not in STRATEGIES:
        raise KeyError(f"unknown strategy {sid!r}; known: {', '.join(sorted(STRATEGIES))}")
    return STRATEGIES[sid]


def sha256(sid):
    return hashlib.sha256(get(sid)["text"].encode("utf-8")).hexdigest()


def apply(sid, prompt):
    """The task prompt with the strategy's preamble in front of it."""
    return get(sid)["text"] + prompt
