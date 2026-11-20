---
name: osiris-claims-grounding-audit
description: Audit responses, research briefs, and training candidates against the SYS-024 claims register, file-citation provenance, histogram-backed metrics, and SYS-026 Speaking Gate rules. Use when evaluating physics or quantum claims, reviewing mentor outputs, checking citations, or filtering exchanges for NCLM continual learning.
---

# OSIRIS Claims and Grounding Audit

Enforce OSIRIS epistemic governance across prompts, code comments, experiment briefs, and mentor replies so unverified claims or hallucinated metrics never enter durable memory or `osiris.nclm` training.

## When to Use

- Reviewing quantum, physics, or benchmark claims before writing docs, briefs, or code
- Auditing mentor model outputs for phantom file citations or fabricated precision numbers
- Updating or testing `osiris_cli/claims.py`, `osiris_cli/knowledge.py`, `osiris_cli/living.py`, or `osiris_cli/train.py`
- Deciding whether a recorded exchange qualifies for online continual learning or batch distillation

## Steps

1. **Enforce the Pre-Model SYS-024 Claims Register:**
   - Cross-check named constants and experimental assertions against `osiris_cli/claims.py` (`/legit list`) before generating conclusions.
   - Never treat refuted or unmeasured historical CRSM parameters as active invariants (`theta_lock = 51.843 deg` is `REFUTED`, `F_max = 0.97871` is `REFUTED`, `Lambda_Phi = 2.176435e-08` is `NOT_MEASURED`).

2. **Verify Explicit File Attribution:**
   - Every local repository or note citation in a synthesized answer must match a path actually present in that turn's retrieved context or deterministic `ProbeReceipt` list.
   - Flag any cited path absent from the retrieved bundle as a phantom citation.

3. **Enforce the Histogram Procedural Rule:**
   - Apply the governing rule: *"No number gets a name until a histogram produced it, and no AI-written summary substitutes for opening the counts."*
   - Require every precision metric (`bits/byte`, `bpb`, `nDCG@10`, `fidelity`, `p-value`, `z-score`, shot counts, or HQC estimates) to trace verbatim to a raw ledger row, benchmark artifact, or deterministic computation receipt with explicit units and uncertainty bounds.

4. **Protect SYS-026 Speaking Gate and Held-Out Isolation:**
   - Keep every 5th exchange (`exchange_idx % 5 == 0`) strictly isolated as held-out evaluation data (`heldout = True`, never trained on).
   - Mark `distill_eligible = False` whenever a reply comes from a degraded low-RAM fallback model, truncates mid-stream, or fails citation or numerical-metric verification.
   - Keep `osiris.nclm` behind the `SYS-026` Speaking Gate until held-out evaluation reaches `<= 2.0 bits/byte` across `30/30` held-out exchanges while beating the unigram baseline.

## Gotchas

- **Confusing retrieval ranking with truth:** `olmo3-embed-7b` and BM25F rank candidate passages by relevance (`nDCG@10`), not truth. Only deterministic code (`claims.py`, `probes.py`, verified ledger counts) issues a verdict.
- **Triggering SYS-024 accidentally in slash commands:** In the REPL, slash commands must begin at index 0 of a clean line; concatenating a slash command onto natural-language text containing registered claim numbers routes the whole buffer through chat and triggers pre-model claim injection.
