# RFC-0001 (DRAFT): `agent` genes in dna::}{::lang

Status: **draft for review. The language does not support agent genes today.** Phase 0 (OSIRIS-side adapter, `osiris_cli/orchestration/agent_gene.py`) is implemented and tested; Phase 1 is not started. Nothing in this
RFC is implemented; no grammar, parser, sema or runtime was modified. Evidence below was produced
by probing `dnalang-core` @ `3de9177` and `osiris.compiler` on 2026-11-09 (throwaway `.dna`
snippets, read-only); the probes that are stable are pinned in
`tests/test_dnalang_agent_gene_probe.py` (they characterize current behavior, they do not test a feature).

## 1. What exists today (verified)

1. **Grammar already admits arbitrary key-value genes.** `gene NAME { key: value ... }` parses
   (`dnalang/parser.py:142 kvgene`); lists and numbers parse as values. An agent-shaped gene
   (`role`, `capabilities`, `budget_tokens`) parsed and its fields were preserved on
   `KVGene.fields`. **No grammar change is needed for a data representation.**
2. **Sema accepts it with a warning only.** A gene with neither `condition` nor `trigger` gives
   `warning: ... (data only)`; zero errors. A made-up capability (`ROOT_EVERYTHING`) is accepted:
   the language has no notion of capabilities.
3. **Lowering ignores the fields.** `rules_ir.lower_all` returns no IR for a data-only gene, and
   adding or removing `role/capabilities/budget_tokens` on a rule gene leaves the RuleSet sha256
   unchanged. So today the canonical hash does not cover anything an agent declares.
   `docs/LANGUAGE.md` states the same ("Unknown fields are preserved ... and ignored by lowering").
4. **Two compilers, two dialects.** `dnalang-core` (`dnalang.parser`, circuit/rule/regulator genes)
   is not `osiris.compiler`. `osiris.compiler.OrganismConverter.compile` returned the *default*
   H/CX/MEASURE circuit for garbage, an empty organism, an agent gene, and for
   `organism G { quantum_state { h 0; } }` (the form `osiris_cli/dnalang/loop.py` mutates). I did
   not trace why; the practical consequence is that `CandidateEvolutionLoop`'s parse/validate step
   does not demonstrably constrain genome content on this converter. **Not touched here; flagged.**
5. The runtime for rule/regulator genes belongs to consumers (`organism_sim`); the language has
   no execution semantics of its own. An agent runtime would likewise live in OSIRIS.

## 2. Proposal

Keep the language a *declaration and identity* layer; keep orchestration, budgets enforcement and
promotion in OSIRIS (`AgentSpec`, `RunBudget`, `PromotionPolicy`, grants).

### Phase 0 — no language change (can ship in OSIRIS alone)
Represent an agent as a data-only kv gene with a discriminator:

```
organism ResearchTrio {
  gene Researcher {
    kind: "agent"
    role: "research"
    version: "1.0"
    capabilities: ["READ_ONLY_ANALYSIS", "RETURN_STRUCTURED_PROPOSAL"]
    deadline_s: 120
    max_tokens: 4096
    evidence_required: true
  }
}
```
A typed adapter `AgentSpec <-> dict` in `osiris_cli/orchestration/` that: (a) accepts only
`kind: "agent"` genes; (b) validates `capabilities` against `CapabilityGovernor.CAPABILITY_TIERS`
(refuses unknown names); (c) computes its **own** `sha256` over canonical JSON of the agent
fields, because the language hash ignores them (finding 3). Cost: none to the language.
Limitation: identity and validation live outside the language; `dnalang check` still says "data only".

### Phase 1 — language change (needs explicit approval; not started)
No parser change. Sema + lowering only, in `dnalang-core`:
* `sema.check_kv`: for `kind: "agent"` genes, require `role`, `version`, `capabilities`; error on
  unknown keys *for that kind only* (other kinds keep today's permissive behavior); error on a
  capability outside a **closed set exported by the consumer** (the set stays defined by
  OSIRIS, passed in as `check(..., capability_set=...)`, never hard-coded in the language).
* `rules_ir.lower_agents -> AgentSet` with canonical JSON + sha256, mirroring `RuleSet`
  (`lower_all` gains an `"agents"` key; existing keys and hashes unchanged).
* Warning `agent-no-evidence` when `evidence_required` is false.

### What the language will NOT hold
Fitness *judgement*, promotion, authorization, ledger writes. The grammar's `fitness IDENT;`
clause may *name* a metric; the metric is computed and compared by OSIRIS (`frozen_suite_pass_rate`
today), so an organism cannot declare itself fit.

### Mutation / selection
Agent genes are mutated only through a closed edit set (as `bridge/edits.py` does for DD
genes): e.g. adjust a numeric budget within bounds, add/remove a capability **from a lower or
equal tier** (never raise tier). A mutated agent gene is a *candidate*: it goes through
`run_trio` + `evaluate_candidate` like a code patch (frozen baseline, firewall, PromotionPolicy,
signed grant bound to its digest). Capability *raising* always requires a human grant.

## 3. Compatibility
Phase 0: none. Phase 1: existing organisms are unaffected (no existing gene has `kind: "agent"`;
`lower_all` output for current inputs is byte-identical — to be asserted by a golden test before
merging). Dialect question (finding 4) must be settled first: which compiler is authoritative for
`.dna` consumed by OSIRIS?

## 4. Tests required before any claim of support
Parser: agent gene round-trips. Sema: missing role/version/capabilities errors; unknown capability
errors; non-agent kv genes unchanged. Lowering: `AgentSet` canonical JSON stable, hash changes when
capabilities/budget change (inverse of today's finding 3), hash of every pre-existing fixture
unchanged. Runtime (OSIRIS side): adapter refuses over-tier capabilities; an `AgentSpec` built from
a gene equals the hand-built one used by `trio.py`. Negative: an agent gene cannot grant itself
`WORKSPACE_WRITE` without a grant.

## 5. Open questions
1. Which compiler/dialect is authoritative for OSIRIS (finding 4)?
2. Should `version` be semantic or a content hash?
3. Is a per-agent token/cost budget a language concern or only an OSIRIS run policy?
