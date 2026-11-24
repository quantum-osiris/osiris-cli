# DNA::}AI{::Lang 0.1 — language reference (dna-ir/0.1)

> Code-derived reference, written 2026-11-24 from the authoritative
> implementation `dna_lang.py` (repo root, `main` @ `5549d0c9`), whose docstring
> cites this file as `bin/DNA_LANG_SPEC.md` (that `bin/` prefix is stale — the
> compiler lives at the repo root; left unedited here).
> Executable contract: `tests/test_dna_ir_conformance.py` (14 tests, pinned
> `SPEC_HASH` and content digests). Cross-layer map:
> `osiris-assessment/proposals/DNA_IR_CORE_SPEC_2026-11-24.md` and
> `docs/dialects/DIALECT_MATRIX.md` (dialect E).
>
> This is a **document** language (organism definitions and metamorphosis
> proposals). It is not the dnalang-core genome/circuit grammar (dialect B),
> despite the shared `organism NAME {` opening.

## 1. Scope and non-goals (0.1)

Parse, check, canonicalize, hash. A compiled DNA-IR document's bytes **are**
the canonical bytes, so `sha256sum organism.dna.json` equals the content ID.

0.1 deliberately does **not**: execute anything, sign anything, touch the
network, resolve `dna:sha256:` references against an artifact store
(format-checked only), support imports, or grant authority. A document only
declares requests and requirements; trusted OSIRIS code decides what happens
to it. Stdlib only (runs on bare Termux).

CLI: `python3 dna_lang.py validate|compile|hash|inspect FILE [--json]`
(FILE is `.dna` source, or compiled DNA-IR if `.json`). Exit codes: 0 valid,
1 invalid (diagnostics), 2 usage/IO error.

## 2. Lexical rules

- UTF-8 source, LF line endings (BOM stripped; bare CR rejected). `//`
  comments. Strings `"…"` with escapes `\" \\ \n \t \/ \uXXXX`, or `"""…"""`
  multi-line (dedented, trailing whitespace stripped).
- Integers only: `-?(0|[1-9][0-9]*)`; fractions/exponents/leading zeros are
  `DNA-E-TYPE`. `|int| ≤ 2^53−1` (largest integer every JCS/IEEE-754 consumer
  agrees on), else `DNA-E-LIMIT`.
- Identifiers: `[A-Za-z_][A-Za-z0-9_.\-]*`, ≤ 128 chars.
- Text rules (`DNA-E-TEXT`): ≤ 8192 chars, Unicode NFC, no bidi controls
  (CVE-2021-42574 Trojan Source), no C0/DEL controls (newline allowed only in
  `"""` strings), no unpaired surrogates.
- Limits: source ≤ 256 KiB; IR ≤ 1 MiB; list ≤ 256 items, no nested lists;
  nesting depth ≤ 6; ≤ 64 named blocks per collection.

## 3. Grammar (EBNF)

```
document := "organism" NAME "{" entry* "}"
entry    := IDENT [NAME] "{" entry* "}"        // block (named if NAME present)
          | IDENT value                        // field: `key value` — no colon
value    := STRING | INT | "true" | "false" | "[" [scalar ("," scalar)* [","]] "]"
```

`null` is not a value (`DNA-E-TYPE`); omit optional fields instead.

## 4. Document schema (the pinned SPEC; `SPEC_HASH` guards it)

Envelope (all required; any other top-level key → `DNA-E-UNKNOWN`, reserved
words → `DNA-E-AUTHORITY` / `DNA-E-EXEC` / `DNA-E-IMPORT`):

| key | rule |
|---|---|
| `schema_version` | const `"dna-ir/0.1"` (else `DNA-E-VERSION`) |
| `kind` | const `"organism_definition"` |
| `name` | name |
| `spec_hash` | const `sha256:90be4b47595b1528139041e344e7ae00ab2991e63acf1893874fbeb41a1a7a34` |

Blocks:

- `meta` — `schema` const `"dna-lang/0.1"`, `title` str, `description` opt str.
- `identity` — `organism_id` URN (`urn:<nid>:<nss>`), `species` name,
  `generation` int [0, 2^53−1], `genome_ref` / `parent_genome_ref` opt ref.
- `intent` — `objective` str, `success` opt **SEQ** of str.
- `constraints` — `require`, `prohibit`: **SET** of name; `allowed_roots`: SET
  of path (min 1); `protected_paths`: opt SET of path. Path: workspace-relative
  POSIX, no leading `/` or `~`, no `\` or `:`, no empty/`.`/`..` segments,
  1–256 chars.
- `capability NAME {…}` — `class` enum {`read_only`, `proposal`, `execution`,
  `append_only`, `controlled_write`}; `authority` enum {`model_adapter`,
  `trusted_runtime`, `sandbox_only`, `operator_plus_runtime`}; opt
  `description`, `profile`; opt block `limits { wall_seconds [1,86400],
  memory_mb [1,65536], network "deny", native_execution "deny" }`.
- `evidence NAME {…}` — `kind` enum {`verified_observation`, `hypothesis`,
  `failure_pattern`, `operator_preference`, `untrusted_reference`},
  `statement` str, `refs` SET of ref.
- `metamorphosis NAME {…}` — `parent` ref, `target_generation` int [1, 2^53−1],
  `objective` str, opt `rationale`; blocks `mutation { kind enum{source_patch,
  prompt_strategy, config}, patch_ref ref, allowed_paths SET of path min 1 }`,
  `verification { required_profiles SET of name min 1, reject_on opt SET }`,
  `promotion { require_operator_approval, require_sandbox_pass,
  require_base_hash_match, require_ledger_attestation }` — each `true`-only.

Field kinds: `ref` = `dna:sha256:[0-9a-f]{64}`; `urn`, `name`, `path`, `str`,
`int` with per-field min/max, `enum`, `const`, `true`, `SET(item)`
(unordered, duplicate-free, canonical = sorted by UTF-16BE code units),
`SEQ(item)` (ordered; source order is part of the content ID).

## 5. Canonical serialization and content ID

RFC 8785 (JCS) over the value domain {object, array, string, integer, bool} —
**floats are outside the domain and refused**, not approximated. Object keys
sorted by UTF-16 code units. `canonical_bytes(ir)` is UTF-8.
`content_id = "dna:sha256:" + sha256(canonical_bytes)`. The hash boundary is
the canonical serialization — not the source text, not the raw file bytes.
A valid but non-canonical JSON file has the same content ID but a different
file sha256; `load_any` reports this (`is_canonical=False`).

## 6. Validation layers and diagnostics

1. **Structural** (lex/parse/lower): syntax, block shapes, reserved words,
   duplicate field/block (`DNA-E-DUP`).
2. **Type/range** (`validate_ir`, also applied to compiled `.json`):
   field types, bounds, enums, ref/urn/name/path formats, canonical set order
   (`DNA-E-CANONICAL`), collection limits. Compiled JSON is parsed with
   duplicate-key rejection (`DNA-E-DUP`) and float rejection (`DNA-E-TYPE`).
3. **Semantic** (`_semantics`, only if 1–2 are clean): lineage
   (`parent_genome_ref` without `genome_ref`, self-parent, metamorphosis
   `parent` must equal `genome_ref`, `target_generation = generation + 1`),
   `require ∩ prohibit = ∅`, allowed-root vs protected-path scope, mutation
   paths inside allowed roots and outside protected paths, `execution`
   capability requires `limits` and non-`model_adapter` authority,
   `controlled_write` requires `operator_plus_runtime`,
   `verified_observation` requires non-empty `refs`.

Diagnostic rules are stable IDs (e.g. `DNA-E-VERSION`, `DNA-E-UNKNOWN`,
`DNA-E-DUP`, `DNA-E-TYPE`, `DNA-E-RANGE`, `DNA-E-REF`, `DNA-E-AUTHORITY`,
`DNA-E-WEAKEN`, `DNA-E-CANONICAL`, `DNA-E-MISSING`, `DNA-E-LIMIT`).

## 7. Trust boundary

Declarations are not execution. A declared capability is a request, not a
grant; an evidence label is not verification; a promotion requirement is not a
completed promotion; `promotion.*` gates cannot be switched off from inside a
document (`DNA-E-WEAKEN`). External authorization (grants, receipts, ledger
attestation) is enforced by OSIRIS governance outside this format.

## 8. Replay semantics

Replaying a document = loading its compiled JSON and re-deriving canonical
bytes. Exact artifact identity = content ID over canonical JCS bytes (no
timestamp/audit exclusions exist in dna-ir/0.1). Numeric outcome replay does
not exist at this layer (no floats); experiment-level tolerances must be
specified per experiment with units, scale and rounding, and are never
byte-identical evidence. Run-level records (`osiris-run/1`, dnalang-core
ledger) use a different canonicalization (sorted-key compact JSON, timestamps
inside the hash) — see the core spec §3.
