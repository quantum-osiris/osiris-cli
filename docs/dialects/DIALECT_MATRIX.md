# `.dna` dialect compatibility matrix (observed 2026-11-09; column E added 2026-11-24)

No dialect is declared authoritative here. This records what each active component accepts today.
Executable: `tests/test_dialect_matrix.py`; fixtures in `tests/fixtures/dialects/`.

## Implementations and consumers
| id | implementation | entry point | active consumers |
|---|---|---|---|
| A | `osiris.compiler` legacy | `osiris/compiler/converter.py` `OrganismConverter.compile` / `dna_parser.py` | `osiris_launcher.py:917` (translate), tests/backends, bare-name API |
| A-strict | same, `strict=True` (branch `bugfix/compiler-fallback-trace`) | `compile(strict=True)` | `osiris_cli/dnalang/loop.py` `run_cycle` (promotion path) |
| B | `dnalang-core` (`dnalang` 0.4.0) | `dnalang.parser.parse`, `sema.check`, `rules_ir.lower_all` | `organism_sim`, `bridge`, `nclm_aer_evaluator.py`, `osiris/quantum/bidirectional_dd.py` (`dnalang.qc_ir`) |
| C | copilot-sdk file compiler | `copilot-sdk-dnalang/src/dnalang_sdk/organisms_compiler.py` `DNACompiler.parse_organism(path)` | SDK/physics tools; accepts INI (`[metadata]`) or block (`ORGANISM`) files |
| D | currency-card generator | `osiris_cli/dnalang/currency_card_gene.py` | emits `quantum_state { H 0 \n CNOT 0 1 }` + `locus 0x..: fn(...)`; validated by its own 6-gate validator, no parser accepts it fully |
| E | DNA::}AI{::Lang 0.1 document compiler (`dna-ir/0.1`) | root-level `dna_lang.py` `compile_source` / `load_ir` / `validate_ir` | `research.py`, `protege.py`, `osiris_termux_console.py` (export), `osiris/lab/autopoietic_loop.py`; language reference `docs/dnalang/DNA_LANG_SPEC.md`; conformance `tests/test_dna_ir_conformance.py` |

## Matrix (✓ accepted and meaningful · ⚠ accepted but body ignored/default · ✗ rejected)
| fixture | A legacy | A-strict | B dnalang-core | C sdk file | E dna-ir/0.1 |
|---|---|---|---|---|---|
| `bare_name.txt` (`BellState`) | ⚠ default circuit (name API) | ✗ SyntaxError | ✗ ParseError | ✗ | ✗ DNA-E-SYNTAX |
| `garbage.txt` | ⚠ default circuit (treated as a name) | ✗ | ✗ | ✗ | ✗ DNA-E-SYNTAX |
| `empty_organism.dna` | ⚠ default circuit | ✗ | ✗ | ✗ | ✗ DNA-E-MISSING |
| `osiris_loop_form.dna` `quantum_state { h 0; cx 0, 1; }` | ⚠ default circuit (body skipped) | ✓ H,CX,MEASURE from the body | ✗ ParseError | ✗ | ✗ DNA-E-SYNTAX |
| `currency_card_form.dna` | ⚠ default circuit | ✗ | ✗ LexError | ✗ | ✗ DNA-E-SYNTAX |
| `dnalang_core_rule_gene.dna` | ⚠ default circuit | ✗ | ✓ 0 errors | ✗ | ✗ DNA-E-SYNTAX |
| `dnalang_core_bell.dna` (`//` comment, circuit gene) | ✗ SyntaxError (legacy has `#` comments only) | ✗ | ✓ 0 errors | ✗ | ✗ DNA-E-SYNTAX |
| `repo_organism_scimitar_sse_128.dna` (tracked `organism/*.dna`) | ✗ SyntaxError | ✗ | ✗ LexError | ✓ OrganismProfile | ✗ DNA-E-SYNTAX |
| `dnalang_ir_organism.dna` (DNA::}AI{::Lang 0.1 document) | ⚠ default circuit (body skipped) | ✗ SyntaxError | ✗ ParseError | ✗ | ✓ compiles to `dna:sha256:f81d76b1…` |

## Observations
1. At least **five** incompatible input dialects are live. Only B (genome/circuit grammar) and E
   (document grammar) have a real grammar, semantic checks and canonical content hashes; they are
   separate languages that happen to share the `organism NAME {` opening and the `.dna` extension.
   A legacy "accepts" most things because it ignores bodies (⚠ ≠ understood).
2. Comment syntax differs (`#` in A vs `//` and `/* */` in B and E), so the same file can be valid
   in one and a SyntaxError in the other.
3. The loop (A-strict) and dnalang-core (B) do not share any genome syntax; `.dna` produced by the
   loop is not parseable by B. RFC-0001's agent genes are B-shaped; the loop's candidates are A-shaped.
   E is the declarative document layer (organism definitions and change proposals → `dna-ir/0.1`);
   it references B-layer artifacts only by `dna:sha256:` hash and never parses them.
4. Historical implementations are untouched; nothing was removed or renamed.

## Proposed strategy (no silent replacement)
1. Keep all five; label each file's dialect explicitly (first-line marker comment `// dialect: ...` is
   not available in A, so detect by content/extension map in an adapter, not in the languages).
2. Add a read-only `detect_dialect(text) -> {A|B|C|D|E|unknown}` in OSIRIS with tests from these
   fixtures; refuse `unknown` on every governed path.
3. Governed paths (promotion, agent genes) accept B only, or A-strict for the existing loop, via typed
   adapters; never legacy A.
4. Decide separately: (a) default `strict` for `parse_and_validate`; (b) migrating the loop's genomes
   to B; (c) retiring A's name/skip-body behavior. Each needs review.
