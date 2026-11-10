# osiris.compiler default-circuit behavior: investigation (2026-11-09)

Base: `main` @ `9896f04a`. Branch `bugfix/compiler-fallback-trace`. Not merged; not on `feature/governed-trio`.

## Reproduction
`OrganismConverter().compile(x)` returns the default `H, CX, MEASURE` circuit for: a bare string
(`"this is not dna at all"`), `organism X { }`, an organism with a gene, and
`organism G { quantum_state { h 0; } }` / `{ x 0; y 1; z 2; swap 0, 1; }` / `{ totally_unsupported_op 99; }`.
Pinned by `tests/test_compiler_fallback_characterization.py` (12 tests, passing on unmodified code).

## Execution path (source-level)
1. `converter.py compile()` → `Lexer(source).tokenize()` → `Parser(tokens).parse()`.
2. `dna_parser.py Parser._parse_organism` reads `organism NAME`, then **advances past every token
   inside `{ ... }` without interpreting it**. `QuantumStateNode`, `QuantumOpNode`, `GenomeNode`,
   `GeneNode` are defined but never constructed. `organism.quantum_state` is always `None`.
3. `converter.py` op-building code (`if hasattr(organism, "quantum_state") and organism.quantum_state:`,
   including `raise ValueError("Unsupported DNA-Lang operation")`) is therefore **unreachable**;
   `ops` stays empty → `IRCompiler.compile(name, qubits=2, operations=None)` → `dna_ir.py:181` default
   `H(0), CX(0,1), MEASURE`.
4. Input with no `organism` keyword: if it contains none of `organism genome gene { } \n ;`
   (`_looks_like_dna_source`) it is treated as an organism *name* and compiled to the same default;
   otherwise `ValueError("... does not contain an organism declaration")`.
5. `lineage_hash` = sha256(`name:qubits:len(ops)`) — independent of genome content.
6. `CandidateEvolutionLoop.parse_and_validate` (`osiris_cli/dnalang/loop.py`) calls `compile()` and
   checks `ALLOWED_CAPABILITIES` against `circuit.operations` (steps 4–5 of its 12-step loop,
   "SYS-025"). Those operations are the default circuit, never the genome's. `acceptance_fn`
   receives that IR, so fitness did not depend on genome content either; `mutate_parent_genome`'s
   edits never reached evaluation. `PromotionPolicy` and the ledger then operate on those results.

## Classification
* **Bare-name input: intentional compatibility behavior.** Commented as a "documented convenience
  API"; pinned by `tests/test_backends.py` (`compile("BellState")`) and `tests/test_dnalang_loop.py`
  (`parse_and_validate("BellState")` is valid). Kept.
* **Skipped organism bodies: an unfinished stub, not a feature.** The parser was added as a minimal
  organism-name reader (commit `303e9167`, "comprehensive unit tests"); the converter code written
  to consume bodies shows the intended design. No doc describes bodies as ignored.
* **Consequence for the loop: a genuine validation defect** (vacuous capability allowlist and
  evaluation) in the promotion path. This is a correctness/governance defect, not shown to be an
  exploitable vulnerability: nothing here executes genome text, and the candidate digest still binds
  the genome source. Whether any historical lineage record was affected is not determined here;
  records in `~/.osiris/dnalang_lineage.jsonl` were not read or changed.

## Correction (commit `b4f49896`) — opt-in strict mode
`Parser(tokens, strict=True)`, `OrganismConverter.compile(..., strict=True)` and
`CandidateEvolutionLoop.parse_and_validate(..., strict=True)`. Strict accepts exactly
`organism NAME { quantum_state { OP INT[, INT]*; ... } }` (the form the loop emits), exactly one
organism, ≥1 operation, known operations, correct arity (`gate_map.QUBIT_COUNT`), distinct qubits in
0..63, nothing outside the organism. Everything else raises. `run_cycle` always uses strict; bare
names, empty organisms, bodies with genes, unknown ops, wrong arity are refused *before*
`acceptance_fn` runs. Default behavior everywhere else is unchanged (legacy tests + 12
characterization tests unchanged and passing).

### Remaining limitations
* `parse_and_validate()` **still defaults to legacy mode**, so a caller outside `run_cycle` keeps the
  old behavior (kept because an existing test pins bare names as valid). Flipping the default is a
  separate decision.
* Strict v0 has no gate parameters (`rz 1` carries no angle, as the loop emits it), no `genome`/`gene`
  bodies, and rejects the currency-card form (newline-separated, uppercase ops, `locus` lines).
* `osiris_launcher.py` (`converter.compile(source_text)`) still uses legacy mode: a user-supplied
  file's body is ignored there. Not changed.
* Legacy lexer supports `#` comments only; `//` comments containing the word "organism" raise
  `SyntaxError` (see dialect matrix).
