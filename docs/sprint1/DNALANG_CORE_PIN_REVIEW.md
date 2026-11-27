# dnalang-core pin review

Reviewed 2026-11-27. Sprint 1 keeps the pin.

| | |
| --- | --- |
| Installed pin | `5ae0748918bec9c63628fed3b0260cdc9b4b4553` |
| Package version at that pin | `0.4.0` (`dnalang/__init__.py`) |
| Compared main | `90b68053959f4e31627e746facc5dbde68c32e14` |
| Range | 13 commits, 0 behind, 32 files, `ledger.py` unchanged |

`pyproject.toml` and `.github/workflows/dnalang-pytest.yml` still install that pin. This review does not bump it.

## What landed after the pin

The compare is additive in the ledger and breaking in the compiler:

- `dnalang/qc_ir.py` (+274/−28), `dnalang/sema.py` (+265/−44), `dnalang/parser.py`, and `dnalang/lower.py` change the circuit IR and checks.
- `dnalang/timing.py` adds a compiler timing contract (`exact` vs `quantize`). `Circuit.to_qiskit` is now called as `to_qiskit(dt, timing_policy="quantize")` from `dnalang/backends/ibm.py`. `qasm` CLI flags changed (`--dt` is no longer `type=float` only; `--timing-policy` and `--require-dt` were added).
- `dnalang/__init__.py` imports `timing` and `seq_ops` at import time. `seq_ops.py` is a new 781-line module.
- New surfaces: `dnalang/qkernel/` (genotype benchmark, statevector, SVM), `dnalang/passes/wen_plaquette_pass.py`, and tests for those. Later commits lazy-load qkernel exports; the package init still imports `seq_ops` eagerly.
- `dnalang/ledger.py` is not in the diff, so the hash-chain writer this repo already uses is the same bytes at both commits.

## Conclusion

Do not bump the pin in Sprint 1, and do not open a pin-bump PR yet. The ledger format is stable, but the compiler, QASM export, and IBM transpile call are not a drop-in change. A bump needs its own PR that runs the osiris-cli suite against that commit and updates every `to_qiskit` / `to_qasm3` caller. That work is Sprint 2, not this branch.
