# OSIRIS-XTOP-005: [[4,2,2]] building blocks and budget

Status: **design and tooling only. Nothing here has run on hardware, and nothing is preregistered.**

## What exists

| File | Contents |
| --- | --- |
| `code422.py` | [[4,2,2]] builder. One gate list emits a native Qiskit `QuantumCircuit` and a native pytket `Circuit`. It provides `|00>_L` and `|++>_L` preparation with optional verification, transversal logical CNOT, logical Bell pairs across two blocks, flagged weight-4 X/Z checks, readout for all 9 logical tomography settings, and a per-shot decoder. |
| `../tests/test_xtop005_code422.py` | 28 exact tests, all passing. See the list below. |
| `budget.py` → `budget.json` | IBM seconds and Quantinuum HQC for the 4-arm × 9-basis × 3-depth matrix, computed from compiled circuits. |

The tests cover:
- the logical algebra;
- preparations;
- all 9 Bell correlators decoded shot by shot;
- the transversal CNOT's action on logical Paulis;
- detection of every single-qubit X/Y/Z error;
- the verified preparation catching the CNOT fault that would otherwise leave an undetectable logical error;
- identical unitaries from the Qiskit and pytket emitters.

## Findings that change the proposed design

1. **The final readout detects errors in only 2 of the 9 settings.** An exhaustive search over single-qubit bases shows that all 9 logical settings are readable. Only the XX and ZZ readouts also contain a stabilizer. The other 7 settings get detection only from mid-circuit checks, so the checks are required and not optional.
2. **The scrambler cannot mix encoded and bare qubits.** The proposal encodes M but leaves B and E bare. A scrambler acting on M and B together leaves the code space, so it is not error-detected. B must be encoded as well, or the arm is not code-space HP.
3. **No Clifford+T scrambler.** T is not transversal on [[4,2,2]]. Fault-detected logical gates here are:
   - transversal CNOT between blocks;
   - physical permutations within a block (logical SWAP and CNOT);
   - H⊗4 (logical H⊗H with SWAP);
   - transversal phase gates (logical CZ-type).

   The scrambler and decoder must come from this group. Designing them, and checking that such a scrambler actually scrambles, is the open task. The bare HP-004 scrambler is a general 4-qubit Clifford and is not in this group as written.
4. **Gate counts.** The encoded layer alone costs 38 two-qubit gates, 18 measurements and 10 resets per circuit. That is more than the entire bare HP-004 protocol, which uses 32 CZ on IBM and 18 ZZ after H2 compilation.

## Budget (from `budget.json`)

Both totals are for 108 circuits: 4 arms × 9 bases × 3 depths.

| Resource | Proposal as written | Half-size variant |
| --- | --- | --- |
| IBM seconds | 4,096 shots: 136 s per backend, 271 s for two backends | 2,048 shots: 69 s per backend, 139 s for two |
| Quantinuum HQC (lower bound) | 1,000 shots: 13,187 | 500 shots: 6,863 |

- **IBM rate:** estimated at 0.30 ms per shot plus 3 s per job. This is this account's measured rate (MAP-04: 0.29 ms per shot; QEC-001 dynamic circuits: 0.28 ms per shot), not an IBM formula.
- **HQC formula:** `5 + C(N1q + 10 N2q + 5 Nm)/5000` per circuit, on circuits compiled offline for H2-1.
- **Why HQC is a lower bound:** the code-space scrambler and decoder don't exist yet, so the encoded arms use bare HP-004 scrambler+decoder counts as a stand-in.
- **Against the proposal's own figures:** it estimated about 1,850 HQC and about 140 IBM seconds. The computed values are about 7× and 2× higher.
- **Credentials:** no Quantinuum credentials are configured on this machine, so nothing can run on H2 or its emulator from here.
