# qByte Classical Statevector Simulator — Correctness & Verification Report

**Status:** `Implemented and tested`  
**Phase 2 Section:** Section 3 (`REPAIR qBYTE`)  
**Branch / Worktree:** `feat/phase2-scientific-repair` (`/home/osiris/osiris-assessment/work/osiris-cli-phase2`)  
**Physical Classification:** Classical CPU statevector simulator (pure NumPy). **Not** an independent physical quantum substrate or error-corrected quantum processor.

---

## 1. Defects Diagnosed & Repaired

Prior to executing any code in `qbyte_system/`, static line-by-line inspection across all 8 files in `qbyte_system/` identified **five distinct mathematical/software defects**, all repaired in this changeset:

1. **Toffoli (CCNOT) Double-Swap Identity Defect (`qbyte_system/sovereign_executor.py`):**
   - *Before:* `_apply_toffoli(c1, c2, target)` iterated `for i in range(self.dim):` and swapped `(self._state[i], self._state[j])` with `j = i ^ (1 << target)` whenever both controls were `1`. Because both `i` and `j` satisfy the control mask, the loop visited `(i, j)` once when `i < j` and a second time when `i > j`, swapping amplitudes back and reducing CCNOT to the **identity operator $I$**.
   - *Fix:* Replaced bit-loop mutation with unified `_apply_k_qubit_gate(toffoli(), (c1, c2, target))` tensor-axis contraction with strict distinct-wire validation (`ValueError` on duplicate or out-of-range wires). Also added `Qbyte.toffoli` / `Qbyte.ccnot` and `Qbyte.fredkin` / `Qbyte.cswap`.
2. **Inconsistent Qubit Indexing Between Tensor-Axis Operations and Bit Masks (`qbyte.py` & `sovereign_executor.py`):**
   - *Before:* 1Q and 2Q gates reshaped `self._state` to `[2] * n_qubits` in C-contiguous order (big-endian $|q_0 q_1 \dots q_{n-1}\rangle$, where wire `q` maps to bit shift `n_qubits - 1 - q`), whereas `Qbyte.measure_qubit` (`(i >> qubit) & 1`) and `SovereignExecutor._apply_toffoli` (`(i >> c1) & 1`, `1 << target`) used little-endian bit shifts (`qubit` instead of `n_qubits - 1 - qubit`). Consequently, `qb = Qbyte(); qb.cleave(0); qb.measure_qubit(0)` deterministically returned `0` instead of `1`.
   - *Fix:* Enforced a single documented **big-endian computational basis convention** across all 1Q, 2Q, and 3Q gates and `Qbyte.measure_qubit` (`bit_shift = self.n_qubits - 1 - qubit`), and added explicit `.get_statevector(endian="big" | "little")` helpers on `Qbyte`, `SovereignExecutor`, and `ExecutionResult` for exact cross-framework comparison with Qiskit's little-endian `Statevector`.
3. **Non-Unitary Default Mid-Circuit Phase Conjugation (`sovereign_executor.py`):**
   - *Before:* `SovereignExecutor.__init__` defaulted to `enable_healing=True`, which computed computational-basis Rényi-2 inverse participation ratio $\Lambda = \sqrt{\sum_k p_k^2}$ after every gate and triggered `np.conj(state)` + amplitude damping whenever $\Lambda < 0.7$ or $1 - \Lambda > 0.3$ (which occurs on *any* uniform superposition of $\ge 2$ qubits), destroying relative quantum phases ($S, T, R_Z, R_X$, QFT) and violating unitarity.
   - *Fix:* Disabled heuristic non-unitary healing by default (`enable_healing=False`) so unitary circuit simulation is strictly unitary unless a caller explicitly opts into heuristic conjugation.
4. **Register Width Mismatch in `SovereignExecutor.run(circuit)` (`sovereign_executor.py`):**
   - *Before:* Running a $k$-qubit `Circuit(k)` on `SovereignExecutor(n_qubits=8)` with $k < 8$ left `self.dim = 256` and formatted sampled indices above $2^k - 1$ with `format(s, f'0{k}b')`, emitting mixed-length bitstrings (e.g., `'00'` alongside `'11000000'` for a 2-qubit Bell circuit).
   - *Fix:* `SovereignExecutor.run(circuit)` sizes the active statevector register to `circuit.n_qubits` (checking `circuit.n_qubits <= self.n_qubits`).
5. **Missing Gate Dispatches (`qft_circuit` `'cphase'`, `QbyteRuntime` `'phase_flip'`, `S†`/`T†`):**
   - *Fix:* Added `s_dg` (`Sdg`), `t_dg` (`Tdg`) in `gates.py`, `Qbyte.phase_flip`, and wired `'cphase'`/`'cp'`, `'fredkin'`/`'cswap'`, `'rxx'`, `'ryy'`, `'rzz'`, `'iswap'`, `'y'`, `'sdg'`, `'tdg'` in `Circuit`, `SovereignExecutor`, and `QbyteRuntime`.

---

## 2. Basis-Order Convention Specification

All `qbyte_system` statevectors use **Big-Endian** wire ordering:
$$|b_0 b_1 \dots b_{n-1}\rangle \longleftrightarrow \text{index } i = \sum_{q=0}^{n-1} b_q \, 2^{\,n - 1 - q}$$
- Wire $q = 0$ is the most-significant bit (tensor axis $0$, bit mask $1 \ll (n - 1)$).
- Wire $q = n - 1$ is the least-significant bit (tensor axis $n - 1$, bit mask $1$).
- `get_statevector(endian="little")` transposes tensor axes $(0, \dots, n-1) \mapsto (n-1, \dots, 0)$, producing Qiskit's $|b_{n-1}\dots b_1 b_0\rangle$ ordering with identical global phase across all supported gates.

---

## 3. Verification & Cross-Framework Differential Testing

- **Test command:**
  ```bash
  /home/osiris/osiris-assessment/venvs/osiris-cli/bin/pytest tests/test_qbyte_correctness.py tests/test_core.py -v
  ```
- **Observed result:** **`109 passed in 1.40s`** (`44` new verification/differential tests in `tests/test_qbyte_correctness.py` + `65` existing regression tests in `tests/test_core.py`).
- **Receipt file:** `docs/phase2/qbyte_test_receipt.txt`
- **Declared numerical tolerance:** `atol = 1e-12` (observed discrepancies $\le 4.4 \times 10^{-16}$, machine $\epsilon$ for IEEE-754 `complex128`).
- **Coverage matrix:**
  - All 8 computational basis states of CCNOT (`000`..`111`) on both `Qbyte` and `SovereignExecutor`.
  - All $3! = 6$ 3-qubit permutations and all $8 \times 7 \times 6 = 336$ ordered `(c1, c2, target)` triples on 8 qubits on complex random statevectors.
  - Wire-wise consistency between `cleave(q)` and projective `measure_qubit(q)` across all 8 wires.
  - Gate unitarity ($U^\dagger U = I$) across every fixed and parameterized 1Q, 2Q, and 3Q gate in `qbyte_system.gates`.
  - Full-matrix Kronecker embedding oracle agreement across 13 heterogeneous 1Q/2Q/3Q gates on 6-qubit random normalized statevectors.
  - Exact inverse-circuit restoration ($|\langle\psi_0 | U^\dagger U | \psi_0\rangle - 1| < 10^{-15}$) on scrambled 8-qubit complex statevectors.
  - All 4 Bell states ($|\Phi^\pm\rangle, |\Psi^\pm\rangle$) and GHZ states ($n = 2, 3, 4, 5, 8$ and non-adjacent wire subsets).
  - Mach-Zehnder phase interference ($H \cdot R_Z(\theta) \cdot H |0\rangle$) across 9 phase angles and 3-qubit Quantum Fourier Transform (`qft_circuit(3)`).
  - Differential comparison against `qiskit.quantum_info.Statevector` on deterministic 6-qubit circuits (including `CCX`, `CSWAP`, `CP`, `RXX`, `RZZ`, `S`, `T`, `RX`, `RY`, `RZ`) and 20 randomized 5-qubit circuits.

---

## 4. Performance & Memory Measurements

Measured via `docs/phase2/qbyte_benchmarks.json` (`H` layer + `CX` chain + `CCX` Toffoli cascade):

| Qubits ($n$) | Statevector Dim ($2^n$) | Gate Count | Wall Time (ms) | Time / Gate (µs) | Peak Memory (KiB) | Max $|\Delta\psi|$ vs Qiskit |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 16 | 9 | 0.3409 | 37.873 | 18.26 | `0.0` |
| 8 | 256 | 21 | 0.4609 | 21.949 | 29.70 | `0.0` |
| 12 | 4,096 | 33 | 1.0255 | 31.077 | 210.92 | `0.0` |
| 16 | 65,536 | 45 | 100.8433 | 2,240.962 | 3,095.19 | N/A (skipped >12Q oracle) |
