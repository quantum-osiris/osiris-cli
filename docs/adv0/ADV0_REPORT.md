# ADV-0: Where generic classical simulators stop recovering a peaked circuit on heavy-hex

**Status:** offline study. No QPU time was used. Nothing here is a claim of quantum advantage.

## Question

For peaked circuits the size of an IBM Heron r2 region, does any generic classical attack still recover the hidden string? If so, how does its success compare with what the hardware is predicted to deliver?

## Setup

**Circuits.** `adv0/peaked.py` builds twirled-mirror circuits: C = X^s · R† · R on a connected n-qubit heavy-hex region of FakeMarrakesh, so ideally C|0⟩ = |s⟩.

- **R:** d layers, each of Haar-random SU(2) on every qubit followed by CZ on one edge matching, then one final SU(2) layer.
- **Obfuscation:** every CZ gets a random RZ gauge and a random Pauli twirl, and all single-qubit runs are merged into native rz/sx/x gates.
- **Region:** grown greedily from the lowest-error CZ edge; qubits with 5% or worse readout or CZ error are excluded.

**Attacks** (`adv0/attacks.py`). All are generic, none looks for mirror structure, and each must recover every bit:

- **Pauli propagation:** Heisenberg picture, coefficient cut-off 1e-5, at most 32,768 terms.
- **Exact light-cone statevector:** limited to cones of 26 qubits or fewer.
- **MPS sampling:** bond dimension 64 or 256, 200 shots. It wins if the most frequent sample is s.

**Hardware prediction.** The product of (1 − error) over every native gate and readout, from the backend's calibration. It ignores idle decoherence and crosstalk. Against a full FakeMarrakesh noisy simulation at n = 12 (`noisy_check.jsonl`), it was optimistic by 1–7% (relative): predicted 0.883 against 0.874 simulated at depth 2, and 0.746 against 0.694 at depth 12.

**Tests.** 9 in `tests/test_adv0.py`, all passing. They check that every circuit peaks with probability 1, that there is no literal palindrome, that Pauli propagation matches the statevector exactly when not truncated, and that the cost model falls with depth.

## Results (`scan.jsonl`, 30 cells)

| n | depth | CZ | hw peak (pred.) | Pauli prop. | light cone | MPS χ=64 | MPS χ=256 |
|---|---|---|---|---|---|---|---|
| 12 | 16 | 120 | 0.691 | fail | win | win (1.00, <1 s) | win |
| 20 | 16 | 206 | 0.492 | fail | win | win (0.73, 2 s) | win (1.00, 59 s) |
| 28 | 8 | 158 | 0.471 | fail | fail | win (0.86, 4 s) | win (1.00, 132 s) |
| 28 | 16 | 304 | 0.288 | fail | fail | win (0.17, 8 s) | win (0.58, 683 s) |
| 40 | 12 | 328 | 0.217 | fail | fail | win (0.18, 10 s) | win (0.66, 617 s) |
| 40 | 16 | 446 | 0.146 | fail | fail | win (0.015, 14 s) | win (0.14, 4,884 s) |
| 56 | 8 | 320 | 0.149 | fail | fail | win (0.52, 7 s) | win (0.97, 1,604 s) |
| 56 | 12 | 456 | 0.080 | fail | fail | win (0.13, 11 s) | win (0.57, 12,547 s) |
| 56 | 16 | 618 | 0.041 | fail | fail | **fail** (0.005, 18 s) | win (0.165, 1,274 s) |

The hw and MPS columns give the peak probability; the MPS columns also give run time. Cells at depths 2–6 are in the JSONL: every attack except light-cone wins there.

The Aer MPS run times vary between runs, so treat them as rough. One example: (56, 16) took less time than (56, 12).

## Findings

1. **In every one of the 30 cells, at least one laptop-scale attack recovers s.** On this family and this hardware, the classical side wins across the whole range.
2. **MPS is the attack that matters.** Pauli propagation fails from depth 8–12, and the light-cone attack fails at 28 qubits or more from depth 6. MPS at bond dimension 64 only fails at the largest cell (n = 56, d = 16), and bond dimension 256 still wins there with a peak of 0.165, against 0.041 predicted for hardware.
3. **Peak detection is forgiving for both sides.** A simulator does not need high fidelity, only that s stays the most frequent sample. Whatever makes the hardware's job easy makes the classical job easy too.
4. **Hardware fidelity falls faster than MPS fidelity as the circuit grows:** the predicted peak is 0.041 at 618 CZ. The region grown for low error is long and thin, and MPS handles low-entanglement, nearly 1D geometries well.
5. **QPU cost is not the constraint.** Each cell needs about 0.3 s of QPU time at 1,000 shots.

## What this rules out, and what comes next

Twirled-mirror peaked circuits on a low-error heavy-hex region are **not** an advantage candidate on Heron r2. Even ignoring the structure attack, which removes the mirror outright, generic MPS recovers s at every size where the hardware still has a usable peak.

Any next candidate has to close two gaps at once:

- **Entanglement geometry.** Use regions with heavy-hex loops (12-qubit rings) and brickwork that crosses them, so that MPS cuts run through many bonds. Cost: a lower hardware peak, from worse qubits and more CZ.
- **No mirror.** Use a peaked construction in which no half-circuit inverts the other, for example a random circuit followed by a peaking block optimised to concentrate on s (Aaronson–Zhang style), or a published challenge circuit. Any such construction also has to be run against a structure-aware attack, which needs to be added to the harness.

**Gate for any hardware run:** a cell qualifies only if MPS at bond dimension 1024 or more, Pauli propagation and light-cone all fail, and the hardware peak predicted from calibration is at least 10× the uniform baseline with 1,000 shots in budget. ADV-0 found no cell that passes.
