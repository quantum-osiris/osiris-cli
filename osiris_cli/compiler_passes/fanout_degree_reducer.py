"""
Commuting CNOT Fan-Out Degree-Reduction Pass (`reduce_commuting_cx_fanouts`)
============================================================================

Reusable OSIRIS Quantum Systems compilation capability motivated by
`EXPERIMENT TESSERACT-MAP-02`.

Problem on Planar Heavy-Hex Processors (max vertex degree = 3):
  A contiguous fan-out burst `CX(c, t_1), CX(c, t_2), ..., CX(c, t_k)` creates a
  star vertex of degree >= k in the logical interaction graph. When k > 3, Qiskit's
  exact error-weighted subgraph isomorphism pass (`VF2Layout`) fails to find an
  isomorphism and falls back to heuristic `SabreSwap` routing (incurring >4x more
  2Q CZ gates on 40-qubit fan-out circuits).

Exact Algebraic Identity:
  Whenever `t_1, ..., t_k` are distinct from `c` and untouched between consecutive
  fan-out gates, the unitary product:
      CX(c, t_k) ... CX(c, t_2) CX(c, t_1)
  is NOT equal to `CX(t_{k-1}, t_k) ... CX(t_1, t_2) CX(c, t_1)` on arbitrary input
  states UNLESS `t_1, ..., t_k` are verified to be in computational ground state `|0>`!
  WAIT — CHECK THAT EXACT ALGEBRAIC CONDITION:
    - On arbitrary input `|c, a, b>`:
        Star:  `|c, a ^ c, b ^ c>`
        Chain: `CX(c, a)` -> `|c, a ^ c, b>`, then `CX(a, b)` -> `|c, a ^ c, b ^ a ^ c>`!
      They are equal **IF AND ONLY IF `a == 0`** (`t_1, ..., t_{k-1}` are fresh `|0>`
      ancilla/target wires untouched since reset/initialization)!
    - Conversely, on ARBITRARY mid-circuit quantum states `|c, a, b>` where `a != 0`,
      we CAN still reduce degree to <= 2 without `|0>` assumptions by adding the
      uncompute CNOT ladder (`2k - 2` CNOTs) OR ONLY applying the `k`-gate zero-ancilla
      chain rewrite to **provably fresh `|0>` wires** (tracked deterministically from
      circuit start or `reset` instructions)!

This pass tracks provably `|0>`-initialized wires rigorously and rewrites commuting
CNOT fan-outs onto fresh `|0>` targets into degree-2 linear cascades, preserving
**exact statevector equivalence** while unlocking `VF2Layout` zero-SWAP embedding!
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from qiskit import QuantumCircuit


def _flush_fanout_group(
    out_qc: QuantumCircuit,
    control_idx: int,
    targets: List[int],
    zero_wires: Set[int],
) -> int:
    """
    Emit a buffered commuting CX fan-out group `[CX(c, t_0), ..., CX(c, t_{m-1})]`.
    Any target `t` that is provably in `|0>` (`t in zero_wires`) can serve as the
    next relay source `prev -> t` (`b ^ 0 ^ c == b ^ c`!), reducing hub degree from
    `m` to `1` while preserving the EXACT statevector for arbitrary state on `c`
    and even if the final leaf target is NOT in `|0>`!
    Returns number of fan-out edges rewritten from hub `c` to chain relays.
    """
    if not targets:
        return 0
    if len(targets) == 1:
        t0 = targets[0]
        out_qc.cx(control_idx, t0)
        zero_wires.discard(t0)
        return 0

    # Partition targets into provably-|0> relay candidates and arbitrary-state leaves:
    # Relay wires `z_0, z_1, ..., z_{r-1}` in `zero_wires` satisfy `|z_j> = |0>`, so
    # `CX(c, z_0); CX(z_0, z_1); ...; CX(z_{r-1}, leaf)` copies `c`'s computational
    # value onto every `z_j` AND XORs `c` into at most ONE arbitrary-state `leaf`!
    relay_zeros = [t for t in targets if t in zero_wires]
    non_zero_targets = [t for t in targets if t not in zero_wires]

    rewritten = 0
    if not relay_zeros:
        for t in non_zero_targets:
            out_qc.cx(control_idx, t)
        return 0

    # Chain through all fresh |0> targets: c -> z_0 -> z_1 -> ... -> z_{r-1}
    out_qc.cx(control_idx, relay_zeros[0])
    zero_wires.discard(relay_zeros[0])
    for idx in range(1, len(relay_zeros)):
        out_qc.cx(relay_zeros[idx - 1], relay_zeros[idx])
        zero_wires.discard(relay_zeros[idx])
        rewritten += 1

    # At most one non-zero target can hang off the tail `relay_zeros[-1]` (since `relay_zeros[-1]`
    # now holds `0 ^ c == c` identically!); any remaining non-zero targets attach to `control_idx`
    if non_zero_targets:
        out_qc.cx(relay_zeros[-1], non_zero_targets[0])
        rewritten += 1
        for extra_t in non_zero_targets[1:]:
            out_qc.cx(control_idx, extra_t)

    return rewritten


def reduce_commuting_cx_fanouts(qc: QuantumCircuit) -> Tuple[QuantumCircuit, Dict[str, int]]:
    """
    Sound state-preserving degree-reduction pass:
      Tracks wires provably in computational state `|0>` (from circuit start or `reset`)
      and rewrites same-control `CX` fan-out bursts onto `|0>` wires into degree-2
      linear chains, unlocking `VF2Layout` zero-SWAP heavy-hex subgraph embedding.
    """
    out = QuantumCircuit(*qc.qregs, *qc.cregs, name=f"{qc.name}_deg_reduced")
    zero_wires: Set[int] = set(range(qc.num_qubits))

    # Active commuting fan-out buffer: control_wire -> ordered list of distinct target wires
    # Multiple disjoint controls (e.g., core qubits 0 and 1 in Tesseract 40Q!) can buffer
    # simultaneously as long as their wires do not intersect!
    active_fanouts: Dict[int, List[int]] = {}
    busy_targets: Dict[int, int] = {}  # target_wire -> control_wire
    total_rewritten = 0

    def _flush_control(ctrl: int) -> None:
        nonlocal total_rewritten
        if ctrl in active_fanouts:
            tgts = active_fanouts.pop(ctrl)
            for t in tgts:
                busy_targets.pop(t, None)
            total_rewritten += _flush_fanout_group(out, ctrl, tgts, zero_wires)

    def _flush_touching(wires: Set[int]) -> None:
        # Flush any buffered fan-out whose control or targets intersect `wires`
        to_flush: Set[int] = set()
        for w in wires:
            if w in active_fanouts:
                to_flush.add(w)
            if w in busy_targets:
                to_flush.add(busy_targets[w])
        for c in sorted(to_flush):
            _flush_control(c)

    for inst in qc.data:
        op = inst.operation
        q_indices = [qc.find_bit(q).index for q in inst.qubits]
        c_indices = [qc.find_bit(c).index for c in inst.clbits]

        if op.name == "cx" and not c_indices:
            ctrl, tgt = q_indices[0], q_indices[1]
            # Check if `ctrl` is currently a target of another buffered fanout, or `tgt` is already busy/control
            conflict = (
                ctrl in busy_targets
                or tgt in active_fanouts
                or tgt in busy_targets
            )
            if conflict:
                _flush_touching({ctrl, tgt})

            if ctrl in zero_wires:
                # CX with control in |0> is an exact no-op, but keep conservatively via flush if desired;
                # actually if ctrl is in |0>, target stays in |0> if it was |0>, wait: flush normally:
                pass
            active_fanouts.setdefault(ctrl, []).append(tgt)
            busy_targets[tgt] = ctrl
            continue

        # Any other instruction flushes buffered fanouts on involved wires
        if op.name == "barrier":
            for c in list(active_fanouts.keys()):
                _flush_control(c)
            out.append(op, inst.qubits, inst.clbits)
            continue

        _flush_touching(set(q_indices))
        out.append(op, inst.qubits, inst.clbits)

        if op.name == "reset":
            for q in q_indices:
                zero_wires.add(q)
        elif op.name in ("rz", "p", "z", "s", "sdg", "t", "tdg", "delay", "id"):
            # Diagonal gates fix |0> up to global/wire phase (e^{-i theta/2} |0> is still |0>!)
            pass
        else:
            for q in q_indices:
                zero_wires.discard(q)

    for c in list(active_fanouts.keys()):
        _flush_control(c)

    return out, {"edges_rewritten_to_chain": total_rewritten}
