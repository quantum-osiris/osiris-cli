#!/usr/bin/env python3
"""OSIRIS-XTOP-006 Quantinuum H2-1 QCCD Tunable Non-Clifford Scrambling & [[4,2,2]] Circuits.

Constructs, offline-compiles (`QuantinuumAPIOffline` -> `H2-1`, 0 routing SWAPs), verifies against
analytical Kraus contractions (`kraus_channel.py`), and simulates under H2-like Pauli + QCCD memory
noise across:
  * theta in {0, pi/8, pi/4, 3*pi/8, pi/2}
  * |psi> in {|0>, |1>, |+>, |->, |+i>, |-i>}  (6-state Pauli witness / tomography)

Two experimental arms:
  1. `P0` (`PHYSICAL_7Q`, 7 qubits):
       7-qubit Landsman-Figgatt tunable non-Clifford Hayden-Preskill recovery compiled to native
       `H2-1` `ZZPhase(alpha)` + `ZZMax` + `PhasedX` + `Rz` (`0` SWAPs).
  2. `E1` (`ENCODED_422_HYBRID`, 12 qubits):
       Bob's idling output + co-encoded reference register encoded in a `[[4,2,2]]` block
       `B_out = (q6, q7, q8, q9)` with ancilla-verified `|00>_L` state preparation (`anc=q10`),
       weight-2 logical `X_{L1}` entanglement with `q3`, `|0>_{L2}` logical reference parity witness,
       and flagged weight-4 `X` (`XXXX`) and `Z` (`ZZZZ`) syndrome checks (`anc=q10, flag=q11`)
       protecting the QCCD storage-zone memory qubits during tunable non-Clifford `ZZPhase(theta)`
       scrambling (`0` SWAPs).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import rfc8785

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from osiris_cli.providers.nexus_sdk import batch_sha256, local_hqc, tket_sha256  # noqa: E402
from xtop005 import code422 as c422  # noqa: E402
from xtop006 import kraus_channel as kc  # noqa: E402

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MANIFEST_FILENAME = "xtop006_h2_manifest.json"
CIRCUITS_BUNDLE_FILENAME = "xtop006_h2_1_compiled_circuits.json"
SHA256SUMS_FILENAME = "SHA256SUMS.txt"

H2_NATIVE_OP_NAMES = frozenset(
    {"PhasedX", "Rz", "ZZMax", "ZZPhase", "Measure", "Reset", "Barrier"}
)

THETA_LABELS: Tuple[Tuple[str, float], ...] = (
    ("0", 0.0),
    ("pi_8", math.pi / 8.0),
    ("pi_4", math.pi / 4.0),
    ("3pi_8", 3.0 * math.pi / 8.0),
    ("pi_2", math.pi / 2.0),
)

STATE_ORDER: Tuple[str, ...] = ("0", "1", "+", "-", "+i", "-i")

ARM_CANONICAL: Dict[str, str] = {
    "P0": "P0",
    "PHYSICAL_7Q": "P0",
    "E1": "E1",
    "ENCODED_422_HYBRID": "E1",
    "ENCODED_422_REF": "E1",
}
ARM_LABELS: Dict[str, str] = {
    "P0": "PHYSICAL_7Q",
    "E1": "ENCODED_422_HYBRID",
}


@dataclass(frozen=True)
class BuiltXtop006Circuit:
    arm: str
    arm_label: str
    theta_name: str
    theta_rad: float
    state_label: str
    readout_basis: str
    ideal_eigenvalue: int
    uncompiled_tket: Any
    compiled_tket: Any
    bell_clbits: Tuple[int, int]
    out_clbits: Tuple[int, ...]
    postselect_clbits: Tuple[int, ...]
    memory_qubits: Tuple[int, ...]


def _prep_input_state_tk(circ: Any, qubit: int, state_label: str) -> None:
    if state_label == "0":
        return
    if state_label == "1":
        circ.X(qubit)
    elif state_label == "+":
        circ.H(qubit)
    elif state_label == "-":
        circ.X(qubit)
        circ.H(qubit)
    elif state_label == "+i":
        circ.H(qubit)
        circ.S(qubit)
    elif state_label == "-i":
        circ.H(qubit)
        circ.Sdg(qubit)
    else:
        raise ValueError(f"Unsupported input state {state_label!r}")


def _add_tunable_cx_tk(circ: Any, theta: float, ctrl: int, tgt: int) -> None:
    """Emit tunable entangler `(I x H) diag(1, 1, 1, exp(i*2*theta)) (I x H)` (up to global phase)
    using a single native `ZZPhase(-theta / pi)` gate plus virtual `Rz(theta / pi)` and `H`.
    """
    from pytket import OpType

    if abs(theta) < 1e-13:
        return
    circ.H(tgt)
    circ.Rz(theta / math.pi, ctrl)
    circ.Rz(theta / math.pi, tgt)
    circ.add_gate(OpType.ZZPhase, [-theta / math.pi], [ctrl, tgt])
    circ.H(tgt)


def add_scrambler_tk(
    circ: Any,
    theta: float,
    q0: int,
    q1: int,
    q2: int,
    conjugate: bool = False,
) -> None:
    """Append 3-qubit tunable non-Clifford scrambler U(theta) (or U*(theta) when `conjugate=True`)."""
    angle = -theta if conjugate else theta
    for q in (q0, q1, q2):
        circ.H(q)
    _add_tunable_cx_tk(circ, angle, q0, q1)
    _add_tunable_cx_tk(circ, angle, q0, q2)
    for q in (q0, q1, q2):
        circ.H(q)
    _add_tunable_cx_tk(circ, angle, q0, q1)
    _add_tunable_cx_tk(circ, angle, q0, q2)


def _offline_h2_backend() -> Any:
    from pytket.extensions.quantinuum import QuantinuumAPIOffline, QuantinuumBackend

    return QuantinuumBackend("H2-1", api_handler=QuantinuumAPIOffline())


def build_xtop006_circuit(
    arm: str,
    theta: float,
    state_label: str,
    theta_name: Optional[str] = None,
    backend: Optional[Any] = None,
    optimisation_level: int = 2,
) -> BuiltXtop006Circuit:
    """Construct and compile an OSIRIS-XTOP-006 circuit (`P0` 7Q or `E1` 12Q) for Quantinuum `H2-1`."""
    from pytket import Circuit, OpType

    canon_arm = ARM_CANONICAL.get(arm)
    if canon_arm is None:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {sorted(ARM_CANONICAL)}")
    if state_label not in kc.INPUT_STATES_6:
        raise ValueError(f"Unknown state {state_label!r}; expected one of {STATE_ORDER}")

    _, basis, eig = kc.INPUT_STATES_6[state_label]
    t_name = theta_name if theta_name is not None else f"{theta:.4f}"
    safe_st = (
        state_label.replace("+", "plus")
        .replace("-", "minus")
    )
    be = backend if backend is not None else _offline_h2_backend()

    if canon_arm == "P0":
        cname = f"xtop006_P0_th_{t_name}_st_{safe_st}"
        circ = Circuit(7, 3, name=cname)
        _prep_input_state_tk(circ, 0, state_label)
        for a, b in ((1, 5), (2, 4), (3, 6)):
            circ.H(a)
            circ.CX(a, b)
        circ.add_barrier(list(range(7)))
        add_scrambler_tk(circ, theta, 0, 1, 2, conjugate=False)
        add_scrambler_tk(circ, theta, 3, 5, 4, conjugate=True)
        circ.add_barrier(list(range(7)))
        # Bell measurement on Hawking pair (q2, q4) -> clbits (0, 1)
        circ.CX(2, 4)
        circ.H(2)
        circ.Measure(2, 0)
        circ.Measure(4, 1)
        # Pauli witness readout on Bob's output q6 -> clbit 2
        if basis == "X":
            circ.H(6)
        elif basis == "Y":
            circ.Sdg(6)
            circ.H(6)
        circ.Measure(6, 2)

        comp = be.get_compiled_circuit(circ, optimisation_level=optimisation_level)
        comp.name = cname
        return BuiltXtop006Circuit(
            arm="P0",
            arm_label=ARM_LABELS["P0"],
            theta_name=t_name,
            theta_rad=float(theta),
            state_label=state_label,
            readout_basis=basis,
            ideal_eigenvalue=eig,
            uncompiled_tket=circ,
            compiled_tket=comp,
            bell_clbits=(0, 1),
            out_clbits=(2,),
            postselect_clbits=(),
            memory_qubits=(6,),
        )

    # Arm E1 (`ENCODED_422_HYBRID`, 12 qubits):
    # q0..q5 active HP registers; B_out = (6, 7, 8, 9) [[4,2,2]] block; anc = 10, flag = 11.
    cname = f"xtop006_E1_th_{t_name}_st_{safe_st}"
    p_init = c422.Program(12)
    ver_cb = c422.prep_zero(p_init, (6, 7, 8, 9), verify_anc=10)
    assert ver_cb is not None
    # Fault-tolerant weight-2 logical X_{L1} = X_6 X_7 entanglement with q3; L2 stays in |0>_{L2}
    p_init.add("h", [3])
    p_init.add("cx", [3, 6])
    p_init.add("cx", [3, 7])
    for a, b in ((1, 5), (2, 4)):
        p_init.add("h", [a])
        p_init.add("cx", [a, b])

    p_tail = c422.Program(12, n_clbits=p_init.n_clbits)
    sx = c422.check(p_tail, (6, 7, 8, 9), "X", 10, 11)
    sz = c422.check(p_tail, (6, 7, 8, 9), "Z", 10, 11)
    bell_cbs = p_tail.new_clbits(2)
    p_tail.add("cx", [2, 4])
    p_tail.add("h", [2])
    p_tail.add("measure", [2], [bell_cbs[0]])
    p_tail.add("measure", [4], [bell_cbs[1]])
    ro_cbs = c422.measure_logical(p_tail, (6, 7, 8, 9), basis, "Z")

    circ = Circuit(12, p_tail.n_clbits, name=cname)
    op_map = {
        "h": OpType.H,
        "x": OpType.X,
        "y": OpType.Y,
        "z": OpType.Z,
        "s": OpType.S,
        "sdg": OpType.Sdg,
        "cx": OpType.CX,
        "cz": OpType.CZ,
        "reset": OpType.Reset,
    }
    _prep_input_state_tk(circ, 0, state_label)
    for g, qs, cs in p_init.gates:
        if g == "measure":
            circ.Measure(qs[0], cs[0])
        else:
            circ.add_gate(op_map[g], list(qs))
    circ.add_barrier(list(range(12)))
    add_scrambler_tk(circ, theta, 0, 1, 2, conjugate=False)
    add_scrambler_tk(circ, theta, 3, 5, 4, conjugate=True)
    circ.add_barrier(list(range(12)))
    for g, qs, cs in p_tail.gates:
        if g == "measure":
            circ.Measure(qs[0], cs[0])
        else:
            circ.add_gate(op_map[g], list(qs))

    comp = be.get_compiled_circuit(circ, optimisation_level=optimisation_level)
    comp.name = cname
    return BuiltXtop006Circuit(
        arm="E1",
        arm_label=ARM_LABELS["E1"],
        theta_name=t_name,
        theta_rad=float(theta),
        state_label=state_label,
        readout_basis=basis,
        ideal_eigenvalue=eig,
        uncompiled_tket=circ,
        compiled_tket=comp,
        bell_clbits=(bell_cbs[0], bell_cbs[1]),
        out_clbits=tuple(ro_cbs),
        postselect_clbits=(ver_cb, sx[0], sx[1], sz[0], sz[1]),
        memory_qubits=(6, 7, 8, 9),
    )


def compiled_gate_counts(tk_circ: Any) -> Dict[str, int]:
    """Count H2-1 ISA operations and assert 0 routing SWAPs and native gate compliance."""
    from pytket import OpType

    n1_total = 0
    n1_phys = 0
    n1_rz = 0
    n2 = 0
    n2_zzphase = 0
    n2_zzmax = 0
    n_meas = 0
    n_reset = 0
    swaps = 0
    for cmd in tk_circ.get_commands():
        op = cmd.op
        while op.type == OpType.Conditional:
            op = op.op
        t = op.type
        name = t.name
        if name == "SWAP":
            swaps += 1
        if name not in H2_NATIVE_OP_NAMES:
            raise ValueError(f"Non-native H2-1 gate {name!r} in {tk_circ.name}")
        if t == OpType.Measure:
            n_meas += 1
        elif t == OpType.Reset:
            n_reset += 1
        elif t == OpType.Barrier or not cmd.qubits:
            continue
        elif len(cmd.qubits) == 1:
            n1_total += 1
            if t == OpType.Rz:
                n1_rz += 1
            else:
                n1_phys += 1
        elif len(cmd.qubits) == 2:
            n2 += 1
            if t == OpType.ZZPhase:
                n2_zzphase += 1
            elif t == OpType.ZZMax:
                n2_zzmax += 1
    n_qubits = int(tk_circ.n_qubits)
    n_m = n_meas + n_reset + n_qubits
    return {
        "qubits": n_qubits,
        "N_1q": n1_total,
        "N_1q_physical": n1_phys,
        "N_1q_virtual_rz": n1_rz,
        "N_2q": n2,
        "N_2q_zzphase": n2_zzphase,
        "N_2q_zzmax": n2_zzmax,
        "N_meas": n_meas,
        "N_reset": n_reset,
        "N_m": n_m,
        "routing_swaps": swaps,
    }


def compute_circuit_hqc(counts: Mapping[str, int], shots: int) -> Dict[str, str]:
    """Return both conservative (`Rz`-counted) and physical (`Rz`-free) HQC strings."""
    cons = 5.0 + shots * (counts["N_1q"] + 10 * counts["N_2q"] + 5 * counts["N_m"]) / 5000.0
    phys = 5.0 + shots * (counts["N_1q_physical"] + 10 * counts["N_2q"] + 5 * counts["N_m"]) / 5000.0
    return {
        "conservative": f"{cons:.4f}",
        "physical": f"{phys:.4f}",
    }


# ---------------------------------------------------------------------------
# Exact Noiseless Verification Against `kraus_channel.py`
# ---------------------------------------------------------------------------


def verify_p0_compiled_against_kraus(
    theta: float,
    state_label: str,
    backend: Optional[Any] = None,
    atol: float = 1e-11,
) -> Dict[str, Any]:
    """Extract the exact 7-qubit pre-measurement statevector from an H2-1 compiled `P0` circuit
    and verify that all 4 Bell-postselected 2x2 output density matrices match `kraus_channel.py`.
    """
    from pytket import Circuit, OpType

    be = backend if backend is not None else _offline_h2_backend()
    psi, _, _ = kc.INPUT_STATES_6[state_label]
    u3 = kc.landsman_figgatt_scrambler(theta)

    # Anchor all 7 logical qubits to classical bits 0..6 so Measure(node, bit) records exact placement
    c_pre = Circuit(7, 7, name="xtop006_p0_pre_meas")
    _prep_input_state_tk(c_pre, 0, state_label)
    for a, b in ((1, 5), (2, 4), (3, 6)):
        c_pre.H(a)
        c_pre.CX(a, b)
    add_scrambler_tk(c_pre, theta, 0, 1, 2, conjugate=False)
    add_scrambler_tk(c_pre, theta, 3, 5, 4, conjugate=True)
    c_pre.add_barrier(list(range(7)))
    for q in range(7):
        c_pre.Measure(q, q)

    c_comp = be.get_compiled_circuit(c_pre, optimisation_level=2)
    c_comp.replace_implicit_wire_swaps()
    comp_qubits = list(c_comp.qubits)
    bit_to_axis: Dict[int, int] = {}
    c_clean = Circuit(7)
    for cmd in c_comp.get_commands():
        t = cmd.op.type
        if t == OpType.Barrier:
            continue
        if t == OpType.Measure:
            bit_idx = cmd.bits[0].index[0]
            bit_to_axis[bit_idx] = comp_qubits.index(cmd.qubits[0])
            continue
        qs_idx = [comp_qubits.index(q) for q in cmd.qubits]
        c_clean.add_gate(cmd.op, qs_idx)

    logical_to_axis = [bit_to_axis[i] for i in range(7)]
    sv_raw = c_clean.get_statevector().reshape(2, 2, 2, 2, 2, 2, 2)
    # Permute axes so axis i corresponds to original logical qubit q_i (0..6)
    psi_tensor = np.transpose(sv_raw, logical_to_axis)
    max_rho_err = 0.0
    max_prob_err = 0.0
    for beta in kc.BELL_OUTCOMES:
        b_mat = kc.BELL_MATRICES[beta]
        # Project (q2, q4) onto <beta|: chi_{i, j, m, n, x} where i=q0, j=q1, m=q5, n=q3, x=q6
        chi = np.einsum("kl,ijknlmx->ijmnx", np.conj(b_mat), psi_tensor).reshape(16, 2)
        rho_tilde_circ = chi.T @ np.conj(chi)  # wait: sum_r |chi_{r,x}><chi_{r,y}| -> chi[r,x]*conj(chi[r,y])
        rho_tilde_circ = np.einsum("rx,ry->xy", chi, np.conj(chi))
        ev = kc.apply_bell_channel(u3, psi, bell_outcome=beta)
        err_rho = float(np.linalg.norm(rho_tilde_circ - ev.unnormalized_rho_out, ord="fro"))
        err_prob = abs(float(np.real(np.trace(rho_tilde_circ))) - ev.success_probability)
        max_rho_err = max(max_rho_err, err_rho)
        max_prob_err = max(max_prob_err, err_prob)

    if max_rho_err > atol or max_prob_err > atol:
        raise AssertionError(
            f"Compiled P0 vs Kraus mismatch at theta={theta}, state={state_label}: "
            f"rho_err={max_rho_err:.3e}, prob_err={max_prob_err:.3e}"
        )
    return {
        "passed": True,
        "max_unnormalized_rho_frobenius_error": max_rho_err,
        "max_probability_error": max_prob_err,
    }


def _exact_noiseless_e1_metrics(built: BuiltXtop006Circuit) -> Dict[str, float]:
    """Deterministically evolve the 12-qubit compiled `E1` circuit in pure NumPy statevector form,
    verifying that every mid-circuit syndrome/flag check deterministically yields 0 (`p_keep = 1.0`)
    and computing exact Born probabilities of all final measurement outcomes.
    """
    from pytket import OpType

    assert built.arm == "E1"
    c_work = built.compiled_tket.copy()
    c_work.replace_implicit_wire_swaps()
    n_q = c_work.n_qubits
    n_c = c_work.n_bits
    assert n_q == 12

    # Statevector of shape (2,)*12
    sv = np.zeros((2,) * n_q, dtype=np.complex128)
    sv[(0,) * n_q] = 1.0

    def apply_1q(mat: np.ndarray, q: int) -> None:
        nonlocal sv
        sv = np.tensordot(mat, sv, axes=([1], [q]))
        sv = np.moveaxis(sv, 0, q)

    def apply_2q(mat4: np.ndarray, q0: int, q1: int) -> None:
        nonlocal sv
        m = mat4.reshape(2, 2, 2, 2)
        sv = np.tensordot(m, sv, axes=([2, 3], [q0, q1]))
        # axes 0, 1 are now the new q0, q1; move them back to positions q0, q1
        sv = np.moveaxis(sv, [0, 1], [q0, q1])

    def phasedx_mat(alpha: float, beta: float) -> np.ndarray:
        # pytket PhasedX(alpha, beta) = Rz(beta) Rx(alpha) Rz(-beta) with angle in half-turns (pi*alpha)
        ca = math.cos(0.5 * math.pi * alpha)
        sa = math.sin(0.5 * math.pi * alpha)
        ep = np.exp(-1.0j * math.pi * beta)
        em = np.exp(+1.0j * math.pi * beta)
        return np.array([[ca, -1.0j * sa * ep], [-1.0j * sa * em, ca]], dtype=np.complex128)

    def rz_mat(alpha: float) -> np.ndarray:
        return np.array(
            [
                [np.exp(-0.5j * math.pi * alpha), 0.0],
                [0.0, np.exp(0.5j * math.pi * alpha)],
            ],
            dtype=np.complex128,
        )

    def zzphase_mat(alpha: float) -> np.ndarray:
        em = np.exp(-0.5j * math.pi * alpha)
        ep = np.exp(+0.5j * math.pi * alpha)
        return np.diag([em, ep, ep, em]).astype(np.complex128)

    post_set = set(built.postselect_clbits)
    final_meas_map: Dict[int, int] = {}  # clbit -> qubit for non-postselected measurements
    comp_qubits = list(c_work.qubits)

    for cmd in c_work.get_commands():
        t = cmd.op.type
        if t == OpType.Barrier:
            continue
        qs = [comp_qubits.index(q) for q in cmd.qubits]
        cs = [b.index[0] for b in cmd.bits]
        params = [float(p) for p in cmd.op.params]
        if t == OpType.PhasedX:
            apply_1q(phasedx_mat(params[0], params[1]), qs[0])
        elif t == OpType.Rz:
            apply_1q(rz_mat(params[0]), qs[0])
        elif t == OpType.ZZPhase:
            apply_2q(zzphase_mat(params[0]), qs[0], qs[1])
        elif t == OpType.ZZMax:
            apply_2q(zzphase_mat(0.5), qs[0], qs[1])
        elif t == OpType.Measure:
            cb = cs[0]
            q = qs[0]
            if cb in post_set:
                # Syndrome / verification ancilla: verify P(bit == 1) == 0 in noiseless limit!
                slicer1 = [slice(None)] * n_q
                slicer1[q] = 1
                p1 = float(np.sum(np.abs(sv[tuple(slicer1)]) ** 2))
                if p1 > 1e-11:
                    raise AssertionError(
                        f"Noiseless E1 syndrome clbit {cb} on q{q} had nonzero P(1)={p1:.3e}"
                    )
                sv[tuple(slicer1)] = 0.0
            else:
                final_meas_map[cb] = q
        elif t == OpType.Reset:
            # Ancilla was just projected onto |0>, verify P(1) == 0
            q = qs[0]
            slicer1 = [slice(None)] * n_q
            slicer1[q] = 1
            p1 = float(np.sum(np.abs(sv[tuple(slicer1)]) ** 2))
            if p1 > 1e-11:
                raise AssertionError(f"Noiseless E1 reset on q{q} saw uncollapsed |1> amplitude {p1:.3e}")
        else:
            raise ValueError(f"Unhandled op {t} in exact E1 statevector verifier")

    # Compute exact Born probabilities over the 6 final measured qubits (2 Bell + 4 B_out)
    probs = np.abs(sv) ** 2
    b0_q = final_meas_map[built.bell_clbits[0]]
    b1_q = final_meas_map[built.bell_clbits[1]]
    out_qs = [final_meas_map[cb] for cb in built.out_clbits]

    # Sum over unmeasured qubits (remnants q0, q1, q3, q5 and ancillas q10, q11)
    keep_axes = [b0_q, b1_q] + out_qs
    sum_axes = tuple(ax for ax in range(n_q) if ax not in keep_axes)
    marg = np.sum(probs, axis=sum_axes)
    # Reorder axes of `marg` to match `keep_axes`
    sorted_keep = sorted(keep_axes)
    perm = [sorted_keep.index(ax) for ax in keep_axes]
    marg6 = np.transpose(marg, perm)  # shape (2, 2, 2, 2, 2, 2) -> (b0, b1, o0, o1, o2, o3)

    p_keep = 0.0
    p_phi_plus = 0.0
    eig_sum = 0.0
    for idx in np.ndindex(2, 2, 2, 2, 2, 2):
        pr = float(marg6[idx])
        if pr <= 1e-16:
            continue
        b0, b1 = idx[0], idx[1]
        obits = idx[2:]
        dec = c422.decode_block(obits, built.readout_basis, "Z")
        ok = bool(dec["check_ok"]) and (dec["L2"] == 1)
        if not ok:
            raise AssertionError(
                f"Noiseless E1 readout check rejected outcome {obits} with probability {pr:.3e}"
            )
        p_keep += pr
        val = -dec["L1"] if built.readout_basis == "Y" else dec["L1"]
        if b0 == 0 and b1 == 0:
            p_phi_plus += pr
            eig_sum += pr * (built.ideal_eigenvalue * val)

    exp_val = eig_sum / p_phi_plus
    fid = 0.5 * (1.0 + exp_val)
    return {
        "p_keep": p_keep,
        "p_phi_plus": p_phi_plus,
        "fidelity": fid,
        "n_clbits": float(n_c),
    }


def verify_e1_compiled_against_kraus(
    built: BuiltXtop006Circuit,
    atol: float = 1e-11,
) -> Dict[str, Any]:
    """Verify that compiled 12Q `E1` matches analytical `kraus_channel.py` in the noiseless limit."""
    psi, _, _ = kc.INPUT_STATES_6[built.state_label]
    u3 = kc.landsman_figgatt_scrambler(built.theta_rad)
    ev = kc.apply_bell_channel(u3, psi, bell_outcome="Phi+")
    exact = _exact_noiseless_e1_metrics(built)

    err_keep = abs(exact["p_keep"] - 1.0)
    err_prob = abs(exact["p_phi_plus"] - ev.success_probability)
    err_fid = abs(exact["fidelity"] - ev.fidelity_with_input)
    if err_keep > atol or err_prob > atol or err_fid > atol:
        raise AssertionError(
            f"Compiled E1 vs Kraus mismatch ({built.compiled_tket.name}): "
            f"keep_err={err_keep:.3e}, prob_err={err_prob:.3e}, fid_err={err_fid:.3e}"
        )
    return {
        "passed": True,
        "p_keep": exact["p_keep"],
        "p_phi_plus": exact["p_phi_plus"],
        "fidelity": exact["fidelity"],
        "prob_error_vs_kraus": err_prob,
        "fidelity_error_vs_kraus": err_fid,
    }


# ---------------------------------------------------------------------------
# H2-Like Pauli + QCCD Storage-Zone Memory Noise Simulation
# ---------------------------------------------------------------------------


def _decode_shot_counts(counts: Mapping[str, int], built: BuiltXtop006Circuit) -> Dict[str, float]:
    total = 0
    code_acc = 0
    bell_phi_plus = 0
    eig_sum = 0.0
    for key, n in counts.items():
        bits = [int(b) for b in key.replace(" ", "")[::-1]]
        total += int(n)
        if built.arm == "E1":
            syn_ok = all(bits[i] == 0 for i in built.postselect_clbits)
            dec = c422.decode_block([bits[i] for i in built.out_clbits], built.readout_basis, "Z")
            ok = syn_ok and bool(dec["check_ok"]) and (dec["L2"] == 1)
            val = -dec["L1"] if built.readout_basis == "Y" else dec["L1"]
        else:
            ok = True
            val = 1 - 2 * bits[built.out_clbits[0]]
        if ok:
            code_acc += int(n)
            b0 = bits[built.bell_clbits[0]]
            b1 = bits[built.bell_clbits[1]]
            if b0 == 0 and b1 == 0:
                bell_phi_plus += int(n)
                eig_sum += float(n) * float(built.ideal_eigenvalue * val)

    p_keep = code_acc / total if total > 0 else 0.0
    p_phi = bell_phi_plus / code_acc if code_acc > 0 else 0.0
    exp_val = eig_sum / bell_phi_plus if bell_phi_plus > 0 else 0.0
    fid = 0.5 * (1.0 + exp_val)
    return {
        "total_shots": float(total),
        "code_accepted_shots": float(code_acc),
        "bell_phi_plus_shots": float(bell_phi_plus),
        "p_keep": p_keep,
        "p_phi_plus": p_phi,
        "fidelity": fid,
    }


def _to_qiskit_with_memory_marker(built: BuiltXtop006Circuit) -> Any:
    """Convert compiled H2-1 pytket circuit to Qiskit (resolving implicit wire swaps) and insert
    `id` markers on QCCD storage-zone `memory_qubits` at the end of the barrier-delimited scrambling window.
    """
    from pytket.extensions.qiskit import tk_to_qiskit
    from qiskit import QuantumCircuit

    qc = tk_to_qiskit(built.compiled_tket, replace_implicit_swaps=True)
    out = QuantumCircuit(*qc.qregs, *qc.cregs, name=qc.name)
    barriers_seen = 0
    for inst in qc.data:
        if inst.operation.name == "barrier":
            barriers_seen += 1
            if barriers_seen == 2:
                for mq in built.memory_qubits:
                    out.id(mq)
        out.append(inst)
    return out


def build_h2_qccd_noise_model(
    p_1q: float = 3.0e-5,
    p_2q: float = 1.0e-3,
    p_spam: float = 1.0e-3,
    p_mem: float = 0.045,
) -> Any:
    """Build Qiskit Aer Pauli noise model for H2-1 gates + QCCD storage-zone memory dephasing `p_mem`.
    Memory channel uses Z-biased trapped-ion phase/depolarizing noise (70% Z, 15% X, 15% Y).
    """
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, pauli_error

    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(
        depolarizing_error(p_1q, 1),
        ["r", "rx", "ry", "h", "x", "y", "sx"],
    )
    nm.add_all_qubit_quantum_error(
        depolarizing_error(p_2q, 2),
        ["rzz", "cx", "cz"],
    )
    nm.add_all_qubit_quantum_error(
        pauli_error([("X", p_spam), ("I", 1.0 - p_spam)]),
        ["reset"],
    )
    nm.add_all_qubit_readout_error(
        ReadoutError([[1.0 - p_spam, p_spam], [p_spam, 1.0 - p_spam]])
    )
    if p_mem > 0.0:
        nm.add_all_qubit_quantum_error(
            pauli_error(
                [
                    ("Z", 0.70 * p_mem),
                    ("X", 0.15 * p_mem),
                    ("Y", 0.15 * p_mem),
                    ("I", 1.0 - p_mem),
                ]
            ),
            ["id"],
        )
    return nm


def run_xtop006_experiment_and_write_manifest(
    artifacts_dir: Path = ARTIFACTS_DIR,
    shots_sim: int = 4000,
    shots_hqc: int = 500,
    p_1q: float = 3.0e-5,
    p_2q: float = 1.0e-3,
    p_spam: float = 1.0e-3,
    p_mem: float = 0.045,
    seed: int = 20261123,
) -> Dict[str, Any]:
    """Build & compile all 60 H2-1 circuits (`P0` & `E1` x 5 thetas x 6 states), verify exact
    noiseless Kraus agreement, run H2 Pauli + memory-dephasing simulation, and emit JCS manifest.
    """
    from qiskit_aer import AerSimulator

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    be = _offline_h2_backend()

    sim_mem = AerSimulator(
        noise_model=build_h2_qccd_noise_model(p_1q=p_1q, p_2q=p_2q, p_spam=p_spam, p_mem=p_mem)
    )
    sim_gate_only = AerSimulator(
        noise_model=build_h2_qccd_noise_model(p_1q=p_1q, p_2q=p_2q, p_spam=p_spam, p_mem=0.0)
    )

    compiled_bundle: List[Dict[str, Any]] = []
    circuit_records: List[Dict[str, Any]] = []
    theta_cells: List[Dict[str, Any]] = []

    arm_totals: Dict[str, Dict[str, Any]] = {
        arm: {
            "arm": arm,
            "label": ARM_LABELS[arm],
            "n_circuits": 0,
            "qubits": 7 if arm == "P0" else 12,
            "N_1q": 0,
            "N_1q_physical": 0,
            "N_1q_virtual_rz": 0,
            "N_2q": 0,
            "N_meas": 0,
            "N_reset": 0,
            "N_m": 0,
            "routing_swaps": 0,
            "hqc_total_500_conservative": 0.0,
            "hqc_total_500_physical": Decimal("0"),
        }
        for arm in ("P0", "E1")
    }

    for t_idx, (t_name, theta) in enumerate(THETA_LABELS):
        u3 = kc.landsman_figgatt_scrambler(theta)
        kraus_summary = kc.evaluate_6state_summary(u3, bell_outcome="Phi+")
        cptp_info = kc.verify_cptp_completeness(u3)
        if not cptp_info["cptp_valid"]:
            raise AssertionError(f"CPTP completeness failed at theta={t_name}")

        per_arm_theta: Dict[str, Any] = {}
        for arm in ("P0", "E1"):
            built_list: List[BuiltXtop006Circuit] = []
            qcs_marked: List[Any] = []
            max_noiseless_err = 0.0

            for st in STATE_ORDER:
                b_circ = build_xtop006_circuit(
                    arm=arm,
                    theta=theta,
                    state_label=st,
                    theta_name=t_name,
                    backend=be,
                )
                built_list.append(b_circ)
                qcs_marked.append(_to_qiskit_with_memory_marker(b_circ))

                if arm == "P0":
                    v_info = verify_p0_compiled_against_kraus(theta, st, backend=be)
                    max_noiseless_err = max(
                        max_noiseless_err, v_info["max_unnormalized_rho_frobenius_error"]
                    )
                else:
                    v_info = verify_e1_compiled_against_kraus(b_circ)
                    max_noiseless_err = max(
                        max_noiseless_err,
                        v_info["prob_error_vs_kraus"],
                        v_info["fidelity_error_vs_kraus"],
                    )

                cnt = compiled_gate_counts(b_circ.compiled_tket)
                hqc_vals = compute_circuit_hqc(cnt, shots_hqc)
                hqc_phys_dec = local_hqc(b_circ.compiled_tket, shots_hqc)
                tk_sha = tket_sha256(b_circ.compiled_tket)
                compiled_bundle.append(
                    {"name": b_circ.compiled_tket.name, "tket": b_circ.compiled_tket.to_dict()}
                )
                circuit_records.append(
                    {
                        "name": b_circ.compiled_tket.name,
                        "arm": arm,
                        "arm_label": ARM_LABELS[arm],
                        "theta_name": t_name,
                        "theta_rad": f"{theta:.8f}",
                        "state": st,
                        "readout_basis": b_circ.readout_basis,
                        "qubits": cnt["qubits"],
                        "N_1q": cnt["N_1q"],
                        "N_1q_physical": cnt["N_1q_physical"],
                        "N_1q_virtual_rz": cnt["N_1q_virtual_rz"],
                        "N_2q": cnt["N_2q"],
                        "N_2q_zzphase": cnt["N_2q_zzphase"],
                        "N_2q_zzmax": cnt["N_2q_zzmax"],
                        "N_meas": cnt["N_meas"],
                        "N_reset": cnt["N_reset"],
                        "N_m": cnt["N_m"],
                        "routing_swaps": cnt["routing_swaps"],
                        "shots": shots_hqc,
                        "hqc_conservative": hqc_vals["conservative"],
                        "hqc_physical": str(hqc_phys_dec),
                        "tket_sha256": tk_sha,
                    }
                )
                tot = arm_totals[arm]
                tot["n_circuits"] += 1
                for k in (
                    "N_1q",
                    "N_1q_physical",
                    "N_1q_virtual_rz",
                    "N_2q",
                    "N_meas",
                    "N_reset",
                    "N_m",
                    "routing_swaps",
                ):
                    tot[k] += cnt[k]
                tot["hqc_total_500_conservative"] += float(hqc_vals["conservative"])
                tot["hqc_total_500_physical"] += hqc_phys_dec

            res_mem = sim_mem.run(
                qcs_marked,
                shots=shots_sim,
                seed_simulator=seed + 100 * t_idx + (0 if arm == "P0" else 50),
            ).result()
            dec_mem = [
                _decode_shot_counts(res_mem.get_counts(k), built_list[k])
                for k in range(len(built_list))
            ]
            res_go = sim_gate_only.run(
                qcs_marked,
                shots=shots_sim,
                seed_simulator=seed + 1000 + 100 * t_idx + (0 if arm == "P0" else 50),
            ).result()
            dec_go = [
                _decode_shot_counts(res_go.get_counts(k), built_list[k])
                for k in range(len(built_list))
            ]

            per_arm_theta[arm] = {
                "arm": arm,
                "label": ARM_LABELS[arm],
                "noiseless_max_error_vs_kraus": f"{max_noiseless_err:.3e}",
                "h2_with_qccd_memory_dephasing": {
                    "p_keep_avg": f"{float(np.mean([d['p_keep'] for d in dec_mem])):.6f}",
                    "p_phi_plus_avg": f"{float(np.mean([d['p_phi_plus'] for d in dec_mem])):.6f}",
                    "fidelity_6state_avg": f"{float(np.mean([d['fidelity'] for d in dec_mem])):.6f}",
                    "verified_scrambling_witness": f"{float(np.mean([d['fidelity'] for d in dec_mem]) - 0.5):.6f}",
                },
                "h2_gate_spam_only_p_mem_0": {
                    "p_keep_avg": f"{float(np.mean([d['p_keep'] for d in dec_go])):.6f}",
                    "p_phi_plus_avg": f"{float(np.mean([d['p_phi_plus'] for d in dec_go])):.6f}",
                    "fidelity_6state_avg": f"{float(np.mean([d['fidelity'] for d in dec_go])):.6f}",
                },
            }

        e1_f = float(per_arm_theta["E1"]["h2_with_qccd_memory_dephasing"]["fidelity_6state_avg"])
        p0_f = float(per_arm_theta["P0"]["h2_with_qccd_memory_dephasing"]["fidelity_6state_avg"])
        theta_cells.append(
            {
                "theta_name": t_name,
                "theta_rad": f"{theta:.8f}",
                "analytical_kraus": {
                    "p_phi_plus": f"{kraus_summary['p_success']:.6f}",
                    "fidelity_6state_avg": f"{kraus_summary['fidelity_6state_avg']:.6f}",
                    "entanglement_fidelity": f"{kraus_summary['entanglement_fidelity']:.6f}",
                    "yk_product_p_times_f_ent": f"{kraus_summary['yk_product_p_times_f_ent']:.6f}",
                    "max_decoupling_deviation": f"{kraus_summary['max_decoupling_deviation']:.3e}",
                    "is_non_clifford": kraus_summary["is_non_clifford"],
                    "stabilizer_renyi2_magic_bits": f"{kraus_summary['stabilizer_renyi2_magic_bits']:.6f}",
                    "cptp_completeness_frobenius_error": f"{cptp_info['completeness_frobenius_error']:.3e}",
                },
                "arms": per_arm_theta,
                "encoded_422_advantage_delta_F": f"{(e1_f - p0_f):.6f}",
            }
        )

    disc = kc.scrambling_vs_decoherence_discriminator()
    disc_serialized = {
        "coherent_noiseless": [
            {k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()}
            for row in disc["coherent_noiseless"]
        ],
        "coherent_noisy": [
            {k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()}
            for row in disc["coherent_noisy"]
        ],
        "pure_decoherence_no_scrambling": [
            {k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()}
            for row in disc["pure_decoherence_no_scrambling"]
        ],
        "separation_summary": {
            "pure_decoherence_max_fidelity": f"{disc['separation_summary']['pure_decoherence_max_fidelity']:.6f}",
            "coherent_noisy_peak_fidelity": f"{disc['separation_summary']['coherent_noisy_peak_fidelity']:.6f}",
            "fidelity_gap_over_decoherence_ceiling": f"{disc['separation_summary']['fidelity_gap_over_decoherence_ceiling']:.6f}",
            "separates_scrambling_from_decoherence": disc["separation_summary"][
                "separates_scrambling_from_decoherence"
            ],
        },
    }

    bundle_bytes = rfc8785.dumps(compiled_bundle)
    bundle_path = artifacts_dir / CIRCUITS_BUNDLE_FILENAME
    bundle_path.write_bytes(bundle_bytes)
    bundle_sha = hashlib.sha256(bundle_bytes).hexdigest()

    arm_summaries_clean: Dict[str, Any] = {}
    total_cons_all = 0.0
    total_phys_all = Decimal("0")
    for arm, tot in arm_totals.items():
        total_cons_all += tot["hqc_total_500_conservative"]
        total_phys_all += tot["hqc_total_500_physical"]
        arm_summaries_clean[arm] = {
            "arm": arm,
            "label": tot["label"],
            "n_circuits": tot["n_circuits"],
            "qubits": tot["qubits"],
            "totals_30_circuits": {
                k: int(tot[k])
                for k in (
                    "N_1q",
                    "N_1q_physical",
                    "N_1q_virtual_rz",
                    "N_2q",
                    "N_meas",
                    "N_reset",
                    "N_m",
                    "routing_swaps",
                )
            },
            "hqc_30_circuits_500_conservative": f"{tot['hqc_total_500_conservative']:.4f}",
            "hqc_30_circuits_500_physical": str(tot["hqc_total_500_physical"]),
        }

    comp_batch_sha = batch_sha256([r["tket_sha256"] for r in circuit_records])
    manifest_obj: Dict[str, Any] = {
        "schema": "osiris.xtop006.h2_scrambling_manifest/1",
        "experiment_id": "OSIRIS-XTOP-006",
        "title": "Tunable Non-Clifford Scrambling & [[4,2,2]] Codespace QCCD Recovery on Quantinuum H2-1",
        "canonicalization": "RFC8785-JCS",
        "evidence_class": "SIMULATED",
        "provider_job_id": None,
        "hardware_executed": False,
        "compiler_backend": "pytket.extensions.quantinuum.QuantinuumAPIOffline",
        "compile_target": "H2-1",
        "optimisation_level": 2,
        "shots_sim": shots_sim,
        "shots_hqc_reference": shots_hqc,
        "seed": seed,
        "n_circuits": len(circuit_records),
        "hqc_budget_all_60_circuits": int(math.ceil(total_cons_all)),
        "hqc_total_all_60_conservative": f"{total_cons_all:.4f}",
        "hqc_total_all_60_physical": str(total_phys_all),
        "noise_model": {
            "p_1q": f"{p_1q:.6f}",
            "p_2q": f"{p_2q:.6f}",
            "p_spam": f"{p_spam:.6f}",
            "p_mem_qccd_storage_zone": f"{p_mem:.6f}",
            "memory_channel": "0.70*Z + 0.15*X + 0.15*Y Pauli dephasing-dominant channel on idle storage qubits",
        },
        "compiled_batch_sha256": comp_batch_sha,
        "circuits_bundle_relpath": CIRCUITS_BUNDLE_FILENAME,
        "circuits_bundle_sha256": bundle_sha,
        "arm_summaries": arm_summaries_clean,
        "theta_sweep_results": theta_cells,
        "scrambling_vs_decoherence_discriminator": disc_serialized,
        "circuits": circuit_records,
    }

    manifest_bytes = rfc8785.dumps(manifest_obj)
    manifest_path = artifacts_dir / MANIFEST_FILENAME
    manifest_path.write_bytes(manifest_bytes)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    sha_lines = [
        f"{bundle_sha}  {CIRCUITS_BUNDLE_FILENAME}",
        f"{manifest_sha}  {MANIFEST_FILENAME}",
    ]
    (artifacts_dir / SHA256SUMS_FILENAME).write_text(
        "\n".join(sorted(sha_lines, key=lambda s: s.split("  ")[1])) + "\n",
        encoding="utf-8",
    )
    return manifest_obj


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    ap.add_argument("--shots-sim", type=int, default=4000)
    ap.add_argument("--shots-hqc", type=int, default=500)
    ap.add_argument("--p-mem", type=float, default=0.045)
    ap.add_argument("--seed", type=int, default=20261123)
    args = ap.parse_args(list(argv) if argv is not None else None)

    manifest = run_xtop006_experiment_and_write_manifest(
        artifacts_dir=args.artifacts_dir,
        shots_sim=args.shots_sim,
        shots_hqc=args.shots_hqc,
        p_mem=args.p_mem,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "experiment_id": manifest["experiment_id"],
                "evidence_class": manifest["evidence_class"],
                "provider_job_id": manifest["provider_job_id"],
                "n_circuits": manifest["n_circuits"],
                "hqc_budget_all_60_circuits": manifest["hqc_budget_all_60_circuits"],
                "hqc_total_all_60_conservative": manifest["hqc_total_all_60_conservative"],
                "hqc_total_all_60_physical": manifest["hqc_total_all_60_physical"],
                "compiled_batch_sha256": manifest["compiled_batch_sha256"],
                "theta_sweep_summary": [
                    {
                        "theta": row["theta_name"],
                        "analytical_F": row["analytical_kraus"]["fidelity_6state_avg"],
                        "analytical_P_Phi+": row["analytical_kraus"]["p_phi_plus"],
                        "P0_noisy_F": row["arms"]["P0"]["h2_with_qccd_memory_dephasing"][
                            "fidelity_6state_avg"
                        ],
                        "E1_noisy_F": row["arms"]["E1"]["h2_with_qccd_memory_dephasing"][
                            "fidelity_6state_avg"
                        ],
                        "delta_F_E1_minus_P0": row["encoded_422_advantage_delta_F"],
                    }
                    for row in manifest["theta_sweep_results"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
