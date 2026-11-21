"""QPU budget for the OSIRIS-XTOP-005 arm matrix, from real compiled circuits.

What is exact here and what is not:
  * Bare arms (A, B): the HP-004 circuits themselves (experiments/hp004), compiled for Quantinuum H2
    offline with pytket-quantinuum (no credentials needed) and counted.
  * Encoded arms (C, D): the code-space layer this repo implements (xtop005/code422.py):
    verified logical Bell-pair prep for R_L/M_L, flagged X and Z checks on both blocks, logical
    readout, compiled and counted.
    THE CODE-SPACE SCRAMBLER AND DECODER ARE NOT DEFINED YET (see README), so encoded-arm totals are
    LOWER BOUNDS: bare scrambler+decoder counts are added as a stand-in.
  * IBM seconds: the empirical rate of this account's Heron r2 jobs. MAP-04 main jobs: 122,880 shots in 36 s
    (0.29 ms/shot, static). QEC-001: 393,216 dynamic-circuit shots in 109 s (0.28 ms/shot). We use
    0.30 ms/shot plus 3 s per job. This is an estimate, not an IBM formula.
  * Quantinuum HQC, H-series published formula per job:  HQC = 5 + C (N1q + 10 N2q + 5 Nm) / 5000,
    where C = shots, N1q/N2q = 1-/2-qubit gates, Nm = state preparations + measurements. Reported with
    Nm = measurements + resets + initial preparations (one per qubit). Rz gates are counted as 1-qubit
    gates in N1q (the conservative reading).
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

from xtop005 import code422 as c

ROOT = Path(__file__).resolve().parents[1]
IBM_S_PER_SHOT = 0.30e-3
IBM_S_PER_JOB = 3.0
BASES = list(product("XYZ", repeat=2))
DEPTHS = (1, 2, 3)


def hqc(n1: int, n2: int, nm: int, shots: int) -> float:
    return 5 + shots * (n1 + 10 * n2 + 5 * nm) / 5000


def _tk_counts(circ) -> dict:
    from pytket import OpType
    from pytket.extensions.quantinuum import QuantinuumAPIOffline, QuantinuumBackend
    be = QuantinuumBackend("H2-1", api_handler=QuantinuumAPIOffline())
    comp = be.get_compiled_circuit(circ, optimisation_level=2)
    n1 = n2 = nm = nr = 0
    for cmd in comp.get_commands():
        t = cmd.op.type
        if t == OpType.Measure:
            nm += 1
        elif t == OpType.Reset:
            nr += 1
        elif t in (OpType.Barrier,):
            continue
        elif len(cmd.qubits) == 1:
            n1 += 1
        elif len(cmd.qubits) == 2:
            n2 += 1
    return {"n1": n1, "n2": n2, "measure": nm, "reset": nr, "qubits": comp.n_qubits,
            "nm_for_hqc": nm + nr + comp.n_qubits}


def qiskit_to_tk(qc):
    """Minimal converter for the gates HP-004 uses (cz, cx, sx, x, h, s, sdg, rz, measure, barrier)."""
    import math
    from pytket import Circuit, OpType
    tk = Circuit(qc.num_qubits, qc.num_clbits)
    simple = {"cz": OpType.CZ, "cx": OpType.CX, "sx": OpType.SX, "x": OpType.X, "h": OpType.H,
              "s": OpType.S, "sdg": OpType.Sdg, "z": OpType.Z, "y": OpType.Y}
    for inst in qc.data:
        name = inst.operation.name
        qs = [qc.find_bit(q).index for q in inst.qubits]
        if name == "barrier":
            continue
        if name == "measure":
            tk.Measure(qs[0], qc.find_bit(inst.clbits[0]).index)
        elif name == "rz":
            tk.Rz(float(inst.operation.params[0]) / math.pi, qs[0])
        elif name in simple:
            tk.add_gate(simple[name], qs)
        else:
            raise ValueError(f"unsupported gate {name}")
    return tk


def bare_hp004_counts() -> dict:
    """HP-004 'Arm B' (2-photon decoder) for each of the 9 bases -> Quantinuum counts."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("hp004", ROOT / "experiments/hp004/hp004_joint_stabilizer.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    out = {}
    for b1, b2 in BASES:
        qc = m.build_hp004_circuit("Arm B", b1, b2)
        out[b1 + b2] = {"ibm_cz": sum(1 for i in qc.data if i.operation.name == "cz"), **_tk_counts(qiskit_to_tk(qc))}
    return out


def encoded_layer_program(b1: str, b2: str) -> c.Program:
    """R_L = q0-3, M_L = q4-7, ancilla 8, flag 9: verified Bell prep, 4 flagged checks, logical readout."""
    p = c.Program(10)
    c.logical_bell_pairs(p, (0, 1, 2, 3), (4, 5, 6, 7), verify_anc=8)
    for blk in ((0, 1, 2, 3), (4, 5, 6, 7)):
        c.check(p, blk, "X", 8, 9)
        c.check(p, blk, "Z", 8, 9)
    c.measure_logical(p, (0, 1, 2, 3), b1, b2)
    c.measure_logical(p, (4, 5, 6, 7), b1, b2)
    return p


def main() -> None:
    try:
        bare = bare_hp004_counts()
        bare_ok = True
    except ImportError:
        bare, bare_ok = {}, False
    enc = {}
    for b1, b2 in BASES:
        p = encoded_layer_program(b1, b2)
        enc[b1 + b2] = {"builder_counts": p.counts(), **_tk_counts(p.to_pytket())}
    avg = lambda d, k: sum(v[k] for v in d.values()) / len(d)

    def arm_cost(n1, n2, nm, shots):
        return hqc(n1, n2, nm, shots)
    rows = []
    for shots_ibm, shots_q in ((4096, 1000), (2048, 500)):
        for arm in ("A", "B", "C", "D"):
            per_basis = []
            for k in enc:
                b = bare.get(k, {"n1": 0, "n2": 0, "nm_for_hqc": 0})
                e = enc[k]
                if arm in ("A", "B"):
                    n1, n2, nm = b["n1"], b["n2"], b["nm_for_hqc"]
                else:  # encoded: code-space layer + bare scrambler/decoder as a lower-bound stand-in
                    n1, n2, nm = e["n1"] + b["n1"], e["n2"] + b["n2"], e["nm_for_hqc"]
                per_basis.append(arm_cost(n1, n2, nm, shots_q))
            rows.append({"shots_ibm": shots_ibm, "shots_quantinuum": shots_q, "arm": arm,
                         "hqc_per_depth_9_bases": round(sum(per_basis), 1),
                         "hqc_all_3_depths": round(3 * sum(per_basis), 1)})
    n_pubs = len(BASES) * 4 * len(DEPTHS)
    ibm = {}
    for shots in (4096, 2048):
        per_backend = n_pubs * shots * IBM_S_PER_SHOT + IBM_S_PER_JOB
        ibm[str(shots)] = {"pubs_per_backend": n_pubs, "seconds_per_backend": round(per_backend, 1),
                           "seconds_two_backends": round(2 * per_backend, 1)}
    out = {"bare_hp004_compiled_h2": bare if bare_ok else "pytket-qiskit not installed",
           "encoded_layer_compiled_h2": enc,
           "encoded_layer_avg": {k: round(avg(enc, k), 1) for k in ("n1", "n2", "measure", "reset", "qubits")},
           "hqc_rows": rows, "ibm_seconds": ibm,
           "caveat": "encoded-arm totals are lower bounds until the code-space scrambler/decoder exist"}
    (ROOT / "xtop005" / "budget.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("encoded_layer_avg", "hqc_rows", "ibm_seconds")}, indent=1))
    if bare_ok:
        print("bare HP-004 Arm B (H2-compiled, avg):", {k: round(avg(bare, k), 1) for k in ("ibm_cz", "n1", "n2", "measure")})


if __name__ == "__main__":
    main()
