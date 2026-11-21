#!/usr/bin/env python3
"""OSIRIS-XTOP-005, Quantinuum H2 design: code-space Hayden-Preskill recovery with all-to-all connectivity.

Why a separate design: on IBM heavy-hex (docs/xtop005/qpu, prereg 10.5281/zenodo.24425416) only the idle reference
R could be encoded cheaply, because transversal gates between [[4,2,2]] blocks need connectivity heavy-hex lacks.
H2's all-to-all gates make every transversal gate a native two-qubit gate and give cheap mid-circuit measurement
and reset, so the full design becomes possible. The DD factor is dropped: idling on H2 is not a user-timed delay.

Arms (same logical program; depths L = 1, 3; 9 readout settings):
  Q0  BARE         8 qubits
  Q1  ENC_R        R encoded (the IBM heavy-hex arm, for cross-platform comparison), 10 qubits
  Q2  ENC_ALL      all four blocks encoded, verified |00>/|++> preparation, 17 qubits
  Q3  ENC_ALL_CHK  Q2 plus flagged weight-4 X/Z checks on R and E after decoding, 18 qubits

This module does not talk to Quantinuum (no credentials on this machine). It:
  * builds every circuit and checks it exactly (noiseless: F = 1, no shot rejected),
  * simulates it with an H2-like Pauli noise model (published H2-1 spec-sheet magnitudes; NOT Quantinuum's
    emulator, no memory/transport error model),
  * compiles it offline for H2-1 with pytket-quantinuum and computes the HQC cost,
  * writes the compiled circuits (pytket JSON) with SHA-256 so the exact submission can be inspected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from xtop005 import osiris_xtop_005 as x

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "xtop005" / "quantinuum"
ARMS_Q = {
    "Q0": ("BARE", dict(arm="A")),
    "Q1": ("ENC_R", dict(arm="C", mode="r")),
    "Q2": ("ENC_ALL", dict(arm="C", mode="full", verify=True, checks=False)),
    "Q3": ("ENC_ALL_CHK", dict(arm="C", mode="full", verify=True, checks=True)),
}
DEPTHS = (1, 3)
# H2-1 published magnitudes (approximate): 1q infidelity 3e-5, 2q 1e-3, SPAM 1e-3. Pauli channels only.
P1, P2, P_MEAS, P_RESET = 3e-5, 1e-3, 1e-3, 1e-3


def build_q(qarm: str, L: int, setting: Tuple[str, str]):
    spec = dict(ARMS_Q[qarm][1])
    arm = spec.pop("arm")
    mode = spec.pop("mode", "r")
    b = x.build(arm, L, setting, 0.0, mode, **spec)
    qc = x.to_qiskit(b)
    qc.data = [i for i in qc.data if i.operation.name not in ("delay", "barrier")]
    qc.name = f"xtop005q_{qarm}_L{L}_{''.join(setting)}"
    return b, qc


def h2_noise():
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, pauli_error
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(P1, 1), ["h", "s", "sdg", "x", "y", "z"])
    nm.add_all_qubit_quantum_error(depolarizing_error(P2, 2), ["cx", "cz"])
    nm.add_all_qubit_quantum_error(pauli_error([("X", P_RESET), ("I", 1 - P_RESET)]), ["reset"])
    nm.add_all_qubit_readout_error(ReadoutError([[1 - P_MEAS, P_MEAS], [P_MEAS, 1 - P_MEAS]]))
    return nm


def tk_compile(qc):
    from pytket.extensions.qiskit import qiskit_to_tk
    from pytket.extensions.quantinuum import QuantinuumAPIOffline, QuantinuumBackend
    be = QuantinuumBackend("H2-1", api_handler=QuantinuumAPIOffline())
    return be.get_compiled_circuit(qiskit_to_tk(qc), optimisation_level=2)


def tk_counts(tk) -> Dict[str, int]:
    from pytket import OpType
    n1 = n2 = nm = nr = 0
    for cmd in tk.get_commands():
        t = cmd.op.type
        if t == OpType.Measure:
            nm += 1
        elif t == OpType.Reset:
            nr += 1
        elif t == OpType.Barrier:
            continue
        elif len(cmd.qubits) == 1:
            n1 += 1
        elif len(cmd.qubits) == 2:
            n2 += 1
    return {"n1": n1, "n2": n2, "measure": nm, "reset": nr, "qubits": tk.n_qubits}


def hqc(c: Dict[str, int], shots: int) -> float:
    """H-series formula: HQC = 5 + C (N1q + 10 N2q + 5 Nm) / 5000; Nm counts preparations + measurements
    (reported conservatively as measurements + resets + one initial preparation per qubit)."""
    nm = c["measure"] + c["reset"] + c["qubits"]
    return 5 + shots * (c["n1"] + 10 * c["n2"] + 5 * nm) / 5000


def run(shots_sim: int = 2000, shots_hqc=(200, 500, 1000), seed: int = 20261121) -> Dict:
    from qiskit_aer import AerSimulator
    OUT.mkdir(parents=True, exist_ok=True)
    ideal = AerSimulator(method="stabilizer")
    noisy = AerSimulator(method="stabilizer", noise_model=h2_noise())
    rng = np.random.default_rng(seed)
    cells, compiled = [], []
    for L in DEPTHS:
        frame = x.logical_frame(L)
        for qarm, (label, _) in ARMS_Q.items():
            items = [build_q(qarm, L, st) for st in x.SETTINGS]
            res0 = ideal.run([q for _, q in items], shots=128, seed_simulator=seed).result()
            per0 = {"".join(b.setting): x.correlators(b, res0.get_counts(k)) for k, (b, _) in enumerate(items)}
            f0 = x.fidelity(per0, frame, "raw")
            if abs(f0 - 1) > 1e-9 or any(v["acc"]["n"] != v["raw"]["n"] for v in per0.values()):
                raise AssertionError(f"Stage 0 failed {qarm} L={L}")
            res = noisy.run([q for _, q in items], shots=shots_sim, seed_simulator=seed + L).result()
            per = {"".join(b.setting): x.correlators(b, res.get_counts(k)) for k, (b, _) in enumerate(items)}
            enc = qarm != "Q0"
            which = "acc" if enc else "raw"
            tks = []
            for b, q in items:
                tk = tk_compile(q)
                tks.append(tk)
                compiled.append({"name": q.name, "tket": tk.to_dict()})
            counts = [tk_counts(t) for t in tks]
            cell = {"arm": qarm, "label": label, "depth": L, "stage0_F": f0, "qubits": items[0][1].num_qubits,
                    "F_raw": x.fidelity(per, frame, "raw"),
                    "F_used": x.fidelity(per, frame, which),
                    "F_used_ci99": x.bootstrap_ci(per, frame, which, rng),
                    "p_keep": sum(v["acc"]["n"] for v in per.values()) / sum(v["raw"]["n"] for v in per.values()),
                    "h2_counts_avg": {k: round(float(np.mean([c_[k] for c_ in counts])), 1) for k in counts[0]},
                    "hqc_9_settings": {str(s): round(sum(hqc(c_, s) for c_ in counts), 1) for s in shots_hqc}}
            cells.append(cell)
            print(json.dumps(x._canon({k: v for k, v in cell.items()})), flush=True)
    totals = {str(s): round(sum(c_["hqc_9_settings"][str(s)] for c_ in cells), 1) for s in shots_hqc}
    blob = json.dumps(compiled, sort_keys=True).encode()
    (OUT / "circuits_h2_1_tket.json").write_bytes(blob)
    report = {"noise_model": {"p1": P1, "p2": P2, "p_meas": P_MEAS, "p_reset": P_RESET,
                              "note": "H2-like Pauli noise from published magnitudes; not Quantinuum's emulator"},
              "shots_sim": shots_sim, "seed": seed, "cells": cells, "hqc_total_all_cells": totals,
              "circuits_sha256": hashlib.sha256(blob).hexdigest(), "n_circuits": len(compiled)}
    body = x.canonical_json(report)
    (OUT / "xtop005_quantinuum_design.json").write_text(body + "\n")
    print("HQC totals", totals, "circuits", len(compiled), "sha", report["circuits_sha256"][:16])
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=2000)
    a = ap.parse_args()
    run(a.shots)
