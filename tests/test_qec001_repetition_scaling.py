"""Unit and threshold-scaling tests for OSIRIS-QEC-001 (`qec001_repetition_scaling.py`)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

from qiskit_aer import AerSimulator  # noqa: E402
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error  # noqa: E402

_MOD_PATH = Path(__file__).resolve().parent.parent / "experiments" / "qec001" / "qec001_repetition_scaling.py"
_SPEC = importlib.util.spec_from_file_location("qec001_repetition_scaling", _MOD_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_QEC = importlib.util.module_from_spec(_SPEC)
sys.modules["qec001_repetition_scaling"] = _QEC
_SPEC.loader.exec_module(_QEC)


@pytest.mark.parametrize("code_basis", ["bit_flip_z", "phase_flip_x"])
@pytest.mark.parametrize("logical_state", [0, 1])
@pytest.mark.parametrize("distance", [3, 5, 7])
def test_noiseless_multi_round_syndrome_zero_defects_and_exact_logical_readout(
    code_basis: str, logical_state: int, distance: int
) -> None:
    qc = _QEC.build_repetition_memory_circuit(
        distance=distance,
        rounds=3,
        code_basis=code_basis,
        logical_state=logical_state,
    )
    sim = AerSimulator()
    counts = sim.run(qc, shots=256).result().get_counts()
    stats = _QEC.evaluate_repetition_counts(counts, distance=distance, rounds=3, logical_state=logical_state)
    assert stats["mwpm_logical_errors"] == 0.0
    assert stats["p_logical_majority"] == 0.0
    assert stats["mean_defect_probability"] == 0.0


def test_mwpm_corrects_all_weight_t_data_and_measurement_faults() -> None:
    # For d=5 (corrects all weight <= 2 data faults) and d=7 (corrects all weight <= 3 data faults)
    for d, max_wt in [(3, 1), (5, 2), (7, 3)]:
        dec = _QEC.build_mwpm_decoder(distance=d, rounds=3)
        syn_zero = np.zeros((3, d - 1), dtype=np.uint8)
        # Exhaustively test all single, double, (and triple for d=7) data bit-flips on |0_L> and |1_L>
        from itertools import combinations

        for logical in (0, 1):
            base = np.full(d, logical, dtype=np.uint8)
            for wt in range(0, max_wt + 1):
                for flipped_sites in combinations(range(d), wt):
                    dat = base.copy()
                    for idx in flipped_sites:
                        dat[idx] ^= 1
                    assert _QEC.decode_shot_mwpm(dec, syn_zero, dat) == logical

        # Pure mid-circuit ancilla measurement faults must never corrupt the logical observable
        for r in range(3):
            for j in range(d - 1):
                syn = np.zeros((3, d - 1), dtype=np.uint8)
                syn[r, j] = 1
                assert _QEC.decode_shot_mwpm(dec, syn, np.zeros(d, dtype=np.uint8)) == 0
                assert _QEC.decode_shot_mwpm(dec, syn, np.ones(d, dtype=np.uint8)) == 1


def test_subthreshold_exponential_logical_error_suppression_across_d3_d5_d7() -> None:
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(0.002, 1), ["x", "sx", "h"])
    noise.add_all_qubit_quantum_error(depolarizing_error(0.008, 2), ["cx"])
    noise.add_all_qubit_quantum_error(depolarizing_error(0.008, 1), ["reset"])
    noise.add_all_qubit_readout_error(ReadoutError([[0.99, 0.01], [0.01, 0.99]]))

    sim = AerSimulator(noise_model=noise, seed_simulator=20261116)
    eps_by_d = {}
    for d in (3, 5, 7):
        qc = _QEC.build_repetition_memory_circuit(distance=d, rounds=5, code_basis="bit_flip_z", logical_state=0)
        counts = sim.run(qc, shots=4096).result().get_counts()
        dec = _QEC.build_mwpm_decoder(distance=d, rounds=5, p_data=0.012, p_meas=0.018, p_hook=0.004)
        res = _QEC.evaluate_repetition_counts(counts, distance=d, rounds=5, logical_state=0, decoder=dec)
        eps_by_d[d] = res["epsilon_per_cycle"]

    lambda_35 = eps_by_d[3] / max(eps_by_d[5], 1e-6)
    lambda_57 = eps_by_d[5] / max(eps_by_d[7], 1e-6)
    assert eps_by_d[3] > eps_by_d[5] > eps_by_d[7]
    assert lambda_35 > 1.5
    assert lambda_57 > 1.5
