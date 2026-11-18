"""CORRELATED-NOISE-01 amendment A-3: pulse timing, model reductions, local-null symmetry, selection
gate, split, and the amendment record."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_ibm_runtime")  # CI does not install qiskit-ibm-runtime

from qiskit import qpy  # noqa: E402

from phase2.correlated_noise_01 import correlated_noise_models as P  # noqa: E402
from phase2.correlated_noise_01.simulate_correlated_noise import (  # noqa: E402
    MODEL_BOUNDS,
    _instantiate_model_from_theta,
)
from phase3 import cn01_amendment_a3 as a3  # noqa: E402
from phase3.execute_and_analyze_live import _build_cn01_stage1_probe_conditions  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs" / "phase3"
SUITE = _build_cn01_stage1_probe_conditions()
BY_ID = {c.condition_id: c for c in SUITE}
FIT_SUITE = [c for c in SUITE if not c.condition_id.startswith("ramsey_")]


def _lock() -> Dict:
    return json.loads((DOCS / "correlated_noise_01_live_prereg_lock.json").read_text())


def _dd_window(qc, wires: Sequence[int], x_dt: int = 9) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """Pulse start ticks and window length per wire, between the first and second barrier."""
    phase = 0
    t = {w: 0 for w in wires}
    pulses: Dict[int, List[int]] = {w: [] for w in wires}
    for inst in qc.data:
        name = inst.operation.name
        if name == "barrier":
            phase += 1
            if phase == 2:
                break
            continue
        if phase != 1:
            continue
        for q in (qc.find_bit(b).index for b in inst.qubits):
            if q not in t:
                continue
            if name == "delay":
                t[q] += int(inst.operation.duration)
            elif name == "x":
                pulses[q].append(t[q])
                t[q] += x_dt
            elif name != "rz":
                raise AssertionError(f"unexpected {name} in the DD window of {qc.name}")
    return pulses, t


def test_hw_pulse_placement_matches_the_locked_circuits() -> None:
    lock = _lock()
    with (DOCS / "correlated_noise_01_live_pubs.qpy").open("rb") as f:
        circuits = qpy.load(f)
    checked = 0
    for rec, qc in zip(lock["pub_schedule"], circuits):
        if rec["block"] != 0 or rec["kind"] == "2q_spam" or (rec["kind"] == "2q_bell" and rec["basis"] != "ZZ"):
            continue
        for slot in lock["coupler_cohort"]:
            u, v = slot["coupler"]
            wires = [u] if rec["kind"] == "1q" else [u, v]
            pulses, length = _dd_window(qc, wires)
            assert length[u] == rec["total_dt"], rec["condition_id"]
            assert pulses[u] == a3.hw_pulse_starts_dt(rec["total_dt"], rec["n_dd"], 0), rec["condition_id"]
            if rec["kind"] != "1q":
                assert pulses[v] == a3.hw_pulse_starts_dt(rec["total_dt"], rec["n_dd"], rec["offset_dt"]), rec["condition_id"]
        checked += 1
    assert checked == 12 + 12  # 12 single-qubit conditions, 12 Bell conditions (incl. the A/A sentinel)


def test_kernel_integrals_match_phase2() -> None:
    h3 = P.H3NonMarkovianMemoryModel()
    for c in SUITE:
        a, b, su, sv = P.piecewise_toggling_intervals(c.total_time_s, c.n_dd_pulses, c.stagger_offset_s)
        ours = a3.ou_kernel_integrals(a, b, su, sv, h3.tau_c_s, h3.f_bath_hz)
        theirs = h3._exact_piecewise_kernel_integrals(c.total_time_s, c.n_dd_pulses, c.stagger_offset_s)
        np.testing.assert_allclose(ours, theirs, rtol=1e-12, atol=0.0)


def test_hw_models_reduce_to_phase2_under_phase2_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(a3, "hw_toggling_intervals", P.piecewise_toggling_intervals)
    monkeypatch.setattr(a3, "hw_pair_exposure", P.exact_pair_toggling_exposure)
    monkeypatch.setattr(a3, "hw_single_exposure", lambda T, n: P.exact_single_toggling_exposure(T, n, 0.0))
    for key in ("H0", "H1", "H2", "H3", "Apparatus"):
        theta = MODEL_BOUNDS[key][0]
        np.testing.assert_allclose(
            a3.instantiate_a3(key, theta).predict_suite(SUITE),
            _instantiate_model_from_theta(key, theta).predict_suite(SUITE),
            rtol=0.0, atol=1e-13, err_msg=key,
        )


def test_h0_1f_is_the_tau_average_of_local_ou_processes() -> None:
    amp, gw = 60e3, 700.0
    m = a3.instantiate_a3("H0_1f", [180e-6, 160e-6, gw, gw, amp, 0.97])
    for c in SUITE:
        chi_u, chi_v, chi_x = m._cumulants(c)
        ou = [a3.instantiate_a3("H3_local", [180e-6, 160e-6, 0.0, 0.0, amp, tau, 0.97])._cumulants(c)
              for tau in a3.TAU_1F_GRID_S]
        assert chi_x == 0.0
        assert chi_u == pytest.approx(gw * c.total_time_s + np.mean([o[0] for o in ou]), rel=1e-12, abs=1e-15)
        assert chi_v == pytest.approx(gw * c.total_time_s + np.mean([o[1] for o in ou]), rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("key", a3.LOCAL_KEYS)
def test_local_models_predict_identical_phi_and_psi(key: str) -> None:
    """No local noise model can produce the Phi+/Psi+ (DFS) asymmetry under DD."""
    m = a3.instantiate_a3(key, a3.A3_BOUNDS[key][0])
    for d in (0, 256):
        assert m.predict(BY_ID[f"phi_dd16_d{d}ns"]) == pytest.approx(m.predict(BY_ID[f"psi_dd16_d{d}ns"]), abs=1e-14)


@pytest.mark.parametrize("key", ("H2", "H3"))
def test_correlated_models_predict_positive_dfs_asymmetry(key: str) -> None:
    m = a3.instantiate_a3(key, a3.A3_BOUNDS[key][0])
    p = {cid: m.predict(BY_ID[cid]) for cid in ("phi_dd16_d0ns", "phi_dd16_d256ns", "psi_dd16_d0ns", "psi_dd16_d256ns")}
    assert (p["phi_dd16_d256ns"] - p["phi_dd16_d0ns"]) - (p["psi_dd16_d256ns"] - p["psi_dd16_d0ns"]) > 0.0


def test_selection_gate_on_noiseless_truths() -> None:
    cov = np.eye(len(FIT_SUITE)) * 0.002 ** 2
    local = a3.instantiate_a3("H0_1f", [180e-6, 160e-6, 800.0, 800.0, 60e3, 0.97])
    res = a3.fit_models_a3(FIT_SUITE, local.predict_suite(FIT_SUITE), cov)
    assert res["selected_model"] in a3.LOCAL_KEYS
    assert res["fits"]["H0"]["bic"] - res["fits"][res["best_local_model"]]["bic"] >= a3.DELTA_BIC_GATE

    correlated = a3.instantiate_a3("H2", [180e-6, 160e-6, 4000.0, 4000.0, 0.97, 0.6, 2.0])
    res = a3.fit_models_a3(FIT_SUITE, correlated.predict_suite(FIT_SUITE), cov)
    assert res["selected_model"] in ("H2", "H3")
    assert res["delta_bic_over_best_local"] >= a3.DELTA_BIC_GATE


def test_random_split_is_deterministic_and_seeded_by_the_locked_qpy() -> None:
    sha = _lock()["qpy_sha256"]
    split = a3.a3_random_split(sha)
    assert split == a3.a3_random_split(sha)
    assert len(set(split)) == a3.N_TRAIN and all(0 <= s < 10 for s in split)
    assert a3.a3_random_split("0" * 64) != split


def test_amendment_record_matches_the_lock_and_the_code() -> None:
    path = DOCS / "correlated_noise_01_A3_amendment.json"
    amendment = json.loads(path.read_text())
    lock_path = DOCS / "correlated_noise_01_live_prereg_lock.json"
    lock = json.loads(lock_path.read_text())
    assert amendment["base_lock_sha256"] == hashlib.sha256(lock_path.read_bytes()).hexdigest()
    assert amendment["qpy_sha256"] == lock["qpy_sha256"]
    assert amendment["train_slots"] == a3.a3_random_split(lock["qpy_sha256"])
    for rel, sha in amendment["analysis_files_sha256"].items():
        assert hashlib.sha256((REPO / rel).read_bytes()).hexdigest() == sha, f"{rel} changed after A-3 was recorded"
    validation = DOCS / amendment["validation"]["file"]
    assert hashlib.sha256(validation.read_bytes()).hexdigest() == amendment["validation"]["sha256"]


def test_runner_accepts_the_recorded_amendment() -> None:
    from phase3.execute_and_analyze_live import verify_cn01_amendment

    amendment = verify_cn01_amendment()
    assert amendment["train_slots"] == a3.a3_random_split(_lock()["qpy_sha256"])
    assert len(amendment["amendment_sha256"]) == 64


def test_a3_output_on_a_synthetic_count_set_is_json_serialisable() -> None:
    from phase3.cn01_a3_validation import draw_truths, synthesize_counts
    from phase3.execute_and_analyze_live import cn01_fidelity_arrays_from_counts

    lock = _lock()
    cal = json.loads((DOCS / "ibm_marrakesh_prelock_calibration.json").read_text())
    rng = np.random.default_rng(7)
    counts = synthesize_counts(lock, draw_truths("correlated_h2", lock["coupler_cohort"], cal, rng), rng)
    assert len(counts) == lock["total_pubs"]
    res = a3.a3_inference(cn01_fidelity_arrays_from_counts(lock, counts), lock["coupler_cohort"],
                          a3.a3_random_split(lock["qpy_sha256"]), cal)
    json.loads(json.dumps(res))
    assert set(res["decisions"]) >= {"correlated_dephasing_supported", "candidate_falsified_by_held_out", "local_noise_only"}
