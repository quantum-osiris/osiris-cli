"""Verification suite for Card C Multi-Distance QEC Threshold & `Lambda_d` Gate
(`experiments/qec_card_c/qec_multidistance_scaling.py`).

Verifies:
  1. Python 3.9--3.14 AST and source-level CI compatibility (`from __future__ import annotations`,
     zero `match`/`case`, zero `int.bit_count()`, zero `zip(..., strict=...)`, zero
     `Path.write_text(..., newline=...)`).
  2. General-`d` (`d in {3, 5, 7}`) repetition `[[d,1,d]]` and Tomita-Svore checkerboard rotated
     surface `[[d^2,1,d]]` (`17Q, 49Q, 97Q`) schedules: zero CNOT step collisions, noiseless
     zero-detector invariance, and 100% op-for-op match with `qec_crossing_gate.py` at `d=3`.
  3. Exact polynomial-time MWPM equivalence (`networkx` Edmonds Blossom vs `qec_crossing_gate.py`
     `2^k` bitmask-DP across `>= 600` random multi-defect syndromes = `100.0%` match, plus `d=7`
     `192`-detector scaling beyond `int64` bitwidth).
  4. Hook-safe circuit fault distance `d_DEM == d` (`3, 5, 7` in `X` and `Z` bases) and zero
     logical failures across `t = (d - 1) // 2` faults (`t=1` exhaustive on `d=3,5,7`; `t=2`
     exhaustive on `d=5` surface & `d=5,7` repetition; `t=3` witness + sampled on `d=7`).
  5. Sub-threshold `Lambda_{3->5} > 1`, `Lambda_{5->7} > 1` vs super-threshold `Lambda < 1`
     inversion, `FEZ_PUBLISHED` / `FEZ_DRIFTED_SPAM` / `H2_1_PUBLISHED` scaling regimes,
     governance metadata (`SIMULATED`, `provider_job_id is None`, `qpu_seconds == 0`), RFC 8785
     JCS digest integrity (including pure-Python ECMA-262 fallback), and no-`qiskit` execution.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

SCALING_PY = REPO_ROOT / "experiments" / "qec_card_c" / "qec_multidistance_scaling.py"
CROSSING_PY = REPO_ROOT / "experiments" / "qec_card_c" / "qec_crossing_gate.py"
REPORT_PATH = (
    REPO_ROOT
    / "experiments"
    / "qec_card_c"
    / "artifacts"
    / "card_c_multidistance_threshold_report.json"
)
THIS_TEST_PY = Path(__file__).resolve()


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"Cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


scaling_mod = _load_module("_test_qec_multidistance_scaling", SCALING_PY)
gate_mod = _load_module("_test_qec_card_c_gate_ref", CROSSING_PY)


@pytest.fixture(scope="module")
def multidistance_report() -> Dict[str, Any]:
    assert REPORT_PATH.is_file(), f"Missing multi-distance QEC artifact: {REPORT_PATH}"
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Python 3.9--3.14 AST & CI Compatibility Checks (No Qiskit Required)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("py_path", [SCALING_PY, THIS_TEST_PY])
def test_python39_to_314_ast_and_source_compatibility(py_path: Path) -> None:
    source = py_path.read_text(encoding="utf-8")
    assert "from __future__ import annotations" in source

    tree = ast.parse(source, filename=str(py_path), feature_version=(3, 9))
    match_cls = getattr(ast, "Match", None)

    for node in ast.walk(tree):
        if match_cls is not None:
            assert not isinstance(node, match_cls), f"Python 3.10+ match/case found in {py_path}"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "bit_count":
                raise AssertionError(f"Python 3.10+ int.bit_count() found in {py_path}")
            if isinstance(node.func, ast.Name) and node.func.id == "zip":
                for kw in node.keywords:
                    assert kw.arg != "strict", f"Python 3.10+ zip(strict=...) found in {py_path}"
            if isinstance(node.func, ast.Attribute) and node.func.attr == "write_text":
                for kw in node.keywords:
                    assert kw.arg != "newline", (
                        f"Python 3.10+ Path.write_text(newline=...) found in {py_path}"
                    )
            if isinstance(node.func, ast.Name) and node.func.id in ("isinstance", "issubclass"):
                if len(node.args) >= 2:
                    assert not isinstance(node.args[1], ast.BinOp), (
                        f"Python 3.10+ PEP 604 union in isinstance/issubclass in {py_path}"
                    )


# ---------------------------------------------------------------------------
# 2. General-d Schedule, Commutation, & Exact d=3 Compatibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("d", "expected_rep_q", "expected_surf_q", "expected_surf_det"),
    [
        (3, 5, 17, 16),
        (5, 9, 49, 72),
        (7, 13, 97, 192),
    ],
)
def test_general_d_schedule_commutation_collision_and_dimensions(
    d: int, expected_rep_q: int, expected_surf_q: int, expected_surf_det: int
) -> None:
    profiles = gate_mod.build_published_target_profiles()
    z_sched, x_sched, z_supp, x_supp, n_tot = scaling_mod.build_rotated_surface_schedule(d)
    assert n_tot == expected_surf_q
    assert len(z_sched) == (d * d - 1) // 2
    assert len(x_sched) == (d * d - 1) // 2

    # Zero data-qubit collision at each of the 4 CNOT steps
    for step in range(4):
        touched = [
            pts[step]
            for pts in list(z_sched.values()) + list(x_sched.values())
            if pts[step] is not None
        ]
        assert len(touched) == len(set(touched)), f"CNOT collision at d={d}, step={step}"

    # Every Z support commutes with every X support (even overlap)
    for zs in z_supp.values():
        z_set = set(zs)
        for xs in x_supp.values():
            assert (len(z_set.intersection(xs)) % 2) == 0

    for basis in ("X", "Z"):
        rep_dem = scaling_mod.build_repetition_dem(d, d, profiles["FEZ_PUBLISHED"], basis=basis)
        assert rep_dem.n_qubits == expected_rep_q
        assert rep_dem.n_detectors == (d - 1) * (d + 1)
        assert scaling_mod.verify_noiseless_zero_syndrome(rep_dem) is True

        surf_dem = scaling_mod.build_rotated_surface_dem(
            d, d, profiles["H2_1_PUBLISHED"], basis=basis
        )
        assert surf_dem.n_qubits == expected_surf_q
        assert surf_dem.n_detectors == expected_surf_det
        assert scaling_mod.verify_noiseless_zero_syndrome(surf_dem) is True


def test_exact_d3_compatibility_with_qec_crossing_gate() -> None:
    profiles = gate_mod.build_published_target_profiles()
    for prof_id in ("FEZ_PUBLISHED", "H2_1_PUBLISHED"):
        prof = profiles[prof_id]
        for basis in ("X", "Z"):
            rep_new = scaling_mod.build_repetition_dem(3, 3, prof, basis=basis)
            rep_ref = gate_mod.build_repetition_d3_dem(3, prof, basis=basis)
            assert rep_new.ops == rep_ref.ops
            assert rep_new.detectors == rep_ref.detectors
            assert rep_new.observable == rep_ref.observable
            assert set(rep_new.mechanisms.keys()) == set(rep_ref.mechanisms.keys())
            for key in rep_ref.mechanisms:
                assert abs(rep_new.mechanisms[key] - rep_ref.mechanisms[key]) < 1e-15

            surf_new = scaling_mod.build_rotated_surface_dem(3, 3, prof, basis=basis)
            surf_ref = gate_mod.build_rotated_surface_d3_dem(3, prof, basis=basis)
            assert surf_new.ops == surf_ref.ops
            assert surf_new.detectors == surf_ref.detectors
            assert surf_new.observable == surf_ref.observable
            assert set(surf_new.mechanisms.keys()) == set(surf_ref.mechanisms.keys())
            for key in surf_ref.mechanisms:
                assert abs(surf_new.mechanisms[key] - surf_ref.mechanisms[key]) < 1e-15


# ---------------------------------------------------------------------------
# 3. Exact Polynomial-Time Blossom MWPM Equivalence vs Bitmask-DP (>= 500)
# ---------------------------------------------------------------------------


def test_polynomial_blossom_mwpm_exact_match_with_bitmask_dp() -> None:
    audit = scaling_mod.audit_polynomial_mwpm_equivalence(
        n_syndromes_per_code=350, seed=scaling_mod.DEFAULT_SEED
    )
    assert audit["total_multi_defect_syndromes_tested"] >= 1000
    assert audit["all_exact_100pct_match"] is True
    for check in audit["per_dem_checks"].values():
        assert check["pure_blossom_mismatches_vs_bitmask_dp"] == 0
        assert check["fast_poly_mismatches_vs_bitmask_dp"] == 0
        assert check["agreement_rate"] == 1.0

    # Verify d=7 (192 detectors > 64-bit integer width) high-defect Blossom consistency
    prof = gate_mod.build_published_target_profiles()["FEZ_PUBLISHED"]
    dem7 = scaling_mod.build_rotated_surface_dem(7, 7, prof, basis="X")
    assert dem7.n_detectors == 192
    dec_pure = scaling_mod.PolynomialMWPMDecoder(dem7, use_component_dp=False)
    dec_fast = scaling_mod.PolynomialMWPMDecoder(dem7, use_component_dp=True)

    rng = np.random.default_rng(20261124)
    for _ in range(80):
        k = int(rng.integers(6, 26))
        chosen = sorted(rng.choice(dem7.n_detectors, size=k, replace=False).tolist())
        big_mask = 0
        for u in chosen:
            big_mask |= 1 << u
        assert dec_pure.decode_nodes(chosen) == dec_fast.decode_mask(big_mask)


# ---------------------------------------------------------------------------
# 4. Hook-Safe Circuit Fault-Distance G_dist & t = (d-1)//2 Zero Failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code_family", "d", "basis"),
    [
        ("repetition", 3, "X"),
        ("repetition", 5, "X"),
        ("repetition", 7, "X"),
        ("rotated_surface", 3, "X"),
        ("rotated_surface", 3, "Z"),
        ("rotated_surface", 5, "X"),
        ("rotated_surface", 5, "Z"),
        ("rotated_surface", 7, "X"),
        ("rotated_surface", 7, "Z"),
    ],
)
def test_hook_safe_exact_dem_fault_distance_and_t_fault_recovery(
    code_family: str, d: int, basis: str
) -> None:
    profiles = gate_mod.build_published_target_profiles()
    prof = profiles["FEZ_PUBLISHED"] if code_family == "repetition" else profiles["H2_1_PUBLISHED"]
    dem = (
        scaling_mod.build_repetition_dem(d, d, prof, basis=basis)
        if code_family == "repetition"
        else scaling_mod.build_rotated_surface_dem(d, d, prof, basis=basis)
    )
    d_exact, witness = scaling_mod.compute_exact_dem_fault_distance(dem)
    assert d_exact == d, f"Expected full hook-safe distance {d}, got {d_exact}"
    assert len(witness) == d

    # Witness combines to zero syndrome and non-trivial logical observable flip 1
    comb_dm = 0
    comb_of = 0
    for dm, of in witness:
        comb_dm ^= dm
        comb_of ^= of
    assert comb_dm == 0
    assert comb_of == 1

    dec_unit = scaling_mod.PolynomialMWPMDecoder(dem, uniform_weights=True)
    dec_weighted = scaling_mod.PolynomialMWPMDecoder(dem, uniform_weights=False)
    keys = sorted(dem.mechanisms.keys())

    # Exhaustive t = 1 across every DEM mechanism
    for dm, of in keys:
        assert dec_unit.decode_mask(dm) == of
        assert dec_weighted.decode_mask(dm) == of

    t_corr = (d - 1) // 2
    rng = np.random.default_rng(20261124 + d)
    if t_corr >= 2:
        if code_family == "repetition":
            # Exhaustive all pairs on repetition d=5 and d=7
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    assert dec_unit.decode_mask(keys[i][0] ^ keys[j][0]) == (
                        keys[i][1] ^ keys[j][1]
                    )
        else:
            # Sampled 3,000 weight-2 pairs here (70,500 exhaustive verified in artifact)
            for _ in range(3000):
                i, j = rng.choice(len(keys), size=2, replace=False).tolist()
                assert dec_unit.decode_mask(keys[i][0] ^ keys[j][0]) == (keys[i][1] ^ keys[j][1])

    if t_corr >= 3:
        # Every 3-fault subset along the minimum-cut logical witness + 2,000 random triples
        for a in range(d):
            for b in range(a + 1, d):
                for c in range(b + 1, d):
                    dm3 = witness[a][0] ^ witness[b][0] ^ witness[c][0]
                    of3 = witness[a][1] ^ witness[b][1] ^ witness[c][1]
                    assert dec_unit.decode_mask(dm3) == of3
        for _ in range(2000):
            i, j, k_idx = rng.choice(len(keys), size=3, replace=False).tolist()
            assert dec_unit.decode_mask(keys[i][0] ^ keys[j][0] ^ keys[k_idx][0]) == (
                keys[i][1] ^ keys[j][1] ^ keys[k_idx][1]
            )


# ---------------------------------------------------------------------------
# 5. Live Sub-Threshold vs Super-Threshold Inversion & Determinism
# ---------------------------------------------------------------------------


def test_live_subthreshold_suppression_vs_superthreshold_inversion() -> None:
    prof_sub = scaling_mod.build_circuit_depolarizing_profile(0.0020, idle_multiplier=2.0)
    prof_sup = scaling_mod.build_circuit_depolarizing_profile(0.0150, idle_multiplier=2.0)

    eps_sub: Dict[int, float] = {}
    eps_sup: Dict[int, float] = {}
    for d in scaling_mod.DISTANCES:
        dem_sub = scaling_mod.build_rotated_surface_dem(
            d, d, prof_sub, basis="X", include_topology_overhead=False
        )
        w_sub, _ = scaling_mod.simulate_dem_failure_weights(
            dem_sub,
            scaling_mod.PolynomialMWPMDecoder(dem_sub),
            n_shots=2500,
            seed=900 + d,
            distance=d,
        )
        eps_sub[d] = gate_mod.logical_error_per_round(float(np.mean(w_sub)), d)

        dem_sup = scaling_mod.build_rotated_surface_dem(
            d, d, prof_sup, basis="X", include_topology_overhead=False
        )
        w_sup, _ = scaling_mod.simulate_dem_failure_weights(
            dem_sup,
            scaling_mod.PolynomialMWPMDecoder(dem_sup),
            n_shots=2500,
            seed=950 + d,
            distance=d,
        )
        eps_sup[d] = gate_mod.logical_error_per_round(float(np.mean(w_sup)), d)

    # Sub-threshold: strict exponential decrease eps(3) > eps(5) > eps(7)
    assert eps_sub[3] > eps_sub[5] > eps_sub[7] > 0.0
    assert (eps_sub[3] / eps_sub[5]) > 1.5
    assert (eps_sub[5] / eps_sub[7]) > 1.5

    # Super-threshold: strict error amplification eps(3) < eps(5) < eps(7) (Lambda < 1)
    assert eps_sup[3] < eps_sup[5] < eps_sup[7]
    assert (eps_sup[3] / eps_sup[5]) < 1.0
    assert (eps_sup[5] / eps_sup[7]) < 1.0

    # Exact deterministic repeatability
    dem_det = scaling_mod.build_repetition_dem(5, 5, prof_sub, basis="X")
    w_a, _ = scaling_mod.simulate_dem_failure_weights(
        dem_det, scaling_mod.PolynomialMWPMDecoder(dem_det), n_shots=1500, seed=20261124, distance=5
    )
    w_b, _ = scaling_mod.simulate_dem_failure_weights(
        dem_det, scaling_mod.PolynomialMWPMDecoder(dem_det), n_shots=1500, seed=20261124, distance=5
    )
    assert np.array_equal(w_a, w_b)


# ---------------------------------------------------------------------------
# 6. Artifact Schema, Governance, RFC 8785 JCS Digest & Target Scaling Laws
# ---------------------------------------------------------------------------


def test_artifact_schema_governance_jcs_hash_and_target_scaling(
    multidistance_report: Dict[str, Any],
) -> None:
    rep = multidistance_report
    assert rep["schema"] == "osiris.qec_card_c.multidistance_threshold_report/1"
    assert rep["execution_tier"] == "SIMULATED"
    assert rep["provider_job_id"] is None
    assert rep["qpu_seconds"] == 0
    assert rep["hardware_execution_authorized"] is False
    assert rep["canonicalization"] == "RFC8785-JCS"
    assert rep["seed"] == 20261124

    # Zero timestamps in artifact JSON
    raw_json = REPORT_PATH.read_text(encoding="utf-8")
    for banned_ts_key in ('"timestamp"', '"created_at"', '"generated_at"', '"run_timestamp"'):
        assert banned_ts_key not in raw_json
    assert not re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", raw_json)

    # RFC 8785 JCS digest verification + pure-Python ECMA-262 fallback byte-for-byte check
    body = {k: v for k, v in rep.items() if k != "jcs_payload_sha256"}
    expected_jcs = scaling_mod.jcs_bytes(body)
    fallback_jcs = scaling_mod._jcs_fallback_encode(body).encode("utf-8")
    assert fallback_jcs == expected_jcs
    assert hashlib.sha256(expected_jcs).hexdigest() == rep["jcs_payload_sha256"]

    # Verification controls embedded in report
    vc = rep["verification_controls"]
    assert vc["polynomial_mwpm_vs_bitmask_dp_equivalence"]["all_exact_100pct_match"] is True
    assert vc["polynomial_mwpm_vs_bitmask_dp_equivalence"][
        "total_multi_defect_syndromes_tested"
    ] >= 3000
    f_audit = vc["fault_distance_and_hook_safety_audit"]
    assert f_audit["all_distance_and_t_fault_audits_passed"] is True
    assert f_audit["per_code_audits"]["rotated_surface_d5_49q__X"]["t2_tested"] == 70500
    assert f_audit["per_code_audits"]["rotated_surface_d5_49q__X"]["t2_unit_mwpm_failures"] == 0

    # Threshold sweep integrity (>= 20,000 shots per cell)
    sweeps = rep["threshold_sweeps"]
    assert sweeps["n_shots_per_cell"] >= 20000
    assert sweeps["alpha_grid"] == [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    surf_th = sweeps["rotated_surface_threshold_sweep"]["threshold_estimate"]
    assert surf_th["subthreshold_at_min_alpha"] is True
    assert surf_th["superthreshold_inversion_at_max_alpha"] is True
    assert 0.007 <= float(surf_th["p_th_consensus"]) <= 0.013

    rep_th = sweeps["repetition_threshold_crossing_sweep"]["threshold_estimate"]
    assert rep_th["subthreshold_at_min_alpha"] is True
    assert rep_th["superthreshold_inversion_at_max_alpha"] is True
    assert float(rep_th["p_th_consensus"]) > float(surf_th["p_th_consensus"]) * 4.0
    rep_matched_rows = sweeps["repetition_matched_p_sweep"]["rows"]
    assert all(
        r["lambda_3_to_5"] > 2.5 and r["lambda_5_to_7"] > 2.5 for r in rep_matched_rows
    )

    # Operational target profiles: FEZ_PUBLISHED, FEZ_DRIFTED_SPAM, H2_1_PUBLISHED
    cells = rep["operational_profile_cells"]

    fez_rep_pf = cells["FEZ_PUBLISHED__repetition__PAULI_FRAME_MWPM"]
    fez_rep_ff = cells["FEZ_PUBLISHED__repetition__PHYSICAL_FEEDFORWARD_MWPM"]
    assert fez_rep_pf["lambda_3_to_5"] > 3.0 and fez_rep_pf["lambda_5_to_7"] > 3.0
    assert fez_rep_ff["lambda_3_to_5"] > 3.0 and fez_rep_ff["lambda_5_to_7"] > 3.0

    fez_surf_pf = cells["FEZ_PUBLISHED__rotated_surface__PAULI_FRAME_MWPM"]
    fez_surf_ff = cells["FEZ_PUBLISHED__rotated_surface__PHYSICAL_FEEDFORWARD_MWPM"]
    assert fez_surf_pf["lambda_3_to_5"] > 1.0 and fez_surf_pf["lambda_5_to_7"] > 1.0
    assert (
        fez_surf_ff["per_distance"]["3"]["epsilon_L_per_round"]
        > fez_surf_pf["per_distance"]["3"]["epsilon_L_per_round"]
    )

    drift_surf_pf = cells["FEZ_DRIFTED_SPAM__rotated_surface__PAULI_FRAME_MWPM"]
    drift_surf_ff = cells["FEZ_DRIFTED_SPAM__rotated_surface__PHYSICAL_FEEDFORWARD_MWPM"]
    assert drift_surf_pf["d3_beats_bare_in_block_rate"] is False
    assert drift_surf_pf["d3_beats_bare_in_differential_rate"] is False
    assert drift_surf_ff["lambda_3_to_5_differential"] < fez_surf_pf["lambda_3_to_5"]

    h2_surf_pf = cells["H2_1_PUBLISHED__rotated_surface__PAULI_FRAME_MWPM"]
    h2_surf_ff = cells["H2_1_PUBLISHED__rotated_surface__PHYSICAL_FEEDFORWARD_MWPM"]
    assert h2_surf_pf["lambda_3_to_5"] > 4.0 and h2_surf_pf["lambda_5_to_7"] > 4.0
    assert h2_surf_ff["lambda_3_to_5"] > 4.0 and h2_surf_ff["lambda_5_to_7"] > 3.0

    # Secret & claim-gate hygiene when osiris_cli is importable
    for secret_pat in ("crn:v1:bluemix:", "ZENODO_TOKEN=", '"apikey"'):
        assert secret_pat not in raw_json

    try:
        from osiris_cli import claim_provenance, claims, plan_gates
    except ImportError:
        return

    assert not claims.flagged_in_reply(raw_json, [])
    for p_id, p_re in plan_gates._QUARANTINED_PLAN_PATTERNS:
        assert not p_re.search(raw_json), f"Banned pattern {p_id} in report"
    for q_id, q_re in claim_provenance._QUARANTINED_CONSTANTS:
        assert not q_re.search(raw_json), f"Quarantined constant {q_id} in report"


# ---------------------------------------------------------------------------
# 7. Optional Qiskit Stabilizer Algebra Cross-Check (Skipped on No-Qiskit CI)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [3, 5, 7])
def test_qiskit_stabilizer_commutation_and_logical_anticommutation(d: int) -> None:
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Pauli, PauliList

    _, _, z_supp, x_supp, _ = scaling_mod.build_rotated_surface_schedule(d)
    n_data = d * d
    stabs = []
    for qs in z_supp.values():
        chars = ["I"] * n_data
        for q in qs:
            chars[n_data - 1 - q] = "Z"
        stabs.append("".join(chars))
    for qs in x_supp.values():
        chars = ["I"] * n_data
        for q in qs:
            chars[n_data - 1 - q] = "X"
        stabs.append("".join(chars))

    plist = PauliList(stabs)
    assert len(plist) == d * d - 1

    # Pairwise stabilizer commutation across all d^2 - 1 generators
    for i in range(len(plist)):
        for j in range(i + 1, len(plist)):
            assert plist[i].commutes(plist[j]), f"Stabilizers {i}, {j} anticommute at d={d}"

    # Logical Z_L (col 0) and X_L (row 0) commute with all stabilizers and anticommute mutually
    zl_chars = ["I"] * n_data
    xl_chars = ["I"] * n_data
    for idx in range(d):
        zl_chars[n_data - 1 - (idx * d)] = "Z"
        xl_chars[n_data - 1 - idx] = "X"
    z_logical = Pauli("".join(zl_chars))
    x_logical = Pauli("".join(xl_chars))

    for gen in plist:
        assert z_logical.commutes(gen)
        assert x_logical.commutes(gen)
    assert z_logical.anticommutes(x_logical)
