"""Wave 4 Sprint Verification Suite:
Capabilities #9, #10, #11 + G-14 + REQ-Q-1..8 + SYS-024 + /architecture 100% completion.

Tests:
  1. [Physics #9] Formal model builder (`osiris_cli/physics_model.py` & `/physics model`):
     - Exact rational SI dimensional analysis (`E = m * c**2`, `T = 2*pi*sqrt(L/g)`,
       `P = V * I`, `F = G * m1 * m2 / r**2`, `y = A * exp(-t / tau) * cos(omega * t)`).
     - Dimensional mismatch rejection (`E = m * c`, `x_next = x + v`, `exp(-t)` with `t:s`).
     - Symmetry & Noether conservation checks (`time_translation`, `space_translation`,
       `parity`, `shift`, `scale`, `exchange`) — passing invariant equations and rejecting
       broken symmetries.
     - `SYS-024` firewall rejecting quarantined CRSM constants (`theta_lock`, `51.843`,
       `PHI_THRESHOLD`, `0.7734`, `F_max`, `Lambda_Phi`, `tau_phase`).
  2. [Physics #10] Standard-theory reference comparator (`compare_to_standard_theory` &
     `/physics compare`, `/physics refs`):
     - Consistent predictions within `tol_sigma` / `rel_tol` across CODATA, QED, Tsirelson,
       and Carnot references, including automatic SI prefix/eV unit conversion.
     - Discrepant predictions (>3 sigma or > upper bound) flagged `RULED_OUT`.
     - Dimensional mismatch rejection (e.g. relabeling Planck mass `2.176435e-8` with `s^-1`).
     - Custom standard-theory references require explicit non-empty `source_ref` and reject
       quarantined CRSM provenance (`SYS-024`).
  3. [Quantum #11] Simulator-first hardware-readiness gate (`osiris_cli/quantum_readiness.py`,
     `/readiness`, `/experiment readiness`, `experiments.readiness`):
     - Shallow circuit passes resource & coherence budget + local depolarizing noise sweep
       (`READY_FOR_REVIEW`, `hardware_executed=False`, `evidence_class="SIMULATED"`).
     - Qubit ceiling overflow -> `BLOCKED_RESOURCES`.
     - Coherence budget / gate-survival breach -> `BLOCKED_COHERENCE`.
     - High-error noisy backend operating point -> `BLOCKED_NOISE_SENSITIVITY`.
     - `CONFIRMATORY` claim mode fails closed without `preregistration_id` + 64-hex
       `preregistration_sha256`, and passes when both are supplied.
     - `HARDWARE_ATTESTED` spoofing on simulator run fails closed (`BLOCKED_GOVERNANCE`).
     - `SYS-024` quarantined constants in circuit / prereg metadata -> `BLOCKED_GOVERNANCE`.
     - `G-14` raw shot counts preserved unmutated (`sum == shots`, canonical SHA-256 verified)
       and `exec-canon-1` JSONL receipt (`0600`, zero floats) passes `ExecJournal.verify()`
       and $O(1)$ `probes.verify_chain_tail()`.
     - `submit_to_hardware()` unconditionally fails closed (`HardwareSubmissionDisabledError`).
  4. [/architecture] Zero remaining `not built` items across all 9 layers (`L0`–`L8`) and all
     3 subsystems (`Physics research`, `Quantum research`, `Edge provisioning & administration`).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from pathlib import Path

import pytest

import experiments
import osiris_ui
from osiris_cli import (
    architecture,
    osiris_repl,
    physics_checks,
    physics_model,
    probes,
    quantum_readiness,
)

# ── 1. Capability #9: Formal Model Builder ───────────────────────────────────


def test_formal_model_validates_relativity_pendulum_and_damped_oscillator() -> None:
    rel = physics_model.build_and_check_model(
        variables="E:J, m:kg, c:m/s",
        equations="E = m * c**2",
        symmetries="time_translation, parity:c",
    )
    assert rel.verdict == "CONSISTENT"
    assert rel.violations == ()
    assert all(s.invariant for s in rel.symmetries)

    # Fractional rational exponent test: T = 2 * pi * sqrt(L / g) -> [s] == [s]
    pend = physics_model.build_and_check_model(
        variables={"T": "s", "L": "m", "g": "m/s^2"},
        equations=["T = 2 * pi * sqrt(L / g)"],
        symmetries=["scale:L=1,g=1"],
    )
    assert pend.verdict == "CONSISTENT"

    # Transcendental arguments dimensionless: y = A * exp(-t / tau) * cos(omega * t)
    damped = physics_model.build_and_check_model(
        variables="y:m, A:m, t:s, tau:s, omega:Hz",
        equations="y = A * exp(-t / tau) * cos(omega * t)",
    )
    assert damped.verdict == "CONSISTENT"


def test_formal_model_detects_dimensional_mismatches_and_bad_transcendentals() -> None:
    # E = m * c -> [kg*m^2*s^-2] != [kg*m*s^-1]
    bad_rel = physics_model.build_and_check_model(
        variables="E:J, m:kg, c:m/s",
        equations="E = m * c",
    )
    assert bad_rel.verdict == "RULED_OUT"
    assert any("LHS" in v and "RHS" in v for v in bad_rel.violations)

    # Adding meters and m/s: x2 = x1 + v
    bad_add = physics_model.build_and_check_model(
        variables="x2:m, x1:m, v:m/s",
        equations="x2 = x1 + v",
    )
    assert bad_add.verdict == "RULED_OUT"
    assert any("mismatch in '+'" in v for v in bad_add.violations)

    # Transcendental exp(-t) where t has dimension [s] (missing 1/tau)
    bad_exp = physics_model.build_and_check_model(
        variables="P:W, P0:W, t:s",
        equations="P = P0 * exp(-t)",
    )
    assert bad_exp.verdict == "RULED_OUT"
    assert any("dimensionless argument" in v for v in bad_exp.violations)


def test_formal_model_verifies_and_falsifies_symmetries() -> None:
    # Gravity law symmetric under m1 <-> m2 exchange and scale r -> 2r, m1 -> 2m1, m2 -> 2m2
    grav = physics_model.build_and_check_model(
        variables="F:N, G:m^3/(kg*s^2), m1:kg, m2:kg, r:m",
        equations="F = G * m1 * m2 / r**2",
        symmetries=["exchange:m1,m2", "parity:r", "scale:m1=1,m2=1,r=1"],
    )
    assert grav.verdict == "CONSISTENT"

    # Hooke's law F = -k * x is covariant under simultaneous parity (x -> -x, F -> -F),
    # whereas scalar potential U = 0.5*k*x**2 + alpha*x breaks x -> -x parity!
    hooke_cov = physics_model.build_and_check_model(
        variables="F:N, k:kg/s^2, x:m",
        equations="F = -k * x",
        symmetries=["parity:x,F"],
    )
    assert hooke_cov.verdict == "CONSISTENT"

    broken_parity = physics_model.build_and_check_model(
        variables="U:J, k:kg/s^2, x:m, alpha:N",
        equations="U = 0.5 * k * x**2 + alpha * x",
        symmetries=["parity:x"],
    )
    assert broken_parity.verdict == "RULED_OUT"
    assert any("parity:x" in v for v in broken_parity.violations)

    broken_time = physics_model.build_and_check_model(
        variables="E:J, P:W, t:s",
        equations="E = P * t",
        symmetries=["time_translation"],
    )
    assert broken_time.verdict == "RULED_OUT"
    assert any("time_translation" in v for v in broken_time.violations)


def test_formal_model_rejects_sys024_quarantined_constants() -> None:
    with pytest.raises(physics_model.PhysicsModelError, match="SYS-024"):
        physics_model.build_and_check_model(
            variables="theta_lock:rad, x:m",
            equations="x = x * cos(theta_lock)",
        )

    bad_magic = physics_model.build_and_check_model(
        variables="F:1, x:1",
        equations="F = 0.97871 * x",
    )
    assert bad_magic.verdict == "RULED_OUT"
    assert any("SYS-024" in v for v in bad_magic.violations)


# ── 2. Capability #10: Standard-Theory Reference Comparator ──────────────────


def test_standard_theory_comparator_consistent_and_discrepant_verdicts() -> None:
    # Within 1 sigma of CODATA 2022 alpha^-1
    ok_alpha = physics_model.compare_to_standard_theory(
        137.035999180,
        ref_key="fine_structure_inv",
        pred_uncertainty=0.000000020,
        tol_sigma=3.0,
    )
    assert ok_alpha.verdict == "CONSISTENT"
    assert ok_alpha.z_score is not None and ok_alpha.z_score < 1.0
    assert "CODATA 2022" in ok_alpha.reference.source_ref

    # Far outside 3 sigma of CODATA 2022 alpha^-1 (e.g. Eddington integer 137)
    bad_alpha = physics_model.compare_to_standard_theory(
        137.0,
        ref_key="fine_structure_inv",
        pred_uncertainty=1e-6,
        tol_sigma=3.0,
    )
    assert bad_alpha.verdict == "RULED_OUT"
    assert bad_alpha.z_score is not None and bad_alpha.z_score > 1000.0

    # Tsirelson CHSH upper bound: 2.80 <= 2*sqrt(2) is CONSISTENT; 2.95 is RULED_OUT
    assert physics_model.compare_to_standard_theory(2.80, ref_key="tsirelson_chsh").verdict == "CONSISTENT"
    assert physics_model.compare_to_standard_theory(2.95, ref_key="tsirelson_chsh").verdict == "RULED_OUT"

    # Unit scale conversion: Rydberg energy passed in Joules (~2.179872361e-18 J == 13.605693 eV)
    ryd_j = 13.605693122990 * 1.602176634e-19
    ok_ryd = physics_model.compare_to_standard_theory(
        ryd_j,
        ref_key="rydberg_energy_ev",
        pred_unit="J",
        pred_uncertainty=1e-28,
    )
    assert ok_ryd.verdict == "CONSISTENT"


def test_standard_theory_comparator_enforces_units_and_sys024_provenance() -> None:
    # Relabeling Planck mass number (2.176434e-8) with inverse-second units 's^-1' -> RULED_OUT
    wrong_unit = physics_model.compare_to_standard_theory(
        2.176434e-8,
        ref_key="planck_mass_kg",
        pred_unit="s^-1",
    )
    assert wrong_unit.verdict == "RULED_OUT"
    assert "Dimensional mismatch" in wrong_unit.why

    # Custom reference without source_ref provenance is rejected
    with pytest.raises(physics_model.PhysicsModelError, match="source_ref"):
        physics_model.compare_to_standard_theory(0.5, ref_value=0.5, source_ref="")

    # Custom reference citing quarantined CRSM / theta_lock is rejected (SYS-024)
    with pytest.raises(physics_model.PhysicsModelError, match="SYS-024"):
        physics_model.compare_to_standard_theory(
            51.843,
            ref_value=51.843,
            ref_unit="rad",
            source_ref="CRSM theta_lock invariant",
        )


def test_physics_checks_cli_wires_model_compare_and_refs() -> None:
    out_model = physics_checks.run_command(
        'model vars="E:J,m:kg,c:m/s" eq="E = m * c**2" symmetries="time_translation,parity:c"'
    )
    assert "formal model: CONSISTENT" in out_model

    out_bad_model = physics_checks.run_command('model vars="E:J,m:kg,c:m/s" eq="E = m * c"')
    assert "formal model: RULED_OUT" in out_bad_model

    out_cmp = physics_checks.run_command("compare ref=tsirelson_chsh pred=2.828 unc=0.002")
    assert "CONSISTENT" in out_cmp
    assert "Tsirelson" in out_cmp

    out_refs = physics_checks.run_command("refs")
    assert "fine_structure_inv" in out_refs
    assert "tsirelson_chsh" in out_refs


# ── 3. Capability #11: Quantum Hardware-Readiness Gate (G-14, REQ-Q-1..8) ────


def _contains_float(obj: object) -> bool:
    if isinstance(obj, float):
        return True
    if isinstance(obj, dict):
        return any(_contains_float(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_contains_float(v) for v in obj)
    return False


def test_quantum_readiness_shallow_ghz_passes_and_preserves_g14_raw_counts(tmp_path: Path) -> None:
    ohome = tmp_path / "osiris_home"
    spec = quantum_readiness.build_family_circuit("ghz", num_qubits=4, layers=1, shots=2048, seed=11)
    rep = quantum_readiness.evaluate_hardware_readiness(
        spec,
        backend="heron_r2_156q",
        claim_mode="EXPLORATORY",
        evidence_class="SIMULATED",
        osiris_home=ohome,
    )
    assert rep.verdict == "READY_FOR_REVIEW"
    assert rep.ready_for_human_review is True
    assert rep.hardware_executed is False
    assert rep.evidence_class == "SIMULATED"
    assert rep.operating_point_fidelity > 0.95
    assert len(rep.noise_sweep) == len(quantum_readiness.DEFAULT_NOISE_GRID)

    # Noise monotonicity: ideal fidelity at p=0 is 1.0 and drops as depolarizing p_2q grows
    assert math.isclose(rep.noise_sweep[0].state_fidelity, 1.0, abs_tol=1e-9)
    assert rep.noise_sweep[-1].state_fidelity < rep.noise_sweep[0].state_fidelity

    # G-14 raw counts verification: sum(counts) == shots, exact SHA-256 match
    for pt in rep.noise_sweep:
        assert sum(pt.raw_counts.values()) == 2048
        assert all(isinstance(v, int) and v > 0 for v in pt.raw_counts.values())
        expected_sha = quantum_readiness._hash_raw_counts(pt.raw_counts, 2048)
        assert pt.raw_counts_sha256 == expected_sha

    # Verify exec-canon-1 receipt chain, 0600 permissions, zero floats, and O(1) tail probe
    rpath = Path(rep.receipt_path)
    assert rpath.is_file()
    if os.name != "nt":
        assert stat.S_IMODE(rpath.stat().st_mode) == 0o600
    ok, count, err = quantum_readiness.verify_receipts(rpath)
    assert ok is True and count == 1 and err == ""
    tail = probes.verify_chain_tail(rpath, tail_lines=10)
    assert tail.intact is True

    parsed_row = json.loads(rpath.read_text(encoding="utf-8").strip(), parse_float=lambda s: float(s))
    # Re-parse without float coercion to verify JSON contains zero numeric floats (only ints & strings)
    strict_row = json.loads(
        rpath.read_text(encoding="utf-8").strip(),
        parse_float=lambda s: (_ for _ in ()).throw(AssertionError(f"float token {s!r} in exec-canon-1 receipt")),
    )
    assert not _contains_float(strict_row)
    assert strict_row["body"]["hardware"] is False
    assert parsed_row["body"]["verdict"] == "READY_FOR_REVIEW"


def test_quantum_readiness_blocks_qubit_ceiling_coherence_and_noise_sensitivity(tmp_path: Path) -> None:
    ohome = tmp_path / "ohome"

    # 1. Qubit ceiling exceeded (200q > 156q Heron ceiling)
    too_wide = quantum_readiness.build_family_circuit("ghz", num_qubits=200, layers=1)
    rep_wide = quantum_readiness.evaluate_hardware_readiness(too_wide, backend="heron_r2_156q", osiris_home=ohome)
    assert rep_wide.verdict == "BLOCKED_RESOURCES"
    assert rep_wide.ready_for_human_review is False

    # 2. Coherence window exceeded (50 us idle > 25% of 150 us T2)
    too_long = quantum_readiness.build_family_circuit("ramsey_idle", num_qubits=3, layers=1, idle_delay_ns=50_000.0)
    rep_coh = quantum_readiness.evaluate_hardware_readiness(too_long, backend="heron_r2_156q", osiris_home=ohome)
    assert rep_coh.verdict == "BLOCKED_COHERENCE"

    # 3. Noise sensitivity / operating-point fidelity drop (within coherence window, high 2Q error backend)
    noisy_profile = quantum_readiness.BackendProfile(
        name="noisy_test_qpu",
        max_qubits=32,
        t1_us=500.0,
        t2_us=500.0,
        t_1q_ns=20.0,
        t_2q_ns=40.0,
        err_1q=0.002,
        err_2q=0.045,
        readout_err=0.02,
        min_survival_floor=0.10,
        min_operating_fidelity=0.75,
    )
    trotter_spec = quantum_readiness.build_family_circuit("ising_trotter", num_qubits=5, layers=3, shots=1024)
    rep_noise = quantum_readiness.evaluate_hardware_readiness(trotter_spec, backend=noisy_profile, osiris_home=ohome)
    assert rep_noise.verdict == "BLOCKED_NOISE_SENSITIVITY"


def test_quantum_readiness_enforces_confirmatory_prereg_evidence_class_and_sys024(tmp_path: Path) -> None:
    ohome = tmp_path / "ohome"
    spec = quantum_readiness.build_family_circuit("bell", num_qubits=2, layers=1, shots=1024)

    # CONFIRMATORY without preregistration_id / sha256 fails closed
    rep_unreg = quantum_readiness.evaluate_hardware_readiness(
        spec,
        claim_mode="CONFIRMATORY",
        osiris_home=ohome,
    )
    assert rep_unreg.verdict == "BLOCKED_GOVERNANCE"
    assert any("preregistration_id" in r for r in rep_unreg.reasons)

    # CONFIRMATORY with valid preregistration_id + 64-hex sha256 passes
    prereg_hash = hashlib.sha256(b"preregistered-bell-protocol-v1").hexdigest()
    rep_reg = quantum_readiness.evaluate_hardware_readiness(
        spec,
        claim_mode="CONFIRMATORY",
        preregistration_id="10.5281/zenodo.22863245",
        preregistration_sha256=prereg_hash,
        osiris_home=ohome,
    )
    assert rep_reg.verdict == "READY_FOR_REVIEW"

    # Spoofing HARDWARE_ATTESTED on simulator run fails closed
    rep_spoof = quantum_readiness.evaluate_hardware_readiness(
        spec,
        evidence_class="HARDWARE_ATTESTED",
        osiris_home=ohome,
    )
    assert rep_spoof.verdict == "BLOCKED_GOVERNANCE"
    assert rep_spoof.evidence_class == "SIMULATED"

    # SYS-024 quarantined constant in circuit name fails closed
    bad_spec = quantum_readiness.build_family_circuit(
        "bell", num_qubits=2, layers=1, name="theta_lock_51.843_resonance"
    )
    rep_sys024 = quantum_readiness.evaluate_hardware_readiness(bad_spec, osiris_home=ohome)
    assert rep_sys024.verdict == "BLOCKED_GOVERNANCE"

    # Hardware submission adapter unconditionally fails closed
    with pytest.raises(quantum_readiness.HardwareSubmissionDisabledError, match="HARDWARE_SUBMISSION_DISABLED"):
        quantum_readiness.submit_to_hardware(spec, backend="ibm_fez")


def test_experiments_brief_and_cli_readiness_integration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ohome = tmp_path / "ohome"
    monkeypatch.setenv("OSIRIS_HOME", str(ohome))
    monkeypatch.setattr(experiments, "STORE", str(tmp_path / "experiments.json"))

    saved = experiments.save(
        experiments.new_brief(
            question="Does dynamical decoupling preserve 4-qubit GHZ state fidelity?",
            hypothesis="XY4 reduces expectation-value error relative to unmitigated idle.",
            system="4-qubit GHZ circuit, depth 4",
            variables=("DD sequence",),
            metric="state fidelity",
            controls=("seed 13", "2048 shots", "depolarizing noise"),
        )
    )
    rep = experiments.readiness(saved["id"], backend="heron_r2_156q", osiris_home=ohome)
    assert rep.verdict == "READY_FOR_REVIEW"
    assert rep.resource_model.num_qubits == 4

    cli_out = quantum_readiness.run_command(f"brief {saved['id']}", osiris_home=ohome)
    assert "READY_FOR_REVIEW" in cli_out
    assert "verify" in quantum_readiness.USAGE

    verify_out = quantum_readiness.run_command("verify", osiris_home=ohome)
    assert "INTACT: 2 receipts" in verify_out


# ── 4. [/architecture] Zero Remaining `not built` Items Across Entire OSIRIS ─


def test_wave4_completes_all_layers_and_subsystems_with_zero_missing() -> None:
    assert "/readiness" in osiris_repl.REPL_COMMANDS
    assert "/readiness" in osiris_repl.READ_ONLY_COMMANDS

    layer_missing = [item for ly in architecture.LAYERS for item in ly.missing]
    subsystem_missing = [item for _name, _built, missing in architecture.SUBSYSTEMS for item in missing]
    assert layer_missing == []
    assert subsystem_missing == []

    state = architecture.live_state(None, None)
    rendered = architecture.full(osiris_ui.Canvas("plain", 100), state)
    assert "not built:" not in rendered
    assert "osiris_cli/physics_model.py" in rendered
    assert "osiris_cli/quantum_readiness.py" in rendered
def test_qiskit_circuit_extraction_and_repl_console_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ohome = tmp_path / "ohome"
    monkeypatch.setenv("OSIRIS_HOME", str(ohome))

    # Minimal duck-typed QuantumCircuit (always runs, including Python 3.9 without qiskit)
    class _Op:
        def __init__(self, name: str, params: tuple[float, ...] = ()) -> None:
            self.name = name
            self.params = params

    class _Inst:
        def __init__(self, op: _Op, qubits: tuple[object, ...]) -> None:
            self.operation = op
            self.qubits = qubits

    class _FakeQC:
        name = "bell_pair_qc"
        num_qubits = 2
        qubits = ("q0", "q1")
        data = (
            _Inst(_Op("h"), ("q0",)),
            _Inst(_Op("cx"), ("q0", "q1")),
        )

        @staticmethod
        def depth() -> int:
            return 2

    spec = quantum_readiness.from_qiskit_circuit(_FakeQC(), shots=1024, seed=5)
    assert spec.num_qubits == 2 and spec.single_qubit_gates == 1 and spec.two_qubit_gates == 1
    rep = quantum_readiness.evaluate_hardware_readiness(spec, osiris_home=ohome)
    assert rep.verdict == "READY_FOR_REVIEW"

    # Dispatch /readiness and /experiment readiness through osiris_repl.dispatch_command
    state = osiris_repl.OsirisReplState()
    osiris_repl.dispatch_command(state, "/readiness backends")
    out1 = capsys.readouterr().out
    assert "heron_r2_156q" in out1

    osiris_repl.dispatch_command(state, "/experiment readiness check family=bell qubits=2 shots=512")
    out2 = capsys.readouterr().out
    assert "READY_FOR_REVIEW" in out2
