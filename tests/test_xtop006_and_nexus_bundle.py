"""Comprehensive tests for OSIRIS-XTOP-006 (`xtop006/`) and OSIRIS-XTOP-005 Nexus H2 bundle (`xtop005/`)."""

from __future__ import annotations

import json
import math
import shutil
import uuid
from collections import Counter

import numpy as np
import pytest
import rfc8785
from scipy.stats import unitary_group

from osiris_cli.providers.quantinuum_nexus import GovernanceError
from osiris_cli.quantinuum_gate import mint_tier3_token
from xtop005 import run_nexus_h2_decisive as rnh
from xtop006 import kraus_channel as kc
from xtop006 import qccd_circuits as qc6


@pytest.fixture(scope="module")
def h2_offline_backend():
    return qc6._offline_h2_backend()


# ---------------------------------------------------------------------------
# 1. Analytical Kraus Channel & Tunable Non-Clifford Scrambler (`kraus_channel.py`)
# ---------------------------------------------------------------------------


def test_kraus_formula_matches_explicit_index_loops():
    """Verify `(M_r)_{xa} = (1/4) sum_{b,c,k} U_{ijk,abc} conj(U_{nmk,xbc})` against literal loops."""
    u = unitary_group.rvs(8, random_state=42)
    m_vec = kc.compute_remnant_kraus_operators(u, bell_outcome="Phi+")
    assert m_vec.shape == (16, 2, 2)

    for r_idx in range(16):
        i = (r_idx >> 3) & 1
        j = (r_idx >> 2) & 1
        m = (r_idx >> 1) & 1
        n = r_idx & 1
        manual = np.zeros((2, 2), dtype=np.complex128)
        for x_out in (0, 1):
            for a_in in (0, 1):
                acc = 0.0j
                for b in (0, 1):
                    for c in (0, 1):
                        for k in (0, 1):
                            row_f = (i << 2) | (j << 1) | k
                            col_f = (a_in << 2) | (b << 1) | c
                            row_c = (n << 2) | (m << 1) | k
                            col_c = (x_out << 2) | (b << 1) | c
                            acc += u[row_f, col_f] * np.conj(u[row_c, col_c])
                manual[x_out, a_in] = 0.25 * acc
        assert np.allclose(m_vec[r_idx], manual, atol=1e-14)


def test_cptp_completeness_haar_random_and_theta_grid():
    """Verify sum_{beta, r} M_{beta, r}^dagger M_{beta, r} == I_2 and Choi CP across all 4 Bell outcomes."""
    for seed in (7, 19, 101):
        u_rand = unitary_group.rvs(8, random_state=seed)
        rep = kc.verify_cptp_completeness(u_rand)
        assert rep["cptp_valid"] is True
        assert rep["completeness_frobenius_error"] < 1e-13
        assert all(eig >= -1e-13 for eig in rep["choi_min_eigenvalues"].values())

    for theta in kc.THETA_GRID_DEFAULT:
        u_th = kc.landsman_figgatt_scrambler(theta)
        rep = kc.verify_cptp_completeness(u_th)
        assert rep["cptp_valid"] is True
        assert rep["completeness_frobenius_error"] < 1e-13


def test_scrambler_theta_zero_is_identity_and_unscrambled():
    """At theta=0, U(0) == I_8, M_r(0) is not proportional to I_2, P_{Phi+} == 1.0, and F == 0.5."""
    u0 = kc.landsman_figgatt_scrambler(0.0)
    assert np.allclose(u0, np.eye(8), atol=1e-14)
    kraus0 = kc.compute_remnant_kraus_operators(u0, "Phi+")
    assert kc.decoupling_deviation(kraus0) == pytest.approx(0.5, abs=1e-12)

    for psi, _, _ in kc.INPUT_STATES_6.values():
        ev = kc.apply_bell_channel(u0, psi, "Phi+")
        assert ev.success_probability == pytest.approx(1.0, abs=1e-13)
        assert np.allclose(ev.normalized_rho_out, 0.5 * np.eye(2), atol=1e-13)
        assert ev.fidelity_with_input == pytest.approx(0.5, abs=1e-13)


def test_scrambler_theta_pi_over_2_maximal_decoupling():
    """At theta=pi/2, M_r(pi/2) = c_r I_2 for all 16 remnants, P_{Phi+} = 1/4, F = 1.0 for every input."""
    u_max = kc.landsman_figgatt_scrambler(math.pi / 2.0)
    kraus_max = kc.compute_remnant_kraus_operators(u_max, "Phi+")
    assert kc.decoupling_deviation(kraus_max) < 1e-14

    c_sq_sum = sum(abs(0.5 * np.trace(m)) ** 2 for m in kraus_max)
    assert c_sq_sum == pytest.approx(0.25, abs=1e-14)

    for psi, _, _ in kc.INPUT_STATES_6.values():
        ev = kc.apply_bell_channel(u_max, psi, "Phi+")
        assert ev.success_probability == pytest.approx(0.25, abs=1e-14)
        assert np.allclose(ev.normalized_rho_out, np.outer(psi, psi.conj()), atol=1e-13)
        assert ev.fidelity_with_input == pytest.approx(1.0, abs=1e-13)

    # Also verify exact recovery on a mixed input density matrix
    rho_mixed = np.array([[0.72, 0.18 - 0.21j], [0.18 + 0.21j, 0.28]], dtype=np.complex128)
    ev_mix = kc.apply_bell_channel(u_max, rho_mixed, "Phi+")
    assert ev_mix.success_probability == pytest.approx(0.25, abs=1e-14)
    assert np.allclose(ev_mix.normalized_rho_out, rho_mixed, atol=1e-13)


@pytest.mark.parametrize(
    "theta,expect_non_clifford",
    [
        (0.0, False),
        (math.pi / 8.0, True),
        (math.pi / 4.0, True),
        (3.0 * math.pi / 8.0, True),
        (math.pi / 2.0, False),
    ],
)
def test_intermediate_theta_non_clifford_magic(theta, expect_non_clifford):
    """Intermediate angles in (0, pi/2) produce genuine non-Clifford unitaries with positive magic M_2."""
    u = kc.landsman_figgatt_scrambler(theta)
    assert np.allclose(u.conj().T @ u, np.eye(8), atol=1e-13)
    non_cliff, magic_m2 = kc.is_non_clifford_unitary(u)
    assert non_cliff is expect_non_clifford
    if expect_non_clifford:
        assert magic_m2 > 2.0
    else:
        assert magic_m2 == pytest.approx(0.0, abs=1e-12)


def test_analytical_closed_form_and_yoshida_kitaev_bound():
    """Verify closed-form P_{Phi+}(theta), F_{6state}(theta), and P_{Phi+} * F_ent == 1/4."""
    for k in range(9):
        theta = k * math.pi / 16.0
        u = kc.landsman_figgatt_scrambler(theta)
        summary = kc.evaluate_6state_summary(u, "Phi+")
        assert summary["p_success"] == pytest.approx(kc.analytical_p_phi_plus(theta), abs=1e-13)
        assert summary["fidelity_6state_avg"] == pytest.approx(
            kc.analytical_fidelity_6state(theta), abs=1e-13
        )
        assert summary["yk_product_p_times_f_ent"] == pytest.approx(0.25, abs=1e-13)


def test_scrambling_vs_decoherence_discriminator():
    """Pure decoherence decays P_{Phi+} to 0.25 with F=0.50; coherent noisy scrambling lifts F > 0.95."""
    disc = kc.scrambling_vs_decoherence_discriminator()
    pure_dec = disc["pure_decoherence_no_scrambling"]
    assert pure_dec[0]["p_phi_plus"] == pytest.approx(1.0, abs=1e-12)
    assert pure_dec[-1]["p_phi_plus"] == pytest.approx(0.25, abs=1e-12)
    for pt in pure_dec:
        assert pt["fidelity_6state"] == pytest.approx(0.5, abs=1e-12)
        assert pt["verified_scrambling_witness"] == pytest.approx(0.0, abs=1e-12)

    noisy_scr = disc["coherent_noisy"]
    for pt in noisy_scr[1:]:
        assert pt["fidelity_6state"] > 0.52
        assert pt["verified_scrambling_witness"] > 0.02
    assert noisy_scr[-1]["fidelity_6state"] > 0.95
    assert disc["separation_summary"]["separates_scrambling_from_decoherence"] is True


# ---------------------------------------------------------------------------
# 2. H2-1 QCCD Circuits (`qccd_circuits.py`) & Exact Kraus Agreement
# ---------------------------------------------------------------------------


def test_p0_and_e1_h2_isa_compliance_and_zero_swaps(h2_offline_backend):
    """Both P0 (7Q) and E1 / ENCODED_422_REF (12Q) compile to native H2-1 ISA with 0 SWAPs."""
    b_p0 = qc6.build_xtop006_circuit("PHYSICAL_7Q", math.pi / 4.0, "+i", backend=h2_offline_backend)
    c_p0 = qc6.compiled_gate_counts(b_p0.compiled_tket)
    assert c_p0["qubits"] == 7
    assert c_p0["routing_swaps"] == 0
    assert c_p0["N_2q_zzphase"] > 0  # non-Clifford continuous-angle ZZPhase gates preserved!

    b_e1 = qc6.build_xtop006_circuit("ENCODED_422_REF", math.pi / 4.0, "+i", backend=h2_offline_backend)
    c_e1 = qc6.compiled_gate_counts(b_e1.compiled_tket)
    assert c_e1["qubits"] == 12
    assert c_e1["routing_swaps"] == 0
    assert c_e1["N_2q_zzphase"] > 0
    assert c_e1["N_reset"] >= 5


@pytest.mark.parametrize(
    "theta,state_label",
    [
        (0.0, "0"),
        (math.pi / 8.0, "+"),
        (math.pi / 4.0, "+i"),
        (3.0 * math.pi / 8.0, "-"),
        (math.pi / 2.0, "-i"),
    ],
)
def test_p0_compiled_circuit_matches_analytical_kraus_all_bell_outcomes(
    theta, state_label, h2_offline_backend
):
    """Compiled 7Q H2-1 P0 circuit matches analytical 2x2 output density matrices across all 4 Bell outcomes."""
    ver = qc6.verify_p0_compiled_against_kraus(theta, state_label, backend=h2_offline_backend)
    assert ver["passed"] is True
    assert ver["max_unnormalized_rho_frobenius_error"] < 1e-12
    assert ver["max_probability_error"] < 1e-12


@pytest.mark.parametrize(
    "theta,state_label",
    [
        (0.0, "+"),
        (math.pi / 4.0, "1"),
        (math.pi / 4.0, "+i"),
        (math.pi / 2.0, "0"),
        (math.pi / 2.0, "-"),
        (math.pi / 2.0, "-i"),
    ],
)
def test_e1_compiled_422_circuit_matches_analytical_kraus_noiseless(
    theta, state_label, h2_offline_backend
):
    """Compiled 12Q H2-1 [[4,2,2]] E1 circuit has p_keep=1.0 and matches analytical Kraus P_{Phi+}, F."""
    b_e1 = qc6.build_xtop006_circuit("E1", theta, state_label, backend=h2_offline_backend)
    ver = qc6.verify_e1_compiled_against_kraus(b_e1)
    assert ver["passed"] is True
    assert ver["p_keep"] == pytest.approx(1.0, abs=1e-12)
    assert ver["prob_error_vs_kraus"] < 1e-12
    assert ver["fidelity_error_vs_kraus"] < 1e-12


def test_e1_detects_every_single_qubit_storage_error(h2_offline_backend):
    """Injecting any single-qubit X, Y, or Z fault on B_out (q6..q9) during storage is 100% rejected."""
    from pytket.extensions.qiskit import tk_to_qiskit
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    sim = AerSimulator()
    b_e1 = qc6.build_xtop006_circuit("E1", math.pi / 2.0, "+i", backend=h2_offline_backend)
    base_qc = tk_to_qiskit(b_e1.compiled_tket, replace_implicit_swaps=True)

    for mem_q in b_e1.memory_qubits:
        for pauli_gate in ("x", "y", "z"):
            fault_qc = QuantumCircuit(*base_qc.qregs, *base_qc.cregs)
            barriers = 0
            for inst in base_qc.data:
                if inst.operation.name == "barrier":
                    barriers += 1
                    if barriers == 2:
                        getattr(fault_qc, pauli_gate)(mem_q)
                fault_qc.append(inst)
            counts = sim.run(fault_qc, shots=64, seed_simulator=99).result().get_counts()
            dec = qc6._decode_shot_counts(counts, b_e1)
            assert dec["code_accepted_shots"] == 0.0, f"Undetected {pauli_gate} on q{mem_q}"


def test_xtop006_manifest_jcs_and_encoded_advantage():
    """Verify xtop006_h2_manifest.json JCS canonicalization, SHA256SUMS, and E1 > P0 under storage noise."""
    manifest_path = qc6.ARTIFACTS_DIR / qc6.MANIFEST_FILENAME
    sums_path = qc6.ARTIFACTS_DIR / qc6.SHA256SUMS_FILENAME
    raw_bytes = manifest_path.read_bytes()
    manifest = json.loads(raw_bytes.decode("utf-8"))

    assert raw_bytes == rfc8785.dumps(manifest)
    assert manifest["evidence_class"] == "SIMULATED"
    assert manifest["provider_job_id"] is None
    assert manifest["hardware_executed"] is False
    assert manifest["n_circuits"] == 60

    for line in sums_path.read_text(encoding="utf-8").splitlines():
        digest, rel = line.strip().split("  ", 1)
        assert rnh._sha256_bytes((qc6.ARTIFACTS_DIR / rel).read_bytes()) == digest

    pi2_cell = next(r for r in manifest["theta_sweep_results"] if r["theta_name"] == "pi_2")
    f_e1 = float(pi2_cell["arms"]["E1"]["h2_with_qccd_memory_dephasing"]["fidelity_6state_avg"])
    f_p0 = float(pi2_cell["arms"]["P0"]["h2_with_qccd_memory_dephasing"]["fidelity_6state_avg"])
    assert f_e1 > f_p0 + 0.02


# ---------------------------------------------------------------------------
# 3. OSIRIS-XTOP-005 Hash-Locked Decisive H2 Bundle & Runner (`xtop005/`)
# ---------------------------------------------------------------------------


def test_xtop005_decisive_manifest_jcs_and_hqc_budget_1336():
    """Verify 18-circuit L=3 decisive manifest JCS lock, HQC=1,336 budget, and Stage-1 predictions."""
    m_path = rnh.BUNDLE_DIR / rnh.MANIFEST_NAME
    raw = m_path.read_bytes()
    m = json.loads(raw.decode("utf-8"))

    assert raw == rfc8785.dumps(m)
    assert m["evidence_class"] == "OFFLINE_COMPILED"
    assert m["provider_job_id"] is None
    assert m["hardware_executed"] is False
    assert m["depth"] == 3
    assert m["shots_per_circuit"] == 500
    assert m["n_circuits"] == 18
    assert m["hqc_budget"] == 1336
    assert m["hqc_total_conservative"] == "1335.6000"
    assert m["arms"]["Q0"]["totals_9_circuits"]["routing_swaps"] == 0
    assert m["arms"]["Q3"]["totals_9_circuits"]["routing_swaps"] == 0
    assert float(m["arms"]["Q3"]["stage1_predicted"]["F_used"]) > float(
        m["arms"]["Q0"]["stage1_predicted"]["F_used"]
    )


def test_run_nexus_h2_decisive_dry_run_cli(capsys):
    """`run_nexus_h2_decisive.py --dry-run` verifies all 18 circuit hashes and Stage-0 F=1.000000."""
    rc = rnh.main(["--dry-run"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "OFFLINE_VERIFIED"
    assert out["evidence_class"] == "OFFLINE_COMPILED"
    assert out["provider_job_id"] is None
    assert out["hqc_budget"] == 1336
    assert out["stage0"]["Q0"]["stage0_F_raw"] == "1.000000"
    assert out["stage0"]["Q3"]["stage0_F_acc"] == "1.000000"
    assert out["stage0"]["Q0"]["stage0_rejected_shots"] == 0
    assert out["stage0"]["Q3"]["stage0_rejected_shots"] == 0


def test_run_nexus_h2_decisive_tamper_detection_fails_closed(tmp_path, capsys):
    """Mutating a locked .tket.json circuit causes `--dry-run` to fail closed with exit code 2."""
    copied = tmp_path / "nexus_h2_bundle"
    shutil.copytree(rnh.BUNDLE_DIR, copied)
    victim = copied / "circuits" / "xtop005q_Q0_L3_XX.tket.json"
    data = json.loads(victim.read_text(encoding="utf-8"))
    data["name"] = "tampered_name"
    victim.write_bytes(rfc8785.dumps(data))

    rc = rnh.main(["--bundle-dir", str(copied), "--dry-run"])
    assert rc == 2
    assert "SHA-256 mismatch" in capsys.readouterr().err


def test_run_nexus_h2_decisive_submit_fails_closed_without_token_or_qnexus(tmp_path, monkeypatch, capsys):
    """`--submit` fails closed when token is absent, invalid, or qnexus/credentials are missing."""
    monkeypatch.setenv("OSIRIS_HOME", str(tmp_path))
    monkeypatch.delenv("QUANTINUUM_NEXUS_TOKEN", raising=False)

    # 1. Missing operator approval token
    assert rnh.main(["--submit", "--device", "H2-1E"]) == 2
    assert "operator-approval-token" in capsys.readouterr().err

    # 2. Tier 3 target with bogus token fails closed before qnexus lookup
    assert rnh.main(["--submit", "--device", "H2-1", "--operator-approval-token", "forged"]) == 2
    assert "BLOCKED_APPROVAL_TOKEN" in capsys.readouterr().err

    # 3. Tier 2 target with approval token fails closed when qnexus is not installed
    assert rnh.main(["--submit", "--device", "H2-1E", "--operator-approval-token", "op-emu-ok"]) == 2
    err = capsys.readouterr().err
    assert "qnexus" in err or "Nexus credentials absent" in err


class _MockRef:
    def __init__(self, payload=None):
        self.id = uuid.uuid4()
        self.payload = payload

    def download_circuit(self):
        return self.payload.copy()


class _MockCompileItem:
    def __init__(self, ref):
        self._ref = ref

    def get_output(self):
        return self._ref


class _MockExecItem:
    def __init__(self, device_name):
        self._dev = device_name

    def download_result(self):
        return type("R", (), {"get_counts": lambda self_: Counter({(0, 0, 0, 0): 500})})()

    def download_backend_info(self):
        return type("BI", (), {"device_name": self._dev})()


class _MockQnexus:
    def __init__(self, report_device=None, report_status="COMPLETED"):
        self.projects = self.circuits = self.jobs = self
        self._jobs = {}
        self._devs = {}
        self.report_device = report_device
        self.report_status = report_status

    def QuantinuumConfig(self, *, device_name, **kw):
        return {"device_name": device_name, **kw}

    def get_or_create(self, *, name, description=None):
        return _MockRef()

    def upload(self, *, circuit, project=None, name=None):
        return _MockRef(circuit)

    def start_compile_job(self, *, programs, backend_config, name, project=None, optimisation_level=0):
        j = _MockRef()
        self._jobs[j.id] = [_MockCompileItem(_MockRef(p.payload)) for p in programs]
        return j

    def cost(self, *, circuit_ref, n_shots, backend_config, syntax_checker=None, project=None):
        assert syntax_checker == backend_config["device_name"] + "SC"
        return float(rnh.local_hqc(circuit_ref.payload, n_shots))

    def start_execute_job(self, *, programs, n_shots, backend_config, name, project=None, max_cost=()):
        j = _MockRef()
        dev = self.report_device or backend_config["device_name"]
        self._devs[j.id] = dev
        self._jobs[j.id] = [_MockExecItem(dev) for _ in programs]
        return j

    def wait_for(self, *, job, timeout=None):
        return None

    def results(self, *, job):
        return self._jobs[job.id]

    def status(self, *, job):
        st_obj = type("E", (), {"value": self.report_status})()
        return type("St", (), {"status": st_obj, "completed_time": None, "cost": 1332.0})()

    def get(self, *, id=None):
        cfg = type("Cfg", (), {"device_name": self._devs[id]})()
        return type("JR", (), {"backend_config_store": cfg})()


def test_run_nexus_h2_decisive_submit_governance_and_attestation(tmp_path, monkeypatch):
    """Emulator never emits HARDWARE_ATTESTED; Tier 3 emits HARDWARE_ATTESTED iff Nexus confirms QPU."""
    monkeypatch.setenv("OSIRIS_HOME", str(tmp_path))
    monkeypatch.delenv("OSIRIS_TIER3_HMAC_SECRET", raising=False)

    # 1. H2-1E emulator run -> SIMULATED (never HARDWARE_ATTESTED)
    emu_res = rnh.submit_decisive_bundle(
        device="H2-1E",
        operator_approval_token="op-approved-h2-1e",
        osiris_home=str(tmp_path),
        qnx_override=_MockQnexus(),
    )
    assert emu_res["evidence_class"] == "SIMULATED"
    assert emu_res["execution"]["disposition"] == "SIMULATED"

    # 2. H2-1E cap exceeded -> GovernanceError
    with pytest.raises(GovernanceError, match="max-emulator-hqc"):
        rnh.submit_decisive_bundle(
            device="H2-1E",
            operator_approval_token="op-approved-h2-1e",
            max_emulator_hqc=200.0,
            osiris_home=str(tmp_path),
            qnx_override=_MockQnexus(),
        )

    # 3. H2-1 Tier 3 live QPU with valid signed token -> HARDWARE_ATTESTED only when device==H2-1 & COMPLETED
    manifest = json.loads((rnh.BUNDLE_DIR / rnh.MANIFEST_NAME).read_text(encoding="utf-8"))
    t3_tok = mint_tier3_token(
        circuit_sha256=manifest["compiled_batch_sha256"],
        target="H2-1",
        max_shots=500,
        max_hqc="1400",
        osiris_home=str(tmp_path),
    )
    qpu_ok = rnh.submit_decisive_bundle(
        device="H2-1",
        operator_approval_token=t3_tok,
        osiris_home=str(tmp_path),
        qnx_override=_MockQnexus(report_device="H2-1", report_status="COMPLETED"),
    )
    assert qpu_ok["evidence_class"] == "HARDWARE_ATTESTED"
    assert qpu_ok["execution"]["disposition"] == "HARDWARE_ATTESTED"

    # 4. If Nexus reports emulator device instead of H2-1 -> NEVER HARDWARE_ATTESTED!
    qpu_spoof = rnh.submit_decisive_bundle(
        device="H2-1",
        operator_approval_token=t3_tok,
        osiris_home=str(tmp_path),
        qnx_override=_MockQnexus(report_device="H2-1E", report_status="COMPLETED"),
    )
    assert qpu_spoof["evidence_class"] == "SIMULATED"
    assert qpu_spoof["execution"]["disposition"] == "HARDWARE_UNVERIFIED"
