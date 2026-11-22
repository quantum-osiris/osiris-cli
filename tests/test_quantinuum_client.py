"""Offline deterministic test suite for TKET lowering, Quantinuum Nexus client, and CLI."""

from __future__ import annotations

import hashlib
import io
import json
import re
import stat
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from osiris_cli import cli, osiris_repl
from osiris_cli.providers.quantinuum_nexus import (
    GovernanceError,
    NexusDisposition,
    QuantinuumNexusClient,
    compute_disposition_receipt_sha256,
    estimate_hqc,
    sign_tier3_approval_token,
    verify_disposition_receipt,
    verify_tier3_approval_token,
)
from osiris_cli.providers.tket_wrapper import (
    QuantinuumCompilerOptions,
    compile_for_quantinuum,
    lower_qasm3_to_quantinuum_qasm2,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_QEC = REPO_ROOT / "examples" / "quantinuum" / "bitflip_qec.qasm"

SAMPLE_QASM3_MID_CIRCUIT = """\
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit[3] c;
h q[0];
cx q[0], q[1];
c[0] = measure q[0];
reset q[0];
if (c[0] == 1) {
    x q[1];
    z q[2];
}
c[1] = measure q[1];
c[2] = measure q[2];
"""


def _make_n_qubit_qasm3(n: int) -> str:
    return (
        'OPENQASM 3.0;\n'
        'include "stdgates.inc";\n'
        f"qubit[{n}] q;\n"
        f"bit[{n}] c;\n"
        "h q[0];\n"
        "c[0] = measure q[0];\n"
    )


class _MockNexusServer:
    """Deterministic offline mock for Quantinuum Nexus HTTP endpoints."""

    def __init__(
        self,
        *,
        preflight_status: str = "COMPLETED",
        preflight_errors: Optional[List[str]] = None,
        target_status_sequence: Optional[List[str]] = None,
        target_job_id: str = "nexus-job-00042",
        target_counts: Optional[Dict[str, int]] = None,
        targets_list: Optional[List[str]] = None,
    ) -> None:
        self.calls: List[Tuple[str, str, Dict[str, str], Optional[Dict[str, Any]]]] = []
        self.preflight_status = preflight_status
        self.preflight_errors = preflight_errors or []
        self.target_status_sequence = list(target_status_sequence or ["QUEUED", "COMPLETED"])
        self.target_job_id = target_job_id
        self.target_counts = target_counts or {"000": 60, "111": 40}
        self.targets_list = targets_list or ["H2-1SC", "H2-1E", "H2-1", "Helios-E", "Helios"]
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._seq_idx: Dict[str, int] = {}

    def __call__(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        payload: Optional[Dict[str, Any]],
    ) -> Tuple[int, Dict[str, Any]]:
        self.calls.append((method, url, dict(headers), payload))
        if method == "GET" and url.endswith("/targets"):
            return 200, {"targets": self.targets_list}

        if method == "POST" and url.endswith("/jobs"):
            assert payload is not None
            target = payload["target"]
            if target.endswith("SC"):
                jid = f"sc-{target.lower()}-001"
                resp: Dict[str, Any] = {
                    "job_id": jid,
                    "target": target,
                    "status": self.preflight_status,
                }
                if self.preflight_errors:
                    resp["errors"] = list(self.preflight_errors)
                self._jobs[jid] = resp
                return 200, resp

            jid = self.target_job_id
            initial = self.target_status_sequence[0]
            self._seq_idx[jid] = 0
            self._jobs[jid] = {
                "job_id": jid,
                "target": target,
                "status": initial,
                "counts": self.target_counts if initial == "COMPLETED" else {},
            }
            return 200, dict(self._jobs[jid])

        if method == "GET" and "/jobs/" in url:
            jid = url.rsplit("/jobs/", 1)[1]
            if jid not in self._jobs:
                return 404, {"error": f"job {jid} not found"}
            idx = self._seq_idx.get(jid, 0) + 1
            seq = self.target_status_sequence
            status = seq[min(idx, len(seq) - 1)]
            self._seq_idx[jid] = idx
            rec = dict(self._jobs[jid])
            rec["status"] = status
            if status == "COMPLETED":
                rec["counts"] = self.target_counts
            self._jobs[jid] = rec
            return 200, rec

        return 404, {"error": f"unhandled mock route {method} {url}"}


# ---------------------------------------------------------------------------
# 1. OpenQASM 3.0 -> Quantinuum QASM 2.0 Lowering & Compiler Options
# ---------------------------------------------------------------------------


def test_lower_qasm3_preserves_mid_circuit_measure_reset_and_classical_control() -> None:
    qasm2 = lower_qasm3_to_quantinuum_qasm2(SAMPLE_QASM3_MID_CIRCUIT)
    assert "OPENQASM 2.0;" in qasm2
    assert 'include "qelib1.inc";' in qasm2
    assert "qreg q[3];" in qasm2
    assert "creg c[3];" in qasm2
    assert "measure q[0] -> c[0];" in qasm2
    assert "reset q[0];" in qasm2
    assert "if(c[0]==1) x q[1];" in qasm2
    assert "if(c[0]==1) z q[2];" in qasm2
    assert "measure q[1] -> c[1];" in qasm2
    assert "measure q[2] -> c[2];" in qasm2

    result = compile_for_quantinuum(SAMPLE_QASM3_MID_CIRCUIT, target="H2-1SC")
    assert result.target == "H2-1SC"
    assert result.compiler_backend in ("pytket", "static_fallback")
    assert len(result.source_sha256) == 64
    assert len(result.compiled_sha256) == 64
    assert result.metrics.n_qubits == 3
    assert result.metrics.n_cbits == 3
    assert result.metrics.two_qubit_gates == 1
    assert result.metrics.has_mid_circuit_measurement is True
    assert result.metrics.has_reset is True
    assert result.metrics.has_classical_control is True


def test_lower_example_bitflip_qec_qasm() -> None:
    source = EXAMPLE_QEC.read_text(encoding="utf-8")
    res = compile_for_quantinuum(source, target="H2-1SC")
    assert res.metrics.n_qubits == 5
    assert res.metrics.n_cbits == 5
    assert res.metrics.two_qubit_gates == 6
    assert res.metrics.has_mid_circuit_measurement is True
    assert res.metrics.has_reset is True
    assert res.metrics.has_classical_control is True
    lowered = lower_qasm3_to_quantinuum_qasm2(source)
    assert "if(syn==1) x data[0];" in lowered
    assert "if(syn==3) x data[1];" in lowered
    assert "if(syn==2) x data[2];" in lowered
    assert "reset anc[0];" in lowered
    # pytket may rewrite x as u3; the compiled circuit must keep every correction and its condition.
    assert res.metrics.feedforward_conditionals == 3
    assert sorted(re.findall(r"if\(syn==(\d)\) \S+ data\[(\d)\]", res.qasm2)) == [("1", "0"), ("2", "2"), ("3", "1")]
    assert "reset anc[0];" in res.qasm2


def test_lower_dnalang_minimal_circuit() -> None:
    dna = """\
    organism BitFlipGene {
        qubits 2
        bits 2
        HELIX(0)
        BOND(0, 1)
        MEASURE(0, 0)
        RESET(0)
        IF(c[0] == 1) X(1)
        MEASURE(1, 1)
    }
    """
    lowered = lower_qasm3_to_quantinuum_qasm2(dna)
    assert "qreg q[2];" in lowered
    assert "creg c[2];" in lowered
    assert "h q[0];" in lowered
    assert "cx q[0], q[1];" in lowered
    assert "measure q[0] -> c[0];" in lowered
    assert "reset q[0];" in lowered
    assert "if(c[0]==1) x q[1];" in lowered
    res = compile_for_quantinuum(dna, target="H2-1SC")
    assert "measure q[0] -> c[0];" in res.qasm2
    assert "reset q[0];" in res.qasm2
    assert re.search(r"if\(c\[0\]==1\) \S+ q\[1\];", res.qasm2)
    assert res.metrics.feedforward_conditionals == 1
    assert res.metrics.has_mid_circuit_measurement is True
    assert res.metrics.has_reset is True
    assert res.metrics.has_classical_control is True


def test_compiler_options_control_redundancy_reduction() -> None:
    redundant = """\
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[2] q;
    bit[2] c;
    h q[0];
    x q[0];
    x q[0];
    rz(0) q[1];
    cx q[0], q[1];
    c[0] = measure q[0];
    """
    opt_res = compile_for_quantinuum(
        redundant,
        target="H2-1SC",
        options=QuantinuumCompilerOptions(tket_opt_level=2),
    )
    no_opt_res = compile_for_quantinuum(
        redundant,
        target="H2-1SC",
        options=QuantinuumCompilerOptions(no_opt=True),
    )
    no_reduce_res = compile_for_quantinuum(
        redundant,
        target="H2-1SC",
        options=QuantinuumCompilerOptions(no_reduce=True),
    )
    assert opt_res.metrics.total_gates < no_opt_res.metrics.total_gates
    assert no_opt_res.metrics.total_gates == 5
    assert no_reduce_res.metrics.total_gates == 5


# ---------------------------------------------------------------------------
# 2. Qubit Cap Enforcement (>56 on H2-1, >98 on Helios)
# ---------------------------------------------------------------------------


def test_qubit_limit_rejected_on_h2_1_over_56() -> None:
    res_56 = compile_for_quantinuum(_make_n_qubit_qasm3(56), target="H2-1")
    assert res_56.metrics.n_qubits == 56

    for tgt in ("H2-1", "H2-1SC", "H2-1E", "H2-2", "H2-2SC", "H2-2E"):
        with pytest.raises(ValueError, match="exceeding target .* cap of 56"):
            compile_for_quantinuum(_make_n_qubit_qasm3(57), target=tgt)


def test_qubit_limit_rejected_on_helios_over_98() -> None:
    res_98 = compile_for_quantinuum(_make_n_qubit_qasm3(98), target="Helios")
    assert res_98.metrics.n_qubits == 98

    for tgt in ("Helios", "Helios-E"):
        with pytest.raises(ValueError, match="exceeding target .* cap of 98"):
            compile_for_quantinuum(_make_n_qubit_qasm3(99), target=tgt)


def test_nexus_client_fails_closed_before_network_on_qubit_overflow() -> None:
    server = _MockNexusServer()
    client = QuantinuumNexusClient(http_request=server)
    with pytest.raises(GovernanceError, match="exceeding target"):
        client.execute(_make_n_qubit_qasm3(57), target="H2-1SC", shots=10)
    assert server.calls == []


# ---------------------------------------------------------------------------
# 3. Fail-Closed Governance Controls (Shots, HQC Budget, Tier 3 Approval)
# ---------------------------------------------------------------------------


def test_estimate_hqc_zero_for_tier1_and_positive_for_tier2_tier3() -> None:
    comp = compile_for_quantinuum(SAMPLE_QASM3_MID_CIRCUIT, target="H2-1SC")
    assert estimate_hqc(comp, shots=1000, target="H2-1SC") == 0.0
    assert estimate_hqc(comp, shots=1000, target="H2-2SC") == 0.0

    hqc_e = estimate_hqc(comp, shots=500, target="H2-1E")
    hqc_hw = estimate_hqc(comp, shots=500, target="H2-1")
    assert hqc_e == hqc_hw
    assert hqc_hw > 5.0


def test_governance_error_when_shots_exceed_cap() -> None:
    server = _MockNexusServer()
    client = QuantinuumNexusClient(max_shots=100, budget_hqc=500.0, http_request=server)
    with pytest.raises(GovernanceError, match="exceed max_shots cap"):
        client.execute(SAMPLE_QASM3_MID_CIRCUIT, target="H2-1E", shots=101)
    assert server.calls == []


def test_governance_error_when_hqc_exceeds_budget() -> None:
    server = _MockNexusServer()
    client = QuantinuumNexusClient(max_shots=10000, budget_hqc=5.1, http_request=server)
    with pytest.raises(GovernanceError, match="exceeds budget_hqc cap"):
        client.execute(SAMPLE_QASM3_MID_CIRCUIT, target="H2-1E", shots=500)
    assert server.calls == []


def test_tier3_approval_token_fail_closed_matrix() -> None:
    secret = "unit-test-tier3-secret-key"
    comp = compile_for_quantinuum(SAMPLE_QASM3_MID_CIRCUIT, target="H2-1")
    now = 1_750_000_000
    valid_token = sign_tier3_approval_token(
        target="H2-1",
        compiled_sha256=comp.compiled_sha256,
        shots=100,
        secret=secret,
        issued_at=now,
        expires_at=now + 600,
    )

    # Sanity check: valid token passes verification directly.
    verified = verify_tier3_approval_token(
        valid_token,
        target="H2-1",
        compiled_sha256=comp.compiled_sha256,
        shots=100,
        secret=secret,
        now=now + 10,
    )
    assert verified["target"] == "H2-1"

    server = _MockNexusServer()
    client = QuantinuumNexusClient(
        approval_secret=secret,
        max_shots=1000,
        budget_hqc=100.0,
        http_request=server,
    )

    # 1. Missing token
    with pytest.raises(GovernanceError, match="requires a valid --approval-token"):
        client.execute(SAMPLE_QASM3_MID_CIRCUIT, target="H2-1", shots=100, approval_token=None, now=now)

    # 2. Tampered signature
    tampered = valid_token[:-1] + ("0" if valid_token[-1] != "0" else "1")
    with pytest.raises(GovernanceError, match="signature verification failed"):
        client.execute(SAMPLE_QASM3_MID_CIRCUIT, target="H2-1", shots=100, approval_token=tampered, now=now)

    # 3. Expired token
    with pytest.raises(GovernanceError, match="expired"):
        client.execute(
            SAMPLE_QASM3_MID_CIRCUIT,
            target="H2-1",
            shots=100,
            approval_token=valid_token,
            now=now + 601,
        )

    # 4. Wrong target (token signed for H2-1, used on Helios)
    with pytest.raises(GovernanceError, match="target mismatch"):
        client.execute(
            SAMPLE_QASM3_MID_CIRCUIT,
            target="Helios",
            shots=100,
            approval_token=valid_token,
            now=now + 10,
        )

    # 5. Wrong circuit hash (circuit modified after token signed)
    mutated_qasm = SAMPLE_QASM3_MID_CIRCUIT.replace("z q[2];", "y q[2];")
    with pytest.raises(GovernanceError, match="compiled_sha256 mismatch"):
        client.execute(
            mutated_qasm,
            target="H2-1",
            shots=100,
            approval_token=valid_token,
            now=now + 10,
        )

    # 6. Wrong shots
    with pytest.raises(GovernanceError, match="shots mismatch"):
        client.execute(
            SAMPLE_QASM3_MID_CIRCUIT,
            target="H2-1",
            shots=150,
            approval_token=valid_token,
            now=now + 10,
        )

    # Zero remote Nexus calls occurred across all failed governance checks.
    assert server.calls == []


def test_tier2_and_tier3_fail_closed_if_preflight_syntax_check_fails() -> None:
    secret = "unit-test-tier3-secret-key"
    comp = compile_for_quantinuum(SAMPLE_QASM3_MID_CIRCUIT, target="H2-1")
    now = 1_750_000_000
    token = sign_tier3_approval_token(
        target="H2-1",
        compiled_sha256=comp.compiled_sha256,
        shots=50,
        secret=secret,
        issued_at=now,
        expires_at=now + 600,
    )

    server = _MockNexusServer(preflight_status="FAILED", preflight_errors=["invalid gate"])
    client = QuantinuumNexusClient(approval_secret=secret, http_request=server)

    with pytest.raises(GovernanceError, match="Syntax pre-flight failed on H2-1SC"):
        client.execute(SAMPLE_QASM3_MID_CIRCUIT, target="H2-1E", shots=50)
    assert len(server.calls) == 1
    assert server.calls[0][3]["target"] == "H2-1SC"

    with pytest.raises(GovernanceError, match="Syntax pre-flight failed on H2-1SC"):
        client.execute(
            SAMPLE_QASM3_MID_CIRCUIT,
            target="H2-1",
            shots=50,
            approval_token=token,
            now=now + 5,
        )
    # Only syntax checker calls were ever attempted; never submitted H2-1E or H2-1.
    assert all(call[3]["target"] == "H2-1SC" for call in server.calls if call[0] == "POST")


# ---------------------------------------------------------------------------
# 4. Strict Attestation & Receipt Integrity Guarantees
# ---------------------------------------------------------------------------


def test_h2_1sc_and_h2_1e_never_produce_hardware_attested() -> None:
    server = _MockNexusServer()
    client = QuantinuumNexusClient(http_request=server)

    disp_sc = client.execute(SAMPLE_QASM3_MID_CIRCUIT, target="H2-1SC", shots=100)
    assert disp_sc.tier == 1
    assert disp_sc.hardware is False
    assert disp_sc.disposition == "SIMULATED"
    assert disp_sc.estimated_hqc == 0.0
    assert verify_disposition_receipt(disp_sc.to_dict()) is True

    disp_emu = client.execute(SAMPLE_QASM3_MID_CIRCUIT, target="H2-1E", shots=100)
    assert disp_emu.tier == 2
    assert disp_emu.hardware is False
    assert disp_emu.disposition == "SIMULATED"
    assert disp_emu.preflight_target == "H2-1SC"
    assert disp_emu.preflight_passed is True
    assert disp_emu.counts == {"000": 60, "111": 40}
    assert verify_disposition_receipt(disp_emu.to_dict()) is True

    # Confirm direct construction of forged Tier 1/2 hardware disposition fails closed.
    for tier, tgt in ((1, "H2-1SC"), (2, "H2-1E")):
        with pytest.raises(GovernanceError, match="can never produce hardware=True"):
            NexusDisposition(
                target=tgt,
                tier=tier,
                disposition="HARDWARE_ATTESTED",
                hardware=True,
                status="COMPLETED",
                job_id="forged-1",
                shots=100,
                estimated_hqc=1.0,
                source_sha256="a" * 64,
                compiled_sha256="b" * 64,
                compiler_backend="static_fallback",
                preflight_target="H2-1SC",
                preflight_job_id="sc-1",
                preflight_passed=True,
            )


def test_tier3_completed_run_produces_hardware_attested_and_detects_tampering() -> None:
    secret = "tier3-production-signing-key"
    comp = compile_for_quantinuum(SAMPLE_QASM3_MID_CIRCUIT, target="H2-1")
    now = 1_750_000_000
    token = sign_tier3_approval_token(
        target="H2-1",
        compiled_sha256=comp.compiled_sha256,
        shots=100,
        secret=secret,
        issued_at=now,
        expires_at=now + 600,
    )

    server = _MockNexusServer(
        target_status_sequence=["QUEUED", "RUNNING", "COMPLETED"],
        target_job_id="quantinuum-h2-1-job-9981",
        target_counts={"000": 51, "111": 49},
    )
    client = QuantinuumNexusClient(
        approval_secret=secret,
        max_shots=1000,
        budget_hqc=100.0,
        http_request=server,
    )

    disp = client.execute(
        SAMPLE_QASM3_MID_CIRCUIT,
        target="H2-1",
        shots=100,
        approval_token=token,
        now=now + 5,
    )
    assert disp.tier == 3
    assert disp.hardware is True
    assert disp.disposition == "HARDWARE_ATTESTED"
    assert disp.job_id == "quantinuum-h2-1-job-9981"
    assert disp.preflight_target == "H2-1SC"
    assert disp.preflight_passed is True
    assert len(disp.receipt_sha256) == 64

    # Verify pre-flight H2-1SC happened strictly BEFORE H2-1 submission.
    post_targets = [c[3]["target"] for c in server.calls if c[0] == "POST"]
    assert post_targets == ["H2-1SC", "H2-1"]

    rec = disp.to_dict()
    assert verify_disposition_receipt(rec) is True

    # Tamper detection on counts, shots, disposition, and compiled_sha256.
    for field_name, bad_val in (
        ("counts", {"000": 100}),
        ("shots", 200),
        ("compiled_sha256", "0" * 64),
        ("job_id", "fake-job"),
    ):
        tampered = dict(rec)
        tampered[field_name] = bad_val
        assert verify_disposition_receipt(tampered) is False

    # Semantic forgery: even with recomputed SHA-256, Tier 2 claiming hardware=True fails!
    forged_t2 = dict(rec)
    forged_t2["tier"] = 2
    forged_t2["target"] = "H2-1E"
    forged_t2["receipt_sha256"] = compute_disposition_receipt_sha256(forged_t2)
    assert verify_disposition_receipt(forged_t2) is False


def test_tier3_refuses_hardware_attestation_if_job_fails_or_missing_job_id() -> None:
    secret = "tier3-secret"
    comp = compile_for_quantinuum(SAMPLE_QASM3_MID_CIRCUIT, target="H2-1")
    now = 1_750_000_000
    token = sign_tier3_approval_token(
        target="H2-1",
        compiled_sha256=comp.compiled_sha256,
        shots=50,
        secret=secret,
        issued_at=now,
        expires_at=now + 600,
    )

    failed_server = _MockNexusServer(target_status_sequence=["RUNNING", "FAILED"])
    client_fail = QuantinuumNexusClient(approval_secret=secret, http_request=failed_server)
    with pytest.raises(GovernanceError, match="refusing hardware attestation"):
        client_fail.execute(
            SAMPLE_QASM3_MID_CIRCUIT,
            target="H2-1",
            shots=50,
            approval_token=token,
            now=now + 5,
        )

    no_id_server = _MockNexusServer(target_job_id="")
    client_no_id = QuantinuumNexusClient(approval_secret=secret, http_request=no_id_server)
    with pytest.raises(GovernanceError, match="no valid job_id"):
        client_no_id.execute(
            SAMPLE_QASM3_MID_CIRCUIT,
            target="H2-1",
            shots=50,
            approval_token=token,
            now=now + 5,
        )


def test_connection_probe_sanitizes_secrets_and_supports_opener() -> None:
    class _FakeResp(io.BytesIO):
        status = 200

        def __enter__(self) -> "_FakeResp":
            return self

        def __exit__(self, *args: Any) -> None:
            self.close()

    class _FakeOpener:
        def __init__(self) -> None:
            self.urls: List[str] = []

        def open(self, req: Any, timeout: float = 10.0) -> _FakeResp:
            self.urls.append(req.full_url)
            body = json.dumps({"targets": ["H2-1SC", "H2-1E", "H2-1"]}).encode("utf-8")
            return _FakeResp(body)

    secret_tok = "nexus-super-secret-token-xyz"
    opener = _FakeOpener()
    client = QuantinuumNexusClient(api_token=secret_tok, opener=opener)
    assert secret_tok not in repr(client)
    res = client.test_connection()
    assert res["ok"] is True
    assert res["authenticated"] is True
    assert "H2-1SC" in res["targets"]
    assert secret_tok not in json.dumps(res)
    assert opener.urls == ["https://nexus.quantinuum.com/api/v1/targets"]

    # Error path must redact secret if upstream echoes it.
    def _leaking_hook(
        method: str, url: str, headers: Dict[str, str], payload: Optional[Dict[str, Any]]
    ) -> Tuple[int, Dict[str, Any]]:
        return 401, {"error": f"bad token {secret_tok}"}

    err_client = QuantinuumNexusClient(api_token=secret_tok, http_request=_leaking_hook)
    err_res = err_client.test_connection()
    assert err_res["ok"] is False
    assert secret_tok not in err_res["error"]
    assert "[REDACTED]" in err_res["error"]


# ---------------------------------------------------------------------------
# 5. CLI Subcommands (verify, run, bench, ledger verify, sync, repl dispatch)
# ---------------------------------------------------------------------------


def test_cli_verify_dry_run_and_ledger_verify_and_tamper_detection(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    receipt_path = tmp_path / "qec_receipt.json"
    rc = cli.main(
        [
            "verify",
            str(EXAMPLE_QEC),
            "--target",
            "H2-1SC",
            "--dry-run",
            "--record-ledger",
            str(receipt_path),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "H2-1SC (Tier 1)" in out
    assert "SIMULATED" in out
    assert "mid_circuit_measure=True" in out
    assert receipt_path.is_file()

    # Offline ledger verify succeeds on untampered receipt.
    rc_ok = cli.main(["ledger", "verify", str(receipt_path)])
    assert rc_ok == 0
    assert "[PASS]" in capsys.readouterr().out

    # Tamper with recorded receipt -> ledger verify MUST fail closed (exit 1).
    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    data["hardware"] = True
    receipt_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    rc_bad = cli.main(["ledger", "verify", str(receipt_path)])
    assert rc_bad == 1
    assert "TAMPERED OR INVALID RECEIPT" in capsys.readouterr().err


def test_cli_run_tier3_gate_and_success(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "cli-test-tier3-secret"
    source = EXAMPLE_QEC.read_text(encoding="utf-8")
    comp = compile_for_quantinuum(source, target="H2-1")
    token = sign_tier3_approval_token(
        target="H2-1",
        compiled_sha256=comp.compiled_sha256,
        shots=100,
        secret=secret,
    )

    server = _MockNexusServer(target_job_id="cli-h2-job-77")
    client = QuantinuumNexusClient(approval_secret=secret, http_request=server)

    # Without approval token -> fails closed with exit code 2.
    rc_fail = cli.main(
        ["run", str(EXAMPLE_QEC), "--target", "H2-1", "--shots", "100"],
        nexus_client=client,
    )
    assert rc_fail == 2
    assert "approval-token" in capsys.readouterr().err

    # With valid approval token -> HARDWARE_ATTESTED and valid receipt written.
    ledger_jsonl = tmp_path / "runs.jsonl"
    rc_ok = cli.main(
        [
            "run",
            str(EXAMPLE_QEC),
            "--target",
            "H2-1",
            "--shots",
            "100",
            "--approval-token",
            token,
            "--record-ledger",
            str(ledger_jsonl),
        ],
        nexus_client=client,
    )
    assert rc_ok == 0
    out = capsys.readouterr().out
    assert "HARDWARE_ATTESTED (hardware=True)" in out
    assert "cli-h2-job-77" in out
    assert cli.main(["ledger", "verify", str(ledger_jsonl)]) == 0


def test_cli_bench_markdown_and_json_formats(capsys: pytest.CaptureFixture[str]) -> None:
    rc_md = cli.main(
        [
            "bench",
            str(EXAMPLE_QEC),
            "--compare",
            "quantinuum_h2_1,quantinuum_helios,ibm_fez",
            "--shots",
            "1024",
            "--format",
            "markdown",
        ]
    )
    assert rc_md == 0
    md_out = capsys.readouterr().out
    assert "`quantinuum_h2_1`" in md_out
    assert "`quantinuum_helios`" in md_out
    assert "`ibm_fez`" in md_out
    assert "hardware_executed=false" in md_out

    rc_json = cli.main(
        [
            "bench",
            str(EXAMPLE_QEC),
            "--compare",
            "quantinuum_h2_1,quantinuum_helios,ibm_fez",
            "--shots",
            "1024",
            "--format",
            "json",
        ]
    )
    assert rc_json == 0
    report = json.loads(capsys.readouterr().out)
    assert report["hardware_executed"] is False
    assert report["shots"] == 1024
    by_backend = {r["backend"]: r for r in report["results"]}
    assert by_backend["quantinuum_h2_1"]["estimated_routing_swaps"] == 0
    assert by_backend["quantinuum_helios"]["estimated_routing_swaps"] == 0
    assert by_backend["ibm_fez"]["estimated_routing_swaps"] > 0


def test_cli_sync_push_pull_token_hygiene_and_409_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    circuit_file = tmp_path / "bell.qasm"
    original_content = 'OPENQASM 3.0;\nqubit[2] q;\nh q[0];\ncx q[0], q[1];\n'
    circuit_file.write_text(original_content, encoding="utf-8")

    # Missing OSIRIS_API_TOKEN fails closed.
    monkeypatch.delenv("OSIRIS_API_TOKEN", raising=False)
    rc_no_tok = cli.main(["sync", "push", "--dir", str(tmp_path), str(circuit_file)])
    assert rc_no_tok == 2
    assert "OSIRIS_API_TOKEN" in capsys.readouterr().err

    # Insecure non-localhost http:// URL rejected.
    token = "osiris-sync-secret-bearer-999"
    monkeypatch.setenv("OSIRIS_API_TOKEN", token)
    rc_http = cli.main(
        [
            "sync",
            "push",
            "--api-url",
            "http://example.com",
            "--dir",
            str(tmp_path),
            str(circuit_file),
        ]
    )
    assert rc_http == 2
    assert "insecure URL" in capsys.readouterr().err

    # 409 SYNC_CONFLICT returns non-zero (3), prints diff guidance, leaves local file untouched.
    local_sha = hashlib.sha256(original_content.encode("utf-8")).hexdigest()

    def _conflict_hook(
        method: str, url: str, headers: Dict[str, str], payload: Optional[Dict[str, Any]]
    ) -> Tuple[int, Dict[str, Any]]:
        assert headers["Authorization"] == f"Bearer {token}"
        assert url == "https://www.quantumagentlab.com/api/osiris/sync"
        return 409, {
            "error": "SYNC_CONFLICT",
            "conflicts": [
                {
                    "path": "bell.qasm",
                    "local_sha256": local_sha,
                    "remote_sha256": "f" * 64,
                    "remote_revision": "rev-42",
                }
            ],
        }

    rc_conflict = cli.main(
        [
            "sync",
            "push",
            "--workspace",
            "ws-1",
            "--org",
            "org-1",
            "--branch",
            "main",
            "--dir",
            str(tmp_path),
            "bell.qasm",
        ],
        sync_http_request=_conflict_hook,
    )
    assert rc_conflict == 3
    err_conflict = capsys.readouterr().err
    assert "409 SYNC_CONFLICT" in err_conflict
    assert "bell.qasm" in err_conflict
    assert "osiris sync pull --branch main" in err_conflict
    assert circuit_file.read_text(encoding="utf-8") == original_content

    # Successful push stores metadata with 0o600 permissions and NEVER writes token to disk.
    def _push_ok_hook(
        method: str, url: str, headers: Dict[str, str], payload: Optional[Dict[str, Any]]
    ) -> Tuple[int, Dict[str, Any]]:
        assert method == "POST"
        assert payload is not None and payload["action"] == "push"
        return 200, {"revision": "rev-43", "synced": 1}

    rc_push = cli.main(
        ["sync", "push", "--workspace", "ws-1", "--dir", str(tmp_path), "bell.qasm"],
        sync_http_request=_push_ok_hook,
    )
    assert rc_push == 0
    state_file = tmp_path / ".osiris_sync_state.json"
    assert state_file.is_file()
    state_text = state_file.read_text(encoding="utf-8")
    assert token not in state_text
    assert stat.S_IMODE(state_file.stat().st_mode) == 0o600

    # Successful pull writes remote file; path traversal is rejected.
    def _pull_ok_hook(
        method: str, url: str, headers: Dict[str, str], payload: Optional[Dict[str, Any]]
    ) -> Tuple[int, Dict[str, Any]]:
        assert method == "GET"
        assert "/api/osiris/sync?" in url
        return 200, {
            "revision": "rev-44",
            "files": [{"path": "pulled/ghz.qasm", "content": "OPENQASM 3.0;\nqubit[3] q;\n"}],
        }

    rc_pull = cli.main(
        ["sync", "pull", "--workspace", "ws-1", "--dir", str(tmp_path)],
        sync_http_request=_pull_ok_hook,
    )
    assert rc_pull == 0
    assert (tmp_path / "pulled" / "ghz.qasm").is_file()

    def _pull_traversal_hook(
        method: str, url: str, headers: Dict[str, str], payload: Optional[Dict[str, Any]]
    ) -> Tuple[int, Dict[str, Any]]:
        return 200, {"files": [{"path": "../escape.qasm", "content": "bad"}]}

    rc_trav = cli.main(
        ["sync", "pull", "--dir", str(tmp_path)],
        sync_http_request=_pull_traversal_hook,
    )
    assert rc_trav == 1
    assert not (tmp_path.parent / "escape.qasm").exists()


def test_osiris_repl_main_dispatches_new_subcommands_preserving_legacy(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # verify routes to osiris_cli.cli and exits 0.
    with pytest.raises(SystemExit) as exc_verify:
        osiris_repl.main(["verify", str(EXAMPLE_QEC), "--target", "H2-1SC", "--dry-run"])
    assert exc_verify.value.code == 0
    assert "OSIRIS Verify" in capsys.readouterr().out

    # bench routes to osiris_cli.cli and exits 0.
    with pytest.raises(SystemExit) as exc_bench:
        osiris_repl.main(["bench", str(EXAMPLE_QEC), "--format", "json"])
    assert exc_bench.value.code == 0
    assert json.loads(capsys.readouterr().out)["hardware_executed"] is False

    # Legacy `osiris sync status` still routes to execute_sync, NOT cloud sync.
    seen: List[Any] = []
    monkeypatch.setattr(osiris_repl, "execute_sync", lambda subargs=None: seen.append(subargs))
    osiris_repl.main(["sync", "status"])
    assert seen == [["status"]]


def test_pytket_path_handles_single_bit_conditions_and_keeps_feedforward() -> None:
    """With pytket installed, single-bit conditions used to raise QASMUnsupportedError at export."""
    pytest.importorskip("pytket")
    res = compile_for_quantinuum(SAMPLE_QASM3_MID_CIRCUIT, target="H2-1SC")
    assert res.ok and res.compiler_backend == "pytket"
    assert res.metrics.feedforward_conditionals == 2
    assert res.metrics.mid_circuit_measurements == compile_for_quantinuum(
        SAMPLE_QASM3_MID_CIRCUIT, target="H2-1SC", options=QuantinuumCompilerOptions(no_opt=True)
    ).metrics.mid_circuit_measurements


def test_static_and_pytket_paths_honour_no_reduce() -> None:
    src = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ncreg c[1];\nx q[0];\nx q[0];\nmeasure q[0] -> c[0];\n'
    res = compile_for_quantinuum(src, target="H2-1SC", options=QuantinuumCompilerOptions(no_reduce=True))
    assert res.metrics.total_gates == 2
