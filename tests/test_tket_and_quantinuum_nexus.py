"""Unit & governance tests for `tket_wrapper.py`, `quantinuum_nexus.py`, and `nexus_ledger.py`."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import payload_gate
import pytest

from osiris_cli.nexus_ledger import (
    NexusAuditLedger,
    compute_receipt_sha256,
    upload_disposition_receipt,
)
from osiris_cli.providers.quantinuum_nexus import (
    GovernanceError,
    NexusConfig,
    NexusDisposition,
    QuantinuumNexusClient,
    estimate_hqc,
    sign_tier3_approval_token,
    verify_tier3_approval_token,
)
from osiris_cli.providers.tket_wrapper import (
    QuantinuumCompilerOptions,
    compile_for_quantinuum,
)
from osiris_cli.quantinuum_cli import dispatch_quantinuum_argv

REPO_ROOT = Path(__file__).resolve().parents[1]
BITFLIP_QASM = (REPO_ROOT / "examples" / "quantinuum" / "bitflip_qec_2round.qasm").read_text(
    encoding="utf-8"
)


def test_new_modules_pass_payload_gate_and_pyproject_has_quantinuum_extra() -> None:
    for rel in (
        "osiris_cli/providers/tket_wrapper.py",
        "osiris_cli/providers/quantinuum_nexus.py",
        "osiris_cli/nexus_ledger.py",
    ):
        src = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert payload_gate.check_payload(src) is None, f"payload_gate rejected {rel}"

    pyproj_txt = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m_q = re.search(r"^quantinuum\s*=\s*\[(.*?)\]", pyproj_txt, re.M | re.S)
    assert m_q is not None and "pytket" in m_q.group(1)


def test_tket_wrapper_compiles_peephole_optimizes_and_enforces_limits() -> None:
    # Redundant self-inverse pairs x;x and cx;cx cancel at optimization_level=2
    redundant = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
x q[0];
x q[0];
cx q[0], q[1];
cx q[0], q[1];
cx q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"""
    res_o0 = compile_for_quantinuum(
        redundant, target="H2-1SC", options=QuantinuumCompilerOptions(optimization_level=0)
    )
    res_o2 = compile_for_quantinuum(
        redundant, target="H2-1SC", options=QuantinuumCompilerOptions(optimization_level=2)
    )
    assert res_o0.ok and res_o2.ok
    assert res_o0.metrics.total_gates == 6
    assert res_o2.metrics.total_gates < res_o0.metrics.total_gates
    assert res_o2.metrics.two_qubit_gates == 1
    assert len(res_o2.compiled_sha256) == 64

    # Mid-circuit reset & feedforward compilation
    bf_res = compile_for_quantinuum(BITFLIP_QASM, target="H2-1")
    assert bf_res.ok is True
    assert bf_res.metrics.mid_circuit_measurements == 4
    assert bf_res.metrics.resets == 4
    assert bf_res.metrics.feedforward_conditionals == 3
    assert "if(syn==" in bf_res.compiled_qasm2

    # Disallowing feedforward returns structured diagnostics instead of raising
    blocked = compile_for_quantinuum(
        BITFLIP_QASM,
        target="H2-1",
        options=QuantinuumCompilerOptions(allow_classical_feedforward=False),
    )
    assert blocked.ok is False
    assert any("feedforward" in d.lower() for d in blocked.diagnostics)


def test_quantinuum_nexus_tier1_tier2_tier3_governance_and_unspoofable_disposition() -> None:
    secret = "test-tier3-approval-secret"
    cfg = NexusConfig(
        token="fake-nexus-bearer-token",
        base_url="https://nexus.quantinuum.com/api/v1",
        max_shots=5000,
        max_hqc=40.0,
        approval_secret=secret,
    )
    client = QuantinuumNexusClient(cfg)
    calls: List[Tuple[str, str, Optional[Dict[str, Any]]]] = []

    def fake_http(method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        calls.append((method, path, body))
        if method == "GET" and path == "/backends":
            return {"backends": ["H2-1SC", "H2-1E", "H2-1", "Helios"]}
        if method == "POST" and path == "/jobs":
            assert body is not None
            return {"job_id": f"job-{body['backend'].lower()}-001", "status": "queued"}
        if method == "GET" and path.startswith("/jobs/"):
            return {
                "job_id": path.rsplit("/", 1)[-1],
                "status": "completed",
                "counts": {"111 00 01": 500},
                "calibration": {"fid_2q": "0.9987"},
            }
        raise AssertionError(f"Unexpected call {method} {path}")

    with patch.object(client, "_request", side_effect=fake_http):
        conn = client.test_connection()
        assert conn["ok"] is True
        calls.clear()

        # 1. Tier 1 syntax checker: free, 0 HQC, SIMULATED, hardware=False
        d1 = client.submit_and_wait(BITFLIP_QASM, target="H2-1SC", shots=10)
        assert d1.tier == 1
        assert d1.estimated_hqc == 0.0
        assert d1.hardware is False
        assert d1.disposition == "SIMULATED"
        assert [c[2]["backend"] for c in calls if c[0] == "POST"] == ["H2-1SC"]
        calls.clear()

        # 2. Tier 2 emulator: runs mandatory H2-1SC pre-flight FIRST, then H2-1E -> SIMULATED
        d2 = client.submit_and_wait(BITFLIP_QASM, target="H2-1E", shots=500, budget_hqc=30.0)
        assert d2.tier == 2
        assert d2.hardware is False
        assert d2.disposition == "SIMULATED"
        assert [c[2]["backend"] for c in calls if c[0] == "POST"] == ["H2-1SC", "H2-1E"]
        calls.clear()

        # 3. Budget cap fail-closed BEFORE any HTTP call!
        with pytest.raises(GovernanceError, match="exceeds budget cap"):
            client.submit_and_wait(BITFLIP_QASM, target="H2-1E", shots=500, budget_hqc=4.0)
        assert calls == []

        # 4. Tier 3 live QPU without approval_token -> fails closed with zero HTTP calls!
        with pytest.raises(GovernanceError, match="approval_token"):
            client.submit_and_wait(BITFLIP_QASM, target="H2-1", shots=500, budget_hqc=30.0)
        assert calls == []

        # 5. Tier 3 live QPU with valid HMAC token over (target, compiled_sha256, shots) -> HARDWARE_ATTESTED
        compiled = compile_for_quantinuum(BITFLIP_QASM, target="H2-1")
        assert estimate_hqc(compiled, 500) > 5.0
        valid_tok = sign_tier3_approval_token(
            target="H2-1",
            compiled_sha256=compiled.compiled_sha256,
            shots=500,
            secret=secret,
        )
        verify_tier3_approval_token(
            valid_tok,
            target="H2-1",
            compiled_sha256=compiled.compiled_sha256,
            shots=500,
            secret=secret,
        )

        # Wrong shot count rejected by token guard
        with pytest.raises(GovernanceError, match="Invalid Tier 3 approval_token"):
            client.submit_and_wait(
                BITFLIP_QASM,
                target="H2-1",
                shots=501,
                budget_hqc=30.0,
                approval_token=valid_tok,
            )
        assert calls == []

        d3 = client.submit_and_wait(
            BITFLIP_QASM,
            target="H2-1",
            shots=500,
            budget_hqc=30.0,
            approval_token=valid_tok,
        )
        assert d3.tier == 3
        assert d3.hardware is True
        assert d3.disposition == "HARDWARE_ATTESTED"
        assert [c[2]["backend"] for c in calls if c[0] == "POST"] == ["H2-1SC", "H2-1"]

    # 6. Direct NexusDisposition construction forbids spoofing HARDWARE_ATTESTED on Tier 1/2
    with pytest.raises(GovernanceError):
        NexusDisposition(
            job_id="spoof",
            target="H2-1E",
            tier=2,
            status="completed",
            hardware=True,
            disposition="HARDWARE_ATTESTED",
            shots=100,
            estimated_hqc=6.0,
            source_sha256="a" * 64,
            compiled_sha256="b" * 64,
            compiler_options={},
        )

    # 7. Failed Tier 3 job becomes QUARANTINED with hardware=False
    def fake_failed(method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if method == "POST":
            assert body is not None
            return {"job_id": f"job-{body['backend']}", "status": "queued"}
        if path.endswith("H2-1SC"):
            return {"job_id": "job-H2-1SC", "status": "completed"}
        return {"job_id": "job-H2-1", "status": "failed"}

    with patch.object(client, "_request", side_effect=fake_failed):
        dfail = client.submit_and_wait(
            BITFLIP_QASM,
            target="H2-1",
            shots=500,
            budget_hqc=30.0,
            approval_token=valid_tok,
        )
        assert dfail.hardware is False
        assert dfail.disposition == "QUARANTINED"


def test_nexus_audit_ledger_and_receipt_upload_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    home = tmp_path / "osiris_home"
    ledger = NexusAuditLedger(home)

    compiled = compile_for_quantinuum(BITFLIP_QASM, target="H2-1")
    disp = NexusDisposition(
        job_id="nexus-job-777",
        target="H2-1",
        tier=3,
        status="completed",
        hardware=True,
        disposition="HARDWARE_ATTESTED",
        shots=500,
        estimated_hqc=8.2,
        source_sha256=compiled.source_sha256,
        compiled_sha256=compiled.compiled_sha256,
        compiler_options=compiled.compiler_options,
        counts={"111 00 01": 495, "111 00 00": 5},
        calibration_snapshot={"fid_2q": "0.9988"},
    )
    saved = ledger.append_disposition(disp)
    assert len(saved["receipt_sha256"]) == 64
    assert saved["verify_url"].endswith(saved["receipt_sha256"])
    assert ledger.verify()["valid"] is True

    found = ledger.find_receipt("nexus-job-777")
    assert compute_receipt_sha256(found) == saved["receipt_sha256"]

    # CLI compile, nexus-sign, and ledger upload --dry-run
    monkeypatch.setenv("OSIRIS_APPROVAL_SECRET", "cli-signing-secret")
    qasm_path = REPO_ROOT / "examples" / "quantinuum" / "bitflip_qec_2round.qasm"
    assert dispatch_quantinuum_argv(["compile", str(qasm_path), "--target", "H2-1", "--json"]) == 0
    comp_out = json.loads(capsys.readouterr().out.strip())
    assert comp_out["compiled_sha256"] == compiled.compiled_sha256

    assert (
        dispatch_quantinuum_argv(
            ["nexus-sign", str(qasm_path), "--target", "H2-1", "--shots", "500", "--quiet"]
        )
        == 0
    )
    tok = capsys.readouterr().out.strip()
    assert tok.startswith("osiris-t3.")

    assert (
        dispatch_quantinuum_argv(
            [
                "ledger",
                "upload",
                "nexus-job-777",
                "--home",
                str(home),
                "--dry-run",
                "--json",
            ]
        )
        == 0
    )
    dry_res = json.loads(capsys.readouterr().out.strip())
    assert dry_res["ok"] is True
    assert dry_res["dry_run"] is True
    assert dry_res["receipt_sha256"] == saved["receipt_sha256"]

    # Mocked live HTTPS upload to quantumagentlab.com/api/verify
    class _FakeResp:
        def __enter__(self) -> "_FakeResp":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "accepted": True,
                    "verify_url": f"https://www.quantumagentlab.com/verify/{saved['receipt_sha256']}",
                }
            ).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=_FakeResp()):
        live_up = upload_disposition_receipt(disp, api_token="qal-test-token", dry_run=False)
        assert live_up["ok"] is True
        assert live_up["dry_run"] is False
        assert live_up["verify_url"].endswith(saved["receipt_sha256"])

    # Tampered receipt payload is rejected BEFORE upload
    bad_payload = disp.to_ledger_dict()
    bad_payload["shots"] = 999
    with pytest.raises(GovernanceError, match="SHA-256 integrity mismatch"):
        upload_disposition_receipt(bad_payload, dry_run=True)
