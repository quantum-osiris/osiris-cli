"""Tests for Quantinuum Nexus gatekeeping, QCCD emulation, Tier 3 HQC governance,
QCCD vs. heavy-hex topology benchmarks, ledger tamper detection, and repo hygiene.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

import payload_gate
from osiris_cli import osiris_repl
from osiris_cli.providers.local_qccd import QCCDNoiseModel, emulate_qccd, parse_quantum_source
from osiris_cli.providers.quantinuum import compare_topologies, compute_hqc
from osiris_cli.quantinuum_cli import dispatch_quantinuum_argv, handle_repl_quantinuum
from osiris_cli.quantinuum_gate import (
    QuantinuumGateError,
    mint_tier3_token,
    run_tier3_governed,
    verify_all_quantum_ledgers,
    verify_quantinuum_circuit,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples" / "quantinuum"


def test_repo_root_hygiene_and_payload_gate_clean() -> None:
    root_names = [p.name for p in REPO_ROOT.iterdir() if p.is_file()]
    forbidden_substrings = (
        "NOBEL",
        "NOBLE",
        "Untitled",
        "2847.3",
        ".bak",
        "QBYTE_MINING",
        "osiris_3dprint",
        "batch_test_",
        "autoadvance_",
        "job-d75",
    )
    offenders = [
        name
        for name in root_names
        if any(bad.lower() in name.lower() for bad in forbidden_substrings)
    ]
    assert offenders == [], f"Clutter files remaining in repository root: {offenders}"

    for rel in (
        "osiris_cli/providers/__init__.py",
        "osiris_cli/providers/local_qccd.py",
        "osiris_cli/providers/quantinuum.py",
        "osiris_cli/quantinuum_gate.py",
        "osiris_cli/quantinuum_cli.py",
    ):
        src = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert payload_gate.check_payload(src) is None, f"payload_gate rejected {rel}"


def test_examples_quantinuum_all_pass_h2_1sc_and_qccd_feedforward() -> None:
    expected_files = (
        "bitflip_qec.qasm",
        "bitflip_qec_2round.qasm",
        "code422_syndrome_detect.qasm",
        "steane_7q_x_syndrome.qasm",
    )
    for fname in expected_files:
        fpath = EXAMPLES_DIR / fname
        assert fpath.is_file(), f"Missing runnable Quantinuum example {fpath}"
        rep = verify_quantinuum_circuit(
            fpath.read_text(encoding="utf-8"),
            target="H2-1SC",
            emulate="qccd",
            shots=256,
            seed=11,
            filename=fname,
        )
        assert rep.passed is True
        assert rep.disposition == "SIMULATED"
        assert rep.hardware is False
        assert rep.routing_swaps == 0
        assert rep.mid_circuit_measurements >= 2
        assert rep.resets >= 2
        assert rep.emulation is not None
        assert sum(rep.emulation.raw_counts.values()) == 256

    # Ideal feedforward check: bitflip_qec injects X on q[0] of |111>; feedforward restores |111> 100%
    bf_parsed = parse_quantum_source((EXAMPLES_DIR / "bitflip_qec_2round.qasm").read_text(encoding="utf-8"))
    ideal_bf = emulate_qccd(bf_parsed, noise_model=QCCDNoiseModel.ideal(), shots=128, seed=3)
    assert ideal_bf.register_counts["data"] == {"111": 128}
    assert ideal_bf.register_counts["syn_r2"] == {"00": 128}

    # Ideal feedforward check on 10-qubit Steane [[7,1,3]]: z_syn==7 corrects q[6] back to code space
    st_parsed = parse_quantum_source(
        (EXAMPLES_DIR / "steane_7q_x_syndrome.qasm").read_text(encoding="utf-8")
    )
    ideal_st = emulate_qccd(st_parsed, noise_model=QCCDNoiseModel.ideal(), shots=64, seed=5)
    assert ideal_st.register_counts["z_syn"] == {"111": 64}
    assert ideal_st.register_counts["x_syn"] == {"000": 64}


def test_guppy_ast_validation_and_qubit_ceiling_enforcement() -> None:
    guppy_src = """
from guppylang import guppy
from guppylang.std.quantum import qubit, h, cx, measure, reset, x

@guppy
def qec_kernel() -> None:
    q0 = qubit()
    q1 = qubit()
    anc = qubit()
    h(q0)
    cx(q0, q1)
    cx(q0, anc)
    cx(q1, anc)
    syn = measure(anc)
    reset(anc)
    if syn:
        x(q1)
"""
    rep = verify_quantinuum_circuit(guppy_src, target="H2-1SC", emulate="qccd", shots=100, filename="k.py")
    assert rep.source_format == "guppy"
    assert rep.n_qubits == 3
    assert rep.mid_circuit_measurements == 1
    assert rep.resets == 1
    assert rep.feedforward_conditionals == 1

    # 64-qubit register exceeds H2-1SC (56q) but fits Helios-SC (98q)
    wide_qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[64];\nh q[0];\ncx q[0], q[63];\n'
    with pytest.raises(QuantinuumGateError) as exc_info:
        verify_quantinuum_circuit(wide_qasm, target="H2-1SC", shots=100)
    assert exc_info.value.code == "BLOCKED_QUBIT_CEILING"

    helios_rep = verify_quantinuum_circuit(wide_qasm, target="Helios-SC", shots=100)
    assert helios_rep.passed is True
    assert helios_rep.max_qubits == 98


def test_propose_verify_ledger_cli_and_tamper_detection(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_qasm = tmp_path / "proposed.qasm"
    home_dir = tmp_path / "osiris_home"

    rc_prop = dispatch_quantinuum_argv(
        [
            "propose",
            "3-qubit bit-flip QEC with mid-circuit syndrome reset",
            "--target",
            "quantinuum",
            "--out",
            str(out_qasm),
        ]
    )
    assert rc_prop == 0
    assert out_qasm.is_file()
    capsys.readouterr()

    rc_ver = dispatch_quantinuum_argv(
        [
            "verify",
            str(out_qasm),
            "--target",
            "H2-1SC",
            "--emulate",
            "qccd",
            "--shots",
            "200",
            "--record-ledger",
            "--home",
            str(home_dir),
            "--json",
        ]
    )
    assert rc_ver == 0
    ver_json = json.loads(capsys.readouterr().out.strip())
    assert ver_json["passed"] is True
    assert ver_json["disposition"] == "SIMULATED"
    assert ver_json["hardware"] is False
    assert len(ver_json["receipt_hash"]) == 64

    rc_led = dispatch_quantinuum_argv(["ledger", "verify", "--home", str(home_dir), "--json"])
    assert rc_led == 0
    led_json = json.loads(capsys.readouterr().out.strip())
    assert led_json["valid"] is True
    assert led_json["ledgers"]["nexus_receipts"]["entries"] == 1

    # Corrupt a single character inside nexus_receipts.jsonl -> must fail closed
    receipt_file = home_dir / "quantum" / "nexus_receipts.jsonl"
    content = receipt_file.read_text(encoding="utf-8")
    tampered = content.replace('"shots":200', '"shots":201', 1)
    receipt_file.write_text(tampered, encoding="utf-8")

    rc_bad = dispatch_quantinuum_argv(["ledger", "verify", "--home", str(home_dir)])
    assert rc_bad == 2
    capsys.readouterr()


def test_tier3_hqc_budget_and_signed_approval_guard_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OSIRIS_ALLOW_LIVE_NEXUS_SUBMIT", raising=False)
    monkeypatch.delenv("QUANTINUUM_NEXUS_TOKEN", raising=False)
    monkeypatch.delenv("OSIRIS_TIER3_HMAC_SECRET", raising=False)

    home_dir = tmp_path / "osiris_home"
    qasm_file = EXAMPLES_DIR / "bitflip_qec.qasm"
    qasm_src = qasm_file.read_text(encoding="utf-8")

    # 1. Missing approval token -> BLOCKED_APPROVAL_TOKEN
    with pytest.raises(QuantinuumGateError) as e_missing:
        run_tier3_governed(qasm_src, target="H2-1", shots=500, budget_hqc="25", approval_token="", osiris_home=home_dir)
    assert e_missing.value.code == "BLOCKED_APPROVAL_TOKEN"

    # 2. Mint Ed25519 operator token via CLI
    rc_grant = dispatch_quantinuum_argv(
        [
            "grant-tier3",
            str(qasm_file),
            "--target",
            "H2-1",
            "--shots",
            "500",
            "--budget-hqc",
            "25",
            "--home",
            str(home_dir),
            "--quiet",
        ]
    )
    assert rc_grant == 0
    token = capsys.readouterr().out.strip()
    assert token.startswith("osiris-t3-ed25519.")

    # 3. Mutate circuit by 1 gate after token minting -> BLOCKED_APPROVAL_TOKEN
    mutated_src = qasm_src + "\nx data[0];\n"
    with pytest.raises(QuantinuumGateError) as e_mut:
        run_tier3_governed(
            mutated_src, target="H2-1", shots=500, budget_hqc="25", approval_token=token, osiris_home=home_dir
        )
    assert e_mut.value.code == "BLOCKED_APPROVAL_TOKEN"

    # 4. Replay H2-1 token against Helios -> BLOCKED_APPROVAL_TOKEN
    with pytest.raises(QuantinuumGateError) as e_tgt:
        run_tier3_governed(
            qasm_src, target="Helios", shots=500, budget_hqc="25", approval_token=token, osiris_home=home_dir
        )
    assert e_tgt.value.code == "BLOCKED_APPROVAL_TOKEN"

    # 5. Tight budget below base 5.0 HQC -> BLOCKED_HQC_BUDGET
    with pytest.raises(QuantinuumGateError) as e_bud:
        run_tier3_governed(
            qasm_src, target="H2-1", shots=500, budget_hqc="4.5", approval_token=token, osiris_home=home_dir
        )
    assert e_bud.value.code == "BLOCKED_HQC_BUDGET"

    # 6. Valid token + valid budget -> passes preflight & records PREVIEW_COMPLETE (hardware=False)
    rc_run = dispatch_quantinuum_argv(
        [
            "run",
            str(qasm_file),
            "--target",
            "H2-1",
            "--shots",
            "500",
            "--budget-hqc",
            "25",
            "--approval-token",
            token,
            "--home",
            str(home_dir),
            "--json",
        ]
    )
    assert rc_run == 0
    run_payload = json.loads(capsys.readouterr().out.strip())
    assert run_payload["status"] == "QNEXUS_LIVE_SUBMIT_DISABLED"
    assert run_payload["disposition"] == "PREVIEW_COMPLETE"
    assert run_payload["hardware"] is False

    # 7. Also verify HMAC mode
    monkeypatch.setenv("OSIRIS_TIER3_HMAC_SECRET", "unit-test-hmac-secret-key")
    c_sha = parse_quantum_source(qasm_src).circuit_sha256
    hmac_tok = mint_tier3_token(c_sha, target="H2-1", max_shots=500, max_hqc=Decimal("25"), osiris_home=home_dir)
    assert hmac_tok.startswith("osiris-t3-hmac1.")
    res_hmac = run_tier3_governed(
        qasm_src, target="H2-1", shots=500, budget_hqc="25", approval_token=hmac_tok, osiris_home=home_dir
    )
    assert res_hmac.disposition == "PREVIEW_COMPLETE"
    assert res_hmac.hardware is False
    assert verify_all_quantum_ledgers(home_dir)["valid"] is True


def test_qccd_vs_heavy_hex_topology_benchmark_and_repl_hook(capsys: pytest.CaptureFixture[str]) -> None:
    steane_path = EXAMPLES_DIR / "steane_7q_x_syndrome.qasm"
    rc = dispatch_quantinuum_argv(
        [
            "bench",
            str(steane_path),
            "--compare",
            "quantinuum_h2_1,quantinuum_helios,ibm_fez",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    bench = json.loads(capsys.readouterr().out.strip())
    rows = {r["backend"]: r for r in bench["rows"]}
    assert rows["quantinuum_h2_1"]["routing_swaps"] == 0
    assert rows["quantinuum_helios"]["routing_swaps"] == 0
    assert rows["ibm_fez"]["routing_swaps"] > 0
    assert (
        rows["ibm_fez"]["routed_2q_gates"]
        == rows["ibm_fez"]["logical_2q_gates"] + 3 * rows["ibm_fez"]["routing_swaps"]
    )
    assert Decimal(rows["quantinuum_helios"]["estimated_fidelity"]) > Decimal(
        rows["ibm_fez"]["estimated_fidelity"]
    )
    assert bench["qccd_swap_saving_vs_heavy_hex"] == rows["ibm_fez"]["routing_swaps"]

    # Shorthand `qccd,heavy-hex` markdown output
    rep_short = compare_topologies(steane_path.read_text(encoding="utf-8"), targets=("qccd,heavy-hex",))
    md = rep_short.to_markdown()
    assert "| `quantinuum_h2_1` |" in md
    assert "| `ibm_fez` |" in md

    #Exact HQC formula sanity check: 5 + 1000 * (10 + 10*20 + 5*8) / 5000 = 5 + 250/5 = 55.0000
    assert compute_hqc(10, 20, 8, 1000) == Decimal("55.0000")

    # REPL /quantinuum hook & main() wiring
    txt = handle_repl_quantinuum("targets")
    assert "H2-1SC" in txt and "Helios" in txt
    assert "/quantinuum" in osiris_repl.REPL_COMMANDS
