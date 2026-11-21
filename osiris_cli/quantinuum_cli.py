"""CLI & REPL entrypoint for Quantinuum Nexus gatekeeping, HQC governance, and QCCD benchmarks.

Subcommands:
  - `osiris propose "<spec>" --target quantinuum [--out FILE] [--model MODEL]`
  - `osiris verify <circuit> --target H2-1SC [--emulate qccd] [--shots 1024] [--record-ledger]`
  - `osiris bench [<circuit>] --compare quantinuum_h2_1,quantinuum_helios,ibm_fez [--format markdown|json]`
  - `osiris grant-tier3 <circuit> --target H2-1 --shots 500 --budget-hqc 25 [--quiet]`
  - `osiris run <circuit> --target H2-1 --shots 500 --budget-hqc 25 --approval-token TOKEN`
  - `osiris ledger verify [--home DIR] [--json]`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .exec_kernel import canon
from .providers.quantinuum import compare_topologies
from .quantinuum_gate import (
    _TEMPLATE_CODE422,
    QuantinuumGateError,
    mint_tier3_token,
    propose_quantinuum_circuit,
    run_tier3_governed,
    verify_all_quantum_ledgers,
    verify_quantinuum_circuit,
)


def _read_circuit_file(path_str: str) -> tuple[str, str]:
    p = Path(path_str).expanduser()
    if not p.is_file():
        raise QuantinuumGateError("FILE_NOT_FOUND", f"Circuit file not found: {path_str}")
    return p.read_text(encoding="utf-8"), p.name


def cmd_propose(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="osiris propose",
        description="Propose a Quantinuum QCCD OpenQASM circuit and pre-verify syntax",
    )
    parser.add_argument("spec", help="Natural-language circuit or QEC specification")
    parser.add_argument("--target", default="quantinuum", help="Target backend (default: quantinuum)")
    parser.add_argument("--out", default=None, help="Output file path for generated .qasm circuit")
    parser.add_argument("--model", default=None, help="Optional local Ollama model proposer")
    args = parser.parse_args(list(argv))

    try:
        res = propose_quantinuum_circuit(args.spec, target=args.target, model=args.model)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(res.qasm_source, encoding="utf-8")
        print(
            f"[PROPOSED] target={res.target} proposer={res.proposer_source} "
            f"sha256={res.circuit_sha256[:16]}… -> {out_path}"
        )
    else:
        sys.stdout.write(res.qasm_source)
    return 0


def cmd_verify(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="osiris verify",
        description="Deterministic H2-1SC / Helios-SC gatekeeper & local QCCD trajectory emulator",
    )
    parser.add_argument("circuit", help="Path to OpenQASM (.qasm) or Guppy (.py) circuit")
    parser.add_argument("--target", default="H2-1SC", help="Quantinuum target (e.g. H2-1SC, Helios-SC, H2-1E)")
    parser.add_argument(
        "--emulate",
        default=None,
        choices=["qccd", "ideal"],
        help="Run deterministic local QCCD trajectory noise emulation",
    )
    parser.add_argument("--shots", type=int, default=1024, help="Shot count (default: 1024)")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic RNG seed (default: 7)")
    parser.add_argument(
        "--record-ledger",
        action="store_true",
        help="Append cryptographic SHA-256 receipt to nexus_receipts.jsonl",
    )
    parser.add_argument("--home", default=None, help="Override OSIRIS_HOME directory")
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON report")
    args = parser.parse_args(list(argv))

    try:
        src, fname = _read_circuit_file(args.circuit)
        report = verify_quantinuum_circuit(
            src,
            target=args.target,
            emulate=args.emulate,
            shots=args.shots,
            seed=args.seed,
            record_ledger=args.record_ledger,
            osiris_home=args.home,
            filename=fname,
        )
    except QuantinuumGateError as exc:
        if args.json:
            print(json.dumps({"passed": False, "error_code": exc.code, "message": exc.detail}))
        else:
            print(f"REJECTED {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(canon(report.to_dict()))
        return 0

    lines = [
        f"[VERIFIED] target={report.target} (Tier {report.tier}) | disposition={report.disposition} | hardware=False",
        f"  circuit_sha256 : {report.circuit_sha256}",
        f"  qubits         : {report.n_qubits} / {report.max_qubits} (0 routing SWAPs on all-to-all QCCD)",
        f"  gates          : 1Q={report.single_qubit_gates} (PhasedX={report.physical_1q_gates}, virtual_Rz={report.virtual_z_gates}) | 2Q_ZZ={report.two_qubit_gates}",
        f"  mid-circuit    : meas={report.mid_circuit_measurements}/{report.measurements} | resets={report.resets} | feedforward_if={report.feedforward_conditionals}",
        f"  hqc_estimate   : {report.hqc_estimate} HQC @ {report.shots} shots",
    ]
    if report.emulation is not None:
        emu = report.emulation
        syn_txt = (
            f" | syndrome_zero_rate={emu.syndrome_zero_rate:.4f}"
            if emu.syndrome_zero_rate is not None
            else ""
        )
        lines.append(
            f"  qccd_emulation : model={emu.noise_model} | top='{emu.dominant_bitstring}' "
            f"({emu.dominant_probability:.4f}){syn_txt} | raw_counts_sha256={emu.raw_counts_sha256[:16]}…"
        )
    if report.receipt_hash is not None:
        lines.append(f"  ledger_receipt : {report.receipt_hash} ({report.receipt_path})")
    print("\n".join(lines))
    return 0


def cmd_bench(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="osiris bench",
        description="Compare QCCD all-to-all connectivity vs. heavy-hex SWAP routing",
    )
    parser.add_argument(
        "circuit",
        nargs="?",
        default=None,
        help="Path to OpenQASM/Guppy circuit (defaults to [[4,2,2]] QEC benchmark circuit)",
    )
    parser.add_argument(
        "--compare",
        default="quantinuum_h2_1,quantinuum_helios,ibm_fez",
        help="Comma-separated backends or 'qccd,heavy-hex'",
    )
    parser.add_argument(
        "--format",
        dest="fmt",
        default="markdown",
        choices=["markdown", "json"],
        help="Output format (markdown or json)",
    )
    parser.add_argument("--shots", type=int, default=1000, help="Reference shot count for HQC sizing")
    args = parser.parse_args(list(argv))

    try:
        if args.circuit:
            src, fname = _read_circuit_file(args.circuit)
        else:
            default_qasm = (
                Path(__file__).resolve().parents[1]
                / "examples"
                / "quantinuum"
                / "code422_syndrome_detect.qasm"
            )
            if default_qasm.is_file():
                src, fname = default_qasm.read_text(encoding="utf-8"), default_qasm.name
            else:
                src, fname = _TEMPLATE_CODE422, "code422_syndrome_detect.qasm"
        report = compare_topologies(
            src,
            targets=[args.compare],
            shots_reference=args.shots,
            filename=fname,
        )
    except Exception as exc:
        print(f"BENCHMARK_ERROR: {exc}", file=sys.stderr)
        return 2

    if args.fmt == "json":
        print(canon(report.to_dict()))
    else:
        print(report.to_markdown())
    return 0


def cmd_grant_tier3(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="osiris grant-tier3",
        description="Mint an operator-signed Tier 3 approval token bound to exact circuit SHA-256 & HQC cap",
    )
    parser.add_argument("circuit", help="Path to pre-verified circuit file")
    parser.add_argument("--target", required=True, help="Live QPU target (H2-1, H2-2, Helios)")
    parser.add_argument("--shots", type=int, required=True, help="Maximum approved shots")
    parser.add_argument("--budget-hqc", required=True, help="Maximum approved HQC budget")
    parser.add_argument("--ttl", type=int, default=3600, help="Token validity in seconds (default: 3600)")
    parser.add_argument("--home", default=None, help="Override OSIRIS_HOME directory")
    parser.add_argument("--quiet", action="store_true", help="Print only the raw signed token")
    args = parser.parse_args(list(argv))

    try:
        src, _ = _read_circuit_file(args.circuit)
        c_sha = hashlib.sha256(src.encode("utf-8")).hexdigest()
        tok = mint_tier3_token(
            circuit_sha256=c_sha,
            target=args.target,
            max_shots=args.shots,
            max_hqc=args.budget_hqc,
            ttl_seconds=args.ttl,
            osiris_home=args.home,
        )
    except Exception as exc:
        print(f"GRANT_ERROR: {exc}", file=sys.stderr)
        return 2

    if args.quiet:
        print(tok)
    else:
        print(f"circuit_sha256={c_sha}")
        print(f"OSIRIS_TIER3_TOKEN={tok}")
    return 0


def cmd_run(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="osiris run",
        description="Submit a governed Tier 3 Quantinuum job (fails closed without preflight, budget & signed token)",
    )
    parser.add_argument("circuit", help="Path to circuit file")
    parser.add_argument("--target", default="H2-1", help="Target QPU (H2-1, H2-2, Helios)")
    parser.add_argument("--shots", type=int, default=500, help="Shot count (default: 500)")
    parser.add_argument("--budget-hqc", default="25", help="Hard HQC budget ceiling (default: 25)")
    parser.add_argument("--approval-token", default="", help="Signed Tier 3 approval token")
    parser.add_argument("--home", default=None, help="Override OSIRIS_HOME directory")
    parser.add_argument("--json", action="store_true", help="Emit canonical JSON receipt summary")
    args = parser.parse_args(list(argv))

    try:
        src, fname = _read_circuit_file(args.circuit)
        res = run_tier3_governed(
            src,
            target=args.target,
            shots=args.shots,
            budget_hqc=args.budget_hqc,
            approval_token=args.approval_token,
            osiris_home=args.home,
            filename=fname,
        )
    except QuantinuumGateError as exc:
        if args.json:
            print(json.dumps({"ok": False, "error_code": exc.code, "message": exc.detail}))
        else:
            print(f"FAIL_CLOSED {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(canon(res.to_dict()))
        return 0

    print(
        f"[{res.status}] target={res.target} (Tier {res.tier}) | "
        f"disposition={res.disposition} | hardware={res.hardware}\n"
        f"  preflight      : {res.preflight_checker} PASSED (circuit_sha256={res.circuit_sha256[:16]}…)\n"
        f"  hqc_budget     : {res.hqc_estimate} / {res.budget_hqc} HQC @ {res.shots} shots (APPROVED)\n"
        f"  ledger_receipt : {res.receipt_hash} ({res.receipt_path})"
    )
    return 0


def cmd_ledger(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="osiris ledger",
        description="Inspect and cryptographically verify local OSIRIS evidence ledgers",
    )
    parser.add_argument("subcmd", nargs="?", default="verify", choices=["verify"])
    parser.add_argument("--home", default=None, help="Override OSIRIS_HOME directory")
    parser.add_argument("--json", action="store_true", help="Emit JSON verification summary")
    args = parser.parse_args(list(argv))

    summary = verify_all_quantum_ledgers(osiris_home=args.home)
    if args.json:
        print(canon(summary))
        return 0 if summary["valid"] else 2

    status = "VALID" if summary["valid"] else "TAMPERED"
    print(f"[LEDGER {status}] osiris_home={summary['osiris_home']}")
    for name, info in summary["ledgers"].items():
        state = "OK" if info["valid"] else "FAIL"
        present = f"{info['entries']} entries" if info["exists"] else "empty (not yet created)"
        print(f"  - {name:20s}: {state} ({present}) tail={info['tail_hash'][:16]}…")
        for err in info["errors"]:
            print(f"      ERROR: {err}", file=sys.stderr)
    return 0 if summary["valid"] else 2


def handle_repl_quantinuum(arg_str: str = "", osiris_home: Optional[str] = None) -> str:
    """Read-only `/quantinuum` REPL summary and deterministic pre-flight dispatcher."""
    tokens = shlex.split(arg_str.strip()) if arg_str.strip() else ["targets"]
    sub = tokens[0].lower()
    if sub in ("targets", "status", "help"):
        lines = [
            "Quantinuum Nexus & QCCD Gatekeeper Targets:",
            "  Tier 1 (Syntax Checker) : H2-1SC (56q), H2-2SC (56q), Helios-SC (98q) -> SIMULATED",
            "  Tier 2 (Noise Emulator) : H2-1E  (56q), H2-2E  (56q), Helios-E  (98q) -> SIMULATED",
            "  Tier 3 (Live Nexus QPU) : H2-1   (56q), H2-2   (56q), Helios    (98q) -> Signed Token + HQC Cap",
            "CLI commands: osiris propose | osiris verify | osiris bench --compare | osiris grant-tier3 | osiris run | osiris ledger verify",
        ]
        return "\n".join(lines)
    if sub == "ledger":
        summary = verify_all_quantum_ledgers(osiris_home=osiris_home)
        nx_info = summary["ledgers"]["nexus_receipts"]
        verdict = "VALID" if summary["valid"] else "TAMPERED"
        return (
            f"Quantum Ledgers [{verdict}] | nexus_receipts={nx_info['entries']} entries "
            f"| tail={nx_info['tail_hash'][:16]}…"
        )
    if sub == "bench":
        cpath = tokens[1] if len(tokens) > 1 else None
        if cpath:
            src, fname = _read_circuit_file(cpath)
        else:
            src, fname = _TEMPLATE_CODE422, "code422_syndrome_detect.qasm"
        rep = compare_topologies(src, filename=fname)
        return rep.to_markdown()
    if sub == "verify" and len(tokens) > 1:
        src, fname = _read_circuit_file(tokens[1])
        rep = verify_quantinuum_circuit(
            src, target="H2-1SC", emulate="qccd", shots=256, record_ledger=False, filename=fname
        )
        return (
            f"[VERIFIED] {fname} on {rep.target}: qubits={rep.n_qubits}/{rep.max_qubits}, "
            f"2Q={rep.two_qubit_gates}, mid_meas={rep.mid_circuit_measurements}, "
            f"resets={rep.resets}, feedforward={rep.feedforward_conditionals}, HQC={rep.hqc_estimate}"
        )
    return "Usage: /quantinuum [targets | verify <file.qasm> | bench [file.qasm] | ledger]"


def dispatch_quantinuum_argv(argv: List[str]) -> Optional[int]:
    """Return exit code if `argv` invokes a Quantinuum / ledger CLI subcommand, else `None`."""
    if not argv:
        return None
    head = argv[0]
    rest = argv[1:]
    if head == "propose":
        return cmd_propose(rest)
    if head == "verify":
        return cmd_verify(rest)
    if head == "bench":
        return cmd_bench(rest)
    if head == "grant-tier3":
        return cmd_grant_tier3(rest)
    if head == "run":
        return cmd_run(rest)
    if head == "ledger":
        return cmd_ledger(rest)
    if head in ("quantinuum", "/quantinuum"):
        try:
            print(handle_repl_quantinuum(" ".join(rest)))
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    return None
