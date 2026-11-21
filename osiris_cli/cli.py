"""Governed OSIRIS Quantum Execution & Quantinuum Nexus CLI entrypoint.

Subcommands:
  - ``verify``: Lower, compile, check qubit caps, and optionally run Tier 1 syntax check.
  - ``run``:    Execute a circuit under Tier 1/2/3 governance with HQC/shot caps and receipts.
  - ``bench``:  Compare static architecture resource projections across QPUs.
  - ``ledger``: Offline SHA-256 verification of recorded Nexus dispositions.
  - ``sync``:   Bi-directional file sync with ``/api/osiris/sync`` using ``OSIRIS_API_TOKEN``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from osiris_cli.providers.quantinuum_nexus import (
    PREFLIGHT_SYNTAX_CHECKER,
    GovernanceError,
    QuantinuumNexusClient,
    estimate_hqc,
    get_target_tier,
    verify_disposition_receipt,
)
from osiris_cli.providers.tket_wrapper import (
    QuantinuumCompilerOptions,
    _analyze_qasm2,
    compile_for_quantinuum,
    lower_qasm3_to_quantinuum_qasm2,
)

DEFAULT_SYNC_API_URL = "https://www.quantumagentlab.com"
DEFAULT_LEDGER_PATH = "~/.osiris/nexus_ledger.jsonl"

SyncHttpHook = Callable[
    [str, str, Dict[str, str], Optional[Dict[str, Any]]],
    Tuple[int, Dict[str, Any]],
]

_BENCH_BACKENDS: Dict[str, Dict[str, Any]] = {
    "quantinuum_h2_1": {
        "label": "Quantinuum H2-1",
        "architecture": "QCCD Trapped-Ion (All-to-All)",
        "quantinuum_target": "H2-1",
        "qubit_cap": 56,
        "all_to_all": True,
        "native_mcm": True,
        "cost_unit": "HQC",
    },
    "quantinuum_h2_2": {
        "label": "Quantinuum H2-2",
        "architecture": "QCCD Trapped-Ion (All-to-All)",
        "quantinuum_target": "H2-2",
        "qubit_cap": 56,
        "all_to_all": True,
        "native_mcm": True,
        "cost_unit": "HQC",
    },
    "quantinuum_helios": {
        "label": "Quantinuum Helios",
        "architecture": "QCCD Trapped-Ion (All-to-All)",
        "quantinuum_target": "Helios",
        "qubit_cap": 98,
        "all_to_all": True,
        "native_mcm": True,
        "cost_unit": "HQC",
    },
    "ibm_fez": {
        "label": "IBM Fez (Heron r2)",
        "architecture": "Superconducting Heavy-Hex",
        "quantinuum_target": None,
        "qubit_cap": 156,
        "all_to_all": False,
        "native_mcm": True,
        "cost_unit": "QPU-s",
    },
    "ibm_marrakesh": {
        "label": "IBM Marrakesh (Heron r2)",
        "architecture": "Superconducting Heavy-Hex",
        "quantinuum_target": None,
        "qubit_cap": 156,
        "all_to_all": False,
        "native_mcm": True,
        "cost_unit": "QPU-s",
    },
    "ibm_kingston": {
        "label": "IBM Kingston (Heron r2)",
        "architecture": "Superconducting Heavy-Hex",
        "quantinuum_target": None,
        "qubit_cap": 156,
        "all_to_all": False,
        "native_mcm": True,
        "cost_unit": "QPU-s",
    },
}


def _read_circuit_file(file_path: str) -> str:
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Circuit file not found: {file_path}")
    return p.read_text(encoding="utf-8")


def _estimate_heavy_hex_swaps(n_qubits: int, two_qubit_gates: int) -> int:
    """Deterministic routing-SWAP overhead heuristic for heavy-hex lattices."""
    if two_qubit_gates <= 0 or n_qubits <= 2:
        return 0
    # Heavy-hex max degree is 3; non-local couplings scale ~0.5 * 2Q gates for n > 2.
    return max(1, int(round(0.5 * two_qubit_gates))) if n_qubits > 3 else 0


def _estimate_ibm_qpu_seconds(
    shots: int,
    depth: int,
    has_mid_circuit_measurement: bool,
) -> float:
    # Conservative static wall-clock model: ~0.25 ms/shot base + dynamic feedback overhead.
    per_shot_s = 0.00025 + (0.00001 * max(1, depth))
    if has_mid_circuit_measurement:
        per_shot_s += 0.00015
    return round(1.5 + float(shots) * per_shot_s, 3)


def cmd_verify(
    args: argparse.Namespace,
    client: Optional[QuantinuumNexusClient] = None,
) -> int:
    try:
        source = _read_circuit_file(args.file)
        opts = QuantinuumCompilerOptions(
            tket_opt_level=args.tket_opt_level,
            no_opt=args.no_opt,
            no_reduce=args.no_reduce,
        )
        canonical_target, tier = get_target_tier(args.target)
        compilation = compile_for_quantinuum(source, target=canonical_target, options=opts)
        nexus = client if client is not None else QuantinuumNexusClient()

        if args.dry_run:
            disp = nexus.execute(
                source=source,
                target=PREFLIGHT_SYNTAX_CHECKER[canonical_target],
                shots=1,
                options=opts,
                record_ledger=args.record_ledger,
                dry_run=True,
            )
            mode_label = "DRY-RUN (local compile + static governance check)"
        else:
            disp = nexus.verify(
                source=source,
                target=canonical_target,
                options=opts,
                record_ledger=args.record_ledger,
                dry_run=False,
            )
            mode_label = f"NEXUS SYNTAX CHECK ({disp.target})"

        m = compilation.metrics
        est_hqc_1000 = estimate_hqc(compilation, shots=1000, target=canonical_target)
        print(f"OSIRIS Verify :: {mode_label}")
        print(f"  File              : {args.file}")
        print(f"  Target            : {canonical_target} (Tier {tier})")
        print(f"  Disposition       : {disp.disposition} (hardware={disp.hardware}, status={disp.status})")
        print(f"  Compiler Backend  : {compilation.compiler_backend}")
        print(f"  Source SHA-256    : {compilation.source_sha256}")
        print(f"  Compiled SHA-256  : {compilation.compiled_sha256}")
        print(
            f"  Circuit Metrics   : qubits={m.n_qubits}, cbits={m.n_cbits}, "
            f"gates={m.total_gates} (2Q={m.two_qubit_gates}), depth={m.depth}"
        )
        print(
            f"  Dynamic Features  : mid_circuit_measure={m.has_mid_circuit_measurement}, "
            f"reset={m.has_reset}, classical_control={m.has_classical_control}"
        )
        print(f"  Estimated HQC     : {disp.estimated_hqc:.4f} (at 1000 shots on {canonical_target}: {est_hqc_1000:.4f})")
        print(f"  Receipt SHA-256   : {disp.receipt_sha256}")
        if args.record_ledger:
            print(f"  Ledger Recorded   : {args.record_ledger}")
        return 0
    except (FileNotFoundError, ValueError, GovernanceError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def cmd_run(
    args: argparse.Namespace,
    client: Optional[QuantinuumNexusClient] = None,
) -> int:
    try:
        source = _read_circuit_file(args.file)
        opts = QuantinuumCompilerOptions(
            tket_opt_level=args.tket_opt_level,
            no_opt=args.no_opt,
            no_reduce=args.no_reduce,
        )
        nexus = client if client is not None else QuantinuumNexusClient(
            max_shots=args.max_shots,
            budget_hqc=args.budget_hqc,
        )
        disp = nexus.execute(
            source=source,
            target=args.target,
            shots=args.shots,
            options=opts,
            max_shots=args.max_shots,
            budget_hqc=args.budget_hqc,
            approval_token=args.approval_token,
            record_ledger=args.record_ledger,
            dry_run=args.dry_run,
        )
        print(f"OSIRIS Run :: {disp.target} (Tier {disp.tier})")
        print(f"  Disposition       : {disp.disposition} (hardware={disp.hardware})")
        print(f"  Status            : {disp.status}")
        print(f"  Job ID            : {disp.job_id}")
        print(f"  Shots             : {disp.shots}")
        print(f"  Estimated HQC     : {disp.estimated_hqc:.4f}")
        print(f"  Compiled SHA-256  : {disp.compiled_sha256}")
        print(f"  Receipt SHA-256   : {disp.receipt_sha256}")
        if args.record_ledger:
            print(f"  Ledger Recorded   : {args.record_ledger}")
        return 0
    except (FileNotFoundError, ValueError, GovernanceError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def cmd_bench(args: argparse.Namespace) -> int:
    try:
        source = _read_circuit_file(args.file)
        if args.shots <= 0:
            raise ValueError(f"--shots must be positive; got {args.shots}")

        opts = QuantinuumCompilerOptions(
            tket_opt_level=args.tket_opt_level,
            no_opt=args.no_opt,
            no_reduce=args.no_reduce,
        )
        requested = [b.strip().lower() for b in args.compare.split(",") if b.strip()]
        if not requested:
            raise ValueError("--compare must specify at least one backend")

        unknown = [b for b in requested if b not in _BENCH_BACKENDS]
        if unknown:
            known = ", ".join(sorted(_BENCH_BACKENDS))
            raise ValueError(
                f"Unknown benchmark backend(s): {', '.join(unknown)}; supported: {known}"
            )

        source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
        lowered_qasm2 = lower_qasm3_to_quantinuum_qasm2(source)
        base_metrics = _analyze_qasm2(lowered_qasm2)
        compiled_sha256 = hashlib.sha256(lowered_qasm2.encode("utf-8")).hexdigest()

        rows: List[Dict[str, Any]] = []
        for key in requested:
            spec = _BENCH_BACKENDS[key]
            cap = int(spec["qubit_cap"])
            fits = base_metrics.n_qubits <= cap

            q_target = spec["quantinuum_target"]
            if q_target is not None and fits:
                comp = compile_for_quantinuum(source, target=q_target, options=opts)
                m = comp.metrics
                est_hqc = estimate_hqc(comp, shots=args.shots, target=q_target)
                routing_swaps = 0
                effective_2q = m.two_qubit_gates
                cost_display = f"{est_hqc:.2f} HQC"
                cost_value = est_hqc
                backend_compiler = comp.compiler_backend
                compiled_sha256 = comp.compiled_sha256
            else:
                m = base_metrics
                routing_swaps = (
                    0
                    if spec["all_to_all"]
                    else _estimate_heavy_hex_swaps(m.n_qubits, m.two_qubit_gates)
                )
                effective_2q = m.two_qubit_gates + 3 * routing_swaps
                if q_target is not None:
                    cost_value = None
                    cost_display = "EXCEEDS_CAP"
                else:
                    qpu_s = _estimate_ibm_qpu_seconds(
                        args.shots,
                        m.depth + 3 * routing_swaps,
                        m.has_mid_circuit_measurement,
                    )
                    cost_value = qpu_s
                    cost_display = f"~{qpu_s:.2f} QPU-s"
                backend_compiler = "static_fallback"

            rows.append(
                {
                    "backend": key,
                    "label": spec["label"],
                    "architecture": spec["architecture"],
                    "qubits_used": m.n_qubits,
                    "qubit_cap": cap,
                    "fits": fits,
                    "logical_2q_gates": m.two_qubit_gates,
                    "estimated_routing_swaps": routing_swaps,
                    "effective_2q_gates": effective_2q,
                    "depth": m.depth,
                    "has_mid_circuit_measurement": m.has_mid_circuit_measurement,
                    "has_reset": m.has_reset,
                    "has_classical_control": m.has_classical_control,
                    "cost_unit": spec["cost_unit"],
                    "estimated_cost": cost_value,
                    "cost_display": cost_display,
                    "compiler_backend": backend_compiler,
                }
            )

        report = {
            "file": args.file,
            "shots": args.shots,
            "source_sha256": source_sha256,
            "compiled_sha256": compiled_sha256,
            "hardware_executed": False,
            "metrics": base_metrics.to_dict(),
            "results": rows,
        }

        if args.format == "json":
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0

        print(f"# OSIRIS Cross-Architecture Benchmark Projection (`{args.file}`, {args.shots} shots)")
        print()
        print(
            "| Backend | Architecture | Qubits | Fits | 2Q Gates | Est. SWAPs | "
            "Eff. 2Q | Depth | Mid-Circuit / Reset | Est. Cost |"
        )
        print("|---|---|---:|:---:|---:|---:|---:|---:|:---:|---:|")
        for r in rows:
            mcm = "Yes" if (r["has_mid_circuit_measurement"] or r["has_reset"]) else "No"
            fits_str = "YES" if r["fits"] else "NO"
            print(
                f"| `{r['backend']}` | {r['architecture']} | "
                f"{r['qubits_used']}/{r['qubit_cap']} | {fits_str} | "
                f"{r['logical_2q_gates']} | {r['estimated_routing_swaps']} | "
                f"{r['effective_2q_gates']} | {r['depth']} | {mcm} | {r['cost_display']} |"
            )
        print()
        print(f"- Source SHA-256: `{source_sha256}`")
        print(f"- Compiled SHA-256: `{compiled_sha256}`")
        print("- Note: Static architecture projection only (`hardware_executed=false`).")
        return 0
    except (FileNotFoundError, ValueError, GovernanceError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def _load_receipt_entries(target_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    entries: List[Tuple[str, Dict[str, Any]]] = []
    if target_path.is_dir():
        files = sorted(
            p for p in target_path.iterdir()
            if p.is_file() and p.suffix.lower() in (".json", ".jsonl")
        )
        for fp in files:
            entries.extend(_load_receipt_entries(fp))
        return entries

    raw = target_path.read_text(encoding="utf-8")
    if target_path.suffix.lower() == ".jsonl":
        for idx, line in enumerate(raw.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if not isinstance(obj, dict):
                raise ValueError(f"{target_path}:{idx} is not a JSON object")
            entries.append((f"{target_path}:{idx}", obj))
    else:
        obj = json.loads(raw)
        if isinstance(obj, list):
            for idx, item in enumerate(obj, start=1):
                if not isinstance(item, dict):
                    raise ValueError(f"{target_path}[{idx}] is not a JSON object")
                entries.append((f"{target_path}[{idx}]", item))
        elif isinstance(obj, dict):
            entries.append((str(target_path), obj))
        else:
            raise ValueError(f"{target_path} does not contain a JSON disposition object")
    return entries


def cmd_ledger_verify(args: argparse.Namespace) -> int:
    path_str = args.receipt or os.environ.get("OSIRIS_NEXUS_LEDGER") or DEFAULT_LEDGER_PATH
    target_path = Path(path_str).expanduser()
    if not target_path.exists():
        print(f"ERROR: Ledger receipt path not found: {target_path}", file=sys.stderr)
        return 1

    try:
        entries = _load_receipt_entries(target_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: Failed to parse ledger receipt(s): {exc}", file=sys.stderr)
        return 1

    if not entries:
        print(f"ERROR: No disposition receipts found in {target_path}", file=sys.stderr)
        return 1

    passed = 0
    failed = 0
    for label, entry in entries:
        ok = verify_disposition_receipt(entry)
        target = entry.get("target", "?")
        tier = entry.get("tier", "?")
        disposition = entry.get("disposition", "?")
        if ok:
            passed += 1
            print(f"[PASS] {label} :: target={target} tier={tier} disposition={disposition}")
        else:
            failed += 1
            print(
                f"[FAIL] {label} :: TAMPERED OR INVALID RECEIPT "
                f"(target={target} tier={tier} disposition={disposition})",
                file=sys.stderr,
            )

    print(f"Ledger verification complete: {passed}/{len(entries)} valid, {failed} failed")
    return 0 if failed == 0 else 1


def _validate_sync_api_url(api_url: str) -> str:
    parsed = urllib.parse.urlparse(api_url.strip())
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    if scheme == "https":
        return api_url.rstrip("/")
    if scheme == "http" and host in ("localhost", "127.0.0.1", "::1"):
        return api_url.rstrip("/")
    raise ValueError(
        f"Refusing to send bearer token over insecure URL {api_url!r}; use https://"
    )


def _sync_http_call(
    method: str,
    url: str,
    headers: Dict[str, str],
    payload: Optional[Dict[str, Any]],
    http_request: Optional[SyncHttpHook] = None,
    timeout: float = 30.0,
) -> Tuple[int, Dict[str, Any]]:
    if http_request is not None:
        return http_request(method.upper(), url, headers, payload)

    data = json.dumps(payload, sort_keys=True).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = int(getattr(resp, "status", 200))
            raw = resp.read().decode("utf-8", errors="replace")
            parsed = json.loads(raw) if raw.strip() else {}
            return status, parsed if isinstance(parsed, dict) else {"data": parsed}
    except urllib.error.HTTPError as exc:
        raw_err = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        try:
            err_obj = json.loads(raw_err) if raw_err.strip() else {}
            if not isinstance(err_obj, dict):
                err_obj = {"error": raw_err}
        except ValueError:
            err_obj = {"error": raw_err or str(exc)}
        return int(exc.code), err_obj
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Sync network error: {exc.reason}") from exc


def _safe_relative_sync_path(base_dir: Path, rel_path: str) -> Path:
    raw = rel_path.replace("\\", "/").lstrip()
    if not raw or raw.startswith("/") or ".." in Path(raw).parts:
        raise ValueError(f"Unsafe sync path rejected: {rel_path!r}")
    resolved_base = base_dir.resolve()
    candidate = (resolved_base / raw).resolve()
    try:
        candidate.relative_to(resolved_base)
    except ValueError as exc:
        raise ValueError(f"Sync path escapes workspace root: {rel_path!r}") from exc
    return candidate


def _format_sync_conflict(data: Dict[str, Any], branch: str) -> str:
    lines = [
        "ERROR: 409 SYNC_CONFLICT — remote workspace has diverged; local files were NOT overwritten.",
    ]
    conflicts = data.get("conflicts")
    if isinstance(conflicts, list) and conflicts:
        for item in conflicts:
            if not isinstance(item, dict):
                continue
            path = item.get("path", "<unknown>")
            local_sha = item.get("local_sha256", "unknown")
            remote_sha = item.get("remote_sha256", "unknown")
            remote_rev = item.get("remote_revision", item.get("server_revision", "unknown"))
            lines.append(
                f"  - {path}: local_sha256={local_sha} vs remote_sha256={remote_sha} "
                f"(remote_revision={remote_rev})"
            )
    elif data.get("message"):
        lines.append(f"  Detail: {data['message']}")
    lines.append("Resolution guidance:")
    lines.append(f"  1. Fetch latest remote state: osiris sync pull --branch {branch}")
    lines.append("  2. Inspect differences (e.g. `git diff` or `diff -u`) and reconcile edits.")
    lines.append(f"  3. Re-run `osiris sync push --branch {branch} <files...>` once merged.")
    return "\n".join(lines)


def _read_sync_state(state_path: Path) -> Dict[str, Dict[str, Any]]:
    if not state_path.is_file():
        return {}
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        files = data.get("files") if isinstance(data, dict) else {}
        return files if isinstance(files, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_sync_state(state_path: Path, files_state: Dict[str, Dict[str, Any]]) -> None:
    payload = {
        "schema_version": "osiris.sync.state.v1",
        "files": files_state,
    }
    raw = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd = os.open(str(state_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(raw)


def cmd_sync(
    args: argparse.Namespace,
    http_request: Optional[SyncHttpHook] = None,
) -> int:
    token = os.environ.get("OSIRIS_API_TOKEN", "").strip()
    if not token:
        print(
            "ERROR: OSIRIS_API_TOKEN environment variable is required for `osiris sync` "
            "(tokens are read strictly from environment and never persisted to disk).",
            file=sys.stderr,
        )
        return 2

    try:
        api_base = _validate_sync_api_url(args.api_url)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    endpoint = f"{api_base}/api/osiris/sync"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "osiris-cli/6.0.0",
    }
    base_dir = Path(getattr(args, "dir", None) or ".").resolve()
    state_path = base_dir / ".osiris_sync_state.json"
    state = _read_sync_state(state_path)

    try:
        if args.sync_action == "push":
            file_paths = list(args.files or [])
            if not file_paths:
                file_paths = sorted(
                    str(p.relative_to(base_dir))
                    for p in base_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in (".qasm", ".dna")
                )
            if not file_paths:
                print("ERROR: No files specified or discovered (.qasm/.dna) to push.", file=sys.stderr)
                return 2

            file_entries: List[Dict[str, Any]] = []
            for fp in file_paths:
                full = Path(fp)
                if not full.is_absolute():
                    full = (base_dir / fp).resolve()
                if not full.is_file():
                    raise FileNotFoundError(f"Sync push file not found: {fp}")
                try:
                    rel = str(full.relative_to(base_dir)).replace("\\", "/")
                except ValueError:
                    rel = full.name
                content = full.read_text(encoding="utf-8")
                sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
                prev = state.get(rel, {})
                file_entries.append(
                    {
                        "path": rel,
                        "content": content,
                        "sha256": sha,
                        "base_sha256": prev.get("sha256"),
                        "base_revision": prev.get("revision"),
                    }
                )

            payload: Dict[str, Any] = {
                "action": "push",
                "workspace": args.workspace,
                "org": args.org,
                "branch": args.branch,
                "files": file_entries,
            }
            status, resp = _sync_http_call(
                "POST", endpoint, headers, payload, http_request=http_request
            )
            if status == 409 or str(resp.get("code") or resp.get("error") or "").upper() == "SYNC_CONFLICT":
                print(_format_sync_conflict(resp, args.branch), file=sys.stderr)
                return 3
            if not (200 <= status < 300):
                err = str(resp.get("error", f"HTTP {status}")).replace(token, "[REDACTED]")
                print(f"ERROR: Sync push failed (HTTP {status}): {err}", file=sys.stderr)
                return 1

            rev = resp.get("revision")
            for item in file_entries:
                state[item["path"]] = {"sha256": item["sha256"], "revision": rev}
            _write_sync_state(state_path, state)
            print(
                f"Sync push complete: {len(file_entries)} file(s) synced to "
                f"branch={args.branch} (revision={rev})"
            )
            return 0

        # pull
        query: Dict[str, str] = {"branch": args.branch}
        if args.workspace:
            query["workspace"] = args.workspace
        if args.org:
            query["org"] = args.org
        if args.files:
            query["files"] = ",".join(args.files)
        pull_url = f"{endpoint}?{urllib.parse.urlencode(query)}"

        status, resp = _sync_http_call(
            "GET", pull_url, headers, None, http_request=http_request
        )
        if status == 409 or str(resp.get("code") or resp.get("error") or "").upper() == "SYNC_CONFLICT":
            print(_format_sync_conflict(resp, args.branch), file=sys.stderr)
            return 3
        if not (200 <= status < 300):
            err = str(resp.get("error", f"HTTP {status}")).replace(token, "[REDACTED]")
            print(f"ERROR: Sync pull failed (HTTP {status}): {err}", file=sys.stderr)
            return 1

        remote_files = resp.get("files") or []
        if not isinstance(remote_files, list):
            print("ERROR: Malformed sync pull response (expected 'files' list)", file=sys.stderr)
            return 1

        written = 0
        for item in remote_files:
            if not isinstance(item, dict) or "path" not in item or "content" not in item:
                continue
            rel = str(item["path"])
            content = str(item["content"])
            dest = _safe_relative_sync_path(base_dir, rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
            state[rel] = {"sha256": sha, "revision": item.get("revision", resp.get("revision"))}
            written += 1

        _write_sync_state(state_path, state)
        print(f"Sync pull complete: {written} file(s) updated on branch={args.branch}")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        safe_msg = str(exc).replace(token, "[REDACTED]")
        print(f"ERROR: {safe_msg}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="osiris",
        description="OSIRIS Governed Quantum Execution & Quantinuum Nexus CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # verify
    p_verify = sub.add_parser(
        "verify",
        help="Compile and verify a quantum circuit against Quantinuum constraints",
    )
    p_verify.add_argument("file", help="Path to .qasm or .dna circuit file")
    p_verify.add_argument("--target", default="H2-1SC", help="Quantinuum target (default: H2-1SC)")
    p_verify.add_argument("--tket-opt-level", type=int, default=2, choices=[0, 1, 2])
    p_verify.add_argument("--no-opt", action="store_true", help="Disable all optimization passes")
    p_verify.add_argument("--no-reduce", action="store_true", help="Disable redundancy removal")
    p_verify.add_argument("--record-ledger", default=None, help="Path to write/append JSON/JSONL receipt")
    p_verify.add_argument("--dry-run", action="store_true", help="Compile & check locally without network calls")

    # run
    p_run = sub.add_parser(
        "run",
        help="Execute a circuit under Tier 1/2/3 Quantinuum Nexus governance",
    )
    p_run.add_argument("file", help="Path to .qasm or .dna circuit file")
    p_run.add_argument("--target", required=True, help="Quantinuum target (e.g. H2-1SC, H2-1E, H2-1, Helios)")
    p_run.add_argument("--shots", type=int, required=True, help="Shot count")
    p_run.add_argument("--budget-hqc", type=float, default=50.0, help="Maximum HQC budget cap")
    p_run.add_argument("--max-shots", type=int, default=1000, help="Maximum shot cap")
    p_run.add_argument("--approval-token", default=None, help="HMAC-SHA256 Tier 3 hardware approval token")
    p_run.add_argument("--record-ledger", default=None, help="Path to write/append JSON/JSONL receipt")
    p_run.add_argument("--tket-opt-level", type=int, default=2, choices=[0, 1, 2])
    p_run.add_argument("--no-opt", action="store_true")
    p_run.add_argument("--no-reduce", action="store_true")
    p_run.add_argument("--dry-run", action="store_true")

    # bench
    p_bench = sub.add_parser(
        "bench",
        help="Compare static circuit resource projections across quantum architectures",
    )
    p_bench.add_argument("file", help="Path to .qasm or .dna circuit file")
    p_bench.add_argument(
        "--compare",
        default="quantinuum_h2_1,quantinuum_helios,ibm_fez",
        help="Comma-separated backend identifiers",
    )
    p_bench.add_argument("--shots", type=int, default=1024, help="Shot count for projection")
    p_bench.add_argument("--format", choices=["markdown", "json"], default="markdown")
    p_bench.add_argument("--tket-opt-level", type=int, default=2, choices=[0, 1, 2])
    p_bench.add_argument("--no-opt", action="store_true")
    p_bench.add_argument("--no-reduce", action="store_true")

    # ledger
    p_ledger = sub.add_parser(
        "ledger",
        help="Verify offline cryptographic receipts for recorded Nexus dispositions",
    )
    ledger_sub = p_ledger.add_subparsers(dest="ledger_action", required=True)
    p_ledger_verify = ledger_sub.add_parser(
        "verify",
        help="Verify SHA-256 receipt integrity and attestation invariants offline",
    )
    p_ledger_verify.add_argument(
        "receipt",
        nargs="?",
        default=None,
        help=f"Receipt .json, .jsonl, or directory (default: {DEFAULT_LEDGER_PATH})",
    )

    # sync
    p_sync = sub.add_parser(
        "sync",
        help="Bi-directional workspace sync with /api/osiris/sync (token via OSIRIS_API_TOKEN)",
    )
    sync_sub = p_sync.add_subparsers(dest="sync_action", required=True)
    for sync_act, sync_help in (
        ("pull", "Pull workspace files from /api/osiris/sync"),
        ("push", "Push local files to /api/osiris/sync"),
    ):
        sp = sync_sub.add_parser(sync_act, help=sync_help)
        sp.add_argument("--workspace", default=None, help="Workspace identifier")
        sp.add_argument("--org", default=None, help="Organization identifier")
        sp.add_argument("--branch", default="main", help="Target sync branch (default: main)")
        sp.add_argument("--api-url", default=DEFAULT_SYNC_API_URL, help="Sync API base URL")
        sp.add_argument("--dir", default=".", help="Local workspace directory")
        sp.add_argument("files", nargs="*", help="Optional files to push or pull")

    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    nexus_client: Optional[QuantinuumNexusClient] = None,
    sync_http_request: Optional[SyncHttpHook] = None,
) -> int:
    """Execute a governed OSIRIS CLI subcommand and return an integer exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "verify":
        return cmd_verify(args, client=nexus_client)
    if args.command == "run":
        return cmd_run(args, client=nexus_client)
    if args.command == "bench":
        return cmd_bench(args)
    if args.command == "ledger" and args.ledger_action == "verify":
        return cmd_ledger_verify(args)
    if args.command == "sync":
        return cmd_sync(args, http_request=sync_http_request)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
