"""
OSIRIS Governed Workspace AST Cartographer & Ledger Utility (Node Alpha / S26 Ultra & Node Beta)
================================================================================================

Replaces legacy un-governed mobile cartographer prototypes with a deterministic,
Claims-Firewall-compliant (SYS-024) workspace auditor:
  1. Zero refuted CRSM numerology (no theta_lock 51.843 deg or fake SU(2) 1.7975 density targets).
  2. Dynamic Snapdragon 8 Elite / ARM64 Prime-core discovery via Linux cpufreq sysfs
     (correctly resolving the 2+6 Oryon topology on Galaxy S26 Ultra rather than hardcoded 4..7).
  3. Accurate AST execution-hazard gating (flags eval/exec/__import__/os.system/shell=True
     without falsely rejecting standard read-only file open() or ast.parse()).
  4. Append-only SHA-256 hash-chained JSONL ledger (`semantic_ledger.jsonl`) with full
     chain verification, filtered CSV export, and dynamic-width ASCII terminal table rendering.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import os
import platform
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple

GENESIS_HASH = "0" * 64
FORBIDDEN_BUILTINS = frozenset({"eval", "exec", "__import__"})
FORBIDDEN_ATTR_CALLS = frozenset({("os", "system"), ("os", "popen"), ("pty", "spawn")})


def detect_prime_cores(sysfs_cpu_root: str = "/sys/devices/system/cpu") -> List[int]:
    """Discover highest-frequency Prime CPU cores via cpufreq sysfs.

    On Snapdragon 8 Elite / Gen 5 (Galaxy S26 Ultra, 2+6 Oryon topology), cores 6-7
    operate at the Prime clock ceiling (~4.32-4.74 GHz) while cores 0-5 form the
    Performance cluster. Falls back to [6, 7] on 8-core ARM64 when SELinux masks cpufreq.
    """
    freqs: Dict[int, int] = {}
    root = Path(sysfs_cpu_root)
    if root.is_dir():
        for entry in sorted(root.iterdir(), key=lambda p: p.name):
            if not (entry.name.startswith("cpu") and entry.name[3:].isdigit()):
                continue
            cpu_id = int(entry.name[3:])
            for fname in ("cpuinfo_max_freq", "scaling_max_freq"):
                fpath = entry / "cpufreq" / fname
                try:
                    val = int(fpath.read_text(encoding="utf-8").strip())
                    if val > 0:
                        freqs[cpu_id] = val
                        break
                except OSError:
                    continue
                except ValueError:
                    continue
    if freqs:
        max_freq = max(freqs.values())
        prime = sorted(cid for cid, hz in freqs.items() if hz == max_freq)
        if prime:
            return prime

    total = os.cpu_count() or 1
    arch = platform.machine().lower()
    if arch in ("aarch64", "arm64") and total == 8:
        return [6, 7]
    if total >= 4:
        return list(range(total // 2, total))
    return list(range(total))


def bind_process_to_prime_cores(cores: Optional[Sequence[int]] = None) -> List[int]:
    """Bind current process CPU affinity to Prime cores when supported by the OS kernel."""
    target = list(cores) if cores is not None else detect_prime_cores()
    sched_set = getattr(os, "sched_setaffinity", None)
    if callable(sched_set) and target:
        try:
            sched_set(0, set(target))
        except OSError:
            pass
    return target


def _max_ast_depth(node: ast.AST, depth: int = 0) -> int:
    children = list(ast.iter_child_nodes(node))
    if not children:
        return depth
    return max(_max_ast_depth(ch, depth + 1) for ch in children)


def analyze_python_ast(source_code: str) -> Dict[str, Any]:
    """Parse Python source and check for dynamic execution or shell-injection hazards."""
    tree = ast.parse(source_code, mode="exec")
    node_count = 0
    fn_count = 0
    cls_count = 0
    imp_count = 0
    violations: List[str] = []

    for node in ast.walk(tree):
        node_count += 1
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn_count += 1
        elif isinstance(node, ast.ClassDef):
            cls_count += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imp_count += 1
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_BUILTINS:
                violations.append(f"forbidden builtin call '{node.func.id}()'")
            elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                pair = (node.func.value.id, node.func.attr)
                if pair in FORBIDDEN_ATTR_CALLS:
                    violations.append(f"forbidden shell call '{pair[0]}.{pair[1]}()'")
                elif pair[0] == "subprocess":
                    for kw in node.keywords:
                        if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            violations.append("subprocess call with shell=True")

    return {
        "ast_nodes": node_count,
        "functions": fn_count,
        "classes": cls_count,
        "imports": imp_count,
        "max_depth": _max_ast_depth(tree),
        "violations": violations,
    }


def inspect_workspace_file(filepath: str) -> Dict[str, Any]:
    path = Path(filepath)
    raw_bytes = path.read_bytes()
    file_hash = hashlib.sha256(raw_bytes).hexdigest()
    content = raw_bytes.decode("utf-8", errors="replace")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if path.suffix == ".py":
        try:
            metrics = analyze_python_ast(content)
        except SyntaxError as exc:
            return {
                "timestamp": ts,
                "file": str(path),
                "sha256": file_hash,
                "file_bytes": len(raw_bytes),
                "ast_nodes": 0,
                "functions": 0,
                "classes": 0,
                "max_depth": 0,
                "status": "REJECTED_POLICY_VIOLATION",
                "reason": f"SyntaxError: {exc.msg} (line {exc.lineno})",
            }
        if metrics["violations"]:
            return {
                "timestamp": ts,
                "file": str(path),
                "sha256": file_hash,
                "file_bytes": len(raw_bytes),
                "ast_nodes": metrics["ast_nodes"],
                "functions": metrics["functions"],
                "classes": metrics["classes"],
                "max_depth": metrics["max_depth"],
                "status": "REJECTED_POLICY_VIOLATION",
                "reason": "; ".join(metrics["violations"]),
            }
        return {
            "timestamp": ts,
            "file": str(path),
            "sha256": file_hash,
            "file_bytes": len(raw_bytes),
            "ast_nodes": metrics["ast_nodes"],
            "functions": metrics["functions"],
            "classes": metrics["classes"],
            "max_depth": metrics["max_depth"],
            "status": "APPROVED_READ_ONLY",
            "reason": "",
        }

    # .dna source files: record cryptographic digest and line/byte geometry
    return {
        "timestamp": ts,
        "file": str(path),
        "sha256": file_hash,
        "file_bytes": len(raw_bytes),
        "ast_nodes": len([ln for ln in content.splitlines() if ln.strip()]),
        "functions": 0,
        "classes": 0,
        "max_depth": 1,
        "status": "APPROVED_READ_ONLY",
        "reason": "",
    }


def _entry_digest(body: Dict[str, Any]) -> str:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _last_ledger_hash(ledger_path: Path) -> str:
    if not ledger_path.is_file():
        return GENESIS_HASH
    last = GENESIS_HASH
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            last = str(rec.get("hash") or last)
    return last


def verify_semantic_ledger(ledger_path: str | Path) -> Tuple[bool, int, str]:
    """Verify hash-chain continuity across semantic_ledger.jsonl."""
    p = Path(ledger_path)
    if not p.is_file():
        return True, 0, GENESIS_HASH
    prev = GENESIS_HASH
    count = 0
    for idx, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        rec = json.loads(line)
        recorded_hash = rec.pop("hash", None)
        if rec.get("prev_hash") != prev:
            return False, idx, prev
        if _entry_digest(rec) != recorded_hash:
            return False, idx, prev
        prev = str(recorded_hash)
        count += 1
    return True, count, prev


def run_cartography(
    workspace_root: str | Path,
    ledger_path: Optional[str | Path] = None,
    *,
    max_workers: int = 4,
    lock_prime_cores: bool = False,
) -> Dict[str, Any]:
    root = Path(workspace_root).expanduser().resolve()
    out_ledger = Path(ledger_path).expanduser().resolve() if ledger_path else (root / "semantic_ledger.jsonl")
    bound_cores = bind_process_to_prime_cores() if lock_prime_cores else detect_prime_cores()

    skip_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", ".next", ".pytest_cache"}
    targets: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in skip_dirs)
        for fname in sorted(filenames):
            if fname.endswith((".py", ".dna")):
                targets.append(os.path.join(dirpath, fname))

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        inspected = list(pool.map(inspect_workspace_file, targets))

    out_ledger.parent.mkdir(parents=True, exist_ok=True)
    prev = _last_ledger_hash(out_ledger)
    chained_entries: List[Dict[str, Any]] = []
    with out_ledger.open("a", encoding="utf-8") as lf:
        for row in inspected:
            body = {**row, "prev_hash": prev}
            digest = _entry_digest(body)
            full = {**body, "hash": digest}
            lf.write(json.dumps(full, sort_keys=True) + "\n")
            prev = digest
            chained_entries.append(full)

    approved = sum(1 for r in chained_entries if r["status"] == "APPROVED_READ_ONLY")
    rejected = sum(1 for r in chained_entries if r["status"] == "REJECTED_POLICY_VIOLATION")
    return {
        "workspace_root": str(root),
        "ledger_path": str(out_ledger),
        "prime_cores": bound_cores,
        "processed": len(chained_entries),
        "approved": approved,
        "rejected": rejected,
        "head_hash": prev,
    }


def load_ledger_records(ledger_path: str | Path, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    p = Path(ledger_path).expanduser()
    if not p.is_file():
        return []
    filt = status_filter.upper() if status_filter else None
    rows: List[Dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if filt and str(rec.get("status", "")).upper() != filt:
            continue
        rows.append(rec)
    return rows


def summarize_ledger(ledger_path: str | Path) -> Dict[str, Any]:
    valid, total, head = verify_semantic_ledger(ledger_path)
    rows = load_ledger_records(ledger_path)
    approved = [r for r in rows if r.get("status") == "APPROVED_READ_ONLY"]
    rejected = [r for r in rows if r.get("status") == "REJECTED_POLICY_VIOLATION"]
    return {
        "chain_valid": valid,
        "total_records": total,
        "approved_count": len(approved),
        "rejected_count": len(rejected),
        "mean_ast_nodes": round(mean(r.get("ast_nodes", 0) for r in approved), 2) if approved else 0.0,
        "mean_max_depth": round(mean(r.get("max_depth", 0) for r in approved), 2) if approved else 0.0,
        "head_hash": head,
    }


def export_ledger_csv(
    ledger_path: str | Path,
    csv_path: str | Path,
    status_filter: Optional[str] = None,
) -> int:
    rows = load_ledger_records(ledger_path, status_filter=status_filter)
    columns = [
        "timestamp",
        "file",
        "sha256",
        "status",
        "ast_nodes",
        "functions",
        "classes",
        "max_depth",
        "reason",
        "hash",
    ]
    out = Path(csv_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in columns})
    return len(rows)


def render_ascii_table(csv_path: str | Path, status_filter: Optional[str] = None) -> str:
    p = Path(csv_path).expanduser()
    if not p.is_file():
        return f"[!] CSV report not found at {p}"
    with p.open("r", encoding="utf-8") as f:
        all_rows = list(csv.reader(f))
    if len(all_rows) < 2:
        return "[!] CSV report contains no data rows."

    header, raw_data = all_rows[0], all_rows[1:]
    status_idx = header.index("status") if "status" in header else 3
    data = [
        r for r in raw_data if not status_filter or r[status_idx].upper() == status_filter.upper()
    ]
    if not data:
        return f"[!] No records matched filter criteria: '{status_filter}'"

    display_cols = [0, 1, 2, 3, 4, 8] if len(header) >= 9 else list(range(len(header)))
    disp_header = [header[i] for i in display_cols]

    def _fmt_cell(col_name: str, val: str) -> str:
        if col_name == "file" and len(val) > 34:
            return "..." + val[-31:]
        if col_name in ("sha256", "hash") and len(val) > 12:
            return val[:12] + "..."
        if col_name == "reason" and len(val) > 32:
            return val[:29] + "..."
        return val

    formatted_rows = [
        [_fmt_cell(header[i], str(row[i] if i < len(row) else "")) for i in display_cols]
        for row in data
    ]
    widths = [
        max(len(disp_header[c]), *(len(r[c]) for r in formatted_rows))
        for c in range(len(disp_header))
    ]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    buf = io.StringIO()
    buf.write(sep + "\n")
    buf.write("|" + "|".join(f" {disp_header[c].upper():<{widths[c]}} " for c in range(len(widths))) + "|\n")
    buf.write(sep + "\n")
    for r in formatted_rows:
        buf.write("|" + "|".join(f" {r[c]:<{widths[c]}} " for c in range(len(widths))) + "|\n")
    buf.write(sep + "\n")
    buf.write(f"Displayed {len(formatted_rows)} of {len(raw_data)} record(s).\n")
    return buf.getvalue()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m osiris_cli.s26_cartographer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_scan = sub.add_parser("scan", help="Audit workspace ASTs into hash-chained semantic_ledger.jsonl")
    p_scan.add_argument("workspace", nargs="?", default=".")
    p_scan.add_argument("--ledger", default=None)
    p_scan.add_argument("--workers", type=int, default=4)
    p_scan.add_argument("--lock-prime", action="store_true")

    p_sum = sub.add_parser("summary", help="Verify hash chain and summarize semantic_ledger.jsonl")
    p_sum.add_argument("ledger")

    p_csv = sub.add_parser("export-csv", help="Export filtered ledger rows to CSV")
    p_csv.add_argument("ledger")
    p_csv.add_argument("csv_out")
    p_csv.add_argument("--status", default=None)

    p_tbl = sub.add_parser("table", help="Render exported CSV as a filtered terminal ASCII table")
    p_tbl.add_argument("csv_in")
    p_tbl.add_argument("--status", default=None)

    args = ap.parse_args(argv)
    if args.cmd == "scan":
        res = run_cartography(
            args.workspace, args.ledger, max_workers=args.workers, lock_prime_cores=args.lock_prime
        )
        print(json.dumps(res, indent=2))
    elif args.cmd == "summary":
        print(json.dumps(summarize_ledger(args.ledger), indent=2))
    elif args.cmd == "export-csv":
        n = export_ledger_csv(args.ledger, args.csv_out, status_filter=args.status)
        print(f"Exported {n} record(s) to {args.csv_out}")
    elif args.cmd == "table":
        print(render_ascii_table(args.csv_in, status_filter=args.status))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
