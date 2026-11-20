#!/usr/bin/env python3
"""Node Beta (WSL2 / x86_64 / RTX 5050 8 GB) multi-mentor benchmark harness.

Substrate-guarded: refuses to run anywhere except Node Beta (x86_64, detected
via osiris_hardware_mesh when available, platform.machine() otherwise). On
Node Beta it launches Ollama with flash attention and a q8_0 KV cache so
4B-9B mentors fit in 8 GB VRAM, verifies the expected model tags against
/api/tags, then runs the advisory 3-task bench comparison
(gaps, claims, backlog -- the same BENCH_QUICK_TASKS /bench uses) once per
installed mentor.

Advisory only: this script never changes OSIRIS's active routing
(CHAT_MODEL/CODE_MODEL, protege rankings). It writes only the bench ledger
(~/.osiris/bench/results.ledger.jsonl) through osiris_bench.py, exactly as
/bench does. Model-selection evidence stays advisory; deterministic code
decides.

Usage:
    python3 scripts/bench_rtx5050_multimentor.py            # verify + run all installed mentors
    python3 scripts/bench_rtx5050_multimentor.py --check    # preflight only: substrate, env, models
    python3 scripts/bench_rtx5050_multimentor.py --models qwen3.5:9b,gemma4:e4b
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

EXPECTED_MODELS = ("qwen3.5:9b", "qwen3.5:4b", "gemma4:e4b", "qwen2.5-coder:7b")
QUICK_TASKS = ("gaps", "claims", "backlog")
OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
RTX_ENV = {"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q8_0"}


def is_node_beta() -> tuple:
    """(True, description) on Node Beta x86_64; (False, reason) otherwise.
    Probes osiris_hardware_mesh first, platform.machine() as fallback."""
    try:
        import osiris_hardware_mesh as mesh
        if mesh.is_node_beta():
            return True, f"osiris_hardware_mesh: {mesh.get_current_profile().node_type.value}"
    except Exception:
        pass
    arch = platform.machine().lower()
    if arch in ("x86_64", "amd64"):
        return True, f"platform.machine(): {arch} (WSL2 workstation class)"
    return False, f"substrate is {arch!r}, not Node Beta x86_64 -- refusing"


def ollama_installed_models(timeout: float = 2.0) -> list:
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        return sorted(m.get("name") for m in data.get("models", []) if m.get("name"))
    except Exception as e:
        return None  # type: ignore[return-value]


def preflight() -> int:
    ok_node, why = is_node_beta()
    print(f"[*] Substrate: {why}")
    if not ok_node:
        print("[!] Refusing: this harness is Node Beta (x86_64 / RTX 5050) only.")
        return 2
    for k, v in RTX_ENV.items():
        os.environ[k] = v
    print(f"[*] Ollama env for 8 GB VRAM: {RTX_ENV}")
    installed = ollama_installed_models()
    if installed is None:
        print(f"[!] Ollama unreachable at {OLLAMA_BASE} -- start the daemon with the env above, e.g.:")
        print(f"    OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve")
        return 2
    print(f"[*] Installed Ollama models: {', '.join(installed)}")
    missing = [m for m in EXPECTED_MODELS if m not in installed and f"{m}:latest" not in installed]
    if missing:
        print(f"[!] Expected mentor tags not installed: {', '.join(missing)}")
        print(f"    pull them first, e.g.: {' '.join('ollama pull ' + m for m in missing)}")
        print("    (honest drift report: the bench below will only cover what IS installed)")
    else:
        print("[*] All expected mentor tags present.")
    return 0


def run_bench(model: str, k: int = 1) -> int:
    argv = [sys.executable, os.path.join(REPO_ROOT, "osiris_bench.py"),
            "--events", "--backend", "ollama", "--model", model,
            "--tasks", ",".join(QUICK_TASKS), "--k", str(k)]
    print(f"\n[*] Advisory bench round: {model} on {','.join(QUICK_TASKS)} (k={k})")
    print(f"    {' '.join(argv)}")
    return subprocess.run(argv, cwd=REPO_ROOT, stdin=subprocess.DEVNULL).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="preflight only")
    ap.add_argument("--models", default=",".join(EXPECTED_MODELS),
                    help="comma-separated mentor tags to bench (default: the expected RTX 5050 set)")
    ap.add_argument("--k", type=int, default=1, help="candidates per task (default 1)")
    args = ap.parse_args()

    rc = preflight()
    if rc != 0 or args.check:
        return rc

    installed = set(ollama_installed_models() or [])
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    todo = [m for m in requested if m in installed or f"{m}:latest" in installed]
    skipped = [m for m in requested if m not in todo]
    if skipped:
        print(f"\n[!] Skipping (not installed): {', '.join(skipped)}")
    if not todo:
        print("[!] Nothing to bench: none of the requested mentors are installed.")
        return 2
    worst = 0
    for m in todo:
        rc = run_bench(m, k=args.k)
        worst = max(worst, rc if rc not in (0, 130) else 0)
    print("\n[*] Multi-mentor advisory round complete. Evidence is in the bench ledger only;")
    print("    OSIRIS routing was not changed. /mentors reads the ledger to rank mentors.")
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
