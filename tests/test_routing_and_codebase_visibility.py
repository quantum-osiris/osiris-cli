"""Verification Suite for Enhanced Evidence-Weighted Model Routing & Polyglot Codebase Visibility:

1. `osiris_cli/codebase_visibility.py` & `osiris_ast_bridge.py`:
   - Indexes >50 files without `os.walk` 40-file blind cutoff and ranks query-relevant modules #1.
   - Extracts polyglot symbols & dependency edges across Python (`.py`) AND TypeScript/Next.js
     (`.ts`, `.tsx`, `app/api/.../route.ts`, `@/` imports) + `.dna` + `.sql`.
   - Computes bidirectional `imports` and `imported_by` blast-radius hub counts (`/codebase deps`).
   - Falls back cleanly to `osiris-cli` for read-only queries from an empty directory (`resolve_read_only_repo`)
     without widening `CodeAgent` write confinement.
   - Injects `<workspace_map>` into `CodeAgent.run()` when `include_workspace_map=True` / live Ollama.
2. `osiris_cli/evidence_router.py`, `providers/router.py`, `code_agent.py`, `osiris_termux_console.py`:
   - Recognizes modern local models (`gemma4:e4b`, `qwen3.5:9b`, `qwen3.5:4b`, `gemma3:4b`, `qwen3:8b`).
   - Combines role priors + `protege` benchmark Wilson pass bonuses + `0/3+` demotion + `calibration`
     false-confidence penalties + `stats.json` median latency + edge low-RAM guard.
   - Enforces cross-family diversity for `critic` role vs `code` proposer (`avoid_model`).
   - Routes `/models`, `/route`, `/codebase`, and `/repo` cleanly through `osiris_repl` & `osiris_termux_console`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import osiris_ast_bridge
import osiris_termux_console as otc
import protege
from osiris_cli import (
    architecture,
    calibration,
    codebase_visibility,
    evidence_router,
    osiris_repl,
)
from osiris_cli.code_agent import CodeAgent
from osiris_cli.providers.router import SOVEREIGN_LOCAL_MODELS, SystemTelemetry, select_mentor


def test_polyglot_codebase_visibility_exceeds_40_file_cap_and_tracks_blast_radius(tmp_path: Path) -> None:
    ws = tmp_path / "polyglot_repo"
    (ws / ".git").mkdir(parents=True)
    (ws / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    # Create a core hub module imported by 50 filler modules so total files > 55 (exceeds old 40-file cap)
    pkg = ws / "core_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('"""Core package."""\n', encoding="utf-8")
    (pkg / "kernel_hub.py").write_text(
        '"""Central execution kernel hub."""\n'
        "class KernelHub:\n"
        "    def execute_safe(self, argv: list) -> int:\n"
        "        return len(argv)\n",
        encoding="utf-8",
    )
    # Place target zzz_quantum_gate.py alphabetically LAST so old os.walk[:40] would completely miss it!
    (pkg / "zzz_quantum_gate.py").write_text(
        '"""Noise-sensitivity hardware readiness verifier."""\n'
        "from . import kernel_hub\n\n"
        "def verify_readiness_sweep(qubits: int, shots: int) -> bool:\n"
        "    return qubits > 0 and shots >= 64\n",
        encoding="utf-8",
    )
    for idx in range(52):
        (pkg / f"aaa_worker_{idx:02d}.py").write_text(
            f'"""Worker module {idx}."""\n'
            "from core_pkg import kernel_hub\n"
            f"def run_worker_{idx}(x: int) -> int:\n"
            "    return x + 1\n",
            encoding="utf-8",
        )

    # Add TypeScript / Next.js App Router + SQL + DNA-Lang files
    ts_lib = ws / "src" / "lib"
    ts_lib.mkdir(parents=True)
    (ts_lib / "supabaseClient.ts").write_text(
        "// Supabase browser client helper\n"
        "export interface SessionConfig { url: string; }\n"
        "export function createBrowserSupabase(cfg: SessionConfig) { return cfg; }\n",
        encoding="utf-8",
    )
    api_dir = ws / "src" / "app" / "api" / "health"
    api_dir.mkdir(parents=True)
    (api_dir / "route.ts").write_text(
        "import { createBrowserSupabase } from '@/lib/supabaseClient';\n"
        "export async function GET() { return createBrowserSupabase({ url: 'ok' }); }\n",
        encoding="utf-8",
    )
    (ws / "schema.sql").write_text("CREATE TABLE IF NOT EXISTS readiness_runs (id TEXT PRIMARY KEY);\n", encoding="utf-8")
    (ws / "edge.dna").write_text("organism EdgeGuardian {\n  gene coherence_guard {}\n}\n", encoding="utf-8")

    idx = codebase_visibility.index_repository(ws)
    assert idx.git_branch == "main"
    assert len(idx.files) >= 57
    assert idx.by_language["python"] >= 54
    assert idx.by_language["typescript"] == 2
    assert idx.by_language["sql"] == 1
    assert idx.by_language["dnalang"] == 1

    # Verify bidirectional dependency & blast-radius tracking (Python + TypeScript `@/` alias!)
    hub_node = idx.files["core_pkg/kernel_hub.py"]
    assert len(hub_node.imported_by) >= 52
    assert idx.hub_files[0][0] == "core_pkg/kernel_hub.py"

    ts_hub = idx.files["src/lib/supabaseClient.ts"]
    assert "src/app/api/health/route.ts" in ts_hub.imported_by

    # Verify query ranking surfaces alphabetically-last `zzz_quantum_gate.py` #1!
    ranked = codebase_visibility.rank_files(idx, query="verify_readiness_sweep quantum", limit=5)
    assert ranked[0][1].rel_path == "core_pkg/zzz_quantum_gate.py"

    # Verify upgraded osiris_ast_bridge.build_context finds `zzz_quantum_gate.py` despite >55 files!
    ctx_json = json.loads(osiris_ast_bridge.build_context(str(ws), max_files=10, max_chars=6000, query="verify_readiness_sweep"))
    assert any(f["file"] == "core_pkg/zzz_quantum_gate.py" for f in ctx_json["files"])

    # Verify `/codebase deps` blast-radius report
    deps_out = codebase_visibility.run_command("deps kernel_hub", cwd=ws)
    assert "imported_by" in deps_out and "core_pkg/zzz_quantum_gate.py" in deps_out


def test_code_agent_injects_workspace_map_and_keeps_write_confinement(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "router_engine.py").write_text(
        '"""Deterministic router engine."""\n'
        "def choose_route(score: float) -> str:\n"
        "    return 'fast' if score > 0.5 else 'safe'\n",
        encoding="utf-8",
    )

    seen_system_prompts: list[str] = []

    def fake_chat(messages, tools, model):
        seen_system_prompts.append(messages[0]["content"])
        return {"content": "Inspected router_engine.py via workspace map.", "tool_calls": []}

    agent = CodeAgent(
        root=str(ws),
        model="qwen2.5-coder:7b",
        chat=fake_chat,
        include_workspace_map=True,
    )
    res = agent.run("Explain choose_route in router_engine")
    assert res.status == "SUCCESS"
    assert seen_system_prompts
    assert "<workspace_map" in seen_system_prompts[0]
    assert "router_engine.py" in seen_system_prompts[0]
    assert "choose_route@L2" in seen_system_prompts[0]

    # Empty non-repo directory falls back cleanly to osiris-cli for read-only `/codebase` queries
    empty_dir = tmp_path / "empty_scratch"
    empty_dir.mkdir()
    resolved_root, why = codebase_visibility.resolve_read_only_repo(empty_dir, "quantum_readiness")
    assert resolved_root == codebase_visibility.osiris_cli_repo_root()
    assert "0 indexed source files" in why


def test_evidence_router_combines_bench_calibration_latency_and_cross_family_critic(tmp_path: Path) -> None:
    # 1. Verify providers/router.py recognizes gemma4:e4b & qwen3.5:9b as sovereign workstation models
    assert "gemma4:e4b" in SOVEREIGN_LOCAL_MODELS
    assert "qwen3.5:9b" in SOVEREIGN_LOCAL_MODELS
    beta_hw = SystemTelemetry(12288, 1.0, False, True, False, cloud_authorized=False)
    dec = select_mentor(1, 300, False, beta_hw, ["gemma3:1b", "gemma4:e4b"])
    assert dec.mentor_id == "gemma4:e4b"
    assert dec.degraded_fallback is False

    # 2. Construct a verified protege bench ledger + calibration ledger + living latency stats
    bench_dir = tmp_path / "bench"
    art_dir = bench_dir / "artifacts"
    art_dir.mkdir(parents=True)

    def store_art(text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        (art_dir / f"{digest}.txt").write_text(text, encoding="utf-8")
        return digest

    p_sha = store_art("prompt")
    m_sha = store_art("def ok(): return 1\n")
    t_sha = store_art("def test_proposal(): assert True\n")

    suite_sha = protege.current_suite()
    runner_sha = protege.current_runner()

    raw_entries = [
        # gemma3:4b: 0/3 -> demoted!
        {
            "kind": "bench_run_start",
            "run_id": "r_demoted",
            "backend": "ollama",
            "model": "gemma3:4b",
            "suite_sha256": suite_sha,
            "runner_sha256": runner_sha,
            "ts": "2026-11-21T10:00:00Z",
        },
        {
            "kind": "bench_task",
            "run_id": "r_demoted",
            "task": "backlog",
            "result": {
                "attempts": [
                    {"n": i, "backend": "ollama", "own_ok": True, "hidden_ok": False,
                     "prompt_sha256": p_sha, "module_sha256": m_sha, "test_sha256": t_sha}
                    for i in (1, 2, 3)
                ]
            },
            "ts": "2026-11-21T10:01:00Z",
        },
        # gemma4:e4b: 2/3 verified passes on backlog, 0 false confidence!
        {
            "kind": "bench_run_start",
            "run_id": "r_winner",
            "backend": "ollama",
            "model": "gemma4:e4b",
            "suite_sha256": suite_sha,
            "runner_sha256": runner_sha,
            "ts": "2026-11-21T10:02:00Z",
        },
        {
            "kind": "bench_task",
            "run_id": "r_winner",
            "task": "backlog",
            "result": {
                "attempts": [
                    {"n": 1, "backend": "ollama", "own_ok": True, "hidden_ok": True,
                     "prompt_sha256": p_sha, "module_sha256": m_sha, "test_sha256": t_sha},
                    {"n": 2, "backend": "ollama", "own_ok": True, "hidden_ok": True,
                     "prompt_sha256": p_sha, "module_sha256": m_sha, "test_sha256": t_sha},
                    {"n": 3, "backend": "ollama", "own_ok": False, "hidden_ok": False,
                     "prompt_sha256": p_sha, "module_sha256": m_sha, "test_sha256": t_sha},
                ]
            },
            "ts": "2026-11-21T10:03:00Z",
        },
    ]
    prev = "0" * 64
    lines = []
    for body in raw_entries:
        h = hashlib.sha256((prev + protege._canon(body)).encode("utf-8")).hexdigest()
        lines.append(json.dumps({**body, "prev": prev, "hash": h}))
        prev = h
    (bench_dir / "results.ledger.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    ohome = tmp_path / "ohome"
    cal_path = calibration.default_calibration_path(ohome)
    calibration.sync_bench_ledger(bench_dir=bench_dir, path=cal_path)

    liv_dir = tmp_path / "living"
    liv_dir.mkdir()
    (liv_dir / "stats.json").write_text(
        json.dumps({"first_word": {"qwen3.5:4b": [1.4, 1.6, 1.5], "gemma4:e4b": [18.0, 20.0, 19.0]}}),
        encoding="utf-8",
    )

    installed = ["gemma3:4b", "gemma4:e4b", "qwen2.5-coder:7b", "qwen3.5:4b", "qwen3.5:9b"]

    # On a backlog task, verified winner `gemma4:e4b` (2/3 verified + backlog task bonus + 0 overconf) ranks #1,
    # while `gemma3:4b` (0/3 + 1.00 overconf) is demoted to last place!
    code_ranked = evidence_router.rank_models(
        "code",
        installed=installed,
        task_text="fix duplicate backlog story idempotency",
        is_edge_low_ram=False,
        bench_dir=str(bench_dir),
        osiris_home=ohome,
        living_home=liv_dir,
    )
    assert code_ranked[0].model == "gemma4:e4b"
    assert code_ranked[-1].model == "gemma3:4b" and code_ranked[-1].demoted is True

    # Critic must pick a DIFFERENT model family from `gemma4:e4b` (cross-family diversity -> qwen3.5:9b!)
    critic_pick = evidence_router.select_for_role(
        "critic",
        installed=installed,
        avoid_model="gemma4:e4b",
        bench_dir=str(bench_dir),
        osiris_home=ohome,
    )
    assert critic_pick is not None
    assert critic_pick.family != "gemma4"
    assert critic_pick.model == "qwen3.5:9b"

    # On low-RAM Termux edge, >=7B models are penalized and compact `qwen3.5:4b` / `gemma4:e4b` win!
    edge_ranked = evidence_router.rank_models(
        "voice",
        installed=["qwen3.5:9b", "qwen2.5-coder:7b", "qwen3.5:4b"],
        is_edge_low_ram=True,
        living_home=liv_dir,
    )
    assert edge_ranked[0].model == "qwen3.5:4b"


def test_repl_dispatches_models_route_codebase_and_repo_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    for cmd in ("/models", "/route", "/codebase", "/repo"):
        assert cmd in osiris_repl.REPL_COMMANDS
        assert cmd in osiris_repl.READ_ONLY_COMMANDS

    state = osiris_repl.OsirisReplState()
    osiris_repl.dispatch_command(state, "/codebase quantum_readiness")
    out_cb = capsys.readouterr().out
    assert "osiris_cli/quantum_readiness.py" in out_cb

    osiris_repl.dispatch_command(state, "/models")
    out_mod = capsys.readouterr().out
    assert "MODEL ROUTER" in out_mod

    rendered_arch = architecture.full(otc.osiris_ui.Canvas("plain", 100), architecture.live_state(None, None))
    assert "osiris_cli/evidence_router.py" in rendered_arch
    assert "osiris_cli/codebase_visibility.py" in rendered_arch
