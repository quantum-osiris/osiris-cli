"""Offline verification suite for Iterations E0-E3:
- E0: Long-paste trailing directive intake (`living.asks_something` & `extract_paste_directive`),
  Ollama GBNF token ceiling (`SCHEMA_MAX_TOKENS=2048`) + transient HTTP 500 retry recovery,
  L5 Numerical-Grounding Critic defeating `claims` false-confidence, and `bench_trial.py --verifier l5`.
- E1: Declarative `dna::}{::lang` Edge Organism compiler (`compile_governed_edge_organism` rejecting
  refuted `51.843` / `1.7975` numerology) + `EdgeOrganismRuntime` 64-step regulatory & ternary rule controller.
- E2: Multi-substrate Edge Node registry (`alpha`, `beta`, `gamma`, `delta`), USB PnP state machine
  blocking `VID_045E&PID_062A` UEFI FFU Flash Mode while allowing `VID_03F0&PID_0155` MTP, and LF-clean
  `.ppkg` / Termux bundle staging + tamper/CRLF verification.
- E3: Non-interactive CLI (`osiris provision ...`) & REPL (`/provision ...`, `/edge ...`) primitives.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import shutil
import urllib.error
from unittest.mock import patch

import pytest

from dnalang.ledger import Ledger
import bench_trial
import osiris_bench
import osiris_termux_console as otc
import prompt_strategies
from osiris_cli import edge_provisioner, living, osiris_repl, production_synth
from osiris_cli.dnalang.edge_organism import (
    EdgeOrganismCompileError,
    EdgeOrganismRuntime,
    EdgeTelemetry,
    build_edge_organism_source,
    compile_governed_edge_organism,
)
from tests.test_living import ScriptedMentor, make

REPO_ROOT = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════════════════
# ITERATION E0: LONG-PASTE DIRECTIVE INTAKE, GBNF HTTP-500 RESILIENCE & L5 TRIAL
# ═══════════════════════════════════════════════════════════════════════════════


def test_e0_long_paste_with_trailing_grokking_and_provisioning_directive_routes_to_mentor(tmp_path):
    """Reproduces the 212KB paste ending with `\" conduct grokking, and recursively refine osiris...`."""
    trailing_directive = (
        '" conduct grokking, and recursively refine osiris, dna::}{::lang and the ecosystems '
        "adaptive and dynamic capabilities to be used for provisioning and administrating next gen "
        "edge node features functions and capabilities. deduce user intent."
    )
    large_paste = ("=== Provisioning Log Line ===\n" * 600) + trailing_directive + "\n"
    assert len(large_paste) > living.LONG_MESSAGE
    assert living.asks_something(large_paste) is True
    extracted = living.extract_paste_directive(large_paste)
    assert extracted is not None and "conduct grokking" in extracted and "deduce user intent" in extracted

    mentor = ScriptedMentor("I deduced your edge-node provisioning intent.")
    o, out_chunks = make(tmp_path, mentor=mentor)
    reply = o.converse(large_paste, input_transport="bracketed_paste")
    rendered = "".join(out_chunks)
    assert "You didn't say what to do with it" not in rendered
    assert reply.strip() == "I deduced your edge-node provisioning intent."
    sent_prompt = mentor.calls[-1][-1]["content"]
    assert "[Explicit operator directive attached to this paste -- address this directly]" in sent_prompt
    assert "conduct grokking" in sent_prompt


def test_e0_query_model_caps_schema_tokens_and_recovers_from_transient_http_500(monkeypatch):
    sent_payloads = []
    call_idx = [0]

    class _OkResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data.decode("utf-8"))
        sent_payloads.append(body)
        call_idx[0] += 1
        if call_idx[0] == 1:
            raise urllib.error.HTTPError(
                req.full_url, 500, "Internal Server Error", hdrs=None, fp=io.BytesIO(b'{"error":"llama runner crashed"}')
            )
        return _OkResp(json.dumps({"model": "qwen2.5-coder:7b", "response": '{"ok": true}', "done": True}).encode("utf-8"))

    monkeypatch.setattr(otc, "_detect_installed_ollama_models", lambda timeout=0.5: ["qwen2.5-coder:7b"])
    monkeypatch.delenv("OSIRIS_MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("OSIRIS_GEMMA_MOCK", raising=False)

    with patch("urllib.request.urlopen", fake_urlopen):
        res = otc.query_model(
            "generate json proposal",
            model="qwen2.5-coder:7b",
            stream=False,
            fmt={"type": "object"},
        )
    assert res == '{"ok": true}'
    assert len(sent_payloads) == 2
    # Attempt 1 under GBNF schema uses SCHEMA_MAX_TOKENS (2048) + num_ctx=8192 (never 4096 runaway)
    assert sent_payloads[0]["options"]["num_predict"] == otc.SCHEMA_MAX_TOKENS == 2048
    assert sent_payloads[0]["options"]["num_ctx"] == 8192
    # Attempt 2 after HTTP 500 automatically retries with compact context ceiling
    assert sent_payloads[1]["options"]["num_predict"] == 1536
    assert sent_payloads[1]["options"]["num_ctx"] == 4096


def test_e0_l5_critic_catches_claims_unsourced_number_false_confidence_and_repairs(tmp_path):
    """Reproduces the exact `claims` false-confidence candidate from `bench-20261118T091638Z-8130b6`."""
    claims_existing = (REPO_ROOT / "bench_tasks" / "claims" / "existing.py").read_text(encoding="utf-8")
    claims_acceptance = (REPO_ROOT / "bench_tasks" / "claims" / "acceptance.py").read_text(encoding="utf-8")
    claims_spec = json.loads((REPO_ROOT / "bench_tasks" / "claims" / "task.json").read_text(encoding="utf-8"))["spec"]

    # Exact false-confidence candidate from bench-20261118T091638Z-8130b6 (ignores numbers completely!)
    attempt_1_false_conf = (
        "=== FILE: claims.py ===\n"
        "def check_claim(text, evidence):\n"
        "    problems = []\n"
        "    if not evidence or not evidence.strip():\n"
        "        problems.append('no evidence cited')\n"
        "    if not text or not text.strip():\n"
        "        problems.append('no claim text')\n"
        "    return problems\n"
        "=== TEST: test_proposal.py ===\n"
        "import claims\n"
        "def test_proposal():\n"
        "    assert claims.check_claim('This is a claim', 'evidence.txt') == []\n"
        "    assert claims.check_claim('', 'evidence.txt') == ['no claim text']\n"
        "=== END ===\n"
    )
    attempt_2_grounded = (
        "=== FILE: claims.py ===\n"
        "import re\n"
        "_NUM_RE = re.compile(r'(?<!\\w)\\d+(?:\\.\\d+)?(?!\\w)')\n"
        "def check_claim(text, evidence):\n"
        "    problems = []\n"
        "    if not evidence or not evidence.strip():\n"
        "        problems.append('no evidence cited')\n"
        "        return problems\n"
        "    ev_nums = set(_NUM_RE.findall(evidence))\n"
        "    for num in _NUM_RE.findall(text or ''):\n"
        "        if num not in ev_nums:\n"
        "            problems.append(f'unsourced number: {num}')\n"
        "    return problems\n"
        "=== TEST: test_proposal.py ===\n"
        "import claims\n"
        "def test_proposal():\n"
        "    assert any('0.9728' in p for p in claims.check_claim('got 0.9728', 'got 0.9722'))\n"
        "    assert claims.check_claim('got 0.9722', 'got 0.9722.') == []\n"
        "=== END ===\n"
    )

    replies = iter([attempt_1_false_conf, attempt_2_grounded])
    outcome = production_synth.run_production_synthesis(
        root=str(tmp_path),
        target_rel_path="claims.py",
        story_text=claims_spec,
        propose_raw_fn=lambda _p: next(replies),
        sandbox_test_fn=otc._run_sandboxed_test_inner,
        existing_source=claims_existing,
        k=3,
    )
    assert outcome.attempts[0]["ok"] is False
    assert "0.9728" in outcome.attempts[0]["reason"]
    assert outcome.winner is not None and outcome.attempts[1]["ok"] is True

    hidden_ok, hidden_out = otc._run_sandboxed_test_inner("claims.py", outcome.winner["module"], claims_acceptance)
    assert hidden_ok is True, hidden_out


def test_e0_bench_trial_l5_verifier_mode_scores_against_hidden_acceptance_and_preserves_runner_hash(
    tmp_path, monkeypatch
):
    # L5 verifier mode lives in bench_trial.py and must not change the runner. The expected value is
    # the audited pin from tests/test_sprint0_routing_and_code_agent.py (af49b2bd, owner-reviewed in
    # 078d43e8); this WIP was written against the earlier e48927f3 pin.
    assert osiris_bench.runner_sha256() == "af49b2bd7707c36ff94cc0f87d307846294afefb5a35cb808f8524d41e53264c"
    assert "contract-critic-v2" in prompt_strategies.STRATEGIES

    osiris_mod = osiris_bench._load_osiris()
    monkeypatch.setattr(osiris_mod, "_sandbox_probe", lambda timeout=20: None)

    bad_first = (
        "=== FILE: claims.py ===\n"
        "def check_claim(text, evidence):\n"
        "    return [] if evidence.strip() else ['no evidence cited']\n"
        "=== TEST: test_proposal.py ===\n"
        "import claims\n"
        "def test_proposal():\n"
        "    assert claims.check_claim('x', 'y') == []\n"
        "=== END ===\n"
    )
    good_second = (
        "=== FILE: claims.py ===\n"
        "import re\n"
        "_N = re.compile(r'(?<!\\w)\\d+(?:\\.\\d+)?(?!\\w)')\n"
        "def check_claim(text, evidence):\n"
        "    if not evidence or not evidence.strip():\n"
        "        return ['no evidence cited']\n"
        "    ev = set(_N.findall(evidence))\n"
        "    return [f'unsourced number: {n}' for n in _N.findall(text or '') if n not in ev]\n"
        "=== TEST: test_proposal.py ===\n"
        "import claims\n"
        "def test_proposal():\n"
        "    assert claims.check_claim('val 9', 'val 9.') == []\n"
        "=== END ===\n"
    )
    seq = iter([bad_first, good_second])

    def fake_synth(prompt, system_prompt="", structured=False, only=None, ollama_model=None):
        osiris_mod._last_backend.update(name="ollama", model="qwen2.5-coder:7b", error=None)
        return next(seq)

    monkeypatch.setattr(osiris_mod, "_synthesize", fake_synth)
    # bench_trial.main() now enforces Ollama preflight before ledger writes; mock it for offline CI.
    monkeypatch.setattr(osiris_bench, "ollama_preflight", lambda model, fetch=None: None)
    rc = bench_trial.main([
        "--strategy", "contract-critic-v2",
        "--verifier", "l5",
        "--backend", "ollama",
        "--model", "qwen2.5-coder:7b",
        "--tasks", "claims",
        "--k", "2",
        "--out", str(tmp_path),
    ])
    assert rc == 0
    rows = [json.loads(ln) for ln in (tmp_path / "results.ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    start_row = next(r for r in rows if r["kind"] == "bench_run_start")
    task_row = next(r for r in rows if r["kind"] == "bench_task")
    assert start_row["verifier"] == "l5"
    assert task_row["result"]["delivered"] is True
    assert task_row["result"]["false_confidence"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# ITERATIONS E1-E3: DNA-LANG EDGE ORGANISM, USB GUARD & PROVISIONING PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════════


def test_e1_edge_organism_compiler_rejects_refuted_51843_numerology_and_cycles():
    # Exact refuted organism shape from the pasted transcript
    refuted_dna = (
        "organism elite_x3_substrate {\n"
        "  meta {\n"
        "    theta_lock_deg: 51.843,\n"
        "    su2_trace: 1.7975\n"
        "  }\n"
        "  gene g1 {\n"
        "    id: \"G1\",\n"
        "    trigger: \"on_genesis\",\n"
        "    action: \"(emit READY)\"\n"
        "  }\n"
        "}\n"
    )
    with pytest.raises(EdgeOrganismCompileError, match="Claims Firewall"):
        compile_governed_edge_organism(refuted_dna)

    # Undeclared metric warning fails closed under strict edge compiler
    undeclared_metric_dna = (
        "organism bad_metric {\n"
        "  meta { metrics: [\"temp_c\"] }\n"
        "  gene g1 {\n"
        "    id: \"G1\",\n"
        "    trigger: \"when unknown_sensor > 5\",\n"
        "    action: \"(emit WARN)\"\n"
        "  }\n"
        "}\n"
    )
    with pytest.raises(EdgeOrganismCompileError, match="undeclared metric"):
        compile_governed_edge_organism(undeclared_metric_dna)


def test_e1_edge_organism_runtime_evaluates_ternary_rules_and_topological_regulators_with_ledger(tmp_path):
    ledger_file = tmp_path / "edge_ticks.jsonl"
    src = build_edge_organism_source("gamma_win10_mobile_ppkg", default_horizon_s=300)
    compiled = compile_governed_edge_organism(src)
    rt = EdgeOrganismRuntime(compiled, step_budget=64, ledger_path=str(ledger_file))

    # Tick 1: Genesis + nominal telemetry
    s1 = rt.step(EdgeTelemetry(temp_c=33.0, battery_pct=90.0, link_rtt_ms=20.0))
    assert s1.active_route == "primary_mentor"
    assert s1.horizon_s == 300.0
    assert "GENESIS_ENROLLED" in s1.emitted_events and "STATE_NOMINAL" in s1.emitted_events
    assert s1.severed_capabilities == set()

    # Tick 2: Thermal + battery + high-latency offline degradation
    s2 = rt.step(EdgeTelemetry(temp_c=46.0, battery_pct=14.0, link_rtt_ms=750.0, network_up=False))
    assert s2.horizon_s == 90.0  # 300 - 120 (thermal) - 90 (battery)
    assert s2.active_route == "local_edge_cache"
    assert {"cloud_burst", "unattended_train"} <= s2.severed_capabilities

    # Tick 3: UEFI FFU Flash Mode ("Lightning Bolt + Gear") blocks MTP/AppX push and notifies operator
    s3 = rt.step(EdgeTelemetry(usb_ffu_mode=1, transport_ready=False))
    assert {"mtp_push", "appx_sideload"} <= s3.severed_capabilities
    assert "HALT_UEFI_FFU_MODE" in s3.emitted_events
    assert ("operator", "EXIT_FFU_HOLD_POWER_VOLDOWN_10S") in s3.operator_notices

    # Tick 4: Broken ledger hash chain forces immediate read-only lockdown
    s4 = rt.step(EdgeTelemetry(ledger_intact=0))
    assert s4.active_route == "lockdown_readonly"
    assert {"apply_gate", "mtp_push", "distill_export"} <= s4.severed_capabilities
    assert "QUARANTINE_LEDGER_TAMPER" in s4.emitted_events

    assert Ledger(str(ledger_file)).verify() is None


def test_e2_usb_pnp_classifier_blocks_uefi_ffu_flash_mode_and_allows_mtp_and_adb(tmp_path):
    ffu_pnp = [{
        "FriendlyName": "QUALCOMM MSM DEVICE",
        "InstanceId": r"USB\VID_045E&PID_062A\45000000-9A62-9D1D-0000-000000000000",
        "Status": "OK",
        "Class": "USBDevice",
    }]
    mtp_and_adb_pnp = [
        {
            "FriendlyName": "HP Elite x3",
            "InstanceId": r"USB\VID_03F0&PID_0155\5&EDED85C&0&7",
            "Status": "OK",
            "Class": "WPD",
        },
        {
            "FriendlyName": "SAMSUNG Android ADB Interface",
            "InstanceId": r"USB\VID_04E8&PID_6860\R5CX12345",
            "Status": "OK",
            "Class": "USB",
        },
    ]

    with pytest.raises(RuntimeError, match="EDGE_USB_FFU_BLOCKED"):
        edge_provisioner.stage_provisioning_bundle(
            tmp_path / "blocked_bundle",
            profile="gamma",
            usb_devices=ffu_pnp,
        )

    classified = edge_provisioner.classify_usb_pnp_devices(mtp_and_adb_pnp)
    assert [c.mode for c in classified] == ["MTP_ONLINE", "ADB_ONLINE"]
    assert all(c.push_allowed for c in classified)


def test_e2_stage_and_verify_gamma_ppkg_and_alpha_termux_bundles_detect_tampering_and_crlf(tmp_path):
    key_dir = tmp_path / "keys"
    gamma_dir = tmp_path / "gamma_bundle"
    has_wimlib = shutil.which("wimlib-imagex") is not None

    manifest_gamma = edge_provisioner.stage_provisioning_bundle(
        gamma_dir,
        profile="gamma",
        account_email="operator@example.com",
        cert_thumbprint="7B5C5F3366AA6E3AF527BD118C4FCD245A392A3C",
        cert_der_b64="MIIBFAKECERTBASE64==",
        key_dir=key_dir,
        capture_wim=has_wimlib,
    )
    assert manifest_gamma["node_tier"] == "NODE_GAMMA"
    assert manifest_gamma["coordinator_pubkey_sha256"] is not None
    if has_wimlib:
        assert (gamma_dir / "OSIRIS.Edge.Provisioning.ppkg").is_file()

    ok, errs, _ = edge_provisioner.verify_provisioning_bundle(gamma_dir)
    assert ok is True, errs

    # Alpha (Android / Termux) bundle
    alpha_dir = tmp_path / "alpha_bundle"
    edge_provisioner.stage_provisioning_bundle(alpha_dir, profile="alpha", key_dir=key_dir)
    ok_alpha, errs_alpha, _ = edge_provisioner.verify_provisioning_bundle(alpha_dir)
    assert ok_alpha is True, errs_alpha

    # Inject CRLF + tamper into staged script -> verify_provisioning_bundle must fail closed
    boot_sh = alpha_dir / "payload" / "bootstrap_edge_node.sh"
    boot_sh.write_bytes(boot_sh.read_bytes().replace(b"\n", b"\r\n"))
    ok_tampered, errs_tampered, _ = edge_provisioner.verify_provisioning_bundle(alpha_dir)
    assert ok_tampered is False
    assert any("SHA-256 mismatch" in e for e in errs_tampered)
    assert any("CRLF" in e for e in errs_tampered)


def test_e3_cli_and_repl_provision_primitives_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        edge_provisioner,
        "discover_edge_topology",
        lambda **_kw: {
            "host_arch": "x86_64",
            "is_wsl2": True,
            "prime_cores": [6, 7],
            "ssh_8022_listening": False,
            "edge_bridge_8080_listening": False,
            "ollama_11434_listening": True,
            "adb_devices": [],
            "usb_endpoints": [{
                "friendly_name": "HP Elite x3",
                "instance_id": r"USB\VID_03F0&PID_0155\1",
                "mode": "MTP_ONLINE",
                "push_allowed": True,
                "advisory": "MTP ready",
            }],
            "uefi_ffu_flash_block": False,
            "wimlib_available": True,
        },
    )

    bundle_out = tmp_path / "cli_bundle"
    with pytest.raises(SystemExit) as ex_stage:
        osiris_repl.main(["provision", "stage", "--profile", "gamma", "--out", str(bundle_out)])
    assert ex_stage.value.code == 0

    with pytest.raises(SystemExit) as ex_verify:
        osiris_repl.main(["provision", "verify", str(bundle_out)])
    assert ex_verify.value.code == 0

    state = osiris_repl.OsirisReplState()
    osiris_repl.dispatch_command(state, "/provision simulate --profile gamma --usb-ffu")
    osiris_repl.dispatch_command(state, "/edge status --json")
    out = capsys.readouterr().out
    assert "[OSIRIS::PROVISION STAGED]" in out
    assert "[OSIRIS::PROVISION VERIFIED] PASS" in out
    assert "HALT_UEFI_FFU_MODE" in out
    assert '"mode": "MTP_ONLINE"' in out
