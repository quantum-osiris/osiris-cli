"""Stage 5 offline tests: telemetry-governed MentorRouter, 100-exchange continual-learning
isolation (20/100 held-out, 0 degraded/audit-failed turns entering training), fail-closed
manual overrides, async shadow scoring, and backward-compatible JSONL ledger integrity."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from osiris_cli import train
from osiris_cli.living import Osiris
from osiris_cli.probes import verify_chain, verify_chain_tail
from osiris_cli.providers.router import (
    CLOUD_BURST_MODEL,
    MentorRouter,
    SystemTelemetry,
    match_installed_model,
    normalize_ollama_tag,
    select_mentor,
)


class _RoutedMentor:
    base = "routed://"

    def __init__(self, reply_fn=None) -> None:
        self._model = None
        self.used: list[str] = []
        self._reply_fn = reply_fn or (lambda model, idx: f"Grounded reply from {model}.")
        self.last_done = {"done_reason": "stop"}

    def model(self) -> str:
        return self._model or "qwen2.5:7b"

    def stream(self, messages, model=None):
        chosen = model or self.model()
        self.used.append(chosen)
        yield self._reply_fn(chosen, len(self.used) - 1)


class _FakeCore:
    name = "fake.nclm"

    def __init__(self, bpb: float = 4.25) -> None:
        self._bpb = bpb
        self._step = 0
        self.learned: list[str] = []

    def step(self) -> int:
        return self._step

    def bits_per_byte(self, text: str) -> float:
        return self._bpb

    def learn(self, lesson: str, steps: int) -> float:
        self.learned.append(lesson)
        self._step += steps
        return 2.5


def test_tag_normalization_and_no_silent_substitution_for_manual_override() -> None:
    assert normalize_ollama_tag("gemma-3-1b") == "gemma3:1b"
    assert match_installed_model("gemma-3-1b", ["gemma3:1b"]) == "gemma3:1b"
    assert match_installed_model("gemma3", ["gemma3:1b"]) is None

    beta_hw = SystemTelemetry(
        free_ram_mb=10240,
        cpu_load_1m=1.0,
        is_termux_edge=False,
        ollama_reachable=True,
        cloud_quota_ok=False,
        cloud_authorized=False,
    )
    # Missing manual override must fail closed, never silently substitute gemma3:1b
    bad = select_mentor(1, 200, False, beta_hw, ["gemma3:1b"], manual_override="qwen2.5-coder:7b")
    assert bad.mentor_id is None
    assert bad.available is False
    assert bad.reason == "manual_override_not_installed"
    assert bad.allow_distillation is False
    assert "not installed" in (bad.unavailable_reason or "")


def test_dual_substrate_routing_edge_workstation_and_authorized_cloud_burst() -> None:
    alpha_hw = SystemTelemetry(
        free_ram_mb=2048,
        cpu_load_1m=0.8,
        is_termux_edge=True,
        ollama_reachable=True,
        cloud_quota_ok=True,
        cloud_authorized=False,
        node_id="NODE_ALPHA_EDGE",
    )
    # Cloud NOT authorized -> stays on local edge model with distillation blocked
    d_edge = select_mentor(1, 300, True, alpha_hw, ["gemma3:1b"])
    assert d_edge.mentor_id == "gemma3:1b"
    assert d_edge.reason == "edge_low_ram"
    assert d_edge.allow_distillation is False
    assert d_edge.degraded_fallback is True

    # Cloud explicitly authorized + multi-hop -> bursts cleanly
    alpha_burst = SystemTelemetry(
        free_ram_mb=2048,
        cpu_load_1m=0.8,
        is_termux_edge=True,
        ollama_reachable=True,
        cloud_quota_ok=True,
        cloud_authorized=True,
        node_id="NODE_ALPHA_EDGE",
    )
    d_burst = select_mentor(1, 300, True, alpha_burst, ["gemma3:1b"])
    assert d_burst.mentor_id == CLOUD_BURST_MODEL
    assert d_burst.reason == "edge_complex_burst"
    assert d_burst.allow_distillation is True

    # Node Beta sovereign default vs load-shed burst vs degraded local fallback
    beta_ok = SystemTelemetry(12288, 1.2, False, True, True, cloud_authorized=True)
    d_beta = select_mentor(2, 400, False, beta_ok, ["gemma3:1b", "qwen2.5:7b"])
    assert d_beta.mentor_id == "qwen2.5:7b"
    assert d_beta.reason == "sovereign_default"
    assert d_beta.allow_distillation is True

    beta_hot = SystemTelemetry(12288, 6.5, False, True, True, cloud_authorized=True)
    d_shed = select_mentor(3, 1800, True, beta_hot, ["qwen2.5:7b"])
    assert d_shed.mentor_id == CLOUD_BURST_MODEL
    assert d_shed.reason == "load_shed_burst"

    d_degraded = select_mentor(4, 200, False, beta_ok, ["gemma3:1b"])
    assert d_degraded.mentor_id == "gemma3:1b"
    assert d_degraded.reason == "degraded_fallback"
    assert d_degraded.allow_distillation is False
    assert d_degraded.degraded_fallback is True


def test_100_exchange_continual_learning_isolation_and_ledger_integrity(tmp_path: Path) -> None:
    """Simulate 100 exchanges across Beta sovereign 7B, Alpha low-RAM degraded fallbacks,
    and grounding-audit failures. Verify exact 20/100 held-out isolation, zero degraded or
    hallucinated replies entering online or batch training, and intact hash chains."""
    beta_hw = SystemTelemetry(12288, 1.0, False, True, False, cloud_authorized=False)
    alpha_hw = SystemTelemetry(2048, 0.9, True, True, False, cloud_authorized=False)

    turn_counter = {"n": 0}

    def telemetry_fn():
        i = turn_counter["n"]
        turn_counter["n"] += 1
        # First 50 turns on Node Beta with 7B; next 50 on Node Alpha low-RAM edge
        if i < 50:
            return beta_hw, ["qwen2.5:7b", "gemma3:1b"]
        return alpha_hw, ["gemma3:1b"]

    def reply_fn(model: str, idx: int) -> str:
        # Inject phantom citation on every 7th non-held-out Beta turn to test audit gating
        if idx < 50 and idx % 5 != 0 and idx % 7 == 0:
            return "Fabricated claim citing `docs/PHANTOM_ROUTER.md` with 0.1111 bits/byte."
        return f"Clean grounded answer from {model} on turn {idx}."

    router = MentorRouter(telemetry_provider=telemetry_fn)
    core = _FakeCore()
    mentor = _RoutedMentor(reply_fn=reply_fn)
    o = Osiris(
        core=core,
        mentor=mentor,
        home=str(tmp_path / "living"),
        out=lambda s: None,
        background=False,
        tty=False,
        router=router,
    )

    for i in range(100):
        o.converse(f"technical query number {i}")

    rows = [json.loads(ln) for ln in Path(o.log_path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) == 100

    held_rows = [r for r in rows if r["heldout"]]
    assert len(held_rows) == 20
    assert [r["index"] for r in held_rows] == [i for i in range(100) if i % 5 == 0]

    # Zero degraded (Alpha) rows or audit-failed rows may be distill_eligible or trained
    alpha_rows = rows[50:]
    assert all(r["routing"]["degraded_fallback"] is True for r in alpha_rows)
    assert all(r["distill_eligible"] is False for r in alpha_rows)

    audit_failed = [r for r in rows if not r["grounding_audit"]["passed"]]
    assert len(audit_failed) > 0
    assert all(r["distill_eligible"] is False for r in audit_failed)

    eligible_rows = [r for r in rows if r["distill_eligible"]]
    assert len(core.learned) == len(eligible_rows)
    assert all("PHANTOM_ROUTER.md" not in lesson for lesson in core.learned)
    assert all("gemma3:1b" not in lesson for lesson in core.learned)

    # train.chat_lessons must match exact train/heldout isolation
    lessons = train.chat_lessons(o.log_path)
    assert len(lessons["train"]) == len(eligible_rows)
    assert len(lessons["heldout"]) == 20
    assert all("PHANTOM_ROUTER.md" not in l for l in lessons["train"])
    assert all("gemma3:1b" not in l for l in lessons["train"])

    # Full chain and O(1) tail verification must pass
    assert verify_chain(o.log_path)["intact"] is True
    assert verify_chain_tail(o.log_path, tail_window=10).intact is True


def test_router_failure_tracking_and_async_shadow_bpb() -> None:
    beta_hw = SystemTelemetry(12288, 1.0, False, True, False, cloud_authorized=False)
    router = MentorRouter(telemetry_provider=lambda: (beta_hw, ["qwen2.5:7b", "gemma3:1b"]))

    d1 = router.route(exchange_idx=1, prompt_tokens=100)
    assert d1.mentor_id == "qwen2.5:7b"

    router.record_failure("qwen2.5:7b", "timeout")
    d2 = router.route(exchange_idx=2, prompt_tokens=100)
    assert d2.mentor_id == "gemma3:1b"
    assert d2.degraded_fallback is True
    assert d2.allow_distillation is False
    assert "qwen2.5:7b" in d2.reason

    core = _FakeCore(bpb=3.75)
    c_bpb, u_bpb = asyncio.run(router.score_shadow_bpb_async(core, "sample reply", lambda t: 4.9))
    assert c_bpb == 3.75
    assert u_bpb == 4.9
