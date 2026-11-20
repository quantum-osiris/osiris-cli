"""Sprint A1.2 / A1.3: one node-ID set, and a natural-language router that proposes a node."""
from __future__ import annotations

import pytest

from osiris_cli import gap_analyzer as ga
from osiris_cli.node_ids import ALIASES, CANONICAL, NODE_ALPHA, NODE_BETA, canonical_node
from osiris_cli.node_router import CAPABILITY_NODES, route_task

HARNESS_TASK = "On the mobile edge node, use the on-device model to explain in one sentence what a Bell state is."


# --- A1.3: no drift ------------------------------------------------------------------------
def test_substrate_reports_only_canonical_ids(monkeypatch):
    from osiris import substrate
    monkeypatch.setattr(substrate, "_profile", lambda: None)
    for arch, alpha in [("aarch64", True), ("x86_64", False), ("riscv64", False)]:
        monkeypatch.setattr(substrate, "_arch", lambda a=arch: a)
        monkeypatch.setattr(substrate, "_fallback_alpha", lambda a=alpha: a)
        assert substrate.node_type() in CANONICAL


def test_hardware_mesh_ids_are_canonical():
    mesh = pytest.importorskip("osiris_hardware_mesh")
    enum = next((getattr(mesh, n) for n in dir(mesh) if n.lower() in ("nodetype", "node_type")
                 and hasattr(getattr(mesh, n), "__members__")), None)
    if enum is None:
        pytest.skip("mesh exposes no node-type enum")
    assert {m.value for m in enum} <= set(CANONICAL)


def test_other_modules_use_known_ids():
    assert all(canonical_node(t) for t in ga.CapabilitySpec("x").supported_targets)
    assert set(ga.CapabilitySpec("x").supported_targets) <= set(CANONICAL)
    assert canonical_node("NODE_ALPHA") == NODE_ALPHA and canonical_node("NODE_BETA") == NODE_BETA  # provisioner tiers
    assert all(v in CANONICAL for v in ALIASES.values())
    assert all(n in CANONICAL for nodes in CAPABILITY_NODES.values() for n in nodes)


def test_repl_labels_cover_canonical_ids():
    import osiris_cli.osiris_repl as repl
    labels = next(v for v in vars(repl).values()
                  if isinstance(v, dict) and "NODE_ALPHA_MOBILE" in v and "NODE_BETA_COMPUTE" in v)
    assert set(labels) <= set(CANONICAL)


def _analyze(target, available, cap="edge_llm_inference", supported=None):
    spec = ga.CapabilitySpec(cap, supported_targets=tuple(supported or CAPABILITY_NODES.get(cap, (NODE_BETA,))))
    step = ga.ProposedStep("s1", cap, target, parameters={"prompt": "x"}, workspace_snapshot_id="snap")
    fp = ga.compute_action_fingerprint(step)
    return ga.CapabilityGapAnalyzer.analyze(ga.AnalysisInput(
        step=step, current_epoch=1.0, active_snapshot_id="snap", registered_capabilities={cap: spec},
        available_targets=frozenset(available), available_verifiers=frozenset(), context_facts=(),
        policy=ga.PolicyAssessment(ga.PolicyState.ALLOWED, bound_action_fingerprint=fp)))


@pytest.mark.parametrize("target,available", [
    ("NODE_ALPHA_EDGE", {NODE_ALPHA}), (NODE_ALPHA, {"NODE_ALPHA_EDGE"}), (NODE_ALPHA, {NODE_ALPHA}),
])
def test_analyzer_treats_aliases_as_the_same_node(target, available):
    assert _analyze(target, available).is_ready


@pytest.mark.parametrize("target,available", [
    (NODE_ALPHA, {NODE_BETA}), ("NODE_GAMMA", {NODE_ALPHA, "NODE_GAMMA"}), (NODE_BETA, {NODE_BETA, NODE_ALPHA}),
])
def test_analyzer_blocks_offline_unknown_or_unsupported_nodes(target, available):
    rep = _analyze(target, available)
    assert not rep.is_ready
    assert any(g.kind == ga.GapKind.EXECUTION_UNAVAILABLE for g in rep.gaps)


def test_alias_does_not_change_the_recorded_fingerprint_formula():
    a = ga.ProposedStep("s", "c", "NODE_ALPHA_EDGE", workspace_snapshot_id="w")
    b = ga.ProposedStep("s", "c", NODE_ALPHA, workspace_snapshot_id="w")
    assert ga.compute_action_fingerprint(a) != ga.compute_action_fingerprint(b)  # raw value is bound, as before


# --- A1.2: router --------------------------------------------------------------------------
def test_harness_task_routes_to_alpha_without_a_harness_rule():
    d = route_task(HARNESS_TASK)
    assert d.routed and d.target_node == NODE_ALPHA and d.capability_id == "edge_llm_inference"
    assert d.rule_id == "N3-edge-inference"
    rep = _analyze(d.target_node, {NODE_ALPHA, NODE_BETA}, cap=d.capability_id)
    assert rep.is_ready
    assert not _analyze(d.target_node, {NODE_BETA}, cap=d.capability_id).is_ready  # phone offline -> blocked


@pytest.mark.parametrize("task,node,cap", [
    ("run the test suite for osiris_cli", NODE_BETA, "workstation_compute"),
    ("simulate a 5-qubit GHZ circuit", NODE_BETA, "workstation_compute"),
    ("fix the edge cases in parser.py", NODE_BETA, "workstation_task"),
    ("rename the variable in calc.py", NODE_BETA, "workstation_task"),
    ("Ask the phone to summarise this paragraph", NODE_ALPHA, "edge_llm_inference"),
    ("translate 'hello' with the edge model", NODE_ALPHA, "edge_llm_inference"),
])
def test_routes(task, node, cap):
    d = route_task(task)
    assert d.routed and d.target_node == node and d.capability_id == cap, d


@pytest.mark.parametrize("task,rule", [
    ("", "N0-empty"),
    ("explain Bell states on the phone and on the workstation", "N1-both-nodes-named"),
    ("run pytest on the phone", "N2-compute-not-on-alpha"),
    ("submit the qiskit job from the mobile device", "N2-compute-not-on-alpha"),
    ("open the settings on the phone", "N3-alpha-without-inference"),
])
def test_refusals_are_explicit(task, rule):
    d = route_task(task)
    assert not d.routed and d.target_node is None and d.rule_id == rule
    assert d.reason


def test_decision_records_what_matched():
    m = route_task(HARNESS_TASK).as_dict()["matched"]
    assert "on-device" in m["alpha"] and m["inference"]
