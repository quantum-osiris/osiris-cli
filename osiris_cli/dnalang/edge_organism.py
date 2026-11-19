"""Declarative `dna::}{::lang` Edge Organism Compiler, Claims-Firewall Auditor, and Adaptive Controller.

Connects `dnalang` v0.4.0's `RuleSet`, `RegulatoryGraph`, `Trigger`, closed 64-step `Interpreter`,
and hash-chained `Ledger` to real Edge Node provisioning and dynamic runtime administration:
  * Rejects refuted CRSM numerology (`51.843`, `46.9787`, `0.7734`, `1.7975`, `2.176435e-8`) at compile time.
  * Evaluates ternary capability bitvectors (`[network_up, battery_ok, thermal_ok, transport_ready, ledger_ok]`)
    and metric triggers (`temp_c`, `battery_pct`, `link_rtt_ms`, `usb_ffu_mode`, `ledger_intact`, `last_error`)
    in topological dependency order under a hard 64-step action budget.
  * Emits deterministic administrative state transitions (`active_route`, `horizon_s`, `severed_capabilities`,
    `emitted_events`, `operator_notices`) and optional tamper-evident `dnalang.Ledger` receipts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Dict, List, Optional, Set, Tuple

import dnalang
from dnalang.action_dsl import Context, Interpreter, matches, parse_sexpr, validate
from dnalang.ledger import Ledger
from dnalang.regulation import Trigger
from dnalang.rules_ir import RegulatoryGraph, RuleSet

REFUTED_NUMERIC_TOLERANCES: Tuple[Tuple[float, float, str], ...] = (
    (51.843, 1e-3, "refuted theta_lock = 51.843 deg constant (SYS-024 / zenodo.23832283)"),
    (0.9048308, 1e-5, "refuted theta_lock = 51.843 deg in radians (0.9048308 rad)"),
    (46.9787, 1e-3, "refuted tau_0 = 46.9787 us revival constant"),
    (0.7734, 1e-4, "refuted Phi = 0.7734 CCCE metric"),
    (1.7975, 1e-4, "unverified su2_trace = 1.7975 numerology"),
    (2.176435e-8, 1e-13, "ungrounded Planck-mass constant in runtime policy"),
)

REFUTED_META_KEYS = frozenset({
    "theta_lock",
    "theta_lock_deg",
    "su2_trace",
    "tau_0_us",
    "ccce_phi",
    "consciousness_threshold",
})

EDGE_DECLARED_METRICS = (
    "temp_c",
    "battery_pct",
    "link_rtt_ms",
    "usb_ffu_mode",
    "ledger_intact",
    "last_error",
)


class EdgeOrganismCompileError(ValueError):
    """Raised when an edge `.dna` organism fails parsing, semantic checking, or the Claims Firewall."""


@dataclass(frozen=True)
class CompiledEdgeOrganism:
    name: str
    source: str
    source_sha256: str
    meta: Dict[str, Any]
    circuit_sha256: Optional[str]
    rules_sha256: Optional[str]
    regulation_sha256: Optional[str]
    circuit: Optional[Any]
    rules: Optional[RuleSet]
    regulation: Optional[RegulatoryGraph]

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "organism": self.name,
            "source_sha256": self.source_sha256,
            "meta": dict(self.meta),
            "circuit_sha256": self.circuit_sha256,
            "rules_sha256": self.rules_sha256,
            "regulation_sha256": self.regulation_sha256,
            "rule_count": len(self.rules.rules) if self.rules else 0,
            "regulator_count": len(self.regulation.regulators) if self.regulation else 0,
            "regulator_order": list(self.regulation.order) if self.regulation else [],
        }


@dataclass
class EdgeTelemetry:
    temp_c: float = 34.0
    battery_pct: float = 85.0
    link_rtt_ms: float = 25.0
    usb_ffu_mode: int = 0       # 1 when device enumerates in UEFI FFU Flash Mode (VID_045E&PID_062A)
    ledger_intact: int = 1      # 0 when hash-chain verification fails
    network_up: bool = True
    transport_ready: bool = True
    last_error: float = 0.0

    def to_metrics(self) -> Dict[str, float]:
        return {
            "temp_c": float(self.temp_c),
            "battery_pct": float(self.battery_pct),
            "link_rtt_ms": float(self.link_rtt_ms),
            "usb_ffu_mode": float(self.usb_ffu_mode),
            "ledger_intact": float(self.ledger_intact),
            "last_error": float(self.last_error),
        }

    def capability_register(self) -> str:
        """5-bit ternary-compatible register: [network_up, battery_ok, thermal_ok, transport_ready, ledger_ok]."""
        bits = [
            "1" if self.network_up and self.link_rtt_ms <= 500.0 else "0",
            "1" if self.battery_pct > 20.0 else "0",
            "1" if self.temp_c < 43.0 else "0",
            "1" if self.transport_ready and self.usb_ffu_mode == 0 else "0",
            "1" if self.ledger_intact == 1 else "0",
        ]
        return "".join(bits)


@dataclass
class EdgeAdminState:
    tick: int = 0
    active_route: str = "primary_mentor"
    horizon_s: float = 300.0
    severed_capabilities: Set[str] = field(default_factory=set)
    vars: Dict[str, Any] = field(default_factory=dict)
    signals_board: Set[str] = field(default_factory=set)
    expressed_last: Set[str] = field(default_factory=set)
    emitted_events: List[str] = field(default_factory=list)
    operator_notices: List[Tuple[str, str]] = field(default_factory=list)
    fired_rules: List[str] = field(default_factory=list)
    fired_regulators: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "active_route": self.active_route,
            "horizon_s": self.horizon_s,
            "severed_capabilities": sorted(self.severed_capabilities),
            "vars": dict(self.vars),
            "emitted_events": list(self.emitted_events),
            "operator_notices": [{"target": t, "message": m} for t, m in self.operator_notices],
            "fired_rules": list(self.fired_rules),
            "fired_regulators": list(self.fired_regulators),
        }


def _audit_value_for_refuted_claims(key: str, val: Any) -> List[str]:
    issues: List[str] = []
    k_low = key.lower()
    if k_low in REFUTED_META_KEYS:
        issues.append(f"forbidden refuted/unverified key '{key}' in organism specification")
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        fval = float(val)
        for target, tol, label in REFUTED_NUMERIC_TOLERANCES:
            if abs(fval - target) <= tol:
                issues.append(f"forbidden constant {val} ({label})")
    elif isinstance(val, str):
        for token, reason in (
            ("51.843", "refuted theta_lock 51.843"),
            ("46.9787", "refuted tau_0 46.9787"),
            ("1.7975", "unverified su2_trace 1.7975"),
            ("0.7734", "refuted Phi 0.7734"),
        ):
            if token in val:
                issues.append(f"forbidden literal '{token}' in '{key}' ({reason})")
    elif isinstance(val, list):
        for item in val:
            issues.extend(_audit_value_for_refuted_claims(key, item))
    return issues


def audit_ast_claims_firewall(org: Any) -> List[str]:
    """Inspect parsed `dnalang.ast.Organism` for refuted CRSM constants before lowering."""
    violations: List[str] = []
    for k, v in (org.meta or {}).items():
        violations.extend(_audit_value_for_refuted_claims(k, v))
    for sec_dict in (org.sections or {}).values():
        for k, v in sec_dict.items():
            violations.extend(_audit_value_for_refuted_claims(k, v))
    for kvg in (org.kv_genes or []):
        for k, v in (kvg.fields or {}).items():
            violations.extend(_audit_value_for_refuted_claims(f"{kvg.name}.{k}", v))
    for inst in (org.genome or []):
        for idx, expr in enumerate(inst.args or ()):
            try:
                val = dnalang.sema.eval_expr(expr, {})
                violations.extend(_audit_value_for_refuted_claims(f"{inst.gene}[{idx}]", val))
            except Exception:
                pass
    return violations


def compile_governed_edge_organism(dna_source: str) -> CompiledEdgeOrganism:
    """Parse, Claims-Firewall-audit, type-check, and lower an edge `.dna` organism."""
    clean_src = dna_source.replace("\r\n", "\n").replace("\r", "\n")
    try:
        org = dnalang.parse(clean_src)
    except Exception as exc:
        raise EdgeOrganismCompileError(f"dnalang parse error: {exc}") from exc

    claim_violations = audit_ast_claims_firewall(org)
    if claim_violations:
        raise EdgeOrganismCompileError(
            "Claims Firewall (SYS-024) rejected .dna organism: " + "; ".join(claim_violations)
        )

    diags = dnalang.check(org)
    if not diags.ok():
        raise EdgeOrganismCompileError("dnalang semantic error(s): " + "; ".join(diags.errors))
    if diags.warnings:
        raise EdgeOrganismCompileError("dnalang strict warning(s): " + "; ".join(diags.warnings))

    lowered = dnalang.lower_all(org)
    circ = lowered.get("circuit")
    rules = lowered.get("rules")
    reg = lowered.get("regulation")

    circ_sha = None
    if circ is not None:
        circ_json = json.dumps(circ.to_dict(), sort_keys=True, separators=(",", ":"))
        circ_sha = hashlib.sha256(circ_json.encode("utf-8")).hexdigest()

    return CompiledEdgeOrganism(
        name=org.name,
        source=clean_src,
        source_sha256=hashlib.sha256(clean_src.encode("utf-8")).hexdigest(),
        meta=dict(org.meta),
        circuit_sha256=circ_sha,
        rules_sha256=rules.sha256() if rules is not None else None,
        regulation_sha256=reg.sha256() if reg is not None else None,
        circuit=circ,
        rules=rules,
        regulation=reg,
    )


def build_edge_organism_source(
    profile_id: str = "alpha_android_termux",
    organism_name: Optional[str] = None,
    default_horizon_s: int = 300,
) -> str:
    """Synthesize a canonical, Claims-Firewall-clean `.dna` edge administration organism."""
    safe_org = organism_name or ("edge_" + "".join(c if c.isalnum() else "_" for c in profile_id).strip("_"))
    metrics_json = ", ".join(f'"{m}"' for m in EDGE_DECLARED_METRICS)
    return (
        f"// OSIRIS Declarative Edge Provisioning & Adaptive Administration Organism\n"
        f"organism {safe_org} {{\n"
        f"  meta {{\n"
        f"    schema_version: \"osiris.edge.dna/1.0\",\n"
        f"    substrate_profile: \"{profile_id}\",\n"
        f"    default_horizon_s: {int(default_horizon_s)},\n"
        f"    metrics: [{metrics_json}]\n"
        f"  }}\n\n"
        f"  // Ternary Classifier Rules over [network_up, battery_ok, thermal_ok, transport_ready, ledger_ok]\n"
        f"  gene rule_nominal {{\n"
        f"    id: \"R_NOMINAL\",\n"
        f"    condition: \"11111\",\n"
        f"    action: \"(seq (route primary_mentor) (emit STATE_NOMINAL))\"\n"
        f"  }}\n"
        f"  gene rule_offline_autonomous {{\n"
        f"    id: \"R_OFFLINE\",\n"
        f"    condition: \"0#1#1\",\n"
        f"    action: \"(seq (sever cloud_burst) (route local_edge_voice) (emit AUTONOMOUS_OFFLINE))\"\n"
        f"  }}\n"
        f"  gene rule_ledger_lockdown {{\n"
        f"    id: \"R_TAMPER\",\n"
        f"    condition: \"####0\",\n"
        f"    action: \"(seq (sever apply_gate) (sever mtp_push) (route lockdown_readonly) (emit LEDGER_QUARANTINE))\"\n"
        f"  }}\n\n"
        f"  // Topological Regulatory Network\n"
        f"  gene reg_genesis {{\n"
        f"    id: \"G_GENESIS\",\n"
        f"    trigger: \"on_genesis\",\n"
        f"    outputs: [\"boot_ready\"],\n"
        f"    action: \"(seq (set stage provisioned) (emit GENESIS_ENROLLED))\"\n"
        f"  }}\n"
        f"  gene reg_thermal_guard {{\n"
        f"    id: \"G_THERMAL\",\n"
        f"    dependencies: [\"G_GENESIS\"],\n"
        f"    trigger: \"when temp_c >= 43\",\n"
        f"    action: \"(seq (adjust horizon_s -120) (route fast_local_quantized) (emit THERMAL_THROTTLE))\"\n"
        f"  }}\n"
        f"  gene reg_battery_guard {{\n"
        f"    id: \"G_BATTERY\",\n"
        f"    dependencies: [\"G_GENESIS\"],\n"
        f"    trigger: \"when battery_pct <= 20\",\n"
        f"    action: \"(seq (sever unattended_train) (adjust horizon_s -90) (emit LOW_BATTERY_GUARD))\"\n"
        f"  }}\n"
        f"  gene reg_latency_fallback {{\n"
        f"    id: \"G_LATENCY\",\n"
        f"    dependencies: [\"G_GENESIS\"],\n"
        f"    trigger: \"when link_rtt_ms > 500\",\n"
        f"    action: \"(seq (sever cloud_burst) (route local_edge_cache) (emit HIGH_LATENCY_LOCAL_FALLBACK))\"\n"
        f"  }}\n"
        f"  gene reg_usb_ffu_safety {{\n"
        f"    id: \"G_USB_FFU\",\n"
        f"    dependencies: [\"G_GENESIS\"],\n"
        f"    trigger: \"when usb_ffu_mode == 1\",\n"
        f"    action: \"(seq (sever mtp_push) (sever appx_sideload) (send operator EXIT_FFU_HOLD_POWER_VOLDOWN_10S) (emit HALT_UEFI_FFU_MODE))\"\n"
        f"  }}\n"
        f"  gene reg_ledger_integrity {{\n"
        f"    id: \"G_LEDGER\",\n"
        f"    dependencies: [\"G_GENESIS\"],\n"
        f"    trigger: \"when ledger_intact == 0\",\n"
        f"    action: \"(seq (sever apply_gate) (sever distill_export) (route lockdown_readonly) (emit QUARANTINE_LEDGER_TAMPER))\"\n"
        f"  }}\n"
        f"  gene reg_error_trap {{\n"
        f"    id: \"G_ERROR\",\n"
        f"    trigger: \"on_error\",\n"
        f"    action: \"(seq (sever apply_gate) (emit RUNTIME_ERROR_FAIL_CLOSED))\"\n"
        f"  }}\n"
        f"}}\n"
    )


class EdgeOrganismRuntime:
    """Bounded deterministic interpreter executing a `CompiledEdgeOrganism` against `EdgeTelemetry`."""

    def __init__(
        self,
        compiled: CompiledEdgeOrganism,
        *,
        step_budget: int = 64,
        ledger_path: Optional[str] = None,
    ):
        self.compiled = compiled
        self.interpreter = Interpreter(step_budget=step_budget)
        self.ledger_path = ledger_path
        default_horizon = float(compiled.meta.get("default_horizon_s", 300.0))
        self.state = EdgeAdminState(horizon_s=default_horizon)

    def _apply_context(self, ctx: Context) -> None:
        self.state.vars.update(ctx.vars)
        if ctx.routes:
            self.state.active_route = ctx.routes[-1]
        if "horizon_s" in ctx.adjusts:
            self.state.horizon_s = max(30.0, min(1800.0, self.state.horizon_s + ctx.adjusts["horizon_s"]))
        for cap in ctx.severs:
            self.state.severed_capabilities.add(cap)
        for ev in ctx.emits:
            self.state.emitted_events.append(ev)
            self.state.signals_board.add(ev)
        for tgt, msg in ctx.sends:
            self.state.operator_notices.append((tgt, msg))

    def step(self, telemetry: EdgeTelemetry) -> EdgeAdminState:
        """Advance one control tick: evaluate matching ternary rules and topological regulators."""
        self.state.tick += 1
        self.state.emitted_events.clear()
        self.state.operator_notices.clear()
        self.state.fired_rules.clear()
        self.state.fired_regulators.clear()
        self.state.horizon_s = float(self.compiled.meta.get("default_horizon_s", 300.0))
        self.state.severed_capabilities.clear()

        reg_bits = telemetry.capability_register()
        metrics = telemetry.to_metrics()
        last_evt = self.state.emitted_events[-1] if self.state.emitted_events else ""

        # 1. Evaluate ternary RuleSet
        if self.compiled.rules is not None and len(reg_bits) == self.compiled.rules.width:
            for rule in self.compiled.rules.rules:
                if matches(rule.condition, reg_bits):
                    node = parse_sexpr(rule.action)
                    validate(node)
                    ctx = Context(register=reg_bits, last=last_evt, vars=dict(self.state.vars))
                    self.interpreter.run(node, ctx)
                    self._apply_context(ctx)
                    self.state.fired_rules.append(rule.id)
                    if ctx.emits:
                        last_evt = ctx.emits[-1]

        # 2. Evaluate RegulatoryGraph in topological dependency order
        expressed_now: Set[str] = set()
        if self.compiled.regulation is not None:
            by_id = {r.id: r for r in self.compiled.regulation.regulators}
            for gid in self.compiled.regulation.order:
                reg = by_id[gid]
                trig = Trigger.parse(reg.trigger)
                if trig.fires(self.state.tick, metrics, self.state.signals_board, self.state.expressed_last):
                    node = parse_sexpr(reg.action)
                    validate(node)
                    ctx = Context(register=reg_bits, last=last_evt, vars=dict(self.state.vars))
                    self.interpreter.run(node, ctx)
                    self._apply_context(ctx)
                    for out_sig in reg.outputs:
                        self.state.signals_board.add(out_sig)
                    expressed_now.add(gid.upper())
                    self.state.fired_regulators.append(gid)
                    if ctx.emits:
                        last_evt = ctx.emits[-1]
        self.state.expressed_last = expressed_now

        if self.ledger_path:
            Ledger(self.ledger_path).append("EDGE_ADMIN_TICK", {
                "organism": self.compiled.name,
                "organism_sha256": self.compiled.source_sha256,
                "register": reg_bits,
                "metrics": metrics,
                "state": self.state.to_dict(),
            })

        return self.state
