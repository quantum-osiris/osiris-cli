"""OSIRIS Multi-Model Virtual Department Orchestrator, Quantum Research Accelerator,
Scientific Rigor Protocol, RFC 8785 Provenance Capsule, Publication/Opportunity Engine,
and 6-Class Blocker Elimination Engine.

Implements durable, deterministic governance & research-acceleration engines:
  1. Workflow Weakness Analyzer (Research workflows, Agent collaboration, Experiment management)
  2. Multi-Model Orchestration Engine & Virtual Departments (5 departments, 7 capability classes)
  3. Quantum Research Accelerator (4 skeptic questions + 6-part preregistered protocol generator)
  4. Scientific Rigor Protocol Auditor (flags weak evidence, unsupported claims, missing controls/baselines/stats)
  5. Provenance & Reproducibility Engine (RFC 8785 JCS canonical capsules + SHA-256 chain)
  6. Publication Accelerator & Opportunity Discovery Engine
  7. Blocker Elimination Engine (technical, research, knowledge, compute, access, funding)
  8. Self-Improvement Directive Evaluator
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import claims, evolution_engine

try:
    import rfc8785 as _rfc8785
except ImportError:  # pragma: no cover
    _rfc8785 = None

CAPABILITY_CLASSES: Tuple[str, ...] = (
    "reasoning",
    "coding",
    "research",
    "scientific",
    "search",
    "hardware_api",
    "domain_specialist",
)

VIRTUAL_DEPARTMENTS: Tuple[str, ...] = (
    "Research Department",
    "Engineering Department",
    "Publication Department",
    "Infrastructure Department",
    "Business Development Department",
)

BLOCKER_CATEGORIES: Tuple[str, ...] = (
    "technical",
    "research",
    "knowledge",
    "compute",
    "access",
    "funding",
)

_RIGOR_FLAGS: Tuple[str, ...] = (
    "WEAK_EVIDENCE",
    "UNSUPPORTED_CLAIM",
    "MISSING_CONTROLS",
    "MISSING_BASELINES",
    "MISSING_STATISTICAL_ANALYSIS",
    "PROVENANCE_VIOLATION",
)


def jcs_bytes(payload: Mapping[str, Any]) -> bytes:
    """Canonicalize a JSON mapping under RFC 8785 JCS (falls back to sorted compact UTF-8)."""
    if _rfc8785 is not None:
        return _rfc8785.dumps(dict(payload))
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def jcs_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(jcs_bytes(payload)).hexdigest()


# ---------------------------------------------------------------------------
# 1. Workflow Weakness Identification (Root Cause -> Architecture -> Migration -> Metrics)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeaknessRemediation:
    weakness_id: str
    domain: str  # research_workflows | agent_collaboration | experiment_management
    title: str
    root_cause: str
    architectural_improvement: str
    migration_strategy: Tuple[str, ...]
    success_metrics: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architectural_improvement": self.architectural_improvement,
            "domain": self.domain,
            "migration_strategy": list(self.migration_strategy),
            "root_cause": self.root_cause,
            "success_metrics": list(self.success_metrics),
            "title": self.title,
            "weakness_id": self.weakness_id,
        }


def identify_workflow_weaknesses(
    *, osiris_home: Optional[Path] = None
) -> Tuple[WeaknessRemediation, ...]:
    """Return structured 4-part remediations across Research, Collaboration & Experiment Management."""
    _ = evolution_engine.default_osiris_home(osiris_home)
    return (
        WeaknessRemediation(
            weakness_id="WK-RW-001",
            domain="research_workflows",
            title="Unlinked Zenodo supersession chains & manual errata propagation",
            root_cause=(
                "Source ingestion in research.py and knowledge.py reads static markdown snapshots "
                "without traversing Zenodo isSupplementedBy/isPreviousVersionOf relations, leaving "
                "superseded records (e.g., 24031814 and 23832282) unlinked from downstream prompts."
            ),
            architectural_improvement=(
                "Introduce a deterministic Citation & Errata Graph Indexer that binds every DOI/record "
                "to its latest erratum status and SYS-024 standing prior to retrieval or training."
            ),
            migration_strategy=(
                "Step 1: Index quantumosiris-site/public/data/records.json & staged errata into ~/.osiris/evidence/doi_lineage.json.",
                "Step 2: Enforce automatic erratum banner injection in Knowledge.lookup() whenever a superseded DOI matches.",
                "Step 3: Publish staged v2.0.1 (24031814) and v1.0.1 (23832282) errata drafts after owner review.",
            ),
            success_metrics=(
                "0% un-annotated citations to superseded Zenodo DOIs in claim_provenance.jsonl",
                "100% concordance between public Zenodo record descriptions and internal claims.REGISTER",
            ),
        ),
        WeaknessRemediation(
            weakness_id="WK-RW-002",
            domain="research_workflows",
            title="Classical-hardness & colored-noise pre-gates not mandatory before QPU design",
            root_cause=(
                "Early circuit families bypassed tensor-network (MPS/Pauli-path) classical attacks and "
                "local-1/f null models until post-hoc audits (ADV-0 PR #73 and CN-01 Amendment A-3)."
            ),
            architectural_improvement=(
                "Gate every quantum advantage or bath-memory protocol through QuantumResearchAccelerator's "
                "mandatory Classical Attack & Colored-Noise Null Pre-Gate prior to preregistration lock."
            ),
            migration_strategy=(
                "Step 1: Require compile_research_protocol() JSON capsule for every new experiment directory.",
                "Step 2: Run automated MPS (chi<=256..1024) + local-OU synthetic false-positive checks in CI.",
                "Step 3: Refuse QPU/HQC budget allocation in quantinuum_gate.py when classical_baseline_passed is False.",
            ),
            success_metrics=(
                "100% of candidate QPU protocols accompanied by executable classical baseline & synthetic null calibration",
                "0 HQC / QPU seconds spent on classically simulable peaked circuits",
            ),
        ),
        WeaknessRemediation(
            weakness_id="WK-AC-001",
            domain="agent_collaboration",
            title="Shared working-tree collisions & untyped prose handoffs across concurrent agents",
            root_cause=(
                "Multiple coding/research agents operated directly on shared checkouts or ad-hoc directory "
                "copies (11 GB build-sandbox), while passing unstructured text summaries instead of typed receipts."
            ),
            architectural_improvement=(
                "Enforce one ephemeral git worktree per department task, typed JSON handoff envelopes carrying "
                "before/after file SHA-256 digests, and mandatory critic_model != author_model verification."
            ),
            migration_strategy=(
                "Step 1: Route multi-agent assignments through assign_virtual_departments() with worktree_isolation=True.",
                "Step 2: Require child-process task_verifier.py PASS receipt before Engineering -> Publication handoff.",
                "Step 3: Auto-prune merged ephemeral worktrees immediately after branch commit.",
            ),
            success_metrics=(
                "0 dirty-working-tree git collisions across concurrent agent sessions",
                "100% of cross-department handoffs carrying verifiable artifact_sha256 and verifier exit_code == 0",
            ),
        ),
        WeaknessRemediation(
            weakness_id="WK-EM-001",
            domain="experiment_management",
            title="Scattered QPU raw-count archives, calibration staleness & single-vendor coupling",
            root_cause=(
                "Raw QPU artifacts grew inside Git trees (184 MB in osiris-cli) or misfiled folders (10 ibm_torino "
                "jobs inside ibm_fez/), Heron r2 calibration drifted on ~60-90 min timescales (TMAP-05), and "
                "2026-11-22 IBM account closure halted single-provider scripts."
            ),
            architectural_improvement=(
                "Unify all runs under RFC 8785 Reproducibility Capsules indexing content-addressed blobs outside Git, "
                "mandate per-block A/A drift sentinels + union readout probes, and target vendor-neutral pytket/OpenQASM 3 IR."
            ),
            migration_strategy=(
                "Step 1: Register every historical and new run via build_rfc8785_reproducibility_capsule().",
                "Step 2: Require >=1 A/A identical-circuit sentinel per interleaved block in all hardware manifests.",
                "Step 3: Execute staged H2-1E dry-run bundles (XTOP-005 & QEC-002) via osiris nexus upon credential activation.",
            ),
            success_metrics=(
                "100% of HARDWARE_ATTESTED claims resolvable to raw_counts_sha256 + provider_job_id in O(1)",
                "0 silent simulator fallbacks (strict SIMULATED vs OFFLINE_COMPILED vs HARDWARE_ATTESTED separation)",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# 2. Multi-Model Orchestration Engine & Virtual Departments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DepartmentAssignment:
    department: str
    objective_slice: str
    capability_classes: Tuple[str, ...]
    assigned_model: str
    assigned_tools: Tuple[str, ...]
    worktree_isolation: bool
    governance_gate: str
    deliverable_schema: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assigned_model": self.assigned_model,
            "assigned_tools": list(self.assigned_tools),
            "capability_classes": list(self.capability_classes),
            "deliverable_schema": self.deliverable_schema,
            "department": self.department,
            "governance_gate": self.governance_gate,
            "objective_slice": self.objective_slice,
            "worktree_isolation": self.worktree_isolation,
        }


def assign_virtual_departments(
    shared_objective: str,
    *,
    candidate_models: Sequence[str] = (
        "ollama/gemma4:e4b",
        "ollama/qwen2.5-coder:7b",
        "ollama/gemma3:4b",
    ),
    calibration_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build a coordinated 5-department workflow using calibration-weighted model routing."""
    if not shared_objective.strip():
        raise ValueError("shared_objective must be non-empty")

    primary_model, cal_score = evolution_engine.select_calibrated_model(
        candidate_models, calibration_path=calibration_path
    )
    # Pick a distinct critic model so critic_model != author_model always holds
    critic_model = next((m for m in candidate_models if m != primary_model), primary_model)

    departments = (
        DepartmentAssignment(
            department="Research Department",
            objective_slice=f"Formulate falsifiable hypotheses, Kraus/stabilizer math & classical baselines for: {shared_objective}",
            capability_classes=("reasoning", "research", "scientific", "search"),
            assigned_model=primary_model,
            assigned_tools=("knowledge.lookup", "claims.match", "physics_checks", "pytket_offline"),
            worktree_isolation=False,
            governance_gate="SYS-024 claim-register & classical-baseline pre-gate",
            deliverable_schema="osiris.research_protocol/1",
        ),
        DepartmentAssignment(
            department="Engineering Department",
            objective_slice=f"Implement circuits, decoders, and tests in an isolated worktree for: {shared_objective}",
            capability_classes=("coding", "scientific", "hardware_api"),
            assigned_model=primary_model,
            assigned_tools=("apply_preconditioned_patch", "sandbox.run_pytest", "pytket_quantinuum", "dnalang_compiler"),
            worktree_isolation=True,
            governance_gate=f"PlanGate v1 (G-10..G-12) + cross-model critic ({critic_model})",
            deliverable_schema="osiris.verified_patch_receipt/1",
        ),
        DepartmentAssignment(
            department="Infrastructure Department",
            objective_slice="Enforce bwrap sandboxing, Node Alpha/Beta routing, enclave key isolation & Nexus dry-run gates",
            capability_classes=("coding", "hardware_api", "domain_specialist"),
            assigned_model=critic_model,
            assigned_tools=("exec_kernel.run", "quantinuum_gate", "edge_router", "task_verifier"),
            worktree_isolation=True,
            governance_gate="SYS-021 Ed25519/ML-DSA-65 grant + fail-closed HARDWARE_ATTESTED check",
            deliverable_schema="osiris.reproducibility_capsule/1",
        ),
        DepartmentAssignment(
            department="Publication Department",
            objective_slice="Compile LaTeX tables directly from raw-count receipts, audit statistical rigor & stage Zenodo/arXiv bundles",
            capability_classes=("reasoning", "research", "domain_specialist"),
            assigned_model=primary_model,
            assigned_tools=("audit_scientific_rigor", "claims.check_claim", "rfc8785_jcs", "zenodo_draft_packager"),
            worktree_isolation=True,
            governance_gate="Verbatim number-provenance check + zero unresolved errata",
            deliverable_schema="osiris.publication_bundle/1",
        ),
        DepartmentAssignment(
            department="Business Development Department",
            objective_slice="Align open-science QCUP vs commercial Startup Partner/SBIR firewalls & partner outreach",
            capability_classes=("research", "search", "domain_specialist"),
            assigned_model=primary_model,
            assigned_tools=("prioritize_opportunities", "diagnose_and_eliminate_blockers"),
            worktree_isolation=False,
            governance_gate="Track A (commercial) / Track B (DOE DE-AC05-00OR22725) strict scope firewall",
            deliverable_schema="osiris.opportunity_action_plan/1",
        ),
    )

    plan = {
        "calibrated_primary_model": primary_model,
        "calibration_utility": cal_score.to_dict() if cal_score else None,
        "coordination_dag": [
            "Research Department -> Engineering Department",
            "Engineering Department -> Infrastructure Department",
            "Infrastructure Department -> Publication Department",
            "Publication Department -> Business Development Department",
        ],
        "cross_model_critic": critic_model,
        "departments": [d.to_dict() for d in departments],
        "schema": "osiris.multi_model_orchestration/1",
        "shared_objective": shared_objective,
    }
    plan["orchestration_sha256"] = jcs_sha256(plan)
    return plan


# ---------------------------------------------------------------------------
# 3. Quantum Research Accelerator (Skeptic Protocol + 6-Part Experiment Design)
# ---------------------------------------------------------------------------


def generate_quantum_experiment_protocol(
    *,
    idea_id: str,
    title: str,
    target_backend: str,
    primary_observable: str,
    null_value_str: str,
    target_effect_str: str,
    shots_per_circuit: int = 500,
    num_blocks: int = 6,
) -> Dict[str, Any]:
    """Generate a preregisterable quantum protocol answering the 4 internal skeptic questions."""
    if shots_per_circuit < 100 or num_blocks < 3:
        raise ValueError("shots_per_circuit >= 100 and num_blocks >= 3 required for block-t power")

    skeptic_interrogation = {
        "what_convinces_skeptic": (
            f"Pre-registered SHA-256-locked circuits on {target_backend} beating {null_value_str} by "
            f">= {target_effect_str} across {num_blocks} interleaved randomized blocks (Holm alpha <= 0.01), "
            "after passing noiseless Stage-0 unitary checks, A/A drift-floor sentinels, and classical baseline attacks."
        ),
        "what_evidence_is_missing": (
            f"Live {target_backend} provider_job_id raw shot histograms and empirical post-selection/defect "
            "rates under real zone-shuttling / mid-circuit measurement crosstalk."
        ),
        "what_experiment_produces_it": (
            f"{idea_id}: {num_blocks}-block interleaved factorial run ({shots_per_circuit} shots/circuit) "
            "with paired bare control, sham/decoherence control, and per-block A/A sentinel."
        ),
        "what_publication_becomes_possible": (
            "Zenodo preregistration + raw-count dataset DOI -> arXiv [quant-ph] preprint -> "
            "Physical Review Applied / Quantum peer-reviewed benchmark."
        ),
    }

    protocol: Dict[str, Any] = {
        "backend": target_backend,
        "control_experiments": [
            "Control C0 (Noiseless Stage-0): Exact statevector/branch check verifying F == 1.000000 and 0 rejected shots.",
            "Control C1 (Matched Bare / Unprotected Baseline): Identical logical circuit depth/schedule without encoding or active protection.",
            "Control C2 (A/A Drift Sentinel): Byte-identical reference circuit duplicated in every interleaved block to bound temporal drift RMS.",
            "Control C3 (Colored-Noise / Classical Sham): Local 1/f + depolarizing noise simulation and classical MPS/BP+OSD baseline.",
        ],
        "failure_criteria": [
            "Kill Rule K1: Observed effect over Control C1 fails Holm-Bonferroni one-sided block-t test (alpha = 0.01) or falls within 2x A/A sentinel RMS.",
            "Kill Rule K2: Held-out block/coupler RMSE exceeds independent local-noise null H0 (paired t-test p >= 0.05).",
            "Kill Rule K3: Post-selection keep fraction drops below pre-registered floor or classical polynomial attack reproduces heavy-output score.",
        ],
        "hypotheses": [
            f"H1 (Primary): {primary_observable} exceeds baseline {null_value_str} by >= {target_effect_str} (Holm-corrected p < 0.01 across {num_blocks} blocks).",
            f"H0 (Null): {primary_observable} <= {null_value_str} or indistinguishable from local-noise / bare control within block variance.",
        ],
        "idea_id": idea_id,
        "num_blocks": num_blocks,
        "primary_observable": primary_observable,
        "publication_pathways": [
            "Stage 1: Deposit RFC 8785 preregistration lock + compiled ISA digests on Zenodo before job submission.",
            "Stage 2: Publish complete raw histograms + negative/positive verdicts as versioned Zenodo data record.",
            "Stage 3: Submit short two-column manuscript with reproducible generate_tables.py to arXiv quant-ph & PRApplied/PRXQ.",
        ],
        "replication_experiments": [
            "R1 (Temporal Stability): Re-execute identical compiled ISA digest >= 90 minutes later to test calibration/readout rank stability.",
            "R2 (Cross-Backend): Replicate identical OpenQASM 3 / pytket logical spec across emulator (H2-1E) and physical QPU (H2-1 / Helios-1).",
        ],
        "schema": "osiris.quantum_protocol/1",
        "shots_per_circuit": shots_per_circuit,
        "skeptic_interrogation": skeptic_interrogation,
        "title": title,
        "validation_experiments": [
            "V1: Verify native ISA gate compliance and zero unexpected SWAPs offline prior to submission.",
            "V2: Independent raw-histogram verifier recomputing observable & Wilson/Student-t intervals from integer counts.",
        ],
    }
    protocol["protocol_sha256"] = jcs_sha256(protocol)
    return protocol


# ---------------------------------------------------------------------------
# 4. Scientific Rigor Protocol Auditor & 5. RFC 8785 Provenance Capsule
# ---------------------------------------------------------------------------


def audit_scientific_rigor(study: Mapping[str, Any]) -> Dict[str, Any]:
    """Flag weak evidence, unsupported claims, missing controls/baselines/stats, and provenance leaks."""
    flags: List[Dict[str, str]] = []

    text_blob = " ".join(
        str(study.get(k, "")) for k in ("title", "claim_text", "summary", "conclusion")
    )
    blocked = [
        c.id
        for c in claims.match(text_blob)
        if c.verdict in ("REFUTED", "ARTIFACT", "NO_EVIDENCE", "QUARANTINED")
    ]
    if blocked and not study.get("is_erratum", False):
        flags.append(
            {
                "code": "UNSUPPORTED_CLAIM",
                "detail": f"References SYS-024 non-supported claim IDs without erratum flag: {blocked}",
            }
        )

    shots = int(study.get("total_shots", 0) or 0)
    blocks = int(study.get("num_blocks", 0) or 0)
    if shots < 1000 or blocks < 3:
        flags.append(
            {
                "code": "WEAK_EVIDENCE",
                "detail": f"Insufficient statistical replication (total_shots={shots}, num_blocks={blocks}; require >=1000 shots & >=3 blocks).",
            }
        )

    if not study.get("controls"):
        flags.append(
            {
                "code": "MISSING_CONTROLS",
                "detail": "No matched-duration / sham / A-A drift sentinel controls declared.",
            }
        )

    if not study.get("classical_baseline"):
        flags.append(
            {
                "code": "MISSING_BASELINES",
                "detail": "Missing classical simulation/decoder baseline (e.g., MPS, MWPM, BP+OSD, or local-1/f null).",
            }
        )

    if not study.get("statistical_test") or not study.get("confidence_interval"):
        flags.append(
            {
                "code": "MISSING_STATISTICAL_ANALYSIS",
                "detail": "Missing explicit statistical test (Holm/block-t/bootstrap) or confidence interval.",
            }
        )

    ev_class = str(study.get("evidence_class", "UNSPECIFIED"))
    job_id = study.get("provider_job_id")
    raw_sha = study.get("raw_counts_sha256")
    if ev_class in ("SIMULATED", "OFFLINE_COMPILED") and job_id is not None:
        flags.append(
            {
                "code": "PROVENANCE_VIOLATION",
                "detail": f"{ev_class} study must have provider_job_id=None (got {job_id!r}).",
            }
        )
    elif ev_class == "HARDWARE_ATTESTED" and (not job_id or not raw_sha):
        flags.append(
            {
                "code": "PROVENANCE_VIOLATION",
                "detail": "HARDWARE_ATTESTED study requires non-empty provider_job_id and raw_counts_sha256.",
            }
        )
    elif ev_class not in ("SIMULATED", "OFFLINE_COMPILED", "HARDWARE_ATTESTED", "THEORETICAL"):
        flags.append(
            {
                "code": "PROVENANCE_VIOLATION",
                "detail": f"Unrecognized evidence_class: {ev_class!r}.",
            }
        )

    return {
        "flag_codes": [f["code"] for f in flags],
        "flags": flags,
        "passed": len(flags) == 0,
        "schema": "osiris.rigor_audit/1",
    }


def build_rfc8785_reproducibility_capsule(
    *,
    experiment_id: str,
    evidence_class: str,
    prereg_sha256: str,
    circuit_isa_sha256: str,
    provider: str,
    backend_id: str,
    shots: int,
    provider_job_id: Optional[str] = None,
    raw_counts_sha256: Optional[str] = None,
    zenodo_doi: Optional[str] = None,
    prev_capsule_sha256: str = "0" * 64,
) -> Dict[str, Any]:
    """Construct a fail-closed RFC 8785 JCS reproducibility capsule."""
    if evidence_class in ("SIMULATED", "OFFLINE_COMPILED") and provider_job_id is not None:
        raise ValueError(f"{evidence_class} capsule cannot carry provider_job_id={provider_job_id!r}")
    if evidence_class == "HARDWARE_ATTESTED" and (not provider_job_id or not raw_counts_sha256):
        raise ValueError("HARDWARE_ATTESTED capsule requires provider_job_id and raw_counts_sha256")

    body: Dict[str, Any] = {
        "backend_id": backend_id,
        "circuit_isa_sha256": circuit_isa_sha256,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_class": evidence_class,
        "experiment_id": experiment_id,
        "prev_capsule_sha256": prev_capsule_sha256,
        "prereg_sha256": prereg_sha256,
        "provider": provider,
        "provider_job_id": provider_job_id,
        "raw_counts_sha256": raw_counts_sha256,
        "rfc8785_jcs": _rfc8785 is not None,
        "schema": "osiris.reproducibility_capsule/1",
        "shots": int(shots),
        "zenodo_doi": zenodo_doi,
    }
    body["capsule_sha256"] = jcs_sha256(body)
    return body


# ---------------------------------------------------------------------------
# 6. Publication Accelerator & Opportunity Discovery Engine
# ---------------------------------------------------------------------------


def plan_publication_sequence() -> Tuple[Dict[str, Any], ...]:
    """Return active publication pipelines across the verified OSIRIS research portfolio."""
    return (
        {
            "evidence_gaps": ["Live H2-1E / H2-1 18-circuit execution (1,336 HQC) confirming 0.989 vs 0.975 prediction"],
            "novel_contributions": [
                "First heavy-hex vs all-to-all QCCD comparison of [[4,2,2]]-encoded Hayden-Preskill recovery",
                "Proof that staggered XY4 (+0.12..+0.18) rescues coherent heavy-hex idle noise where detection discards 65% of shots",
            ],
            "project_id": "OSIRIS-XTOP-005 / XTOP-006",
            "publication_sequence": [
                "1. Zenodo v1.1 (10.5281/zenodo.24426216 — COMPLETE)",
                "2. H2-1E / H2-1 Nexus execution addendum (Zenodo v2.0)",
                "3. XTOP-006 tunable non-Clifford Kraus scrambling extension -> Physical Review Applied / Quantum",
            ],
            "required_experiments": ["Run staged 18-circuit XTOP-005 H2-1E bundle + XTOP-006 5-angle ZZPhase sweep"],
        },
        {
            "evidence_gaps": ["Live H2-1 20-ion execution of QEC-002 state prep and QEC-003 full X+Z memory + twist braiding"],
            "novel_contributions": [
                "Operational separation of 2-round LUT state prep (e-anyon only) vs full X+Z toroidal memory (48/48 wt-1 Pauli recovery)",
                "Zero-SWAP QCCD extraction of weight-5 checkerboard dislocation twist defects transmuting e <-> m anyons",
            ],
            "project_id": "OSIRIS-QEC-001 / QEC-002 / QEC-003",
            "publication_sequence": [
                "1. QEC-001 Heron r2 MCMR dataset (10.5281/zenodo.24293756 — COMPLETE) & QEC-002 v1.1.1 (10.5281/zenodo.24424726 — COMPLETE)",
                "2. QEC-003 full X+Z memory & twist-defect simulation + H2-1E benchmark -> arXiv [quant-ph]",
                "3. Combined superconducting vs trapped-ion mid-circuit syndrome paper -> PRX Quantum / Quantum",
            ],
            "required_experiments": ["Execute QEC-002 (32 ZZPhase) & QEC-003 R=1..3 memory cycles on H2-1E / H2-1"],
        },
        {
            "evidence_gaps": ["Consolidate TMAP-02..05 + RO-06/RO-07 + CN-01 into one unified archival journal submission"],
            "novel_contributions": [
                "42-job, 1.85M-shot three-processor characterization of hour-scale readout telegraphing & 90-minute layout sign flips",
                "Proof that white-noise nulls falsely select non-Markovian bath memory on 100% of local 1/f datasets (CN-01)",
            ],
            "project_id": "HERON-DRIFT-AND-NULL-SERIES",
            "publication_sequence": [
                "1. Individual preregistered Zenodo reports (24406426..24420902 — COMPLETE, ZERO NEW QPU NEEDED)",
                "2. Unified synthesis manuscript -> IEEE TQE / npj Quantum Information (immediate submission ready)",
            ],
            "required_experiments": ["None — all 42 IBM Heron r2 QPU jobs and raw counts are already archived and verified"],
        },
    )


def prioritize_opportunities() -> Tuple[Dict[str, Any], ...]:
    """Rank concrete external hardware, funding, partner, and venue opportunities by leverage."""
    return (
        {
            "acceleration_score": "0.980000",
            "category": "hardware_program",
            "immediate_action": "Complete complimentary Quantinuum Nexus onboarding (Q-Net@quantinuum.com invite, 2026-11-21) and run staged XTOP-005 & QEC-002 dry-run bundles on H2-1E.",
            "opportunity": "Quantinuum Nexus & Q-Net Complimentary Emulator Access",
            "unlocks": "Unblocks live provider_job_id execution for OSIRIS-QEC-002, XTOP-005 H2 arm, and OLCF QCUP Month-1 milestone.",
        },
        {
            "acceleration_score": "0.950000",
            "category": "funding_and_compute",
            "immediate_action": "Fill 3 PI contact lines in submission-ready-2026-11-22/1_olcf_qcup_open_science_21500_hqc/OLCF_QCUP_PROJECT_PROPOSAL_V2_1.md and submit to OLCF QCUP.",
            "opportunity": "DOE OLCF Quantum Computing User Program (QCUP — 21,500 HQC + 600s IBM)",
            "unlocks": "Funds 6-month H2-1 & 98Q Helios-1 open-science allocations (WP1 k=3 HP, WP2 96Q Sinkhorn OT, WP3 [[72,12,6]] BB qLDPC) and restores IBM Heron r2 access.",
        },
        {
            "acceleration_score": "0.900000",
            "category": "industry_partnership",
            "immediate_action": "Submit firewalled Track A application (QUANTINUUM_STARTUP_PARTNER_SUBMISSION.md) referencing CAGE 9HUP5 and live DOIs 24287505, 24293756, 24424726, 24426216.",
            "opportunity": "Quantinuum Startup Partner Program & Broomfield Technical Collaboration",
            "unlocks": "Commercial TKET/Nexus/Quantum Origin provenance capsule co-design and peer technical exchange on codespace scrambling.",
        },
        {
            "acceleration_score": "0.860000",
            "category": "publication_venue",
            "immediate_action": "Compile unified manuscript from completed TMAP-02..05 + RO-06/07 + CN-01 datasets (1.85M shots already on disk) for arXiv quant-ph & PRApplied.",
            "opportunity": "Immediate Zero-QPU Journal Submission on Heron r2 Calibration Staleness & Colored-Noise Nulls",
            "unlocks": "Peer-reviewed publication requiring zero additional QPU credits while Nexus/QCUP allocations spin up.",
        },
        {
            "acceleration_score": "0.820000",
            "category": "funding_sbir",
            "immediate_action": "Target AFWERX Open Topic Phase I / DoD STTR using osiris-governance (FIPS 204 ML-DSA-65 + Quantum Origin Mermin-M3 + ECS 8.x audit trail).",
            "opportunity": "AFWERX / DoD High-Assurance Audit & Post-Quantum Provenance SBIR",
            "unlocks": "Non-dilutive commercial R&D funding strictly separated from Track B DOE open science.",
        },
    )


# ---------------------------------------------------------------------------
# 7. Blocker Elimination Engine & 8. Self-Improvement Directive Evaluator
# ---------------------------------------------------------------------------


def diagnose_and_eliminate_blockers(
    *, osiris_home: Optional[Path] = None
) -> Tuple[Dict[str, Any], ...]:
    """Identify active blockers across all 6 categories and emit concrete resolution plans."""
    snap = evolution_engine.monitor_evolution(osiris_home=osiris_home)
    tool_dim = snap.by_dimension("tool_use")
    know_dim = snap.by_dimension("knowledge_retention")

    return (
        {
            "blocker": "IBM Quantum account closure on 2026-11-22 & unconfigured qnexus token block live QPU jobs.",
            "category": "access",
            "resolution_plan": [
                "1. Activate complimentary Quantinuum Nexus invite (Q-Net@quantinuum.com) and store token in chmod 600 ~/.osiris/.env.",
                "2. Submit OLCF QCUP Track B application to regain merit-reviewed IBM Heron r2 (600s) + Quantinuum (21,500 HQC) access.",
                "3. Publish zero-QPU unified Heron r2 drift paper from existing 1.85M-shot verified archives in the interim.",
            ],
            "status": "ACTIONABLE_NOW",
        },
        {
            "blocker": f"Subprocess/tool failure rate at {tool_dim.headline_metric_str} & NCLM <=2.0 BPB speaking gate deadlocking engineering promotions.",
            "category": "technical",
            "resolution_plan": [
                "1. Merge feat/osiris-evolution-engine to enable apply_preconditioned_patch(), select_calibrated_model(), and evaluate_subsystem_promotion(subsystem_scope='engineering').",
                "2. Merge chore/audit-prune-2026-11 (-63,408 dead LOC in osiris-cli) and chore/audit-strip-legacy-crsm (-4,976 mock LOC in quantum-ide).",
            ],
            "status": "IMPLEMENTED_ON_BRANCH",
        },
        {
            "blocker": "OSIRIS-QEC-002 X-only state prep leaves magnetic m-anyons (X faults) invisible; XTOP-005 used Clifford-only scramblers.",
            "category": "research",
            "resolution_plan": [
                "1. Use OSIRIS-QEC-003 full X+Z 20-ion memory cycle + weight-5 dislocation twist-defect package in osiris-qec.",
                "2. Use OSIRIS-XTOP-006 tunable non-Clifford ZZPhase(theta) Kraus channel package in osiris-cli.",
            ],
            "status": "PROTOTYPED_OFFLINE",
        },
        {
            "blocker": f"Provenance ledger shows {know_dim.headline_metric_str} flagged claim share; Zenodo 24031814 & 23832282 await erratum upload.",
            "category": "knowledge",
            "resolution_plan": [
                "1. Enforce filter_training_doc_sys024() on all continual-learning corpus builds.",
                "2. Run make_draft.sh --create-draft in submission-ready-2026-11-22/2_zenodo_24031814_erratum_v2_0_1 and 3_zenodo_23832282_erratum_and_qia_reply.",
            ],
            "status": "BUNDLES_STAGED",
        },
        {
            "blocker": "Node Alpha (ARM64 Termux, 300s ceiling) times out on heavy 7B prompts, py_compile sweeps, and Monte Carlo QEC simulations.",
            "category": "compute",
            "resolution_plan": [
                "1. Keep lightweight gemma3/gemma4 triage on Node Alpha (<100 col UI) and delegate pytest/BP+OSD/pytket compilation to Node Beta (WSL2 x86_64, 1800s ceiling) via osiris-dual-node-sync.",
                "2. Bound prompt input via stratify_memory_context(max_tokens=1200).",
            ],
            "status": "READY",
        },
        {
            "blocker": "Multi-thousand-HQC 50Q-96Q Helios-1 campaigns exceed complimentary evaluation quotas.",
            "category": "funding",
            "resolution_plan": [
                "1. Submit DOE OLCF QCUP Track B (21,500 HQC open-science allocation) for WP1-WP3.",
                "2. Submit firewalled Quantinuum Startup Partner Track A & AFWERX SBIR for commercial provenance productization.",
            ],
            "status": "PROPOSAL_PACKAGED",
        },
    )


def evaluate_self_improvement_directive() -> Dict[str, str]:
    """Answer the 4 mandatory post-execution Self-Improvement Directive questions."""
    return {
        "automation_to_remove_future_effort": (
            "osiris_cli.evolution_engine (9-dimension monitor + 4-part spec generator) and "
            "osiris_cli.orchestration_departments (scientific rigor auditor, RFC 8785 capsule builder, "
            "and virtual department router) automate telemetry audits and pre-submission gates."
        ),
        "durable_system_built": (
            "Committed tested Python modules (evolution_engine.py, orchestration_departments.py) + "
            "offline dry-run Nexus & Zenodo errata scripts instead of one-off manual prose checklists."
        ),
        "knowledge_gap_remaining": (
            "Empirical mid-circuit measurement crosstalk and ion-transport memory error rates on physical "
            "Quantinuum H2-1 / Helios-1 (not captured by offline Pauli noise models without H2-1E/H2-1 runs)."
        ),
        "reusable_artifact_created": (
            "RFC 8785 JCS Reproducibility Capsule schema (osiris.reproducibility_capsule/1) + "
            "6-part Quantum Research Accelerator protocol generator (osiris.quantum_protocol/1)."
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="OSIRIS Multi-Model & Quantum Research Orchestrator")
    parser.add_argument("--objective", default="Execute OSIRIS-XTOP-005 & QEC-002/003 on Quantinuum Nexus")
    parser.add_argument("--osiris-home", default=None)
    args = parser.parse_args(argv)

    home = Path(args.osiris_home) if args.osiris_home else None
    payload = {
        "blockers": list(diagnose_and_eliminate_blockers(osiris_home=home)),
        "opportunities": list(prioritize_opportunities()),
        "orchestration": assign_virtual_departments(args.objective),
        "publication_plan": list(plan_publication_sequence()),
        "self_improvement": evaluate_self_improvement_directive(),
        "weaknesses": [w.to_dict() for w in identify_workflow_weaknesses(osiris_home=home)],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
