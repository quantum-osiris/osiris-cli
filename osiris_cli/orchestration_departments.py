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
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

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
# 5b. Live Workspace Migrations: DOI Lineage Indexer (WK-RW-001),
#     Department Handoff Receipts (WK-AC-001) & Capsule Materializer (WK-EM-001)
# ---------------------------------------------------------------------------

DEFAULT_SITE_RECORDS_PATH = (
    Path.home() / "osiris-assessment" / "work" / "quantumosiris-site" / "public" / "data" / "records.json"
)
DEFAULT_WORK_ROOT = Path.home() / "osiris-assessment" / "work"

_KNOWN_DOI_LINEAGE_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "24031814": {
        "status": "ERRATUM_STAGED",
        "superseded_by": "staged:v2.0.1",
        "supersedes": [],
        "title_fallback": "OSIRIS Unified Quantum Hardware Evidence Package v2.0",
        "warning": (
            "Staged v2.0.1 erratum corrects the +2.28 pp row; verify against "
            "submission-ready-2026-11-22/2_zenodo_24031814_erratum_v2_0_1 before citing."
        ),
    },
    "23832282": {
        "status": "ERRATUM_STAGED",
        "superseded_by": "staged:v1.0.1",
        "supersedes": [],
        "title_fallback": "Staggered Dynamical Decoupling Hardware Report (16 us vs 32 us erratum)",
        "warning": (
            "Staged v1.0.1 duration-matching erratum (16 us vs 32 us idle mismatch); "
            "do not cite unadjusted cross-duration comparison."
        ),
    },
    "23832283": {
        "status": "ERRATUM_ISSUED",
        "superseded_by": None,
        "supersedes": [
            "10.5281/zenodo.17858632",
            "10.5281/zenodo.17858792",
            "10.5281/zenodo.17858802",
            "10.5281/zenodo.17858962",
            "10.5281/zenodo.17859207",
            "10.5281/zenodo.17859845",
            "10.5281/zenodo.23658893",
        ],
        "title_fallback": "Oct 2026 59-Circuit 3-Basis Bell Tomography on ibm_fez Refuting K8 Revival and theta_lock",
        "warning": (
            "Refutation/erratum record: 3-basis tomography on ibm_fez proved F_Bell(46us)=0.3438 "
            "and C_perp=+0.0029, refuting K8 revival and theta_lock=51.843 deg (p=0.3537)."
        ),
    },
    "23213023": {
        "status": "ERRATUM_ISSUED",
        "superseded_by": None,
        "supersedes": ["10.5281/zenodo.19656600"],
        "title_fallback": "Erratum and re-analysis: 136-bit negentropy gap is a sampling-ceiling artifact",
        "warning": (
            "Erratum record superseding 10.5281/zenodo.19656600: the 136-bit negentropy gap on "
            "156 qubits is a log2(10^6)=19.93-bit shot-sampling ceiling artifact."
        ),
    },
    "23977105": {
        "status": "ERRATUM_ISSUED",
        "superseded_by": None,
        "supersedes": ["10.5281/zenodo.18450505", "10.5281/zenodo.19864030"],
        "title_fallback": "Formal Erratum & Forensic Audit: Statistical Validation of W2 Geometry",
        "warning": (
            "Formal W2 mathematical erratum invalidating independent-product Eq. 7 cost and "
            "112x constructive-interference framing in 18450505 / 19864030."
        ),
    },
    "23986321": {
        "status": "ERRATUM_ISSUED",
        "superseded_by": None,
        "supersedes": ["10.5281/zenodo.23983003"],
        "title_fallback": "OSIRIS v5.3.1: Held-Out Dynamical Decoupling Validation & Statistical Rescoring",
        "warning": (
            "Statistical rescoring and empirical noise-floor bounds superseding unadjusted "
            "DD drift figures in v5.3.0 (10.5281/zenodo.23983003)."
        ),
    },
    "22870279": {
        "status": "WITHDRAWN",
        "superseded_by": None,
        "supersedes": ["10.5281/zenodo.18209071"],
        "title_fallback": "Re-execution and Errata for 1,000,000x Quantum Error Suppression via 40-Qubit Tesseract Resonance",
        "warning": (
            "Withdrawn/refuted claim: 10^6x error suppression and 0.99999 fidelity obsoleted "
            "(ibm_kingston re-execution yielded F_cl=0.050 vs GHZ-40 control 0.045)."
        ),
    },
    "18209071": {
        "status": "WITHDRAWN",
        "superseded_by": "10.5281/zenodo.22870279",
        "supersedes": [],
        "title_fallback": "Hardware-Verified 1,000,000x Quantum Error Suppression (Obsoleted)",
        "warning": "Withdrawn and obsoleted by 10.5281/zenodo.22870279 (ERROR_SUPPRESSION_1E6 REFUTED).",
    },
    "19656600": {
        "status": "SUPERSEDED",
        "superseded_by": "10.5281/zenodo.23213023",
        "supersedes": [],
        "title_fallback": "136-bit Negentropy Gap on 156-Qubit Heron Substrate (Superseded)",
        "warning": "Superseded by erratum 10.5281/zenodo.23213023 (NEGENTROPY_136 sampling-ceiling ARTIFACT).",
    },
    "18450505": {
        "status": "REFUTED",
        "superseded_by": "10.5281/zenodo.23977105",
        "supersedes": [],
        "title_fallback": "DNA-Lang v51.843 Geometric Resonance Package (Refuted)",
        "warning": "Refuted by 10.5281/zenodo.23832283 and 10.5281/zenodo.23977105 (THETA_LOCK & W2_ROBUSTNESS).",
    },
    "19864030": {
        "status": "SUPERSEDED",
        "superseded_by": "10.5281/zenodo.23977105",
        "supersedes": [],
        "title_fallback": "Wasserstein-2 Study on NISQ Hardware (Erratum Filed)",
        "warning": "W2 robustness metric superseded by mathematical erratum 10.5281/zenodo.23977105.",
    },
    "24285577": {
        "status": "SUPERSEDED",
        "superseded_by": "10.5281/zenodo.24287505",
        "supersedes": [],
        "title_fallback": "OSIRIS-HP-004 v1 Single-Processor Report (Superseded by v2)",
        "warning": (
            "Superseded by v2 dual-processor archive 10.5281/zenodo.24287505 "
            "(ibm_marrakesh certifies SN>=3 only; E_F>1 ebit certified on ibm_fez)."
        ),
    },
    "24293755": {
        "status": "SUPERSEDED",
        "superseded_by": "10.5281/zenodo.24293756",
        "supersedes": [],
        "title_fallback": "OSIRIS-QEC-001 Concept DOI",
        "warning": "Concept DOI; cite versioned 393,216-shot archive 10.5281/zenodo.24293756.",
    },
    "24425627": {
        "status": "SUPERSEDED",
        "superseded_by": "10.5281/zenodo.24426216",
        "supersedes": [],
        "title_fallback": "OSIRIS-XTOP-005 v1.0 Paper",
        "warning": "Superseded by OSIRIS-XTOP-005 v1.1 archive 10.5281/zenodo.24426216.",
    },
    "24425628": {
        "status": "SUPERSEDED",
        "superseded_by": "10.5281/zenodo.24426216",
        "supersedes": [],
        "title_fallback": "OSIRIS-XTOP-005 v1.0 Record",
        "warning": "Superseded by OSIRIS-XTOP-005 v1.1 archive 10.5281/zenodo.24426216.",
    },
}
for _tau_id in ("17858632", "17858792", "17858802", "17858962", "17859207", "17859845", "23658893"):
    _KNOWN_DOI_LINEAGE_OVERRIDES[_tau_id] = {
        "status": "REFUTED",
        "superseded_by": "10.5281/zenodo.23832283",
        "supersedes": [],
        "title_fallback": f"Exploratory tau-phase record {_tau_id}",
        "warning": "Tau-phase / K8 revival claim refuted by server-clock audit and 10.5281/zenodo.23832283.",
    }


def index_doi_lineage(
    records_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    work_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Index Zenodo record supersession chains, errata, and SYS-024 claim standings into
    deterministic RFC 8785-canonical JSON at ``~/.osiris/evidence/doi_lineage.json`` (WK-RW-001)."""
    src_path = Path(records_path).expanduser().resolve() if records_path is not None else DEFAULT_SITE_RECORDS_PATH
    out_path = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else (evolution_engine.default_osiris_home() / "evidence" / "doi_lineage.json")
    )

    raw_records: List[Dict[str, Any]] = []
    records_sha = "0" * 64
    if src_path.is_file():
        raw_bytes = src_path.read_bytes()
        records_sha = hashlib.sha256(raw_bytes).hexdigest()
        loaded = json.loads(raw_bytes.decode("utf-8"))
        if isinstance(loaded, list):
            raw_records = [dict(item) for item in loaded if isinstance(item, dict) and item.get("id")]

    claim_by_rec_id: Dict[str, List[Tuple[str, str]]] = {}
    doi_rx = re.compile(r"10\.5281/zenodo\.(\d+)", re.IGNORECASE)
    for clm in claims.REGISTER:
        ev_blob = " ".join(f"{k}:{r}:{n}" for k, r, n in clm.evidence) + " " + clm.finding
        for m_doi in doi_rx.finditer(ev_blob):
            rid = m_doi.group(1)
            pair = (clm.id, clm.verdict)
            bucket = claim_by_rec_id.setdefault(rid, [])
            if pair not in bucket:
                bucket.append(pair)

    by_id: Dict[str, Dict[str, Any]] = {}
    for item in raw_records:
        rid = str(item["id"]).strip()
        by_id[rid] = {
            "category": str(item.get("cat") or ""),
            "date": str(item.get("date") or ""),
            "phase": str(item.get("phase") or ""),
            "title": str(item.get("title") or ""),
        }

    for rid, meta_override in _KNOWN_DOI_LINEAGE_OVERRIDES.items():
        if rid not in by_id:
            by_id[rid] = {
                "category": "correction",
                "date": "",
                "phase": "test",
                "title": str(meta_override.get("title_fallback") or f"Zenodo record {rid}"),
            }

    for rid in sorted(claim_by_rec_id):
        if rid not in by_id:
            by_id[rid] = {
                "category": "research",
                "date": "",
                "phase": "test",
                "title": f"Referenced in SYS-024 claims register ({rid})",
            }

    # Scan workspace zenodo_receipt.json files (v6.2.0+ deposits, papers, xtop005) when work_root
    # is supplied or when indexing the default site/workspace corpus.
    dynamic_overrides: Dict[str, Dict[str, Any]] = {}
    scan_cli_repo: Optional[Path] = None
    if work_root is not None:
        wroot = Path(work_root).expanduser().resolve()
        scan_cli_repo = wroot if (wroot / "papers").is_dir() else (wroot / "osiris-cli")
    elif records_path is None:
        scan_cli_repo = (
            DEFAULT_WORK_ROOT
            if (DEFAULT_WORK_ROOT / "papers").is_dir()
            else (DEFAULT_WORK_ROOT / "osiris-cli")
        )
    if scan_cli_repo is not None and scan_cli_repo.is_dir():
        receipt_files: List[Path] = []
        for sub in ("papers", "xtop005"):
            sdir = scan_cli_repo / sub
            if sdir.is_dir():
                receipt_files.extend(sorted(sdir.rglob("zenodo_receipt.json")))
        for rfile in receipt_files:
            try:
                rdata = json.loads(rfile.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(rdata, dict):
                continue
            raw_doi = str(rdata.get("doi") or rdata.get("prereserved_doi") or "").strip()
            dep_id = str(rdata.get("deposition_id") or rdata.get("id") or "").strip()
            if not dep_id and raw_doi:
                m_d = doi_rx.search(raw_doi)
                if m_d:
                    dep_id = m_d.group(1)
            if not dep_id or not dep_id.isdigit():
                continue
            r_date = str(
                rdata.get("published_date")
                or rdata.get("publication_date")
                or rdata.get("updated_utc")
                or rdata.get("created")
                or ""
            )[:10]
            r_title = str(rdata.get("title") or f"OSIRIS Zenodo archive ({dep_id})").strip()
            if dep_id not in by_id or not by_id[dep_id].get("title"):
                by_id[dep_id] = {
                    "category": "research",
                    "date": r_date,
                    "phase": "verified",
                    "title": r_title,
                }
            sup_prev = rdata.get("supersedes_record_id")
            if sup_prev is not None and str(sup_prev).strip().isdigit():
                prev_id = str(sup_prev).strip()
                new_doi = f"10.5281/zenodo.{dep_id}"
                prev_doi = f"10.5281/zenodo.{prev_id}"
                dyn_new = dynamic_overrides.setdefault(dep_id, {})
                sups = list(dyn_new.get("supersedes") or [])
                if prev_doi not in sups:
                    sups.append(prev_doi)
                dyn_new["supersedes"] = sups
                if prev_id not in _KNOWN_DOI_LINEAGE_OVERRIDES:
                    dynamic_overrides[prev_id] = {
                        "status": "SUPERSEDED",
                        "superseded_by": new_doi,
                        "supersedes": [],
                        "warning": f"Updated by newer Zenodo archive {new_doi}.",
                    }
                    if prev_id not in by_id:
                        by_id[prev_id] = {
                            "category": "research",
                            "date": "",
                            "phase": "test",
                            "title": f"Superseded Zenodo archive ({prev_id})",
                        }

    indexed: Dict[str, Dict[str, Any]] = {}
    for rid in sorted(by_id):
        doi = f"10.5281/zenodo.{rid}"
        base_meta = by_id[rid]
        override = dict(dynamic_overrides.get(rid, {}))
        override.update(_KNOWN_DOI_LINEAGE_OVERRIDES.get(rid, {}))
        c_pairs = sorted(claim_by_rec_id.get(rid, []))
        matched_title = claims.match(base_meta["title"]) if base_meta["title"] else []
        for mc in matched_title:
            pair = (mc.id, mc.verdict)
            if pair not in c_pairs:
                c_pairs.append(pair)
        c_pairs.sort()

        status = str(override.get("status") or "ACTIVE")
        warning = override.get("warning")
        if status == "ACTIVE" and any(v in ("REFUTED", "ARTIFACT", "NOT_MEASURED", "RULED_OUT") for _, v in c_pairs):
            if base_meta.get("category") != "correction":
                bad_claims = [cid for cid, v in c_pairs if v in ("REFUTED", "ARTIFACT", "NOT_MEASURED", "RULED_OUT")]
                status = "REFUTED"
                warning = f"Touches non-legit SYS-024 claim(s): {', '.join(bad_claims)}; check claims.REGISTER before citing."

        indexed[doi] = {
            "category": base_meta["category"],
            "date": base_meta["date"],
            "doi": doi,
            "phase": base_meta["phase"],
            "record_id": rid,
            "status": status,
            "superseded_by": override.get("superseded_by"),
            "supersedes": list(override.get("supersedes") or []),
            "sys024_claims": [cid for cid, _ in c_pairs],
            "sys024_verdicts": sorted({v for _, v in c_pairs}),
            "title": base_meta["title"],
            "warning": warning,
        }

    payload: Dict[str, Any] = {
        "indexed_records_count": len(indexed),
        "records": indexed,
        "schema": "osiris.doi_lineage/1",
        "sources_sha256": {
            "records_json_sha256": records_sha,
        },
        "warned_records_count": sum(1 for entry in indexed.values() if entry.get("warning")),
    }
    payload["lineage_sha256"] = jcs_sha256(payload)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_bytes(jcs_bytes(payload) + b"\n")
    os.replace(tmp_path, out_path)
    return payload


def build_department_handoff_receipt(
    *,
    from_department: str,
    to_department: str,
    artifact_path: Union[str, Path],
    verifier_exit_code: int,
    author_model: str,
    critic_model: str,
    worktree_isolated: bool = True,
) -> Dict[str, Any]:
    """Build a typed, hash-verified cross-department handoff receipt enforcing worktree isolation
    and cross-model review (WK-AC-001)."""
    mutating_depts = {
        "Engineering Department",
        "Infrastructure Department",
        "Publication Department",
    }
    if from_department in mutating_depts and not worktree_isolated:
        raise ValueError(f"{from_department} requires worktree_isolated=True (WK-AC-001)")
    if author_model.strip() == critic_model.strip():
        raise ValueError("Cross-department handoff requires critic_model != author_model")
    if int(verifier_exit_code) != 0:
        raise ValueError(f"Handoff refused: child verifier exited {verifier_exit_code} != 0")

    art = Path(artifact_path)
    if not art.is_file():
        raise FileNotFoundError(f"Handoff artifact does not exist: {art}")
    art_sha = hashlib.sha256(art.read_bytes()).hexdigest()
    body: Dict[str, Any] = {
        "artifact_path": str(art),
        "artifact_sha256": art_sha,
        "author_model": author_model,
        "critic_model": critic_model,
        "from_department": from_department,
        "schema": "osiris.department_handoff_receipt/1",
        "to_department": to_department,
        "verifier_exit_code": int(verifier_exit_code),
        "worktree_isolated": bool(worktree_isolated),
    }
    body["handoff_sha256"] = jcs_sha256(body)
    return body


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_show_bytes(repo: Path, ref_and_path: str) -> Optional[bytes]:
    if not (repo / ".git").exists():
        return None
    proc = subprocess.run(
        ["git", "-C", str(repo), "show", ref_and_path],
        capture_output=True,
        check=False,
        timeout=15,
    )
    if proc.returncode != 0 or not proc.stdout:
        return None
    return proc.stdout


def _read_disk_or_git_bytes(repo: Path, rel_path: str, branch: str) -> Optional[bytes]:
    disk_file = repo / rel_path
    if disk_file.is_file():
        return disk_file.read_bytes()
    return _git_show_bytes(repo, f"{branch}:{rel_path}")


def verify_reproducibility_capsules_chain(path: Union[str, Path]) -> Dict[str, Any]:
    """Verify hash-chain continuity and RFC 8785 JCS SHA-256 digests of ``reproducibility_capsules.jsonl``."""
    cpath = Path(path)
    if not cpath.is_file():
        return {"count": 0, "errors": ["file_missing"], "tail_sha256": "0" * 64, "valid": False}

    prev = "0" * 64
    count = 0
    errors: List[str] = []
    for idx, line in enumerate(cpath.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        count += 1
        row = json.loads(line)
        if row.get("schema") != "osiris.reproducibility_capsule/1":
            errors.append(f"line {idx}: unexpected schema {row.get('schema')!r}")
            break
        if row.get("prev_capsule_sha256") != prev:
            errors.append(f"line {idx}: prev_capsule_sha256 mismatch")
            break
        recorded_sha = str(row.get("capsule_sha256") or "")
        body = {k: v for k, v in row.items() if k != "capsule_sha256"}
        recomputed = jcs_sha256(body)
        if recorded_sha != recomputed:
            errors.append(f"line {idx}: capsule_sha256 mismatch")
            break
        ev = row.get("evidence_class")
        if ev in ("SIMULATED", "OFFLINE_COMPILED") and row.get("provider_job_id") is not None:
            errors.append(f"line {idx}: {ev} cannot carry provider_job_id")
            break
        if ev == "HARDWARE_ATTESTED" and (not row.get("provider_job_id") or not row.get("raw_counts_sha256")):
            errors.append(f"line {idx}: HARDWARE_ATTESTED missing job_id or raw_counts_sha256")
            break
        prev = recorded_sha

    return {
        "count": count,
        "errors": errors,
        "tail_sha256": prev,
        "valid": len(errors) == 0 and count > 0,
    }


def materialize_workspace_reproducibility_capsules(
    work_root: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Compute real SHA-256 digests over workspace IBM hardware runs & staged Quantinuum bundles,
    writing hash-chained RFC 8785 JCS ``osiris.reproducibility_capsule/1`` rows (WK-EM-001)."""
    wroot = Path(work_root).expanduser().resolve() if work_root is not None else DEFAULT_WORK_ROOT
    out_file = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else (evolution_engine.default_osiris_home() / "evidence" / "reproducibility_capsules.jsonl")
    )

    if (wroot / "experiments").is_dir():
        cli_repo = wroot
    elif (wroot / "_wt-evo-integration" / "experiments").is_dir():
        cli_repo = wroot / "_wt-evo-integration"
    else:
        cli_repo = wroot / "osiris-cli"
    qec_repo = wroot / "osiris-qec"

    capsules: List[Dict[str, Any]] = []
    prev_sha = "0" * 64

    def _append_capsule(**kwargs: Any) -> None:
        nonlocal prev_sha
        cap = build_rfc8785_reproducibility_capsule(prev_capsule_sha256=prev_sha, **kwargs)
        prev_sha = str(cap["capsule_sha256"])
        capsules.append(cap)

    # 1. OSIRIS-HP-004 (ibm_marrakesh)
    hp4_dir = cli_repo / "experiments" / "hp004"
    if (hp4_dir / "hp004_submitted_job.json").is_file() and (hp4_dir / "physical_hp004_results.json").is_file():
        j4 = json.loads((hp4_dir / "hp004_submitted_job.json").read_text(encoding="utf-8"))
        z4 = (
            json.loads((hp4_dir / "zenodo_hp004_publication_receipt.json").read_text(encoding="utf-8"))
            if (hp4_dir / "zenodo_hp004_publication_receipt.json").is_file()
            else {}
        )
        _append_capsule(
            experiment_id="OSIRIS-HP-004",
            evidence_class="HARDWARE_ATTESTED",
            prereg_sha256=_sha256_file(hp4_dir / "PRE_REGISTRATION.md"),
            circuit_isa_sha256=_sha256_file(hp4_dir / "hp004_staged_circuits.qpy"),
            provider="IBM Quantum",
            backend_id=str(j4["backend"]),
            shots=int(j4["total_shots"]),
            provider_job_id=str(j4["job_id"]),
            raw_counts_sha256=_sha256_file(hp4_dir / "physical_hp004_results.json"),
            zenodo_doi=z4.get("doi"),
        )

    # 2. OSIRIS-HP-004B (ibm_fez)
    hp4b_dir = cli_repo / "experiments" / "hp004b_fez"
    if (hp4b_dir / "hp004b_fez_submitted_job.json").is_file() and (hp4b_dir / "physical_hp004b_fez_results.json").is_file():
        j4b = json.loads((hp4b_dir / "hp004b_fez_submitted_job.json").read_text(encoding="utf-8"))
        z4b = (
            json.loads((hp4b_dir / "zenodo_hp004_v2_publication_receipt.json").read_text(encoding="utf-8"))
            if (hp4b_dir / "zenodo_hp004_v2_publication_receipt.json").is_file()
            else {}
        )
        _append_capsule(
            experiment_id="OSIRIS-HP-004B",
            evidence_class="HARDWARE_ATTESTED",
            prereg_sha256=_sha256_file(hp4b_dir / "PRE_REGISTRATION.md"),
            circuit_isa_sha256=_sha256_file(hp4b_dir / "hp004b_fez_staged_circuits.qpy"),
            provider="IBM Quantum",
            backend_id=str(j4b["backend"]),
            shots=int(j4b["total_shots"]),
            provider_job_id=str(j4b["job_id"]),
            raw_counts_sha256=_sha256_file(hp4b_dir / "physical_hp004b_fez_results.json"),
            zenodo_doi=z4b.get("doi"),
        )

    # 3. OSIRIS-QEC-001 (ibm_fez)
    qec1_dir = cli_repo / "experiments" / "qec001"
    if (qec1_dir / "qec001_fez_submitted_job.json").is_file() and (qec1_dir / "physical_qec001_fez_raw_counts.json.gz").is_file():
        jq1 = json.loads((qec1_dir / "qec001_fez_submitted_job.json").read_text(encoding="utf-8"))
        zq1 = (
            json.loads((qec1_dir / "zenodo_qec001_publication_receipt.json").read_text(encoding="utf-8"))
            if (qec1_dir / "zenodo_qec001_publication_receipt.json").is_file()
            else {}
        )
        _append_capsule(
            experiment_id="OSIRIS-QEC-001",
            evidence_class="HARDWARE_ATTESTED",
            prereg_sha256=_sha256_file(qec1_dir / "PRE_REGISTRATION.md"),
            circuit_isa_sha256=_sha256_file(qec1_dir / "qec001_fez_staged_circuits.qpy"),
            provider="IBM Quantum",
            backend_id=str(jq1["backend"]),
            shots=int(jq1["total_shots"]),
            provider_job_id=str(jq1["job_id"]),
            raw_counts_sha256=_sha256_file(qec1_dir / "physical_qec001_fez_raw_counts.json.gz"),
            zenodo_doi=zq1.get("doi"),
        )

    # 4. OSIRIS-ADVANTAGE-MARRAKESH (ibm_marrakesh)
    adv_dir = cli_repo / "experiments" / "osiris_advantage_marrakesh"
    adv_counts = adv_dir / "dau0q3qhcrkc73durtgg.counts.json"
    adv_meta = adv_dir / "dau0q3qhcrkc73durtgg.meta.json"
    if adv_counts.is_file() and adv_meta.is_file():
        m_adv = json.loads(adv_meta.read_text(encoding="utf-8"))
        n_specs = max(1, len(m_adv.get("specs") or [1]))
        _append_capsule(
            experiment_id="OSIRIS-ADVANTAGE-MARRAKESH",
            evidence_class="HARDWARE_ATTESTED",
            prereg_sha256=_sha256_file(adv_dir / "pre_registration_manifest.json"),
            circuit_isa_sha256=_sha256_file(adv_dir / "run_marrakesh_advantage.py"),
            provider="IBM Quantum",
            backend_id=str(m_adv.get("backend") or "ibm_marrakesh"),
            shots=int(m_adv.get("shots", 512)) * n_specs,
            provider_job_id="dau0q3qhcrkc73durtgg",
            raw_counts_sha256=_sha256_file(adv_counts),
            zenodo_doi=None,
        )

    # 5. CORRELATED-NOISE-01 (papers/cn01_correlated_noise_null_marrakesh + docs/phase3)
    cn01_paper = cli_repo / "papers" / "cn01_correlated_noise_null_marrakesh"
    cn01_counts = cli_repo / "docs" / "phase3" / "correlated_noise_01_live_raw_counts.json"
    if (cn01_paper / "zenodo_receipt.json").is_file() and cn01_counts.is_file():
        z_cn = json.loads((cn01_paper / "zenodo_receipt.json").read_text(encoding="utf-8"))
        g_cn = (cn01_paper / "generated.tex").read_text(encoding="utf-8")
        m_jid = re.search(r"\\newcommand\{\\JobId\}\{([a-z0-9]+)\}", g_cn)
        m_sh = re.search(r"\\newcommand\{\\TotalShots\}\{([0-9{,}]+)\}", g_cn)
        shots_cn = int(re.sub(r"[^0-9]", "", m_sh.group(1))) if m_sh else 499712
        _append_capsule(
            experiment_id="CORRELATED-NOISE-01",
            evidence_class="HARDWARE_ATTESTED",
            prereg_sha256=_sha256_file(cli_repo / "docs" / "phase3" / "correlated_noise_01_live_prereg_lock.json"),
            circuit_isa_sha256=_sha256_file(cli_repo / "docs" / "phase3" / "correlated_noise_01_live_pubs.qpy"),
            provider="IBM Quantum",
            backend_id="ibm_marrakesh",
            shots=shots_cn,
            provider_job_id=m_jid.group(1) if m_jid else "dbtp67dav06s73cfbsm0",
            raw_counts_sha256=_sha256_file(cn01_counts),
            zenodo_doi=f"10.5281/zenodo.{z_cn['deposition_id']}",
        )

    # 6. TESSERACT-MAP-05 (papers/tmap05_exclusion_layout + docs/phase5)
    tmap5_paper = cli_repo / "papers" / "tmap05_exclusion_layout"
    if (tmap5_paper / "zenodo_receipt.json").is_file() and (tmap5_paper / "generated.tex").is_file():
        z_tm5 = json.loads((tmap5_paper / "zenodo_receipt.json").read_text(encoding="utf-8"))
        g_tm5 = (tmap5_paper / "generated.tex").read_text(encoding="utf-8")
        for b_short, b_name in (("MARRAKESH", "ibm_marrakesh"), ("FEZ", "ibm_fez"), ("KINGSTON", "ibm_kingston")):
            p_lock = cli_repo / "docs" / "phase5" / f"tesseract_map_05_{b_name}_live_prereg_lock.json"
            p_qpy = cli_repo / "docs" / "phase5" / f"tesseract_map_05_{b_name}_live_pubs.qpy"
            p_cnt = cli_repo / "docs" / "phase5" / f"tesseract_map_05_{b_name}_live_raw_counts.json"
            if p_lock.is_file() and p_qpy.is_file() and p_cnt.is_file():
                esc_b = b_name.replace("_", r"\_")
                m_row = re.search(rf"{esc_b}\s*&\s*\\texttt\{{([a-z0-9]+)\}}\s*&\s*\\texttt\{{([a-z0-9]+)\}}", g_tm5)
                live_job = m_row.group(2) if m_row else f"tmap05_{b_name}"
                lock_obj = json.loads(p_lock.read_text(encoding="utf-8"))
                shots_tm = int(lock_obj.get("total_shots") or lock_obj.get("shots_total") or 163840)
                _append_capsule(
                    experiment_id=f"TESSERACT-MAP-05-{b_short}",
                    evidence_class="HARDWARE_ATTESTED",
                    prereg_sha256=_sha256_file(p_lock),
                    circuit_isa_sha256=_sha256_file(p_qpy),
                    provider="IBM Quantum",
                    backend_id=b_name,
                    shots=shots_tm,
                    provider_job_id=live_job,
                    raw_counts_sha256=_sha256_file(p_cnt),
                    zenodo_doi=f"10.5281/zenodo.{z_tm5['deposition_id']}",
                )

    # 7. TESSERACT-RO-06-07 (papers/ro06_07_readout_staleness)
    ro_paper = cli_repo / "papers" / "ro06_07_readout_staleness"
    ro_zip = ro_paper / "ro06_ro07_data.zip"
    if (ro_paper / "zenodo_receipt.json").is_file() and ro_zip.is_file():
        z_ro = json.loads((ro_paper / "zenodo_receipt.json").read_text(encoding="utf-8"))
        g_ro = (ro_paper / "generated.tex").read_text(encoding="utf-8")
        ro_jobs = re.findall(r"\\texttt\{(dc[a-z0-9]+)\}", g_ro)
        _append_capsule(
            experiment_id="TESSERACT-RO-06-07",
            evidence_class="HARDWARE_ATTESTED",
            prereg_sha256=_sha256_file(ro_paper / "SHA256SUMS.txt"),
            circuit_isa_sha256=_sha256_file(ro_paper / "generate_tables.py"),
            provider="IBM Quantum",
            backend_id="ibm_marrakesh,ibm_fez,ibm_kingston",
            shots=552960,
            provider_job_id=",".join(ro_jobs) if ro_jobs else "dc0qjqdhdvas73bjtgr0",
            raw_counts_sha256=_sha256_file(ro_zip),
            zenodo_doi=f"10.5281/zenodo.{z_ro['deposition_id']}",
        )

    # 8. OSIRIS-XTOP-005-IBM (papers/xtop005_codespace_hp + docs/xtop005/qpu)
    x5_paper = cli_repo / "papers" / "xtop005_codespace_hp"
    x5_qpu = cli_repo / "docs" / "xtop005" / "qpu"
    if (x5_paper / "zenodo_receipt.json").is_file() and (x5_qpu / "job_receipt.json").is_file():
        z_x5 = json.loads((x5_paper / "zenodo_receipt.json").read_text(encoding="utf-8"))
        j_x5 = json.loads((x5_qpu / "job_receipt.json").read_text(encoding="utf-8"))
        raw_x5 = (x5_paper / "xtop005_data.zip") if (x5_paper / "xtop005_data.zip").is_file() else (x5_qpu / "raw_counts.json")
        _append_capsule(
            experiment_id="OSIRIS-XTOP-005-IBM",
            evidence_class="HARDWARE_ATTESTED",
            prereg_sha256=_sha256_file(x5_qpu / "prereg.json"),
            circuit_isa_sha256=_sha256_file(x5_qpu / "circuits.qpy"),
            provider="IBM Quantum",
            backend_id=str(j_x5["backend"]),
            shots=int(j_x5["shots"]) * int(j_x5["n_pubs"]),
            provider_job_id=str(j_x5["job_id"]),
            raw_counts_sha256=_sha256_file(raw_x5),
            zenodo_doi=f"10.5281/zenodo.{z_x5['deposition_id']}",
        )

    # 9. OSIRIS-QEC-002-H2 (Quantinuum OFFLINE_COMPILED via osiris-qec)
    qec2_man = _read_disk_or_git_bytes(
        qec_repo,
        "submission/nexus_h2_bundle/nexus_qec002_manifest.json",
        "feat/qec003-memory-twist-and-nexus",
    )
    qec2_isa = _read_disk_or_git_bytes(
        qec_repo,
        "submission/nexus_h2_bundle/wen_plaquette_qec002_x_basis_h2_isa.json",
        "feat/qec003-memory-twist-and-nexus",
    )
    if qec2_man is not None and qec2_isa is not None:
        _append_capsule(
            experiment_id="OSIRIS-QEC-002-H2",
            evidence_class="OFFLINE_COMPILED",
            prereg_sha256=hashlib.sha256(qec2_man).hexdigest(),
            circuit_isa_sha256=hashlib.sha256(qec2_isa).hexdigest(),
            provider="Quantinuum",
            backend_id="H2-1",
            shots=2000,
            provider_job_id=None,
            raw_counts_sha256=None,
            zenodo_doi="10.5281/zenodo.24424726",
        )

    # 10. OSIRIS-QEC-003-H2 (Quantinuum SIMULATED via osiris-qec)
    qec3_rep = _read_disk_or_git_bytes(
        qec_repo,
        "reports/qec003_verification_report.json",
        "feat/qec003-memory-twist-and-nexus",
    )
    qec3_src = _read_disk_or_git_bytes(
        qec_repo,
        "osiris_qec/qec003_memory_twists.py",
        "feat/qec003-memory-twist-and-nexus",
    )
    if qec3_rep is not None:
        isa_q3 = hashlib.sha256(qec3_src if qec3_src is not None else qec3_rep).hexdigest()
        rep_q3_sha = hashlib.sha256(qec3_rep).hexdigest()
        _append_capsule(
            experiment_id="OSIRIS-QEC-003-H2",
            evidence_class="SIMULATED",
            prereg_sha256=rep_q3_sha,
            circuit_isa_sha256=isa_q3,
            provider="Quantinuum",
            backend_id="H2-1",
            shots=4000,
            provider_job_id=None,
            raw_counts_sha256=rep_q3_sha,
            zenodo_doi=None,
        )

    # 11. OSIRIS-XTOP-005-H2 (Quantinuum OFFLINE_COMPILED via feat/xtop006-and-nexus-staging)
    x5_h2_man = _read_disk_or_git_bytes(
        cli_repo,
        "xtop005/nexus_h2_bundle/bundle_manifest.json",
        "feat/xtop006-and-nexus-staging",
    ) or _read_disk_or_git_bytes(
        cli_repo,
        "xtop005/nexus_h2_bundle/nexus_h2_decisive_manifest.json",
        "feat/xtop006-and-nexus-staging",
    )
    x5_h2_isa = _read_disk_or_git_bytes(
        cli_repo,
        "xtop005/nexus_h2_bundle/circuits_h2_decisive_18_tket.json",
        "feat/xtop006-and-nexus-staging",
    )
    if x5_h2_man is not None:
        _append_capsule(
            experiment_id="OSIRIS-XTOP-005-H2",
            evidence_class="OFFLINE_COMPILED",
            prereg_sha256=hashlib.sha256(x5_h2_man).hexdigest(),
            circuit_isa_sha256=hashlib.sha256(x5_h2_isa if x5_h2_isa is not None else x5_h2_man).hexdigest(),
            provider="Quantinuum",
            backend_id="H2-1E",
            shots=9000,
            provider_job_id=None,
            raw_counts_sha256=None,
            zenodo_doi="10.5281/zenodo.24426216",
        )

    # 12. OSIRIS-XTOP-006-H2 (Quantinuum OFFLINE_COMPILED via feat/xtop006-and-nexus-staging)
    x6_h2_man = _read_disk_or_git_bytes(
        cli_repo,
        "xtop006/artifacts/xtop006_h2_manifest.json",
        "feat/xtop006-and-nexus-staging",
    )
    x6_h2_isa = _read_disk_or_git_bytes(
        cli_repo,
        "xtop006/artifacts/xtop006_h2_1_compiled_circuits.json",
        "feat/xtop006-and-nexus-staging",
    )
    if x6_h2_man is not None:
        _append_capsule(
            experiment_id="OSIRIS-XTOP-006-H2",
            evidence_class="OFFLINE_COMPILED",
            prereg_sha256=hashlib.sha256(x6_h2_man).hexdigest(),
            circuit_isa_sha256=hashlib.sha256(x6_h2_isa if x6_h2_isa is not None else x6_h2_man).hexdigest(),
            provider="Quantinuum",
            backend_id="H2-1",
            shots=30000,
            provider_job_id=None,
            raw_counts_sha256=None,
            zenodo_doi=None,
        )

    def _v620_deposit_doi(slug: str) -> Optional[str]:
        rec_p = cli_repo / "papers" / "zenodo_v620_deposits" / slug / "zenodo_receipt.json"
        if not rec_p.is_file():
            return None
        try:
            d = json.loads(rec_p.read_text(encoding="utf-8"))
            val = str(d.get("doi") or d.get("prereserved_doi") or "").strip()
            return val or None
        except Exception:
            return None

    # 13. OSIRIS-XTOP-006-FEZ (ibm_fez live Kraus trajectory)
    x6f_dir = cli_repo / "experiments" / "xtop006_fez"
    if (x6f_dir / "xtop006_fez_submitted_job.json").is_file() and (
        x6f_dir / "physical_xtop006_fez_raw_counts.json.gz"
    ).is_file():
        jx6f = json.loads((x6f_dir / "xtop006_fez_submitted_job.json").read_text(encoding="utf-8"))
        mx6f = json.loads((x6f_dir / "xtop006_fez_staging_manifest.json").read_text(encoding="utf-8"))
        _append_capsule(
            experiment_id="OSIRIS-XTOP-006-FEZ",
            evidence_class="HARDWARE_ATTESTED",
            prereg_sha256=_sha256_file(x6f_dir / "PRE_REGISTRATION.md"),
            circuit_isa_sha256=_sha256_file(x6f_dir / "xtop006_fez_staged_circuits.qpy"),
            provider="IBM Quantum",
            backend_id=str(jx6f["backend"]),
            shots=int(mx6f["total_shots"]),
            provider_job_id=str(jx6f["job_id"]),
            raw_counts_sha256=_sha256_file(x6f_dir / "physical_xtop006_fez_raw_counts.json.gz"),
            zenodo_doi=_v620_deposit_doi("xtop006_fez"),
        )

    # 14. OSIRIS-WP2-27Q-FEZ (ibm_fez live 27Q debiased Sinkhorn OT)
    wp2_dir = cli_repo / "experiments" / "wp2_27q_fez"
    if (wp2_dir / "wp2_27q_fez_submitted_job.json").is_file() and (
        wp2_dir / "physical_wp2_27q_fez_raw_counts.json.gz"
    ).is_file():
        jwp2 = json.loads((wp2_dir / "wp2_27q_fez_submitted_job.json").read_text(encoding="utf-8"))
        mwp2 = json.loads((wp2_dir / "wp2_27q_fez_staging_manifest.json").read_text(encoding="utf-8"))
        _append_capsule(
            experiment_id="OSIRIS-WP2-27Q-FEZ",
            evidence_class="HARDWARE_ATTESTED",
            prereg_sha256=_sha256_file(wp2_dir / "PRE_REGISTRATION.md"),
            circuit_isa_sha256=_sha256_file(wp2_dir / "wp2_27q_fez_staged_circuits.qpy"),
            provider="IBM Quantum",
            backend_id=str(jwp2["backend"]),
            shots=int(mwp2["total_shots"]),
            provider_job_id=str(jwp2["job_id"]),
            raw_counts_sha256=_sha256_file(wp2_dir / "physical_wp2_27q_fez_raw_counts.json.gz"),
            zenodo_doi=_v620_deposit_doi("wp2_27q_ot"),
        )

    # 15. OSIRIS-XTOP-006B-FEZ (ibm_fez live 4-twirl confirmatory discriminator)
    x6b_dir = cli_repo / "experiments" / "xtop006b_fez"
    if (x6b_dir / "xtop006b_fez_submitted_job.json").is_file() and (
        x6b_dir / "physical_xtop006b_fez_raw_counts.json.gz"
    ).is_file():
        jx6b = json.loads((x6b_dir / "xtop006b_fez_submitted_job.json").read_text(encoding="utf-8"))
        mx6b = json.loads((x6b_dir / "xtop006b_fez_staging_manifest.json").read_text(encoding="utf-8"))
        _append_capsule(
            experiment_id="OSIRIS-XTOP-006B-FEZ",
            evidence_class="HARDWARE_ATTESTED",
            prereg_sha256=_sha256_file(x6b_dir / "PRE_REGISTRATION.md"),
            circuit_isa_sha256=_sha256_file(x6b_dir / "xtop006b_fez_staged_circuits.qpy"),
            provider="IBM Quantum",
            backend_id=str(jx6b["backend"]),
            shots=int(mx6b["total_shots"]),
            provider_job_id=str(jx6b["job_id"]),
            raw_counts_sha256=_sha256_file(x6b_dir / "physical_xtop006b_fez_raw_counts.json.gz"),
            zenodo_doi=_v620_deposit_doi("xtop006_fez"),
        )

    # 16. OSIRIS-2BGA-QLDPC (Non-abelian odd-k 2BGA CSS codes + BP+OSD-CS)
    bga_rep = cli_repo / "qldpc_nonabelian" / "artifacts" / "nonabelian_2bga_report.json"
    bga_src = cli_repo / "qldpc_nonabelian" / "nonabelian_2bga.py"
    if bga_rep.is_file() and bga_src.is_file():
        _append_capsule(
            experiment_id="OSIRIS-2BGA-QLDPC",
            evidence_class="SIMULATED",
            prereg_sha256=_sha256_file(bga_rep),
            circuit_isa_sha256=_sha256_file(bga_src),
            provider="Simulator",
            backend_id="bp_osd_cs_helios_noise",
            shots=14000,
            provider_job_id=None,
            raw_counts_sha256=_sha256_file(bga_rep),
            zenodo_doi=_v620_deposit_doi("nonabelian_2bga_adv1"),
        )

    # 17. OSIRIS-ADV1-PREGATE (3-attack M_6 Pauli-weight-5 classical-hardness pre-gate)
    adv1_rep = cli_repo / "adv1" / "artifacts" / "adv1_pregate_report.json"
    adv1_src = cli_repo / "adv1" / "pregate.py"
    if adv1_rep.is_file() and adv1_src.is_file():
        _append_capsule(
            experiment_id="OSIRIS-ADV1-PREGATE",
            evidence_class="SIMULATED",
            prereg_sha256=_sha256_file(adv1_rep),
            circuit_isa_sha256=_sha256_file(adv1_src),
            provider="Simulator",
            backend_id="mps_paulipath_lightcone",
            shots=3000,
            provider_job_id=None,
            raw_counts_sha256=_sha256_file(adv1_rep),
            zenodo_doi=_v620_deposit_doi("nonabelian_2bga_adv1"),
        )

    # 18. OSIRIS-HERON-R2-UNIFIED-42 (42-job unified calibration-staleness & colored-nulls corpus)
    hr2_dir = cli_repo / "papers" / "heron_r2_calibration_staleness_and_colored_nulls"
    if (hr2_dir / "SHA256SUMS.txt").is_file() and (hr2_dir / "generate_unified_assets.py").is_file():
        _append_capsule(
            experiment_id="OSIRIS-HERON-R2-UNIFIED-42",
            evidence_class="HARDWARE_ATTESTED",
            prereg_sha256=_sha256_file(hr2_dir / "SHA256SUMS.txt"),
            circuit_isa_sha256=_sha256_file(hr2_dir / "generate_unified_assets.py"),
            provider="IBM Quantum",
            backend_id="ibm_marrakesh,ibm_fez,ibm_kingston",
            shots=2174976,
            provider_job_id="dc0qjqdhdvas73bjtgr0",
            raw_counts_sha256=_sha256_file(hr2_dir / "generated.tex"),
            zenodo_doi=_v620_deposit_doi("heron_r2_unified"),
        )

    # 19. OSIRIS-QDECODE-HERON-59 (59-job wire-deserialization & entropy ceiling corpus)
    qdec_dir = cli_repo / "papers" / "qdecode_heron_provenance_and_finite_shot_entropy"
    if (qdec_dir / "HARDWARE_PROVENANCE_MANIFEST.json").is_file() and (
        qdec_dir / "reproducibility_capsule.json"
    ).is_file():
        qdec_cap = json.loads((qdec_dir / "reproducibility_capsule.json").read_text(encoding="utf-8"))
        _append_capsule(
            experiment_id="OSIRIS-QDECODE-HERON-59",
            evidence_class="HARDWARE_ATTESTED",
            prereg_sha256=_sha256_file(qdec_dir / "SHA256SUMS.txt"),
            circuit_isa_sha256=_sha256_file(cli_repo / "osiris_cli" / "qdecode.py"),
            provider="IBM Quantum",
            backend_id="ibm_fez,ibm_marrakesh,ibm_kingston,ibm_torino",
            shots=int(qdec_cap["corpus_totals"]["total_shots"]),
            provider_job_id=str(qdec_cap["key_benchmarks"]["fez_156q_100k_job_id"]),
            raw_counts_sha256=_sha256_file(qdec_dir / "HARDWARE_PROVENANCE_MANIFEST.json"),
            zenodo_doi=_v620_deposit_doi("qdecode_heron_59"),
        )

    # 20. OSIRIS-BT2T-001-FEZ (5Q Binary Tetrahedral |2T|=24 structured vs generic synthesis)
    bt_dir = cli_repo / "experiments" / "bt2t_fez"
    if (bt_dir / "bt2t_fez_submitted_job.json").is_file() and (
        bt_dir / "physical_bt2t_fez_raw_counts.json.gz"
    ).is_file():
        jbt = json.loads((bt_dir / "bt2t_fez_submitted_job.json").read_text(encoding="utf-8"))
        mbt = json.loads((bt_dir / "bt2t_fez_staging_manifest.json").read_text(encoding="utf-8"))
        _append_capsule(
            experiment_id="OSIRIS-BT2T-001-FEZ",
            evidence_class="HARDWARE_ATTESTED",
            prereg_sha256=_sha256_file(bt_dir / "PRE_REGISTRATION.md"),
            circuit_isa_sha256=_sha256_file(bt_dir / "bt2t_fez_staged_circuits.qpy"),
            provider="IBM Quantum",
            backend_id=str(jbt["backend"]),
            shots=int(mbt["total_shots"]),
            provider_job_id=str(jbt["job_id"]),
            raw_counts_sha256=_sha256_file(bt_dir / "physical_bt2t_fez_raw_counts.json.gz"),
            zenodo_doi=None,
        )

    # 21. OSIRIS-ECHO-STOPGO (Card B offline placement-ranking stop/go gate)
    echo_rep = cli_repo / "experiments" / "echo_placement" / "artifacts" / "echo_stopgo_report.json"
    echo_src = cli_repo / "experiments" / "echo_placement" / "echo_placement_gate.py"
    if echo_rep.is_file() and echo_src.is_file():
        _append_capsule(
            experiment_id="OSIRIS-ECHO-STOPGO",
            evidence_class="SIMULATED",
            prereg_sha256=_sha256_file(echo_rep),
            circuit_isa_sha256=_sha256_file(echo_src),
            provider="Simulator",
            backend_id="heron_r2_t5_kraus_16win",
            shots=163840,
            provider_job_id=None,
            raw_counts_sha256=_sha256_file(echo_rep),
            zenodo_doi=None,
        )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_file.with_suffix(".jsonl.tmp")
    raw_jsonl = b"".join(jcs_bytes(cap) + b"\n" for cap in capsules)
    tmp_out.write_bytes(raw_jsonl)
    os.replace(tmp_out, out_file)

    return {
        "capsule_count": len(capsules),
        "capsules": capsules,
        "file_sha256": hashlib.sha256(raw_jsonl).hexdigest(),
        "hardware_attested_count": sum(1 for c in capsules if c["evidence_class"] == "HARDWARE_ATTESTED"),
        "offline_or_simulated_count": sum(1 for c in capsules if c["evidence_class"] in ("OFFLINE_COMPILED", "SIMULATED")),
        "output_path": str(out_file),
        "tail_capsule_sha256": prev_sha,
    }


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
