"""
Recursive DNA-Lang Governed Evolution Loop
==========================================
Executes the 12-step recursive engineering loop:
  1. Capture user intent
  2. Normalize into typed task specification
  3. Generate candidate .dna genome (parent-derived mutation)
  4. Parse and validate genome
  5. Reject unauthorized capabilities before execution (SYS-025)
  6. Execute within approved sandbox/runtime
  7. Evaluate against deterministic acceptance criteria & emit EvaluationReceipt
  8. Record performance, resource usage, failures, provenance
  9. Compare with frozen parent and baseline
  10. Retain candidate only if predefined promotion criteria met (no ties, no NaN)
  11. Record lineage and reproducible rollback reference into hash-chained ledger
  12. Propose next improvement from observed capability gap
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from osiris.compiler.converter import OrganismConverter
from osiris.compiler.dna_ir import QuantumCircuitIR
from osiris_cli.providers import BaseModelProvider, get_active_provider
from .receipts import EvaluationReceipt, PromotionPolicy
from .falsification import AdversarialFalsificationEngine, FalsificationReceipt


@dataclass
class GenomeCandidate:
    candidate_id: str
    parent_id: Optional[str]
    genome_source: str  # .dna source code
    parsed_ir: Optional[QuantumCircuitIR] = None
    fitness_score: float = 0.0
    passed_acceptance: bool = False
    rejection_reason: str = ""
    resource_usage_ms: float = 0.0
    provenance: str = "dnalang_evolution_loop"
    evaluation_receipt: Optional[EvaluationReceipt] = None
    falsification_receipt: Optional[FalsificationReceipt] = None
    created_at: float = field(default_factory=time.time)

    def content_digest(self) -> str:
        return hashlib.sha256(self.genome_source.encode("utf-8")).hexdigest()


@dataclass
class LineageRecord:
    generation: int
    candidate_id: str
    parent_id: Optional[str]
    digest: str
    fitness: float
    promoted: bool
    rollback_ref: str
    receipt_digest: str = ""
    timestamp: float = field(default_factory=time.time)


class CandidateEvolutionLoop:
    """Governed 12-step recursive DNA-Lang improvement pipeline."""

    ALLOWED_CAPABILITIES = {"H", "CX", "RZ", "X", "Y", "Z", "MEASURE"}

    def __init__(
        self,
        provider: Optional[BaseModelProvider] = None,
        converter: Optional[OrganismConverter] = None,
        lineage_ledger_path: Optional[str] = None,
        promotion_policy: Optional[PromotionPolicy] = None,
        falsification_engine: Optional[AdversarialFalsificationEngine] = None,
    ):
        self.provider = provider or get_active_provider()
        self.converter = converter or OrganismConverter()
        self.lineage_ledger_path = lineage_ledger_path or os.path.expanduser("~/.osiris/dnalang_lineage.jsonl")
        self.promotion_policy = promotion_policy or PromotionPolicy()
        self.falsification_engine = falsification_engine
        self.generation_count = 0
        self.current_champion: Optional[GenomeCandidate] = None
        self.current_champion_receipt: Optional[EvaluationReceipt] = None
        self.last_ledger_hash: str = "GENESIS_ROOT"

    def mutate_parent_genome(self, parent_source: str, gen_num: int, intent_text: str) -> str:
        """
        Derive a new candidate genome directly from parent source code.
        Mutates operators while preserving parent lineage.
        """
        match = re.search(r"quantum_state\s*\{(.*?)\}", parent_source, re.DOTALL)
        if match:
            inner_body = match.group(1).strip()
            statements = [s.strip() for s in inner_body.split(";") if s.strip()]
            
            # Apply lineage-aware mutation based on intent or sequence expansion
            intent_lower = intent_text.lower()
            if "rz" in intent_lower and not any("rz" in s for s in statements):
                statements.append("rz 1")
            elif "phase" in intent_lower and not any("rz" in s for s in statements):
                statements.append("rz 1")
            elif "depth" in intent_lower or "optimize" in intent_lower:
                if not any("rz" in s for s in statements):
                    statements.append("rz 1")
                elif not any("x" in s for s in statements):
                    statements.append("x 0")
                elif not any("h 1" in s for s in statements):
                    statements.append("h 1")
                else:
                    statements.append("cx 1, 0")
            else:
                # Default parent mutation
                if not any("rz" in s for s in statements):
                    statements.append("rz 1")
                elif not any("x" in s for s in statements):
                    statements.append("x 0")
                else:
                    statements.append("h 1")

            mutated_body = "; ".join(statements) + ";"
            return f"organism Gen_{gen_num} {{ quantum_state {{ {mutated_body} }} }}"

        return f"organism Gen_{gen_num} {{ quantum_state {{ h 0; cx 0, 1; rz 1; }} }}"

    def parse_and_validate(self, source: str, strict: bool = False
                           ) -> Tuple[bool, Optional[QuantumCircuitIR], str]:
        """Steps 4 & 5: Parse genome and enforce capability allowlist.

        strict=True compiles with the converter's strict mode so the allowlist is checked against
        the genome's own operations. run_cycle always uses it; the default keeps the legacy
        behavior (bare organism names, skipped bodies) for existing callers."""
        try:
            circuit = self.converter.compile(source, strict=True) if strict \
                else self.converter.compile(source)
            # Capability enforcement
            for op in circuit.operations:
                op_name = op.op_type.name
                if op_name not in self.ALLOWED_CAPABILITIES:
                    return False, None, f"Unauthorized capability '{op_name}' rejected before execution"
            return True, circuit, "Valid genome"
        except Exception as e:
            return False, None, f"DNA syntax/validation failure: {e}"

    def run_cycle(
        self,
        intent_text: str,
        acceptance_fn: Callable[
            [QuantumCircuitIR],
            Union[Tuple[bool, float], Tuple[bool, float, float], EvaluationReceipt]
        ],
        parent_candidate: Optional[GenomeCandidate] = None,
    ) -> Dict[str, Any]:
        """Execute one complete 12-step cycle with proof-carrying evolution."""
        # 1. Capture intent & 2. Normalize
        normalized_spec = {
            "intent": intent_text.strip(),
            "target_system": "quantum_dna",
            "cycle_id": str(uuid.uuid4())[:8],
        }

        # 3. Generate candidate .dna from parent lineage
        parent = parent_candidate or self.current_champion
        parent_id = parent.candidate_id if parent else None
        baseline_fitness = self.current_champion_receipt.fitness_score if self.current_champion_receipt else 0.0

        gen_num = self.generation_count + 1
        candidate_id = f"cand_{gen_num}_{normalized_spec['cycle_id']}"

        if parent:
            dna_source = self.mutate_parent_genome(parent.genome_source, gen_num, intent_text)
        else:
            dna_source = f"organism Gen_{gen_num} {{ quantum_state {{ h 0; cx 0, 1; }} }}"

        candidate = GenomeCandidate(
            candidate_id=candidate_id,
            parent_id=parent_id,
            genome_source=dna_source,
        )

        t0 = time.monotonic()

        # 4 & 5. Parse and validate against capability policy
        is_valid, ir, val_reason = self.parse_and_validate(candidate.genome_source, strict=True)
        if not is_valid:
            candidate.rejection_reason = val_reason
            candidate.passed_acceptance = False
            return {
                "candidate": candidate,
                "promoted": False,
                "reason": val_reason,
                "step": "validation",
                "generation": self.generation_count,
            }

        candidate.parsed_ir = ir

        # 6 & 7. Sandboxed evaluation against deterministic acceptance criteria
        eval_result = acceptance_fn(ir)
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        if isinstance(eval_result, EvaluationReceipt):
            receipt = eval_result
            passed = receipt.passed
            fitness = receipt.fitness_score
        elif isinstance(eval_result, tuple):
            if len(eval_result) == 2:
                passed, fitness = eval_result
                uncertainty = 0.0
            elif len(eval_result) == 3:
                passed, fitness, uncertainty = eval_result
            else:
                passed, fitness, uncertainty = eval_result[0], eval_result[1], 0.0

            receipt = EvaluationReceipt(
                candidate_id=candidate.candidate_id,
                candidate_digest=candidate.content_digest(),
                parent_id=parent_id,
                evaluator_id="deterministic_acceptance_fn",
                evaluator_version="1.0.0",
                metric_name="fitness",
                fitness_score=fitness,
                uncertainty=uncertainty,
                passed=passed,
                input_artifacts_hash=normalized_spec["cycle_id"],
            )
        else:
            passed = False
            fitness = 0.0
            receipt = EvaluationReceipt(
                candidate_id=candidate.candidate_id,
                candidate_digest=candidate.content_digest(),
                parent_id=parent_id,
                evaluator_id="deterministic_acceptance_fn",
                evaluator_version="1.0.0",
                metric_name="fitness",
                fitness_score=0.0,
                passed=False,
            )

        candidate.resource_usage_ms = elapsed_ms
        candidate.fitness_score = fitness
        candidate.passed_acceptance = passed
        candidate.evaluation_receipt = receipt

        # Adversarial Falsification Firewall (SYS-024)
        falsification_passed = True
        falsif_reason = ""
        if self.falsification_engine is not None and passed:
            falsif_receipt = self.falsification_engine.evaluate_candidate(
                candidate_id=candidate.candidate_id,
                candidate_source=candidate.genome_source,
                candidate_metric=fitness,
            )
            candidate.falsification_receipt = falsif_receipt
            if not falsif_receipt.all_controls_passed:
                falsification_passed = False
                falsif_reason = falsif_receipt.quarantine_notice or "Falsified by null control battery"

        # 8, 9 & 10. Compare with parent / baseline under PromotionPolicy (no ties, no NaN)
        if falsification_passed:
            promoted, promo_reason = self.promotion_policy.evaluate_promotion(
                candidate_receipt=receipt,
                champion_receipt=self.current_champion_receipt,
                expected_candidate_digest=candidate.content_digest(),
            )
        else:
            promoted = False
            promo_reason = falsif_reason

        rollback_ref = parent_id or "GENESIS"
        next_gap = ""

        if promoted:
            self.current_champion = candidate
            self.current_champion_receipt = receipt
            self.generation_count += 1
            next_gap = f"Target {candidate_id}: Optimize circuit depth beyond {ir.depth}"
            candidate.rejection_reason = ""
        else:
            candidate.rejection_reason = promo_reason
            next_gap = f"Gap observed: failed to achieve fitness > {baseline_fitness:.4f} ({promo_reason})"

        # 11. Record lineage & reproducible rollback reference into hash-chained ledger
        record = LineageRecord(
            generation=self.generation_count,
            candidate_id=candidate.candidate_id,
            parent_id=parent_id,
            digest=candidate.content_digest(),
            fitness=fitness,
            promoted=promoted,
            rollback_ref=rollback_ref,
            receipt_digest=receipt.receipt_digest,
        )
        self._record_lineage(record, receipt)

        return {
            "candidate": candidate,
            "promoted": promoted,
            "baseline_fitness": baseline_fitness,
            "new_fitness": fitness,
            "promo_reason": promo_reason,
            "receipt": receipt,
            "falsification_receipt": candidate.falsification_receipt,
            "next_gap": next_gap,
            "rollback_ref": rollback_ref,
            "generation": self.generation_count,
        }

    def _record_lineage(self, record: LineageRecord, receipt: EvaluationReceipt) -> None:
        """Append to tamper-evident hash-chained lineage ledger."""
        os.makedirs(os.path.dirname(self.lineage_ledger_path), exist_ok=True)
        
        block = {
            "generation": record.generation,
            "candidate_id": record.candidate_id,
            "parent_id": record.parent_id,
            "digest": record.digest,
            "fitness": record.fitness,
            "promoted": record.promoted,
            "rollback_ref": record.rollback_ref,
            "receipt_digest": record.receipt_digest,
            "timestamp": record.timestamp,
            "prev_ledger_hash": self.last_ledger_hash,
        }
        
        block_bytes = json.dumps(block, sort_keys=True, separators=(",", ":")).encode("utf-8")
        current_hash = hashlib.sha256(block_bytes).hexdigest()
        block["block_hash"] = current_hash
        self.last_ledger_hash = current_hash

        with open(self.lineage_ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(block) + "\n")
