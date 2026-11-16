"""DNA-Lang source → compile → governed simulator run → OSIRIS evaluation → governance verification → receipt.

Every stage calls the module that owns it; nothing here fabricates output:

  parse / check / lower   dnalang (dnalang-core)
  admission               osiris_governance.execution_gate.ExecutionGate, mode SIMULATION
  simulation              dnalang.backends.aer.run (qiskit-aer AerSimulator, noiseless)
  evaluation (OSIRIS)     exact distribution from qiskit Statevector; TVD with a sampling bound
                          derived by multinomial resampling of the exact distribution; classical
                          fidelity with a bootstrap CI; claimed metrics computed here
  verification            osiris_governance.ExternalVerifier recomputes every claimed number from the
                          raw counts, sealed with evidence_class SIMULATED
  receipt                 osiris_cli.trio_authz.Receipts, an append-only SHA-256 chain

Results are always labelled SIMULATED. Nothing in this module talks to quantum hardware.
"""

from __future__ import annotations

import math
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np

SCHEMA = "osiris-slice-result/v1"
DEFAULT_RECEIPTS_PATH = os.path.join(os.path.expanduser("~"), ".osiris", "receipts", "vertical_slice.jsonl")
MAX_SOURCE_BYTES = 64 * 1024
MAX_SHOTS = 100_000
MAX_QUBITS = 16
MAX_OPS = 5_000       # circuit operations admitted to the simulator
MAX_DEPTH = 2_000     # circuit depth admitted to the simulator
TVD_RESAMPLES = 400
BOOTSTRAP_RESAMPLES = 1000
_receipt_lock = threading.Lock()


class SliceInputError(ValueError):
    pass


def _sha256(data: bytes) -> str:
    import hashlib
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _stage(stages: List[Dict[str, Any]], name: str):
    class _S:
        def __enter__(self_):
            self_.t0 = time.perf_counter()
            self_.rec = {"name": name, "status": "OK"}
            stages.append(self_.rec)
            return self_.rec

        def __exit__(self_, et, ev, tb):
            self_.rec["ms"] = round((time.perf_counter() - self_.t0) * 1000, 2)
            if et is not None:
                self_.rec["status"] = "FAILED"
                self_.rec["error"] = f"{et.__name__}: {ev}"
            return False
    return _S()


def compile_source(source: str) -> Dict[str, Any]:
    """Parse, check and lower DNA-Lang source. Never raises for bad source: errors come back as diagnostics."""
    import dnalang
    from dnalang.lexer import LexError
    from dnalang.parser import ParseError
    from dnalang.sema import SemaError

    if not isinstance(source, str):
        raise SliceInputError("source must be a string")
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise SliceInputError(f"source exceeds {MAX_SOURCE_BYTES} bytes")
    out: Dict[str, Any] = {"ok": False, "source_sha256": _sha256(source.encode("utf-8")),
                           "compiler": {"name": "dnalang", "version": dnalang.__version__},
                           "diagnostics": {"errors": [], "warnings": [], "notes": []}}
    try:
        org = dnalang.parse(source)
    except (LexError, ParseError) as e:
        out["diagnostics"]["errors"].append({"stage": "parse", "message": str(e), **_pos(str(e))})
        return out
    out["organism"] = {"name": org.name, "circuit_genes": len(org.genes), "rule_genes": len(org.rules),
                       "regulator_genes": len(org.regulators), "instances": len(org.genome)}
    # Say which parts of the source cannot change the compiled circuit, so that a field such as
    # `habitat: "some_backend"` is never mistaken for a setting that was applied.
    ignored = [f"meta.{k}" for k in (org.meta or {})] + [f"{name}.*" for name in (org.sections or {})]
    if ignored:
        out["diagnostics"]["notes"].append({
            "stage": "lower", "code": "metadata-not-compiled",
            "message": "kept as metadata only; these fields select no backend and change no gate: "
                       + ", ".join(sorted(ignored)[:20])})
    if org.fitness:
        out["diagnostics"]["notes"].append({"stage": "lower", "code": "fitness-not-run",
                                            "message": f"fitness '{org.fitness}' is declared; compiling does not run evolution"})
    diag = dnalang.check(org)
    out["diagnostics"]["errors"] += [{"stage": "check", "message": m, **_pos(m)} for m in diag.errors]
    out["diagnostics"]["warnings"] += [{"stage": "check", "message": m, **_pos(m)} for m in diag.warnings]
    if diag.errors:
        return out
    if not org.genome:
        out["diagnostics"]["errors"].append({"stage": "lower", "message": "organism has no genome to execute"})
        return out
    try:
        circ = dnalang.lower(org)
    except (SemaError, ValueError) as e:
        out["diagnostics"]["errors"].append({"stage": "lower", "message": str(e)})
        return out
    out["ok"] = True
    out["circuit"] = {"n_qubits": circ.n_qubits, "n_cbits": circ.n_cbits, "ops": len(circ.ops),
                      "depth": circ.depth(), "two_qubit": circ.two_qubit_count(), "sha256": circ.sha256(),
                      "qasm3": circ.to_qasm3()}
    out["_circuit"] = circ
    return out


def _pos(message: str) -> Dict[str, int]:
    import re
    m = re.match(r"^\s*(\d+):(\d+)", message)
    return {"line": int(m.group(1)), "column": int(m.group(2))} if m else {}


def exact_distribution(circ) -> Optional[Dict[str, float]]:
    """Exact outcome probabilities, keyed like Aer counts (classical bit 0 rightmost).

    Returns None when the circuit is not "unitary then final measurements" (a measured qubit is used
    again, or a classical bit is written twice), because no single exact distribution applies then."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    measured: Dict[int, int] = {}
    for op in circ.ops:
        if op.name == "measure":
            q, c = op.qubits[0], op.cbit
            if c in measured.values() or q in measured:
                return None
            measured[q] = c
        elif op.name not in ("barrier",) and any(q in measured for q in op.qubits):
            return None
    if not measured:
        return None
    qc = circ.to_qiskit()
    unitary = QuantumCircuit(qc.num_qubits)
    for inst in qc.data:
        if inst.operation.name in ("measure", "barrier", "delay"):
            continue
        unitary.append(inst.operation, [qc.find_bit(q).index for q in inst.qubits])
    probs = Statevector(unitary).probabilities()
    n_c = circ.n_cbits
    dist: Dict[str, float] = {}
    for basis, p in enumerate(probs):
        if p < 1e-12:
            continue
        bits = ["0"] * n_c
        for q, c in measured.items():
            bits[n_c - 1 - c] = "1" if (basis >> q) & 1 else "0"
        key = "".join(bits)
        dist[key] = dist.get(key, 0.0) + float(p)
    return dist


def _tvd(p: Dict[str, float], q: Dict[str, float]) -> float:
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


def _classical_fidelity(p: Dict[str, float], q: Dict[str, float]) -> float:
    return sum(math.sqrt(p.get(k, 0.0) * q.get(k, 0.0)) for k in set(p) & set(q)) ** 2


def evaluate(counts: Dict[str, int], circ, seed: int) -> Dict[str, Any]:
    """OSIRIS evaluation of one simulator run, with predeclared acceptance criteria."""
    from dnalang.metrics.core import parity_expectation

    n = sum(counts.values())
    emp = {k: v / n for k, v in counts.items()}
    width = max(len(k) for k in counts)
    z = {q: (sum(v for k, v in counts.items() if k[-(q + 1)] == "0") * 2 - n) / n for q in range(width)}
    exact = exact_distribution(circ)
    out: Dict[str, Any] = {"shots": n, "exact_available": exact is not None}
    target = None
    if exact is not None:
        rng = np.random.default_rng(seed)
        keys = sorted(exact)
        pv = np.array([exact[k] for k in keys]); pv = pv / pv.sum()
        null = sorted(_tvd({k: c / n for k, c in zip(keys, draw)}, exact)
                      for draw in rng.multinomial(n, pv, size=TVD_RESAMPLES))
        bound = null[int(0.99 * (TVD_RESAMPLES - 1))]
        obs_keys = sorted(counts)
        ov = np.array([counts[k] for k in obs_keys], dtype=float) / n
        boots = sorted(_classical_fidelity({k: c / n for k, c in zip(obs_keys, d)}, exact)
                       for d in rng.multinomial(n, ov, size=BOOTSTRAP_RESAMPLES))
        tvd = _tvd(emp, exact)
        target = max(sorted(exact), key=lambda k: exact[k])
        out.update(
            exact_distribution={k: round(v, 12) for k, v in sorted(exact.items())},
            # 1e-12 absorbs float round-off when the exact distribution is a single outcome (bound ≈ 0)
            tvd=tvd, tvd_bound_p99=bound, consistent_with_exact=bool(tvd <= bound + 1e-12),
            classical_fidelity=_classical_fidelity(emp, exact),
            classical_fidelity_ci95=[boots[int(0.025 * (BOOTSTRAP_RESAMPLES - 1))],
                                     boots[int(0.975 * (BOOTSTRAP_RESAMPLES - 1))]],
            acceptance_criterion=("TVD(empirical, exact) <= 99th percentile of TVD over "
                                  f"{TVD_RESAMPLES} multinomial samples of the exact distribution at the same shots"),
        )
    claimed = {"total_shots": float(n), "parity": float(parity_expectation(counts)),
               **{f"z_exp_q{q}": float(v) for q, v in z.items()},
               "mean_qubit_zero_probability": float(sum((1 + v) / 2 for v in z.values()) / len(z))}
    if target is not None:
        claimed["survival_fidelity"] = counts.get(target, 0) / n
        out["target_state"] = target
    out["claimed_metrics"] = claimed
    return out


def _versions() -> Dict[str, str]:
    import dnalang, osiris_governance, qiskit, qiskit_aer
    return {"dnalang": dnalang.__version__, "osiris_governance": getattr(osiris_governance, "__version__", "?"),
            "qiskit": qiskit.__version__, "qiskit_aer": qiskit_aer.__version__}


def run_slice(source: str, *, shots: int = 2000, seed: int = 1234, receipts_path: str = DEFAULT_RECEIPTS_PATH,
              simulator: Optional[Callable[..., List[Dict[str, int]]]] = None,
              binding: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run the whole slice. Returns a result whose ``status`` is COMPLETED, REJECTED (bad input or
    compile errors: nothing executed), or FAILED (a stage after admission failed: receipted).

    ``binding`` names the stored file revision this source was read from (project_id, file_id, path,
    revision). It is copied into the result and the receipt beside the source digest; the caller is
    responsible for having read exactly that revision."""
    from osiris_governance.evidence_verifier import ExternalVerifier, RawDataPackage
    from osiris_governance.execution_gate import ExecutionContract, ExecutionGate
    from osiris_governance.models.evidence import EvidenceClass
    from osiris_governance.models.ledger import EvidencePlane
    from osiris_governance.models.release import ExecutionMode
    from osiris_cli.trio_authz import Receipts

    if isinstance(shots, bool) or not isinstance(shots, int) or not 1 <= shots <= MAX_SHOTS:
        raise SliceInputError(f"shots must be an integer in [1, {MAX_SHOTS}]")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise SliceInputError("seed must be an integer in [0, 2^31)")
    run_id = f"slice-{uuid.uuid4().hex[:12]}"
    stages: List[Dict[str, Any]] = []
    result: Dict[str, Any] = {"schema": SCHEMA, "run_id": run_id, "evidence_class": "SIMULATED",
                              "hardware": False, "stages": stages, "versions": _versions(),
                              "requested": {"operation": "simulate", "shots": shots, "seed": seed},
                              "source_binding": dict(binding) if binding else None}

    with _stage(stages, "compile") as st:
        comp = compile_source(source)
        result["compile"] = {k: v for k, v in comp.items() if not k.startswith("_")}
        if not comp["ok"]:
            st["status"] = "FAILED"; st["error"] = "; ".join(e["message"] for e in comp["diagnostics"]["errors"])
    if not comp["ok"]:
        result["status"] = "REJECTED"
        return result
    circ = comp["_circuit"]
    over = ([f"{circ.n_qubits} qubits > {MAX_QUBITS}"] if circ.n_qubits > MAX_QUBITS else []) + \
           ([f"{len(circ.ops)} operations > {MAX_OPS}"] if len(circ.ops) > MAX_OPS else []) + \
           ([f"depth {circ.depth()} > {MAX_DEPTH}"] if circ.depth() > MAX_DEPTH else [])
    if over:
        stages.append({"name": "admission", "status": "FAILED", "error": "simulator limit exceeded: " + "; ".join(over)})
        result["status"] = "REJECTED"
        return result

    with _stage(stages, "admission") as st:
        contract = ExecutionContract(execution_id=run_id, target_artifact_digest=circ.sha256(),
                                     mode=ExecutionMode.SIMULATION,
                                     frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                                     parameters={"require_physical_hardware": False, "shots": shots, "seed": seed})
        mode = ExecutionGate.admit_execution(contract)
        result["admission"] = {"mode": mode.value, "contract_sha256": contract.compute_hash(),
                               "authority": "osiris_governance.ExecutionGate"}

    receipts = Receipts(receipts_path)
    os.makedirs(os.path.dirname(receipts_path), mode=0o700, exist_ok=True)

    def finish(kind: str, payload: Dict[str, Any]) -> None:
        with _receipt_lock:
            row = receipts.append(kind, payload)
            valid = Receipts.verify(receipts_path)
        result["receipt"] = {"path": receipts_path, "hash": row["hash"], "prev": row["prev"], "kind": kind,
                             "chain_valid": valid}

    base_payload = {"run_id": run_id, "source_sha256": comp["source_sha256"], "circuit_sha256": circ.sha256(),
                    "shots": shots, "seed": seed, "evidence_class": "SIMULATED", "versions": result["versions"],
                    "admission": result["admission"], "source_binding": result["source_binding"]}
    try:
        with _stage(stages, "simulate") as st:
            if simulator is None:
                from dnalang.backends import aer
                simulator = aer.run
            counts = simulator([circ], shots=shots, seed=seed)[0]
            if not isinstance(counts, dict) or not counts or sum(counts.values()) != shots:
                raise RuntimeError(f"simulator returned {sum(counts.values()) if isinstance(counts, dict) else counts!r}"
                                   f" shots, expected {shots}")
            result["simulation"] = {"backend": "qiskit_aer.AerSimulator (noiseless, optimization_level=0)",
                                    "label": "SIMULATED", "shots": shots, "seed": seed,
                                    "counts": dict(sorted(counts.items()))}
        with _stage(stages, "evaluate"):
            result["evaluation"] = evaluate(counts, circ, seed)
        with _stage(stages, "verify"):
            raw = RawDataPackage(raw_id=run_id, source_plane=EvidencePlane.REPRODUCIBILITY, provider_receipt_id=None,
                                 raw_payload={"counts": dict(counts)}, captured_at_utc=datetime.now(timezone.utc).isoformat(),
                                 metadata={"target_state": result["evaluation"].get("target_state", "0" * circ.n_cbits)})
            pack = ExternalVerifier().verify_and_seal(raw, result["evaluation"]["claimed_metrics"],
                                                      claim_id=f"CLAIM-{run_id}", evidence_class=EvidenceClass.SIMULATED)
            result["governance"] = {
                "authority": "osiris_governance.ExternalVerifier", "raw_payload_sha256": pack.raw_digest,
                "all_matched": pack.all_matched, "epistemic_status": pack.epistemic_status.value,
                "evidence_class": pack.evidence_record.evidence_class.value,
                "verification": [{"metric": v.metric_name, "claimed": v.declared_value, "recomputed": v.recomputed_value,
                                  "matched": v.matched} for v in pack.verification_results],
            }
    except Exception as e:  # any failure after admission is a negative result and is preserved
        result["status"] = "FAILED"
        finish("slice.failed", {**base_payload, "failed_stage": stages[-1]["name"], "error": stages[-1].get("error",
                                                                                                         repr(e))})
        return result

    ev, gov = result["evaluation"], result["governance"]
    accepted = gov["all_matched"] and ev.get("consistent_with_exact", False)
    result["claim"] = {
        "category": "SIMULATOR_VALIDATED" if accepted else "NOT_SUPPORTED",
        "statement": (f"Noiseless Aer sampling of circuit {circ.sha256()[:16]} at {shots} shots is consistent with "
                      f"its exact distribution (TVD {ev['tvd']:.4f} <= bound {ev['tvd_bound_p99']:.4f}), and governance "
                      f"recomputed all {len(gov['verification'])} claimed metrics from the raw counts."
                      if accepted else
                      "Not accepted: " + ("governance recomputation mismatch" if not gov["all_matched"] else
                                          "no exact distribution or sampling inconsistent with it")),
        "not_claimed": "Nothing about quantum hardware, noise, advantage or any physical effect.",
    }
    result["status"] = "COMPLETED"
    finish("slice.evidence", {**base_payload, "raw_payload_sha256": gov["raw_payload_sha256"],
                              "counts": result["simulation"]["counts"], "epistemic_status": gov["epistemic_status"],
                              "all_matched": gov["all_matched"], "claim_category": result["claim"]["category"],
                              "tvd": ev.get("tvd"), "tvd_bound_p99": ev.get("tvd_bound_p99")})
    return result
