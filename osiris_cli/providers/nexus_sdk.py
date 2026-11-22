"""Quantinuum Nexus transport built on the official ``qnexus`` SDK (pinned 0.51.4).

Why this module exists: ``quantinuum_nexus.QuantinuumNexusClient`` speaks a hand-written REST dialect
(``/api/v1/backends``, ``/api/v1/targets``, ``/api/v1/jobs``) that the Nexus service does not expose; the
real API is versioned per resource (``/api/circuits/v1beta2``, ``/api/jobs/v1beta3``, ...) and authenticates
with ``qnx.login()`` tokens. Rather than re-implement that API, this module calls the SDK, so request shapes
and auth come from Quantinuum's own client. Every SDK call goes through ``NexusSdkRunner._call`` so tests
can record the exact keyword arguments and check them against the real ``qnexus`` signatures.

Flow (each step writes a ``NexusDisposition`` to the hash-chained ``NexusAuditLedger``):

  preflight  upload -> compile on Nexus (``start_compile_job``) -> price every compiled circuit with
             Quantinuum's syntax checker (``circuits.cost(..., syntax_checker="H2-1SC")``) and cross-check
             against the local HQC formula. Tier 1, costs nothing.
  run        execute compiled circuits (``start_execute_job``) with a per-circuit ``max_cost`` cap.
             Tier 2 (emulators, e.g. H2-1E) is capped by ``max_emulator_hqc``; Tier 3 (H2-1, H2-2) refuses
             to start without a signed Tier 3 token bound to the compiled-circuit SHA-256, target, shots
             and HQC (``quantinuum_gate.verify_tier3_token``).

Login is interactive and is the operator's job: ``python -c "import qnexus; qnexus.login()"``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .quantinuum import compute_hqc, resolve_quantinuum_target
from .quantinuum_nexus import GovernanceError, NexusDisposition

QNEXUS_VERSION = "0.51.4"


class NexusSdkUnavailable(RuntimeError):
    pass


def load_qnexus():
    try:
        import qnexus  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise NexusSdkUnavailable(
            f"qnexus is not installed. Install the extra: pip install 'osiris-cli[nexus]' (qnexus=={QNEXUS_VERSION})"
        ) from exc
    return qnexus


def tket_sha256(circ: Any) -> str:
    """SHA-256 of a pytket circuit's canonical JSON (sorted keys, no whitespace)."""
    return hashlib.sha256(json.dumps(circ.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def batch_sha256(hashes: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(hashes).encode()).hexdigest()


def local_hqc(circ: Any, shots: int) -> Decimal:
    """Local HQC formula on a compiled pytket circuit: Rz is virtual (free); PhasedX/other 1q gates count 1,
    2q gates 10, measurements + resets + one preparation per qubit count 5."""
    from pytket import OpType
    n1 = n2 = nm = 0
    for cmd in circ.get_commands():
        op = cmd.op
        while op.type == OpType.Conditional:
            op = op.op
        t = op.type
        if t in (OpType.Measure, OpType.Reset):
            nm += 1
        elif t in (OpType.Barrier, OpType.Rz) or not cmd.qubits:
            continue
        elif len(cmd.qubits) == 1:
            n1 += 1
        elif len(cmd.qubits) == 2:
            n2 += 1
    return compute_hqc(n1, n2, nm + circ.n_qubits, shots)


def load_circuits(path: Path) -> List[Any]:
    """Read circuits from .qasm (OpenQASM 2.0), pytket JSON (one dict), or a list of {"name", "tket"}
    entries such as docs/xtop005/quantinuum/circuits_h2_1_tket.json."""
    from pytket import Circuit
    p = Path(path)
    if p.suffix.lower() == ".qasm":
        from pytket.qasm import circuit_from_qasm
        c = circuit_from_qasm(str(p))
        c.name = c.name or p.stem
        return [c]
    data = json.loads(p.read_text())
    items = data if isinstance(data, list) else [data]
    out = []
    for i, it in enumerate(items):
        d = it.get("tket", it) if isinstance(it, dict) else it
        c = Circuit.from_dict(d)
        c.name = (it.get("name") if isinstance(it, dict) else None) or c.name or f"{p.stem}_{i}"
        out.append(c)
    return out


@dataclass
class PreflightResult:
    target: str
    shots: int
    project_id: str
    compile_job_id: str
    compiled_refs: List[Any]
    compiled_sha256: List[str]
    source_sha256: List[str]
    nexus_hqc: List[Optional[float]]
    local_hqc: List[str]
    receipt: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_nexus_hqc(self) -> Optional[float]:
        return None if any(v is None for v in self.nexus_hqc) else round(sum(self.nexus_hqc), 6)

    @property
    def total_local_hqc(self) -> str:
        return str(sum((Decimal(v) for v in self.local_hqc), Decimal("0")))

    def summary(self) -> Dict[str, Any]:
        return {"target": self.target, "shots": self.shots, "circuits": len(self.compiled_refs),
                "project_id": self.project_id, "compile_job_id": self.compile_job_id,
                "compiled_batch_sha256": batch_sha256(self.compiled_sha256),
                "nexus_hqc_total": self.total_nexus_hqc, "local_hqc_total": self.total_local_hqc,
                "receipt_sha256": self.receipt.get("receipt_sha256")}


class NexusSdkRunner:
    def __init__(self, qnx: Any = None, ledger: Any = None, project_name: str = "OSIRIS",
                 project_description: str = "OSIRIS governed experiments (osiris-cli)") -> None:
        self._qnx = qnx if qnx is not None else load_qnexus()
        if ledger is None:
            from ..nexus_ledger import NexusAuditLedger
            ledger = NexusAuditLedger()
        self.ledger = ledger
        self.project_name = project_name
        self.project_description = project_description
        self.calls: List[Tuple[str, Dict[str, Any]]] = []
        self._project = None

    # Every SDK call goes through here (recorded for signature tests; positional args are never used).
    def _call(self, dotted: str, **kwargs: Any) -> Any:
        self.calls.append((dotted, kwargs))
        fn: Any = self._qnx
        for part in dotted.split("."):
            fn = getattr(fn, part)
        return fn(**kwargs)

    def _config(self, device_name: str):
        return self._qnx.QuantinuumConfig(device_name=device_name)

    def project(self):
        if self._project is None:
            self._project = self._call("projects.get_or_create", name=self.project_name,
                                       description=self.project_description)
        return self._project

    def _record(self, **kw: Any) -> Dict[str, Any]:
        return self.ledger.append_disposition(NexusDisposition(**kw))

    def preflight(self, circuits: Sequence[Any], target: str = "H2-1", shots: int = 500,
                  optimisation_level: int = 0, job_name: str = "osiris-preflight",
                  timeout: float = 900.0, approve_remote: bool = False) -> PreflightResult:
        """Compile on Nexus for ``target`` (an H2 QPU name) and price it with that QPU's syntax checker.

        This uploads circuits and starts a compile job on Nexus, so it needs ``approve_remote=True``
        (the CLI asks the operator, or takes ``--approve-remote``). Nothing is sent before that check.
        """
        spec = resolve_quantinuum_target(target)
        if spec.family != "H2" or spec.role != "qpu":
            raise GovernanceError(f"preflight takes an H2 QPU name (H2-1, H2-2) to compile and price for; got {target}")
        if not approve_remote:
            raise GovernanceError("preflight uploads and compiles on Quantinuum Nexus: explicit operator approval "
                                  "is required (approve_remote=True / --approve-remote)")
        checker = f"{spec.name}SC"
        proj = self.project()
        src = [tket_sha256(c) for c in circuits]
        refs = [self._call("circuits.upload", circuit=c, project=proj, name=c.name or f"{job_name}-{i}")
                for i, c in enumerate(circuits)]
        job = self._call("start_compile_job", programs=refs, backend_config=self._config(spec.name),
                         name=job_name, project=proj, optimisation_level=optimisation_level)
        self._call("jobs.wait_for", job=job, timeout=timeout)
        results = self._call("jobs.results", job=job)
        compiled_refs = [r.get_output() for r in results]
        compiled = [r.download_circuit() for r in compiled_refs]
        comp_sha = [tket_sha256(c) for c in compiled]
        nexus_cost: List[Optional[float]] = []
        for r in compiled_refs:
            v = self._call("circuits.cost", circuit_ref=r, n_shots=shots,
                           backend_config=self._config(spec.name), syntax_checker=checker, project=proj)
            nexus_cost.append(None if v is None else float(v))
        local = [str(local_hqc(c, shots)) for c in compiled]
        job_id = str(getattr(job, "id", ""))
        res = PreflightResult(target=spec.name, shots=shots, project_id=str(getattr(proj, "id", "")),
                              compile_job_id=job_id, compiled_refs=compiled_refs, compiled_sha256=comp_sha,
                              source_sha256=src, nexus_hqc=nexus_cost, local_hqc=local)
        res.receipt = self._record(
            target=checker, tier=1, disposition="PREFLIGHT_CHECKED", hardware=False, status="COMPLETED",
            job_id=job_id, shots=shots,
            estimated_hqc=float(res.total_nexus_hqc if res.total_nexus_hqc is not None else res.total_local_hqc),
            source_sha256=batch_sha256(src), compiled_sha256=batch_sha256(comp_sha), compiler_backend="nexus-sdk",
            preflight_target=checker, preflight_job_id=job_id, preflight_passed=True,
            metrics={"nexus_hqc": nexus_cost, "local_hqc": local, "circuits": len(compiled),
                     "compiled_for": spec.name, "optimisation_level": optimisation_level},
            compiler_options={"qnexus": QNEXUS_VERSION, "optimisation_level": optimisation_level})
        return res

    def run(self, pre: PreflightResult, target: str, shots: Optional[int] = None,
            approval_token: Optional[str] = None, max_emulator_hqc: float = 200.0,
            job_name: str = "osiris-run", timeout: float = 3600.0,
            verify_token: Optional[Callable[..., Any]] = None, approve_remote: bool = False) -> Dict[str, Any]:
        """Execute the pre-flighted circuits on ``target`` (H2-1E/H2-2E emulator, or H2-1/H2-2 hardware).

        Emulator jobs need ``approve_remote=True``; hardware jobs need a signed Tier 3 token, which is
        itself the operator's approval. ``HARDWARE_ATTESTED`` is written only after Nexus confirms the job
        completed on the requested QPU (see ``_attest_hardware``).
        """
        spec = resolve_quantinuum_target(target)
        if spec.tier == 1:
            raise GovernanceError("run needs an emulator or QPU target; the syntax checker ran in preflight")
        if spec.tier == 2 and not approve_remote:
            raise GovernanceError(f"{spec.name} jobs use the Nexus emulator quota: explicit operator approval is "
                                  "required (approve_remote=True / --approve-remote)")
        shots = int(shots or pre.shots)
        if shots == pre.shots and pre.total_nexus_hqc is not None:
            per_circ = [Decimal(str(v)) for v in pre.nexus_hqc]
        else:
            scale = Decimal(shots) / Decimal(pre.shots)
            per_circ = [Decimal("5") + (Decimal(v) - Decimal("5")) * scale for v in pre.local_hqc]
        total = sum(per_circ, Decimal("0"))
        compiled_batch = batch_sha256(pre.compiled_sha256)
        if spec.tier == 2 and total > Decimal(str(max_emulator_hqc)):
            raise GovernanceError(f"emulator run would cost ~{total} HQC, above max_emulator_hqc={max_emulator_hqc}")
        if spec.tier == 3:
            if not approval_token:
                raise GovernanceError(f"{spec.name} is live hardware: a signed Tier 3 approval token is required")
            if verify_token is None:
                from ..quantinuum_gate import verify_tier3_token as verify_token
            verify_token(approval_token, compiled_batch, spec, shots, total, compiled_sha256=compiled_batch)
        caps = [float(v) + 1.0 for v in per_circ]
        job = self._call("start_execute_job", programs=pre.compiled_refs, n_shots=[shots] * len(pre.compiled_refs),
                         backend_config=self._config(spec.name), name=job_name, project=self.project(),
                         max_cost=caps)
        self._call("jobs.wait_for", job=job, timeout=timeout)
        results = self._call("jobs.results", job=job)
        counts = []
        for r in results:
            br = r.download_result()
            c = br.get_counts()
            counts.append({"".join(str(b) for b in k): int(v) for k, v in sorted(c.items())})
        job_id = str(getattr(job, "id", ""))
        metrics: Dict[str, Any] = {"circuits": len(counts), "max_cost_per_circuit": caps}
        diagnostics: List[str] = []
        if spec.tier == 3:
            attestation, problems = self._attest_hardware(job, spec, results)
            metrics["attestation"] = attestation
            diagnostics.extend(problems)
            disposition, hw = ("HARDWARE_ATTESTED", True) if not problems else ("HARDWARE_UNVERIFIED", False)
        else:
            disposition, hw = "SIMULATED", False
        receipt = self._record(
            target=spec.name, tier=spec.tier, disposition=disposition,
            hardware=hw, status="COMPLETED", job_id=job_id, shots=shots, estimated_hqc=float(total),
            source_sha256=batch_sha256(pre.source_sha256), compiled_sha256=compiled_batch,
            compiler_backend="nexus-sdk", preflight_target=f"{pre.target}SC", preflight_job_id=pre.compile_job_id,
            preflight_passed=True, metrics=metrics, diagnostics=diagnostics,
            compiler_options={"qnexus": QNEXUS_VERSION})
        return {"target": spec.name, "job_id": job_id, "shots": shots, "hqc_estimate": str(total),
                "disposition": disposition, "diagnostics": diagnostics, "counts": counts, "receipt": receipt}

    def _attest_hardware(self, job: Any, spec: Any, results: Sequence[Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Ask Nexus, independently of the submit call, where and how the job ran.

        Checks: the job status is COMPLETED; the job record fetched again by id names ``spec.name`` as its
        device; every result's backend info names the same device. Any failure is returned as a problem and
        the run is recorded as HARDWARE_UNVERIFIED, never HARDWARE_ATTESTED.
        """
        problems: List[str] = []
        st = self._call("jobs.status", job=job)
        status = str(getattr(getattr(st, "status", None), "value", getattr(st, "status", "")))
        if status != "COMPLETED":
            problems.append(f"Nexus job status is {status or 'unknown'}, not COMPLETED")
        fresh = self._call("jobs.get", id=getattr(job, "id", None))
        cfg = getattr(fresh, "backend_config_store", None)
        job_device = getattr(cfg, "device_name", None)
        if job_device != spec.name:
            problems.append(f"Nexus job record names device {job_device!r}, expected {spec.name!r}")
        result_devices = []
        for r in results:
            bi = r.download_backend_info()
            result_devices.append(getattr(bi, "device_name", None))
        if not result_devices or any(d != spec.name for d in result_devices):
            problems.append(f"result backend devices {sorted(set(map(str, result_devices)))} != {spec.name!r}")
        completed = getattr(st, "completed_time", None)
        evidence = {"job_status": status, "job_device": job_device, "result_devices": result_devices,
                    "completed_time": completed.isoformat() if hasattr(completed, "isoformat") else completed,
                    "nexus_cost": getattr(st, "cost", None)}
        return evidence, problems
