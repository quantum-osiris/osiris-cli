#!/usr/bin/env python3
"""
Quantum Data Loader: Discover, decode, and validate IBM Quantum hardware job results.

All measurement payloads are decoded through ``osiris_cli.qdecode`` (supporting both
SamplerV2 ``PrimitiveResult`` and IBM Executor ``schema_version="v0.2"`` wire formats).
Virtual environments, site-packages, and Qiskit ``fake_provider`` directories are
explicitly excluded, and unmeasured ``ALL_ZERO`` registers are filtered before
registering valid hardware measurements.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import re
import sys
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from osiris_cli import qdecode, qscore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SKIP_DIRS = frozenset({
    "venv",
    ".venv",
    "venvs",
    "site-packages",
    "fake_provider",
    "__pycache__",
    ".git",
    "node_modules",
    ".pytest_cache",
    ".ruff_cache",
    "tests",
})

_JOB_ID_RE = re.compile(r"job-([a-z0-9]{16,24})-result\.json", re.I)


@dataclass
class CircuitMetadata:
    """Metadata for a decoded hardware measurement register."""
    circuit_id: str
    num_qubits: int
    circuit_depth: int
    gate_count: int
    result_path: str
    shots: int
    has_measurement_counts: bool
    has_statevector: bool
    has_noise_info: bool
    creation_timestamp: str
    file_size_bytes: int
    circuit_type: Optional[str] = None
    parameter_values: Optional[Dict] = None
    job_id: Optional[str] = None
    backend: Optional[str] = None
    wire_format: str = "unknown"
    schema_version: Optional[str] = None
    source_sha256: Optional[str] = None
    payload_sha256: Optional[str] = None
    flags: List[str] = field(default_factory=list)


@dataclass
class MeasurementData:
    """Normalized measurement outcome data."""
    circuit_id: str
    counts: Dict[str, int]
    shots: int
    probabilities: Dict[str, float]
    num_qubits: int
    entropy: Optional[float] = None
    entropy_ceiling: Optional[float] = None
    entropy_ceiling_ratio: Optional[float] = None
    purity_estimate: Optional[float] = None
    flags: List[str] = field(default_factory=list)


def _should_skip_path(path: Path, root: Optional[Path] = None) -> bool:
    if root is not None:
        try:
            parts = path.resolve().relative_to(root.resolve()).parts
            return any(part in SKIP_DIRS for part in parts[:-1])
        except ValueError:
            pass
    return any(part in SKIP_DIRS for part in path.parts[:-1])


class QuantumDataLoader:
    """Discover and decode genuine quantum hardware results via ``osiris_cli.qdecode``."""

    def __init__(self, workspace_root: Optional[str | Path] = None):
        self.workspace_root = str(Path(workspace_root) if workspace_root is not None else _REPO_ROOT)
        self.inventory: Dict[str, CircuitMetadata] = {}
        self.measurement_data: Dict[str, MeasurementData] = {}
        self.filtered_all_zero: List[str] = []

    def audit_filesystem(self, max_results: Optional[int] = None) -> Dict[str, Any]:
        """Recursively scan workspace (excluding ``SKIP_DIRS``) for decodable QPU job files."""
        root = Path(self.workspace_root)
        logger.info(f"Starting filesystem audit of {root} (skipping {sorted(SKIP_DIRS)})...")

        results: Dict[str, Any] = {
            "json_files": 0,
            "qasm_files": 0,
            "zip_archives": 0,
            "ibm_result_files": 0,
            "potential_job_results": [],
            "total_size_bytes": 0,
            "directories_scanned": 0,
            "skipped_dirs": sorted(SKIP_DIRS),
        }

        seen = 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            results["directories_scanned"] += 1

            for fname in sorted(filenames):
                fpath = Path(dirpath) / fname
                if _should_skip_path(fpath, root) or not fpath.is_file():
                    continue
                if max_results is not None and seen >= max_results:
                    break

                fsize = fpath.stat().st_size
                results["total_size_bytes"] += fsize
                seen += 1

                if fname.endswith(".json") or fname.endswith(".json.gz"):
                    results["json_files"] += 1
                    if self._is_ibm_job_result(fpath, root=root):
                        results["ibm_result_files"] += 1
                        results["potential_job_results"].append(str(fpath))
                elif fname.endswith(".qasm"):
                    results["qasm_files"] += 1
                elif fname == "workloads.zip" or (fname.endswith(".zip") and "workload" in fname.lower()):
                    results["zip_archives"] += 1
                    for zentry in self._list_zip_job_entries(fpath):
                        results["ibm_result_files"] += 1
                        results["potential_job_results"].append(f"{fpath}!{zentry}")

            if max_results is not None and seen >= max_results:
                break

        results["total_size_gb"] = results["total_size_bytes"] / (1024 ** 3)
        logger.info(
            f"Audit complete: {results['json_files']} JSON files, "
            f"{results['ibm_result_files']} decodable QPU result candidates"
        )
        return results

    def _list_zip_job_entries(self, zip_path: Path) -> List[str]:
        entries: List[str] = []
        try:
            with zipfile.ZipFile(zip_path) as zf:
                for name in sorted(zf.namelist()):
                    if name.endswith("-result.json") or name.endswith("-result.json.gz"):
                        entries.append(name)
        except (OSError, zipfile.BadZipFile) as exc:
            logger.debug(f"Could not inspect zip {zip_path}: {exc}")
        return entries

    def _is_ibm_job_result(self, filepath: Path, root: Optional[Path] = None) -> bool:
        """Check if a JSON or gzip-JSON file carries a ``PrimitiveResult`` or Executor wire header."""
        if _should_skip_path(filepath, root) or filepath.name.endswith("-info.json"):
            return False
        try:
            if filepath.name.endswith(".gz"):
                with gzip.open(filepath, "rt", encoding="utf-8", errors="replace") as fh:
                    head = fh.read(2048)
            else:
                with filepath.open("r", encoding="utf-8", errors="replace") as fh:
                    head = fh.read(2048)
            return (
                '"PrimitiveResult"' in head
                or ('"schema_version"' in head and '"results"' in head)
            )
        except (OSError, EOFError, gzip.BadGzipFile):
            return False

    def _infer_job_info(self, result_spec: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract ``(job_id, backend, created_iso)`` from filename or sibling ``-info.json``."""
        job_id: Optional[str] = None
        backend: Optional[str] = None
        created: Optional[str] = None

        m = _JOB_ID_RE.search(result_spec)
        if m:
            job_id = m.group(1)

        if "!" in result_spec:
            zpath_s, member = result_spec.split("!", 1)
            info_member = member.replace("-result.json.gz", "-info.json").replace("-result.json", "-info.json")
            try:
                with zipfile.ZipFile(zpath_s) as zf:
                    if info_member in zf.namelist():
                        info = json.loads(zf.read(info_member))
                        job_id = info.get("id") or info.get("job_id") or job_id
                        backend = info.get("backend") or info.get("backend_name")
                        created = info.get("created")
            except (OSError, zipfile.BadZipFile, json.JSONDecodeError):
                pass
        else:
            p = Path(result_spec)
            info_p = p.with_name(p.name.replace("-result.json.gz", "-info.json").replace("-result.json", "-info.json"))
            if info_p.is_file() and info_p != p:
                try:
                    info = json.loads(info_p.read_text(encoding="utf-8"))
                    job_id = info.get("id") or info.get("job_id") or job_id
                    backend = info.get("backend") or info.get("backend_name")
                    created = info.get("created")
                except (OSError, json.JSONDecodeError):
                    pass

        return job_id, backend, created

    def load_job_result(self, result_path: str, circuit_id: Optional[str] = None) -> Optional[CircuitMetadata]:
        """Decode a QPU job file (or ``archive.zip!member.json``) via ``osiris_cli.qdecode``.

        Registers every valid measured register in ``self.inventory`` and
        ``self.measurement_data``, discarding ``ALL_ZERO`` unmeasured registers.
        Returns the first valid ``CircuitMetadata`` loaded from the job, or ``None``.
        """
        root = Path(self.workspace_root)
        if "!" in result_path:
            zpath_s, member = result_path.split("!", 1)
            zpath = Path(zpath_s)
            if _should_skip_path(zpath, root) or not zpath.is_file():
                return None
            with zipfile.ZipFile(zpath) as zf:
                blob = zf.read(member)
            created_fallback = datetime.fromtimestamp(zpath.stat().st_mtime, tz=timezone.utc).isoformat()
        else:
            p = Path(result_path)
            if _should_skip_path(p, root) or not p.is_file():
                logger.warning(f"Skipping missing or excluded path: {result_path}")
                return None
            blob = p.read_bytes()
            created_fallback = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()

        job_rec = qdecode.decode_job_bytes(blob, source=result_path)
        if job_rec.status not in (qdecode.OK, qdecode.SUSPECT_UNMEASURED):
            logger.warning(f"Skipping {result_path}: qdecode status={job_rec.status} ({job_rec.warnings})")
            return None

        job_id, backend, created_iso = self._infer_job_info(result_path)
        base_id = circuit_id or job_id or hashlib.sha256(result_path.encode("utf-8")).hexdigest()[:16]
        total_regs = sum(len(pub.registers) for pub in job_rec.pubs)

        first_meta: Optional[CircuitMetadata] = None
        for pub in job_rec.pubs:
            for reg in pub.registers:
                loc_suffix = ("_" + "_".join(map(str, reg.loc))) if reg.loc else ""
                cid = (
                    base_id
                    if (total_regs == 1 and circuit_id is not None)
                    else f"{base_id}_pub{pub.index}_{reg.name}{loc_suffix}"
                )
                if "ALL_ZERO" in reg.flags or qdecode.is_all_zero_register(reg.counts):
                    self.filtered_all_zero.append(cid)
                    continue
                if reg.num_bits <= 0 or reg.shots <= 0 or not reg.counts:
                    continue

                mdata = self.normalize_measurement_counts(cid, reg.counts, num_qubits=reg.num_bits)
                meta = CircuitMetadata(
                    circuit_id=cid,
                    num_qubits=reg.num_bits,
                    circuit_depth=0,
                    gate_count=0,
                    result_path=result_path,
                    shots=reg.shots,
                    has_measurement_counts=True,
                    has_statevector=False,
                    has_noise_info=False,
                    creation_timestamp=created_iso or created_fallback,
                    file_size_bytes=job_rec.source_bytes,
                    job_id=job_id,
                    backend=backend,
                    wire_format=job_rec.wire_format,
                    schema_version=job_rec.schema_version,
                    source_sha256=job_rec.source_sha256,
                    payload_sha256=reg.payload_sha256,
                    flags=list(mdata.flags),
                )
                self.inventory[cid] = meta
                if first_meta is None:
                    first_meta = meta

        return first_meta

    def normalize_measurement_counts(
        self,
        circuit_id: str,
        counts: Dict[str, int],
        num_qubits: Optional[int] = None,
    ) -> MeasurementData:
        """Normalize raw measurement counts to probabilities and finite-shot entropy metrics."""
        total_shots = sum(counts.values())
        inferred_q = len(next(iter(counts.keys()))) if counts else 0
        nq = int(num_qubits if num_qubits is not None else inferred_q)

        probabilities = {b: c / total_shots for b, c in counts.items()} if total_shots > 0 else {}
        escore = qscore.score_entropy(counts, num_bits=nq)
        purity = self._compute_purity_estimate(probabilities)

        data = MeasurementData(
            circuit_id=circuit_id,
            counts=dict(counts),
            shots=total_shots,
            probabilities=probabilities,
            num_qubits=nq,
            entropy=escore.shannon_entropy,
            entropy_ceiling=escore.ceiling,
            entropy_ceiling_ratio=escore.ceiling_ratio,
            purity_estimate=purity,
            flags=list(escore.flags),
        )
        self.measurement_data[circuit_id] = data
        return data

    def _compute_entropy(self, probabilities: Dict[str, float]) -> float:
        entropy = 0.0
        for p in probabilities.values():
            if p > 0:
                entropy -= p * np.log2(p)
        return float(entropy)

    def _compute_purity_estimate(self, probabilities: Dict[str, float]) -> float:
        return float(sum(p ** 2 for p in probabilities.values()))

    def save_inventory(self, output_path: str = "data_inventory.json"):
        inventory_data = {cid: asdict(meta) for cid, meta in self.inventory.items()}
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(inventory_data, fh, indent=2)
        logger.info(f"Saved inventory of {len(self.inventory)} measured registers to {output_path}")

    def export_provenance_manifest(self, output_path: str = "HARDWARE_PROVENANCE_MANIFEST.json") -> Dict[str, Any]:
        """Export canonical per-register QPU provenance and finite-shot entropy manifest."""
        records: List[Dict[str, Any]] = []
        for cid, meta in sorted(self.inventory.items()):
            mdata = self.measurement_data.get(cid)
            records.append({
                "circuit_id": cid,
                "job_id": meta.job_id,
                "backend": meta.backend,
                "wire_format": meta.wire_format,
                "schema_version": meta.schema_version,
                "shots": meta.shots,
                "num_bits": meta.num_qubits,
                "source_sha256": meta.source_sha256,
                "payload_sha256": meta.payload_sha256,
                "entropy": None if mdata is None else mdata.entropy,
                "ceiling": None if mdata is None else mdata.entropy_ceiling,
                "ceiling_ratio": None if mdata is None else mdata.entropy_ceiling_ratio,
                "flags": list(meta.flags),
            })
        manifest = {
            "manifest_version": 1,
            "decoder": qdecode.DECODER_NAME,
            "decoder_version": qdecode.DECODER_VERSION,
            "total_measured_registers": len(records),
            "filtered_all_zero_registers": len(self.filtered_all_zero),
            "distinct_jobs": len({r["job_id"] for r in records if r["job_id"]}),
            "registers": records,
        }
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        logger.info(f"Exported hardware provenance manifest ({len(records)} registers) to {output_path}")
        return manifest

    def get_inventory_summary(self) -> Dict[str, Any]:
        if not self.inventory:
            return {"error": "No circuits loaded", "filtered_all_zero": len(self.filtered_all_zero)}

        qubits = [m.num_qubits for m in self.inventory.values()]
        depths = [m.circuit_depth for m in self.inventory.values()]
        shots = [m.shots for m in self.inventory.values()]
        jobs = {m.job_id for m in self.inventory.values() if m.job_id}

        return {
            "total_circuits": len(self.inventory),
            "distinct_jobs": len(jobs),
            "filtered_all_zero": len(self.filtered_all_zero),
            "qubit_range": [int(min(qubits)), int(max(qubits))],
            "depth_range": [int(min(depths)), int(max(depths))],
            "shots_range": [int(min(shots)), int(max(shots))],
            "avg_shots": float(np.mean(shots)),
            "circuits_with_measurement_counts": sum(1 for m in self.inventory.values() if m.has_measurement_counts),
            "circuits_with_statevector": sum(1 for m in self.inventory.values() if m.has_statevector),
        }


def main():
    loader = QuantumDataLoader()
    audit_results = loader.audit_filesystem(max_results=10000)
    for result_path in audit_results["potential_job_results"]:
        loader.load_job_result(result_path)
    loader.save_inventory("data_inventory.json")
    print(json.dumps(loader.get_inventory_summary(), indent=2))


if __name__ == "__main__":
    main()
