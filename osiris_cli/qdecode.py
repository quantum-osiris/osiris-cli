"""
osiris_qdecode — corrected decoder for IBM Quantum (Qiskit Runtime) sampler results.

Replaces the decoding path in analyze_existing_hardware.py.

The defect this fixes
---------------------
Qiskit Runtime stores each measurement register as a BitArray whose packed
uint8 array is serialized as base64(zlib(np.save(array))). np.save writes a
self-describing .npy header first: the magic bytes b"\\x93NUMPY", a version, and
a Python-dict header string space-padded to a 64-byte boundary (128 bytes for
typical sampler arrays). Casting the decompressed stream with np.frombuffer
treats that header as measurement bytes and pushes the same number of genuine
shots off the end.

This module loads every payload with np.load(..., allow_pickle=False), which
parses and strips the header. It never casts a raw buffer to measurement data.

Two independent decode paths
----------------------------
1. A numpy-only implementation in this file (no Qiskit required).
2. qiskit_ibm_runtime.RuntimeDecoder + BitArray.get_counts(), the reference.
When Qiskit is installed both run and their counts must agree exactly. A
disagreement is reported as DECODER_DISAGREEMENT and never silently resolved.

Bit order
---------
Bitstrings follow Qiskit convention: classical bit 0 is the RIGHTMOST
character. Each packed row is big-endian, with padding bits in the high
positions of the first byte. Padding bits must be zero; nonzero padding is
flagged because it means the bytes are not what the header says they are.

Usage
-----
    python osiris_qdecode.py JOB.json [MORE.json | DIR ...] -o decoded.json
    python osiris_qdecode.py archive/ -o decoded.json --compare-legacy
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import gzip
import hashlib
import io
import json
import math
import platform
import sys
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

DECODER_NAME = "osiris_qdecode"
DECODER_VERSION = "1.0.0"
NPY_MAGIC = b"\x93NUMPY"

# Job status values
OK = "OK"
SUSPECT_UNMEASURED = "SUSPECT_UNMEASURED"
NO_MEASUREMENTS = "NO_MEASUREMENTS"
DECODER_DISAGREEMENT = "DECODER_DISAGREEMENT"
UNRECOGNIZED = "UNRECOGNIZED"
ERROR = "ERROR"


class DecodeError(ValueError):
    """Raised when a payload is not what its declared format says it is."""


# --------------------------------------------------------------------------
# Low-level payload handling
# --------------------------------------------------------------------------

def decompress_ndarray_payload(b64: str) -> bytes:
    """base64 -> zlib-decompressed bytes. Returns the raw .npy stream."""
    blob = base64.standard_b64decode(b64)
    try:
        return zlib.decompress(blob)
    except zlib.error:
        return blob  # payload stored without compression


def npy_header_length(raw: bytes) -> int:
    """Total length in bytes of the .npy header (magic + version + dict + padding)."""
    if not raw.startswith(NPY_MAGIC):
        raise DecodeError("payload is not a .npy stream (missing \\x93NUMPY magic)")
    major = raw[6]
    if major == 1:
        return 10 + int.from_bytes(raw[8:10], "little")
    if major in (2, 3):
        return 12 + int.from_bytes(raw[8:12], "little")
    raise DecodeError(f"unsupported .npy format version {major}")


def load_ndarray_payload(b64: str) -> np.ndarray:
    """Correct decode: parse the .npy header and return only the array data."""
    raw = decompress_ndarray_payload(b64)
    npy_header_length(raw)  # validates the magic before handing to numpy
    return np.load(io.BytesIO(raw), allow_pickle=False)


def packed_rows_to_counts(packed: np.ndarray, num_bits: int) -> tuple[dict[str, int], int]:
    """Convert a packed (shots, nbytes) uint8 array to Qiskit-order counts.

    Returns (counts, rows_with_nonzero_padding_bits).
    """
    if packed.dtype != np.uint8:
        raise DecodeError(f"expected uint8 packed bits, got {packed.dtype}")
    if packed.ndim != 2:
        raise DecodeError(f"expected a (shots, nbytes) array, got shape {packed.shape}")
    shots, nbytes = packed.shape
    if num_bits == 0:
        return {}, 0
    expected = math.ceil(num_bits / 8)
    if nbytes != expected:
        raise DecodeError(
            f"num_bits={num_bits} needs {expected} bytes per shot, payload has {nbytes}"
        )
    if shots == 0:
        return {}, 0
    pad = 8 * nbytes - num_bits
    bad_padding = 0
    if pad:
        mask = (0xFF << (8 - pad)) & 0xFF
        bad_padding = int(np.count_nonzero(packed[:, 0] & mask))
    rows, freq = np.unique(packed, axis=0, return_counts=True)
    counts: dict[str, int] = {}
    for row, c in zip(rows, freq):
        key = "".join(f"{b:08b}" for b in row.tolist())[pad:]
        counts[key] = counts.get(key, 0) + int(c)
    return counts, bad_padding


def emulate_legacy_decode(b64: str, num_bits: int, shots: int) -> dict[str, Any]:
    """Reproduce the defect described in the forensic dossier, for side-by-side
    comparison only.

    This is an emulation of the described behaviour (np.frombuffer over the
    whole decompressed .npy stream, keeping the first shots*nbytes values). It is
    not the original analyze_existing_hardware.py source.
    """
    raw = decompress_ndarray_payload(b64)
    header = npy_header_length(raw)
    nbytes = max(1, math.ceil(num_bits / 8))
    flat = np.frombuffer(raw, dtype=np.uint8)[: shots * nbytes]
    rows = flat[: (flat.size // nbytes) * nbytes].reshape(-1, nbytes)
    counts, _ = packed_rows_to_counts(rows.copy(), num_bits)
    return {
        "counts": counts,
        "header_bytes": header,
        "header_rows_read_as_shots": math.ceil(header / nbytes),
        "genuine_bytes_dropped_from_end": header,
        # When the header length is not a multiple of the row width, every
        # row after the header straddles two real shots, so all rows are garbled.
        "row_aligned": header % nbytes == 0,
    }


def total_variation_distance(a: dict[str, int], b: dict[str, int]) -> float:
    na, nb = sum(a.values()), sum(b.values())
    if na == 0 or nb == 0:
        return float("nan")
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb) for k in keys)


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------

@dataclass
class RegisterRecord:
    name: str
    loc: list[int]          # index into the PUB's sweep shape; [] when not swept
    num_bits: int
    shots: int
    unique: int
    counts: dict[str, int]
    flags: list[str] = field(default_factory=list)
    legacy: dict[str, Any] | None = None


@dataclass
class PubRecord:
    index: int
    shape: list[int]
    metadata: dict[str, Any]
    registers: list[RegisterRecord]


@dataclass
class JobRecord:
    source: str
    source_sha256: str
    source_bytes: int
    status: str
    decoder: str = DECODER_NAME
    decoder_version: str = DECODER_VERSION
    paths_run: list[str] = field(default_factory=list)
    paths_agree: bool | None = None
    warnings: list[str] = field(default_factory=list)
    pubs: list[PubRecord] = field(default_factory=list)

    @property
    def total_shots(self) -> int:
        # Count each PUB's shots once (registers in one PUB share shots).
        total = 0
        for p in self.pubs:
            per_loc: dict[tuple, int] = {}
            for r in p.registers:
                per_loc[tuple(r.loc)] = max(per_loc.get(tuple(r.loc), 0), r.shots)
            total += sum(per_loc.values())
        return total

    def counts(self, pub: int, register: str | None = None, loc: Iterable[int] = ()) -> dict[str, int]:
        """Convenience accessor: counts for one register of one PUB."""
        loc = list(loc)
        regs = [r for r in self.pubs[pub].registers if r.loc == loc]
        if register is None:
            if len(regs) != 1:
                names = sorted({r.name for r in regs})
                raise KeyError(f"PUB {pub} has registers {names}; name one")
            return regs[0].counts
        for r in regs:
            if r.name == register:
                return r.counts
        raise KeyError(f"PUB {pub} has no register {register!r} at loc {loc}")


# --------------------------------------------------------------------------
# Decode paths
# --------------------------------------------------------------------------

def _unwrap(node: Any, types: tuple[str, ...]) -> dict:
    if not (isinstance(node, dict) and node.get("__type__") in types):
        got = node.get("__type__") if isinstance(node, dict) else type(node).__name__
        raise DecodeError(f"expected {'/'.join(types)}, found {got}")
    return node["__value__"]


def _register_flags(counts: dict[str, int], num_bits: int, shots: int, bad_padding: int) -> list[str]:
    flags = []
    if num_bits == 0:
        flags.append("ZERO_WIDTH")
    if shots == 0:
        flags.append("ZERO_SHOTS")
    if num_bits and shots and counts == {"0" * num_bits: shots}:
        flags.append("ALL_ZERO")  # every shot all-zeros: register may never have been measured
    if bad_padding:
        flags.append(f"NONZERO_PADDING_BITS:{bad_padding}")
    return flags


def _pubs_from_raw(doc: Any, compare_legacy: bool) -> list[PubRecord]:
    """numpy-only path over the raw RuntimeEncoder JSON."""
    pr = _unwrap(doc, ("PrimitiveResult",))
    out = []
    for i, p in enumerate(pr["pub_results"]):
        pv = _unwrap(p, ("SamplerPubResult", "PubResult"))
        data = _unwrap(pv["data"], ("DataBin",))
        shape = list(data.get("shape", []))
        regs: list[RegisterRecord] = []
        for name, node in data.get("fields", {}).items():
            if not (isinstance(node, dict) and node.get("__type__") == "BitArray"):
                continue  # not a measurement register
            bv = node["__value__"]
            b64 = bv["array"]["__value__"]
            num_bits = int(bv["num_bits"])
            arr = load_ndarray_payload(b64)
            if arr.ndim < 2:
                raise DecodeError(f"PUB {i} register {name!r}: array has shape {arr.shape}")
            for loc in np.ndindex(*arr.shape[:-2]):
                packed = np.ascontiguousarray(arr[loc])
                counts, bad = packed_rows_to_counts(packed, num_bits)
                shots = packed.shape[0]
                rec = RegisterRecord(
                    name=name, loc=list(loc), num_bits=num_bits, shots=shots,
                    unique=len(counts), counts=counts,
                    flags=_register_flags(counts, num_bits, shots, bad),
                )
                if compare_legacy and loc == () and num_bits and shots:
                    leg = emulate_legacy_decode(b64, num_bits, shots)
                    leg["total_variation_distance"] = total_variation_distance(counts, leg["counts"])
                    leg["legacy_unique"] = len(leg["counts"])
                    rec.legacy = leg
                regs.append(rec)
        out.append(PubRecord(index=i, shape=shape, metadata=pv.get("metadata", {}), registers=regs))
    return out


def _qiskit_counts(text: str) -> dict[tuple, dict[str, int]] | None:
    """Reference path. Returns {(pub, register, loc): counts}, or None if Qiskit is absent."""
    try:
        from qiskit_ibm_runtime import RuntimeDecoder
    except ImportError:
        return None
    result = json.loads(text, cls=RuntimeDecoder)
    out: dict[tuple, dict[str, int]] = {}
    for i, pub in enumerate(result):
        data = pub.data
        for name in data.keys():
            ba = data[name]
            if type(ba).__name__ != "BitArray":
                continue
            for loc in np.ndindex(*ba.shape):
                out[(i, name, loc)] = dict(ba.get_counts(loc=loc if loc else None))
    return out


# --------------------------------------------------------------------------
# File-level entry point
# --------------------------------------------------------------------------

def _read_text(path: Path) -> tuple[bytes, str]:
    blob = path.read_bytes()
    raw = gzip.decompress(blob) if blob[:2] == b"\x1f\x8b" else blob
    return blob, raw.decode("utf-8")


def decode_job_file(path: str | Path, compare_legacy: bool = False, use_qiskit: bool = True) -> JobRecord:
    path = Path(path)
    blob, text = _read_text(path)
    job = JobRecord(
        source=str(path),
        source_sha256=hashlib.sha256(blob).hexdigest(),
        source_bytes=len(blob),
        status=OK,
    )
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        job.status = UNRECOGNIZED
        job.warnings.append(f"not JSON: {exc}")
        return job

    try:
        job.pubs = _pubs_from_raw(doc, compare_legacy)
        job.paths_run.append("numpy")
    except (DecodeError, KeyError, TypeError, ValueError) as exc:
        job.status = UNRECOGNIZED
        job.warnings.append(f"not a sampler PrimitiveResult: {exc}")
        return job

    if use_qiskit:
        try:
            ref = _qiskit_counts(text)
        except Exception as exc:  # reference decoder failed: record it, keep numpy result
            ref = None
            job.warnings.append(f"qiskit reference decode failed: {exc!r}")
        if ref is not None:
            job.paths_run.append("qiskit_ibm_runtime")
            mine = {
                (p.index, r.name, tuple(r.loc)): r.counts
                for p in job.pubs for r in p.registers
            }
            job.paths_agree = mine == ref
            if not job.paths_agree:
                job.status = DECODER_DISAGREEMENT
                diff = sorted(set(mine) ^ set(ref)) or [k for k in mine if mine[k] != ref.get(k)]
                job.warnings.append(f"numpy and qiskit decoders disagree on {len(diff)} register(s): {diff[:5]}")
                return job

    regs = [r for p in job.pubs for r in p.registers]
    if not regs:
        job.status = NO_MEASUREMENTS
        job.warnings.append("no measurement registers in any PUB")
    elif all({"ZERO_WIDTH", "ZERO_SHOTS"} & set(r.flags) for r in regs):
        job.status = NO_MEASUREMENTS
    elif all("ALL_ZERO" in r.flags for r in regs):
        job.status = SUSPECT_UNMEASURED
        job.warnings.append("every register reads all-zeros on every shot; check the circuits measured anything")
    for p in job.pubs:
        for r in p.registers:
            for f in r.flags:
                if f.startswith("NONZERO_PADDING_BITS"):
                    job.warnings.append(f"PUB {p.index} register {r.name!r} {f}")
    return job


def _collect(paths: Iterable[str], exclude: Path | None = None) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(q for q in p.rglob("*") if q.is_file() and (q.name.endswith(".json") or q.name.endswith(".json.gz")))
        else:
            files.append(p)
    if exclude is not None:  # never ingest our own report on a rerun
        files = [f for f in files if f.resolve() != exclude.resolve()]
    return files


def _environment() -> dict[str, str]:
    env = {"python": platform.python_version(), "numpy": np.__version__}
    for mod in ("qiskit", "qiskit_ibm_runtime"):
        try:
            env[mod] = __import__(mod).__version__
        except ImportError:
            env[mod] = "not installed"
    return env


def _safe_decode(path: Path, compare_legacy: bool) -> JobRecord:
    """One bad file must not stop a 1,400-file run: record it as ERROR and continue."""
    try:
        return decode_job_file(path, compare_legacy=compare_legacy)
    except Exception as exc:
        try:
            blob = path.read_bytes()
            sha, size = hashlib.sha256(blob).hexdigest(), len(blob)
        except OSError:
            sha, size = "", 0
        return JobRecord(source=str(path), source_sha256=sha, source_bytes=size,
                         status=ERROR, warnings=[f"{type(exc).__name__}: {exc}"])


def build_report(files: list[Path], compare_legacy: bool, summary_only: bool) -> dict[str, Any]:
    jobs = [_safe_decode(f, compare_legacy) for f in files]
    status_counts: dict[str, int] = {}
    for j in jobs:
        status_counts[j.status] = status_counts.get(j.status, 0) + 1
    entries = []
    for j in jobs:
        d = asdict(j)
        d["total_shots"] = j.total_shots
        if summary_only:
            for p in d["pubs"]:
                for r in p["registers"]:
                    r.pop("counts", None)
                    if r.get("legacy"):
                        r["legacy"].pop("counts", None)
        entries.append(d)
    return {
        "decoder": DECODER_NAME,
        "decoder_version": DECODER_VERSION,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "environment": _environment(),
        "summary": {
            "files": len(jobs),
            "pubs": sum(len(j.pubs) for j in jobs),
            "shots": sum(j.total_shots for j in jobs),
            "status": status_counts,
        },
        "jobs": entries,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="job result .json/.json.gz files or directories")
    ap.add_argument("-o", "--output", default="decoded.json", help="report path (default decoded.json)")
    ap.add_argument("--compare-legacy", action="store_true",
                    help="also record what the header-bug decoder would have produced")
    ap.add_argument("--summary-only", action="store_true", help="omit per-bitstring counts from the report")
    args = ap.parse_args(argv)

    files = _collect(args.paths, exclude=Path(args.output))
    if not files:
        print("no .json or .json.gz files found", file=sys.stderr)
        return 2
    report = build_report(files, args.compare_legacy, args.summary_only)
    Path(args.output).write_text(json.dumps(report, indent=2))

    for j in report["jobs"]:
        print(f"{j['status']:<21} pubs={len(j['pubs']):<4} shots={j['total_shots']:<8} {Path(j['source']).name}")
    s = report["summary"]
    print(f"\n{s['files']} files, {s['pubs']} PUBs, {s['shots']} shots -> {args.output}")
    print("status:", ", ".join(f"{k}={v}" for k, v in sorted(s["status"].items())))
    bad = {DECODER_DISAGREEMENT, ERROR}
    return 1 if any(j["status"] in bad for j in report["jobs"]) else 0


if __name__ == "__main__":
    sys.exit(main())
