"""
osiris_qdecode — corrected decoder for IBM Quantum (Qiskit Runtime & Executor) results.

Replaces the decoding path in analyze_existing_hardware.py.

Supported wire formats
----------------------
1. SamplerV2 ``PrimitiveResult`` JSON (``{"__type__": "PrimitiveResult", ...}``):
   Qiskit Runtime stores each measurement register as a ``BitArray`` whose packed
   uint8 array is serialized as ``base64(zlib(np.save(array)))``. ``np.save`` writes a
   self-describing ``.npy`` header first: the magic bytes ``b"\\x93NUMPY"``, a version,
   and a Python-dict header string space-padded to a 64-byte boundary. Casting the
   decompressed stream with ``np.frombuffer`` treats that header as measurement bytes
   and pushes the same number of genuine shots off the end. This module loads every
   ``.npy`` payload with ``np.load(..., allow_pickle=False)``.
2. IBM Executor ``QuantumProgramResult`` JSON (``schema_version`` in ``v0.1``, ``v0.2``,
   ``v1.0``, ``v1.1``, ``v2.0``):
   Each register in ``data[i].results.<reg>`` carries ``{"data": "<base64>", "shape":
   [..., shots, num_bits], "dtype": "bool"}`` packed with ``bitorder="little"`` (raw
   bytes in ``v0.1``/``v0.2``, ``zlib``-compressed in ``v1.0+``) and reconstructed in
   Qiskit via ``BitArray.from_bool_array(meas_data, order="little")``.

Two independent decode paths (PrimitiveResult)
----------------------------------------------
1. A numpy-only implementation in this file (no Qiskit required).
2. ``qiskit_ibm_runtime.RuntimeDecoder`` + ``BitArray.get_counts()``, the reference.
When Qiskit is installed both run on ``PrimitiveResult`` files and their counts must
agree exactly. A disagreement is reported as ``DECODER_DISAGREEMENT`` and never
silently resolved.

Bit order
---------
Bitstrings follow Qiskit convention: classical bit 0 is the RIGHTMOST
character (``"c_{n-1}...c_1c_0"``). In ``PrimitiveResult`` payloads each packed row is
big-endian with zero padding in the high bits of byte 0. In Executor ``v0.2`` payloads
unpacked axis ``-1`` index ``0`` is ``c_0`` (``order="little"``), so axis ``-1`` is
reversed before packing to Qiskit order.

Usage
-----
    python -m osiris_cli.qdecode JOB.json [MORE.json | DIR | ARCHIVE.zip ...] -o decoded.json
    python -m osiris_cli.qdecode archive/ -o decoded.json --compare-legacy
"""

from __future__ import annotations

import argparse
import base64
import binascii
import datetime as _dt
import gzip
import hashlib
import io
import json
import math
import platform
import sys
import zipfile
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

DECODER_NAME = "osiris_qdecode"
DECODER_VERSION = "1.1.0"
NPY_MAGIC = b"\x93NUMPY"
EXECUTOR_SCHEMA_VERSIONS = ("v0.1", "v0.2", "v1.0", "v1.1", "v2.0")

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
# Format & register classification helpers
# --------------------------------------------------------------------------

def is_executor_v02(payload: Any) -> bool:
    """Return True when ``payload`` is an IBM Executor schema_version=v0.2 document."""
    return isinstance(payload, dict) and payload.get("schema_version") == "v0.2"


def detect_format(payload: Any) -> str:
    """Identify the wire schema of a parsed JSON job result."""
    if not isinstance(payload, dict):
        return "unknown"
    sv = payload.get("schema_version")
    if sv == "v0.2":
        return "executor_v02"
    if sv in EXECUTOR_SCHEMA_VERSIONS and isinstance(payload.get("data"), list):
        return f"executor_{sv.replace('.', '_')}"
    if payload.get("__type__") == "PrimitiveResult" or payload.get("_type") == "PrimitiveResult":
        return "primitive_result"
    if "__type__" in payload:
        return "primitive_result"
    return "unknown"


def is_all_zero_register(counts: dict[str, int]) -> bool:
    """Return True when every recorded shot landed in the all-zero bitstring."""
    if len(counts) != 1:
        return False
    key, shots = next(iter(counts.items()))
    return bool(key) and shots > 0 and set(key) == {"0"}


def classify_register(counts: dict[str, int]) -> str | None:
    """Return ``'ALL_ZERO'`` for unmeasured/all-zero registers, else ``None``."""
    return "ALL_ZERO" if is_all_zero_register(counts) else None


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


def decode_executor_v02_register(register_payload: dict[str, Any]) -> np.ndarray:
    """Decode an IBM Executor ``dtype='bool'`` register payload into Qiskit bit order.

    Wire format (``ibm_quantum_schemas.common.tensor`` + ``executor_sampler``):
      - ``data``: base64-encoded little-endian packed bits (raw in v0.1/v0.2,
        zlib-compressed in v1.0+).
      - ``shape``: ``[..., shots, num_bits]``.
      - ``dtype``: ``"bool"``.

    In IBM's wire array, last-axis index ``0`` is classical bit ``c_0``
    (``BitArray.from_bool_array(meas_data, order="little")``). This function
    reverses the last axis so column ``-1`` is ``c_0`` and column ``0`` is
    ``c_{num_bits-1}``, matching Qiskit's bitstring convention everywhere.
    """
    if not isinstance(register_payload, dict):
        raise DecodeError(f"expected dict register payload, got {type(register_payload).__name__}")
    dtype = register_payload.get("dtype")
    if dtype != "bool":
        raise DecodeError(f"expected bool executor register, got dtype={dtype!r}")
    raw_shape = register_payload.get("shape")
    if not isinstance(raw_shape, (list, tuple)) or len(raw_shape) < 2:
        raise DecodeError(f"expected shape [..., shots, num_bits], got {raw_shape!r}")
    shape = tuple(int(x) for x in raw_shape)
    if any(x < 0 for x in shape):
        raise DecodeError(f"negative dimension in shape {shape}")

    b64_data = register_payload.get("data")
    if not isinstance(b64_data, str):
        raise DecodeError("missing or non-string base64 'data' in executor register")
    try:
        raw_b64 = base64.b64decode(b64_data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise DecodeError(f"corrupt base64 in executor register: {exc}") from exc

    expected_bits = int(math.prod(shape))
    expected_bytes = math.ceil(expected_bits / 8)

    if len(raw_b64) == expected_bytes:
        raw = raw_b64
    else:
        try:
            raw = zlib.decompress(raw_b64)
        except zlib.error:
            raw = raw_b64
        if len(raw) < expected_bytes:
            raise DecodeError(
                f"truncated payload: got {len(raw) * 8} bits ({len(raw)} bytes), "
                f"expected {expected_bits} bits ({expected_bytes} bytes)"
            )
        if len(raw) != expected_bytes:
            raise DecodeError(
                f"executor payload byte length {len(raw)} inconsistent with shape {shape} "
                f"(expected {expected_bytes} bytes)"
            )

    if expected_bits == 0:
        return np.zeros(shape, dtype=np.uint8)

    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")[:expected_bits]
    matrix = bits.reshape(shape)
    # Reverse bit axis so c_0 is rightmost (matching Qiskit BitArray.from_bool_array(..., order="little"))
    return np.ascontiguousarray(matrix[..., ::-1])


def qiskit_bit_matrix_to_counts(matrix: np.ndarray) -> dict[str, int]:
    """Convert a 2D ``(shots, num_bits)`` matrix in Qiskit column order (``col -1 == c_0``) to counts."""
    if matrix.ndim != 2:
        raise DecodeError(f"expected 2D (shots, num_bits) matrix, got shape {matrix.shape}")
    shots, num_bits = matrix.shape
    if shots == 0 or num_bits == 0:
        return {}
    pad = (-num_bits) % 8
    work = np.pad(matrix, ((0, 0), (pad, 0)), mode="constant", constant_values=0) if pad else matrix
    packed = np.packbits(work.astype(np.uint8, copy=False), axis=1, bitorder="big")
    counts, _ = packed_rows_to_counts(packed, num_bits)
    return counts


def decode_executor_v02(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Convenience helper returning flat register dicts for an Executor payload."""
    pubs = _pubs_from_executor(payload)
    out: list[dict[str, Any]] = []
    sv = payload.get("schema_version")
    for pub in pubs:
        for reg in pub.registers:
            out.append({
                "pub_index": pub.index,
                "register": reg.name,
                "loc": list(reg.loc),
                "shots": reg.shots,
                "num_bits": reg.num_bits,
                "unique": reg.unique,
                "counts": reg.counts,
                "flags": list(reg.flags),
                "schema_version": sv,
                "payload_sha256": reg.payload_sha256,
            })
    return out


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
    payload_sha256: str | None = None


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
    wire_format: str = "unknown"
    schema_version: str | None = None
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
    if num_bits and shots and is_all_zero_register(counts):
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
            raw_bytes = base64.standard_b64decode(b64)
            payload_sha = hashlib.sha256(raw_bytes).hexdigest()
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
                    payload_sha256=payload_sha,
                )
                if compare_legacy and loc == () and num_bits and shots:
                    leg = emulate_legacy_decode(b64, num_bits, shots)
                    leg["total_variation_distance"] = total_variation_distance(counts, leg["counts"])
                    leg["legacy_unique"] = len(leg["counts"])
                    rec.legacy = leg
                regs.append(rec)
        out.append(PubRecord(index=i, shape=shape, metadata=pv.get("metadata", {}), registers=regs))
    return out


def _pubs_from_executor(doc: dict[str, Any]) -> list[PubRecord]:
    """Decode IBM Executor QuantumProgramResult (schema_version v0.1..v2.0) into PubRecords."""
    if not isinstance(doc, dict) or doc.get("schema_version") not in EXECUTOR_SCHEMA_VERSIONS:
        raise DecodeError(f"unsupported executor schema_version: {doc.get('schema_version')!r}")
    items = doc.get("data")
    if not isinstance(items, list):
        raise DecodeError("executor payload missing 'data' list")

    out: list[PubRecord] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise DecodeError(f"executor item {i} is not a dict")
        results = item.get("results")
        if not isinstance(results, dict):
            raise DecodeError(f"executor item {i} missing 'results' dict")

        regs: list[RegisterRecord] = []
        pub_shape: list[int] = []
        for name, rv in results.items():
            if not isinstance(rv, dict):
                raise DecodeError(f"executor item {i} register {name!r} is not a dict")
            if rv.get("dtype") in ("f64", "c128", "u8"):
                # Non-bitstring (e.g. kerneled IQ) register: skip classified bitstring extraction
                continue
            matrix = decode_executor_v02_register(rv)
            raw_bytes = base64.b64decode(rv["data"], validate=True)
            payload_sha = hashlib.sha256(raw_bytes).hexdigest()
            sweep_shape = list(matrix.shape[:-2])
            if sweep_shape and not pub_shape:
                pub_shape = sweep_shape
            shots, num_bits = int(matrix.shape[-2]), int(matrix.shape[-1])
            for loc in np.ndindex(*matrix.shape[:-2]):
                sub = np.ascontiguousarray(matrix[loc])
                counts = qiskit_bit_matrix_to_counts(sub)
                rec = RegisterRecord(
                    name=name,
                    loc=list(loc),
                    num_bits=num_bits,
                    shots=shots,
                    unique=len(counts),
                    counts=counts,
                    flags=_register_flags(counts, num_bits, shots, 0),
                    payload_sha256=payload_sha,
                )
                regs.append(rec)
        item_meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        out.append(PubRecord(index=i, shape=pub_shape, metadata=item_meta, registers=regs))
    return out


def _qiskit_counts(text: str) -> dict[tuple, dict[str, int]] | None:
    """Reference path for PrimitiveResult. Returns {(pub, register, loc): counts}, or None if Qiskit is absent."""
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


def _qiskit_executor_counts(text: str) -> dict[tuple, dict[str, int]] | None:
    """Reference path for Executor QuantumProgramResult via ExecutorResultDecoder + BitArray."""
    try:
        from qiskit.primitives.containers import BitArray
        from qiskit_ibm_runtime.decoders.executor.decoder import ExecutorResultDecoder
    except ImportError:
        return None
    qpr = ExecutorResultDecoder.decode(text)
    out: dict[tuple, dict[str, int]] = {}
    for i, item in enumerate(qpr):
        for name, meas_data in item.items():
            if not isinstance(meas_data, np.ndarray) or meas_data.dtype != np.bool_:
                continue
            ba = BitArray.from_bool_array(meas_data, order="little")
            for loc in np.ndindex(*ba.shape):
                out[(i, name, loc)] = dict(ba.get_counts(loc=loc if loc else None))
    return out


# --------------------------------------------------------------------------
# File- and byte-level entry points
# --------------------------------------------------------------------------

def _read_text(path: Path) -> tuple[bytes, str]:
    blob = path.read_bytes()
    raw = gzip.decompress(blob) if blob[:2] == b"\x1f\x8b" else blob
    return blob, raw.decode("utf-8")


def decode_job_bytes(
    blob: bytes,
    source: str = "<memory>",
    compare_legacy: bool = False,
    use_qiskit: bool = True,
) -> JobRecord:
    """Decode raw bytes (plain or gzip-compressed JSON) of a job result."""
    raw = gzip.decompress(blob) if blob[:2] == b"\x1f\x8b" else blob
    job = JobRecord(
        source=source,
        source_sha256=hashlib.sha256(blob).hexdigest(),
        source_bytes=len(blob),
        status=OK,
    )
    try:
        text = raw.decode("utf-8")
        doc = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        job.status = UNRECOGNIZED
        job.warnings.append(f"not JSON: {exc}")
        return job

    fmt = detect_format(doc)
    job.wire_format = fmt
    if isinstance(doc, dict) and isinstance(doc.get("schema_version"), str):
        job.schema_version = doc["schema_version"]

    if fmt == "primitive_result":
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
    elif fmt.startswith("executor_"):
        try:
            job.pubs = _pubs_from_executor(doc)
            job.paths_run.append("executor_numpy")
        except (DecodeError, KeyError, TypeError, ValueError) as exc:
            job.status = ERROR
            job.warnings.append(f"invalid executor payload ({job.schema_version}): {exc}")
            return job

        if use_qiskit:
            try:
                ref_exec = _qiskit_executor_counts(text)
            except Exception as exc:
                ref_exec = None
                job.warnings.append(f"qiskit executor reference decode failed: {exc!r}")
            if ref_exec is not None:
                job.paths_run.append("qiskit_ibm_runtime_executor")
                mine = {
                    (p.index, r.name, tuple(r.loc)): r.counts
                    for p in job.pubs for r in p.registers
                }
                job.paths_agree = mine == ref_exec
                if not job.paths_agree:
                    job.status = DECODER_DISAGREEMENT
                    diff = sorted(set(mine) ^ set(ref_exec)) or [k for k in mine if mine[k] != ref_exec.get(k)]
                    job.warnings.append(
                        f"executor_numpy and qiskit decoders disagree on {len(diff)} register(s): {diff[:5]}"
                    )
                    return job
    else:
        job.status = UNRECOGNIZED
        job.warnings.append("unrecognized JSON schema (neither PrimitiveResult nor Executor)")
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


def decode_job_file(path: str | Path, compare_legacy: bool = False, use_qiskit: bool = True) -> JobRecord:
    """Decode a ``PrimitiveResult`` or Executor ``v0.2`` JSON/gzip result file."""
    path = Path(path)
    blob = path.read_bytes()
    return decode_job_bytes(blob, source=str(path), compare_legacy=compare_legacy, use_qiskit=use_qiskit)


def _collect_inputs(paths: Iterable[str | Path], exclude: Path | None = None) -> list[tuple[str, bytes]]:
    """Collect ``(source_label, raw_bytes)`` from files, directories, and ``.zip`` archives."""
    items: list[tuple[str, bytes]] = []
    excl = exclude.resolve() if exclude is not None else None
    for raw_p in paths:
        p = Path(raw_p)
        if p.is_dir():
            for q in sorted(p.rglob("*")):
                if not q.is_file():
                    continue
                if excl is not None and q.resolve() == excl:
                    continue
                if q.name.endswith(".json") or q.name.endswith(".json.gz"):
                    items.append((str(q), q.read_bytes()))
        elif p.is_file() and p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as zf:
                for name in sorted(zf.namelist()):
                    if name.endswith("-info.json"):
                        continue
                    if name.endswith(".json") or name.endswith(".json.gz"):
                        items.append((f"{p}!{name}", zf.read(name)))
        elif p.is_file():
            if excl is None or p.resolve() != excl:
                items.append((str(p), p.read_bytes()))
    return items


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
    jobs: list[JobRecord] = []
    for f in files:
        if f.is_file() and f.suffix.lower() == ".zip":
            for label, blob in _collect_inputs([f]):
                try:
                    jobs.append(decode_job_bytes(blob, source=label, compare_legacy=compare_legacy))
                except Exception as exc:
                    jobs.append(JobRecord(
                        source=label,
                        source_sha256=hashlib.sha256(blob).hexdigest(),
                        source_bytes=len(blob),
                        status=ERROR,
                        warnings=[f"{type(exc).__name__}: {exc}"],
                    ))
        else:
            jobs.append(_safe_decode(f, compare_legacy))
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
    ap.add_argument("paths", nargs="+", help="job result .json/.json.gz/.zip files or directories")
    ap.add_argument("-o", "--output", default="decoded.json", help="report path (default decoded.json)")
    ap.add_argument("--compare-legacy", action="store_true",
                    help="also record what the header-bug decoder would have produced")
    ap.add_argument("--summary-only", action="store_true", help="omit per-bitstring counts from the report")
    args = ap.parse_args(argv)

    files = _collect(args.paths, exclude=Path(args.output))
    if not files:
        print("no .json, .json.gz, or .zip files found", file=sys.stderr)
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
