"""Tests for osiris_cli.qdecode / osiris_cli.qscore (recovered from chat-only
home files into tracked code during release/production-reconciliation-20261119).

Covers: .npy-header-aware dual-path decoding (numpy-only vs qiskit_ibm_runtime
RuntimeDecoder must agree exactly), legacy frombuffer defect emulation, padding
flags, decode-error fail-closed behavior, and GHZ population-vs-fidelity scoring.
"""

from __future__ import annotations

import base64
import io
import json
import zlib

import numpy as np
import pytest

from osiris_cli import qdecode, qscore


def _packed_bits(rows: list[str]) -> tuple[np.ndarray, int]:
    """Pack bitstrings (Qiskit order, c0 rightmost) into a (shots, nbytes) uint8 array."""
    nbits = len(rows[0])
    nbytes = (nbits + 7) // 8
    arr = np.zeros((len(rows), nbytes), dtype=np.uint8)
    for i, r in enumerate(rows):
        v = int(r, 2)  # r = "c_{n-1}...c0"; int(...,2) puts c0 at bit 0 (Qiskit convention)
        # BitArray rows are big-endian: byte 0 holds the highest bits, padding in its high bits
        arr[i] = np.frombuffer(v.to_bytes(nbytes, "big"), dtype=np.uint8)
    return arr, nbits


def _b64_npy(packed: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, packed)
    return base64.b64encode(zlib.compress(buf.getvalue())).decode()


def _runtime_job_json(rows: list[str], num_bits: int) -> str:
    """Serialize a real one-PUB sampler result via qiskit's own RuntimeEncoder."""
    from qiskit.primitives.containers import BitArray, DataBin, SamplerPubResult, PrimitiveResult
    from qiskit_ibm_runtime import RuntimeEncoder

    packed, _ = _packed_bits(rows)
    ba = BitArray(packed, num_bits=num_bits)
    pub = SamplerPubResult(data=DataBin(c=ba), metadata={"arm": "test"})
    return json.dumps(PrimitiveResult([pub]), cls=RuntimeEncoder)


# ----------------------------------------------------------------------
# qdecode
# ----------------------------------------------------------------------

def test_dual_path_decode_agrees_on_synthetic_runtime_job():
    rows = (["0000"] * 700 + ["1111"] * 200 + ["0101"] * 124)
    text = _runtime_job_json(rows, num_bits=4)

    doc = json.loads(text)
    pubs = qdecode._pubs_from_raw(doc, compare_legacy=False)
    numpy_counts = pubs[0].registers[0].counts

    ref = qdecode._qiskit_counts(text)
    assert ref is not None, "qiskit_ibm_runtime reference path must be available"
    qiskit_counts = ref[(0, "c", ())]

    assert numpy_counts == qiskit_counts
    assert sum(numpy_counts.values()) == 1024
    assert numpy_counts["0000"] == 700 and numpy_counts["1111"] == 200 and numpy_counts["0101"] == 124


def test_roundtrip_various_widths_and_shots():
    rng = np.random.default_rng(20261119)
    for num_bits, shots in [(1, 8), (3, 100), (8, 512), (13, 1024), (16, 2048)]:
        rows = ["".join(rng.choice(list("01"), size=num_bits)) for _ in range(shots)]
        text = _runtime_job_json(rows, num_bits=num_bits)
        doc = json.loads(text)
        pubs = qdecode._pubs_from_raw(doc, compare_legacy=False)
        got = pubs[0].registers[0].counts
        ref = qdecode._qiskit_counts(text)[(0, "c", ())]
        assert got == ref, f"dual-path disagreement at num_bits={num_bits}"
        expected: dict[str, int] = {}
        for r in rows:
            expected[r] = expected.get(r, 0) + 1
        assert got == expected


def test_legacy_frombuffer_defect_is_detected_not_silent():
    rows = (["00"] * 500 + ["11"] * 500)
    text = _runtime_job_json(rows, num_bits=2)
    doc = json.loads(text)
    pubs = qdecode._pubs_from_raw(doc, compare_legacy=True)
    leg = pubs[0].registers[0].legacy
    assert leg is not None
    # The legacy path misreads the .npy header as shot bytes -> divergent counts
    assert leg["header_bytes"] > 0
    assert leg["total_variation_distance"] > 0.0
    assert leg["counts"] != pubs[0].registers[0].counts


def test_nonzero_padding_bits_flagged():
    packed = np.array([[0b11111111, 0b00000101]], dtype=np.uint8)  # num_bits=9 -> 7 pad bits, top byte nonzero
    b64 = _b64_npy(packed)
    arr = qdecode.load_ndarray_payload(b64)
    counts, bad = qdecode.packed_rows_to_counts(arr, num_bits=9)
    assert bad == 1


def test_decode_errors_fail_closed():
    with pytest.raises(qdecode.DecodeError):
        qdecode.npy_header_length(b"NOTNUMPY....")
    with pytest.raises(qdecode.DecodeError):
        qdecode.packed_rows_to_counts(np.zeros((4, 3), dtype=np.uint8), num_bits=8)  # wrong nbytes
    with pytest.raises(qdecode.DecodeError):
        qdecode.packed_rows_to_counts(np.zeros((4, 1), dtype=np.float64), num_bits=8)  # wrong dtype


# ----------------------------------------------------------------------
# qscore
# ----------------------------------------------------------------------

def _ideal_ghz_settings(n: int = 3, shots: int = 1024) -> dict[str, dict[str, int]]:
    # For each Pauli setting choose an outcome whose parity makes every covered
    # stabilizer term contribute +1 (terms with coefficient -1 need odd parity).
    assert n == 3
    terms = qscore.ghz_stabilizer_terms(n)
    settings: dict[str, dict[str, int]] = {}
    for s in ["ZZZ", "XXX", "XYY", "YXY", "YYX"]:
        covered = [(t, c) for t, c in terms if all(p == "I" or p == q for p, q in zip(t, s)) and set(t) != {"I"}]
        assert covered
        want_odd = all(c < 0 for _, c in covered)
        for v in range(8):
            if (bin(v).count("1") % 2 == 1) == want_odd:
                settings[s] = {format(v, "03b"): shots}
                break
    return settings


def test_ghz_ideal_state_fidelity_is_one():
    res = qscore.ghz_fidelity(_ideal_ghz_settings())
    assert res.fidelity.value == pytest.approx(1.0, abs=1e-12)
    assert res.population == pytest.approx(1.0)
    assert res.witness_sigma > 0


def test_population_is_not_fidelity_for_classical_mixture():
    # 50/50 classical mixture of |000> and |111>: population 1.0, fidelity 0.5
    uniform8 = {format(i, "03b"): 128 for i in range(8)}  # fully mixed in X/Y bases
    settings = {
        "ZZZ": {"000": 512, "111": 512},
        "XXX": dict(uniform8),
        "XYY": dict(uniform8),
        "YXY": dict(uniform8),
        "YYX": dict(uniform8),
    }
    res = qscore.ghz_fidelity(settings)
    assert res.population == pytest.approx(1.0)
    assert res.fidelity.value == pytest.approx(0.5, abs=1e-9)
    pop = qscore.ghz_population(settings["ZZZ"])
    assert pop.value == pytest.approx(1.0)


def test_ghz_fidelity_refuses_when_xy_settings_missing():
    with pytest.raises(ValueError, match="not measured"):
        qscore.ghz_fidelity({"ZZZ": {"000": 100, "111": 100}})


def test_ghz_fidelity_rejects_bad_labels():
    with pytest.raises(ValueError):
        qscore.ghz_fidelity({"ZZQ": {"000": 10}})
    with pytest.raises(ValueError):
        qscore.ghz_fidelity({})
