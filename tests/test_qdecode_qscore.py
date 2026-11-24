"""Tests for osiris_cli.qdecode / osiris_cli.qscore and Phase 1 hardware provenance.

Covers:
  - .npy-header-aware dual-path decoding (numpy-only vs qiskit_ibm_runtime RuntimeDecoder)
  - IBM Executor schema_version="v0.2" decoding, Qiskit bit-order parity, truncation,
    corrupt base64, and multi-register payloads
  - ALL_ZERO unmeasured-register classification & filtering
  - Finite-shot entropy ceilings min(num_bits, log2(shots)) & SATURATED_ENTROPY flagging
    (including 127Q/4096-shot and 156Q/100k-shot saturation regressions)
  - Loader SKIP_DIRS hardening, removal of synthetic fallback (_generate_synthetic_counts),
    quarantined anomalies_week4.csv write protection, and 59/59 repo job decoding
"""

from __future__ import annotations

import base64
import io
import json
import math
import sys
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pytest

from osiris_cli import qdecode, qscore

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


def load_fixture(name: str):
    with (FIXTURE_DIR / name).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def runtime_decoder_counts(payload: dict) -> dict[str, int]:
    """Decode an Executor v0.2 fixture through Qiskit's reference ExecutorResultDecoder + BitArray."""
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit.primitives.containers import BitArray
    from qiskit_ibm_runtime.decoders.executor.decoder import ExecutorResultDecoder

    qpr = ExecutorResultDecoder.decode(json.dumps(payload))
    first_item = qpr[0]
    reg_name = next(iter(first_item.keys()))
    ba = BitArray.from_bool_array(first_item[reg_name], order="little")
    return dict(ba.get_counts())


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


def _executor_v02_b64_from_qiskit_bitstrings(rows: list[str]) -> tuple[str, list[int]]:
    """Encode Qiskit-order bitstrings ('c_{n-1}...c_0') into IBM Executor v0.2 wire format."""
    shots = len(rows)
    nbits = len(rows[0])
    # Wire order has index 0 == c_0 (rightmost character of Qiskit string)
    little_cols = np.array([[c == "1" for c in r[::-1]] for r in rows], dtype=np.uint8)
    packed = np.packbits(little_cols.reshape(-1), bitorder="little")
    return base64.b64encode(packed.tobytes()).decode("ascii"), [shots, nbits]


def _runtime_job_json(rows: list[str], num_bits: int) -> str:
    """Serialize a real one-PUB sampler result via qiskit's own RuntimeEncoder."""
    pytest.importorskip("qiskit", reason="needs qiskit to build a real sampler result")
    pytest.importorskip("qiskit_ibm_runtime", reason="needs qiskit-ibm-runtime's RuntimeEncoder; "
                        "CI does not install qiskit-ibm-runtime")
    from qiskit.primitives.containers import BitArray, DataBin, SamplerPubResult, PrimitiveResult
    from qiskit_ibm_runtime import RuntimeEncoder

    packed, _ = _packed_bits(rows)
    ba = BitArray(packed, num_bits=num_bits)
    pub = SamplerPubResult(data=DataBin(c=ba), metadata={"arm": "test"})
    return json.dumps(PrimitiveResult([pub]), cls=RuntimeEncoder)


# ----------------------------------------------------------------------
# qdecode: PrimitiveResult
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
# qdecode: Executor schema_version="v0.2" & ALL_ZERO classification
# ----------------------------------------------------------------------

def test_executor_v02_decode():
    rows = ["0000"] * 6 + ["1111"] * 4
    b64, shape = _executor_v02_b64_from_qiskit_bitstrings(rows)
    payload = {
        "schema_version": "v0.2",
        "data": [{"results": {"c": {"data": b64, "shape": shape, "dtype": "bool"}}, "metadata": {}}],
        "metadata": {},
        "passthrough_data": {"post_processor": {"context": "sampler_v2", "version": "v1"}},
    }
    assert qdecode.detect_format(payload) == "executor_v02"
    assert qdecode.is_executor_v02(payload)

    decoded = qdecode.decode_executor_v02(payload)
    assert len(decoded) == 1
    assert decoded[0]["shots"] == 10
    assert decoded[0]["num_bits"] == 4
    assert decoded[0]["counts"] == {"0000": 6, "1111": 4}
    assert decoded[0]["payload_sha256"] and len(decoded[0]["payload_sha256"]) == 64

    jr = qdecode.decode_job_bytes(json.dumps(payload).encode("utf-8"), source="mem_v02.json")
    assert jr.status == qdecode.OK
    assert jr.wire_format == "executor_v02"
    assert jr.schema_version == "v0.2"
    assert jr.counts(0, "c") == {"0000": 6, "1111": 4}


def test_executor_v02_bit_order():
    """Guard against c_0 <-> c_{n-1} reversal on asymmetric multi-bit outcomes."""
    # Shot 0: c0=1, c1=0, c2=0 -> Qiskit string "001"
    # Shot 1: c0=0, c1=1, c2=1 -> Qiskit string "110"
    # Shot 2: c0=1, c1=0, c2=0 -> Qiskit string "001"
    rows = ["001", "110", "001"]
    b64, shape = _executor_v02_b64_from_qiskit_bitstrings(rows)
    mat = qdecode.decode_executor_v02_register({"data": b64, "shape": shape, "dtype": "bool"})
    assert mat.shape == (3, 3)
    counts = qdecode.qiskit_bit_matrix_to_counts(mat)
    assert counts == {"001": 2, "110": 1}
    assert "100" not in counts and "011" not in counts

    # Cross-check directly against Qiskit's BitArray.from_bool_array(..., order="little") when installed
    try:
        from qiskit.primitives.containers import BitArray
    except ImportError:
        pass
    else:
        raw_bytes = base64.b64decode(b64)
        wire_bool = np.unpackbits(np.frombuffer(raw_bytes, dtype=np.uint8), bitorder="little")[:9].astype(bool).reshape(3, 3)
        ref_counts = dict(BitArray.from_bool_array(wire_bool, order="little").get_counts())
        assert counts == ref_counts


def test_executor_v02_truncated_payload():
    # 100 shots x 4 bits = 400 bits (50 bytes); supply only 10 bytes
    short_b64 = base64.b64encode(b"\x0f" * 10).decode("ascii")
    with pytest.raises(qdecode.DecodeError, match="truncated payload"):
        qdecode.decode_executor_v02_register({"data": short_b64, "shape": [100, 4], "dtype": "bool"})


def test_executor_v02_multiple_registers():
    b64_a, shape_a = _executor_v02_b64_from_qiskit_bitstrings(["01", "01", "10", "00"])
    b64_b, shape_b = _executor_v02_b64_from_qiskit_bitstrings(["111", "000", "111", "101"])
    payload = {
        "schema_version": "v0.2",
        "data": [{
            "results": {
                "meas": {"data": b64_a, "shape": shape_a, "dtype": "bool"},
                "synd": {"data": b64_b, "shape": shape_b, "dtype": "bool"},
            },
            "metadata": {},
        }],
    }
    jr = qdecode.decode_job_bytes(json.dumps(payload).encode("utf-8"))
    assert jr.status == qdecode.OK
    assert jr.total_shots == 4  # registers in one PUB share shots
    assert jr.counts(0, "meas") == {"01": 2, "10": 1, "00": 1}
    assert jr.counts(0, "synd") == {"111": 2, "000": 1, "101": 1}


def test_executor_v02_corrupt_base64():
    with pytest.raises(qdecode.DecodeError, match="corrupt base64"):
        qdecode.decode_executor_v02_register({
            "data": "!!!not-valid-base64@@@",
            "shape": [8, 2],
            "dtype": "bool",
        })


def test_all_zero_filter():
    assert qdecode.classify_register({"0" * 84: 2048}) == "ALL_ZERO"
    assert qdecode.is_all_zero_register({"000": 2048})
    assert qdecode.classify_register({"000": 2047, "001": 1}) is None

    b64_unmeas, shape_u = _executor_v02_b64_from_qiskit_bitstrings(["000000000"] * 32)
    payload = {
        "schema_version": "v0.2",
        "data": [{"results": {"c": {"data": b64_unmeas, "shape": shape_u, "dtype": "bool"}}}],
    }
    jr = qdecode.decode_job_bytes(json.dumps(payload).encode("utf-8"))
    assert jr.status == qdecode.SUSPECT_UNMEASURED
    assert "ALL_ZERO" in jr.pubs[0].registers[0].flags

    es = qscore.score_entropy({"0" * 84: 2048}, num_bits=84)
    assert qscore.ALL_ZERO in es.flags
    assert qscore.UNMEASURED_REGISTER in es.flags


# ----------------------------------------------------------------------
# qscore: Finite-shot ceilings, saturation & GHZ fidelity
# ----------------------------------------------------------------------

def test_entropy_ceiling_127q_4096():
    ceil_h = qscore.entropy_ceiling(127, 4096)
    assert ceil_h == pytest.approx(12.0, abs=1e-12)
    # 4,096 unique 127-bit outcomes -> H = 12.0 bits = 100% of finite-shot ceiling
    unique_counts = {format(i, "0127b"): 1 for i in range(4096)}
    es = qscore.score_entropy(unique_counts, num_bits=127)
    assert es.shannon_entropy == pytest.approx(12.0, abs=1e-12)
    assert es.ceiling_ratio == pytest.approx(1.0, abs=1e-12)
    assert qscore.SATURATED_ENTROPY in es.flags


def test_entropy_ceiling_156q_100k():
    ceil_h = qscore.entropy_ceiling(156, 100_000)
    assert ceil_h == pytest.approx(math.log2(100_000), rel=1e-12)
    assert ceil_h < 16.61


def test_saturated_entropy_flag():
    # 3 qubits, 2048 shots, uniform across all 8 states -> H = 3.0 / ceiling 3.0
    counts = {format(i, "03b"): 256 for i in range(8)}
    es = qscore.score_entropy(counts, num_bits=3)
    assert es.ceiling == pytest.approx(3.0)
    assert es.ceiling_ratio == pytest.approx(1.0)
    assert qscore.SATURATED_ENTROPY in es.flags
    assert "ENTROPY_SUPPRESSION" not in es.flags


def test_156q_finite_shot_saturation():
    """Forensic regression against 156Q ibm_fez executor v0.2 job d77uq7jc6das739h8k50."""
    H_forensic = 16.5051
    ceiling = qscore.entropy_ceiling(156, 100_000)
    assert H_forensic / ceiling > 0.99

    wl_zip = REPO_ROOT / "workloads.zip"
    if wl_zip.is_file():
        with zipfile.ZipFile(wl_zip) as zf:
            blob = zf.read("job-d77uq7jc6das739h8k50-result.json")
        jr = qdecode.decode_job_bytes(blob, source="job-d77uq7jc6das739h8k50-result.json")
        assert jr.status == qdecode.OK
        assert jr.wire_format == "executor_v02"
        reg = jr.pubs[0].registers[0]
        assert (reg.shots, reg.num_bits) == (100_000, 156)
        es = qscore.score_entropy(reg.counts, num_bits=156)
        assert es.shannon_entropy == pytest.approx(16.5051, abs=0.02)
        assert es.ceiling_ratio > 0.99
        assert qscore.SATURATED_ENTROPY in es.flags
        assert "ENTROPY_SUPPRESSION" not in es.flags


def _ideal_ghz_settings(n: int = 3, shots: int = 1024) -> dict[str, dict[str, int]]:
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
    uniform8 = {format(i, "03b"): 128 for i in range(8)}
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


# ----------------------------------------------------------------------
# Phase 1 loader & fail-closed provenance regressions
# ----------------------------------------------------------------------

def test_loader_skips_venv_and_phase1_executor_fails_closed(tmp_path: Path):
    p1_dir = REPO_ROOT / "quantum_discovery" / "phase1_analysis"
    if str(p1_dir) not in sys.path:
        sys.path.insert(0, str(p1_dir))
    from anomaly_detector import AnomalyDetector
    from phase1_executor import Phase1Executor
    from quantum_data_loader import QuantumDataLoader

    # 1. Fake backend file under venv/.../fake_provider must be ignored
    fake_dir = tmp_path / "venv" / "lib" / "site-packages" / "fake_provider"
    fake_dir.mkdir(parents=True)
    (fake_dir / "props_fake.json").write_text(json.dumps({"__type__": "PrimitiveResult", "counts": {"00": 10}}))

    loader = QuantumDataLoader(tmp_path)
    audit = loader.audit_filesystem()
    assert audit["potential_job_results"] == []

    # 2. Phase1Executor must not have _generate_synthetic_counts and must fail closed
    assert not hasattr(Phase1Executor, "_generate_synthetic_counts")
    executor = Phase1Executor(workspace_root=tmp_path, report_path=str(tmp_path / "hw_report.csv"))
    with pytest.raises(RuntimeError, match="No valid measured hardware circuits loaded"):
        executor.full_analysis()

    # 3. Quarantined anomalies_week4.csv cannot be overwritten, and legacy suppression fails closed
    det = AnomalyDetector()
    with pytest.raises(RuntimeError, match="ENTROPY_SUPPRESSION_Z29"):
        det.detect_entropy_suppression({"c0": 2.0})
    with pytest.raises(ValueError, match="quarantined"):
        det.export_flags(str(tmp_path / "anomalies_week4.csv"))


def test_all_59_repo_hardware_jobs_decode_and_filter_all_zero():
    raw_dir = REPO_ROOT / "data" / "raw_jobs"
    wl_zip = REPO_ROOT / "workloads.zip"
    if not (raw_dir.is_dir() and wl_zip.is_file()):
        pytest.skip("repository hardware archives not present in this checkout")

    prim_ok = 0
    exec_ok = 0
    all_zero_count = 0
    valid_count = 0

    for f in sorted(raw_dir.glob("job-*-result.json")):
        jr = qdecode.decode_job_file(f, use_qiskit=False)
        assert jr.status == qdecode.OK
        prim_ok += 1
        for p in jr.pubs:
            for r in p.registers:
                if "ALL_ZERO" in r.flags:
                    all_zero_count += 1
                else:
                    valid_count += 1

    with zipfile.ZipFile(wl_zip) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith("-result.json"):
                continue
            jr = qdecode.decode_job_bytes(zf.read(name), source=name, use_qiskit=False)
            assert jr.status == qdecode.OK
            if jr.wire_format == "executor_v02":
                exec_ok += 1
            else:
                prim_ok += 1
            for p in jr.pubs:
                for r in p.registers:
                    if "ALL_ZERO" in r.flags:
                        all_zero_count += 1
                    else:
                        valid_count += 1

    assert prim_ok == 51
    assert exec_ok == 8
    assert prim_ok + exec_ok == 59
    assert all_zero_count == 24
    assert valid_count == 559


# ----------------------------------------------------------------------
# Permanent Executor v0.2 fixtures, Qiskit parity & provenance guards
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "fixture_name",
    [
        "executor_v02_known_counts.json",
        "executor_v02_asymmetric.json",
        "executor_v02_qiskit_parity.json",
    ],
)
def test_decoder_fixture_metadata(fixture_name: str):
    payload = load_fixture(fixture_name)
    assert payload["fixture_version"] == 1
    assert payload["schema_version"] == "v0.2"
    assert payload["wire_format"] == "executor_v02"
    assert payload["bit_order"] == "little"
    assert payload.get("purpose")


def test_executor_v02_known_counts():
    payload = load_fixture("executor_v02_known_counts.json")
    expected = load_fixture("executor_v02_known_counts.expected.json")
    decoded = qdecode.decode_executor_v02(payload)
    assert len(decoded) == 1
    assert decoded[0]["shots"] == 8
    assert decoded[0]["num_bits"] == 4
    assert decoded[0]["counts"] == expected


def test_executor_v02_asymmetric_bit_order():
    payload = load_fixture("executor_v02_asymmetric.json")
    expected = load_fixture("executor_v02_asymmetric.expected.json")
    decoded = qdecode.decode_executor_v02(payload)
    counts = decoded[0]["counts"]
    assert counts == expected
    assert counts.get("01", 0) == 5
    assert counts.get("10", 0) == 0


@pytest.mark.parametrize(
    ("fixture_name", "expected_name"),
    [
        ("executor_v02_known_counts.json", "executor_v02_known_counts.expected.json"),
        ("executor_v02_asymmetric.json", "executor_v02_asymmetric.expected.json"),
        ("executor_v02_qiskit_parity.json", "executor_v02_qiskit_parity.expected.json"),
    ],
)
def test_executor_v02_qiskit_parity(fixture_name: str, expected_name: str):
    payload = load_fixture(fixture_name)
    expected = load_fixture(expected_name)

    ours = qdecode.decode_executor_v02(payload)[0]["counts"]
    assert ours == expected

    theirs = runtime_decoder_counts(payload)
    assert ours == theirs

    # Also verify decode_job_bytes dual-path agreement flag when Qiskit is installed
    jr = qdecode.decode_job_bytes(json.dumps(payload).encode("utf-8"), source=fixture_name, use_qiskit=True)
    assert jr.status == qdecode.OK
    assert jr.paths_agree is True
    assert "qiskit_ibm_runtime_executor" in jr.paths_run


def test_entropy_saturation_156q():
    ceiling = qscore.entropy_ceiling(156, 100_000)
    ratio = 16.505144 / ceiling
    assert ratio > 0.99


def test_no_synthetic_fallback_artifacts():
    p1_dir = REPO_ROOT / "quantum_discovery" / "phase1_analysis"
    checked = sorted(p1_dir.glob("*.py"))
    assert checked

    forbidden = [
        "_generate_synthetic_counts",
        "demo_0",
        "demo_19",
        "baseline_mean = 5.0",
        "baseline_std = 0.1",
    ]
    for path in checked:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"Forbidden synthetic fallback token {token!r} found in {path.name}"


def test_hardware_provenance_manifest_export(tmp_path: Path):
    p1_dir = REPO_ROOT / "quantum_discovery" / "phase1_analysis"
    if str(p1_dir) not in sys.path:
        sys.path.insert(0, str(p1_dir))
    from quantum_data_loader import QuantumDataLoader

    loader = QuantumDataLoader(FIXTURE_DIR)
    loader.load_job_result(str(FIXTURE_DIR / "executor_v02_qiskit_parity.json"))
    out_file = tmp_path / "HARDWARE_PROVENANCE_MANIFEST.json"
    manifest = loader.export_provenance_manifest(str(out_file))

    assert out_file.is_file()
    assert manifest["total_measured_registers"] == 1
    rec = manifest["registers"][0]
    assert rec["wire_format"] == "executor_v02"
    assert rec["schema_version"] == "v0.2"
    assert rec["shots"] == 32
    assert rec["num_bits"] == 10
    assert rec["payload_sha256"] and len(rec["payload_sha256"]) == 64
    assert rec["ceiling"] == pytest.approx(5.0)
