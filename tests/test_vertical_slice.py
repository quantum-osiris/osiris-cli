"""DNA-Lang → compile → simulate → evaluate → governance → receipt, on real modules (no mocked physics)."""

import json

import pytest

pytest.importorskip("dnalang")
pytest.importorskip("qiskit_aer")
pytest.importorskip("osiris_governance.github_policy")

from osiris_cli import vertical_slice as vs  # noqa: E402
from osiris_cli.trio_authz import Receipts  # noqa: E402

BELL = """organism Bell {
  gene bell_pair() on a, b { helix a; bond a, b; cleave a -> ca; cleave b -> cb; }
  genome { bell_pair() @ [0, 1]; }
}"""


@pytest.fixture
def receipts(tmp_path):
    return str(tmp_path / "slice.jsonl")


def test_bell_end_to_end(receipts):
    r = vs.run_slice(BELL, shots=1000, seed=11, receipts_path=receipts)
    assert r["status"] == "COMPLETED" and r["evidence_class"] == "SIMULATED" and r["hardware"] is False
    assert [s["name"] for s in r["stages"]] == ["compile", "admission", "simulate", "evaluate", "verify"]
    counts = r["simulation"]["counts"]
    assert sum(counts.values()) == 1000 and set(counts) <= {"00", "11"}
    assert r["evaluation"]["exact_distribution"] == {"00": pytest.approx(0.5), "11": pytest.approx(0.5)}
    g = r["governance"]
    assert g["all_matched"] and g["epistemic_status"] == "SIMULATED" and g["evidence_class"] == "SIMULATED"
    assert r["claim"]["category"] == "SIMULATOR_VALIDATED"
    assert r["receipt"]["chain_valid"] and Receipts.verify(receipts)


def test_same_seed_same_counts(receipts):
    a = vs.run_slice(BELL, shots=500, seed=3, receipts_path=receipts)["simulation"]["counts"]
    b = vs.run_slice(BELL, shots=500, seed=3, receipts_path=receipts)["simulation"]["counts"]
    assert a == b


@pytest.mark.parametrize("src,stage", [
    ("organism { ", "parse"),
    ("organism X { gene g() on a { helix a; } genome { nope() @ [0]; } }", "check"),
    ("organism X { gene g() on a { frobnicate a; } genome { g() @ [0]; } }", None),
])
def test_invalid_dnalang_is_rejected_and_nothing_executes(receipts, src, stage):
    r = vs.run_slice(src, receipts_path=receipts)
    assert r["status"] == "REJECTED"
    errs = r["compile"]["diagnostics"]["errors"]
    assert errs and (stage is None or errs[0]["stage"] == stage)
    assert "simulation" not in r and "receipt" not in r
    import os
    assert not os.path.exists(receipts)


def test_parse_errors_carry_positions():
    errs = vs.compile_source("organism { ")["diagnostics"]["errors"]
    assert errs[0]["line"] == 1 and errs[0]["column"] >= 1


@pytest.mark.parametrize("kw", [{"shots": 0}, {"shots": 10**6}, {"shots": True}, {"seed": -1}, {"shots": 1.5}])
def test_bad_run_parameters(receipts, kw):
    with pytest.raises(vs.SliceInputError):
        vs.run_slice(BELL, receipts_path=receipts, **kw)


def test_oversized_and_non_string_source():
    with pytest.raises(vs.SliceInputError):
        vs.compile_source("x" * (vs.MAX_SOURCE_BYTES + 1))
    with pytest.raises(vs.SliceInputError):
        vs.compile_source(None)


def test_simulator_failure_is_a_receipted_negative_result(receipts):
    def broken(circuits, shots, seed):
        raise RuntimeError("Aer backend crashed")
    r = vs.run_slice(BELL, shots=100, seed=1, receipts_path=receipts, simulator=broken)
    assert r["status"] == "FAILED" and r["stages"][-1] == {**r["stages"][-1], "name": "simulate", "status": "FAILED"}
    assert r["receipt"]["kind"] == "slice.failed" and r["receipt"]["chain_valid"]
    row = json.loads(open(receipts).read().splitlines()[-1])
    assert row["payload"]["failed_stage"] == "simulate" and "crashed" in row["payload"]["error"]


def test_simulator_returning_wrong_shot_total_fails(receipts):
    r = vs.run_slice(BELL, shots=100, seed=1, receipts_path=receipts,
                     simulator=lambda c, shots, seed: [{"00": 7}])
    assert r["status"] == "FAILED" and "expected 100" in r["stages"][-1]["error"]


def test_governance_catches_an_inflated_evaluation(receipts, monkeypatch):
    real = vs.evaluate

    def lying(counts, circ, seed):
        out = real(counts, circ, seed)
        out["claimed_metrics"]["survival_fidelity"] = 0.99
        return out
    monkeypatch.setattr(vs, "evaluate", lying)
    r = vs.run_slice(BELL, shots=1000, seed=2, receipts_path=receipts)
    assert r["governance"]["all_matched"] is False and r["governance"]["epistemic_status"] == "REFUTED"
    assert r["claim"]["category"] == "NOT_SUPPORTED"


def test_evidence_receipt_integrity(receipts):
    for seed in (1, 2, 3):
        vs.run_slice(BELL, shots=200, seed=seed, receipts_path=receipts)
    lines = open(receipts).read().splitlines()
    assert Receipts.verify(receipts)
    row = json.loads(lines[1]); row["payload"]["counts"] = {"00": 200}
    open(receipts, "w").write("\n".join([lines[0], json.dumps(row, sort_keys=True), lines[2]]) + "\n")
    assert Receipts.verify(receipts) is False
    open(receipts, "w").write("\n".join([lines[0], lines[2]]) + "\n")
    assert Receipts.verify(receipts) is False


def test_exact_distribution_refused_for_reused_measured_qubit():
    src = """organism M { gene g() on a { helix a; cleave a -> c0; helix a; cleave a -> c1; } genome { g() @ [0]; } }"""
    comp = vs.compile_source(src)
    assert comp["ok"]
    assert vs.exact_distribution(comp["_circuit"]) is None


def test_shipped_examples_validate(receipts):
    import os
    import dnalang
    ex = os.path.join(os.path.dirname(os.path.dirname(dnalang.__file__)), "examples")
    if not os.path.isdir(ex):
        pytest.skip("dnalang-core examples not installed")
    for name in sorted(os.listdir(ex)):
        if name.endswith(".dna"):
            r = vs.run_slice(open(os.path.join(ex, name)).read(), shots=1000, seed=5, receipts_path=receipts)
            assert r["status"] == "COMPLETED", name
            assert r["governance"]["all_matched"], name
