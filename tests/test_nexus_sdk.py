"""Offline tests for the qnexus-SDK Nexus transport (osiris_cli/providers/nexus_sdk.py)."""

import inspect
import json
import uuid
from collections import Counter

import pytest

pytket = pytest.importorskip("pytket")
from pytket import Circuit, OpType  # noqa: E402

from osiris_cli.nexus_ledger import NexusAuditLedger  # noqa: E402
from osiris_cli.providers import nexus_sdk as N  # noqa: E402
from osiris_cli.providers.quantinuum_nexus import GovernanceError  # noqa: E402


def _bell(name="bell"):
    c = Circuit(2, 2, name=name)
    c.add_gate(OpType.PhasedX, [0.5, 0.0], [0])
    c.add_gate(OpType.Rz, [0.25], [0])
    c.add_gate(OpType.ZZPhase, [0.5], [0, 1])
    c.add_gate(OpType.PhasedX, [0.5, 0.5], [1])
    c.Measure(0, 0)
    c.Measure(1, 1)
    return c


class _Ref:
    def __init__(self, kind, payload=None):
        self.id = uuid.uuid4()
        self.kind = kind
        self.payload = payload


class _CircRef(_Ref):
    def download_circuit(self):
        return self.payload.copy()


class _CompileItem:
    def __init__(self, ref):
        self._ref = ref

    def get_output(self):
        return self._ref


class _Result:
    def get_counts(self):
        return Counter({(0, 0): 51, (1, 1): 49})


class _BackendInfo:
    def __init__(self, device_name):
        self.device_name = device_name


class _ExecItem:
    def __init__(self, device_name):
        self._device = device_name

    def download_result(self):
        return _Result()

    def download_backend_info(self):
        return _BackendInfo(self._device)


class _Status:
    def __init__(self, status="COMPLETED"):
        self.status = type("E", (), {"value": status})()
        self.completed_time = None
        self.cost = 7.5


class FakeQnexus:
    """Minimal stand-in with the same call shapes as qnexus 0.51.4 (keyword arguments only)."""

    def __init__(self, report_device=None, report_status="COMPLETED"):
        self.projects = self.circuits = self.jobs = self
        self._jobs = {}
        self._devices = {}
        self.report_device = report_device  # simulate Nexus reporting a different device than requested
        self.report_status = report_status

    def QuantinuumConfig(self, *, device_name, **kw):
        return {"type": "QuantinuumConfig", "device_name": device_name, **kw}

    def get_or_create(self, *, name, description=None, properties=None):
        return _Ref("project")

    def upload(self, *, circuit, project=None, name=None, description=None, properties=None):
        return _CircRef("circuit", circuit)

    def start_compile_job(self, *, programs, backend_config, name, project=None, optimisation_level=2, **kw):
        j = _Ref("compile")
        self._jobs[j.id] = [_CompileItem(_CircRef("circuit", r.payload)) for r in programs]
        return j

    def cost(self, *, circuit_ref, n_shots, backend_config, syntax_checker=None, project=None, timeout=None):
        assert syntax_checker == backend_config["device_name"] + "SC"
        assert project is not None
        return float(N.local_hqc(circuit_ref.payload, n_shots))

    def start_execute_job(self, *, programs, n_shots, backend_config, name, project=None, max_cost=(), **kw):
        assert len(max_cost) == len(programs) == len(n_shots)
        j = _Ref("execute")
        dev = self.report_device or backend_config["device_name"]
        self._devices[j.id] = dev
        self._jobs[j.id] = [_ExecItem(dev) for _ in programs]
        return j

    def status(self, *, job, **kw):
        return _Status(self.report_status)

    def get(self, *, id=None, **kw):
        return type("JobRecord", (), {"backend_config_store": type("Cfg", (), {"device_name": self._devices[id]})()})()

    def wait_for(self, *, job, timeout=None, **kw):
        return None

    def results(self, *, job, **kw):
        return self._jobs[job.id]


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setenv("OSIRIS_HOME", str(tmp_path))
    monkeypatch.delenv("OSIRIS_NEXUS_DISPOSITIONS_LOG", raising=False)
    monkeypatch.delenv("OSIRIS_TIER3_HMAC_SECRET", raising=False)
    return N.NexusSdkRunner(qnx=FakeQnexus(), ledger=NexusAuditLedger(tmp_path))


def _pre(runner, **kw):
    return runner.preflight([_bell()], target="H2-1", approve_remote=True, **kw)


def test_local_hqc_counts_rz_as_free():
    # 2 PhasedX + 10 x 1 ZZPhase + 5 x (2 measure + 2 prep) = 32 units; 5 + 500*32/5000 = 8.2
    assert str(N.local_hqc(_bell(), 500)) == "8.2000"


def test_preflight_prices_with_syntax_checker_and_records_preflight_checked(runner):
    res = runner.preflight([_bell("a"), _bell("b")], target="H2-1", shots=500, approve_remote=True)
    assert res.nexus_hqc == [8.2, 8.2]
    assert res.total_nexus_hqc == 16.4
    assert res.compiled_sha256 == res.source_sha256  # fake compile is the identity
    body = runner.ledger.list_entries()[-1]["body"]
    assert body["tier"] == 1 and body["hardware"] is False and body["target"] == "H2-1SC"
    assert body["disposition"] == "PREFLIGHT_CHECKED"
    cost_calls = [kw for name, kw in runner.calls if name == "circuits.cost"]
    assert len(cost_calls) == 2 and all(kw["project"] is not None for kw in cost_calls)
    v = runner.ledger.verify()
    assert v["valid"] is True and v["entries"] == 1, v
    with pytest.raises(GovernanceError):
        runner.preflight([_bell()], target="H2-1E", approve_remote=True)


def test_remote_side_effects_need_operator_approval(runner):
    with pytest.raises(GovernanceError, match="operator approval"):
        runner.preflight([_bell()], target="H2-1")
    assert runner.calls == []  # nothing uploaded, no project created
    pre = _pre(runner)
    n = len(runner.calls)
    with pytest.raises(GovernanceError, match="operator approval"):
        runner.run(pre, target="H2-1E", shots=100)
    assert len(runner.calls) == n  # no emulator job started


def test_emulator_run_records_simulated_and_counts(runner):
    pre = _pre(runner, shots=100)
    out = runner.run(pre, target="H2-1E", shots=100, approve_remote=True)
    assert out["counts"] == [{"00": 51, "11": 49}]
    body = runner.ledger.list_entries()[-1]["body"]
    assert body["tier"] == 2 and body["disposition"] == "SIMULATED" and body["hardware"] is False


def test_emulator_cap_and_hardware_token_fail_closed_before_dispatch(runner):
    pre = _pre(runner, shots=500)
    with pytest.raises(GovernanceError, match="max_emulator_hqc"):
        runner.run(pre, target="H2-1E", max_emulator_hqc=1.0, approve_remote=True)
    with pytest.raises(GovernanceError, match="approval token"):
        runner.run(pre, target="H2-1", approve_remote=True)
    assert not [c for c in runner.calls if c[0] == "start_execute_job"]


def _token(pre, tmp_path, batch=None):
    from osiris_cli.quantinuum_gate import mint_tier3_token
    return mint_tier3_token(circuit_sha256=batch or N.batch_sha256(pre.compiled_sha256), target="H2-1",
                            max_shots=500, max_hqc="20", osiris_home=tmp_path)


def test_hardware_attested_only_after_nexus_confirms_device_and_status(runner, tmp_path):
    pre = _pre(runner, shots=500)
    out = runner.run(pre, target="H2-1", approval_token=_token(pre, tmp_path))
    body = runner.ledger.list_entries()[-1]["body"]
    assert body["disposition"] == "HARDWARE_ATTESTED" and body["hardware"] is True
    assert body["tier"] == 3 and body["job_id"] == out["job_id"]
    att = body["metrics"]["attestation"]
    assert att["job_status"] == "COMPLETED" and att["job_device"] == "H2-1" and att["result_devices"] == ["H2-1"]
    assert {"jobs.status", "jobs.get"} <= {c[0] for c in runner.calls}
    with pytest.raises(Exception):
        runner.run(pre, target="H2-1", approval_token=_token(pre, tmp_path, batch="0" * 64))


@pytest.mark.parametrize("kw, why", [({"report_device": "H2-1E"}, "device"),
                                     ({"report_status": "ERROR"}, "status")])
def test_hardware_run_not_attested_when_nexus_disagrees(tmp_path, monkeypatch, kw, why):
    monkeypatch.setenv("OSIRIS_HOME", str(tmp_path))
    runner = N.NexusSdkRunner(qnx=FakeQnexus(**kw), ledger=NexusAuditLedger(tmp_path))
    pre = _pre(runner, shots=500)
    out = runner.run(pre, target="H2-1", approval_token=_token(pre, tmp_path))
    body = runner.ledger.list_entries()[-1]["body"]
    assert out["disposition"] == body["disposition"] == "HARDWARE_UNVERIFIED"
    assert body["hardware"] is False and any(why in d for d in body["diagnostics"])


def test_load_circuits_reads_xtop005_style_list(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps([{"name": "q0", "tket": _bell().to_dict()}, {"name": "q1", "tket": _bell().to_dict()}]))
    cs = N.load_circuits(p)
    assert [c.name for c in cs] == ["q0", "q1"] and cs[0].n_qubits == 2


def test_every_sdk_call_binds_to_the_real_qnexus_signature(runner, tmp_path):
    """Guards against invented APIs: each recorded call must bind to qnexus 0.51.4's real function."""
    qnx = pytest.importorskip("qnexus")
    pre = _pre(runner, shots=500)
    runner.run(pre, target="H2-1E", shots=100, approve_remote=True)
    runner.run(pre, target="H2-1", approval_token=_token(pre, tmp_path))
    assert {c[0] for c in runner.calls} >= {"projects.get_or_create", "circuits.upload", "start_compile_job",
                                           "jobs.wait_for", "jobs.results", "circuits.cost", "start_execute_job",
                                           "jobs.status", "jobs.get"}
    for dotted, kwargs in runner.calls:
        fn = qnx
        for part in dotted.split("."):
            fn = getattr(fn, part)
        inspect.signature(fn).bind(**kwargs)
    qnx.QuantinuumConfig(device_name="H2-1E")


def test_cli_refuses_without_approval_when_not_interactive(tmp_path, monkeypatch, capsys):
    from osiris_cli import quantinuum_cli as Q
    p = tmp_path / "c.json"
    p.write_text(json.dumps([{"name": "q0", "tket": _bell().to_dict()}]))
    monkeypatch.setattr(N, "load_qnexus", lambda: FakeQnexus())
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
    assert Q.cmd_nexus(["preflight", str(p)]) == 2
    assert "operator approval" in capsys.readouterr().err
