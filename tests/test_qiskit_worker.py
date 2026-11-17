"""F-34: Qiskit/Aer work runs on one long-lived thread with a bounded backlog.

UNIT tests exercise osiris_cli.qiskit_worker directly. LIVE LOCAL tests start the real IDE API on a
loopback port and run the real dnalang compiler and Aer simulator through /api/v1/slice/run; one
test also runs the service as a subprocess and restarts it. No store, no network, no hardware.
"""
import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time

import pytest

pytest.importorskip("dnalang")
pytest.importorskip("qiskit_aer")
pytest.importorskip("osiris_governance.github_policy")

from osiris_cli import ide_api  # noqa: E402
from osiris_cli.qiskit_worker import Busy, QiskitWorker  # noqa: E402
from osiris_cli.trio_authz import Receipts  # noqa: E402
from osiris_cli.vertical_slice import run_slice  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BELL = open(os.path.join(ROOT, "docs", "dnalang", "templates", "bell.dna")).read()
BROKEN = "organism Broken { genome { nothing_here() @ [0]; } }"


# ───────────────────────── UNIT ─────────────────────────
def test_successive_callers_share_one_thread():
    w = QiskitWorker(2)
    seen = []
    for _ in range(4):
        t = threading.Thread(target=lambda: seen.append(w.submit(threading.get_ident)))
        t.start(); t.join()
    assert len(set(seen)) == 1 and threading.get_ident() not in seen
    w.shutdown()


def test_exception_propagates_and_worker_survives():
    w = QiskitWorker(1)
    with pytest.raises(ZeroDivisionError):
        w.submit(lambda: 1 / 0)
    assert w.submit(lambda: 42) == 42
    assert w.stats()["in_flight"] == 0
    w.shutdown()


def test_backlog_is_bounded_and_slots_return():
    w = QiskitWorker(2)
    gate = threading.Event()
    out = []
    callers = [threading.Thread(target=lambda: out.append(w.submit(lambda: gate.wait(5) and "ran"))) for _ in range(2)]
    for c in callers:
        c.start()
    deadline = time.time() + 5
    while w.stats()["in_flight"] < 2 and time.time() < deadline:
        time.sleep(0.01)
    with pytest.raises(Busy):
        w.submit(lambda: "third")              # refused at once, nothing queued
    gate.set()
    for c in callers:
        c.join()
    assert out == ["ran", "ran"] and w.stats()["rejected_busy"] == 1
    assert w.submit(lambda: "after") == "after"
    w.shutdown()


def test_abandoned_call_still_completes_and_frees_its_slot():
    w = QiskitWorker(1)
    done = threading.Event()
    def slow():
        time.sleep(0.3); done.set(); return "late"
    t = threading.Thread(target=lambda: w.submit(slow), daemon=True)
    t.start()                                   # this caller "gives up": nobody reads its result
    time.sleep(0.05)
    with pytest.raises(Busy):
        w.submit(lambda: "x")
    assert done.wait(2)
    time.sleep(0.05)
    assert w.submit(lambda: "next") == "next"
    w.shutdown()


def test_shutdown_waits_for_admitted_work():
    w = QiskitWorker(2)
    finished = []
    t = threading.Thread(target=lambda: w.submit(lambda: (time.sleep(0.2), finished.append(1))))
    t.start(); time.sleep(0.05)
    w.shutdown()
    assert finished == [1]


def test_every_qiskit_route_is_routed_through_the_worker():
    names = set(ide_api.ROUTES.values())
    assert set(ide_api.QISKIT_OPERATIONS) <= names
    assert {"compile", "run_slice", "organism", "files_run", "dnalang_info"} <= ide_api.QISKIT_OPERATIONS
    src = open(ide_api.__file__).read()
    assert "compile_fn=lambda src: on_qiskit_thread(" in src      # assist compiles proposals on the worker


# ───────────────────────── LIVE LOCAL: real server, real Aer ─────────────────────────
@pytest.fixture()
def api(tmp_path, monkeypatch):
    def start(max_pending=4, runner=None):
        monkeypatch.setattr(ide_api, "QISKIT_WORKER", QiskitWorker(max_pending))
        backend = ide_api.Backend(gateway_factory=lambda: None, slice_receipts=str(tmp_path / "slice.jsonl"),
                                  examples_dir=str(tmp_path), slice_runner=runner)
        httpd, token = ide_api.serve(backend, port=0, console_dir=None, token_file=str(tmp_path / "tok"))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        started.append(httpd)
        return {"port": httpd.server_address[1], "token": token, "receipts": str(tmp_path / "slice.jsonl")}
    started = []
    yield start
    for h in started:
        h.shutdown(); h.server_close()


def post(s, path, body, timeout=120):
    c = http.client.HTTPConnection("127.0.0.1", s["port"], timeout=timeout)
    c.request("POST", path, body=json.dumps(body), headers={"Host": f"127.0.0.1:{s['port']}", "X-OSIRIS-Token": s["token"],
                                                           "Content-Type": "application/json"})
    r = c.getresponse(); data = json.loads(r.read()); c.close()
    return r.status, data


def receipts_by_run(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.strip():
                e = json.loads(line); out[e["payload"]["run_id"]] = e
    return out


def test_sequential_simulations_from_different_request_threads(api):
    s = api()
    runs = [post(s, "/api/v1/slice/run", {"source": BELL, "shots": 100 + i, "seed": i}) for i in range(4)]
    assert all(st == 200 and r["status"] == "COMPLETED" for st, r in runs)
    rec = receipts_by_run(s["receipts"])
    for i, (_, r) in enumerate(runs):
        assert sum(r["simulation"]["counts"].values()) == 100 + i
        assert rec[r["run_id"]]["payload"]["shots"] == 100 + i and rec[r["run_id"]]["hash"] == r["receipt"]["hash"]
    assert Receipts.verify(s["receipts"]) is True


def test_concurrent_clients_each_get_their_own_result(api):
    s = api(max_pending=8)
    results = {}
    def client(i):
        results[i] = post(s, "/api/v1/slice/run", {"source": BELL, "shots": 200 + i, "seed": 10 + i})
    ts = [threading.Thread(target=client, args=(i,)) for i in range(6)]
    for t in ts: t.start()
    for t in ts: t.join()
    rec = receipts_by_run(s["receipts"])
    assert len(rec) == 6 and Receipts.verify(s["receipts"]) is True
    for i, (st, r) in results.items():
        assert st == 200 and r["status"] == "COMPLETED"
        assert sum(r["simulation"]["counts"].values()) == 200 + i and r["requested"]["seed"] == 10 + i
        assert rec[r["run_id"]]["payload"]["shots"] == 200 + i and rec[r["run_id"]]["payload"]["seed"] == 10 + i


def test_overload_is_refused_not_queued(api):
    def slow(*a, **k):
        time.sleep(0.4); return run_slice(*a, **k)
    s = api(max_pending=1, runner=slow)
    results = []
    ts = [threading.Thread(target=lambda: results.append(post(s, "/api/v1/slice/run", {"source": BELL, "shots": 50}))) for _ in range(4)]
    for t in ts: t.start()
    for t in ts: t.join()
    codes = sorted(st for st, _ in results)
    busy = [r for st, r in results if st == 503]
    assert codes.count(200) >= 1 and busy and all(r["error"]["kind"] == "SIMULATOR_BUSY" for r in busy)
    assert len(receipts_by_run(s["receipts"])) == codes.count(200)
    assert post(s, "/api/v1/slice/run", {"source": BELL, "shots": 50})[0] == 200


def test_failed_simulation_then_successful_one(api):
    calls = {"n": 0}
    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated backend fault")
        return run_slice(*a, **k)
    s = api(runner=flaky)
    st, rejected = post(s, "/api/v1/slice/run", {"source": BROKEN})
    assert st == 200 and rejected["status"] == "REJECTED" and "receipt" not in rejected
    st, err = post(s, "/api/v1/slice/run", {"source": BELL, "shots": 64})
    assert st == 500 and err["error"]["kind"] == "INTERNAL"
    st, ok = post(s, "/api/v1/slice/run", {"source": BELL, "shots": 64})
    assert st == 200 and ok["status"] == "COMPLETED"
    assert list(receipts_by_run(s["receipts"])) == [ok["run_id"]]


def test_client_timeout_then_next_request(api):
    def slow(*a, **k):
        time.sleep(0.8); return run_slice(*a, **k)
    s = api(max_pending=2, runner=slow)
    with pytest.raises((socket.timeout, TimeoutError)):
        post(s, "/api/v1/slice/run", {"source": BELL, "shots": 32}, timeout=0.2)
    st, r = post(s, "/api/v1/slice/run", {"source": BELL, "shots": 33})
    assert st == 200 and r["status"] == "COMPLETED"
    rec = receipts_by_run(s["receipts"])
    # the abandoned run was admitted, so it finished and was receipted; its result went nowhere
    assert sorted(e["payload"]["shots"] for e in rec.values()) == [32, 33] and Receipts.verify(s["receipts"]) is True


def _free_port():
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0)); return so.getsockname()[1]


def test_service_restart_keeps_one_valid_receipt_chain(tmp_path):
    receipts, tok = str(tmp_path / "slice.jsonl"), str(tmp_path / "tok")
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    run_ids = []
    for round_ in range(2):
        port = _free_port()
        p = subprocess.Popen([sys.executable, "-m", "osiris_cli.ide_api", "--port", str(port), "--slice-receipts", receipts,
                              "--github-receipts", str(tmp_path / "gh.jsonl"), "--token-file", tok, "--examples-dir", str(tmp_path)],
                             cwd=str(tmp_path), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for _ in range(200):
                try:
                    socket.create_connection(("127.0.0.1", port), timeout=0.2).close(); break
                except OSError:
                    time.sleep(0.1)
            s = {"port": port, "token": open(tok).read().strip()}
            for i in range(2):                      # two runs per process: the F-34 crash needed exactly this
                st, r = post(s, "/api/v1/slice/run", {"source": BELL, "shots": 40 + round_ * 10 + i})
                assert st == 200 and r["status"] == "COMPLETED", r
                run_ids.append(r["run_id"])
            assert p.poll() is None, "service exited during the runs"
        finally:
            p.send_signal(2)                        # SIGINT: the KeyboardInterrupt path shuts the worker down
            p.wait(30)
    assert list(receipts_by_run(receipts)) == run_ids and Receipts.verify(receipts) is True
