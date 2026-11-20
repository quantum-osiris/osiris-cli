"""IDE API 1.6.0: read-only views of governed /code runs for Mission Control.

Runs are produced by the real headless runner with a scripted model; nothing reaches Ollama.
"""

import http.client
import json
import os
import subprocess
import sys
import threading

import pytest

jsonschema = pytest.importorskip("jsonschema")

from osiris_cli import ide_api  # noqa: E402
from osiris_cli.code_runs import RunStore  # noqa: E402
from osiris_cli.code_task import run_task  # noqa: E402

CONTRACT = json.load(open(ide_api.CONTRACT_PATH))
FIXED = "def add(a, b):\n    return a + b\n"
ACCEPT = f"{sys.executable} -m pytest -q -p no:cacheprovider test_calc.py"


def call(name, **args):
    return {"content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def scripted(*steps):
    script = list(steps) + [{"content": "done"}]
    return lambda m, t, model: script.pop(0) if script else {"content": "done"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".osiris" / "code").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    repo = home / "repo"
    repo.mkdir()
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (repo / "test_calc.py").write_text("from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    runs = home / ".osiris" / "code" / "runs"
    receipts = home / ".osiris" / "code" / "receipts.jsonl"
    store = RunStore(str(runs))

    def make(*steps, **kw):
        kw.setdefault("accept", ACCEPT)
        return run_task("fix add in calc.py", root=str(repo), chat=scripted(*steps), out=lambda s: None,
                        store=store, receipts=str(receipts), confirm=lambda p: False, **kw)
    backend = ide_api.Backend(gateway_factory=lambda: None, slice_receipts=str(tmp_path / "slice.jsonl"),
                              examples_dir="", code_runs_dir=str(runs), code_receipts=str(receipts))
    return {"backend": backend, "make": make, "repo": repo, "runs": runs, "receipts": receipts, "home": home}


def check(route, payload):
    jsonschema.validate(payload, CONTRACT["routes"][route]["response"])


def test_empty_store_lists_nothing(tmp_path):
    b = ide_api.Backend(gateway_factory=lambda: None, slice_receipts=str(tmp_path / "s.jsonl"), examples_dir="",
                        code_runs_dir=str(tmp_path / "missing"))
    out = b.code_runs()
    check("GET /api/v1/code/runs", out)
    assert out["count"] == 0 and out["runs"] == []


def test_list_newest_first_without_contents(env):
    ok = env["make"](call("write_file", path="calc.py", content=FIXED), approve_writes=True)
    bad = env["make"](call("write_file", path="test_calc.py", content="def test_add():\n    pass\n"),
                      approve_writes=True)
    out = env["backend"].code_runs()
    check("GET /api/v1/code/runs", out)
    assert [r["run_id"] for r in out["runs"]] == [bad["run_id"], ok["run_id"]]
    first, second = out["runs"]
    assert first["state"] == "BLOCKED" and first["critic"] == "BLOCK" and first["files_changed"] == []
    assert second["state"] == "SUCCEEDED" and second["verification"] == "VERIFIED" and second["exit_code"] == 0
    assert second["files_changed"] == ["calc.py"] and second["root"] == "~/repo"
    blob = json.dumps(out)
    assert FIXED not in blob and str(env["home"]) not in blob and "output_tail" not in blob


def test_detail_has_lifecycle_diff_and_bound_receipts(env):
    s = env["make"](call("write_file", path="calc.py", content=FIXED), approve_writes=True)
    d = env["backend"].code_run({"run_id": s["run_id"]})
    check("POST /api/v1/code/run", d)
    assert [h["state"] for h in d["history"] if "state" in h][-1] == "SUCCEEDED"
    assert d["diffs"][0]["path"] == "calc.py" and "+    return a + b" in d["diffs"][0]["diff"]
    assert [r["kind"] for r in d["receipts"]["rows"]] == ["code.agent.run", "code.agent.apply_write",
                                                           "code.agent.verify"]
    assert d["receipts"]["chain_valid"] is True
    assert d["verification"]["receipt_hash"] == d["receipts"]["rows"][-1]["hash"]
    blob = json.dumps(d)
    assert str(env["home"]) not in blob and "content\"" not in blob and "argv" not in d["verification"]


def test_home_directory_is_not_exposed_in_command_or_output(env):
    store = RunStore(str(env["runs"]))
    s = run_task("x", root=str(env["repo"]), chat=scripted(), out=lambda m: None, store=store,
                 receipts=str(env["receipts"]), accept=f"{sys.executable} -c \"print('{env['home']}/secret/path')\"",
                 require_change=False)
    d = env["backend"].code_run({"run_id": s["run_id"]})
    assert str(env["home"]) not in json.dumps(d), d["verification"]
    assert "~/secret/path" in d["verification"]["output_tail"]


def test_tampered_receipts_report_invalid_chain(env):
    s = env["make"](call("write_file", path="calc.py", content=FIXED), approve_writes=True)
    lines = env["receipts"].read_text().splitlines()
    row = json.loads(lines[1])
    row["payload"]["after_sha256"] = "0" * 64
    lines[1] = json.dumps(row, sort_keys=True)
    env["receipts"].write_text("\n".join(lines) + "\n")
    assert env["backend"].code_run({"run_id": s["run_id"]})["receipts"]["chain_valid"] is False


def test_receipts_outside_allowed_locations_are_not_read(env, tmp_path):
    outside = tmp_path / "elsewhere.jsonl"
    store = RunStore(str(env["runs"]))
    s = run_task("x", root=str(env["repo"]), chat=scripted(call("write_file", path="calc.py", content=FIXED)),
                 out=lambda m: None, store=store, receipts=str(outside), accept=ACCEPT, approve_writes=True)
    d = env["backend"].code_run({"run_id": s["run_id"]})
    assert d["receipts"]["available"] is False and "outside" in d["receipts"]["reason"]


@pytest.mark.parametrize("rid", ["../../etc/passwd", "code-XYZ", "", None, "code-0123456789ab/../x"])
def test_bad_run_ids_rejected(env, rid):
    with pytest.raises(ide_api.ApiError) as e:
        env["backend"].code_run({"run_id": rid})
    assert e.value.status == 400


def test_unknown_run_is_404(env):
    with pytest.raises(ide_api.ApiError) as e:
        env["backend"].code_run({"run_id": "code-000000000000"})
    assert e.value.status == 404


def test_corrupt_record_is_skipped_not_fatal(env):
    env["make"](call("write_file", path="calc.py", content=FIXED), approve_writes=True)
    (env["runs"] / "code-ffffffffffff.json").write_text("{not json")
    assert env["backend"].code_runs()["count"] == 1


def test_routes_over_http_require_token(env):
    httpd, token = ide_api.serve(env["backend"], port=0)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        port = httpd.server_address[1]
        s = env["make"](call("write_file", path="calc.py", content=FIXED), approve_writes=True)

        def req(method, path, body=None, tok=True):
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
            h = {"Host": f"127.0.0.1:{port}"}
            if tok:
                h["X-OSIRIS-Token"] = token
            data = None
            if body is not None:
                data, h["Content-Type"] = json.dumps(body).encode(), "application/json"
            c.request(method, path, body=data, headers=h)
            r = c.getresponse()
            out = json.loads(r.read())
            c.close()
            return r.status, out
        assert req("GET", "/api/v1/code/runs", tok=False)[0] == 401
        st, out = req("GET", "/api/v1/code/runs")
        assert st == 200 and out["api_version"] == "1.6.0" and out["runs"][0]["run_id"] == s["run_id"]
        st, out = req("POST", "/api/v1/code/run", {"run_id": s["run_id"]})
        assert st == 200 and out["state"] == "SUCCEEDED"
    finally:
        httpd.shutdown()
        httpd.server_close()
