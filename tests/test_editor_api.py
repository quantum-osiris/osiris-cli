"""DNA-Lang editor routes of OSIRIS IDE API 1.3.0.

Two kinds of test, kept apart on purpose:

  LOCAL INTEGRATION (fixture ``stack``): osiris_cli.editor_store and the API talk over HTTP to
      supabase/local_stack/server.mjs, which runs the real migrations in an in-process PostgreSQL.
      Ownership, cross-user denial and revision conflicts are decided by the real policies and
      trigger. The sign-in transport is a stand-in and no email is sent. Skipped when Node or
      PGlite is not installed. Nothing here contacts the hosted Supabase project.
  MOCKED: an injected simulator or model function; named ``*_mocked``.
"""

import hashlib
import json
import os
import socket
import subprocess
import time

import pytest

pytest.importorskip("dnalang")
pytest.importorskip("osiris_governance.github_policy")
jsonschema = pytest.importorskip("jsonschema")

from osiris_cli import editor_assist, ide_api  # noqa: E402
from osiris_cli.editor_store import EditorStore, target_status, valid_path  # noqa: E402
from osiris_cli.vertical_slice import compile_source, run_slice  # noqa: E402
from osiris_cli.workspace_store import StoreError  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACT = json.load(open(ide_api.CONTRACT_PATH))
TOOLCHAINS = os.path.expanduser("~/osiris-assessment/toolchains")
NODE = os.environ.get("OSIRIS_NODE", os.path.join(TOOLCHAINS, "node24", "bin", "node"))
PGLITE = os.environ.get("OSIRIS_PGLITE_NODE_MODULES", os.path.join(TOOLCHAINS, "pglite", "node_modules"))
BELL = open(os.path.join(ROOT, "docs", "dnalang", "templates", "bell.dna")).read()
LEGACY = """# DNA::}{::LANG Organism Definition
ORGANISM BellStateOrganism {
    META { habitat: "ibm_brisbane" consciousness_target: 0.7734 }
    GENOME EntangledPair { CHROMOSOME qubits: 2 }
    EVOLUTION { generations: 100 }
}
result = BellStateOrganism.express()
"""


def check(route, payload):
    jsonschema.validate(payload, CONTRACT["routes"][route]["response"])


# ───────────────────────── compiler: supported template, no fallback ─────────────────────────
def test_supported_template_compiles_to_the_bell_circuit():
    r = compile_source(BELL)
    assert r["ok"] and r["diagnostics"]["errors"] == []
    assert r["compiler"]["name"] == "dnalang" and r["compiler"]["version"] == __import__("dnalang").__version__
    c = r["circuit"]
    assert (c["n_qubits"], c["n_cbits"], c["two_qubit"]) == (2, 2, 1)
    assert "h q[0];" in c["qasm3"] and "cx q[0], q[1];" in c["qasm3"]
    assert r["source_sha256"] == "sha256:" + hashlib.sha256(BELL.encode()).hexdigest()


@pytest.mark.parametrize("source", [
    LEGACY,
    "",
    "print('hello')",
    "organism X { gene g() on a { frobnicate a; } genome { g() @ [0]; } }",
    "organism X { gene g() on a { helix a; } }",                       # no genome
    "organism X { evolution { generations: 100 } genome { } }",        # unsupported block
    "organism X { gene g() on a, b { bond a, b; } genome { g() @ [0]; } }",   # arity
])
def test_unsupported_source_fails_and_yields_no_circuit(source):
    r = compile_source(source)
    assert r["ok"] is False and r["diagnostics"]["errors"]
    assert "circuit" not in r and "_circuit" not in r      # nothing was lowered: no default H/CX circuit


def test_unsupported_source_is_never_simulated(tmp_path):
    called = []
    r = run_slice(LEGACY, shots=100, receipts_path=str(tmp_path / "r.jsonl"),
                  simulator=lambda *a, **k: called.append(1) or [{"00": 100}])
    assert r["status"] == "REJECTED" and not called
    assert "simulation" not in r and "receipt" not in r and not (tmp_path / "r.jsonl").exists()


def test_legacy_diagnostic_carries_the_compilers_own_position():
    e = compile_source(LEGACY)["diagnostics"]["errors"][0]
    assert (e["stage"], e["line"], e["column"]) == ("parse", 1, 1) and "'#'" in e["message"]
    # a message without a position gets no invented line
    e = compile_source("organism X { gene g() on a { helix a; } genome { nope() @ [0]; } }")["diagnostics"]["errors"][0]
    assert set(e) - {"stage", "message"} <= {"line", "column"}


def test_metadata_is_reported_as_not_compiled():
    src = BELL.replace("meta {", 'meta { habitat: "ibm_brisbane",', 1)
    with_meta, plain = compile_source(src), compile_source(BELL)
    assert with_meta["ok"] and with_meta["circuit"]["sha256"] == plain["circuit"]["sha256"]
    notes = [n for n in with_meta["diagnostics"]["notes"] if n["code"] == "metadata-not-compiled"]
    assert notes and "meta.habitat" in notes[0]["message"]


def test_simulator_limits_are_enforced_before_running(tmp_path, monkeypatch):
    from osiris_cli import vertical_slice as vs
    monkeypatch.setattr(vs, "MAX_OPS", 3)
    called = []
    r = run_slice(BELL, shots=10, receipts_path=str(tmp_path / "r.jsonl"), simulator=lambda *a, **k: called.append(1))
    assert r["status"] == "REJECTED" and not called and "operations" in r["stages"][-1]["error"]
    with pytest.raises(vs.SliceInputError):
        run_slice(BELL, shots=vs.MAX_SHOTS + 1, receipts_path=str(tmp_path / "r.jsonl"))


# ───────────────────────── run results (real Aer simulator, no store) ─────────────────────────
def test_real_simulation_counts_sum_to_the_requested_shots(tmp_path):
    r = run_slice(BELL, shots=777, seed=5, receipts_path=str(tmp_path / "r.jsonl"))
    assert r["status"] == "COMPLETED" and r["evidence_class"] == "SIMULATED" and r["hardware"] is False
    assert sum(r["simulation"]["counts"].values()) == 777 and set(r["simulation"]["counts"]) <= {"00", "11"}
    assert r["requested"] == {"operation": "simulate", "shots": 777, "seed": 5}


def test_simulator_failure_is_reported_as_failure_mocked(tmp_path):
    def broken(circuits, shots, seed):
        raise RuntimeError("backend exploded")
    r = run_slice(BELL, shots=50, receipts_path=str(tmp_path / "r.jsonl"), simulator=broken)
    assert r["status"] == "FAILED" and "simulation" not in r and "claim" not in r and "evaluation" not in r
    assert r["receipt"]["kind"] == "slice.failed"
    short = run_slice(BELL, shots=50, receipts_path=str(tmp_path / "r.jsonl"), simulator=lambda *a, **k: [{"00": 49}])
    assert short["status"] == "FAILED" and "simulation" not in short       # wrong shot total is not a result


def test_a_valid_receipt_is_not_a_scientific_claim(tmp_path):
    r = run_slice(BELL, shots=200, receipts_path=str(tmp_path / "r.jsonl"))
    assert r["receipt"]["chain_valid"] is True
    assert r["claim"]["category"] in ("SIMULATOR_VALIDATED", "NOT_SUPPORTED")
    assert "hardware" in r["claim"]["not_claimed"]
    # The metric named classical_fidelity compares count distributions; no state fidelity is produced.
    assert "state_fidelity" not in json.dumps(r) and "consciousness" not in json.dumps(r).lower()
    failed = run_slice(BELL, shots=50, receipts_path=str(tmp_path / "r.jsonl"),
                       simulator=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert failed["receipt"]["chain_valid"] is True and "claim" not in failed   # chain valid, nothing established


# ───────────────────────── service description and unavailable states ─────────────────────────
def backend(tmp_path, **kw):
    return ide_api.Backend(gateway_factory=lambda: None, slice_receipts=str(tmp_path / "slice.jsonl"),
                           examples_dir=str(tmp_path), **kw)


def test_info_reports_hardware_evolution_and_queue_as_unavailable(tmp_path, monkeypatch):
    for k in ("OSIRIS_SUPABASE_URL", "OSIRIS_SUPABASE_ANON_KEY", "OSIRIS_SUPABASE_LOCAL_URL", "OSIRIS_SUPABASE_LOCAL_ANON_KEY"):
        monkeypatch.delenv(k, raising=False)
    info = backend(tmp_path).dnalang_info()
    check("GET /api/v1/dnalang/info", info)
    assert info["hardware"]["available"] is False and info["operations"]["evolve"] is False
    assert info["queue"] == {**info["queue"], "supported": False, "cancel_supported": False}
    assert info["files"]["configured"] is False and info["files"]["reason"] == "NOT_CONFIGURED"
    assert not any("hardware" in p or "qpu" in p or "ibm" in p for _, p in ide_api.ROUTES)


def test_unconfigured_store_is_unavailable_not_a_fake_save(tmp_path, monkeypatch):
    for k in ("OSIRIS_SUPABASE_URL", "OSIRIS_SUPABASE_ANON_KEY", "OSIRIS_SUPABASE_LOCAL_URL", "OSIRIS_SUPABASE_LOCAL_ANON_KEY"):
        monkeypatch.delenv(k, raising=False)
    b = backend(tmp_path)
    for call, body in ((b.files_save, {"user_token": "t" * 40, "file_id": "x", "base_revision": 1, "content": ""}),
                       (b.files_projects, {"user_token": "t" * 40}), (b.auth_request_code, {"email": "a@example.test"})):
        with pytest.raises(ide_api.ApiError) as e:
            call(body)
        assert e.value.status == 503 and e.value.kind == "EDITOR_NOT_CONFIGURED"


def test_store_target_rules(monkeypatch):
    monkeypatch.setenv("OSIRIS_SUPABASE_LOCAL_URL", "http://192.168.1.5:54321")      # not loopback
    monkeypatch.setenv("OSIRIS_SUPABASE_LOCAL_ANON_KEY", "k")
    assert target_status()["reason"] == "BAD_CONFIG"
    monkeypatch.delenv("OSIRIS_SUPABASE_LOCAL_URL")
    monkeypatch.setenv("OSIRIS_SUPABASE_URL", "https://abcdefghijklmnopqrst.supabase.co")   # another project
    monkeypatch.setenv("OSIRIS_SUPABASE_ANON_KEY", "k")
    assert target_status()["reason"] == "WRONG_PROJECT"


def test_store_failure_is_an_error_not_a_saved_state_mocked():
    def down(method, url, headers, body):
        raise StoreError("UNAVAILABLE", "ConnectionRefusedError")
    s = EditorStore("http://127.0.0.1:1", "anon", http=down)
    with pytest.raises(StoreError) as e:
        s.save_file("t" * 40, "00000000-0000-4000-8000-000000000001", 1, "x")
    assert e.value.code == "UNAVAILABLE"

    def lying(method, url, headers, body):     # answers 200 with a row that does not confirm the write
        return [{"id": "00000000-0000-4000-8000-000000000001", "revision": 2, "content_sha256": "0" * 64}]
    with pytest.raises(StoreError) as e:
        EditorStore("http://127.0.0.1:1", "anon", http=lying, verify_session=False).save_file("t" * 40, "00000000-0000-4000-8000-000000000001", 1, "x")
    assert e.value.code == "BAD_RESPONSE"


def test_store_never_sends_an_owner_and_never_uses_a_service_key_mocked():
    seen = []

    def http(method, url, headers, body):
        seen.append((method, url, headers, body))
        return [{"id": "00000000-0000-4000-8000-000000000001", "name": "p"}]
    s = EditorStore("http://127.0.0.1:1", "anon-key", http=http, verify_session=False)
    s.create_project("user-token-" + "t" * 30, "p")
    method, url, headers, body = seen[0]
    assert json.loads(body) == {"name": "p"} and headers["apikey"] == "anon-key"
    assert headers["Authorization"] == "Bearer user-token-" + "t" * 30
    assert "SERVICE" not in open(os.path.join(ROOT, "osiris_cli", "editor_store.py")).read().replace("service-role", "")


@pytest.mark.parametrize("path,ok", [
    ("bell.dna", True), ("src/organisms/bell.dna", True), ("a_b-c.1/d.dna", True),
    ("../etc/passwd", False), ("/abs.dna", False), ("a/../b.dna", False), ("a//b.dna", False), ("a\\b.dna", False),
    (".hidden", False), ("a/.git/config", False), ("", False), ("a b.dna", False), ("x" * 201, False), ("a..b", False),
    (None, False), ("a/", False), ("~/x", False), ("a;rm -rf.dna", False), ("%2e%2e/x", False),
])
def test_path_rules(path, ok):
    assert valid_path(path) is ok


# ───────────────────────── assistance (mocked model) ─────────────────────────
def fake_route(text, **record):
    def route(role, system, prompt, **kw):
        return text, {"policy": "osiris-model-routes/v1", "role": role, "status": "ANSWERED", "runtime": "ollama",
                      "requested_model": "gemma3:4b", "resolved_model": "gemma3:4b", "resolved_digest": "abc",
                      "left_device": False, "cloud_allowed": kw["cloud_allowed"], "attempts": [], **record}
    return route


def test_assist_proposal_is_returned_as_data_and_applies_nothing_mocked(tmp_path):
    changed = BELL.replace("helix a;", "helix a;\n    x b;")
    b = backend(tmp_path, assist_route_fn=fake_route(f"Adds an X.\n```dna\n{changed}```"))
    r = b.assist({"action": "propose_change", "source": BELL, "instruction": "add an x on b"})
    check("POST /api/v1/assist", r)
    assert r["applied"] is False and r["proposed_source"].strip() == changed.strip() and r["proposal_compiles"] is True
    assert r["route"]["resolved_model"] == "gemma3:4b" and r["route"]["left_device"] is False
    assert not os.listdir(tmp_path)                       # nothing was written or run


def test_assist_reports_a_proposal_the_compiler_rejects_mocked(tmp_path):
    b = backend(tmp_path, assist_route_fn=fake_route("Here.\n```\nORGANISM X { EVOLUTION { } }\n```"))
    r = b.assist({"action": "generate_example", "instruction": "bell"})
    assert r["proposal_compiles"] is False and r["proposal_diagnostics"]["errors"]


def test_assist_cloud_needs_a_literal_true_and_unavailable_is_honest_mocked(tmp_path):
    seen = []

    def route(role, system, prompt, **kw):
        seen.append(kw["cloud_allowed"])
        from osiris_cli.orchestration.model_router import RouteUnavailable
        raise RouteUnavailable("none", {"status": "MODEL_UNAVAILABLE", "attempts": [], "left_device": False})
    b = backend(tmp_path, assist_route_fn=route)
    for value in (None, "true", 1, True):
        r = b.assist({"action": "explain_selection", "source": BELL, "selection": "helix a;", "allow_cloud": value})
        assert r["status"] == "UNAVAILABLE" and r["text"] is None and r["applied"] is False
    assert seen == [False, False, False, True]
    with pytest.raises(ide_api.ApiError):
        b.assist({"action": "run_shell", "source": BELL})
    assert editor_assist.extract_source("no code here") is None


# ───────────────────────── LOCAL INTEGRATION: real policies, stand-in transport ─────────────────────────
@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    if not os.path.isfile(NODE) or not os.path.isdir(os.path.join(PGLITE, "@electric-sql", "pglite")):
        pytest.skip("Node or PGlite is not installed (OSIRIS_NODE / OSIRIS_PGLITE_NODE_MODULES)")
    state = tmp_path_factory.mktemp("stack")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen([NODE, os.path.join(ROOT, "supabase", "local_stack", "server.mjs"), "--port", str(port),
                             "--state", str(state), "--memory", "--pglite", PGLITE,
                             "--migrations", os.path.join(ROOT, "supabase", "migrations")],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 60
    while time.time() < deadline and not (state / "anon.key").exists():
        if proc.poll() is not None:
            pytest.fail("local stack exited: " + proc.stdout.read())
        time.sleep(0.2)
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
            break
        except OSError:
            time.sleep(0.2)
    store = EditorStore(f"http://127.0.0.1:{port}", (state / "anon.key").read_text(), target="local-stand-in")

    def sign_in(email):
        store.request_code(email)
        code = (state / "outbox.log").read_text().strip().splitlines()[-1].rsplit("code=", 1)[1]
        return store.verify_code(email, code)
    tmp = tmp_path_factory.mktemp("api")
    api = ide_api.Backend(gateway_factory=lambda: None, slice_receipts=str(tmp / "slice.jsonl"), examples_dir=str(tmp),
                          editor_store_factory=lambda: store)
    yield {"store": store, "sign_in": sign_in, "api": api, "tmp": tmp, "state": state}
    proc.terminate()


def test_sign_in_issues_a_session_and_a_wrong_code_does_not(stack):
    s = stack["store"]
    a = stack["sign_in"]("alice@example.test")
    assert s.user(a["access_token"])["email"] == "alice@example.test"
    s.request_code("mallory@example.test")
    with pytest.raises(StoreError) as e:
        s.verify_code("mallory@example.test", "000000" if not (stack["state"] / "outbox.log").read_text().endswith("code=000000\n") else "111111")
    assert e.value.code == "CODE_REJECTED"
    renewed = s.refresh(a["refresh_token"])
    assert renewed["user"]["id"] == a["user"]["id"]
    with pytest.raises(StoreError):
        s.refresh(a["refresh_token"])                    # a refresh token works once


def test_owner_saves_and_reopens_a_file(stack):
    s, a = stack["store"], stack["sign_in"]("alice@example.test")["access_token"]
    p = s.create_project(a, "Bell work")
    assert p["owner_id"] == s.user(a)["id"]
    f = s.create_file(a, p["id"], "src/bell.dna", BELL)
    assert f["revision"] == 1 and f["content_sha256"] == hashlib.sha256(BELL.encode()).hexdigest()
    edited = BELL + "// edited\n"
    saved = s.save_file(a, f["id"], 1, edited)
    assert saved["status"] == "SAVED" and saved["file"]["revision"] == 2
    reopened = s.file(a, f["id"])
    assert reopened["content"] == edited and reopened["revision"] == 2 and reopened["path"] == "src/bell.dna"
    assert [x["path"] for x in s.files(a, p["id"])] == ["src/bell.dna"]
    check("POST /api/v1/files/open", {"file": reopened})


def test_stale_revision_is_a_conflict_and_does_not_overwrite(stack):
    s, a = stack["store"], stack["sign_in"]("alice@example.test")["access_token"]
    p = s.create_project(a, "Conflict")
    f = s.create_file(a, p["id"], "c.dna", "one")
    assert s.save_file(a, f["id"], 1, "two")["status"] == "SAVED"          # tab 1
    stale = s.save_file(a, f["id"], 1, "from a stale tab")                 # tab 2 still holds revision 1
    assert stale["status"] == "CONFLICT" and stale["current"]["revision"] == 2
    assert s.file(a, f["id"])["content"] == "two"
    check("POST /api/v1/files/save", stale)


def test_another_user_cannot_read_or_change_the_file(stack):
    s = stack["store"]
    a = stack["sign_in"]("alice@example.test")["access_token"]
    b = stack["sign_in"]("bob@example.test")["access_token"]
    p = s.create_project(a, "Private")
    f = s.create_file(a, p["id"], "secret.dna", "alice's text")
    assert p["id"] not in [x["id"] for x in s.projects(b)] and s.files(b, p["id"]) == []
    for attempt in (lambda: s.file(b, f["id"]), lambda: s.save_file(b, f["id"], 1, "bob was here"),
                    lambda: s.delete_file(b, f["id"])):
        with pytest.raises(StoreError) as e:
            attempt()
        assert e.value.code == "NOT_FOUND"
    with pytest.raises(StoreError) as e:
        s.create_file(b, p["id"], "planted.dna", "x")
    assert e.value.code == "FORBIDDEN"
    still = s.file(a, f["id"])
    assert still["content"] == "alice's text" and still["revision"] == 1


def test_owner_and_server_columns_cannot_be_forged(stack):
    s = stack["store"]
    a, b = (stack["sign_in"](e) for e in ("alice@example.test", "bob@example.test"))
    raw = lambda method, path, tok, body=None: s._rows(method, path, tok, body)   # what a hostile client could send
    with pytest.raises(StoreError) as e:                      # a project created in someone else's name
        raw("POST", "/osiris_projects", b["access_token"], {"name": "x", "owner_id": a["user"]["id"]})
    assert e.value.code == "FORBIDDEN"
    p = s.create_project(a["access_token"], "Forge")
    for tok in (a["access_token"], b["access_token"]):        # owner_id is not a writable column for anyone
        with pytest.raises(StoreError) as e:
            raw("PATCH", f"/osiris_projects?id=eq.{p['id']}", tok, {"owner_id": b["user"]["id"]})
        assert e.value.code == "FORBIDDEN"
    assert raw("PATCH", f"/osiris_projects?id=eq.{p['id']}", b["access_token"], {"name": "taken"}) == []   # rename: owner only
    assert s.projects(a["access_token"])[0]["owner_id"] == a["user"]["id"]
    f = raw("POST", "/osiris_files", a["access_token"], {"project_id": p["id"], "path": "f.dna", "content": "abc",
                                                          "revision": 99, "content_sha256": "f" * 64})[0]
    assert f["revision"] == 1 and f["content_sha256"] == hashlib.sha256(b"abc").hexdigest()
    bumped = raw("PATCH", f"/osiris_files?id=eq.{f['id']}", a["access_token"], {"revision": 500})[0]
    assert bumped["revision"] == 2
    other = s.create_project(a["access_token"], "Other")
    with pytest.raises(StoreError):                           # a file cannot be moved to another project
        raw("PATCH", f"/osiris_files?id=eq.{f['id']}", a["access_token"], {"project_id": other["id"]})


def test_paths_are_unique_per_project_and_limits_hold(stack):
    s, a = stack["store"], stack["sign_in"]("alice@example.test")["access_token"]
    p, q = s.create_project(a, "U1"), s.create_project(a, "U2")
    s.create_file(a, p["id"], "same.dna", "")
    with pytest.raises(StoreError) as e:
        s.create_file(a, p["id"], "same.dna", "")
    assert e.value.code == "PATH_EXISTS"
    s.create_file(a, q["id"], "same.dna", "")                 # the same path in another project is fine
    with pytest.raises(StoreError) as e:
        s.create_file(a, p["id"], "big.dna", "x" * 65537)
    assert e.value.code == "TOO_LARGE"
    with pytest.raises(StoreError):                           # the database enforces the limits too
        s._rows("POST", "/osiris_files", a, {"project_id": p["id"], "path": "big.dna", "content": "x" * 65537})
    with pytest.raises(StoreError):
        s._rows("POST", "/osiris_files", a, {"project_id": p["id"], "path": "../up.dna", "content": ""})


def test_delete_removes_the_file_for_good(stack):
    s, a = stack["store"], stack["sign_in"]("alice@example.test")["access_token"]
    p = s.create_project(a, "Del")
    f = s.create_file(a, p["id"], "gone.dna", "x")
    assert s.delete_file(a, f["id"])["deleted"]["path"] == "gone.dna"
    with pytest.raises(StoreError) as e:
        s.file(a, f["id"])
    assert e.value.code == "NOT_FOUND" and s.files(a, p["id"]) == []


def test_missing_or_ended_session_cannot_use_protected_operations(stack):
    s, api = stack["store"], stack["api"]
    session = stack["sign_in"]("carol@example.test")
    tok = session["access_token"]
    p = s.create_project(tok, "Session")
    f = s.create_file(tok, p["id"], "s.dna", BELL)
    protected = [(api.files_projects, {}), (api.files_list, {"project_id": p["id"]}), (api.files_open, {"file_id": f["id"]}),
                 (api.files_save, {"file_id": f["id"], "base_revision": 1, "content": "x"}),
                 (api.files_create, {"project_id": p["id"], "path": "n.dna"}), (api.files_delete, {"file_id": f["id"]}),
                 (api.files_run, {"file_id": f["id"], "revision": 1}), (api.files_runs, {"file_id": f["id"]}),
                 (api.auth_user, {})]
    for bad in (None, "", "not-a-token", "x" * 60, s._key):       # s._key: the publishable key alone is not a user
        for fn, body in protected:
            with pytest.raises(ide_api.ApiError) as e:
                fn({**body, **({} if bad is None else {"user_token": bad})})
            assert e.value.status in (401, 403, 404), (fn.__name__, bad, e.value.status)
    assert s.file(tok, f["id"])["revision"] == 1 and not os.path.exists(api.slice_receipts)
    s.sign_out(tok)
    for attempt in (lambda: s.user(tok), lambda: s.file(tok, f["id"]), lambda: s.save_file(tok, f["id"], 1, "after sign-out"),
                    lambda: s.projects(tok)):
        with pytest.raises(StoreError) as e:          # the token has not expired, but its session has ended
            attempt()
        assert e.value.code == "UNAUTHORIZED"
    with pytest.raises(ide_api.ApiError) as e:
        api.files_run({"user_token": tok, "file_id": f["id"], "revision": 1})
    assert e.value.status == 401 and not os.path.exists(api.slice_receipts)


def test_run_is_bound_to_the_stored_revision_it_executed(stack):
    s, api = stack["store"], stack["api"]
    tok = stack["sign_in"]("dave@example.test")["access_token"]
    p = s.create_project(tok, "Runs")
    f = s.create_file(tok, p["id"], "bell.dna", BELL)
    # the request names a revision; source text in the request is ignored
    r = api.files_run({"user_token": tok, "file_id": f["id"], "revision": 1, "shots": 512, "seed": 3,
                       "source": "organism Evil { }"})
    check("POST /api/v1/files/run", r)
    digest = "sha256:" + hashlib.sha256(BELL.encode()).hexdigest()
    assert r["status"] == "COMPLETED" and r["executed"] is True
    assert r["source_binding"] == {"project_id": p["id"], "file_id": f["id"], "path": "bell.dna", "revision": 1,
                                   "content_sha256": digest}
    assert r["compile"]["source_sha256"] == digest and sum(r["simulation"]["counts"].values()) == 512
    assert r["versions"]["dnalang"] == r["compile"]["compiler"]["version"]

    # the user edits and saves while looking at the old result: the old run still names revision 1
    three = BELL.replace("bell_pair() @ [0, 1];", "bell_pair() @ [0, 2];")
    assert s.save_file(tok, f["id"], 1, three)["status"] == "SAVED"
    stale = api.files_run({"user_token": tok, "file_id": f["id"], "revision": 1})
    assert stale["status"] == "REVISION_MISMATCH" and stale["executed"] is False and "simulation" not in stale
    r2 = api.files_run({"user_token": tok, "file_id": f["id"], "revision": 2, "shots": 100})
    assert r2["source_binding"]["revision"] == 2 and r2["compile"]["circuit"]["n_qubits"] == 3
    assert r2["compile"]["source_sha256"] == "sha256:" + hashlib.sha256(three.encode()).hexdigest()

    runs = api.files_runs({"user_token": tok, "file_id": f["id"]})
    check("POST /api/v1/files/runs", runs)
    assert [x["source_binding"]["revision"] for x in runs["runs"]] == [1, 2] and runs["chain_valid"] is True
    assert runs["runs"][0]["counts"] == r["simulation"]["counts"] and runs["runs"][0]["receipt_hash"] == r["receipt"]["hash"]
    other = stack["sign_in"]("erin@example.test")["access_token"]
    for fn in (api.files_run, api.files_runs):               # another user can neither run nor list them
        with pytest.raises(ide_api.ApiError) as e:
            fn({"user_token": other, "file_id": f["id"], "revision": 2})
        assert e.value.status == 404


def test_run_of_a_stored_file_that_does_not_compile_executes_nothing(stack):
    s, api = stack["store"], stack["api"]
    tok = stack["sign_in"]("dave@example.test")["access_token"]
    f = s.create_file(tok, s.create_project(tok, "Bad")["id"], "legacy.dna", LEGACY)
    r = api.files_run({"user_token": tok, "file_id": f["id"], "revision": 1})
    assert r["status"] == "REJECTED" and r["executed"] is False and "simulation" not in r and "receipt" not in r


# ───────────────────────── IDE API 1.4.0: notebook cells ─────────────────────────
def _notebook(*cells):
    return json.dumps({"format": "osiris-notebook", "version": 1, "title": "t",
                       "cells": [{"id": i, "type": t, "source": s} for i, t, s in cells]}, sort_keys=True)


def test_notebook_cell_run_is_bound_to_notebook_revision_and_cell(stack):
    s, api = stack["store"], stack["api"]
    tok = stack["sign_in"]("nora@example.test")["access_token"]
    p = s.create_project(tok, "Notebook")
    text = _notebook(("intro", "markdown", "# Bell"), ("bell", "dnalang", BELL))
    f = s.create_file(tok, p["id"], "notebooks/bell.dnanb", text)
    r = api.notebook_run({"user_token": tok, "file_id": f["id"], "revision": 1, "cell_id": "bell", "shots": 400,
                          "seed": 5, "source": "organism Evil { }"})
    check("POST /api/v1/notebook/run", r)
    assert r["status"] == "COMPLETED" and r["executed"] is True and sum(r["simulation"]["counts"].values()) == 400
    b = r["source_binding"]
    assert b["cell_id"] == "bell" and b["revision"] == 1 and b["notebook_format"] == "osiris-notebook/1"
    assert b["cell_source_sha256"] == "sha256:" + hashlib.sha256(BELL.encode()).hexdigest() == r["compile"]["source_sha256"]
    assert b["content_sha256"] == "sha256:" + hashlib.sha256(text.encode()).hexdigest()
    runs = api.files_runs({"user_token": tok, "file_id": f["id"]})
    assert runs["chain_valid"] is True and runs["runs"][-1]["source_binding"]["cell_id"] == "bell"
    assert runs["runs"][-1]["receipt_hash"] == r["receipt"]["hash"]

    # stale revision executes nothing
    assert s.save_file(tok, f["id"], 1, _notebook(("bell", "dnalang", BELL + "// v2\n")))["status"] == "SAVED"
    stale = api.notebook_run({"user_token": tok, "file_id": f["id"], "revision": 1, "cell_id": "bell"})
    assert stale["status"] == "REVISION_MISMATCH" and stale["executed"] is False
    # markdown, unknown cells, other users, non-notebooks are refused
    for cell, kind in (("intro", "NOTEBOOK_CELL_NOT_FOUND"), ("nope", "NOTEBOOK_CELL_NOT_FOUND")):
        with pytest.raises(ide_api.ApiError) as e:
            api.notebook_run({"user_token": tok, "file_id": f["id"], "revision": 2, "cell_id": cell})
        assert e.value.kind == kind
    other = stack["sign_in"]("oscar@example.test")["access_token"]
    with pytest.raises(ide_api.ApiError) as e:
        api.notebook_run({"user_token": other, "file_id": f["id"], "revision": 2, "cell_id": "bell"})
    assert e.value.status == 404
    plain = s.create_file(tok, p["id"], "bell.dna", BELL)
    with pytest.raises(ide_api.ApiError) as e:
        api.notebook_run({"user_token": tok, "file_id": plain["id"], "revision": 1, "cell_id": "bell"})
    assert e.value.kind == "NOTEBOOK_INVALID"


def test_markdown_cell_is_not_executable(stack):
    s, api = stack["store"], stack["api"]
    tok = stack["sign_in"]("nora@example.test")["access_token"]
    f = s.create_file(tok, s.create_project(tok, "NB2")["id"], "n.dnanb", _notebook(("m", "markdown", "text")))
    with pytest.raises(ide_api.ApiError) as e:
        api.notebook_run({"user_token": tok, "file_id": f["id"], "revision": 1, "cell_id": "m"})
    assert e.value.kind == "NOTEBOOK_CELL_NOT_EXECUTABLE" and e.value.status == 400


def test_notebook_format_rejects_malformed_content():
    from osiris_cli import notebook_format as nbf
    for bad in ("x", "{}", '{"format":"osiris-notebook","version":2,"cells":[]}',
                _notebook(("a", "python", "print(1)")), _notebook(("a", "markdown", ""), ("a", "markdown", "")),
                _notebook(("bad id!", "markdown", ""))):
        with pytest.raises(nbf.NotebookFormatError):
            nbf.parse(bad)
    assert nbf.parse(_notebook(("a", "dnalang", BELL)))["cells"][0]["id"] == "a"


# ───────────────────────── IDE API 1.5.0: rename ─────────────────────────
def test_rename_keeps_identity_and_receipts_and_refuses_stale_or_taken_paths(stack):
    s, api = stack["store"], stack["api"]
    tok = stack["sign_in"]("rita@example.test")["access_token"]
    p = s.create_project(tok, "Rename")
    f = s.create_file(tok, p["id"], "notebooks/a.dnanb", _notebook(("bell", "dnalang", BELL)))
    s.create_file(tok, p["id"], "notebooks/taken.dnanb", _notebook())
    run = api.notebook_run({"user_token": tok, "file_id": f["id"], "revision": 1, "cell_id": "bell", "shots": 50})
    r = api.files_rename({"user_token": tok, "file_id": f["id"], "base_revision": 1, "path": "notebooks/b.dnanb"})
    check("POST /api/v1/files/rename", r)
    assert r["status"] == "RENAMED" and r["file"]["id"] == f["id"] and r["file"]["revision"] == 2
    assert r["file"]["content_sha256"] == f["content_sha256"]
    assert [x["receipt_hash"] for x in api.files_runs({"user_token": tok, "file_id": f["id"]})["runs"]] == [run["receipt"]["hash"]]
    stale = api.files_rename({"user_token": tok, "file_id": f["id"], "base_revision": 1, "path": "notebooks/c.dnanb"})
    assert stale["status"] == "CONFLICT" and stale["current"]["path"] == "notebooks/b.dnanb"
    with pytest.raises(ide_api.ApiError) as e:
        api.files_rename({"user_token": tok, "file_id": f["id"], "base_revision": 2, "path": "notebooks/taken.dnanb"})
    assert e.value.kind == "EDITOR_PATH_EXISTS" and e.value.status == 409
    with pytest.raises(ide_api.ApiError) as e:
        api.files_rename({"user_token": tok, "file_id": f["id"], "base_revision": 2, "path": "../escape"})
    assert e.value.status == 400
    other = stack["sign_in"]("sam@example.test")["access_token"]
    with pytest.raises(ide_api.ApiError) as e:
        api.files_rename({"user_token": other, "file_id": f["id"], "base_revision": 2, "path": "notebooks/mine.dnanb"})
    assert e.value.status == 404
    assert s.file(tok, f["id"])["path"] == "notebooks/b.dnanb"
