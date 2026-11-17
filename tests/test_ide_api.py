"""OSIRIS IDE API v1: contract conformance, console compatibility and loopback exposure rules."""

import http.client
import json
import os
import re

import pytest

pytest.importorskip("osiris_governance.github_policy")
pytest.importorskip("dnalang")
jsonschema = pytest.importorskip("jsonschema")

from osiris_cli import ide_api  # noqa: E402
from osiris_cli.github_gateway import GitHubGateway  # noqa: E402

from tests.test_github_gateway import FakeBridge, write_policy  # noqa: E402

CONTRACT = json.load(open(ide_api.CONTRACT_PATH))
BELL = "organism B { gene g() on a, b { helix a; bond a, b; cleave a -> x; cleave b -> y; } genome { g() @ [0, 1]; } }"


def fake_agent_model(role, system, prompt):
    """Stands in for the model gateway: fixed text per role, no network."""
    text = {"AURA": "Proposal: compile the Bell organism, then simulate it. Risk: none known.",
            "AIDEN": "The proposal names no receipt to check.\nVERDICT: PROCEED",
            "IRIS": "Objective: a Bell baseline. AURA proposes a simulator run. AIDEN asks for a receipt."}[role]
    return text, "fake:test-model"


class FakeSupabaseHttp:
    """In-memory PostgREST stand-in so IDE API store routes validate against the contract offline."""
    def __init__(self):
        self.projects = {}
        self.tasks = {}
        self.dispositions = {}
        self._seq = 0

    def _uuid(self):
        self._seq += 1
        return f"00000000-0000-4000-8000-{self._seq:012d}"

    def __call__(self, method, url, headers, body):
        payload = json.loads(body) if body else None
        if method == "POST" and url.endswith("/osiris_projects"):
            pid = self._uuid()
            row = {"id": pid, "name": payload["name"], "description": payload.get("description", ""),
                   "archived": False, "created_at": "2026-11-16T12:00:00Z"}
            self.projects[pid] = row
            return [row]
        if method == "GET" and "/osiris_projects?select=" in url:
            return list(self.projects.values())
        if method == "GET" and "/osiris_projects?id=eq." in url:
            pid = url.split("id=eq.", 1)[1].split("&", 1)[0]
            if pid not in self.projects:
                return []
            p = dict(self.projects[pid])
            p["osiris_tasks"] = [t for t in self.tasks.values() if t["project_id"] == pid]
            p["osiris_dispositions"] = [d for d in self.dispositions.values() if d["project_id"] == pid]
            return [p]
        if method == "POST" and url.endswith("/osiris_tasks"):
            tid = self._uuid()
            row = {"id": tid, "project_id": payload["project_id"], "title": payload["title"],
                   "status": payload.get("status", "todo"), "created_at": "2026-11-16T12:00:00Z"}
            self.tasks[tid] = row
            return [row]
        if method == "PATCH" and "/osiris_tasks?id=eq." in url:
            tid = url.split("id=eq.", 1)[1]
            if tid not in self.tasks:
                return []
            self.tasks[tid].update(payload)
            return [dict(self.tasks[tid])]
        if method == "POST" and url.endswith("/osiris_dispositions"):
            did = self._uuid()
            row = {"id": did, **payload, "created_at": "2026-11-16T12:00:00Z"}
            self.dispositions[did] = row
            return [row]
        return []


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    import threading
    from osiris_cli.workspace_store import PROJECT_REF, WorkspaceStore
    tmp = tmp_path_factory.mktemp("ide")
    console = tmp / "console"
    console.mkdir()
    (console / "index.html").write_text('<meta name="osiris-api-token" content="__OSIRIS_SESSION_TOKEN__">')
    (console / "app.js").write_text("// test")
    (console / "secret.txt").write_text("not served")
    examples = tmp / "examples"
    examples.mkdir()
    (examples / "bell.dna").write_text(BELL)
    policy = write_policy(tmp)
    fake_sb = FakeSupabaseHttp()
    store = WorkspaceStore(f"https://{PROJECT_REF}.supabase.co", "test-service-key", http=fake_sb)
    seeded = store.create_project("Seeded Project", "Contract fixture")
    backend = ide_api.Backend(
        gateway_factory=lambda: GitHubGateway(policy_path=policy, receipts_path=str(tmp / "gh.jsonl"), bridge=FakeBridge()),
        slice_receipts=str(tmp / "slice.jsonl"), examples_dir=str(examples), workspace_dirs=(str(tmp / "nope"),),
        agent_model_fn=fake_agent_model, store_factory=lambda: store)
    httpd, token = ide_api.serve(backend, port=0, console_dir=str(console), token_file=str(tmp / "tok"))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield {"port": httpd.server_address[1], "token": token, "tmp": tmp, "project_id": seeded["id"], "fake_sb": fake_sb}
    httpd.shutdown()
    httpd.server_close()


def call(server, method, path, body=None, *, token=True, headers=None, raw=None):
    conn = http.client.HTTPConnection("127.0.0.1", server["port"], timeout=60)
    h = {"Host": f"127.0.0.1:{server['port']}"}
    if token:
        h["X-OSIRIS-Token"] = server["token"]
    data = raw
    if body is not None:
        data = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    h.update(headers or {})
    conn.request(method, path, body=data, headers=h)
    r = conn.getresponse()
    payload = r.read()
    conn.close()
    ctype = r.getheader("Content-Type", "")
    return r.status, (json.loads(payload) if ctype.startswith("application/json") else payload), r


def check(route, payload):
    jsonschema.validate(payload, CONTRACT["routes"][route]["response"])


# ── contract ──
def test_routes_and_contract_agree():
    served = {f"{m} {p}" for m, p in ide_api.ROUTES}
    assert served == set(CONTRACT["routes"])
    assert CONTRACT["api_version"] == ide_api.API_VERSION


def test_console_endpoints_are_in_the_contract():
    console = os.environ.get("OSIRIS_CONSOLE_DIR")
    if not console or not os.path.isfile(os.path.join(console, "app.js")):
        pytest.skip("set OSIRIS_CONSOLE_DIR to the quantum-ide public/osiris-console directory")
    js = open(os.path.join(console, "app.js")).read()
    used = set(re.findall(r'"((?:GET|POST) /api/v1/[a-z/]+)"', js))
    assert used and used <= set(CONTRACT["routes"]), used - set(CONTRACT["routes"])
    assert re.search(r"const API_MAJOR = (\d+);", js).group(1) == ide_api.API_VERSION.split(".")[0]
    html = open(os.path.join(console, "index.html")).read()
    assert ide_api.TOKEN_PLACEHOLDER in html and "<script>" not in html and " style=" not in html


@pytest.mark.parametrize("route,body", [
    ("GET /api/v1/meta", None),
    ("GET /api/v1/workspace", None),
    ("GET /api/v1/examples", None),
    ("POST /api/v1/dnalang/compile", {"source": BELL}),
    ("POST /api/v1/dnalang/compile", {"source": "organism {"}),
    ("POST /api/v1/slice/run", {"source": BELL, "shots": 300, "seed": 4}),
    ("POST /api/v1/slice/run", {"source": "organism {"}),
    ("POST /api/v1/organism/run", {"ticks": 40, "seed": 2}),
    ("POST /api/v1/governance/github/propose", {"op": "repo.view", "repo": "quantum-osiris/osiris-cli"}),
    ("POST /api/v1/governance/github/propose",
     {"op": "pr.create", "repo": "quantum-osiris/osiris-cli", "args": {"title": "t", "head": "h", "base": "main"}}),
    ("GET /api/v1/governance/receipts", None),
    ("GET /api/v1/research/claims", None),
    ("GET /api/v1/research/experiments", None),
    ("POST /api/v1/agents/task", {"objective": "Establish a Bell baseline on the simulator"}),
    ("GET /api/v1/agents/tasks", None),
    ("GET /api/v1/store/status", None),
    ("GET /api/v1/store/projects", None),
    ("POST /api/v1/store/projects", {"name": "Bell Workspace", "description": "Supabase slice"}),
    ("POST /api/v1/store/project", {"project_id": "00000000-0000-4000-8000-000000000001"}),
    ("POST /api/v1/store/tasks", {"project_id": "00000000-0000-4000-8000-000000000001", "title": "Simulate Bell"}),
])
def test_every_response_matches_the_contract(server, route, body):
    method, path = route.split(" ")
    status, payload, _ = call(server, method, path, body)
    assert status == 200, payload
    check(route, payload)


def test_slice_over_http_is_real_and_receipted(server):
    _, r, _ = call(server, "POST", "/api/v1/slice/run", {"source": BELL, "shots": 400, "seed": 9})
    assert r["status"] == "COMPLETED" and sum(r["simulation"]["counts"].values()) == 400
    assert r["receipt"]["chain_valid"]
    _, ex, _ = call(server, "GET", "/api/v1/research/experiments")
    assert any(x["run_id"] == r["run_id"] for x in ex["runs"]) and ex["chain_valid"]


def test_write_proposals_are_dry_run_only(server):
    _, r, _ = call(server, "POST", "/api/v1/governance/github/propose",
                   {"op": "pr.create", "repo": "quantum-osiris/osiris-cli", "args": {"title": "t", "head": "h", "base": "main"}})
    assert r["is_write"] and r["decision"]["decision"] == "DENY" and r["dry_run"]["status"] == "DENIED"
    assert "not available through the API" in r["execution"]
    assert not any("execute" in p for _, p in ide_api.ROUTES)


def test_bad_inputs_get_contract_errors(server):
    for method, path, body in [("POST", "/api/v1/slice/run", {"source": BELL, "shots": 0}),
                               ("POST", "/api/v1/organism/run", {"ticks": 10**6}),
                               ("POST", "/api/v1/governance/github/propose", {"op": "repo.delete", "repo": "a/b"})]:
        status, payload, _ = call(server, method, path, body)
        assert status == 400
        jsonschema.validate(payload, CONTRACT["error"])


# ── exposure rules ──
def test_token_required(server):
    status, payload, _ = call(server, "GET", "/api/v1/meta", token=False)
    assert status == 401 and payload["error"]["kind"] == "TOKEN_REQUIRED"


def test_wrong_token(server):
    status, _, _ = call(server, "GET", "/api/v1/meta", token=False, headers={"X-OSIRIS-Token": "x" * 43})
    assert status == 401


def test_dns_rebinding_host_rejected(server):
    status, payload, _ = call(server, "GET", "/api/v1/meta", headers={"Host": f"evil.example:{server['port']}"})
    assert status == 421 and payload["error"]["kind"] == "HOST_REJECTED"


def test_cross_origin_rejected(server):
    status, _, _ = call(server, "GET", "/api/v1/meta", headers={"Origin": "http://evil.example"})
    assert status == 403


def test_form_posts_rejected(server):
    status, _, _ = call(server, "POST", "/api/v1/dnalang/compile", raw=b"source=x",
                        headers={"Content-Type": "application/x-www-form-urlencoded"})
    assert status == 415


def test_oversized_body_rejected(server):
    status, _, _ = call(server, "POST", "/api/v1/dnalang/compile", raw=b"{}",
                        headers={"Content-Type": "application/json", "Content-Length": str(ide_api.MAX_BODY + 1)})
    assert status == 413


def test_unknown_route_and_method(server):
    assert call(server, "GET", "/api/v1/nope")[0] == 404
    assert call(server, "GET", "/api/v1/slice/run")[0] == 405
    assert call(server, "DELETE", "/api/v1/meta")[0] == 405


def test_static_serving_is_confined(server):
    status, page, r = call(server, "GET", "/", token=False)
    assert status == 200 and server["token"].encode() in page and b"__OSIRIS" not in page
    assert "frame-ancestors 'none'" in r.getheader("Content-Security-Policy")
    assert call(server, "GET", "/secret.txt", token=False)[0] == 404
    assert call(server, "GET", "/../../etc/passwd", token=False)[0] == 404


def test_token_never_echoed_in_api_responses(server):
    for route in CONTRACT["routes"]:
        method, path = route.split(" ")
        if method == "GET":
            _, payload, _ = call(server, method, path)
            assert server["token"] not in json.dumps(payload)


def test_token_file_is_private(server):
    assert oct(os.stat(server["tmp"] / "tok").st_mode & 0o777) == "0o600"


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "::"])
def test_non_loopback_bind_refused(host):
    with pytest.raises(ValueError):
        ide_api.serve(ide_api.Backend(gateway_factory=lambda: None, slice_receipts="x", examples_dir="x"), host=host, port=0)


def test_end_to_end_project_task_council_proposal_persisted_and_retrieved(server):
    # 1. Create project
    st, proj_resp, _ = call(server, "POST", "/api/v1/store/projects",
                            {"name": "Hayden-Preskill Audit", "description": "Track offline estimator verification"})
    assert st == 200
    pid = proj_resp["project"]["id"]

    # 2. Create task & move status
    st, task_resp, _ = call(server, "POST", "/api/v1/store/tasks",
                            {"project_id": pid, "title": "Verify marginal vs joint Bell estimator"})
    assert st == 200
    tid = task_resp["task"]["id"]
    st, moved_resp, _ = call(server, "POST", "/api/v1/store/tasks", {"task_id": tid, "status": "doing"})
    assert st == 200 and moved_resp["task"]["status"] == "doing"

    # 3. Run OSIRIS council proposal bound to (project_id, task_id) -> persists disposition + receipt hash
    st, agent_resp, _ = call(server, "POST", "/api/v1/agents/task", {
        "objective": "Design synthetic 4-qubit test comparing marginal product vs joint Bell overlap",
        "context": "Project: Hayden-Preskill Audit",
        "project_id": pid,
        "task_id": tid,
    })
    assert st == 200
    assert agent_resp["authority"]["executed"] is False
    saved = agent_resp["saved_disposition"]
    assert saved["project_id"] == pid and saved["task_id"] == tid
    assert saved["receipt_hash"] == agent_resp["receipt"]["hash"]
    assert saved["executed"] is False

    # 4. Retrieve project -> includes task and saved disposition with receipt reference, never raw prose
    st, detail_resp, _ = call(server, "POST", "/api/v1/store/project", {"project_id": pid})
    assert st == 200
    proj = detail_resp["project"]
    assert any(t["id"] == tid and t["status"] == "doing" for t in proj["osiris_tasks"])
    assert any(d["council_task_id"] == agent_resp["task_id"] and d["receipt_hash"] == agent_resp["receipt"]["hash"]
               for d in proj["osiris_dispositions"])
    assert "test-service-key" not in json.dumps(detail_resp)



def test_qiskit_operations_share_one_worker_thread():
    """Regression: simulations from successive request threads segfaulted qiskit (see QISKIT_OPERATIONS)."""
    import threading
    seen = []
    for _ in range(3):
        t = threading.Thread(target=lambda: seen.append(ide_api._QISKIT_THREAD.submit(lambda: threading.get_ident()).result()))
        t.start(); t.join()
    assert len(set(seen)) == 1
    assert {"compile", "run_slice", "files_run", "notebook_run"} <= ide_api.QISKIT_OPERATIONS
    assert set(ide_api.QISKIT_OPERATIONS) <= set(ide_api.ROUTES.values())
