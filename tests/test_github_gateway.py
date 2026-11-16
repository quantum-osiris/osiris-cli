"""Governed GitHub gateway: identity, allowlists, policy, proposal binding, grants, receipts, failures.

The GitHub API is replaced by FakeBridge (no network). Subprocess-level behaviour of the real
bridge (timeouts, redaction, argv shape) is tested with a fake gh executable.
"""

import json
import os
import stat
import sys
import textwrap

import pytest

pytest.importorskip("osiris_governance.github_policy")

from osiris_governance.canonical import canonicalize_json, strict_parse_json  # noqa: E402

from osiris_cli.github_bridge import GitHubBridge, classify_failure, redact  # noqa: E402
from osiris_cli.github_gateway import DEFAULT_POLICY_PATH, GatewayError, GitHubGateway  # noqa: E402
from osiris_cli.trio_authz import Receipts  # noqa: E402

FAKE_TOKEN = "gho_" + "A1b2C3d4" * 5


class FakeBridge:
    def __init__(self, login="quantum-osiris", login_error=None, fail=None):
        self.login, self.login_error, self.fail = login, login_error, fail
        self.calls = []

    def authenticated_login(self):
        if self.login_error:
            return {"status": "ERROR", "kind": self.login_error, "reason": "simulated"}
        return {"status": "SUCCESS", "login": self.login}

    def api(self, endpoint, method="GET", body=None, timeout=30):
        self.calls.append((method, endpoint, body))
        if self.fail:
            return {"status": "ERROR", "kind": self.fail, "reason": f"dial tcp: lookup api.github.com ({FAKE_TOKEN})"}
        if endpoint.endswith("/branches?per_page=30"):
            return {"status": "SUCCESS", "data": [{"name": "main", "commit": {"sha": "a" * 40}, "protected": False}]}
        if "/pulls" in endpoint and method == "POST":
            return {"status": "SUCCESS", "data": {"number": 42, "html_url": "https://github.com/x/y/pull/42"}}
        if endpoint.startswith("repos/") and endpoint.count("/") == 2:
            return {"status": "SUCCESS", "data": {"full_name": endpoint[6:], "private": False, "default_branch": "main",
                                                  "permissions": {"admin": True}}}
        return {"status": "SUCCESS", "data": []}

    @property
    def posts(self):
        return [c for c in self.calls if c[0] == "POST"]


def write_policy(tmp_path, **over):
    doc = json.load(open(DEFAULT_POLICY_PATH))
    doc.update(over)
    p = tmp_path / "policy.json"
    p.write_text(json.dumps(doc))
    return str(p)


def gateway(tmp_path, bridge=None, **policy_over):
    return GitHubGateway(policy_path=write_policy(tmp_path, **policy_over),
                         receipts_path=str(tmp_path / "receipts.jsonl"), bridge=bridge or FakeBridge())


OPEN = dict(writes_enabled=True, security_gates={"S0": "PASSED"})


def kinds(gw):
    return [r["kind"] for r in gw.receipts()["rows"]]


class Granted:
    """Stands in for grants.authorize: approves only the digest and scope it was issued for, once."""

    def __init__(self, digest=None, scope=None):
        self.digest, self.scope, self.used, self.calls = digest, scope, False, []

    def __call__(self, grant_path, payload_digest, scope, signature_path=None):
        self.calls.append((payload_digest, scope))
        ok = (not self.used and payload_digest == self.digest and scope == self.scope)
        self.used = self.used or ok

        class D:
            detail = "authorized once" if ok else "grant does not match"

            def __bool__(s):
                return ok
        return D()


# ── identity and credentials ──
def test_identity_mismatch_denies_reads(tmp_path):
    gw = gateway(tmp_path, FakeBridge(login="someone-else"))
    assert gw.status()["identity_matches"] is False
    with pytest.raises(GatewayError) as e:
        gw.read("repo.view", "quantum-osiris/osiris-cli")
    assert e.value.kind == "POLICY_DENIED" and "IDENTITY_MISMATCH" in e.value.message
    assert kinds(gw) == ["github.read.denied"]


def test_missing_credentials(tmp_path):
    gw = gateway(tmp_path, FakeBridge(login_error="AUTH"))
    st = gw.status()
    assert st["authenticated"] is False and st["error"] == "NOT_AUTHENTICATED"
    with pytest.raises(GatewayError) as e:
        gw.read("repo.view", "quantum-osiris/osiris-cli")
    assert "IDENTITY_UNAVAILABLE" in e.value.message


def test_gh_not_installed(tmp_path):
    gw = gateway(tmp_path, GitHubBridge(gh_binary=None))
    gw.bridge.gh_binary = None
    assert gw.status()["error"] == "GH_NOT_INSTALLED"


# ── reads ──
def test_read_only_access_works_and_never_posts(tmp_path):
    b = FakeBridge()
    gw = gateway(tmp_path, b)
    r = gw.read("branches.list", "quantum-osiris/osiris-cli")
    assert r["data"] == [{"name": "main", "sha": "a" * 40, "protected": False}]
    assert b.calls == [("GET", "repos/quantum-osiris/osiris-cli/branches?per_page=30", None)]
    assert gw.ecosystem()["repositories"][0]["ok"] is True
    assert not b.posts


def test_read_of_unlisted_repository_denied_and_receipted(tmp_path):
    b = FakeBridge()
    gw = gateway(tmp_path, b)
    with pytest.raises(GatewayError) as e:
        gw.read("repo.view", "attacker/osiris-cli")
    assert "REPOSITORY_NOT_ALLOWLISTED" in e.value.message and b.calls == []
    assert kinds(gw) == ["github.read.denied"]


def test_reads_refuse_write_operations(tmp_path):
    with pytest.raises(GatewayError) as e:
        gateway(tmp_path).read("pr.create", "quantum-osiris/osiris-cli", title="t", head="h", base="main")
    assert e.value.kind == "NOT_A_READ"


@pytest.mark.parametrize("op,args", [
    ("branches.list", {"limit": 0}),
    ("branches.list", {"limit": True}),
    ("commits.list", {"ref": "../../etc"}),
    ("commits.list", {"ref": "-rf"}),
    ("branches.list", {"bogus": 1}),
    ("pull.view", {}),
    ("repo.delete", {}),
])
def test_invalid_arguments_rejected_before_any_request(tmp_path, op, args):
    b = FakeBridge()
    with pytest.raises(GatewayError) as e:
        gateway(tmp_path, b).read(op, "quantum-osiris/osiris-cli", **args)
    assert e.value.kind in ("INVALID_ARGUMENT", "OPERATION_NOT_ALLOWLISTED") and b.calls == []


def test_network_failure_on_read_is_reported(tmp_path):
    gw = gateway(tmp_path, FakeBridge(fail="NETWORK"))
    with pytest.raises(GatewayError) as e:
        gw.read("repo.view", "quantum-osiris/osiris-cli")
    assert e.value.kind == "NETWORK"


# ── writes ──
def pr(gw, repo="quantum-osiris/osiris-cli", **kw):
    args = dict(title="Sprint 0 hardening", head="sprint0/hardening", base="main")
    args.update(kw)
    return gw.propose("pr.create", repo, **args)


def test_shipped_policy_keeps_writes_disabled_while_gates_unmet():
    doc = json.load(open(DEFAULT_POLICY_PATH))
    if any(v != "PASSED" for v in doc["security_gates"].values()):
        assert doc["writes_enabled"] is False


def test_policy_denial_blocks_write_and_is_receipted(tmp_path):
    b = FakeBridge()
    gw = gateway(tmp_path, b)
    r = gw.execute(pr(gw), dry_run=False, grant_path="unused")
    assert r["status"] == "DENIED"
    assert "WRITES_DISABLED_BY_POLICY" in r["reasons"]
    assert not b.posts and kinds(gw) == ["github.write.denied"]


def test_unauthorized_repository_write_denied_even_when_writes_enabled(tmp_path):
    b = FakeBridge()
    gw = gateway(tmp_path, b, **OPEN)
    r = gw.execute(pr(gw, repo="enki-420/quantum-ide"), dry_run=False, grant_path="g")
    assert r["status"] == "DENIED" and any("CAPABILITY_NOT_GRANTED" in x for x in r["reasons"])
    assert not b.posts


def test_dry_run_plans_without_sending(tmp_path):
    b = FakeBridge()
    gw = gateway(tmp_path, b, **OPEN)
    r = gw.execute(pr(gw), dry_run=True)
    assert r["status"] == "DRY_RUN" and r["plan"]["endpoint"] == "repos/quantum-osiris/osiris-cli/pulls"
    assert r["body"]["draft"] is True and not b.posts
    assert kinds(gw) == ["github.write.dry_run"]


def test_write_without_grant_is_denied(tmp_path):
    b = FakeBridge()
    gw = gateway(tmp_path, b, **OPEN)
    r = gw.execute(pr(gw), dry_run=False)
    assert r["reasons"] == ["SYS021_GRANT_REQUIRED"] and not b.posts


def test_real_sys021_authorize_refuses_an_unsigned_grant(tmp_path):
    """No owner-signed grant exists on this machine, so the production authorizer must refuse."""
    b = FakeBridge()
    gw = gateway(tmp_path, b, **OPEN)
    grant = tmp_path / "grant.json"
    grant.write_text("{}")
    r = gw.execute(pr(gw), dry_run=False, grant_path=str(grant),
                   authorize_kwargs={"ledger_path": str(tmp_path / "auth.jsonl"),
                                     "allowed_signers": str(tmp_path / "allowed_signers")})
    assert r["status"] == "DENIED" and r["reasons"][0].startswith("SYS021_GRANT_REFUSED")
    assert not b.posts


def test_granted_write_executes_once_with_write_ahead_receipt(tmp_path):
    b = FakeBridge()
    gw = gateway(tmp_path, b, **OPEN)
    prop = pr(gw)
    digest = gw.decide(prop).proposal_sha256
    auth = Granted(digest, "github:github.pr.create:quantum-osiris/osiris-cli")
    r = gw.execute(prop, dry_run=False, grant_path="g", authorize=auth)
    assert r["status"] == "EXECUTED" and r["result"]["number"] == 42
    assert kinds(gw) == ["github.write.attempt", "github.write.result"]
    method, endpoint, body = b.posts[0]
    assert endpoint == "repos/quantum-osiris/osiris-cli/pulls" and body["head"] == "sprint0/hardening"
    assert gw.receipts()["chain_valid"]


def test_duplicate_request_refused(tmp_path):
    b = FakeBridge()
    gw = gateway(tmp_path, b, **OPEN)
    prop = pr(gw)
    auth = Granted(gw.decide(prop).proposal_sha256, "github:github.pr.create:quantum-osiris/osiris-cli")
    assert gw.execute(prop, dry_run=False, grant_path="g", authorize=auth)["status"] == "EXECUTED"
    again = gw.execute(prop, dry_run=False, grant_path="g", authorize=auth)
    assert again["status"] == "DUPLICATE_REFUSED" and len(b.posts) == 1


def test_altered_payload_does_not_inherit_the_grant(tmp_path):
    b = FakeBridge()
    gw = gateway(tmp_path, b, **OPEN)
    original = pr(gw)
    auth = Granted(gw.decide(original).proposal_sha256, "github:github.pr.create:quantum-osiris/osiris-cli")
    tampered = canonicalize_json({**strict_parse_json(original),
                                  "arguments": {**strict_parse_json(original)["arguments"], "base": "release"}})
    r = gw.execute(tampered, dry_run=False, grant_path="g", authorize=auth)
    assert r["status"] == "DENIED" and r["reasons"][0].startswith("SYS021_GRANT_REFUSED")
    assert auth.calls[0][0] != auth.digest and not b.posts


def test_non_canonical_payload_denied(tmp_path):
    gw = gateway(tmp_path, **OPEN)
    pretty = json.dumps(strict_parse_json(pr(gw)), indent=1).encode()
    r = gw.execute(pretty, dry_run=True)
    assert r["status"] == "DENIED" and "PROPOSAL_NOT_CANONICAL" in r["reasons"]


def test_capability_and_operation_binding_mismatch(tmp_path):
    gw = gateway(tmp_path, **OPEN)
    p = strict_parse_json(pr(gw))
    p["arguments"] = {"op": "issue.comment", "number": 1, "body": "x"}  # capability still github.pr.create
    with pytest.raises(GatewayError) as e:
        gw.execute(canonicalize_json(p), dry_run=True)
    assert e.value.kind == "BINDING_MISMATCH"


def test_network_failure_on_write_is_receipted_and_redacted(tmp_path):
    b = FakeBridge(fail="NETWORK")
    gw = gateway(tmp_path, b, **OPEN)
    prop = pr(gw)
    auth = Granted(gw.decide(prop).proposal_sha256, "github:github.pr.create:quantum-osiris/osiris-cli")
    r = gw.execute(prop, dry_run=False, grant_path="g", authorize=auth)
    assert r["status"] == "FAILED" and r["kind"] == "NETWORK"
    assert kinds(gw) == ["github.write.attempt", "github.write.result"]
    assert FAKE_TOKEN not in open(gw.receipts_path).read()


def test_receipt_tampering_detected(tmp_path):
    gw = gateway(tmp_path, **OPEN)
    gw.execute(pr(gw), dry_run=True)
    gw.execute(pr(gw), dry_run=True)
    rows = open(gw.receipts_path).read().splitlines()
    row = json.loads(rows[0]); row["payload"]["plan"]["endpoint"] = "repos/attacker/x/pulls"
    open(gw.receipts_path, "w").write("\n".join([json.dumps(row, sort_keys=True), rows[1]]) + "\n")
    assert Receipts.verify(gw.receipts_path) is False


# ── the real bridge, with a fake gh executable ──
@pytest.fixture
def fake_gh(tmp_path):
    script = tmp_path / "gh"
    script.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import json, os, sys, time
        log = os.environ["FAKE_GH_LOG"]
        stdin = sys.stdin.read() if "--input" in sys.argv else None
        open(log, "a").write(json.dumps({{"argv": sys.argv[1:], "stdin": stdin}}) + "\\n")
        mode = os.environ.get("FAKE_GH_MODE", "ok")
        if mode == "sleep": time.sleep(5)
        if mode == "network":
            sys.stderr.write("error connecting to api.github.com: dial tcp: lookup api.github.com: no such host\\n"); sys.exit(1)
        if mode == "leak":
            sys.stderr.write("HTTP 401: Bad credentials token {FAKE_TOKEN}\\n"); sys.exit(1)
        print(json.dumps({{"login": "quantum-osiris"}}))
    """))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "gh.log"
    os.environ["FAKE_GH_LOG"] = str(log)
    yield str(script), log
    os.environ.pop("FAKE_GH_LOG", None)
    os.environ.pop("FAKE_GH_MODE", None)


def test_bridge_identity_via_api(fake_gh):
    assert GitHubBridge(gh_binary=fake_gh[0]).authenticated_login() == {"status": "SUCCESS", "login": "quantum-osiris"}


def test_bridge_timeout(fake_gh):
    os.environ["FAKE_GH_MODE"] = "sleep"
    r = GitHubBridge(gh_binary=fake_gh[0]).run_gh(["api", "user"], timeout=0.5)
    assert r["kind"] == "TIMEOUT"


def test_bridge_network_failure_classified(fake_gh):
    os.environ["FAKE_GH_MODE"] = "network"
    assert GitHubBridge(gh_binary=fake_gh[0]).api("user")["kind"] == "NETWORK"


def test_bridge_redacts_tokens_in_errors(fake_gh):
    os.environ["FAKE_GH_MODE"] = "leak"
    r = GitHubBridge(gh_binary=fake_gh[0]).api("user")
    assert r["kind"] == "AUTH" and FAKE_TOKEN not in r["reason"] and "<REDACTED>" in r["reason"]


def test_bridge_never_puts_request_values_in_argv(fake_gh):
    hostile = {"title": "--hostname=evil.example @/etc/passwd", "body": "$(rm -rf ~)"}
    GitHubBridge(gh_binary=fake_gh[0]).api("repos/a/b/issues", method="POST", body=hostile)
    call = json.loads(fake_gh[1].read_text().splitlines()[-1])
    assert call["argv"] == ["api", "--method", "POST", "-H", "Accept: application/vnd.github+json",
                            "repos/a/b/issues", "--input", "-"]
    assert json.loads(call["stdin"]) == hostile


@pytest.mark.parametrize("endpoint", ["-X", "https://evil.example/x", "user\n--hostname"])
def test_bridge_rejects_non_relative_endpoints(endpoint):
    with pytest.raises(ValueError):
        GitHubBridge(gh_binary="/bin/true").api(endpoint)


def test_redact_and_classify():
    assert FAKE_TOKEN not in redact(f"token {FAKE_TOKEN}")
    assert redact("  - Token: gho_************************************") == "  - Token: <REDACTED>"
    assert classify_failure(1, "HTTP 404: Not Found") == "NOT_FOUND"
    assert classify_failure(1, "HTTP 403: forbidden") == "FORBIDDEN"
    assert classify_failure(1, "something odd") == "GH_ERROR"
