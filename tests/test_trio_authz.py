"""Signed, single-use, payload-bound authorization for /trio run. Synthetic repos, temporary keys only;
no live ~/.osiris state, no network, no model."""

import json
import multiprocessing
import os
import shutil
import subprocess

import pytest

from osiris_cli import grants, sandbox, trio_authz
from osiris_cli import trio_command as tc
from osiris_cli.orchestration.trio import tree_digest

pytestmark = [pytest.mark.skipif(shutil.which("ssh-keygen") is None, reason="needs ssh-keygen"),
              pytest.mark.skipif(not sandbox.available(), reason="needs bwrap")]

BUGGY = "def mean(xs):\n    return sum(xs) / (len(xs) + 1)\n"
FIXED = "def mean(xs):\n    return sum(xs) / len(xs)\n"
TESTS = "from calc import mean\n\ndef test_mean():\n    assert mean([2, 4]) == 3\n"
CLOCK = lambda: (True, "test clock")  # noqa: E731


@pytest.fixture
def env(tmp_path):
    work = tmp_path / "work"
    repo = work / "repo"
    repo.mkdir(parents=True)
    (repo / "calc.py").write_text(BUGGY)
    (repo / "test_calc.py").write_text(TESTS)
    patch = work / "patch.json"
    patch.write_text(json.dumps({"patched_files": {"calc.py": FIXED}, "report": "Changed the divisor."}))
    key = tmp_path / "owner"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "t", "-f", str(key)], check=True)
    kind, blob = (tmp_path / "owner.pub").read_text().split()[:2]
    signers = tmp_path / "allowed_signers"
    signers.write_text(f'owner namespaces="osiris-grant" {kind} {blob}\n')
    return {"root": str(work), "repo": str(repo), "patch": str(patch), "out": str(tmp_path / "out"),
            "key": key, "signers": str(signers), "ledger": str(tmp_path / "authz.jsonl"), "tmp": tmp_path}


def bundle_digest(env, patch=None):
    plan, _ = tc.build_plan(env["repo"], patch or env["patch"], workspace_root=env["root"])
    return plan.bundle_digest


def sign(env, digest, name="grant.json", expires=900, signed_ago=0, scope=trio_authz.SCOPE, now=None):
    import time
    now = now or time.time()
    g = env["tmp"] / name
    g.write_bytes(grants.draft(digest, scope, expires, now=now - signed_ago))
    subprocess.run(["ssh-keygen", "-q", "-Y", "sign", "-f", str(env["key"]), "-n", "osiris-grant", str(g)],
                   check=True, capture_output=True)
    return str(g)


def run(env, grant, out=None, **kw):
    argv = ["run", env["repo"], "--patch", env["patch"], "--out", out or env["out"], "--grant", grant,
            "--ledger", env["ledger"], "--allowed-signers", env["signers"]]
    return tc.run_trio_command(argv, workspace_root=env["root"], _grant_kw={"clock": CLOCK, **kw})


def receipts(env):
    p = os.path.join(env["out"], "authz_receipts.jsonl")
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else []


def test_valid_grant_runs_sandboxed_once_and_receipts_are_attributable(env):
    g = sign(env, bundle_digest(env))
    before = tree_digest(env["repo"])
    text = run(env, g)
    assert "VERIFIED_PENDING_AUTHORIZATION" in text and "nothing promoted" in text
    assert tree_digest(env["repo"]) == before
    rs = receipts(env)
    assert [r["kind"] for r in rs] == ["AUTHORIZATION_DECISION", "EXECUTION"]
    assert rs[0]["payload"]["authorized"] and rs[0]["payload"]["key_fingerprint"].startswith("SHA256:")
    assert rs[1]["payload"]["sandboxed"] is True and rs[1]["payload"]["applied_to_repo"] is False
    assert trio_authz.Receipts.verify(os.path.join(env["out"], "authz_receipts.jsonl"))
    blob = open(os.path.join(env["out"], "authz_receipts.jsonl")).read()
    assert "BEGIN" not in blob and "PRIVATE" not in blob and open(g).read() not in blob
    assert [json.loads(x)["kind"] for x in open(env["ledger"])] == [grants.GRANTED, grants.CONSUMED]


def test_replay_is_refused(env):
    g = sign(env, bundle_digest(env))
    run(env, g)
    with pytest.raises(tc.TrioCommandError, match="already used"):
        run(env, g, out=str(env["tmp"] / "out2"))


def test_wrong_digest_refused_and_nothing_runs(env):
    g = sign(env, "ab" * 32)
    with pytest.raises(tc.TrioCommandError, match="not authorized: grant is bound to payload"):
        run(env, g)
    assert [r["kind"] for r in receipts(env)] == ["AUTHORIZATION_DECISION"]
    assert not os.path.exists(env["ledger"])


@pytest.mark.parametrize("what", ["repo", "patch", "scope"])
def test_changed_plan_or_scope_invalidates_the_grant(env, what):
    g = sign(env, bundle_digest(env), scope="trio.candidate.promote" if what == "scope" else trio_authz.SCOPE)
    if what == "repo":
        open(os.path.join(env["repo"], "calc.py"), "a").write("# changed after signing\n")
    if what == "patch":
        open(env["patch"], "w").write(json.dumps({"patched_files": {"calc.py": FIXED + "#\n"}, "report": "x"}))
    with pytest.raises(tc.TrioCommandError, match="not authorized"):
        run(env, g)
    assert not os.path.exists(os.path.join(env["out"], "lineage.jsonl"))


def test_altered_budget_sandbox_capability_or_policy_changes_the_bundle_digest(env, monkeypatch):
    base = bundle_digest(env)
    g = sign(env, base)
    monkeypatch.setattr(tc, "TEST_TIMEOUT_S", 61.0)                       # budget / sandbox wall limit
    assert bundle_digest(env) != base
    with pytest.raises(tc.TrioCommandError, match="not authorized"):
        run(env, g)
    monkeypatch.undo()
    from osiris_cli.orchestration import agent_gene
    monkeypatch.setitem(agent_gene.CAPABILITY_TIERS, "REPLAY_FIXTURE", 2)  # capability policy
    assert bundle_digest(env) != base
    monkeypatch.undo()
    monkeypatch.setattr(trio_authz, "POLICY_VERSION", "trio-authz/2")      # policy version
    assert bundle_digest(env) != base
    monkeypatch.undo()
    b = tc.build_plan(env["repo"], env["patch"], workspace_root=env["root"])[0].bundle
    b2 = json.loads(json.dumps(b))
    b2["sandbox"]["network"] = True                                        # sandbox policy
    assert trio_authz.bundle_digest(b2) != base
    b3 = json.loads(json.dumps(b))
    b3["capabilities"].append("WORKSPACE_WRITE")                           # requested capabilities
    assert trio_authz.bundle_digest(b3) != base


def test_expired_and_future_grants_refused(env):
    d = bundle_digest(env)
    with pytest.raises(tc.TrioCommandError, match="expired"):
        run(env, sign(env, d, expires=60, signed_ago=3600))
    with pytest.raises(tc.TrioCommandError, match="not valid before"):
        run(env, sign(env, d, name="g2.json", signed_ago=-3600), out=str(env["tmp"] / "o2"))


def test_revoked_grant_refused(env):
    g = sign(env, bundle_digest(env))
    gid = grants._sha256(open(g, "rb").read())
    grants.revoke(gid, "test revocation", env["ledger"])
    with pytest.raises(tc.TrioCommandError, match="revoked"):
        run(env, g)


def test_tampered_grant_and_foreign_signer_refused(env):
    g = sign(env, bundle_digest(env))
    open(g, "ab").write(b" ")
    with pytest.raises(tc.TrioCommandError, match="signature refused"):
        run(env, g)
    other = env["tmp"] / "other"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "x", "-f", str(other)], check=True)
    env2 = dict(env, key=other)
    with pytest.raises(tc.TrioCommandError, match="signature refused"):
        run(env, sign(env2, bundle_digest(env), name="g3.json"), out=str(env["tmp"] / "o3"))


def test_fail_closed_when_clock_ledger_or_signers_unavailable(env, tmp_path):
    g = sign(env, bundle_digest(env))
    with pytest.raises(tc.TrioCommandError, match="NTP"):
        run(env, g, clock=lambda: (False, "the system clock is not NTP-synchronized"))
    bad = tmp_path / "ledger_is_a_dir"
    bad.mkdir()
    env["ledger"] = str(bad)
    with pytest.raises(tc.TrioCommandError, match="ledger unavailable"):
        run(env, g, out=str(tmp_path / "o4"))
    env["ledger"] = str(tmp_path / "l5.jsonl")
    argv = ["run", env["repo"], "--patch", env["patch"], "--out", str(tmp_path / "o5"), "--grant", g,
            "--ledger", env["ledger"], "--allowed-signers", str(tmp_path / "missing_signers")]
    with pytest.raises(tc.TrioCommandError, match="no trust root"):
        tc.run_trio_command(argv, workspace_root=env["root"], _grant_kw={"clock": CLOCK})


def test_no_sandbox_means_no_spend_and_no_run(env, monkeypatch):
    g = sign(env, bundle_digest(env))
    monkeypatch.setattr(sandbox, "available", lambda: False)
    with pytest.raises(tc.TrioCommandError, match="no unsandboxed mode"):
        run(env, g)
    assert not os.path.exists(env["ledger"]), "grant must not be consumed when the sandbox is unavailable"
    assert not os.path.exists(env["out"])


def test_bare_approve_is_no_longer_accepted_and_missing_grant_refused(env):
    with pytest.raises(tc.TrioCommandError, match="not human authorization"):
        tc.run_trio_command(["run", env["repo"], "--patch", env["patch"], "--out", env["out"], "--approve", "x"],
                            workspace_root=env["root"])
    with pytest.raises(tc.TrioCommandError, match="--grant"):
        tc.run_trio_command(["run", env["repo"], "--patch", env["patch"], "--out", env["out"]],
                            workspace_root=env["root"])
    assert not os.path.exists(env["out"])


def test_inputs_swapped_after_authorization_abort_but_grant_stays_spent(env, monkeypatch):
    g = sign(env, bundle_digest(env))
    real = tc.build_plan
    calls = {"n": 0}

    def swapping(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:                                   # the re-derivation after consumption
            open(os.path.join(env["repo"], "calc.py"), "a").write("# swapped\n")
        return real(*a, **k)
    monkeypatch.setattr(tc, "build_plan", swapping)
    with pytest.raises(tc.TrioCommandError, match="changed between authorization and execution"):
        run(env, g)
    assert "EXECUTION_ABORTED" in [r["kind"] for r in receipts(env)]
    monkeypatch.undo()
    open(os.path.join(env["repo"], "calc.py"), "w").write(BUGGY)      # restore: now only replay protection applies
    with pytest.raises(tc.TrioCommandError, match="already used"):
        run(env, g, out=str(env["tmp"] / "o9"))


def _race(args):
    grant, bd, signers, ledger, i = args
    d = grants.authorize(grant, bd, trio_authz.SCOPE, allowed_signers=signers, ledger_path=ledger,
                         clock=lambda: (True, "t"))
    return bool(d.authorized)


def test_concurrent_use_of_one_grant_succeeds_exactly_once(env):
    bd = bundle_digest(env)
    g = sign(env, bd)
    with multiprocessing.get_context("fork").Pool(6) as pool:
        wins = pool.map(_race, [(g, bd, env["signers"], env["ledger"], i) for i in range(12)])
    assert sum(wins) == 1, wins
    kinds = [json.loads(x)["kind"] for x in open(env["ledger"])]
    assert kinds == [grants.GRANTED, grants.CONSUMED]


def test_live_repl_run_stays_disabled(env, monkeypatch, capsys):
    from osiris_cli import osiris_repl
    monkeypatch.chdir(env["root"])
    osiris_repl.execute_trio("run %s --patch %s --out %s --grant g --ledger l" % (env["repo"], env["patch"], env["out"]))
    assert "disabled in the REPL" in capsys.readouterr().out and not os.path.exists(env["out"])
