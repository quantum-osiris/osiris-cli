"""``/trio``: a dry-run-first front end for the governed Research -> Developer -> Critic slice.

NOT registered in the live REPL (osiris_repl.py is untouched); see docs/TRIO_REPL_INTEGRATION.md for
the wiring and the review checklist. Behavior:

  /trio plan <repo> [--patch FILE]
      Dry run (also what bare ``/trio <repo>`` does). Reads files, runs NOTHING (no subprocess, no
      network, no model), writes NOTHING, and prints the plan with a plan digest.

  /trio run <repo> --patch FILE --out DIR --approve <plan_digest>
      Executes the repo's own test suite (arbitrary code, scrubbed env, copy of the repo, not a
      container) on the supplied patch. Refused unless --approve equals the digest of the plan
      recomputed NOW, so a changed repo/patch/agent spec/budget invalidates the approval. Records the
      outcome and a lineage row under DIR. Never applies the patch, never promotes (no authorizer is
      ever passed), never calls a model or any paid service. The best possible status is
      VERIFIED_PENDING_AUTHORIZATION.

The Developer's proposal comes from a human-supplied JSON file: {"patched_files": {rel: text}, "report": text}.
Provider integrations are deliberately not part of v0.

The approval must be typed by a human. Wire this as a slash command only; never expose it as a model tool.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
from dataclasses import dataclass
from typing import List, Optional, Sequence

from osiris_cli import trio_authz
from osiris_cli.orchestration import evolution_bridge as eb
from osiris_cli.orchestration.agent_gene import spec_digest
from osiris_cli.orchestration.agent_spec import RunBudget
from osiris_cli.orchestration.trio import (CRITIC, DEVELOPER, RESEARCH, DeveloperOutput, run_trio,
                                           sandbox_limits, tree_digest)

MAX_FILES = 2000
MAX_BYTES = 50 * 1024 * 1024
MAX_PATCH_BYTES = 256 * 1024
TEST_TIMEOUT_S = 60.0
USAGE = ("usage: /trio plan <repo> [--patch FILE]\n"
         "       /trio run <repo> --patch FILE --out DIR --grant FILE --ledger FILE [--signature FILE] "
         "[--allowed-signers FILE]")


class TrioCommandError(ValueError):
    pass


@dataclass(frozen=True)
class TrioPlan:
    repo: str
    repo_tree_digest: str
    n_files: int
    patch_digest: str
    patched_paths: List[str]
    agents: List[dict]
    budget: dict
    plan_digest: str
    bundle: Optional[dict] = None
    bundle_digest: str = ""

    def render(self) -> str:
        lines = ["/trio plan (dry run; nothing executed, nothing written)",
                 f"  repo            {self.repo}  ({self.n_files} files, tree {self.repo_tree_digest[:16]}…)",
                 f"  patch           {self.patch_digest[:16] + '…' if self.patch_digest else '(none supplied)'}  "
                 f"{self.patched_paths}",
                 f"  budget          {self.budget}"]
        for a in self.agents:
            lines.append(f"  agent {a['role']:9} {a['digest'][:16]}…  caps={a['capabilities']}"
                         f"{'  [needs human approval]' if a['requires_human_approval'] else ''}")
        lines += ["  'run' would EXECUTE this repo's tests inside a bwrap namespace sandbox (no network, no home dir, rlimits; refuses if unavailable).",
                  "  'run' never applies the patch and never promotes; best outcome VERIFIED_PENDING_AUTHORIZATION.",
                  f"  plan digest     {self.plan_digest}",
                  f"  bundle digest   {self.bundle_digest}",
                  "  to authorize, the owner signs a grant for exactly this bundle digest:",
                  f"    python -m osiris_cli.grants draft --payload-digest {self.bundle_digest} "
                  f"--scope {trio_authz.SCOPE} > grant.json",
                  "    ssh-keygen -Y sign -f <owner key> -n osiris-grant grant.json",
                  "  then: /trio run <repo> --patch FILE --out DIR --grant grant.json --ledger FILE"]
        return "\n".join(lines)


def _within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _scan(repo: str):
    n = size = 0
    for d, dirs, files in os.walk(repo, followlinks=False):
        dirs[:] = [x for x in dirs if x not in (".git", "__pycache__", ".venv", "venv", "node_modules",
                                                ".pytest_cache")]
        for f in files:
            n += 1
            p = os.path.join(d, f)
            if os.path.islink(p):
                raise TrioCommandError(f"refusing repo containing a symlink: {os.path.relpath(p, repo)}")
            size += os.path.getsize(p)
            if n > MAX_FILES or size > MAX_BYTES:
                raise TrioCommandError(f"repo exceeds the v0 size bounds ({MAX_FILES} files / {MAX_BYTES >> 20} MB)")
    return n


def _load_patch(path: Optional[str]):
    if not path:
        return None, "", []
    if os.path.getsize(path) > MAX_PATCH_BYTES:
        raise TrioCommandError("patch file too large")
    try:
        data = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise TrioCommandError(f"patch file unreadable: {e}")
    files, report = data.get("patched_files") if isinstance(data, dict) else None, data.get("report") if isinstance(data, dict) else None
    if (not isinstance(files, dict) or not files or not isinstance(report, str)
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in files.items())
            or set(data) - {"patched_files", "report"}):
        raise TrioCommandError('patch file must be {"patched_files": {path: text}, "report": text} and nothing else')
    dg = hashlib.sha256(json.dumps({"files": files, "report": report}, sort_keys=True).encode()).hexdigest()
    return DeveloperOutput(files, report), dg, sorted(files)


def build_plan(repo: str, patch_file: Optional[str] = None, *, workspace_root: Optional[str] = None,
               provider: str = "none") -> tuple:
    """Returns (TrioPlan, DeveloperOutput | None). Pure read; no subprocess, no writes."""
    if provider != "none":
        raise TrioCommandError("/trio v0 has no model integration; only provider 'none' (a human-supplied patch)")
    root = os.path.realpath(workspace_root or os.getcwd())
    real = os.path.realpath(repo)
    if not os.path.isdir(real):
        raise TrioCommandError(f"not a directory: {repo}")
    if not _within(real, root):
        raise TrioCommandError(f"repo must be inside the workspace root {root}")
    if os.path.realpath(os.path.expanduser("~")) == real or real == os.path.sep:
        raise TrioCommandError("refusing to use a home or filesystem root directory as the repo")
    n = _scan(real)
    patch, pdg, paths = _load_patch(patch_file)
    agents = []
    for spec in (RESEARCH, DEVELOPER, CRITIC):
        blob, dg = spec_digest(spec)
        c = json.loads(blob)
        agents.append({"role": spec.role, "digest": dg, "capabilities": sorted(spec.capabilities),
                       "requires_human_approval": c["requires_human_approval"]})
    budget = {"max_agents": 3, "max_depth": 1, "test_timeout_s": TEST_TIMEOUT_S, "run_deadline_s": 300}
    tdig = tree_digest(real)
    pd = hashlib.sha256(json.dumps({"v": 1, "tree": tdig, "patch": pdg, "agents": [a["digest"] for a in agents],
                                    "budget": budget}, sort_keys=True).encode()).hexdigest()
    cand = eb.candidate_digest(patch.patched_files) if patch else ""
    bundle = trio_authz.make_bundle(plan_digest=pd, candidate_digest=cand, agents=agents, budget=budget,
                                    limits=sandbox_limits(TEST_TIMEOUT_S))
    return TrioPlan(real, tdig, n, pdg, paths, agents, budget, pd, bundle, trio_authz.bundle_digest(bundle)), patch


def run_trio_command(argv: Sequence[str], *, workspace_root: Optional[str] = None,
                     _grant_kw: Optional[dict] = None) -> str:
    """argv is what follows ``/trio``. Returns text for the REPL to print; raises nothing user-facing
    except TrioCommandError, which callers print."""
    args = list(argv)
    if not args:
        return USAGE
    sub = args[0] if args[0] in ("plan", "run") else "plan"
    rest = args[1:] if args[0] in ("plan", "run") else args
    opts = {"--patch": None, "--out": None, "--grant": None, "--ledger": None, "--signature": None,
            "--allowed-signers": None}
    pos: List[str] = []
    i = 0
    while i < len(rest):
        if rest[i] in opts:
            if i + 1 >= len(rest):
                raise TrioCommandError(f"{rest[i]} needs a value\n{USAGE}")
            opts[rest[i]] = rest[i + 1]
            i += 2
        elif rest[i] == "--approve":
            raise TrioCommandError("--approve (a bare digest) is no longer accepted: it is not human authorization. "
                                   "Use a signed single-use grant (--grant/--ledger); see `/trio plan`.")
        elif rest[i].startswith("--"):
            raise TrioCommandError(f"unknown option {rest[i]}\n{USAGE}")
        else:
            pos.append(rest[i])
            i += 1
    if len(pos) != 1:
        raise TrioCommandError(USAGE)

    plan, patch = build_plan(pos[0], opts["--patch"], workspace_root=workspace_root)
    if sub == "plan":
        return plan.render()

    if patch is None:
        raise TrioCommandError("run needs --patch FILE (the Developer's proposal)")
    if not opts["--out"]:
        raise TrioCommandError("run needs an explicit --out DIR (outside the repo); there is no default location")
    if not opts["--grant"] or not opts["--ledger"]:
        raise TrioCommandError("run needs --grant FILE (signed, single-use, bound to the bundle digest) and an "
                               "explicit --ledger FILE for its consumption record.\n" + plan.render())
    out = os.path.realpath(opts["--out"])
    if _within(out, plan.repo) or _within(plan.repo, out):
        raise TrioCommandError("--out must not be inside the repo (or contain it)")
    if _within(os.path.realpath(opts["--ledger"]), plan.repo):
        raise TrioCommandError("--ledger must not be inside the repo")
    from osiris_cli import grants, sandbox
    if not sandbox.available():          # checked BEFORE the grant is consumed: no sandbox, no spend, no run
        raise TrioCommandError("no namespace sandbox is available here; refusing (there is no unsandboxed mode)")
    os.makedirs(out, exist_ok=True)
    receipts = trio_authz.Receipts(os.path.join(out, "authz_receipts.jsonl"))
    kw = dict(_grant_kw or {})
    d = trio_authz.authorize_execution(
        plan.bundle, opts["--grant"], allowed_signers=opts["--allowed-signers"] or grants.ALLOWED_SIGNERS,
        ledger_path=opts["--ledger"], receipts=receipts, signature_path=opts["--signature"], **kw)
    if not d.authorized:
        raise TrioCommandError(f"not authorized: {d.reason}")
    # the grant is now spent; re-derive the bundle from disk so a swap after planning cannot ride on it
    plan2, patch2 = build_plan(pos[0], opts["--patch"], workspace_root=workspace_root)
    if plan2.bundle_digest != plan.bundle_digest:
        receipts.append("EXECUTION_ABORTED", {"reason": "inputs changed after authorization",
                                              "authorized_bundle": plan.bundle_digest,
                                              "current_bundle": plan2.bundle_digest})
        raise TrioCommandError("inputs changed between authorization and execution; nothing was run (grant spent)")
    res = run_trio(plan.repo, lambda f, sb: patch, out, timeout=TEST_TIMEOUT_S, budget=RunBudget(),
                   sandboxed=True)
    dec = eb.evaluate_candidate(res, os.path.join(out, "lineage.jsonl"))     # no authorizer: cannot promote
    assert dec.status != "PROMOTED"
    receipts.append("EXECUTION", {"run_id": res.run_id, "outcome": res.outcome, "candidate_status": dec.status,
                                  "candidate_digest": dec.candidate_digest, "bundle_digest": plan.bundle_digest,
                                  "sandboxed": True, "applied_to_repo": False,
                                  "grant_id": (d.record or {}).get("grant_id")})
    return "\n".join([
        f"/trio run {res.run_id}: trio outcome {res.outcome}; candidate {dec.status}",
        f"  authorized once by grant {(d.record or {}).get('grant_id', '')[:16]}… (key {(d.record or {}).get('key_fingerprint')})",
        f"  candidate {dec.candidate_id} ({dec.candidate_digest[:16]}…)  "
        f"baseline→candidate fitness: {dec.baseline_fitness}→{dec.candidate_fitness}",
        *[f"  - {r[:200]}" for r in dec.reasons[:4]],
        f"  record: {res.run_dir}   lineage: {os.path.join(out, 'lineage.jsonl')}   receipts: {receipts.path}",
        "  repo NOT modified; patch NOT applied; nothing promoted (promotion needs its own grant, scope "
        f"{eb.SCOPE}, bound to {dec.candidate_digest})."])


def split(line: str) -> List[str]:
    return shlex.split(line)
