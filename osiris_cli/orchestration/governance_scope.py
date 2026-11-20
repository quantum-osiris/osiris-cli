"""Governance files no agent may write, matched by repo-relative path (SYS-025, REQ-SEC-6).

The earlier check compared basenames against {policy.json, cloudbuild.yaml, gemini.md, ...}. That
missed the files that actually hold authority: ``osiris_cli/policy/exec_policy.json`` (basename
``exec_policy.json``) and ``governance/allowed_signers``. Here a target is resolved with
``realpath`` (so a symlink cannot launder it) and compared as a path relative to every repository
root that contains it: each allowed scope root, the nearest enclosing git checkout, and this
package's own checkout. The old basenames stay refused too, so the rule only got stricter.

Agent origins (model, skill, claude_code) can never target these paths. A person changes them by
hand, under review.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional

#: exact repo-relative files
IMMUTABLE_GOVERNANCE_FILES = frozenset({
    "osiris_cli/policy/exec_policy.json",
    "governance/allowed_signers",
    "cloudbuild.yaml",
    "GEMINI.md",
    "POLICY.md",
    "AGENTS.md",
    "CLAUDE.md",
    "quarantine_notice.md",
})

#: repo-relative directories: everything below them
IMMUTABLE_GOVERNANCE_DIRS = frozenset({
    ".osiris_compliance",
    "contracts",
    "governance",
    "osiris_cli/policy",
    ".git",
    ".github/workflows",
})

#: basenames refused anywhere (the earlier rule, kept so nothing it refused becomes writable)
LEGACY_BASENAMES = frozenset({
    "policy.json", "exec_policy.json", "allowed_signers", "cloudbuild.yaml",
    "quarantine_notice.md", "gemini.md", "policy.md", "agents.md", "claude.md",
    ".osiris_compliance",
})

_PACKAGE_REPO = Path(__file__).resolve().parents[2]


def _git_root(start: str) -> Optional[str]:
    cur = start if os.path.isdir(start) else os.path.dirname(start)
    while True:
        if os.path.exists(os.path.join(cur, ".git")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def _relative(path: str, root: str) -> Optional[str]:
    root = os.path.realpath(root)
    if path == root:
        return ""
    if path.startswith(root.rstrip(os.sep) + os.sep):
        return Path(os.path.relpath(path, root)).as_posix()
    return None


def candidate_roots(target: str, roots: Iterable[str] = ()) -> List[str]:
    resolved = os.path.realpath(target)
    found = [os.path.realpath(r) for r in roots]
    git = _git_root(resolved)
    if git:
        found.append(os.path.realpath(git))
    found.append(str(_PACKAGE_REPO))
    out: List[str] = []
    for r in found:
        if r not in out:
            out.append(r)
    return out


def protected_reason(target: str, roots: Iterable[str] = ()) -> Optional[str]:
    """Why ``target`` is a governance file, or None if it is not one.

    Both the path as given and its realpath are checked, so neither a symlink to a governance file
    nor a symlink named like one gets through.
    """
    paths = {os.path.abspath(target), os.path.realpath(target)}
    for p in paths:
        base = os.path.basename(p).lower()
        if base in LEGACY_BASENAMES:
            return f"'{target}' is an immutable governance artifact ({base})"
        for root in candidate_roots(p, roots):
            rel = _relative(p, root)
            if rel is None or rel == "":
                continue
            if rel in IMMUTABLE_GOVERNANCE_FILES:
                return f"'{target}' is an immutable governance artifact ({rel})"
            for d in IMMUTABLE_GOVERNANCE_DIRS:
                if rel == d or rel.startswith(d + "/"):
                    return f"'{target}' is inside the governance directory {d}/"
    return None


def is_protected(target: str, roots: Iterable[str] = ()) -> bool:
    return protected_reason(target, roots) is not None
