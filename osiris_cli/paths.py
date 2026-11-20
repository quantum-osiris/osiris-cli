"""Where OSIRIS looks for optional sibling checkouts and where it writes local results.

Earlier versions put one machine's absolute paths (including the whole home directory)
at the *front* of ``sys.path``. That let stale copies in the home directory shadow the
installed package, and made tests import the live working tree. Now:

* the source checkout, entries of ``OSIRIS_EXTRA_PATHS`` (``os.pathsep``-separated) and
  sibling research checkouts that exist under the home directory are *appended*, so
  installed packages always take precedence;
* results go under the checkout's ``results/`` when running from a source checkout,
  otherwise under ``~/.osiris/results/``.
"""

import os
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent

# Separately published research repositories the console can use when checked out
# next to the home directory (each import of them is guarded).
SIBLING_CHECKOUTS = (
    "dnalang-core",
    "organism_sim",
    "bridge",
    "flywheel-2026",
    "qbyte_system",
    "fold",
    "osiris-mobile-termux",
    "osiris-governance/src",
)


def add_optional_paths() -> List[str]:
    """Append (never prepend) existing optional locations to ``sys.path``; return those added."""
    work_root = Path.home() / "osiris-assessment" / "work"
    candidates = [str(REPO_ROOT)]
    candidates += [p for p in os.environ.get("OSIRIS_EXTRA_PATHS", "").split(os.pathsep) if p]
    candidates += [str(Path.home() / name) for name in SIBLING_CHECKOUTS]
    candidates += [str(work_root / name) for name in SIBLING_CHECKOUTS]
    candidates += [str(Path.home())]
    added = []
    for path in candidates:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)
            added.append(path)
    return added


def results_dir(*parts: str) -> Path:
    """``<checkout>/results/...`` in a source checkout, else ``~/.osiris/results/...``."""
    if (REPO_ROOT / "pyproject.toml").is_file():
        base = REPO_ROOT / "results"
    else:
        base = Path.home() / ".osiris" / "results"
    return base.joinpath(*parts)


def _resolve_osiris_home(environ: dict = None, home: Path = None) -> Path:
    """``$OSIRIS_HOME`` when set, else ``~/.osiris``. Never a hardcoded absolute
    path: Node Alpha (Termux) and Node Beta (WSL2) resolve their own home."""
    env = os.environ if environ is None else environ
    root = Path(env.get("OSIRIS_HOME") or "").expanduser() if env.get("OSIRIS_HOME") else None
    if root is not None:
        return root
    return (home or Path.home()) / ".osiris"


def _resolve_dnalang_ledger(osiris_home: Path, repo_root: Path = None, environ: dict = None,
                           home: Path = None) -> Path:
    """Locate dnalang-core's ``dnalang/ledger.py`` without assuming one machine's
    layout. Lookup order:

    1. ``$OSIRIS_DNALANG_LEDGER`` (explicit override);
    2. ``~/osiris-repos/dnalang-core/dnalang/ledger.py`` (Node Alpha checkout);
    3. ``<osiris_home>/src/dnalang-core/dnalang/ledger.py`` (installed under OSIRIS_HOME);
    4. ``~/dnalang-core/dnalang/ledger.py`` (home checkout);
    5. ``<repo_root>/../.osiris/src/dnalang-core/...`` and ``<repo_root>/../dnalang-core/...``
       (legacy sibling-workdir layouts, probed relative to the checkout, not hardcoded);
    6. an installed ``dnalang`` package (located without importing it).

    Returns the first candidate that exists, else candidate 2 (the canonical
    OSIRIS_HOME location) so error messages name a stable path.
    """
    env = os.environ if environ is None else environ
    home = home or Path.home()
    root = repo_root or REPO_ROOT
    candidates = []
    if env.get("OSIRIS_DNALANG_LEDGER"):
        candidates.append(Path(env["OSIRIS_DNALANG_LEDGER"]).expanduser())
    candidates += [
        home / "osiris-repos" / "dnalang-core" / "dnalang" / "ledger.py",
        osiris_home / "src" / "dnalang-core" / "dnalang" / "ledger.py",
        home / "dnalang-core" / "dnalang" / "ledger.py",
        root.parent / ".osiris" / "src" / "dnalang-core" / "dnalang" / "ledger.py",
        root.parent / "dnalang-core" / "dnalang" / "ledger.py",
    ]
    try:
        import importlib.util
        spec = importlib.util.find_spec("dnalang")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.submodule_search_locations:
        candidates += [Path(p) / "ledger.py" for p in spec.submodule_search_locations]
    for cand in candidates:
        if cand.is_file():
            return cand
    return candidates[1] if len(candidates) > 1 else candidates[0]
