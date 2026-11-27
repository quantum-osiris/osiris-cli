"""Where OSIRIS looks for optional sibling checkouts and where it writes local results.

Earlier versions put one machine's absolute paths (including the whole home directory)
at the *front* of ``sys.path``. That let stale copies in the home directory shadow the
installed package, and made tests import the live working tree. Now:

* the source checkout, entries of ``OSIRIS_EXTRA_PATHS`` (``os.pathsep``-separated) and
  named sibling research checkouts are *appended*, so installed packages always
  take precedence. The home directory itself is not put on ``sys.path``.
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
    added = []
    for path in candidates:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)
            added.append(path)
    return added


def trusted_module_origin(module_name: str):
    """Return the file a module would load from, if that file is an install or this checkout.

    ``importlib.util.find_spec`` does not execute the module. A file planted directly
    in the home directory is not trusted, even when that directory is on ``sys.path``.
    """
    import importlib.util

    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ValueError):
        return None
    origin = getattr(spec, "origin", None) if spec is not None else None
    if not origin or origin == "namespace" or not os.path.isfile(origin):
        return None
    resolved = Path(origin).resolve()
    roots = [REPO_ROOT.resolve(), Path(sys.prefix).resolve()]
    try:
        import site
        roots.extend(Path(path).resolve() for path in site.getsitepackages())
        user_site = site.getusersitepackages()
        if user_site:
            roots.append(Path(user_site).resolve())
    except Exception:
        pass
    if resolved.parent == Path.home().resolve():
        return None
    if not any(resolved == root or root in resolved.parents for root in roots):
        return None
    return resolved


def import_trusted_attr(module_name: str, attr: str):
    """Import ``attr`` only after its module file is inside a trusted root."""
    import importlib

    origin = trusted_module_origin(module_name)
    if origin is None:
        raise ImportError(f"{module_name} is not in a trusted install or checkout")
    module = importlib.import_module(module_name)
    loaded = Path(getattr(module, "__file__", "")).resolve()
    if loaded != origin:
        raise ImportError(f"{module_name} origin {loaded} is not the trusted file {origin}")
    if not hasattr(module, attr):
        raise ImportError(f"{module_name} has no attribute {attr}")
    return getattr(module, attr)


def results_dir(*parts: str) -> Path:
    """``<checkout>/results/...`` in a source checkout, else ``~/.osiris/results/...``."""
    if (REPO_ROOT / "pyproject.toml").is_file():
        base = REPO_ROOT / "results"
    else:
        base = Path.home() / ".osiris" / "results"
    return base.joinpath(*parts)
