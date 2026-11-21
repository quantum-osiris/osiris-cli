"""Polyglot AST, Import-Graph, and Multi-Workspace Codebase Visibility (`osiris_cli/codebase_visibility.py`).

Eliminates OSIRIS's codebase blind spots:
  1. Full-repository indexing with `(mtime_ns, size)` caching (no arbitrary 40-file `os.walk` cutoff).
  2. Polyglot symbol & route extraction:
     - Python (`.py`): AST classes, methods, functions, signatures, `L<start>-L<end>` spans,
       docstrings, and internal module import edges.
     - TypeScript / JavaScript / React / Next.js (`.ts`, `.tsx`, `.js`, `.jsx`): exported
       functions, classes, interfaces, types, Next.js App Router pages/API routes, and relative
       module imports.
     - DNA-Lang (`.dna`) and SQL (`.sql`): organisms, genes, tables, and migrations.
  3. Bidirectional dependency & blast-radius graph: both outgoing `imports` and incoming
     `imported_by` reverse-dependencies (hub centrality).
  4. Query-aware relevance ranking (token overlap across path, symbols, docstrings + hub boost)
     so Architect, Synthesizer, `/code`, and `/ask` receive the exact relevant files within a
     strict character budget.
  5. Multi-repo ecosystem awareness (`discover_ecosystem_repos`): discovers git root of `cwd`,
     OSIRIS's own source checkout (`osiris-cli`), and sibling workspace repositories
     (`quantum-ide`, `dnalang-core`, `osiris-governance`) for read-only visibility without
     widening write confinement.
"""

from __future__ import annotations

import ast
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

PathArg = Optional[Union[os.PathLike, str]]

SKIP_DIRS: Set[str] = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "node_modules",
    ".next",
    ".turbo",
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "site-packages",
    "dist",
    "build",
    "coverage",
    ".idea",
    ".vscode",
    "osiris_advantage_marrakesh",
}

SOURCE_EXTENSIONS: Tuple[str, ...] = (".py", ".ts", ".tsx", ".js", ".jsx", ".dna", ".sql")
MAX_SOURCE_BYTES = 512_000

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}")
_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# TypeScript / JS / DNA / SQL symbol patterns
_TS_EXPORT_FN_RE = re.compile(
    r"^\s*export\s+(?:default\s+)?(?:async\s+)?function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(([^)]*)\)",
    re.MULTILINE,
)
_TS_EXPORT_CONST_FN_RE = re.compile(
    r"^\s*export\s+const\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*(?::[^=]+)?=\s*(?:async\s*)?\(([^)]*)\)\s*=>",
    re.MULTILINE,
)
_TS_EXPORT_CLASS_RE = re.compile(
    r"^\s*export\s+(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][A-Za-z0-9_$]*)",
    re.MULTILINE,
)
_TS_EXPORT_TYPE_RE = re.compile(
    r"^\s*export\s+(?:interface|type|enum)\s+([A-Za-z_$][A-Za-z0-9_$]*)",
    re.MULTILINE,
)
_TS_IMPORT_RE = re.compile(
    r"""^\s*import\s+(?:type\s+)?(?:[^'"]*?\s+from\s+)?['"]([^'"]+)['"]""",
    re.MULTILINE,
)
_DNA_DECL_RE = re.compile(r"^\s*(organism|gene|rule|state|evidence)\s+([A-Za-z_][A-Za-z0-9_.-]*)", re.MULTILINE)
_SQL_TABLE_RE = re.compile(
    r"create\s+table\s+(?:if\s+not\s+exists\s+)?([A-Za-z0-9_.\"]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SymbolSummary:
    name: str
    kind: str  # class | function | method | interface | route | organism | table
    line: int
    end_line: int
    signature: str = ""
    doc: str = ""
    methods: Tuple[Tuple[str, str, int], ...] = ()  # (method_name, args_sig, line)


@dataclass
class FileNode:
    rel_path: str
    language: str
    line_count: int
    module_doc: str
    symbols: Tuple[SymbolSummary, ...]
    raw_imports: Tuple[str, ...]
    imports: Tuple[str, ...] = ()
    imported_by: Tuple[str, ...] = ()
    tokens: Set[str] = field(default_factory=set)


@dataclass(frozen=True)
class RepoIndex:
    root: str
    name: str
    git_branch: str
    files: Dict[str, FileNode]
    by_language: Dict[str, int]
    hub_files: Tuple[Tuple[str, int], ...]  # (rel_path, imported_by_count)


# Fast in-process file parse cache: (abs_path, mtime_ns, size) -> FileNode template
_FILE_PARSE_CACHE: Dict[Tuple[str, int, int], Optional[FileNode]] = {}


def _split_tokens(text: str) -> Set[str]:
    expanded = _CAMEL_SPLIT_RE.sub(" ", text or "")
    out: Set[str] = set()
    for tok in _TOKEN_RE.findall(expanded.lower()):
        out.add(tok)
        if "_" in tok:
            for sub in tok.split("_"):
                if len(sub) >= 2:
                    out.add(sub)
    return out


def _first_doc_line(doc: Optional[str]) -> str:
    if not doc:
        return ""
    for ln in doc.strip().splitlines():
        s = ln.strip()
        if s:
            return s[:140]
    return ""


def _py_args_sig(fn: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
    parts = [a.arg for a in getattr(fn.args, "posonlyargs", ())] + [a.arg for a in fn.args.args]
    if parts and parts[0] in ("self", "cls"):
        parts = parts[1:]
    if fn.args.vararg:
        parts.append("*" + fn.args.vararg.arg)
    for kw in fn.args.kwonlyargs:
        parts.append(kw.arg + "=")
    if fn.args.kwarg:
        parts.append("**" + fn.args.kwarg.arg)
    return ", ".join(parts[:8]) + (", ..." if len(parts) > 8 else "")


def _parse_python_file(src: str, rel_path: str) -> Optional[FileNode]:
    try:
        tree = ast.parse(src, filename=rel_path)
    except (SyntaxError, ValueError):
        return None

    symbols: List[SymbolSummary] = []
    raw_imports: List[str] = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                raw_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            dots = "." * (node.level or 0)
            mod = node.module or ""
            if dots or mod:
                raw_imports.append(f"{dots}{mod}")
            for alias in node.names:
                if mod:
                    raw_imports.append(f"{dots}{mod}.{alias.name}")
                elif dots:
                    raw_imports.append(f"{dots}{alias.name}")
        elif isinstance(node, ast.ClassDef):
            methods: List[Tuple[str, str, int]] = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and not item.name.startswith("_test"):
                    methods.append((item.name, _py_args_sig(item), getattr(item, "lineno", node.lineno)))
            end_ln = getattr(node, "end_lineno", None) or node.lineno
            symbols.append(
                SymbolSummary(
                    name=node.name,
                    kind="class",
                    line=node.lineno,
                    end_line=end_ln,
                    doc=_first_doc_line(ast.get_docstring(node)),
                    methods=tuple(methods[:16]),
                )
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end_ln = getattr(node, "end_lineno", None) or node.lineno
            symbols.append(
                SymbolSummary(
                    name=node.name,
                    kind="function",
                    line=node.lineno,
                    end_line=end_ln,
                    signature=_py_args_sig(node),
                    doc=_first_doc_line(ast.get_docstring(node)),
                )
            )

    # Also capture module-level imports inside functions/try blocks for accurate dependency edges
    for sub in ast.walk(tree):
        if isinstance(sub, ast.Import):
            for alias in sub.names:
                raw_imports.append(alias.name)
        elif isinstance(sub, ast.ImportFrom):
            dots = "." * (sub.level or 0)
            mod = sub.module or ""
            if dots or mod:
                raw_imports.append(f"{dots}{mod}")

    mod_doc = _first_doc_line(ast.get_docstring(tree))
    line_count = src.count("\n") + 1
    toks = _split_tokens(rel_path) | _split_tokens(mod_doc)
    for sym in symbols:
        toks |= _split_tokens(sym.name) | _split_tokens(sym.doc)
        for mname, _msig, _mln in sym.methods:
            toks |= _split_tokens(mname)

    return FileNode(
        rel_path=rel_path,
        language="python",
        line_count=line_count,
        module_doc=mod_doc,
        symbols=tuple(symbols),
        raw_imports=tuple(dict.fromkeys(raw_imports)),
        tokens=toks,
    )


def _line_of_offset(src: str, offset: int) -> int:
    return src.count("\n", 0, max(0, offset)) + 1


def _parse_ts_js_file(src: str, rel_path: str, ext: str) -> FileNode:
    symbols: List[SymbolSummary] = []
    norm_rel = rel_path.replace(os.sep, "/")

    # Next.js App Router detection (page.tsx / route.ts / layout.tsx)
    base = os.path.basename(norm_rel)
    if base in ("page.tsx", "page.ts", "page.jsx", "route.ts", "route.js"):
        kind_label = "api_route" if base.startswith("route.") else "page_route"
        symbols.append(SymbolSummary(name=norm_rel, kind=kind_label, line=1, end_line=1))

    for m in _TS_EXPORT_CLASS_RE.finditer(src):
        ln = _line_of_offset(src, m.start())
        symbols.append(SymbolSummary(name=m.group(1), kind="class", line=ln, end_line=ln))
    for m in _TS_EXPORT_FN_RE.finditer(src):
        ln = _line_of_offset(src, m.start())
        sig = " ".join(m.group(2).split())[:60]
        symbols.append(SymbolSummary(name=m.group(1), kind="function", line=ln, end_line=ln, signature=sig))
    for m in _TS_EXPORT_CONST_FN_RE.finditer(src):
        ln = _line_of_offset(src, m.start())
        sig = " ".join(m.group(2).split())[:60]
        symbols.append(SymbolSummary(name=m.group(1), kind="function", line=ln, end_line=ln, signature=sig))
    for m in _TS_EXPORT_TYPE_RE.finditer(src):
        ln = _line_of_offset(src, m.start())
        symbols.append(SymbolSummary(name=m.group(1), kind="interface", line=ln, end_line=ln))

    raw_imports = [m.group(1).strip() for m in _TS_IMPORT_RE.finditer(src)]
    first_comment = ""
    cm = re.match(r"\s*(?:/\*\*?(.*?)\*/|//\s*(.*))", src, re.DOTALL)
    if cm:
        first_comment = _first_doc_line((cm.group(1) or cm.group(2) or "").replace("*", " "))

    lang = "typescript" if ext in (".ts", ".tsx") else "javascript"
    toks = _split_tokens(rel_path) | _split_tokens(first_comment)
    for sym in symbols:
        toks |= _split_tokens(sym.name)

    return FileNode(
        rel_path=rel_path,
        language=lang,
        line_count=src.count("\n") + 1,
        module_doc=first_comment,
        symbols=tuple(symbols[:30]),
        raw_imports=tuple(dict.fromkeys(raw_imports)),
        tokens=toks,
    )


def _parse_dna_or_sql_file(src: str, rel_path: str, ext: str) -> FileNode:
    symbols: List[SymbolSummary] = []
    if ext == ".dna":
        for m in _DNA_DECL_RE.finditer(src):
            ln = _line_of_offset(src, m.start())
            symbols.append(SymbolSummary(name=m.group(2), kind=m.group(1), line=ln, end_line=ln))
        lang = "dnalang"
    else:
        for m in _SQL_TABLE_RE.finditer(src):
            ln = _line_of_offset(src, m.start())
            symbols.append(SymbolSummary(name=m.group(1).strip('"'), kind="table", line=ln, end_line=ln))
        lang = "sql"
    toks = _split_tokens(rel_path)
    for sym in symbols:
        toks |= _split_tokens(sym.name)
    return FileNode(
        rel_path=rel_path,
        language=lang,
        line_count=src.count("\n") + 1,
        module_doc="",
        symbols=tuple(symbols[:25]),
        raw_imports=(),
        tokens=toks,
    )


def _parse_source_file(abs_path: str, rel_path: str) -> Optional[FileNode]:
    try:
        st = os.stat(abs_path)
    except OSError:
        return None
    if st.st_size <= 0 or st.st_size > MAX_SOURCE_BYTES:
        return None

    cache_key = (abs_path, st.st_mtime_ns, st.st_size)
    if cache_key in _FILE_PARSE_CACHE:
        cached = _FILE_PARSE_CACHE[cache_key]
        if cached is None:
            return None
        return FileNode(
            rel_path=rel_path,
            language=cached.language,
            line_count=cached.line_count,
            module_doc=cached.module_doc,
            symbols=cached.symbols,
            raw_imports=cached.raw_imports,
            tokens=set(cached.tokens),
        )

    ext = os.path.splitext(abs_path)[1].lower()
    try:
        with open(abs_path, "r", encoding="utf-8", errors="ignore") as fh:
            src = fh.read()
    except OSError:
        return None

    if ext == ".py":
        parsed = _parse_python_file(src, rel_path)
    elif ext in (".ts", ".tsx", ".js", ".jsx"):
        parsed = _parse_ts_js_file(src, rel_path, ext)
    elif ext in (".dna", ".sql"):
        parsed = _parse_dna_or_sql_file(src, rel_path, ext)
    else:
        parsed = None

    if len(_FILE_PARSE_CACHE) > 2500:
        _FILE_PARSE_CACHE.clear()
    _FILE_PARSE_CACHE[cache_key] = parsed
    return parsed


def read_git_branch(root: str) -> str:
    """Read `.git/HEAD` directly from disk without spawning a subprocess."""
    dot_git = os.path.join(root, ".git")
    head_path: Optional[str] = None
    if os.path.isdir(dot_git):
        head_path = os.path.join(dot_git, "HEAD")
    elif os.path.isfile(dot_git):
        # Git worktree pointer: "gitdir: /path/to/.git/worktrees/..."
        try:
            raw = Path(dot_git).read_text(encoding="utf-8").strip()
            if raw.startswith("gitdir:"):
                gdir = raw.split(":", 1)[1].strip()
                if not os.path.isabs(gdir):
                    gdir = os.path.normpath(os.path.join(root, gdir))
                head_path = os.path.join(gdir, "HEAD")
        except OSError:
            head_path = None
    if not head_path or not os.path.isfile(head_path):
        return ""
    try:
        head_txt = Path(head_path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if head_txt.startswith("ref: refs/heads/"):
        return head_txt[len("ref: refs/heads/"):]
    return head_txt[:12]


def find_git_root(start: PathArg = None) -> Optional[str]:
    cur = os.path.realpath(str(start) if start is not None else os.getcwd())
    while True:
        if os.path.exists(os.path.join(cur, ".git")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


def osiris_cli_repo_root() -> str:
    return os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def discover_ecosystem_repos(cwd: PathArg = None) -> List[Dict[str, str]]:
    """Discover the active `cwd` repo, OSIRIS's own checkout, and sibling ecosystem repos."""
    base_cwd = os.path.realpath(str(cwd) if cwd is not None else os.getcwd())
    home = os.path.expanduser("~")
    own_root = osiris_cli_repo_root()

    candidate_dirs: List[Tuple[str, str]] = []
    cwd_git = find_git_root(base_cwd)
    if cwd_git:
        candidate_dirs.append(("cwd", cwd_git))
    candidate_dirs.append(("osiris-cli", own_root))

    for parent in (
        os.path.dirname(own_root),
        os.path.join(home, "osiris-assessment", "work"),
        os.path.join(home, "osiris-repos"),
        home,
    ):
        for sub in ("osiris-cli", "quantum-ide", "dnalang-core", "osiris-governance"):
            cand = os.path.join(parent, sub)
            if os.path.isdir(cand) and os.path.exists(os.path.join(cand, ".git")):
                candidate_dirs.append((sub, os.path.realpath(cand)))

    seen: Set[str] = set()
    repos: List[Dict[str, str]] = []
    for role_hint, rpath in candidate_dirs:
        real = os.path.realpath(rpath)
        if real in seen or not os.path.isdir(real):
            continue
        seen.add(real)
        repos.append({
            "name": os.path.basename(real),
            "role": role_hint,
            "root": real,
            "branch": read_git_branch(real) or "(untracked)",
        })
    return repos


def _resolve_import_edges(files: Dict[str, FileNode]) -> None:
    """Link Python dotted imports and TS/JS relative imports to internal `rel_path` targets."""
    py_module_map: Dict[str, str] = {}
    ts_stem_map: Dict[str, str] = {}
    for rel in files:
        norm = rel.replace(os.sep, "/")
        if norm.endswith(".py"):
            dotted = norm[:-3].replace("/", ".")
            py_module_map[dotted] = norm
            if dotted.endswith(".__init__"):
                py_module_map[dotted[: -len(".__init__")]] = norm
        elif norm.endswith((".ts", ".tsx", ".js", ".jsx")):
            stem = os.path.splitext(norm)[0]
            ts_stem_map[stem] = norm
            if stem.endswith("/index"):
                ts_stem_map[stem[: -len("/index")]] = norm

    incoming: Dict[str, Set[str]] = {rel: set() for rel in files}
    for rel, node in files.items():
        outgoing: Set[str] = set()
        pkg_parts = rel.replace(os.sep, "/").split("/")[:-1]
        for raw in node.raw_imports:
            if node.language == "python":
                if raw.startswith("."):
                    level = len(raw) - len(raw.lstrip("."))
                    tail = raw.lstrip(".")
                    up = max(0, len(pkg_parts) - (level - 1))
                    prefix = ".".join(pkg_parts[:up])
                    full_mod = f"{prefix}.{tail}".strip(".") if tail else prefix
                else:
                    full_mod = raw
                target = py_module_map.get(full_mod)
                if target is None and "." in full_mod:
                    target = py_module_map.get(full_mod.rsplit(".", 1)[0])
                if target and target != rel:
                    outgoing.add(target)
                    incoming[target].add(rel)
            elif node.language in ("typescript", "javascript"):
                cand_rel: Optional[str] = None
                if raw.startswith("."):
                    joined = os.path.normpath(os.path.join(os.path.dirname(rel), raw)).replace(os.sep, "/")
                    cand_rel = ts_stem_map.get(joined) or (joined if joined in files else None)
                elif raw.startswith("@/"):
                    stripped = raw[2:]
                    for prefix_dir in ("src/", ""):
                        cand_rel = ts_stem_map.get(prefix_dir + stripped)
                        if cand_rel:
                            break
                if cand_rel and cand_rel != rel and cand_rel in incoming:
                    outgoing.add(cand_rel)
                    incoming[cand_rel].add(rel)

        node.imports = tuple(sorted(outgoing))

    for rel, node in files.items():
        node.imported_by = tuple(sorted(incoming.get(rel, set())))


def index_repository(root: PathArg = None, *, max_scan_files: int = 1500) -> RepoIndex:
    """Scan all source files under `root`, build symbol + dependency graphs, and identify hub modules."""
    abs_root = os.path.realpath(str(root) if root is not None else os.getcwd())
    files: Dict[str, FileNode] = {}
    by_lang: Dict[str, int] = {}

    if os.path.isdir(abs_root):
        for dirpath, dirnames, filenames in os.walk(abs_root):
            dirnames[:] = sorted(
                d
                for d in dirnames
                if d not in SKIP_DIRS
                and not d.startswith(".")
                and "venv" not in d.lower()
                and not d.endswith((".egg-info", ".dist-info"))
            )
            for fn in sorted(filenames):
                ext = os.path.splitext(fn)[1].lower()
                if ext not in SOURCE_EXTENSIONS:
                    continue
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, abs_root).replace(os.sep, "/")
                parsed = _parse_source_file(full, rel)
                if parsed is not None:
                    files[rel] = parsed
                    by_lang[parsed.language] = by_lang.get(parsed.language, 0) + 1
                    if len(files) >= max_scan_files:
                        break
            if len(files) >= max_scan_files:
                break

    _resolve_import_edges(files)
    hubs = sorted(
        (
            (rel, len(node.imported_by))
            for rel, node in files.items()
            if node.imported_by and os.path.basename(rel) != "__init__.py"
        ),
        key=lambda item: (-item[1], item[0]),
    )[:12]

    return RepoIndex(
        root=abs_root,
        name=os.path.basename(abs_root) or abs_root,
        git_branch=read_git_branch(abs_root),
        files=files,
        by_language=by_lang,
        hub_files=tuple(hubs),
    )


def score_file_for_query(node: FileNode, query_tokens: Set[str], raw_query_lower: str) -> float:
    """Rank a `FileNode` by exact filename/symbol hits, token overlap, and dependency centrality."""
    is_test = node.rel_path.startswith("tests/") or "/test_" in ("/" + node.rel_path)
    hub_score = min(3.0, len(node.imported_by) * 0.35)
    base_prior = (0.2 if is_test else 1.2) + hub_score

    if not query_tokens:
        return base_prior + min(1.5, len(node.symbols) * 0.08)

    overlap = query_tokens & node.tokens
    score = float(len(overlap)) * 2.5 + hub_score * 0.5
    base_name = os.path.splitext(os.path.basename(node.rel_path))[0].lower()
    if base_name in query_tokens or node.rel_path.lower() in raw_query_lower:
        score += 8.0
    for sym in node.symbols:
        sym_low = sym.name.lower()
        if sym_low in query_tokens or (len(sym_low) >= 4 and sym_low in raw_query_lower):
            score += 5.0
    if is_test and "test" not in query_tokens and "pytest" not in query_tokens:
        score *= 0.45
    return score


def rank_files(index: RepoIndex, query: str = "", *, limit: int = 25) -> List[Tuple[float, FileNode]]:
    q_tokens = _split_tokens(query)
    q_low = (query or "").lower()
    scored = [(score_file_for_query(node, q_tokens, q_low), node) for node in index.files.values()]
    scored.sort(key=lambda item: (-item[0], item[1].rel_path))
    if q_tokens:
        # Prefer positive matches first, then top architectural hubs so context is never empty
        matched = [it for it in scored if (q_tokens & it[1].tokens)]
        rest = [it for it in scored if not (q_tokens & it[1].tokens)]
        ordered = matched + rest
    else:
        ordered = scored
    return ordered[: max(1, limit)]


def resolve_read_only_repo(cwd: PathArg = None, query: str = "") -> Tuple[str, str]:
    """Choose the best repository root for read-only codebase queries (`/codebase`, Architect context).

    If `cwd` contains source files, uses `cwd` (or its git root) unless `query` explicitly names a
    sibling repo (`quantum-ide`, `osiris-cli`, `dnalang-core`). If `cwd` has 0 source files (e.g. `~`),
    transparently falls back to `osiris-cli` and notes why.
    """
    base_cwd = os.path.realpath(str(cwd) if cwd is not None else os.getcwd())
    ecosys = discover_ecosystem_repos(base_cwd)
    q_low = (query or "").lower()

    for repo in ecosys:
        rname = repo["name"].lower()
        if rname in q_low or rname.replace("-", "_") in q_low:
            return repo["root"], f"matched ecosystem repo {repo['name']}"
    if any(w in q_low for w in ("nextjs", "tsx", "react", "vercel", "quantum-ide", "ide ui")):
        for repo in ecosys:
            if repo["name"] == "quantum-ide":
                return repo["root"], "matched quantum-ide frontend/API repo"

    cwd_git = find_git_root(base_cwd)
    probe_root = cwd_git or base_cwd
    probe_idx = index_repository(probe_root, max_scan_files=40)
    if probe_idx.files:
        return probe_root, "active working directory"

    own = osiris_cli_repo_root()
    return own, f"cwd ({base_cwd}) had 0 indexed source files; grounded in osiris-cli ({own})"


def format_workspace_map_for_prompt(
    root: PathArg = None,
    query: str = "",
    *,
    max_files: int = 12,
    max_chars: int = 1800,
) -> str:
    """Compact `<workspace_map>` block injected into `/code` and Architect synthesis prompts."""
    idx = index_repository(root)
    if not idx.files:
        return ""
    ranked = rank_files(idx, query=query, limit=max_files)
    lang_str = ", ".join(f"{k}:{v}" for k, v in sorted(idx.by_language.items()))
    lines = [
        f"<workspace_map root={idx.root!r} branch={idx.git_branch or 'n/a'!r} "
        f"total_files={len(idx.files)} languages={lang_str!r}>"
    ]
    for _score, node in ranked:
        sym_parts: List[str] = []
        for sym in node.symbols[:6]:
            if sym.kind == "class":
                mnames = ",".join(m[0] for m in sym.methods[:4])
                sym_parts.append(f"class {sym.name}@L{sym.line}" + (f"({mnames})" if mnames else ""))
            else:
                sym_parts.append(f"{sym.name}@L{sym.line}")
        dep_note = f" [imported_by={len(node.imported_by)}]" if node.imported_by else ""
        sym_txt = (": " + ", ".join(sym_parts)) if sym_parts else ""
        entry = f"  {node.rel_path} ({node.line_count}L){dep_note}{sym_txt}"
        if sum(len(x) + 1 for x in lines) + len(entry) + 24 > max_chars:
            break
        lines.append(entry)
    lines.append("</workspace_map>")
    return "\n".join(lines)


def format_dependency_inspector(index: RepoIndex, target: str) -> str:
    clean = target.strip().strip("'\"").replace("\\", "/")
    exact = index.files.get(clean)
    if exact is None:
        matches = [
            node
            for rel, node in index.files.items()
            if clean in rel or os.path.splitext(os.path.basename(rel))[0] == clean.replace(".py", "")
        ]
        if not matches:
            return f"Module/file {target!r} not found under {index.root} ({len(index.files)} indexed files)."
        matches.sort(key=lambda n: (len(n.rel_path), n.rel_path))
        exact = matches[0]

    lines = [
        f"DEPENDENCY & BLAST-RADIUS MAP · {exact.rel_path} ({exact.language}, {exact.line_count} lines)",
        f"  repo: {index.root}" + (f" [{index.git_branch}]" if index.git_branch else ""),
    ]
    if exact.module_doc:
        lines.append(f"  doc:  {exact.module_doc}")
    lines.append(f"  imports ({len(exact.imports)} internal):")
    if exact.imports:
        for imp in exact.imports[:20]:
            lines.append(f"    -> {imp}")
    else:
        lines.append("    (none -- leaf/stdlib-only module)")
    lines.append(f"  imported_by ({len(exact.imported_by)} reverse dependents / blast radius):")
    if exact.imported_by:
        for rev in exact.imported_by[:60]:
            lines.append(f"    <- {rev}")
        if len(exact.imported_by) > 60:
            lines.append(f"    ... and {len(exact.imported_by) - 60} more")
    else:
        lines.append("    (0 internal callers)")
    return "\n".join(lines)


USAGE = (
    "Usage:\n"
    "  /codebase                       summary of active workspace, hub modules & ecosystem repos\n"
    "  /codebase <query>               rank relevant files & symbols (classes/functions @ line)\n"
    "  /codebase deps <file_or_module> show internal imports + reverse-dependency blast radius\n"
    "  /codebase repos                 list discovered OSIRIS ecosystem git repositories"
)


def run_command(args: str = "", *, cwd: PathArg = None) -> str:
    """CLI handler for `/codebase` and `/repo`."""
    try:
        tokens = shlex.split(args or "")
    except ValueError:
        tokens = (args or "").strip().split()

    base_cwd = os.path.realpath(str(cwd) if cwd is not None else os.getcwd())
    if tokens and tokens[0].lower() in ("help", "-h", "--help"):
        return USAGE

    ecosys = discover_ecosystem_repos(base_cwd)
    if tokens and tokens[0].lower() == "repos":
        out = ["DISCOVERED OSIRIS ECOSYSTEM REPOSITORIES:"]
        for r in ecosys:
            idx = index_repository(r["root"])
            langs = ", ".join(f"{k}={v}" for k, v in sorted(idx.by_language.items())) or "0 source files"
            out.append(f"  {r['name']:<20} [{r['branch']}]  {r['root']}  ({len(idx.files)} files: {langs})")
        return "\n".join(out)

    if tokens and tokens[0].lower() == "deps" and len(tokens) >= 2:
        target_arg = " ".join(tokens[1:])
        repo_root, _why = resolve_read_only_repo(base_cwd, target_arg)
        return format_dependency_inspector(index_repository(repo_root), target_arg)

    query = " ".join(tokens).strip()
    repo_root, reason = resolve_read_only_repo(base_cwd, query)
    idx = index_repository(repo_root)
    langs = ", ".join(f"{k}={v}" for k, v in sorted(idx.by_language.items())) or "none"

    if not query:
        lines = [
            f"CODEBASE VISIBILITY · {idx.name} ({len(idx.files)} source files · {langs})",
            f"  root:   {idx.root}  (branch: {idx.git_branch or 'n/a'} · {reason})",
        ]
        if idx.hub_files:
            lines.append("  architectural hubs (most-imported internal modules):")
            for rel, cnt in idx.hub_files[:8]:
                fnode = idx.files[rel]
                doc_part = f" -- {fnode.module_doc[:52]}" if fnode.module_doc else ""
                lines.append(f"    {rel:<42} imported_by={cnt:<2}{doc_part}")
        if len(ecosys) > 1:
            lines.append("  ecosystem repos (query by name or `/codebase repos`):")
            for r in ecosys[:5]:
                lines.append(f"    - {r['name']} [{r['branch']}] -> {r['root']}")
        lines.append("  tip: `/codebase <topic>` ranks symbols; `/codebase deps <module>` shows blast radius.")
        return "\n".join(lines)

    ranked = rank_files(idx, query=query, limit=10)
    lines = [
        f"CODEBASE SEARCH · query={query!r} in {idx.name} ({idx.root})",
    ]
    for score, node in ranked:
        lines.append(
            f"  [{score:4.1f}] {node.rel_path} ({node.language}, {node.line_count}L, imported_by={len(node.imported_by)})"
        )
        if node.module_doc:
            lines.append(f"         doc: {node.module_doc[:76]}")
        sym_strs: List[str] = []
        for sym in node.symbols[:6]:
            sig = f"({sym.signature})" if sym.signature else ""
            sym_strs.append(f"{sym.kind} {sym.name}{sig}@L{sym.line}")
        if sym_strs:
            lines.append("         symbols: " + ", ".join(sym_strs))
    return "\n".join(lines)
