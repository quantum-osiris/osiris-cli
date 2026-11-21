"""
osiris_ast_bridge.py
Lightweight, stdlib-only (ast + os + json) filesystem-to-context bridge for
the OSIRIS Dual-Engine REPL. Scans real .py files under a project root and
emits a compact JSON summary of classes/functions/signatures/docstrings, so
the Architect model (llama3.2:1b) grounds its "technical specification" in
the actual codebase instead of free-associating a generic boilerplate.

No network calls, no third-party deps -- must run cheaply on-device.
"""

import ast
import json
import os

SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env", "node_modules",
    ".cache", ".mypy_cache", ".pytest_cache", "site-packages",
    "dist", "build", ".idea", ".vscode",
}


def _first_line(doc):
    if not doc:
        return ""
    return doc.strip().splitlines()[0][:120]


def _args_sig(fn: ast.FunctionDef) -> str:
    parts = [a.arg for a in fn.args.args]
    if fn.args.vararg:
        parts.append("*" + fn.args.vararg.arg)
    if fn.args.kwarg:
        parts.append("**" + fn.args.kwarg.arg)
    return ", ".join(parts)


def _summarize_file(path: str, rel: str):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            src = f.read()
        tree = ast.parse(src, filename=rel)
    except (SyntaxError, ValueError, UnicodeDecodeError):
        return None

    classes = []
    functions = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [
                {"name": n.name, "args": _args_sig(n)}
                for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not n.name.startswith("_test")
            ][:12]
            classes.append({
                "name": node.name,
                "doc": _first_line(ast.get_docstring(node)),
                "methods": methods,
            })
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append({
                "name": node.name,
                "args": _args_sig(node),
                "doc": _first_line(ast.get_docstring(node)),
            })

    if not classes and not functions:
        return None

    return {
        "file": rel,
        "module_doc": _first_line(ast.get_docstring(tree)),
        "classes": classes[:15],
        "functions": functions[:15],
    }


def scan_repo(root: str, max_files: int = 40, query: str = ""):
    """Index `root` across Python/TS/DNA/SQL files and return the top `max_files` ranked summaries."""
    try:
        from osiris_cli import codebase_visibility as cv

        idx = cv.index_repository(root)
        ranked = cv.rank_files(idx, query=query, limit=max_files)
        results = []
        for _score, node in ranked:
            classes = []
            functions = []
            for sym in node.symbols:
                if sym.kind == "class":
                    classes.append({
                        "name": sym.name,
                        "doc": sym.doc,
                        "methods": [{"name": m[0], "args": m[1]} for m in sym.methods[:12]],
                    })
                else:
                    functions.append({
                        "name": sym.name,
                        "args": sym.signature,
                        "doc": sym.doc,
                    })
            if not classes and not functions and not node.module_doc:
                continue
            results.append({
                "file": node.rel_path,
                "language": node.language,
                "imported_by": len(node.imported_by),
                "module_doc": node.module_doc,
                "classes": classes[:15],
                "functions": functions[:15],
            })
        return results
    except Exception:
        results = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            for fn in sorted(filenames):
                if not fn.endswith(".py"):
                    continue
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, root)
                summary = _summarize_file(full, rel)
                if summary:
                    results.append(summary)
                if len(results) >= max_files:
                    return results
        return results


def build_context(root: str = ".", max_files: int = 40, max_chars: int = 6000, query: str = "") -> str:
    """
    Returns a compact JSON string summarizing the real codebase at `root`,
    ranked against `query` and truncated to fit the target context window.
    """
    root = os.path.abspath(root)
    files = scan_repo(root, max_files=max_files, query=query)
    payload = {"root": root, "file_count": len(files), "files": files}
    text = json.dumps(payload, separators=(",", ":"))

    if len(text) > max_chars:
        # Drop files from the tail until it fits; keep file_count truthful.
        payload["truncated"] = True
        while files and len(text) > max_chars:
            files.pop()
            payload["files"] = files
            text = json.dumps(payload, separators=(",", ":"))

    return text


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    out = build_context(target)
    print(out)
    print(f"\n[bridge] {len(out)} chars, root={os.path.abspath(target)}", file=sys.stderr)
