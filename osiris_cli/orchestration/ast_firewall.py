"""One deny-by-default AST firewall for candidate code (SYS-025).

Before this module there were two firewalls: ``controller.ASTFirewallVisitor`` (used by the
governed engine and the trio) and ``critic_verifier.ASTSecurityVisitor`` (used by the L5 critic and
``code_critic``). They disagreed: neither caught ``os.execvp``, ``os.spawnv`` or ``os.execl``, the
critic missed ``os.popen``, and both missed ``from subprocess import run`` followed by ``run(...)``
or ``import subprocess as s`` followed by ``s.run(...)``. Both visitors now delegate here.

What is refused, wherever it appears in the tree:

* the builtins ``exec``, ``eval``, ``compile``, ``__import__``;
* every process-spawning or process-replacing function in ``os``/``posix``/``nt`` (``system``,
  ``popen``, all ``exec*``, all ``spawn*``, ``posix_spawn*``, ``fork*``, ``kill*``), in
  ``subprocess`` (every public call: ``run``, ``Popen``, ``call``, ``check_*``, ``getoutput``,
  ``getstatusoutput``), ``pty.spawn``, and ``importlib.import_module``;
* the same names reached through an alias (``import os as o``; ``from os import system as s``),
  through ``from <module> import *``, or by name in ``getattr(module, "<name>")``.

Names are resolved statically from the module's own imports. The check never imports or runs the
code it inspects. A file that does not parse is refused.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

BANNED_BUILTINS = frozenset({"exec", "eval", "compile", "__import__"})

_OS_BANNED = frozenset({
    "system", "popen",
    "execl", "execle", "execlp", "execlpe", "execv", "execve", "execvp", "execvpe",
    "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe",
    "posix_spawn", "posix_spawnp", "fork", "forkpty", "kill", "killpg",
})

#: module -> names in it that are refused. ``None`` means every attribute of the module.
BANNED_MODULE_NAMES: Dict[str, Optional[frozenset]] = {
    "os": _OS_BANNED,
    "posix": _OS_BANNED,
    "nt": _OS_BANNED,
    "subprocess": None,
    "pty": frozenset({"spawn", "fork"}),
    "importlib": frozenset({"import_module"}),
    "commands": None,
}


@dataclass(frozen=True)
class Violation:
    kind: str    # "builtin" | "call" | "import"
    name: str    # "exec", "os.system", "subprocess.run", "os.*"
    line: int


def _banned(module: str, attr: str) -> bool:
    names = BANNED_MODULE_NAMES.get(module, frozenset())
    if names is None:
        return not attr.startswith("_")
    return attr in names


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: List[Violation] = []
        self.module_alias: Dict[str, str] = {}             # local name -> module
        self.name_alias: Dict[str, Tuple[str, str]] = {}   # local name -> (module, attr)

    # -- imports ----------------------------------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if alias.name in BANNED_MODULE_NAMES:
                self.module_alias[alias.asname or root] = alias.name
            elif root in BANNED_MODULE_NAMES and alias.asname is None:
                self.module_alias[root] = root
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = (node.module or "").split(".")[0] if node.level == 0 else ""
        if module in BANNED_MODULE_NAMES:
            for alias in node.names:
                if alias.name == "*":
                    self.violations.append(Violation("import", f"{module}.*", node.lineno))
                elif _banned(module, alias.name):
                    self.name_alias[alias.asname or alias.name] = (module, alias.name)
                    self.violations.append(Violation("import", f"{module}.{alias.name}", node.lineno))
        self.generic_visit(node)

    # -- calls ------------------------------------------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in BANNED_BUILTINS and func.id not in self.name_alias:
                self.violations.append(Violation("builtin", func.id, node.lineno))
            elif func.id in self.name_alias:
                module, attr = self.name_alias[func.id]
                self.violations.append(Violation("call", f"{module}.{attr}", node.lineno))
            elif func.id == "getattr" and len(node.args) >= 2:
                self._check_getattr(node)
        elif isinstance(func, ast.Attribute):
            module = self._module_of(func.value)
            if module is not None and _banned(module, func.attr):
                self.violations.append(Violation("call", f"{module}.{func.attr}", node.lineno))
        self.generic_visit(node)

    def _module_of(self, value: ast.AST) -> Optional[str]:
        if isinstance(value, ast.Name):
            if value.id in self.module_alias:
                return self.module_alias[value.id]
            if value.id in BANNED_MODULE_NAMES:   # used without a visible import
                return value.id
        return None

    def _check_getattr(self, node: ast.Call) -> None:
        target, name = node.args[0], node.args[1]
        if isinstance(name, ast.Constant) and isinstance(name.value, str):
            module = self._module_of(target)
            if module is not None and _banned(module, name.value):
                self.violations.append(Violation("call", f"{module}.{name.value}", node.lineno))
            elif isinstance(target, ast.Name) and target.id in ("__builtins__", "builtins") \
                    and name.value in BANNED_BUILTINS:
                self.violations.append(Violation("builtin", name.value, node.lineno))


def scan(source: str) -> List[Violation]:
    """Every refused construct in ``source``. Raises SyntaxError if it does not parse."""
    visitor = _Visitor()
    visitor.visit(ast.parse(source))
    return visitor.violations


def scan_tree(tree: ast.AST) -> List[Violation]:
    visitor = _Visitor()
    visitor.visit(tree)
    return visitor.violations


def check(source: str) -> Tuple[bool, List[Violation]]:
    """(is_clean, violations). Code that does not parse is not clean."""
    try:
        found = scan(source)
    except SyntaxError as exc:
        return False, [Violation("syntax", str(exc), getattr(exc, "lineno", 0) or 0)]
    return not found, found


# Two renderings, one rule set: controller-style messages and critic-style tokens.

def controller_message(v: Violation) -> str:
    if v.kind == "builtin":
        return f"Forbidden call: '{v.name}' at line {v.line}"
    if v.kind == "import":
        return f"Forbidden import: '{v.name}' at line {v.line}"
    if v.kind == "syntax":
        return f"SyntaxError in candidate code: {v.name}"
    if v.name.split(".")[0] in ("os", "posix", "nt"):
        return f"Forbidden os call: '{v.name}' at line {v.line}"
    return f"Forbidden call: '{v.name}' at line {v.line}"


def critic_token(v: Violation) -> str:
    kind = {"builtin": "banned_primitive", "call": "banned_call", "import": "banned_import"}.get(v.kind, v.kind)
    return f"{kind}:{v.name}:L{v.line}"
