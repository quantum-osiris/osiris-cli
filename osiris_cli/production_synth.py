"""Closed-loop production code synthesis, AST contract extraction, and independent L5 critique.

Implements Iterations R1-R3 of the OSIRIS production-code pipeline ("models propose, code decides"):
  * Stage 1 (R1): Deterministic AST Contract Extractor (`extract_module_contract`,
    `verify_contract_preservation`) -- extracts and enforces public function/class/method
    signatures, `__init__` container attribute types (e.g. `self.stories: list`), and
    non-`None` return invariants before any test runs.
  * Stage 1B (R1): Delimited Multi-File Proposal Transport (`delimited_instructions`,
    `parse_delimited_or_json_proposal`) -- eliminates JSON string double-escaping corruption
    while remaining backward-compatible with `structured_proposal.parse_proposal`.
  * Stage 2 (R2): Structured Invariant Specifier (`build_invariant_spec`,
    `format_contract_prompt_block`) -- expands vague modify/synthesis stories + AST contracts
    into explicit pre/post-condition and boundary invariants.
  * Stage 4B (R2): Independent L5 Adversarial Critic (`synthesize_independent_critic_tests`,
    `combine_verification_tests`) -- strictly isolated from candidate implementation code;
    generates contract & boundary falsification assertions from `(spec, contract, existing_py)` only.
  * Stage 4C (R3): Hermetic Closed-Loop Synthesis (`ProductionConfinementGuard`,
    `diagnostic_feedback_block`, `discover_repo_pytest_targets`, `run_production_synthesis`) --
    drives `verify_loop.verified_first_passing` (`k=3` default) with SHA-256 duplicate-attempt
    circuit breaking, digest-bound receipts, and matching repository `pytest` execution.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import payload_gate
import structured_proposal
import verify_loop

_BOUNDED_REGEX_CAP_RE = re.compile(r"\.\{\d+,\s*(\d+)\}")
_DELIM_FILE_RE = re.compile(
    r"^===\s*FILE:\s*([^\n=]+?)\s*===\s*$\n(.*?)(?=^===\s*(?:FILE:|TEST(?::|\s*=)|END\b)|\Z)",
    re.MULTILINE | re.DOTALL,
)
_DELIM_TEST_RE = re.compile(
    r"^===\s*TEST(?::\s*[^\n=]+?)?\s*===\s*$\n(.*?)(?=^===\s*(?:FILE:|END\b)|\Z)",
    re.MULTILINE | re.DOTALL,
)
_CODE_FENCE_BLOCK_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_OUTER_FENCE_RE = re.compile(r"^\s*```(?:[a-zA-Z0-9_+-]+)?[ \t]*\n(.*?)\n?```\s*$", re.DOTALL)

_UNSAFE_AST_IMPORTS = frozenset({"subprocess", "socket", "ctypes"})
_UNSAFE_AST_CALLS = frozenset({"eval", "exec", "__import__"})


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 (R1): DETERMINISTIC AST CONTRACT EXTRACTOR & PRESERVATION VERIFIER
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class FunctionContract:
    name: str
    arg_names: Tuple[str, ...]
    min_positional: int
    max_positional: Optional[int]  # None when *args is accepted
    kwonly_names: Tuple[str, ...]
    required_kwonly: Tuple[str, ...]
    has_varkw: bool
    returns_non_none: bool


@dataclass(frozen=True)
class ClassContract:
    name: str
    init_min_args: int
    init_attr_types: Dict[str, str]
    methods: Dict[str, FunctionContract]


@dataclass(frozen=True)
class ModuleContract:
    public_names: Tuple[str, ...]
    functions: Dict[str, FunctionContract]
    classes: Dict[str, ClassContract]
    bounded_regex_caps: Tuple[str, ...] = ()


def _infer_expr_type_name(expr: ast.AST) -> Optional[str]:
    """Infer deterministic Python builtin type name for simple `__init__` initializers."""
    if isinstance(expr, (ast.List, ast.ListComp)):
        return "list"
    if isinstance(expr, (ast.Dict, ast.DictComp)):
        return "dict"
    if isinstance(expr, ast.SetComp) or isinstance(expr, ast.Set):
        return "set"
    if isinstance(expr, ast.Tuple):
        return "tuple"
    if isinstance(expr, ast.Constant):
        v = expr.value
        if isinstance(v, bool):
            return "bool"
        if isinstance(v, int):
            return "int"
        if isinstance(v, float):
            return "float"
        if isinstance(v, str):
            return "str"
        return None
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
        if expr.func.id in ("list", "dict", "set", "tuple", "int", "float", "str", "bool"):
            return expr.func.id
    return None


def _direct_returns(body: Sequence[ast.stmt]) -> List[ast.Return]:
    """Collect `return` statements belonging to `body`, skipping nested defs/classes/lambdas."""
    out: List[ast.Return] = []

    def visit_nodes(nodes: Sequence[ast.AST]) -> None:
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            if isinstance(node, ast.Return):
                out.append(node)
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                    continue
                visit_nodes([child])

    visit_nodes(body)
    return out


def _block_always_terminates(stmts: Sequence[ast.stmt]) -> bool:
    """True if control flow in `stmts` is guaranteed to return or raise without falling off the end."""
    for stmt in stmts:
        if isinstance(stmt, (ast.Return, ast.Raise)):
            return True
        if isinstance(stmt, ast.If):
            if stmt.orelse and _block_always_terminates(stmt.body) and _block_always_terminates(stmt.orelse):
                return True
        if isinstance(stmt, ast.Try):
            if stmt.finalbody and _block_always_terminates(stmt.finalbody):
                return True
            if (
                _block_always_terminates(stmt.body)
                and stmt.handlers
                and all(_block_always_terminates(h.body) for h in stmt.handlers)
            ):
                return True
    return False


def _extract_function_contract(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    is_method: bool = False,
) -> FunctionContract:
    pos_nodes = list(fn.args.posonlyargs) + list(fn.args.args)
    if is_method and pos_nodes and pos_nodes[0].arg in ("self", "cls"):
        pos_nodes = pos_nodes[1:]
    arg_names = tuple(a.arg for a in pos_nodes)
    n_defaults = len(fn.args.defaults)
    min_pos = max(0, len(pos_nodes) - n_defaults)
    max_pos = None if fn.args.vararg is not None else len(pos_nodes)
    kwonly_names = tuple(a.arg for a in fn.args.kwonlyargs)
    req_kw = tuple(
        a.arg for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults) if d is None
    )

    rets = _direct_returns(fn.body)
    returns_non_none = bool(rets) and _block_always_terminates(fn.body) and all(
        r.value is not None and not (isinstance(r.value, ast.Constant) and r.value.value is None)
        for r in rets
    )
    return FunctionContract(
        name=fn.name,
        arg_names=arg_names,
        min_positional=min_pos,
        max_positional=max_pos,
        kwonly_names=kwonly_names,
        required_kwonly=req_kw,
        has_varkw=fn.args.kwarg is not None,
        returns_non_none=returns_non_none,
    )


def _extract_init_attrs(init_fn: ast.FunctionDef | ast.AsyncFunctionDef) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for stmt in init_fn.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            tgt = stmt.targets[0]
            if (
                isinstance(tgt, ast.Attribute)
                and isinstance(tgt.value, ast.Name)
                and tgt.value.id == "self"
                and not tgt.attr.startswith("_")
            ):
                inferred = _infer_expr_type_name(stmt.value)
                if inferred:
                    attrs[tgt.attr] = inferred
        elif isinstance(stmt, ast.AnnAssign):
            tgt = stmt.target
            if (
                isinstance(tgt, ast.Attribute)
                and isinstance(tgt.value, ast.Name)
                and tgt.value.id == "self"
                and not tgt.attr.startswith("_")
                and stmt.value is not None
            ):
                inferred = _infer_expr_type_name(stmt.value)
                if inferred:
                    attrs[tgt.attr] = inferred
    return attrs


def extract_module_contract(source: str) -> ModuleContract:
    """Parse Python module `source` and extract its public API, attribute types, and regex caps."""
    tree = ast.parse(source)
    public_names: List[str] = []
    functions: Dict[str, FunctionContract] = {}
    classes: Dict[str, ClassContract] = {}

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            public_names.append(node.name)
            functions[node.name] = _extract_function_contract(node, is_method=False)
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            public_names.append(node.name)
            methods: Dict[str, FunctionContract] = {}
            init_attrs: Dict[str, str] = {}
            init_min_args = 0
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name == "__init__":
                        mc = _extract_function_contract(item, is_method=True)
                        methods["__init__"] = mc
                        init_min_args = mc.min_positional
                        init_attrs = _extract_init_attrs(item)
                    elif not item.name.startswith("_"):
                        methods[item.name] = _extract_function_contract(item, is_method=True)
            classes[node.name] = ClassContract(
                name=node.name,
                init_min_args=init_min_args,
                init_attr_types=init_attrs,
                methods=methods,
            )
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and not tgt.id.startswith("_"):
                    public_names.append(tgt.id)

    caps: Set[str] = set()
    for sub in ast.walk(tree):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            for m in _BOUNDED_REGEX_CAP_RE.finditer(sub.value):
                caps.add(m.group(0))

    return ModuleContract(
        public_names=tuple(dict.fromkeys(public_names)),
        functions=functions,
        classes=classes,
        bounded_regex_caps=tuple(sorted(caps)),
    )


def _check_callable_compat(
    label: str,
    old: FunctionContract,
    new: FunctionContract,
) -> List[str]:
    issues: List[str] = []
    if new.min_positional > old.min_positional:
        issues.append(
            f"{label} requires {new.min_positional} positional arg(s), breaking callers passing {old.min_positional}"
        )
    if old.max_positional is None and new.max_positional is not None:
        issues.append(f"{label} dropped *args support")
    elif old.max_positional is not None and new.max_positional is not None and new.max_positional < old.max_positional:
        issues.append(
            f"{label} accepts at most {new.max_positional} positional arg(s), previously accepted {old.max_positional}"
        )
    new_req_kw = set(new.required_kwonly) - set(old.required_kwonly)
    if new_req_kw:
        issues.append(f"{label} added required keyword-only arg(s) {sorted(new_req_kw)}")
    if not new.has_varkw:
        accepted_kw = set(new.arg_names) | set(new.kwonly_names)
        dropped_kw = (set(old.arg_names) | set(old.kwonly_names)) - accepted_kw
        if dropped_kw:
            issues.append(f"{label} renamed or dropped parameter(s) {sorted(dropped_kw)}")
    if old.returns_non_none and not new.returns_non_none:
        issues.append(
            f"{label} previously always returned a non-None value, but candidate can return None "
            f"(preserve non-None return value on all paths, including duplicates/idempotent calls)"
        )
    return issues


def verify_contract_preservation(contract: ModuleContract, candidate_source: str) -> List[str]:
    """Deterministically check that `candidate_source` preserves `contract` without silent drift."""
    try:
        cand = extract_module_contract(candidate_source)
    except SyntaxError as exc:
        return [f"syntax error at line {exc.lineno}: {exc.msg}"]

    issues: List[str] = []
    dropped_top = set(contract.public_names) - set(cand.public_names)
    if dropped_top:
        issues.append(f"drops existing public symbol(s): {sorted(dropped_top)}")

    for fn_name, old_fn in contract.functions.items():
        new_fn = cand.functions.get(fn_name)
        if new_fn is None:
            continue
        issues.extend(_check_callable_compat(f"function '{fn_name}'", old_fn, new_fn))

    for cls_name, old_cls in contract.classes.items():
        new_cls = cand.classes.get(cls_name)
        if new_cls is None:
            continue
        for attr, old_type in old_cls.init_attr_types.items():
            new_type = new_cls.init_attr_types.get(attr)
            if new_type is not None and new_type != old_type:
                issues.append(
                    f"class '{cls_name}.__init__' changed attribute 'self.{attr}' from {old_type} to {new_type}"
                )
        for m_name, old_m in old_cls.methods.items():
            new_m = new_cls.methods.get(m_name)
            if new_m is None:
                if m_name != "__init__":
                    issues.append(f"class '{cls_name}' drops public method '{m_name}'")
                continue
            issues.extend(_check_callable_compat(f"method '{cls_name}.{m_name}'", old_m, new_m))

    return issues


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1B (R1): DELIMITED MULTI-FILE PROPOSAL PARSER (NO JSON DOUBLE-ESCAPING)
# ═══════════════════════════════════════════════════════════════════════════════


def strip_code_fences(source: str) -> str:
    """Remove accidental outer markdown fences inside extracted Python source."""
    if not source:
        return ""
    stripped = source.strip()
    m = _OUTER_FENCE_RE.match(stripped)
    if m:
        return m.group(1).strip() + "\n"
    lines = stripped.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip() + ("\n" if lines else "")


def delimited_instructions(path: str, module_name: str) -> str:
    """Unescaped multi-block output contract eliminating JSON string double-escaping failures."""
    return (
        f"Output ONLY the following plain-text delimited blocks (NO JSON escaping, NO prose outside blocks):\n"
        f"=== FILE: {path} ===\n"
        f"# complete Python source of {path}\n"
        f"=== TEST: test_proposal.py ===\n"
        f"import {module_name}\n\n"
        f"def test_proposal():\n"
        f"    # plain assert statements testing normal, duplicate/idempotent, and boundary cases\n"
        f"    assert ...\n"
        f"=== END ==="
    )


def parse_delimited_or_json_proposal(raw: str, default_path: str = "") -> Optional[Dict[str, Any]]:
    """Parse delimited `=== FILE: ... ===` blocks, JSON proposals, legacy XML tags, or dual Python fences."""
    if not raw or not raw.strip():
        return None

    # 1. Delimited multi-block format (zero JSON-escaping corruption)
    file_matches = list(_DELIM_FILE_RE.finditer(raw))
    test_match = _DELIM_TEST_RE.search(raw)
    if file_matches and test_match:
        primary_path = file_matches[0].group(1).strip() or default_path
        primary_mod = strip_code_fences(file_matches[0].group(2))
        test_src = strip_code_fences(test_match.group(1))
        extra_files = [
            (m.group(1).strip(), strip_code_fences(m.group(2)))
            for m in file_matches[1:]
            if m.group(1).strip() and strip_code_fences(m.group(2)).strip()
        ]
        if primary_path and primary_mod.strip() and test_src.strip():
            return {
                "path": primary_path,
                "module": primary_mod,
                "test": test_src,
                "extra_files": extra_files,
                "format": "delimited",
            }

    # 2. Standard JSON / <WRITE_FILE> tag parser (cleaning inner markdown fences if present)
    parsed = structured_proposal.parse_proposal(raw)
    if parsed is not None:
        parsed["module"] = strip_code_fences(parsed["module"])
        parsed["test"] = strip_code_fences(parsed["test"])
        parsed.setdefault("extra_files", [])
        return parsed

    # 3. Embedded JSON object inside surrounding commentary
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if 0 <= first_brace < last_brace:
        sub = raw[first_brace:last_brace + 1]
        try:
            obj = json.loads(sub)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and all(
            isinstance(obj.get(k), str) and obj[k].strip() for k in structured_proposal.FIELDS
        ):
            return {
                "path": obj["path"].strip(),
                "module": strip_code_fences(obj["module"]),
                "test": strip_code_fences(obj["test"]),
                "extra_files": [],
                "format": "json_embedded",
            }

    # 4. Dual ```python``` fenced blocks where second block defines `def test_proposal`
    fenced_blocks = [b.strip() for b in _CODE_FENCE_BLOCK_RE.findall(raw) if b.strip()]
    if len(fenced_blocks) >= 2 and default_path:
        test_idx = next((i for i, b in enumerate(fenced_blocks) if "def test_proposal" in b), None)
        if test_idx is not None:
            mod_blocks = [b for i, b in enumerate(fenced_blocks) if i != test_idx]
            if mod_blocks:
                return {
                    "path": default_path,
                    "module": mod_blocks[0] + "\n",
                    "test": fenced_blocks[test_idx] + "\n",
                    "extra_files": [],
                    "format": "fenced",
                }

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 (R2): STRUCTURED INVARIANT SPECIFIER
# ═══════════════════════════════════════════════════════════════════════════════


def build_invariant_spec(story_text: str, contract: Optional[ModuleContract] = None) -> List[str]:
    """Deterministic L4 invariant decomposition from story prose + extracted AST contract."""
    invariants: List[str] = []
    low = (story_text or "").lower()

    if contract is not None:
        if contract.public_names:
            invariants.append(
                f"Preserve all existing public symbols and compatible call signatures: {', '.join(contract.public_names)}."
            )
        for cls_name, cls_c in contract.classes.items():
            if cls_c.init_attr_types:
                pairs = ", ".join(f"self.{k}: {v}" for k, v in sorted(cls_c.init_attr_types.items()))
                invariants.append(
                    f"Preserve `{cls_name}.__init__` attribute container/scalar types ({pairs}); "
                    f"do not replace a `list` with a `dict` (callers index `[-1]` and iterate in insertion order)."
                )
            non_none_methods = [
                m_name for m_name, mc in cls_c.methods.items()
                if m_name != "__init__" and mc.returns_non_none
            ]
            if non_none_methods:
                invariants.append(
                    f"Methods on `{cls_name}` that currently always return a non-None value "
                    f"({', '.join(non_none_methods)}) MUST still return a valid non-None value on "
                    f"duplicate/idempotent calls (e.g. return the existing item's ID, NEVER None)."
                )
        non_none_fns = [fn_name for fn_name, fc in contract.functions.items() if fc.returns_non_none]
        if non_none_fns:
            invariants.append(
                f"Top-level functions ({', '.join(non_none_fns)}) must keep returning non-None values across edge cases."
            )
        if contract.bounded_regex_caps:
            invariants.append(
                f"Existing source contains fixed-length regex quantifier(s) {list(contract.bounded_regex_caps)}; "
                f"remove arbitrary character-length ceilings so inputs > 250 characters match up to newline or end-of-string."
            )

    if re.search(r"\b(duplicate|twice|already|idempoten|repeat|existing|dedup)", low):
        invariants.append(
            "Idempotency: repeating the same operation with identical or whitespace/case-normalized keys must not "
            "create duplicate entries, advance counters, or return None when a handle/ID is expected."
        )
    if re.search(r"\b(truncat|long|line|regex|pattern|extract|parse|match|claim|gap|number)", low):
        invariants.append(
            "Unbounded & embedded text: handle >250-character inputs without fixed-length truncation, "
            r"extract numbers via regex (`r'\d+(?:\.\d+)?'`) rather than calling `int()`/`float()` on prose words "
            "like `'parity'`, and never split sentences on decimal points inside numbers (e.g. `'0.97'`)."
        )
    if re.search(r"\b(breaker|auth|401|403|status|error|retry| sticky|transient|timeout)", low):
        invariants.append(
            "Failure classification: distinguish permanent/auth failures (sticky until explicit reset) from "
            "transient timeouts/5xx errors, and never silently swallow error codes."
        )
    invariants.append(
        "Write assertions covering normal behavior, repeated/duplicate calls, empty/invalid inputs, and >250-char boundary inputs."
    )
    return invariants


def format_contract_prompt_block(story_text: str, contract: Optional[ModuleContract] = None) -> str:
    """Format the Stage 1 + Stage 2 contract and invariant specification for synthesis prompts."""
    invariants = build_invariant_spec(story_text, contract)
    bullets = "\n".join(f"  {i}. {inv}" for i, inv in enumerate(invariants, start=1))
    return f"MANDATORY CONTRACT & INVARIANTS (enforced deterministically by OSIRIS L5 gate):\n{bullets}\n"


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4B (R2): INDEPENDENT L5 CRITIC (NEVER SEES CANDIDATE IMPLEMENTATION)
# ═══════════════════════════════════════════════════════════════════════════════


def _static_safe_test(test_code: str) -> Optional[str]:
    try:
        tree = ast.parse(test_code)
    except SyntaxError as exc:
        return f"critic test syntax error: {exc}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _UNSAFE_AST_IMPORTS:
                    return f"disallowed import in critic test: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in _UNSAFE_AST_IMPORTS:
                return f"disallowed import in critic test: {node.module}"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _UNSAFE_AST_CALLS:
                return f"disallowed call in critic test: {node.func.id}()"
    return None


def synthesize_independent_critic_tests(
    module_name: str,
    story_text: str,
    contract: Optional[ModuleContract] = None,
    existing_source: Optional[str] = None,
    critic_model_fn: Optional[Callable[[str], str]] = None,
) -> str:
    """Generate independent L5 falsification assertions from `(story_text, contract, existing_source)` ONLY.

    By construction, this function has NO parameter for the candidate module or proposer test,
    making self-confirming "false confidence" collusion structurally impossible.
    """
    lines: List[str] = [
        "import inspect",
        f"import {module_name} as _osiris_target",
        "",
        "def _osiris_critic_contract():",
        "    assert _osiris_target is not None",
    ]

    if contract is not None:
        for pub in contract.public_names:
            lines.append(f"    assert hasattr(_osiris_target, {pub!r}), 'missing public symbol {pub}'")
        for fn_name, fc in contract.functions.items():
            lines.append(f"    assert callable(getattr(_osiris_target, {fn_name!r}, None))")
            lines.append(f"    _sig = inspect.signature(_osiris_target.{fn_name})")
            lines.append(
                "    _req = [p for p in _sig.parameters.values() "
                "if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty]"
            )
            lines.append(
                f"    assert len(_req) <= {fc.min_positional}, "
                f"'{fn_name} requires more positional args than contract'"
            )
            if (
                fc.min_positional == 2
                and re.search(r"\b(numbers?|unsourced|made-up|invented|claim|cite|evidence)\b", (story_text or "").lower())
            ):
                lines.append(
                    f"    _bad_num = _osiris_target.{fn_name}('parity telemetry reached 0.9728', 'measured fidelity 0.9722 (FAIL)')"
                )
                lines.append(
                    f"    assert _osiris_target.{fn_name}('the ledger is intact', 'all chains verified') == [], "
                    f"'{fn_name} must accept prose without unsourced numbers'"
                )
                lines.append(
                    f"    assert _bad_num and any('0.9728' in str(_p) for _p in _bad_num), "
                    f"'{fn_name} must report unsourced number 0.9728 by value'"
                )
                lines.append(
                    f"    assert _osiris_target.{fn_name}('fidelity 0.9722 on node', 'measured fidelity 0.9722.') == [], "
                    f"'{fn_name} must accept sourced number followed by sentence punctuation'"
                )
                lines.append(
                    f"    _sub_num = _osiris_target.{fn_name}('ran 3 seeds', 'seeds 13')"
                )
                lines.append(
                    f"    assert _sub_num and any('3' in str(_p) for _p in _sub_num), "
                    f"'{fn_name} must treat numbers as whole tokens (3 != 13)'"
                )
            if (
                fc.min_positional == 1
                and fn_name.startswith("extract_")
                and (contract.bounded_regex_caps or re.search(r"\b(gap|truncat|long)\b", (story_text or "").lower()))
            ):
                lines.append(
                    "    _long_gap = 'a hash-chained ledger that records ' + ('every genome and prompt change ' * 8) + 'cleanly'"
                )
                lines.append(
                    f"    _g_long = _osiris_target.{fn_name}([f'We have a capability gap: {{_long_gap}}. Unrelated tail.'])"
                )
                lines.append(
                    f"    assert _g_long == [_long_gap], '{fn_name} must extract >250-char description whole without truncation'"
                )
                lines.append(
                    f"    _g_dec = _osiris_target.{fn_name}(['Capability Gap: export 0.97 fidelity values. More.'])"
                )
                lines.append(
                    f"    assert _g_dec == ['export 0.97 fidelity values'], '{fn_name} must not split on decimal point 0.97'"
                )
        for cls_name, cc in contract.classes.items():
            lines.append(f"    _cls = getattr(_osiris_target, {cls_name!r})")
            for m_name in cc.methods:
                if m_name != "__init__":
                    lines.append(
                        f"    assert callable(getattr(_cls, {m_name!r}, None)), "
                        f"'missing method {cls_name}.{m_name}'"
                    )
            if cc.init_min_args == 0:
                lines.append("    _inst = _cls()")
                for attr, tname in sorted(cc.init_attr_types.items()):
                    lines.append(
                        f"    assert hasattr(_inst, {attr!r}) and isinstance(getattr(_inst, {attr!r}), {tname}), "
                        f"'{cls_name}.{attr} must remain {tname}'"
                    )
                for m_name, mc in sorted(cc.methods.items()):
                    if (
                        m_name in ("add", "register", "append", "insert", "record")
                        and mc.min_positional == 1
                        and mc.returns_non_none
                    ):
                        lines.append(f"    _r1 = _inst.{m_name}('OSIRIS Critic Probe Alpha')")
                        lines.append(f"    _r2 = _inst.{m_name}('OSIRIS Critic Probe Alpha')")
                        lines.append(
                            f"    assert _r1 is not None and _r2 is not None, "
                            f"'{cls_name}.{m_name} returned None on initial or duplicate call'"
                        )
                        if re.search(r"\b(duplicate|twice|idempoten|existing)\b", (story_text or "").lower()):
                            lines.append(
                                f"    assert _r1 == _r2, '{cls_name}.{m_name} must return the existing id on duplicate open entry'"
                            )
                        for attr, tname in sorted(cc.init_attr_types.items()):
                            lines.append(
                                f"    assert isinstance(getattr(_inst, {attr!r}), {tname}) and type(getattr(_inst, {attr!r})) is {tname}, "
                                f"'{cls_name}.{attr} must remain an exact {tname} (never a dict subclass)'"
                            )
                            if tname == "list":
                                lines.append(
                                    f"    if len(getattr(_inst, {attr!r})) > 0: _ = getattr(_inst, {attr!r})[-1]"
                                )

    critic_extra_fn_present = False
    if critic_model_fn is not None:
        invariants_block = format_contract_prompt_block(story_text, contract)
        existing_snippet = (
            f"\nEXISTING MODULE SOURCE (before change):\n```python\n{existing_source[:4500]}\n```\n"
            if existing_source
            else ""
        )
        critic_prompt = (
            "You are OSIRIS L5 Independent Critic. You DO NOT see the candidate implementation.\n"
            "Write a Python function `def _osiris_critic_invariants():` using ONLY `import "
            f"{module_name}` and plain `assert` statements that try to falsify buggy implementations "
            "of the specification below (test duplicates/idempotency, >250-char strings without truncation, "
            "boundary inputs, and preserved return types).\n"
            "Do NOT import subprocess, socket, or ctypes. Output ONLY Python code.\n\n"
            f"SPECIFICATION:\n{story_text}\n\n{invariants_block}{existing_snippet}"
        )
        raw_critic = strip_code_fences(critic_model_fn(critic_prompt) or "")
        if "def _osiris_critic_invariants" in raw_critic and _static_safe_test(raw_critic) is None:
            try:
                tree = ast.parse(raw_critic)
                has_assert = any(
                    isinstance(n, ast.FunctionDef)
                    and n.name == "_osiris_critic_invariants"
                    and any(isinstance(sub, ast.Assert) for sub in ast.walk(n))
                    for n in tree.body
                )
                if has_assert:
                    lines.append("")
                    lines.append(raw_critic.rstrip())
                    critic_extra_fn_present = True
            except SyntaxError:
                pass

    lines.append("")
    lines.append("def run_all_critic_checks():")
    lines.append("    _osiris_critic_contract()")
    if critic_extra_fn_present:
        lines.append("    _osiris_critic_invariants()")
    return "\n".join(lines) + "\n"


def combine_verification_tests(proposer_test: str, critic_test: str) -> Tuple[Optional[str], Optional[str]]:
    """Validate `proposer_test` on its own first, then bind `critic_test` into `test_proposal()`.

    Returns `(combined_source, error_reason)`.
    """
    prop_err = payload_gate.check_test(proposer_test)
    if prop_err:
        return None, f"placeholder gate rejected test: {prop_err}"
    safe_err = _static_safe_test(proposer_test)
    if safe_err:
        return None, f"static safety check rejected proposal: {safe_err}"
    if not critic_test or not critic_test.strip():
        return proposer_test, None

    renamed_proposer = re.sub(
        r"^(\s*def\s+)test_proposal(\s*\()",
        r"\1_osiris_proposer_test_proposal\2",
        proposer_test,
        count=1,
        flags=re.MULTILINE,
    )
    combined = (
        f"{renamed_proposer.rstrip()}\n\n"
        f"# --- OSIRIS L5 Independent Critic Suite ---\n"
        f"{critic_test.rstrip()}\n\n"
        f"def test_proposal():\n"
        f"    assert callable(_osiris_proposer_test_proposal)\n"
        f"    _osiris_proposer_test_proposal()\n"
        f"    run_all_critic_checks()\n"
    )
    return combined, None


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4C (R3): CONFINEMENT GUARD, FEEDBACK COMPRESSOR & CLOSED-LOOP SYNTHESIS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AuditDecision:
    is_admissible: bool
    message: str
    candidate_id: str = ""


class ProductionConfinementGuard:
    """Self-contained AST, contract, and path confinement guard for `verify_loop.verified_first_passing`."""

    def __init__(
        self,
        root: str,
        target_rel_path: str,
        contract: Optional[ModuleContract] = None,
    ):
        self.root = os.path.realpath(root)
        self.target_rel_path = target_rel_path
        self.contract = contract

    def _confine_rel_path(self, rel_path: str) -> Optional[str]:
        if not rel_path or os.path.isabs(rel_path):
            return f"path {rel_path!r} must be relative to workspace root"
        resolved = os.path.realpath(os.path.join(self.root, rel_path))
        if not (resolved == self.root or resolved.startswith(self.root.rstrip(os.sep) + os.sep)):
            return f"path {rel_path!r} escapes workspace root {self.root}"
        norm_rel = os.path.relpath(resolved, self.root)
        if norm_rel == "tests" or norm_rel.startswith("tests" + os.sep):
            return f"candidate module may not overwrite reference test path {norm_rel!r}"
        return None

    def audit_patch_paths(self, patch_payload: str, candidate_id: str = "") -> AuditDecision:
        for line in patch_payload.splitlines():
            if line.startswith(("+++ ", "--- ")):
                raw = line[4:].strip().split("\t")[0]
                if raw == "/dev/null":
                    continue
                for pfx in ("a/", "b/"):
                    if raw.startswith(pfx):
                        raw = raw[len(pfx):]
                err = self._confine_rel_path(raw)
                if err:
                    return AuditDecision(False, err, candidate_id)
        return AuditDecision(True, "patch paths confined", candidate_id)

    def audit_ast_safety(self, source_payload: str, candidate_id: str = "") -> AuditDecision:
        path_err = self._confine_rel_path(self.target_rel_path)
        if path_err:
            return AuditDecision(False, path_err, candidate_id)
        try:
            tree = ast.parse(source_payload)
        except SyntaxError as exc:
            snippet = (exc.text or "").strip()
            return AuditDecision(
                False,
                f"SyntaxError at line {exc.lineno}: {exc.msg}" + (f" (`{snippet}`)" if snippet else ""),
                candidate_id,
            )
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec"):
                return AuditDecision(False, f"disallowed builtin call `{node.func.id}()`", candidate_id)

        stub_reason = payload_gate.check_payload(source_payload)
        if stub_reason:
            return AuditDecision(False, f"placeholder gate: {stub_reason}", candidate_id)

        if self.contract is not None:
            violations = verify_contract_preservation(self.contract, source_payload)
            if violations:
                return AuditDecision(
                    False,
                    "contract drift: " + "; ".join(violations),
                    candidate_id,
                )
        return AuditDecision(True, "AST, payload, and contract checks passed", candidate_id)


def diagnostic_feedback_block(reason: str, format_hint: str = "delimited") -> str:
    """Compress raw verification failure output into an actionable root-cause repair block."""
    raw = (reason or "").strip()
    lines = [ln.rstrip() for ln in raw.splitlines() if ln.strip()]
    focus = [
        ln for ln in lines
        if any(
            tok in ln
            for tok in (
                "AssertionError",
                "SyntaxError",
                "TypeError",
                "ValueError",
                "AttributeError",
                "KeyError",
                "contract drift",
                "placeholder gate",
                "FORBIDDEN_AST_PRIMITIVE",
                "DUPLICATE_CANDIDATE",
                "assert ",
                "FAILED",
            )
        )
    ]
    excerpt = "\n".join((focus[-8:] if focus else lines[-10:]))[-1200:]
    fmt_note = (
        "Return a corrected complete module and test using the exact `=== FILE: ... ===` / `=== TEST: ... ===` blocks."
        if format_hint == "delimited"
        else "Return a corrected candidate in the exact requested format."
    )
    return (
        "\n\nPREVIOUS CANDIDATE FAILED DETERMINISTIC L5 VERIFICATION:\n"
        f"{excerpt}\n"
        "Fix the root cause above (do NOT weaken tests or return the identical code).\n"
        f"{fmt_note}\n"
    )


def discover_repo_pytest_targets(root: str, target_rel_path: str) -> List[str]:
    """Find matching `tests/test_<module>.py` or `test_<module>.py` inside `root`."""
    mod = os.path.splitext(os.path.basename(target_rel_path))[0]
    if not mod or mod.startswith("test_"):
        return []
    candidates = [
        os.path.join("tests", f"test_{mod}.py"),
        f"test_{mod}.py",
    ]
    found: List[str] = []
    real_root = os.path.realpath(root)
    for rel in candidates:
        full = os.path.realpath(os.path.join(real_root, rel))
        if full.startswith(real_root.rstrip(os.sep) + os.sep) and os.path.isfile(full):
            found.append(rel)
    return found


def run_repo_pytest_in_overlay(
    root: str,
    target_rel_path: str,
    candidate_source: str,
    pytest_rel_paths: Sequence[str],
    timeout: int = 25,
) -> Tuple[bool, str]:
    """Stage `candidate_source` in a temporary overlay without touching `root` and run matching repo pytest files."""
    if not pytest_rel_paths:
        return True, ""
    real_root = os.path.realpath(root)
    tmpdir = tempfile.mkdtemp(prefix="osiris_repo_pytest_")
    try:
        overlay_pkg_root = os.path.join(tmpdir, "overlay")
        staged_target = os.path.join(overlay_pkg_root, target_rel_path)
        os.makedirs(os.path.dirname(staged_target), exist_ok=True)
        # If target lives inside a package (e.g. osiris_cli/foo.py), symlink sibling modules so package imports resolve
        rel_dir = os.path.dirname(target_rel_path)
        if rel_dir:
            src_pkg_dir = os.path.join(real_root, rel_dir)
            dst_pkg_dir = os.path.join(overlay_pkg_root, rel_dir)
            if os.path.isdir(src_pkg_dir):
                for entry in os.listdir(src_pkg_dir):
                    src_entry = os.path.join(src_pkg_dir, entry)
                    dst_entry = os.path.join(dst_pkg_dir, entry)
                    if os.path.abspath(dst_entry) == os.path.abspath(staged_target):
                        continue
                    if not os.path.exists(dst_entry):
                        os.symlink(src_entry, dst_entry)
        with open(staged_target, "w", encoding="utf-8") as f:
            f.write(candidate_source)

        env = {
            k: v for k, v in os.environ.items()
            if k in ("PATH", "LANG", "LC_ALL", "TERM", "SYSTEMROOT")
        }
        existing_py = os.environ.get("PYTHONPATH", "")
        py_parts = [overlay_pkg_root, real_root] + ([existing_py] if existing_py else [])
        env.update(
            HOME=os.path.join(tmpdir, "_home"),
            TMPDIR=tmpdir,
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONNOUSERSITE="1",
            PYTHONPATH=os.pathsep.join(py_parts),
        )
        os.makedirs(env["HOME"], exist_ok=True)
        cache_dir = os.path.join(tmpdir, "pytest_cache")
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            f"cache_dir={cache_dir}",
            "-q",
            *[os.path.join(real_root, p) for p in pytest_rel_paths],
        ]
        proc = subprocess.run(
            cmd,
            cwd=tmpdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
        output = (proc.stdout or "") + (f"\n{proc.stderr}" if proc.stderr else "")
        return proc.returncode == 0, output.strip()
    except subprocess.TimeoutExpired:
        return False, f"repository pytest timed out after {timeout}s"
    except OSError as exc:
        return False, f"repository pytest failed to start: {exc}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@dataclass
class SynthesisOutcome:
    winner: Optional[Dict[str, Any]]
    attempts: List[Dict[str, Any]]
    contract: Optional[ModuleContract]
    invariants: List[str]
    critic_source: str
    receipt: Optional[Dict[str, Any]] = None


def run_production_synthesis(
    *,
    root: str,
    target_rel_path: str,
    story_text: str,
    propose_raw_fn: Callable[[str], str],
    sandbox_test_fn: Callable[[str, str, str], Tuple[bool, str]],
    existing_source: Optional[str] = None,
    critic_model_fn: Optional[Callable[[str], str]] = None,
    k: int = 3,
    use_delimited_prompt: bool = True,
    enforce_target_path: bool = True,
    run_repo_pytest: bool = True,
    on_attempt: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> SynthesisOutcome:
    """Execute the complete R1-R3 contract -> propose -> static/contract gate -> L5 critic -> sandbox/pytest loop."""
    module_name = os.path.splitext(os.path.basename(target_rel_path))[0]
    contract: Optional[ModuleContract] = None
    if existing_source:
        try:
            contract = extract_module_contract(existing_source)
        except SyntaxError:
            contract = None

    invariants = build_invariant_spec(story_text, contract)
    contract_block = format_contract_prompt_block(story_text, contract)
    current_block = (
        f"\nCURRENT CONTENT OF {target_rel_path} (modify THIS file; output the COMPLETE updated file):\n"
        f"{existing_source}\n"
        if existing_source is not None
        else ""
    )
    format_block = (
        delimited_instructions(target_rel_path, module_name)
        if use_delimited_prompt
        else structured_proposal.instructions(target_rel_path, module_name)
    )
    base_prompt = (
        f"You are OSIRIS Sovereign Synthesizer implementing verified production Python code.\n\n"
        f"SPECIFICATION:\n{story_text}\n\n"
        f"{contract_block}"
        f"{current_block}\n"
        f"{format_block}\n"
    )

    critic_source = synthesize_independent_critic_tests(
        module_name=module_name,
        story_text=story_text,
        contract=contract,
        existing_source=existing_source,
        critic_model_fn=critic_model_fn,
    )
    guard = ProductionConfinementGuard(root=root, target_rel_path=target_rel_path, contract=contract)
    repo_pytest_targets = discover_repo_pytest_targets(root, target_rel_path) if run_repo_pytest else []

    seen_module_hashes: Set[str] = set()
    parsed_by_id: Dict[str, Dict[str, Any]] = {}
    winning_receipt: Dict[str, Any] = {}
    attempt_counter = [0]

    def generate(feedback: Optional[str]) -> Tuple[Optional[verify_loop.CandidateArtifact], str]:
        attempt_counter[0] += 1
        idx = attempt_counter[0]
        full_prompt = base_prompt + (
            diagnostic_feedback_block(feedback, "delimited" if use_delimited_prompt else "json")
            if feedback
            else ""
        )
        raw = propose_raw_fn(full_prompt)
        parsed = parse_delimited_or_json_proposal(raw, default_path=target_rel_path)
        if parsed is None:
            return None, "no usable proposal (empty or malformed output)"
        cid = f"cand-{idx}"
        parsed_by_id[cid] = parsed
        return (
            verify_loop.CandidateArtifact(
                candidate_id=cid,
                artifact_type="module",
                payload=parsed["module"],
                metadata={
                    "path": parsed["path"],
                    "format": parsed.get("format", "unknown"),
                },
            ),
            "",
        )

    def hermetic_verify(cand: verify_loop.CandidateArtifact) -> Tuple[bool, str, Dict[str, Any]]:
        parsed = parsed_by_id.get(cand.candidate_id, {})
        prop_path = os.path.normpath(parsed.get("path", target_rel_path))
        expected_path = os.path.normpath(target_rel_path)
        if enforce_target_path and prop_path != expected_path:
            return False, f"proposed '{prop_path}' instead of assigned '{expected_path}'", {}
        if not enforce_target_path:
            path_err = guard._confine_rel_path(prop_path)
            if path_err:
                return False, path_err, {}
            if not prop_path.endswith(".py"):
                return False, f"proposed path '{prop_path}' is not a .py module", {}
        effective_rel_path = expected_path if enforce_target_path else prop_path
        effective_mod_name = os.path.splitext(os.path.basename(effective_rel_path))[0]

        mod_sha = hashlib.sha256(cand.payload.encode("utf-8")).hexdigest()
        if mod_sha in seen_module_hashes:
            return (
                False,
                "DUPLICATE_CANDIDATE: byte-identical module already failed verification; modify logic to fix failure",
                {},
            )
        seen_module_hashes.add(mod_sha)

        effective_critic = (
            critic_source
            if effective_mod_name == module_name
            else synthesize_independent_critic_tests(
                module_name=effective_mod_name,
                story_text=story_text,
                contract=contract,
                existing_source=existing_source,
            )
        )
        combined_test, combine_err = combine_verification_tests(parsed.get("test", ""), effective_critic)
        if combine_err or combined_test is None:
            return False, combine_err or "invalid test block", {}

        ok, out = sandbox_test_fn(os.path.basename(effective_rel_path), cand.payload, combined_test)
        if not ok:
            return False, out, {}

        eff_pytest_targets = (
            repo_pytest_targets
            if effective_rel_path == expected_path
            else (discover_repo_pytest_targets(root, effective_rel_path) if run_repo_pytest else [])
        )
        if eff_pytest_targets:
            py_ok, py_out = run_repo_pytest_in_overlay(
                root=root,
                target_rel_path=effective_rel_path,
                candidate_source=cand.payload,
                pytest_rel_paths=eff_pytest_targets,
            )
            if not py_ok:
                return False, f"existing repository pytest failed ({', '.join(eff_pytest_targets)}):\n{py_out}", {}

        receipt = {
            "evaluator_id": "osiris.production_synth.v1",
            "policy_version": "OSIRIS-SPEC-VERIF-015",
            "candidate_digest": cand.compute_digest(),
            "module_sha256": mod_sha,
            "critic_executed": True,
            "repo_pytest_targets": list(eff_pytest_targets),
        }
        winning_receipt.clear()
        winning_receipt.update(receipt)
        return True, out, receipt

    winner_art, raw_attempts = verify_loop.verified_first_passing(
        generate=generate,
        hermetic_verify=hermetic_verify,
        k=max(1, int(k)),
        confinement_guard=guard,
        evaluator_id="osiris.production_synth.v1",
        on_attempt=on_attempt,
    )

    normalized_attempts: List[Dict[str, Any]] = []
    for att in raw_attempts:
        art = att.get("candidate")
        prop_dict = parsed_by_id.get(art.candidate_id) if isinstance(art, verify_loop.CandidateArtifact) else None
        normalized_attempts.append({
            "n": att["n"],
            "ok": att["ok"],
            "reason": att["reason"],
            "candidate": prop_dict,
            "artifact": art,
        })

    winner_prop = (
        parsed_by_id.get(winner_art.candidate_id)
        if isinstance(winner_art, verify_loop.CandidateArtifact)
        else None
    )
    return SynthesisOutcome(
        winner=winner_prop,
        attempts=normalized_attempts,
        contract=contract,
        invariants=invariants,
        critic_source=critic_source,
        receipt=dict(winning_receipt) if winner_prop else None,
    )
