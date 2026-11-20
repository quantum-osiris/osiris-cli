"""Unified Agent Skills runtime for OSIRIS v6.0.0.

Discovers, validates, autocompletes, and hydrates SKILL.md bundles across
project (./.agents/skills) and global (~/.agents/skills) stores with strict
cross-CLI schema parity (Claude Code, Google Antigravity, Mistral Vibe,
GitHub Copilot CLI, and OSIRIS).

Invariants enforced:
- Project-over-global precedence (./.agents/skills overrides ~/.agents/skills)
- Fail-closed project shadowing (invalid project skill blocks global fallback)
- Universal 2-key YAML frontmatter (name, description only) + UTF-8 LF checks
- SafePathGuard confinement (blocks '..' traversal, escaping symlinks, and secrets)
- Coordinator context hygiene (companion references/*.md stay unexpanded unless requested)
- Deterministic SHA-256 receipts ready for ~/.osiris/living/exchanges.jsonl
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SkillScope = Literal["project", "global"]

SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.+)$", re.DOTALL)
COMPANION_REF_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])((?:references|scripts|assets)/[A-Za-z0-9_./-]+\.[A-Za-z0-9]+)"
)
FORBIDDEN_SECRET_RE = re.compile(
    r"(^\.env($|\.)|^apikey\.json$|credentials|secret|token|"
    r"^id_rsa|^id_ed25519|\.pem$|\.key$|\.netrc$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SkillValidationIssue:
    path: str
    code: str
    message: str


@dataclass(frozen=True)
class SkillRecord:
    name: str
    description: str
    scope: SkillScope
    skill_dir: Path
    skill_file: Path
    sha256: str
    body: str
    referenced_files: tuple[str, ...]
    valid: bool
    issues: tuple[SkillValidationIssue, ...]


@dataclass(frozen=True)
class SkillHydrationBundle:
    name: str
    scope: SkillScope
    skill_dir: Path
    skill_file: Path
    system_prompt: str
    user_prompt: str
    skill_sha256: str
    loaded_resources: dict[str, str]
    ledger_receipt: dict[str, object]


def resolve_osiris_home(explicit_home: Path | str | None = None) -> Path:
    """Resolve active OSIRIS state root dynamically via $OSIRIS_HOME or ~/.osiris."""
    if explicit_home is not None:
        return Path(explicit_home).expanduser().resolve()
    env_home = os.environ.get("OSIRIS_HOME", "").strip()
    if env_home:
        return Path(env_home).expanduser().resolve()
    return (Path.home() / ".osiris").resolve()


def resolve_skill_roots(
    workspace_root: Path | str | None = None,
    home_dir: Path | str | None = None,
) -> tuple[Path, Path]:
    """Return canonical (project_skills_dir, global_skills_dir) paths."""
    ws = Path(workspace_root).expanduser().resolve() if workspace_root else Path.cwd().resolve()
    project_root = ws / ".agents" / "skills"

    env_global = os.environ.get("OSIRIS_GLOBAL_SKILLS_DIR", "").strip()
    if home_dir is not None:
        global_root = Path(home_dir).expanduser().resolve() / ".agents" / "skills"
    elif env_global:
        global_root = Path(env_global).expanduser().resolve()
    else:
        global_root = Path.home().resolve() / ".agents" / "skills"

    return project_root, global_root


def is_forbidden_secret_path(candidate: Path, base_dir: Path | None = None) -> bool:
    """Return True if candidate or any relative segment matches forbidden secret patterns."""
    parts_to_check = [candidate.name]
    if base_dir is not None:
        try:
            rel = candidate.relative_to(base_dir)
            parts_to_check = list(rel.parts)
        except ValueError:
            parts_to_check = [candidate.name]
    return any(bool(FORBIDDEN_SECRET_RE.search(part)) for part in parts_to_check)


def safe_resolve_skill_resource(skill_dir: Path, relative_path: str) -> Path:
    """Confine resource access strictly within skill_dir and block secrets/symlink escapes."""
    clean_rel = relative_path.strip()
    if not clean_rel or "\x00" in clean_rel or clean_rel.startswith("/") or "\\" in clean_rel:
        raise PermissionError(f"Unsafe resource path outside relative skill scope: {relative_path!r}")

    canonical_skill_dir = skill_dir.resolve()
    raw_target = skill_dir / clean_rel

    for part in Path(clean_rel).parts:
        if part == "..":
            raise PermissionError(f"Parent traversal blocked by SafePathGuard: {relative_path!r}")
        if FORBIDDEN_SECRET_RE.search(part):
            raise PermissionError(f"Credential file blocked by SafePathGuard: {relative_path!r}")

    resolved_target = raw_target.resolve(strict=False)
    if not resolved_target.is_relative_to(canonical_skill_dir):
        raise PermissionError(
            f"Resource escapes skill directory ({canonical_skill_dir}): {relative_path!r}"
        )
    if is_forbidden_secret_path(resolved_target, canonical_skill_dir):
        raise PermissionError(f"Resolved secret file blocked by SafePathGuard: {resolved_target.name}")
    if not resolved_target.is_file():
        raise FileNotFoundError(f"Skill companion file not found: {clean_rel}")

    return resolved_target


def validate_skill_file(
    skill_file: Path,
    scope: SkillScope = "project",
    require_trigger_phrase: bool = True,
) -> SkillRecord:
    """Parse and validate a SKILL.md file against the universal cross-CLI schema."""
    issues: list[SkillValidationIssue] = []
    resolved_file = skill_file.resolve()
    skill_dir = resolved_file.parent
    fallback_name = skill_file.parent.name

    if not resolved_file.is_file():
        issues.append(SkillValidationIssue(str(skill_file), "MISSING_FILE", "SKILL.md does not exist"))
        return SkillRecord(
            name=fallback_name,
            description="",
            scope=scope,
            skill_dir=skill_dir,
            skill_file=resolved_file,
            sha256="",
            body="",
            referenced_files=(),
            valid=False,
            issues=tuple(issues),
        )

    raw_bytes = resolved_file.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()

    if b"\r" in raw_bytes:
        issues.append(
            SkillValidationIssue(str(skill_file), "CRLF_DETECTED", "CR byte (0x0D) detected; require LF")
        )

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        issues.append(SkillValidationIssue(str(skill_file), "INVALID_UTF8", f"Invalid UTF-8: {exc}"))
        return SkillRecord(
            name=fallback_name,
            description="",
            scope=scope,
            skill_dir=skill_dir,
            skill_file=resolved_file,
            sha256=digest,
            body="",
            referenced_files=(),
            valid=False,
            issues=tuple(issues),
        )

    if not text.endswith("\n"):
        issues.append(
            SkillValidationIssue(str(skill_file), "MISSING_TRAILING_LF", "File must end with LF newline")
        )

    if "\u2014" in text or "\u2013" in text:
        issues.append(
            SkillValidationIssue(
                str(skill_file), "TYPOGRAPHIC_DASH", "Replace em-dash or en-dash with ASCII hyphen"
            )
        )

    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    match = FRONTMATTER_RE.match(normalized_text)
    if not match:
        issues.append(
            SkillValidationIssue(
                str(skill_file), "INVALID_FRONTMATTER", "Missing or malformed --- YAML frontmatter ---"
            )
        )
        return SkillRecord(
            name=fallback_name,
            description="",
            scope=scope,
            skill_dir=skill_dir,
            skill_file=resolved_file,
            sha256=digest,
            body=normalized_text.strip(),
            referenced_files=(),
            valid=False,
            issues=tuple(issues),
        )

    fm_raw, body_raw = match.group(1), match.group(2)
    meta: dict[str, str] = {}
    for line in fm_raw.splitlines():
        if ":" not in line:
            issues.append(
                SkillValidationIssue(
                    str(skill_file),
                    "MALFORMED_YAML_LINE",
                    f"Frontmatter must use single-line key: value pairs, got {line!r}",
                )
            )
            continue
        key, val = line.split(":", 1)
        clean_key = key.strip()
        clean_val = val.strip().strip('"').strip("'")
        if clean_key in meta:
            issues.append(
                SkillValidationIssue(str(skill_file), "DUPLICATE_YAML_KEY", f"Duplicate key {clean_key!r}")
            )
        meta[clean_key] = clean_val

    if set(meta.keys()) != {"name", "description"}:
        extra_or_missing = sorted(set(meta.keys()) ^ {"name", "description"})
        issues.append(
            SkillValidationIssue(
                str(skill_file),
                "NON_UNIVERSAL_KEYS",
                f"Frontmatter must contain only ['description', 'name']; delta={extra_or_missing}",
            )
        )

    name = meta.get("name", fallback_name)
    description = meta.get("description", "")

    if not SKILL_NAME_RE.match(name) or len(name) > 64:
        issues.append(
            SkillValidationIssue(
                str(skill_file),
                "INVALID_SKILL_NAME",
                f"Skill name {name!r} must match lowercase a-z0-9- (max 64 chars)",
            )
        )

    if name != skill_file.parent.name:
        issues.append(
            SkillValidationIssue(
                str(skill_file),
                "DIR_NAME_MISMATCH",
                f"Frontmatter name {name!r} does not match directory {skill_file.parent.name!r}",
            )
        )

    if not (20 <= len(description) <= 1024):
        issues.append(
            SkillValidationIssue(
                str(skill_file),
                "DESCRIPTION_BOUNDS",
                f"Description length {len(description)} must be between 20 and 1024 characters",
            )
        )
    elif require_trigger_phrase and "use when" not in description.lower():
        issues.append(
            SkillValidationIssue(
                str(skill_file),
                "MISSING_TRIGGER_PHRASE",
                "Description should include explicit 'Use when...' trigger phrasing",
            )
        )

    body_lines = body_raw.splitlines()
    if not body_raw.strip() or len(body_lines) > 500:
        issues.append(
            SkillValidationIssue(
                str(skill_file),
                "BODY_LINE_BOUNDS",
                f"Markdown body line count ({len(body_lines)}) must be between 1 and 500",
            )
        )

    refs = tuple(sorted(set(COMPANION_REF_RE.findall(body_raw))))
    for rel_ref in refs:
        try:
            safe_resolve_skill_resource(skill_dir, rel_ref)
        except PermissionError as exc:
            issues.append(SkillValidationIssue(str(skill_file), "UNSAFE_COMPANION_PATH", str(exc)))
        except FileNotFoundError as exc:
            issues.append(SkillValidationIssue(str(skill_file), "MISSING_COMPANION_FILE", str(exc)))

    return SkillRecord(
        name=name,
        description=description,
        scope=scope,
        skill_dir=skill_dir,
        skill_file=resolved_file,
        sha256=digest,
        body=body_raw.strip(),
        referenced_files=refs,
        valid=(len(issues) == 0),
        issues=tuple(issues),
    )


def _scan_skills_root(root: Path, scope: SkillScope, require_trigger_phrase: bool) -> dict[str, SkillRecord]:
    records: dict[str, SkillRecord] = {}
    if not root.exists() or not root.is_dir():
        return records

    canonical_root = root.resolve()
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        resolved_entry = entry.resolve()
        if not resolved_entry.is_relative_to(canonical_root):
            continue
        skill_md = entry / "SKILL.md"
        if not skill_md.exists():
            continue
        resolved_md = skill_md.resolve()
        if not resolved_md.is_relative_to(resolved_entry):
            continue
        rec = validate_skill_file(skill_md, scope=scope, require_trigger_phrase=require_trigger_phrase)
        records[entry.name] = rec
    return records


def discover_skills(
    workspace_root: Path | str | None = None,
    home_dir: Path | str | None = None,
    include_invalid: bool = False,
    require_trigger_phrase: bool = True,
) -> dict[str, SkillRecord]:
    """Discover unified skills with project-over-global precedence."""
    project_root, global_root = resolve_skill_roots(workspace_root=workspace_root, home_dir=home_dir)
    merged: dict[str, SkillRecord] = {}

    global_records = _scan_skills_root(global_root, "global", require_trigger_phrase)
    for key, rec in global_records.items():
        if include_invalid or rec.valid:
            merged[key] = rec

    if not (
        project_root.exists()
        and global_root.exists()
        and project_root.resolve() == global_root.resolve()
    ):
        project_records = _scan_skills_root(project_root, "project", require_trigger_phrase)
        for key, rec in project_records.items():
            if include_invalid or rec.valid:
                merged[key] = rec
            elif key in merged and not include_invalid:
                # Project skill explicitly shadows global skill of same name; fail closed if invalid
                del merged[key]
    else:
        for key, rec in list(merged.items()):
            merged[key] = SkillRecord(
                name=rec.name,
                description=rec.description,
                scope="project",
                skill_dir=rec.skill_dir,
                skill_file=rec.skill_file,
                sha256=rec.sha256,
                body=rec.body,
                referenced_files=rec.referenced_files,
                valid=rec.valid,
                issues=rec.issues,
            )

    return dict(sorted(merged.items(), key=lambda kv: kv[0]))


def load_skill_resource(record: SkillRecord, relative_path: str) -> tuple[str, str]:
    """Safely load a companion file (e.g. references/researcher.md) and return (text, sha256)."""
    target = safe_resolve_skill_resource(record.skill_dir, relative_path)
    raw = target.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    return raw.decode("utf-8"), digest


def hydrate_skill(
    skill_name: str,
    user_task: str = "",
    workspace_root: Path | str | None = None,
    home_dir: Path | str | None = None,
    include_references: tuple[str, ...] = (),
) -> SkillHydrationBundle:
    """Hydrate a verified SKILL.md into a governed prompt bundle with SHA-256 receipt.

    By default, companion references/*.md files are NOT inlined into the prompt so
    orchestrator skills (such as deep-research) preserve clean coordinator context.
    """
    clean_name = skill_name.strip()
    all_records = discover_skills(
        workspace_root=workspace_root,
        home_dir=home_dir,
        include_invalid=True,
    )
    if clean_name not in all_records:
        raise KeyError(f"Skill {clean_name!r} not found in ./.agents/skills or ~/.agents/skills")

    rec = all_records[clean_name]
    if not rec.valid:
        codes = ", ".join(f"{iss.code}: {iss.message}" for iss in rec.issues)
        raise ValueError(f"Skill {clean_name!r} failed validation ({codes})")

    ws = Path(workspace_root).expanduser().resolve() if workspace_root else Path.cwd().resolve()
    effective_task = (
        user_task.strip()
        if user_task.strip()
        else f"Execute the '{rec.name}' skill protocol against workspace {ws}."
    )

    loaded_digests: dict[str, str] = {}
    inlined_sections: list[str] = []
    for rel_ref in include_references:
        ref_text, ref_sha = load_skill_resource(rec, rel_ref)
        loaded_digests[rel_ref] = ref_sha
        inlined_sections.append(
            f"\n\n--- COMPANION RESOURCE: {rel_ref} (SHA-256: {ref_sha[:12]}) ---\n{ref_text.strip()}"
        )

    hydrated_body = rec.body.replace("{path_to_skill}", str(rec.skill_dir))
    system_prompt = (
        f"OSIRIS v6.0.0 Governed Skill Execution\n"
        f"Skill Name: {rec.name} (scope={rec.scope}, sha256={rec.sha256[:12]})\n"
        f"Skill Base Directory: {rec.skill_dir}\n"
        f"Canonical Workspace Root: {ws}\n"
        f"Governance Contract: Models propose; deterministic code decides, measures, and records. "
        f"Respect SYS-024 claims verdicts, SafePathGuard secret isolation, and explicit file provenance.\n\n"
        f"--- SKILL PROTOCOL ---\n"
        f"{hydrated_body}"
        + "".join(inlined_sections)
    )

    task_sha = hashlib.sha256(effective_task.encode("utf-8")).hexdigest()
    receipt: dict[str, object] = {
        "schema": "osiris.skill_receipt.v1",
        "skill_name": rec.name,
        "scope": rec.scope,
        "skill_file": str(rec.skill_file),
        "skill_sha256": rec.sha256,
        "loaded_resources": dict(loaded_digests),
        "task_sha256": task_sha,
    }

    return SkillHydrationBundle(
        name=rec.name,
        scope=rec.scope,
        skill_dir=rec.skill_dir,
        skill_file=rec.skill_file,
        system_prompt=system_prompt,
        user_prompt=effective_task,
        skill_sha256=rec.sha256,
        loaded_resources=loaded_digests,
        ledger_receipt=receipt,
    )


def build_completer_dict(
    base_commands: dict[str, dict[str, None] | None] | None = None,
    workspace_root: Path | str | None = None,
    home_dir: Path | str | None = None,
) -> dict[str, dict[str, None] | None]:
    """Build a NestedCompleter-compatible mapping refreshed with live valid skills."""
    discovered = discover_skills(workspace_root=workspace_root, home_dir=home_dir, include_invalid=False)
    skill_children: dict[str, None] | None = {name: None for name in discovered} if discovered else None

    merged: dict[str, dict[str, None] | None] = {
        "/status": None,
        "/check": {"trainer": None, "git": None, "ledger": None, "system": None, "evidence": None},
        "/architecture": None,
        "/legit": {"list": None},
        "/mentor": None,
        "/bench": {"log": None},
        "/focus": None,
        "/gate": {"strict": None, "auto": None},
        "/ledger": None,
        "/help": None,
    }
    if base_commands:
        merged.update(base_commands)

    merged["/skill"] = skill_children
    merged["/skills"] = {"list": None, "audit": None}
    return merged


def complete_skill_input(
    raw_buffer: str,
    workspace_root: Path | str | None = None,
    home_dir: Path | str | None = None,
) -> list[str]:
    """Pure-Python prefix completion for fallback/non-prompt_toolkit REPL modes."""
    discovered = discover_skills(workspace_root=workspace_root, home_dir=home_dir, include_invalid=False)
    stripped = raw_buffer.lstrip()

    if stripped.startswith("/skill "):
        remainder = stripped[len("/skill ") :]
        if " " in remainder:
            return []
        return [f"/skill {name}" for name in discovered if name.startswith(remainder)]

    if stripped in ("/", "/s", "/sk", "/ski", "/skil", "/skill"):
        base = ["/skill", "/skills"]
        return [cmd for cmd in base if cmd.startswith(stripped)]

    return []


def format_skills_table(
    workspace_root: Path | str | None = None,
    home_dir: Path | str | None = None,
    max_width: int = 80,
    include_invalid: bool = False,
) -> str:
    """Format discovered skills into a Termux-safe compact ASCII/UTF-8 table."""
    records = discover_skills(
        workspace_root=workspace_root,
        home_dir=home_dir,
        include_invalid=include_invalid,
    )
    safe_width = max(48, min(max_width, 120))
    if not records:
        return "No valid unified skills found in ./.agents/skills or ~/.agents/skills."

    name_w = min(28, max(len(r.name) for r in records.values()))
    scope_w = 7
    fixed_overhead = name_w + scope_w + 9
    desc_w = max(12, safe_width - fixed_overhead)

    header = f"{'ST':<2} {'SKILL':<{name_w}}  {'SCOPE':<{scope_w}}  {'DESCRIPTION':<{desc_w}}"
    divider = "-" * min(safe_width, len(header))
    lines = [header, divider]

    for rec in records.values():
        glyph = "OK" if rec.valid else "!!"
        summary = rec.description if rec.valid else f"[{rec.issues[0].code}] {rec.issues[0].message}"
        if len(summary) > desc_w:
            summary = summary[: desc_w - 3].rstrip() + "..."
        line = f"{glyph:<2} {rec.name:<{name_w}}  {rec.scope:<{scope_w}}  {summary}"
        lines.append(line[:safe_width])

    return "\n".join(lines)
