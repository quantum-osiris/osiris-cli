"""
OSIRIS Canonical Contract Registry & Fail-Closed Validator (SYS-021 / P0)
=========================================================================

Validates incoming and outgoing payloads against canonical JSON schemas in
`contracts/{plan,execution,receipt,agent_task}_v1.json` with:
  1. Zero silent fallbacks or implicit type coercion.
  2. Standardized fail-closed refusal payloads (`SCHEMA_VALIDATION_FAILED`,
     `BACKEND_REQUIRED`, `UNAUTHORIZED_COMMAND`, `UNAUTHORIZED`).
  3. Unified header credential extraction (`X-OSIRIS-Token`, `X-Session-Token`,
     `Authorization: Bearer <token>`).
  4. Cryptographic receipt invariants (`evidence_class == "SIMULATED"` strictly
     requires `hardware == False`).
"""

from __future__ import annotations

import hmac
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "contracts"
CONTRACT_FILES = {
    "plan_v1.json": CONTRACTS_DIR / "plan_v1.json",
    "execution_v1.json": CONTRACTS_DIR / "execution_v1.json",
    "receipt_v1.json": CONTRACTS_DIR / "receipt_v1.json",
    "agent_task_v1.json": CONTRACTS_DIR / "agent_task_v1.json",
}

GOVERNED_TERMINAL_VERBS = frozenset({"ls", "cat", "express", "status", "claims", "examples"})
_ISO_DT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$")


def load_contract(name: str) -> Dict[str, Any]:
    path = CONTRACT_FILES.get(name) or (CONTRACTS_DIR / name)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_auth_token(headers: Mapping[str, str]) -> Optional[str]:
    """Extract bearer/session credential from X-OSIRIS-Token, X-Session-Token, or Authorization."""
    lowered = {k.lower(): v.strip() for k, v in headers.items() if isinstance(v, str)}
    for key in ("x-osiris-token", "x-session-token"):
        val = lowered.get(key, "")
        if val:
            return val
    auth = lowered.get("authorization", "")
    if auth.lower().startswith("bearer "):
        tok = auth[7:].strip()
        if tok:
            return tok
    return None


def verify_auth_headers(headers: Mapping[str, str], expected_token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Constant-time verification across the three recognized OSIRIS authentication headers."""
    if not expected_token:
        return False, {
            "error_code": "UNAUTHORIZED",
            "status": "REFUSED",
            "message": "Server authentication token is unconfigured",
            "violating_fields": [],
            "policy_blockers": ["MISSING_CREDENTIAL_CONFIG"],
        }
    provided = extract_auth_token(headers)
    if not provided or not hmac.compare_digest(provided.encode("utf-8"), expected_token.encode("utf-8")):
        return False, {
            "error_code": "UNAUTHORIZED",
            "status": "REFUSED",
            "message": "Missing or invalid X-OSIRIS-Token, X-Session-Token, or Bearer Authorization header",
            "violating_fields": [],
            "policy_blockers": ["UNAUTHENTICATED_REQUEST"],
        }
    return True, None


def _json_type_matches(val: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(val, dict)
    if expected == "array":
        return isinstance(val, list)
    if expected == "string":
        return isinstance(val, str)
    if expected == "boolean":
        return isinstance(val, bool)
    if expected == "integer":
        return isinstance(val, int) and not isinstance(val, bool)
    if expected == "number":
        return isinstance(val, (int, float)) and not isinstance(val, bool)
    return True


def _validate_node(val: Any, schema: Mapping[str, Any], path: str, violations: List[Dict[str, str]]) -> None:
    expected_type = schema.get("type")
    if isinstance(expected_type, str):
        if not _json_type_matches(val, expected_type):
            violations.append({"field": path or "$", "issue": f"Expected type '{expected_type}'"})
            return
    elif isinstance(expected_type, list):
        if not any(isinstance(t, str) and _json_type_matches(val, t) for t in expected_type):
            violations.append({"field": path or "$", "issue": f"Expected one of types {expected_type!r}"})
            return

    if "const" in schema and val != schema["const"]:
        violations.append({"field": path or "$", "issue": f"Must equal constant {schema['const']!r}"})
    if "enum" in schema and val not in schema["enum"]:
        violations.append({"field": path or "$", "issue": f"Value {val!r} not in allowed enum {schema['enum']}"})

    not_schema = schema.get("not")
    if isinstance(not_schema, Mapping):
        sub_violations: List[Dict[str, str]] = []
        _validate_node(val, not_schema, path, sub_violations)
        if not sub_violations:
            violations.append({"field": path or "$", "issue": "Value matched prohibited 'not' schema"})

    if isinstance(val, str):
        if "minLength" in schema and len(val) < int(schema["minLength"]):
            violations.append({"field": path or "$", "issue": f"String shorter than minLength {schema['minLength']}"})
        if "maxLength" in schema and len(val) > int(schema["maxLength"]):
            violations.append({"field": path or "$", "issue": f"String exceeds maxLength {schema['maxLength']}"})
        if "pattern" in schema and not re.search(schema["pattern"], val):
            violations.append({"field": path or "$", "issue": f"Does not match pattern {schema['pattern']}"})
        if schema.get("format") == "date-time" and not _ISO_DT_RE.match(val):
            violations.append({"field": path or "$", "issue": "Invalid ISO-8601 date-time string"})

    if isinstance(val, (int, float)) and not isinstance(val, bool):
        if "minimum" in schema and val < schema["minimum"]:
            violations.append({"field": path or "$", "issue": f"Value {val} below minimum {schema['minimum']}"})
        if "exclusiveMinimum" in schema and val <= schema["exclusiveMinimum"]:
            violations.append(
                {"field": path or "$", "issue": f"Value {val} must exceed {schema['exclusiveMinimum']}"}
            )
        if "maximum" in schema and val > schema["maximum"]:
            violations.append({"field": path or "$", "issue": f"Value {val} exceeds maximum {schema['maximum']}"})

    if isinstance(val, list):
        if "minItems" in schema and len(val) < int(schema["minItems"]):
            violations.append({"field": path or "$", "issue": f"Array has fewer than {schema['minItems']} item(s)"})
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for idx, item in enumerate(val):
                _validate_node(item, item_schema, f"{path}[{idx}]", violations)

    if isinstance(val, dict):
        props = schema.get("properties") or {}
        for req in schema.get("required") or ():
            if req not in val:
                field_path = f"{path}.{req}" if path else req
                violations.append({"field": field_path, "issue": f"Missing required property '{req}'"})
        if schema.get("additionalProperties") is False:
            for k in val:
                if k not in props:
                    field_path = f"{path}.{k}" if path else k
                    violations.append({"field": field_path, "issue": f"Unexpected property '{k}'"})
        for k, sub_schema in props.items():
            if k in val and isinstance(sub_schema, Mapping):
                field_path = f"{path}.{k}" if path else k
                _validate_node(val[k], sub_schema, field_path, violations)


def validate_contract(
    contract_name: str, payload: Any, *, require_execution_mode_on_plan: bool = False
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Validate payload against a canonical contract; returns (True, None) or (False, refusal_dict)."""
    schema = load_contract(contract_name)
    violations: List[Dict[str, str]] = []
    _validate_node(payload, schema, "", violations)

    if isinstance(payload, dict):
        if contract_name == "plan_v1.json" and require_execution_mode_on_plan and "execution_mode" not in payload:
            violations.append({"field": "execution_mode", "issue": "Missing required property 'execution_mode'"})

        if contract_name == "receipt_v1.json":
            if (
                payload.get("evidence_class") in ("SIMULATED", "PREVIEW", "SYNDROME_SIMULATED", "STATIC_RECORD")
                and payload.get("hardware") is not False
            ):
                violations.append(
                    {
                        "field": "hardware",
                        "issue": "Non-hardware execution receipts must set hardware strictly to false",
                    }
                )

        if contract_name == "agent_task_v1.json":
            kind = payload.get("type")
            req_by_kind = {
                "proposal": ("summary", "options"),
                "review": ("risks", "issues", "recommendations"),
                "refusal": ("status", "blockers", "required_remediations"),
                "event": ("stream", "timestamp", "payload"),
            }
            for req in req_by_kind.get(str(kind), ()):
                if req not in payload:
                    violations.append({"field": req, "issue": f"Missing required property '{req}' for type={kind!r}"})
            if kind == "refusal" and payload.get("status") != "REFUSED":
                violations.append({"field": "status", "issue": "Refusal payload must set status='REFUSED'"})

    if violations:
        return False, schema_refusal(contract_name, violations)
    return True, None


def schema_refusal(contract_name: str, violating_fields: List[Dict[str, str]]) -> Dict[str, Any]:
    return {
        "error_code": "SCHEMA_VALIDATION_FAILED",
        "status": "REFUSED",
        "message": f"Payload failed validation against canonical contract {contract_name}",
        "violating_fields": violating_fields,
        "policy_blockers": ["UNVALIDATED_INTENT_STRUCTURE"],
    }


def backend_required_refusal(route_label: str = "IDE API v1 backend route") -> Dict[str, Any]:
    return {
        "error_code": "BACKEND_REQUIRED",
        "status": "REFUSED",
        "message": f"{route_label} is unconfigured for this operation",
        "violating_fields": [],
        "policy_blockers": ["UNWIRED_SERVICE_ADAPTER"],
    }


def unauthorized_command_refusal() -> Dict[str, Any]:
    return {
        "error_code": "UNAUTHORIZED_COMMAND",
        "status": "REFUSED",
        "message": "Arbitrary shell execution and unvalidated subprocesses are prohibited.",
    }


def validate_governed_terminal_command(raw_command: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Allow only governed read-only workspace verbs; reject shell flags, pipes, and subprocesses."""
    stripped = (raw_command or "").strip()
    if not stripped or any(sep in stripped for sep in (";", "|", "&", "`", "$(", ">", "<", "\n")):
        return False, unauthorized_command_refusal()
    parts = stripped.split()
    verb = parts[0].lower()
    if verb not in GOVERNED_TERMINAL_VERBS:
        return False, unauthorized_command_refusal()
    if any(arg.startswith("-") or ".." in arg for arg in parts[1:]):
        return False, unauthorized_command_refusal()
    return True, None
