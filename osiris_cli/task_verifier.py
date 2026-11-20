"""OSIRIS Independent Task Verifier — separate OS process (REQ-VER-4, G-8).

Integrity and acceptance are strictly separated:
  1. Integrity: re-reads every declared artifact file directly from disk and recomputes its
     SHA-256 digest from raw bytes (never from an in-memory object passed by the executor).
     Any single-byte mismatch returns integrity="FAIL", code="INTEGRITY_FAIL".
  2. Acceptance: evaluates explicit node acceptance criteria against observed disk artifacts
     and exit status (`exit_code_zero`, `stdout_contains:<text>`, `file_exists:<rel>`,
     `non_empty_output`, `json_field_equals:<key>=<value>`).

A node reaches VERIFIED only when BOTH integrity == "PASS" AND acceptance == "PASS".
Invoked in a child process through `exec_kernel.run` (`REQ-EXEC-1`) so `verifier_pid != os.getpid()`.

Usage:
    python -m osiris_cli.task_verifier --manifest /path/to/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

VERIFIER_VERSION = "task-verifier/1.0"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _read_artifact_text(path: Path) -> str:
    """Read stdout/text from either a kernel output.jsonl artifact or a plain text/JSON file."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".jsonl":
        chunks: List[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and isinstance(obj.get("text"), str):
                    chunks.append(obj["text"])
                    continue
            except ValueError:
                pass
            chunks.append(line)
        return "".join(chunks)
    return raw


def verify_manifest_dict(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Run separate integrity and acceptance checks from a manifest on disk."""
    workdir = Path(str(manifest.get("workdir") or ".")).resolve()
    artifacts = manifest.get("artifacts") or []
    criteria = manifest.get("acceptance_criteria") or []
    observed_exit_code = manifest.get("exit_code")
    primary_artifact_path: Optional[Path] = None

    integrity_checks: List[Dict[str, Any]] = []
    recomputed_digests: Dict[str, str] = {}
    integrity_ok = True

    if not isinstance(artifacts, list) or len(artifacts) == 0:
        integrity_ok = False
        integrity_checks.append(
            {
                "check": "artifact_present",
                "passed": False,
                "reason": "no disk artifacts declared for verification",
            }
        )
    else:
        for item in artifacts:
            if not isinstance(item, Mapping):
                integrity_ok = False
                integrity_checks.append(
                    {"check": "artifact_entry", "passed": False, "reason": "malformed artifact entry"}
                )
                continue
            raw_path = str(item.get("path") or "").strip()
            expected_sha = str(item.get("expected_sha256") or "").strip().lower()
            p = Path(raw_path)
            if primary_artifact_path is None:
                primary_artifact_path = p
            if not raw_path or not p.is_file():
                integrity_ok = False
                integrity_checks.append(
                    {
                        "check": "disk_digest",
                        "path": raw_path,
                        "passed": False,
                        "reason": "artifact file missing on disk",
                    }
                )
                continue
            actual_sha = _sha256_file(p)
            recomputed_digests[raw_path] = actual_sha
            if not expected_sha or actual_sha != expected_sha:
                integrity_ok = False
                integrity_checks.append(
                    {
                        "check": "disk_digest",
                        "path": raw_path,
                        "expected_sha256": expected_sha,
                        "actual_sha256": actual_sha,
                        "passed": False,
                        "reason": "INTEGRITY_FAIL: disk bytes digest does not match expected SHA-256",
                    }
                )
            else:
                integrity_checks.append(
                    {
                        "check": "disk_digest",
                        "path": raw_path,
                        "actual_sha256": actual_sha,
                        "passed": True,
                        "reason": "disk SHA-256 verified",
                    }
                )

    # Acceptance phase runs independently of integrity so both results are explicit.
    acceptance_checks: List[Dict[str, Any]] = []
    acceptance_ok = True

    if not isinstance(criteria, list) or len(criteria) == 0:
        acceptance_ok = False
        acceptance_checks.append(
            {
                "criterion": "non_empty_criteria",
                "passed": False,
                "reason": "node declared no acceptance criteria",
            }
        )
    else:
        cached_text: Optional[str] = None

        def _primary_text() -> str:
            nonlocal cached_text
            if cached_text is None:
                if primary_artifact_path is not None and primary_artifact_path.is_file():
                    cached_text = _read_artifact_text(primary_artifact_path)
                else:
                    cached_text = ""
            return cached_text

        for crit in criteria:
            c = str(crit).strip()
            if c == "exit_code_zero":
                ok = observed_exit_code == 0
                acceptance_checks.append(
                    {
                        "criterion": c,
                        "passed": ok,
                        "reason": "exit_code is 0" if ok else f"observed exit_code={observed_exit_code!r}",
                    }
                )
                if not ok:
                    acceptance_ok = False
            elif c == "non_empty_output":
                txt = _primary_text()
                ok = bool(txt.strip())
                acceptance_checks.append(
                    {
                        "criterion": c,
                        "passed": ok,
                        "reason": "output non-empty on disk" if ok else "captured output on disk is empty",
                    }
                )
                if not ok:
                    acceptance_ok = False
            elif c.startswith("stdout_contains:"):
                needle = c.split(":", 1)[1]
                txt = _primary_text()
                ok = needle in txt
                acceptance_checks.append(
                    {
                        "criterion": c,
                        "passed": ok,
                        "reason": f"found {needle!r} in disk output" if ok else f"missing {needle!r} in disk output",
                    }
                )
                if not ok:
                    acceptance_ok = False
            elif c.startswith("file_exists:"):
                rel = c.split(":", 1)[1].strip()
                target = (workdir / rel).resolve()
                inside = str(target) == str(workdir) or str(target).startswith(str(workdir) + os.sep)
                ok = inside and target.is_file()
                acceptance_checks.append(
                    {
                        "criterion": c,
                        "passed": ok,
                        "reason": f"file exists: {rel}" if ok else f"missing or escaped workdir: {rel}",
                    }
                )
                if not ok:
                    acceptance_ok = False
            elif c.startswith("json_field_equals:"):
                expr = c.split(":", 1)[1]
                if "=" not in expr:
                    acceptance_ok = False
                    acceptance_checks.append(
                        {"criterion": c, "passed": False, "reason": "malformed json_field_equals criterion"}
                    )
                    continue
                key, expected_val = expr.split("=", 1)
                ok = False
                reason = "primary artifact is not JSON"
                if primary_artifact_path is not None and primary_artifact_path.is_file():
                    try:
                        parsed = json.loads(primary_artifact_path.read_text(encoding="utf-8"))
                        cur: Any = parsed
                        for seg in key.split("."):
                            cur = cur[seg]
                        ok = str(cur) == expected_val
                        reason = f"{key}=={cur!r}" if ok else f"{key} was {cur!r}, expected {expected_val!r}"
                    except Exception as exc:
                        reason = f"could not evaluate {key}: {exc}"
                acceptance_checks.append({"criterion": c, "passed": ok, "reason": reason})
                if not ok:
                    acceptance_ok = False
            else:
                # Unknown acceptance criterion fails closed (never silently accepted).
                acceptance_ok = False
                acceptance_checks.append(
                    {"criterion": c, "passed": False, "reason": f"unknown acceptance criterion: {c!r}"}
                )

    integrity_label = "PASS" if integrity_ok else "FAIL"
    acceptance_label = "PASS" if acceptance_ok else "FAIL"
    if integrity_ok and acceptance_ok:
        overall = "PASS"
        code = "VERIFIED"
    elif not integrity_ok:
        overall = "FAIL"
        code = "INTEGRITY_FAIL"
    else:
        overall = "FAIL"
        code = "ACCEPTANCE_FAIL"

    return {
        "verifier_version": VERIFIER_VERSION,
        "verifier_pid": os.getpid(),
        "integrity": integrity_label,
        "acceptance": acceptance_label,
        "overall": overall,
        "code": code,
        "integrity_checks": integrity_checks,
        "acceptance_checks": acceptance_checks,
        "recomputed_digests": recomputed_digests,
    }


def run_separate_verifier(
    *,
    workdir: str,
    artifacts: List[Dict[str, str]],
    acceptance_criteria: List[str],
    exit_code: Optional[int],
    exec_journal: Any,
    artifacts_dir: Path,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Run task_verifier in a separate child process via exec_kernel.run (REQ-VER-4, REQ-EXEC-1).

    Returns `(verification_report, exec_journal_entry_hash)`.
    """
    from .exec_kernel import CommandRequest, _last_line, run as kernel_run

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifacts_dir / f"verify-manifest-{uuid.uuid4().hex[:12]}.json"
    manifest = {
        "workdir": os.path.realpath(workdir),
        "artifacts": artifacts,
        "acceptance_criteria": acceptance_criteria,
        "exit_code": exit_code,
    }
    fd = os.open(manifest_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, sort_keys=True, indent=2)
        fh.flush()
        os.fsync(fh.fileno())

    verifier_script = str(Path(__file__).resolve())
    req = CommandRequest(
        argv=(sys.executable, verifier_script, "--manifest", str(manifest_path)),
        cwd=os.path.realpath(workdir),
        origin="human",
        root=os.path.realpath(workdir),
        timeout_s=30.0,
    )
    cmd_res = kernel_run(
        req,
        confirm=lambda _prompt, _cls: True,
        journal=exec_journal,
        artifacts_dir=artifacts_dir,
    )

    journal_hash: Optional[str] = None
    last = _last_line(exec_journal.path)
    if last is not None:
        try:
            journal_hash = json.loads(last.decode("utf-8")).get("hash")
        except Exception:
            journal_hash = None

    raw_out = (cmd_res.stdout_view or "").strip()
    try:
        report = json.loads(raw_out)
    except ValueError:
        return (
            {
                "verifier_version": VERIFIER_VERSION,
                "verifier_pid": None,
                "integrity": "FAIL",
                "acceptance": "FAIL",
                "overall": "FAIL",
                "code": "VERIFIER_ERROR",
                "integrity_checks": [
                    {
                        "check": "verifier_subprocess",
                        "passed": False,
                        "reason": f"verifier child process returned non-JSON (decision={cmd_res.decision}, exit={cmd_res.exit_code}): {(cmd_res.stderr_view or raw_out)[:240]}",
                    }
                ],
                "acceptance_checks": [],
                "recomputed_digests": {},
            },
            journal_hash,
        )

    if report.get("verifier_pid") == os.getpid():
        report["integrity"] = "FAIL"
        report["overall"] = "FAIL"
        report["code"] = "SAME_PROCESS_VERIFIER_FORBIDDEN"
    return report, journal_hash


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m osiris_cli.task_verifier", description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", required=True, help="Path to verification manifest JSON")
    args = ap.parse_args(argv)
    try:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    except Exception as exc:
        err = {
            "verifier_version": VERIFIER_VERSION,
            "verifier_pid": os.getpid(),
            "integrity": "FAIL",
            "acceptance": "FAIL",
            "overall": "FAIL",
            "code": "MANIFEST_UNREADABLE",
            "integrity_checks": [{"check": "manifest", "passed": False, "reason": str(exc)}],
            "acceptance_checks": [],
            "recomputed_digests": {},
        }
        print(json.dumps(err, sort_keys=True))
        return 1
    result = verify_manifest_dict(manifest)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
