"""
OSIRIS GitHub Bridge & PR Management Subsystem
=============================================
Provides governed CLI and REPL access to GitHub repositories and PR workflows.
Enforces OSIRIS Dual-Node Governance:
  - Read-Only / Inspection Tier: `gh auth status`, `gh pr list`, `gh pr view`, `gh repo list`
  - Mutating / State-Advancing Tier: `gh pr create`, `gh pr merge`, `git push`
    Strictly gated behind SYS-021 payload-bound authorization grants.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Sequence, Union

# Anything that looks like a GitHub credential is removed from every string this module returns.
_SECRET_PATTERNS = (
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_*]{8,}|github_pat_[A-Za-z0-9_]{20,})"),
    re.compile(r"(?im)^(\s*-?\s*token:\s*).*$"),
    re.compile(r"(?i)(authorization:\s*(?:bearer|token)\s+)\S+"),
)
DEFAULT_TIMEOUT_S = 30
_NETWORK_HINTS = ("dial tcp", "could not resolve", "connection refused", "connection reset", "no such host",
                  "network is unreachable", "tls handshake", "i/o timeout", "timeout awaiting", "eof")


def redact(text: Optional[str]) -> str:
    """Remove token-like material from gh output before it is logged, returned or displayed."""
    out = text or ""
    out = _SECRET_PATTERNS[0].sub("<REDACTED>", out)
    out = _SECRET_PATTERNS[1].sub(lambda m: m.group(1) + "<REDACTED>", out)
    out = _SECRET_PATTERNS[2].sub(lambda m: m.group(1) + "<REDACTED>", out)
    return out


def classify_failure(returncode: int, stderr: str) -> str:
    """Map a failed gh invocation to a stable error kind."""
    low = (stderr or "").lower()
    if "http 401" in low or "bad credentials" in low or "not logged" in low or "gh auth login" in low:
        return "AUTH"
    if "http 403" in low:
        return "FORBIDDEN"
    if "http 404" in low or "not found" in low:
        return "NOT_FOUND"
    if any(h in low for h in _NETWORK_HINTS):
        return "NETWORK"
    if "http 4" in low or "http 5" in low:
        return "HTTP_ERROR"
    return "GH_ERROR"


class GitHubBridge:
    """Governed GitHub CLI interface for OSIRIS."""

    def __init__(self, gh_binary: Optional[str] = None):
        self.gh_binary = gh_binary or shutil.which("gh")

    def is_installed(self) -> bool:
        return self.gh_binary is not None

    def run_gh(self, args: Sequence[str], *, stdin: Optional[str] = None,
               timeout: float = DEFAULT_TIMEOUT_S) -> Dict[str, Any]:
        """Run gh with a fixed argument vector (never a shell). Output is redacted.

        Returns {"ok": bool, "stdout", "stderr", "kind"} where kind is None on success or one of
        NOT_INSTALLED, TIMEOUT, NETWORK, AUTH, FORBIDDEN, NOT_FOUND, HTTP_ERROR, GH_ERROR, EXEC_ERROR.
        """
        if not self.is_installed():
            return {"ok": False, "stdout": "", "stderr": "gh CLI not installed", "kind": "NOT_INSTALLED"}
        if not all(isinstance(a, str) for a in args):
            raise TypeError("gh arguments must be strings")
        env = dict(os.environ, GH_PROMPT_DISABLED="1", GH_NO_UPDATE_NOTIFIER="1", NO_COLOR="1")
        try:
            res = subprocess.run([self.gh_binary, *args], input=stdin, capture_output=True, text=True,
                                 check=False, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return {"ok": False, "stdout": "", "stderr": f"gh timed out after {timeout}s", "kind": "TIMEOUT"}
        except OSError as e:
            return {"ok": False, "stdout": "", "stderr": redact(str(e)), "kind": "EXEC_ERROR"}
        out, err = redact(res.stdout), redact(res.stderr)
        if res.returncode != 0:
            return {"ok": False, "stdout": out, "stderr": err.strip(), "kind": classify_failure(res.returncode, err)}
        return {"ok": True, "stdout": out, "stderr": err, "kind": None}

    def api(self, endpoint: str, *, method: str = "GET", body: Optional[Dict[str, Any]] = None,
            timeout: float = DEFAULT_TIMEOUT_S) -> Dict[str, Any]:
        """Call the GitHub REST API through ``gh api``. A request body travels on stdin as JSON,
        so no user-supplied value is ever parsed by gh as a flag or an @file reference."""
        if method not in ("GET", "POST"):
            raise ValueError(f"method {method!r} not supported")
        if endpoint.startswith("-") or "\n" in endpoint or "://" in endpoint:
            raise ValueError("endpoint must be a relative API path")
        args = ["api", "--method", method, "-H", "Accept: application/vnd.github+json", endpoint]
        stdin = None
        if body is not None:
            args += ["--input", "-"]
            stdin = json.dumps(body)
        r = self.run_gh(args, stdin=stdin, timeout=timeout)
        if not r["ok"]:
            return {"status": "ERROR", "kind": r["kind"], "reason": r["stderr"] or "gh api failed"}
        try:
            data = json.loads(r["stdout"]) if r["stdout"].strip() else None
        except json.JSONDecodeError:
            return {"status": "ERROR", "kind": "BAD_RESPONSE", "reason": "gh api returned non-JSON output"}
        return {"status": "SUCCESS", "data": data}

    def authenticated_login(self) -> Dict[str, Any]:
        """The login the GitHub API says this token belongs to (not what gh's config claims)."""
        r = self.api("user")
        if r["status"] != "SUCCESS":
            return r
        login = (r["data"] or {}).get("login")
        if not login:
            return {"status": "ERROR", "kind": "BAD_RESPONSE", "reason": "user endpoint returned no login"}
        return {"status": "SUCCESS", "login": login}

    def check_auth(self) -> Dict[str, Any]:
        """Check authentication status with GitHub CLI."""
        if not self.is_installed():
            return {
                "installed": False,
                "authenticated": False,
                "reason": "GitHub CLI ('gh') is not installed in the current environment PATH."
            }

        try:
            env = dict(os.environ, GH_PROMPT_DISABLED="1", GH_NO_UPDATE_NOTIFIER="1", NO_COLOR="1")
            res = subprocess.run(
                [self.gh_binary, "auth", "status"],
                capture_output=True,
                text=True,
                check=False,
                timeout=DEFAULT_TIMEOUT_S,
                env=env,
            )
            # gh auth status exits 0 when fully authenticated
            is_auth = (res.returncode == 0)
            combined_output = (res.stdout or "") + (res.stderr or "")
            account = "unknown"
            scopes = []
            for line in combined_output.splitlines():
                line_s = line.strip()
                if "account" in line_s:
                    parts = line_s.split()
                    if "account" in parts:
                        idx = parts.index("account")
                        if idx + 1 < len(parts):
                            account = parts[idx + 1]
                if "Token scopes:" in line_s:
                    scopes_raw = line_s.split("Token scopes:", 1)[1].strip()
                    scopes = [s.strip(" '\"") for s in scopes_raw.split(",")]

            return {
                "installed": True,
                "authenticated": is_auth,
                "account": account,
                "scopes": scopes,
                "raw": redact(combined_output).strip()
            }
        except Exception as e:
            return {
                "installed": True,
                "authenticated": False,
                "reason": redact(f"Execution error probing gh auth status: {e}")
            }

    def list_prs(self, limit: int = 10) -> Dict[str, Any]:
        """List open pull requests (Read-Only tier)."""
        r = self.run_gh([
            "pr", "list",
            "--limit", str(limit),
            "--json", "number,title,state,headRefName,url,updatedAt",
        ])
        if not r["ok"]:
            return {"status": "ERROR", "kind": r["kind"], "reason": r["stderr"] or "Failed to list PRs"}
        try:
            prs = json.loads(r["stdout"]) if r["stdout"].strip() else []
            return {"status": "SUCCESS", "prs": prs}
        except Exception as e:
            return {"status": "ERROR", "kind": "BAD_RESPONSE", "reason": redact(str(e))}

    def view_pr(self, pr_id: Union[int, str]) -> Dict[str, Any]:
        """View PR details avoiding classic projectCards GraphQL issues (Read-Only tier)."""
        r = self.run_gh([
            "pr", "view", str(pr_id),
            "--json", "number,title,state,headRefName,baseRefName,url,body",
        ])
        if not r["ok"]:
            return {"status": "ERROR", "kind": r["kind"], "reason": r["stderr"] or f"Failed to view PR #{pr_id}"}
        try:
            pr_data = json.loads(r["stdout"]) if r["stdout"].strip() else {}
            return {"status": "SUCCESS", "pr": pr_data}
        except Exception as e:
            return {"status": "ERROR", "kind": "BAD_RESPONSE", "reason": redact(str(e))}

    def checks_pr(self, pr_id: Union[int, str]) -> Dict[str, Any]:
        """View CI/CD check runs for a PR (Read-Only tier)."""
        if not self.is_installed():
            return {"status": "ERROR", "reason": "gh CLI not installed"}
        r = self.run_gh(["pr", "checks", str(pr_id)])
        return {
            "status": "SUCCESS" if r["ok"] else "FAILED",
            "output": (r["stdout"] or r["stderr"]).strip(),
        }

    def list_repos(self, limit: int = 10) -> Dict[str, Any]:
        """List repositories for authenticated user (Read-Only tier)."""
        r = self.run_gh([
            "repo", "list",
            "--limit", str(limit),
            "--json", "nameWithOwner,isPrivate,updatedAt,url",
        ])
        if not r["ok"]:
            return {"status": "ERROR", "kind": r["kind"], "reason": r["stderr"] or "Failed to list repos"}
        try:
            repos = json.loads(r["stdout"]) if r["stdout"].strip() else []
            return {"status": "SUCCESS", "repos": repos}
        except Exception as e:
            return {"status": "ERROR", "kind": "BAD_RESPONSE", "reason": redact(str(e))}

    def merge_pr(self, pr_id: Union[int, str], grant_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Merge a pull request.
        MUTATING / STATE-ADVANCING: strictly requires SYS-021 payload-bound grant.
        """
        if not grant_id or not grant_id.strip():
            return {
                "status": "BLOCKED",
                "reason": (
                    "[SYS-021 BLOCKED] Bare merge directives forbidden. "
                    "Provide explicit payload-bound grant_id binding PR digest."
                )
            }
        return {
            "status": "HOLD",
            "reason": f"Grant {grant_id} registered. Remote merge held for human review."
        }

    def format_dashboard(self) -> str:
        """Format a human-readable GitHub governance card for REPL / CLI."""
        lines = []
        lines.append("\033[1;36m[OSIRIS::GITHUB REPOSITORY & PR SUBSYSTEM]\033[0m")
        auth = self.check_auth()
        if not auth["installed"]:
            lines.append("  \033[1;31mGitHub CLI ('gh') is not installed in PATH.\033[0m")
            return "\n".join(lines)

        if not auth["authenticated"]:
            lines.append("  \033[1;33mGitHub CLI installed but not authenticated. Run 'gh auth login'.\033[0m")
            return "\n".join(lines)

        account = auth.get("account", "unknown")
        scopes = ", ".join(auth.get("scopes", [])) or "default"
        lines.append(f"  ├─ Authenticated Account : \033[1;32m{account}\033[0m")
        lines.append(f"  ├─ Authorized Scopes    : {scopes}")
        lines.append("  ├─ Governance Tier      : Read-Only (Inspection Active · Mutating Ops Gated)")

        pr_res = self.list_prs(limit=5)
        if pr_res["status"] == "SUCCESS" and pr_res["prs"]:
            lines.append("  └─ Open Pull Requests   :")
            for pr in pr_res["prs"]:
                num = pr.get("number", "?")
                title = pr.get("title", "")
                head = pr.get("headRefName", "")
                lines.append(f"     • \033[1;37m#{num}\033[0m: {title} (\033[1;34m{head}\033[0m)")
        elif pr_res["status"] == "SUCCESS":
            lines.append("  └─ Open Pull Requests   : None currently open")
        else:
            lines.append(f"  └─ Pull Requests Error  : {pr_res.get('reason')}")

        return "\n".join(lines)
