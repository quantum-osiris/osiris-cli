"""Deterministic evidence firewall for agent-authored reports.

A report sentence that asserts a test result, a completed action, a measurement or a
registered claim is checked against ExecutionEvidence (produced only by the OSIRIS test
runner) and cited artifacts. No model is consulted. Verdicts reuse the register's words:

  SUPPORTED   the evidence on hand backs the sentence
  REFUTED     the evidence on hand contradicts it (or it restates a refuted/null registered claim)
  ARTIFACT    it restates a registered claim recorded as a measurement artifact
  UNVERIFIED  it asserts something checkable and no evidence was supplied
  UNMEASURED  it states a number/interval with no citation to a recorded measurement

Decision: any REFUTED/ARTIFACT -> REJECT; else any UNVERIFIED/UNMEASURED -> QUARANTINE; else ACCEPT.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .. import claims as claims_register


@dataclass(frozen=True)
class ExecutionEvidence:
    """What the runner observed. Constructed by the runner, never by an agent."""
    argv: tuple
    returncode: int
    passed: int
    failed: int
    errors: int
    stdout_digest: str
    stdout_tail: str
    workdir_digest: str = ""

    @property
    def green(self) -> bool:
        return self.returncode == 0 and self.passed > 0 and self.failed == 0 and self.errors == 0


@dataclass(frozen=True)
class ClaimFinding:
    sentence: str
    kind: str
    verdict: str
    reason: str


@dataclass
class FirewallResult:
    findings: List[ClaimFinding] = field(default_factory=list)

    @property
    def decision(self) -> str:
        verdicts = {f.verdict for f in self.findings}
        if verdicts & {"REFUTED", "ARTIFACT"}:
            return "REJECT"
        if verdicts & {"UNVERIFIED", "UNMEASURED"}:
            return "QUARANTINE"
        return "ACCEPT"


_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_TEST_PASS = re.compile(
    r"\b(all\s+)?(unit\s+|regression\s+|the\s+)?tests?\s+(all\s+)?(now\s+)?(pass(es|ed)?|are\s+green|succeed(ed)?)\b"
    r"|\b\d+\s+(tests?\s+)?passed\b|\bpytest\s+(passes|passed|is\s+green)\b", re.I)
_ACTION = re.compile(r"\bI\s+(have\s+)?(ran|run|executed|verified|benchmarked|tested|measured)\b"
                     r"|\b(was|were|has\s+been|have\s+been)\s+(run|executed|verified|benchmarked|measured)\b",
                     re.I)
_NUMBER = re.compile(r"(?:\b\d+(?:\.\d+)?\s*%|\b\d+(?:\.\d+)?\s*(?:ms|µs|us|s|x|×)\b|[±]\s*\d|"
                     r"\b(?:CI|confidence interval|p\s*[=<]|fidelity|latency|speedup|accuracy|"
                     r"F1|AUC)\b[^.\n]{0,25}\d)", re.I)
_CITE = re.compile(r"\[evidence:([A-Za-z0-9_.:-]+)\]")
_BACKEND = re.compile(r"\bibm_[a-z]+\b|\bibmq_[a-z_]+\b", re.I)
_NEGATION = re.compile(r"\b(not|no|never|refuted|failed|artifact|null|cannot|can't|isn't|"
                       r"unsupported|retracted|withdrawn|was\s+wrong)\b", re.I)


def check_report(report: str, evidence: Optional[ExecutionEvidence] = None,
                 artifacts: Optional[Dict[str, str]] = None) -> FirewallResult:
    """artifacts maps a citation id (``[evidence:<id>]``) to recorded artifact text."""
    artifacts = artifacts or {}
    out = FirewallResult()
    for sent in (s.strip() for s in _SENT.split(report or "")):
        if not sent:
            continue
        cites = _CITE.findall(sent)

        if _TEST_PASS.search(sent):
            if evidence is None:
                out.findings.append(ClaimFinding(sent, "test_result", "UNVERIFIED",
                                                 "asserts tests passed; no execution evidence"))
            elif evidence.green:
                out.findings.append(ClaimFinding(sent, "test_result", "SUPPORTED",
                                                 f"runner: {evidence.passed} passed, rc=0"))
            else:
                out.findings.append(ClaimFinding(
                    sent, "test_result", "REFUTED",
                    f"runner: rc={evidence.returncode}, passed={evidence.passed}, "
                    f"failed={evidence.failed}, errors={evidence.errors}"))
        elif _ACTION.search(sent) and evidence is None:
            out.findings.append(ClaimFinding(sent, "completed_action", "UNVERIFIED",
                                             "asserts an action was completed; no execution evidence"))

        if _NUMBER.search(sent) and not _TEST_PASS.search(sent):
            nums = re.findall(r"\d+(?:\.\d+)?", _CITE.sub("", sent))
            missing = [c for c in cites if c not in artifacts]
            backed = [c for c in cites if c in artifacts
                      and all(n in artifacts[c] for n in nums)]
            if not cites:
                out.findings.append(ClaimFinding(sent, "measurement", "UNMEASURED",
                                                 "states a number with no [evidence:<id>] citation"))
            elif missing:
                out.findings.append(ClaimFinding(sent, "measurement", "UNMEASURED",
                                                 f"cites unknown evidence {missing}"))
            elif backed:
                out.findings.append(ClaimFinding(sent, "measurement", "SUPPORTED",
                                                 f"every number appears in {backed}"))
            else:
                out.findings.append(ClaimFinding(sent, "measurement", "REFUTED",
                                                 "cited artifact does not contain the stated number(s)"))

        for b in _BACKEND.findall(sent):
            if not any(b.lower() in t.lower() for t in artifacts.values()):
                out.findings.append(ClaimFinding(sent, "backend", "UNVERIFIED",
                                                 f"names backend {b}; no cited artifact mentions it"))

        if not _NEGATION.search(sent):
            for c in claims_register.match(sent):
                if c.verdict == "ARTIFACT":
                    out.findings.append(ClaimFinding(sent, "registered_claim", "ARTIFACT",
                                                     f"restates {c.id}, recorded as an artifact"))
                elif c.verdict in ("REFUTED", "RULED_OUT", "NULL", "NOT_MEASURED"):
                    out.findings.append(ClaimFinding(sent, "registered_claim", "REFUTED",
                                                     f"restates {c.id}, recorded as {c.verdict}"))
    return out
