"""Deterministic evidence firewall for agent-authored reports.

A report sentence that asserts a test result, a completed action, a measurement or a
registered claim is checked against ExecutionEvidence (produced only by the OSIRIS test
runner) and cited artifacts. No model is consulted. Verdicts reuse the register's words:

  SUPPORTED   the evidence on hand backs the sentence (unquoted agent prose only)
  REFUTED     the evidence on hand contradicts it (or it endorses a refuted/null registered claim)
  ARTIFACT    it endorses a registered claim recorded as a measurement artifact
  UNVERIFIED  it asserts something checkable and no evidence was supplied
  UNMEASURED  it states a number/interval with no citation to a recorded measurement
  AMBIGUOUS   it quotes untrusted assertions or touches a refuted claim ambiguously without rejecting it

Decision: any REFUTED/ARTIFACT -> REJECT; else any UNVERIFIED/UNMEASURED/AMBIGUOUS -> QUARANTINE; else ACCEPT.
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
        if verdicts & {"UNVERIFIED", "UNMEASURED", "AMBIGUOUS"}:
            return "QUARANTINE"
        if any(v != "SUPPORTED" for v in verdicts):
            return "QUARANTINE"
        return "ACCEPT"


_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_QUOTE_SPAN = re.compile(r'"[^"\n]{3,}"|“[^”\n]{3,}”|(?<!\w)\'[^\'\n]{4,}\'(?!\w)')
_TEST_PASS = re.compile(
    r"\b(all\s+)?(unit\s+|regression\s+|the\s+)?tests?\s+(all\s+)?(now\s+)?(pass(es|ed)?|are\s+green|succeed(ed)?)\b"
    r"|\b\d+\s+(tests?\s+)?passed\b|\bpytest\s+(passes|passed|is\s+green)\b", re.I)
_ACTION = re.compile(r"\bI\s+(have\s+)?(ran|run|executed|verified|benchmarked|tested|measured|deployed|pushed)\b"
                     r"|\b(was|were|has\s+been|have\s+been)\s+(run|executed|verified|benchmarked|measured|deployed)\b",
                     re.I)
_NUMBER = re.compile(r"(?:\b\d+(?:\.\d+)?\s*%|\b\d+(?:\.\d+)?\s*(?:ms|µs|us|s|x|×)\b|[±]\s*\d|"
                     r"\b(?:CI|confidence interval|p\s*[=<]|fidelity|latency|speedup|accuracy|"
                     r"F1|AUC)\b[^.\n]{0,25}\d)", re.I)
_CITE = re.compile(r"\[evidence:([A-Za-z0-9_.:-]+)\]")
_BACKEND = re.compile(r"\bibm_[a-z]+\b|\bibmq_[a-z_]+\b", re.I)
_REJECTION = re.compile(
    r"\b(?:not|no|never|neither|nor|isn'?t|aren'?t|wasn'?t|weren'?t|doesn'?t|didn'?t|cannot|can'?t|"
    r"won'?t|shouldn'?t|mustn'?t|refut\w*|fail\w*|artifact\w*|ruled[\s_-]*out|not[\s_-]*measured|"
    r"unmeasured|no[\s_-]*evidence|overclaim\w*|unsupported|unverified|retract\w*|withdraw\w*|"
    r"disprov\w*|debunk\w*|invalid\w*|false|wrong|myth|null|errat\w*|object\w*|reject\w*|"
    r"critic\w*|skeptic\w*|avoid\w*|forbid\w*|prohibit\w*|exclude\w*|discard\w*)\b", re.I)
_AMBIGUOUS_HEDGE = re.compile(
    r"\?\s*$|\b(?:whether|unclear|ambiguous|questionable|alleged\w*|purported\w*|unconfirmed|"
    r"might|could\s+be|if\s+true)\b", re.I)
_ENDORSE_PREDICATE = re.compile(
    r"\b(?:is|are|was|were|be|has|have|had|proves?|proved|shows?|showed|demonstrates?|demonstrated|"
    r"achieves?|achieved|measures?|measured|confirms?|confirmed|validates?|validated|establishes?|"
    r"established|locks?|locked|stabilizes?|stabilized|yields?|yielded|gives?|gave|produces?|"
    r"produced|delivers?|delivered|reaches?|reached|found|operates?|runs?|simulates?|simulated|"
    r"exceeds?|exceeded|enables?|enabled|harvest\w*|suppress\w*|oscillat\w*|quantiz\w*)\b"
    r"|[=≈~]\s*\d", re.I)
_BARE_ID_LIST = re.compile(r"^[A-Za-z0-9_,\s:;().-]+$")


def _is_bare_claim_listing(text: str) -> bool:
    """True when text is only a label/header plus uppercase claim IDs (e.g. 'Evidence: CHSH_TEST, LAMBDA_PHI')."""
    body = text.split(":", 1)[-1].strip() if ":" in text else text.strip()
    if not body or not _BARE_ID_LIST.match(body):
        return False
    tokens = [t for t in re.split(r"[\s,;().]+", body) if t]
    return bool(tokens) and all(t == t.upper() and any(ch.isalpha() for ch in t) for t in tokens)


def check_report(report: str, evidence: Optional[ExecutionEvidence] = None,
                 artifacts: Optional[Dict[str, str]] = None) -> FirewallResult:
    """artifacts maps a citation id (``[evidence:<id>]``) to recorded artifact text."""
    artifacts = artifacts or {}
    out = FirewallResult()
    for raw_sent in _SENT.split(report or ""):
        sent = raw_sent.strip()
        if not sent:
            continue
        is_blockquote = sent.startswith(">")
        unquoted = "" if is_blockquote else _QUOTE_SPAN.sub(" ", sent).strip()
        has_quote = is_blockquote or (unquoted != sent)
        rejected_in_prose = bool(unquoted and _REJECTION.search(unquoted))
        cites = _CITE.findall(unquoted)   # citations inside untrusted quotes carry no authority

        if _TEST_PASS.search(unquoted):
            if rejected_in_prose:
                pass
            elif evidence is None:
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
        elif has_quote and _TEST_PASS.search(sent):
            if not rejected_in_prose:
                out.findings.append(ClaimFinding(
                    sent, "quoted_untrusted", "AMBIGUOUS",
                    "quotes a test-pass assertion without rejecting it; quoted text carries no authority"))
        elif _ACTION.search(unquoted) and not rejected_in_prose and evidence is None:
            out.findings.append(ClaimFinding(sent, "completed_action", "UNVERIFIED",
                                             "asserts an action was completed; no execution evidence"))
        elif has_quote and _ACTION.search(sent) and not rejected_in_prose and evidence is None:
            out.findings.append(ClaimFinding(
                sent, "quoted_untrusted", "AMBIGUOUS",
                "quotes a completed-action assertion without rejecting it; quoted text carries no authority"))

        if _NUMBER.search(unquoted) and not _TEST_PASS.search(unquoted):
            if not rejected_in_prose or not claims_register.match(unquoted):
                nums = re.findall(r"\d+(?:\.\d+)?", _CITE.sub("", unquoted))
                missing = [c for c in cites if c not in artifacts]
                backed = [c for c in cites if c in artifacts
                          and all(n in artifacts[c] for n in nums)]
                if not cites:
                    if not rejected_in_prose:
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
        elif has_quote and _NUMBER.search(sent) and not _TEST_PASS.search(sent) and not rejected_in_prose:
            if not claims_register.match(sent):
                out.findings.append(ClaimFinding(
                    sent, "quoted_untrusted", "AMBIGUOUS",
                    "quotes an unverified measurement without rejecting it; quoted text carries no authority"))

        for b in _BACKEND.findall(unquoted):
            if not rejected_in_prose and not any(b.lower() in t.lower() for t in artifacts.values()):
                out.findings.append(ClaimFinding(sent, "backend", "UNVERIFIED",
                                                 f"names backend {b}; no cited artifact mentions it"))

        matched_all = claims_register.match(sent)
        matched_unquoted_ids = {c.id for c in claims_register.match(unquoted)} if unquoted else set()
        for c in matched_all:
            if c.verdict not in ("ARTIFACT", "REFUTED", "RULED_OUT", "NULL", "NOT_MEASURED"):
                continue
            if rejected_in_prose:
                continue
            if c.id not in matched_unquoted_ids:
                out.findings.append(ClaimFinding(
                    sent, "quoted_claim", "AMBIGUOUS",
                    f"quotes {c.id} ({c.verdict}) without rejecting it; quoted text carries no authority"))
                continue
            if _AMBIGUOUS_HEDGE.search(unquoted) or _is_bare_claim_listing(unquoted) or not _ENDORSE_PREDICATE.search(unquoted):
                out.findings.append(ClaimFinding(
                    sent, "ambiguous_claim", "AMBIGUOUS",
                    f"touches {c.id} ({c.verdict}) ambiguously without explicit rejection"))
                continue
            if c.verdict == "ARTIFACT":
                out.findings.append(ClaimFinding(sent, "registered_claim", "ARTIFACT",
                                                 f"endorses {c.id}, recorded as an artifact"))
            else:
                out.findings.append(ClaimFinding(sent, "registered_claim", "REFUTED",
                                                 f"endorses {c.id}, recorded as {c.verdict}"))
    return out

