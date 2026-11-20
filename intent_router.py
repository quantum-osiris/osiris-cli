"""
intent_router.py -- what free-form operator text is FOR, decided by code.

Before this, every non-command message went to the code-generation
pipeline: on 2026-09-24 "hello osiris" produced an invented OSIRISSystems
class ("linear algebra, calculus") saved as osiris_organism_*.py. Now each
message gets one fixed class from word cues, and each class has a fixed,
code-defined action list. No model chooses the class or the actions; a
model only ever answers, plans or drafts inside an action the operator
picked (or, for read-only answers, one this module marks auto_answer).

  conversation  greetings, thanks, chit-chat          -> answered at once (read-only)
  question      why/what/how/?, "explain", "show me"  -> answered at once (read-only)
  status        "status", "what are you doing"        -> answered at once (read-only)
  benchmark     evaluate/benchmark/compare models     -> bench or mentor evidence
  research      quantum/experiment/hypothesis/...     -> experiment brief, design, arXiv
  engineering   build/add/fix/make/implement/...      -> gap draft, plan, or engines
  session_log   a pasted terminal session             -> read-only digest of what happened
                (paste_digest.looks_like_session: prompts, Termux banner, panels)

A single token of 1-3 characters that is no greeting ("l3", "abc", "??")
returns UNCLEAR, which is not a class: it gets "did you mean", no answer.

Ties: research > engineering > benchmark > status > question > conversation.
"build an experiment" is research; "can you add X to the bench?" is a build
request (engineering), not a benchmark run; a question ABOUT a research or
benchmark topic ("what is dynamical decoupling?") stays a question. Text
with no cues is answered (a short line is conversation).

assess(text) adds, by the same kind of fixed rule, what a wrong answer to the
request would cost: a cost class (read_only, workspace_write, external_action,
hardware_action) and a risk tier (LOW, MEDIUM, HIGH, CRITICAL). See RUBRIC.
"""
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import paste_digest

CLASSES = ("conversation", "question", "status", "benchmark", "research", "engineering", "session_log")
UNCLEAR = "unclear"  # returned by classify(), deliberately not a class: nothing is answered or offered

_CUES = {
    "benchmark": r"\b(bench(mark)?s?|evaluate|evaluation|score|scorecard|mentors?|which model|compare (the )?models?|"
                 r"qwen|deepseek|llama|gemini)\b",
    "research": r"\b(quantum|qubits?|circuits?|experiments?|hypothes[ie]s|simulat(or|ion|e)|noise|mitigation|"
                r"decoherence|fidelity|entangle\w*|hamiltonian|vqe|qaoa|aer|qiskit|dynamical decoupling|"
                r"research|arxiv|paper)\b",
    "engineering": r"\b(build|add|fix|make|implement|create|write|refactor|change|improve|enhance|support|"
                   r"modify|patch|update|remove|rename)\b",
    "status": r"\b(status|what are you doing|what('s| is) (osiris )?doing|what('s| is) (pending|running|next)|"
              r"where are we|progress|state of)\b",
    "question": r"(\?\s*$|^\s*(why|what|how|when|where|which|who|is|are|does|do|did|can|could|should|explain|"
                r"tell me|show me|describe)\b)",
    "conversation": r"^\s*(hi|hello|hey|yo|thanks|thank you|good (morning|afternoon|evening|night)|gm|ok|okay|"
                    r"cool|nice|great)\b",
}
_ORDER = ("research", "engineering", "benchmark", "status", "question", "conversation")

# Fixed actions per class: (label, command). Commands are REPL commands owned
# by code; {text} is the operator's own words (never model output).
ACTIONS = {
    "conversation": [("Ask a follow-up (just type it)", None), ("Full status", "/status"),
                     ("Capture a capability gap", "/gap"), ("Draft an experiment brief", "/experiment")],
    "question": [("Full status (every fact)", "/status"), ("Mentor evidence", "/mentors"),
                 ("Capture it as a capability gap", "/gap {text}")],
    "status": [("Full status (every fact)", "/status"), ("Check everything", "/check"),
               ("Mentor evidence", "/mentors")],
    "benchmark": [("Benchmark qwen2.5-coder:7b on 3 tasks (local)", "/bench"),
                  ("Mentor evidence so far", "/mentors"), ("Ask OSIRIS about it", "/ask {text}")],
    "research": [("Draft an experiment brief (guided, simulator first)", "/experiment {text}"),
                 ("Ask the Architect for an experiment design (text only)", "/plan {text}"),
                 ("Fetch recent arXiv abstracts on it", "/research {text}")],
    "engineering": [("Capture it as a capability gap (guided draft)", "/gap {text}"),
                    ("Ask the Architect for a plan (text only, no files)", "/plan {text}"),
                    ("Send to the engines (proposes a file you /apply)", "/engage")],
    "session_log": [("What happened, again: commands, results, problems", "/digest"),
                    ("Full status now, to compare with the log", "/status"),
                    ("Capture a capability gap (guided draft)", "/gap")],
}
AUTO_ANSWER = {"conversation", "question", "status"}

# Questions about OSIRIS ITSELF are answered by code from verified state, never
# by a model: on 2026-09-24 both llama3.2:1b and qwen2.5-coder:7b, given the
# facts, still asserted things they did not say ("has been benchmarked", "the
# best model is deepseek-coder"). Models answer only knowledge questions.
_ABOUT_OSIRIS = re.compile(
    r"\b(osiris|you|your|yourself|gaps?|stor(y|ies)|sprints?|backlog|bench(mark)?s?|mentors?|scorecard|pending|"
    r"apply|ledger|runs?|status|next|genome|generation|briefs?|which model|best model|models? (is|are) best|"
    r"what can (you|i) do|help)\b", re.IGNORECASE)


# OSIRIS's own terms: their meaning here, and the data behind them, come from OSIRIS.
_OSIRIS_TERMS = re.compile(r"\b(false confidence|hidden tests?|own tests?|integrity exclusions?)\b", re.IGNORECASE)
# "why did it fail?" -- a pronoun plus an outcome verb asks about something that just happened here.
_PRONOUN = re.compile(r"\b(it|that|this|they|them)\b", re.IGNORECASE)
_OUTCOME = re.compile(r"\b(fail\w*|stop\w*|pass\w*|break\w*|broke|happen\w*|go wrong|went wrong|score\w*)\b",
                      re.IGNORECASE)


def about_osiris(text):
    """True when the message asks about OSIRIS's own state, work, models or
    terms -- including "why did it fail?" right after a round."""
    return bool(_ABOUT_OSIRIS.search(text) or _OSIRIS_TERMS.search(text)
                or (_PRONOUN.search(text) and _OUTCOME.search(text)))


def classify(text):
    """(class, {class: [matched cues]}). Deterministic; no model.

    A long pasted block (more than 5 lines) is classified by its FIRST line --
    the operator's framing -- not by every word in it: on 2026-09-24 a pasted
    462-line design note on ledger reconciliation routed as research because
    its body mentioned "dynamical decoupling" and "experiment" in passing. A
    research, benchmark or build first line keeps that class; anything else
    (a question, a title) makes the block material to work on: engineering."""
    if paste_digest.looks_like_session(text):
        # A pasted terminal session is a record of what happened, not a request:
        # on-device a paste opening "Welcome to Termux" became an engineering
        # goal of that name (2026-09-24).
        return "session_log", {"session_log": ["pasted terminal session"]}
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if len(lines) > 5:
        head, head_hits = classify(lines[0])
        if head in ("research", "benchmark", "engineering"):
            return head, {head: head_hits.get(head, []) + ["first line of a pasted block"]}
        return "engineering", {"engineering": [f"pasted block of {len(lines)} lines"]}
    flat = " ".join(text.split()).lower()
    hits = {}
    for cls, pattern in _CUES.items():
        found = [m.group(0).strip() for m in re.finditer(pattern, flat)]
        if found:
            hits[cls] = sorted(set(found))[:4]
    if re.fullmatch(r"\S{1,3}", flat) and "conversation" not in hits:
        # "l3" got a full status answer on-device: a stray fragment is no request.
        return UNCLEAR, {UNCLEAR: [f"'{flat}' is no command, menu key or sentence"]}
    if "\n" in text.strip() and len(text.strip().splitlines()) > 3:
        # A pasted block (log, traceback, notes) is material to work on, not a chat line.
        hits.setdefault("engineering", ["pasted block"])
    if "conversation" in hits and len(flat.split()) <= 5 and not ({"engineering", "research", "benchmark"} & set(hits)):
        return "conversation", hits  # "hey osiris, you there?" is a greeting, not a question
    for cls in _ORDER:
        if cls in hits:
            # A question ABOUT a topic stays a question unless it asks for work.
            if cls in ("benchmark", "research") and "question" in hits and "engineering" not in hits \
                    and not re.search(r"\b(run|start|do|design|draft|plan|compare)\b", flat) \
                    and not re.search(r"\b(benchmark|evaluate|score|test) (them|it|all|the models?|"
                                      r"qwen|deepseek|llama|gemini)\b", flat):
                return "question", hits
            return cls, hits
    return "conversation" if len(flat.split()) <= 3 else "question", hits


def actions_for(cls, text):
    """The fixed actions for a class, with the operator's words filled in."""
    words = " ".join(text.split())[:240]
    return [(label, cmd.replace("{text}", words) if cmd else None) for label, cmd in ACTIONS[cls]]


# ---------------------------------------------------------------- risk and cost
#
# What a wrong answer to a request would cost, decided by code from the same
# word cues that decide the class. No model reads the text and no model can
# change the result. The assessment grants nothing: it is recorded with the
# task (osiris_cli/task_context.py) and sets how short a leash the task gets.

RISK_TIERS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
COST_CLASSES = ("read_only", "workspace_write", "external_action", "hardware_action")

# The rubric: for each cost class, the tier it starts at, what is at stake, and
# whether it can be undone. A request's class is the most costly one it touches.
RUBRIC = {
    "read_only": {
        "tier": "LOW", "reversible": True,
        "at_stake": "a wrong or unhelpful answer on screen; nothing is written or run",
        "undo": "ask again"},
    "workspace_write": {
        "tier": "MEDIUM", "reversible": True,
        "at_stake": "a wrong change proposed to local files or OSIRIS's own local records; "
                    "it is written only after you approve it",
        "undo": "/discard before /apply; git afterwards"},
    "external_action": {
        "tier": "HIGH", "reversible": False,
        "at_stake": "something leaves this machine -- a push, merge, release, publication, upload, "
                    "deployment or message -- where others can see or depend on it",
        "undo": "may not be undoable once it has left"},
    "hardware_action": {
        "tier": "CRITICAL", "reversible": False,
        "at_stake": "a device or quantum backend acts: QPU time is spent, or a device is flashed, "
                    "wiped or provisioned",
        "undo": "cannot be undone"},
}

# What each class leads to before any cue is read. Answer-only classes run
# nothing; the others lead to local drafts, evidence or proposed changes.
_CLASS_COST = {"conversation": "read_only", "question": "read_only", "status": "read_only",
               "session_log": "read_only", UNCLEAR: "read_only",
               "benchmark": "workspace_write", "research": "workspace_write", "engineering": "workspace_write"}
_ANSWER_ONLY = {"conversation", "question", "status", "session_log", UNCLEAR}

_HW_TARGET = (r"(qpu|quantum (hardware|computer|processor|device)|real (hardware|device|backend|qpu)|"
              r"hardware backend|ibm[ _-]?(quantum|q|cloud)?|ibm_[a-z]+|heron|eagle|ionq|rigetti|quantinuum|"
              r"braket)")
_HW_VERB = r"(submit|submits|submitting|run|runs|running|execute|executing|send|sending|launch|queue|dispatch)"
_DEVICE = r"(device|phone|tablet|handset|usb|disk|drive|partition|firmware|bootloader|rom|ffu|image|sd card)"
_EXT_OBJECT = (r"(github|gitlab|remote|origin|upstream|main|master|branch|pr|pull request|zenodo|pypi|npm|"
               r"production|prod|cloud run|vercel|server|registry|release|tag|website|site|bucket|doi)")

# (cost class, tier, rule name, pattern). Patterns run on lower-cased, single-spaced text.
_COST_RULES = (
    ("hardware_action", "CRITICAL", "qpu_submission",
     rf"\b{_HW_VERB}\b[^?!]{{0,80}}\b(on|to|against|at)\b[^?!]{{0,30}}\b{_HW_TARGET}\b"),
    ("hardware_action", "CRITICAL", "qpu_job",
     rf"\b{_HW_VERB}\b[^?!]{{0,40}}\b(qpu|ibm|hardware) (job|jobs|run|runs|submission|experiment)s?\b"),
    ("hardware_action", "CRITICAL", "device_flash",
     rf"\b(flash|reflash|wipe|factory[ -]reset|repartition|provision|unlock the bootloader|root)\b"
     rf"[^?!]{{0,60}}\b{_DEVICE}\b"),
    ("external_action", "HIGH", "git_push",
     r"\b(git push|force[ -]push|push --force|push (it |this |that |them |the \w+ )?(to|up)\b|"
     r"push (the |my |this )?(branch|commit|commits|changes|tag|tags|fix)\b)"),
    ("external_action", "HIGH", "merge_or_pr",
     r"\b(merge (the |this |that |my )?(pr|pull request|branch)|merge (it |this )?(in)?to (main|master)|"
     r"(open|create|file|submit) (a |the )?(pr|pull request))\b"),
    ("external_action", "HIGH", "publish_release",
     rf"\b(publish|release|deploy|upload|ship|roll out|rollout)\b[^?!]{{0,60}}\b(to|on|onto|at|as)\b"
     rf"[^?!]{{0,30}}\b{_EXT_OBJECT}\b"),
    ("external_action", "HIGH", "publish_release",
     r"\b((create|cut|make|draft|tag) (a |the )?(github )?release|publish (it|this|the \w+)|"
     r"deploy (it|this|the \w+)|mint (a |the )?doi)\b"),
    ("external_action", "HIGH", "send_message",
     r"\b(send|post|email|e-mail|mail|tweet|dm)\b[^?!]{0,40}\b(email|e-mail|message|slack|discord|"
     r"twitter|linkedin|reviewers?|the team|them|him|her|everyone)\b"),
)

# Tier-only rules: the request stays in its cost class, but a wrong answer costs more.
_TIER_RULES = (
    ("HIGH", "destructive_change",
     r"\b(delete|deletes|erase|wipe|purge|truncate|drop (the |a )?(table|database|ledger)|rm -rf|"
     r"reset --hard|overwrite|force[ -]overwrite|remove all|clean out)\b"),
    ("HIGH", "keys_or_secrets",
     r"\b(private key|signing key|operator key|enclave|trusted_operator_keys|credentials?|api[ _-]?key|"
     r"secret|secrets|passwords?|(api|access|auth|bearer|refresh)[ _-]tokens?|\.env)\b"),
    ("HIGH", "ledger_rewrite",
     r"\b(rewrite|re-?hash|repair|edit|rebuild|reorder|backfill|amend)\b[^?!]{0,40}\b(ledger|hash chain|"
     r"receipts?|exchange log|evidence)\b"),
    ("HIGH", "self_modify", r"\b(--allow-self-modify|modify yourself|rewrite yourself|change your own (code|gate|rules))\b"),
)


@dataclass(frozen=True)
class RiskAssessment:
    """What a wrong answer to one request would cost. Built only by assess()."""
    cls: str                     # the intent class (classify)
    cost_class: str              # one of COST_CLASSES
    tier: str                    # one of RISK_TIERS
    reversible: bool
    at_stake: str                # the rubric's line for the cost class
    reasons: Tuple[str, ...]     # every rule that fired, with the words that fired it

    def line(self) -> str:
        return (f"risk {self.tier} · {self.cost_class.replace('_', ' ')} · "
                f"{'can be undone' if self.reversible else 'cannot be undone'} · if wrong: {self.at_stake}")


def _request_text(text):
    """The words that are the operator's request. For a pasted block of more
    than 5 lines that is its first line, as in classify(): what a pasted
    document mentions in passing is not something the operator asked for."""
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    head = lines[0] if len(lines) > 5 else text
    return " ".join(head.split()).lower(), len(lines) > 5


def assess(text, cls: Optional[str] = None) -> RiskAssessment:
    """(cost class, risk tier) for a request, with every rule that fired.

    Deterministic; no model. The cost class starts from what the intent class
    leads to, and each cost rule that matches raises it; the tier is the
    highest of the class's rubric tier and any tier rule that matches.

    Something that only asks -- a greeting, a status request, a pasted session,
    or a question with a question cue ("how do I push to main?") -- runs
    nothing, so its cost class stays read_only whatever it mentions; but an
    answer about an action that cannot be undone is one you may act on, so its
    tier is MEDIUM rather than LOW.

    classify() also calls any longer text with no cue a question. When such
    text has no question cue and names an action ("submit the job to the
    QPU"), it is an instruction, and it is assessed as a request for that
    action even though OSIRIS has no path that performs it: the cost of a
    wrong answer there is a reply that claims or encourages the action.

    Text that cannot be assessed (not a string, or empty) and an unknown class
    get the highest tier rather than being waved through."""
    if not isinstance(text, str) or not text.strip():
        return RiskAssessment(UNCLEAR, "hardware_action", "CRITICAL", False, RUBRIC["hardware_action"]["at_stake"],
                              ("nothing to assess: treated as the most costly class, and nothing is run",))
    cls = cls or classify(text)[0]
    if cls not in _CLASS_COST:
        return RiskAssessment(cls, "hardware_action", "CRITICAL", False, RUBRIC["hardware_action"]["at_stake"],
                              (f"unknown intent class {cls!r}: treated as the most costly class",))
    flat, pasted_block = _request_text(text)
    instruction = cls == "question" and not re.search(_CUES["question"], flat)
    asks_only = cls in _ANSWER_ONLY and not instruction
    cost, reasons = _CLASS_COST[cls], [f"class {cls}: {_CLASS_COST[cls].replace('_', ' ')}"]
    if pasted_block:
        reasons.append("pasted block: assessed by its first line only")
    cost_hits = [(rule_cost, name, m.group(0)) for rule_cost, _tier, name, pattern in _COST_RULES
                 for m in [re.search(pattern, flat)] if m]
    tier_hits = [(rule_tier, name, m.group(0)) for rule_tier, name, pattern in _TIER_RULES
                 for m in [re.search(pattern, flat)] if m]
    if asks_only:
        tier = RUBRIC[cost]["tier"]
        for rule_cost, name, words in cost_hits:
            reasons.append(f"{name}: asks about '{words}' ({rule_cost.replace('_', ' ')}); nothing is run")
        for _rule_tier, name, words in tier_hits:
            reasons.append(f"{name}: asks about '{words}'; nothing is run")
        if cost_hits or tier_hits:
            tier = "MEDIUM"
    else:
        if instruction and (cost_hits or tier_hits):
            cost = "workspace_write"
            reasons.append("no question cue: an instruction, assessed as a request for the action it names "
                           "(OSIRIS answers it; it has no path that performs it)")
        for rule_cost, name, words in cost_hits:
            reasons.append(f"{name}: '{words}' ({rule_cost.replace('_', ' ')})")
            if COST_CLASSES.index(rule_cost) > COST_CLASSES.index(cost):
                cost = rule_cost
        tier = RUBRIC[cost]["tier"]
        for rule_tier, name, words in tier_hits:
            reasons.append(f"{name}: '{words}'")
            if RISK_TIERS.index(rule_tier) > RISK_TIERS.index(tier):
                tier = rule_tier
    rubric = RUBRIC[cost]
    return RiskAssessment(cls, cost, tier, rubric["reversible"], rubric["at_stake"], tuple(reasons))
