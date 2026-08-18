"""Custom guardrails actions (auto-loaded from the rails config directory).

Both output rails are deterministic Python — no LLM call:
  - soften_config_blocks: no-config-execution rail (design §"Guardrails")
  - apply_citation_rail: citation-required rail; delegates to the existing
    validator in agent/validators/citations.py
The retrieved tagmap crosses into colang as a context variable
(``tagmap_json``), set by rails_runtime.apply_output_rails.
"""
from __future__ import annotations

import json
import re

from nemoguardrails.actions import action

from agent.validators.citations import TAG_RE, validate

# Imperative vendor-CLI config lines: Cisco/Juniper/Huawei-style commands or
# config prompts. Two or more such lines inside one fenced block => config block.
_CLI_LINE = re.compile(
    r"^\s*(?:"
    r"conf(?:igure)?(?:\s+t(?:erminal)?)?\b|set\s|no\s|commit\b|interface\s|"
    r"router\s|enable\b|delete\s|request\s|mml\b|(?:ADD|MOD|SET|RMV)\s|"
    r"\S*\((?:config[^)]*)\)#|\S+#\s*\S|\S+>\s*\S"
    r")",
    re.IGNORECASE,
)
_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)

_SOFTEN_NOTE = ("> Recommendation only — not an authoritative configuration. "
                "Verify against your vendor's documentation and the cited clauses "
                "before applying anything to live equipment.\n\n")


def _looks_like_config(body: str) -> bool:
    return sum(bool(_CLI_LINE.match(l)) for l in body.splitlines()) >= 2


def soften_config_text(msg: str) -> str:
    """Prefix imperative CLI config blocks with a recommendation disclaimer."""
    def _sub(m: re.Match) -> str:
        return _SOFTEN_NOTE + m.group(0) if _looks_like_config(m.group(1)) else m.group(0)
    return _FENCE.sub(_sub, msg)


def citation_rail_text(msg: str, tagmap: dict[str, dict]) -> str:
    """Citation-required check over the final message.

    Empty tagmap = abstain/refusal path, nothing to validate. An answer with
    zero validatable citations is replaced by an abstain message naming the
    specs searched; partial failures get an explicit caution line (quotes were
    already repaired/stripped upstream by answer.ask)."""
    if not tagmap:
        return msg
    report = validate(msg, tagmap)
    if report.passed:
        return msg
    specs = sorted({c["spec"] for c in tagmap.values()})
    if not TAG_RE.search(msg) or not any(c.ok for c in report.checks):
        return ("No supporting clause could be validated for this answer in "
                f"[{', '.join(specs)}]; treat it as unverified.")
    failed = sum(not c.ok for c in report.checks)
    return (msg + f"\n\n> Caution: {failed} citation check(s) failed validation — "
            "verify the flagged claims against the cited clauses.")


@action(name="soften_config_blocks")
async def soften_config_blocks(context: dict) -> str:
    return soften_config_text(context.get("bot_message", ""))


@action(name="apply_citation_rail")
async def apply_citation_rail(context: dict) -> str:
    tagmap = json.loads(context.get("tagmap_json") or "{}")
    return citation_rail_text(context.get("bot_message", ""), tagmap)
