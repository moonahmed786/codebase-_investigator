"""
Independent audit agent.

This agent runs in a completely separate API call and context from the
investigation agent. It never sees the investigator's reasoning — only:
  - The final answer text
  - Programmatically-verified citation validity (what the files actually say)
  - The conversation history (to catch contradictions)

The prompt explicitly instructs the model not to give the investigator
the benefit of the doubt.
"""
import json
import re
from typing import List, Optional

import anthropic

from models.schemas import AuditFlag, AuditResult, CitationCheck

AUDIT_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 2048


def _build_citation_block(checks: List[CitationCheck]) -> str:
    if not checks:
        return "No citations detected in the answer."
    lines = []
    for c in checks:
        status = "VALID" if c.valid else f"INVALID ({c.reason})"
        lines.append(f"[{c.raw}] → {status}")
        if c.valid and c.snippet:
            # Show first 6 lines of the actual snippet
            snippet_lines = c.snippet.splitlines()[:6]
            lines.append("  Actual content:")
            for sl in snippet_lines:
                lines.append(f"    {sl}")
    return "\n".join(lines)


def _audit_system() -> str:
    return """\
You are an independent code-review auditor. Your job is to audit the quality of \
an AI assistant's answer about a codebase. You have no affiliation with that \
assistant and should NOT give it the benefit of the doubt.

You will be given:
1. The answer to audit
2. Programmatic citation checks — whether each cited file:line actually exists and \
   what the code at that location really says
3. The prior conversation turns (to spot contradictions)

Audit for:
- Citation accuracy: does the cited code actually support the claim?
- Hallucinated details: facts stated without evidence, or that contradict the files
- Contradictions: does this answer conflict with something said in a prior turn?
- Over-confidence: strong claims without adequate exploration (e.g. "there is no X" \
  without searching for X)
- Dangerous suggestions: would any proposed change break other code?
- Coverage gaps: important aspects of the question that were not investigated

Return ONLY a JSON object with this exact shape (no markdown, no preamble):
{
  "verdict": "trustworthy" | "caution" | "unreliable",
  "flags": [
    {"severity": "error"|"warning"|"info", "text": "..."}
  ],
  "contradictions": ["..."],
  "missing_context": ["..."],
  "summary": "1-2 sentence plain-English verdict"
}

verdict guide:
- "trustworthy": citations are valid and support the claims, no contradictions, \
  reasoning is sound
- "caution": minor unverified claims, some extrapolation, or small gaps — still \
  largely useful
- "unreliable": invalid citations, contradictions, unsupported strong claims, or \
  dangerous suggestions
"""


async def run_audit(
    answer: str,
    citation_checks: List[CitationCheck],
    prior_conversation: str,
    anthropic_client: anthropic.AsyncAnthropic,
) -> AuditResult:
    citation_block = _build_citation_block(citation_checks)

    user_content = f"""\
ANSWER TO AUDIT:
---
{answer}
---

PROGRAMMATIC CITATION CHECKS:
{citation_block}

PRIOR CONVERSATION (all earlier turns):
{prior_conversation}

Audit the answer now. Return only the JSON object.
"""

    response = await anthropic_client.messages.create(
        model=AUDIT_MODEL,
        max_tokens=MAX_TOKENS,
        system=_audit_system(),
        messages=[{"role": "user", "content": user_content}],
    )

    raw = response.content[0].text.strip()
    return _parse_audit(raw, citation_checks)


def _parse_audit(raw: str, citation_checks: List[CitationCheck]) -> AuditResult:
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: extract what we can
        return AuditResult(
            verdict="caution",
            citation_checks=citation_checks,
            flags=[AuditFlag(severity="warning", text="Audit response could not be parsed as JSON.")],
            contradictions=[],
            missing_context=[],
            summary="Audit parsing failed — treat with caution.",
        )

    flags = [
        AuditFlag(severity=f.get("severity", "info"), text=f.get("text", ""))
        for f in data.get("flags", [])
    ]
    verdict = data.get("verdict", "caution")
    if verdict not in ("trustworthy", "caution", "unreliable"):
        verdict = "caution"

    # Auto-downgrade verdict if any citations are invalid and auditor didn't catch it
    invalid_citations = [c for c in citation_checks if not c.valid]
    if invalid_citations and verdict == "trustworthy":
        verdict = "caution"
        flags.append(AuditFlag(
            severity="error",
            text=f"{len(invalid_citations)} citation(s) failed programmatic validation."
        ))

    return AuditResult(
        verdict=verdict,
        citation_checks=citation_checks,
        flags=flags,
        contradictions=data.get("contradictions", []),
        missing_context=data.get("missing_context", []),
        summary=data.get("summary", ""),
    )
