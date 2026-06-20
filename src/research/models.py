from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field


class FactKind(str, Enum):
    """Why a fact was retrieved — tracks the pipeline pass that produced it."""

    BASE_RATE = "base_rate"  # analogous past situations / reference-class data (Exa)
    CURRENT = "current"  # fresh dated facts about the present situation (Brave)
    DISCONFIRMING = "disconfirming"  # evidence against the leading hypothesis


class Provider(str, Enum):
    EXA = "exa"
    BRAVE = "brave"


class RetrievedFact(BaseModel):
    """A single fact pulled from a provider. FACTS ONLY — no opinions, no synthesis.

    Every fact is stamped with its source and (when available) its date.
    """

    text: str
    source_url: str = ""
    source_name: str = ""
    published_date: date | None = None
    retrieved_at: datetime
    provider: Provider
    originating_query: str
    kind: FactKind
    # Set when a current-state fact has no date — caller should treat as weaker.
    low_confidence: bool = False

    def dedupe_key(self) -> tuple[str, str]:
        """URL + a content fingerprint, used to drop near-duplicates."""
        return (self.source_url.strip().lower(), self.text.strip().lower()[:120])


class ResearchBriefing(BaseModel):
    """A bundle of facts handed to the BLIND forecast stage.

    Contains facts only: no probabilities, no recommendation, no synthesis, and
    crucially no market price. Each rendered line carries its source and date.
    """

    question: str
    sub_questions: list[str] = Field(default_factory=list)
    facts: list[RetrievedFact] = Field(default_factory=list)

    def facts_of(self, kind: FactKind) -> list[RetrievedFact]:
        return [f for f in self.facts if f.kind == kind]

    def render(self) -> str:
        """Render as a plain-text briefing for the forecaster's prompt."""
        lines: list[str] = [
            "RESEARCH BRIEFING — facts only. No market prices, no conclusions.",
            f"Question: {self.question}",
        ]
        if self.sub_questions:
            lines.append("Research sub-questions:")
            lines.extend(f"  - {q}" for q in self.sub_questions)

        sections = [
            ("REFERENCE CLASS / BASE RATES (analogous past situations)", FactKind.BASE_RATE),
            ("CURRENT STATE (fresh dated facts)", FactKind.CURRENT),
            ("DISCONFIRMING EVIDENCE (against the leading hypothesis)", FactKind.DISCONFIRMING),
        ]
        for header, kind in sections:
            facts = self.facts_of(kind)
            lines.append("")
            lines.append(f"{header}:")
            if not facts:
                lines.append("  (none found)")
                continue
            for f in facts:
                lines.append(f"  {_render_fact(f)}")

        return "\n".join(lines)


def _render_fact(f: RetrievedFact) -> str:
    if f.published_date is not None:
        date_str = f.published_date.isoformat()
    elif f.kind == FactKind.CURRENT:
        date_str = "undated, low-confidence"
    else:
        date_str = "undated"
    source = f.source_name or f.source_url or f.provider.value
    return (
        f"[{f.provider.value}] ({date_str}) {f.text}\n"
        f"      source: {source} <{f.source_url}>\n"
        f"      found via: \"{f.originating_query}\""
    )
