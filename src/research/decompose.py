from __future__ import annotations

import json
import logging

import anthropic

logger = logging.getLogger(__name__)


DECOMPOSITION_SYSTEM_PROMPT = """\
You break a prediction-market question into specific, independently answerable
research sub-questions that would help someone estimate how the market will
resolve, from first principles.

Produce 2-4 sub-questions. Aim for a mix of:
- REFERENCE-CLASS questions: "how often do things of this sort happen?" — these
  let us look up base rates from analogous past situations.
- CURRENT-STATE questions: concrete, datable facts about the present situation
  that would move the estimate.
- RESOLUTION questions (only when the wording is non-trivial or ambiguous): how
  similar markets or criteria have resolved before, or what exactly the
  resolution source tracks — to surface edge-resolution or ambiguity risk.

Rules:
- Each sub-question must be specific and searchable on its own (a web search or
  semantic search should return useful results). Avoid vague or compound questions.
- Do NOT ask about market prices, betting odds, implied probabilities, or "what
  the market thinks." We are forming an independent estimate, not reading a price.
- Do NOT answer the questions. Only produce the questions.
"""

DECOMPOSITION_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "sub_questions": {
            "type": "array",
            "description": "2-4 specific, independently answerable research sub-questions.",
            "items": {"type": "string"},
        }
    },
    "required": ["sub_questions"],
    "additionalProperties": False,
}


class QuestionDecomposer:
    """Turns a market question into research sub-questions via a cheap LLM call.

    Deliberately separate from the forecaster so retrieval stays split from
    judgment. Never sees or emits a market price.
    """

    def __init__(self, client: anthropic.AsyncAnthropic, model: str) -> None:
        self._client = client
        self._model = model

    async def decompose(
        self,
        question: str,
        resolution_criteria: str = "",
        end_date: str = "",
    ) -> list[str]:
        user_parts = [f"Market question: {question}"]
        if resolution_criteria.strip():
            user_parts.append(f"Resolution criteria: {resolution_criteria.strip()[:1500]}")
        if end_date:
            user_parts.append(f"Resolution date: {end_date}")
        user_parts.append(
            "Return 2-4 research sub-questions that would most help estimate this."
        )
        user_prompt = "\n\n".join(user_parts)

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=DECOMPOSITION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            output_config={
                "format": {"type": "json_schema", "schema": DECOMPOSITION_JSON_SCHEMA}
            },
        )
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise RuntimeError("Decomposition returned no text block")

        data = json.loads(text)
        subs = [s.strip() for s in data.get("sub_questions", []) if s and s.strip()]
        # Keep it tight: 2-4 sub-questions.
        subs = subs[:4]
        logger.info("Decomposed into %d sub-questions", len(subs))
        return subs
