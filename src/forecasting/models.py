from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from src.news.models import Article


class Recommendation(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    AVOID = "AVOID"


class Confidence(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


# Maps the self-reported confidence label onto the DB's REAL column.
CONFIDENCE_TO_FLOAT: dict[Confidence, float] = {
    Confidence.LOW: 0.33,
    Confidence.MEDIUM: 0.67,
    Confidence.HIGH: 1.0,
}


class OutcomeProbability(BaseModel):
    """A single outcome's probability — part of the structured-output schema."""

    outcome: str
    probability: float


class ForecastOutput(BaseModel):
    """Schema Claude fills via structured outputs (output_config.format).

    Replaces the old free-text PROBABILITIES block — probabilities are now
    guaranteed one-per-outcome and machine-parseable.
    """

    briefing: str
    confidence: Confidence
    key_assumption: str
    probabilities: list[OutcomeProbability]


# Hand-written JSON Schema sent to the API. Kept in sync with ForecastOutput.
# Note: numeric/string constraints (minimum/maximum/minLength) are unsupported
# by structured outputs, so they're omitted here.
FORECAST_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "briefing": {
            "type": "string",
            "description": (
                "A 300-500 word briefing: one-line summary and lean, base-rate "
                "context, the 2-3 most important drivers, key risk factors, and "
                "the bottom-line probability with reasoning."
            ),
        },
        "confidence": {
            "type": "string",
            "enum": ["Low", "Medium", "High"],
            "description": (
                "Confidence in the PRECISION of your estimate, not the probability "
                "itself. Low = the true probability could be 15+ points away; "
                "High = you'd be surprised if it were more than 5 points off."
            ),
        },
        "key_assumption": {
            "type": "string",
            "description": "The single assumption that, if wrong, would most change your estimate.",
        },
        "probabilities": {
            "type": "array",
            "description": "Exactly one entry per possible outcome. Probabilities must sum to 1.0.",
            "items": {
                "type": "object",
                "properties": {
                    "outcome": {
                        "type": "string",
                        "description": "The exact outcome name as given in the prompt.",
                    },
                    "probability": {
                        "type": "number",
                        "description": "Probability for this outcome, between 0 and 1.",
                    },
                },
                "required": ["outcome", "probability"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["briefing", "confidence", "key_assumption", "probabilities"],
    "additionalProperties": False,
}


class OutcomeForecast(BaseModel):
    outcome: str
    bot_probability: float
    market_probability: float
    ev_per_dollar: float
    kelly_fraction: float
    recommendation: Recommendation
    has_market_price: bool = True


class ForecastResult(BaseModel):
    condition_id: str
    question: str
    slug: str
    reasoning: str
    outcomes: list[OutcomeForecast]
    confidence: float = 0.0  # self-reported 0-1 (mapped from Confidence label)
    confidence_label: str = ""
    key_assumption: str = ""
    prompt_version: str = ""
    news_article_count: int = 0
    articles: list[Article] = Field(default_factory=list)

    @property
    def best_opportunity(self) -> OutcomeForecast | None:
        """Return the outcome with the highest positive EV, if any."""
        positive = [o for o in self.outcomes if o.ev_per_dollar > 0]
        if not positive:
            return None
        return max(positive, key=lambda o: o.ev_per_dollar)
