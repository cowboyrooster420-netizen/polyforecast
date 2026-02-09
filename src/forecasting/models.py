from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class Recommendation(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    AVOID = "AVOID"


class OutcomeForecast(BaseModel):
    outcome: str
    bot_probability: float
    market_probability: float
    ev_per_dollar: float
    kelly_fraction: float
    recommendation: Recommendation


class ForecastResult(BaseModel):
    condition_id: str
    question: str
    slug: str
    reasoning: str
    outcomes: list[OutcomeForecast]
    confidence: float = 0.0  # self-reported 0-1
    prompt_version: str = ""
    news_article_count: int = 0

    @property
    def best_opportunity(self) -> OutcomeForecast | None:
        """Return the outcome with the highest positive EV, if any."""
        positive = [o for o in self.outcomes if o.ev_per_dollar > 0]
        if not positive:
            return None
        return max(positive, key=lambda o: o.ev_per_dollar)
