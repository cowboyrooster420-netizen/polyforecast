from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import anthropic

from src.config import Settings
from src.forecasting.ev_calculator import evaluate_outcome
from src.forecasting.models import ForecastResult, OutcomeForecast
from src.forecasting.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from src.news.client import NewsClient
from src.news.models import Article
from src.polymarket.client import PolymarketClient
from src.polymarket.models import Market
from src.utils.rate_limiter import AsyncTokenBucket

logger = logging.getLogger(__name__)


def _format_articles(articles: list[Article]) -> str:
    parts: list[str] = []
    for i, art in enumerate(articles, 1):
        date_str = (
            art.published_at.strftime("%Y-%m-%d") if art.published_at else "unknown"
        )
        parts.append(
            f"{i}. [{art.source}] {art.title} ({date_str})\n   {art.description}"
        )
    return "\n\n".join(parts)


def _parse_probabilities(text: str, outcomes: list[str]) -> dict[str, float]:
    """Extract outcome probabilities from Claude's response."""
    probs: dict[str, float] = {}

    # Look for the PROBABILITIES: block
    prob_section = text.split("PROBABILITIES:")[-1] if "PROBABILITIES:" in text else text

    for outcome in outcomes:
        # Match "Outcome: 0.XX" or "Outcome: .XX"
        pattern = re.compile(
            rf"{re.escape(outcome)}\s*:\s*(0?\.\d+|1\.0+|0+\.0+|1)",
            re.IGNORECASE,
        )
        match = pattern.search(prob_section)
        if match:
            probs[outcome] = float(match.group(1))

    # Normalise so they sum to 1.0 if close
    total = sum(probs.values())
    if probs and 0.9 < total < 1.1 and total != 1.0:
        probs = {k: v / total for k, v in probs.items()}

    return probs


class ForecastingEngine:
    def __init__(
        self,
        settings: Settings,
        polymarket: PolymarketClient,
        news: NewsClient,
    ) -> None:
        self._settings = settings
        self._polymarket = polymarket
        self._news = news
        self._anthropic = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._rate_limiter = AsyncTokenBucket(settings.anthropic_rpm / 60.0)

    async def analyze_market(self, market: Market) -> ForecastResult:
        """Full pipeline: fetch news → prompt Claude → parse → compute EV."""
        # 1. Fetch news articles
        articles = await self._news.fetch_articles_for_market(market.question)
        articles_text = _format_articles(articles)

        # 2. Build prompt — intentionally exclude market prices to avoid anchoring
        outcomes = [t.outcome for t in market.tokens]
        end_date_str = (
            market.end_date.strftime("%Y-%m-%d") if market.end_date else "unspecified"
        )
        today_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

        user_prompt = build_user_prompt(
            question=market.question,
            description=market.description[:2000],
            outcomes=outcomes,
            end_date=end_date_str,
            today=today_str,
            articles_text=articles_text,
        )

        # 3. Call Claude
        await self._rate_limiter.acquire()
        response = await self._anthropic.messages.create(
            model=self._settings.claude_model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        reasoning = response.content[0].text

        # 4. Parse probabilities from response
        probs = _parse_probabilities(reasoning, outcomes)

        # 5. Compute EV for each outcome by comparing against market prices
        outcome_forecasts: list[OutcomeForecast] = []
        for token in market.tokens:
            bot_prob = probs.get(token.outcome, 0.5 / len(market.tokens))
            market_prob = token.price if token.price > 0 else 0.5
            of = evaluate_outcome(token.outcome, bot_prob, market_prob)
            outcome_forecasts.append(of)

        return ForecastResult(
            condition_id=market.condition_id,
            question=market.question,
            slug=market.slug,
            reasoning=reasoning,
            outcomes=outcome_forecasts,
            prompt_version=PROMPT_VERSION,
            news_article_count=len(articles),
        )

    async def analyze_by_ref(self, ref: str) -> ForecastResult | None:
        """Convenience: resolve a URL/slug/condition_id and analyze."""
        market = await self._polymarket.get_market(ref)
        if not market:
            return None
        return await self.analyze_market(market)
