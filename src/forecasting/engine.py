from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import anthropic

from src.config import Settings
from src.forecasting.ev_calculator import evaluate_outcome
from src.forecasting.models import (
    CONFIDENCE_TO_FLOAT,
    FORECAST_JSON_SCHEMA,
    ForecastOutput,
    ForecastResult,
    OutcomeForecast,
)
from src.forecasting.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from src.forecasting.validators import validate_forecast
from src.news.client import NewsClient
from src.news.models import Article
from src.polymarket.client import PolymarketClient
from src.polymarket.models import Market
from src.research.models import RetrievedFact
from src.research.retriever import ResearchRetriever
from src.utils.rate_limiter import AsyncTokenBucket

logger = logging.getLogger(__name__)


def _fact_to_article(fact: RetrievedFact) -> Article:
    """Map a research fact onto the Article shape so provenance still lands in
    the news_articles table when the research layer is the context source."""
    published = (
        datetime(
            fact.published_date.year,
            fact.published_date.month,
            fact.published_date.day,
            tzinfo=timezone.utc,
        )
        if fact.published_date
        else None
    )
    return Article(
        title=fact.text[:120],
        source=fact.source_name or fact.provider.value,
        url=fact.source_url,
        published_at=published,
        description=fact.text[:500],
    )


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


def _extract_text_block(response: anthropic.types.Message) -> str:
    """Return the first text content block.

    With adaptive thinking enabled the first block is a thinking block, so we
    can't assume content[0] is text.
    """
    for block in response.content:
        if block.type == "text":
            return block.text
    raise RuntimeError("No text block in Claude response")


def _normalize_probs(probs: dict[str, float]) -> dict[str, float]:
    """Renormalize to sum to 1.0 when the model's output is close but not exact."""
    total = sum(probs.values())
    if probs and 0.9 < total < 1.1 and total != 1.0:
        return {k: v / total for k, v in probs.items()}
    return probs


class ForecastingEngine:
    def __init__(
        self,
        settings: Settings,
        polymarket: PolymarketClient,
        news: NewsClient,
        research: ResearchRetriever | None = None,
    ) -> None:
        self._settings = settings
        self._polymarket = polymarket
        self._news = news
        self._research = research
        self._anthropic = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            max_retries=settings.anthropic_max_retries,
        )
        self._rate_limiter = AsyncTokenBucket(settings.anthropic_rpm / 60.0)

    async def _gather_context(
        self, market: Market, end_date_str: str
    ) -> tuple[str, int, list[Article]]:
        """Assemble the source material for the BLIND forecast.

        Prefers the question-driven research layer (price-blind by design); falls
        back to the news pipeline if research is disabled, unconfigured, or errors.
        Returns (context_text, source_count, articles_for_db).
        """
        if (
            self._research is not None
            and self._settings.use_research_retrieval
            and self._research.enabled
        ):
            try:
                logger.info("Step 1: Research retrieval for: %s", market.question[:60])
                briefing = await self._research.research(
                    question=market.question,
                    resolution_criteria=market.description[:2000],
                    end_date=end_date_str,
                )
                facts = briefing.facts
                logger.info("Step 1 done: %d research facts", len(facts))
                return briefing.render(), len(facts), [_fact_to_article(f) for f in facts]
            except Exception as exc:
                logger.warning("Research retrieval failed (%s); falling back to news", exc)

        logger.info("Step 1: Fetching news for: %s", market.question[:60])
        articles = await self._news.fetch_articles_for_market(market.question)
        logger.info("Step 1 done: got %d articles", len(articles))
        return _format_articles(articles), len(articles), articles

    async def analyze_market(self, market: Market) -> ForecastResult:
        """Full pipeline: gather source material → prompt Claude → parse → compute EV."""
        outcomes = [t.outcome for t in market.tokens]
        end_date_str = (
            market.end_date.strftime("%Y-%m-%d") if market.end_date else "unspecified"
        )
        today_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

        # 1. Gather source material (research layer, falling back to news).
        articles_text, source_count, articles = await self._gather_context(
            market, end_date_str
        )

        # 2. Build prompt — intentionally exclude market prices to avoid anchoring
        user_prompt = build_user_prompt(
            question=market.question,
            description=market.description[:2000],
            outcomes=outcomes,
            end_date=end_date_str,
            today=today_str,
            articles_text=articles_text,
        )

        # 3. Call Claude with adaptive thinking + structured output.
        #    The system prompt is cached (prefix match) to cut cost on repeat calls.
        num_outcomes = len(outcomes)
        max_tokens = 4096 if num_outcomes <= 3 else min(4096 + num_outcomes * 512, 8192)
        await self._rate_limiter.acquire()
        logger.info("Calling Claude for: %s (%d outcomes)", market.question[:60], num_outcomes)
        response = await self._anthropic.messages.create(
            model=self._settings.claude_model,
            max_tokens=max_tokens,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
            thinking={"type": "adaptive"},
            output_config={
                "format": {"type": "json_schema", "schema": FORECAST_JSON_SCHEMA}
            },
            timeout=180.0,
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("Claude refused to analyze this market")

        # 4. Parse the structured forecast (guaranteed valid JSON in the text block).
        forecast = ForecastOutput.model_validate(json.loads(_extract_text_block(response)))
        logger.info(
            "Claude responded (confidence=%s, %d probabilities)",
            forecast.confidence.value,
            len(forecast.probabilities),
        )

        probs = _normalize_probs(
            {p.outcome.strip().lower(): p.probability for p in forecast.probabilities}
        )

        # 5. Compute EV per outcome against market prices.
        outcome_forecasts: list[OutcomeForecast] = []
        for token in market.tokens:
            bot_prob = probs.get(token.outcome.strip().lower())
            if bot_prob is None:
                # The model didn't return a probability for this outcome — don't
                # fabricate one. Skip it rather than bet on a guess.
                logger.warning("No model probability for outcome %r; skipping", token.outcome)
                continue
            of = evaluate_outcome(
                token.outcome,
                bot_prob,
                token.price,
                has_market_price=token.price > 0,
            )
            outcome_forecasts.append(of)

        result = ForecastResult(
            condition_id=market.condition_id,
            question=market.question,
            slug=market.slug,
            reasoning=forecast.briefing,
            outcomes=outcome_forecasts,
            confidence=CONFIDENCE_TO_FLOAT.get(forecast.confidence, 0.0),
            confidence_label=forecast.confidence.value,
            key_assumption=forecast.key_assumption,
            prompt_version=PROMPT_VERSION,
            news_article_count=len(articles),
            articles=articles,
        )

        # 6. Pre-commit validation: hard logic assertions raise (a bad forecast
        #    must never reach the user); soft gates flag for manual review and
        #    downgrade confidence on staleness.
        report = validate_forecast(result, market)
        if report.downgrade_confidence and result.confidence_label != "Low":
            logger.warning("Forecast downgraded to Low confidence (staleness gate).")
            result.confidence = 0.33
            result.confidence_label = "Low"
        result.flags = report.flags
        for flag in report.flags:
            logger.info("Forecast flag: %s", flag)
        return result

    async def analyze_by_ref(self, ref: str) -> ForecastResult | None:
        """Convenience: resolve a URL/slug/condition_id and analyze."""
        market = await self._polymarket.get_market(ref)
        if not market:
            return None
        return await self.analyze_market(market)
