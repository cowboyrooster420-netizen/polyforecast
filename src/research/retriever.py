from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta

import anthropic
import httpx

from src.config import Settings
from src.research._util import domain_blocked, looks_like_market_odds, utcnow
from src.research.brave_client import BraveClient
from src.research.decompose import QuestionDecomposer
from src.research.exa_client import ExaClient
from src.research.models import FactKind, ResearchBriefing, RetrievedFact
from src.utils.rate_limiter import AsyncTokenBucket

logger = logging.getLogger(__name__)


class ResearchRetriever:
    """Question-driven retrieval for the BLIND forecast stage.

    Pipeline per market:
      1. Decompose the question into sub-questions (Haiku).
      2. Base-rate pass: Exa semantic search on the question + each sub-question
         for analogous past situations / reference-class data.
      3. Current-state pass: Brave LLM Context on each sub-question for fresh facts.
      4. Disconfirming pass: one Exa query for evidence against the leading hypothesis.
      5. Normalize → dedupe → drop stale current facts → cap → assemble briefing.

    Returns FACTS ONLY. Never fetches or passes a market price.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._http = httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": "Polyforecast-Research/1.0"},
        )
        self._anthropic = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._exa = ExaClient(
            settings, self._http, AsyncTokenBucket(settings.exa_rpm / 60.0)
        )
        self._brave = BraveClient(
            settings, self._http, AsyncTokenBucket(settings.brave_rpm / 60.0)
        )
        self._decomposer = QuestionDecomposer(self._anthropic, settings.decomposition_model)
        self._max_facts = settings.research_max_facts
        self._freshness_days = settings.research_freshness_days
        self._blocked_domains = {d.lower() for d in settings.research_exclude_domains}

    async def close(self) -> None:
        await self._http.aclose()

    async def research(
        self,
        question: str,
        resolution_criteria: str = "",
        end_date: str = "",
    ) -> ResearchBriefing:
        """Run the full retrieval pipeline. `question`/`resolution_criteria` only —
        callers must NOT pass any price information."""
        # 1. Decompose
        sub_questions = await self._decomposer.decompose(
            question, resolution_criteria, end_date
        )

        # 2-4. Fan out all retrieval passes concurrently.
        #   Base rate: the overall question + each sub-question (Exa, semantic).
        #   Current:   each sub-question (Brave, fresh).
        #   Disconfirm: one explicit "evidence against" query (Exa).
        base_rate_queries = [question, *sub_questions]
        disconfirming_query = (
            f"evidence and reasons that this will NOT happen: {question}"
        )

        tasks: list = []
        tasks += [self._exa.search(q, FactKind.BASE_RATE) for q in base_rate_queries]
        tasks += [self._brave.search(q, FactKind.CURRENT) for q in sub_questions]
        tasks.append(self._exa.search(disconfirming_query, FactKind.DISCONFIRMING))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        facts: list[RetrievedFact] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Retrieval pass failed: %s", r)
                continue
            facts.extend(r)

        facts = self._postprocess(facts)
        logger.info(
            "Research complete: %d facts (%d base, %d current, %d disconfirming)",
            len(facts),
            sum(f.kind == FactKind.BASE_RATE for f in facts),
            sum(f.kind == FactKind.CURRENT for f in facts),
            sum(f.kind == FactKind.DISCONFIRMING for f in facts),
        )
        return ResearchBriefing(
            question=question, sub_questions=sub_questions, facts=facts
        )

    def _postprocess(self, facts: list[RetrievedFact]) -> list[RetrievedFact]:
        cutoff = utcnow().date() - timedelta(days=max(self._freshness_days, 1))

        # Dedupe by URL + content fingerprint, and drop stale current-state facts.
        seen: set[tuple[str, str]] = set()
        kept: list[RetrievedFact] = []
        dropped_price = 0
        for f in facts:
            if not f.text.strip():
                continue
            # Anti-anchoring: never let a market price reach the blind forecast.
            # Drop facts from prediction-market/odds aggregators or whose text
            # quotes odds/implied probabilities (catches mirror sites too).
            if domain_blocked(f.source_url, self._blocked_domains) or looks_like_market_odds(
                f.text
            ):
                dropped_price += 1
                continue
            # Current facts that are dated but older than the window are dropped;
            # undated current facts are kept (already flagged low-confidence).
            if (
                f.kind == FactKind.CURRENT
                and f.published_date is not None
                and f.published_date < cutoff
            ):
                continue
            key = f.dedupe_key()
            if key in seen:
                continue
            seen.add(key)
            kept.append(f)

        if dropped_price:
            logger.info("Dropped %d price/odds-contaminated facts (anti-anchoring)", dropped_price)
        return self._cap(kept)

    def _cap(self, facts: list[RetrievedFact]) -> list[RetrievedFact]:
        """Cap total facts, keeping diversity across kinds via round-robin.

        Within a kind, prefer dated facts and more recent ones.
        """
        if len(facts) <= self._max_facts:
            return facts

        def sort_key(f: RetrievedFact):
            has_date = f.published_date is not None
            # date.min as the floor for undated facts
            d = f.published_date or date.min
            return (has_date, d)

        buckets: dict[FactKind, list[RetrievedFact]] = {}
        for f in facts:
            buckets.setdefault(f.kind, []).append(f)
        for kind in buckets:
            buckets[kind].sort(key=sort_key, reverse=True)

        order = [FactKind.BASE_RATE, FactKind.CURRENT, FactKind.DISCONFIRMING]
        capped: list[RetrievedFact] = []
        idx = 0
        while len(capped) < self._max_facts:
            progressed = False
            for kind in order:
                bucket = buckets.get(kind, [])
                if idx < len(bucket):
                    capped.append(bucket[idx])
                    progressed = True
                    if len(capped) >= self._max_facts:
                        break
            if not progressed:
                break
            idx += 1
        return capped
