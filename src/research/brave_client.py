from __future__ import annotations

import logging
from datetime import timedelta

import httpx

from src.config import Settings
from src.research._util import hostname, parse_brave_age, utcnow
from src.research.models import FactKind, Provider, RetrievedFact
from src.utils.rate_limiter import AsyncTokenBucket
from src.utils.retry import with_retry

logger = logging.getLogger(__name__)


class BraveClient:
    """Brave LLM Context API for fresh, dated current-state facts.

    Returns raw ranked, LLM-ready chunks (the `grounding` payload). Does NOT use
    any answer/summarizer mode — we want raw context, not a pre-baked conclusion.
    """

    def __init__(
        self,
        settings: Settings,
        http: httpx.AsyncClient,
        limiter: AsyncTokenBucket,
    ) -> None:
        self._key = settings.brave_api_key
        self._url = settings.brave_llm_context_url
        self._http = http
        self._limiter = limiter
        self._max_snippets = settings.brave_snippets_per_query
        self._freshness = _freshness_window(settings.research_freshness_days)

    @property
    def enabled(self) -> bool:
        return bool(self._key)

    @with_retry(
        max_attempts=3,
        retry_on=(httpx.TransportError, httpx.TimeoutException, httpx.HTTPStatusError),
    )
    async def _get(self, params: dict) -> dict:
        resp = await self._http.get(
            self._url,
            params=params,
            headers={
                "X-Subscription-Token": self._key,
                "Accept": "application/json",
            },
        )
        resp.raise_for_status()
        return resp.json()

    async def search(
        self, query: str, kind: FactKind = FactKind.CURRENT
    ) -> list[RetrievedFact]:
        """Fetch fresh context chunks for one query; normalize to RetrievedFact."""
        if not self.enabled:
            logger.warning("Brave disabled (no BRAVE_API_KEY); skipping: %s", query[:60])
            return []

        params = {
            "q": query[:400],
            "country": "us",
            "search_lang": "en",
            "freshness": self._freshness,
            "maximum_number_of_snippets": self._max_snippets,
            "context_threshold_mode": "balanced",
        }
        await self._limiter.acquire()
        try:
            data = await self._get(params)
        except Exception as exc:
            logger.warning("Brave search failed for %r: %s", query[:60], exc)
            return []

        grounding = (data.get("grounding") or {}).get("generic") or []
        sources = data.get("sources") or {}

        facts: list[RetrievedFact] = []
        for item in grounding:
            url = item.get("url", "")
            src = sources.get(url, {}) if isinstance(sources, dict) else {}
            published = parse_brave_age(src.get("age"))
            name = src.get("hostname") or hostname(url) or item.get("title", "") or "Brave"
            for snippet in item.get("snippets", []) or []:
                text = (snippet or "").strip()
                if not text:
                    continue
                facts.append(
                    RetrievedFact(
                        text=text,
                        source_url=url,
                        source_name=name,
                        published_date=published,
                        retrieved_at=utcnow(),
                        provider=Provider.BRAVE,
                        originating_query=query,
                        kind=kind,
                        # Undated current facts can't be checked against the
                        # freshness window — treat as weaker signal.
                        low_confidence=published is None,
                    )
                )
        logger.info("Brave: %d facts for %r", len(facts), query[:60])
        return facts


def _freshness_window(days: int) -> str:
    """Build Brave's `freshness` value as a custom date range ending today."""
    end = utcnow().date()
    start = end - timedelta(days=max(days, 1))
    return f"{start.isoformat()}to{end.isoformat()}"
