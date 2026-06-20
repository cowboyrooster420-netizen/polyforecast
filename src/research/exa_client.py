from __future__ import annotations

import logging

import httpx

from src.config import Settings
from src.research._util import hostname, parse_iso_date, utcnow
from src.research.models import FactKind, Provider, RetrievedFact
from src.utils.rate_limiter import AsyncTokenBucket
from src.utils.retry import with_retry

logger = logging.getLogger(__name__)


class ExaClient:
    """Exa semantic (neural) search for reference-class / base-rate research.

    Uses /search with contents+highlights. Does NOT use Exa's answer mode — we
    want analogous past situations and source material, not a pre-baked answer.

    Note: Exa's old bare `type="neural"` was removed; `type="auto"` is the
    current embeddings-first semantic search (the `deep*` types are the
    answer-style agentic modes we deliberately avoid).
    """

    def __init__(
        self,
        settings: Settings,
        http: httpx.AsyncClient,
        limiter: AsyncTokenBucket,
    ) -> None:
        self._key = settings.exa_api_key
        self._url = f"{settings.exa_api_base}/search"
        self._http = http
        self._limiter = limiter
        self._num_results = settings.exa_results_per_query
        self._exclude_domains = list(settings.research_exclude_domains)

    @property
    def enabled(self) -> bool:
        return bool(self._key)

    @with_retry(
        max_attempts=3,
        retry_on=(httpx.TransportError, httpx.TimeoutException, httpx.HTTPStatusError),
    )
    async def _post(self, body: dict) -> dict:
        resp = await self._http.post(
            self._url,
            json=body,
            headers={"x-api-key": self._key, "Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()

    async def search(self, query: str, kind: FactKind) -> list[RetrievedFact]:
        """Semantic search for a single query; normalize results to RetrievedFact.

        Used for both base-rate and disconfirming passes (both are semantic).
        No freshness filter — for base rates we want historical analogues.
        """
        if not self.enabled:
            logger.warning("Exa disabled (no EXA_API_KEY); skipping: %s", query[:60])
            return []

        body = {
            "query": query,
            "type": "auto",
            "numResults": self._num_results,
            "contents": {
                "highlights": {"query": query, "maxCharacters": 800},
                "text": {"maxCharacters": 600},
            },
        }
        # Keep prediction-market / odds aggregators out at the source.
        if self._exclude_domains:
            body["excludeDomains"] = self._exclude_domains
        await self._limiter.acquire()
        try:
            data = await self._post(body)
        except Exception as exc:
            logger.warning("Exa search failed for %r: %s", query[:60], exc)
            return []

        facts: list[RetrievedFact] = []
        for r in data.get("results", []):
            highlights = r.get("highlights") or []
            text = " … ".join(h.strip() for h in highlights if h.strip())
            if not text:
                text = (r.get("text") or "").strip()[:600]
            if not text:
                continue
            url = r.get("url", "")
            facts.append(
                RetrievedFact(
                    text=text,
                    source_url=url,
                    source_name=hostname(url) or (r.get("author") or "Exa"),
                    published_date=parse_iso_date(r.get("publishedDate")),
                    retrieved_at=utcnow(),
                    provider=Provider.EXA,
                    originating_query=query,
                    kind=kind,
                )
            )
        logger.info("Exa: %d facts for %r", len(facts), query[:60])
        return facts
