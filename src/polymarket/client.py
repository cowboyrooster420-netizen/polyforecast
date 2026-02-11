from __future__ import annotations

import logging
from datetime import datetime

import httpx

from src.config import Settings
from src.polymarket.models import Event, Market, Token
from src.polymarket.parser import ParsedMarketRef, parse_market_ref
from src.utils.retry import with_retry

logger = logging.getLogger(__name__)

# Gamma API category tags used for filtering
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "politics": ["politics", "election", "government", "president", "congress"],
    "crypto": ["crypto", "bitcoin", "ethereum", "defi", "blockchain", "token"],
    "science": ["science", "tech", "ai", "space", "climate", "health"],
    "sports": ["sports", "nfl", "nba", "soccer", "football", "baseball"],
    "finance": ["finance", "economy", "stock", "fed", "inflation", "gdp"],
    "entertainment": ["entertainment", "oscars", "grammy", "movie", "music"],
}


def _extract_outcome_name(market_question: str, event_title: str) -> str:
    """Try to extract a clean outcome name from a sub-market question.

    E.g. event "Who will win the 2026 election?" with sub-market
    "Will Donald Trump win the 2026 election?" → "Donald Trump"
    """
    import re

    q = market_question.strip().rstrip("?").strip()

    # Common patterns: "Will X win/happen/be...", "X to win/happen..."
    for pattern in [
        r"^Will\s+(.+?)\s+(?:win|be |become |get |reach |pass |capture )",
        r"^(.+?)\s+to\s+(?:win|be |become )",
    ]:
        m = re.match(pattern, q, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            if len(name) > 3:
                return name

    # Fallback: if question is different from event title, use the full question
    # but trim shared prefix/suffix with event title
    if len(q) < 60:
        return q

    return q[:60]


class PolymarketClient:
    def __init__(self, settings: Settings) -> None:
        self._gamma_base = settings.gamma_api_base
        self._clob_base = settings.clob_api_base
        self._http = httpx.AsyncClient(timeout=30.0)

    async def close(self) -> None:
        await self._http.aclose()

    # ── Gamma API helpers ────────────────────────────────────

    @with_retry(max_attempts=3)
    async def _gamma_get(self, path: str, params: dict | None = None) -> list[dict]:
        url = f"{self._gamma_base}{path}"
        resp = await self._http.get(url, params=params or {})
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else [data]

    @with_retry(max_attempts=2, retry_on=(httpx.TransportError, httpx.TimeoutException))
    async def _clob_get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self._clob_base}{path}"
        resp = await self._http.get(url, params=params or {})
        resp.raise_for_status()
        return resp.json()

    # ── Public methods ───────────────────────────────────────

    async def get_active_markets(
        self,
        limit: int = 10,
        category: str | None = None,
    ) -> list[Market]:
        """Fetch active markets from Gamma API, optionally filtered by category keyword."""
        params: dict[str, str | int] = {
            "active": "true",
            "closed": "false",
            "limit": min(limit * 3, 100),  # overfetch to allow filtering
            "order": "volume",
            "ascending": "false",
        }
        if category:
            params["tag"] = category

        raw_markets = await self._gamma_get("/markets", params)
        markets: list[Market] = []
        for item in raw_markets:
            try:
                market = self._parse_gamma_market(item)
                if category and not self._matches_category(market, category):
                    continue
                markets.append(market)
            except Exception:
                logger.debug("Skipping unparseable market: %s", item.get("question", "?"))
                continue
            if len(markets) >= limit:
                break

        # Enrich with CLOB prices
        for market in markets:
            await self._enrich_prices(market)

        return markets

    async def get_market(self, ref: str | ParsedMarketRef) -> Market | None:
        """Fetch a single market by URL, slug, or condition ID.

        For multi-outcome events, merges all sub-markets into one unified
        Market with all outcomes and their prices.
        """
        if isinstance(ref, str):
            ref = parse_market_ref(ref)

        if ref.condition_id:
            return await self._get_market_by_condition_id(ref.condition_id)

        if ref.slug:
            return await self._get_market_by_slug(ref.slug)

        if ref.event_slug:
            event = await self.get_event_by_slug(ref.event_slug)
            if not event or not event.markets:
                return None
            # Single sub-market → return directly
            if len(event.markets) == 1:
                return event.markets[0]
            # Multi-outcome event → merge into one virtual Market
            return self._merge_event_markets(event)

        return None

    @staticmethod
    def _merge_event_markets(event: Event) -> Market:
        """Merge multiple binary sub-markets into one multi-outcome Market.

        Polymarket represents multi-outcome events (e.g. "Who will win?")
        as separate binary markets per outcome. Each has a Yes token whose
        price represents the implied probability for that outcome.
        """
        tokens: list[Token] = []
        total_volume = 0.0
        total_liquidity = 0.0
        end_date = None
        description_parts: list[str] = []

        for m in event.markets:
            # Use the Yes token price as the outcome's implied probability
            yes_price = m.outcome_price("Yes") or 0.0
            yes_token_id = ""
            for t in m.tokens:
                if t.outcome.lower() == "yes":
                    yes_token_id = t.token_id
                    break

            # Derive outcome name from the sub-market question
            # e.g. "Will Donald Trump win?" → "Donald Trump"
            # or just use the question if we can't simplify it
            outcome_name = _extract_outcome_name(m.question, event.title)

            tokens.append(Token(
                token_id=yes_token_id,
                outcome=outcome_name,
                price=yes_price,
            ))
            total_volume += m.volume
            total_liquidity += m.liquidity
            if m.end_date:
                end_date = m.end_date
            if m.description and len(description_parts) < 3:
                description_parts.append(m.description)

        # Use event-level info for the merged market
        description = event.description or "\n---\n".join(description_parts)

        return Market(
            condition_id=event.event_id or event.markets[0].condition_id,
            question=event.title or event.markets[0].question,
            slug=event.slug,
            description=description[:3000],
            end_date=end_date,
            active=True,
            volume=total_volume,
            liquidity=total_liquidity,
            tokens=tokens,
            category=event.category,
        )

    async def get_event_by_slug(self, slug: str) -> Event | None:
        """Fetch an event and its markets by slug."""
        try:
            items = await self._gamma_get("/events", {"slug": slug})
        except httpx.HTTPStatusError:
            return None
        if not items:
            return None

        item = items[0]
        event = Event(
            event_id=str(item.get("id", "")),
            title=item.get("title", ""),
            slug=item.get("slug", slug),
            description=item.get("description", ""),
            category=item.get("category", ""),
        )

        raw_markets = item.get("markets", [])
        for rm in raw_markets:
            try:
                m = self._parse_gamma_market(rm)
                await self._enrich_prices(m)
                event.markets.append(m)
            except Exception:
                continue

        return event

    async def get_market_prices(self, token_ids: list[str]) -> dict[str, float]:
        """Get latest prices for token IDs from CLOB (per-token requests)."""
        if not token_ids:
            return {}
        prices: dict[str, float] = {}
        for tid in token_ids:
            try:
                data = await self._clob_get(f"/price", {"token_id": tid})
                if isinstance(data, dict) and "price" in data:
                    prices[tid] = float(data["price"])
            except Exception:
                logger.debug("CLOB price fetch failed for token %s", tid[:16])
        return prices

    # ── Private helpers ──────────────────────────────────────

    async def _get_market_by_condition_id(self, condition_id: str) -> Market | None:
        try:
            items = await self._gamma_get("/markets", {"condition_id": condition_id})
        except httpx.HTTPStatusError:
            return None
        if not items:
            return None
        market = self._parse_gamma_market(items[0])
        await self._enrich_prices(market)
        return market

    async def _get_market_by_slug(self, slug: str) -> Market | None:
        try:
            items = await self._gamma_get("/markets", {"slug": slug})
        except httpx.HTTPStatusError:
            return None
        if not items:
            return None
        market = self._parse_gamma_market(items[0])
        await self._enrich_prices(market)
        return market

    async def _enrich_prices(self, market: Market) -> None:
        """Fill in token prices from CLOB if Gamma didn't already provide them."""
        # Skip if all tokens already have prices from Gamma's outcomePrices
        if all(t.price > 0 for t in market.tokens):
            return
        token_ids = [t.token_id for t in market.tokens if t.token_id and t.price <= 0]
        if not token_ids:
            return
        prices = await self.get_market_prices(token_ids)
        for token in market.tokens:
            if token.token_id in prices:
                token.price = prices[token.token_id]

    def _parse_gamma_market(self, item: dict) -> Market:
        """Parse a raw Gamma API market dict into a Market model."""
        tokens: list[Token] = []
        for outcome_raw in item.get("outcomes", []):
            # outcomes can be a JSON string list like '["Yes", "No"]' or a Python list
            pass

        # Gamma API returns outcomes as a JSON string or list
        outcomes_raw = item.get("outcomes", "")
        if isinstance(outcomes_raw, str):
            import json
            try:
                outcomes_list = json.loads(outcomes_raw)
            except (json.JSONDecodeError, TypeError):
                outcomes_list = []
        else:
            outcomes_list = outcomes_raw

        clob_token_ids_raw = item.get("clobTokenIds", "")
        if isinstance(clob_token_ids_raw, str):
            import json
            try:
                clob_ids = json.loads(clob_token_ids_raw)
            except (json.JSONDecodeError, TypeError):
                clob_ids = []
        else:
            clob_ids = clob_token_ids_raw or []

        for i, outcome in enumerate(outcomes_list):
            tid = clob_ids[i] if i < len(clob_ids) else ""
            tokens.append(Token(token_id=tid, outcome=str(outcome)))

        # Parse out outcome prices from Gamma if available
        outcome_prices_raw = item.get("outcomePrices", "")
        if isinstance(outcome_prices_raw, str):
            import json
            try:
                outcome_prices = json.loads(outcome_prices_raw)
            except (json.JSONDecodeError, TypeError):
                outcome_prices = []
        else:
            outcome_prices = outcome_prices_raw or []

        for i, tok in enumerate(tokens):
            if i < len(outcome_prices):
                try:
                    tok.price = float(outcome_prices[i])
                except (ValueError, TypeError):
                    pass

        end_date = None
        if item.get("endDate"):
            try:
                end_date = datetime.fromisoformat(
                    str(item["endDate"]).replace("Z", "+00:00")
                )
            except ValueError:
                pass

        return Market(
            condition_id=item.get("conditionId", item.get("condition_id", "")),
            question=item.get("question", ""),
            slug=item.get("slug", ""),
            description=item.get("description", ""),
            end_date=end_date,
            active=bool(item.get("active", True)),
            closed=bool(item.get("closed", False)),
            resolved=bool(item.get("resolved", False)),
            resolution=item.get("resolution", ""),
            volume=float(item.get("volume", 0) or 0),
            liquidity=float(item.get("liquidity", 0) or 0),
            tokens=tokens,
            category=item.get("category", ""),
            image=item.get("image", ""),
        )

    @staticmethod
    def _matches_category(market: Market, category: str) -> bool:
        """Fuzzy check if a market matches a category by keywords."""
        category = category.lower()
        keywords = CATEGORY_KEYWORDS.get(category, [category])
        searchable = f"{market.question} {market.description} {market.category}".lower()
        return any(kw in searchable for kw in keywords)
