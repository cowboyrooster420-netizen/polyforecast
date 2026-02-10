from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import feedparser
import httpx
from newsapi import NewsApiClient
from newsapi.newsapi_exception import NewsAPIException

from src.config import Settings
from src.news.models import Article
from src.news.relevance import extract_search_queries

logger = logging.getLogger(__name__)

# ── Curated RSS feeds by category ────────────────────────────
# These are free, no API key needed, and provide high-quality sources.

RSS_FEEDS: dict[str, list[tuple[str, str]]] = {
    "general": [
        ("Reuters", "https://feeds.reuters.com/reuters/topNews"),
        ("AP News", "https://rsshub.app/apnews/topics/apf-topnews"),
        ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ],
    "politics": [
        ("Reuters Politics", "https://feeds.reuters.com/Reuters/PoliticsNews"),
        ("BBC Politics", "https://feeds.bbci.co.uk/news/politics/rss.xml"),
        ("The Hill", "https://thehill.com/feed/"),
    ],
    "crypto": [
        ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("CoinTelegraph", "https://cointelegraph.com/rss"),
        ("The Block", "https://www.theblock.co/rss.xml"),
    ],
    "finance": [
        ("Reuters Business", "https://feeds.reuters.com/reuters/businessNews"),
        ("MarketWatch", "https://feeds.marketwatch.com/marketwatch/topstories/"),
        ("CNBC", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
    ],
    "science": [
        ("Reuters Tech", "https://feeds.reuters.com/reuters/technologyNews"),
        ("Ars Technica", "https://feeds.arstechnica.com/arstechnica/index"),
        ("TechCrunch", "https://techcrunch.com/feed/"),
    ],
    "geopolitics": [
        ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
        ("Kyiv Independent", "https://kyivindependent.com/feed/"),
        ("Defense One", "https://www.defenseone.com/rss/"),
        ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
        ("Reuters World", "https://feeds.reuters.com/Reuters/worldNews"),
    ],
}


class NewsClient:
    def __init__(self, settings: Settings) -> None:
        self._newsapi: NewsApiClient | None = None
        if settings.newsapi_key and len(settings.newsapi_key) > 10:
            self._newsapi = NewsApiClient(api_key=settings.newsapi_key)
        self._guardian_key: str | None = None
        if settings.guardian_api_key and len(settings.guardian_api_key) > 5:
            self._guardian_key = settings.guardian_api_key
        self._http = httpx.AsyncClient(timeout=15.0)

    async def close(self) -> None:
        await self._http.aclose()

    async def fetch_articles_for_market(
        self,
        question: str,
        max_articles: int = 15,
    ) -> list[Article]:
        """Fetch relevant news from all sources in parallel."""
        queries = extract_search_queries(question)
        primary_query = queries[0] if queries else question

        # Fire all sources in parallel
        logger.info("News: fetching from all sources for: %s", primary_query[:60])
        results = await asyncio.gather(
            self._search_newsapi(primary_query),
            self._search_guardian(primary_query),
            self._search_google_rss(primary_query),
            self._search_rss_feeds(primary_query),
            return_exceptions=True,
        )

        # Merge and deduplicate
        articles: list[Article] = []
        seen_urls: set[str] = set()
        source_counts: dict[str, int] = {}

        for result in results:
            if isinstance(result, Exception):
                logger.debug("News source error: %s", result)
                continue
            for art in result:
                if art.url and art.url not in seen_urls:
                    seen_urls.add(art.url)
                    articles.append(art)
                    source_counts[art.source] = source_counts.get(art.source, 0) + 1

        # Sort by recency
        articles.sort(
            key=lambda a: a.published_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )

        logger.info(
            "News: got %d unique articles from %d sources",
            len(articles),
            len(source_counts),
        )
        return articles[:max_articles]

    async def search_topic(self, topic: str, max_articles: int = 10) -> list[Article]:
        """Search for news on a general topic."""
        return (await self.fetch_articles_for_market(topic, max_articles))

    # ── NewsAPI ──────────────────────────────────────────────

    async def _search_newsapi(self, query: str) -> list[Article]:
        """Search using NewsAPI (sync SDK wrapped in thread)."""
        if not self._newsapi:
            return []
        try:
            from_date = (datetime.now(tz=timezone.utc) - timedelta(days=7)).strftime(
                "%Y-%m-%d"
            )
            newsapi = self._newsapi
            resp = await asyncio.to_thread(
                newsapi.get_everything,
                q=query,
                from_param=from_date,
                sort_by="relevancy",
                page_size=10,
                language="en",
            )
            articles: list[Article] = []
            for item in resp.get("articles", []):
                published = None
                if item.get("publishedAt"):
                    try:
                        published = datetime.fromisoformat(
                            item["publishedAt"].replace("Z", "+00:00")
                        )
                    except ValueError:
                        pass
                articles.append(
                    Article(
                        title=item.get("title", ""),
                        source=item.get("source", {}).get("name", ""),
                        url=item.get("url", ""),
                        published_at=published,
                        description=item.get("description", "") or "",
                    )
                )
            return articles
        except NewsAPIException as exc:
            logger.warning("NewsAPI error: %s", exc)
            return []
        except Exception as exc:
            logger.warning("NewsAPI unexpected error: %s", exc)
            return []

    # ── The Guardian ─────────────────────────────────────────

    async def _search_guardian(self, query: str) -> list[Article]:
        """Search The Guardian's Open Platform API."""
        if not self._guardian_key:
            return []
        try:
            resp = await self._http.get(
                "https://content.guardianapis.com/search",
                params={
                    "q": query,
                    "api-key": self._guardian_key,
                    "page-size": 10,
                    "order-by": "relevance",
                    "show-fields": "trailText",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            articles: list[Article] = []
            for item in data.get("response", {}).get("results", []):
                published = None
                if item.get("webPublicationDate"):
                    try:
                        published = datetime.fromisoformat(
                            item["webPublicationDate"].replace("Z", "+00:00")
                        )
                    except ValueError:
                        pass
                articles.append(
                    Article(
                        title=item.get("webTitle", ""),
                        source="The Guardian",
                        url=item.get("webUrl", ""),
                        published_at=published,
                        description=item.get("fields", {}).get("trailText", ""),
                    )
                )
            return articles
        except Exception as exc:
            logger.warning("Guardian API error: %s", exc)
            return []

    # ── Google News RSS ──────────────────────────────────────

    async def _search_google_rss(self, query: str) -> list[Article]:
        """Google News RSS search."""
        try:
            encoded_query = httpx.URL("", params={"q": query}).params["q"]
            url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
            resp = await self._http.get(url)
            resp.raise_for_status()
            return self._parse_rss_feed(resp.text, "Google News")
        except Exception as exc:
            logger.warning("Google News RSS error: %s", exc)
            return []

    # ── Curated RSS feeds ────────────────────────────────────

    async def _search_rss_feeds(self, query: str) -> list[Article]:
        """Search curated RSS feeds for relevant articles."""
        query_lower = query.lower()
        keywords = set(query_lower.split())

        # Pick relevant feed categories based on query keywords
        categories_to_check = ["general"]
        for category, cat_keywords in {
            "politics": {"politics", "president", "congress", "election", "government", "senate", "bill", "law", "shutdown"},
            "crypto": {"crypto", "bitcoin", "btc", "ethereum", "eth", "solana", "defi", "token", "blockchain"},
            "finance": {"stock", "market", "fed", "inflation", "gdp", "economy", "gold", "oil", "rate", "treasury"},
            "science": {"ai", "tech", "space", "climate", "science", "model", "chip", "gpu", "openai"},
            "geopolitics": {"war", "ukraine", "russia", "ceasefire", "nato", "military", "conflict", "invasion", "troops", "weapons", "sanctions", "crimea", "zelensky", "putin", "peace", "missile", "drone", "frontline", "china", "taiwan", "iran", "israel", "gaza", "hamas", "hezbollah", "syria", "nuclear", "treaty"},
        }.items():
            if keywords & cat_keywords:
                categories_to_check.append(category)

        # Fetch all relevant feeds in parallel
        feeds_to_fetch: list[tuple[str, str]] = []
        for cat in categories_to_check:
            feeds_to_fetch.extend(RSS_FEEDS.get(cat, []))

        if not feeds_to_fetch:
            return []

        tasks = [self._fetch_single_rss(name, url) for name, url in feeds_to_fetch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter articles by keyword relevance
        all_articles: list[Article] = []
        for result in results:
            if isinstance(result, Exception):
                continue
            for art in result:
                # Basic relevance check — at least one query keyword in title
                title_lower = art.title.lower()
                if any(kw in title_lower for kw in keywords if len(kw) > 3):
                    all_articles.append(art)

        return all_articles

    async def _fetch_single_rss(self, source_name: str, feed_url: str) -> list[Article]:
        """Fetch and parse a single RSS feed."""
        try:
            resp = await self._http.get(feed_url)
            resp.raise_for_status()
            return self._parse_rss_feed(resp.text, source_name)[:5]
        except Exception:
            return []

    @staticmethod
    def _parse_rss_feed(text: str, default_source: str) -> list[Article]:
        """Parse RSS/Atom feed text into Article objects."""
        feed = feedparser.parse(text)
        articles: list[Article] = []
        for entry in feed.entries[:10]:
            published = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                except Exception:
                    pass
            source = default_source
            if hasattr(entry, "source") and isinstance(entry.source, dict):
                source = entry.source.get("title", default_source)
            articles.append(
                Article(
                    title=entry.get("title", ""),
                    source=source,
                    url=entry.get("link", ""),
                    published_at=published,
                    description=entry.get("summary", ""),
                )
            )
        return articles
