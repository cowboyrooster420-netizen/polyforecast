from __future__ import annotations

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


class NewsClient:
    def __init__(self, settings: Settings) -> None:
        self._newsapi: NewsApiClient | None = None
        if settings.newsapi_key:
            self._newsapi = NewsApiClient(api_key=settings.newsapi_key)
        self._http = httpx.AsyncClient(timeout=15.0)

    async def close(self) -> None:
        await self._http.aclose()

    async def fetch_articles_for_market(
        self,
        question: str,
        max_articles: int = 10,
    ) -> list[Article]:
        """Fetch relevant news articles for a market question."""
        queries = extract_search_queries(question)
        articles: list[Article] = []
        seen_urls: set[str] = set()

        for query in queries:
            if len(articles) >= max_articles:
                break
            batch = await self._search(query)
            for art in batch:
                if art.url not in seen_urls:
                    seen_urls.add(art.url)
                    articles.append(art)
                    if len(articles) >= max_articles:
                        break

        return articles[:max_articles]

    async def search_topic(self, topic: str, max_articles: int = 10) -> list[Article]:
        """Search for news on a general topic."""
        return (await self._search(topic))[:max_articles]

    async def _search(self, query: str) -> list[Article]:
        """Try NewsAPI first, fall back to Google News RSS."""
        articles = await self._search_newsapi(query)
        if not articles:
            articles = await self._search_google_rss(query)
        return articles

    async def _search_newsapi(self, query: str) -> list[Article]:
        """Search using NewsAPI (sync SDK, but fast enough for our use)."""
        if not self._newsapi:
            return []
        try:
            from_date = (datetime.now(tz=timezone.utc) - timedelta(days=7)).strftime(
                "%Y-%m-%d"
            )
            resp = self._newsapi.get_everything(
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

    async def _search_google_rss(self, query: str) -> list[Article]:
        """Fallback: Google News RSS feed."""
        try:
            encoded_query = httpx.URL("", params={"q": query}).params["q"]
            url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
            resp = await self._http.get(url)
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)
            articles: list[Article] = []
            for entry in feed.entries[:10]:
                published = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    try:
                        published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                    except Exception:
                        pass
                articles.append(
                    Article(
                        title=entry.get("title", ""),
                        source=entry.get("source", {}).get("title", "Google News"),
                        url=entry.get("link", ""),
                        published_at=published,
                        description=entry.get("summary", ""),
                    )
                )
            return articles
        except Exception as exc:
            logger.warning("Google News RSS error: %s", exc)
            return []
