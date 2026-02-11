from __future__ import annotations

import asyncio
import logging
import re
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
        ("AP News", "https://apnews.com/feed"),
        ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ],
    "politics": [
        ("Reuters Politics", "https://feeds.reuters.com/Reuters/PoliticsNews"),
        ("BBC Politics", "https://feeds.bbci.co.uk/news/politics/rss.xml"),
        ("The Hill", "https://thehill.com/news/feed/"),
        ("Politico", "https://rss.politico.com/congress.xml"),
        ("Politico Picks", "https://rss.politico.com/politicopicks.xml"),
        ("NPR Politics", "https://feeds.npr.org/1014/rss.xml"),
        ("ProPublica", "https://feeds.propublica.org/propublica/main"),
        ("Roll Call", "https://rollcall.com/feed/"),
        ("Axios", "https://www.axios.com/feeds/feed.rss"),
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
        ("Kyiv Independent", "https://kyivindependent.com/news-archive/rss/"),
        ("Defense One", "https://www.defenseone.com/rss/all/"),
        ("Politico Defense", "https://rss.politico.com/defense.xml"),
        ("The Intercept", "https://theintercept.com/feed/?rss"),
        ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
        ("Reuters World", "https://feeds.reuters.com/Reuters/worldNews"),
        # Telegram — geopolitics & conflict aggregators
        ("DD Geopolitics", "tg://DDGeopolitics"),
        ("Geopolitics Watch", "tg://GeoPWatch"),
        ("War Monitor Global", "tg://warmonitors"),
        ("Middle East Spectator", "tg://Middle_East_Spectator"),
        ("Geopolitics Live", "tg://geopolitics_live"),
    ],
    "frontline": [
        # ISW daily assessments via WordPress REST API (no RSS available)
        ("ISW", "isw_api://"),
        # Telegram channels — scraped from public web previews
        ("DeepState UA", "tg://DeepStateUA"),
        ("Rybar EN", "tg://rybar_in_english"),
        ("Ukraine NOW", "tg://ukrainenowenglish"),
        ("WarMonitor", "tg://WarMonitor1"),
        # RSS feeds
        ("Kyiv Independent", "https://kyivindependent.com/news-archive/rss/"),
        ("Defense One", "https://www.defenseone.com/rss/all/"),
        ("Militaryland", "https://militaryland.net/feed/"),
        ("War Zone", "https://www.twz.com/feed"),
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
        self._http = httpx.AsyncClient(
            timeout=15.0,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; Polyforecast/1.0; +https://github.com/cowboyrooster420-netizen/polyforecast)",
            },
        )

    async def close(self) -> None:
        await self._http.aclose()

    async def fetch_articles_for_market(
        self,
        question: str,
        max_articles: int = 25,
    ) -> list[Article]:
        """Fetch relevant news from all sources in parallel."""
        queries = extract_search_queries(question)
        primary_query = queries[0] if queries else question
        # Shorter keyword query for APIs that need concise input (GDELT)
        keyword_query = queries[-1] if len(queries) > 1 else primary_query

        # Fire all sources in parallel
        logger.info("News: fetching from all sources for: %s", primary_query[:60])
        results = await asyncio.gather(
            self._search_newsapi(primary_query),
            self._search_guardian(primary_query),
            self._search_google_rss(primary_query),
            self._search_gdelt(keyword_query),
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
            from_date = (datetime.now(tz=timezone.utc) - timedelta(days=30)).strftime(
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
        """Google News RSS search with 30-day lookback."""
        try:
            after_date = (datetime.now(tz=timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
            full_query = f"{query} after:{after_date}"
            encoded_query = httpx.URL("", params={"q": full_query}).params["q"]
            url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
            resp = await self._http.get(url)
            resp.raise_for_status()
            return self._parse_rss_feed(resp.text, "Google News")
        except Exception as exc:
            logger.warning("Google News RSS error: %s", exc)
            return []

    # ── GDELT (Global Database of Events, Language, and Tone) ─

    async def _search_gdelt(self, query: str) -> list[Article]:
        """Search GDELT for historical news coverage. Free, no API key, searches back months."""
        try:
            resp = await self._http.get(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                params={
                    "query": f"{query} sourcelang:eng",
                    "mode": "artlist",
                    "maxrecords": 15,
                    "format": "json",
                    "sort": "hybridrel",  # relevance + recency blend
                },
            )
            resp.raise_for_status()
            text = resp.text.strip()
            if not text or text.startswith("<!"):
                # GDELT returns empty or HTML error page for no results
                logger.info("GDELT: no results for: %s", query[:40])
                return []
            data = resp.json()
            articles: list[Article] = []
            for item in data.get("articles", []):
                published = None
                if item.get("seendate"):
                    try:
                        # GDELT format: "20250210T143000Z"
                        published = datetime.strptime(
                            item["seendate"], "%Y%m%dT%H%M%SZ"
                        ).replace(tzinfo=timezone.utc)
                    except ValueError:
                        pass
                articles.append(
                    Article(
                        title=item.get("title", ""),
                        source=item.get("domain", "GDELT"),
                        url=item.get("url", ""),
                        published_at=published,
                        description=item.get("title", ""),
                    )
                )
            logger.info("GDELT: found %d articles for: %s", len(articles), query[:40])
            return articles
        except Exception as exc:
            logger.warning("GDELT API error: %s", exc)
            return []

    # ── Curated RSS feeds ────────────────────────────────────

    async def _search_rss_feeds(self, query: str) -> list[Article]:
        """Search curated RSS feeds for relevant articles."""
        query_lower = query.lower()
        keywords = set(query_lower.split())

        # Pick relevant feed categories based on query keywords
        categories_to_check = ["general"]
        for category, cat_keywords in {
            "politics": {"politics", "president", "congress", "election", "government", "senate", "bill", "law", "shutdown", "democrat", "republican", "gop", "biden", "trump", "vote", "poll", "legislation", "impeach", "scotus", "supreme", "speaker", "governor", "primary", "ballot", "midterm", "veto", "executive", "cabinet"},
            "crypto": {"crypto", "bitcoin", "btc", "ethereum", "eth", "solana", "defi", "token", "blockchain"},
            "finance": {"stock", "market", "fed", "inflation", "gdp", "economy", "gold", "oil", "rate", "treasury"},
            "science": {"ai", "tech", "space", "climate", "science", "model", "chip", "gpu", "openai"},
            "geopolitics": {"war", "ukraine", "russia", "ceasefire", "nato", "military", "conflict", "invasion", "troops", "weapons", "sanctions", "crimea", "zelensky", "putin", "peace", "missile", "drone", "frontline", "china", "taiwan", "iran", "israel", "gaza", "hamas", "hezbollah", "syria", "nuclear", "treaty"},
            "frontline": {"capture", "captured", "frontline", "advance", "offensive", "assault", "battalion", "brigade", "regiment", "oblast", "zaporizhzhia", "donetsk", "luhansk", "kherson", "bakhmut", "avdiivka", "huliaipole", "tokmak", "robotyne", "kupyansk", "chasiv", "pokrovsk", "vuhledar", "marinka", "isw", "deepstate", "counterattack", "defense", "fortification", "trench", "artillery", "position"},
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
        """Fetch and parse a single RSS feed, scrape Telegram, or call ISW API."""
        try:
            if feed_url.startswith("tg://"):
                return await self._scrape_telegram_channel(
                    source_name, feed_url[5:]
                )
            if feed_url == "isw_api://":
                return await self._fetch_isw_api()
            resp = await self._http.get(feed_url)
            resp.raise_for_status()
            return self._parse_rss_feed(resp.text, source_name)[:10]
        except Exception:
            return []

    async def _fetch_isw_api(self) -> list[Article]:
        """Fetch ISW daily assessments via WordPress REST API."""
        try:
            resp = await self._http.get(
                "https://understandingwar.org/wp-json/wp/v2/posts",
                params={"per_page": 10},
            )
            resp.raise_for_status()
            posts = resp.json()
            articles: list[Article] = []
            for post in posts:
                published = None
                if post.get("date_gmt"):
                    try:
                        published = datetime.fromisoformat(
                            post["date_gmt"] + "+00:00"
                        )
                    except ValueError:
                        pass
                # Title comes as {"rendered": "..."}
                title = post.get("title", {}).get("rendered", "")
                # Excerpt as description
                excerpt = post.get("excerpt", {}).get("rendered", "")
                excerpt = re.sub(r"<[^>]+>", "", excerpt).strip()[:500]
                articles.append(
                    Article(
                        title=title,
                        source="ISW",
                        url=post.get("link", ""),
                        published_at=published,
                        description=excerpt,
                    )
                )
            logger.info("ISW API: fetched %d posts", len(articles))
            return articles
        except Exception as exc:
            logger.warning("ISW API error: %s", exc)
            return []

    async def _scrape_telegram_channel(
        self, source_name: str, channel: str
    ) -> list[Article]:
        """Scrape recent posts from a public Telegram channel's web preview."""
        try:
            url = f"https://t.me/s/{channel}"
            resp = await self._http.get(url, follow_redirects=True)
            resp.raise_for_status()
            html = resp.text
            articles: list[Article] = []
            # Each message is in a tgme_widget_message div
            messages = re.findall(
                r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>',
                html,
                re.DOTALL,
            )
            # Extract timestamps
            dates = re.findall(
                r'<time[^>]*datetime="([^"]+)"', html
            )

            for i, msg_html in enumerate(messages[-15:]):
                # Strip HTML tags to get plain text
                text = re.sub(r"<[^>]+>", " ", msg_html).strip()
                text = re.sub(r"\s+", " ", text)
                if len(text) < 20:
                    continue

                published = None
                date_idx = len(messages) - 15 + i
                if 0 <= date_idx < len(dates):
                    try:
                        published = datetime.fromisoformat(
                            dates[date_idx].replace("+00:00", "+00:00")
                        )
                        if published.tzinfo is None:
                            published = published.replace(tzinfo=timezone.utc)
                    except ValueError:
                        pass

                # Use first ~100 chars as title, full text as description
                title = text[:100] + ("..." if len(text) > 100 else "")
                articles.append(
                    Article(
                        title=title,
                        source=source_name,
                        url=f"https://t.me/{channel}",
                        published_at=published,
                        description=text[:500],
                    )
                )
            logger.info("Telegram %s: scraped %d posts", channel, len(articles))
            return articles
        except Exception as exc:
            logger.warning("Telegram scrape error for %s: %s", channel, exc)
            return []

    @staticmethod
    def _parse_rss_feed(text: str, default_source: str) -> list[Article]:
        """Parse RSS/Atom feed text into Article objects."""
        feed = feedparser.parse(text)
        articles: list[Article] = []
        for entry in feed.entries[:20]:
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
