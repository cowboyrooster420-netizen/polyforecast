from __future__ import annotations

import re
from datetime import date, datetime, timezone
from urllib.parse import urlparse


def utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


# Patterns that signal a retrieved snippet is carrying market/betting odds or an
# implied probability — these must never reach the BLIND forecast (anti-anchoring).
_ODDS_PATTERNS = re.compile(
    r"(\d+\s*¢|¢|implied\s+probability|priced\s+at|shares?\s*@|open\s+interest"
    r"|order\s*book|\bbid\b.{0,24}\bask\b"
    r"|market\s+price|market\s+is\s+pricing|trading\s+at\s+\d"
    # betting recommendations are opinion/synthesis, not facts
    r"|\bno\s+bet\b|\bi\s+recommend\b|recommend\s+(a\s+|no\s+)?bet"
    r"|\b(kalshi|polymarket|predictit|manifold|metaculus|betfair|smarkets"
    r"|oddschecker|electionbettingodds|insightprediction)\b)",
    re.IGNORECASE,
)


def looks_like_market_odds(text: str) -> bool:
    """True if the text appears to quote betting/prediction-market odds or prices."""
    return bool(_ODDS_PATTERNS.search(text or ""))


def domain_blocked(url: str, blocked: set[str]) -> bool:
    """True if the URL's host equals or is a subdomain of any blocked domain."""
    host = hostname(url)
    return any(host == d or host.endswith("." + d) for d in blocked)


def hostname(url: str) -> str:
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def parse_iso_date(value: str | None) -> date | None:
    """Parse an ISO8601 date or datetime string into a date."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except ValueError:
        # Fall back to a bare YYYY-MM-DD prefix if present.
        try:
            return date.fromisoformat(str(value)[:10])
        except ValueError:
            return None


def parse_brave_age(age: list | None) -> date | None:
    """Brave's `sources[url].age` is [human, iso, relative] (or null).

    The ISO form is the second element; fall back to scanning for any parseable
    YYYY-MM-DD token.
    """
    if not age or not isinstance(age, list):
        return None
    if len(age) >= 2:
        parsed = parse_iso_date(age[1])
        if parsed:
            return parsed
    for entry in age:
        parsed = parse_iso_date(entry if isinstance(entry, str) else None)
        if parsed:
            return parsed
    return None
