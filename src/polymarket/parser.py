from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ParsedMarketRef:
    slug: str | None = None
    condition_id: str | None = None
    event_slug: str | None = None


_POLYMARKET_URL_RE = re.compile(
    r"polymarket\.com/event/(?P<event_slug>[^/?#]+)"
    r"(?:/(?P<market_slug>[^/?#]+))?"
)

_CONDITION_ID_RE = re.compile(r"^0x[0-9a-fA-F]{64}$")


def parse_market_ref(text: str) -> ParsedMarketRef:
    """Parse a Polymarket URL, slug, or condition ID from user input."""
    text = text.strip()

    # Try URL match
    m = _POLYMARKET_URL_RE.search(text)
    if m:
        return ParsedMarketRef(
            event_slug=m.group("event_slug"),
            slug=m.group("market_slug"),
        )

    # Try condition ID (hex)
    if _CONDITION_ID_RE.match(text):
        return ParsedMarketRef(condition_id=text)

    # Treat as slug
    return ParsedMarketRef(slug=text)
