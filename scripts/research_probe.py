#!/usr/bin/env python3
"""Probe the question-driven retrieval layer on one real market.

Usage:
    python -m scripts.research_probe <market-slug-or-url>

Resolves the market, runs the full retrieval pipeline (decompose → Exa base-rate
→ Brave current → disconfirming), and prints the assembled ResearchBriefing.

This is a STANDALONE probe — it is NOT wired into live /analyze. It exists so you
can eyeball that:
  * we query the sub-questions / reference classes, not "latest news about X"
  * every fact carries a source and date
  * NO market price is ever passed into the retrieval layer

Requires EXA_API_KEY and BRAVE_API_KEY for the retrieval passes. Decomposition
runs on ANTHROPIC_API_KEY alone, so the sub-questions still print without them.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from src.config import Settings
from src.polymarket.client import PolymarketClient
from src.research.retriever import ResearchRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


async def run(ref: str) -> None:
    settings = Settings.from_env()

    if not settings.exa_api_key:
        logger.warning("EXA_API_KEY not set — base-rate/disconfirming passes will be empty.")
    if not settings.brave_api_key:
        logger.warning("BRAVE_API_KEY not set — current-state pass will be empty.")

    polymarket = PolymarketClient(settings)
    retriever = ResearchRetriever(settings)

    try:
        market = await polymarket.get_market(ref)
        if not market:
            logger.error("Could not resolve market for %r", ref)
            return

        end_date = (
            market.end_date.strftime("%Y-%m-%d") if market.end_date else "unspecified"
        )

        # Anti-anchoring check: show the prices to the OPERATOR only. They are
        # deliberately NOT passed into research() below.
        print("\n" + "=" * 70)
        print("MARKET (resolved from Polymarket)")
        print("=" * 70)
        print(f"Question: {market.question}")
        print(f"Resolution date: {end_date}")
        print(
            "Prices (FYI ONLY — NOT sent to the retrieval layer): "
            + ", ".join(f"{t.outcome}={t.price}" for t in market.tokens)
        )

        # Only the question + resolution criteria reach research — never a price.
        briefing = await retriever.research(
            question=market.question,
            resolution_criteria=market.description,
            end_date=end_date,
        )

        print("\n" + "=" * 70)
        print("ASSEMBLED RESEARCH BRIEFING")
        print("=" * 70)
        print(briefing.render())

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Sub-questions generated: {len(briefing.sub_questions)}")
        for q in briefing.sub_questions:
            print(f"  - {q}")
        print(f"Total facts: {len(briefing.facts)}")
        for kind in ("base_rate", "current", "disconfirming"):
            n = sum(f.kind.value == kind for f in briefing.facts)
            print(f"  {kind}: {n}")
        undated_current = sum(
            f.kind.value == "current" and f.published_date is None
            for f in briefing.facts
        )
        print(f"  current facts flagged low-confidence (undated): {undated_current}")
    finally:
        await retriever.close()
        await polymarket.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe the research retrieval layer")
    parser.add_argument("ref", help="Polymarket market slug, event slug, URL, or condition ID")
    args = parser.parse_args()
    asyncio.run(run(args.ref))


if __name__ == "__main__":
    main()
