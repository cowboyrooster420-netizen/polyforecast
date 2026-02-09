#!/usr/bin/env python3
"""Backtest the forecasting engine against resolved Polymarket markets.

Usage:
    python -m scripts.backtest [--limit 20]

Fetches recently resolved markets, runs the superforecasting pipeline on each
(using only pre-resolution news where possible), saves predictions, then
auto-resolves them to compute Brier scores.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from src.config import Settings
from src.database.db import init_db
from src.database.repository import Repository
from src.forecasting.engine import ForecastingEngine
from src.news.client import NewsClient
from src.polymarket.client import PolymarketClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


async def run_backtest(limit: int = 20) -> None:
    settings = Settings.from_env()
    conn = await init_db(settings.db_path)
    repo = Repository(conn)
    polymarket = PolymarketClient(settings)
    news = NewsClient(settings)
    engine = ForecastingEngine(settings, polymarket, news)

    logger.info("Fetching resolved markets...")

    # Fetch resolved markets from Gamma API
    try:
        raw = await polymarket._gamma_get(
            "/markets",
            {
                "closed": "true",
                "resolved": "true",
                "limit": limit * 2,
                "order": "volume",
                "ascending": "false",
            },
        )
    except Exception as exc:
        logger.error("Failed to fetch resolved markets: %s", exc)
        return

    analyzed = 0
    total_brier = 0.0
    brier_count = 0

    for item in raw:
        if analyzed >= limit:
            break

        resolution = item.get("resolution", "")
        question = item.get("question", "")
        if not resolution or not question:
            continue

        logger.info("Analyzing: %s", question[:80])

        try:
            market = polymarket._parse_gamma_market(item)
            await polymarket._enrich_prices(market)

            # Run forecasting pipeline
            result = await engine.analyze_market(market)
            pred_ids = await repo.save_prediction(result)

            # Auto-resolve with known outcome
            count = await repo.resolve_prediction(
                result.condition_id, resolution
            )
            logger.info(
                "  Resolved %d prediction rows for '%s' -> %s",
                count, question[:50], resolution,
            )

            # Track Brier
            for of in result.outcomes:
                actual = 1.0 if of.outcome.lower() == resolution.lower() else 0.0
                brier = (of.bot_probability - actual) ** 2
                total_brier += brier
                brier_count += 1

            analyzed += 1
            logger.info(
                "  Bot probs: %s | Market probs: %s",
                {o.outcome: f"{o.bot_probability:.2f}" for o in result.outcomes},
                {o.outcome: f"{o.market_probability:.2f}" for o in result.outcomes},
            )

        except Exception as exc:
            logger.warning("  Skipping due to error: %s", exc)
            continue

    # Summary
    logger.info("=" * 60)
    logger.info("Backtest complete: %d markets analyzed", analyzed)
    if brier_count:
        avg_brier = total_brier / brier_count
        logger.info("Average Brier score: %.4f", avg_brier)
        logger.info("  (0.0 = perfect, 0.25 = coin flip, 0.5 = always wrong)")
    else:
        logger.info("No Brier scores computed (no resolved outcomes).")

    # Show calibration from DB
    buckets = await repo.get_calibration_data()
    if buckets:
        logger.info("\nCalibration buckets:")
        for b in buckets:
            logger.info(
                "  %.0f%%-%.0f%%: predicted=%.2f actual=%.2f (n=%d)",
                b["bucket_lower"] * 100,
                b["bucket_upper"] * 100,
                b["predicted_avg"],
                b["actual_frequency"],
                b["count"],
            )

    await polymarket.close()
    await news.close()
    await conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Polyforecast")
    parser.add_argument("--limit", type=int, default=20, help="Number of markets to backtest")
    args = parser.parse_args()
    asyncio.run(run_backtest(limit=args.limit))


if __name__ == "__main__":
    main()
