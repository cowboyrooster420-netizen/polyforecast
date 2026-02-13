from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import aiosqlite

from src.forecasting.models import ForecastResult
from src.news.models import Article
from src.polymarket.models import Market


class Repository:
    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    # ── Predictions ──────────────────────────────────────────

    async def save_prediction(
        self,
        result: ForecastResult,
        articles: list[Article] | None = None,
        telegram_user_id: int | None = None,
    ) -> list[int]:
        """Save all outcome forecasts for a market. Returns list of prediction IDs."""
        pred_ids: list[int] = []
        for of in result.outcomes:
            cursor = await self._conn.execute(
                """INSERT INTO predictions
                   (condition_id, market_question, market_slug, outcome,
                    bot_probability, market_probability, ev_per_dollar,
                    kelly_fraction, recommendation, confidence,
                    reasoning_text, prompt_version, news_article_count,
                    telegram_user_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result.condition_id,
                    result.question,
                    result.slug,
                    of.outcome,
                    of.bot_probability,
                    of.market_probability,
                    of.ev_per_dollar,
                    of.kelly_fraction,
                    of.recommendation.value,
                    result.confidence,
                    result.reasoning,
                    result.prompt_version,
                    result.news_article_count,
                    telegram_user_id,
                ),
            )
            pred_ids.append(cursor.lastrowid)  # type: ignore[arg-type]

        # Save linked articles
        if articles:
            for pid in pred_ids:
                for art in articles:
                    pub = (
                        art.published_at.isoformat() if art.published_at else None
                    )
                    await self._conn.execute(
                        """INSERT INTO news_articles
                           (prediction_id, title, source, url, published_at, description)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (pid, art.title, art.source, art.url, pub, art.description),
                    )

        await self._conn.commit()
        return pred_ids

    async def save_market_snapshot(self, market: Market) -> None:
        for token in market.tokens:
            await self._conn.execute(
                """INSERT INTO market_snapshots
                   (condition_id, market_question, outcome, token_id,
                    price, volume, liquidity)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    market.condition_id,
                    market.question,
                    token.outcome,
                    token.token_id,
                    token.price,
                    market.volume,
                    market.liquidity,
                ),
            )
        await self._conn.commit()

    async def get_predictions_for_user(
        self, telegram_user_id: int, limit: int = 20
    ) -> list[dict[str, Any]]:
        cursor = await self._conn.execute(
            """SELECT id, created_at, condition_id, market_question, outcome,
                      bot_probability, market_probability, ev_per_dollar,
                      kelly_fraction, recommendation, resolved,
                      actual_outcome, brier_component
               FROM predictions
               WHERE telegram_user_id = ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (telegram_user_id, limit),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def resolve_prediction(
        self,
        condition_id: str,
        winning_outcome: str,
    ) -> int:
        """Mark predictions as resolved and compute Brier components.

        Returns number of updated rows.
        """
        cursor = await self._conn.execute(
            "SELECT id, outcome, bot_probability FROM predictions "
            "WHERE condition_id = ? AND resolved = 0",
            (condition_id,),
        )
        rows = await cursor.fetchall()
        count = 0
        now = datetime.now(tz=timezone.utc).isoformat()
        for row in rows:
            actual = 1.0 if row["outcome"].lower() == winning_outcome.lower() else 0.0
            brier = (row["bot_probability"] - actual) ** 2
            await self._conn.execute(
                """UPDATE predictions
                   SET resolved = 1, actual_outcome = ?, resolution_date = ?,
                       brier_component = ?
                   WHERE id = ?""",
                (winning_outcome, now, brier, row["id"]),
            )
            count += 1
        await self._conn.commit()
        return count

    async def get_unresolved_predictions(
        self, telegram_user_id: int | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Get distinct unresolved markets."""
        where = "WHERE resolved = 0"
        params: tuple = ()
        if telegram_user_id:
            where += " AND telegram_user_id = ?"
            params = (telegram_user_id,)
        cursor = await self._conn.execute(
            f"""SELECT condition_id, market_question, market_slug,
                       MIN(created_at) as first_analyzed,
                       GROUP_CONCAT(DISTINCT outcome) as outcomes
                FROM predictions
                {where}
                GROUP BY condition_id
                ORDER BY first_analyzed DESC
                LIMIT ?""",
            (*params, limit),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ── Calibration / stats ──────────────────────────────────

    async def get_brier_score(self, telegram_user_id: int | None = None) -> float | None:
        """Average Brier score across all resolved predictions."""
        where = "WHERE resolved = 1 AND brier_component IS NOT NULL"
        params: tuple = ()
        if telegram_user_id:
            where += " AND telegram_user_id = ?"
            params = (telegram_user_id,)
        cursor = await self._conn.execute(
            f"SELECT AVG(brier_component) as avg_brier, COUNT(*) as cnt "
            f"FROM predictions {where}",
            params,
        )
        row = await cursor.fetchone()
        if not row or row["cnt"] == 0:
            return None
        return row["avg_brier"]

    async def get_win_rate(self, telegram_user_id: int | None = None) -> dict[str, Any]:
        """Win rate for resolved predictions where we recommended BUY/STRONG_BUY."""
        where = "WHERE resolved = 1 AND recommendation IN ('BUY', 'STRONG_BUY')"
        params: tuple = ()
        if telegram_user_id:
            where += " AND telegram_user_id = ?"
            params = (telegram_user_id,)

        cursor = await self._conn.execute(
            f"""SELECT
                COUNT(*) as total,
                SUM(CASE WHEN outcome = actual_outcome THEN 1 ELSE 0 END) as wins
            FROM predictions {where}""",
            params,
        )
        row = await cursor.fetchone()
        total = row["total"] if row else 0
        wins = row["wins"] if row else 0
        return {
            "total": total,
            "wins": wins,
            "win_rate": wins / total if total > 0 else None,
        }

    async def get_calibration_data(
        self, telegram_user_id: int | None = None
    ) -> list[dict[str, Any]]:
        """Return calibration buckets (0.0-0.1, 0.1-0.2, ... 0.9-1.0)."""
        where = "WHERE resolved = 1 AND brier_component IS NOT NULL"
        params: tuple = ()
        if telegram_user_id:
            where += " AND telegram_user_id = ?"
            params = (telegram_user_id,)

        buckets: list[dict[str, Any]] = []
        for low in [i / 10 for i in range(10)]:
            high = low + 0.1
            cursor = await self._conn.execute(
                f"""SELECT
                    AVG(bot_probability) as predicted_avg,
                    AVG(CASE WHEN outcome = actual_outcome THEN 1.0 ELSE 0.0 END) as actual_freq,
                    COUNT(*) as cnt
                FROM predictions
                {where}
                AND bot_probability >= ? AND bot_probability < ?""",
                (*params, low, high),
            )
            row = await cursor.fetchone()
            if row and row["cnt"] > 0:
                buckets.append(
                    {
                        "bucket_lower": low,
                        "bucket_upper": high,
                        "predicted_avg": row["predicted_avg"],
                        "actual_frequency": row["actual_freq"],
                        "count": row["cnt"],
                    }
                )
        return buckets

    async def get_prediction_count(self, telegram_user_id: int | None = None) -> int:
        where = ""
        params: tuple = ()
        if telegram_user_id:
            where = "WHERE telegram_user_id = ?"
            params = (telegram_user_id,)
        cursor = await self._conn.execute(
            f"SELECT COUNT(DISTINCT condition_id) as cnt FROM predictions {where}",
            params,
        )
        row = await cursor.fetchone()
        return row["cnt"] if row else 0

    # ── User state ───────────────────────────────────────────

    async def get_user_categories(self, telegram_user_id: int) -> list[str]:
        cursor = await self._conn.execute(
            "SELECT default_categories FROM user_state WHERE telegram_user_id = ?",
            (telegram_user_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return ["science", "crypto", "politics"]
        return json.loads(row["default_categories"])

    async def set_user_categories(
        self, telegram_user_id: int, categories: list[str]
    ) -> None:
        cats_json = json.dumps(categories)
        await self._conn.execute(
            """INSERT INTO user_state (telegram_user_id, default_categories)
               VALUES (?, ?)
               ON CONFLICT(telegram_user_id)
               DO UPDATE SET default_categories = ?, last_active = datetime('now')""",
            (telegram_user_id, cats_json, cats_json),
        )
        await self._conn.commit()

    async def touch_user(self, telegram_user_id: int) -> None:
        await self._conn.execute(
            """INSERT INTO user_state (telegram_user_id)
               VALUES (?)
               ON CONFLICT(telegram_user_id)
               DO UPDATE SET last_active = datetime('now')""",
            (telegram_user_id,),
        )
        await self._conn.commit()
