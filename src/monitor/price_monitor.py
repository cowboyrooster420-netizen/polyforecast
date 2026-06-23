from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

from src.config import Settings
from src.forecasting.engine import ForecastingEngine
from src.forecasting.models import ForecastResult, OutcomeForecast
from src.monitor.detector import MoveEvent, detect_moves
from src.polymarket.client import PolymarketClient
from src.polymarket.models import Market

logger = logging.getLogger(__name__)

AlertFn = Callable[[str], Awaitable[None]]

# Lightweight category tagging so the paper phase can show WHERE fades pay
# (e.g. "fades win in politics, lose in crypto"). Order matters — first hit wins.
_CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("crypto", ("bitcoin", "btc", "ethereum", "eth", "solana", "crypto", "dogecoin", "xrp")),
    ("sports", ("nba", "nfl", "mlb", "ufc", "champion", "cup", "playoff", " vs ", "f1", "tennis", "soccer", "league")),
    ("entertainment", ("box office", "movie", "film", "oscar", "emmy", "grammy", "rotten", "netflix", "weekend", "album")),
    ("geopolitics", ("invade", "war", "ceasefire", "nuclear", "missile", "regime", "coup", "sanction", "hostage")),
    ("politics", ("election", "president", "senate", "congress", "governor", "trump", "putin", "vote", "nominee", "primary", "minister", "parliament")),
    ("macro", ("fed", "rate cut", "rate hike", "fomc", "cpi", "inflation", "gdp", "jobs", "payroll", "recession")),
    ("science", ("fda", "approval", "vaccine", "drug", "clinical", "launch", "spacex", "nasa", "nobel")),
    ("business", ("ipo", "merger", "acquisition", "earnings", "ceo", "bankruptcy", "largest company")),
]


def classify_category(question: str) -> str:
    q = question.lower()
    for label, kws in _CATEGORY_KEYWORDS:
        if any(k in q for k in kws):
            return label
    return "other"


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


class PriceMonitor:
    """Polls a fit-tag universe for sharp price moves, then runs the BLIND
    forecaster against the movers to judge whether fair value actually moved
    (news is real → no trade) or the move was FOMO (fair value unchanged →
    a fade candidate). Observation-only: it logs + alerts, never trades.
    """

    def __init__(
        self,
        settings: Settings,
        polymarket: PolymarketClient,
        engine: ForecastingEngine,
        repo,
        alert: AlertFn | None = None,
    ) -> None:
        self.settings = settings
        self.polymarket = polymarket
        self.engine = engine
        self.repo = repo
        self.alert = alert
        self._universe: dict[str, Market] = {}
        self._universe_at: datetime | None = None

    async def run_forever(self) -> None:
        s = self.settings
        logger.info(
            "PriceMonitor live: poll %dm, window %.1fh, trigger ≥%.0fpp, "
            "≤%d forecasts/cycle, ≤%d/day",
            s.monitor_poll_minutes,
            s.monitor_window_hours,
            s.monitor_move_threshold * 100,
            s.monitor_max_forecasts_per_cycle,
            s.monitor_daily_forecast_cap,
        )
        while True:
            try:
                await self.poll_once()
            except Exception as exc:  # never let the loop die
                logger.exception("monitor poll failed: %s", exc)
            await asyncio.sleep(s.monitor_poll_minutes * 60)

    # ── one cycle ────────────────────────────────────────────

    async def poll_once(self) -> None:
        await self._ensure_universe()
        if not self._universe:
            return
        await self._snapshot_prices()

        since = (_now() - timedelta(hours=self.settings.monitor_window_hours)).isoformat()
        rows = await self.repo.get_recent_prices(since)
        moves = detect_moves(rows, self.settings.monitor_move_threshold)
        if not moves:
            return
        logger.info("monitor: %d market(s) moved ≥ threshold", len(moves))

        # Cost guards: daily cap, per-cycle cap, per-market cooldown.
        day_ago = (_now() - timedelta(hours=24)).isoformat()
        used_today = await self.repo.count_fade_candidates_since(day_ago)
        budget = self.settings.monitor_daily_forecast_cap - used_today
        if budget <= 0:
            logger.info("monitor: daily forecast cap reached, skipping")
            return
        cooldown_since = (
            _now() - timedelta(hours=self.settings.monitor_cooldown_hours)
        ).isoformat()

        processed = 0
        per_cycle = min(self.settings.monitor_max_forecasts_per_cycle, budget)
        for ev in moves:
            if processed >= per_cycle:
                break
            market = self._universe.get(ev.condition_id)
            if market is None:
                continue
            if await self.repo.was_triggered_since(ev.condition_id, cooldown_since):
                continue
            try:
                await self._handle_move(ev, market)
                processed += 1
            except Exception as exc:
                logger.warning("monitor: handling %s failed: %s", ev.condition_id, exc)

    # ── helpers ──────────────────────────────────────────────

    async def _ensure_universe(self) -> None:
        """Refresh the watched-market set at most hourly (discovery is slow)."""
        if (
            self._universe
            and self._universe_at
            and _now() - self._universe_at < timedelta(hours=1)
        ):
            return
        found: dict[str, Market] = {}
        if self.settings.monitor_broad_mode:
            # Broad-minus-noise: top markets by volume across all categories
            # (where attention/liquidity/overreaction live), but drop the
            # categories that are poor fade targets (crypto-price tracks an
            # external price; sports move on real in-game info). Polling is free;
            # the daily forecast cap governs spend, not the universe size.
            exclude = {c.lower() for c in self.settings.monitor_exclude_categories}
            try:
                markets = await self.polymarket.get_active_markets(
                    limit=self.settings.monitor_universe_limit, category=None
                )
                dropped = 0
                for m in markets:
                    if classify_category(m.question) in exclude:
                        dropped += 1
                        continue
                    found[m.condition_id] = m
                if exclude:
                    logger.info(
                        "monitor: broad universe dropped %d markets in %s",
                        dropped,
                        sorted(exclude),
                    )
            except Exception as exc:
                logger.warning("monitor: broad universe fetch failed: %s", exc)
        else:
            tags = self.settings.monitor_universe_tags or [None]  # type: ignore[list-item]
            per_tag = max(5, self.settings.monitor_universe_limit // max(1, len(tags)))
            for tag in tags:
                try:
                    markets = await self.polymarket.get_active_markets(
                        limit=per_tag, category=tag
                    )
                    for m in markets:
                        found[m.condition_id] = m
                except Exception as exc:
                    logger.warning("monitor: universe fetch failed for %r: %s", tag, exc)
        # Keep at most universe_limit, dropping already-resolved markets.
        self._universe = {
            cid: m
            for cid, m in list(found.items())[: self.settings.monitor_universe_limit]
            if not m.closed
        }
        self._universe_at = _now()
        logger.info("monitor: watching %d markets", len(self._universe))

    async def _snapshot_prices(self) -> None:
        """Cheap per-cycle price refresh + snapshot for the whole universe."""
        for market in self._universe.values():
            token_ids = [t.token_id for t in market.tokens if t.token_id]
            if not token_ids:
                continue
            try:
                prices = await self.polymarket.get_market_prices(token_ids)
                for token in market.tokens:
                    if token.token_id in prices:
                        token.price = prices[token.token_id]
                await self.repo.save_market_snapshot(market)
            except Exception as exc:
                logger.debug("monitor: snapshot failed for %s: %s", market.condition_id, exc)

    async def _handle_move(self, ev: MoveEvent, market: Market) -> None:
        logger.info(
            "monitor: forecasting mover %s (%s %.0f%%→%.0f%%)",
            market.question[:50],
            ev.outcome,
            ev.ref_price * 100,
            ev.new_price * 100,
        )
        result = await self.engine.analyze_market(market)

        moved = self._find_outcome(result, ev.outcome)
        bot_p = moved.bot_probability if moved else None
        # edge on the moved outcome: negative => the spike overshot fair value.
        edge = (bot_p - ev.new_price) if bot_p is not None else None

        # The fade trade is the cheapest mispriced side the bot would actually
        # buy after the move (best_opportunity is already BUY/STRONG_BUY only).
        best = result.best_opportunity
        is_fade = best is not None

        category = classify_category(result.question)
        await self.repo.save_fade_candidate(
            {
                "condition_id": market.condition_id,
                "market_question": result.question,
                "market_slug": result.slug,
                "category": category,
                "moved_outcome": ev.outcome,
                "ref_price": round(ev.ref_price, 4),
                "new_price": round(ev.new_price, 4),
                "price_move": round(ev.move, 4),
                "bot_probability": round(bot_p, 4) if bot_p is not None else None,
                "edge": round(edge, 4) if edge is not None else None,
                "is_fade": is_fade,
                "fade_outcome": best.outcome if best else None,
                "fade_recommendation": best.recommendation.value if best else None,
                "fade_edge": round(best.edge, 4) if best else None,
                "confidence": result.confidence,
                "reasoning_text": result.reasoning,
            }
        )
        if self.alert:
            try:
                await self.alert(self._format_alert(ev, result, moved, best, category))
            except Exception as exc:
                logger.warning("monitor: alert send failed: %s", exc)

    @staticmethod
    def _find_outcome(result: ForecastResult, name: str) -> OutcomeForecast | None:
        target = name.strip().lower()
        return next(
            (o for o in result.outcomes if o.outcome.strip().lower() == target), None
        )

    @staticmethod
    def _format_alert(
        ev: MoveEvent,
        result: ForecastResult,
        moved: OutcomeForecast | None,
        best: OutcomeForecast | None,
        category: str = "",
    ) -> str:
        arrow = "▲" if ev.move > 0 else "▼"
        tag = f" [{category}]" if category else ""
        lines = [
            f"⚡ <b>Price move detected</b>{tag}",
            f"<b>{result.question[:90]}</b>",
            f"{arrow} <b>{ev.outcome}</b>: {ev.ref_price:.0%} → {ev.new_price:.0%} "
            f"({ev.move:+.0%})",
        ]
        if moved is not None:
            verdict = (
                "looks OVERPRICED — fade candidate"
                if moved.bot_probability < ev.new_price
                else "fair value moved with it — no clear fade"
            )
            lines.append(
                f"Blind fair value: {moved.bot_probability:.0%} → {verdict}"
            )
        if best is not None:
            lines.append(
                f"Cheap side: <b>{best.outcome}</b> {best.recommendation.value} "
                f"(edge {best.edge:+.0%}, Kelly {best.kelly_fraction:.1%})"
            )
        else:
            lines.append("No actionable fade after the bot's read.")
        lines.append("\n<i>Observation-only — paper, not executed.</i>")
        return "\n".join(lines)
