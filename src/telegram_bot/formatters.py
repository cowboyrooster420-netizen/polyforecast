from __future__ import annotations

import io
import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.forecasting.ev_calculator import MIN_TRADABLE_PRICE
from src.forecasting.models import ForecastResult, Recommendation
from src.polymarket.models import Market

logger = logging.getLogger(__name__)

# Telegram HTML formatting helpers

_REC_EMOJI = {
    Recommendation.STRONG_BUY: "!!!",
    Recommendation.BUY: ">>",
    Recommendation.HOLD: "--",
    Recommendation.AVOID: "xx",
}


def format_market_list(markets: list[Market]) -> str:
    if not markets:
        return "No active markets found."

    lines: list[str] = []
    for i, m in enumerate(markets, 1):
        prices = " / ".join(
            f"{_escape(t.outcome)}: {t.price:.0%}" for t in m.tokens if t.price > 0
        )
        vol = f"${m.volume:,.0f}" if m.volume else "n/a"
        lines.append(
            f"<b>{i}.</b> {_escape(m.question)}\n"
            f"   Prices: {prices}\n"
            f"   Volume: {vol}\n"
            f"   <code>{m.slug or m.condition_id[:16]}</code>"
        )
    return "\n\n".join(lines)


def format_forecast(result: ForecastResult) -> str:
    """Format a full forecast analysis for Telegram (HTML)."""
    lines: list[str] = [
        f"<b>{_escape(result.question)}</b>",
    ]

    if result.slug:
        lines.append(f"https://polymarket.com/event/{result.slug}")

    # ── Manual-review alarms (freshness / divergence) ──
    if result.flags:
        lines.append("")
        lines.append("<b>⚠ REVIEW BEFORE SIZING</b>")
        for flag in result.flags:
            lines.append(_escape(flag))

    lines.append("")

    # ── Outcome comparison table ──
    lines.append("<b>PROBABILITY COMPARISON</b>")
    lines.append("<pre>")
    # Dynamically size outcome column based on longest name
    max_name = max((len(of.outcome) for of in result.outcomes), default=7)
    col_w = max(max_name + 1, 8)
    lines.append(f"{'Outcome':<{col_w}} {'Bot':>7} {'Market':>7} {'Edge':>7}")
    lines.append("-" * (col_w + 23))
    for of in result.outcomes:
        # Escape the name (outcomes like "<52m"/">70m" contain HTML-special chars)
        # but pad by visible width so the column still lines up inside <pre>.
        raw = of.outcome[:col_w]
        name = _escape(raw) + " " * max(0, col_w - len(raw))
        if not of.has_market_price:
            lines.append(
                f"{name} {of.bot_probability:>6.1%} {'n/a':>7} {'n/a':>7}"
            )
            continue
        lines.append(
            f"{name} {of.bot_probability:>6.1%} {of.market_probability:>6.1%} {of.edge:>+6.1%}"
        )
    lines.append("</pre>")

    # ── EV & Recommendation ──
    lines.append("\n<b>RECOMMENDATIONS</b>")
    priced = [of for of in result.outcomes if of.has_market_price]
    for of in priced:
        rec_tag = _REC_EMOJI.get(of.recommendation, "")
        # On sub-floor (penny) markets EV-per-dollar is a denominator artifact —
        # show why it's untradable instead of a mirage greenlight number.
        if of.market_probability < MIN_TRADABLE_PRICE:
            ev_line = (
                f"    EV per dollar: n/a "
                f"(price {of.market_probability:.1%} below {MIN_TRADABLE_PRICE:.0%} floor — too thin to trade)"
            )
        else:
            ev_line = f"    EV per dollar: {of.ev_per_dollar:+.1%}"
        lines.append(
            f"  <b>{_escape(of.outcome)}</b>: {of.recommendation.value} {rec_tag}\n"
            f"    Edge: {of.edge:+.1%}\n"
            f"{ev_line}\n"
            f"    Kelly fraction: {of.kelly_fraction:.1%}"
        )
    if not priced:
        lines.append("  No market prices available — EV cannot be computed.")

    best = result.best_opportunity
    if best and best.edge > 0:
        lines.append(
            f"\n<b>Best opportunity: {_escape(best.outcome)}</b> "
            f"(Edge {best.edge:+.1%}, EV {best.ev_per_dollar:+.1%}, "
            f"Kelly {best.kelly_fraction:.1%})"
        )
    else:
        lines.append("\nNo +EV opportunity found — market appears fairly priced.")

    # ── Full reasoning ──
    lines.append("\n<b>ANALYSIS</b>")
    lines.append(f"<i>{_escape(result.reasoning)}</i>")

    if result.key_assumption:
        lines.append(f"\n<b>Key assumption:</b> {_escape(result.key_assumption)}")

    # ── Footer ──
    if result.confidence_label:
        lines.append(f"\nForecast confidence: {result.confidence_label}")
    lines.append(f"News sources used: {result.news_article_count}")

    return "\n".join(lines)


def format_portfolio(
    predictions: list[dict[str, Any]],
    stats: dict[str, Any],
) -> str:
    brier = stats.get("brier_score")
    win_rate = stats.get("win_rate")
    total_markets = stats.get("total_markets", 0)

    lines: list[str] = [
        "<b>Portfolio Summary</b>\n",
        f"Markets analyzed: {total_markets}",
    ]
    if brier is not None:
        lines.append(f"Brier score: {brier:.4f} (lower is better)")
    if win_rate is not None and win_rate.get("win_rate") is not None:
        lines.append(
            f"Win rate (BUY+): {win_rate['wins']}/{win_rate['total']} "
            f"({win_rate['win_rate']:.0%})"
        )

    if predictions:
        lines.append("\n<b>Recent predictions:</b>")
        # Group by condition_id, show latest per market
        seen_conditions: set[str] = set()
        for p in predictions:
            cid = p["condition_id"]
            if cid in seen_conditions:
                continue
            seen_conditions.add(cid)
            resolved_str = "Resolved" if p["resolved"] else "Open"
            lines.append(
                f"\n  {_escape(p['market_question'][:60])}\n"
                f"    {_escape(p['outcome'])}: bot {p['bot_probability']:.0%} vs market {p['market_probability']:.0%}\n"
                f"    Rec: {p['recommendation']} | {resolved_str}"
            )
            if len(seen_conditions) >= 10:
                break
    else:
        lines.append("\nNo predictions yet. Use /analyze to get started.")

    return "\n".join(lines)


def format_calibration_table(buckets: list[dict[str, Any]]) -> str:
    if not buckets:
        return "No resolved predictions yet for calibration data."

    lines: list[str] = ["<b>Calibration Table</b>\n", "<pre>"]
    lines.append(f"{'Bucket':>10} {'Pred':>6} {'Actual':>6} {'Count':>5}")
    lines.append("-" * 30)
    for b in buckets:
        bucket_str = f"{b['bucket_lower']:.0%}-{b['bucket_upper']:.0%}"
        lines.append(
            f"{bucket_str:>10} {b['predicted_avg']:>5.0%} "
            f"{b['actual_frequency']:>5.0%} {b['count']:>5}"
        )
    lines.append("</pre>")
    return "\n".join(lines)


def generate_calibration_chart(buckets: list[dict[str, Any]]) -> bytes | None:
    """Generate a calibration plot and return PNG bytes."""
    if not buckets:
        return None

    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        predicted = [b["predicted_avg"] for b in buckets]
        actual = [b["actual_frequency"] for b in buckets]

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")

        # Actual calibration
        ax.scatter(predicted, actual, s=80, zorder=3, color="#5c6bc0")
        ax.plot(predicted, actual, color="#5c6bc0", alpha=0.7, label="Polyforecast")

        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Frequency")
        ax.set_title("Calibration Plot")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception as exc:
        logger.warning("Failed to generate calibration chart: %s", exc)
        return None


def format_news_articles(articles: list[dict[str, str]]) -> str:
    if not articles:
        return "No articles found."
    lines: list[str] = []
    for i, art in enumerate(articles, 1):
        lines.append(
            f"<b>{i}.</b> {_escape(art.get('title', ''))}\n"
            f"   <i>{_escape(art.get('source', ''))}</i> — "
            f"{art.get('published_at', 'unknown date')}\n"
            f"   {_escape(art.get('url', ''))}"
        )
    return "\n\n".join(lines)


def format_movers(rows: list[dict[str, Any]]) -> str:
    """Render recent price-move / fade candidates from the monitor."""
    if not rows:
        return (
            "No price-move triggers logged yet.\n"
            "The monitor logs sharp moves and the bot's blind read on each."
        )
    lines = ["<b>RECENT PRICE MOVES (paper / observation)</b>\n"]
    for r in rows:
        when = str(r.get("detected_at", ""))[:16]
        q = _escape(str(r.get("market_question", ""))[:70])
        moved = _escape(str(r.get("moved_outcome", "")))
        ref = r.get("ref_price") or 0.0
        new = r.get("new_price") or 0.0
        move = r.get("price_move") or 0.0
        arrow = "▲" if move > 0 else "▼"
        botp = r.get("bot_probability")
        cat = str(r.get("category", "") or "")
        tag = f" [{cat}]" if cat else ""
        line = [
            f"<b>{q}</b>{tag}",
            f"  {arrow} {moved}: {ref:.0%} → {new:.0%} ({move:+.0%})  <i>{when}</i>",
        ]
        if botp is not None:
            verdict = "overpriced → fade" if botp < new else "fair value moved"
            line.append(f"  blind fair value {botp:.0%} ({verdict})")
        if r.get("is_fade") and r.get("fade_outcome"):
            fe = r.get("fade_edge") or 0.0
            line.append(
                f"  → buy <b>{_escape(str(r['fade_outcome']))}</b> "
                f"{r.get('fade_recommendation', '')} (edge {fe:+.0%})"
            )
        if r.get("resolved"):
            line.append(f"  resolved: {_escape(str(r.get('actual_outcome', '?')))}")
        lines.append("\n".join(line))
    return "\n\n".join(lines)


def _escape(text: str) -> str:
    """Escape HTML special chars for Telegram."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
