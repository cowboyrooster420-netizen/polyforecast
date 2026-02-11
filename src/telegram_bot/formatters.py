from __future__ import annotations

import io
import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
            f"{t.outcome}: {t.price:.0%}" for t in m.tokens if t.price > 0
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

    lines.append("")

    # ── Outcome comparison table ──
    lines.append("<b>PROBABILITY COMPARISON</b>")
    lines.append("<pre>")
    lines.append(f"{'Outcome':<12} {'Bot':>7} {'Market':>7} {'Edge':>7}")
    lines.append("-" * 36)
    for of in result.outcomes:
        edge = of.bot_probability - of.market_probability
        lines.append(
            f"{of.outcome:<12} {of.bot_probability:>6.1%} {of.market_probability:>6.1%} {edge:>+6.1%}"
        )
    lines.append("</pre>")

    # ── EV & Recommendation ──
    lines.append("\n<b>RECOMMENDATIONS</b>")
    for of in result.outcomes:
        rec_tag = _REC_EMOJI.get(of.recommendation, "")
        lines.append(
            f"  <b>{_escape(of.outcome)}</b>: {of.recommendation.value} {rec_tag}\n"
            f"    EV per dollar: {of.ev_per_dollar:+.2%}\n"
            f"    Kelly fraction: {of.kelly_fraction:.1%}"
        )

    best = result.best_opportunity
    if best and best.ev_per_dollar > 0:
        lines.append(
            f"\n<b>Best opportunity: {_escape(best.outcome)}</b> "
            f"(EV {best.ev_per_dollar:+.2%}, Kelly {best.kelly_fraction:.1%})"
        )
    else:
        lines.append("\nNo +EV opportunity found — market appears fairly priced.")

    # ── Full reasoning ──
    lines.append("\n<b>ANALYSIS</b>")
    lines.append(f"<i>{_escape(result.reasoning)}</i>")

    # ── Footer ──
    lines.append(f"\nNews sources used: {result.news_article_count}")

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
                f"    {p['outcome']}: bot {p['bot_probability']:.0%} vs market {p['market_probability']:.0%}\n"
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


def _escape(text: str) -> str:
    """Escape HTML special chars for Telegram."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
