from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import anthropic
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import ContextTypes

from src.forecasting.validators import ForecastValidationError
from src.polymarket.models import Market
from src.telegram_bot.formatters import (
    format_calibration_table,
    format_forecast,
    format_market_list,
    format_movers,
    format_news_articles,
    format_portfolio,
    generate_calibration_chart,
)

if TYPE_CHECKING:
    from src.telegram_bot.bot import BotApp

logger = logging.getLogger(__name__)

# Hard ceiling on a single analysis so the handler always replies, even if the
# pipeline hangs or Anthropic stays overloaded through every retry.
ANALYSIS_TIMEOUT_SECONDS = 240


def _infer_winning_outcome(market: Market) -> str | None:
    """Determine the winning outcome of a resolved market.

    Handles both binary markets (resolution string is "Yes"/"No") and merged
    multi-outcome events (winner's token trades at ~1.0 and carries the
    extracted outcome name we stored at prediction time).
    """
    if not market.resolved:
        return None
    # Map the resolution string onto a known outcome name when possible.
    if market.resolution:
        for t in market.tokens:
            if t.outcome.lower() == market.resolution.lower():
                return t.outcome
        return market.resolution
    # Multi-outcome event: the winning outcome's YES token settles near 1.0.
    if market.tokens:
        top = max(market.tokens, key=lambda t: t.price)
        if top.price >= 0.9:
            return top.outcome
    return None


def _get_app(context: ContextTypes.DEFAULT_TYPE) -> BotApp:
    return context.bot_data["app"]  # type: ignore[return-value]


def _authorized(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    app = _get_app(context)
    if not app.settings.telegram_authorized_users:
        return True  # no allowlist = open access
    user_id = update.effective_user.id if update.effective_user else 0
    return user_id in app.settings.telegram_authorized_users


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    if update.effective_user:
        logger.info("User ID: %s  Name: %s", update.effective_user.id, update.effective_user.first_name)
    await update.message.reply_text(
        "<b>Welcome to Polyforecast!</b>\n\n"
        "I'm a superforecasting assistant for Polymarket.\n\n"
        "Commands:\n"
        "/markets [category] - Browse active markets\n"
        "/analyze &lt;url or slug&gt; - Full analysis with EV\n"
        "/setcategories - Set default categories\n"
        "/portfolio - Your tracked predictions\n"
        "/calibration - Calibration chart\n"
        "/news &lt;topic&gt; - Latest news\n"
        "/help - Command reference",
        parse_mode=ParseMode.HTML,
    )


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    await update.message.reply_text(
        "<b>Polyforecast Commands</b>\n\n"
        "<b>/markets</b> [category]\n"
        "  Show top active markets. Categories: politics, crypto, science, sports, finance\n\n"
        "<b>/analyze</b> &lt;url or slug or condition_id&gt;\n"
        "  Run superforecasting analysis. Fetches news, gets Claude's independent estimate, compares to market.\n\n"
        "<b>/setcategories</b> cat1 cat2 ...\n"
        "  Set your default categories for /markets\n\n"
        "<b>/portfolio</b>\n"
        "  View your tracked predictions and accuracy stats\n\n"
        "<b>/calibration</b>\n"
        "  Show calibration table and chart for resolved predictions\n\n"
        "<b>/movers</b>\n"
        "  Recent sharp price moves the monitor caught, with the bot's blind read "
        "and any fade candidate (paper/observation only)\n\n"
        "<b>/news</b> &lt;topic&gt;\n"
        "  Search for recent news on a topic\n\n"
        "<b>/resolve</b> [slug] [outcome]\n"
        "  Resolve a prediction. No args = show unresolved. With slug = auto-check Polymarket. With slug + outcome = manual resolve.\n",
        parse_mode=ParseMode.HTML,
    )


async def markets_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not _authorized(update, context):
        return

    app = _get_app(context)
    user_id = update.effective_user.id if update.effective_user else 0

    await update.message.chat.send_action(ChatAction.TYPING)

    # Determine category
    category: str | None = None
    if context.args:
        category = " ".join(context.args).strip().lower()
    else:
        # Use saved defaults
        categories = await app.repo.get_user_categories(user_id)
        if categories:
            category = categories[0]  # use first saved category

    try:
        markets = await app.polymarket.get_active_markets(limit=10, category=category)
        text = format_market_list(markets)
        if category:
            text = f"<b>Category: {category}</b>\n\n" + text
    except Exception as exc:
        logger.error("Failed to fetch markets: %s", exc)
        text = "Failed to fetch markets. Please try again later."

    await _send_long_message(update, text)


async def analyze_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not _authorized(update, context):
        return

    app = _get_app(context)
    user_id = update.effective_user.id if update.effective_user else 0

    if not context.args:
        await update.message.reply_text(
            "Usage: /analyze &lt;polymarket URL, slug, or condition ID&gt;",
            parse_mode=ParseMode.HTML,
        )
        return

    ref = " ".join(context.args).strip()
    await update.message.chat.send_action(ChatAction.TYPING)
    await update.message.reply_text("Analyzing... this may take up to a minute or two.")

    try:
        # Resolve market first
        market = await app.polymarket.get_market(ref)
        if not market:
            await update.message.reply_text("Could not find that market.")
            return

        # Run analysis with a hard ceiling so a hang or retry storm never
        # leaves the user with the "Analyzing..." message and no reply.
        result = await asyncio.wait_for(
            app.engine.analyze_market(market), timeout=ANALYSIS_TIMEOUT_SECONDS
        )

        # Save prediction + snapshot (include the articles that informed it)
        await app.repo.save_prediction(
            result, articles=result.articles, telegram_user_id=user_id
        )
        await app.repo.save_market_snapshot(market)
        await app.repo.touch_user(user_id)

        text = format_forecast(result)
    except asyncio.TimeoutError:
        logger.error("Analysis timed out after %ss for: %s", ANALYSIS_TIMEOUT_SECONDS, ref)
        await update.message.reply_text(
            "Analysis timed out — Claude may be overloaded. Please try /analyze again in a minute."
        )
        return
    except (anthropic.OverloadedError, anthropic.RateLimitError, anthropic.InternalServerError):
        logger.warning("Anthropic temporarily unavailable during analysis", exc_info=True)
        await update.message.reply_text(
            "Claude is temporarily overloaded. Please try /analyze again in a minute."
        )
        return
    except ForecastValidationError as exc:
        logger.error("Forecast failed validation for %s: %s", ref, exc)
        await update.message.reply_text(
            "⛔ Forecast failed a sanity check and was withheld "
            f"(not logged):\n{exc}\n\nThis is a logic bug, not a market call — "
            "please report it."
        )
        return
    except Exception as exc:
        logger.error("Analysis failed: %s", exc, exc_info=True)
        await update.message.reply_text(f"Analysis failed: {exc}")
        return

    await _send_long_message(update, text)


async def setcategories_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    if not update.message or not _authorized(update, context):
        return

    app = _get_app(context)
    user_id = update.effective_user.id if update.effective_user else 0

    valid = {"politics", "crypto", "science", "sports", "finance", "entertainment"}

    if not context.args:
        current = await app.repo.get_user_categories(user_id)
        await update.message.reply_text(
            f"Current categories: {', '.join(current)}\n\n"
            f"Usage: /setcategories cat1 cat2 ...\n"
            f"Valid: {', '.join(sorted(valid))}",
        )
        return

    chosen = [a.lower().strip() for a in context.args if a.lower().strip() in valid]
    if not chosen:
        await update.message.reply_text(f"No valid categories. Choose from: {', '.join(sorted(valid))}")
        return

    await app.repo.set_user_categories(user_id, chosen)
    await update.message.reply_text(f"Categories saved: {', '.join(chosen)}")


async def portfolio_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not _authorized(update, context):
        return

    app = _get_app(context)
    user_id = update.effective_user.id if update.effective_user else 0

    await update.message.chat.send_action(ChatAction.TYPING)

    predictions = await app.repo.get_predictions_for_user(user_id)
    brier = await app.repo.get_brier_score(user_id)
    win_rate = await app.repo.get_win_rate(user_id)
    total_markets = await app.repo.get_prediction_count(user_id)

    stats = {
        "brier_score": brier,
        "win_rate": win_rate,
        "total_markets": total_markets,
    }
    text = format_portfolio(predictions, stats)
    await _send_long_message(update, text)


async def movers_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not _authorized(update, context):
        return
    app = _get_app(context)
    await update.message.chat.send_action(ChatAction.TYPING)
    rows = await app.repo.get_recent_fade_candidates(limit=15)
    await _send_long_message(update, format_movers(rows))


async def calibration_handler(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    if not update.message or not _authorized(update, context):
        return

    app = _get_app(context)
    user_id = update.effective_user.id if update.effective_user else 0

    await update.message.chat.send_action(ChatAction.TYPING)

    buckets = await app.repo.get_calibration_data(user_id)
    text = format_calibration_table(buckets)
    await _send_long_message(update, text)

    # Send chart if we have data
    chart_bytes = generate_calibration_chart(buckets)
    if chart_bytes:
        await update.message.reply_photo(photo=chart_bytes, caption="Calibration plot")


async def news_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not _authorized(update, context):
        return

    app = _get_app(context)

    if not context.args:
        await update.message.reply_text("Usage: /news &lt;topic&gt;", parse_mode=ParseMode.HTML)
        return

    topic = " ".join(context.args).strip()
    await update.message.chat.send_action(ChatAction.TYPING)

    try:
        articles = await app.news.search_topic(topic, max_articles=10)
        article_dicts = [
            {
                "title": a.title,
                "source": a.source,
                "url": a.url,
                "published_at": (
                    a.published_at.strftime("%Y-%m-%d") if a.published_at else "unknown"
                ),
            }
            for a in articles
        ]
        text = format_news_articles(article_dicts)
    except Exception as exc:
        logger.error("News fetch failed: %s", exc)
        text = "Failed to fetch news. Please try again."

    await _send_long_message(update, text)


async def resolve_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not _authorized(update, context):
        return

    app = _get_app(context)
    user_id = update.effective_user.id if update.effective_user else 0

    # No args → show unresolved predictions
    if not context.args:
        await update.message.chat.send_action(ChatAction.TYPING)
        unresolved = await app.repo.get_unresolved_predictions(user_id)
        if not unresolved:
            await update.message.reply_text("No unresolved predictions.")
            return
        lines = ["<b>Unresolved predictions:</b>\n"]
        for p in unresolved:
            lines.append(
                f"  <code>{p['market_slug'] or p['condition_id'][:16]}</code>\n"
                f"  {p['market_question'][:60]}\n"
            )
        lines.append(
            "\nTo resolve: /resolve &lt;slug&gt;\n"
            "(Auto-checks Polymarket for result)\n\n"
            "Or manually: /resolve &lt;slug&gt; &lt;winning outcome&gt;"
        )
        await _send_long_message(update, "\n".join(lines))
        return

    ref = context.args[0].strip()
    manual_outcome = " ".join(context.args[1:]).strip() if len(context.args) > 1 else None

    await update.message.chat.send_action(ChatAction.TYPING)

    # Find the condition_id for this slug
    unresolved = await app.repo.get_unresolved_predictions(user_id)
    match = None
    for p in unresolved:
        if ref.lower() in (
            (p.get("market_slug") or "").lower(),
            (p.get("condition_id") or "").lower(),
        ):
            match = p
            break

    if not match:
        await update.message.reply_text(
            f"No unresolved prediction found for '{ref}'.\n"
            "Use /resolve with no args to see your unresolved predictions."
        )
        return

    condition_id = match["condition_id"]
    winning_outcome = manual_outcome

    # Auto-check Polymarket if no manual outcome given
    if not winning_outcome:
        # Prefer the slug for re-fetch: merged multi-outcome events are stored
        # under a numeric event id that won't resolve as a condition_id or slug.
        fetch_ref = match.get("market_slug") or condition_id
        try:
            market = await app.polymarket.get_market(fetch_ref)
            if market:
                winning_outcome = _infer_winning_outcome(market)
        except Exception as exc:
            logger.warning("Failed to auto-resolve from Polymarket: %s", exc)

    if not winning_outcome:
        await update.message.reply_text(
            f"Market not yet resolved on Polymarket.\n"
            f"Resolve manually: /resolve {ref} &lt;winning outcome&gt;\n\n"
            f"Outcomes: {match.get('outcomes', 'unknown')}",
            parse_mode=ParseMode.HTML,
        )
        return

    # Resolve in database
    count = await app.repo.resolve_prediction(condition_id, winning_outcome)
    brier = await app.repo.get_brier_score(user_id)
    brier_str = f"\nOverall Brier score: {brier:.4f}" if brier is not None else ""

    await update.message.reply_text(
        f"Resolved {count} predictions for:\n"
        f"<b>{match['market_question'][:80]}</b>\n"
        f"Winner: <b>{winning_outcome}</b>{brier_str}",
        parse_mode=ParseMode.HTML,
    )


async def _send_long_message(
    update: Update,
    text: str,
    max_len: int = 4000,
) -> None:
    """Split long messages to stay within Telegram's limit."""
    if not update.message:
        return
    if len(text) <= max_len:
        try:
            await update.message.reply_text(text, parse_mode=ParseMode.HTML)
        except Exception:
            # Bad HTML (e.g. an unescaped char) must never swallow the result —
            # fall back to plain text so the forecast is always delivered.
            logger.warning("HTML send failed; retrying as plain text", exc_info=True)
            await update.message.reply_text(text)
        return

    # Split on double newlines or force-split
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break
        # Try to split at a paragraph boundary
        split_pos = remaining.rfind("\n\n", 0, max_len)
        if split_pos == -1:
            split_pos = remaining.rfind("\n", 0, max_len)
        if split_pos == -1:
            split_pos = max_len
        chunks.append(remaining[:split_pos])
        remaining = remaining[split_pos:].lstrip("\n")

    # Fix unclosed HTML tags across chunks so Telegram doesn't reject them
    open_tags = {
        "<i>": "</i>",
        "<b>": "</b>",
        "<pre>": "</pre>",
        "<code>": "</code>",
    }
    for i, chunk in enumerate(chunks):
        # Check for unclosed tags — count opens vs closes
        for open_tag, close_tag in open_tags.items():
            opens = chunk.count(open_tag)
            closes = chunk.count(close_tag)
            if opens > closes:
                # Close tag at end of this chunk, reopen at start of next
                chunks[i] = chunk + close_tag
                if i + 1 < len(chunks):
                    chunks[i + 1] = open_tag + chunks[i + 1]

    for idx, chunk in enumerate(chunks):
        suffix = f"\n\n<i>({idx + 1}/{len(chunks)})</i>" if len(chunks) > 1 else ""
        try:
            await update.message.reply_text(
                chunk + suffix, parse_mode=ParseMode.HTML
            )
        except Exception:
            # Fallback: send without HTML if parsing fails
            await update.message.reply_text(chunk)
