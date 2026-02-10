from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import ContextTypes

from src.telegram_bot.formatters import (
    format_calibration_table,
    format_forecast,
    format_market_list,
    format_news_articles,
    format_portfolio,
    generate_calibration_chart,
)

if TYPE_CHECKING:
    from src.telegram_bot.bot import BotApp

logger = logging.getLogger(__name__)


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
        "<b>/news</b> &lt;topic&gt;\n"
        "  Search for recent news on a topic\n",
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
    await update.message.reply_text("Analyzing... this may take 30-60 seconds.")

    try:
        # Resolve market first
        market = await app.polymarket.get_market(ref)
        if not market:
            await update.message.reply_text("Could not find that market.")
            return

        # Run analysis
        result = await app.engine.analyze_market(market)

        # Save prediction + snapshot
        await app.repo.save_prediction(result, telegram_user_id=user_id)
        await app.repo.save_market_snapshot(market)
        await app.repo.touch_user(user_id)

        text = format_forecast(result)
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


async def _send_long_message(
    update: Update,
    text: str,
    max_len: int = 4000,
) -> None:
    """Split long messages to stay within Telegram's limit."""
    if not update.message:
        return
    if len(text) <= max_len:
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)
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

    for chunk in chunks:
        try:
            await update.message.reply_text(chunk, parse_mode=ParseMode.HTML)
        except Exception:
            # Fallback: send without HTML if parsing fails
            await update.message.reply_text(chunk)
