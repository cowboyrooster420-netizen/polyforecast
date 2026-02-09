from __future__ import annotations

import logging

from telegram.ext import ApplicationBuilder, CommandHandler

from src.config import Settings
from src.database.repository import Repository
from src.forecasting.engine import ForecastingEngine
from src.news.client import NewsClient
from src.polymarket.client import PolymarketClient
from src.telegram_bot.handlers import (
    analyze_handler,
    calibration_handler,
    help_handler,
    markets_handler,
    news_handler,
    portfolio_handler,
    setcategories_handler,
    start_handler,
)

logger = logging.getLogger(__name__)


class BotApp:
    """Holds shared resources and wires up the Telegram application."""

    def __init__(
        self,
        settings: Settings,
        polymarket: PolymarketClient,
        news: NewsClient,
        engine: ForecastingEngine,
        repo: Repository,
    ) -> None:
        self.settings = settings
        self.polymarket = polymarket
        self.news = news
        self.engine = engine
        self.repo = repo

    def build_application(self):
        application = (
            ApplicationBuilder()
            .token(self.settings.telegram_bot_token)
            .build()
        )

        # Inject self into bot_data so handlers can access shared resources
        application.bot_data["app"] = self

        # Register command handlers
        application.add_handler(CommandHandler("start", start_handler))
        application.add_handler(CommandHandler("help", help_handler))
        application.add_handler(CommandHandler("markets", markets_handler))
        application.add_handler(CommandHandler("analyze", analyze_handler))
        application.add_handler(CommandHandler("setcategories", setcategories_handler))
        application.add_handler(CommandHandler("portfolio", portfolio_handler))
        application.add_handler(CommandHandler("calibration", calibration_handler))
        application.add_handler(CommandHandler("news", news_handler))

        return application
