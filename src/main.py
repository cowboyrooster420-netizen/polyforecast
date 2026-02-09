from __future__ import annotations

import asyncio
import logging
import sys

from src.config import Settings
from src.database.db import init_db
from src.database.repository import Repository
from src.forecasting.engine import ForecastingEngine
from src.news.client import NewsClient
from src.polymarket.client import PolymarketClient
from src.telegram_bot.bot import BotApp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


async def _run() -> None:
    settings = Settings.from_env()
    logger.info("Starting Polyforecast bot...")

    # Initialise database
    conn = await init_db(settings.db_path)
    repo = Repository(conn)

    # Initialise clients
    polymarket = PolymarketClient(settings)
    news = NewsClient(settings)
    engine = ForecastingEngine(settings, polymarket, news)

    # Build and run Telegram bot
    bot_app = BotApp(settings, polymarket, news, engine, repo)
    application = bot_app.build_application()

    logger.info("Bot is ready. Polling for updates...")
    try:
        await application.initialize()
        await application.start()
        await application.updater.start_polling()

        # Run until interrupted
        stop_event = asyncio.Event()
        try:
            await stop_event.wait()
        except (KeyboardInterrupt, SystemExit):
            pass
    finally:
        logger.info("Shutting down...")
        await application.updater.stop()
        await application.stop()
        await application.shutdown()
        await polymarket.close()
        await news.close()
        await conn.close()
        logger.info("Shutdown complete.")


def main() -> None:
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
