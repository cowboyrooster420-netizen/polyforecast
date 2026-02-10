from __future__ import annotations

import asyncio
import logging
import signal
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

    # Retry initialization in case of transient network issues (e.g. Railway cold start)
    for attempt in range(5):
        try:
            logger.info("Initializing bot (attempt %d/5)...", attempt + 1)
            await application.initialize()
            break
        except Exception as exc:
            if attempt < 4:
                wait = 2 ** (attempt + 1)
                logger.warning("Init failed: %s — retrying in %ds", exc, wait)
                await asyncio.sleep(wait)
            else:
                logger.error("Failed to initialize after 5 attempts: %s", exc)
                raise

    await application.start()
    await application.updater.start_polling()
    logger.info("Bot is live. Polling for updates...")

    # Run until interrupted
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass  # Windows

    await stop_event.wait()

    # Graceful shutdown
    logger.info("Shutting down...")
    try:
        if application.updater and application.updater.running:
            await application.updater.stop()
        if application.running:
            await application.stop()
        await application.shutdown()
    except Exception as exc:
        logger.warning("Shutdown error (non-fatal): %s", exc)
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
