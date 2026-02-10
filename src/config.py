from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str = field(repr=False)
    newsapi_key: str = field(repr=False)
    telegram_bot_token: str = field(repr=False)
    guardian_api_key: str = field(repr=False, default="")
    telegram_authorized_users: list[int] = field(default_factory=list)

    # Polymarket endpoints
    gamma_api_base: str = "https://gamma-api.polymarket.com"
    clob_api_base: str = "https://clob.polymarket.com"

    # Claude model
    claude_model: str = "claude-sonnet-4-5-20250929"

    # Database — use RAILWAY_VOLUME_MOUNT_PATH if available for persistence
    db_path: str = str(
        Path(os.environ.get("RAILWAY_VOLUME_MOUNT_PATH", ""))
        / "polyforecast.db"
        if os.environ.get("RAILWAY_VOLUME_MOUNT_PATH")
        else Path(__file__).resolve().parent.parent / "polyforecast.db"
    )

    # Rate limits
    anthropic_rpm: int = 30
    newsapi_rpm: int = 100

    # Default categories for market discovery
    default_categories: list[str] = field(
        default_factory=lambda: ["science", "crypto", "politics"]
    )

    @classmethod
    def from_env(cls) -> Settings:
        auth_users_raw = os.environ.get("TELEGRAM_AUTHORIZED_USERS", "")
        auth_users = [
            int(uid.strip())
            for uid in auth_users_raw.split(",")
            if uid.strip().isdigit()
        ]
        return cls(
            anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
            newsapi_key=os.environ.get("NEWSAPI_KEY", ""),
            guardian_api_key=os.environ.get("GUARDIAN_API_KEY", ""),
            telegram_bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
            telegram_authorized_users=auth_users,
        )
