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
    # Question-driven retrieval providers (research layer)
    exa_api_key: str = field(repr=False, default="")
    brave_api_key: str = field(repr=False, default="")
    telegram_authorized_users: list[int] = field(default_factory=list)

    # Polymarket endpoints
    gamma_api_base: str = "https://gamma-api.polymarket.com"
    clob_api_base: str = "https://clob.polymarket.com"

    # Claude model — Opus 4.8 with adaptive thinking for reasoning-heavy forecasting
    claude_model: str = "claude-opus-4-8"
    # Cheap model for question decomposition (kept separate from the forecaster)
    decomposition_model: str = "claude-haiku-4-5"

    # Research layer endpoints
    exa_api_base: str = "https://api.exa.ai"
    brave_llm_context_url: str = "https://api.search.brave.com/res/v1/llm/context"

    # Research layer knobs
    # When true (and provider keys exist), the blind forecast uses the Exa/Brave
    # research layer instead of the news RSS pipeline. Falls back to news if the
    # research layer is unavailable or errors.
    use_research_retrieval: bool = True
    research_freshness_days: int = 30  # current-state facts older than this are dropped
    research_max_facts: int = 40  # hard cap on facts handed to the forecaster
    exa_results_per_query: int = 5
    brave_snippets_per_query: int = 10
    # Prediction-market / betting-odds aggregators — excluded so their published
    # odds never leak a market price into the BLIND forecast (anti-anchoring).
    research_exclude_domains: list[str] = field(
        default_factory=lambda: [
            "polymarket.com",
            "kalshi.com",
            "predictit.org",
            "manifold.markets",
            "metaculus.com",
            "electionbettingodds.com",
            "betfair.com",
            "smarkets.com",
            "oddschecker.com",
            "insightprediction.com",
            "accrue.io",
            "picksbyodds.com",
            "simplefunctions.dev",
            "rekko.ai",
        ]
    )

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
    exa_rpm: int = 60
    brave_rpm: int = 60

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
        use_research = os.environ.get("RESEARCH_RETRIEVAL", "true").strip().lower() not in (
            "false",
            "0",
            "no",
        )
        return cls(
            use_research_retrieval=use_research,
            anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
            newsapi_key=os.environ.get("NEWSAPI_KEY", ""),
            guardian_api_key=os.environ.get("GUARDIAN_API_KEY", ""),
            exa_api_key=os.environ.get("EXA_API_KEY", ""),
            brave_api_key=os.environ.get("BRAVE_API_KEY", ""),
            telegram_bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
            telegram_authorized_users=auth_users,
        )
