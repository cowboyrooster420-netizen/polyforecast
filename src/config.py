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
    # SDK auto-retries 429/529/5xx with exponential backoff; bump above the
    # default of 2 so transient Anthropic overload doesn't fail a whole analysis.
    anthropic_max_retries: int = 5
    newsapi_rpm: int = 100
    exa_rpm: int = 60
    brave_rpm: int = 60

    # Default categories for market discovery
    default_categories: list[str] = field(
        default_factory=lambda: ["science", "crypto", "politics"]
    )

    # ── Price-move monitor (overreaction-fade watcher) ───────────────────────
    # Polls a universe of markets for sharp price moves, then runs the BLIND
    # forecaster against the movers to see whether fair value actually changed
    # or the move was FOMO. Observation-only: it logs + notifies, never trades.
    monitor_enabled: bool = False
    monitor_poll_minutes: int = 10  # how often to snapshot prices
    monitor_window_hours: float = 6.0  # rolling window a "move" is measured over
    monitor_move_threshold: float = 0.15  # min absolute price move to trigger (15pp)
    # Relative trigger for tail-event overreactions: a low-probability outcome
    # (ref <= relative_max_ref) that spikes UP by >= relative_multiple× to a
    # tradable price. Catches "regime collapses by <date>" jumping 4%→16% — only
    # +12pp absolute (below threshold) but 4× relative — the salience-driven
    # spike that's the bot's best fade. Set relative_multiple very high to disable.
    monitor_relative_multiple: float = 1.8
    monitor_relative_max_ref: float = 0.25
    monitor_relative_min_price: float = 0.08
    monitor_max_forecasts_per_cycle: int = 2  # cost guard: only the N biggest movers
    monitor_daily_forecast_cap: int = 12  # cost guard: hard ceiling on auto-forecasts/day
    monitor_cooldown_hours: float = 12.0  # don't re-trigger same market within this window
    monitor_universe_limit: int = 80  # how many markets to watch
    # Fade detection wants attention/volume (where overreactions happen), so by
    # default we watch the top markets by volume across ALL categories. Polling
    # is free; the daily forecast cap — not the universe size — governs spend.
    # Set monitor_broad_mode=False to instead watch only the fit-tag topics.
    monitor_broad_mode: bool = True
    monitor_universe_tags: list[str] = field(
        default_factory=lambda: ["box-office", "movies", "film", "economy", "awards"]
    )
    # In broad mode, drop these categories from the watched universe. Crypto-price
    # markets track an external price (no overreaction to fade); sports move on
    # real in-game info and resolve too fast. Both dominate raw volume but are
    # poor fade targets, so we exclude them by default.
    monitor_exclude_categories: list[str] = field(
        default_factory=lambda: ["crypto", "sports"]
    )
    # Telegram chat to push trigger alerts to (defaults to first authorized user).
    monitor_alert_chat_id: int | None = None

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
        monitor_enabled = os.environ.get("MONITOR_ENABLED", "false").strip().lower() in (
            "true",
            "1",
            "yes",
        )
        alert_chat_raw = os.environ.get("MONITOR_ALERT_CHAT_ID", "").strip()
        alert_chat = int(alert_chat_raw) if alert_chat_raw.lstrip("-").isdigit() else None
        broad_mode = os.environ.get("MONITOR_BROAD_MODE", "true").strip().lower() not in (
            "false",
            "0",
            "no",
        )
        exclude_raw = os.environ.get("MONITOR_EXCLUDE_CATEGORIES", "").strip()
        exclude_cats = (
            [c.strip().lower() for c in exclude_raw.split(",") if c.strip()]
            if exclude_raw
            else ["crypto", "sports"]
        )
        return cls(
            use_research_retrieval=use_research,
            monitor_enabled=monitor_enabled,
            monitor_broad_mode=broad_mode,
            monitor_exclude_categories=exclude_cats,
            monitor_alert_chat_id=alert_chat,
            anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
            newsapi_key=os.environ.get("NEWSAPI_KEY", ""),
            guardian_api_key=os.environ.get("GUARDIAN_API_KEY", ""),
            exa_api_key=os.environ.get("EXA_API_KEY", ""),
            brave_api_key=os.environ.get("BRAVE_API_KEY", ""),
            telegram_bot_token=os.environ["TELEGRAM_BOT_TOKEN"],
            telegram_authorized_users=auth_users,
        )
