from __future__ import annotations

import aiosqlite

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    condition_id TEXT NOT NULL,
    market_question TEXT NOT NULL,
    market_slug TEXT NOT NULL DEFAULT '',
    outcome TEXT NOT NULL,
    bot_probability REAL NOT NULL,
    market_probability REAL NOT NULL,
    ev_per_dollar REAL NOT NULL,
    kelly_fraction REAL NOT NULL DEFAULT 0.0,
    recommendation TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.0,
    reasoning_text TEXT NOT NULL DEFAULT '',
    prompt_version TEXT NOT NULL DEFAULT '',
    resolved INTEGER NOT NULL DEFAULT 0,
    actual_outcome TEXT,
    resolution_date TEXT,
    brier_component REAL,
    news_article_count INTEGER NOT NULL DEFAULT 0,
    telegram_user_id INTEGER
);

CREATE TABLE IF NOT EXISTS market_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at TEXT NOT NULL DEFAULT (datetime('now')),
    condition_id TEXT NOT NULL,
    market_question TEXT NOT NULL DEFAULT '',
    outcome TEXT NOT NULL DEFAULT '',
    token_id TEXT NOT NULL DEFAULT '',
    price REAL NOT NULL DEFAULT 0.0,
    volume REAL NOT NULL DEFAULT 0.0,
    liquidity REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS news_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER REFERENCES predictions(id),
    title TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    url TEXT NOT NULL DEFAULT '',
    published_at TEXT,
    description TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS calibration_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    computed_at TEXT NOT NULL DEFAULT (datetime('now')),
    bucket_lower REAL NOT NULL,
    bucket_upper REAL NOT NULL,
    predicted_avg REAL NOT NULL,
    actual_frequency REAL NOT NULL,
    count INTEGER NOT NULL,
    brier_score REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS user_state (
    telegram_user_id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_active TEXT NOT NULL DEFAULT (datetime('now')),
    default_categories TEXT NOT NULL DEFAULT '["science","crypto","politics"]',
    notifications INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_predictions_condition ON predictions(condition_id);
CREATE INDEX IF NOT EXISTS idx_predictions_user ON predictions(telegram_user_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_condition ON market_snapshots(condition_id);
"""


async def get_connection(db_path: str) -> aiosqlite.Connection:
    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA foreign_keys=ON")
    return conn


async def init_db(db_path: str) -> aiosqlite.Connection:
    conn = await get_connection(db_path)
    await conn.executescript(SCHEMA_SQL)
    await conn.commit()
    return conn
