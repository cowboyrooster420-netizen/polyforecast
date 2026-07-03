"""Deterministic pre-commit guards. Every bug we surface becomes a case here.

Run: python -m pytest tests/test_validators.py   (or: python tests/test_validators.py)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.forecasting.ev_calculator import evaluate_outcome
from src.forecasting.models import ForecastResult, Recommendation
from src.forecasting.validators import (
    ForecastValidationError,
    assert_all_legs_scored,
    assert_ev_matches_inputs,
    assert_probabilities_sum_to_one,
    assert_resolution_in_future,
    assert_summary_not_contradictory,
    check_divergence,
    check_freshness,
    validate_forecast,
)
from src.news.models import Article
from src.polymarket.models import Market, Token

NOW = datetime(2026, 6, 26, tzinfo=timezone.utc)


def _market(outcomes, days_to_resolution=30):
    return Market(
        condition_id="c",
        question="Will X happen?",
        end_date=NOW + timedelta(days=days_to_resolution),
        tokens=[Token(token_id=str(i), outcome=o, price=p) for i, (o, p) in enumerate(outcomes)],
    )


def _result(outcomes, articles=None, **kw):
    ofs = [evaluate_outcome(o, p_bot, p_mkt) for o, p_bot, p_mkt in outcomes]
    return ForecastResult(
        condition_id="c", question="Will X happen?", slug="x", reasoning="r",
        outcomes=ofs, articles=articles or [], **kw,
    )


# ── sum to one ───────────────────────────────────────────────────────────────

def test_probabilities_sum_ok():
    assert_probabilities_sum_to_one(_result([("Yes", 0.6, 0.5), ("No", 0.4, 0.5)]))


def test_probabilities_sum_bad_raises():
    with pytest.raises(ForecastValidationError, match="sum to"):
        assert_probabilities_sum_to_one(_result([("Yes", 0.6, 0.5), ("No", 0.1, 0.5)]))


# ── all legs scored (YES-only / dropped-leg blind spot) ──────────────────────

def test_all_legs_scored_ok():
    m = _market([("Yes", 0.5), ("No", 0.5)])
    assert_all_legs_scored(_result([("Yes", 0.6, 0.5), ("No", 0.4, 0.5)]), m)


def test_dropped_no_leg_raises():
    m = _market([("Yes", 0.5), ("No", 0.5)])
    res = _result([("Yes", 1.0, 0.5)])  # NO leg never scored
    with pytest.raises(ForecastValidationError, match="never scored"):
        assert_all_legs_scored(res, m)


# ── EV matches inputs (the display/calc bug) ─────────────────────────────────

def test_ev_matches_ok():
    assert_ev_matches_inputs(_result([("Yes", 0.34, 0.07), ("No", 0.66, 0.93)]))


def test_ev_tampered_raises():
    res = _result([("Yes", 0.34, 0.07), ("No", 0.66, 0.93)])
    res.outcomes[0].ev_per_dollar = 0.27  # the old bug: edge, not EV/dollar
    with pytest.raises(ForecastValidationError, match="EV mismatch"):
        assert_ev_matches_inputs(res)


# ── summary self-contradiction (Letlow) ──────────────────────────────────────

def test_summary_consistent_ok():
    assert_summary_not_contradictory(_result([("Yes", 0.34, 0.07), ("No", 0.66, 0.93)]))


def test_summary_contradiction_raises():
    buy_leg = SimpleNamespace(recommendation=Recommendation.STRONG_BUY)
    fake = SimpleNamespace(outcomes=[buy_leg], best_opportunity=None)
    with pytest.raises(ForecastValidationError, match="contradiction"):
        assert_summary_not_contradictory(fake)  # type: ignore[arg-type]


# ── resolution in the future / wrong date ────────────────────────────────────

def test_resolution_future_ok():
    assert_resolution_in_future(_market([("Yes", 0.5)], days_to_resolution=10), NOW)


def test_resolution_past_raises():
    with pytest.raises(ForecastValidationError, match="not in the future"):
        assert_resolution_in_future(_market([("Yes", 0.5)], days_to_resolution=-3), NOW)


# ── divergence alarm (Louisiana) ─────────────────────────────────────────────

def test_large_divergence_flags():
    flags = check_divergence(_result([("Yes", 0.20, 0.55), ("No", 0.80, 0.45)]), 0.25)
    assert flags and "divergence" in flags[0].lower()


def test_small_divergence_quiet():
    assert check_divergence(_result([("Yes", 0.52, 0.50), ("No", 0.48, 0.50)]), 0.25) == []


# ── freshness gate ───────────────────────────────────────────────────────────

def test_stale_near_resolution_downgrades():
    stale = [Article(title="t", published_at=NOW - timedelta(days=40))]
    m = _market([("Yes", 0.5), ("No", 0.5)], days_to_resolution=4)
    flags, downgrade = check_freshness(_result([("Yes", 0.6, 0.5), ("No", 0.4, 0.5)], stale), m, NOW)
    assert downgrade and flags


def test_fresh_near_resolution_ok():
    fresh = [Article(title="t", published_at=NOW - timedelta(days=1))]
    m = _market([("Yes", 0.5), ("No", 0.5)], days_to_resolution=4)
    flags, downgrade = check_freshness(_result([("Yes", 0.6, 0.5), ("No", 0.4, 0.5)], fresh), m, NOW)
    assert not downgrade and not flags


def test_far_resolution_skips_freshness():
    m = _market([("Yes", 0.5), ("No", 0.5)], days_to_resolution=60)
    flags, downgrade = check_freshness(_result([("Yes", 0.6, 0.5), ("No", 0.4, 0.5)], []), m, NOW)
    assert not downgrade and not flags


# ── end-to-end ───────────────────────────────────────────────────────────────

def test_validate_clean_forecast_passes():
    m = _market([("Yes", 0.5), ("No", 0.5)], days_to_resolution=30)
    res = _result([("Yes", 0.6, 0.5), ("No", 0.4, 0.5)])
    report = validate_forecast(res, m, now=NOW)
    assert report.flags == [] and not report.downgrade_confidence


def test_validate_stale_diverging_forecast_flags_both():
    stale = [Article(title="t", published_at=NOW - timedelta(days=40))]
    m = _market([("Yes", 0.5), ("No", 0.5)], days_to_resolution=3)
    res = _result([("Yes", 0.20, 0.55), ("No", 0.80, 0.45)], stale)
    report = validate_forecast(res, m, now=NOW)
    assert report.downgrade_confidence
    assert any("divergence" in f.lower() for f in report.flags)
    assert any("stale" in f.lower() or "freshest" in f.lower() for f in report.flags)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
