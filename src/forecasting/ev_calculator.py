from __future__ import annotations

from src.forecasting.models import OutcomeForecast, Recommendation


def compute_ev(bot_prob: float, market_prob: float) -> float:
    """EV per dollar = bot_probability - market_probability."""
    return bot_prob - market_prob


def compute_kelly(bot_prob: float, market_prob: float) -> float:
    """Kelly criterion fraction.

    b = decimal odds = (1 - market_prob) / market_prob
    f* = (b*p - q) / b
    where p = bot_prob, q = 1 - bot_prob
    """
    if market_prob <= 0 or market_prob >= 1:
        return 0.0
    b = (1.0 - market_prob) / market_prob
    if b <= 0:
        return 0.0
    p = bot_prob
    q = 1.0 - p
    kelly = (b * p - q) / b
    # Clamp: never recommend shorting, cap at half-Kelly for safety
    return max(0.0, kelly * 0.5)


def classify_recommendation(ev: float) -> Recommendation:
    if ev > 0.10:
        return Recommendation.STRONG_BUY
    if ev > 0.05:
        return Recommendation.BUY
    if ev > 0:
        return Recommendation.HOLD
    return Recommendation.AVOID


def evaluate_outcome(
    outcome: str,
    bot_prob: float,
    market_prob: float,
) -> OutcomeForecast:
    ev = compute_ev(bot_prob, market_prob)
    kelly = compute_kelly(bot_prob, market_prob)
    rec = classify_recommendation(ev)
    return OutcomeForecast(
        outcome=outcome,
        bot_probability=bot_prob,
        market_probability=market_prob,
        ev_per_dollar=round(ev, 4),
        kelly_fraction=round(kelly, 4),
        recommendation=rec,
    )
