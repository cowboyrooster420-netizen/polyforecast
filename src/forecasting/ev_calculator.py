from __future__ import annotations

from src.forecasting.models import OutcomeForecast, Recommendation


def compute_edge(bot_prob: float, market_prob: float) -> float:
    """Probability edge = bot_probability - market_probability.

    This is the directional advantage in probability space. It drives the
    recommendation thresholds and opportunity ranking — NOT the same thing as
    expected value per dollar staked (see compute_ev_per_dollar).
    """
    return bot_prob - market_prob


def compute_ev_per_dollar(bot_prob: float, market_prob: float) -> float:
    """Expected profit per $1 staked on YES at the market price.

    Buying YES at price `market_prob` gets you 1/market_prob shares per dollar,
    each paying $1 with probability `bot_prob`:

        EV = bot_prob * (1 / market_prob) - 1 = (bot_prob - market_prob) / market_prob

    So a $0.07 YES you believe is 34% has EV ≈ +386% per dollar, not +27%
    (the +27% is the probability *edge*). Bounded below by -100%.
    """
    if market_prob <= 0:
        return 0.0
    return (bot_prob - market_prob) / market_prob


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


def classify_recommendation(edge: float) -> Recommendation:
    """Classify on the probability edge (bot_prob - market_prob)."""
    if edge > 0.10:
        return Recommendation.STRONG_BUY
    if edge > 0.05:
        return Recommendation.BUY
    if edge > 0:
        return Recommendation.HOLD
    return Recommendation.AVOID


def evaluate_outcome(
    outcome: str,
    bot_prob: float,
    market_prob: float,
    has_market_price: bool = True,
) -> OutcomeForecast:
    # No market price → we can't compute a real edge. Record the bot's
    # probability but emit a neutral, non-actionable recommendation rather
    # than inventing an EV against a fabricated price.
    if not has_market_price:
        return OutcomeForecast(
            outcome=outcome,
            bot_probability=bot_prob,
            market_probability=0.0,
            ev_per_dollar=0.0,
            kelly_fraction=0.0,
            recommendation=Recommendation.AVOID,
            has_market_price=False,
        )

    edge = compute_edge(bot_prob, market_prob)
    ev = compute_ev_per_dollar(bot_prob, market_prob)
    kelly = compute_kelly(bot_prob, market_prob)
    rec = classify_recommendation(edge)
    return OutcomeForecast(
        outcome=outcome,
        bot_probability=bot_prob,
        market_probability=market_prob,
        ev_per_dollar=round(ev, 4),
        kelly_fraction=round(kelly, 4),
        recommendation=rec,
    )
