from __future__ import annotations

from src.forecasting.models import OutcomeForecast, Recommendation

# ── Recommendation / sizing policy ───────────────────────────────────────────
# The recommendation is driven by the probability EDGE (bot_prob - market_prob),
# never by EV-per-dollar. EV-per-dollar is price-scaled, so on a penny market a
# 1pt edge reads as +100% EV — a mirage that must never greenlight a position.

# Below this market price, EV-per-dollar is dominated by the tiny denominator
# and the order book is too thin to fill at size. Such outcomes are never
# actionable regardless of how large the nominal edge or EV looks.
MIN_TRADABLE_PRICE = 0.05  # 5¢

# Edge tiers. A ~12-13pt edge on a single bracket is a small BUY, not a
# STRONG_BUY — STRONG_BUY is reserved for edges large enough that even a
# mis-estimate leaves real value.
STRONG_BUY_EDGE = 0.20
BUY_EDGE = 0.07

# Sizing: quarter-Kelly, hard-capped. A ~13pt edge should be a small nibble
# (~3% of bankroll), not a 6-7% slug, and no single outcome exceeds the cap.
KELLY_FRACTION_MULTIPLIER = 0.25
MAX_KELLY = 0.05  # 5% of bankroll, hard ceiling per outcome


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
    (the +27% is the probability *edge*). Bounded below by -100%. Note this
    inflates without bound as the price falls — which is exactly why it must
    NOT drive recommendations (see MIN_TRADABLE_PRICE).
    """
    if market_prob <= 0:
        return 0.0
    return (bot_prob - market_prob) / market_prob


def compute_kelly(bot_prob: float, market_prob: float) -> float:
    """Quarter-Kelly criterion fraction, hard-capped at MAX_KELLY.

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
    # Never recommend shorting; fractional-Kelly for safety; hard cap.
    return max(0.0, min(kelly * KELLY_FRACTION_MULTIPLIER, MAX_KELLY))


def classify_recommendation(edge: float, market_prob: float) -> Recommendation:
    """Classify on the probability edge, gated by a minimum tradable price.

    Penny-priced outcomes are never actionable: their EV-per-dollar is a
    mirage and the book is too thin to fill, so a positive edge there is not a
    buy. Above the price floor, tiers gate purely on edge.
    """
    if market_prob < MIN_TRADABLE_PRICE:
        return Recommendation.AVOID
    if edge > STRONG_BUY_EDGE:
        return Recommendation.STRONG_BUY
    if edge > BUY_EDGE:
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
    rec = classify_recommendation(edge, market_prob)
    # Don't size a position we've judged non-actionable (negative edge or a
    # sub-floor longshot), even though the raw Kelly math would be positive.
    kelly = compute_kelly(bot_prob, market_prob) if rec != Recommendation.AVOID else 0.0
    return OutcomeForecast(
        outcome=outcome,
        bot_probability=bot_prob,
        market_probability=market_prob,
        ev_per_dollar=round(ev, 4),
        kelly_fraction=round(kelly, 4),
        recommendation=rec,
    )
