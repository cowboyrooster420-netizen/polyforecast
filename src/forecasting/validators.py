from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.forecasting.ev_calculator import compute_ev_per_dollar
from src.forecasting.models import ForecastResult, Recommendation
from src.polymarket.models import Market


class ForecastValidationError(Exception):
    """A forecast violated a hard logic invariant and must NOT be acted on.

    These are deterministic bugs (probabilities that don't sum to 1, a dropped
    leg, an EV that doesn't match its inputs, a self-contradicting summary) —
    pure logic, no judgment. Better to fail loudly than to surface a wrong call.
    """


@dataclass
class ValidationReport:
    """Result of the soft (judgment-adjacent) gates that flag rather than raise."""

    flags: list[str] = field(default_factory=list)
    downgrade_confidence: bool = False


def _norm(s: str) -> str:
    return s.strip().lower()


def _as_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ── Hard assertions (raise) ──────────────────────────────────────────────────


def assert_probabilities_sum_to_one(
    result: ForecastResult, tolerance: float = 0.03
) -> None:
    total = sum(o.bot_probability for o in result.outcomes)
    if abs(total - 1.0) > tolerance:
        raise ForecastValidationError(
            f"Probabilities sum to {total:.3f}, not ~1.0 "
            f"(±{tolerance}) across {len(result.outcomes)} outcomes."
        )


def assert_all_legs_scored(result: ForecastResult, market: Market) -> None:
    """Every outcome the market offers must be scored — catches the YES-only /
    dropped-leg blind spot where a short opportunity silently vanishes."""
    offered = {_norm(t.outcome) for t in market.tokens if t.outcome.strip()}
    scored = {_norm(o.outcome) for o in result.outcomes}
    missing = offered - scored
    if missing:
        raise ForecastValidationError(
            f"Market offers outcomes that were never scored: {sorted(missing)}. "
            "Every leg (incl. the NO/short side) must be evaluated."
        )


def assert_ev_matches_inputs(result: ForecastResult, tolerance: float = 1e-3) -> None:
    """Recompute EV independently from (p, price) and assert it matches the
    stored value — guards the display/calc bug from ever regressing."""
    for o in result.outcomes:
        if not o.has_market_price:
            continue
        expected = compute_ev_per_dollar(o.bot_probability, o.market_probability)
        if abs(expected - o.ev_per_dollar) > tolerance:
            raise ForecastValidationError(
                f"EV mismatch on {o.outcome!r}: stored {o.ev_per_dollar:+.4f} but "
                f"(p={o.bot_probability:.3f}, price={o.market_probability:.3f}) "
                f"implies {expected:+.4f}."
            )


def assert_summary_not_contradictory(result: ForecastResult) -> None:
    """If any leg is a BUY/STRONG_BUY, the summary cannot also claim there's no
    +EV opportunity (best_opportunity is None) — the Letlow contradiction."""
    has_buy = any(
        o.recommendation in (Recommendation.BUY, Recommendation.STRONG_BUY)
        for o in result.outcomes
    )
    if has_buy and result.best_opportunity is None:
        raise ForecastValidationError(
            "Summary contradiction: a leg is BUY/STRONG_BUY but best_opportunity "
            "is None ('no +EV / fairly priced')."
        )


def assert_resolution_in_future(market: Market, now: datetime) -> None:
    if market.end_date is None:
        return  # absence handled as a soft flag elsewhere
    if _as_utc(market.end_date) <= _as_utc(now):
        raise ForecastValidationError(
            f"Resolution date {market.end_date:%Y-%m-%d} is not in the future "
            f"(now {now:%Y-%m-%d}); market may already be resolved."
        )


# ── Soft gates (flag / downgrade) ────────────────────────────────────────────


def _days_to_resolution(market: Market, now: datetime) -> float | None:
    if market.end_date is None:
        return None
    return (_as_utc(market.end_date) - _as_utc(now)).total_seconds() / 86400.0


def check_divergence(result: ForecastResult, threshold: float) -> list[str]:
    """A large bot-vs-market gap on a liquid event market usually means the
    market knows something you don't (e.g. a poll) — treat it as an alarm
    requiring a manual look, not an automatic green light."""
    flags: list[str] = []
    worst = None
    for o in result.outcomes:
        if not o.has_market_price:
            continue
        if abs(o.edge) >= threshold and (worst is None or abs(o.edge) > abs(worst.edge)):
            worst = o
    if worst is not None:
        flags.append(
            f"⚠ Large divergence on {worst.outcome!r}: bot {worst.bot_probability:.0%} "
            f"vs market {worst.market_probability:.0%} ({worst.edge:+.0%}). On a "
            "liquid market a big gap often means the market knows something — "
            "manual look before sizing."
        )
    return flags


def check_freshness(
    result: ForecastResult, market: Market, now: datetime, near_days: int = 7
) -> tuple[list[str], bool]:
    """For a market resolving soon, the forecast must rest on recent sources.
    If the freshest source is older than the time-to-resolution window, that's a
    staleness alarm: flag it and downgrade confidence. (This is the cheap
    precursor to a full source-date gate.)"""
    days = _days_to_resolution(market, now)
    if days is None:
        return (["⚠ No resolution date on the market — verify the deadline manually."], False)
    if days > near_days:
        return ([], False)

    dated = [a.published_at for a in result.articles if a.published_at is not None]
    if not dated:
        return (
            [
                f"⚠ Resolves in {days:.0f}d but no dated sources backed the forecast "
                "— do a 60-second fresh-source check before trading."
            ],
            True,
        )
    freshest = max(_as_utc(d) for d in dated)
    age_days = (_as_utc(now) - freshest).total_seconds() / 86400.0
    if age_days > near_days:
        return (
            [
                f"⚠ Resolves in {days:.0f}d but the freshest source is {age_days:.0f}d "
                "old — likely stale; confidence downgraded, manual poll check advised."
            ],
            True,
        )
    return ([], False)


# ── Orchestration ────────────────────────────────────────────────────────────


def validate_forecast(
    result: ForecastResult,
    market: Market,
    *,
    now: datetime | None = None,
    ev_tolerance: float = 1e-3,
    prob_tolerance: float = 0.03,
    divergence_threshold: float = 0.25,
    near_resolution_days: int = 7,
) -> ValidationReport:
    """Run hard assertions (raise on violation) then soft gates (flag/downgrade).

    Returns a ValidationReport with human-facing flags and a confidence-downgrade
    signal. Raises ForecastValidationError on any hard logic bug.
    """
    now = now or datetime.now(tz=timezone.utc)

    # Hard — deterministic logic, no judgment.
    assert_probabilities_sum_to_one(result, prob_tolerance)
    assert_all_legs_scored(result, market)
    assert_ev_matches_inputs(result, ev_tolerance)
    assert_summary_not_contradictory(result)
    assert_resolution_in_future(market, now)

    # Soft — flag for manual review, downgrade where warranted.
    report = ValidationReport()
    report.flags.extend(check_divergence(result, divergence_threshold))
    fresh_flags, downgrade = check_freshness(result, market, now, near_resolution_days)
    report.flags.extend(fresh_flags)
    report.downgrade_confidence = downgrade
    return report
