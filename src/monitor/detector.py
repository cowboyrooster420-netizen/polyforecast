from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class MoveEvent:
    """A qualifying price move for one market outcome over the window."""

    condition_id: str
    outcome: str
    ref_price: float  # earliest price in the window
    new_price: float  # latest price in the window
    move: float  # new_price - ref_price (signed)

    @property
    def magnitude(self) -> float:
        return abs(self.move)


def detect_moves(
    price_rows: Iterable[Mapping[str, object]],
    threshold: float,
    min_price: float = 0.05,
    max_price: float = 0.97,
    relative_multiple: float = 1.8,
    relative_max_ref: float = 0.25,
    relative_min_price: float = 0.08,
) -> list[MoveEvent]:
    """Find sharp price moves, one (the largest) per market.

    `price_rows` is the window's price history: each row must have
    `condition_id`, `outcome`, `captured_at` (ISO8601 string, sortable), and
    `price`. Rows are expected to already be scoped to the rolling window; this
    function only needs them grouped, not time-filtered.

    For each (market, outcome) we compare the earliest and latest price in the
    window. A move qualifies, and the new price must sit in a tradable band
    (ignoring resolved 0/1 rails and penny rails), when EITHER:

    - **Absolute**: |move| >= `threshold` (mid-range overreactions), OR
    - **Relative**: a low-probability outcome (ref <= `relative_max_ref`) spiked
      UP by at least `relative_multiple`× to >= `relative_min_price`. This is the
      tail-event case — a "regime collapses by <date>" market jumping 4%→16% is
      only +12pp absolute but 4× relative, and is exactly the salience-driven
      overreaction we want to fade. A pure absolute threshold would miss it.

    We keep the single largest-magnitude outcome per market (preferring the
    upward spike on ties) — its headline move.
    """
    # Group rows by (condition_id, outcome), preserving chronological order.
    series: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    for row in price_rows:
        cid = str(row["condition_id"])
        outcome = str(row["outcome"])
        ts = str(row["captured_at"])
        try:
            price = float(row["price"])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        series[(cid, outcome)].append((ts, price))

    # Largest qualifying move per market (condition_id).
    best_per_market: dict[str, MoveEvent] = {}
    for (cid, outcome), points in series.items():
        if len(points) < 2:
            continue
        points.sort(key=lambda p: p[0])
        ref_price = points[0][1]
        new_price = points[-1][1]
        move = new_price - ref_price

        absolute_hit = abs(move) >= threshold
        relative_hit = (
            move > 0
            and 0 < ref_price <= relative_max_ref
            and new_price >= relative_min_price
            and new_price >= ref_price * relative_multiple
        )
        if not (absolute_hit or relative_hit):
            continue
        if not (min_price <= new_price <= max_price):
            continue

        event = MoveEvent(
            condition_id=cid,
            outcome=outcome,
            ref_price=ref_price,
            new_price=new_price,
            move=move,
        )
        current = best_per_market.get(cid)
        # Largest magnitude wins; on a tie prefer the upward spike (clearer to
        # read as "this outcome got bid up"), since binary YES/NO moves tie.
        if (
            current is None
            or event.magnitude > current.magnitude
            or (event.magnitude == current.magnitude and event.move > current.move)
        ):
            best_per_market[cid] = event

    # Biggest movers first.
    return sorted(best_per_market.values(), key=lambda e: e.magnitude, reverse=True)
