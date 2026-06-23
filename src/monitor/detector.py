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
) -> list[MoveEvent]:
    """Find sharp price moves, one (the largest) per market.

    `price_rows` is the window's price history: each row must have
    `condition_id`, `outcome`, `captured_at` (ISO8601 string, sortable), and
    `price`. Rows are expected to already be scoped to the rolling window; this
    function only needs them grouped, not time-filtered.

    For each (market, outcome) we compare the earliest and latest price in the
    window. A move qualifies when its magnitude clears `threshold` AND the new
    price sits in a tradable band (so we ignore already-resolved 0/1 rails and
    penny rails where the fade is unfillable). We then keep only the single
    largest-magnitude outcome per market — its headline move.
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
        if abs(move) < threshold:
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
        if current is None or event.magnitude > current.magnitude:
            best_per_market[cid] = event

    # Biggest movers first.
    return sorted(best_per_market.values(), key=lambda e: e.magnitude, reverse=True)
