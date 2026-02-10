from __future__ import annotations

import asyncio
import time


class AsyncTokenBucket:
    """Async token-bucket rate limiter."""

    def __init__(self, rate: float, capacity: float | None = None) -> None:
        self._rate = rate  # tokens per second
        self._capacity = max(capacity or rate, 1.0)
        self._tokens = self._capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now

    async def acquire(self, tokens: float = 1.0) -> None:
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                wait = (tokens - self._tokens) / self._rate
            await asyncio.sleep(wait)
