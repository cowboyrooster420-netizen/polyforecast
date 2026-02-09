from __future__ import annotations

import functools
import logging
from typing import Any, Callable, TypeVar

from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def _log_retry(retry_state: RetryCallState) -> None:
    exc = retry_state.outcome and retry_state.outcome.exception()
    logger.warning(
        "Retry attempt %d for %s: %s",
        retry_state.attempt_number,
        retry_state.fn.__name__ if retry_state.fn else "unknown",
        exc,
    )


def with_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 30.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[F], F]:
    """Decorator that adds exponential backoff retry to an async function."""

    def decorator(fn: F) -> F:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=min_wait, max=max_wait),
            retry=retry_if_exception_type(retry_on),
            before_sleep=_log_retry,
            reraise=True,
        )
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
