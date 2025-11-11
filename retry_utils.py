"""Retry utilities for handling transient failures with exponential backoff."""

import asyncio
import functools
import logging
import random
from typing import Any, Callable, Coroutine, Type, Union

logger = logging.getLogger(__name__)


def retry_with_backoff(
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: Union[Type[Exception], tuple[Type[Exception], ...]] = Exception,
) -> Callable:
    """Decorator for retrying async functions with exponential backoff."""
    def decorator(func: Callable[..., Coroutine[Any, Any, Any]]) -> Callable[..., Coroutine[Any, Any, Any]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts - 1:
                        logger.error(
                            f"Failed after {max_attempts} attempts for {func.__name__}. "
                            f"Final error: {str(e)}"
                        )
                        raise

                    delay = min(base_delay * (exponential_base ** attempt), max_delay)

                    if jitter:
                        delay = delay * (0.5 + random.random() * 0.5)

                    if hasattr(e, 'message') and "retry after" in str(e.message).lower():
                        try:
                            retry_after = int("".join(filter(str.isdigit, str(e.message))))
                            delay = min(retry_after, max_delay)
                            logger.warning(
                                f"Telegram rate limit hit for {func.__name__}. "
                                f"Waiting {delay:.1f}s before retry {attempt + 1}/{max_attempts}"
                            )
                        except (ValueError, AttributeError):
                            pass
                    else:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}. "
                            f"Error: {str(e)}. Retrying in {delay:.1f}s..."
                        )

                    await asyncio.sleep(delay)

            if last_exception:
                raise last_exception

        return wrapper
    return decorator


def retry_with_backoff_sync(
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: Union[Type[Exception], tuple[Type[Exception], ...]] = Exception,
) -> Callable:
    """Decorator for retrying synchronous functions with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts - 1:
                        logger.error(
                            f"Failed after {max_attempts} attempts for {func.__name__}. "
                            f"Final error: {str(e)}"
                        )
                        raise

                    delay = min(base_delay * (exponential_base ** attempt), max_delay)

                    if jitter:
                        delay = delay * (0.5 + random.random() * 0.5)

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}. "
                        f"Error: {str(e)}. Retrying in {delay:.1f}s..."
                    )

                    import time
                    time.sleep(delay)

            if last_exception:
                raise last_exception

        return wrapper
    return decorator