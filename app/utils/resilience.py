"""Resilience utilities: retries, circuit breaker, rate limiting."""

import asyncio
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3  # successes needed to close from half-open


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None


class CircuitBreaker:
    """Circuit breaker implementation."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.stats.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.stats.state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise CircuitBreakerError("Circuit breaker is OPEN")

        try:
            result = (
                await func(*args, **kwargs)
                if asyncio.iscoroutinefunction(func)
                else func(*args, **kwargs)
            )
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.stats.last_failure_time is None:
            return True
        return (
            time.time() - self.stats.last_failure_time >= self.config.recovery_timeout
        )

    async def _on_success(self):
        """Handle successful call."""
        async with self._lock:
            self.stats.success_count += 1
            self.stats.last_success_time = time.time()

            if self.stats.state == CircuitState.HALF_OPEN:
                if self.stats.success_count >= self.config.success_threshold:
                    self.stats.state = CircuitState.CLOSED
                    self.stats.failure_count = 0
                    self.stats.success_count = 0
                    logger.info("Circuit breaker CLOSED after successful recovery")

    async def _on_failure(self):
        """Handle failed call."""
        async with self._lock:
            self.stats.failure_count += 1
            self.stats.last_failure_time = time.time()

            if self.stats.state == CircuitState.HALF_OPEN:
                self.stats.state = CircuitState.OPEN
                logger.warning("Circuit breaker OPEN after failure in HALF_OPEN state")
            elif self.stats.failure_count >= self.config.failure_threshold:
                self.stats.state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker OPEN after {self.stats.failure_count} failures"
                )


class CircuitBreakerError(Exception):
    """Circuit breaker is open."""

    pass


@dataclass
class RetryConfig:
    """Retry configuration."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


async def retry_with_exponential_backoff(
    func: Callable, config: RetryConfig, *args, **kwargs
) -> Any:
    """Retry function with exponential backoff."""
    last_exception = None

    for attempt in range(config.max_attempts):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except Exception as e:
            last_exception = e

            if attempt == config.max_attempts - 1:
                logger.error(f"All {config.max_attempts} retry attempts failed")
                break

            delay = min(
                config.base_delay * (config.exponential_base**attempt), config.max_delay
            )

            if config.jitter:
                delay *= 0.5 + random.random() * 0.5  # Add 0-50% jitter

            logger.warning(
                f"Attempt {attempt + 1} failed: {e}. " f"Retrying in {delay:.2f}s..."
            )
            await asyncio.sleep(delay)

    raise last_exception


def retry(config: RetryConfig | None = None):
    """Retry decorator."""
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await retry_with_exponential_backoff(func, config, *args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(
                retry_with_exponential_backoff(func, config, *args, **kwargs)
            )

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


@dataclass
class TokenBucketConfig:
    """Token bucket rate limiter configuration."""

    capacity: int = 100  # Maximum tokens
    refill_rate: float = 10.0  # Tokens per second
    burst_capacity: int = 20  # Burst allowance


class TokenBucket:
    """Token bucket rate limiter."""

    def __init__(self, config: TokenBucketConfig):
        self.config = config
        self.tokens = float(config.capacity)
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens."""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_refill

            # Refill tokens
            self.tokens = min(
                self.config.capacity, self.tokens + elapsed * self.config.refill_rate
            )
            self.last_refill = now

            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False

    async def wait_for_tokens(
        self, tokens: int = 1, timeout: float | None = None
    ) -> bool:
        """Wait for tokens to become available."""
        start_time = time.time()

        while True:
            if await self.acquire(tokens):
                return True

            if timeout and (time.time() - start_time) >= timeout:
                return False

            await asyncio.sleep(0.1)


class RateLimitError(Exception):
    """Rate limit exceeded."""

    pass


class RateLimiter:
    """Rate limiter using token bucket algorithm."""

    def __init__(self, config: TokenBucketConfig):
        self.bucket = TokenBucket(config)

    async def check_rate_limit(self, tokens: int = 1) -> None:
        """Check rate limit and raise exception if exceeded."""
        if not await self.bucket.acquire(tokens):
            raise RateLimitError("Rate limit exceeded")

    def rate_limit(self, tokens: int = 1):
        """Rate limiting decorator."""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                await self.check_rate_limit(tokens)
                return await func(*args, **kwargs)

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                asyncio.run(self.check_rate_limit(tokens))
                return func(*args, **kwargs)

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator


# Global instances (configured from settings)
_circuit_breakers: dict[str, CircuitBreaker] = {}
_rate_limiters: dict[str, RateLimiter] = {}


def get_circuit_breaker(
    name: str, config: CircuitBreakerConfig | None = None
) -> CircuitBreaker:
    """Get or create circuit breaker instance."""
    if name not in _circuit_breakers:
        if config is None:
            config = CircuitBreakerConfig()
        _circuit_breakers[name] = CircuitBreaker(config)
    return _circuit_breakers[name]


def get_rate_limiter(name: str, config: TokenBucketConfig | None = None) -> RateLimiter:
    """Get or create rate limiter instance."""
    if name not in _rate_limiters:
        if config is None:
            config = TokenBucketConfig()
        _rate_limiters[name] = RateLimiter(config)
    return _rate_limiters[name]


# Convenience decorators
def with_circuit_breaker(name: str, config: CircuitBreakerConfig | None = None):
    """Circuit breaker decorator."""
    breaker = get_circuit_breaker(name, config)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)

        return wrapper

    return decorator


def with_rate_limit(
    name: str, tokens: int = 1, config: TokenBucketConfig | None = None
):
    """Rate limiting decorator."""
    limiter = get_rate_limiter(name, config)
    return limiter.rate_limit(tokens)
