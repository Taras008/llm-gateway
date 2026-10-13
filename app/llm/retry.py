import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from app.llm.base import LLMProvider, LLMResult, LLMStreamChunk
from app.llm.errors import LLMUnavailableError

logger = logging.getLogger(__name__)

SleepFunc = Callable[[float], Awaitable[None]]


class RetryingProvider:
    """Wraps any LLMProvider and retries transient failures with exponential backoff."""

    def __init__(
        self,
        inner: LLMProvider,
        max_attempts: int = 3,
        base_delay: float = 0.5,
        sleep: SleepFunc = asyncio.sleep,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.inner = inner
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.sleep = sleep

    async def chat(
        self,
        message: str,
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResult:
        for attempt in range(1, self.max_attempts + 1):
            try:
                return await self.inner.chat(message, system=system, json_schema=json_schema)
            except LLMUnavailableError as exc:
                if attempt == self.max_attempts:
                    logger.error("LLM unavailable, giving up after %d attempts: %s", attempt, exc)
                    raise

                delay = self.base_delay * 2 ** (attempt - 1)
                logger.warning(
                    "LLM unavailable (attempt %d/%d), retrying in %.1fs: %s",
                    attempt,
                    self.max_attempts,
                    delay,
                    exc,
                )
                await self.sleep(delay)

        raise AssertionError("unreachable")  # loop always returns or raises

    def stream(
        self,
        message: str,
        *,
        system: str | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        # Стрім не повторюємо: якщо він обірвався посередині, клієнт уже отримав
        # частину тексту, і повтор продублював би її.
        return self.inner.stream(message, system=system)
