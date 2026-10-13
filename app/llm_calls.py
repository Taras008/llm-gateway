import logging
import time
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.llm.base import LLMProvider, LLMResult
from app.llm.errors import LLMBadResponseError, LLMError
from app.request_log import RequestLogEntry, RequestLogRepository

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMCall:
    result: LLMResult
    latency_ms: int


async def call_llm(
    provider: LLMProvider,
    log_repo: RequestLogRepository,
    *,
    endpoint: str,
    message: str,
    system: str | None = None,
) -> LLMCall:
    call, _ = await _execute(
        provider, log_repo, endpoint=endpoint, message=message, system=system, schema=None
    )
    return call


async def call_llm_structured(
    provider: LLMProvider,
    log_repo: RequestLogRepository,
    *,
    endpoint: str,
    message: str,
    system: str | None,
    schema: type[T],
) -> tuple[T, LLMCall]:
    call, parsed = await _execute(
        provider, log_repo, endpoint=endpoint, message=message, system=system, schema=schema
    )
    assert parsed is not None
    return parsed, call


async def _execute(
    provider: LLMProvider,
    log_repo: RequestLogRepository,
    *,
    endpoint: str,
    message: str,
    system: str | None,
    schema: type[T] | None,
) -> tuple[LLMCall, T | None]:
    start = time.perf_counter()
    result: LLMResult | None = None
    try:
        result = await provider.chat(
            message,
            system=system,
            json_schema=schema.model_json_schema() if schema else None,
        )
        parsed = _parse(schema, result) if schema else None
    except LLMError as exc:
        await _safe_log(
            log_repo,
            RequestLogEntry(
                endpoint=endpoint,
                status="error",
                error_type=type(exc).__name__,
                # якщо модель відповіла, але відповідь не пройшла валідацію,
                # токени вже витрачено — їх теж треба врахувати
                model=result.model if result else None,
                input_chars=len(message),
                prompt_tokens=result.prompt_tokens if result else None,
                completion_tokens=result.completion_tokens if result else None,
                latency_ms=_elapsed_ms(start),
            ),
        )
        raise

    latency_ms = _elapsed_ms(start)
    await _safe_log(
        log_repo,
        RequestLogEntry(
            endpoint=endpoint,
            status="success",
            error_type=None,
            model=result.model,
            input_chars=len(message),
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_ms=latency_ms,
        ),
    )
    return LLMCall(result=result, latency_ms=latency_ms), parsed


def _parse(schema: type[T], result: LLMResult) -> T:
    try:
        return schema.model_validate_json(result.text)
    except ValidationError as exc:
        logger.warning(
            "Invalid structured output from %s: %s | raw=%r",
            result.model,
            exc.errors(include_url=False, include_input=False),
            result.text[:500],
        )
        raise LLMBadResponseError("Structured output failed validation") from exc


async def _safe_log(log_repo: RequestLogRepository, entry: RequestLogEntry) -> None:
    # Лог — допоміжна річ: якщо БД недоступна, користувач однаково має отримати відповідь.
    try:
        await log_repo.add(entry)
    except Exception:
        logger.exception("Failed to write request log for %s", entry.endpoint)


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
