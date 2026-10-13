import pytest

from app.llm.base import LLMResult
from app.llm.errors import LLMBadResponseError, LLMTimeoutError, LLMUnavailableError
from app.llm.retry import RetryingProvider


class FlakyProvider:
    """Raises the given errors one by one, then succeeds."""

    def __init__(self, errors: list[Exception]) -> None:
        self.errors = list(errors)
        self.calls = 0
        self.last_kwargs: dict = {}

    async def chat(self, message, *, system=None, json_schema=None):
        self.calls += 1
        self.last_kwargs = {"system": system, "json_schema": json_schema}
        if self.errors:
            raise self.errors.pop(0)
        return LLMResult(text="ok", model="flaky", prompt_tokens=1, completion_tokens=1)


class FakeSleep:
    """Records delays instead of actually sleeping."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def make_provider(inner, sleep):
    return RetryingProvider(inner, max_attempts=3, base_delay=0.5, sleep=sleep)


@pytest.mark.anyio
async def test_no_retry_when_first_call_succeeds():
    inner, sleep = FlakyProvider([]), FakeSleep()

    result = await make_provider(inner, sleep).chat("hi")

    assert result.text == "ok"
    assert inner.calls == 1
    assert sleep.delays == []


@pytest.mark.anyio
async def test_retries_unavailable_with_exponential_backoff():
    inner = FlakyProvider([LLMUnavailableError("down"), LLMUnavailableError("down")])
    sleep = FakeSleep()

    result = await make_provider(inner, sleep).chat("hi")

    assert result.text == "ok"
    assert inner.calls == 3
    assert sleep.delays == [0.5, 1.0]


@pytest.mark.anyio
async def test_gives_up_after_max_attempts():
    inner = FlakyProvider([LLMUnavailableError("down")] * 5)
    sleep = FakeSleep()

    with pytest.raises(LLMUnavailableError):
        await make_provider(inner, sleep).chat("hi")

    assert inner.calls == 3
    assert sleep.delays == [0.5, 1.0]


@pytest.mark.parametrize("error", [LLMTimeoutError("slow"), LLMBadResponseError("bad")])
@pytest.mark.anyio
async def test_does_not_retry_non_transient_errors(error):
    inner, sleep = FlakyProvider([error]), FakeSleep()

    with pytest.raises(type(error)):
        await make_provider(inner, sleep).chat("hi")

    assert inner.calls == 1
    assert sleep.delays == []


@pytest.mark.anyio
async def test_passes_all_arguments_to_inner_provider():
    inner, sleep = FlakyProvider([]), FakeSleep()
    schema = {"type": "object"}

    await make_provider(inner, sleep).chat("hi", system="be brief", json_schema=schema)

    assert inner.last_kwargs == {"system": "be brief", "json_schema": schema}


def test_rejects_zero_attempts():
    with pytest.raises(ValueError):
        RetryingProvider(FlakyProvider([]), max_attempts=0)
