from collections.abc import AsyncIterator
from typing import Any, Protocol

from pydantic import BaseModel


class LLMResult(BaseModel):
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class LLMStreamChunk(BaseModel):
    text: str
    done: bool = False
    # заповнюються тільки в останньому чанку (done=True)
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMProvider(Protocol):
    async def chat(
        self,
        message: str,
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResult: ...

    def stream(
        self,
        message: str,
        *,
        system: str | None = None,
    ) -> AsyncIterator[LLMStreamChunk]: ...
