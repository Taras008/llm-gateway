from typing import Any, Protocol

from pydantic import BaseModel


class LLMResult(BaseModel):
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class LLMProvider(Protocol):
    async def chat(
        self,
        message: str,
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResult: ...
