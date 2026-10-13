from typing import Protocol

from pydantic import BaseModel


class LLMResult(BaseModel):
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class LLMProvider(Protocol):
    async def chat(self, message: str) -> LLMResult: ...