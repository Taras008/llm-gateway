import re
from collections.abc import AsyncIterator
from typing import Any

from app.llm.base import LLMResult, LLMStreamChunk


class FakeProvider:
    def __init__(self, text: str | None = None) -> None:
        self.text = text

    def _answer(self, message: str) -> str:
        return self.text if self.text is not None else f"fake answer to: {message}"

    async def chat(
        self,
        message: str,
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResult:
        return LLMResult(
            text=self._answer(message),
            model="fake",
            prompt_tokens=len(message.split()),
            completion_tokens=4,
        )

    async def stream(
        self,
        message: str,
        *,
        system: str | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        # слово разом з пробілом після нього: "fake ", "answer ", ...
        for piece in re.findall(r"\S+\s*", self._answer(message)):
            yield LLMStreamChunk(text=piece)
        yield LLMStreamChunk(
            text="",
            done=True,
            model="fake",
            prompt_tokens=len(message.split()),
            completion_tokens=4,
        )
