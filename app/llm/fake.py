from typing import Any

from app.llm.base import LLMResult


class FakeProvider:
    def __init__(self, text: str | None = None) -> None:
        self.text = text

    async def chat(
        self,
        message: str,
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResult:
        text = self.text if self.text is not None else f"fake answer to: {message}"
        return LLMResult(
            text=text,
            model="fake",
            prompt_tokens=len(message.split()),
            completion_tokens=4,
        )
