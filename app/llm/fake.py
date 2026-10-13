from app.llm.base import LLMResult


class FakeProvider:
    async def chat(self, message: str) -> LLMResult:
        return LLMResult(
            text=f"fake answer to: {message}",
            model="fake",
            prompt_tokens=len(message.split()),
            completion_tokens=4,
        )