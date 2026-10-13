import httpx

from app.llm.base import LLMResult


class OllamaProvider:
    def __init__(self, base_url: str, model: str, timeout: float = 60.0) -> None:
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    async def chat(self, message: str) -> LLMResult:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": message}],
            "stream": False,
        }
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            response = await client.post("/api/chat", json=payload)
            response.raise_for_status()

        data = response.json()
        return LLMResult(
            text=data["message"]["content"],
            model=data["model"],
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
        )