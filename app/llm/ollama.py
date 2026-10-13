from typing import Any

import httpx

from app.llm.base import LLMResult
from app.llm.errors import LLMBadResponseError, LLMTimeoutError, LLMUnavailableError


class OllamaProvider:
    def __init__(self, base_url: str, model: str, timeout: float = 60.0) -> None:
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    async def chat(
        self,
        message: str,
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if json_schema is not None:
            payload["format"] = json_schema
            payload["options"] = {"temperature": 0}

        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
                response = await client.post("/api/chat", json=payload)
                response.raise_for_status()
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise LLMUnavailableError(f"Ollama is not reachable at {self.base_url}") from exc
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Ollama did not respond within {self.timeout}s") from exc
        except httpx.HTTPStatusError as exc:
            raise LLMBadResponseError(
                f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc

        try:
            data = response.json()
            return LLMResult(
                text=data["message"]["content"],
                model=data["model"],
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
            )
        except (ValueError, KeyError) as exc:
            raise LLMBadResponseError("Ollama returned an unexpected response format") from exc
