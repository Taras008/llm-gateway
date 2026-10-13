import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.llm.base import LLMResult, LLMStreamChunk
from app.llm.errors import LLMBadResponseError, LLMTimeoutError, LLMUnavailableError


class OllamaProvider:
    def __init__(self, base_url: str, model: str, timeout: float = 60.0) -> None:
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    def _messages(self, message: str, system: str | None) -> list[dict[str, str]]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})
        return messages

    async def chat(
        self,
        message: str,
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(message, system),
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

    async def stream(
        self,
        message: str,
        *,
        system: str | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        payload = {
            "model": self.model,
            "messages": self._messages(message, system),
            "stream": True,
        }
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
                async with client.stream("POST", "/api/chat", json=payload) as response:
                    if response.status_code >= 400:
                        body = await response.aread()
                        raise LLMBadResponseError(
                            f"Ollama returned HTTP {response.status_code}: {body[:200]!r}"
                        )
                    # Ollama віддає NDJSON: один JSON-об'єкт на рядок
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        data = json.loads(line)
                        if data.get("done"):
                            yield LLMStreamChunk(
                                text=data.get("message", {}).get("content", ""),
                                done=True,
                                model=data.get("model"),
                                prompt_tokens=data.get("prompt_eval_count", 0),
                                completion_tokens=data.get("eval_count", 0),
                            )
                        else:
                            yield LLMStreamChunk(text=data["message"]["content"])
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise LLMUnavailableError(f"Ollama is not reachable at {self.base_url}") from exc
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Ollama did not respond within {self.timeout}s") from exc
        except (ValueError, KeyError) as exc:
            raise LLMBadResponseError("Ollama returned an unexpected stream format") from exc
