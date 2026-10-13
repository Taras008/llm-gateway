import time

import httpx

from app.config import OLLAMA_MODEL, OLLAMA_URL
from app.schemas import ChatResponse


async def chat(message: str) -> ChatResponse:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": message}],
        "stream": False,
    }
    start = time.perf_counter()
    async with httpx.AsyncClient(base_url=OLLAMA_URL, timeout=60.0) as client:
        response = await client.post("/api/chat", json=payload)
        response.raise_for_status()
    latency_ms = int((time.perf_counter() - start) * 1000)

    data = response.json()
    return ChatResponse(
        answer=data["message"]["content"],
        model=data["model"],
        prompt_tokens=data.get("prompt_eval_count", 0),
        completion_tokens=data.get("eval_count", 0),
        latency_ms=latency_ms,
    )