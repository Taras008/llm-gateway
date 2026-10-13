import time

from fastapi import Depends, FastAPI

from app.llm.base import LLMProvider
from app.llm.factory import get_provider
from app.schemas import ChatRequest, ChatResponse, HealthResponse

app = FastAPI(title="llm-gateway")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    provider: LLMProvider = Depends(get_provider),
) -> ChatResponse:
    start = time.perf_counter()
    result = await provider.chat(request.message)
    latency_ms = int((time.perf_counter() - start) * 1000)

    return ChatResponse(
        answer=result.text,
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        latency_ms=latency_ms,
    )