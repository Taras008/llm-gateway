import logging
import time

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.config import LLM_PROVIDER
from app.llm.base import LLMProvider
from app.llm.errors import (
    LLMBadResponseError,
    LLMError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.llm.factory import get_provider
from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ReviewAnalysis,
)

logger = logging.getLogger(__name__)

ANALYZE_SYSTEM_PROMPT = (
    "You analyze customer reviews. Return JSON with fields: "
    "sentiment (positive, negative or neutral), "
    "topics (up to 5 short topics mentioned in the review), "
    "summary (one short sentence in the language of the review)."
)

# Тип помилки -> (HTTP-статус, повідомлення для клієнта)
LLM_ERROR_RESPONSES: dict[type[LLMError], tuple[int, str]] = {
    LLMUnavailableError: (503, "LLM provider is unavailable, try again later"),
    LLMTimeoutError: (504, "LLM provider did not respond in time"),
    LLMBadResponseError: (502, "LLM provider returned an invalid response"),
}

app = FastAPI(title="llm-gateway")


@app.exception_handler(LLMError)
async def handle_llm_error(request: Request, exc: LLMError) -> JSONResponse:
    status_code, detail = LLM_ERROR_RESPONSES.get(type(exc), (502, "LLM provider error"))
    logger.warning("LLM error on %s %s: %s: %s", request.method, request.url.path, type(exc).__name__, exc)
    return JSONResponse(status_code=status_code, content={"detail": detail})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", llm_provider=LLM_PROVIDER)


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


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: AnalyzeRequest,
    provider: LLMProvider = Depends(get_provider),
) -> AnalyzeResponse:
    start = time.perf_counter()
    result = await provider.chat(
        request.text,
        system=ANALYZE_SYSTEM_PROMPT,
        json_schema=ReviewAnalysis.model_json_schema(),
    )
    latency_ms = int((time.perf_counter() - start) * 1000)

    try:
        analysis = ReviewAnalysis.model_validate_json(result.text)
    except ValidationError as exc:
        logger.warning(
            "Invalid structured output from %s: %s | raw=%r",
            result.model,
            exc.errors(include_url=False, include_input=False),
            result.text[:500],
        )
        raise LLMBadResponseError("Structured output failed validation") from exc

    return AnalyzeResponse(
        analysis=analysis,
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        latency_ms=latency_ms,
    )
