import logging

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import LLM_PROVIDER
from app.llm.base import LLMProvider
from app.llm.errors import (
    LLMBadResponseError,
    LLMError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.llm.factory import get_provider
from app.llm_calls import call_llm, call_llm_structured, start_stream
from app.request_log import RequestLogRepository, get_request_log_repository
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
    logger.warning(
        "LLM error on %s %s: %s: %s", request.method, request.url.path, type(exc).__name__, exc
    )
    return JSONResponse(status_code=status_code, content={"detail": detail})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", llm_provider=LLM_PROVIDER)


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    provider: LLMProvider = Depends(get_provider),
    log_repo: RequestLogRepository = Depends(get_request_log_repository),
) -> ChatResponse:
    call = await call_llm(provider, log_repo, endpoint="/chat", message=request.message)

    return ChatResponse(
        answer=call.result.text,
        model=call.result.model,
        prompt_tokens=call.result.prompt_tokens,
        completion_tokens=call.result.completion_tokens,
        latency_ms=call.latency_ms,
    )


@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    provider: LLMProvider = Depends(get_provider),
    log_repo: RequestLogRepository = Depends(get_request_log_repository),
) -> StreamingResponse:
    chunks = await start_stream(
        provider, log_repo, endpoint="/chat/stream", message=request.message
    )
    return StreamingResponse(chunks, media_type="text/plain; charset=utf-8")


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: AnalyzeRequest,
    provider: LLMProvider = Depends(get_provider),
    log_repo: RequestLogRepository = Depends(get_request_log_repository),
) -> AnalyzeResponse:
    analysis, call = await call_llm_structured(
        provider,
        log_repo,
        endpoint="/analyze",
        message=request.text,
        system=ANALYZE_SYSTEM_PROMPT,
        schema=ReviewAnalysis,
    )

    return AnalyzeResponse(
        analysis=analysis,
        model=call.result.model,
        prompt_tokens=call.result.prompt_tokens,
        completion_tokens=call.result.completion_tokens,
        latency_ms=call.latency_ms,
    )
