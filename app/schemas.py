from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    answer: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    llm_provider: str


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class ReviewAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    topics: list[str] = Field(max_length=5)
    summary: str = Field(max_length=300)


class AnalyzeResponse(BaseModel):
    analysis: ReviewAnalysis
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
