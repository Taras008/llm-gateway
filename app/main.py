from fastapi import FastAPI
from pydantic import BaseModel
from app import ollama_client
from app.schemas import ChatRequest, ChatResponse


app = FastAPI(title="llm-gateway")


class HealthResponse(BaseModel):
    status: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    return await ollama_client.chat(request.message)