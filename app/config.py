import os

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+asyncpg://gateway:gateway@localhost:5433/gateway"
)
