from app.config import LLM_PROVIDER, OLLAMA_MODEL, OLLAMA_URL
from app.llm.base import LLMProvider
from app.llm.fake import FakeProvider
from app.llm.ollama import OllamaProvider
from app.llm.retry import RetryingProvider


def get_provider() -> LLMProvider:
    if LLM_PROVIDER == "ollama":
        return RetryingProvider(
            OllamaProvider(base_url=OLLAMA_URL, model=OLLAMA_MODEL),
            max_attempts=3,
            base_delay=0.5,
        )
    if LLM_PROVIDER == "fake":
        return FakeProvider()
    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")
