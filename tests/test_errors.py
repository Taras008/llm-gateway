import pytest

from app.llm.errors import LLMBadResponseError, LLMTimeoutError, LLMUnavailableError
from app.llm.factory import get_provider
from app.llm.ollama import OllamaProvider
from app.main import app


class FailingProvider:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    async def chat(self, message, *, system=None, json_schema=None):
        raise self.exc


@pytest.mark.parametrize(
    ("exc", "expected_status"),
    [
        (LLMUnavailableError("down"), 503),
        (LLMTimeoutError("slow"), 504),
        (LLMBadResponseError("garbage"), 502),
    ],
)
def test_llm_errors_are_mapped_to_http_status(client, exc, expected_status):
    app.dependency_overrides[get_provider] = lambda: FailingProvider(exc)

    response = client.post("/chat", json={"message": "hi"})

    assert response.status_code == expected_status
    assert "detail" in response.json()


@pytest.mark.anyio
async def test_ollama_provider_raises_unavailable_when_server_is_down():
    provider = OllamaProvider(base_url="http://127.0.0.1:1", model="any")

    with pytest.raises(LLMUnavailableError):
        await provider.chat("hi")
